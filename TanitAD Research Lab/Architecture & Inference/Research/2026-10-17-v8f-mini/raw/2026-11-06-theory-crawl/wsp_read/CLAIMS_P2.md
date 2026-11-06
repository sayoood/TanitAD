# CLAIMS_P2: what the weights of C0 s0 say about plan speed, written BEFORE any forward pass (WSP-READ agent, 2026-11-06)

Checkpoint: `E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt` (7,233,343 parameters; the C0C rows of P1 use the same weights with a post-hoc composite).
Model source: `dagt1_tree_b43c7a9a/stack/tanitad/models/v8f_mini.py`. Frozen theory: `THEORY_CRAWL.md` sha 6e83451c, `THEORY_CRAWL_AMEND_1.md` sha 9d39ac53.
Method of this file: tensor arithmetic on `ck["model"]` only (`p2a_weights.py` sha ec7b51cb..., output `P2A_WEIGHTS.json` sha 339a585a...). No model object was built, no input was made, no forward pass ran.
Evidence class of every number below: MEASURED (tensor arithmetic, artifact `P2A_WEIGHTS.json`), unless marked STRUCTURE (read from source, not from weights).

## 0. Quantities and conventions
* **Plan speed** `s` = (x30[11] - x30[7]) / 0.8 s, the 0.8 s secant at +2.0 s of the plan's longitudinal channel (channel 0) on the 0.2 s grid. This is the P1 definition.
* The plan is `plan30[j] = p30[j] + res60[2j+1]`. Hence `s = s_base + s_net + s_res`:
  * `s_base`: the kd_x base (a function of v0, a0, kappa, not learned; STRUCTURE). Forward-Euler const-accel gives `s_base = v0 + 1.9 a0` while unclamped.
  * `s_net`: the frozen P10 MLP (8 to 256 to 60), a function of the 8 standardised ego features only.
  * `s_res`: the learned residual `head_o(reader(cond, scene))`, 12 cosine-ramp coefficients per axis times 2.0.
* **Ego features** f8 = (v, a, k, v^2, v*a, v*k, v^2*k, a*k), `k` = curvature in yaw mode. a0 is the 0.2 s backward difference of speed. So the whole "ego-speed history" is two speed samples plus curvature.
* **z-units.** Ego: the model's own standardiser `prior.xs` = (7.714, 0.809, 0.02257) for (v0, a0, kappa). Vision: the output coordinates of `adapter.0` (a LayerNorm with gamma rms 1.001, so unit variance per coordinate; 4 frames x 40 tokens x 1024 = 163,840 coordinates). A block norm is the L2 norm of the gradient over the block's coordinates.

## 1. Claims from the weights alone (each can fail at 2b)
**W1. The P10 MLP is not a level carrier; it corrects the base's acceleration extrapolation.**
* Linear-limit gain of `s_net` on the standardised features: norm 1.0045, dominated by `a` (-0.921 per z), `v*a` (-0.304), `k` (-0.123), `v` (+0.127), `a*k` (+0.120), `v*k` (+0.112), `v^2*k` (+0.099); `v^2` is 0.006.
* In raw units the linear-limit gain is +0.0165 m/s per m/s on v0 and -1.138 m/s per m/s^2 on a0 (before the chain terms through v*a, which add -0.038 v).
* Upper bound on the gain norm (spectral, GELU slope 1.1289): 14.24 per z (loose).
* **Claim W1:** on the held-out windows with v0 in 3-8 m/s, the median over windows of `d s_net / d v0` has |.| <= 0.15 m/s per m/s, and the median of `d s_net / d a0` lies in [-2.0, -0.6] m/s per m/s^2. (Window-level gradients, GELU slopes at the actual features.)
* **Consequence (STRUCTURE plus W1):** the speed LEVEL enters `s` through the base (d s_base / d v0 = 1, z-gain 7.71 m/s per sigma of v0), and the prior's net acts on the acceleration: `d(s_base+s_net)/d a0` is about 1.9 - 1.1 = +0.8 m/s per m/s^2. Check range: median in [0.0, 1.3].

**W2. Inside the learned residual head, ego and vision have comparable capacity; the ego dominance is not learned.**
* Residual readout: `r_res` (12 coefficients, norm 3.21) times `head_o.2` x-rows (12 x 512; singular values 1.35 ... 0.086, effective rank 6.46). `||r_res W_o2x||` = 0.9999; `||r_res W_o2x W_o1||` = 2.573 (GELU slope 1), spread evenly over the 4 query outputs (1.29, 1.29, 1.26, 1.30). The speed readout lies mostly in low-gain singular directions of the head (alignment with the top left singular vector 0.078).
* Structure of the reader (STRUCTURE from source): the stream starts at `q + mean(6 cond tokens)`, so each cond token (ego, nav, limit, S's checkpoints, T's code, ages) reaches every query with weight 1/6 by a SKIP path that does not depend on attention; scene tokens (160) reach the queries only through cross-attention over 166 memory tokens.
* Capacities (block norm of the gradient of `s_res` w.r.t. the group's input coordinates, common final-LayerNorm factor 1/sigma dropped for all groups alike, GELU slope 1; `P2A_WEIGHTS.json` key `capacity_weight_only`):
  * ego f8-z (8 coords), skip path 0.349; attention path 0.026 at uniform attention, 4.38 at attention 1.
  * vision LN coords (163,840), attention path 0.320 at uniform attention (alpha 1/166 on all 160 tokens), 4.20 if ALL attention sat on one token.
  * T's code (60 coords) skip 0.363; nav (5) 0.289; limit 0.201; S checkpoints 0.181; ages 0.232.
* Ratios: ego(skip) / vision(uniform) = **1.09**; ego(skip) / vision(one token) = 0.083; ego(skip+attention, uniform) / vision(uniform) = 1.17. Code / ego (skip) = **1.04**.
* **Claim W2:** for the RESIDUAL component `s_res` on the 3-8 m/s windows, the ratio of mean per-window Jacobian block norms, ego f8-z over vision, lies in **[0.08, 1.3]**. In particular it is below 2, so the residual path alone would FAIL the P2 bar (<= 2x).
* **Claim W3 (T-code):** the same ratio, T's code (partial derivative at the code O actually reads) over ego f8-z, lies in **[0.35, 3.1]** (a factor 3 around 1.04). The weights give no support to "the T-code path to plan speed is weak": per input coordinate its capacity is not below the ego's.

**W4. What the weights cannot say.**
* The size of `s_res` (m/s). The weight-only bound is 254 m/s (`bound_abs_res_speed`), vacuous; no claim is made on the magnitude of the residual, only on gains.
* The attention distribution (data-dependent), hence the position inside W2's bracket.

## 2. Prediction for P2 as written (stated now, decided by 2b)
* **P2 text:** gain of plan speed on the ego-speed history exceeds the gain on vision tokens by >= 5x as the Jacobian block norm at 3-8 m/s windows (holds); <= 2x (fails); between (INCONCLUSIVE).
* **Verdict statistic (fixed here):** `R_tot` = (mean over the 3-8 m/s windows of the L2 norm of `[d s/d v0, d s/d a0, d s/d kappa] * prior.xs`, with `p10f` and `base` rebuilt consistently from the perturbed (v0, a0, kappa)) divided by (mean over the same windows of the L2 norm of `d s / d (adapter.0 output)`). The total derivative of `s` (all paths, including S and T). Window-cluster bootstrap CI over episodes, B 2000, seed 0, reported; the verdict is read from the point estimate as P2 is written, and a CI that crosses a threshold is named.
* **Prediction:** `R_tot` >= 5 (P2 as written would HOLD) because of the non-learned base and the frozen prior (W1), and NOT because of anything O learned (W2). Falsified if `R_tot` < 5.
* **Companion statistics (not the verdict):** `R_res` (W2), the learned-only ratio with the base and the frozen P10 MLP excluded; the same ratio in raw input units (sensitivity of the z-unit choice); the zero and shuffle ablation effects on `s` per group and per band (shuffle within the speed band is primary; global shuffle and zero reported).

## 3. 2b measurement protocol (fixed before the first forward)
* Windows: the T1-kin eval-141 windows (141 clips, 2,399 windows), tick 0 of each: logged history, dead-reckoned nav, causal speed bucket, a_t = a_s = 0, `scene logged`. Frozen-feature cache `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval`, grid `pool4x10`; join and grid under `dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw`. Model batch built by the tree's own `V8FMiniWindows.item` and `collate`. No post-processing (no `sg72`, no `kbox`): the raw `forward_infer` plan.
* Speed bands by the window's v0: [0,3), [3,8), [8,15), [15,inf). The P2 verdict reads [3,8).
* Held-out status: the eval-141 clips are the programme's evaluation clips, not training windows.
* The replicated forward must match `forward_infer` plans to 1e-5 m before any Jacobian is read (control).
* Controls that must read exactly: a Jacobian block of an input the model does not read (`pose`, `act`) is 0.0; the gradient of `s_res` w.r.t. the P10 prior's inputs through `base` is 0.0.
* No criterion above changes after the data are seen. Anything added later is labelled POST-HOC.
