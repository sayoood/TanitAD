# v7 — the ledger of PROVEN improvements

**Scope.** Every change to the v7 line that is supported by a measurement, from the
residual-head init bug forward. **Nothing hypothetical appears here.** Open
hypotheses live in `Project Steering/GOALS_AND_CLAIMS.md`; retracted claims live in
`RETRACTION_LOG.md`. If a row is here, it has an artifact path and an estimator.

**Tier.** ⛔ **T0-DIAGNOSTIC throughout.** No arm in this ledger has a planner, a
scorer or a closed loop. Per `EVAL_DOCTRINE.md` a T0 number may never be presented
as driving performance.

**Estimator conventions used below** (all four were themselves fixed this campaign):

| quantity | admissible estimator | why the previous one was void |
|---|---|---|
| decodability | **leave-one-episode-out PAIRED** — fit on n−1 episodes, score the held-out one, pair columns on the SAME episode | the episode bootstrap returns ±1.6 on a 0.2 effect at 12 clusters (each resample refits λ and the PCA basis) |
| prediction | **mean-centred cos vs a ≥100-draw permutation null**, report z; read h≥2 by **ROLLING** the h=1 head | divergence-over-movement divided by an ARM property spanning 468× (C137) |
| rank | participation ratio, **stated with corpus, episode count and ambient dimension** | the same encoder reads 5.756 / 20.228 on two corpora — 3.51× from episode diversity alone (C135) |
| "A beats B" | A and B **paired on the same units** | two deltas against a common third column are not a comparison (C138) |

Every panel carries a **CONSTANT control that must read exactly 0.0000** and a
**RAW-PIXEL floor**, and reports **EGO AND ENVIRONMENT** — ego alone hid a total
absence of scene content for an entire campaign.

---

## 1. `RESIDUAL_HEAD_INIT_SCALE = 1e-3` — the init bug

**The defect.** `forward` computes `out[k] = z_t + delta`, with `delta` produced from
a `LayerNorm`, whose output is O(1) *per dimension* whatever the latent's scale. v6's
operative latent has mean|z| **0.015581** and moves **0.000892 per tick** (MEASURED,
stride-1 latents at the true dt = 0.1 s). **A default-init head therefore starts
emitting a delta ~1000× larger than the movement it must predict**, and the run
spends itself shrinking it. v6F-SW-30k's own `o5_step1` was still **535× worse than
predicting NO CHANGE at step 20,000**, and a scalar rescale of the trained heads
could not rescue it — the error fell monotonically to α = 0, i.e. the learned delta
carried no signal.

**The fix.** Down-scale the residual heads at init by `1e-3`; env-overridable via
`TANITAD_RESIDUAL_INIT_SCALE`. Initialisation only — `state_dict` shapes unchanged,
so every existing checkpoint loads byte-identically.

⛔ **Down-scaled, NOT zero-init.** Zeroing the OUTPUT head sets
`dL/dh_last = Wᵀ·dL/dout = 0`, which stops gradient reaching the ENTIRE predictor
body — blocks, `in_proj`, `act_emb`, `intent_proj` all stall until the head leaves
zero. `test_v6_staged.py::test_planner_surface_is_total` caught exactly that:
`intent_proj` went invisible to the gradient probe. This is why FiLM (an *internal*
modulation, main path untouched) is zero-init while an *output* head must not be.

**Pinned by** `stack/tests/test_residual_init_scale.py` (17 tests).

---

## 2. `SubspaceSigReg` — Sub-JEPA's subspace regulariser

**Basis.** `PUBLISHED` (banked `2605.09241`): SIGReg's isotropic-Gaussian prior in
the FULL embedding space is an overly strong bias because latents live on low-dim
manifolds; Sub-JEPA applies Gaussian constraints over multiple **random orthonormal
subspaces** instead, and beats LeWM on 4 continuous-control environments.

**Implementation.** `K` random orthonormal subspaces of width `d_s = ⌊D/K⌉`, drawn
once by seeded QR and registered as a frozen buffer; the inner SIGReg statistic is
averaged over subspaces. `--sigreg-subspaces` (default 1 ⇒ the incumbent `SigReg`,
bit-identical).

⚠️ **Honest bound:** on our arms the subspace variant showed **no val-side gain**
(H-RANK-5 REFUTED for the estimator-conditioning hypothesis it was introduced to
test). It is retained because it is the published-correct form and costs nothing,
**not** because we measured it to help.

**Pinned by** `stack/tests/test_subspace_sigreg.py` (11 tests).

---

## 3. The λ confound in the SIGReg row bank

**The defect.** The Epps-Pulley statistic is deliberately **not n-normalised** — its
batch scale is part of LeJEPA's validated λ = 0.1 operating point. Pooling rows
across accumulation steps therefore **multiplied the effective λ**: 24 rows → `o6`
3.10; 192 rows → **46.3**. Two sweeps were invalidated before this was found.

**The fix.** Renormalise when the bank is larger than the base batch:
`l6 = l6 * (base_rows / z6.shape[0])`.

---

## 4. `--spectrum-accum` reaching the gate

**The defect.** The pooled spectrum was written to the log while `run_stage_gate`
received the **single batch** — so the preflight banner promised a gate that could
rule on a statistic the gate never saw.

**The fix.** Track `spectrum_pooled_last` and pass
`run_stage_gate(..., spectrum=(spectrum_pooled_last or spectrum_last))`.

---

## 5. ⭐ The readout geometry — the ego lever (E-DEC-2)

`z_op` is the operative latent **after** pooling 16×40 patch tokens to a 4×4 grid:
**four azimuth bins over a 120° CYLINDRICAL FOV = 30°/bin**. (The column axis is
linear in azimuth under this projection; ⛔ the pinhole formula gives 92.6° and is
wrong here — the rig's own name is `camera_front_wide_120fov`.)

**The information is in the encoder and the pooling discards it.** LOEO paired,
self-pairing control exactly 0.0000, `…/raw/e_dec1c_loeo.json`:

| target | `enc − z_op` | t | episodes | `enc − pixels` | t |
|---|---|---|---|---|---|
| speed | **+0.2117** | 7.86 | **12/12** | +0.2535 | 3.68 |
| `d_ego` | **+0.1691** | 12.38 | **12/12** | +0.3603 | 4.21 |

`z_op − pixels` is **not separable** (t 0.71) — after the pooling the latent does not
beat raw pixels.

**Training WITH a wider azimuth axis recovers it, at IDENTICAL latent dimension**
(`grid 4 × grid_w 8 × d_readout 64` = 2048 = the incumbent `d_op`),
`…/raw/e_dec2_arm_compare.json`:

| target | `lewm` (4×4×128) | **`rdw8` (4×8×64)** | Δ | t | episodes |
|---|---|---|---|---|---|
| speed | +0.0374 | **+0.2830** | +0.2456 | 5.17 | **12/12** |
| `d_ego` | +0.1532 | **+0.3601** | +0.2069 | 6.12 | **12/12** |

⚠️ **Two bounds that must travel with this decision.**
1. **`rdw20` (20 bins) is WORSE than `rdw8` when TRAINED** (+0.1331 speed, ±0 on
   `d_ego`) although a frozen-encoder re-pooling curve put the optimum at 20 bins.
   **The curve was a pointer, not a design.**
2. **Widening REVERSES on ENVIRONMENT targets**: it helps `n_agents` (+0.0830,
   t 7.68) but HURTS `lead_gap_m` (−0.0509, t −11.7, **0/24**) and `nearest_bearing`
   (−0.0683, t −13.6) — a scalar that depends on one region straight ahead is
   diluted by more bins. **`rdw8` is chosen on EGO evidence; the environment cost is
   real and must be re-measured at full scale.**

⇒ **v7 config: `--readout-grid 4 --readout-grid-w 8 --readout-dim 64`.**

---

## 6. ⭐⭐ `--o5-k ≥ 4` — a correctness fix, not a tuning choice (C139)

**The defect.** `rollout_transitions` builds `zhat_steps` from **`t[1]` — the h=1
head ONLY** — applied autoregressively `k_roll` times. With `--o5-k 1`, `k_roll = 1`,
and with O1/O2/O3 off **nothing else consumes the h=2 / h=4 heads**: they receive no
gradient in the entire run.

**Measured head weight norms** (`…/raw/h_proof6_head_norms.json`):

| arm | step | h=1 | h=2 | h=4 |
|---|---|---|---|---|
| champ30k | 30,000 | 4.29010 | **0.02612** | **0.02614** |
| rdw8 | 2,000 | 0.38423 | **0.02613** | **0.02615** |
| o7w1p0 | 2,000 | 0.41153 | **0.02613** | **0.02615** |
| lewm | 2,000 | 0.35806 | **0.02612** | **0.02614** |
| scale1 | 30,000 | 6.31850 | **0.02614** | **0.02612** |

**Identical to five significant figures across five arms, two architectures and a
15× range of training steps** — the untouched `1e-3` init. An untrained `1e-3` head
emits delta ≈ 0, so `ẑ = z_t`: exactly the 0.0002 "identity ratio" that was reported
as a property of the model.

⚠️ **v6F is CLEAN** (`o5_k = 60`, all six terms on ⇒ head norms **24.27 / 26.11 /
26.12**). **The condition was INTRODUCED BY THE TWO-TERM SIMPLIFICATION**, which
removed the only consumers of the h≥2 heads. It does not contaminate v6F or any
six-term result.

**The gain, measured by ROLLING the h=1 head** (`…/raw/h_proof7_rolled.json`):

| rolled step | k=1 | k=4 | **k=8** | k=8 vs k=1 |
|---|---|---|---|---|
| j=1 | 0.0428 (z 3.5) | 0.1140 (z 7.3) | **0.1198 (z 6.1)** | 2.80× |
| j=4 | 0.0693 (z 5.6) | 0.1778 (z 10.3) | **0.2051 (z 11.6)** | 2.96× |
| j=6 | 0.0686 (z 5.7) | 0.1987 (z 10.7) | **0.2468 (z 11.8)** | **3.60×** |

Monotone in depth, and **the advantage GROWS with horizon**. ⚠️ Absolute level still
low: cos 0.25 ≈ **6 % of variance explained** — *predicts* is not *predicts well*.

⇒ **v7 config: `--o5-k 8` — the MEASURED optimum. ⛔ NEVER 1, ⛔ not 16.**

⭐ **UPGRADED 2026-08-24 — `--o5-k 8` IS NOW THE MEASURED OPTIMUM, NOT AN EXTRAPOLATION.**
The "4 minimum, 8 better" line came from H-PROOF-7's **read-out-time** rolling depth `k_roll`,
which is a DIFFERENT KNOB from the **training-time** `--o5-k`. The training-time curve has now
been read out directly over four banked arms (`depth_panel.json`; `rdw8` **is** k=1):

| metric | k=1 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| participation val / held24 | 3.80/3.62 | 3.48/4.87 | 3.53/**6.39** | 4.27/4.21 |
| predictor cos h=1 | 0.0541 | 0.1234 | **0.1328** | 0.1327 |
| speed | +0.2830 | +0.3062 | **+0.3252** | **−0.2377** |
| `d_ego` | **+0.3601** | +0.2157 | +0.1212 | **−0.4543** |
| `lead_gap_m` | −0.3290 | −0.3495 | −0.2940 | **−0.2062** |
| `n_agents` | −1.0407 | −0.4434 | −0.4648 | −0.5241 |

* predictor cos **saturates at k=8** — k=16 adds nothing (0.1328 → 0.1327);
* ego **survives to k=8 and collapses at k=16** — speed falls below the constant control;
* `lead_gap_m` is the one metric that keeps wanting depth (monotone to k=16);
* ⚠️ **`d_ego` is BEST at k=1** and declines monotonically — the two ego targets dissociate,
  so depth **trades `d_ego` for speed and `lead_gap_m`**. Quote both or the curve reads as
  uniformly good.
* ⛔ at **every** depth the arm is **below the constant control on both environment targets**,
  and the raw-pixel floor beats it on `lead_gap_m` throughout. **Depth tunes the objective; it
  does not put the scene in the latent.**

⇒ **v7 ships `--o5-k 8`.** ⛔ never 1 (a correctness bug, C139) · ⛔ not 16 (ego collapses).


---

## 7. ⭐ External prediction targets — O7 / O8 (E-DEC-7/8/9/10)

**The root cause they address.** Every pre-existing term has a **self-generated
target**: O5 predicts our own next latent, O3 our own masked readout cells, O6 asks
only for isotropy, O1 only for per-action difference. Writing `z = (u, η)` with `u` a
low-dimensional ego code and `η` isotropic noise: O5 is small (kinematics are
smooth), O6 is satisfied **exactly** (η is isotropic by construction), participation
is high (η fills the dimensions). **"Ego + noise" is a non-collapsed optimum carrying
no scene content** — and it reproduces every number the campaign had measured.

⭐ `PUBLISHED` corroboration — PhyLatent (ICLR 2025, banked `2608.05720`):
*"preventing global latent collapse does not ensure that a representation preserves
physical states and action consequences."*

**O7 (`--w-o7-distill`) — distillation into a frozen teacher.** Proven in isolation
(`…/raw/e_dec8a_distill_readout.json`), an encoder of identical architecture trained
on **nothing but** MSE into frozen DINOv3 cells:

| target | two-term | **distilled** | Δ | t | episodes |
|---|---|---|---|---|---|
| `n_agents` | −1.0407 | **+0.3274** | +1.3681 | **12.63** | **24/24** |
| `lead_gap_m` | −0.3290 | −0.0330 | +0.2960 | 6.42 | 20/24 |
| speed (held-out) | +0.2830 | **+0.3940** | +0.1110 | 3.07 | 11/12 |

`n_agents` clears **zero and the raw-pixel floor for the first time in the
programme** — every prior arm sat *below a constant predictor* — and ego rises at the
same time. ⚠️ We do **not** claim to beat the teacher (C138): "level with" is what
the data supports.

⚠️ **In the full objective it reaches only −0.2553** — the self-referential terms
fight it. Raising the weight does not fix it: `o7w50` reaches the best in-objective
`n_agents` (+0.2214) but is **REJECTED by the kill-gate** (participation 2.57/3.13,
below baseline on BOTH held-out sets; `d_ego` −0.3703).

⛔ **O8 (`--w-o8-pixel`) REFUTED — externality alone is not sufficient.** The
teacher-free raw-pixel target REGRESSES ego hard (speed +0.2830 → **+0.0560**,
t −4.00, 1/12). Its target is **low-pass**: pooling averages 8×8 pixel blocks, so
structure finer than ≈8 px — roughly a distant vehicle — is destroyed. **The target's
CONTENT matters, not just its externality.**

⭐ **External targets do NOT hurt collapse resilience** (o7 rank 3.60/**4.52** vs
baseline 3.80/3.62) — unlike O1 (collapsed rank 4.43 → 2.94) and O3 (drove ego BELOW
the constant control).

Both default to **0.0** and neither head is CONSTRUCTED when off ⇒ loss, RNG stream
and `state_dict` bit-identical (verified: 118 passed on the determinism guards).
**Pinned by** `stack/tests/test_external_targets.py` (10 tests), including *the
target carries no gradient* and *the frozen teacher never enters our state_dict*.

---

## 8. ⭐ Two-stage training — the teacher is needed ONCE (E-DEC-12)

Init from the distilled encoder, then continue with **O5+O6 only, no teacher**
(`…/raw/postrain_o7_panel.json`):

| target | two-term | **distill → self-supervised** | Δ | t |
|---|---|---|---|---|
| `n_agents` | −1.0407 | **+0.1327** | +1.1734 | **12.29** |
| `lead_gap_m` | −0.3290 | −0.0714 | +0.2576 | 6.30 |
| `d_ego` | +0.3601 | +0.3122 | −0.0479 | −1.15 (n.s.) |
| cos h=1 | 0.0541 (z 3.99) | **0.1766 (z 7.11)** | — | best in-objective |

**Passes the kill-gate** (rank 3.35/3.74; held24 ABOVE baseline). ⇒ **the degeneracy
ERODES acquired content but does not destroy it** (+0.3274 → +0.1327 over 2k steps).
⚠️ Ego speed fell to +0.0084 but **t = −1.52, NOT significant** at n=12 — a longer
run is required before calling that erosion.

---

## 9. ⭐ Scale transfer — what tiny arms can and cannot tell us (H-SCALE-1/2)

`scale1` (enc 256×6 = **4×** the tiny encoder, 30k steps, PARITY corpus) vs `rdw8`
(0.97M, 2k), same `d_op` 2048 (`…/raw/scale1_panel.json`):

| | tiny | **scale1** |
|---|---|---|
| participation val / held24 | 3.80 / 3.62 | **8.54 / 7.54** |
| predictor cos h=1 | 0.0541 (z 3.99) | **0.6090 (z 15.65)** |
| `n_agents` | −1.0407 | **+0.0263** (positive) |
| speed | +0.2830 | **+0.0381** |

**Rank, prediction and environment all scale; ego decodability does not** — because
`scale1` uses the **old 4×4 readout** and reads speed +0.0381 against the tiny 4×4
arm's **+0.0374**. **4× encoder, 15× steps, 18× data — no change.**

⇒ **the readout caps ego decodability irrespective of scale**, and:
**H-SCALE-2 — screen ARCHITECTURE on tiny arms; NEVER quote a tiny-arm rank or
predictor number as capability.** Absolute values moved 2–11× with size.

⚠️ Confounded (size, steps and corpus all differ); the matched-readout contrast is
the clean part. `rdw8p30k` (rdw8 geometry, 30k, parity — guard verified
`e438721ae894`, sha256 matches manifest) is the disambiguating run.

---

## 9b. ⭐⭐⭐ The SPLIT ENCODER, and the control that says why it works (E-DEC-14/17)

**The arm.** Frozen distilled encoder + trainable readout and predictor, O5+O6
only, `--o5-k 4`, no teacher in the loop. `n_agents` **+0.4035** (paired LOEO,
t 13.45, **24/24**) — above pure distillation (+0.3274) and above a frozen
DINOv3 that never saw our data (+0.2754). Predictor cos h=1 **0.1872** (z 7.99),
the best measured in the programme.

⭐ **The freeze is verified by CONTENT, not by a flag.** All 41 `encoder.*`
tensors are byte-identical to the initialisation (sha256 match, **max|Δ| = 0**)
where every trainable arm moved 0.08–0.33. ⚠️ This check exists because the
arm's own `config.json` reads `'encoder': {'trainable': 972032, 'frozen': 0}` —
that block records the STAGE freeze plan, not the `--freeze-encoder` override,
and reading it as runtime state nearly retracted the result (**C146**).

**The control (E-DEC-17) — freeze a RANDOM encoder, change nothing else:**

| metric | two-term baseline | frozen **distilled** | frozen **random** | pixel floor | constant |
|---|---|---|---|---|---|
| predictor cos h=1 (z) | 0.1234 (6.61) | **0.1872 (7.99)** | **0.0016 (0.51)** | — | — |
| speed | +0.3062 | +0.2465 | **+0.3552** | +0.1156 | 0.0000 |
| `d_ego` | +0.2157 | +0.2878 | +0.2024 | −0.0781 | 0.0000 |
| `lead_gap_m` | −0.3495 | −0.0172 | +0.0064 | +0.0064 | 0.0000 |
| `n_agents` | −0.4434 | **+0.4035** | **−1.1027** | −0.2156 | 0.0000 |

⇒ **FREEZING IS NECESSARY BUT NOT SUFFICIENT.** On a frozen *random* encoder the
predictor is indistinguishable from noise (z 0.51); `n_agents` moves **1.5062**
between the two frozen arms and the random one sits below the constant control
*and* the pixel floor. The teacher supplied **content**, and a teacher-free
source of content is the remaining problem (E-DEC-18 / PSG).

⛔ **AND A STANDING CONSTRAINT ON THIS WHOLE LEDGER: EGO DECODABILITY IS FREE.**
The frozen *random* encoder has the **best speed of the three arms** (+0.3552)
and healthy `d_ego` (+0.2024) — from weights never trained on anything. That is
exactly what the `z = (u, η)` degeneracy predicts. **No ego number may be cited
as evidence that an objective worked**; a representation claim rests on scene
content or on prediction. Where this ledger says "ego retained", read it as *"the
arm did not go backwards"*, never as *"the arm learned something"*.

⚠️ The split arm **fails the rank kill-gate** (participation 2.28/2.17, the
lowest measured). That is not dismissed — it is weighed against C131, C135 and
the degeneracy derivation, all of which say participation does not track
capability — and it is why the parity-scale replication is queued rather than
the design being adopted on a 2 k tiny arm.

## 10. Other proven trainer changes

| change | flag | status |
|---|---|---|
| O1 confined to the predictor | `--o1-detach-encoder` | gradient topology **pinned** (`test_o1_detach_encoder.py`, 5 tests incl. two negative controls: encoder gradient exactly 0.0 while the predictor still receives it). Effect: **partially** decouples rank from action response (val 3.48 vs 3.42) but does NOT replicate on held24 (4.43 → 3.40); the restored response is 79× weaker than the joint path |
| encoder freeze | `--freeze-encoder` | implemented, default off; freeze confirmed in the run log (0.97 M params). Read-out in progress |
| O5 loss form | `--o5-form l1` | in use across the campaign |
| newest-frame-only data path | `--newest-frame-only` | implemented; `test_newest_frame_only.py` |

---

## 11. ⛔ What is NOT proven, and must not be read into this ledger

* **No driving claim.** T0 throughout.
* **Environment decodability inside the full objective is still BELOW a constant
  predictor.** Only the isolated (+0.3274) and two-stage (+0.1327) forms clear it.
* **We have never beaten frozen DINOv3** (C138 — paired: speed −0.1251, t −2.72).
* **Absolute prediction quality is low** — cos 0.25 ≈ 6 % of variance.
* **No teacher-free route to scene content has been found.** The pixel target is
  refuted; the EMA-target masked-latent route is pre-registered and unrun.
* **The subspace SIGReg showed no val-side gain** — retained on published grounds,
  not measured ones.
