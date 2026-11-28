# PREREG_V8F_HSPEC: does H dominate its kinematic prior on EVERY family once its residual is shrunk per cosine coefficient? (written 2026-10-25, before any HSPEC data)

**Status: pre-registered.** Zero training. Offline, from banked outputs, on the dev box's CPU, under `PI-V8F-2026-10-25b` and the PI's 2026-10-25 instruction *"Go ahead with your proposed ideas … our priority is to proof very quickly dominance"*.

## Why (MEASURED; `raw/2026-10-25-hblend/` §5–6, paired episode-cluster bootstrap, eval-141)

Against P10-yaw-4512, H (HBLEND, PASSED) is already separated-better on:
* ADE2 and ADE6;
* speed MAE and along MAE;
* cross-track and heading MAE;
* both plan-derived tactical agreements;
* CONSISTENCY(a) 'any' (0.260 vs 0.311).

It is separated-WORSE on exactly two metrics, at both seeds, and both are second-derivative metrics:

| metric | H − P10 s0 | H − P10 s1 |
|---|---|---|
| accel MAE (m/s²) | +0.104 [+0.080, +0.134] | +0.066 [+0.040, +0.102] |
| curvature MAE (1/m) | +0.0074 [+0.0036, +0.0119] | +0.0110 [+0.0045, +0.0190] |

Yaw-rate MAE, also a second-derivative metric, is worse but not separated (+0.09 / +0.11 deg/s). HBLEND's weights are per STEP, so they cannot tell the residual's smooth part from its rough part. Its own report named the next lever: shrink per cosine coefficient (`RESULT_V8F_HBLEND.md` §8.3).

## The arm (ONE lever on top of H)

O's residual is $r = \mathbf{B}c$, with $c \in \mathbb{R}^{12\times2}$ on the ramped DCT basis (`v8f_mini.py`, `basis`, `head_o`). Recover $c$ per window from the banked `traj30 − p30` by least squares on the basis rows at the 0.2 s points; the recovery residual must be ≤ 1e-4 m (control 1).

$$\tau_{HS} = \mathrm{up}(P) + w \odot \mathbf{B}\,(v \odot c)$$

* $w$ = H's banked per-step, per-axis weights (unchanged).
* $v \in [0,1]^{12\times2}$, one weight per coefficient and channel.

**Stage 1 (position objective):** fit $v$ by bounded least squares (`scipy.optimize.lsq_linear`, bounds [0, 1]) of $\tau_{HS}$ against the GT on HBLEND's training-window set (9,582 windows, 141 clips, k = 32), per seed. No monotonicity constraint, no tuning on eval.

**Stage 2 (runs AUTOMATICALLY if stage 1's PRIMARY-DOM fails):** the same fit with the objective
$$\|\tau - y\|^2 + \lambda\,\|\Delta^2 \tau - \Delta^2 y\|^2$$
where $\Delta^2$ is the second difference on the 0.2 s grid. $\lambda$ is fixed by a TRAIN-only unit-balancing rule: the ratio of the two terms' sums at $v = 1$ on the training windows. It is computed once, reported, and never tuned.

## Controls

1. Coefficient recovery residual ≤ 1e-4 m; with $v = 1$ the plan equals H exactly (Δ = 0.0 through the eval's own `evaluate()`).
2. With $v = 0$ the plan equals P10 exactly.
3. **Deliberate regression:** with the GT replaced by P10 in the fit, $v$ collapses to 0, and the plan reads as P10, never as a gain.
4. **Noise control:** with $c$ permuted across windows in the fit, $v$ collapses toward 0.

## The reading (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m; the eval's OWN `evaluate()` and `families()`; distance keeping through `v8f_dk_read.read_dk`)

| read | definition | bar |
|---|---|---|
| **PRIMARY-DOM** (the dominance row) | HS − P10 at BOTH seeds | **separated-better** on ADE2, ADE6, accel MAE and curvature MAE; AND **not separated-worse** on any other metric of the four families: speed, along, cross-track, heading, yaw-rate, both plan-derived tactical agreements, CONSISTENCY(a) 'any', distance keeping (headway, time-gap, LEAD-VIOL) |
| **PRIMARY-KEEP** | ADE6 HS − H | upper bound ≤ +F/2 = +0.060 m at both seeds |
| beside | HS − B on everything; $v$ per coefficient; the per-horizon ADE table; CONSISTENCY(a) per rule | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **DOMINATES ITS PRIOR** | PRIMARY-DOM and PRIMARY-KEEP at both seeds, at stage 1 or stage 2 | HS becomes the deployed v8F output, with $v$ a model buffer beside $w$. The register row states it as dominance over the programme's **kinematic floor** only, ⛔ never as state of the art. Next: the same dominance row against FLAT (`PREREG_V8F_FLAT`) |
| **SMOOTHER, NOT DOMINANT** | accel and curvature HS − H < 0 separated, but PRIMARY-DOM fails after stage 2 | the roughness is partly in the coefficients' amplitudes, and partly elsewhere. Next: a training arm with a second-difference penalty on O's plan, on box 2 (a loss term: G-DVB entry and launch gate) |
| **NO EFFECT** | neither | the roughness is not in O's residual. Next: read the prior's up-sampling and the 0.2 s subsampling as the source (zero training) |
| **GAIN LOST** | PRIMARY-DOM passes, PRIMARY-KEEP fails | reported; HS is not deployed over H |

⛔ No objective, $\lambda$ rule, bound or bar changes after any HSPEC number is seen.
