# PREREG_V8F_HBLEND: horizon-wise shrinkage of O's residual toward its own prior. Zero training, deployable (written 2026-10-25, before any HBLEND data)

**Status: pre-registered.** It runs eval-only on the BANKED M2b s0 / s1 checkpoints, on the dev box's CPU, under the PI's go of 2026-10-25 (`PI-V8F-2026-10-25b`: autonomous, allocated resources only). No training and no GPU.

## Why (MEASURED, `RESULT_FULL.md`, `raw/full/eval/`; T1-class open loop, eval-141)

v8F-mini beats its kinematic prior at 6 s and LOSES to it at short horizons and on acceleration:

| metric | P10-yaw-4512 | M2b s0 | M2b s1 |
|---|---|---|---|
| ADE6 (m) | 2.4870 | 2.1753 | 2.0546 |
| ADE2 (m) | 0.222 | 0.28–0.32 across the four full runs (to be re-read from the raw eval JSON by the instrument; INHERITED until then) | |
| accel MAE (m/s²) | 0.638 | 1.206 | 0.931 |

A planner that is worse than a constant-kinematics prior in its first two seconds, and twice as wrong on acceleration, does not dominate that prior. The first seconds are what a controller actually executes.

**The mechanism this targets.** O's plan is $\tau = \mathrm{up}(P10) + r$, where the residual $r = B\,c$ lives on a ramped DCT basis (`v8f_mini.py`, `run_o`). The ramp makes $r$ small near $t = 0$, but not zero, and it is trained on one loss over all 60 steps with whole-code teacher forcing. At eval, with T's own code, the residual's short-horizon part may carry more error than signal. Shrinking it there is the classical bias–variance answer (James–Stein shrinkage toward a prior), and it uses only model outputs at inference.

## The arm (ONE lever: a fixed per-step, per-axis shrinkage weight on O's residual)

$$\tau_H(t) = \mathrm{up}(P10)(t) + w_a(t)\,\big(\tau(t) - \mathrm{up}(P10)(t)\big)_a, \qquad a \in \{\text{along}, \text{cross}\}\ \text{(the ego frame at NOW)}$$

* **The weights are fitted on TRAINING windows only**, in closed form per step and axis:
  $$w_a(t) = \mathrm{clip}\Big(\tfrac{\sum r_a(t)\,(y_a(t) - P_a(t))}{\sum r_a(t)^2},\ 0,\ 1\Big)$$
  * The sums run over a deterministic subsample of B's training windows: every k-th train clip in sorted-sha order, with the smallest k whose subsample fits a 3 GB copy budget and gives at least 100 clips; k is declared. About 21 MB per clip on box 2's cache, so about 140 clips.
  * ⚠️ **The bias of a train-window fit is CONSERVATIVE, by construction.** O was trained on these windows, so its residual is at least as accurate there as on eval. The fitted weights therefore shrink LESS than an eval-optimal weight would, and the eval read under-states the lever, never over-states it.
  * The fit is then made **monotone non-decreasing in t** (isotonic), separately per seed, each from its own checkpoint.
  * $y$ is the logged future.
  * ⛔ The eval-141 windows never enter the fit.
* **Fallback, declared now:** if the train cache cannot be reached within the disk budget, use a **2-fold episode cross-fit on eval-141**. The split is by parity of the sorted-episode index: fit on one fold, score the other, then swap. It is reported as CROSS-FITTED, not as a deployable weight.
* Inference reads only model outputs (P10 and τ): vision-only, no label, no situation classifier. The weights are 120 constants per seed.
* **Identity controls:** with $w \equiv 1$ the eval must reproduce B's banked per-window predictions (max|Δ| ≤ 1e-4 m, CPU vs box 2's GPU) and B's ADE6 to 1e-4. With $w \equiv 0$ it must reproduce P10-yaw-4512's ADE6, 2.4870, to 1e-3.

## The reading (paired episode-cluster bootstrap, B = 2,000, 141 episodes; F = 0.1207 m; inference is deterministic, so no inference-seed floor applies)

| read | definition | bar |
|---|---|---|
| **PRIMARY-S (short horizon)** | ADE2: H − B | **< 0 and separated, at BOTH seeds** |
| **PRIMARY-N (no harm at 6 s)** | ADE6: H − B | **upper 95 % bound ≤ +F/2 = +0.060 m at BOTH seeds** |
| **SECONDARY-A** | accel MAE: H − B | < 0, separated, both seeds |
| **DOMINANCE vs P10** | ADE2, ADE6 and accel MAE of H against P10-yaw-4512 | descriptive: "H dominates P10" iff all three point estimates are ≤ P10's and ADE6 is separated |
| families | longitudinal (incl. distance keeping), lateral, tactical (unchanged by construction: T is untouched), strategic (unchanged: S is untouched) | beside, per family. Tactical and strategic must read EXACTLY B's values (a structural identity) |
| horizons | ADE / FDE at 1, 2, 3, 4, 5, 6 s | beside |

| outcome | condition | next (committed) |
|---|---|---|
| **PASS** | PRIMARY-S and PRIMARY-N at both seeds | H becomes the deployed v8F output. The shrinkage weights become a model buffer in the next trained arm, and the FLAT and MIXO reads are re-scored with H applied to each arm's own fitted weights (descriptive) |
| **SHORT ONLY** | PRIMARY-S passes, PRIMARY-N fails | the residual's 6 s signal and its short-horizon error are entangled. Next: a TRAINED horizon-weighted loss arm under its own PREREG (box 2) |
| **NO GAIN** | PRIMARY-S fails | the short-horizon loss is not residual noise that shrinkage can remove. Next: a diagnosis of the short-horizon error by component (P10's own error vs the residual vs the speed code), before any training arm |

⛔ The weights are never re-fitted after the eval is seen, and the functional form, the clip range and the isotonic step are fixed above.
