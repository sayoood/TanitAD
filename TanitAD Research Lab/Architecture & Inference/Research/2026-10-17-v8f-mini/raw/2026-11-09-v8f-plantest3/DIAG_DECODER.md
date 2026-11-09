# DIAG_DECODER: how much speed and yaw rate can be read from M2b's level-1 latents? (Master Mind, 2026-11-09)
Beside `PREREG_V8F_PLANTEST3.md` (`1f9609b4`), deciding nothing. Tier T0, a readout diagnostic on encoded latents, never a driving result. MEASURED: `runs/pre_M2b_s*.json`, `DIAG_RIDGE.log`, `DIAG_MULTIFRAME.json`, `DIAG_MLP.json`. Every read uses the card's TUNE inner split (53 fit episodes, 17 held-out episodes, 238 held-out windows); no SCORE window is read. R2 is pooled over held-out (window, knot) samples; it has no bootstrap interval, so small differences here carry no weight.

| readout (M2b s0 / s1) | speed R2 | yaw-rate R2 | bar (card) |
|---|---|---|---|
| card's gate: one latent, ridge 1e-3 | 0.271 / 0.314 | -0.122 / 0.268 | 0.70 / 0.50 |
| linear, stack of 1 to 8 past latents, best of three ridge values (chosen on the held-out episodes, so optimistic) | up to 0.323 / 0.331 | up to 0.277 / 0.424 | |
| MLP, one latent | 0.420 / 0.387 | 0.348 / 0.222 | |
| MLP, four latents | 0.406 / 0.355 | 0.431 / 0.105 | |

* **The card's DECODER-WEAK flag is raised at both seeds.** The run continues as the card orders.
* **Stacking past latents does not lift the linear read.** Single-frame blindness to motion is therefore not the explanation. The yaw target's alignment is not the explanation either: the rate ahead (`A[t]`, the gate's target) and the rate behind (`A[t-1]`) read within 0.004 of each other at m = 1 (best ridge).
* **A small nonlinear readout adds about 0.1 on speed and stays far below the bar.** Shuffled-target controls read -0.281 to +0.061 (linear) and -0.143 to +0.028 (MLP).
* **Limits.** About 11,000 training samples from 42 episodes. A weak probe bounds what this readout finds with this data; it does not show that the information is absent or unlearnable.
* **Structural fact (source, `tanitad/models/v8f_mini.py:666-669`):** `z1` is the mean-pooled vision adapter token, projected. The measured ego state reaches the readers S, T and O; it does not enter the world-model latent. A planner searching in that latent has weak access to the quantities its actions change.

Committed next under the card (DECODER-WEAK): a nonlinear readout and multi-frame decoding on the ladder latents. HYPOTHESIS for a further card, not tested here: the measured ego state as part of the world-model state.
