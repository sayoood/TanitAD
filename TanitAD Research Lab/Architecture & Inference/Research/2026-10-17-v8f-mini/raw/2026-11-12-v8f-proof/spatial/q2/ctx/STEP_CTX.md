# One-step and +2.0 s latent error of the level-1 predictor over the error of the copy

MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Ratio of summed squared errors over the held-out windows, predictor over copy; below 1 means better than the copy. Episode-cluster bootstrap, B 2,000.

## Controls (first)

* G_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* G_s1: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* W_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* P_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* ALL_PASS: True

| arm | windows | one step, context 1 | one step, context 3 | +2.0 s from frame 0, context 1 | +2.0 s from frame 4, context 1 | +2.0 s from frame 4, true history |
|---|---|---|---|---|---|---|
| G_s0 | 1948 | 1.233 [1.180, 1.295] | 1.216 [1.162, 1.278] | 0.926 [0.869, 0.988] | 0.922 [0.867, 0.983] | 0.886 [0.835, 0.941] |
| G_s1 | 1948 | 1.254 [1.197, 1.317] | 1.238 [1.181, 1.302] | 0.951 [0.895, 1.013] | 0.945 [0.889, 1.007] | 0.900 [0.850, 0.956] |
| W_s0 | 1948 | 1.026 [1.013, 1.042] | 0.974 [0.956, 0.992] | 1.156 [1.084, 1.230] | 1.166 [1.092, 1.243] | 1.031 [0.975, 1.090] |
| P_s0 | 1948 | 0.788 [0.770, 0.807] | 0.720 [0.705, 0.737] | 0.782 [0.745, 0.821] | 0.780 [0.742, 0.819] | 0.695 [0.667, 0.724] |

Limits: held-out windows only; a ratio below 1 at +2.0 s is also what a shrinkage toward the mean state gives; one checkpoint per arm and seed.
