# One-step and +2.0 s latent error of the level-1 predictor over the error of the copy

MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Ratio of summed squared errors over the held-out windows, predictor over copy; below 1 means better than the copy. Episode-cluster bootstrap, B 2,000.

## Controls (first)

* G_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* G_s1: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* P_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* P_s1: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* PR_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* PR_s1: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* PE_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* PE_s1: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* PF_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* RES_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* RR_s0: {"s1_copy_over_copy_is_1": true, "s2_shuffle_keeps_point": true}
* ALL_PASS: True

| arm | windows | one step, context 1 | one step, context 3 | +2.0 s from frame 0, context 1 | +2.0 s from frame 4, context 1 | +2.0 s from frame 4, true history |
|---|---|---|---|---|---|---|
| G_s0 | 1948 | 1.233 [1.180, 1.295] | 1.216 [1.162, 1.278] | 0.926 [0.869, 0.988] | 0.922 [0.867, 0.983] | 0.886 [0.835, 0.941] |
| G_s1 | 1948 | 1.254 [1.197, 1.317] | 1.238 [1.181, 1.302] | 0.951 [0.895, 1.013] | 0.945 [0.889, 1.007] | 0.900 [0.850, 0.956] |
| P_s0 | 1948 | 0.788 [0.770, 0.807] | 0.720 [0.705, 0.737] | 0.782 [0.745, 0.821] | 0.780 [0.742, 0.819] | 0.695 [0.667, 0.724] |
| P_s1 | 1948 | 0.796 [0.779, 0.814] | 0.731 [0.717, 0.746] | 0.775 [0.739, 0.812] | 0.772 [0.736, 0.810] | 0.702 [0.675, 0.731] |
| PR_s0 | 1948 | 0.844 [0.831, 0.857] | 0.829 [0.816, 0.842] | 0.751 [0.714, 0.789] | 0.749 [0.713, 0.788] | 0.643 [0.617, 0.670] |
| PR_s1 | 1948 | 0.873 [0.864, 0.883] | 0.861 [0.851, 0.872] | 0.762 [0.732, 0.794] | 0.761 [0.730, 0.793] | 0.664 [0.640, 0.689] |
| PE_s0 | 1948 | 0.785 [0.770, 0.801] | 0.728 [0.713, 0.742] | 0.769 [0.735, 0.805] | 0.767 [0.734, 0.804] | 0.704 [0.677, 0.733] |
| PE_s1 | 1948 | 0.791 [0.776, 0.807] | 0.735 [0.722, 0.749] | 0.775 [0.742, 0.811] | 0.774 [0.740, 0.808] | 0.718 [0.691, 0.746] |
| PF_s0 | 1948 | 0.914 [0.896, 0.933] | 0.805 [0.792, 0.819] | 1.625 [1.533, 1.726] | 1.618 [1.527, 1.718] | 0.810 [0.780, 0.841] |
| RES_s0 | 1948 | 0.829 [0.814, 0.845] | 0.809 [0.795, 0.825] | 0.780 [0.757, 0.804] | 0.779 [0.756, 0.804] | 0.763 [0.741, 0.786] |
| RR_s0 | 1948 | 0.870 [0.858, 0.882] | 0.874 [0.862, 0.886] | 0.689 [0.664, 0.715] | 0.687 [0.662, 0.713] | 0.681 [0.656, 0.707] |

Limits: held-out windows only; a ratio below 1 at +2.0 s is also what a shrinkage toward the mean state gives; one checkpoint per arm and seed.
