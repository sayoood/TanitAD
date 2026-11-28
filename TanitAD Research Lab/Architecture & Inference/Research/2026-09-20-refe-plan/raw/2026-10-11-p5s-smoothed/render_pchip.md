n = 1018 rows, 11 logs, 0 rows raised; 270 s.

**Controls (§4)**

| control | result |
|---|---|
| C1_identity | PASS |
| C2_knot_exact | PASS |
| C4_d1_join | PASS |
| C5_review7 | PASS |
| C6_margin_direction | PASS |
| C7_batch_independence | PASS |
| C3_wrong_sign | PASS |

raw: knot_exact True, knot_checks 9162, ident_label_bit_exact True, ident_p8_bit_exact True, ident_sim8_eq_T True, batch_indep_max_m 0.0, margin_neg_not_interp8 0, d1_sim8_mismatch 0, d1_missing 0, r7_dac_T 1018, r7_dac_H 1018, r7_ddc_T 1018, r7_ddc_H 1018, r7_lk_T 1018, r7_lk_H 1018, r7_n 1018, knot_maxdev 4.440892098500626e-16

**Per reference, scored on its executed trajectory (§3)**

| ref | intended | n | sim8 fail | interp8 fail | DDC mean | DDC < 1 | lk10 violation | comfort | path-length ratio (median) |
|---|---|---|---|---|---|---|---|---|---|
| T | T | 1018 | 8.06 % | 0.59 % | 0.9818 | 2.75 % | 13.36 % | 100.00 % | 0.9972 |
| Tc20 | T | 1018 | 0.88 % | 13.06 % | 0.9887 | 1.96 % | 10.90 % | 99.90 % | 1.0009 |
| Tc8 | T | 1018 | 0.98 % | 13.16 % | 0.9887 | 1.96 % | 10.90 % | 99.90 % | 1.0009 |
| Tw | T | 1018 | 34.09 % | 36.25 % | 0.8841 | 16.11 % | 16.90 % | 88.21 % | 0.9952 |
| ident | T | 1018 | 8.06 % | 0.59 % | 0.9818 | 2.75 % | 13.36 % | 100.00 % | 0.9972 |
| H | H | 1018 | 0.00 % | 0.79 % | 0.9764 | 3.24 % | 13.95 % | 100.00 % | 1.0089 |
| Hc | H | 1018 | 0.59 % | 14.44 % | 0.9823 | 2.55 % | 11.98 % | 99.51 % | 1.0057 |

**Executed vs intended** (mean / median / p95)

| ref | intended | max_xy | along_4s | mean_abs_along | mean_abs_cross | heading_mae_deg |
|---|---|---|---|---|---|---|
| T | T | 0.473 / 0.414 / 1.021 | -0.075 / -0.131 / 0.667 | 0.170 / 0.140 / 0.411 | 0.121 / 0.065 / 0.392 | 1.645 / 0.801 / 5.792 |
| Tc20 | T | 0.073 / 0.057 / 0.185 | 0.018 / 0.016 / 0.075 | 0.022 / 0.016 / 0.054 | 0.028 / 0.019 / 0.084 | 0.656 / 0.420 / 1.903 |
| Tc8 | T | 0.072 / 0.057 / 0.184 | 0.019 / 0.016 / 0.076 | 0.022 / 0.016 / 0.053 | 0.028 / 0.019 / 0.084 | 0.654 / 0.417 / 1.904 |
| Tw | T | 3.041 / 2.673 / 6.166 | -0.668 / -1.008 / 4.565 | 1.038 / 0.859 / 2.443 | 0.630 / 0.346 / 2.092 | 8.077 / 3.561 / 30.468 |
| ident | T | 0.473 / 0.414 / 1.021 | -0.075 / -0.131 / 0.667 | 0.170 / 0.140 / 0.411 | 0.121 / 0.065 / 0.392 | 1.645 / 0.801 / 5.792 |
| H | H | 0.553 / 0.498 / 1.047 | 0.089 / 0.165 / 0.841 | 0.263 / 0.247 / 0.535 | 0.148 / 0.078 / 0.505 | 1.383 / 0.572 / 5.612 |
| Hc | H | 0.195 / 0.183 / 0.366 | 0.092 / 0.099 / 0.245 | 0.113 / 0.103 / 0.238 | 0.070 / 0.049 / 0.201 | 0.713 / 0.324 / 2.770 |

**Executed vs H (descriptive)**

| ref | max_xy | mean_abs_along | mean_abs_cross | heading_mae_deg |
|---|---|---|---|---|
| T | 2.623 | 1.043 | 0.247 | 2.771 |
| Tc20 | 2.760 | 1.099 | 0.224 | 2.558 |
| Tc8 | 2.760 | 1.099 | 0.224 | 2.557 |

**Arm (ii): Tc20 − T, paired log-cluster bootstrap (§6)** — verdict **CONFIRMED**; adverse guards: none

| metric | Δ [95 % CI] |
|---|---|
| sim8 | -7.17 pp [-13.50, -1.83] |
| ddc | +0.69 pp [+0.00, +1.99] |
| comfort | -0.10 pp [-0.41, +0.00] |
| lk10 | -2.46 pp [-5.58, +0.51] |
| mean_abs_along | -0.1481 [-0.1830, -0.1221] |
| mean_abs_cross | -0.0930 [-0.1256, -0.0534] |
| max_xy | -0.3999 [-0.4594, -0.3393] |
| heading_mae_deg | -0.9891 [-1.3727, -0.5050] |

**Control 3, wrong sign: Tw − T (same estimator)** — verdict **REFUTED**; adverse guards: ddc, comfort, mean_abs_along, mean_abs_cross, max_xy, heading_mae_deg

| metric | Δ [95 % CI] |
|---|---|
| sim8 | +26.03 pp [+15.23, +35.40] |
| ddc | -9.77 pp [-16.43, -3.54] |
| comfort | -11.79 pp [-18.80, -5.05] |
| lk10 | +3.54 pp [-0.43, +6.73] |
| mean_abs_along | +0.8686 [+0.7427, +1.0545] |
| mean_abs_cross | +0.5085 [+0.2941, +0.6814] |
| max_xy | +2.5683 [+2.2321, +2.9179] |
| heading_mae_deg | +6.4316 [+3.6572, +8.6302] |

**Learnability** (flag: **none**; RISK iff a Tc20 median > 2× t20's)

| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |
|---|---|---|---|---|
| t20 | 1.371 | 3.136 | 5.110 | 32.742 |
| Tc20 | 2.497 | 5.353 | 9.151 | 45.598 |
| Tw | 2.284 | 11.065 | 10.543 | 84.020 |

**D1 class breakdown of what Tc20 repairs and breaks**

| D1 class | n | T sim8 fail | Tc20 sim8 fail | repaired | broken |
|---|---|---|---|---|---|
| pass | 934 | 0 | 2 | 0 | 2 |
| C_navsim_tracker | 78 | 78 | 1 | 77 | 0 |
| A_teacher_path | 6 | 4 | 6 | 0 | 2 |

**Arm (i): margin < 0.6 m on T's plan footprint** — verdict **NOT VIABLE**

- at 0.6 m: flagged 469 (46.07 %), recall 85.37 % (70 of 82 sim8 failures), precision 14.93 %
- drop-if-sim8 comparison point: flagged 8.06 %, recall 100.00 %
- margin of T: median 0.632 m, p05 0.234 m; H median 0.685 m (descriptive)

| threshold m (descriptive sweep) | flagged share | recall | precision |
|---|---|---|---|
| 0.3 | 12.08 % | 29.27 % | 19.51 % |
| 0.45 | 28.00 % | 69.51 % | 20.00 % |
| 0.6 | 46.07 % | 85.37 % | 14.93 % |
| 0.75 | 61.89 % | 97.56 % | 12.70 % |

**Iteration traces** (median max-xy deviation, m, k = 0..3)

| ref | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| Tc20 | 0.4136 | 0.1214 | 0.0744 | 0.0576 |
| Tw | 0.4136 | 0.7663 | 1.4392 | 2.6729 |
| ident | 0.4136 | 0.4136 | 0.4136 | 0.4136 |
| Tc8 | 0.4136 | 0.1212 | 0.0739 | 0.0573 |

`max |to_navsim(Tc20) − Tc8|`: median 0.0062 m, max 0.2559 m.

**Dual compliance: executed path passes sim8 AND the drawn 8 poses pass interp8** (descriptive, counted from `rows`)

| ref | n | sim8 fail | interp8 fail | fail both | pass both |
|---|---|---|---|---|---|
| T | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| Tc20 | 1018 | 9 | 133 | 6 | 882 (86.64 %) |
| Tc8 | 1018 | 10 | 134 | 7 | 881 (86.54 %) |
| Tw | 1018 | 347 | 369 | 336 | 638 (62.67 %) |
| ident | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| H | 1018 | 0 | 8 | 0 | 1010 (99.21 %) |
| Hc | 1018 | 6 | 147 | 3 | 868 (85.27 %) |
| Tsel (DERIVED: Tc20 where T fails sim8, else T) | 1018 | 5 | 54 | 2 | 961 (94.40 %) |

Tc20 interp8 failures: 133; on rows where T's interp8 passes: 130; on rows where T fails sim8: 52.
Learnability on the 82 rows Tsel compensates (median per-row max): |acc| t20 2.456 vs Tc20 4.306 m/s² (1.75×); |heading rate| t20 26.867 vs Tc20 36.018 °/s (1.34×).

**Four families** (tier T1, n 1018, 11 logs, value ± 95 % half-width; descriptive only). STRATEGIC: UNAVAILABLE in NAVSIM; PAUSED (PI ruling 2026-09-19).

| pair (executed vs intended) | speed MAE m/s | along MAE m | accel MAE m/s² | heading MAE ° | yaw-rate MAE °/s | curvature MAE 1/m | cross MAE m | lat-decision κ | lon-decision κ | goal-point err m | STRATEGIC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exec_T_vs_T | 0.105 ± 0.013 | 0.167 ± 0.027 | 0.139 ± 0.013 | 1.530 ± 0.544 | 2.128 ± 0.714 | 0.0101 ± 0.0040 | 0.120 ± 0.042 | 0.971 [0.959, 0.980] | 0.971 [0.952, 0.992] | 0.429 [0.362, 0.497] | UNAVAILABLE |
| exec_Tc20_vs_T | 0.026 ± 0.003 | 0.024 ± 0.004 | 0.075 ± 0.008 | 0.577 ± 0.156 | 1.150 ± 0.295 | 0.0061 ± 0.0024 | 0.027 ± 0.007 | 0.966 [0.935, 0.977] | 0.992 [0.988, 0.998] | 0.052 [0.040, 0.064] | UNAVAILABLE |
| exec_Tc8_vs_T | 0.025 ± 0.003 | 0.024 ± 0.004 | 0.075 ± 0.008 | 0.575 ± 0.156 | 1.147 ± 0.295 | 0.0060 ± 0.0024 | 0.026 ± 0.007 | 0.967 [0.938, 0.978] | 0.992 [0.988, 0.998] | 0.052 [0.040, 0.064] | UNAVAILABLE |
| exec_Tw_vs_T | 0.610 ± 0.095 | 0.974 ± 0.153 | 0.551 ± 0.058 | 7.075 ± 2.547 | 8.896 ± 3.448 | 0.0369 ± 0.0151 | 0.677 ± 0.257 | 0.744 [0.627, 0.809] | 0.547 [0.448, 0.679] | 2.896 [2.519, 3.330] | UNAVAILABLE |
| exec_T_vs_H | 0.668 ± 0.153 | 1.014 ± 0.217 | 0.394 ± 0.058 | 2.618 ± 0.818 | 2.846 ± 0.817 | 0.0129 ± 0.0046 | 0.309 ± 0.090 | 0.885 [0.826, 0.923] | 0.483 [0.309, 0.671] | 2.592 [1.962, 3.198] | UNAVAILABLE |
| exec_Tc20_vs_H | 0.712 ± 0.154 | 1.053 ± 0.211 | 0.458 ± 0.055 | 2.341 ± 0.664 | 2.865 ± 0.789 | 0.0134 ± 0.0048 | 0.305 ± 0.087 | 0.874 [0.817, 0.911] | 0.479 [0.308, 0.668] | 2.719 [2.069, 3.350] | UNAVAILABLE |
| exec_Tc8_vs_H | 0.712 ± 0.154 | 1.053 ± 0.211 | 0.458 ± 0.054 | 2.340 ± 0.664 | 2.864 ± 0.790 | 0.0134 ± 0.0048 | 0.304 ± 0.087 | 0.872 [0.814, 0.909] | 0.479 [0.308, 0.668] | 2.719 [2.069, 3.350] | UNAVAILABLE |
| exec_H_vs_H | 0.121 ± 0.018 | 0.268 ± 0.034 | 0.128 ± 0.018 | 1.564 ± 0.542 | 1.956 ± 0.549 | 0.0104 ± 0.0035 | 0.145 ± 0.056 | 0.963 [0.941, 0.985] | 0.907 [0.882, 0.929] | 0.486 [0.437, 0.553] | UNAVAILABLE |
| exec_Hc_vs_H | 0.051 ± 0.008 | 0.123 ± 0.026 | 0.100 ± 0.013 | 0.900 ± 0.241 | 1.602 ± 0.399 | 0.0089 ± 0.0031 | 0.062 ± 0.021 | 0.978 [0.964, 0.992] | 0.933 [0.906, 0.954] | 0.156 [0.122, 0.191] | UNAVAILABLE |
