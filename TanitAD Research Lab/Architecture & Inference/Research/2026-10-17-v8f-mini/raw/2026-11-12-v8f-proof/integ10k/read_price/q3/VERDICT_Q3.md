# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RQ50"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RQ50 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0580 / +0.0565 | +0.0572 | [-0.0338, +0.1486] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3333 / +0.3344 | +0.3339 | [+0.1600, +0.5405] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0402 / -0.0240 | -0.0321 | [-0.0581, -0.0072] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5669 / +0.5337 | +0.5503 | [+0.3387, +0.7956] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0198 / +0.0164 | +0.0181 | [+0.0109, +0.0264] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.8792 / +0.7800 | +1.3296 | [-0.2464, +2.7075] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0054 / +0.0000 | -0.0027 | [-0.0131, +0.0071] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0100 / +0.0163 | +0.0131 | [-0.0004, +0.0271] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0063 / +0.0063 | +0.0063 | [-0.0019, +0.0146] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3888 / +0.4088 | +0.3988 | [+0.1675, +0.6661] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2359 / +0.1729 | +0.2044 | [+0.0569, +0.3882] | 0.0229 | WORSE |
| ade6_m|bulk | +0.3479 / +0.4382 | +0.3931 | [+0.1649, +0.6635] | 0.1243 | WORSE |
| ade6_m|all | +0.2870 / +0.2939 | +0.2904 | [+0.1249, +0.4841] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3888 / +0.4088 | +0.3988 | [+0.1675, +0.6661] | 0.0699 | WORSE |

## I10RQ50 minus I10RLb

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0190 / -0.0108 | -0.0149 | [-0.0687, +0.0353] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0271 / +0.0464 | +0.0368 | [-0.0535, +0.1258] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0174 / -0.0010 | -0.0092 | [-0.0340, +0.0116] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0032 / +0.1541 | +0.0787 | [-0.1467, +0.3115] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0095 / +0.0022 | +0.0058 | [-0.0027, +0.0142] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.1750 / -0.3321 | -0.0785 | [-1.2879, +1.0191] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0054 / -0.0050 | -0.0052 | [-0.0108, +0.0002] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0033 / +0.0071 | +0.0052 | [-0.0042, +0.0152] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0004 / -0.0017 | -0.0006 | [-0.0056, +0.0048] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0126 / +0.0521 | +0.0324 | [-0.0952, +0.1530] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0953 / +0.0396 | +0.0675 | [-0.0197, +0.1947] | 0.0229 | OPEN |
| ade6_m|bulk | -0.0825 / +0.0281 | -0.0272 | [-0.1820, +0.1085] | 0.1243 | OPEN |
| ade6_m|all | +0.0142 / +0.0344 | +0.0243 | [-0.0658, +0.1109] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0126 / +0.0521 | +0.0324 | [-0.0952, +0.1530] | 0.0699 | OPEN |

## I10RQ50 minus I10RR

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1891 / -0.1337 | -0.1614 | [-0.2410, -0.0939] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.4641 / -0.3735 | -0.4188 | [-0.6109, -0.2589] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0244 / -0.0094 | -0.0169 | [-0.0372, +0.0069] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.5703 / -0.3069 | -0.4386 | [-0.7060, -0.1944] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0088 / -0.0173 | -0.0130 | [-0.0231, -0.0059] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.3130 / -1.8685 | -2.0907 | [-3.0994, -1.1893] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0133 / +0.0004 | -0.0065 | [-0.0192, +0.0042] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0154 / -0.0104 | -0.0129 | [-0.0233, -0.0040] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0117 / -0.0008 | -0.0063 | [-0.0177, +0.0029] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.6414 / -0.4988 | -0.5701 | [-0.8352, -0.3492] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2469 / -0.2666 | -0.2568 | [-0.3962, -0.1374] | 0.0229 | BETTER |
| ade6_m|bulk | -0.6965 / -0.4751 | -0.5858 | [-0.9320, -0.2856] | 0.1243 | BETTER |
| ade6_m|all | -0.4519 / -0.3617 | -0.4068 | [-0.5919, -0.2516] | 0.0442 | BETTER |
| ade_tac_m|all | -0.6414 / -0.4988 | -0.5701 | [-0.8352, -0.3492] | 0.0699 | BETTER |

## I10RQ35 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1300 / +0.1139 | +0.1219 | [+0.0164, +0.2264] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5254 / +0.4845 | +0.5050 | [+0.2856, +0.7420] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0314 / -0.0185 | -0.0249 | [-0.0527, +0.0079] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.7663 / +0.5952 | +0.6807 | [+0.4226, +0.9774] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0227 / +0.0225 | +0.0226 | [+0.0149, +0.0312] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.8072 / +1.7839 | +2.2955 | [+0.6663, +3.8530] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0025 / -0.0025 | +0.0000 | [-0.0106, +0.0108] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0192 / +0.0200 | +0.0196 | [+0.0052, +0.0346] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0129 / +0.0088 | +0.0108 | [+0.0015, +0.0213] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.6537 / +0.6086 | +0.6311 | [+0.3389, +0.9426] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2799 / +0.2883 | +0.2841 | [+0.1105, +0.4787] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6992 / +0.6189 | +0.6590 | [+0.3351, +1.0049] | 0.1243 | WORSE |
| ade6_m|all | +0.4711 / +0.4391 | +0.4551 | [+0.2443, +0.6787] | 0.0442 | WORSE |
| ade_tac_m|all | +0.6537 / +0.6086 | +0.6311 | [+0.3389, +0.9426] | 0.0699 | WORSE |

## I10RQ35 minus I10RLb

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0530 / +0.0465 | +0.0498 | [-0.0170, +0.1208] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2192 / +0.1965 | +0.2079 | [+0.0884, +0.3454] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0086 / +0.0045 | -0.0020 | [-0.0278, +0.0264] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.2028 / +0.2420 | +0.2224 | [+0.0062, +0.4500] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0125 / +0.0087 | +0.0106 | [+0.0033, +0.0178] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.1033 / +0.5672 | +0.8352 | [-0.4543, +2.0438] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0025 / -0.0075 | -0.0025 | [-0.0119, +0.0060] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0125 / +0.0108 | +0.0117 | [+0.0017, +0.0229] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0071 / +0.0008 | +0.0040 | [-0.0029, +0.0117] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.2775 / +0.2519 | +0.2647 | [+0.1024, +0.4532] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1393 / +0.1551 | +0.1472 | [+0.0349, +0.2755] | 0.0229 | WORSE |
| ade6_m|bulk | +0.2688 / +0.2088 | +0.2388 | [+0.0437, +0.4642] | 0.1243 | WORSE |
| ade6_m|all | +0.1983 / +0.1796 | +0.1889 | [+0.0730, +0.3221] | 0.0442 | WORSE |
| ade_tac_m|all | +0.2775 / +0.2519 | +0.2647 | [+0.1024, +0.4532] | 0.0699 | WORSE |

## I10RRaw minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0560 / +0.0445 | +0.0502 | [-0.0403, +0.1450] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3179 / +0.2894 | +0.3036 | [+0.1311, +0.5106] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0352 / -0.0364 | -0.0358 | [-0.0586, -0.0149] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5610 / +0.4114 | +0.4862 | [+0.2777, +0.7415] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0194 / +0.0156 | +0.0175 | [+0.0105, +0.0259] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.8917 / +0.4586 | +1.1752 | [-0.4728, +2.6479] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0046 / +0.0017 | -0.0015 | [-0.0125, +0.0090] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0088 / +0.0142 | +0.0115 | [-0.0010, +0.0244] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0054 / +0.0083 | +0.0069 | [-0.0017, +0.0161] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3755 / +0.3369 | +0.3562 | [+0.1245, +0.6270] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2248 / +0.1356 | +0.1802 | [+0.0358, +0.3645] | 0.0229 | WORSE |
| ade6_m|bulk | +0.3350 / +0.3750 | +0.3550 | [+0.1355, +0.5980] | 0.1243 | WORSE |
| ade6_m|all | +0.2750 / +0.2448 | +0.2599 | [+0.0956, +0.4541] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3755 / +0.3369 | +0.3562 | [+0.1245, +0.6270] | 0.0699 | WORSE |

## I10RRaw minus I10RLb

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0210 / -0.0229 | -0.0219 | [-0.0734, +0.0284] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0116 / +0.0014 | +0.0065 | [-0.0760, +0.0940] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0124 / -0.0134 | -0.0129 | [-0.0335, +0.0060] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.0026 / +0.0314 | +0.0144 | [-0.1838, +0.2256] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0092 / +0.0014 | +0.0053 | [-0.0036, +0.0137] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.1874 / -0.6548 | -0.2337 | [-1.5521, +0.9223] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0046 / -0.0033 | -0.0040 | [-0.0100, +0.0019] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0021 / +0.0050 | +0.0035 | [-0.0052, +0.0131] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0004 / +0.0004 | +0.0000 | [-0.0054, +0.0054] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.0007 / -0.0198 | -0.0102 | [-0.1201, +0.1062] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0841 / +0.0023 | +0.0432 | [-0.0323, +0.1605] | 0.0229 | OPEN |
| ade6_m|bulk | -0.0954 / -0.0350 | -0.0652 | [-0.2145, +0.0579] | 0.1243 | OPEN |
| ade6_m|all | +0.0022 / -0.0147 | -0.0062 | [-0.0852, +0.0772] | 0.0442 | OPEN |
| ade_tac_m|all | -0.0007 / -0.0198 | -0.0102 | [-0.1201, +0.1062] | 0.0699 | OPEN |

## I10RR minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2471 / +0.1902 | +0.2187 | [+0.0825, +0.3629] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.7974 / +0.7078 | +0.7526 | [+0.4611, +1.0507] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0157 / -0.0146 | -0.0152 | [-0.0498, +0.0186] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +1.1586 / +0.8251 | +0.9919 | [+0.6415, +1.3924] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0287 / +0.0328 | +0.0307 | [+0.0206, +0.0431] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.2975 / +2.9565 | +3.6270 | [+1.7790, +5.4347] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0079 / -0.0004 | +0.0038 | [-0.0094, +0.0181] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0254 / +0.0267 | +0.0261 | [+0.0094, +0.0440] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0179 / +0.0071 | +0.0125 | [-0.0002, +0.0271] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +1.0301 / +0.9076 | +0.9689 | [+0.5763, +1.3781] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4828 / +0.4395 | +0.4612 | [+0.2345, +0.7069] | 0.0229 | WORSE |
| ade6_m|bulk | +1.0444 / +0.9133 | +0.9789 | [+0.5388, +1.4366] | 0.1243 | WORSE |
| ade6_m|all | +0.7389 / +0.6556 | +0.6973 | [+0.4171, +0.9909] | 0.0442 | WORSE |
| ade_tac_m|all | +1.0301 / +0.9076 | +0.9689 | [+0.5763, +1.3781] | 0.0699 | WORSE |

## I10RR minus I10RLb

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1701 / +0.1229 | +0.1465 | [+0.0521, +0.2500] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4912 / +0.4199 | +0.4555 | [+0.2638, +0.6826] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0071 / +0.0084 | +0.0077 | [-0.0214, +0.0370] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.6013 / +0.4584 | +0.5298 | [+0.2669, +0.8484] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0191 / +0.0191 | +0.0191 | [+0.0103, +0.0286] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.6319 / +1.6152 | +2.1235 | [+0.5593, +3.6800] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0079 / -0.0054 | +0.0013 | [-0.0098, +0.0144] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0188 / +0.0175 | +0.0181 | [+0.0056, +0.0327] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0121 / -0.0008 | +0.0056 | [-0.0050, +0.0186] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.6539 / +0.5510 | +0.6025 | [+0.3444, +0.9099] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3422 / +0.3062 | +0.3242 | [+0.1548, +0.5195] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6140 / +0.5033 | +0.5586 | [+0.2623, +0.9343] | 0.1243 | WORSE |
| ade6_m|all | +0.4661 / +0.3961 | +0.4311 | [+0.2475, +0.6480] | 0.0442 | WORSE |
| ade_tac_m|all | +0.6539 / +0.5510 | +0.6025 | [+0.3444, +0.9099] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10RLb_s0 | 3.0076 | 3.2533 | 1.6737 | 1.2717 | 0.6528 |
| I10RLb_s1 | 2.9501 | 3.2689 | 1.7153 | 1.1656 | 0.6794 |
| I10RQ35_s0 | 3.2059 | 3.3926 | 1.7267 | 1.2631 | 0.6599 |
| I10RQ35_s1 | 3.1297 | 3.4240 | 1.7618 | 1.1701 | 0.6803 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RR_s0 | 3.4738 | 3.5955 | 1.8438 | 1.2787 | 0.6649 |
| I10RR_s1 | 3.3462 | 3.5751 | 1.8382 | 1.1740 | 0.6786 |
| I10RRaw_s0 | 3.0099 | 3.3375 | 1.6527 | 1.2592 | 0.6524 |
| I10RRaw_s1 | 2.9354 | 3.2712 | 1.6924 | 1.1522 | 0.6799 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
