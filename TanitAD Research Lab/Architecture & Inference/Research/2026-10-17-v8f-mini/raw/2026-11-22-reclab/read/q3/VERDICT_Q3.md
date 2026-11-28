# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 28}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RL_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.16894383952236636, "D_read": 0.16894383952236638, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10RL_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RL R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1895 / -0.3185 | -0.2540 | [-0.3809, -0.1398] | 0.0563 (0.0563) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1222 / -0.3075 | -0.2148 | [-0.3902, -0.0564] | 0.0083 (0.0083) | BETTER |
| LATERAL:cross_mae_m | -0.0258 / -0.0540 | -0.0399 | [-0.0666, -0.0165] | 0.0151 (0.0151) | BETTER |
| LATERAL:heading_mae_deg | -0.8619 / -0.9962 | -0.9291 | [-1.1448, -0.7496] | 0.5735 (0.5735) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0870 / -0.0999 | -0.0934 | [-0.1232, -0.0702] | 0.0373 (0.0373) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.6250 / -6.0012 | -5.3131 | [-6.9456, -3.9097] | 2.5751 (2.5751) | BETTER |
| TACTICAL:lon_err_rate | -0.0575 / -0.0913 | -0.0744 | [-0.1081, -0.0450] | 0.0500 (0.0500) | BETTER |
| TACTICAL:lat_err_rate | -0.0542 / -0.0421 | -0.0481 | [-0.0623, -0.0355] | 0.0288 (0.0288) | BETTER |
| TACTICAL:joint_err_rate | -0.0709 / -0.1030 | -0.0869 | [-0.1194, -0.0583] | 0.0342 (0.0342) | BETTER |
| TACTICAL:ade_tac_m | -0.2231 / -0.5129 | -0.3680 | [-0.6286, -0.1291] | 0.0036 (0.0036) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0377 / -0.1460 | -0.0542 | [-0.1825, +0.0784] | 0.0488 | OPEN |
| ade6_m|bulk | -0.3596 / -0.5639 | -0.4617 | [-0.7447, -0.2315] | 0.0535 | BETTER |
| ade6_m|all | -0.1435 / -0.3366 | -0.2400 | [-0.4142, -0.0803] | 0.0022 | BETTER |
| ade_tac_m|all | -0.2231 / -0.5129 | -0.3680 | [-0.6286, -0.1291] | 0.0036 | BETTER |

## I10RL R0C10XHb20 minus I10RL R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2160 / +0.2452 | +0.2306 | [+0.1841, +0.2769] | 0.0563 (0.0563) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3128 / +0.3197 | +0.3163 | [+0.2537, +0.3845] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | +0.0002 / +0.0032 | +0.0017 | [-0.0090, +0.0130] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | +0.6328 / +1.3882 | +1.0105 | [+0.5296, +1.5958] | 0.5735 (0.5735) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0935 / +0.1009 | +0.0972 | [+0.0746, +0.1230] | 0.0373 (0.0373) | WORSE |
| LATERAL:yaw_rate_mae_degps | +9.0326 / +10.4606 | +9.7466 | [+7.9202, +11.8776] | 2.5751 (2.5751) | WORSE |
| TACTICAL:lon_err_rate | +0.1180 / +0.1672 | +0.1426 | [+0.0989, +0.1917] | 0.0500 (0.0500) | WORSE |
| TACTICAL:lat_err_rate | +0.0204 / +0.0379 | +0.0292 | [+0.0131, +0.0461] | 0.0288 (0.0288) | WORSE |
| TACTICAL:joint_err_rate | +0.1105 / +0.1676 | +0.1390 | [+0.0915, +0.1905] | 0.0342 (0.0342) | WORSE |
| TACTICAL:ade_tac_m | +0.4186 / +0.4406 | +0.4296 | [+0.3396, +0.5264] | 0.0036 (0.0036) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1232 / +0.0941 | +0.1086 | [+0.0281, +0.1864] | 0.0488 | WORSE |
| ade6_m|bulk | +0.4651 / +0.5319 | +0.4985 | [+0.4097, +0.5957] | 0.0535 | WORSE |
| ade6_m|all | +0.2791 / +0.2937 | +0.2864 | [+0.2264, +0.3510] | 0.0022 | WORSE |
| ade_tac_m|all | +0.4186 / +0.4406 | +0.4296 | [+0.3396, +0.5264] | 0.0036 | WORSE |

## I10RL R0G50Hb20 minus I10RL R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0996 / +0.0900 | +0.0948 | [+0.0648, +0.1220] | 0.0563 (0.0563) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1397 / +0.1183 | +0.1290 | [+0.0862, +0.1700] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | -0.0278 / -0.0663 | -0.0470 | [-0.0777, -0.0203] | 0.0151 (0.0151) | BETTER |
| LATERAL:heading_mae_deg | +0.1600 / +0.3254 | +0.2427 | [+0.0936, +0.4538] | 0.5735 (0.5735) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0400 / +0.0614 | +0.0507 | [+0.0347, +0.0718] | 0.0373 (0.0373) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.3586 / +6.1217 | +5.2401 | [+3.8915, +6.9313] | 2.5751 (2.5751) | WORSE |
| TACTICAL:lon_err_rate | +0.0667 / +0.1025 | +0.0846 | [+0.0523, +0.1202] | 0.0500 (0.0500) | WORSE |
| TACTICAL:lat_err_rate | -0.0263 / -0.0638 | -0.0450 | [-0.0758, -0.0198] | 0.0288 (0.0288) | BETTER |
| TACTICAL:joint_err_rate | +0.0279 / +0.0225 | +0.0252 | [+0.0054, +0.0446] | 0.0342 (0.0342) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1504 / +0.0785 | +0.1144 | [+0.0412, +0.1826] | 0.0036 (0.0036) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0294 / -0.0321 | -0.0013 | [-0.0667, +0.0572] | 0.0488 | OPEN |
| ade6_m|bulk | +0.1847 / +0.1530 | +0.1689 | [+0.1231, +0.2160] | 0.0535 | WORSE |
| ade6_m|all | +0.1002 / +0.0523 | +0.0763 | [+0.0275, +0.1217] | 0.0022 | WORSE |
| ade_tac_m|all | +0.1504 / +0.0785 | +0.1144 | [+0.0412, +0.1826] | 0.0036 | WORSE |

## I10RL R0C10Hb20 minus I10RL R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0974 / +0.1447 | +0.1210 | [+0.0857, +0.1590] | 0.0563 (0.0563) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1310 / +0.1674 | +0.1492 | [+0.1150, +0.1852] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | +0.0210 / -0.0023 | +0.0094 | [-0.0119, +0.0304] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | +0.5114 / +0.8390 | +0.6752 | [+0.3956, +0.9686] | 0.5735 (0.5735) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0433 / +0.0514 | +0.0473 | [+0.0330, +0.0633] | 0.0373 (0.0373) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.4617 / +8.7373 | +7.5995 | [+5.8253, +9.6646] | 2.5751 (2.5751) | WORSE |
| TACTICAL:lon_err_rate | +0.0375 / +0.0779 | +0.0577 | [+0.0271, +0.0942] | 0.0500 (0.0500) | WORSE |
| TACTICAL:lat_err_rate | +0.0075 / -0.0204 | -0.0065 | [-0.0317, +0.0135] | 0.0288 (0.0288) | OPEN |
| TACTICAL:joint_err_rate | +0.0363 / +0.0738 | +0.0550 | [+0.0258, +0.0891] | 0.0342 (0.0342) | WORSE |
| TACTICAL:ade_tac_m | +0.1969 / +0.2193 | +0.2081 | [+0.1704, +0.2513] | 0.0036 (0.0036) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0948 / +0.1029 | +0.0988 | [+0.0670, +0.1298] | 0.0488 | WORSE |
| ade6_m|bulk | +0.1748 / +0.1979 | +0.1863 | [+0.1527, +0.2264] | 0.0535 | WORSE |
| ade6_m|all | +0.1313 / +0.1462 | +0.1387 | [+0.1136, +0.1675] | 0.0022 | WORSE |
| ade_tac_m|all | +0.1969 / +0.2193 | +0.2081 | [+0.1704, +0.2513] | 0.0036 | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4253 / +0.5783 | +0.5018 | [+0.3633, +0.6525] | 0.0563 (0.0563) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5148 / +0.6779 | +0.5963 | [+0.4294, +0.7851] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0281 | +0.0151 | [-0.0064, +0.0392] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | +1.2984 / +2.1302 | +1.7143 | [+1.1487, +2.3855] | 0.5735 (0.5735) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1632 / +0.1844 | +0.1738 | [+0.1396, +0.2133] | 0.0373 (0.0373) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8306 / +15.7625 | +14.2966 | [+11.4789, +17.5884] | 2.5751 (2.5751) | WORSE |
| TACTICAL:lon_err_rate | +0.1734 / +0.2543 | +0.2138 | [+0.1512, +0.2812] | 0.0500 (0.0500) | WORSE |
| TACTICAL:lat_err_rate | +0.0646 / +0.0638 | +0.0642 | [+0.0500, +0.0794] | 0.0288 (0.0288) | WORSE |
| TACTICAL:joint_err_rate | +0.1788 / +0.2580 | +0.2184 | [+0.1561, +0.2826] | 0.0342 (0.0342) | WORSE |
| TACTICAL:ade_tac_m | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0036 (0.0036) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1535 / +0.2743 | +0.2139 | [+0.0940, +0.3374] | 0.0488 | WORSE |
| ade6_m|bulk | +0.8602 / +1.1053 | +0.9827 | [+0.7095, +1.3144] | 0.0535 | WORSE |
| ade6_m|all | +0.4758 / +0.6532 | +0.5645 | [+0.3925, +0.7571] | 0.0022 | WORSE |
| ade_tac_m|all | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0036 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1390 / +0.1740 | +0.1565 | [+0.1087, +0.2098] | 0.0563 (0.0563) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1801 / +0.2231 | +0.2016 | [+0.1391, +0.2668] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | -0.0540 / -0.0648 | -0.0594 | [-0.0833, -0.0397] | 0.0151 (0.0151) | BETTER |
| LATERAL:heading_mae_deg | +0.2941 / +0.8362 | +0.5652 | [+0.2123, +0.9892] | 0.5735 (0.5735) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0682 / +0.0883 | +0.0783 | [+0.0594, +0.1021] | 0.0373 (0.0373) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7579 / +7.9876 | +6.8727 | [+5.1833, +9.1092] | 2.5751 (2.5751) | WORSE |
| TACTICAL:lon_err_rate | +0.0767 / +0.1463 | +0.1115 | [+0.0594, +0.1676] | 0.0500 (0.0500) | WORSE |
| TACTICAL:lat_err_rate | -0.0113 / -0.0129 | -0.0121 | [-0.0231, -0.0017] | 0.0288 (0.0288) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0429 / +0.0713 | +0.0571 | [+0.0304, +0.0869] | 0.0342 (0.0342) | WORSE |
| TACTICAL:ade_tac_m | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0036 (0.0036) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0033 / +0.0210 | +0.0122 | [-0.0397, +0.0695] | 0.0488 | OPEN |
| ade6_m|bulk | +0.2261 / +0.2684 | +0.2473 | [+0.1874, +0.3092] | 0.0535 | WORSE |
| ade6_m|all | +0.1049 / +0.1338 | +0.1194 | [+0.0744, +0.1683] | 0.0022 | WORSE |
| ade_tac_m|all | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0036 | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2919 / +0.4426 | +0.3672 | [+0.2558, +0.4843] | 0.0563 (0.0563) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3451 / +0.5030 | +0.4241 | [+0.2983, +0.5604] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | -0.0080 / -0.0177 | -0.0128 | [-0.0332, +0.0072] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | +1.4153 / +1.5714 | +1.4933 | [+1.2106, +1.8094] | 0.5735 (0.5735) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1058 / +0.1205 | +0.1131 | [+0.0895, +0.1417] | 0.0373 (0.0373) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.2146 / +14.0424 | +12.6285 | [+10.0108, +15.6291] | 2.5751 (2.5751) | WORSE |
| TACTICAL:lon_err_rate | +0.0942 / +0.1105 | +0.1023 | [+0.0763, +0.1301] | 0.0500 (0.0500) | WORSE |
| TACTICAL:lat_err_rate | +0.0538 / +0.0396 | +0.0467 | [+0.0311, +0.0628] | 0.0288 (0.0288) | WORSE |
| TACTICAL:joint_err_rate | +0.0763 / +0.1159 | +0.0961 | [+0.0650, +0.1312] | 0.0342 (0.0342) | WORSE |
| TACTICAL:ade_tac_m | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0036 (0.0036) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1666 / +0.2602 | +0.2134 | [+0.1412, +0.2884] | 0.0488 | WORSE |
| ade6_m|bulk | +0.4298 / +0.6170 | +0.5234 | [+0.3460, +0.7226] | 0.0535 | WORSE |
| ade6_m|all | +0.2866 / +0.4229 | +0.3548 | [+0.2432, +0.4718] | 0.0022 | WORSE |
| ade_tac_m|all | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0036 | WORSE |

## I10RL R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2358 / +0.2598 | +0.2478 | [+0.2021, +0.2953] | 0.0563 (0.0563) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3926 / +0.3704 | +0.3815 | [+0.3095, +0.4518] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | -0.0237 / -0.0259 | -0.0248 | [-0.0448, -0.0063] | 0.0151 (0.0151) | BETTER |
| LATERAL:heading_mae_deg | +0.4276 / +1.0900 | +0.7588 | [+0.3194, +1.2897] | 0.5735 (0.5735) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0781 / +0.0852 | +0.0816 | [+0.0606, +0.1063] | 0.0373 (0.0373) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.2556 / +9.7259 | +8.9907 | [+7.2078, +11.0691] | 2.5751 (2.5751) | WORSE |
| TACTICAL:lon_err_rate | +0.1159 / +0.1630 | +0.1394 | [+0.0957, +0.1884] | 0.0500 (0.0500) | WORSE |
| TACTICAL:lat_err_rate | +0.0104 / +0.0217 | +0.0160 | [+0.0027, +0.0300] | 0.0288 (0.0288) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.1080 / +0.1551 | +0.1315 | [+0.0873, +0.1797] | 0.0342 (0.0342) | WORSE |
| TACTICAL:ade_tac_m | +0.4906 / +0.4670 | +0.4788 | [+0.3771, +0.5814] | 0.0036 (0.0036) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1912 / +0.1283 | +0.1598 | [+0.0668, +0.2499] | 0.0488 | WORSE |
| ade6_m|bulk | +0.5006 / +0.5414 | +0.5210 | [+0.4291, +0.6207] | 0.0535 | WORSE |
| ade6_m|all | +0.3323 / +0.3167 | +0.3245 | [+0.2566, +0.3926] | 0.0022 | WORSE |
| ade_tac_m|all | +0.4906 / +0.4670 | +0.4788 | [+0.3771, +0.5814] | 0.0036 | WORSE |

## I10RL R0Hb20 minus I10B0 R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0417 / -0.0660 | -0.0538 | [-0.0956, -0.0162] | 0.0563 (0.0563) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0255 / -0.0530 | -0.0137 | [-0.0843, +0.0569] | 0.0083 (0.0083) | OPEN |
| LATERAL:cross_mae_m | -0.0097 / -0.0205 | -0.0151 | [-0.0387, +0.0091] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | -0.9638 / -1.1444 | -1.0541 | [-1.2970, -0.8410] | 0.5735 (0.5735) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0649 / -0.0644 | -0.0647 | [-0.0793, -0.0515] | 0.0373 (0.0373) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.0477 / -3.7388 | -3.8933 | [-4.8136, -3.0527] | 2.5751 (2.5751) | BETTER |
| TACTICAL:lon_err_rate | -0.0413 / -0.0621 | -0.0517 | [-0.0773, -0.0260] | 0.0500 (0.0500) | BETTER |
| TACTICAL:lat_err_rate | -0.0488 / -0.0642 | -0.0565 | [-0.0767, -0.0325] | 0.0288 (0.0288) | BETTER |
| TACTICAL:joint_err_rate | -0.0459 / -0.0767 | -0.0613 | [-0.0894, -0.0356] | 0.0342 (0.0342) | BETTER |
| TACTICAL:ade_tac_m | +0.0180 / -0.1029 | -0.0424 | [-0.1399, +0.0563] | 0.0036 (0.0036) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0856 / -0.0181 | +0.0338 | [-0.0539, +0.1284] | 0.0488 | OPEN |
| ade6_m|bulk | -0.0644 / -0.1170 | -0.0907 | [-0.1501, -0.0373] | 0.0535 | BETTER |
| ade6_m|all | +0.0172 / -0.0632 | -0.0230 | [-0.0884, +0.0437] | 0.0022 | OPEN |
| ade_tac_m|all | +0.0180 / -0.1029 | -0.0424 | [-0.1399, +0.0563] | 0.0036 | OPEN |

## I10RL R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2129 / -0.3661 | -0.2895 | [-0.4241, -0.1619] | 0.0563 (0.0563) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1819 / -0.4078 | -0.2948 | [-0.4843, -0.1097] | 0.0083 (0.0083) | BETTER |
| LATERAL:cross_mae_m | +0.0158 / -0.0167 | -0.0004 | [-0.0336, +0.0363] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | -1.4376 / -1.4952 | -1.4664 | [-1.7437, -1.2309] | 0.5735 (0.5735) | BETTER |
| LATERAL:curvature_mae_1pm | -0.1020 / -0.1108 | -0.1064 | [-0.1349, -0.0825] | 0.0373 (0.0373) | BETTER |
| LATERAL:yaw_rate_mae_degps | -6.7994 / -7.8489 | -7.3242 | [-9.1071, -5.7576] | 2.5751 (2.5751) | BETTER |
| TACTICAL:lon_err_rate | -0.0663 / -0.0809 | -0.0736 | [-0.0971, -0.0494] | 0.0500 (0.0500) | BETTER |
| TACTICAL:lat_err_rate | -0.1063 / -0.1346 | -0.1205 | [-0.1549, -0.0923] | 0.0288 (0.0288) | BETTER |
| TACTICAL:joint_err_rate | -0.0654 / -0.1067 | -0.0861 | [-0.1203, -0.0534] | 0.0342 (0.0342) | BETTER |
| TACTICAL:ade_tac_m | -0.1918 / -0.5320 | -0.3619 | [-0.6297, -0.1200] | 0.0036 (0.0036) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0363 / -0.1691 | -0.0664 | [-0.2092, +0.0794] | 0.0488 | OPEN |
| ade6_m|bulk | -0.3122 / -0.5643 | -0.4383 | [-0.6984, -0.2176] | 0.0535 | BETTER |
| ade6_m|all | -0.1226 / -0.3493 | -0.2360 | [-0.4152, -0.0735] | 0.0022 | BETTER |
| ade_tac_m|all | -0.1918 / -0.5320 | -0.3619 | [-0.6297, -0.1200] | 0.0036 | BETTER |

## I10RL R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0184 / -0.0682 | -0.0433 | [-0.0871, -0.0011] | 0.0563 (0.0563) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0323 / -0.0722 | -0.0199 | [-0.1007, +0.0588] | 0.0083 (0.0083) | OPEN |
| LATERAL:cross_mae_m | -0.0131 / -0.0321 | -0.0226 | [-0.0466, +0.0031] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | -0.5346 / -0.7609 | -0.6478 | [-0.8705, -0.4534] | 0.5735 (0.5735) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0358 / -0.0370 | -0.0364 | [-0.0478, -0.0266] | 0.0373 (0.0373) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.9746 / -2.3875 | -2.1810 | [-2.9267, -1.5916] | 2.5751 (2.5751) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0096 / -0.0484 | -0.0290 | [-0.0600, -0.0006] | 0.0500 (0.0500) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0600 / -0.0746 | -0.0673 | [-0.0890, -0.0490] | 0.0288 (0.0288) | BETTER |
| TACTICAL:joint_err_rate | -0.0254 / -0.0646 | -0.0450 | [-0.0742, -0.0156] | 0.0342 (0.0342) | BETTER |
| TACTICAL:ade_tac_m | +0.0413 / -0.1169 | -0.0378 | [-0.1523, +0.0746] | 0.0036 (0.0036) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1081 / -0.0117 | +0.0482 | [-0.0444, +0.1407] | 0.0488 | OPEN |
| ade6_m|bulk | -0.0572 / -0.1451 | -0.1012 | [-0.1755, -0.0339] | 0.0535 | BETTER |
| ade6_m|all | +0.0327 / -0.0726 | -0.0199 | [-0.0971, +0.0555] | 0.0022 | OPEN |
| ade_tac_m|all | +0.0413 / -0.1169 | -0.0378 | [-0.1523, +0.0746] | 0.0036 | OPEN |

## I10RL R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0211 / +0.0158 | +0.0184 | [+0.0074, +0.0286] | 0.0563 (0.0563) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0727 / +0.0326 | +0.0527 | [+0.0162, +0.0876] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | -0.0394 / -0.0306 | -0.0350 | [-0.0556, -0.0153] | 0.0151 (0.0151) | BETTER |
| LATERAL:heading_mae_deg | -0.4107 / -0.2558 | -0.3333 | [-0.5049, -0.2026] | 0.5735 (0.5735) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0111 / -0.0120 | -0.0116 | [-0.0180, -0.0062] | 0.0373 (0.0373) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.6713 / -0.5872 | -0.6292 | [-0.9288, -0.3882] | 2.5751 (2.5751) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0004 / -0.0046 | -0.0021 | [-0.0121, +0.0083] | 0.0500 (0.0500) | OPEN |
| TACTICAL:lat_err_rate | -0.0450 / -0.0238 | -0.0344 | [-0.0523, -0.0173] | 0.0288 (0.0288) | BETTER |
| TACTICAL:joint_err_rate | -0.0104 / -0.0158 | -0.0131 | [-0.0238, -0.0029] | 0.0342 (0.0342) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0483 / +0.0054 | +0.0268 | [-0.0202, +0.0749] | 0.0036 (0.0036) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0820 / +0.0414 | +0.0617 | [+0.0091, +0.1130] | 0.0488 | WORSE |
| ade6_m|bulk | -0.0157 / -0.0298 | -0.0228 | [-0.0477, +0.0034] | 0.0535 | OPEN |
| ade6_m|all | +0.0374 / +0.0089 | +0.0232 | [-0.0090, +0.0559] | 0.0022 | OPEN |
| ade_tac_m|all | +0.0483 / +0.0054 | +0.0268 | [-0.0202, +0.0749] | 0.0036 | OPEN |

## I10RL R0KHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0198 / +0.0145 | +0.0172 | [+0.0070, +0.0272] | 0.0563 (0.0563) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0798 / +0.0508 | +0.0653 | [+0.0333, +0.0967] | 0.0083 (0.0083) | WORSE |
| LATERAL:cross_mae_m | -0.0239 / -0.0291 | -0.0265 | [-0.0444, -0.0118] | 0.0151 (0.0151) | BETTER |
| LATERAL:heading_mae_deg | -0.2095 / -0.2961 | -0.2528 | [-0.3587, -0.1370] | 0.5735 (0.5735) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0144 / -0.0142 | -0.0143 | [-0.0217, -0.0081] | 0.0373 (0.0373) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.6990 / -0.7184 | -0.7087 | [-1.0466, -0.4295] | 2.5751 (2.5751) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0021 / -0.0042 | -0.0031 | [-0.0119, +0.0063] | 0.0500 (0.0500) | OPEN |
| TACTICAL:lat_err_rate | -0.0100 / -0.0163 | -0.0131 | [-0.0202, -0.0060] | 0.0288 (0.0288) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0025 / -0.0125 | -0.0075 | [-0.0169, +0.0031] | 0.0342 (0.0342) | OPEN |
| TACTICAL:ade_tac_m | +0.0720 / +0.0263 | +0.0492 | [-0.0006, +0.0971] | 0.0036 (0.0036) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0680 / +0.0342 | +0.0511 | [+0.0006, +0.1005] | 0.0488 | WORSE |
| ade6_m|bulk | +0.0355 / +0.0095 | +0.0225 | [-0.0038, +0.0481] | 0.0535 | OPEN |
| ade6_m|all | +0.0532 / +0.0229 | +0.0381 | [+0.0043, +0.0705] | 0.0022 | WORSE |
| ade_tac_m|all | +0.0720 / +0.0263 | +0.0492 | [-0.0006, +0.0971] | 0.0036 | OPEN |

## I10RL R0Hb20 minus I10RH R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0046 / +0.0263 | +0.0109 | [-0.0129, +0.0347] | 0.0563 (0.0563) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0021 / +0.0256 | +0.0139 | [-0.0118, +0.0382] | 0.0083 (0.0083) | OPEN |
| LATERAL:cross_mae_m | +0.0071 / -0.0207 | -0.0068 | [-0.0299, +0.0167] | 0.0151 (0.0151) | OPEN |
| LATERAL:heading_mae_deg | -0.0062 / -0.0323 | -0.0192 | [-0.1049, +0.0596] | 0.5735 (0.5735) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0021 / +0.0011 | -0.0005 | [-0.0059, +0.0052] | 0.0373 (0.0373) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.1081 / +0.7964 | +0.4522 | [-0.0775, +1.0957] | 2.5751 (2.5751) | OPEN |
| TACTICAL:lon_err_rate | +0.0092 / +0.0071 | +0.0081 | [+0.0002, +0.0165] | 0.0500 (0.0500) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0029 / -0.0063 | -0.0017 | [-0.0133, +0.0123] | 0.0288 (0.0288) | OPEN |
| TACTICAL:joint_err_rate | +0.0100 / +0.0113 | +0.0106 | [-0.0002, +0.0217] | 0.0342 (0.0342) | OPEN |
| TACTICAL:ade_tac_m | +0.0101 / +0.0142 | +0.0122 | [-0.0198, +0.0434] | 0.0036 (0.0036) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0186 / +0.0058 | -0.0064 | [-0.0466, +0.0309] | 0.0488 | OPEN |
| ade6_m|bulk | +0.0355 / +0.0124 | +0.0239 | [+0.0010, +0.0471] | 0.0535 | INSIDE-SPREAD |
| ade6_m|all | +0.0061 / +0.0088 | +0.0074 | [-0.0141, +0.0285] | 0.0022 | OPEN |
| ade_tac_m|all | +0.0101 / +0.0142 | +0.0122 | [-0.0198, +0.0434] | 0.0036 | OPEN |

## I10RL R0C10XHb20 minus I10RH R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0065 / +0.0142 | +0.0103 | [-0.0005, +0.0217] | 0.0563 (0.0563) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0006 / +0.0186 | +0.0096 | [-0.0116, +0.0320] | 0.0083 (0.0083) | OPEN |
| LATERAL:cross_mae_m | -0.0150 / -0.0212 | -0.0181 | [-0.0287, -0.0077] | 0.0151 (0.0151) | BETTER |
| LATERAL:heading_mae_deg | -0.0805 / -0.1566 | -0.1186 | [-0.2242, -0.0342] | 0.5735 (0.5735) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0031 / -0.0073 | -0.0021 | [-0.0132, +0.0075] | 0.0373 (0.0373) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.5646 / -0.2621 | +0.1513 | [-0.5460, +0.7826] | 2.5751 (2.5751) | OPEN |
| TACTICAL:lon_err_rate | +0.0092 / +0.0083 | +0.0088 | [-0.0010, +0.0188] | 0.0500 (0.0500) | OPEN |
| TACTICAL:lat_err_rate | +0.0021 / +0.0004 | +0.0013 | [-0.0065, +0.0090] | 0.0288 (0.0288) | OPEN |
| TACTICAL:joint_err_rate | +0.0063 / +0.0113 | +0.0088 | [-0.0015, +0.0196] | 0.0342 (0.0342) | OPEN |
| TACTICAL:ade_tac_m | -0.0175 / +0.0068 | -0.0053 | [-0.0358, +0.0271] | 0.0036 (0.0036) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0345 / -0.0190 | -0.0268 | [-0.0541, -0.0002] | 0.0488 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0141 / +0.0312 | +0.0226 | [-0.0009, +0.0478] | 0.0535 | OPEN |
| ade6_m|all | -0.0123 / +0.0039 | -0.0042 | [-0.0247, +0.0174] | 0.0022 | OPEN |
| ade_tac_m|all | -0.0175 / +0.0068 | -0.0053 | [-0.0358, +0.0271] | 0.0036 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10B0_R0C10Hb20_s0 | 2.7607 | 3.1459 | 1.3815 | 1.0227 | 0.6590 |
| I10B0_R0C10Hb20_s1 | 2.8982 | 3.2241 | 1.5476 | 1.0286 | 0.7278 |
| I10B0_R0C10XHb20_s0 | 2.6464 | 3.1096 | 1.3833 | 0.8429 | 0.6432 |
| I10B0_R0C10XHb20_s1 | 2.7558 | 3.1537 | 1.5089 | 0.8373 | 0.7053 |
| I10B0_R0G0Hb20_s0 | 2.3692 | 2.9760 | 0.9507 | 1.0847 | 0.5398 |
| I10B0_R0G0Hb20_s1 | 2.3414 | 2.9428 | 0.9310 | 1.1110 | 0.5406 |
| I10B0_R0G50Hb20_s0 | 2.4741 | 2.9794 | 1.0897 | 1.0307 | 0.5827 |
| I10B0_R0G50Hb20_s1 | 2.4752 | 2.9638 | 1.1050 | 1.0462 | 0.6119 |
| I10B0_R0Hb20_s0 | 2.4794 | 2.9596 | 1.1540 | 1.0323 | 0.5940 |
| I10B0_R0Hb20_s1 | 2.5576 | 3.0145 | 1.2346 | 1.0582 | 0.6590 |
| I10B0_R0KHb20_s0 | 2.1706 | 2.9561 | 0.9580 | 0.8408 | 0.4644 |
| I10B0_R0KHb20_s1 | 2.1026 | 2.8794 | 0.9306 | 0.8092 | 0.4473 |
| I10RH_R0C10XHb20_s0 | 2.5152 | 3.1818 | 1.1873 | 0.8321 | 0.5661 |
| I10RH_R0C10XHb20_s1 | 2.4154 | 3.0267 | 1.1762 | 0.8046 | 0.5911 |
| I10RH_R0Hb20_s0 | 2.4906 | 3.0639 | 1.1169 | 1.0155 | 0.5381 |
| I10RH_R0Hb20_s1 | 2.4857 | 2.9907 | 1.1424 | 1.0584 | 0.5711 |
| I10RL_R0C10Hb20_s0 | 2.6381 | 3.1822 | 1.1687 | 1.0385 | 0.5936 |
| I10RL_R0C10Hb20_s1 | 2.5489 | 3.0550 | 1.1815 | 1.0118 | 0.6211 |
| I10RL_R0C10XHb20_s0 | 2.5029 | 3.1473 | 1.1938 | 0.8171 | 0.5723 |
| I10RL_R0C10XHb20_s1 | 2.4193 | 3.0077 | 1.1904 | 0.7833 | 0.6023 |
| I10RL_R0G0Hb20_s0 | 2.4066 | 3.0580 | 0.9717 | 1.0453 | 0.5294 |
| I10RL_R0G0Hb20_s1 | 2.3504 | 2.9842 | 0.9468 | 1.0805 | 0.5248 |
| I10RL_R0G50Hb20_s0 | 2.5068 | 3.0874 | 1.0713 | 1.0175 | 0.5573 |
| I10RL_R0G50Hb20_s1 | 2.4027 | 2.9521 | 1.0369 | 1.0141 | 0.5473 |
| I10RL_R0Hb20_s0 | 2.4966 | 3.0452 | 1.1123 | 1.0226 | 0.5481 |
| I10RL_R0Hb20_s1 | 2.4945 | 2.9965 | 1.1686 | 1.0377 | 0.5823 |
| I10RL_R0KHb20_s0 | 2.2238 | 3.0241 | 0.9778 | 0.8169 | 0.4619 |
| I10RL_R0KHb20_s1 | 2.1255 | 2.9136 | 0.9451 | 0.7801 | 0.4348 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
