# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 24}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RH_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.07646714794720211, "D_read": 0.0764671479472021, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10RH_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RH R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1960 / -0.3327 | -0.2643 | [-0.3941, -0.1469] | 0.0255 (0.0255) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1228 / -0.3260 | -0.2244 | [-0.4040, -0.0564] | 0.0317 (0.0317) | BETTER |
| LATERAL:cross_mae_m | -0.0108 / -0.0327 | -0.0218 | [-0.0438, -0.0027] | 0.0429 (0.0429) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.7826 / -0.8596 | -0.8211 | [-1.0001, -0.6655] | 0.6101 (0.6101) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0894 / -0.0923 | -0.0909 | [-0.1176, -0.0699] | 0.0353 (0.0353) | BETTER |
| LATERAL:yaw_rate_mae_degps | -5.1755 / -5.7295 | -5.4525 | [-6.7520, -4.3357] | 1.9535 (1.9535) | BETTER |
| TACTICAL:lon_err_rate | -0.0667 / -0.0996 | -0.0832 | [-0.1173, -0.0535] | 0.0521 (0.0521) | BETTER |
| TACTICAL:lat_err_rate | -0.0563 / -0.0425 | -0.0494 | [-0.0650, -0.0358] | 0.0196 (0.0196) | BETTER |
| TACTICAL:joint_err_rate | -0.0771 / -0.1142 | -0.0957 | [-0.1303, -0.0629] | 0.0329 (0.0329) | BETTER |
| TACTICAL:ade_tac_m | -0.2056 / -0.5197 | -0.3627 | [-0.6368, -0.1083] | 0.0077 (0.0077) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0722 / -0.1270 | -0.0274 | [-0.1681, +0.1167] | 0.0732 | OPEN |
| ade6_m|bulk | -0.3737 / -0.5950 | -0.4844 | [-0.7702, -0.2472] | 0.0766 | BETTER |
| ade6_m|all | -0.1312 / -0.3404 | -0.2358 | [-0.4187, -0.0653] | 0.0049 | BETTER |
| ade_tac_m|all | -0.2056 / -0.5197 | -0.3627 | [-0.6368, -0.1083] | 0.0077 | BETTER |

## I10RH R0C10XHb20 minus I10RH R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2100 / +0.2354 | +0.2227 | [+0.1778, +0.2689] | 0.0255 (0.0255) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3082 / +0.3005 | +0.3044 | [+0.2402, +0.3703] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | +0.0006 / +0.0064 | +0.0035 | [-0.0076, +0.0146] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | +0.6883 / +1.4563 | +1.0723 | [+0.5810, +1.6885] | 0.6101 (0.6101) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0922 / +0.1055 | +0.0988 | [+0.0750, +0.1263] | 0.0353 (0.0353) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.6000 / +10.6197 | +9.6098 | [+7.6318, +11.9796] | 1.9535 (1.9535) | WORSE |
| TACTICAL:lon_err_rate | +0.1088 / +0.1601 | +0.1344 | [+0.0906, +0.1825] | 0.0521 (0.0521) | WORSE |
| TACTICAL:lat_err_rate | +0.0179 / +0.0367 | +0.0273 | [+0.0098, +0.0454] | 0.0196 (0.0196) | WORSE |
| TACTICAL:joint_err_rate | +0.1021 / +0.1555 | +0.1288 | [+0.0830, +0.1787] | 0.0329 (0.0329) | WORSE |
| TACTICAL:ade_tac_m | +0.4135 / +0.4168 | +0.4152 | [+0.3258, +0.5105] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1189 / +0.0873 | +0.1031 | [+0.0242, +0.1824] | 0.0732 | WORSE |
| ade6_m|bulk | +0.4628 / +0.5052 | +0.4840 | [+0.4037, +0.5746] | 0.0766 | WORSE |
| ade6_m|all | +0.2757 / +0.2778 | +0.2768 | [+0.2172, +0.3403] | 0.0049 | WORSE |
| ade_tac_m|all | +0.4135 / +0.4168 | +0.4152 | [+0.3258, +0.5105] | 0.0077 | WORSE |

## I10RH R0G50Hb20 minus I10RH R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0978 / +0.0833 | +0.0905 | [+0.0603, +0.1187] | 0.0255 (0.0255) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1357 / +0.1086 | +0.1221 | [+0.0785, +0.1636] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | -0.0353 / -0.0628 | -0.0490 | [-0.0743, -0.0275] | 0.0429 (0.0429) | BETTER |
| LATERAL:heading_mae_deg | +0.1342 / +0.3869 | +0.2606 | [+0.0825, +0.5025] | 0.6101 (0.6101) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0475 / +0.0539 | +0.0507 | [+0.0383, +0.0664] | 0.0353 (0.0353) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.4955 / +5.3736 | +4.9345 | [+3.9403, +6.1887] | 1.9535 (1.9535) | WORSE |
| TACTICAL:lon_err_rate | +0.0679 / +0.1005 | +0.0842 | [+0.0538, +0.1172] | 0.0521 (0.0521) | WORSE |
| TACTICAL:lat_err_rate | -0.0296 / -0.0600 | -0.0448 | [-0.0734, -0.0229] | 0.0196 (0.0196) | BETTER |
| TACTICAL:joint_err_rate | +0.0300 / +0.0288 | +0.0294 | [+0.0104, +0.0488] | 0.0329 (0.0329) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1410 / +0.0676 | +0.1043 | [+0.0306, +0.1730] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0234 / -0.0345 | -0.0056 | [-0.0704, +0.0516] | 0.0732 | OPEN |
| ade6_m|bulk | +0.1782 / +0.1400 | +0.1591 | [+0.1112, +0.2091] | 0.0766 | WORSE |
| ade6_m|all | +0.0940 / +0.0451 | +0.0695 | [+0.0204, +0.1153] | 0.0049 | WORSE |
| ade_tac_m|all | +0.1410 / +0.0676 | +0.1043 | [+0.0306, +0.1730] | 0.0077 | WORSE |

## I10RH R0C10Hb20 minus I10RH R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0933 / +0.1413 | +0.1173 | [+0.0832, +0.1551] | 0.0255 (0.0255) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1252 / +0.1603 | +0.1427 | [+0.1097, +0.1783] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | +0.0154 / +0.0005 | +0.0080 | [-0.0089, +0.0252] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | +0.4595 / +0.9748 | +0.7172 | [+0.3464, +1.1180] | 0.6101 (0.6101) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0414 / +0.0577 | +0.0496 | [+0.0327, +0.0691] | 0.0353 (0.0353) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.0643 / +8.5765 | +7.3204 | [+5.4615, +9.5424] | 1.9535 (1.9535) | WORSE |
| TACTICAL:lon_err_rate | +0.0325 / +0.0754 | +0.0540 | [+0.0225, +0.0917] | 0.0521 (0.0521) | WORSE |
| TACTICAL:lat_err_rate | -0.0029 / -0.0171 | -0.0100 | [-0.0265, +0.0031] | 0.0196 (0.0196) | OPEN |
| TACTICAL:joint_err_rate | +0.0325 / +0.0629 | +0.0477 | [+0.0223, +0.0784] | 0.0329 (0.0329) | WORSE |
| TACTICAL:ade_tac_m | +0.1871 / +0.2084 | +0.1977 | [+0.1610, +0.2422] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0883 / +0.1034 | +0.0958 | [+0.0637, +0.1293] | 0.0732 | WORSE |
| ade6_m|bulk | +0.1682 / +0.1814 | +0.1748 | [+0.1419, +0.2120] | 0.0766 | WORSE |
| ade6_m|all | +0.1247 / +0.1389 | +0.1318 | [+0.1074, +0.1615] | 0.0049 | WORSE |
| ade_tac_m|all | +0.1871 / +0.2084 | +0.1977 | [+0.1610, +0.2422] | 0.0077 | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4253 / +0.5783 | +0.5018 | [+0.3633, +0.6525] | 0.0255 (0.0255) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5148 / +0.6779 | +0.5963 | [+0.4294, +0.7851] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0281 | +0.0151 | [-0.0064, +0.0392] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | +1.2984 / +2.1302 | +1.7143 | [+1.1487, +2.3855] | 0.6101 (0.6101) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1632 / +0.1844 | +0.1738 | [+0.1396, +0.2133] | 0.0353 (0.0353) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8306 / +15.7625 | +14.2966 | [+11.4789, +17.5884] | 1.9535 (1.9535) | WORSE |
| TACTICAL:lon_err_rate | +0.1734 / +0.2543 | +0.2138 | [+0.1512, +0.2812] | 0.0521 (0.0521) | WORSE |
| TACTICAL:lat_err_rate | +0.0646 / +0.0638 | +0.0642 | [+0.0500, +0.0794] | 0.0196 (0.0196) | WORSE |
| TACTICAL:joint_err_rate | +0.1788 / +0.2580 | +0.2184 | [+0.1561, +0.2826] | 0.0329 (0.0329) | WORSE |
| TACTICAL:ade_tac_m | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1535 / +0.2743 | +0.2139 | [+0.0940, +0.3374] | 0.0732 | WORSE |
| ade6_m|bulk | +0.8602 / +1.1053 | +0.9827 | [+0.7095, +1.3144] | 0.0766 | WORSE |
| ade6_m|all | +0.4758 / +0.6532 | +0.5645 | [+0.3925, +0.7571] | 0.0049 | WORSE |
| ade_tac_m|all | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0077 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1390 / +0.1740 | +0.1565 | [+0.1087, +0.2098] | 0.0255 (0.0255) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1801 / +0.2231 | +0.2016 | [+0.1391, +0.2668] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | -0.0540 / -0.0648 | -0.0594 | [-0.0833, -0.0397] | 0.0429 (0.0429) | BETTER |
| LATERAL:heading_mae_deg | +0.2941 / +0.8362 | +0.5652 | [+0.2123, +0.9892] | 0.6101 (0.6101) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0682 / +0.0883 | +0.0783 | [+0.0594, +0.1021] | 0.0353 (0.0353) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7579 / +7.9876 | +6.8727 | [+5.1833, +9.1092] | 1.9535 (1.9535) | WORSE |
| TACTICAL:lon_err_rate | +0.0767 / +0.1463 | +0.1115 | [+0.0594, +0.1676] | 0.0521 (0.0521) | WORSE |
| TACTICAL:lat_err_rate | -0.0113 / -0.0129 | -0.0121 | [-0.0231, -0.0017] | 0.0196 (0.0196) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0429 / +0.0713 | +0.0571 | [+0.0304, +0.0869] | 0.0329 (0.0329) | WORSE |
| TACTICAL:ade_tac_m | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0033 / +0.0210 | +0.0122 | [-0.0397, +0.0695] | 0.0732 | OPEN |
| ade6_m|bulk | +0.2261 / +0.2684 | +0.2473 | [+0.1874, +0.3092] | 0.0766 | WORSE |
| ade6_m|all | +0.1049 / +0.1338 | +0.1194 | [+0.0744, +0.1683] | 0.0049 | WORSE |
| ade_tac_m|all | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0077 | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2919 / +0.4426 | +0.3672 | [+0.2558, +0.4843] | 0.0255 (0.0255) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3451 / +0.5030 | +0.4241 | [+0.2983, +0.5604] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | -0.0080 / -0.0177 | -0.0128 | [-0.0332, +0.0072] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | +1.4153 / +1.5714 | +1.4933 | [+1.2106, +1.8094] | 0.6101 (0.6101) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1058 / +0.1205 | +0.1131 | [+0.0895, +0.1417] | 0.0353 (0.0353) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.2146 / +14.0424 | +12.6285 | [+10.0108, +15.6291] | 1.9535 (1.9535) | WORSE |
| TACTICAL:lon_err_rate | +0.0942 / +0.1105 | +0.1023 | [+0.0763, +0.1301] | 0.0521 (0.0521) | WORSE |
| TACTICAL:lat_err_rate | +0.0538 / +0.0396 | +0.0467 | [+0.0311, +0.0628] | 0.0196 (0.0196) | WORSE |
| TACTICAL:joint_err_rate | +0.0763 / +0.1159 | +0.0961 | [+0.0650, +0.1312] | 0.0329 (0.0329) | WORSE |
| TACTICAL:ade_tac_m | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1666 / +0.2602 | +0.2134 | [+0.1412, +0.2884] | 0.0732 | WORSE |
| ade6_m|bulk | +0.4298 / +0.6170 | +0.5234 | [+0.3460, +0.7226] | 0.0766 | WORSE |
| ade6_m|all | +0.2866 / +0.4229 | +0.3548 | [+0.2432, +0.4718] | 0.0049 | WORSE |
| ade_tac_m|all | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0077 | WORSE |

## I10RH R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2293 / +0.2456 | +0.2375 | [+0.1940, +0.2814] | 0.0255 (0.0255) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3919 / +0.3519 | +0.3719 | [+0.2974, +0.4457] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | -0.0087 / -0.0046 | -0.0067 | [-0.0244, +0.0103] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | +0.5026 / +1.2231 | +0.8629 | [+0.3897, +1.4152] | 0.6101 (0.6101) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0762 / +0.0911 | +0.0836 | [+0.0610, +0.1106] | 0.0353 (0.0353) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.7357 / +9.9860 | +8.8608 | [+6.8464, +11.2812] | 1.9535 (1.9535) | WORSE |
| TACTICAL:lon_err_rate | +0.1067 / +0.1546 | +0.1307 | [+0.0863, +0.1797] | 0.0521 (0.0521) | WORSE |
| TACTICAL:lat_err_rate | +0.0083 / +0.0213 | +0.0148 | [+0.0002, +0.0300] | 0.0196 (0.0196) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.1017 / +0.1438 | +0.1228 | [+0.0817, +0.1674] | 0.0329 (0.0329) | WORSE |
| TACTICAL:ade_tac_m | +0.5081 / +0.4602 | +0.4841 | [+0.3812, +0.5882] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2257 / +0.1473 | +0.1865 | [+0.0888, +0.2829] | 0.0732 | WORSE |
| ade6_m|bulk | +0.4864 / +0.5103 | +0.4984 | [+0.4146, +0.5888] | 0.0766 | WORSE |
| ade6_m|all | +0.3446 / +0.3128 | +0.3287 | [+0.2602, +0.3975] | 0.0049 | WORSE |
| ade_tac_m|all | +0.5081 / +0.4602 | +0.4841 | [+0.3812, +0.5882] | 0.0077 | WORSE |

## I10RH R0Hb20 minus I10B0 R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0371 / -0.0922 | -0.0647 | [-0.1197, -0.0146] | 0.0255 (0.0255) | BETTER |
| LONGITUDINAL:along_mae_m | +0.0234 / -0.0785 | -0.0276 | [-0.1134, +0.0579] | 0.0317 (0.0317) | OPEN |
| LATERAL:cross_mae_m | -0.0168 / +0.0002 | -0.0083 | [-0.0298, +0.0157] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | -0.9598 / -1.1231 | -1.0414 | [-1.2658, -0.8387] | 0.6101 (0.6101) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0625 / -0.0658 | -0.0642 | [-0.0798, -0.0508] | 0.0353 (0.0353) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.1469 / -4.5522 | -4.3496 | [-5.3758, -3.4276] | 1.9535 (1.9535) | BETTER |
| TACTICAL:lon_err_rate | -0.0504 / -0.0692 | -0.0598 | [-0.0852, -0.0337] | 0.0521 (0.0521) | BETTER |
| TACTICAL:lat_err_rate | -0.0517 / -0.0579 | -0.0548 | [-0.0729, -0.0369] | 0.0196 (0.0196) | BETTER |
| TACTICAL:joint_err_rate | -0.0559 / -0.0880 | -0.0719 | [-0.1011, -0.0440] | 0.0329 (0.0329) | BETTER |
| TACTICAL:ade_tac_m | +0.0079 / -0.1171 | -0.0546 | [-0.1576, +0.0520] | 0.0077 (0.0077) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1043 / -0.0239 | +0.0402 | [-0.0613, +0.1493] | 0.0732 | OPEN |
| ade6_m|bulk | -0.0999 / -0.1294 | -0.1146 | [-0.1682, -0.0692] | 0.0766 | BETTER |
| ade6_m|all | +0.0112 / -0.0720 | -0.0304 | [-0.0995, +0.0414] | 0.0049 | OPEN |
| ade_tac_m|all | +0.0079 / -0.1171 | -0.0546 | [-0.1576, +0.0520] | 0.0077 | OPEN |

## I10RH R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2131 / -0.3794 | -0.2962 | [-0.4410, -0.1626] | 0.0255 (0.0255) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1925 / -0.4310 | -0.3117 | [-0.5101, -0.1162] | 0.0317 (0.0317) | BETTER |
| LATERAL:cross_mae_m | -0.0035 / +0.0131 | +0.0048 | [-0.0195, +0.0302] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | -1.5029 / -1.3272 | -1.4151 | [-1.7060, -1.1748] | 0.6101 (0.6101) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0992 / -0.1114 | -0.1053 | [-0.1351, -0.0802] | 0.0353 (0.0353) | BETTER |
| LATERAL:yaw_rate_mae_degps | -7.2794 / -8.7190 | -7.9992 | [-10.0120, -6.1740] | 1.9535 (1.9535) | BETTER |
| TACTICAL:lon_err_rate | -0.0659 / -0.0900 | -0.0779 | [-0.1057, -0.0496] | 0.0521 (0.0521) | BETTER |
| TACTICAL:lat_err_rate | -0.1192 / -0.1267 | -0.1230 | [-0.1509, -0.0992] | 0.0196 (0.0196) | BETTER |
| TACTICAL:joint_err_rate | -0.0671 / -0.1117 | -0.0894 | [-0.1246, -0.0554] | 0.0329 (0.0329) | BETTER |
| TACTICAL:ade_tac_m | -0.2182 / -0.5350 | -0.3766 | [-0.6311, -0.1313] | 0.0077 (0.0077) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0482 / -0.1567 | -0.0542 | [-0.2000, +0.0928] | 0.0732 | OPEN |
| ade6_m|bulk | -0.3635 / -0.5819 | -0.4727 | [-0.7253, -0.2607] | 0.0766 | BETTER |
| ade6_m|all | -0.1395 / -0.3506 | -0.2451 | [-0.4148, -0.0812] | 0.0049 | BETTER |
| ade_tac_m|all | -0.2182 / -0.5350 | -0.3766 | [-0.6311, -0.1313] | 0.0077 | BETTER |

## I10RH R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0146 / -0.0780 | -0.0463 | [-0.0992, +0.0034] | 0.0255 (0.0255) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0275 / -0.0883 | -0.0304 | [-0.1199, +0.0578] | 0.0317 (0.0317) | OPEN |
| LATERAL:cross_mae_m | -0.0269 / -0.0051 | -0.0160 | [-0.0393, +0.0099] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | -0.5590 / -0.7338 | -0.6464 | [-0.8590, -0.4779] | 0.6101 (0.6101) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0315 / -0.0436 | -0.0376 | [-0.0539, -0.0246] | 0.0353 (0.0353) | BETTER |
| LATERAL:yaw_rate_mae_degps | -2.0101 / -3.1223 | -2.5662 | [-3.6859, -1.6575] | 1.9535 (1.9535) | BETTER |
| TACTICAL:lon_err_rate | -0.0042 / -0.0550 | -0.0296 | [-0.0688, +0.0060] | 0.0521 (0.0521) | OPEN |
| TACTICAL:lat_err_rate | -0.0625 / -0.0700 | -0.0663 | [-0.0867, -0.0496] | 0.0196 (0.0196) | BETTER |
| TACTICAL:joint_err_rate | -0.0233 / -0.0588 | -0.0411 | [-0.0686, -0.0127] | 0.0329 (0.0329) | BETTER |
| TACTICAL:ade_tac_m | +0.0247 / -0.1090 | -0.0421 | [-0.1446, +0.0642] | 0.0077 (0.0077) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1265 / +0.0002 | +0.0633 | [-0.0329, +0.1622] | 0.0732 | OPEN |
| ade6_m|bulk | -0.1019 / -0.1462 | -0.1241 | [-0.1815, -0.0717] | 0.0766 | BETTER |
| ade6_m|all | +0.0223 / -0.0666 | -0.0221 | [-0.0904, +0.0491] | 0.0049 | OPEN |
| ade_tac_m|all | +0.0247 / -0.1090 | -0.0421 | [-0.1446, +0.0642] | 0.0077 | OPEN |

## I10RH R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0266 / +0.0128 | +0.0197 | [+0.0061, +0.0339] | 0.0255 (0.0255) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0719 / +0.0263 | +0.0491 | [+0.0106, +0.0909] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | -0.0457 / -0.0071 | -0.0264 | [-0.0576, +0.0046] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | -0.4031 / -0.2849 | -0.3440 | [-0.4947, -0.2122] | 0.6101 (0.6101) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0114 / -0.0127 | -0.0120 | [-0.0192, -0.0061] | 0.0353 (0.0353) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7242 / -0.5894 | -0.6568 | [-0.9725, -0.3968] | 1.9535 (1.9535) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0046 / -0.0092 | -0.0023 | [-0.0152, +0.0108] | 0.0521 (0.0521) | OPEN |
| TACTICAL:lat_err_rate | -0.0442 / -0.0229 | -0.0336 | [-0.0511, -0.0169] | 0.0196 (0.0196) | BETTER |
| TACTICAL:joint_err_rate | -0.0104 / -0.0163 | -0.0133 | [-0.0236, -0.0031] | 0.0329 (0.0329) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0411 / +0.0242 | +0.0326 | [-0.0088, +0.0771] | 0.0077 (0.0077) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1064 / +0.0558 | +0.0811 | [+0.0274, +0.1422] | 0.0732 | WORSE |
| ade6_m|bulk | -0.0539 / -0.0179 | -0.0359 | [-0.0695, -0.0029] | 0.0766 | INSIDE-SPREAD |
| ade6_m|all | +0.0333 / +0.0222 | +0.0277 | [-0.0009, +0.0584] | 0.0049 | OPEN |
| ade_tac_m|all | +0.0411 / +0.0242 | +0.0326 | [-0.0088, +0.0771] | 0.0077 | OPEN |

## I10RH R0KHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0193 / +0.0102 | +0.0148 | [+0.0031, +0.0266] | 0.0255 (0.0255) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0837 / +0.0513 | +0.0675 | [+0.0348, +0.1039] | 0.0317 (0.0317) | WORSE |
| LATERAL:cross_mae_m | -0.0093 / -0.0110 | -0.0102 | [-0.0256, +0.0033] | 0.0429 (0.0429) | OPEN |
| LATERAL:heading_mae_deg | -0.1807 / -0.2322 | -0.2065 | [-0.3257, -0.0977] | 0.6101 (0.6101) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0148 / -0.0137 | -0.0142 | [-0.0219, -0.0080] | 0.0353 (0.0353) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.8203 / -0.6258 | -0.7230 | [-1.0918, -0.4422] | 1.9535 (1.9535) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0021 / -0.0054 | -0.0038 | [-0.0133, +0.0071] | 0.0521 (0.0521) | OPEN |
| TACTICAL:lat_err_rate | -0.0096 / -0.0154 | -0.0125 | [-0.0206, -0.0050] | 0.0196 (0.0196) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0004 / -0.0117 | -0.0060 | [-0.0160, +0.0050] | 0.0329 (0.0329) | OPEN |
| TACTICAL:ade_tac_m | +0.0945 / +0.0434 | +0.0690 | [+0.0193, +0.1212] | 0.0077 (0.0077) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1068 / +0.0600 | +0.0834 | [+0.0309, +0.1427] | 0.0732 | WORSE |
| ade6_m|bulk | +0.0237 / +0.0051 | +0.0144 | [-0.0071, +0.0361] | 0.0766 | OPEN |
| ade6_m|all | +0.0689 / +0.0350 | +0.0519 | [+0.0183, +0.0876] | 0.0049 | WORSE |
| ade_tac_m|all | +0.0945 / +0.0434 | +0.0690 | [+0.0193, +0.1212] | 0.0077 | WORSE |

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
| I10RH_R0C10Hb20_s0 | 2.6212 | 3.1941 | 1.1684 | 1.0192 | 0.5919 |
| I10RH_R0C10Hb20_s1 | 2.5476 | 3.0674 | 1.1683 | 1.0416 | 0.6161 |
| I10RH_R0C10XHb20_s0 | 2.5152 | 3.1818 | 1.1873 | 0.8321 | 0.5661 |
| I10RH_R0C10XHb20_s1 | 2.4154 | 3.0267 | 1.1762 | 0.8046 | 0.5911 |
| I10RH_R0G0Hb20_s0 | 2.4025 | 3.0824 | 0.9773 | 1.0390 | 0.5294 |
| I10RH_R0G0Hb20_s1 | 2.3636 | 2.9986 | 0.9438 | 1.1040 | 0.5244 |
| I10RH_R0G50Hb20_s0 | 2.4965 | 3.1058 | 1.0751 | 1.0037 | 0.5594 |
| I10RH_R0G50Hb20_s1 | 2.4086 | 2.9641 | 1.0270 | 1.0412 | 0.5531 |
| I10RH_R0Hb20_s0 | 2.4906 | 3.0639 | 1.1169 | 1.0155 | 0.5381 |
| I10RH_R0Hb20_s1 | 2.4857 | 2.9907 | 1.1424 | 1.0584 | 0.5711 |
| I10RH_R0KHb20_s0 | 2.2395 | 3.0629 | 0.9774 | 0.8314 | 0.4639 |
| I10RH_R0KHb20_s1 | 2.1375 | 2.9394 | 0.9408 | 0.7982 | 0.4356 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
