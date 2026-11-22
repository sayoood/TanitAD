# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 28}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RA_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.1418964335263626, "D_read": 0.1418964335263626, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10RA_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RA R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2035 / -0.3106 | -0.2571 | [-0.3759, -0.1527] | 0.0473 (0.0473) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1374 / -0.3006 | -0.2190 | [-0.3845, -0.0728] | 0.0108 (0.0108) | BETTER |
| LATERAL:cross_mae_m | -0.0254 / -0.0372 | -0.0313 | [-0.0504, -0.0153] | 0.0533 (0.0533) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.7972 / -0.8682 | -0.8327 | [-0.9983, -0.6769] | 0.6686 (0.6686) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0922 / -0.0925 | -0.0923 | [-0.1187, -0.0707] | 0.0336 (0.0336) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.6452 / -5.5940 | -5.1196 | [-6.4949, -3.8736] | 2.5640 (2.5640) | BETTER |
| TACTICAL:lon_err_rate | -0.0696 / -0.0921 | -0.0809 | [-0.1102, -0.0555] | 0.0592 (0.0592) | BETTER |
| TACTICAL:lat_err_rate | -0.0542 / -0.0459 | -0.0500 | [-0.0638, -0.0375] | 0.0208 (0.0208) | BETTER |
| TACTICAL:joint_err_rate | -0.0809 / -0.1067 | -0.0938 | [-0.1251, -0.0659] | 0.0371 (0.0371) | BETTER |
| TACTICAL:ade_tac_m | -0.2486 / -0.4890 | -0.3688 | [-0.6124, -0.1546] | 0.0338 (0.0338) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0171 / -0.1126 | -0.0478 | [-0.1535, +0.0566] | 0.0158 | OPEN |
| ade6_m|bulk | -0.3716 / -0.5676 | -0.4696 | [-0.7503, -0.2464] | 0.0692 | BETTER |
| ade6_m|all | -0.1602 / -0.3201 | -0.2401 | [-0.4030, -0.0975] | 0.0230 | BETTER |
| ade_tac_m|all | -0.2486 / -0.4890 | -0.3688 | [-0.6124, -0.1546] | 0.0338 | BETTER |

## I10RA R0C10XHb20 minus I10RA R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2009 / +0.2550 | +0.2279 | [+0.1744, +0.2829] | 0.0473 (0.0473) | WORSE |
| LONGITUDINAL:along_mae_m | +0.2992 / +0.3233 | +0.3112 | [+0.2458, +0.3818] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | -0.0021 / +0.0066 | +0.0022 | [-0.0103, +0.0149] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | +0.7722 / +1.4801 | +1.1261 | [+0.6729, +1.7110] | 0.6686 (0.6686) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0905 / +0.1085 | +0.0995 | [+0.0744, +0.1284] | 0.0336 (0.0336) | WORSE |
| LATERAL:yaw_rate_mae_degps | +9.1592 / +10.8572 | +10.0082 | [+8.1196, +12.2595] | 2.5640 (2.5640) | WORSE |
| TACTICAL:lon_err_rate | +0.0980 / +0.1596 | +0.1288 | [+0.0817, +0.1841] | 0.0592 (0.0592) | WORSE |
| TACTICAL:lat_err_rate | +0.0275 / +0.0304 | +0.0290 | [+0.0171, +0.0414] | 0.0208 (0.0208) | WORSE |
| TACTICAL:joint_err_rate | +0.0975 / +0.1567 | +0.1271 | [+0.0809, +0.1824] | 0.0371 (0.0371) | WORSE |
| TACTICAL:ade_tac_m | +0.3920 / +0.4491 | +0.4206 | [+0.3229, +0.5255] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1046 / +0.1131 | +0.1088 | [+0.0308, +0.1890] | 0.0158 | WORSE |
| ade6_m|bulk | +0.4484 / +0.5217 | +0.4850 | [+0.3943, +0.5831] | 0.0692 | WORSE |
| ade6_m|all | +0.2614 / +0.2994 | +0.2804 | [+0.2152, +0.3503] | 0.0230 | WORSE |
| ade_tac_m|all | +0.3920 / +0.4491 | +0.4206 | [+0.3229, +0.5255] | 0.0338 | WORSE |

## I10RA R0G50Hb20 minus I10RA R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0922 / +0.0893 | +0.0907 | [+0.0627, +0.1182] | 0.0473 (0.0473) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1321 / +0.1145 | +0.1233 | [+0.0816, +0.1630] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | -0.0334 / -0.0529 | -0.0432 | [-0.0614, -0.0248] | 0.0533 (0.0533) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1118 / +0.3597 | +0.2358 | [+0.0352, +0.4632] | 0.6686 (0.6686) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0466 / +0.0609 | +0.0538 | [+0.0397, +0.0720] | 0.0336 (0.0336) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.5906 / +6.1600 | +5.3753 | [+4.1124, +6.9825] | 2.5640 (2.5640) | WORSE |
| TACTICAL:lon_err_rate | +0.0496 / +0.0988 | +0.0742 | [+0.0371, +0.1146] | 0.0592 (0.0592) | WORSE |
| TACTICAL:lat_err_rate | -0.0342 / -0.0421 | -0.0381 | [-0.0531, -0.0250] | 0.0208 (0.0208) | BETTER |
| TACTICAL:joint_err_rate | +0.0163 / +0.0254 | +0.0208 | [+0.0033, +0.0400] | 0.0371 (0.0371) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1362 / +0.0852 | +0.1107 | [+0.0477, +0.1739] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0212 / -0.0301 | -0.0045 | [-0.0666, +0.0528] | 0.0158 | OPEN |
| ade6_m|bulk | +0.1738 / +0.1606 | +0.1672 | [+0.1269, +0.2123] | 0.0692 | WORSE |
| ade6_m|all | +0.0908 / +0.0568 | +0.0738 | [+0.0318, +0.1160] | 0.0230 | WORSE |
| ade_tac_m|all | +0.1362 / +0.0852 | +0.1107 | [+0.0477, +0.1739] | 0.0338 | WORSE |

## I10RA R0C10Hb20 minus I10RA R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0805 / +0.1572 | +0.1188 | [+0.0713, +0.1718] | 0.0473 (0.0473) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1147 / +0.1723 | +0.1435 | [+0.0989, +0.1905] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | +0.0074 / +0.0254 | +0.0164 | [-0.0008, +0.0348] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | +0.6392 / +1.0006 | +0.8199 | [+0.5377, +1.1576] | 0.6686 (0.6686) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0496 / +0.0528 | +0.0512 | [+0.0377, +0.0665] | 0.0336 (0.0336) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.0238 / +8.4027 | +7.7132 | [+6.2836, +9.3779] | 2.5640 (2.5640) | WORSE |
| TACTICAL:lon_err_rate | +0.0288 / +0.0784 | +0.0536 | [+0.0183, +0.0944] | 0.0592 (0.0592) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0008 / -0.0200 | -0.0104 | [-0.0306, +0.0090] | 0.0208 (0.0208) | OPEN |
| TACTICAL:joint_err_rate | +0.0221 / +0.0704 | +0.0463 | [+0.0113, +0.0863] | 0.0371 (0.0371) | WORSE |
| TACTICAL:ade_tac_m | +0.1586 / +0.2506 | +0.2046 | [+0.1380, +0.2774] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0822 / +0.1293 | +0.1057 | [+0.0646, +0.1509] | 0.0158 | WORSE |
| ade6_m|bulk | +0.1338 / +0.2121 | +0.1730 | [+0.1149, +0.2367] | 0.0692 | WORSE |
| ade6_m|all | +0.1057 / +0.1671 | +0.1364 | [+0.0920, +0.1849] | 0.0230 | WORSE |
| ade_tac_m|all | +0.1586 / +0.2506 | +0.2046 | [+0.1380, +0.2774] | 0.0338 | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4253 / +0.5783 | +0.5018 | [+0.3633, +0.6525] | 0.0473 (0.0473) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5148 / +0.6779 | +0.5963 | [+0.4294, +0.7851] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0281 | +0.0151 | [-0.0064, +0.0392] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | +1.2984 / +2.1302 | +1.7143 | [+1.1487, +2.3855] | 0.6686 (0.6686) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1632 / +0.1844 | +0.1738 | [+0.1396, +0.2133] | 0.0336 (0.0336) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8306 / +15.7625 | +14.2966 | [+11.4789, +17.5884] | 2.5640 (2.5640) | WORSE |
| TACTICAL:lon_err_rate | +0.1734 / +0.2543 | +0.2138 | [+0.1512, +0.2812] | 0.0592 (0.0592) | WORSE |
| TACTICAL:lat_err_rate | +0.0646 / +0.0638 | +0.0642 | [+0.0500, +0.0794] | 0.0208 (0.0208) | WORSE |
| TACTICAL:joint_err_rate | +0.1788 / +0.2580 | +0.2184 | [+0.1561, +0.2826] | 0.0371 (0.0371) | WORSE |
| TACTICAL:ade_tac_m | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1535 / +0.2743 | +0.2139 | [+0.0940, +0.3374] | 0.0158 | WORSE |
| ade6_m|bulk | +0.8602 / +1.1053 | +0.9827 | [+0.7095, +1.3144] | 0.0692 | WORSE |
| ade6_m|all | +0.4758 / +0.6532 | +0.5645 | [+0.3925, +0.7571] | 0.0230 | WORSE |
| ade_tac_m|all | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0338 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1390 / +0.1740 | +0.1565 | [+0.1087, +0.2098] | 0.0473 (0.0473) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1801 / +0.2231 | +0.2016 | [+0.1391, +0.2668] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | -0.0540 / -0.0648 | -0.0594 | [-0.0833, -0.0397] | 0.0533 (0.0533) | BETTER |
| LATERAL:heading_mae_deg | +0.2941 / +0.8362 | +0.5652 | [+0.2123, +0.9892] | 0.6686 (0.6686) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0682 / +0.0883 | +0.0783 | [+0.0594, +0.1021] | 0.0336 (0.0336) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7579 / +7.9876 | +6.8727 | [+5.1833, +9.1092] | 2.5640 (2.5640) | WORSE |
| TACTICAL:lon_err_rate | +0.0767 / +0.1463 | +0.1115 | [+0.0594, +0.1676] | 0.0592 (0.0592) | WORSE |
| TACTICAL:lat_err_rate | -0.0113 / -0.0129 | -0.0121 | [-0.0231, -0.0017] | 0.0208 (0.0208) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0429 / +0.0713 | +0.0571 | [+0.0304, +0.0869] | 0.0371 (0.0371) | WORSE |
| TACTICAL:ade_tac_m | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0033 / +0.0210 | +0.0122 | [-0.0397, +0.0695] | 0.0158 | OPEN |
| ade6_m|bulk | +0.2261 / +0.2684 | +0.2473 | [+0.1874, +0.3092] | 0.0692 | WORSE |
| ade6_m|all | +0.1049 / +0.1338 | +0.1194 | [+0.0744, +0.1683] | 0.0230 | WORSE |
| ade_tac_m|all | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0338 | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2919 / +0.4426 | +0.3672 | [+0.2558, +0.4843] | 0.0473 (0.0473) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3451 / +0.5030 | +0.4241 | [+0.2983, +0.5604] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | -0.0080 / -0.0177 | -0.0128 | [-0.0332, +0.0072] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | +1.4153 / +1.5714 | +1.4933 | [+1.2106, +1.8094] | 0.6686 (0.6686) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1058 / +0.1205 | +0.1131 | [+0.0895, +0.1417] | 0.0336 (0.0336) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.2146 / +14.0424 | +12.6285 | [+10.0108, +15.6291] | 2.5640 (2.5640) | WORSE |
| TACTICAL:lon_err_rate | +0.0942 / +0.1105 | +0.1023 | [+0.0763, +0.1301] | 0.0592 (0.0592) | WORSE |
| TACTICAL:lat_err_rate | +0.0538 / +0.0396 | +0.0467 | [+0.0311, +0.0628] | 0.0208 (0.0208) | WORSE |
| TACTICAL:joint_err_rate | +0.0763 / +0.1159 | +0.0961 | [+0.0650, +0.1312] | 0.0371 (0.0371) | WORSE |
| TACTICAL:ade_tac_m | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1666 / +0.2602 | +0.2134 | [+0.1412, +0.2884] | 0.0158 | WORSE |
| ade6_m|bulk | +0.4298 / +0.6170 | +0.5234 | [+0.3460, +0.7226] | 0.0692 | WORSE |
| ade6_m|all | +0.2866 / +0.4229 | +0.3548 | [+0.2432, +0.4718] | 0.0230 | WORSE |
| ade_tac_m|all | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0338 | WORSE |

## I10RA R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2218 / +0.2676 | +0.2447 | [+0.1927, +0.2976] | 0.0473 (0.0473) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3773 / +0.3773 | +0.3773 | [+0.3099, +0.4493] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | -0.0233 / -0.0091 | -0.0162 | [-0.0373, +0.0040] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | +0.4903 / +1.2120 | +0.8512 | [+0.3810, +1.4146] | 0.6686 (0.6686) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0743 / +0.0901 | +0.0822 | [+0.0604, +0.1079] | 0.0336 (0.0336) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.2946 / +10.0991 | +9.1969 | [+7.3396, +11.4181] | 2.5640 (2.5640) | WORSE |
| TACTICAL:lon_err_rate | +0.1038 / +0.1622 | +0.1330 | [+0.0844, +0.1879] | 0.0592 (0.0592) | WORSE |
| TACTICAL:lat_err_rate | +0.0104 / +0.0179 | +0.0142 | [+0.0010, +0.0273] | 0.0208 (0.0208) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0980 / +0.1513 | +0.1246 | [+0.0791, +0.1753] | 0.0371 (0.0371) | WORSE |
| TACTICAL:ade_tac_m | +0.4651 / +0.4909 | +0.4780 | [+0.3799, +0.5821] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1706 / +0.1617 | +0.1661 | [+0.0814, +0.2486] | 0.0158 | WORSE |
| ade6_m|bulk | +0.4885 / +0.5377 | +0.5131 | [+0.4211, +0.6133] | 0.0692 | WORSE |
| ade6_m|all | +0.3156 / +0.3331 | +0.3244 | [+0.2597, +0.3937] | 0.0230 | WORSE |
| ade_tac_m|all | +0.4651 / +0.4909 | +0.4780 | [+0.3799, +0.5821] | 0.0338 | WORSE |

## I10RA R0Hb20 minus I10B0 R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0350 / -0.0682 | -0.0516 | [-0.0998, -0.0114] | 0.0473 (0.0473) | BETTER |
| LONGITUDINAL:along_mae_m | +0.0303 / -0.0508 | -0.0102 | [-0.0835, +0.0660] | 0.0108 (0.0108) | OPEN |
| LATERAL:cross_mae_m | -0.0122 / +0.0152 | +0.0015 | [-0.0244, +0.0342] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | -0.9121 / -0.9817 | -0.9469 | [-1.1231, -0.7767] | 0.6686 (0.6686) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0598 / -0.0645 | -0.0621 | [-0.0768, -0.0488] | 0.0336 (0.0336) | BETTER |
| LATERAL:yaw_rate_mae_degps | -3.8220 / -3.5927 | -3.7073 | [-4.5763, -2.8982] | 2.5640 (2.5640) | BETTER |
| TACTICAL:lon_err_rate | -0.0517 / -0.0634 | -0.0575 | [-0.0813, -0.0348] | 0.0592 (0.0592) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0488 / -0.0563 | -0.0525 | [-0.0708, -0.0317] | 0.0208 (0.0208) | BETTER |
| TACTICAL:joint_err_rate | -0.0513 / -0.0792 | -0.0652 | [-0.0921, -0.0406] | 0.0371 (0.0371) | BETTER |
| TACTICAL:ade_tac_m | +0.0165 / -0.0669 | -0.0252 | [-0.1092, +0.0629] | 0.0338 (0.0338) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0809 / +0.0102 | +0.0455 | [-0.0332, +0.1295] | 0.0158 | OPEN |
| ade6_m|bulk | -0.0601 / -0.0970 | -0.0786 | [-0.1346, -0.0298] | 0.0692 | BETTER |
| ade6_m|all | +0.0166 / -0.0387 | -0.0111 | [-0.0681, +0.0483] | 0.0230 | OPEN |
| ade_tac_m|all | +0.0165 / -0.0669 | -0.0252 | [-0.1092, +0.0629] | 0.0338 | OPEN |

## I10RA R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2332 / -0.3558 | -0.2945 | [-0.4180, -0.1831] | 0.0473 (0.0473) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2108 / -0.4013 | -0.3061 | [-0.4794, -0.1381] | 0.0108 (0.0108) | BETTER |
| LATERAL:cross_mae_m | +0.0066 / +0.0494 | +0.0280 | [-0.0086, +0.0708] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | -1.3501 / -1.1984 | -1.2743 | [-1.5334, -1.0582] | 0.6686 (0.6686) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0942 / -0.1129 | -0.1036 | [-0.1336, -0.0765] | 0.0336 (0.0336) | BETTER |
| LATERAL:yaw_rate_mae_degps | -6.2663 / -8.1248 | -7.1955 | [-9.3010, -5.3377] | 2.5640 (2.5640) | BETTER |
| TACTICAL:lon_err_rate | -0.0855 / -0.0788 | -0.0821 | [-0.1081, -0.0600] | 0.0592 (0.0592) | BETTER |
| TACTICAL:lat_err_rate | -0.1146 / -0.1042 | -0.1094 | [-0.1324, -0.0877] | 0.0208 (0.0208) | BETTER |
| TACTICAL:joint_err_rate | -0.0834 / -0.1005 | -0.0919 | [-0.1157, -0.0686] | 0.0371 (0.0371) | BETTER |
| TACTICAL:ade_tac_m | -0.2423 / -0.4613 | -0.3518 | [-0.5533, -0.1596] | 0.0338 (0.0338) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0056 / -0.1218 | -0.0581 | [-0.1670, +0.0492] | 0.0158 | OPEN |
| ade6_m|bulk | -0.3487 / -0.5164 | -0.4325 | [-0.6484, -0.2472] | 0.0692 | BETTER |
| ade6_m|all | -0.1560 / -0.3017 | -0.2289 | [-0.3636, -0.0995] | 0.0230 | BETTER |
| ade_tac_m|all | -0.2423 / -0.4613 | -0.3518 | [-0.5533, -0.1596] | 0.0338 | BETTER |

## I10RA R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0218 / -0.0703 | -0.0461 | [-0.0913, -0.0031] | 0.0473 (0.0473) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0196 / -0.0707 | -0.0255 | [-0.1013, +0.0499] | 0.0108 (0.0108) | OPEN |
| LATERAL:cross_mae_m | -0.0088 / +0.0064 | -0.0012 | [-0.0233, +0.0268] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | -0.5676 / -0.6427 | -0.6052 | [-0.7944, -0.4636] | 0.6686 (0.6686) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0337 / -0.0397 | -0.0367 | [-0.0495, -0.0256] | 0.0336 (0.0336) | BETTER |
| LATERAL:yaw_rate_mae_degps | -1.9187 / -2.3588 | -2.1387 | [-2.9004, -1.5035] | 2.5640 (2.5640) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0200 / -0.0467 | -0.0333 | [-0.0587, -0.0096] | 0.0592 (0.0592) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0600 / -0.0446 | -0.0523 | [-0.0711, -0.0365] | 0.0208 (0.0208) | BETTER |
| TACTICAL:joint_err_rate | -0.0292 / -0.0550 | -0.0421 | [-0.0642, -0.0192] | 0.0371 (0.0371) | BETTER |
| TACTICAL:ade_tac_m | +0.0290 / -0.0775 | -0.0243 | [-0.1111, +0.0669] | 0.0338 (0.0338) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0900 / +0.0092 | +0.0496 | [-0.0285, +0.1268] | 0.0158 | OPEN |
| ade6_m|bulk | -0.0527 / -0.1114 | -0.0821 | [-0.1377, -0.0263] | 0.0692 | BETTER |
| ade6_m|all | +0.0249 / -0.0458 | -0.0105 | [-0.0686, +0.0512] | 0.0230 | OPEN |
| ade_tac_m|all | +0.0290 / -0.0775 | -0.0243 | [-0.1111, +0.0669] | 0.0338 | OPEN |

## I10RA R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0250 / +0.0145 | +0.0197 | [+0.0071, +0.0330] | 0.0473 (0.0473) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0677 / +0.0380 | +0.0529 | [+0.0199, +0.0881] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | -0.0294 / -0.0055 | -0.0174 | [-0.0387, +0.0081] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | -0.3948 / -0.1902 | -0.2925 | [-0.4529, -0.1317] | 0.6686 (0.6686) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0122 / -0.0139 | -0.0130 | [-0.0202, -0.0073] | 0.0336 (0.0336) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7280 / -0.5586 | -0.6433 | [-0.9614, -0.3801] | 2.5640 (2.5640) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0071 / +0.0008 | +0.0040 | [-0.0063, +0.0160] | 0.0592 (0.0592) | OPEN |
| TACTICAL:lat_err_rate | -0.0371 / -0.0154 | -0.0263 | [-0.0431, -0.0100] | 0.0208 (0.0208) | BETTER |
| TACTICAL:joint_err_rate | -0.0025 / -0.0092 | -0.0058 | [-0.0156, +0.0052] | 0.0371 (0.0371) | OPEN |
| TACTICAL:ade_tac_m | +0.0503 / +0.0380 | +0.0441 | [+0.0033, +0.0855] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0721 / +0.0603 | +0.0662 | [+0.0187, +0.1155] | 0.0158 | WORSE |
| ade6_m|bulk | -0.0003 / -0.0036 | -0.0020 | [-0.0254, +0.0219] | 0.0692 | OPEN |
| ade6_m|all | +0.0391 / +0.0312 | +0.0351 | [+0.0069, +0.0641] | 0.0230 | WORSE |
| ade_tac_m|all | +0.0503 / +0.0380 | +0.0441 | [+0.0033, +0.0855] | 0.0338 | WORSE |

## I10RA R0KHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0209 / +0.0126 | +0.0168 | [+0.0051, +0.0286] | 0.0473 (0.0473) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0781 / +0.0540 | +0.0661 | [+0.0357, +0.0995] | 0.0108 (0.0108) | WORSE |
| LATERAL:cross_mae_m | -0.0212 / -0.0157 | -0.0184 | [-0.0345, -0.0037] | 0.0533 (0.0533) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2759 / -0.2355 | -0.2557 | [-0.3547, -0.1640] | 0.6686 (0.6686) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0149 / -0.0164 | -0.0156 | [-0.0234, -0.0090] | 0.0336 (0.0336) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.8035 / -0.7128 | -0.7582 | [-1.1111, -0.4738] | 2.5640 (2.5640) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0058 / +0.0025 | +0.0042 | [-0.0054, +0.0154] | 0.0592 (0.0592) | OPEN |
| TACTICAL:lat_err_rate | -0.0171 / -0.0125 | -0.0148 | [-0.0221, -0.0079] | 0.0208 (0.0208) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0004 / -0.0054 | -0.0025 | [-0.0113, +0.0092] | 0.0371 (0.0371) | OPEN |
| TACTICAL:ade_tac_m | +0.0730 / +0.0418 | +0.0574 | [+0.0121, +0.1034] | 0.0338 (0.0338) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0661 / +0.0486 | +0.0573 | [+0.0108, +0.1046] | 0.0158 | WORSE |
| ade6_m|bulk | +0.0402 / +0.0160 | +0.0281 | [+0.0027, +0.0572] | 0.0692 | INSIDE-SPREAD |
| ade6_m|all | +0.0543 / +0.0337 | +0.0440 | [+0.0132, +0.0754] | 0.0230 | WORSE |
| ade_tac_m|all | +0.0730 / +0.0418 | +0.0574 | [+0.0121, +0.1034] | 0.0338 | WORSE |

## I10RA R0Hb20 minus I10RH R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0022 / +0.0240 | +0.0131 | [-0.0070, +0.0347] | 0.0473 (0.0473) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0069 / +0.0278 | +0.0173 | [-0.0051, +0.0376] | 0.0108 (0.0108) | OPEN |
| LATERAL:cross_mae_m | +0.0046 / +0.0150 | +0.0098 | [-0.0057, +0.0247] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | +0.0423 / +0.1257 | +0.0840 | [-0.0081, +0.1820] | 0.6686 (0.6686) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0028 / -0.0023 | +0.0003 | [-0.0059, +0.0066] | 0.0336 (0.0336) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.3460 / +0.8577 | +0.6019 | [+0.1029, +1.1002] | 2.5640 (2.5640) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0013 / +0.0058 | +0.0023 | [-0.0081, +0.0113] | 0.0592 (0.0592) | OPEN |
| TACTICAL:lat_err_rate | +0.0029 / +0.0017 | +0.0023 | [-0.0079, +0.0142] | 0.0208 (0.0208) | OPEN |
| TACTICAL:joint_err_rate | +0.0046 / +0.0088 | +0.0067 | [-0.0042, +0.0187] | 0.0371 (0.0371) | OPEN |
| TACTICAL:ade_tac_m | +0.0086 / +0.0502 | +0.0294 | [-0.0117, +0.0678] | 0.0338 (0.0338) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0234 / +0.0340 | +0.0053 | [-0.0432, +0.0529] | 0.0158 | OPEN |
| ade6_m|bulk | +0.0398 / +0.0323 | +0.0360 | [+0.0174, +0.0563] | 0.0692 | INSIDE-SPREAD |
| ade6_m|all | +0.0054 / +0.0333 | +0.0193 | [-0.0084, +0.0450] | 0.0230 | OPEN |
| ade_tac_m|all | +0.0086 / +0.0502 | +0.0294 | [-0.0117, +0.0678] | 0.0338 | OPEN |

## I10RA R0C10XHb20 minus I10RH R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0075 / +0.0220 | +0.0072 | [-0.0142, +0.0279] | 0.0473 (0.0473) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0146 / +0.0254 | +0.0054 | [-0.0264, +0.0353] | 0.0108 (0.0108) | OPEN |
| LATERAL:cross_mae_m | -0.0146 / -0.0045 | -0.0096 | [-0.0211, +0.0004] | 0.0533 (0.0533) | OPEN |
| LATERAL:heading_mae_deg | -0.0173 / -0.0294 | -0.0233 | [-0.0997, +0.0574] | 0.6686 (0.6686) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0023 / -0.0003 | -0.0013 | [-0.0116, +0.0080] | 0.0336 (0.0336) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.5365 / +0.0826 | +0.3096 | [-0.1727, +0.8915] | 2.5640 (2.5640) | OPEN |
| TACTICAL:lon_err_rate | -0.0029 / +0.0075 | +0.0023 | [-0.0096, +0.0150] | 0.0592 (0.0592) | OPEN |
| TACTICAL:lat_err_rate | +0.0021 / -0.0033 | -0.0006 | [-0.0081, +0.0075] | 0.0208 (0.0208) | OPEN |
| TACTICAL:joint_err_rate | -0.0038 / +0.0075 | +0.0019 | [-0.0104, +0.0152] | 0.0371 (0.0371) | OPEN |
| TACTICAL:ade_tac_m | -0.0430 / +0.0308 | -0.0061 | [-0.0605, +0.0443] | 0.0338 (0.0338) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0551 / +0.0144 | -0.0204 | [-0.0715, +0.0265] | 0.0158 | OPEN |
| ade6_m|bulk | +0.0021 / +0.0274 | +0.0147 | [-0.0109, +0.0387] | 0.0692 | OPEN |
| ade6_m|all | -0.0290 / +0.0203 | -0.0044 | [-0.0407, +0.0293] | 0.0230 | OPEN |
| ade_tac_m|all | -0.0430 / +0.0308 | -0.0061 | [-0.0605, +0.0443] | 0.0338 | OPEN |

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
| I10RA_R0C10Hb20_s0 | 2.6047 | 3.1515 | 1.1484 | 1.0293 | 0.5757 |
| I10RA_R0C10Hb20_s1 | 2.5965 | 3.1023 | 1.1919 | 1.0780 | 0.6273 |
| I10RA_R0C10XHb20_s0 | 2.4862 | 3.1267 | 1.1798 | 0.8174 | 0.5623 |
| I10RA_R0C10XHb20_s1 | 2.4357 | 3.0411 | 1.1982 | 0.8001 | 0.5986 |
| I10RA_R0G0Hb20_s0 | 2.4083 | 3.0481 | 0.9757 | 1.0553 | 0.5373 |
| I10RA_R0G0Hb20_s1 | 2.3726 | 3.0031 | 0.9455 | 1.1055 | 0.5315 |
| I10RA_R0G50Hb20_s0 | 2.4990 | 3.0693 | 1.0679 | 1.0219 | 0.5536 |
| I10RA_R0G50Hb20_s1 | 2.4294 | 2.9730 | 1.0347 | 1.0526 | 0.5569 |
| I10RA_R0Hb20_s0 | 2.4960 | 3.0405 | 1.1190 | 1.0201 | 0.5427 |
| I10RA_R0Hb20_s1 | 2.5189 | 3.0247 | 1.1663 | 1.0734 | 0.5798 |
| I10RA_R0KHb20_s0 | 2.2249 | 3.0222 | 0.9789 | 0.8195 | 0.4648 |
| I10RA_R0KHb20_s1 | 2.1363 | 2.9280 | 0.9432 | 0.7936 | 0.4419 |
| I10RH_R0C10XHb20_s0 | 2.5152 | 3.1818 | 1.1873 | 0.8321 | 0.5661 |
| I10RH_R0C10XHb20_s1 | 2.4154 | 3.0267 | 1.1762 | 0.8046 | 0.5911 |
| I10RH_R0Hb20_s0 | 2.4906 | 3.0639 | 1.1169 | 1.0155 | 0.5381 |
| I10RH_R0Hb20_s1 | 2.4857 | 2.9907 | 1.1424 | 1.0584 | 0.5711 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
