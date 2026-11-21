# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 28}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RHp3_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.08771976070433585, "D_read": 0.08771976070433583, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10RHp3_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RHp3 R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2410 / -0.3878 | -0.3144 | [-0.4554, -0.1841] | 0.0292 (0.0292) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1248 / -0.3265 | -0.2257 | [-0.4302, -0.0339] | 0.0528 (0.0528) | BETTER |
| LATERAL:cross_mae_m | -0.0266 / -0.0478 | -0.0372 | [-0.0702, -0.0088] | 0.0471 (0.0471) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.3706 / -1.3975 | -1.3840 | [-1.6216, -1.1802] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.1543 / -0.1247 | -0.1395 | [-0.1797, -0.1085] | 0.0348 (0.0348) | BETTER |
| LATERAL:yaw_rate_mae_degps | -8.4820 / -8.2608 | -8.3714 | [-9.9572, -6.9922] | 2.0057 (2.0057) | BETTER |
| TACTICAL:lon_err_rate | -0.0975 / -0.1321 | -0.1148 | [-0.1518, -0.0805] | 0.0292 (0.0292) | BETTER |
| TACTICAL:lat_err_rate | -0.0796 / -0.0625 | -0.0711 | [-0.0913, -0.0536] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.1146 / -0.1580 | -0.1363 | [-0.1778, -0.0977] | 0.0196 (0.0196) | BETTER |
| TACTICAL:ade_tac_m | -0.2593 / -0.5529 | -0.4061 | [-0.6991, -0.1278] | 0.0132 (0.0132) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1618 / -0.0369 | +0.0624 | [-0.1127, +0.2582] | 0.0797 | OPEN |
| ade6_m|bulk | -0.5194 / -0.7179 | -0.6186 | [-0.9345, -0.3706] | 0.0696 | BETTER |
| ade6_m|all | -0.1488 / -0.3475 | -0.2482 | [-0.4469, -0.0586] | 0.0116 | BETTER |
| ade_tac_m|all | -0.2593 / -0.5529 | -0.4061 | [-0.6991, -0.1278] | 0.0132 | BETTER |

## I10RHp3 R0C10XHb20 minus I10RHp3 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1213 / +0.1411 | +0.1312 | [+0.0990, +0.1659] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1773 / +0.1826 | +0.1800 | [+0.1321, +0.2297] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0039 / +0.0015 | -0.0012 | [-0.0100, +0.0083] | 0.0471 (0.0471) | OPEN |
| LATERAL:heading_mae_deg | +0.4016 / +1.0681 | +0.7348 | [+0.3372, +1.2663] | 0.4482 (0.4482) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0428 / +0.0809 | +0.0619 | [+0.0354, +0.0949] | 0.0348 (0.0348) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.1406 / +8.5095 | +7.3251 | [+5.4720, +9.5565] | 2.0057 (2.0057) | WORSE |
| TACTICAL:lon_err_rate | +0.0734 / +0.1226 | +0.0980 | [+0.0571, +0.1411] | 0.0292 (0.0292) | WORSE |
| TACTICAL:lat_err_rate | +0.0067 / +0.0296 | +0.0181 | [-0.0004, +0.0363] | 0.0063 (0.0063) | OPEN |
| TACTICAL:joint_err_rate | +0.0667 / +0.1146 | +0.0907 | [+0.0517, +0.1330] | 0.0196 (0.0196) | WORSE |
| TACTICAL:ade_tac_m | +0.2255 / +0.2492 | +0.2373 | [+0.1709, +0.3073] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0488 / +0.0494 | +0.0491 | [-0.0046, +0.1043] | 0.0797 | OPEN |
| ade6_m|bulk | +0.2715 / +0.3054 | +0.2884 | [+0.2287, +0.3537] | 0.0696 | WORSE |
| ade6_m|all | +0.1503 / +0.1661 | +0.1582 | [+0.1139, +0.2049] | 0.0116 | WORSE |
| ade_tac_m|all | +0.2255 / +0.2492 | +0.2373 | [+0.1709, +0.3073] | 0.0132 | WORSE |

## I10RHp3 R0G50Hb20 minus I10RHp3 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0561 / +0.0551 | +0.0556 | [+0.0351, +0.0756] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.0880 / +0.0701 | +0.0790 | [+0.0463, +0.1092] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0259 / -0.0554 | -0.0407 | [-0.0649, -0.0188] | 0.0471 (0.0471) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1711 / +0.1563 | +0.1637 | [+0.0223, +0.3432] | 0.4482 (0.4482) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0201 / +0.0446 | +0.0324 | [+0.0175, +0.0518] | 0.0348 (0.0348) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.9320 / +4.6388 | +3.7854 | [+2.6167, +5.2440] | 2.0057 (2.0057) | WORSE |
| TACTICAL:lon_err_rate | +0.0383 / +0.0613 | +0.0498 | [+0.0265, +0.0761] | 0.0292 (0.0292) | WORSE |
| TACTICAL:lat_err_rate | -0.0088 / -0.0600 | -0.0344 | [-0.0725, -0.0013] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | +0.0275 / +0.0021 | +0.0148 | [-0.0127, +0.0400] | 0.0196 (0.0196) | OPEN |
| TACTICAL:ade_tac_m | +0.0842 / +0.0376 | +0.0609 | [+0.0056, +0.1113] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0065 / -0.0240 | -0.0087 | [-0.0553, +0.0317] | 0.0797 | OPEN |
| ade6_m|bulk | +0.1153 / +0.0836 | +0.0995 | [+0.0532, +0.1437] | 0.0696 | WORSE |
| ade6_m|all | +0.0561 / +0.0251 | +0.0406 | [+0.0037, +0.0742] | 0.0116 | WORSE |
| ade_tac_m|all | +0.0842 / +0.0376 | +0.0609 | [+0.0056, +0.1113] | 0.0132 | WORSE |

## I10RHp3 R0C10Hb20 minus I10RHp3 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0320 / +0.0683 | +0.0501 | [+0.0219, +0.0781] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.0388 / +0.0662 | +0.0525 | [+0.0204, +0.0795] | 0.0528 (0.0528) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0043 / +0.0004 | +0.0024 | [-0.0104, +0.0149] | 0.0471 (0.0471) | OPEN |
| LATERAL:heading_mae_deg | +0.2106 / +0.5616 | +0.3861 | [+0.1101, +0.6998] | 0.4482 (0.4482) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0209 / +0.0381 | +0.0295 | [+0.0153, +0.0462] | 0.0348 (0.0348) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.4180 / +6.2918 | +5.3549 | [+3.8971, +6.9986] | 2.0057 (2.0057) | WORSE |
| TACTICAL:lon_err_rate | +0.0042 / +0.0463 | +0.0252 | [-0.0058, +0.0598] | 0.0292 (0.0292) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / -0.0167 | -0.0075 | [-0.0267, +0.0073] | 0.0063 (0.0063) | OPEN |
| TACTICAL:joint_err_rate | +0.0117 / +0.0459 | +0.0288 | [+0.0027, +0.0598] | 0.0196 (0.0196) | WORSE |
| TACTICAL:ade_tac_m | +0.0651 / +0.0887 | +0.0769 | [+0.0448, +0.1125] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0473 / +0.0658 | +0.0565 | [+0.0293, +0.0842] | 0.0797 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0388 / +0.0512 | +0.0450 | [+0.0157, +0.0757] | 0.0696 | INSIDE-SPREAD |
| ade6_m|all | +0.0434 / +0.0591 | +0.0513 | [+0.0299, +0.0750] | 0.0116 | WORSE |
| ade_tac_m|all | +0.0651 / +0.0887 | +0.0769 | [+0.0448, +0.1125] | 0.0132 | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4253 / +0.5783 | +0.5018 | [+0.3633, +0.6525] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5148 / +0.6779 | +0.5963 | [+0.4294, +0.7851] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0281 | +0.0151 | [-0.0064, +0.0392] | 0.0471 (0.0471) | OPEN |
| LATERAL:heading_mae_deg | +1.2984 / +2.1302 | +1.7143 | [+1.1487, +2.3855] | 0.4482 (0.4482) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1632 / +0.1844 | +0.1738 | [+0.1396, +0.2133] | 0.0348 (0.0348) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8306 / +15.7625 | +14.2966 | [+11.4789, +17.5884] | 2.0057 (2.0057) | WORSE |
| TACTICAL:lon_err_rate | +0.1734 / +0.2543 | +0.2138 | [+0.1512, +0.2812] | 0.0292 (0.0292) | WORSE |
| TACTICAL:lat_err_rate | +0.0646 / +0.0638 | +0.0642 | [+0.0500, +0.0794] | 0.0063 (0.0063) | WORSE |
| TACTICAL:joint_err_rate | +0.1788 / +0.2580 | +0.2184 | [+0.1561, +0.2826] | 0.0196 (0.0196) | WORSE |
| TACTICAL:ade_tac_m | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1535 / +0.2743 | +0.2139 | [+0.0940, +0.3374] | 0.0797 | WORSE |
| ade6_m|bulk | +0.8602 / +1.1053 | +0.9827 | [+0.7095, +1.3144] | 0.0696 | WORSE |
| ade6_m|all | +0.4758 / +0.6532 | +0.5645 | [+0.3925, +0.7571] | 0.0116 | WORSE |
| ade_tac_m|all | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0132 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1390 / +0.1740 | +0.1565 | [+0.1087, +0.2098] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1801 / +0.2231 | +0.2016 | [+0.1391, +0.2668] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0540 / -0.0648 | -0.0594 | [-0.0833, -0.0397] | 0.0471 (0.0471) | BETTER |
| LATERAL:heading_mae_deg | +0.2941 / +0.8362 | +0.5652 | [+0.2123, +0.9892] | 0.4482 (0.4482) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0682 / +0.0883 | +0.0783 | [+0.0594, +0.1021] | 0.0348 (0.0348) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7579 / +7.9876 | +6.8727 | [+5.1833, +9.1092] | 2.0057 (2.0057) | WORSE |
| TACTICAL:lon_err_rate | +0.0767 / +0.1463 | +0.1115 | [+0.0594, +0.1676] | 0.0292 (0.0292) | WORSE |
| TACTICAL:lat_err_rate | -0.0113 / -0.0129 | -0.0121 | [-0.0231, -0.0017] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | +0.0429 / +0.0713 | +0.0571 | [+0.0304, +0.0869] | 0.0196 (0.0196) | WORSE |
| TACTICAL:ade_tac_m | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0033 / +0.0210 | +0.0122 | [-0.0397, +0.0695] | 0.0797 | OPEN |
| ade6_m|bulk | +0.2261 / +0.2684 | +0.2473 | [+0.1874, +0.3092] | 0.0696 | WORSE |
| ade6_m|all | +0.1049 / +0.1338 | +0.1194 | [+0.0744, +0.1683] | 0.0116 | WORSE |
| ade_tac_m|all | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0132 | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2919 / +0.4426 | +0.3672 | [+0.2558, +0.4843] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3451 / +0.5030 | +0.4241 | [+0.2983, +0.5604] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0080 / -0.0177 | -0.0128 | [-0.0332, +0.0072] | 0.0471 (0.0471) | OPEN |
| LATERAL:heading_mae_deg | +1.4153 / +1.5714 | +1.4933 | [+1.2106, +1.8094] | 0.4482 (0.4482) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1058 / +0.1205 | +0.1131 | [+0.0895, +0.1417] | 0.0348 (0.0348) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.2146 / +14.0424 | +12.6285 | [+10.0108, +15.6291] | 2.0057 (2.0057) | WORSE |
| TACTICAL:lon_err_rate | +0.0942 / +0.1105 | +0.1023 | [+0.0763, +0.1301] | 0.0292 (0.0292) | WORSE |
| TACTICAL:lat_err_rate | +0.0538 / +0.0396 | +0.0467 | [+0.0311, +0.0628] | 0.0063 (0.0063) | WORSE |
| TACTICAL:joint_err_rate | +0.0763 / +0.1159 | +0.0961 | [+0.0650, +0.1312] | 0.0196 (0.0196) | WORSE |
| TACTICAL:ade_tac_m | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1666 / +0.2602 | +0.2134 | [+0.1412, +0.2884] | 0.0797 | WORSE |
| ade6_m|bulk | +0.4298 / +0.6170 | +0.5234 | [+0.3460, +0.7226] | 0.0696 | WORSE |
| ade6_m|all | +0.2866 / +0.4229 | +0.3548 | [+0.2432, +0.4718] | 0.0116 | WORSE |
| ade_tac_m|all | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0132 | WORSE |

## I10RHp3 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1843 / +0.1905 | +0.1874 | [+0.1531, +0.2230] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3899 / +0.3514 | +0.3707 | [+0.2959, +0.4455] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0245 / -0.0197 | -0.0221 | [-0.0523, +0.0057] | 0.0471 (0.0471) | OPEN |
| LATERAL:heading_mae_deg | -0.0794 / +0.6641 | +0.2923 | [-0.1818, +0.8530] | 0.4482 (0.4482) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0127 / +0.0575 | +0.0351 | [+0.0039, +0.0700] | 0.0348 (0.0348) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.4792 / +7.4148 | +5.9470 | [+3.8170, +8.3965] | 2.0057 (2.0057) | WORSE |
| TACTICAL:lon_err_rate | +0.0759 / +0.1221 | +0.0990 | [+0.0575, +0.1419] | 0.0292 (0.0292) | WORSE |
| TACTICAL:lat_err_rate | -0.0150 / +0.0013 | -0.0069 | [-0.0235, +0.0100] | 0.0063 (0.0063) | OPEN |
| TACTICAL:joint_err_rate | +0.0642 / +0.1000 | +0.0821 | [+0.0452, +0.1203] | 0.0196 (0.0196) | WORSE |
| TACTICAL:ade_tac_m | +0.4544 / +0.4270 | +0.4407 | [+0.3353, +0.5406] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3153 / +0.2374 | +0.2764 | [+0.1567, +0.3971] | 0.0797 | WORSE |
| ade6_m|bulk | +0.3408 / +0.3874 | +0.3641 | [+0.2880, +0.4467] | 0.0696 | WORSE |
| ade6_m|all | +0.3269 / +0.3058 | +0.3164 | [+0.2433, +0.3867] | 0.0116 | WORSE |
| ade_tac_m|all | +0.4544 / +0.4270 | +0.4407 | [+0.3353, +0.5406] | 0.0132 | WORSE |

## I10RHp3 R0Hb20 minus I10B0 R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0277 / -0.0791 | -0.0534 | [-0.1190, +0.0061] | 0.0292 (0.0292) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1248 / +0.0018 | +0.0633 | [-0.0547, +0.1893] | 0.0528 (0.0528) | OPEN |
| LATERAL:cross_mae_m | -0.0527 / -0.0315 | -0.0421 | [-0.0758, -0.0071] | 0.0471 (0.0471) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.5844 / -1.8326 | -1.7085 | [-2.0485, -1.4241] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.1048 / -0.1007 | -0.1027 | [-0.1239, -0.0836] | 0.0348 (0.0348) | BETTER |
| LATERAL:yaw_rate_mae_degps | -7.4355 / -7.5106 | -7.4731 | [-8.9316, -6.1867] | 2.0057 (2.0057) | BETTER |
| TACTICAL:lon_err_rate | -0.0567 / -0.0984 | -0.0775 | [-0.1144, -0.0409] | 0.0292 (0.0292) | BETTER |
| TACTICAL:lat_err_rate | -0.0959 / -0.0888 | -0.0923 | [-0.1149, -0.0725] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.0817 / -0.1271 | -0.1044 | [-0.1432, -0.0683] | 0.0196 (0.0196) | BETTER |
| TACTICAL:ade_tac_m | +0.0819 / -0.0485 | +0.0167 | [-0.1240, +0.1761] | 0.0132 (0.0132) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2809 / +0.1462 | +0.2135 | [+0.0608, +0.3859] | 0.0797 | WORSE |
| ade6_m|bulk | -0.1626 / -0.1990 | -0.1808 | [-0.2572, -0.1143] | 0.0696 | BETTER |
| ade6_m|all | +0.0786 / -0.0112 | +0.0337 | [-0.0628, +0.1446] | 0.0116 | OPEN |
| ade_tac_m|all | +0.0819 / -0.0485 | +0.0167 | [-0.1240, +0.1761] | 0.0132 | OPEN |

## I10RHp3 R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2747 / -0.4425 | -0.3586 | [-0.5128, -0.2153] | 0.0292 (0.0292) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2018 / -0.4522 | -0.3270 | [-0.5602, -0.0995] | 0.0528 (0.0528) | BETTER |
| LATERAL:cross_mae_m | -0.0523 / -0.0279 | -0.0401 | [-0.0741, -0.0055] | 0.0471 (0.0471) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -2.1936 / -2.1967 | -2.1951 | [-2.5343, -1.8967] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.1630 / -0.1478 | -0.1554 | [-0.1908, -0.1255] | 0.0348 (0.0348) | BETTER |
| LATERAL:yaw_rate_mae_degps | -11.3298 / -12.0815 | -11.7057 | [-13.9325, -9.6505] | 2.0057 (2.0057) | BETTER |
| TACTICAL:lon_err_rate | -0.1201 / -0.1484 | -0.1342 | [-0.1718, -0.0999] | 0.0292 (0.0292) | BETTER |
| TACTICAL:lat_err_rate | -0.1526 / -0.1546 | -0.1536 | [-0.1840, -0.1270] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.1175 / -0.1730 | -0.1453 | [-0.1925, -0.1026] | 0.0196 (0.0196) | BETTER |
| TACTICAL:ade_tac_m | -0.2971 / -0.6044 | -0.4507 | [-0.7299, -0.1664] | 0.0132 (0.0132) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1493 / -0.0574 | +0.0459 | [-0.1390, +0.2487] | 0.0797 | OPEN |
| ade6_m|bulk | -0.5597 / -0.7687 | -0.6642 | [-0.9333, -0.4418] | 0.0696 | BETTER |
| ade6_m|all | -0.1740 / -0.3818 | -0.2779 | [-0.4669, -0.0846] | 0.0116 | BETTER |
| ade_tac_m|all | -0.2971 / -0.6044 | -0.4507 | [-0.7299, -0.1664] | 0.0132 | BETTER |

## I10RHp3 R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0149 / -0.0681 | -0.0415 | [-0.0999, +0.0154] | 0.0292 (0.0292) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1046 / -0.0153 | +0.0446 | [-0.0694, +0.1673] | 0.0528 (0.0528) | OPEN |
| LATERAL:cross_mae_m | -0.0646 / -0.0459 | -0.0553 | [-0.0874, -0.0221] | 0.0471 (0.0471) | BETTER |
| LATERAL:heading_mae_deg | -0.9862 / -1.1835 | -1.0849 | [-1.3391, -0.8670] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0756 / -0.0604 | -0.0680 | [-0.0870, -0.0509] | 0.0348 (0.0348) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.4285 / -4.2152 | -4.3218 | [-5.2731, -3.4512] | 2.0057 (2.0057) | BETTER |
| TACTICAL:lon_err_rate | -0.0300 / -0.0842 | -0.0571 | [-0.1025, -0.0171] | 0.0292 (0.0292) | BETTER |
| TACTICAL:lat_err_rate | -0.1005 / -0.0984 | -0.0994 | [-0.1239, -0.0784] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.0529 / -0.1030 | -0.0779 | [-0.1159, -0.0398] | 0.0196 (0.0196) | BETTER |
| TACTICAL:ade_tac_m | +0.0677 / -0.0586 | +0.0045 | [-0.1296, +0.1550] | 0.0132 (0.0132) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2686 / +0.1370 | +0.2028 | [+0.0610, +0.3582] | 0.0797 | WORSE |
| ade6_m|bulk | -0.1687 / -0.2029 | -0.1858 | [-0.2588, -0.1196] | 0.0696 | BETTER |
| ade6_m|all | +0.0692 / -0.0180 | +0.0256 | [-0.0668, +0.1293] | 0.0116 | OPEN |
| ade_tac_m|all | +0.0677 / -0.0586 | +0.0045 | [-0.1296, +0.1550] | 0.0132 | OPEN |

## I10RHp3 R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0680 / +0.0508 | +0.0594 | [+0.0359, +0.0859] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1967 / +0.1377 | +0.1672 | [+0.0984, +0.2457] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0927 / -0.0553 | -0.0740 | [-0.1125, -0.0370] | 0.0471 (0.0471) | BETTER |
| LATERAL:heading_mae_deg | -0.8592 / -0.5540 | -0.7066 | [-0.9561, -0.4726] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0273 / -0.0198 | -0.0235 | [-0.0335, -0.0143] | 0.0348 (0.0348) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.5612 / -0.9621 | -1.2617 | [-1.8367, -0.7701] | 2.0057 (2.0057) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0083 / +0.0008 | +0.0046 | [-0.0092, +0.0194] | 0.0292 (0.0292) | OPEN |
| TACTICAL:lat_err_rate | -0.1030 / -0.0513 | -0.0771 | [-0.1155, -0.0413] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.0375 / -0.0338 | -0.0356 | [-0.0519, -0.0210] | 0.0196 (0.0196) | BETTER |
| TACTICAL:ade_tac_m | +0.1409 / +0.1045 | +0.1227 | [+0.0366, +0.2203] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2653 / +0.1821 | +0.2237 | [+0.1136, +0.3487] | 0.0797 | WORSE |
| ade6_m|bulk | -0.0579 / -0.0181 | -0.0380 | [-0.0871, +0.0127] | 0.0696 | OPEN |
| ade6_m|all | +0.1180 / +0.0908 | +0.1044 | [+0.0439, +0.1731] | 0.0116 | WORSE |
| ade_tac_m|all | +0.1409 / +0.1045 | +0.1227 | [+0.0366, +0.2203] | 0.0132 | WORSE |

## I10RHp3 R0KHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0629 / +0.0494 | +0.0561 | [+0.0350, +0.0792] | 0.0292 (0.0292) | WORSE |
| LONGITUDINAL:along_mae_m | +0.2126 / +0.1688 | +0.1907 | [+0.1291, +0.2620] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0207 / -0.0213 | -0.0210 | [-0.0500, +0.0039] | 0.0471 (0.0471) | OPEN |
| LATERAL:heading_mae_deg | -0.4751 / -0.4287 | -0.4519 | [-0.5971, -0.3161] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0304 / -0.0224 | -0.0264 | [-0.0375, -0.0165] | 0.0348 (0.0348) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.6035 / -1.0918 | -1.3477 | [-1.9230, -0.8640] | 2.0057 (2.0057) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0025 / -0.0004 | +0.0010 | [-0.0110, +0.0133] | 0.0292 (0.0292) | OPEN |
| TACTICAL:lat_err_rate | -0.0217 / -0.0283 | -0.0250 | [-0.0346, -0.0154] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.0025 / -0.0146 | -0.0085 | [-0.0213, +0.0052] | 0.0196 (0.0196) | OPEN |
| TACTICAL:ade_tac_m | +0.2289 / +0.1778 | +0.2034 | [+0.1135, +0.3002] | 0.0132 (0.0132) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2665 / +0.1880 | +0.2273 | [+0.1192, +0.3515] | 0.0797 | WORSE |
| ade6_m|bulk | +0.0693 / +0.0820 | +0.0757 | [+0.0409, +0.1148] | 0.0696 | WORSE |
| ade6_m|all | +0.1766 / +0.1397 | +0.1581 | [+0.0954, +0.2270] | 0.0116 | WORSE |
| ade_tac_m|all | +0.2289 / +0.1778 | +0.2034 | [+0.1135, +0.3002] | 0.0132 | WORSE |

## I10RHp3 R0Hb20 minus I10RH R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0094 / +0.0131 | +0.0113 | [-0.0142, +0.0376] | 0.0292 (0.0292) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1014 / +0.0803 | +0.0909 | [+0.0386, +0.1465] | 0.0528 (0.0528) | WORSE |
| LATERAL:cross_mae_m | -0.0359 / -0.0317 | -0.0338 | [-0.0538, -0.0137] | 0.0471 (0.0471) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.6321 / -0.7229 | -0.6775 | [-0.8320, -0.5479] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0424 / -0.0368 | -0.0396 | [-0.0499, -0.0297] | 0.0348 (0.0348) | BETTER |
| LATERAL:yaw_rate_mae_degps | -3.2670 / -3.0084 | -3.1377 | [-3.8368, -2.4740] | 2.0057 (2.0057) | BETTER |
| TACTICAL:lon_err_rate | -0.0063 / -0.0292 | -0.0177 | [-0.0375, +0.0054] | 0.0292 (0.0292) | OPEN |
| TACTICAL:lat_err_rate | -0.0442 / -0.0308 | -0.0375 | [-0.0554, -0.0221] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.0258 / -0.0392 | -0.0325 | [-0.0486, -0.0146] | 0.0196 (0.0196) | BETTER |
| TACTICAL:ade_tac_m | +0.0740 / +0.0685 | +0.0713 | [-0.0007, +0.1459] | 0.0132 (0.0132) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1766 / +0.1701 | +0.1733 | [+0.0974, +0.2614] | 0.0797 | WORSE |
| ade6_m|bulk | -0.0627 / -0.0697 | -0.0662 | [-0.1056, -0.0272] | 0.0696 | INSIDE-SPREAD |
| ade6_m|all | +0.0675 / +0.0607 | +0.0641 | [+0.0150, +0.1160] | 0.0116 | WORSE |
| ade_tac_m|all | +0.0740 / +0.0685 | +0.0713 | [-0.0007, +0.1459] | 0.0132 | OPEN |

## I10RHp3 R0C10XHb20 minus I10RH R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0450 / -0.0551 | -0.0501 | [-0.0757, -0.0229] | 0.0292 (0.0292) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0020 / -0.0005 | -0.0012 | [-0.0528, +0.0551] | 0.0528 (0.0528) | OPEN |
| LATERAL:cross_mae_m | -0.0158 / -0.0151 | -0.0155 | [-0.0336, +0.0020] | 0.0471 (0.0471) | OPEN |
| LATERAL:heading_mae_deg | -0.5881 / -0.5315 | -0.5598 | [-0.6947, -0.4321] | 0.4482 (0.4482) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0640 / -0.0342 | -0.0491 | [-0.0743, -0.0271] | 0.0348 (0.0348) | BETTER |
| LATERAL:yaw_rate_mae_degps | -3.2729 / -2.5800 | -2.9264 | [-3.7905, -2.1848] | 2.0057 (2.0057) | BETTER |
| TACTICAL:lon_err_rate | -0.0308 / -0.0325 | -0.0317 | [-0.0483, -0.0152] | 0.0292 (0.0292) | BETTER |
| TACTICAL:lat_err_rate | -0.0233 / -0.0200 | -0.0217 | [-0.0315, -0.0123] | 0.0063 (0.0063) | BETTER |
| TACTICAL:joint_err_rate | -0.0375 / -0.0438 | -0.0406 | [-0.0584, -0.0240] | 0.0196 (0.0196) | BETTER |
| TACTICAL:ade_tac_m | -0.0537 / -0.0331 | -0.0434 | [-0.1196, +0.0337] | 0.0132 (0.0132) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0896 / +0.0901 | +0.0898 | [+0.0113, +0.1777] | 0.0797 | WORSE |
| ade6_m|bulk | -0.1456 / -0.1229 | -0.1343 | [-0.1884, -0.0848] | 0.0696 | BETTER |
| ade6_m|all | -0.0177 / -0.0070 | -0.0124 | [-0.0652, +0.0421] | 0.0116 | OPEN |
| ade_tac_m|all | -0.0537 / -0.0331 | -0.0434 | [-0.1196, +0.0337] | 0.0132 | OPEN |

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
| I10RHp3_R0C10Hb20_s0 | 2.5867 | 3.2952 | 1.1068 | 0.9704 | 0.5415 |
| I10RHp3_R0C10Hb20_s1 | 2.5164 | 3.1666 | 1.1052 | 1.0007 | 0.5548 |
| I10RHp3_R0C10XHb20_s0 | 2.4976 | 3.2714 | 1.1423 | 0.8162 | 0.5286 |
| I10RHp3_R0C10XHb20_s1 | 2.4084 | 3.1168 | 1.1211 | 0.7895 | 0.5473 |
| I10RHp3_R0G0Hb20_s0 | 2.4871 | 3.2414 | 1.0187 | 0.9920 | 0.5023 |
| I10RHp3_R0G0Hb20_s1 | 2.4322 | 3.1249 | 0.9818 | 1.0557 | 0.5069 |
| I10RHp3_R0G50Hb20_s0 | 2.5433 | 3.2479 | 1.0748 | 0.9661 | 0.5298 |
| I10RHp3_R0G50Hb20_s1 | 2.4573 | 3.1009 | 1.0369 | 1.0003 | 0.5090 |
| I10RHp3_R0Hb20_s0 | 2.5580 | 3.2404 | 1.1263 | 0.9796 | 0.5123 |
| I10RHp3_R0Hb20_s1 | 2.5464 | 3.1607 | 1.1555 | 1.0267 | 0.5319 |
| I10RHp3_R0KHb20_s0 | 2.3472 | 3.2226 | 1.0210 | 0.8201 | 0.4619 |
| I10RHp3_R0KHb20_s1 | 2.2422 | 3.0674 | 0.9800 | 0.7879 | 0.4327 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
