# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 30}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RK_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.11405300737052437, "D_read": 0.11405300737052436, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10RK_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RK R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1879 / -0.3081 | -0.2480 | [-0.3740, -0.1415] | 0.0380 (0.0380) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1244 / -0.2939 | -0.2092 | [-0.3789, -0.0625] | 0.0248 (0.0248) | BETTER |
| LATERAL:cross_mae_m | -0.0323 / -0.0479 | -0.0401 | [-0.0616, -0.0227] | 0.0270 (0.0270) | BETTER |
| LATERAL:heading_mae_deg | -0.8612 / -0.7458 | -0.8035 | [-0.9893, -0.6286] | 0.5784 (0.5784) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0840 / -0.0504 | -0.0672 | [-0.0992, -0.0415] | 0.0454 (0.0454) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.7190 / -3.4875 | -4.1033 | [-5.4307, -3.0305] | 2.8181 (2.8181) | BETTER |
| TACTICAL:lon_err_rate | -0.0617 / -0.0971 | -0.0794 | [-0.1145, -0.0483] | 0.0584 (0.0584) | BETTER |
| TACTICAL:lat_err_rate | -0.0650 / -0.0354 | -0.0502 | [-0.0740, -0.0296] | 0.0225 (0.0225) | BETTER |
| TACTICAL:joint_err_rate | -0.0800 / -0.1080 | -0.0940 | [-0.1261, -0.0652] | 0.0375 (0.0375) | BETTER |
| TACTICAL:ade_tac_m | -0.2284 / -0.4850 | -0.3567 | [-0.6079, -0.1375] | 0.0050 (0.0050) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0159 / -0.1248 | -0.0545 | [-0.1633, +0.0554] | 0.0366 | OPEN |
| ade6_m|bulk | -0.3442 / -0.5528 | -0.4485 | [-0.7366, -0.2223] | 0.0354 | BETTER |
| ade6_m|all | -0.1484 / -0.3200 | -0.2342 | [-0.4020, -0.0873] | 0.0037 | BETTER |
| ade_tac_m|all | -0.2284 / -0.4850 | -0.3567 | [-0.6079, -0.1375] | 0.0050 | BETTER |

## I10RK R0C10XHb20 minus I10RK R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2169 / +0.2616 | +0.2392 | [+0.1886, +0.2916] | 0.0380 (0.0380) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3189 / +0.3470 | +0.3330 | [+0.2644, +0.4057] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | -0.0017 / +0.0069 | +0.0026 | [-0.0098, +0.0151] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | +0.7556 / +1.5944 | +1.1750 | [+0.6593, +1.8248] | 0.5784 (0.5784) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0955 / +0.1412 | +0.1183 | [+0.0812, +0.1658] | 0.0454 (0.0454) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.9758 / +12.5942 | +10.7850 | [+7.9873, +14.1336] | 2.8181 (2.8181) | WORSE |
| TACTICAL:lon_err_rate | +0.1163 / +0.1596 | +0.1380 | [+0.0977, +0.1832] | 0.0584 (0.0584) | WORSE |
| TACTICAL:lat_err_rate | +0.0158 / +0.0413 | +0.0286 | [+0.0085, +0.0499] | 0.0225 (0.0225) | WORSE |
| TACTICAL:joint_err_rate | +0.1059 / +0.1580 | +0.1319 | [+0.0883, +0.1817] | 0.0375 (0.0375) | WORSE |
| TACTICAL:ade_tac_m | +0.4233 / +0.4792 | +0.4512 | [+0.3525, +0.5576] | 0.0050 (0.0050) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1094 / +0.1302 | +0.1198 | [+0.0395, +0.2020] | 0.0366 | WORSE |
| ade6_m|bulk | +0.4883 / +0.5453 | +0.5168 | [+0.4282, +0.6126] | 0.0354 | WORSE |
| ade6_m|all | +0.2822 / +0.3195 | +0.3008 | [+0.2350, +0.3718] | 0.0037 | WORSE |
| ade_tac_m|all | +0.4233 / +0.4792 | +0.4512 | [+0.3525, +0.5576] | 0.0050 | WORSE |

## I10RK R0G50Hb20 minus I10RK R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0973 / +0.0894 | +0.0934 | [+0.0638, +0.1214] | 0.0380 (0.0380) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1443 / +0.1245 | +0.1344 | [+0.0919, +0.1763] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | -0.0315 / -0.0576 | -0.0445 | [-0.0668, -0.0238] | 0.0270 (0.0270) | BETTER |
| LATERAL:heading_mae_deg | +0.1986 / +0.3696 | +0.2841 | [+0.1160, +0.4875] | 0.5784 (0.5784) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0415 / +0.0613 | +0.0514 | [+0.0360, +0.0712] | 0.0454 (0.0454) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.4230 / +5.9958 | +5.2094 | [+3.9425, +6.7838] | 2.8181 (2.8181) | WORSE |
| TACTICAL:lon_err_rate | +0.0659 / +0.1000 | +0.0830 | [+0.0504, +0.1182] | 0.0584 (0.0584) | WORSE |
| TACTICAL:lat_err_rate | -0.0296 / -0.0421 | -0.0358 | [-0.0540, -0.0213] | 0.0225 (0.0225) | BETTER |
| TACTICAL:joint_err_rate | +0.0296 / +0.0383 | +0.0340 | [+0.0148, +0.0548] | 0.0375 (0.0375) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1496 / +0.0972 | +0.1234 | [+0.0584, +0.1870] | 0.0050 (0.0050) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0207 / -0.0181 | +0.0013 | [-0.0564, +0.0550] | 0.0366 | OPEN |
| ade6_m|bulk | +0.1940 / +0.1638 | +0.1789 | [+0.1332, +0.2294] | 0.0354 | WORSE |
| ade6_m|all | +0.0997 / +0.0648 | +0.0823 | [+0.0389, +0.1246] | 0.0037 | WORSE |
| ade_tac_m|all | +0.1496 / +0.0972 | +0.1234 | [+0.0584, +0.1870] | 0.0050 | WORSE |

## I10RK R0C10Hb20 minus I10RK R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1005 / +0.1645 | +0.1325 | [+0.0898, +0.1791] | 0.0380 (0.0380) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1301 / +0.2040 | +0.1671 | [+0.1159, +0.2236] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | +0.0191 / +0.0051 | +0.0121 | [-0.0052, +0.0284] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | +0.5974 / +1.1726 | +0.8850 | [+0.4914, +1.3461] | 0.5784 (0.5784) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0516 / +0.0741 | +0.0628 | [+0.0427, +0.0872] | 0.0454 (0.0454) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.1515 / +9.5908 | +8.3711 | [+6.4533, +10.6689] | 2.8181 (2.8181) | WORSE |
| TACTICAL:lon_err_rate | +0.0246 / +0.0775 | +0.0511 | [+0.0142, +0.0921] | 0.0584 (0.0584) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0042 / -0.0250 | -0.0146 | [-0.0358, +0.0027] | 0.0225 (0.0225) | OPEN |
| TACTICAL:joint_err_rate | +0.0192 / +0.0646 | +0.0419 | [+0.0102, +0.0792] | 0.0375 (0.0375) | WORSE |
| TACTICAL:ade_tac_m | +0.1941 / +0.2685 | +0.2313 | [+0.1716, +0.2976] | 0.0050 (0.0050) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0970 / +0.1331 | +0.1151 | [+0.0720, +0.1590] | 0.0366 | WORSE |
| ade6_m|bulk | +0.1680 / +0.2338 | +0.2009 | [+0.1510, +0.2609] | 0.0354 | WORSE |
| ade6_m|all | +0.1294 / +0.1790 | +0.1542 | [+0.1144, +0.1984] | 0.0037 | WORSE |
| ade_tac_m|all | +0.1941 / +0.2685 | +0.2313 | [+0.1716, +0.2976] | 0.0050 | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4253 / +0.5783 | +0.5018 | [+0.3633, +0.6525] | 0.0380 (0.0380) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5148 / +0.6779 | +0.5963 | [+0.4294, +0.7851] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0281 | +0.0151 | [-0.0064, +0.0392] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | +1.2984 / +2.1302 | +1.7143 | [+1.1487, +2.3855] | 0.5784 (0.5784) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1632 / +0.1844 | +0.1738 | [+0.1396, +0.2133] | 0.0454 (0.0454) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8306 / +15.7625 | +14.2966 | [+11.4789, +17.5884] | 2.8181 (2.8181) | WORSE |
| TACTICAL:lon_err_rate | +0.1734 / +0.2543 | +0.2138 | [+0.1512, +0.2812] | 0.0584 (0.0584) | WORSE |
| TACTICAL:lat_err_rate | +0.0646 / +0.0638 | +0.0642 | [+0.0500, +0.0794] | 0.0225 (0.0225) | WORSE |
| TACTICAL:joint_err_rate | +0.1788 / +0.2580 | +0.2184 | [+0.1561, +0.2826] | 0.0375 (0.0375) | WORSE |
| TACTICAL:ade_tac_m | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0050 (0.0050) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1535 / +0.2743 | +0.2139 | [+0.0940, +0.3374] | 0.0366 | WORSE |
| ade6_m|bulk | +0.8602 / +1.1053 | +0.9827 | [+0.7095, +1.3144] | 0.0354 | WORSE |
| ade6_m|all | +0.4758 / +0.6532 | +0.5645 | [+0.3925, +0.7571] | 0.0037 | WORSE |
| ade_tac_m|all | +0.7137 / +0.9799 | +0.8468 | [+0.5888, +1.1357] | 0.0050 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1390 / +0.1740 | +0.1565 | [+0.1087, +0.2098] | 0.0380 (0.0380) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1801 / +0.2231 | +0.2016 | [+0.1391, +0.2668] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | -0.0540 / -0.0648 | -0.0594 | [-0.0833, -0.0397] | 0.0270 (0.0270) | BETTER |
| LATERAL:heading_mae_deg | +0.2941 / +0.8362 | +0.5652 | [+0.2123, +0.9892] | 0.5784 (0.5784) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0682 / +0.0883 | +0.0783 | [+0.0594, +0.1021] | 0.0454 (0.0454) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.7579 / +7.9876 | +6.8727 | [+5.1833, +9.1092] | 2.8181 (2.8181) | WORSE |
| TACTICAL:lon_err_rate | +0.0767 / +0.1463 | +0.1115 | [+0.0594, +0.1676] | 0.0584 (0.0584) | WORSE |
| TACTICAL:lat_err_rate | -0.0113 / -0.0129 | -0.0121 | [-0.0231, -0.0017] | 0.0225 (0.0225) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0429 / +0.0713 | +0.0571 | [+0.0304, +0.0869] | 0.0375 (0.0375) | WORSE |
| TACTICAL:ade_tac_m | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0050 (0.0050) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0033 / +0.0210 | +0.0122 | [-0.0397, +0.0695] | 0.0366 | OPEN |
| ade6_m|bulk | +0.2261 / +0.2684 | +0.2473 | [+0.1874, +0.3092] | 0.0354 | WORSE |
| ade6_m|all | +0.1049 / +0.1338 | +0.1194 | [+0.0744, +0.1683] | 0.0037 | WORSE |
| ade_tac_m|all | +0.1574 / +0.2007 | +0.1791 | [+0.1116, +0.2525] | 0.0050 | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2919 / +0.4426 | +0.3672 | [+0.2558, +0.4843] | 0.0380 (0.0380) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3451 / +0.5030 | +0.4241 | [+0.2983, +0.5604] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | -0.0080 / -0.0177 | -0.0128 | [-0.0332, +0.0072] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | +1.4153 / +1.5714 | +1.4933 | [+1.2106, +1.8094] | 0.5784 (0.5784) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1058 / +0.1205 | +0.1131 | [+0.0895, +0.1417] | 0.0454 (0.0454) | WORSE |
| LATERAL:yaw_rate_mae_degps | +11.2146 / +14.0424 | +12.6285 | [+10.0108, +15.6291] | 2.8181 (2.8181) | WORSE |
| TACTICAL:lon_err_rate | +0.0942 / +0.1105 | +0.1023 | [+0.0763, +0.1301] | 0.0584 (0.0584) | WORSE |
| TACTICAL:lat_err_rate | +0.0538 / +0.0396 | +0.0467 | [+0.0311, +0.0628] | 0.0225 (0.0225) | WORSE |
| TACTICAL:joint_err_rate | +0.0763 / +0.1159 | +0.0961 | [+0.0650, +0.1312] | 0.0375 (0.0375) | WORSE |
| TACTICAL:ade_tac_m | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0050 (0.0050) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1666 / +0.2602 | +0.2134 | [+0.1412, +0.2884] | 0.0366 | WORSE |
| ade6_m|bulk | +0.4298 / +0.6170 | +0.5234 | [+0.3460, +0.7226] | 0.0354 | WORSE |
| ade6_m|all | +0.2866 / +0.4229 | +0.3548 | [+0.2432, +0.4718] | 0.0037 | WORSE |
| ade_tac_m|all | +0.4299 / +0.6344 | +0.5322 | [+0.3648, +0.7077] | 0.0050 | WORSE |

## I10RK R0C10XHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2374 / +0.2702 | +0.2538 | [+0.2045, +0.3035] | 0.0380 (0.0380) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3903 / +0.3840 | +0.3872 | [+0.3181, +0.4592] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | -0.0302 / -0.0198 | -0.0250 | [-0.0462, -0.0030] | 0.0270 (0.0270) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4281 / +1.3116 | +0.8698 | [+0.3250, +1.5281] | 0.5784 (0.5784) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0814 / +0.1312 | +0.1063 | [+0.0682, +0.1542] | 0.0454 (0.0454) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.1890 / +12.1764 | +10.1827 | [+7.2801, +13.6696] | 2.8181 (2.8181) | WORSE |
| TACTICAL:lon_err_rate | +0.1117 / +0.1571 | +0.1344 | [+0.0916, +0.1827] | 0.0584 (0.0584) | WORSE |
| TACTICAL:lat_err_rate | -0.0004 / +0.0283 | +0.0140 | [-0.0079, +0.0367] | 0.0225 (0.0225) | OPEN |
| TACTICAL:joint_err_rate | +0.0988 / +0.1501 | +0.1244 | [+0.0794, +0.1732] | 0.0375 (0.0375) | WORSE |
| TACTICAL:ade_tac_m | +0.4853 / +0.4949 | +0.4901 | [+0.3926, +0.5968] | 0.0050 (0.0050) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1694 / +0.1495 | +0.1594 | [+0.0736, +0.2428] | 0.0366 | WORSE |
| ade6_m|bulk | +0.5159 / +0.5525 | +0.5342 | [+0.4426, +0.6315] | 0.0354 | WORSE |
| ade6_m|all | +0.3274 / +0.3333 | +0.3303 | [+0.2652, +0.4016] | 0.0037 | WORSE |
| ade_tac_m|all | +0.4853 / +0.4949 | +0.4901 | [+0.3926, +0.5968] | 0.0050 | WORSE |

## I10RK R0Hb20 minus I10B0 R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0366 / -0.0792 | -0.0579 | [-0.1066, -0.0152] | 0.0380 (0.0380) | BETTER |
| LONGITUDINAL:along_mae_m | +0.0306 / -0.0644 | -0.0169 | [-0.0945, +0.0612] | 0.0248 (0.0248) | OPEN |
| LATERAL:cross_mae_m | -0.0207 / -0.0195 | -0.0201 | [-0.0403, +0.0009] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | -0.9321 / -1.1096 | -1.0208 | [-1.2475, -0.8197] | 0.5784 (0.5784) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0635 / -0.0619 | -0.0627 | [-0.0769, -0.0501] | 0.0454 (0.0454) | BETTER |
| LATERAL:yaw_rate_mae_degps | -3.9935 / -3.6547 | -3.8241 | [-4.7216, -3.0027] | 2.8181 (2.8181) | BETTER |
| TACTICAL:lon_err_rate | -0.0575 / -0.0700 | -0.0638 | [-0.0882, -0.0402] | 0.0584 (0.0584) | BETTER |
| TACTICAL:lat_err_rate | -0.0542 / -0.0634 | -0.0588 | [-0.0777, -0.0387] | 0.0225 (0.0225) | BETTER |
| TACTICAL:joint_err_rate | -0.0663 / -0.0938 | -0.0800 | [-0.1088, -0.0550] | 0.0375 (0.0375) | BETTER |
| TACTICAL:ade_tac_m | +0.0148 / -0.1074 | -0.0463 | [-0.1444, +0.0569] | 0.0050 (0.0050) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0787 / -0.0127 | +0.0330 | [-0.0507, +0.1210] | 0.0366 | OPEN |
| ade6_m|bulk | -0.0638 / -0.1345 | -0.0991 | [-0.1649, -0.0404] | 0.0354 | BETTER |
| ade6_m|all | +0.0137 / -0.0683 | -0.0273 | [-0.0934, +0.0426] | 0.0037 | OPEN |
| ade_tac_m|all | +0.0148 / -0.1074 | -0.0463 | [-0.1444, +0.0569] | 0.0050 | OPEN |

## I10RK R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2091 / -0.3500 | -0.2796 | [-0.4113, -0.1593] | 0.0380 (0.0380) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1822 / -0.3783 | -0.2802 | [-0.4564, -0.1138] | 0.0248 (0.0248) | BETTER |
| LATERAL:cross_mae_m | +0.0124 / -0.0034 | +0.0045 | [-0.0224, +0.0304] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | -1.3256 / -1.2367 | -1.2811 | [-1.5348, -1.0715] | 0.5784 (0.5784) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0919 / -0.0748 | -0.0834 | [-0.1075, -0.0630] | 0.0454 (0.0454) | BETTER |
| LATERAL:yaw_rate_mae_degps | -6.1161 / -6.6026 | -6.3593 | [-7.7851, -5.0475] | 2.8181 (2.8181) | BETTER |
| TACTICAL:lon_err_rate | -0.0796 / -0.0900 | -0.0848 | [-0.1103, -0.0582] | 0.0584 (0.0584) | BETTER |
| TACTICAL:lat_err_rate | -0.1142 / -0.1330 | -0.1236 | [-0.1553, -0.0978] | 0.0225 (0.0225) | BETTER |
| TACTICAL:joint_err_rate | -0.0784 / -0.1130 | -0.0957 | [-0.1284, -0.0658] | 0.0375 (0.0375) | BETTER |
| TACTICAL:ade_tac_m | -0.1924 / -0.4746 | -0.3335 | [-0.5732, -0.1166] | 0.0050 (0.0050) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0315 / -0.1389 | -0.0537 | [-0.1829, +0.0728] | 0.0366 | OPEN |
| ade6_m|bulk | -0.3102 / -0.5208 | -0.4155 | [-0.6537, -0.2138] | 0.0354 | BETTER |
| ade6_m|all | -0.1244 / -0.3130 | -0.2187 | [-0.3789, -0.0733] | 0.0037 | BETTER |
| ade_tac_m|all | -0.1924 / -0.4746 | -0.3335 | [-0.5732, -0.1166] | 0.0050 | BETTER |

## I10RK R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0178 / -0.0719 | -0.0448 | [-0.0923, -0.0003] | 0.0380 (0.0380) | BETTER |
| LONGITUDINAL:along_mae_m | +0.0328 / -0.0793 | -0.0232 | [-0.1093, +0.0613] | 0.0248 (0.0248) | OPEN |
| LATERAL:cross_mae_m | -0.0147 / -0.0262 | -0.0204 | [-0.0423, +0.0016] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | -0.4797 / -0.8518 | -0.6658 | [-0.9782, -0.4089] | 0.5784 (0.5784) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0349 / -0.0343 | -0.0346 | [-0.0450, -0.0251] | 0.0454 (0.0454) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.9569 / -2.3671 | -2.1620 | [-2.8582, -1.5752] | 2.8181 (2.8181) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0100 / -0.0571 | -0.0336 | [-0.0717, -0.0002] | 0.0584 (0.0584) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0563 / -0.0684 | -0.0623 | [-0.0832, -0.0446] | 0.0225 (0.0225) | BETTER |
| TACTICAL:joint_err_rate | -0.0213 / -0.0617 | -0.0415 | [-0.0715, -0.0106] | 0.0375 (0.0375) | BETTER |
| TACTICAL:ade_tac_m | +0.0434 / -0.1087 | -0.0326 | [-0.1444, +0.0793] | 0.0050 (0.0050) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1010 / -0.0117 | +0.0446 | [-0.0453, +0.1347] | 0.0366 | OPEN |
| ade6_m|bulk | -0.0484 / -0.1375 | -0.0929 | [-0.1680, -0.0258] | 0.0354 | BETTER |
| ade6_m|all | +0.0329 / -0.0691 | -0.0181 | [-0.0936, +0.0571] | 0.0037 | OPEN |
| ade_tac_m|all | +0.0434 / -0.1087 | -0.0326 | [-0.1444, +0.0793] | 0.0050 | OPEN |

## I10RK R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0239 / +0.0127 | +0.0183 | [+0.0064, +0.0309] | 0.0380 (0.0380) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0686 / +0.0194 | +0.0440 | [+0.0027, +0.0858] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | -0.0371 / -0.0334 | -0.0353 | [-0.0527, -0.0178] | 0.0270 (0.0270) | BETTER |
| LATERAL:heading_mae_deg | -0.3937 / -0.3896 | -0.3917 | [-0.5125, -0.2847] | 0.5784 (0.5784) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0100 / -0.0068 | -0.0084 | [-0.0143, -0.0035] | 0.0454 (0.0454) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.6662 / -0.3494 | -0.5078 | [-0.8543, -0.2481] | 2.8181 (2.8181) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0008 / -0.0108 | -0.0050 | [-0.0169, +0.0075] | 0.0584 (0.0584) | OPEN |
| TACTICAL:lat_err_rate | -0.0379 / -0.0392 | -0.0386 | [-0.0509, -0.0273] | 0.0225 (0.0225) | BETTER |
| TACTICAL:joint_err_rate | -0.0079 / -0.0288 | -0.0183 | [-0.0355, -0.0021] | 0.0375 (0.0375) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0513 / -0.0052 | +0.0230 | [-0.0287, +0.0768] | 0.0050 (0.0050) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0837 / +0.0274 | +0.0555 | [-0.0008, +0.1145] | 0.0366 | OPEN |
| ade6_m|bulk | -0.0163 / -0.0329 | -0.0246 | [-0.0490, +0.0008] | 0.0354 | OPEN |
| ade6_m|all | +0.0381 / -0.0001 | +0.0190 | [-0.0166, +0.0560] | 0.0037 | OPEN |
| ade_tac_m|all | +0.0513 / -0.0052 | +0.0230 | [-0.0287, +0.0768] | 0.0050 | OPEN |

## I10RK R0KHb20 minus I10B0 R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0206 / +0.0086 | +0.0146 | [+0.0026, +0.0271] | 0.0380 (0.0380) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0715 / +0.0369 | +0.0542 | [+0.0208, +0.0890] | 0.0248 (0.0248) | WORSE |
| LATERAL:cross_mae_m | -0.0285 / -0.0267 | -0.0276 | [-0.0453, -0.0113] | 0.0270 (0.0270) | BETTER |
| LATERAL:heading_mae_deg | -0.3257 / -0.2778 | -0.3018 | [-0.4142, -0.2019] | 0.5784 (0.5784) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0140 / -0.0092 | -0.0116 | [-0.0183, -0.0062] | 0.0454 (0.0454) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7949 / -0.4138 | -0.6043 | [-1.0048, -0.3112] | 2.8181 (2.8181) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0046 / -0.0025 | -0.0035 | [-0.0127, +0.0056] | 0.0584 (0.0584) | OPEN |
| TACTICAL:lat_err_rate | -0.0163 / -0.0129 | -0.0146 | [-0.0213, -0.0085] | 0.0225 (0.0225) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0071 / -0.0079 | -0.0075 | [-0.0146, -0.0000] | 0.0375 (0.0375) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0620 / +0.0156 | +0.0388 | [-0.0113, +0.0871] | 0.0050 (0.0050) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0600 / +0.0193 | +0.0396 | [-0.0133, +0.0934] | 0.0366 | OPEN |
| ade6_m|bulk | +0.0276 / +0.0072 | +0.0174 | [-0.0036, +0.0403] | 0.0354 | OPEN |
| ade6_m|all | +0.0452 / +0.0138 | +0.0295 | [-0.0042, +0.0627] | 0.0037 | OPEN |
| ade_tac_m|all | +0.0620 / +0.0156 | +0.0388 | [-0.0113, +0.0871] | 0.0050 | OPEN |

## I10RK R0Hb20 minus I10RH R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0005 / +0.0130 | +0.0068 | [-0.0103, +0.0201] | 0.0380 (0.0380) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0072 / +0.0141 | +0.0107 | [-0.0082, +0.0293] | 0.0248 (0.0248) | OPEN |
| LATERAL:cross_mae_m | -0.0039 / -0.0197 | -0.0118 | [-0.0287, +0.0059] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | +0.0314 / +0.0117 | +0.0215 | [-0.0516, +0.0974] | 0.5784 (0.5784) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0016 / +0.0041 | +0.0013 | [-0.0052, +0.0079] | 0.0454 (0.0454) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.1501 / +0.8944 | +0.5222 | [-0.0684, +1.1359] | 2.8181 (2.8181) | OPEN |
| TACTICAL:lon_err_rate | -0.0071 / -0.0008 | -0.0040 | [-0.0150, +0.0056] | 0.0584 (0.0584) | OPEN |
| TACTICAL:lat_err_rate | -0.0025 / -0.0054 | -0.0040 | [-0.0131, +0.0054] | 0.0225 (0.0225) | OPEN |
| TACTICAL:joint_err_rate | -0.0104 / -0.0058 | -0.0081 | [-0.0196, +0.0027] | 0.0375 (0.0375) | OPEN |
| TACTICAL:ade_tac_m | +0.0068 / +0.0096 | +0.0082 | [-0.0206, +0.0380] | 0.0050 (0.0050) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0255 / +0.0111 | -0.0072 | [-0.0479, +0.0276] | 0.0366 | OPEN |
| ade6_m|bulk | +0.0361 / -0.0051 | +0.0155 | [-0.0154, +0.0507] | 0.0354 | OPEN |
| ade6_m|all | +0.0026 / +0.0037 | +0.0031 | [-0.0162, +0.0234] | 0.0037 | OPEN |
| ade_tac_m|all | +0.0068 / +0.0096 | +0.0082 | [-0.0206, +0.0380] | 0.0050 | OPEN |

## I10RK R0C10XHb20 minus I10RH R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0081 / +0.0246 | +0.0164 | [+0.0011, +0.0311] | 0.0380 (0.0380) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0016 / +0.0321 | +0.0153 | [-0.0127, +0.0432] | 0.0248 (0.0248) | OPEN |
| LATERAL:cross_mae_m | -0.0215 / -0.0152 | -0.0183 | [-0.0294, -0.0078] | 0.0270 (0.0270) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0802 / +0.0647 | -0.0078 | [-0.1184, +0.1278] | 0.5784 (0.5784) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0053 / +0.0396 | +0.0225 | [+0.0003, +0.0549] | 0.0454 (0.0454) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.4432 / +2.1807 | +1.3120 | [+0.1980, +2.7382] | 2.8181 (2.8181) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0050 / +0.0025 | +0.0038 | [-0.0054, +0.0125] | 0.0584 (0.0584) | OPEN |
| TACTICAL:lat_err_rate | -0.0088 / +0.0071 | -0.0008 | [-0.0131, +0.0115] | 0.0225 (0.0225) | OPEN |
| TACTICAL:joint_err_rate | -0.0029 / +0.0063 | +0.0017 | [-0.0085, +0.0127] | 0.0375 (0.0375) | OPEN |
| TACTICAL:ade_tac_m | -0.0228 / +0.0347 | +0.0059 | [-0.0406, +0.0517] | 0.0050 (0.0050) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0563 / +0.0022 | -0.0271 | [-0.0712, +0.0169] | 0.0366 | OPEN |
| ade6_m|bulk | +0.0295 / +0.0422 | +0.0359 | [+0.0129, +0.0602] | 0.0354 | WORSE |
| ade6_m|all | -0.0172 / +0.0204 | +0.0016 | [-0.0290, +0.0318] | 0.0037 | OPEN |
| ade_tac_m|all | -0.0228 / +0.0347 | +0.0059 | [-0.0406, +0.0517] | 0.0050 | OPEN |

## I10RK R0C10XHb20 minus I10RL R0C10XHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0017 / +0.0104 | +0.0060 | [-0.0050, +0.0172] | 0.0380 (0.0380) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0022 / +0.0135 | +0.0056 | [-0.0142, +0.0245] | 0.0248 (0.0248) | OPEN |
| LATERAL:cross_mae_m | -0.0066 / +0.0061 | -0.0002 | [-0.0115, +0.0103] | 0.0270 (0.0270) | OPEN |
| LATERAL:heading_mae_deg | +0.0003 / +0.2183 | +0.1093 | [-0.0353, +0.2759] | 0.5784 (0.5784) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0021 / +0.0453 | +0.0237 | [-0.0015, +0.0605] | 0.0454 (0.0454) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.1214 / +2.4208 | +1.1497 | [-0.3081, +3.0284] | 2.8181 (2.8181) | OPEN |
| TACTICAL:lon_err_rate | -0.0042 / -0.0058 | -0.0050 | [-0.0154, +0.0050] | 0.0584 (0.0584) | OPEN |
| TACTICAL:lat_err_rate | -0.0108 / +0.0067 | -0.0021 | [-0.0156, +0.0106] | 0.0225 (0.0225) | OPEN |
| TACTICAL:joint_err_rate | -0.0092 / -0.0050 | -0.0071 | [-0.0177, +0.0025] | 0.0375 (0.0375) | OPEN |
| TACTICAL:ade_tac_m | -0.0053 / +0.0279 | +0.0113 | [-0.0210, +0.0432] | 0.0050 (0.0050) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0218 / +0.0212 | -0.0003 | [-0.0338, +0.0316] | 0.0366 | OPEN |
| ade6_m|bulk | +0.0154 / +0.0111 | +0.0132 | [-0.0113, +0.0391] | 0.0354 | OPEN |
| ade6_m|all | -0.0049 / +0.0166 | +0.0059 | [-0.0156, +0.0269] | 0.0037 | OPEN |
| ade_tac_m|all | -0.0053 / +0.0279 | +0.0113 | [-0.0210, +0.0432] | 0.0050 | OPEN |

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
| I10RK_R0C10Hb20_s0 | 2.6364 | 3.1774 | 1.1724 | 1.0351 | 0.5807 |
| I10RK_R0C10Hb20_s1 | 2.5851 | 3.0852 | 1.1977 | 1.0252 | 0.6148 |
| I10RK_R0C10XHb20_s0 | 2.4980 | 3.1255 | 1.1954 | 0.8106 | 0.5632 |
| I10RK_R0C10XHb20_s1 | 2.4358 | 3.0289 | 1.2008 | 0.7894 | 0.5973 |
| I10RK_R0G0Hb20_s0 | 2.4073 | 3.0597 | 0.9746 | 1.0475 | 0.5319 |
| I10RK_R0G0Hb20_s1 | 2.3413 | 2.9702 | 0.9437 | 1.0776 | 0.5119 |
| I10RK_R0G50Hb20_s0 | 2.5070 | 3.0804 | 1.0719 | 1.0160 | 0.5615 |
| I10RK_R0G50Hb20_s1 | 2.4061 | 2.9521 | 1.0332 | 1.0200 | 0.5502 |
| I10RK_R0Hb20_s0 | 2.4931 | 3.0383 | 1.1174 | 1.0116 | 0.5277 |
| I10RK_R0Hb20_s1 | 2.4894 | 3.0018 | 1.1554 | 1.0386 | 0.5652 |
| I10RK_R0KHb20_s0 | 2.2159 | 3.0161 | 0.9786 | 0.8123 | 0.4573 |
| I10RK_R0KHb20_s1 | 2.1164 | 2.8987 | 0.9392 | 0.7825 | 0.4393 |
| I10RL_R0C10XHb20_s0 | 2.5029 | 3.1473 | 1.1938 | 0.8171 | 0.5723 |
| I10RL_R0C10XHb20_s1 | 2.4193 | 3.0077 | 1.1904 | 0.7833 | 0.6023 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
