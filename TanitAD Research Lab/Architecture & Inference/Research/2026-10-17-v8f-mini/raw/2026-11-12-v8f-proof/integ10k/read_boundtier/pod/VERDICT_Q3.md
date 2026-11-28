# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 32}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0C10Hb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867476, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0C10Hb20 minus I10R0C10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.4591 / -0.6217 | -0.5404 | [-0.7372, -0.3778] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0983 / -0.1955 | -0.1469 | [-0.2359, -0.0657] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.2359 / -0.1442 | -0.1901 | [-0.2703, -0.1216] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.2035 / -2.0362 | -2.1199 | [-2.5735, -1.6952] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.1257 / +0.0541 | +0.0899 | [+0.0410, +0.1497] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -6.3291 / -11.4741 | -8.9016 | [-13.2482, -5.3477] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0150 / +0.0038 | -0.0056 | [-0.0267, +0.0161] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0496 / -0.0442 | -0.0469 | [-0.0646, -0.0298] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0250 / -0.0183 | -0.0217 | [-0.0342, -0.0094] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.3224 / -0.3680 | -0.3452 | [-0.4585, -0.2404] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2939 / -0.3228 | -0.3083 | [-0.4140, -0.2111] | 0.0229 | BETTER |
| ade6_m|bulk | -0.1208 / -0.1530 | -0.1369 | [-0.2106, -0.0768] | 0.1243 | BETTER |
| ade6_m|all | -0.2150 / -0.2454 | -0.2302 | [-0.3057, -0.1603] | 0.0442 | BETTER |
| ade_tac_m|all | -0.3224 / -0.3680 | -0.3452 | [-0.4585, -0.2404] | 0.0699 | BETTER |

## I10R0C10XHb20 minus I10R0C10X

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1499 / -0.1479 | -0.1489 | [-0.2321, -0.0737] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1003 / -0.0526 | -0.0764 | [-0.1471, -0.0238] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1050 / -0.0957 | -0.1004 | [-0.1401, -0.0614] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.8957 / -1.6189 | -1.7573 | [-2.2087, -1.2936] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0866 / +0.0606 | +0.0736 | [+0.0429, +0.1145] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -5.6661 / -7.2549 | -6.4605 | [-9.0890, -4.1990] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0025 / +0.0038 | +0.0006 | [-0.0177, +0.0185] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0467 / -0.0300 | -0.0383 | [-0.0583, -0.0184] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0200 / -0.0096 | -0.0148 | [-0.0288, -0.0017] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.2266 / -0.1618 | -0.1942 | [-0.3107, -0.1023] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1977 / -0.1609 | -0.1793 | [-0.2757, -0.1005] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0955 / -0.0447 | -0.0701 | [-0.1467, -0.0072] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.1511 / -0.1079 | -0.1295 | [-0.2071, -0.0682] | 0.0442 | BETTER |
| ade_tac_m|all | -0.2266 / -0.1618 | -0.1942 | [-0.3107, -0.1023] | 0.0699 | BETTER |

## I10RLcHb20 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1839 / -0.1858 | -0.1848 | [-0.2596, -0.1232] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0258 / -0.0712 | -0.0485 | [-0.0870, -0.0119] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1161 / -0.1434 | -0.1298 | [-0.1735, -0.0910] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -1.3009 / -1.8679 | -1.5844 | [-2.2821, -0.9866] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0044 / -0.0063 | -0.0010 | [-0.0135, +0.0108] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -4.2147 / -4.1575 | -4.1861 | [-6.0951, -2.6427] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.0008 / -0.0021 | -0.0006 | [-0.0131, +0.0110] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0346 / -0.0371 | -0.0358 | [-0.0494, -0.0236] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0088 / -0.0138 | -0.0113 | [-0.0204, -0.0027] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1364 / -0.2190 | -0.1777 | [-0.2635, -0.1053] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1338 / -0.2052 | -0.1695 | [-0.2507, -0.1003] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0398 / -0.0754 | -0.0576 | [-0.1011, -0.0194] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.0909 / -0.1460 | -0.1185 | [-0.1757, -0.0702] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1364 / -0.2190 | -0.1777 | [-0.2635, -0.1053] | 0.0699 | BETTER |

## I10RQ50Hb20 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1438 / -0.1486 | -0.1462 | [-0.2038, -0.0952] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0230 / -0.0743 | -0.0486 | [-0.0897, -0.0102] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1055 / -0.1347 | -0.1201 | [-0.1613, -0.0845] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -1.3049 / -1.9803 | -1.6426 | [-2.3858, -1.0042] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0010 / -0.0110 | -0.0050 | [-0.0188, +0.0072] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -4.1057 / -4.1366 | -4.1212 | [-5.8749, -2.6895] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0004 / +0.0000 | -0.0002 | [-0.0117, +0.0106] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0304 / -0.0375 | -0.0340 | [-0.0488, -0.0206] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0096 / -0.0129 | -0.0113 | [-0.0200, -0.0029] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1223 / -0.2125 | -0.1674 | [-0.2517, -0.0961] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1221 / -0.1989 | -0.1605 | [-0.2428, -0.0910] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0332 / -0.0734 | -0.0533 | [-0.0929, -0.0230] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.0816 / -0.1417 | -0.1116 | [-0.1678, -0.0641] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1223 / -0.2125 | -0.1674 | [-0.2517, -0.0961] | 0.0699 | BETTER |

## I10R0C10X minus I10R0C10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3450 / -0.7150 | -0.5300 | [-0.8184, -0.2814] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0715 / -0.2384 | -0.1550 | [-0.2811, -0.0422] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3165 / -0.0146 | -0.1655 | [-0.3846, +0.0240] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -2.5185 / -1.5715 | -2.0450 | [-2.7677, -1.3306] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0538 / -0.0023 | +0.0258 | [-0.0102, +0.0628] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -7.6226 / -10.3222 | -8.9724 | [-11.9245, -6.3749] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0025 / +0.0025 | +0.0000 | [-0.0133, +0.0119] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.1434 / -0.0271 | -0.0852 | [-0.1656, -0.0123] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0288 / -0.0029 | -0.0158 | [-0.0367, +0.0036] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.2636 / -0.2474 | -0.2555 | [-0.3719, -0.1366] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0528 / -0.0672 | -0.0600 | [-0.1669, +0.0411] | 0.0229 | OPEN |
| ade6_m|bulk | -0.3224 / -0.2814 | -0.3019 | [-0.4023, -0.2051] | 0.1243 | BETTER |
| ade6_m|all | -0.1757 / -0.1649 | -0.1703 | [-0.2480, -0.0910] | 0.0442 | BETTER |
| ade_tac_m|all | -0.2636 / -0.2474 | -0.2555 | [-0.3719, -0.1366] | 0.0699 | BETTER |

## I10R0C10XHb20 minus I10R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.1868 / +1.3285 | +1.2576 | [+1.0872, +1.4390] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.1409 / +1.3416 | +1.2413 | [+0.9830, +1.5019] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.1702 / +0.2133 | +0.1917 | [+0.1451, +0.2469] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +4.1452 / +4.8064 | +4.4758 | [+3.7575, +5.2476] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.3616 / +0.3294 | +0.3455 | [+0.2937, +0.4039] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +37.8733 / +38.9835 | +38.4284 | [+33.9937, +43.5403] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2355 / +0.3556 | +0.2955 | [+0.2109, +0.3854] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1847 / +0.1992 | +0.1920 | [+0.1628, +0.2229] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2664 / +0.3893 | +0.3278 | [+0.2428, +0.4176] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.7694 / +2.0644 | +1.9169 | [+1.5303, +2.3040] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.8770 / +0.9044 | +0.8907 | [+0.7388, +1.0405] | 0.0229 | WORSE |
| ade6_m|bulk | +1.5406 / +1.9392 | +1.7399 | [+1.2638, +2.2485] | 0.1243 | WORSE |
| ade6_m|all | +1.1796 / +1.3763 | +1.2780 | [+1.0202, +1.5360] | 0.0442 | WORSE |
| ade_tac_m|all | +1.7694 / +2.0644 | +1.9169 | [+1.5303, +2.3040] | 0.0699 | WORSE |

## I10R0KHb20 minus I10R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3700 / -0.4062 | -0.3881 | [-0.4789, -0.3018] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.3283 / -0.3135 | -0.3209 | [-0.4303, -0.2321] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.2958 / -0.1479 | -0.2219 | [-0.3529, -0.0913] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -1.6869 / -1.3409 | -1.5139 | [-1.8956, -1.1377] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0266 / -0.0304 | -0.0285 | [-0.0383, -0.0204] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -6.2747 / -6.8123 | -6.5435 | [-7.8321, -5.4644] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1255 / -0.1792 | -0.1524 | [-0.2052, -0.1021] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1705 / -0.1213 | -0.1459 | [-0.1901, -0.1065] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1634 / -0.2072 | -0.1853 | [-0.2330, -0.1405] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.6703 / -0.5125 | -0.5914 | [-0.8258, -0.3912] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0600 / +0.0136 | -0.0232 | [-0.1428, +0.0969] | 0.0229 | OPEN |
| ade6_m|bulk | -0.9084 / -0.7655 | -0.8369 | [-1.1108, -0.6293] | 0.1243 | BETTER |
| ade6_m|all | -0.4469 / -0.3417 | -0.3943 | [-0.5506, -0.2608] | 0.0442 | BETTER |
| ade_tac_m|all | -0.6703 / -0.5125 | -0.5914 | [-0.8258, -0.3912] | 0.0699 | BETTER |

## I10RLcKF5Hb20 minus I10RLcHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.4452 / -0.4685 | -0.4569 | [-0.5520, -0.3695] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.4673 / -0.3885 | -0.4279 | [-0.5691, -0.3110] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.2960 / -0.1515 | -0.2237 | [-0.3546, -0.0944] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.0445 / -1.5972 | -1.8209 | [-2.2520, -1.3800] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0338 / -0.0376 | -0.0357 | [-0.0462, -0.0269] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -8.1528 / -8.2555 | -8.2041 | [-9.5983, -7.0275] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1288 / -0.1847 | -0.1567 | [-0.2100, -0.1072] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1859 / -0.1351 | -0.1605 | [-0.2065, -0.1198] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1709 / -0.2151 | -0.1930 | [-0.2410, -0.1497] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.8467 / -0.6142 | -0.7305 | [-1.0165, -0.4957] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2009 / -0.0852 | -0.1431 | [-0.3006, +0.0017] | 0.0229 | OPEN |
| ade6_m|bulk | -0.9965 / -0.7945 | -0.8955 | [-1.1843, -0.6637] | 0.1243 | BETTER |
| ade6_m|all | -0.5637 / -0.4087 | -0.4862 | [-0.6767, -0.3297] | 0.0442 | BETTER |
| ade_tac_m|all | -0.8467 / -0.6142 | -0.7305 | [-1.0165, -0.4957] | 0.0699 | BETTER |

## I10RQ50KF5Hb20 minus I10RQ50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.4742 / -0.5210 | -0.4976 | [-0.6097, -0.3989] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.4769 / -0.4915 | -0.4842 | [-0.6256, -0.3544] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.2950 / -0.1625 | -0.2288 | [-0.3571, -0.1039] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.2216 / -1.7098 | -1.9657 | [-2.4513, -1.4752] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0386 / -0.0414 | -0.0400 | [-0.0507, -0.0306] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -8.9084 / -8.9827 | -8.9456 | [-10.3329, -7.6518] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1338 / -0.1901 | -0.1619 | [-0.2161, -0.1122] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1917 / -0.1438 | -0.1678 | [-0.2113, -0.1271] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1792 / -0.2188 | -0.1990 | [-0.2454, -0.1555] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.8714 / -0.7726 | -0.8220 | [-1.0697, -0.5914] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2119 / -0.1739 | -0.1929 | [-0.3361, -0.0568] | 0.0229 | BETTER |
| ade6_m|bulk | -1.0161 / -0.9187 | -0.9674 | [-1.2355, -0.7433] | 0.1243 | BETTER |
| ade6_m|all | -0.5786 / -0.5135 | -0.5461 | [-0.7104, -0.3935] | 0.0442 | BETTER |
| ade_tac_m|all | -0.8714 / -0.7726 | -0.8220 | [-1.0697, -0.5914] | 0.0699 | BETTER |

## I10R0Hb05 minus I10R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0417 / -0.0322 | -0.0369 | [-0.0644, -0.0152] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0122 / +0.0232 | +0.0055 | [-0.0233, +0.0342] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0360 / -0.0534 | -0.0447 | [-0.0671, -0.0243] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.0667 / -1.1478 | -1.1073 | [-1.4108, -0.8476] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0373 / -0.0501 | -0.0437 | [-0.0601, -0.0308] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.9534 / -5.7611 | -5.3573 | [-6.8660, -4.0471] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.0121 / +0.0021 | +0.0071 | [-0.0029, +0.0169] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0367 / -0.0529 | -0.0448 | [-0.0631, -0.0277] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0025 / -0.0250 | -0.0113 | [-0.0319, +0.0079] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.0418 / -0.0136 | -0.0277 | [-0.0576, +0.0011] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0474 / -0.0211 | -0.0343 | [-0.0653, -0.0040] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0045 / +0.0053 | +0.0004 | [-0.0139, +0.0169] | 0.1243 | OPEN |
| ade6_m|all | -0.0278 / -0.0091 | -0.0185 | [-0.0384, +0.0008] | 0.0442 | OPEN |
| ade_tac_m|all | -0.0418 / -0.0136 | -0.0277 | [-0.0576, +0.0011] | 0.0699 | OPEN |

## I10R0Hb10 minus I10R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0393 / -0.0284 | -0.0338 | [-0.0537, -0.0186] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0226 / -0.0028 | -0.0127 | [-0.0316, +0.0030] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0198 / -0.0287 | -0.0242 | [-0.0378, -0.0117] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.7160 / -0.7010 | -0.7085 | [-0.9216, -0.5329] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0283 / -0.0343 | -0.0313 | [-0.0432, -0.0223] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -3.4707 / -3.6234 | -3.5470 | [-4.6066, -2.6560] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.0029 / +0.0017 | +0.0023 | [-0.0044, +0.0092] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0138 / -0.0271 | -0.0204 | [-0.0329, -0.0081] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0004 / -0.0121 | -0.0058 | [-0.0165, +0.0048] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.0459 / -0.0303 | -0.0381 | [-0.0593, -0.0210] | 0.0699 (0.0699) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0456 / -0.0334 | -0.0395 | [-0.0619, -0.0233] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0128 / -0.0045 | -0.0087 | [-0.0200, +0.0029] | 0.1243 | OPEN |
| ade6_m|all | -0.0306 / -0.0202 | -0.0254 | [-0.0395, -0.0140] | 0.0442 | INSIDE-SPREAD |
| ade_tac_m|all | -0.0459 / -0.0303 | -0.0381 | [-0.0593, -0.0210] | 0.0699 | INSIDE-SPREAD |

## I10R0Hb40 minus I10R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0473 / +0.0232 | +0.0352 | [+0.0133, +0.0665] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0084 / +0.0121 | +0.0103 | [-0.0012, +0.0239] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.0238 / +0.0259 | +0.0248 | [+0.0120, +0.0404] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5733 / +0.5538 | +0.5635 | [+0.3533, +0.8065] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0321 / +0.0279 | +0.0300 | [+0.0181, +0.0453] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +3.8513 / +2.5808 | +3.2160 | [+1.8789, +4.8581] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0050 / +0.0021 | +0.0035 | [-0.0035, +0.0100] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0075 / +0.0233 | +0.0154 | [+0.0035, +0.0306] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0050 / +0.0108 | +0.0079 | [+0.0017, +0.0156] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0348 / +0.0383 | +0.0365 | [+0.0165, +0.0641] | 0.0699 (0.0699) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0337 / +0.0342 | +0.0340 | [+0.0103, +0.0656] | 0.0229 | WORSE |
| ade6_m|bulk | +0.0106 / +0.0151 | +0.0129 | [+0.0056, +0.0230] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0232 / +0.0255 | +0.0243 | [+0.0110, +0.0427] | 0.0442 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0348 / +0.0383 | +0.0365 | [+0.0165, +0.0641] | 0.0699 | INSIDE-SPREAD |

## I10R0C10XHb20 minus I10R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.8168 / +0.9223 | +0.8696 | [+0.7567, +0.9931] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.8126 / +1.0281 | +0.9203 | [+0.7073, +1.1267] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.1256 / +0.0654 | -0.0301 | [-0.1911, +0.1196] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +2.4614 / +3.4540 | +2.9577 | [+2.1301, +3.8116] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.3353 / +0.2987 | +0.3170 | [+0.2674, +0.3731] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +31.6199 / +32.2351 | +31.9275 | [+28.0716, +36.3767] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.1100 / +0.1763 | +0.1432 | [+0.0925, +0.1977] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.0142 / +0.0779 | +0.0461 | [-0.0050, +0.0934] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.1030 / +0.1822 | +0.1426 | [+0.0852, +0.2017] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.0991 / +1.5519 | +1.3255 | [+0.9538, +1.6855] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.8170 / +0.9180 | +0.8675 | [+0.7106, +1.0253] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6322 / +1.1738 | +0.9030 | [+0.4540, +1.3218] | 0.1243 | WORSE |
| ade6_m|all | +0.7327 / +1.0346 | +0.8837 | [+0.6359, +1.1237] | 0.0442 | WORSE |
| ade_tac_m|all | +1.0991 / +1.5519 | +1.3255 | [+0.9538, +1.6855] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0C10_s0 | 3.6723 | 4.0229 | 2.6732 | 1.4494 | 0.7849 |
| I10R0C10_s1 | 3.8213 | 4.0431 | 3.1893 | 1.1953 | 0.8508 |
| I10R0C10Hb20_s0 | 3.4574 | 3.7290 | 2.2141 | 1.2134 | 0.7599 |
| I10R0C10Hb20_s1 | 3.5759 | 3.7203 | 2.5675 | 1.0511 | 0.8324 |
| I10R0C10X_s0 | 3.4966 | 3.9701 | 2.3281 | 1.1328 | 0.7561 |
| I10R0C10X_s1 | 3.6564 | 3.9758 | 2.4743 | 1.1807 | 0.8479 |
| I10R0C10XHb20_s0 | 3.3455 | 3.7725 | 2.1782 | 1.0278 | 0.7361 |
| I10R0C10XHb20_s1 | 3.5485 | 3.8150 | 2.3264 | 1.0850 | 0.8383 |
| I10R0Hb05_s0 | 2.5849 | 2.9081 | 1.3197 | 1.1174 | 0.6357 |
| I10R0Hb05_s1 | 2.5048 | 2.8759 | 1.3719 | 0.9662 | 0.6311 |
| I10R0Hb10_s0 | 2.5822 | 2.9099 | 1.3221 | 1.1336 | 0.6336 |
| I10R0Hb10_s1 | 2.4936 | 2.8636 | 1.3757 | 0.9910 | 0.6440 |
| I10R0Hb20_s0 | 2.6128 | 2.9555 | 1.3614 | 1.1534 | 0.6332 |
| I10R0Hb20_s1 | 2.5138 | 2.8970 | 1.4041 | 1.0196 | 0.6561 |
| I10R0Hb40_s0 | 2.6360 | 2.9892 | 1.4086 | 1.1772 | 0.6382 |
| I10R0Hb40_s1 | 2.5394 | 2.9312 | 1.4274 | 1.0455 | 0.6669 |
| I10R0KHb20_s0 | 2.1659 | 2.8955 | 0.9914 | 0.8576 | 0.4698 |
| I10R0KHb20_s1 | 2.1722 | 2.9106 | 0.9979 | 0.8717 | 0.4489 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcHb20_s0 | 2.8592 | 3.1080 | 1.4672 | 1.1511 | 0.6394 |
| I10RLcHb20_s1 | 2.7335 | 3.0355 | 1.5079 | 1.0203 | 0.6653 |
| I10RLcKF5Hb20_s0 | 2.2955 | 2.9071 | 1.0220 | 0.8551 | 0.4685 |
| I10RLcKF5Hb20_s1 | 2.3249 | 2.9503 | 1.0394 | 0.8688 | 0.4502 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50Hb20_s0 | 2.9403 | 3.2265 | 1.5110 | 1.1487 | 0.6436 |
| I10RQ50Hb20_s1 | 2.8428 | 3.1096 | 1.5558 | 1.0299 | 0.6649 |
| I10RQ50KF5Hb20_s0 | 2.3616 | 3.0146 | 1.0367 | 0.8537 | 0.4644 |
| I10RQ50KF5Hb20_s1 | 2.3293 | 2.9357 | 1.0349 | 0.8674 | 0.4460 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
