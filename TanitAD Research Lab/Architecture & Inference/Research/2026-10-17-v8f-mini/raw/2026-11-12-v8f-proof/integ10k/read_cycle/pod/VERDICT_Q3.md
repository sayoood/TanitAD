# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0M5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.1536295382986747, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0M5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +10.0703 / +11.9542 | +11.0122 | [+9.6755, +12.2970] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +11.6936 / +14.6671 | +13.1803 | [+11.2596, +15.0195] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +3.8253 / +4.2896 | +4.0575 | [+3.5384, +4.6282] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +5.4728 / +4.3501 | +4.9115 | [+3.9533, +6.1583] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0647 / -0.0964 | -0.0805 | [-0.1135, -0.0541] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +20.2784 / +12.9993 | +16.6389 | [+10.5730, +22.6852] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2755 / +0.2518 | +0.2637 | [+0.2152, +0.3097] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1571 / +0.3368 | +0.2470 | [+0.1354, +0.3559] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2651 / +0.2634 | +0.2643 | [+0.2278, +0.2982] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +19.3350 / +24.0224 | +21.6787 | [+18.5772, +24.6430] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +14.0632 / +17.2584 | +15.6608 | [+13.5964, +17.7510] | 0.0229 | WORSE |
| ade6_m|bulk | +11.6014 / +14.6608 | +13.1311 | [+10.7172, +15.2497] | 0.1243 | WORSE |
| ade6_m|all | +12.9406 / +16.0739 | +14.5072 | [+12.4341, +16.4944] | 0.0442 | WORSE |
| ade_tac_m|all | +19.3350 / +24.0224 | +21.6787 | [+18.5772, +24.6430] | 0.0699 | WORSE |

## I10RLcM5 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +7.3267 / +8.6658 | +7.9962 | [+6.9605, +9.0757] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +8.7317 / +10.8471 | +9.7894 | [+8.2902, +11.3232] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +3.0637 / +3.2261 | +3.1449 | [+2.7647, +3.5690] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +7.1021 / +6.7037 | +6.9029 | [+6.0499, +7.7900] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0129 / -0.0516 | -0.0323 | [-0.0664, -0.0008] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +31.7615 / +27.1829 | +29.4722 | [+24.5205, +34.7422] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2201 / +0.1888 | +0.2045 | [+0.1547, +0.2488] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1872 / +0.3385 | +0.2628 | [+0.1661, +0.3578] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2426 / +0.2397 | +0.2411 | [+0.2056, +0.2742] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +14.5814 / +17.8368 | +16.2091 | [+13.8067, +18.6225] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +10.1940 / +12.4610 | +11.3275 | [+9.6418, +13.0546] | 0.0229 | WORSE |
| ade6_m|bulk | +9.2713 / +11.3422 | +10.3067 | [+8.4702, +12.0807] | 0.1243 | WORSE |
| ade6_m|all | +9.7732 / +11.9508 | +10.8620 | [+9.2523, +12.4758] | 0.0442 | WORSE |
| ade_tac_m|all | +14.5814 / +17.8368 | +16.2091 | [+13.8067, +18.6225] | 0.0699 | WORSE |

## I10RQ50M5 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +6.2152 / +7.0674 | +6.6413 | [+5.8701, +7.4382] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +7.9004 / +9.2604 | +8.5804 | [+7.4337, +9.7087] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +2.7037 / +2.7243 | +2.7140 | [+2.3708, +3.0901] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +7.7177 / +6.8476 | +7.2827 | [+6.3364, +8.3598] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0093 / -0.0318 | -0.0113 | [-0.0428, +0.0191] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +35.0273 / +29.1518 | +32.0896 | [+26.7738, +37.5162] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2030 / +0.1613 | +0.1822 | [+0.1305, +0.2282] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1855 / +0.3180 | +0.2518 | [+0.1656, +0.3364] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2272 / +0.2363 | +0.2318 | [+0.1972, +0.2669] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +13.1092 / +15.2048 | +14.1570 | [+12.3498, +15.9668] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +9.1549 / +10.6302 | +9.8926 | [+8.6088, +11.1628] | 0.0229 | WORSE |
| ade6_m|bulk | +8.3662 / +9.6824 | +9.0243 | [+7.5647, +10.3982] | 0.1243 | WORSE |
| ade6_m|all | +8.7953 / +10.1980 | +9.4966 | [+8.2850, +10.7067] | 0.0442 | WORSE |
| ade_tac_m|all | +13.1092 / +15.2048 | +14.1570 | [+12.3498, +15.9668] | 0.0699 | WORSE |

## I10RLcM5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -2.6893 / -3.2427 | -2.9660 | [-3.6742, -2.3220] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -2.7095 / -3.6035 | -3.1565 | [-4.0945, -2.3287] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.7888 / -1.0884 | -0.9386 | [-1.3290, -0.6145] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | +2.2909 / +2.6961 | +2.4935 | [+1.7232, +3.2543] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0687 / +0.0611 | +0.0649 | [+0.0471, +0.0855] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.4262 / +13.9356 | +13.1809 | [+9.7706, +16.7846] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0588 / -0.0600 | -0.0594 | [-0.0769, -0.0425] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0354 / +0.0083 | +0.0219 | [-0.0025, +0.0494] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0213 / -0.0163 | -0.0188 | [-0.0285, -0.0098] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -4.4619 / -5.9332 | -5.1976 | [-6.7519, -3.8536] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -3.7400 / -4.6923 | -4.2162 | [-5.4066, -3.1282] | 0.0229 | BETTER |
| ade6_m|bulk | -2.0120 / -3.0296 | -2.5208 | [-3.6458, -1.5673] | 0.1243 | BETTER |
| ade6_m|all | -2.9520 / -3.9341 | -3.4430 | [-4.4811, -2.5471] | 0.0442 | BETTER |
| ade_tac_m|all | -4.4619 / -5.9332 | -5.1976 | [-6.7519, -3.8536] | 0.0699 | BETTER |

## I10RQ50M5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -3.7971 / -4.8303 | -4.3137 | [-5.2707, -3.4374] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -3.4598 / -5.0723 | -4.2661 | [-5.5611, -3.0531] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -1.1618 / -1.5893 | -1.3755 | [-1.8722, -0.9906] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | +3.1577 / +2.9310 | +3.0443 | [+2.1864, +3.9771] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1006 / +0.0844 | +0.0925 | [+0.0681, +0.1209] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +16.7761 / +16.1237 | +16.4499 | [+12.5555, +20.4009] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0779 / -0.0905 | -0.0842 | [-0.1067, -0.0636] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0383 / -0.0025 | +0.0179 | [-0.0127, +0.0535] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0317 / -0.0208 | -0.0263 | [-0.0375, -0.0148] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -5.8370 / -8.4088 | -7.1229 | [-9.2239, -5.1935] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -4.6723 / -6.4553 | -5.5638 | [-7.1903, -4.1634] | 0.0229 | BETTER |
| ade6_m|bulk | -2.8873 / -4.5403 | -3.7138 | [-5.1530, -2.4063] | 0.1243 | BETTER |
| ade6_m|all | -3.8583 / -5.5820 | -4.7202 | [-6.1211, -3.4169] | 0.0442 | BETTER |
| ade_tac_m|all | -5.8370 / -8.4088 | -7.1229 | [-9.2239, -5.1935] | 0.0699 | BETTER |

## I10RLc minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0543 / +0.0458 | +0.0501 | [-0.0220, +0.1299] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2524 / +0.2165 | +0.2344 | [+0.0689, +0.4342] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0272 / -0.0249 | -0.0261 | [-0.0431, -0.0100] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4046 / +0.2709 | +0.3377 | [+0.1335, +0.5319] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0159 / +0.0147 | +0.0153 | [+0.0084, +0.0232] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.0810 / +0.1411 | +0.6110 | [-0.4238, +1.5898] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0033 / +0.0029 | -0.0002 | [-0.0098, +0.0088] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0054 / +0.0067 | +0.0060 | [-0.0042, +0.0171] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0013 / +0.0075 | +0.0044 | [-0.0048, +0.0138] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.2917 / +0.2524 | +0.2721 | [+0.0598, +0.5289] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1291 / +0.1052 | +0.1171 | [-0.0193, +0.2947] | 0.0229 | OPEN |
| ade6_m|bulk | +0.3182 / +0.2890 | +0.3036 | [+0.1100, +0.5312] | 0.1243 | WORSE |
| ade6_m|all | +0.2153 / +0.1890 | +0.2022 | [+0.0475, +0.3897] | 0.0442 | WORSE |
| ade_tac_m|all | +0.2917 / +0.2524 | +0.2721 | [+0.0598, +0.5289] | 0.0699 | WORSE |

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

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcM5_s0 | 12.7234 | 13.4358 | 8.9777 | 4.3309 | 0.8908 |
| I10RLcM5_s1 | 14.8304 | 15.7017 | 10.3595 | 4.3898 | 0.9187 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50M5_s0 | 11.8171 | 12.5035 | 7.8699 | 3.9580 | 0.8804 |
| I10RQ50M5_s1 | 13.1824 | 13.9387 | 8.7719 | 3.8890 | 0.9141 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
