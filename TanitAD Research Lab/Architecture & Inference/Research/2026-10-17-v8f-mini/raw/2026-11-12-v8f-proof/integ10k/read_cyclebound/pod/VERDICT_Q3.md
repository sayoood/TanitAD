# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 36}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0M5Hb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867476, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0M5Hb20 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -1.1916 / -1.2915 | -1.2415 | [-1.7076, -0.8337] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -1.1067 / -0.6086 | -0.8576 | [-1.3372, -0.4864] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.2469 / -0.5874 | -0.4172 | [-0.7761, -0.1289] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -0.1843 / -0.5755 | -0.3799 | [-0.8640, +0.0681] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0015 / +0.0039 | +0.0027 | [-0.0031, +0.0089] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.4942 / -3.4706 | -3.4824 | [-5.8655, -1.2297] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0004 / +0.0000 | -0.0002 | [-0.0079, +0.0075] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0263 / -0.0250 | -0.0256 | [-0.0388, -0.0140] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0104 / -0.0033 | -0.0069 | [-0.0156, +0.0013] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -1.5922 / -1.3206 | -1.4564 | [-2.0390, -0.9207] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -1.5348 / -1.2474 | -1.3911 | [-1.9600, -0.8647] | 0.0229 | BETTER |
| ade6_m|bulk | -0.5133 / -0.4610 | -0.4871 | [-0.8877, -0.1858] | 0.1243 | BETTER |
| ade6_m|all | -1.0690 / -0.8888 | -0.9789 | [-1.3677, -0.6191] | 0.0442 | BETTER |
| ade_tac_m|all | -1.5922 / -1.3206 | -1.4564 | [-2.0390, -0.9207] | 0.0699 | BETTER |

## I10RLcM5Hb20 minus I10RLcM5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.7357 / -0.8311 | -0.7834 | [-1.1197, -0.4753] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.6811 / -0.4621 | -0.5716 | [-0.8573, -0.3447] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1924 / -0.3159 | -0.2542 | [-0.4502, -0.0868] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -1.0110 / -1.3874 | -1.1992 | [-1.7386, -0.7142] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0019 / +0.0107 | +0.0063 | [-0.0036, +0.0162] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -2.8603 / -4.7563 | -3.8083 | [-6.3246, -1.5530] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.0038 / -0.0013 | +0.0013 | [-0.0106, +0.0127] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0317 / -0.0304 | -0.0311 | [-0.0442, -0.0190] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0100 / -0.0071 | -0.0085 | [-0.0167, -0.0008] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -1.0086 / -0.8293 | -0.9190 | [-1.3462, -0.5446] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.9258 / -0.8400 | -0.8829 | [-1.2643, -0.5191] | 0.0229 | BETTER |
| ade6_m|bulk | -0.3824 / -0.2242 | -0.3033 | [-0.6242, -0.0825] | 0.1243 | BETTER |
| ade6_m|all | -0.6780 / -0.5591 | -0.6186 | [-0.9048, -0.3675] | 0.0442 | BETTER |
| ade_tac_m|all | -1.0086 / -0.8293 | -0.9190 | [-1.3462, -0.5446] | 0.0699 | BETTER |

## I10RQ50M5Hb20 minus I10RQ50M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.6144 / -0.6807 | -0.6476 | [-0.9407, -0.3953] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.6167 / -0.3548 | -0.4858 | [-0.7442, -0.2747] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.1823 / -0.2064 | -0.1944 | [-0.3302, -0.0578] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -1.7270 / -1.4915 | -1.6093 | [-2.1972, -1.1197] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0058 / +0.0052 | +0.0055 | [-0.0042, +0.0150] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -2.4421 / -5.9821 | -4.2121 | [-7.4661, -1.2450] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0017 / -0.0042 | -0.0029 | [-0.0140, +0.0094] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0321 / -0.0342 | -0.0331 | [-0.0476, -0.0200] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0071 / -0.0108 | -0.0090 | [-0.0171, -0.0010] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.9213 / -0.5848 | -0.7531 | [-1.1577, -0.4149] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.8916 / -0.6202 | -0.7559 | [-1.1623, -0.4155] | 0.0229 | BETTER |
| ade6_m|bulk | -0.2958 / -0.1305 | -0.2132 | [-0.4075, -0.0811] | 0.1243 | BETTER |
| ade6_m|all | -0.6199 / -0.3969 | -0.5084 | [-0.7785, -0.2822] | 0.0442 | BETTER |
| ade_tac_m|all | -0.9213 / -0.5848 | -0.7531 | [-1.1577, -0.4149] | 0.0699 | BETTER |

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

## I10R0M5Hb20 minus I10R0Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +9.1141 / +10.9066 | +10.0103 | [+8.7767, +11.1689] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +10.6284 / +14.1426 | +12.3855 | [+10.2284, +14.5225] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +3.7194 / +3.8712 | +3.7953 | [+3.3804, +4.2481] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +5.3735 / +5.6518 | +5.5127 | [+4.7438, +6.3014] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0534 / -0.0668 | -0.0601 | [-0.0804, -0.0437] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +19.9180 / +15.6469 | +17.7824 | [+13.9648, +22.0011] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2793 / +0.2564 | +0.2678 | [+0.2187, +0.3142] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1717 / +0.3535 | +0.2626 | [+0.1501, +0.3715] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2684 / +0.2755 | +0.2720 | [+0.2364, +0.3073] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +17.9258 / +22.9669 | +20.4464 | [+17.2005, +23.5575] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +12.6856 / +16.2496 | +14.4676 | [+12.2400, +16.7152] | 0.0229 | WORSE |
| ade6_m|bulk | +11.1683 / +14.3028 | +12.7355 | [+10.2653, +14.9024] | 0.1243 | WORSE |
| ade6_m|all | +11.9936 / +15.3618 | +13.6777 | [+11.5088, +15.7595] | 0.0442 | WORSE |
| ade_tac_m|all | +17.9258 / +22.9669 | +20.4464 | [+17.2005, +23.5575] | 0.0699 | WORSE |

## I10R0KM5Hb20 minus I10R0KHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3859 / +0.3978 | +0.3918 | [+0.2941, +0.5127] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.0483 / +1.1099 | +1.0791 | [+0.8959, +1.2899] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.5105 / +0.4408 | +0.4756 | [+0.3595, +0.6040] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +1.4840 / +1.4697 | +1.4768 | [+1.0464, +1.8906] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0363 / -0.0361 | -0.0362 | [-0.0512, -0.0231] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -1.1696 / -0.5912 | -0.8804 | [-1.9157, +0.2243] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0300 / +0.0563 | +0.0431 | [+0.0056, +0.0838] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.0117 / +0.0229 | +0.0173 | [-0.0050, +0.0405] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0275 / +0.0629 | +0.0452 | [+0.0075, +0.0833] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.9939 / +2.0062 | +2.0000 | [+1.6988, +2.3356] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.9168 / +1.8857 | +1.9013 | [+1.6536, +2.1558] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6297 / +0.6804 | +0.6550 | [+0.3992, +1.0319] | 0.1243 | WORSE |
| ade6_m|all | +1.3298 / +1.3361 | +1.3329 | [+1.1338, +1.5545] | 0.0442 | WORSE |
| ade_tac_m|all | +1.9939 / +2.0062 | +2.0000 | [+1.6988, +2.3356] | 0.0699 | WORSE |

## I10RLcKM5Hb20 minus I10RLcKF5Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3317 / +0.3296 | +0.3306 | [+0.2326, +0.4533] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.9733 / +1.0037 | +0.9885 | [+0.8094, +1.1991] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.5074 / +0.4378 | +0.4726 | [+0.3549, +0.6022] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +1.4517 / +1.4186 | +1.4352 | [+0.9952, +1.8567] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0353 / -0.0331 | -0.0342 | [-0.0483, -0.0216] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -1.1452 / -0.4767 | -0.8109 | [-1.8644, +0.3073] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0129 / +0.0396 | +0.0263 | [-0.0115, +0.0669] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0150 / +0.0221 | +0.0185 | [-0.0029, +0.0409] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0200 / +0.0525 | +0.0363 | [-0.0008, +0.0742] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +1.8731 / +1.8410 | +1.8570 | [+1.5453, +2.1957] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.7186 / +1.6436 | +1.6811 | [+1.4203, +1.9347] | 0.0229 | WORSE |
| ade6_m|bulk | +0.7003 / +0.7387 | +0.7195 | [+0.4443, +1.1149] | 0.1243 | WORSE |
| ade6_m|all | +1.2542 / +1.2310 | +1.2426 | [+1.0355, +1.4666] | 0.0442 | WORSE |
| ade_tac_m|all | +1.8731 / +1.8410 | +1.8570 | [+1.5453, +2.1957] | 0.0699 | WORSE |

## I10RQ50KM5Hb20 minus I10RQ50KF5Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3088 / +0.3086 | +0.3087 | [+0.2150, +0.4300] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.9451 / +0.9824 | +0.9637 | [+0.7824, +1.1705] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.5031 / +0.4354 | +0.4692 | [+0.3530, +0.5960] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +1.4337 / +1.4284 | +1.4311 | [+1.0016, +1.8446] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0283 / -0.0307 | -0.0295 | [-0.0436, -0.0176] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -1.0448 / -0.4841 | -0.7644 | [-1.7438, +0.3271] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0054 / +0.0192 | +0.0123 | [-0.0219, +0.0477] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0088 / +0.0204 | +0.0146 | [-0.0079, +0.0382] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0079 / +0.0350 | +0.0215 | [-0.0144, +0.0556] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +1.8334 / +1.8129 | +1.8231 | [+1.5137, +2.1589] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.7467 / +1.6686 | +1.7076 | [+1.4585, +1.9503] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6067 / +0.6654 | +0.6361 | [+0.3717, +1.0152] | 0.1243 | WORSE |
| ade6_m|all | +1.2268 / +1.2111 | +1.2190 | [+1.0144, +1.4400] | 0.0442 | WORSE |
| ade_tac_m|all | +1.8334 / +1.8129 | +1.8231 | [+1.5137, +2.1589] | 0.0699 | WORSE |

## I10RLcM5Hb20 minus I10RLcHb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +6.7748 / +8.0205 | +7.3977 | [+6.4170, +8.3816] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +8.0764 / +10.4561 | +9.2663 | [+7.6280, +10.9249] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +2.9874 / +3.0536 | +3.0205 | [+2.6504, +3.4246] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +7.0660 / +7.3271 | +7.1965 | [+6.3745, +8.0728] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0011 / -0.0141 | -0.0076 | [-0.0251, +0.0097] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +32.1753 / +26.9331 | +29.5542 | [+24.5689, +34.7768] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2230 / +0.1897 | +0.2063 | [+0.1539, +0.2531] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1901 / +0.3451 | +0.2676 | [+0.1700, +0.3628] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2414 / +0.2464 | +0.2439 | [+0.2098, +0.2791] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +13.7091 / +17.2265 | +15.4678 | [+12.9514, +17.9873] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +9.4020 / +11.8262 | +10.6141 | [+8.8554, +12.4121] | 0.0229 | WORSE |
| ade6_m|bulk | +8.9286 / +11.1935 | +10.0610 | [+8.1008, +11.8883] | 0.1243 | WORSE |
| ade6_m|all | +9.1861 / +11.5377 | +10.3619 | [+8.6786, +12.0467] | 0.0442 | WORSE |
| ade_tac_m|all | +13.7091 / +17.2265 | +15.4678 | [+12.9514, +17.9873] | 0.0699 | WORSE |

## I10RQ50M5Hb20 minus I10RQ50Hb20

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +5.7445 / +6.5353 | +6.1399 | [+5.4082, +6.8773] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +7.3067 / +8.9800 | +8.1433 | [+6.8771, +9.4208] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +2.6270 / +2.6526 | +2.6398 | [+2.2893, +3.0388] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +7.1111 / +7.4764 | +7.2937 | [+6.4544, +8.1845] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0232 / +0.0034 | +0.0133 | [-0.0078, +0.0363] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +35.7217 / +27.5844 | +31.6531 | [+25.4067, +38.1005] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2018 / +0.1571 | +0.1794 | [+0.1253, +0.2285] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1838 / +0.3214 | +0.2526 | [+0.1642, +0.3399] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2297 / +0.2384 | +0.2341 | [+0.1982, +0.2689] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +12.3103 / +14.8324 | +13.5713 | [+11.5757, +15.5622] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +8.3855 / +10.2089 | +9.2972 | [+7.8907, +10.7207] | 0.0229 | WORSE |
| ade6_m|bulk | +8.1036 / +9.6252 | +8.8644 | [+7.2837, +10.3265] | 0.1243 | WORSE |
| ade6_m|all | +8.2569 / +9.9427 | +9.0998 | [+7.7631, +10.4312] | 0.0442 | WORSE |
| ade_tac_m|all | +12.3103 / +14.8324 | +13.5713 | [+11.5757, +15.5622] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0Hb20_s0 | 2.6128 | 2.9555 | 1.3614 | 1.1534 | 0.6332 |
| I10R0Hb20_s1 | 2.5138 | 2.8970 | 1.4041 | 1.0196 | 0.6561 |
| I10R0KHb20_s0 | 2.1659 | 2.8955 | 0.9914 | 0.8576 | 0.4698 |
| I10R0KHb20_s1 | 2.1722 | 2.9106 | 0.9979 | 0.8717 | 0.4489 |
| I10R0KM5Hb20_s0 | 3.4957 | 4.8123 | 1.3773 | 1.3681 | 0.4973 |
| I10R0KM5Hb20_s1 | 3.5082 | 4.7963 | 1.3957 | 1.3125 | 0.5119 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |
| I10R0M5Hb20_s0 | 14.6064 | 15.6411 | 10.4754 | 4.8728 | 0.9016 |
| I10R0M5Hb20_s1 | 17.8757 | 19.1466 | 12.3107 | 4.8908 | 0.9316 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcHb20_s0 | 2.8592 | 3.1080 | 1.4672 | 1.1511 | 0.6394 |
| I10RLcHb20_s1 | 2.7335 | 3.0355 | 1.5079 | 1.0203 | 0.6653 |
| I10RLcKF5Hb20_s0 | 2.2955 | 2.9071 | 1.0220 | 0.8551 | 0.4685 |
| I10RLcKF5Hb20_s1 | 2.3249 | 2.9503 | 1.0394 | 0.8688 | 0.4502 |
| I10RLcKM5Hb20_s0 | 3.5497 | 4.6257 | 1.3536 | 1.3626 | 0.4885 |
| I10RLcKM5Hb20_s1 | 3.5558 | 4.5939 | 1.3690 | 1.3066 | 0.5027 |
| I10RLcM5_s0 | 12.7234 | 13.4358 | 8.9777 | 4.3309 | 0.8908 |
| I10RLcM5_s1 | 14.8304 | 15.7017 | 10.3595 | 4.3898 | 0.9187 |
| I10RLcM5Hb20_s0 | 12.0454 | 12.5100 | 8.2420 | 4.1385 | 0.8808 |
| I10RLcM5Hb20_s1 | 14.2712 | 14.8618 | 9.5285 | 4.0739 | 0.9116 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50Hb20_s0 | 2.9403 | 3.2265 | 1.5110 | 1.1487 | 0.6436 |
| I10RQ50Hb20_s1 | 2.8428 | 3.1096 | 1.5558 | 1.0299 | 0.6649 |
| I10RQ50KF5Hb20_s0 | 2.3616 | 3.0146 | 1.0367 | 0.8537 | 0.4644 |
| I10RQ50KF5Hb20_s1 | 2.3293 | 2.9357 | 1.0349 | 0.8674 | 0.4460 |
| I10RQ50KM5Hb20_s0 | 3.5885 | 4.7613 | 1.3455 | 1.3569 | 0.4723 |
| I10RQ50KM5Hb20_s1 | 3.5404 | 4.6043 | 1.3435 | 1.3028 | 0.4810 |
| I10RQ50M5_s0 | 11.8171 | 12.5035 | 7.8699 | 3.9580 | 0.8804 |
| I10RQ50M5_s1 | 13.1824 | 13.9387 | 8.7719 | 3.8890 | 0.9141 |
| I10RQ50M5Hb20_s0 | 11.1972 | 11.6120 | 7.2555 | 3.7757 | 0.8733 |
| I10RQ50M5Hb20_s1 | 12.7855 | 13.3185 | 8.0911 | 3.6826 | 0.9033 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
