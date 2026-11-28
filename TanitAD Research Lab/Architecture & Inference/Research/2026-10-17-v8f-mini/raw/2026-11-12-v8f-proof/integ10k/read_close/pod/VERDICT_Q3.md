# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RLc"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RLc minus I10RLb

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0227 / -0.0215 | -0.0221 | [-0.0491, +0.0024] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0539 / -0.0715 | -0.0627 | [-0.1186, -0.0067] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0044 / -0.0019 | -0.0032 | [-0.0154, +0.0080] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.2507 / -0.1782 | -0.2145 | [-0.4025, -0.0594] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0006 / -0.0053 | -0.0029 | [-0.0103, +0.0026] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.4808 / -0.6394 | -0.5601 | [-1.1727, -0.0173] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0033 / -0.0021 | -0.0027 | [-0.0077, +0.0021] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0013 / -0.0025 | -0.0019 | [-0.0075, +0.0038] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0046 / -0.0004 | -0.0025 | [-0.0083, +0.0031] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.0845 / -0.1043 | -0.0944 | [-0.1734, -0.0137] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0115 / -0.0281 | -0.0198 | [-0.0745, +0.0436] | 0.0229 | OPEN |
| ade6_m|bulk | -0.1122 / -0.1211 | -0.1167 | [-0.2244, -0.0270] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.0574 / -0.0705 | -0.0640 | [-0.1171, -0.0095] | 0.0442 | BETTER |
| ade_tac_m|all | -0.0845 / -0.1043 | -0.0944 | [-0.1734, -0.0137] | 0.0699 | BETTER |

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

## I10RQ50c minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0040 / -0.0343 | -0.0151 | [-0.0478, +0.0246] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0290 / -0.0996 | -0.0643 | [-0.1305, +0.0192] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0041 / -0.0076 | -0.0059 | [-0.0160, +0.0039] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.0012 / -0.2061 | -0.1037 | [-0.2898, +0.0921] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0032 / -0.0056 | -0.0044 | [-0.0101, +0.0008] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2284 / -0.8140 | -0.5212 | [-1.1876, +0.1481] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0029 / -0.0046 | -0.0038 | [-0.0090, +0.0010] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0004 / -0.0067 | -0.0035 | [-0.0102, +0.0031] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0021 / -0.0029 | -0.0025 | [-0.0065, +0.0017] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.0376 / -0.1432 | -0.0904 | [-0.1884, +0.0286] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0121 / -0.1004 | -0.0562 | [-0.1396, +0.0416] | 0.0229 | OPEN |
| ade6_m|bulk | -0.0441 / -0.0952 | -0.0696 | [-0.1554, +0.0187] | 0.1243 | OPEN |
| ade6_m|all | -0.0267 / -0.0980 | -0.0623 | [-0.1293, +0.0183] | 0.0442 | OPEN |
| ade_tac_m|all | -0.0376 / -0.1432 | -0.0904 | [-0.1884, +0.0286] | 0.0699 | OPEN |

## I10RQ50c minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0620 / +0.0222 | +0.0421 | [-0.0509, +0.1368] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3044 / +0.2348 | +0.2696 | [+0.0962, +0.4617] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0443 / -0.0316 | -0.0379 | [-0.0609, -0.0160] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.6399 / +0.3746 | +0.5073 | [+0.2619, +0.7951] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0227 / +0.0179 | +0.0203 | [+0.0120, +0.0295] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.3033 / -0.3381 | +0.4826 | [-1.1576, +1.9202] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0083 / -0.0046 | -0.0065 | [-0.0167, +0.0027] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0096 / +0.0096 | +0.0096 | [-0.0015, +0.0213] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0042 / +0.0033 | +0.0038 | [-0.0035, +0.0113] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3511 / +0.2656 | +0.3084 | [+0.0827, +0.5569] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2238 / +0.0725 | +0.1482 | [-0.0176, +0.3583] | 0.0229 | OPEN |
| ade6_m|bulk | +0.3038 / +0.3430 | +0.3234 | [+0.1382, +0.5336] | 0.1243 | WORSE |
| ade6_m|all | +0.2603 / +0.1959 | +0.2281 | [+0.0655, +0.4081] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3511 / +0.2656 | +0.3084 | [+0.0827, +0.5569] | 0.0699 | WORSE |

## I10RLb minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0770 / +0.0673 | +0.0721 | [-0.0126, +0.1691] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3062 / +0.2880 | +0.2971 | [+0.1038, +0.5317] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0228 / -0.0230 | -0.0229 | [-0.0427, -0.0023] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5849 / +0.4145 | +0.4997 | [+0.2736, +0.7683] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0108 / +0.0167 | +0.0137 | [+0.0069, +0.0229] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.5804 / +1.0971 | +1.3387 | [+0.3406, +2.4109] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0000 / +0.0050 | +0.0025 | [-0.0073, +0.0113] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0067 / +0.0092 | +0.0079 | [-0.0040, +0.0210] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0058 / +0.0079 | +0.0069 | [-0.0006, +0.0154] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3762 / +0.3566 | +0.3664 | [+0.1133, +0.6690] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1406 / +0.1333 | +0.1370 | [+0.0134, +0.2967] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4304 / +0.4101 | +0.4202 | [+0.1435, +0.7461] | 0.1243 | WORSE |
| ade6_m|all | +0.2728 / +0.2595 | +0.2661 | [+0.0837, +0.4851] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3762 / +0.3566 | +0.3664 | [+0.1133, +0.6690] | 0.0699 | WORSE |

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
| I10RLb_s0 | 3.0076 | 3.2533 | 1.6737 | 1.2717 | 0.6528 |
| I10RLb_s1 | 2.9501 | 3.2689 | 1.7153 | 1.1656 | 0.6794 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50c_s0 | 2.9951 | 3.3365 | 1.6588 | 1.2502 | 0.6511 |
| I10RQ50c_s1 | 2.8865 | 3.2081 | 1.6701 | 1.1571 | 0.6749 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
