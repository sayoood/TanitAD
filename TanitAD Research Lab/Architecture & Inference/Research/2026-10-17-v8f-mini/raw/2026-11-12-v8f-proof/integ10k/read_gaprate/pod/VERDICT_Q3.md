# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RQ50d"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RQ50d minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0365 / +0.0219 | +0.0292 | [+0.0100, +0.0506] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0755 / +0.0376 | +0.0566 | [+0.0140, +0.1126] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0183 / -0.0027 | -0.0105 | [-0.0243, +0.0033] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0760 / -0.0428 | +0.0166 | [-0.0983, +0.1817] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0096 / -0.0005 | +0.0045 | [-0.0033, +0.0210] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.4192 / +0.2336 | +0.3264 | [-0.1376, +0.9487] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0058 / -0.0008 | +0.0025 | [-0.0040, +0.0090] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0029 / +0.0025 | -0.0002 | [-0.0061, +0.0054] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0029 / +0.0038 | +0.0033 | [-0.0010, +0.0075] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0922 / +0.0496 | +0.0709 | [+0.0135, +0.1437] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0628 / +0.0112 | +0.0370 | [-0.0109, +0.0994] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0658 / +0.0652 | +0.0655 | [+0.0105, +0.1358] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0641 / +0.0358 | +0.0500 | [+0.0109, +0.0994] | 0.0442 | WORSE |
| ade_tac_m|all | +0.0922 / +0.0496 | +0.0709 | [+0.0135, +0.1437] | 0.0699 | WORSE |

## I10RQ50x minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0173 / +0.0012 | +0.0092 | [-0.0405, +0.0569] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0221 / -0.0078 | +0.0072 | [-0.0952, +0.1059] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0080 / -0.0149 | -0.0114 | [-0.0251, +0.0018] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.0290 / -0.1465 | -0.0878 | [-0.2822, +0.0873] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0057 / +0.0003 | -0.0027 | [-0.0094, +0.0037] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2680 / -0.0389 | -0.1535 | [-0.8589, +0.5188] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0017 / -0.0029 | -0.0006 | [-0.0083, +0.0065] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0038 / -0.0083 | -0.0060 | [-0.0146, +0.0021] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0038 / -0.0050 | -0.0006 | [-0.0090, +0.0073] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0294 / -0.0169 | +0.0062 | [-0.1331, +0.1380] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0479 / +0.0513 | +0.0017 | [-0.1273, +0.1373] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0861 / -0.0717 | +0.0072 | [-0.1578, +0.1665] | 0.1243 | OPEN |
| ade6_m|all | +0.0132 / -0.0048 | +0.0042 | [-0.0924, +0.0990] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0294 / -0.0169 | +0.0062 | [-0.1331, +0.1380] | 0.0699 | OPEN |

## I10RQ50d minus I10RQ50c

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0325 / +0.0562 | +0.0444 | [+0.0124, +0.0784] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1044 / +0.1373 | +0.1208 | [+0.0548, +0.1929] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0142 / +0.0049 | -0.0047 | [-0.0217, +0.0123] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0773 / +0.1567 | +0.1170 | [-0.0843, +0.3197] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0128 / +0.0021 | +0.0075 | [-0.0023, +0.0259] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.6476 / +0.9598 | +0.8037 | [+0.0160, +1.6183] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0088 / +0.0038 | +0.0063 | [+0.0004, +0.0119] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0025 / +0.0092 | +0.0033 | [-0.0069, +0.0129] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0050 / +0.0067 | +0.0058 | [+0.0006, +0.0108] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1299 / +0.1928 | +0.1613 | [+0.0618, +0.2679] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0749 / +0.1116 | +0.0932 | [+0.0049, +0.1744] | 0.0229 | WORSE |
| ade6_m|bulk | +0.1099 / +0.1604 | +0.1351 | [+0.0338, +0.2582] | 0.1243 | WORSE |
| ade6_m|all | +0.0908 / +0.1338 | +0.1123 | [+0.0439, +0.1854] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1299 / +0.1928 | +0.1613 | [+0.0618, +0.2679] | 0.0699 | WORSE |

## I10RQ50d minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0945 / +0.0784 | +0.0864 | [-0.0114, +0.1852] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.4088 / +0.3720 | +0.3904 | [+0.2043, +0.6015] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0585 / -0.0267 | -0.0426 | [-0.0750, -0.0097] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.6429 / +0.4780 | +0.5605 | [+0.3055, +0.8873] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0292 / +0.0159 | +0.0226 | [+0.0109, +0.0443] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.3065 / +0.9783 | +1.6424 | [-0.0541, +3.3668] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0004 / -0.0008 | -0.0002 | [-0.0106, +0.0102] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0071 / +0.0188 | +0.0129 | [-0.0021, +0.0281] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0092 / +0.0100 | +0.0096 | [+0.0004, +0.0192] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.4810 / +0.4584 | +0.4697 | [+0.2254, +0.7429] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2987 / +0.1841 | +0.2414 | [+0.0743, +0.4614] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4137 / +0.5034 | +0.4586 | [+0.2166, +0.7357] | 0.1243 | WORSE |
| ade6_m|all | +0.3511 / +0.3297 | +0.3404 | [+0.1660, +0.5382] | 0.0442 | WORSE |
| ade_tac_m|all | +0.4810 / +0.4584 | +0.4697 | [+0.2254, +0.7429] | 0.0699 | WORSE |

## I10RQ50d minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0402 / +0.0326 | +0.0364 | [-0.0150, +0.0872] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1565 / +0.1555 | +0.1560 | [+0.0785, +0.2489] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0313 / -0.0018 | -0.0165 | [-0.0462, +0.0127] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.2732 / +0.2055 | +0.2393 | [-0.0041, +0.5444] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0196 / +0.0063 | +0.0129 | [+0.0020, +0.0334] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.0957 / +0.3980 | +0.7468 | [-0.6168, +2.1537] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0038 / -0.0038 | +0.0000 | [-0.0079, +0.0083] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / +0.0121 | +0.0069 | [-0.0035, +0.0177] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0079 / +0.0025 | +0.0052 | [-0.0012, +0.0121] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1893 / +0.2060 | +0.1976 | [+0.0834, +0.3191] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1696 / +0.0789 | +0.1242 | [+0.0168, +0.2925] | 0.0229 | WORSE |
| ade6_m|bulk | +0.0955 / +0.2144 | +0.1550 | [+0.0269, +0.2922] | 0.1243 | WORSE |
| ade6_m|all | +0.1358 / +0.1407 | +0.1383 | [+0.0581, +0.2273] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1893 / +0.2060 | +0.1976 | [+0.0834, +0.3191] | 0.0699 | WORSE |

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
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50c_s0 | 2.9951 | 3.3365 | 1.6588 | 1.2502 | 0.6511 |
| I10RQ50c_s1 | 2.8865 | 3.2081 | 1.6701 | 1.1571 | 0.6749 |
| I10RQ50d_s0 | 3.0860 | 3.4114 | 1.6913 | 1.2360 | 0.6561 |
| I10RQ50d_s1 | 3.0203 | 3.3197 | 1.7263 | 1.1619 | 0.6815 |
| I10RQ50x_s0 | 3.0350 | 3.3007 | 1.6720 | 1.2463 | 0.6569 |
| I10RQ50x_s1 | 2.9797 | 3.3598 | 1.7057 | 1.1497 | 0.6728 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
