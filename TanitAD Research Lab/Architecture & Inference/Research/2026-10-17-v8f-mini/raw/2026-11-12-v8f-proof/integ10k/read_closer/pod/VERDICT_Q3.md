# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RQ50h"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RQ50h minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0312 / +0.0019 | +0.0165 | [-0.0125, +0.0448] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0258 / -0.0046 | +0.0106 | [-0.0387, +0.0611] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0034 / +0.0053 | +0.0010 | [-0.0128, +0.0133] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0920 / -0.1220 | -0.0150 | [-0.2176, +0.2123] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0055 / +0.0014 | -0.0020 | [-0.0091, +0.0066] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.0523 / +0.2421 | +0.0949 | [-0.4753, +0.6922] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0025 / -0.0021 | +0.0002 | [-0.0058, +0.0067] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0075 / -0.0013 | +0.0031 | [-0.0050, +0.0131] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0017 / +0.0013 | +0.0015 | [-0.0023, +0.0056] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0391 / -0.0050 | +0.0170 | [-0.0559, +0.0907] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0051 / -0.0162 | -0.0056 | [-0.0565, +0.0462] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0493 / +0.0113 | +0.0303 | [-0.0453, +0.1150] | 0.1243 | OPEN |
| ade6_m|all | +0.0252 / -0.0037 | +0.0108 | [-0.0380, +0.0606] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0391 / -0.0050 | +0.0170 | [-0.0559, +0.0907] | 0.0699 | OPEN |

## I10RQ50h minus I10RQ50c

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0272 / +0.0362 | +0.0317 | [+0.0040, +0.0589] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0548 / +0.0951 | +0.0749 | [-0.0017, +0.1414] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.0007 / +0.0129 | +0.0068 | [-0.0063, +0.0216] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0442 / +0.0762 | +0.0602 | [-0.0786, +0.2000] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0003 / +0.0061 | +0.0032 | [-0.0037, +0.0108] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.2125 / +1.1000 | +0.6563 | [-0.2299, +1.4754] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0054 / +0.0025 | +0.0040 | [-0.0025, +0.0108] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0079 / +0.0054 | +0.0067 | [+0.0004, +0.0142] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0038 / +0.0042 | +0.0040 | [-0.0013, +0.0096] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0767 / +0.1382 | +0.1074 | [-0.0028, +0.2050] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0172 / +0.0842 | +0.0507 | [-0.0478, +0.1453] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0934 / +0.1065 | +0.0999 | [+0.0190, +0.1836] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0519 / +0.0943 | +0.0731 | [-0.0021, +0.1394] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0767 / +0.1382 | +0.1074 | [-0.0028, +0.2050] | 0.0699 | OPEN |

## I10RQ50h minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0892 / +0.0584 | +0.0738 | [-0.0193, +0.1666] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3592 / +0.3298 | +0.3445 | [+0.1755, +0.5484] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0436 / -0.0187 | -0.0311 | [-0.0578, -0.0017] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.8125 / +0.4472 | +0.6299 | [+0.3141, +1.0398] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0281 / +0.0202 | +0.0242 | [+0.0128, +0.0380] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.3997 / +0.6785 | +1.0391 | [-0.3336, +2.2741] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0029 / -0.0021 | -0.0025 | [-0.0117, +0.0065] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0175 / +0.0150 | +0.0163 | [+0.0048, +0.0282] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0079 / +0.0075 | +0.0077 | [-0.0004, +0.0163] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.4278 / +0.4037 | +0.4158 | [+0.1907, +0.6810] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2410 / +0.1567 | +0.1988 | [+0.0432, +0.3859] | 0.0229 | WORSE |
| ade6_m|bulk | +0.3972 / +0.4495 | +0.4234 | [+0.2015, +0.6759] | 0.1243 | WORSE |
| ade6_m|all | +0.3122 / +0.2902 | +0.3012 | [+0.1400, +0.4927] | 0.0442 | WORSE |
| ade_tac_m|all | +0.4278 / +0.4037 | +0.4158 | [+0.1907, +0.6810] | 0.0699 | WORSE |

## I10RQ50h minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0349 / +0.0125 | +0.0237 | [-0.0288, +0.0764] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1068 / +0.1133 | +0.1101 | [+0.0349, +0.1919] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0164 / +0.0063 | -0.0050 | [-0.0312, +0.0214] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.2583 / +0.1916 | +0.2250 | [+0.0196, +0.4502] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0070 / +0.0101 | +0.0085 | [+0.0011, +0.0179] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.5364 / +0.4745 | +0.5055 | [-0.5621, +1.5034] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0004 / -0.0050 | -0.0023 | [-0.0106, +0.0058] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0121 / +0.0083 | +0.0102 | [+0.0008, +0.0202] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0067 / +0.0000 | +0.0033 | [-0.0052, +0.0110] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1361 / +0.1514 | +0.1437 | [+0.0398, +0.2525] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1119 / +0.0515 | +0.0817 | [-0.0123, +0.1992] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0790 / +0.1605 | +0.1198 | [-0.0208, +0.2649] | 0.1243 | OPEN |
| ade6_m|all | +0.0969 / +0.1012 | +0.0991 | [+0.0264, +0.1756] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1361 / +0.1514 | +0.1437 | [+0.0398, +0.2525] | 0.0699 | WORSE |

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
| I10RQ50h_s0 | 3.0471 | 3.3537 | 1.6860 | 1.2509 | 0.6549 |
| I10RQ50h_s1 | 2.9808 | 3.2923 | 1.7063 | 1.1700 | 0.6790 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
