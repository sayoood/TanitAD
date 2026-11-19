# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RLcB8"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867476, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RLcB8 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0549 / +0.0462 | +0.0506 | [+0.0192, +0.0861] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0877 / +0.0967 | +0.0922 | [+0.0160, +0.1822] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0109 / +0.0019 | -0.0045 | [-0.0182, +0.0074] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.1402 / +0.0915 | +0.1158 | [+0.0174, +0.2316] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0005 / +0.0028 | +0.0011 | [-0.0021, +0.0043] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.8916 / +0.6221 | +0.7568 | [+0.3185, +1.3764] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0000 / -0.0029 | -0.0015 | [-0.0052, +0.0027] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0004 / +0.0008 | +0.0006 | [-0.0038, +0.0054] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0017 / -0.0017 | +0.0000 | [-0.0035, +0.0042] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1235 / +0.1402 | +0.1318 | [+0.0223, +0.2641] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0400 / +0.0664 | +0.0532 | [+0.0019, +0.1147] | 0.0229 | WORSE |
| ade6_m|bulk | +0.1376 / +0.1305 | +0.1341 | [+0.0221, +0.2728] | 0.1243 | WORSE |
| ade6_m|all | +0.0845 / +0.0956 | +0.0901 | [+0.0157, +0.1785] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1235 / +0.1402 | +0.1318 | [+0.0223, +0.2641] | 0.0699 | WORSE |

## I10RQ50B8 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0716 / +0.0315 | +0.0515 | [+0.0161, +0.0960] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.1153 / +0.0197 | +0.0675 | [-0.0073, +0.1539] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0142 / -0.0013 | -0.0077 | [-0.0199, +0.0043] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.2910 / +0.0631 | +0.1770 | [-0.0077, +0.3756] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0095 / +0.0019 | +0.0057 | [+0.0001, +0.0125] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.9668 / +0.5384 | +1.2526 | [+0.2296, +2.4901] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0008 / -0.0025 | -0.0017 | [-0.0062, +0.0031] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0033 / -0.0008 | +0.0013 | [-0.0048, +0.0073] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0008 / +0.0008 | +0.0008 | [-0.0025, +0.0042] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1544 / +0.0263 | +0.0903 | [-0.0098, +0.2082] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0643 / -0.0377 | +0.0133 | [-0.0555, +0.1098] | 0.0229 | OPEN |
| ade6_m|bulk | +0.1541 / +0.0878 | +0.1210 | [+0.0275, +0.2211] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.1053 / +0.0195 | +0.0624 | [-0.0049, +0.1411] | 0.0442 | OPEN |
| ade_tac_m|all | +0.1544 / +0.0263 | +0.0903 | [-0.0098, +0.2082] | 0.0699 | OPEN |

## I10RLcB10 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0515 / +0.0484 | +0.0499 | [+0.0164, +0.0899] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0181 / +0.0531 | +0.0356 | [-0.0265, +0.1111] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0029 / +0.0120 | +0.0045 | [-0.0070, +0.0196] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.2346 / +0.2101 | +0.2224 | [+0.0840, +0.3803] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0024 / +0.0038 | +0.0031 | [-0.0007, +0.0066] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.3173 / +1.1704 | +1.2438 | [+0.5276, +2.0752] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0000 / -0.0004 | -0.0002 | [-0.0052, +0.0048] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0029 / +0.0042 | +0.0035 | [-0.0010, +0.0085] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0008 / -0.0029 | -0.0010 | [-0.0048, +0.0033] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0274 / +0.0813 | +0.0544 | [-0.0359, +0.1664] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0347 / +0.0701 | +0.0524 | [-0.0089, +0.1316] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0071 / +0.0439 | +0.0255 | [-0.0491, +0.1103] | 0.1243 | OPEN |
| ade6_m|all | +0.0221 / +0.0581 | +0.0401 | [-0.0193, +0.1158] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0274 / +0.0813 | +0.0544 | [-0.0359, +0.1664] | 0.0699 | OPEN |

## I10RLcB10 minus I10RLcB8

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0034 / +0.0021 | -0.0006 | [-0.0257, +0.0214] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0696 / -0.0437 | -0.0566 | [-0.1608, +0.0171] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.0080 / +0.0100 | +0.0090 | [-0.0006, +0.0209] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0945 / +0.1186 | +0.1065 | [-0.0044, +0.2389] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0029 / +0.0010 | +0.0020 | [-0.0009, +0.0052] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.4257 / +0.5483 | +0.4870 | [-0.1131, +1.2201] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0000 / +0.0025 | +0.0013 | [-0.0040, +0.0063] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0025 / +0.0033 | +0.0029 | [-0.0010, +0.0073] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0008 / -0.0013 | -0.0010 | [-0.0044, +0.0023] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.0961 / -0.0588 | -0.0775 | [-0.2285, +0.0274] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0053 / +0.0037 | -0.0008 | [-0.0577, +0.0462] | 0.0229 | OPEN |
| ade6_m|bulk | -0.1305 / -0.0866 | -0.1086 | [-0.2677, +0.0078] | 0.1243 | OPEN |
| ade6_m|all | -0.0624 / -0.0375 | -0.0499 | [-0.1500, +0.0189] | 0.0442 | OPEN |
| ade_tac_m|all | -0.0961 / -0.0588 | -0.0775 | [-0.2285, +0.0274] | 0.0699 | OPEN |

## I10RLcB8 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1092 / +0.0920 | +0.1006 | [+0.0057, +0.2125] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3400 / +0.3132 | +0.3266 | [+0.1000, +0.6067] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0381 / -0.0230 | -0.0305 | [-0.0513, -0.0118] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5448 / +0.3623 | +0.4535 | [+0.2195, +0.7187] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0154 / +0.0175 | +0.0165 | [+0.0096, +0.0244] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.9726 / +0.7632 | +1.3679 | [+0.1265, +2.7369] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0033 / +0.0000 | -0.0017 | [-0.0106, +0.0067] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0058 / +0.0075 | +0.0067 | [-0.0042, +0.0186] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0029 / +0.0058 | +0.0044 | [-0.0035, +0.0129] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.4152 / +0.3926 | +0.4039 | [+0.1003, +0.7698] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1691 / +0.1716 | +0.1703 | [+0.0021, +0.3896] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4559 / +0.4194 | +0.4376 | [+0.1511, +0.7937] | 0.1243 | WORSE |
| ade6_m|all | +0.2999 / +0.2846 | +0.2922 | [+0.0741, +0.5572] | 0.0442 | WORSE |
| ade_tac_m|all | +0.4152 / +0.3926 | +0.4039 | [+0.1003, +0.7698] | 0.0699 | WORSE |

## I10RQ50B8 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1295 / +0.0880 | +0.1088 | [-0.0037, +0.2275] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.4487 / +0.3541 | +0.4014 | [+0.1830, +0.6507] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0543 / -0.0252 | -0.0398 | [-0.0715, -0.0068] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.8579 / +0.5968 | +0.7273 | [+0.4187, +1.0986] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0293 / +0.0179 | +0.0236 | [+0.0130, +0.0362] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.8461 / +1.3028 | +2.5744 | [+0.3104, +4.9950] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0063 / -0.0025 | -0.0044 | [-0.0152, +0.0058] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0133 / +0.0154 | +0.0144 | [+0.0013, +0.0282] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0071 / +0.0071 | +0.0071 | [-0.0013, +0.0158] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.5432 / +0.4351 | +0.4891 | [+0.2043, +0.8100] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3002 / +0.1352 | +0.2177 | [+0.0406, +0.4694] | 0.0229 | WORSE |
| ade6_m|bulk | +0.5021 / +0.5260 | +0.5141 | [+0.2319, +0.8341] | 0.1243 | WORSE |
| ade6_m|all | +0.3922 / +0.3134 | +0.3528 | [+0.1496, +0.5853] | 0.0442 | WORSE |
| ade_tac_m|all | +0.5432 / +0.4351 | +0.4891 | [+0.2043, +0.8100] | 0.0699 | WORSE |

## I10RQ50B8 minus I10RLcB8

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0203 / -0.0040 | +0.0081 | [-0.0495, +0.0667] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1086 / +0.0409 | +0.0747 | [-0.0421, +0.2122] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0163 / -0.0023 | -0.0093 | [-0.0341, +0.0142] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.3475 / +0.2448 | +0.2962 | [+0.0517, +0.5668] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0205 / +0.0070 | +0.0137 | [+0.0032, +0.0265] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.7476 / +0.1731 | +0.9603 | [-0.6728, +2.5743] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0029 / -0.0025 | -0.0027 | [-0.0115, +0.0052] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0075 / +0.0079 | +0.0077 | [-0.0013, +0.0169] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0042 / +0.0013 | +0.0027 | [-0.0044, +0.0094] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1280 / +0.0425 | +0.0853 | [-0.0723, +0.2704] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1311 / -0.0363 | +0.0474 | [-0.0952, +0.2555] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0462 / +0.1066 | +0.0764 | [-0.0983, +0.2422] | 0.1243 | OPEN |
| ade6_m|all | +0.0924 / +0.0288 | +0.0606 | [-0.0506, +0.1924] | 0.0442 | OPEN |
| ade_tac_m|all | +0.1280 / +0.0425 | +0.0853 | [-0.0723, +0.2704] | 0.0699 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcB10_s0 | 2.9723 | 3.2765 | 1.7025 | 1.2644 | 0.6490 |
| I10RLcB10_s1 | 2.9377 | 3.3109 | 1.7421 | 1.1757 | 0.6761 |
| I10RLcB8_s0 | 3.0347 | 3.2818 | 1.7060 | 1.2563 | 0.6499 |
| I10RLcB8_s1 | 2.9752 | 3.3072 | 1.7400 | 1.1656 | 0.6774 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50B8_s0 | 3.1271 | 3.4129 | 1.7263 | 1.2401 | 0.6540 |
| I10RQ50B8_s1 | 3.0040 | 3.2708 | 1.7360 | 1.1634 | 0.6786 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
