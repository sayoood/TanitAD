# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RH_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.44464669811771973, "D_read": 0.4446466981177198, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10UEA_R0C10XHb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RH R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1227 | -0.1227 | [-0.2236, -0.0448] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0944 | -0.0944 | [-0.2180, +0.0039] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0081 | +0.0081 | [-0.0109, +0.0280] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | -0.8315 | -0.8315 | [-1.0430, -0.6325] | 0.6001 (0.6001) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0902 | -0.0902 | [-0.1327, -0.0576] | 0.0202 (0.0202) | BETTER |
| LATERAL:yaw_rate_mae_degps | -4.3722 | -4.3722 | [-5.8835, -3.0640] | 1.4612 (1.4612) | BETTER |
| TACTICAL:lon_err_rate | -0.0588 | -0.0588 | [-0.0946, -0.0279] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0488 | -0.0488 | [-0.0688, -0.0304] | 0.0117 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0729 | -0.0729 | [-0.1096, -0.0430] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1422 | -0.1422 | [-0.3253, +0.0015] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0631 | +0.0631 | [-0.0239, +0.1527] | 0.0425 | OPEN |
| ade6_m|bulk | -0.2702 | -0.2702 | [-0.5006, -0.1099] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | -0.0889 | -0.0889 | [-0.2118, +0.0089] | 0.2029 | OPEN |
| ade_tac_m|all | -0.1422 | -0.1422 | [-0.3253, +0.0015] | 0.3008 | OPEN |

## I10RH R0C10XHb20 minus I10RH R0KHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1569 | +0.1569 | [+0.1053, +0.2071] | 0.1482 (0.1482) | WORSE |
| LONGITUDINAL:along_mae_m | +0.2013 | +0.2013 | [+0.1277, +0.2773] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0032 | +0.0032 | [-0.0116, +0.0180] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | +0.6957 | +0.6957 | [+0.4690, +0.9631] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0839 | +0.0839 | [+0.0533, +0.1240] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.3892 | +7.3892 | [+5.4120, +9.7764] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.1000 | +0.1000 | [+0.0608, +0.1408] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0192 | +0.0192 | [+0.0071, +0.0333] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0975 | +0.0975 | [+0.0583, +0.1351] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.2719 | +0.2719 | [+0.1676, +0.3782] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0437 | +0.0437 | [-0.0364, +0.1235] | 0.0425 | OPEN |
| ade6_m|bulk | +0.3454 | +0.3454 | [+0.2501, +0.4524] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.1813 | +0.1813 | [+0.1117, +0.2522] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2719 | +0.2719 | [+0.1676, +0.3782] | 0.3008 | INSIDE-SPREAD |

## I10RH R0G50Hb20 minus I10RH R0G0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0472 | +0.0472 | [+0.0187, +0.0743] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0616 | +0.0616 | [+0.0204, +0.1025] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0302 | -0.0302 | [-0.0488, -0.0138] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1887 | +0.1887 | [-0.0065, +0.5062] | 0.6001 (0.6001) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0321 | +0.0321 | [+0.0216, +0.0431] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +3.5270 | +3.5270 | [+2.6149, +4.4766] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0525 | +0.0525 | [+0.0246, +0.0805] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0438 | -0.0438 | [-0.0650, -0.0262] | 0.0117 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | +0.0154 | +0.0154 | [-0.0058, +0.0363] | 0.1021 (0.1021) | OPEN |
| TACTICAL:ade_tac_m | +0.0429 | +0.0429 | [-0.0162, +0.1024] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0285 | -0.0285 | [-0.0775, +0.0206] | 0.0425 | OPEN |
| ade6_m|bulk | +0.0967 | +0.0967 | [+0.0420, +0.1531] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0286 | +0.0286 | [-0.0108, +0.0683] | 0.2029 | OPEN |
| ade_tac_m|all | +0.0429 | +0.0429 | [-0.0162, +0.1024] | 0.3008 | OPEN |

## I10RH R0C10Hb20 minus I10RH R0G50Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1129 | +0.1129 | [+0.0844, +0.1447] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1127 | +0.1127 | [+0.0810, +0.1443] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0245 | +0.0245 | [+0.0088, +0.0426] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4952 | +0.4952 | [+0.2972, +0.7044] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0492 | +0.0492 | [+0.0285, +0.0769] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +5.8945 | +5.8945 | [+4.4301, +7.6528] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0521 | +0.0521 | [+0.0271, +0.0767] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0146 | +0.0146 | [+0.0004, +0.0292] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0604 | +0.0604 | [+0.0358, +0.0838] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1769 | +0.1769 | [+0.1273, +0.2245] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0895 | +0.0895 | [+0.0440, +0.1329] | 0.0425 | WORSE |
| ade6_m|bulk | +0.1519 | +0.1519 | [+0.1122, +0.1921] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.1180 | +0.1180 | [+0.0849, +0.1496] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.1769 | +0.1769 | [+0.1273, +0.2245] | 0.3008 | INSIDE-SPREAD |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2943 | +0.2943 | [+0.1911, +0.4196] | 0.1482 (0.1482) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3506 | +0.3506 | [+0.2180, +0.5093] | 0.2027 (0.2027) | WORSE |
| LATERAL:cross_mae_m | -0.0062 | -0.0062 | [-0.0256, +0.0127] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | +1.2360 | +1.2360 | [+0.9217, +1.6120] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1549 | +0.1549 | [+0.1029, +0.2238] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +10.7292 | +10.7292 | [+8.0219, +13.9126] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.1371 | +0.1371 | [+0.0871, +0.1876] | 0.1071 (0.1071) | WORSE |
| TACTICAL:lat_err_rate | +0.0534 | +0.0534 | [+0.0305, +0.0775] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.1505 | +0.1505 | [+0.1017, +0.1996] | 0.1021 (0.1021) | WORSE |
| TACTICAL:ade_tac_m | +0.4809 | +0.4809 | [+0.2870, +0.7102] | 0.3008 (0.3008) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0602 | +0.0602 | [-0.0515, +0.1701] | 0.0425 | OPEN |
| ade6_m|bulk | +0.6312 | +0.6312 | [+0.4216, +0.9180] | 0.3943 | WORSE |
| ade6_m|all | +0.3206 | +0.3206 | [+0.1913, +0.4735] | 0.2029 | WORSE |
| ade_tac_m|all | +0.4809 | +0.4809 | [+0.2870, +0.7102] | 0.3008 | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0892 | +0.0892 | [+0.0432, +0.1411] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0978 | +0.0978 | [+0.0389, +0.1637] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0383 | -0.0383 | [-0.0620, -0.0196] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.3137 | +0.3137 | [+0.1267, +0.5687] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0565 | +0.0565 | [+0.0384, +0.0761] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.8143 | +4.8143 | [+3.5674, +6.1603] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0846 | +0.0846 | [+0.0471, +0.1246] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0096 | -0.0096 | [-0.0296, +0.0088] | 0.0117 (0.0117) | OPEN |
| TACTICAL:joint_err_rate | +0.0358 | +0.0358 | [+0.0121, +0.0592] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0683 | +0.0683 | [+0.0011, +0.1335] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0446 | -0.0446 | [-0.1018, +0.0132] | 0.0425 | OPEN |
| ade6_m|bulk | +0.1531 | +0.1531 | [+0.0991, +0.2126] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0455 | +0.0455 | [+0.0007, +0.0890] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0683 | +0.0683 | [+0.0011, +0.1335] | 0.3008 | INSIDE-SPREAD |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2630 | +0.2630 | [+0.1916, +0.3482] | 0.1482 (0.1482) | WORSE |
| LONGITUDINAL:along_mae_m | +0.2796 | +0.2796 | [+0.1909, +0.3893] | 0.2027 (0.2027) | WORSE |
| LATERAL:cross_mae_m | +0.0104 | +0.0104 | [-0.0115, +0.0336] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | +1.3001 | +1.3001 | [+0.9604, +1.6578] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1021 | +0.1021 | [+0.0669, +0.1470] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +9.5482 | +9.5482 | [+7.2337, +11.9831] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0875 | +0.0875 | [+0.0584, +0.1184] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0400 | +0.0400 | [+0.0237, +0.0579] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0855 | +0.0855 | [+0.0601, +0.1126] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3535 | +0.3535 | [+0.2450, +0.4963] | 0.3008 (0.3008) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1284 | +0.1284 | [+0.0642, +0.1990] | 0.0425 | WORSE |
| ade6_m|bulk | +0.3636 | +0.3636 | [+0.2459, +0.5455] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.2357 | +0.2357 | [+0.1634, +0.3309] | 0.2029 | WORSE |
| ade_tac_m|all | +0.3535 | +0.3535 | [+0.2450, +0.4963] | 0.3008 | WORSE |

## I10RH R0C10XHb20 minus I10B0 R0KHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1716 | +0.1716 | [+0.1230, +0.2210] | 0.1482 (0.1482) | WORSE |
| LONGITUDINAL:along_mae_m | +0.2561 | +0.2561 | [+0.1833, +0.3309] | 0.2027 (0.2027) | WORSE |
| LATERAL:cross_mae_m | +0.0019 | +0.0019 | [-0.0231, +0.0263] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | +0.3623 | +0.3623 | [+0.1345, +0.6202] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0636 | +0.0636 | [+0.0374, +0.0993] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.3276 | +6.3276 | [+4.5577, +8.4911] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0784 | +0.0784 | [+0.0400, +0.1184] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0046 | +0.0046 | [-0.0096, +0.0200] | 0.0117 (0.0117) | OPEN |
| TACTICAL:joint_err_rate | +0.0775 | +0.0775 | [+0.0400, +0.1147] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3386 | +0.3386 | [+0.2332, +0.4436] | 0.3008 (0.3008) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1232 | +0.1232 | [+0.0452, +0.1974] | 0.0425 | WORSE |
| ade6_m|bulk | +0.3610 | +0.3610 | [+0.2619, +0.4737] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.2317 | +0.2317 | [+0.1614, +0.3020] | 0.2029 | WORSE |
| ade_tac_m|all | +0.3386 | +0.3386 | [+0.2332, +0.4436] | 0.3008 | WORSE |

## I10RH R0Hb20 minus I10B0 R0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0244 | -0.0244 | [-0.0733, +0.0164] | 0.1482 (0.1482) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0204 | +0.0204 | [-0.0430, +0.0805] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0023 | +0.0023 | [-0.0203, +0.0247] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | -0.5634 | -0.5634 | [-0.7706, -0.3652] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0528 | -0.0528 | [-0.0704, -0.0374] | 0.0202 (0.0202) | BETTER |
| LATERAL:yaw_rate_mae_degps | -2.7276 | -2.7276 | [-3.6508, -1.9231] | 1.4612 (1.4612) | BETTER |
| TACTICAL:lon_err_rate | -0.0283 | -0.0283 | [-0.0608, -0.0004] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0579 | -0.0579 | [-0.0780, -0.0396] | 0.0117 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0338 | -0.0338 | [-0.0601, -0.0096] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0293 | +0.0293 | [-0.0487, +0.1077] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1209 | +0.1209 | [+0.0402, +0.2053] | 0.0425 | WORSE |
| ade6_m|bulk | -0.0884 | -0.0884 | [-0.1337, -0.0428] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0254 | +0.0254 | [-0.0280, +0.0788] | 0.2029 | OPEN |
| ade_tac_m|all | +0.0293 | +0.0293 | [-0.0487, +0.1077] | 0.3008 | OPEN |

## I10RH R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1795 | -0.1795 | [-0.3033, -0.0811] | 0.1482 (0.1482) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1546 | -0.1546 | [-0.3030, -0.0349] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0058 | -0.0058 | [-0.0301, +0.0208] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | -1.2264 | -1.2264 | [-1.5261, -0.9237] | 0.6001 (0.6001) | BETTER |
| LATERAL:curvature_mae_1pm | -0.1100 | -0.1100 | [-0.1598, -0.0702] | 0.0202 (0.0202) | BETTER |
| LATERAL:yaw_rate_mae_degps | -6.2438 | -6.2438 | [-8.2100, -4.4885] | 1.4612 (1.4612) | BETTER |
| TACTICAL:lon_err_rate | -0.0792 | -0.0792 | [-0.1167, -0.0467] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0975 | -0.0975 | [-0.1284, -0.0670] | 0.0117 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0754 | -0.0754 | [-0.1059, -0.0471] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1738 | -0.1738 | [-0.3531, -0.0302] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0603 | +0.0603 | [-0.0340, +0.1551] | 0.0425 | OPEN |
| ade6_m|bulk | -0.3130 | -0.3130 | [-0.5330, -0.1650] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | -0.1099 | -0.1099 | [-0.2302, -0.0136] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | -0.1738 | -0.1738 | [-0.3531, -0.0302] | 0.3008 | INSIDE-SPREAD |

## I10RH R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0293 | -0.0293 | [-0.0726, +0.0060] | 0.1482 (0.1482) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0124 | +0.0124 | [-0.0497, +0.0672] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | -0.0200 | -0.0200 | [-0.0432, +0.0024] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | -0.3824 | -0.3824 | [-0.5406, -0.2132] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0430 | -0.0430 | [-0.0587, -0.0289] | 0.0202 (0.0202) | BETTER |
| LATERAL:yaw_rate_mae_degps | -2.1032 | -2.1032 | [-2.8134, -1.4527] | 1.4612 (1.4612) | BETTER |
| TACTICAL:lon_err_rate | -0.0438 | -0.0438 | [-0.0771, -0.0171] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0721 | -0.0721 | [-0.0975, -0.0484] | 0.0117 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0504 | -0.0504 | [-0.0746, -0.0267] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0028 | +0.0028 | [-0.0699, +0.0747] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0992 | +0.0992 | [+0.0296, +0.1680] | 0.0425 | WORSE |
| ade6_m|bulk | -0.1013 | -0.1013 | [-0.1593, -0.0523] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0078 | +0.0078 | [-0.0423, +0.0575] | 0.2029 | OPEN |
| ade_tac_m|all | +0.0028 | +0.0028 | [-0.0699, +0.0747] | 0.3008 | OPEN |

## I10RH R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0127 | +0.0127 | [-0.0013, +0.0277] | 0.1482 (0.1482) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0485 | +0.0485 | [+0.0134, +0.0866] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0281 | -0.0281 | [-0.0521, -0.0053] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2588 | -0.2588 | [-0.3945, -0.1176] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0183 | -0.0183 | [-0.0277, -0.0098] | 0.0202 (0.0202) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.8022 | -0.8022 | [-1.1751, -0.4560] | 1.4612 (1.4612) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0117 | -0.0117 | [-0.0208, -0.0025] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0379 | -0.0379 | [-0.0567, -0.0208] | 0.0117 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0300 | -0.0300 | [-0.0484, -0.0133] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0282 | +0.0282 | [-0.0273, +0.0841] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0831 | +0.0831 | [+0.0231, +0.1445] | 0.0425 | WORSE |
| ade6_m|bulk | -0.0449 | -0.0449 | [-0.0774, -0.0141] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0247 | +0.0247 | [-0.0140, +0.0633] | 0.2029 | OPEN |
| ade_tac_m|all | +0.0282 | +0.0282 | [-0.0273, +0.0841] | 0.3008 | OPEN |

## I10RH R0KHb20 minus I10B0 R0KHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0147 | +0.0147 | [+0.0009, +0.0288] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0549 | +0.0549 | [+0.0196, +0.0908] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0013 | -0.0013 | [-0.0189, +0.0152] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | -0.3010 | -0.3010 | [-0.4266, -0.1721] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0198 | -0.0198 | [-0.0294, -0.0111] | 0.0202 (0.0202) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.9959 | -0.9959 | [-1.4430, -0.5998] | 1.4612 (1.4612) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0217 | -0.0217 | [-0.0358, -0.0096] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0146 | -0.0146 | [-0.0233, -0.0063] | 0.0117 (0.0117) | BETTER |
| TACTICAL:joint_err_rate | -0.0200 | -0.0200 | [-0.0346, -0.0079] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0668 | +0.0668 | [+0.0164, +0.1172] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0796 | +0.0796 | [+0.0208, +0.1403] | 0.0425 | WORSE |
| ade6_m|bulk | +0.0156 | +0.0156 | [-0.0033, +0.0351] | 0.3943 | OPEN |
| ade6_m|all | +0.0504 | +0.0504 | [+0.0155, +0.0850] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0668 | +0.0668 | [+0.0164, +0.1172] | 0.3008 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10B0_R0C10Hb20_s2 | 2.5343 | 2.9374 | 1.2615 | 1.0037 | 0.6357 |
| I10B0_R0C10XHb20_s2 | 2.3805 | 2.8752 | 1.2014 | 0.7823 | 0.5940 |
| I10B0_R0G0Hb20_s2 | 2.2531 | 2.8536 | 0.9093 | 1.0316 | 0.5144 |
| I10B0_R0G50Hb20_s2 | 2.2986 | 2.8089 | 0.9985 | 0.9933 | 0.5502 |
| I10B0_R0Hb20_s2 | 2.3534 | 2.7931 | 1.0659 | 1.0070 | 0.5286 |
| I10B0_R0KHb20_s2 | 2.0599 | 2.8150 | 0.9072 | 0.7884 | 0.4435 |
| I10RH_R0C10Hb20_s2 | 2.4243 | 2.9976 | 1.0820 | 0.9979 | 0.5602 |
| I10RH_R0C10XHb20_s2 | 2.2916 | 2.9383 | 1.0788 | 0.7904 | 0.5211 |
| I10RH_R0G0Hb20_s2 | 2.2778 | 2.9366 | 0.9220 | 1.0035 | 0.4844 |
| I10RH_R0G50Hb20_s2 | 2.3064 | 2.9081 | 0.9692 | 0.9733 | 0.4998 |
| I10RH_R0Hb20_s2 | 2.3788 | 2.9140 | 1.0415 | 1.0094 | 0.4948 |
| I10RH_R0KHb20_s2 | 2.1103 | 2.8946 | 0.9219 | 0.7872 | 0.4235 |
| I10UEA_R0C10XHb20_s0 | 3.3455 | 3.7725 | 2.1782 | 1.0278 | 0.7361 |
| I10UEA_R0C10XHb20_s1 | 3.5485 | 3.8150 | 2.3264 | 1.0850 | 0.8383 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
