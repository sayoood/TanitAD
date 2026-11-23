# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 16}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10KI_R0C10XHb20"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.44464669811771973, "D_read": 0.4446466981177198, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10UEA_R0C10XHb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10KI R0C10XHb20 minus I10B0 R0C10XHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0092 | -0.0092 | [-0.0761, +0.0491] | 0.1482 (0.1482) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0016 | +0.0016 | [-0.0792, +0.0724] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0408 | +0.0408 | [+0.0198, +0.0634] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.7645 | +1.7645 | [+1.3014, +2.2324] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0547 | +0.0547 | [+0.0296, +0.0830] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +3.7084 | +3.7084 | [+2.5437, +4.9156] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0117 | +0.0117 | [-0.0279, +0.0446] | 0.1071 (0.1071) | OPEN |
| TACTICAL:lat_err_rate | +0.0392 | +0.0392 | [+0.0179, +0.0605] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0063 | +0.0063 | [-0.0325, +0.0388] | 0.1021 (0.1021) | OPEN |
| TACTICAL:ade_tac_m | +0.0484 | +0.0484 | [-0.0705, +0.1520] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1316 | +0.1316 | [+0.0741, +0.1900] | 0.0425 | WORSE |
| ade6_m|bulk | -0.0855 | -0.0855 | [-0.2366, +0.0431] | 0.3943 | OPEN |
| ade6_m|all | +0.0326 | +0.0326 | [-0.0467, +0.1018] | 0.2029 | OPEN |
| ade_tac_m|all | +0.0484 | +0.0484 | [-0.0705, +0.1520] | 0.3008 | OPEN |

## I10KI R0C10XHb20 minus I10KI R0KHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2635 | +0.2635 | [+0.1876, +0.3436] | 0.1482 (0.1482) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3323 | +0.3323 | [+0.2239, +0.4473] | 0.2027 (0.2027) | WORSE |
| LATERAL:cross_mae_m | +0.0005 | +0.0005 | [-0.0219, +0.0225] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | +2.6304 | +2.6304 | [+1.9472, +3.3482] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1931 | +0.1931 | [+0.1349, +0.2726] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +13.7848 | +13.7848 | [+10.5157, +17.5574] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.1380 | +0.1380 | [+0.0920, +0.1844] | 0.1071 (0.1071) | WORSE |
| TACTICAL:lat_err_rate | +0.0909 | +0.0909 | [+0.0625, +0.1188] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.1426 | +0.1426 | [+0.0975, +0.1847] | 0.1021 (0.1021) | WORSE |
| TACTICAL:ade_tac_m | +0.4547 | +0.4547 | [+0.2951, +0.6213] | 0.3008 (0.3008) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1241 | +0.1241 | [+0.0070, +0.2417] | 0.0425 | WORSE |
| ade6_m|bulk | +0.5167 | +0.5167 | [+0.3710, +0.6999] | 0.3943 | WORSE |
| ade6_m|all | +0.3031 | +0.3031 | [+0.1968, +0.4142] | 0.2029 | WORSE |
| ade_tac_m|all | +0.4547 | +0.4547 | [+0.2951, +0.6213] | 0.3008 | WORSE |

## I10KI R0G50Hb20 minus I10KI R0G0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0924 | +0.0924 | [+0.0512, +0.1352] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1207 | +0.1207 | [+0.0636, +0.1816] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0306 | -0.0306 | [-0.0467, -0.0152] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.9196 | +0.9196 | [+0.5829, +1.3395] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0865 | +0.0865 | [+0.0624, +0.1117] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.3539 | +7.3539 | [+5.5821, +9.1806] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0817 | +0.0817 | [+0.0459, +0.1160] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0100 | +0.0100 | [-0.0050, +0.0242] | 0.0117 (0.0117) | OPEN |
| TACTICAL:joint_err_rate | +0.0496 | +0.0496 | [+0.0279, +0.0704] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1134 | +0.1134 | [+0.0390, +0.1892] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0102 | -0.0102 | [-0.0741, +0.0570] | 0.0425 | OPEN |
| ade6_m|bulk | +0.1779 | +0.1779 | [+0.1175, +0.2472] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0756 | +0.0756 | [+0.0260, +0.1261] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.1134 | +0.1134 | [+0.0390, +0.1892] | 0.3008 | INSIDE-SPREAD |

## I10KI R0C10Hb20 minus I10KI R0G50Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1963 | +0.1963 | [+0.1547, +0.2422] | 0.1482 (0.1482) | WORSE |
| LONGITUDINAL:along_mae_m | +0.2390 | +0.2390 | [+0.1837, +0.3022] | 0.2027 (0.2027) | WORSE |
| LATERAL:cross_mae_m | +0.0198 | +0.0198 | [+0.0013, +0.0423] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.4490 | +2.4490 | [+1.7719, +3.1005] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1259 | +0.1259 | [+0.0876, +0.1740] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.8285 | +12.8285 | [+9.8428, +15.9246] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0934 | +0.0934 | [+0.0633, +0.1256] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0521 | +0.0521 | [+0.0325, +0.0701] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0855 | +0.0855 | [+0.0621, +0.1092] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3318 | +0.3318 | [+0.2552, +0.4114] | 0.3008 (0.3008) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1703 | +0.1703 | [+0.1108, +0.2364] | 0.0425 | WORSE |
| ade6_m|bulk | +0.2818 | +0.2818 | [+0.2107, +0.3713] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.2212 | +0.2212 | [+0.1702, +0.2743] | 0.2029 | WORSE |
| ade_tac_m|all | +0.3318 | +0.3318 | [+0.2552, +0.4114] | 0.3008 | WORSE |

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

## I10KI R0C10XHb20 minus I10B0 R0KHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2850 | +0.2850 | [+0.2041, +0.3706] | 0.1482 (0.1482) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3522 | +0.3522 | [+0.2368, +0.4765] | 0.2027 (0.2027) | WORSE |
| LATERAL:cross_mae_m | +0.0347 | +0.0347 | [+0.0093, +0.0609] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.9206 | +2.9206 | [+2.2158, +3.6491] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.2103 | +0.2103 | [+0.1465, +0.2959] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +14.4797 | +14.4797 | [+11.0795, +18.4376] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.1488 | +0.1488 | [+0.1035, +0.1942] | 0.1071 (0.1071) | WORSE |
| TACTICAL:lat_err_rate | +0.0925 | +0.0925 | [+0.0642, +0.1204] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.1567 | +0.1567 | [+0.1113, +0.1993] | 0.1021 (0.1021) | WORSE |
| TACTICAL:ade_tac_m | +0.5293 | +0.5293 | [+0.3584, +0.7129] | 0.3008 (0.3008) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1918 | +0.1918 | [+0.0535, +0.3303] | 0.0425 | WORSE |
| ade6_m|bulk | +0.5456 | +0.5456 | [+0.3898, +0.7344] | 0.3943 | WORSE |
| ade6_m|all | +0.3532 | +0.3532 | [+0.2391, +0.4755] | 0.2029 | WORSE |
| ade_tac_m|all | +0.5293 | +0.5293 | [+0.3584, +0.7129] | 0.3008 | WORSE |

## I10KI R0Hb20 minus I10B0 R0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0206 | +0.0206 | [-0.0170, +0.0515] | 0.1482 (0.1482) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0383 | +0.0383 | [-0.0049, +0.0795] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0214 | +0.0214 | [-0.0042, +0.0467] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | +1.0806 | +1.0806 | [+0.7984, +1.3818] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0511 | +0.0511 | [+0.0349, +0.0686] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +3.6855 | +3.6855 | [+2.5229, +4.9099] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | -0.0050 | -0.0050 | [-0.0375, +0.0263] | 0.1071 (0.1071) | OPEN |
| TACTICAL:lat_err_rate | -0.0083 | -0.0083 | [-0.0271, +0.0083] | 0.0117 (0.0117) | OPEN |
| TACTICAL:joint_err_rate | -0.0029 | -0.0029 | [-0.0329, +0.0254] | 0.1021 (0.1021) | OPEN |
| TACTICAL:ade_tac_m | +0.0733 | +0.0733 | [+0.0184, +0.1266] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0612 | +0.0612 | [+0.0085, +0.1129] | 0.0425 | WORSE |
| ade6_m|bulk | +0.0348 | +0.0348 | [-0.0054, +0.0753] | 0.3943 | OPEN |
| ade6_m|all | +0.0492 | +0.0492 | [+0.0124, +0.0849] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0733 | +0.0733 | [+0.0184, +0.1266] | 0.3008 | INSIDE-SPREAD |

## I10KI R0C10Hb20 minus I10B0 R0C10Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0498 | -0.0498 | [-0.1209, +0.0135] | 0.1482 (0.1482) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0028 | +0.0028 | [-0.0840, +0.0822] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0433 | +0.0433 | [+0.0108, +0.0741] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.0546 | +2.0546 | [+1.5580, +2.5484] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0737 | +0.0737 | [+0.0446, +0.1070] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.6502 | +6.6502 | [+4.6317, +8.9053] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0092 | +0.0092 | [-0.0200, +0.0367] | 0.1071 (0.1071) | OPEN |
| TACTICAL:lat_err_rate | +0.0425 | +0.0425 | [+0.0250, +0.0601] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0196 | +0.0196 | [-0.0021, +0.0413] | 0.1021 (0.1021) | OPEN |
| TACTICAL:ade_tac_m | +0.0819 | +0.0819 | [-0.0107, +0.1676] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1038 | +0.1038 | [+0.0474, +0.1605] | 0.0425 | WORSE |
| ade6_m|bulk | -0.0034 | -0.0034 | [-0.1205, +0.0896] | 0.3943 | OPEN |
| ade6_m|all | +0.0549 | +0.0549 | [-0.0067, +0.1123] | 0.2029 | OPEN |
| ade_tac_m|all | +0.0819 | +0.0819 | [-0.0107, +0.1676] | 0.3008 | OPEN |

## I10KI R0G50Hb20 minus I10B0 R0G50Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0169 | +0.0169 | [-0.0104, +0.0429] | 0.1482 (0.1482) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0434 | +0.0434 | [+0.0101, +0.0780] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0338 | +0.0338 | [+0.0043, +0.0621] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.9102 | +0.9102 | [+0.6618, +1.1820] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0443 | +0.0443 | [+0.0303, +0.0596] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +3.2287 | +3.2287 | [+2.3030, +4.2503] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0033 | +0.0033 | [-0.0250, +0.0271] | 0.1071 (0.1071) | OPEN |
| TACTICAL:lat_err_rate | +0.0304 | +0.0304 | [+0.0129, +0.0492] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0196 | +0.0196 | [+0.0041, +0.0350] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1037 | +0.1037 | [+0.0554, +0.1538] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0619 | +0.0619 | [+0.0154, +0.1115] | 0.0425 | WORSE |
| ade6_m|bulk | +0.0784 | +0.0784 | [+0.0441, +0.1141] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0694 | +0.0694 | [+0.0371, +0.1032] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.1037 | +0.1037 | [+0.0554, +0.1538] | 0.3008 | INSIDE-SPREAD |

## I10KI R0G0Hb20 minus I10B0 R0G0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0136 | +0.0136 | [+0.0019, +0.0255] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0205 | +0.0205 | [-0.0012, +0.0419] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0262 | +0.0262 | [+0.0012, +0.0515] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2965 | +0.2965 | [+0.1723, +0.4346] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0159 | +0.0159 | [+0.0085, +0.0250] | 0.0202 (0.0202) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.7278 | +0.7278 | [+0.4669, +1.0497] | 1.4612 (1.4612) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0063 | +0.0063 | [-0.0033, +0.0163] | 0.1071 (0.1071) | OPEN |
| TACTICAL:lat_err_rate | +0.0108 | +0.0108 | [+0.0012, +0.0217] | 0.0117 (0.0117) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0058 | +0.0058 | [-0.0038, +0.0154] | 0.1021 (0.1021) | OPEN |
| TACTICAL:ade_tac_m | +0.0586 | +0.0586 | [+0.0188, +0.0999] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0275 | +0.0275 | [-0.0127, +0.0703] | 0.0425 | OPEN |
| ade6_m|bulk | +0.0536 | +0.0536 | [+0.0298, +0.0793] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0394 | +0.0394 | [+0.0127, +0.0673] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0586 | +0.0586 | [+0.0188, +0.0999] | 0.3008 | INSIDE-SPREAD |

## I10KI R0KHb20 minus I10B0 R0KHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0216 | +0.0216 | [+0.0094, +0.0342] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0198 | +0.0198 | [-0.0017, +0.0420] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0341 | +0.0341 | [+0.0127, +0.0564] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2904 | +0.2904 | [+0.1792, +0.4165] | 0.6001 (0.6001) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0181 | +0.0181 | [+0.0099, +0.0279] | 0.0202 (0.0202) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.7152 | +0.7152 | [+0.4362, +1.0597] | 1.4612 (1.4612) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0108 | +0.0108 | [-0.0021, +0.0246] | 0.1071 (0.1071) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 | +0.0017 | [-0.0063, +0.0104] | 0.0117 (0.0117) | OPEN |
| TACTICAL:joint_err_rate | +0.0142 | +0.0142 | [-0.0004, +0.0279] | 0.1021 (0.1021) | OPEN |
| TACTICAL:ade_tac_m | +0.0745 | +0.0745 | [+0.0366, +0.1154] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0677 | +0.0677 | [+0.0246, +0.1133] | 0.0425 | WORSE |
| ade6_m|bulk | +0.0290 | +0.0290 | [+0.0092, +0.0498] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0500 | +0.0500 | [+0.0248, +0.0775] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0745 | +0.0745 | [+0.0366, +0.1154] | 0.3008 | INSIDE-SPREAD |

## I10KI R0Hb20 minus I10RH R0Hb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0449 | +0.0449 | [+0.0036, +0.0863] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0179 | +0.0179 | [-0.0567, +0.0915] | 0.2027 (0.2027) | OPEN |
| LATERAL:cross_mae_m | +0.0191 | +0.0191 | [-0.0150, +0.0484] | 0.0572 (0.0572) | OPEN |
| LATERAL:heading_mae_deg | +1.6424 | +1.6424 | [+1.2416, +2.0739] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1040 | +0.1040 | [+0.0778, +0.1330] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.4251 | +6.4251 | [+4.9426, +8.0255] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0233 | +0.0233 | [+0.0042, +0.0446] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0496 | +0.0496 | [+0.0275, +0.0734] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0308 | +0.0308 | [+0.0079, +0.0533] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0440 | +0.0440 | [-0.0595, +0.1410] | 0.3008 (0.3008) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0596 | -0.0596 | [-0.1714, +0.0452] | 0.0425 | OPEN |
| ade6_m|bulk | +0.1232 | +0.1232 | [+0.0725, +0.1814] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.0237 | +0.0237 | [-0.0468, +0.0899] | 0.2029 | OPEN |
| ade_tac_m|all | +0.0440 | +0.0440 | [-0.0595, +0.1410] | 0.3008 | OPEN |

## I10KI R0C10XHb20 minus I10RH R0C10XHb20

Seeds read [2] of [2]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1135 | +0.1135 | [+0.0584, +0.1715] | 0.1482 (0.1482) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0960 | +0.0960 | [+0.0119, +0.1867] | 0.2027 (0.2027) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0327 | +0.0327 | [+0.0041, +0.0635] | 0.0572 (0.0572) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.5622 | +2.5622 | [+1.9549, +3.1586] | 0.6001 (0.6001) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1449 | +0.1449 | [+0.0955, +0.2100] | 0.0202 (0.0202) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.0806 | +8.0806 | [+5.8900, +10.5737] | 1.4612 (1.4612) | WORSE |
| TACTICAL:lon_err_rate | +0.0704 | +0.0704 | [+0.0421, +0.1008] | 0.1071 (0.1071) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0880 | +0.0880 | [+0.0617, +0.1138] | 0.0117 (0.0117) | WORSE |
| TACTICAL:joint_err_rate | +0.0792 | +0.0792 | [+0.0542, +0.1067] | 0.1021 (0.1021) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1906 | +0.1906 | [+0.0638, +0.3251] | 0.3008 (0.3008) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0686 | +0.0686 | [-0.0527, +0.1858] | 0.0425 | OPEN |
| ade6_m|bulk | +0.1847 | +0.1847 | [+0.0915, +0.3064] | 0.3943 | INSIDE-SPREAD |
| ade6_m|all | +0.1215 | +0.1215 | [+0.0362, +0.2115] | 0.2029 | INSIDE-SPREAD |
| ade_tac_m|all | +0.1906 | +0.1906 | [+0.0638, +0.3251] | 0.3008 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10B0_R0C10Hb20_s2 | 2.5343 | 2.9374 | 1.2615 | 1.0037 | 0.6357 |
| I10B0_R0C10XHb20_s2 | 2.3805 | 2.8752 | 1.2014 | 0.7823 | 0.5940 |
| I10B0_R0G0Hb20_s2 | 2.2531 | 2.8536 | 0.9093 | 1.0316 | 0.5144 |
| I10B0_R0G50Hb20_s2 | 2.2986 | 2.8089 | 0.9985 | 0.9933 | 0.5502 |
| I10B0_R0Hb20_s2 | 2.3534 | 2.7931 | 1.0659 | 1.0070 | 0.5286 |
| I10B0_R0KHb20_s2 | 2.0599 | 2.8150 | 0.9072 | 0.7884 | 0.4435 |
| I10KI_R0C10Hb20_s2 | 2.5892 | 3.0412 | 1.2117 | 1.0470 | 0.6553 |
| I10KI_R0C10XHb20_s2 | 2.4131 | 3.0068 | 1.1922 | 0.8231 | 0.6003 |
| I10KI_R0G0Hb20_s2 | 2.2925 | 2.8810 | 0.9229 | 1.0578 | 0.5202 |
| I10KI_R0G50Hb20_s2 | 2.3681 | 2.8709 | 1.0154 | 1.0271 | 0.5698 |
| I10KI_R0Hb20_s2 | 2.4026 | 2.8544 | 1.0865 | 1.0285 | 0.5256 |
| I10KI_R0KHb20_s2 | 2.1099 | 2.8827 | 0.9287 | 0.8226 | 0.4577 |
| I10RH_R0C10XHb20_s2 | 2.2916 | 2.9383 | 1.0788 | 0.7904 | 0.5211 |
| I10RH_R0Hb20_s2 | 2.3788 | 2.9140 | 1.0415 | 1.0094 | 0.4948 |
| I10UEA_R0C10XHb20_s0 | 3.3455 | 3.7725 | 2.1782 | 1.0278 | 0.7361 |
| I10UEA_R0C10XHb20_s1 | 3.5485 | 3.8150 | 2.3264 | 1.0850 | 0.8383 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
