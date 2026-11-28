# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RLcF5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RLcF5 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0128 / +0.0087 | +0.0107 | [+0.0023, +0.0205] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0229 / +0.0241 | +0.0235 | [+0.0076, +0.0416] | 0.0298 (0.0298) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0000 / +0.0013 | +0.0007 | [-0.0036, +0.0051] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0478 / +0.0386 | +0.0432 | [-0.0037, +0.0899] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0035 / +0.0039 | +0.0037 | [+0.0012, +0.0068] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.2928 / +0.2641 | +0.2785 | [+0.0701, +0.5040] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0008 / +0.0000 | -0.0004 | [-0.0025, +0.0015] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0013 / +0.0029 | +0.0021 | [-0.0008, +0.0052] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0004 / +0.0017 | +0.0006 | [-0.0029, +0.0036] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0298 / +0.0345 | +0.0322 | [+0.0097, +0.0591] | 0.0699 (0.0699) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0131 / +0.0115 | +0.0123 | [+0.0016, +0.0246] | 0.0229 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0296 / +0.0385 | +0.0341 | [+0.0000, +0.0757] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0206 / +0.0238 | +0.0222 | [+0.0069, +0.0404] | 0.0442 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0298 / +0.0345 | +0.0322 | [+0.0097, +0.0591] | 0.0699 | INSIDE-SPREAD |

## I10RQ50F5 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0309 / +0.0297 | +0.0303 | [+0.0132, +0.0500] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0769 / +0.0685 | +0.0727 | [+0.0346, +0.1166] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0029 / +0.0072 | +0.0021 | [-0.0073, +0.0144] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.2335 / +0.1067 | +0.1701 | [+0.0533, +0.3424] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0073 / +0.0043 | +0.0058 | [+0.0024, +0.0106] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.9418 / +0.6815 | +0.8117 | [+0.4312, +1.2679] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0033 / -0.0033 | +0.0000 | [-0.0058, +0.0067] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0008 / +0.0104 | +0.0056 | [-0.0017, +0.0135] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0042 / +0.0046 | +0.0044 | [+0.0004, +0.0088] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1032 / +0.1022 | +0.1027 | [+0.0487, +0.1672] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0309 / +0.0278 | +0.0293 | [+0.0025, +0.0602] | 0.0229 | WORSE |
| ade6_m|bulk | +0.1191 / +0.1195 | +0.1193 | [+0.0544, +0.1996] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0711 / +0.0696 | +0.0704 | [+0.0335, +0.1144] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1032 / +0.1022 | +0.1027 | [+0.0487, +0.1672] | 0.0699 | WORSE |

## I10RLcF2 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0190 / +0.0299 | +0.0244 | [+0.0027, +0.0487] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0672 / +0.0837 | +0.0755 | [+0.0355, +0.1251] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0033 / +0.0059 | +0.0046 | [-0.0020, +0.0117] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.1274 / +0.0955 | +0.1115 | [+0.0419, +0.1874] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0036 / +0.0046 | +0.0041 | [-0.0002, +0.0078] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.3762 / +0.4045 | +0.3903 | [+0.0867, +0.7097] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0017 / -0.0013 | +0.0002 | [-0.0031, +0.0033] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0033 / +0.0042 | +0.0038 | [+0.0002, +0.0079] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0008 / -0.0008 | +0.0000 | [-0.0027, +0.0025] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0950 / +0.1218 | +0.1084 | [+0.0500, +0.1815] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0200 / +0.0363 | +0.0281 | [-0.0044, +0.0622] | 0.0229 | OPEN |
| ade6_m|bulk | +0.1206 / +0.1401 | +0.1303 | [+0.0579, +0.2229] | 0.1243 | WORSE |
| ade6_m|all | +0.0659 / +0.0837 | +0.0748 | [+0.0351, +0.1250] | 0.0442 | WORSE |
| ade_tac_m|all | +0.0950 / +0.1218 | +0.1084 | [+0.0500, +0.1815] | 0.0699 | WORSE |

## I10RLcF2 minus I10RLcF5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0062 / +0.0212 | +0.0137 | [-0.0070, +0.0357] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0443 / +0.0596 | +0.0520 | [+0.0195, +0.0935] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0034 / +0.0045 | +0.0039 | [-0.0013, +0.0098] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0796 / +0.0569 | +0.0682 | [+0.0033, +0.1361] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0001 / +0.0007 | +0.0004 | [-0.0036, +0.0041] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0834 / +0.1403 | +0.1118 | [-0.1923, +0.4395] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0025 / -0.0013 | +0.0006 | [-0.0033, +0.0038] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0021 / +0.0013 | +0.0017 | [-0.0019, +0.0050] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0013 / -0.0025 | -0.0006 | [-0.0042, +0.0027] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0652 / +0.0872 | +0.0762 | [+0.0286, +0.1367] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0069 / +0.0248 | +0.0158 | [-0.0130, +0.0479] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0910 / +0.1016 | +0.0963 | [+0.0334, +0.1710] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.0452 / +0.0598 | +0.0525 | [+0.0203, +0.0931] | 0.0442 | WORSE |
| ade_tac_m|all | +0.0652 / +0.0872 | +0.0762 | [+0.0286, +0.1367] | 0.0699 | WORSE |

## I10RLcF5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0671 / +0.0545 | +0.0608 | [-0.0152, +0.1468] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2752 / +0.2406 | +0.2579 | [+0.0873, +0.4616] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0272 / -0.0236 | -0.0254 | [-0.0442, -0.0072] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4524 / +0.3095 | +0.3810 | [+0.1786, +0.5854] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0194 / +0.0186 | +0.0190 | [+0.0113, +0.0275] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.3738 / +0.4052 | +0.8895 | [-0.1974, +1.9066] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0042 / +0.0029 | -0.0006 | [-0.0108, +0.0088] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0067 / +0.0096 | +0.0081 | [-0.0023, +0.0196] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0008 / +0.0092 | +0.0050 | [-0.0054, +0.0146] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3215 / +0.2869 | +0.3042 | [+0.0816, +0.5650] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1422 / +0.1167 | +0.1294 | [-0.0124, +0.3155] | 0.0229 | OPEN |
| ade6_m|bulk | +0.3478 / +0.3275 | +0.3377 | [+0.1302, +0.5655] | 0.1243 | WORSE |
| ade6_m|all | +0.2360 / +0.2128 | +0.2244 | [+0.0616, +0.4165] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3215 / +0.2869 | +0.3042 | [+0.0816, +0.5650] | 0.0699 | WORSE |

## I10RQ50F5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0888 / +0.0862 | +0.0875 | [-0.0160, +0.1896] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.4102 / +0.4029 | +0.4066 | [+0.2086, +0.6396] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0431 / -0.0168 | -0.0299 | [-0.0608, +0.0038] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.8391 / +0.6404 | +0.7397 | [+0.4510, +1.1067] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0276 / +0.0207 | +0.0241 | [+0.0143, +0.0364] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.8132 / +1.4615 | +2.1374 | [+0.3053, +3.7994] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0021 / -0.0033 | -0.0027 | [-0.0137, +0.0083] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0108 / +0.0267 | +0.0188 | [+0.0000, +0.0377] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0104 / +0.0108 | +0.0106 | [+0.0010, +0.0210] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.4920 / +0.5110 | +0.5015 | [+0.2368, +0.8042] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2668 / +0.2007 | +0.2337 | [+0.0701, +0.4341] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4671 / +0.5577 | +0.5124 | [+0.2456, +0.8229] | 0.1243 | WORSE |
| ade6_m|all | +0.3581 / +0.3635 | +0.3608 | [+0.1725, +0.5765] | 0.0442 | WORSE |
| ade_tac_m|all | +0.4920 / +0.5110 | +0.5015 | [+0.2368, +0.8042] | 0.0699 | WORSE |

## I10RLcF2 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0733 / +0.0757 | +0.0745 | [-0.0070, +0.1672] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3196 / +0.3002 | +0.3099 | [+0.1171, +0.5306] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0239 / -0.0191 | -0.0215 | [-0.0416, -0.0014] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5320 / +0.3603 | +0.4462 | [+0.2157, +0.6771] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0195 / +0.0188 | +0.0191 | [+0.0118, +0.0273] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.4572 / +0.6685 | +1.0628 | [+0.0449, +2.0716] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0017 / +0.0017 | +0.0000 | [-0.0098, +0.0090] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0088 / +0.0108 | +0.0098 | [-0.0015, +0.0221] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0021 / +0.0067 | +0.0044 | [-0.0052, +0.0142] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3867 / +0.3742 | +0.3805 | [+0.1274, +0.6667] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1491 / +0.1415 | +0.1453 | [+0.0009, +0.3270] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4388 / +0.4291 | +0.4339 | [+0.1745, +0.7243] | 0.1243 | WORSE |
| ade6_m|all | +0.2812 / +0.2726 | +0.2769 | [+0.0932, +0.4859] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3867 / +0.3742 | +0.3805 | [+0.1274, +0.6667] | 0.0699 | WORSE |

## I10RQ50F5 minus I10RLcF5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0217 / +0.0317 | +0.0267 | [-0.0327, +0.0835] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1350 / +0.1623 | +0.1487 | [+0.0690, +0.2412] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0158 / +0.0068 | -0.0045 | [-0.0334, +0.0262] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.4081 / +0.3240 | +0.3660 | [+0.1327, +0.6853] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0121 / +0.0084 | +0.0103 | [+0.0027, +0.0192] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.2687 / +0.6963 | +0.9825 | [-0.3376, +2.2495] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0021 / -0.0063 | -0.0021 | [-0.0113, +0.0071] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0042 / +0.0171 | +0.0106 | [-0.0033, +0.0242] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0096 / +0.0017 | +0.0056 | [-0.0025, +0.0142] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.1704 / +0.2240 | +0.1972 | [+0.0816, +0.3294] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1246 / +0.0840 | +0.1043 | [+0.0119, +0.2335] | 0.0229 | WORSE |
| ade6_m|bulk | +0.1192 / +0.2302 | +0.1747 | [+0.0411, +0.3410] | 0.1243 | WORSE |
| ade6_m|all | +0.1221 / +0.1507 | +0.1364 | [+0.0568, +0.2282] | 0.0442 | WORSE |
| ade_tac_m|all | +0.1704 / +0.2240 | +0.1972 | [+0.0816, +0.3294] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcF2_s0 | 3.0160 | 3.2618 | 1.6701 | 1.2706 | 0.6490 |
| I10RLcF2_s1 | 2.9632 | 3.2771 | 1.7236 | 1.1696 | 0.6782 |
| I10RLcF5_s0 | 2.9708 | 3.2549 | 1.6639 | 1.2672 | 0.6478 |
| I10RLcF5_s1 | 2.9034 | 3.2523 | 1.7024 | 1.1650 | 0.6807 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50F5_s0 | 3.0929 | 3.3795 | 1.6856 | 1.2514 | 0.6574 |
| I10RQ50F5_s1 | 3.0541 | 3.3363 | 1.7342 | 1.1719 | 0.6824 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
