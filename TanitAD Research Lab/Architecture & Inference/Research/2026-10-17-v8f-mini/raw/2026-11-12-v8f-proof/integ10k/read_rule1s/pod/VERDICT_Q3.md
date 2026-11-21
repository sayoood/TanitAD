# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 24}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0B5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0B5 minus I10R0Bk10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1965 / +0.2010 | +0.1987 | [+0.1141, +0.2892] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4935 / +0.4941 | +0.4938 | [+0.4121, +0.5795] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4762 / +0.4216 | +0.4489 | [+0.3301, +0.5771] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +3.8663 / +2.3504 | +3.1083 | [+1.9398, +4.4588] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0770 / -0.0061 | +0.0355 | [-0.0330, +0.0965] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +19.6390 / +15.5751 | +17.6070 | [+12.0005, +24.2846] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.1071 / -0.1242 | -0.1157 | [-0.1569, -0.0758] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0183 / +0.0029 | +0.0106 | [-0.0177, +0.0373] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0675 / -0.1021 | -0.0848 | [-0.1298, -0.0433] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +1.2644 / +1.2153 | +1.2398 | [+1.0499, +1.4345] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.5438 / +1.4868 | +1.5153 | [+1.3432, +1.6910] | 0.0229 | WORSE |
| ade6_m|bulk | +0.0062 / +0.0016 | +0.0039 | [-0.0621, +0.0707] | 0.1243 | OPEN |
| ade6_m|all | +0.8426 / +0.8096 | +0.8261 | [+0.6983, +0.9569] | 0.0442 | WORSE |
| ade_tac_m|all | +1.2644 / +1.2153 | +1.2398 | [+1.0499, +1.4345] | 0.0699 | WORSE |

## I10RLcBk5 minus I10RLcBk10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1064 / +0.1298 | +0.1181 | [+0.0380, +0.2075] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4592 / +0.4941 | +0.4766 | [+0.3832, +0.5758] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4750 / +0.4000 | +0.4375 | [+0.3063, +0.5704] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +3.3572 / +2.2344 | +2.7958 | [+1.8194, +3.8645] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0911 / +0.0396 | +0.0654 | [+0.0187, +0.1140] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +17.2573 / +14.0032 | +15.6303 | [+10.6848, +21.7118] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.1109 / -0.1301 | -0.1205 | [-0.1630, -0.0848] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0121 / -0.0058 | +0.0031 | [-0.0279, +0.0310] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0734 / -0.1025 | -0.0880 | [-0.1299, -0.0481] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +1.2060 / +1.1981 | +1.2021 | [+1.0173, +1.3992] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.4083 / +1.3650 | +1.3866 | [+1.2039, +1.5734] | 0.0229 | WORSE |
| ade6_m|bulk | +0.0955 / +0.1336 | +0.1145 | [+0.0092, +0.2308] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | +0.8096 / +0.8034 | +0.8065 | [+0.6794, +0.9407] | 0.0442 | WORSE |
| ade_tac_m|all | +1.2060 / +1.1981 | +1.2021 | [+1.0173, +1.3992] | 0.0699 | WORSE |

## I10RQ50Bk5 minus I10RQ50Bk10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0249 / +0.1003 | +0.0626 | [-0.0249, +0.1643] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.4152 / +0.4793 | +0.4473 | [+0.3516, +0.5457] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4727 / +0.4071 | +0.4399 | [+0.3146, +0.5705] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +3.3183 / +2.3293 | +2.8238 | [+1.9267, +3.8191] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0893 / +0.0312 | +0.0603 | [+0.0101, +0.1115] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +15.4267 / +12.2397 | +13.8332 | [+9.4685, +18.9837] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.1359 / -0.1417 | -0.1388 | [-0.1763, -0.1024] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0004 / -0.0138 | -0.0067 | [-0.0365, +0.0215] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0909 / -0.1105 | -0.1007 | [-0.1409, -0.0636] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +1.1327 / +1.1716 | +1.1522 | [+0.9711, +1.3500] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.3774 / +1.3603 | +1.3688 | [+1.1929, +1.5553] | 0.0229 | WORSE |
| ade6_m|bulk | +0.0288 / +0.1012 | +0.0650 | [-0.0231, +0.1795] | 0.1243 | OPEN |
| ade6_m|all | +0.7624 / +0.7861 | +0.7743 | [+0.6520, +0.9088] | 0.0442 | WORSE |
| ade_tac_m|all | +1.1327 / +1.1716 | +1.1522 | [+0.9711, +1.3500] | 0.0699 | WORSE |

## I10RLcBk5 minus I10R0B5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0148 / +0.0001 | +0.0075 | [-0.0786, +0.1040] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1903 / +0.1877 | +0.1890 | [-0.0248, +0.4402] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0030 / -0.0122 | -0.0076 | [-0.0227, +0.0026] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0699 / +0.1698 | +0.1199 | [-0.0008, +0.2877] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0227 / +0.0184 | +0.0205 | [+0.0069, +0.0376] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.1503 / -2.0584 | -1.6044 | [-3.2427, -0.4676] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0033 / -0.0042 | -0.0038 | [-0.0206, +0.0138] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0013 / +0.0021 | +0.0017 | [-0.0035, +0.0060] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0021 / +0.0025 | +0.0023 | [-0.0129, +0.0188] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.2368 / +0.2265 | +0.2317 | [-0.0618, +0.5699] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0585 / -0.0640 | -0.0612 | [-0.2281, +0.1354] | 0.0229 | OPEN |
| ade6_m|bulk | +0.4746 / +0.4647 | +0.4697 | [+0.1659, +0.8047] | 0.1243 | WORSE |
| ade6_m|all | +0.1846 / +0.1771 | +0.1809 | [-0.0313, +0.4320] | 0.0442 | OPEN |
| ade_tac_m|all | +0.2368 / +0.2265 | +0.2317 | [-0.0618, +0.5699] | 0.0699 | OPEN |

## I10RQ50Bk5 minus I10R0B5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0005 / -0.0069 | -0.0037 | [-0.0868, +0.0837] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2850 / +0.2212 | +0.2531 | [+0.0583, +0.4797] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0013 / -0.0025 | -0.0019 | [-0.0135, +0.0087] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.1583 / +0.2950 | +0.2266 | [-0.1002, +0.5594] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0358 / +0.0154 | +0.0256 | [+0.0064, +0.0560] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.8699 / -3.7675 | -2.8187 | [-5.3342, -0.1093] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0154 / -0.0108 | -0.0131 | [-0.0304, +0.0044] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / -0.0004 | +0.0006 | [-0.0040, +0.0054] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0063 / +0.0000 | -0.0031 | [-0.0186, +0.0131] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3523 / +0.2652 | +0.3087 | [+0.0459, +0.6169] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1008 / -0.0394 | +0.0307 | [-0.1461, +0.2456] | 0.0229 | OPEN |
| ade6_m|bulk | +0.4723 / +0.4940 | +0.4832 | [+0.2198, +0.7791] | 0.1243 | WORSE |
| ade6_m|all | +0.2702 / +0.2038 | +0.2370 | [+0.0447, +0.4579] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3523 / +0.2652 | +0.3087 | [+0.0459, +0.6169] | 0.0699 | WORSE |

## I10RLcBk10 minus I10R0Bk10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1049 / +0.0714 | +0.0881 | [+0.0148, +0.1759] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.2246 / +0.1878 | +0.2062 | [+0.0376, +0.4084] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0017 / +0.0095 | +0.0039 | [-0.0173, +0.0330] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.3310 / +0.1764 | +0.2537 | [+0.1052, +0.4521] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0017 / -0.0154 | -0.0069 | [-0.0280, +0.0035] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.3584 / -0.5591 | -0.1003 | [-1.0459, +0.4887] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0004 / +0.0017 | +0.0010 | [-0.0100, +0.0115] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0075 / +0.0108 | +0.0092 | [-0.0023, +0.0225] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0079 / +0.0029 | +0.0054 | [-0.0054, +0.0177] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.2952 / +0.2437 | +0.2694 | [+0.0368, +0.5497] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0771 / +0.0578 | +0.0675 | [-0.0639, +0.2301] | 0.0229 | OPEN |
| ade6_m|bulk | +0.3853 / +0.3328 | +0.3590 | [+0.1278, +0.6260] | 0.1243 | WORSE |
| ade6_m|all | +0.2176 / +0.1832 | +0.2004 | [+0.0337, +0.4051] | 0.0442 | WORSE |
| ade_tac_m|all | +0.2952 / +0.2437 | +0.2694 | [+0.0368, +0.5497] | 0.0699 | WORSE |

## I10RQ50Bk10 minus I10R0Bk10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1711 / +0.0938 | +0.1324 | [+0.0472, +0.2339] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3633 / +0.2360 | +0.2996 | [+0.1243, +0.5037] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0022 / +0.0120 | +0.0071 | [-0.0124, +0.0301] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.4867 / +0.2173 | +0.3520 | [+0.1261, +0.7146] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0107 / -0.0022 | +0.0042 | [-0.0067, +0.0157] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.7844 / -0.0885 | +0.3480 | [-0.3663, +1.0998] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0133 / +0.0067 | +0.0100 | [-0.0011, +0.0231] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0196 / +0.0163 | +0.0179 | [+0.0060, +0.0306] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0171 / +0.0083 | +0.0127 | [+0.0008, +0.0256] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.4839 / +0.3089 | +0.3964 | [+0.1537, +0.6743] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2672 / +0.0872 | +0.1772 | [+0.0145, +0.3886] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4497 / +0.3944 | +0.4220 | [+0.1963, +0.6721] | 0.1243 | WORSE |
| ade6_m|all | +0.3504 / +0.2273 | +0.2889 | [+0.1140, +0.4889] | 0.0442 | WORSE |
| ade_tac_m|all | +0.4839 / +0.3089 | +0.3964 | [+0.1537, +0.6743] | 0.0699 | WORSE |

## I10RLcBk5 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2979 / -0.3553 | -0.3266 | [-0.4269, -0.2240] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | +0.1516 / +0.1508 | +0.1512 | [+0.0102, +0.2958] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.1411 / +0.2205 | +0.1808 | [+0.0093, +0.3720] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +0.6355 / +0.0774 | +0.3565 | [-0.4474, +1.1786] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0607 / +0.0216 | +0.0412 | [-0.0041, +0.0918] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +4.6885 / +2.1045 | +3.3965 | [-0.6433, +7.9009] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.1226 / -0.1726 | -0.1476 | [-0.2031, -0.0961] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1242 / -0.0809 | -0.1025 | [-0.1502, -0.0586] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1246 / -0.1676 | -0.1461 | [-0.1971, -0.0971] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.5213 / +0.5878 | +0.5546 | [+0.2656, +0.8493] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.2618 / +1.2367 | +1.2492 | [+1.0106, +1.4936] | 0.0229 | WORSE |
| ade6_m|bulk | -0.7307 / -0.6055 | -0.6681 | [-0.9246, -0.4618] | 0.1243 | BETTER |
| ade6_m|all | +0.3532 / +0.3966 | +0.3749 | [+0.1796, +0.5746] | 0.0442 | WORSE |
| ade_tac_m|all | +0.5213 / +0.5878 | +0.5546 | [+0.2656, +0.8493] | 0.0699 | WORSE |

## I10RQ50Bk5 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3169 / -0.3730 | -0.3450 | [-0.4451, -0.2481] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | +0.1653 / +0.0663 | +0.1158 | [-0.0477, +0.2716] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.1557 / +0.2293 | +0.1925 | [+0.0197, +0.3837] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +0.5999 / -0.0039 | +0.2980 | [-0.6076, +1.1736] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0591 / +0.0134 | +0.0363 | [-0.0139, +0.0884] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +3.0506 / +0.3845 | +1.7176 | [-2.2476, +5.9820] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.1326 / -0.1763 | -0.1544 | [-0.2086, -0.1057] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1284 / -0.0930 | -0.1107 | [-0.1543, -0.0690] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1380 / -0.1688 | -0.1534 | [-0.2007, -0.1078] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.5398 / +0.4700 | +0.5049 | [+0.2143, +0.8096] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.3143 / +1.1935 | +1.2539 | [+0.9852, +1.5038] | 0.0229 | WORSE |
| ade6_m|bulk | -0.7627 / -0.7254 | -0.7441 | [-0.9643, -0.5640] | 0.1243 | BETTER |
| ade6_m|all | +0.3671 / +0.3184 | +0.3428 | [+0.1438, +0.5498] | 0.0442 | WORSE |
| ade_tac_m|all | +0.5398 / +0.4700 | +0.5049 | [+0.2143, +0.8096] | 0.0699 | WORSE |

## I10R0Bk10 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.4550 / -0.5106 | -0.4828 | [-0.5804, -0.3973] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2799 / -0.3146 | -0.2972 | [-0.4042, -0.2008] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3594 / -0.2139 | -0.2866 | [-0.4190, -0.1554] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.6407 / -2.0694 | -2.3551 | [-2.9432, -1.6851] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0147 / +0.0308 | +0.0081 | [-0.0249, +0.0520] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -12.4329 / -11.2215 | -11.8272 | [-14.5645, -9.5033] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0154 / -0.0413 | -0.0283 | [-0.0562, +0.0008] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.1384 / -0.0792 | -0.1088 | [-0.1574, -0.0633] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0579 / -0.0604 | -0.0592 | [-0.0809, -0.0373] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.6881 / -0.6016 | -0.6449 | [-0.8623, -0.4618] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0945 / -0.0810 | -0.0877 | [-0.2010, +0.0277] | 0.0229 | OPEN |
| ade6_m|bulk | -0.8933 / -0.7829 | -0.8381 | [-1.0891, -0.6334] | 0.1243 | BETTER |
| ade6_m|all | -0.4587 / -0.4011 | -0.4299 | [-0.5749, -0.3079] | 0.0442 | BETTER |
| ade_tac_m|all | -0.6881 / -0.6016 | -0.6449 | [-0.8623, -0.4618] | 0.0699 | BETTER |

## I10RLcBk10 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.4044 / -0.4851 | -0.4447 | [-0.5458, -0.3538] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.3076 / -0.3433 | -0.3255 | [-0.4365, -0.2297] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3339 / -0.1794 | -0.2567 | [-0.3928, -0.1218] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.6872 / -2.1571 | -2.4221 | [-2.9954, -1.8066] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0314 / -0.0180 | -0.0247 | [-0.0453, -0.0005] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -12.6787 / -11.8988 | -12.2887 | [-14.6230, -10.2396] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0117 / -0.0425 | -0.0271 | [-0.0573, +0.0038] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.1363 / -0.0750 | -0.1057 | [-0.1561, -0.0600] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0513 / -0.0650 | -0.0581 | [-0.0817, -0.0356] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.6847 / -0.6103 | -0.6475 | [-0.8538, -0.4689] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1465 / -0.1283 | -0.1374 | [-0.2571, -0.0165] | 0.0229 | BETTER |
| ade6_m|bulk | -0.8262 / -0.7391 | -0.7827 | [-1.0330, -0.5854] | 0.1243 | BETTER |
| ade6_m|all | -0.4565 / -0.4069 | -0.4317 | [-0.5692, -0.3126] | 0.0442 | BETTER |
| ade_tac_m|all | -0.6847 / -0.6103 | -0.6475 | [-0.8538, -0.4689] | 0.0699 | BETTER |

## I10RQ50Bk10 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3419 / -0.4733 | -0.4076 | [-0.5274, -0.2984] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2499 / -0.4130 | -0.3314 | [-0.4932, -0.1868] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3170 / -0.1779 | -0.2474 | [-0.3753, -0.1206] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -2.6628 / -2.3779 | -2.5203 | [-3.0687, -1.9561] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0302 / -0.0182 | -0.0242 | [-0.0449, +0.0009] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -12.4451 / -11.7434 | -12.0942 | [-14.3045, -10.1541] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.0033 / -0.0346 | -0.0156 | [-0.0487, +0.0177] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.1288 / -0.0792 | -0.1040 | [-0.1472, -0.0638] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0471 / -0.0584 | -0.0527 | [-0.0748, -0.0304] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.5930 / -0.7015 | -0.6473 | [-0.8661, -0.4623] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0631 / -0.1668 | -0.1149 | [-0.2624, +0.0058] | 0.0229 | OPEN |
| ade6_m|bulk | -0.7916 / -0.8267 | -0.8091 | [-1.0513, -0.6104] | 0.1243 | BETTER |
| ade6_m|all | -0.3953 / -0.4677 | -0.4315 | [-0.5774, -0.3082] | 0.0442 | BETTER |
| ade_tac_m|all | -0.5930 / -0.7015 | -0.6473 | [-0.8661, -0.4623] | 0.0699 | BETTER |

## I10RLcZ5 minus I10R0Z5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0232 / -0.0314 | -0.0273 | [-0.1046, +0.0569] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0559 / +0.0327 | +0.0443 | [-0.1592, +0.2811] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0630 / -0.0415 | -0.0523 | [-0.0831, -0.0268] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.3022 / -0.1145 | -0.2084 | [-0.4430, -0.0266] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0199 / -0.0182 | +0.0009 | [-0.0273, +0.0301] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.2459 / -0.8923 | +0.1768 | [-1.3199, +1.8142] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0125 / -0.0204 | -0.0165 | [-0.0362, +0.0042] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0121 / -0.0108 | -0.0115 | [-0.0194, -0.0040] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0088 / -0.0108 | -0.0098 | [-0.0260, +0.0071] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0099 / -0.0018 | +0.0040 | [-0.2598, +0.3042] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2080 / -0.2365 | -0.2222 | [-0.4117, -0.0234] | 0.0229 | BETTER |
| ade6_m|bulk | +0.3168 / +0.3329 | +0.3248 | [+0.0811, +0.6106] | 0.1243 | WORSE |
| ade6_m|all | +0.0313 / +0.0232 | +0.0273 | [-0.1648, +0.2511] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0099 / -0.0018 | +0.0040 | [-0.2598, +0.3042] | 0.0699 | OPEN |

## I10RQ50Z5 minus I10R0Z5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0309 / -0.0611 | -0.0460 | [-0.1147, +0.0255] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0986 / +0.0193 | +0.0589 | [-0.1102, +0.2518] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.0711 / -0.0427 | -0.0569 | [-0.0932, -0.0289] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0651 / -0.1247 | -0.0949 | [-0.3241, +0.1262] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0243 / -0.0146 | +0.0048 | [-0.0205, +0.0392] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.4235 / -0.7351 | +0.3442 | [-1.0345, +2.2386] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0379 / -0.0467 | -0.0423 | [-0.0613, -0.0239] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.0138 / -0.0096 | -0.0117 | [-0.0202, -0.0042] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0304 / -0.0288 | -0.0296 | [-0.0444, -0.0148] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.0438 / -0.0325 | +0.0057 | [-0.2052, +0.2446] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0889 / -0.2109 | -0.1499 | [-0.3297, +0.0452] | 0.0229 | OPEN |
| ade6_m|bulk | +0.2413 / +0.2599 | +0.2506 | [+0.0653, +0.4640] | 0.1243 | WORSE |
| ade6_m|all | +0.0617 / +0.0038 | +0.0327 | [-0.1239, +0.2068] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0438 / -0.0325 | +0.0057 | [-0.2052, +0.2446] | 0.0699 | OPEN |

## I10RLcZ5 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3377 / -0.3289 | -0.3333 | [-0.4340, -0.2302] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | +0.2640 / +0.4396 | +0.3518 | [+0.1584, +0.5726] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0761 / +0.0069 | +0.0415 | [-0.0432, +0.1286] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.1742 / -1.0827 | -0.6284 | [-1.5638, +0.3697] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0351 / -0.0330 | +0.0011 | [-0.0520, +0.0645] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -11.1314 / -14.5027 | -12.8170 | [-16.6345, -9.7226] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1221 / -0.1442 | -0.1332 | [-0.1815, -0.0872] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1363 / -0.1409 | -0.1386 | [-0.1639, -0.1138] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1351 / -0.1880 | -0.1615 | [-0.2165, -0.1136] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.4596 / +0.6355 | +0.5476 | [+0.2937, +0.8232] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6505 / +0.7641 | +0.7073 | [+0.4779, +0.9442] | 0.0229 | WORSE |
| ade6_m|bulk | -0.0877 / +0.0329 | -0.0274 | [-0.2088, +0.2207] | 0.1243 | OPEN |
| ade6_m|all | +0.3139 / +0.4306 | +0.3722 | [+0.2049, +0.5555] | 0.0442 | WORSE |
| ade_tac_m|all | +0.4596 / +0.6355 | +0.5476 | [+0.2937, +0.8232] | 0.0699 | WORSE |

## I10RQ50Z5 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3491 / -0.3693 | -0.3592 | [-0.4621, -0.2603] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | +0.2257 / +0.3082 | +0.2670 | [+0.0952, +0.4684] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0810 / +0.0048 | +0.0429 | [-0.0451, +0.1325] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.3677 / -1.2879 | -0.8278 | [-1.7443, +0.0826] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0388 / -0.0350 | +0.0019 | [-0.0547, +0.0657] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -10.7869 / -14.2586 | -12.5227 | [-16.0188, -9.4680] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1455 / -0.1676 | -0.1565 | [-0.2041, -0.1092] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1426 / -0.1492 | -0.1459 | [-0.1717, -0.1199] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1617 / -0.2047 | -0.1832 | [-0.2348, -0.1393] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.3965 / +0.4485 | +0.4225 | [+0.1698, +0.6931] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6627 / +0.7219 | +0.6923 | [+0.4696, +0.9260] | 0.0229 | WORSE |
| ade6_m|bulk | -0.1928 / -0.1894 | -0.1911 | [-0.3979, +0.0330] | 0.1243 | OPEN |
| ade6_m|all | +0.2726 / +0.3063 | +0.2895 | [+0.1210, +0.4688] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3965 / +0.4485 | +0.4225 | [+0.1698, +0.6931] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0B5_s0 | 3.1187 | 4.5621 | 1.3383 | 1.4113 | 0.5215 |
| I10R0B5_s1 | 3.0991 | 4.5415 | 1.3383 | 1.3964 | 0.5090 |
| I10R0Bk10_s0 | 2.2761 | 3.0182 | 1.1418 | 0.9350 | 0.5890 |
| I10R0Bk10_s1 | 2.2895 | 3.0546 | 1.1373 | 0.9748 | 0.6111 |
| I10R0Z5_s0 | 3.2327 | 4.1002 | 1.3365 | 1.4064 | 0.5219 |
| I10R0Z5_s1 | 3.2870 | 4.2413 | 1.3962 | 1.2121 | 0.5019 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcBk10_s0 | 2.4937 | 3.0953 | 1.2467 | 0.9333 | 0.5969 |
| I10RLcBk10_s1 | 2.4727 | 3.1124 | 1.2087 | 0.9843 | 0.6140 |
| I10RLcBk5_s0 | 3.3034 | 4.5036 | 1.3531 | 1.4083 | 0.5236 |
| I10RLcBk5_s1 | 3.2761 | 4.4774 | 1.3385 | 1.3842 | 0.5115 |
| I10RLcZ5_s0 | 3.2640 | 3.8923 | 1.3133 | 1.3433 | 0.5131 |
| I10RLcZ5_s1 | 3.3102 | 4.0049 | 1.3648 | 1.1706 | 0.4910 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50Bk10_s0 | 2.6265 | 3.2855 | 1.3129 | 0.9372 | 0.6061 |
| I10RQ50Bk10_s1 | 2.5168 | 3.1418 | 1.2311 | 0.9868 | 0.6194 |
| I10RQ50Bk5_s0 | 3.3889 | 4.6629 | 1.3378 | 1.4100 | 0.5152 |
| I10RQ50Bk5_s1 | 3.3029 | 4.5020 | 1.3314 | 1.3939 | 0.5090 |
| I10RQ50Z5_s0 | 3.2944 | 4.0113 | 1.3057 | 1.3352 | 0.4915 |
| I10RQ50Z5_s1 | 3.2908 | 4.0305 | 1.3352 | 1.1694 | 0.4731 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
