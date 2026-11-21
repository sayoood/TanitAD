# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0W5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.1536295382986748, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0W5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -9.2440 / -10.9010 | -10.0725 | [-11.1699, -8.9035] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -10.3345 / -12.7650 | -11.5497 | [-13.1303, -9.9095] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.3015 / -3.8307 | -3.5661 | [-4.1395, -3.0483] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.0566 / -3.7199 | -3.8883 | [-4.5720, -3.1732] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0292 / +0.0245 | +0.0269 | [+0.0144, +0.0404] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -19.8173 / -17.3242 | -18.5708 | [-21.8319, -15.2663] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1992 / -0.0784 | -0.1388 | [-0.2215, -0.0669] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2647 / -0.4414 | -0.3531 | [-0.4631, -0.2408] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.2230 / -0.1392 | -0.1811 | [-0.2424, -0.1261] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -16.9866 / -21.0192 | -19.0029 | [-21.5987, -16.3046] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -11.8310 / -14.5926 | -13.2118 | [-15.0539, -11.3873] | 0.0229 | BETTER |
| ade6_m|bulk | -10.8241 / -13.4434 | -12.1338 | [-13.9796, -9.9779] | 0.1243 | BETTER |
| ade6_m|all | -11.3718 / -14.0686 | -12.7202 | [-14.4516, -10.9152] | 0.0442 | BETTER |
| ade_tac_m|all | -16.9866 / -21.0192 | -19.0029 | [-21.5987, -16.3046] | 0.0699 | BETTER |

## I10R0Z5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -10.3305 / -12.2060 | -11.2682 | [-12.5744, -9.9239] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -11.2332 / -14.0437 | -12.6384 | [-14.3981, -10.7950] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.7134 / -4.2661 | -3.9897 | [-4.6058, -3.4454] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.6608 / -4.9871 | -4.8240 | [-5.5190, -4.1536] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.1024 / +0.0946 | +0.0985 | [+0.0580, +0.1491] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -31.7246 / -27.3321 | -29.5283 | [-35.4138, -24.2411] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.3885 / -0.3727 | -0.3806 | [-0.4300, -0.3278] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2759 / -0.4602 | -0.3681 | [-0.4825, -0.2538] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.3902 / -0.4331 | -0.4116 | [-0.4649, -0.3580] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -18.5936 / -23.1326 | -20.8631 | [-23.7222, -17.8354] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -13.0757 / -16.1527 | -14.6142 | [-16.5794, -12.5989] | 0.0229 | BETTER |
| ade6_m|bulk | -11.6876 / -14.6719 | -13.1798 | [-15.2529, -10.8119] | 0.1243 | BETTER |
| ade6_m|all | -12.4427 / -15.4774 | -13.9601 | [-15.8755, -11.9332] | 0.0442 | BETTER |
| ade_tac_m|all | -18.5936 / -23.1326 | -20.8631 | [-23.7222, -17.8354] | 0.0699 | BETTER |

## I10R0W5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.8263 / +1.0533 | +0.9398 | [+0.6979, +1.1893] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.3591 / +1.9021 | +1.6306 | [+1.2226, +2.0719] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.5238 / +0.4589 | +0.4913 | [+0.3867, +0.6001] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +1.4162 / +0.6302 | +1.0232 | [+0.1235, +2.0358] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0354 / -0.0716 | -0.0535 | [-0.0865, -0.0273] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +0.4730 / -4.3060 | -1.9165 | [-5.8209, +1.7573] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0763 / +0.1734 | +0.1248 | [+0.0496, +0.2053] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.1075 / -0.1046 | -0.1061 | [-0.1308, -0.0842] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | +0.0421 / +0.1242 | +0.0832 | [+0.0204, +0.1506] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +2.3484 / +3.0032 | +2.6758 | [+2.1263, +3.2743] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +2.2321 / +2.6658 | +2.4490 | [+2.0457, +2.9015] | 0.0229 | WORSE |
| ade6_m|bulk | +0.7773 / +1.2174 | +0.9974 | [+0.5929, +1.4280] | 0.1243 | WORSE |
| ade6_m|all | +1.5687 / +2.0053 | +1.7870 | [+1.4211, +2.1861] | 0.0442 | WORSE |
| ade_tac_m|all | +2.3484 / +3.0032 | +2.6758 | [+2.1263, +3.2743] | 0.0699 | WORSE |

## I10R0Z5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2602 / -0.2517 | -0.2560 | [-0.3690, -0.1535] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | +0.4604 / +0.6233 | +0.5419 | [+0.3567, +0.7589] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.1119 / +0.0235 | +0.0677 | [-0.0263, +0.1582] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.8121 / -0.6371 | +0.0875 | [-1.1133, +1.4520] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0376 / -0.0016 | +0.0180 | [-0.0226, +0.0684] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -11.4457 / -14.3217 | -12.8837 | [-16.6829, -9.8118] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1130 / -0.1209 | -0.1169 | [-0.1641, -0.0713] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1188 / -0.1234 | -0.1211 | [-0.1461, -0.0973] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1251 / -0.1697 | -0.1474 | [-0.1991, -0.1018] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.7414 / +0.8898 | +0.8156 | [+0.5762, +1.0930] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.9875 / +1.1057 | +1.0466 | [+0.8034, +1.2924] | 0.0229 | WORSE |
| ade6_m|bulk | -0.0862 / -0.0111 | -0.0486 | [-0.2152, +0.1679] | 0.1243 | OPEN |
| ade6_m|all | +0.4979 / +0.5964 | +0.5472 | [+0.3866, +0.7301] | 0.0442 | WORSE |
| ade_tac_m|all | +0.7414 / +0.8898 | +0.8156 | [+0.5762, +1.0930] | 0.0699 | WORSE |

## I10R0W10 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0449 / -0.0206 | -0.0327 | [-0.1426, +0.0650] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0013 / +0.1161 | +0.0587 | [-0.0717, +0.1716] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.1551 / +0.0247 | +0.0899 | [-0.0116, +0.1849] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +2.5488 / +2.6716 | +2.6102 | [+2.1315, +3.1899] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1187 / +0.1889 | +0.1538 | [+0.1061, +0.2155] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +15.1862 / +15.5376 | +15.3619 | [+11.9071, +19.0542] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0421 / +0.0167 | -0.0127 | [-0.0594, +0.0388] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0575 / +0.0092 | +0.0333 | [-0.0073, +0.0714] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0050 / -0.0071 | -0.0010 | [-0.0277, +0.0238] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.2184 / +0.1874 | +0.2029 | [+0.0715, +0.3228] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2866 / +0.2255 | +0.2560 | [+0.1423, +0.3584] | 0.0229 | WORSE |
| ade6_m|bulk | -0.0225 / +0.0049 | -0.0088 | [-0.1546, +0.1071] | 0.1243 | OPEN |
| ade6_m|all | +0.1456 / +0.1249 | +0.1353 | [+0.0477, +0.2152] | 0.0442 | WORSE |
| ade_tac_m|all | +0.2184 / +0.1874 | +0.2029 | [+0.0715, +0.3228] | 0.0699 | WORSE |

## I10R0W5 minus I10R0Z5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.0865 / +1.3050 | +1.1957 | [+0.9515, +1.4561] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.8987 / +1.2788 | +1.0887 | [+0.7779, +1.3871] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4119 / +0.4354 | +0.4236 | [+0.3273, +0.5305] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +0.6041 / +1.2673 | +0.9357 | [+0.2447, +1.5551] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0732 / -0.0701 | -0.0716 | [-0.1127, -0.0379] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +11.9073 / +10.0078 | +10.9576 | [+8.0643, +14.2261] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.1892 / +0.2943 | +0.2418 | [+0.1641, +0.3228] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.0113 / +0.0188 | +0.0150 | [+0.0021, +0.0285] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.1672 / +0.2939 | +0.2305 | [+0.1458, +0.3217] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.6070 / +2.1135 | +1.8602 | [+1.4122, +2.3142] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.2446 / +1.5601 | +1.4024 | [+1.1063, +1.7255] | 0.0229 | WORSE |
| ade6_m|bulk | +0.8635 / +1.2285 | +1.0460 | [+0.6489, +1.4151] | 0.1243 | WORSE |
| ade6_m|all | +1.0708 / +1.4089 | +1.2399 | [+0.9402, +1.5426] | 0.0442 | WORSE |
| ade_tac_m|all | +1.6070 / +2.1135 | +1.8602 | [+1.4122, +2.3142] | 0.0699 | WORSE |

## I10R0W5 minus I10R0C5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -7.4391 / -9.8997 | -8.6694 | [-10.1709, -7.1327] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -7.1664 / -11.3521 | -9.2592 | [-11.6667, -6.9113] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -2.9059 / -3.5504 | -3.2281 | [-3.8441, -2.6314] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.2446 / -5.2014 | -4.7230 | [-5.6453, -3.7273] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0111 / +0.0022 | +0.0067 | [-0.0110, +0.0234] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -20.2896 / -20.2378 | -20.2637 | [-23.1878, -17.5911] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.1855 / -0.0796 | -0.1326 | [-0.2066, -0.0684] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2547 / -0.5010 | -0.3779 | [-0.5246, -0.2335] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.2109 / -0.1459 | -0.1784 | [-0.2296, -0.1326] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -12.4987 / -18.7747 | -15.6367 | [-19.2777, -11.9096] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -8.8398 / -13.0168 | -10.9283 | [-13.4336, -8.4996] | 0.0229 | BETTER |
| ade6_m|bulk | -7.7946 / -12.0098 | -9.9022 | [-12.4804, -7.1125] | 0.1243 | BETTER |
| ade6_m|all | -8.3632 / -12.5576 | -10.4604 | [-12.8940, -7.9695] | 0.0442 | BETTER |
| ade_tac_m|all | -12.4987 / -18.7747 | -15.6367 | [-19.2777, -11.9096] | 0.0699 | BETTER |

## I10R0Z5 minus I10R0C5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -8.5256 / -11.2047 | -9.8651 | [-11.5484, -8.1323] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -8.0650 / -12.6308 | -10.3479 | [-12.9341, -7.7718] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.3178 / -3.9858 | -3.6518 | [-4.3147, -3.0322] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.8487 / -6.4687 | -5.6587 | [-6.9917, -4.2819] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0843 / +0.0723 | +0.0783 | [+0.0392, +0.1262] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -32.1969 / -30.2456 | -31.2213 | [-36.3046, -26.6722] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.3747 / -0.3739 | -0.3743 | [-0.4245, -0.3246] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2659 / -0.5198 | -0.3929 | [-0.5435, -0.2461] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.3781 / -0.4398 | -0.4089 | [-0.4706, -0.3495] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -14.1057 / -20.8882 | -17.4969 | [-21.4332, -13.4832] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -10.0844 / -14.5769 | -12.3307 | [-15.0097, -9.6912] | 0.0229 | BETTER |
| ade6_m|bulk | -8.6581 / -13.2382 | -10.9482 | [-13.7632, -7.9483] | 0.1243 | BETTER |
| ade6_m|all | -9.4340 / -13.9664 | -11.7002 | [-14.3296, -9.0173] | 0.0442 | BETTER |
| ade_tac_m|all | -14.1057 / -20.8882 | -17.4969 | [-21.4332, -13.4832] | 0.0699 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0C5_s0 | 12.6667 | 14.1846 | 9.8621 | 4.7241 | 0.9000 |
| I10R0C5_s1 | 17.2535 | 18.8182 | 12.6009 | 5.1979 | 0.9416 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |
| I10R0W10_s0 | 2.8804 | 3.3993 | 1.5519 | 1.4495 | 0.6519 |
| I10R0W10_s1 | 2.8155 | 3.3612 | 1.6274 | 1.2134 | 0.6644 |
| I10R0W5_s0 | 4.3035 | 5.3448 | 2.4230 | 1.8182 | 0.6890 |
| I10R0W5_s1 | 4.6959 | 5.8014 | 2.7012 | 1.6475 | 0.7957 |
| I10R0Z5_s0 | 3.2327 | 4.1002 | 1.3365 | 1.4064 | 0.5219 |
| I10R0Z5_s1 | 3.2870 | 4.2413 | 1.3962 | 1.2121 | 0.5019 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
