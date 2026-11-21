# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0CE5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0CE5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -9.1789 / -11.0302 | -10.1045 | [-11.4039, -8.7542] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -9.3715 / -12.0318 | -10.7016 | [-12.4009, -8.9618] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.5812 / -4.1140 | -3.8476 | [-4.4310, -3.3047] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -3.7055 / -3.6832 | -3.6944 | [-4.2412, -3.1990] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0362 / +0.0565 | +0.0463 | [+0.0248, +0.0749] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -22.9276 / -19.3592 | -21.1434 | [-25.3062, -16.9733] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0546 / -0.0496 | -0.0521 | [-0.0777, -0.0302] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2776 / -0.3877 | -0.3326 | [-0.4105, -0.2547] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0930 / -0.1025 | -0.0977 | [-0.1246, -0.0740] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -15.9012 / -20.2047 | -18.0530 | [-20.8887, -15.1420] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -11.3357 / -14.2996 | -12.8176 | [-14.7492, -10.8608] | 0.0229 | BETTER |
| ade6_m|bulk | -9.8009 / -12.5702 | -11.1855 | [-13.3282, -8.8790] | 0.1243 | BETTER |
| ade6_m|all | -10.6358 / -13.5109 | -12.0734 | [-13.9747, -10.1263] | 0.0442 | BETTER |
| ade_tac_m|all | -15.9012 / -20.2047 | -18.0530 | [-20.8887, -15.1420] | 0.0699 | BETTER |

## I10R0CE5 minus I10R0C5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -7.3739 / -10.0290 | -8.7015 | [-10.3817, -6.9831] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -6.2033 / -10.6189 | -8.4111 | [-10.9162, -5.9143] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.1856 / -3.8337 | -3.5097 | [-4.1250, -2.9019] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -3.8935 / -5.1647 | -4.5291 | [-5.5135, -3.4903] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0181 / +0.0342 | +0.0261 | [+0.0126, +0.0446] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -23.3999 / -22.2728 | -22.8363 | [-26.2581, -19.8579] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0409 / -0.0509 | -0.0459 | [-0.0710, -0.0256] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2676 / -0.4473 | -0.3574 | [-0.4720, -0.2491] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0809 / -0.1092 | -0.0950 | [-0.1265, -0.0686] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -11.4134 / -17.9603 | -14.6868 | [-18.5465, -10.7630] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -8.3445 / -12.7237 | -10.5341 | [-13.0956, -7.9733] | 0.0229 | BETTER |
| ade6_m|bulk | -6.7714 / -11.1365 | -8.9539 | [-11.7280, -6.0297] | 0.1243 | BETTER |
| ade6_m|all | -7.6271 / -11.9999 | -9.8135 | [-12.3916, -7.1909] | 0.0442 | BETTER |
| ade_tac_m|all | -11.4134 / -17.9603 | -14.6868 | [-18.5465, -10.7630] | 0.0699 | BETTER |

## I10R0CE5 minus I10R0E5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.5841 / -0.5113 | -0.5477 | [-0.6204, -0.4828] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -1.0133 / -0.8800 | -0.9467 | [-1.0865, -0.8295] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0247 / -0.0212 | -0.0230 | [-0.0624, +0.0171] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +1.3160 / +0.7684 | +1.0422 | [+0.4480, +1.7963] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0265 / +0.0311 | +0.0288 | [+0.0132, +0.0473] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +7.1336 / +4.3805 | +5.7570 | [+3.3992, +8.6177] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0450 / -0.0379 | -0.0415 | [-0.0669, -0.0196] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0333 / +0.0471 | +0.0402 | [+0.0254, +0.0569] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0292 / -0.0292 | -0.0292 | [-0.0527, -0.0083] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -1.4186 / -1.2654 | -1.3420 | [-1.5117, -1.1910] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -1.0144 / -0.7540 | -0.8842 | [-1.0857, -0.6964] | 0.0229 | BETTER |
| ade6_m|bulk | -0.8874 / -0.9736 | -0.9305 | [-1.0596, -0.7667] | 0.1243 | BETTER |
| ade6_m|all | -0.9565 / -0.8541 | -0.9053 | [-1.0199, -0.8036] | 0.0442 | BETTER |
| ade_tac_m|all | -1.4186 / -1.2654 | -1.3420 | [-1.5117, -1.1910] | 0.0699 | BETTER |

## I10R0CE5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.8914 / +0.9240 | +0.9077 | [+0.7547, +1.0562] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +2.3221 / +2.6353 | +2.4787 | [+2.1818, +2.7949] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.2441 / +0.1756 | +0.2098 | [+0.1079, +0.3133] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +1.7673 / +0.6669 | +1.2171 | [+0.1932, +2.4177] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0329 / -0.0396 | -0.0363 | [-0.0560, -0.0200] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -2.7995 / -6.3283 | -4.5639 | [-7.6389, -1.8655] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.2209 / +0.2022 | +0.2115 | [+0.1567, +0.2628] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.1205 / -0.0509 | -0.0857 | [-0.1408, -0.0343] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | +0.1722 / +0.1609 | +0.1665 | [+0.1189, +0.2140] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +3.4338 / +3.8177 | +3.6257 | [+3.2145, +4.0519] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +2.7274 / +2.9589 | +2.8432 | [+2.5378, +3.1717] | 0.0229 | WORSE |
| ade6_m|bulk | +1.8005 / +2.0907 | +1.9456 | [+1.5563, +2.3258] | 0.1243 | WORSE |
| ade6_m|all | +2.3048 / +2.5630 | +2.4339 | [+2.1571, +2.7198] | 0.0442 | WORSE |
| ade_tac_m|all | +3.4338 / +3.8177 | +3.6257 | [+3.2145, +4.0519] | 0.0699 | WORSE |

## I10R0CE10 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3103 / +0.4087 | +0.3595 | [+0.2360, +0.4849] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.7878 / +0.9137 | +0.8508 | [+0.7054, +0.9946] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0618 / -0.1053 | -0.0835 | [-0.1548, -0.0232] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +4.4205 / +3.7670 | +4.0938 | [+3.4104, +4.8103] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.2560 / +0.2909 | +0.2734 | [+0.2317, +0.3266] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +24.0406 / +23.8770 | +23.9588 | [+20.5462, +27.7386] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0909 / +0.1626 | +0.1267 | [+0.0736, +0.1827] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1509 / +0.1050 | +0.1280 | [+0.0890, +0.1660] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.1271 / +0.1780 | +0.1526 | [+0.1107, +0.1964] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.0290 / +1.1686 | +1.0988 | [+0.9014, +1.3060] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6918 / +0.6832 | +0.6875 | [+0.5505, +0.8215] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6627 / +0.8756 | +0.7692 | [+0.5409, +1.0157] | 0.1243 | WORSE |
| ade6_m|all | +0.6786 / +0.7710 | +0.7248 | [+0.5927, +0.8631] | 0.0442 | WORSE |
| ade_tac_m|all | +1.0290 / +1.1686 | +1.0988 | [+0.9014, +1.3060] | 0.0699 | WORSE |

## I10R0CE10 minus I10R0E10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.6097 / +0.7036 | +0.6567 | [+0.5575, +0.7609] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.7901 / +0.9062 | +0.8481 | [+0.7094, +0.9901] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0709 / +0.0602 | +0.0656 | [+0.0319, +0.0953] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +4.8663 / +3.9358 | +4.4011 | [+3.6203, +5.2252] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.2402 / +0.2638 | +0.2520 | [+0.2148, +0.3000] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +29.2000 / +29.5816 | +29.3908 | [+25.9939, +33.3541] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.1100 / +0.1867 | +0.1484 | [+0.0902, +0.2081] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1880 / +0.1163 | +0.1521 | [+0.0997, +0.2045] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.1517 / +0.1951 | +0.1734 | [+0.1350, +0.2158] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.1362 / +1.2863 | +1.2112 | [+1.0252, +1.4053] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.7727 / +0.7811 | +0.7769 | [+0.6582, +0.8949] | 0.0229 | WORSE |
| ade6_m|bulk | +0.7393 / +0.9487 | +0.8440 | [+0.6287, +1.0739] | 0.1243 | WORSE |
| ade6_m|all | +0.7575 / +0.8575 | +0.8075 | [+0.6834, +0.9369] | 0.0442 | WORSE |
| ade_tac_m|all | +1.1362 / +1.2863 | +1.2112 | [+1.0252, +1.4053] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0C5_s0 | 12.6667 | 14.1846 | 9.8621 | 4.7241 | 0.9000 |
| I10R0C5_s1 | 17.2535 | 18.8182 | 12.6009 | 5.1979 | 0.9416 |
| I10R0CE10_s0 | 3.4134 | 3.8045 | 1.9070 | 1.2327 | 0.7741 |
| I10R0CE10_s1 | 3.4616 | 3.8188 | 2.0566 | 1.0834 | 0.8495 |
| I10R0CE5_s0 | 5.0396 | 5.8402 | 2.4882 | 1.5385 | 0.8191 |
| I10R0CE5_s1 | 5.2535 | 6.0945 | 2.5719 | 1.3642 | 0.8324 |
| I10R0E10_s0 | 2.6559 | 3.0319 | 1.2973 | 1.1618 | 0.6223 |
| I10R0E10_s1 | 2.6041 | 3.0378 | 1.3530 | 1.0231 | 0.6544 |
| I10R0E5_s0 | 5.9961 | 6.8545 | 3.0722 | 1.5633 | 0.8483 |
| I10R0E5_s1 | 6.1077 | 6.8485 | 3.0832 | 1.3854 | 0.8616 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
