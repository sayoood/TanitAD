# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 22}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0E5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0E5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -8.5948 / -10.5190 | -9.5569 | [-10.8976, -8.1861] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -8.3582 / -11.1517 | -9.7549 | [-11.5171, -7.9623] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.5565 / -4.0928 | -3.8246 | [-4.4222, -3.2810] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -5.0215 / -4.4515 | -4.7365 | [-5.6156, -3.9938] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0097 / +0.0254 | +0.0175 | [+0.0058, +0.0340] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -30.0612 / -23.7396 | -26.9004 | [-33.7283, -21.1376] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0096 / -0.0117 | -0.0106 | [-0.0225, -0.0017] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.3110 / -0.4348 | -0.3729 | [-0.4577, -0.2865] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0638 / -0.0734 | -0.0686 | [-0.0863, -0.0513] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -14.4826 / -18.9393 | -16.7110 | [-19.6308, -13.7480] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -10.3213 / -13.5456 | -11.9334 | [-13.9682, -9.8307] | 0.0229 | BETTER |
| ade6_m|bulk | -8.9135 / -11.5965 | -10.2550 | [-12.3621, -8.0045] | 0.1243 | BETTER |
| ade6_m|all | -9.6793 / -12.6568 | -11.1680 | [-13.1183, -9.1862] | 0.0442 | BETTER |
| ade_tac_m|all | -14.4826 / -18.9393 | -16.7110 | [-19.6308, -13.7480] | 0.0699 | BETTER |

## I10RLcE5 minus I10RLcM5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -6.1825 / -7.5242 | -6.8533 | [-7.9133, -5.8037] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -5.8806 / -7.8240 | -6.8523 | [-8.2388, -5.4802] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -2.6289 / -2.8571 | -2.7430 | [-3.1972, -2.3478] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.6220 / -5.2357 | -4.9288 | [-5.6826, -4.1923] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0096 / +0.0179 | +0.0138 | [+0.0017, +0.0283] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -32.6500 / -29.9407 | -31.2954 | [-36.5642, -26.9534] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0208 / -0.0117 | -0.0163 | [-0.0336, +0.0002] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.2514 / -0.3710 | -0.3112 | [-0.3945, -0.2298] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0617 / -0.0792 | -0.0704 | [-0.0917, -0.0498] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -10.2987 / -13.3244 | -11.8115 | [-14.0799, -9.5398] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -7.1130 / -9.3541 | -8.2336 | [-9.8490, -6.6453] | 0.0229 | BETTER |
| ade6_m|bulk | -6.6194 / -8.3813 | -7.5004 | [-9.1919, -5.7894] | 0.1243 | BETTER |
| ade6_m|all | -6.8879 / -8.9105 | -7.8992 | [-9.4158, -6.3779] | 0.0442 | BETTER |
| ade_tac_m|all | -10.2987 / -13.3244 | -11.8115 | [-14.0799, -9.5398] | 0.0699 | BETTER |

## I10RQ50E5 minus I10RQ50M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -5.1516 / -6.0125 | -5.5821 | [-6.3572, -4.8046] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -5.0525 / -6.3565 | -5.7045 | [-6.7414, -4.6556] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -2.2294 / -2.3159 | -2.2726 | [-2.6685, -1.9266] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.6155 / -5.0995 | -4.8575 | [-5.5133, -4.1825] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0010 / +0.0102 | +0.0056 | [-0.0071, +0.0200] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -33.4050 / -29.5208 | -31.4629 | [-36.1267, -27.1655] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0246 / -0.0250 | -0.0248 | [-0.0421, -0.0079] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.2218 / -0.3322 | -0.2770 | [-0.3528, -0.2015] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0542 / -0.0917 | -0.0729 | [-0.1025, -0.0444] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -8.8296 / -10.8292 | -9.8294 | [-11.5501, -8.0946] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -6.0725 / -7.6807 | -6.8766 | [-8.1369, -5.5837] | 0.0229 | BETTER |
| ade6_m|bulk | -5.7157 / -6.7329 | -6.2243 | [-7.6152, -4.8335] | 0.1243 | BETTER |
| ade6_m|all | -5.9098 / -7.2485 | -6.5792 | [-7.7315, -5.4239] | 0.0442 | BETTER |
| ade_tac_m|all | -8.8296 / -10.8292 | -9.8294 | [-11.5501, -8.0946] | 0.0699 | BETTER |

## I10R0E5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.4755 / +1.4353 | +1.4554 | [+1.2861, +1.6163] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +3.3354 / +3.5154 | +3.4254 | [+3.1310, +3.7015] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.2688 / +0.1968 | +0.2328 | [+0.1313, +0.3349] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +0.4513 / -0.1015 | +0.1749 | [-0.5785, +1.0243] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0549 / -0.0709 | -0.0629 | [-0.0861, -0.0434] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -9.7776 / -10.7341 | -10.2559 | [-13.2468, -7.6351] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.2659 / +0.2401 | +0.2530 | [+0.2022, +0.3005] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.1538 / -0.0980 | -0.1259 | [-0.1753, -0.0830] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | +0.2013 / +0.1901 | +0.1957 | [+0.1530, +0.2363] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +4.8524 / +5.0831 | +4.9677 | [+4.5642, +5.3730] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.7418 / +3.7129 | +3.7274 | [+3.4135, +4.0280] | 0.0229 | WORSE |
| ade6_m|bulk | +2.6880 / +3.0643 | +2.8761 | [+2.3812, +3.3234] | 0.1243 | WORSE |
| ade6_m|all | +3.2612 / +3.4171 | +3.3392 | [+3.0662, +3.6135] | 0.0442 | WORSE |
| ade_tac_m|all | +4.8524 / +5.0831 | +4.9677 | [+4.5642, +5.3730] | 0.0699 | WORSE |

## I10RLcE5 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.1442 / +1.1416 | +1.1429 | [+0.9759, +1.3110] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +2.8511 / +3.0230 | +2.9371 | [+2.6364, +3.2275] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4348 / +0.3691 | +0.4019 | [+0.2752, +0.5402] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +2.4867 / +1.4780 | +1.9823 | [+1.0858, +3.0009] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0029 / -0.0340 | -0.0185 | [-0.0467, +0.0091] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.6482 / -2.6542 | -1.6512 | [-4.7474, +1.2853] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.1992 / +0.1772 | +0.1882 | [+0.1385, +0.2365] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.0642 / -0.0325 | -0.0484 | [-0.0898, -0.0136] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.1809 / +0.1605 | +0.1707 | [+0.1280, +0.2102] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +4.2827 / +4.5124 | +4.3975 | [+3.9499, +4.8162] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.0810 / +3.1069 | +3.0939 | [+2.7859, +3.4167] | 0.0229 | WORSE |
| ade6_m|bulk | +2.6518 / +2.9609 | +2.8064 | [+2.2932, +3.2538] | 0.1243 | WORSE |
| ade6_m|all | +2.8853 / +3.0403 | +2.9628 | [+2.6600, +3.2456] | 0.0442 | WORSE |
| ade_tac_m|all | +4.2827 / +4.5124 | +4.3975 | [+3.9499, +4.8162] | 0.0699 | WORSE |

## I10RQ50E5 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.0636 / +1.0549 | +1.0593 | [+0.9000, +1.2194] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +2.8479 / +2.9039 | +2.8759 | [+2.6058, +3.1496] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4744 / +0.4084 | +0.4414 | [+0.3102, +0.5703] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +3.0318 / +1.7150 | +2.3734 | [+1.3181, +3.5875] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0125 / -0.0217 | -0.0046 | [-0.0327, +0.0242] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +2.2315 / +0.0375 | +1.1345 | [-1.3585, +3.7024] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.1784 / +0.1363 | +0.1574 | [+0.1025, +0.2084] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.0363 / -0.0142 | -0.0252 | [-0.0596, +0.0062] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.1730 / +0.1446 | +0.1588 | [+0.1135, +0.2006] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +4.2797 / +4.3756 | +4.3276 | [+3.9259, +4.7193] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.0824 / +2.9494 | +3.0159 | [+2.7043, +3.3261] | 0.0229 | WORSE |
| ade6_m|bulk | +2.6505 / +2.9494 | +2.8000 | [+2.2866, +3.2416] | 0.1243 | WORSE |
| ade6_m|all | +2.8855 / +2.9494 | +2.9174 | [+2.6467, +3.1813] | 0.0442 | WORSE |
| ade_tac_m|all | +4.2797 / +4.3756 | +4.3276 | [+3.9259, +4.7193] | 0.0699 | WORSE |

## I10R0E10 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2995 / -0.2949 | -0.2972 | [-0.3721, -0.2337] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0023 / +0.0075 | +0.0026 | [-0.0230, +0.0288] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.1326 / -0.1655 | -0.1491 | [-0.2126, -0.1015] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -0.4458 / -0.1688 | -0.3073 | [-0.5923, +0.0011] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0157 / +0.0273 | +0.0215 | [+0.0121, +0.0353] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -5.1594 / -5.6531 | -5.4062 | [-7.2368, -3.8370] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0192 / -0.0242 | -0.0217 | [-0.0390, -0.0069] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0371 / -0.0113 | -0.0242 | [-0.0456, -0.0044] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0246 / -0.0171 | -0.0208 | [-0.0344, -0.0083] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1072 / -0.1176 | -0.1124 | [-0.1707, -0.0617] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0808 / -0.0978 | -0.0893 | [-0.1505, -0.0370] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0766 / -0.0730 | -0.0748 | [-0.1149, -0.0386] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.0789 / -0.0865 | -0.0827 | [-0.1218, -0.0473] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1072 / -0.1176 | -0.1124 | [-0.1707, -0.0617] | 0.0699 | BETTER |

## I10RLcE5 minus I10R0E5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2770 / -0.2479 | -0.2624 | [-0.3731, -0.1564] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2320 / -0.2758 | -0.2539 | [-0.5322, +0.0538] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.1388 / +0.1473 | +0.1430 | [+0.0363, +0.2606] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +2.6267 / +1.9464 | +2.2866 | [+1.5334, +3.0628] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0685 / +0.0555 | +0.0620 | [+0.0415, +0.0849] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +9.7825 / +7.9346 | +8.8586 | [+6.3629, +11.4425] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0700 / -0.0600 | -0.0650 | [-0.0867, -0.0440] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0950 / +0.0721 | +0.0836 | [+0.0550, +0.1122] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | -0.0192 / -0.0221 | -0.0206 | [-0.0340, -0.0065] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.2780 / -0.3182 | -0.2981 | [-0.7269, +0.1738] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.5317 / -0.5009 | -0.5163 | [-0.8365, -0.2023] | 0.0229 | BETTER |
| ade6_m|bulk | +0.2821 / +0.1856 | +0.2338 | [-0.1547, +0.6807] | 0.1243 | OPEN |
| ade6_m|all | -0.1606 / -0.1878 | -0.1742 | [-0.4739, +0.1618] | 0.0442 | OPEN |
| ade_tac_m|all | -0.2780 / -0.3182 | -0.2981 | [-0.7269, +0.1738] | 0.0699 | OPEN |

## I10RQ50E5 minus I10R0E5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3540 / -0.3238 | -0.3389 | [-0.4613, -0.2220] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1542 / -0.2771 | -0.2156 | [-0.5099, +0.1177] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.1654 / +0.1876 | +0.1765 | [+0.0541, +0.3029] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +3.4021 / +2.3795 | +2.8908 | [+1.9137, +3.9284] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0917 / +0.0716 | +0.0817 | [+0.0562, +0.1112] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +13.0774 / +10.4136 | +11.7455 | [+8.7452, +15.0231] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0930 / -0.1038 | -0.0984 | [-0.1238, -0.0728] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.1276 / +0.1000 | +0.1138 | [+0.0819, +0.1453] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | -0.0221 / -0.0392 | -0.0306 | [-0.0488, -0.0117] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.1839 / -0.2987 | -0.2413 | [-0.6984, +0.2536] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.4235 / -0.5905 | -0.5070 | [-0.8500, -0.1587] | 0.0229 | BETTER |
| ade6_m|bulk | +0.3105 / +0.3234 | +0.3169 | [-0.1046, +0.7910] | 0.1243 | OPEN |
| ade6_m|all | -0.0888 / -0.1738 | -0.1313 | [-0.4486, +0.2186] | 0.0442 | OPEN |
| ade_tac_m|all | -0.1839 / -0.2987 | -0.2413 | [-0.6984, +0.2536] | 0.0699 | OPEN |

## I10R0E5L minus I10R0E5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.9630 / -0.9037 | -0.9333 | [-1.0091, -0.8624] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -1.5669 / -1.5672 | -1.5670 | [-1.6865, -1.4516] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.2424 / -0.2584 | -0.2504 | [-0.3064, -0.1941] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -0.4833 / -0.7402 | -0.6117 | [-1.1022, -0.0786] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0514 / +0.0354 | +0.0434 | [+0.0232, +0.0683] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +0.3980 / -0.8549 | -0.2284 | [-1.5734, +1.3801] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0017 / -0.0183 | -0.0083 | [-0.0263, +0.0094] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0046 / -0.0121 | -0.0083 | [-0.0196, +0.0023] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0050 / -0.0250 | -0.0150 | [-0.0325, +0.0017] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -2.4544 / -2.4241 | -2.4393 | [-2.6266, -2.2625] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -1.8743 / -1.8042 | -1.8392 | [-1.9754, -1.6985] | 0.0229 | BETTER |
| ade6_m|bulk | -1.3870 / -1.4313 | -1.4091 | [-1.5859, -1.2409] | 0.1243 | BETTER |
| ade6_m|all | -1.6520 / -1.6342 | -1.6431 | [-1.7692, -1.5230] | 0.0442 | BETTER |
| ade_tac_m|all | -2.4544 / -2.4241 | -2.4393 | [-2.6266, -2.2625] | 0.0699 | BETTER |

## I10R0E5L minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -9.5577 / -11.4227 | -10.4902 | [-11.8201, -9.1234] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -9.9250 / -12.7189 | -11.3220 | [-13.1185, -9.4770] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.7989 / -4.3511 | -4.0750 | [-4.6917, -3.5017] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -5.5048 / -5.1917 | -5.3483 | [-6.0052, -4.7223] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0611 / +0.0608 | +0.0609 | [+0.0383, +0.0878] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -29.6632 / -24.5945 | -27.1289 | [-33.0382, -21.7912] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0079 / -0.0300 | -0.0190 | [-0.0417, -0.0029] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.3155 / -0.4469 | -0.3812 | [-0.4693, -0.2911] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0688 / -0.0984 | -0.0836 | [-0.1112, -0.0598] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -16.9370 / -21.3635 | -19.1502 | [-22.0841, -16.1481] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -12.1956 / -15.3498 | -13.7727 | [-15.8144, -11.7028] | 0.0229 | BETTER |
| ade6_m|bulk | -10.3004 / -13.0278 | -11.6641 | [-13.7881, -9.3883] | 0.1243 | BETTER |
| ade6_m|all | -11.3314 / -14.2909 | -12.8111 | [-14.7748, -10.8020] | 0.0442 | BETTER |
| ade_tac_m|all | -16.9370 / -21.3635 | -19.1502 | [-22.0841, -16.1481] | 0.0699 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0E10_s0 | 2.6559 | 3.0319 | 1.2973 | 1.1618 | 0.6223 |
| I10R0E10_s1 | 2.6041 | 3.0378 | 1.3530 | 1.0231 | 0.6544 |
| I10R0E5_s0 | 5.9961 | 6.8545 | 3.0722 | 1.5633 | 0.8483 |
| I10R0E5_s1 | 6.1077 | 6.8485 | 3.0832 | 1.3854 | 0.8616 |
| I10R0E5L_s0 | 4.3440 | 4.9803 | 2.1093 | 1.3209 | 0.8433 |
| I10R0E5L_s1 | 4.4735 | 5.0443 | 2.1795 | 1.1271 | 0.8366 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcE5_s0 | 5.8355 | 6.3228 | 2.7952 | 1.7020 | 0.8291 |
| I10RLcE5_s1 | 5.9199 | 6.3476 | 2.8353 | 1.5328 | 0.8395 |
| I10RLcM5_s0 | 12.7234 | 13.4358 | 8.9777 | 4.3309 | 0.8908 |
| I10RLcM5_s1 | 14.8304 | 15.7017 | 10.3595 | 4.3898 | 0.9187 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50E5_s0 | 5.9073 | 6.4310 | 2.7183 | 1.7286 | 0.8262 |
| I10RQ50E5_s1 | 5.9339 | 6.2580 | 2.7594 | 1.5731 | 0.8224 |
| I10RQ50M5_s0 | 11.8171 | 12.5035 | 7.8699 | 3.9580 | 0.8804 |
| I10RQ50M5_s1 | 13.1824 | 13.9387 | 8.7719 | 3.8890 | 0.9141 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
