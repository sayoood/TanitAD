# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0V5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867476, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0V5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -9.5926 / -11.3966 | -10.4946 | [-11.7013, -9.2334] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -10.6634 / -13.2586 | -11.9610 | [-13.6080, -10.2499] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.3112 / -3.9453 | -3.6282 | [-4.2538, -3.0557] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.1166 / -3.3388 | -3.7277 | [-4.6045, -2.8741] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0479 / +0.0372 | +0.0426 | [+0.0259, +0.0622] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -20.8326 / -18.2289 | -19.5308 | [-23.5117, -15.8071] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.2497 / -0.1546 | -0.2022 | [-0.2735, -0.1391] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2714 / -0.4335 | -0.3524 | [-0.4559, -0.2458] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.2730 / -0.2088 | -0.2409 | [-0.2935, -0.1922] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -17.4171 / -21.7918 | -19.6045 | [-22.3566, -16.7203] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -12.2713 / -15.2694 | -13.7704 | [-15.7198, -11.8120] | 0.0229 | BETTER |
| ade6_m|bulk | -10.9283 / -13.7684 | -12.3484 | [-14.2916, -10.0756] | 0.1243 | BETTER |
| ade6_m|all | -11.6589 / -14.5849 | -13.1219 | [-14.9627, -11.1989] | 0.0442 | BETTER |
| ade_tac_m|all | -17.4171 / -21.7918 | -19.6045 | [-22.3566, -16.7203] | 0.0699 | BETTER |

## I10R0V5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4777 / +0.5576 | +0.5176 | [+0.3813, +0.6767] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.0302 / +1.4085 | +1.2194 | [+0.9250, +1.5696] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.5141 / +0.3443 | +0.4292 | [+0.2883, +0.5811] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +1.3562 / +1.0112 | +1.1837 | [+0.4349, +1.9572] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0167 / -0.0590 | -0.0379 | [-0.0739, -0.0064] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -0.5511 / -5.2238 | -2.8874 | [-6.9320, +0.5200] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0258 / +0.0971 | +0.0615 | [-0.0017, +0.1266] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.1142 / -0.0967 | -0.1055 | [-0.1337, -0.0812] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0079 / +0.0546 | +0.0233 | [-0.0298, +0.0804] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +1.9179 / +2.2305 | +2.0742 | [+1.7050, +2.4964] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.7918 / +1.9890 | +1.8904 | [+1.5972, +2.1992] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6731 / +0.8924 | +0.7828 | [+0.4942, +1.1784] | 0.1243 | WORSE |
| ade6_m|all | +1.2817 / +1.4890 | +1.3853 | [+1.1404, +1.6660] | 0.0442 | WORSE |
| ade_tac_m|all | +1.9179 / +2.2305 | +2.0742 | [+1.7050, +2.4964] | 0.0699 | WORSE |

## I10R0V10 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3780 / -0.4402 | -0.4091 | [-0.5136, -0.3077] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2350 / -0.2533 | -0.2442 | [-0.3566, -0.1502] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | +0.0792 / -0.0296 | +0.0248 | [-0.0618, +0.1082] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.9302 / +1.0377 | +0.9839 | [+0.6291, +1.4364] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0722 / +0.0840 | +0.0781 | [+0.0592, +0.1028] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +2.2367 / +0.0327 | +1.1347 | [-1.6018, +3.7414] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.1146 / -0.0971 | -0.1059 | [-0.1395, -0.0729] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0071 / -0.0138 | -0.0033 | [-0.0327, +0.0244] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0546 / -0.0992 | -0.0769 | [-0.1176, -0.0402] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.1429 / -0.3172 | -0.2300 | [-0.4016, -0.0665] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0435 / -0.1461 | -0.0948 | [-0.2069, +0.0018] | 0.0229 | OPEN |
| ade6_m|bulk | -0.1569 / -0.2895 | -0.2232 | [-0.4048, -0.0570] | 0.1243 | BETTER |
| ade6_m|all | -0.0952 / -0.2115 | -0.1534 | [-0.2677, -0.0443] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1429 / -0.3172 | -0.2300 | [-0.4016, -0.0665] | 0.0699 | BETTER |

## I10R0V5 minus I10R0W5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3486 / -0.4957 | -0.4221 | [-0.5664, -0.2769] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.3289 / -0.4936 | -0.4112 | [-0.5805, -0.2143] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0097 / -0.1146 | -0.0621 | [-0.1464, +0.0176] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.0600 / +0.3810 | +0.1605 | [-0.2701, +0.5483] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0187 / +0.0127 | +0.0157 | [+0.0081, +0.0258] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.0153 / -0.9047 | -0.9600 | [-2.4558, +0.4157] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0504 / -0.0763 | -0.0634 | [-0.0852, -0.0411] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.0067 / +0.0079 | +0.0006 | [-0.0123, +0.0136] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0500 / -0.0696 | -0.0598 | [-0.0784, -0.0419] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.4306 / -0.7727 | -0.6016 | [-0.8958, -0.2827] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.4403 / -0.6768 | -0.5586 | [-0.7678, -0.3828] | 0.0229 | BETTER |
| ade6_m|bulk | -0.1042 / -0.3249 | -0.2146 | [-0.4558, +0.1138] | 0.1243 | OPEN |
| ade6_m|all | -0.2871 / -0.5163 | -0.4017 | [-0.5985, -0.1876] | 0.0442 | BETTER |
| ade_tac_m|all | -0.4306 / -0.7727 | -0.6016 | [-0.8958, -0.2827] | 0.0699 | BETTER |

## I10R0V5 minus I10R0Z5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.7379 / +0.8093 | +0.7736 | [+0.6395, +0.9203] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.5698 / +0.7852 | +0.6775 | [+0.5025, +0.8657] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4022 / +0.3208 | +0.3615 | [+0.2666, +0.4678] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +0.5442 / +1.6483 | +1.0962 | [+0.1960, +1.9511] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0545 / -0.0574 | -0.0559 | [-0.0936, -0.0241] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +10.8920 / +9.1031 | +9.9976 | [+7.4727, +12.8014] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.1388 / +0.2180 | +0.1784 | [+0.1176, +0.2447] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.0046 / +0.0267 | +0.0156 | [-0.0044, +0.0346] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.1171 / +0.2243 | +0.1707 | [+0.0969, +0.2474] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.1764 / +1.3408 | +1.2586 | [+1.0450, +1.4977] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.8043 / +0.8833 | +0.8438 | [+0.6749, +1.0209] | 0.0229 | WORSE |
| ade6_m|bulk | +0.7593 / +0.9035 | +0.8314 | [+0.6482, +1.0578] | 0.1243 | WORSE |
| ade6_m|all | +0.7838 / +0.8925 | +0.8382 | [+0.6955, +0.9982] | 0.0442 | WORSE |
| ade_tac_m|all | +1.1764 / +1.3408 | +1.2586 | [+1.0450, +1.4977] | 0.0699 | WORSE |

## I10R0V10 minus I10R0W10

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3331 / -0.4196 | -0.3764 | [-0.4758, -0.2961] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2363 / -0.3694 | -0.3029 | [-0.4134, -0.2082] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0759 / -0.0543 | -0.0651 | [-0.0929, -0.0363] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.6186 / -1.6339 | -1.6262 | [-1.9503, -1.3436] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0465 / -0.1049 | -0.0757 | [-0.1246, -0.0374] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -12.9494 / -15.5049 | -14.2272 | [-17.5108, -11.6257] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0725 / -0.1138 | -0.0932 | [-0.1265, -0.0617] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.0504 / -0.0229 | -0.0367 | [-0.0584, -0.0139] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0596 / -0.0921 | -0.0759 | [-0.1045, -0.0511] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.3613 / -0.5046 | -0.4330 | [-0.5533, -0.3232] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.3301 / -0.3716 | -0.3509 | [-0.4132, -0.2893] | 0.0229 | BETTER |
| ade6_m|bulk | -0.1344 / -0.2944 | -0.2144 | [-0.3416, -0.1127] | 0.1243 | BETTER |
| ade6_m|all | -0.2408 / -0.3364 | -0.2886 | [-0.3689, -0.2155] | 0.0442 | BETTER |
| ade_tac_m|all | -0.3613 / -0.5046 | -0.4330 | [-0.5533, -0.3232] | 0.0699 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |
| I10R0V10_s0 | 2.6396 | 3.0692 | 1.2187 | 1.3736 | 0.5923 |
| I10R0V10_s1 | 2.4791 | 2.9895 | 1.2078 | 1.1591 | 0.5723 |
| I10R0V5_s0 | 4.0165 | 4.9045 | 2.0744 | 1.8085 | 0.6390 |
| I10R0V5_s1 | 4.1795 | 5.1246 | 2.2055 | 1.5330 | 0.7261 |
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
