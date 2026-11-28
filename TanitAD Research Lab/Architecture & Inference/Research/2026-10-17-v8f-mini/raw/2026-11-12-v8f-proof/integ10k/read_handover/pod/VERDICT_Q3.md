# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0C5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867476, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0C5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -1.8049 / -1.0013 | -1.4031 | [-2.0141, -0.7702] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -3.1681 / -1.4129 | -2.2905 | [-3.4344, -1.1995] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.3956 / -0.2803 | -0.3380 | [-0.5995, -0.0475] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | +0.1879 / +1.4815 | +0.8347 | [-0.1184, +1.7641] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0181 / +0.0223 | +0.0202 | [+0.0069, +0.0359] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.4723 / +2.9136 | +1.6929 | [-0.8287, +4.6933] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0138 / +0.0013 | -0.0063 | [-0.0223, +0.0033] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0100 / +0.0596 | +0.0248 | [-0.0236, +0.0709] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0121 / +0.0067 | -0.0027 | [-0.0208, +0.0108] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -4.4879 / -2.2444 | -3.3662 | [-4.9080, -1.8005] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -2.9912 / -1.5758 | -2.2835 | [-3.3548, -1.1354] | 0.0229 | BETTER |
| ade6_m|bulk | -3.0295 / -1.4337 | -2.2316 | [-3.4573, -1.1345] | 0.1243 | BETTER |
| ade6_m|all | -3.0087 / -1.5110 | -2.2598 | [-3.2914, -1.2122] | 0.0442 | BETTER |
| ade_tac_m|all | -4.4879 / -2.2444 | -3.3662 | [-4.9080, -1.8005] | 0.0699 | BETTER |

## I10R0B5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -10.3287 / -12.2638 | -11.2963 | [-12.5971, -9.9236] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -11.4800 / -14.4875 | -12.9837 | [-14.8646, -11.0420] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.7085 / -4.0818 | -3.8951 | [-4.4887, -3.3417] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.3032 / -4.0691 | -4.1862 | [-4.9515, -3.4797] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.1271 / +0.1209 | +0.1240 | [+0.0827, +0.1746] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -12.9525 / -8.6674 | -10.8100 | [-14.6130, -6.7261] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.3981 / -0.4173 | -0.4077 | [-0.4648, -0.3507] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.2772 / -0.4131 | -0.3451 | [-0.4398, -0.2503] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.3906 / -0.4260 | -0.4083 | [-0.4627, -0.3556] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -18.7587 / -23.4087 | -21.0837 | [-24.1006, -17.9283] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -12.6138 / -15.8526 | -14.2332 | [-16.3400, -12.1400] | 0.0229 | BETTER |
| ade6_m|bulk | -12.4886 / -15.4421 | -13.9653 | [-16.0936, -11.5123] | 0.1243 | BETTER |
| ade6_m|all | -12.5567 / -15.6654 | -14.1110 | [-16.1269, -11.9992] | 0.0442 | BETTER |
| ade_tac_m|all | -18.7587 / -23.4087 | -21.0837 | [-24.1006, -17.9283] | 0.0699 | BETTER |

## I10R0C5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +8.2653 / +10.9530 | +9.6092 | [+7.8963, +11.2888] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +8.5255 / +13.2542 | +10.8898 | [+8.2217, +13.5643] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +3.4297 / +4.0093 | +3.7195 | [+3.1328, +4.3153] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +5.6608 / +5.8316 | +5.7462 | [+4.8628, +6.6454] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0508 / -0.0740 | -0.0624 | [-0.0878, -0.0419] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +20.6062 / +15.9344 | +18.2703 | [+14.0661, +22.7003] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.2618 / +0.2530 | +0.2574 | [+0.2116, +0.3034] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1471 / +0.3964 | +0.2718 | [+0.1270, +0.4173] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.2530 / +0.2701 | +0.2616 | [+0.2234, +0.2982] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +14.8471 / +21.7780 | +18.3125 | [+14.2158, +22.3279] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +11.0719 / +15.6826 | +13.3773 | [+10.6737, +16.1573] | 0.0229 | WORSE |
| ade6_m|bulk | +8.5719 / +13.2272 | +10.8995 | [+7.8499, +13.7460] | 0.1243 | WORSE |
| ade6_m|all | +9.9319 / +14.5629 | +12.2474 | [+9.5104, +14.9318] | 0.0442 | WORSE |
| ade_tac_m|all | +14.8471 / +21.7780 | +18.3125 | [+14.2158, +22.3279] | 0.0699 | WORSE |

## I10R0B5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2585 / -0.3096 | -0.2841 | [-0.3931, -0.1832] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | +0.2136 / +0.1795 | +0.1966 | [+0.0519, +0.3386] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.1168 / +0.2078 | +0.1623 | [-0.0161, +0.3575] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +1.1700 / +0.2810 | +0.7255 | [-0.2611, +1.7209] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0622 / +0.0248 | +0.0435 | [+0.0008, +0.0941] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.2746 / +4.3536 | +5.8141 | [+1.4058, +10.6702] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.1226 / -0.1655 | -0.1440 | [-0.1973, -0.0932] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | -0.1201 / -0.0763 | -0.0982 | [-0.1451, -0.0544] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.1255 / -0.1626 | -0.1440 | [-0.1933, -0.0969] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | +0.5763 / +0.6137 | +0.5950 | [+0.3067, +0.8946] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.4494 / +1.4058 | +1.4276 | [+1.1863, +1.6637] | 0.0229 | WORSE |
| ade6_m|bulk | -0.8871 / -0.7813 | -0.8342 | [-1.0673, -0.6467] | 0.1243 | BETTER |
| ade6_m|all | +0.3839 / +0.4085 | +0.3962 | [+0.2006, +0.5989] | 0.0442 | WORSE |
| ade_tac_m|all | +0.5763 / +0.6137 | +0.5950 | [+0.3067, +0.8946] | 0.0699 | WORSE |

## I10R0C10 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.0764 / +1.5413 | +1.3089 | [+0.9861, +1.6691] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.9430 / +1.2349 | +1.0890 | [+0.8433, +1.3594] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.1549 / +0.0066 | +0.0808 | [-0.0308, +0.1842] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +5.4758 / +4.5174 | +4.9966 | [+4.1467, +5.8621] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.1983 / +0.2330 | +0.2157 | [+0.1798, +0.2631] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +40.2308 / +45.1434 | +42.6871 | [+36.9256, +49.3225] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.1109 / +0.1655 | +0.1382 | [+0.0927, +0.1869] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.1634 / +0.0934 | +0.1284 | [+0.0761, +0.1800] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.1380 / +0.1792 | +0.1586 | [+0.1209, +0.1983] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.4062 / +1.6960 | +1.5511 | [+1.2644, +1.8573] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.9102 / +0.9075 | +0.9088 | [+0.7760, +1.0363] | 0.0229 | WORSE |
| ade6_m|bulk | +0.9700 / +1.3969 | +1.1835 | [+0.8053, +1.6039] | 0.1243 | WORSE |
| ade6_m|all | +0.9375 / +1.1307 | +1.0341 | [+0.8429, +1.2382] | 0.0442 | WORSE |
| ade_tac_m|all | +1.4062 / +1.6960 | +1.5511 | [+1.2644, +1.8573] | 0.0699 | WORSE |

## I10R0C5 minus I10R0B5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +8.5238 / +11.2626 | +9.8932 | [+8.1399, +11.6003] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +8.3118 / +13.0746 | +10.6932 | [+8.0006, +13.3926] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +3.3129 / +3.8015 | +3.5572 | [+2.9536, +4.1694] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +4.4774 / +5.5506 | +5.0140 | [+3.9395, +6.0624] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.1089 / -0.0987 | -0.1038 | [-0.1525, -0.0632] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | +13.4297 / +11.5810 | +12.5054 | [+9.3350, +15.3880] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.3843 / +0.4185 | +0.4014 | [+0.3413, +0.4612] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.2672 / +0.4727 | +0.3699 | [+0.2412, +0.4992] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.3785 / +0.4327 | +0.4056 | [+0.3484, +0.4657] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +14.2708 / +21.1643 | +17.7176 | [+13.5609, +21.7758] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +9.6225 / +14.2768 | +11.9497 | [+9.1689, +14.7509] | 0.0229 | WORSE |
| ade6_m|bulk | +9.4591 / +14.0084 | +11.7337 | [+8.6628, +14.5638] | 0.1243 | WORSE |
| ade6_m|all | +9.5480 / +14.1544 | +11.8512 | [+9.0763, +14.5650] | 0.0442 | WORSE |
| ade_tac_m|all | +14.2708 / +21.1643 | +17.7176 | [+13.5609, +21.7758] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0B5_s0 | 3.1187 | 4.5621 | 1.3383 | 1.4113 | 0.5215 |
| I10R0B5_s1 | 3.0991 | 4.5415 | 1.3383 | 1.3964 | 0.5090 |
| I10R0C10_s0 | 3.6723 | 4.0229 | 2.6732 | 1.4494 | 0.7849 |
| I10R0C10_s1 | 3.8213 | 4.0431 | 3.1893 | 1.1953 | 0.8508 |
| I10R0C5_s0 | 12.6667 | 14.1846 | 9.8621 | 4.7241 | 0.9000 |
| I10R0C5_s1 | 17.2535 | 18.8182 | 12.6009 | 5.1979 | 0.9416 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
