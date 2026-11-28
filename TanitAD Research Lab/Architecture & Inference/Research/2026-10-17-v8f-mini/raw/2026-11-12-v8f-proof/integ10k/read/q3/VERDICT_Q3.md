# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RR"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RR minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2471 / +0.1902 | +0.2187 | [+0.0825, +0.3629] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.7974 / +0.7078 | +0.7526 | [+0.4611, +1.0507] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0157 / -0.0146 | -0.0152 | [-0.0498, +0.0186] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +1.1586 / +0.8251 | +0.9919 | [+0.6415, +1.3924] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0287 / +0.0328 | +0.0307 | [+0.0206, +0.0431] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.2975 / +2.9565 | +3.6270 | [+1.7790, +5.4347] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0079 / -0.0004 | +0.0038 | [-0.0094, +0.0181] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0254 / +0.0267 | +0.0261 | [+0.0094, +0.0440] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0179 / +0.0071 | +0.0125 | [-0.0002, +0.0271] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +1.0301 / +0.9076 | +0.9689 | [+0.5763, +1.3781] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4828 / +0.4395 | +0.4612 | [+0.2345, +0.7069] | 0.0229 | WORSE |
| ade6_m|bulk | +1.0444 / +0.9133 | +0.9789 | [+0.5388, +1.4366] | 0.1243 | WORSE |
| ade6_m|all | +0.7389 / +0.6556 | +0.6973 | [+0.4171, +0.9909] | 0.0442 | WORSE |
| ade_tac_m|all | +1.0301 / +0.9076 | +0.9689 | [+0.5763, +1.3781] | 0.0699 | WORSE |

## I10RR minus I10RLb

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1701 / +0.1229 | +0.1465 | [+0.0521, +0.2500] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4912 / +0.4199 | +0.4555 | [+0.2638, +0.6826] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0071 / +0.0084 | +0.0077 | [-0.0214, +0.0370] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.6013 / +0.4584 | +0.5298 | [+0.2669, +0.8484] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0191 / +0.0191 | +0.0191 | [+0.0103, +0.0286] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.6319 / +1.6152 | +2.1235 | [+0.5593, +3.6800] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0079 / -0.0054 | +0.0013 | [-0.0098, +0.0144] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0188 / +0.0175 | +0.0181 | [+0.0056, +0.0327] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0121 / -0.0008 | +0.0056 | [-0.0050, +0.0186] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.6539 / +0.5510 | +0.6025 | [+0.3444, +0.9099] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3422 / +0.3062 | +0.3242 | [+0.1548, +0.5195] | 0.0229 | WORSE |
| ade6_m|bulk | +0.6140 / +0.5033 | +0.5586 | [+0.2623, +0.9343] | 0.1243 | WORSE |
| ade6_m|all | +0.4661 / +0.3961 | +0.4311 | [+0.2475, +0.6480] | 0.0442 | WORSE |
| ade_tac_m|all | +0.6539 / +0.5510 | +0.6025 | [+0.3444, +0.9099] | 0.0699 | WORSE |

## I10RLb minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0770 / +0.0673 | +0.0721 | [-0.0126, +0.1691] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3062 / +0.2880 | +0.2971 | [+0.1038, +0.5317] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0228 / -0.0230 | -0.0229 | [-0.0427, -0.0023] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5849 / +0.4145 | +0.4997 | [+0.2736, +0.7683] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0108 / +0.0167 | +0.0137 | [+0.0069, +0.0229] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.5804 / +1.0971 | +1.3387 | [+0.3406, +2.4109] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0000 / +0.0050 | +0.0025 | [-0.0073, +0.0113] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0067 / +0.0092 | +0.0079 | [-0.0040, +0.0210] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0058 / +0.0079 | +0.0069 | [-0.0006, +0.0154] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.3762 / +0.3566 | +0.3664 | [+0.1133, +0.6690] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1406 / +0.1333 | +0.1370 | [+0.0134, +0.2967] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4304 / +0.4101 | +0.4202 | [+0.1435, +0.7461] | 0.1243 | WORSE |
| ade6_m|all | +0.2728 / +0.2595 | +0.2661 | [+0.0837, +0.4851] | 0.0442 | WORSE |
| ade_tac_m|all | +0.3762 / +0.3566 | +0.3664 | [+0.1133, +0.6690] | 0.0699 | WORSE |

## I10RR0 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.4788 / +1.5772 | +1.5280 | [+1.2311, +1.8532] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +4.1432 / +4.5564 | +4.3498 | [+3.6167, +5.1833] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.2232 / +0.1957 | +0.2095 | [+0.0939, +0.3297] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +5.9406 / +5.1422 | +5.5414 | [+4.5100, +6.5576] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.2342 / +0.2193 | +0.2267 | [+0.1834, +0.2734] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +29.5086 / +26.8254 | +28.1670 | [+23.5954, +32.5665] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0438 / +0.0534 | +0.0486 | [+0.0238, +0.0723] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | +0.0955 / +0.1067 | +0.1011 | [+0.0719, +0.1318] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.0721 / +0.0771 | +0.0746 | [+0.0515, +0.0973] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +5.6293 / +6.1483 | +5.8888 | [+4.8999, +7.0446] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.5508 / +3.5627 | +3.5568 | [+2.9302, +4.2210] | 0.0229 | WORSE |
| ade6_m|bulk | +4.8027 / +5.6405 | +5.2216 | [+4.0055, +6.6374] | 0.1243 | WORSE |
| ade6_m|all | +4.1217 / +4.5102 | +4.3160 | [+3.5953, +5.1441] | 0.0442 | WORSE |
| ade_tac_m|all | +5.6293 / +6.1483 | +5.8888 | [+4.8999, +7.0446] | 0.0699 | WORSE |

## I10RRid minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0512 (0.0512) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0229 | OPEN |
| ade6_m|bulk | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1243 | OPEN |
| ade6_m|all | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0442 | OPEN |
| ade_tac_m|all | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0699 | OPEN |

## I10RRx minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2242 / +0.2220 | +0.2231 | [+0.0932, +0.3549] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.7492 / +0.7437 | +0.7464 | [+0.4639, +1.0577] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | -0.0275 / +0.0163 | -0.0056 | [-0.0476, +0.0383] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +1.1312 / +0.8970 | +1.0141 | [+0.6575, +1.3852] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0356 / +0.0272 | +0.0314 | [+0.0202, +0.0442] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +4.1005 / +3.1596 | +3.6300 | [+1.7779, +5.3744] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0071 / -0.0008 | +0.0031 | [-0.0104, +0.0167] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0225 / +0.0317 | +0.0271 | [+0.0098, +0.0458] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0204 / +0.0133 | +0.0169 | [+0.0046, +0.0298] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.9390 / +0.9817 | +0.9604 | [+0.5825, +1.3613] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4469 / +0.4621 | +0.4545 | [+0.2297, +0.7037] | 0.0229 | WORSE |
| ade6_m|bulk | +0.9509 / +0.9981 | +0.9745 | [+0.5615, +1.4277] | 0.1243 | WORSE |
| ade6_m|all | +0.6767 / +0.7066 | +0.6916 | [+0.4233, +0.9832] | 0.0442 | WORSE |
| ade_tac_m|all | +0.9390 / +0.9817 | +0.9604 | [+0.5825, +1.3613] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10RLb_s0 | 3.0076 | 3.2533 | 1.6737 | 1.2717 | 0.6528 |
| I10RLb_s1 | 2.9501 | 3.2689 | 1.7153 | 1.1656 | 0.6794 |
| I10RR_s0 | 3.4738 | 3.5955 | 1.8438 | 1.2787 | 0.6649 |
| I10RR_s1 | 3.3462 | 3.5751 | 1.8382 | 1.1740 | 0.6786 |
| I10RR0_s0 | 6.8565 | 6.6635 | 3.0755 | 1.5176 | 0.7190 |
| I10RR0_s1 | 7.2008 | 6.6983 | 3.2252 | 1.3844 | 0.7486 |
| I10RRid_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10RRid_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10RRx_s0 | 3.4116 | 3.5596 | 1.8210 | 1.2669 | 0.6674 |
| I10RRx_s1 | 3.3971 | 3.5978 | 1.8700 | 1.2050 | 0.6849 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
