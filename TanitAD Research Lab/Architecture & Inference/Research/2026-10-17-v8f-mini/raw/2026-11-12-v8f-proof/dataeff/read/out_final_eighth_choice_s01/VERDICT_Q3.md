# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "S1_8"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## S1 8 minus R1 8

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1199 / +0.2508 | +0.1854 | [+0.0361, +0.3288] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.2885 / +0.2229 | +0.2557 | [+0.0815, +0.4396] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0954 / +0.0596 | -0.0179 | [-0.1467, +0.1061] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | +0.8050 / +1.4911 | +1.1481 | [+0.4185, +1.9848] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0587 / +0.0398 | +0.0492 | [+0.0241, +0.0769] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.2805 / +3.2199 | +2.7502 | [+0.6641, +4.9843] | 10.6485 (4.6266) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0800 / +0.0521 | +0.0661 | [+0.0287, +0.1042] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0138 / +0.0909 | +0.0523 | [-0.0019, +0.1111] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0779 / +0.0617 | +0.0698 | [+0.0365, +0.1057] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.2762 / +0.3568 | +0.3165 | [+0.0533, +0.5673] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1966 / -0.1479 | -0.1722 | [-0.3719, +0.0245] | 0.4093 | OPEN |
| ade6_m|bulk | +0.6405 / +0.6958 | +0.6681 | [+0.4101, +0.9193] | 0.3850 | WORSE |
| ade6_m|all | +0.1851 / +0.2369 | +0.2110 | [+0.0334, +0.3795] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2762 / +0.3568 | +0.3165 | [+0.0533, +0.5673] | 0.4169 | INSIDE-SPREAD |

## R1 8 minus F1 8

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1582 / +0.0196 | +0.0889 | [-0.0320, +0.2197] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1025 / +0.0789 | +0.0907 | [-0.0320, +0.2139] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.2158 / -0.1645 | -0.1901 | [-0.2716, -0.1059] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.6905 / -1.0558 | -1.3731 | [-2.1520, -0.7159] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0161 / -0.0050 | -0.0106 | [-0.0316, +0.0082] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.5678 / -0.7834 | -1.1756 | [-3.4043, +0.9126] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0438 / +0.0496 | +0.0467 | [+0.0158, +0.0753] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0542 / -0.0484 | -0.0513 | [-0.0761, -0.0250] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0079 / +0.0150 | +0.0035 | [-0.0255, +0.0344] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.1088 / -0.0890 | -0.0989 | [-0.2742, +0.0827] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2084 / -0.1349 | -0.1717 | [-0.3466, +0.0027] | 0.4093 | OPEN |
| ade6_m|bulk | +0.0973 / +0.0428 | +0.0700 | [-0.0745, +0.2332] | 0.3850 | OPEN |
| ade6_m|all | -0.0690 / -0.0539 | -0.0614 | [-0.1798, +0.0621] | 0.2845 | OPEN |
| ade_tac_m|all | -0.1088 / -0.0890 | -0.0989 | [-0.2742, +0.0827] | 0.4169 | OPEN |

## S1 8 minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0213 / +0.2881 | +0.1547 | [-0.0692, +0.3556] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3339 / +0.6696 | +0.5017 | [+0.2130, +0.7674] | 0.4550 (0.1815) | WORSE |
| LATERAL:cross_mae_m | +0.1464 / -0.0546 | +0.0459 | [-0.1000, +0.1955] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -1.8854 / +0.2428 | -0.8213 | [-2.2260, +0.6700] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0709 / +0.0369 | -0.0170 | [-0.0864, +0.0518] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -9.0091 / +0.5386 | -4.2352 | [-10.4070, +1.8400] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0233 / +0.0054 | +0.0144 | [-0.0192, +0.0540] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0267 / -0.0638 | -0.0452 | [-0.0886, -0.0081] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0188 / -0.0358 | -0.0085 | [-0.0542, +0.0450] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | +0.5995 / +0.8743 | +0.7369 | [+0.4282, +1.0278] | 0.4169 (0.1897) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1047 / +0.4131 | +0.2589 | [+0.0005, +0.5132] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | +0.7450 / +0.7907 | +0.7678 | [+0.5153, +1.0187] | 0.3850 | WORSE |
| ade6_m|all | +0.3967 / +0.5853 | +0.4910 | [+0.2804, +0.6879] | 0.2845 | WORSE |
| ade_tac_m|all | +0.5995 / +0.8743 | +0.7369 | [+0.4282, +1.0278] | 0.4169 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| F1_8_s0 | 3.4737 | 4.2130 | 1.7430 | 1.7329 | 0.6444 |
| F1_8_s1 | 3.3792 | 4.1211 | 1.7926 | 1.5380 | 0.6186 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| R1_8_s0 | 3.4047 | 4.0046 | 1.9012 | 1.5171 | 0.6365 |
| R1_8_s1 | 3.3253 | 3.9861 | 1.8123 | 1.3735 | 0.6336 |
| S1_8_s0 | 3.5898 | 3.8079 | 2.0211 | 1.4217 | 0.7145 |
| S1_8_s1 | 3.5622 | 3.8383 | 2.0631 | 1.4332 | 0.6953 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
