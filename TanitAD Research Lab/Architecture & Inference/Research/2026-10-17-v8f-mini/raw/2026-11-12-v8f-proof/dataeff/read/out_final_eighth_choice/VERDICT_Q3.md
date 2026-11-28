# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 9}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "S1_8"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## S1 8 minus R1 8

Seeds read [0] of [0]. **Q3: COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1199 | +0.1199 | [-0.0565, +0.2960] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2885 | +0.2885 | [+0.0371, +0.5571] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0954 | -0.0954 | [-0.2223, +0.0217] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | +0.8050 | +0.8050 | [+0.0222, +1.6581] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0587 | +0.0587 | [+0.0264, +0.0966] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.2805 | +2.2805 | [-0.8485, +5.5454] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0800 | +0.0800 | [+0.0300, +0.1352] | 0.0746 (0.0329) | WORSE |
| TACTICAL:lat_err_rate | +0.0138 | +0.0138 | [-0.0234, +0.0509] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0779 | +0.0779 | [+0.0283, +0.1302] | 0.1205 (0.0438) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.2762 | +0.2762 | [-0.1160, +0.6626] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1966 | -0.1966 | [-0.4837, +0.0892] | 0.4093 | OPEN |
| ade6_m|bulk | +0.6405 | +0.6405 | [+0.2163, +1.0280] | 0.3850 | WORSE |
| ade6_m|all | +0.1851 | +0.1851 | [-0.0805, +0.4482] | 0.2845 | OPEN |
| ade_tac_m|all | +0.2762 | +0.2762 | [-0.1160, +0.6626] | 0.4169 | OPEN |

## R1 8 minus F1 8

Seeds read [0] of [0]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1582 | +0.1582 | [+0.0351, +0.2933] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1025 | +0.1025 | [-0.0664, +0.2922] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.2158 | -0.2158 | [-0.3170, -0.1154] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.6905 | -1.6905 | [-2.7322, -0.8078] | 2.5508 (1.1975) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0161 | -0.0161 | [-0.0442, +0.0082] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.5678 | -1.5678 | [-4.7790, +1.3096] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0438 | +0.0438 | [+0.0033, +0.0872] | 0.0746 (0.0329) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0542 | -0.0542 | [-0.0913, -0.0163] | 0.1305 (0.0500) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0079 | -0.0079 | [-0.0438, +0.0292] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.1088 | -0.1088 | [-0.3719, +0.1691] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2084 | -0.2084 | [-0.4506, +0.0441] | 0.4093 | OPEN |
| ade6_m|bulk | +0.0973 | +0.0973 | [-0.1226, +0.3167] | 0.3850 | OPEN |
| ade6_m|all | -0.0690 | -0.0690 | [-0.2464, +0.1168] | 0.2845 | OPEN |
| ade_tac_m|all | -0.1088 | -0.1088 | [-0.3719, +0.1691] | 0.4169 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| F1_8_s0 | 3.4737 | 4.2130 | 1.7430 | 1.7329 | 0.6444 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| R1_8_s0 | 3.4047 | 4.0046 | 1.9012 | 1.5171 | 0.6365 |
| S1_8_s0 | 3.5898 | 3.8079 | 2.0211 | 1.4217 | 0.7145 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
