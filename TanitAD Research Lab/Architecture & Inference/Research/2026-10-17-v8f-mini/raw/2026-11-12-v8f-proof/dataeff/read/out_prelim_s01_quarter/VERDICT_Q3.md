# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD13. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 6}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "R1_4"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.6745357826426672, "D_read": 0.6745357826426673, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): False. A difference inside the spread is not an effect.
Dumps not present: HIER_s4, HIER_s5.

## R1 4 minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2119 / +0.1573 | -0.0273 | [-0.2677, +0.2209] | 0.2248 (0.1124) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0722 / +0.3649 | +0.1464 | [-0.1522, +0.4519] | 0.3864 (0.1932) | OPEN |
| LATERAL:cross_mae_m | +0.0448 / -0.1005 | -0.0278 | [-0.1472, +0.0878] | 0.2943 (0.1988) | OPEN |
| LATERAL:heading_mae_deg | -1.6004 / -0.9078 | -1.2541 | [-1.9952, -0.6007] | 2.5508 (1.2754) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0924 / -0.0066 | -0.0495 | [-0.1072, +0.0009] | 0.1049 (0.0525) | OPEN |
| LATERAL:yaw_rate_mae_degps | -8.1196 / -1.8479 | -4.9838 | [-9.5824, -0.7882] | 10.6485 (5.3243) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0425 / +0.0229 | -0.0098 | [-0.0661, +0.0453] | 0.0746 (0.0575) | OPEN |
| TACTICAL:lat_err_rate | -0.0217 / -0.1021 | -0.0619 | [-0.1236, -0.0069] | 0.1305 (0.0654) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0217 / -0.0017 | -0.0117 | [-0.0480, +0.0229] | 0.1205 (0.0602) | OPEN |
| TACTICAL:ade_tac_m | -0.0402 / +0.4005 | +0.1801 | [-0.1535, +0.5305] | 0.4005 (0.2914) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0423 / +0.2639 | +0.1108 | [-0.1505, +0.3822] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0398 / +0.2664 | +0.1133 | [-0.1521, +0.3787] | 0.3476 | OPEN |
| ade6_m|all | -0.0411 / +0.2650 | +0.1119 | [-0.1178, +0.3529] | 0.2730 | OPEN |
| ade_tac_m|all | -0.0402 / +0.4005 | +0.1801 | [-0.1535, +0.5305] | 0.4005 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| R1_4_s0 | 3.1520 | 3.6609 | 1.7880 | 1.3202 | 0.6740 |
| R1_4_s1 | 3.2420 | 3.6891 | 1.9323 | 1.3872 | 0.7295 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
