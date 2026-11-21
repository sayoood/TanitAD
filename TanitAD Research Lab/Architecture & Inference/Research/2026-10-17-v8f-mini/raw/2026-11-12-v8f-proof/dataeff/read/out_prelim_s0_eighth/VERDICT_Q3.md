# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD13 (DATAEFF), preliminary, seed 0 only. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 6}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "R1_8"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.6745357826426672, "D_read": 0.6745357826426673, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## R1 8 minus HIER

Seeds read [0] of [0]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0986 | -0.0986 | [-0.2507, +0.0470] | 0.2248 (0.2108) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0453 | +0.0453 | [-0.1696, +0.2625] | 0.3864 (0.2048) | OPEN |
| LATERAL:cross_mae_m | +0.2417 | +0.2417 | [+0.1264, +0.3618] | 0.2124 (0.1853) | WORSE |
| LATERAL:heading_mae_deg | -2.6904 | -2.6904 | [-3.6943, -1.7905] | 2.5508 (1.3533) | BETTER |
| LATERAL:curvature_mae_1pm | -0.1292 | -0.1292 | [-0.1674, -0.0933] | 0.1049 (0.0830) | BETTER |
| LATERAL:yaw_rate_mae_degps | -11.2820 | -11.2820 | [-15.4105, -7.5155] | 10.6485 (8.1641) | BETTER |
| TACTICAL:lon_err_rate | -0.0567 | -0.0567 | [-0.1008, -0.0121] | 0.0746 (0.0625) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0404 | -0.0404 | [-0.0759, -0.0075] | 0.1305 (0.0734) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0592 | -0.0592 | [-0.1009, -0.0196] | 0.1205 (0.0850) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3233 | +0.3233 | [-0.0038, +0.6634] | 0.3119 (0.2709) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3014 | +0.3014 | [-0.0080, +0.6164] | 0.2780 | OPEN |
| ade6_m|bulk | +0.1045 | +0.1045 | [-0.1574, +0.3710] | 0.3476 | OPEN |
| ade6_m|all | +0.2116 | +0.2116 | [-0.0093, +0.4415] | 0.2161 | OPEN |
| ade_tac_m|all | +0.3233 | +0.3233 | [-0.0038, +0.6634] | 0.3119 | OPEN |

## R1 8 minus R1 4

Seeds read [0] of [0]. **Q3: PRELIMINARY (COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1132 | +0.1132 | [-0.0116, +0.2388] | 0.2248 (0.2108) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1176 | +0.1176 | [-0.0654, +0.3248] | 0.3864 (0.2048) | OPEN |
| LATERAL:cross_mae_m | +0.1969 | +0.1969 | [+0.0969, +0.3088] | 0.2124 (0.1853) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.0900 | -1.0900 | [-2.0167, -0.2334] | 2.5508 (1.3533) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0370 | -0.0370 | [-0.0618, -0.0138] | 0.1049 (0.0830) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -3.1664 | -3.1664 | [-6.2669, -0.3048] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0142 | -0.0142 | [-0.0592, +0.0288] | 0.0746 (0.0625) | OPEN |
| TACTICAL:lat_err_rate | -0.0188 | -0.0188 | [-0.0509, +0.0158] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | -0.0375 | -0.0375 | [-0.0830, +0.0059] | 0.1205 (0.0850) | OPEN |
| TACTICAL:ade_tac_m | +0.3635 | +0.3635 | [+0.0985, +0.6575] | 0.3119 (0.2709) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3436 | +0.3436 | [+0.0943, +0.6031] | 0.2780 | WORSE |
| ade6_m|bulk | +0.1443 | +0.1443 | [-0.0807, +0.4042] | 0.3476 | OPEN |
| ade6_m|all | +0.2527 | +0.2527 | [+0.0750, +0.4510] | 0.2161 | WORSE |
| ade_tac_m|all | +0.3635 | +0.3635 | [+0.0985, +0.6575] | 0.3119 | WORSE |

## R1 8 minus R1 2

Seeds read [0] of [0]. **Q3: PRELIMINARY (COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0475 | +0.0475 | [-0.1181, +0.2093] | 0.2248 (0.2108) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1310 | +0.1310 | [-0.0790, +0.3421] | 0.3864 (0.2048) | OPEN |
| LATERAL:cross_mae_m | +0.1851 | +0.1851 | [+0.0742, +0.3081] | 0.2124 (0.1853) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.7682 | -1.7682 | [-2.8313, -0.7930] | 2.5508 (1.3533) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.1196 | -0.1196 | [-0.1703, -0.0746] | 0.1049 (0.0830) | BETTER |
| LATERAL:yaw_rate_mae_degps | -8.2035 | -8.2035 | [-12.3779, -4.2285] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0892 | -0.0892 | [-0.1388, -0.0379] | 0.0746 (0.0625) | BETTER |
| TACTICAL:lat_err_rate | -0.0383 | -0.0383 | [-0.0784, +0.0050] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | -0.1005 | -0.1005 | [-0.1480, -0.0521] | 0.1205 (0.0850) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3878 | +0.3878 | [+0.0665, +0.7097] | 0.3119 (0.2709) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4389 | +0.4389 | [+0.1661, +0.7156] | 0.2780 | WORSE |
| ade6_m|bulk | +0.0491 | +0.0491 | [-0.2533, +0.3572] | 0.3476 | OPEN |
| ade6_m|all | +0.2611 | +0.2611 | [+0.0434, +0.4784] | 0.2161 | WORSE |
| ade_tac_m|all | +0.3878 | +0.3878 | [+0.0665, +0.7097] | 0.3119 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| R1_2_s0 | 3.1435 | 3.5657 | 1.8537 | 1.3320 | 0.7370 |
| R1_4_s0 | 3.1520 | 3.6609 | 1.7880 | 1.3202 | 0.6740 |
| R1_8_s0 | 3.4047 | 4.0046 | 1.9012 | 1.5171 | 0.6365 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
