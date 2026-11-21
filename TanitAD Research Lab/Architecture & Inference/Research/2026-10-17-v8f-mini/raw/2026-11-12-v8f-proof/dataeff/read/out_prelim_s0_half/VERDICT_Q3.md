# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD13 (DATAEFF), preliminary, seed 0 only. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 5}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "R1_2"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.6745357826426672, "D_read": 0.6745357826426673, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## R1 2 minus HIER

Seeds read [0] of [0]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1461 | -0.1461 | [-0.2750, -0.0129] | 0.2248 (0.2108) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0857 | -0.0857 | [-0.2493, +0.0818] | 0.3864 (0.2048) | OPEN |
| LATERAL:cross_mae_m | +0.0566 | +0.0566 | [-0.0163, +0.1324] | 0.2124 (0.1853) | OPEN |
| LATERAL:heading_mae_deg | -0.9223 | -0.9223 | [-1.7982, -0.0587] | 2.5508 (1.3533) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0096 | -0.0096 | [-0.0450, +0.0239] | 0.1049 (0.0830) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.0786 | -3.0786 | [-6.3001, -0.0255] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0325 | +0.0325 | [-0.0067, +0.0713] | 0.0746 (0.0625) | OPEN |
| TACTICAL:lat_err_rate | -0.0021 | -0.0021 | [-0.0413, +0.0342] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | +0.0413 | +0.0413 | [+0.0050, +0.0779] | 0.1205 (0.0850) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0646 | -0.0646 | [-0.2974, +0.1639] | 0.3119 (0.2709) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1375 | -0.1375 | [-0.3016, +0.0263] | 0.2780 | OPEN |
| ade6_m|bulk | +0.0554 | +0.0554 | [-0.1920, +0.3106] | 0.3476 | OPEN |
| ade6_m|all | -0.0495 | -0.0495 | [-0.2067, +0.1044] | 0.2161 | OPEN |
| ade_tac_m|all | -0.0646 | -0.0646 | [-0.2974, +0.1639] | 0.3119 | OPEN |

## R1 4 minus R1 2

Seeds read [0] of [0]. **Q3: PRELIMINARY (NO-COST GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0657 | -0.0657 | [-0.1993, +0.0725] | 0.2248 (0.2108) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0134 | +0.0134 | [-0.1689, +0.1896] | 0.3864 (0.2048) | OPEN |
| LATERAL:cross_mae_m | -0.0118 | -0.0118 | [-0.0929, +0.0701] | 0.2124 (0.1853) | OPEN |
| LATERAL:heading_mae_deg | -0.6781 | -0.6781 | [-1.6800, +0.3445] | 2.5508 (1.3533) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0828 | -0.0828 | [-0.1288, -0.0414] | 0.1049 (0.0830) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -5.0410 | -5.0410 | [-8.6837, -1.8328] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0750 | -0.0750 | [-0.1280, -0.0242] | 0.0746 (0.0625) | BETTER |
| TACTICAL:lat_err_rate | -0.0196 | -0.0196 | [-0.0542, +0.0158] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | -0.0629 | -0.0629 | [-0.1116, -0.0163] | 0.1205 (0.0850) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0244 | +0.0244 | [-0.2392, +0.2969] | 0.3119 (0.2709) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0953 | +0.0953 | [-0.1228, +0.3205] | 0.2780 | OPEN |
| ade6_m|bulk | -0.0952 | -0.0952 | [-0.3674, +0.1494] | 0.3476 | OPEN |
| ade6_m|all | +0.0084 | +0.0084 | [-0.1699, +0.1916] | 0.2161 | OPEN |
| ade_tac_m|all | +0.0244 | +0.0244 | [-0.2392, +0.2969] | 0.3119 | OPEN |

## R1 4 minus HIER

Seeds read [0] of [0]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2119 | -0.2119 | [-0.3459, -0.0756] | 0.2248 (0.2108) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0722 | -0.0722 | [-0.2533, +0.0994] | 0.3864 (0.2048) | OPEN |
| LATERAL:cross_mae_m | +0.0448 | +0.0448 | [-0.0402, +0.1301] | 0.2124 (0.1853) | OPEN |
| LATERAL:heading_mae_deg | -1.6004 | -1.6004 | [-2.3939, -0.8162] | 2.5508 (1.3533) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0924 | -0.0924 | [-0.1254, -0.0610] | 0.1049 (0.0830) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -8.1196 | -8.1196 | [-11.5508, -4.9925] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0425 | -0.0425 | [-0.0925, +0.0092] | 0.0746 (0.0625) | OPEN |
| TACTICAL:lat_err_rate | -0.0217 | -0.0217 | [-0.0592, +0.0133] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | -0.0217 | -0.0217 | [-0.0701, +0.0258] | 0.1205 (0.0850) | OPEN |
| TACTICAL:ade_tac_m | -0.0402 | -0.0402 | [-0.3167, +0.2100] | 0.3119 (0.2709) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0423 | -0.0423 | [-0.2953, +0.2041] | 0.2780 | OPEN |
| ade6_m|bulk | -0.0398 | -0.0398 | [-0.2797, +0.1846] | 0.3476 | OPEN |
| ade6_m|all | -0.0411 | -0.0411 | [-0.2282, +0.1287] | 0.2161 | OPEN |
| ade_tac_m|all | -0.0402 | -0.0402 | [-0.3167, +0.2100] | 0.3119 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| R1_2_s0 | 3.1435 | 3.5657 | 1.8537 | 1.3320 | 0.7370 |
| R1_4_s0 | 3.1520 | 3.6609 | 1.7880 | 1.3202 | 0.6740 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
