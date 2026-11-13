# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD13.md, section 4 (rule) and section 5 (names); preliminary, seed 0 only. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 7}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "R1_4"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.6745357826426672, "D_read": 0.6745357826426673, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): False. A difference inside the spread is not an effect.
Dumps not present: HIER_s3, HIER_s4, HIER_s5.

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

## S1 4 minus R1 4

Seeds read [0] of [0]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3478 | +0.3478 | [+0.1707, +0.5367] | 0.2248 (0.2108) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3290 | +0.3290 | [+0.0870, +0.5876] | 0.3864 (0.2048) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.1182 | +0.1182 | [+0.0325, +0.2064] | 0.2124 (0.1853) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.6503 | -0.6503 | [-1.5122, +0.1621] | 2.5508 (1.3533) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0007 | -0.0007 | [-0.0298, +0.0312] | 0.1049 (0.0830) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.5513 | -1.5513 | [-4.4027, +1.3672] | 10.6485 (8.1641) | OPEN |
| TACTICAL:lon_err_rate | +0.0642 | +0.0642 | [+0.0167, +0.1151] | 0.0746 (0.0625) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0371 | +0.0371 | [-0.0017, +0.0780] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | +0.0567 | +0.0567 | [+0.0079, +0.1080] | 0.1205 (0.0850) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.5644 | +0.5644 | [+0.2130, +0.9281] | 0.3119 (0.2709) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0225 | +0.0225 | [-0.2426, +0.2868] | 0.2780 | OPEN |
| ade6_m|bulk | +0.8026 | +0.8026 | [+0.4823, +1.1647] | 0.3476 | WORSE |
| ade6_m|all | +0.3782 | +0.3782 | [+0.1425, +0.6224] | 0.2161 | WORSE |
| ade_tac_m|all | +0.5644 | +0.5644 | [+0.2130, +0.9281] | 0.3119 | WORSE |

## R1 4 minus F1 4

Seeds read [0] of [0]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1201 | +0.1201 | [+0.0040, +0.2377] | 0.2248 (0.2108) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1566 | +0.1566 | [+0.0121, +0.3126] | 0.3864 (0.2048) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.2126 | -0.2126 | [-0.3210, -0.1095] | 0.2124 (0.1853) | BETTER |
| LATERAL:heading_mae_deg | -1.2618 | -1.2618 | [-2.1238, -0.4159] | 2.5508 (1.3533) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0296 | -0.0296 | [-0.0614, +0.0000] | 0.1049 (0.0830) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.7103 | -0.7103 | [-4.1132, +2.6176] | 10.6485 (8.1641) | OPEN |
| TACTICAL:lon_err_rate | +0.0604 | +0.0604 | [+0.0171, +0.1056] | 0.0746 (0.0625) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0417 | -0.0417 | [-0.0780, -0.0054] | 0.1305 (0.0734) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0242 | +0.0242 | [-0.0133, +0.0629] | 0.1205 (0.0850) | OPEN |
| TACTICAL:ade_tac_m | -0.0210 | -0.0210 | [-0.2562, +0.2256] | 0.3119 (0.2709) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2457 | -0.2457 | [-0.4487, -0.0343] | 0.2780 | INSIDE-SPREAD |
| ade6_m|bulk | +0.2589 | +0.2589 | [+0.0442, +0.4901] | 0.3476 | INSIDE-SPREAD |
| ade6_m|all | -0.0156 | -0.0156 | [-0.1738, +0.1520] | 0.2161 | OPEN |
| ade_tac_m|all | -0.0210 | -0.0210 | [-0.2562, +0.2256] | 0.3119 | OPEN |

## HIER minus FLAT

Seeds read [0] of [0]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4999 | +0.4999 | [+0.3625, +0.6533] | 0.2248 (0.2108) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4551 | +0.4551 | [+0.2944, +0.6371] | 0.3864 (0.2048) | WORSE |
| LATERAL:cross_mae_m | -0.1788 | -0.1788 | [-0.2698, -0.0946] | 0.2124 (0.1853) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.7751 | -0.7751 | [-1.5597, +0.0006] | 2.5508 (1.3533) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0619 | +0.0619 | [+0.0353, +0.0905] | 0.1049 (0.0830) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +6.9265 | +6.9265 | [+4.1663, +9.9982] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.1167 | +0.1167 | [+0.0717, +0.1654] | 0.0746 (0.0625) | WORSE |
| TACTICAL:lat_err_rate | +0.0088 | +0.0088 | [-0.0246, +0.0412] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | +0.0900 | +0.0900 | [+0.0476, +0.1351] | 0.1205 (0.0850) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.3976 | +0.3976 | [+0.1651, +0.6539] | 0.3119 (0.2709) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1085 | +0.1085 | [-0.0751, +0.2886] | 0.2780 | OPEN |
| ade6_m|bulk | +0.4817 | +0.4817 | [+0.2157, +0.7804] | 0.3476 | WORSE |
| ade6_m|all | +0.2787 | +0.2787 | [+0.1210, +0.4560] | 0.2161 | WORSE |
| ade_tac_m|all | +0.3976 | +0.3976 | [+0.1651, +0.6539] | 0.3119 | WORSE |

## S1 4 minus HIER

Seeds read [0] of [0]. **Q3: PRELIMINARY (COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1359 | +0.1359 | [+0.0012, +0.2875] | 0.2248 (0.2108) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.2567 | +0.2567 | [+0.0473, +0.4830] | 0.3864 (0.2048) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.1630 | +0.1630 | [+0.0716, +0.2583] | 0.2124 (0.1853) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -2.2507 | -2.2507 | [-3.0867, -1.4686] | 2.5508 (1.3533) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0933 | -0.0933 | [-0.1288, -0.0602] | 0.1049 (0.0830) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -9.6749 | -9.6749 | [-13.3054, -6.3868] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0217 | +0.0217 | [-0.0288, +0.0759] | 0.0746 (0.0625) | OPEN |
| TACTICAL:lat_err_rate | +0.0154 | +0.0154 | [-0.0167, +0.0471] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | +0.0350 | +0.0350 | [-0.0129, +0.0863] | 0.1205 (0.0850) | OPEN |
| TACTICAL:ade_tac_m | +0.5242 | +0.5242 | [+0.2189, +0.8520] | 0.3119 (0.2709) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0197 | -0.0197 | [-0.2659, +0.2473] | 0.2780 | OPEN |
| ade6_m|bulk | +0.7628 | +0.7628 | [+0.5142, +1.0371] | 0.3476 | WORSE |
| ade6_m|all | +0.3371 | +0.3371 | [+0.1329, +0.5559] | 0.2161 | WORSE |
| ade_tac_m|all | +0.5242 | +0.5242 | [+0.2189, +0.8520] | 0.3119 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| F1_4_s0 | 3.1676 | 3.9067 | 1.6678 | 1.5328 | 0.6499 |
| FLAT_s0 | 2.9144 | 3.5947 | 1.5000 | 1.4542 | 0.6057 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| R1_4_s0 | 3.1520 | 3.6609 | 1.7880 | 1.3202 | 0.6740 |
| S1_4_s0 | 3.5302 | 3.6835 | 2.1357 | 1.4384 | 0.7307 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
