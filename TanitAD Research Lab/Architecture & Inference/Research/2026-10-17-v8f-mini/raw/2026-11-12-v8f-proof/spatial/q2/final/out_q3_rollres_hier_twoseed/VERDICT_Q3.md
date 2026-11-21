# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "ROLL"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## ROLL minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1550 / -0.0876 | -0.1213 | [-0.1998, -0.0470] | 0.2997 (0.1392) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.1602 / -0.1010 | -0.1306 | [-0.2171, -0.0510] | 0.4550 (0.1815) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0491 / -0.1200 | -0.0846 | [-0.1460, -0.0210] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.6299 / -0.0463 | +0.2918 | [-0.2451, +1.0138] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0301 / +0.0184 | +0.0243 | [+0.0102, +0.0419] | 0.1049 (0.0500) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.4615 / +1.7742 | +1.6179 | [-0.1052, +3.4589] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | -0.0154 / -0.0058 | -0.0106 | [-0.0323, +0.0117] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0096 / -0.0717 | -0.0406 | [-0.0874, +0.0036] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | -0.0054 / -0.0271 | -0.0163 | [-0.0444, +0.0091] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.2252 / -0.2430 | -0.2341 | [-0.3388, -0.1251] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1623 / -0.1205 | -0.1414 | [-0.2303, -0.0541] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | -0.1427 / -0.2263 | -0.1845 | [-0.2922, -0.0504] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | -0.1534 / -0.1687 | -0.1611 | [-0.2314, -0.0869] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | -0.2252 / -0.2430 | -0.2341 | [-0.3388, -0.1251] | 0.4169 | INSIDE-SPREAD |

## RES minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0635 / +0.0362 | -0.0136 | [-0.1152, +0.0871] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1091 / -0.0390 | -0.0741 | [-0.1734, +0.0197] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0665 / -0.1173 | -0.0919 | [-0.1453, -0.0357] | 0.2943 (0.1407) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4589 / -0.8617 | -0.2014 | [-1.0393, +0.7841] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0077 / -0.0000 | -0.0038 | [-0.0181, +0.0073] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.6909 / -2.3232 | -0.3161 | [-3.0759, +2.8316] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0204 / -0.0013 | +0.0096 | [-0.0171, +0.0357] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0371 / -0.0642 | -0.0135 | [-0.0775, +0.0542] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0308 / -0.0196 | +0.0056 | [-0.0346, +0.0455] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.2185 / -0.1619 | -0.1902 | [-0.3267, -0.0614] | 0.4169 (0.1897) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0958 / -0.0876 | -0.0917 | [-0.1915, -0.0012] | 0.4093 | INSIDE-SPREAD |
| ade6_m|bulk | -0.2004 / -0.1434 | -0.1719 | [-0.3139, -0.0419] | 0.3850 | INSIDE-SPREAD |
| ade6_m|all | -0.1435 / -0.1130 | -0.1283 | [-0.2199, -0.0421] | 0.2845 | INSIDE-SPREAD |
| ade_tac_m|all | -0.2185 / -0.1619 | -0.1902 | [-0.3267, -0.0614] | 0.4169 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| RES_s0 | 3.0496 | 3.6074 | 1.9364 | 1.2088 | 0.7266 |
| RES_s1 | 2.8639 | 3.3376 | 1.8112 | 1.3704 | 0.7115 |
| ROLL_s0 | 3.0397 | 3.5409 | 1.8449 | 1.2262 | 0.6903 |
| ROLL_s1 | 2.8082 | 3.3047 | 1.6874 | 1.3677 | 0.7040 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
