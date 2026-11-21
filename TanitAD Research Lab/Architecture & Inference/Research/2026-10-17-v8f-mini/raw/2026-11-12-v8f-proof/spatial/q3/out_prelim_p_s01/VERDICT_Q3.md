# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD16. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 8}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "P"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.6745357826426672, "D_read": 0.6745357826426673, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): False. A difference inside the spread is not an effect.
Dumps not present: HIER_s4, HIER_s5.

## P minus G

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0016 / +0.2640 | +0.1328 | [-0.0417, +0.3036] | 0.2248 (0.1124) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0783 / +0.2898 | +0.1057 | [-0.1324, +0.3336] | 0.3864 (0.1932) | OPEN |
| LATERAL:cross_mae_m | +0.1450 / +0.0404 | +0.0927 | [+0.0149, +0.1709] | 0.2943 (0.1988) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.6413 / +1.8053 | +0.0820 | [-1.9599, +2.0185] | 2.5508 (1.2754) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0654 / +0.0429 | -0.0113 | [-0.0768, +0.0507] | 0.1049 (0.0525) | OPEN |
| LATERAL:yaw_rate_mae_degps | -8.0131 / +6.3893 | -0.8119 | [-9.4408, +7.4093] | 10.6485 (5.3243) | OPEN |
| TACTICAL:lon_err_rate | +0.0088 / +0.0721 | +0.0404 | [-0.0098, +0.0881] | 0.0746 (0.0575) | OPEN |
| TACTICAL:lat_err_rate | -0.0021 / +0.0905 | +0.0442 | [-0.0183, +0.1073] | 0.1305 (0.0654) | OPEN |
| TACTICAL:joint_err_rate | +0.0058 / +0.1109 | +0.0584 | [-0.0106, +0.1247] | 0.1205 (0.0602) | OPEN |
| TACTICAL:ade_tac_m | +0.0517 / +0.4020 | +0.2268 | [-0.0217, +0.4673] | 0.4005 (0.2914) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0816 / +0.2868 | +0.1026 | [-0.1431, +0.3465] | 0.4093 | OPEN |
| ade6_m|bulk | +0.1684 / +0.2546 | +0.2115 | [+0.0950, +0.3301] | 0.3476 | INSIDE-SPREAD |
| ade6_m|all | +0.0324 / +0.2721 | +0.1523 | [-0.0174, +0.3164] | 0.2730 | OPEN |
| ade_tac_m|all | +0.0517 / +0.4020 | +0.2268 | [-0.0217, +0.4673] | 0.4005 | OPEN |

## P minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0864 / +0.2207 | +0.0672 | [-0.1329, +0.2676] | 0.2248 (0.1124) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1654 / +0.2411 | +0.0379 | [-0.2180, +0.2952] | 0.3864 (0.1932) | OPEN |
| LATERAL:cross_mae_m | +0.0968 / -0.1256 | -0.0144 | [-0.1544, +0.1369] | 0.2943 (0.1988) | OPEN |
| LATERAL:heading_mae_deg | +0.2786 / +0.5215 | +0.4001 | [+0.0026, +0.7801] | 2.5508 (1.2754) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0279 / +0.0342 | +0.0032 | [-0.0372, +0.0426] | 0.1049 (0.0525) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.1691 / +3.5982 | +0.2146 | [-4.1922, +4.6944] | 10.6485 (5.3243) | OPEN |
| TACTICAL:lon_err_rate | +0.0254 / +0.0830 | +0.0542 | [+0.0106, +0.0997] | 0.0746 (0.0575) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0625 / -0.0050 | +0.0288 | [-0.0211, +0.0777] | 0.1305 (0.0654) | OPEN |
| TACTICAL:joint_err_rate | +0.0488 / +0.0838 | +0.0663 | [+0.0346, +0.1007] | 0.1205 (0.0602) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0953 / +0.1989 | +0.0518 | [-0.1757, +0.2822] | 0.4005 (0.2914) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1046 / +0.1016 | -0.0015 | [-0.1722, +0.1634] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0259 / +0.1721 | +0.0731 | [-0.1162, +0.2544] | 0.3476 | OPEN |
| ade6_m|all | -0.0687 / +0.1337 | +0.0325 | [-0.1213, +0.1899] | 0.2730 | OPEN |
| ade_tac_m|all | -0.0953 / +0.1989 | +0.0518 | [-0.1757, +0.2822] | 0.4005 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| P_s0 | 3.1244 | 3.5986 | 1.9134 | 1.3721 | 0.7445 |
| P_s1 | 3.1107 | 3.5268 | 1.9957 | 1.3621 | 0.8149 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
