# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD16 (CELLPRED) C3, preliminary, seed 0 only. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 5}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "P"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.6745357826426672, "D_read": 0.6745357826426673, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## P minus G

Seeds read [0] of [0]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0016 | +0.0016 | [-0.0940, +0.0987] | 0.2248 (0.2108) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0783 | -0.0783 | [-0.1886, +0.0332] | 0.3864 (0.2048) | OPEN |
| LATERAL:cross_mae_m | +0.1450 | +0.1450 | [+0.0902, +0.2007] | 0.2124 (0.1853) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.6413 | -1.6413 | [-2.4206, -0.9242] | 2.5508 (1.3533) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0654 | -0.0654 | [-0.0926, -0.0417] | 0.1049 (0.0830) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -8.0131 | -8.0131 | [-11.0582, -5.2387] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0088 | +0.0088 | [-0.0325, +0.0488] | 0.0746 (0.0625) | OPEN |
| TACTICAL:lat_err_rate | -0.0021 | -0.0021 | [-0.0379, +0.0334] | 0.1305 (0.0734) | OPEN |
| TACTICAL:joint_err_rate | +0.0058 | +0.0058 | [-0.0288, +0.0396] | 0.1205 (0.0850) | OPEN |
| TACTICAL:ade_tac_m | +0.0517 | +0.0517 | [-0.1032, +0.2176] | 0.3119 (0.2709) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0816 | -0.0816 | [-0.2108, +0.0445] | 0.2780 | OPEN |
| ade6_m|bulk | +0.1684 | +0.1684 | [+0.0203, +0.3217] | 0.3476 | INSIDE-SPREAD |
| ade6_m|all | +0.0324 | +0.0324 | [-0.0735, +0.1446] | 0.2161 | OPEN |
| ade_tac_m|all | +0.0517 | +0.0517 | [-0.1032, +0.2176] | 0.3119 | OPEN |

## P minus HIER

Seeds read [0] of [0]. **Q3: PRELIMINARY (NO-COST)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0864 | -0.0864 | [-0.1917, +0.0162] | 0.2248 (0.2108) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1654 | -0.1654 | [-0.2883, -0.0473] | 0.3864 (0.2048) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0968 | +0.0968 | [+0.0207, +0.1746] | 0.2124 (0.1853) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2786 | +0.2786 | [-0.3471, +0.9067] | 2.5508 (1.3533) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0279 | -0.0279 | [-0.0489, -0.0081] | 0.1049 (0.0830) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -3.1691 | -3.1691 | [-5.5619, -0.7854] | 10.6485 (8.1641) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0254 | +0.0254 | [-0.0071, +0.0588] | 0.0746 (0.0625) | OPEN |
| TACTICAL:lat_err_rate | +0.0625 | +0.0625 | [+0.0291, +0.0957] | 0.1305 (0.0734) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0488 | +0.0488 | [+0.0179, +0.0797] | 0.1205 (0.0850) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.0953 | -0.0953 | [-0.2789, +0.0841] | 0.3119 (0.2709) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1046 | -0.1046 | [-0.2565, +0.0361] | 0.2780 | OPEN |
| ade6_m|bulk | -0.0259 | -0.0259 | [-0.2231, +0.1724] | 0.3476 | OPEN |
| ade6_m|all | -0.0687 | -0.0687 | [-0.1920, +0.0521] | 0.2161 | OPEN |
| ade_tac_m|all | -0.0953 | -0.0953 | [-0.2789, +0.0841] | 0.3119 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| P_s0 | 3.1244 | 3.5986 | 1.9134 | 1.3721 | 0.7445 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
