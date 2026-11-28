# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RR"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RR minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0479 / +0.1733 | +0.0627 | [-0.0900, +0.2295] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0979 / +0.0534 | -0.0222 | [-0.1486, +0.1113] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0254 / +0.0897 | +0.0322 | [-0.0603, +0.1157] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -1.9430 / +2.5620 | +0.3095 | [-2.2995, +2.8681] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0486 / +0.0396 | -0.0045 | [-0.0601, +0.0464] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -6.0283 / +7.1122 | +0.5419 | [-7.4462, +8.1566] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0133 / +0.0113 | -0.0010 | [-0.0319, +0.0258] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0521 / +0.0771 | +0.0125 | [-0.0684, +0.0938] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0313 / +0.0513 | +0.0100 | [-0.0469, +0.0646] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.1612 / +0.1327 | -0.0142 | [-0.2336, +0.2157] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1697 / +0.1856 | +0.0079 | [-0.2297, +0.2524] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0303 / -0.0332 | -0.0317 | [-0.1525, +0.0982] | 0.3535 | OPEN |
| ade6_m|all | -0.1061 / +0.0858 | -0.0101 | [-0.1560, +0.1418] | 0.4002 | OPEN |
| ade_tac_m|all | -0.1612 / +0.1327 | -0.0142 | [-0.2336, +0.2157] | 0.5955 | OPEN |

## RR minus HIER

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1360 / +0.1300 | -0.0030 | [-0.1982, +0.1771] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1849 / +0.0048 | -0.0901 | [-0.2342, +0.0544] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0736 / -0.0763 | -0.0750 | [-0.1219, -0.0271] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0231 / +1.2782 | +0.6275 | [-0.2434, +1.5853] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0112 / +0.0308 | +0.0098 | [-0.0211, +0.0381] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.1843 / +4.3175 | +1.5666 | [-2.2894, +5.4277] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0033 / +0.0221 | +0.0127 | [-0.0144, +0.0407] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0125 / -0.0183 | -0.0029 | [-0.0336, +0.0285] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0117 / +0.0242 | +0.0179 | [-0.0075, +0.0444] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.3082 / -0.0703 | -0.1893 | [-0.3878, +0.0048] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1927 / +0.0004 | -0.0961 | [-0.2509, +0.0717] | 0.4517 | OPEN |
| ade6_m|bulk | -0.2246 / -0.1158 | -0.1702 | [-0.3250, -0.0489] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.2072 / -0.0526 | -0.1299 | [-0.2599, -0.0019] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.3082 / -0.0703 | -0.1893 | [-0.3878, +0.0048] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| RR_s0 | 2.9859 | 3.5105 | 1.8639 | 1.2017 | 0.7074 |
| RR_s1 | 2.9244 | 3.4256 | 1.9050 | 1.4114 | 0.7553 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
