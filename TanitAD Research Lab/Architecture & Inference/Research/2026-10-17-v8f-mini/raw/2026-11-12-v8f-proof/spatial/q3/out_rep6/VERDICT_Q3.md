# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "G"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859553, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## G minus HIER

Seeds read [0, 1, 2, 3, 4, 5] of [0, 1, 2, 3, 4, 5]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0880 / -0.0433 / -0.0183 / -0.1177 / +0.0225 / -0.1412 | -0.0643 | [-0.1295, -0.0012] | 0.2997 (0.1392) | MIXED-SIGN |
| LONGITUDINAL:along_mae_m | -0.0871 / -0.0486 / +0.0750 / -0.1050 / +0.0149 / -0.0954 | -0.0410 | [-0.1138, +0.0336] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0482 / -0.1660 / -0.0307 / -0.0440 / +0.0543 / -0.2194 | -0.0757 | [-0.1555, -0.0031] | 0.2943 (0.1407) | MIXED-SIGN |
| LATERAL:heading_mae_deg | +1.9199 / -1.2838 / -0.2208 / -1.4355 / -0.4829 / -1.6976 | -0.5335 | [-1.3974, +0.5462] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0373 / -0.0086 / +0.0402 / -0.0077 / -0.0037 / +0.0125 | +0.0117 | [-0.0056, +0.0311] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | +4.8412 / -2.7912 / +1.6406 / -2.0854 / -1.6055 / -1.1130 | -0.1856 | [-2.2661, +2.2567] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0167 / +0.0108 / -0.0513 / -0.0208 / -0.0717 / -0.0179 | -0.0224 | [-0.0504, +0.0059] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | +0.0646 / -0.0955 / -0.1121 / -0.0509 / +0.0146 / -0.0750 | -0.0424 | [-0.0901, +0.0133] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | +0.0429 / -0.0271 / -0.0796 / -0.0375 / -0.0629 / -0.0321 | -0.0327 | [-0.0634, +0.0021] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.1470 / -0.2031 / +0.0657 / -0.1703 / +0.0500 / -0.3338 | -0.1231 | [-0.2492, +0.0097] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0230 / -0.1852 / +0.2200 / -0.0479 / +0.0151 / -0.1566 | -0.0296 | [-0.1419, +0.0937] | 0.4093 | OPEN |
| ade6_m|bulk | -0.1943 / -0.0826 / -0.1700 / -0.2003 / +0.0474 / -0.3134 | -0.1522 | [-0.2623, -0.0312] | 0.3850 | MIXED-SIGN |
| ade6_m|all | -0.1011 / -0.1384 / +0.0421 / -0.1174 / +0.0298 / -0.2281 | -0.0855 | [-0.1705, +0.0044] | 0.2845 | OPEN |
| ade_tac_m|all | -0.1470 / -0.2031 / +0.0657 / -0.1703 / +0.0500 / -0.3338 | -0.1231 | [-0.2492, +0.0097] | 0.4169 | OPEN |

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
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
