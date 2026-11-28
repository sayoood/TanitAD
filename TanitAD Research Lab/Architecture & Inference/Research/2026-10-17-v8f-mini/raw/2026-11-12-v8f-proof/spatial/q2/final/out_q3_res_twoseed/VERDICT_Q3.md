# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 10}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "RES"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8990895386859552, "D_read": 0.8990895386859552, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## RES minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0246 / +0.0795 | +0.0521 | [-0.0246, +0.1207] | 0.2997 (0.1392) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0220 / +0.0096 | -0.0062 | [-0.0841, +0.0656] | 0.4550 (0.1815) | OPEN |
| LATERAL:cross_mae_m | -0.0183 / +0.0487 | +0.0152 | [-0.0500, +0.0751] | 0.2943 (0.1407) | OPEN |
| LATERAL:heading_mae_deg | -1.4610 / +0.4221 | -0.5195 | [-1.7358, +0.5993] | 2.5508 (1.1975) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0450 / +0.0087 | -0.0182 | [-0.0561, +0.0135] | 0.1049 (0.0500) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.1503 / +0.4663 | -1.3420 | [-4.4149, +1.1750] | 10.6485 (4.6266) | OPEN |
| TACTICAL:lon_err_rate | +0.0038 / -0.0121 | -0.0042 | [-0.0290, +0.0189] | 0.0746 (0.0329) | OPEN |
| TACTICAL:lat_err_rate | -0.0275 / +0.0313 | +0.0019 | [-0.0432, +0.0442] | 0.1305 (0.0500) | OPEN |
| TACTICAL:joint_err_rate | -0.0121 / +0.0075 | -0.0023 | [-0.0298, +0.0231] | 0.1205 (0.0438) | OPEN |
| TACTICAL:ade_tac_m | -0.0715 / +0.0412 | -0.0151 | [-0.1458, +0.1058] | 0.4169 (0.1897) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0728 / +0.0976 | +0.0124 | [-0.1322, +0.1439] | 0.4093 | OPEN |
| ade6_m|bulk | -0.0061 / -0.0608 | -0.0335 | [-0.1360, +0.0768] | 0.3850 | OPEN |
| ade6_m|all | -0.0424 / +0.0254 | -0.0085 | [-0.0932, +0.0697] | 0.2845 | OPEN |
| ade_tac_m|all | -0.0715 / +0.0412 | -0.0151 | [-0.1458, +0.1058] | 0.4169 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| HIER_s0 | 3.1931 | 3.7032 | 1.9998 | 1.2754 | 0.6957 |
| HIER_s1 | 2.9769 | 3.4252 | 1.7750 | 1.4877 | 0.7311 |
| HIER_s2 | 3.1607 | 3.4717 | 1.9858 | 1.4606 | 0.8162 |
| HIER_s3 | 2.9201 | 3.2939 | 1.8914 | 1.1934 | 0.7724 |
| HIER_s4 | 2.9086 | 3.3309 | 1.7001 | 1.4618 | 0.7524 |
| HIER_s5 | 3.0344 | 3.4157 | 1.8393 | 1.4161 | 0.7624 |
| RES_s0 | 3.0496 | 3.6074 | 1.9364 | 1.2088 | 0.7266 |
| RES_s1 | 2.8639 | 3.3376 | 1.8112 | 1.3704 | 0.7115 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
