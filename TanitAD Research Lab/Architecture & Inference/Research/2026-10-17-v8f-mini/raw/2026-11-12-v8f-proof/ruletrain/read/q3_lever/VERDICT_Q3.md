# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "GRT"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## GRT minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0881 / +0.0967 | +0.0924 | [+0.0298, +0.1625] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0988 / +0.0692 | +0.0840 | [+0.0001, +0.1829] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0513 / +0.0435 | -0.0039 | [-0.0609, +0.0542] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.8796 / -0.2934 | -0.5865 | [-1.0452, -0.1770] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0333 / -0.0058 | -0.0196 | [-0.0422, -0.0014] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.6015 / -0.6450 | -1.6232 | [-3.4274, -0.2166] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0175 / +0.0192 | +0.0008 | [-0.0258, +0.0292] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0158 / +0.0163 | +0.0002 | [-0.0283, +0.0234] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0138 / +0.0304 | +0.0083 | [-0.0229, +0.0396] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0852 / +0.1346 | +0.1099 | [-0.0074, +0.2432] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0286 / +0.0702 | +0.0208 | [-0.0726, +0.1261] | 0.4517 | OPEN |
| ade6_m|bulk | +0.1585 / +0.1132 | +0.1359 | [+0.0370, +0.2553] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.0567 / +0.0898 | +0.0732 | [-0.0053, +0.1621] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0852 / +0.1346 | +0.1099 | [-0.0074, +0.2432] | 0.5955 | OPEN |

## GFTb minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2084 / -0.1002 | +0.0541 | [-0.1332, +0.2575] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1888 / -0.0037 | +0.0926 | [-0.0378, +0.2546] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0951 / -0.0176 | -0.0564 | [-0.1160, +0.0027] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.6049 / +0.1168 | -0.2440 | [-0.8328, +0.3033] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0049 / +0.0043 | +0.0046 | [-0.0072, +0.0161] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.7391 / -0.1098 | -0.4245 | [-1.8243, +0.8546] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0021 / -0.0129 | -0.0054 | [-0.0290, +0.0167] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0242 / -0.0042 | -0.0142 | [-0.0377, +0.0073] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0054 / +0.0100 | +0.0023 | [-0.0202, +0.0269] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1677 / -0.0348 | +0.0665 | [-0.0867, +0.2623] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0350 / -0.0319 | -0.0335 | [-0.1064, +0.0357] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3037 / -0.0219 | +0.1409 | [-0.0726, +0.4335] | 0.3535 | OPEN |
| ade6_m|all | +0.1194 / -0.0274 | +0.0460 | [-0.0620, +0.1833] | 0.4002 | OPEN |
| ade_tac_m|all | +0.1677 / -0.0348 | +0.0665 | [-0.0867, +0.2623] | 0.5955 | OPEN |

## GFT minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2212 / -0.1247 | -0.1729 | [-0.2694, -0.0580] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.2936 / -0.1022 | -0.1979 | [-0.3436, -0.0244] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0351 / -0.1037 | -0.0343 | [-0.1300, +0.0517] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -1.9790 / +0.6128 | -0.6831 | [-2.3041, +0.8851] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0579 / +0.0338 | -0.0120 | [-0.0699, +0.0426] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -6.5048 / +3.7101 | -1.3974 | [-7.8768, +4.6502] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0150 / -0.0559 | -0.0204 | [-0.0755, +0.0279] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0321 / -0.0129 | -0.0225 | [-0.0481, +0.0012] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0063 / -0.0567 | -0.0252 | [-0.0761, +0.0204] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.3506 / -0.2349 | -0.2927 | [-0.4359, -0.1322] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2917 / -0.1260 | -0.2089 | [-0.3524, -0.0592] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.1911 / -0.1908 | -0.1910 | [-0.3233, -0.0668] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.2458 / -0.1556 | -0.2007 | [-0.3011, -0.0869] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.3506 / -0.2349 | -0.2927 | [-0.4359, -0.1322] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| GFT_s0 | 2.8462 | 3.3885 | 1.6906 | 1.2622 | 0.7449 |
| GFT_s1 | 2.6830 | 3.1140 | 1.6070 | 1.2180 | 0.6474 |
| GFTb_s0 | 2.9656 | 3.3535 | 1.8990 | 1.1671 | 0.7395 |
| GFTb_s1 | 2.6556 | 3.0821 | 1.5067 | 1.2004 | 0.6574 |
| GRT_s0 | 2.9029 | 3.3599 | 1.7788 | 1.2109 | 0.7311 |
| GRT_s1 | 2.7728 | 3.1842 | 1.7037 | 1.2614 | 0.6778 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
