# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "GOW"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## GOW minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.4045 / +1.0562 | +1.2303 | [+0.8249, +1.6384] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +2.8532 / +2.5220 | +2.6876 | [+1.8110, +3.6093] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.1246 / +0.2605 | +0.1925 | [+0.0856, +0.3092] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +4.8069 / +4.1449 | +4.4759 | [+3.3952, +5.5550] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.1197 / +0.0767 | +0.0982 | [+0.0627, +0.1366] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +16.0382 / +12.5189 | +14.2785 | [+10.1597, +18.4298] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0959 / +0.0188 | +0.0573 | [+0.0023, +0.1104] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.1797 / +0.1271 | +0.1534 | [+0.1079, +0.1970] | 0.1042 (0.0534) | WORSE |
| TACTICAL:joint_err_rate | +0.1092 / +0.0567 | +0.0830 | [+0.0421, +0.1221] | 0.0492 (0.0263) | WORSE |
| TACTICAL:ade_tac_m | +4.0489 / +3.6674 | +3.8581 | [+2.6631, +5.1286] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.9879 / +2.0098 | +1.9989 | [+1.3088, +2.7382] | 0.4517 | WORSE |
| ade6_m|bulk | +3.7359 / +3.2159 | +3.4759 | [+2.1297, +4.8562] | 0.3535 | WORSE |
| ade6_m|all | +2.7850 / +2.5598 | +2.6724 | [+1.8344, +3.5540] | 0.4002 | WORSE |
| ade_tac_m|all | +4.0489 / +3.6674 | +3.8581 | [+2.6631, +5.1286] | 0.5955 | WORSE |

## GOW4 minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +2.3439 / +1.0402 | +1.6920 | [+0.8113, +2.5872] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +5.0855 / +2.8230 | +3.9542 | [+2.2154, +5.7678] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.1340 / +0.1828 | +0.1584 | [+0.0801, +0.2425] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +8.3908 / +5.2250 | +6.8079 | [+4.4148, +9.0410] | 4.5570 (1.9414) | WORSE |
| LATERAL:curvature_mae_1pm | +0.2527 / +0.1329 | +0.1928 | [+0.1153, +0.2777] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +30.2955 / +14.6201 | +22.4578 | [+12.1954, +32.9581] | 15.8479 (7.7515) | WORSE |
| TACTICAL:lon_err_rate | +0.1405 / +0.0279 | +0.0842 | [+0.0104, +0.1586] | 0.0838 (0.0204) | WORSE |
| TACTICAL:lat_err_rate | +0.2989 / +0.0913 | +0.1951 | [+0.0719, +0.3208] | 0.1042 (0.0534) | WORSE |
| TACTICAL:joint_err_rate | +0.1730 / +0.0517 | +0.1123 | [+0.0356, +0.1899] | 0.0492 (0.0263) | WORSE |
| TACTICAL:ade_tac_m | +7.0926 / +3.9534 | +5.5230 | [+3.1190, +8.0276] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.3645 / +2.3310 | +2.8478 | [+1.8252, +3.9767] | 0.4517 | WORSE |
| ade6_m|bulk | +6.7851 / +3.3491 | +5.0671 | [+2.3966, +7.6786] | 0.3535 | WORSE |
| ade6_m|all | +4.9243 / +2.7953 | +3.8598 | [+2.1978, +5.5776] | 0.4002 | WORSE |
| ade_tac_m|all | +7.0926 / +3.9534 | +5.5230 | [+3.1190, +8.0276] | 0.5955 | WORSE |

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
| GOW_s0 | 5.6312 | 5.3764 | 3.0951 | 1.3868 | 0.8541 |
| GOW_s1 | 5.2428 | 5.1238 | 2.6631 | 1.4785 | 0.7040 |
| GOW4_s0 | 7.7705 | 6.7530 | 4.0345 | 1.3962 | 0.9179 |
| GOW4_s1 | 5.4782 | 5.4450 | 2.6471 | 1.4008 | 0.6990 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
