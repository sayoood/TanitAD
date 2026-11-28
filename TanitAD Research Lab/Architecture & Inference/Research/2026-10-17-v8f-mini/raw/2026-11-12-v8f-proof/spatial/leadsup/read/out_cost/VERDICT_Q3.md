# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 12}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LS"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LS minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0361 / -0.1508 | -0.0574 | [-0.1956, +0.0905] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1084 / -0.0874 | -0.0979 | [-0.1775, -0.0244] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0357 / -0.0850 | -0.0247 | [-0.1229, +0.0592] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -1.6196 / +0.8488 | -0.3854 | [-1.8903, +1.1731] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0522 / +0.0380 | -0.0071 | [-0.0642, +0.0474] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -4.7570 / +1.6397 | -1.5586 | [-6.0054, +2.3472] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0659 / -0.0050 | -0.0354 | [-0.0828, +0.0121] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0038 / +0.0271 | +0.0154 | [-0.0136, +0.0458] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0679 / +0.0004 | -0.0338 | [-0.0847, +0.0175] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.1209 / -0.2267 | -0.1738 | [-0.3026, -0.0412] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1661 / -0.0696 | -0.1178 | [-0.2283, -0.0117] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0169 / -0.2399 | -0.1115 | [-0.3142, +0.0967] | 0.3535 | OPEN |
| ade6_m|all | -0.0827 / -0.1473 | -0.1150 | [-0.1995, -0.0274] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.1209 / -0.2267 | -0.1738 | [-0.3026, -0.0412] | 0.5955 | INSIDE-SPREAD |

## LSH minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0965 / -0.0248 | -0.0606 | [-0.1448, +0.0270] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1287 / -0.0469 | -0.0878 | [-0.1854, +0.0179] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0696 / +0.1312 | +0.1004 | [+0.0414, +0.1652] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.5698 / +1.7743 | +0.1022 | [-1.8805, +2.0138] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0606 / +0.0262 | -0.0172 | [-0.0717, +0.0316] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -7.6819 / +4.3551 | -1.6634 | [-8.9695, +5.2239] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0054 / +0.0321 | +0.0188 | [-0.0121, +0.0471] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0075 / +0.0742 | +0.0333 | [-0.0254, +0.0938] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0071 / +0.0759 | +0.0415 | [-0.0094, +0.0919] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.1048 / +0.0532 | -0.0258 | [-0.1853, +0.1383] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0710 / +0.2020 | +0.0655 | [-0.1391, +0.2610] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0679 / -0.1670 | -0.1175 | [-0.2638, -0.0062] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.0696 / +0.0337 | -0.0179 | [-0.1239, +0.0909] | 0.4002 | OPEN |
| ade_tac_m|all | -0.1048 / +0.0532 | -0.0258 | [-0.1853, +0.1383] | 0.5955 | OPEN |

## LSD minus G

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0218 / +0.2741 | +0.1479 | [-0.0206, +0.3172] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0021 / +0.1790 | +0.0905 | [-0.0423, +0.2242] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0188 / +0.0432 | +0.0122 | [-0.0494, +0.0728] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -1.4746 / +1.0672 | -0.2037 | [-1.8099, +1.2166] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0099 / +0.0214 | +0.0058 | [-0.0195, +0.0289] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.0644 / +3.1348 | +1.0352 | [-2.1985, +3.7451] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0367 / +0.0229 | +0.0298 | [+0.0056, +0.0544] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0821 / +0.0525 | -0.0148 | [-0.1005, +0.0676] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0092 / +0.0442 | +0.0267 | [-0.0081, +0.0598] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0296 / +0.2512 | +0.1108 | [-0.1039, +0.3188] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1001 / +0.2666 | +0.0833 | [-0.1644, +0.3308] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0726 / +0.0559 | +0.0643 | [-0.0373, +0.1597] | 0.3535 | OPEN |
| ade6_m|all | -0.0213 / +0.1705 | +0.0746 | [-0.0707, +0.2154] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0296 / +0.2512 | +0.1108 | [-0.1039, +0.3188] | 0.5955 | OPEN |

## LS minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST GAIN-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0142 / -0.4248 | -0.2053 | [-0.4840, +0.0761] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1105 / -0.2664 | -0.1885 | [-0.3179, -0.0532] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0544 / -0.1282 | -0.0369 | [-0.1640, +0.0781] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1449 / -0.2184 | -0.1816 | [-0.6234, +0.2879] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0423 / +0.0166 | -0.0128 | [-0.0529, +0.0276] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.6926 / -1.4951 | -2.5938 | [-4.6598, -0.7726] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.1025 / -0.0279 | -0.0652 | [-0.1189, -0.0100] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0859 / -0.0254 | +0.0302 | [-0.0404, +0.1057] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0771 / -0.0438 | -0.0604 | [-0.0953, -0.0248] | 0.0492 (0.0263) | BETTER |
| TACTICAL:ade_tac_m | -0.0913 / -0.4780 | -0.2846 | [-0.5504, -0.0098] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0660 / -0.3362 | -0.2011 | [-0.4161, +0.0089] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0558 / -0.2958 | -0.1758 | [-0.3453, +0.0198] | 0.3535 | OPEN |
| ade6_m|all | -0.0614 / -0.3178 | -0.1896 | [-0.3662, -0.0073] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.0913 / -0.4780 | -0.2846 | [-0.5504, -0.0098] | 0.5955 | INSIDE-SPREAD |

## LSH minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1183 / -0.2988 | -0.2085 | [-0.3511, -0.0763] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.1309 / -0.2259 | -0.1784 | [-0.2827, -0.0804] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0884 / +0.0880 | +0.0882 | [+0.0444, +0.1345] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.0951 / +0.7071 | +0.3060 | [-0.3961, +0.9056] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0507 / +0.0048 | -0.0230 | [-0.0614, +0.0114] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -6.6208 / +1.2203 | -2.7002 | [-7.7592, +2.1048] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0313 / +0.0092 | -0.0110 | [-0.0463, +0.0242] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0746 / +0.0217 | +0.0481 | [+0.0092, +0.0894] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0021 / +0.0317 | +0.0148 | [-0.0177, +0.0453] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0752 / -0.1981 | -0.1366 | [-0.2828, +0.0089] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0291 / -0.0646 | -0.0177 | [-0.1345, +0.1058] | 0.4517 | OPEN |
| ade6_m|bulk | -0.1405 / -0.2229 | -0.1817 | [-0.3061, -0.0610] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | -0.0483 / -0.1368 | -0.0925 | [-0.1932, +0.0072] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0752 / -0.1981 | -0.1366 | [-0.2828, +0.0089] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LS_s0 | 3.0093 | 3.5142 | 1.9479 | 1.2628 | 0.6707 |
| LS_s1 | 2.6913 | 3.1704 | 1.5809 | 1.2366 | 0.7045 |
| LSD_s0 | 3.0707 | 3.5802 | 1.9336 | 1.2084 | 0.7478 |
| LSD_s1 | 3.0090 | 3.5066 | 2.0057 | 1.3649 | 0.7482 |
| LSH_s0 | 3.0224 | 3.6093 | 1.8153 | 1.2967 | 0.7457 |
| LSH_s1 | 2.8723 | 3.4420 | 1.7069 | 1.4529 | 0.7799 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
