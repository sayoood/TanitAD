# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LA"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LA minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0108 / +0.0895 | +0.0394 | [-0.0318, +0.1172] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0535 / +0.0959 | +0.0212 | [-0.0802, +0.1253] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0341 / +0.0398 | +0.0369 | [+0.0102, +0.0623] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2670 / +0.8069 | +0.5369 | [+0.1155, +0.9682] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0018 / +0.0191 | +0.0087 | [-0.0061, +0.0280] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.3031 / +3.6608 | +1.9819 | [-0.1370, +4.5951] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0321 / +0.0271 | -0.0025 | [-0.0438, +0.0377] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0292 / +0.0021 | +0.0156 | [-0.0079, +0.0382] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0200 / +0.0192 | -0.0004 | [-0.0329, +0.0300] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0198 / +0.1811 | +0.0806 | [-0.0640, +0.2268] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0583 / +0.1319 | +0.0951 | [+0.0213, +0.1696] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.0990 / +0.1054 | +0.0032 | [-0.1430, +0.1565] | 0.3535 | OPEN |
| ade6_m|all | -0.0134 / +0.1198 | +0.0532 | [-0.0430, +0.1509] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0198 / +0.1811 | +0.0806 | [-0.0640, +0.2268] | 0.5955 | OPEN |

## LA01 minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0887 / +0.0680 | +0.0784 | [+0.0404, +0.1147] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0762 / +0.1171 | +0.0967 | [+0.0518, +0.1554] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0444 / +0.0216 | +0.0330 | [+0.0109, +0.0540] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2442 / +0.8007 | +0.5225 | [+0.1426, +0.9602] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0022 / +0.0096 | +0.0059 | [-0.0031, +0.0172] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2933 / +1.7701 | +0.7384 | [-0.8599, +2.6041] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0067 / +0.0238 | +0.0152 | [-0.0029, +0.0356] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0308 / +0.0133 | +0.0221 | [+0.0056, +0.0392] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0208 / +0.0200 | +0.0204 | [+0.0056, +0.0350] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.1459 / +0.1773 | +0.1616 | [+0.1051, +0.2316] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0590 / +0.0946 | +0.0768 | [+0.0307, +0.1237] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.1441 / +0.1424 | +0.1433 | [+0.0851, +0.2168] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.0978 / +0.1164 | +0.1071 | [+0.0693, +0.1534] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.1459 / +0.1773 | +0.1616 | [+0.1051, +0.2316] | 0.5955 | INSIDE-SPREAD |

## LFTb minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0181 / +0.0322 | +0.0070 | [-0.0353, +0.0569] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0119 / +0.0458 | +0.0288 | [-0.0032, +0.0618] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0048 / +0.0187 | +0.0118 | [-0.0071, +0.0324] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1910 / +0.5933 | +0.2012 | [-0.2913, +0.7570] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0034 / +0.0324 | +0.0145 | [-0.0069, +0.0406] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.3861 / +3.3274 | +0.9706 | [-1.8298, +4.1054] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0088 / +0.0071 | +0.0079 | [-0.0040, +0.0198] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0371 / +0.0196 | +0.0283 | [+0.0140, +0.0458] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0179 / +0.0075 | +0.0127 | [-0.0002, +0.0256] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0258 / +0.0770 | +0.0514 | [+0.0036, +0.1033] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0127 / +0.0373 | +0.0250 | [-0.0086, +0.0650] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0232 / +0.0668 | +0.0450 | [-0.0026, +0.0863] | 0.3535 | OPEN |
| ade6_m|all | +0.0174 / +0.0507 | +0.0341 | [+0.0026, +0.0686] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0258 / +0.0770 | +0.0514 | [+0.0036, +0.1033] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LA_s0 | 2.6243 | 3.1328 | 1.6868 | 1.2231 | 0.6686 |
| LA_s1 | 2.7023 | 3.2228 | 1.5842 | 1.1971 | 0.6732 |
| LA01_s0 | 2.7356 | 3.1335 | 1.7864 | 1.2334 | 0.7095 |
| LA01_s1 | 2.6989 | 3.1855 | 1.5627 | 1.1789 | 0.6740 |
| LFT_s0 | 2.6377 | 3.0744 | 1.6976 | 1.1890 | 0.6886 |
| LFT_s1 | 2.5825 | 3.0910 | 1.4947 | 1.1573 | 0.6540 |
| LFTb_s0 | 2.6552 | 3.0871 | 1.6795 | 1.1938 | 0.7065 |
| LFTb_s1 | 2.6332 | 3.1282 | 1.5268 | 1.1760 | 0.6615 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
