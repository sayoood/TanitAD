# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LCS"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LCS minus LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1498 / +0.2070 | +0.0286 | [-0.2013, +0.2617] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.1372 / +0.0840 | -0.0266 | [-0.1835, +0.1384] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0497 / +0.1521 | +0.0512 | [-0.0734, +0.1885] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -1.0795 / +0.2142 | -0.4327 | [-1.3353, +0.4457] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0287 / -0.0220 | -0.0253 | [-0.0411, -0.0118] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -3.7943 / +0.6263 | -1.5840 | [-4.9107, +1.1766] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0521 / +0.0196 | +0.0358 | [+0.0025, +0.0678] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.1092 / +0.0046 | -0.0523 | [-0.1274, +0.0219] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0413 / +0.0292 | +0.0352 | [+0.0079, +0.0613] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.2437 / +0.2743 | +0.0153 | [-0.3134, +0.3450] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1328 / +0.2220 | +0.0446 | [-0.1955, +0.2886] | 0.4517 | OPEN |
| ade6_m|bulk | -0.1985 / +0.1355 | -0.0315 | [-0.2592, +0.1884] | 0.3535 | OPEN |
| ade6_m|all | -0.1627 / +0.1825 | +0.0099 | [-0.2096, +0.2315] | 0.4002 | OPEN |
| ade_tac_m|all | -0.2437 / +0.2743 | +0.0153 | [-0.3134, +0.3450] | 0.5955 | OPEN |

## LFTRL minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0008 / +0.0654 | +0.0331 | [-0.0485, +0.1239] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1907 / +0.2095 | +0.2001 | [+0.0515, +0.3800] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0226 / -0.0178 | -0.0202 | [-0.0438, +0.0065] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3457 / +0.4608 | +0.4032 | [+0.2425, +0.5971] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0163 / +0.0171 | +0.0167 | [+0.0092, +0.0256] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.3052 / +1.5944 | +0.9498 | [-0.2285, +2.0579] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0108 / +0.0075 | -0.0017 | [-0.0173, +0.0146] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0054 / +0.0138 | +0.0096 | [-0.0019, +0.0236] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0079 | +0.0040 | [-0.0058, +0.0148] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2231 / +0.2400 | +0.2315 | [+0.0302, +0.4763] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0789 / +0.0945 | +0.0867 | [-0.0476, +0.2510] | 0.4517 | OPEN |
| ade6_m|bulk | +0.2775 / +0.2844 | +0.2809 | [+0.0971, +0.4939] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.1694 / +0.1811 | +0.1753 | [+0.0271, +0.3556] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2231 / +0.2400 | +0.2315 | [+0.0302, +0.4763] | 0.5955 | INSIDE-SPREAD |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LCS_s0 | 2.8466 | 3.3814 | 1.7981 | 1.2131 | 0.7120 |
| LCS_s1 | 2.8738 | 3.3924 | 1.7879 | 1.3888 | 0.7336 |
| LFT_s0 | 2.6377 | 3.0744 | 1.6976 | 1.1890 | 0.6886 |
| LFT_s1 | 2.5825 | 3.0910 | 1.4947 | 1.1573 | 0.6540 |
| LFTRL_s0 | 2.8072 | 3.1533 | 1.6984 | 1.1664 | 0.6886 |
| LFTRL_s1 | 2.7636 | 3.1854 | 1.5601 | 1.1395 | 0.6619 |
| LS_s0 | 3.0093 | 3.5142 | 1.9479 | 1.2628 | 0.6707 |
| LS_s1 | 2.6913 | 3.1704 | 1.5809 | 1.2366 | 0.7045 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
