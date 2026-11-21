# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "GRT6"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## GRT6 minus GFT6

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0153 / +0.0151 | +0.0152 | [-0.0431, +0.0771] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0443 / +0.0615 | +0.0529 | [-0.0427, +0.1680] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0397 / +0.0095 | +0.0246 | [-0.0049, +0.0520] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.3700 / -0.0330 | -0.2015 | [-0.5266, +0.1278] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0208 / -0.0001 | -0.0104 | [-0.0262, +0.0054] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.0560 / -0.9481 | -1.0021 | [-2.0425, +0.0247] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0388 / +0.0229 | -0.0079 | [-0.0469, +0.0321] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0100 / +0.0221 | +0.0060 | [-0.0171, +0.0290] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0413 / +0.0292 | -0.0060 | [-0.0482, +0.0379] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1020 / +0.1157 | +0.1088 | [-0.0254, +0.2738] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0540 / +0.0493 | +0.0517 | [-0.0301, +0.1467] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0817 / +0.1104 | +0.0961 | [-0.0380, +0.2461] | 0.3535 | OPEN |
| ade6_m|all | +0.0667 / +0.0772 | +0.0719 | [-0.0185, +0.1823] | 0.4002 | OPEN |
| ade_tac_m|all | +0.1020 / +0.1157 | +0.1088 | [-0.0254, +0.2738] | 0.5955 | OPEN |

## GFT6b minus GFT6

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0852 / -0.0777 | +0.0038 | [-0.1209, +0.1240] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1766 / -0.0764 | +0.0501 | [-0.1215, +0.2326] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0148 / +0.0680 | +0.0414 | [-0.0043, +0.0881] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.3077 / -0.2028 | +0.5525 | [-0.3661, +1.5797] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0587 / -0.0214 | +0.0187 | [-0.0282, +0.0686] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +4.4245 / -1.8916 | +1.2664 | [-2.6544, +5.1291] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0179 / -0.0163 | +0.0008 | [-0.0290, +0.0286] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0067 / -0.0208 | -0.0071 | [-0.0331, +0.0186] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / -0.0183 | -0.0092 | [-0.0321, +0.0110] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2457 / -0.0296 | +0.1080 | [-0.0989, +0.3216] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1162 / -0.0512 | +0.0325 | [-0.0970, +0.1591] | 0.4517 | OPEN |
| ade6_m|bulk | +0.2267 / +0.0171 | +0.1219 | [-0.0512, +0.3273] | 0.3535 | OPEN |
| ade6_m|all | +0.1666 / -0.0201 | +0.0733 | [-0.0667, +0.2181] | 0.4002 | OPEN |
| ade_tac_m|all | +0.2457 / -0.0296 | +0.1080 | [-0.0989, +0.3216] | 0.5955 | OPEN |

## GFT6 minus GFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0202 / +0.0131 | +0.0166 | [-0.0574, +0.0928] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | -0.0799 / +0.0337 | -0.0231 | [-0.1312, +0.0905] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.1035 / -0.0454 | -0.0745 | [-0.1236, -0.0237] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -1.3412 / +0.0596 | -0.6408 | [-1.6090, +0.2497] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0346 / +0.0005 | -0.0171 | [-0.0441, +0.0079] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -3.4597 / +0.2200 | -1.6198 | [-4.3990, +1.1186] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0513 / -0.0104 | -0.0308 | [-0.0673, +0.0035] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0046 / -0.0083 | -0.0019 | [-0.0217, +0.0219] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0413 / +0.0004 | -0.0204 | [-0.0575, +0.0150] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.2141 / -0.0259 | -0.1200 | [-0.2846, +0.0524] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2133 / -0.0025 | -0.1079 | [-0.2639, +0.0500] | 0.4517 | OPEN |
| ade6_m|bulk | -0.0548 / -0.0381 | -0.0464 | [-0.1607, +0.0741] | 0.3535 | OPEN |
| ade6_m|all | -0.1410 / -0.0187 | -0.0799 | [-0.1889, +0.0328] | 0.4002 | OPEN |
| ade_tac_m|all | -0.2141 / -0.0259 | -0.1200 | [-0.2846, +0.0524] | 0.5955 | OPEN |

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
| GFT6_s0 | 2.7051 | 3.1752 | 1.7108 | 1.1587 | 0.7036 |
| GFT6_s1 | 2.6642 | 3.1115 | 1.6200 | 1.1726 | 0.6478 |
| GFT6b_s0 | 2.8717 | 3.2914 | 1.7960 | 1.1735 | 0.7036 |
| GFT6b_s1 | 2.6441 | 3.0602 | 1.5423 | 1.2405 | 0.6294 |
| GRT6_s0 | 2.7718 | 3.2292 | 1.7261 | 1.1984 | 0.6624 |
| GRT6_s1 | 2.7414 | 3.1608 | 1.6352 | 1.1820 | 0.6769 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
