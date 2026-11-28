# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 16}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LC"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LC minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0693 / +0.1002 | +0.0847 | [+0.0471, +0.1261] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.0700 / +0.1078 | +0.0889 | [+0.0500, +0.1293] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0008 / +0.0214 | +0.0103 | [-0.0119, +0.0353] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.0705 / +1.0797 | +0.5046 | [-0.1906, +1.3536] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0074 / +0.0390 | +0.0232 | [+0.0022, +0.0485] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.4203 / +4.4755 | +2.0276 | [-0.8997, +5.5126] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0238 / +0.0171 | +0.0204 | [+0.0071, +0.0342] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0075 / +0.0200 | +0.0138 | [+0.0010, +0.0273] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0267 / +0.0133 | +0.0200 | [+0.0050, +0.0357] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0950 / +0.1680 | +0.1315 | [+0.0679, +0.1966] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0480 / +0.1251 | +0.0866 | [+0.0282, +0.1520] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0809 / +0.0958 | +0.0884 | [+0.0455, +0.1287] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.0630 / +0.1118 | +0.0874 | [+0.0447, +0.1307] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.0950 / +0.1680 | +0.1315 | [+0.0679, +0.1966] | 0.5955 | INSIDE-SPREAD |

## LO minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4485 / +0.8198 | +0.6341 | [+0.3523, +1.0067] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.1063 / +1.7988 | +1.4525 | [+0.8328, +2.2549] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0952 / +0.1652 | +0.1302 | [+0.0668, +0.2032] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.9200 / +3.0797 | +2.4998 | [+1.5481, +3.5869] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0460 / +0.0439 | +0.0449 | [+0.0247, +0.0681] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +6.2225 / +9.1363 | +7.6794 | [+4.7858, +11.2574] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0304 / +0.0358 | +0.0027 | [-0.0448, +0.0504] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0554 / +0.1359 | +0.0957 | [+0.0417, +0.1544] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0042 / +0.0629 | +0.0294 | [-0.0198, +0.0778] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.5379 / +2.6042 | +2.0710 | [+1.1987, +3.2244] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.9485 / +1.5536 | +1.2511 | [+0.7108, +2.0281] | 0.4517 | WORSE |
| ade6_m|bulk | +1.3238 / +2.1655 | +1.7447 | [+0.9229, +2.7919] | 0.3535 | WORSE |
| ade6_m|all | +1.1197 / +1.8327 | +1.4762 | [+0.8539, +2.2773] | 0.4002 | WORSE |
| ade_tac_m|all | +1.5379 / +2.6042 | +2.0710 | [+1.1987, +3.2244] | 0.5955 | WORSE |

## LCO minus LFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.6210 / +0.7363 | +0.6786 | [+0.4544, +0.9468] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.5858 / +1.6987 | +1.6422 | [+1.0287, +2.3316] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.2692 / +0.1358 | +0.2025 | [+0.1074, +0.3210] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +2.3822 / +2.1903 | +2.2862 | [+1.7073, +2.9342] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0506 / +0.0268 | +0.0387 | [+0.0164, +0.0606] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +5.1296 / +7.1198 | +6.1247 | [+3.8603, +8.6371] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0050 / +0.0417 | +0.0233 | [-0.0085, +0.0581] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0746 / +0.1000 | +0.0873 | [+0.0591, +0.1195] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0146 / +0.0588 | +0.0367 | [+0.0006, +0.0744] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +2.2882 / +2.4287 | +2.3585 | [+1.5112, +3.2781] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.2182 / +1.2697 | +1.2439 | [+0.7714, +1.8052] | 0.4517 | WORSE |
| ade6_m|bulk | +2.1057 / +2.2653 | +2.1855 | [+1.2792, +3.2476] | 0.3535 | WORSE |
| ade6_m|all | +1.6229 / +1.7237 | +1.6733 | [+1.0651, +2.3314] | 0.4002 | WORSE |
| ade_tac_m|all | +2.2882 / +2.4287 | +2.3585 | [+1.5112, +3.2781] | 0.5955 | WORSE |

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

## LCO minus LO

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1725 / -0.0835 | +0.0445 | [-0.1304, +0.2181] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.4794 / -0.1001 | +0.1897 | [-0.2072, +0.5941] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.1740 / -0.0295 | +0.0723 | [-0.0538, +0.2276] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.4623 / -0.8895 | -0.2136 | [-1.1738, +0.7277] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0046 / -0.0171 | -0.0062 | [-0.0279, +0.0111] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.0929 / -2.0165 | -1.5547 | [-3.5646, +0.2194] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0354 / +0.0058 | +0.0206 | [-0.0017, +0.0444] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0192 / -0.0358 | -0.0083 | [-0.0450, +0.0279] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0188 / -0.0042 | +0.0073 | [-0.0127, +0.0281] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.7504 / -0.1755 | +0.2874 | [-0.3219, +0.9255] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2697 / -0.2839 | -0.0071 | [-0.3896, +0.3631] | 0.4517 | OPEN |
| ade6_m|bulk | +0.7819 / +0.0997 | +0.4408 | [-0.0677, +0.9757] | 0.3535 | OPEN |
| ade6_m|all | +0.5032 / -0.1090 | +0.1971 | [-0.2173, +0.6184] | 0.4002 | OPEN |
| ade_tac_m|all | +0.7504 / -0.1755 | +0.2874 | [-0.3219, +0.9255] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| LC_s0 | 2.7007 | 3.1225 | 1.7669 | 1.1882 | 0.7153 |
| LC_s1 | 2.6943 | 3.2161 | 1.5949 | 1.1787 | 0.6674 |
| LCO_s0 | 4.2606 | 4.2926 | 2.3186 | 1.4582 | 0.7032 |
| LCO_s1 | 4.3062 | 4.3607 | 2.2309 | 1.2931 | 0.7128 |
| LFT_s0 | 2.6377 | 3.0744 | 1.6976 | 1.1890 | 0.6886 |
| LFT_s1 | 2.5825 | 3.0910 | 1.4947 | 1.1573 | 0.6540 |
| LFTb_s0 | 2.6552 | 3.0871 | 1.6795 | 1.1938 | 0.7065 |
| LFTb_s1 | 2.6332 | 3.1282 | 1.5268 | 1.1760 | 0.6615 |
| LO_s0 | 3.7574 | 4.0229 | 2.1461 | 1.2842 | 0.6845 |
| LO_s1 | 4.4152 | 4.6446 | 2.3144 | 1.3225 | 0.7170 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
