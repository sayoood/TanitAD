# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 14}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSQE"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSQE minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1920 / +0.2856 | +0.2388 | [+0.0812, +0.3955] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.9204 / +0.9381 | +0.9292 | [+0.5990, +1.2772] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0129 / +0.0475 | +0.0173 | [-0.0451, +0.0760] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.9015 / +1.7018 | +1.3016 | [+0.6061, +1.9834] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0294 / +0.0564 | +0.0429 | [+0.0223, +0.0669] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.2955 / +7.6500 | +4.9727 | [+0.8089, +8.9813] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0054 / -0.0088 | -0.0017 | [-0.0188, +0.0156] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0079 / +0.0179 | +0.0129 | [-0.0054, +0.0325] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0167 / +0.0029 | +0.0098 | [-0.0048, +0.0258] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.1849 / +1.2730 | +1.2290 | [+0.7770, +1.7135] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6384 / +0.5629 | +0.6006 | [+0.3230, +0.9018] | 0.4517 | WORSE |
| ade6_m|bulk | +1.1119 / +1.3356 | +1.2238 | [+0.6989, +1.8286] | 0.3535 | WORSE |
| ade6_m|all | +0.8543 / +0.9153 | +0.8848 | [+0.5602, +1.2326] | 0.4002 | WORSE |
| ade_tac_m|all | +1.1849 / +1.2730 | +1.2290 | [+0.7770, +1.7135] | 0.5955 | WORSE |

## LSQE minus LSQC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0051 / +0.0196 | +0.0124 | [-0.0447, +0.0695] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1541 / +0.0188 | +0.0864 | [-0.0662, +0.2530] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0368 / -0.0044 | -0.0206 | [-0.0574, +0.0102] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1879 / +0.2276 | +0.0199 | [-0.3854, +0.3679] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0037 / +0.0097 | +0.0067 | [-0.0014, +0.0152] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.9878 / +1.7665 | +0.3893 | [-1.6424, +2.4344] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0025 / +0.0000 | +0.0013 | [-0.0060, +0.0085] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0121 / +0.0013 | -0.0054 | [-0.0177, +0.0063] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0008 / +0.0000 | +0.0004 | [-0.0052, +0.0065] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1790 / +0.0194 | +0.0992 | [-0.1077, +0.3203] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1172 / -0.0520 | +0.0326 | [-0.1065, +0.1962] | 0.4517 | OPEN |
| ade6_m|bulk | +0.1534 / +0.0923 | +0.1228 | [-0.0961, +0.3635] | 0.3535 | OPEN |
| ade6_m|all | +0.1337 / +0.0138 | +0.0738 | [-0.0735, +0.2329] | 0.4002 | OPEN |
| ade_tac_m|all | +0.1790 / +0.0194 | +0.0992 | [-0.1077, +0.3203] | 0.5955 | OPEN |

## LSQE minus LSMC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0012 / -0.0424 | -0.0218 | [-0.0833, +0.0496] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0672 / -0.1765 | -0.0546 | [-0.2596, +0.1805] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0359 / -0.0001 | -0.0180 | [-0.0561, +0.0119] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2245 / -0.5316 | -0.3780 | [-0.8155, +0.0253] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0101 / -0.0268 | -0.0185 | [-0.0364, -0.0024] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.9973 / -2.2839 | -1.6406 | [-3.3376, -0.0599] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0008 / +0.0000 | +0.0004 | [-0.0077, +0.0081] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0063 / +0.0013 | -0.0025 | [-0.0138, +0.0090] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / -0.0017 | -0.0008 | [-0.0081, +0.0058] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0621 / -0.2150 | -0.0764 | [-0.3323, +0.2156] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0328 / -0.2685 | -0.1506 | [-0.3517, +0.0370] | 0.4517 | OPEN |
| ade6_m|bulk | +0.1489 / -0.0548 | +0.0470 | [-0.2011, +0.3526] | 0.3535 | OPEN |
| ade6_m|all | +0.0500 / -0.1710 | -0.0605 | [-0.2545, +0.1591] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0621 / -0.2150 | -0.0764 | [-0.3323, +0.2156] | 0.5955 | OPEN |

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
| LSMC_s0 | 3.8137 | 4.1854 | 2.1411 | 1.2858 | 0.6874 |
| LSMC_s1 | 3.7776 | 4.0018 | 1.9089 | 1.2842 | 0.7090 |
| LSQC_s0 | 3.7300 | 4.0354 | 2.1348 | 1.2867 | 0.6865 |
| LSQC_s1 | 3.5927 | 3.7853 | 1.8469 | 1.2885 | 0.7074 |
| LSQE_s0 | 3.8637 | 4.1525 | 2.1399 | 1.2499 | 0.6874 |
| LSQE_s1 | 3.6065 | 3.7333 | 1.8665 | 1.2841 | 0.7074 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
