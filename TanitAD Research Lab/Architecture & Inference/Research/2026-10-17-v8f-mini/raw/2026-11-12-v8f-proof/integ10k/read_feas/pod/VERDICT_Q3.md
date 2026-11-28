# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 22}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10R0A5"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10R0A5 minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -8.6016 / -10.5110 | -9.5563 | [-10.8886, -8.1954] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -8.3544 / -11.1390 | -9.7467 | [-11.5108, -7.9622] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.5773 / -4.0927 | -3.8350 | [-4.4236, -3.2980] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.8403 / -4.2417 | -4.5410 | [-5.3531, -3.8419] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0124 / +0.0240 | +0.0182 | [+0.0080, +0.0317] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -29.5037 / -24.5281 | -27.0159 | [-33.1332, -21.7865] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0113 / -0.0117 | -0.0115 | [-0.0236, -0.0025] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.3151 / -0.4323 | -0.3737 | [-0.4549, -0.2911] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0659 / -0.0750 | -0.0704 | [-0.0880, -0.0536] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -14.4984 / -18.9293 | -16.7138 | [-19.6248, -13.7524] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -10.3279 / -13.5291 | -11.9285 | [-13.9590, -9.8256] | 0.0229 | BETTER |
| ade6_m|bulk | -8.9297 / -11.6019 | -10.2658 | [-12.3676, -8.0280] | 0.1243 | BETTER |
| ade6_m|all | -9.6903 / -12.6503 | -11.1703 | [-13.1147, -9.1890] | 0.0442 | BETTER |
| ade_tac_m|all | -14.4984 / -18.9293 | -16.7138 | [-19.6248, -13.7524] | 0.0699 | BETTER |

## I10RLcA5 minus I10RLcM5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -6.1760 / -7.5141 | -6.8450 | [-7.9030, -5.7986] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -5.8454 / -7.7924 | -6.8189 | [-8.2076, -5.4429] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -2.6521 / -2.8798 | -2.7659 | [-3.2175, -2.3708] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.3765 / -5.1441 | -4.7603 | [-5.5164, -3.9503] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0024 / +0.0247 | +0.0135 | [-0.0042, +0.0342] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -33.1004 / -29.8911 | -31.4958 | [-36.6748, -27.0333] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0213 / -0.0179 | -0.0196 | [-0.0357, -0.0042] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.2543 / -0.3618 | -0.3080 | [-0.3861, -0.2331] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0642 / -0.0830 | -0.0736 | [-0.0953, -0.0525] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -10.2698 / -13.3062 | -11.7880 | [-14.0556, -9.5147] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -7.0778 / -9.3271 | -8.2025 | [-9.8237, -6.6080] | 0.0229 | BETTER |
| ade6_m|bulk | -6.6195 / -8.3872 | -7.5034 | [-9.1944, -5.7930] | 0.1243 | BETTER |
| ade6_m|all | -6.8688 / -8.8985 | -7.8837 | [-9.4036, -6.3627] | 0.0442 | BETTER |
| ade_tac_m|all | -10.2698 / -13.3062 | -11.7880 | [-14.0556, -9.5147] | 0.0699 | BETTER |

## I10RQ50A5 minus I10RQ50M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -5.1306 / -5.9946 | -5.5626 | [-6.3427, -4.7859] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -4.9813 / -6.3272 | -5.6542 | [-6.7127, -4.5834] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -2.2414 / -2.3312 | -2.2863 | [-2.6834, -1.9391] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -4.3039 / -4.8630 | -4.5835 | [-5.2594, -3.9119] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | -0.0043 / +0.0107 | +0.0032 | [-0.0116, +0.0203] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -33.9716 / -30.4604 | -32.2160 | [-36.9195, -27.8593] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0233 / -0.0242 | -0.0238 | [-0.0398, -0.0083] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.2247 / -0.3243 | -0.2745 | [-0.3444, -0.2036] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0592 / -0.0888 | -0.0740 | [-0.0999, -0.0494] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -8.7528 / -10.8072 | -9.7800 | [-11.5295, -8.0218] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -6.0179 / -7.6683 | -6.8431 | [-8.1287, -5.5352] | 0.0229 | BETTER |
| ade6_m|bulk | -5.6685 / -6.7158 | -6.1921 | [-7.5839, -4.7935] | 0.1243 | BETTER |
| ade6_m|all | -5.8586 / -7.2339 | -6.5463 | [-7.7172, -5.3626] | 0.0442 | BETTER |
| ade_tac_m|all | -8.7528 / -10.8072 | -9.7800 | [-11.5295, -8.0218] | 0.0699 | BETTER |

## I10R0A5 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL GAIN-LATERAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.4687 / +1.4432 | +1.4560 | [+1.2848, +1.6129] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +3.3392 / +3.5281 | +3.4336 | [+3.1358, +3.7097] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.2481 / +0.1969 | +0.2225 | [+0.1302, +0.3128] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +0.6326 / +0.1084 | +0.3705 | [-0.3955, +1.2519] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0522 / -0.0723 | -0.0622 | [-0.0860, -0.0423] | 0.0295 (0.0295) | BETTER |
| LATERAL:yaw_rate_mae_degps | -9.2198 / -11.5213 | -10.3706 | [-13.9876, -7.5000] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | +0.2643 / +0.2401 | +0.2522 | [+0.2025, +0.2996] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.1580 / -0.0955 | -0.1267 | [-0.1788, -0.0803] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | +0.1992 / +0.1884 | +0.1938 | [+0.1508, +0.2341] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +4.8366 / +5.0931 | +4.9649 | [+4.5557, +5.3760] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.7353 / +3.7293 | +3.7323 | [+3.4151, +4.0383] | 0.0229 | WORSE |
| ade6_m|bulk | +2.6718 / +3.0589 | +2.8653 | [+2.3606, +3.3125] | 0.1243 | WORSE |
| ade6_m|all | +3.2503 / +3.4236 | +3.3369 | [+3.0597, +3.6157] | 0.0442 | WORSE |
| ade_tac_m|all | +4.8366 / +5.0931 | +4.9649 | [+4.5557, +5.3760] | 0.0699 | WORSE |

## I10RLcA5 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.1507 / +1.1517 | +1.1512 | [+0.9837, +1.3196] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +2.8864 / +3.0546 | +2.9705 | [+2.6612, +3.2625] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4116 / +0.3463 | +0.3790 | [+0.2559, +0.5142] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +2.7043 / +1.5951 | +2.1497 | [+1.1881, +3.2456] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | -0.0105 / -0.0290 | -0.0198 | [-0.0418, +0.0011] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.1327 / -2.6689 | -1.9008 | [-4.8101, +0.6983] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.1988 / +0.1709 | +0.1849 | [+0.1334, +0.2333] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.0671 / -0.0233 | -0.0452 | [-0.0922, -0.0065] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.1784 / +0.1567 | +0.1676 | [+0.1242, +0.2073] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +4.3116 / +4.5306 | +4.4211 | [+3.9698, +4.8457] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.1162 / +3.1338 | +3.1250 | [+2.8189, +3.4395] | 0.0229 | WORSE |
| ade6_m|bulk | +2.6517 / +2.9550 | +2.8034 | [+2.2872, +3.2474] | 0.1243 | WORSE |
| ade6_m|all | +2.9044 / +3.0523 | +2.9784 | [+2.6716, +3.2638] | 0.0442 | WORSE |
| ade_tac_m|all | +4.3116 / +4.5306 | +4.4211 | [+3.9698, +4.8457] | 0.0699 | WORSE |

## I10RQ50A5 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.0846 / +1.0728 | +1.0787 | [+0.9180, +1.2377] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +2.9191 / +2.9332 | +2.9262 | [+2.6553, +3.1932] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.4623 / +0.3931 | +0.4277 | [+0.2948, +0.5582] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +3.4032 / +1.9677 | +2.6854 | [+1.5514, +3.9432] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0082 / -0.0181 | -0.0050 | [-0.0299, +0.0182] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.8333 / -0.7989 | +0.5172 | [-2.2676, +3.1894] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.1797 / +0.1371 | +0.1584 | [+0.1029, +0.2086] | 0.0413 (0.0413) | WORSE |
| TACTICAL:lat_err_rate | -0.0392 / -0.0063 | -0.0227 | [-0.0603, +0.0119] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | +0.1680 / +0.1476 | +0.1578 | [+0.1158, +0.1979] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +4.3564 / +4.3976 | +4.3770 | [+3.9791, +4.7744] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.1370 / +2.9619 | +3.0495 | [+2.7206, +3.3733] | 0.0229 | WORSE |
| ade6_m|bulk | +2.6977 / +2.9666 | +2.8321 | [+2.3371, +3.2749] | 0.1243 | WORSE |
| ade6_m|all | +2.9367 / +2.9640 | +2.9504 | [+2.6803, +3.2189] | 0.0442 | WORSE |
| ade_tac_m|all | +4.3564 / +4.3976 | +4.3770 | [+3.9791, +4.7744] | 0.0699 | WORSE |

## I10R0A10 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2931 / -0.2878 | -0.2904 | [-0.3631, -0.2284] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0023 / +0.0046 | +0.0012 | [-0.0247, +0.0277] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | -0.1374 / -0.1722 | -0.1548 | [-0.2198, -0.1064] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -0.2308 / -0.2938 | -0.2623 | [-0.4970, -0.0558] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0193 / +0.0255 | +0.0224 | [+0.0142, +0.0339] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -4.1324 / -5.4913 | -4.8119 | [-6.8406, -3.2619] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0200 / -0.0217 | -0.0208 | [-0.0383, -0.0062] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0338 / -0.0092 | -0.0215 | [-0.0425, -0.0025] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0229 / -0.0175 | -0.0202 | [-0.0338, -0.0083] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1109 / -0.1269 | -0.1189 | [-0.1797, -0.0662] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0839 / -0.1055 | -0.0947 | [-0.1607, -0.0398] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0783 / -0.0774 | -0.0779 | [-0.1184, -0.0411] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.0814 / -0.0927 | -0.0870 | [-0.1289, -0.0507] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1109 / -0.1269 | -0.1189 | [-0.1797, -0.0662] | 0.0699 | BETTER |

## I10RLcA5 minus I10R0A5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2637 / -0.2457 | -0.2547 | [-0.3660, -0.1495] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2004 / -0.2570 | -0.2287 | [-0.5126, +0.0835] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.1363 / +0.1245 | +0.1304 | [+0.0214, +0.2498] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +2.7501 / +1.8504 | +2.3003 | [+1.4109, +3.2294] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0586 / +0.0618 | +0.0602 | [+0.0430, +0.0794] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.7743 / +8.5564 | +8.6653 | [+6.5938, +10.8493] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0688 / -0.0663 | -0.0675 | [-0.0880, -0.0477] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.0963 / +0.0788 | +0.0875 | [+0.0611, +0.1139] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | -0.0196 / -0.0242 | -0.0219 | [-0.0356, -0.0077] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.2333 / -0.3101 | -0.2717 | [-0.7124, +0.2033] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.4899 / -0.4903 | -0.4901 | [-0.8150, -0.1627] | 0.0229 | BETTER |
| ade6_m|bulk | +0.2982 / +0.1851 | +0.2416 | [-0.1522, +0.6875] | 0.1243 | OPEN |
| ade6_m|all | -0.1305 / -0.1823 | -0.1564 | [-0.4637, +0.1816] | 0.0442 | OPEN |
| ade_tac_m|all | -0.2333 / -0.3101 | -0.2717 | [-0.7124, +0.2033] | 0.0699 | OPEN |

## I10RQ50A5 minus I10R0A5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-TACTICAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3261 / -0.3139 | -0.3200 | [-0.4412, -0.2016] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.0867 / -0.2605 | -0.1736 | [-0.4889, +0.1793] | 0.0298 (0.0298) | OPEN |
| LATERAL:cross_mae_m | +0.1741 / +0.1722 | +0.1732 | [+0.0490, +0.3036] | 0.1058 (0.1058) | WORSE |
| LATERAL:heading_mae_deg | +3.6851 / +2.4002 | +3.0427 | [+1.9416, +4.2153] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0847 / +0.0713 | +0.0780 | [+0.0555, +0.1036] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +12.0483 / +10.1571 | +11.1027 | [+8.4810, +13.9685] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | -0.0900 / -0.1030 | -0.0965 | [-0.1206, -0.0717] | 0.0413 (0.0413) | BETTER |
| TACTICAL:lat_err_rate | +0.1288 / +0.1055 | +0.1171 | [+0.0857, +0.1482] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | -0.0250 / -0.0346 | -0.0298 | [-0.0455, -0.0135] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -0.0914 / -0.2868 | -0.1891 | [-0.6774, +0.3378] | 0.0699 (0.0699) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.3623 / -0.5945 | -0.4784 | [-0.8502, -0.0729] | 0.0229 | BETTER |
| ade6_m|bulk | +0.3739 / +0.3459 | +0.3599 | [-0.0719, +0.8386] | 0.1243 | OPEN |
| ade6_m|all | -0.0266 / -0.1657 | -0.0961 | [-0.4348, +0.2751] | 0.0442 | OPEN |
| ade_tac_m|all | -0.0914 / -0.2868 | -0.1891 | [-0.6774, +0.3378] | 0.0699 | OPEN |

## I10R0A5L minus I10R0A5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.9487 / -0.9093 | -0.9290 | [-0.9981, -0.8622] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -1.5312 / -1.5575 | -1.5444 | [-1.6626, -1.4273] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.2214 / -0.2568 | -0.2391 | [-0.2966, -0.1826] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -0.2518 / -0.8261 | -0.5389 | [-1.0899, +0.0723] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0312 / +0.0279 | +0.0296 | [+0.0112, +0.0492] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -1.0633 / -0.9065 | -0.9849 | [-2.0585, +0.1462] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | +0.0067 / -0.0208 | -0.0071 | [-0.0282, +0.0140] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0058 / -0.0158 | -0.0050 | [-0.0223, +0.0119] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0021 / -0.0246 | -0.0133 | [-0.0323, +0.0046] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -2.3851 / -2.4113 | -2.3982 | [-2.5791, -2.2226] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -1.8126 / -1.7926 | -1.8026 | [-1.9286, -1.6681] | 0.0229 | BETTER |
| ade6_m|bulk | -1.3580 / -1.4259 | -1.3920 | [-1.5659, -1.2216] | 0.1243 | BETTER |
| ade6_m|all | -1.6053 / -1.6254 | -1.6153 | [-1.7381, -1.4954] | 0.0442 | BETTER |
| ade_tac_m|all | -2.3851 / -2.4113 | -2.3982 | [-2.5791, -2.2226] | 0.0699 | BETTER |

## I10R0A5L minus I10R0M5

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL GAIN-LATERAL GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -9.5503 / -11.4203 | -10.4853 | [-11.8146, -9.1172] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -9.8857 / -12.6965 | -11.2911 | [-13.0949, -9.4457] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -3.7986 / -4.3495 | -4.0741 | [-4.6912, -3.4976] | 0.1058 (0.1058) | BETTER |
| LATERAL:heading_mae_deg | -5.0920 / -5.0678 | -5.0799 | [-5.7002, -4.4700] | 0.4743 (0.4743) | BETTER |
| LATERAL:curvature_mae_1pm | +0.0436 / +0.0519 | +0.0478 | [+0.0278, +0.0727] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | -30.5670 / -25.4346 | -28.0008 | [-34.1789, -22.6481] | 2.2206 (2.2206) | BETTER |
| TACTICAL:lon_err_rate | -0.0046 / -0.0325 | -0.0185 | [-0.0444, +0.0004] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.3093 / -0.4481 | -0.3787 | [-0.4701, -0.2850] | 0.0513 (0.0513) | BETTER |
| TACTICAL:joint_err_rate | -0.0679 / -0.0996 | -0.0838 | [-0.1122, -0.0588] | 0.0246 (0.0246) | BETTER |
| TACTICAL:ade_tac_m | -16.8835 / -21.3406 | -19.1120 | [-22.0654, -16.0944] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -12.1405 / -15.3217 | -13.7311 | [-15.7841, -11.6485] | 0.0229 | BETTER |
| ade6_m|bulk | -10.2877 / -13.0278 | -11.6578 | [-13.7891, -9.3793] | 0.1243 | BETTER |
| ade6_m|all | -11.2956 / -14.2757 | -12.7856 | [-14.7641, -10.7667] | 0.0442 | BETTER |
| ade_tac_m|all | -16.8835 / -21.3406 | -19.1120 | [-22.0654, -16.0944] | 0.0699 | BETTER |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10R0A10_s0 | 2.6535 | 3.0288 | 1.3037 | 1.1570 | 0.6240 |
| I10R0A10_s1 | 2.5979 | 3.0301 | 1.3602 | 1.0165 | 0.6540 |
| I10R0A5_s0 | 5.9851 | 6.8480 | 3.0654 | 1.5425 | 0.8462 |
| I10R0A5_s1 | 6.1142 | 6.8649 | 3.0912 | 1.3855 | 0.8599 |
| I10R0A5L_s0 | 4.3798 | 5.0354 | 2.1168 | 1.3211 | 0.8441 |
| I10R0A5L_s1 | 4.4888 | 5.0723 | 2.1819 | 1.1287 | 0.8353 |
| I10R0M5_s0 | 15.6754 | 17.1759 | 11.6670 | 5.1198 | 0.9120 |
| I10R0M5_s1 | 18.7645 | 20.3940 | 13.6022 | 5.4782 | 0.9350 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcA5_s0 | 5.8546 | 6.3580 | 2.8017 | 1.6788 | 0.8266 |
| I10RLcA5_s1 | 5.9319 | 6.3746 | 2.8455 | 1.5100 | 0.8358 |
| I10RLcM5_s0 | 12.7234 | 13.4358 | 8.9777 | 4.3309 | 0.8908 |
| I10RLcM5_s1 | 14.8304 | 15.7017 | 10.3595 | 4.3898 | 0.9187 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50A5_s0 | 5.9585 | 6.4856 | 2.7393 | 1.7166 | 0.8212 |
| I10RQ50A5_s1 | 5.9485 | 6.2704 | 2.7773 | 1.5578 | 0.8253 |
| I10RQ50M5_s0 | 11.8171 | 12.5035 | 7.8699 | 3.9580 | 0.8804 |
| I10RQ50M5_s1 | 13.1824 | 13.9387 | 8.7719 | 3.8890 | 0.9141 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
