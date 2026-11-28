# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 22}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "I10RLcT08"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.15362953829867476, "D_read": 0.15362953829867473, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False. A difference inside the spread is not an effect.

## I10RLcT08 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0675 / -0.0769 | -0.0722 | [-0.1176, -0.0341] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1918 / -0.1613 | -0.1765 | [-0.2891, -0.0787] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0070 / +0.0008 | -0.0031 | [-0.0152, +0.0080] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.1959 / -0.1000 | -0.1479 | [-0.2812, -0.0002] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0083 / -0.0042 | -0.0063 | [-0.0107, -0.0021] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.9859 / -0.5426 | -0.7642 | [-1.3368, -0.2800] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0079 / -0.0096 | -0.0088 | [-0.0140, -0.0038] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0113 / -0.0004 | -0.0058 | [-0.0152, +0.0021] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0063 / -0.0088 | -0.0075 | [-0.0127, -0.0019] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.2480 / -0.2052 | -0.2266 | [-0.3754, -0.0987] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1323 / -0.0944 | -0.1134 | [-0.2034, -0.0375] | 0.0229 | BETTER |
| ade6_m|bulk | -0.2354 / -0.2179 | -0.2267 | [-0.3762, -0.0951] | 0.1243 | BETTER |
| ade6_m|all | -0.1793 / -0.1508 | -0.1651 | [-0.2739, -0.0714] | 0.0442 | BETTER |
| ade_tac_m|all | -0.2480 / -0.2052 | -0.2266 | [-0.3754, -0.0987] | 0.0699 | BETTER |

## I10RLcT10 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0300 / -0.0428 | -0.0364 | [-0.0675, -0.0100] | 0.0512 (0.0512) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0904 / -0.0753 | -0.0828 | [-0.1342, -0.0392] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0031 / +0.0019 | -0.0006 | [-0.0084, +0.0084] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.0857 / -0.0288 | -0.0572 | [-0.1627, +0.0807] | 0.4743 (0.4743) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0048 / -0.0018 | -0.0033 | [-0.0068, +0.0000] | 0.0295 (0.0295) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.4538 / -0.2001 | -0.3269 | [-0.7938, +0.0836] | 2.2206 (2.2206) | OPEN |
| TACTICAL:lon_err_rate | -0.0075 / -0.0092 | -0.0083 | [-0.0131, -0.0038] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0054 / -0.0021 | -0.0038 | [-0.0096, +0.0019] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0054 / -0.0083 | -0.0069 | [-0.0119, -0.0019] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1175 / -0.0962 | -0.1069 | [-0.1781, -0.0468] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0869 / -0.0644 | -0.0756 | [-0.1445, -0.0224] | 0.0229 | BETTER |
| ade6_m|bulk | -0.0820 / -0.0776 | -0.0798 | [-0.1626, -0.0110] | 0.1243 | INSIDE-SPREAD |
| ade6_m|all | -0.0846 / -0.0704 | -0.0775 | [-0.1269, -0.0362] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1175 / -0.0962 | -0.1069 | [-0.1781, -0.0468] | 0.0699 | BETTER |

## I10RLcT15 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1320 / +0.1055 | +0.1187 | [+0.0669, +0.1719] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.3074 / +0.2927 | +0.3000 | [+0.1697, +0.4392] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0264 / +0.0238 | +0.0251 | [+0.0046, +0.0478] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4286 / +0.2782 | +0.3534 | [+0.1742, +0.5331] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0097 / +0.0124 | +0.0111 | [+0.0053, +0.0185] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.8677 / +1.4242 | +1.6459 | [+0.9134, +2.4468] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0075 / +0.0050 | +0.0063 | [+0.0004, +0.0125] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0104 / +0.0096 | +0.0100 | [+0.0040, +0.0165] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0113 / +0.0063 | +0.0088 | [+0.0015, +0.0160] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.4362 / +0.4197 | +0.4279 | [+0.2510, +0.6230] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2152 / +0.2094 | +0.2123 | [+0.1080, +0.3363] | 0.0229 | WORSE |
| ade6_m|bulk | +0.4135 / +0.3962 | +0.4048 | [+0.2177, +0.6103] | 0.1243 | WORSE |
| ade6_m|all | +0.3056 / +0.2946 | +0.3001 | [+0.1764, +0.4390] | 0.0442 | WORSE |
| ade_tac_m|all | +0.4362 / +0.4197 | +0.4279 | [+0.2510, +0.6230] | 0.0699 | WORSE |

## I10RLcT20 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3169 / +0.2867 | +0.3018 | [+0.1892, +0.4210] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.7630 / +0.7228 | +0.7429 | [+0.4649, +1.0461] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0632 / +0.0618 | +0.0625 | [+0.0251, +0.1056] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.1190 / +0.8651 | +0.9921 | [+0.6497, +1.3721] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0210 / +0.0193 | +0.0202 | [+0.0119, +0.0294] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +5.2213 / +3.4245 | +4.3229 | [+2.6802, +6.2090] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0208 / +0.0071 | +0.0140 | [+0.0008, +0.0284] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0325 / +0.0238 | +0.0281 | [+0.0146, +0.0434] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0296 / +0.0146 | +0.0221 | [+0.0075, +0.0379] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +1.0744 / +1.0184 | +1.0464 | [+0.6641, +1.4626] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.5076 / +0.4713 | +0.4895 | [+0.2754, +0.7284] | 0.0229 | WORSE |
| ade6_m|bulk | +1.0735 / +1.0347 | +1.0541 | [+0.5900, +1.5514] | 0.1243 | WORSE |
| ade6_m|all | +0.7657 / +0.7282 | +0.7470 | [+0.4725, +1.0424] | 0.0442 | WORSE |
| ade_tac_m|all | +1.0744 / +1.0184 | +1.0464 | [+0.6641, +1.4626] | 0.0699 | WORSE |

## I10RQ50T08 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1108 / -0.1402 | -0.1255 | [-0.1939, -0.0676] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.2592 / -0.3045 | -0.2819 | [-0.4370, -0.1503] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0114 / -0.0193 | -0.0154 | [-0.0303, -0.0028] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.3051 / -0.4361 | -0.3706 | [-0.5543, -0.2018] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0060 / -0.0090 | -0.0075 | [-0.0129, -0.0024] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.1785 / -1.6146 | -1.3966 | [-2.1991, -0.6613] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0092 / -0.0092 | -0.0092 | [-0.0156, -0.0029] | 0.0413 (0.0413) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0117 / -0.0158 | -0.0138 | [-0.0225, -0.0052] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0117 / -0.0079 | -0.0098 | [-0.0173, -0.0033] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.3434 / -0.4119 | -0.3777 | [-0.5828, -0.2025] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.2239 / -0.1934 | -0.2086 | [-0.3443, -0.0980] | 0.0229 | BETTER |
| ade6_m|bulk | -0.2803 / -0.4107 | -0.3455 | [-0.5770, -0.1721] | 0.1243 | BETTER |
| ade6_m|all | -0.2496 / -0.2925 | -0.2710 | [-0.4180, -0.1445] | 0.0442 | BETTER |
| ade_tac_m|all | -0.3434 / -0.4119 | -0.3777 | [-0.5828, -0.2025] | 0.0699 | BETTER |

## I10RQ50T10 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0503 / -0.0967 | -0.0735 | [-0.1268, -0.0314] | 0.0512 (0.0512) | BETTER |
| LONGITUDINAL:along_mae_m | -0.1378 / -0.1846 | -0.1612 | [-0.2570, -0.0754] | 0.0298 (0.0298) | BETTER |
| LATERAL:cross_mae_m | -0.0058 / -0.0183 | -0.0121 | [-0.0274, +0.0015] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | -0.1777 / -0.3050 | -0.2414 | [-0.3796, -0.1210] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0028 / -0.0056 | -0.0042 | [-0.0082, -0.0012] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.5014 / -1.0131 | -0.7573 | [-1.4041, -0.1886] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0017 / -0.0046 | -0.0031 | [-0.0092, +0.0017] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | -0.0021 / -0.0113 | -0.0067 | [-0.0152, +0.0010] | 0.0513 (0.0513) | OPEN |
| TACTICAL:joint_err_rate | -0.0017 / -0.0058 | -0.0038 | [-0.0104, +0.0010] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | -0.1781 / -0.2589 | -0.2185 | [-0.3547, -0.0986] | 0.0699 (0.0699) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0992 / -0.1410 | -0.1201 | [-0.1986, -0.0545] | 0.0229 | BETTER |
| ade6_m|bulk | -0.1696 / -0.2332 | -0.2014 | [-0.3387, -0.0832] | 0.1243 | BETTER |
| ade6_m|all | -0.1313 / -0.1831 | -0.1572 | [-0.2524, -0.0740] | 0.0442 | BETTER |
| ade_tac_m|all | -0.1781 / -0.2589 | -0.2185 | [-0.3547, -0.0986] | 0.0699 | BETTER |

## I10RQ50T15 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1715 / +0.1180 | +0.1447 | [+0.0817, +0.2139] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4010 / +0.3009 | +0.3509 | [+0.2076, +0.5082] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0150 / +0.0066 | +0.0108 | [-0.0062, +0.0265] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.5862 / +0.2282 | +0.4072 | [+0.1626, +0.7005] | 0.4743 (0.4743) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0145 / +0.0055 | +0.0100 | [+0.0034, +0.0188] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.4469 / +1.0572 | +1.7521 | [+0.7068, +3.0218] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0200 / -0.0029 | +0.0085 | [-0.0065, +0.0252] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0208 / +0.0071 | +0.0140 | [+0.0031, +0.0269] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0171 / +0.0000 | +0.0085 | [-0.0029, +0.0217] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.5466 / +0.3961 | +0.4714 | [+0.2714, +0.6842] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2352 / +0.1477 | +0.1914 | [+0.0949, +0.2948] | 0.0229 | WORSE |
| ade6_m|bulk | +0.5688 / +0.4555 | +0.5121 | [+0.2759, +0.7874] | 0.1243 | WORSE |
| ade6_m|all | +0.3873 / +0.2880 | +0.3377 | [+0.1975, +0.4877] | 0.0442 | WORSE |
| ade_tac_m|all | +0.5466 / +0.3961 | +0.4714 | [+0.2714, +0.6842] | 0.0699 | WORSE |

## I10RQ50T20 minus I10RQ50

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4051 / +0.3795 | +0.3923 | [+0.2773, +0.5193] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.0064 / +0.9720 | +0.9892 | [+0.6975, +1.3275] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0602 / +0.0799 | +0.0700 | [+0.0403, +0.1042] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.4258 / +1.0644 | +1.2451 | [+0.8510, +1.6606] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0322 / +0.0295 | +0.0309 | [+0.0220, +0.0414] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.7319 / +4.8103 | +5.7711 | [+3.8269, +7.8379] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0225 / +0.0029 | +0.0127 | [-0.0035, +0.0302] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0475 / +0.0317 | +0.0396 | [+0.0231, +0.0585] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0321 / +0.0154 | +0.0238 | [+0.0098, +0.0404] | 0.0246 (0.0246) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +1.3945 / +1.3598 | +1.3771 | [+0.9831, +1.8354] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6202 / +0.5912 | +0.6057 | [+0.3923, +0.8389] | 0.0229 | WORSE |
| ade6_m|bulk | +1.4405 / +1.4288 | +1.4347 | [+0.9406, +1.9793] | 0.1243 | WORSE |
| ade6_m|all | +0.9943 / +0.9732 | +0.9837 | [+0.7027, +1.3121] | 0.0442 | WORSE |
| ade_tac_m|all | +1.3945 / +1.3598 | +1.3771 | [+0.9831, +1.8354] | 0.0699 | WORSE |

## I10RQ50T15 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1751 / +0.1286 | +0.1519 | [+0.0654, +0.2408] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +0.4820 / +0.4188 | +0.4504 | [+0.2851, +0.6338] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0021 / +0.0076 | +0.0048 | [-0.0209, +0.0318] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +0.7833 / +0.4916 | +0.6374 | [+0.3064, +1.0298] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0249 / +0.0131 | +0.0190 | [+0.0079, +0.0322] | 0.0295 (0.0295) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.1227 / +1.2527 | +2.1877 | [+0.2802, +4.1526] | 2.2206 (2.2206) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0179 / -0.0058 | +0.0060 | [-0.0104, +0.0233] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0254 / +0.0167 | +0.0211 | [+0.0083, +0.0352] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0221 / -0.0013 | +0.0104 | [-0.0054, +0.0279] | 0.0246 (0.0246) | OPEN |
| TACTICAL:ade_tac_m | +0.6437 / +0.5525 | +0.5981 | [+0.3697, +0.8496] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3420 / +0.2155 | +0.2787 | [+0.1328, +0.4555] | 0.0229 | WORSE |
| ade6_m|bulk | +0.5985 / +0.6047 | +0.6016 | [+0.3348, +0.9292] | 0.1243 | WORSE |
| ade6_m|all | +0.4590 / +0.3930 | +0.4260 | [+0.2632, +0.6064] | 0.0442 | WORSE |
| ade_tac_m|all | +0.6437 / +0.5525 | +0.5981 | [+0.3697, +0.8496] | 0.0699 | WORSE |

## I10RQ50T20 minus I10RLc

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4088 / +0.3902 | +0.3995 | [+0.2580, +0.5502] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.0874 / +1.0899 | +1.0887 | [+0.7656, +1.4638] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0472 / +0.0808 | +0.0640 | [+0.0232, +0.1093] | 0.1058 (0.1058) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.6229 / +1.3196 | +1.4713 | [+1.0129, +1.9880] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0426 / +0.0373 | +0.0400 | [+0.0268, +0.0555] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +7.4076 / +5.0711 | +6.2394 | [+3.6555, +8.9472] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0204 / +0.0000 | +0.0102 | [-0.0067, +0.0290] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0521 / +0.0413 | +0.0467 | [+0.0279, +0.0671] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0371 / +0.0142 | +0.0256 | [+0.0075, +0.0462] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.4916 / +1.5162 | +1.5039 | [+1.0634, +2.0266] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.7270 / +0.6589 | +0.6930 | [+0.4441, +0.9674] | 0.0229 | WORSE |
| ade6_m|bulk | +1.4702 / +1.5781 | +1.5242 | [+0.9899, +2.1284] | 0.1243 | WORSE |
| ade6_m|all | +1.0660 / +1.0781 | +1.0720 | [+0.7578, +1.4422] | 0.0442 | WORSE |
| ade_tac_m|all | +1.4916 / +1.5162 | +1.5039 | [+1.0634, +2.0266] | 0.0699 | WORSE |

## I10RLcT20 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3712 / +0.3325 | +0.3519 | [+0.1884, +0.5179] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.0154 / +0.9393 | +0.9773 | [+0.5838, +1.3889] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0360 / +0.0369 | +0.0365 | [-0.0068, +0.0831] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +1.5236 / +1.1360 | +1.3298 | [+0.8998, +1.8204] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0369 / +0.0340 | +0.0355 | [+0.0249, +0.0478] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +6.3023 / +3.5655 | +4.9339 | [+2.5727, +7.3723] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0175 / +0.0100 | +0.0138 | [-0.0017, +0.0302] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0379 / +0.0304 | +0.0342 | [+0.0169, +0.0527] | 0.0513 (0.0513) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0308 / +0.0221 | +0.0265 | [+0.0110, +0.0433] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.3661 / +1.2708 | +1.3185 | [+0.7816, +1.8763] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6367 / +0.5765 | +0.6066 | [+0.3141, +0.9292] | 0.0229 | WORSE |
| ade6_m|bulk | +1.3917 / +1.3236 | +1.3577 | [+0.7805, +1.9820] | 0.1243 | WORSE |
| ade6_m|all | +0.9810 / +0.9172 | +0.9491 | [+0.5653, +1.3529] | 0.0442 | WORSE |
| ade_tac_m|all | +1.3661 / +1.2708 | +1.3185 | [+0.7816, +1.8763] | 0.0699 | WORSE |

## I10RQ50T20 minus I10R0

Seeds read [0, 1] of [0, 1]. **Q3: PRELIMINARY (COST-LATERAL COST-LONGITUDINAL COST-TACTICAL)** (decides nothing: the name needs both seeds of the pair and six baseline seeds)

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4631 / +0.4360 | +0.4495 | [+0.2645, +0.6337] | 0.0512 (0.0512) | WORSE |
| LONGITUDINAL:along_mae_m | +1.3398 / +1.3064 | +1.3231 | [+0.9047, +1.7698] | 0.0298 (0.0298) | WORSE |
| LATERAL:cross_mae_m | +0.0200 / +0.0559 | +0.0380 | [-0.0056, +0.0866] | 0.1058 (0.1058) | OPEN |
| LATERAL:heading_mae_deg | +1.9927 / +1.5982 | +1.7954 | [+1.3113, +2.3307] | 0.4743 (0.4743) | WORSE |
| LATERAL:curvature_mae_1pm | +0.0519 / +0.0459 | +0.0489 | [+0.0347, +0.0652] | 0.0295 (0.0295) | WORSE |
| LATERAL:yaw_rate_mae_degps | +8.6111 / +5.5903 | +7.1007 | [+4.0964, +10.0958] | 2.2206 (2.2206) | WORSE |
| TACTICAL:lon_err_rate | +0.0171 / +0.0029 | +0.0100 | [-0.0081, +0.0292] | 0.0413 (0.0413) | OPEN |
| TACTICAL:lat_err_rate | +0.0575 / +0.0479 | +0.0527 | [+0.0321, +0.0761] | 0.0513 (0.0513) | WORSE |
| TACTICAL:joint_err_rate | +0.0383 / +0.0217 | +0.0300 | [+0.0133, +0.0500] | 0.0246 (0.0246) | WORSE |
| TACTICAL:ade_tac_m | +1.7833 / +1.7686 | +1.7759 | [+1.2094, +2.3913] | 0.0699 (0.0699) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.8561 / +0.7641 | +0.8101 | [+0.4936, +1.1512] | 0.0229 | WORSE |
| ade6_m|bulk | +1.7884 / +1.8671 | +1.8278 | [+1.2033, +2.5235] | 0.1243 | WORSE |
| ade6_m|all | +1.2813 / +1.2671 | +1.2742 | [+0.8694, +1.7137] | 0.0442 | WORSE |
| ade_tac_m|all | +1.7833 / +1.7686 | +1.7759 | [+1.2094, +2.3913] | 0.0699 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| I10R0_s0 | 2.7348 | 3.1127 | 1.5967 | 1.2944 | 0.6469 |
| I10R0_s1 | 2.6906 | 3.1356 | 1.6480 | 1.1886 | 0.6715 |
| I10RLc_s0 | 2.9502 | 3.2418 | 1.6510 | 1.2672 | 0.6482 |
| I10RLc_s1 | 2.8796 | 3.2408 | 1.6938 | 1.1637 | 0.6790 |
| I10RLcT08_s0 | 2.7708 | 3.1095 | 1.5835 | 1.2602 | 0.6419 |
| I10RLcT08_s1 | 2.7288 | 3.1463 | 1.6169 | 1.1645 | 0.6703 |
| I10RLcT10_s0 | 2.8655 | 3.1549 | 1.6210 | 1.2641 | 0.6428 |
| I10RLcT10_s1 | 2.8092 | 3.1764 | 1.6509 | 1.1656 | 0.6707 |
| I10RLcT15_s0 | 3.2558 | 3.4570 | 1.7831 | 1.2936 | 0.6594 |
| I10RLcT15_s1 | 3.1742 | 3.4502 | 1.7992 | 1.1875 | 0.6853 |
| I10RLcT20_s0 | 3.7159 | 3.7494 | 1.9680 | 1.3305 | 0.6778 |
| I10RLcT20_s1 | 3.6078 | 3.7121 | 1.9805 | 1.2255 | 0.6936 |
| I10RQ50_s0 | 3.0218 | 3.3486 | 1.6547 | 1.2543 | 0.6532 |
| I10RQ50_s1 | 2.9845 | 3.3085 | 1.7045 | 1.1647 | 0.6778 |
| I10RQ50T08_s0 | 2.7722 | 3.1247 | 1.5439 | 1.2428 | 0.6415 |
| I10RQ50T08_s1 | 2.6920 | 3.1151 | 1.5643 | 1.1454 | 0.6699 |
| I10RQ50T10_s0 | 2.8905 | 3.2494 | 1.6045 | 1.2485 | 0.6515 |
| I10RQ50T10_s1 | 2.8014 | 3.1675 | 1.6078 | 1.1463 | 0.6719 |
| I10RQ50T15_s0 | 3.4091 | 3.5838 | 1.8262 | 1.2693 | 0.6703 |
| I10RQ50T15_s1 | 3.2725 | 3.4562 | 1.8224 | 1.1713 | 0.6778 |
| I10RQ50T20_s0 | 4.0161 | 3.9688 | 2.0598 | 1.3145 | 0.6853 |
| I10RQ50T20_s1 | 3.9576 | 3.8997 | 2.0840 | 1.2445 | 0.6932 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
