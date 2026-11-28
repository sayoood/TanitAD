# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 26}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSXP"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSXP minus LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0252 / +0.0176 | -0.0038 | [-0.0807, +0.0757] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2360 / +0.2157 | +0.2259 | [+0.0770, +0.3980] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0379 / -0.0259 | -0.0319 | [-0.0575, -0.0092] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2171 / +0.2757 | +0.2464 | [+0.0533, +0.4374] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0146 / +0.0166 | +0.0156 | [+0.0080, +0.0242] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7099 / +0.8130 | +0.0516 | [-1.3752, +1.3575] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0042 / -0.0129 | -0.0085 | [-0.0196, +0.0027] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0067 / -0.0025 | -0.0046 | [-0.0158, +0.0081] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0033 / -0.0033 | +0.0000 | [-0.0081, +0.0092] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2647 / +0.2538 | +0.2593 | [+0.0562, +0.4963] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1094 / +0.0935 | +0.1014 | [-0.0369, +0.2766] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3032 / +0.3055 | +0.3044 | [+0.1266, +0.5034] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.1978 / +0.1902 | +0.1940 | [+0.0456, +0.3693] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2647 / +0.2538 | +0.2593 | [+0.0562, +0.4963] | 0.5955 | INSIDE-SPREAD |

## LSXG minus LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0214 / +0.0858 | +0.0536 | [-0.0253, +0.1462] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2221 / +0.3459 | +0.2840 | [+0.0992, +0.5136] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0139 / -0.0083 | -0.0111 | [-0.0325, +0.0118] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.1367 / +0.7025 | +0.4196 | [+0.0448, +0.8804] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0031 / +0.0250 | +0.0140 | [+0.0012, +0.0317] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.4245 / +2.3102 | +0.9428 | [-1.0960, +2.9962] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0063 / -0.0025 | -0.0044 | [-0.0137, +0.0050] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0029 / -0.0063 | -0.0017 | [-0.0135, +0.0104] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0029 / +0.0000 | +0.0015 | [-0.0050, +0.0085] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2647 / +0.4418 | +0.3532 | [+0.0985, +0.6726] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1222 / +0.2773 | +0.1997 | [+0.0280, +0.4222] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.2810 / +0.3806 | +0.3308 | [+0.1219, +0.6070] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.1946 / +0.3244 | +0.2595 | [+0.0745, +0.4899] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2647 / +0.4418 | +0.3532 | [+0.0985, +0.6726] | 0.5955 | INSIDE-SPREAD |

## LSXC minus LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0508 / +0.0794 | +0.0651 | [-0.0119, +0.1479] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2950 / +0.3209 | +0.3080 | [+0.1308, +0.5121] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0059 / +0.0067 | +0.0063 | [-0.0164, +0.0291] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3609 / +0.5097 | +0.4353 | [+0.1990, +0.6729] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0206 / +0.0198 | +0.0202 | [+0.0121, +0.0302] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.2411 / +1.9629 | +1.1020 | [-0.7725, +2.7185] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0050 / -0.0050 | +0.0000 | [-0.0121, +0.0115] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0092 / +0.0017 | +0.0054 | [-0.0069, +0.0177] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0121 / +0.0017 | +0.0069 | [-0.0042, +0.0181] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3882 / +0.4269 | +0.4075 | [+0.1669, +0.6854] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2032 / +0.1747 | +0.1890 | [+0.0562, +0.3527] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.3750 / +0.4631 | +0.4190 | [+0.1556, +0.7179] | 0.3535 | WORSE |
| ade6_m|all | +0.2815 / +0.3062 | +0.2939 | [+0.1198, +0.4934] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3882 / +0.4269 | +0.4075 | [+0.1669, +0.6854] | 0.5955 | INSIDE-SPREAD |

## LSXZ minus LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0417 / +0.0557 | +0.0487 | [-0.0252, +0.1310] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3063 / +0.2610 | +0.2836 | [+0.1016, +0.4988] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0109 / -0.0089 | -0.0099 | [-0.0289, +0.0088] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.4018 / +0.3973 | +0.3995 | [+0.1948, +0.6350] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0144 / +0.0147 | +0.0145 | [+0.0086, +0.0218] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.6569 / +1.2708 | +0.9638 | [-0.1208, +2.0858] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0017 / +0.0017 | +0.0017 | [-0.0071, +0.0108] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0013 / +0.0004 | -0.0004 | [-0.0123, +0.0125] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0054 / +0.0050 | +0.0052 | [-0.0031, +0.0148] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3902 / +0.3303 | +0.3603 | [+0.1180, +0.6518] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1331 / +0.1105 | +0.1218 | [+0.0111, +0.2653] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.4592 / +0.3971 | +0.4281 | [+0.1538, +0.7610] | 0.3535 | WORSE |
| ade6_m|all | +0.2818 / +0.2412 | +0.2615 | [+0.0852, +0.4719] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3902 / +0.3303 | +0.3603 | [+0.1180, +0.6518] | 0.5955 | INSIDE-SPREAD |

## LSDXP minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0017 / -0.0949 | -0.0483 | [-0.1505, +0.0514] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2521 / +0.1794 | +0.2158 | [+0.0518, +0.3986] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0181 / -0.0767 | -0.0474 | [-0.0979, -0.0043] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2670 / +0.2989 | +0.2829 | [+0.0809, +0.4830] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0153 / +0.0104 | +0.0128 | [+0.0054, +0.0228] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.3583 / -0.5632 | -0.4607 | [-1.4450, +0.4269] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0138 / -0.0167 | -0.0152 | [-0.0235, -0.0069] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0242 / -0.0050 | +0.0096 | [-0.0111, +0.0336] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0013 / -0.0071 | -0.0029 | [-0.0108, +0.0054] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3112 / +0.1603 | +0.2358 | [-0.0094, +0.5058] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1176 / +0.0339 | +0.0757 | [-0.0841, +0.2627] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3616 / +0.2325 | +0.2971 | [+0.0933, +0.5382] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.2289 / +0.1244 | +0.1767 | [+0.0014, +0.3675] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3112 / +0.1603 | +0.2358 | [-0.0094, +0.5058] | 0.5955 | OPEN |

## LSDXG minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0338 / -0.0286 | +0.0026 | [-0.0693, +0.0796] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2450 / +0.1663 | +0.2056 | [+0.0585, +0.3896] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0146 / -0.0388 | -0.0121 | [-0.0517, +0.0286] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3975 / +0.1897 | +0.2936 | [+0.0948, +0.5530] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0077 / +0.0086 | +0.0081 | [+0.0028, +0.0144] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.0027 / +0.1243 | +0.0608 | [-0.7692, +0.8578] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0063 / -0.0108 | -0.0085 | [-0.0154, -0.0017] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0175 / +0.0008 | +0.0092 | [-0.0052, +0.0256] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0029 / +0.0000 | +0.0015 | [-0.0035, +0.0067] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3212 / +0.1817 | +0.2515 | [+0.0388, +0.5090] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1655 / +0.1115 | +0.1385 | [+0.0089, +0.2906] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.3159 / +0.1691 | +0.2425 | [+0.0516, +0.4944] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.2341 / +0.1378 | +0.1859 | [+0.0337, +0.3716] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3212 / +0.1817 | +0.2515 | [+0.0388, +0.5090] | 0.5955 | INSIDE-SPREAD |

## LSDXC minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0395 / -0.0052 | +0.0172 | [-0.0644, +0.1016] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2990 / +0.2707 | +0.2848 | [+0.1056, +0.4890] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0056 / -0.0442 | -0.0249 | [-0.0598, +0.0062] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.2871 / +0.4994 | +0.3932 | [+0.1312, +0.6621] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0080 / +0.0119 | +0.0099 | [+0.0039, +0.0178] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.5675 / +0.1420 | -0.2127 | [-1.6674, +0.9685] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0033 / -0.0096 | -0.0065 | [-0.0135, +0.0008] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0196 / +0.0038 | +0.0117 | [-0.0031, +0.0271] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0054 / -0.0004 | +0.0025 | [-0.0038, +0.0090] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3843 / +0.3136 | +0.3490 | [+0.1036, +0.6316] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1487 / +0.1333 | +0.1410 | [+0.0041, +0.3106] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.4320 / +0.3465 | +0.3892 | [+0.1460, +0.6712] | 0.3535 | WORSE |
| ade6_m|all | +0.2779 / +0.2306 | +0.2542 | [+0.0764, +0.4609] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3843 / +0.3136 | +0.3490 | [+0.1036, +0.6316] | 0.5955 | INSIDE-SPREAD |

## LSDXZ minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0195 / -0.0296 | -0.0051 | [-0.0843, +0.0774] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2607 / +0.2286 | +0.2447 | [+0.0788, +0.4404] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0255 / -0.0519 | -0.0387 | [-0.0728, -0.0106] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2672 / +0.3643 | +0.3158 | [+0.1128, +0.5078] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0172 / +0.0079 | +0.0126 | [+0.0048, +0.0238] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.2127 / +0.0704 | +0.1416 | [-0.5714, +0.8379] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0054 / -0.0117 | -0.0085 | [-0.0169, -0.0006] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0167 / -0.0008 | +0.0079 | [-0.0085, +0.0250] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0042 / -0.0042 | +0.0000 | [-0.0077, +0.0083] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.3143 / +0.2520 | +0.2832 | [+0.0586, +0.5510] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1035 / +0.0951 | +0.0993 | [-0.0239, +0.2554] | 0.4517 | OPEN |
| ade6_m|bulk | +0.3830 / +0.2990 | +0.3410 | [+0.1089, +0.6066] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.2310 / +0.1880 | +0.2095 | [+0.0470, +0.4044] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.3143 / +0.2520 | +0.2832 | [+0.0586, +0.5510] | 0.5955 | INSIDE-SPREAD |

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
| LSDXC_s0 | 3.3486 | 3.7289 | 1.9731 | 1.2028 | 0.7532 |
| LSDXC_s1 | 3.2396 | 3.6400 | 2.0006 | 1.3206 | 0.7478 |
| LSDXG_s0 | 3.3048 | 3.7456 | 1.9674 | 1.2230 | 0.7507 |
| LSDXG_s1 | 3.1468 | 3.6181 | 1.9772 | 1.3261 | 0.7482 |
| LSDXP_s0 | 3.2996 | 3.6978 | 1.9320 | 1.1902 | 0.7491 |
| LSDXP_s1 | 3.1335 | 3.5405 | 1.9109 | 1.2881 | 0.7411 |
| LSDXZ_s0 | 3.3017 | 3.6837 | 1.9531 | 1.1829 | 0.7520 |
| LSDXZ_s1 | 3.1971 | 3.6017 | 1.9762 | 1.3130 | 0.7441 |
| LSXC_s0 | 3.2909 | 3.7174 | 1.9987 | 1.2687 | 0.6828 |
| LSXC_s1 | 2.9975 | 3.3451 | 1.6603 | 1.2433 | 0.7061 |
| LSXG_s0 | 3.2040 | 3.6363 | 1.9693 | 1.2489 | 0.6736 |
| LSXG_s1 | 3.0157 | 3.4477 | 1.6667 | 1.2284 | 0.7045 |
| LSXP_s0 | 3.2071 | 3.6235 | 1.9226 | 1.2249 | 0.6740 |
| LSXP_s1 | 2.8815 | 3.2639 | 1.5985 | 1.2108 | 0.7011 |
| LSXZ_s0 | 3.2912 | 3.6473 | 1.9896 | 1.2519 | 0.6761 |
| LSXZ_s1 | 2.9324 | 3.2809 | 1.6366 | 1.2277 | 0.7095 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
