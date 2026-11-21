# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IPlx_RLFT"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IPlx RLFT minus IPox RLFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1984 / +0.1344 | +0.1664 | [-0.0005, +0.3510] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.6533 / +0.6009 | +0.6271 | [+0.2273, +1.1079] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0526 / -0.0502 | -0.0514 | [-0.0764, -0.0276] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4366 / +0.2359 | +0.3363 | [+0.1032, +0.5782] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0174 / +0.0172 | +0.0173 | [+0.0094, +0.0273] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.9416 / +0.0004 | +0.4710 | [-0.7478, +1.7879] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0004 / +0.0071 | +0.0033 | [-0.0052, +0.0146] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0013 / +0.0017 | +0.0002 | [-0.0086, +0.0119] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0004 / +0.0058 | +0.0031 | [-0.0046, +0.0133] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.8618 / +0.7952 | +0.8285 | [+0.2712, +1.5057] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3454 / +0.3049 | +0.3251 | [+0.0475, +0.6864] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.9155 / +0.8686 | +0.8920 | [+0.3411, +1.5510] | 0.3535 | WORSE |
| ade6_m|all | +0.6054 / +0.5619 | +0.5836 | [+0.1926, +1.0601] | 0.4002 | WORSE |
| ade_tac_m|all | +0.8618 / +0.7952 | +0.8285 | [+0.2712, +1.5057] | 0.5955 | WORSE |

## IPkx RLFT minus IPox RLFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1979 / +0.2427 | +0.2203 | [+0.0608, +0.4032] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6755 / +0.8452 | +0.7604 | [+0.3339, +1.2316] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0487 / -0.0460 | -0.0474 | [-0.0700, -0.0261] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2423 / +0.1561 | +0.1992 | [+0.0369, +0.3641] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0107 / +0.0168 | +0.0137 | [+0.0064, +0.0216] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.6175 / +0.4194 | +0.5185 | [-0.3370, +1.4369] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0013 / +0.0192 | +0.0090 | [-0.0054, +0.0252] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0063 / +0.0038 | -0.0013 | [-0.0113, +0.0117] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0138 | +0.0069 | [-0.0033, +0.0198] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.8956 / +1.1445 | +1.0201 | [+0.4332, +1.6814] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4021 / +0.5804 | +0.4912 | [+0.1514, +0.9152] | 0.4517 | WORSE |
| ade6_m|bulk | +0.9005 / +1.0749 | +0.9877 | [+0.4367, +1.6315] | 0.3535 | WORSE |
| ade6_m|all | +0.6294 / +0.8059 | +0.7176 | [+0.3048, +1.1833] | 0.4002 | WORSE |
| ade_tac_m|all | +0.8956 / +1.1445 | +1.0201 | [+0.4332, +1.6814] | 0.5955 | WORSE |

## IPkx RLFT minus IPcx RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0727 / +0.0365 | +0.0546 | [+0.0132, +0.0944] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1178 / +0.0013 | +0.0595 | [-0.0510, +0.1752] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0126 / +0.0149 | +0.0138 | [+0.0042, +0.0241] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2019 / -0.1327 | -0.1673 | [-0.3095, -0.0308] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0061 / -0.0081 | -0.0071 | [-0.0127, -0.0028] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.4335 / -0.0941 | +0.1697 | [-0.4587, +0.7572] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0046 / +0.0017 | +0.0031 | [-0.0023, +0.0083] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0042 / +0.0029 | +0.0035 | [-0.0006, +0.0079] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0029 / +0.0029 | +0.0029 | [-0.0006, +0.0067] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1773 / +0.0158 | +0.0966 | [-0.0616, +0.2569] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0722 / -0.0235 | +0.0244 | [-0.0541, +0.0986] | 0.4517 | OPEN |
| ade6_m|bulk | +0.1845 / +0.0486 | +0.1165 | [-0.0660, +0.3182] | 0.3535 | OPEN |
| ade6_m|all | +0.1234 / +0.0094 | +0.0664 | [-0.0451, +0.1795] | 0.4002 | OPEN |
| ade_tac_m|all | +0.1773 / +0.0158 | +0.0966 | [-0.0616, +0.2569] | 0.5955 | OPEN |

## IPkx RLFT minus IPlx RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0005 / +0.1083 | +0.0539 | [-0.0288, +0.1662] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0222 / +0.2444 | +0.1333 | [-0.0506, +0.3908] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0038 / +0.0043 | +0.0040 | [-0.0142, +0.0234] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1642 / -0.0411 | -0.1027 | [-0.2763, +0.0703] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0058 / +0.0015 | -0.0022 | [-0.0097, +0.0059] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.6121 / +0.1351 | -0.2385 | [-0.9148, +0.5471] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0008 / +0.0121 | +0.0056 | [-0.0038, +0.0169] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0050 / +0.0021 | -0.0015 | [-0.0096, +0.0075] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0004 / +0.0079 | +0.0038 | [-0.0025, +0.0117] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0339 / +0.3493 | +0.1916 | [-0.0628, +0.5532] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0567 / +0.2755 | +0.1661 | [+0.0063, +0.4516] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.0150 / +0.2063 | +0.0957 | [-0.1584, +0.4121] | 0.3535 | OPEN |
| ade6_m|all | +0.0240 / +0.2440 | +0.1340 | [-0.0437, +0.3866] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0339 / +0.3493 | +0.1916 | [-0.0628, +0.5532] | 0.5955 | OPEN |

## IPcx RLFT minus IPox RLFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1252 / +0.2062 | +0.1657 | [+0.0085, +0.3607] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.5577 / +0.8439 | +0.7008 | [+0.2757, +1.2185] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0614 / -0.0609 | -0.0611 | [-0.0868, -0.0378] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4465 / +0.3514 | +0.3989 | [+0.1709, +0.6237] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0167 / +0.0232 | +0.0200 | [+0.0113, +0.0302] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1845 / +0.6948 | +0.4397 | [-0.6863, +1.5571] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0058 / +0.0175 | +0.0058 | [-0.0115, +0.0248] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0104 / +0.0008 | -0.0048 | [-0.0156, +0.0094] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0029 / +0.0108 | +0.0040 | [-0.0069, +0.0173] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.7183 / +1.1288 | +0.9235 | [+0.3244, +1.6489] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3298 / +0.6039 | +0.4669 | [+0.1119, +0.9246] | 0.4517 | WORSE |
| ade6_m|bulk | +0.7160 / +1.0263 | +0.8712 | [+0.3449, +1.4923] | 0.3535 | WORSE |
| ade6_m|all | +0.5059 / +0.7965 | +0.6512 | [+0.2305, +1.1622] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7183 / +1.1288 | +0.9235 | [+0.3244, +1.6489] | 0.5955 | WORSE |

## IPksx RLFT minus IPkx RLFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.2300 / +1.3002 | +1.2651 | [+1.0167, +1.5226] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +3.7574 / +3.9013 | +3.8293 | [+3.1514, +4.5344] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0718 / +0.1309 | +0.1014 | [-0.0000, +0.2239] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +3.7973 / +3.1275 | +3.4624 | [+2.3235, +4.5810] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.2054 / +0.1992 | +0.2023 | [+0.1562, +0.2478] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +14.1359 / +13.9877 | +14.0618 | [+10.6714, +17.3909] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0634 / +0.0750 | +0.0692 | [+0.0485, +0.0911] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0017 / +0.0254 | +0.0135 | [-0.0204, +0.0467] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0304 / +0.0509 | +0.0406 | [+0.0206, +0.0621] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +5.1399 / +5.3923 | +5.2661 | [+4.3086, +6.2470] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.5346 / +3.7481 | +3.6413 | [+2.8784, +4.4531] | 0.4517 | WORSE |
| ade6_m|bulk | +3.8598 / +4.0014 | +3.9306 | [+2.9033, +4.9969] | 0.3535 | WORSE |
| ade6_m|all | +3.6829 / +3.8636 | +3.7732 | [+3.0894, +4.4856] | 0.4002 | WORSE |
| ade_tac_m|all | +5.1399 / +5.3923 | +5.2661 | [+4.3086, +6.2470] | 0.5955 | WORSE |

## IPkx RLFT minus IPix RLFT

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1165 / -0.1074 | -0.1120 | [-0.1717, -0.0591] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.3143 / -0.2996 | -0.3069 | [-0.4668, -0.1693] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0093 / -0.0026 | -0.0060 | [-0.0196, +0.0037] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2453 / -0.2815 | -0.2634 | [-0.4237, -0.1314] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0110 / -0.0133 | -0.0121 | [-0.0196, -0.0055] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.2000 / -1.3725 | -1.2862 | [-2.1654, -0.5621] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0013 / -0.0004 | -0.0008 | [-0.0029, +0.0013] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0050 / -0.0092 | -0.0071 | [-0.0135, -0.0025] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0000 / -0.0013 | -0.0006 | [-0.0025, +0.0010] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.4425 / -0.4211 | -0.4318 | [-0.6623, -0.2372] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1819 / -0.1223 | -0.1521 | [-0.2950, -0.0505] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.4647 / -0.5035 | -0.4841 | [-0.7568, -0.2468] | 0.3535 | BETTER |
| ade6_m|all | -0.3109 / -0.2961 | -0.3035 | [-0.4645, -0.1672] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.4425 / -0.4211 | -0.4318 | [-0.6623, -0.2372] | 0.5955 | INSIDE-SPREAD |

## IPix RLFT minus IPox RLFT

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3144 / +0.3501 | +0.3323 | [+0.1400, +0.5443] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +0.9897 / +1.1449 | +1.0673 | [+0.5622, +1.6401] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0394 / -0.0434 | -0.0414 | [-0.0683, -0.0144] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4899 / +0.4376 | +0.4637 | [+0.2417, +0.7059] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0217 / +0.0301 | +0.0259 | [+0.0143, +0.0387] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.8202 / +1.7919 | +1.8061 | [+0.5430, +3.1643] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0000 / +0.0196 | +0.0098 | [-0.0038, +0.0258] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0013 / +0.0129 | +0.0058 | [-0.0065, +0.0217] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0150 | +0.0075 | [-0.0031, +0.0212] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.3381 / +1.5657 | +1.4519 | [+0.7465, +2.2618] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.5840 / +0.7027 | +0.6434 | [+0.2608, +1.1009] | 0.4517 | WORSE |
| ade6_m|bulk | +1.3651 / +1.5784 | +1.4718 | [+0.7608, +2.2859] | 0.3535 | WORSE |
| ade6_m|all | +0.9402 / +1.1020 | +1.0211 | [+0.5253, +1.5893] | 0.4002 | WORSE |
| ade_tac_m|all | +1.3381 / +1.5657 | +1.4519 | [+0.7465, +2.2618] | 0.5955 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| IPcx_RLFT_s0 | 3.2480 | 3.5575 | 1.8118 | 1.1964 | 0.7061 |
| IPcx_RLFT_s1 | 3.5248 | 3.8400 | 1.9608 | 1.0986 | 0.6415 |
| IPix_RLFT_s0 | 3.6822 | 3.8117 | 2.0010 | 1.2183 | 0.7090 |
| IPix_RLFT_s1 | 3.8303 | 3.9388 | 2.1048 | 1.1161 | 0.6457 |
| IPksx_RLFT_s0 | 7.0543 | 7.1643 | 3.1145 | 1.2809 | 0.7395 |
| IPksx_RLFT_s1 | 7.3978 | 7.5646 | 3.2976 | 1.2444 | 0.6953 |
| IPkx_RLFT_s0 | 3.3714 | 3.6298 | 1.8844 | 1.2090 | 0.7090 |
| IPkx_RLFT_s1 | 3.5342 | 3.8165 | 1.9974 | 1.1135 | 0.6444 |
| IPlx_RLFT_s0 | 3.3474 | 3.5731 | 1.8849 | 1.2052 | 0.7095 |
| IPlx_RLFT_s1 | 3.2902 | 3.5410 | 1.8891 | 1.1092 | 0.6365 |
| IPox_RLFT_s0 | 2.7420 | 3.2277 | 1.6865 | 1.2578 | 0.7090 |
| IPox_RLFT_s1 | 2.7283 | 3.2361 | 1.7547 | 1.1595 | 0.6307 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
