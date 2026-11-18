# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IPlx_RLAm"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045083, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IPlx RLAm minus IPox RLAm

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2052 / +0.1289 | +0.1671 | [-0.0035, +0.3537] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.6511 / +0.5980 | +0.6246 | [+0.2248, +1.1065] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0471 / -0.0505 | -0.0488 | [-0.0708, -0.0273] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5335 / +0.1791 | +0.3563 | [+0.0282, +0.6954] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0198 / +0.0181 | +0.0190 | [+0.0103, +0.0294] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.8671 / -0.2577 | +0.3047 | [-1.1500, +1.6189] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0017 / +0.0121 | +0.0052 | [-0.0060, +0.0192] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0033 / +0.0088 | +0.0027 | [-0.0090, +0.0190] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0021 / +0.0117 | +0.0048 | [-0.0058, +0.0194] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.8665 / +0.7926 | +0.8295 | [+0.2671, +1.5039] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3415 / +0.2892 | +0.3153 | [+0.0334, +0.6775] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.9277 / +0.8832 | +0.9054 | [+0.3473, +1.5722] | 0.3535 | WORSE |
| ade6_m|all | +0.6088 / +0.5601 | +0.5844 | [+0.1891, +1.0577] | 0.4002 | WORSE |
| ade_tac_m|all | +0.8665 / +0.7926 | +0.8295 | [+0.2671, +1.5039] | 0.5955 | WORSE |

## IPkx RLAm minus IPox RLAm

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2243 / +0.2312 | +0.2277 | [+0.0672, +0.4105] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6973 / +0.8350 | +0.7661 | [+0.3422, +1.2402] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0391 / -0.0484 | -0.0437 | [-0.0673, -0.0227] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2060 / +0.1619 | +0.1839 | [+0.0323, +0.3481] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0100 / +0.0175 | +0.0137 | [+0.0062, +0.0221] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.6989 / +0.2332 | +0.4661 | [-0.5318, +1.4550] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0029 / +0.0242 | +0.0135 | [+0.0002, +0.0307] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0038 / +0.0075 | +0.0019 | [-0.0098, +0.0169] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0013 / +0.0192 | +0.0102 | [-0.0013, +0.0258] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.9379 / +1.1317 | +1.0348 | [+0.4484, +1.6917] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4272 / +0.5921 | +0.5096 | [+0.1653, +0.9383] | 0.4517 | WORSE |
| ade6_m|bulk | +0.9332 / +1.0396 | +0.9864 | [+0.4609, +1.5930] | 0.3535 | WORSE |
| ade6_m|all | +0.6579 / +0.7962 | +0.7270 | [+0.3138, +1.1896] | 0.4002 | WORSE |
| ade_tac_m|all | +0.9379 / +1.1317 | +1.0348 | [+0.4484, +1.6917] | 0.5955 | WORSE |

## IPkx RLAm minus IPcx RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0943 / +0.0424 | +0.0684 | [+0.0198, +0.1131] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.1466 / -0.0105 | +0.0681 | [-0.0607, +0.1934] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0182 / +0.0160 | +0.0171 | [+0.0065, +0.0288] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.2575 / -0.1334 | -0.1954 | [-0.4105, -0.0229] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0099 / -0.0061 | -0.0080 | [-0.0155, -0.0008] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.5196 / -0.0861 | +0.2168 | [-0.4349, +0.9363] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0100 / +0.0017 | +0.0058 | [-0.0025, +0.0140] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0050 / +0.0029 | +0.0040 | [-0.0006, +0.0090] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0067 / +0.0017 | +0.0042 | [-0.0008, +0.0092] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2261 / +0.0021 | +0.1141 | [-0.0692, +0.2877] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0922 / +0.0045 | +0.0483 | [-0.0354, +0.1212] | 0.4517 | OPEN |
| ade6_m|bulk | +0.2327 / -0.0082 | +0.1122 | [-0.0946, +0.3204] | 0.3535 | OPEN |
| ade6_m|all | +0.1562 / -0.0013 | +0.0775 | [-0.0513, +0.1988] | 0.4002 | OPEN |
| ade_tac_m|all | +0.2261 / +0.0021 | +0.1141 | [-0.0692, +0.2877] | 0.5955 | OPEN |

## IPkx RLAm minus IPlx RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0191 / +0.1023 | +0.0607 | [-0.0131, +0.1638] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0462 / +0.2370 | +0.1416 | [-0.0261, +0.3725] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0080 / +0.0021 | +0.0050 | [-0.0121, +0.0235] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2302 / +0.0323 | -0.0990 | [-0.3702, +0.1369] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0090 / +0.0017 | -0.0036 | [-0.0139, +0.0067] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2617 / +0.0963 | -0.0827 | [-0.7305, +0.6305] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0046 / +0.0121 | +0.0083 | [+0.0006, +0.0177] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0004 / -0.0013 | -0.0008 | [-0.0083, +0.0065] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0033 / +0.0075 | +0.0054 | [+0.0000, +0.0121] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.0714 / +0.3391 | +0.2053 | [-0.0237, +0.5292] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0857 / +0.3029 | +0.1943 | [+0.0362, +0.4794] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.0055 / +0.1564 | +0.0809 | [-0.1498, +0.3561] | 0.3535 | OPEN |
| ade6_m|all | +0.0491 / +0.2361 | +0.1426 | [-0.0184, +0.3685] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0714 / +0.3391 | +0.2053 | [-0.0237, +0.5292] | 0.5955 | OPEN |

## IPcx RLAm minus IPox RLAm

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1300 / +0.1887 | +0.1594 | [+0.0010, +0.3464] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.5507 / +0.8455 | +0.6981 | [+0.2692, +1.2179] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0572 / -0.0644 | -0.0608 | [-0.0881, -0.0377] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4746 / +0.3601 | +0.4174 | [+0.1688, +0.6909] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0200 / +0.0227 | +0.0213 | [+0.0126, +0.0321] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.2284 / +0.5718 | +0.4001 | [-0.6670, +1.5735] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0071 / +0.0225 | +0.0077 | [-0.0127, +0.0302] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0088 / +0.0046 | -0.0021 | [-0.0142, +0.0144] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0054 / +0.0175 | +0.0060 | [-0.0096, +0.0252] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.7118 / +1.1296 | +0.9207 | [+0.3219, +1.6435] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3350 / +0.5877 | +0.4613 | [+0.1119, +0.9008] | 0.4517 | WORSE |
| ade6_m|bulk | +0.7005 / +1.0478 | +0.8741 | [+0.3522, +1.5092] | 0.3535 | WORSE |
| ade6_m|all | +0.5017 / +0.7975 | +0.6496 | [+0.2286, +1.1580] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7118 / +1.1296 | +0.9207 | [+0.3219, +1.6435] | 0.5955 | WORSE |

## IPksx RLAm minus IPkx RLAm

Seeds read [0, 1] of [0, 1]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +1.1835 / +1.2794 | +1.2315 | [+0.9726, +1.5022] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +3.6984 / +3.8473 | +3.7728 | [+3.0844, +4.4823] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0548 / +0.1067 | +0.0808 | [-0.0213, +0.1996] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +3.9960 / +2.9480 | +3.4720 | [+2.2084, +4.6861] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.2164 / +0.2146 | +0.2155 | [+0.1681, +0.2639] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +15.6875 / +13.8034 | +14.7454 | [+10.7934, +18.5794] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0459 / +0.0625 | +0.0542 | [+0.0321, +0.0767] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0004 / +0.0371 | +0.0183 | [-0.0202, +0.0578] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0233 / +0.0488 | +0.0361 | [+0.0138, +0.0599] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +5.0489 / +5.3150 | +5.1820 | [+4.2234, +6.1699] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +3.4892 / +3.6352 | +3.5622 | [+2.8082, +4.3440] | 0.4517 | WORSE |
| ade6_m|bulk | +3.7773 / +4.0108 | +3.8941 | [+2.8712, +4.9859] | 0.3535 | WORSE |
| ade6_m|all | +3.6206 / +3.8065 | +3.7136 | [+3.0303, +4.4145] | 0.4002 | WORSE |
| ade_tac_m|all | +5.0489 / +5.3150 | +5.1820 | [+4.2234, +6.1699] | 0.5955 | WORSE |

## IPkx RLAm minus IPix RLAm

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0135 / +0.0080 | +0.0108 | [-0.0015, +0.0230] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0239 / -0.0001 | +0.0119 | [-0.0214, +0.0437] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0004 / +0.0025 | +0.0014 | [-0.0021, +0.0056] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.0290 / -0.0079 | -0.0185 | [-0.0706, +0.0195] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0014 / -0.0012 | -0.0013 | [-0.0034, +0.0007] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0513 / -0.1931 | -0.0709 | [-0.3384, +0.1261] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0008 / +0.0000 | +0.0004 | [-0.0015, +0.0023] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0004 / +0.0013 | +0.0004 | [-0.0021, +0.0033] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0008 / -0.0004 | +0.0002 | [-0.0015, +0.0017] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0354 / +0.0050 | +0.0202 | [-0.0255, +0.0651] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0215 / -0.0131 | +0.0042 | [-0.0254, +0.0392] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0280 / +0.0227 | +0.0254 | [-0.0223, +0.0778] | 0.3535 | OPEN |
| ade6_m|all | +0.0245 / +0.0032 | +0.0139 | [-0.0182, +0.0454] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0354 / +0.0050 | +0.0202 | [-0.0255, +0.0651] | 0.5955 | OPEN |

## IPix RLAm minus IPox RLAm

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2108 / +0.2232 | +0.2170 | [+0.0582, +0.3992] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6734 / +0.8350 | +0.7542 | [+0.3370, +1.2323] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0394 / -0.0509 | -0.0452 | [-0.0696, -0.0229] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.2350 / +0.1698 | +0.2024 | [+0.0318, +0.3750] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0114 / +0.0187 | +0.0150 | [+0.0071, +0.0241] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.6476 / +0.4263 | +0.5370 | [-0.4403, +1.5042] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0021 / +0.0242 | +0.0131 | [-0.0010, +0.0311] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0033 / +0.0063 | +0.0015 | [-0.0094, +0.0156] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0004 / +0.0196 | +0.0100 | [-0.0019, +0.0263] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.9025 / +1.1267 | +1.0146 | [+0.4373, +1.6879] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4056 / +0.6052 | +0.5054 | [+0.1638, +0.9393] | 0.4517 | WORSE |
| ade6_m|bulk | +0.9052 / +1.0168 | +0.9610 | [+0.4407, +1.5587] | 0.3535 | WORSE |
| ade6_m|all | +0.6334 / +0.7929 | +0.7132 | [+0.3080, +1.1868] | 0.4002 | WORSE |
| ade_tac_m|all | +0.9025 / +1.1267 | +1.0146 | [+0.4373, +1.6879] | 0.5955 | WORSE |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| IPcx_RLAm_s0 | 3.2525 | 3.5401 | 1.8057 | 1.2002 | 0.7203 |
| IPcx_RLAm_s1 | 3.5459 | 3.8327 | 2.0100 | 1.1193 | 0.6511 |
| IPix_RLAm_s0 | 3.3842 | 3.6108 | 1.8865 | 1.2180 | 0.7261 |
| IPix_RLAm_s1 | 3.5414 | 3.8503 | 2.0444 | 1.1328 | 0.6532 |
| IPksx_RLAm_s0 | 7.0294 | 7.1216 | 3.0835 | 1.2731 | 0.7503 |
| IPksx_RLAm_s1 | 7.3511 | 7.4724 | 3.3318 | 1.2420 | 0.7015 |
| IPkx_RLAm_s0 | 3.4087 | 3.6323 | 1.9000 | 1.2183 | 0.7270 |
| IPkx_RLAm_s1 | 3.5446 | 3.8371 | 2.0524 | 1.1353 | 0.6528 |
| IPlx_RLAm_s0 | 3.3596 | 3.5466 | 1.8809 | 1.2103 | 0.7236 |
| IPlx_RLAm_s1 | 3.3085 | 3.5342 | 1.9501 | 1.1332 | 0.6453 |
| IPox_RLAm_s0 | 2.7508 | 3.2052 | 1.6757 | 1.2574 | 0.7257 |
| IPox_RLAm_s1 | 2.7484 | 3.2450 | 1.8212 | 1.1837 | 0.6336 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
