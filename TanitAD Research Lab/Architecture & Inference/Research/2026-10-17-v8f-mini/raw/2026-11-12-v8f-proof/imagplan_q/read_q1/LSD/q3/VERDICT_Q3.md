# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IPlq_LSD"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IPlq LSD minus IPoq LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1790 / +0.1163 | +0.1476 | [-0.0097, +0.3287] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.6727 / +0.6304 | +0.6515 | [+0.2465, +1.1446] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0443 / -0.0685 | -0.0564 | [-0.0905, -0.0299] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.3055 / +0.5121 | +0.4088 | [+0.1062, +0.7250] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0271 / +0.0174 | +0.0223 | [+0.0112, +0.0363] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.0147 / +0.8989 | +0.4568 | [-1.0248, +2.1236] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0025 / -0.0050 | -0.0038 | [-0.0106, +0.0033] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0138 / -0.0029 | +0.0054 | [-0.0098, +0.0223] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0063 / -0.0025 | +0.0019 | [-0.0065, +0.0108] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.9034 / +0.8211 | +0.8623 | [+0.2847, +1.5540] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3776 / +0.3050 | +0.3413 | [+0.0451, +0.7136] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.9389 / +0.9018 | +0.9203 | [+0.3516, +1.5881] | 0.3535 | WORSE |
| ade6_m|all | +0.6336 / +0.5771 | +0.6053 | [+0.2015, +1.0906] | 0.4002 | WORSE |
| ade_tac_m|all | +0.9034 / +0.8211 | +0.8623 | [+0.2847, +1.5540] | 0.5955 | WORSE |

## IPcq LSD minus IPoq LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2111 / +0.1019 | +0.1565 | [-0.0060, +0.3512] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.8017 / +0.6529 | +0.7273 | [+0.3209, +1.2132] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0330 / -0.0656 | -0.0493 | [-0.0830, -0.0156] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.7475 / +0.3841 | +0.5658 | [+0.2533, +0.9625] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0213 / +0.0182 | +0.0197 | [+0.0109, +0.0312] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.3024 / +1.4280 | +0.5628 | [-1.2927, +2.3658] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0029 / -0.0033 | -0.0002 | [-0.0075, +0.0077] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0208 / +0.0017 | +0.0113 | [-0.0035, +0.0286] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0108 / +0.0000 | +0.0054 | [-0.0031, +0.0150] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.0836 / +0.8684 | +0.9760 | [+0.3986, +1.6481] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.5317 / +0.3401 | +0.4359 | [+0.1062, +0.8473] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +1.0305 / +0.9270 | +0.9788 | [+0.4292, +1.6302] | 0.3535 | WORSE |
| ade6_m|all | +0.7592 / +0.6077 | +0.6835 | [+0.2793, +1.1547] | 0.4002 | WORSE |
| ade_tac_m|all | +1.0836 / +0.8684 | +0.9760 | [+0.3986, +1.6481] | 0.5955 | WORSE |

## IPcq LSD minus IPlq LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0320 / -0.0144 | +0.0088 | [-0.0573, +0.0817] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.1290 / +0.0225 | +0.0758 | [-0.0760, +0.2415] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0112 / +0.0029 | +0.0071 | [-0.0182, +0.0352] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3378 / -0.1466 | +0.0956 | [-0.2535, +0.5006] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0054 / +0.0004 | -0.0025 | [-0.0143, +0.0064] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0970 / +0.3643 | +0.2307 | [-0.5867, +1.0319] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0054 / +0.0017 | +0.0035 | [-0.0025, +0.0098] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0071 / +0.0046 | +0.0058 | [-0.0017, +0.0140] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0046 / +0.0025 | +0.0035 | [-0.0015, +0.0083] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.1802 / +0.0473 | +0.1138 | [-0.0970, +0.3413] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1541 / +0.0351 | +0.0946 | [-0.0124, +0.2569] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0917 / +0.0253 | +0.0585 | [-0.1871, +0.3360] | 0.3535 | OPEN |
| ade6_m|all | +0.1256 / +0.0306 | +0.0781 | [-0.0706, +0.2387] | 0.4002 | OPEN |
| ade_tac_m|all | +0.1802 / +0.0473 | +0.1138 | [-0.0970, +0.3413] | 0.5955 | OPEN |

## IPcqq LSD minus IPcq LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2234 / +0.2666 | +0.2450 | [+0.1437, +0.3553] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.7667 / +0.8210 | +0.7939 | [+0.5325, +1.0874] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0308 / -0.0122 | -0.0215 | [-0.0505, +0.0148] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.6679 / +0.6509 | +0.6594 | [+0.3818, +0.9536] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0553 / +0.0313 | +0.0433 | [+0.0239, +0.0680] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.1369 / +2.6572 | +2.8971 | [+1.5072, +4.4100] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0021 / +0.0100 | +0.0040 | [-0.0061, +0.0133] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0113 / +0.0017 | +0.0065 | [-0.0065, +0.0186] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0008 / +0.0017 | +0.0004 | [-0.0052, +0.0054] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.0361 / +1.1195 | +1.0778 | [+0.7097, +1.4891] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3809 / +0.5449 | +0.4629 | [+0.2439, +0.7898] | 0.4517 | WORSE |
| ade6_m|bulk | +1.1451 / +1.0767 | +1.1109 | [+0.6642, +1.5744] | 0.3535 | WORSE |
| ade6_m|all | +0.7294 / +0.7874 | +0.7584 | [+0.5003, +1.0471] | 0.4002 | WORSE |
| ade_tac_m|all | +1.0361 / +1.1195 | +1.0778 | [+0.7097, +1.4891] | 0.5955 | WORSE |

## IPcqq LSD minus IPlq LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2554 / +0.2523 | +0.2538 | [+0.1329, +0.3805] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.8957 / +0.8436 | +0.8696 | [+0.5621, +1.2064] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0195 / -0.0092 | -0.0144 | [-0.0496, +0.0268] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.9942 / +0.5637 | +0.7790 | [+0.3915, +1.2016] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0517 / +0.0320 | +0.0419 | [+0.0239, +0.0659] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.2371 / +2.6257 | +2.9314 | [+1.4322, +4.5569] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0033 / +0.0117 | +0.0075 | [-0.0013, +0.0169] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0183 / +0.0063 | +0.0123 | [-0.0010, +0.0256] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0038 / +0.0042 | +0.0040 | [-0.0010, +0.0090] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.2163 / +1.1668 | +1.1915 | [+0.7593, +1.6599] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.5350 / +0.5800 | +0.5575 | [+0.2843, +0.8744] | 0.4517 | WORSE |
| ade6_m|bulk | +1.2367 / +1.1019 | +1.1693 | [+0.6181, +1.7771] | 0.3535 | WORSE |
| ade6_m|all | +0.8550 / +0.8180 | +0.8365 | [+0.5326, +1.1634] | 0.4002 | WORSE |
| ade_tac_m|all | +1.2163 / +1.1668 | +1.1915 | [+0.7593, +1.6599] | 0.5955 | WORSE |

## IPcqq LSD minus IPoq LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4344 / +0.3685 | +0.4015 | [+0.1825, +0.6312] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.5684 / +1.4740 | +1.5212 | [+0.9569, +2.1160] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0638 / -0.0777 | -0.0708 | [-0.1183, -0.0226] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.2983 / +1.0021 | +1.1502 | [+0.7398, +1.6259] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0768 / +0.0483 | +0.0625 | [+0.0374, +0.0941] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.3008 / +3.8968 | +3.5988 | [+1.4436, +5.8773] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0008 / +0.0067 | +0.0038 | [-0.0069, +0.0129] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0321 / +0.0033 | +0.0177 | [-0.0063, +0.0421] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0100 / +0.0017 | +0.0058 | [-0.0029, +0.0158] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +2.1197 / +1.9879 | +2.0538 | [+1.2701, +2.8859] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.9126 / +0.8850 | +0.8988 | [+0.4496, +1.3817] | 0.4517 | WORSE |
| ade6_m|bulk | +2.1756 / +2.0037 | +2.0896 | [+1.2229, +3.0486] | 0.3535 | WORSE |
| ade6_m|all | +1.4886 / +1.3951 | +1.4418 | [+0.8931, +2.0252] | 0.4002 | WORSE |
| ade_tac_m|all | +2.1197 / +1.9879 | +2.0538 | [+1.2701, +2.8859] | 0.5955 | WORSE |

## IPcq0q LSD minus IPcq LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +2.3166 / +1.8861 | +2.1013 | [+1.6819, +2.5315] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +7.3375 / +6.1983 | +6.7679 | [+5.5916, +7.9387] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.2740 / +0.1220 | +0.1980 | [+0.0437, +0.3604] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +11.4517 / +11.8926 | +11.6721 | [+9.9571, +13.4583] | 4.5570 (1.9414) | WORSE |
| LATERAL:curvature_mae_1pm | +0.6103 / +0.4268 | +0.5186 | [+0.3880, +0.6632] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +39.6249 / +33.3572 | +36.4910 | [+29.6597, +43.4864] | 15.8479 (7.7515) | WORSE |
| TACTICAL:lon_err_rate | +0.0567 / +0.0713 | +0.0640 | [+0.0440, +0.0842] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.1442 / +0.0775 | +0.1109 | [+0.0575, +0.1657] | 0.1042 (0.0534) | WORSE |
| TACTICAL:joint_err_rate | +0.0696 / +0.0621 | +0.0659 | [+0.0473, +0.0856] | 0.0492 (0.0263) | WORSE |
| TACTICAL:ade_tac_m | +10.1399 / +8.4474 | +9.2937 | [+7.6030, +10.9626] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +6.4669 / +5.6430 | +6.0550 | [+5.0169, +7.0771] | 0.4517 | WORSE |
| ade6_m|bulk | +8.2864 / +6.6019 | +7.4441 | [+5.6787, +9.2586] | 0.3535 | WORSE |
| ade6_m|all | +7.2967 / +6.0803 | +6.6885 | [+5.4804, +7.8852] | 0.4002 | WORSE |
| ade_tac_m|all | +10.1399 / +8.4474 | +9.2937 | [+7.6030, +10.9626] | 0.5955 | WORSE |

## IPcqidq LSD minus IPcq LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3535 | OPEN |
| ade6_m|all | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| IPcq0q_LSD_s0 | 11.1265 | 10.5788 | 4.4612 | 1.4494 | 0.8283 |
| IPcq0q_LSD_s1 | 9.6971 | 9.4897 | 3.9937 | 1.4212 | 0.8103 |
| IPcq_LSD_s0 | 3.8299 | 4.1119 | 2.1447 | 1.1754 | 0.7586 |
| IPcq_LSD_s1 | 3.6168 | 3.8467 | 2.1076 | 1.2993 | 0.7482 |
| IPcqidq_LSD_s0 | 3.8299 | 4.1119 | 2.1447 | 1.1754 | 0.7586 |
| IPcqidq_LSD_s1 | 3.6168 | 3.8467 | 2.1076 | 1.2993 | 0.7482 |
| IPcqq_LSD_s0 | 4.5593 | 4.4928 | 2.3680 | 1.1446 | 0.7578 |
| IPcqq_LSD_s1 | 4.4042 | 4.3916 | 2.3743 | 1.2871 | 0.7499 |
| IPlq_LSD_s0 | 3.7043 | 3.9578 | 2.1126 | 1.1641 | 0.7541 |
| IPlq_LSD_s1 | 3.5862 | 3.8116 | 2.1220 | 1.2964 | 0.7457 |
| IPoq_LSD_s0 | 3.0707 | 3.5802 | 1.9336 | 1.2084 | 0.7478 |
| IPoq_LSD_s1 | 3.0090 | 3.5066 | 2.0057 | 1.3649 | 0.7482 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
