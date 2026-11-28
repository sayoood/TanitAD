# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 18}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IPlq_LS"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IPlq LS minus IPoq LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1698 / +0.2172 | +0.1935 | [+0.0294, +0.3922] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6760 / +0.6747 | +0.6753 | [+0.2569, +1.1802] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0427 / -0.0452 | -0.0440 | [-0.0655, -0.0246] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4096 / +0.3762 | +0.3929 | [+0.1324, +0.6887] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0269 / +0.0331 | +0.0300 | [+0.0166, +0.0464] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.9006 / +1.7354 | +1.3180 | [-0.2247, +3.0868] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0096 / +0.0038 | +0.0067 | [-0.0027, +0.0171] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0058 / -0.0025 | +0.0017 | [-0.0102, +0.0154] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0129 / +0.0038 | +0.0083 | [-0.0019, +0.0204] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.9011 / +0.9094 | +0.9053 | [+0.3196, +1.6166] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3768 / +0.3535 | +0.3652 | [+0.0722, +0.7482] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.9371 / +0.9755 | +0.9563 | [+0.3559, +1.6438] | 0.3535 | WORSE |
| ade6_m|all | +0.6323 / +0.6371 | +0.6347 | [+0.2254, +1.1334] | 0.4002 | WORSE |
| ade_tac_m|all | +0.9011 / +0.9094 | +0.9053 | [+0.3196, +1.6166] | 0.5955 | WORSE |

## IPcq LS minus IPoq LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1377 / +0.3327 | +0.2352 | [+0.0274, +0.4717] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6888 / +1.0848 | +0.8868 | [+0.3953, +1.4736] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0642 / -0.0447 | -0.0544 | [-0.0850, -0.0142] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.1470 / +0.9386 | +0.5428 | [-0.0082, +1.1810] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0129 / +0.0540 | +0.0334 | [+0.0093, +0.0673] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.7801 / +3.8625 | +1.5412 | [-1.7459, +4.9022] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0096 / +0.0029 | +0.0063 | [-0.0060, +0.0184] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0088 / -0.0088 | +0.0000 | [-0.0177, +0.0156] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0171 / +0.0025 | +0.0098 | [-0.0033, +0.0233] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.8929 / +1.4862 | +1.1895 | [+0.4869, +2.0246] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3505 / +0.7882 | +0.5694 | [+0.1418, +1.1050] | 0.4517 | WORSE |
| ade6_m|bulk | +0.9577 / +1.3443 | +1.1510 | [+0.5000, +1.9370] | 0.3535 | WORSE |
| ade6_m|all | +0.6274 / +1.0418 | +0.8346 | [+0.3421, +1.4196] | 0.4002 | WORSE |
| ade_tac_m|all | +0.8929 / +1.4862 | +1.1895 | [+0.4869, +2.0246] | 0.5955 | WORSE |

## IPcq LS minus IPlq LS

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0321 / +0.1155 | +0.0417 | [-0.0732, +0.1756] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0128 / +0.4101 | +0.2115 | [-0.0704, +0.5755] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0214 / +0.0005 | -0.0105 | [-0.0379, +0.0258] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2115 / +0.5696 | +0.1790 | [-0.3384, +0.7834] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | -0.0128 / +0.0253 | +0.0063 | [-0.0193, +0.0377] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.4084 / +2.0483 | +0.3199 | [-2.0763, +2.9281] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0000 / -0.0008 | -0.0004 | [-0.0079, +0.0071] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0029 / -0.0063 | -0.0017 | [-0.0138, +0.0083] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0042 / -0.0013 | +0.0015 | [-0.0058, +0.0077] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0082 / +0.5768 | +0.2843 | [-0.1271, +0.8091] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0263 / +0.4347 | +0.2042 | [-0.0697, +0.6567] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0206 / +0.3688 | +0.1947 | [-0.1422, +0.5926] | 0.3535 | OPEN |
| ade6_m|all | -0.0049 / +0.4046 | +0.1999 | [-0.0886, +0.5671] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0082 / +0.5768 | +0.2843 | [-0.1271, +0.8091] | 0.5955 | OPEN |

## IPcqq LS minus IPcq LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3527 / +0.2347 | +0.2937 | [+0.1651, +0.4475] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.0170 / +0.6969 | +0.8570 | [+0.5232, +1.2506] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0236 / +0.0054 | +0.0145 | [-0.0183, +0.0588] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.0138 / +1.1181 | +1.0660 | [+0.6846, +1.4936] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0635 / +0.0575 | +0.0605 | [+0.0407, +0.0835] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.4182 / +4.8860 | +4.6521 | [+3.0938, +6.3845] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0179 / +0.0100 | +0.0140 | [+0.0035, +0.0244] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0108 / +0.0163 | +0.0135 | [+0.0015, +0.0252] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0121 / +0.0063 | +0.0092 | [+0.0015, +0.0179] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +1.4135 / +0.9656 | +1.1896 | [+0.7179, +1.7427] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.7731 / +0.3914 | +0.5823 | [+0.3001, +0.9736] | 0.4517 | WORSE |
| ade6_m|bulk | +1.2559 / +1.0227 | +1.1393 | [+0.6159, +1.7171] | 0.3535 | WORSE |
| ade6_m|all | +0.9933 / +0.6793 | +0.8363 | [+0.5057, +1.2246] | 0.4002 | WORSE |
| ade_tac_m|all | +1.4135 / +0.9656 | +1.1896 | [+0.7179, +1.7427] | 0.5955 | WORSE |

## IPcqq LS minus IPlq LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.3206 / +0.3502 | +0.3354 | [+0.1872, +0.4822] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.0298 / +1.1071 | +1.0684 | [+0.7126, +1.4633] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0022 / +0.0059 | +0.0040 | [-0.0397, +0.0527] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.8482 / +1.7230 | +1.2856 | [+0.6188, +2.0964] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0524 / +0.0831 | +0.0678 | [+0.0395, +0.1018] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.2937 / +7.0145 | +5.1541 | [+2.3109, +8.3892] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0179 / +0.0092 | +0.0135 | [+0.0010, +0.0254] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0138 / +0.0100 | +0.0119 | [-0.0008, +0.0233] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0163 / +0.0050 | +0.0106 | [-0.0006, +0.0209] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.4053 / +1.5424 | +1.4739 | [+0.9708, +2.0325] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.7468 / +0.8261 | +0.7865 | [+0.4607, +1.1422] | 0.4517 | WORSE |
| ade6_m|bulk | +1.2765 / +1.3916 | +1.3341 | [+0.7306, +2.0084] | 0.3535 | WORSE |
| ade6_m|all | +0.9884 / +1.0840 | +1.0362 | [+0.6804, +1.4276] | 0.4002 | WORSE |
| ade_tac_m|all | +1.4053 / +1.5424 | +1.4739 | [+0.9708, +2.0325] | 0.5955 | WORSE |

## IPcqq LS minus IPoq LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4904 / +0.5674 | +0.5289 | [+0.2913, +0.7768] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.7058 / +1.7818 | +1.7438 | [+1.1588, +2.3782] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0406 / -0.0393 | -0.0399 | [-0.0874, +0.0096] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.2227 / +2.0730 | +1.6479 | [+0.9134, +2.5083] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0750 / +0.1105 | +0.0927 | [+0.0580, +0.1329] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.1440 / +8.9273 | +6.5356 | [+2.6727, +10.7768] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0275 / +0.0129 | +0.0202 | [+0.0021, +0.0379] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0196 / +0.0075 | +0.0135 | [-0.0058, +0.0317] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0292 / +0.0088 | +0.0190 | [+0.0010, +0.0381] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +2.3064 / +2.4518 | +2.3791 | [+1.5526, +3.2815] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.1236 / +1.1796 | +1.1516 | [+0.6876, +1.6665] | 0.4517 | WORSE |
| ade6_m|bulk | +2.2136 / +2.3670 | +2.2903 | [+1.3629, +3.3175] | 0.3535 | WORSE |
| ade6_m|all | +1.6207 / +1.7211 | +1.6709 | [+1.0945, +2.2997] | 0.4002 | WORSE |
| ade_tac_m|all | +2.3064 / +2.4518 | +2.3791 | [+1.5526, +3.2815] | 0.5955 | WORSE |

## IPcq0q LS minus IPcq LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LATERAL COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +2.0067 / +1.9853 | +1.9960 | [+1.6956, +2.3144] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +6.3935 / +6.2202 | +6.3068 | [+5.4050, +7.2268] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.2532 / +0.2796 | +0.2664 | [+0.1392, +0.4050] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +9.4020 / +13.6277 | +11.5149 | [+8.5138, +14.9652] | 4.5570 (1.9414) | WORSE |
| LATERAL:curvature_mae_1pm | +0.5423 / +0.5665 | +0.5544 | [+0.4762, +0.6366] | 0.1528 (0.0760) | WORSE |
| LATERAL:yaw_rate_mae_degps | +40.0412 / +45.4815 | +42.7613 | [+36.1997, +49.9257] | 15.8479 (7.7515) | WORSE |
| TACTICAL:lon_err_rate | +0.0850 / +0.0613 | +0.0732 | [+0.0475, +0.0985] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0575 / +0.1042 | +0.0809 | [+0.0381, +0.1222] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0738 / +0.0704 | +0.0721 | [+0.0556, +0.0909] | 0.0492 (0.0263) | WORSE |
| TACTICAL:ade_tac_m | +8.8323 / +8.6172 | +8.7248 | [+7.4762, +10.0080] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +6.1712 / +5.6935 | +5.9324 | [+4.8883, +7.0270] | 0.4517 | WORSE |
| ade6_m|bulk | +6.5638 / +6.7724 | +6.6681 | [+5.3645, +8.0594] | 0.3535 | WORSE |
| ade6_m|all | +6.3503 / +6.1855 | +6.2679 | [+5.3809, +7.1752] | 0.4002 | WORSE |
| ade_tac_m|all | +8.8323 / +8.6172 | +8.7248 | [+7.4762, +10.0080] | 0.5955 | WORSE |

## IPcqidq LS minus IPcq LS

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
| IPcq0q_LS_s0 | 9.9870 | 10.0359 | 4.0922 | 1.4518 | 0.7616 |
| IPcq0q_LS_s1 | 9.9185 | 9.6521 | 3.8990 | 1.4716 | 0.7774 |
| IPcq_LS_s0 | 3.6368 | 3.8647 | 2.0855 | 1.1986 | 0.6878 |
| IPcq_LS_s1 | 3.7330 | 3.9586 | 1.9136 | 1.1920 | 0.7070 |
| IPcqidq_LS_s0 | 3.6368 | 3.8647 | 2.0855 | 1.1986 | 0.6878 |
| IPcqidq_LS_s1 | 3.7330 | 3.9586 | 1.9136 | 1.1920 | 0.7070 |
| IPcqq_LS_s0 | 4.6300 | 4.6378 | 2.4382 | 1.2222 | 0.6999 |
| IPcqq_LS_s1 | 4.4124 | 4.3500 | 2.1483 | 1.1974 | 0.7132 |
| IPlq_LS_s0 | 3.6417 | 3.8910 | 2.1177 | 1.2201 | 0.6836 |
| IPlq_LS_s1 | 3.3284 | 3.5239 | 1.7981 | 1.1915 | 0.7082 |
| IPoq_LS_s0 | 3.0093 | 3.5142 | 1.9479 | 1.2628 | 0.6707 |
| IPoq_LS_s1 | 2.6913 | 3.1704 | 1.5809 | 1.2366 | 0.7045 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
