# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: PAUSED (PI item 25).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 30}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSMC"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSMC minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1932 / +0.3280 | +0.2606 | [+0.0904, +0.4280] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.8531 / +1.1146 | +0.9839 | [+0.6044, +1.3997] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0230 / +0.0475 | +0.0353 | [-0.0209, +0.0899] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.1078 / +2.2368 | +1.6723 | [+0.7906, +2.6144] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0440 / +0.0751 | +0.0596 | [+0.0335, +0.0913] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.4024 / +9.9457 | +6.6741 | [+1.7971, +11.5396] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0046 / -0.0088 | -0.0021 | [-0.0181, +0.0146] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0142 / +0.0167 | +0.0154 | [-0.0021, +0.0338] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0167 / +0.0046 | +0.0106 | [-0.0033, +0.0256] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.1228 / +1.4880 | +1.3054 | [+0.7721, +1.8703] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6712 / +0.8314 | +0.7513 | [+0.4212, +1.1435] | 0.4517 | WORSE |
| ade6_m|bulk | +0.9631 / +1.3904 | +1.1767 | [+0.6151, +1.8029] | 0.3535 | WORSE |
| ade6_m|all | +0.8043 / +1.0863 | +0.9453 | [+0.5571, +1.3590] | 0.4002 | WORSE |
| ade_tac_m|all | +1.1228 / +1.4880 | +1.3054 | [+0.7721, +1.8703] | 0.5955 | WORSE |

## LSMT minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0333 / +0.1153 | +0.0743 | [-0.0299, +0.1786] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3377 / +0.4948 | +0.4163 | [+0.1919, +0.6759] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0194 / +0.0044 | -0.0075 | [-0.0359, +0.0284] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.3876 / +0.9271 | +0.6574 | [+0.2764, +1.1458] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0074 / +0.0307 | +0.0191 | [+0.0052, +0.0400] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.3167 / +3.5942 | +1.6388 | [-1.1859, +4.3975] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0033 / -0.0058 | -0.0046 | [-0.0152, +0.0060] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0058 / -0.0025 | +0.0017 | [-0.0121, +0.0140] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0054 / -0.0004 | +0.0025 | [-0.0065, +0.0115] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.4189 / +0.6548 | +0.5368 | [+0.2233, +0.8887] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1447 / +0.3525 | +0.2486 | [+0.0526, +0.5040] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.4864 / +0.6148 | +0.5506 | [+0.2347, +0.9142] | 0.3535 | WORSE |
| ade6_m|all | +0.3005 / +0.4721 | +0.3863 | [+0.1597, +0.6437] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.4189 / +0.6548 | +0.5368 | [+0.2233, +0.8887] | 0.5955 | INSIDE-SPREAD |

## LSMG minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1979 / +0.3280 | +0.2629 | [+0.0927, +0.4285] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.8456 / +1.1146 | +0.9801 | [+0.5999, +1.3997] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0297 / +0.0475 | +0.0386 | [-0.0159, +0.0911] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.0913 / +2.2368 | +1.6641 | [+0.7673, +2.6138] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0421 / +0.0751 | +0.0586 | [+0.0315, +0.0913] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.3041 / +9.9457 | +6.6249 | [+1.6551, +11.5396] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0054 / -0.0088 | -0.0017 | [-0.0181, +0.0154] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0150 / +0.0167 | +0.0158 | [-0.0015, +0.0340] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0171 / +0.0046 | +0.0108 | [-0.0033, +0.0260] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.1209 / +1.4880 | +1.3045 | [+0.7722, +1.8703] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.6761 / +0.8314 | +0.7537 | [+0.4211, +1.1440] | 0.4517 | WORSE |
| ade6_m|bulk | +0.9535 / +1.3904 | +1.1719 | [+0.6096, +1.7994] | 0.3535 | WORSE |
| ade6_m|all | +0.8026 / +1.0863 | +0.9444 | [+0.5584, +1.3560] | 0.4002 | WORSE |
| ade_tac_m|all | +1.1209 / +1.4880 | +1.3045 | [+0.7722, +1.8703] | 0.5955 | WORSE |

## LSML minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0730 / +0.1527 | +0.1129 | [-0.0035, +0.2366] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.5049 / +0.5643 | +0.5346 | [+0.2883, +0.8201] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0099 / +0.0076 | -0.0012 | [-0.0338, +0.0322] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.6287 / +1.1857 | +0.9072 | [+0.4099, +1.4195] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0142 / +0.0413 | +0.0278 | [+0.0100, +0.0540] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1643 / +4.6225 | +2.3934 | [-1.0189, +5.6293] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0017 / -0.0108 | -0.0046 | [-0.0190, +0.0102] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0067 / +0.0013 | +0.0040 | [-0.0106, +0.0179] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0133 / -0.0013 | +0.0060 | [-0.0077, +0.0213] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.6454 / +0.7456 | +0.6955 | [+0.3557, +1.0903] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3445 / +0.3849 | +0.3647 | [+0.1520, +0.6028] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.5997 / +0.7219 | +0.6608 | [+0.3019, +1.0882] | 0.3535 | WORSE |
| ade6_m|all | +0.4609 / +0.5386 | +0.4997 | [+0.2563, +0.7812] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6454 / +0.7456 | +0.6955 | [+0.3557, +1.0903] | 0.5955 | WORSE |

## LSMH minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.4557 / +0.6775 | +0.5666 | [+0.3267, +0.8221] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.6876 / +2.0702 | +1.8789 | [+1.3463, +2.5068] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0454 / +0.1473 | +0.0963 | [-0.0065, +0.1943] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +2.3346 / +3.6632 | +2.9989 | [+1.9094, +4.1165] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0875 / +0.1220 | +0.1047 | [+0.0717, +0.1433] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +10.4779 / +17.7256 | +14.1017 | [+8.4456, +20.0097] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0150 / -0.0071 | +0.0040 | [-0.0160, +0.0259] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0317 / +0.0509 | +0.0413 | [+0.0167, +0.0673] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0308 / +0.0104 | +0.0206 | [+0.0033, +0.0408] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +2.2127 / +2.8243 | +2.5185 | [+1.7611, +3.4059] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.2574 / +1.5876 | +1.4225 | [+0.9689, +1.9565] | 0.4517 | WORSE |
| ade6_m|bulk | +2.0525 / +2.6427 | +2.3476 | [+1.5616, +3.2933] | 0.3535 | WORSE |
| ade6_m|all | +1.6200 / +2.0688 | +1.8444 | [+1.2992, +2.4836] | 0.4002 | WORSE |
| ade_tac_m|all | +2.2127 / +2.8243 | +2.5185 | [+1.7611, +3.4059] | 0.5955 | WORSE |

## LSDMC minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2292 / +0.0951 | +0.1622 | [-0.0008, +0.3285] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.9841 / +0.8548 | +0.9194 | [+0.5842, +1.2876] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0531 / -0.0826 | -0.0148 | [-0.1181, +0.0794] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +1.0157 / +1.0388 | +1.0272 | [+0.6671, +1.3958] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0238 / +0.0342 | +0.0290 | [+0.0163, +0.0443] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.9441 / +4.4004 | +3.1723 | [+0.5968, +5.5167] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0121 / -0.0163 | -0.0142 | [-0.0246, -0.0039] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0467 / +0.0017 | +0.0242 | [-0.0092, +0.0594] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0092 / -0.0063 | +0.0015 | [-0.0108, +0.0146] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.3075 / +1.0412 | +1.1744 | [+0.6817, +1.6923] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.7221 / +0.4681 | +0.5951 | [+0.2656, +0.9865] | 0.4517 | WORSE |
| ade6_m|bulk | +1.2263 / +1.1180 | +1.1722 | [+0.6716, +1.7192] | 0.3535 | WORSE |
| ade6_m|all | +0.9520 / +0.7644 | +0.8582 | [+0.5077, +1.2294] | 0.4002 | WORSE |
| ade_tac_m|all | +1.3075 / +1.0412 | +1.1744 | [+0.6817, +1.6923] | 0.5955 | WORSE |

## LSDMT minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0498 / -0.0350 | +0.0074 | [-0.0899, +0.1090] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.3685 / +0.3023 | +0.3354 | [+0.1595, +0.5384] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0156 / -0.0488 | -0.0166 | [-0.0640, +0.0284] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.5851 / +0.4475 | +0.5163 | [+0.3055, +0.7427] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0095 / +0.0072 | +0.0084 | [+0.0038, +0.0141] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.2561 / -0.3112 | -0.2836 | [-1.5082, +0.9005] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0017 / -0.0083 | -0.0050 | [-0.0133, +0.0031] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0296 / +0.0033 | +0.0165 | [-0.0027, +0.0394] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0092 / -0.0004 | +0.0044 | [-0.0035, +0.0129] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.4944 / +0.3621 | +0.4283 | [+0.1766, +0.7159] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2425 / +0.1427 | +0.1926 | [+0.0258, +0.3920] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.4860 / +0.4012 | +0.4436 | [+0.2043, +0.7225] | 0.3535 | WORSE |
| ade6_m|all | +0.3535 / +0.2605 | +0.3070 | [+0.1263, +0.5157] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.4944 / +0.3621 | +0.4283 | [+0.1766, +0.7159] | 0.5955 | INSIDE-SPREAD |

## LSDMG minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.2230 / +0.1072 | +0.1651 | [+0.0090, +0.3276] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.9637 / +0.8415 | +0.9026 | [+0.5694, +1.2707] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0488 / -0.0772 | -0.0142 | [-0.1118, +0.0750] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.9633 / +1.0292 | +0.9962 | [+0.6312, +1.3694] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0236 / +0.0342 | +0.0289 | [+0.0162, +0.0442] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.8549 / +4.4136 | +3.1342 | [+0.5742, +5.5430] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0096 / -0.0158 | -0.0127 | [-0.0231, -0.0019] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0454 / +0.0008 | +0.0231 | [-0.0100, +0.0579] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0113 / -0.0054 | +0.0029 | [-0.0100, +0.0163] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +1.2788 / +1.0299 | +1.1544 | [+0.6675, +1.6713] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.7006 / +0.4692 | +0.5849 | [+0.2679, +0.9651] | 0.4517 | WORSE |
| ade6_m|bulk | +1.2068 / +1.0970 | +1.1519 | [+0.6488, +1.6948] | 0.3535 | WORSE |
| ade6_m|all | +0.9314 / +0.7555 | +0.8435 | [+0.4938, +1.2140] | 0.4002 | WORSE |
| ade_tac_m|all | +1.2788 / +1.0299 | +1.1544 | [+0.6675, +1.6713] | 0.5955 | WORSE |

## LSDML minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1091 / -0.0389 | +0.0351 | [-0.1009, +0.1790] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.5701 / +0.4081 | +0.4891 | [+0.2506, +0.7685] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0172 / -0.0754 | -0.0291 | [-0.0965, +0.0364] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.8199 / +0.5166 | +0.6683 | [+0.3863, +1.0180] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0171 / +0.0222 | +0.0197 | [+0.0118, +0.0291] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -0.0023 / +0.7176 | +0.3577 | [-1.4046, +1.7895] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0008 / -0.0113 | -0.0060 | [-0.0167, +0.0050] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0467 / +0.0054 | +0.0261 | [-0.0010, +0.0575] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0121 / -0.0033 | +0.0044 | [-0.0063, +0.0169] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.7538 / +0.4682 | +0.6110 | [+0.2609, +1.0030] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.4056 / +0.1707 | +0.2882 | [+0.0608, +0.5834] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.7047 / +0.5499 | +0.6273 | [+0.2992, +1.0159] | 0.3535 | WORSE |
| ade6_m|all | +0.5420 / +0.3436 | +0.4428 | [+0.1928, +0.7272] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7538 / +0.4682 | +0.6110 | [+0.2609, +1.0030] | 0.5955 | WORSE |

## LSDMH minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.5274 / +0.4753 | +0.5013 | [+0.3166, +0.6941] | 0.2693 (0.1381) | WORSE |
| LONGITUDINAL:along_mae_m | +1.9219 / +1.8809 | +1.9014 | [+1.4503, +2.3808] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0975 / +0.0034 | +0.0505 | [-0.0448, +0.1397] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +2.7350 / +2.4383 | +2.5866 | [+2.0341, +3.1295] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0875 / +0.0780 | +0.0827 | [+0.0614, +0.1075] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +9.3834 / +12.6436 | +11.0135 | [+7.5455, +14.3401] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0021 / -0.0083 | -0.0052 | [-0.0196, +0.0090] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0846 / +0.0467 | +0.0657 | [+0.0319, +0.1008] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0179 / +0.0079 | +0.0129 | [+0.0000, +0.0267] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +2.5573 / +2.4372 | +2.4973 | [+1.8732, +3.1528] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +1.3847 / +1.2812 | +1.3329 | [+0.9443, +1.7773] | 0.4517 | WORSE |
| ade6_m|bulk | +2.4632 / +2.4125 | +2.4379 | [+1.7432, +3.2010] | 0.3535 | WORSE |
| ade6_m|all | +1.8765 / +1.7971 | +1.8368 | [+1.3832, +2.3113] | 0.4002 | WORSE |
| ade_tac_m|all | +2.5573 / +2.4372 | +2.4973 | [+1.8732, +3.1528] | 0.5955 | WORSE |

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
| LSDMC_s0 | 4.0227 | 4.3023 | 2.1629 | 1.2615 | 0.7570 |
| LSDMC_s1 | 3.7735 | 3.9747 | 2.1008 | 1.2822 | 0.7420 |
| LSDMG_s0 | 4.0021 | 4.2808 | 2.1566 | 1.2572 | 0.7591 |
| LSDMG_s1 | 3.7645 | 3.9758 | 2.1130 | 1.2876 | 0.7428 |
| LSDMH_s0 | 4.9472 | 4.9649 | 2.4610 | 1.3058 | 0.7657 |
| LSDMH_s1 | 4.8061 | 4.7878 | 2.4810 | 1.3683 | 0.7561 |
| LSDML_s0 | 3.6127 | 3.9858 | 2.0428 | 1.2256 | 0.7599 |
| LSDML_s1 | 3.3527 | 3.6773 | 1.9668 | 1.2894 | 0.7449 |
| LSDMT_s0 | 3.4242 | 3.8226 | 1.9834 | 1.2239 | 0.7570 |
| LSDMT_s1 | 3.2696 | 3.6493 | 1.9708 | 1.3160 | 0.7478 |
| LSMC_s0 | 3.8137 | 4.1854 | 2.1411 | 1.2858 | 0.6874 |
| LSMC_s1 | 3.7776 | 4.0018 | 1.9089 | 1.2842 | 0.7090 |
| LSMG_s0 | 3.8119 | 4.1903 | 2.1458 | 1.2925 | 0.6878 |
| LSMG_s1 | 3.7776 | 4.0018 | 1.9089 | 1.2842 | 0.7090 |
| LSMH_s0 | 4.6294 | 4.7716 | 2.4035 | 1.3082 | 0.7015 |
| LSMH_s1 | 4.7600 | 4.7580 | 2.2584 | 1.3839 | 0.7149 |
| LSML_s0 | 3.4702 | 3.8586 | 2.0209 | 1.2529 | 0.6840 |
| LSML_s1 | 3.2298 | 3.5553 | 1.7336 | 1.2442 | 0.7032 |
| LSMT_s0 | 3.3099 | 3.6588 | 1.9811 | 1.2434 | 0.6761 |
| LSMT_s1 | 3.1634 | 3.5230 | 1.6962 | 1.2410 | 0.7040 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
