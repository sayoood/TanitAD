# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 22}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "LSQH"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## LSQH minus LS

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0785 / +0.1580 | +0.1182 | [+0.0008, +0.2396] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.4976 / +0.6157 | +0.5567 | [+0.3064, +0.8306] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0028 / +0.0206 | +0.0117 | [-0.0242, +0.0518] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.5550 / +1.1460 | +0.8505 | [+0.3919, +1.3826] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0160 / +0.0347 | +0.0253 | [+0.0117, +0.0438] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1337 / +4.1424 | +2.1380 | [-0.9985, +5.1235] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0038 / -0.0079 | -0.0021 | [-0.0167, +0.0123] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0150 / +0.0038 | +0.0094 | [-0.0067, +0.0240] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0154 / -0.0017 | +0.0069 | [-0.0079, +0.0223] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.6478 / +0.8245 | +0.7362 | [+0.3912, +1.1006] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3299 / +0.4163 | +0.3731 | [+0.1487, +0.6281] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.6246 / +0.8047 | +0.7146 | [+0.3702, +1.1200] | 0.3535 | WORSE |
| ade6_m|all | +0.4643 / +0.5934 | +0.5289 | [+0.2794, +0.7930] | 0.4002 | WORSE |
| ade_tac_m|all | +0.6478 / +0.8245 | +0.7362 | [+0.3912, +1.1006] | 0.5955 | WORSE |

## LSQH minus LSQC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1084 / -0.1081 | -0.1082 | [-0.1598, -0.0622] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.2687 / -0.3035 | -0.2861 | [-0.4225, -0.1711] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0211 / -0.0313 | -0.0262 | [-0.0533, +0.0019] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.4675 / -0.3282 | -0.3979 | [-0.6780, -0.1378] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0159 / -0.0124 | -0.0141 | [-0.0232, -0.0064] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.2560 / -1.6863 | -1.9711 | [-3.0978, -0.9296] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0008 / +0.0008 | +0.0008 | [-0.0067, +0.0079] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0050 / -0.0129 | -0.0090 | [-0.0171, -0.0004] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0004 / -0.0046 | -0.0025 | [-0.0090, +0.0042] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.3581 / -0.4291 | -0.3936 | [-0.5957, -0.2279] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.1913 / -0.1985 | -0.1949 | [-0.3304, -0.0752] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.3339 / -0.4387 | -0.3863 | [-0.6094, -0.2109] | 0.3535 | BETTER |
| ade6_m|all | -0.2564 / -0.3080 | -0.2822 | [-0.4249, -0.1631] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | -0.3581 / -0.4291 | -0.3936 | [-0.5957, -0.2279] | 0.5955 | INSIDE-SPREAD |

## LSQH minus LSMC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST GAIN-LONGITUDINAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1147 / -0.1700 | -0.1424 | [-0.2066, -0.0738] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.3555 / -0.4989 | -0.4272 | [-0.5997, -0.2577] | 0.3530 (0.2400) | BETTER |
| LATERAL:cross_mae_m | -0.0202 / -0.0270 | -0.0236 | [-0.0586, +0.0147] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.5330 / -1.1218 | -0.8274 | [-1.4101, -0.3224] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0317 / -0.0419 | -0.0368 | [-0.0548, -0.0211] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.2446 / -5.4183 | -3.8315 | [-6.5518, -1.2245] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0008 / +0.0008 | +0.0000 | [-0.0085, +0.0092] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0008 / -0.0129 | -0.0060 | [-0.0198, +0.0079] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0013 / -0.0063 | -0.0038 | [-0.0121, +0.0038] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.4750 / -0.6635 | -0.5693 | [-0.8052, -0.3223] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.3413 / -0.4150 | -0.3782 | [-0.5537, -0.2117] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.3385 / -0.5857 | -0.4621 | [-0.7379, -0.2143] | 0.3535 | BETTER |
| ade6_m|all | -0.3400 / -0.4929 | -0.4165 | [-0.5901, -0.2369] | 0.4002 | BETTER |
| ade_tac_m|all | -0.4750 / -0.6635 | -0.5693 | [-0.8052, -0.3223] | 0.5955 | INSIDE-SPREAD |

## LSDQH minus LSD

Seeds read [0, 1] of [0, 1]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0868 / -0.0280 | +0.0294 | [-0.0930, +0.1583] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.5579 / +0.4162 | +0.4871 | [+0.2566, +0.7519] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | +0.0121 / -0.0661 | -0.0270 | [-0.0874, +0.0286] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.7447 / +0.3822 | +0.5635 | [+0.2545, +0.9162] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0143 / +0.0146 | +0.0144 | [+0.0084, +0.0214] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.1627 / -0.1490 | +0.0069 | [-1.3955, +1.2950] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0079 / -0.0125 | -0.0102 | [-0.0192, -0.0017] | 0.0838 (0.0204) | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0392 / +0.0054 | +0.0223 | [-0.0010, +0.0502] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0079 / -0.0033 | +0.0023 | [-0.0067, +0.0125] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.7304 / +0.4898 | +0.6101 | [+0.2777, +0.9871] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2865 / +0.2149 | +0.2507 | [+0.0507, +0.4775] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.8082 / +0.5225 | +0.6653 | [+0.3191, +1.0940] | 0.3535 | WORSE |
| ade6_m|all | +0.5244 / +0.3552 | +0.4398 | [+0.2017, +0.7110] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7304 / +0.4898 | +0.6101 | [+0.2777, +0.9871] | 0.5955 | WORSE |

## LSDQH minus LSDQC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1705 / -0.1339 | -0.1522 | [-0.2229, -0.0871] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.4670 / -0.4255 | -0.4462 | [-0.6171, -0.2892] | 0.3530 (0.2400) | BETTER |
| LATERAL:cross_mae_m | -0.0425 / -0.0047 | -0.0236 | [-0.0609, +0.0063] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.6232 / -0.7569 | -0.6900 | [-0.9735, -0.4267] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0192 / -0.0198 | -0.0195 | [-0.0274, -0.0130] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -2.5260 / -2.8790 | -2.7025 | [-3.8314, -1.5654] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0000 / +0.0021 | +0.0010 | [-0.0065, +0.0083] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0242 / -0.0179 | -0.0211 | [-0.0329, -0.0100] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0038 / -0.0038 | -0.0038 | [-0.0088, +0.0010] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.6448 / -0.5710 | -0.6079 | [-0.8490, -0.3840] | 0.5955 (0.2199) | BETTER |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.3990 / -0.1922 | -0.2956 | [-0.5099, -0.1140] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.5383 / -0.6720 | -0.6051 | [-0.8997, -0.3597] | 0.3535 | BETTER |
| ade6_m|all | -0.4625 / -0.4110 | -0.4367 | [-0.6061, -0.2777] | 0.4002 | BETTER |
| ade_tac_m|all | -0.6448 / -0.5710 | -0.6079 | [-0.8490, -0.3840] | 0.5955 | BETTER |

## LSDQH minus LSDMC

Seeds read [0, 1] of [0, 1]. **Q3: NO-COST GAIN-LONGITUDINAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1424 / -0.1231 | -0.1328 | [-0.1917, -0.0749] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.4262 / -0.4385 | -0.4324 | [-0.5832, -0.2906] | 0.3530 (0.2400) | BETTER |
| LATERAL:cross_mae_m | -0.0410 / +0.0165 | -0.0122 | [-0.0582, +0.0324] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.2619 / -0.6766 | -0.4693 | [-0.8378, -0.0440] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0108 / -0.0254 | -0.0181 | [-0.0327, -0.0054] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -1.0768 / -3.5507 | -2.3137 | [-4.3734, -0.3848] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0042 / +0.0038 | +0.0040 | [-0.0035, +0.0115] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0075 / +0.0038 | -0.0019 | [-0.0144, +0.0123] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0013 / +0.0029 | +0.0008 | [-0.0060, +0.0067] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.5771 / -0.5513 | -0.5642 | [-0.7731, -0.3568] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.4356 / -0.2531 | -0.3443 | [-0.5498, -0.1589] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | -0.4182 / -0.5955 | -0.5068 | [-0.7690, -0.2841] | 0.3535 | BETTER |
| ade6_m|all | -0.4276 / -0.4092 | -0.4184 | [-0.5692, -0.2700] | 0.4002 | BETTER |
| ade_tac_m|all | -0.5771 / -0.5513 | -0.5642 | [-0.7731, -0.3568] | 0.5955 | INSIDE-SPREAD |

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
| LSDQC_s0 | 4.0576 | 4.2657 | 2.1910 | 1.2630 | 0.7595 |
| LSDQC_s1 | 3.7752 | 3.9137 | 2.1116 | 1.3035 | 0.7486 |
| LSDQH_s0 | 3.5951 | 3.8667 | 2.0205 | 1.2205 | 0.7557 |
| LSDQH_s1 | 3.3642 | 3.7215 | 1.9777 | 1.2988 | 0.7449 |
| LSMC_s0 | 3.8137 | 4.1854 | 2.1411 | 1.2858 | 0.6874 |
| LSMC_s1 | 3.7776 | 4.0018 | 1.9089 | 1.2842 | 0.7090 |
| LSQC_s0 | 3.7300 | 4.0354 | 2.1348 | 1.2867 | 0.6865 |
| LSQC_s1 | 3.5927 | 3.7853 | 1.8469 | 1.2885 | 0.7074 |
| LSQH_s0 | 3.4736 | 3.8440 | 2.0264 | 1.2656 | 0.6861 |
| LSQH_s1 | 3.2847 | 3.5867 | 1.7389 | 1.2572 | 0.7028 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
