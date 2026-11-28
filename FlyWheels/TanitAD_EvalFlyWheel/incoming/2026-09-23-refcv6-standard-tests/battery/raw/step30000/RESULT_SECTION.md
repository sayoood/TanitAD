### step30000: step 30000, ckpt md5 `0c5c67b3ecf28495d54348e3436371bf`, SPEC sha at run `a8095594d2d1…`

**Tier and estimator.** T1 = self-action open loop; T0 = `oracle_sel` only. Paired episode-cluster bootstrap. One training seed. Every separated cell answers only the EPISODE question; the INFERENCE question is answered by the two-seed replicate below.

⛔ **Run-defect stamp (pre-switch checkpoint, step ≤ 34,500):** F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early (D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; audit 92337fa6). No weakness below may be attributed to the REGISTERED design while these hold. TACTICAL is read under BOTH label clocks (SPEC A5); this checkpoint's PRIMARY clock is the OLD one it was trained on. Physical-unit rates use dt = 0.5 s for a true 0.5033 s (row = 0.100667 s), identically for every arm and baseline.

**Gate.** G0 as registered **FAIL**, G0-A1 **FAIL**, G0-A2 (operative) **PASS**.
* A2 max wrapper rel: {'P1_as_run': 0.008076489252922635, 'P2_cudnn_det': 0.008074943843892514, 'P3_fp32_det': 2.074791357176956e-06}.
* Reasons: A1 ['wrapper control failed: max rel 8.076e-03']; A2 [].

| bar | statement | seed 0 | seed 1 | verdict |
|---|---|---|---|---|
| BAR-R6-1 | refcv6 beats the ECHO control: os - ha0_ext < 0, CI upper < 0 | +0.0225 [+0.0068, +0.0395] **sep** | +0.0224 [+0.0070, +0.0387] **sep** | **FAIL** |
| BAR-R6-2 | refcv6 beats HOLD-ACTION: os - ha < 0, separated | +0.0101 [-0.0066, +0.0280] | +0.0100 [-0.0066, +0.0272] | **FAIL** |
| BAR-R6-3 | refcv6 beats refcv4b: os - refcv4b.os < 0, separated | +0.0137 [-0.0029, +0.0317] | +0.0136 [-0.0028, +0.0313] | **FAIL** |
| BAR-R6-4 | refcv6 beats refcv5-v2 (seed 0): os - refcv5v2.os < 0, separated | +0.0021 [-0.0160, +0.0212] | +0.0020 [-0.0160, +0.0209] | **FAIL** |
| BAR-R6-5 | at 6 s: os - ha0_ext (ADE 0-6 s) < 0, separated | -0.7475 [-1.0603, -0.4409] **sep** | -0.7472 [-1.0537, -0.4377] **sep** | **PASS** |

⭐ **Primary bar BAR-R6-1 = FAIL.** At this checkpoint refcv6 is **NOT PROVEN** to beat the echo control at 0–2 s.

**Inference-seed replicate** (os seed 0 − os seed 1, same windows): ADE +0.0001 [-0.0027, +0.0027]; max |path diff| 3.5125 m.

**Key paired cells, S2 (0–2 s), inference seed 0**

| cell (b − a) | ADE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|
| os − ha0_ext (echo) | +0.0225 [+0.0068, +0.0395] **sep** | +0.0192 [+0.0041, +0.0346] **sep** | +0.0109 [-0.0032, +0.0249] | +0.0521 [-0.0555, +0.1668] | -0.0018 [-0.0044, +0.0009] | +0.0163 [+0.0068, +0.0266] **sep** | +0.0158 [+0.0036, +0.0269] **sep** | -0.0116 [-0.0274, +0.0036] |
| os − ha (hold) | +0.0101 [-0.0066, +0.0280] | +0.0192 [+0.0041, +0.0346] **sep** | +0.0102 [-0.0039, +0.0243] | -0.0653 [-0.1828, +0.0586] | -0.0038 [-0.0068, -0.0008] **sep** | +0.0005 [-0.0109, +0.0120] | +0.0196 [+0.0069, +0.0315] **sep** | -0.0116 [-0.0274, +0.0036] |
| os − ha0 (CV) | -0.3653 [-0.4294, -0.3049] **sep** | -0.2151 [-0.2529, -0.1792] **sep** | -0.2262 [-0.2608, -0.1932] **sep** | -1.3138 [-1.6865, -0.9638] **sep** | -0.0202 [-0.0271, -0.0140] **sep** | -0.1928 [-0.2536, -0.1370] **sep** | +0.0930 [+0.0605, +0.1259] **sep** | +0.0751 [+0.0441, +0.1040] **sep** |
| os − refcv4b | +0.0137 [-0.0029, +0.0317] | -0.0169 [-0.0365, +0.0022] | -0.0097 [-0.0291, +0.0093] | +0.1266 [+0.0269, +0.2433] **sep** | -0.0098 [-0.0131, -0.0068] **sep** | +0.0263 [+0.0158, +0.0377] **sep** | -0.0004 [-0.0088, +0.0078] | +0.0057 [-0.0103, +0.0210] |
| os − refcv5-v2 s0 | +0.0021 [-0.0160, +0.0212] | -0.0189 [-0.0389, +0.0016] | -0.0210 [-0.0408, -0.0012] **sep** | +0.1991 [+0.1086, +0.3079] **sep** | +0.0032 [+0.0017, +0.0051] **sep** | +0.0249 [+0.0149, +0.0360] **sep** | +0.0013 [-0.0059, +0.0084] | -0.0025 [-0.0196, +0.0139] |
| nav withheld − os | +0.0266 [+0.0167, +0.0361] **sep** | +0.0278 [+0.0184, +0.0375] **sep** | +0.0288 [+0.0197, +0.0382] **sep** | +0.0668 [-0.0124, +0.1541] | +0.0014 [-0.0002, +0.0032] | +0.0008 [-0.0045, +0.0063] | -0.0042 [-0.0097, +0.0011] | -0.0097 [-0.0210, +0.0011] |
| nav shuffled − os | +0.0050 [+0.0002, +0.0097] **sep** | +0.0041 [+0.0001, +0.0079] **sep** | +0.0043 [+0.0005, +0.0080] **sep** | +0.0575 [-0.0036, +0.1196] | +0.0013 [+0.0001, +0.0027] **sep** | +0.0021 [-0.0021, +0.0060] | -0.0053 [-0.0109, -0.0002] **sep** | -0.0042 [-0.0107, +0.0029] |
| max-speed withheld − os | +0.0000 [-0.0028, +0.0030] | +0.0004 [-0.0029, +0.0039] | +0.0005 [-0.0025, +0.0037] | +0.0028 [-0.0061, +0.0135] | +0.0001 [-0.0001, +0.0004] | -0.0002 [-0.0013, +0.0008] | -0.0002 [-0.0023, +0.0019] | -0.0002 [-0.0065, +0.0059] |

n = 4754 windows / 139 episodes; paired episode-cluster bootstrap (n_boot 2000, seed 0, cluster = clip); **sep** = CI excludes 0

**Key paired cells, S2 (0–2 s), inference seed 1**

| cell (b − a) | ADE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|
| os − ha0_ext (echo) | +0.0224 [+0.0070, +0.0387] **sep** | +0.0197 [+0.0051, +0.0350] **sep** | +0.0111 [-0.0030, +0.0251] | +0.0490 [-0.0576, +0.1627] | -0.0018 [-0.0044, +0.0008] | +0.0161 [+0.0065, +0.0262] **sep** | +0.0154 [+0.0031, +0.0266] **sep** | -0.0116 [-0.0265, +0.0023] |
| os − ha (hold) | +0.0100 [-0.0066, +0.0272] | +0.0197 [+0.0051, +0.0350] **sep** | +0.0103 [-0.0038, +0.0245] | -0.0684 [-0.1861, +0.0545] | -0.0038 [-0.0068, -0.0009] **sep** | +0.0002 [-0.0108, +0.0116] | +0.0191 [+0.0065, +0.0309] **sep** | -0.0116 [-0.0265, +0.0023] |
| os − ha0 (CV) | -0.3654 [-0.4300, -0.3045] **sep** | -0.2147 [-0.2528, -0.1790] **sep** | -0.2260 [-0.2607, -0.1924] **sep** | -1.3178 [-1.6863, -0.9694] **sep** | -0.0202 [-0.0271, -0.0140] **sep** | -0.1930 [-0.2537, -0.1374] **sep** | +0.0926 [+0.0606, +0.1249] **sep** | +0.0751 [+0.0457, +0.1025] **sep** |
| os − refcv4b | +0.0136 [-0.0028, +0.0313] | -0.0164 [-0.0360, +0.0027] | -0.0095 [-0.0288, +0.0095] | +0.1248 [+0.0269, +0.2391] **sep** | -0.0099 [-0.0131, -0.0068] **sep** | +0.0261 [+0.0156, +0.0373] **sep** | -0.0008 [-0.0092, +0.0076] | +0.0057 [-0.0091, +0.0202] |
| os − refcv5-v2 s0 | +0.0020 [-0.0160, +0.0209] | -0.0185 [-0.0383, +0.0016] | -0.0209 [-0.0405, -0.0012] **sep** | +0.1974 [+0.1092, +0.3047] **sep** | +0.0032 [+0.0017, +0.0051] **sep** | +0.0246 [+0.0147, +0.0360] **sep** | +0.0008 [-0.0063, +0.0082] | -0.0025 [-0.0187, +0.0122] |
| nav withheld − os | +0.0258 [+0.0158, +0.0363] **sep** | +0.0258 [+0.0165, +0.0359] **sep** | +0.0273 [+0.0183, +0.0369] **sep** | +0.0662 [-0.0118, +0.1517] | +0.0012 [-0.0003, +0.0030] | +0.0009 [-0.0046, +0.0065] | -0.0034 [-0.0090, +0.0017] | -0.0124 [-0.0236, -0.0025] **sep** |
| nav shuffled − os | +0.0039 [-0.0011, +0.0089] | +0.0021 [-0.0018, +0.0061] | +0.0029 [-0.0008, +0.0068] | +0.0590 [-0.0070, +0.1269] | +0.0012 [-0.0002, +0.0025] | +0.0021 [-0.0023, +0.0064] | -0.0046 [-0.0097, -0.0002] **sep** | -0.0053 [-0.0126, +0.0017] |
| max-speed withheld − os | -0.0018 [-0.0049, +0.0010] | -0.0011 [-0.0038, +0.0014] | -0.0005 [-0.0032, +0.0021] | -0.0106 [-0.0337, +0.0056] | -0.0001 [-0.0005, +0.0002] | -0.0013 [-0.0034, +0.0003] | +0.0004 [-0.0017, +0.0023] | +0.0032 [-0.0017, +0.0082] |

n = 4754 windows / 139 episodes; paired episode-cluster bootstrap (n_boot 2000, seed 0, cluster = clip); **sep** = CI excludes 0

**Levels, S2, inference seed 0** (4754 windows / 139 eps)

| arm | tier | ADE 0–2 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 0.3111 [0.2847, 0.3409] | 0.6632 | 0.2740 | 0.8418 | 0.2456 | 1.4938 | 0.004240 | 1.2461 | 0.1243 | 0.8223 | 0.5663 |
| refcv6 nav withheld | T1 | 0.3377 [0.3096, 0.3699] | 0.7173 | 0.3017 | 0.8143 | 0.2744 | 1.5582 | 0.004823 | 1.3191 | 0.1251 | 0.8014 | 0.5315 |
| refcv6 max-speed withheld | T1 | 0.3111 [0.2850, 0.3409] | 0.6626 | 0.2743 | 0.8424 | 0.2461 | 1.4952 | 0.004278 | 1.2527 | 0.1240 | 0.8214 | 0.5641 |
| refcv4b `os` (banked) | T1 | 0.2974 [0.2689, 0.3303] | 0.6363 | 0.2908 | 0.8321 | 0.2553 | 1.3103 | 0.008266 | 1.7557 | 0.0979 | 0.8274 | 0.5215 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 0.3090 [0.2797, 0.3427] | 0.6629 | 0.2929 | 0.8338 | 0.2666 | 1.2241 | 0.003529 | 1.0619 | 0.0994 | 0.8190 | 0.5450 |
| `ha` hold-action | T1 | 0.3010 [0.2761, 0.3298] | 0.6623 | 0.2548 | 0.8655 | 0.2354 | 1.5706 | 0.004089 | 1.4738 | 0.1238 | 0.7370 | 0.6066 |
| `ha0` constant velocity | T1 | 0.6764 [0.6033, 0.7529] | 1.4113 | 0.4891 | 0.7046 | 0.4718 | 2.8200 | 0.006903 | 2.4052 | 0.3170 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 0.2886 [0.2652, 0.3158] | 0.6355 | 0.2548 | 0.8655 | 0.2347 | 1.4522 | 0.003767 | 1.3575 | 0.1079 | 0.7544 | 0.6066 |
| `oracle_sel` (T0 ceiling) | T0 | 0.2435 [0.2266, 0.2623] | 0.4900 | 0.2017 | 0.9107 | 0.1839 | 0.8948 | 0.003143 | 1.0432 | 0.1098 | 0.8581 | 0.7309 |

**Distance keeping (LONGITUDINAL)**

lead block: `b1_eval_lead_block.npz`, status PRESENT, instants [1.0, 2.0] s

| arm | status | n with lead (eps) | min headway m [CI] | min time-gap s [CI] (n) | min TTC s [CI] | n closing (rest censored at 30 s) |
|---|---|---|---|---|---|---|
| **refcv6 `os`** | OK | 1314 (69) | 27.8790 [23.8031, 32.3277] | 4.0746 [3.3073, 5.0658] (1173) | 24.6659 [22.9649, 26.1466] | 508 |
| refcv6 nav withheld | OK | 1306 (68) | 27.8516 [23.8309, 32.1673] | 4.1385 [3.2984, 5.1611] (1165) | 24.9775 [23.3242, 26.5557] | 474 |
| refcv6 max-speed withheld | OK | 1304 (70) | 27.9460 [23.9915, 32.2928] | 4.1172 [3.3074, 5.0833] (1163) | 24.7069 [23.0508, 26.1266] | 503 |
| refcv4b `os` (banked) | OK | 1224 (67) | 28.4517 [24.4003, 32.7611] | 4.1118 [3.2775, 5.0945] (1154) | 24.8434 [23.3648, 26.1655] | 470 |
| refcv5-v2 `os` seed 0 (banked) | OK | 1308 (68) | 27.8285 [23.9407, 32.4796] | 4.0885 [3.2646, 5.0234] (1167) | 24.6454 [23.2247, 26.0281] | 524 |
| `ha` hold-action | OK | 1291 (67) | 28.0480 [23.8033, 32.3785] | 4.2073 [3.3867, 5.2420] (1150) | 25.5883 [24.2766, 26.7972] | 447 |
| `ha0` constant velocity | OK | 1315 (70) | 27.4550 [23.7229, 31.7446] | 4.1178 [3.2811, 5.1281] (1174) | 24.1077 [22.3616, 25.6733] | 499 |
| `ha0_ext` echo | OK | 1285 (67) | 28.0211 [23.7324, 32.3570] | 4.1966 [3.3733, 5.2300] (1144) | 25.5453 [24.2463, 26.7708] | 446 |
| `oracle_sel` (T0 ceiling) | OK | 1311 (70) | 27.9034 [24.0025, 32.2111] | 4.1013 [3.3127, 5.0721] (1170) | 24.9039 [23.4558, 26.1997] | 511 |

**TACTICAL, inference seed 0.** Declared heads use the trainer's label clock (stamp above).

**LAT** (v8 labels, in-band windows n = 1141 / 139 eps; classes ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'ABORT_LC', 'NUDGE_L', 'NUDGE_R', 'TURN_L', 'TURN_R'])

| surface | acc [CI] | κ | majority-class rate |
|---|---|---|---|
| v6_behaviour_decoder | 0.7546 [0.7002, 0.8121] | 0.5181 | 0.6713 |
| z_tac_v7_heads | 0.7117 [0.6427, 0.7791] | 0.3242 | 0.6713 |
| v6_behaviour_decoder_NAVZERO | 0.7187 [0.6541, 0.7764] | 0.4081 | 0.6713 |

**LON** (v8 labels, in-band windows n = 1141 / 139 eps; classes ['FOLLOW', 'CRUISE', 'YIELD_MERGE', 'BRAKE_TO', 'CREEP', 'HOLD', 'ADAPT_SPEED_FOR_CURVE', 'ACCELERATE'])

| surface | acc [CI] | κ | majority-class rate |
|---|---|---|---|
| v6_behaviour_decoder | 0.4943 [0.4252, 0.5615] | 0.3400 | 0.3024 |
| z_tac_v7_heads | 0.5320 [0.4592, 0.6009] | 0.3828 | 0.3024 |
| v6_behaviour_decoder_NAVZERO | 0.4514 [0.3817, 0.5184] | 0.2925 | 0.3024 |

**22-token goal selection** (per class, nav true; floor n_pos 200)

| token | n_pos / n_neg | eps | AUROC | AP | prevalence | P@0.5 | R@0.5 | AUROC nav-zero | status |
|---|---|---|---|---|---|---|---|---|---|
| FOLLOW_LANE | 911 / 230 | 139 | 0.7999 | 0.8818 | 0.7984 | 0.9227 | 0.8386 | 0.8125 | SCOREABLE |
| TURN_L | 40 / 1101 | 139 | 0.9881 | 0.8633 | 0.0351 | 0.5614 | 0.8000 | 0.9506 | UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 1074 | 139 | 0.9695 | 0.6142 | 0.0587 | 0.3688 | 0.8806 | 0.7689 | UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 1125 | 139 | 0.9983 | 0.8568 | 0.0140 | 0.6667 | 1.0000 | 0.9761 | UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / 1141 | 139 | — | — | 0.0000 | 0.0000 | — | — | UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / 0 | 17 | — | 1.0000 | 1.0000 | 1.0000 | 0.0647 | — | UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 1067 | 139 | 0.8598 | 0.3074 | 0.0649 | 0.2257 | 0.7838 | 0.8328 | UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / 0 | 139 | — | 1.0000 | 1.0000 | 1.0000 | 0.1543 | — | SCOREABLE |
| CORRIDOR_OFFSET | 200 / 0 | 24 | — | 1.0000 | 1.0000 | 1.0000 | 0.2000 | — | SCOREABLE |
| EVADE_IN_CORRIDOR | 49 / 9 | 7 | 0.9546 | 0.9924 | 0.8448 | 1.0000 | 0.8367 | 0.9977 | UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 911 | 112 | 0.9889 | 0.6532 | 0.0098 | 0.0918 | 1.0000 | 0.9948 | UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 911 | 116 | 0.6168 | 0.0564 | 0.0421 | 0.0578 | 0.4000 | 0.6292 | UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / 0 | 12 | — | 1.0000 | 1.0000 | — | 0.0000 | — | UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / 0 | 10 | — | 1.0000 | 1.0000 | 1.0000 | 0.2195 | — | UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / 108 | 13 | — | — | 0.0000 | 0.0000 | — | — | UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 40 | 10 | 0.8915 | 0.8571 | 0.5062 | 0.8367 | 1.0000 | 0.9012 | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / 212 | 26 | — | — | 0.0000 | 0.0000 | — | — | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 147 | 26 | 0.8759 | 0.8367 | 0.3066 | 0.8000 | 0.8000 | 0.7798 | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 187 | 26 | 0.6340 | 0.1466 | 0.1179 | 0.0000 | 0.0000 | 0.6090 | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 90 | 26 | 0.7087 | 0.6645 | 0.5755 | 0.7474 | 0.5820 | 0.7007 | UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / 1141 | 139 | — | — | 0.0000 | 0.0000 | — | — | UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 67 | 9 | 1.0000 | 1.0000 | 0.1067 | 1.0000 | 1.0000 | 1.0000 | UNSCOREABLE (n_pos < 200) |

**TACTICAL, inference seed 1:** LAT: v6 decoder acc 0.7546 [0.7002, 0.8121] κ 0.5181; LON: v6 decoder acc 0.4943 [0.4252, 0.5615] κ 0.3400

### TACTICAL under both label clocks (SPEC A5): step30000, step 30000. PRIMARY clock: **OLD**

Clock and dt source: CORRECTED = grid_start + (t + w - 1 + n_stack - 1) * dt; 136/139 eval clips on the sidecar clock (dt median 0.10066662995966351), 3 on nominal dt 0.1 s with grid_start 0, 0 on pose dt; OLD = (t + w - 1) * 0.1 s.

Tier T1 (UNRULED for an action-free model). Declared-head accuracy [episode-cluster CI], κ full-set; in-band n per clock. Label tables sha256 `83b47aae4aff…`; controls C1–C4 PASS (`raw/label_clock/label_clock_record.json`).

**Inference seed 0**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7546 [0.7002, 0.8121] · 0.5181 (1141) ⭐ | 0.7514 [0.6959, 0.8105] · 0.5083 (1106) |
| LAT | z_tac_v7_heads | 0.7117 [0.6427, 0.7791] · 0.3242 (1141) ⭐ | 0.7107 [0.6417, 0.7796] · 0.3143 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.7187 [0.6541, 0.7764] · 0.4081 (1141) ⭐ | 0.7134 [0.6492, 0.7729] · 0.3964 (1106) |
| LON | v6_behaviour_decoder | 0.4943 [0.4252, 0.5615] · 0.3400 (1141) ⭐ | 0.4901 [0.4224, 0.5575] · 0.3354 (1106) |
| LON | z_tac_v7_heads | 0.5320 [0.4592, 0.6009] · 0.3828 (1141) ⭐ | 0.5289 [0.4551, 0.5969] · 0.3791 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.4514 [0.3817, 0.5184] · 0.2925 (1141) ⭐ | 0.4458 [0.3736, 0.5118] · 0.2853 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7999 / SCOREABLE | 882 / 0.7980 / SCOREABLE |
| TURN_L | 40 / 0.9881 / UNSCOREABLE (n_pos < 200) | 40 / 0.9883 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9695 / UNSCOREABLE (n_pos < 200) | 63 / 0.9684 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9983 / UNSCOREABLE (n_pos < 200) | 16 / 0.9975 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8598 / UNSCOREABLE (n_pos < 200) | 73 / 0.8547 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.9546 / UNSCOREABLE (n_pos < 200) | 48 / 0.9505 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.9889 / UNSCOREABLE (n_pos < 200) | 8 / 0.9940 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.6168 / UNSCOREABLE (n_pos < 200) | 40 / 0.6092 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 0.8915 / UNSCOREABLE (n_pos < 200) | 40 / 0.8919 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.8759 / UNSCOREABLE (n_pos < 200) | 63 / 0.8831 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6340 / UNSCOREABLE (n_pos < 200) | 24 / 0.6335 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7087 / UNSCOREABLE (n_pos < 200) | 118 / 0.6974 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

**Inference seed 1**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7546 [0.7002, 0.8121] · 0.5181 (1141) ⭐ | 0.7514 [0.6959, 0.8105] · 0.5083 (1106) |
| LAT | z_tac_v7_heads | 0.7117 [0.6427, 0.7791] · 0.3242 (1141) ⭐ | 0.7107 [0.6417, 0.7796] · 0.3143 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.7187 [0.6541, 0.7764] · 0.4081 (1141) ⭐ | 0.7134 [0.6492, 0.7729] · 0.3964 (1106) |
| LON | v6_behaviour_decoder | 0.4943 [0.4252, 0.5615] · 0.3400 (1141) ⭐ | 0.4901 [0.4224, 0.5575] · 0.3354 (1106) |
| LON | z_tac_v7_heads | 0.5320 [0.4592, 0.6009] · 0.3828 (1141) ⭐ | 0.5289 [0.4551, 0.5969] · 0.3791 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.4514 [0.3817, 0.5184] · 0.2925 (1141) ⭐ | 0.4458 [0.3736, 0.5118] · 0.2853 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7999 / SCOREABLE | 882 / 0.7980 / SCOREABLE |
| TURN_L | 40 / 0.9881 / UNSCOREABLE (n_pos < 200) | 40 / 0.9883 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9695 / UNSCOREABLE (n_pos < 200) | 63 / 0.9684 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9983 / UNSCOREABLE (n_pos < 200) | 16 / 0.9975 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8598 / UNSCOREABLE (n_pos < 200) | 73 / 0.8547 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.9546 / UNSCOREABLE (n_pos < 200) | 48 / 0.9505 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.9889 / UNSCOREABLE (n_pos < 200) | 8 / 0.9940 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.6168 / UNSCOREABLE (n_pos < 200) | 40 / 0.6092 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 0.8915 / UNSCOREABLE (n_pos < 200) | 40 / 0.8919 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.8759 / UNSCOREABLE (n_pos < 200) | 63 / 0.8831 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6340 / UNSCOREABLE (n_pos < 200) | 24 / 0.6335 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7087 / UNSCOREABLE (n_pos < 200) | 118 / 0.6974 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

⚠ Nav-echo caveat (Master Mind register REFCV6-BATTERY-5K): on PhysicalAI the nav input is derived from the ego's own future path, so every nav-true LAT κ here is optimistic by construction (the nav-echo family); the NAVZERO rows are the nav-free reading.

⭐ = the PRIMARY clock for this checkpoint. A tactical comparison across step 34,500 must show both clocks, and it mixes training time with the fix.


**STRATEGIC: NOT APPLICABLE, n = 0.** The strategic layer is OFF (`--no-strategic`); its route CE is gated off in training, so the route head is untrained and is not scored.

**S6: the 6 s horizon, ADE 1–6 s** (BAR-R6-5 reads the os − ha0_ext column)

| seed | os ADE 1–6 s [CI] | os − ha0_ext | os − refcv4b | os − refcv5-v2 s0 |
|---|---|---|---|---|
| 0 | 2.8004 [2.5672, 3.0383] | -0.7475 [-1.0603, -0.4409] **sep** | +0.0779 [-0.0651, +0.2413] | +0.0863 [-0.0732, +0.2616] |
| 1 | 2.8007 [2.5677, 3.0416] | -0.7472 [-1.0537, -0.4377] **sep** | +0.0782 [-0.0706, +0.2431] | +0.0867 [-0.0734, +0.2604] |

**Frozen acceptance instruments, inference seed 0.**
* **T-FLIP: FAIL.** follows_FED 0.2159 CI [0.1295, 0.3037], against a bar of 0.50 (refcv5-v2: 0.205).
  * true − shuffled: 0.1562 CI [0.1054, 0.2111], against a bar of 0.38 (refcv5-v2: 0.099).
  * n = 352 windows / 29 episodes (quote the n with it).
* **OBEDIENCE: FAIL.** Obeys 0.0126; 2232 of 2294 rows have NO compliant candidate. The bar is structurally unsatisfiable on this population (§3 F8).

**Frozen acceptance instruments, inference seed 1.**
* **T-FLIP: FAIL.** follows_FED 0.2216 CI [0.1323, 0.3107], against a bar of 0.50 (refcv5-v2: 0.205).
  * true − shuffled: 0.142 CI [0.0968, 0.1929], against a bar of 0.38 (refcv5-v2: 0.099).
  * n = 352 windows / 29 episodes (quote the n with it).
* **OBEDIENCE: FAIL.** Obeys 0.0126; 2232 of 2294 rows have NO compliant candidate. The bar is structurally unsatisfiable on this population (§3 F8).
* VOID gates, seed 0: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.4697, 'n_distinct_selected': 52, 'os_trivial_frac': 0.0042}
* VOID gates, seed 1: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.4716, 'n_distinct_selected': 53, 'os_trivial_frac': 0.0044}

### A4 lever panel — 4754 windows / 139 episodes, S2 ADE 0–2 s, T1 (self-action open loop); every lever is a DIFFERENT planner from the registered arm

Estimator: paired episode-cluster bootstrap, n_boot 2000, seed 0, cluster = clip. SPEC sha `314a4845f7de…`.

**L1 inference-seed average** (sign vs single seeds guaranteed; magnitude only)

| cell | ADE m [CI] |
|---|---|
| os_avg − ha0_ext | +0.0172 [+0.0016, +0.0336] **sep** |
| os_avg − os_s0 | -0.0053 [-0.0067, -0.0039] **sep** |
| os_avg − os_s1 | -0.0052 [-0.0067, -0.0038] **sep** |
| replicate floor os_s0 − os_s1 | +0.0001 [-0.0027, +0.0027] ns |

|os_s0 − os_s1| per instant (0.5/1/1.5/2 s): mean [0.01769999973475933, 0.06159999966621399, 0.11490000039339066, 0.1793999969959259], p95 [0.03849999979138374, 0.1412000060081482, 0.295199990272522, 0.5411999821662903] m

**L2 causal-hold blend (base `ha`)**

| seed | blend − ha0_ext | blend − base | blend − blend_shuf | blend_shuf − base | w (fold0→1 / fold1→0) | identity |
|---|---|---|---|---|---|---|
| seed0 | -0.0364 [-0.0464, -0.0265] **sep** | -0.0488 [-0.0605, -0.0371] **sep** | -0.0488 [-0.0605, -0.0371] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.65, 0.45, 0.45, 0.5] / [0.65, 0.45, 0.45, 0.55] | PASS |
| seed1 | -0.0361 [-0.0461, -0.0266] **sep** | -0.0485 [-0.0603, -0.0371] **sep** | -0.0485 [-0.0603, -0.0371] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.65, 0.45, 0.45, 0.5] / [0.6, 0.4, 0.45, 0.55] | PASS |

**L2e echo blend (base `ha0_ext`, DIAGNOSTIC ONLY)**

| seed | blend − ha0_ext | blend − base | blend − blend_shuf | blend_shuf − base | w (fold0→1 / fold1→0) | identity |
|---|---|---|---|---|---|---|
| seed0 | -0.0419 [-0.0519, -0.0321] **sep** | -0.0419 [-0.0519, -0.0321] **sep** | -0.0419 [-0.0519, -0.0321] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.65, 0.4, 0.4, 0.45] / [0.6, 0.4, 0.4, 0.5] | PASS |
| seed1 | -0.0418 [-0.0517, -0.0322] **sep** | -0.0418 [-0.0517, -0.0322] **sep** | -0.0418 [-0.0517, -0.0322] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.65, 0.4, 0.4, 0.45] / [0.6, 0.4, 0.4, 0.5] | PASS |


Full tables: `raw/step30000/TABLES_s0.md`, `TABLES_s1.md`.
