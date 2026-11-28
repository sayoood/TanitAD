## step30000 — ckpt md5 `0c5c67b3ecf28495d54348e3436371bf`, step 30000, SPEC sha256 `a8095594d2d1bddf…`

**Tier:** every number is T1 = self-action OPEN loop (UNRULED for an action-free model) except `oracle_sel` (T0). Not closed loop, not driving. **Estimator:** full-set pooled means; paired episode-cluster bootstrap (n_boot 2000, seed 0, cluster = clip). One training seed: every separated cell answers the EPISODE question only; the INFERENCE question is answered by the seed replicate.

⛔ **Run-defect stamp (pre-switch checkpoint, step ≤ 34,500):** F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early (D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; audit 92337fa6). No weakness below may be attributed to the REGISTERED design while these hold. TACTICAL is read under BOTH label clocks (SPEC A5); this checkpoint's PRIMARY clock is the OLD one it was trained on. Physical-unit rates use dt = 0.5 s for a true 0.5033 s (row = 0.100667 s), identically for every arm and baseline.

### Gate

* G0 as registered: **FAIL**; G0-A1: **FAIL**; G0-A2 (operative): **PASS**; A2 wrapper max rel by condition: {'P1_as_run': 0.008076489252922635, 'P2_cudnn_det': 0.008074943843892514, 'P3_fp32_det': 2.074791357176956e-06}; A1 reasons ['wrapper control failed: max rel 8.076e-03']; A2 reasons []
* criteria_s0: {"registry_version": "2.9.0", "per_artifact": {"analysis_s0.json": {"scope": "IN_SCOPE", "tier": "T1", "n_violations": 0, "n_work_items": 2}}}
* criteria_s1: {"registry_version": "2.9.0", "per_artifact": {"analysis_s1.json": {"scope": "IN_SCOPE", "tier": "T1", "n_violations": 0, "n_work_items": 2}}}
* echo_gate_s0 (echo_gate GATE 1 only; margins INHERITED from refcv5_compare): {"ha": {"n_slots": 4, "n_slots_passing": 1, "all_slots_pass": false, "relative_margin_by_slot": [0.14006, -0.11548, -0.0865, -0.0013], "separated_by_slot": [true, true, true, false]}, "ha0": {"n_slots": 4, "n_slots_passing": 4, "all_slots_pass": true, "relative_margin_by_slot": [0.57377, 0.5585, 0.54457, 0.53008], "separated_by_slot": [true, true, true, true]}, "ha0_ext": {"n_slots": 4, "n_slots_passing": 1, "all_slots_pass": false, "relative_margin_by_slot": [0.12448, -0.16541, -0.13811, -0.04365], "separated_by_slot": [true, true, true, false]}}
* echo_gate_s1 (echo_gate GATE 1 only; margins INHERITED from refcv5_compare): {"ha": {"n_slots": 4, "n_slots_passing": 1, "all_slots_pass": false, "relative_margin_by_slot": [0.13655, -0.11743, -0.08623, -0.0001], "separated_by_slot": [true, true, true, false]}, "ha0": {"n_slots": 4, "n_slots_passing": 4, "all_slots_pass": true, "relative_margin_by_slot": [0.57203, 0.55773, 0.54468, 0.53065], "separated_by_slot": [true, true, true, true]}, "ha0_ext": {"n_slots": 4, "n_slots_passing": 1, "all_slots_pass": false, "relative_margin_by_slot": [0.12091, -0.16745, -0.13782, -0.04239], "separated_by_slot": [true, true, true, false]}}
* panel_s0 pairing gates (g + ha/ha0/ha0_ext bit-identical vs every baseline): {'pass': True, 'failures': []}
* panel_s1 pairing gates (g + ha/ha0/ha0_ext bit-identical vs every baseline): {'pass': True, 'failures': []}
* roll_s0: 4754 windows / 139 eps, skipped 0, device cuda, wall 3444.6 s, peak 2.749 GiB, frame memo {'hits': 213100, 'misses': 47540}
* roll_s1: 4754 windows / 139 eps, skipped 0, device cuda, wall 3339.9 s, peak 2.749 GiB, frame memo {'hits': 213100, 'misses': 47540}
* void_gates_s0: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.4697, 'n_distinct_selected': 52, 'os_trivial_frac': 0.0042}
* void_gates_s1: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.4716, 'n_distinct_selected': 53, 'os_trivial_frac': 0.0044}
* inference-seed replicate (os seed A − seed B, same windows): {"max_abs_path_diff_m": 3.512540633790195, "ade": {"delta": 0.0001, "lo": -0.0027, "hi": 0.0027, "ci95": 0.0027, "p_delta_gt0": 0.5265, "separated": false, "reducer": "mean", "n_windows": 4754, "n_episodes": 139, "n_boot": 2000, "estimator": "paired_episode_cluster_bootstrap", "n_dropped_nonfinite": 0}}
* parent_cuda_initialized: False

### Four families + ADE, S2 (4754 windows / 139 episodes), inference seed 1

| arm | tier | ADE 0–2 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 0.3110 [0.2850, 0.3402] | 0.6624 | 0.2744 | 0.8385 | 0.2458 | 1.4904 | 0.004229 | 1.2465 | 0.1240 | 0.8199 | 0.5656 |
| refcv6 nav withheld | T1 | 0.3368 [0.3084, 0.3690] | 0.7154 | 0.3003 | 0.8140 | 0.2731 | 1.5527 | 0.004740 | 1.3136 | 0.1249 | 0.8023 | 0.5198 |
| refcv6 nav shuffled | T1 | 0.3149 [0.2893, 0.3450] | 0.6730 | 0.2766 | 0.8383 | 0.2486 | 1.5491 | 0.004684 | 1.3102 | 0.1261 | 0.7987 | 0.5538 |
| refcv6 nav flipped | T1 | 0.3192 [0.2913, 0.3515] | 0.6828 | 0.2756 | 0.8375 | 0.2470 | 1.7124 | 0.005954 | 1.5257 | 0.1329 | 0.7663 | 0.5662 |
| refcv6 max-speed withheld | T1 | 0.3092 [0.2838, 0.3390] | 0.6596 | 0.2733 | 0.8416 | 0.2453 | 1.4805 | 0.004203 | 1.2445 | 0.1227 | 0.8220 | 0.5726 |
| refcv4b `os` (banked) | T1 | 0.2974 [0.2689, 0.3303] | 0.6363 | 0.2908 | 0.8321 | 0.2553 | 1.3103 | 0.008266 | 1.7557 | 0.0979 | 0.8274 | 0.5215 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 0.3090 [0.2797, 0.3427] | 0.6629 | 0.2929 | 0.8338 | 0.2666 | 1.2241 | 0.003529 | 1.0619 | 0.0994 | 0.8190 | 0.5450 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 0.3091 [0.2793, 0.3429] | 0.6628 | 0.2931 | 0.8329 | 0.2666 | 1.2154 | 0.003533 | 1.0671 | 0.0998 | 0.8264 | 0.5334 |
| `ha` hold-action | T1 | 0.3010 [0.2761, 0.3298] | 0.6623 | 0.2548 | 0.8655 | 0.2354 | 1.5706 | 0.004089 | 1.4738 | 0.1238 | 0.7370 | 0.6066 |
| `ha0` constant velocity | T1 | 0.6764 [0.6033, 0.7529] | 1.4113 | 0.4891 | 0.7046 | 0.4718 | 2.8200 | 0.006903 | 2.4052 | 0.3170 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 0.2886 [0.2652, 0.3158] | 0.6355 | 0.2548 | 0.8655 | 0.2347 | 1.4522 | 0.003767 | 1.3575 | 0.1079 | 0.7544 | 0.6066 |
| `stop` STOP | T1 | 14.1180 [12.5982, 15.7166] | 22.5650 | 11.2911 | 0.0557 | 14.0984 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 0.3170 | 0.0000 | 0.0000 |
| `oracle_sel` (T0 ceiling) | T0 | 0.2434 [0.2264, 0.2623] | 0.4898 | 0.2022 | 0.9109 | 0.1843 | 0.9055 | 0.003137 | 1.0395 | 0.1096 | 0.8591 | 0.7291 |

### Distance keeping

lead block: `b1_eval_lead_block.npz`, status PRESENT, instants [1.0, 2.0] s

| arm | status | n with lead (eps) | min headway m [CI] | min time-gap s [CI] (n) | min TTC s [CI] | n closing (rest censored at 30 s) |
|---|---|---|---|---|---|---|
| **refcv6 `os`** | OK | 1313 (69) | 27.8386 [23.7086, 32.2592] | 4.0884 [3.2978, 5.1162] (1172) | 24.6236 [22.9053, 26.1206] | 514 |
| refcv6 nav withheld | OK | 1311 (69) | 27.7843 [23.6811, 32.2049] | 4.0931 [3.2904, 5.1291] (1170) | 24.8797 [23.1234, 26.3797] | 481 |
| refcv6 nav shuffled | OK | 1312 (70) | 27.9252 [23.9785, 32.2175] | 4.0394 [3.2669, 4.9921] (1171) | 24.6374 [22.9631, 26.0843] | 507 |
| refcv6 nav flipped | OK | 1309 (69) | 27.6505 [23.5412, 32.1219] | 4.0901 [3.2816, 5.1457] (1168) | 24.4354 [22.6920, 26.0024] | 523 |
| refcv6 max-speed withheld | OK | 1304 (69) | 27.7690 [23.6359, 32.2101] | 4.0713 [3.2880, 5.0955] (1163) | 24.5916 [22.8877, 26.0600] | 505 |
| refcv4b `os` (banked) | OK | 1224 (67) | 28.4517 [24.4003, 32.7611] | 4.1118 [3.2775, 5.0945] (1154) | 24.8434 [23.3648, 26.1655] | 470 |
| refcv5-v2 `os` seed 0 (banked) | OK | 1308 (68) | 27.8285 [23.9407, 32.4796] | 4.0885 [3.2646, 5.0234] (1167) | 24.6454 [23.2247, 26.0281] | 524 |
| refcv5-v2 `os` seed 1 (banked) | OK | 1307 (71) | 27.8319 [23.9316, 31.9794] | 4.0289 [3.2461, 5.0218] (1165) | 24.5477 [23.0821, 25.9259] | 541 |
| `ha` hold-action | OK | 1291 (67) | 28.0480 [23.8033, 32.3785] | 4.2073 [3.3867, 5.2420] (1150) | 25.5883 [24.2766, 26.7972] | 447 |
| `ha0` constant velocity | OK | 1315 (70) | 27.4550 [23.7229, 31.7446] | 4.1178 [3.2811, 5.1281] (1174) | 24.1077 [22.3616, 25.6733] | 499 |
| `ha0_ext` echo | OK | 1285 (67) | 28.0211 [23.7324, 32.3570] | 4.1966 [3.3733, 5.2300] (1144) | 25.5453 [24.2463, 26.7708] | 446 |
| `stop` STOP | OK | 1315 (70) | 38.6200 [33.3724, 44.1749] | 5.3059 [4.4617, 6.3288] (1174) | 29.9515 [29.8730, 30.0000] | 21 |
| `oracle_sel` (T0 ceiling) | OK | 1311 (70) | 27.9492 [24.0402, 32.2560] | 4.1532 [3.3474, 5.1151] (1170) | 24.8492 [23.3829, 26.1608] | 519 |

### Paired cells, S2, inference seed 1

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | 0.0224 [0.0070, 0.0387] **sep** | 0.0269 [-0.0119, 0.0668] ns | 0.0197 [0.0051, 0.0350] **sep** | 0.0111 [-0.0030, 0.0251] ns | 0.0490 [-0.0576, 0.1627] ns | -0.0018 [-0.0044, 0.0008] ns | 0.0161 [0.0065, 0.0262] **sep** | 0.0154 [0.0031, 0.0266] **sep** | -0.0116 [-0.0265, 0.0023] ns |
| os - ha | 0.0100 [-0.0066, 0.0272] ns | 0.0001 [-0.0423, 0.0429] ns | 0.0197 [0.0051, 0.0350] **sep** | 0.0103 [-0.0038, 0.0245] ns | -0.0684 [-0.1861, 0.0545] ns | -0.0038 [-0.0068, -0.0009] **sep** | 0.0002 [-0.0108, 0.0116] ns | 0.0191 [0.0065, 0.0309] **sep** | -0.0116 [-0.0265, 0.0023] ns |
| os - ha0 | -0.3654 [-0.4300, -0.3045] **sep** | -0.7489 [-0.8888, -0.6144] **sep** | -0.2147 [-0.2528, -0.1790] **sep** | -0.2260 [-0.2607, -0.1924] **sep** | -1.3178 [-1.6863, -0.9694] **sep** | -0.0202 [-0.0271, -0.0140] **sep** | -0.1930 [-0.2537, -0.1374] **sep** | 0.0926 [0.0606, 0.1249] **sep** | 0.0751 [0.0457, 0.1025] **sep** |
| os - stop | -13.8070 [-15.3307, -12.3036] **sep** | -21.9026 [-24.3478, -19.4791] **sep** | -11.0166 [-12.2479, -9.8039] **sep** | -13.8526 [-15.3750, -12.3449] **sep** | — | — | -0.1930 [-0.2537, -0.1374] **sep** | 0.0926 [0.0606, 0.1249] **sep** | 0.0751 [0.0457, 0.1025] **sep** |
| os - b_refcv4b | 0.0136 [-0.0028, 0.0313] ns | 0.0261 [-0.0092, 0.0656] ns | -0.0164 [-0.0360, 0.0027] ns | -0.0095 [-0.0288, 0.0095] ns | 0.1248 [0.0269, 0.2391] **sep** | -0.0099 [-0.0131, -0.0068] **sep** | 0.0261 [0.0156, 0.0373] **sep** | -0.0008 [-0.0092, 0.0076] ns | 0.0057 [-0.0091, 0.0202] ns |
| os - b_refcv5v2_s0 | 0.0020 [-0.0160, 0.0209] ns | -0.0005 [-0.0395, 0.0404] ns | -0.0185 [-0.0383, 0.0016] ns | -0.0209 [-0.0405, -0.0012] **sep** | 0.1974 [0.1092, 0.3047] **sep** | 0.0032 [0.0017, 0.0051] **sep** | 0.0246 [0.0147, 0.0360] **sep** | 0.0008 [-0.0063, 0.0082] ns | -0.0025 [-0.0187, 0.0122] ns |
| os - b_refcv5v2_s1 | 0.0019 [-0.0166, 0.0214] ns | -0.0004 [-0.0404, 0.0419] ns | -0.0186 [-0.0385, 0.0025] ns | -0.0209 [-0.0415, -0.0005] **sep** | 0.1971 [0.1083, 0.3050] **sep** | 0.0032 [0.0016, 0.0050] **sep** | 0.0242 [0.0142, 0.0356] **sep** | -0.0011 [-0.0084, 0.0063] ns | 0.0008 [-0.0162, 0.0166] ns |
| os_navzero - os | 0.0258 [0.0158, 0.0363] **sep** | 0.0530 [0.0311, 0.0752] **sep** | 0.0258 [0.0165, 0.0359] **sep** | 0.0273 [0.0183, 0.0369] **sep** | 0.0662 [-0.0118, 0.1517] ns | 0.0012 [-0.0003, 0.0030] ns | 0.0009 [-0.0046, 0.0065] ns | -0.0034 [-0.0090, 0.0017] ns | -0.0124 [-0.0236, -0.0025] **sep** |
| os_navshuf - os | 0.0039 [-0.0011, 0.0089] ns | 0.0106 [-0.0003, 0.0223] ns | 0.0021 [-0.0018, 0.0061] ns | 0.0029 [-0.0008, 0.0068] ns | 0.0590 [-0.0070, 0.1269] ns | 0.0012 [-0.0002, 0.0025] ns | 0.0021 [-0.0023, 0.0064] ns | -0.0046 [-0.0097, -0.0002] **sep** | -0.0053 [-0.0126, 0.0017] ns |
| os_vmaxzero - os | -0.0018 [-0.0049, 0.0010] ns | -0.0028 [-0.0092, 0.0028] ns | -0.0011 [-0.0038, 0.0014] ns | -0.0005 [-0.0032, 0.0021] ns | -0.0106 [-0.0337, 0.0056] ns | -0.0001 [-0.0005, 0.0002] ns | -0.0013 [-0.0034, 0.0003] ns | 0.0004 [-0.0017, 0.0023] ns | 0.0032 [-0.0017, 0.0082] ns |
| os_navzero - ha0_ext | 0.0482 [0.0302, 0.0663] **sep** | 0.0800 [0.0374, 0.1240] **sep** | 0.0455 [0.0279, 0.0653] **sep** | 0.0384 [0.0214, 0.0566] **sep** | 0.1131 [-0.0033, 0.2419] ns | -0.0006 [-0.0033, 0.0024] ns | 0.0170 [0.0079, 0.0266] **sep** | 0.0120 [0.0002, 0.0231] **sep** | -0.0240 [-0.0429, -0.0065] **sep** |
| b_refcv4b - ha0_ext | 0.0088 [-0.0058, 0.0249] ns | 0.0008 [-0.0316, 0.0369] ns | 0.0361 [0.0185, 0.0546] **sep** | 0.0206 [0.0035, 0.0384] **sep** | -0.0840 [-0.2002, 0.0265] ns | 0.0073 [0.0036, 0.0112] **sep** | -0.0100 [-0.0221, 0.0019] ns | 0.0162 [0.0065, 0.0255] **sep** | -0.0172 [-0.0315, -0.0036] **sep** |
| b_refcv5v2_s0 - ha0_ext | 0.0204 [0.0046, 0.0383] **sep** | 0.0275 [-0.0078, 0.0659] ns | 0.0381 [0.0194, 0.0579] **sep** | 0.0319 [0.0135, 0.0509] **sep** | -0.1504 [-0.2507, -0.0529] **sep** | -0.0051 [-0.0078, -0.0024] **sep** | -0.0085 [-0.0197, 0.0022] ns | 0.0145 [0.0038, 0.0248] **sep** | -0.0090 [-0.0231, 0.0044] ns |

n = 4754 windows / 139 episodes · estimator: paired episode-cluster bootstrap

### TACTICAL — declared heads and the 22-token goal set, inference seed 1

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

### S6 — the 6 s horizon, inference seed 1

| arm | tier | ADE 1–6 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 2.8007 [2.5677, 3.0416] | 6.9303 | 1.0255 | 0.4895 | 2.3644 | 3.5197 | 0.005446 | 1.7708 | 0.9367 | 0.6837 | 0.2602 |
| refcv6 nav withheld | T1 | 2.9677 [2.7152, 3.2293] | 7.2966 | 1.0600 | 0.4768 | 2.5077 | 3.9423 | 0.006045 | 1.9187 | 0.9852 | 0.6545 | 0.2387 |
| refcv6 nav shuffled | T1 | 2.9076 [2.6719, 3.1646] | 7.2435 | 1.0551 | 0.4812 | 2.4444 | 3.8830 | 0.006012 | 1.9115 | 0.9877 | 0.6480 | 0.2430 |
| refcv6 nav flipped | T1 | 2.9536 [2.6894, 3.2469] | 7.3457 | 1.0295 | 0.4900 | 2.3873 | 4.6007 | 0.007154 | 2.1599 | 1.0979 | 0.6117 | 0.2608 |
| refcv4b `os` (banked) | T1 | 2.7225 [2.4832, 2.9711] | 6.6964 | 0.9392 | 0.5373 | 2.2347 | 3.5207 | 0.005894 | 1.8225 | 0.9982 | 0.6790 | 0.3426 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 2.7140 [2.4798, 2.9674] | 6.6030 | 0.9257 | 0.5361 | 2.2110 | 3.5208 | 0.005141 | 1.7695 | 1.0255 | 0.6552 | 0.3475 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 2.7096 [2.4726, 2.9634] | 6.5888 | 0.9238 | 0.5381 | 2.1968 | 3.4947 | 0.005229 | 1.7852 | 1.0355 | 0.6557 | 0.3460 |
| `ha` hold-action | T1 | 3.6158 [3.2069, 4.0256] | 9.3416 | 1.0955 | 0.5251 | 2.6612 | 6.1561 | 0.006997 | 2.8433 | 1.7404 | 0.4887 | 0.3172 |
| `ha0` constant velocity | T1 | 4.9926 [4.3774, 5.6601] | 11.3583 | 1.2903 | 0.4470 | 3.4452 | 6.9005 | 0.006519 | 2.4079 | 2.4322 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 3.5479 [3.1510, 3.9432] | 9.2071 | 1.0957 | 0.5249 | 2.6460 | 5.9407 | 0.006850 | 2.7751 | 1.6680 | 0.4997 | 0.3172 |
| `stop` STOP | T1 | 39.2767 [35.0040, 43.7743] | 66.9735 | 11.2321 | 0.0498 | 38.8590 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 2.4322 | 0.0000 | 0.0000 |

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | -0.7472 [-1.0537, -0.4377] **sep** | -2.2768 [-3.1286, -1.4342] **sep** | -0.0701 [-0.1515, 0.0154] ns | -0.2817 [-0.5184, -0.0433] **sep** | -2.3251 [-3.1156, -1.6012] **sep** | -0.0175 [-0.0240, -0.0111] **sep** | -0.7313 [-0.9516, -0.5298] **sep** | 0.0867 [0.0641, 0.1099] **sep** | -0.0461 [-0.0963, 0.0003] ns |
| os - ha | -0.8151 [-1.1393, -0.4912] **sep** | -2.4113 [-3.2963, -1.5433] **sep** | -0.0700 [-0.1513, 0.0155] ns | -0.2969 [-0.5397, -0.0554] **sep** | -2.5387 [-3.3707, -1.7671] **sep** | -0.0188 [-0.0256, -0.0120] **sep** | -0.8036 [-1.0350, -0.5866] **sep** | 0.0911 [0.0676, 0.1147] **sep** | -0.0461 [-0.0963, 0.0003] ns |
| os - ha0 | -2.1919 [-2.7613, -1.6786] **sep** | -4.4280 [-5.7514, -3.2529] **sep** | -0.2648 [-0.3848, -0.1444] **sep** | -1.0809 [-1.4029, -0.7667] **sep** | -3.3374 [-4.4546, -2.3988] **sep** | -0.0116 [-0.0170, -0.0071] **sep** | -1.4955 [-2.0279, -1.0503] **sep** | 0.1598 [0.1163, 0.2053] **sep** | 0.0673 [-0.0071, 0.1378] ns |
| os - stop | -36.4760 [-40.8374, -32.1777] **sep** | -60.0432 [-67.5996, -52.6199] **sep** | -10.2066 [-11.4474, -8.9803] **sep** | -36.4947 [-40.8774, -32.1874] **sep** | — | — | -1.4955 [-2.0279, -1.0503] **sep** | 0.1598 [0.1163, 0.2053] **sep** | 0.0673 [-0.0071, 0.1378] ns |
| os - b_refcv4b | 0.0782 [-0.0706, 0.2431] ns | 0.2339 [-0.1453, 0.6577] ns | 0.0864 [0.0250, 0.1525] **sep** | 0.1296 [-0.0151, 0.2852] ns | -0.0751 [-0.3976, 0.2469] ns | -0.0012 [-0.0028, 0.0004] ns | -0.0615 [-0.1417, 0.0247] ns | 0.0038 [-0.0092, 0.0164] ns | -0.0886 [-0.1408, -0.0391] **sep** |
| os - b_refcv5v2_s0 | 0.0867 [-0.0734, 0.2604] ns | 0.3273 [-0.0886, 0.7731] ns | 0.0998 [0.0348, 0.1673] **sep** | 0.1534 [-0.0100, 0.3166] ns | -0.1168 [-0.3755, 0.1767] ns | -0.0001 [-0.0016, 0.0013] ns | -0.0887 [-0.1647, -0.0023] **sep** | 0.0123 [-0.0005, 0.0256] ns | -0.0932 [-0.1487, -0.0425] **sep** |
| os - b_refcv5v2_s1 | 0.0910 [-0.0739, 0.2662] ns | 0.3415 [-0.0805, 0.7833] ns | 0.1017 [0.0378, 0.1681] **sep** | 0.1675 [0.0045, 0.3331] **sep** | -0.1447 [-0.4097, 0.1445] ns | -0.0005 [-0.0020, 0.0009] ns | -0.0987 [-0.1746, -0.0113] **sep** | 0.0123 [-0.0005, 0.0255] ns | -0.0927 [-0.1475, -0.0416] **sep** |
| os_navzero - os | 0.1670 [0.0801, 0.2581] **sep** | 0.3664 [0.1465, 0.6008] **sep** | 0.0345 [0.0060, 0.0643] **sep** | 0.1434 [0.0723, 0.2185] **sep** | 0.4243 [0.1415, 0.7576] **sep** | 0.0028 [0.0012, 0.0047] **sep** | 0.0485 [-0.0060, 0.1114] ns | -0.0106 [-0.0196, -0.0019] **sep** | -0.0136 [-0.0338, 0.0049] ns |
| os_navshuf - os | 0.1069 [0.0359, 0.1840] **sep** | 0.3132 [0.1290, 0.5176] **sep** | 0.0296 [0.0105, 0.0495] **sep** | 0.0800 [0.0332, 0.1289] **sep** | 0.3858 [0.1238, 0.6756] **sep** | 0.0030 [0.0013, 0.0048] **sep** | 0.0510 [-0.0012, 0.1117] ns | -0.0134 [-0.0215, -0.0057] **sep** | -0.0128 [-0.0294, 0.0041] ns |
| vmaxzero_minus_os | ABSENT ['os_vmaxzero'] | | | | | | | | |
| os_navzero - ha0_ext | -0.5802 [-0.8899, -0.2802] **sep** | -1.9104 [-2.7579, -1.1045] **sep** | -0.0357 [-0.1200, 0.0505] ns | -0.1383 [-0.3702, 0.0940] ns | -1.9201 [-2.6369, -1.1810] **sep** | -0.0148 [-0.0211, -0.0088] **sep** | -0.6828 [-0.8890, -0.4891] **sep** | 0.0761 [0.0557, 0.0966] **sep** | -0.0597 [-0.1108, -0.0120] **sep** |
| b_refcv4b - ha0_ext | -0.8254 [-1.1031, -0.5708] **sep** | -2.5107 [-3.2872, -1.7973] **sep** | -0.1565 [-0.2206, -0.0953] **sep** | -0.4113 [-0.5881, -0.2356] **sep** | -2.3478 [-3.1344, -1.5806] **sep** | -0.0164 [-0.0235, -0.0096] **sep** | -0.6698 [-0.8890, -0.4591] **sep** | 0.0829 [0.0616, 0.1037] **sep** | 0.0425 [0.0071, 0.0754] **sep** |
| b_refcv5v2_s0 - ha0_ext | -0.8338 [-1.1206, -0.5756] **sep** | -2.6041 [-3.4076, -1.8950] **sep** | -0.1699 [-0.2342, -0.1075] **sep** | -0.4350 [-0.6193, -0.2611] **sep** | -2.2749 [-3.0421, -1.5259] **sep** | -0.0174 [-0.0242, -0.0108] **sep** | -0.6425 [-0.8494, -0.4412] **sep** | 0.0744 [0.0524, 0.0966] **sep** | 0.0472 [0.0109, 0.0826] **sep** |

n = 3668 windows / 139 episodes · estimator: paired episode-cluster bootstrap

### Frozen refcv6 acceptance instruments

```json
{
 "tflip": {
  "test": "T-FLIP",
  "bars": {
   "follows_FED_command": 0.5,
   "paired_true_minus_shuffled": 0.38
  },
  "today_refcv5_v2": {
   "follows_FED_command": 0.205,
   "paired_true_minus_shuffled": 0.099
  },
  "measured": {
   "follows_FED_command": {
    "mean": 0.2216,
    "ci": [
     0.1323,
     0.3107
    ],
    "n_windows": 352,
    "n_episodes": 29
   },
   "follows_TRUE_command": {
    "mean": 0.25,
    "ci": [
     0.1742,
     0.3243
    ]
   },
   "paired_true_minus_shuffled": {
    "delta": 0.142,
    "ci": [
     0.0968,
     0.1929
    ],
    "separated": true,
    "n_windows": 352
   }
  },
  "gates": {
   "follows_FED_lower_bound_clears_bar": false,
   "true_minus_shuffled_lower_bound_clears_bar_and_separated": false
  },
  "verdict": "FAIL",
  "reason": "the plan does not follow a flipped command: ['follows_FED_lower_bound_clears_bar', 'true_minus_shuffled_lower_bound_clears_bar_and_separated'] did not clear. The operative plan is still vision/ego-driven and only weakly steerable by nav \u2014 the refcv5-v2 reading (0.205).",
  "nav_compliance_status": null,
  "nav_compliance_reason": null
 },
 "obedience": {
  "test": "speed obedience",
  "forced_limit_kmh": 30.0,
  "population": "windows whose GT max speed exceeds 40.0 km/h",
  "n_windows": 2294,
  "n_episodes": 81,
  "bar_obeys_fraction": 0.99,
  "tol_ms": 0.0,
  "rows_with_no_compliant_candidate": 2232,
  "obeys": {
   "mean": 0.0126,
   "lo": 0.0022,
   "hi": 0.0264,
   "ci95": 0.0121,
   "se": 0.0063,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "episode_cluster_bootstrap"
  },
  "violation_ms": {
   "max": 28.26328913370768,
   "p95": 23.06548124949137,
   "mean_over_violators": 9.515196068029004,
   "n_violating": 2265
  },
  "ade_cost": {
   "delta": 0.0013,
   "lo": -0.0018,
   "hi": 0.0046,
   "ci95": 0.0032,
   "p_delta_gt0": 0.786,
   "separated": false,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "paired_episode_cluster_bootstrap",
   "_reads": "forced minus baseline, in metres. A model can obey ANY ceiling by planning to stop; this delta is what says it did not."
  },
  "verdict": "FAIL",
  "reason": "obedience lower bound 0.0022 < 0.99. 2265 of 2294 windows exceeded the forced ceiling; 2232 of them had no compliant candidate in the fan at all, which is a VOCABULARY limit, not a model one \u2014 separate the two before reading this as disobedience."
 },
 "tzero": {
  "test": "T-ZERO (non-turn diagnostic)",
  "classes": {
   "FOLLOW_LANE": {
    "n": 911,
    "recall_nav_true": 0.8386388583973655,
    "recall_nav_zero": 0.8507135016465422,
    "retained_fraction": 1.0143979057591623,
    "is_turn_class": false,
    "powered": true
   },
   "TURN_L": {
    "n": 40,
    "recall_nav_true": 0.8,
    "recall_nav_zero": 0.675,
    "retained_fraction": 0.84375,
    "is_turn_class": true,
    "powered": true,
    "_note": "EXCLUDED from the summary. De
```

### Bars

* **BAR-R6-1** — refcv6 beats the ECHO control: os - ha0_ext < 0, CI upper < 0 — **FAIL** — seed 0: 0.0225 [0.0068, 0.0395]; seed 1: 0.0224 [0.007, 0.0387]
* **BAR-R6-2** — refcv6 beats HOLD-ACTION: os - ha < 0, separated — **FAIL** — seed 0: 0.0101 [-0.0066, 0.028]; seed 1: 0.01 [-0.0066, 0.0272]
* **BAR-R6-3** — refcv6 beats refcv4b: os - refcv4b.os < 0, separated — **FAIL** — seed 0: 0.0137 [-0.0029, 0.0317]; seed 1: 0.0136 [-0.0028, 0.0313]
* **BAR-R6-4** — refcv6 beats refcv5-v2 (seed 0): os - refcv5v2.os < 0, separated — **FAIL** — seed 0: 0.0021 [-0.016, 0.0212]; seed 1: 0.002 [-0.016, 0.0209]
* **BAR-R6-5** — at 6 s: os - ha0_ext (ADE 0-6 s) < 0, separated — **PASS** — seed 0: -0.7475 [-1.0603, -0.4409]; seed 1: -0.7472 [-1.0537, -0.4377]
