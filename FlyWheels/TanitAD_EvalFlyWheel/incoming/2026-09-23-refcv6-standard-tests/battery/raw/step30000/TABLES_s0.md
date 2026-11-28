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

### Four families + ADE, S2 (4754 windows / 139 episodes), inference seed 0

| arm | tier | ADE 0–2 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 0.3111 [0.2847, 0.3409] | 0.6632 | 0.2740 | 0.8418 | 0.2456 | 1.4938 | 0.004240 | 1.2461 | 0.1243 | 0.8223 | 0.5663 |
| refcv6 nav withheld | T1 | 0.3377 [0.3096, 0.3699] | 0.7173 | 0.3017 | 0.8143 | 0.2744 | 1.5582 | 0.004823 | 1.3191 | 0.1251 | 0.8014 | 0.5315 |
| refcv6 nav shuffled | T1 | 0.3161 [0.2897, 0.3460] | 0.6745 | 0.2781 | 0.8361 | 0.2499 | 1.5487 | 0.004659 | 1.3188 | 0.1264 | 0.7970 | 0.5599 |
| refcv6 nav flipped | T1 | 0.3206 [0.2927, 0.3527] | 0.6856 | 0.2759 | 0.8380 | 0.2475 | 1.7192 | 0.005900 | 1.5230 | 0.1342 | 0.7638 | 0.5631 |
| refcv6 max-speed withheld | T1 | 0.3111 [0.2850, 0.3409] | 0.6626 | 0.2743 | 0.8424 | 0.2461 | 1.4952 | 0.004278 | 1.2527 | 0.1240 | 0.8214 | 0.5641 |
| refcv4b `os` (banked) | T1 | 0.2974 [0.2689, 0.3303] | 0.6363 | 0.2908 | 0.8321 | 0.2553 | 1.3103 | 0.008266 | 1.7557 | 0.0979 | 0.8274 | 0.5215 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 0.3090 [0.2797, 0.3427] | 0.6629 | 0.2929 | 0.8338 | 0.2666 | 1.2241 | 0.003529 | 1.0619 | 0.0994 | 0.8190 | 0.5450 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 0.3091 [0.2793, 0.3429] | 0.6628 | 0.2931 | 0.8329 | 0.2666 | 1.2154 | 0.003533 | 1.0671 | 0.0998 | 0.8264 | 0.5334 |
| `ha` hold-action | T1 | 0.3010 [0.2761, 0.3298] | 0.6623 | 0.2548 | 0.8655 | 0.2354 | 1.5706 | 0.004089 | 1.4738 | 0.1238 | 0.7370 | 0.6066 |
| `ha0` constant velocity | T1 | 0.6764 [0.6033, 0.7529] | 1.4113 | 0.4891 | 0.7046 | 0.4718 | 2.8200 | 0.006903 | 2.4052 | 0.3170 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 0.2886 [0.2652, 0.3158] | 0.6355 | 0.2548 | 0.8655 | 0.2347 | 1.4522 | 0.003767 | 1.3575 | 0.1079 | 0.7544 | 0.6066 |
| `stop` STOP | T1 | 14.1180 [12.5982, 15.7166] | 22.5650 | 11.2911 | 0.0557 | 14.0984 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 0.3170 | 0.0000 | 0.0000 |
| `oracle_sel` (T0 ceiling) | T0 | 0.2435 [0.2266, 0.2623] | 0.4900 | 0.2017 | 0.9107 | 0.1839 | 0.8948 | 0.003143 | 1.0432 | 0.1098 | 0.8581 | 0.7309 |

### Distance keeping

lead block: `b1_eval_lead_block.npz`, status PRESENT, instants [1.0, 2.0] s

| arm | status | n with lead (eps) | min headway m [CI] | min time-gap s [CI] (n) | min TTC s [CI] | n closing (rest censored at 30 s) |
|---|---|---|---|---|---|---|
| **refcv6 `os`** | OK | 1314 (69) | 27.8790 [23.8031, 32.3277] | 4.0746 [3.3073, 5.0658] (1173) | 24.6659 [22.9649, 26.1466] | 508 |
| refcv6 nav withheld | OK | 1306 (68) | 27.8516 [23.8309, 32.1673] | 4.1385 [3.2984, 5.1611] (1165) | 24.9775 [23.3242, 26.5557] | 474 |
| refcv6 nav shuffled | OK | 1312 (69) | 27.8775 [23.7657, 32.3338] | 4.0428 [3.2852, 5.0508] (1171) | 24.6565 [22.9401, 26.1751] | 511 |
| refcv6 nav flipped | OK | 1306 (70) | 27.6417 [23.7056, 31.9660] | 4.0696 [3.2690, 5.0279] (1165) | 24.6076 [22.9218, 26.0905] | 507 |
| refcv6 max-speed withheld | OK | 1304 (70) | 27.9460 [23.9915, 32.2928] | 4.1172 [3.3074, 5.0833] (1163) | 24.7069 [23.0508, 26.1266] | 503 |
| refcv4b `os` (banked) | OK | 1224 (67) | 28.4517 [24.4003, 32.7611] | 4.1118 [3.2775, 5.0945] (1154) | 24.8434 [23.3648, 26.1655] | 470 |
| refcv5-v2 `os` seed 0 (banked) | OK | 1308 (68) | 27.8285 [23.9407, 32.4796] | 4.0885 [3.2646, 5.0234] (1167) | 24.6454 [23.2247, 26.0281] | 524 |
| refcv5-v2 `os` seed 1 (banked) | OK | 1307 (71) | 27.8319 [23.9316, 31.9794] | 4.0289 [3.2461, 5.0218] (1165) | 24.5477 [23.0821, 25.9259] | 541 |
| `ha` hold-action | OK | 1291 (67) | 28.0480 [23.8033, 32.3785] | 4.2073 [3.3867, 5.2420] (1150) | 25.5883 [24.2766, 26.7972] | 447 |
| `ha0` constant velocity | OK | 1315 (70) | 27.4550 [23.7229, 31.7446] | 4.1178 [3.2811, 5.1281] (1174) | 24.1077 [22.3616, 25.6733] | 499 |
| `ha0_ext` echo | OK | 1285 (67) | 28.0211 [23.7324, 32.3570] | 4.1966 [3.3733, 5.2300] (1144) | 25.5453 [24.2463, 26.7708] | 446 |
| `stop` STOP | OK | 1315 (70) | 38.6200 [33.3724, 44.1749] | 5.3059 [4.4617, 6.3288] (1174) | 29.9515 [29.8730, 30.0000] | 21 |
| `oracle_sel` (T0 ceiling) | OK | 1311 (70) | 27.9034 [24.0025, 32.2111] | 4.1013 [3.3127, 5.0721] (1170) | 24.9039 [23.4558, 26.1997] | 511 |

### Paired cells, S2, inference seed 0

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | 0.0225 [0.0068, 0.0395] **sep** | 0.0277 [-0.0112, 0.0684] ns | 0.0192 [0.0041, 0.0346] **sep** | 0.0109 [-0.0032, 0.0249] ns | 0.0521 [-0.0555, 0.1668] ns | -0.0018 [-0.0044, 0.0009] ns | 0.0163 [0.0068, 0.0266] **sep** | 0.0158 [0.0036, 0.0269] **sep** | -0.0116 [-0.0274, 0.0036] ns |
| os - ha | 0.0101 [-0.0066, 0.0280] ns | 0.0009 [-0.0417, 0.0455] ns | 0.0192 [0.0041, 0.0346] **sep** | 0.0102 [-0.0039, 0.0243] ns | -0.0653 [-0.1828, 0.0586] ns | -0.0038 [-0.0068, -0.0008] **sep** | 0.0005 [-0.0109, 0.0120] ns | 0.0196 [0.0069, 0.0315] **sep** | -0.0116 [-0.0274, 0.0036] ns |
| os - ha0 | -0.3653 [-0.4294, -0.3049] **sep** | -0.7481 [-0.8885, -0.6137] **sep** | -0.2151 [-0.2529, -0.1792] **sep** | -0.2262 [-0.2608, -0.1932] **sep** | -1.3138 [-1.6865, -0.9638] **sep** | -0.0202 [-0.0271, -0.0140] **sep** | -0.1928 [-0.2536, -0.1370] **sep** | 0.0930 [0.0605, 0.1259] **sep** | 0.0751 [0.0441, 0.1040] **sep** |
| os - stop | -13.8069 [-15.3309, -12.3026] **sep** | -21.9018 [-24.3463, -19.4817] **sep** | -11.0171 [-12.2476, -9.8053] **sep** | -13.8528 [-15.3752, -12.3447] **sep** | — | — | -0.1928 [-0.2536, -0.1370] **sep** | 0.0930 [0.0605, 0.1259] **sep** | 0.0751 [0.0441, 0.1040] **sep** |
| os - b_refcv4b | 0.0137 [-0.0029, 0.0317] ns | 0.0269 [-0.0089, 0.0660] ns | -0.0169 [-0.0365, 0.0022] ns | -0.0097 [-0.0291, 0.0093] ns | 0.1266 [0.0269, 0.2433] **sep** | -0.0098 [-0.0131, -0.0068] **sep** | 0.0263 [0.0158, 0.0377] **sep** | -0.0004 [-0.0088, 0.0078] ns | 0.0057 [-0.0103, 0.0210] ns |
| os - b_refcv5v2_s0 | 0.0021 [-0.0160, 0.0212] ns | 0.0003 [-0.0392, 0.0420] ns | -0.0189 [-0.0389, 0.0016] ns | -0.0210 [-0.0408, -0.0012] **sep** | 0.1991 [0.1086, 0.3079] **sep** | 0.0032 [0.0017, 0.0051] **sep** | 0.0249 [0.0149, 0.0360] **sep** | 0.0013 [-0.0059, 0.0084] ns | -0.0025 [-0.0196, 0.0139] ns |
| os - b_refcv5v2_s1 | 0.0020 [-0.0167, 0.0216] ns | 0.0004 [-0.0405, 0.0431] ns | -0.0191 [-0.0392, 0.0025] ns | -0.0210 [-0.0413, -0.0003] **sep** | 0.1989 [0.1082, 0.3095] **sep** | 0.0031 [0.0016, 0.0050] **sep** | 0.0245 [0.0145, 0.0359] **sep** | -0.0006 [-0.0082, 0.0067] ns | 0.0008 [-0.0173, 0.0179] ns |
| os_navzero - os | 0.0266 [0.0167, 0.0361] **sep** | 0.0541 [0.0327, 0.0750] **sep** | 0.0278 [0.0184, 0.0375] **sep** | 0.0288 [0.0197, 0.0382] **sep** | 0.0668 [-0.0124, 0.1541] ns | 0.0014 [-0.0002, 0.0032] ns | 0.0008 [-0.0045, 0.0063] ns | -0.0042 [-0.0097, 0.0011] ns | -0.0097 [-0.0210, 0.0011] ns |
| os_navshuf - os | 0.0050 [0.0002, 0.0097] **sep** | 0.0113 [0.0007, 0.0223] **sep** | 0.0041 [0.0001, 0.0079] **sep** | 0.0043 [0.0005, 0.0080] **sep** | 0.0575 [-0.0036, 0.1196] ns | 0.0013 [0.0001, 0.0027] **sep** | 0.0021 [-0.0021, 0.0060] ns | -0.0053 [-0.0109, -0.0002] **sep** | -0.0042 [-0.0107, 0.0029] ns |
| os_vmaxzero - os | 0.0000 [-0.0028, 0.0030] ns | -0.0006 [-0.0066, 0.0059] ns | 0.0004 [-0.0029, 0.0039] ns | 0.0005 [-0.0025, 0.0037] ns | 0.0028 [-0.0061, 0.0135] ns | 0.0001 [-0.0001, 0.0004] ns | -0.0002 [-0.0013, 0.0008] ns | -0.0002 [-0.0023, 0.0019] ns | -0.0002 [-0.0065, 0.0059] ns |
| os_navzero - ha0_ext | 0.0491 [0.0312, 0.0672] **sep** | 0.0819 [0.0397, 0.1260] **sep** | 0.0470 [0.0295, 0.0673] **sep** | 0.0397 [0.0232, 0.0578] **sep** | 0.1185 [-0.0001, 0.2488] ns | -0.0005 [-0.0033, 0.0026] ns | 0.0171 [0.0079, 0.0270] **sep** | 0.0116 [-0.0011, 0.0236] ns | -0.0212 [-0.0400, -0.0040] **sep** |
| b_refcv4b - ha0_ext | 0.0088 [-0.0058, 0.0249] ns | 0.0008 [-0.0316, 0.0369] ns | 0.0361 [0.0185, 0.0546] **sep** | 0.0206 [0.0035, 0.0384] **sep** | -0.0840 [-0.2002, 0.0265] ns | 0.0073 [0.0036, 0.0112] **sep** | -0.0100 [-0.0221, 0.0019] ns | 0.0162 [0.0065, 0.0255] **sep** | -0.0172 [-0.0315, -0.0036] **sep** |
| b_refcv5v2_s0 - ha0_ext | 0.0204 [0.0046, 0.0383] **sep** | 0.0275 [-0.0078, 0.0659] ns | 0.0381 [0.0194, 0.0579] **sep** | 0.0319 [0.0135, 0.0509] **sep** | -0.1504 [-0.2507, -0.0529] **sep** | -0.0051 [-0.0078, -0.0024] **sep** | -0.0085 [-0.0197, 0.0022] ns | 0.0145 [0.0038, 0.0248] **sep** | -0.0090 [-0.0231, 0.0044] ns |

n = 4754 windows / 139 episodes · estimator: paired episode-cluster bootstrap

### TACTICAL — declared heads and the 22-token goal set, inference seed 0

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

### S6 — the 6 s horizon, inference seed 0

| arm | tier | ADE 1–6 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 2.8004 [2.5672, 3.0383] | 6.9317 | 1.0253 | 0.4886 | 2.3637 | 3.5293 | 0.005439 | 1.7738 | 0.9400 | 0.6775 | 0.2579 |
| refcv6 nav withheld | T1 | 2.9574 [2.7105, 3.2123] | 7.2589 | 1.0602 | 0.4752 | 2.5042 | 3.8845 | 0.006038 | 1.8950 | 0.9718 | 0.6616 | 0.2398 |
| refcv6 nav shuffled | T1 | 2.9017 [2.6645, 3.1543] | 7.2234 | 1.0521 | 0.4845 | 2.4362 | 3.8739 | 0.006032 | 1.9128 | 0.9897 | 0.6542 | 0.2442 |
| refcv6 nav flipped | T1 | 2.9501 [2.6855, 3.2377] | 7.3217 | 1.0255 | 0.4900 | 2.3837 | 4.5812 | 0.007113 | 2.1493 | 1.0946 | 0.6173 | 0.2690 |
| refcv4b `os` (banked) | T1 | 2.7225 [2.4832, 2.9711] | 6.6964 | 0.9392 | 0.5373 | 2.2347 | 3.5207 | 0.005894 | 1.8225 | 0.9982 | 0.6790 | 0.3426 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 2.7140 [2.4798, 2.9674] | 6.6030 | 0.9257 | 0.5361 | 2.2110 | 3.5208 | 0.005141 | 1.7695 | 1.0255 | 0.6552 | 0.3475 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 2.7096 [2.4726, 2.9634] | 6.5888 | 0.9238 | 0.5381 | 2.1968 | 3.4947 | 0.005229 | 1.7852 | 1.0355 | 0.6557 | 0.3460 |
| `ha` hold-action | T1 | 3.6158 [3.2069, 4.0256] | 9.3416 | 1.0955 | 0.5251 | 2.6612 | 6.1561 | 0.006997 | 2.8433 | 1.7404 | 0.4887 | 0.3172 |
| `ha0` constant velocity | T1 | 4.9926 [4.3774, 5.6601] | 11.3583 | 1.2903 | 0.4470 | 3.4452 | 6.9005 | 0.006519 | 2.4079 | 2.4322 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 3.5479 [3.1510, 3.9432] | 9.2071 | 1.0957 | 0.5249 | 2.6460 | 5.9407 | 0.006850 | 2.7751 | 1.6680 | 0.4997 | 0.3172 |
| `stop` STOP | T1 | 39.2767 [35.0040, 43.7743] | 66.9735 | 11.2321 | 0.0498 | 38.8590 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 2.4322 | 0.0000 | 0.0000 |

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | -0.7475 [-1.0603, -0.4409] **sep** | -2.2754 [-3.1353, -1.4529] **sep** | -0.0703 [-0.1526, 0.0132] ns | -0.2823 [-0.5187, -0.0471] **sep** | -2.3179 [-3.0856, -1.5886] **sep** | -0.0175 [-0.0240, -0.0110] **sep** | -0.7280 [-0.9428, -0.5239] **sep** | 0.0840 [0.0615, 0.1070] **sep** | -0.0477 [-0.0970, -0.0014] **sep** |
| os - ha | -0.8155 [-1.1417, -0.4958] **sep** | -2.4099 [-3.2964, -1.5662] **sep** | -0.0702 [-0.1523, 0.0133] ns | -0.2975 [-0.5401, -0.0575] **sep** | -2.5314 [-3.3402, -1.7558] **sep** | -0.0187 [-0.0255, -0.0119] **sep** | -0.8004 [-1.0311, -0.5827] **sep** | 0.0883 [0.0654, 0.1115] **sep** | -0.0477 [-0.0970, -0.0014] **sep** |
| os - ha0 | -2.1922 [-2.7513, -1.6910] **sep** | -4.4266 [-5.7397, -3.2527] **sep** | -0.2649 [-0.3837, -0.1470] **sep** | -1.0815 [-1.4026, -0.7694] **sep** | -3.3245 [-4.4450, -2.3847] **sep** | -0.0115 [-0.0170, -0.0071] **sep** | -1.4922 [-2.0234, -1.0483] **sep** | 0.1570 [0.1135, 0.2028] **sep** | 0.0657 [-0.0106, 0.1361] ns |
| os - stop | -36.4764 [-40.8452, -32.1726] **sep** | -60.0418 [-67.6039, -52.6125] **sep** | -10.2068 [-11.4489, -8.9761] **sep** | -36.4953 [-40.8778, -32.1876] **sep** | — | — | -1.4922 [-2.0234, -1.0483] **sep** | 0.1570 [0.1135, 0.2028] **sep** | 0.0657 [-0.0106, 0.1361] ns |
| os - b_refcv4b | 0.0779 [-0.0651, 0.2413] ns | 0.2353 [-0.1312, 0.6535] ns | 0.0862 [0.0270, 0.1488] **sep** | 0.1290 [-0.0129, 0.2784] ns | -0.0652 [-0.3903, 0.2695] ns | -0.0012 [-0.0028, 0.0004] ns | -0.0582 [-0.1373, 0.0258] ns | 0.0011 [-0.0125, 0.0142] ns | -0.0902 [-0.1425, -0.0404] **sep** |
| os - b_refcv5v2_s0 | 0.0863 [-0.0732, 0.2616] ns | 0.3287 [-0.0585, 0.7571] ns | 0.0996 [0.0356, 0.1675] **sep** | 0.1527 [-0.0042, 0.3108] ns | -0.1036 [-0.3624, 0.1915] ns | -0.0001 [-0.0016, 0.0012] ns | -0.0854 [-0.1595, -0.0017] **sep** | 0.0095 [-0.0038, 0.0240] ns | -0.0949 [-0.1500, -0.0444] **sep** |
| os - b_refcv5v2_s1 | 0.0907 [-0.0737, 0.2640] ns | 0.3429 [-0.0585, 0.7723] ns | 0.1015 [0.0379, 0.1671] **sep** | 0.1669 [0.0088, 0.3283] **sep** | -0.1304 [-0.3890, 0.1609] ns | -0.0005 [-0.0019, 0.0009] ns | -0.0954 [-0.1709, -0.0103] **sep** | 0.0095 [-0.0038, 0.0239] ns | -0.0943 [-0.1493, -0.0431] **sep** |
| os_navzero - os | 0.1570 [0.0749, 0.2454] **sep** | 0.3272 [0.1130, 0.5497] **sep** | 0.0349 [0.0058, 0.0647] **sep** | 0.1405 [0.0669, 0.2164] **sep** | 0.3579 [0.0892, 0.6721] **sep** | 0.0025 [0.0009, 0.0044] **sep** | 0.0318 [-0.0215, 0.0950] ns | -0.0052 [-0.0142, 0.0041] ns | -0.0115 [-0.0327, 0.0101] ns |
| os_navshuf - os | 0.1013 [0.0347, 0.1709] **sep** | 0.2917 [0.1173, 0.4777] **sep** | 0.0268 [0.0080, 0.0461] **sep** | 0.0725 [0.0268, 0.1173] **sep** | 0.3654 [0.1178, 0.6411] **sep** | 0.0030 [0.0014, 0.0048] **sep** | 0.0496 [-0.0014, 0.1090] ns | -0.0076 [-0.0166, 0.0014] ns | -0.0098 [-0.0284, 0.0093] ns |
| vmaxzero_minus_os | ABSENT ['os_vmaxzero'] | | | | | | | | |
| os_navzero - ha0_ext | -0.5905 [-0.9025, -0.2894] **sep** | -1.9482 [-2.7967, -1.1302] **sep** | -0.0354 [-0.1195, 0.0522] ns | -0.1418 [-0.3734, 0.0897] ns | -1.9614 [-2.6887, -1.2262] **sep** | -0.0151 [-0.0214, -0.0090] **sep** | -0.6962 [-0.9081, -0.4973] **sep** | 0.0788 [0.0572, 0.1007] **sep** | -0.0592 [-0.1088, -0.0122] **sep** |
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
    "mean": 0.2159,
    "ci": [
     0.1295,
     0.3037
    ],
    "n_windows": 352,
    "n_episodes": 29
   },
   "follows_TRUE_command": {
    "mean": 0.2472,
    "ci": [
     0.1734,
     0.3199
    ]
   },
   "paired_true_minus_shuffled": {
    "delta": 0.1562,
    "ci": [
     0.1054,
     0.2111
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
   "lo": 0.0026,
   "hi": 0.026,
   "ci95": 0.0117,
   "se": 0.0061,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "episode_cluster_bootstrap"
  },
  "violation_ms": {
   "max": 28.29799143473307,
   "p95": 23.062460009256995,
   "mean_over_violators": 9.509645703296787,
   "n_violating": 2265
  },
  "ade_cost": {
   "delta": 0.0011,
   "lo": -0.0021,
   "hi": 0.0043,
   "ci95": 0.0032,
   "p_delta_gt0": 0.756,
   "separated": false,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "paired_episode_cluster_bootstrap",
   "_reads": "forced minus baseline, in metres. A model can obey ANY ceiling by planning to stop; this delta is what says it did not."
  },
  "verdict": "FAIL",
  "reason": "obedience lower bound 0.0026 < 0.99. 2265 of 2294 windows exceeded the forced ceiling; 2232 of them had no compliant candidate in the fan at all, which is a VOCABULARY limit, not a model one \u2014 separate the two before reading this as disobedience."
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
    "_note": "EXCLUDED from the summary.
```

### Bars

* **BAR-R6-1** — refcv6 beats the ECHO control: os - ha0_ext < 0, CI upper < 0 — **FAIL** — seed 0: 0.0225 [0.0068, 0.0395]; seed 1: 0.0224 [0.007, 0.0387]
* **BAR-R6-2** — refcv6 beats HOLD-ACTION: os - ha < 0, separated — **FAIL** — seed 0: 0.0101 [-0.0066, 0.028]; seed 1: 0.01 [-0.0066, 0.0272]
* **BAR-R6-3** — refcv6 beats refcv4b: os - refcv4b.os < 0, separated — **FAIL** — seed 0: 0.0137 [-0.0029, 0.0317]; seed 1: 0.0136 [-0.0028, 0.0313]
* **BAR-R6-4** — refcv6 beats refcv5-v2 (seed 0): os - refcv5v2.os < 0, separated — **FAIL** — seed 0: 0.0021 [-0.016, 0.0212]; seed 1: 0.002 [-0.016, 0.0209]
* **BAR-R6-5** — at 6 s: os - ha0_ext (ADE 0-6 s) < 0, separated — **PASS** — seed 0: -0.7475 [-1.0603, -0.4409]; seed 1: -0.7472 [-1.0537, -0.4377]
