## step5000 — ckpt md5 `8a1e4da0dc6561353b28e3947146751e`, step 5000, SPEC sha256 `7bfca6cf2b81a754…`

**Tier:** every number is T1 = self-action OPEN loop (UNRULED for an action-free model) except `oracle_sel` (T0). Not closed loop, not driving. **Estimator:** full-set pooled means; paired episode-cluster bootstrap (n_boot 2000, seed 0, cluster = clip). One training seed: every separated cell answers the EPISODE question only; the INFERENCE question is answered by the seed replicate.

⛔ **Run-defect stamp (pre-switch checkpoint, step ≤ 34,500):** F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early (D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; audit 92337fa6). No weakness below may be attributed to the REGISTERED design while these hold. TACTICAL is read under BOTH label clocks (SPEC A5); this checkpoint's PRIMARY clock is the OLD one it was trained on. Physical-unit rates use dt = 0.5 s for a true 0.5033 s (row = 0.100667 s), identically for every arm and baseline.

### Gate

* G0 as registered: **FAIL**; G0-A1: **FAIL**; G0-A2 (operative): **PASS**; A2 wrapper max rel by condition: {'P1_as_run': 0.003920739304811635, 'P2_cudnn_det': 0.003920684765843788, 'P3_fp32_det': 1.8905077336620297e-06}; A1 reasons ['wrapper control failed: max rel 3.921e-03']; A2 reasons []
* criteria_s0: {"registry_version": "2.9.0", "per_artifact": {"analysis_s0.json": {"scope": "IN_SCOPE", "tier": "T1", "n_violations": 0, "n_work_items": 2}}}
* criteria_s1: {"registry_version": "2.9.0", "per_artifact": {"analysis_s1.json": {"scope": "IN_SCOPE", "tier": "T1", "n_violations": 0, "n_work_items": 2}}}
* echo_gate_s0 (echo_gate GATE 1 only; margins INHERITED from refcv5_compare): {"ha": {"n_slots": 4, "n_slots_passing": 0, "all_slots_pass": false, "relative_margin_by_slot": [-0.2278, -0.44118, -0.33173, -0.17184], "separated_by_slot": [true, true, true, true]}, "ha0": {"n_slots": 4, "n_slots_passing": 4, "all_slots_pass": true, "relative_margin_by_slot": [0.39144, 0.4296, 0.44178, 0.45005], "separated_by_slot": [true, true, true, true]}, "ha0_ext": {"n_slots": 4, "n_slots_passing": 0, "all_slots_pass": false, "relative_margin_by_slot": [-0.25004, -0.50568, -0.39498, -0.2214], "separated_by_slot": [true, true, true, true]}}
* echo_gate_s1 (echo_gate GATE 1 only; margins INHERITED from refcv5_compare): {"ha": {"n_slots": 4, "n_slots_passing": 0, "all_slots_pass": false, "relative_margin_by_slot": [-0.2323, -0.4425, -0.33126, -0.17116], "separated_by_slot": [true, true, true, true]}, "ha0": {"n_slots": 4, "n_slots_passing": 4, "all_slots_pass": true, "relative_margin_by_slot": [0.38921, 0.42907, 0.44197, 0.45037], "separated_by_slot": [true, true, true, true]}, "ha0_ext": {"n_slots": 4, "n_slots_passing": 0, "all_slots_pass": false, "relative_margin_by_slot": [-0.25462, -0.50707, -0.3945, -0.22069], "separated_by_slot": [true, true, true, true]}}
* panel_s0 pairing gates (g + ha/ha0/ha0_ext bit-identical vs every baseline): {'pass': True, 'failures': []}
* panel_s1 pairing gates (g + ha/ha0/ha0_ext bit-identical vs every baseline): {'pass': True, 'failures': []}
* roll_s0: 4754 windows / 139 eps, skipped 0, device cuda, wall 6106.3 s, peak 2.749 GiB, frame memo {'hits': 213100, 'misses': 47540}
* roll_s1: 4754 windows / 139 eps, skipped 0, device cuda, wall 4931.0 s, peak 2.749 GiB, frame memo {'hits': 213100, 'misses': 47540}
* void_gates_s0: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.5191, 'n_distinct_selected': 45, 'os_trivial_frac': 0.0189}
* void_gates_s1: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.5187, 'n_distinct_selected': 46, 'os_trivial_frac': 0.0189}
* inference-seed replicate (os seed A − seed B, same windows): {"max_abs_path_diff_m": 3.329801082611084, "ade": {"delta": 0.0, "lo": -0.0025, "hi": 0.0027, "ci95": 0.0026, "p_delta_gt0": 0.516, "separated": false, "reducer": "mean", "n_windows": 4754, "n_episodes": 139, "n_boot": 2000, "estimator": "paired_episode_cluster_bootstrap", "n_dropped_nonfinite": 0}}
* parent_cuda_initialized: False

### Four families + ADE, S2 (4754 windows / 139 episodes), inference seed 0

| arm | tier | ADE 0–2 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 0.3771 [0.3458, 0.4138] | 0.7762 | 0.3043 | 0.8101 | 0.2833 | 1.8623 | 0.005321 | 1.5649 | 0.1668 | 0.7184 | 0.5309 |
| refcv6 nav withheld | T1 | 0.4017 [0.3688, 0.4400] | 0.8223 | 0.3285 | 0.7900 | 0.3104 | 1.8942 | 0.005655 | 1.5803 | 0.1686 | 0.7183 | 0.4994 |
| refcv6 nav shuffled | T1 | 0.3871 [0.3540, 0.4272] | 0.7979 | 0.3118 | 0.8032 | 0.2907 | 1.9362 | 0.005914 | 1.6458 | 0.1709 | 0.7051 | 0.5237 |
| refcv6 nav flipped | T1 | 0.3925 [0.3585, 0.4337] | 0.8117 | 0.3051 | 0.8102 | 0.2847 | 2.1721 | 0.007459 | 1.9234 | 0.1824 | 0.6532 | 0.5300 |
| refcv6 max-speed withheld | T1 | 0.3792 [0.3479, 0.4156] | 0.7800 | 0.3056 | 0.8086 | 0.2846 | 1.8699 | 0.005344 | 1.5745 | 0.1679 | 0.7207 | 0.5324 |
| refcv4b `os` (banked) | T1 | 0.2974 [0.2689, 0.3303] | 0.6363 | 0.2908 | 0.8321 | 0.2553 | 1.3103 | 0.008266 | 1.7557 | 0.0979 | 0.8274 | 0.5215 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 0.3090 [0.2797, 0.3427] | 0.6629 | 0.2929 | 0.8338 | 0.2666 | 1.2241 | 0.003529 | 1.0619 | 0.0994 | 0.8190 | 0.5450 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 0.3091 [0.2793, 0.3429] | 0.6628 | 0.2931 | 0.8329 | 0.2666 | 1.2154 | 0.003533 | 1.0671 | 0.0998 | 0.8264 | 0.5334 |
| `ha` hold-action | T1 | 0.3010 [0.2761, 0.3298] | 0.6623 | 0.2548 | 0.8655 | 0.2354 | 1.5706 | 0.004089 | 1.4738 | 0.1238 | 0.7370 | 0.6066 |
| `ha0` constant velocity | T1 | 0.6764 [0.6033, 0.7529] | 1.4113 | 0.4891 | 0.7046 | 0.4718 | 2.8200 | 0.006903 | 2.4052 | 0.3170 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 0.2886 [0.2652, 0.3158] | 0.6355 | 0.2548 | 0.8655 | 0.2347 | 1.4522 | 0.003767 | 1.3575 | 0.1079 | 0.7544 | 0.6066 |
| `stop` STOP | T1 | 14.1180 [12.5982, 15.7166] | 22.5650 | 11.2911 | 0.0557 | 14.0984 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 0.3170 | 0.0000 | 0.0000 |
| `oracle_sel` (T0 ceiling) | T0 | 0.3164 [0.2878, 0.3517] | 0.6201 | 0.2400 | 0.8841 | 0.2319 | 1.3521 | 0.003838 | 1.2155 | 0.1489 | 0.8077 | 0.6950 |

### Distance keeping

lead block: `b1_eval_lead_block.npz`, status PRESENT, instants [1.0, 2.0] s

| arm | status | n with lead (eps) | min headway m [CI] | min time-gap s [CI] (n) | min TTC s [CI] | n closing (rest censored at 30 s) |
|---|---|---|---|---|---|---|
| **refcv6 `os`** | OK | 1317 (69) | 27.6045 [23.5467, 31.9670] | 4.1242 [3.3039, 5.1812] (1176) | 24.0720 [22.3495, 25.5676] | 515 |
| refcv6 nav withheld | OK | 1324 (69) | 27.6290 [23.5783, 31.9496] | 4.1282 [3.3049, 5.1872] (1183) | 24.4452 [22.7535, 25.9688] | 490 |
| refcv6 nav shuffled | OK | 1328 (69) | 27.7203 [23.5750, 32.0894] | 4.1303 [3.3119, 5.1900] (1187) | 24.1363 [22.3592, 25.6991] | 514 |
| refcv6 nav flipped | OK | 1324 (69) | 27.5532 [23.5144, 31.8837] | 4.0855 [3.2796, 5.1235] (1183) | 24.0540 [22.3033, 25.5720] | 520 |
| refcv6 max-speed withheld | OK | 1320 (68) | 27.6536 [23.8710, 32.0282] | 4.1068 [3.2758, 5.1405] (1179) | 24.1271 [22.5181, 25.6914] | 516 |
| refcv4b `os` (banked) | OK | 1224 (67) | 28.4517 [24.4003, 32.7611] | 4.1118 [3.2775, 5.0945] (1154) | 24.8434 [23.3648, 26.1655] | 470 |
| refcv5-v2 `os` seed 0 (banked) | OK | 1308 (68) | 27.8285 [23.9407, 32.4796] | 4.0885 [3.2646, 5.0234] (1167) | 24.6454 [23.2247, 26.0281] | 524 |
| refcv5-v2 `os` seed 1 (banked) | OK | 1307 (71) | 27.8319 [23.9316, 31.9794] | 4.0289 [3.2461, 5.0218] (1165) | 24.5477 [23.0821, 25.9259] | 541 |
| `ha` hold-action | OK | 1291 (67) | 28.0480 [23.8033, 32.3785] | 4.2073 [3.3867, 5.2420] (1150) | 25.5883 [24.2766, 26.7972] | 447 |
| `ha0` constant velocity | OK | 1315 (70) | 27.4550 [23.7229, 31.7446] | 4.1178 [3.2811, 5.1281] (1174) | 24.1077 [22.3616, 25.6733] | 499 |
| `ha0_ext` echo | OK | 1285 (67) | 28.0211 [23.7324, 32.3570] | 4.1966 [3.3733, 5.2300] (1144) | 25.5453 [24.2463, 26.7708] | 446 |
| `stop` STOP | OK | 1315 (70) | 38.6200 [33.3724, 44.1749] | 5.3059 [4.4617, 6.3288] (1174) | 29.9515 [29.8730, 30.0000] | 21 |
| `oracle_sel` (T0 ceiling) | OK | 1327 (70) | 27.5702 [23.7201, 31.7699] | 4.0343 [3.2347, 4.9692] (1186) | 24.5335 [23.0731, 25.8609] | 509 |

### Paired cells, S2, inference seed 0

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | 0.0885 [0.0695, 0.1094] **sep** | 0.1407 [0.0988, 0.1875] **sep** | 0.0495 [0.0312, 0.0687] **sep** | 0.0486 [0.0303, 0.0674] **sep** | 0.4205 [0.2980, 0.5602] **sep** | 0.0039 [0.0012, 0.0067] **sep** | 0.0588 [0.0466, 0.0732] **sep** | -0.0076 [-0.0209, 0.0053] ns | -0.0187 [-0.0318, -0.0057] **sep** |
| os - ha | 0.0761 [0.0569, 0.0974] **sep** | 0.1138 [0.0692, 0.1626] **sep** | 0.0495 [0.0312, 0.0687] **sep** | 0.0479 [0.0295, 0.0669] **sep** | 0.3030 [0.1788, 0.4454] **sep** | 0.0019 [-0.0011, 0.0049] ns | 0.0430 [0.0312, 0.0568] **sep** | -0.0038 [-0.0172, 0.0093] ns | -0.0187 [-0.0318, -0.0057] **sep** |
| os - ha0 | -0.2993 [-0.3582, -0.2419] **sep** | -0.6352 [-0.7630, -0.5109] **sep** | -0.1849 [-0.2227, -0.1496] **sep** | -0.1885 [-0.2246, -0.1543] **sep** | -0.9443 [-1.2634, -0.6420] **sep** | -0.0145 [-0.0204, -0.0091] **sep** | -0.1503 [-0.2049, -0.1001] **sep** | 0.0696 [0.0408, 0.1014] **sep** | 0.0679 [0.0427, 0.0936] **sep** |
| os - stop | -13.7409 [-15.2651, -12.2370] **sep** | -21.7888 [-24.2345, -19.3720] **sep** | -10.9868 [-12.2092, -9.7790] **sep** | -13.8151 [-15.3359, -12.3103] **sep** | — | — | -0.1503 [-0.2049, -0.1001] **sep** | 0.0696 [0.0408, 0.1014] **sep** | 0.0679 [0.0427, 0.0936] **sep** |
| os - b_refcv4b | 0.0797 [0.0574, 0.1017] **sep** | 0.1399 [0.0948, 0.1854] **sep** | 0.0134 [-0.0107, 0.0368] ns | 0.0280 [0.0023, 0.0531] **sep** | 0.4982 [0.3504, 0.6498] **sep** | -0.0042 [-0.0075, -0.0011] **sep** | 0.0689 [0.0524, 0.0869] **sep** | -0.0238 [-0.0357, -0.0128] **sep** | -0.0015 [-0.0154, 0.0122] ns |
| os - b_refcv5v2_s0 | 0.0681 [0.0456, 0.0909] **sep** | 0.1132 [0.0672, 0.1597] **sep** | 0.0114 [-0.0125, 0.0349] ns | 0.0167 [-0.0085, 0.0412] ns | 0.5699 [0.4350, 0.7136] **sep** | 0.0089 [0.0067, 0.0113] **sep** | 0.0674 [0.0519, 0.0850] **sep** | -0.0221 [-0.0336, -0.0105] **sep** | -0.0097 [-0.0238, 0.0048] ns |
| os - b_refcv5v2_s1 | 0.0680 [0.0457, 0.0916] **sep** | 0.1134 [0.0671, 0.1609] **sep** | 0.0112 [-0.0124, 0.0352] ns | 0.0167 [-0.0087, 0.0421] ns | 0.5702 [0.4311, 0.7172] **sep** | 0.0089 [0.0065, 0.0113] **sep** | 0.0670 [0.0511, 0.0851] **sep** | -0.0240 [-0.0356, -0.0122] **sep** | -0.0063 [-0.0206, 0.0082] ns |
| os_navzero - os | 0.0246 [0.0157, 0.0340] **sep** | 0.0461 [0.0272, 0.0661] **sep** | 0.0242 [0.0154, 0.0335] **sep** | 0.0271 [0.0186, 0.0361] **sep** | 0.0350 [-0.0240, 0.0972] ns | 0.0004 [-0.0007, 0.0017] ns | 0.0018 [-0.0022, 0.0056] ns | 0.0025 [-0.0038, 0.0089] ns | -0.0006 [-0.0093, 0.0084] ns |
| os_navshuf - os | 0.0100 [0.0042, 0.0166] **sep** | 0.0217 [0.0087, 0.0368] **sep** | 0.0076 [0.0025, 0.0126] **sep** | 0.0074 [0.0027, 0.0122] **sep** | 0.0737 [0.0095, 0.1489] **sep** | 0.0014 [0.0002, 0.0028] **sep** | 0.0042 [-0.0004, 0.0092] ns | -0.0017 [-0.0067, 0.0034] ns | -0.0004 [-0.0076, 0.0065] ns |
| os_vmaxzero - os | 0.0020 [-0.0010, 0.0049] ns | 0.0039 [-0.0026, 0.0101] ns | 0.0013 [-0.0016, 0.0041] ns | 0.0013 [-0.0015, 0.0040] ns | 0.0076 [-0.0121, 0.0310] ns | 0.0001 [-0.0002, 0.0005] ns | 0.0011 [-0.0008, 0.0030] ns | 0.0006 [-0.0023, 0.0034] ns | 0.0023 [-0.0032, 0.0082] ns |
| os_navzero - ha0_ext | 0.1131 [0.0916, 0.1380] **sep** | 0.1868 [0.1387, 0.2426] **sep** | 0.0737 [0.0517, 0.0970] **sep** | 0.0757 [0.0539, 0.0989] **sep** | 0.4536 [0.3197, 0.6011] **sep** | 0.0041 [0.0013, 0.0072] **sep** | 0.0607 [0.0488, 0.0742] **sep** | -0.0050 [-0.0167, 0.0067] ns | -0.0194 [-0.0337, -0.0063] **sep** |
| b_refcv4b - ha0_ext | 0.0088 [-0.0058, 0.0249] ns | 0.0008 [-0.0316, 0.0369] ns | 0.0361 [0.0185, 0.0546] **sep** | 0.0206 [0.0035, 0.0384] **sep** | -0.0840 [-0.2002, 0.0265] ns | 0.0073 [0.0036, 0.0112] **sep** | -0.0100 [-0.0221, 0.0019] ns | 0.0162 [0.0065, 0.0255] **sep** | -0.0172 [-0.0315, -0.0036] **sep** |
| b_refcv5v2_s0 - ha0_ext | 0.0204 [0.0046, 0.0383] **sep** | 0.0275 [-0.0078, 0.0659] ns | 0.0381 [0.0194, 0.0579] **sep** | 0.0319 [0.0135, 0.0509] **sep** | -0.1504 [-0.2507, -0.0529] **sep** | -0.0051 [-0.0078, -0.0024] **sep** | -0.0085 [-0.0197, 0.0022] ns | 0.0145 [0.0038, 0.0248] **sep** | -0.0090 [-0.0231, 0.0044] ns |

n = 4754 windows / 139 episodes · estimator: paired episode-cluster bootstrap

### TACTICAL — declared heads and the 22-token goal set, inference seed 0

**LAT** (v8 labels, in-band windows n = 1141 / 139 eps; classes ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'ABORT_LC', 'NUDGE_L', 'NUDGE_R', 'TURN_L', 'TURN_R'])

| surface | acc [CI] | κ | majority-class rate |
|---|---|---|---|
| v6_behaviour_decoder | 0.7169 [0.6427, 0.7857] | 0.2418 | 0.6713 |
| z_tac_v7_heads | 0.7038 [0.6279, 0.7759] | 0.1975 | 0.6713 |
| v6_behaviour_decoder_NAVZERO | 0.6713 [0.5925, 0.7483] | 0.0066 | 0.6713 |

**LON** (v8 labels, in-band windows n = 1141 / 139 eps; classes ['FOLLOW', 'CRUISE', 'YIELD_MERGE', 'BRAKE_TO', 'CREEP', 'HOLD', 'ADAPT_SPEED_FOR_CURVE', 'ACCELERATE'])

| surface | acc [CI] | κ | majority-class rate |
|---|---|---|---|
| v6_behaviour_decoder | 0.3707 [0.3003, 0.4432] | 0.1821 | 0.3024 |
| z_tac_v7_heads | 0.3883 [0.3173, 0.4618] | 0.2131 | 0.3024 |
| v6_behaviour_decoder_NAVZERO | 0.3655 [0.2948, 0.4378] | 0.1790 | 0.3024 |

**22-token goal selection** (per class, nav true; floor n_pos 200)

| token | n_pos / n_neg | eps | AUROC | AP | prevalence | P@0.5 | R@0.5 | AUROC nav-zero | status |
|---|---|---|---|---|---|---|---|---|---|
| FOLLOW_LANE | 911 / 230 | 139 | 0.7372 | 0.9066 | 0.7984 | 0.9234 | 0.4632 | 0.7641 | SCOREABLE |
| TURN_L | 40 / 1101 | 139 | 0.9751 | 0.8408 | 0.0351 | 0.3368 | 0.8000 | 0.8627 | UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 1074 | 139 | 0.9571 | 0.4805 | 0.0587 | 0.2683 | 0.9851 | 0.8035 | UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 1125 | 139 | 0.9939 | 0.5196 | 0.0140 | 0.5385 | 0.4375 | 0.8766 | UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / 1141 | 139 | — | — | 0.0000 | 0.0000 | — | — | UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / 0 | 17 | — | 1.0000 | 1.0000 | 1.0000 | 0.1007 | — | UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 1067 | 139 | 0.8648 | 0.3212 | 0.0649 | 0.2333 | 0.7568 | 0.8654 | UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / 0 | 139 | — | 1.0000 | 1.0000 | 1.0000 | 0.0482 | — | SCOREABLE |
| CORRIDOR_OFFSET | 200 / 0 | 24 | — | 1.0000 | 1.0000 | 1.0000 | 0.0250 | — | SCOREABLE |
| EVADE_IN_CORRIDOR | 49 / 9 | 7 | 0.0952 | 0.6949 | 0.8448 | — | 0.0000 | 0.2653 | UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 911 | 112 | 0.6914 | 0.0163 | 0.0098 | — | 0.0000 | 0.3430 | UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 911 | 116 | 0.4499 | 0.0386 | 0.0421 | — | 0.0000 | 0.5538 | UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / 0 | 12 | — | 1.0000 | 1.0000 | 1.0000 | 0.0515 | — | UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / 0 | 10 | — | 1.0000 | 1.0000 | — | 0.0000 | — | UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / 108 | 13 | — | — | 0.0000 | — | — | — | UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 40 | 10 | 1.0000 | 1.0000 | 0.5062 | 1.0000 | 0.8049 | 0.9634 | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / 212 | 26 | — | — | 0.0000 | 0.0000 | — | — | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 147 | 26 | 0.5646 | 0.3838 | 0.3066 | 0.0000 | 0.0000 | 0.3751 | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 187 | 26 | 0.6526 | 0.2281 | 0.1179 | 0.0000 | 0.0000 | 0.6584 | UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 90 | 26 | 0.7104 | 0.6673 | 0.5755 | 0.3750 | 0.0738 | 0.4648 | UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / 1141 | 139 | — | — | 0.0000 | — | — | — | UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 67 | 9 | 1.0000 | 1.0000 | 0.1067 | 1.0000 | 1.0000 | 1.0000 | UNSCOREABLE (n_pos < 200) |

### S6 — the 6 s horizon, inference seed 0

| arm | tier | ADE 1–6 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 3.0749 [2.7977, 3.3745] | 7.4484 | 1.0141 | 0.4906 | 2.4568 | 4.6396 | 0.006954 | 2.3039 | 1.2344 | 0.6259 | 0.2931 |
| refcv6 nav withheld | T1 | 3.1878 [2.9048, 3.4928] | 7.6681 | 1.0486 | 0.4761 | 2.5798 | 4.7300 | 0.007398 | 2.3463 | 1.2447 | 0.5770 | 0.2922 |
| refcv6 nav shuffled | T1 | 3.2164 [2.9346, 3.5294] | 7.8137 | 1.0583 | 0.4766 | 2.5561 | 4.8805 | 0.007410 | 2.3789 | 1.3093 | 0.5867 | 0.2611 |
| refcv6 nav flipped | T1 | 3.2413 [2.9382, 3.5868] | 7.8492 | 1.0262 | 0.4890 | 2.4872 | 5.5222 | 0.008374 | 2.5721 | 1.4071 | 0.5377 | 0.2888 |
| refcv4b `os` (banked) | T1 | 2.7225 [2.4832, 2.9711] | 6.6964 | 0.9392 | 0.5373 | 2.2347 | 3.5207 | 0.005894 | 1.8225 | 0.9982 | 0.6790 | 0.3426 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 2.7140 [2.4798, 2.9674] | 6.6030 | 0.9257 | 0.5361 | 2.2110 | 3.5208 | 0.005141 | 1.7695 | 1.0255 | 0.6552 | 0.3475 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 2.7096 [2.4726, 2.9634] | 6.5888 | 0.9238 | 0.5381 | 2.1968 | 3.4947 | 0.005229 | 1.7852 | 1.0355 | 0.6557 | 0.3460 |
| `ha` hold-action | T1 | 3.6158 [3.2069, 4.0256] | 9.3416 | 1.0955 | 0.5251 | 2.6612 | 6.1561 | 0.006997 | 2.8433 | 1.7404 | 0.4887 | 0.3172 |
| `ha0` constant velocity | T1 | 4.9926 [4.3774, 5.6601] | 11.3583 | 1.2903 | 0.4470 | 3.4452 | 6.9005 | 0.006519 | 2.4079 | 2.4322 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 3.5479 [3.1510, 3.9432] | 9.2071 | 1.0957 | 0.5249 | 2.6460 | 5.9407 | 0.006850 | 2.7751 | 1.6680 | 0.4997 | 0.3172 |
| `stop` STOP | T1 | 39.2767 [35.0040, 43.7743] | 66.9735 | 11.2321 | 0.0498 | 38.8590 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 2.4322 | 0.0000 | 0.0000 |

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | -0.4729 [-0.7531, -0.1780] **sep** | -1.7586 [-2.5222, -0.9770] **sep** | -0.0816 [-0.1588, -0.0023] **sep** | -0.1892 [-0.4134, 0.0312] ns | -1.2023 [-1.8378, -0.5796] **sep** | -0.0081 [-0.0138, -0.0024] **sep** | -0.4336 [-0.6304, -0.2388] **sep** | 0.0676 [0.0464, 0.0891] **sep** | 0.0030 [-0.0332, 0.0381] ns |
| os - ha | -0.5409 [-0.8350, -0.2366] **sep** | -1.8931 [-2.6867, -1.0966] **sep** | -0.0814 [-0.1587, -0.0022] **sep** | -0.2044 [-0.4309, 0.0186] ns | -1.4161 [-2.0937, -0.7533] **sep** | -0.0093 [-0.0153, -0.0033] **sep** | -0.5060 [-0.7160, -0.2993] **sep** | 0.0720 [0.0501, 0.0944] **sep** | 0.0030 [-0.0332, 0.0381] ns |
| os - ha0 | -1.9177 [-2.3941, -1.4777] **sep** | -3.9098 [-4.9995, -2.9262] **sep** | -0.2762 [-0.3745, -0.1789] **sep** | -0.9885 [-1.2594, -0.7223] **sep** | -2.2007 [-3.1222, -1.4155] **sep** | -0.0019 [-0.0066, 0.0021] ns | -1.1979 [-1.6762, -0.7962] **sep** | 0.1407 [0.1022, 0.1815] **sep** | 0.1164 [0.0670, 0.1633] **sep** |
| os - stop | -36.2018 [-40.5613, -31.9050] **sep** | -59.5251 [-67.0227, -52.2409] **sep** | -10.2180 [-11.4533, -8.9830] **sep** | -36.4023 [-40.7965, -32.1370] **sep** | — | — | -1.1979 [-1.6762, -0.7962] **sep** | 0.1407 [0.1022, 0.1815] **sep** | 0.1164 [0.0670, 0.1633] **sep** |
| os - b_refcv4b | 0.3524 [0.1955, 0.5140] **sep** | 0.7520 [0.3818, 1.1361] **sep** | 0.0749 [0.0209, 0.1302] **sep** | 0.2221 [0.0801, 0.3655] **sep** | 1.0828 [0.6919, 1.5082] **sep** | 0.0085 [0.0059, 0.0114] **sep** | 0.2361 [0.1250, 0.3561] **sep** | -0.0153 [-0.0303, -0.0014] **sep** | -0.0395 [-0.0695, -0.0114] **sep** |
| os - b_refcv5v2_s0 | 0.3609 [0.2066, 0.5231] **sep** | 0.8455 [0.4835, 1.2295] **sep** | 0.0883 [0.0369, 0.1427] **sep** | 0.2458 [0.0981, 0.3902] **sep** | 1.0382 [0.6887, 1.4049] **sep** | 0.0094 [0.0071, 0.0119] **sep** | 0.2089 [0.1039, 0.3323] **sep** | -0.0068 [-0.0194, 0.0057] ns | -0.0442 [-0.0739, -0.0153] **sep** |
| os - b_refcv5v2_s1 | 0.3653 [0.2128, 0.5269] **sep** | 0.8597 [0.5027, 1.2398] **sep** | 0.0903 [0.0380, 0.1434] **sep** | 0.2600 [0.1134, 0.4033] **sep** | 1.0259 [0.6607, 1.4040] **sep** | 0.0092 [0.0069, 0.0115] **sep** | 0.1989 [0.0972, 0.3210] **sep** | -0.0068 [-0.0193, 0.0060] ns | -0.0436 [-0.0750, -0.0142] **sep** |
| os_navzero - os | 0.1129 [0.0453, 0.1780] **sep** | 0.2197 [0.0527, 0.3791] **sep** | 0.0345 [0.0121, 0.0568] **sep** | 0.1230 [0.0621, 0.1844] **sep** | 0.1110 [-0.0712, 0.3300] ns | 0.0013 [-0.0004, 0.0034] ns | 0.0103 [-0.0322, 0.0526] ns | -0.0166 [-0.0275, -0.0074] **sep** | 0.0087 [-0.0109, 0.0303] ns |
| os_navshuf - os | 0.1415 [0.0827, 0.2066] **sep** | 0.3652 [0.2185, 0.5268] **sep** | 0.0443 [0.0249, 0.0650] **sep** | 0.0993 [0.0515, 0.1497] **sep** | 0.2734 [0.0530, 0.5214] **sep** | 0.0020 [0.0002, 0.0041] **sep** | 0.0749 [0.0260, 0.1320] **sep** | -0.0147 [-0.0240, -0.0052] **sep** | -0.0243 [-0.0493, -0.0016] **sep** |
| vmaxzero_minus_os | ABSENT ['os_vmaxzero'] | | | | | | | | |
| os_navzero - ha0_ext | -0.3600 [-0.6537, -0.0596] **sep** | -1.5390 [-2.3219, -0.7486] **sep** | -0.0471 [-0.1255, 0.0382] ns | -0.0662 [-0.2996, 0.1746] ns | -1.1531 [-1.7757, -0.5244] **sep** | -0.0078 [-0.0134, -0.0022] **sep** | -0.4234 [-0.6120, -0.2325] **sep** | 0.0510 [0.0300, 0.0721] **sep** | 0.0117 [-0.0253, 0.0496] ns |
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
    "mean": 0.25,
    "ci": [
     0.1562,
     0.3551
    ],
    "n_windows": 352,
    "n_episodes": 29
   },
   "follows_TRUE_command": {
    "mean": 0.1903,
    "ci": [
     0.1036,
     0.2811
    ]
   },
   "paired_true_minus_shuffled": {
    "delta": 0.1534,
    "ci": [
     0.101,
     0.209
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
  "rows_with_no_compliant_candidate": 2216,
  "obeys": {
   "mean": 0.0153,
   "lo": 0.0038,
   "hi": 0.0305,
   "ci95": 0.0133,
   "se": 0.0069,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "episode_cluster_bootstrap"
  },
  "violation_ms": {
   "max": 27.909086863199867,
   "p95": 22.79382492701212,
   "mean_over_violators": 9.452687733360179,
   "n_violating": 2259
  },
  "ade_cost": {
   "delta": 0.0004,
   "lo": -0.003,
   "hi": 0.0038,
   "ci95": 0.0034,
   "p_delta_gt0": 0.598,
   "separated": false,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "paired_episode_cluster_bootstrap",
   "_reads": "forced minus baseline, in metres. A model can obey ANY ceiling by planning to stop; this delta is what says it did not."
  },
  "verdict": "FAIL",
  "reason": "obedience lower bound 0.0038 < 0.99. 2259 of 2294 windows exceeded the forced ceiling; 2216 of them had no compliant candidate in the fan at all, which is a VOCABULARY limit, not a model one \u2014 separate the two before reading this as disobedience."
 },
 "tzero": {
  "test": "T-ZERO (non-turn diagnostic)",
  "classes": {
   "FOLLOW_LANE": {
    "n": 911,
    "recall_nav_true": 0.4632272228320527,
    "recall_nav_zero": 0.4105378704720088,
    "retained_fraction": 0.8862559241706162,
    "is_turn_class": false,
    "powered": true
   },
   "TURN_L": {
    "n": 40,
    "recall_nav_true": 0.8,
    "recall_nav_zero": 0.6,
    "retained_fraction": 0.7499999999999999,
    "is_turn_class": true,
    "powered": true,
    "_note": "EXCLUDED from the sum
```

### Bars

* **BAR-R6-1** — refcv6 beats the ECHO control: os - ha0_ext < 0, CI upper < 0 — **FAIL** — seed 0: 0.0885 [0.0695, 0.1094]; seed 1: 0.0885 [0.0695, 0.1092]
* **BAR-R6-2** — refcv6 beats HOLD-ACTION: os - ha < 0, separated — **FAIL** — seed 0: 0.0761 [0.0569, 0.0974]; seed 1: 0.0761 [0.0569, 0.0971]
* **BAR-R6-3** — refcv6 beats refcv4b: os - refcv4b.os < 0, separated — **FAIL** — seed 0: 0.0797 [0.0574, 0.1017]; seed 1: 0.0797 [0.0577, 0.1023]
* **BAR-R6-4** — refcv6 beats refcv5-v2 (seed 0): os - refcv5v2.os < 0, separated — **FAIL** — seed 0: 0.0681 [0.0456, 0.0909]; seed 1: 0.0681 [0.0455, 0.0909]
* **BAR-R6-5** — at 6 s: os - ha0_ext (ADE 0-6 s) < 0, separated — **PASS** — seed 0: -0.4729 [-0.7531, -0.178]; seed 1: -0.4601 [-0.7453, -0.1658]
