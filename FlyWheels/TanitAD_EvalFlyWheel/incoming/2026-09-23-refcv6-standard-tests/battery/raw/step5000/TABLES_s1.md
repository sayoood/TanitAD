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

### Four families + ADE, S2 (4754 windows / 139 episodes), inference seed 1

| arm | tier | ADE 0–2 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 0.3771 [0.3454, 0.4137] | 0.7757 | 0.3037 | 0.8096 | 0.2823 | 1.8703 | 0.005357 | 1.5717 | 0.1681 | 0.7232 | 0.5268 |
| refcv6 nav withheld | T1 | 0.4002 [0.3666, 0.4386] | 0.8201 | 0.3275 | 0.7908 | 0.3096 | 1.8967 | 0.005670 | 1.5870 | 0.1687 | 0.7091 | 0.5014 |
| refcv6 nav shuffled | T1 | 0.3871 [0.3544, 0.4259] | 0.7975 | 0.3127 | 0.8032 | 0.2923 | 1.9267 | 0.005888 | 1.6360 | 0.1692 | 0.7078 | 0.5180 |
| refcv6 nav flipped | T1 | 0.3926 [0.3579, 0.4348] | 0.8115 | 0.3042 | 0.8102 | 0.2838 | 2.1786 | 0.007468 | 1.9260 | 0.1832 | 0.6589 | 0.5320 |
| refcv6 max-speed withheld | T1 | 0.3785 [0.3465, 0.4151] | 0.7797 | 0.3077 | 0.8081 | 0.2862 | 1.8557 | 0.005324 | 1.5642 | 0.1665 | 0.7198 | 0.5295 |
| refcv4b `os` (banked) | T1 | 0.2974 [0.2689, 0.3303] | 0.6363 | 0.2908 | 0.8321 | 0.2553 | 1.3103 | 0.008266 | 1.7557 | 0.0979 | 0.8274 | 0.5215 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 0.3090 [0.2797, 0.3427] | 0.6629 | 0.2929 | 0.8338 | 0.2666 | 1.2241 | 0.003529 | 1.0619 | 0.0994 | 0.8190 | 0.5450 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 0.3091 [0.2793, 0.3429] | 0.6628 | 0.2931 | 0.8329 | 0.2666 | 1.2154 | 0.003533 | 1.0671 | 0.0998 | 0.8264 | 0.5334 |
| `ha` hold-action | T1 | 0.3010 [0.2761, 0.3298] | 0.6623 | 0.2548 | 0.8655 | 0.2354 | 1.5706 | 0.004089 | 1.4738 | 0.1238 | 0.7370 | 0.6066 |
| `ha0` constant velocity | T1 | 0.6764 [0.6033, 0.7529] | 1.4113 | 0.4891 | 0.7046 | 0.4718 | 2.8200 | 0.006903 | 2.4052 | 0.3170 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 0.2886 [0.2652, 0.3158] | 0.6355 | 0.2548 | 0.8655 | 0.2347 | 1.4522 | 0.003767 | 1.3575 | 0.1079 | 0.7544 | 0.6066 |
| `stop` STOP | T1 | 14.1180 [12.5982, 15.7166] | 22.5650 | 11.2911 | 0.0557 | 14.0984 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 0.3170 | 0.0000 | 0.0000 |
| `oracle_sel` (T0 ceiling) | T0 | 0.3158 [0.2863, 0.3512] | 0.6191 | 0.2405 | 0.8837 | 0.2322 | 1.3415 | 0.003869 | 1.2185 | 0.1484 | 0.8111 | 0.7029 |

### Distance keeping

lead block: `b1_eval_lead_block.npz`, status PRESENT, instants [1.0, 2.0] s

| arm | status | n with lead (eps) | min headway m [CI] | min time-gap s [CI] (n) | min TTC s [CI] | n closing (rest censored at 30 s) |
|---|---|---|---|---|---|---|
| **refcv6 `os`** | OK | 1324 (68) | 27.5965 [23.8799, 31.9844] | 4.1197 [3.2974, 5.1631] (1183) | 24.0486 [22.4540, 25.6537] | 517 |
| refcv6 nav withheld | OK | 1319 (68) | 27.6336 [23.8735, 31.8905] | 4.1205 [3.3081, 5.1388] (1178) | 24.4185 [22.8379, 25.9802] | 494 |
| refcv6 nav shuffled | OK | 1326 (68) | 27.7700 [23.9560, 32.1489] | 4.1057 [3.2691, 5.1414] (1185) | 24.1106 [22.4109, 25.7228] | 512 |
| refcv6 nav flipped | OK | 1327 (70) | 27.6415 [23.6315, 32.0310] | 4.1101 [3.3106, 5.0408] (1186) | 23.9833 [22.2675, 25.5083] | 530 |
| refcv6 max-speed withheld | OK | 1322 (68) | 27.6292 [23.8672, 32.0078] | 4.1074 [3.2992, 5.1162] (1181) | 24.0354 [22.4222, 25.6089] | 525 |
| refcv4b `os` (banked) | OK | 1224 (67) | 28.4517 [24.4003, 32.7611] | 4.1118 [3.2775, 5.0945] (1154) | 24.8434 [23.3648, 26.1655] | 470 |
| refcv5-v2 `os` seed 0 (banked) | OK | 1308 (68) | 27.8285 [23.9407, 32.4796] | 4.0885 [3.2646, 5.0234] (1167) | 24.6454 [23.2247, 26.0281] | 524 |
| refcv5-v2 `os` seed 1 (banked) | OK | 1307 (71) | 27.8319 [23.9316, 31.9794] | 4.0289 [3.2461, 5.0218] (1165) | 24.5477 [23.0821, 25.9259] | 541 |
| `ha` hold-action | OK | 1291 (67) | 28.0480 [23.8033, 32.3785] | 4.2073 [3.3867, 5.2420] (1150) | 25.5883 [24.2766, 26.7972] | 447 |
| `ha0` constant velocity | OK | 1315 (70) | 27.4550 [23.7229, 31.7446] | 4.1178 [3.2811, 5.1281] (1174) | 24.1077 [22.3616, 25.6733] | 499 |
| `ha0_ext` echo | OK | 1285 (67) | 28.0211 [23.7324, 32.3570] | 4.1966 [3.3733, 5.2300] (1144) | 25.5453 [24.2463, 26.7708] | 446 |
| `stop` STOP | OK | 1315 (70) | 38.6200 [33.3724, 44.1749] | 5.3059 [4.4617, 6.3288] (1174) | 29.9515 [29.8730, 30.0000] | 21 |
| `oracle_sel` (T0 ceiling) | OK | 1334 (70) | 27.6627 [23.7989, 31.8758] | 4.0552 [3.2532, 4.9875] (1193) | 24.4560 [22.9694, 25.7996] | 517 |

### Paired cells, S2, inference seed 1

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | 0.0885 [0.0695, 0.1092] **sep** | 0.1402 [0.0972, 0.1873] **sep** | 0.0489 [0.0303, 0.0679] **sep** | 0.0476 [0.0290, 0.0666] **sep** | 0.4273 [0.3070, 0.5699] **sep** | 0.0040 [0.0013, 0.0069] **sep** | 0.0602 [0.0481, 0.0742] **sep** | -0.0063 [-0.0198, 0.0069] ns | -0.0179 [-0.0320, -0.0053] **sep** |
| os - ha | 0.0761 [0.0569, 0.0971] **sep** | 0.1134 [0.0691, 0.1621] **sep** | 0.0489 [0.0303, 0.0679] **sep** | 0.0469 [0.0282, 0.0660] **sep** | 0.3098 [0.1871, 0.4538] **sep** | 0.0020 [-0.0009, 0.0051] ns | 0.0443 [0.0328, 0.0580] **sep** | -0.0025 [-0.0160, 0.0107] ns | -0.0179 [-0.0320, -0.0053] **sep** |
| os - ha0 | -0.2993 [-0.3590, -0.2417] **sep** | -0.6356 [-0.7647, -0.5089] **sep** | -0.1854 [-0.2232, -0.1502] **sep** | -0.1895 [-0.2256, -0.1557] **sep** | -0.9391 [-1.2605, -0.6402] **sep** | -0.0145 [-0.0205, -0.0090] **sep** | -0.1489 [-0.2044, -0.0986] **sep** | 0.0709 [0.0419, 0.1020] **sep** | 0.0688 [0.0432, 0.0939] **sep** |
| os - stop | -13.7409 [-15.2674, -12.2374] **sep** | -21.7893 [-24.2363, -19.3726] **sep** | -10.9874 [-12.2108, -9.7768] **sep** | -13.8161 [-15.3383, -12.3100] **sep** | — | — | -0.1489 [-0.2044, -0.0986] **sep** | 0.0709 [0.0419, 0.1020] **sep** | 0.0688 [0.0432, 0.0939] **sep** |
| os - b_refcv4b | 0.0797 [0.0577, 0.1023] **sep** | 0.1394 [0.0947, 0.1858] **sep** | 0.0128 [-0.0113, 0.0363] ns | 0.0270 [0.0012, 0.0517] **sep** | 0.5050 [0.3588, 0.6570] **sep** | -0.0041 [-0.0073, -0.0010] **sep** | 0.0702 [0.0535, 0.0881] **sep** | -0.0225 [-0.0339, -0.0118] **sep** | -0.0006 [-0.0135, 0.0126] ns |
| os - b_refcv5v2_s0 | 0.0681 [0.0455, 0.0909] **sep** | 0.1128 [0.0668, 0.1607] **sep** | 0.0108 [-0.0132, 0.0344] ns | 0.0157 [-0.0099, 0.0408] ns | 0.5762 [0.4384, 0.7196] **sep** | 0.0090 [0.0067, 0.0114] **sep** | 0.0687 [0.0533, 0.0863] **sep** | -0.0208 [-0.0318, -0.0099] **sep** | -0.0088 [-0.0225, 0.0051] ns |
| os - b_refcv5v2_s1 | 0.0680 [0.0453, 0.0913] **sep** | 0.1129 [0.0661, 0.1609] **sep** | 0.0106 [-0.0135, 0.0345] ns | 0.0157 [-0.0104, 0.0406] ns | 0.5765 [0.4347, 0.7244] **sep** | 0.0089 [0.0067, 0.0114] **sep** | 0.0684 [0.0527, 0.0863] **sep** | -0.0227 [-0.0339, -0.0118] **sep** | -0.0055 [-0.0189, 0.0082] ns |
| os_navzero - os | 0.0232 [0.0144, 0.0324] **sep** | 0.0444 [0.0255, 0.0637] **sep** | 0.0238 [0.0143, 0.0334] **sep** | 0.0273 [0.0183, 0.0364] **sep** | 0.0307 [-0.0249, 0.0877] ns | 0.0004 [-0.0007, 0.0015] ns | 0.0006 [-0.0032, 0.0040] ns | -0.0006 [-0.0071, 0.0059] ns | -0.0004 [-0.0082, 0.0078] ns |
| os_navshuf - os | 0.0101 [0.0043, 0.0162] **sep** | 0.0218 [0.0093, 0.0352] **sep** | 0.0090 [0.0043, 0.0138] **sep** | 0.0100 [0.0057, 0.0146] **sep** | 0.0574 [-0.0007, 0.1216] ns | 0.0012 [0.0000, 0.0025] **sep** | 0.0011 [-0.0033, 0.0061] ns | -0.0019 [-0.0071, 0.0034] ns | -0.0023 [-0.0080, 0.0038] ns |
| os_vmaxzero - os | 0.0014 [-0.0015, 0.0041] ns | 0.0040 [-0.0020, 0.0098] ns | 0.0040 [0.0012, 0.0070] **sep** | 0.0039 [0.0012, 0.0067] **sep** | -0.0113 [-0.0260, 0.0033] ns | -0.0001 [-0.0004, 0.0002] ns | -0.0016 [-0.0033, 0.0001] ns | -0.0008 [-0.0036, 0.0021] ns | 0.0004 [-0.0046, 0.0055] ns |
| os_navzero - ha0_ext | 0.1116 [0.0899, 0.1362] **sep** | 0.1847 [0.1363, 0.2402] **sep** | 0.0727 [0.0507, 0.0960] **sep** | 0.0749 [0.0535, 0.0976] **sep** | 0.4572 [0.3247, 0.6060] **sep** | 0.0042 [0.0016, 0.0072] **sep** | 0.0608 [0.0488, 0.0749] **sep** | -0.0069 [-0.0183, 0.0046] ns | -0.0183 [-0.0335, -0.0048] **sep** |
| b_refcv4b - ha0_ext | 0.0088 [-0.0058, 0.0249] ns | 0.0008 [-0.0316, 0.0369] ns | 0.0361 [0.0185, 0.0546] **sep** | 0.0206 [0.0035, 0.0384] **sep** | -0.0840 [-0.2002, 0.0265] ns | 0.0073 [0.0036, 0.0112] **sep** | -0.0100 [-0.0221, 0.0019] ns | 0.0162 [0.0065, 0.0255] **sep** | -0.0172 [-0.0315, -0.0036] **sep** |
| b_refcv5v2_s0 - ha0_ext | 0.0204 [0.0046, 0.0383] **sep** | 0.0275 [-0.0078, 0.0659] ns | 0.0381 [0.0194, 0.0579] **sep** | 0.0319 [0.0135, 0.0509] **sep** | -0.1504 [-0.2507, -0.0529] **sep** | -0.0051 [-0.0078, -0.0024] **sep** | -0.0085 [-0.0197, 0.0022] ns | 0.0145 [0.0038, 0.0248] **sep** | -0.0090 [-0.0231, 0.0044] ns |

n = 4754 windows / 139 episodes · estimator: paired episode-cluster bootstrap

### TACTICAL — declared heads and the 22-token goal set, inference seed 1

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

### S6 — the 6 s horizon, inference seed 1

| arm | tier | ADE 1–6 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 3.0877 [2.8126, 3.3835] | 7.4821 | 1.0209 | 0.4878 | 2.4727 | 4.6165 | 0.006930 | 2.2951 | 1.2332 | 0.6234 | 0.3078 |
| refcv6 nav withheld | T1 | 3.1915 [2.9097, 3.5025] | 7.6795 | 1.0451 | 0.4788 | 2.5799 | 4.7537 | 0.007387 | 2.3520 | 1.2565 | 0.5811 | 0.3004 |
| refcv6 nav shuffled | T1 | 3.2081 [2.9288, 3.5181] | 7.7925 | 1.0600 | 0.4741 | 2.5671 | 4.8422 | 0.007397 | 2.3678 | 1.2807 | 0.5851 | 0.2558 |
| refcv6 nav flipped | T1 | 3.2509 [2.9505, 3.6018] | 7.8707 | 1.0278 | 0.4874 | 2.4840 | 5.5488 | 0.008340 | 2.5761 | 1.4171 | 0.5406 | 0.2919 |
| refcv4b `os` (banked) | T1 | 2.7225 [2.4832, 2.9711] | 6.6964 | 0.9392 | 0.5373 | 2.2347 | 3.5207 | 0.005894 | 1.8225 | 0.9982 | 0.6790 | 0.3426 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 2.7140 [2.4798, 2.9674] | 6.6030 | 0.9257 | 0.5361 | 2.2110 | 3.5208 | 0.005141 | 1.7695 | 1.0255 | 0.6552 | 0.3475 |
| refcv5-v2 `os` seed 1 (banked) | T1 | 2.7096 [2.4726, 2.9634] | 6.5888 | 0.9238 | 0.5381 | 2.1968 | 3.4947 | 0.005229 | 1.7852 | 1.0355 | 0.6557 | 0.3460 |
| `ha` hold-action | T1 | 3.6158 [3.2069, 4.0256] | 9.3416 | 1.0955 | 0.5251 | 2.6612 | 6.1561 | 0.006997 | 2.8433 | 1.7404 | 0.4887 | 0.3172 |
| `ha0` constant velocity | T1 | 4.9926 [4.3774, 5.6601] | 11.3583 | 1.2903 | 0.4470 | 3.4452 | 6.9005 | 0.006519 | 2.4079 | 2.4322 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 3.5479 [3.1510, 3.9432] | 9.2071 | 1.0957 | 0.5249 | 2.6460 | 5.9407 | 0.006850 | 2.7751 | 1.6680 | 0.4997 | 0.3172 |
| `stop` STOP | T1 | 39.2767 [35.0040, 43.7743] | 66.9735 | 11.2321 | 0.0498 | 38.8590 | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | UNAVAILABLE (n=0) | 2.4322 | 0.0000 | 0.0000 |

| cell (b − a) | ADE m [CI] sep | FDE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (valid steps, A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|---|
| os - ha0_ext | -0.4601 [-0.7453, -0.1658] **sep** | -1.7250 [-2.4873, -0.9407] **sep** | -0.0748 [-0.1498, 0.0046] ns | -0.1733 [-0.3932, 0.0474] ns | -1.2286 [-1.8819, -0.5903] **sep** | -0.0082 [-0.0139, -0.0024] **sep** | -0.4348 [-0.6332, -0.2392] **sep** | 0.0671 [0.0464, 0.0882] **sep** | 0.0131 [-0.0237, 0.0466] ns |
| os - ha | -0.5281 [-0.8285, -0.2181] **sep** | -1.8595 [-2.6476, -1.0544] **sep** | -0.0746 [-0.1497, 0.0047] ns | -0.1885 [-0.4122, 0.0369] ns | -1.4424 [-2.1365, -0.7612] **sep** | -0.0094 [-0.0154, -0.0034] **sep** | -0.5072 [-0.7220, -0.2992] **sep** | 0.0714 [0.0496, 0.0925] **sep** | 0.0131 [-0.0237, 0.0466] ns |
| os - ha0 | -1.9049 [-2.3869, -1.4597] **sep** | -3.8762 [-4.9932, -2.8862] **sep** | -0.2694 [-0.3672, -0.1744] **sep** | -0.9726 [-1.2475, -0.7052] **sep** | -2.2184 [-3.1561, -1.4190] **sep** | -0.0020 [-0.0066, 0.0022] ns | -1.1990 [-1.6812, -0.7958] **sep** | 0.1401 [0.1025, 0.1803] **sep** | 0.1265 [0.0799, 0.1733] **sep** |
| os - stop | -36.1890 [-40.5686, -31.8817] **sep** | -59.4914 [-67.0210, -52.1670] **sep** | -10.2112 [-11.4418, -8.9714] **sep** | -36.3864 [-40.7912, -32.1055] **sep** | — | — | -1.1990 [-1.6812, -0.7958] **sep** | 0.1401 [0.1025, 0.1803] **sep** | 0.1265 [0.0799, 0.1733] **sep** |
| os - b_refcv4b | 0.3652 [0.2097, 0.5275] **sep** | 0.7857 [0.4083, 1.1600] **sep** | 0.0817 [0.0280, 0.1374] **sep** | 0.2380 [0.0977, 0.3852] **sep** | 1.0705 [0.6924, 1.4791] **sep** | 0.0085 [0.0058, 0.0113] **sep** | 0.2350 [0.1294, 0.3513] **sep** | -0.0158 [-0.0289, -0.0035] **sep** | -0.0294 [-0.0587, 0.0003] ns |
| os - b_refcv5v2_s0 | 0.3737 [0.2196, 0.5358] **sep** | 0.8791 [0.5159, 1.2709] **sep** | 0.0952 [0.0428, 0.1489] **sep** | 0.2617 [0.1152, 0.4070] **sep** | 1.0167 [0.6682, 1.3772] **sep** | 0.0094 [0.0071, 0.0118] **sep** | 0.2078 [0.1072, 0.3271] **sep** | -0.0074 [-0.0188, 0.0044] ns | -0.0341 [-0.0635, -0.0049] **sep** |
| os - b_refcv5v2_s1 | 0.3781 [0.2270, 0.5416] **sep** | 0.8933 [0.5429, 1.2773] **sep** | 0.0971 [0.0451, 0.1503] **sep** | 0.2759 [0.1289, 0.4212] **sep** | 1.0051 [0.6457, 1.3712] **sep** | 0.0092 [0.0068, 0.0115] **sep** | 0.1977 [0.0995, 0.3101] **sep** | -0.0074 [-0.0190, 0.0044] ns | -0.0335 [-0.0646, -0.0033] **sep** |
| os_navzero - os | 0.1038 [0.0349, 0.1755] **sep** | 0.1974 [0.0294, 0.3692] **sep** | 0.0243 [-0.0008, 0.0495] ns | 0.1072 [0.0386, 0.1789] **sep** | 0.1489 [-0.0413, 0.3773] ns | 0.0014 [-0.0001, 0.0033] ns | 0.0233 [-0.0171, 0.0639] ns | -0.0139 [-0.0242, -0.0046] **sep** | 0.0033 [-0.0172, 0.0254] ns |
| os_navshuf - os | 0.1204 [0.0662, 0.1763] **sep** | 0.3105 [0.1774, 0.4497] **sep** | 0.0391 [0.0188, 0.0588] **sep** | 0.0944 [0.0485, 0.1413] **sep** | 0.2558 [0.0386, 0.4967] **sep** | 0.0019 [0.0003, 0.0038] **sep** | 0.0475 [-0.0024, 0.1064] ns | -0.0147 [-0.0238, -0.0063] **sep** | -0.0384 [-0.0644, -0.0125] **sep** |
| vmaxzero_minus_os | ABSENT ['os_vmaxzero'] | | | | | | | | |
| os_navzero - ha0_ext | -0.3563 [-0.6451, -0.0635] **sep** | -1.5276 [-2.2898, -0.7466] **sep** | -0.0505 [-0.1285, 0.0344] ns | -0.0661 [-0.2966, 0.1724] ns | -1.1219 [-1.7397, -0.5094] **sep** | -0.0075 [-0.0130, -0.0019] **sep** | -0.4115 [-0.6025, -0.2223] **sep** | 0.0532 [0.0322, 0.0739] **sep** | 0.0164 [-0.0218, 0.0531] ns |
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
     0.1593,
     0.3508
    ],
    "n_windows": 352,
    "n_episodes": 29
   },
   "follows_TRUE_command": {
    "mean": 0.1903,
    "ci": [
     0.1039,
     0.2804
    ]
   },
   "paired_true_minus_shuffled": {
    "delta": 0.1335,
    "ci": [
     0.0858,
     0.1825
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
   "mean": 0.0166,
   "lo": 0.0038,
   "hi": 0.0331,
   "ci95": 0.0146,
   "se": 0.0076,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "episode_cluster_bootstrap"
  },
  "violation_ms": {
   "max": 27.878256479899086,
   "p95": 22.787918154398596,
   "mean_over_violators": 9.46054356656176,
   "n_violating": 2256
  },
  "ade_cost": {
   "delta": 0.0023,
   "lo": -0.0015,
   "hi": 0.006,
   "ci95": 0.0037,
   "p_delta_gt0": 0.8845,
   "separated": false,
   "reducer": "mean",
   "n_windows": 2294,
   "n_episodes": 81,
   "n_boot": 2000,
   "estimator": "paired_episode_cluster_bootstrap",
   "_reads": "forced minus baseline, in metres. A model can obey ANY ceiling by planning to stop; this delta is what says it did not."
  },
  "verdict": "FAIL",
  "reason": "obedience lower bound 0.0038 < 0.99. 2256 of 2294 windows exceeded the forced ceiling; 2216 of them had no compliant candidate in the fan at all, which is a VOCABULARY limit, not a model one \u2014 separate the two before reading this as disobedience."
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
    "_note": "EXCLUDED from the 
```

### Bars

* **BAR-R6-1** — refcv6 beats the ECHO control: os - ha0_ext < 0, CI upper < 0 — **FAIL** — seed 0: 0.0885 [0.0695, 0.1094]; seed 1: 0.0885 [0.0695, 0.1092]
* **BAR-R6-2** — refcv6 beats HOLD-ACTION: os - ha < 0, separated — **FAIL** — seed 0: 0.0761 [0.0569, 0.0974]; seed 1: 0.0761 [0.0569, 0.0971]
* **BAR-R6-3** — refcv6 beats refcv4b: os - refcv4b.os < 0, separated — **FAIL** — seed 0: 0.0797 [0.0574, 0.1017]; seed 1: 0.0797 [0.0577, 0.1023]
* **BAR-R6-4** — refcv6 beats refcv5-v2 (seed 0): os - refcv5v2.os < 0, separated — **FAIL** — seed 0: 0.0681 [0.0456, 0.0909]; seed 1: 0.0681 [0.0455, 0.0909]
* **BAR-R6-5** — at 6 s: os - ha0_ext (ADE 0-6 s) < 0, separated — **PASS** — seed 0: -0.4729 [-0.7531, -0.178]; seed 1: -0.4601 [-0.7453, -0.1658]
