### step5000: step 5000, ckpt md5 `8a1e4da0dc6561353b28e3947146751e`, SPEC sha at run `7bfca6cf2b81…`

**Tier and estimator.** T1 = self-action open loop; T0 = `oracle_sel` only. Paired episode-cluster bootstrap. One training seed. Every separated cell answers only the EPISODE question; the INFERENCE question is answered by the two-seed replicate below.

⛔ **Run-defect stamp (pre-switch checkpoint, step ≤ 34,500):** F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early (D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; audit 92337fa6). No weakness below may be attributed to the REGISTERED design while these hold. TACTICAL is read under BOTH label clocks (SPEC A5); this checkpoint's PRIMARY clock is the OLD one it was trained on. Physical-unit rates use dt = 0.5 s for a true 0.5033 s (row = 0.100667 s), identically for every arm and baseline.

**Gate.** G0 as registered **FAIL**, G0-A1 **FAIL**, G0-A2 (operative) **PASS**.
* A2 max wrapper rel: {'P1_as_run': 0.003920739304811635, 'P2_cudnn_det': 0.003920684765843788, 'P3_fp32_det': 1.8905077336620297e-06}.
* Reasons: A1 ['wrapper control failed: max rel 3.921e-03']; A2 [].

| bar | statement | seed 0 | seed 1 | verdict |
|---|---|---|---|---|
| BAR-R6-1 | refcv6 beats the ECHO control: os - ha0_ext < 0, CI upper < 0 | +0.0885 [+0.0695, +0.1094] **sep** | +0.0885 [+0.0695, +0.1092] **sep** | **FAIL** |
| BAR-R6-2 | refcv6 beats HOLD-ACTION: os - ha < 0, separated | +0.0761 [+0.0569, +0.0974] **sep** | +0.0761 [+0.0569, +0.0971] **sep** | **FAIL** |
| BAR-R6-3 | refcv6 beats refcv4b: os - refcv4b.os < 0, separated | +0.0797 [+0.0574, +0.1017] **sep** | +0.0797 [+0.0577, +0.1023] **sep** | **FAIL** |
| BAR-R6-4 | refcv6 beats refcv5-v2 (seed 0): os - refcv5v2.os < 0, separated | +0.0681 [+0.0456, +0.0909] **sep** | +0.0681 [+0.0455, +0.0909] **sep** | **FAIL** |
| BAR-R6-5 | at 6 s: os - ha0_ext (ADE 0-6 s) < 0, separated | -0.4729 [-0.7531, -0.1780] **sep** | -0.4601 [-0.7453, -0.1658] **sep** | **PASS** |

⭐ **Primary bar BAR-R6-1 = FAIL.** At this checkpoint refcv6 is **NOT PROVEN** to beat the echo control at 0–2 s.

**Inference-seed replicate** (os seed 0 − os seed 1, same windows): ADE +0.0000 [-0.0025, +0.0027]; max |path diff| 3.3298 m.

**Key paired cells, S2 (0–2 s), inference seed 0**

| cell (b − a) | ADE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|
| os − ha0_ext (echo) | +0.0885 [+0.0695, +0.1094] **sep** | +0.0495 [+0.0312, +0.0687] **sep** | +0.0486 [+0.0303, +0.0674] **sep** | +0.4205 [+0.2980, +0.5602] **sep** | +0.0039 [+0.0012, +0.0067] **sep** | +0.0588 [+0.0466, +0.0732] **sep** | -0.0076 [-0.0209, +0.0053] | -0.0187 [-0.0318, -0.0057] **sep** |
| os − ha (hold) | +0.0761 [+0.0569, +0.0974] **sep** | +0.0495 [+0.0312, +0.0687] **sep** | +0.0479 [+0.0295, +0.0669] **sep** | +0.3030 [+0.1788, +0.4454] **sep** | +0.0019 [-0.0011, +0.0049] | +0.0430 [+0.0312, +0.0568] **sep** | -0.0038 [-0.0172, +0.0093] | -0.0187 [-0.0318, -0.0057] **sep** |
| os − ha0 (CV) | -0.2993 [-0.3582, -0.2419] **sep** | -0.1849 [-0.2227, -0.1496] **sep** | -0.1885 [-0.2246, -0.1543] **sep** | -0.9443 [-1.2634, -0.6420] **sep** | -0.0145 [-0.0204, -0.0091] **sep** | -0.1503 [-0.2049, -0.1001] **sep** | +0.0696 [+0.0408, +0.1014] **sep** | +0.0679 [+0.0427, +0.0936] **sep** |
| os − refcv4b | +0.0797 [+0.0574, +0.1017] **sep** | +0.0134 [-0.0107, +0.0368] | +0.0280 [+0.0023, +0.0531] **sep** | +0.4982 [+0.3504, +0.6498] **sep** | -0.0042 [-0.0075, -0.0011] **sep** | +0.0689 [+0.0524, +0.0869] **sep** | -0.0238 [-0.0357, -0.0128] **sep** | -0.0015 [-0.0154, +0.0122] |
| os − refcv5-v2 s0 | +0.0681 [+0.0456, +0.0909] **sep** | +0.0114 [-0.0125, +0.0349] | +0.0167 [-0.0085, +0.0412] | +0.5699 [+0.4350, +0.7136] **sep** | +0.0089 [+0.0067, +0.0113] **sep** | +0.0674 [+0.0519, +0.0850] **sep** | -0.0221 [-0.0336, -0.0105] **sep** | -0.0097 [-0.0238, +0.0048] |
| nav withheld − os | +0.0246 [+0.0157, +0.0340] **sep** | +0.0242 [+0.0154, +0.0335] **sep** | +0.0271 [+0.0186, +0.0361] **sep** | +0.0350 [-0.0240, +0.0972] | +0.0004 [-0.0007, +0.0017] | +0.0018 [-0.0022, +0.0056] | +0.0025 [-0.0038, +0.0089] | -0.0006 [-0.0093, +0.0084] |
| nav shuffled − os | +0.0100 [+0.0042, +0.0166] **sep** | +0.0076 [+0.0025, +0.0126] **sep** | +0.0074 [+0.0027, +0.0122] **sep** | +0.0737 [+0.0095, +0.1489] **sep** | +0.0014 [+0.0002, +0.0028] **sep** | +0.0042 [-0.0004, +0.0092] | -0.0017 [-0.0067, +0.0034] | -0.0004 [-0.0076, +0.0065] |
| max-speed withheld − os | +0.0020 [-0.0010, +0.0049] | +0.0013 [-0.0016, +0.0041] | +0.0013 [-0.0015, +0.0040] | +0.0076 [-0.0121, +0.0310] | +0.0001 [-0.0002, +0.0005] | +0.0011 [-0.0008, +0.0030] | +0.0006 [-0.0023, +0.0034] | +0.0023 [-0.0032, +0.0082] |

n = 4754 windows / 139 episodes; paired episode-cluster bootstrap (n_boot 2000, seed 0, cluster = clip); **sep** = CI excludes 0

**Key paired cells, S2 (0–2 s), inference seed 1**

| cell (b − a) | ADE m | speed MAE | along MAE | heading ° | yaw-rate rad/s (A3) | cross-track | traj lat correct | traj lon correct |
|---|---|---|---|---|---|---|---|---|
| os − ha0_ext (echo) | +0.0885 [+0.0695, +0.1092] **sep** | +0.0489 [+0.0303, +0.0679] **sep** | +0.0476 [+0.0290, +0.0666] **sep** | +0.4273 [+0.3070, +0.5699] **sep** | +0.0040 [+0.0013, +0.0069] **sep** | +0.0602 [+0.0481, +0.0742] **sep** | -0.0063 [-0.0198, +0.0069] | -0.0179 [-0.0320, -0.0053] **sep** |
| os − ha (hold) | +0.0761 [+0.0569, +0.0971] **sep** | +0.0489 [+0.0303, +0.0679] **sep** | +0.0469 [+0.0282, +0.0660] **sep** | +0.3098 [+0.1871, +0.4538] **sep** | +0.0020 [-0.0009, +0.0051] | +0.0443 [+0.0328, +0.0580] **sep** | -0.0025 [-0.0160, +0.0107] | -0.0179 [-0.0320, -0.0053] **sep** |
| os − ha0 (CV) | -0.2993 [-0.3590, -0.2417] **sep** | -0.1854 [-0.2232, -0.1502] **sep** | -0.1895 [-0.2256, -0.1557] **sep** | -0.9391 [-1.2605, -0.6402] **sep** | -0.0145 [-0.0205, -0.0090] **sep** | -0.1489 [-0.2044, -0.0986] **sep** | +0.0709 [+0.0419, +0.1020] **sep** | +0.0688 [+0.0432, +0.0939] **sep** |
| os − refcv4b | +0.0797 [+0.0577, +0.1023] **sep** | +0.0128 [-0.0113, +0.0363] | +0.0270 [+0.0012, +0.0517] **sep** | +0.5050 [+0.3588, +0.6570] **sep** | -0.0041 [-0.0073, -0.0010] **sep** | +0.0702 [+0.0535, +0.0881] **sep** | -0.0225 [-0.0339, -0.0118] **sep** | -0.0006 [-0.0135, +0.0126] |
| os − refcv5-v2 s0 | +0.0681 [+0.0455, +0.0909] **sep** | +0.0108 [-0.0132, +0.0344] | +0.0157 [-0.0099, +0.0408] | +0.5762 [+0.4384, +0.7196] **sep** | +0.0090 [+0.0067, +0.0114] **sep** | +0.0687 [+0.0533, +0.0863] **sep** | -0.0208 [-0.0318, -0.0099] **sep** | -0.0088 [-0.0225, +0.0051] |
| nav withheld − os | +0.0232 [+0.0144, +0.0324] **sep** | +0.0238 [+0.0143, +0.0334] **sep** | +0.0273 [+0.0183, +0.0364] **sep** | +0.0307 [-0.0249, +0.0877] | +0.0004 [-0.0007, +0.0015] | +0.0006 [-0.0032, +0.0040] | -0.0006 [-0.0071, +0.0059] | -0.0004 [-0.0082, +0.0078] |
| nav shuffled − os | +0.0101 [+0.0043, +0.0162] **sep** | +0.0090 [+0.0043, +0.0138] **sep** | +0.0100 [+0.0057, +0.0146] **sep** | +0.0574 [-0.0007, +0.1216] | +0.0012 [+0.0000, +0.0025] **sep** | +0.0011 [-0.0033, +0.0061] | -0.0019 [-0.0071, +0.0034] | -0.0023 [-0.0080, +0.0038] |
| max-speed withheld − os | +0.0014 [-0.0015, +0.0041] | +0.0040 [+0.0012, +0.0070] **sep** | +0.0039 [+0.0012, +0.0067] **sep** | -0.0113 [-0.0260, +0.0033] | -0.0001 [-0.0004, +0.0002] | -0.0016 [-0.0033, +0.0001] | -0.0008 [-0.0036, +0.0021] | +0.0004 [-0.0046, +0.0055] |

n = 4754 windows / 139 episodes; paired episode-cluster bootstrap (n_boot 2000, seed 0, cluster = clip); **sep** = CI excludes 0

**Levels, S2, inference seed 0** (4754 windows / 139 eps)

| arm | tier | ADE 0–2 s m [CI] | FDE m | speed MAE m/s | tgt-speed acc@0.5 | along MAE m | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | lat κ (traj) | lon κ (traj) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **refcv6 `os`** | T1 | 0.3771 [0.3458, 0.4138] | 0.7762 | 0.3043 | 0.8101 | 0.2833 | 1.8623 | 0.005321 | 1.5649 | 0.1668 | 0.7184 | 0.5309 |
| refcv6 nav withheld | T1 | 0.4017 [0.3688, 0.4400] | 0.8223 | 0.3285 | 0.7900 | 0.3104 | 1.8942 | 0.005655 | 1.5803 | 0.1686 | 0.7183 | 0.4994 |
| refcv6 max-speed withheld | T1 | 0.3792 [0.3479, 0.4156] | 0.7800 | 0.3056 | 0.8086 | 0.2846 | 1.8699 | 0.005344 | 1.5745 | 0.1679 | 0.7207 | 0.5324 |
| refcv4b `os` (banked) | T1 | 0.2974 [0.2689, 0.3303] | 0.6363 | 0.2908 | 0.8321 | 0.2553 | 1.3103 | 0.008266 | 1.7557 | 0.0979 | 0.8274 | 0.5215 |
| refcv5-v2 `os` seed 0 (banked) | T1 | 0.3090 [0.2797, 0.3427] | 0.6629 | 0.2929 | 0.8338 | 0.2666 | 1.2241 | 0.003529 | 1.0619 | 0.0994 | 0.8190 | 0.5450 |
| `ha` hold-action | T1 | 0.3010 [0.2761, 0.3298] | 0.6623 | 0.2548 | 0.8655 | 0.2354 | 1.5706 | 0.004089 | 1.4738 | 0.1238 | 0.7370 | 0.6066 |
| `ha0` constant velocity | T1 | 0.6764 [0.6033, 0.7529] | 1.4113 | 0.4891 | 0.7046 | 0.4718 | 2.8200 | 0.006903 | 2.4052 | 0.3170 | 0.0000 | 0.0000 |
| `ha0_ext` echo | T1 | 0.2886 [0.2652, 0.3158] | 0.6355 | 0.2548 | 0.8655 | 0.2347 | 1.4522 | 0.003767 | 1.3575 | 0.1079 | 0.7544 | 0.6066 |
| `oracle_sel` (T0 ceiling) | T0 | 0.3164 [0.2878, 0.3517] | 0.6201 | 0.2400 | 0.8841 | 0.2319 | 1.3521 | 0.003838 | 1.2155 | 0.1489 | 0.8077 | 0.6950 |

**Distance keeping (LONGITUDINAL)**

lead block: `b1_eval_lead_block.npz`, status PRESENT, instants [1.0, 2.0] s

| arm | status | n with lead (eps) | min headway m [CI] | min time-gap s [CI] (n) | min TTC s [CI] | n closing (rest censored at 30 s) |
|---|---|---|---|---|---|---|
| **refcv6 `os`** | OK | 1317 (69) | 27.6045 [23.5467, 31.9670] | 4.1242 [3.3039, 5.1812] (1176) | 24.0720 [22.3495, 25.5676] | 515 |
| refcv6 nav withheld | OK | 1324 (69) | 27.6290 [23.5783, 31.9496] | 4.1282 [3.3049, 5.1872] (1183) | 24.4452 [22.7535, 25.9688] | 490 |
| refcv6 max-speed withheld | OK | 1320 (68) | 27.6536 [23.8710, 32.0282] | 4.1068 [3.2758, 5.1405] (1179) | 24.1271 [22.5181, 25.6914] | 516 |
| refcv4b `os` (banked) | OK | 1224 (67) | 28.4517 [24.4003, 32.7611] | 4.1118 [3.2775, 5.0945] (1154) | 24.8434 [23.3648, 26.1655] | 470 |
| refcv5-v2 `os` seed 0 (banked) | OK | 1308 (68) | 27.8285 [23.9407, 32.4796] | 4.0885 [3.2646, 5.0234] (1167) | 24.6454 [23.2247, 26.0281] | 524 |
| `ha` hold-action | OK | 1291 (67) | 28.0480 [23.8033, 32.3785] | 4.2073 [3.3867, 5.2420] (1150) | 25.5883 [24.2766, 26.7972] | 447 |
| `ha0` constant velocity | OK | 1315 (70) | 27.4550 [23.7229, 31.7446] | 4.1178 [3.2811, 5.1281] (1174) | 24.1077 [22.3616, 25.6733] | 499 |
| `ha0_ext` echo | OK | 1285 (67) | 28.0211 [23.7324, 32.3570] | 4.1966 [3.3733, 5.2300] (1144) | 25.5453 [24.2463, 26.7708] | 446 |
| `oracle_sel` (T0 ceiling) | OK | 1327 (70) | 27.5702 [23.7201, 31.7699] | 4.0343 [3.2347, 4.9692] (1186) | 24.5335 [23.0731, 25.8609] | 509 |

**TACTICAL, inference seed 0.** Declared heads use the trainer's label clock (stamp above).

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

**TACTICAL, inference seed 1:** LAT: v6 decoder acc 0.7169 [0.6427, 0.7857] κ 0.2418; LON: v6 decoder acc 0.3707 [0.3003, 0.4432] κ 0.1821

### TACTICAL under both label clocks (SPEC A5): step5000, step 5000. PRIMARY clock: **OLD**

Clock and dt source: CORRECTED = grid_start + (t + w - 1 + n_stack - 1) * dt; 136/139 eval clips on the sidecar clock (dt median 0.10066662995966351), 3 on nominal dt 0.1 s with grid_start 0, 0 on pose dt; OLD = (t + w - 1) * 0.1 s.

Tier T1 (UNRULED for an action-free model). Declared-head accuracy [episode-cluster CI], κ full-set; in-band n per clock. Label tables sha256 `83b47aae4aff…`; controls C1–C4 PASS (`raw/label_clock/label_clock_record.json`).

**Inference seed 0**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7169 [0.6427, 0.7857] · 0.2418 (1141) ⭐ | 0.7134 [0.6392, 0.7821] · 0.2300 (1106) |
| LAT | z_tac_v7_heads | 0.7038 [0.6279, 0.7759] · 0.1975 (1141) ⭐ | 0.7016 [0.6268, 0.7749] · 0.1937 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.6713 [0.5925, 0.7483] · 0.0066 (1141) ⭐ | 0.6718 [0.5928, 0.7489] · 0.0068 (1106) |
| LON | v6_behaviour_decoder | 0.3707 [0.3003, 0.4432] · 0.1821 (1141) ⭐ | 0.3680 [0.2965, 0.4418] · 0.1786 (1106) |
| LON | z_tac_v7_heads | 0.3883 [0.3173, 0.4618] · 0.2131 (1141) ⭐ | 0.3843 [0.3140, 0.4587] · 0.2083 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.3655 [0.2948, 0.4378] · 0.1790 (1141) ⭐ | 0.3644 [0.2924, 0.4370] · 0.1769 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7372 / SCOREABLE | 882 / 0.7369 / SCOREABLE |
| TURN_L | 40 / 0.9751 / UNSCOREABLE (n_pos < 200) | 40 / 0.9769 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9571 / UNSCOREABLE (n_pos < 200) | 63 / 0.9532 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9939 / UNSCOREABLE (n_pos < 200) | 16 / 0.9942 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8648 / UNSCOREABLE (n_pos < 200) | 73 / 0.8620 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.0952 / UNSCOREABLE (n_pos < 200) | 48 / 0.1042 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.6914 / UNSCOREABLE (n_pos < 200) | 8 / 0.6937 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.4499 / UNSCOREABLE (n_pos < 200) | 40 / 0.4520 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 1.0000 / UNSCOREABLE (n_pos < 200) | 40 / 1.0000 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.5646 / UNSCOREABLE (n_pos < 200) | 63 / 0.5544 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6526 / UNSCOREABLE (n_pos < 200) | 24 / 0.6485 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7104 / UNSCOREABLE (n_pos < 200) | 118 / 0.7155 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

**Inference seed 1**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7169 [0.6427, 0.7857] · 0.2418 (1141) ⭐ | 0.7134 [0.6392, 0.7821] · 0.2300 (1106) |
| LAT | z_tac_v7_heads | 0.7038 [0.6279, 0.7759] · 0.1975 (1141) ⭐ | 0.7016 [0.6268, 0.7749] · 0.1937 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.6713 [0.5925, 0.7483] · 0.0066 (1141) ⭐ | 0.6718 [0.5928, 0.7489] · 0.0068 (1106) |
| LON | v6_behaviour_decoder | 0.3707 [0.3003, 0.4432] · 0.1821 (1141) ⭐ | 0.3680 [0.2965, 0.4418] · 0.1786 (1106) |
| LON | z_tac_v7_heads | 0.3883 [0.3173, 0.4618] · 0.2131 (1141) ⭐ | 0.3843 [0.3140, 0.4587] · 0.2083 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.3655 [0.2948, 0.4378] · 0.1790 (1141) ⭐ | 0.3644 [0.2924, 0.4370] · 0.1769 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7372 / SCOREABLE | 882 / 0.7369 / SCOREABLE |
| TURN_L | 40 / 0.9751 / UNSCOREABLE (n_pos < 200) | 40 / 0.9769 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9571 / UNSCOREABLE (n_pos < 200) | 63 / 0.9532 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9939 / UNSCOREABLE (n_pos < 200) | 16 / 0.9942 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8648 / UNSCOREABLE (n_pos < 200) | 73 / 0.8620 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.0952 / UNSCOREABLE (n_pos < 200) | 48 / 0.1042 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.6914 / UNSCOREABLE (n_pos < 200) | 8 / 0.6937 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.4499 / UNSCOREABLE (n_pos < 200) | 40 / 0.4520 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 1.0000 / UNSCOREABLE (n_pos < 200) | 40 / 1.0000 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.5646 / UNSCOREABLE (n_pos < 200) | 63 / 0.5544 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6526 / UNSCOREABLE (n_pos < 200) | 24 / 0.6485 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7104 / UNSCOREABLE (n_pos < 200) | 118 / 0.7155 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

⭐ = the PRIMARY clock for this checkpoint. A tactical comparison across step 34,500 must show both clocks, and it mixes training time with the fix.


**STRATEGIC: NOT APPLICABLE, n = 0.** The strategic layer is OFF (`--no-strategic`); its route CE is gated off in training, so the route head is untrained and is not scored.

**S6: the 6 s horizon, ADE 1–6 s** (BAR-R6-5 reads the os − ha0_ext column)

| seed | os ADE 1–6 s [CI] | os − ha0_ext | os − refcv4b | os − refcv5-v2 s0 |
|---|---|---|---|---|
| 0 | 3.0749 [2.7977, 3.3745] | -0.4729 [-0.7531, -0.1780] **sep** | +0.3524 [+0.1955, +0.5140] **sep** | +0.3609 [+0.2066, +0.5231] **sep** |
| 1 | 3.0877 [2.8126, 3.3835] | -0.4601 [-0.7453, -0.1658] **sep** | +0.3652 [+0.2097, +0.5275] **sep** | +0.3737 [+0.2196, +0.5358] **sep** |

**Frozen acceptance instruments, inference seed 0.**
* **T-FLIP: FAIL.** follows_FED 0.25 CI [0.1562, 0.3551], against a bar of 0.50 (refcv5-v2: 0.205).
  * true − shuffled: 0.1534 CI [0.101, 0.209], against a bar of 0.38 (refcv5-v2: 0.099).
  * n = 352 windows.
* **OBEDIENCE: FAIL.** Obeys 0.0153; 2216 of 2294 rows have NO compliant candidate. The bar is structurally unsatisfiable on this population (§3 F8).

**Frozen acceptance instruments, inference seed 1.**
* **T-FLIP: FAIL.** follows_FED 0.25 CI [0.1593, 0.3508], against a bar of 0.50 (refcv5-v2: 0.205).
  * true − shuffled: 0.1335 CI [0.0858, 0.1825], against a bar of 0.38 (refcv5-v2: 0.099).
  * n = 352 windows.
* **OBEDIENCE: FAIL.** Obeys 0.0166; 2216 of 2294 rows have NO compliant candidate. The bar is structurally unsatisfiable on this population (§3 F8).
* VOID gates, seed 0: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.5191, 'n_distinct_selected': 45, 'os_trivial_frac': 0.0189}
* VOID gates, seed 1: STOP True, ha0 True, profiles {'selection_degenerate': False, 'selection_modal_frac': 0.5187, 'n_distinct_selected': 46, 'os_trivial_frac': 0.0189}

### A4 lever panel — 4754 windows / 139 episodes, S2 ADE 0–2 s, T1 (self-action open loop); every lever is a DIFFERENT planner from the registered arm

Estimator: paired episode-cluster bootstrap, n_boot 2000, seed 0, cluster = clip. SPEC sha `a8095594d2d1…`.

**L1 inference-seed average** (sign vs single seeds guaranteed; magnitude only)

| cell | ADE m [CI] |
|---|---|
| os_avg − ha0_ext | +0.0831 [+0.0640, +0.1043] **sep** |
| os_avg − os_s0 | -0.0054 [-0.0069, -0.0040] **sep** |
| os_avg − os_s1 | -0.0054 [-0.0068, -0.0040] **sep** |
| replicate floor os_s0 − os_s1 | +0.0000 [-0.0025, +0.0027] ns |

|os_s0 − os_s1| per instant (0.5/1/1.5/2 s): mean [0.01850000023841858, 0.06629999727010727, 0.1282999962568283, 0.20559999346733093], p95 [0.03999999910593033, 0.1542000025510788, 0.33640000224113464, 0.6241000294685364] m

**L2 causal-hold blend (base `ha`)**

| seed | blend − ha0_ext | blend − base | blend − blend_shuf | blend_shuf − base | w (fold0→1 / fold1→0) | identity |
|---|---|---|---|---|---|---|
| seed0 | -0.0122 [-0.0200, -0.0042] **sep** | -0.0246 [-0.0338, -0.0151] **sep** | -0.0246 [-0.0338, -0.0151] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.2, 0.2, 0.3, 0.4] / [0.2, 0.25, 0.3, 0.4] | PASS |
| seed1 | -0.0123 [-0.0200, -0.0045] **sep** | -0.0247 [-0.0339, -0.0153] **sep** | -0.0247 [-0.0339, -0.0153] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.2, 0.2, 0.3, 0.4] / [0.2, 0.25, 0.3, 0.4] | PASS |

**L2e echo blend (base `ha0_ext`, DIAGNOSTIC ONLY)**

| seed | blend − ha0_ext | blend − base | blend − blend_shuf | blend_shuf − base | w (fold0→1 / fold1→0) | identity |
|---|---|---|---|---|---|---|
| seed0 | -0.0202 [-0.0277, -0.0125] **sep** | -0.0202 [-0.0277, -0.0125] **sep** | -0.0202 [-0.0277, -0.0125] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.15, 0.2, 0.25, 0.35] / [0.2, 0.2, 0.25, 0.35] | PASS |
| seed1 | -0.0203 [-0.0277, -0.0125] **sep** | -0.0203 [-0.0277, -0.0125] **sep** | -0.0203 [-0.0277, -0.0125] **sep** | +0.0000 [+0.0000, +0.0000] ns | [0.15, 0.2, 0.25, 0.35] / [0.15, 0.2, 0.25, 0.35] | PASS |


Full tables: `raw/step5000/TABLES_s0.md`, `TABLES_s1.md`.
