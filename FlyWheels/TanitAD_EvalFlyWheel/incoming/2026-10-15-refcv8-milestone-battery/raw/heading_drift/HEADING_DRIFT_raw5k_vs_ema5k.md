# refcv8 heading-drift localisation: RAW-5k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-17T01:23:54; trees: `RAW-5k` = C:/Users/Admin/refcv8_eval_archive/battery/step5000_unc; `EMA-5k` = C:/Users/Admin/refcv8_eval_archive/battery/step5000ema_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- RAW-5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4477 | 139 | 1.6874 | 1.3729 | 0.3144 [0.2188, 0.4222] | **SEP** |  |
| turn: straight <0.02 | 2882 | 132 | 1.3304 | 1.1019 | 0.2286 [0.1524, 0.3151] | **SEP** | 0.644 / 0.468 / -0.2411 [-0.5540, 0.0411] |
| turn: gentle 0.02-0.06 | 717 | 97 | 1.3573 | 1.0080 | 0.3493 [0.1478, 0.5996] | **SEP** | 0.160 / 0.178 / 0.0415 [-0.2818, 0.3992] |
| turn: turn 0.06-0.15 | 506 | 72 | 2.1078 | 1.9201 | 0.1877 [-0.0237, 0.4254] |  | 0.113 / 0.067 / -0.1429 [-0.4658, 0.1966] |
| turn: sharp >=0.15 | 365 | 42 | 4.1023 | 2.9937 | 1.1086 [0.5223, 1.8133] | **SEP** | 0.082 / 0.287 / 0.8647 [-0.0121, 2.0374] |
| turn: undef (GT crawling) (low n) | 7 | 6 | 26.1490 | 26.2907 | -0.1417 [-0.7251, 0.1400] |  | 0.002 / -0.001 / -0.4568 [-1.8208, -0.0510] BONF-SEP |
| speed: v0 <2 | 188 | 37 | 14.7149 | 14.4792 | 0.2357 [-0.0384, 0.6168] |  | 0.042 / 0.031 / -0.0822 [-0.5349, 0.6858] |
| speed: v0 2-5 | 516 | 58 | 2.2384 | 1.8694 | 0.3690 [-0.0601, 0.7638] |  | 0.115 / 0.135 / 0.0617 [-0.6405, 0.6889] |
| speed: v0 5-10 | 1390 | 84 | 1.7324 | 1.0092 | 0.7232 [0.5236, 0.9449] | **SEP** | 0.310 / 0.714 / 0.5928 [0.2883, 0.9204] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.7029 | 0.5744 | 0.1285 [0.0788, 0.1802] | **SEP** | 0.266 / 0.109 / -0.2535 [-0.4803, -0.0687] BONF-SEP |
| speed: v0 15-25 | 856 | 35 | 0.3689 | 0.3734 | -0.0045 [-0.0409, 0.0253] |  | 0.191 / -0.003 / -0.3943 [-0.5873, -0.2281] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.2129 | 0.1576 | 0.0553 [0.0047, 0.1176] | **SEP** | 0.075 / 0.013 / -0.2801 [-0.4595, -0.1058] BONF-SEP |
| nav: nav=follow | 2863 | 88 | 0.8685 | 0.5997 | 0.2687 [0.1667, 0.3863] | **SEP** | 0.639 / 0.547 / -0.1268 [-0.4629, 0.1884] |
| nav: nav=left (low n) | 398 | 13 | 6.6045 | 6.5255 | 0.0791 [-0.0509, 0.1846] |  | 0.089 / 0.022 / -0.2583 [-0.5389, -0.0385] BONF-SEP |
| nav: nav=right | 1216 | 38 | 2.0061 | 1.5070 | 0.4991 [0.2571, 0.7464] | **SEP** | 0.272 / 0.431 / 0.2535 [-0.1377, 0.6637] |

episode concentration: 0.777 of episodes have mean delta > 0; top-5 episodes carry 0.266 of the summed delta; delta excluding the top 5 = 0.2400 (overall 0.3144).

### S2 -- RAW-5k - EMA-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4476 | 139 | 1.6876 | 1.3872 | 0.3004 [0.2137, 0.3971] | **SEP** |  |
| turn: straight <0.02 | 2881 | 132 | 1.3306 | 1.1153 | 0.2153 [0.1427, 0.2964] | **SEP** | 0.644 / 0.461 / -0.2388 [-0.5449, 0.0277] |
| turn: gentle 0.02-0.06 | 717 | 97 | 1.3573 | 1.0248 | 0.3325 [0.1201, 0.5897] | **SEP** | 0.160 / 0.177 / 0.0382 [-0.2711, 0.3937] |
| turn: turn 0.06-0.15 | 506 | 72 | 2.1078 | 1.9015 | 0.2064 [0.0018, 0.4356] | **SEP** | 0.113 / 0.078 / -0.1061 [-0.4064, 0.2489] |
| turn: sharp >=0.15 | 365 | 42 | 4.1023 | 3.0542 | 1.0481 [0.5449, 1.6662] | **SEP** | 0.082 / 0.285 / 0.8141 [-0.0154, 1.7466] |
| turn: undef (GT crawling) (low n) | 7 | 6 | 26.1490 | 26.3012 | -0.1522 [-0.5629, 0.0420] |  | 0.002 / -0.001 / -0.4533 [-1.3696, -0.1551] BONF-SEP |
| speed: v0 <2 | 187 | 37 | 14.7901 | 14.5596 | 0.2305 [-0.1238, 0.6585] |  | 0.042 / 0.032 / -0.0730 [-0.6478, 0.7609] |
| speed: v0 2-5 | 516 | 58 | 2.2384 | 1.9040 | 0.3345 [-0.0340, 0.6920] |  | 0.115 / 0.128 / 0.0385 [-0.5698, 0.5515] |
| speed: v0 5-10 | 1390 | 84 | 1.7324 | 1.0618 | 0.6705 [0.4812, 0.8764] | **SEP** | 0.311 / 0.693 / 0.5368 [0.2393, 0.8604] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.7029 | 0.5717 | 0.1312 [0.0810, 0.1875] | **SEP** | 0.266 / 0.116 / -0.2307 [-0.4389, -0.0621] BONF-SEP |
| speed: v0 15-25 | 856 | 35 | 0.3689 | 0.3482 | 0.0207 [-0.0062, 0.0438] |  | 0.191 / 0.013 / -0.3458 [-0.5177, -0.1976] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.2129 | 0.1448 | 0.0682 [0.0246, 0.1259] | **SEP** | 0.075 / 0.017 / -0.2510 [-0.4239, -0.0955] BONF-SEP |
| nav: nav=follow | 2862 | 88 | 0.8685 | 0.6074 | 0.2611 [0.1644, 0.3748] | **SEP** | 0.639 / 0.556 / -0.1089 [-0.4291, 0.1856] |
| nav: nav=left (low n) | 398 | 13 | 6.6045 | 6.5171 | 0.0874 [-0.0762, 0.2288] |  | 0.089 / 0.026 / -0.2338 [-0.5405, 0.0324] |
| nav: nav=right | 1216 | 38 | 2.0061 | 1.5435 | 0.4626 [0.2523, 0.6885] | **SEP** | 0.272 / 0.418 / 0.2226 [-0.1338, 0.6075] |

episode concentration: 0.806 of episodes have mean delta > 0; top-5 episodes carry 0.265 of the summed delta; delta excluding the top 5 = 0.2294 (overall 0.3004).

### S2 -- EMA-5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4518 | 139 | 1.4616 | 1.4471 | 0.0145 [-0.0144, 0.0436] |  |  |
| turn: straight <0.02 | 2899 | 132 | 1.1726 | 1.1593 | 0.0133 [-0.0083, 0.0393] |  | 0.642 / 0.587 / -0.0035 [-0.1431, 0.1046] |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0252 | 1.0095 | 0.0157 [-0.0228, 0.0604] |  | 0.160 / 0.172 / 0.0013 [-0.0681, 0.0768] |
| turn: turn 0.06-0.15 | 509 | 72 | 1.9091 | 1.9295 | -0.0204 [-0.1283, 0.0816] |  | 0.113 / -0.158 / -0.0393 [-0.2345, 0.1168] |
| turn: sharp >=0.15 | 367 | 42 | 3.0631 | 2.9870 | 0.0761 [-0.1596, 0.3724] |  | 0.081 / 0.426 / 0.0671 [-0.2501, 0.5536] |
| turn: undef (GT crawling) (low n) | 22 | 18 | 16.7858 | 16.8657 | -0.0799 [-0.2458, 0.0381] |  | 0.005 / -0.027 / -0.0949 [-0.3562, 0.0649] |
| speed: v0 <2 | 229 | 38 | 13.6121 | 13.5957 | 0.0164 [-0.1129, 0.1237] |  | 0.051 / 0.057 / 0.0020 [-0.2251, 0.1838] |
| speed: v0 2-5 | 516 | 58 | 1.9040 | 1.8694 | 0.0345 [-0.1385, 0.2320] |  | 0.114 / 0.271 / 0.0226 [-0.2288, 0.3376] |
| speed: v0 5-10 | 1390 | 84 | 1.0618 | 1.0092 | 0.0526 [0.0049, 0.1051] | **SEP** | 0.308 / 1.115 / 0.0550 [-0.0315, 0.1490] |
| speed: v0 10-15 | 1192 | 64 | 0.5717 | 0.5744 | -0.0027 [-0.0357, 0.0224] |  | 0.264 / -0.049 / -0.0234 [-0.0943, 0.0530] |
| speed: v0 15-25 | 856 | 35 | 0.3482 | 0.3734 | -0.0252 [-0.0541, 0.0038] |  | 0.189 / -0.329 / -0.0490 [-0.1132, 0.0137] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1448 | 0.1576 | -0.0129 [-0.0345, 0.0044] |  | 0.074 / -0.066 / -0.0296 [-0.0889, 0.0235] |
| nav: nav=follow | 2879 | 88 | 0.6053 | 0.5982 | 0.0072 [-0.0185, 0.0332] |  | 0.637 / 0.315 / -0.0202 [-0.1407, 0.0900] |
| nav: nav=left (low n) | 406 | 13 | 7.3105 | 7.3200 | -0.0095 [-0.1057, 0.0757] |  | 0.090 / -0.059 / -0.0264 [-0.2023, 0.1061] |
| nav: nav=right | 1233 | 38 | 1.5351 | 1.4955 | 0.0396 [-0.0403, 0.1212] |  | 0.273 / 0.744 / 0.0345 [-0.0851, 0.1788] |

episode concentration: 0.525 of episodes have mean delta > 0; top-5 episodes carry 1.358 of the summed delta; delta excluding the top 5 = -0.0054 (overall 0.0145).

### S2 -- RAW-5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4463 | 139 | 0.9060 | 0.5881 | 0.3179 [0.0577, 0.8152] | **SEP** | 0.3310 [-0.0998, 1.1647] |
| seg1 | straight | 2876 | 132 | 0.9379 | 0.5077 | 0.4302 [0.0401, 1.2619] | **SEP** | 0.5356 [-0.1110, 1.9387] |
| seg1 | curved | 1582 | 101 | 0.7372 | 0.6227 | 0.1145 [0.0647, 0.1697] | **SEP** | -0.0397 [-0.1305, 0.0423] |
| seg2 | all | 4441 | 139 | 1.2042 | 0.8596 | 0.3446 [0.2036, 0.5371] | **SEP** | -0.1056 [-0.2707, 0.1152] |
| seg2 | straight | 2858 | 132 | 0.8524 | 0.5348 | 0.3177 [0.1417, 0.6382] | **SEP** | -0.1190 [-0.3116, 0.1889] |
| seg2 | curved | 1583 | 101 | 1.8393 | 1.4461 | 0.3931 [0.2382, 0.5657] | **SEP** | -0.0815 [-0.3494, 0.1608] |
| seg3 | all | 4428 | 139 | 1.5374 | 1.1659 | 0.3715 [0.2263, 0.5239] | **SEP** | -0.2039 [-0.3917, -0.0187] SEP |
| seg3 | straight | 2849 | 132 | 0.9417 | 0.7151 | 0.2265 [0.0544, 0.3736] | **SEP** | -0.2675 [-0.4268, -0.0849] SEP |
| seg3 | curved | 1579 | 101 | 2.6124 | 1.9792 | 0.6331 [0.3750, 0.9106] | **SEP** | -0.0893 [-0.4859, 0.2677] |
| seg4 | all | 4409 | 139 | 1.9506 | 1.4133 | 0.5373 [0.3551, 0.7416] | **SEP** | -0.2189 [-0.5072, 0.1356] |
| seg4 | straight | 2836 | 131 | 1.1292 | 0.7164 | 0.4129 [0.2524, 0.6157] | **SEP** | -0.2392 [-0.5439, 0.2463] |
| seg4 | curved | 1571 | 101 | 3.4340 | 2.6704 | 0.7636 [0.3977, 1.1575] | **SEP** | -0.1835 [-0.7138, 0.3091] |

### S2 -- RAW-5k - EMA-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4462 | 139 | 0.9061 | 0.4715 | 0.4346 [0.0571, 1.1770] | **SEP** | 0.3144 [-0.0717, 1.0641] |
| seg1 | straight | 2875 | 132 | 0.9379 | 0.3224 | 0.6155 [0.0399, 1.8685] | **SEP** | 0.5053 [-0.0768, 1.7671] |
| seg1 | curved | 1582 | 101 | 0.7372 | 0.6299 | 0.1073 [0.0644, 0.1515] | **SEP** | -0.0316 [-0.0991, 0.0309] |
| seg2 | all | 4441 | 139 | 1.2042 | 0.7801 | 0.4240 [0.2077, 0.7694] | **SEP** | 0.0252 [-0.2114, 0.4049] |
| seg2 | straight | 2858 | 132 | 0.8524 | 0.4124 | 0.4400 [0.1416, 1.0389] | **SEP** | 0.0781 [-0.2350, 0.6704] |
| seg2 | curved | 1583 | 101 | 1.8393 | 1.4441 | 0.3952 [0.2528, 0.5495] | **SEP** | -0.0702 [-0.2874, 0.1236] |
| seg3 | all | 4428 | 139 | 1.5374 | 1.0997 | 0.4378 [0.3020, 0.5884] | **SEP** | -0.0710 [-0.3042, 0.2539] |
| seg3 | straight | 2849 | 132 | 0.9417 | 0.6000 | 0.3417 [0.2116, 0.5323] | **SEP** | -0.0691 [-0.3442, 0.3982] |
| seg3 | curved | 1579 | 101 | 2.6124 | 2.0012 | 0.6112 [0.3681, 0.8834] | **SEP** | -0.0744 [-0.4161, 0.2349] |
| seg4 | all | 4409 | 139 | 1.9506 | 1.4776 | 0.4731 [0.3235, 0.6434] | **SEP** | -0.1007 [-0.4126, 0.3366] |
| seg4 | straight | 2836 | 131 | 1.1292 | 0.8010 | 0.3282 [0.2225, 0.4580] | **SEP** | -0.0868 [-0.4412, 0.5153] |
| seg4 | curved | 1571 | 101 | 3.4340 | 2.6982 | 0.7358 [0.4060, 1.0985] | **SEP** | -0.1267 [-0.5763, 0.3068] |

### S2 -- EMA-5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4480 | 139 | 0.5111 | 0.6679 | -0.1568 [-0.4850, 0.0109] |  | -0.0228 [-0.0357, -0.0086] SEP |
| seg1 | straight | 2879 | 132 | 0.3846 | 0.5697 | -0.1851 [-0.6013, 0.0075] |  | -0.0944 [-0.2362, -0.0254] SEP |
| seg1 | curved | 1587 | 101 | 0.6295 | 0.6227 | 0.0068 [-0.0152, 0.0299] |  | -0.0070 [-0.0415, 0.0236] |
| seg2 | all | 4475 | 139 | 0.8589 | 0.9381 | -0.0792 [-0.2477, 0.0180] |  | -0.2088 [-0.5428, -0.0223] SEP |
| seg2 | straight | 2884 | 132 | 0.5354 | 0.6569 | -0.1215 [-0.4025, 0.0168] |  | -0.3195 [-0.8844, -0.0550] SEP |
| seg2 | curved | 1591 | 101 | 1.4454 | 1.4478 | -0.0025 [-0.0596, 0.0590] |  | -0.0082 [-0.0938, 0.0671] |
| seg3 | all | 4465 | 139 | 1.1796 | 1.2049 | -0.0253 [-0.1154, 0.0381] |  | -0.0891 [-0.1850, -0.0153] SEP |
| seg3 | straight | 2876 | 132 | 0.7216 | 0.7739 | -0.0523 [-0.2033, 0.0350] |  | -0.1339 [-0.2883, -0.0496] SEP |
| seg3 | curved | 1589 | 101 | 2.0085 | 1.9850 | 0.0235 [-0.0644, 0.1180] |  | -0.0081 [-0.1186, 0.0975] |
| seg4 | all | 4450 | 139 | 1.6406 | 1.4938 | 0.1468 [-0.0147, 0.4187] |  | -0.0346 [-0.1222, 0.0707] |
| seg4 | straight | 2863 | 131 | 1.0484 | 0.8392 | 0.2092 [-0.0094, 0.6385] |  | -0.0263 [-0.1235, 0.1296] |
| seg4 | curved | 1579 | 101 | 2.7195 | 2.6846 | 0.0349 [-0.1000, 0.1790] |  | -0.0491 [-0.2181, 0.1149] |

### S2 -- RAW-5k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 423 | 47 | 6.0219 | 5.7552 | 0.2668 [0.0080, 0.5176] | **SEP** |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.7037 | 0.3674 | 0.3363 [0.2267, 0.4744] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.2320 | 0.1947 | 0.0373 [0.0151, 0.0635] | **SEP** |  |
| gentle 0.02-0.06 | v0 <5 | 108 | 28 | 1.8235 | 1.7810 | 0.0425 [-0.7072, 0.6985] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.5205 | 0.9711 | 0.5494 [0.2871, 0.8498] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.5087 | 0.5590 | -0.0503 [-0.1210, 0.0228] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 76 | 25 | 4.8136 | 5.2289 | -0.4153 [-1.2993, 0.2310] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.8281 | 1.4275 | 0.4006 [0.1713, 0.7447] | **SEP** |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 0.8469 | 0.9719 | -0.1250 [-0.3873, 0.0099] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 90 | 16 | 6.9815 | 5.3167 | 1.6648 [0.2144, 3.2412] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 3.2776 | 2.2857 | 0.9919 [0.4231, 1.6873] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.2565 | 1.3866 | -0.1301 [-0.2728, 0.1394] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 7 | 6 | 26.1490 | 26.2907 | -0.1417 [-0.7251, 0.1400] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW-5k s0-s1 [CI] | EMA-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0161 [-0.0334, 0.0008] | 0.0080 [-0.0053, 0.0227] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0212 [-0.0379, -0.0051] SEP | -0.0025 [-0.0152, 0.0107] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | -0.0019 [-0.0291, 0.0225] | 0.0222 [-0.0046, 0.0560] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | 0.0382 [-0.0075, 0.0961] | 0.0029 [-0.0637, 0.0729] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0785 [-0.2171, 0.0460] | 0.0710 [-0.0077, 0.1690] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.0511 [-0.2590, 0.0676] | -0.0061 [-0.1265, 0.1043] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | 0.0407 [-0.0974, 0.1929] | 0.0507 [-0.0497, 0.1881] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.1039 [-0.2163, -0.0093] SEP | 0.0044 [-0.0592, 0.0734] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0220 [-0.0551, 0.0157] | 0.0164 [-0.0160, 0.0478] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0040 [-0.0105, 0.0180] | -0.0014 [-0.0124, 0.0091] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | 0.0020 [-0.0062, 0.0111] | 0.0006 [-0.0037, 0.0050] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | -0.0065 [-0.0241, 0.0083] | 0.0018 [-0.0030, 0.0066] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0127 [-0.0307, 0.0056] | 0.0030 [-0.0075, 0.0161] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | -0.0176 [-0.0773, 0.0512] | 0.0540 [-0.0031, 0.1154] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0237 [-0.0680, 0.0187] | 0.0044 [-0.0318, 0.0433] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- RAW-5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.3144 | 0.3007 / 0.3274 | True / True / True | 0.0161 / 0.0110 | 19.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.2286 | 0.2171 / 0.2396 | True / True / True | 0.0212 / 0.0019 | 10.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.3493 | 0.3529 / 0.3424 | True / True / True | 0.0019 / 0.0161 | 21.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | 0.1877 | 0.1943 / 0.1810 | False / False / False | 0.0382 / 0.0331 | 4.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 1.1086 | 1.0122 / 1.2050 | True / True / True | 0.0785 / 0.1080 | 10.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.1417 | -0.0770 / -0.2064 | False / False / False | 0.0511 / 0.0237 | 2.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.2357 | 0.2160 / 0.2374 | False / False / False | 0.0407 / 0.0573 | 4.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.3690 | 0.2811 / 0.4569 | False / True / False | 0.1039 / 0.0719 | 3.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.7232 | 0.7119 / 0.7345 | True / True / True | 0.0220 / 0.0006 | 32.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1285 | 0.1305 / 0.1264 | True / True / True | 0.0040 / 0.0001 | 32.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.0045 | -0.0023 / -0.0067 | False / False / False | 0.0020 / 0.0023 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0553 | 0.0524 / 0.0583 | True / True / True | 0.0065 / 0.0005 | 8.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.2687 | 0.2641 / 0.2731 | True / True / True | 0.0127 / 0.0036 | 21.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0791 | 0.0435 / 0.1132 | False / False / False | 0.0176 / 0.0608 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.4991 | 0.4708 / 0.5253 | True / True / True | 0.0237 / 0.0283 | 17.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- RAW-5k - EMA-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.3004 | 0.2883 / 0.3122 | True / True / True | 0.0161 / 0.0080 | 18.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.2153 | 0.2059 / 0.2243 | True / True / True | 0.0212 / 0.0025 | 10.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.3325 | 0.3199 / 0.3452 | True / True / True | 0.0019 / 0.0222 | 15.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | 0.2064 | 0.2240 / 0.1887 | True / False / True | 0.0382 / 0.0029 | 5.4 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|sharp >=0.15 | 1.0481 | 0.9739 / 1.1223 | True / True / True | 0.0785 / 0.0710 | 13.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.1522 | -0.1206 / -0.1838 | False / True / False | 0.0511 / 0.0061 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.2305 | 0.2214 / 0.2336 | False / False / False | 0.0407 / 0.0507 | 4.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.3345 | 0.2803 / 0.3886 | False / True / False | 0.1039 / 0.0044 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.6705 | 0.6514 / 0.6897 | True / True / True | 0.0220 / 0.0164 | 30.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1312 | 0.1338 / 0.1285 | True / True / True | 0.0040 / 0.0014 | 33.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.0207 | 0.0214 / 0.0200 | False / False / False | 0.0020 / 0.0006 | 10.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0682 | 0.0640 / 0.0723 | True / True / True | 0.0065 / 0.0018 | 10.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.2611 | 0.2532 / 0.2689 | True / True / True | 0.0127 / 0.0030 | 20.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0874 | 0.0539 / 0.1196 | False / False / False | 0.0176 / 0.0540 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.4626 | 0.4479 / 0.4772 | True / True / True | 0.0237 / 0.0044 | 19.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- EMA-5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0145 | 0.0133 / 0.0157 | False / False / False | 0.0080 / 0.0110 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0133 | 0.0112 / 0.0155 | False / False / False | 0.0025 / 0.0019 | 5.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0157 | 0.0348 / -0.0035 | False / False / False | 0.0222 / 0.0161 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0204 | -0.0337 / -0.0070 | False / False / False | 0.0029 / 0.0331 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.0761 | 0.0576 / 0.0947 | False / False / False | 0.0710 / 0.1080 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.0799 | -0.0575 / -0.1099 | False / False / False | 0.0061 / 0.0237 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0164 | 0.0174 / 0.0157 | False / False / False | 0.0507 / 0.0573 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0345 | 0.0008 / 0.0683 | False / False / False | 0.0044 / 0.0719 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0526 | 0.0605 / 0.0448 | True / False / True | 0.0164 / 0.0006 | 3.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 10-15 | -0.0027 | -0.0033 / -0.0021 | False / False / False | 0.0014 / 0.0001 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0252 | -0.0238 / -0.0267 | False / False / False | 0.0006 / 0.0023 | 11.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0129 | -0.0117 / -0.0140 | False / False / False | 0.0018 / 0.0005 | 7.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0072 | 0.0106 / 0.0038 | False / False / False | 0.0030 / 0.0036 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.0095 | -0.0111 / -0.0080 | False / False / False | 0.0540 / 0.0608 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0396 | 0.0278 / 0.0514 | False / False / False | 0.0044 / 0.0283 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1582): RAW-5k bias -0.142, refcv7-50400 bias +0.083, EMA-5k bias -0.008; RAW-5k MAE 0.737, refcv7-50400 MAE 0.623, EMA-5k MAE 0.630
* seg2 (n 1583): RAW-5k bias -0.674, refcv7-50400 bias -0.090, EMA-5k bias -0.277; RAW-5k MAE 1.839, refcv7-50400 MAE 1.446, EMA-5k MAE 1.444
* seg3 (n 1579): RAW-5k bias -1.363, refcv7-50400 bias -0.607, EMA-5k bias -0.788; RAW-5k MAE 2.612, refcv7-50400 MAE 1.979, EMA-5k MAE 2.001
* seg4 (n 1571): RAW-5k bias -1.754, refcv7-50400 bias -0.697, EMA-5k bias -0.996; RAW-5k MAE 3.434, refcv7-50400 MAE 2.670, EMA-5k MAE 2.698
**sharp (>= 0.15)**

* seg1 (n 364): RAW-5k bias -0.768, refcv7-50400 bias +0.010, EMA-5k bias -0.267; RAW-5k MAE 1.106, refcv7-50400 MAE 0.917, EMA-5k MAE 0.881
* seg2 (n 364): RAW-5k bias -2.470, refcv7-50400 bias -0.515, EMA-5k bias -1.076; RAW-5k MAE 3.189, refcv7-50400 MAE 2.410, EMA-5k MAE 2.362
* seg3 (n 363): RAW-5k bias -4.267, refcv7-50400 bias -1.835, EMA-5k bias -2.362; RAW-5k MAE 5.210, refcv7-50400 MAE 3.631, EMA-5k MAE 3.770
* seg4 (n 361): RAW-5k bias -5.233, refcv7-50400 bias -1.787, EMA-5k bias -2.665; RAW-5k MAE 6.857, refcv7-50400 MAE 4.854, EMA-5k MAE 5.057
**curved & v0 5-10**

* seg1 (n 627): RAW-5k bias -0.326, refcv7-50400 bias -0.038, EMA-5k bias -0.134; RAW-5k MAE 0.766, refcv7-50400 MAE 0.535, EMA-5k MAE 0.553
* seg2 (n 627): RAW-5k bias -1.091, refcv7-50400 bias -0.282, EMA-5k bias -0.477; RAW-5k MAE 2.165, refcv7-50400 MAE 1.363, EMA-5k MAE 1.377
* seg3 (n 627): RAW-5k bias -1.942, refcv7-50400 bias -0.871, EMA-5k bias -1.035; RAW-5k MAE 3.294, refcv7-50400 MAE 2.019, EMA-5k MAE 2.036
* seg4 (n 627): RAW-5k bias -2.505, refcv7-50400 bias -1.194, EMA-5k bias -1.432; RAW-5k MAE 4.229, refcv7-50400 MAE 2.730, EMA-5k MAE 2.759
**sharp & v0 5-10**

* seg1 (n 164): RAW-5k bias -0.772, refcv7-50400 bias +0.103, EMA-5k bias -0.151; RAW-5k MAE 1.127, refcv7-50400 MAE 0.857, EMA-5k MAE 0.787
* seg2 (n 164): RAW-5k bias -2.771, refcv7-50400 bias -0.422, EMA-5k bias -0.929; RAW-5k MAE 3.359, refcv7-50400 MAE 2.243, EMA-5k MAE 2.117
* seg3 (n 164): RAW-5k bias -5.059, refcv7-50400 bias -2.032, EMA-5k bias -2.456; RAW-5k MAE 5.585, refcv7-50400 MAE 3.465, EMA-5k MAE 3.412
* seg4 (n 164): RAW-5k bias -6.283, refcv7-50400 bias -2.403, EMA-5k bias -3.048; RAW-5k MAE 7.248, refcv7-50400 MAE 4.635, EMA-5k MAE 4.706

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1571 / 1571 | 10.2 | RAW-5k -0.2119 [-0.3032, -0.1364]; refcv7-50400 -0.0831 [-0.1540, -0.0266]; EMA-5k -0.1176 [-0.2034, -0.0517] | RAW-5k -0.1288 [-0.1740, -0.0866] SEP; EMA-5k -0.0345 [-0.0582, -0.0161] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1586 / 4733 | 7.2 | RAW-5k -0.2715 [-0.3918, -0.1688]; refcv7-50400 -0.1456 [-0.2647, -0.0512]; EMA-5k -0.1792 [-0.3027, -0.0799] | RAW-5k -0.1259 [-0.1706, -0.0842] SEP; EMA-5k -0.0336 [-0.0540, -0.0180] SEP |
| sharp (>= 0.15) | last segment | 361 / 361 | 25.4 | RAW-5k -0.2340 [-0.3402, -0.1455]; refcv7-50400 -0.0908 [-0.1741, -0.0240]; EMA-5k -0.1283 [-0.2304, -0.0520] | RAW-5k -0.1432 [-0.1967, -0.0953] SEP; EMA-5k -0.0375 [-0.0639, -0.0169] SEP |
| sharp (>= 0.15) | segments 2..K | 364 / 1088 | 18.0 | RAW-5k -0.2504 [-0.3566, -0.1626]; refcv7-50400 -0.1030 [-0.1861, -0.0347]; EMA-5k -0.1409 [-0.2423, -0.0644] | RAW-5k -0.1474 [-0.1994, -0.1007] SEP; EMA-5k -0.0379 [-0.0629, -0.0190] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | RAW-5k -0.2329 [-0.3196, -0.1651]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; EMA-5k -0.1128 [-0.1915, -0.0594] | RAW-5k -0.1404 [-0.2057, -0.0827] SEP; EMA-5k -0.0202 [-0.0429, +0.0055] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | RAW-5k -0.2510 [-0.3376, -0.1825]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; EMA-5k -0.1233 [-0.2007, -0.0703] | RAW-5k -0.1517 [-0.2132, -0.0962] SEP; EMA-5k -0.0240 [-0.0453, +0.0004] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | RAW-5k -0.2390 [-0.3324, -0.1610]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; EMA-5k -0.1116 [-0.1998, -0.0515] | RAW-5k -0.1477 [-0.2206, -0.0862] SEP; EMA-5k -0.0203 [-0.0436, +0.0069] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | RAW-5k -0.2546 [-0.3487, -0.1783]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; EMA-5k -0.1186 [-0.2069, -0.0577] | RAW-5k -0.1591 [-0.2286, -0.0991] SEP; EMA-5k -0.0231 [-0.0459, +0.0041] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW-5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4477 | 0.3144 | 0.05659 | 0.11630 | 0.628 / 0.372 / 0.000 | 505.1 / 1407.7 |
| turn|straight <0.02 | 2882 | 0.2286 | 0.04620 | 0.08233 | 0.646 / 0.354 / 0.000 | 246.1 / 658.7 |
| turn|gentle 0.02-0.06 | 717 | 0.3493 | 0.07606 | 0.12011 | 0.586 / 0.414 / 0.000 | 79.0 / 250.4 |
| turn|turn 0.06-0.15 | 506 | 0.1877 | 0.10029 | 0.12260 | 0.579 / 0.421 / 0.000 | 44.6 / 95.0 |
| turn|sharp >=0.15 | 365 | 1.1086 | 0.37790 | 0.72790 | 0.641 / 0.359 / 0.000 | 66.0 / 404.6 |
| turn|undef (GT crawling) | 7 | -0.1417 | 0.06701 | 0.06697 | 0.714 / 0.286 / 0.000 | 0.2 / -1.0 |
| speed|v0 <2 | 188 | 0.2357 | 0.07468 | 0.11328 | 0.585 / 0.415 / 0.000 | 19.2 / 44.3 |
| speed|v0 2-5 | 516 | 0.3690 | 0.21714 | 0.32133 | 0.696 / 0.304 / 0.000 | 87.3 / 190.4 |
| speed|v0 5-10 | 1390 | 0.7232 | 0.13631 | 0.27912 | 0.655 / 0.345 / 0.000 | 168.8 / 1005.2 |
| speed|v0 10-15 | 1192 | 0.1285 | 0.06226 | 0.07909 | 0.633 / 0.367 / 0.000 | 35.3 / 153.1 |
| speed|v0 15-25 | 856 | -0.0045 | 0.00780 | 0.01352 | 0.556 / 0.444 / 0.000 | 8.4 / -3.9 |
| speed|v0 >=25 | 335 | 0.0553 | 0.02018 | 0.02611 | 0.606 / 0.394 / 0.000 | 4.8 / 18.5 |
| nav|nav=follow | 2863 | 0.2687 | 0.03932 | 0.07526 | 0.608 / 0.392 / 0.000 | 312.4 / 769.4 |
| nav|nav=left | 398 | 0.0791 | 0.06164 | 0.09968 | 0.643 / 0.357 / 0.000 | 14.8 / 31.5 |
| nav|nav=right | 1216 | 0.4991 | 0.14900 | 0.25426 | 0.672 / 0.328 / 0.000 | 156.3 / 606.9 |
**RAW-5k - EMA-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4476 | 0.3004 | 0.05488 | 0.11468 | 0.640 / 0.360 / 0.000 | 476.7 / 1344.7 |
| turn|straight <0.02 | 2881 | 0.2153 | 0.04503 | 0.08077 | 0.649 / 0.351 / 0.000 | 245.7 / 620.3 |
| turn|gentle 0.02-0.06 | 717 | 0.3325 | 0.07728 | 0.11717 | 0.608 / 0.392 / 0.000 | 79.1 / 238.4 |
| turn|turn 0.06-0.15 | 506 | 0.2064 | 0.08798 | 0.12120 | 0.597 / 0.403 / 0.000 | 43.6 / 104.4 |
| turn|sharp >=0.15 | 365 | 1.0481 | 0.43624 | 0.73413 | 0.688 / 0.312 / 0.000 | 54.9 / 382.6 |
| turn|undef (GT crawling) | 7 | -0.1522 | -0.04881 | -0.01766 | 0.429 / 0.571 / 0.000 | 0.1 / -1.1 |
| speed|v0 <2 | 187 | 0.2305 | 0.05976 | 0.12960 | 0.610 / 0.390 / 0.000 | 19.1 / 43.1 |
| speed|v0 2-5 | 516 | 0.3345 | 0.23044 | 0.30644 | 0.692 / 0.308 / 0.000 | 69.5 / 172.6 |
| speed|v0 5-10 | 1390 | 0.6705 | 0.12568 | 0.24769 | 0.663 / 0.337 / 0.000 | 167.4 / 932.0 |
| speed|v0 10-15 | 1192 | 0.1312 | 0.05167 | 0.07858 | 0.637 / 0.363 / 0.000 | 34.8 / 156.4 |
| speed|v0 15-25 | 856 | 0.0207 | 0.02028 | 0.02818 | 0.597 / 0.403 / 0.000 | 7.4 / 17.7 |
| speed|v0 >=25 | 335 | 0.0682 | 0.01259 | 0.02926 | 0.600 / 0.400 / 0.000 | 4.7 / 22.8 |
| nav|nav=follow | 2862 | 0.2611 | 0.03948 | 0.08206 | 0.625 / 0.375 / 0.000 | 302.6 / 747.4 |
| nav|nav=left | 398 | 0.0874 | 0.04447 | 0.07838 | 0.621 / 0.379 / 0.000 | 13.6 / 34.8 |
| nav|nav=right | 1216 | 0.4626 | 0.13804 | 0.23657 | 0.679 / 0.321 / 0.000 | 135.9 / 562.5 |
**EMA-5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4518 | 0.0145 | 0.00098 | 0.00031 | 0.504 / 0.496 / 0.000 | 203.4 / 65.6 |
| turn|straight <0.02 | 2899 | 0.0133 | 0.00000 | -0.00155 | 0.500 / 0.500 / 0.000 | 88.7 / 38.5 |
| turn|gentle 0.02-0.06 | 721 | 0.0157 | -0.00091 | 0.00216 | 0.497 / 0.503 / 0.000 | 21.8 / 11.3 |
| turn|turn 0.06-0.15 | 509 | -0.0204 | 0.00607 | 0.00126 | 0.513 / 0.487 / 0.000 | 21.0 / -10.4 |
| turn|sharp >=0.15 | 367 | 0.0761 | 0.05080 | 0.02528 | 0.537 / 0.463 / 0.000 | 38.3 / 27.9 |
| turn|undef (GT crawling) | 22 | -0.0799 | -0.01009 | -0.00502 | 0.455 / 0.545 / 0.000 | 0.3 / -1.8 |
| speed|v0 <2 | 229 | 0.0164 | -0.00388 | 0.00269 | 0.480 / 0.520 / 0.000 | 10.1 / 3.8 |
| speed|v0 2-5 | 516 | 0.0345 | 0.00515 | 0.01925 | 0.508 / 0.492 / 0.000 | 44.3 / 17.8 |
| speed|v0 5-10 | 1390 | 0.0526 | 0.00582 | 0.01205 | 0.519 / 0.481 / 0.000 | 79.0 / 73.2 |
| speed|v0 10-15 | 1192 | -0.0027 | 0.00432 | 0.00316 | 0.523 / 0.477 / 0.000 | 16.0 / -3.2 |
| speed|v0 15-25 | 856 | -0.0252 | -0.00653 | -0.01192 | 0.464 / 0.536 / 0.000 | 6.7 / -21.6 |
| speed|v0 >=25 | 335 | -0.0129 | -0.00237 | -0.00258 | 0.484 / 0.516 / 0.000 | 0.4 / -4.3 |
| nav|nav=follow | 2879 | 0.0072 | -0.00233 | -0.00610 | 0.490 / 0.510 / 0.000 | 86.8 / 20.7 |
| nav|nav=left | 406 | -0.0095 | 0.01469 | 0.01876 | 0.559 / 0.441 / 0.000 | 15.9 / -3.9 |
| nav|nav=right | 1233 | 0.0396 | 0.00632 | 0.01408 | 0.517 / 0.483 / 0.000 | 87.5 / 48.8 |

### S2 -- the episodes carrying the summed delta (panel file index only)

**RAW-5k - refcv7-50400** (sum 1407.7 over 139 episodes)

* ep116: n 33, share 0.067, mean delta 2.8389, sharp-window frac 0.00, curved frac 0.24, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 33}
* ep023: n 34, share 0.055, mean delta 2.2945, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep000: n 35, share 0.048, mean delta 1.9399, sharp-window frac 0.20, curved frac 0.46, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 35}
* ep011: n 34, share 0.048, mean delta 1.9770, sharp-window frac 0.00, curved frac 0.88, mean v0 6.6, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep111: n 34, share 0.048, mean delta 1.9764, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share 0.045, mean delta 1.8069, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
**RAW-5k - EMA-5k** (sum 1344.7 over 139 episodes)

* ep116: n 33, share 0.068, mean delta 2.7561, sharp-window frac 0.00, curved frac 0.24, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 33}
* ep023: n 34, share 0.052, mean delta 2.0421, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep011: n 34, share 0.051, mean delta 2.0149, sharp-window frac 0.00, curved frac 0.88, mean v0 6.6, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share 0.048, mean delta 1.8312, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep000: n 35, share 0.048, mean delta 1.8259, sharp-window frac 0.20, curved frac 0.46, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 35}
* ep111: n 34, share 0.047, mean delta 1.8592, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
**EMA-5k - refcv7-50400** (sum 65.6 over 139 episodes)

* ep013: n 34, share 0.527, mean delta 1.0183, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep075: n 34, share 0.342, mean delta 0.6610, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep067: n 34, share 0.165, mean delta 0.3191, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep021: n 34, share 0.162, mean delta 0.3134, sharp-window frac 0.15, curved frac 0.53, mean v0 5.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep012: n 34, share 0.161, mean delta 0.3100, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep023: n 34, share 0.131, mean delta 0.2524, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW-5k vs EMA-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1156 / 0.0854 [0.0565, 0.1185] | 46 / 0.9054 [0.4445, 1.5425] |
| only new agrees | 395 / 0.0302 [-0.0452, 0.1073] | 17 / 1.1438 [0.4345, 1.8407] |
| only earlier agrees | 725 / 0.4653 [0.2986, 0.6537] | 52 / 1.6970 [0.4951, 2.9777] |
| neither agrees | 2203 / 0.3829 [0.2498, 0.5248] | 250 / 0.8245 [0.3349, 1.3686] |
**RAW-5k vs EMA-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1161 / 0.0870 [0.0562, 0.1215] | 42 / 1.0022 [0.4743, 1.7680] |
| only new agrees | 393 / -0.0037 [-0.1224, 0.0995] | 26 / 0.3776 [-0.5523, 1.2470] |
| only earlier agrees | 706 / 0.5396 [0.3590, 0.7482] | 59 / 1.9583 [0.7900, 3.2259] |
| neither agrees | 2217 / 0.4137 [0.2882, 0.5565] | 238 / 1.0176 [0.4281, 1.6814] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW-5k vs EMA-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4464 | 0.5097 [0.3343, 0.7032] | 0.0377 [-0.0311, 0.1106] | 0.4734 [0.3102, 0.6599] SEP |
| turn|straight <0.02 | 2874 | 0.4994 [0.2834, 0.7683] | 0.0066 [-0.0273, 0.0472] | 0.4928 [0.2762, 0.7529] SEP |
| turn|gentle 0.02-0.06 | 718 | 0.4875 [0.2402, 0.7901] | -0.0222 [-0.1579, 0.1103] | 0.5100 [0.2463, 0.8304] SEP |
| turn|turn 0.06-0.15 | 502 | 0.3058 [0.0388, 0.6665] | 0.1061 [-0.0326, 0.3074] | 0.1994 [-0.0748, 0.5230] |
| turn|sharp >=0.15 | 364 | 0.9252 [0.3435, 1.7333] | 0.3019 [-0.2941, 0.7995] | 0.6364 [-0.0158, 1.4272] |
| turn|undef (GT crawling) | 6 | -0.0099 [-0.0477, 0.0135] | 0.1364 [-0.0364, 0.3356] | -0.1847 [-0.4459, 0.0718] |
| speed|v0 <2 | 175 | 0.0840 [-0.3249, 0.5254] | 0.2537 [0.0988, 0.4898] | -0.1867 [-0.7699, 0.3200] |
| speed|v0 2-5 | 516 | 1.9375 [1.0587, 2.9928] | 0.0959 [-0.3776, 0.5625] | 1.8416 [0.9969, 2.7420] SEP |
| speed|v0 5-10 | 1390 | 0.8818 [0.5827, 1.1816] | 0.0594 [-0.0384, 0.1665] | 0.8223 [0.5429, 1.1235] SEP |
| speed|v0 10-15 | 1192 | -0.0042 [-0.0371, 0.0293] | -0.0114 [-0.0289, 0.0049] | 0.0071 [-0.0275, 0.0437] |
| speed|v0 15-25 | 856 | 0.0134 [-0.0080, 0.0330] | -0.0039 [-0.0198, 0.0083] | 0.0173 [-0.0073, 0.0428] |
| speed|v0 >=25 | 335 | 0.0864 [0.0064, 0.1941] | -0.0007 [-0.0094, 0.0106] | 0.0872 [0.0070, 0.1949] SEP |
**RAW-5k vs EMA-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* RAW-5k vs EMA-5k|seed0: anchor flip rate 0.505; heading delta on flipped windows 0.4697 [0.3137, 0.6411] (n 2313) vs unflipped 0.0946 [0.0661, 0.1244]; goal_dist_sel proxy new-earlier 0.3238 [0.0335, 0.6104]; corr(window dHeading, window dGoalDist) -0.040
* RAW-5k vs EMA-5k|seed1: anchor flip rate 0.511; heading delta on flipped windows 0.5052 [0.3598, 0.6761] (n 2343) vs unflipped 0.1003 [0.0699, 0.1326]; goal_dist_sel proxy new-earlier 0.3248 [0.0372, 0.6078]; corr(window dHeading, window dGoalDist) -0.055

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- RAW-5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3483 | 139 | 3.7109 | 2.9801 | 0.7308 [0.4890, 0.9911] | **SEP** |  |
| turn: straight <0.02 | 2068 | 122 | 1.6336 | 1.1404 | 0.4932 [0.3705, 0.6216] | **SEP** | 0.594 / 0.401 / -0.5849 [-1.4322, 0.3072] |
| turn: gentle 0.02-0.06 | 673 | 76 | 3.1168 | 2.7201 | 0.3967 [-0.2284, 0.8792] |  | 0.193 / 0.105 / -0.4141 [-1.5493, 0.3704] |
| turn: turn 0.06-0.15 | 431 | 59 | 7.0678 | 5.9824 | 1.0854 [0.6222, 1.6244] | **SEP** | 0.124 / 0.184 / 0.4047 [-0.3136, 1.3197] |
| turn: sharp >=0.15 | 301 | 33 | 14.0153 | 11.3957 | 2.6196 [0.7308, 4.7111] | **SEP** | 0.086 / 0.310 / 2.0675 [-0.5352, 5.5492] |
| turn: undef (GT crawling) (low n) | 10 | 8 | 18.4175 | 18.1990 | 0.2185 [-0.0517, 0.5337] |  | 0.003 / 0.001 / -0.5138 [-1.2059, 0.1535] |
| speed: v0 <2 | 145 | 23 | 10.2966 | 10.4679 | -0.1713 [-1.1511, 0.6205] |  | 0.042 / -0.010 / -0.9413 [-3.0956, 0.4219] |
| speed: v0 2-5 | 422 | 52 | 5.0329 | 4.1501 | 0.8828 [-0.3267, 1.9795] |  | 0.121 / 0.146 / 0.1729 [-1.9172, 2.0463] |
| speed: v0 5-10 | 1040 | 79 | 4.8774 | 3.4941 | 1.3833 [0.8993, 1.9672] | **SEP** | 0.299 / 0.565 / 0.9302 [0.1992, 1.8886] BONF-SEP |
| speed: v0 10-15 | 952 | 62 | 3.0719 | 2.4870 | 0.5848 [0.2929, 0.9461] | **SEP** | 0.273 / 0.219 / -0.2009 [-0.8254, 0.6035] |
| speed: v0 15-25 | 679 | 34 | 1.7173 | 1.4456 | 0.2717 [0.1464, 0.4286] | **SEP** | 0.195 / 0.072 / -0.5703 [-1.0723, -0.0844] BONF-SEP |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5923 | 0.5193 | 0.0730 [-0.0597, 0.2210] |  | 0.070 / 0.007 / -0.7076 [-1.1663, -0.2062] BONF-SEP |
| nav: nav=follow | 2228 | 88 | 2.6331 | 1.9842 | 0.6489 [0.3939, 0.9699] | **SEP** | 0.640 / 0.568 / -0.2272 [-0.9770, 0.6146] |
| nav: nav=left (low n) | 308 | 13 | 7.0938 | 6.8461 | 0.2477 [-0.0816, 0.5464] |  | 0.088 / 0.030 / -0.5300 [-1.2573, 0.1026] |
| nav: nav=right | 947 | 38 | 5.1463 | 4.0657 | 1.0806 [0.4268, 1.6263] | **SEP** | 0.272 / 0.402 / 0.4804 [-0.6018, 1.2904] |

episode concentration: 0.813 of episodes have mean delta > 0; top-5 episodes carry 0.293 of the summed delta; delta excluding the top 5 = 0.5371 (overall 0.7308).

### S6 -- RAW-5k - EMA-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3483 | 139 | 3.7109 | 3.0852 | 0.6257 [0.4297, 0.8364] | **SEP** |  |
| turn: straight <0.02 | 2068 | 122 | 1.6336 | 1.2108 | 0.4228 [0.3151, 0.5444] | **SEP** | 0.594 / 0.401 / -0.4995 [-1.1915, 0.1650] |
| turn: gentle 0.02-0.06 | 673 | 76 | 3.1168 | 2.7177 | 0.3991 [-0.1439, 0.8234] |  | 0.193 / 0.123 / -0.2809 [-1.2693, 0.4367] |
| turn: turn 0.06-0.15 | 431 | 59 | 7.0678 | 6.1256 | 0.9421 [0.4803, 1.4269] | **SEP** | 0.124 / 0.186 / 0.3611 [-0.3178, 1.2289] |
| turn: sharp >=0.15 | 301 | 33 | 14.0153 | 11.9238 | 2.0915 [0.4230, 3.8627] | **SEP** | 0.086 / 0.289 / 1.6044 [-0.8732, 4.3319] |
| turn: undef (GT crawling) (low n) | 10 | 8 | 18.4175 | 18.3381 | 0.0794 [-0.2556, 0.4033] |  | 0.003 / 0.000 / -0.5479 [-1.2561, 0.1262] |
| speed: v0 <2 | 145 | 23 | 10.2966 | 10.3895 | -0.0928 [-1.0348, 0.9014] |  | 0.042 / -0.006 / -0.7498 [-2.5945, 1.0079] |
| speed: v0 2-5 | 422 | 52 | 5.0329 | 4.3226 | 0.7103 [-0.3621, 1.6829] |  | 0.121 / 0.138 / 0.0962 [-1.6270, 1.6621] |
| speed: v0 5-10 | 1040 | 79 | 4.8774 | 3.6560 | 1.2214 [0.8020, 1.7284] | **SEP** | 0.299 / 0.583 / 0.8493 [0.2476, 1.6237] BONF-SEP |
| speed: v0 10-15 | 952 | 62 | 3.0719 | 2.6253 | 0.4466 [0.2677, 0.6576] | **SEP** | 0.273 / 0.195 / -0.2465 [-0.7586, 0.2673] |
| speed: v0 15-25 | 679 | 34 | 1.7173 | 1.4734 | 0.2439 [0.1021, 0.3919] | **SEP** | 0.195 / 0.076 / -0.4743 [-0.9026, -0.0579] BONF-SEP |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5923 | 0.4614 | 0.1309 [0.0261, 0.2678] | **SEP** | 0.070 / 0.015 / -0.5323 [-0.9006, -0.0838] BONF-SEP |
| nav: nav=follow | 2228 | 88 | 2.6331 | 2.0803 | 0.5528 [0.3566, 0.7945] | **SEP** | 0.640 / 0.565 / -0.2025 [-0.8707, 0.5582] |
| nav: nav=left (low n) | 308 | 13 | 7.0938 | 6.9161 | 0.1776 [-0.4246, 0.5810] |  | 0.088 / 0.025 / -0.4916 [-1.6507, 0.1915] |
| nav: nav=right | 947 | 38 | 5.1463 | 4.2032 | 0.9431 [0.3603, 1.4503] | **SEP** | 0.272 / 0.410 / 0.4359 [-0.4762, 1.2649] |

episode concentration: 0.835 of episodes have mean delta > 0; top-5 episodes carry 0.284 of the summed delta; delta excluding the top 5 = 0.4654 (overall 0.6257).

### S6 -- EMA-5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3541 | 139 | 3.3694 | 3.2663 | 0.1031 [0.0009, 0.2105] | **SEP** |  |
| turn: straight <0.02 | 2103 | 122 | 1.5255 | 1.4585 | 0.0670 [-0.0014, 0.1482] |  | 0.594 / 0.386 / -0.0888 [-0.4844, 0.2439] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.8890 | 2.8900 | -0.0011 [-0.1769, 0.1642] |  | 0.191 / -0.002 / -0.1287 [-0.4357, 0.1678] |
| turn: turn 0.06-0.15 | 437 | 59 | 6.4953 | 6.3490 | 0.1463 [-0.1561, 0.4543] |  | 0.123 / 0.175 / 0.0493 [-0.3595, 0.5159] |
| turn: sharp >=0.15 | 307 | 33 | 12.1191 | 11.5771 | 0.5421 [-0.2105, 1.4704] |  | 0.087 / 0.456 / 0.4807 [-0.5988, 1.9760] |
| turn: undef (GT crawling) (low n) | 19 | 10 | 11.2453 | 11.5385 | -0.2931 [-0.7032, 0.1586] |  | 0.005 / -0.015 / -0.3983 [-1.0973, 0.2968] |
| speed: v0 <2 | 203 | 25 | 13.2602 | 13.3215 | -0.0613 [-0.3400, 0.1491] |  | 0.057 / -0.034 / -0.1744 [-0.6486, 0.1491] |
| speed: v0 2-5 | 422 | 52 | 4.3226 | 4.1501 | 0.1725 [-0.2426, 0.7556] |  | 0.119 / 0.199 / 0.0788 [-0.5196, 0.9445] |
| speed: v0 5-10 | 1040 | 79 | 3.6560 | 3.4941 | 0.1619 [-0.0457, 0.3709] |  | 0.294 / 0.461 / 0.0832 [-0.2652, 0.4450] |
| speed: v0 10-15 | 952 | 62 | 2.6253 | 2.4870 | 0.1383 [-0.0437, 0.3427] |  | 0.269 / 0.361 / 0.0481 [-0.2840, 0.4412] |
| speed: v0 15-25 | 679 | 34 | 1.4734 | 1.4456 | 0.0278 [-0.0664, 0.1377] |  | 0.192 / 0.052 / -0.0931 [-0.3396, 0.1481] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4614 | 0.5193 | -0.0579 [-0.1214, -0.0014] | **SEP** | 0.069 / -0.039 / -0.1730 [-0.3808, -0.0069] BONF-SEP |
| nav: nav=follow | 2251 | 88 | 2.0698 | 1.9774 | 0.0924 [-0.0068, 0.1992] |  | 0.636 / 0.570 / -0.0294 [-0.4576, 0.3213] |
| nav: nav=left (low n) | 322 | 13 | 10.0026 | 9.9171 | 0.0855 [-0.3436, 0.5053] |  | 0.091 / 0.075 / -0.0193 [-0.7236, 0.6515] |
| nav: nav=right | 968 | 38 | 4.1849 | 4.0511 | 0.1338 [-0.1056, 0.4025] |  | 0.273 / 0.355 / 0.0423 [-0.3196, 0.5710] |

episode concentration: 0.540 of episodes have mean delta > 0; top-5 episodes carry 0.827 of the summed delta; delta excluding the top 5 = 0.0185 (overall 0.1031).

### S6 -- RAW-5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3472 | 139 | 0.9772 | 0.6622 | 0.3150 [0.1231, 0.6477] | **SEP** | 0.0265 [-0.1847, 0.3815] |
| seg1 | straight | 2062 | 122 | 0.7453 | 0.4583 | 0.2870 [0.0821, 0.6923] | **SEP** | 0.1918 [-0.2003, 0.9691] |
| seg1 | curved | 1400 | 88 | 1.1944 | 0.8375 | 0.3569 [0.1510, 0.6766] | **SEP** | -0.2172 [-0.5566, 0.0369] |
| seg2 | all | 3451 | 139 | 1.7323 | 1.3063 | 0.4259 [0.2877, 0.5810] | **SEP** | -0.1591 [-0.4109, 0.1411] |
| seg2 | straight | 2052 | 122 | 1.0041 | 0.7580 | 0.2461 [0.1652, 0.3445] | **SEP** | -0.1483 [-0.4027, 0.2665] |
| seg2 | curved | 1399 | 88 | 2.8003 | 2.1106 | 0.6897 [0.3857, 1.0087] | **SEP** | -0.1749 [-0.5994, 0.2342] |
| seg3 | all | 3387 | 139 | 2.6289 | 1.8777 | 0.7513 [0.4648, 1.0782] | **SEP** | -0.1331 [-0.5481, 0.4337] |
| seg3 | straight | 2012 | 121 | 1.3407 | 0.8069 | 0.5337 [0.2857, 0.9854] | **SEP** | -0.0020 [-0.4536, 0.8083] |
| seg3 | curved | 1375 | 88 | 4.5140 | 3.4444 | 1.0696 [0.5311, 1.5851] | **SEP** | -0.3249 [-0.9634, 0.2863] |
| seg4 | all | 3316 | 139 | 3.7367 | 2.7614 | 0.9753 [0.6254, 1.3441] | **SEP** | -0.2078 [-0.6299, 0.3169] |
| seg4 | straight | 1990 | 121 | 1.5319 | 0.8714 | 0.6605 [0.4599, 0.9260] | **SEP** | -0.1366 [-0.5414, 0.5281] |
| seg4 | curved | 1326 | 87 | 7.0456 | 5.5978 | 1.4478 [0.6805, 2.2075] | **SEP** | -0.3146 [-1.0971, 0.4956] |
| seg5 | all | 3252 | 139 | 5.2058 | 4.0441 | 1.1617 [0.7546, 1.6116] | **SEP** | -0.4962 [-0.9243, -0.0681] SEP |
| seg5 | straight | 1963 | 120 | 1.7274 | 1.0302 | 0.6972 [0.5124, 0.8975] | **SEP** | -0.4206 [-0.6666, -0.1737] SEP |
| seg5 | curved | 1289 | 87 | 10.5030 | 8.6339 | 1.8691 [0.8652, 2.9201] | **SEP** | -0.6114 [-1.6032, 0.3858] |
| seg6 | all | 3192 | 136 | 6.9586 | 5.6120 | 1.3465 [0.8345, 1.8760] | **SEP** | -0.8752 [-1.3864, -0.3645] SEP |
| seg6 | straight | 1950 | 119 | 2.1567 | 1.4421 | 0.7146 [0.4954, 0.9650] | **SEP** | -0.5396 [-0.8592, -0.2309] SEP |
| seg6 | curved | 1242 | 85 | 14.4977 | 12.1591 | 2.3387 [1.1399, 3.6188] | **SEP** | -1.4020 [-2.6790, -0.1923] SEP |

### S6 -- RAW-5k - EMA-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3472 | 139 | 0.9772 | 0.6127 | 0.3645 [0.1255, 0.7907] | **SEP** | 0.1239 [-0.1319, 0.5758] |
| seg1 | straight | 2062 | 122 | 0.7453 | 0.3758 | 0.3696 [0.0774, 0.9693] | **SEP** | 0.3264 [-0.1500, 1.2967] |
| seg1 | curved | 1400 | 88 | 1.1944 | 0.8355 | 0.3589 [0.1704, 0.6468] | **SEP** | -0.1754 [-0.4919, 0.0484] |
| seg2 | all | 3451 | 139 | 1.7323 | 1.3289 | 0.4034 [0.2830, 0.5413] | **SEP** | 0.0168 [-0.3166, 0.5327] |
| seg2 | straight | 2052 | 122 | 1.0041 | 0.7683 | 0.2358 [0.1621, 0.3248] | **SEP** | 0.1042 [-0.3226, 0.8916] |
| seg2 | curved | 1399 | 88 | 2.8003 | 2.1511 | 0.6492 [0.3849, 0.9286] | **SEP** | -0.1113 [-0.4766, 0.2355] |
| seg3 | all | 3388 | 139 | 2.6283 | 1.9473 | 0.6810 [0.4205, 0.9734] | **SEP** | -0.1734 [-0.4766, 0.1949] |
| seg3 | straight | 2013 | 121 | 1.3403 | 0.8431 | 0.4973 [0.2593, 0.9266] | **SEP** | -0.1070 [-0.3786, 0.3206] |
| seg3 | curved | 1375 | 88 | 4.5140 | 3.5640 | 0.9501 [0.4899, 1.3846] | **SEP** | -0.2705 [-0.8393, 0.2623] |
| seg4 | all | 3307 | 139 | 3.7198 | 2.8584 | 0.8614 [0.5488, 1.2117] | **SEP** | -0.2257 [-0.6064, 0.2074] |
| seg4 | straight | 1988 | 121 | 1.5228 | 0.8543 | 0.6685 [0.3915, 1.1197] | **SEP** | -0.1592 [-0.4768, 0.3096] |
| seg4 | curved | 1319 | 87 | 7.0311 | 5.8791 | 1.1520 [0.5057, 1.7909] | **SEP** | -0.3260 [-1.0516, 0.4161] |
| seg5 | all | 3247 | 139 | 5.2010 | 4.2389 | 0.9621 [0.6204, 1.3341] | **SEP** | -0.6130 [-1.0760, -0.2023] SEP |
| seg5 | straight | 1963 | 120 | 1.7164 | 1.1131 | 0.6032 [0.4305, 0.7872] | **SEP** | -0.5040 [-0.9918, -0.1432] SEP |
| seg5 | curved | 1284 | 87 | 10.5285 | 9.0177 | 1.5108 [0.6432, 2.3274] | **SEP** | -0.7797 [-1.6904, 0.0965] |
| seg6 | all | 3186 | 136 | 6.9567 | 5.8304 | 1.1263 [0.7246, 1.5612] | **SEP** | -0.8484 [-1.3484, -0.3723] SEP |
| seg6 | straight | 1948 | 119 | 2.1581 | 1.4904 | 0.6677 [0.4436, 0.9101] | **SEP** | -0.4278 [-0.7376, -0.1281] SEP |
| seg6 | curved | 1238 | 85 | 14.5073 | 12.6594 | 1.8479 [0.8290, 2.8509] | **SEP** | -1.5103 [-2.6508, -0.3637] SEP |

### S6 -- EMA-5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3499 | 139 | 0.7139 | 0.8147 | -0.1007 [-0.3129, 0.0156] |  | -0.1473 [-0.3616, -0.0235] SEP |
| seg1 | straight | 2072 | 122 | 0.3745 | 0.5433 | -0.1689 [-0.5493, 0.0131] |  | -0.2208 [-0.6070, -0.0363] SEP |
| seg1 | curved | 1410 | 88 | 1.0897 | 1.0917 | -0.0020 [-0.0479, 0.0462] |  | -0.0416 [-0.1049, 0.0210] |
| seg2 | all | 3485 | 139 | 1.5335 | 1.4599 | 0.0737 [-0.0206, 0.2012] |  | -0.0197 [-0.1160, 0.1046] |
| seg2 | straight | 2076 | 122 | 1.0214 | 0.8386 | 0.1827 [-0.0095, 0.5655] |  | -0.0779 [-0.1200, -0.0426] SEP |
| seg2 | curved | 1409 | 88 | 2.2882 | 2.3752 | -0.0870 [-0.3698, 0.1342] |  | 0.0659 [-0.1761, 0.3528] |
| seg3 | all | 3458 | 139 | 2.1877 | 2.0612 | 0.1266 [0.0013, 0.2739] | **SEP** | 0.2843 [-0.1610, 1.0533] |
| seg3 | straight | 2062 | 121 | 1.1110 | 0.9771 | 0.1339 [0.0068, 0.3535] | **SEP** | 0.3505 [-0.1424, 1.2954] |
| seg3 | curved | 1396 | 88 | 3.7782 | 3.6624 | 0.1158 [-0.1063, 0.3638] |  | 0.1864 [-0.2853, 0.7581] |
| seg4 | all | 3412 | 139 | 3.0729 | 2.9680 | 0.1048 [-0.0503, 0.2594] |  | 0.1403 [-0.1373, 0.5042] |
| seg4 | straight | 2048 | 121 | 1.1405 | 1.1380 | 0.0025 [-0.2122, 0.1648] |  | 0.1852 [-0.1518, 0.7673] |
| seg4 | curved | 1364 | 87 | 5.9742 | 5.7157 | 0.2585 [-0.0311, 0.5955] |  | 0.0729 [-0.3150, 0.4498] |
| seg5 | all | 3363 | 139 | 4.4549 | 4.3053 | 0.1496 [-0.0426, 0.3543] |  | 0.1626 [-0.1800, 0.5764] |
| seg5 | straight | 2019 | 121 | 1.3218 | 1.3072 | 0.0146 [-0.2184, 0.2112] |  | 0.1545 [-0.2090, 0.7653] |
| seg5 | curved | 1344 | 88 | 9.1615 | 8.8091 | 0.3524 [-0.0267, 0.7608] |  | 0.1747 [-0.4080, 0.7129] |
| seg6 | all | 3325 | 138 | 6.0944 | 5.9267 | 0.1677 [-0.0589, 0.3944] |  | 0.0769 [-0.2255, 0.3965] |
| seg6 | straight | 2006 | 120 | 1.7029 | 1.7264 | -0.0235 [-0.2615, 0.1769] |  | 0.0006 [-0.2025, 0.2650] |
| seg6 | curved | 1317 | 88 | 12.7719 | 12.3079 | 0.4640 [0.0002, 0.9878] | **SEP** | 0.1885 [-0.5509, 0.8637] |

### S6 -- RAW-5k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 334 | 39 | 4.2290 | 3.5306 | 0.6984 [0.4350, 0.9973] | **SEP** |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 1.4300 | 0.7632 | 0.6668 [0.4902, 0.8687] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.6789 | 0.5523 | 0.1265 [0.0465, 0.2327] | **SEP** |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 96 | 20 | 3.3946 | 4.3920 | -0.9974 [-5.0425, 1.1096] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 3.2925 | 2.5310 | 0.7616 [0.3749, 1.1838] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.4557 | 2.1952 | 0.2604 [-0.0186, 0.5575] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 60 | 20 | 9.0826 | 8.8908 | 0.1919 [-1.0566, 1.6088] |  |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 7.1376 | 5.8272 | 1.3104 [0.6682, 2.0459] | **SEP** |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 5.2555 | 4.3281 | 0.9274 [0.4336, 1.4802] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 67 | 16 | 17.1550 | 14.2227 | 2.9323 [-1.0996, 8.3226] |  |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 13.3692 | 10.7541 | 2.6151 [0.4646, 4.9780] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 6.7955 | 6.3909 | 0.4046 [-1.0651, 5.3439] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 10 | 8 | 18.4175 | 18.1990 | 0.2185 [-0.0517, 0.5337] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW-5k s0-s1 [CI] | EMA-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0092 [-0.0548, 0.0335] | -0.0153 [-0.0751, 0.0424] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0222 [-0.0586, 0.0109] | -0.0024 [-0.0263, 0.0190] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | -0.0331 [-0.1566, 0.0983] | 0.0493 [-0.0553, 0.1670] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.0294 [-0.1727, 0.0941] | 0.0108 [-0.1388, 0.1615] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.1538 [-0.1558, 0.4832] | -0.2362 [-0.7709, 0.2315] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.2483 [-0.0632, 0.7205] | -0.7635 [-1.9674, 0.4258] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | 0.1279 [-0.1377, 0.5456] | 0.2812 [0.0192, 0.7088] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.0357 [-0.1747, 0.1170] | -0.1594 [-0.3925, 0.0077] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | -0.0247 [-0.1237, 0.1011] | -0.0367 [-0.1984, 0.0987] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | 0.0083 [-0.0517, 0.0713] | -0.0091 [-0.0758, 0.0548] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | -0.0203 [-0.0677, 0.0198] | 0.0031 [-0.0236, 0.0284] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | -0.0163 [-0.0532, 0.0239] | 0.0016 [-0.0069, 0.0122] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0053 [-0.0471, 0.0425] | 0.0235 [-0.0170, 0.0654] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.1081 [-0.0251, 0.3113] | 0.0874 [-0.0438, 0.2535] | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | -0.0566 [-0.1710, 0.0566] | -0.1398 [-0.3317, 0.0343] | 0.0235 [-0.1175, 0.1775] |

### S6 -- RAW-5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.7308 | 0.7182 / 0.7434 | True / True / True | 0.0092 / 0.0128 | 57.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.4932 | 0.4952 / 0.4912 | True / True / True | 0.0222 / 0.0216 | 22.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.3967 | 0.3529 / 0.4406 | False / False / False | 0.0331 / 0.0713 | 5.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 1.0854 | 1.0398 / 1.1310 | True / True / True | 0.0294 / 0.0611 | 17.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 2.6196 | 2.6175 / 2.6218 | True / True / True | 0.1538 / 0.2004 | 13.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.2185 | 0.3855 / 0.0515 | True / False / False | 0.2483 / 1.9278 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.1713 | -0.1689 / -0.1737 | False / False / False | 0.1279 / 0.0328 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.8828 | 0.7644 / 1.0011 | False / False / False | 0.0357 / 0.2009 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 1.3833 | 1.3831 / 1.3835 | True / True / True | 0.0247 / 0.0243 | 56.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.5848 | 0.6024 / 0.5673 | True / True / True | 0.0083 / 0.0268 | 21.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.2717 | 0.2589 / 0.2846 | True / True / True | 0.0203 / 0.0055 | 13.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | 0.0730 | 0.0644 / 0.0815 | False / False / False | 0.0163 / 0.0008 | 4.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.6489 | 0.6415 / 0.6564 | True / True / True | 0.0053 / 0.0080 | 81.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.2477 | 0.3389 / 0.1564 | True / False / False | 0.1081 / 0.0140 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 1.0806 | 1.0220 / 1.1392 | True / True / True | 0.0566 / 0.0235 | 19.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- RAW-5k - EMA-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.6257 | 0.6305 / 0.6209 | True / True / True | 0.0092 / 0.0153 | 40.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.4228 | 0.4136 / 0.4320 | True / True / True | 0.0222 / 0.0024 | 19.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.3991 | 0.3673 / 0.4310 | False / False / False | 0.0331 / 0.0493 | 8.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.9421 | 0.9231 / 0.9611 | True / True / True | 0.0294 / 0.0108 | 32.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 2.0915 | 2.3084 / 1.8746 | True / True / True | 0.1538 / 0.2362 | 8.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.0794 | 0.1027 / 0.0562 | False / False / False | 0.2483 / 0.7635 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0928 | -0.1870 / 0.0013 | False / False / False | 0.1279 / 0.2812 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.7103 | 0.7721 / 0.6485 | False / False / False | 0.0357 / 0.1594 | 4.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 1.2214 | 1.2274 / 1.2154 | True / True / True | 0.0247 / 0.0367 | 33.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.4466 | 0.4553 / 0.4379 | True / True / True | 0.0083 / 0.0091 | 49.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.2439 | 0.2322 / 0.2556 | True / True / True | 0.0203 / 0.0031 | 12.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | 0.1309 | 0.1219 / 0.1399 | True / True / True | 0.0163 / 0.0016 | 8.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.5528 | 0.5389 / 0.5666 | True / True / True | 0.0053 / 0.0235 | 23.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.1776 | 0.2139 / 0.1413 | False / False / False | 0.1081 / 0.0874 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.9431 | 0.9816 / 0.9046 | True / True / True | 0.0566 / 0.1398 | 6.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- EMA-5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1031 | 0.0901 / 0.1207 | False / True / True | 0.0153 / 0.0128 | 6.7 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|straight <0.02 | 0.0670 | 0.0788 / 0.0542 | True / False / False | 0.0024 / 0.0216 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.0011 | -0.0125 / 0.0104 | False / False / False | 0.0493 / 0.0713 | 0.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.1463 | 0.1163 / 0.1621 | False / False / False | 0.0108 / 0.0611 | 2.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.5421 | 0.3322 / 0.8313 | False / True / False | 0.2362 / 0.2004 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.2931 | 0.4706 / -1.0569 | False / False / False | 0.7635 / 1.9278 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0613 | 0.0806 / -0.1180 | False / False / False | 0.2812 / 0.0328 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1725 | -0.0077 / 0.3526 | False / False / False | 0.1594 / 0.2009 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1619 | 0.1557 / 0.1681 | False / False / False | 0.0367 / 0.0243 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.1383 | 0.1471 / 0.1294 | False / False / False | 0.0091 / 0.0268 | 5.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0278 | 0.0267 / 0.0290 | False / False / False | 0.0031 / 0.0055 | 5.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0579 | -0.0575 / -0.0583 | True / True / True | 0.0016 / 0.0008 | 35.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.0924 | 0.0991 / 0.0851 | False / False / False | 0.0235 / 0.0080 | 3.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | 0.0855 | 0.1340 / 0.1149 | False / False / False | 0.0874 / 0.0140 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1338 | 0.0545 / 0.2055 | False / False / False | 0.1398 / 0.0235 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1400): RAW-5k bias -0.258, refcv7-50400 bias +0.042, EMA-5k bias -0.004; RAW-5k MAE 1.194, refcv7-50400 MAE 0.837, EMA-5k MAE 0.836
* seg2 (n 1399): RAW-5k bias -0.657, refcv7-50400 bias -0.217, EMA-5k bias -0.347; RAW-5k MAE 2.800, refcv7-50400 MAE 2.111, EMA-5k MAE 2.151
* seg3 (n 1375): RAW-5k bias -1.438, refcv7-50400 bias -0.880, EMA-5k bias -1.082; RAW-5k MAE 4.514, refcv7-50400 MAE 3.444, EMA-5k MAE 3.564
* seg4 (n 1319): RAW-5k bias -2.869, refcv7-50400 bias -2.369, EMA-5k bias -2.601; RAW-5k MAE 7.031, refcv7-50400 MAE 5.618, EMA-5k MAE 5.879
* seg5 (n 1284): RAW-5k bias -4.359, refcv7-50400 bias -4.237, EMA-5k bias -4.293; RAW-5k MAE 10.528, refcv7-50400 MAE 8.639, EMA-5k MAE 9.018
* seg6 (n 1238): RAW-5k bias -5.825, refcv7-50400 bias -6.645, EMA-5k bias -6.334; RAW-5k MAE 14.507, refcv7-50400 MAE 12.160, EMA-5k MAE 12.659
**sharp (>= 0.15)**

* seg1 (n 300): RAW-5k bias -1.277, refcv7-50400 bias +0.188, EMA-5k bias -0.017; RAW-5k MAE 2.311, refcv7-50400 MAE 1.441, EMA-5k MAE 1.364
* seg2 (n 297): RAW-5k bias -3.096, refcv7-50400 bias -0.483, EMA-5k bias -1.107; RAW-5k MAE 5.284, refcv7-50400 MAE 3.782, EMA-5k MAE 3.800
* seg3 (n 292): RAW-5k bias -6.235, refcv7-50400 bias -2.122, EMA-5k bias -3.222; RAW-5k MAE 9.309, refcv7-50400 MAE 6.340, EMA-5k MAE 6.742
* seg4 (n 287): RAW-5k bias -10.447, refcv7-50400 bias -5.904, EMA-5k bias -7.048; RAW-5k MAE 15.265, refcv7-50400 MAE 11.275, EMA-5k MAE 12.370
* seg5 (n 282): RAW-5k bias -14.673, refcv7-50400 bias -10.715, EMA-5k bias -11.344; RAW-5k MAE 23.059, refcv7-50400 MAE 18.190, EMA-5k MAE 19.432
* seg6 (n 276): RAW-5k bias -17.950, refcv7-50400 bias -16.348, EMA-5k bias -16.300; RAW-5k MAE 30.090, refcv7-50400 MAE 24.180, EMA-5k MAE 25.597
**curved & v0 5-10**

* seg1 (n 547): RAW-5k bias -0.163, refcv7-50400 bias +0.049, EMA-5k bias +0.052; RAW-5k MAE 1.351, refcv7-50400 MAE 0.896, EMA-5k MAE 0.900
* seg2 (n 547): RAW-5k bias -0.973, refcv7-50400 bias -0.516, EMA-5k bias -0.472; RAW-5k MAE 3.637, refcv7-50400 MAE 2.231, EMA-5k MAE 2.292
* seg3 (n 537): RAW-5k bias -2.002, refcv7-50400 bias -1.674, EMA-5k bias -1.546; RAW-5k MAE 5.697, refcv7-50400 MAE 3.809, EMA-5k MAE 3.920
* seg4 (n 496): RAW-5k bias -3.797, refcv7-50400 bias -3.693, EMA-5k bias -3.484; RAW-5k MAE 8.351, refcv7-50400 MAE 6.368, EMA-5k MAE 6.400
* seg5 (n 474): RAW-5k bias -4.985, refcv7-50400 bias -5.838, EMA-5k bias -5.342; RAW-5k MAE 12.351, refcv7-50400 MAE 9.457, EMA-5k MAE 9.524
* seg6 (n 442): RAW-5k bias -5.452, refcv7-50400 bias -7.940, EMA-5k bias -7.009; RAW-5k MAE 16.544, refcv7-50400 MAE 12.597, EMA-5k MAE 12.865
**sharp & v0 5-10**

* seg1 (n 139): RAW-5k bias -0.881, refcv7-50400 bias +0.062, EMA-5k bias +0.031; RAW-5k MAE 1.740, refcv7-50400 MAE 1.251, EMA-5k MAE 1.109
* seg2 (n 139): RAW-5k bias -4.116, refcv7-50400 bias -1.267, EMA-5k bias -1.331; RAW-5k MAE 5.859, refcv7-50400 MAE 3.564, EMA-5k MAE 3.402
* seg3 (n 138): RAW-5k bias -8.326, refcv7-50400 bias -4.253, EMA-5k bias -4.386; RAW-5k MAE 10.584, refcv7-50400 MAE 6.990, EMA-5k MAE 6.905
* seg4 (n 134): RAW-5k bias -12.780, refcv7-50400 bias -9.037, EMA-5k bias -8.665; RAW-5k MAE 16.649, refcv7-50400 MAE 12.509, EMA-5k MAE 12.661
* seg5 (n 132): RAW-5k bias -15.528, refcv7-50400 bias -13.786, EMA-5k bias -12.454; RAW-5k MAE 24.975, refcv7-50400 MAE 19.364, EMA-5k MAE 19.659
* seg6 (n 126): RAW-5k bias -15.255, refcv7-50400 bias -17.206, EMA-5k bias -14.819; RAW-5k MAE 31.464, refcv7-50400 MAE 24.803, EMA-5k MAE 25.263

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1238 / 1238 | 26.6 | RAW-5k -0.2307 [-0.3281, -0.1432]; refcv7-50400 -0.2333 [-0.3029, -0.1676]; EMA-5k -0.2234 [-0.3038, -0.1535] | RAW-5k +0.0027 [-0.0566, +0.0546]; EMA-5k +0.0099 [-0.0211, +0.0372] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1404 / 6615 | 16.3 | RAW-5k -0.2071 [-0.3032, -0.1239]; refcv7-50400 -0.1633 [-0.2260, -0.1063]; EMA-5k -0.1655 [-0.2403, -0.1024] | RAW-5k -0.0438 [-0.1044, +0.0078]; EMA-5k -0.0022 [-0.0324, +0.0235] |
| sharp (>= 0.15) | last segment | 276 / 276 | 63.4 | RAW-5k -0.2765 [-0.4025, -0.1750]; refcv7-50400 -0.2480 [-0.3290, -0.1734]; EMA-5k -0.2462 [-0.3457, -0.1647] | RAW-5k -0.0285 [-0.1075, +0.0366]; EMA-5k +0.0018 [-0.0382, +0.0321] |
| sharp (>= 0.15) | segments 2..K | 300 / 1434 | 39.8 | RAW-5k -0.2573 [-0.3839, -0.1582]; refcv7-50400 -0.1780 [-0.2559, -0.1112]; EMA-5k -0.1896 [-0.2867, -0.1125] | RAW-5k -0.0793 [-0.1639, -0.0125] SEP; EMA-5k -0.0116 [-0.0529, +0.0183] |
| curved & v0 5-10 | last segment | 442 / 442 | 29.4 | RAW-5k -0.2168 [-0.3393, -0.1122]; refcv7-50400 -0.2489 [-0.3704, -0.1544]; EMA-5k -0.2235 [-0.3443, -0.1240] | RAW-5k +0.0322 [-0.0683, +0.1252]; EMA-5k +0.0254 [-0.0129, +0.0624] |
| curved & v0 5-10 | segments 2..K | 547 / 2496 | 18.1 | RAW-5k -0.2194 [-0.3342, -0.1236]; refcv7-50400 -0.1931 [-0.3055, -0.1112]; EMA-5k -0.1799 [-0.2870, -0.0967] | RAW-5k -0.0263 [-0.1231, +0.0577]; EMA-5k +0.0132 [-0.0205, +0.0449] |
| sharp & v0 5-10 | last segment | 126 / 126 | 67.7 | RAW-5k -0.2500 [-0.3902, -0.1335]; refcv7-50400 -0.2485 [-0.3800, -0.1430]; EMA-5k -0.2276 [-0.3613, -0.1170] | RAW-5k -0.0015 [-0.1271, +0.1059]; EMA-5k +0.0209 [-0.0256, +0.0627] |
| sharp & v0 5-10 | segments 2..K | 139 / 669 | 43.9 | RAW-5k -0.2574 [-0.3981, -0.1457]; refcv7-50400 -0.1979 [-0.3241, -0.1044]; EMA-5k -0.1892 [-0.3139, -0.0906] | RAW-5k -0.0595 [-0.1821, +0.0387]; EMA-5k +0.0087 [-0.0336, +0.0480] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW-5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3483 | 0.7308 | 0.21345 | 0.41124 | 0.672 / 0.328 / 0.000 | 945.5 / 2545.5 |
| turn|straight <0.02 | 2068 | 0.4932 | 0.19339 | 0.30284 | 0.716 / 0.284 / 0.000 | 195.1 / 1020.0 |
| turn|gentle 0.02-0.06 | 673 | 0.3967 | 0.21590 | 0.38089 | 0.626 / 0.374 / 0.000 | 104.1 / 267.0 |
| turn|turn 0.06-0.15 | 431 | 1.0854 | 0.44385 | 0.78828 | 0.610 / 0.390 / 0.000 | 101.9 / 467.8 |
| turn|sharp >=0.15 | 301 | 2.6196 | 0.95499 | 1.58064 | 0.561 / 0.439 / 0.000 | 172.8 / 788.5 |
| turn|undef (GT crawling) | 10 | 0.2185 | 0.09415 | 0.19893 | 0.700 / 0.300 / 0.000 | 1.1 / 2.2 |
| speed|v0 <2 | 145 | -0.1713 | 0.12323 | 0.16187 | 0.593 / 0.407 / 0.000 | 21.0 / -24.8 |
| speed|v0 2-5 | 422 | 0.8828 | 0.66097 | 0.83507 | 0.756 / 0.244 / 0.000 | 184.9 / 372.5 |
| speed|v0 5-10 | 1040 | 1.3833 | 0.41019 | 0.78464 | 0.676 / 0.324 / 0.000 | 375.4 / 1438.6 |
| speed|v0 10-15 | 952 | 0.5848 | 0.19918 | 0.36165 | 0.703 / 0.297 / 0.000 | 135.4 / 556.8 |
| speed|v0 15-25 | 679 | 0.2717 | 0.08322 | 0.13858 | 0.616 / 0.384 / 0.000 | 42.5 / 184.5 |
| speed|v0 >=25 | 245 | 0.0730 | 0.07998 | 0.07790 | 0.592 / 0.408 / 0.000 | 5.5 / 17.9 |
| nav|nav=follow | 2228 | 0.6489 | 0.15733 | 0.26845 | 0.655 / 0.345 / 0.000 | 506.1 / 1445.8 |
| nav|nav=left | 308 | 0.2477 | 0.15377 | 0.34702 | 0.679 / 0.321 / 0.000 | 55.6 / 76.3 |
| nav|nav=right | 947 | 1.0806 | 0.57516 | 0.80801 | 0.710 / 0.290 / 0.000 | 326.3 / 1023.3 |
**RAW-5k - EMA-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3483 | 0.6257 | 0.18718 | 0.37640 | 0.672 / 0.328 / 0.000 | 816.3 / 2179.4 |
| turn|straight <0.02 | 2068 | 0.4228 | 0.16526 | 0.27192 | 0.698 / 0.302 / 0.000 | 181.4 / 874.4 |
| turn|gentle 0.02-0.06 | 673 | 0.3991 | 0.22695 | 0.40735 | 0.681 / 0.319 / 0.000 | 101.7 / 268.6 |
| turn|turn 0.06-0.15 | 431 | 0.9421 | 0.25104 | 0.65555 | 0.608 / 0.392 / 0.000 | 80.9 / 406.1 |
| turn|sharp >=0.15 | 301 | 2.0915 | 0.57570 | 1.20978 | 0.561 / 0.439 / 0.000 | 145.9 / 629.5 |
| turn|undef (GT crawling) | 10 | 0.0794 | 0.06836 | 0.07255 | 0.600 / 0.400 / 0.000 | 0.9 / 0.8 |
| speed|v0 <2 | 145 | -0.0928 | 0.09585 | 0.15459 | 0.566 / 0.434 / 0.000 | 22.8 / -13.5 |
| speed|v0 2-5 | 422 | 0.7103 | 0.57769 | 0.73592 | 0.737 / 0.263 / 0.000 | 153.1 / 299.7 |
| speed|v0 5-10 | 1040 | 1.2214 | 0.38157 | 0.73524 | 0.670 / 0.330 / 0.000 | 335.5 / 1270.3 |
| speed|v0 10-15 | 952 | 0.4466 | 0.14414 | 0.30536 | 0.700 / 0.300 / 0.000 | 112.4 / 425.1 |
| speed|v0 15-25 | 679 | 0.2439 | 0.10771 | 0.14236 | 0.645 / 0.355 / 0.000 | 32.5 / 165.6 |
| speed|v0 >=25 | 245 | 0.1309 | 0.09079 | 0.10393 | 0.592 / 0.408 / 0.000 | 5.7 / 32.1 |
| nav|nav=follow | 2228 | 0.5528 | 0.14048 | 0.24477 | 0.658 / 0.342 / 0.000 | 453.2 / 1231.6 |
| nav|nav=left | 308 | 0.1776 | 0.10296 | 0.22922 | 0.636 / 0.364 / 0.000 | 52.9 / 54.7 |
| nav|nav=right | 947 | 0.9431 | 0.54694 | 0.78603 | 0.716 / 0.284 / 0.000 | 269.9 / 893.1 |
**EMA-5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3541 | 0.1031 | 0.01419 | 0.00679 | 0.528 / 0.472 / 0.000 | 517.2 / 365.0 |
| turn|straight <0.02 | 2103 | 0.0670 | 0.02291 | 0.01520 | 0.552 / 0.448 / 0.000 | 171.4 / 141.0 |
| turn|gentle 0.02-0.06 | 675 | -0.0011 | -0.02399 | -0.04162 | 0.474 / 0.526 / 0.000 | 61.9 / -0.7 |
| turn|turn 0.06-0.15 | 437 | 0.1463 | 0.02104 | 0.07108 | 0.519 / 0.481 / 0.000 | 48.6 / 63.9 |
| turn|sharp >=0.15 | 307 | 0.5421 | -0.03805 | 0.25309 | 0.492 / 0.508 / 0.000 | 122.1 / 166.4 |
| turn|undef (GT crawling) | 19 | -0.2931 | 0.00301 | -0.01160 | 0.579 / 0.421 / 0.000 | 1.0 / -5.6 |
| speed|v0 <2 | 203 | -0.0613 | -0.01959 | -0.03344 | 0.473 / 0.527 / 0.000 | 17.3 / -12.4 |
| speed|v0 2-5 | 422 | 0.1725 | 0.04604 | 0.06773 | 0.550 / 0.450 / 0.000 | 106.0 / 72.8 |
| speed|v0 5-10 | 1040 | 0.1619 | 0.02298 | 0.01830 | 0.530 / 0.470 / 0.000 | 211.6 / 168.4 |
| speed|v0 10-15 | 952 | 0.1383 | 0.03129 | 0.02136 | 0.579 / 0.421 / 0.000 | 101.0 / 131.6 |
| speed|v0 15-25 | 679 | 0.0278 | -0.00051 | -0.00854 | 0.499 / 0.501 / 0.000 | 37.7 / 18.9 |
| speed|v0 >=25 | 245 | -0.0579 | -0.02692 | -0.03108 | 0.408 / 0.592 / 0.000 | 1.4 / -14.2 |
| nav|nav=follow | 2251 | 0.0924 | 0.00301 | -0.00746 | 0.506 / 0.494 / 0.000 | 242.1 / 207.9 |
| nav|nav=left | 322 | 0.0855 | 0.03608 | 0.07483 | 0.584 / 0.416 / 0.000 | 50.6 / 27.5 |
| nav|nav=right | 968 | 0.1338 | 0.03718 | 0.02930 | 0.560 / 0.440 / 0.000 | 211.4 / 129.6 |

### S6 -- the episodes carrying the summed delta (panel file index only)

**RAW-5k - refcv7-50400** (sum 2545.5 over 139 episodes)

* ep111: n 26, share 0.089, mean delta 8.7416, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep085: n 26, share 0.063, mean delta 6.1198, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.058, mean delta 5.6953, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.042, mean delta 3.9683, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep041: n 26, share 0.041, mean delta 3.9814, sharp-window frac 0.12, curved frac 0.38, mean v0 8.6, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.037, mean delta 3.4972, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**RAW-5k - EMA-5k** (sum 2179.4 over 139 episodes)

* ep111: n 26, share 0.077, mean delta 6.4933, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.061, mean delta 5.1207, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.055, mean delta 4.4338, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.046, mean delta 3.8166, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep041: n 26, share 0.045, mean delta 3.7873, sharp-window frac 0.12, curved frac 0.38, mean v0 8.6, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep110: n 26, share 0.035, mean delta 2.9645, sharp-window frac 0.46, curved frac 0.85, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
**EMA-5k - refcv7-50400** (sum 365.0 over 139 episodes)

* ep013: n 27, share 0.248, mean delta 3.3550, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.164, mean delta 2.3032, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.160, mean delta 2.2483, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share 0.133, mean delta 1.8028, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep000: n 27, share 0.121, mean delta 1.6405, sharp-window frac 0.33, curved frac 0.67, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep100: n 18, share 0.102, mean delta 2.0699, sharp-window frac 0.50, curved frac 1.00, mean v0 5.2, nav {'nav=follow': 0, 'nav=left': 18, 'nav=right': 0}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW-5k vs EMA-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 917 / 0.2664 [0.1828, 0.3621] | 35 / 1.2683 [0.6144, 2.2493] |
| only new agrees | 318 / -0.2037 [-0.5480, 0.1181] | 14 / -1.0483 [-2.5582, 0.8735] |
| only earlier agrees | 565 / 1.5365 [1.1455, 1.9783] | 44 / 7.5415 [4.2854, 12.2795] |
| neither agrees | 1683 / 0.6825 [0.3452, 1.0100] | 208 / 1.6024 [-0.1879, 3.4717] |
**RAW-5k vs EMA-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 934 / 0.2706 [0.1973, 0.3563] | 36 / 1.3386 [0.6806, 2.3024] |
| only new agrees | 318 / -0.4097 [-0.9057, 0.0097] | 15 / -5.9647 [-14.3218, -1.0684] |
| only earlier agrees | 552 / 1.7131 [1.2323, 2.3092] | 49 / 7.9148 [4.3850, 12.8076] |
| neither agrees | 1679 / 0.6519 [0.3390, 0.9358] | 201 / 1.0831 [-0.6812, 2.7624] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW-5k vs EMA-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3447 | 0.5780 [0.3356, 0.8458] | 0.2206 [-0.0735, 0.5536] | 0.3571 [0.0928, 0.6272] SEP |
| turn|straight <0.02 | 2046 | 0.4004 [0.1539, 0.6899] | -0.0575 [-0.1372, 0.0084] | 0.4607 [0.2174, 0.7501] SEP |
| turn|gentle 0.02-0.06 | 673 | 0.5311 [0.1944, 0.8649] | -0.3451 [-0.8539, 0.0238] | 0.8805 [0.3976, 1.4444] SEP |
| turn|turn 0.06-0.15 | 423 | 0.3326 [-0.2227, 0.9146] | 0.1441 [-0.1844, 0.4765] | 0.1860 [-0.3246, 0.7429] |
| turn|sharp >=0.15 | 299 | 2.2395 [0.6061, 4.2331] | 3.4834 [0.8893, 6.7820] | -1.3029 [-3.3224, 0.7100] |
| turn|undef (GT crawling) | 6 | 0.9170 [0.3116, 1.4676] | -0.0513 [-0.3960, 0.1713] | 1.1079 [0.3402, 2.1163] SEP |
| speed|v0 <2 | 109 | 0.9165 [0.0601, 2.4457] | 0.0184 [-0.5053, 0.4379] | 1.0328 [0.1252, 2.6955] SEP |
| speed|v0 2-5 | 422 | 1.8982 [0.4836, 3.4811] | 0.7573 [-0.9143, 2.6853] | 1.1409 [-0.3745, 2.8174] |
| speed|v0 5-10 | 1040 | 1.1156 [0.6851, 1.5621] | 0.5257 [0.0316, 1.0835] | 0.5899 [0.0895, 1.0575] SEP |
| speed|v0 10-15 | 952 | -0.1191 [-0.2445, 0.0019] | -0.0878 [-0.1902, 0.0053] | -0.0313 [-0.1733, 0.1120] |
| speed|v0 15-25 | 679 | 0.0393 [-0.0394, 0.1095] | -0.0227 [-0.0751, 0.0173] | 0.0620 [-0.0365, 0.1639] |
| speed|v0 >=25 | 245 | 0.0733 [-0.0112, 0.1503] | 0.0277 [-0.0188, 0.1225] | 0.0457 [-0.0580, 0.1261] |
**RAW-5k vs EMA-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* RAW-5k vs EMA-5k|seed0: anchor flip rate 0.498; heading delta on flipped windows 0.9683 [0.5927, 1.3601] (n 1755) vs unflipped 0.2875 [0.2045, 0.3799]; goal_dist_sel proxy new-earlier 0.2917 [-0.0251, 0.5905]; corr(window dHeading, window dGoalDist) -0.023
* RAW-5k vs EMA-5k|seed1: anchor flip rate 0.504; heading delta on flipped windows 0.9357 [0.5896, 1.2929] (n 1780) vs unflipped 0.2919 [0.2150, 0.3796]; goal_dist_sel proxy new-earlier 0.2906 [-0.0226, 0.5928]; corr(window dHeading, window dGoalDist) -0.026
