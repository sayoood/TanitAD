# refcv8 heading-drift localisation: EMA8-30k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-18T18:18:52; trees: `EMA8-30k` = C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc; `R7-30k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step30000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- EMA8-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4517 | 139 | 1.4425 | 1.4069 | 0.0356 [-0.0036, 0.0759] |  |  |
| turn: straight <0.02 | 2902 | 132 | 1.2018 | 1.1584 | 0.0433 [0.0126, 0.0758] | **SEP** | 0.642 / 0.783 / 0.0217 [-0.1297, 0.1640] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0069 | 1.0094 | -0.0025 [-0.0705, 0.0736] |  | 0.159 / -0.011 / -0.0453 [-0.1607, 0.0841] |
| turn: turn 0.06-0.15 | 510 | 72 | 1.9472 | 1.9285 | 0.0187 [-0.1051, 0.1530] |  | 0.113 / 0.059 / -0.0190 [-0.1924, 0.2032] |
| turn: sharp >=0.15 | 366 | 42 | 3.0691 | 2.9907 | 0.0784 [-0.2219, 0.4233] |  | 0.081 / 0.179 / 0.0466 [-0.3637, 0.6150] |
| turn: undef (GT crawling) (low n) | 19 | 13 | 9.8393 | 9.9210 | -0.0817 [-0.1752, 0.0342] |  | 0.004 / -0.010 / -0.1178 [-0.2925, 0.0658] |
| speed: v0 <2 | 228 | 37 | 12.8883 | 12.8536 | 0.0347 [-0.0899, 0.1740] |  | 0.050 / 0.049 / -0.0009 [-0.2484, 0.2330] |
| speed: v0 2-5 | 516 | 58 | 1.8480 | 1.8694 | -0.0214 [-0.2189, 0.1994] |  | 0.114 / -0.069 / -0.0643 [-0.3391, 0.3453] |
| speed: v0 5-10 | 1390 | 84 | 1.1384 | 1.0092 | 0.1292 [0.0391, 0.2170] | **SEP** | 0.308 / 1.118 / 0.1353 [-0.0058, 0.2896] |
| speed: v0 10-15 | 1192 | 64 | 0.5839 | 0.5744 | 0.0094 [-0.0340, 0.0508] |  | 0.264 / 0.070 / -0.0355 [-0.1351, 0.0716] |
| speed: v0 15-25 | 856 | 35 | 0.3420 | 0.3734 | -0.0314 [-0.0598, -0.0070] | **SEP** | 0.190 / -0.168 / -0.0827 [-0.1731, -0.0001] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1570 | 0.1576 | -0.0006 [-0.0230, 0.0240] |  | 0.074 / -0.001 / -0.0390 [-0.1104, 0.0314] |
| nav: nav=follow | 2882 | 88 | 0.6449 | 0.5978 | 0.0471 [0.0124, 0.0863] | **SEP** | 0.638 / 0.844 / 0.0317 [-0.1162, 0.1754] |
| nav: nav=left (low n) | 405 | 13 | 6.7659 | 6.8876 | -0.1217 [-0.3231, 0.0317] |  | 0.090 / -0.307 / -0.1728 [-0.5139, 0.0499] |
| nav: nav=right | 1230 | 38 | 1.5586 | 1.4982 | 0.0604 [-0.0418, 0.1650] |  | 0.272 / 0.463 / 0.0342 [-0.1195, 0.1992] |

episode concentration: 0.540 of episodes have mean delta > 0; top-5 episodes carry 0.706 of the summed delta; delta excluding the top 5 = 0.0109 (overall 0.0356).

### S2 -- EMA8-30k - R7-30k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4509 | 139 | 1.4447 | 1.4634 | -0.0187 [-0.0582, 0.0204] |  |  |
| turn: straight <0.02 | 2897 | 132 | 1.2034 | 1.1920 | 0.0114 [-0.0240, 0.0453] |  | 0.642 / -0.390 / 0.0841 [-0.0481, 0.2156] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0069 | 1.0281 | -0.0212 [-0.0926, 0.0548] |  | 0.160 / 0.181 / -0.0029 [-0.1118, 0.1207] |
| turn: turn 0.06-0.15 | 510 | 72 | 1.9472 | 2.0033 | -0.0561 [-0.1830, 0.0779] |  | 0.113 / 0.339 / -0.0422 [-0.2489, 0.1973] |
| turn: sharp >=0.15 | 366 | 42 | 3.0691 | 3.2746 | -0.2055 [-0.5180, 0.0790] |  | 0.081 / 0.891 / -0.2033 [-0.6400, 0.2641] |
| turn: undef (GT crawling) (low n) | 16 | 12 | 11.6670 | 11.5561 | 0.1109 [-0.0341, 0.3307] |  | 0.004 / -0.021 / 0.1301 [-0.0736, 0.5328] |
| speed: v0 <2 | 220 | 37 | 13.3495 | 13.3246 | 0.0249 [-0.1449, 0.2127] |  | 0.049 / -0.065 / 0.0458 [-0.2079, 0.3473] |
| speed: v0 2-5 | 516 | 58 | 1.8480 | 1.9548 | -0.1067 [-0.2849, 0.0987] |  | 0.114 / 0.653 / -0.0994 [-0.3519, 0.2365] |
| speed: v0 5-10 | 1390 | 84 | 1.1384 | 1.1066 | 0.0318 [-0.0528, 0.1140] |  | 0.308 / -0.524 / 0.0730 [-0.0682, 0.2129] |
| speed: v0 10-15 | 1192 | 64 | 0.5839 | 0.6061 | -0.0222 [-0.0577, 0.0175] |  | 0.264 / 0.314 / -0.0048 [-0.1018, 0.1001] |
| speed: v0 15-25 | 856 | 35 | 0.3420 | 0.3950 | -0.0530 [-0.0916, -0.0189] | **SEP** | 0.190 / 0.537 / -0.0423 [-0.1336, 0.0429] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1570 | 0.1784 | -0.0214 [-0.0517, 0.0130] |  | 0.074 / 0.085 / -0.0029 [-0.0838, 0.0841] |
| nav: nav=follow | 2874 | 88 | 0.6461 | 0.6372 | 0.0089 [-0.0225, 0.0422] |  | 0.637 / -0.301 / 0.0760 [-0.0705, 0.2242] |
| nav: nav=left (low n) | 405 | 13 | 6.7659 | 7.0377 | -0.2718 [-0.4936, -0.0969] | **SEP** | 0.090 / 1.305 / -0.2781 [-0.6559, -0.0273] BONF-SEP |
| nav: nav=right | 1230 | 38 | 1.5586 | 1.5584 | 0.0002 [-0.0966, 0.1000] |  | 0.273 / -0.003 / 0.0260 [-0.1150, 0.1855] |

episode concentration: 0.360 of episodes have mean delta > 0; top-5 episodes carry -1.126 of the summed delta; delta excluding the top 5 = -0.0414 (overall -0.0187).

### S2 -- R7-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4527 | 139 | 1.5031 | 1.4485 | 0.0546 [0.0265, 0.0834] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.1921 | 1.1597 | 0.0323 [0.0166, 0.0496] | **SEP** | 0.640 / 0.379 / -0.0620 [-0.1925, 0.0424] |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0278 | 1.0095 | 0.0182 [-0.0155, 0.0493] |  | 0.159 / 0.053 / -0.0433 [-0.1147, 0.0278] |
| turn: turn 0.06-0.15 | 512 | 72 | 2.0092 | 1.9319 | 0.0773 [-0.0342, 0.1987] |  | 0.113 / 0.160 / 0.0256 [-0.1460, 0.2053] |
| turn: sharp >=0.15 | 368 | 42 | 3.2871 | 3.0070 | 0.2802 [0.0169, 0.5718] | **SEP** | 0.081 / 0.417 / 0.2455 [-0.1160, 0.6978] |
| turn: undef (GT crawling) (low n) | 28 | 22 | 13.2368 | 13.3146 | -0.0778 [-0.2004, 0.0202] |  | 0.006 / -0.009 / -0.1333 [-0.3174, 0.0243] |
| speed: v0 <2 | 238 | 39 | 13.1830 | 13.1626 | 0.0203 [-0.1078, 0.1516] |  | 0.053 / 0.020 / -0.0362 [-0.2413, 0.1696] |
| speed: v0 2-5 | 516 | 58 | 1.9548 | 1.8694 | 0.0853 [-0.0437, 0.2063] |  | 0.114 / 0.178 / 0.0346 [-0.1492, 0.2258] |
| speed: v0 5-10 | 1390 | 84 | 1.1066 | 1.0092 | 0.0974 [0.0383, 0.1648] | **SEP** | 0.307 / 0.547 / 0.0617 [-0.0313, 0.1862] |
| speed: v0 10-15 | 1192 | 64 | 0.6061 | 0.5744 | 0.0317 [-0.0180, 0.0720] |  | 0.263 / 0.153 / -0.0312 [-0.1187, 0.0452] |
| speed: v0 15-25 | 856 | 35 | 0.3950 | 0.3734 | 0.0215 [-0.0067, 0.0530] |  | 0.189 / 0.075 / -0.0408 [-0.1081, 0.0281] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1784 | 0.1576 | 0.0208 [0.0018, 0.0382] | **SEP** | 0.074 / 0.028 / -0.0365 [-0.0907, 0.0161] |
| nav: nav=follow | 2876 | 88 | 0.6369 | 0.5986 | 0.0384 [0.0058, 0.0734] | **SEP** | 0.635 / 0.446 / -0.0447 [-0.1283, 0.0402] |
| nav: nav=left (low n) | 414 | 13 | 7.3407 | 7.1974 | 0.1433 [0.0770, 0.2275] | **SEP** | 0.091 / 0.240 / 0.0975 [-0.0159, 0.2373] |
| nav: nav=right | 1237 | 38 | 1.5633 | 1.5005 | 0.0629 [-0.0003, 0.1199] |  | 0.273 / 0.314 / 0.0113 [-0.0876, 0.1210] |

episode concentration: 0.647 of episodes have mean delta > 0; top-5 episodes carry 0.357 of the summed delta; delta excluding the top 5 = 0.0365 (overall 0.0546).

### S2 -- EMA8-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4479 | 139 | 0.5791 | 0.6274 | -0.0484 [-0.1338, 0.0000] |  | 0.0309 [-0.0223, 0.1209] |
| seg1 | straight | 2880 | 132 | 0.5100 | 0.5695 | -0.0595 [-0.2011, 0.0098] |  | 0.0503 [-0.0202, 0.1913] |
| seg1 | curved | 1588 | 101 | 0.5943 | 0.6229 | -0.0286 [-0.0545, -0.0031] | **SEP** | -0.0043 [-0.0507, 0.0375] |
| seg2 | all | 4474 | 139 | 1.0288 | 0.9391 | 0.0897 [-0.0173, 0.2676] |  | -0.0313 [-0.0802, 0.0171] |
| seg2 | straight | 2881 | 132 | 0.8110 | 0.6572 | 0.1538 [0.0102, 0.4416] | **SEP** | -0.0352 [-0.0663, -0.0050] SEP |
| seg2 | curved | 1593 | 101 | 1.4228 | 1.4490 | -0.0262 [-0.1001, 0.0476] |  | -0.0243 [-0.1474, 0.0863] |
| seg3 | all | 4464 | 139 | 1.3078 | 1.2048 | 0.1031 [0.0245, 0.2083] | **SEP** | -0.0306 [-0.1304, 0.0818] |
| seg3 | straight | 2877 | 132 | 0.9016 | 0.7737 | 0.1279 [0.0349, 0.2932] | **SEP** | 0.0018 [-0.0990, 0.1610] |
| seg3 | curved | 1587 | 101 | 2.0443 | 1.9861 | 0.0581 [-0.0589, 0.1750] |  | -0.0893 [-0.2754, 0.0748] |
| seg4 | all | 4448 | 139 | 1.7240 | 1.4920 | 0.2320 [0.0134, 0.5958] | **SEP** | 0.2066 [-0.2011, 0.9009] |
| seg4 | straight | 2863 | 131 | 1.1516 | 0.8364 | 0.3153 [0.0234, 0.8833] | **SEP** | 0.4133 [-0.1355, 1.5207] |
| seg4 | curved | 1577 | 101 | 2.7682 | 2.6860 | 0.0822 [-0.0965, 0.2736] |  | -0.1673 [-0.4751, 0.0932] |

### S2 -- EMA8-30k - R7-30k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4478 | 139 | 0.5792 | 0.5744 | 0.0048 [-0.0451, 0.0906] |  | -0.1231 [-0.2084, -0.0704] SEP |
| seg1 | straight | 2880 | 132 | 0.5100 | 0.4621 | 0.0480 [-0.0222, 0.1885] |  | -0.1327 [-0.2780, -0.0596] SEP |
| seg1 | curved | 1588 | 101 | 0.5943 | 0.6680 | -0.0737 [-0.1044, -0.0436] | **SEP** | -0.1060 [-0.1467, -0.0655] SEP |
| seg2 | all | 4472 | 139 | 1.0292 | 0.9111 | 0.1181 [-0.0665, 0.4535] |  | -0.1218 [-0.2452, 0.0710] |
| seg2 | straight | 2879 | 132 | 0.8115 | 0.5582 | 0.2532 [-0.0160, 0.8100] |  | -0.0373 [-0.1919, 0.2561] |
| seg2 | curved | 1593 | 101 | 1.4228 | 1.5489 | -0.1262 [-0.2027, -0.0518] | **SEP** | -0.2744 [-0.3900, -0.1678] SEP |
| seg3 | all | 4461 | 139 | 1.3085 | 1.2642 | 0.0443 [-0.0295, 0.1433] |  | -0.1533 [-0.3404, 0.1330] |
| seg3 | straight | 2874 | 132 | 0.9022 | 0.8066 | 0.0956 [-0.0037, 0.2622] |  | -0.0014 [-0.2308, 0.4334] |
| seg3 | curved | 1587 | 101 | 2.0443 | 2.0930 | -0.0487 [-0.1517, 0.0602] |  | -0.4283 [-0.6172, -0.2596] SEP |
| seg4 | all | 4445 | 139 | 1.7253 | 1.6533 | 0.0720 [-0.0635, 0.2688] |  | -0.3768 [-0.5675, -0.1548] SEP |
| seg4 | straight | 2862 | 131 | 1.1523 | 1.0129 | 0.1394 [-0.0356, 0.4385] |  | -0.1771 [-0.3585, 0.1292] |
| seg4 | curved | 1577 | 101 | 2.7682 | 2.8191 | -0.0509 [-0.2232, 0.1269] |  | -0.7401 [-1.0548, -0.4540] SEP |

### S2 -- R7-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4487 | 139 | 0.5743 | 0.6674 | -0.0931 [-0.3383, 0.0332] |  | 0.1938 [0.0645, 0.4428] SEP |
| seg1 | straight | 2879 | 132 | 0.4624 | 0.5697 | -0.1073 [-0.3865, 0.0230] |  | 0.1833 [0.0513, 0.4658] SEP |
| seg1 | curved | 1588 | 101 | 0.6680 | 0.6229 | 0.0451 [0.0238, 0.0649] | **SEP** | 0.1017 [0.0678, 0.1344] SEP |
| seg2 | all | 4479 | 139 | 0.9112 | 0.9386 | -0.0274 [-0.1946, 0.0683] |  | 0.0910 [-0.0901, 0.2029] |
| seg2 | straight | 2886 | 132 | 0.5592 | 0.6568 | -0.0976 [-0.3756, 0.0382] |  | 0.0032 [-0.2804, 0.1458] |
| seg2 | curved | 1593 | 101 | 1.5489 | 1.4490 | 0.0999 [0.0418, 0.1542] | **SEP** | 0.2501 [0.1593, 0.3354] SEP |
| seg3 | all | 4468 | 139 | 1.2635 | 1.2049 | 0.0586 [0.0245, 0.0944] | **SEP** | 0.1226 [-0.0604, 0.2474] |
| seg3 | straight | 2878 | 132 | 0.8063 | 0.7737 | 0.0326 [0.0125, 0.0536] | **SEP** | 0.0029 [-0.2798, 0.1460] |
| seg3 | curved | 1590 | 101 | 2.0911 | 1.9852 | 0.1058 [0.0155, 0.1967] | **SEP** | 0.3392 [0.2022, 0.4652] SEP |
| seg4 | all | 4454 | 139 | 1.6564 | 1.4968 | 0.1596 [0.0323, 0.3455] | **SEP** | 0.5845 [0.2990, 1.0855] SEP |
| seg4 | straight | 2864 | 131 | 1.0140 | 0.8394 | 0.1746 [0.0279, 0.4642] | **SEP** | 0.5910 [0.1924, 1.4137] SEP |
| seg4 | curved | 1582 | 101 | 2.8243 | 2.6906 | 0.1337 [-0.0235, 0.2923] |  | 0.5754 [0.3835, 0.7649] SEP |

### S2 -- EMA8-30k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 443 | 47 | 5.9287 | 5.9156 | 0.0130 [-0.0780, 0.1016] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.4486 | 0.3674 | 0.0812 [0.0341, 0.1356] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.1911 | 0.1947 | -0.0035 [-0.0160, 0.0095] |  |  |
| gentle 0.02-0.06 | v0 <5 | 111 | 29 | 1.6588 | 1.7693 | -0.1105 [-0.3688, 0.1761] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.0188 | 0.9711 | 0.0476 [-0.0390, 0.1467] |  |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.4812 | 0.5590 | -0.0777 [-0.1533, -0.0189] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 80 | 25 | 5.1102 | 5.1166 | -0.0064 [-0.3492, 0.3460] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.4829 | 1.4275 | 0.0554 [-0.0960, 0.2290] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 0.8689 | 0.9719 | -0.1030 [-0.2194, -0.0215] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 91 | 16 | 5.3388 | 5.2793 | 0.0596 [-0.6713, 1.1737] |  |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.3882 | 2.2857 | 0.1025 [-0.2195, 0.3964] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.1805 | 1.3866 | -0.2061 [-0.3177, -0.0155] | **SEP** |  |
| undef (GT crawling) | v0 <5 (low n) | 19 | 13 | 9.8393 | 9.9210 | -0.0817 [-0.1752, 0.0342] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA8-30k s0-s1 [CI] | R7-30k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0059 [-0.0202, 0.0070] | -0.0060 [-0.0196, 0.0079] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0052 [-0.0161, 0.0043] | -0.0050 [-0.0158, 0.0047] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | -0.0149 [-0.0488, 0.0191] | 0.0210 [-0.0180, 0.0668] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | 0.0306 [-0.0063, 0.0728] | 0.0090 [-0.0310, 0.0579] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0400 [-0.1755, 0.0736] | -0.0870 [-0.1984, 0.0173] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.0799 [-0.2282, 0.0340] | -0.0060 [-0.0909, 0.0632] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | 0.0120 [-0.0444, 0.0723] | 0.0226 [-0.0617, 0.1081] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0358 [-0.1335, 0.0394] | -0.0262 [-0.1192, 0.0801] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0092 [-0.0383, 0.0182] | -0.0190 [-0.0394, 0.0006] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | -0.0032 [-0.0170, 0.0076] | 0.0070 [-0.0025, 0.0174] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | 0.0070 [-0.0027, 0.0165] | -0.0017 [-0.0088, 0.0049] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | -0.0003 [-0.0077, 0.0077] | 0.0017 [-0.0086, 0.0125] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0158 [-0.0315, -0.0024] SEP | 0.0003 [-0.0067, 0.0069] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0192 [-0.0354, 0.0718] | -0.0076 [-0.0615, 0.0391] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | 0.0093 [-0.0231, 0.0416] | -0.0201 [-0.0627, 0.0255] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- EMA8-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0356 | 0.0270 / 0.0441 | False / True / False | 0.0059 / 0.0110 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0433 | 0.0398 / 0.0469 | True / True / True | 0.0052 / 0.0019 | 8.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.0025 | -0.0021 / -0.0029 | False / False / False | 0.0149 / 0.0161 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0187 | 0.0183 / 0.0191 | False / False / False | 0.0306 / 0.0331 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.0784 | 0.0018 / 0.1549 | False / False / False | 0.0400 / 0.1080 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.0817 | -0.0949 / -0.0736 | False / False / False | 0.0799 / 0.0237 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0347 | 0.0074 / 0.0612 | False / False / False | 0.0120 / 0.0573 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.0214 | -0.0753 / 0.0325 | False / False / False | 0.0358 / 0.0719 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1292 | 0.1243 / 0.1341 | True / True / True | 0.0092 / 0.0006 | 14.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0094 | 0.0079 / 0.0110 | False / False / False | 0.0032 / 0.0001 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0314 | -0.0268 / -0.0361 | True / True / True | 0.0070 / 0.0023 | 4.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0006 | -0.0005 / -0.0007 | False / False / False | 0.0003 / 0.0005 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0471 | 0.0409 / 0.0532 | True / True / True | 0.0158 / 0.0036 | 3.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.1217 | -0.1443 / -0.0993 | False / False / False | 0.0192 / 0.0608 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0604 | 0.0508 / 0.0700 | False / False / False | 0.0093 / 0.0283 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- EMA8-30k - R7-30k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0187 | -0.0185 / -0.0188 | False / False / False | 0.0059 / 0.0060 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0114 | 0.0114 / 0.0115 | False / False / False | 0.0052 / 0.0050 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.0212 | -0.0393 / -0.0030 | False / False / False | 0.0149 / 0.0210 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0561 | -0.0441 / -0.0682 | False / False / False | 0.0306 / 0.0090 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | -0.2055 | -0.1821 / -0.2289 | False / False / False | 0.0400 / 0.0870 | 2.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.1109 | 0.0484 / 0.1623 | False / False / False | 0.0799 / 0.0060 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0249 | 0.0210 / 0.0303 | False / False / False | 0.0120 / 0.0226 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.1067 | -0.1115 / -0.1019 | False / False / False | 0.0358 / 0.0262 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0318 | 0.0367 / 0.0269 | False / False / False | 0.0092 / 0.0190 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.0222 | -0.0273 / -0.0171 | False / False / False | 0.0032 / 0.0070 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0530 | -0.0486 / -0.0573 | True / True / True | 0.0070 / 0.0017 | 7.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0214 | -0.0224 / -0.0204 | False / False / False | 0.0003 / 0.0017 | 12.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0089 | 0.0009 / 0.0170 | False / False / False | 0.0158 / 0.0003 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.2718 | -0.2588 / -0.2845 | True / True / True | 0.0192 / 0.0076 | 14.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.0002 | 0.0155 / -0.0150 | False / False / False | 0.0093 / 0.0201 | 0.0 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- R7-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0546 | 0.0462 / 0.0631 | True / True / True | 0.0060 / 0.0110 | 5.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0323 | 0.0289 / 0.0359 | True / True / True | 0.0050 / 0.0019 | 6.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0182 | 0.0368 / -0.0003 | True / False / False | 0.0210 / 0.0161 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0773 | 0.0653 / 0.0894 | False / False / False | 0.0090 / 0.0331 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.2802 | 0.1827 / 0.3776 | False / True / True | 0.0870 / 0.1080 | 2.6 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | -0.0778 | -0.0663 / -0.1043 | False / False / False | 0.0060 / 0.0237 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0203 | 0.0025 / 0.0372 | False / False / False | 0.0226 / 0.0573 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0853 | 0.0363 / 0.1344 | False / False / False | 0.0262 / 0.0719 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0974 | 0.0876 / 0.1072 | True / True / True | 0.0190 / 0.0006 | 5.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0317 | 0.0352 / 0.0281 | False / False / False | 0.0070 / 0.0001 | 4.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0215 | 0.0219 / 0.0212 | False / False / False | 0.0017 / 0.0023 | 9.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0208 | 0.0219 / 0.0197 | True / True / True | 0.0017 / 0.0005 | 12.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.0384 | 0.0404 / 0.0365 | True / True / True | 0.0003 / 0.0036 | 10.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.1433 | 0.1091 / 0.1774 | True / True / True | 0.0076 / 0.0608 | 2.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.0629 | 0.0386 / 0.0866 | False / True / False | 0.0201 / 0.0283 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1588): EMA8-30k bias +0.001, refcv7-50400 bias +0.081, R7-30k bias +0.046; EMA8-30k MAE 0.594, refcv7-50400 MAE 0.623, R7-30k MAE 0.668
* seg2 (n 1593): EMA8-30k bias -0.243, refcv7-50400 bias -0.098, R7-30k bias -0.188; EMA8-30k MAE 1.423, refcv7-50400 MAE 1.449, R7-30k MAE 1.549
* seg3 (n 1587): EMA8-30k bias -0.715, refcv7-50400 bias -0.620, R7-30k bias -0.808; EMA8-30k MAE 2.044, refcv7-50400 MAE 1.986, R7-30k MAE 2.093
* seg4 (n 1577): EMA8-30k bias -0.888, refcv7-50400 bias -0.720, R7-30k bias -1.132; EMA8-30k MAE 2.768, refcv7-50400 MAE 2.686, R7-30k MAE 2.819
**sharp (>= 0.15)**

* seg1 (n 366): EMA8-30k bias -0.249, refcv7-50400 bias +0.010, R7-30k bias -0.123; EMA8-30k MAE 0.856, refcv7-50400 MAE 0.913, R7-30k MAE 0.994
* seg2 (n 366): EMA8-30k bias -1.026, refcv7-50400 bias -0.528, R7-30k bias -0.822; EMA8-30k MAE 2.384, refcv7-50400 MAE 2.413, R7-30k MAE 2.645
* seg3 (n 365): EMA8-30k bias -2.292, refcv7-50400 bias -1.863, R7-30k bias -2.425; EMA8-30k MAE 3.834, refcv7-50400 MAE 3.649, R7-30k MAE 3.969
* seg4 (n 363): EMA8-30k bias -2.647, refcv7-50400 bias -1.837, R7-30k bias -3.178; EMA8-30k MAE 5.130, refcv7-50400 MAE 4.887, R7-30k MAE 5.332
**curved & v0 5-10**

* seg1 (n 627): EMA8-30k bias -0.100, refcv7-50400 bias -0.038, R7-30k bias -0.061; EMA8-30k MAE 0.520, refcv7-50400 MAE 0.535, R7-30k MAE 0.598
* seg2 (n 627): EMA8-30k bias -0.391, refcv7-50400 bias -0.282, R7-30k bias -0.325; EMA8-30k MAE 1.413, refcv7-50400 MAE 1.363, R7-30k MAE 1.525
* seg3 (n 627): EMA8-30k bias -0.905, refcv7-50400 bias -0.871, R7-30k bias -0.963; EMA8-30k MAE 2.213, refcv7-50400 MAE 2.019, R7-30k MAE 2.199
* seg4 (n 627): EMA8-30k bias -1.210, refcv7-50400 bias -1.194, R7-30k bias -1.457; EMA8-30k MAE 2.999, refcv7-50400 MAE 2.730, R7-30k MAE 2.978
**sharp & v0 5-10**

* seg1 (n 164): EMA8-30k bias -0.100, refcv7-50400 bias +0.103, R7-30k bias +0.060; EMA8-30k MAE 0.761, refcv7-50400 MAE 0.857, R7-30k MAE 0.971
* seg2 (n 164): EMA8-30k bias -0.787, refcv7-50400 bias -0.422, R7-30k bias -0.444; EMA8-30k MAE 2.220, refcv7-50400 MAE 2.243, R7-30k MAE 2.557
* seg3 (n 164): EMA8-30k bias -2.249, refcv7-50400 bias -2.032, R7-30k bias -2.198; EMA8-30k MAE 3.749, refcv7-50400 MAE 3.465, R7-30k MAE 3.740
* seg4 (n 164): EMA8-30k bias -2.866, refcv7-50400 bias -2.403, R7-30k bias -3.228; EMA8-30k MAE 5.181, refcv7-50400 MAE 4.635, R7-30k MAE 5.151

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1577 / 1577 | 10.2 | EMA8-30k -0.1199 [-0.2038, -0.0573]; refcv7-50400 -0.0847 [-0.1561, -0.0276]; R7-30k -0.1292 [-0.2017, -0.0666] | EMA8-30k -0.0352 [-0.0623, -0.0103] SEP; R7-30k -0.0445 [-0.0653, -0.0227] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1596 / 4757 | 7.2 | EMA8-30k -0.1793 [-0.3021, -0.0807]; refcv7-50400 -0.1468 [-0.2658, -0.0524]; R7-30k -0.1805 [-0.2999, -0.0840] | EMA8-30k -0.0325 [-0.0560, -0.0104] SEP; R7-30k -0.0337 [-0.0539, -0.0135] SEP |
| sharp (>= 0.15) | last segment | 363 / 363 | 25.4 | EMA8-30k -0.1337 [-0.2298, -0.0605]; refcv7-50400 -0.0921 [-0.1765, -0.0255]; R7-30k -0.1434 [-0.2287, -0.0705] | EMA8-30k -0.0415 [-0.0734, -0.0132] SEP; R7-30k -0.0513 [-0.0760, -0.0269] SEP |
| sharp (>= 0.15) | segments 2..K | 366 / 1094 | 17.9 | EMA8-30k -0.1446 [-0.2426, -0.0700]; refcv7-50400 -0.1041 [-0.1873, -0.0360]; R7-30k -0.1455 [-0.2338, -0.0702] | EMA8-30k -0.0405 [-0.0693, -0.0143] SEP; R7-30k -0.0414 [-0.0650, -0.0173] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | EMA8-30k -0.1148 [-0.1813, -0.0614]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-30k -0.1170 [-0.1981, -0.0539] | EMA8-30k -0.0223 [-0.0586, +0.0222]; R7-30k -0.0245 [-0.0621, +0.0105] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | EMA8-30k -0.1221 [-0.1904, -0.0668]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-30k -0.1147 [-0.1962, -0.0510] | EMA8-30k -0.0228 [-0.0572, +0.0181]; R7-30k -0.0154 [-0.0515, +0.0188] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | EMA8-30k -0.1177 [-0.1920, -0.0594]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-30k -0.1195 [-0.2080, -0.0502] | EMA8-30k -0.0265 [-0.0655, +0.0189]; R7-30k -0.0282 [-0.0696, +0.0083] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | EMA8-30k -0.1217 [-0.1977, -0.0607]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-30k -0.1131 [-0.2023, -0.0424] | EMA8-30k -0.0262 [-0.0630, +0.0165]; R7-30k -0.0176 [-0.0578, +0.0180] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA8-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4517 | 0.0356 | -0.00071 | -0.00269 | 0.497 / 0.503 / 0.000 | 299.0 / 160.6 |
| turn|straight <0.02 | 2902 | 0.0433 | 0.00144 | 0.00304 | 0.509 / 0.491 / 0.000 | 134.1 / 125.8 |
| turn|gentle 0.02-0.06 | 720 | -0.0025 | -0.02985 | -0.03484 | 0.453 / 0.547 / 0.000 | 41.0 / -1.8 |
| turn|turn 0.06-0.15 | 510 | 0.0187 | -0.00620 | -0.01787 | 0.490 / 0.510 / 0.000 | 34.6 / 9.5 |
| turn|sharp >=0.15 | 366 | 0.0784 | 0.00833 | -0.00880 | 0.508 / 0.492 / 0.000 | 55.3 / 28.7 |
| turn|undef (GT crawling) | 19 | -0.0817 | -0.03382 | -0.06171 | 0.368 / 0.632 / 0.000 | 0.2 / -1.6 |
| speed|v0 <2 | 228 | 0.0347 | 0.01031 | 0.00899 | 0.522 / 0.478 / 0.000 | 14.2 / 7.9 |
| speed|v0 2-5 | 516 | -0.0214 | -0.03722 | -0.04958 | 0.442 / 0.558 / 0.000 | 63.9 / -11.0 |
| speed|v0 5-10 | 1390 | 0.1292 | 0.00537 | 0.02298 | 0.515 / 0.485 / 0.000 | 115.3 / 179.6 |
| speed|v0 10-15 | 1192 | 0.0094 | 0.00343 | 0.00443 | 0.517 / 0.483 / 0.000 | 25.1 / 11.2 |
| speed|v0 15-25 | 856 | -0.0314 | -0.00654 | -0.01381 | 0.472 / 0.528 / 0.000 | 6.2 / -26.9 |
| speed|v0 >=25 | 335 | -0.0006 | -0.00104 | -0.00176 | 0.487 / 0.513 / 0.000 | 2.5 / -0.2 |
| nav|nav=follow | 2882 | 0.0471 | -0.00143 | -0.00385 | 0.494 / 0.506 / 0.000 | 152.5 / 135.6 |
| nav|nav=left | 405 | -0.1217 | 0.01106 | -0.00753 | 0.523 / 0.477 / 0.000 | 15.2 / -49.3 |
| nav|nav=right | 1230 | 0.0604 | -0.00271 | -0.00004 | 0.496 / 0.504 / 0.000 | 107.9 / 74.3 |
**EMA8-30k - R7-30k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4509 | -0.0187 | -0.03134 | -0.03968 | 0.422 / 0.578 / 0.000 | 286.0 / -84.4 |
| turn|straight <0.02 | 2897 | 0.0114 | -0.02427 | -0.02675 | 0.421 / 0.579 / 0.000 | 142.1 / 32.9 |
| turn|gentle 0.02-0.06 | 720 | -0.0212 | -0.04030 | -0.05823 | 0.432 / 0.568 / 0.000 | 40.3 / -15.2 |
| turn|turn 0.06-0.15 | 510 | -0.0561 | -0.09463 | -0.06918 | 0.429 / 0.571 / 0.000 | 33.7 / -28.6 |
| turn|sharp >=0.15 | 366 | -0.2055 | -0.17787 | -0.22784 | 0.407 / 0.593 / 0.000 | 43.9 / -75.2 |
| turn|undef (GT crawling) | 16 | 0.1109 | -0.02104 | 0.02830 | 0.438 / 0.562 / 0.000 | 1.2 / 1.8 |
| speed|v0 <2 | 220 | 0.0249 | -0.00186 | 0.01146 | 0.495 / 0.505 / 0.000 | 12.4 / 5.5 |
| speed|v0 2-5 | 516 | -0.1067 | -0.08698 | -0.11300 | 0.374 / 0.626 / 0.000 | 69.5 / -55.1 |
| speed|v0 5-10 | 1390 | 0.0318 | -0.03515 | -0.03941 | 0.433 / 0.567 / 0.000 | 102.5 / 44.2 |
| speed|v0 10-15 | 1192 | -0.0222 | -0.02832 | -0.03365 | 0.426 / 0.574 / 0.000 | 23.5 / -26.5 |
| speed|v0 15-25 | 856 | -0.0530 | -0.02720 | -0.03681 | 0.436 / 0.564 / 0.000 | 7.9 / -45.3 |
| speed|v0 >=25 | 335 | -0.0214 | -0.02759 | -0.02384 | 0.358 / 0.642 / 0.000 | 2.5 / -7.2 |
| nav|nav=follow | 2874 | 0.0089 | -0.02452 | -0.02926 | 0.428 / 0.572 / 0.000 | 139.8 / 25.4 |
| nav|nav=left | 405 | -0.2718 | -0.04408 | -0.08656 | 0.417 / 0.583 / 0.000 | 13.2 / -110.1 |
| nav|nav=right | 1230 | 0.0002 | -0.05194 | -0.05979 | 0.411 / 0.589 / 0.000 | 110.2 / 0.3 |
**R7-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4527 | 0.0546 | 0.02944 | 0.03463 | 0.593 / 0.407 / 0.000 | 188.0 / 247.4 |
| turn|straight <0.02 | 2898 | 0.0323 | 0.02691 | 0.02930 | 0.607 / 0.393 / 0.000 | 45.8 / 93.7 |
| turn|gentle 0.02-0.06 | 721 | 0.0182 | 0.02245 | 0.02962 | 0.556 / 0.444 / 0.000 | 12.0 / 13.1 |
| turn|turn 0.06-0.15 | 512 | 0.0773 | 0.04893 | 0.04963 | 0.568 / 0.432 / 0.000 | 34.8 / 39.6 |
| turn|sharp >=0.15 | 368 | 0.2802 | 0.19304 | 0.17998 | 0.595 / 0.405 / 0.000 | 34.4 / 103.1 |
| turn|undef (GT crawling) | 28 | -0.0778 | -0.01158 | -0.03977 | 0.464 / 0.536 / 0.000 | 0.5 / -2.2 |
| speed|v0 <2 | 238 | 0.0203 | -0.00323 | -0.00507 | 0.479 / 0.521 / 0.000 | 7.5 / 4.8 |
| speed|v0 2-5 | 516 | 0.0853 | 0.04064 | 0.06172 | 0.568 / 0.432 / 0.000 | 30.4 / 44.0 |
| speed|v0 5-10 | 1390 | 0.0974 | 0.04238 | 0.05221 | 0.601 / 0.399 / 0.000 | 82.2 / 135.4 |
| speed|v0 10-15 | 1192 | 0.0317 | 0.03376 | 0.03713 | 0.617 / 0.383 / 0.000 | 18.6 / 37.7 |
| speed|v0 15-25 | 856 | 0.0215 | 0.01964 | 0.02224 | 0.585 / 0.415 / 0.000 | 9.5 / 18.4 |
| speed|v0 >=25 | 335 | 0.0208 | 0.02487 | 0.01984 | 0.609 / 0.391 / 0.000 | 2.1 / 7.0 |
| nav|nav=follow | 2876 | 0.0384 | 0.02257 | 0.02400 | 0.585 / 0.415 / 0.000 | 91.8 / 110.3 |
| nav|nav=left | 414 | 0.1433 | 0.04963 | 0.07269 | 0.659 / 0.341 / 0.000 | 24.5 / 59.3 |
| nav|nav=right | 1237 | 0.0629 | 0.04477 | 0.05271 | 0.589 / 0.411 / 0.000 | 58.4 / 77.8 |

### S2 -- the episodes carrying the summed delta (panel file index only)

**EMA8-30k - refcv7-50400** (sum 160.6 over 139 episodes)

* ep013: n 34, share 0.194, mean delta 0.9159, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep027: n 34, share 0.147, mean delta 0.6941, sharp-window frac 0.00, curved frac 0.09, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep061: n 34, share 0.131, mean delta 0.6182, sharp-window frac 0.62, curved frac 1.00, mean v0 9.5, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep059: n 34, share 0.122, mean delta 0.5771, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep041: n 34, share 0.112, mean delta 0.5286, sharp-window frac 0.18, curved frac 0.47, mean v0 9.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep080: n 34, share 0.100, mean delta 0.4724, sharp-window frac 0.18, curved frac 0.32, mean v0 9.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**EMA8-30k - R7-30k** (sum -84.4 over 139 episodes)

* ep078: n 34, share -0.274, mean delta 0.6791, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep027: n 34, share -0.264, mean delta 0.6543, sharp-window frac 0.00, curved frac 0.09, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep013: n 34, share -0.221, mean delta 0.5495, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep059: n 34, share -0.200, mean delta 0.4972, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep029: n 34, share -0.167, mean delta 0.4152, sharp-window frac 0.24, curved frac 1.00, mean v0 10.8, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep036: n 34, share -0.167, mean delta 0.4147, sharp-window frac 0.26, curved frac 0.50, mean v0 12.4, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
**R7-30k - refcv7-50400** (sum 247.4 over 139 episodes)

* ep111: n 34, share 0.121, mean delta 0.8803, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep067: n 34, share 0.064, mean delta 0.4648, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep035: n 34, share 0.063, mean delta 0.4600, sharp-window frac 0.03, curved frac 0.24, mean v0 13.4, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep028: n 34, share 0.056, mean delta 0.4047, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep080: n 34, share 0.054, mean delta 0.3912, sharp-window frac 0.18, curved frac 0.32, mean v0 9.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep013: n 34, share 0.050, mean delta 0.3664, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA8-30k vs R7-30k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1642 / -0.0607 [-0.0823, -0.0413] | 54 / -0.3271 [-0.7284, -0.0200] |
| only new agrees | 359 / -0.2188 [-0.3482, -0.1045] | 45 / -0.4522 [-1.2615, 0.1376] |
| only earlier agrees | 895 / 0.1097 [0.0409, 0.1996] | 46 / 0.6924 [-0.2533, 1.8913] |
| neither agrees | 1615 / -0.0022 [-0.0811, 0.0729] | 221 / -0.2737 [-0.5474, -0.0312] |
**EMA8-30k vs R7-30k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1655 / -0.0522 [-0.0717, -0.0353] | 64 / -0.1776 [-0.5087, 0.0783] |
| only new agrees | 329 / -0.2816 [-0.4650, -0.1322] | 37 / -1.5141 [-2.9466, -0.3582] |
| only earlier agrees | 886 / 0.1121 [0.0266, 0.2183] | 44 / 0.9628 [-0.2735, 2.7921] |
| neither agrees | 1639 / -0.0031 [-0.1114, 0.0984] | 221 / -0.2659 [-0.6486, 0.0591] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA8-30k vs R7-30k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4486 | 0.0547 [-0.0062, 0.1188] | 0.0309 [-0.0266, 0.1012] | 0.0293 [-0.0213, 0.0778] |
| turn|straight <0.02 | 2892 | 0.0334 [-0.0214, 0.0955] | -0.0240 [-0.0506, 0.0055] | 0.0576 [0.0052, 0.1180] SEP |
| turn|gentle 0.02-0.06 | 719 | -0.0549 [-0.1827, 0.0390] | -0.0180 [-0.1334, 0.0773] | -0.0344 [-0.1216, 0.0378] |
| turn|turn 0.06-0.15 | 500 | 0.0675 [-0.0614, 0.2293] | 0.1191 [-0.0362, 0.3461] | -0.0453 [-0.1989, 0.0791] |
| turn|sharp >=0.15 | 365 | 0.4190 [0.0479, 0.8677] | 0.4356 [0.0750, 0.8390] | 0.0347 [-0.2144, 0.3272] |
| turn|undef (GT crawling) | 10 | 0.1385 [-0.0055, 0.3179] | 0.0519 [-0.1242, 0.2388] | -0.0461 [-0.5253, 0.2745] |
| speed|v0 <2 | 197 | 0.0371 [-0.1303, 0.1928] | 0.3004 [0.1136, 0.5311] | -0.1855 [-0.4895, 0.0480] |
| speed|v0 2-5 | 516 | 0.4172 [-0.0239, 0.8760] | 0.2345 [-0.1783, 0.7598] | 0.1828 [-0.1493, 0.5537] |
| speed|v0 5-10 | 1390 | 0.0264 [-0.0607, 0.1031] | -0.0151 [-0.0764, 0.0472] | 0.0415 [-0.0252, 0.1033] |
| speed|v0 10-15 | 1192 | -0.0106 [-0.0353, 0.0105] | -0.0153 [-0.0289, -0.0004] | 0.0047 [-0.0190, 0.0268] |
| speed|v0 15-25 | 856 | -0.0008 [-0.0216, 0.0181] | -0.0135 [-0.0236, -0.0044] | 0.0127 [-0.0044, 0.0284] |
| speed|v0 >=25 | 335 | -0.0018 [-0.0193, 0.0106] | -0.0001 [-0.0096, 0.0089] | -0.0017 [-0.0236, 0.0162] |
**EMA8-30k vs R7-30k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* EMA8-30k vs R7-30k|seed0: anchor flip rate 0.435; heading delta on flipped windows 0.0410 [-0.0428, 0.1208] (n 1991) vs unflipped -0.0656 [-0.0858, -0.0446]; goal_dist_sel proxy new-earlier -0.2311 [-0.3878, -0.0825]; corr(window dHeading, window dGoalDist) 0.001
* EMA8-30k vs R7-30k|seed1: anchor flip rate 0.436; heading delta on flipped windows 0.0405 [-0.0637, 0.1458] (n 1987) vs unflipped -0.0655 [-0.0857, -0.0453]; goal_dist_sel proxy new-earlier -0.2324 [-0.3892, -0.0845]; corr(window dHeading, window dGoalDist) -0.003

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- EMA8-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3542 | 139 | 3.4092 | 3.2462 | 0.1630 [-0.0098, 0.3535] |  |  |
| turn: straight <0.02 | 2107 | 123 | 1.6963 | 1.4627 | 0.2336 [0.0697, 0.4866] | **SEP** | 0.595 / 0.853 / 0.1744 [-0.4217, 0.7635] |
| turn: gentle 0.02-0.06 | 676 | 77 | 3.3812 | 2.8878 | 0.4934 [0.1450, 0.8820] | **SEP** | 0.191 / 0.578 / 0.4084 [-0.1651, 1.1324] |
| turn: turn 0.06-0.15 | 436 | 59 | 6.1347 | 6.3573 | -0.2225 [-0.5134, 0.0803] |  | 0.123 / -0.168 / -0.4396 [-0.9301, 0.0705] |
| turn: sharp >=0.15 | 306 | 33 | 10.9445 | 11.4553 | -0.5108 [-1.7459, 0.7764] |  | 0.086 / -0.271 / -0.7375 [-2.5730, 1.1874] |
| turn: undef (GT crawling) (low n) | 17 | 12 | 11.2732 | 10.9951 | 0.2782 [0.0731, 0.5069] | **SEP** | 0.005 / 0.008 / 0.1158 [-0.3119, 0.5509] |
| speed: v0 <2 | 204 | 27 | 12.9442 | 12.9236 | 0.0206 [-0.6561, 0.7651] |  | 0.058 / 0.007 / -0.1510 [-1.3156, 1.1063] |
| speed: v0 2-5 | 422 | 52 | 4.5102 | 4.1501 | 0.3601 [-0.3058, 1.0506] |  | 0.119 / 0.263 / 0.2238 [-0.7229, 1.2738] |
| speed: v0 5-10 | 1040 | 79 | 3.8183 | 3.4941 | 0.3242 [-0.1444, 0.8734] |  | 0.294 / 0.584 / 0.2282 [-0.5217, 1.0642] |
| speed: v0 10-15 | 952 | 62 | 2.5042 | 2.4870 | 0.0171 [-0.1586, 0.2005] |  | 0.269 / 0.028 / -0.1994 [-0.6523, 0.2528] |
| speed: v0 15-25 | 679 | 34 | 1.5580 | 1.4456 | 0.1124 [0.0082, 0.2333] | **SEP** | 0.192 / 0.132 / -0.0625 [-0.4279, 0.2972] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4836 | 0.5193 | -0.0357 [-0.0791, 0.0222] |  | 0.069 / -0.015 / -0.2134 [-0.5371, 0.0777] |
| nav: nav=follow | 2256 | 88 | 2.0783 | 1.9749 | 0.1034 [-0.0131, 0.2214] |  | 0.637 / 0.404 / -0.1640 [-0.9491, 0.4869] |
| nav: nav=left (low n) | 325 | 13 | 9.3558 | 9.7225 | -0.3667 [-1.2230, 0.4430] |  | 0.092 / -0.206 / -0.5832 [-2.0572, 0.9356] |
| nav: nav=right | 961 | 38 | 4.5223 | 4.0404 | 0.4819 [-0.0276, 1.0826] |  | 0.271 / 0.802 / 0.4377 [-0.3727, 1.3201] |

episode concentration: 0.576 of episodes have mean delta > 0; top-5 episodes carry 0.944 of the summed delta; delta excluding the top 5 = 0.0095 (overall 0.1630).

### S6 -- EMA8-30k - R7-30k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3527 | 139 | 3.4170 | 3.4311 | -0.0141 [-0.2357, 0.2138] |  |  |
| turn: straight <0.02 | 2095 | 122 | 1.6968 | 1.4949 | 0.2019 [0.0346, 0.4300] | **SEP** | 0.594 / -8.534 / 0.5319 [-0.1955, 1.2987] |
| turn: gentle 0.02-0.06 | 675 | 77 | 3.3826 | 2.9231 | 0.4595 [0.0249, 0.9928] | **SEP** | 0.191 / -6.257 / 0.5856 [-0.0941, 1.5449] |
| turn: turn 0.06-0.15 | 436 | 59 | 6.1347 | 6.7069 | -0.5722 [-0.9513, -0.2135] | **SEP** | 0.124 / 5.033 / -0.6368 [-1.1992, 0.0353] |
| turn: sharp >=0.15 | 306 | 33 | 10.9445 | 12.6947 | -1.7502 [-3.4772, -0.2798] | **SEP** | 0.087 / 10.805 / -1.9011 [-4.6055, 0.4150] |
| turn: undef (GT crawling) (low n) | 15 | 10 | 12.6598 | 12.5055 | 0.1543 [-0.0423, 0.4050] |  | 0.004 / -0.047 / 0.1691 [-0.3391, 0.6780] |
| speed: v0 <2 | 189 | 26 | 13.8473 | 13.8549 | -0.0076 [-0.6363, 0.6404] |  | 0.054 / 0.029 / 0.0068 [-1.1778, 1.0540] |
| speed: v0 2-5 | 422 | 52 | 4.5102 | 4.4408 | 0.0694 [-0.7578, 0.8083] |  | 0.120 / -0.591 / 0.0948 [-1.0270, 1.4250] |
| speed: v0 5-10 | 1040 | 79 | 3.8183 | 3.7918 | 0.0265 [-0.5507, 0.6201] |  | 0.295 / -0.555 / 0.0575 [-0.8583, 0.8619] |
| speed: v0 10-15 | 952 | 62 | 2.5042 | 2.6243 | -0.1201 [-0.3057, 0.0314] |  | 0.270 / 2.307 / -0.1453 [-0.6430, 0.3827] |
| speed: v0 15-25 | 679 | 34 | 1.5580 | 1.5277 | 0.0303 [-0.0911, 0.1667] |  | 0.193 / -0.415 / 0.0549 [-0.3603, 0.5106] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4836 | 0.5291 | -0.0455 [-0.1266, 0.0422] |  | 0.069 / 0.225 / -0.0338 [-0.4243, 0.3269] |
| nav: nav=follow | 2243 | 88 | 2.0848 | 2.1144 | -0.0295 [-0.1396, 0.0661] |  | 0.636 / 1.337 / -0.0426 [-0.8980, 0.8250] |
| nav: nav=left (low n) | 323 | 13 | 9.3796 | 10.1162 | -0.7366 [-1.7771, 0.1574] |  | 0.092 / 4.800 / -0.7954 [-2.6419, 0.7630] |
| nav: nav=right | 961 | 38 | 4.5223 | 4.2573 | 0.2650 [-0.4288, 1.0313] |  | 0.272 / -5.137 / 0.3835 [-0.6678, 1.3770] |

episode concentration: 0.525 of episodes have mean delta > 0; top-5 episodes carry -11.118 of the summed delta; delta excluding the top 5 = -0.1770 (overall -0.0141).

### S6 -- R7-30k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3549 | 139 | 3.4846 | 3.2938 | 0.1908 [0.0588, 0.3349] | **SEP** |  |
| turn: straight <0.02 | 2100 | 122 | 1.4952 | 1.4589 | 0.0362 [-0.0110, 0.0859] |  | 0.592 / 0.112 / -0.3785 [-0.8609, 0.0334] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.9231 | 2.8900 | 0.0331 [-0.1546, 0.1783] |  | 0.190 / 0.033 / -0.1947 [-0.5714, 0.1053] |
| turn: turn 0.06-0.15 | 441 | 59 | 6.6920 | 6.3357 | 0.3563 [0.1050, 0.6272] | **SEP** | 0.124 / 0.232 / 0.1890 [-0.2408, 0.6244] |
| turn: sharp >=0.15 | 312 | 33 | 13.0227 | 11.7318 | 1.2909 [0.0867, 2.7274] | **SEP** | 0.088 / 0.595 / 1.2061 [-0.4307, 3.5327] |
| turn: undef (GT crawling) (low n) | 21 | 10 | 11.4093 | 10.5155 | 0.8938 [0.0024, 1.9559] | **SEP** | 0.006 / 0.028 / 0.7072 [-0.4215, 2.2930] |
| speed: v0 <2 | 211 | 28 | 13.6683 | 13.4029 | 0.2654 [-0.1792, 0.6725] |  | 0.059 / 0.083 / 0.0793 [-0.8546, 0.7086] |
| speed: v0 2-5 | 422 | 52 | 4.4408 | 4.1501 | 0.2907 [-0.1266, 0.7510] |  | 0.119 / 0.181 / 0.1133 [-0.5921, 0.8690] |
| speed: v0 5-10 | 1040 | 79 | 3.7918 | 3.4941 | 0.2977 [-0.0265, 0.7041] |  | 0.293 / 0.457 / 0.1512 [-0.3190, 0.8380] |
| speed: v0 10-15 | 952 | 62 | 2.6243 | 2.4870 | 0.1373 [-0.0097, 0.2911] |  | 0.268 / 0.193 / -0.0732 [-0.4397, 0.2685] |
| speed: v0 15-25 | 679 | 34 | 1.5277 | 1.4456 | 0.0821 [-0.0317, 0.2002] |  | 0.191 / 0.082 / -0.1343 [-0.4292, 0.1215] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5291 | 0.5193 | 0.0098 [-0.0491, 0.0663] |  | 0.069 / 0.004 / -0.1944 [-0.4372, -0.0034] BONF-SEP |
| nav: nav=follow | 2246 | 88 | 2.1125 | 1.9797 | 0.1327 [0.0064, 0.2914] | **SEP** | 0.633 / 0.440 / -0.1582 [-0.6592, 0.2335] |
| nav: nav=left (low n) | 327 | 13 | 10.3337 | 9.9260 | 0.4077 [-0.0314, 0.9220] |  | 0.092 / 0.197 / 0.2389 [-0.4585, 1.1807] |
| nav: nav=right | 976 | 38 | 4.3474 | 4.0957 | 0.2518 [-0.0511, 0.5890] |  | 0.275 / 0.363 / 0.0841 [-0.3758, 0.6657] |

episode concentration: 0.604 of episodes have mean delta > 0; top-5 episodes carry 0.633 of the summed delta; delta excluding the top 5 = 0.0727 (overall 0.1908).

### S6 -- EMA8-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3495 | 139 | 1.0163 | 0.7642 | 0.2521 [-0.0264, 0.7844] |  | 0.1298 [-0.0530, 0.4575] |
| seg1 | straight | 2070 | 122 | 0.8217 | 0.5436 | 0.2780 [0.0056, 0.8533] | **SEP** | 0.2382 [-0.0392, 0.8182] |
| seg1 | curved | 1409 | 88 | 1.1794 | 0.9648 | 0.2146 [-0.0833, 0.7824] |  | -0.0247 [-0.1099, 0.0575] |
| seg2 | all | 3484 | 139 | 1.5849 | 1.4596 | 0.1253 [0.0170, 0.2637] | **SEP** | 0.2572 [-0.1848, 1.0228] |
| seg2 | straight | 2077 | 122 | 0.9912 | 0.8383 | 0.1529 [0.0222, 0.3631] | **SEP** | 0.3552 [-0.1380, 1.3162] |
| seg2 | curved | 1407 | 88 | 2.4613 | 2.3767 | 0.0845 [-0.0687, 0.2410] |  | 0.1126 [-0.3397, 0.6984] |
| seg3 | all | 3454 | 139 | 2.4217 | 2.0640 | 0.3577 [0.0480, 0.8475] | **SEP** | 0.3918 [-0.3805, 1.7091] |
| seg3 | straight | 2061 | 121 | 1.4531 | 0.9769 | 0.4762 [0.0284, 1.2787] | **SEP** | 0.7104 [-0.2810, 2.6367] |
| seg3 | curved | 1393 | 88 | 3.8548 | 3.6724 | 0.1824 [-0.0986, 0.4764] |  | -0.0796 [-0.7223, 0.5998] |
| seg4 | all | 3423 | 139 | 3.2893 | 2.9644 | 0.3249 [0.0356, 0.6601] | **SEP** | 0.0530 [-0.5866, 0.9724] |
| seg4 | straight | 2050 | 121 | 1.5549 | 1.1381 | 0.4168 [0.0709, 0.9094] | **SEP** | 0.4638 [-0.4380, 2.0639] |
| seg4 | curved | 1373 | 87 | 5.8788 | 5.6911 | 0.1877 [-0.2667, 0.6555] |  | -0.5604 [-1.2308, 0.0682] |
| seg5 | all | 3374 | 139 | 4.6469 | 4.2943 | 0.3526 [-0.0414, 0.7722] |  | -0.0326 [-0.8796, 1.1916] |
| seg5 | straight | 2023 | 122 | 1.8745 | 1.3127 | 0.5618 [0.1355, 1.1500] | **SEP** | 0.5069 [-0.6917, 2.6013] |
| seg5 | curved | 1351 | 88 | 8.7982 | 8.7590 | 0.0392 [-0.6530, 0.7569] |  | -0.8404 [-1.7515, 0.0478] |
| seg6 | all | 3346 | 139 | 6.1834 | 5.9256 | 0.2578 [-0.1578, 0.7293] |  | -0.6358 [-1.3269, 0.1210] |
| seg6 | straight | 2019 | 122 | 2.1654 | 1.7338 | 0.4316 [-0.0088, 0.9649] |  | -0.1124 [-0.9746, 1.0825] |
| seg6 | curved | 1326 | 88 | 12.3050 | 12.3125 | -0.0075 [-0.8041, 0.8230] |  | -1.4342 [-2.5451, -0.3517] SEP |

### S6 -- EMA8-30k - R7-30k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3493 | 139 | 1.0168 | 0.8019 | 0.2149 [-0.0604, 0.7420] |  | 0.0018 [-0.1850, 0.3357] |
| seg1 | straight | 2069 | 122 | 0.8220 | 0.5622 | 0.2599 [-0.0135, 0.8310] |  | 0.1421 [-0.1370, 0.7190] |
| seg1 | curved | 1409 | 88 | 1.1794 | 1.0306 | 0.1488 [-0.1446, 0.7014] |  | -0.1984 [-0.2770, -0.1264] SEP |
| seg2 | all | 3482 | 139 | 1.5858 | 1.4254 | 0.1604 [-0.0417, 0.5039] |  | -0.2315 [-0.4853, 0.1297] |
| seg2 | straight | 2075 | 122 | 0.9922 | 0.7862 | 0.2060 [-0.0131, 0.6051] |  | -0.0840 [-0.3171, 0.3195] |
| seg2 | curved | 1407 | 88 | 2.4613 | 2.3682 | 0.0931 [-0.1304, 0.4007] |  | -0.4491 [-0.7760, -0.0643] SEP |
| seg3 | all | 3447 | 139 | 2.4257 | 2.4080 | 0.0178 [-0.1403, 0.1840] |  | -0.6250 [-0.9162, -0.3095] SEP |
| seg3 | straight | 2054 | 121 | 1.4566 | 1.3778 | 0.0788 [-0.0423, 0.2445] |  | -0.2672 [-0.5695, 0.1680] |
| seg3 | curved | 1393 | 88 | 3.8548 | 3.9269 | -0.0721 [-0.3806, 0.2597] |  | -1.1527 [-1.6568, -0.7004] SEP |
| seg4 | all | 3408 | 139 | 3.3002 | 3.3020 | -0.0018 [-0.2926, 0.2982] |  | -0.9131 [-1.2651, -0.5835] SEP |
| seg4 | straight | 2037 | 121 | 1.5591 | 1.3578 | 0.2013 [-0.0006, 0.4870] |  | -0.5092 [-0.8109, -0.2827] SEP |
| seg4 | curved | 1371 | 87 | 5.8871 | 6.1907 | -0.3035 [-0.9095, 0.3019] |  | -1.5132 [-2.2759, -0.7898] SEP |
| seg5 | all | 3358 | 139 | 4.6573 | 4.6443 | 0.0130 [-0.4217, 0.4428] |  | -0.9139 [-1.4179, -0.4359] SEP |
| seg5 | straight | 2009 | 121 | 1.8778 | 1.4078 | 0.4699 [0.1281, 0.9234] | **SEP** | -0.4625 [-0.9195, -0.0636] SEP |
| seg5 | curved | 1349 | 88 | 8.7968 | 9.4642 | -0.6675 [-1.5599, 0.2387] |  | -1.5861 [-2.6427, -0.5832] SEP |
| seg6 | all | 3322 | 139 | 6.2053 | 6.2655 | -0.0602 [-0.5685, 0.4794] |  | -1.1256 [-1.7251, -0.5050] SEP |
| seg6 | straight | 2001 | 121 | 2.1648 | 1.6341 | 0.5307 [0.1504, 1.0429] | **SEP** | -0.7945 [-1.4940, -0.2601] SEP |
| seg6 | curved | 1321 | 88 | 12.3256 | 13.2809 | -0.9552 [-1.9879, 0.0819] |  | -1.6271 [-2.9002, -0.4051] SEP |

### S6 -- R7-30k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3503 | 139 | 0.8005 | 0.8142 | -0.0137 [-0.1206, 0.0493] |  | 0.1795 [0.1104, 0.2916] SEP |
| seg1 | straight | 2074 | 122 | 0.5614 | 0.5429 | 0.0185 [0.0082, 0.0302] | **SEP** | 0.0957 [0.0798, 0.1134] SEP |
| seg1 | curved | 1410 | 88 | 1.0299 | 1.0917 | -0.0618 [-0.3267, 0.0898] |  | 0.3011 [0.1362, 0.5761] SEP |
| seg2 | all | 3487 | 139 | 1.4261 | 1.4610 | -0.0349 [-0.2550, 0.1046] |  | 0.4897 [0.2436, 0.9216] SEP |
| seg2 | straight | 2076 | 122 | 0.7862 | 0.8392 | -0.0530 [-0.2493, 0.0510] |  | 0.4392 [0.1599, 1.0155] SEP |
| seg2 | curved | 1411 | 88 | 2.3676 | 2.3758 | -0.0082 [-0.2936, 0.2017] |  | 0.5640 [0.3356, 0.8379] SEP |
| seg3 | all | 3457 | 139 | 2.4200 | 2.0736 | 0.3464 [0.0432, 0.8129] | **SEP** | 1.0200 [0.4230, 2.0968] SEP |
| seg3 | straight | 2055 | 121 | 1.3783 | 0.9797 | 0.3986 [0.0163, 1.1768] | **SEP** | 0.9783 [0.2482, 2.4994] SEP |
| seg3 | curved | 1402 | 88 | 3.9468 | 3.6768 | 0.2699 [-0.0079, 0.5586] |  | 1.0811 [0.6257, 1.6396] SEP |
| seg4 | all | 3429 | 139 | 3.3254 | 2.9935 | 0.3318 [0.0788, 0.6548] | **SEP** | 0.9641 [0.4202, 1.8581] SEP |
| seg4 | straight | 2043 | 121 | 1.3570 | 1.1396 | 0.2174 [0.0015, 0.6218] | **SEP** | 0.9700 [0.2262, 2.5038] SEP |
| seg4 | curved | 1386 | 87 | 6.2269 | 5.7263 | 0.5006 [0.0836, 0.9849] | **SEP** | 0.9553 [0.4742, 1.4362] SEP |
| seg5 | all | 3392 | 139 | 4.6946 | 4.3488 | 0.3458 [0.1019, 0.6151] | **SEP** | 0.8804 [0.2164, 1.9076] SEP |
| seg5 | straight | 2020 | 121 | 1.4102 | 1.3111 | 0.0991 [-0.0424, 0.3198] |  | 0.9737 [0.1177, 2.7351] SEP |
| seg5 | curved | 1372 | 88 | 9.5303 | 8.8212 | 0.7091 [0.2110, 1.2606] | **SEP** | 0.7430 [0.1153, 1.3508] SEP |
| seg6 | all | 3354 | 139 | 6.3528 | 6.0130 | 0.3398 [0.0689, 0.6388] | **SEP** | 0.5037 [-0.0831, 1.3207] |
| seg6 | straight | 2006 | 121 | 1.6402 | 1.7276 | -0.0874 [-0.3060, 0.0751] |  | 0.7044 [-0.0084, 2.0729] |
| seg6 | curved | 1346 | 88 | 13.3484 | 12.3837 | 0.9647 [0.3682, 1.6342] | **SEP** | 0.2176 [-0.5266, 0.9362] |

### S6 -- EMA8-30k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 373 | 40 | 5.2520 | 5.1011 | 0.1509 [-0.1023, 0.4422] |  |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 1.1870 | 0.7632 | 0.4238 [0.1289, 0.8264] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.5391 | 0.5523 | -0.0132 [-0.0448, 0.0217] |  |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 99 | 22 | 6.3274 | 5.4867 | 0.8407 [0.2181, 1.8302] | **SEP** |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.9782 | 2.5310 | 0.4472 [0.0272, 0.9790] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.5919 | 2.1952 | 0.3967 [0.0337, 0.7787] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 65 | 20 | 10.2804 | 11.1818 | -0.9014 [-1.6182, -0.1873] | **SEP** |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 5.6225 | 5.8272 | -0.2047 [-0.5598, 0.1804] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 4.6041 | 4.3281 | 0.2760 [-0.0083, 0.6145] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 72 | 16 | 15.2590 | 14.2798 | 0.9792 [-1.5315, 4.5110] |  |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 9.7709 | 10.7541 | -0.9832 [-2.6574, 0.3225] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 5.7692 | 6.3909 | -0.6217 [-1.0595, 0.4950] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 17 | 12 | 11.2732 | 10.9951 | 0.2782 [0.0731, 0.5069] | **SEP** |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA8-30k s0-s1 [CI] | R7-30k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0415 [-0.0963, 0.0064] | 0.0185 [-0.0369, 0.0709] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0334 [-0.0874, 0.0085] | -0.0084 [-0.0264, 0.0108] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | -0.0054 [-0.0859, 0.0923] | 0.0693 [-0.0460, 0.2432] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.0024 [-0.1543, 0.1785] | 0.0928 [-0.0314, 0.2301] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | -0.2409 [-0.7843, 0.1795] | 0.0475 [-0.4823, 0.4413] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.0881 [-0.2105, 0.4451] | -0.9093 [-2.3766, 0.3252] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | 0.1343 [-0.1363, 0.6115] | 0.2784 [-0.0018, 0.6372] | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.1596 [-0.4473, 0.0471] | -0.0736 [-0.4282, 0.2864] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | -0.1151 [-0.2428, 0.0015] | 0.0440 [-0.0405, 0.1290] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0004 [-0.0391, 0.0356] | 0.0092 [-0.0307, 0.0556] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | 0.0088 [-0.0282, 0.0479] | -0.0311 [-0.0639, -0.0040] SEP | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | 0.0263 [-0.0054, 0.0648] | 0.0189 [-0.0164, 0.0666] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0455 [-0.1018, -0.0023] SEP | 0.0236 [-0.0124, 0.0651] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.0884 [-0.1817, 0.3811] | 0.1426 [-0.0193, 0.3162] | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | -0.0761 [-0.2079, 0.0323] | -0.0347 [-0.1898, 0.1397] | 0.0235 [-0.1175, 0.1775] |

### S6 -- EMA8-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1630 | 0.1295 / 0.1935 | False / True / False | 0.0415 / 0.0128 | 3.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.2336 | 0.2230 / 0.2373 | True / True / True | 0.0334 / 0.0216 | 7.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.4934 | 0.4543 / 0.5310 | True / True / True | 0.0054 / 0.0713 | 6.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.2225 | -0.2467 / -0.1984 | False / False / False | 0.0024 / 0.0611 | 3.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | -0.5108 | -0.7059 / -0.2978 | False / False / False | 0.2409 / 0.2004 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.2782 | 0.3491 / 0.2074 | True / False / True | 0.0881 / 1.9278 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0206 | -0.0374 / 0.0299 | False / False / False | 0.1343 / 0.0328 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.3601 | 0.1799 / 0.5403 | False / False / False | 0.1596 / 0.2009 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.3242 | 0.2788 / 0.3696 | False / False / False | 0.1151 / 0.0243 | 2.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0171 | 0.0303 / 0.0039 | False / False / False | 0.0004 / 0.0268 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.1124 | 0.1141 / 0.1108 | True / True / True | 0.0088 / 0.0055 | 12.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0357 | -0.0230 / -0.0485 | False / False / False | 0.0263 / 0.0008 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1034 | 0.0767 / 0.1301 | False / True / False | 0.0455 / 0.0080 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.3667 | -0.3480 / -0.4133 | False / False / False | 0.0884 / 0.0140 | 4.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.4819 | 0.4158 / 0.5476 | False / False / False | 0.0761 / 0.0235 | 6.3 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- EMA8-30k - R7-30k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0141 | -0.0432 / 0.0160 | False / False / False | 0.0415 / 0.0185 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.2019 | 0.1916 / 0.2122 | True / True / True | 0.0334 / 0.0084 | 6.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.4595 | 0.4221 / 0.4969 | True / True / True | 0.0054 / 0.0693 | 6.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.5722 | -0.6161 / -0.5282 | True / True / True | 0.0024 / 0.0928 | 6.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -1.7502 | -1.8640 / -1.6204 | True / True / True | 0.2409 / 0.0475 | 7.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.1543 | 0.1387 / 0.1698 | False / False / False | 0.0881 / 0.9093 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0076 | -0.0704 / 0.0720 | False / False / False | 0.1343 / 0.2784 | 0.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0694 | 0.0265 / 0.1124 | False / False / False | 0.1596 / 0.0736 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0265 | -0.0531 / 0.1060 | False / False / False | 0.1151 / 0.0440 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.1201 | -0.1249 / -0.1153 | False / False / False | 0.0004 / 0.0092 | 13.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0303 | 0.0502 / 0.0103 | False / False / False | 0.0088 / 0.0311 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0455 | -0.0418 / -0.0492 | False / False / False | 0.0263 / 0.0189 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.0295 | -0.0637 / 0.0046 | False / False / False | 0.0455 / 0.0236 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.7366 | -0.7382 / -0.7233 | False / False / False | 0.0884 / 0.1426 | 5.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2650 | 0.2390 / 0.2908 | False / False / False | 0.0761 / 0.0347 | 3.5 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- R7-30k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1908 | 0.1958 / 0.1870 | True / True / True | 0.0185 / 0.0128 | 10.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0362 | 0.0448 / 0.0298 | False / False / False | 0.0084 / 0.0216 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0331 | 0.0316 / 0.0346 | False / False / False | 0.0693 / 0.0713 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.3563 | 0.3818 / 0.3308 | True / True / True | 0.0928 / 0.0611 | 3.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 1.2909 | 1.2144 / 1.3673 | False / True / True | 0.0475 / 0.2004 | 6.4 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | 0.8938 | 1.5319 / 0.2558 | True / False / True | 0.9093 / 1.9278 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.2654 | 0.4228 / 0.1281 | False / False / False | 0.2784 / 0.0328 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.2907 | 0.1534 / 0.4279 | False / False / False | 0.0736 / 0.2009 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2977 | 0.3318 / 0.2636 | False / False / False | 0.0440 / 0.0243 | 6.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.1373 | 0.1553 / 0.1192 | False / False / False | 0.0092 / 0.0268 | 5.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0821 | 0.0638 / 0.1005 | False / False / False | 0.0311 / 0.0055 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0098 | 0.0188 / 0.0007 | False / False / False | 0.0189 / 0.0008 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1327 | 0.1404 / 0.1251 | True / True / True | 0.0236 / 0.0080 | 5.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.4077 | 0.4797 / 0.3520 | False / False / False | 0.1426 / 0.0140 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2518 | 0.2282 / 0.2738 | False / False / False | 0.0347 / 0.0235 | 7.3 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1409): EMA8-30k bias +0.090, refcv7-50400 bias +0.164, R7-30k bias +0.107; EMA8-30k MAE 1.179, refcv7-50400 MAE 0.965, R7-30k MAE 1.031
* seg2 (n 1407): EMA8-30k bias -0.089, refcv7-50400 bias -0.239, R7-30k bias -0.393; EMA8-30k MAE 2.461, refcv7-50400 MAE 2.377, R7-30k MAE 2.368
* seg3 (n 1393): EMA8-30k bias -0.939, refcv7-50400 bias -1.113, R7-30k bias -1.459; EMA8-30k MAE 3.855, refcv7-50400 MAE 3.672, R7-30k MAE 3.927
* seg4 (n 1371): EMA8-30k bias -2.378, refcv7-50400 bias -2.509, R7-30k bias -3.321; EMA8-30k MAE 5.887, refcv7-50400 MAE 5.698, R7-30k MAE 6.191
* seg5 (n 1349): EMA8-30k bias -3.916, refcv7-50400 bias -4.458, R7-30k bias -5.215; EMA8-30k MAE 8.797, refcv7-50400 MAE 8.765, R7-30k MAE 9.464
* seg6 (n 1321): EMA8-30k bias -5.745, refcv7-50400 bias -6.878, R7-30k bias -7.237; EMA8-30k MAE 12.326, refcv7-50400 MAE 12.343, R7-30k MAE 13.281
**sharp (>= 0.15)**

* seg1 (n 304): EMA8-30k bias -0.695, refcv7-50400 bias +0.169, R7-30k bias -0.026; EMA8-30k MAE 1.895, refcv7-50400 MAE 1.439, R7-30k MAE 1.542
* seg2 (n 299): EMA8-30k bias -1.015, refcv7-50400 bias -0.559, R7-30k bias -1.560; EMA8-30k MAE 4.017, refcv7-50400 MAE 3.836, R7-30k MAE 4.113
* seg3 (n 296): EMA8-30k bias -3.081, refcv7-50400 bias -2.491, R7-30k bias -4.808; EMA8-30k MAE 7.129, refcv7-50400 MAE 6.736, R7-30k MAE 7.708
* seg4 (n 297): EMA8-30k bias -6.677, refcv7-50400 bias -6.436, R7-30k bias -9.722; EMA8-30k MAE 11.556, refcv7-50400 MAE 11.627, R7-30k MAE 13.599
* seg5 (n 292): EMA8-30k bias -10.483, refcv7-50400 bias -11.592, R7-30k bias -15.042; EMA8-30k MAE 17.224, refcv7-50400 MAE 18.811, R7-30k MAE 20.901
* seg6 (n 287): EMA8-30k bias -15.112, refcv7-50400 bias -17.232, R7-30k bias -19.769; EMA8-30k MAE 23.266, refcv7-50400 MAE 25.034, R7-30k MAE 27.389
**curved & v0 5-10**

* seg1 (n 547): EMA8-30k bias +0.006, refcv7-50400 bias +0.049, R7-30k bias -0.010; EMA8-30k MAE 0.893, refcv7-50400 MAE 0.896, R7-30k MAE 0.993
* seg2 (n 547): EMA8-30k bias -0.486, refcv7-50400 bias -0.516, R7-30k bias -0.769; EMA8-30k MAE 2.440, refcv7-50400 MAE 2.231, R7-30k MAE 2.434
* seg3 (n 545): EMA8-30k bias -1.537, refcv7-50400 bias -1.649, R7-30k bias -2.275; EMA8-30k MAE 4.034, refcv7-50400 MAE 3.822, R7-30k MAE 4.235
* seg4 (n 533): EMA8-30k bias -3.049, refcv7-50400 bias -3.577, R7-30k bias -4.359; EMA8-30k MAE 6.072, refcv7-50400 MAE 6.233, R7-30k MAE 6.904
* seg5 (n 514): EMA8-30k bias -4.921, refcv7-50400 bias -5.891, R7-30k bias -6.440; EMA8-30k MAE 8.871, refcv7-50400 MAE 9.399, R7-30k MAE 10.231
* seg6 (n 493): EMA8-30k bias -6.824, refcv7-50400 bias -8.469, R7-30k bias -8.540; EMA8-30k MAE 12.409, refcv7-50400 MAE 12.812, R7-30k MAE 13.840
**sharp & v0 5-10**

* seg1 (n 139): EMA8-30k bias -0.064, refcv7-50400 bias +0.062, R7-30k bias +0.021; EMA8-30k MAE 1.183, refcv7-50400 MAE 1.251, R7-30k MAE 1.454
* seg2 (n 139): EMA8-30k bias -1.260, refcv7-50400 bias -1.267, R7-30k bias -1.794; EMA8-30k MAE 3.846, refcv7-50400 MAE 3.564, R7-30k MAE 3.957
* seg3 (n 139): EMA8-30k bias -3.881, refcv7-50400 bias -4.133, R7-30k bias -5.957; EMA8-30k MAE 6.999, refcv7-50400 MAE 7.029, R7-30k MAE 8.355
* seg4 (n 138): EMA8-30k bias -7.786, refcv7-50400 bias -9.099, R7-30k bias -11.691; EMA8-30k MAE 11.049, refcv7-50400 MAE 12.470, R7-30k MAE 14.579
* seg5 (n 136): EMA8-30k bias -11.316, refcv7-50400 bias -14.484, R7-30k bias -16.561; EMA8-30k MAE 16.627, refcv7-50400 MAE 19.898, R7-30k MAE 22.220
* seg6 (n 132): EMA8-30k bias -14.617, refcv7-50400 bias -18.938, R7-30k bias -19.739; EMA8-30k MAE 22.970, refcv7-50400 MAE 26.189, R7-30k MAE 28.288

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1321 / 1321 | 26.0 | EMA8-30k -0.2011 [-0.2688, -0.1345]; refcv7-50400 -0.2477 [-0.3212, -0.1800]; R7-30k -0.2758 [-0.3564, -0.2002] | EMA8-30k +0.0467 [+0.0070, +0.0909] SEP; R7-30k -0.0281 [-0.0616, +0.0055] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1417 / 6841 | 16.4 | EMA8-30k -0.1342 [-0.2140, -0.0590]; refcv7-50400 -0.1784 [-0.2447, -0.1197]; R7-30k -0.2008 [-0.2826, -0.1263] | EMA8-30k +0.0443 [-0.0113, +0.1192]; R7-30k -0.0224 [-0.0690, +0.0370] |
| sharp (>= 0.15) | last segment | 287 / 287 | 63.0 | EMA8-30k -0.2272 [-0.3103, -0.1534]; refcv7-50400 -0.2638 [-0.3533, -0.1873]; R7-30k -0.3059 [-0.4051, -0.2199] | EMA8-30k +0.0366 [-0.0140, +0.0899]; R7-30k -0.0421 [-0.0858, +0.0003] |
| sharp (>= 0.15) | segments 2..K | 306 / 1471 | 39.8 | EMA8-30k -0.1871 [-0.2739, -0.1174]; refcv7-50400 -0.1934 [-0.2797, -0.1220]; R7-30k -0.2556 [-0.3518, -0.1770] | EMA8-30k +0.0063 [-0.0474, +0.0577]; R7-30k -0.0622 [-0.1070, -0.0214] SEP |
| curved & v0 5-10 | last segment | 493 / 493 | 28.2 | EMA8-30k -0.2092 [-0.2795, -0.1354]; refcv7-50400 -0.2748 [-0.4006, -0.1729]; R7-30k -0.2939 [-0.4024, -0.2014] | EMA8-30k +0.0656 [-0.0078, +0.1573]; R7-30k -0.0191 [-0.0802, +0.0397] |
| curved & v0 5-10 | segments 2..K | 547 / 2632 | 17.8 | EMA8-30k -0.1699 [-0.2296, -0.1133]; refcv7-50400 -0.2066 [-0.3224, -0.1203]; R7-30k -0.2429 [-0.3462, -0.1604] | EMA8-30k +0.0367 [-0.0315, +0.1244]; R7-30k -0.0363 [-0.0925, +0.0157] |
| sharp & v0 5-10 | last segment | 132 / 132 | 66.9 | EMA8-30k -0.2176 [-0.2929, -0.1447]; refcv7-50400 -0.2738 [-0.4111, -0.1615]; R7-30k -0.3020 [-0.4236, -0.2028] | EMA8-30k +0.0562 [-0.0208, +0.1573]; R7-30k -0.0282 [-0.1034, +0.0433] |
| sharp & v0 5-10 | segments 2..K | 139 / 684 | 43.7 | EMA8-30k -0.1809 [-0.2463, -0.1203]; refcv7-50400 -0.2120 [-0.3465, -0.1138]; R7-30k -0.2602 [-0.3816, -0.1663] | EMA8-30k +0.0311 [-0.0440, +0.1349]; R7-30k -0.0482 [-0.1188, +0.0180] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA8-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3542 | 0.1630 | 0.01702 | 0.02688 | 0.527 / 0.473 / 0.000 | 1008.8 / 577.2 |
| turn|straight <0.02 | 2107 | 0.2336 | 0.03323 | 0.05424 | 0.562 / 0.438 / 0.000 | 384.9 / 492.3 |
| turn|gentle 0.02-0.06 | 676 | 0.4934 | 0.00929 | 0.04193 | 0.515 / 0.485 / 0.000 | 192.8 / 333.5 |
| turn|turn 0.06-0.15 | 436 | -0.2225 | -0.08087 | -0.14894 | 0.466 / 0.534 / 0.000 | 48.1 / -97.0 |
| turn|sharp >=0.15 | 306 | -0.5108 | -0.56090 | -0.60360 | 0.389 / 0.611 / 0.000 | 179.6 / -156.3 |
| turn|undef (GT crawling) | 17 | 0.2782 | 0.19579 | 0.22882 | 0.647 / 0.353 / 0.000 | 1.5 / 4.7 |
| speed|v0 <2 | 204 | 0.0206 | 0.02808 | 0.04635 | 0.549 / 0.451 / 0.000 | 60.4 / 4.2 |
| speed|v0 2-5 | 422 | 0.3601 | 0.04662 | 0.09540 | 0.536 / 0.464 / 0.000 | 151.2 / 152.0 |
| speed|v0 5-10 | 1040 | 0.3242 | -0.00785 | -0.01549 | 0.496 / 0.504 / 0.000 | 438.3 / 337.1 |
| speed|v0 10-15 | 952 | 0.0171 | 0.05398 | 0.07100 | 0.591 / 0.409 / 0.000 | 76.5 / 16.3 |
| speed|v0 15-25 | 679 | 0.1124 | 0.00340 | 0.00477 | 0.517 / 0.483 / 0.000 | 47.7 / 76.3 |
| speed|v0 >=25 | 245 | -0.0357 | -0.03072 | -0.03875 | 0.400 / 0.600 / 0.000 | 5.1 / -8.8 |
| nav|nav=follow | 2256 | 0.1034 | 0.00260 | -0.00066 | 0.508 / 0.492 / 0.000 | 348.9 / 233.3 |
| nav|nav=left | 325 | -0.3667 | 0.06041 | 0.04268 | 0.563 / 0.437 / 0.000 | 111.3 / -119.2 |
| nav|nav=right | 961 | 0.4819 | 0.09665 | 0.06944 | 0.560 / 0.440 / 0.000 | 380.8 / 463.1 |
**EMA8-30k - R7-30k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3527 | -0.0141 | -0.02325 | -0.04471 | 0.473 / 0.527 / 0.000 | 1012.1 / -49.6 |
| turn|straight <0.02 | 2095 | 0.2019 | 0.01151 | 0.02458 | 0.518 / 0.482 / 0.000 | 377.1 / 423.0 |
| turn|gentle 0.02-0.06 | 675 | 0.4595 | -0.05946 | -0.05824 | 0.456 / 0.544 / 0.000 | 216.8 / 310.2 |
| turn|turn 0.06-0.15 | 436 | -0.5722 | -0.23679 | -0.43157 | 0.362 / 0.638 / 0.000 | 49.1 / -249.5 |
| turn|sharp >=0.15 | 306 | -1.7502 | -0.92306 | -1.31992 | 0.350 / 0.650 / 0.000 | 168.9 / -535.6 |
| turn|undef (GT crawling) | 15 | 0.1543 | 0.16548 | 0.15125 | 0.600 / 0.400 / 0.000 | 1.1 / 2.3 |
| speed|v0 <2 | 189 | -0.0076 | 0.01036 | -0.06868 | 0.503 / 0.497 / 0.000 | 51.6 / -1.4 |
| speed|v0 2-5 | 422 | 0.0694 | 0.00618 | -0.00819 | 0.505 / 0.495 / 0.000 | 170.4 / 29.3 |
| speed|v0 5-10 | 1040 | 0.0265 | -0.05196 | -0.11556 | 0.461 / 0.539 / 0.000 | 407.8 / 27.5 |
| speed|v0 10-15 | 952 | -0.1201 | -0.00934 | -0.01293 | 0.494 / 0.506 / 0.000 | 72.9 / -114.3 |
| speed|v0 15-25 | 679 | 0.0303 | -0.02290 | -0.02663 | 0.454 / 0.546 / 0.000 | 43.2 / 20.5 |
| speed|v0 >=25 | 245 | -0.0455 | -0.03941 | -0.04059 | 0.420 / 0.580 / 0.000 | 4.3 / -11.1 |
| nav|nav=follow | 2243 | -0.0295 | -0.03239 | -0.04220 | 0.455 / 0.545 / 0.000 | 319.0 / -66.3 |
| nav|nav=left | 323 | -0.7366 | -0.02537 | -0.17860 | 0.483 / 0.517 / 0.000 | 112.6 / -237.9 |
| nav|nav=right | 961 | 0.2650 | 0.02617 | -0.01081 | 0.511 / 0.489 / 0.000 | 397.0 / 254.6 |
**R7-30k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3549 | 0.1908 | 0.04782 | 0.06696 | 0.581 / 0.419 / 0.000 | 624.9 / 677.1 |
| turn|straight <0.02 | 2100 | 0.0362 | 0.03351 | 0.03370 | 0.575 / 0.425 / 0.000 | 66.0 / 76.1 |
| turn|gentle 0.02-0.06 | 675 | 0.0331 | 0.07457 | 0.09536 | 0.587 / 0.413 / 0.000 | 34.5 / 22.3 |
| turn|turn 0.06-0.15 | 441 | 0.3563 | 0.15301 | 0.20016 | 0.587 / 0.413 / 0.000 | 55.0 / 157.1 |
| turn|sharp >=0.15 | 312 | 1.2909 | 0.21509 | 0.56419 | 0.593 / 0.407 / 0.000 | 134.1 / 402.8 |
| turn|undef (GT crawling) | 21 | 0.8938 | 0.07032 | 0.21360 | 0.714 / 0.286 / 0.000 | 14.6 / 18.8 |
| speed|v0 <2 | 211 | 0.2654 | 0.08221 | 0.16092 | 0.578 / 0.422 / 0.000 | 29.5 / 56.0 |
| speed|v0 2-5 | 422 | 0.2907 | 0.08953 | 0.11149 | 0.585 / 0.415 / 0.000 | 118.8 / 122.7 |
| speed|v0 5-10 | 1040 | 0.2977 | 0.07701 | 0.09054 | 0.565 / 0.435 / 0.000 | 289.3 / 309.6 |
| speed|v0 10-15 | 952 | 0.1373 | 0.06305 | 0.07700 | 0.618 / 0.382 / 0.000 | 80.6 / 130.7 |
| speed|v0 15-25 | 679 | 0.0821 | 0.02171 | 0.02008 | 0.564 / 0.436 / 0.000 | 39.1 / 55.8 |
| speed|v0 >=25 | 245 | 0.0098 | 0.01260 | 0.00239 | 0.547 / 0.453 / 0.000 | 4.4 / 2.4 |
| nav|nav=follow | 2246 | 0.1327 | 0.03370 | 0.03982 | 0.569 / 0.431 / 0.000 | 286.3 / 298.1 |
| nav|nav=left | 327 | 0.4077 | 0.09678 | 0.19025 | 0.602 / 0.398 / 0.000 | 50.4 / 133.3 |
| nav|nav=right | 976 | 0.2518 | 0.08841 | 0.10862 | 0.602 / 0.398 / 0.000 | 247.9 / 245.7 |

### S6 -- the episodes carrying the summed delta (panel file index only)

**EMA8-30k - refcv7-50400** (sum 577.2 over 139 episodes)

* ep059: n 27, share 0.263, mean delta 5.6167, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep027: n 26, share 0.257, mean delta 5.6958, sharp-window frac 0.00, curved frac 0.00, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep078: n 26, share 0.174, mean delta 3.8569, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep013: n 27, share 0.140, mean delta 2.9945, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep048: n 27, share 0.111, mean delta 2.3643, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep005: n 27, share 0.097, mean delta 2.0651, sharp-window frac 0.00, curved frac 0.33, mean v0 6.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**EMA8-30k - R7-30k** (sum -49.6 over 139 episodes)

* ep027: n 26, share -2.958, mean delta 5.6391, sharp-window frac 0.00, curved frac 0.00, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep059: n 27, share -2.937, mean delta 5.3908, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep078: n 26, share -2.931, mean delta 5.5874, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep048: n 27, share -1.234, mean delta 2.2655, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep005: n 27, share -1.059, mean delta 1.9435, sharp-window frac 0.00, curved frac 0.33, mean v0 6.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep013: n 27, share -1.028, mean delta 1.8871, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**R7-30k - refcv7-50400** (sum 677.1 over 139 episodes)

* ep111: n 26, share 0.189, mean delta 4.9261, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.183, mean delta 4.7761, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.094, mean delta 2.3661, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep016: n 22, share 0.084, mean delta 2.5708, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.082, mean delta 2.1458, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep061: n 26, share 0.072, mean delta 1.8755, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA8-30k vs R7-30k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1320 / -0.0566 [-0.0940, -0.0212] | 49 / -0.4141 [-1.0141, 0.0255] |
| only new agrees | 303 / -1.4506 [-2.2672, -0.7955] | 32 / -8.4335 [-14.4489, -4.2400] |
| only earlier agrees | 659 / 0.9050 [0.5266, 1.3888] | 25 / 7.7476 [2.2325, 15.4958] |
| neither agrees | 1246 / -0.1883 [-0.6076, 0.2344] | 201 / -2.3671 [-4.2115, -0.6376] |
**EMA8-30k vs R7-30k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1329 / -0.0628 [-0.0996, -0.0258] | 52 / -0.5830 [-1.1196, -0.1654] |
| only new agrees | 276 / -1.7021 [-2.6811, -0.8944] | 32 / -10.8809 [-17.7284, -5.3373] |
| only earlier agrees | 658 / 0.9673 [0.5691, 1.4811] | 29 / 8.6448 [2.6661, 16.1786] |
| neither agrees | 1265 / -0.0211 [-0.4998, 0.4885] | 193 / -1.9069 [-3.7496, -0.3858] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA8-30k vs R7-30k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3493 | 0.1591 [-0.1674, 0.5009] | 0.2684 [-0.0013, 0.6224] | -0.0876 [-0.3108, 0.1179] |
| turn|straight <0.02 | 2082 | -0.1186 [-0.3566, 0.0489] | -0.0116 [-0.0377, 0.0146] | -0.1216 [-0.3325, 0.0406] |
| turn|gentle 0.02-0.06 | 675 | -0.6885 [-1.6292, -0.0371] | -0.3040 [-0.7972, 0.0186] | -0.3831 [-0.8389, -0.0303] SEP |
| turn|turn 0.06-0.15 | 426 | 0.1710 [-0.2072, 0.5898] | 0.1798 [-0.2220, 0.6523] | 0.0840 [-0.1731, 0.3295] |
| turn|sharp >=0.15 | 301 | 3.9963 [1.6121, 6.8094] | 3.5454 [1.2035, 6.9651] | 0.5803 [-1.0346, 2.4241] |
| turn|undef (GT crawling) | 9 | -0.3601 [-0.9884, 0.0621] | 0.0560 [-0.2548, 0.3435] | -0.5171 [-1.3105, 0.1067] |
| speed|v0 <2 | 155 | 0.3584 [-0.1114, 1.0589] | 0.5921 [-0.0256, 1.2883] | 0.1854 [-0.2859, 0.9840] |
| speed|v0 2-5 | 422 | 0.6932 [-0.9153, 2.3548] | 1.1090 [-0.5204, 3.3488] | -0.4158 [-1.3303, 0.3817] |
| speed|v0 5-10 | 1040 | 0.3462 [-0.3446, 1.0552] | 0.3891 [0.0293, 0.8012] | -0.0429 [-0.5985, 0.5055] |
| speed|v0 10-15 | 952 | -0.1701 [-0.3593, -0.0344] | -0.0485 [-0.1090, 0.0053] | -0.1216 [-0.2670, -0.0093] SEP |
| speed|v0 15-25 | 679 | 0.0102 [-0.0579, 0.1009] | 0.0007 [-0.0284, 0.0271] | 0.0096 [-0.0587, 0.0952] |
| speed|v0 >=25 | 245 | -0.0021 [-0.0405, 0.0274] | 0.0198 [-0.0076, 0.0472] | -0.0219 [-0.0689, 0.0205] |
**EMA8-30k vs R7-30k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* EMA8-30k vs R7-30k|seed0: anchor flip rate 0.424; heading delta on flipped windows 0.0439 [-0.4663, 0.5424] (n 1507) vs unflipped -0.1082 [-0.1561, -0.0676]; goal_dist_sel proxy new-earlier -0.2059 [-0.3712, -0.0458]; corr(window dHeading, window dGoalDist) -0.007
* EMA8-30k vs R7-30k|seed1: anchor flip rate 0.428; heading delta on flipped windows 0.1823 [-0.3830, 0.7413] (n 1517) vs unflipped -0.1094 [-0.1507, -0.0700]; goal_dist_sel proxy new-earlier -0.2134 [-0.3797, -0.0520]; corr(window dHeading, window dGoalDist) -0.009
