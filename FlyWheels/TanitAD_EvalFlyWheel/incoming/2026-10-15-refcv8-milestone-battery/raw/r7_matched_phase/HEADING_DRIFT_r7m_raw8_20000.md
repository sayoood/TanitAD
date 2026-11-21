# refcv8 heading-drift localisation: RAW8-20k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-18T06:05:12; trees: `RAW8-20k` = C:/Users/Admin/refcv8_eval_archive/battery/step20000_unc; `R7-20k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step20000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- RAW8-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4506 | 139 | 1.4953 | 1.4044 | 0.0909 [0.0464, 0.1383] | **SEP** |  |
| turn: straight <0.02 | 2901 | 132 | 1.2185 | 1.1586 | 0.0599 [0.0275, 0.0999] | **SEP** | 0.644 / 0.424 / -0.0871 [-0.2973, 0.0531] |
| turn: gentle 0.02-0.06 | 719 | 97 | 1.0725 | 1.0091 | 0.0634 [-0.0202, 0.1627] |  | 0.160 / 0.111 / -0.0328 [-0.1633, 0.1504] |
| turn: turn 0.06-0.15 | 503 | 72 | 2.0364 | 1.9043 | 0.1321 [-0.0092, 0.2802] |  | 0.112 / 0.162 / 0.0464 [-0.1642, 0.2833] |
| turn: sharp >=0.15 | 366 | 42 | 3.3246 | 2.9907 | 0.3339 [-0.0212, 0.7360] |  | 0.081 / 0.298 / 0.2644 [-0.2455, 1.0095] |
| turn: undef (GT crawling) (low n) | 17 | 14 | 11.2171 | 11.1183 | 0.0988 [-0.1057, 0.4003] |  | 0.004 / 0.004 / 0.0079 [-0.2923, 0.6366] |
| speed: v0 <2 | 217 | 37 | 13.4794 | 13.3810 | 0.0984 [-0.0553, 0.2884] |  | 0.048 / 0.052 / 0.0079 [-0.2089, 0.3822] |
| speed: v0 2-5 | 516 | 58 | 1.9947 | 1.8694 | 0.1253 [-0.1060, 0.4134] |  | 0.115 / 0.158 / 0.0388 [-0.2890, 0.5829] |
| speed: v0 5-10 | 1390 | 84 | 1.1767 | 1.0092 | 0.1675 [0.0679, 0.2729] | **SEP** | 0.308 / 0.568 / 0.1108 [-0.0614, 0.2900] |
| speed: v0 10-15 | 1192 | 64 | 0.6324 | 0.5744 | 0.0580 [0.0118, 0.1067] | **SEP** | 0.265 / 0.169 / -0.0447 [-0.1643, 0.0610] |
| speed: v0 15-25 | 856 | 35 | 0.3903 | 0.3734 | 0.0169 [-0.0215, 0.0573] |  | 0.190 / 0.035 / -0.0914 [-0.2029, -0.0102] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1791 | 0.1576 | 0.0215 [-0.0064, 0.0549] |  | 0.074 / 0.018 / -0.0750 [-0.1701, 0.0137] |
| nav: nav=follow | 2880 | 88 | 0.6855 | 0.5981 | 0.0875 [0.0479, 0.1318] | **SEP** | 0.639 / 0.615 / -0.0096 [-0.1966, 0.1484] |
| nav: nav=left (low n) | 404 | 13 | 6.9248 | 6.8947 | 0.0302 [-0.0437, 0.0925] |  | 0.090 / 0.030 / -0.0667 [-0.1985, 0.0484] |
| nav: nav=right | 1222 | 38 | 1.6088 | 1.4896 | 0.1192 [-0.0020, 0.2572] |  | 0.271 / 0.355 / 0.0387 [-0.1533, 0.2790] |

episode concentration: 0.676 of episodes have mean delta > 0; top-5 episodes carry 0.446 of the summed delta; delta excluding the top 5 = 0.0523 (overall 0.0909).

### S2 -- RAW8-20k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4497 | 139 | 1.4583 | 1.4410 | 0.0173 [-0.0387, 0.0729] |  |  |
| turn: straight <0.02 | 2897 | 132 | 1.2199 | 1.1939 | 0.0261 [-0.0158, 0.0674] |  | 0.644 / 0.971 / 0.0246 [-0.1848, 0.2251] |
| turn: gentle 0.02-0.06 | 719 | 97 | 1.0725 | 0.9813 | 0.0911 [-0.0198, 0.2162] |  | 0.160 / 0.842 / 0.0879 [-0.0662, 0.2928] |
| turn: turn 0.06-0.15 | 503 | 72 | 2.0364 | 2.1784 | -0.1419 [-0.3751, 0.0645] |  | 0.112 / -0.917 / -0.1793 [-0.5485, 0.1462] |
| turn: sharp >=0.15 | 366 | 42 | 3.3246 | 3.3165 | 0.0080 [-0.3648, 0.3604] |  | 0.081 / 0.038 / -0.0101 [-0.5363, 0.6064] |
| turn: undef (GT crawling) (low n) | 12 | 11 | 0.9632 | 0.5282 | 0.4351 [-0.1946, 1.3176] |  | 0.003 / 0.067 / 0.4189 [-0.4560, 2.1467] |
| speed: v0 <2 | 208 | 37 | 13.1972 | 13.2089 | -0.0116 [-0.2531, 0.2526] |  | 0.046 / -0.031 / -0.0303 [-0.3856, 0.4852] |
| speed: v0 2-5 | 516 | 58 | 1.9947 | 2.0874 | -0.0927 [-0.3816, 0.1886] |  | 0.115 / -0.614 / -0.1242 [-0.5506, 0.3235] |
| speed: v0 5-10 | 1390 | 84 | 1.1767 | 1.1092 | 0.0676 [-0.0435, 0.1778] |  | 0.309 / 1.207 / 0.0727 [-0.1027, 0.2466] |
| speed: v0 10-15 | 1192 | 64 | 0.6324 | 0.6040 | 0.0284 [-0.0137, 0.0716] |  | 0.265 / 0.435 / 0.0151 [-0.0966, 0.1238] |
| speed: v0 15-25 | 856 | 35 | 0.3903 | 0.3964 | -0.0061 [-0.0490, 0.0320] |  | 0.190 / -0.067 / -0.0289 [-0.1433, 0.0793] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1791 | 0.1627 | 0.0164 [-0.0072, 0.0457] |  | 0.074 / 0.071 / -0.0009 [-0.0932, 0.0960] |
| nav: nav=follow | 2872 | 88 | 0.6870 | 0.6506 | 0.0364 [-0.0007, 0.0727] |  | 0.639 / 1.341 / 0.0527 [-0.1710, 0.2732] |
| nav: nav=left (low n) | 404 | 13 | 6.9248 | 6.9301 | -0.0052 [-0.2498, 0.2633] |  | 0.090 / -0.027 / -0.0248 [-0.4189, 0.4290] |
| nav: nav=right | 1221 | 38 | 1.4638 | 1.4839 | -0.0200 [-0.1698, 0.1489] |  | 0.272 / -0.314 / -0.0512 [-0.2975, 0.1995] |

episode concentration: 0.540 of episodes have mean delta > 0; top-5 episodes carry 2.143 of the summed delta; delta excluding the top 5 = -0.0205 (overall 0.0173).

### S2 -- R7-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4520 | 139 | 1.4855 | 1.4107 | 0.0748 [0.0298, 0.1207] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.1935 | 1.1596 | 0.0339 [0.0124, 0.0591] | **SEP** | 0.641 / 0.291 / -0.1139 [-0.3208, 0.0621] |
| turn: gentle 0.02-0.06 | 721 | 98 | 0.9847 | 1.0095 | -0.0248 [-0.1099, 0.0403] |  | 0.160 / -0.053 / -0.1185 [-0.2785, -0.0092] BONF-SEP |
| turn: turn 0.06-0.15 | 512 | 72 | 2.2085 | 1.9319 | 0.2766 [0.1421, 0.4417] | **SEP** | 0.113 / 0.419 / 0.2276 [0.0336, 0.4964] BONF-SEP |
| turn: sharp >=0.15 | 368 | 42 | 3.3254 | 3.0070 | 0.3185 [-0.0559, 0.7023] |  | 0.081 / 0.347 / 0.2653 [-0.3030, 0.9886] |
| turn: undef (GT crawling) (low n) | 21 | 17 | 9.0951 | 9.1504 | -0.0553 [-0.3755, 0.1854] |  | 0.005 / -0.003 / -0.1307 [-0.6283, 0.1948] |
| speed: v0 <2 | 231 | 38 | 12.9074 | 12.7780 | 0.1294 [-0.0116, 0.2725] |  | 0.051 / 0.088 / 0.0575 [-0.1563, 0.2727] |
| speed: v0 2-5 | 516 | 58 | 2.0874 | 1.8694 | 0.2180 [-0.0860, 0.5492] |  | 0.114 / 0.333 / 0.1616 [-0.3265, 0.7098] |
| speed: v0 5-10 | 1390 | 84 | 1.1092 | 1.0092 | 0.1000 [0.0416, 0.1655] | **SEP** | 0.308 / 0.411 / 0.0364 [-0.0810, 0.1531] |
| speed: v0 10-15 | 1192 | 64 | 0.6040 | 0.5744 | 0.0296 [-0.0108, 0.0696] |  | 0.264 / 0.104 / -0.0613 [-0.1738, 0.0382] |
| speed: v0 15-25 | 856 | 35 | 0.3964 | 0.3734 | 0.0230 [-0.0030, 0.0579] |  | 0.189 / 0.058 / -0.0639 [-0.1638, 0.0329] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1627 | 0.1576 | 0.0051 [-0.0069, 0.0182] |  | 0.074 / 0.005 / -0.0753 [-0.1564, -0.0009] BONF-SEP |
| nav: nav=follow | 2872 | 88 | 0.6506 | 0.5988 | 0.0518 [0.0196, 0.0903] | **SEP** | 0.635 / 0.441 / -0.0629 [-0.2319, 0.1204] |
| nav: nav=left (low n) | 412 | 13 | 7.2617 | 7.2314 | 0.0302 [-0.2494, 0.2404] |  | 0.091 / 0.037 / -0.0490 [-0.5216, 0.2698] |
| nav: nav=right | 1236 | 38 | 1.4999 | 1.3570 | 0.1429 [0.0189, 0.2619] | **SEP** | 0.273 / 0.523 / 0.0938 [-0.0821, 0.2815] |

episode concentration: 0.683 of episodes have mean delta > 0; top-5 episodes carry 0.461 of the summed delta; delta excluding the top 5 = 0.0418 (overall 0.0748).

### S2 -- RAW8-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4470 | 139 | 0.7651 | 0.6277 | 0.1374 [0.0084, 0.3860] | **SEP** | 0.3107 [-0.0676, 1.0552] |
| seg1 | straight | 2880 | 132 | 0.7655 | 0.5695 | 0.1960 [0.0027, 0.6105] | **SEP** | 0.5381 [-0.0326, 1.7875] |
| seg1 | curved | 1580 | 101 | 0.6533 | 0.6222 | 0.0311 [0.0035, 0.0611] | **SEP** | -0.1015 [-0.1575, -0.0513] SEP |
| seg2 | all | 4453 | 139 | 1.0483 | 0.8974 | 0.1509 [0.0473, 0.3277] | **SEP** | -0.0505 [-0.1785, 0.1453] |
| seg2 | straight | 2874 | 132 | 0.7700 | 0.5954 | 0.1747 [0.0314, 0.4641] | **SEP** | 0.0772 [-0.0758, 0.3732] |
| seg2 | curved | 1579 | 101 | 1.5549 | 1.4472 | 0.1077 [0.0301, 0.1937] | **SEP** | -0.2830 [-0.4482, -0.1443] SEP |
| seg3 | all | 4446 | 139 | 1.4268 | 1.1607 | 0.2660 [0.0643, 0.6135] | **SEP** | -0.0226 [-0.2637, 0.3537] |
| seg3 | straight | 2870 | 132 | 1.0328 | 0.7114 | 0.3214 [0.0401, 0.8919] | **SEP** | 0.2064 [-0.0900, 0.7890] |
| seg3 | curved | 1576 | 101 | 2.1442 | 1.9790 | 0.1652 [0.0395, 0.3043] | **SEP** | -0.4397 [-0.7026, -0.2227] SEP |
| seg4 | all | 4433 | 138 | 1.7697 | 1.4047 | 0.3650 [0.0999, 0.8518] | **SEP** | -0.0627 [-0.3786, 0.4445] |
| seg4 | straight | 2859 | 131 | 1.1267 | 0.7139 | 0.4128 [0.0494, 1.1233] | **SEP** | 0.2694 [-0.1072, 0.9793] |
| seg4 | curved | 1567 | 100 | 2.9460 | 2.6675 | 0.2785 [0.0783, 0.4946] | **SEP** | -0.6663 [-1.0645, -0.3203] SEP |

### S2 -- RAW8-20k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4470 | 139 | 0.7252 | 0.6282 | 0.0970 [-0.0348, 0.3462] |  | 0.0560 [-0.0874, 0.3138] |
| seg1 | straight | 2882 | 132 | 0.7650 | 0.5835 | 0.1816 [-0.0137, 0.5944] |  | 0.1481 [-0.0523, 0.5679] |
| seg1 | curved | 1580 | 101 | 0.6533 | 0.7109 | -0.0577 [-0.0950, -0.0162] | **SEP** | -0.1108 [-0.1812, -0.0469] SEP |
| seg2 | all | 4451 | 139 | 1.0477 | 1.0635 | -0.0158 [-0.0641, 0.0322] |  | -0.1842 [-0.2663, -0.1150] SEP |
| seg2 | straight | 2873 | 132 | 0.7703 | 0.7562 | 0.0141 [-0.0209, 0.0461] |  | -0.0747 [-0.1255, -0.0258] SEP |
| seg2 | curved | 1578 | 101 | 1.5526 | 1.6229 | -0.0703 [-0.1817, 0.0545] |  | -0.3835 [-0.5652, -0.2230] SEP |
| seg3 | all | 4441 | 139 | 1.4276 | 1.2457 | 0.1819 [-0.0257, 0.5297] |  | -0.0835 [-0.3376, 0.3016] |
| seg3 | straight | 2866 | 132 | 1.0335 | 0.7449 | 0.2885 [-0.0043, 0.8587] |  | 0.2238 [-0.0847, 0.8118] |
| seg3 | curved | 1575 | 101 | 2.1448 | 2.1570 | -0.0122 [-0.1742, 0.1561] |  | -0.6426 [-0.8968, -0.4108] SEP |
| seg4 | all | 4424 | 138 | 1.7717 | 1.6377 | 0.1340 [0.0188, 0.2673] | **SEP** | -0.2195 [-0.4036, -0.0286] SEP |
| seg4 | straight | 2854 | 131 | 1.1274 | 1.0047 | 0.1227 [0.0012, 0.2861] | **SEP** | 0.1069 [-0.0252, 0.2834] |
| seg4 | curved | 1566 | 100 | 2.9459 | 2.7937 | 0.1522 [-0.0952, 0.4072] |  | -0.8122 [-1.2008, -0.4665] SEP |

### S2 -- R7-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4483 | 139 | 0.6686 | 0.6280 | 0.0406 [0.0246, 0.0559] | **SEP** | 0.2541 [-0.0036, 0.7510] |
| seg1 | straight | 2880 | 132 | 0.5838 | 0.5695 | 0.0143 [0.0056, 0.0240] | **SEP** | 0.3898 [0.0024, 1.2288] SEP |
| seg1 | curved | 1588 | 101 | 0.7117 | 0.6229 | 0.0888 [0.0507, 0.1229] | **SEP** | 0.0098 [-0.0456, 0.0712] |
| seg2 | all | 4478 | 139 | 1.1048 | 0.9384 | 0.1665 [0.0530, 0.3494] | **SEP** | 0.2129 [0.0126, 0.5602] SEP |
| seg2 | straight | 2886 | 132 | 0.8169 | 0.6569 | 0.1600 [0.0202, 0.4464] | **SEP** | 0.2754 [-0.0039, 0.8447] |
| seg2 | curved | 1592 | 101 | 1.6268 | 1.4487 | 0.1782 [0.0727, 0.2766] | **SEP** | 0.0997 [-0.0402, 0.2522] |
| seg3 | all | 4465 | 139 | 1.2910 | 1.2053 | 0.0858 [0.0258, 0.1474] | **SEP** | 0.1440 [0.0106, 0.3285] SEP |
| seg3 | straight | 2876 | 132 | 0.8073 | 0.7741 | 0.0333 [0.0026, 0.0697] | **SEP** | 0.1089 [-0.0537, 0.4104] |
| seg3 | curved | 1589 | 101 | 2.1665 | 1.9857 | 0.1808 [0.0307, 0.3259] | **SEP** | 0.2075 [0.0256, 0.3916] SEP |
| seg4 | all | 4449 | 139 | 1.7720 | 1.4975 | 0.2745 [0.0066, 0.7239] | **SEP** | 0.2838 [-0.0685, 0.8901] |
| seg4 | straight | 2862 | 131 | 1.1923 | 0.8393 | 0.3530 [0.0164, 1.0423] | **SEP** | 0.3517 [-0.1307, 1.3262] |
| seg4 | curved | 1581 | 101 | 2.8250 | 2.6914 | 0.1336 [-0.0669, 0.3364] |  | 0.1607 [-0.1063, 0.4168] |

### S2 -- RAW8-20k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 442 | 47 | 5.9623 | 5.9278 | 0.0345 [-0.0413, 0.1116] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.4653 | 0.3674 | 0.0979 [0.0472, 0.1688] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.2052 | 0.1947 | 0.0105 [-0.0032, 0.0259] |  |  |
| gentle 0.02-0.06 | v0 <5 | 110 | 28 | 1.6389 | 1.7738 | -0.1349 [-0.5598, 0.2308] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.1109 | 0.9711 | 0.1398 [0.0149, 0.2938] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.5319 | 0.5590 | -0.0270 [-0.1122, 0.0409] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 73 | 25 | 5.5131 | 5.2557 | 0.2574 [-0.1629, 0.7373] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.5433 | 1.4275 | 0.1158 [-0.0563, 0.2870] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 1.0635 | 0.9719 | 0.0916 [-0.1777, 0.2504] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 91 | 16 | 5.9949 | 5.2793 | 0.7157 [-0.3498, 2.2562] |  |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.4758 | 2.2857 | 0.1901 [-0.0981, 0.4780] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.8755 | 1.3866 | 0.4889 [0.1047, 0.7045] | **SEP** |  |
| undef (GT crawling) | v0 <5 (low n) | 17 | 14 | 11.2171 | 11.1183 | 0.0988 [-0.1057, 0.4003] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW8-20k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0024 [-0.0193, 0.0183] | -0.0015 [-0.0157, 0.0150] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0148 [-0.0311, -0.0013] SEP | 0.0079 [-0.0056, 0.0266] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0148 [-0.0195, 0.0556] | -0.0271 [-0.0663, 0.0095] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | 0.0333 [-0.0137, 0.0846] | -0.0187 [-0.0706, 0.0340] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | 0.0154 [-0.1396, 0.2203] | -0.0092 [-0.1307, 0.1113] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.0362 [-0.1631, 0.0889] | 0.1339 [-0.0506, 0.3279] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | -0.0126 [-0.0969, 0.0884] | 0.0129 [-0.1193, 0.1887] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0280 [-0.0975, 0.0515] | 0.0189 [-0.0664, 0.1076] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0067 [-0.0441, 0.0314] | -0.0115 [-0.0432, 0.0284] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0109 [-0.0040, 0.0291] | 0.0005 [-0.0131, 0.0140] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | 0.0058 [-0.0041, 0.0186] | -0.0017 [-0.0070, 0.0041] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | -0.0064 [-0.0175, 0.0039] | -0.0075 [-0.0163, -0.0008] SEP | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | 0.0030 [-0.0085, 0.0154] | -0.0062 [-0.0176, 0.0054] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0601 [-0.0416, 0.2251] | -0.0472 [-0.1021, -0.0024] SEP | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0357 [-0.0756, 0.0020] | 0.0250 [-0.0162, 0.0755] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- RAW8-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0909 | 0.0840 / 0.0973 | True / True / True | 0.0024 / 0.0110 | 8.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0599 | 0.0515 / 0.0683 | True / True / True | 0.0148 / 0.0019 | 4.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0634 | 0.0776 / 0.0492 | False / False / False | 0.0148 / 0.0161 | 3.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.1321 | 0.1323 / 0.1271 | False / False / False | 0.0333 / 0.0331 | 4.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.3339 | 0.2850 / 0.3827 | False / True / False | 0.0154 / 0.1080 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.0988 | 0.1262 / 0.0643 | False / False / False | 0.0362 / 0.0237 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0984 | 0.0546 / 0.1304 | False / False / False | 0.0126 / 0.0573 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1253 | 0.0753 / 0.1753 | False / False / False | 0.0280 / 0.0719 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1675 | 0.1639 / 0.1712 | True / True / True | 0.0067 / 0.0006 | 25.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0580 | 0.0635 / 0.0525 | True / True / True | 0.0109 / 0.0001 | 5.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.0169 | 0.0209 / 0.0129 | False / False / False | 0.0058 / 0.0023 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0215 | 0.0186 / 0.0245 | False / False / False | 0.0064 / 0.0005 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0875 | 0.0908 / 0.0842 | True / True / True | 0.0030 / 0.0036 | 24.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0302 | 0.0286 / 0.0317 | False / False / False | 0.0601 / 0.0608 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1192 | 0.0862 / 0.1499 | False / True / False | 0.0357 / 0.0283 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- RAW8-20k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0173 | 0.0175 / 0.0165 | False / False / False | 0.0024 / 0.0015 | 7.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0261 | 0.0152 / 0.0369 | False / False / False | 0.0148 / 0.0079 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0911 | 0.1124 / 0.0699 | False / False / False | 0.0148 / 0.0271 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.1419 | -0.1155 / -0.1728 | False / False / False | 0.0333 / 0.0187 | 4.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.0080 | 0.0193 / -0.0032 | False / False / False | 0.0154 / 0.0092 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.4351 | 0.3995 / 0.4102 | False / False / False | 0.0362 / 0.1339 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0116 | -0.0108 / -0.0249 | False / False / False | 0.0126 / 0.0129 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.0927 | -0.1161 / -0.0692 | False / False / False | 0.0280 / 0.0189 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0676 | 0.0700 / 0.0651 | False / False / False | 0.0067 / 0.0115 | 5.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0284 | 0.0336 / 0.0232 | False / False / False | 0.0109 / 0.0005 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0061 | -0.0023 / -0.0098 | False / False / False | 0.0058 / 0.0017 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0164 | 0.0170 / 0.0159 | False / False / False | 0.0064 / 0.0075 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0364 | 0.0409 / 0.0317 | True / False / False | 0.0030 / 0.0062 | 5.8 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.0052 | 0.0531 / -0.0635 | False / False / False | 0.0601 / 0.0472 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.0200 | -0.0491 / 0.0072 | False / False / False | 0.0357 / 0.0250 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- R7-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0748 | 0.0681 / 0.0822 | True / True / True | 0.0015 / 0.0110 | 6.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0339 | 0.0364 / 0.0314 | True / True / True | 0.0079 / 0.0019 | 4.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.0248 | -0.0303 / -0.0194 | False / False / False | 0.0271 / 0.0161 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.2766 | 0.2510 / 0.3076 | True / True / True | 0.0187 / 0.0331 | 8.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 0.3185 | 0.2599 / 0.3771 | False / True / False | 0.0092 / 0.1080 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.0553 | 0.0054 / -0.0831 | False / False / False | 0.1339 / 0.0237 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1294 | 0.0965 / 0.1757 | False / True / False | 0.0129 / 0.0573 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.2180 | 0.1914 / 0.2445 | False / False / False | 0.0189 / 0.0719 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1000 | 0.0939 / 0.1061 | True / True / True | 0.0115 / 0.0006 | 8.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0296 | 0.0299 / 0.0293 | False / False / False | 0.0005 / 0.0001 | 54.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0230 | 0.0233 / 0.0227 | False / False / False | 0.0017 / 0.0023 | 10.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0051 | 0.0016 / 0.0086 | False / False / False | 0.0075 / 0.0005 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0518 | 0.0505 / 0.0531 | True / True / True | 0.0062 / 0.0036 | 8.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0302 | -0.0271 / 0.0947 | False / False / False | 0.0472 / 0.0608 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1429 | 0.1406 / 0.1455 | True / True / True | 0.0250 / 0.0283 | 5.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1580): RAW8-20k bias +0.084, refcv7-50400 bias +0.085, R7-20k bias +0.141; RAW8-20k MAE 0.653, refcv7-50400 MAE 0.622, R7-20k MAE 0.711
* seg2 (n 1578): RAW8-20k bias -0.053, refcv7-50400 bias -0.086, R7-20k bias +0.010; RAW8-20k MAE 1.553, refcv7-50400 MAE 1.447, R7-20k MAE 1.623
* seg3 (n 1575): RAW8-20k bias -0.518, refcv7-50400 bias -0.604, R7-20k bias -0.644; RAW8-20k MAE 2.145, refcv7-50400 MAE 1.979, R7-20k MAE 2.157
* seg4 (n 1566): RAW8-20k bias -0.621, refcv7-50400 bias -0.693, R7-20k bias -0.853; RAW8-20k MAE 2.946, refcv7-50400 MAE 2.668, R7-20k MAE 2.794
**sharp (>= 0.15)**

* seg1 (n 364): RAW8-20k bias -0.088, refcv7-50400 bias +0.010, R7-20k bias +0.286; RAW8-20k MAE 0.977, refcv7-50400 MAE 0.917, R7-20k MAE 1.108
* seg2 (n 365): RAW8-20k bias -0.636, refcv7-50400 bias -0.528, R7-20k bias +0.085; RAW8-20k MAE 2.640, refcv7-50400 MAE 2.418, R7-20k MAE 2.818
* seg3 (n 365): RAW8-20k bias -1.843, refcv7-50400 bias -1.863, R7-20k bias -1.602; RAW8-20k MAE 4.022, refcv7-50400 MAE 3.649, R7-20k MAE 4.038
* seg4 (n 363): RAW8-20k bias -2.083, refcv7-50400 bias -1.837, R7-20k bias -2.018; RAW8-20k MAE 5.501, refcv7-50400 MAE 4.887, R7-20k MAE 5.156
**curved & v0 5-10**

* seg1 (n 627): RAW8-20k bias -0.009, refcv7-50400 bias -0.038, R7-20k bias +0.030; RAW8-20k MAE 0.590, refcv7-50400 MAE 0.535, R7-20k MAE 0.661
* seg2 (n 627): RAW8-20k bias -0.176, refcv7-50400 bias -0.282, R7-20k bias -0.169; RAW8-20k MAE 1.522, refcv7-50400 MAE 1.363, R7-20k MAE 1.598
* seg3 (n 627): RAW8-20k bias -0.662, refcv7-50400 bias -0.871, R7-20k bias -0.896; RAW8-20k MAE 2.216, refcv7-50400 MAE 2.019, R7-20k MAE 2.199
* seg4 (n 627): RAW8-20k bias -0.859, refcv7-50400 bias -1.194, R7-20k bias -1.283; RAW8-20k MAE 3.076, refcv7-50400 MAE 2.730, R7-20k MAE 2.839
**sharp & v0 5-10**

* seg1 (n 164): RAW8-20k bias +0.083, refcv7-50400 bias +0.103, R7-20k bias +0.482; RAW8-20k MAE 0.854, refcv7-50400 MAE 0.857, R7-20k MAE 1.100
* seg2 (n 164): RAW8-20k bias -0.325, refcv7-50400 bias -0.422, R7-20k bias +0.462; RAW8-20k MAE 2.332, refcv7-50400 MAE 2.243, R7-20k MAE 2.699
* seg3 (n 164): RAW8-20k bias -1.725, refcv7-50400 bias -2.032, R7-20k bias -1.436; RAW8-20k MAE 3.624, refcv7-50400 MAE 3.465, R7-20k MAE 3.740
* seg4 (n 164): RAW8-20k bias -2.203, refcv7-50400 bias -2.403, R7-20k bias -2.138; RAW8-20k MAE 5.135, refcv7-50400 MAE 4.635, R7-20k MAE 4.837

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1566 / 1566 | 10.3 | RAW8-20k -0.0980 [-0.1961, -0.0226]; refcv7-50400 -0.0833 [-0.1550, -0.0269]; R7-20k -0.0976 [-0.1805, -0.0361] | RAW8-20k -0.0147 [-0.0565, +0.0210]; R7-20k -0.0143 [-0.0485, +0.0190] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1582 / 4719 | 7.2 | RAW8-20k -0.1580 [-0.2921, -0.0487]; refcv7-50400 -0.1457 [-0.2637, -0.0522]; R7-20k -0.1512 [-0.2768, -0.0497] | RAW8-20k -0.0122 [-0.0489, +0.0194]; R7-20k -0.0055 [-0.0372, +0.0265] |
| sharp (>= 0.15) | last segment | 363 / 363 | 25.4 | RAW8-20k -0.1149 [-0.2276, -0.0296]; refcv7-50400 -0.0921 [-0.1765, -0.0255]; R7-20k -0.1059 [-0.2043, -0.0351] | RAW8-20k -0.0228 [-0.0725, +0.0173]; R7-20k -0.0138 [-0.0533, +0.0250] |
| sharp (>= 0.15) | segments 2..K | 366 / 1093 | 17.9 | RAW8-20k -0.1240 [-0.2378, -0.0366]; refcv7-50400 -0.1041 [-0.1873, -0.0360]; R7-20k -0.1061 [-0.2062, -0.0315] | RAW8-20k -0.0199 [-0.0660, +0.0176]; R7-20k -0.0020 [-0.0424, +0.0386] |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | RAW8-20k -0.0838 [-0.1568, -0.0265]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-20k -0.0925 [-0.1693, -0.0394] | RAW8-20k +0.0087 [-0.0241, +0.0467]; R7-20k -0.0000 [-0.0233, +0.0247] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | RAW8-20k -0.0915 [-0.1659, -0.0331]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-20k -0.0932 [-0.1722, -0.0382] | RAW8-20k +0.0078 [-0.0217, +0.0412]; R7-20k +0.0061 [-0.0160, +0.0286] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | RAW8-20k -0.0875 [-0.1719, -0.0246]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-20k -0.0853 [-0.1691, -0.0256] | RAW8-20k +0.0038 [-0.0280, +0.0396]; R7-20k +0.0059 [-0.0173, +0.0316] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | RAW8-20k -0.0896 [-0.1754, -0.0248]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-20k -0.0795 [-0.1657, -0.0181] | RAW8-20k +0.0059 [-0.0251, +0.0391]; R7-20k +0.0160 [-0.0068, +0.0412] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW8-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4506 | 0.0909 | 0.01177 | 0.02999 | 0.539 / 0.461 / 0.000 | 352.4 / 409.7 |
| turn|straight <0.02 | 2901 | 0.0599 | 0.00980 | 0.01804 | 0.544 / 0.456 / 0.000 | 131.1 / 173.8 |
| turn|gentle 0.02-0.06 | 719 | 0.0634 | -0.00856 | -0.00207 | 0.477 / 0.523 / 0.000 | 68.7 / 45.6 |
| turn|turn 0.06-0.15 | 503 | 0.1321 | 0.10546 | 0.10606 | 0.594 / 0.406 / 0.000 | 40.1 / 66.5 |
| turn|sharp >=0.15 | 366 | 0.3339 | 0.10534 | 0.15368 | 0.541 / 0.459 / 0.000 | 64.0 / 122.2 |
| turn|undef (GT crawling) | 17 | 0.0988 | 0.01642 | 0.02319 | 0.529 / 0.471 / 0.000 | 2.0 / 1.7 |
| speed|v0 <2 | 217 | 0.0984 | -0.01297 | -0.02396 | 0.465 / 0.535 / 0.000 | 21.0 / 21.4 |
| speed|v0 2-5 | 516 | 0.1253 | 0.01957 | 0.04952 | 0.521 / 0.479 / 0.000 | 83.8 / 64.6 |
| speed|v0 5-10 | 1390 | 0.1675 | 0.01444 | 0.04313 | 0.539 / 0.461 / 0.000 | 135.9 / 232.9 |
| speed|v0 10-15 | 1192 | 0.0580 | 0.02192 | 0.04200 | 0.563 / 0.437 / 0.000 | 22.4 / 69.2 |
| speed|v0 15-25 | 856 | 0.0169 | 0.00525 | 0.00722 | 0.518 / 0.482 / 0.000 | 11.0 / 14.5 |
| speed|v0 >=25 | 335 | 0.0215 | 0.00961 | 0.01656 | 0.582 / 0.418 / 0.000 | 1.7 / 7.2 |
| nav|nav=follow | 2880 | 0.0875 | 0.01252 | 0.03112 | 0.549 / 0.451 / 0.000 | 140.9 / 251.9 |
| nav|nav=left | 404 | 0.0302 | 0.03014 | 0.03479 | 0.557 / 0.443 / 0.000 | 25.0 / 12.2 |
| nav|nav=right | 1222 | 0.1192 | 0.00355 | 0.01739 | 0.509 / 0.491 / 0.000 | 158.8 / 145.6 |
**RAW8-20k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4497 | 0.0173 | -0.00665 | -0.00634 | 0.481 / 0.519 / 0.000 | 346.1 / 77.8 |
| turn|straight <0.02 | 2897 | 0.0261 | -0.00311 | -0.00024 | 0.490 / 0.510 / 0.000 | 135.9 / 75.5 |
| turn|gentle 0.02-0.06 | 719 | 0.0911 | -0.03095 | -0.01672 | 0.460 / 0.540 / 0.000 | 73.5 / 65.5 |
| turn|turn 0.06-0.15 | 503 | -0.1419 | -0.01750 | -0.04604 | 0.487 / 0.513 / 0.000 | 38.8 / -71.4 |
| turn|sharp >=0.15 | 366 | 0.0080 | -0.13433 | -0.14268 | 0.445 / 0.555 / 0.000 | 41.3 / 2.9 |
| turn|undef (GT crawling) | 12 | 0.4351 | 0.02812 | 0.08434 | 0.583 / 0.417 / 0.000 | 4.3 / 5.2 |
| speed|v0 <2 | 208 | -0.0116 | -0.02760 | -0.07644 | 0.462 / 0.538 / 0.000 | 20.2 / -2.4 |
| speed|v0 2-5 | 516 | -0.0927 | -0.03102 | -0.07799 | 0.457 / 0.543 / 0.000 | 62.2 / -47.8 |
| speed|v0 5-10 | 1390 | 0.0676 | -0.02303 | -0.00190 | 0.463 / 0.537 / 0.000 | 127.1 / 93.9 |
| speed|v0 10-15 | 1192 | 0.0284 | -0.00101 | 0.01020 | 0.498 / 0.502 / 0.000 | 32.1 / 33.9 |
| speed|v0 15-25 | 856 | -0.0061 | -0.00562 | -0.00584 | 0.477 / 0.523 / 0.000 | 15.4 / -5.2 |
| speed|v0 >=25 | 335 | 0.0164 | 0.01016 | 0.01327 | 0.558 / 0.442 / 0.000 | 1.7 / 5.5 |
| nav|nav=follow | 2872 | 0.0364 | -0.00075 | 0.00656 | 0.498 / 0.502 / 0.000 | 119.9 / 104.4 |
| nav|nav=left | 404 | -0.0052 | 0.00099 | 0.01120 | 0.507 / 0.493 / 0.000 | 45.2 / -2.1 |
| nav|nav=right | 1221 | -0.0200 | -0.03468 | -0.06124 | 0.434 / 0.566 / 0.000 | 130.9 / -24.4 |
**R7-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4520 | 0.0748 | 0.01970 | 0.03531 | 0.567 / 0.433 / 0.000 | 264.4 / 338.0 |
| turn|straight <0.02 | 2898 | 0.0339 | 0.01302 | 0.01926 | 0.565 / 0.435 / 0.000 | 82.4 / 98.2 |
| turn|gentle 0.02-0.06 | 721 | -0.0248 | 0.02203 | 0.01421 | 0.530 / 0.470 / 0.000 | 15.5 / -17.9 |
| turn|turn 0.06-0.15 | 512 | 0.2766 | 0.11826 | 0.16205 | 0.619 / 0.381 / 0.000 | 34.9 / 141.6 |
| turn|sharp >=0.15 | 368 | 0.3185 | 0.16131 | 0.22209 | 0.584 / 0.416 / 0.000 | 54.9 / 117.2 |
| turn|undef (GT crawling) | 21 | -0.0553 | 0.03690 | 0.06354 | 0.571 / 0.429 / 0.000 | 1.0 / -1.2 |
| speed|v0 <2 | 231 | 0.1294 | 0.02261 | 0.05327 | 0.550 / 0.450 / 0.000 | 10.0 / 29.9 |
| speed|v0 2-5 | 516 | 0.2180 | 0.07220 | 0.11227 | 0.576 / 0.424 / 0.000 | 83.4 / 112.5 |
| speed|v0 5-10 | 1390 | 0.1000 | 0.03468 | 0.04677 | 0.573 / 0.427 / 0.000 | 72.8 / 138.9 |
| speed|v0 10-15 | 1192 | 0.0296 | 0.01950 | 0.03041 | 0.580 / 0.420 / 0.000 | 18.5 / 35.3 |
| speed|v0 15-25 | 856 | 0.0230 | 0.01076 | 0.01462 | 0.556 / 0.444 / 0.000 | 11.7 / 19.7 |
| speed|v0 >=25 | 335 | 0.0051 | 0.00395 | 0.00567 | 0.525 / 0.475 / 0.000 | 1.8 / 1.7 |
| nav|nav=follow | 2872 | 0.0518 | 0.01277 | 0.02296 | 0.554 / 0.446 / 0.000 | 107.9 / 148.9 |
| nav|nav=left | 412 | 0.0302 | 0.03076 | 0.03300 | 0.602 / 0.398 / 0.000 | 43.6 / 12.5 |
| nav|nav=right | 1236 | 0.1429 | 0.03541 | 0.07311 | 0.587 / 0.413 / 0.000 | 89.8 / 176.6 |

### S2 -- the episodes carrying the summed delta (panel file index only)

**RAW8-20k - refcv7-50400** (sum 409.7 over 139 episodes)

* ep013: n 34, share 0.107, mean delta 1.2904, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep027: n 34, share 0.102, mean delta 1.2323, sharp-window frac 0.00, curved frac 0.09, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep059: n 34, share 0.099, mean delta 1.1940, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep061: n 34, share 0.072, mean delta 0.8686, sharp-window frac 0.62, curved frac 1.00, mean v0 9.5, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep041: n 34, share 0.066, mean delta 0.7926, sharp-window frac 0.18, curved frac 0.47, mean v0 9.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep121: n 35, share 0.061, mean delta 0.7137, sharp-window frac 0.20, curved frac 0.74, mean v0 6.1, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
**RAW8-20k - R7-20k** (sum 77.8 over 139 episodes)

* ep104: n 35, share 0.525, mean delta 1.1667, sharp-window frac 0.29, curved frac 0.31, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 35, 'nav=right': 0}
* ep059: n 34, share 0.509, mean delta 1.1651, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep027: n 34, share 0.444, mean delta 1.0165, sharp-window frac 0.00, curved frac 0.09, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep078: n 34, share 0.395, mean delta 0.9047, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep137: n 29, share 0.270, mean delta 0.7247, sharp-window frac 0.41, curved frac 0.90, mean v0 3.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 29}
* ep121: n 35, share 0.257, mean delta 0.5711, sharp-window frac 0.20, curved frac 0.74, mean v0 6.1, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
**R7-20k - refcv7-50400** (sum 338.0 over 139 episodes)

* ep067: n 34, share 0.140, mean delta 1.3928, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep041: n 34, share 0.101, mean delta 1.0037, sharp-window frac 0.18, curved frac 0.47, mean v0 9.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.088, mean delta 0.8721, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep082: n 31, share 0.076, mean delta 0.8308, sharp-window frac 0.00, curved frac 0.45, mean v0 8.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 31}
* ep124: n 34, share 0.056, mean delta 0.5595, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep129: n 31, share 0.053, mean delta 0.5812, sharp-window frac 0.32, curved frac 0.52, mean v0 4.7, nav {'nav=follow': 31, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW8-20k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1658 / -0.0137 [-0.0350, 0.0076] | 69 / -0.0176 [-0.3637, 0.2686] |
| only new agrees | 344 / -0.2448 [-0.4249, -0.0807] | 38 / -1.0514 [-2.1263, -0.2515] |
| only earlier agrees | 792 / 0.1101 [0.0417, 0.1829] | 40 / 0.1996 [-0.9789, 1.2629] |
| neither agrees | 1707 / 0.0577 [-0.0584, 0.1917] | 219 / 0.1837 [-0.3517, 0.8233] |
**RAW8-20k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1655 / -0.0080 [-0.0313, 0.0140] | 68 / 0.0864 [-0.2381, 0.3371] |
| only new agrees | 340 / -0.1269 [-0.2713, -0.0068] | 42 / -0.3102 [-0.8314, 0.2801] |
| only earlier agrees | 788 / 0.1315 [0.0468, 0.2255] | 34 / -0.0640 [-1.2329, 1.1568] |
| neither agrees | 1716 / 0.0158 [-0.0983, 0.1357] | 222 / 0.0367 [-0.3911, 0.4751] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW8-20k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4477 | 0.0064 [-0.0592, 0.0680] | 0.0589 [-0.0136, 0.1443] | -0.0487 [-0.1356, 0.0252] |
| turn|straight <0.02 | 2888 | 0.0035 [-0.0428, 0.0492] | -0.0207 [-0.0518, 0.0116] | 0.0232 [-0.0146, 0.0653] |
| turn|gentle 0.02-0.06 | 718 | -0.0662 [-0.2164, 0.0534] | 0.0215 [-0.0281, 0.0865] | -0.0864 [-0.2366, 0.0406] |
| turn|turn 0.06-0.15 | 499 | -0.0293 [-0.2293, 0.1214] | 0.0702 [-0.1827, 0.3249] | -0.0893 [-0.4017, 0.1642] |
| turn|sharp >=0.15 | 365 | 0.2208 [-0.4379, 0.7690] | 0.7306 [0.0919, 1.5744] | -0.4834 [-1.3590, 0.1597] |
| turn|undef (GT crawling) | 7 | 0.0272 [-0.1596, 0.2019] | 0.2725 [-0.0434, 0.8481] | -0.2637 [-1.0963, 0.1886] |
| speed|v0 <2 | 188 | 0.2531 [0.0383, 0.6597] | 0.1807 [-0.0229, 0.3663] | 0.1397 [-0.1607, 0.5874] |
| speed|v0 2-5 | 516 | -0.1119 [-0.5481, 0.2828] | 0.0293 [-0.2999, 0.4019] | -0.1413 [-0.5864, 0.2434] |
| speed|v0 5-10 | 1390 | 0.0239 [-0.0937, 0.1319] | 0.1443 [-0.0058, 0.3801] | -0.1204 [-0.3557, 0.0389] |
| speed|v0 10-15 | 1192 | -0.0122 [-0.0332, 0.0062] | 0.0070 [-0.0086, 0.0222] | -0.0192 [-0.0442, 0.0053] |
| speed|v0 15-25 | 856 | 0.0223 [0.0040, 0.0446] | -0.0014 [-0.0103, 0.0073] | 0.0237 [0.0051, 0.0456] SEP |
| speed|v0 >=25 | 335 | -0.0008 [-0.0137, 0.0093] | 0.0033 [-0.0054, 0.0136] | -0.0041 [-0.0171, 0.0083] |
**RAW8-20k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* RAW8-20k vs R7-20k|seed0: anchor flip rate 0.430; heading delta on flipped windows 0.0837 [-0.0305, 0.1988] (n 1923) vs unflipped -0.0319 [-0.0594, -0.0059]; goal_dist_sel proxy new-earlier -0.4350 [-0.7031, -0.1594]; corr(window dHeading, window dGoalDist) 0.026
* RAW8-20k vs R7-20k|seed1: anchor flip rate 0.432; heading delta on flipped windows 0.0845 [-0.0309, 0.2104] (n 1933) vs unflipped -0.0347 [-0.0632, -0.0067]; goal_dist_sel proxy new-earlier -0.4398 [-0.7067, -0.1675]; corr(window dHeading, window dGoalDist) 0.038

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- RAW8-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3515 | 139 | 3.5115 | 3.1074 | 0.4041 [0.1869, 0.6517] | **SEP** |  |
| turn: straight <0.02 | 2095 | 123 | 1.7434 | 1.2983 | 0.4451 [0.2168, 0.7775] | **SEP** | 0.596 / 0.656 / 0.1014 [-0.6381, 0.7559] |
| turn: gentle 0.02-0.06 | 677 | 76 | 3.0555 | 2.7384 | 0.3171 [0.1032, 0.5635] | **SEP** | 0.193 / 0.151 / -0.1078 [-0.6294, 0.3845] |
| turn: turn 0.06-0.15 | 427 | 59 | 6.4716 | 6.3397 | 0.1319 [-0.2003, 0.4718] |  | 0.121 / 0.040 / -0.3099 [-0.8942, 0.2673] |
| turn: sharp >=0.15 | 302 | 33 | 12.1487 | 11.4497 | 0.6991 [-0.7339, 2.3939] |  | 0.086 / 0.149 / 0.3227 [-1.8682, 2.9375] |
| turn: undef (GT crawling) (low n) | 14 | 11 | 13.5561 | 13.1323 | 0.4238 [0.0863, 0.8782] | **SEP** | 0.004 / 0.004 / 0.0197 [-0.6212, 0.8734] |
| speed: v0 <2 | 177 | 25 | 12.3523 | 11.6434 | 0.7088 [-0.4191, 1.9768] |  | 0.050 / 0.088 / 0.3209 [-1.8910, 2.5594] |
| speed: v0 2-5 | 422 | 52 | 4.9951 | 4.1501 | 0.8449 [-0.0163, 1.9392] |  | 0.120 / 0.251 / 0.5009 [-0.7702, 2.4920] |
| speed: v0 5-10 | 1040 | 79 | 4.0648 | 3.4941 | 0.5707 [0.0281, 1.2049] | **SEP** | 0.296 / 0.418 / 0.2365 [-0.6032, 1.2996] |
| speed: v0 10-15 | 952 | 62 | 2.6610 | 2.4870 | 0.1739 [0.0189, 0.3238] | **SEP** | 0.271 / 0.117 / -0.3157 [-0.8380, 0.0888] |
| speed: v0 15-25 | 679 | 34 | 1.7151 | 1.4456 | 0.2695 [0.1531, 0.3935] | **SEP** | 0.193 / 0.129 / -0.1668 [-0.6383, 0.2311] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5047 | 0.5193 | -0.0146 [-0.0543, 0.0357] |  | 0.070 / -0.003 / -0.4501 [-0.8670, -0.1339] BONF-SEP |
| nav: nav=follow | 2245 | 88 | 2.2558 | 1.9793 | 0.2765 [0.1383, 0.4539] | **SEP** | 0.639 / 0.437 / -0.3533 [-1.2829, 0.3925] |
| nav: nav=left (low n) | 321 | 13 | 8.8368 | 8.3102 | 0.5266 [-0.1356, 1.2919] |  | 0.091 / 0.119 / 0.1347 [-0.9029, 1.3923] |
| nav: nav=right | 949 | 38 | 4.6809 | 4.0162 | 0.6647 [0.0412, 1.4375] | **SEP** | 0.270 / 0.444 / 0.3569 [-0.5745, 1.5347] |

episode concentration: 0.698 of episodes have mean delta > 0; top-5 episodes carry 0.552 of the summed delta; delta excluding the top 5 = 0.1881 (overall 0.4041).

### S6 -- RAW8-20k - R7-20k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3502 | 139 | 3.5185 | 3.5721 | -0.0536 [-0.2946, 0.1989] |  |  |
| turn: straight <0.02 | 2083 | 123 | 1.7447 | 1.5286 | 0.2161 [0.0131, 0.5063] | **SEP** | 0.595 / -2.397 / 0.6657 [-0.2724, 1.4793] |
| turn: gentle 0.02-0.06 | 677 | 76 | 3.0555 | 2.8404 | 0.2151 [-0.1375, 0.7455] |  | 0.193 / -0.775 / 0.3331 [-0.3295, 1.2539] |
| turn: turn 0.06-0.15 | 427 | 59 | 6.4716 | 7.1464 | -0.6748 [-1.2943, -0.0781] | **SEP** | 0.122 / 1.534 / -0.7075 [-1.6279, 0.1555] |
| turn: sharp >=0.15 | 302 | 33 | 12.1487 | 13.7935 | -1.6448 [-3.2248, -0.0349] | **SEP** | 0.086 / 2.644 / -1.7413 [-4.0272, 0.6685] |
| turn: undef (GT crawling) (low n) | 13 | 10 | 14.3706 | 14.2732 | 0.0974 [-0.0807, 0.3084] |  | 0.004 / -0.007 / 0.1516 [-0.3336, 0.6439] |
| speed: v0 <2 | 164 | 25 | 13.2019 | 13.4745 | -0.2725 [-0.7390, 0.3510] |  | 0.047 / 0.238 / -0.2296 [-0.9634, 0.8628] |
| speed: v0 2-5 | 422 | 52 | 4.9951 | 5.2980 | -0.3029 [-1.2595, 0.7575] |  | 0.121 / 0.681 / -0.2835 [-1.6727, 1.4717] |
| speed: v0 5-10 | 1040 | 79 | 4.0648 | 4.0421 | 0.0227 [-0.5877, 0.6794] |  | 0.297 / -0.126 / 0.1085 [-0.8200, 1.1521] |
| speed: v0 10-15 | 952 | 62 | 2.6610 | 2.7810 | -0.1200 [-0.3524, 0.1122] |  | 0.272 / 0.608 / -0.0912 [-0.7599, 0.4488] |
| speed: v0 15-25 | 679 | 34 | 1.7151 | 1.6197 | 0.0954 [-0.0266, 0.2133] |  | 0.194 / -0.345 / 0.1849 [-0.3383, 0.6324] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5047 | 0.4613 | 0.0433 [-0.0301, 0.1257] |  | 0.070 / -0.057 / 0.1043 [-0.3544, 0.5088] |
| nav: nav=follow | 2232 | 88 | 2.2594 | 2.3286 | -0.0691 [-0.2227, 0.0914] |  | 0.637 / 0.822 / -0.0428 [-1.0944, 0.8270] |
| nav: nav=left (low n) | 321 | 13 | 8.8368 | 8.8748 | -0.0380 [-0.7401, 0.5200] |  | 0.092 / 0.065 / 0.0172 [-1.2414, 0.9760] |
| nav: nav=right | 949 | 38 | 4.6809 | 4.7033 | -0.0224 [-0.7392, 0.9032] |  | 0.271 / 0.113 / 0.0428 [-1.1345, 1.3980] |

episode concentration: 0.482 of episodes have mean delta > 0; top-5 episodes carry -3.939 of the summed delta; delta excluding the top 5 = -0.2749 (overall -0.0536).

### S6 -- R7-20k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3553 | 139 | 3.7779 | 3.3013 | 0.4766 [0.3208, 0.6392] | **SEP** |  |
| turn: straight <0.02 | 2099 | 123 | 1.7062 | 1.4684 | 0.2379 [0.1461, 0.3431] | **SEP** | 0.591 / 0.295 / -0.5834 [-1.1715, -0.0085] BONF-SEP |
| turn: gentle 0.02-0.06 | 679 | 77 | 3.0078 | 2.8937 | 0.1141 [-0.2425, 0.3726] |  | 0.191 / 0.046 / -0.4481 [-1.0838, -0.0448] BONF-SEP |
| turn: turn 0.06-0.15 | 442 | 59 | 7.1717 | 6.3377 | 0.8340 [0.3322, 1.3347] | **SEP** | 0.124 / 0.218 / 0.4082 [-0.3512, 1.2173] |
| turn: sharp >=0.15 | 312 | 33 | 14.1420 | 11.7318 | 2.4102 [1.2515, 3.7058] | **SEP** | 0.088 / 0.444 / 2.1197 [0.5754, 4.3614] BONF-SEP |
| turn: undef (GT crawling) (low n) | 21 | 11 | 10.3328 | 10.5237 | -0.1909 [-0.5355, 0.1873] |  | 0.006 / -0.002 / -0.6715 [-1.2239, -0.0640] BONF-SEP |
| speed: v0 <2 | 215 | 28 | 14.5255 | 13.3387 | 1.1869 [0.4042, 2.2910] | **SEP** | 0.061 / 0.151 / 0.7560 [-0.4419, 2.6216] |
| speed: v0 2-5 | 422 | 52 | 5.2980 | 4.1501 | 1.1479 [0.2733, 2.0555] | **SEP** | 0.119 / 0.286 / 0.7617 [-0.6335, 2.4793] |
| speed: v0 5-10 | 1040 | 79 | 4.0421 | 3.4941 | 0.5480 [0.2695, 0.8686] | **SEP** | 0.293 / 0.337 / 0.1009 [-0.4337, 0.6905] |
| speed: v0 10-15 | 952 | 62 | 2.7810 | 2.4870 | 0.2940 [0.0543, 0.5494] | **SEP** | 0.268 / 0.165 / -0.2495 [-0.6955, 0.2878] |
| speed: v0 15-25 | 679 | 34 | 1.6197 | 1.4456 | 0.1741 [0.0571, 0.3164] | **SEP** | 0.191 / 0.070 / -0.3740 [-0.6823, -0.0098] BONF-SEP |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4613 | 0.5193 | -0.0580 [-0.1242, 0.0010] |  | 0.069 / -0.008 / -0.5742 [-0.8548, -0.2880] BONF-SEP |
| nav: nav=follow | 2240 | 88 | 2.3251 | 1.9803 | 0.3448 [0.1893, 0.5162] | **SEP** | 0.630 / 0.456 / -0.3566 [-0.9083, 0.1819] |
| nav: nav=left (low n) | 334 | 13 | 10.4774 | 9.8401 | 0.6373 [0.1485, 1.2859] | **SEP** | 0.094 / 0.126 / 0.1773 [-0.5691, 1.3100] |
| nav: nav=right | 979 | 38 | 4.8163 | 4.0930 | 0.7233 [0.3019, 1.1034] | **SEP** | 0.276 / 0.418 / 0.3405 [-0.3174, 0.9958] |

episode concentration: 0.705 of episodes have mean delta > 0; top-5 episodes carry 0.275 of the summed delta; delta excluding the top 5 = 0.3589 (overall 0.4766).

### S6 -- RAW8-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3474 | 139 | 0.9153 | 0.7119 | 0.2033 [0.0291, 0.5290] | **SEP** | 0.0588 [-0.1351, 0.3989] |
| seg1 | straight | 2064 | 122 | 0.6721 | 0.4574 | 0.2147 [0.0242, 0.6025] | **SEP** | 0.1258 [-0.0746, 0.5150] |
| seg1 | curved | 1397 | 88 | 1.1496 | 0.9635 | 0.1862 [0.0153, 0.4818] | **SEP** | -0.0358 [-0.2533, 0.2722] |
| seg2 | all | 3469 | 139 | 1.7208 | 1.3490 | 0.3718 [0.1070, 0.8224] | **SEP** | 0.0720 [-0.3537, 0.7678] |
| seg2 | straight | 2072 | 122 | 1.1252 | 0.6654 | 0.4599 [0.0499, 1.2486] | **SEP** | 0.2524 [-0.1759, 1.0345] |
| seg2 | curved | 1397 | 88 | 2.6041 | 2.3629 | 0.2411 [0.0971, 0.4179] | **SEP** | -0.1957 [-0.7289, 0.4607] |
| seg3 | all | 3438 | 138 | 2.4199 | 1.8901 | 0.5298 [0.1920, 1.0516] | **SEP** | 0.0063 [-0.5274, 0.7977] |
| seg3 | straight | 2053 | 121 | 1.3994 | 0.8029 | 0.5965 [0.1035, 1.4218] | **SEP** | 0.4808 [-0.2267, 1.6840] |
| seg3 | curved | 1385 | 87 | 3.9325 | 3.5017 | 0.4308 [0.0980, 0.8179] | **SEP** | -0.6972 [-1.4238, -0.0891] SEP |
| seg4 | all | 3408 | 138 | 3.5306 | 2.8903 | 0.6403 [0.3244, 1.0331] | **SEP** | 0.0798 [-0.5616, 0.9126] |
| seg4 | straight | 2040 | 121 | 1.7386 | 1.0511 | 0.6875 [0.2792, 1.2357] | **SEP** | 0.5961 [-0.2109, 1.8648] |
| seg4 | curved | 1368 | 86 | 6.2029 | 5.6329 | 0.5700 [0.0891, 1.1416] | **SEP** | -0.6901 [-1.5932, 0.1861] |
| seg5 | all | 3367 | 138 | 5.0023 | 4.2253 | 0.7770 [0.3646, 1.2478] | **SEP** | 0.2169 [-0.6566, 1.3886] |
| seg5 | straight | 2019 | 122 | 2.1711 | 1.2277 | 0.9434 [0.4448, 1.6205] | **SEP** | 0.8197 [-0.3180, 2.6058] |
| seg5 | curved | 1348 | 87 | 9.2427 | 8.7150 | 0.5277 [-0.1396, 1.2750] |  | -0.6860 [-1.8543, 0.4854] |
| seg6 | all | 3336 | 138 | 6.6472 | 5.8827 | 0.7644 [0.3073, 1.2791] | **SEP** | -0.0728 [-0.9964, 0.9933] |
| seg6 | straight | 2010 | 122 | 2.5567 | 1.6463 | 0.9104 [0.4392, 1.5850] | **SEP** | 0.6333 [-0.4351, 2.0736] |
| seg6 | curved | 1325 | 87 | 12.8552 | 12.3136 | 0.5416 [-0.2603, 1.4279] |  | -1.1416 [-2.6951, 0.3499] |

### S6 -- RAW8-20k - R7-20k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3474 | 139 | 0.9153 | 0.9328 | -0.0175 [-0.0507, 0.0149] |  | -0.1269 [-0.1874, -0.0752] SEP |
| seg1 | straight | 2064 | 122 | 0.6721 | 0.5718 | 0.1003 [-0.0032, 0.2973] |  | 0.0091 [-0.1098, 0.2115] |
| seg1 | curved | 1397 | 88 | 1.1496 | 1.3430 | -0.1934 [-0.5032, -0.0038] | **SEP** | -0.3266 [-0.6279, -0.1160] SEP |
| seg2 | all | 3464 | 139 | 1.7231 | 1.4903 | 0.2328 [0.0135, 0.5728] | **SEP** | -0.1193 [-0.4120, 0.2618] |
| seg2 | straight | 2067 | 122 | 1.1277 | 0.8816 | 0.2460 [0.0116, 0.6599] | **SEP** | 0.0969 [-0.1596, 0.5122] |
| seg2 | curved | 1397 | 88 | 2.6041 | 2.3908 | 0.2132 [-0.0575, 0.5464] |  | -0.4391 [-0.8814, 0.0200] |
| seg3 | all | 3428 | 138 | 2.4262 | 2.3098 | 0.1164 [-0.0821, 0.3306] |  | -0.2324 [-0.5543, 0.0693] |
| seg3 | straight | 2043 | 121 | 1.4051 | 1.2445 | 0.1606 [0.0074, 0.3792] | **SEP** | 0.1298 [-0.0957, 0.3007] |
| seg3 | curved | 1385 | 87 | 3.9325 | 3.8812 | 0.0513 [-0.3381, 0.5306] |  | -0.7667 [-1.4819, -0.0432] SEP |
| seg4 | all | 3391 | 138 | 3.5353 | 3.4325 | 0.1028 [-0.2178, 0.4350] |  | -0.0297 [-0.4909, 0.4189] |
| seg4 | straight | 2027 | 121 | 1.7443 | 1.3488 | 0.3955 [0.0903, 0.7718] | **SEP** | 0.4646 [0.0909, 0.7987] SEP |
| seg4 | curved | 1364 | 86 | 6.1968 | 6.5289 | -0.3321 [-0.9541, 0.3693] |  | -0.7643 [-1.7279, 0.2201] |
| seg5 | all | 3355 | 138 | 5.0106 | 5.0944 | -0.0838 [-0.5193, 0.3512] |  | 0.1414 [-0.5090, 0.7929] |
| seg5 | straight | 2008 | 122 | 2.1730 | 1.6985 | 0.4745 [0.0742, 0.9925] | **SEP** | 0.7165 [0.2131, 1.1564] SEP |
| seg5 | curved | 1347 | 87 | 9.2407 | 10.1568 | -0.9161 [-1.7203, 0.0095] |  | -0.7159 [-2.0626, 0.6635] |
| seg6 | all | 3321 | 137 | 6.6526 | 6.9693 | -0.3167 [-0.8498, 0.2414] |  | 0.3910 [-0.4240, 1.3289] |
| seg6 | straight | 1999 | 121 | 2.5537 | 2.0625 | 0.4912 [-0.0056, 1.0558] |  | 1.0807 [0.3319, 1.9111] SEP |
| seg6 | curved | 1322 | 87 | 12.8505 | 14.3888 | -1.5384 [-2.5148, -0.4379] | **SEP** | -0.6518 [-2.2387, 1.0248] |

### S6 -- R7-20k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3501 | 139 | 1.0349 | 0.8146 | 0.2203 [0.0403, 0.5454] | **SEP** | 0.2892 [0.0013, 0.8230] SEP |
| seg1 | straight | 2072 | 122 | 0.6572 | 0.5433 | 0.1139 [0.0175, 0.3106] | **SEP** | 0.2903 [0.0082, 0.8694] SEP |
| seg1 | curved | 1410 | 88 | 1.4700 | 1.0917 | 0.3782 [0.0662, 0.9424] | **SEP** | 0.2908 [-0.0309, 0.8461] |
| seg2 | all | 3485 | 139 | 1.6554 | 1.4614 | 0.1940 [0.0195, 0.4512] | **SEP** | 0.3507 [-0.0159, 0.9989] |
| seg2 | straight | 2074 | 122 | 1.0543 | 0.8393 | 0.2151 [0.0246, 0.5990] | **SEP** | 0.3294 [-0.0552, 1.0995] |
| seg2 | curved | 1411 | 88 | 2.5389 | 2.3758 | 0.1632 [-0.0471, 0.3748] |  | 0.3822 [-0.0005, 0.9312] |
| seg3 | all | 3458 | 139 | 2.4449 | 2.0719 | 0.3730 [0.1132, 0.7424] | **SEP** | 0.5079 [-0.1707, 1.7158] |
| seg3 | straight | 2053 | 122 | 1.3310 | 0.9779 | 0.3531 [0.0590, 0.9477] | **SEP** | 0.6138 [-0.2312, 2.3653] |
| seg3 | curved | 1405 | 88 | 4.0726 | 3.6705 | 0.4021 [0.0589, 0.7412] | **SEP** | 0.3532 [-0.1600, 0.9748] |
| seg4 | all | 3428 | 139 | 3.5949 | 2.9802 | 0.6147 [0.3368, 0.9412] | **SEP** | 0.2839 [-0.3123, 1.2199] |
| seg4 | straight | 2041 | 122 | 1.5269 | 1.1359 | 0.3910 [0.1465, 0.8095] | **SEP** | 0.3913 [-0.4062, 1.9697] |
| seg4 | curved | 1387 | 87 | 6.6381 | 5.6943 | 0.9438 [0.4438, 1.4316] | **SEP** | 0.1257 [-0.4377, 0.6708] |
| seg5 | all | 3396 | 139 | 5.2468 | 4.3430 | 0.9037 [0.6252, 1.2007] | **SEP** | 0.2115 [-0.5311, 1.3192] |
| seg5 | straight | 2020 | 122 | 1.7976 | 1.3137 | 0.4839 [0.2685, 0.7550] | **SEP** | 0.2696 [-0.7013, 2.1164] |
| seg5 | curved | 1376 | 88 | 10.3102 | 8.7902 | 1.5201 [0.8910, 2.1643] | **SEP** | 0.1261 [-0.6410, 0.9120] |
| seg6 | all | 3361 | 138 | 7.1408 | 6.0071 | 1.1337 [0.7956, 1.4877] | **SEP** | -0.2801 [-0.8634, 0.4655] |
| seg6 | straight | 2009 | 121 | 2.1736 | 1.7386 | 0.4351 [0.1206, 0.7248] | **SEP** | -0.2906 [-0.9830, 0.8212] |
| seg6 | curved | 1350 | 88 | 14.5231 | 12.3433 | 2.1798 [1.4171, 2.9309] | **SEP** | -0.2697 [-1.1622, 0.6428] |

### S6 -- RAW8-20k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 361 | 40 | 4.7908 | 4.2681 | 0.5227 [0.2869, 0.8645] | **SEP** |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 1.4371 | 0.7632 | 0.6739 [0.2685, 1.2336] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.6052 | 0.5523 | 0.0529 [-0.0001, 0.1159] |  |  |
| gentle 0.02-0.06 | v0 <5 | 100 | 21 | 5.0327 | 4.4489 | 0.5837 [-0.0449, 1.3187] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.7032 | 2.5310 | 0.1722 [-0.1060, 0.5455] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.7395 | 2.1952 | 0.5443 [0.2636, 0.7861] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 56 | 20 | 11.3655 | 11.8230 | -0.4575 [-1.9063, 1.2770] |  |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 5.9259 | 5.8272 | 0.0987 [-0.2046, 0.4072] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 5.0078 | 4.3281 | 0.6797 [0.4197, 1.0014] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 68 | 16 | 18.1654 | 14.4209 | 3.7445 [0.2167, 8.7629] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 10.5297 | 10.7541 | -0.2243 [-1.9238, 1.4041] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 7.1652 | 6.3909 | 0.7743 [-1.5129, 1.1440] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 14 | 11 | 13.5561 | 13.1323 | 0.4238 [0.0863, 0.8782] | **SEP** |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW8-20k s0-s1 [CI] | R7-20k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0193 [-0.1016, 0.0723] | 0.0032 [-0.0496, 0.0598] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0483 [-0.1427, 0.0116] | 0.0262 [-0.0207, 0.0960] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0616 [-0.0166, 0.1615] | 0.0292 [-0.0744, 0.1218] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.1519 [-0.4037, 0.0681] | -0.0675 [-0.2282, 0.1022] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.1695 [-0.4457, 1.0524] | -0.1077 [-0.5568, 0.3422] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.3576 [-0.1445, 0.9004] | 0.0091 [-0.1871, 0.2769] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | -0.1127 [-0.3226, 0.0765] | 0.4396 [0.0260, 1.0407] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.0240 [-0.2636, 0.1872] | -0.1181 [-0.4121, 0.0983] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | 0.0054 [-0.2072, 0.2751] | 0.0358 [-0.1045, 0.1956] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0460 [-0.1178, 0.0243] | -0.0728 [-0.1371, -0.0117] SEP | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | -0.0021 [-0.0364, 0.0304] | 0.0008 [-0.0513, 0.0624] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | 0.0078 [-0.0348, 0.0613] | -0.0123 [-0.0357, 0.0129] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0257 [-0.0691, 0.0141] | -0.0214 [-0.0692, 0.0290] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.2371 [-0.3469, 0.9947] | 0.0776 [0.0039, 0.1823] SEP | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | -0.0909 [-0.3079, 0.0812] | 0.0340 [-0.1064, 0.1917] | 0.0235 [-0.1175, 0.1775] |

### S6 -- RAW8-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.4041 | 0.3812 / 0.4252 | True / True / True | 0.0193 / 0.0128 | 21.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.4451 | 0.4310 / 0.4600 | True / True / True | 0.0483 / 0.0216 | 9.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.3171 | 0.3123 / 0.3219 | True / True / True | 0.0616 / 0.0713 | 4.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | 0.1319 | 0.0192 / 0.2250 | False / False / False | 0.1519 / 0.0611 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.6991 | 0.6953 / 0.7055 | False / False / False | 0.1695 / 0.2004 | 3.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.4238 | 0.5803 / 0.2672 | True / True / True | 0.3576 / 1.9278 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.7088 | 0.4936 / 0.8736 | False / False / False | 0.1127 / 0.0328 | 6.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.8449 | 0.7325 / 0.9574 | False / True / False | 0.0240 / 0.2009 | 4.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.5707 | 0.5855 / 0.5558 | True / False / True | 0.0054 / 0.0243 | 23.5 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 10-15 | 0.1739 | 0.1643 / 0.1835 | False / True / True | 0.0460 / 0.0268 | 3.8 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 15-25 | 0.2695 | 0.2658 / 0.2733 | True / True / True | 0.0021 / 0.0055 | 49.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0146 | -0.0112 / -0.0181 | False / False / False | 0.0078 / 0.0008 | 1.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.2765 | 0.2593 / 0.2944 | True / True / True | 0.0257 / 0.0080 | 10.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.5266 | 0.6311 / 0.4220 | True / False / False | 0.2371 / 0.0140 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.6647 | 0.5847 / 0.7352 | True / True / True | 0.0909 / 0.0235 | 7.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- RAW8-20k - R7-20k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0536 | -0.0566 / -0.0519 | False / False / False | 0.0193 / 0.0032 | 2.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.2161 | 0.1847 / 0.2463 | True / True / True | 0.0483 / 0.0262 | 4.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.2151 | 0.2430 / 0.1872 | False / False / False | 0.0616 / 0.0292 | 3.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.6748 | -0.7137 / -0.6471 | True / False / True | 0.1519 / 0.0675 | 4.4 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|sharp >=0.15 | -1.6448 | -1.4690 / -1.8035 | False / True / True | 0.1695 / 0.1077 | 9.7 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | 0.0974 | 0.1234 / 0.0714 | False / False / False | 0.3576 / 0.0091 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.2725 | -0.4389 / -0.1209 | False / False / False | 0.1127 / 0.4396 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.3029 | -0.2559 / -0.3500 | False / False / False | 0.0240 / 0.1181 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0227 | 0.0075 / 0.0378 | False / False / False | 0.0054 / 0.0358 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.1200 | -0.1066 / -0.1335 | False / False / False | 0.0460 / 0.0728 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0954 | 0.0940 / 0.0968 | False / False / False | 0.0021 / 0.0008 | 46.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0433 | 0.0534 / 0.0333 | False / False / False | 0.0078 / 0.0123 | 3.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.0691 | -0.0719 / -0.0669 | False / False / False | 0.0257 / 0.0214 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.0380 | 0.1058 / -0.1818 | False / False / False | 0.2371 / 0.0776 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.0224 | -0.0755 / 0.0271 | False / False / False | 0.0909 / 0.0340 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- R7-20k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.4766 | 0.4724 / 0.4773 | True / True / True | 0.0032 / 0.0128 | 37.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.2379 | 0.2626 / 0.2136 | True / True / True | 0.0262 / 0.0216 | 9.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.1141 | 0.0930 / 0.1352 | False / False / False | 0.0292 / 0.0713 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.8340 | 0.7697 / 0.8982 | True / True / True | 0.0675 / 0.0611 | 12.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 2.4102 | 2.2609 / 2.5126 | True / True / True | 0.1077 / 0.2004 | 12.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.1909 | 0.9116 / -1.2935 | True / False / False | 0.0091 / 1.9278 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 1.1869 | 1.3893 / 0.9127 | True / True / True | 0.4396 / 0.0328 | 2.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 1.1479 | 0.9884 / 1.3073 | True / True / True | 0.1181 / 0.2009 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | 0.5480 | 0.5780 / 0.5180 | True / True / True | 0.0358 / 0.0243 | 15.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.2940 | 0.2710 / 0.3170 | True / True / True | 0.0728 / 0.0268 | 4.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.1741 | 0.1718 / 0.1765 | True / True / True | 0.0008 / 0.0055 | 31.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0580 | -0.0645 / -0.0514 | False / False / False | 0.0123 / 0.0008 | 4.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.3448 | 0.3299 / 0.3599 | True / True / True | 0.0214 / 0.0080 | 16.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.6373 | 0.6777 / 0.5589 | True / True / True | 0.0776 / 0.0140 | 8.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.7233 | 0.7284 / 0.7182 | True / True / True | 0.0340 / 0.0235 | 21.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1397): RAW8-20k bias +0.362, refcv7-50400 bias +0.175, R7-20k bias +0.539; RAW8-20k MAE 1.150, refcv7-50400 MAE 0.963, R7-20k MAE 1.343
* seg2 (n 1397): RAW8-20k bias +0.167, refcv7-50400 bias -0.211, R7-20k bias -0.133; RAW8-20k MAE 2.604, refcv7-50400 MAE 2.363, R7-20k MAE 2.391
* seg3 (n 1385): RAW8-20k bias -0.768, refcv7-50400 bias -0.937, R7-20k bias -1.213; RAW8-20k MAE 3.932, refcv7-50400 MAE 3.502, R7-20k MAE 3.881
* seg4 (n 1364): RAW8-20k bias -2.244, refcv7-50400 bias -2.457, R7-20k bias -3.022; RAW8-20k MAE 6.197, refcv7-50400 MAE 5.615, R7-20k MAE 6.529
* seg5 (n 1346): RAW8-20k bias -4.114, refcv7-50400 bias -4.430, R7-20k bias -5.286; RAW8-20k MAE 9.247, refcv7-50400 MAE 8.702, R7-20k MAE 10.163
* seg6 (n 1320): RAW8-20k bias -6.310, refcv7-50400 bias -6.909, R7-20k bias -7.423; RAW8-20k MAE 12.867, refcv7-50400 MAE 12.316, R7-20k MAE 14.403
**sharp (>= 0.15)**

* seg1 (n 301): RAW8-20k bias +0.188, refcv7-50400 bias +0.178, R7-20k bias +0.989; RAW8-20k MAE 1.569, refcv7-50400 MAE 1.446, R7-20k MAE 2.408
* seg2 (n 298): RAW8-20k bias -0.660, refcv7-50400 bias -0.532, R7-20k bias -1.205; RAW8-20k MAE 4.537, refcv7-50400 MAE 3.819, R7-20k MAE 4.326
* seg3 (n 295): RAW8-20k bias -2.908, refcv7-50400 bias -2.422, R7-20k bias -4.817; RAW8-20k MAE 8.004, refcv7-50400 MAE 6.681, R7-20k MAE 8.032
* seg4 (n 291): RAW8-20k bias -7.042, refcv7-50400 bias -6.542, R7-20k bias -10.406; RAW8-20k MAE 13.006, refcv7-50400 MAE 11.597, R7-20k MAE 14.512
* seg5 (n 287): RAW8-20k bias -11.650, refcv7-50400 bias -11.684, R7-20k bias -16.685; RAW8-20k MAE 19.629, refcv7-50400 MAE 18.786, R7-20k MAE 23.006
* seg6 (n 283): RAW8-20k bias -16.800, refcv7-50400 bias -17.407, R7-20k bias -21.202; RAW8-20k MAE 26.216, refcv7-50400 MAE 25.101, R7-20k MAE 30.928
**curved & v0 5-10**

* seg1 (n 547): RAW8-20k bias +0.156, refcv7-50400 bias +0.049, R7-20k bias +0.199; RAW8-20k MAE 0.974, refcv7-50400 MAE 0.896, R7-20k MAE 1.009
* seg2 (n 547): RAW8-20k bias -0.173, refcv7-50400 bias -0.516, R7-20k bias -0.417; RAW8-20k MAE 2.497, refcv7-50400 MAE 2.231, R7-20k MAE 2.356
* seg3 (n 545): RAW8-20k bias -1.054, refcv7-50400 bias -1.649, R7-20k bias -1.729; RAW8-20k MAE 4.030, refcv7-50400 MAE 3.822, R7-20k MAE 4.200
* seg4 (n 534): RAW8-20k bias -2.620, refcv7-50400 bias -3.564, R7-20k bias -3.755; RAW8-20k MAE 6.220, refcv7-50400 MAE 6.214, R7-20k MAE 7.028
* seg5 (n 519): RAW8-20k bias -4.597, refcv7-50400 bias -5.884, R7-20k bias -6.128; RAW8-20k MAE 9.222, refcv7-50400 MAE 9.372, R7-20k MAE 10.661
* seg6 (n 496): RAW8-20k bias -6.727, refcv7-50400 bias -8.488, R7-20k bias -8.150; RAW8-20k MAE 12.822, refcv7-50400 MAE 12.771, R7-20k MAE 14.743
**sharp & v0 5-10**

* seg1 (n 139): RAW8-20k bias +0.347, refcv7-50400 bias +0.062, R7-20k bias +0.642; RAW8-20k MAE 1.355, refcv7-50400 MAE 1.251, R7-20k MAE 1.538
* seg2 (n 139): RAW8-20k bias -0.539, refcv7-50400 bias -1.267, R7-20k bias -0.900; RAW8-20k MAE 4.007, refcv7-50400 MAE 3.564, R7-20k MAE 3.694
* seg3 (n 139): RAW8-20k bias -3.107, refcv7-50400 bias -4.133, R7-20k bias -5.173; RAW8-20k MAE 7.190, refcv7-50400 MAE 7.029, R7-20k MAE 7.618
* seg4 (n 138): RAW8-20k bias -7.413, refcv7-50400 bias -9.099, R7-20k bias -10.983; RAW8-20k MAE 11.819, refcv7-50400 MAE 12.470, R7-20k MAE 14.258
* seg5 (n 137): RAW8-20k bias -11.447, refcv7-50400 bias -14.439, R7-20k bias -16.121; RAW8-20k MAE 18.687, refcv7-50400 MAE 19.813, R7-20k MAE 22.584
* seg6 (n 133): RAW8-20k bias -14.584, refcv7-50400 bias -18.868, R7-20k bias -18.114; RAW8-20k MAE 25.732, refcv7-50400 MAE 26.065, R7-20k MAE 29.800

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1320 / 1320 | 25.9 | RAW8-20k -0.2300 [-0.3070, -0.1511]; refcv7-50400 -0.2487 [-0.3221, -0.1809]; R7-20k -0.2839 [-0.3676, -0.2132] | RAW8-20k +0.0187 [-0.0524, +0.0798]; R7-20k -0.0352 [-0.0767, -0.0013] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1405 / 6812 | 16.2 | RAW8-20k -0.1591 [-0.2416, -0.0795]; refcv7-50400 -0.1732 [-0.2401, -0.1130]; R7-20k -0.2101 [-0.2869, -0.1449] | RAW8-20k +0.0141 [-0.0509, +0.0776]; R7-20k -0.0370 [-0.0750, -0.0021] SEP |
| sharp (>= 0.15) | last segment | 283 / 283 | 63.1 | RAW8-20k -0.2571 [-0.3498, -0.1694]; refcv7-50400 -0.2647 [-0.3539, -0.1878]; R7-20k -0.3147 [-0.4247, -0.2313] | RAW8-20k +0.0076 [-0.0804, +0.0792]; R7-20k -0.0500 [-0.1089, -0.0065] SEP |
| sharp (>= 0.15) | segments 2..K | 301 / 1454 | 39.8 | RAW8-20k -0.2072 [-0.3108, -0.1212]; refcv7-50400 -0.1933 [-0.2798, -0.1208]; R7-20k -0.2606 [-0.3680, -0.1804] | RAW8-20k -0.0139 [-0.0931, +0.0524]; R7-20k -0.0673 [-0.1214, -0.0279] SEP |
| curved & v0 5-10 | last segment | 496 / 496 | 28.3 | RAW8-20k -0.2195 [-0.3051, -0.1045]; refcv7-50400 -0.2732 [-0.3983, -0.1730]; R7-20k -0.2741 [-0.4007, -0.1651] | RAW8-20k +0.0536 [-0.0592, +0.1750]; R7-20k -0.0010 [-0.0508, +0.0380] |
| curved & v0 5-10 | segments 2..K | 547 / 2641 | 17.8 | RAW8-20k -0.1676 [-0.2424, -0.0789]; refcv7-50400 -0.2060 [-0.3211, -0.1205]; R7-20k -0.2237 [-0.3292, -0.1358] | RAW8-20k +0.0383 [-0.0556, +0.1375]; R7-20k -0.0177 [-0.0571, +0.0173] |
| sharp & v0 5-10 | last segment | 133 / 133 | 67.1 | RAW8-20k -0.2257 [-0.3179, -0.0984]; refcv7-50400 -0.2716 [-0.4102, -0.1612]; R7-20k -0.2705 [-0.4107, -0.1513] | RAW8-20k +0.0459 [-0.0788, +0.1772]; R7-20k +0.0010 [-0.0588, +0.0460] |
| sharp & v0 5-10 | segments 2..K | 139 / 686 | 43.8 | RAW8-20k -0.1810 [-0.2636, -0.0811]; refcv7-50400 -0.2110 [-0.3442, -0.1138]; R7-20k -0.2351 [-0.3579, -0.1293] | RAW8-20k +0.0300 [-0.0745, +0.1392]; R7-20k -0.0241 [-0.0768, +0.0180] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW8-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3515 | 0.4041 | 0.12155 | 0.18184 | 0.611 / 0.389 / 0.000 | 1120.6 / 1420.5 |
| turn|straight <0.02 | 2095 | 0.4451 | 0.13415 | 0.18953 | 0.646 / 0.354 / 0.000 | 486.8 / 932.5 |
| turn|gentle 0.02-0.06 | 677 | 0.3171 | 0.11022 | 0.13062 | 0.592 / 0.408 / 0.000 | 120.6 / 214.7 |
| turn|turn 0.06-0.15 | 427 | 0.1319 | 0.08632 | 0.11646 | 0.539 / 0.461 / 0.000 | 65.8 / 56.3 |
| turn|sharp >=0.15 | 302 | 0.6991 | 0.02401 | 0.00547 | 0.507 / 0.493 / 0.000 | 200.9 / 211.1 |
| turn|undef (GT crawling) | 14 | 0.4238 | 0.08474 | 0.21504 | 0.786 / 0.214 / 0.000 | 2.7 / 5.9 |
| speed|v0 <2 | 177 | 0.7088 | 0.47523 | 0.44641 | 0.689 / 0.311 / 0.000 | 42.1 / 125.5 |
| speed|v0 2-5 | 422 | 0.8449 | 0.23317 | 0.33007 | 0.635 / 0.365 / 0.000 | 250.5 / 356.6 |
| speed|v0 5-10 | 1040 | 0.5707 | 0.11511 | 0.16139 | 0.586 / 0.414 / 0.000 | 489.7 / 593.5 |
| speed|v0 10-15 | 952 | 0.1739 | 0.15298 | 0.19605 | 0.648 / 0.352 / 0.000 | 68.9 / 165.6 |
| speed|v0 15-25 | 679 | 0.2695 | 0.11726 | 0.13273 | 0.629 / 0.371 / 0.000 | 50.1 / 183.0 |
| speed|v0 >=25 | 245 | -0.0146 | -0.02222 | -0.02337 | 0.429 / 0.571 / 0.000 | 3.9 / -3.6 |
| nav|nav=follow | 2245 | 0.2765 | 0.09342 | 0.13327 | 0.599 / 0.401 / 0.000 | 341.1 / 620.7 |
| nav|nav=left | 321 | 0.5266 | 0.60531 | 0.62271 | 0.726 / 0.274 / 0.000 | 83.3 / 169.0 |
| nav|nav=right | 949 | 0.6647 | 0.11508 | 0.15828 | 0.601 / 0.399 / 0.000 | 522.8 / 630.8 |
**RAW8-20k - R7-20k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3502 | -0.0536 | 0.00072 | -0.05034 | 0.501 / 0.499 / 0.000 | 1047.6 / -187.8 |
| turn|straight <0.02 | 2083 | 0.2161 | 0.05381 | 0.06588 | 0.554 / 0.446 / 0.000 | 408.0 / 450.2 |
| turn|gentle 0.02-0.06 | 677 | 0.2151 | -0.03822 | -0.04395 | 0.473 / 0.527 / 0.000 | 172.1 / 145.6 |
| turn|turn 0.06-0.15 | 427 | -0.6748 | -0.39044 | -0.61211 | 0.379 / 0.621 / 0.000 | 105.1 / -288.2 |
| turn|sharp >=0.15 | 302 | -1.6448 | -1.12055 | -1.57931 | 0.364 / 0.636 / 0.000 | 167.9 / -496.7 |
| turn|undef (GT crawling) | 13 | 0.0974 | -0.00396 | 0.03820 | 0.462 / 0.538 / 0.000 | 1.0 / 1.3 |
| speed|v0 <2 | 164 | -0.2725 | -0.19129 | -0.31629 | 0.415 / 0.585 / 0.000 | 41.4 / -44.7 |
| speed|v0 2-5 | 422 | -0.3029 | -0.21020 | -0.35819 | 0.400 / 0.600 / 0.000 | 179.9 / -127.8 |
| speed|v0 5-10 | 1040 | 0.0227 | -0.05232 | -0.10293 | 0.473 / 0.527 / 0.000 | 455.4 / 23.6 |
| speed|v0 10-15 | 952 | -0.1200 | 0.06198 | 0.03377 | 0.543 / 0.457 / 0.000 | 94.6 / -114.3 |
| speed|v0 15-25 | 679 | 0.0954 | 0.05856 | 0.06250 | 0.555 / 0.445 / 0.000 | 53.4 / 64.8 |
| speed|v0 >=25 | 245 | 0.0433 | 0.01179 | 0.01079 | 0.531 / 0.469 / 0.000 | 4.1 / 10.6 |
| nav|nav=follow | 2232 | -0.0691 | 0.00839 | -0.01861 | 0.510 / 0.490 / 0.000 | 321.9 / -154.3 |
| nav|nav=left | 321 | -0.0380 | 0.37927 | 0.33027 | 0.645 / 0.355 / 0.000 | 85.4 / -12.2 |
| nav|nav=right | 949 | -0.0224 | -0.15393 | -0.27173 | 0.429 / 0.571 / 0.000 | 458.2 / -21.3 |
**R7-20k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3553 | 0.4766 | 0.10028 | 0.20807 | 0.620 / 0.380 / 0.000 | 855.2 / 1693.4 |
| turn|straight <0.02 | 2099 | 0.2379 | 0.07444 | 0.12359 | 0.627 / 0.373 / 0.000 | 182.9 / 499.3 |
| turn|gentle 0.02-0.06 | 679 | 0.1141 | 0.10517 | 0.15251 | 0.589 / 0.411 / 0.000 | 53.7 / 77.5 |
| turn|turn 0.06-0.15 | 442 | 0.8340 | 0.43138 | 0.64440 | 0.631 / 0.369 / 0.000 | 106.9 / 368.6 |
| turn|sharp >=0.15 | 312 | 2.4102 | 0.75762 | 1.57327 | 0.625 / 0.375 / 0.000 | 176.8 / 752.0 |
| turn|undef (GT crawling) | 21 | -0.1909 | 0.01568 | 0.03616 | 0.524 / 0.476 / 0.000 | 1.0 / -4.0 |
| speed|v0 <2 | 215 | 1.1869 | 0.64327 | 0.82983 | 0.712 / 0.288 / 0.000 | 54.3 / 255.2 |
| speed|v0 2-5 | 422 | 1.1479 | 0.55297 | 0.70166 | 0.732 / 0.268 / 0.000 | 215.5 / 484.4 |
| speed|v0 5-10 | 1040 | 0.5480 | 0.16613 | 0.23231 | 0.627 / 0.373 / 0.000 | 287.0 / 569.9 |
| speed|v0 10-15 | 952 | 0.2940 | 0.08256 | 0.13515 | 0.633 / 0.367 / 0.000 | 111.5 / 279.8 |
| speed|v0 15-25 | 679 | 0.1741 | 0.02646 | 0.04804 | 0.558 / 0.442 / 0.000 | 40.0 / 118.2 |
| speed|v0 >=25 | 245 | -0.0580 | -0.03136 | -0.03594 | 0.433 / 0.567 / 0.000 | 2.1 / -14.2 |
| nav|nav=follow | 2240 | 0.3448 | 0.07247 | 0.13962 | 0.604 / 0.396 / 0.000 | 362.0 / 772.4 |
| nav|nav=left | 334 | 0.6373 | 0.08693 | 0.19686 | 0.608 / 0.392 / 0.000 | 117.6 / 212.9 |
| nav|nav=right | 979 | 0.7233 | 0.22736 | 0.42770 | 0.659 / 0.341 / 0.000 | 300.7 / 708.1 |

### S6 -- the episodes carrying the summed delta (panel file index only)

**RAW8-20k - refcv7-50400** (sum 1420.5 over 139 episodes)

* ep027: n 26, share 0.146, mean delta 7.9625, sharp-window frac 0.00, curved frac 0.00, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep059: n 27, share 0.145, mean delta 7.6407, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.102, mean delta 5.5498, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.091, mean delta 4.7659, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep100: n 23, share 0.069, mean delta 4.2365, sharp-window frac 0.39, curved frac 0.91, mean v0 4.1, nav {'nav=follow': 0, 'nav=left': 23, 'nav=right': 0}
* ep137: n 22, share 0.062, mean delta 3.9894, sharp-window frac 0.36, curved frac 1.00, mean v0 4.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 22}
**RAW8-20k - R7-20k** (sum -187.8 over 139 episodes)

* ep059: n 27, share -0.977, mean delta 6.7972, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep027: n 26, share -0.890, mean delta 6.4266, sharp-window frac 0.00, curved frac 0.00, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep137: n 22, share -0.800, mean delta 6.8344, sharp-window frac 0.36, curved frac 1.00, mean v0 4.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 22}
* ep078: n 26, share -0.767, mean delta 5.5393, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep061: n 26, share -0.505, mean delta 3.6495, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep029: n 26, share -0.226, mean delta 1.6307, sharp-window frac 0.00, curved frac 1.00, mean v0 12.1, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**R7-20k - refcv7-50400** (sum 1693.4 over 139 episodes)

* ep085: n 26, share 0.058, mean delta 3.7941, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep100: n 26, share 0.056, mean delta 3.6358, sharp-window frac 0.35, curved frac 0.85, mean v0 3.6, nav {'nav=follow': 0, 'nav=left': 26, 'nav=right': 0}
* ep013: n 27, share 0.055, mean delta 3.4561, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep041: n 26, share 0.054, mean delta 3.5224, sharp-window frac 0.12, curved frac 0.38, mean v0 8.6, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.051, mean delta 3.3460, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep067: n 26, share 0.045, mean delta 2.9568, sharp-window frac 0.35, curved frac 0.96, mean v0 4.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW8-20k vs R7-20k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1325 / -0.0041 [-0.0763, 0.0618] | 45 / 0.2428 [-0.5246, 0.8879] |
| only new agrees | 278 / -1.7714 [-2.7557, -1.0192] | 33 / -10.2747 [-16.7826, -5.5776] |
| only earlier agrees | 563 / 0.7470 [0.3343, 1.2953] | 29 / 7.6619 [1.2082, 15.1702] |
| neither agrees | 1340 / -0.0904 [-0.5416, 0.4209] | 195 / -1.7318 [-3.0545, -0.4585] |
**RAW8-20k vs R7-20k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1321 / 0.0016 [-0.0709, 0.0661] | 47 / 0.4106 [-0.2826, 1.0779] |
| only new agrees | 272 / -1.4797 [-2.1681, -0.8664] | 30 / -7.2675 [-11.0085, -4.3603] |
| only earlier agrees | 562 / 0.9504 [0.5029, 1.5207] | 27 / 7.5628 [2.3264, 13.5195] |
| neither agrees | 1352 / -0.2335 [-0.7600, 0.3301] | 199 / -2.7735 [-4.7302, -0.9156] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW8-20k vs R7-20k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3477 | 0.0469 [-0.2582, 0.3458] | 0.2188 [-0.0458, 0.5550] | -0.1537 [-0.5018, 0.1576] |
| turn|straight <0.02 | 2069 | -0.1601 [-0.4404, 0.0274] | -0.1226 [-0.2141, -0.0514] | -0.0475 [-0.2730, 0.1141] |
| turn|gentle 0.02-0.06 | 677 | -0.4004 [-1.2250, 0.0840] | -0.2909 [-0.6428, 0.0101] | -0.1304 [-0.6783, 0.2296] |
| turn|turn 0.06-0.15 | 424 | -0.0389 [-0.5838, 0.4654] | -0.1136 [-0.5918, 0.3094] | 0.1756 [-0.3804, 0.7031] |
| turn|sharp >=0.15 | 299 | 2.6197 [0.5338, 4.7880] | 4.1074 [1.6804, 7.1485] | -1.4196 [-4.8706, 1.3283] |
| turn|undef (GT crawling) | 8 | 0.1148 [-0.6235, 1.0893] | -0.3843 [-0.8422, 0.1214] | 0.2957 [-0.4551, 1.4040] |
| speed|v0 <2 | 139 | 0.3916 [-0.4097, 1.6615] | -0.0013 [-0.8575, 0.8521] | 0.9929 [0.0583, 2.3971] SEP |
| speed|v0 2-5 | 422 | -0.5269 [-2.0408, 0.7121] | 0.5277 [-0.6093, 1.9391] | -1.0547 [-2.6570, 0.1826] |
| speed|v0 5-10 | 1040 | 0.3621 [-0.3684, 1.0555] | 0.5521 [-0.0683, 1.4855] | -0.1900 [-1.1715, 0.5372] |
| speed|v0 10-15 | 952 | -0.0303 [-0.1448, 0.0816] | 0.0021 [-0.0834, 0.0813] | -0.0324 [-0.1612, 0.0872] |
| speed|v0 15-25 | 679 | -0.0388 [-0.1267, 0.0481] | -0.0380 [-0.1045, 0.0137] | -0.0008 [-0.0983, 0.1178] |
| speed|v0 >=25 | 245 | 0.0244 [-0.0047, 0.0532] | 0.0175 [-0.0140, 0.0529] | 0.0070 [-0.0213, 0.0451] |
**RAW8-20k vs R7-20k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* RAW8-20k vs R7-20k|seed0: anchor flip rate 0.411; heading delta on flipped windows -0.0036 [-0.5129, 0.5700] (n 1428) vs unflipped -0.0931 [-0.1707, -0.0197]; goal_dist_sel proxy new-earlier -0.3827 [-0.6682, -0.0806]; corr(window dHeading, window dGoalDist) -0.010
* RAW8-20k vs R7-20k|seed1: anchor flip rate 0.414; heading delta on flipped windows 0.0091 [-0.5556, 0.7026] (n 1442) vs unflipped -0.0944 [-0.1724, -0.0267]; goal_dist_sel proxy new-earlier -0.3894 [-0.6756, -0.0857]; corr(window dHeading, window dGoalDist) -0.005
