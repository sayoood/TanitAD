# refcv8 heading-drift localisation: RAW8-15k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-19T18:22:42; trees: `RAW8-15k` = C:/Users/Admin/refcv8_eval_archive/battery/step15000_unc; `R7-15k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step15000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- RAW8-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4492 | 139 | 1.4592 | 1.3255 | 0.1338 [0.0754, 0.1982] | **SEP** |  |
| turn: straight <0.02 | 2894 | 132 | 1.2065 | 1.0989 | 0.1076 [0.0631, 0.1597] | **SEP** | 0.644 / 0.518 / -0.0735 [-0.3161, 0.1535] |
| turn: gentle 0.02-0.06 | 718 | 97 | 1.0637 | 1.0085 | 0.0551 [-0.0348, 0.1523] |  | 0.160 / 0.066 / -0.0936 [-0.2582, 0.0930] |
| turn: turn 0.06-0.15 | 501 | 72 | 1.8822 | 1.9056 | -0.0234 [-0.1863, 0.1212] |  | 0.112 / -0.020 / -0.1769 [-0.4107, 0.0363] |
| turn: sharp >=0.15 | 364 | 42 | 3.6977 | 2.9797 | 0.7180 [0.1908, 1.3679] | **SEP** | 0.081 / 0.435 / 0.6358 [-0.1422, 1.6425] |
| turn: undef (GT crawling) (low n) | 15 | 13 | 0.7026 | 0.6920 | 0.0105 [-0.0684, 0.0812] |  | 0.003 / 0.000 / -0.1236 [-0.2649, 0.0138] |
| speed: v0 <2 | 203 | 37 | 12.6644 | 12.4606 | 0.2038 [-0.0467, 0.4664] |  | 0.045 / 0.069 / 0.0734 [-0.3055, 0.4701] |
| speed: v0 2-5 | 516 | 58 | 2.0718 | 1.8694 | 0.2023 [-0.1275, 0.5500] |  | 0.115 / 0.174 / 0.0775 [-0.3851, 0.6417] |
| speed: v0 5-10 | 1390 | 84 | 1.2763 | 1.0092 | 0.2671 [0.1436, 0.4251] | **SEP** | 0.309 / 0.618 / 0.1931 [-0.0216, 0.4622] |
| speed: v0 10-15 | 1192 | 64 | 0.6313 | 0.5744 | 0.0569 [-0.0009, 0.1094] |  | 0.265 / 0.113 / -0.1046 [-0.2398, 0.0301] |
| speed: v0 15-25 | 856 | 35 | 0.3773 | 0.3734 | 0.0039 [-0.0395, 0.0475] |  | 0.191 / 0.006 / -0.1605 [-0.2818, -0.0414] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1954 | 0.1576 | 0.0377 [0.0104, 0.0686] | **SEP** | 0.075 / 0.021 / -0.1038 [-0.2168, 0.0016] |
| nav: nav=follow | 2874 | 88 | 0.7296 | 0.5989 | 0.1307 [0.0632, 0.2152] | **SEP** | 0.640 / 0.625 / -0.0083 [-0.1918, 0.2181] |
| nav: nav=left (low n) | 399 | 13 | 6.7135 | 6.4978 | 0.2157 [-0.0120, 0.4552] |  | 0.089 / 0.143 / 0.0899 [-0.2943, 0.4856] |
| nav: nav=right | 1219 | 38 | 1.4596 | 1.3456 | 0.1140 [-0.0055, 0.2380] |  | 0.271 / 0.231 / -0.0271 [-0.2562, 0.1917] |

episode concentration: 0.705 of episodes have mean delta > 0; top-5 episodes carry 0.359 of the summed delta; delta excluding the top 5 = 0.0890 (overall 0.1338).

### S2 -- RAW8-15k - R7-15k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4487 | 139 | 1.4605 | 1.4857 | -0.0252 [-0.1003, 0.0495] |  |  |
| turn: straight <0.02 | 2891 | 132 | 1.2075 | 1.1410 | 0.0666 [0.0176, 0.1224] | **SEP** | 0.644 / -1.702 / 0.2579 [0.0293, 0.4653] BONF-SEP |
| turn: gentle 0.02-0.06 | 718 | 97 | 1.0637 | 1.0404 | 0.0232 [-0.0782, 0.1353] |  | 0.160 / -0.147 / 0.0576 [-0.1051, 0.2548] |
| turn: turn 0.06-0.15 | 501 | 72 | 1.8822 | 2.2426 | -0.3604 [-0.5100, -0.2292] | **SEP** | 0.112 / 1.597 / -0.3773 [-0.5935, -0.1620] BONF-SEP |
| turn: sharp >=0.15 | 364 | 42 | 3.6977 | 4.0965 | -0.3988 [-0.9331, 0.2141] |  | 0.081 / 1.284 / -0.4066 [-1.1736, 0.4301] |
| turn: undef (GT crawling) (low n) | 13 | 11 | 0.7514 | 0.4756 | 0.2758 [-0.0685, 0.8211] |  | 0.003 / -0.032 / 0.3018 [-0.1458, 1.3216] |
| speed: v0 <2 | 198 | 37 | 12.9768 | 13.0367 | -0.0599 [-0.2876, 0.1349] |  | 0.044 / 0.105 / -0.0363 [-0.4019, 0.2853] |
| speed: v0 2-5 | 516 | 58 | 2.0718 | 2.1713 | -0.0995 [-0.4793, 0.3100] |  | 0.115 / 0.454 / -0.0840 [-0.6575, 0.5666] |
| speed: v0 5-10 | 1390 | 84 | 1.2763 | 1.2623 | 0.0140 [-0.1108, 0.1452] |  | 0.310 / -0.172 / 0.0568 [-0.1218, 0.2575] |
| speed: v0 10-15 | 1192 | 64 | 0.6313 | 0.6836 | -0.0523 [-0.1257, 0.0171] |  | 0.266 / 0.551 / -0.0369 [-0.1888, 0.1181] |
| speed: v0 15-25 | 856 | 35 | 0.3773 | 0.3947 | -0.0174 [-0.0649, 0.0299] |  | 0.191 / 0.132 / 0.0096 [-0.1380, 0.1764] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1954 | 0.1716 | 0.0237 [-0.0089, 0.0573] |  | 0.075 / -0.070 / 0.0529 [-0.0856, 0.1838] |
| nav: nav=follow | 2869 | 88 | 0.7304 | 0.7030 | 0.0274 [-0.0343, 0.1015] |  | 0.639 / -0.694 / 0.1457 [-0.0993, 0.4154] |
| nav: nav=left (low n) | 399 | 13 | 6.7135 | 6.7352 | -0.0217 [-0.5288, 0.4308] |  | 0.089 / 0.077 / 0.0038 [-0.9174, 0.7125] |
| nav: nav=right | 1219 | 38 | 1.4596 | 1.6096 | -0.1500 [-0.2944, -0.0017] | **SEP** | 0.272 / 1.618 / -0.1714 [-0.4100, 0.0712] |

episode concentration: 0.439 of episodes have mean delta > 0; top-5 episodes carry -2.010 of the summed delta; delta excluding the top 5 = -0.0787 (overall -0.0252).

### S2 -- R7-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4526 | 139 | 1.6096 | 1.4461 | 0.1635 [0.1035, 0.2261] | **SEP** |  |
| turn: straight <0.02 | 2901 | 132 | 1.1998 | 1.1585 | 0.0412 [0.0093, 0.0729] | **SEP** | 0.641 / 0.162 / -0.3405 [-0.5623, -0.1322] BONF-SEP |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0470 | 1.0095 | 0.0374 [-0.0610, 0.1252] |  | 0.159 / 0.036 / -0.1499 [-0.3216, -0.0073] BONF-SEP |
| turn: turn 0.06-0.15 | 512 | 72 | 2.2671 | 1.9319 | 0.3352 [0.2047, 0.4766] | **SEP** | 0.113 / 0.232 / 0.1936 [0.0196, 0.4046] BONF-SEP |
| turn: sharp >=0.15 | 367 | 42 | 4.1379 | 2.9870 | 1.1509 [0.6812, 1.6913] | **SEP** | 0.081 / 0.571 / 1.0745 [0.3241, 1.8708] BONF-SEP |
| turn: undef (GT crawling) (low n) | 25 | 20 | 14.8139 | 14.8398 | -0.0259 [-0.3863, 0.2094] |  | 0.006 / -0.001 / -0.1904 [-0.6872, 0.1378] |
| speed: v0 <2 | 237 | 38 | 13.5014 | 13.1669 | 0.3346 [0.1393, 0.5755] | **SEP** | 0.052 / 0.107 / 0.1805 [-0.1326, 0.5912] |
| speed: v0 2-5 | 516 | 58 | 2.1713 | 1.8694 | 0.3018 [-0.0861, 0.6533] |  | 0.114 / 0.210 / 0.1562 [-0.4184, 0.7190] |
| speed: v0 5-10 | 1390 | 84 | 1.2623 | 1.0092 | 0.2531 [0.1492, 0.3660] | **SEP** | 0.307 / 0.476 / 0.1294 [-0.0527, 0.3202] |
| speed: v0 10-15 | 1192 | 64 | 0.6836 | 0.5744 | 0.1092 [0.0467, 0.1829] | **SEP** | 0.263 / 0.176 / -0.0737 [-0.2228, 0.0870] |
| speed: v0 15-25 | 856 | 35 | 0.3947 | 0.3734 | 0.0213 [-0.0072, 0.0524] |  | 0.189 / 0.025 / -0.1754 [-0.2949, -0.0602] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1716 | 0.1576 | 0.0140 [0.0030, 0.0260] | **SEP** | 0.074 / 0.006 / -0.1614 [-0.2556, -0.0630] BONF-SEP |
| nav: nav=follow | 2877 | 88 | 0.7017 | 0.5980 | 0.1037 [0.0560, 0.1544] | **SEP** | 0.636 / 0.403 / -0.1642 [-0.4024, 0.0271] |
| nav: nav=left (low n) | 412 | 13 | 7.4995 | 7.2301 | 0.2694 [-0.0227, 0.6493] |  | 0.091 / 0.150 / 0.1165 [-0.3220, 0.8226] |
| nav: nav=right | 1237 | 38 | 1.7595 | 1.4922 | 0.2673 [0.0938, 0.4149] | **SEP** | 0.273 / 0.447 / 0.1429 [-0.1121, 0.3899] |

episode concentration: 0.755 of episodes have mean delta > 0; top-5 episodes carry 0.304 of the summed delta; delta excluding the top 5 = 0.1181 (overall 0.1635).

### S2 -- RAW8-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4458 | 139 | 0.7454 | 0.5885 | 0.1568 [0.0205, 0.4109] | **SEP** | 0.4015 [0.0202, 1.1475] SEP |
| seg1 | straight | 2874 | 132 | 0.7910 | 0.5706 | 0.2204 [0.0212, 0.6434] | **SEP** | 0.5733 [-0.0041, 1.8308] |
| seg1 | curved | 1576 | 101 | 0.6629 | 0.6217 | 0.0413 [0.0004, 0.0820] | **SEP** | 0.0909 [0.0324, 0.1522] SEP |
| seg2 | all | 4444 | 139 | 1.2392 | 0.9369 | 0.3023 [0.0595, 0.7391] | **SEP** | 0.3224 [-0.0090, 0.9204] |
| seg2 | straight | 2870 | 132 | 1.0627 | 0.6589 | 0.4038 [0.0597, 1.1062] | **SEP** | 0.4130 [-0.0645, 1.3980] |
| seg2 | curved | 1574 | 101 | 1.5611 | 1.4438 | 0.1174 [0.0026, 0.2359] | **SEP** | 0.1572 [-0.0005, 0.3227] |
| seg3 | all | 4433 | 139 | 1.5085 | 1.1996 | 0.3089 [0.0841, 0.6854] | **SEP** | 0.2834 [-0.0261, 0.8028] |
| seg3 | straight | 2864 | 132 | 1.1356 | 0.7751 | 0.3604 [0.0700, 0.9330] | **SEP** | 0.3574 [-0.0632, 1.2115] |
| seg3 | curved | 1569 | 101 | 2.1892 | 1.9744 | 0.2148 [0.0384, 0.4146] | **SEP** | 0.1484 [-0.0799, 0.3904] |
| seg4 | all | 4419 | 138 | 1.8569 | 1.4430 | 0.4139 [0.1215, 0.9251] | **SEP** | 0.5027 [0.2228, 0.9822] SEP |
| seg4 | straight | 2848 | 131 | 1.2493 | 0.7769 | 0.4724 [0.1084, 1.1796] | **SEP** | 0.5361 [0.1708, 1.2341] SEP |
| seg4 | curved | 1564 | 100 | 2.9678 | 2.6587 | 0.3091 [0.0457, 0.6091] | **SEP** | 0.4442 [0.0983, 0.8226] SEP |

### S2 -- RAW8-15k - R7-15k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4457 | 139 | 0.7455 | 0.6200 | 0.1256 [-0.0515, 0.4611] |  | 0.1289 [-0.0587, 0.4705] |
| seg1 | straight | 2874 | 132 | 0.7910 | 0.5446 | 0.2464 [-0.0179, 0.8050] |  | 0.1851 [-0.0850, 0.7472] |
| seg1 | curved | 1576 | 101 | 0.6629 | 0.7576 | -0.0946 [-0.1365, -0.0509] | **SEP** | 0.0278 [-0.0399, 0.1035] |
| seg2 | all | 4439 | 139 | 1.2397 | 1.0872 | 0.1525 [-0.0970, 0.5847] |  | 0.1137 [-0.1573, 0.5590] |
| seg2 | straight | 2866 | 131 | 1.0632 | 0.7148 | 0.3484 [-0.0013, 1.0455] |  | 0.1469 [-0.2191, 0.8402] |
| seg2 | curved | 1573 | 101 | 1.5614 | 1.7656 | -0.2043 [-0.3301, -0.0764] | **SEP** | 0.0534 [-0.1212, 0.2427] |
| seg3 | all | 4424 | 139 | 1.5104 | 1.4297 | 0.0807 [-0.1177, 0.3593] |  | 0.0666 [-0.1471, 0.3472] |
| seg3 | straight | 2857 | 131 | 1.1371 | 0.8597 | 0.2774 [0.0458, 0.7152] | **SEP** | 0.0817 [-0.1672, 0.5159] |
| seg3 | curved | 1567 | 101 | 2.1910 | 2.4689 | -0.2779 [-0.4887, -0.0587] | **SEP** | 0.0390 [-0.2123, 0.2939] |
| seg4 | all | 4414 | 138 | 1.8585 | 1.7799 | 0.0787 [-0.0974, 0.3020] |  | 0.0593 [-0.0803, 0.1987] |
| seg4 | straight | 2845 | 131 | 1.2503 | 1.0165 | 0.2338 [0.0541, 0.5387] | **SEP** | 0.0440 [-0.0341, 0.1242] |
| seg4 | curved | 1563 | 100 | 2.9695 | 3.1750 | -0.2056 [-0.4727, 0.0740] |  | 0.0893 [-0.2755, 0.4473] |

### S2 -- R7-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4483 | 139 | 0.6582 | 0.6675 | -0.0093 [-0.1754, 0.0817] |  | 0.3106 [0.0532, 0.8035] SEP |
| seg1 | straight | 2881 | 132 | 0.5435 | 0.5693 | -0.0258 [-0.1671, 0.0460] |  | 0.3875 [0.0608, 1.0848] SEP |
| seg1 | curved | 1585 | 101 | 0.7572 | 0.6233 | 0.1339 [0.0994, 0.1699] | **SEP** | 0.0611 [0.0014, 0.1218] SEP |
| seg2 | all | 4471 | 139 | 1.0888 | 0.9387 | 0.1501 [0.1001, 0.1998] | **SEP** | 0.2064 [0.0806, 0.3816] SEP |
| seg2 | straight | 2882 | 131 | 0.7121 | 0.6572 | 0.0549 [0.0274, 0.0850] | **SEP** | 0.2655 [0.1103, 0.5489] SEP |
| seg2 | curved | 1589 | 101 | 1.7720 | 1.4492 | 0.3228 [0.2144, 0.4298] | **SEP** | 0.0993 [-0.0618, 0.2542] |
| seg3 | all | 4460 | 139 | 1.4370 | 1.2055 | 0.2316 [0.1177, 0.3674] | **SEP** | 0.2145 [0.0254, 0.4889] SEP |
| seg3 | straight | 2875 | 131 | 0.8564 | 0.7742 | 0.0822 [-0.0059, 0.2317] |  | 0.2766 [0.0514, 0.7008] SEP |
| seg3 | curved | 1585 | 101 | 2.4902 | 1.9877 | 0.5026 [0.3083, 0.7004] | **SEP** | 0.1019 [-0.1714, 0.3495] |
| seg4 | all | 4453 | 139 | 1.8374 | 1.4941 | 0.3433 [0.1339, 0.6482] | **SEP** | 0.4392 [0.1330, 0.8943] SEP |
| seg4 | straight | 2865 | 131 | 1.0765 | 0.8387 | 0.2377 [0.0139, 0.6664] | **SEP** | 0.4898 [0.1276, 1.1999] SEP |
| seg4 | curved | 1580 | 101 | 3.2230 | 2.6866 | 0.5364 [0.2775, 0.7963] | **SEP** | 0.3461 [-0.0859, 0.7443] |

### S2 -- RAW8-15k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 435 | 47 | 5.7820 | 5.6070 | 0.1749 [-0.0059, 0.4040] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.5017 | 0.3674 | 0.1343 [0.0779, 0.2001] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.2280 | 0.1947 | 0.0334 [0.0125, 0.0599] | **SEP** |  |
| gentle 0.02-0.06 | v0 <5 | 109 | 28 | 1.5828 | 1.7771 | -0.1943 [-0.5357, 0.1593] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.1064 | 0.9711 | 0.1353 [0.0300, 0.2458] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.5480 | 0.5590 | -0.0109 [-0.0752, 0.0408] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 71 | 25 | 5.0267 | 5.3596 | -0.3329 [-1.0862, 0.4315] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.4750 | 1.4275 | 0.0475 [-0.0790, 0.1828] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 0.9213 | 0.9719 | -0.0506 [-0.2342, 0.1861] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 89 | 16 | 6.5705 | 5.2858 | 1.2847 [0.1470, 2.6704] | **SEP** |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.8897 | 2.2857 | 0.6040 [0.0702, 1.3674] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 0.7990 | 1.3866 | -0.5876 [-0.8701, -0.0711] | **SEP** |  |
| undef (GT crawling) | v0 <5 (low n) | 15 | 13 | 0.7026 | 0.6920 | 0.0105 [-0.0684, 0.0812] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW8-15k s0-s1 [CI] | R7-15k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0060 [-0.0201, 0.0088] | 0.0102 [-0.0033, 0.0246] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0110 [-0.0305, 0.0106] | 0.0073 [0.0005, 0.0153] SEP | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0111 [-0.0331, 0.0502] | 0.0177 [-0.0188, 0.0628] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | -0.0149 [-0.0627, 0.0257] | 0.0313 [-0.0224, 0.0874] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | 0.0178 [-0.0676, 0.1126] | -0.0152 [-0.1389, 0.1082] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.1186 [-0.2692, 0.0150] | 0.0587 [-0.0663, 0.2319] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | -0.0344 [-0.0825, 0.0108] | 0.0943 [0.0353, 0.1870] SEP | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0099 [-0.1126, 0.1039] | 0.0373 [-0.0539, 0.1359] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0164 [-0.0496, 0.0152] | -0.0004 [-0.0255, 0.0239] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0026 [-0.0057, 0.0108] | 0.0077 [-0.0040, 0.0196] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | 0.0052 [-0.0022, 0.0127] | -0.0047 [-0.0124, 0.0024] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | 0.0020 [-0.0057, 0.0107] | -0.0024 [-0.0100, 0.0042] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0108 [-0.0232, -0.0001] SEP | 0.0077 [-0.0004, 0.0163] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0245 [-0.0222, 0.0720] | 0.0155 [-0.0735, 0.1327] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0046 [-0.0461, 0.0396] | 0.0142 [-0.0181, 0.0514] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- RAW8-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1338 | 0.1261 / 0.1409 | True / True / True | 0.0060 / 0.0110 | 12.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.1076 | 0.1012 / 0.1141 | True / True / True | 0.0110 / 0.0019 | 9.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0551 | 0.0674 / 0.0434 | False / False / False | 0.0111 / 0.0161 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0234 | -0.0403 / -0.0109 | False / False / False | 0.0149 / 0.0331 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.7180 | 0.6751 / 0.7609 | True / True / True | 0.0178 / 0.1080 | 6.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.0105 | -0.0100 / 0.0295 | False / False / False | 0.1186 / 0.0237 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.2038 | 0.1713 / 0.2238 | False / False / False | 0.0344 / 0.0573 | 3.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.2023 | 0.1614 / 0.2432 | False / False / False | 0.0099 / 0.0719 | 2.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2671 | 0.2586 / 0.2756 | True / True / True | 0.0164 / 0.0006 | 16.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0569 | 0.0582 / 0.0556 | False / False / False | 0.0026 / 0.0001 | 22.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0039 | 0.0076 / 0.0001 | False / False / False | 0.0052 / 0.0023 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0377 | 0.0390 / 0.0365 | True / True / True | 0.0020 / 0.0005 | 19.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.1307 | 0.1273 / 0.1343 | True / True / True | 0.0108 / 0.0036 | 12.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.2157 | 0.2033 / 0.2273 | True / False / False | 0.0245 / 0.0608 | 3.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1140 | 0.0981 / 0.1282 | False / True / False | 0.0046 / 0.0283 | 4.0 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- RAW8-15k - R7-15k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0252 | -0.0329 / -0.0178 | False / False / False | 0.0060 / 0.0102 | 2.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0666 | 0.0576 / 0.0756 | True / True / True | 0.0110 / 0.0073 | 6.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0232 | 0.0217 / 0.0255 | False / False / False | 0.0111 / 0.0177 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.3604 | -0.3813 / -0.3433 | True / True / True | 0.0149 / 0.0313 | 11.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -0.3988 | -0.3850 / -0.4125 | False / False / False | 0.0178 / 0.0152 | 22.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.2758 | 0.0819 / 0.4069 | False / False / False | 0.1186 / 0.0587 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0599 | -0.1258 / -0.0014 | False / False / False | 0.0344 / 0.0943 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.0995 | -0.1231 / -0.0759 | False / False / False | 0.0099 / 0.0373 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0140 | 0.0060 / 0.0220 | False / False / False | 0.0164 / 0.0004 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.0523 | -0.0548 / -0.0497 | False / False / False | 0.0026 / 0.0077 | 6.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0174 | -0.0125 / -0.0223 | False / False / False | 0.0052 / 0.0047 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0237 | 0.0259 / 0.0215 | False / False / False | 0.0020 / 0.0024 | 9.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0274 | 0.0184 / 0.0365 | False / False / False | 0.0108 / 0.0077 | 2.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.0217 | -0.0118 / -0.0314 | False / False / False | 0.0245 / 0.0155 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.1500 | -0.1605 / -0.1412 | True / False / True | 0.0046 / 0.0142 | 10.6 | CLEARS 2x floor + pooled-separated, not separated at each seed |

### S2 -- R7-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1635 | 0.1624 / 0.1652 | True / True / True | 0.0102 / 0.0110 | 14.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0412 | 0.0439 / 0.0385 | True / True / True | 0.0073 / 0.0019 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0374 | 0.0542 / 0.0206 | False / False / False | 0.0177 / 0.0161 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.3352 | 0.3288 / 0.3410 | True / True / True | 0.0313 / 0.0331 | 10.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 1.1509 | 1.0893 / 1.2161 | True / True / True | 0.0152 / 0.1080 | 10.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.0259 | 0.0077 / -0.0316 | False / False / False | 0.0587 / 0.0237 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.3346 | 0.3402 / 0.3369 | True / True / True | 0.0943 / 0.0573 | 3.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 0.3018 | 0.2845 / 0.3191 | False / False / False | 0.0373 / 0.0719 | 4.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2531 | 0.2526 / 0.2536 | True / True / True | 0.0004 / 0.0006 | 390.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1092 | 0.1131 / 0.1053 | True / True / True | 0.0077 / 0.0001 | 14.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.0213 | 0.0201 / 0.0225 | False / False / False | 0.0047 / 0.0023 | 4.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0140 | 0.0130 / 0.0150 | True / True / True | 0.0024 / 0.0005 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.1037 | 0.1093 / 0.0980 | True / True / True | 0.0077 / 0.0036 | 13.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.2694 | 0.2392 / 0.2978 | False / True / False | 0.0155 / 0.0608 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2673 | 0.2602 / 0.2767 | True / True / True | 0.0142 / 0.0283 | 9.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1576): RAW8-15k bias -0.075, refcv7-50400 bias +0.087, R7-15k bias +0.040; RAW8-15k MAE 0.663, refcv7-50400 MAE 0.622, R7-15k MAE 0.758
* seg2 (n 1573): RAW8-15k bias -0.479, refcv7-50400 bias -0.077, R7-15k bias -0.314; RAW8-15k MAE 1.561, refcv7-50400 MAE 1.444, R7-15k MAE 1.766
* seg3 (n 1567): RAW8-15k bias -1.144, refcv7-50400 bias -0.596, R7-15k bias -1.229; RAW8-15k MAE 2.191, refcv7-50400 MAE 1.976, R7-15k MAE 2.469
* seg4 (n 1563): RAW8-15k bias -1.621, refcv7-50400 bias -0.683, R7-15k bias -1.810; RAW8-15k MAE 2.969, refcv7-50400 MAE 2.660, R7-15k MAE 3.175
**sharp (>= 0.15)**

* seg1 (n 363): RAW8-15k bias -0.522, refcv7-50400 bias +0.013, R7-15k bias -0.122; RAW8-15k MAE 1.006, refcv7-50400 MAE 0.916, R7-15k MAE 1.147
* seg2 (n 363): RAW8-15k bias -1.825, refcv7-50400 bias -0.508, R7-15k bias -1.184; RAW8-15k MAE 2.789, refcv7-50400 MAE 2.408, R7-15k MAE 3.122
* seg3 (n 362): RAW8-15k bias -3.620, refcv7-50400 bias -1.834, R7-15k bias -3.702; RAW8-15k MAE 4.520, refcv7-50400 MAE 3.635, R7-15k MAE 5.185
* seg4 (n 361): RAW8-15k bias -4.843, refcv7-50400 bias -1.793, R7-15k bias -5.191; RAW8-15k MAE 6.266, refcv7-50400 MAE 4.859, R7-15k MAE 6.815
**curved & v0 5-10**

* seg1 (n 627): RAW8-15k bias -0.220, refcv7-50400 bias -0.038, R7-15k bias -0.095; RAW8-15k MAE 0.612, refcv7-50400 MAE 0.535, R7-15k MAE 0.735
* seg2 (n 627): RAW8-15k bias -0.769, refcv7-50400 bias -0.282, R7-15k bias -0.534; RAW8-15k MAE 1.621, refcv7-50400 MAE 1.363, R7-15k MAE 1.843
* seg3 (n 627): RAW8-15k bias -1.557, refcv7-50400 bias -0.871, R7-15k bias -1.477; RAW8-15k MAE 2.474, refcv7-50400 MAE 2.019, R7-15k MAE 2.653
* seg4 (n 627): RAW8-15k bias -2.250, refcv7-50400 bias -1.194, R7-15k bias -2.210; RAW8-15k MAE 3.335, refcv7-50400 MAE 2.730, R7-15k MAE 3.352
**sharp & v0 5-10**

* seg1 (n 164): RAW8-15k bias -0.444, refcv7-50400 bias +0.103, R7-15k bias +0.093; RAW8-15k MAE 0.932, refcv7-50400 MAE 0.857, R7-15k MAE 1.156
* seg2 (n 164): RAW8-15k bias -1.852, refcv7-50400 bias -0.422, R7-15k bias -0.744; RAW8-15k MAE 2.760, refcv7-50400 MAE 2.243, R7-15k MAE 3.001
* seg3 (n 164): RAW8-15k bias -4.042, refcv7-50400 bias -2.032, R7-15k bias -3.460; RAW8-15k MAE 4.768, refcv7-50400 MAE 3.465, R7-15k MAE 4.827
* seg4 (n 164): RAW8-15k bias -5.613, refcv7-50400 bias -2.403, R7-15k bias -5.287; RAW8-15k MAE 6.707, refcv7-50400 MAE 4.635, R7-15k MAE 6.586

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1563 / 1563 | 10.2 | RAW8-15k -0.1986 [-0.2888, -0.1252]; refcv7-50400 -0.0823 [-0.1530, -0.0266]; R7-15k -0.2151 [-0.2983, -0.1472] | RAW8-15k -0.1163 [-0.1618, -0.0814] SEP; R7-15k -0.1327 [-0.1742, -0.0923] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1576 / 4703 | 7.2 | RAW8-15k -0.2495 [-0.3656, -0.1477]; refcv7-50400 -0.1450 [-0.2634, -0.0512]; R7-15k -0.2583 [-0.3736, -0.1608] | RAW8-15k -0.1044 [-0.1449, -0.0719] SEP; R7-15k -0.1132 [-0.1535, -0.0758] SEP |
| sharp (>= 0.15) | last segment | 361 / 361 | 25.4 | RAW8-15k -0.2181 [-0.3243, -0.1328]; refcv7-50400 -0.0910 [-0.1740, -0.0241]; R7-15k -0.2333 [-0.3311, -0.1544] | RAW8-15k -0.1271 [-0.1812, -0.0881] SEP; R7-15k -0.1423 [-0.1916, -0.0983] SEP |
| sharp (>= 0.15) | segments 2..K | 363 / 1086 | 17.9 | RAW8-15k -0.2235 [-0.3283, -0.1392]; refcv7-50400 -0.1030 [-0.1864, -0.0349]; R7-15k -0.2289 [-0.3295, -0.1472] | RAW8-15k -0.1205 [-0.1710, -0.0836] SEP; R7-15k -0.1258 [-0.1727, -0.0839] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | RAW8-15k -0.2132 [-0.2990, -0.1420]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-15k -0.2026 [-0.2773, -0.1468] | RAW8-15k -0.1207 [-0.2022, -0.0553] SEP; R7-15k -0.1100 [-0.1454, -0.0722] SEP |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | RAW8-15k -0.2166 [-0.2998, -0.1476]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-15k -0.2000 [-0.2767, -0.1423] | RAW8-15k -0.1173 [-0.1949, -0.0574] SEP; R7-15k -0.1008 [-0.1343, -0.0660] SEP |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | RAW8-15k -0.2175 [-0.3153, -0.1372]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-15k -0.2004 [-0.2813, -0.1383] | RAW8-15k -0.1263 [-0.2180, -0.0577] SEP; R7-15k -0.1091 [-0.1504, -0.0674] SEP |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | RAW8-15k -0.2173 [-0.3111, -0.1400]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-15k -0.1889 [-0.2718, -0.1251] | RAW8-15k -0.1218 [-0.2115, -0.0569] SEP; R7-15k -0.0934 [-0.1317, -0.0545] SEP |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW8-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4492 | 0.1338 | 0.02957 | 0.04790 | 0.581 / 0.419 / 0.000 | 369.1 / 600.8 |
| turn|straight <0.02 | 2894 | 0.1076 | 0.03137 | 0.04728 | 0.616 / 0.384 / 0.000 | 171.2 / 311.5 |
| turn|gentle 0.02-0.06 | 718 | 0.0551 | 0.00754 | 0.01794 | 0.517 / 0.483 / 0.000 | 38.3 / 39.6 |
| turn|turn 0.06-0.15 | 501 | -0.0234 | -0.01649 | 0.00640 | 0.485 / 0.515 / 0.000 | 26.1 / -11.7 |
| turn|sharp >=0.15 | 364 | 0.7180 | 0.11906 | 0.36507 | 0.563 / 0.437 / 0.000 | 71.0 / 261.4 |
| turn|undef (GT crawling) | 15 | 0.0105 | -0.01666 | -0.00020 | 0.467 / 0.533 / 0.000 | 0.3 / 0.2 |
| speed|v0 <2 | 203 | 0.2038 | 0.00901 | 0.02620 | 0.522 / 0.478 / 0.000 | 15.5 / 41.4 |
| speed|v0 2-5 | 516 | 0.2023 | 0.05428 | 0.08562 | 0.566 / 0.434 / 0.000 | 73.0 / 104.4 |
| speed|v0 5-10 | 1390 | 0.2671 | 0.04817 | 0.09189 | 0.583 / 0.417 / 0.000 | 133.1 / 371.3 |
| speed|v0 10-15 | 1192 | 0.0569 | 0.02467 | 0.04446 | 0.591 / 0.409 / 0.000 | 30.9 / 67.8 |
| speed|v0 15-25 | 856 | 0.0039 | 0.01476 | 0.01864 | 0.546 / 0.454 / 0.000 | 9.3 / 3.3 |
| speed|v0 >=25 | 335 | 0.0377 | 0.03559 | 0.03868 | 0.681 / 0.319 / 0.000 | 1.8 / 12.6 |
| nav|nav=follow | 2874 | 0.1307 | 0.02342 | 0.03400 | 0.577 / 0.423 / 0.000 | 221.8 / 375.8 |
| nav|nav=left | 399 | 0.2157 | 0.08465 | 0.11610 | 0.629 / 0.371 / 0.000 | 25.1 / 86.1 |
| nav|nav=right | 1219 | 0.1140 | 0.04265 | 0.06297 | 0.574 / 0.426 / 0.000 | 112.1 / 139.0 |
**RAW8-15k - R7-15k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4487 | -0.0252 | -0.01584 | -0.03213 | 0.464 / 0.536 / 0.000 | 356.2 / -113.0 |
| turn|straight <0.02 | 2891 | 0.0666 | 0.00329 | 0.00952 | 0.512 / 0.488 / 0.000 | 185.6 / 192.4 |
| turn|gentle 0.02-0.06 | 718 | 0.0232 | -0.05600 | -0.05078 | 0.416 / 0.584 / 0.000 | 52.4 / 16.7 |
| turn|turn 0.06-0.15 | 501 | -0.3604 | -0.22556 | -0.28708 | 0.325 / 0.675 / 0.000 | 16.0 / -180.5 |
| turn|sharp >=0.15 | 364 | -0.3988 | -0.32055 | -0.41512 | 0.365 / 0.635 / 0.000 | 64.2 / -145.1 |
| turn|undef (GT crawling) | 13 | 0.2758 | -0.02062 | 0.04321 | 0.462 / 0.538 / 0.000 | 2.8 / 3.6 |
| speed|v0 <2 | 198 | -0.0599 | -0.03800 | -0.07091 | 0.424 / 0.576 / 0.000 | 8.2 / -11.9 |
| speed|v0 2-5 | 516 | -0.0995 | -0.10218 | -0.14385 | 0.415 / 0.585 / 0.000 | 76.8 / -51.4 |
| speed|v0 5-10 | 1390 | 0.0140 | -0.03664 | -0.03696 | 0.444 / 0.556 / 0.000 | 116.3 / 19.4 |
| speed|v0 10-15 | 1192 | -0.0523 | -0.01437 | -0.02692 | 0.456 / 0.544 / 0.000 | 37.1 / -62.3 |
| speed|v0 15-25 | 856 | -0.0174 | -0.00633 | -0.00423 | 0.480 / 0.520 / 0.000 | 11.6 / -14.9 |
| speed|v0 >=25 | 335 | 0.0237 | 0.02641 | 0.02699 | 0.636 / 0.364 / 0.000 | 1.8 / 8.0 |
| nav|nav=follow | 2869 | 0.0274 | -0.01147 | -0.02177 | 0.472 / 0.528 / 0.000 | 203.7 / 78.5 |
| nav|nav=left | 399 | -0.0217 | 0.03163 | 0.06870 | 0.544 / 0.456 / 0.000 | 37.0 / -8.7 |
| nav|nav=right | 1219 | -0.1500 | -0.05754 | -0.10789 | 0.419 / 0.581 / 0.000 | 105.2 / -182.9 |
**R7-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4526 | 0.1635 | 0.03533 | 0.07819 | 0.615 / 0.385 / 0.000 | 338.4 / 739.9 |
| turn|straight <0.02 | 2901 | 0.0412 | 0.02299 | 0.03728 | 0.609 / 0.391 / 0.000 | 56.8 / 119.6 |
| turn|gentle 0.02-0.06 | 721 | 0.0374 | 0.05040 | 0.06975 | 0.583 / 0.417 / 0.000 | 24.3 / 27.0 |
| turn|turn 0.06-0.15 | 512 | 0.3352 | 0.18964 | 0.26054 | 0.641 / 0.359 / 0.000 | 29.8 / 171.6 |
| turn|sharp >=0.15 | 367 | 1.1509 | 0.59272 | 0.88909 | 0.695 / 0.305 / 0.000 | 70.1 / 422.4 |
| turn|undef (GT crawling) | 25 | -0.0259 | 0.03725 | 0.02737 | 0.520 / 0.480 / 0.000 | 1.9 / -0.6 |
| speed|v0 <2 | 237 | 0.3346 | 0.02631 | 0.11500 | 0.574 / 0.426 / 0.000 | 15.6 / 79.3 |
| speed|v0 2-5 | 516 | 0.3018 | 0.18655 | 0.24091 | 0.655 / 0.345 / 0.000 | 97.8 / 155.7 |
| speed|v0 5-10 | 1390 | 0.2531 | 0.06923 | 0.12250 | 0.618 / 0.382 / 0.000 | 88.2 / 351.8 |
| speed|v0 10-15 | 1192 | 0.1092 | 0.04512 | 0.06968 | 0.655 / 0.345 / 0.000 | 32.2 / 130.1 |
| speed|v0 15-25 | 856 | 0.0213 | 0.01273 | 0.01539 | 0.564 / 0.436 / 0.000 | 11.6 / 18.2 |
| speed|v0 >=25 | 335 | 0.0140 | 0.00753 | 0.00874 | 0.552 / 0.448 / 0.000 | 3.0 / 4.7 |
| nav|nav=follow | 2877 | 0.1037 | 0.02413 | 0.05110 | 0.597 / 0.403 / 0.000 | 128.0 / 298.3 |
| nav|nav=left | 412 | 0.2694 | 0.02697 | 0.07506 | 0.597 / 0.403 / 0.000 | 50.8 / 111.0 |
| nav|nav=right | 1237 | 0.2673 | 0.08259 | 0.16826 | 0.662 / 0.338 / 0.000 | 126.6 / 330.7 |

### S2 -- the episodes carrying the summed delta (panel file index only)

**RAW8-15k - refcv7-50400** (sum 600.8 over 139 episodes)

* ep111: n 34, share 0.132, mean delta 2.3369, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep108: n 23, share 0.061, mean delta 1.5901, sharp-window frac 0.00, curved frac 0.57, mean v0 9.3, nav {'nav=follow': 23, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share 0.061, mean delta 1.0409, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep104: n 35, share 0.054, mean delta 0.9351, sharp-window frac 0.29, curved frac 0.31, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 35, 'nav=right': 0}
* ep013: n 34, share 0.050, mean delta 0.8909, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep059: n 34, share 0.048, mean delta 0.8562, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**RAW8-15k - R7-15k** (sum -113.0 over 139 episodes)

* ep104: n 35, share -0.488, mean delta 1.5753, sharp-window frac 0.29, curved frac 0.31, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 35, 'nav=right': 0}
* ep111: n 34, share -0.479, mean delta 1.5922, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep108: n 23, share -0.387, mean delta 1.9028, sharp-window frac 0.00, curved frac 0.57, mean v0 9.3, nav {'nav=follow': 23, 'nav=left': 0, 'nav=right': 0}
* ep078: n 34, share -0.337, mean delta 1.1220, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep024: n 35, share -0.319, mean delta 1.0296, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep048: n 34, share -0.248, mean delta 0.8236, sharp-window frac 0.00, curved frac 0.53, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 34, 'nav=right': 0}
**R7-15k - refcv7-50400** (sum 739.9 over 139 episodes)

* ep016: n 29, share 0.089, mean delta 2.2669, sharp-window frac 0.28, curved frac 0.34, mean v0 7.7, nav {'nav=follow': 0, 'nav=left': 29, 'nav=right': 0}
* ep067: n 34, share 0.062, mean delta 1.3404, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep028: n 34, share 0.059, mean delta 1.2862, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep061: n 34, share 0.049, mean delta 1.0613, sharp-window frac 0.62, curved frac 1.00, mean v0 9.5, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.045, mean delta 0.9865, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep124: n 34, share 0.044, mean delta 0.9502, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW8-15k vs R7-15k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1321 / -0.0508 [-0.0759, -0.0263] | 42 / -0.2881 [-0.6208, 0.0225] |
| only new agrees | 321 / -0.3510 [-0.5359, -0.1796] | 48 / -1.5462 [-2.4193, -0.8463] |
| only earlier agrees | 1140 / 0.1302 [0.0486, 0.2256] | 47 / 0.6964 [-0.2475, 1.9042] |
| neither agrees | 1708 / -0.0682 [-0.2261, 0.0858] | 227 / -0.3813 [-1.1504, 0.4578] |
**RAW8-15k vs R7-15k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1325 / -0.0644 [-0.0909, -0.0397] | 45 / -0.4617 [-0.8437, -0.1389] |
| only new agrees | 332 / -0.2495 [-0.4129, -0.0996] | 53 / -1.4132 [-2.2826, -0.7696] |
| only earlier agrees | 1150 / 0.1320 [0.0444, 0.2285] | 49 / 0.5270 [-0.6486, 1.7572] |
| neither agrees | 1684 / -0.0379 [-0.2013, 0.1228] | 217 / -0.3701 [-1.1509, 0.4529] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW8-15k vs R7-15k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4461 | 0.2283 [0.1311, 0.3406] | 0.0165 [-0.0143, 0.0507] | 0.2173 [0.1178, 0.3310] SEP |
| turn|straight <0.02 | 2873 | 0.1693 [0.0899, 0.2621] | -0.0310 [-0.0482, -0.0166] | 0.2002 [0.1157, 0.2961] SEP |
| turn|gentle 0.02-0.06 | 717 | 0.2463 [0.0419, 0.4810] | -0.0661 [-0.1205, -0.0179] | 0.3117 [0.0927, 0.5584] SEP |
| turn|turn 0.06-0.15 | 499 | 0.2718 [0.0733, 0.5328] | -0.0382 [-0.1221, 0.0391] | 0.3309 [0.1369, 0.5830] SEP |
| turn|sharp >=0.15 | 363 | 0.6012 [0.0601, 1.2641] | 0.6337 [0.3031, 1.0562] | 0.0057 [-0.6057, 0.6323] |
| turn|undef (GT crawling) | 9 | 0.1819 [-0.0526, 0.3707] | -0.0275 [-0.2339, 0.1080] | 0.4035 [0.0247, 0.8324] SEP |
| speed|v0 <2 | 172 | 0.0457 [-0.4019, 0.5788] | 0.1199 [-0.0475, 0.3007] | 0.0228 [-0.5836, 0.6500] |
| speed|v0 2-5 | 516 | 1.1304 [0.6125, 1.6597] | 0.0502 [-0.1799, 0.3512] | 1.0802 [0.4984, 1.6544] SEP |
| speed|v0 5-10 | 1390 | 0.3218 [0.1762, 0.4912] | 0.0120 [-0.0514, 0.0726] | 0.3099 [0.1744, 0.4724] SEP |
| speed|v0 10-15 | 1192 | -0.0089 [-0.0330, 0.0141] | 0.0005 [-0.0233, 0.0265] | -0.0094 [-0.0431, 0.0221] |
| speed|v0 15-25 | 856 | -0.0108 [-0.0384, 0.0103] | -0.0016 [-0.0112, 0.0088] | -0.0092 [-0.0396, 0.0162] |
| speed|v0 >=25 | 335 | 0.0013 [-0.0158, 0.0152] | 0.0117 [0.0006, 0.0224] | -0.0104 [-0.0353, 0.0087] |
**RAW8-15k vs R7-15k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* RAW8-15k vs R7-15k|seed0: anchor flip rate 0.530; heading delta on flipped windows -0.0003 [-0.1318, 0.1306] (n 2379) vs unflipped -0.0697 [-0.1022, -0.0412]; goal_dist_sel proxy new-earlier -0.3068 [-0.6893, 0.1047]; corr(window dHeading, window dGoalDist) -0.002
* RAW8-15k vs R7-15k|seed1: anchor flip rate 0.532; heading delta on flipped windows 0.0330 [-0.0951, 0.1608] (n 2387) vs unflipped -0.0755 [-0.1072, -0.0480]; goal_dist_sel proxy new-earlier -0.3165 [-0.6967, 0.0962]; corr(window dHeading, window dGoalDist) -0.006

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- RAW8-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3504 | 139 | 3.6876 | 3.1801 | 0.5075 [0.2938, 0.7293] | **SEP** |  |
| turn: straight <0.02 | 2092 | 122 | 1.9011 | 1.3768 | 0.5243 [0.3676, 0.7101] | **SEP** | 0.597 / 0.617 / 0.0415 [-0.7589, 0.6550] |
| turn: gentle 0.02-0.06 | 674 | 77 | 3.2657 | 2.8914 | 0.3742 [0.0961, 0.6833] | **SEP** | 0.192 / 0.142 / -0.1650 [-0.7032, 0.3491] |
| turn: turn 0.06-0.15 | 426 | 59 | 6.2446 | 6.3476 | -0.1030 [-0.5427, 0.2623] |  | 0.122 / -0.025 / -0.6951 [-1.4213, -0.1706] BONF-SEP |
| turn: sharp >=0.15 | 301 | 33 | 12.9515 | 11.3886 | 1.5629 [-0.1441, 3.5856] |  | 0.086 / 0.265 / 1.1546 [-1.1745, 4.5503] |
| turn: undef (GT crawling) (low n) | 11 | 8 | 16.7926 | 16.5344 | 0.2581 [-0.0814, 0.8073] |  | 0.003 / 0.002 / -0.2502 [-0.9465, 0.9205] |
| speed: v0 <2 | 167 | 25 | 14.1706 | 13.6692 | 0.5014 [-0.6872, 1.5752] |  | 0.048 / 0.047 / -0.0064 [-2.1321, 1.7182] |
| speed: v0 2-5 | 421 | 52 | 5.2156 | 4.1566 | 1.0590 [0.2860, 1.7950] | **SEP** | 0.120 / 0.251 / 0.6268 [-0.5818, 1.9103] |
| speed: v0 5-10 | 1040 | 79 | 4.3652 | 3.4941 | 0.8711 [0.3899, 1.5012] | **SEP** | 0.297 / 0.509 / 0.5171 [-0.1789, 1.5789] |
| speed: v0 10-15 | 952 | 62 | 2.7466 | 2.4870 | 0.2595 [0.0746, 0.4642] | **SEP** | 0.272 / 0.139 / -0.3405 [-0.8738, 0.1545] |
| speed: v0 15-25 | 679 | 34 | 1.5909 | 1.4456 | 0.1453 [0.0027, 0.2854] | **SEP** | 0.194 / 0.055 / -0.4493 [-0.8705, -0.0496] BONF-SEP |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5072 | 0.5193 | -0.0121 [-0.1545, 0.1260] |  | 0.070 / -0.002 / -0.5586 [-1.0154, -0.1905] BONF-SEP |
| nav: nav=follow | 2240 | 88 | 2.5320 | 1.9826 | 0.5494 [0.3329, 0.8423] | **SEP** | 0.639 / 0.692 / 0.1161 [-0.5823, 0.8506] |
| nav: nav=left (low n) | 315 | 13 | 9.6389 | 9.1689 | 0.4700 [-0.5846, 1.4510] |  | 0.090 / 0.083 / -0.0413 [-1.9897, 1.5991] |
| nav: nav=right | 949 | 38 | 4.4398 | 4.0187 | 0.4211 [0.0441, 0.8417] | **SEP** | 0.271 / 0.225 / -0.1185 [-0.8433, 0.6799] |

episode concentration: 0.770 of episodes have mean delta > 0; top-5 episodes carry 0.365 of the summed delta; delta excluding the top 5 = 0.3352 (overall 0.5075).

### S6 -- RAW8-15k - R7-15k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3499 | 139 | 3.6900 | 3.6778 | 0.0122 [-0.2528, 0.2666] |  |  |
| turn: straight <0.02 | 2087 | 122 | 1.9007 | 1.5381 | 0.3626 [0.2245, 0.5242] | **SEP** | 0.596 / 17.758 / 0.8684 [0.0322, 1.7562] BONF-SEP |
| turn: gentle 0.02-0.06 | 674 | 77 | 3.2657 | 2.7624 | 0.5033 [0.0874, 1.0522] | **SEP** | 0.193 / 7.960 / 0.6083 [-0.0622, 1.7116] |
| turn: turn 0.06-0.15 | 426 | 59 | 6.2446 | 6.8979 | -0.6533 [-1.1767, -0.1295] | **SEP** | 0.122 / -6.531 / -0.7577 [-1.6200, 0.0754] |
| turn: sharp >=0.15 | 301 | 33 | 12.9515 | 15.5360 | -2.5845 [-4.7514, -0.6436] | **SEP** | 0.086 / -18.255 / -2.8411 [-6.0459, 0.1987] |
| turn: undef (GT crawling) (low n) | 11 | 8 | 16.7926 | 16.5302 | 0.2623 [-0.0721, 0.7551] |  | 0.003 / 0.068 / 0.2509 [-0.3975, 1.4755] |
| speed: v0 <2 | 162 | 25 | 14.5454 | 14.3798 | 0.1656 [-0.8199, 1.0917] |  | 0.046 / 0.629 / 0.1608 [-1.4410, 1.5353] |
| speed: v0 2-5 | 421 | 52 | 5.2156 | 5.0083 | 0.2074 [-1.1985, 1.3403] |  | 0.120 / 2.049 / 0.2219 [-2.0600, 2.0028] |
| speed: v0 5-10 | 1040 | 79 | 4.3652 | 4.1750 | 0.1903 [-0.2395, 0.6637] |  | 0.297 / 4.644 / 0.2534 [-0.4180, 1.0893] |
| speed: v0 10-15 | 952 | 62 | 2.7466 | 2.9686 | -0.2221 [-0.5532, 0.0612] |  | 0.272 / -4.961 / -0.3218 [-0.9886, 0.3064] |
| speed: v0 15-25 | 679 | 34 | 1.5909 | 1.6750 | -0.0841 [-0.2246, 0.0403] |  | 0.194 / -1.340 / -0.1194 [-0.6335, 0.4196] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5072 | 0.5110 | -0.0038 [-0.1494, 0.1410] |  | 0.070 / -0.022 / -0.0171 [-0.4793, 0.5024] |
| nav: nav=follow | 2235 | 88 | 2.5332 | 2.4647 | 0.0684 [-0.1585, 0.2947] |  | 0.639 / 3.590 / 0.1558 [-0.7672, 1.2694] |
| nav: nav=left (low n) | 315 | 13 | 9.6389 | 9.9450 | -0.3062 [-2.3136, 1.1724] |  | 0.090 / -2.263 / -0.3498 [-4.3431, 1.7697] |
| nav: nav=right | 949 | 38 | 4.4398 | 4.4545 | -0.0147 [-0.5748, 0.5940] |  | 0.271 / -0.327 / -0.0368 [-1.0232, 0.9570] |

episode concentration: 0.547 of episodes have mean delta > 0; top-5 episodes carry 12.877 of the summed delta; delta excluding the top 5 = -0.1500 (overall 0.0122).

### S6 -- R7-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3566 | 139 | 3.8393 | 3.2893 | 0.5500 [0.3022, 0.8206] | **SEP** |  |
| turn: straight <0.02 | 2112 | 123 | 1.6340 | 1.4638 | 0.1703 [0.1143, 0.2353] | **SEP** | 0.592 / 0.183 / -0.9314 [-1.7834, -0.0650] BONF-SEP |
| turn: gentle 0.02-0.06 | 678 | 77 | 2.7762 | 2.8833 | -0.1070 [-0.6270, 0.2487] |  | 0.190 / -0.037 / -0.8113 [-1.7698, -0.2045] BONF-SEP |
| turn: turn 0.06-0.15 | 442 | 59 | 6.9775 | 6.3377 | 0.6398 [0.0153, 1.2417] | **SEP** | 0.124 / 0.144 / 0.1025 [-0.8572, 1.0610] |
| turn: sharp >=0.15 | 312 | 33 | 16.2410 | 11.7318 | 4.5092 [2.6898, 6.5514] | **SEP** | 0.087 / 0.717 / 4.3387 [1.9625, 7.8780] BONF-SEP |
| turn: undef (GT crawling) (low n) | 22 | 13 | 9.3763 | 10.0709 | -0.6945 [-1.6720, 0.1677] |  | 0.006 / -0.008 / -1.2523 [-2.8231, -0.0823] BONF-SEP |
| speed: v0 <2 | 228 | 26 | 13.8187 | 12.5782 | 1.2405 [0.3495, 2.8822] | **SEP** | 0.064 / 0.144 / 0.7376 [-0.4697, 3.4350] |
| speed: v0 2-5 | 422 | 52 | 4.9992 | 4.1501 | 0.8490 [-0.2946, 2.0662] |  | 0.118 / 0.183 / 0.3391 [-1.5476, 2.3402] |
| speed: v0 5-10 | 1040 | 79 | 4.1750 | 3.4941 | 0.6809 [0.3172, 1.0586] | **SEP** | 0.292 / 0.361 / 0.1847 [-0.4058, 0.8490] |
| speed: v0 10-15 | 952 | 62 | 2.9686 | 2.4870 | 0.4816 [0.1433, 0.8711] | **SEP** | 0.267 / 0.234 / -0.0934 [-0.7067, 0.6017] |
| speed: v0 15-25 | 679 | 34 | 1.6750 | 1.4456 | 0.2294 [0.1145, 0.3645] | **SEP** | 0.190 / 0.079 / -0.3960 [-0.8928, 0.0924] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5110 | 0.5193 | -0.0083 [-0.0721, 0.0607] |  | 0.069 / -0.001 / -0.5995 [-1.0604, -0.2118] BONF-SEP |
| nav: nav=follow | 2248 | 88 | 2.4554 | 1.9748 | 0.4806 [0.2623, 0.7260] | **SEP** | 0.630 / 0.551 / -0.1878 [-1.1747, 0.5980] |
| nav: nav=left (low n) | 335 | 13 | 10.9818 | 9.7883 | 1.1935 [0.1320, 3.0265] | **SEP** | 0.094 / 0.204 / 0.7101 [-0.6100, 4.2706] |
| nav: nav=right | 983 | 38 | 4.5700 | 4.0805 | 0.4895 [-0.0955, 1.0138] |  | 0.276 / 0.245 / -0.0836 [-1.0731, 0.8410] |

episode concentration: 0.748 of episodes have mean delta > 0; top-5 episodes carry 0.414 of the summed delta; delta excluding the top 5 = 0.3340 (overall 0.5500).

### S6 -- RAW8-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3473 | 139 | 1.0839 | 0.7626 | 0.3213 [0.0305, 0.8648] | **SEP** | 0.4062 [0.0077, 1.1523] SEP |
| seg1 | straight | 2064 | 122 | 0.8644 | 0.5442 | 0.3202 [0.0345, 0.9039] | **SEP** | 0.4169 [-0.0486, 1.3875] |
| seg1 | curved | 1398 | 88 | 1.2845 | 0.9613 | 0.3232 [-0.0039, 0.9257] |  | 0.3964 [0.0629, 0.9409] SEP |
| seg2 | all | 3457 | 139 | 1.6810 | 1.3997 | 0.2813 [0.0875, 0.5544] | **SEP** | 0.5034 [0.1099, 1.1568] SEP |
| seg2 | straight | 2062 | 122 | 1.0575 | 0.7543 | 0.3032 [0.0801, 0.7131] | **SEP** | 0.4244 [0.0184, 1.2073] SEP |
| seg2 | curved | 1395 | 88 | 2.6026 | 2.3537 | 0.2489 [0.0100, 0.5004] | **SEP** | 0.6202 [0.1795, 1.1773] SEP |
| seg3 | all | 3419 | 138 | 2.4302 | 1.9347 | 0.4955 [0.2476, 0.8192] | **SEP** | 0.7216 [0.4690, 1.0312] SEP |
| seg3 | straight | 2042 | 121 | 1.4280 | 0.8908 | 0.5372 [0.2745, 0.9856] | **SEP** | 0.6856 [0.4240, 1.1187] SEP |
| seg3 | curved | 1377 | 87 | 3.9165 | 3.4829 | 0.4336 [0.0136, 0.9494] | **SEP** | 0.7750 [0.2502, 1.3128] SEP |
| seg4 | all | 3382 | 138 | 3.5876 | 2.8823 | 0.7053 [0.4074, 1.0448] | **SEP** | 0.8361 [0.3291, 1.2727] SEP |
| seg4 | straight | 2029 | 119 | 1.8094 | 1.0506 | 0.7588 [0.5199, 1.0584] | **SEP** | 0.7022 [0.0420, 1.1110] SEP |
| seg4 | curved | 1353 | 86 | 6.2543 | 5.6293 | 0.6250 [0.0393, 1.3094] | **SEP** | 1.0368 [0.2148, 1.7949] SEP |
| seg5 | all | 3338 | 138 | 5.1482 | 4.2171 | 0.9311 [0.5373, 1.3849] | **SEP** | 1.0704 [0.5888, 1.5370] SEP |
| seg5 | straight | 2006 | 119 | 2.3040 | 1.2205 | 1.0835 [0.7038, 1.5768] | **SEP** | 1.1240 [0.6125, 1.5109] SEP |
| seg5 | curved | 1332 | 87 | 9.4315 | 8.7299 | 0.7016 [-0.0224, 1.5476] |  | 0.9896 [0.0045, 1.9154] SEP |
| seg6 | all | 3308 | 138 | 6.8092 | 5.8791 | 0.9301 [0.5231, 1.3679] | **SEP** | 0.9666 [0.4301, 1.5009] SEP |
| seg6 | straight | 1996 | 119 | 2.6108 | 1.6373 | 0.9735 [0.6278, 1.3582] | **SEP** | 1.4033 [1.0301, 1.7859] SEP |
| seg6 | curved | 1312 | 87 | 13.1963 | 12.3322 | 0.8641 [0.0973, 1.7721] | **SEP** | 0.3021 [-0.8630, 1.3971] |

### S6 -- RAW8-15k - R7-15k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3470 | 139 | 1.0847 | 0.9173 | 0.1674 [-0.0746, 0.6024] |  | 0.1520 [-0.1078, 0.5913] |
| seg1 | straight | 2062 | 122 | 0.8652 | 0.6866 | 0.1786 [-0.0219, 0.5682] |  | 0.0410 [-0.1706, 0.4331] |
| seg1 | curved | 1397 | 88 | 1.2852 | 1.1351 | 0.1500 [-0.1810, 0.7248] |  | 0.3205 [-0.0376, 0.9041] |
| seg2 | all | 3450 | 139 | 1.6833 | 1.5924 | 0.0910 [-0.1080, 0.3432] |  | 0.1395 [-0.0718, 0.3937] |
| seg2 | straight | 2055 | 121 | 1.0593 | 0.9080 | 0.1513 [0.0169, 0.3681] | **SEP** | 0.0086 [-0.1479, 0.2266] |
| seg2 | curved | 1395 | 88 | 2.6026 | 2.6006 | 0.0020 [-0.3322, 0.3964] |  | 0.3322 [-0.0523, 0.7248] |
| seg3 | all | 3414 | 138 | 2.4329 | 2.2800 | 0.1529 [-0.0943, 0.4606] |  | 0.0036 [-0.3481, 0.2903] |
| seg3 | straight | 2037 | 121 | 1.4299 | 1.0710 | 0.3589 [0.1133, 0.7999] | **SEP** | -0.0254 [-0.4407, 0.2326] |
| seg3 | curved | 1377 | 87 | 3.9165 | 4.0684 | -0.1519 [-0.5817, 0.3157] |  | 0.0464 [-0.5212, 0.6002] |
| seg4 | all | 3373 | 138 | 3.5939 | 3.5212 | 0.0728 [-0.2356, 0.4088] |  | -0.0492 [-0.7875, 0.4748] |
| seg4 | straight | 2021 | 119 | 1.8112 | 1.3139 | 0.4973 [0.2936, 0.7678] | **SEP** | 0.0038 [-1.0345, 0.5549] |
| seg4 | curved | 1352 | 86 | 6.2589 | 6.8207 | -0.5618 [-1.2350, 0.1512] |  | -0.1284 [-0.9416, 0.6270] |
| seg5 | all | 3336 | 138 | 5.1496 | 4.9406 | 0.2090 [-0.3271, 0.8558] |  | 0.0805 [-0.8516, 0.7894] |
| seg5 | straight | 2003 | 119 | 2.3028 | 1.2380 | 1.0648 [0.5340, 1.9397] | **SEP** | 0.3213 [-0.9482, 1.0559] |
| seg5 | curved | 1333 | 87 | 9.4273 | 10.5042 | -1.0769 [-2.0089, -0.1253] | **SEP** | -0.2814 [-1.4631, 0.8553] |
| seg6 | all | 3302 | 138 | 6.8136 | 6.7447 | 0.0689 [-0.4771, 0.6613] |  | 0.2871 [-0.4076, 0.9387] |
| seg6 | straight | 1993 | 119 | 2.6061 | 1.6087 | 0.9974 [0.5533, 1.5422] | **SEP** | 0.9264 [0.3336, 1.4199] SEP |
| seg6 | curved | 1309 | 87 | 13.2197 | 14.5645 | -1.3448 [-2.3985, -0.2249] | **SEP** | -0.6862 [-2.1243, 0.7251] |

### S6 -- R7-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3496 | 139 | 0.9170 | 0.8149 | 0.1022 [0.0638, 0.1394] | **SEP** | 0.3045 [0.0662, 0.7287] SEP |
| seg1 | straight | 2071 | 122 | 0.6849 | 0.5435 | 0.1413 [0.0414, 0.3401] | **SEP** | 0.3754 [0.0941, 0.9549] SEP |
| seg1 | curved | 1407 | 88 | 1.1386 | 1.0929 | 0.0458 [-0.2240, 0.2248] |  | 0.2041 [0.0046, 0.4934] SEP |
| seg2 | all | 3483 | 139 | 1.6601 | 1.4614 | 0.1988 [0.0816, 0.3187] | **SEP** | 0.3599 [0.0521, 0.8084] SEP |
| seg2 | straight | 2073 | 121 | 0.9908 | 0.8390 | 0.1518 [0.0365, 0.3605] | **SEP** | 0.4137 [0.1128, 1.0026] SEP |
| seg2 | curved | 1410 | 88 | 2.6441 | 2.3764 | 0.2678 [-0.0656, 0.5807] |  | 0.2807 [-0.1114, 0.6728] |
| seg3 | all | 3465 | 139 | 2.3871 | 2.0678 | 0.3193 [0.0865, 0.5461] | **SEP** | 0.7654 [0.2995, 1.3814] SEP |
| seg3 | straight | 2061 | 122 | 1.1558 | 0.9749 | 0.1809 [0.1169, 0.2481] | **SEP** | 0.7075 [0.2995, 1.4833] SEP |
| seg3 | curved | 1404 | 88 | 4.1946 | 3.6721 | 0.5225 [-0.0499, 1.0894] |  | 0.8504 [0.1569, 1.5063] SEP |
| seg4 | all | 3439 | 139 | 3.6267 | 2.9850 | 0.6417 [0.3033, 0.9828] | **SEP** | 0.9080 [0.4299, 1.4149] SEP |
| seg4 | straight | 2050 | 122 | 1.3147 | 1.1340 | 0.1808 [-0.0505, 0.3330] |  | 0.7837 [0.4447, 1.3805] SEP |
| seg4 | curved | 1389 | 87 | 7.0390 | 5.7169 | 1.3221 [0.5397, 2.1416] | **SEP** | 1.0914 [0.0637, 2.1003] SEP |
| seg5 | all | 3412 | 139 | 5.1587 | 4.3337 | 0.8250 [0.3927, 1.2782] | **SEP** | 1.0672 [0.3385, 1.9196] SEP |
| seg5 | straight | 2035 | 122 | 1.3487 | 1.3111 | 0.0376 [-0.3929, 0.2876] |  | 0.9575 [0.3426, 2.1567] SEP |
| seg5 | curved | 1377 | 88 | 10.7894 | 8.8006 | 1.9888 [1.0115, 2.9559] | **SEP** | 1.2293 [-0.1088, 2.5020] |
| seg6 | all | 3380 | 139 | 6.9595 | 5.9834 | 0.9761 [0.4910, 1.4814] | **SEP** | 0.7965 [0.1103, 1.5188] SEP |
| seg6 | straight | 2026 | 122 | 1.7329 | 1.7347 | -0.0017 [-0.4553, 0.2679] |  | 0.6433 [0.1603, 1.4938] SEP |
| seg6 | curved | 1350 | 88 | 14.8095 | 12.3511 | 2.4583 [1.4295, 3.5040] | **SEP** | 1.0169 [-0.3976, 2.4642] |

### S6 -- RAW8-15k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 358 | 39 | 6.0909 | 4.7515 | 1.3393 [0.9449, 1.7694] | **SEP** |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 1.3139 | 0.7632 | 0.5506 [0.3906, 0.7344] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.6095 | 0.5523 | 0.0572 [-0.0098, 0.1238] |  |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 97 | 22 | 5.9784 | 5.5654 | 0.4130 [-0.6740, 1.8625] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.8894 | 2.5310 | 0.3584 [-0.0004, 0.8090] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.5887 | 2.1952 | 0.3935 [0.0187, 0.7838] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 55 | 19 | 10.9394 | 11.9842 | -1.0448 [-2.7325, 0.7201] |  |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 5.8212 | 5.8272 | -0.0059 [-0.3997, 0.3627] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 4.5244 | 4.3281 | 0.1963 [-0.3831, 0.7607] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 67 | 16 | 15.1562 | 14.1908 | 0.9653 [-2.5503, 4.9972] |  |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 12.6414 | 10.7541 | 1.8874 [-0.0493, 4.3262] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 4.2907 | 6.3909 | -2.1002 [-3.4079, 1.8658] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 11 | 8 | 16.7926 | 16.5344 | 0.2581 [-0.0814, 0.8073] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | RAW8-15k s0-s1 [CI] | R7-15k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | 0.0249 [-0.0363, 0.0991] | 0.0385 [-0.0243, 0.1139] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0156 [-0.0464, 0.0131] | -0.0089 [-0.0410, 0.0275] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0576 [-0.0955, 0.2915] | 0.0091 [-0.0537, 0.0850] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | 0.0292 [-0.1440, 0.2320] | 0.0410 [-0.0646, 0.1414] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.2264 [-0.1064, 0.6622] | 0.4196 [-0.0921, 1.0546] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.0419 [-0.0728, 0.1863] | 0.0169 [-0.2314, 0.3191] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | 0.6156 [-0.0537, 1.8145] | 0.5111 [-0.0175, 1.3551] | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | 0.1901 [-0.0045, 0.4643] | 0.2492 [0.0195, 0.5701] SEP | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | -0.0779 [-0.2361, 0.0598] | -0.0596 [-0.1349, 0.0154] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0155 [-0.0658, 0.0366] | -0.0267 [-0.0712, 0.0117] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | 0.0015 [-0.0467, 0.0469] | 0.0013 [-0.0456, 0.0457] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | -0.0026 [-0.0295, 0.0199] | -0.0103 [-0.0282, 0.0092] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0308 [-0.0652, 0.0006] | -0.0124 [-0.0413, 0.0165] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.3547 [-0.0812, 1.1057] | 0.4812 [0.0017, 1.1772] SEP | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | 0.0471 [-0.0703, 0.1675] | 0.0029 [-0.0995, 0.1193] | 0.0235 [-0.1175, 0.1775] |

### S6 -- RAW8-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.5075 | 0.5117 / 0.5068 | True / True / True | 0.0249 / 0.0128 | 20.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.5243 | 0.5297 / 0.5197 | True / True / True | 0.0156 / 0.0216 | 24.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.3742 | 0.3667 / 0.3868 | True / True / True | 0.0576 / 0.0713 | 5.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.1030 | -0.1062 / -0.0829 | False / False / False | 0.0292 / 0.0611 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 1.5629 | 1.5905 / 1.5354 | False / False / False | 0.2264 / 0.2004 | 6.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.2581 | 0.3921 / 0.1713 | True / False / False | 0.0419 / 1.9278 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.5014 | 0.7458 / 0.3304 | False / False / False | 0.6156 / 0.0328 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 1.0590 | 1.0534 / 1.0647 | True / True / True | 0.1901 / 0.2009 | 5.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | 0.8711 | 0.8443 / 0.8980 | True / True / True | 0.0779 / 0.0243 | 11.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.2595 | 0.2652 / 0.2539 | True / True / True | 0.0155 / 0.0268 | 9.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.1453 | 0.1433 / 0.1473 | False / True / True | 0.0015 / 0.0055 | 26.6 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 >=25 | -0.0121 | -0.0138 / -0.0103 | False / False / False | 0.0026 / 0.0008 | 4.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.5494 | 0.5295 / 0.5702 | True / True / True | 0.0308 / 0.0080 | 17.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.4700 | 0.6592 / 0.2797 | False / False / False | 0.3547 / 0.0140 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.4211 | 0.4205 / 0.4328 | True / True / True | 0.0471 / 0.0235 | 8.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- RAW8-15k - R7-15k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0122 | 0.0205 / 0.0052 | False / False / False | 0.0249 / 0.0385 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.3626 | 0.3673 / 0.3588 | True / True / True | 0.0156 / 0.0089 | 23.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.5033 | 0.5313 / 0.4725 | True / True / True | 0.0576 / 0.0091 | 8.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.6533 | -0.6438 / -0.6530 | True / True / True | 0.0292 / 0.0410 | 16.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -2.5845 | -2.5900 / -2.5791 | True / True / True | 0.2264 / 0.4196 | 6.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.2623 | 0.0552 / 0.4622 | False / True / False | 0.0419 / 0.0169 | 6.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1656 | 0.4378 / -0.0811 | False / False / False | 0.6156 / 0.5111 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.2074 | 0.1775 / 0.2372 | False / False / False | 0.1901 / 0.2492 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1903 | 0.1811 / 0.1994 | False / False / False | 0.0779 / 0.0596 | 2.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.2221 | -0.2164 / -0.2277 | False / False / False | 0.0155 / 0.0267 | 8.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0841 | -0.0840 / -0.0842 | False / False / False | 0.0015 / 0.0013 | 57.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0038 | 0.0001 / -0.0076 | False / False / False | 0.0026 / 0.0103 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0684 | 0.0591 / 0.0787 | False / False / False | 0.0308 / 0.0124 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.3062 | -0.2120 / -0.3981 | False / False / False | 0.3547 / 0.4812 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.0147 | 0.0071 / -0.0342 | False / False / False | 0.0471 / 0.0029 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- R7-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.5500 | 0.5546 / 0.5416 | True / True / True | 0.0385 / 0.0128 | 14.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.1703 | 0.1769 / 0.1636 | True / True / True | 0.0089 / 0.0216 | 7.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.1070 | -0.1385 / -0.0843 | False / False / False | 0.0091 / 0.0713 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.6398 | 0.6298 / 0.6499 | True / True / True | 0.0410 / 0.0611 | 10.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 4.5092 | 4.5279 / 4.4560 | True / True / True | 0.4196 / 0.2004 | 10.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | -0.6945 | 0.3228 / -1.7119 | False / False / False | 0.0169 / 1.9278 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 1.2405 | 1.3603 / 1.0532 | True / True / True | 0.5111 / 0.0328 | 2.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 0.8490 | 0.8732 / 0.8249 | False / False / False | 0.2492 / 0.2009 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.6809 | 0.6632 / 0.6985 | True / True / True | 0.0596 / 0.0243 | 11.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.4816 | 0.4816 / 0.4815 | True / True / True | 0.0267 / 0.0268 | 18.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.2294 | 0.2273 / 0.2315 | True / True / True | 0.0013 / 0.0055 | 42.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0083 | -0.0138 / -0.0027 | False / False / False | 0.0103 / 0.0008 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.4806 | 0.4693 / 0.4917 | True / True / True | 0.0124 / 0.0080 | 38.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 1.1935 | 1.3424 / 1.0016 | True / True / True | 0.4812 / 0.0140 | 2.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.4895 | 0.4812 / 0.4978 | False / False / False | 0.0029 / 0.0235 | 20.9 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1397): RAW8-15k bias +0.264, refcv7-50400 bias +0.177, R7-15k bias +0.166; RAW8-15k MAE 1.285, refcv7-50400 MAE 0.962, R7-15k MAE 1.135
* seg2 (n 1395): RAW8-15k bias -0.472, refcv7-50400 bias -0.201, R7-15k bias -0.611; RAW8-15k MAE 2.603, refcv7-50400 MAE 2.354, R7-15k MAE 2.601
* seg3 (n 1377): RAW8-15k bias -1.949, refcv7-50400 bias -0.931, R7-15k bias -2.292; RAW8-15k MAE 3.916, refcv7-50400 MAE 3.483, R7-15k MAE 4.068
* seg4 (n 1352): RAW8-15k bias -3.780, refcv7-50400 bias -2.408, R7-15k bias -4.561; RAW8-15k MAE 6.259, refcv7-50400 MAE 5.633, R7-15k MAE 6.821
* seg5 (n 1332): RAW8-15k bias -5.876, refcv7-50400 bias -4.384, R7-15k bias -7.209; RAW8-15k MAE 9.432, refcv7-50400 MAE 8.730, R7-15k MAE 10.509
* seg6 (n 1308): RAW8-15k bias -8.067, refcv7-50400 bias -6.863, R7-15k bias -9.347; RAW8-15k MAE 13.222, refcv7-50400 MAE 12.351, R7-15k MAE 14.566
**sharp (>= 0.15)**

* seg1 (n 300): RAW8-15k bias -0.434, refcv7-50400 bias +0.188, R7-15k bias -0.092; RAW8-15k MAE 1.497, refcv7-50400 MAE 1.441, R7-15k MAE 1.814
* seg2 (n 296): RAW8-15k bias -2.533, refcv7-50400 bias -0.481, R7-15k bias -2.746; RAW8-15k MAE 4.563, refcv7-50400 MAE 3.791, R7-15k MAE 5.160
* seg3 (n 292): RAW8-15k bias -6.346, refcv7-50400 bias -2.394, R7-15k bias -7.856; RAW8-15k MAE 8.392, refcv7-50400 MAE 6.600, R7-15k MAE 9.688
* seg4 (n 292): RAW8-15k bias -11.438, refcv7-50400 bias -6.256, R7-15k bias -14.589; RAW8-15k MAE 13.894, refcv7-50400 MAE 11.530, R7-15k MAE 16.940
* seg5 (n 288): RAW8-15k bias -16.763, refcv7-50400 bias -11.352, R7-15k bias -21.702; RAW8-15k MAE 20.952, refcv7-50400 MAE 18.671, R7-15k MAE 25.711
* seg6 (n 284): RAW8-15k bias -22.012, refcv7-50400 bias -17.208, R7-15k bias -25.589; RAW8-15k MAE 27.910, refcv7-50400 MAE 25.090, R7-15k MAE 33.256
**curved & v0 5-10**

* seg1 (n 547): RAW8-15k bias -0.158, refcv7-50400 bias +0.049, R7-15k bias +0.061; RAW8-15k MAE 1.046, refcv7-50400 MAE 0.896, R7-15k MAE 1.142
* seg2 (n 547): RAW8-15k bias -1.115, refcv7-50400 bias -0.516, R7-15k bias -0.897; RAW8-15k MAE 2.744, refcv7-50400 MAE 2.231, R7-15k MAE 2.791
* seg3 (n 544): RAW8-15k bias -2.861, refcv7-50400 bias -1.675, R7-15k bias -2.875; RAW8-15k MAE 4.743, refcv7-50400 MAE 3.806, R7-15k MAE 4.491
* seg4 (n 527): RAW8-15k bias -5.266, refcv7-50400 bias -3.611, R7-15k bias -5.503; RAW8-15k MAE 7.425, refcv7-50400 MAE 6.280, R7-15k MAE 7.589
* seg5 (n 512): RAW8-15k bias -7.736, refcv7-50400 bias -5.934, R7-15k bias -8.265; RAW8-15k MAE 10.846, refcv7-50400 MAE 9.456, R7-15k MAE 11.218
* seg6 (n 491): RAW8-15k bias -9.769, refcv7-50400 bias -8.411, R7-15k bias -10.182; RAW8-15k MAE 14.516, refcv7-50400 MAE 12.794, R7-15k MAE 15.060
**sharp & v0 5-10**

* seg1 (n 139): RAW8-15k bias -0.572, refcv7-50400 bias +0.062, R7-15k bias +0.172; RAW8-15k MAE 1.400, refcv7-50400 MAE 1.251, R7-15k MAE 1.664
* seg2 (n 139): RAW8-15k bias -3.441, refcv7-50400 bias -1.267, R7-15k bias -2.425; RAW8-15k MAE 4.920, refcv7-50400 MAE 3.564, R7-15k MAE 5.063
* seg3 (n 138): RAW8-15k bias -8.574, refcv7-50400 bias -4.253, R7-15k bias -8.511; RAW8-15k MAE 9.982, refcv7-50400 MAE 6.990, R7-15k MAE 9.948
* seg4 (n 138): RAW8-15k bias -14.854, refcv7-50400 bias -9.099, R7-15k bias -15.913; RAW8-15k MAE 16.438, refcv7-50400 MAE 12.470, R7-15k MAE 17.466
* seg5 (n 137): RAW8-15k bias -20.464, refcv7-50400 bias -14.439, R7-15k bias -22.322; RAW8-15k MAE 24.169, refcv7-50400 MAE 19.813, R7-15k MAE 25.636
* seg6 (n 133): RAW8-15k bias -23.874, refcv7-50400 bias -18.868, R7-15k bias -24.850; RAW8-15k MAE 30.430, refcv7-50400 MAE 26.065, R7-15k MAE 32.363

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1308 / 1308 | 26.0 | RAW8-15k -0.3098 [-0.3992, -0.2264]; refcv7-50400 -0.2472 [-0.3203, -0.1787]; R7-15k -0.3604 [-0.4478, -0.2782] | RAW8-15k -0.0626 [-0.1196, -0.0071] SEP; R7-15k -0.1132 [-0.1622, -0.0685] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1401 / 6764 | 16.3 | RAW8-15k -0.2412 [-0.3367, -0.1485]; refcv7-50400 -0.1701 [-0.2363, -0.1102]; R7-15k -0.2982 [-0.3797, -0.2208] | RAW8-15k -0.0711 [-0.1308, -0.0087] SEP; R7-15k -0.1281 [-0.1749, -0.0831] SEP |
| sharp (>= 0.15) | last segment | 284 / 284 | 63.0 | RAW8-15k -0.3455 [-0.4583, -0.2475]; refcv7-50400 -0.2635 [-0.3528, -0.1861]; R7-15k -0.3943 [-0.5069, -0.2962] | RAW8-15k -0.0820 [-0.1594, -0.0132] SEP; R7-15k -0.1308 [-0.1924, -0.0773] SEP |
| sharp (>= 0.15) | segments 2..K | 301 / 1452 | 39.8 | RAW8-15k -0.3038 [-0.4221, -0.2084]; refcv7-50400 -0.1914 [-0.2772, -0.1193]; R7-15k -0.3610 [-0.4733, -0.2667] | RAW8-15k -0.1124 [-0.1927, -0.0472] SEP; R7-15k -0.1696 [-0.2332, -0.1190] SEP |
| curved & v0 5-10 | last segment | 491 / 491 | 28.4 | RAW8-15k -0.3478 [-0.4629, -0.2412]; refcv7-50400 -0.2721 [-0.3972, -0.1731]; R7-15k -0.3559 [-0.4640, -0.2588] | RAW8-15k -0.0757 [-0.1876, +0.0401]; R7-15k -0.0838 [-0.1379, -0.0293] SEP |
| curved & v0 5-10 | segments 2..K | 547 / 2621 | 17.9 | RAW8-15k -0.3050 [-0.4231, -0.2085]; refcv7-50400 -0.2057 [-0.3208, -0.1200]; R7-15k -0.3147 [-0.4097, -0.2306] | RAW8-15k -0.0994 [-0.2066, +0.0069]; R7-15k -0.1091 [-0.1603, -0.0566] SEP |
| sharp & v0 5-10 | last segment | 133 / 133 | 67.1 | RAW8-15k -0.3676 [-0.4989, -0.2483]; refcv7-50400 -0.2716 [-0.4102, -0.1612]; R7-15k -0.3679 [-0.4926, -0.2601] | RAW8-15k -0.0960 [-0.2278, +0.0390]; R7-15k -0.0963 [-0.1537, -0.0488] SEP |
| sharp & v0 5-10 | segments 2..K | 139 / 685 | 43.9 | RAW8-15k -0.3336 [-0.4733, -0.2236]; refcv7-50400 -0.2111 [-0.3443, -0.1139]; R7-15k -0.3389 [-0.4516, -0.2407] | RAW8-15k -0.1225 [-0.2510, +0.0013]; R7-15k -0.1278 [-0.1865, -0.0769] SEP |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW8-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3504 | 0.5075 | 0.23023 | 0.32862 | 0.648 / 0.352 / 0.000 | 876.6 / 1778.4 |
| turn|straight <0.02 | 2092 | 0.5243 | 0.28205 | 0.35742 | 0.714 / 0.286 / 0.000 | 214.2 / 1096.8 |
| turn|gentle 0.02-0.06 | 674 | 0.3742 | 0.11246 | 0.13791 | 0.561 / 0.439 / 0.000 | 148.5 / 252.2 |
| turn|turn 0.06-0.15 | 426 | -0.1030 | 0.12062 | 0.16721 | 0.538 / 0.462 / 0.000 | 47.0 / -43.9 |
| turn|sharp >=0.15 | 301 | 1.5629 | 0.21514 | 0.65372 | 0.545 / 0.455 / 0.000 | 188.6 / 470.4 |
| turn|undef (GT crawling) | 11 | 0.2581 | 0.08460 | 0.18659 | 0.545 / 0.455 / 0.000 | 1.9 / 2.8 |
| speed|v0 <2 | 167 | 0.5014 | 0.34651 | 0.54738 | 0.695 / 0.305 / 0.000 | 41.7 / 83.7 |
| speed|v0 2-5 | 421 | 1.0590 | 0.87362 | 1.02901 | 0.729 / 0.271 / 0.000 | 151.1 / 445.8 |
| speed|v0 5-10 | 1040 | 0.8711 | 0.30875 | 0.45243 | 0.633 / 0.367 / 0.000 | 383.7 / 906.0 |
| speed|v0 10-15 | 952 | 0.2595 | 0.29141 | 0.29336 | 0.693 / 0.307 / 0.000 | 86.6 / 247.1 |
| speed|v0 15-25 | 679 | 0.1453 | 0.10034 | 0.11942 | 0.594 / 0.406 / 0.000 | 45.2 / 98.7 |
| speed|v0 >=25 | 245 | -0.0121 | 0.01496 | 0.00686 | 0.514 / 0.486 / 0.000 | 3.5 / -3.0 |
| nav|nav=follow | 2240 | 0.5494 | 0.21196 | 0.29869 | 0.645 / 0.355 / 0.000 | 491.4 / 1230.7 |
| nav|nav=left | 315 | 0.4700 | 0.46770 | 0.55314 | 0.711 / 0.289 / 0.000 | 96.1 / 148.0 |
| nav|nav=right | 949 | 0.4211 | 0.22342 | 0.32317 | 0.634 / 0.366 / 0.000 | 258.2 / 399.6 |
**RAW8-15k - R7-15k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3499 | 0.0122 | 0.10828 | 0.11585 | 0.577 / 0.423 / 0.000 | 799.8 / 42.6 |
| turn|straight <0.02 | 2087 | 0.3626 | 0.17912 | 0.23620 | 0.659 / 0.341 / 0.000 | 186.9 / 756.8 |
| turn|gentle 0.02-0.06 | 674 | 0.5033 | 0.05536 | 0.05575 | 0.527 / 0.473 / 0.000 | 218.2 / 339.2 |
| turn|turn 0.06-0.15 | 426 | -0.6533 | -0.26048 | -0.45733 | 0.430 / 0.570 / 0.000 | 74.9 / -278.3 |
| turn|sharp >=0.15 | 301 | -2.5845 | -1.27885 | -2.13077 | 0.326 / 0.674 / 0.000 | 142.4 / -777.9 |
| turn|undef (GT crawling) | 11 | 0.2623 | 0.07084 | 0.20821 | 0.727 / 0.273 / 0.000 | 1.6 / 2.9 |
| speed|v0 <2 | 162 | 0.1656 | 0.10198 | 0.24826 | 0.593 / 0.407 / 0.000 | 41.2 / 26.8 |
| speed|v0 2-5 | 421 | 0.2074 | 0.56580 | 0.67841 | 0.653 / 0.347 / 0.000 | 174.0 / 87.3 |
| speed|v0 5-10 | 1040 | 0.1903 | 0.14103 | 0.14645 | 0.562 / 0.438 / 0.000 | 294.3 / 197.9 |
| speed|v0 10-15 | 952 | -0.2221 | 0.12742 | 0.07700 | 0.597 / 0.403 / 0.000 | 93.6 / -211.4 |
| speed|v0 15-25 | 679 | -0.0841 | 0.03344 | 0.01215 | 0.533 / 0.467 / 0.000 | 38.4 / -57.1 |
| speed|v0 >=25 | 245 | -0.0038 | 0.02106 | 0.01138 | 0.543 / 0.457 / 0.000 | 3.9 / -0.9 |
| nav|nav=follow | 2235 | 0.0684 | 0.09632 | 0.11661 | 0.579 / 0.421 / 0.000 | 390.2 / 153.0 |
| nav|nav=left | 315 | -0.3062 | 0.29695 | 0.21923 | 0.654 / 0.346 / 0.000 | 99.5 / -96.4 |
| nav|nav=right | 949 | -0.0147 | 0.09987 | 0.06975 | 0.547 / 0.453 / 0.000 | 272.2 / -13.9 |
**R7-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3566 | 0.5500 | 0.11045 | 0.19996 | 0.623 / 0.377 / 0.000 | 1051.1 / 1961.4 |
| turn|straight <0.02 | 2112 | 0.1703 | 0.09098 | 0.12972 | 0.642 / 0.358 / 0.000 | 122.5 / 359.6 |
| turn|gentle 0.02-0.06 | 678 | -0.1070 | 0.04909 | 0.10347 | 0.534 / 0.466 / 0.000 | 39.9 / -72.6 |
| turn|turn 0.06-0.15 | 442 | 0.6398 | 0.33087 | 0.60070 | 0.633 / 0.367 / 0.000 | 69.0 / 282.8 |
| turn|sharp >=0.15 | 312 | 4.5092 | 1.68441 | 2.90109 | 0.673 / 0.327 / 0.000 | 228.2 / 1406.9 |
| turn|undef (GT crawling) | 22 | -0.6945 | 0.02016 | -0.07831 | 0.591 / 0.409 / 0.000 | 0.9 / -15.3 |
| speed|v0 <2 | 228 | 1.2405 | 0.29979 | 0.50663 | 0.706 / 0.294 / 0.000 | 111.9 / 282.8 |
| speed|v0 2-5 | 422 | 0.8490 | 0.31561 | 0.35721 | 0.673 / 0.327 / 0.000 | 239.4 / 358.3 |
| speed|v0 5-10 | 1040 | 0.6809 | 0.16305 | 0.22989 | 0.595 / 0.405 / 0.000 | 277.0 / 708.1 |
| speed|v0 10-15 | 952 | 0.4816 | 0.13341 | 0.20676 | 0.666 / 0.334 / 0.000 | 169.8 / 458.5 |
| speed|v0 15-25 | 679 | 0.2294 | 0.06110 | 0.11237 | 0.589 / 0.411 / 0.000 | 46.3 / 155.8 |
| speed|v0 >=25 | 245 | -0.0083 | 0.00484 | -0.01481 | 0.502 / 0.498 / 0.000 | 5.8 / -2.0 |
| nav|nav=follow | 2248 | 0.4806 | 0.09591 | 0.17776 | 0.624 / 0.376 / 0.000 | 467.6 / 1080.4 |
| nav|nav=left | 335 | 1.1935 | 0.20606 | 0.34476 | 0.663 / 0.337 / 0.000 | 203.7 / 399.8 |
| nav|nav=right | 983 | 0.4895 | 0.15393 | 0.21111 | 0.606 / 0.394 / 0.000 | 313.4 / 481.2 |

### S6 -- the episodes carrying the summed delta (panel file index only)

**RAW8-15k - refcv7-50400** (sum 1778.4 over 139 episodes)

* ep111: n 26, share 0.130, mean delta 8.9116, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share 0.070, mean delta 4.6297, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.062, mean delta 4.2379, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.054, mean delta 3.5712, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep048: n 27, share 0.048, mean delta 3.1582, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep136: n 18, share 0.045, mean delta 4.4846, sharp-window frac 0.33, curved frac 0.56, mean v0 1.4, nav {'nav=follow': 0, 'nav=left': 18, 'nav=right': 0}
**RAW8-15k - R7-15k** (sum 42.6 over 139 episodes)

* ep078: n 26, share 3.749, mean delta 6.1448, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep111: n 26, share 2.999, mean delta 4.9147, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share 2.590, mean delta 4.0878, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep048: n 27, share 2.147, mean delta 3.3884, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep136: n 18, share 1.392, mean delta 3.2963, sharp-window frac 0.33, curved frac 0.56, mean v0 1.4, nav {'nav=follow': 0, 'nav=left': 18, 'nav=right': 0}
* ep024: n 27, share 1.174, mean delta 1.8525, sharp-window frac 0.00, curved frac 0.00, mean v0 5.1, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
**R7-15k - refcv7-50400** (sum 1961.4 over 139 episodes)

* ep016: n 22, share 0.141, mean delta 12.5467, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.091, mean delta 6.8567, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep028: n 27, share 0.069, mean delta 5.0322, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.060, mean delta 4.5602, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.053, mean delta 3.9969, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.051, mean delta 3.7104, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW8-15k vs R7-15k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1068 / 0.1354 [0.0445, 0.2386] | 25 / -1.2636 [-2.3313, -0.2012] |
| only new agrees | 260 / -1.1300 [-1.9858, -0.3359] | 37 / -7.9433 [-12.8445, -4.3238] |
| only earlier agrees | 829 / 0.7441 [0.4320, 1.1372] | 31 / 4.8265 [-0.0994, 8.4591] |
| neither agrees | 1344 / -0.2945 [-0.8607, 0.2805] | 208 / -2.9025 [-5.3569, -0.2499] |
**RAW8-15k vs R7-15k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1065 / 0.1113 [0.0121, 0.2174] | 33 / -1.3876 [-2.6344, -0.4479] |
| only new agrees | 272 / -1.6860 [-2.8249, -0.6777] | 44 / -9.3191 [-14.0765, -5.2705] |
| only earlier agrees | 841 / 0.7756 [0.4442, 1.2280] | 32 / 5.5485 [-0.3901, 9.6707] |
| neither agrees | 1324 / -0.2221 [-0.7932, 0.3321] | 192 / -2.5939 [-5.2281, 0.1455] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW8-15k vs R7-15k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3455 | 0.3254 [0.0782, 0.5850] | 0.1371 [0.0004, 0.2890] | 0.2105 [-0.0087, 0.4385] |
| turn|straight <0.02 | 2058 | 0.0900 [-0.0492, 0.2208] | -0.0618 [-0.1170, -0.0192] | 0.1414 [-0.0025, 0.2812] |
| turn|gentle 0.02-0.06 | 672 | -0.0474 [-0.6952, 0.4447] | -0.1127 [-0.2438, 0.0024] | 0.0693 [-0.4831, 0.5211] |
| turn|turn 0.06-0.15 | 422 | 0.3576 [-0.0830, 0.8468] | -0.1208 [-0.4650, 0.2215] | 0.5181 [0.1389, 0.9703] SEP |
| turn|sharp >=0.15 | 296 | 2.7777 [0.8821, 5.2341] | 2.4018 [1.1364, 3.9775] | 0.5676 [-1.1821, 2.8395] |
| turn|undef (GT crawling) | 7 | -0.1820 [-0.7158, 0.1596] | -0.0902 [-0.6115, 0.3636] | 0.4331 [0.0772, 0.9013] SEP |
| speed|v0 <2 | 118 | 0.8918 [-0.0136, 2.3283] | 0.4735 [-0.0257, 1.1232] | 0.7628 [-0.2358, 2.3069] |
| speed|v0 2-5 | 421 | 0.9122 [-0.1437, 2.0808] | 0.1310 [-0.4453, 0.7842] | 0.7810 [-0.2949, 1.9848] |
| speed|v0 5-10 | 1040 | 0.5297 [0.0498, 1.0702] | 0.3403 [-0.0556, 0.7960] | 0.1894 [-0.2225, 0.6092] |
| speed|v0 10-15 | 952 | 0.0468 [-0.0493, 0.1359] | -0.0435 [-0.1644, 0.0865] | 0.0903 [-0.0630, 0.2411] |
| speed|v0 15-25 | 679 | 0.0536 [-0.0433, 0.1543] | 0.0007 [-0.0521, 0.0591] | 0.0530 [-0.0312, 0.1562] |
| speed|v0 >=25 | 245 | 0.0037 [-0.0093, 0.0211] | 0.0465 [0.0124, 0.0901] | -0.0428 [-0.0764, -0.0128] SEP |
**RAW8-15k vs R7-15k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* RAW8-15k vs R7-15k|seed0: anchor flip rate 0.511; heading delta on flipped windows -0.0425 [-0.5781, 0.4376] (n 1780) vs unflipped 0.0857 [-0.0161, 0.1798]; goal_dist_sel proxy new-earlier -0.3192 [-0.7103, 0.1064]; corr(window dHeading, window dGoalDist) -0.014
* RAW8-15k vs R7-15k|seed1: anchor flip rate 0.518; heading delta on flipped windows -0.0558 [-0.5044, 0.3847] (n 1804) vs unflipped 0.0700 [-0.0314, 0.1708]; goal_dist_sel proxy new-earlier -0.3346 [-0.7224, 0.0883]; corr(window dHeading, window dGoalDist) -0.009
