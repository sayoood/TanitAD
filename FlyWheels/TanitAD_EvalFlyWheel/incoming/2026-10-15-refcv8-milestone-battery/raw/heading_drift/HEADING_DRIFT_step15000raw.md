# refcv8 heading-drift localisation: RAW-15k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-17T10:59:51; trees: `RAW-15k` = E:/refcv8_eval_kit/battery/step15000_unc; `RAW-10.5k` = C:/Users/Admin/refcv8_eval_archive/battery/step10500_unc; `RAW-5k` = C:/Users/Admin/refcv8_eval_archive/battery/step5000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- RAW-15k - refcv7-50400, 2-seed mean of the per-window metric

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

### S2 -- RAW-15k - RAW-10.5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4485 | 139 | 1.4605 | 1.5771 | -0.1166 [-0.1840, -0.0528] | **SEP** |  |
| turn: straight <0.02 | 2890 | 132 | 1.2066 | 1.2835 | -0.0769 [-0.1208, -0.0323] | **SEP** | 0.644 / 0.425 / 0.1117 [-0.1298, 0.3804] |
| turn: gentle 0.02-0.06 | 718 | 97 | 1.0637 | 1.2182 | -0.1546 [-0.2540, -0.0536] | **SEP** | 0.160 / 0.212 / -0.0452 [-0.2028, 0.1178] |
| turn: turn 0.06-0.15 | 501 | 72 | 1.8822 | 2.2042 | -0.3220 [-0.5239, -0.1653] | **SEP** | 0.112 / 0.309 / -0.2312 [-0.4969, 0.0083] |
| turn: sharp >=0.15 | 364 | 42 | 3.6977 | 3.7798 | -0.0821 [-0.6665, 0.4816] |  | 0.081 / 0.057 / 0.0376 [-0.9399, 0.9500] |
| turn: undef (GT crawling) (low n) | 12 | 10 | 0.8595 | 0.7460 | 0.1135 [-0.1490, 0.4252] |  | 0.003 / -0.003 / 0.2307 [-0.2222, 0.7928] |
| speed: v0 <2 | 196 | 37 | 13.0928 | 13.1913 | -0.0985 [-0.3679, 0.1035] |  | 0.044 / 0.037 / 0.0190 [-0.4966, 0.3326] |
| speed: v0 2-5 | 516 | 58 | 2.0718 | 2.3452 | -0.2735 [-0.6235, 0.0603] |  | 0.115 / 0.270 / -0.1773 [-0.6578, 0.2862] |
| speed: v0 5-10 | 1390 | 84 | 1.2763 | 1.4436 | -0.1673 [-0.3224, -0.0277] | **SEP** | 0.310 / 0.445 / -0.0735 [-0.3275, 0.1494] |
| speed: v0 10-15 | 1192 | 64 | 0.6313 | 0.7422 | -0.1109 [-0.1433, -0.0815] | **SEP** | 0.266 / 0.253 / 0.0078 [-0.1338, 0.1571] |
| speed: v0 15-25 | 856 | 35 | 0.3773 | 0.3780 | -0.0007 [-0.0281, 0.0273] |  | 0.191 / 0.001 / 0.1432 [0.0242, 0.2697] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1954 | 0.1868 | 0.0085 [-0.0279, 0.0484] |  | 0.075 / -0.005 / 0.1352 [0.0294, 0.2513] BONF-SEP |
| nav: nav=follow | 2872 | 88 | 0.7300 | 0.8028 | -0.0727 [-0.1508, -0.0065] | **SEP** | 0.640 / 0.399 / 0.1220 [-0.0790, 0.3588] |
| nav: nav=left (low n) | 397 | 13 | 6.7371 | 6.8795 | -0.1424 [-0.3195, 0.0390] |  | 0.089 / 0.108 / -0.0283 [-0.3455, 0.2969] |
| nav: nav=right | 1216 | 38 | 1.4629 | 1.6747 | -0.2118 [-0.3976, -0.0474] | **SEP** | 0.271 / 0.492 / -0.1306 [-0.4175, 0.1150] |

episode concentration: 0.317 of episodes have mean delta > 0; top-5 episodes carry -0.240 of the summed delta; delta excluding the top 5 = -0.1500 (overall -0.1166).

### S2 -- RAW-10.5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4501 | 139 | 1.6586 | 1.4091 | 0.2496 [0.1860, 0.3214] | **SEP** |  |
| turn: straight <0.02 | 2894 | 132 | 1.3443 | 1.1604 | 0.1839 [0.1342, 0.2465] | **SEP** | 0.643 / 0.474 / -0.1840 [-0.4519, 0.0361] |
| turn: gentle 0.02-0.06 | 719 | 97 | 1.2192 | 1.0091 | 0.2101 [0.0821, 0.3429] | **SEP** | 0.160 / 0.135 / -0.0469 [-0.2448, 0.1668] |
| turn: turn 0.06-0.15 | 507 | 72 | 2.2091 | 1.9197 | 0.2894 [0.1323, 0.4670] | **SEP** | 0.113 / 0.131 / 0.0449 [-0.1970, 0.3295] |
| turn: sharp >=0.15 | 366 | 42 | 3.7840 | 2.9907 | 0.7933 [0.2905, 1.4437] | **SEP** | 0.081 / 0.258 / 0.5918 [-0.1026, 1.5291] |
| turn: undef (GT crawling) (low n) | 15 | 11 | 12.9113 | 12.7087 | 0.2026 [-0.2654, 0.6809] |  | 0.003 / 0.003 / -0.0472 [-0.7576, 0.7235] |
| speed: v0 <2 | 212 | 37 | 14.0468 | 13.7627 | 0.2841 [0.0556, 0.5953] | **SEP** | 0.047 / 0.054 / 0.0363 [-0.3261, 0.7168] |
| speed: v0 2-5 | 516 | 58 | 2.3452 | 1.8694 | 0.4758 [0.1030, 0.8766] | **SEP** | 0.115 / 0.219 / 0.2555 [-0.3088, 0.8976] |
| speed: v0 5-10 | 1390 | 84 | 1.4436 | 1.0092 | 0.4344 [0.3048, 0.5962] | **SEP** | 0.309 / 0.538 / 0.2675 [0.0560, 0.5433] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.7422 | 0.5744 | 0.1678 [0.1153, 0.2168] | **SEP** | 0.265 / 0.178 / -0.1112 [-0.2755, 0.0498] |
| speed: v0 15-25 | 856 | 35 | 0.3780 | 0.3734 | 0.0046 [-0.0389, 0.0439] |  | 0.190 / 0.003 / -0.3025 [-0.4540, -0.1810] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1868 | 0.1576 | 0.0292 [0.0037, 0.0574] | **SEP** | 0.074 / 0.009 / -0.2381 [-0.3631, -0.1318] BONF-SEP |
| nav: nav=follow | 2875 | 88 | 0.8027 | 0.5989 | 0.2038 [0.1300, 0.3014] | **SEP** | 0.639 / 0.522 / -0.1266 [-0.3355, 0.0864] |
| nav: nav=left (low n) | 401 | 13 | 7.2847 | 6.9367 | 0.3480 [0.1968, 0.5226] | **SEP** | 0.089 / 0.124 / 0.1080 [-0.1464, 0.4028] |
| nav: nav=right | 1225 | 38 | 1.8258 | 1.5011 | 0.3247 [0.1658, 0.4715] | **SEP** | 0.272 / 0.354 / 0.1033 [-0.1666, 0.3558] |

episode concentration: 0.827 of episodes have mean delta > 0; top-5 episodes carry 0.263 of the summed delta; delta excluding the top 5 = 0.1910 (overall 0.2496).

### S2 -- RAW-15k - RAW-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4464 | 139 | 1.4229 | 1.6064 | -0.1835 [-0.2764, -0.1026] | **SEP** |  |
| turn: straight <0.02 | 2878 | 132 | 1.1482 | 1.2696 | -0.1214 [-0.1977, -0.0576] | **SEP** | 0.645 / 0.426 / 0.1750 [-0.0736, 0.4573] |
| turn: gentle 0.02-0.06 | 716 | 97 | 1.0639 | 1.3581 | -0.2942 [-0.5252, -0.1045] | **SEP** | 0.160 / 0.257 / -0.1318 [-0.4377, 0.1412] |
| turn: turn 0.06-0.15 | 501 | 72 | 1.8822 | 2.0996 | -0.2174 [-0.4538, -0.0195] | **SEP** | 0.112 / 0.133 / -0.0382 [-0.4005, 0.2341] |
| turn: sharp >=0.15 | 363 | 42 | 3.6838 | 4.1009 | -0.4171 [-0.8731, 0.0480] |  | 0.081 / 0.185 / -0.2543 [-0.9511, 0.4305] |
| turn: undef (GT crawling) (low n) | 6 | 5 | 0.8953 | 0.7216 | 0.1736 [-0.1139, 0.7348] |  | 0.001 / -0.001 / 0.3576 [-0.0343, 1.5191] |
| speed: v0 <2 | 175 | 37 | 13.5310 | 13.6181 | -0.0870 [-0.4957, 0.3252] |  | 0.039 / 0.019 / 0.1004 [-0.5512, 0.8032] |
| speed: v0 2-5 | 516 | 58 | 2.0718 | 2.2384 | -0.1667 [-0.4527, 0.1642] |  | 0.116 / 0.105 / 0.0190 [-0.4278, 0.5418] |
| speed: v0 5-10 | 1390 | 84 | 1.2763 | 1.7324 | -0.4561 [-0.6585, -0.2769] | **SEP** | 0.311 / 0.774 / -0.3958 [-0.7146, -0.1271] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.6313 | 0.7029 | -0.0716 [-0.1323, -0.0183] | **SEP** | 0.267 / 0.104 / 0.1527 [-0.0199, 0.3445] |
| speed: v0 15-25 | 856 | 35 | 0.3773 | 0.3689 | 0.0084 [-0.0290, 0.0483] |  | 0.192 / -0.009 / 0.2374 [0.0825, 0.4119] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1954 | 0.2129 | -0.0176 [-0.0800, 0.0341] |  | 0.075 / 0.007 / 0.1794 [0.0149, 0.3533] BONF-SEP |
| nav: nav=follow | 2860 | 88 | 0.7313 | 0.8690 | -0.1377 [-0.2224, -0.0640] | **SEP** | 0.641 / 0.481 / 0.1274 [-0.1619, 0.4662] |
| nav: nav=left (low n) | 395 | 13 | 6.3080 | 6.1786 | 0.1294 [-0.0575, 0.3674] |  | 0.088 / -0.062 / 0.3433 [0.0372, 0.7456] BONF-SEP |
| nav: nav=right | 1209 | 38 | 1.4631 | 1.8571 | -0.3940 [-0.6148, -0.1918] | **SEP** | 0.271 / 0.582 / -0.2887 [-0.6634, 0.0697] |

episode concentration: 0.353 of episodes have mean delta > 0; top-5 episodes carry -0.168 of the summed delta; delta excluding the top 5 = -0.2220 (overall -0.1835).

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

### S2 -- RAW-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

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

### S2 -- RAW-15k - RAW-10.5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4456 | 139 | 0.7456 | 0.8211 | -0.0755 [-0.1604, -0.0229] | **SEP** | -0.0119 [-0.1085, 0.0536] |
| seg1 | straight | 2874 | 132 | 0.7910 | 0.8830 | -0.0921 [-0.2388, -0.0185] | **SEP** | -0.0931 [-0.2419, -0.0138] SEP |
| seg1 | curved | 1576 | 101 | 0.6629 | 0.7092 | -0.0463 [-0.0864, -0.0077] | **SEP** | 0.1371 [0.0677, 0.2128] SEP |
| seg2 | all | 4434 | 139 | 1.1586 | 1.2620 | -0.1034 [-0.1545, -0.0544] | **SEP** | 0.0154 [-0.0763, 0.1047] |
| seg2 | straight | 2863 | 132 | 0.9387 | 1.0101 | -0.0715 [-0.1109, -0.0324] | **SEP** | -0.1379 [-0.2062, -0.0768] SEP |
| seg2 | curved | 1571 | 101 | 1.5594 | 1.7210 | -0.1615 [-0.2818, -0.0431] | **SEP** | 0.2948 [0.1085, 0.4979] SEP |
| seg3 | all | 4422 | 139 | 1.4233 | 1.5734 | -0.1501 [-0.2367, -0.0704] | **SEP** | 0.0266 [-0.1024, 0.1563] |
| seg3 | straight | 2855 | 132 | 1.0048 | 1.1121 | -0.1073 [-0.1642, -0.0500] | **SEP** | -0.1647 [-0.2534, -0.0791] SEP |
| seg3 | curved | 1567 | 101 | 2.1858 | 2.4138 | -0.2279 [-0.4405, -0.0272] | **SEP** | 0.3751 [0.0981, 0.6768] SEP |
| seg4 | all | 4409 | 138 | 1.7772 | 1.9204 | -0.1432 [-0.2968, 0.0143] |  | 0.2575 [0.0547, 0.4640] SEP |
| seg4 | straight | 2840 | 131 | 1.1232 | 1.1709 | -0.0477 [-0.1792, 0.1337] |  | -0.0337 [-0.2041, 0.1105] |
| seg4 | curved | 1563 | 100 | 2.9686 | 3.2861 | -0.3175 [-0.6405, -0.0053] | **SEP** | 0.7861 [0.3677, 1.2432] SEP |

### S2 -- RAW-10.5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4470 | 139 | 0.9008 | 0.6288 | 0.2719 [0.0567, 0.6876] | **SEP** | 0.4517 [-0.0144, 1.3634] |
| seg1 | straight | 2877 | 132 | 0.9450 | 0.5700 | 0.3750 [0.0503, 1.0734] | **SEP** | 0.7279 [0.0272, 2.2606] SEP |
| seg1 | curved | 1585 | 101 | 0.7097 | 0.6229 | 0.0868 [0.0501, 0.1250] | **SEP** | -0.0474 [-0.1288, 0.0263] |
| seg2 | all | 4450 | 139 | 1.3028 | 0.9397 | 0.3631 [0.1603, 0.7140] | **SEP** | 0.2654 [-0.0435, 0.7916] |
| seg2 | straight | 2869 | 132 | 1.0723 | 0.6585 | 0.4138 [0.1310, 0.9827] | **SEP** | 0.4874 [0.0745, 1.3199] SEP |
| seg2 | curved | 1581 | 101 | 1.7211 | 1.4500 | 0.2711 [0.1555, 0.3927] | **SEP** | -0.1373 [-0.3660, 0.0734] |
| seg3 | all | 4441 | 139 | 1.6194 | 1.1650 | 0.4544 [0.2282, 0.8223] | **SEP** | 0.2614 [-0.0839, 0.8010] |
| seg3 | straight | 2863 | 132 | 1.1776 | 0.7126 | 0.4650 [0.1793, 1.0370] | **SEP** | 0.5295 [0.1015, 1.3635] SEP |
| seg3 | curved | 1578 | 101 | 2.4209 | 1.9858 | 0.4351 [0.2230, 0.6647] | **SEP** | -0.2249 [-0.5683, 0.0967] |
| seg4 | all | 4429 | 139 | 1.9709 | 1.4530 | 0.5178 [0.3001, 0.8282] | **SEP** | 0.2850 [-0.1455, 0.9209] |
| seg4 | straight | 2850 | 131 | 1.2386 | 0.7768 | 0.4618 [0.2268, 0.8960] | **SEP** | 0.6338 [0.1127, 1.6124] SEP |
| seg4 | curved | 1572 | 101 | 3.3005 | 2.6820 | 0.6186 [0.3162, 0.9416] | **SEP** | -0.3415 [-0.8483, 0.1218] |

### S2 -- RAW-15k - RAW-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4451 | 139 | 0.7053 | 0.8264 | -0.1211 [-0.2880, -0.0262] | **SEP** | 0.1102 [0.0815, 0.1396] SEP |
| seg1 | straight | 2872 | 132 | 0.7288 | 0.8764 | -0.1476 [-0.4325, -0.0106] | **SEP** | 0.0997 [0.0773, 0.1238] SEP |
| seg1 | curved | 1575 | 101 | 0.6633 | 0.7363 | -0.0730 [-0.1156, -0.0295] | **SEP** | 0.1295 [0.0717, 0.1885] SEP |
| seg2 | all | 4425 | 139 | 0.9986 | 1.2032 | -0.2046 [-0.3100, -0.1144] | **SEP** | 0.2671 [0.1622, 0.3810] SEP |
| seg2 | straight | 2854 | 132 | 0.6892 | 0.8531 | -0.1639 [-0.3324, -0.0621] | **SEP** | 0.2833 [0.1722, 0.4541] SEP |
| seg2 | curved | 1571 | 101 | 1.5607 | 1.8391 | -0.2784 [-0.4238, -0.1413] | **SEP** | 0.2376 [0.0621, 0.4185] SEP |
| seg3 | all | 4410 | 139 | 1.2665 | 1.5354 | -0.2689 [-0.3943, -0.1590] | **SEP** | 0.2870 [0.1548, 0.4279] SEP |
| seg3 | straight | 2843 | 132 | 0.7598 | 0.9431 | -0.1833 [-0.2828, -0.0965] | **SEP** | 0.3153 [0.2093, 0.4317] SEP |
| seg3 | curved | 1567 | 101 | 2.1857 | 2.6099 | -0.4242 [-0.6733, -0.1890] | **SEP** | 0.2357 [-0.0436, 0.5328] |
| seg4 | all | 4390 | 138 | 1.6975 | 1.9463 | -0.2489 [-0.4384, -0.0573] | **SEP** | 0.5984 [0.3798, 0.8105] SEP |
| seg4 | straight | 2826 | 131 | 0.9985 | 1.1304 | -0.1319 [-0.2962, 0.0529] |  | 0.5890 [0.3779, 0.7831] SEP |
| seg4 | curved | 1562 | 100 | 2.9615 | 3.4232 | -0.4616 [-0.7782, -0.1553] | **SEP** | 0.6170 [0.2409, 1.0045] SEP |

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

### S2 -- RAW-15k - refcv7-50400: turn x speed cross

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

| bin | RAW-15k s0-s1 [CI] | RAW-10.5k s0-s1 [CI] | RAW-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|---|
| overall | -0.0060 [-0.0201, 0.0088] | 0.0013 [-0.0193, 0.0202] | -0.0161 [-0.0334, 0.0008] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0110 [-0.0305, 0.0106] | -0.0015 [-0.0224, 0.0186] | -0.0212 [-0.0379, -0.0051] SEP | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0111 [-0.0331, 0.0502] | 0.0561 [0.0004, 0.1339] SEP | -0.0019 [-0.0291, 0.0225] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | -0.0149 [-0.0627, 0.0257] | -0.0008 [-0.0383, 0.0351] | 0.0382 [-0.0075, 0.0961] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | 0.0178 [-0.0676, 0.1126] | -0.0795 [-0.1922, 0.0200] | -0.0785 [-0.2171, 0.0460] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.1186 [-0.2692, 0.0150] | -0.0410 [-0.3427, 0.2041] | -0.0511 [-0.2590, 0.0676] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | -0.0344 [-0.0825, 0.0108] | -0.0636 [-0.1842, 0.0571] | 0.0407 [-0.0974, 0.1929] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0099 [-0.1126, 0.1039] | 0.0422 [-0.0315, 0.1338] | -0.1039 [-0.2163, -0.0093] SEP | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | -0.0164 [-0.0496, 0.0152] | -0.0004 [-0.0507, 0.0559] | -0.0220 [-0.0551, 0.0157] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0026 [-0.0057, 0.0108] | -0.0032 [-0.0199, 0.0129] | 0.0040 [-0.0105, 0.0180] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | 0.0052 [-0.0022, 0.0127] | 0.0045 [-0.0023, 0.0113] | 0.0020 [-0.0062, 0.0111] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | 0.0020 [-0.0057, 0.0107] | -0.0059 [-0.0120, -0.0001] SEP | -0.0065 [-0.0241, 0.0083] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0108 [-0.0232, -0.0001] SEP | 0.0012 [-0.0158, 0.0162] | -0.0127 [-0.0307, 0.0056] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0245 [-0.0222, 0.0720] | -0.0156 [-0.1567, 0.1151] | -0.0176 [-0.0773, 0.0512] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0046 [-0.0461, 0.0396] | 0.0069 [-0.0314, 0.0549] | -0.0237 [-0.0680, 0.0187] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- RAW-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

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

### S2 -- RAW-15k - RAW-10.5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.1166 | -0.1204 / -0.1132 | True / True / True | 0.0060 / 0.0013 | 19.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | -0.0769 | -0.0820 / -0.0718 | True / True / True | 0.0110 / 0.0015 | 7.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.1546 | -0.1775 / -0.1319 | True / True / True | 0.0111 / 0.0561 | 2.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.3220 | -0.3288 / -0.3175 | True / True / True | 0.0149 / 0.0008 | 21.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -0.0821 | -0.0335 / -0.1306 | False / False / False | 0.0178 / 0.0795 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.1135 | 0.1035 / 0.0991 | False / False / False | 0.1186 / 0.0410 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0985 | -0.0874 / -0.1202 | False / False / False | 0.0344 / 0.0636 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.2735 | -0.2995 / -0.2474 | False / False / False | 0.0099 / 0.0422 | 6.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.1673 | -0.1753 / -0.1594 | True / True / True | 0.0164 / 0.0004 | 10.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | -0.1109 | -0.1080 / -0.1138 | True / True / True | 0.0026 / 0.0032 | 35.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.0007 | -0.0003 / -0.0010 | False / False / False | 0.0052 / 0.0045 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0085 | 0.0125 / 0.0046 | False / False / False | 0.0020 / 0.0059 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.0727 | -0.0787 / -0.0669 | True / True / True | 0.0108 / 0.0012 | 6.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.1424 | -0.1257 / -0.1591 | False / False / False | 0.0245 / 0.0156 | 5.8 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.2118 | -0.2173 / -0.2074 | True / True / True | 0.0046 / 0.0069 | 30.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- RAW-10.5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.2496 | 0.2450 / 0.2541 | True / True / True | 0.0013 / 0.0110 | 22.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.1839 | 0.1828 / 0.1851 | True / True / True | 0.0015 / 0.0019 | 95.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.2101 | 0.2449 / 0.1750 | True / True / True | 0.0561 / 0.0161 | 3.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | 0.2894 | 0.2748 / 0.3041 | True / True / True | 0.0008 / 0.0331 | 8.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 0.7933 | 0.6970 / 0.8896 | True / True / True | 0.0795 / 0.1080 | 7.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.2026 | 0.2197 / 0.1831 | False / False / False | 0.0410 / 0.0237 | 4.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.2841 | 0.2262 / 0.3407 | False / True / True | 0.0636 / 0.0573 | 4.5 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 2-5 | 0.4758 | 0.4609 / 0.4906 | True / True / True | 0.0422 / 0.0719 | 6.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | 0.4344 | 0.4339 / 0.4350 | True / True / True | 0.0004 / 0.0006 | 669.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1678 | 0.1663 / 0.1693 | True / True / True | 0.0032 / 0.0001 | 53.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.0046 | 0.0079 / 0.0012 | False / False / False | 0.0045 / 0.0023 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0292 | 0.0265 / 0.0319 | True / True / True | 0.0059 / 0.0005 | 5.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.2038 | 0.2064 / 0.2013 | True / True / True | 0.0012 / 0.0036 | 57.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.3480 | 0.3127 / 0.3832 | True / True / True | 0.0156 / 0.0608 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | 0.3247 | 0.3134 / 0.3357 | True / True / True | 0.0069 / 0.0283 | 11.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S2 -- RAW-15k - RAW-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.1835 | -0.1778 / -0.1887 | True / True / True | 0.0060 / 0.0161 | 11.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | -0.1214 | -0.1161 / -0.1260 | True / True / True | 0.0110 / 0.0212 | 5.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.2942 | -0.2872 / -0.2988 | True / True / True | 0.0111 / 0.0019 | 26.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.2174 | -0.2409 / -0.1964 | True / False / True | 0.0149 / 0.0382 | 5.7 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|sharp >=0.15 | -0.4171 | -0.3686 / -0.4656 | False / False / False | 0.0178 / 0.0785 | 5.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.1736 | 0.1066 / 0.2407 | False / False / False | 0.1186 / 0.0511 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0870 | -0.1131 / -0.0512 | False / False / False | 0.0344 / 0.0407 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.1667 | -0.1197 / -0.2137 | False / False / False | 0.0099 / 0.1039 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.4561 | -0.4533 / -0.4589 | True / True / True | 0.0164 / 0.0220 | 20.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | -0.0716 | -0.0723 / -0.0709 | True / True / True | 0.0026 / 0.0040 | 18.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | 0.0084 | 0.0100 / 0.0068 | False / False / False | 0.0052 / 0.0020 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0176 | -0.0134 / -0.0218 | False / False / False | 0.0020 / 0.0065 | 2.7 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.1377 | -0.1365 / -0.1385 | True / True / True | 0.0108 / 0.0127 | 10.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.1294 | 0.1504 / 0.1092 | False / False / False | 0.0245 / 0.0176 | 5.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.3940 | -0.3827 / -0.4046 | True / True / True | 0.0046 / 0.0237 | 16.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

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

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1575): RAW-15k bias -0.075, refcv7-50400 bias +0.087, RAW-10.5k bias +0.038, RAW-5k bias -0.139; RAW-15k MAE 0.663, refcv7-50400 MAE 0.622, RAW-10.5k MAE 0.710, RAW-5k MAE 0.736
* seg2 (n 1569): RAW-15k bias -0.476, refcv7-50400 bias -0.076, RAW-10.5k bias -0.201, RAW-5k bias -0.664; RAW-15k MAE 1.560, refcv7-50400 MAE 1.443, RAW-10.5k MAE 1.722, RAW-5k MAE 1.839
* seg3 (n 1566): RAW-15k bias -1.139, refcv7-50400 bias -0.594, RAW-10.5k bias -0.775, RAW-5k bias -1.356; RAW-15k MAE 2.186, refcv7-50400 MAE 1.975, RAW-10.5k MAE 2.415, RAW-5k MAE 2.611
* seg4 (n 1561): RAW-15k bias -1.613, refcv7-50400 bias -0.679, RAW-10.5k bias -1.038, RAW-5k bias -1.735; RAW-15k MAE 2.962, refcv7-50400 MAE 2.660, RAW-10.5k MAE 3.279, RAW-5k MAE 3.424
**sharp (>= 0.15)**

* seg1 (n 363): RAW-15k bias -0.522, refcv7-50400 bias +0.013, RAW-10.5k bias -0.181, RAW-5k bias -0.765; RAW-15k MAE 1.006, refcv7-50400 MAE 0.916, RAW-10.5k MAE 1.043, RAW-5k MAE 1.104
* seg2 (n 362): RAW-15k bias -1.820, refcv7-50400 bias -0.506, RAW-10.5k bias -0.967, RAW-5k bias -2.472; RAW-15k MAE 2.787, refcv7-50400 MAE 2.411, RAW-10.5k MAE 2.937, RAW-5k MAE 3.195
* seg3 (n 361): RAW-15k bias -3.607, refcv7-50400 bias -1.831, RAW-10.5k bias -2.443, RAW-5k bias -4.276; RAW-15k MAE 4.509, refcv7-50400 MAE 3.637, RAW-10.5k MAE 4.561, RAW-5k MAE 5.213
* seg4 (n 360): RAW-15k bias -4.817, refcv7-50400 bias -1.790, RAW-10.5k bias -2.917, RAW-5k bias -5.219; RAW-15k MAE 6.244, refcv7-50400 MAE 4.865, RAW-10.5k MAE 6.298, RAW-5k MAE 6.847
**curved & v0 5-10**

* seg1 (n 627): RAW-15k bias -0.220, refcv7-50400 bias -0.038, RAW-10.5k bias -0.109, RAW-5k bias -0.326; RAW-15k MAE 0.612, refcv7-50400 MAE 0.535, RAW-10.5k MAE 0.668, RAW-5k MAE 0.766
* seg2 (n 627): RAW-15k bias -0.769, refcv7-50400 bias -0.282, RAW-10.5k bias -0.480, RAW-5k bias -1.091; RAW-15k MAE 1.621, refcv7-50400 MAE 1.363, RAW-10.5k MAE 1.813, RAW-5k MAE 2.165
* seg3 (n 627): RAW-15k bias -1.557, refcv7-50400 bias -0.871, RAW-10.5k bias -1.171, RAW-5k bias -1.942; RAW-15k MAE 2.474, refcv7-50400 MAE 2.019, RAW-10.5k MAE 2.729, RAW-5k MAE 3.294
* seg4 (n 627): RAW-15k bias -2.250, refcv7-50400 bias -1.194, RAW-10.5k bias -1.665, RAW-5k bias -2.505; RAW-15k MAE 3.335, refcv7-50400 MAE 2.730, RAW-10.5k MAE 3.645, RAW-5k MAE 4.229
**sharp & v0 5-10**

* seg1 (n 164): RAW-15k bias -0.444, refcv7-50400 bias +0.103, RAW-10.5k bias -0.081, RAW-5k bias -0.772; RAW-15k MAE 0.932, refcv7-50400 MAE 0.857, RAW-10.5k MAE 1.045, RAW-5k MAE 1.127
* seg2 (n 164): RAW-15k bias -1.852, refcv7-50400 bias -0.422, RAW-10.5k bias -0.895, RAW-5k bias -2.771; RAW-15k MAE 2.760, refcv7-50400 MAE 2.243, RAW-10.5k MAE 2.980, RAW-5k MAE 3.359
* seg3 (n 164): RAW-15k bias -4.042, refcv7-50400 bias -2.032, RAW-10.5k bias -2.770, RAW-5k bias -5.059; RAW-15k MAE 4.768, refcv7-50400 MAE 3.465, RAW-10.5k MAE 4.689, RAW-5k MAE 5.585
* seg4 (n 164): RAW-15k bias -5.613, refcv7-50400 bias -2.403, RAW-10.5k bias -3.797, RAW-5k bias -6.283; RAW-15k MAE 6.707, refcv7-50400 MAE 4.635, RAW-10.5k MAE 6.409, RAW-5k MAE 7.248

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1561 / 1561 | 10.2 | RAW-15k -0.1979 [-0.2882, -0.1237]; refcv7-50400 -0.0822 [-0.1530, -0.0265]; RAW-10.5k -0.1232 [-0.2230, -0.0369]; RAW-5k -0.2108 [-0.3020, -0.1349] | RAW-15k -0.1157 [-0.1612, -0.0809] SEP; RAW-10.5k -0.0410 [-0.1012, +0.0100]; RAW-5k -0.1286 [-0.1733, -0.0866] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1572 / 4696 | 7.2 | RAW-15k -0.2490 [-0.3652, -0.1470]; refcv7-50400 -0.1450 [-0.2636, -0.0511]; RAW-10.5k -0.1813 [-0.3136, -0.0668]; RAW-5k -0.2712 [-0.3915, -0.1686] | RAW-15k -0.1040 [-0.1442, -0.0714] SEP; RAW-10.5k -0.0364 [-0.0889, +0.0081]; RAW-5k -0.1263 [-0.1710, -0.0850] SEP |
| sharp (>= 0.15) | last segment | 360 / 360 | 25.4 | RAW-15k -0.2174 [-0.3233, -0.1319]; refcv7-50400 -0.0909 [-0.1741, -0.0240]; RAW-10.5k -0.1365 [-0.2556, -0.0381]; RAW-5k -0.2335 [-0.3399, -0.1454] | RAW-15k -0.1264 [-0.1808, -0.0875] SEP; RAW-10.5k -0.0456 [-0.1164, +0.0139]; RAW-5k -0.1426 [-0.1955, -0.0951] SEP |
| sharp (>= 0.15) | segments 2..K | 362 / 1083 | 18.0 | RAW-15k -0.2229 [-0.3283, -0.1383]; refcv7-50400 -0.1030 [-0.1864, -0.0346]; RAW-10.5k -0.1450 [-0.2634, -0.0477]; RAW-5k -0.2506 [-0.3575, -0.1621] | RAW-15k -0.1199 [-0.1703, -0.0825] SEP; RAW-10.5k -0.0421 [-0.1111, +0.0144]; RAW-5k -0.1476 [-0.1997, -0.1007] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | RAW-15k -0.2132 [-0.2990, -0.1420]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; RAW-10.5k -0.1394 [-0.2555, -0.0496]; RAW-5k -0.2329 [-0.3196, -0.1651] | RAW-15k -0.1207 [-0.2022, -0.0553] SEP; RAW-10.5k -0.0469 [-0.1489, +0.0290]; RAW-5k -0.1404 [-0.2057, -0.0827] SEP |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | RAW-15k -0.2166 [-0.2998, -0.1476]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; RAW-10.5k -0.1431 [-0.2553, -0.0571]; RAW-5k -0.2510 [-0.3376, -0.1825] | RAW-15k -0.1173 [-0.1949, -0.0574] SEP; RAW-10.5k -0.0438 [-0.1409, +0.0285]; RAW-5k -0.1517 [-0.2132, -0.0962] SEP |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | RAW-15k -0.2175 [-0.3153, -0.1372]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; RAW-10.5k -0.1380 [-0.2686, -0.0406]; RAW-5k -0.2390 [-0.3324, -0.1610] | RAW-15k -0.1263 [-0.2180, -0.0577] SEP; RAW-10.5k -0.0467 [-0.1633, +0.0386]; RAW-5k -0.1477 [-0.2206, -0.0862] SEP |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | RAW-15k -0.2173 [-0.3111, -0.1400]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; RAW-10.5k -0.1373 [-0.2644, -0.0428]; RAW-5k -0.2546 [-0.3487, -0.1783] | RAW-15k -0.1218 [-0.2115, -0.0569] SEP; RAW-10.5k -0.0418 [-0.1540, +0.0400]; RAW-5k -0.1591 [-0.2286, -0.0991] SEP |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW-15k - refcv7-50400**

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
**RAW-15k - RAW-10.5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4485 | -0.1166 | -0.04450 | -0.07174 | 0.413 / 0.587 / 0.000 | 285.1 / -522.9 |
| turn|straight <0.02 | 2890 | -0.0769 | -0.03982 | -0.06041 | 0.402 / 0.598 / 0.000 | 119.0 / -222.1 |
| turn|gentle 0.02-0.06 | 718 | -0.1546 | -0.07141 | -0.11605 | 0.390 / 0.610 / 0.000 | 40.4 / -111.0 |
| turn|turn 0.06-0.15 | 501 | -0.3220 | -0.07385 | -0.13542 | 0.419 / 0.581 / 0.000 | 16.4 / -161.3 |
| turn|sharp >=0.15 | 364 | -0.0821 | 0.04719 | 0.08565 | 0.533 / 0.467 / 0.000 | 54.8 / -29.9 |
| turn|undef (GT crawling) | 12 | 0.1135 | -0.00744 | 0.05145 | 0.500 / 0.500 / 0.000 | 1.4 / 1.4 |
| speed|v0 <2 | 196 | -0.0985 | -0.02661 | -0.03587 | 0.469 / 0.531 / 0.000 | 4.1 / -19.3 |
| speed|v0 2-5 | 516 | -0.2735 | -0.14596 | -0.20443 | 0.405 / 0.595 / 0.000 | 59.7 / -141.1 |
| speed|v0 5-10 | 1390 | -0.1673 | -0.11048 | -0.11483 | 0.383 / 0.617 / 0.000 | 112.1 / -232.6 |
| speed|v0 10-15 | 1192 | -0.1109 | -0.08122 | -0.09237 | 0.322 / 0.678 / 0.000 | 19.6 / -132.2 |
| speed|v0 15-25 | 856 | -0.0007 | 0.00510 | 0.00465 | 0.525 / 0.475 / 0.000 | 6.8 / -0.6 |
| speed|v0 >=25 | 335 | 0.0085 | 0.01628 | 0.01357 | 0.549 / 0.451 / 0.000 | 1.2 / 2.9 |
| nav|nav=follow | 2872 | -0.0727 | -0.03123 | -0.05285 | 0.430 / 0.570 / 0.000 | 155.0 / -208.9 |
| nav|nav=left | 397 | -0.1424 | -0.07363 | -0.07190 | 0.395 / 0.605 / 0.000 | 15.4 / -56.5 |
| nav|nav=right | 1216 | -0.2118 | -0.09205 | -0.12869 | 0.377 / 0.623 / 0.000 | 99.9 / -257.5 |
**RAW-10.5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4501 | 0.2496 | 0.08120 | 0.12748 | 0.641 / 0.359 / 0.000 | 443.3 / 1123.3 |
| turn|straight <0.02 | 2894 | 0.1839 | 0.07829 | 0.11564 | 0.679 / 0.321 / 0.000 | 168.2 / 532.1 |
| turn|gentle 0.02-0.06 | 719 | 0.2101 | 0.07891 | 0.12491 | 0.580 / 0.420 / 0.000 | 50.5 / 151.1 |
| turn|turn 0.06-0.15 | 507 | 0.2894 | 0.10137 | 0.16123 | 0.576 / 0.424 / 0.000 | 34.8 / 146.7 |
| turn|sharp >=0.15 | 366 | 0.7933 | 0.14264 | 0.27320 | 0.557 / 0.443 / 0.000 | 68.7 / 290.3 |
| turn|undef (GT crawling) | 15 | 0.2026 | 0.03459 | 0.00635 | 0.533 / 0.467 / 0.000 | 4.3 / 3.0 |
| speed|v0 <2 | 212 | 0.2841 | 0.03157 | 0.06081 | 0.552 / 0.448 / 0.000 | 27.3 / 60.2 |
| speed|v0 2-5 | 516 | 0.4758 | 0.21920 | 0.32483 | 0.661 / 0.339 / 0.000 | 77.4 / 245.5 |
| speed|v0 5-10 | 1390 | 0.4344 | 0.18911 | 0.22378 | 0.672 / 0.328 / 0.000 | 167.3 / 603.9 |
| speed|v0 10-15 | 1192 | 0.1678 | 0.12372 | 0.13677 | 0.695 / 0.305 / 0.000 | 34.2 / 200.0 |
| speed|v0 15-25 | 856 | 0.0046 | 0.00907 | 0.01651 | 0.539 / 0.461 / 0.000 | 9.0 / 3.9 |
| speed|v0 >=25 | 335 | 0.0292 | 0.02186 | 0.02337 | 0.609 / 0.391 / 0.000 | 1.9 / 9.8 |
| nav|nav=follow | 2875 | 0.2038 | 0.05093 | 0.09005 | 0.622 / 0.378 / 0.000 | 253.1 / 586.0 |
| nav|nav=left | 401 | 0.3480 | 0.17874 | 0.20661 | 0.743 / 0.257 / 0.000 | 41.9 / 139.5 |
| nav|nav=right | 1225 | 0.3247 | 0.15198 | 0.20594 | 0.653 / 0.347 / 0.000 | 133.3 / 397.8 |
**RAW-15k - RAW-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4464 | -0.1835 | -0.02964 | -0.07006 | 0.453 / 0.547 / 0.000 | 301.0 / -819.2 |
| turn|straight <0.02 | 2878 | -0.1214 | -0.01606 | -0.03552 | 0.467 / 0.533 / 0.000 | 130.8 / -349.2 |
| turn|gentle 0.02-0.06 | 716 | -0.2942 | -0.07324 | -0.12120 | 0.427 / 0.573 / 0.000 | 45.6 / -210.7 |
| turn|turn 0.06-0.15 | 501 | -0.2174 | -0.09084 | -0.10827 | 0.437 / 0.563 / 0.000 | 27.1 / -108.9 |
| turn|sharp >=0.15 | 363 | -0.4171 | -0.23039 | -0.33263 | 0.416 / 0.584 / 0.000 | 45.6 / -151.4 |
| turn|undef (GT crawling) | 6 | 0.1736 | -0.06371 | -0.02785 | 0.333 / 0.667 / 0.000 | 1.3 / 1.0 |
| speed|v0 <2 | 175 | -0.0870 | -0.08162 | -0.11542 | 0.383 / 0.617 / 0.000 | 10.7 / -15.2 |
| speed|v0 2-5 | 516 | -0.1667 | -0.14290 | -0.22493 | 0.419 / 0.581 / 0.000 | 65.8 / -86.0 |
| speed|v0 5-10 | 1390 | -0.4561 | -0.10361 | -0.16879 | 0.398 / 0.602 / 0.000 | 93.0 / -633.9 |
| speed|v0 10-15 | 1192 | -0.0716 | -0.02345 | -0.03156 | 0.466 / 0.534 / 0.000 | 24.1 / -85.3 |
| speed|v0 15-25 | 856 | 0.0084 | 0.00233 | 0.00492 | 0.506 / 0.494 / 0.000 | 13.0 / 7.2 |
| speed|v0 >=25 | 335 | -0.0176 | 0.02312 | 0.01096 | 0.594 / 0.406 / 0.000 | 2.8 / -5.9 |
| nav|nav=follow | 2860 | -0.1377 | -0.01774 | -0.04427 | 0.465 / 0.535 / 0.000 | 142.5 / -393.9 |
| nav|nav=left | 395 | 0.1294 | 0.01790 | 0.02477 | 0.527 / 0.473 / 0.000 | 28.2 / 51.1 |
| nav|nav=right | 1209 | -0.3940 | -0.11746 | -0.18928 | 0.401 / 0.599 / 0.000 | 98.5 / -476.4 |
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

### S2 -- the episodes carrying the summed delta (panel file index only)

**RAW-15k - refcv7-50400** (sum 600.8 over 139 episodes)

* ep111: n 34, share 0.132, mean delta 2.3369, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep108: n 23, share 0.061, mean delta 1.5901, sharp-window frac 0.00, curved frac 0.57, mean v0 9.3, nav {'nav=follow': 23, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share 0.061, mean delta 1.0409, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep104: n 35, share 0.054, mean delta 0.9351, sharp-window frac 0.29, curved frac 0.31, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 35, 'nav=right': 0}
* ep013: n 34, share 0.050, mean delta 0.8909, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep059: n 34, share 0.048, mean delta 0.8562, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**RAW-15k - RAW-10.5k** (sum -522.9 over 139 episodes)

* ep111: n 34, share -0.072, mean delta 1.1125, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep078: n 34, share -0.052, mean delta 0.7997, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep041: n 34, share -0.045, mean delta 0.6856, sharp-window frac 0.18, curved frac 0.47, mean v0 9.7, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep066: n 25, share -0.039, mean delta 0.8080, sharp-window frac 0.00, curved frac 0.00, mean v0 3.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 25}
* ep064: n 34, share -0.033, mean delta 0.5035, sharp-window frac 0.24, curved frac 0.56, mean v0 8.9, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep048: n 34, share -0.026, mean delta 0.3991, sharp-window frac 0.00, curved frac 0.53, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 34, 'nav=right': 0}
**RAW-10.5k - refcv7-50400** (sum 1123.3 over 139 episodes)

* ep129: n 31, share 0.081, mean delta 2.9473, sharp-window frac 0.32, curved frac 0.52, mean v0 4.7, nav {'nav=follow': 31, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share 0.053, mean delta 1.7012, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep063: n 34, share 0.049, mean delta 1.6043, sharp-window frac 0.26, curved frac 0.91, mean v0 6.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep067: n 34, share 0.043, mean delta 1.4235, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep111: n 34, share 0.037, mean delta 1.2244, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.034, mean delta 1.1297, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
**RAW-15k - RAW-5k** (sum -819.2 over 139 episodes)

* ep078: n 34, share -0.048, mean delta 1.1560, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep136: n 17, share -0.034, mean delta 1.6473, sharp-window frac 0.29, curved frac 0.35, mean v0 1.4, nav {'nav=follow': 0, 'nav=left': 17, 'nav=right': 0}
* ep104: n 34, share -0.034, mean delta 0.8128, sharp-window frac 0.29, curved frac 0.32, mean v0 8.0, nav {'nav=follow': 0, 'nav=left': 34, 'nav=right': 0}
* ep059: n 34, share -0.028, mean delta 0.6831, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep063: n 34, share -0.024, mean delta 0.5814, sharp-window frac 0.26, curved frac 0.91, mean v0 6.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep111: n 34, share -0.015, mean delta 0.3605, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
**RAW-5k - refcv7-50400** (sum 1407.7 over 139 episodes)

* ep116: n 33, share 0.067, mean delta 2.8389, sharp-window frac 0.00, curved frac 0.24, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 33}
* ep023: n 34, share 0.055, mean delta 2.2945, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep000: n 35, share 0.048, mean delta 1.9399, sharp-window frac 0.20, curved frac 0.46, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 35}
* ep011: n 34, share 0.048, mean delta 1.9770, sharp-window frac 0.00, curved frac 0.88, mean v0 6.6, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep111: n 34, share 0.048, mean delta 1.9764, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share 0.045, mean delta 1.8069, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW-15k vs RAW-10.5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1204 / -0.0657 [-0.1012, -0.0323] | 49 / -0.0607 [-0.4550, 0.3683] |
| only new agrees | 438 / -0.2874 [-0.4134, -0.1723] | 41 / -1.2790 [-2.3137, -0.4203] |
| only earlier agrees | 544 / 0.1086 [-0.0233, 0.2688] | 47 / 0.4153 [-0.5691, 1.9430] |
| neither agrees | 2300 / -0.1715 [-0.2833, -0.0612] | 227 / 0.1044 [-0.6427, 0.8411] |
**RAW-15k vs RAW-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 918 / -0.0469 [-0.0883, -0.0114] | 28 / -0.7018 [-1.3028, -0.2201] |
| only new agrees | 720 / -0.3749 [-0.5483, -0.2289] | 62 / -1.3479 [-2.1630, -0.6054] |
| only earlier agrees | 632 / 0.1631 [0.0414, 0.3152] | 35 / 0.4096 [-0.7413, 2.4690] |
| neither agrees | 2197 / -0.2658 [-0.4048, -0.1349] | 238 / -0.1888 [-0.6956, 0.3541] |
**RAW-15k vs RAW-10.5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1226 / -0.0800 [-0.1231, -0.0421] | 58 / -0.3692 [-0.9362, 0.1273] |
| only new agrees | 429 / -0.2805 [-0.4124, -0.1502] | 40 / -0.7989 [-1.6182, 0.0299] |
| only earlier agrees | 529 / 0.1668 [0.0361, 0.3296] | 38 / 0.9303 [-0.2383, 2.5054] |
| neither agrees | 2306 / -0.1640 [-0.2874, -0.0514] | 228 / -0.1295 [-1.0298, 0.7057] |
**RAW-15k vs RAW-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 939 / -0.0567 [-0.0939, -0.0217] | 33 / -0.5504 [-1.0115, -0.1642] |
| only new agrees | 711 / -0.4136 [-0.5964, -0.2514] | 65 / -1.6230 [-2.6638, -0.7348] |
| only earlier agrees | 615 / 0.1635 [0.0540, 0.3034] | 35 / 0.7086 [-0.3901, 2.4355] |
| neither agrees | 2204 / -0.2706 [-0.4049, -0.1508] | 230 / -0.3051 [-0.8482, 0.2328] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW-15k vs RAW-10.5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4457 | 0.2283 [0.1311, 0.3406] | 0.0930 [0.0237, 0.1837] | 0.1347 [0.0281, 0.2457] SEP |
| turn|straight <0.02 | 2870 | 0.1693 [0.0899, 0.2621] | 0.0793 [0.0243, 0.1447] | 0.0908 [0.0179, 0.1628] SEP |
| turn|gentle 0.02-0.06 | 717 | 0.2463 [0.0419, 0.4810] | 0.0171 [-0.0948, 0.1313] | 0.2287 [0.0165, 0.4888] SEP |
| turn|turn 0.06-0.15 | 499 | 0.2718 [0.0733, 0.5328] | 0.0742 [-0.1482, 0.3464] | 0.1883 [-0.1041, 0.5234] |
| turn|sharp >=0.15 | 363 | 0.6012 [0.0601, 1.2641] | 0.3752 [-0.0155, 0.8896] | 0.2260 [-0.2979, 0.7718] |
| turn|undef (GT crawling) | 8 | 0.1819 [-0.0526, 0.3707] | 0.2142 [-0.2017, 0.7107] | -0.0388 [-0.6822, 0.4170] |
| speed|v0 <2 | 168 | 0.0457 [-0.4019, 0.5788] | 0.0393 [-0.3812, 0.6017] | -0.0157 [-0.1869, 0.1973] |
| speed|v0 2-5 | 516 | 1.1304 [0.6125, 1.6597] | 0.4984 [0.0613, 1.0709] | 0.6320 [-0.0190, 1.2500] |
| speed|v0 5-10 | 1390 | 0.3218 [0.1762, 0.4912] | 0.1246 [0.0277, 0.2284] | 0.1973 [0.0372, 0.3704] SEP |
| speed|v0 10-15 | 1192 | -0.0089 [-0.0330, 0.0141] | -0.0090 [-0.0419, 0.0231] | 0.0002 [-0.0370, 0.0393] |
| speed|v0 15-25 | 856 | -0.0108 [-0.0384, 0.0103] | -0.0099 [-0.0229, 0.0009] | -0.0009 [-0.0212, 0.0168] |
| speed|v0 >=25 | 335 | 0.0013 [-0.0158, 0.0152] | -0.0077 [-0.0167, -0.0004] | 0.0090 [-0.0093, 0.0261] |
**RAW-15k vs RAW-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4448 | 0.2283 [0.1311, 0.3406] | 0.5097 [0.3343, 0.7032] | -0.2846 [-0.4455, -0.1311] SEP |
| turn|straight <0.02 | 2865 | 0.1693 [0.0899, 0.2621] | 0.4994 [0.2834, 0.7683] | -0.3312 [-0.5532, -0.1558] SEP |
| turn|gentle 0.02-0.06 | 716 | 0.2463 [0.0419, 0.4810] | 0.4875 [0.2402, 0.7901] | -0.2403 [-0.5531, 0.0702] |
| turn|turn 0.06-0.15 | 499 | 0.2718 [0.0733, 0.5328] | 0.3058 [0.0388, 0.6665] | -0.0560 [-0.3146, 0.1691] |
| turn|sharp >=0.15 | 363 | 0.6012 [0.0601, 1.2641] | 0.9252 [0.3435, 1.7333] | -0.3267 [-1.1427, 0.3828] |
| turn|undef (GT crawling) | 5 | 0.1819 [-0.0526, 0.3707] | -0.0099 [-0.0477, 0.0135] | 0.2634 [-0.0592, 0.5211] |
| speed|v0 <2 | 159 | 0.0457 [-0.4019, 0.5788] | 0.0840 [-0.3249, 0.5254] | -0.1043 [-0.5498, 0.3131] |
| speed|v0 2-5 | 516 | 1.1304 [0.6125, 1.6597] | 1.9375 [1.0587, 2.9928] | -0.8071 [-1.7100, 0.0079] |
| speed|v0 5-10 | 1390 | 0.3218 [0.1762, 0.4912] | 0.8818 [0.5827, 1.1816] | -0.5599 [-0.8321, -0.3005] SEP |
| speed|v0 10-15 | 1192 | -0.0089 [-0.0330, 0.0141] | -0.0042 [-0.0371, 0.0293] | -0.0046 [-0.0412, 0.0325] |
| speed|v0 15-25 | 856 | -0.0108 [-0.0384, 0.0103] | 0.0134 [-0.0080, 0.0330] | -0.0242 [-0.0539, 0.0037] |
| speed|v0 >=25 | 335 | 0.0013 [-0.0158, 0.0152] | 0.0864 [0.0064, 0.1941] | -0.0851 [-0.1873, -0.0104] SEP |
**RAW-15k vs RAW-10.5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
**RAW-15k vs RAW-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* RAW-15k vs RAW-10.5k|seed0: anchor flip rate 0.439; heading delta on flipped windows -0.1786 [-0.3027, -0.0574] (n 2036) vs unflipped -0.0722 [-0.1084, -0.0380]; goal_dist_sel proxy new-earlier -0.1648 [-0.5235, 0.1771]; corr(window dHeading, window dGoalDist) -0.023
* RAW-15k vs RAW-5k|seed0: anchor flip rate 0.589; heading delta on flipped windows -0.2565 [-0.3986, -0.1347] (n 2726) vs unflipped -0.0545 [-0.0888, -0.0231]; goal_dist_sel proxy new-earlier -0.1626 [-0.5577, 0.2555]; corr(window dHeading, window dGoalDist) -0.023
* RAW-15k vs RAW-10.5k|seed1: anchor flip rate 0.441; heading delta on flipped windows -0.1527 [-0.2844, -0.0272] (n 2040) vs unflipped -0.0803 [-0.1184, -0.0453]; goal_dist_sel proxy new-earlier -0.1742 [-0.5298, 0.1687]; corr(window dHeading, window dGoalDist) -0.018
* RAW-15k vs RAW-5k|seed1: anchor flip rate 0.586; heading delta on flipped windows -0.2684 [-0.4096, -0.1440] (n 2720) vs unflipped -0.0647 [-0.0990, -0.0292]; goal_dist_sel proxy new-earlier -0.1676 [-0.5665, 0.2497]; corr(window dHeading, window dGoalDist) -0.027

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- RAW-15k - refcv7-50400, 2-seed mean of the per-window metric

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

### S6 -- RAW-15k - RAW-10.5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3491 | 139 | 3.6147 | 3.6673 | -0.0526 [-0.2926, 0.1714] |  |  |
| turn: straight <0.02 | 2082 | 122 | 1.7713 | 1.6421 | 0.1292 [-0.0052, 0.2690] |  | 0.596 / -1.464 / 0.4503 [-0.3521, 1.3029] |
| turn: gentle 0.02-0.06 | 673 | 77 | 3.2515 | 2.9777 | 0.2738 [-0.2049, 0.8664] |  | 0.193 / -1.004 / 0.4044 [-0.3320, 1.4550] |
| turn: turn 0.06-0.15 | 426 | 59 | 6.2446 | 6.7839 | -0.5393 [-1.0842, -0.0984] | **SEP** | 0.122 / 1.251 / -0.5543 [-1.3202, 0.0834] |
| turn: sharp >=0.15 | 301 | 33 | 12.9515 | 14.3060 | -1.3545 [-3.6405, 0.3720] |  | 0.086 / 2.220 / -1.4247 [-5.0367, 1.2188] |
| turn: undef (GT crawling) (low n) | 9 | 8 | 20.4690 | 20.3986 | 0.0704 [-0.2910, 0.4933] |  | 0.003 / -0.003 / 0.1233 [-0.6241, 1.0049] |
| speed: v0 <2 | 154 | 24 | 13.4025 | 13.6638 | -0.2613 [-1.7836, 1.0912] |  | 0.044 / 0.219 / -0.2183 [-2.7472, 1.7675] |
| speed: v0 2-5 | 421 | 52 | 5.2156 | 5.6744 | -0.4588 [-1.4684, 0.4988] |  | 0.121 / 1.052 / -0.4619 [-1.7992, 0.9975] |
| speed: v0 5-10 | 1040 | 79 | 4.3652 | 4.4047 | -0.0395 [-0.6099, 0.4815] |  | 0.298 / 0.223 / 0.0187 [-0.7903, 0.9234] |
| speed: v0 10-15 | 952 | 62 | 2.7466 | 2.6578 | 0.0888 [-0.0761, 0.2481] |  | 0.273 / -0.460 / 0.1944 [-0.3512, 0.7270] |
| speed: v0 15-25 | 679 | 34 | 1.5909 | 1.5548 | 0.0361 [-0.0518, 0.1207] |  | 0.195 / -0.133 / 0.1101 [-0.3375, 0.5698] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5072 | 0.5817 | -0.0744 [-0.2473, 0.0905] |  | 0.070 / 0.099 / -0.0235 [-0.4629, 0.4456] |
| nav: nav=follow | 2237 | 88 | 2.5335 | 2.4982 | 0.0353 [-0.2219, 0.2361] |  | 0.641 / -0.430 / 0.2446 [-0.5452, 1.1118] |
| nav: nav=left (low n) | 311 | 13 | 8.8199 | 8.7338 | 0.0860 [-1.1214, 1.2298] |  | 0.089 / -0.146 / 0.1522 [-1.9440, 1.9477] |
| nav: nav=right | 943 | 38 | 4.4628 | 4.7696 | -0.3067 [-0.8756, 0.3154] |  | 0.270 / 1.575 / -0.3482 [-1.3329, 0.6823] |

episode concentration: 0.547 of episodes have mean delta > 0; top-5 episodes carry -2.581 of the summed delta; delta excluding the top 5 = -0.1951 (overall -0.0526).

### S6 -- RAW-10.5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3511 | 139 | 3.7336 | 3.1706 | 0.5631 [0.3375, 0.8011] | **SEP** |  |
| turn: straight <0.02 | 2086 | 122 | 1.7267 | 1.3293 | 0.3975 [0.2629, 0.5502] | **SEP** | 0.594 / 0.419 / -0.4081 [-1.3179, 0.3863] |
| turn: gentle 0.02-0.06 | 674 | 77 | 2.9837 | 2.8783 | 0.1054 [-0.4434, 0.5215] |  | 0.192 / 0.036 / -0.5664 [-1.5460, 0.0563] |
| turn: turn 0.06-0.15 | 434 | 59 | 6.8110 | 6.3471 | 0.4639 [0.0988, 0.8666] | **SEP** | 0.124 / 0.102 / -0.1131 [-0.8723, 0.5688] |
| turn: sharp >=0.15 | 304 | 33 | 14.3032 | 11.4481 | 2.8551 [0.9590, 5.2476] | **SEP** | 0.087 / 0.439 / 2.5093 [-0.1598, 6.5950] |
| turn: undef (GT crawling) (low n) | 13 | 10 | 14.7431 | 14.1627 | 0.5804 [0.0462, 1.5432] | **SEP** | 0.004 / 0.004 / 0.0173 [-0.7966, 1.8247] |
| speed: v0 <2 | 173 | 24 | 13.9263 | 13.1222 | 0.8040 [0.2852, 1.4340] | **SEP** | 0.049 / 0.070 / 0.2534 [-0.7671, 1.3458] |
| speed: v0 2-5 | 422 | 52 | 5.6641 | 4.1501 | 1.5140 [0.3855, 2.6250] | **SEP** | 0.120 / 0.323 / 1.0808 [-0.9091, 2.6367] |
| speed: v0 5-10 | 1040 | 79 | 4.4047 | 3.4941 | 0.9106 [0.4313, 1.5554] | **SEP** | 0.296 / 0.479 / 0.4938 [-0.2718, 1.4154] |
| speed: v0 10-15 | 952 | 62 | 2.6578 | 2.4870 | 0.1707 [-0.0309, 0.3958] |  | 0.271 / 0.082 / -0.5383 [-1.0936, -0.0051] BONF-SEP |
| speed: v0 15-25 | 679 | 34 | 1.5548 | 1.4456 | 0.1093 [-0.0030, 0.2302] |  | 0.193 / 0.038 / -0.5626 [-1.0324, -0.1655] BONF-SEP |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5817 | 0.5193 | 0.0624 [-0.0227, 0.1576] |  | 0.070 / 0.008 / -0.5383 [-0.9939, -0.1827] BONF-SEP |
| nav: nav=follow | 2241 | 88 | 2.4988 | 1.9818 | 0.5171 [0.2887, 0.8133] | **SEP** | 0.638 / 0.586 / -0.1273 [-0.9217, 0.6294] |
| nav: nav=left (low n) | 314 | 13 | 9.3336 | 8.9486 | 0.3850 [-0.1456, 0.9256] |  | 0.089 / 0.061 / -0.1956 [-1.1799, 0.7327] |
| nav: nav=right | 956 | 38 | 4.7889 | 4.0594 | 0.7295 [0.1265, 1.2501] | **SEP** | 0.272 / 0.353 / 0.2287 [-0.6769, 1.1776] |

episode concentration: 0.763 of episodes have mean delta > 0; top-5 episodes carry 0.369 of the summed delta; delta excluding the top 5 = 0.3689 (overall 0.5631).

### S6 -- RAW-15k - RAW-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3468 | 139 | 3.4682 | 3.7051 | -0.2369 [-0.4492, -0.0184] | **SEP** |  |
| turn: straight <0.02 | 2064 | 122 | 1.6670 | 1.6347 | 0.0323 [-0.1367, 0.2095] |  | 0.595 / -0.081 / 0.6648 [-0.0380, 1.3005] |
| turn: gentle 0.02-0.06 | 672 | 76 | 3.0315 | 3.1172 | -0.0857 [-0.6145, 0.5802] |  | 0.194 / 0.070 / 0.1875 [-0.5968, 1.4102] |
| turn: turn 0.06-0.15 | 424 | 59 | 5.8329 | 7.0946 | -1.2617 [-1.7844, -0.8279] | **SEP** | 0.122 / 0.651 / -1.1676 [-2.1372, -0.4881] BONF-SEP |
| turn: sharp >=0.15 | 300 | 33 | 12.9756 | 13.9706 | -0.9951 [-2.3267, 0.1911] |  | 0.087 / 0.363 / -0.8300 [-2.8748, 0.7955] |
| turn: undef (GT crawling) (low n) | 8 | 7 | 23.0067 | 22.6348 | 0.3720 [-0.1570, 0.9953] |  | 0.002 / -0.004 / 0.6103 [-0.3675, 1.7595] |
| speed: v0 <2 | 131 | 23 | 11.2432 | 10.8287 | 0.4144 [-0.3346, 1.0557] |  | 0.038 / -0.066 / 0.6769 [-0.5566, 1.7003] |
| speed: v0 2-5 | 421 | 52 | 5.2156 | 5.0417 | 0.1739 [-0.8429, 1.2185] |  | 0.121 / -0.089 / 0.4675 [-1.0629, 2.1614] |
| speed: v0 5-10 | 1040 | 79 | 4.3652 | 4.8774 | -0.5122 [-0.9649, -0.0507] | **SEP** | 0.300 / 0.648 / -0.3932 [-1.0592, 0.3573] |
| speed: v0 10-15 | 952 | 62 | 2.7466 | 3.0719 | -0.3253 [-0.5780, -0.0900] | **SEP** | 0.275 / 0.377 / -0.1219 [-0.7102, 0.4408] |
| speed: v0 15-25 | 679 | 34 | 1.5909 | 1.7173 | -0.1264 [-0.3059, 0.0512] |  | 0.196 / 0.104 / 0.1374 [-0.3199, 0.6160] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.5072 | 0.5923 | -0.0850 [-0.2236, 0.0240] |  | 0.071 / 0.025 / 0.1634 [-0.2400, 0.5310] |
| nav: nav=follow | 2224 | 88 | 2.5351 | 2.6359 | -0.1008 [-0.2918, 0.0885] |  | 0.641 / 0.273 / 0.3793 [-0.4829, 1.1885] |
| nav: nav=left (low n) | 307 | 13 | 7.1373 | 7.0275 | 0.1098 [-0.5127, 0.7412] |  | 0.089 / -0.041 / 0.3803 [-0.8259, 1.4182] |
| nav: nav=right | 937 | 38 | 4.4808 | 5.1542 | -0.6734 [-1.2211, 0.0113] |  | 0.270 / 0.768 / -0.5981 [-1.4923, 0.5113] |

episode concentration: 0.424 of episodes have mean delta > 0; top-5 episodes carry -0.524 of the summed delta; delta excluding the top 5 = -0.3754 (overall -0.2369).

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

### S6 -- RAW-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

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

### S6 -- RAW-15k - RAW-10.5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3466 | 139 | 0.9816 | 1.0992 | -0.1176 [-0.2315, -0.0382] | **SEP** | -0.0218 [-0.1557, 0.0796] |
| seg1 | straight | 2060 | 122 | 0.6911 | 0.8297 | -0.1386 [-0.3363, -0.0337] | **SEP** | -0.1782 [-0.3804, -0.0667] SEP |
| seg1 | curved | 1397 | 88 | 1.2845 | 1.3731 | -0.0886 [-0.1731, -0.0024] | **SEP** | 0.2106 [0.0729, 0.3586] SEP |
| seg2 | all | 3450 | 139 | 1.5780 | 1.8407 | -0.2627 [-0.5065, -0.0734] | **SEP** | 0.0735 [-0.2155, 0.3156] |
| seg2 | straight | 2056 | 122 | 0.8844 | 1.1570 | -0.2726 [-0.6599, -0.0670] | **SEP** | -0.2648 [-0.6593, -0.0435] SEP |
| seg2 | curved | 1394 | 88 | 2.6011 | 2.8492 | -0.2481 [-0.5280, 0.0355] |  | 0.5725 [0.1875, 0.9702] SEP |
| seg3 | all | 3407 | 138 | 2.3802 | 2.4916 | -0.1114 [-0.3686, 0.1226] |  | 0.4499 [-0.2390, 0.9381] |
| seg3 | straight | 2034 | 121 | 1.3424 | 1.3057 | 0.0367 [-0.1584, 0.1894] |  | -0.0477 [-1.0171, 0.4736] |
| seg3 | curved | 1373 | 87 | 3.9176 | 4.2484 | -0.3309 [-0.9154, 0.1983] |  | 1.1870 [0.5341, 1.8293] SEP |
| seg4 | all | 3355 | 138 | 3.5336 | 3.6033 | -0.0697 [-0.4652, 0.2816] |  | 0.9331 [-0.1640, 1.6432] |
| seg4 | straight | 2014 | 119 | 1.7217 | 1.5352 | 0.1864 [-0.2175, 0.4767] |  | 0.4037 [-1.2234, 1.2634] |
| seg4 | curved | 1341 | 86 | 6.2549 | 6.7093 | -0.4544 [-1.2321, 0.2852] |  | 1.7283 [0.8940, 2.5403] SEP |
| seg5 | all | 3310 | 137 | 5.1442 | 5.0984 | 0.0458 [-0.3616, 0.4394] |  | 1.5235 [0.1808, 2.4332] SEP |
| seg5 | straight | 1991 | 118 | 2.3094 | 1.9096 | 0.3998 [0.1708, 0.6372] | **SEP** | 0.9522 [-1.1004, 2.0768] |
| seg5 | curved | 1319 | 86 | 9.4232 | 9.9118 | -0.4887 [-1.4294, 0.4032] |  | 2.3859 [1.3152, 3.3695] SEP |
| seg6 | all | 3268 | 137 | 6.7879 | 6.7966 | -0.0086 [-0.4572, 0.4575] |  | 2.2602 [1.3650, 3.0178] SEP |
| seg6 | straight | 1977 | 118 | 2.6196 | 2.3931 | 0.2266 [-0.1433, 0.5790] |  | 1.9736 [0.6634, 2.8533] SEP |
| seg6 | curved | 1291 | 87 | 13.1711 | 13.5400 | -0.3688 [-1.3476, 0.6324] |  | 2.6990 [1.4931, 3.7634] SEP |

### S6 -- RAW-10.5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3480 | 139 | 1.1506 | 0.7653 | 0.3854 [0.0938, 0.9277] | **SEP** | 0.3743 [-0.0381, 1.1332] |
| seg1 | straight | 2062 | 122 | 0.9168 | 0.5448 | 0.3719 [0.0857, 0.9573] | **SEP** | 0.5078 [0.0514, 1.4708] SEP |
| seg1 | curved | 1406 | 88 | 1.3718 | 0.9645 | 0.4073 [0.0803, 0.9999] | **SEP** | 0.1838 [-0.1880, 0.7696] |
| seg2 | all | 3466 | 139 | 1.9012 | 1.4087 | 0.4925 [0.2403, 0.8673] | **SEP** | 0.4803 [-0.1230, 1.4837] |
| seg2 | straight | 2063 | 122 | 1.2487 | 0.7538 | 0.4949 [0.1917, 1.0766] | **SEP** | 0.7743 [0.1159, 2.1043] SEP |
| seg2 | curved | 1403 | 88 | 2.8606 | 2.3717 | 0.4890 [0.1981, 0.7813] | **SEP** | 0.0480 [-0.5499, 0.7200] |
| seg3 | all | 3426 | 139 | 2.5648 | 2.0581 | 0.5067 [0.2826, 0.7461] | **SEP** | 0.3682 [-0.3271, 1.3832] |
| seg3 | straight | 2041 | 121 | 1.4025 | 0.9792 | 0.4232 [0.1887, 0.8302] | **SEP** | 0.8187 [0.0502, 2.3379] SEP |
| seg3 | curved | 1385 | 88 | 4.2776 | 3.6479 | 0.6297 [0.0666, 1.1926] | **SEP** | -0.2957 [-1.0929, 0.4049] |
| seg4 | all | 3377 | 139 | 3.6874 | 2.9112 | 0.7762 [0.3810, 1.2514] | **SEP** | -0.0330 [-0.6792, 0.8433] |
| seg4 | straight | 2021 | 120 | 1.6245 | 1.0523 | 0.5722 [0.2269, 1.2169] | **SEP** | 0.4644 [-0.2545, 1.8635] |
| seg4 | curved | 1356 | 87 | 6.7619 | 5.6818 | 1.0801 [0.3836, 1.8209] | **SEP** | -0.7743 [-1.7383, 0.0953] |
| seg5 | all | 3333 | 138 | 5.1933 | 4.2949 | 0.8984 [0.4720, 1.3812] | **SEP** | -0.4058 [-1.3236, 0.9896] |
| seg5 | straight | 1996 | 119 | 2.0115 | 1.3089 | 0.7026 [0.3757, 1.1652] | **SEP** | 0.3291 [-0.7292, 2.3597] |
| seg5 | curved | 1337 | 87 | 9.9433 | 8.7526 | 1.1908 [0.3253, 2.1043] | **SEP** | -1.5029 [-2.6410, -0.3844] SEP |
| seg6 | all | 3288 | 138 | 6.8823 | 5.9184 | 0.9639 [0.5262, 1.3945] | **SEP** | -1.2137 [-2.0706, -0.0696] SEP |
| seg6 | straight | 1983 | 119 | 2.4841 | 1.7271 | 0.7571 [0.4751, 1.0723] | **SEP** | -0.3934 [-1.3444, 1.2625] |
| seg6 | curved | 1304 | 88 | 13.5718 | 12.2965 | 1.2753 [0.3069, 2.2408] | **SEP** | -2.4578 [-3.6396, -1.2375] SEP |

### S6 -- RAW-15k - RAW-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3459 | 139 | 0.7758 | 0.9769 | -0.2010 [-0.4167, -0.0711] | **SEP** | 0.1732 [0.1216, 0.2283] SEP |
| seg1 | straight | 2058 | 122 | 0.6048 | 0.7461 | -0.1413 [-0.3458, -0.0299] | **SEP** | 0.0510 [-0.1559, 0.1707] |
| seg1 | curved | 1393 | 88 | 0.9009 | 1.1937 | -0.2928 [-0.5742, -0.1145] | **SEP** | 0.3590 [0.1621, 0.6424] SEP |
| seg2 | all | 3434 | 139 | 1.4807 | 1.7288 | -0.2481 [-0.3763, -0.1294] | **SEP** | 0.4545 [0.3101, 0.6046] SEP |
| seg2 | straight | 2043 | 122 | 0.8894 | 1.0066 | -0.1171 [-0.2093, -0.0331] | **SEP** | 0.3980 [0.3010, 0.5127] SEP |
| seg2 | curved | 1391 | 88 | 2.3491 | 2.7896 | -0.4405 [-0.6975, -0.1742] | **SEP** | 0.5375 [0.2367, 0.8304] SEP |
| seg3 | all | 3372 | 138 | 2.2626 | 2.6188 | -0.3562 [-0.6558, -0.0946] | **SEP** | 0.7542 [0.1243, 1.1711] SEP |
| seg3 | straight | 2006 | 121 | 1.1734 | 1.3415 | -0.1682 [-0.6064, 0.1089] |  | 0.5154 [-0.3175, 1.0212] |
| seg3 | curved | 1366 | 87 | 3.8622 | 4.4944 | -0.6322 [-1.0299, -0.1980] | **SEP** | 1.1049 [0.5857, 1.5600] SEP |
| seg4 | all | 3290 | 138 | 3.4668 | 3.7066 | -0.2397 [-0.5307, 0.0586] |  | 1.0902 [0.4042, 1.5746] SEP |
| seg4 | straight | 1979 | 119 | 1.6344 | 1.5233 | 0.1111 [-0.1049, 0.3613] |  | 1.0021 [0.0882, 1.5818] SEP |
| seg4 | curved | 1311 | 86 | 6.2329 | 7.0022 | -0.7693 [-1.3150, -0.1697] | **SEP** | 1.2230 [0.4971, 1.8789] SEP |
| seg5 | all | 3233 | 138 | 4.9160 | 5.1859 | -0.2699 [-0.6849, 0.1622] |  | 1.5436 [1.0722, 1.9812] SEP |
| seg5 | straight | 1958 | 118 | 2.0306 | 1.7247 | 0.3059 [-0.0642, 0.7141] |  | 1.6248 [1.2117, 2.0283] SEP |
| seg5 | curved | 1275 | 86 | 9.3469 | 10.5012 | -1.1542 [-1.9400, -0.2582] | **SEP** | 1.4189 [0.4887, 2.3020] SEP |
| seg6 | all | 3171 | 136 | 6.5204 | 6.9434 | -0.4230 [-0.9492, 0.1040] |  | 1.8084 [1.2849, 2.3510] SEP |
| seg6 | straight | 1943 | 117 | 2.3999 | 2.1549 | 0.2450 [-0.1642, 0.6810] |  | 1.9463 [1.4951, 2.4207] SEP |
| seg6 | curved | 1228 | 85 | 13.0401 | 14.5201 | -1.4800 [-2.5765, -0.4235] | **SEP** | 1.5904 [0.3909, 2.7427] SEP |

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

### S6 -- RAW-15k - refcv7-50400: turn x speed cross

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

| bin | RAW-15k s0-s1 [CI] | RAW-10.5k s0-s1 [CI] | RAW-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|---|
| overall | 0.0249 [-0.0363, 0.0991] | 0.0063 [-0.0750, 0.0931] | -0.0092 [-0.0548, 0.0335] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0156 [-0.0464, 0.0131] | 0.0571 [-0.0171, 0.1659] | -0.0222 [-0.0586, 0.0109] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0576 [-0.0955, 0.2915] | 0.0742 [-0.0416, 0.2042] | -0.0331 [-0.1566, 0.0983] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | 0.0292 [-0.1440, 0.2320] | -0.0482 [-0.2489, 0.1601] | -0.0294 [-0.1727, 0.0941] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.2264 [-0.1064, 0.6622] | -0.4293 [-0.8584, -0.1048] SEP | 0.1538 [-0.1558, 0.4832] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.0419 [-0.0728, 0.1863] | 0.3347 [-0.1234, 0.9280] | 0.2483 [-0.0632, 0.7205] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | 0.6156 [-0.0537, 1.8145] | -0.1800 [-0.5844, 0.0899] | 0.1279 [-0.1377, 0.5456] | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | 0.1901 [-0.0045, 0.4643] | 0.0450 [-0.1339, 0.2487] | -0.0357 [-0.1747, 0.1170] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | -0.0779 [-0.2361, 0.0598] | 0.0715 [-0.1215, 0.3061] | -0.0247 [-0.1237, 0.1011] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0155 [-0.0658, 0.0366] | -0.0587 [-0.1108, -0.0085] SEP | 0.0083 [-0.0517, 0.0713] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | 0.0015 [-0.0467, 0.0469] | 0.0268 [-0.0094, 0.0634] | -0.0203 [-0.0677, 0.0198] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | -0.0026 [-0.0295, 0.0199] | -0.0076 [-0.0276, 0.0144] | -0.0163 [-0.0532, 0.0239] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0308 [-0.0652, 0.0006] | -0.0183 [-0.0986, 0.0452] | -0.0053 [-0.0471, 0.0425] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.3547 [-0.0812, 1.1057] | -0.1089 [-0.5018, 0.2463] | 0.1081 [-0.0251, 0.3113] | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | 0.0471 [-0.0703, 0.1675] | 0.1020 [-0.0707, 0.3570] | -0.0566 [-0.1710, 0.0566] | 0.0235 [-0.1175, 0.1775] |

### S6 -- RAW-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

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

### S6 -- RAW-15k - RAW-10.5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0526 | -0.0529 / -0.0630 | False / False / False | 0.0249 / 0.0063 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.1292 | 0.0748 / 0.1666 | False / True / False | 0.0156 / 0.0571 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.2738 | 0.2621 / 0.2815 | False / False / False | 0.0576 / 0.0742 | 3.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.5393 | -0.4963 / -0.5823 | True / True / True | 0.0292 / 0.0482 | 11.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -1.3545 | -1.0166 / -1.6924 | False / False / False | 0.2264 / 0.4293 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.0704 | 0.0245 / 0.0857 | False / False / False | 0.0419 / 0.3347 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.2613 | -0.0442 / -0.7130 | False / False / False | 0.6156 / 0.1800 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.4588 | -0.3863 / -0.5313 | False / False / False | 0.1901 / 0.0450 | 2.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.0395 | -0.1142 / 0.0353 | False / False / False | 0.0779 / 0.0715 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0888 | 0.1104 / 0.0672 | False / False / False | 0.0155 / 0.0587 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0361 | 0.0234 / 0.0487 | False / False / False | 0.0015 / 0.0268 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0744 | -0.0719 / -0.0769 | False / False / False | 0.0026 / 0.0076 | 9.8 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0353 | 0.0297 / 0.0408 | False / False / False | 0.0308 / 0.0183 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | 0.0860 | 0.1990 / -0.1396 | False / False / False | 0.3547 / 0.1089 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.3067 | -0.3319 / -0.2838 | False / False / False | 0.0471 / 0.1020 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- RAW-10.5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.5631 | 0.5670 / 0.5646 | True / True / True | 0.0063 / 0.0128 | 44.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.3975 | 0.4575 / 0.3551 | True / True / True | 0.0571 / 0.0216 | 7.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.1054 | 0.1147 / 0.0961 | False / False / False | 0.0742 / 0.0713 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.4639 | 0.4082 / 0.5159 | True / True / True | 0.0482 / 0.0611 | 7.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | 2.8551 | 2.5398 / 3.1102 | True / True / True | 0.4293 / 0.2004 | 6.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.5804 | 0.7631 / 0.3976 | True / False / True | 0.3347 / 1.9278 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.8040 | 0.8334 / 0.8776 | True / True / True | 0.1800 / 0.0328 | 4.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 2-5 | 1.5140 | 1.4360 / 1.5920 | True / True / True | 0.0450 / 0.2009 | 7.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 5-10 | 0.9106 | 0.9585 / 0.8627 | True / True / True | 0.0715 / 0.0243 | 12.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1707 | 0.1548 / 0.1867 | False / False / False | 0.0587 / 0.0268 | 2.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.1093 | 0.1199 / 0.0986 | True / False / False | 0.0268 / 0.0055 | 4.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0624 | 0.0581 / 0.0666 | False / False / False | 0.0076 / 0.0008 | 8.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.5171 | 0.5047 / 0.5304 | True / True / True | 0.0183 / 0.0080 | 28.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.3850 | 0.4678 / 0.4119 | True / False / False | 0.1089 / 0.0140 | 3.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.7295 | 0.7457 / 0.6947 | True / True / True | 0.1020 / 0.0235 | 7.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

### S6 -- RAW-15k - RAW-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.2369 | -0.2325 / -0.2397 | True / True / True | 0.0249 / 0.0092 | 9.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0323 | 0.0349 / 0.0307 | False / False / False | 0.0156 / 0.0222 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.0857 | -0.0838 / -0.0832 | False / False / False | 0.0576 / 0.0331 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -1.2617 | -1.2676 / -1.2557 | True / True / True | 0.0292 / 0.0294 | 42.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -0.9951 | -0.9547 / -1.0355 | False / False / False | 0.2264 / 0.1538 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.3720 | 0.1890 / 0.5081 | False / False / False | 0.0419 / 0.2483 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.4144 | 0.3923 / 0.4645 | False / False / False | 0.6156 / 0.1279 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1739 | 0.2869 / 0.0610 | False / False / False | 0.1901 / 0.0357 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.5122 | -0.5388 / -0.4855 | True / False / True | 0.0779 / 0.0247 | 6.6 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 10-15 | -0.3253 | -0.3372 / -0.3134 | True / True / True | 0.0155 / 0.0083 | 21.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.1264 | -0.1155 / -0.1373 | False / False / False | 0.0015 / 0.0203 | 6.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0850 | -0.0781 / -0.0919 | False / False / False | 0.0026 / 0.0163 | 5.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.1008 | -0.1141 / -0.0865 | False / False / False | 0.0308 / 0.0053 | 3.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | 0.1098 | 0.0971 / 0.1225 | False / False / False | 0.3547 / 0.1081 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.6734 | -0.6208 / -0.7215 | False / True / False | 0.0471 / 0.0566 | 11.9 | NOT PROVEN (not separated and/or within 2x floor) |

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

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1393): RAW-15k bias -0.121, refcv7-50400 bias +0.049, RAW-10.5k bias +0.195, RAW-5k bias -0.253; RAW-15k MAE 0.901, refcv7-50400 MAE 0.835, RAW-10.5k MAE 1.118, RAW-5k MAE 1.194
* seg2 (n 1391): RAW-15k bias -0.727, refcv7-50400 bias -0.200, RAW-10.5k bias -0.306, RAW-5k bias -0.636; RAW-15k MAE 2.349, refcv7-50400 MAE 2.100, RAW-10.5k MAE 2.598, RAW-5k MAE 2.790
* seg3 (n 1363): RAW-15k bias -1.898, refcv7-50400 bias -0.852, RAW-10.5k bias -1.159, RAW-5k bias -1.383; RAW-15k MAE 3.866, refcv7-50400 MAE 3.420, RAW-10.5k MAE 4.186, RAW-5k MAE 4.482
* seg4 (n 1303): RAW-15k bias -3.738, refcv7-50400 bias -2.335, RAW-10.5k bias -2.709, RAW-5k bias -2.844; RAW-15k MAE 6.232, refcv7-50400 MAE 5.589, RAW-10.5k MAE 6.684, RAW-5k MAE 7.009
* seg5 (n 1266): RAW-15k bias -5.739, refcv7-50400 bias -4.229, RAW-10.5k bias -4.503, RAW-5k bias -4.385; RAW-15k MAE 9.380, refcv7-50400 MAE 8.682, RAW-10.5k MAE 9.919, RAW-5k MAE 10.539
* seg6 (n 1220): RAW-15k bias -7.888, refcv7-50400 bias -6.640, RAW-10.5k bias -5.828, RAW-5k bias -5.866; RAW-15k MAE 13.094, refcv7-50400 MAE 12.218, RAW-10.5k MAE 13.586, RAW-5k MAE 14.573
**sharp (>= 0.15)**

* seg1 (n 300): RAW-15k bias -0.434, refcv7-50400 bias +0.188, RAW-10.5k bias +0.541, RAW-5k bias -1.277; RAW-15k MAE 1.497, refcv7-50400 MAE 1.441, RAW-10.5k MAE 2.334, RAW-5k MAE 2.311
* seg2 (n 296): RAW-15k bias -2.533, refcv7-50400 bias -0.481, RAW-10.5k bias -1.824, RAW-5k bias -3.078; RAW-15k MAE 4.563, refcv7-50400 MAE 3.791, RAW-10.5k MAE 5.216, RAW-5k MAE 5.274
* seg3 (n 291): RAW-15k bias -6.101, refcv7-50400 bias -2.135, RAW-10.5k bias -5.213, RAW-5k bias -6.196; RAW-15k MAE 8.154, refcv7-50400 MAE 6.356, RAW-10.5k MAE 9.359, RAW-5k MAE 9.281
* seg4 (n 285): RAW-15k bias -11.253, refcv7-50400 bias -5.975, RAW-10.5k bias -9.863, RAW-5k bias -10.411; RAW-15k MAE 13.709, refcv7-50400 MAE 11.325, RAW-10.5k MAE 15.608, RAW-5k MAE 15.262
* seg5 (n 280): RAW-15k bias -16.287, refcv7-50400 bias -10.762, RAW-10.5k bias -14.322, RAW-5k bias -14.607; RAW-15k MAE 20.597, refcv7-50400 MAE 18.291, RAW-10.5k MAE 22.985, RAW-5k MAE 23.053
* seg6 (n 274): RAW-15k bias -21.040, refcv7-50400 bias -16.363, RAW-10.5k bias -17.049, RAW-5k bias -17.873; RAW-15k MAE 27.153, refcv7-50400 MAE 24.252, RAW-10.5k MAE 29.030, RAW-5k MAE 30.101
**curved & v0 5-10**

* seg1 (n 547): RAW-15k bias -0.158, refcv7-50400 bias +0.049, RAW-10.5k bias +0.081, RAW-5k bias -0.163; RAW-15k MAE 1.046, refcv7-50400 MAE 0.896, RAW-10.5k MAE 1.159, RAW-5k MAE 1.351
* seg2 (n 547): RAW-15k bias -1.115, refcv7-50400 bias -0.516, RAW-10.5k bias -0.599, RAW-5k bias -0.973; RAW-15k MAE 2.744, refcv7-50400 MAE 2.231, RAW-10.5k MAE 3.062, RAW-5k MAE 3.637
* seg3 (n 534): RAW-15k bias -2.899, refcv7-50400 bias -1.665, RAW-10.5k bias -1.908, RAW-5k bias -1.956; RAW-15k MAE 4.767, refcv7-50400 MAE 3.811, RAW-10.5k MAE 5.056, RAW-5k MAE 5.671
* seg4 (n 488): RAW-15k bias -5.457, refcv7-50400 bias -3.723, RAW-10.5k bias -3.974, RAW-5k bias -3.860; RAW-15k MAE 7.616, refcv7-50400 MAE 6.385, RAW-10.5k MAE 8.004, RAW-5k MAE 8.383
* seg5 (n 462): RAW-15k bias -7.770, refcv7-50400 bias -5.914, RAW-10.5k bias -6.147, RAW-5k bias -5.180; RAW-15k MAE 11.000, refcv7-50400 MAE 9.597, RAW-10.5k MAE 11.385, RAW-5k MAE 12.455
* seg6 (n 429): RAW-15k bias -9.592, refcv7-50400 bias -7.965, RAW-10.5k bias -6.501, RAW-5k bias -5.599; RAW-15k MAE 14.480, refcv7-50400 MAE 12.756, RAW-10.5k MAE 14.526, RAW-5k MAE 16.761
**sharp & v0 5-10**

* seg1 (n 139): RAW-15k bias -0.572, refcv7-50400 bias +0.062, RAW-10.5k bias -0.023, RAW-5k bias -0.881; RAW-15k MAE 1.400, refcv7-50400 MAE 1.251, RAW-10.5k MAE 1.725, RAW-5k MAE 1.740
* seg2 (n 139): RAW-15k bias -3.441, refcv7-50400 bias -1.267, RAW-10.5k bias -2.349, RAW-5k bias -4.116; RAW-15k MAE 4.920, refcv7-50400 MAE 3.564, RAW-10.5k MAE 5.510, RAW-5k MAE 5.859
* seg3 (n 138): RAW-15k bias -8.574, refcv7-50400 bias -4.253, RAW-10.5k bias -7.021, RAW-5k bias -8.326; RAW-15k MAE 9.982, refcv7-50400 MAE 6.990, RAW-10.5k MAE 10.446, RAW-5k MAE 10.584
* seg4 (n 133): RAW-15k bias -15.057, refcv7-50400 bias -9.164, RAW-10.5k bias -12.610, RAW-5k bias -12.862; RAW-15k MAE 16.597, refcv7-50400 MAE 12.544, RAW-10.5k MAE 16.841, RAW-5k MAE 16.760
* seg5 (n 131): RAW-15k bias -20.122, refcv7-50400 bias -13.884, RAW-10.5k bias -17.362, RAW-5k bias -15.570; RAW-15k MAE 23.997, refcv7-50400 MAE 19.504, RAW-10.5k MAE 24.488, RAW-5k MAE 25.089
* seg6 (n 125): RAW-15k bias -22.468, refcv7-50400 bias -17.289, RAW-10.5k bias -16.493, RAW-5k bias -15.273; RAW-15k MAE 29.444, refcv7-50400 MAE 24.947, RAW-10.5k MAE 29.128, RAW-5k MAE 31.612

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1220 / 1220 | 26.7 | RAW-15k -0.2959 [-0.3843, -0.2132]; refcv7-50400 -0.2331 [-0.3031, -0.1667]; RAW-10.5k -0.2286 [-0.3321, -0.1304]; RAW-5k -0.2298 [-0.3268, -0.1413] | RAW-15k -0.0628 [-0.1227, -0.0065] SEP; RAW-10.5k +0.0046 [-0.0578, +0.0605]; RAW-5k +0.0034 [-0.0570, +0.0556] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1395 / 6543 | 16.3 | RAW-15k -0.2512 [-0.3377, -0.1754]; refcv7-50400 -0.1630 [-0.2264, -0.1054]; RAW-10.5k -0.1955 [-0.2972, -0.1057]; RAW-5k -0.2058 [-0.3030, -0.1223] | RAW-15k -0.0882 [-0.1467, -0.0351] SEP; RAW-10.5k -0.0325 [-0.1012, +0.0247]; RAW-5k -0.0428 [-0.1039, +0.0091] |
| sharp (>= 0.15) | last segment | 274 / 274 | 63.3 | RAW-15k -0.3312 [-0.4402, -0.2364]; refcv7-50400 -0.2486 [-0.3315, -0.1736]; RAW-10.5k -0.2672 [-0.3997, -0.1569]; RAW-5k -0.2754 [-0.4032, -0.1727] | RAW-15k -0.0826 [-0.1638, -0.0107] SEP; RAW-10.5k -0.0186 [-0.1034, +0.0522]; RAW-5k -0.0268 [-0.1068, +0.0406] |
| sharp (>= 0.15) | segments 2..K | 299 / 1426 | 39.7 | RAW-15k -0.2930 [-0.4078, -0.2007]; refcv7-50400 -0.1790 [-0.2579, -0.1112]; RAW-10.5k -0.2431 [-0.3775, -0.1367]; RAW-5k -0.2563 [-0.3841, -0.1561] | RAW-15k -0.1139 [-0.1966, -0.0477] SEP; RAW-10.5k -0.0640 [-0.1562, +0.0100]; RAW-5k -0.0773 [-0.1626, -0.0101] SEP |
| curved & v0 5-10 | last segment | 429 / 429 | 29.8 | RAW-15k -0.3274 [-0.4468, -0.2168]; refcv7-50400 -0.2490 [-0.3710, -0.1538]; RAW-10.5k -0.2392 [-0.3838, -0.1158]; RAW-5k -0.2174 [-0.3420, -0.1122] | RAW-15k -0.0784 [-0.1965, +0.0460]; RAW-10.5k +0.0098 [-0.1133, +0.1226]; RAW-5k +0.0316 [-0.0699, +0.1251] |
| curved & v0 5-10 | segments 2..K | 547 / 2460 | 18.2 | RAW-15k -0.2950 [-0.4126, -0.1977]; refcv7-50400 -0.1936 [-0.3072, -0.1114]; RAW-10.5k -0.2295 [-0.3718, -0.1179]; RAW-5k -0.2203 [-0.3370, -0.1243] | RAW-15k -0.1014 [-0.2110, +0.0066]; RAW-10.5k -0.0359 [-0.1593, +0.0676]; RAW-5k -0.0268 [-0.1243, +0.0572] |
| sharp & v0 5-10 | last segment | 125 / 125 | 67.6 | RAW-15k -0.3495 [-0.4820, -0.2259]; refcv7-50400 -0.2497 [-0.3826, -0.1437]; RAW-10.5k -0.2629 [-0.4323, -0.1270]; RAW-5k -0.2506 [-0.3938, -0.1332] | RAW-15k -0.0997 [-0.2370, +0.0417]; RAW-10.5k -0.0131 [-0.1582, +0.1194]; RAW-5k -0.0008 [-0.1275, +0.1076] |
| sharp & v0 5-10 | segments 2..K | 139 / 666 | 43.8 | RAW-15k -0.3247 [-0.4666, -0.2129]; refcv7-50400 -0.1991 [-0.3251, -0.1059]; RAW-10.5k -0.2635 [-0.4326, -0.1354]; RAW-5k -0.2581 [-0.4008, -0.1457] | RAW-15k -0.1256 [-0.2602, +0.0019]; RAW-10.5k -0.0644 [-0.2148, +0.0602]; RAW-5k -0.0590 [-0.1821, +0.0398] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**RAW-15k - refcv7-50400**

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
**RAW-15k - RAW-10.5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3491 | -0.0526 | 0.07453 | 0.07221 | 0.537 / 0.463 / 0.000 | 720.7 / -183.6 |
| turn|straight <0.02 | 2082 | 0.1292 | 0.09090 | 0.10537 | 0.556 / 0.444 / 0.000 | 163.2 / 268.9 |
| turn|gentle 0.02-0.06 | 673 | 0.2738 | 0.05058 | 0.03280 | 0.527 / 0.473 / 0.000 | 222.1 / 184.3 |
| turn|turn 0.06-0.15 | 426 | -0.5393 | -0.04477 | -0.19669 | 0.484 / 0.516 / 0.000 | 45.9 / -229.7 |
| turn|sharp >=0.15 | 301 | -1.3545 | 0.00646 | -0.38984 | 0.502 / 0.498 / 0.000 | 144.6 / -407.7 |
| turn|undef (GT crawling) | 9 | 0.0704 | 0.21417 | 0.07297 | 0.556 / 0.444 / 0.000 | 1.1 / 0.6 |
| speed|v0 <2 | 154 | -0.2613 | -0.09435 | -0.17199 | 0.468 / 0.532 / 0.000 | 41.5 / -40.2 |
| speed|v0 2-5 | 421 | -0.4588 | 0.00189 | -0.07007 | 0.501 / 0.499 / 0.000 | 173.0 / -193.2 |
| speed|v0 5-10 | 1040 | -0.0395 | 0.09846 | 0.11486 | 0.529 / 0.471 / 0.000 | 287.6 / -41.0 |
| speed|v0 10-15 | 952 | 0.0888 | 0.15691 | 0.14130 | 0.607 / 0.393 / 0.000 | 71.1 / 84.5 |
| speed|v0 15-25 | 679 | 0.0361 | 0.02807 | 0.06277 | 0.520 / 0.480 / 0.000 | 32.2 / 24.5 |
| speed|v0 >=25 | 245 | -0.0744 | -0.05279 | -0.07272 | 0.453 / 0.547 / 0.000 | 3.4 / -18.2 |
| nav|nav=follow | 2237 | 0.0353 | 0.10830 | 0.12249 | 0.566 / 0.434 / 0.000 | 339.2 / 78.9 |
| nav|nav=left | 311 | 0.0860 | 0.20718 | 0.28954 | 0.585 / 0.415 / 0.000 | 94.7 / 26.8 |
| nav|nav=right | 943 | -0.3067 | -0.10040 | -0.14127 | 0.453 / 0.547 / 0.000 | 260.8 / -289.3 |
**RAW-10.5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3511 | 0.5631 | 0.16273 | 0.26095 | 0.652 / 0.348 / 0.000 | 997.2 / 1977.0 |
| turn|straight <0.02 | 2086 | 0.3975 | 0.16996 | 0.23109 | 0.706 / 0.294 / 0.000 | 199.7 / 829.1 |
| turn|gentle 0.02-0.06 | 674 | 0.1054 | 0.10113 | 0.15550 | 0.568 / 0.432 / 0.000 | 64.3 / 71.0 |
| turn|turn 0.06-0.15 | 434 | 0.4639 | 0.16336 | 0.33781 | 0.567 / 0.433 / 0.000 | 64.5 / 201.4 |
| turn|sharp >=0.15 | 304 | 2.8551 | 0.50372 | 1.01294 | 0.589 / 0.411 / 0.000 | 173.4 / 868.0 |
| turn|undef (GT crawling) | 13 | 0.5804 | 0.27198 | 0.25796 | 0.615 / 0.385 / 0.000 | 5.0 / 7.5 |
| speed|v0 <2 | 173 | 0.8040 | 0.52537 | 0.82405 | 0.723 / 0.277 / 0.000 | 16.8 / 139.1 |
| speed|v0 2-5 | 422 | 1.5140 | 0.70121 | 1.05518 | 0.770 / 0.230 / 0.000 | 198.2 / 638.9 |
| speed|v0 5-10 | 1040 | 0.9106 | 0.22615 | 0.33505 | 0.639 / 0.361 / 0.000 | 442.8 / 947.0 |
| speed|v0 10-15 | 952 | 0.1707 | 0.14003 | 0.14974 | 0.650 / 0.350 / 0.000 | 92.9 / 162.5 |
| speed|v0 15-25 | 679 | 0.1093 | 0.08193 | 0.07871 | 0.596 / 0.404 / 0.000 | 41.9 / 74.2 |
| speed|v0 >=25 | 245 | 0.0624 | 0.04696 | 0.06760 | 0.612 / 0.388 / 0.000 | 3.5 / 15.3 |
| nav|nav=follow | 2241 | 0.5171 | 0.12449 | 0.19300 | 0.640 / 0.360 / 0.000 | 513.8 / 1158.7 |
| nav|nav=left | 314 | 0.3850 | 0.26403 | 0.28883 | 0.729 / 0.271 / 0.000 | 96.9 / 120.9 |
| nav|nav=right | 956 | 0.7295 | 0.29734 | 0.43004 | 0.655 / 0.345 / 0.000 | 335.1 / 697.4 |
**RAW-15k - RAW-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3468 | -0.2369 | 0.01277 | -0.10025 | 0.506 / 0.494 / 0.000 | 715.0 / -821.5 |
| turn|straight <0.02 | 2064 | 0.0323 | 0.06641 | 0.05405 | 0.546 / 0.454 / 0.000 | 197.2 / 66.6 |
| turn|gentle 0.02-0.06 | 672 | -0.0857 | -0.08476 | -0.24444 | 0.461 / 0.539 / 0.000 | 219.1 / -57.6 |
| turn|turn 0.06-0.15 | 424 | -1.2617 | -0.43247 | -0.87514 | 0.413 / 0.587 / 0.000 | 47.1 / -535.0 |
| turn|sharp >=0.15 | 300 | -0.9951 | -0.30834 | -0.82996 | 0.463 / 0.537 / 0.000 | 142.9 / -298.5 |
| turn|undef (GT crawling) | 8 | 0.3720 | 0.11183 | 0.33611 | 0.625 / 0.375 / 0.000 | 1.6 / 3.0 |
| speed|v0 <2 | 131 | 0.4144 | 0.24413 | 0.37857 | 0.626 / 0.374 / 0.000 | 12.1 / 54.3 |
| speed|v0 2-5 | 421 | 0.1739 | 0.13677 | 0.18080 | 0.530 / 0.470 / 0.000 | 185.9 / 73.2 |
| speed|v0 5-10 | 1040 | -0.5122 | -0.14101 | -0.36274 | 0.457 / 0.543 / 0.000 | 275.4 / -532.6 |
| speed|v0 10-15 | 952 | -0.3253 | 0.06451 | -0.09429 | 0.527 / 0.473 / 0.000 | 58.3 / -309.7 |
| speed|v0 15-25 | 679 | -0.1264 | 0.04359 | 0.00893 | 0.538 / 0.462 / 0.000 | 34.3 / -85.8 |
| speed|v0 >=25 | 245 | -0.0850 | -0.03033 | -0.05171 | 0.445 / 0.555 / 0.000 | 4.3 / -20.8 |
| nav|nav=follow | 2224 | -0.1008 | 0.06747 | 0.03200 | 0.545 / 0.455 / 0.000 | 288.3 / -224.3 |
| nav|nav=left | 307 | 0.1098 | 0.27705 | 0.14338 | 0.593 / 0.407 / 0.000 | 73.0 / 33.7 |
| nav|nav=right | 937 | -0.6734 | -0.32886 | -0.57861 | 0.385 / 0.615 / 0.000 | 306.2 / -630.9 |
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

### S6 -- the episodes carrying the summed delta (panel file index only)

**RAW-15k - refcv7-50400** (sum 1778.4 over 139 episodes)

* ep111: n 26, share 0.130, mean delta 8.9116, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share 0.070, mean delta 4.6297, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.062, mean delta 4.2379, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.054, mean delta 3.5712, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep048: n 27, share 0.048, mean delta 3.1582, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep136: n 18, share 0.045, mean delta 4.4846, sharp-window frac 0.33, curved frac 0.56, mean v0 1.4, nav {'nav=follow': 0, 'nav=left': 18, 'nav=right': 0}
**RAW-15k - RAW-10.5k** (sum -183.6 over 139 episodes)

* ep078: n 26, share -0.892, mean delta 6.2995, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep048: n 27, share -0.440, mean delta 2.9953, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep136: n 16, share -0.424, mean delta 4.8651, sharp-window frac 0.38, curved frac 0.62, mean v0 1.4, nav {'nav=follow': 0, 'nav=left': 16, 'nav=right': 0}
* ep111: n 26, share -0.414, mean delta 2.9268, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep061: n 26, share -0.410, mean delta 2.8945, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share -0.308, mean delta 2.0942, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**RAW-10.5k - refcv7-50400** (sum 1977.0 over 139 episodes)

* ep129: n 25, share 0.106, mean delta 8.4095, sharp-window frac 0.52, curved frac 0.60, mean v0 3.6, nav {'nav=follow': 25, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.079, mean delta 5.9847, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.077, mean delta 5.8760, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.057, mean delta 4.1863, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep013: n 27, share 0.050, mean delta 3.6476, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep024: n 27, share 0.047, mean delta 3.4598, sharp-window frac 0.00, curved frac 0.00, mean v0 5.1, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
**RAW-15k - RAW-5k** (sum -821.5 over 139 episodes)

* ep078: n 26, share -0.218, mean delta 6.8746, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep059: n 27, share -0.137, mean delta 4.1631, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep048: n 27, share -0.063, mean delta 1.9248, sharp-window frac 0.00, curved frac 0.67, mean v0 6.9, nav {'nav=follow': 0, 'nav=left': 27, 'nav=right': 0}
* ep019: n 28, share -0.054, mean delta 1.5935, sharp-window frac 0.00, curved frac 0.43, mean v0 4.5, nav {'nav=follow': 28, 'nav=left': 0, 'nav=right': 0}
* ep001: n 26, share -0.052, mean delta 1.6343, sharp-window frac 0.00, curved frac 0.00, mean v0 3.2, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep010: n 27, share -0.051, mean delta 1.5379, sharp-window frac 0.00, curved frac 0.00, mean v0 3.5, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
**RAW-5k - refcv7-50400** (sum 2545.5 over 139 episodes)

* ep111: n 26, share 0.089, mean delta 8.7416, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep085: n 26, share 0.063, mean delta 6.1198, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.058, mean delta 5.6953, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep028: n 27, share 0.042, mean delta 3.9683, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep041: n 26, share 0.041, mean delta 3.9814, sharp-window frac 0.12, curved frac 0.38, mean v0 8.6, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.037, mean delta 3.4972, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**RAW-15k vs RAW-10.5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 972 / 0.1187 [0.0134, 0.2180] | 35 / 0.0129 [-0.4934, 0.5201] |
| only new agrees | 354 / -0.7211 [-1.2103, -0.3200] | 27 / -6.5882 [-11.1935, -3.0269] |
| only earlier agrees | 425 / 0.8833 [0.5034, 1.3486] | 44 / 2.8357 [0.5730, 5.5821] |
| neither agrees | 1742 / -0.2413 [-0.6404, 0.1536] | 195 / -1.2991 [-3.9081, 1.1922] |
**RAW-15k vs RAW-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 742 / -0.0051 [-0.1437, 0.1282] | 21 / -0.9590 [-2.7147, 0.3777] |
| only new agrees | 582 / -0.7560 [-1.1227, -0.4362] | 40 / -4.9931 [-9.5175, -2.0424] |
| only earlier agrees | 491 / 0.7109 [0.3418, 1.1499] | 28 / 3.0899 [-0.5146, 11.4678] |
| neither agrees | 1654 / -0.4303 [-0.7712, -0.0645] | 211 / -0.7254 [-2.0476, 0.6348] |
**RAW-15k vs RAW-10.5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 983 / 0.1263 [0.0161, 0.2331] | 41 / -0.3790 [-1.4330, 0.2797] |
| only new agrees | 349 / -1.3530 [-2.1189, -0.7089] | 36 / -10.1228 [-15.3067, -5.4963] |
| only earlier agrees | 416 / 0.9070 [0.4517, 1.5019] | 33 / 3.5773 [0.2496, 7.0621] |
| neither agrees | 1745 / -0.1428 [-0.5989, 0.2973] | 191 / -1.2959 [-4.4915, 1.5710] |
**RAW-15k vs RAW-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 760 / 0.0153 [-0.1471, 0.1753] | 26 / -1.3033 [-3.1292, 0.0637] |
| only new agrees | 570 / -1.1199 [-1.6991, -0.6555] | 50 / -6.5286 [-11.6724, -2.9427] |
| only earlier agrees | 492 / 0.6341 [0.3485, 1.0264] | 25 / 4.3322 [0.4822, 12.6912] |
| neither agrees | 1648 / -0.3136 [-0.6510, 0.0742] | 199 / -0.2946 [-1.8169, 1.1077] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**RAW-15k vs RAW-10.5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3446 | 0.3254 [0.0782, 0.5850] | 0.2217 [-0.0690, 0.5801] | 0.0960 [-0.3114, 0.4331] |
| turn|straight <0.02 | 2049 | 0.0900 [-0.0492, 0.2208] | -0.0696 [-0.1961, 0.0287] | 0.1633 [0.0875, 0.2471] SEP |
| turn|gentle 0.02-0.06 | 672 | -0.0474 [-0.6952, 0.4447] | -0.1268 [-0.3501, 0.0903] | 0.0751 [-0.4983, 0.5859] |
| turn|turn 0.06-0.15 | 422 | 0.3576 [-0.0830, 0.8468] | 0.0968 [-0.3638, 0.5633] | 0.2471 [-0.4421, 0.9992] |
| turn|sharp >=0.15 | 296 | 2.7777 [0.8821, 5.2341] | 3.2164 [0.4508, 7.0176] | -0.5363 [-4.1847, 2.4827] |
| turn|undef (GT crawling) | 7 | -0.1820 [-0.7158, 0.1596] | -0.2310 [-0.5225, 0.1564] | 0.0490 [-0.4686, 0.4486] |
| speed|v0 <2 | 109 | 0.8918 [-0.0136, 2.3283] | -0.0438 [-1.5053, 1.8061] | 0.7861 [-0.2547, 1.9128] |
| speed|v0 2-5 | 421 | 0.9122 [-0.1437, 2.0808] | 0.6912 [-0.6315, 2.8741] | 0.2196 [-2.0135, 1.9854] |
| speed|v0 5-10 | 1040 | 0.5297 [0.0498, 1.0702] | 0.4523 [-0.0929, 1.0271] | 0.0774 [-0.4780, 0.7007] |
| speed|v0 10-15 | 952 | 0.0468 [-0.0493, 0.1359] | 0.0472 [-0.0704, 0.1785] | -0.0004 [-0.1610, 0.1513] |
| speed|v0 15-25 | 679 | 0.0536 [-0.0433, 0.1543] | -0.0490 [-0.0951, -0.0072] | 0.1026 [0.0059, 0.2017] SEP |
| speed|v0 >=25 | 245 | 0.0037 [-0.0093, 0.0211] | -0.0080 [-0.0342, 0.0188] | 0.0118 [-0.0217, 0.0502] |
**RAW-15k vs RAW-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3435 | 0.3254 [0.0782, 0.5850] | 0.5780 [0.3356, 0.8458] | -0.2475 [-0.5109, 0.0045] |
| turn|straight <0.02 | 2039 | 0.0900 [-0.0492, 0.2208] | 0.4004 [0.1539, 0.6899] | -0.3116 [-0.5829, -0.0725] SEP |
| turn|gentle 0.02-0.06 | 672 | -0.0474 [-0.6952, 0.4447] | 0.5311 [0.1944, 0.8649] | -0.5708 [-1.1815, -0.0536] SEP |
| turn|turn 0.06-0.15 | 422 | 0.3576 [-0.0830, 0.8468] | 0.3326 [-0.2227, 0.9146] | 0.0454 [-0.5059, 0.5031] |
| turn|sharp >=0.15 | 296 | 2.7777 [0.8821, 5.2341] | 2.2395 [0.6061, 4.2331] | 0.5227 [-1.3751, 2.5754] |
| turn|undef (GT crawling) | 6 | -0.1820 [-0.7158, 0.1596] | 0.9170 [0.3116, 1.4676] | -0.8659 [-1.3653, -0.2272] SEP |
| speed|v0 <2 | 98 | 0.8918 [-0.0136, 2.3283] | 0.9165 [0.0601, 2.4457] | 0.2603 [-0.4982, 1.0755] |
| speed|v0 2-5 | 421 | 0.9122 [-0.1437, 2.0808] | 1.8982 [0.4836, 3.4811] | -0.9903 [-2.3842, 0.3722] |
| speed|v0 5-10 | 1040 | 0.5297 [0.0498, 1.0702] | 1.1156 [0.6851, 1.5621] | -0.5859 [-1.0781, -0.0866] SEP |
| speed|v0 10-15 | 952 | 0.0468 [-0.0493, 0.1359] | -0.1191 [-0.2445, 0.0019] | 0.1659 [0.0150, 0.3201] SEP |
| speed|v0 15-25 | 679 | 0.0536 [-0.0433, 0.1543] | 0.0393 [-0.0394, 0.1095] | 0.0143 [-0.0922, 0.1278] |
| speed|v0 >=25 | 245 | 0.0037 [-0.0093, 0.0211] | 0.0733 [-0.0112, 0.1503] | -0.0696 [-0.1558, 0.0253] |
**RAW-15k vs RAW-10.5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
**RAW-15k vs RAW-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* RAW-15k vs RAW-10.5k|seed0: anchor flip rate 0.437; heading delta on flipped windows -0.2155 [-0.6701, 0.2400] (n 1556) vs unflipped 0.0777 [-0.0197, 0.1689]; goal_dist_sel proxy new-earlier -0.1547 [-0.5328, 0.2215]; corr(window dHeading, window dGoalDist) -0.003
* RAW-15k vs RAW-5k|seed0: anchor flip rate 0.588; heading delta on flipped windows -0.3111 [-0.6247, 0.0160] (n 2091) vs unflipped -0.1131 [-0.2638, 0.0306]; goal_dist_sel proxy new-earlier -0.1215 [-0.5346, 0.3093]; corr(window dHeading, window dGoalDist) 0.020
* RAW-15k vs RAW-10.5k|seed1: anchor flip rate 0.441; heading delta on flipped windows -0.2423 [-0.7873, 0.2899] (n 1568) vs unflipped 0.0830 [-0.0165, 0.1772]; goal_dist_sel proxy new-earlier -0.1676 [-0.5488, 0.2046]; corr(window dHeading, window dGoalDist) -0.004
* RAW-15k vs RAW-5k|seed1: anchor flip rate 0.589; heading delta on flipped windows -0.3442 [-0.6797, -0.0004] (n 2098) vs unflipped -0.0798 [-0.2253, 0.0578]; goal_dist_sel proxy new-earlier -0.1294 [-0.5414, 0.3049]; corr(window dHeading, window dGoalDist) 0.017
