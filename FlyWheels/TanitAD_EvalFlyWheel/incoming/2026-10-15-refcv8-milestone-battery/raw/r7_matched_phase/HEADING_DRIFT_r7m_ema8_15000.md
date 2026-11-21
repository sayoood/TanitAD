# refcv8 heading-drift localisation: EMA8-15k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-19T18:18:15; trees: `EMA8-15k` = C:/Users/Admin/refcv8_eval_archive/battery/step15000ema_unc; `R7-15k` = C:/Users/Admin/refcv8_eval_archive/battery/r7step15000_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- EMA8-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4510 | 139 | 1.4845 | 1.4076 | 0.0770 [0.0341, 0.1222] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.2129 | 1.1597 | 0.0531 [0.0095, 0.1049] | **SEP** | 0.643 / 0.444 / -0.0667 [-0.2334, 0.0694] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0815 | 1.0094 | 0.0720 [-0.0027, 0.1609] |  | 0.160 / 0.149 / -0.0059 [-0.1437, 0.1462] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9488 | 1.9207 | 0.0281 [-0.0768, 0.1379] |  | 0.113 / 0.041 / -0.0551 [-0.2339, 0.1162] |
| turn: sharp >=0.15 | 366 | 42 | 3.3408 | 2.9907 | 0.3501 [0.0352, 0.7223] | **SEP** | 0.081 / 0.369 / 0.2973 [-0.1280, 0.9140] |
| turn: undef (GT crawling) (low n) | 18 | 14 | 10.5016 | 10.5645 | -0.0629 [-0.1689, 0.0408] |  | 0.004 / -0.003 / -0.1404 [-0.3296, 0.0448] |
| speed: v0 <2 | 221 | 37 | 13.3705 | 13.2289 | 0.1416 [-0.0417, 0.3721] |  | 0.049 / 0.090 / 0.0680 [-0.2232, 0.4689] |
| speed: v0 2-5 | 516 | 58 | 1.9464 | 1.8694 | 0.0770 [-0.1485, 0.3352] |  | 0.114 / 0.114 / 0.0000 [-0.2988, 0.4261] |
| speed: v0 5-10 | 1390 | 84 | 1.2090 | 1.0092 | 0.1998 [0.1031, 0.3013] | **SEP** | 0.308 / 0.800 / 0.1775 [0.0105, 0.3585] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.6019 | 0.5744 | 0.0275 [-0.0146, 0.0698] |  | 0.264 / 0.094 / -0.0673 [-0.1689, 0.0342] |
| speed: v0 15-25 | 856 | 35 | 0.3348 | 0.3734 | -0.0386 [-0.0727, -0.0078] | **SEP** | 0.190 / -0.095 / -0.1427 [-0.2354, -0.0574] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1538 | 0.1576 | -0.0038 [-0.0339, 0.0188] |  | 0.074 / -0.004 / -0.0873 [-0.1723, -0.0133] BONF-SEP |
| nav: nav=follow | 2879 | 88 | 0.6814 | 0.5982 | 0.0832 [0.0348, 0.1408] | **SEP** | 0.638 / 0.690 / 0.0171 [-0.1370, 0.1586] |
| nav: nav=left (low n) | 402 | 13 | 6.9194 | 6.9250 | -0.0056 [-0.1187, 0.0741] |  | 0.089 / -0.006 / -0.0906 [-0.3038, 0.0354] |
| nav: nav=right | 1229 | 38 | 1.5882 | 1.4987 | 0.0895 [-0.0077, 0.1920] |  | 0.273 / 0.317 / 0.0172 [-0.1488, 0.2111] |

episode concentration: 0.647 of episodes have mean delta > 0; top-5 episodes carry 0.498 of the summed delta; delta excluding the top 5 = 0.0401 (overall 0.0770).

### S2 -- EMA8-15k - R7-15k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4506 | 139 | 1.4855 | 1.5691 | -0.0836 [-0.1507, -0.0190] | **SEP** |  |
| turn: straight <0.02 | 2895 | 132 | 1.2139 | 1.2019 | 0.0120 [-0.0403, 0.0687] |  | 0.642 / -0.092 / 0.2674 [0.0814, 0.4705] BONF-SEP |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0815 | 1.0440 | 0.0375 [-0.0589, 0.1586] |  | 0.160 / -0.072 / 0.1441 [-0.0144, 0.3313] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9488 | 2.2533 | -0.3045 [-0.4282, -0.1831] | **SEP** | 0.113 / 0.411 / -0.2490 [-0.4310, -0.0613] BONF-SEP |
| turn: sharp >=0.15 | 366 | 42 | 3.3408 | 4.1216 | -0.7808 [-1.2361, -0.3714] | **SEP** | 0.081 / 0.759 / -0.7588 [-1.4605, -0.1570] BONF-SEP |
| turn: undef (GT crawling) (low n) | 17 | 13 | 11.0809 | 10.9581 | 0.1228 [-0.1444, 0.5833] |  | 0.004 / -0.006 / 0.2072 [-0.1648, 1.1366] |
| speed: v0 <2 | 217 | 37 | 13.6106 | 13.7577 | -0.1471 [-0.3728, 0.1012] |  | 0.048 / 0.085 / -0.0667 [-0.4263, 0.4110] |
| speed: v0 2-5 | 516 | 58 | 1.9464 | 2.1713 | -0.2249 [-0.5822, 0.1386] |  | 0.115 / 0.308 / -0.1595 [-0.7317, 0.4839] |
| speed: v0 5-10 | 1390 | 84 | 1.2090 | 1.2623 | -0.0533 [-0.1721, 0.0571] |  | 0.308 / 0.197 / 0.0438 [-0.1298, 0.2258] |
| speed: v0 10-15 | 1192 | 64 | 0.6019 | 0.6836 | -0.0817 [-0.1460, -0.0157] | **SEP** | 0.265 / 0.259 / 0.0026 [-0.1277, 0.1272] |
| speed: v0 15-25 | 856 | 35 | 0.3348 | 0.3947 | -0.0599 [-0.0955, -0.0293] | **SEP** | 0.190 / 0.136 / 0.0292 [-0.0879, 0.1673] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1538 | 0.1716 | -0.0178 [-0.0577, 0.0119] |  | 0.074 / 0.016 / 0.0711 [-0.0418, 0.1853] |
| nav: nav=follow | 2875 | 88 | 0.6819 | 0.7020 | -0.0202 [-0.0772, 0.0428] |  | 0.638 / 0.154 / 0.1752 [-0.0252, 0.3935] |
| nav: nav=left (low n) | 402 | 13 | 6.9194 | 7.1815 | -0.2621 [-0.7144, 0.0481] |  | 0.089 / 0.280 / -0.1959 [-1.0240, 0.2211] |
| nav: nav=right | 1229 | 38 | 1.5882 | 1.7618 | -0.1736 [-0.3041, -0.0372] | **SEP** | 0.273 / 0.566 / -0.1238 [-0.3495, 0.0984] |

episode concentration: 0.338 of episodes have mean delta > 0; top-5 episodes carry -0.420 of the summed delta; delta excluding the top 5 = -0.1231 (overall -0.0836).

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

### S2 -- EMA8-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4474 | 139 | 0.7265 | 0.6284 | 0.0981 [0.0070, 0.2664] | **SEP** | 0.0608 [-0.0336, 0.2323] |
| seg1 | straight | 2878 | 132 | 0.7077 | 0.5699 | 0.1379 [0.0046, 0.4155] | **SEP** | 0.1074 [-0.0303, 0.3930] |
| seg1 | curved | 1585 | 101 | 0.6497 | 0.6231 | 0.0266 [0.0005, 0.0537] | **SEP** | -0.0236 [-0.0715, 0.0202] |
| seg2 | all | 4470 | 139 | 1.1181 | 0.9389 | 0.1793 [0.0285, 0.4378] | **SEP** | -0.1005 [-0.2009, -0.0156] SEP |
| seg2 | straight | 2880 | 132 | 0.8903 | 0.6574 | 0.2329 [0.0170, 0.6677] | **SEP** | -0.1236 [-0.2871, -0.0313] SEP |
| seg2 | curved | 1590 | 101 | 1.5309 | 1.4487 | 0.0822 [0.0093, 0.1561] | **SEP** | -0.0587 [-0.1873, 0.0591] |
| seg3 | all | 4460 | 139 | 1.2194 | 1.2044 | 0.0150 [-0.1685, 0.1390] |  | -0.1613 [-0.3527, -0.0149] SEP |
| seg3 | straight | 2874 | 132 | 0.7199 | 0.7743 | -0.0544 [-0.3695, 0.1242] |  | -0.2179 [-0.5463, -0.0458] SEP |
| seg3 | curved | 1586 | 101 | 2.1246 | 1.9840 | 0.1406 [0.0280, 0.2669] | **SEP** | -0.0587 [-0.2398, 0.1110] |
| seg4 | all | 4440 | 139 | 1.6770 | 1.4933 | 0.1837 [0.0457, 0.3796] | **SEP** | 0.1194 [-0.1236, 0.4856] |
| seg4 | straight | 2857 | 131 | 1.0121 | 0.8401 | 0.1720 [0.0038, 0.4735] | **SEP** | 0.2009 [-0.1061, 0.7637] |
| seg4 | curved | 1576 | 101 | 2.8866 | 2.6802 | 0.2063 [0.0288, 0.4001] | **SEP** | -0.0273 [-0.2970, 0.2217] |

### S2 -- EMA8-15k - R7-15k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4473 | 139 | 0.7265 | 0.6593 | 0.0672 [-0.0660, 0.3172] |  | -0.2104 [-0.4618, -0.0717] SEP |
| seg1 | straight | 2878 | 132 | 0.7077 | 0.5440 | 0.1638 [-0.0358, 0.5808] |  | -0.2804 [-0.7006, -0.0768] SEP |
| seg1 | curved | 1584 | 101 | 0.6498 | 0.7572 | -0.1074 [-0.1423, -0.0711] | **SEP** | -0.0848 [-0.1425, -0.0270] SEP |
| seg2 | all | 4460 | 139 | 1.1192 | 1.0906 | 0.0286 [-0.1304, 0.2874] |  | -0.3072 [-0.5661, -0.1403] SEP |
| seg2 | straight | 2873 | 131 | 0.8914 | 0.7139 | 0.1775 [-0.0448, 0.6020] |  | -0.3894 [-0.8266, -0.1616] SEP |
| seg2 | curved | 1587 | 101 | 1.5315 | 1.7724 | -0.2410 [-0.3509, -0.1291] | **SEP** | -0.1583 [-0.3042, -0.0196] SEP |
| seg3 | all | 4448 | 139 | 1.2195 | 1.4369 | -0.2174 [-0.5146, -0.0230] | **SEP** | -0.3759 [-0.8131, -0.1035] SEP |
| seg3 | straight | 2867 | 131 | 0.7198 | 0.8582 | -0.1384 [-0.5807, 0.1096] |  | -0.4939 [-1.2093, -0.1276] SEP |
| seg3 | curved | 1581 | 101 | 2.1257 | 2.4864 | -0.3607 [-0.5485, -0.1818] | **SEP** | -0.1619 [-0.3891, 0.0699] |
| seg4 | all | 4437 | 139 | 1.6775 | 1.8327 | -0.1552 [-0.3057, -0.0270] | **SEP** | -0.3217 [-0.4779, -0.1710] SEP |
| seg4 | straight | 2856 | 131 | 1.0128 | 1.0790 | -0.0661 [-0.2322, 0.0462] |  | -0.2902 [-0.4608, -0.1738] SEP |
| seg4 | curved | 1575 | 101 | 2.8863 | 3.2053 | -0.3190 [-0.5691, -0.0864] | **SEP** | -0.3778 [-0.7229, -0.0262] SEP |

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

### S2 -- EMA8-15k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 439 | 47 | 6.0086 | 5.9675 | 0.0411 [-0.1133, 0.2670] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.4606 | 0.3674 | 0.0931 [0.0331, 0.1666] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.1888 | 0.1947 | -0.0058 [-0.0189, 0.0060] |  |  |
| gentle 0.02-0.06 | v0 <5 | 111 | 29 | 1.8547 | 1.7693 | 0.0855 [-0.1773, 0.4694] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.0873 | 0.9711 | 0.1162 [0.0281, 0.2148] | **SEP** |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.4832 | 0.5590 | -0.0758 [-0.1553, -0.0046] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 78 | 25 | 5.1591 | 5.1476 | 0.0115 [-0.4061, 0.3572] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.5008 | 1.4275 | 0.0732 [-0.0438, 0.2093] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 0.8369 | 0.9719 | -0.1350 [-0.2527, -0.0036] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 91 | 16 | 5.7599 | 5.2793 | 0.4806 [-0.2278, 1.5448] |  |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.6344 | 2.2857 | 0.3487 [0.0539, 0.7338] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.0163 | 1.3866 | -0.3703 [-0.6006, 0.0656] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 18 | 14 | 10.5016 | 10.5645 | -0.0629 [-0.1689, 0.0408] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA8-15k s0-s1 [CI] | R7-15k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0005 [-0.0127, 0.0123] | 0.0102 [-0.0033, 0.0246] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0018 [-0.0136, 0.0091] | 0.0073 [0.0005, 0.0153] SEP | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0037 [-0.0392, 0.0479] | 0.0177 [-0.0188, 0.0628] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | 0.0098 [-0.0370, 0.0556] | 0.0313 [-0.0224, 0.0874] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0106 [-0.0602, 0.0395] | -0.0152 [-0.1389, 0.1082] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.0306 [-0.1758, 0.0775] | 0.0587 [-0.0663, 0.2319] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | -0.0244 [-0.0824, 0.0341] | 0.0943 [0.0353, 0.1870] SEP | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0220 [-0.0847, 0.0342] | 0.0373 [-0.0539, 0.1359] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | 0.0054 [-0.0246, 0.0319] | -0.0004 [-0.0255, 0.0239] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0088 [-0.0029, 0.0208] | 0.0077 [-0.0040, 0.0196] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | -0.0042 [-0.0124, 0.0037] | -0.0047 [-0.0124, 0.0024] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | 0.0008 [-0.0049, 0.0064] | -0.0024 [-0.0100, 0.0042] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | 0.0022 [-0.0101, 0.0155] | 0.0077 [-0.0004, 0.0163] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0120 [-0.0189, 0.0388] | 0.0155 [-0.0735, 0.1327] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0109 [-0.0458, 0.0198] | 0.0142 [-0.0181, 0.0514] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- EMA8-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0770 | 0.0713 / 0.0826 | True / True / True | 0.0005 / 0.0110 | 7.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0531 | 0.0513 / 0.0550 | True / True / True | 0.0018 / 0.0019 | 27.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0720 | 0.0817 / 0.0623 | True / False / False | 0.0037 / 0.0161 | 4.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0281 | 0.0184 / 0.0378 | False / False / False | 0.0098 / 0.0331 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.3501 | 0.2882 / 0.4120 | False / True / True | 0.0106 / 0.1080 | 3.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | -0.0629 | -0.0478 / -0.0780 | False / True / False | 0.0306 / 0.0237 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1416 | 0.0988 / 0.1837 | False / False / False | 0.0244 / 0.0573 | 2.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0770 | 0.0300 / 0.1239 | False / False / False | 0.0220 / 0.0719 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1998 | 0.2021 / 0.1974 | True / True / True | 0.0054 / 0.0006 | 37.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0275 | 0.0319 / 0.0230 | False / False / False | 0.0088 / 0.0001 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0386 | -0.0396 / -0.0377 | True / True / True | 0.0042 / 0.0023 | 9.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0038 | -0.0031 / -0.0045 | False / False / False | 0.0008 / 0.0005 | 4.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0832 | 0.0861 / 0.0802 | True / True / True | 0.0022 / 0.0036 | 23.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.0056 | -0.0302 / 0.0191 | False / False / False | 0.0120 / 0.0608 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0895 | 0.0698 / 0.1091 | False / True / False | 0.0109 / 0.0283 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- EMA8-15k - R7-15k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.0836 | -0.0883 / -0.0789 | True / True / True | 0.0005 / 0.0102 | 8.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0120 | 0.0075 / 0.0164 | False / False / False | 0.0018 / 0.0073 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0375 | 0.0306 / 0.0443 | False / False / False | 0.0037 / 0.0177 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.3045 | -0.3094 / -0.2996 | True / True / True | 0.0098 / 0.0313 | 9.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|sharp >=0.15 | -0.7808 | -0.7802 / -0.7813 | True / True / True | 0.0106 / 0.0152 | 51.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.1228 | 0.0595 / 0.1861 | False / False / False | 0.0306 / 0.0587 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.1471 | -0.1998 / -0.0941 | False / False / False | 0.0244 / 0.0943 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.2249 | -0.2545 / -0.1952 | False / False / False | 0.0220 / 0.0373 | 6.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.0533 | -0.0504 / -0.0562 | False / False / False | 0.0054 / 0.0004 | 9.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.0817 | -0.0811 / -0.0823 | True / True / True | 0.0088 / 0.0077 | 9.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.0599 | -0.0597 / -0.0602 | True / True / True | 0.0042 / 0.0047 | 12.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0178 | -0.0162 / -0.0194 | False / False / False | 0.0008 / 0.0024 | 7.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.0202 | -0.0229 / -0.0175 | False / False / False | 0.0022 / 0.0077 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.2621 | -0.2575 / -0.2666 | False / False / False | 0.0120 / 0.0155 | 17.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.1736 | -0.1861 / -0.1611 | True / True / True | 0.0109 / 0.0142 | 12.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |

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

* seg1 (n 1584): EMA8-15k bias -0.027, refcv7-50400 bias +0.082, R7-15k bias +0.036; EMA8-15k MAE 0.650, refcv7-50400 MAE 0.623, R7-15k MAE 0.757
* seg2 (n 1587): EMA8-15k bias -0.347, refcv7-50400 bias -0.093, R7-15k bias -0.332; EMA8-15k MAE 1.531, refcv7-50400 MAE 1.449, R7-15k MAE 1.772
* seg3 (n 1581): EMA8-15k bias -0.912, refcv7-50400 bias -0.617, R7-15k bias -1.255; EMA8-15k MAE 2.126, refcv7-50400 MAE 1.986, R7-15k MAE 2.486
* seg4 (n 1575): EMA8-15k bias -1.192, refcv7-50400 bias -0.715, R7-15k bias -1.850; EMA8-15k MAE 2.886, refcv7-50400 MAE 2.681, R7-15k MAE 3.205
**sharp (>= 0.15)**

* seg1 (n 364): EMA8-15k bias -0.298, refcv7-50400 bias +0.010, R7-15k bias -0.125; EMA8-15k MAE 0.933, refcv7-50400 MAE 0.917, R7-15k MAE 1.147
* seg2 (n 366): EMA8-15k bias -1.204, refcv7-50400 bias -0.528, R7-15k bias -1.213; EMA8-15k MAE 2.593, refcv7-50400 MAE 2.413, R7-15k MAE 3.135
* seg3 (n 365): EMA8-15k bias -2.587, refcv7-50400 bias -1.863, R7-15k bias -3.746; EMA8-15k MAE 4.103, refcv7-50400 MAE 3.649, R7-15k MAE 5.217
* seg4 (n 363): EMA8-15k bias -3.040, refcv7-50400 bias -1.837, R7-15k bias -5.252; EMA8-15k MAE 5.637, refcv7-50400 MAE 4.887, R7-15k MAE 6.867
**curved & v0 5-10**

* seg1 (n 627): EMA8-15k bias -0.164, refcv7-50400 bias -0.038, R7-15k bias -0.095; EMA8-15k MAE 0.590, refcv7-50400 MAE 0.535, R7-15k MAE 0.735
* seg2 (n 627): EMA8-15k bias -0.612, refcv7-50400 bias -0.282, R7-15k bias -0.534; EMA8-15k MAE 1.559, refcv7-50400 MAE 1.363, R7-15k MAE 1.843
* seg3 (n 627): EMA8-15k bias -1.301, refcv7-50400 bias -0.871, R7-15k bias -1.477; EMA8-15k MAE 2.334, refcv7-50400 MAE 2.019, R7-15k MAE 2.653
* seg4 (n 627): EMA8-15k bias -1.811, refcv7-50400 bias -1.194, R7-15k bias -2.210; EMA8-15k MAE 3.150, refcv7-50400 MAE 2.730, R7-15k MAE 3.352
**sharp & v0 5-10**

* seg1 (n 164): EMA8-15k bias -0.168, refcv7-50400 bias +0.103, R7-15k bias +0.093; EMA8-15k MAE 0.858, refcv7-50400 MAE 0.857, R7-15k MAE 1.156
* seg2 (n 164): EMA8-15k bias -1.062, refcv7-50400 bias -0.422, R7-15k bias -0.744; EMA8-15k MAE 2.495, refcv7-50400 MAE 2.243, R7-15k MAE 3.001
* seg3 (n 164): EMA8-15k bias -2.805, refcv7-50400 bias -2.032, R7-15k bias -3.460; EMA8-15k MAE 4.150, refcv7-50400 MAE 3.465, R7-15k MAE 4.827
* seg4 (n 164): EMA8-15k bias -3.654, refcv7-50400 bias -2.403, R7-15k bias -5.287; EMA8-15k MAE 5.851, refcv7-50400 MAE 4.635, R7-15k MAE 6.586

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1575 / 1575 | 10.2 | EMA8-15k -0.1327 [-0.2195, -0.0647]; refcv7-50400 -0.0844 [-0.1557, -0.0272]; R7-15k -0.2176 [-0.3010, -0.1494] | EMA8-15k -0.0483 [-0.0800, -0.0225] SEP; R7-15k -0.1332 [-0.1748, -0.0923] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1593 / 4743 | 7.2 | EMA8-15k -0.1920 [-0.3157, -0.0893]; refcv7-50400 -0.1465 [-0.2654, -0.0521]; R7-15k -0.2602 [-0.3752, -0.1625] | EMA8-15k -0.0455 [-0.0737, -0.0221] SEP; R7-15k -0.1137 [-0.1540, -0.0763] SEP |
| sharp (>= 0.15) | last segment | 363 / 363 | 25.4 | EMA8-15k -0.1409 [-0.2445, -0.0616]; refcv7-50400 -0.0921 [-0.1765, -0.0255]; R7-15k -0.2351 [-0.3342, -0.1555] | EMA8-15k -0.0488 [-0.0867, -0.0176] SEP; R7-15k -0.1430 [-0.1924, -0.0983] SEP |
| sharp (>= 0.15) | segments 2..K | 366 / 1094 | 17.9 | EMA8-15k -0.1523 [-0.2562, -0.0713]; refcv7-50400 -0.1041 [-0.1873, -0.0360]; R7-15k -0.2305 [-0.3309, -0.1489] | EMA8-15k -0.0482 [-0.0838, -0.0187] SEP; R7-15k -0.1264 [-0.1732, -0.0841] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | EMA8-15k -0.1397 [-0.2276, -0.0723]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; R7-15k -0.2026 [-0.2773, -0.1468] | EMA8-15k -0.0472 [-0.0991, -0.0027] SEP; R7-15k -0.1100 [-0.1454, -0.0722] SEP |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | EMA8-15k -0.1470 [-0.2318, -0.0818]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; R7-15k -0.2000 [-0.2767, -0.1423] | EMA8-15k -0.0478 [-0.0961, -0.0057] SEP; R7-15k -0.1008 [-0.1343, -0.0660] SEP |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | EMA8-15k -0.1337 [-0.2294, -0.0595]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; R7-15k -0.2004 [-0.2813, -0.1383] | EMA8-15k -0.0424 [-0.1008, +0.0060]; R7-15k -0.1091 [-0.1504, -0.0674] SEP |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | EMA8-15k -0.1369 [-0.2322, -0.0631]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; R7-15k -0.1889 [-0.2718, -0.1251] | EMA8-15k -0.0414 [-0.0964, +0.0052]; R7-15k -0.0934 [-0.1317, -0.0545] SEP |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA8-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4510 | 0.0770 | 0.00393 | 0.00610 | 0.515 / 0.485 / 0.000 | 366.9 / 347.1 |
| turn|straight <0.02 | 2898 | 0.0531 | 0.00465 | 0.00335 | 0.524 / 0.476 / 0.000 | 173.8 / 154.0 |
| turn|gentle 0.02-0.06 | 720 | 0.0720 | 0.00734 | 0.01273 | 0.512 / 0.487 / 0.000 | 52.5 / 51.8 |
| turn|turn 0.06-0.15 | 508 | 0.0281 | -0.02370 | 0.00480 | 0.467 / 0.533 / 0.000 | 31.0 / 14.3 |
| turn|sharp >=0.15 | 366 | 0.3501 | 0.01251 | 0.09777 | 0.516 / 0.484 / 0.000 | 64.6 / 128.1 |
| turn|undef (GT crawling) | 18 | -0.0629 | -0.01327 | -0.03186 | 0.500 / 0.500 / 0.000 | 0.3 / -1.1 |
| speed|v0 <2 | 221 | 0.1416 | -0.01541 | 0.00979 | 0.471 / 0.529 / 0.000 | 25.2 / 31.3 |
| speed|v0 2-5 | 516 | 0.0770 | 0.00623 | 0.01152 | 0.510 / 0.490 / 0.000 | 75.5 / 39.7 |
| speed|v0 5-10 | 1390 | 0.1998 | 0.01470 | 0.03885 | 0.542 / 0.458 / 0.000 | 130.9 / 277.7 |
| speed|v0 10-15 | 1192 | 0.0275 | 0.00382 | 0.00629 | 0.518 / 0.482 / 0.000 | 35.7 / 32.7 |
| speed|v0 15-25 | 856 | -0.0386 | -0.00517 | -0.01635 | 0.471 / 0.529 / 0.000 | 6.5 / -33.1 |
| speed|v0 >=25 | 335 | -0.0038 | 0.00436 | 0.00213 | 0.543 / 0.457 / 0.000 | 0.9 / -1.3 |
| nav|nav=follow | 2879 | 0.0832 | 0.00114 | -0.00101 | 0.505 / 0.495 / 0.000 | 228.6 / 239.4 |
| nav|nav=left | 402 | -0.0056 | 0.00430 | 0.01379 | 0.507 / 0.493 / 0.000 | 16.3 / -2.2 |
| nav|nav=right | 1229 | 0.0895 | 0.01369 | 0.02642 | 0.541 / 0.459 / 0.000 | 112.4 / 109.9 |
**EMA8-15k - R7-15k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4506 | -0.0836 | -0.03322 | -0.07061 | 0.395 / 0.605 / 0.000 | 350.0 / -376.7 |
| turn|straight <0.02 | 2895 | 0.0120 | -0.01936 | -0.03258 | 0.410 / 0.590 / 0.000 | 188.4 / 34.7 |
| turn|gentle 0.02-0.06 | 720 | 0.0375 | -0.04702 | -0.04690 | 0.421 / 0.579 / 0.000 | 65.4 / 27.0 |
| turn|turn 0.06-0.15 | 508 | -0.3045 | -0.18133 | -0.26587 | 0.333 / 0.667 / 0.000 | 24.1 / -154.7 |
| turn|sharp >=0.15 | 366 | -0.7808 | -0.50731 | -0.70634 | 0.309 / 0.691 / 0.000 | 41.6 / -285.8 |
| turn|undef (GT crawling) | 17 | 0.1228 | 0.00121 | -0.01838 | 0.529 / 0.471 / 0.000 | 3.2 / 2.1 |
| speed|v0 <2 | 217 | -0.1471 | -0.03684 | -0.11582 | 0.461 / 0.539 / 0.000 | 18.2 / -31.9 |
| speed|v0 2-5 | 516 | -0.2249 | -0.17167 | -0.20249 | 0.386 / 0.614 / 0.000 | 73.0 / -116.0 |
| speed|v0 5-10 | 1390 | -0.0533 | -0.05598 | -0.07993 | 0.407 / 0.593 / 0.000 | 117.0 / -74.1 |
| speed|v0 10-15 | 1192 | -0.0817 | -0.03805 | -0.06001 | 0.364 / 0.636 / 0.000 | 40.6 / -97.4 |
| speed|v0 15-25 | 856 | -0.0599 | -0.01868 | -0.03283 | 0.397 / 0.603 / 0.000 | 7.6 / -51.3 |
| speed|v0 >=25 | 335 | -0.0178 | -0.00605 | -0.00883 | 0.427 / 0.573 / 0.000 | 1.2 / -6.0 |
| nav|nav=follow | 2875 | -0.0202 | -0.02604 | -0.05078 | 0.395 / 0.605 / 0.000 | 209.3 / -58.0 |
| nav|nav=left | 402 | -0.2621 | -0.00837 | -0.04425 | 0.470 / 0.530 / 0.000 | 27.4 / -105.4 |
| nav|nav=right | 1229 | -0.1736 | -0.07670 | -0.14337 | 0.373 / 0.627 / 0.000 | 104.1 / -213.4 |
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

**EMA8-15k - refcv7-50400** (sum 347.1 over 139 episodes)

* ep024: n 35, share 0.113, mean delta 1.1239, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.102, mean delta 1.0372, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep108: n 24, share 0.100, mean delta 1.4422, sharp-window frac 0.00, curved frac 0.54, mean v0 8.9, nav {'nav=follow': 24, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.095, mean delta 0.9709, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep111: n 34, share 0.088, mean delta 0.9004, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep075: n 34, share 0.063, mean delta 0.6431, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
**EMA8-15k - R7-15k** (sum -376.7 over 139 episodes)

* ep108: n 24, share -0.112, mean delta 1.7510, sharp-window frac 0.00, curved frac 0.54, mean v0 8.9, nav {'nav=follow': 24, 'nav=left': 0, 'nav=right': 0}
* ep024: n 35, share -0.103, mean delta 1.1127, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep078: n 34, share -0.099, mean delta 1.0956, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep104: n 35, share -0.053, mean delta 0.5732, sharp-window frac 0.29, curved frac 0.31, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 35, 'nav=right': 0}
* ep075: n 34, share -0.053, mean delta 0.5847, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep072: n 17, share -0.040, mean delta 0.8773, sharp-window frac 0.00, curved frac 0.82, mean v0 6.5, nav {'nav=follow': 17, 'nav=left': 0, 'nav=right': 0}
**R7-15k - refcv7-50400** (sum 739.9 over 139 episodes)

* ep016: n 29, share 0.089, mean delta 2.2669, sharp-window frac 0.28, curved frac 0.34, mean v0 7.7, nav {'nav=follow': 0, 'nav=left': 29, 'nav=right': 0}
* ep067: n 34, share 0.062, mean delta 1.3404, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep028: n 34, share 0.059, mean delta 1.2862, sharp-window frac 0.21, curved frac 0.68, mean v0 5.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep061: n 34, share 0.049, mean delta 1.0613, sharp-window frac 0.62, curved frac 1.00, mean v0 9.5, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.045, mean delta 0.9865, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep124: n 34, share 0.044, mean delta 0.9502, sharp-window frac 0.32, curved frac 0.88, mean v0 9.7, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA8-15k vs R7-15k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1564 / -0.0814 [-0.1113, -0.0527] | 54 / -0.1568 [-0.7019, 0.3764] |
| only new agrees | 358 / -0.4455 [-0.6074, -0.2945] | 49 / -1.8104 [-2.8368, -1.1120] |
| only earlier agrees | 900 / 0.0533 [-0.0260, 0.1347] | 35 / -0.1956 [-0.6538, 0.2950] |
| neither agrees | 1684 / -0.0945 [-0.2233, 0.0423] | 228 / -0.7962 [-1.4225, -0.2408] |
**EMA8-15k vs R7-15k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1545 / -0.0845 [-0.1164, -0.0561] | 52 / -0.4017 [-0.9941, 0.1864] |
| only new agrees | 369 / -0.3854 [-0.5286, -0.2579] | 58 / -1.5315 [-2.3898, -0.9407] |
| only earlier agrees | 932 / 0.0371 [-0.0447, 0.1352] | 43 / -0.2690 [-1.0052, 0.6305] |
| neither agrees | 1661 / -0.0706 [-0.2109, 0.0597] | 213 / -0.7732 [-1.4248, -0.2236] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA8-15k vs R7-15k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4477 | 0.0497 [-0.0139, 0.1126] | 0.0165 [-0.0143, 0.0507] | 0.0369 [-0.0185, 0.0934] |
| turn|straight <0.02 | 2882 | 0.0337 [-0.0102, 0.0835] | -0.0310 [-0.0482, -0.0166] | 0.0653 [0.0189, 0.1163] SEP |
| turn|gentle 0.02-0.06 | 719 | -0.0030 [-0.1151, 0.0918] | -0.0661 [-0.1205, -0.0179] | 0.0629 [-0.0605, 0.1792] |
| turn|turn 0.06-0.15 | 502 | 0.1073 [-0.0487, 0.2777] | -0.0382 [-0.1221, 0.0391] | 0.1624 [0.0139, 0.3276] SEP |
| turn|sharp >=0.15 | 365 | 0.2025 [-0.3538, 0.6720] | 0.6337 [0.3031, 1.0562] | -0.4157 [-0.9104, -0.0160] SEP |
| turn|undef (GT crawling) | 9 | -0.0348 [-0.3072, 0.1884] | -0.0275 [-0.2339, 0.1080] | 0.2104 [-0.1321, 0.7057] |
| speed|v0 <2 | 188 | -0.0442 [-0.2750, 0.1016] | 0.1199 [-0.0475, 0.3007] | -0.1070 [-0.3759, 0.1271] |
| speed|v0 2-5 | 516 | 0.1507 [-0.2666, 0.5068] | 0.0502 [-0.1799, 0.3512] | 0.1005 [-0.3380, 0.4894] |
| speed|v0 5-10 | 1390 | 0.1186 [0.0172, 0.2236] | 0.0120 [-0.0514, 0.0726] | 0.1067 [0.0100, 0.2019] SEP |
| speed|v0 10-15 | 1192 | -0.0102 [-0.0358, 0.0150] | 0.0005 [-0.0233, 0.0265] | -0.0108 [-0.0518, 0.0262] |
| speed|v0 15-25 | 856 | 0.0030 [-0.0073, 0.0130] | -0.0016 [-0.0112, 0.0088] | 0.0046 [-0.0080, 0.0176] |
| speed|v0 >=25 | 335 | -0.0057 [-0.0188, 0.0052] | 0.0117 [0.0006, 0.0224] | -0.0175 [-0.0307, -0.0057] SEP |
**EMA8-15k vs R7-15k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* EMA8-15k vs R7-15k|seed0: anchor flip rate 0.466; heading delta on flipped windows -0.0871 [-0.2202, 0.0395] (n 2091) vs unflipped -0.0894 [-0.1205, -0.0614]; goal_dist_sel proxy new-earlier -0.4704 [-0.7525, -0.2000]; corr(window dHeading, window dGoalDist) 0.032
* EMA8-15k vs R7-15k|seed1: anchor flip rate 0.473; heading delta on flipped windows -0.0622 [-0.1941, 0.0605] (n 2126) vs unflipped -0.0937 [-0.1231, -0.0683]; goal_dist_sel proxy new-earlier -0.4741 [-0.7533, -0.2018]; corr(window dHeading, window dGoalDist) 0.029

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- EMA8-15k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3527 | 139 | 3.4208 | 3.2448 | 0.1760 [0.0267, 0.3309] | **SEP** |  |
| turn: straight <0.02 | 2098 | 122 | 1.6032 | 1.4599 | 0.1433 [0.0455, 0.2540] | **SEP** | 0.595 / 0.484 / -0.0806 [-0.7031, 0.3727] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.8791 | 2.8900 | -0.0110 [-0.2169, 0.2004] |  | 0.191 / -0.012 / -0.2312 [-0.6845, 0.1810] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.4533 | 6.3633 | 0.0899 [-0.2242, 0.3913] |  | 0.123 / 0.063 / -0.0982 [-0.6160, 0.3384] |
| turn: sharp >=0.15 | 304 | 33 | 12.3655 | 11.4378 | 0.9277 [-0.2964, 2.5226] |  | 0.086 / 0.454 / 0.8227 [-0.8156, 3.1784] |
| turn: undef (GT crawling) (low n) | 15 | 11 | 12.8086 | 12.3899 | 0.4188 [0.0798, 0.9252] | **SEP** | 0.004 / 0.010 / 0.2438 [-0.3104, 1.1452] |
| speed: v0 <2 | 189 | 25 | 13.4093 | 13.6660 | -0.2567 [-1.0211, 0.1654] |  | 0.054 / -0.078 / -0.4572 [-1.9379, 0.1084] |
| speed: v0 2-5 | 422 | 52 | 4.4559 | 4.1501 | 0.3057 [-0.2512, 0.9407] |  | 0.120 / 0.208 / 0.1474 [-0.6896, 1.1550] |
| speed: v0 5-10 | 1040 | 79 | 3.8916 | 3.4941 | 0.3975 [0.0486, 0.8194] | **SEP** | 0.295 / 0.666 / 0.3141 [-0.1991, 0.9847] |
| speed: v0 10-15 | 952 | 62 | 2.5970 | 2.4870 | 0.1100 [-0.0855, 0.3231] |  | 0.270 / 0.169 / -0.0905 [-0.4844, 0.3768] |
| speed: v0 15-25 | 679 | 34 | 1.5006 | 1.4456 | 0.0550 [-0.0133, 0.1277] |  | 0.193 / 0.060 / -0.1499 [-0.4472, 0.1069] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4576 | 0.5193 | -0.0617 [-0.1611, 0.0182] |  | 0.069 / -0.024 / -0.2555 [-0.5753, -0.0082] BONF-SEP |
| nav: nav=follow | 2250 | 88 | 2.1555 | 1.9779 | 0.1776 [0.0288, 0.3565] | **SEP** | 0.638 / 0.644 / 0.0045 [-0.5020, 0.5055] |
| nav: nav=left (low n) | 319 | 13 | 9.5980 | 9.7688 | -0.1708 [-0.7216, 0.2418] |  | 0.090 / -0.088 / -0.3813 [-1.4783, 0.1871] |
| nav: nav=right | 958 | 38 | 4.3357 | 4.0480 | 0.2877 [-0.0362, 0.6499] |  | 0.272 / 0.444 / 0.1533 [-0.3643, 0.8326] |

episode concentration: 0.604 of episodes have mean delta > 0; top-5 episodes carry 0.776 of the summed delta; delta excluding the top 5 = 0.0410 (overall 0.1760).

### S6 -- EMA8-15k - R7-15k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3521 | 139 | 3.4239 | 3.7636 | -0.3397 [-0.5928, -0.1337] | **SEP** |  |
| turn: straight <0.02 | 2092 | 122 | 1.6032 | 1.6227 | -0.0195 [-0.1181, 0.0823] |  | 0.594 / 0.034 / 0.7889 [0.0190, 1.5852] BONF-SEP |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.8791 | 2.7712 | 0.1079 [-0.1851, 0.5158] |  | 0.192 / -0.061 / 0.5537 [0.0138, 1.4228] BONF-SEP |
| turn: turn 0.06-0.15 | 435 | 59 | 6.4533 | 6.9440 | -0.4908 [-0.9894, 0.0522] |  | 0.124 / 0.178 / -0.1724 [-0.9798, 0.6966] |
| turn: sharp >=0.15 | 304 | 33 | 12.3655 | 15.7206 | -3.3550 [-5.3268, -1.6193] | **SEP** | 0.086 / 0.853 / -3.3003 [-6.2484, -0.6206] BONF-SEP |
| turn: undef (GT crawling) (low n) | 15 | 11 | 12.8086 | 12.4474 | 0.3612 [-0.0685, 0.8729] |  | 0.004 / -0.005 / 0.7039 [-0.0395, 1.8230] |
| speed: v0 <2 | 183 | 25 | 13.7964 | 14.8166 | -1.0202 [-1.9674, -0.4486] | **SEP** | 0.052 / 0.156 / -0.7178 [-2.2567, 0.0835] |
| speed: v0 2-5 | 422 | 52 | 4.4559 | 4.9992 | -0.5433 [-1.8699, 0.5196] |  | 0.120 / 0.192 / -0.2313 [-2.0332, 1.5608] |
| speed: v0 5-10 | 1040 | 79 | 3.8916 | 4.1750 | -0.2834 [-0.6235, 0.0417] |  | 0.295 / 0.246 / 0.0799 [-0.4512, 0.5951] |
| speed: v0 10-15 | 952 | 62 | 2.5970 | 2.9686 | -0.3716 [-0.6611, -0.1039] | **SEP** | 0.270 / 0.296 / -0.0438 [-0.5845, 0.4942] |
| speed: v0 15-25 | 679 | 34 | 1.5006 | 1.6750 | -0.1744 [-0.2750, -0.0900] | **SEP** | 0.193 / 0.099 / 0.2047 [-0.2207, 0.7023] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4576 | 0.5110 | -0.0534 [-0.1448, 0.0347] |  | 0.070 / 0.011 / 0.3077 [-0.0745, 0.7968] |
| nav: nav=follow | 2244 | 88 | 2.1570 | 2.4597 | -0.3027 [-0.5021, -0.1247] | **SEP** | 0.637 / 0.568 / 0.1020 [-0.6472, 1.1302] |
| nav: nav=left (low n) | 319 | 13 | 9.5980 | 10.7060 | -1.1080 [-2.9633, -0.0723] | **SEP** | 0.091 / 0.296 / -0.8449 [-4.6788, 0.4572] |
| nav: nav=right | 958 | 38 | 4.3357 | 4.5062 | -0.1705 [-0.6891, 0.3457] |  | 0.272 / 0.137 / 0.2324 [-0.6925, 1.1100] |

episode concentration: 0.345 of episodes have mean delta > 0; top-5 episodes carry -0.241 of the summed delta; delta excluding the top 5 = -0.4375 (overall -0.3397).

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

### S6 -- EMA8-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3493 | 139 | 1.0010 | 0.7640 | 0.2370 [0.0084, 0.6640] | **SEP** | -0.0429 [-0.0867, -0.0014] SEP |
| seg1 | straight | 2071 | 122 | 0.6664 | 0.5434 | 0.1229 [0.0144, 0.3225] | **SEP** | 0.0461 [-0.0693, 0.2501] |
| seg1 | curved | 1408 | 88 | 1.3694 | 0.9644 | 0.4049 [-0.0228, 1.2557] |  | -0.1723 [-0.4852, 0.0274] |
| seg2 | all | 3478 | 139 | 1.5497 | 1.4593 | 0.0904 [0.0174, 0.1646] | **SEP** | 0.1486 [-0.1369, 0.6180] |
| seg2 | straight | 2072 | 122 | 1.0796 | 0.8398 | 0.2398 [0.0185, 0.6413] | **SEP** | 0.2702 [-0.1374, 1.0458] |
| seg2 | curved | 1406 | 88 | 2.2425 | 2.3722 | -0.1297 [-0.7091, 0.2393] |  | -0.0305 [-0.2618, 0.1880] |
| seg3 | all | 3444 | 139 | 2.2113 | 2.0620 | 0.1492 [0.0276, 0.2935] | **SEP** | 0.4405 [-0.0946, 1.3511] |
| seg3 | straight | 2053 | 121 | 1.1381 | 0.9786 | 0.1595 [0.0171, 0.3823] | **SEP** | 0.5958 [-0.0793, 1.9405] |
| seg3 | curved | 1391 | 88 | 3.7952 | 3.6612 | 0.1340 [-0.2971, 0.5324] |  | 0.2113 [-0.2141, 0.6431] |
| seg4 | all | 3405 | 139 | 3.1955 | 2.9513 | 0.2443 [0.0593, 0.4522] | **SEP** | 0.4445 [-0.0094, 1.1489] |
| seg4 | straight | 2038 | 121 | 1.2835 | 1.1392 | 0.1443 [0.0430, 0.2736] | **SEP** | 0.5479 [-0.0742, 1.7206] |
| seg4 | curved | 1367 | 87 | 6.0461 | 5.6528 | 0.3932 [-0.0304, 0.9064] |  | 0.2905 [-0.1290, 0.7499] |
| seg5 | all | 3358 | 139 | 4.5087 | 4.2754 | 0.2334 [-0.0201, 0.5048] |  | 0.3259 [-0.0203, 0.7502] |
| seg5 | straight | 2011 | 121 | 1.4018 | 1.3099 | 0.0919 [-0.1656, 0.3204] |  | 0.2616 [-0.1312, 0.9034] |
| seg5 | curved | 1347 | 88 | 9.1471 | 8.7027 | 0.4445 [-0.0850, 1.0505] |  | 0.4219 [-0.1457, 1.0290] |
| seg6 | all | 3315 | 138 | 6.2196 | 5.8849 | 0.3347 [0.0500, 0.6486] | **SEP** | 0.0886 [-0.2050, 0.4231] |
| seg6 | straight | 2001 | 120 | 1.8935 | 1.7279 | 0.1656 [-0.0090, 0.3610] |  | -0.0331 [-0.2730, 0.1827] |
| seg6 | curved | 1313 | 88 | 12.8147 | 12.2245 | 0.5901 [-0.0712, 1.3123] |  | 0.2717 [-0.3939, 0.9970] |

### S6 -- EMA8-15k - R7-15k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3487 | 139 | 1.0020 | 0.9187 | 0.0833 [-0.1028, 0.4047] |  | -0.2962 [-0.6187, -0.1097] SEP |
| seg1 | straight | 2068 | 122 | 0.6672 | 0.6857 | -0.0185 [-0.0499, 0.0170] |  | -0.3294 [-0.7295, -0.1211] SEP |
| seg1 | curved | 1405 | 88 | 1.3707 | 1.1392 | 0.2316 [-0.2068, 1.0654] |  | -0.2485 [-0.5462, -0.0573] SEP |
| seg2 | all | 3472 | 139 | 1.5509 | 1.6553 | -0.1044 [-0.2235, 0.0128] |  | -0.2128 [-0.3499, -0.0930] SEP |
| seg2 | straight | 2067 | 121 | 1.0806 | 0.9929 | 0.0877 [-0.0494, 0.3043] |  | -0.1431 [-0.2988, 0.0764] |
| seg2 | curved | 1405 | 88 | 2.2428 | 2.6299 | -0.3871 [-0.7887, -0.0448] | **SEP** | -0.3153 [-0.6879, 0.0028] |
| seg3 | all | 3437 | 139 | 2.2149 | 2.3622 | -0.1474 [-0.3598, 0.0472] |  | -0.3364 [-0.6996, 0.1123] |
| seg3 | straight | 2046 | 121 | 1.1404 | 1.1599 | -0.0194 [-0.1652, 0.1984] |  | -0.1126 [-0.4546, 0.4912] |
| seg3 | curved | 1391 | 88 | 3.7952 | 4.1307 | -0.3355 [-0.8059, 0.0839] |  | -0.6656 [-1.2392, -0.0875] SEP |
| seg4 | all | 3396 | 139 | 3.2020 | 3.5551 | -0.3532 [-0.6499, -0.0829] | **SEP** | -0.4783 [-0.9458, 0.0355] |
| seg4 | straight | 2029 | 121 | 1.2858 | 1.3147 | -0.0289 [-0.1943, 0.2003] |  | -0.2386 [-0.6102, 0.3912] |
| seg4 | curved | 1367 | 87 | 6.0461 | 6.8806 | -0.8345 [-1.4955, -0.1890] | **SEP** | -0.8340 [-1.7342, 0.0841] |
| seg5 | all | 3354 | 139 | 4.5113 | 5.0322 | -0.5210 [-0.9204, -0.1807] | **SEP** | -0.7481 [-1.3149, -0.1841] SEP |
| seg5 | straight | 2007 | 121 | 1.3999 | 1.3361 | 0.0638 [-0.1323, 0.3181] |  | -0.7152 [-1.3788, -0.3250] SEP |
| seg5 | curved | 1347 | 88 | 9.1471 | 10.5394 | -1.3923 [-2.2513, -0.5748] | **SEP** | -0.7971 [-1.9767, 0.3934] |
| seg6 | all | 3310 | 138 | 6.2107 | 6.7799 | -0.5692 [-1.0342, -0.0594] | **SEP** | -0.6758 [-1.4185, 0.0019] |
| seg6 | straight | 1997 | 120 | 1.8882 | 1.7118 | 0.1764 [-0.1497, 0.6597] |  | -0.6866 [-1.6322, -0.1724] SEP |
| seg6 | curved | 1312 | 88 | 12.7922 | 14.4992 | -1.7071 [-2.6379, -0.7642] | **SEP** | -0.6623 [-2.0907, 0.6889] |

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

### S6 -- EMA8-15k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 364 | 39 | 5.3515 | 5.1747 | 0.1768 [-0.0202, 0.4568] |  |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 1.0025 | 0.7632 | 0.2393 [0.0759, 0.4338] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.5305 | 0.5523 | -0.0219 [-0.0712, 0.0240] |  |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 98 | 22 | 5.4087 | 5.5283 | -0.1196 [-0.8818, 0.3654] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.4992 | 2.5310 | -0.0317 [-0.2864, 0.2885] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.3115 | 2.1952 | 0.1162 [-0.0499, 0.2974] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 64 | 20 | 10.7059 | 11.2985 | -0.5926 [-1.3838, 0.0815] |  |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 5.9912 | 5.8272 | 0.1640 [-0.1863, 0.5283] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 4.6997 | 4.3281 | 0.3716 [0.1258, 0.7146] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 70 | 16 | 15.1347 | 14.2844 | 0.8502 [-1.5993, 4.4273] |  |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 11.7858 | 10.7541 | 1.0317 [-0.4431, 2.7565] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 5.3219 | 6.3909 | -1.0689 [-2.0364, 1.6319] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 15 | 11 | 12.8086 | 12.3899 | 0.4188 [0.0798, 0.9252] | **SEP** |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA8-15k s0-s1 [CI] | R7-15k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0062 [-0.0477, 0.0363] | 0.0385 [-0.0243, 0.1139] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0084 [-0.0312, 0.0115] | -0.0089 [-0.0410, 0.0275] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0191 [-0.0540, 0.0983] | 0.0091 [-0.0537, 0.0850] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.1019 [-0.2216, 0.0227] | 0.0410 [-0.0646, 0.1414] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.0928 [-0.2494, 0.4389] | 0.4196 [-0.0921, 1.0546] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | -0.0822 [-0.5704, 0.3910] | 0.0169 [-0.2314, 0.3191] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | -0.1561 [-0.3462, -0.0084] SEP | 0.5111 [-0.0175, 1.3551] | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.0806 [-0.2142, 0.0344] | 0.2492 [0.0195, 0.5701] SEP | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | 0.0315 [-0.0543, 0.1270] | -0.0596 [-0.1349, 0.0154] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | 0.0076 [-0.0473, 0.0831] | -0.0267 [-0.0712, 0.0117] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | -0.0017 [-0.0317, 0.0242] | 0.0013 [-0.0456, 0.0457] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | 0.0121 [-0.0033, 0.0310] | -0.0103 [-0.0282, 0.0092] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0074 [-0.0492, 0.0343] | -0.0124 [-0.0413, 0.0165] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.0066 [-0.1012, 0.0914] | 0.4812 [0.0017, 1.1772] SEP | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | -0.0078 [-0.1120, 0.1111] | 0.0029 [-0.0995, 0.1193] | 0.0235 [-0.1175, 0.1775] |

### S6 -- EMA8-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1760 | 0.1624 / 0.1895 | True / True / True | 0.0062 / 0.0128 | 13.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.1433 | 0.1516 / 0.1349 | True / True / True | 0.0084 / 0.0216 | 6.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | -0.0110 | -0.0375 / 0.0156 | False / False / False | 0.0191 / 0.0713 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0899 | 0.0169 / 0.1629 | False / False / False | 0.1019 / 0.0611 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.9277 | 0.8805 / 0.9750 | False / False / False | 0.0928 / 0.2004 | 4.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.4188 | 0.3414 / 0.4961 | True / False / True | 0.0822 / 1.9278 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.2567 | -0.4268 / -0.0841 | False / False / False | 0.1561 / 0.0328 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.3057 | 0.1650 / 0.4465 | False / False / False | 0.0806 / 0.2009 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.3975 | 0.4253 / 0.3696 | True / True / True | 0.0315 / 0.0243 | 12.6 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1100 | 0.1271 / 0.0928 | False / False / False | 0.0076 / 0.0268 | 4.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0550 | 0.0514 / 0.0586 | False / False / False | 0.0017 / 0.0055 | 10.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0617 | -0.0561 / -0.0674 | False / False / False | 0.0121 / 0.0008 | 5.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1776 | 0.1687 / 0.1863 | True / True / True | 0.0074 / 0.0080 | 22.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.1708 | -0.1641 / -0.1775 | False / False / False | 0.0066 / 0.0140 | 12.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.2877 | 0.2563 / 0.3190 | False / False / False | 0.0078 / 0.0235 | 12.3 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- EMA8-15k - R7-15k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | -0.3397 | -0.3466 / -0.3327 | True / True / True | 0.0062 / 0.0385 | 8.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | -0.0195 | -0.0103 / -0.0289 | False / False / False | 0.0084 / 0.0089 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.1079 | 0.1144 / 0.1013 | False / False / False | 0.0191 / 0.0091 | 5.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.4908 | -0.5456 / -0.4359 | True / False / False | 0.1019 / 0.0410 | 4.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | -3.3550 | -3.4269 / -3.2832 | True / True / True | 0.0928 / 0.4196 | 8.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.3612 | 0.2089 / 0.5135 | False / False / False | 0.0822 / 0.0169 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -1.0202 | -1.1307 / -0.9020 | True / True / True | 0.1561 / 0.5111 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.5433 | -0.7082 / -0.3784 | False / False / False | 0.0806 / 0.2492 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.2834 | -0.2378 / -0.3290 | False / True / False | 0.0315 / 0.0596 | 4.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.3716 | -0.3545 / -0.3888 | True / True / True | 0.0076 / 0.0267 | 13.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 15-25 | -0.1744 | -0.1759 / -0.1729 | True / True / True | 0.0017 / 0.0013 | 101.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0534 | -0.0422 / -0.0646 | False / False / False | 0.0121 / 0.0103 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.3027 | -0.2996 / -0.3057 | True / True / True | 0.0074 / 0.0124 | 24.5 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -1.1080 | -1.1888 / -1.0273 | True / True / True | 0.0066 / 0.4812 | 2.3 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=right | -0.1705 | -0.1763 / -0.1647 | False / False / False | 0.0078 / 0.0029 | 21.9 | NOT PROVEN (not separated and/or within 2x floor) |

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

* seg1 (n 1405): EMA8-15k bias -0.019, refcv7-50400 bias +0.167, R7-15k bias +0.154; EMA8-15k MAE 1.371, refcv7-50400 MAE 0.965, R7-15k MAE 1.139
* seg2 (n 1405): EMA8-15k bias -0.426, refcv7-50400 bias -0.234, R7-15k bias -0.654; EMA8-15k MAE 2.243, refcv7-50400 MAE 2.373, R7-15k MAE 2.630
* seg3 (n 1391): EMA8-15k bias -1.374, refcv7-50400 bias -1.098, R7-15k bias -2.372; EMA8-15k MAE 3.795, refcv7-50400 MAE 3.661, R7-15k MAE 4.131
* seg4 (n 1367): EMA8-15k bias -2.963, refcv7-50400 bias -2.462, R7-15k bias -4.653; EMA8-15k MAE 6.046, refcv7-50400 MAE 5.653, R7-15k MAE 6.881
* seg5 (n 1347): EMA8-15k bias -4.834, refcv7-50400 bias -4.384, R7-15k bias -7.256; EMA8-15k MAE 9.147, refcv7-50400 MAE 8.703, R7-15k MAE 10.539
* seg6 (n 1311): EMA8-15k bias -6.885, refcv7-50400 bias -6.758, R7-15k bias -9.331; EMA8-15k MAE 12.798, refcv7-50400 MAE 12.229, R7-15k MAE 14.504
**sharp (>= 0.15)**

* seg1 (n 301): EMA8-15k bias -0.665, refcv7-50400 bias +0.178, R7-15k bias -0.105; EMA8-15k MAE 2.011, refcv7-50400 MAE 1.446, R7-15k MAE 1.821
* seg2 (n 298): EMA8-15k bias -1.477, refcv7-50400 bias -0.532, R7-15k bias -2.816; EMA8-15k MAE 4.215, refcv7-50400 MAE 3.819, R7-15k MAE 5.214
* seg3 (n 295): EMA8-15k bias -4.497, refcv7-50400 bias -2.422, R7-15k bias -7.949; EMA8-15k MAE 7.786, refcv7-50400 MAE 6.681, R7-15k MAE 9.763
* seg4 (n 293): EMA8-15k bias -9.112, refcv7-50400 bias -6.245, R7-15k bias -14.751; EMA8-15k MAE 13.109, refcv7-50400 MAE 11.506, R7-15k MAE 17.093
* seg5 (n 289): EMA8-15k bias -14.242, refcv7-50400 bias -11.337, R7-15k bias -21.911; EMA8-15k MAE 20.486, refcv7-50400 MAE 18.631, R7-15k MAE 25.907
* seg6 (n 283): EMA8-15k bias -18.885, refcv7-50400 bias -16.801, R7-15k bias -25.460; EMA8-15k MAE 27.041, refcv7-50400 MAE 24.710, R7-15k MAE 33.154
**curved & v0 5-10**

* seg1 (n 547): EMA8-15k bias +0.030, refcv7-50400 bias +0.049, R7-15k bias +0.061; EMA8-15k MAE 0.971, refcv7-50400 MAE 0.896, R7-15k MAE 1.142
* seg2 (n 547): EMA8-15k bias -0.606, refcv7-50400 bias -0.516, R7-15k bias -0.897; EMA8-15k MAE 2.550, refcv7-50400 MAE 2.231, R7-15k MAE 2.791
* seg3 (n 545): EMA8-15k bias -1.933, refcv7-50400 bias -1.649, R7-15k bias -2.874; EMA8-15k MAE 4.371, refcv7-50400 MAE 3.822, R7-15k MAE 4.488
* seg4 (n 532): EMA8-15k bias -3.842, refcv7-50400 bias -3.511, R7-15k bias -5.389; EMA8-15k MAE 6.764, refcv7-50400 MAE 6.171, R7-15k MAE 7.468
* seg5 (n 515): EMA8-15k bias -6.085, refcv7-50400 bias -5.765, R7-15k bias -8.060; EMA8-15k MAE 9.997, refcv7-50400 MAE 9.280, R7-15k MAE 11.014
* seg6 (n 487): EMA8-15k bias -8.067, refcv7-50400 bias -8.083, R7-15k bias -9.881; EMA8-15k MAE 13.352, refcv7-50400 MAE 12.451, R7-15k MAE 14.706
**sharp & v0 5-10**

* seg1 (n 139): EMA8-15k bias -0.037, refcv7-50400 bias +0.062, R7-15k bias +0.172; EMA8-15k MAE 1.259, refcv7-50400 MAE 1.251, R7-15k MAE 1.664
* seg2 (n 139): EMA8-15k bias -2.021, refcv7-50400 bias -1.267, R7-15k bias -2.425; EMA8-15k MAE 4.256, refcv7-50400 MAE 3.564, R7-15k MAE 5.063
* seg3 (n 139): EMA8-15k bias -6.144, refcv7-50400 bias -4.133, R7-15k bias -8.469; EMA8-15k MAE 8.548, refcv7-50400 MAE 7.029, R7-15k MAE 9.896
* seg4 (n 136): EMA8-15k bias -11.668, refcv7-50400 bias -8.928, R7-15k bias -15.835; EMA8-15k MAE 14.471, refcv7-50400 MAE 12.348, R7-15k MAE 17.410
* seg5 (n 135): EMA8-15k bias -17.059, refcv7-50400 bias -14.156, R7-15k bias -22.149; EMA8-15k MAE 22.735, refcv7-50400 MAE 19.609, R7-15k MAE 25.512
* seg6 (n 130): EMA8-15k bias -19.848, refcv7-50400 bias -17.837, R7-15k bias -23.974; EMA8-15k MAE 28.643, refcv7-50400 MAE 25.199, R7-15k MAE 31.661

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1311 / 1311 | 26.0 | EMA8-15k -0.2562 [-0.3527, -0.1725]; refcv7-50400 -0.2414 [-0.3105, -0.1759]; R7-15k -0.3581 [-0.4429, -0.2780] | EMA8-15k -0.0149 [-0.0625, +0.0213]; R7-15k -0.1167 [-0.1665, -0.0703] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1414 / 6821 | 16.3 | EMA8-15k -0.1982 [-0.2909, -0.1211]; refcv7-50400 -0.1744 [-0.2383, -0.1169]; R7-15k -0.2970 [-0.3803, -0.2203] | EMA8-15k -0.0238 [-0.0719, +0.0156]; R7-15k -0.1225 [-0.1765, -0.0654] SEP |
| sharp (>= 0.15) | last segment | 283 / 283 | 63.0 | EMA8-15k -0.2857 [-0.4116, -0.1871]; refcv7-50400 -0.2564 [-0.3394, -0.1824]; R7-15k -0.3910 [-0.5005, -0.2934] | EMA8-15k -0.0294 [-0.0976, +0.0167]; R7-15k -0.1346 [-0.1969, -0.0812] SEP |
| sharp (>= 0.15) | segments 2..K | 304 / 1458 | 39.8 | EMA8-15k -0.2379 [-0.3613, -0.1396]; refcv7-50400 -0.1883 [-0.2712, -0.1189]; R7-15k -0.3608 [-0.4716, -0.2677] | EMA8-15k -0.0496 [-0.1202, -0.0020] SEP; R7-15k -0.1725 [-0.2366, -0.1204] SEP |
| curved & v0 5-10 | last segment | 487 / 487 | 28.2 | EMA8-15k -0.2825 [-0.4134, -0.1602]; refcv7-50400 -0.2585 [-0.3806, -0.1637]; R7-15k -0.3446 [-0.4495, -0.2515] | EMA8-15k -0.0241 [-0.1162, +0.0504]; R7-15k -0.0861 [-0.1420, -0.0308] SEP |
| curved & v0 5-10 | segments 2..K | 547 / 2626 | 17.7 | EMA8-15k -0.2367 [-0.3565, -0.1303]; refcv7-50400 -0.1986 [-0.3113, -0.1151]; R7-15k -0.3089 [-0.3991, -0.2272] | EMA8-15k -0.0381 [-0.1224, +0.0300]; R7-15k -0.1103 [-0.1618, -0.0568] SEP |
| sharp & v0 5-10 | last segment | 130 / 130 | 67.1 | EMA8-15k -0.2967 [-0.4571, -0.1567]; refcv7-50400 -0.2565 [-0.3869, -0.1508]; R7-15k -0.3553 [-0.4741, -0.2527] | EMA8-15k -0.0402 [-0.1551, +0.0493]; R7-15k -0.0988 [-0.1591, -0.0506] SEP |
| sharp & v0 5-10 | segments 2..K | 139 / 679 | 43.8 | EMA8-15k -0.2585 [-0.4084, -0.1306]; refcv7-50400 -0.2031 [-0.3332, -0.1089]; R7-15k -0.3325 [-0.4420, -0.2368] | EMA8-15k -0.0554 [-0.1632, +0.0279]; R7-15k -0.1294 [-0.1891, -0.0774] SEP |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA8-15k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3527 | 0.1760 | 0.03006 | 0.04284 | 0.540 / 0.460 / 0.000 | 696.4 / 620.7 |
| turn|straight <0.02 | 2098 | 0.1433 | 0.04183 | 0.04666 | 0.568 / 0.432 / 0.000 | 226.8 / 300.7 |
| turn|gentle 0.02-0.06 | 675 | -0.0110 | -0.01698 | -0.03020 | 0.486 / 0.514 / 0.000 | 57.1 / -7.4 |
| turn|turn 0.06-0.15 | 435 | 0.0899 | 0.04127 | 0.11620 | 0.529 / 0.471 / 0.000 | 44.6 / 39.1 |
| turn|sharp >=0.15 | 304 | 0.9277 | -0.12407 | 0.21739 | 0.480 / 0.520 / 0.000 | 167.1 / 282.0 |
| turn|undef (GT crawling) | 15 | 0.4188 | 0.33794 | 0.27458 | 0.733 / 0.267 / 0.000 | 3.2 / 6.3 |
| speed|v0 <2 | 189 | -0.2567 | 0.00038 | 0.00785 | 0.503 / 0.497 / 0.000 | 8.9 / -48.5 |
| speed|v0 2-5 | 422 | 0.3057 | 0.07127 | 0.11502 | 0.545 / 0.455 / 0.000 | 120.8 / 129.0 |
| speed|v0 5-10 | 1040 | 0.3975 | 0.02477 | 0.06251 | 0.523 / 0.477 / 0.000 | 354.2 / 413.4 |
| speed|v0 10-15 | 952 | 0.1100 | 0.06512 | 0.07245 | 0.613 / 0.387 / 0.000 | 84.2 / 104.7 |
| speed|v0 15-25 | 679 | 0.0550 | 0.00902 | 0.00959 | 0.513 / 0.487 / 0.000 | 35.0 / 37.3 |
| speed|v0 >=25 | 245 | -0.0617 | -0.03928 | -0.03858 | 0.429 / 0.571 / 0.000 | 1.9 / -15.1 |
| nav|nav=follow | 2250 | 0.1776 | 0.01636 | 0.02476 | 0.528 / 0.472 / 0.000 | 340.4 / 399.6 |
| nav|nav=left | 319 | -0.1708 | 0.05857 | 0.09148 | 0.596 / 0.404 / 0.000 | 35.7 / -54.5 |
| nav|nav=right | 958 | 0.2877 | 0.07619 | 0.07226 | 0.550 / 0.450 / 0.000 | 294.3 / 275.6 |
**EMA8-15k - R7-15k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3521 | -0.3397 | -0.08338 | -0.14770 | 0.411 / 0.589 / 0.000 | 659.4 / -1196.0 |
| turn|straight <0.02 | 2092 | -0.0195 | -0.05073 | -0.06955 | 0.421 / 0.579 / 0.000 | 214.9 / -40.8 |
| turn|gentle 0.02-0.06 | 675 | 0.1079 | -0.08968 | -0.09772 | 0.449 / 0.551 / 0.000 | 151.0 / 72.8 |
| turn|turn 0.06-0.15 | 435 | -0.4908 | -0.28944 | -0.49542 | 0.379 / 0.621 / 0.000 | 88.2 / -213.5 |
| turn|sharp >=0.15 | 304 | -3.3550 | -1.52602 | -2.48091 | 0.296 / 0.704 / 0.000 | 95.2 / -1019.9 |
| turn|undef (GT crawling) | 15 | 0.3612 | 0.11337 | 0.20792 | 0.600 / 0.400 / 0.000 | 3.1 / 5.4 |
| speed|v0 <2 | 183 | -1.0202 | -0.17445 | -0.37618 | 0.328 / 0.672 / 0.000 | 5.8 / -186.7 |
| speed|v0 2-5 | 422 | -0.5433 | -0.25490 | -0.23608 | 0.389 / 0.611 / 0.000 | 152.0 / -229.3 |
| speed|v0 5-10 | 1040 | -0.2834 | -0.10603 | -0.16654 | 0.423 / 0.577 / 0.000 | 237.9 / -294.7 |
| speed|v0 10-15 | 952 | -0.3716 | -0.07735 | -0.13027 | 0.410 / 0.590 / 0.000 | 108.0 / -353.8 |
| speed|v0 15-25 | 679 | -0.1744 | -0.04889 | -0.09530 | 0.412 / 0.588 / 0.000 | 29.5 / -118.4 |
| speed|v0 >=25 | 245 | -0.0534 | -0.02619 | -0.02885 | 0.461 / 0.539 / 0.000 | 2.0 / -13.1 |
| nav|nav=follow | 2244 | -0.3027 | -0.08266 | -0.14095 | 0.403 / 0.597 / 0.000 | 279.5 / -679.2 |
| nav|nav=left | 319 | -1.1080 | -0.08401 | -0.20685 | 0.408 / 0.592 / 0.000 | 60.3 / -353.5 |
| nav|nav=right | 958 | -0.1705 | -0.08884 | -0.13410 | 0.431 / 0.569 / 0.000 | 276.1 / -163.4 |
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

**EMA8-15k - refcv7-50400** (sum 620.7 over 139 episodes)

* ep111: n 26, share 0.222, mean delta 5.3025, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.171, mean delta 4.0729, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep013: n 27, share 0.141, mean delta 3.2369, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep059: n 27, share 0.122, mean delta 2.7936, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.121, mean delta 2.8840, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep012: n 26, share 0.064, mean delta 1.5216, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
**EMA8-15k - R7-15k** (sum -1196.0 over 139 episodes)

* ep078: n 26, share -0.093, mean delta 4.2599, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep059: n 27, share -0.051, mean delta 2.2517, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep029: n 26, share -0.040, mean delta 1.8307, sharp-window frac 0.00, curved frac 1.00, mean v0 12.1, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep137: n 23, share -0.029, mean delta 1.5292, sharp-window frac 0.39, curved frac 1.00, mean v0 4.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 23}
* ep111: n 26, share -0.028, mean delta 1.3056, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep108: n 22, share -0.027, mean delta 1.4899, sharp-window frac 0.00, curved frac 0.45, mean v0 9.6, nav {'nav=follow': 22, 'nav=left': 0, 'nav=right': 0}
**R7-15k - refcv7-50400** (sum 1961.4 over 139 episodes)

* ep016: n 22, share 0.141, mean delta 12.5467, sharp-window frac 0.41, curved frac 0.59, mean v0 5.6, nav {'nav=follow': 0, 'nav=left': 22, 'nav=right': 0}
* ep085: n 26, share 0.091, mean delta 6.8567, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep028: n 27, share 0.069, mean delta 5.0322, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep061: n 26, share 0.060, mean delta 4.5602, sharp-window frac 0.88, curved frac 1.00, mean v0 9.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.053, mean delta 3.9969, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep013: n 27, share 0.051, mean delta 3.7104, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA8-15k vs R7-15k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1242 / -0.1617 [-0.2178, -0.1107] | 39 / -1.3442 [-2.3500, -0.7346] |
| only new agrees | 295 / -2.0023 [-2.7874, -1.2901] | 44 / -8.3535 [-12.6154, -5.2505] |
| only earlier agrees | 658 / 0.3163 [0.0799, 0.5784] | 17 / 2.5948 [-1.1567, 6.6285] |
| neither agrees | 1326 / -0.4804 [-1.0187, -0.0482] | 204 / -3.2643 [-5.4692, -1.1895] |
**EMA8-15k vs R7-15k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1236 / -0.1829 [-0.2428, -0.1316] | 38 / -1.5197 [-2.9247, -0.8749] |
| only new agrees | 308 / -2.1775 [-3.0722, -1.3541] | 52 / -8.4378 [-12.8504, -5.0272] |
| only earlier agrees | 675 / 0.4038 [0.1346, 0.7117] | 28 / 3.0802 [-1.1513, 7.8106] |
| neither agrees | 1304 / -0.4201 [-0.9453, 0.0676] | 186 / -3.1603 [-6.0154, -0.7599] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA8-15k vs R7-15k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3477 | 0.1712 [-0.0401, 0.3918] | 0.1371 [0.0004, 0.2890] | 0.0391 [-0.1286, 0.2188] |
| turn|straight <0.02 | 2069 | -0.0561 [-0.1608, 0.0230] | -0.0618 [-0.1170, -0.0192] | -0.0030 [-0.1038, 0.0880] |
| turn|gentle 0.02-0.06 | 675 | -0.1769 [-0.6635, 0.1963] | -0.1127 [-0.2438, 0.0024] | -0.0356 [-0.4239, 0.2810] |
| turn|turn 0.06-0.15 | 425 | 0.1911 [-0.1608, 0.5216] | -0.1208 [-0.4650, 0.2215] | 0.3510 [0.0141, 0.6748] SEP |
| turn|sharp >=0.15 | 301 | 2.4945 [0.7321, 4.4620] | 2.4018 [1.1364, 3.9775] | 0.0421 [-1.7024, 1.6690] |
| turn|undef (GT crawling) | 7 | -0.0195 [-0.5854, 0.5401] | -0.0902 [-0.6115, 0.3636] | 0.5956 [0.1271, 1.3141] SEP |
| speed|v0 <2 | 139 | 0.2680 [0.0426, 0.5587] | 0.4735 [-0.0257, 1.1232] | -0.3025 [-1.5946, 0.4373] |
| speed|v0 2-5 | 422 | 0.0637 [-1.1347, 1.1446] | 0.1310 [-0.4453, 0.7842] | -0.0672 [-1.2418, 0.9812] |
| speed|v0 5-10 | 1040 | 0.5104 [0.0600, 0.9608] | 0.3403 [-0.0556, 0.7960] | 0.1701 [-0.1685, 0.4663] |
| speed|v0 10-15 | 952 | -0.0344 [-0.1250, 0.0560] | -0.0435 [-0.1644, 0.0865] | 0.0091 [-0.1165, 0.1409] |
| speed|v0 15-25 | 679 | 0.0520 [-0.0070, 0.1232] | 0.0007 [-0.0521, 0.0591] | 0.0514 [-0.0020, 0.1252] |
| speed|v0 >=25 | 245 | -0.0116 [-0.0362, 0.0077] | 0.0465 [0.0124, 0.0901] | -0.0581 [-0.1046, -0.0176] SEP |
**EMA8-15k vs R7-15k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* EMA8-15k vs R7-15k|seed0: anchor flip rate 0.455; heading delta on flipped windows -0.5595 [-1.0846, -0.1029] (n 1598) vs unflipped -0.1697 [-0.2332, -0.1140]; goal_dist_sel proxy new-earlier -0.4985 [-0.7820, -0.2090]; corr(window dHeading, window dGoalDist) 0.002
* EMA8-15k vs R7-15k|seed1: anchor flip rate 0.460; heading delta on flipped windows -0.4991 [-1.0587, -0.0324] (n 1617) vs unflipped -0.1915 [-0.2513, -0.1420]; goal_dist_sel proxy new-earlier -0.5066 [-0.7893, -0.2180]; corr(window dHeading, window dGoalDist) 0.000
