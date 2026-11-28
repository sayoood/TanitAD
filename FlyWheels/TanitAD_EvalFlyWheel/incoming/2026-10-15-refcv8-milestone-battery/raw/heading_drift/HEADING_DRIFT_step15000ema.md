# refcv8 heading-drift localisation: EMA-15k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-17T03:21:20; trees: `EMA-15k` = E:/refcv8_eval_kit/battery/step15000ema_unc; `EMA-10.5k` = E:/refcv8_eval_kit/battery/step10500ema_unc; `EMA-5k` = C:/Users/Admin/refcv8_eval_archive/battery/step5000ema_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- EMA-15k - refcv7-50400, 2-seed mean of the per-window metric

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

### S2 -- EMA-15k - EMA-10.5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4509 | 139 | 1.4846 | 1.4532 | 0.0313 [0.0062, 0.0608] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.2129 | 1.1884 | 0.0244 [0.0039, 0.0511] | **SEP** | 0.643 / 0.501 / -0.0194 [-0.1070, 0.0515] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0815 | 0.9896 | 0.0919 [0.0097, 0.2035] | **SEP** | 0.160 / 0.468 / 0.0720 [-0.0338, 0.2404] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9488 | 1.9557 | -0.0069 [-0.0504, 0.0461] |  | 0.113 / -0.025 / -0.0431 [-0.1278, 0.0414] |
| turn: sharp >=0.15 | 366 | 42 | 3.3408 | 3.3195 | 0.0213 [-0.1102, 0.1838] |  | 0.081 / 0.055 / -0.0109 [-0.2093, 0.3078] |
| turn: undef (GT crawling) (low n) | 17 | 13 | 11.0488 | 11.0413 | 0.0075 [-0.0479, 0.0676] |  | 0.004 / 0.001 / -0.0239 [-0.1312, 0.0778] |
| speed: v0 <2 | 220 | 37 | 13.4259 | 13.3927 | 0.0332 [-0.0942, 0.2379] |  | 0.049 / 0.052 / 0.0019 [-0.1911, 0.4233] |
| speed: v0 2-5 | 516 | 58 | 1.9464 | 1.8093 | 0.1371 [-0.0030, 0.3312] |  | 0.114 / 0.501 / 0.1194 [-0.0677, 0.4913] |
| speed: v0 5-10 | 1390 | 84 | 1.2090 | 1.1784 | 0.0306 [-0.0084, 0.0840] |  | 0.308 / 0.301 / -0.0011 [-0.0892, 0.1092] |
| speed: v0 10-15 | 1192 | 64 | 0.6019 | 0.5878 | 0.0141 [-0.0063, 0.0411] |  | 0.264 / 0.119 / -0.0234 [-0.0930, 0.0340] |
| speed: v0 15-25 | 856 | 35 | 0.3348 | 0.3330 | 0.0018 [-0.0042, 0.0082] |  | 0.190 / 0.011 / -0.0365 [-0.0962, 0.0047] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1538 | 0.1466 | 0.0072 [-0.0030, 0.0204] |  | 0.074 / 0.017 / -0.0261 [-0.0827, 0.0150] |
| nav: nav=follow | 2880 | 88 | 0.6812 | 0.6477 | 0.0335 [0.0068, 0.0671] | **SEP** | 0.639 / 0.682 / 0.0059 [-0.1016, 0.0879] |
| nav: nav=left (low n) | 402 | 13 | 6.9194 | 6.9332 | -0.0138 [-0.0653, 0.0272] |  | 0.089 / -0.039 / -0.0495 [-0.1453, 0.0242] |
| nav: nav=right | 1227 | 38 | 1.5897 | 1.5486 | 0.0411 [-0.0125, 0.1263] |  | 0.272 / 0.357 / 0.0134 [-0.0746, 0.1474] |

episode concentration: 0.554 of episodes have mean delta > 0; top-5 episodes carry 0.855 of the summed delta; delta excluding the top 5 = 0.0047 (overall 0.0313).

### S2 -- EMA-10.5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4510 | 139 | 1.4542 | 1.4077 | 0.0464 [0.0007, 0.0919] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.1884 | 1.1597 | 0.0287 [-0.0053, 0.0648] |  | 0.643 / 0.398 / -0.0495 [-0.2079, 0.0914] |
| turn: gentle 0.02-0.06 | 720 | 98 | 0.9896 | 1.0094 | -0.0199 [-0.1232, 0.0677] |  | 0.160 / -0.068 / -0.0789 [-0.2462, 0.0545] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9557 | 1.9207 | 0.0350 [-0.0851, 0.1512] |  | 0.113 / 0.085 / -0.0129 [-0.2081, 0.1573] |
| turn: sharp >=0.15 | 367 | 42 | 3.3249 | 2.9870 | 0.3379 [0.0225, 0.7386] | **SEP** | 0.081 / 0.593 / 0.3174 [-0.1074, 0.9382] |
| turn: undef (GT crawling) (low n) | 17 | 13 | 11.0545 | 11.1380 | -0.0835 [-0.2200, 0.0355] |  | 0.004 / -0.007 / -0.1304 [-0.3509, 0.0868] |
| speed: v0 <2 | 221 | 37 | 13.3571 | 13.2326 | 0.1246 [-0.0429, 0.3078] |  | 0.049 / 0.132 / 0.0822 [-0.1801, 0.4226] |
| speed: v0 2-5 | 516 | 58 | 1.8093 | 1.8694 | -0.0601 [-0.3546, 0.1978] |  | 0.114 / -0.148 / -0.1203 [-0.5812, 0.2910] |
| speed: v0 5-10 | 1390 | 84 | 1.1784 | 1.0092 | 0.1692 [0.0780, 0.2667] | **SEP** | 0.308 / 1.124 / 0.1775 [0.0250, 0.3232] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.5878 | 0.5744 | 0.0133 [-0.0258, 0.0479] |  | 0.264 / 0.076 / -0.0449 [-0.1492, 0.0684] |
| speed: v0 15-25 | 856 | 35 | 0.3330 | 0.3734 | -0.0404 [-0.0783, -0.0071] | **SEP** | 0.190 / -0.165 / -0.1072 [-0.1937, -0.0138] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1466 | 0.1576 | -0.0110 [-0.0406, 0.0079] |  | 0.074 / -0.018 / -0.0620 [-0.1392, 0.0225] |
| nav: nav=follow | 2880 | 88 | 0.6478 | 0.5982 | 0.0496 [0.0071, 0.0963] | **SEP** | 0.639 / 0.683 / 0.0089 [-0.1520, 0.1794] |
| nav: nav=left (low n) | 403 | 13 | 6.9292 | 6.9119 | 0.0173 [-0.0821, 0.1172] |  | 0.089 / 0.033 / -0.0320 [-0.2038, 0.1507] |
| nav: nav=right | 1227 | 38 | 1.5486 | 1.5002 | 0.0484 [-0.0904, 0.1709] |  | 0.272 / 0.284 / 0.0027 [-0.2033, 0.2134] |

episode concentration: 0.597 of episodes have mean delta > 0; top-5 episodes carry 0.682 of the summed delta; delta excluding the top 5 = 0.0153 (overall 0.0464).

### S2 -- EMA-15k - EMA-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4508 | 139 | 1.4850 | 1.4223 | 0.0628 [0.0313, 0.0970] | **SEP** |  |
| turn: straight <0.02 | 2897 | 132 | 1.2131 | 1.1731 | 0.0399 [0.0053, 0.0840] | **SEP** | 0.643 / 0.409 / -0.0639 [-0.1730, 0.0368] |
| turn: gentle 0.02-0.06 | 720 | 98 | 1.0815 | 1.0251 | 0.0564 [-0.0058, 0.1365] |  | 0.160 / 0.143 / -0.0076 [-0.1125, 0.1177] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9488 | 1.9002 | 0.0486 [-0.0257, 0.1439] |  | 0.113 / 0.087 / -0.0160 [-0.1441, 0.1367] |
| turn: sharp >=0.15 | 366 | 42 | 3.3408 | 3.0633 | 0.2774 [0.0990, 0.5239] | **SEP** | 0.081 / 0.359 / 0.2336 [0.0064, 0.6047] BONF-SEP |
| turn: undef (GT crawling) (low n) | 17 | 14 | 11.1151 | 11.0860 | 0.0291 [-0.0660, 0.1562] |  | 0.004 / 0.002 / -0.0338 [-0.1730, 0.1998] |
| speed: v0 <2 | 219 | 37 | 13.4895 | 13.3568 | 0.1327 [-0.0016, 0.3569] |  | 0.049 / 0.103 / 0.0735 [-0.1128, 0.5120] |
| speed: v0 2-5 | 516 | 58 | 1.9464 | 1.9040 | 0.0424 [-0.1125, 0.2466] |  | 0.114 / 0.077 / -0.0230 [-0.2522, 0.3575] |
| speed: v0 5-10 | 1390 | 84 | 1.2090 | 1.0618 | 0.1471 [0.0735, 0.2339] | **SEP** | 0.308 / 0.723 / 0.1220 [-0.0018, 0.2758] |
| speed: v0 10-15 | 1192 | 64 | 0.6019 | 0.5717 | 0.0302 [0.0002, 0.0660] | **SEP** | 0.264 / 0.127 / -0.0443 [-0.1300, 0.0237] |
| speed: v0 15-25 | 856 | 35 | 0.3348 | 0.3482 | -0.0134 [-0.0277, -0.0012] | **SEP** | 0.190 / -0.041 / -0.0941 [-0.1603, -0.0369] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1538 | 0.1448 | 0.0091 [-0.0031, 0.0236] |  | 0.074 / 0.011 / -0.0580 [-0.1197, -0.0100] BONF-SEP |
| nav: nav=follow | 2877 | 88 | 0.6816 | 0.6057 | 0.0759 [0.0365, 0.1251] | **SEP** | 0.638 / 0.772 / 0.0363 [-0.0603, 0.1395] |
| nav: nav=left (low n) | 402 | 13 | 6.9194 | 6.9119 | 0.0075 [-0.0486, 0.0691] |  | 0.089 / 0.011 / -0.0606 [-0.1662, 0.0583] |
| nav: nav=right | 1229 | 38 | 1.5882 | 1.5381 | 0.0501 [-0.0077, 0.1126] |  | 0.273 / 0.218 / -0.0174 [-0.1273, 0.0880] |

episode concentration: 0.597 of episodes have mean delta > 0; top-5 episodes carry 0.493 of the summed delta; delta excluding the top 5 = 0.0329 (overall 0.0628).

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

### S2 -- EMA-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

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

### S2 -- EMA-15k - EMA-10.5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4473 | 139 | 0.7264 | 0.6807 | 0.0457 [0.0005, 0.1298] | **SEP** | 0.0372 [-0.0101, 0.1227] |
| seg1 | straight | 2878 | 132 | 0.7077 | 0.6391 | 0.0687 [0.0012, 0.2084] | **SEP** | 0.0701 [0.0008, 0.2134] SEP |
| seg1 | curved | 1585 | 101 | 0.6497 | 0.6454 | 0.0043 [-0.0086, 0.0199] |  | -0.0222 [-0.0409, -0.0060] SEP |
| seg2 | all | 4468 | 139 | 1.1186 | 1.0098 | 0.1088 [0.0136, 0.2768] | **SEP** | 0.0568 [-0.0465, 0.2342] |
| seg2 | straight | 2878 | 132 | 0.8908 | 0.7418 | 0.1490 [0.0095, 0.4311] | **SEP** | 0.1315 [-0.0136, 0.4192] |
| seg2 | curved | 1590 | 101 | 1.5309 | 1.4948 | 0.0361 [-0.0039, 0.0845] |  | -0.0784 [-0.1365, -0.0296] SEP |
| seg3 | all | 4461 | 139 | 1.2191 | 1.1348 | 0.0843 [0.0217, 0.1734] | **SEP** | -0.0162 [-0.0931, 0.0873] |
| seg3 | straight | 2875 | 132 | 0.7197 | 0.6188 | 0.1009 [0.0156, 0.2583] | **SEP** | 0.0498 [-0.0409, 0.2006] |
| seg3 | curved | 1586 | 101 | 2.1246 | 2.0703 | 0.0543 [-0.0085, 0.1315] |  | -0.1357 [-0.2380, -0.0507] SEP |
| seg4 | all | 4441 | 139 | 1.6742 | 1.5624 | 0.1118 [0.0027, 0.2867] | **SEP** | 0.0144 [-0.1125, 0.2034] |
| seg4 | straight | 2858 | 131 | 1.0080 | 0.8721 | 0.1359 [-0.0065, 0.4161] |  | 0.1294 [-0.0127, 0.4070] |
| seg4 | curved | 1576 | 101 | 2.8866 | 2.8181 | 0.0684 [-0.0306, 0.1875] |  | -0.1943 [-0.3366, -0.0688] SEP |

### S2 -- EMA-10.5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4474 | 139 | 0.6806 | 0.6282 | 0.0524 [0.0007, 0.1396] | **SEP** | 0.0234 [-0.0301, 0.1128] |
| seg1 | straight | 2879 | 132 | 0.6389 | 0.5697 | 0.0692 [-0.0006, 0.2091] |  | 0.0373 [-0.0353, 0.1818] |
| seg1 | curved | 1586 | 101 | 0.6450 | 0.6229 | 0.0222 [-0.0075, 0.0530] |  | -0.0016 [-0.0495, 0.0422] |
| seg2 | all | 4469 | 139 | 1.0096 | 0.9390 | 0.0707 [0.0002, 0.1658] | **SEP** | -0.1572 [-0.4207, 0.0080] |
| seg2 | straight | 2879 | 132 | 0.7417 | 0.6575 | 0.0842 [-0.0001, 0.2379] |  | -0.2550 [-0.6860, -0.0406] SEP |
| seg2 | curved | 1590 | 101 | 1.4948 | 1.4487 | 0.0461 [-0.0314, 0.1277] |  | 0.0197 [-0.1008, 0.1320] |
| seg3 | all | 4462 | 139 | 1.1352 | 1.2044 | -0.0692 [-0.3275, 0.0890] |  | -0.1455 [-0.4217, 0.0405] |
| seg3 | straight | 2875 | 132 | 0.6189 | 0.7742 | -0.1553 [-0.5963, 0.0691] |  | -0.2679 [-0.7136, -0.0402] SEP |
| seg3 | curved | 1587 | 101 | 2.0705 | 1.9836 | 0.0869 [-0.0395, 0.2205] |  | 0.0762 [-0.0841, 0.2274] |
| seg4 | all | 4445 | 139 | 1.5630 | 1.4925 | 0.0705 [-0.0133, 0.1548] |  | 0.1056 [-0.0403, 0.3086] |
| seg4 | straight | 2860 | 131 | 0.8717 | 0.8399 | 0.0318 [-0.0255, 0.0950] |  | 0.0747 [-0.1019, 0.3737] |
| seg4 | curved | 1577 | 101 | 2.8215 | 2.6798 | 0.1418 [-0.0521, 0.3470] |  | 0.1629 [-0.0671, 0.3930] |

### S2 -- EMA-15k - EMA-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4472 | 139 | 0.7267 | 0.5117 | 0.2149 [0.0068, 0.6252] | **SEP** | 0.1240 [-0.0055, 0.3730] |
| seg1 | straight | 2877 | 132 | 0.7078 | 0.3848 | 0.3230 [0.0036, 1.0173] | **SEP** | 0.2022 [0.0045, 0.6266] SEP |
| seg1 | curved | 1585 | 101 | 0.6497 | 0.6297 | 0.0200 [0.0059, 0.0348] | **SEP** | -0.0165 [-0.0380, 0.0041] |
| seg2 | all | 4469 | 139 | 1.1183 | 0.8590 | 0.2592 [0.0353, 0.6727] | **SEP** | 0.1081 [-0.0416, 0.3683] |
| seg2 | straight | 2880 | 132 | 0.8903 | 0.5355 | 0.3548 [0.0197, 1.0490] | **SEP** | 0.1966 [-0.0229, 0.6262] |
| seg2 | curved | 1589 | 101 | 1.5315 | 1.4454 | 0.0861 [0.0412, 0.1382] | **SEP** | -0.0521 [-0.1196, 0.0125] |
| seg3 | all | 4461 | 139 | 1.2191 | 1.1788 | 0.0404 [-0.0585, 0.1204] |  | -0.0721 [-0.1747, 0.0129] |
| seg3 | straight | 2875 | 132 | 0.7197 | 0.7218 | -0.0021 [-0.1593, 0.1086] |  | -0.0839 [-0.2562, 0.0171] |
| seg3 | curved | 1586 | 101 | 2.1246 | 2.0072 | 0.1174 [0.0484, 0.2022] | **SEP** | -0.0508 [-0.1645, 0.0579] |
| seg4 | all | 4440 | 139 | 1.6770 | 1.6399 | 0.0371 [-0.0659, 0.1191] |  | 0.1532 [-0.0225, 0.4253] |
| seg4 | straight | 2857 | 131 | 1.0121 | 1.0499 | -0.0378 [-0.1934, 0.0634] |  | 0.2268 [0.0028, 0.6548] SEP |
| seg4 | curved | 1576 | 101 | 2.8866 | 2.7137 | 0.1729 [0.0611, 0.3031] | **SEP** | 0.0203 [-0.1517, 0.1814] |

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

### S2 -- EMA-15k - refcv7-50400: turn x speed cross

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

| bin | EMA-15k s0-s1 [CI] | EMA-10.5k s0-s1 [CI] | EMA-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|---|
| overall | -0.0005 [-0.0127, 0.0123] | 0.0031 [-0.0109, 0.0180] | 0.0080 [-0.0053, 0.0227] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | -0.0018 [-0.0136, 0.0091] | 0.0028 [-0.0175, 0.0223] | -0.0025 [-0.0152, 0.0107] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0037 [-0.0392, 0.0479] | 0.0039 [-0.0207, 0.0359] | 0.0222 [-0.0046, 0.0560] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | 0.0098 [-0.0370, 0.0556] | -0.0099 [-0.0410, 0.0178] | 0.0029 [-0.0637, 0.0729] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | -0.0106 [-0.0602, 0.0395] | 0.0286 [-0.0320, 0.0952] | 0.0710 [-0.0077, 0.1690] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.0306 [-0.1758, 0.0775] | -0.1140 [-0.2743, 0.0250] | -0.0061 [-0.1265, 0.1043] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | -0.0244 [-0.0824, 0.0341] | -0.0346 [-0.0718, 0.0181] | 0.0507 [-0.0497, 0.1881] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | -0.0220 [-0.0847, 0.0342] | 0.0219 [-0.0389, 0.1008] | 0.0044 [-0.0592, 0.0734] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | 0.0054 [-0.0246, 0.0319] | 0.0111 [-0.0255, 0.0414] | 0.0164 [-0.0160, 0.0478] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | 0.0088 [-0.0029, 0.0208] | -0.0039 [-0.0181, 0.0088] | -0.0014 [-0.0124, 0.0091] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | -0.0042 [-0.0124, 0.0037] | 0.0012 [-0.0064, 0.0076] | 0.0006 [-0.0037, 0.0050] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | 0.0008 [-0.0049, 0.0064] | -0.0036 [-0.0103, 0.0025] | 0.0018 [-0.0030, 0.0066] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | 0.0022 [-0.0101, 0.0155] | -0.0035 [-0.0139, 0.0086] | 0.0030 [-0.0075, 0.0161] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | 0.0120 [-0.0189, 0.0388] | -0.0000 [-0.0180, 0.0186] | 0.0540 [-0.0031, 0.1154] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | -0.0109 [-0.0458, 0.0198] | 0.0197 [-0.0260, 0.0663] | 0.0044 [-0.0318, 0.0433] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- EMA-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

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

### S2 -- EMA-15k - EMA-10.5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0313 | 0.0297 / 0.0330 | True / True / True | 0.0005 / 0.0031 | 10.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0244 | 0.0221 / 0.0267 | True / True / True | 0.0018 / 0.0028 | 8.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0919 | 0.0918 / 0.0920 | True / True / True | 0.0037 / 0.0039 | 23.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|turn 0.06-0.15 | -0.0069 | 0.0030 / -0.0167 | False / False / False | 0.0098 / 0.0099 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.0213 | 0.0035 / 0.0391 | False / False / False | 0.0106 / 0.0286 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.0075 | 0.0463 / -0.0312 | False / False / False | 0.0306 / 0.1140 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0332 | 0.0409 / 0.0254 | False / False / False | 0.0244 / 0.0346 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1371 | 0.1152 / 0.1590 | False / False / False | 0.0220 / 0.0219 | 6.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0306 | 0.0277 / 0.0335 | False / False / False | 0.0054 / 0.0111 | 2.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0141 | 0.0205 / 0.0078 | False / False / False | 0.0088 / 0.0039 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0018 | -0.0009 / 0.0045 | False / False / False | 0.0042 / 0.0012 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0072 | 0.0094 / 0.0050 | False / False / False | 0.0008 / 0.0036 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0335 | 0.0363 / 0.0306 | True / True / True | 0.0022 / 0.0035 | 9.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.0138 | -0.0061 / -0.0214 | False / False / False | 0.0120 / 0.0000 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0411 | 0.0258 / 0.0565 | False / False / False | 0.0109 / 0.0197 | 2.1 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- EMA-10.5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0464 | 0.0428 / 0.0500 | False / True / True | 0.0031 / 0.0110 | 4.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|straight <0.02 | 0.0287 | 0.0292 / 0.0283 | False / False / False | 0.0028 / 0.0019 | 10.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.0199 | -0.0101 / -0.0296 | False / False / False | 0.0039 / 0.0161 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0350 | 0.0154 / 0.0545 | False / False / False | 0.0099 / 0.0331 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.3379 | 0.2982 / 0.3777 | False / True / True | 0.0286 / 0.1080 | 3.1 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | -0.0835 | -0.0960 / -0.0711 | False / False / False | 0.1140 / 0.0237 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1246 | 0.0818 / 0.1674 | False / False / False | 0.0346 / 0.0573 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.0601 | -0.0852 / -0.0351 | False / False / False | 0.0219 / 0.0719 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1692 | 0.1744 / 0.1640 | True / True / True | 0.0111 / 0.0006 | 15.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0133 | 0.0114 / 0.0152 | False / False / False | 0.0039 / 0.0001 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0404 | -0.0387 / -0.0422 | True / True / True | 0.0012 / 0.0023 | 17.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0110 | -0.0125 / -0.0094 | False / False / False | 0.0036 / 0.0005 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0496 | 0.0497 / 0.0495 | True / True / True | 0.0035 / 0.0036 | 13.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0173 | -0.0111 / 0.0457 | False / False / False | 0.0000 / 0.0608 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0484 | 0.0441 / 0.0527 | False / False / False | 0.0197 / 0.0283 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- EMA-15k - EMA-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0628 | 0.0585 / 0.0671 | True / True / True | 0.0005 / 0.0080 | 7.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|straight <0.02 | 0.0399 | 0.0401 / 0.0398 | True / True / True | 0.0018 / 0.0025 | 16.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|gentle 0.02-0.06 | 0.0564 | 0.0469 / 0.0658 | False / True / False | 0.0037 / 0.0222 | 2.5 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0486 | 0.0520 / 0.0452 | False / False / False | 0.0098 / 0.0029 | 5.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.2774 | 0.2373 / 0.3176 | True / True / True | 0.0106 / 0.0710 | 3.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.0291 | 0.0253 / 0.0329 | False / False / False | 0.0306 / 0.0061 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1327 | 0.0925 / 0.1720 | False / True / False | 0.0244 / 0.0507 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0424 | 0.0292 / 0.0556 | False / False / False | 0.0220 / 0.0044 | 1.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1471 | 0.1416 / 0.1526 | True / True / True | 0.0054 / 0.0164 | 9.0 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0302 | 0.0353 / 0.0251 | True / False / True | 0.0088 / 0.0014 | 3.4 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 15-25 | -0.0134 | -0.0158 / -0.0110 | True / False / True | 0.0042 / 0.0006 | 3.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 >=25 | 0.0091 | 0.0085 / 0.0096 | False / False / False | 0.0008 / 0.0018 | 5.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0759 | 0.0755 / 0.0763 | True / True / True | 0.0022 / 0.0030 | 24.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0075 | -0.0132 / 0.0282 | False / False / False | 0.0120 / 0.0540 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0501 | 0.0420 / 0.0582 | False / False / False | 0.0109 / 0.0044 | 4.6 | NOT PROVEN (not separated and/or within 2x floor) |

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

* seg1 (n 1585): EMA-15k bias -0.027, refcv7-50400 bias +0.082, EMA-10.5k bias -0.049, EMA-5k bias -0.009; EMA-15k MAE 0.650, refcv7-50400 MAE 0.623, EMA-10.5k MAE 0.645, EMA-5k MAE 0.630
* seg2 (n 1589): EMA-15k bias -0.349, refcv7-50400 bias -0.097, EMA-10.5k bias -0.410, EMA-5k bias -0.283; EMA-15k MAE 1.532, refcv7-50400 MAE 1.448, EMA-10.5k MAE 1.495, EMA-5k MAE 1.445
* seg3 (n 1586): EMA-15k bias -0.910, refcv7-50400 bias -0.617, EMA-10.5k bias -1.008, EMA-5k bias -0.798; EMA-15k MAE 2.125, refcv7-50400 MAE 1.984, EMA-10.5k MAE 2.070, EMA-5k MAE 2.007
* seg4 (n 1576): EMA-15k bias -1.189, refcv7-50400 bias -0.713, EMA-10.5k bias -1.356, EMA-5k bias -1.016; EMA-15k MAE 2.887, refcv7-50400 MAE 2.680, EMA-10.5k MAE 2.818, EMA-5k MAE 2.714
**sharp (>= 0.15)**

* seg1 (n 364): EMA-15k bias -0.298, refcv7-50400 bias +0.010, EMA-10.5k bias -0.357, EMA-5k bias -0.267; EMA-15k MAE 0.933, refcv7-50400 MAE 0.917, EMA-10.5k MAE 0.920, EMA-5k MAE 0.881
* seg2 (n 366): EMA-15k bias -1.204, refcv7-50400 bias -0.528, EMA-10.5k bias -1.337, EMA-5k bias -1.088; EMA-15k MAE 2.593, refcv7-50400 MAE 2.413, EMA-10.5k MAE 2.543, EMA-5k MAE 2.367
* seg3 (n 365): EMA-15k bias -2.587, refcv7-50400 bias -1.863, EMA-10.5k bias -2.756, EMA-5k bias -2.394; EMA-15k MAE 4.103, refcv7-50400 MAE 3.649, EMA-10.5k MAE 4.100, EMA-5k MAE 3.794
* seg4 (n 363): EMA-15k bias -3.040, refcv7-50400 bias -1.837, EMA-10.5k bias -3.347, EMA-5k bias -2.733; EMA-15k MAE 5.637, refcv7-50400 MAE 4.887, EMA-10.5k MAE 5.556, EMA-5k MAE 5.111
**curved & v0 5-10**

* seg1 (n 627): EMA-15k bias -0.164, refcv7-50400 bias -0.038, EMA-10.5k bias -0.184, EMA-5k bias -0.134; EMA-15k MAE 0.590, refcv7-50400 MAE 0.535, EMA-10.5k MAE 0.585, EMA-5k MAE 0.553
* seg2 (n 627): EMA-15k bias -0.612, refcv7-50400 bias -0.282, EMA-10.5k bias -0.641, EMA-5k bias -0.477; EMA-15k MAE 1.559, refcv7-50400 MAE 1.363, EMA-10.5k MAE 1.512, EMA-5k MAE 1.377
* seg3 (n 627): EMA-15k bias -1.301, refcv7-50400 bias -0.871, EMA-10.5k bias -1.317, EMA-5k bias -1.035; EMA-15k MAE 2.334, refcv7-50400 MAE 2.019, EMA-10.5k MAE 2.239, EMA-5k MAE 2.036
* seg4 (n 627): EMA-15k bias -1.811, refcv7-50400 bias -1.194, EMA-10.5k bias -1.867, EMA-5k bias -1.432; EMA-15k MAE 3.150, refcv7-50400 MAE 2.730, EMA-10.5k MAE 3.061, EMA-5k MAE 2.759
**sharp & v0 5-10**

* seg1 (n 164): EMA-15k bias -0.168, refcv7-50400 bias +0.103, EMA-10.5k bias -0.249, EMA-5k bias -0.151; EMA-15k MAE 0.858, refcv7-50400 MAE 0.857, EMA-10.5k MAE 0.824, EMA-5k MAE 0.787
* seg2 (n 164): EMA-15k bias -1.062, refcv7-50400 bias -0.422, EMA-10.5k bias -1.209, EMA-5k bias -0.929; EMA-15k MAE 2.495, refcv7-50400 MAE 2.243, EMA-10.5k MAE 2.344, EMA-5k MAE 2.117
* seg3 (n 164): EMA-15k bias -2.805, refcv7-50400 bias -2.032, EMA-10.5k bias -2.919, EMA-5k bias -2.456; EMA-15k MAE 4.150, refcv7-50400 MAE 3.465, EMA-10.5k MAE 3.897, EMA-5k MAE 3.412
* seg4 (n 164): EMA-15k bias -3.654, refcv7-50400 bias -2.403, EMA-10.5k bias -3.856, EMA-5k bias -3.048; EMA-15k MAE 5.851, refcv7-50400 MAE 4.635, EMA-10.5k MAE 5.463, EMA-5k MAE 4.706

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1576 / 1576 | 10.2 | EMA-15k -0.1326 [-0.2195, -0.0647]; refcv7-50400 -0.0843 [-0.1557, -0.0272]; EMA-10.5k -0.1454 [-0.2321, -0.0784]; EMA-5k -0.1193 [-0.2052, -0.0539] | EMA-15k -0.0483 [-0.0799, -0.0224] SEP; EMA-10.5k -0.0610 [-0.0908, -0.0367] SEP; EMA-5k -0.0350 [-0.0581, -0.0171] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1592 / 4751 | 7.2 | EMA-15k -0.1919 [-0.3155, -0.0892]; refcv7-50400 -0.1465 [-0.2654, -0.0521]; EMA-10.5k -0.2036 [-0.3257, -0.1048]; EMA-5k -0.1803 [-0.3036, -0.0809] | EMA-15k -0.0454 [-0.0737, -0.0221] SEP; EMA-10.5k -0.0571 [-0.0841, -0.0350] SEP; EMA-5k -0.0338 [-0.0541, -0.0184] SEP |
| sharp (>= 0.15) | last segment | 363 / 363 | 25.4 | EMA-15k -0.1409 [-0.2445, -0.0616]; refcv7-50400 -0.0921 [-0.1765, -0.0255]; EMA-10.5k -0.1545 [-0.2571, -0.0754]; EMA-5k -0.1302 [-0.2321, -0.0533] | EMA-15k -0.0488 [-0.0867, -0.0176] SEP; EMA-10.5k -0.0623 [-0.0979, -0.0337] SEP; EMA-5k -0.0381 [-0.0647, -0.0174] SEP |
| sharp (>= 0.15) | segments 2..K | 366 / 1094 | 17.9 | EMA-15k -0.1523 [-0.2562, -0.0713]; refcv7-50400 -0.1041 [-0.1873, -0.0360]; EMA-10.5k -0.1652 [-0.2676, -0.0867]; EMA-5k -0.1423 [-0.2435, -0.0657] | EMA-15k -0.0482 [-0.0838, -0.0187] SEP; EMA-10.5k -0.0612 [-0.0937, -0.0338] SEP; EMA-5k -0.0382 [-0.0626, -0.0194] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | EMA-15k -0.1397 [-0.2276, -0.0723]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; EMA-10.5k -0.1479 [-0.2329, -0.0826]; EMA-5k -0.1128 [-0.1915, -0.0594] | EMA-15k -0.0472 [-0.0991, -0.0027] SEP; EMA-10.5k -0.0553 [-0.1025, -0.0150] SEP; EMA-5k -0.0202 [-0.0429, +0.0055] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | EMA-15k -0.1470 [-0.2318, -0.0818]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; EMA-10.5k -0.1560 [-0.2400, -0.0922]; EMA-5k -0.1233 [-0.2007, -0.0703] | EMA-15k -0.0478 [-0.0961, -0.0057] SEP; EMA-10.5k -0.0568 [-0.1007, -0.0192] SEP; EMA-5k -0.0240 [-0.0453, +0.0004] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | EMA-15k -0.1337 [-0.2294, -0.0595]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; EMA-10.5k -0.1439 [-0.2364, -0.0718]; EMA-5k -0.1116 [-0.1998, -0.0515] | EMA-15k -0.0424 [-0.1008, +0.0060]; EMA-10.5k -0.0527 [-0.1039, -0.0100] SEP; EMA-5k -0.0203 [-0.0436, +0.0069] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | EMA-15k -0.1369 [-0.2322, -0.0631]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; EMA-10.5k -0.1483 [-0.2400, -0.0764]; EMA-5k -0.1186 [-0.2069, -0.0577] | EMA-15k -0.0414 [-0.0964, +0.0052]; EMA-10.5k -0.0528 [-0.1023, -0.0107] SEP; EMA-5k -0.0231 [-0.0459, +0.0041] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA-15k - refcv7-50400**

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
**EMA-15k - EMA-10.5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4509 | 0.0313 | 0.00000 | -0.00029 | 0.500 / 0.500 / 0.000 | 208.1 / 141.3 |
| turn|straight <0.02 | 2898 | 0.0244 | 0.00111 | 0.00294 | 0.514 / 0.486 / 0.000 | 85.2 / 70.8 |
| turn|gentle 0.02-0.06 | 720 | 0.0919 | -0.00272 | -0.00297 | 0.485 / 0.515 / 0.000 | 63.9 / 66.1 |
| turn|turn 0.06-0.15 | 508 | -0.0069 | -0.01818 | -0.02234 | 0.443 / 0.557 / 0.000 | 11.5 / -3.5 |
| turn|sharp >=0.15 | 366 | 0.0213 | -0.00106 | -0.02049 | 0.500 / 0.500 / 0.000 | 26.4 / 7.8 |
| turn|undef (GT crawling) | 17 | 0.0075 | -0.00222 | 0.00042 | 0.471 / 0.529 / 0.000 | 0.3 / 0.1 |
| speed|v0 <2 | 220 | 0.0332 | -0.00152 | 0.00197 | 0.491 / 0.509 / 0.000 | 17.8 / 7.3 |
| speed|v0 2-5 | 516 | 0.1371 | -0.00731 | 0.00479 | 0.473 / 0.527 / 0.000 | 60.5 / 70.8 |
| speed|v0 5-10 | 1390 | 0.0306 | -0.00254 | -0.00697 | 0.487 / 0.513 / 0.000 | 70.8 / 42.5 |
| speed|v0 10-15 | 1192 | 0.0141 | 0.00214 | 0.00303 | 0.517 / 0.483 / 0.000 | 23.4 / 16.8 |
| speed|v0 15-25 | 856 | 0.0018 | -0.00033 | 0.00068 | 0.496 / 0.504 / 0.000 | 3.6 / 1.5 |
| speed|v0 >=25 | 335 | 0.0072 | 0.00219 | 0.00369 | 0.552 / 0.448 / 0.000 | 1.3 / 2.4 |
| nav|nav=follow | 2880 | 0.0335 | 0.00006 | 0.00097 | 0.501 / 0.499 / 0.000 | 120.2 / 96.4 |
| nav|nav=left | 402 | -0.0138 | -0.00392 | -0.01216 | 0.478 / 0.522 / 0.000 | 6.8 / -5.5 |
| nav|nav=right | 1227 | 0.0411 | 0.00111 | -0.00063 | 0.504 / 0.496 / 0.000 | 73.6 / 50.5 |
**EMA-10.5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4510 | 0.0464 | 0.00375 | 0.00724 | 0.512 / 0.488 / 0.000 | 304.1 / 209.3 |
| turn|straight <0.02 | 2898 | 0.0287 | 0.00124 | -0.00057 | 0.507 / 0.493 / 0.000 | 140.5 / 83.2 |
| turn|gentle 0.02-0.06 | 720 | -0.0199 | 0.00615 | 0.01147 | 0.511 / 0.489 / 0.000 | 30.0 / -14.3 |
| turn|turn 0.06-0.15 | 508 | 0.0350 | 0.01513 | 0.02973 | 0.514 / 0.486 / 0.000 | 30.4 / 17.8 |
| turn|sharp >=0.15 | 367 | 0.3379 | 0.11961 | 0.15752 | 0.561 / 0.439 / 0.000 | 54.2 / 124.0 |
| turn|undef (GT crawling) | 17 | -0.0835 | -0.03532 | -0.03805 | 0.412 / 0.588 / 0.000 | 0.3 / -1.4 |
| speed|v0 <2 | 221 | 0.1246 | -0.00662 | 0.02402 | 0.493 / 0.507 / 0.000 | 14.9 / 27.5 |
| speed|v0 2-5 | 516 | -0.0601 | 0.01604 | 0.02236 | 0.525 / 0.475 / 0.000 | 57.5 / -31.0 |
| speed|v0 5-10 | 1390 | 0.1692 | 0.02043 | 0.04247 | 0.544 / 0.456 / 0.000 | 115.1 / 235.2 |
| speed|v0 10-15 | 1192 | 0.0133 | 0.00539 | 0.00412 | 0.519 / 0.481 / 0.000 | 25.3 / 15.9 |
| speed|v0 15-25 | 856 | -0.0404 | -0.00795 | -0.01784 | 0.452 / 0.548 / 0.000 | 7.0 / -34.6 |
| speed|v0 >=25 | 335 | -0.0110 | 0.00040 | -0.00042 | 0.501 / 0.499 / 0.000 | 0.7 / -3.7 |
| nav|nav=follow | 2880 | 0.0496 | -0.00076 | -0.00291 | 0.495 / 0.505 / 0.000 | 171.4 / 143.0 |
| nav|nav=left | 403 | 0.0173 | 0.02063 | 0.02719 | 0.563 / 0.437 / 0.000 | 18.7 / 7.0 |
| nav|nav=right | 1227 | 0.0484 | 0.01116 | 0.03215 | 0.535 / 0.465 / 0.000 | 104.0 / 59.4 |
**EMA-15k - EMA-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4508 | 0.0628 | 0.00210 | 0.00915 | 0.516 / 0.484 / 0.000 | 284.4 / 283.0 |
| turn|straight <0.02 | 2897 | 0.0399 | 0.00198 | 0.00526 | 0.520 / 0.480 / 0.000 | 141.2 / 115.7 |
| turn|gentle 0.02-0.06 | 720 | 0.0564 | -0.00047 | 0.00574 | 0.497 / 0.503 / 0.000 | 44.2 / 40.6 |
| turn|turn 0.06-0.15 | 508 | 0.0486 | -0.00505 | 0.01312 | 0.488 / 0.512 / 0.000 | 24.8 / 24.7 |
| turn|sharp >=0.15 | 366 | 0.2774 | 0.06801 | 0.12104 | 0.571 / 0.429 / 0.000 | 43.6 / 101.5 |
| turn|undef (GT crawling) | 17 | 0.0291 | -0.02330 | -0.01793 | 0.235 / 0.765 / 0.000 | 0.6 / 0.5 |
| speed|v0 <2 | 219 | 0.1327 | -0.00440 | 0.01952 | 0.479 / 0.521 / 0.000 | 20.6 / 29.1 |
| speed|v0 2-5 | 516 | 0.0424 | 0.00525 | 0.02220 | 0.519 / 0.481 / 0.000 | 49.2 / 21.9 |
| speed|v0 5-10 | 1390 | 0.1471 | 0.01048 | 0.02674 | 0.539 / 0.461 / 0.000 | 122.2 / 204.5 |
| speed|v0 10-15 | 1192 | 0.0302 | 0.00153 | 0.00700 | 0.513 / 0.487 / 0.000 | 35.3 / 36.0 |
| speed|v0 15-25 | 856 | -0.0134 | -0.00195 | -0.00381 | 0.481 / 0.519 / 0.000 | 4.7 / -11.5 |
| speed|v0 >=25 | 335 | 0.0091 | 0.00162 | 0.00464 | 0.534 / 0.466 / 0.000 | 1.4 / 3.0 |
| nav|nav=follow | 2877 | 0.0759 | 0.00148 | 0.00573 | 0.514 / 0.486 / 0.000 | 198.3 / 218.4 |
| nav|nav=left | 402 | 0.0075 | 0.00301 | 0.00327 | 0.515 / 0.485 / 0.000 | 12.5 / 3.0 |
| nav|nav=right | 1229 | 0.0501 | 0.00703 | 0.02363 | 0.520 / 0.480 / 0.000 | 69.1 / 61.6 |
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

**EMA-15k - refcv7-50400** (sum 347.1 over 139 episodes)

* ep024: n 35, share 0.113, mean delta 1.1239, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep013: n 34, share 0.102, mean delta 1.0372, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep108: n 24, share 0.100, mean delta 1.4422, sharp-window frac 0.00, curved frac 0.54, mean v0 8.9, nav {'nav=follow': 24, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.095, mean delta 0.9709, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep111: n 34, share 0.088, mean delta 0.9004, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep075: n 34, share 0.063, mean delta 0.6431, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
**EMA-15k - EMA-10.5k** (sum 141.3 over 139 episodes)

* ep078: n 34, share 0.261, mean delta 1.0839, sharp-window frac 0.00, curved frac 0.29, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep024: n 35, share 0.192, mean delta 0.7745, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep129: n 32, share 0.153, mean delta 0.6776, sharp-window frac 0.31, curved frac 0.50, mean v0 4.6, nav {'nav=follow': 32, 'nav=left': 0, 'nav=right': 0}
* ep108: n 24, share 0.139, mean delta 0.8204, sharp-window frac 0.00, curved frac 0.54, mean v0 8.9, nav {'nav=follow': 24, 'nav=left': 0, 'nav=right': 0}
* ep059: n 34, share 0.110, mean delta 0.4578, sharp-window frac 0.00, curved frac 0.24, mean v0 6.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep072: n 17, share 0.105, mean delta 0.8750, sharp-window frac 0.00, curved frac 0.82, mean v0 6.5, nav {'nav=follow': 17, 'nav=left': 0, 'nav=right': 0}
**EMA-10.5k - refcv7-50400** (sum 209.3 over 139 episodes)

* ep013: n 34, share 0.175, mean delta 1.0755, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep111: n 34, share 0.154, mean delta 0.9475, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.136, mean delta 0.8349, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep023: n 34, share 0.109, mean delta 0.6723, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep075: n 34, share 0.109, mean delta 0.6682, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep121: n 35, share 0.089, mean delta 0.5295, sharp-window frac 0.20, curved frac 0.74, mean v0 6.1, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
**EMA-15k - EMA-5k** (sum 283.0 over 139 episodes)

* ep024: n 35, share 0.142, mean delta 1.1481, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
* ep108: n 23, share 0.120, mean delta 1.4724, sharp-window frac 0.00, curved frac 0.57, mean v0 9.3, nav {'nav=follow': 23, 'nav=left': 0, 'nav=right': 0}
* ep111: n 34, share 0.094, mean delta 0.7832, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.079, mean delta 0.6609, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep072: n 17, share 0.058, mean delta 0.9616, sharp-window frac 0.00, curved frac 0.82, mean v0 6.5, nav {'nav=follow': 17, 'nav=left': 0, 'nav=right': 0}
* ep066: n 27, share 0.054, mean delta 0.5659, sharp-window frac 0.00, curved frac 0.00, mean v0 3.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**EMA-5k - refcv7-50400** (sum 65.6 over 139 episodes)

* ep013: n 34, share 0.527, mean delta 1.0183, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep075: n 34, share 0.342, mean delta 0.6610, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep067: n 34, share 0.165, mean delta 0.3191, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep021: n 34, share 0.162, mean delta 0.3134, sharp-window frac 0.15, curved frac 0.53, mean v0 5.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep012: n 34, share 0.161, mean delta 0.3100, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep023: n 34, share 0.131, mean delta 0.2524, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA-15k vs EMA-10.5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1722 / -0.0101 [-0.0167, -0.0041] | 91 / -0.0798 [-0.1837, 0.0053] |
| only new agrees | 199 / -0.0351 [-0.0949, 0.0210] | 12 / -0.1197 [-0.9057, 0.7699] |
| only earlier agrees | 290 / 0.1078 [0.0073, 0.2460] | 10 / -0.0616 [-0.9057, 0.7552] |
| neither agrees | 2298 / 0.0553 [0.0176, 0.0972] | 253 / 0.0418 [-0.1318, 0.2546] |
**EMA-15k vs EMA-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1452 / 0.0013 [-0.0116, 0.0154] | 73 / 0.0929 [-0.0936, 0.2588] |
| only new agrees | 469 / -0.0865 [-0.1424, -0.0392] | 30 / -0.2375 [-0.5475, 0.0366] |
| only earlier agrees | 434 / 0.1922 [0.0646, 0.3728] | 25 / 1.1857 [-0.1264, 3.6251] |
| neither agrees | 2154 / 0.1016 [0.0474, 0.1603] | 238 / 0.2417 [0.0764, 0.4284] |
**EMA-15k vs EMA-10.5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1704 / -0.0052 [-0.0119, 0.0012] | 94 / -0.0534 [-0.1606, 0.0253] |
| only new agrees | 209 / -0.0130 [-0.0995, 0.0532] | 16 / -0.3524 [-1.0042, 0.3557] |
| only earlier agrees | 296 / 0.1239 [0.0085, 0.3049] | 9 / 0.0889 [-0.9041, 0.9227] |
| neither agrees | 2300 / 0.0538 [0.0059, 0.1170] | 247 / 0.0979 [-0.0971, 0.3498] |
**EMA-15k vs EMA-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1455 / 0.0025 [-0.0093, 0.0155] | 80 / 0.0677 [-0.0844, 0.1960] |
| only new agrees | 458 / -0.0677 [-0.1274, -0.0169] | 30 / -0.0909 [-0.3408, 0.2214] |
| only earlier agrees | 416 / 0.2267 [0.0683, 0.4426] | 21 / 1.6809 [0.0178, 4.0136] |
| neither agrees | 2180 / 0.1080 [0.0550, 0.1637] | 235 / 0.3331 [0.1308, 0.5734] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA-15k vs EMA-10.5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4479 | 0.0497 [-0.0139, 0.1126] | 0.0414 [-0.0150, 0.0973] | 0.0085 [-0.0205, 0.0376] |
| turn|straight <0.02 | 2885 | 0.0337 [-0.0102, 0.0835] | 0.0330 [-0.0111, 0.0910] | 0.0006 [-0.0326, 0.0324] |
| turn|gentle 0.02-0.06 | 718 | -0.0030 [-0.1151, 0.0918] | 0.0278 [-0.0292, 0.0880] | -0.0304 [-0.1270, 0.0457] |
| turn|turn 0.06-0.15 | 501 | 0.1073 [-0.0487, 0.2777] | 0.0665 [-0.0712, 0.2335] | 0.0424 [-0.0378, 0.1525] |
| turn|sharp >=0.15 | 365 | 0.2025 [-0.3538, 0.6720] | 0.1006 [-0.4657, 0.5041] | 0.1019 [-0.0691, 0.3019] |
| turn|undef (GT crawling) | 10 | -0.0348 [-0.3072, 0.1884] | 0.0236 [-0.1437, 0.1779] | -0.0239 [-0.1083, 0.0297] |
| speed|v0 <2 | 190 | -0.0442 [-0.2750, 0.1016] | -0.1405 [-0.3612, 0.0558] | 0.1050 [-0.0720, 0.2968] |
| speed|v0 2-5 | 516 | 0.1507 [-0.2666, 0.5068] | 0.1001 [-0.2674, 0.4425] | 0.0506 [-0.1567, 0.2472] |
| speed|v0 5-10 | 1390 | 0.1186 [0.0172, 0.2236] | 0.1209 [0.0333, 0.2160] | -0.0022 [-0.0590, 0.0478] |
| speed|v0 10-15 | 1192 | -0.0102 [-0.0358, 0.0150] | -0.0034 [-0.0302, 0.0272] | -0.0068 [-0.0307, 0.0115] |
| speed|v0 15-25 | 856 | 0.0030 [-0.0073, 0.0130] | -0.0054 [-0.0197, 0.0059] | 0.0084 [-0.0016, 0.0199] |
| speed|v0 >=25 | 335 | -0.0057 [-0.0188, 0.0052] | 0.0057 [-0.0033, 0.0194] | -0.0115 [-0.0300, 0.0010] |
**EMA-15k vs EMA-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4481 | 0.0497 [-0.0139, 0.1126] | 0.0377 [-0.0311, 0.1106] | 0.0131 [-0.0423, 0.0609] |
| turn|straight <0.02 | 2886 | 0.0337 [-0.0102, 0.0835] | 0.0066 [-0.0273, 0.0472] | 0.0270 [-0.0144, 0.0711] |
| turn|gentle 0.02-0.06 | 719 | -0.0030 [-0.1151, 0.0918] | -0.0222 [-0.1579, 0.1103] | 0.0191 [-0.0691, 0.1073] |
| turn|turn 0.06-0.15 | 501 | 0.1073 [-0.0487, 0.2777] | 0.1061 [-0.0326, 0.3074] | 0.0027 [-0.1580, 0.1543] |
| turn|sharp >=0.15 | 365 | 0.2025 [-0.3538, 0.6720] | 0.3019 [-0.2941, 0.7995] | -0.0877 [-0.3819, 0.1503] |
| turn|undef (GT crawling) | 10 | -0.0348 [-0.3072, 0.1884] | 0.1364 [-0.0364, 0.3356] | -0.2495 [-0.5511, -0.0501] SEP |
| speed|v0 <2 | 192 | -0.0442 [-0.2750, 0.1016] | 0.2537 [0.0988, 0.4898] | -0.2992 [-0.6221, -0.0847] SEP |
| speed|v0 2-5 | 516 | 0.1507 [-0.2666, 0.5068] | 0.0959 [-0.3776, 0.5625] | 0.0548 [-0.3370, 0.3701] |
| speed|v0 5-10 | 1390 | 0.1186 [0.0172, 0.2236] | 0.0594 [-0.0384, 0.1665] | 0.0592 [-0.0072, 0.1217] |
| speed|v0 10-15 | 1192 | -0.0102 [-0.0358, 0.0150] | -0.0114 [-0.0289, 0.0049] | 0.0011 [-0.0281, 0.0323] |
| speed|v0 15-25 | 856 | 0.0030 [-0.0073, 0.0130] | -0.0039 [-0.0198, 0.0083] | 0.0069 [-0.0066, 0.0217] |
| speed|v0 >=25 | 335 | -0.0057 [-0.0188, 0.0052] | -0.0007 [-0.0094, 0.0106] | -0.0050 [-0.0188, 0.0042] |
**EMA-15k vs EMA-10.5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
**EMA-15k vs EMA-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* EMA-15k vs EMA-10.5k|seed0: anchor flip rate 0.204; heading delta on flipped windows 0.1509 [0.0495, 0.2648] (n 952) vs unflipped -0.0028 [-0.0095, 0.0038]; goal_dist_sel proxy new-earlier -0.0528 [-0.1011, -0.0051]; corr(window dHeading, window dGoalDist) -0.069
* EMA-15k vs EMA-5k|seed0: anchor flip rate 0.377; heading delta on flipped windows 0.1430 [0.0626, 0.2361] (n 1748) vs unflipped 0.0050 [-0.0067, 0.0174]; goal_dist_sel proxy new-earlier -0.0024 [-0.0741, 0.0710]; corr(window dHeading, window dGoalDist) -0.079
* EMA-15k vs EMA-10.5k|seed1: anchor flip rate 0.204; heading delta on flipped windows 0.1726 [0.0490, 0.3276] (n 946) vs unflipped -0.0040 [-0.0111, 0.0033]; goal_dist_sel proxy new-earlier -0.0544 [-0.1004, -0.0086]; corr(window dHeading, window dGoalDist) -0.030
* EMA-15k vs EMA-5k|seed1: anchor flip rate 0.372; heading delta on flipped windows 0.1733 [0.0930, 0.2615] (n 1728) vs unflipped 0.0010 [-0.0100, 0.0123]; goal_dist_sel proxy new-earlier -0.0004 [-0.0729, 0.0725]; corr(window dHeading, window dGoalDist) -0.064

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- EMA-15k - refcv7-50400, 2-seed mean of the per-window metric

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

### S6 -- EMA-15k - EMA-10.5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3524 | 139 | 3.4212 | 3.3842 | 0.0370 [-0.0476, 0.1484] |  |  |
| turn: straight <0.02 | 2098 | 122 | 1.6029 | 1.5342 | 0.0687 [0.0040, 0.1589] | **SEP** | 0.595 / 1.106 / 0.0783 [-0.3534, 0.3560] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.8791 | 2.7017 | 0.1774 [-0.0378, 0.5616] |  | 0.192 / 0.919 / 0.1737 [-0.1318, 0.9024] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.4533 | 6.4718 | -0.0186 [-0.1307, 0.0955] |  | 0.123 / -0.062 / -0.0633 [-0.3436, 0.1611] |
| turn: sharp >=0.15 | 303 | 33 | 12.3838 | 12.7954 | -0.4115 [-0.8894, 0.1077] |  | 0.086 / -0.957 / -0.4907 [-1.1936, 0.3324] |
| turn: undef (GT crawling) (low n) | 13 | 11 | 14.6558 | 14.7134 | -0.0576 [-0.2887, 0.1640] |  | 0.004 / -0.006 / -0.0949 [-0.4989, 0.3208] |
| speed: v0 <2 | 186 | 24 | 13.5772 | 13.5362 | 0.0410 [-0.0729, 0.1938] |  | 0.053 / 0.059 / 0.0043 [-0.2472, 0.3046] |
| speed: v0 2-5 | 422 | 52 | 4.4559 | 4.0290 | 0.4268 [-0.0004, 1.0998] |  | 0.120 / 1.382 / 0.4429 [-0.1027, 1.6884] |
| speed: v0 5-10 | 1040 | 79 | 3.8916 | 3.8977 | -0.0061 [-0.1783, 0.2055] |  | 0.295 / -0.049 / -0.0612 [-0.4002, 0.3385] |
| speed: v0 10-15 | 952 | 62 | 2.5970 | 2.6434 | -0.0464 [-0.1315, 0.0236] |  | 0.270 / -0.339 / -0.1143 [-0.3769, 0.0793] |
| speed: v0 15-25 | 679 | 34 | 1.5006 | 1.5135 | -0.0129 [-0.0541, 0.0247] |  | 0.193 / -0.067 / -0.0618 [-0.2780, 0.0922] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4576 | 0.4497 | 0.0078 [-0.0290, 0.0549] |  | 0.070 / 0.015 / -0.0313 [-0.2398, 0.1223] |
| nav: nav=follow | 2249 | 88 | 2.1562 | 2.1598 | -0.0036 [-0.0518, 0.0415] |  | 0.638 / -0.063 / -0.1123 [-0.5743, 0.2094] |
| nav: nav=left (low n) | 319 | 13 | 9.5980 | 9.6462 | -0.0482 [-0.1549, 0.0443] |  | 0.091 / -0.118 / -0.0937 [-0.3358, 0.0939] |
| nav: nav=right | 956 | 38 | 4.3361 | 4.1752 | 0.1609 [-0.1183, 0.5543] |  | 0.271 / 1.181 / 0.1701 [-0.2432, 0.7572] |

episode concentration: 0.583 of episodes have mean delta > 0; top-5 episodes carry 2.083 of the summed delta; delta excluding the top 5 = -0.0416 (overall 0.0370).

### S6 -- EMA-10.5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3525 | 139 | 3.3919 | 3.2465 | 0.1454 [-0.0342, 0.3191] |  |  |
| turn: straight <0.02 | 2098 | 122 | 1.5346 | 1.4595 | 0.0751 [0.0110, 0.1398] | **SEP** | 0.595 / 0.307 / -0.1737 [-0.8037, 0.5115] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.7017 | 2.8900 | -0.1883 [-0.7293, 0.1880] |  | 0.191 / -0.248 / -0.4128 [-1.3495, 0.1290] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.4718 | 6.3633 | 0.1085 [-0.2301, 0.4297] |  | 0.123 / 0.092 / -0.0421 [-0.6004, 0.4532] |
| turn: sharp >=0.15 | 304 | 33 | 12.8508 | 11.4418 | 1.4090 [0.1902, 2.8853] | **SEP** | 0.086 / 0.836 / 1.3829 [-0.4234, 3.6670] |
| turn: undef (GT crawling) (low n) | 13 | 11 | 14.7134 | 14.2040 | 0.5095 [0.0545, 1.2103] | **SEP** | 0.004 / 0.013 / 0.3654 [-0.3154, 1.6921] |
| speed: v0 <2 | 187 | 24 | 13.6258 | 13.8079 | -0.1821 [-0.9210, 0.2642] |  | 0.053 / -0.066 / -0.3458 [-1.9259, 0.3882] |
| speed: v0 2-5 | 422 | 52 | 4.0290 | 4.1501 | -0.1211 [-1.1064, 0.6765] |  | 0.120 / -0.100 / -0.3027 [-2.0481, 0.8554] |
| speed: v0 5-10 | 1040 | 79 | 3.8977 | 3.4941 | 0.4036 [0.0670, 0.8126] | **SEP** | 0.295 / 0.819 / 0.3663 [-0.1342, 1.0121] |
| speed: v0 10-15 | 952 | 62 | 2.6434 | 2.4870 | 0.1564 [-0.0500, 0.3872] |  | 0.270 / 0.290 / 0.0151 [-0.4361, 0.5478] |
| speed: v0 15-25 | 679 | 34 | 1.5135 | 1.4456 | 0.0679 [-0.0136, 0.1623] |  | 0.193 / 0.090 / -0.0960 [-0.4224, 0.2553] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4497 | 0.5193 | -0.0696 [-0.1632, 0.0083] |  | 0.070 / -0.033 / -0.2310 [-0.5296, 0.0996] |
| nav: nav=follow | 2249 | 88 | 2.1601 | 1.9783 | 0.1818 [0.0345, 0.3566] | **SEP** | 0.638 / 0.798 / 0.1005 [-0.5135, 0.7141] |
| nav: nav=left (low n) | 320 | 13 | 9.7087 | 9.7553 | -0.0467 [-0.4725, 0.3121] |  | 0.091 / -0.029 / -0.2113 [-0.9843, 0.3613] |
| nav: nav=right | 956 | 38 | 4.1752 | 4.0510 | 0.1241 [-0.4050, 0.5937] |  | 0.271 / 0.232 / -0.0292 [-0.8284, 0.7446] |

episode concentration: 0.612 of episodes have mean delta > 0; top-5 episodes carry 0.918 of the summed delta; delta excluding the top 5 = 0.0124 (overall 0.1454).

### S6 -- EMA-15k - EMA-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3526 | 139 | 3.4208 | 3.3497 | 0.0711 [-0.0199, 0.1677] |  |  |
| turn: straight <0.02 | 2098 | 122 | 1.6032 | 1.5269 | 0.0763 [0.0029, 0.1498] | **SEP** | 0.595 / 0.638 / 0.0128 [-0.3231, 0.2738] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.8791 | 2.8890 | -0.0099 [-0.1278, 0.1091] |  | 0.191 / -0.027 / -0.1002 [-0.4162, 0.1545] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.4533 | 6.5123 | -0.0590 [-0.2436, 0.1064] |  | 0.123 / -0.102 / -0.1484 [-0.5131, 0.1438] |
| turn: sharp >=0.15 | 304 | 33 | 12.3655 | 11.9701 | 0.3955 [-0.4854, 1.4725] |  | 0.086 / 0.480 / 0.3550 [-0.7983, 2.1047] |
| turn: undef (GT crawling) (low n) | 14 | 10 | 13.4806 | 13.2809 | 0.1997 [0.0857, 0.3372] | **SEP** | 0.004 / 0.011 / 0.1291 [-0.0984, 0.3797] |
| speed: v0 <2 | 188 | 25 | 13.4625 | 13.6810 | -0.2184 [-0.7352, 0.1185] |  | 0.053 / -0.164 / -0.3058 [-1.3978, 0.2334] |
| speed: v0 2-5 | 422 | 52 | 4.4559 | 4.3226 | 0.1332 [-0.1797, 0.4904] |  | 0.120 / 0.224 / 0.0706 [-0.3885, 0.7368] |
| speed: v0 5-10 | 1040 | 79 | 3.8916 | 3.6560 | 0.2356 [-0.0300, 0.5631] |  | 0.295 / 0.977 / 0.2333 [-0.1564, 0.7296] |
| speed: v0 10-15 | 952 | 62 | 2.5970 | 2.6253 | -0.0283 [-0.1508, 0.0871] |  | 0.270 / -0.107 / -0.1362 [-0.4150, 0.1193] |
| speed: v0 15-25 | 679 | 34 | 1.5006 | 1.4734 | 0.0271 [-0.0448, 0.0994] |  | 0.193 / 0.073 / -0.0545 [-0.2667, 0.1310] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4576 | 0.4614 | -0.0038 [-0.0475, 0.0405] |  | 0.069 / -0.004 / -0.0805 [-0.2493, 0.0817] |
| nav: nav=follow | 2249 | 88 | 2.1550 | 2.0711 | 0.0839 [-0.0079, 0.1930] |  | 0.638 / 0.753 / 0.0353 [-0.3045, 0.3545] |
| nav: nav=left (low n) | 319 | 13 | 9.5980 | 9.8407 | -0.2427 [-0.5430, 0.0272] |  | 0.090 / -0.309 / -0.3450 [-0.9089, 0.0730] |
| nav: nav=right | 958 | 38 | 4.3357 | 4.1901 | 0.1456 [-0.0733, 0.3945] |  | 0.272 / 0.556 / 0.1022 [-0.2546, 0.5059] |

episode concentration: 0.576 of episodes have mean delta > 0; top-5 episodes carry 1.102 of the summed delta; delta excluding the top 5 = -0.0075 (overall 0.0711).

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

### S6 -- EMA-15k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

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

### S6 -- EMA-15k - EMA-10.5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3490 | 139 | 1.0014 | 0.9343 | 0.0671 [0.0047, 0.1742] | **SEP** | 0.0399 [-0.0282, 0.1529] |
| seg1 | straight | 2071 | 122 | 0.6664 | 0.6464 | 0.0200 [0.0034, 0.0419] | **SEP** | -0.0048 [-0.0269, 0.0126] |
| seg1 | curved | 1407 | 88 | 1.3703 | 1.2333 | 0.1370 [-0.0078, 0.4154] |  | 0.1071 [-0.0482, 0.3933] |
| seg2 | all | 3479 | 139 | 1.5424 | 1.2967 | 0.2457 [0.0098, 0.6738] | **SEP** | 0.1503 [-0.1020, 0.5948] |
| seg2 | straight | 2073 | 122 | 1.0675 | 0.5975 | 0.4700 [0.0081, 1.4257] | **SEP** | 0.4044 [-0.0611, 1.3612] |
| seg2 | curved | 1406 | 88 | 2.2425 | 2.3275 | -0.0850 [-0.4123, 0.1357] |  | -0.2243 [-0.5149, -0.0316] SEP |
| seg3 | all | 3443 | 139 | 2.2066 | 2.1563 | 0.0503 [-0.0302, 0.1502] |  | -0.0236 [-0.2068, 0.2323] |
| seg3 | straight | 2054 | 121 | 1.1375 | 1.0095 | 0.1281 [-0.0048, 0.3418] |  | 0.1916 [-0.1244, 0.7768] |
| seg3 | curved | 1389 | 88 | 3.7876 | 3.8523 | -0.0646 [-0.4204, 0.2258] |  | -0.3418 [-0.6544, -0.0898] SEP |
| seg4 | all | 3402 | 139 | 3.1977 | 3.2324 | -0.0348 [-0.1981, 0.1332] |  | 0.0682 [-0.2970, 0.6557] |
| seg4 | straight | 2036 | 121 | 1.2843 | 1.3275 | -0.0432 [-0.2710, 0.1271] |  | 0.3160 [-0.2073, 1.2929] |
| seg4 | curved | 1366 | 87 | 6.0495 | 6.0717 | -0.0222 [-0.2450, 0.2746] |  | -0.3011 [-0.5767, -0.0644] SEP |
| seg5 | all | 3342 | 139 | 4.4757 | 4.5002 | -0.0244 [-0.1740, 0.1642] |  | -0.1799 [-0.4164, 0.0976] |
| seg5 | straight | 2009 | 121 | 1.3842 | 1.3259 | 0.0583 [-0.0496, 0.2150] |  | -0.0081 [-0.3030, 0.4298] |
| seg5 | curved | 1333 | 88 | 9.1351 | 9.2842 | -0.1491 [-0.4583, 0.2569] |  | -0.4387 [-0.8274, -0.1319] SEP |
| seg6 | all | 3302 | 138 | 6.2084 | 6.1713 | 0.0371 [-0.1625, 0.2672] |  | -0.4344 [-0.6844, -0.2225] SEP |
| seg6 | straight | 1996 | 120 | 1.8768 | 1.6895 | 0.1873 [-0.0106, 0.4595] |  | -0.3362 [-0.6409, -0.1067] SEP |
| seg6 | curved | 1305 | 88 | 12.8358 | 13.0277 | -0.1919 [-0.5607, 0.2771] |  | -0.5842 [-1.0433, -0.1960] SEP |

### S6 -- EMA-10.5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3490 | 139 | 0.9343 | 0.7643 | 0.1701 [-0.0084, 0.4937] |  | -0.0829 [-0.2010, 0.0021] |
| seg1 | straight | 2071 | 122 | 0.6464 | 0.5434 | 0.1030 [0.0051, 0.2954] | **SEP** | 0.0509 [-0.0525, 0.2477] |
| seg1 | curved | 1407 | 88 | 1.2333 | 0.9651 | 0.2683 [-0.0398, 0.8534] |  | -0.2794 [-0.8722, 0.0520] |
| seg2 | all | 3482 | 139 | 1.2975 | 1.4585 | -0.1610 [-0.6012, 0.0998] |  | 0.0034 [-0.0935, 0.1095] |
| seg2 | straight | 2075 | 122 | 0.5973 | 0.8394 | -0.2420 [-0.8143, 0.0476] |  | -0.1235 [-0.3273, -0.0031] SEP |
| seg2 | curved | 1407 | 88 | 2.3300 | 2.3716 | -0.0416 [-0.3516, 0.2112] |  | 0.1906 [-0.0976, 0.5140] |
| seg3 | all | 3448 | 139 | 2.1609 | 2.0566 | 0.1043 [-0.0512, 0.2585] |  | 0.4564 [0.0436, 1.1346] SEP |
| seg3 | straight | 2057 | 121 | 1.0093 | 0.9782 | 0.0310 [-0.0150, 0.0813] |  | 0.4034 [0.0132, 1.1768] SEP |
| seg3 | curved | 1391 | 88 | 3.8638 | 3.6512 | 0.2126 [-0.1694, 0.5849] |  | 0.5347 [0.0185, 1.1532] SEP |
| seg4 | all | 3411 | 139 | 3.2565 | 2.9618 | 0.2947 [0.0349, 0.5561] | **SEP** | 0.3572 [0.1465, 0.6058] SEP |
| seg4 | straight | 2041 | 121 | 1.3280 | 1.1386 | 0.1894 [0.0464, 0.4136] | **SEP** | 0.2319 [0.0866, 0.4568] SEP |
| seg4 | curved | 1370 | 87 | 6.1294 | 5.6779 | 0.4515 [-0.1076, 1.0330] |  | 0.5439 [0.0571, 1.0927] SEP |
| seg5 | all | 3353 | 139 | 4.5244 | 4.2807 | 0.2437 [-0.0579, 0.5502] |  | 0.5181 [0.2458, 0.8257] SEP |
| seg5 | straight | 2015 | 121 | 1.3294 | 1.3087 | 0.0207 [-0.2005, 0.1828] |  | 0.2873 [0.1177, 0.5292] SEP |
| seg5 | curved | 1338 | 88 | 9.3361 | 8.7564 | 0.5797 [-0.1034, 1.2836] |  | 0.8658 [0.2385, 1.5201] SEP |
| seg6 | all | 3311 | 138 | 6.1911 | 5.9044 | 0.2867 [-0.0765, 0.6294] |  | 0.5107 [0.1943, 0.8534] SEP |
| seg6 | straight | 2000 | 120 | 1.6929 | 1.7283 | -0.0354 [-0.2800, 0.1497] |  | 0.3152 [0.1066, 0.5747] SEP |
| seg6 | curved | 1310 | 88 | 13.0599 | 12.2845 | 0.7754 [-0.0459, 1.6004] |  | 0.8065 [0.0673, 1.5491] SEP |

### S6 -- EMA-15k - EMA-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3493 | 139 | 1.0010 | 0.7144 | 0.2866 [0.0142, 0.8111] | **SEP** | 0.1563 [-0.0188, 0.4809] |
| seg1 | straight | 2071 | 122 | 0.6664 | 0.3744 | 0.2919 [0.0104, 0.8663] | **SEP** | 0.2670 [-0.0197, 0.8493] |
| seg1 | curved | 1408 | 88 | 1.3694 | 1.0902 | 0.2792 [0.0030, 0.8371] | **SEP** | -0.0028 [-0.0413, 0.0354] |
| seg2 | all | 3479 | 139 | 1.5493 | 1.5324 | 0.0170 [-0.1075, 0.1057] |  | 0.1676 [-0.0398, 0.5178] |
| seg2 | straight | 2073 | 122 | 1.0792 | 1.0227 | 0.0565 [0.0056, 0.1281] | **SEP** | 0.3476 [-0.0522, 1.1176] |
| seg2 | curved | 1406 | 88 | 2.2425 | 2.2838 | -0.0413 [-0.3411, 0.1523] |  | -0.0977 [-0.4183, 0.1197] |
| seg3 | all | 3444 | 139 | 2.2090 | 2.1835 | 0.0254 [-0.1218, 0.1483] |  | 0.1511 [0.0163, 0.3142] SEP |
| seg3 | straight | 2054 | 121 | 1.1375 | 1.1125 | 0.0250 [-0.0208, 0.0731] |  | 0.2430 [0.0275, 0.6397] SEP |
| seg3 | curved | 1390 | 88 | 3.7922 | 3.7662 | 0.0260 [-0.3341, 0.3201] |  | 0.0153 [-0.3596, 0.3145] |
| seg4 | all | 3392 | 139 | 3.1703 | 3.0463 | 0.1240 [-0.0177, 0.2822] |  | 0.3203 [0.0437, 0.7163] SEP |
| seg4 | straight | 2036 | 121 | 1.2807 | 1.1247 | 0.1560 [0.0178, 0.3762] | **SEP** | 0.3483 [0.0322, 0.9428] SEP |
| seg4 | curved | 1356 | 87 | 6.0074 | 5.9315 | 0.0759 [-0.1867, 0.3918] |  | 0.2783 [-0.0768, 0.6650] |
| seg5 | all | 3342 | 139 | 4.4834 | 4.4215 | 0.0619 [-0.0831, 0.2233] |  | 0.1800 [-0.0291, 0.3989] |
| seg5 | straight | 2006 | 121 | 1.3749 | 1.3011 | 0.0738 [-0.0076, 0.1561] |  | 0.1149 [0.0082, 0.2147] SEP |
| seg5 | curved | 1336 | 88 | 9.1508 | 9.1068 | 0.0440 [-0.2838, 0.4266] |  | 0.2777 [-0.2235, 0.8135] |
| seg6 | all | 3302 | 138 | 6.2027 | 6.0576 | 0.1451 [-0.0585, 0.3790] |  | 0.0232 [-0.2793, 0.3165] |
| seg6 | straight | 1995 | 120 | 1.8717 | 1.7002 | 0.1715 [0.0028, 0.4255] | **SEP** | -0.0303 [-0.2949, 0.1812] |
| seg6 | curved | 1307 | 88 | 12.8136 | 12.7088 | 0.1048 [-0.3077, 0.5546] |  | 0.1048 [-0.5309, 0.7500] |

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

### S6 -- EMA-15k - refcv7-50400: turn x speed cross

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

| bin | EMA-15k s0-s1 [CI] | EMA-10.5k s0-s1 [CI] | EMA-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|---|
| overall | -0.0062 [-0.0477, 0.0363] | -0.0064 [-0.0498, 0.0457] | -0.0153 [-0.0751, 0.0424] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | -0.0084 [-0.0312, 0.0115] | 0.0000 [-0.0297, 0.0321] | -0.0024 [-0.0263, 0.0190] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0191 [-0.0540, 0.0983] | 0.0243 [-0.1031, 0.2032] | 0.0493 [-0.0553, 0.1670] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.1019 [-0.2216, 0.0227] | -0.0237 [-0.1328, 0.0888] | 0.0108 [-0.1388, 0.1615] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | 0.0928 [-0.2494, 0.4389] | -0.1128 [-0.4015, 0.1588] | -0.2362 [-0.7709, 0.2315] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | -0.0822 [-0.5704, 0.3910] | 0.4538 [0.0969, 0.9207] SEP | -0.7635 [-1.9674, 0.4258] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | -0.1561 [-0.3462, -0.0084] SEP | -0.0574 [-0.2403, 0.1549] | 0.2812 [0.0192, 0.7088] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | -0.0806 [-0.2142, 0.0344] | 0.0116 [-0.2115, 0.3108] | -0.1594 [-0.3925, 0.0077] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | 0.0315 [-0.0543, 0.1270] | 0.0409 [-0.0569, 0.1368] | -0.0367 [-0.1984, 0.0987] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | 0.0076 [-0.0473, 0.0831] | -0.0616 [-0.1130, -0.0170] SEP | -0.0091 [-0.0758, 0.0548] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | -0.0017 [-0.0317, 0.0242] | -0.0007 [-0.0271, 0.0258] | 0.0031 [-0.0236, 0.0284] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | 0.0121 [-0.0033, 0.0310] | 0.0003 [-0.0086, 0.0111] | 0.0016 [-0.0069, 0.0122] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0074 [-0.0492, 0.0343] | -0.0193 [-0.0557, 0.0184] | 0.0235 [-0.0170, 0.0654] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | 0.0066 [-0.1012, 0.0914] | -0.0334 [-0.1654, 0.1030] | 0.0874 [-0.0438, 0.2535] | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | -0.0078 [-0.1120, 0.1111] | 0.0332 [-0.1061, 0.1944] | -0.1398 [-0.3317, 0.0343] | 0.0235 [-0.1175, 0.1775] |

### S6 -- EMA-15k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

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

### S6 -- EMA-15k - EMA-10.5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0370 | 0.0380 / 0.0359 | False / False / False | 0.0062 / 0.0064 | 5.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0687 | 0.0642 / 0.0731 | False / True / True | 0.0084 / 0.0000 | 8.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|gentle 0.02-0.06 | 0.1774 | 0.1748 / 0.1800 | False / False / False | 0.0191 / 0.0243 | 7.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0186 | -0.0576 / 0.0205 | False / False / False | 0.1019 / 0.0237 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | -0.4115 | -0.2952 / -0.5279 | False / False / False | 0.0928 / 0.1128 | 3.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.0576 | -0.3337 / 0.2185 | True / False / False | 0.0822 / 0.4538 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0410 | 0.0087 / 0.0733 | False / False / False | 0.1561 / 0.0574 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.4268 | 0.3807 / 0.4729 | True / False / False | 0.0806 / 0.0116 | 5.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | -0.0061 | -0.0109 / -0.0014 | False / False / False | 0.0315 / 0.0409 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.0464 | -0.0119 / -0.0810 | False / False / False | 0.0076 / 0.0616 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0129 | -0.0134 / -0.0124 | False / False / False | 0.0017 / 0.0007 | 7.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | 0.0078 | 0.0138 / 0.0019 | False / False / False | 0.0121 / 0.0003 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | -0.0036 | 0.0021 / -0.0094 | False / False / False | 0.0074 / 0.0193 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.0482 | -0.0147 / -0.0817 | False / False / False | 0.0066 / 0.0334 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1609 | 0.1400 / 0.1818 | False / False / False | 0.0078 / 0.0332 | 4.8 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- EMA-10.5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1454 | 0.1323 / 0.1822 | False / True / False | 0.0064 / 0.0128 | 11.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0751 | 0.0878 / 0.0624 | True / False / True | 0.0000 / 0.0216 | 3.5 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|gentle 0.02-0.06 | -0.1883 | -0.2123 / -0.1643 | False / False / False | 0.0243 / 0.0713 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.1085 | 0.0745 / 0.1424 | False / False / False | 0.0237 / 0.0611 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 1.4090 | 1.2607 / 1.8208 | False / True / True | 0.1128 / 0.2004 | 7.0 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | 0.5095 | 0.7430 / 0.2759 | True / False / True | 0.4538 / 1.9278 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.1821 | -0.2976 / 0.3771 | False / False / False | 0.0574 / 0.0328 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.1211 | -0.2157 / -0.0264 | False / False / False | 0.0116 / 0.2009 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.4036 | 0.4362 / 0.3710 | True / True / True | 0.0409 / 0.0243 | 9.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1564 | 0.1390 / 0.1738 | False / False / False | 0.0616 / 0.0268 | 2.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0679 | 0.0648 / 0.0710 | False / False / False | 0.0007 / 0.0055 | 12.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0696 | -0.0698 / -0.0693 | False / False / False | 0.0003 / 0.0008 | 83.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1818 | 0.1672 / 0.1963 | True / True / True | 0.0193 / 0.0080 | 9.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.0467 | -0.0598 / 0.2267 | False / False / False | 0.0334 / 0.0140 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1241 | 0.1143 / 0.1340 | False / False / False | 0.0332 / 0.0235 | 3.7 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- EMA-15k - EMA-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0711 | 0.0735 / 0.0687 | False / False / False | 0.0062 / 0.0153 | 4.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0763 | 0.0736 / 0.0790 | False / True / True | 0.0084 / 0.0024 | 9.1 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|gentle 0.02-0.06 | -0.0099 | -0.0250 / 0.0052 | False / False / False | 0.0191 / 0.0493 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0590 | -0.1148 / -0.0032 | False / False / False | 0.1019 / 0.0108 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.3955 | 0.5624 / 0.2285 | False / False / False | 0.0928 / 0.2362 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.1997 | 0.0429 / 0.3565 | False / True / True | 0.0822 / 0.7635 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.2184 | -0.4906 / 0.0538 | True / False / False | 0.1561 / 0.2812 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1332 | 0.1726 / 0.0939 | False / False / False | 0.0806 / 0.1594 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2356 | 0.2697 / 0.2015 | False / False / False | 0.0315 / 0.0367 | 6.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | -0.0283 | -0.0200 / -0.0366 | False / False / False | 0.0076 / 0.0091 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0271 | 0.0247 / 0.0295 | False / False / False | 0.0017 / 0.0031 | 8.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0038 | 0.0015 / -0.0090 | False / False / False | 0.0121 / 0.0016 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0839 | 0.0685 / 0.0992 | False / False / False | 0.0074 / 0.0235 | 3.6 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.2427 | -0.2828 / -0.2026 | False / False / False | 0.0066 / 0.0874 | 2.8 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1456 | 0.2038 / 0.0873 | False / False / False | 0.0078 / 0.1398 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |

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

* seg1 (n 1407): EMA-15k bias -0.021, refcv7-50400 bias +0.166, EMA-10.5k bias -0.157, EMA-5k bias -0.008; EMA-15k MAE 1.370, refcv7-50400 MAE 0.965, EMA-10.5k MAE 1.233, EMA-5k MAE 1.091
* seg2 (n 1406): EMA-15k bias -0.425, refcv7-50400 bias -0.232, EMA-10.5k bias -0.320, EMA-5k bias -0.233; EMA-15k MAE 2.242, refcv7-50400 MAE 2.372, EMA-10.5k MAE 2.328, EMA-5k MAE 2.284
* seg3 (n 1389): EMA-15k bias -1.374, refcv7-50400 bias -1.103, EMA-10.5k bias -1.251, EMA-5k bias -1.027; EMA-15k MAE 3.788, refcv7-50400 MAE 3.652, EMA-10.5k MAE 3.852, EMA-5k MAE 3.763
* seg4 (n 1355): EMA-15k bias -3.045, refcv7-50400 bias -2.471, EMA-10.5k bias -3.068, EMA-5k bias -2.688; EMA-15k MAE 6.011, refcv7-50400 MAE 5.684, EMA-10.5k MAE 6.018, EMA-5k MAE 5.934
* seg5 (n 1330): EMA-15k bias -4.930, refcv7-50400 bias -4.409, EMA-10.5k bias -4.930, EMA-5k bias -4.455; EMA-15k MAE 9.148, refcv7-50400 MAE 8.748, EMA-10.5k MAE 9.293, EMA-5k MAE 9.110
* seg6 (n 1300): EMA-15k bias -6.952, refcv7-50400 bias -6.783, EMA-10.5k bias -6.946, EMA-5k bias -6.517; EMA-15k MAE 12.844, refcv7-50400 MAE 12.274, EMA-10.5k MAE 13.070, EMA-5k MAE 12.742
**sharp (>= 0.15)**

* seg1 (n 302): EMA-15k bias -0.666, refcv7-50400 bias +0.176, EMA-10.5k bias -0.728, EMA-5k bias -0.028; EMA-15k MAE 2.008, refcv7-50400 MAE 1.443, EMA-10.5k MAE 1.982, EMA-5k MAE 1.366
* seg2 (n 298): EMA-15k bias -1.477, refcv7-50400 bias -0.532, EMA-10.5k bias -1.661, EMA-5k bias -1.145; EMA-15k MAE 4.215, refcv7-50400 MAE 3.819, EMA-10.5k MAE 4.175, EMA-5k MAE 3.830
* seg3 (n 294): EMA-15k bias -4.539, refcv7-50400 bias -2.472, EMA-10.5k bias -4.776, EMA-5k bias -3.557; EMA-15k MAE 7.785, refcv7-50400 MAE 6.662, EMA-10.5k MAE 7.843, EMA-5k MAE 7.053
* seg4 (n 292): EMA-15k bias -9.139, refcv7-50400 bias -6.266, EMA-10.5k bias -9.436, EMA-5k bias -7.390; EMA-15k MAE 13.150, refcv7-50400 MAE 11.545, EMA-10.5k MAE 13.732, EMA-5k MAE 12.620
* seg5 (n 288): EMA-15k bias -14.258, refcv7-50400 bias -11.354, EMA-10.5k bias -14.450, EMA-5k bias -11.969; EMA-15k MAE 20.523, refcv7-50400 MAE 18.673, EMA-10.5k MAE 21.546, EMA-5k MAE 19.888
* seg6 (n 283): EMA-15k bias -18.971, refcv7-50400 bias -16.801, EMA-10.5k bias -18.985, EMA-5k bias -16.873; EMA-15k MAE 27.127, refcv7-50400 MAE 24.711, EMA-10.5k MAE 28.230, EMA-5k MAE 25.986
**curved & v0 5-10**

* seg1 (n 547): EMA-15k bias +0.030, refcv7-50400 bias +0.049, EMA-10.5k bias +0.018, EMA-5k bias +0.052; EMA-15k MAE 0.971, refcv7-50400 MAE 0.896, EMA-10.5k MAE 0.959, EMA-5k MAE 0.900
* seg2 (n 547): EMA-15k bias -0.606, refcv7-50400 bias -0.516, EMA-10.5k bias -0.639, EMA-5k bias -0.472; EMA-15k MAE 2.550, refcv7-50400 MAE 2.231, EMA-10.5k MAE 2.509, EMA-5k MAE 2.292
* seg3 (n 544): EMA-15k bias -1.951, refcv7-50400 bias -1.675, EMA-10.5k bias -2.006, EMA-5k bias -1.543; EMA-15k MAE 4.364, refcv7-50400 MAE 3.806, EMA-10.5k MAE 4.411, EMA-5k MAE 3.914
* seg4 (n 521): EMA-15k bias -4.071, refcv7-50400 bias -3.553, EMA-10.5k bias -4.256, EMA-5k bias -3.330; EMA-15k MAE 6.679, refcv7-50400 MAE 6.254, EMA-10.5k MAE 6.917, EMA-5k MAE 6.271
* seg5 (n 502): EMA-15k bias -6.342, refcv7-50400 bias -5.847, EMA-10.5k bias -6.499, EMA-5k bias -5.323; EMA-15k MAE 10.015, refcv7-50400 MAE 9.403, EMA-10.5k MAE 10.465, EMA-5k MAE 9.462
* seg6 (n 477): EMA-15k bias -8.218, refcv7-50400 bias -8.173, EMA-10.5k bias -8.368, EMA-5k bias -7.262; EMA-15k MAE 13.422, refcv7-50400 MAE 12.564, EMA-10.5k MAE 14.142, EMA-5k MAE 12.836
**sharp & v0 5-10**

* seg1 (n 139): EMA-15k bias -0.037, refcv7-50400 bias +0.062, EMA-10.5k bias -0.094, EMA-5k bias +0.031; EMA-15k MAE 1.259, refcv7-50400 MAE 1.251, EMA-10.5k MAE 1.139, EMA-5k MAE 1.109
* seg2 (n 139): EMA-15k bias -2.021, refcv7-50400 bias -1.267, EMA-10.5k bias -2.182, EMA-5k bias -1.331; EMA-15k MAE 4.256, refcv7-50400 MAE 3.564, EMA-10.5k MAE 4.040, EMA-5k MAE 3.402
* seg3 (n 138): EMA-15k bias -6.246, refcv7-50400 bias -4.253, EMA-10.5k bias -6.570, EMA-5k bias -4.386; EMA-15k MAE 8.552, refcv7-50400 MAE 6.990, EMA-10.5k MAE 8.727, EMA-5k MAE 6.905
* seg4 (n 136): EMA-15k bias -11.668, refcv7-50400 bias -8.928, EMA-10.5k bias -12.317, EMA-5k bias -8.568; EMA-15k MAE 14.471, refcv7-50400 MAE 12.348, EMA-10.5k MAE 15.421, EMA-5k MAE 12.505
* seg5 (n 135): EMA-15k bias -17.059, refcv7-50400 bias -14.156, EMA-10.5k bias -17.625, EMA-5k bias -12.848; EMA-15k MAE 22.735, refcv7-50400 MAE 19.609, EMA-10.5k MAE 24.221, EMA-5k MAE 19.893
* seg6 (n 130): EMA-15k bias -19.848, refcv7-50400 bias -17.837, EMA-10.5k bias -20.348, EMA-5k bias -15.494; EMA-15k MAE 28.643, refcv7-50400 MAE 25.199, EMA-10.5k MAE 30.658, EMA-5k MAE 25.617

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1300 / 1300 | 26.1 | EMA-15k -0.2572 [-0.3549, -0.1727]; refcv7-50400 -0.2412 [-0.3098, -0.1759]; EMA-10.5k -0.2571 [-0.3533, -0.1730]; EMA-5k -0.2325 [-0.3138, -0.1614] | EMA-15k -0.0160 [-0.0636, +0.0199]; EMA-10.5k -0.0159 [-0.0614, +0.0194]; EMA-5k +0.0087 [-0.0217, +0.0347] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1413 / 6780 | 16.4 | EMA-15k -0.1986 [-0.2910, -0.1217]; refcv7-50400 -0.1743 [-0.2383, -0.1170]; EMA-10.5k -0.1851 [-0.2826, -0.1010]; EMA-5k -0.1553 [-0.2399, -0.0814] | EMA-15k -0.0243 [-0.0729, +0.0144]; EMA-10.5k -0.0108 [-0.0642, +0.0488]; EMA-5k +0.0190 [-0.0243, +0.0725] |
| sharp (>= 0.15) | last segment | 283 / 283 | 63.0 | EMA-15k -0.2866 [-0.4125, -0.1878]; refcv7-50400 -0.2563 [-0.3393, -0.1824]; EMA-10.5k -0.2855 [-0.4105, -0.1881]; EMA-5k -0.2552 [-0.3572, -0.1723] | EMA-15k -0.0303 [-0.0982, +0.0157]; EMA-10.5k -0.0291 [-0.0922, +0.0158]; EMA-5k +0.0012 [-0.0383, +0.0302] |
| sharp (>= 0.15) | segments 2..K | 303 / 1455 | 39.8 | EMA-15k -0.2382 [-0.3615, -0.1400]; refcv7-50400 -0.1883 [-0.2712, -0.1188]; EMA-10.5k -0.2383 [-0.3606, -0.1407]; EMA-5k -0.1999 [-0.3041, -0.1177] | EMA-15k -0.0499 [-0.1206, -0.0026] SEP; EMA-10.5k -0.0501 [-0.1155, -0.0058] SEP; EMA-5k -0.0117 [-0.0520, +0.0178] |
| curved & v0 5-10 | last segment | 477 / 477 | 28.5 | EMA-15k -0.2832 [-0.4157, -0.1611]; refcv7-50400 -0.2583 [-0.3799, -0.1642]; EMA-10.5k -0.2863 [-0.4211, -0.1574]; EMA-5k -0.2341 [-0.3528, -0.1329] | EMA-15k -0.0250 [-0.1175, +0.0495]; EMA-10.5k -0.0280 [-0.1096, +0.0439]; EMA-5k +0.0241 [-0.0136, +0.0608] |
| curved & v0 5-10 | segments 2..K | 547 / 2591 | 17.9 | EMA-15k -0.2372 [-0.3571, -0.1309]; refcv7-50400 -0.1984 [-0.3113, -0.1151]; EMA-10.5k -0.2415 [-0.3626, -0.1317]; EMA-5k -0.1857 [-0.2931, -0.1021] | EMA-15k -0.0388 [-0.1234, +0.0292]; EMA-10.5k -0.0430 [-0.1193, +0.0200]; EMA-5k +0.0128 [-0.0208, +0.0442] |
| sharp & v0 5-10 | last segment | 130 / 130 | 67.1 | EMA-15k -0.2967 [-0.4571, -0.1567]; refcv7-50400 -0.2565 [-0.3869, -0.1508]; EMA-10.5k -0.3023 [-0.4588, -0.1555]; EMA-5k -0.2357 [-0.3698, -0.1224] | EMA-15k -0.0402 [-0.1551, +0.0493]; EMA-10.5k -0.0458 [-0.1498, +0.0376]; EMA-5k +0.0208 [-0.0245, +0.0620] |
| sharp & v0 5-10 | segments 2..K | 139 / 678 | 43.8 | EMA-15k -0.2585 [-0.4084, -0.1306]; refcv7-50400 -0.2031 [-0.3332, -0.1089]; EMA-10.5k -0.2651 [-0.4106, -0.1321]; EMA-5k -0.1944 [-0.3207, -0.0941] | EMA-15k -0.0554 [-0.1632, +0.0279]; EMA-10.5k -0.0619 [-0.1599, +0.0138]; EMA-5k +0.0087 [-0.0330, +0.0474] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA-15k - refcv7-50400**

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
**EMA-15k - EMA-10.5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3524 | 0.0370 | 0.00462 | 0.00439 | 0.517 / 0.483 / 0.000 | 419.6 / 130.3 |
| turn|straight <0.02 | 2098 | 0.0687 | 0.00841 | 0.01270 | 0.540 / 0.460 / 0.000 | 165.3 / 144.1 |
| turn|gentle 0.02-0.06 | 675 | 0.1774 | 0.00749 | 0.01089 | 0.523 / 0.477 / 0.000 | 132.5 / 119.7 |
| turn|turn 0.06-0.15 | 435 | -0.0186 | -0.03987 | -0.04098 | 0.437 / 0.563 / 0.000 | 29.0 / -8.1 |
| turn|sharp >=0.15 | 303 | -0.4115 | -0.04386 | -0.19776 | 0.472 / 0.528 / 0.000 | 54.3 / -124.7 |
| turn|undef (GT crawling) | 13 | -0.0576 | -0.06490 | -0.07153 | 0.308 / 0.692 / 0.000 | 0.7 / -0.7 |
| speed|v0 <2 | 186 | 0.0410 | -0.00481 | -0.00863 | 0.473 / 0.527 / 0.000 | 4.6 / 7.6 |
| speed|v0 2-5 | 422 | 0.4268 | 0.00324 | 0.03266 | 0.507 / 0.493 / 0.000 | 140.8 / 180.1 |
| speed|v0 5-10 | 1040 | -0.0061 | -0.00697 | -0.02693 | 0.487 / 0.513 / 0.000 | 156.3 / -6.4 |
| speed|v0 10-15 | 952 | -0.0464 | 0.01438 | 0.02151 | 0.567 / 0.433 / 0.000 | 43.4 / -44.2 |
| speed|v0 15-25 | 679 | -0.0129 | 0.00444 | 0.01041 | 0.518 / 0.482 / 0.000 | 9.5 / -8.8 |
| speed|v0 >=25 | 245 | 0.0078 | -0.00021 | 0.00501 | 0.498 / 0.502 / 0.000 | 2.1 / 1.9 |
| nav|nav=follow | 2249 | -0.0036 | 0.00273 | 0.00430 | 0.513 / 0.487 / 0.000 | 138.2 / -8.2 |
| nav|nav=left | 319 | -0.0482 | -0.01393 | -0.01157 | 0.476 / 0.524 / 0.000 | 12.6 / -15.4 |
| nav|nav=right | 956 | 0.1609 | 0.01600 | 0.01304 | 0.541 / 0.459 / 0.000 | 250.5 / 153.9 |
**EMA-10.5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3525 | 0.1454 | 0.02354 | 0.03780 | 0.543 / 0.457 / 0.000 | 697.7 / 512.5 |
| turn|straight <0.02 | 2098 | 0.0751 | 0.02532 | 0.03289 | 0.564 / 0.436 / 0.000 | 139.0 / 157.5 |
| turn|gentle 0.02-0.06 | 675 | -0.1883 | -0.00741 | -0.03732 | 0.480 / 0.520 / 0.000 | 63.9 / -127.1 |
| turn|turn 0.06-0.15 | 435 | 0.1085 | 0.08640 | 0.13987 | 0.540 / 0.460 / 0.000 | 47.7 / 47.2 |
| turn|sharp >=0.15 | 304 | 1.4090 | 0.12766 | 0.49622 | 0.536 / 0.464 / 0.000 | 165.8 / 428.3 |
| turn|undef (GT crawling) | 13 | 0.5095 | 0.28851 | 0.25924 | 0.692 / 0.308 / 0.000 | 4.1 / 6.6 |
| speed|v0 <2 | 187 | -0.1821 | -0.02632 | 0.00017 | 0.492 / 0.508 / 0.000 | 24.2 / -34.0 |
| speed|v0 2-5 | 422 | -0.1211 | 0.06449 | 0.11784 | 0.562 / 0.438 / 0.000 | 102.9 / -51.1 |
| speed|v0 5-10 | 1040 | 0.4036 | 0.03626 | 0.05879 | 0.542 / 0.458 / 0.000 | 346.9 / 419.7 |
| speed|v0 10-15 | 952 | 0.1564 | 0.04183 | 0.04775 | 0.577 / 0.423 / 0.000 | 107.5 / 148.9 |
| speed|v0 15-25 | 679 | 0.0679 | 0.01194 | 0.01205 | 0.539 / 0.461 / 0.000 | 37.2 / 46.1 |
| speed|v0 >=25 | 245 | -0.0696 | -0.02818 | -0.03919 | 0.433 / 0.567 / 0.000 | 1.9 / -17.0 |
| nav|nav=follow | 2249 | 0.1818 | 0.01702 | 0.02416 | 0.532 / 0.468 / 0.000 | 350.2 / 408.8 |
| nav|nav=left | 320 | -0.0467 | 0.06156 | 0.10276 | 0.591 / 0.409 / 0.000 | 53.0 / -14.9 |
| nav|nav=right | 956 | 0.1241 | 0.03795 | 0.04970 | 0.553 / 0.447 / 0.000 | 281.0 / 118.7 |
**EMA-15k - EMA-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3526 | 0.0711 | 0.02071 | 0.02907 | 0.537 / 0.463 / 0.000 | 503.6 / 250.7 |
| turn|straight <0.02 | 2098 | 0.0763 | 0.02400 | 0.03125 | 0.558 / 0.442 / 0.000 | 180.3 / 160.0 |
| turn|gentle 0.02-0.06 | 675 | -0.0099 | 0.02185 | 0.03086 | 0.526 / 0.474 / 0.000 | 36.4 / -6.7 |
| turn|turn 0.06-0.15 | 435 | -0.0590 | -0.00867 | 0.00248 | 0.492 / 0.508 / 0.000 | 30.0 / -25.7 |
| turn|sharp >=0.15 | 304 | 0.3955 | -0.05140 | -0.03933 | 0.474 / 0.526 / 0.000 | 132.0 / 120.2 |
| turn|undef (GT crawling) | 14 | 0.1997 | 0.11972 | 0.17094 | 0.786 / 0.214 / 0.000 | 0.9 / 2.8 |
| speed|v0 <2 | 188 | -0.2184 | -0.00261 | 0.01111 | 0.500 / 0.500 / 0.000 | 5.6 / -41.1 |
| speed|v0 2-5 | 422 | 0.1332 | 0.06162 | 0.07776 | 0.578 / 0.422 / 0.000 | 68.6 / 56.2 |
| speed|v0 5-10 | 1040 | 0.2356 | 0.01550 | 0.03324 | 0.524 / 0.476 / 0.000 | 267.2 / 245.0 |
| speed|v0 10-15 | 952 | -0.0283 | 0.03288 | 0.03262 | 0.550 / 0.450 / 0.000 | 72.7 / -26.9 |
| speed|v0 15-25 | 679 | 0.0271 | 0.01127 | 0.02290 | 0.543 / 0.457 / 0.000 | 22.8 / 18.4 |
| speed|v0 >=25 | 245 | -0.0038 | -0.00533 | -0.00382 | 0.486 / 0.514 / 0.000 | 2.1 / -0.9 |
| nav|nav=follow | 2249 | 0.0839 | 0.01598 | 0.02475 | 0.539 / 0.461 / 0.000 | 267.2 / 188.7 |
| nav|nav=left | 319 | -0.2427 | 0.01360 | 0.00430 | 0.520 / 0.480 / 0.000 | 19.6 / -77.4 |
| nav|nav=right | 958 | 0.1456 | 0.03509 | 0.04689 | 0.539 / 0.461 / 0.000 | 202.9 / 139.4 |
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

**EMA-15k - refcv7-50400** (sum 620.7 over 139 episodes)

* ep111: n 26, share 0.222, mean delta 5.3025, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.171, mean delta 4.0729, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep013: n 27, share 0.141, mean delta 3.2369, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep059: n 27, share 0.122, mean delta 2.7936, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.121, mean delta 2.8840, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep012: n 26, share 0.064, mean delta 1.5216, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
**EMA-15k - EMA-10.5k** (sum 130.3 over 139 episodes)

* ep078: n 26, share 0.911, mean delta 4.5645, sharp-window frac 0.00, curved frac 0.54, mean v0 7.1, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep059: n 27, share 0.575, mean delta 2.7736, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep137: n 22, share 0.301, mean delta 1.7836, sharp-window frac 0.36, curved frac 1.00, mean v0 4.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 22}
* ep069: n 27, share 0.153, mean delta 0.7371, sharp-window frac 0.04, curved frac 0.26, mean v0 7.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep024: n 27, share 0.143, mean delta 0.6919, sharp-window frac 0.00, curved frac 0.00, mean v0 5.1, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
* ep129: n 26, share 0.130, mean delta 0.6507, sharp-window frac 0.50, curved frac 0.58, mean v0 3.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**EMA-10.5k - refcv7-50400** (sum 512.5 over 139 episodes)

* ep111: n 26, share 0.266, mean delta 5.2389, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.226, mean delta 4.4648, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep013: n 27, share 0.174, mean delta 3.3096, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.149, mean delta 2.9453, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep012: n 26, share 0.102, mean delta 2.0024, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep036: n 26, share 0.086, mean delta 1.7029, sharp-window frac 0.27, curved frac 0.65, mean v0 12.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**EMA-15k - EMA-5k** (sum 250.7 over 139 episodes)

* ep080: n 26, share 0.363, mean delta 3.4983, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep111: n 26, share 0.317, mean delta 3.0542, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep122: n 26, share 0.157, mean delta 1.5115, sharp-window frac 0.38, curved frac 0.73, mean v0 8.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep024: n 27, share 0.151, mean delta 1.4009, sharp-window frac 0.00, curved frac 0.00, mean v0 5.1, nav {'nav=follow': 27, 'nav=left': 0, 'nav=right': 0}
* ep036: n 26, share 0.114, mean delta 1.1033, sharp-window frac 0.27, curved frac 0.65, mean v0 12.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share 0.107, mean delta 0.9908, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**EMA-5k - refcv7-50400** (sum 365.0 over 139 episodes)

* ep013: n 27, share 0.248, mean delta 3.3550, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.164, mean delta 2.3032, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.160, mean delta 2.2483, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share 0.133, mean delta 1.8028, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep000: n 27, share 0.121, mean delta 1.6405, sharp-window frac 0.33, curved frac 0.67, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep100: n 18, share 0.102, mean delta 2.0699, sharp-window frac 0.50, curved frac 1.00, mean v0 5.2, nav {'nav=follow': 0, 'nav=left': 18, 'nav=right': 0}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA-15k vs EMA-10.5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1375 / -0.0104 [-0.0339, 0.0093] | 69 / -0.1468 [-0.5052, 0.0968] |
| only new agrees | 162 / -0.4622 [-0.7951, -0.1847] | 13 / -3.5167 [-7.1376, -1.1298] |
| only earlier agrees | 224 / 0.4490 [0.1207, 0.9884] | 9 / 0.4413 [-2.7938, 3.3674] |
| neither agrees | 1763 / 0.0695 [-0.0582, 0.2240] | 212 / -0.1772 [-0.9002, 0.6128] |
**EMA-15k vs EMA-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1156 / -0.0008 [-0.0262, 0.0234] | 57 / -0.0780 [-0.3268, 0.1802] |
| only new agrees | 381 / -0.3800 [-0.5982, -0.1924] | 26 / -2.9103 [-4.0300, -2.0293] |
| only earlier agrees | 331 / 0.9132 [0.4056, 1.7082] | 22 / 8.0544 [1.6846, 17.8153] |
| neither agrees | 1658 / 0.0619 [-0.1016, 0.2339] | 199 / 0.3713 [-0.5203, 1.4154] |
**EMA-15k vs EMA-10.5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1354 / -0.0084 [-0.0307, 0.0103] | 75 / -0.1337 [-0.4953, 0.0856] |
| only new agrees | 190 / -0.4232 [-0.7208, -0.1697] | 14 / -3.5778 [-5.6002, -1.4182] |
| only earlier agrees | 227 / 0.4823 [0.1212, 1.0046] | 9 / 1.2491 [-1.8567, 3.8949] |
| neither agrees | 1753 / 0.0622 [-0.1062, 0.2893] | 205 / -0.5418 [-1.2670, 0.2333] |
**EMA-15k vs EMA-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1160 / -0.0106 [-0.0390, 0.0162] | 65 / -0.1763 [-0.5596, 0.1195] |
| only new agrees | 384 / -0.4841 [-0.7252, -0.2520] | 25 / -4.2952 [-6.1035, -2.8786] |
| only earlier agrees | 331 / 1.0782 [0.4677, 1.9679] | 20 / 11.0169 [2.8888, 19.4482] |
| neither agrees | 1652 / 0.0506 [-0.1014, 0.2085] | 194 / -0.1651 [-1.2898, 0.9127] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA-15k vs EMA-10.5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3477 | 0.1712 [-0.0401, 0.3918] | 0.1721 [-0.0134, 0.3537] | 0.0073 [-0.0932, 0.0985] |
| turn|straight <0.02 | 2072 | -0.0561 [-0.1608, 0.0230] | -0.0055 [-0.0696, 0.0632] | -0.0509 [-0.1490, 0.0208] |
| turn|gentle 0.02-0.06 | 674 | -0.1769 [-0.6635, 0.1963] | -0.0765 [-0.3387, 0.1613] | -0.1015 [-0.4176, 0.1258] |
| turn|turn 0.06-0.15 | 424 | 0.1911 [-0.1608, 0.5216] | 0.0570 [-0.3001, 0.3767] | 0.1344 [-0.0601, 0.3639] |
| turn|sharp >=0.15 | 300 | 2.4945 [0.7321, 4.4620] | 2.1150 [0.5477, 3.8549] | 0.4737 [-0.0725, 1.1278] |
| turn|undef (GT crawling) | 7 | -0.0195 [-0.5854, 0.5401] | -0.0435 [-0.7224, 0.6428] | 0.0240 [-0.6747, 0.6204] |
| speed|v0 <2 | 139 | 0.2680 [0.0426, 0.5587] | 0.3626 [-0.1288, 1.0876] | 0.1019 [-0.1677, 0.5133] |
| speed|v0 2-5 | 422 | 0.0637 [-1.1347, 1.1446] | 0.1757 [-0.7928, 1.0442] | -0.1119 [-0.6733, 0.3185] |
| speed|v0 5-10 | 1040 | 0.5104 [0.0600, 0.9608] | 0.4750 [0.1547, 0.8209] | 0.0354 [-0.2050, 0.2445] |
| speed|v0 10-15 | 952 | -0.0344 [-0.1250, 0.0560] | -0.0139 [-0.1278, 0.1037] | -0.0205 [-0.0828, 0.0490] |
| speed|v0 15-25 | 679 | 0.0520 [-0.0070, 0.1232] | -0.0215 [-0.0708, 0.0191] | 0.0735 [0.0192, 0.1474] SEP |
| speed|v0 >=25 | 245 | -0.0116 [-0.0362, 0.0077] | 0.0246 [-0.0128, 0.0928] | -0.0362 [-0.1087, 0.0081] |
**EMA-15k vs EMA-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3482 | 0.1712 [-0.0401, 0.3918] | 0.2206 [-0.0735, 0.5536] | -0.0512 [-0.2514, 0.1087] |
| turn|straight <0.02 | 2074 | -0.0561 [-0.1608, 0.0230] | -0.0575 [-0.1372, 0.0084] | 0.0031 [-0.0587, 0.0668] |
| turn|gentle 0.02-0.06 | 675 | -0.1769 [-0.6635, 0.1963] | -0.3451 [-0.8539, 0.0238] | 0.1682 [-0.0210, 0.3600] |
| turn|turn 0.06-0.15 | 425 | 0.1911 [-0.1608, 0.5216] | 0.1441 [-0.1844, 0.4765] | 0.0444 [-0.2420, 0.3111] |
| turn|sharp >=0.15 | 301 | 2.4945 [0.7321, 4.4620] | 3.4834 [0.8893, 6.7820] | -1.0570 [-3.0352, 0.4240] |
| turn|undef (GT crawling) | 7 | -0.0195 [-0.5854, 0.5401] | -0.0513 [-0.3960, 0.1713] | 0.1437 [-0.5258, 0.8763] |
| speed|v0 <2 | 144 | 0.2680 [0.0426, 0.5587] | 0.0184 [-0.5053, 0.4379] | 0.2662 [-0.1255, 0.7870] |
| speed|v0 2-5 | 422 | 0.0637 [-1.1347, 1.1446] | 0.7573 [-0.9143, 2.6853] | -0.6935 [-1.8999, 0.1089] |
| speed|v0 5-10 | 1040 | 0.5104 [0.0600, 0.9608] | 0.5257 [0.0316, 1.0835] | -0.0153 [-0.3448, 0.3028] |
| speed|v0 10-15 | 952 | -0.0344 [-0.1250, 0.0560] | -0.0878 [-0.1902, 0.0053] | 0.0534 [-0.0621, 0.1833] |
| speed|v0 15-25 | 679 | 0.0520 [-0.0070, 0.1232] | -0.0227 [-0.0751, 0.0173] | 0.0747 [0.0203, 0.1428] SEP |
| speed|v0 >=25 | 245 | -0.0116 [-0.0362, 0.0077] | 0.0277 [-0.0188, 0.1225] | -0.0393 [-0.1304, 0.0034] |
**EMA-15k vs EMA-10.5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
**EMA-15k vs EMA-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* EMA-15k vs EMA-10.5k|seed0: anchor flip rate 0.201; heading delta on flipped windows 0.1896 [-0.1797, 0.6337] (n 723) vs unflipped -0.0011 [-0.0247, 0.0239]; goal_dist_sel proxy new-earlier -0.0596 [-0.1125, -0.0092]; corr(window dHeading, window dGoalDist) 0.015
* EMA-15k vs EMA-5k|seed0: anchor flip rate 0.374; heading delta on flipped windows 0.1775 [-0.0731, 0.4297] (n 1345) vs unflipped 0.0094 [-0.0232, 0.0387]; goal_dist_sel proxy new-earlier -0.0092 [-0.0836, 0.0665]; corr(window dHeading, window dGoalDist) -0.010
* EMA-15k vs EMA-10.5k|seed1: anchor flip rate 0.203; heading delta on flipped windows 0.1944 [-0.2708, 0.7809] (n 730) vs unflipped -0.0054 [-0.0261, 0.0159]; goal_dist_sel proxy new-earlier -0.0629 [-0.1147, -0.0131]; corr(window dHeading, window dGoalDist) 0.027
* EMA-15k vs EMA-5k|seed1: anchor flip rate 0.374; heading delta on flipped windows 0.1773 [-0.1039, 0.4512] (n 1345) vs unflipped 0.0017 [-0.0281, 0.0303]; goal_dist_sel proxy new-earlier -0.0108 [-0.0869, 0.0650]; corr(window dHeading, window dGoalDist) -0.032
