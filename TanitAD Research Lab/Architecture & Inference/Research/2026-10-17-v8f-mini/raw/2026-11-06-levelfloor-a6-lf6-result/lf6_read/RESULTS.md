# LF6 read: the bounded-lift floor F_B at kappa 1.0 (A6), T1-kin, generated 2026-11-06T08:14:25+0100

**Joint reading (A6 section 5): THE-BOUND-REMOVES-THE-CLOSURE.** LFOB is CEILING-SHORT at both doses, or SHORT at one and HARMS at the other: the bound removes the closure; the floor's closure needs a lift larger than O's plan supports, and no measured estimator supports such a lift; the family is exhausted on CPU, and T2 and the A40 are reported to the PI

**LFB (the only admissible arms): delta 0.5: P-FAILS-M1-MOVED (counts as a result: True); delta 1.0: P-FAILS-M1-MOVED (counts as a result: True).** The heads-floor family has now read five doses (kappa 0.50, 0.75 and 1.0, then delta 0.5 and 1.0 at kappa 1): a counted DOMINANT names its kappa and delta and says that five were read. Adaptivity, disclosed: the form and the delta doses were chosen after looking at eval-141 rollouts and this read is on the same windows (A6 card).

Ceiling arms (every line CEILING): LFOB05 CEILING-SHORT; LFOB10 CEILING-SHORT.

CEILING: the level input is the human's logged route speed (exact), a privileged channel that is NOT admissible at inference. Not a driving result, not a capability claim, never a counted DOMINANT; it ranks levers (A6 card).

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

## LFOB05 (CEILING): **CEILING-SHORT**

| rule | s0 | s1 | holds |
|---|---|---|---|
| P_O share of M1 closed >= 0.5, separated-better | sep True, share 0.384 | sep True, share 0.329 | **False** |
| G_O ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 -0.1165, worse: cross-track MAE | ADE6 -0.1027, worse: cross-track MAE | **False** |
| F_O (beside) | 0.0205 vs 0.0362 | 0.0196 vs 0.0038 | **False** |
| ceiling of D (never counted): read A False | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | counted: False |

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.1165 [-0.1612, -0.0727] * | -0.1027 [-0.1465, -0.0591] * |
| speed MAE | LONGITUDINAL | -0.0779 [-0.1065, -0.0493] * | -0.0727 [-0.1002, -0.0463] * |
| along-track MAE | LONGITUDINAL | -0.1476 [-0.1953, -0.0997] * | -0.1366 [-0.1846, -0.0876] * |
| accel MAE | LONGITUDINAL | -0.0054 [-0.0322, +0.0222] | +0.0113 [-0.0167, +0.0450] |
| DK-VIOL | LONGITUDINAL | +0.0034 [+0.0000, +0.0108] | +0.0067 [+0.0000, +0.0155] |
| DK-HW | LONGITUDINAL | +0.0444 [-0.0346, +0.1194] | +0.0116 [-0.1159, +0.1392] |
| DK-TG | LONGITUDINAL | +0.0086 [-0.0030, +0.0213] | +0.0007 [-0.0168, +0.0183] |
| heading MAE | LATERAL | -0.1931 [-0.3231, -0.0697] * | -0.2071 [-0.3291, -0.1065] * |
| yaw-rate MAE | LATERAL | -1.9888 [-2.9016, -1.1733] * | -1.8040 [-2.5687, -1.1062] * |
| curvature MAE | LATERAL | -0.0240 [-0.0353, -0.0145] * | -0.0187 [-0.0301, -0.0096] * |
| cross-track MAE | LATERAL | +0.0189 [+0.0014, +0.0355] * | +0.0210 [+0.0066, +0.0361] * |
| L1 lateral-accel MAE | LATERAL | -0.0313 [-0.1039, +0.0434] | -0.0337 [-0.0817, +0.0111] |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0288 [-0.0499, -0.0141] * | -0.0204 [-0.0387, -0.0067] * |
| TAC-LAT | TACTICAL | +0.0154 [+0.0075, +0.0238] * | +0.0171 [+0.0075, +0.0271] * |
| TAC-LON | TACTICAL | +0.0392 [+0.0192, +0.0612] * | +0.0625 [+0.0413, +0.0854] * |
| S-POS@30 | STRATEGIC | -0.2508 [-0.3491, -0.1625] * | -0.3498 [-0.4952, -0.2221] * |
| S-POS@50 | STRATEGIC | -0.4182 [-0.6051, -0.2470] * | -0.5867 [-0.8660, -0.3484] * |
| S-POS@80 | STRATEGIC | -0.6110 [-0.9574, -0.2969] * | -0.9362 [-1.4101, -0.5311] * |
| M1 crawl share | crawl | -0.0205 [-0.0283, -0.0138] * | -0.0196 [-0.0284, -0.0120] * |

Attribution (beside): LFOB05 - RILFO100 (the same exact level at the unbounded lift)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.1498 [-0.2442, -0.0639] * | -0.1095 [-0.1911, -0.0291] * |
| speed MAE | LONGITUDINAL | -0.0144 [-0.0794, +0.0470] | -0.0166 [-0.0801, +0.0403] |
| along-track MAE | LONGITUDINAL | -0.0766 [-0.1672, +0.0087] | -0.0560 [-0.1323, +0.0218] |
| accel MAE | LONGITUDINAL | -0.0039 [-0.0701, +0.0584] | -0.0470 [-0.1427, +0.0304] |
| DK-VIOL | LONGITUDINAL | -0.0202 [-0.0445, -0.0033] * | -0.0101 [-0.0206, -0.0016] * |
| DK-HW | LONGITUDINAL | -0.1064 [-0.2837, +0.0461] | +0.0080 [-0.1517, +0.1662] |
| DK-TG | LONGITUDINAL | -0.0146 [-0.0379, +0.0047] | +0.0017 [-0.0170, +0.0222] |
| heading MAE | LATERAL | +0.1396 [-0.0483, +0.3344] | +0.0574 [-0.0766, +0.2185] |
| yaw-rate MAE | LATERAL | +2.3029 [+1.2678, +3.6156] * | +2.0270 [+1.1763, +2.9608] * |
| curvature MAE | LATERAL | +0.0181 [+0.0106, +0.0273] * | +0.0238 [+0.0145, +0.0342] * |
| cross-track MAE | LATERAL | -0.1043 [-0.1476, -0.0654] * | -0.0858 [-0.1180, -0.0541] * |
| L1 lateral-accel MAE | LATERAL | +0.0214 [-0.1223, +0.1639] | -0.0696 [-0.2337, +0.0581] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0075 [+0.0023, +0.0143] * | +0.0188 [+0.0066, +0.0361] * |
| TAC-LAT | TACTICAL | -0.0054 [-0.0167, +0.0050] | -0.0083 [-0.0221, +0.0046] |
| TAC-LON | TACTICAL | -0.0725 [-0.0980, -0.0487] * | -0.0604 [-0.0851, -0.0375] * |
| S-POS@30 | STRATEGIC | +0.1757 [+0.0660, +0.2964] * | +0.1512 [+0.0547, +0.2612] * |
| S-POS@50 | STRATEGIC | +0.2545 [+0.0719, +0.4412] * | +0.1998 [+0.0393, +0.3843] * |
| S-POS@80 | STRATEGIC | +0.3978 [+0.1361, +0.6647] * | +0.2756 [+0.0139, +0.5684] * |
| M1 crawl share | crawl | +0.0193 [+0.0116, +0.0286] * | +0.0245 [+0.0155, +0.0359] * |

## LFOB10 (CEILING): **CEILING-SHORT**

| rule | s0 | s1 | holds |
|---|---|---|---|
| P_O share of M1 closed >= 0.5, separated-better | sep True, share 0.544 | sep True, share 0.488 | **False** |
| G_O ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 -0.1163, worse: DK-VIOL, cross-track MAE | ADE6 -0.0930, worse: accel MAE, cross-track MAE | **False** |
| F_O (beside) | 0.0290 vs 0.0362 | 0.0291 vs 0.0038 | **False** |
| ceiling of D (never counted): read A False | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | counted: False |

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.1163 [-0.1845, -0.0511] * | -0.0930 [-0.1647, -0.0246] * |
| speed MAE | LONGITUDINAL | -0.1035 [-0.1465, -0.0588] * | -0.0808 [-0.1221, -0.0384] * |
| along-track MAE | LONGITUDINAL | -0.1739 [-0.2470, -0.1042] * | -0.1524 [-0.2252, -0.0793] * |
| accel MAE | LONGITUDINAL | +0.0083 [-0.0275, +0.0445] | +0.0458 [+0.0147, +0.0805] * |
| DK-VIOL | LONGITUDINAL | +0.0134 [+0.0033, +0.0272] * | +0.0050 [+0.0000, +0.0114] |
| DK-HW | LONGITUDINAL | +0.0958 [-0.0293, +0.2221] | -0.0480 [-0.2039, +0.1068] |
| DK-TG | LONGITUDINAL | +0.0142 [-0.0035, +0.0339] | -0.0097 [-0.0311, +0.0107] |
| heading MAE | LATERAL | -0.1609 [-0.3125, -0.0116] * | -0.1935 [-0.3228, -0.0755] * |
| yaw-rate MAE | LATERAL | -2.2786 [-3.1734, -1.4643] * | -2.2799 [-3.3400, -1.4091] * |
| curvature MAE | LATERAL | -0.0287 [-0.0418, -0.0179] * | -0.0298 [-0.0433, -0.0180] * |
| cross-track MAE | LATERAL | +0.0488 [+0.0223, +0.0755] * | +0.0582 [+0.0333, +0.0851] * |
| L1 lateral-accel MAE | LATERAL | +0.0031 [-0.0842, +0.0912] | +0.0198 [-0.0427, +0.0924] |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0320 [-0.0547, -0.0152] * | -0.0317 [-0.0546, -0.0127] * |
| TAC-LAT | TACTICAL | +0.0113 [-0.0004, +0.0234] | +0.0175 [+0.0033, +0.0325] * |
| TAC-LON | TACTICAL | +0.0709 [+0.0450, +0.0983] * | +0.0863 [+0.0600, +0.1138] * |
| S-POS@30 | STRATEGIC | -0.2459 [-0.3750, -0.1241] * | -0.3420 [-0.4818, -0.2188] * |
| S-POS@50 | STRATEGIC | -0.3917 [-0.6240, -0.1738] * | -0.5532 [-0.8341, -0.3282] * |
| S-POS@80 | STRATEGIC | -0.5908 [-0.9890, -0.2330] * | -0.8610 [-1.3421, -0.4795] * |
| M1 crawl share | crawl | -0.0290 [-0.0399, -0.0194] * | -0.0291 [-0.0414, -0.0185] * |

Attribution (beside): LFOB10 - RILFO100 (the same exact level at the unbounded lift)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.1496 [-0.2379, -0.0719] * | -0.0999 [-0.1637, -0.0400] * |
| speed MAE | LONGITUDINAL | -0.0400 [-0.0995, +0.0134] | -0.0247 [-0.0766, +0.0193] |
| along-track MAE | LONGITUDINAL | -0.1028 [-0.1834, -0.0310] * | -0.0718 [-0.1322, -0.0136] * |
| accel MAE | LONGITUDINAL | +0.0097 [-0.0565, +0.0693] | -0.0125 [-0.0857, +0.0438] |
| DK-VIOL | LONGITUDINAL | -0.0101 [-0.0271, +0.0018] | -0.0118 [-0.0215, -0.0035] * |
| DK-HW | LONGITUDINAL | -0.0550 [-0.1664, +0.0415] | -0.0517 [-0.1737, +0.0677] |
| DK-TG | LONGITUDINAL | -0.0091 [-0.0244, +0.0032] | -0.0088 [-0.0238, +0.0045] |
| heading MAE | LATERAL | +0.1718 [+0.0347, +0.3320] * | +0.0710 [-0.0509, +0.2341] |
| yaw-rate MAE | LATERAL | +2.0131 [+1.1274, +3.1760] * | +1.5511 [+0.8928, +2.3272] * |
| curvature MAE | LATERAL | +0.0133 [+0.0075, +0.0212] * | +0.0128 [+0.0070, +0.0199] * |
| cross-track MAE | LATERAL | -0.0743 [-0.1124, -0.0412] * | -0.0486 [-0.0710, -0.0282] * |
| L1 lateral-accel MAE | LATERAL | +0.0558 [-0.0751, +0.1805] | -0.0160 [-0.1516, +0.0874] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0039 [+0.0012, +0.0071] * | +0.0074 [+0.0026, +0.0133] * |
| TAC-LAT | TACTICAL | -0.0096 [-0.0183, -0.0008] * | -0.0079 [-0.0188, +0.0017] |
| TAC-LON | TACTICAL | -0.0409 [-0.0625, -0.0212] * | -0.0367 [-0.0559, -0.0200] * |
| S-POS@30 | STRATEGIC | +0.1805 [+0.0851, +0.2778] * | +0.1589 [+0.0699, +0.2617] * |
| S-POS@50 | STRATEGIC | +0.2809 [+0.1252, +0.4462] * | +0.2333 [+0.0839, +0.4053] * |
| S-POS@80 | STRATEGIC | +0.4180 [+0.1762, +0.6878] * | +0.3508 [+0.1210, +0.6238] * |
| M1 crawl share | crawl | +0.0103 [+0.0052, +0.0166] * | +0.0150 [+0.0087, +0.0228] * |

## LFB05 (admissible, vision-only): **P-FAILS-M1-MOVED** (the A3 rules)

Next (A3 wording; the joint reading above governs where they differ): the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data).

P False, G False, F False, D (read A, four families) False, LEAK-BORNE False.

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.1588 [+0.0748, +0.2500] * | +0.1336 [+0.0561, +0.2177] * |
| speed MAE | LONGITUDINAL | +0.0199 [-0.0335, +0.0734] | +0.0113 [-0.0283, +0.0536] |
| along-track MAE | LONGITUDINAL | +0.1568 [+0.0687, +0.2537] * | +0.1316 [+0.0499, +0.2192] * |
| accel MAE | LONGITUDINAL | -0.1095 [-0.2058, -0.0278] * | -0.0522 [-0.1089, -0.0062] * |
| DK-VIOL | LONGITUDINAL | +0.0134 [+0.0000, +0.0324] | +0.0118 [+0.0017, +0.0253] * |
| DK-HW | LONGITUDINAL | +0.2344 [+0.0611, +0.4335] * | +0.0455 [-0.0758, +0.1675] |
| DK-TG | LONGITUDINAL | +0.0338 [-0.0283, +0.0971] | -0.0205 [-0.1187, +0.0433] |
| heading MAE | LATERAL | -0.8276 [-1.3453, -0.3809] * | -0.7839 [-1.2575, -0.3886] * |
| yaw-rate MAE | LATERAL | -6.3579 [-9.7198, -3.8697] * | -5.6209 [-8.6419, -3.2825] * |
| curvature MAE | LATERAL | -0.0536 [-0.0771, -0.0344] * | -0.0513 [-0.0790, -0.0289] * |
| cross-track MAE | LATERAL | +0.0125 [-0.0147, +0.0418] | +0.0114 [-0.0116, +0.0352] |
| L1 lateral-accel MAE | LATERAL | -0.4970 [-0.9405, -0.1866] * | -0.3656 [-0.6338, -0.1720] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0322 [-0.0549, -0.0162] * | -0.0288 [-0.0506, -0.0100] * |
| TAC-LAT | TACTICAL | +0.0296 [+0.0167, +0.0425] * | +0.0454 [+0.0292, +0.0646] * |
| TAC-LON | TACTICAL | -0.0188 [-0.0525, +0.0125] | +0.0125 [-0.0159, +0.0413] |
| S-POS@30 | STRATEGIC | -0.4152 [-0.6408, -0.2096] * | -0.5226 [-0.7591, -0.3047] * |
| S-POS@50 | STRATEGIC | -0.7391 [-1.1320, -0.3768] * | -0.8466 [-1.2752, -0.4792] * |
| S-POS@80 | STRATEGIC | -1.1065 [-1.7407, -0.5403] * | -1.3609 [-2.0712, -0.7451] * |
| M1 crawl share | crawl | -0.0204 [-0.0284, -0.0130] * | -0.0195 [-0.0287, -0.0114] * |

Attribution (beside): LFB05 - RILF100 (the unbounded heads floor, A5)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.0970 [-1.3235, -0.8870] * | -0.9555 [-1.1639, -0.7717] * |
| speed MAE | LONGITUDINAL | -0.4428 [-0.5619, -0.3345] * | -0.3611 [-0.4556, -0.2745] * |
| along-track MAE | LONGITUDINAL | -0.9701 [-1.1765, -0.7742] * | -0.8598 [-1.0661, -0.6751] * |
| accel MAE | LONGITUDINAL | -0.1511 [-0.2545, -0.0500] * | -0.0695 [-0.1121, -0.0285] * |
| DK-VIOL | LONGITUDINAL | -0.0319 [-0.0601, -0.0096] * | -0.0353 [-0.0557, -0.0196] * |
| DK-HW | LONGITUDINAL | -0.7718 [-1.2518, -0.3665] * | -0.7999 [-1.3367, -0.3708] * |
| DK-TG | LONGITUDINAL | -0.1520 [-0.2566, -0.0698] * | -0.1200 [-0.1882, -0.0616] * |
| heading MAE | LATERAL | +0.0574 [-0.1806, +0.3061] | +0.2501 [-0.0316, +0.6052] |
| yaw-rate MAE | LATERAL | +2.0507 [+1.0709, +3.1148] * | +2.9108 [+1.6548, +4.2822] * |
| curvature MAE | LATERAL | +0.0283 [+0.0157, +0.0419] * | +0.0368 [+0.0214, +0.0547] * |
| cross-track MAE | LATERAL | -0.2817 [-0.3836, -0.1919] * | -0.2311 [-0.3074, -0.1647] * |
| L1 lateral-accel MAE | LATERAL | -0.1909 [-0.4009, -0.0071] * | -0.0784 [-0.2098, +0.0311] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0086 [+0.0030, +0.0163] * | +0.0153 [+0.0061, +0.0268] * |
| TAC-LAT | TACTICAL | -0.0117 [-0.0263, +0.0033] | -0.0121 [-0.0246, +0.0013] |
| TAC-LON | TACTICAL | +0.0217 [-0.0046, +0.0500] | +0.0296 [+0.0033, +0.0579] * |
| S-POS@30 | STRATEGIC | +0.0501 [-0.0754, +0.1811] | +0.1701 [+0.0333, +0.3144] * |
| S-POS@50 | STRATEGIC | +0.0672 [-0.1606, +0.3165] | +0.2808 [+0.0416, +0.5279] * |
| S-POS@80 | STRATEGIC | +0.2679 [-0.1120, +0.6817] | +0.3626 [-0.0436, +0.7886] |
| M1 crawl share | crawl | +0.0281 [+0.0174, +0.0402] * | +0.0272 [+0.0162, +0.0402] * |

## LFB10 (admissible, vision-only): **P-FAILS-M1-MOVED** (the A3 rules)

Next (A3 wording; the joint reading above governs where they differ): the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data).

P False, G False, F False, D (read A, four families) False, LEAK-BORNE False.

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.4344 [+0.2995, +0.5852] * | +0.3956 [+0.2664, +0.5345] * |
| speed MAE | LONGITUDINAL | +0.1141 [+0.0466, +0.1910] * | +0.1065 [+0.0475, +0.1720] * |
| along-track MAE | LONGITUDINAL | +0.4202 [+0.2757, +0.5810] * | +0.3641 [+0.2258, +0.5081] * |
| accel MAE | LONGITUDINAL | -0.0558 [-0.1541, +0.0321] | +0.0170 [-0.0594, +0.0981] |
| DK-VIOL | LONGITUDINAL | +0.0336 [+0.0135, +0.0585] * | +0.0218 [+0.0080, +0.0405] * |
| DK-HW | LONGITUDINAL | +0.5950 [+0.2670, +1.0013] * | +0.3265 [+0.1120, +0.5659] * |
| DK-TG | LONGITUDINAL | +0.0916 [+0.0162, +0.1842] * | +0.0309 [-0.0792, +0.1214] |
| heading MAE | LATERAL | -0.7850 [-1.3240, -0.2681] * | -0.9076 [-1.5526, -0.3884] * |
| yaw-rate MAE | LATERAL | -6.9047 [-10.4613, -4.1799] * | -6.6273 [-9.9870, -4.0504] * |
| curvature MAE | LATERAL | -0.0663 [-0.0952, -0.0421] * | -0.0694 [-0.1027, -0.0417] * |
| cross-track MAE | LATERAL | +0.0601 [+0.0192, +0.1071] * | +0.0723 [+0.0344, +0.1154] * |
| L1 lateral-accel MAE | LATERAL | -0.3675 [-0.8133, -0.0384] * | -0.2653 [-0.5396, -0.0517] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0363 [-0.0625, -0.0182] * | -0.0352 [-0.0599, -0.0141] * |
| TAC-LAT | TACTICAL | +0.0346 [+0.0196, +0.0500] * | +0.0471 [+0.0296, +0.0667] * |
| TAC-LON | TACTICAL | -0.0258 [-0.0675, +0.0138] | +0.0088 [-0.0275, +0.0425] |
| S-POS@30 | STRATEGIC | -0.3768 [-0.6202, -0.1396] * | -0.5938 [-0.8672, -0.3486] * |
| S-POS@50 | STRATEGIC | -0.6547 [-1.0815, -0.2466] * | -0.9589 [-1.4392, -0.5435] * |
| S-POS@80 | STRATEGIC | -1.0063 [-1.7264, -0.3380] * | -1.5136 [-2.3458, -0.8107] * |
| M1 crawl share | crawl | -0.0304 [-0.0418, -0.0197] * | -0.0293 [-0.0416, -0.0189] * |

Attribution (beside): LFB10 - RILF100 (the unbounded heads floor, A5)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.8214 [-1.0040, -0.6541] * | -0.6935 [-0.8504, -0.5541] * |
| speed MAE | LONGITUDINAL | -0.3485 [-0.4481, -0.2614] * | -0.2660 [-0.3401, -0.2008] * |
| along-track MAE | LONGITUDINAL | -0.7068 [-0.8693, -0.5563] * | -0.6272 [-0.7804, -0.4903] * |
| accel MAE | LONGITUDINAL | -0.0973 [-0.1946, -0.0188] * | -0.0003 [-0.0653, +0.0711] |
| DK-VIOL | LONGITUDINAL | -0.0118 [-0.0337, +0.0058] | -0.0252 [-0.0434, -0.0114] * |
| DK-HW | LONGITUDINAL | -0.4091 [-0.7378, -0.1497] * | -0.5187 [-0.9661, -0.1602] * |
| DK-TG | LONGITUDINAL | -0.0939 [-0.1806, -0.0278] * | -0.0685 [-0.1164, -0.0258] * |
| heading MAE | LATERAL | +0.1000 [-0.0843, +0.2768] | +0.1263 [-0.0620, +0.3629] |
| yaw-rate MAE | LATERAL | +1.5039 [+0.7534, +2.3990] * | +1.9044 [+0.9258, +3.1139] * |
| curvature MAE | LATERAL | +0.0155 [+0.0080, +0.0244] * | +0.0187 [+0.0096, +0.0304] * |
| cross-track MAE | LATERAL | -0.2340 [-0.3225, -0.1584] * | -0.1702 [-0.2298, -0.1174] * |
| L1 lateral-accel MAE | LATERAL | -0.0613 [-0.2671, +0.1067] | +0.0219 [-0.1195, +0.1710] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0045 [+0.0008, +0.0086] * | +0.0089 [+0.0028, +0.0168] * |
| TAC-LAT | TACTICAL | -0.0067 [-0.0183, +0.0054] | -0.0104 [-0.0200, -0.0013] * |
| TAC-LON | TACTICAL | +0.0146 [-0.0038, +0.0338] | +0.0258 [+0.0071, +0.0450] * |
| S-POS@30 | STRATEGIC | +0.0885 [-0.0290, +0.2106] | +0.0989 [-0.0089, +0.2141] |
| S-POS@50 | STRATEGIC | +0.1516 [-0.0759, +0.3923] | +0.1685 [-0.0116, +0.3451] |
| S-POS@80 | STRATEGIC | +0.3681 [-0.0415, +0.7922] | +0.2099 [-0.0686, +0.5015] |
| M1 crawl share | crawl | +0.0175 [+0.0097, +0.0268] * | +0.0173 [+0.0095, +0.0268] * |

## Instruments (beside, never deciding)

| rollout | bind overall | bind in M1 ticks (n) | cap overall | mean lift where bound, m/s | share of bound ticks where the delta term binds | max (s' - s - delta t), m | v_hat vs logged: MAE m/s, relative MAE, share v_hat > logged (n) |
|---|---|---|---|---|---|---|---|
| RILFB10_s0 | 0.459 | 0.687 (182) | 0.125 | 0.553 | 0.842 | -9.55e-04 | 2.593, 1.444, 0.516 (6075) |
| RILFB10_s1 | 0.433 | 0.625 (184) | 0.096 | 0.543 | 0.858 | -6.88e-03 | 2.606, 1.248, 0.495 (6079) |
| RILFB10C_s0 | 0.461 | 0.709 (182) | 0.125 | 0.555 | 0.842 | -4.70e-02 | 2.587, 1.197, 0.516 (6073) |
| RILFB10C_s1 | 0.435 | 0.656 (183) | 0.096 | 0.544 | 0.858 | -4.51e-02 | 2.603, 1.073, 0.495 (6082) |
| RILFB05_s0 | 0.463 | 0.713 (216) | 0.124 | 0.336 | 0.947 | 8.88e-16 | 2.593, 1.441, 0.516 (6075) |
| RILFB05_s1 | 0.439 | 0.649 (202) | 0.096 | 0.326 | 0.961 | 8.88e-16 | 2.608, 1.217, 0.496 (6079) |
| RILFB05C_s0 | 0.467 | 0.731 (216) | 0.124 | 0.336 | 0.947 | 4.44e-16 | 2.588, 1.201, 0.515 (6073) |
| RILFB05C_s1 | 0.442 | 0.673 (202) | 0.097 | 0.328 | 0.960 | 1.78e-15 | 2.606, 1.080, 0.496 (6082) |
| RILFOB10_s0 | 0.453 | 0.731 (175) | 0.092 | 0.309 | 0.617 | -7.29e-02 | 0.000, 0.000, 0.496 (6108) |
| RILFOB10_s1 | 0.442 | 0.760 (175) | 0.084 | 0.295 | 0.624 | -1.48e-02 | 0.000, 0.000, 0.496 (6108) |
| RILFOB10C_s0 | 0.456 | 0.806 (175) | 0.092 | 0.310 | 0.618 | -4.94e-02 | 0.000, 0.000, 0.496 (6108) |
| RILFOB10C_s1 | 0.444 | 0.789 (175) | 0.084 | 0.297 | 0.624 | -7.28e-03 | 0.000, 0.000, 0.496 (6108) |
| RILFOB05_s0 | 0.449 | 0.740 (208) | 0.092 | 0.201 | 0.831 | -2.78e-17 | 0.000, 0.000, 0.496 (6108) |
| RILFOB05_s1 | 0.441 | 0.751 (201) | 0.084 | 0.194 | 0.842 | 1.11e-16 | 0.000, 0.000, 0.496 (6108) |
| RILFOB05C_s0 | 0.452 | 0.803 (208) | 0.092 | 0.202 | 0.831 | 4.44e-16 | 0.000, 0.000, 0.496 (6108) |
| RILFOB05C_s1 | 0.444 | 0.796 (201) | 0.084 | 0.196 | 0.842 | 4.44e-16 | 0.000, 0.000, 0.496 (6108) |
| LFB10_s0 | 0.460 | 0.700 (180) | 0.125 | 0.552 | 0.845 | -6.13e-02 | 2.592, 1.220, 0.518 (6080) |
| LFB10_s1 | 0.433 | 0.625 (184) | 0.097 | 0.543 | 0.857 | -4.14e-04 | 2.603, 1.094, 0.498 (6087) |
| LFB10C_s0 | 0.464 | 0.739 (180) | 0.125 | 0.553 | 0.846 | 1.11e-16 | 2.592, 1.220, 0.518 (6080) |
| LFB10C_s1 | 0.435 | 0.661 (183) | 0.097 | 0.544 | 0.856 | -4.51e-02 | 2.603, 1.094, 0.498 (6087) |
| LFB05_s0 | 0.464 | 0.735 (215) | 0.125 | 0.334 | 0.948 | 8.88e-16 | 2.592, 1.220, 0.518 (6080) |
| LFB05_s1 | 0.440 | 0.658 (202) | 0.097 | 0.326 | 0.961 | 8.88e-16 | 2.603, 1.094, 0.498 (6087) |
| LFB05C_s0 | 0.470 | 0.767 (215) | 0.125 | 0.335 | 0.949 | 4.44e-16 | 2.592, 1.220, 0.518 (6080) |
| LFB05C_s1 | 0.443 | 0.678 (202) | 0.097 | 0.327 | 0.960 | 1.78e-15 | 2.603, 1.094, 0.498 (6087) |

## Controls

| control | PASS |
|---|---|
| reused_arrays | True |
| patched_tool_pin | True |
| rollout_RILFB10_s0 | True |
| rollout_RILFB10_s1 | True |
| rollout_RILFB10C_s0 | True |
| rollout_RILFB10C_s1 | True |
| rollout_RILFB05_s0 | True |
| rollout_RILFB05_s1 | True |
| rollout_RILFB05C_s0 | True |
| rollout_RILFB05C_s1 | True |
| rollout_RILFOB10_s0 | True |
| rollout_RILFOB10_s1 | True |
| rollout_RILFOB10C_s0 | True |
| rollout_RILFOB10C_s1 | True |
| rollout_RILFOB05_s0 | True |
| rollout_RILFOB05_s1 | True |
| rollout_RILFOB05C_s0 | True |
| rollout_RILFOB05C_s1 | True |
| rollout_LFB10_s0 | True |
| rollout_LFB10_s1 | True |
| rollout_LFB10C_s0 | True |
| rollout_LFB10C_s1 | True |
| rollout_LFB05_s0 | True |
| rollout_LFB05_s1 | True |
| rollout_LFB05C_s0 | True |
| rollout_LFB05C_s1 | True |
| one_checkpoint_per_seed | True |
| LF6_PRECHECK | True |
| card_a6 | True |
| cl_card_rules | True |
| lfo_read_py | True |
| lfo_read_json | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: O is C0's own checkpoint in every contrast with C0 / RIC0, so O's training variance is absent from it; the heads come from one LLR run per seed (not measured); F_O / F read against the C0REP floor
* inference: every A6 arm is deterministic (no noise arm): the interval answers another draw of EPISODES only
