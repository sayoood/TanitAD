# R7 matched-phase contrasts at N = 30000 (S2/S6 heading-error slope; PREREG_R7_MATCHED_PHASE_TURN.md)

* generated 2026-10-18T18:26:46; trees: `R7` = C:/Users/Admin/refcv8_eval_archive/battery/r7step30000_unc; `EMA8` = C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc
* estimator: episode-cluster bootstrap over the eval episodes, models FIXED (not another training run; not another inference run); B 2000; tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* CONTROL vs `heading_drift.shrinkage` on the same arms (must be < 1e-9): S2|EMA8: 0.00e+00 (ok); S6|EMA8: 0.00e+00 (ok)
* baseline identity (embedded refcv7-50,400 arm, max abs diff vs the R7 tree; must be 0): S2|EMA8|s0 0.0e+00, S2|EMA8|s1 0.0e+00, S6|EMA8|s0 0.0e+00, S6|EMA8|s1 0.0e+00

## S2 ([0.5, 1.0, 1.5, 2.0] s), 4754 windows / 139 episodes


### S2 -- EMA8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1577 / 1577 | -0.0887 | -0.1254 | -0.1167 | -0.0366 [-0.0567, -0.0163] SEP | +0.0087 [-0.0158, +0.0315] | -0.0280 [-0.0542, -0.0040] SEP | 1.31 [0.53, 4.95] |
| seed 1 | 1577 / 1577 | -0.0808 | -0.1331 | -0.1232 | -0.0523 [-0.0782, -0.0268] SEP | +0.0099 [-0.0246, +0.0438] | -0.0424 [-0.0722, -0.0156] SEP | 1.23 [0.58, 3.26] |
| pooled (2-seed mean heading error) | 1577 / 1577 | -0.0847 | -0.1292 | -0.1199 | -0.0445 [-0.0653, -0.0227] SEP | +0.0093 [-0.0198, +0.0372] | -0.0352 [-0.0623, -0.0103] SEP | 1.26 [0.61, 3.66] |

### S2 -- EMA8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1596 / 4757 | -0.1503 | -0.1769 | -0.1766 | -0.0266 [-0.0459, -0.0073] SEP | +0.0003 [-0.0206, +0.0207] | -0.0263 [-0.0493, -0.0048] SEP | 1.01 [0.33, 3.30] |
| seed 1 | 1596 / 4757 | -0.1432 | -0.1840 | -0.1819 | -0.0408 [-0.0643, -0.0175] SEP | +0.0021 [-0.0268, +0.0315] | -0.0387 [-0.0643, -0.0151] SEP | 1.05 [0.47, 2.60] |
| pooled (2-seed mean heading error) | 1596 / 4757 | -0.1468 | -0.1805 | -0.1793 | -0.0337 [-0.0539, -0.0135] SEP | +0.0012 [-0.0230, +0.0257] | -0.0325 [-0.0560, -0.0104] SEP | 1.04 [0.46, 2.82] |

### S2 -- EMA8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 363 / 363 | -0.0968 | -0.1385 | -0.1301 | -0.0417 [-0.0661, -0.0187] SEP | +0.0084 [-0.0189, +0.0328] | -0.0333 [-0.0622, -0.0066] SEP | 1.25 [0.60, 3.84] |
| seed 1 | 363 / 363 | -0.0874 | -0.1483 | -0.1372 | -0.0609 [-0.0912, -0.0318] SEP | +0.0111 [-0.0271, +0.0484] | -0.0498 [-0.0856, -0.0191] SEP | 1.22 [0.61, 2.97] |
| pooled (2-seed mean heading error) | 363 / 363 | -0.0921 | -0.1434 | -0.1337 | -0.0513 [-0.0760, -0.0269] SEP | +0.0098 [-0.0220, +0.0394] | -0.0415 [-0.0734, -0.0132] SEP | 1.23 [0.63, 3.17] |

### S2 -- EMA8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 366 / 1094 | -0.1084 | -0.1406 | -0.1415 | -0.0323 [-0.0554, -0.0097] SEP | -0.0009 [-0.0252, +0.0215] | -0.0331 [-0.0594, -0.0075] SEP | 0.97 [0.38, 2.66] |
| seed 1 | 366 / 1094 | -0.0998 | -0.1503 | -0.1477 | -0.0505 [-0.0789, -0.0222] SEP | +0.0026 [-0.0317, +0.0363] | -0.0479 [-0.0794, -0.0200] SEP | 1.05 [0.51, 2.42] |
| pooled (2-seed mean heading error) | 366 / 1094 | -0.1041 | -0.1455 | -0.1446 | -0.0414 [-0.0650, -0.0173] SEP | +0.0009 [-0.0279, +0.0280] | -0.0405 [-0.0693, -0.0143] SEP | 1.02 [0.49, 2.42] |

### S2 -- EMA8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 627 | -0.0906 | -0.1132 | -0.1131 | -0.0226 [-0.0627, +0.0137] | +0.0001 [-0.0306, +0.0347] | -0.0225 [-0.0605, +0.0222] | 1.00 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 627 / 627 | -0.0945 | -0.1209 | -0.1166 | -0.0264 [-0.0632, +0.0080] | +0.0043 [-0.0282, +0.0416] | -0.0221 [-0.0577, +0.0229] | 1.19 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 627 / 627 | -0.0925 | -0.1170 | -0.1148 | -0.0245 [-0.0621, +0.0105] | +0.0022 [-0.0291, +0.0376] | -0.0223 [-0.0586, +0.0222] | 1.10 (CI undefined: the CI of the denominator (gap) includes 0) |

### S2 -- EMA8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 1881 | -0.0979 | -0.1111 | -0.1211 | -0.0132 [-0.0509, +0.0218] | -0.0101 [-0.0370, +0.0211] | -0.0233 [-0.0585, +0.0171] | 0.57 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 627 / 1881 | -0.1007 | -0.1183 | -0.1231 | -0.0177 [-0.0521, +0.0161] | -0.0047 [-0.0330, +0.0276] | -0.0224 [-0.0558, +0.0189] | 0.79 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 627 / 1881 | -0.0993 | -0.1147 | -0.1221 | -0.0154 [-0.0515, +0.0188] | -0.0074 [-0.0347, +0.0244] | -0.0228 [-0.0572, +0.0181] | 0.68 (CI undefined: the CI of the denominator (gap) includes 0) |

### S2 -- EMA8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 164 | -0.0901 | -0.1151 | -0.1158 | -0.0251 [-0.0684, +0.0133] | -0.0006 [-0.0299, +0.0353] | -0.0257 [-0.0680, +0.0198] | 0.98 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 164 / 164 | -0.0924 | -0.1238 | -0.1197 | -0.0314 [-0.0705, +0.0047] | +0.0042 [-0.0245, +0.0407] | -0.0273 [-0.0651, +0.0181] | 1.15 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 164 / 164 | -0.0912 | -0.1195 | -0.1177 | -0.0282 [-0.0696, +0.0083] | +0.0018 [-0.0271, +0.0377] | -0.0265 [-0.0655, +0.0189] | 1.07 (CI undefined: the CI of the denominator (gap) includes 0) |

### S2 -- EMA8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 492 | -0.0947 | -0.1089 | -0.1205 | -0.0142 [-0.0561, +0.0222] | -0.0116 [-0.0369, +0.0200] | -0.0259 [-0.0659, +0.0179] | 0.55 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 164 / 492 | -0.0963 | -0.1172 | -0.1229 | -0.0209 [-0.0598, +0.0155] | -0.0057 [-0.0305, +0.0270] | -0.0266 [-0.0623, +0.0168] | 0.79 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 164 / 492 | -0.0955 | -0.1131 | -0.1217 | -0.0176 [-0.0578, +0.0180] | -0.0087 [-0.0334, +0.0225] | -0.0262 [-0.0630, +0.0165] | 0.67 (CI undefined: the CI of the denominator (gap) includes 0) |

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s), 3668 windows / 139 episodes


### S6 -- EMA8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1326 / 1326 | -0.2466 | -0.2755 | -0.1986 | -0.0289 [-0.0671, +0.0103] | +0.0769 [+0.0360, +0.1239] SEP | +0.0480 [+0.0091, +0.0933] SEP | -0.60 [-4.11, 0.25] |
| seed 1 | 1322 / 1322 | -0.2485 | -0.2800 | -0.2046 | -0.0315 [-0.0678, +0.0032] | +0.0754 [+0.0291, +0.1303] SEP | +0.0439 [+0.0019, +0.0892] SEP | -0.72 [-5.45, 0.22] |
| pooled (2-seed mean heading error) | 1321 / 1321 | -0.2477 | -0.2758 | -0.2011 | -0.0281 [-0.0616, +0.0055] | +0.0748 [+0.0330, +0.1242] SEP | +0.0467 [+0.0070, +0.0909] SEP | -0.60 [-4.06, 0.18] |

### S6 -- EMA8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1418 / 6852 | -0.1732 | -0.2000 | -0.1389 | -0.0268 [-0.0683, +0.0205] | +0.0611 [+0.0238, +0.1007] SEP | +0.0343 [-0.0096, +0.0841] | -0.78 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 1417 / 6849 | -0.1838 | -0.2046 | -0.1304 | -0.0208 [-0.0759, +0.0545] | +0.0741 [+0.0222, +0.1320] SEP | +0.0534 [-0.0179, +0.1602] | -0.39 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 1417 / 6841 | -0.1784 | -0.2008 | -0.1342 | -0.0224 [-0.0690, +0.0370] | +0.0667 [+0.0254, +0.1131] SEP | +0.0443 [-0.0113, +0.1192] | -0.51 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 289 / 289 | -0.2622 | -0.3054 | -0.2221 | -0.0432 [-0.0929, +0.0062] | +0.0833 [+0.0312, +0.1418] SEP | +0.0401 [-0.0115, +0.0954] | -1.08 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 287 / 287 | -0.2651 | -0.3115 | -0.2331 | -0.0464 [-0.0919, -0.0033] SEP | +0.0783 [+0.0229, +0.1478] SEP | +0.0319 [-0.0204, +0.0832] | -1.45 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 287 / 287 | -0.2638 | -0.3059 | -0.2272 | -0.0421 [-0.0858, +0.0003] | +0.0787 [+0.0261, +0.1406] SEP | +0.0366 [-0.0140, +0.0899] | -1.15 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 307 / 1477 | -0.1953 | -0.2542 | -0.1824 | -0.0589 [-0.1073, -0.0141] SEP | +0.0718 [+0.0231, +0.1257] SEP | +0.0129 [-0.0386, +0.0663] | -4.57 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 306 / 1473 | -0.1915 | -0.2607 | -0.1927 | -0.0692 [-0.1166, -0.0268] SEP | +0.0680 [+0.0100, +0.1360] SEP | -0.0012 [-0.0581, +0.0505] | 58.66 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 306 / 1471 | -0.1934 | -0.2556 | -0.1871 | -0.0622 [-0.1070, -0.0214] SEP | +0.0685 [+0.0166, +0.1277] SEP | +0.0063 [-0.0474, +0.0577] | -9.85 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 496 / 496 | -0.2694 | -0.2984 | -0.2063 | -0.0290 [-0.1007, +0.0407] | +0.0921 [+0.0211, +0.1811] SEP | +0.0631 [-0.0120, +0.1538] | -0.46 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 494 / 494 | -0.2780 | -0.2959 | -0.2126 | -0.0179 [-0.0790, +0.0382] | +0.0833 [+0.0117, +0.1787] SEP | +0.0654 [-0.0094, +0.1621] | -0.27 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 493 / 493 | -0.2748 | -0.2939 | -0.2092 | -0.0191 [-0.0802, +0.0397] | +0.0847 [+0.0162, +0.1765] SEP | +0.0656 [-0.0078, +0.1573] | -0.29 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 547 / 2638 | -0.2044 | -0.2450 | -0.1688 | -0.0407 [-0.1078, +0.0212] | +0.0763 [+0.0096, +0.1580] SEP | +0.0356 [-0.0334, +0.1234] | -1.14 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 547 / 2637 | -0.2080 | -0.2458 | -0.1716 | -0.0377 [-0.0930, +0.0141] | +0.0741 [+0.0071, +0.1626] SEP | +0.0364 [-0.0297, +0.1297] | -1.04 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 547 / 2632 | -0.2066 | -0.2429 | -0.1699 | -0.0363 [-0.0925, +0.0157] | +0.0730 [+0.0082, +0.1571] SEP | +0.0367 [-0.0315, +0.1244] | -0.99 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 133 / 133 | -0.2683 | -0.3075 | -0.2089 | -0.0392 [-0.1280, +0.0476] | +0.0985 [+0.0194, +0.2010] SEP | +0.0593 [-0.0217, +0.1638] | -0.66 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 132 / 132 | -0.2769 | -0.3045 | -0.2257 | -0.0276 [-0.0984, +0.0406] | +0.0788 [+0.0062, +0.1771] SEP | +0.0512 [-0.0235, +0.1488] | -0.54 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 132 / 132 | -0.2738 | -0.3020 | -0.2176 | -0.0282 [-0.1034, +0.0433] | +0.0844 [+0.0115, +0.1825] SEP | +0.0562 [-0.0208, +0.1573] | -0.50 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- EMA8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(EMA8) | P7 = s7-s7f [CI] | D = s(EMA8)-s7 [CI] | gap = s(EMA8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 139 / 686 | -0.2092 | -0.2629 | -0.1758 | -0.0537 [-0.1381, +0.0225] | +0.0872 [+0.0113, +0.1876] SEP | +0.0334 [-0.0449, +0.1408] | -1.61 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 139 / 684 | -0.2138 | -0.2633 | -0.1861 | -0.0495 [-0.1188, +0.0150] | +0.0772 [+0.0038, +0.1786] SEP | +0.0276 [-0.0441, +0.1298] | -1.79 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 139 / 684 | -0.2120 | -0.2602 | -0.1809 | -0.0482 [-0.1188, +0.0180] | +0.0794 [+0.0063, +0.1780] SEP | +0.0311 [-0.0440, +0.1349] | -1.55 (CI undefined: the CI of the denominator (gap) includes 0) |

## MECHANICAL READ of the pre-registered outcome table (cell S2/sharp (>= 0.15)/last segment)

* contrast **EMA8**: per-seed outcomes ['PHASE', 'PHASE']; both seeds agree = True; **PHASE**
