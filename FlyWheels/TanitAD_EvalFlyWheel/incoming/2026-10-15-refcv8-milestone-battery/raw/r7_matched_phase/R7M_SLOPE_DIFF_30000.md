# R7 matched-phase contrasts at N = 30000 (S2/S6 heading-error slope; PREREG_R7_MATCHED_PHASE_TURN.md)

* generated 2026-10-18T23:14:03; trees: `R7` = C:/Users/Admin/refcv8_eval_archive/battery/r7step30000_unc; `EMA8` = C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc; `RAW8` = C:/Users/Admin/refcv8_eval_archive/battery/step30000_unc
* estimator: episode-cluster bootstrap over the eval episodes, models FIXED (not another training run; not another inference run); B 2000; tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* CONTROL vs `heading_drift.shrinkage` on the same arms (must be < 1e-9): S2|EMA8: 0.00e+00 (ok); S2|RAW8: 0.00e+00 (ok); S6|EMA8: 0.00e+00 (ok); S6|RAW8: 0.00e+00 (ok)
* baseline identity (embedded refcv7-50,400 arm, max abs diff vs the R7 tree; must be 0): S2|EMA8|s0 0.0e+00, S2|RAW8|s0 0.0e+00, S2|EMA8|s1 0.0e+00, S2|RAW8|s1 0.0e+00, S6|EMA8|s0 0.0e+00, S6|RAW8|s0 0.0e+00, S6|EMA8|s1 0.0e+00, S6|RAW8|s1 0.0e+00

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

### S2 -- RAW8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1574 / 1574 | -0.0880 | -0.1247 | -0.1388 | -0.0367 [-0.0567, -0.0163] SEP | -0.0141 [-0.0482, +0.0234] | -0.0507 [-0.0890, -0.0130] SEP | 0.72 [0.32, 2.19] |
| seed 1 | 1574 / 1574 | -0.0800 | -0.1329 | -0.1516 | -0.0529 [-0.0785, -0.0273] SEP | -0.0187 [-0.0553, +0.0199] | -0.0716 [-0.1089, -0.0358] SEP | 0.74 [0.39, 1.46] |
| pooled (2-seed mean heading error) | 1572 / 1572 | -0.0838 | -0.1286 | -0.1449 | -0.0448 [-0.0656, -0.0231] SEP | -0.0162 [-0.0512, +0.0215] | -0.0610 [-0.0966, -0.0259] SEP | 0.73 [0.37, 1.67] |

### S2 -- RAW8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1590 / 4740 | -0.1497 | -0.1764 | -0.1917 | -0.0266 [-0.0459, -0.0073] SEP | -0.0153 [-0.0457, +0.0181] | -0.0419 [-0.0744, -0.0079] SEP | 0.64 [0.17, 2.11] |
| seed 1 | 1590 / 4740 | -0.1426 | -0.1839 | -0.2025 | -0.0412 [-0.0645, -0.0180] SEP | -0.0186 [-0.0500, +0.0150] | -0.0599 [-0.0930, -0.0273] SEP | 0.69 [0.33, 1.43] |
| pooled (2-seed mean heading error) | 1588 / 4735 | -0.1460 | -0.1800 | -0.1968 | -0.0340 [-0.0539, -0.0137] SEP | -0.0168 [-0.0472, +0.0169] | -0.0508 [-0.0826, -0.0188] SEP | 0.67 [0.29, 1.64] |

### S2 -- RAW8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 365 / 365 | -0.0972 | -0.1389 | -0.1534 | -0.0417 [-0.0660, -0.0187] SEP | -0.0144 [-0.0510, +0.0235] | -0.0561 [-0.0971, -0.0171] SEP | 0.74 [0.35, 2.05] |
| seed 1 | 364 / 364 | -0.0879 | -0.1488 | -0.1658 | -0.0609 [-0.0912, -0.0318] SEP | -0.0170 [-0.0579, +0.0266] | -0.0779 [-0.1204, -0.0375] SEP | 0.78 [0.42, 1.57] |
| pooled (2-seed mean heading error) | 364 / 364 | -0.0926 | -0.1439 | -0.1594 | -0.0513 [-0.0760, -0.0269] SEP | -0.0156 [-0.0538, +0.0254] | -0.0669 [-0.1076, -0.0291] SEP | 0.77 [0.41, 1.73] |

### S2 -- RAW8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 368 / 1097 | -0.1086 | -0.1409 | -0.1597 | -0.0322 [-0.0554, -0.0097] SEP | -0.0189 [-0.0531, +0.0171] | -0.0511 [-0.0891, -0.0142] SEP | 0.63 [0.22, 1.84] |
| seed 1 | 367 / 1095 | -0.1000 | -0.1505 | -0.1711 | -0.0505 [-0.0789, -0.0222] SEP | -0.0206 [-0.0586, +0.0194] | -0.0711 [-0.1102, -0.0324] SEP | 0.71 [0.36, 1.50] |
| pooled (2-seed mean heading error) | 367 / 1095 | -0.1043 | -0.1457 | -0.1653 | -0.0414 [-0.0650, -0.0173] SEP | -0.0196 [-0.0555, +0.0193] | -0.0610 [-0.0986, -0.0249] SEP | 0.68 [0.31, 1.59] |

### S2 -- RAW8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 627 | -0.0906 | -0.1132 | -0.1559 | -0.0226 [-0.0627, +0.0137] | -0.0428 [-0.0765, -0.0025] SEP | -0.0653 [-0.1133, -0.0079] SEP | 0.35 [-0.53, 0.91] |
| seed 1 | 627 / 627 | -0.0945 | -0.1209 | -0.1583 | -0.0264 [-0.0632, +0.0080] | -0.0374 [-0.0727, +0.0050] | -0.0638 [-0.1112, -0.0128] SEP | 0.41 [-0.27, 1.05] |
| pooled (2-seed mean heading error) | 627 / 627 | -0.0925 | -0.1170 | -0.1571 | -0.0245 [-0.0621, +0.0105] | -0.0401 [-0.0738, +0.0005] | -0.0646 [-0.1122, -0.0116] SEP | 0.38 [-0.35, 0.94] |

### S2 -- RAW8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 627 / 1881 | -0.0979 | -0.1111 | -0.1562 | -0.0132 [-0.0509, +0.0218] | -0.0452 [-0.0738, -0.0079] SEP | -0.0584 [-0.1009, -0.0065] SEP | 0.23 [-0.92, 0.78] |
| seed 1 | 627 / 1881 | -0.1007 | -0.1183 | -0.1574 | -0.0177 [-0.0521, +0.0161] | -0.0391 [-0.0683, -0.0029] SEP | -0.0567 [-0.0991, -0.0119] SEP | 0.31 [-0.59, 0.93] |
| pooled (2-seed mean heading error) | 627 / 1881 | -0.0993 | -0.1147 | -0.1568 | -0.0154 [-0.0515, +0.0188] | -0.0421 [-0.0706, -0.0066] SEP | -0.0576 [-0.0997, -0.0095] SEP | 0.27 [-0.72, 0.85] |

### S2 -- RAW8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 164 | -0.0901 | -0.1151 | -0.1649 | -0.0251 [-0.0684, +0.0133] | -0.0497 [-0.0821, -0.0112] SEP | -0.0748 [-0.1281, -0.0167] SEP | 0.34 [-0.41, 0.76] |
| seed 1 | 164 / 164 | -0.0924 | -0.1238 | -0.1642 | -0.0314 [-0.0705, +0.0047] | -0.0404 [-0.0746, -0.0004] SEP | -0.0718 [-0.1227, -0.0188] SEP | 0.44 [-0.11, 0.97] |
| pooled (2-seed mean heading error) | 164 / 164 | -0.0912 | -0.1195 | -0.1645 | -0.0282 [-0.0696, +0.0083] | -0.0451 [-0.0786, -0.0063] SEP | -0.0733 [-0.1247, -0.0181] SEP | 0.39 [-0.24, 0.84] |

### S2 -- RAW8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 164 / 492 | -0.0947 | -0.1089 | -0.1640 | -0.0142 [-0.0561, +0.0222] | -0.0550 [-0.0836, -0.0210] SEP | -0.0693 [-0.1176, -0.0169] SEP | 0.21 [-0.89, 0.62] |
| seed 1 | 164 / 492 | -0.0963 | -0.1172 | -0.1625 | -0.0209 [-0.0598, +0.0155] | -0.0453 [-0.0745, -0.0088] SEP | -0.0662 [-0.1130, -0.0175] SEP | 0.32 [-0.48, 0.78] |
| pooled (2-seed mean heading error) | 164 / 492 | -0.0955 | -0.1131 | -0.1633 | -0.0176 [-0.0578, +0.0180] | -0.0502 [-0.0783, -0.0154] SEP | -0.0678 [-0.1149, -0.0177] SEP | 0.26 [-0.60, 0.70] |

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

### S6 -- RAW8 -- curved (GT |yaw| >= 0.02) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1323 / 1323 | -0.2451 | -0.2781 | -0.2329 | -0.0330 [-0.0728, +0.0068] | +0.0452 [-0.0044, +0.1030] | +0.0122 [-0.0445, +0.0766] | -2.70 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 1322 / 1322 | -0.2465 | -0.2817 | -0.2255 | -0.0352 [-0.0720, +0.0006] | +0.0561 [-0.0001, +0.1240] | +0.0210 [-0.0379, +0.0853] | -1.68 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 1316 / 1316 | -0.2460 | -0.2780 | -0.2284 | -0.0321 [-0.0676, +0.0030] | +0.0496 [-0.0022, +0.1133] | +0.0175 [-0.0394, +0.0818] | -1.83 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- RAW8 -- curved (GT |yaw| >= 0.02) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 1413 / 6838 | -0.1713 | -0.2006 | -0.1607 | -0.0293 [-0.0720, +0.0188] | +0.0399 [-0.0126, +0.0987] | +0.0106 [-0.0597, +0.1024] | -2.75 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 1413 / 6835 | -0.1819 | -0.2050 | -0.1600 | -0.0232 [-0.0791, +0.0521] | +0.0450 [-0.0167, +0.1121] | +0.0218 [-0.0630, +0.1357] | -1.06 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 1411 / 6820 | -0.1765 | -0.2014 | -0.1598 | -0.0249 [-0.0726, +0.0354] | +0.0416 [-0.0147, +0.1040] | +0.0167 [-0.0606, +0.1202] | -1.49 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- RAW8 -- sharp (>= 0.15) | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 289 / 289 | -0.2594 | -0.3084 | -0.2749 | -0.0490 [-0.1026, +0.0026] | +0.0335 [-0.0218, +0.1009] | -0.0154 [-0.0823, +0.0600] | 3.17 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 288 / 288 | -0.2621 | -0.3133 | -0.2619 | -0.0512 [-0.1008, -0.0067] SEP | +0.0514 [-0.0150, +0.1364] | +0.0002 [-0.0730, +0.0752] | -323.20 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 287 / 287 | -0.2613 | -0.3088 | -0.2680 | -0.0475 [-0.0938, -0.0027] SEP | +0.0408 [-0.0189, +0.1169] | -0.0067 [-0.0788, +0.0677] | 7.04 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- RAW8 -- sharp (>= 0.15) | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 307 / 1476 | -0.1926 | -0.2550 | -0.2368 | -0.0623 [-0.1111, -0.0165] SEP | +0.0181 [-0.0378, +0.0820] | -0.0442 [-0.1126, +0.0279] | 1.41 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 306 / 1474 | -0.1890 | -0.2611 | -0.2320 | -0.0721 [-0.1212, -0.0289] SEP | +0.0292 [-0.0437, +0.1102] | -0.0430 [-0.1233, +0.0328] | 1.68 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 306 / 1472 | -0.1910 | -0.2564 | -0.2341 | -0.0654 [-0.1100, -0.0241] SEP | +0.0223 [-0.0408, +0.0941] | -0.0431 [-0.1144, +0.0293] | 1.52 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- RAW8 -- curved & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 493 / 493 | -0.2705 | -0.2997 | -0.2635 | -0.0292 [-0.1009, +0.0407] | +0.0362 [-0.0435, +0.1403] | +0.0070 [-0.0870, +0.1306] | -4.20 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 494 / 494 | -0.2785 | -0.2962 | -0.2479 | -0.0177 [-0.0785, +0.0382] | +0.0483 [-0.0402, +0.1619] | +0.0306 [-0.0659, +0.1521] | -0.58 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 489 / 489 | -0.2755 | -0.2947 | -0.2550 | -0.0192 [-0.0805, +0.0398] | +0.0397 [-0.0428, +0.1489] | +0.0205 [-0.0764, +0.1413] | -0.94 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- RAW8 -- curved & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 547 / 2639 | -0.2047 | -0.2455 | -0.2251 | -0.0408 [-0.1079, +0.0209] | +0.0205 [-0.0564, +0.1149] | -0.0204 [-0.1069, +0.0948] | 2.00 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 547 / 2637 | -0.2083 | -0.2460 | -0.2146 | -0.0377 [-0.0931, +0.0140] | +0.0313 [-0.0539, +0.1369] | -0.0064 [-0.0953, +0.1105] | 5.90 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 547 / 2631 | -0.2068 | -0.2433 | -0.2194 | -0.0365 [-0.0924, +0.0154] | +0.0239 [-0.0538, +0.1240] | -0.0126 [-0.1002, +0.1032] | 2.89 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- RAW8 -- sharp & v0 5-10 | last segment

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 132 / 132 | -0.2691 | -0.3087 | -0.2835 | -0.0396 [-0.1292, +0.0476] | +0.0252 [-0.0594, +0.1346] | -0.0144 [-0.1107, +0.1159] | 2.75 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 132 / 132 | -0.2769 | -0.3045 | -0.2599 | -0.0276 [-0.0984, +0.0406] | +0.0446 [-0.0516, +0.1765] | +0.0171 [-0.0881, +0.1533] | -1.61 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 131 / 131 | -0.2746 | -0.3031 | -0.2717 | -0.0285 [-0.1038, +0.0432] | +0.0314 [-0.0570, +0.1486] | +0.0029 [-0.0978, +0.1351] | -9.96 (CI undefined: the CI of the denominator (gap) includes 0) |

### S6 -- RAW8 -- sharp & v0 5-10 | segments 2..K

| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(RAW8) | P7 = s7-s7f [CI] | D = s(RAW8)-s7 [CI] | gap = s(RAW8)-s7f [CI] | phase fraction |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 139 / 685 | -0.2094 | -0.2633 | -0.2496 | -0.0539 [-0.1384, +0.0225] | +0.0137 [-0.0687, +0.1224] | -0.0402 [-0.1388, +0.0866] | 1.34 (CI undefined: the CI of the denominator (gap) includes 0) |
| seed 1 | 139 / 684 | -0.2138 | -0.2633 | -0.2314 | -0.0495 [-0.1188, +0.0150] | +0.0319 [-0.0645, +0.1617] | -0.0176 [-0.1173, +0.1153] | 2.81 (CI undefined: the CI of the denominator (gap) includes 0) |
| pooled (2-seed mean heading error) | 139 / 683 | -0.2122 | -0.2606 | -0.2403 | -0.0484 [-0.1188, +0.0177] | +0.0203 [-0.0682, +0.1382] | -0.0281 [-0.1279, +0.1009] | 1.72 (CI undefined: the CI of the denominator (gap) includes 0) |

## MECHANICAL READ of the pre-registered outcome table (cell S2/sharp (>= 0.15)/last segment)

* contrast **EMA8**: per-seed outcomes ['PHASE', 'PHASE']; both seeds agree = True; **PHASE**
* contrast **RAW8**: per-seed outcomes ['PHASE', 'PHASE']; both seeds agree = True; **PHASE**
