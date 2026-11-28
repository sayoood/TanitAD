# FAN vs SELECTOR -- candidate plans or selection? (refcv8 tight-turn under-steering)

- **Tier:** T1-style open loop on the S2 panel windows for `os` (own action); `oracle_sel` is a T0 ceiling (nearest-GT anchor, refined)
- **Evidence class:** MEASURED (ours; per-window dumps listed under `inputs`)
- **Estimator:** paired episode-cluster bootstrap, ONE shared resample matrix (E=139 episodes, B=2000, rng seed 29); answers 'another draw of EPISODES, models fixed' only -- NOT another training run (H-ESTIM-SEED-1), NOT another inference run
- **n:** 4754 S2 windows / 139 episodes; sharp (GT mean |yaw rate| >= 0.15 rad/s) = 368 windows; curved (>= 0.02) = 1604.
- **Metric:** through-origin slope of turn-aligned signed heading error on turn-aligned GT chord heading (-0.10 = plan turns 10 % less than GT), `hd.shrinkage` definition, 2-seed mean of the signed per-segment error, segments valid on every arm of the contrast.
- **Not answered:** another TRAINING run (one checkpoint per arm; H-ESTIM-SEED-1) and another INFERENCE run (per-seed rows only).

## 0. Reproduction control (rule 6; asserted before anything else)

banked EMA-15k sharp last-segment slope -0.140937, refcv7 -0.092130; reproduced through the ORIGINAL `hd.shrinkage` -0.140937 / -0.092130 (n_seg 363); max |diff| slope 0.00e+00, CI 0.00e+00; **PASS = True**. The new `Ctx` slope code matches `hd.shrinkage` to 1e-12.

Other controls (all MEASURED):

- r7 own os vs embedded b_refcv7_50400_s0 (max abs diff, m): `0`
- nav_cmd r7 == r8 EMA-15k, s0 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), EMA-15k|s0: `0`
- sel_agrees_oracle == (sel_idx==a_star), EMA-15k|s0 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), EMA-10.5k|s0: `0`
- sel_agrees_oracle == (sel_idx==a_star), EMA-10.5k|s0 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), EMA-5k|s0: `0`
- sel_agrees_oracle == (sel_idx==a_star), EMA-5k|s0 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), RAW-5k|s0: `0`
- sel_agrees_oracle == (sel_idx==a_star), RAW-5k|s0 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), r7|s0: `0`
- r7 max sel_idx s0 (117 anchors -> <=116): `113`
- r8 EMA-15k max sel_idx s0 (149 candidates -> <=148): `148`
- oracle_sel seed difference max abs (EMA-15k s0 vs s1, m): `0.527925`
- r7 own os vs embedded b_refcv7_50400_s1 (max abs diff, m): `0`
- nav_cmd r7 == r8 EMA-15k, s1 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), EMA-15k|s1: `0`
- sel_agrees_oracle == (sel_idx==a_star), EMA-15k|s1 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), EMA-10.5k|s1: `0`
- sel_agrees_oracle == (sel_idx==a_star), EMA-10.5k|s1 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), EMA-5k|s1: `0`
- sel_agrees_oracle == (sel_idx==a_star), EMA-5k|s1 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), RAW-5k|s1: `0`
- sel_agrees_oracle == (sel_idx==a_star), RAW-5k|s1 (frac equal): `1`
- os==oracle_sel where sel_idx==a_star (max abs m), r7|s1: `0`
- r7 max sel_idx s1 (117 anchors -> <=116): `113`
- r8 EMA-15k max sel_idx s1 (149 candidates -> <=148): `148`
- nav arms finite fraction EMA-15k|s0: `{'os_navzero': 1.0, 'os_navflip': 1.0, 'os_navshuf': 1.0}`
- nav arms finite fraction EMA-15k|s1: `{'os_navzero': 0.0, 'os_navflip': 0.0, 'os_navshuf': 0.0}`
- nav arms finite fraction r7|s0: `{'os_navzero': 1.0, 'os_navflip': 1.0, 'os_navshuf': 1.0}`
- nav arms finite fraction r7|s1: `{'os_navzero': 0.0, 'os_navflip': 0.0, 'os_navshuf': 0.0}`
- anchor bank: r7 == r8 ckpt (anchors, controls) bitwise: `True`
- anchor bank sha256[:16] (anchors bytes | controls bytes), r8 ckpt: `['51f930dc6f3564ff', 'b072f4c052331beb']`
- manifest anchor_file_vs_ckpt_buffers max_abs_diff (all trees): `{'EMA-15k': {'core.decoder.anchors': 0.0, 'core.decoder.anchor_controls': 0.0}, 'EMA-10.5k': {'core.decoder.anchors': 0.0, 'core.decoder.anchor_controls': 0.0}, 'EMA-5k': {'core.decoder.anchors': 0.0, 'core.decoder.anchor_controls': 0.0}, 'RAW-5k': {'core.decoder.anchors': 0.0, 'core.decoder.anchor_controls': 0.0}}`

## 1. Slope decomposition (identity asserted at the point estimate and on all bootstrap draws)

gap = CANDIDATE (anchor coverage + refinement: r8 `oracle_sel` - r7 `oracle_sel`) + SELECTION ((r8 os - r8 orc) - (r7 os - r7 orc)). Negative = more under-rotation.

### EMA-15k vs refcv7-50,400

**2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 363, n_seg 363, n_ep 42, mean GT turn-aligned heading 25.4 deg; identity err 3.5e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1409 [-0.2445, -0.0616] SEP |
| slope r8 oracle_sel | -0.0826 [-0.1315, -0.0396] SEP |
| slope r7 os | -0.0921 [-0.1765, -0.0255] SEP |
| slope r7 oracle_sel | -0.0479 [-0.0899, -0.0137] SEP |
| GAP (r8 os - r7 os) | -0.0488 [-0.0867, -0.0176] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0347 [-0.0527, -0.0183] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0141 [-0.0430, +0.0102] |
| selection cost r8 (os - orc) | -0.0583 [-0.1278, -0.0067] SEP |
| selection cost r7 (os - orc) | -0.0442 [-0.1056, +0.0030] |
| share CANDIDATE of GAP | 0.71 [0.43, 1.51] |
| share SELECTION of GAP | 0.29 [-0.51, 0.57] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 0.999)

**2-seed mean | sharp (GT |yaw| >= 0.15) | segments 2..K** -- n_win 366, n_seg 1094, n_ep 42, mean GT turn-aligned heading 17.9 deg; identity err 3.5e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1523 [-0.2562, -0.0713] SEP |
| slope r8 oracle_sel | -0.0940 [-0.1476, -0.0474] SEP |
| slope r7 os | -0.1041 [-0.1873, -0.0360] SEP |
| slope r7 oracle_sel | -0.0568 [-0.1033, -0.0179] SEP |
| GAP (r8 os - r7 os) | -0.0482 [-0.0838, -0.0187] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0372 [-0.0555, -0.0200] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0111 [-0.0370, +0.0108] |
| selection cost r8 (os - orc) | -0.0584 [-0.1272, -0.0071] SEP |
| selection cost r7 (os - orc) | -0.0473 [-0.1088, +0.0004] |
| share CANDIDATE of GAP | 0.77 [0.49, 1.46] |
| share SELECTION of GAP | 0.23 [-0.46, 0.51] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

**2-seed mean | curved (GT |yaw| >= 0.02) | last segment (1.5->2.0 s chord)** -- n_win 1574, n_seg 1574, n_ep 101, mean GT turn-aligned heading 10.2 deg; identity err 4.9e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1325 [-0.2191, -0.0645] SEP |
| slope r8 oracle_sel | -0.0795 [-0.1235, -0.0420] SEP |
| slope r7 os | -0.0842 [-0.1557, -0.0272] SEP |
| slope r7 oracle_sel | -0.0447 [-0.0823, -0.0143] SEP |
| GAP (r8 os - r7 os) | -0.0483 [-0.0800, -0.0224] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0348 [-0.0502, -0.0207] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0136 [-0.0391, +0.0079] |
| selection cost r8 (os - orc) | -0.0530 [-0.1124, -0.0087] SEP |
| selection cost r7 (os - orc) | -0.0394 [-0.0911, +0.0005] |
| share CANDIDATE of GAP | 0.72 [0.45, 1.29] |
| share SELECTION of GAP | 0.28 [-0.29, 0.55] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

**2-seed mean | curved (GT |yaw| >= 0.02) | segments 2..K** -- n_win 1592, n_seg 4746, n_ep 101, mean GT turn-aligned heading 7.2 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1458 [-0.2323, -0.0775] SEP |
| slope r8 oracle_sel | -0.0941 [-0.1415, -0.0535] SEP |
| slope r7 os | -0.0976 [-0.1692, -0.0399] SEP |
| slope r7 oracle_sel | -0.0574 [-0.0984, -0.0231] SEP |
| GAP (r8 os - r7 os) | -0.0482 [-0.0768, -0.0238] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0367 [-0.0520, -0.0223] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0115 [-0.0340, +0.0075] |
| selection cost r8 (os - orc) | -0.0518 [-0.1084, -0.0081] SEP |
| selection cost r7 (os - orc) | -0.0402 [-0.0920, -0.0004] SEP |
| share CANDIDATE of GAP | 0.76 [0.50, 1.26] |
| share SELECTION of GAP | 0.24 [-0.26, 0.50] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

**seed 0 | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 363, n_seg 363, n_ep 42, mean GT turn-aligned heading 25.4 deg; identity err 4.5e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1397 [-0.2424, -0.0592] SEP |
| slope r8 oracle_sel | -0.0817 [-0.1317, -0.0383] SEP |
| slope r7 os | -0.0968 [-0.1860, -0.0263] SEP |
| slope r7 oracle_sel | -0.0473 [-0.0896, -0.0124] SEP |
| GAP (r8 os - r7 os) | -0.0429 [-0.0791, -0.0130] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0344 [-0.0521, -0.0182] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0085 [-0.0380, +0.0177] |
| selection cost r8 (os - orc) | -0.0580 [-0.1270, -0.0074] SEP |
| selection cost r7 (os - orc) | -0.0496 [-0.1151, +0.0008] |
| share CANDIDATE of GAP | 0.80 [0.46, 2.05] |
| share SELECTION of GAP | 0.20 [-1.05, 0.54] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 0.999)

**seed 1 | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 363, n_seg 363, n_ep 42, mean GT turn-aligned heading 25.4 deg; identity err 4.9e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1422 [-0.2466, -0.0625] SEP |
| slope r8 oracle_sel | -0.0835 [-0.1321, -0.0410] SEP |
| slope r7 os | -0.0874 [-0.1662, -0.0247] SEP |
| slope r7 oracle_sel | -0.0486 [-0.0907, -0.0141] SEP |
| GAP (r8 os - r7 os) | -0.0547 [-0.0937, -0.0227] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0349 [-0.0532, -0.0183] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0198 [-0.0510, +0.0049] |
| selection cost r8 (os - orc) | -0.0586 [-0.1294, -0.0068] SEP |
| selection cost r7 (os - orc) | -0.0388 [-0.0960, +0.0059] |
| share CANDIDATE of GAP | 0.64 [0.40, 1.22] |
| share SELECTION of GAP | 0.36 [-0.22, 0.60] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

### EMA-10.5k vs refcv7-50,400

**2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 364, n_seg 364, n_ep 42, mean GT turn-aligned heading 25.4 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1548 [-0.2571, -0.0759] SEP |
| slope r8 oracle_sel | -0.0982 [-0.1471, -0.0570] SEP |
| slope r7 os | -0.0922 [-0.1765, -0.0257] SEP |
| slope r7 oracle_sel | -0.0479 [-0.0899, -0.0137] SEP |
| GAP (r8 os - r7 os) | -0.0626 [-0.0980, -0.0339] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0503 [-0.0656, -0.0363] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0123 [-0.0385, +0.0089] |
| selection cost r8 (os - orc) | -0.0566 [-0.1265, -0.0038] SEP |
| selection cost r7 (os - orc) | -0.0442 [-0.1056, +0.0028] |
| share CANDIDATE of GAP | 0.80 [0.59, 1.23] |
| share SELECTION of GAP | 0.20 [-0.23, 0.41] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

**2-seed mean | sharp (GT |yaw| >= 0.15) | segments 2..K** -- n_win 367, n_seg 1096, n_ep 42, mean GT turn-aligned heading 17.9 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1654 [-0.2676, -0.0867] SEP |
| slope r8 oracle_sel | -0.1100 [-0.1640, -0.0656] SEP |
| slope r7 os | -0.1041 [-0.1873, -0.0361] SEP |
| slope r7 oracle_sel | -0.0568 [-0.1033, -0.0179] SEP |
| GAP (r8 os - r7 os) | -0.0613 [-0.0938, -0.0340] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0532 [-0.0689, -0.0391] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0081 [-0.0323, +0.0118] |
| selection cost r8 (os - orc) | -0.0554 [-0.1235, -0.0049] SEP |
| selection cost r7 (os - orc) | -0.0473 [-0.1088, +0.0004] |
| share CANDIDATE of GAP | 0.87 [0.64, 1.31] |
| share SELECTION of GAP | 0.13 [-0.31, 0.36] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

### EMA-5k vs refcv7-50,400

**2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 364, n_seg 364, n_ep 42, mean GT turn-aligned heading 25.4 deg; identity err 2.8e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1303 [-0.2322, -0.0535] SEP |
| slope r8 oracle_sel | -0.0864 [-0.1318, -0.0487] SEP |
| slope r7 os | -0.0922 [-0.1765, -0.0257] SEP |
| slope r7 oracle_sel | -0.0479 [-0.0899, -0.0137] SEP |
| GAP (r8 os - r7 os) | -0.0382 [-0.0648, -0.0175] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0384 [-0.0470, -0.0302] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | +0.0003 [-0.0238, +0.0195] |
| selection cost r8 (os - orc) | -0.0440 [-0.1229, +0.0117] |
| selection cost r7 (os - orc) | -0.0442 [-0.1056, +0.0028] |
| share CANDIDATE of GAP | 1.01 [0.62, 2.08] |
| share SELECTION of GAP | -0.01 [-1.08, 0.38] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

**2-seed mean | sharp (GT |yaw| >= 0.15) | segments 2..K** -- n_win 367, n_seg 1096, n_ep 42, mean GT turn-aligned heading 17.9 deg; identity err 3.5e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.1424 [-0.2435, -0.0658] SEP |
| slope r8 oracle_sel | -0.0977 [-0.1475, -0.0568] SEP |
| slope r7 os | -0.1041 [-0.1873, -0.0361] SEP |
| slope r7 oracle_sel | -0.0568 [-0.1033, -0.0179] SEP |
| GAP (r8 os - r7 os) | -0.0383 [-0.0626, -0.0194] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.0409 [-0.0495, -0.0324] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | +0.0026 [-0.0183, +0.0198] |
| selection cost r8 (os - orc) | -0.0447 [-0.1215, +0.0099] |
| selection cost r7 (os - orc) | -0.0473 [-0.1088, +0.0004] |
| share CANDIDATE of GAP | 1.07 [0.70, 1.96] |
| share SELECTION of GAP | -0.07 [-0.96, 0.30] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

### RAW-5k vs refcv7-50,400

**2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 361, n_seg 361, n_ep 42, mean GT turn-aligned heading 25.4 deg; identity err 6.9e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.2340 [-0.3402, -0.1455] SEP |
| slope r8 oracle_sel | -0.1638 [-0.2168, -0.1131] SEP |
| slope r7 os | -0.0908 [-0.1741, -0.0240] SEP |
| slope r7 oracle_sel | -0.0473 [-0.0897, -0.0129] SEP |
| GAP (r8 os - r7 os) | -0.1432 [-0.1967, -0.0953] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.1165 [-0.1438, -0.0877] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0267 [-0.0605, +0.0042] |
| selection cost r8 (os - orc) | -0.0702 [-0.1451, -0.0148] SEP |
| selection cost r7 (os - orc) | -0.0435 [-0.1056, +0.0040] |
| share CANDIDATE of GAP | 0.81 [0.67, 1.04] |
| share SELECTION of GAP | 0.19 [-0.04, 0.33] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)

**2-seed mean | sharp (GT |yaw| >= 0.15) | segments 2..K** -- n_win 364, n_seg 1088, n_ep 42, mean GT turn-aligned heading 18.0 deg; identity err 7.6e-17

| term | est [95 % CI] |
|---|---|
| slope r8 os | -0.2504 [-0.3566, -0.1626] SEP |
| slope r8 oracle_sel | -0.1819 [-0.2378, -0.1286] SEP |
| slope r7 os | -0.1030 [-0.1861, -0.0347] SEP |
| slope r7 oracle_sel | -0.0564 [-0.1033, -0.0174] SEP |
| GAP (r8 os - r7 os) | -0.1474 [-0.1994, -0.1007] SEP |
| CANDIDATE (r8 orc - r7 orc) | -0.1255 [-0.1529, -0.0960] SEP |
| SELECTION ((r8 os-orc) - (r7 os-orc)) | -0.0219 [-0.0540, +0.0071] |
| selection cost r8 (os - orc) | -0.0685 [-0.1409, -0.0148] SEP |
| selection cost r7 (os - orc) | -0.0466 [-0.1079, +0.0011] |
| share CANDIDATE of GAP | 0.85 [0.71, 1.06] |
| share SELECTION of GAP | 0.15 [-0.06, 0.29] |

(shares are ratios of correlated bootstrap terms; P(boot gap has the point sign) = 1.000)


## 2. Selection detail

### 2c. Agreement with the oracle anchor, allocator-extra selection rate, a_star identity

| tree|seed | group | n | r8 selects extra (117-148) | sel_agrees_oracle r8 | r7 | r8 - r7 | a_star same r8/r7 |
|---|---|---|---|---|---|---|---|
| EMA-15k|s0 | all | 4754 | +0.124 [+0.104, +0.145] SEP | +0.428 [+0.394, +0.463] SEP | +0.534 [+0.493, +0.575] SEP | -0.106 [-0.138, -0.075] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-15k|s0 | sharp | 368 | +0.068 [+0.038, +0.106] SEP | +0.280 [+0.208, +0.350] SEP | +0.283 [+0.214, +0.348] SEP | -0.003 [-0.079, +0.078] | +1.000 [+1.000, +1.000] SEP |
| EMA-15k|s0 | curved | 1604 | +0.123 [+0.095, +0.154] SEP | +0.314 [+0.276, +0.353] SEP | +0.372 [+0.327, +0.419] SEP | -0.058 [-0.092, -0.024] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-15k|s1 | all | 4754 | +0.126 [+0.108, +0.146] SEP | +0.426 [+0.393, +0.461] SEP | +0.534 [+0.491, +0.577] SEP | -0.107 [-0.139, -0.076] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-15k|s1 | sharp | 368 | +0.043 [+0.018, +0.075] SEP | +0.299 [+0.220, +0.371] SEP | +0.280 [+0.208, +0.349] SEP | +0.019 [-0.053, +0.097] | +1.000 [+1.000, +1.000] SEP |
| EMA-15k|s1 | curved | 1604 | +0.123 [+0.097, +0.152] SEP | +0.314 [+0.274, +0.354] SEP | +0.373 [+0.325, +0.420] SEP | -0.059 [-0.092, -0.024] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-10.5k|s0 | all | 4754 | +0.122 [+0.105, +0.140] SEP | +0.447 [+0.413, +0.482] SEP | +0.534 [+0.493, +0.575] SEP | -0.087 [-0.113, -0.061] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-10.5k|s0 | sharp | 368 | +0.065 [+0.031, +0.108] SEP | +0.274 [+0.202, +0.340] SEP | +0.283 [+0.214, +0.348] SEP | -0.008 [-0.080, +0.066] | +1.000 [+1.000, +1.000] SEP |
| EMA-10.5k|s0 | curved | 1604 | +0.117 [+0.090, +0.144] SEP | +0.320 [+0.280, +0.360] SEP | +0.372 [+0.327, +0.419] SEP | -0.052 [-0.085, -0.022] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-10.5k|s1 | all | 4754 | +0.125 [+0.107, +0.144] SEP | +0.444 [+0.410, +0.478] SEP | +0.534 [+0.491, +0.577] SEP | -0.090 [-0.115, -0.063] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-10.5k|s1 | sharp | 368 | +0.049 [+0.028, +0.077] SEP | +0.280 [+0.203, +0.350] SEP | +0.280 [+0.208, +0.349] SEP | +0.000 [-0.065, +0.073] | +1.000 [+1.000, +1.000] SEP |
| EMA-10.5k|s1 | curved | 1604 | +0.113 [+0.087, +0.140] SEP | +0.322 [+0.279, +0.364] SEP | +0.373 [+0.325, +0.420] SEP | -0.051 [-0.083, -0.020] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-5k|s0 | all | 4754 | +0.177 [+0.157, +0.198] SEP | +0.419 [+0.386, +0.452] SEP | +0.534 [+0.493, +0.575] SEP | -0.115 [-0.139, -0.090] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-5k|s0 | sharp | 368 | +0.079 [+0.043, +0.121] SEP | +0.266 [+0.198, +0.325] SEP | +0.283 [+0.214, +0.348] SEP | -0.016 [-0.073, +0.040] | +1.000 [+1.000, +1.000] SEP |
| EMA-5k|s0 | curved | 1604 | +0.151 [+0.122, +0.184] SEP | +0.300 [+0.261, +0.339] SEP | +0.372 [+0.327, +0.419] SEP | -0.072 [-0.102, -0.042] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-5k|s1 | all | 4754 | +0.189 [+0.169, +0.211] SEP | +0.416 [+0.384, +0.448] SEP | +0.534 [+0.491, +0.577] SEP | -0.118 [-0.143, -0.093] SEP | +1.000 [+1.000, +1.000] SEP |
| EMA-5k|s1 | sharp | 368 | +0.054 [+0.027, +0.088] SEP | +0.274 [+0.210, +0.330] SEP | +0.280 [+0.208, +0.349] SEP | -0.005 [-0.053, +0.043] | +1.000 [+1.000, +1.000] SEP |
| EMA-5k|s1 | curved | 1604 | +0.156 [+0.129, +0.187] SEP | +0.304 [+0.266, +0.342] SEP | +0.373 [+0.325, +0.420] SEP | -0.069 [-0.098, -0.041] SEP | +1.000 [+1.000, +1.000] SEP |
| RAW-5k|s0 | all | 4754 | +0.149 [+0.129, +0.169] SEP | +0.352 [+0.324, +0.380] SEP | +0.534 [+0.493, +0.575] SEP | -0.182 [-0.211, -0.152] SEP | +1.000 [+1.000, +1.000] SEP |
| RAW-5k|s0 | sharp | 368 | +0.060 [+0.033, +0.093] SEP | +0.171 [+0.111, +0.234] SEP | +0.283 [+0.214, +0.348] SEP | -0.111 [-0.200, -0.024] SEP | +1.000 [+1.000, +1.000] SEP |
| RAW-5k|s0 | curved | 1604 | +0.110 [+0.086, +0.136] SEP | +0.218 [+0.189, +0.249] SEP | +0.372 [+0.327, +0.419] SEP | -0.154 [-0.191, -0.116] SEP | +1.000 [+1.000, +1.000] SEP |
| RAW-5k|s1 | all | 4754 | +0.148 [+0.130, +0.168] SEP | +0.352 [+0.324, +0.382] SEP | +0.534 [+0.491, +0.577] SEP | -0.182 [-0.211, -0.151] SEP | +1.000 [+1.000, +1.000] SEP |
| RAW-5k|s1 | sharp | 368 | +0.041 [+0.020, +0.067] SEP | +0.185 [+0.127, +0.245] SEP | +0.280 [+0.208, +0.349] SEP | -0.095 [-0.187, -0.005] SEP | +1.000 [+1.000, +1.000] SEP |
| RAW-5k|s1 | curved | 1604 | +0.105 [+0.083, +0.131] SEP | +0.223 [+0.193, +0.253] SEP | +0.373 [+0.325, +0.420] SEP | -0.150 [-0.190, -0.110] SEP | +1.000 [+1.000, +1.000] SEP |

### 2a. Selected vs oracle anchor (EMA-15k bank; sharp windows, v0 >= 2 m/s, in-bank selections)

orientation/units check: `{"corr(alat_col1[a_star], gt_yaw_signed) on sharp windows (EMA-15k s0)": -0.05511106514793033, "orientation_sign_used": -1.0, "controls_col0_vs_col1_corr_with_anchor_terminal_y": [-4.180846583684545e-18, 0.7449138367005517], "units_note": "controls col1 = lateral acceleration (m/s^2) per the refc anchor builder (alat); verified only by the correlation with the anchors' terminal lateral offset above"}`

anchor-index faithfulness check: `{"corr(alat[a_star], GT lateral y@2s), all windows": -0.14329294675667642, "corr(alat[a_star], GT lateral y@2s), windows with |y@2s|>3 m": -0.4209853062393363, "corr(bank anchor terminal y[a_star], GT lateral y@2s), all windows": -0.16205209648171637, "corr(oracle_sel lateral y@2s, GT lateral y@2s), all windows": 0.9728604422530963, "n_windows_|y@2s|>3": 295, "reading": "if the refined oracle plan tracks GT (corr~1) while the raw anchor control of a_star does not, the anchor INDEX is not a geometric prototype of the final plan: the refinement, not the bank, carries the turn"}`

**CAUTION: the |alat| / implied-turn-rate rows below are only interpretable if the anchor index is a geometric prototype of the final plan; see the faithfulness check above.**

**r8|s0**

| quantity | est [95 % CI] |
|---|---|
| n_sharp_v0>=2 | 349 |
| n_sharp_v0>=2_in_bank_selection | 330 |
| |alat| selected anchor (m/s^2) | +0.6295 [+0.5070, +0.7695] SEP |
| |alat| oracle anchor (m/s^2) | +0.8182 [+0.6763, +0.9594] SEP |
| |alat| selected - oracle | -0.1886 [-0.2851, -0.0756] SEP |
| implied-turn-rate / GT-turn-rate slope, selected anchor (ESTIMATED) | +0.2328 [+0.1068, +0.3926] SEP |
| implied-turn-rate / GT-turn-rate slope, oracle anchor (ESTIMATED) | +0.1307 [-0.0366, +0.3427] |
| frac selected anchor curves the WRONG way (turn-aligned alat<0) | +0.2182 [+0.1709, +0.2674] SEP |
| frac oracle anchor curves the wrong way | +0.3030 [+0.2451, +0.3633] SEP |

**r7|s0**

| quantity | est [95 % CI] |
|---|---|
| n_sharp_v0>=2 | 349 |
| n_sharp_v0>=2_in_bank_selection | 349 |
| |alat| selected anchor (m/s^2) | +0.7457 [+0.6103, +0.8942] SEP |
| |alat| oracle anchor (m/s^2) | +0.8274 [+0.6872, +0.9663] SEP |
| |alat| selected - oracle | -0.0817 [-0.1623, +0.0139] |
| implied-turn-rate / GT-turn-rate slope, selected anchor (ESTIMATED) | +0.2735 [+0.1479, +0.4358] SEP |
| implied-turn-rate / GT-turn-rate slope, oracle anchor (ESTIMATED) | +0.1105 [-0.0595, +0.3203] |
| frac selected anchor curves the WRONG way (turn-aligned alat<0) | +0.2292 [+0.1823, +0.2797] SEP |
| frac oracle anchor curves the wrong way | +0.3123 [+0.2527, +0.3761] SEP |

**r8|s1**

| quantity | est [95 % CI] |
|---|---|
| n_sharp_v0>=2 | 349 |
| n_sharp_v0>=2_in_bank_selection | 338 |
| |alat| selected anchor (m/s^2) | +0.6590 [+0.5338, +0.8036] SEP |
| |alat| oracle anchor (m/s^2) | +0.8388 [+0.6899, +0.9855] SEP |
| |alat| selected - oracle | -0.1797 [-0.2788, -0.0653] SEP |
| implied-turn-rate / GT-turn-rate slope, selected anchor (ESTIMATED) | +0.2259 [+0.1048, +0.3822] SEP |
| implied-turn-rate / GT-turn-rate slope, oracle anchor (ESTIMATED) | +0.1161 [-0.0549, +0.3270] |
| frac selected anchor curves the WRONG way (turn-aligned alat<0) | +0.2219 [+0.1782, +0.2687] SEP |
| frac oracle anchor curves the wrong way | +0.3107 [+0.2500, +0.3761] SEP |

**r7|s1**

| quantity | est [95 % CI] |
|---|---|
| n_sharp_v0>=2 | 349 |
| n_sharp_v0>=2_in_bank_selection | 349 |
| |alat| selected anchor (m/s^2) | +0.7479 [+0.6124, +0.8942] SEP |
| |alat| oracle anchor (m/s^2) | +0.8274 [+0.6872, +0.9663] SEP |
| |alat| selected - oracle | -0.0795 [-0.1641, +0.0187] |
| implied-turn-rate / GT-turn-rate slope, selected anchor (ESTIMATED) | +0.2677 [+0.1469, +0.4294] SEP |
| implied-turn-rate / GT-turn-rate slope, oracle anchor (ESTIMATED) | +0.1105 [-0.0595, +0.3203] |
| frac selected anchor curves the WRONG way (turn-aligned alat<0) | +0.2292 [+0.1800, +0.2817] SEP |
| frac oracle anchor curves the wrong way | +0.3123 [+0.2527, +0.3761] SEP |


### 2b. Allocator extras and selection-class decomposition (sharp turns, last segment, per inference seed)

**EMA-15k | s0** -- n_win 363; frac r8-extra 0.063; full GAP -0.0429; r8 selection cost -0.0580; r7 selection cost -0.0496

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 23 | -0.6165 [-0.8844, -0.2884] SEP | -0.2396 [-0.3143, -0.1314] SEP | -0.3696 [-0.5243, -0.1589] SEP | -0.0156 [-0.0374, -0.0022] SEP | -0.0238 [-0.0552, -0.0045] SEP |
| IN-BANK agrees with oracle anchor | 103 | -0.0447 [-0.1058, +0.0106] | -0.0447 [-0.1058, +0.0106] | -0.0382 [-0.1005, +0.0204] | -0.0020 [-0.0093, +0.0050] | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 237 | -0.1383 [-0.2356, -0.0567] SEP | -0.0839 [-0.1359, -0.0371] SEP | -0.0981 [-0.1943, -0.0190] SEP | -0.0253 [-0.0505, -0.0053] SEP | -0.0342 [-0.0770, +0.0004] |
| REST (all non-extra) | 340 | -0.1076 [-0.1876, -0.0456] SEP | -0.0711 [-0.1208, -0.0291] SEP | -0.0784 [-0.1572, -0.0180] SEP | -0.0273 [-0.0548, -0.0052] SEP | -0.0342 [-0.0770, +0.0004] |

r8 os slope EXTRA - REST: -0.5089 [-0.7575, -0.2008] SEP

r7 classes: r7 agrees with oracle anchor: n 103, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 260, contrib. to r7 sel. cost -0.0496 [-0.1151, +0.0008]

**EMA-15k | s1** -- n_win 363; frac r8-extra 0.041; full GAP -0.0547; r8 selection cost -0.0586; r7 selection cost -0.0388

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 15 | -0.7469 [-1.0216, -0.3468] SEP | -0.2738 [-0.3218, -0.1949] SEP | -0.3273 [-0.4638, -0.1940] SEP | -0.0172 [-0.0479, -0.0016] SEP | -0.0193 [-0.0508, -0.0008] SEP |
| IN-BANK agrees with oracle anchor | 110 | -0.0401 [-0.0942, +0.0085] | -0.0401 [-0.0942, +0.0085] | -0.0211 [-0.0746, +0.0272] | -0.0060 [-0.0127, +0.0010] | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 238 | -0.1541 [-0.2478, -0.0715] SEP | -0.0929 [-0.1459, -0.0474] SEP | -0.1050 [-0.1984, -0.0244] SEP | -0.0315 [-0.0585, -0.0083] SEP | -0.0393 [-0.0861, -0.0006] SEP |
| REST (all non-extra) | 348 | -0.1164 [-0.1963, -0.0514] SEP | -0.0754 [-0.1239, -0.0342] SEP | -0.0772 [-0.1524, -0.0174] SEP | -0.0376 [-0.0675, -0.0131] SEP | -0.0393 [-0.0861, -0.0006] SEP |

r8 os slope EXTRA - REST: -0.6305 [-0.8733, -0.2451] SEP

r7 classes: r7 agrees with oracle anchor: n 103, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 260, contrib. to r7 sel. cost -0.0388 [-0.0960, +0.0059]

**EMA-10.5k | s0** -- n_win 364; frac r8-extra 0.060; full GAP -0.0602; r8 selection cost -0.0597; r7 selection cost -0.0495

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 22 | -0.5644 [-0.8133, -0.3327] SEP | -0.2203 [-0.3126, -0.1365] SEP | -0.3293 [-0.4754, -0.1404] SEP | -0.0127 [-0.0337, -0.0008] SEP | -0.0186 [-0.0393, -0.0045] SEP |
| IN-BANK agrees with oracle anchor | 101 | -0.0598 [-0.1170, -0.0062] SEP | -0.0598 [-0.1170, -0.0062] SEP | -0.0376 [-0.0975, +0.0188] | -0.0062 [-0.0124, +0.0001] | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 241 | -0.1649 [-0.2720, -0.0789] SEP | -0.1030 [-0.1549, -0.0577] SEP | -0.1029 [-0.2108, -0.0183] SEP | -0.0412 [-0.0670, -0.0202] SEP | -0.0411 [-0.0931, -0.0007] SEP |
| REST (all non-extra) | 342 | -0.1337 [-0.2233, -0.0658] SEP | -0.0902 [-0.1408, -0.0485] SEP | -0.0835 [-0.1708, -0.0192] SEP | -0.0475 [-0.0754, -0.0249] SEP | -0.0411 [-0.0931, -0.0007] SEP |

r8 os slope EXTRA - REST: -0.4307 [-0.6584, -0.2113] SEP

r7 classes: r7 agrees with oracle anchor: n 104, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 260, contrib. to r7 sel. cost -0.0495 [-0.1151, +0.0008]

**EMA-10.5k | s1** -- n_win 364; frac r8-extra 0.047; full GAP -0.0650; r8 selection cost -0.0534; r7 selection cost -0.0390

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 17 | -0.6142 [-0.9333, -0.2205] SEP | -0.2152 [-0.2803, -0.1090] SEP | -0.2541 [-0.3418, -0.1518] SEP | -0.0183 [-0.0485, -0.0004] SEP | -0.0202 [-0.0522, -0.0025] SEP |
| IN-BANK agrees with oracle anchor | 103 | -0.0632 [-0.1214, -0.0111] SEP | -0.0632 [-0.1214, -0.0111] SEP | -0.0265 [-0.0853, +0.0269] | -0.0103 [-0.0182, -0.0025] SEP | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 244 | -0.1549 [-0.2525, -0.0735] SEP | -0.1053 [-0.1554, -0.0634] SEP | -0.1005 [-0.1930, -0.0234] SEP | -0.0365 [-0.0601, -0.0177] SEP | -0.0332 [-0.0840, +0.0061] |
| REST (all non-extra) | 347 | -0.1279 [-0.2094, -0.0643] SEP | -0.0929 [-0.1421, -0.0528] SEP | -0.0787 [-0.1562, -0.0173] SEP | -0.0467 [-0.0746, -0.0255] SEP | -0.0332 [-0.0840, +0.0061] |

r8 os slope EXTRA - REST: -0.4863 [-0.7867, -0.1141] SEP

r7 classes: r7 agrees with oracle anchor: n 103, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 261, contrib. to r7 sel. cost -0.0390 [-0.0961, +0.0057]

**EMA-5k | s0** -- n_win 364; frac r8-extra 0.080; full GAP -0.0348; r8 selection cost -0.0462; r7 selection cost -0.0495

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 29 | -0.2621 [-0.5083, -0.0494] SEP | -0.1145 [-0.1729, -0.0534] SEP | -0.1792 [-0.3140, -0.0096] SEP | -0.0082 [-0.0302, +0.0088] | -0.0146 [-0.0370, +0.0028] |
| IN-BANK agrees with oracle anchor | 98 | -0.0643 [-0.1153, -0.0182] SEP | -0.0643 [-0.1153, -0.0182] SEP | -0.0396 [-0.0919, +0.0136] | -0.0068 [-0.0115, -0.0023] SEP | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 237 | -0.1406 [-0.2542, -0.0515] SEP | -0.0901 [-0.1444, -0.0454] SEP | -0.1089 [-0.2214, -0.0217] SEP | -0.0198 [-0.0285, -0.0119] SEP | -0.0316 [-0.0883, +0.0097] |
| REST (all non-extra) | 335 | -0.1173 [-0.2054, -0.0519] SEP | -0.0822 [-0.1315, -0.0432] SEP | -0.0877 [-0.1748, -0.0219] SEP | -0.0266 [-0.0366, -0.0175] SEP | -0.0316 [-0.0883, +0.0097] |

r8 os slope EXTRA - REST: -0.1448 [-0.3417, +0.0255]

r7 classes: r7 agrees with oracle anchor: n 104, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 260, contrib. to r7 sel. cost -0.0495 [-0.1151, +0.0008]

**EMA-5k | s1** -- n_win 364; frac r8-extra 0.055; full GAP -0.0415; r8 selection cost -0.0417; r7 selection cost -0.0390

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 20 | -0.3898 [-0.7537, -0.0772] SEP | -0.1307 [-0.1954, -0.0570] SEP | -0.1369 [-0.2493, -0.0175] SEP | -0.0181 [-0.0481, -0.0015] SEP | -0.0186 [-0.0536, +0.0017] |
| IN-BANK agrees with oracle anchor | 101 | -0.0607 [-0.1123, -0.0179] SEP | -0.0607 [-0.1123, -0.0179] SEP | -0.0390 [-0.1015, +0.0132] | -0.0062 [-0.0139, +0.0034] | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 243 | -0.1306 [-0.2257, -0.0492] SEP | -0.0945 [-0.1457, -0.0531] SEP | -0.1038 [-0.1976, -0.0250] SEP | -0.0172 [-0.0282, -0.0066] SEP | -0.0232 [-0.0727, +0.0144] |
| REST (all non-extra) | 344 | -0.1089 [-0.1876, -0.0485] SEP | -0.0840 [-0.1316, -0.0450] SEP | -0.0837 [-0.1625, -0.0223] SEP | -0.0234 [-0.0369, -0.0099] SEP | -0.0232 [-0.0727, +0.0144] |

r8 os slope EXTRA - REST: -0.2809 [-0.5996, -0.0003] SEP

r7 classes: r7 agrees with oracle anchor: n 103, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 261, contrib. to r7 sel. cost -0.0390 [-0.0961, +0.0057]

**RAW-5k | s0** -- n_win 361; frac r8-extra 0.061; full GAP -0.1370; r8 selection cost -0.0699; r7 selection cost -0.0487

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 22 | -0.5227 [-0.8072, -0.2316] SEP | -0.2263 [-0.3348, -0.1283] SEP | -0.1812 [-0.3770, -0.0384] SEP | -0.0207 [-0.0447, -0.0032] SEP | -0.0180 [-0.0364, -0.0040] SEP |
| IN-BANK agrees with oracle anchor | 63 | -0.0809 [-0.1468, -0.0090] SEP | -0.0809 [-0.1468, -0.0090] SEP | -0.0199 [-0.0849, +0.0375] | -0.0110 [-0.0193, -0.0035] SEP | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 276 | -0.2449 [-0.3464, -0.1584] SEP | -0.1766 [-0.2310, -0.1255] SEP | -0.1063 [-0.2083, -0.0271] SEP | -0.1053 [-0.1475, -0.0677] SEP | -0.0519 [-0.1104, -0.0067] SEP |
| REST (all non-extra) | 339 | -0.2136 [-0.3082, -0.1355] SEP | -0.1583 [-0.2140, -0.1073] SEP | -0.0898 [-0.1801, -0.0219] SEP | -0.1163 [-0.1626, -0.0744] SEP | -0.0519 [-0.1104, -0.0067] SEP |

r8 os slope EXTRA - REST: -0.3092 [-0.5718, -0.0573] SEP

r7 classes: r7 agrees with oracle anchor: n 103, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 258, contrib. to r7 sel. cost -0.0487 [-0.1143, +0.0014]

**RAW-5k | s1** -- n_win 361; frac r8-extra 0.039; full GAP -0.1494; r8 selection cost -0.0704; r7 selection cost -0.0382

| r8 selection class | n_win | slope r8 os | slope r8 orc | slope r7 os | contrib. to GAP | contrib. to r8 sel. cost |
|---|---|---|---|---|---|---|
| EXTRA (r8 sel_idx>=117) | 14 | -0.3049 [-0.5371, -0.1417] SEP | -0.1089 [-0.2570, +0.0207] | -0.1041 [-0.2337, +0.0460] | -0.0074 [-0.0196, -0.0001] SEP | -0.0072 [-0.0171, -0.0007] SEP |
| IN-BANK agrees with oracle anchor | 68 | -0.1137 [-0.2067, -0.0365] SEP | -0.1137 [-0.2067, -0.0365] SEP | -0.0338 [-0.1133, +0.0327] | -0.0159 [-0.0257, -0.0075] SEP | +0.0000 [+0.0000, +0.0000] |
| IN-BANK disagrees with oracle anchor | 279 | -0.2642 [-0.3862, -0.1596] SEP | -0.1814 [-0.2321, -0.1349] SEP | -0.0991 [-0.1898, -0.0253] SEP | -0.1261 [-0.1760, -0.0804] SEP | -0.0632 [-0.1388, -0.0073] SEP |
| REST (all non-extra) | 347 | -0.2330 [-0.3425, -0.1414] SEP | -0.1674 [-0.2224, -0.1171] SEP | -0.0856 [-0.1664, -0.0202] SEP | -0.1420 [-0.1948, -0.0948] SEP | -0.0632 [-0.1388, -0.0073] SEP |

r8 os slope EXTRA - REST: -0.0719 [-0.3294, +0.1280]

r7 classes: r7 agrees with oracle anchor: n 103, contrib. to r7 sel. cost +0.0000 [+0.0000, +0.0000]; r7 disagrees: n 258, contrib. to r7 sel. cost -0.0382 [-0.0962, +0.0071]


## 3. Nav: does refcv8 use the turn signal less? (EMA-15k vs refcv7; last segment)

### seed0

**sharp** -- n_win 358, n_ep 42

| quantity | est [95 % CI] |
|---|---|
| slope r8 os | -0.1393 [-0.2431, -0.0585] SEP |
| slope r8 os_navzero | -0.1822 [-0.2679, -0.1067] SEP |
| slope r8 os_navflip | -0.2963 [-0.4079, -0.1936] SEP |
| slope r8 os_navshuf | -0.2050 [-0.3156, -0.1156] SEP |
| slope r7 os | -0.0968 [-0.1864, -0.0253] SEP |
| slope r7 os_navzero | -0.1434 [-0.2227, -0.0780] SEP |
| slope r7 os_navflip | -0.1706 [-0.2625, -0.0940] SEP |
| slope r7 os_navshuf | -0.1177 [-0.1982, -0.0516] SEP |
| r8 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0430 [-0.0277, +0.0962] |
| r8 flip response: slope(os) - slope(navflip) | +0.1570 [+0.0501, +0.2648] SEP |
| r8 shuffle response: slope(os) - slope(navshuf) | +0.0657 [+0.0388, +0.0954] SEP |
| r7 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0467 [+0.0137, +0.0793] SEP |
| r7 flip response: slope(os) - slope(navflip) | +0.0738 [+0.0049, +0.1481] SEP |
| r7 shuffle response: slope(os) - slope(navshuf) | +0.0209 [-0.0013, +0.0488] |
| r8 - r7: nav response | -0.0037 [-0.0520, +0.0361] |
| r8 - r7: flip response | +0.0832 [+0.0289, +0.1446] SEP |
| r8 - r7: slope(os) in this valid set | -0.0425 [-0.0793, -0.0123] SEP |
| r8 - r7: slope(navzero)  [the nav-free plan] | -0.0388 [-0.0828, +0.0026] |

**sharp & nav=left/right** -- n_win 157, n_ep 22

| quantity | est [95 % CI] |
|---|---|
| slope r8 os | -0.1924 [-0.3473, -0.0658] SEP |
| slope r8 os_navzero | -0.2645 [-0.3810, -0.1619] SEP |
| slope r8 os_navflip | -0.4738 [-0.6021, -0.3575] SEP |
| slope r8 os_navshuf | -0.3040 [-0.4573, -0.1755] SEP |
| slope r7 os | -0.1561 [-0.2843, -0.0442] SEP |
| slope r7 os_navzero | -0.2164 [-0.3275, -0.1212] SEP |
| slope r7 os_navflip | -0.2863 [-0.4051, -0.1844] SEP |
| slope r7 os_navshuf | -0.1834 [-0.2959, -0.0772] SEP |
| r8 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0720 [-0.0568, +0.1627] |
| r8 flip response: slope(os) - slope(navflip) | +0.2814 [+0.1033, +0.4399] SEP |
| r8 shuffle response: slope(os) - slope(navshuf) | +0.1116 [+0.0724, +0.1553] SEP |
| r7 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0604 [+0.0003, +0.1190] SEP |
| r7 flip response: slope(os) - slope(navflip) | +0.1302 [+0.0067, +0.2562] SEP |
| r7 shuffle response: slope(os) - slope(navshuf) | +0.0273 [-0.0119, +0.0793] |
| r8 - r7: nav response | +0.0117 [-0.0761, +0.0809] |
| r8 - r7: flip response | +0.1512 [+0.0596, +0.2467] SEP |
| r8 - r7: slope(os) in this valid set | -0.0364 [-0.0782, +0.0000] |
| r8 - r7: slope(navzero)  [the nav-free plan] | -0.0480 [-0.1202, +0.0212] |

**sharp & nav=follow** -- n_win 201, n_ep 20

| quantity | est [95 % CI] |
|---|---|
| slope r8 os | -0.0724 [-0.1678, -0.0023] SEP |
| slope r8 os_navzero | -0.0788 [-0.1748, -0.0101] SEP |
| slope r8 os_navflip | -0.0731 [-0.1694, -0.0026] SEP |
| slope r8 os_navshuf | -0.0805 [-0.1670, -0.0129] SEP |
| slope r7 os | -0.0222 [-0.0969, +0.0377] |
| slope r7 os_navzero | -0.0516 [-0.1270, +0.0089] |
| slope r7 os_navflip | -0.0252 [-0.1007, +0.0336] |
| slope r7 os_navshuf | -0.0351 [-0.1157, +0.0246] |
| r8 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0064 [-0.0078, +0.0208] |
| r8 flip response: slope(os) - slope(navflip) | +0.0007 [-0.0050, +0.0056] |
| r8 shuffle response: slope(os) - slope(navshuf) | +0.0081 [-0.0032, +0.0223] |
| r7 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0294 [+0.0194, +0.0410] SEP |
| r7 flip response: slope(os) - slope(navflip) | +0.0030 [-0.0004, +0.0071] |
| r7 shuffle response: slope(os) - slope(navshuf) | +0.0129 [+0.0047, +0.0230] SEP |
| r8 - r7: nav response | -0.0230 [-0.0439, -0.0027] SEP |
| r8 - r7: flip response | -0.0023 [-0.0093, +0.0036] |
| r8 - r7: slope(os) in this valid set | -0.0502 [-0.1207, -0.0023] SEP |
| r8 - r7: slope(navzero)  [the nav-free plan] | -0.0272 [-0.0877, +0.0138] |

**curved** -- n_win 1555, n_ep 100

| quantity | est [95 % CI] |
|---|---|
| slope r8 os | -0.1301 [-0.2179, -0.0614] SEP |
| slope r8 os_navzero | -0.1750 [-0.2469, -0.1096] SEP |
| slope r8 os_navflip | -0.2784 [-0.3750, -0.1891] SEP |
| slope r8 os_navshuf | -0.1955 [-0.2873, -0.1206] SEP |
| slope r7 os | -0.0869 [-0.1628, -0.0273] SEP |
| slope r7 os_navzero | -0.1361 [-0.2062, -0.0796] SEP |
| slope r7 os_navflip | -0.1656 [-0.2489, -0.0987] SEP |
| slope r7 os_navshuf | -0.1145 [-0.1852, -0.0572] SEP |
| r8 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0449 [-0.0133, +0.0896] |
| r8 flip response: slope(os) - slope(navflip) | +0.1482 [+0.0591, +0.2407] SEP |
| r8 shuffle response: slope(os) - slope(navshuf) | +0.0653 [+0.0401, +0.0935] SEP |
| r7 nav response: slope(os) - slope(navzero)  [+ = nav makes the plan turn MORE] | +0.0492 [+0.0192, +0.0818] SEP |
| r7 flip response: slope(os) - slope(navflip) | +0.0787 [+0.0177, +0.1469] SEP |
| r7 shuffle response: slope(os) - slope(navshuf) | +0.0276 [+0.0061, +0.0551] SEP |
| r8 - r7: nav response | -0.0044 [-0.0446, +0.0304] |
| r8 - r7: flip response | +0.0695 [+0.0236, +0.1205] SEP |
| r8 - r7: slope(os) in this valid set | -0.0433 [-0.0751, -0.0172] SEP |
| r8 - r7: slope(navzero)  [the nav-free plan] | -0.0389 [-0.0767, -0.0025] SEP |

### seed1

nav arms not present for this seed (all-NaN) -- skipped


runtime 4 s; free commit at start/end 6.87/5.98 GB.

---

## 4. Mechanism verdict (appended after reading sections 0-3; tables above are script-generated)

**MEASURED (this artifact, S2 panels, 4,754 windows / 139 episodes; sharp turns = 363 windows from 42 episodes; one training run per arm; CIs = episode-cluster bootstrap, models fixed)**

1. Control PASS: EMA-15k sharp last-segment slope -0.140937 and refcv7 -0.092130 reproduce through the original `hd.shrinkage` (diff 0.0; CI diff 0.0).
2. **EMA-15k vs refcv7, 2-seed mean, sharp, last segment.** GAP -0.0488 [-0.0867, -0.0176] = **CANDIDATE -0.0347 [-0.0527, -0.0183] (separated, point share 0.71)** + **SELECTION -0.0141 [-0.0430, +0.0102] (NOT separated, point share 0.29)**. The share CIs are wide (candidate [0.43, 1.51], selection [-0.51, 0.57]) because they are ratios of correlated draws. Per inference seed CANDIDATE is stable (-0.0344 / -0.0349) while SELECTION moves (-0.0085 / -0.0198). Same split on segments 2..K (-0.0372 / -0.0111) and on all curved windows (-0.0348 / -0.0136).
3. **The oracle ceiling itself under-steers, in both models**: r8 `oracle_sel` slope -0.0826, r7 -0.0479. A perfect selector would leave r8 at -0.083 (59 % of its -0.141). Selection costs are similar: r8 -0.0583 [-0.1278, -0.0067], r7 -0.0442 [-0.1056, +0.0030].
4. **Trend (same decomposition, 2-seed, sharp last segment):** EMA-5k GAP -0.0382 = CAND -0.0384 + SEL +0.0003; EMA-10.5k -0.0626 = -0.0503 + -0.0123; EMA-15k as above; RAW-5k -0.1432 = -0.1165 + -0.0267. CANDIDATE is separated at all four points, SELECTION at none. (Four points, one run each: no trend exponent is quotable.)
5. **Selection detail.** `a_star` is identical for refcv7 and refcv8 on 4,754/4,754 windows (both seeds, all four trees), and the anchor/control bank is bit-identical across the two ckpts (sha256 `51f930dc...`): item 3 of the audit holds. On sharp turns `sel_agrees_oracle` is r8 0.280 vs r7 0.283 (diff -0.003 [-0.079, +0.078]); the overall deficit (0.428 vs 0.534) lives off the tight turns. Allocator extras (sel_idx >= 117) are chosen on 6.8 % / 4.3 % of sharp windows (23 / 15 of 363; seed 0/1) vs 12.4 % overall. Those windows are hard for every model (r7 os -0.37 / -0.33; r8 oracle -0.24 / -0.27) but r8 os is -0.62 / -0.75 (extras minus rest -0.51 [-0.76, -0.20] / -0.63 [-0.87, -0.25]). They carry -0.0156 / -0.0172 of the GAP numerator (about 36 % / 31 %, separated) and about 41 % / 33 % of r8's selection cost; the rest of the selection cost is in-bank disagreements with the oracle anchor (-0.034 / -0.039), and in-bank agreements cost exactly 0.
6. **Nav (seed 0 only; nav arms were not dumped for seed 1; n = 358 sharp windows, 42 episodes).** nav response slope(os)-slope(navzero): r8 +0.043 [-0.028, +0.096], r7 +0.047 [+0.014, +0.079]; r8 - r7 -0.004 [-0.052, +0.036]. Flip response r8 +0.157 vs r7 +0.074 (diff +0.083 [+0.029, +0.145], separated, i.e. r8 reacts MORE to a reversed command). r8's nav-free plan (navzero) is -0.039 [-0.083, +0.003] worse than r7's, the size of the whole gap. No evidence that refcv8 uses the turn signal less. (Sharp & nav=follow: r8 - r7 nav response -0.023 [-0.044, -0.003]; n = 201 windows, 20 episodes; not pursued.)
7. **Anchor-control analysis is NOT interpretable and must not be quoted as 'coverage'.** corr(alat[a_star], GT lateral y@2 s) = -0.14 on all windows and -0.42 on the 295 windows with |y@2 s| > 3 m, while corr(oracle_sel y@2 s, GT y@2 s) = 0.973. So `a_star` is the best REFINED output slot; its raw bank control is not a geometric prototype of that output. The selected slot's raw trajectory (`sel_bank_nav_true`) does track GT (corr 0.889 vs 0.954 for the refined plan) but is moved by up to 5.9 m at 2 s by the refinement. The |alat| / implied-turn-rate rows in section 2a are therefore retained only as a record; they do not separate bank coverage from refinement.

**Mechanism verdict.** MEASURED: the r8-minus-r7 under-steering on tight turns is mostly in the **candidate plans** (about 70 % of the gap, separated), with **selection a smaller, unseparated ~30 %**; roughly a third of the gap sits on the 4-7 % of sharp windows where r8's selector takes an allocator extra. Under-steering is also present in BOTH models' oracle ceilings, so most of the defect is shared and lives in what the decoder/refinement produces, not in which candidate is picked. This is a statement about the episodes drawn and these two checkpoints (H-ESTIM-SEED-1 does not allow a statement about retraining).
**HYPOTHESIS (not tested here):** the shrinkage is introduced by the refinement (decoder) step rather than by the bank: the bank index is not geometric (item 7), and the refined plan tracks GT lateral offset well (corr 0.954) yet still under-rotates in heading. The hypothesis that the allocator extras are poorly calibrated on tight turns is supported only as 'hard windows where r8 is worse than its own oracle and than r7'.

**Cheapest next experiments (0 GPU first).**
(a) Slope of the RAW selected-slot trajectory (`sel_bank_nav_true`, in-bank windows) vs the refined `plan_full_nav_true` on the same sharp windows, first 4 instants: if raw is near -0.0 and refined is -0.14, the refinement is the shrinker (direct, ~20 lines on existing sidecars). (b) Counterfactual re-selection without extras is not possible from these dumps (only the selected plan is stored); the eval-time arm `n_alloc=0` on the dev box (GPU, cheap) removes the 36 %/31 % extras share exactly. (c) Design lever if (a) confirms: a terminal-heading / curvature term on the refined candidates weighted by GT |yaw| (tight turns), because a perfect selector alone cannot get r8 below -0.083.
