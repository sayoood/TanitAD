# S6 (0-6 s) longitudinal decomposition: refcv8 EMA-30k vs refcv7-50,400 -- where the regression lives

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**
* estimator: episode-cluster bootstrap over the 139 eval episodes, models FIXED, B 2000, seed 0 (draws identical to taniteval.ci._draws): answers 'another draw of EPISODES' only; NOT another training run (single training seed per arm), NOT another inference run
* surface: S6 0-6 s (instants 1..6 s, dt 1.0); n **3668 windows / 139 episodes** (inference seed 0 unless marked); single TRAINING seed per arm.
* arms: refcv8 `os` (ceiling filter ACTING, as banked) = **ON**; `filter_off.traj[:, slots]` (same DDIM draw, filter off) = **OFF** = the matched instrument; `vmax_zero.traj[:, slots]` = the max-speed (road-type) label WITHHELD; refcv7-50,400 = `b_refcv7_50400_s{seed}` (its filter is inert on S6, re-verified below).
* **contribution** = sum of the bin's per-window deltas / N (adds up to the total mean delta, metric units); **share** = contribution / total; **enrichment** = share / window fraction. CIs: joint episode-cluster draws (B 2000), percentile 95 %. SEP = CI excludes 0.
* signed along-track at 6 s = P_x(6 s) - G_x(6 s) in the t0 ego frame: **negative = the plan ends SHORT of GT (too slow)**, positive = overshoots (too fast).
* privileged inputs: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction).
* artifacts: trees C:/Users/Admin/refcv8_eval_archive/battery\step30000ema_unc, C:/Users/Admin/refcv8_eval_archive/battery\step20000ema_unc, C:/Users/Admin/refcv8_eval_archive/battery\step20000_unc; refcv7 D:/refcv7_eval_kit/battery/step50400_a6text; wall 18.9 s.

## Controls

* **Z (known-zero)**: refcv7 vs itself and refcv8 vs itself read exactly 0.0 (contribution, CI, bin mean) in every bin of every partition and metric: **True**.
* **R (reproduction)**: all-window deltas re-read through `taniteval.ci` equal the banked PAIRED_READ S6 cells (ON: 30k EMA s0+s1, 20k EMA, 20k RAW; OFF: NR_MATCHED_INSTRUMENT) field-for-field, and the joint-draw total equals ci's at its rounding: **True**.
* **X (independent derivations / identity)**: **True** -- numpy |signed along| vs components max abs 3.8e-06; numpy ADE vs components 2.4e-06; S6 `os` vs extras `nav_true.traj[:, slots]` 0.0; refcv7 S6 arm vs its extras 0.0; refcv7 filter changes selection on 0 S6 windows; refcv7 vs refcv8 speed label max abs 13.727778434753418; GT end-point world vs ego max abs 2.1e-05 m; window keys + GT identical across all trees and seed 1.
* **MEASURED, not a control failure -- the two arms consume DIFFERENT speed labels**: refcv7-50,400's `v_max_ms` differs from refcv8's on 3668 / 3668 S6 windows (max abs 13.73 m/s); refcv7: 136 distinct values, constant within every clip = True, mean (per-clip label minus the clip's max 6-s GT speed) -1.94 m/s; refcv8: 4 values (the 4-class road-type label).
* **P (random partition)**: v0-bin labels permuted across windows; ADE shares vs window fractions: v0<2 11 % [-4, 38] vs 9 %; v0 2-8 19 % [-29, 38] vs 28 %; v0 8-15 49 % [31, 104] vs 38 %; v0 15-25 11 % [-26, 27] vs 19 %; v0>=25 9 % [-0, 24] vs 7 %

## C0 -- THE TARGET: refcv8 EMA-30k ON minus refcv7-50,400, seed 0

### (a) GT initial speed

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| v0<2 | 330 / 29 | 9.0 | -0.0455 [-0.1018, +0.0045] | -27 % [-186, 4] | -0.5061 [-0.9928, +0.0562] | -18 % [-115, 5] | -0.962 [-2.055, +0.317] | -21 % [-132, 5] | +2.30 / +4.53 / -2.225 [-3.405, -0.952] **SEP** | 41 |
| v0 2-8 | 1017 / 71 | 27.7 | +0.0050 [-0.0506, +0.0604] | 3 % [-70, 39] | +0.0181 [-0.1810, +0.2151] | 6 % [-61, 45] | +0.107 [-0.465, +0.660] | 6 % [-54, 44] | -0.42 / +1.62 / -2.040 [-2.702, -1.343] **SEP** | 54 |
| v0 8-15 | 1397 / 83 | 38.1 | +0.1267 [+0.0338, +0.2248] **SEP** | 76 % [30, 198] | +0.3327 [+0.0915, +0.5719] **SEP** | 67 % [20, 158] | +0.834 [+0.168, +1.499] **SEP** | 75 % [28, 175] | -1.06 / +1.95 / -3.016 [-3.874, -2.179] **SEP** | 56 |
| v0 15-25 | 679 / 34 | 18.5 | +0.0727 [+0.0106, +0.1502] **SEP** | 44 % [7, 150] | +0.3927 [+0.0633, +0.7792] **SEP** | 41 % [2, 124] | +1.043 [+0.090, +2.190] **SEP** | 37 % [-1, 116] | -2.40 / -0.53 / -1.868 [-3.104, -0.895] **SEP** | 65 |
| v0>=25 | 245 / 11 | 6.7 | +0.0069 [-0.0059, +0.0250] | 4 % [-5, 22] | +0.1033 [-0.0975, +0.4807] | 4 % [-5, 20] | +0.266 [-0.260, +1.193] | 3 % [-6, 19] | -1.96 / -1.47 / -0.487 [-1.494, +0.262] | 69 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **v0 8-15** (cumulative point share 76 %). |along| 6 s: **v0 8-15** (67 %).

### (a) GT speed change over 6 s

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| start_from_stop | 101 / 15 | 2.8 | +0.0046 [-0.0136, +0.0235] | 3 % [-15, 28] | +0.1667 [-0.5332, +0.8927] | 2 % [-11, 20] | +0.370 [-1.159, +1.913] | 3 % [-9, 23] | -2.03 / +0.20 / -2.225 [-3.628, -0.721] **SEP** | 61 |
| stopping | 165 / 23 | 4.5 | -0.0115 [-0.0306, +0.0101] | -7 % [-47, 7] | -0.2552 [-0.6972, +0.2441] | -5 % [-35, 10] | -0.507 [-1.699, +0.777] | -5 % [-37, 9] | +7.49 / +8.20 / -0.706 [-1.980, +0.705] | 16 |
| slow_both | 165 / 21 | 4.5 | -0.0408 [-0.0926, +0.0014] | -25 % [-179, 2] | -0.9077 [-1.5056, +0.0509] | -17 % [-100, 3] | -1.804 [-3.318, +0.443] | -20 % [-129, 3] | +5.30 / +7.53 / -2.224 [-3.924, +0.329] | 32 |
| accelerating | 781 / 76 | 21.3 | +0.0304 [-0.0222, +0.0808] | 18 % [-22, 75] | +0.1430 [-0.1058, +0.3755] | 19 % [-13, 71] | +0.429 [-0.135, +0.945] | 15 % [-12, 59] | -4.01 / -2.64 / -1.371 [-2.006, -0.654] **SEP** | 73 |
| decelerating | 717 / 83 | 19.5 | -0.0268 [-0.0845, +0.0346] | -16 % [-148, 20] | -0.1370 [-0.4362, +0.1809] | -35 % [-217, 1] | -0.852 [-1.616, -0.070] **SEP** | -24 % [-152, 10] | +2.51 / +6.26 / -3.747 [-4.668, -2.875] **SEP** | 38 |
| cruising | 1739 / 128 | 47.4 | +0.2099 [+0.1089, +0.3179] **SEP** | 127 % [77, 390] | +0.4427 [+0.2346, +0.6579] **SEP** | 136 % [88, 375] | +1.350 [+0.743, +2.006] **SEP** | 132 % [86, 339] | -2.21 / +0.06 / -2.268 [-2.999, -1.565] **SEP** | 63 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **cruising** (cumulative point share 127 %). |along| 6 s: **cruising** (136 %).

### (b) sign of the along-track error at 6 s (new = refcv8, ref = refcv7; OUTCOME-defined)

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| new slow / ref slow | 1537 / 136 | 41.9 | +0.2674 [+0.1752, +0.3602] **SEP** | 161 % [99, 536] | +0.6381 [+0.4304, +0.8469] **SEP** | 167 % [107, 506] | +1.876 [+1.303, +2.463] **SEP** | 153 % [99, 463] | -6.37 / -4.50 / -1.876 [-2.463, -1.303] **SEP** | 100 |
| new slow / ref fast | 542 / 114 | 14.8 | +0.0772 [+0.0216, +0.1300] **SEP** | 47 % [20, 126] | +0.5226 [+0.1595, +0.8557] **SEP** | 37 % [8, 82] | +1.181 [+0.170, +2.065] **SEP** | 44 % [20, 96] | -4.75 / +3.57 / -8.327 [-9.120, -7.501] **SEP** | 100 |
| new fast / ref slow | 129 / 67 | 3.5 | +0.0025 [-0.0119, +0.0187] | 2 % [-15, 18] | +0.0713 [-0.3599, +0.5197] | 2 % [-10, 15] | +0.227 [-0.775, +1.369] | -0 % [-15, 12] | +3.25 / -3.03 / +6.281 [+5.219, +7.348] **SEP** | 0 |
| new fast / ref fast | 1460 / 131 | 39.8 | -0.1813 [-0.2410, -0.1159] **SEP** | -109 % [-555, -39] | -0.4555 [-0.5980, -0.2954] **SEP** | -105 % [-477, -41] | -1.249 [-1.622, -0.863] **SEP** | -97 % [-441, -37] | +5.95 / +7.20 / -1.249 [-1.622, -0.863] **SEP** | 0 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **new slow / ref slow** (cumulative point share 161 %). |along| 6 s: **new slow / ref slow** (167 %).

### (c) does the max-speed ceiling filter act (refcv8 EMA-30k)

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| ceiling ACTS (ema30k) | 239 / 50 | 6.5 | +0.0413 [-0.0076, +0.0954] | 25 % [-8, 93] | +0.6346 [-0.1293, +1.4603] | 22 % [-6, 75] | +1.559 [-0.233, +3.435] | 20 % [-8, 70] | -1.59 / +5.13 / -6.726 [-8.527, -5.028] **SEP** | 54 |
| ceiling inert (ema30k) | 3429 / 139 | 93.5 | +0.1245 [-0.0030, +0.2650] | 75 % [7, 108] | +0.1331 [-0.0032, +0.2833] | 78 % [25, 106] | +0.395 [+0.010, +0.811] **SEP** | 80 % [30, 108] | -0.84 / +1.14 / -1.984 [-2.458, -1.511] **SEP** | 57 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **ceiling inert (ema30k)** (cumulative point share 75 %). |along| 6 s: **ceiling inert (ema30k)** (78 %).

### (c) road-type max-speed label class

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 km/h (8.33) | 1246 / 47 | 34.0 | -0.0431 [-0.1265, +0.0386] | -26 % [-253, 21] | -0.1268 [-0.3580, +0.1227] | -15 % [-164, 28] | -0.206 [-0.823, +0.437] | -17 % [-185, 25] | +0.58 / +3.39 / -2.806 [-3.530, -2.080] **SEP** | 48 |
| 50 km/h (13.89) | 1291 / 49 | 35.2 | +0.1284 [+0.0538, +0.2212] **SEP** | 77 % [32, 265] | +0.3649 [+0.1660, +0.6050] **SEP** | 69 % [25, 197] | +0.928 [+0.317, +1.646] **SEP** | 72 % [30, 206] | -1.31 / +1.19 / -2.493 [-3.334, -1.638] **SEP** | 59 |
| 100 km/h (27.78) | 947 / 36 | 25.8 | +0.0792 [+0.0012, +0.1660] **SEP** | 48 % [-2, 137] | +0.3068 [+0.0042, +0.6317] **SEP** | 46 % [-3, 123] | +0.847 [-0.002, +1.809] | 46 % [-0, 122] | -1.98 / -0.22 / -1.763 [-2.826, -0.840] **SEP** | 63 |
| 120 km/h (33.33) | 184 / 7 | 5.0 | +0.0012 [-0.0073, +0.0099] | 1 % [-7, 10] | +0.0247 [-0.1557, +0.2160] | -1 % [-9, 4] | -0.083 [-0.492, +0.221] | -1 % [-9, 4] | -2.32 / -2.19 / -0.130 [-0.920, +0.644] | 70 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **50 km/h (13.89)** (cumulative point share 77 %). |along| 6 s: **50 km/h (13.89)** (69 %).

### (c) GT max speed over 6 s relative to the label

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| GT_above_label | 417 / 43 | 11.4 | +0.0606 [+0.0124, +0.1115] **SEP** | 37 % [7, 131] | +0.5329 [+0.1221, +0.9770] **SEP** | 30 % [3, 103] | +1.238 [+0.203, +2.315] **SEP** | 28 % [3, 103] | -2.84 / +1.91 / -4.755 [-6.089, -3.400] **SEP** | 65 |
| GT_near_label | 1821 / 105 | 49.6 | +0.1362 [+0.0467, +0.2421] **SEP** | 82 % [37, 224] | +0.2743 [+0.0981, +0.4737] **SEP** | 83 % [39, 201] | +0.792 [+0.258, +1.395] **SEP** | 84 % [39, 188] | -1.98 / +0.16 / -2.138 [-2.824, -1.492] **SEP** | 64 |
| GT_well_below_label | 1430 / 91 | 39.0 | -0.0310 [-0.1178, +0.0638] | -19 % [-247, 31] | -0.0794 [-0.2979, +0.1659] | -13 % [-176, 35] | -0.161 [-0.707, +0.494] | -12 % [-176, 35] | +1.07 / +2.84 / -1.772 [-2.467, -1.074] **SEP** | 46 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **GT_near_label** (cumulative point share 82 %). |along| 6 s: **GT_near_label** (83 %).

### (c) speed-label INFORMATION gap: refcv8 4-class value minus refcv7's per-clip continuous value

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| gap<0 | 26 / 1 | 0.7 | +0.0001 [+0.0000, +0.0002] | 0 % [0, 0] | +0.0074 [+0.0074, +0.0074] **SEP** | 0 % [0, 2] | +0.205 [+0.205, +0.205] **SEP** | 0 % [0, 2] | -0.27 / +0.06 / -0.327 [-0.327, -0.327] **SEP** | 62 |
| gap 0-2 | 1082 / 41 | 29.5 | +0.0834 [+0.0050, +0.1710] **SEP** | 50 % [3, 144] | +0.2828 [+0.0171, +0.5558] **SEP** | 53 % [7, 143] | +0.855 [+0.126, +1.638] **SEP** | 58 % [16, 163] | -2.67 / +0.31 / -2.979 [-3.865, -2.129] **SEP** | 65 |
| gap 2-5 | 1401 / 53 | 38.2 | +0.0415 [-0.0283, +0.1180] | 25 % [-33, 93] | +0.1086 [-0.0780, +0.3131] | 24 % [-38, 80] | +0.299 [-0.248, +0.869] | 21 % [-39, 71] | -0.12 / +1.91 / -2.031 [-2.705, -1.361] **SEP** | 53 |
| gap 5-10 | 711 / 27 | 19.4 | +0.0239 [-0.0446, +0.1053] | 14 % [-60, 73] | +0.1234 [-0.2418, +0.5379] | 11 % [-52, 63] | +0.276 [-0.613, +1.365] | 5 % [-74, 54] | +0.17 / +2.06 / -1.891 [-3.140, -0.715] **SEP** | 51 |
| gap>=10 | 448 / 17 | 12.2 | +0.0169 [-0.0403, +0.0816] | 10 % [-53, 54] | +0.1386 [-0.3371, +0.7120] | 11 % [-45, 55] | +0.408 [-0.881, +2.065] | 15 % [-35, 62] | -0.70 / +1.50 / -2.204 [-4.051, -0.639] **SEP** | 55 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **gap 0-2** (cumulative point share 50 %). |along| 6 s: **gap 0-2** (53 %).

### GT tactical longitudinal label (lon_v7)

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| unlabelled (-100) | 259 / 22 | 7.1 | -0.0133 [-0.0749, +0.0562] | -8 % [-139, 29] | -0.1877 [-1.0573, +0.7720] | 2 % [-59, 39] | +0.143 [-1.888, +2.665] | -3 % [-85, 34] | +1.18 / +4.03 / -2.854 [-4.853, -0.859] **SEP** | 46 |
| 0 FOLLOW | 676 / 38 | 18.4 | -0.0014 [-0.0274, +0.0263] | -1 % [-34, 22] | -0.0076 [-0.1605, +0.1439] | -2 % [-27, 16] | -0.041 [-0.369, +0.287] | -0 % [-24, 17] | -1.49 / -0.51 / -0.981 [-1.648, -0.357] **SEP** | 57 |
| 1 CRUISE | 1139 / 92 | 31.1 | +0.1574 [+0.0737, +0.2476] **SEP** | 95 % [51, 308] | +0.5070 [+0.2484, +0.7747] **SEP** | 103 % [60, 300] | +1.571 [+0.809, +2.376] **SEP** | 103 % [61, 273] | -2.43 / -0.08 / -2.358 [-3.278, -1.480] **SEP** | 67 |
| 3 BRAKE_TO | 809 / 77 | 22.1 | +0.0101 [-0.0519, +0.0760] | 6 % [-87, 47] | +0.0458 [-0.2376, +0.3432] | -6 % [-114, 27] | -0.133 [-0.862, +0.610] | -2 % [-85, 32] | +2.39 / +5.94 / -3.550 [-4.444, -2.641] **SEP** | 39 |
| 4 CREEP | 30 / 14 | 0.8 | -0.0000 [-0.0050, +0.0047] | -0 % [-5, 5] | -0.0030 [-0.6236, +0.6456] | -0 % [-4, 4] | -0.062 [-1.555, +1.383] | 0 % [-4, 4] | +2.33 / +2.39 / -0.057 [-1.617, +1.575] | 23 |
| 5 HOLD | 31 / 6 | 0.8 | -0.0038 [-0.0150, +0.0035] | -2 % [-21, 3] | -0.4465 [-1.7895, +1.2078] | -2 % [-19, 3] | -1.333 [-5.259, +2.988] | -3 % [-21, 3] | +1.32 / +2.81 / -1.494 [-5.518, +2.505] | 45 |
| 7 ACCELERATE | 724 / 69 | 19.7 | +0.0167 [-0.0389, +0.0698] | 10 % [-48, 56] | +0.0848 [-0.2002, +0.3625] | 5 % [-57, 39] | +0.111 [-0.578, +0.787] | 5 % [-33, 35] | -2.53 / -0.59 / -1.936 [-2.642, -1.211] **SEP** | 65 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **1 CRUISE** (cumulative point share 95 %). |along| 6 s: **1 CRUISE** (103 %).

### (a) GT initial speed x speed change: the 8 largest ADE contributions (C0)

| cell | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s contrib [CI] m | signed along: refcv8 / refcv7 m |
|---|---|---|---|---|---|---|---|
| v0 8-15 | cruising | 755 / 65 | 20.6 | +0.1421 [+0.0691, +0.2248] **SEP** | 86 % [41, 267] | +0.6903 [+0.3707, +1.0522] **SEP** | +0.4151 [+0.1984, +0.6609] **SEP** | -2.85 / +0.10 |
| v0 15-25 | cruising | 426 / 29 | 11.6 | +0.0595 [+0.0071, +0.1288] **SEP** | 36 % [4, 128] | +0.5121 [+0.0666, +1.0591] **SEP** | +0.1694 [+0.0186, +0.3656] **SEP** | -2.93 / -1.00 |
| v0 2-8 | accelerating | 412 / 47 | 11.2 | +0.0207 [-0.0048, +0.0486] | 12 % [-6, 55] | +0.1844 [-0.0454, +0.4174] | +0.0495 [-0.0158, +0.1190] | -3.32 / -1.70 |
| v0 15-25 | accelerating | 125 / 11 | 3.4 | +0.0152 [+0.0032, +0.0325] **SEP** | 9 % [1, 40] | +0.4456 [+0.1105, +0.8971] **SEP** | +0.0351 [+0.0063, +0.0765] **SEP** | -7.52 / -6.51 |
| v0 2-8 | cruising | 333 / 47 | 9.1 | +0.0101 [-0.0240, +0.0477] | 6 % [-31, 42] | +0.1116 [-0.3116, +0.4672] | +0.0555 [-0.0386, +0.1611] | -0.73 / +1.75 |
| v0>=25 | accelerating | 24 / 4 | 0.7 | +0.0077 [-0.0001, +0.0228] | 5 % [-0, 24] | +1.1826 [-0.2184, +6.6014] | +0.0164 [-0.0021, +0.0523] | -8.20 / -6.02 |
| v0<2 | start_from_stop | 101 / 15 | 2.8 | +0.0046 [-0.0136, +0.0235] | 3 % [-15, 28] | +0.1667 [-0.5332, +0.8927] | +0.0102 [-0.0294, +0.0524] | -2.03 / +0.20 |
| v0 8-15 | stopping | 49 / 13 | 1.3 | +0.0016 [-0.0084, +0.0162] | 1 % [-10, 13] | +0.1180 [-0.7114, +1.1868] | +0.0024 [-0.0217, +0.0392] | +10.24 / +10.38 |

## The same decomposition on the other contrasts (ADE contribution per bin, share in brackets)

### C0s1: refcv8 EMA-30k ON minus refcv7-50,400 (inference seed 1 replicate)

* **v0bin** ADE: total +0.1895 [+0.0464, +0.3391]; v0<2: -0.0404 (-21 % [-119, 3]); v0 2-8: +0.0132 (7 % [-55, 39]); v0 8-15: +0.1425 SEP (75 % [37, 174]); v0 15-25: +0.0708 SEP (37 % [4, 102]); v0>=25: +0.0034 (2 % [-8, 13])
  * |along| 6 s: total +0.4998 [+0.1148, +0.8971]; v0<2: -0.0760 (-15 % [-99, 5]); v0 2-8: +0.0406 (8 % [-54, 43]); v0 8-15: +0.3311 SEP (66 % [23, 151]); v0 15-25: +0.1953 SEP (39 % [2, 110]); v0>=25: +0.0088 (2 % [-7, 12])
  * signed along 6 s (bin mean delta): v0<2 -2.222 [-3.386, -0.963] **SEP**; v0 2-8 -1.865 [-2.546, -1.174] **SEP**; v0 8-15 -2.981 [-3.827, -2.146] **SEP**; v0 15-25 -1.862 [-3.111, -0.840] **SEP**; v0>=25 -0.467 [-1.294, +0.176]
* **dvclass** ADE: total +0.1895 [+0.0464, +0.3391]; start_from_stop: +0.0047 (2 % [-8, 18]); stopping: -0.0045 (-2 % [-24, 11]); slow_both: -0.0374 (-20 % [-109, 2]); accelerating: +0.0410 (22 % [-8, 65]); decelerating: -0.0288 (-15 % [-104, 13]); cruising: +0.2144 SEP (113 % [74, 273])
  * |along| 6 s: total +0.4998 [+0.1148, +0.8971]; start_from_stop: +0.0088 (2 % [-8, 15]); stopping: -0.0156 (-3 % [-28, 10]); slow_both: -0.0733 (-15 % [-91, 3]); accelerating: +0.1101 (22 % [-1, 74]); decelerating: -0.1694 SEP (-34 % [-201, -3]); cruising: +0.6393 SEP (128 % [87, 329])
  * signed along 6 s (bin mean delta): start_from_stop -2.622 [-4.129, -1.009] **SEP**; stopping -0.779 [-1.922, +0.666]; slow_both -2.098 [-3.802, +0.524]; accelerating -1.098 [-1.704, -0.411] **SEP**; decelerating -3.700 [-4.618, -2.838] **SEP**; cruising -2.256 [-3.009, -1.558] **SEP**
* **label_gap** ADE: total +0.1895 [+0.0464, +0.3391]; gap<0: -0.0001 (-0 % [-0, 0]); gap 0-2: +0.0983 SEP (52 % [14, 129]); gap 2-5: +0.0452 (24 % [-24, 69]); gap 5-10: +0.0330 (17 % [-37, 65]); gap>=10: +0.0130 (7 % [-48, 41])
  * |along| 6 s: total +0.4998 [+0.1148, +0.8971]; gap<0: +0.0015 (0 % [0, 2]); gap 0-2: +0.2804 SEP (56 % [16, 145]); gap 2-5: +0.1098 (22 % [-34, 70]); gap 5-10: +0.0713 (14 % [-48, 61]); gap>=10: +0.0368 (7 % [-51, 45])
  * signed along 6 s (bin mean delta): gap<0 -0.348 [-0.348, -0.348] **SEP**; gap 0-2 -2.858 [-3.777, -1.976] **SEP**; gap 2-5 -2.004 [-2.673, -1.380] **SEP**; gap 5-10 -1.847 [-3.114, -0.666] **SEP**; gap>=10 -2.120 [-3.911, -0.565] **SEP**
* **label_vs_gt** ADE: total +0.1895 [+0.0464, +0.3391]; GT_above_label: +0.0679 SEP (36 % [13, 108]); GT_near_label: +0.1440 SEP (76 % [38, 172]); GT_well_below_label: -0.0224 (-12 % [-168, 31])
  * |along| 6 s: total +0.4998 [+0.1148, +0.8971]; GT_above_label: +0.1524 SEP (31 % [8, 103]); GT_near_label: +0.4096 SEP (82 % [43, 174]); GT_well_below_label: -0.0622 (-12 % [-156, 31])
  * signed along 6 s (bin mean delta): GT_above_label -4.539 [-6.024, -3.089] **SEP**; GT_near_label -2.128 [-2.820, -1.456] **SEP**; GT_well_below_label -1.681 [-2.358, -0.997] **SEP**

### C1: refcv8 EMA-30k OFF (ceiling filter off = matched instrument) minus refcv7-50,400

* **v0bin** ADE: total +0.1447 [+0.0154, +0.2842]; v0<2: -0.0241 (-17 % [-166, 22]); v0 2-8: +0.0227 (16 % [-57, 74]); v0 8-15: +0.0793 (55 % [-3, 156]); v0 15-25: +0.0701 SEP (48 % [3, 183]); v0>=25: -0.0033 (-2 % [-15, 3])
  * |along| 6 s: total +0.4041 [+0.0360, +0.8026]; v0<2: -0.0402 (-10 % [-108, 23]); v0 2-8: +0.0420 (10 % [-67, 64]); v0 8-15: +0.2239 (55 % [-11, 148]); v0 15-25: +0.1862 SEP (46 % [-1, 171]); v0>=25: -0.0079 (-2 % [-14, 4])
  * signed along 6 s (bin mean delta): v0<2 -1.539 [-2.963, -0.029] **SEP**; v0 2-8 -1.486 [-2.198, -0.743] **SEP**; v0 8-15 -2.331 [-3.140, -1.541] **SEP**; v0 15-25 -1.831 [-3.076, -0.855] **SEP**; v0>=25 -0.044 [-0.555, +0.517]
* **dvclass** ADE: total +0.1447 [+0.0154, +0.2842]; start_from_stop: +0.0070 (5 % [-12, 36]); stopping: -0.0107 (-7 % [-56, 10]); slow_both: -0.0262 (-18 % [-170, 13]); accelerating: +0.0218 (15 % [-27, 68]); decelerating: -0.0635 SEP (-44 % [-274, -8]); cruising: +0.2162 SEP (149 % [91, 537])
  * |along| 6 s: total +0.4041 [+0.0360, +0.8026]; start_from_stop: +0.0163 (4 % [-10, 27]); stopping: -0.0272 (-7 % [-49, 12]); slow_both: -0.0480 (-12 % [-115, 16]); accelerating: +0.0542 (13 % [-24, 64]); decelerating: -0.2378 SEP (-59 % [-373, -14]); cruising: +0.6466 SEP (160 % [98, 566])
  * signed along 6 s (bin mean delta): start_from_stop -1.585 [-3.258, +0.152]; stopping -0.812 [-2.020, +0.533]; slow_both -1.486 [-3.766, +1.307]; accelerating -0.699 [-1.282, -0.038] **SEP**; decelerating -2.822 [-3.557, -2.093] **SEP**; cruising -1.967 [-2.703, -1.264] **SEP**
* **label_gap** ADE: total +0.1447 [+0.0154, +0.2842]; gap<0: +0.0001 (0 % [0, 0]); gap 0-2: +0.0437 (30 % [-42, 107]); gap 2-5: +0.0535 (37 % [-27, 132]); gap 5-10: +0.0305 (21 % [-71, 91]); gap>=10: +0.0169 (12 % [-68, 66])
  * |along| 6 s: total +0.4041 [+0.0360, +0.8026]; gap<0: +0.0014 (0 % [0, 3]); gap 0-2: +0.1412 (35 % [-43, 114]); gap 2-5: +0.1408 (35 % [-33, 128]); gap 5-10: +0.0707 (17 % [-73, 90]); gap>=10: +0.0499 (12 % [-59, 69])
  * signed along 6 s (bin mean delta): gap<0 -0.327 [-0.327, -0.327] **SEP**; gap 0-2 -2.008 [-2.869, -1.172] **SEP**; gap 2-5 -1.556 [-2.274, -0.867] **SEP**; gap 5-10 -1.660 [-2.933, -0.486] **SEP**; gap>=10 -2.204 [-4.051, -0.639] **SEP**
* **label_vs_gt** ADE: total +0.1447 [+0.0154, +0.2842]; GT_above_label: -0.0244 SEP (-17 % [-105, 1]); GT_near_label: +0.1586 SEP (110 % [55, 390]); GT_well_below_label: +0.0105 (7 % [-184, 62])
  * |along| 6 s: total +0.4041 [+0.0360, +0.8026]; GT_above_label: -0.0518 (-13 % [-85, 6]); GT_near_label: +0.4380 SEP (108 % [53, 342]); GT_well_below_label: +0.0178 (4 % [-164, 61])
  * signed along 6 s (bin mean delta): GT_above_label -1.832 [-2.911, -0.821] **SEP**; GT_near_label -1.935 [-2.632, -1.280] **SEP**; GT_well_below_label -1.568 [-2.286, -0.823] **SEP**
* **ceiling** ADE: total +0.1447 [+0.0154, +0.2842]; ceiling ACTS (ema30k): +0.0203 SEP (14 % [-0, 72]); ceiling inert (ema30k): +0.1245 (86 % [28, 100])
  * |along| 6 s: total +0.4041 [+0.0360, +0.8026]; ceiling ACTS (ema30k): +0.0344 (9 % [-7, 45]); ceiling inert (ema30k): +0.3697 SEP (91 % [55, 107])
  * signed along 6 s (bin mean delta): ceiling ACTS (ema30k) +1.141 [+0.389, +1.928] **SEP**; ceiling inert (ema30k) -1.984 [-2.458, -1.511] **SEP**

### C2: refcv8 EMA-30k ON minus OFF (what the ceiling filter does to refcv8)

* **v0bin** ADE: total +0.0211 [-0.0397, +0.0817]; v0<2: -0.0215 SEP (-102 % [-1057, 1333]); v0 2-8: -0.0177 (-84 % [-964, 1035]); v0 8-15: +0.0474 SEP (225 % [-1739, 1546]); v0 15-25: +0.0026 (12 % [-82, 90]); v0>=25: +0.0102 (48 % [-332, 350])
  * |along| 6 s: total +0.0672 [-0.0816, +0.2110]; v0<2: -0.0463 (-69 % [-946, 809]); v0 2-8: -0.0124 (-19 % [-373, 443]); v0 8-15: +0.0936 SEP (139 % [-811, 927]); v0 15-25: +0.0067 (10 % [-58, 72]); v0>=25: +0.0256 (38 % [-228, 288])
  * signed along 6 s (bin mean delta): v0<2 -0.687 [-1.518, -0.074] **SEP**; v0 2-8 -0.554 [-0.842, -0.309] **SEP**; v0 8-15 -0.685 [-1.109, -0.354] **SEP**; v0 15-25 -0.036 [-0.118, +0.000]; v0>=25 -0.443 [-1.447, +0.000]
* **dvclass** ADE: total +0.0211 [-0.0397, +0.0817]; start_from_stop: -0.0024 (-12 % [-182, 144]); stopping: -0.0007 (-4 % [-69, 83]); slow_both: -0.0146 (-69 % [-718, 1029]); accelerating: +0.0086 (41 % [-355, 424]); decelerating: +0.0367 SEP (174 % [-1308, 1181]); cruising: -0.0064 (-30 % [-409, 443])
  * |along| 6 s: total +0.0672 [-0.0816, +0.2110]; start_from_stop: -0.0061 (-9 % [-133, 123]); stopping: +0.0044 (7 % [-55, 57]); slow_both: -0.0332 (-49 % [-724, 598]); accelerating: +0.0372 (55 % [-262, 385]); decelerating: +0.0713 SEP (106 % [-627, 766]); cruising: -0.0065 (-10 % [-264, 348])
  * signed along 6 s (bin mean delta): start_from_stop -0.640 [-1.595, +0.000]; stopping +0.106 [+0.000, +0.265] **SEP**; slow_both -0.738 [-2.162, +0.000]; accelerating -0.673 [-1.085, -0.340] **SEP**; decelerating -0.924 [-1.624, -0.369] **SEP**; cruising -0.301 [-0.454, -0.179] **SEP**
* **label_gap** ADE: total +0.0211 [-0.0397, +0.0817]; gap<0: +0.0000 (0 % [0, 0]); gap 0-2: +0.0397 SEP (188 % [-1525, 1341]); gap 2-5: -0.0120 (-57 % [-850, 948]); gap 5-10: -0.0066 (-31 % [-583, 755]); gap>=10: +0.0000 (0 % [0, 0])
  * |along| 6 s: total +0.0672 [-0.0816, +0.2110]; gap<0: +0.0000 (0 % [0, 0]); gap 0-2: +0.1109 SEP (165 % [-1033, 1225]); gap 2-5: -0.0265 (-39 % [-680, 686]); gap 5-10: -0.0171 (-26 % [-528, 494]); gap>=10: +0.0000 (0 % [0, 0])
  * signed along 6 s (bin mean delta): gap<0 +0.000 [+0.000, +0.000]; gap 0-2 -0.971 [-1.368, -0.598] **SEP**; gap 2-5 -0.475 [-0.851, -0.211] **SEP**; gap 5-10 -0.231 [-0.673, +0.025]; gap>=10 +0.000 [+0.000, +0.000]
* **label_vs_gt** ADE: total +0.0211 [-0.0397, +0.0817]; GT_above_label: +0.0850 SEP (403 % [-3240, 3233]); GT_near_label: -0.0225 SEP (-107 % [-1018, 1140]); GT_well_below_label: -0.0415 SEP (-197 % [-1993, 2313])
  * |along| 6 s: total +0.0672 [-0.0816, +0.2110]; GT_above_label: +0.1925 SEP (287 % [-1873, 2155]); GT_near_label: -0.0446 SEP (-66 % [-637, 653]); GT_well_below_label: -0.0807 SEP (-120 % [-1361, 1351])
  * signed along 6 s (bin mean delta): GT_above_label -2.923 [-4.188, -1.778] **SEP**; GT_near_label -0.202 [-0.292, -0.122] **SEP**; GT_well_below_label -0.205 [-0.421, -0.044] **SEP**
* **ceiling** ADE: total +0.0211 [-0.0397, +0.0817]; ceiling ACTS (ema30k): +0.0211 (100 % [100, 100]); ceiling inert (ema30k): +0.0000 (0 % [0, 0])
  * |along| 6 s: total +0.0672 [-0.0816, +0.2110]; ceiling ACTS (ema30k): +0.0672 (100 % [100, 100]); ceiling inert (ema30k): +0.0000 (0 % [0, 0])
  * signed along 6 s (bin mean delta): ceiling ACTS (ema30k) -7.867 [-9.515, -6.366] **SEP**; ceiling inert (ema30k) +0.000 [+0.000, +0.000]

### C3: refcv8 EMA-30k max-speed label WITHHELD minus refcv7-50,400 label WITHHELD

* **v0bin** ADE: total +0.1448 [+0.0136, +0.2817]; v0<2: -0.0275 (-19 % [-215, 17]); v0 2-8: +0.0368 (25 % [-28, 111]); v0 8-15: +0.0880 SEP (61 % [9, 193]); v0 15-25: +0.0640 SEP (44 % [-4, 168]); v0>=25: -0.0165 SEP (-11 % [-79, 0])
  * |along| 6 s: total +0.3779 [-0.0002, +0.7767]; v0<2: -0.0435 (-12 % [-138, 39]); v0 2-8: +0.0645 (17 % [-60, 97]); v0 8-15: +0.2374 SEP (63 % [-13, 213]); v0 15-25: +0.1610 (43 % [-36, 157]); v0>=25: -0.0415 (-11 % [-83, 8])
  * signed along 6 s (bin mean delta): v0<2 -1.720 [-3.060, -0.198] **SEP**; v0 2-8 -1.637 [-2.346, -0.931] **SEP**; v0 8-15 -2.416 [-3.257, -1.668] **SEP**; v0 15-25 -1.590 [-2.863, -0.515] **SEP**; v0>=25 +0.292 [-0.691, +1.506]
* **dvclass** ADE: total +0.1448 [+0.0136, +0.2817]; start_from_stop: +0.0054 (4 % [-14, 30]); stopping: -0.0017 (-1 % [-39, 20]); slow_both: -0.0259 (-18 % [-183, 13]); accelerating: +0.0244 (17 % [-19, 74]); decelerating: -0.0583 SEP (-40 % [-279, -7]); cruising: +0.2008 SEP (139 % [82, 520])
  * |along| 6 s: total +0.3779 [-0.0002, +0.7767]; start_from_stop: +0.0112 (3 % [-16, 24]); stopping: -0.0073 (-2 % [-41, 26]); slow_both: -0.0466 (-12 % [-132, 34]); accelerating: +0.0565 (15 % [-22, 82]); decelerating: -0.2365 SEP (-63 % [-420, 148]); cruising: +0.6006 SEP (159 % [-66, 609])
  * signed along 6 s (bin mean delta): start_from_stop -2.113 [-3.708, -0.636] **SEP**; stopping -0.795 [-2.042, +0.656]; slow_both -1.609 [-3.772, +1.362]; accelerating -0.569 [-1.154, +0.021]; decelerating -3.109 [-3.853, -2.415] **SEP**; cruising -1.916 [-2.656, -1.233] **SEP**
* **label_gap** ADE: total +0.1448 [+0.0136, +0.2817]; gap<0: -0.0000 (-0 % [-0, 0]); gap 0-2: +0.0444 (31 % [-36, 128]); gap 2-5: +0.0594 (41 % [-18, 152]); gap 5-10: +0.0126 (9 % [-135, 69]); gap>=10: +0.0285 (20 % [-36, 82])
  * |along| 6 s: total +0.3779 [-0.0002, +0.7767]; gap<0: +0.0023 (1 % [0, 5]); gap 0-2: +0.1469 (39 % [-50, 158]); gap 2-5: +0.1320 (35 % [-49, 148]); gap 5-10: +0.0262 (7 % [-130, 83]); gap>=10: +0.0705 (19 % [-49, 102])
  * signed along 6 s (bin mean delta): gap<0 -0.308 [-0.308, -0.308] **SEP**; gap 0-2 -2.071 [-2.980, -1.200] **SEP**; gap 2-5 -1.673 [-2.390, -1.004] **SEP**; gap 5-10 -1.608 [-2.950, -0.407] **SEP**; gap>=10 -1.966 [-3.764, -0.532] **SEP**
* **label_vs_gt** ADE: total +0.1448 [+0.0136, +0.2817]; GT_above_label: -0.0095 (-7 % [-73, 9]); GT_near_label: +0.1341 SEP (93 % [40, 306]); GT_well_below_label: +0.0202 (14 % [-148, 72])
  * |along| 6 s: total +0.3779 [-0.0002, +0.7767]; GT_above_label: -0.0243 (-6 % [-64, 27]); GT_near_label: +0.3706 SEP (98 % [16, 302]); GT_well_below_label: +0.0316 (8 % [-159, 96])
  * signed along 6 s (bin mean delta): GT_above_label -2.064 [-3.220, -0.980] **SEP**; GT_near_label -1.948 [-2.648, -1.264] **SEP**; GT_well_below_label -1.544 [-2.240, -0.813] **SEP**
* **ceiling** ADE: total +0.1448 [+0.0136, +0.2817]; ceiling ACTS (ema30k): +0.0117 (8 % [-6, 49]); ceiling inert (ema30k): +0.1331 SEP (92 % [51, 106])
  * |along| 6 s: total +0.3779 [-0.0002, +0.7767]; ceiling ACTS (ema30k): +0.0196 (5 % [-18, 49]); ceiling inert (ema30k): +0.3583 (95 % [51, 118])
  * signed along 6 s (bin mean delta): ceiling ACTS (ema30k) +0.653 [-0.210, +1.483]; ceiling inert (ema30k) -1.975 [-2.449, -1.481] **SEP**

### C3b: refcv8 EMA-30k ON minus refcv8 EMA-30k label WITHHELD (refcv8's label effect)

* **v0bin** ADE: total -0.0021 [-0.0635, +0.0635]; v0<2: -0.0245 SEP (1163 % [-1040, 1394]); v0 2-8: -0.0273 SEP (1295 % [-1162, 1431]); v0 8-15: +0.0499 SEP (-2369 % [-2990, 2247]); v0 15-25: -0.0071 (337 % [-378, 563]); v0>=25: +0.0069 (-326 % [-366, 371])
  * |along| 6 s: total +0.0259 [-0.1259, +0.1808]; v0<2: -0.0557 SEP (-215 % [-1086, 986]); v0 2-8: -0.0170 (-66 % [-408, 508]); v0 8-15: +0.0976 SEP (377 % [-1278, 1479]); v0 15-25: -0.0147 (-57 % [-363, 307]); v0>=25: +0.0155 (60 % [-282, 267])
  * signed along 6 s (bin mean delta): v0<2 -0.542 [-1.367, +0.138]; v0 2-8 -0.618 [-0.963, -0.332] **SEP**; v0 8-15 -0.622 [-1.072, -0.252] **SEP**; v0 15-25 +0.123 [-0.103, +0.348]; v0>=25 -0.269 [-1.164, +0.309]
* **dvclass** ADE: total -0.0021 [-0.0635, +0.0635]; start_from_stop: -0.0056 (266 % [-303, 299]); stopping: -0.0055 (262 % [-295, 349]); slow_both: -0.0124 (590 % [-475, 706]); accelerating: -0.0028 (131 % [-522, 590]); decelerating: +0.0359 SEP (-1703 % [-2275, 1669]); cruising: -0.0117 (554 % [-558, 691])
  * |along| 6 s: total +0.0259 [-0.1259, +0.1808]; start_from_stop: -0.0119 (-46 % [-224, 217]); stopping: -0.0017 (-7 % [-111, 125]); slow_both: -0.0296 (-114 % [-539, 610]); accelerating: +0.0132 (51 % [-426, 422]); decelerating: +0.0801 SEP (309 % [-1018, 1111]); cruising: -0.0241 (-93 % [-541, 538])
  * signed along 6 s (bin mean delta): start_from_stop -0.185 [-1.435, +1.003]; stopping +0.284 [-0.063, +0.670]; slow_both -0.618 [-1.868, +0.183]; accelerating -0.627 [-1.111, -0.221] **SEP**; decelerating -0.820 [-1.532, -0.246] **SEP**; cruising -0.292 [-0.481, -0.126] **SEP**
* **label_gap** ADE: total -0.0021 [-0.0635, +0.0635]; gap<0: -0.0005 (25 % [-24, 32]); gap 0-2: +0.0302 (-1432 % [-1819, 1401]); gap 2-5: -0.0254 (1207 % [-1170, 1396]); gap 5-10: -0.0029 (136 % [-462, 706]); gap>=10: -0.0035 (164 % [-200, 198])
  * |along| 6 s: total +0.0259 [-0.1259, +0.1808]; gap<0: -0.0012 (-5 % [-22, 24]); gap 0-2: +0.0912 (352 % [-1337, 1346]); gap 2-5: -0.0472 (-182 % [-818, 1008]); gap 5-10: -0.0095 (-37 % [-510, 426]); gap>=10: -0.0073 (-28 % [-147, 183])
  * signed along 6 s (bin mean delta): gap<0 -0.018 [-0.018, -0.018] **SEP**; gap 0-2 -1.065 [-1.470, -0.659] **SEP**; gap 2-5 -0.334 [-0.742, +0.001]; gap 5-10 -0.050 [-0.521, +0.318]; gap>=10 -0.006 [-0.223, +0.206]
* **label_vs_gt** ADE: total -0.0021 [-0.0635, +0.0635]; GT_above_label: +0.0659 SEP (-3128 % [-3928, 3244]); GT_near_label: -0.0208 (989 % [-1012, 1238]); GT_well_below_label: -0.0472 SEP (2239 % [-2206, 2623])
  * |along| 6 s: total +0.0259 [-0.1259, +0.1808]; GT_above_label: +0.1569 SEP (606 % [-2168, 2433]); GT_near_label: -0.0400 (-155 % [-787, 777]); GT_well_below_label: -0.0910 SEP (-352 % [-1687, 1589])
  * signed along 6 s (bin mean delta): GT_above_label -3.016 [-4.300, -1.799] **SEP**; GT_near_label -0.112 [-0.271, +0.048]; GT_well_below_label -0.137 [-0.365, +0.059]
* **ceiling** ADE: total -0.0021 [-0.0635, +0.0635]; ceiling ACTS (ema30k): +0.0306 (-1452 % [-1980, 1683]); ceiling inert (ema30k): -0.0327 SEP (1552 % [-1583, 2080])
  * |along| 6 s: total +0.0259 [-0.1259, +0.1808]; ceiling ACTS (ema30k): +0.0803 (310 % [-873, 1144]); ceiling inert (ema30k): -0.0544 (-210 % [-1044, 973])
  * signed along 6 s (bin mean delta): ceiling ACTS (ema30k) -7.413 [-9.083, -5.848] **SEP**; ceiling inert (ema30k) +0.033 [-0.060, +0.126]

### C3c: refcv7-50,400 ON minus refcv7 label WITHHELD (refcv7's label effect)

* **v0bin** ADE: total -0.0231 [-0.0601, +0.0157]; v0<2: -0.0065 (28 % [-169, 242]); v0 2-8: +0.0044 (-19 % [-536, 549]); v0 8-15: +0.0113 (-49 % [-638, 601]); v0 15-25: -0.0158 SEP (68 % [-407, 551]); v0>=25: -0.0165 SEP (72 % [-397, 575])
  * |along| 6 s: total -0.0675 [-0.1604, +0.0291]; v0<2: -0.0126 (19 % [-130, 156]); v0 2-8: +0.0180 (-27 % [-567, 643]); v0 8-15: +0.0174 (-26 % [-383, 272]); v0 15-25: -0.0466 SEP (69 % [-383, 447]); v0>=25: -0.0437 SEP (65 % [-313, 439])
  * signed along 6 s (bin mean delta): v0<2 -0.036 [-0.553, +0.622]; v0 2-8 -0.215 [-0.471, +0.012]; v0 8-15 -0.023 [-0.185, +0.124]; v0 15-25 +0.401 [+0.179, +0.650] **SEP**; v0>=25 +0.511 [-0.119, +1.392]
* **dvclass** ADE: total -0.0231 [-0.0601, +0.0157]; start_from_stop: -0.0048 (21 % [-116, 182]); stopping: +0.0043 (-19 % [-197, 170]); slow_both: +0.0025 (-11 % [-174, 123]); accelerating: -0.0088 (38 % [-235, 266]); decelerating: +0.0044 (-19 % [-368, 300]); cruising: -0.0208 (90 % [-437, 648])
  * |along| 6 s: total -0.0675 [-0.1604, +0.0291]; start_from_stop: -0.0110 (16 % [-99, 134]); stopping: +0.0138 SEP (-20 % [-219, 188]); slow_both: +0.0049 (-7 % [-143, 85]); accelerating: -0.0217 (32 % [-160, 211]); decelerating: +0.0101 (-15 % [-362, 268]); cruising: -0.0637 SEP (94 % [-428, 633])
  * signed along 6 s (bin mean delta): start_from_stop -0.074 [-1.239, +1.510]; stopping +0.195 [-0.109, +0.564]; slow_both -0.003 [-0.381, +0.487]; accelerating +0.174 [-0.152, +0.487]; decelerating -0.182 [-0.449, +0.069]; cruising +0.060 [-0.069, +0.181]
* **label_gap** ADE: total -0.0231 [-0.0601, +0.0157]; gap<0: -0.0006 (3 % [-16, 26]); gap 0-2: -0.0089 (38 % [-267, 375]); gap 2-5: -0.0075 (33 % [-240, 337]); gap 5-10: -0.0142 (62 % [-367, 472]); gap>=10: +0.0081 (-35 % [-488, 355])
  * |along| 6 s: total -0.0675 [-0.1604, +0.0291]; gap<0: -0.0004 (1 % [-3, 5]); gap 0-2: -0.0139 (21 % [-253, 316]); gap 2-5: -0.0295 (44 % [-187, 315]); gap 5-10: -0.0369 (55 % [-311, 338]); gap>=10: +0.0132 (-20 % [-248, 191])
  * signed along 6 s (bin mean delta): gap<0 +0.001 [+0.001, +0.001] **SEP**; gap 0-2 -0.156 [-0.446, +0.138]; gap 2-5 +0.024 [-0.192, +0.242]; gap 5-10 +0.233 [+0.026, +0.432] **SEP**; gap>=10 +0.232 [-0.056, +0.488]
* **label_vs_gt** ADE: total -0.0231 [-0.0601, +0.0157]; GT_above_label: -0.0042 (18 % [-142, 174]); GT_near_label: -0.0229 (99 % [-347, 609]); GT_well_below_label: +0.0039 (-17 % [-578, 537])
  * |along| 6 s: total -0.0675 [-0.1604, +0.0291]; GT_above_label: -0.0081 (12 % [-116, 140]); GT_near_label: -0.0628 (93 % [-275, 496]); GT_well_below_label: +0.0034 (-5 % [-382, 346])
  * signed along 6 s (bin mean delta): GT_above_label -0.325 [-0.684, -0.018] **SEP**; GT_near_label +0.077 [-0.098, +0.252]; GT_well_below_label +0.091 [-0.102, +0.288]

### C4: refcv8 EMA-20k ON minus refcv7-50,400

* **v0bin** ADE: total +0.0847 [-0.0568, +0.2250]; v0<2: -0.0616 SEP (-73 % [-816, 666]); v0 2-8: -0.0107 (-13 % [-337, 352]); v0 8-15: +0.1018 SEP (120 % [-499, 797]); v0 15-25: +0.0468 (55 % [-281, 416]); v0>=25: +0.0083 (10 % [-66, 93])
  * |along| 6 s: total +0.2534 [-0.1347, +0.6342]; v0<2: -0.1259 SEP (-50 % [-563, 483]); v0 2-8: -0.0013 (-1 % [-237, 245]); v0 8-15: +0.2542 SEP (100 % [-340, 574]); v0 15-25: +0.1086 (43 % [-160, 335]); v0>=25: +0.0179 (7 % [-56, 66])
  * signed along 6 s (bin mean delta): v0<2 -2.972 [-4.254, -1.661] **SEP**; v0 2-8 -2.498 [-3.185, -1.814] **SEP**; v0 8-15 -3.158 [-3.893, -2.425] **SEP**; v0 15-25 -1.399 [-2.435, -0.524] **SEP**; v0>=25 -0.660 [-1.711, +0.323]
* **dvclass** ADE: total +0.0847 [-0.0568, +0.2250]; start_from_stop: +0.0051 (6 % [-82, 69]); stopping: -0.0182 (-21 % [-223, 213]); slow_both: -0.0581 SEP (-69 % [-760, 591]); accelerating: +0.0225 (27 % [-208, 210]); decelerating: -0.0474 (-56 % [-588, 614]); cruising: +0.1807 SEP (213 % [-1141, 1528])
  * |along| 6 s: total +0.2534 [-0.1347, +0.6342]; start_from_stop: +0.0117 (5 % [-47, 59]); stopping: -0.0534 (-21 % [-216, 153]); slow_both: -0.1264 SEP (-50 % [-559, 429]); accelerating: +0.0763 (30 % [-165, 215]); decelerating: -0.1996 SEP (-79 % [-940, 738]); cruising: +0.5448 SEP (215 % [-1040, 1468])
  * signed along 6 s (bin mean delta): start_from_stop -2.548 [-3.946, -1.128] **SEP**; stopping -2.230 [-3.319, -1.000] **SEP**; slow_both -3.451 [-5.179, -1.020] **SEP**; accelerating -1.529 [-2.165, -0.831] **SEP**; decelerating -3.531 [-4.435, -2.703] **SEP**; cruising -2.372 [-2.996, -1.736] **SEP**
* **label_gap** ADE: total +0.0847 [-0.0568, +0.2250]; gap<0: +0.0007 (1 % [-7, 8]); gap 0-2: +0.0993 SEP (117 % [-584, 798]); gap 2-5: +0.0040 (5 % [-288, 294]); gap 5-10: -0.0240 (-28 % [-516, 348]); gap>=10: +0.0046 (5 % [-220, 227])
  * |along| 6 s: total +0.2534 [-0.1347, +0.6342]; gap<0: +0.0035 (1 % [-11, 13]); gap 0-2: +0.2863 SEP (113 % [-561, 680]); gap 2-5: +0.0230 (9 % [-236, 253]); gap 5-10: -0.0784 (-31 % [-465, 340]); gap>=10: +0.0190 (7 % [-161, 237])
  * signed along 6 s (bin mean delta): gap<0 -1.092 [-1.092, -1.092] **SEP**; gap 0-2 -3.169 [-4.039, -2.317] **SEP**; gap 2-5 -2.187 [-2.820, -1.556] **SEP**; gap 5-10 -1.978 [-3.106, -0.904] **SEP**; gap>=10 -2.495 [-3.975, -1.184] **SEP**
* **label_vs_gt** ADE: total +0.0847 [-0.0568, +0.2250]; GT_above_label: +0.0666 SEP (79 % [-441, 605]); GT_near_label: +0.1098 SEP (130 % [-569, 846]); GT_well_below_label: -0.0917 (-108 % [-1263, 1076])
  * |along| 6 s: total +0.2534 [-0.1347, +0.6342]; GT_above_label: +0.1449 SEP (57 % [-287, 392]); GT_near_label: +0.3160 SEP (125 % [-452, 822]); GT_well_below_label: -0.2075 (-82 % [-1034, 879])
  * signed along 6 s (bin mean delta): GT_above_label -4.720 [-6.047, -3.445] **SEP**; GT_near_label -2.135 [-2.765, -1.521] **SEP**; GT_well_below_label -2.230 [-2.866, -1.611] **SEP**
* **ceiling** ADE: total +0.0847 [-0.0568, +0.2250]; ceiling ACTS (ema20k): +0.0637 SEP (75 % [-455, 575]); ceiling inert (ema20k): +0.0209 (25 % [-475, 555])
  * |along| 6 s: total +0.2534 [-0.1347, +0.6342]; ceiling ACTS (ema20k): +0.1360 SEP (54 % [-295, 403]); ceiling inert (ema20k): +0.1175 (46 % [-303, 395])
  * signed along 6 s (bin mean delta): ceiling ACTS (ema20k) -8.491 [-10.174, -6.743] **SEP**; ceiling inert (ema20k) -2.166 [-2.569, -1.729] **SEP**

### C5: refcv8 RAW-20k ON minus refcv7-50,400

* **v0bin** ADE: total +0.1433 [+0.0166, +0.2685]; v0<2: -0.0122 (-9 % [-222, 33]); v0 2-8: -0.0075 (-5 % [-114, 33]); v0 8-15: +0.1107 SEP (77 % [33, 259]); v0 15-25: +0.0449 SEP (31 % [5, 135]); v0>=25: +0.0075 (5 % [-15, 33])
  * |along| 6 s: total +0.2375 [-0.0913, +0.5470]; v0<2: -0.0137 (-6 % [-366, 219]); v0 2-8: -0.0478 (-20 % [-453, 288]); v0 8-15: +0.2284 SEP (96 % [-270, 691]); v0 15-25: +0.0653 (28 % [-110, 271]); v0>=25: +0.0053 (2 % [-72, 58])
  * signed along 6 s (bin mean delta): v0<2 -2.428 [-4.290, -0.503] **SEP**; v0 2-8 -1.556 [-2.277, -0.806] **SEP**; v0 8-15 -1.698 [-2.461, -0.985] **SEP**; v0 15-25 +0.460 [-0.126, +1.038]; v0>=25 +0.579 [-0.525, +1.498]
* **dvclass** ADE: total +0.1433 [+0.0166, +0.2685]; start_from_stop: +0.0270 SEP (19 % [3, 84]); stopping: -0.0044 (-3 % [-32, 18]); slow_both: -0.0376 (-26 % [-281, 7]); accelerating: +0.0256 (18 % [-47, 69]); decelerating: -0.0019 (-1 % [-96, 35]); cruising: +0.1346 SEP (94 % [51, 378])
  * |along| 6 s: total +0.2375 [-0.0913, +0.5470]; start_from_stop: +0.0593 SEP (25 % [-133, 202]); stopping: -0.0041 (-2 % [-81, 63]); slow_both: -0.0782 (-33 % [-563, 322]); accelerating: +0.0209 (9 % [-175, 152]); decelerating: -0.1177 SEP (-50 % [-658, 382]); cruising: +0.3573 SEP (150 % [-648, 1358])
  * signed along 6 s (bin mean delta): start_from_stop -2.132 [-4.332, +0.006]; stopping -0.079 [-1.303, +1.232]; slow_both -2.791 [-5.628, +0.760]; accelerating -0.628 [-1.419, +0.250]; decelerating -2.276 [-3.175, -1.377] **SEP**; cruising -0.857 [-1.401, -0.365] **SEP**
* **label_gap** ADE: total +0.1433 [+0.0166, +0.2685]; gap<0: +0.0008 (1 % [0, 4]); gap 0-2: +0.0645 (45 % [-11, 148]); gap 2-5: +0.0323 (23 % [-31, 84]); gap 5-10: +0.0272 (19 % [-104, 76]); gap>=10: +0.0186 (13 % [-37, 59])
  * |along| 6 s: total +0.2375 [-0.0913, +0.5470]; gap<0: +0.0028 (1 % [-5, 12]); gap 0-2: +0.1600 (67 % [-198, 392]); gap 2-5: +0.0348 (15 % [-189, 186]); gap 5-10: +0.0130 (5 % [-300, 231]); gap>=10: +0.0269 (11 % [-111, 169])
  * signed along 6 s (bin mean delta): gap<0 +0.157 [+0.157, +0.157] **SEP**; gap 0-2 -1.972 [-2.887, -1.060] **SEP**; gap 2-5 -1.245 [-1.944, -0.579] **SEP**; gap 5-10 -0.310 [-1.666, +0.928]; gap>=10 -0.463 [-1.951, +0.888]
* **label_vs_gt** ADE: total +0.1433 [+0.0166, +0.2685]; GT_above_label: +0.0617 SEP (43 % [13, 207]); GT_near_label: +0.0820 SEP (57 % [11, 208]); GT_well_below_label: -0.0003 (-0 % [-279, 46])
  * |along| 6 s: total +0.2375 [-0.0913, +0.5470]; GT_above_label: +0.1459 SEP (61 % [-241, 490]); GT_near_label: +0.1551 (65 % [-155, 339]); GT_well_below_label: -0.0636 (-27 % [-670, 451])
  * signed along 6 s (bin mean delta): GT_above_label -4.004 [-5.485, -2.528] **SEP**; GT_near_label -0.676 [-1.225, -0.133] **SEP**; GT_well_below_label -0.980 [-1.737, -0.224] **SEP**
* **ceiling** ADE: total +0.1433 [+0.0166, +0.2685]; ceiling ACTS (raw20k): +0.0340 (24 % [-15, 99]); ceiling inert (raw20k): +0.1093 (76 % [1, 115])
  * |along| 6 s: total +0.2375 [-0.0913, +0.5470]; ceiling ACTS (raw20k): +0.0707 (30 % [-119, 238]); ceiling inert (raw20k): +0.1668 (70 % [-138, 219])
  * signed along 6 s (bin mean delta): ceiling ACTS (raw20k) -5.811 [-7.219, -4.450] **SEP**; ceiling inert (raw20k) -0.773 [-1.262, -0.294] **SEP**

### C6: refcv8 EMA-30k ON minus refcv8 EMA-20k ON (what moved 20k -> 30k)

* **v0bin** ADE: total +0.0811 [+0.0268, +0.1357]; v0<2: +0.0160 SEP (20 % [3, 56]); v0 2-8: +0.0157 (19 % [-22, 52]); v0 8-15: +0.0249 (31 % [-20, 65]); v0 15-25: +0.0259 SEP (32 % [3, 79]); v0>=25: -0.0014 (-2 % [-12, 4])
  * |along| 6 s: total +0.2178 [+0.0894, +0.3525]; v0<2: +0.0394 SEP (18 % [3, 46]); v0 2-8: +0.0309 (14 % [-22, 42]); v0 8-15: +0.0634 (29 % [-10, 63]); v0 15-25: +0.0844 SEP (39 % [7, 79]); v0>=25: -0.0002 (-0 % [-8, 6])
  * signed along 6 s (bin mean delta): v0<2 +0.747 [+0.229, +1.276] **SEP**; v0 2-8 +0.459 [+0.201, +0.715] **SEP**; v0 8-15 +0.142 [-0.156, +0.452]; v0 15-25 -0.468 [-0.943, -0.028] **SEP**; v0>=25 +0.172 [-0.282, +0.559]
* **dvclass** ADE: total +0.0811 [+0.0268, +0.1357]; start_from_stop: -0.0005 (-1 % [-12, 8]); stopping: +0.0067 (8 % [-18, 35]); slow_both: +0.0173 SEP (21 % [7, 60]); accelerating: +0.0079 (10 % [-35, 44]); decelerating: +0.0206 (25 % [-4, 72]); cruising: +0.0291 (36 % [-14, 75])
  * |along| 6 s: total +0.2178 [+0.0894, +0.3525]; start_from_stop: -0.0015 (-1 % [-9, 5]); stopping: +0.0306 (14 % [-2, 41]); slow_both: +0.0452 SEP (21 % [6, 51]); accelerating: +0.0151 (7 % [-28, 32]); decelerating: +0.0332 (15 % [-17, 44]); cruising: +0.0953 SEP (44 % [4, 78])
  * signed along 6 s (bin mean delta): start_from_stop +0.323 [-0.504, +1.156]; stopping +1.523 [+0.905, +2.111] **SEP**; slow_both +1.227 [+0.635, +2.002] **SEP**; accelerating +0.158 [-0.191, +0.520]; decelerating -0.216 [-0.771, +0.249]; cruising +0.104 [-0.144, +0.346]
* **label_gap** ADE: total +0.0811 [+0.0268, +0.1357]; gap<0: -0.0007 (-1 % [-4, 0]); gap 0-2: -0.0159 (-20 % [-132, 13]); gap 2-5: +0.0375 SEP (46 % [14, 101]); gap 5-10: +0.0479 SEP (59 % [28, 140]); gap>=10: +0.0123 (15 % [-7, 50])
  * |along| 6 s: total +0.2178 [+0.0894, +0.3525]; gap<0: -0.0021 (-1 % [-4, 0]); gap 0-2: -0.0342 (-16 % [-91, 13]); gap 2-5: +0.0913 SEP (42 % [14, 86]); gap 5-10: +0.1319 SEP (61 % [30, 120]); gap>=10: +0.0309 (14 % [-5, 41])
  * signed along 6 s (bin mean delta): gap<0 +0.765 [+0.765, +0.765] **SEP**; gap 0-2 +0.190 [-0.072, +0.461]; gap 2-5 +0.156 [-0.147, +0.475]; gap 5-10 +0.087 [-0.405, +0.552]; gap>=10 +0.291 [-0.297, +0.850]
* **label_vs_gt** ADE: total +0.0811 [+0.0268, +0.1357]; GT_above_label: -0.0060 (-7 % [-84, 21]); GT_near_label: +0.0264 (33 % [-11, 77]); GT_well_below_label: +0.0608 SEP (75 % [41, 165])
  * |along| 6 s: total +0.2178 [+0.0894, +0.3525]; GT_above_label: -0.0042 (-2 % [-48, 22]); GT_near_label: +0.0774 (36 % [-8, 71]); GT_well_below_label: +0.1446 SEP (66 % [36, 129])
  * signed along 6 s (bin mean delta): GT_above_label -0.035 [-0.773, +0.626]; GT_near_label -0.002 [-0.240, +0.238]; GT_well_below_label +0.458 [+0.193, +0.736] **SEP**
* **ceiling** ADE: total +0.0811 [+0.0268, +0.1357]; ceiling ACTS (ema30k): -0.0041 (-5 % [-64, 22]); ceiling inert (ema30k): +0.0852 SEP (105 % [78, 164])
  * |along| 6 s: total +0.2178 [+0.0894, +0.3525]; ceiling ACTS (ema30k): -0.0014 (-1 % [-34, 21]); ceiling inert (ema30k): +0.2193 SEP (101 % [79, 134])
  * signed along 6 s (bin mean delta): ceiling ACTS (ema30k) -0.421 [-1.626, +0.596]; ceiling inert (ema30k) +0.215 [+0.039, +0.396] **SEP**

### C7: refcv8 RAW-20k ON minus refcv8 EMA-20k ON (raw vs EMA at 20k)

* **v0bin** ADE: total +0.0587 [-0.0388, +0.1649]; v0<2: +0.0493 SEP (84 % [-596, 539]); v0 2-8: +0.0032 (5 % [-322, 319]); v0 8-15: +0.0088 (15 % [-273, 306]); v0 15-25: -0.0019 (-3 % [-304, 279]); v0>=25: -0.0007 (-1 % [-69, 84])
  * |along| 6 s: total -0.0160 [-0.2568, +0.2382]; v0<2: +0.1122 SEP (-703 % [-1123, 1137]); v0 2-8: -0.0464 (291 % [-511, 581]); v0 8-15: -0.0258 (162 % [-593, 617]); v0 15-25: -0.0433 (271 % [-518, 501]); v0>=25: -0.0126 (79 % [-164, 194])
  * signed along 6 s (bin mean delta): v0<2 +0.544 [-0.725, +2.048]; v0 2-8 +0.942 [+0.505, +1.419] **SEP**; v0 8-15 +1.460 [+1.065, +1.844] **SEP**; v0 15-25 +1.860 [+1.320, +2.509] **SEP**; v0>=25 +1.238 [+0.745, +1.786] **SEP**
* **dvclass** ADE: total +0.0587 [-0.0388, +0.1649]; start_from_stop: +0.0219 SEP (37 % [-377, 264]); stopping: +0.0138 (24 % [-193, 186]); slow_both: +0.0205 (35 % [-230, 253]); accelerating: +0.0031 (5 % [-224, 330]); decelerating: +0.0455 SEP (78 % [-710, 525]); cruising: -0.0461 (-79 % [-980, 1279])
  * |along| 6 s: total -0.0160 [-0.2568, +0.2382]; start_from_stop: +0.0475 SEP (-298 % [-555, 472]); stopping: +0.0493 SEP (-309 % [-576, 482]); slow_both: +0.0482 (-302 % [-545, 531]); accelerating: -0.0553 (346 % [-511, 648]); decelerating: +0.0819 SEP (-513 % [-899, 889]); cruising: -0.1875 SEP (1175 % [-1794, 1871])
  * signed along 6 s (bin mean delta): start_from_stop +0.416 [-1.364, +1.877]; stopping +2.150 [+1.290, +2.860] **SEP**; slow_both +0.660 [-1.388, +3.716]; accelerating +0.900 [+0.444, +1.368] **SEP**; decelerating +1.255 [+0.769, +1.726] **SEP**; cruising +1.515 [+1.204, +1.824] **SEP**
* **label_gap** ADE: total +0.0587 [-0.0388, +0.1649]; gap<0: +0.0001 (0 % [-2, 1]); gap 0-2: -0.0349 (-59 % [-692, 904]); gap 2-5: +0.0284 (48 % [-249, 366]); gap 5-10: +0.0512 (87 % [-501, 551]); gap>=10: +0.0139 (24 % [-296, 273])
  * |along| 6 s: total -0.0160 [-0.2568, +0.2382]; gap<0: -0.0008 (5 % [-8, 11]); gap 0-2: -0.1264 SEP (792 % [-1176, 1323]); gap 2-5: +0.0118 (-74 % [-410, 460]); gap 5-10: +0.0914 (-573 % [-1025, 1091]); gap>=10: +0.0079 (-50 % [-409, 433])
  * signed along 6 s (bin mean delta): gap<0 +1.249 [+1.249, +1.249] **SEP**; gap 0-2 +1.197 [+0.757, +1.672] **SEP**; gap 2-5 +0.942 [+0.536, +1.338] **SEP**; gap 5-10 +1.668 [+0.922, +2.454] **SEP**; gap>=10 +2.032 [+1.042, +2.998] **SEP**
* **label_vs_gt** ADE: total +0.0587 [-0.0388, +0.1649]; GT_above_label: -0.0050 (-8 % [-177, 246]); GT_near_label: -0.0278 (-47 % [-684, 975]); GT_well_below_label: +0.0914 SEP (156 % [-1064, 887])
  * |along| 6 s: total -0.0160 [-0.2568, +0.2382]; GT_above_label: +0.0010 (-6 % [-281, 278]); GT_near_label: -0.1608 SEP (1008 % [-1471, 1419]); GT_well_below_label: +0.1439 (-902 % [-1415, 1486])
  * signed along 6 s (bin mean delta): GT_above_label +0.715 [-0.062, +1.471]; GT_near_label +1.460 [+1.156, +1.777] **SEP**; GT_well_below_label +1.250 [+0.744, +1.745] **SEP**
* **ceiling** ADE: total +0.0587 [-0.0388, +0.1649]; ceiling ACTS (raw20k): -0.0124 (-21 % [-236, 246]); ceiling inert (raw20k): +0.0710 (121 % [-146, 336])
  * |along| 6 s: total -0.0160 [-0.2568, +0.2382]; ceiling ACTS (raw20k): -0.0226 (142 % [-280, 307]); ceiling inert (raw20k): +0.0066 (-42 % [-207, 380])
  * signed along 6 s (bin mean delta): ceiling ACTS (raw20k) -0.476 [-1.389, +0.379]; ceiling inert (raw20k) +1.446 [+1.153, +1.738] **SEP**

## Plan-derived longitudinal decision vs GT (decel / cruise / accel from the last-second speed minus v0, +-1.5 m/s; all windows, seed 0)

GT class counts: {'decel': 832, 'cruise': 1949, 'accel': 887}

| arm | accuracy | GT decel -> plan decel/cruise/accel | GT cruise -> plan decel/cruise/accel | GT accel -> plan decel/cruise/accel |
|---|---|---|---|---|
| r7_ON | 0.661 | 376/406/50 | 188/1493/268 | 31/299/557 |
| ema30k_ON | 0.613 | 545/239/48 | 535/1195/219 | 53/326/508 |
| ema30k_OFF | 0.613 | 531/249/52 | 520/1186/243 | 47/310/530 |
| ema20k_ON | 0.625 | 563/226/43 | 541/1240/168 | 45/351/491 |

Paired refcv8 EMA-30k minus refcv7-50,400 (per-window indicators, episode-cluster bootstrap): **false_decel_in_gt_cruise** +0.1780 [+0.1228, +0.2404] SEP (refcv8 0.274 vs refcv7 0.096; n 1949/138); **correct** -0.0485 [-0.0898, -0.0090] SEP (refcv8 0.613 vs refcv7 0.661; n 3668/139); **missed_decel_in_gt_decel** -0.2031 [-0.2707, -0.1435] SEP (refcv8 0.345 vs refcv7 0.548; n 832/80)

refcv8's tactical lon head argmax (`nav_true.tacv6_lon_logits`) per decision cell (counts):

* GT cruise, refcv8 plan decel, refcv7 plan cruise (n 374): FOLLOW 52, CRUISE 42, BRAKE_TO 274, ACCELERATE 6
* GT cruise, refcv8 plan cruise (n 1195): FOLLOW 331, CRUISE 607, BRAKE_TO 121, HOLD 21, ACCELERATE 115
* GT decel, refcv8 plan decel, refcv7 plan cruise (n 189): FOLLOW 29, CRUISE 38, BRAKE_TO 120, ACCELERATE 2
* all windows (n 3668): FOLLOW 920, CRUISE 1095, BRAKE_TO 992, CREEP 4, HOLD 47, ACCELERATE 610

refcv8's tactical lon head argmax given the GT `lon_v7` label (rows = label; counts):

* label FOLLOW (n 676): FOLLOW 565, CRUISE 41, BRAKE_TO 41, ACCELERATE 29
* label CRUISE (n 1139): FOLLOW 91, CRUISE 617, BRAKE_TO 354, HOLD 5, ACCELERATE 72
* label BRAKE_TO (n 809): FOLLOW 110, CRUISE 189, BRAKE_TO 444, HOLD 1, ACCELERATE 65
* label CREEP (n 30): FOLLOW 3, CRUISE 3, BRAKE_TO 5, HOLD 9, ACCELERATE 10
* label HOLD (n 31): CRUISE 1, BRAKE_TO 1, HOLD 14, ACCELERATE 15
* label ACCELERATE (n 724): FOLLOW 91, CRUISE 184, BRAKE_TO 114, CREEP 4, HOLD 9, ACCELERATE 322

### C0 by plan-derived decision of refcv8 (new) and refcv7 (ref) -- OUTCOME-defined

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| new decel / ref decel | 528 / 78 | 14.4 | +0.0538 [+0.0122, +0.0978] **SEP** | 32 % [8, 128] | +0.3739 [+0.0957, +0.6854] **SEP** | 27 % [5, 96] | +0.894 [+0.178, +1.695] **SEP** | 30 % [9, 95] | -3.16 / -1.03 / -2.133 [-2.985, -1.172] **SEP** | 61 |
| new decel / ref cruise | 591 / 93 | 16.1 | +0.1867 [+0.0857, +0.2951] **SEP** | 113 % [68, 306] | +1.1586 [+0.6137, +1.6076] **SEP** | 108 % [67, 246] | +3.156 [+1.553, +4.488] **SEP** | 112 % [72, 261] | -5.80 / +2.66 / -8.458 [-9.190, -7.657] **SEP** | 82 |
| new decel / ref accel | 14 / 12 | 0.4 | -0.0009 [-0.0174, +0.0161] | -1 % [-20, 13] | -0.2313 [-4.7480, +4.2725] | 0 % [-15, 12] | +0.109 [-10.986, +11.216] | 0 % [-14, 13] | -8.87 / +10.50 / -19.375 [-24.134, -13.996] **SEP** | 71 |
| new cruise / ref decel | 64 / 38 | 1.7 | +0.0034 [-0.0130, +0.0184] | 2 % [-16, 17] | +0.1951 [-0.7816, +1.0346] | 2 % [-10, 16] | +0.593 [-1.608, +2.618] | 1 % [-11, 12] | +2.52 / -2.51 / +5.029 [+3.425, +6.543] **SEP** | 45 |
| new cruise / ref cruise | 1483 / 132 | 40.4 | -0.0092 [-0.0266, +0.0089] | -6 % [-36, 7] | -0.0227 [-0.0661, +0.0222] | -9 % [-46, 3] | -0.105 [-0.230, +0.022] | -4 % [-27, 6] | +0.20 / +0.72 / -0.524 [-0.679, -0.379] **SEP** | 55 |
| new cruise / ref accel | 213 / 62 | 5.8 | -0.0435 [-0.0916, -0.0012] **SEP** | -26 % [-188, 1] | -0.7494 [-1.4441, -0.0207] **SEP** | -22 % [-138, 1] | -1.767 [-3.492, -0.020] **SEP** | -26 % [-148, -2] | -2.04 / +4.88 / -6.915 [-8.012, -5.695] **SEP** | 63 |
| new accel / ref decel | 3 / 2 | 0.1 | -0.0027 [-0.0073, +0.0000] | -2 % [-10, 0] | -3.2459 [-3.5462, -2.6453] **SEP** | -0 % [-3, 1] | -1.136 [-8.469, +2.531] | -1 % [-7, 0] | +3.95 / -5.08 / +9.030 [+7.303, +12.485] **SEP** | 0 |
| new accel / ref cruise | 124 / 49 | 3.4 | +0.0280 [+0.0056, +0.0564] **SEP** | 17 % [2, 82] | +0.8269 [+0.1927, +1.4855] **SEP** | 14 % [3, 59] | +1.904 [+0.608, +3.259] **SEP** | 9 % [-2, 43] | +4.65 / -0.27 / +4.918 [+3.640, +6.225] **SEP** | 27 |
| new accel / ref accel | 648 / 77 | 17.7 | -0.0498 [-0.0840, -0.0178] **SEP** | -30 % [-168, -7] | -0.2819 [-0.4547, -0.1109] **SEP** | -20 % [-108, -2] | -0.538 [-0.960, -0.120] **SEP** | -20 % [-99, -3] | +2.07 / +3.19 / -1.113 [-1.603, -0.573] **SEP** | 40 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **new decel / ref cruise** (cumulative point share 113 %). |along| 6 s: **new decel / ref cruise** (108 %).

### C0 by refcv8's tactical lon head argmax (`nav_true.tacv6_lon_logits`) -- OUTCOME-defined

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| pred FOLLOW (ema30k) | 920 / 62 | 25.1 | -0.0131 [-0.0595, +0.0340] | -8 % [-93, 21] | -0.0521 [-0.2394, +0.1317] | -5 % [-63, 19] | -0.099 [-0.506, +0.332] | -4 % [-50, 20] | +0.16 / +1.37 / -1.214 [-1.807, -0.686] **SEP** | 52 |
| pred CRUISE (ema30k) | 1095 / 68 | 29.9 | +0.0296 [-0.0126, +0.0777] | 18 % [-13, 68] | +0.0992 [-0.0439, +0.2564] | 13 % [-17, 52] | +0.197 [-0.147, +0.567] | 16 % [-5, 61] | +0.15 / +1.04 / -0.895 [-1.377, -0.440] **SEP** | 54 |
| pred BRAKE_TO (ema30k) | 992 / 71 | 27.0 | +0.1807 [+0.0780, +0.2968] **SEP** | 109 % [61, 308] | +0.6682 [+0.3220, +1.0019] **SEP** | 104 % [64, 241] | +1.820 [+0.754, +2.839] **SEP** | 111 % [70, 243] | -3.40 / +1.84 / -5.234 [-6.011, -4.320] **SEP** | 67 |
| pred CREEP (ema30k) | 4 / 1 | 0.1 | -0.0008 [-0.0024, +0.0000] | -0 % [-3, 0] | -0.7270 [-0.7270, -0.7270] **SEP** | -0 % [-2, 0] | -1.574 [-1.574, -1.574] **SEP** | -0 % [-2, 0] | +2.10 / +4.94 / -2.838 [-2.838, -2.838] **SEP** | 50 |
| pred HOLD (ema30k) | 47 / 6 | 1.3 | +0.0032 [-0.0025, +0.0099] | 2 % [-3, 11] | +0.2487 [-0.7006, +0.5459] | 1 % [-2, 7] | +0.411 [-1.840, +1.175] | 1 % [-2, 5] | -0.64 / +0.65 / -1.282 [-3.534, +0.330] | 60 |
| pred ACCELERATE (ema30k) | 610 / 69 | 16.6 | -0.0338 [-0.0913, +0.0224] | -20 % [-170, 13] | -0.2034 [-0.5287, +0.1399] | -12 % [-101, 17] | -0.352 [-1.117, +0.502] | -24 % [-152, 4] | -0.30 / +1.42 / -1.719 [-2.498, -0.879] **SEP** | 52 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **pred BRAKE_TO (ema30k)** (cumulative point share 109 %). |along| 6 s: **pred BRAKE_TO (ema30k)** (104 %).

### C0 by whether refcv8's tactical lon head matches the GT lon_v7 label

| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |
|---|---|---|---|---|---|---|---|---|---|---|
| tac lon pred == lon_v7 | 1962 / 127 | 53.5 | +0.0309 [-0.0293, +0.0906] | 19 % [-43, 68] | +0.0577 [-0.0534, +0.1695] | 5 % [-69, 36] | +0.042 [-0.243, +0.314] | 8 % [-53, 41] | -1.14 / +0.70 / -1.834 [-2.334, -1.355] **SEP** | 58 |
| tac lon pred != lon_v7 | 1447 / 111 | 39.4 | +0.1482 [+0.0489, +0.2555] **SEP** | 89 % [50, 226] | +0.3756 [+0.1307, +0.6120] **SEP** | 93 % [55, 202] | +1.112 [+0.401, +1.826] **SEP** | 95 % [58, 210] | -0.93 / +1.89 / -2.815 [-3.522, -2.072] **SEP** | 57 |
| unlabelled | 259 / 22 | 7.1 | -0.0133 [-0.0749, +0.0562] | -8 % [-139, 29] | -0.1877 [-1.0573, +0.7720] | 2 % [-59, 39] | +0.143 [-1.888, +2.665] | -3 % [-85, 34] | +1.18 / +4.03 / -2.854 [-4.853, -0.859] **SEP** | 46 |
| **all** | 3668 / 139 | 100 | +0.1658 [+0.0268, +0.3153] | 100 % | | | | | | |

ADE: bins reaching >= 50 % of the delta (largest share first): **tac lon pred != lon_v7** (cumulative point share 89 %). |along| 6 s: **tac lon pred != lon_v7** (93 %).

## Bias structure: OFFSET or COMPRESSION? (all 3,668 windows, seed 0)

Plan x(6 s) regressed on GT x(6 s) (OLS; slope < 1 = the speed profile is compressed toward the mean; slope CI and slope minus refcv7 on the same draws).

| arm | mean signed along 6 s [CI] m | windows too slow | mean signed speed err m/s | OLS slope [CI] | intercept m | slope minus refcv7 [CI] |
|---|---|---|---|---|---|---|
| r7_ON | +1.402 [+0.711, +2.090] | 45 % | +0.223 | 0.9525 [0.9360, 0.9665] | +4.529 |  |
| ema30k_ON | -0.890 [-1.699, -0.117] | 57 % | -0.151 | 0.9604 [0.9418, 0.9754] | +1.716 | +0.0079 [-0.0020, +0.0158] |
| ema30k_OFF | -0.378 [-1.215, +0.413] | 55 % | -0.059 | 0.9581 [0.9402, 0.9732] | +2.376 | +0.0057 [-0.0039, +0.0136] |
| ema30k_vmaxzero | -0.438 [-1.307, +0.385] | 55 % | -0.070 | 0.9576 [0.9390, 0.9727] | +2.352 | +0.0051 [-0.0050, +0.0131] |
| r7_vmaxzero | +1.366 [+0.638, +2.082] | 46 % | +0.218 | 0.9476 [0.9299, 0.9631] | +4.812 | -0.0048 [-0.0084, -0.0017] |
| ema20k_ON | -1.064 [-1.753, -0.399] | 60 % | -0.196 | 0.9684 [0.9528, 0.9812] | +1.018 | +0.0159 [+0.0077, +0.0233] |
| raw20k_ON | +0.229 [-0.451, +0.904] | 49 % | +0.030 | 0.9727 [0.9583, 0.9846] | +2.023 | +0.0203 [+0.0119, +0.0292] |

## Lever L1 RUN: a post-hoc uniform speed offset (zero GPU, zero training) -- DIAGNOSTIC, fit on eval episodes

* each waypoint moved along the plan's own polyline by beta*t (beta m/s, grid [-2.0, 2.0, 0.05]); episode position parity (eid % 2) 2-fold cross-fit, criterion mean ADE 0-6 s on the fit fold; the scored fold is never tuned on. Known-value control: beta = 0 reproduces the plan, max abs 0.0e+00 m.
* applied to BOTH arms (a calibrated refcv8 against a raw refcv7 is not a fair comparison); per-v0-bin uses only the measured v0 at t0 (admissible at inference).
* betas chosen: refcv8 {'global': {'betas_chosen_mps': {'fold0:all': 0.05, 'fold1:all': 0.05}}, 'per_v0bin': {'betas_chosen_mps': {'fold0:v0 15-25': 0.15, 'fold0:v0 2-8': 0.05, 'fold0:v0 8-15': 0.05, 'fold0:v0<2': -0.65, 'fold0:v0>=25': 0.1, 'fold1:v0 15-25': 0.1, 'fold1:v0 2-8': -0.05, 'fold1:v0 8-15': 0.05, 'fold1:v0<2': -0.55, 'fold1:v0>=25': 0.1}}}; refcv7 {'global': {'betas_chosen_mps': {'fold0:all': -0.05, 'fold1:all': -0.05}}, 'per_v0bin': {'betas_chosen_mps': {'fold0:v0 15-25': 0.1, 'fold0:v0 2-8': -0.1, 'fold0:v0 8-15': -0.05, 'fold0:v0<2': -0.7, 'fold0:v0>=25': 0.05, 'fold1:v0 15-25': 0.0, 'fold1:v0 2-8': -0.2, 'fold1:v0 8-15': -0.05, 'fold1:v0<2': -1.05, 'fold1:v0>=25': 0.1}}}

| mode | metric | refcv8 cal - refcv7 RAW [CI] | refcv8 cal - refcv7 cal [CI] | refcv8 cal - refcv8 raw [CI] | refcv7 cal - refcv7 raw [CI] |
|---|---|---|---|---|---|
| global | ADE 0-6 s | +0.1598 [+0.0262, +0.3032] SEP | +0.1703 [+0.0412, +0.3116] SEP | -0.0060 [-0.0180, +0.0056] | -0.0105 [-0.0220, +0.0015] |
| global | |along-track| at 6 s | +0.4402 [+0.0717, +0.8313] SEP | +0.4594 [+0.0968, +0.8542] SEP | -0.0311 [-0.0544, -0.0081] SEP | -0.0192 [-0.0427, +0.0059] |
| global | speed MAE 0-6 s | +0.0759 [+0.0157, +0.1391] SEP | +0.0781 [+0.0186, +0.1412] SEP | -0.0020 [-0.0054, +0.0011] | -0.0022 [-0.0055, +0.0012] |
| per_v0bin | ADE 0-6 s | +0.1219 [-0.0286, +0.2806] | +0.1782 [+0.0566, +0.3095] SEP | -0.0440 [-0.0870, -0.0060] SEP | -0.0563 [-0.1275, +0.0093] |
| per_v0bin | |along-track| at 6 s | +0.3413 [-0.0531, +0.7517] | +0.4940 [+0.1399, +0.8723] SEP | -0.1300 [-0.2148, -0.0577] SEP | -0.1527 [-0.2940, -0.0231] SEP |
| per_v0bin | speed MAE 0-6 s | +0.0675 [+0.0021, +0.1331] SEP | +0.0803 [+0.0235, +0.1409] SEP | -0.0104 [-0.0201, -0.0020] SEP | -0.0128 [-0.0311, +0.0044] |

reference (uncalibrated): refcv8 - refcv7 ADE +0.1658 [+0.0268, +0.3153] SEP.
* global, refcv8 cal - refcv7 cal ADE by dvclass: total +0.1703 [+0.0412, +0.3116]; start_from_stop +0.0050; stopping -0.0024; slow_both -0.0316; accelerating +0.0107; decelerating -0.0019; cruising +0.1904 SEP
* per_v0bin, refcv8 cal - refcv7 cal ADE by dvclass: total +0.1782 [+0.0566, +0.3095]; start_from_stop -0.0011; stopping +0.0007; slow_both -0.0204; accelerating +0.0061; decelerating -0.0014; cruising +0.1943 SEP

## VERDICT (analyst-authored from `S6_LON_DECOMP.json` of this run; every number below is MEASURED from that file unless marked)

Stamp for every number: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; S6 0-6 s; n 3,668 windows / 139 episodes, inference seed 0 (seed 1 where stated); paired episode-cluster bootstrap, B 2000. It answers "another draw of EPISODES" only. It does NOT answer "another training run": each arm has one training seed, so every refcv8-vs-refcv7 difference below is exposed to `H-ESTIM-SEED-1`. It does NOT answer "another inference run" either. Shares are contribution / total of the C0 ADE delta +0.1658 [+0.0268, +0.3153] m. Controls Z, R, X and the L1 identity all PASS.

**1. Where the delta lives (bins carrying >= 50 %).** One bin carries all of it: **GT-CRUISING windows (|dv| < 1.5 m/s over 6 s): 1,739 win / 128 ep (47 % of windows). ADE contribution +0.2099 [+0.1089, +0.3179] m = 127 % [77, 390] of the delta; |along| 6 s 136 % [88, 375]; speed MAE 132 % [86, 339].** By initial speed it is **v0 8-15 m/s: 76 % [30, 198]** plus **v0 15-25: 44 % [7, 150]**. No other speed bin contributes anything separated. The largest single cell is v0 8-15 x cruising (755 win / 65 ep, 86 % [41, 267]). Seed 1 puts it in the same place (cruising 113 % [74, 273]; v0 8-15 75 % [37, 174]). A share above 100 % means the other bins partly cancel it: refcv8 is BETTER than refcv7 in slow, stopping and decelerating windows (v0 < 2: -27 %; slow_both -25 %; decelerating |along| -35 %).

**2. Too slow or too fast: refcv8 is TOO SLOW. It brakes where the human cruised.** In the cruising bin refcv8 ends 2.21 m short at 6 s, while refcv7 is +0.06 m (delta -2.268 [-2.999, -1.565] m). Over all windows, refcv8's signed along-track at 6 s is -0.890 [-1.699, -0.117] m and refcv7's is +1.402 [+0.711, +2.090] m. This is a DECISION error, not a gain error. Regressing plan x(6 s) on GT x(6 s) gives a slope of 0.9604 vs 0.9525, difference +0.0079 [-0.0020, +0.0158], so the speed profile is not compressed. We also derived a longitudinal decision from each plan (last-second speed minus v0, +-1.5 m/s, the same rule for GT):
* **False deceleration in GT-cruise windows: 0.274 vs 0.096, +0.1780 [+0.1228, +0.2404] SEP (n 1,949 / 138).**
* In return refcv8 **misses fewer real decelerations: 0.345 vs 0.548, -0.2031 [-0.2707, -0.1435] SEP (n 832 / 80).**
* Net decision accuracy: 0.613 vs 0.661, -0.0485 [-0.0898, -0.0090] SEP.

The cell "refcv8 plans decel / refcv7 plans cruise" (591 win / 93 ep, 16 %) carries **113 % [68, 306]** of the ADE delta (seed 1: 95 % [59, 219]). This is a TRADE-OFF, not only a defect. refcv8 under-brakes less, which is the safety-relevant direction, and over-brakes more. S6 ADE weights the cruise windows, which are 2.3x more numerous than the deceleration windows.

**3. The false brake is decided at the tactical longitudinal (lon) head. The association is MEASURED; causation is UNVERIFIED.** Windows where refcv8's tactical lon head argmax (`nav_true.tacv6_lon_logits`) is BRAKE_TO (992 win / 71 ep) carry **109 % [61, 308]** of the ADE delta. Take the 374 GT-cruise windows where refcv8 plans decel and refcv7 cruises: the head says BRAKE_TO in **274 of them (73 %)**. Where refcv8's plan correctly cruises, it says BRAKE_TO in only 121 / 1,195 (10 %). When the GT `lon_v7` label is CRUISE (n 1,139), the head says BRAKE_TO in 354 (31 %). Windows where the head disagrees with `lon_v7` carry 89 % [50, 226]. The head's logits DO feed the planner: in `stack/tanitad/refs/refc_v3.py:1999-2027`, `hypothesis_posteriors(lat_logits, lon_logits)` goes into `r8_feed` and `lon_prior`. So a causal path exists, but an association is not an intervention.

**4. The ceiling and the speed label are ELIMINATED as the main cause (MEASURED).**
* The ceiling filter acts on 239 / 3,668 S6 windows. Its whole effect on refcv8 (ON - OFF, C2) is +0.0211 [-0.0397, +0.0817] m, not separated. The matched read with the ceiling off (C1) is still +0.1447 [+0.0154, +0.2842], with cruising at 149 % [91, 537].
* Where GT drives faster than the road-type label (417 win), the ceiling does cut the plan (-2.92 m signed; C2 +0.0850 SEP). That effect is real but at most about 13 % of the total.
* **The two arms do NOT get the same speed label.** refcv7-50,400 gets a per-clip CONTINUOUS `v_max_ms`: 136 values, constant within every clip, on average 1.94 m/s below the clip's max 6-s GT speed. refcv8 gets the 4-class road-type label. The two differ on 3,668 / 3,668 windows. Where refcv7's value comes from is UNVERIFIED.
* That information gap does NOT explain the regression. **With the label WITHHELD from BOTH arms (C3), the delta is +0.1448 [+0.0136, +0.2817] SEP**, with cruising at 139 % [82, 520]. refcv8's own label effect (C3b) is -0.0021 [-0.0635, +0.0635].

**5. What changed from 20k to 30k (C6 = EMA-30k minus EMA-20k on refcv8 itself): +0.0811 [+0.0268, +0.1357] SEP.**
* It is NOT the false brake. GT-cruise windows planned as decel: 541 at EMA-20k, 535 at EMA-30k.
* The growth comes from refcv8 **losing its advantage in slow and stopping windows**, where it became faster. Signed along-track change: +0.747 m at v0 < 2, +1.523 m in `stopping`, +1.227 m in `slow_both`, all SEP. v0 15-25 adds 32 % [3, 79] SEP, and cruise |along| adds 44 % [4, 78] SEP.
* EMA-20k vs refcv7 was +0.0847 [-0.0568, +0.2250], not separated, but it already had the same cruise false-brake pattern (cruising +0.1807 SEP). RAW-20k vs refcv7 is +0.1433 [+0.0166, +0.2685] SEP.
* S6 is getting WORSE as training continues, which matters for the 48,852 final.

**6. Lever L1 was RUN in this turn and FAILED (zero GPU): a post-hoc speed offset does not fix it.** We shifted each plan along its own path by a uniform speed offset, cross-fitted over 2 EPISODE folds (the fold used for fitting is never scored), and applied it to BOTH arms. The ADE-optimal global offset for refcv8 is +0.05 m/s in both folds. Calibrated refcv8 minus calibrated refcv7 is **+0.1703 [+0.0412, +0.3116] SEP**; with per-v0-bin offsets it is +0.1782 [+0.0566, +0.3095] SEP. A single bias, even one conditioned on speed, cannot tell "cruise" from "decelerate" at the same v0. The error is a per-window decision. This is the same conclusion as item 2, now shown by intervention.

### Next levers, RANKED BY MEASURED SHARE

| rank | lever | measured share of the C0 ADE delta | the experiment that tests it | compute | status |
|---|---|---|---|---|---|
| 1 | **Tactical lon over-braking in cruise** (BRAKE_TO false positives lead to a decel plan) | 113 % [68, 306] (decel-vs-cruise cell); 109 % [61, 308] (head says BRAKE_TO); seed 1: 95 % | **(a) Causal test, eval-only, no training.** Re-roll S6 with the tactical lon hypothesis FORCED via `set_r8_force(lon6=...)` (`refc_v3.py:2081`; EVAL-only, refused in training; it takes ONE class per call, so batch the windows by class). Arm F1 forces the GT-`lon_v7`-mapped LON6 class (oracle tactical lon). Arm F2 forces CRUISE wherever the head's argmax is BRAKE_TO. Pre-register: if F1 closes >= 50 % of the cruise contribution, the defect is in the head, not the decoder. **(b) The fix, if (a) confirms:** a cost-sensitive / prior-calibrated lon head (BRAKE_TO precision), fit on TRAIN, never on eval. | (a) GPU forward only, about one S6 battery roll; (b) retraining | **BLOCKED** for (a): the dev-box GPU is held by the G0 job under the shared lock, and the forcing API has **no caller in the battery** (its only caller is `stack/tests/test_refcv8_warm_start.py`; the wiring is a new roll script, which can be written on CPU now). (b) is a PI compute decision. |
| 2 | **20k -> 30k drift** in slow and stopping windows (refcv8 got faster where it used to win) | C6 +0.0811 [+0.0268, +0.1357] SEP; v0<2 20 % [3, 56], slow_both 21 % [7, 60], v0 15-25 32 % [3, 79] | Score this decomposition on a NON-eval validation split across 20k / 30k / final (the checkpoint must not be chosen on eval139). If the drift is monotone, change the LR decay or the EMA horizon. | GPU eval on a val split; retraining only if a schedule change is chosen | needs a val-split battery (none banked) |
| 3 | **Ceiling filter vs a wrong road-type label** (GT faster than the class) | C2 total +0.0211 [-0.0397, +0.0817] (n.s.); GT_above_label +0.0850 SEP; at most about 13 % of the total | Re-roll with a ceiling tolerance (cap at label x 1.1-1.2). The `filter_off` arm is its banked upper bound (C1: still +0.1447 SEP). | GPU re-roll; no training | small; cannot close the gap alone |

ELIMINATED this turn (logged, MEASURED): the speed-label information gap (C3), the ceiling as the primary cause (C1, C2), a scalar or v0-conditional speed bias (L1), and speed-profile compression (OLS slope). **Done-state under Rule Zero: the bar (BAR-R8-NR S6 ADE) is still FAILED. The remaining lever with the largest measured share (rank 1a) is blocked on the GPU lock plus the roll-script wiring, and its fix (1b) is blocked on a PI retraining decision.**
