# REFINE SHRINK -- does the refinement step shrink tight turns? (refcv8 vs refcv7)

- **Tier:** T1-style open loop on the S2 panel windows (own action, `os`); the unrefined anchor is the selection surface of the same window
- **Evidence class:** MEASURED (ours; per-window dumps listed under `inputs`)
- **Estimator:** paired episode-cluster bootstrap, ONE shared resample matrix (E=139 episodes, B=2000, rng seed 29); 'another draw of EPISODES, models fixed' only -- NOT another training run (H-ESTIM-SEED-1), NOT another inference run
- **n:** 4754 S2 windows / 139 episodes; sharp = 368 windows; curved = 1604. In-bank restriction (sel_idx < 117 in the view's seeds) shrinks the window set; n is printed per table.
- **Not answered:** another TRAINING run (one checkpoint per arm; H-ESTIM-SEED-1) and another INFERENCE run (seed rows only).

## 0. Controls (asserted before anything else)

**Rule 6 -- banked numbers reproduced through the same Ctx slope code, against `FAN_VS_SELECTOR.json`:**

| tree | slope r8 os | slope r8 oracle_sel | slope r7 os | slope r7 oracle_sel | n_seg | max abs diff (est, lo, hi vs banked) |
|---|---|---|---|---|---|---|
| EMA-15k | -0.140937 | -0.082618 | -0.092130 | -0.047932 | 363 | 0.0e+00 |
| EMA-5k | -0.130321 | -0.086370 | -0.092167 | -0.047922 | 364 | 0.0e+00 |

Literal task values: EMA-15k os -0.1409, oracle r8 -0.083, r7 -0.048 -> **PASS = True**.

**Frame / pairing controls (MEASURED):**

- plan_full[:, :4] == os (max abs m), EMA-15k|s0: `0`
- sel_bank rows finite iff sel_idx<117, EMA-15k|s0: `True`
- frac in-bank (all windows / sharp), EMA-15k|s0: `[0.8756836348338242, 0.9320652173913043]`
- plan_full[:, :4] == os (max abs m), EMA-15k|s1: `0`
- sel_bank rows finite iff sel_idx<117, EMA-15k|s1: `True`
- frac in-bank (all windows / sharp), EMA-15k|s1: `[0.8740008413967185, 0.9565217391304348]`
- plan_full[:, :4] == os (max abs m), EMA-5k|s0: `0`
- sel_bank rows finite iff sel_idx<117, EMA-5k|s0: `True`
- frac in-bank (all windows / sharp), EMA-5k|s0: `[0.8228859907446361, 0.9211956521739131]`
- plan_full[:, :4] == os (max abs m), EMA-5k|s1: `0`
- sel_bank rows finite iff sel_idx<117, EMA-5k|s1: `True`
- frac in-bank (all windows / sharp), EMA-5k|s1: `[0.8108960875052588, 0.9456521739130435]`
- plan_full[:, :4] == os (max abs m), r7|s0: `0`
- sel_bank rows finite iff sel_idx<117, r7|s0: `True`
- frac in-bank (all windows / sharp), r7|s0: `[1.0, 1.0]`
- plan_full[:, :4] == os (max abs m), r7|s1: `0`
- sel_bank rows finite iff sel_idx<117, r7|s1: `True`
- frac in-bank (all windows / sharp), r7|s1: `[1.0, 1.0]`
- same selected idx r8/r7 -> sel_bank identical (max abs m; n windows), s0: `[0.0, 2654]`
- same selected idx r8/r7 -> sel_bank identical (max abs m; n windows), s1: `[0.0, 2638]`
- gt_future_ext identical r8 EMA-15k vs r7 (max abs): `0`
- GT frame/time control: ego_xy(gt_future_ext[:, i]) vs panel g[:,0] (0.5 s), max abs m for i=3/4/5: `{3: 3.650768238446018, 4: 2.328524139727506e-06, 5: 3.6507684267128866}`
- known-value control: slope of GT-as-plan: `0`
- mispaired control (unrefined r8, shifted by 1009 in-bank windows (stride-1 windows: a 1-window shift hits a near-identical neighbour)): slope vs paired slope: `[0.017647348737319636, -0.23858466087449123]`

## 1. Refinement effect: unrefined selected anchor (a) vs refined emitted plan (b)

Negative slope = the path turns LESS than GT. effect = b - a (negative = refinement shrinks the turn). `GAP_refined = GAP_unrefined + DIFF_EFFECT` is an exact identity (asserted on all bootstrap draws).

### EMA-15k vs refcv7-50,400

**2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 335, n_seg 335, n_ep 41, mean GT turn-aligned heading 25.5 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2386 [-0.3155, -0.1647] SEP |
| slope r8 REFINED emitted plan | -0.1032 [-0.1835, -0.0422] SEP |
| slope r7 UNREFINED selected anchor | -0.2714 [-0.3499, -0.1998] SEP |
| slope r7 REFINED emitted plan | -0.0711 [-0.1458, -0.0129] SEP |
| refinement effect r8 (refined - unrefined) | +0.1354 [+0.0889, +0.1847] SEP |
| refinement effect r7 (refined - unrefined) | +0.2003 [+0.1465, +0.2651] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0649 [-0.0923, -0.0398] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0328 [-0.0010, +0.0630] |
| GAP_refined (r8 - r7, emitted plans) | -0.0321 [-0.0615, -0.0089] SEP |
| share DIFF EFFECT of GAP refined | 2.02 [0.95, 6.15] |
| share GAP unrefined of GAP refined | -1.02 [-5.15, 0.05] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.567 [+0.377, +0.783] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.738 [+0.542, +0.945] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.171 [-0.248, -0.097] SEP |

**2-seed mean | sharp (GT |yaw| >= 0.15) | segments 2..K** -- n_win 336, n_seg 1007, n_ep 41, mean GT turn-aligned heading 18.0 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2577 [-0.3370, -0.1803] SEP |
| slope r8 REFINED emitted plan | -0.1172 [-0.1986, -0.0543] SEP |
| slope r7 UNREFINED selected anchor | -0.2932 [-0.3742, -0.2173] SEP |
| slope r7 REFINED emitted plan | -0.0842 [-0.1600, -0.0234] SEP |
| refinement effect r8 (refined - unrefined) | +0.1405 [+0.0903, +0.1918] SEP |
| refinement effect r7 (refined - unrefined) | +0.2090 [+0.1514, +0.2760] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0685 [-0.0956, -0.0430] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0356 [+0.0043, +0.0641] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0330 [-0.0602, -0.0105] SEP |
| share DIFF EFFECT of GAP refined | 2.08 [1.07, 5.75] |
| share GAP unrefined of GAP refined | -1.08 [-4.75, -0.07] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.545 [+0.362, +0.742] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.713 [+0.526, +0.909] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.168 [-0.238, -0.099] SEP |

**2-seed mean | curved (GT |yaw| >= 0.02) | last segment (1.5->2.0 s chord)** -- n_win 1281, n_seg 1281, n_ep 99, mean GT turn-aligned heading 11.1 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2110 [-0.2772, -0.1466] SEP |
| slope r8 REFINED emitted plan | -0.0972 [-0.1656, -0.0440] SEP |
| slope r7 UNREFINED selected anchor | -0.2414 [-0.3070, -0.1772] SEP |
| slope r7 REFINED emitted plan | -0.0633 [-0.1273, -0.0125] SEP |
| refinement effect r8 (refined - unrefined) | +0.1138 [+0.0718, +0.1574] SEP |
| refinement effect r7 (refined - unrefined) | +0.1781 [+0.1303, +0.2331] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0642 [-0.0873, -0.0428] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0303 [+0.0002, +0.0568] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0339 [-0.0590, -0.0134] SEP |
| share DIFF EFFECT of GAP refined | 1.89 [1.00, 4.53] |
| share GAP unrefined of GAP refined | -0.89 [-3.53, -0.00] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.539 [+0.350, +0.742] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.738 [+0.546, +0.933] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.198 [-0.274, -0.130] SEP |

**2-seed mean | curved (GT |yaw| >= 0.02) | segments 2..K** -- n_win 1294, n_seg 3864, n_ep 99, mean GT turn-aligned heading 7.8 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2800 [-0.3968, -0.1816] SEP |
| slope r8 REFINED emitted plan | -0.1653 [-0.2928, -0.0666] SEP |
| slope r7 UNREFINED selected anchor | -0.3114 [-0.4242, -0.2139] SEP |
| slope r7 REFINED emitted plan | -0.1326 [-0.2627, -0.0371] SEP |
| refinement effect r8 (refined - unrefined) | +0.1146 [+0.0715, +0.1613] SEP |
| refinement effect r7 (refined - unrefined) | +0.1788 [+0.1279, +0.2383] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0642 [-0.0884, -0.0435] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0315 [+0.0050, +0.0563] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0327 [-0.0551, -0.0139] SEP |
| share DIFF EFFECT of GAP refined | 1.96 [1.11, 4.19] |
| share GAP unrefined of GAP refined | -0.96 [-3.19, -0.11] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.409 [+0.233, +0.656] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.574 [+0.361, +0.848] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.165 [-0.240, -0.105] SEP |

**seed 0 | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 340, n_seg 340, n_ep 42, mean GT turn-aligned heading 25.4 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2415 [-0.3192, -0.1671] SEP |
| slope r8 REFINED emitted plan | -0.1076 [-0.1876, -0.0456] SEP |
| slope r7 UNREFINED selected anchor | -0.2792 [-0.3605, -0.2054] SEP |
| slope r7 REFINED emitted plan | -0.0784 [-0.1572, -0.0180] SEP |
| refinement effect r8 (refined - unrefined) | +0.1339 [+0.0867, +0.1832] SEP |
| refinement effect r7 (refined - unrefined) | +0.2007 [+0.1443, +0.2657] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0668 [-0.0962, -0.0413] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0377 [+0.0016, +0.0692] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0291 [-0.0598, -0.0056] SEP |
| share DIFF EFFECT of GAP refined | 2.29 [0.98, 9.53] |
| share GAP unrefined of GAP refined | -1.29 [-8.53, 0.02] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.555 [+0.365, +0.761] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.719 [+0.522, +0.924] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.164 [-0.244, -0.094] SEP |

**seed 1 | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 347, n_seg 347, n_ep 41, mean GT turn-aligned heading 25.5 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2445 [-0.3203, -0.1703] SEP |
| slope r8 REFINED emitted plan | -0.1132 [-0.1937, -0.0473] SEP |
| slope r7 UNREFINED selected anchor | -0.2698 [-0.3480, -0.2015] SEP |
| slope r7 REFINED emitted plan | -0.0763 [-0.1520, -0.0157] SEP |
| refinement effect r8 (refined - unrefined) | +0.1313 [+0.0872, +0.1777] SEP |
| refinement effect r7 (refined - unrefined) | +0.1935 [+0.1407, +0.2548] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0622 [-0.0897, -0.0379] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0253 [-0.0062, +0.0540] |
| GAP_refined (r8 - r7, emitted plans) | -0.0369 [-0.0670, -0.0122] SEP |
| share DIFF EFFECT of GAP refined | 1.69 [0.89, 4.67] |
| share GAP unrefined of GAP refined | -0.69 [-3.67, 0.11] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.537 [+0.352, +0.754] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.717 [+0.517, +0.933] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.180 [-0.260, -0.103] SEP |

### EMA-5k vs refcv7-50,400

**2-seed mean | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 321, n_seg 321, n_ep 42, mean GT turn-aligned heading 25.0 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2054 [-0.2912, -0.1320] SEP |
| slope r8 REFINED emitted plan | -0.1041 [-0.1802, -0.0464] SEP |
| slope r7 UNREFINED selected anchor | -0.2662 [-0.3504, -0.1911] SEP |
| slope r7 REFINED emitted plan | -0.0768 [-0.1533, -0.0179] SEP |
| refinement effect r8 (refined - unrefined) | +0.1013 [+0.0570, +0.1509] SEP |
| refinement effect r7 (refined - unrefined) | +0.1895 [+0.1429, +0.2435] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0882 [-0.1072, -0.0711] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0608 [+0.0417, +0.0816] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0273 [-0.0394, -0.0158] SEP |
| share DIFF EFFECT of GAP refined | 3.23 [2.19, 5.47] |
| share GAP unrefined of GAP refined | -2.23 [-4.47, -1.19] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.493 [+0.299, +0.709] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.712 [+0.529, +0.915] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.218 [-0.301, -0.159] SEP |

**seed 0 | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 335, n_seg 335, n_ep 42, mean GT turn-aligned heading 25.1 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2158 [-0.3062, -0.1386] SEP |
| slope r8 REFINED emitted plan | -0.1173 [-0.2054, -0.0519] SEP |
| slope r7 UNREFINED selected anchor | -0.2784 [-0.3684, -0.1989] SEP |
| slope r7 REFINED emitted plan | -0.0877 [-0.1748, -0.0219] SEP |
| refinement effect r8 (refined - unrefined) | +0.0985 [+0.0498, +0.1526] SEP |
| refinement effect r7 (refined - unrefined) | +0.1906 [+0.1386, +0.2523] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0921 [-0.1145, -0.0730] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0626 [+0.0419, +0.0848] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0295 [-0.0412, -0.0191] SEP |
| share DIFF EFFECT of GAP refined | 3.12 [2.14, 4.82] |
| share GAP unrefined of GAP refined | -2.12 [-3.82, -1.14] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.457 [+0.244, +0.692] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.685 [+0.484, +0.907] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.228 [-0.321, -0.164] SEP |

**seed 1 | sharp (GT |yaw| >= 0.15) | last segment (1.5->2.0 s chord)** -- n_win 344, n_seg 344, n_ep 42, mean GT turn-aligned heading 25.2 deg; identity err 4.2e-17

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2146 [-0.2980, -0.1454] SEP |
| slope r8 REFINED emitted plan | -0.1089 [-0.1876, -0.0485] SEP |
| slope r7 UNREFINED selected anchor | -0.2720 [-0.3509, -0.2023] SEP |
| slope r7 REFINED emitted plan | -0.0837 [-0.1625, -0.0223] SEP |
| refinement effect r8 (refined - unrefined) | +0.1056 [+0.0600, +0.1584] SEP |
| refinement effect r7 (refined - unrefined) | +0.1883 [+0.1392, +0.2447] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0826 [-0.0977, -0.0686] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0574 [+0.0375, +0.0800] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0252 [-0.0401, -0.0109] SEP |
| share DIFF EFFECT of GAP refined | 3.28 [2.02, 8.23] |
| share GAP unrefined of GAP refined | -2.28 [-7.23, -1.02] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.492 [+0.290, +0.716] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.692 [+0.500, +0.906] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.200 [-0.261, -0.148] SEP |


### 1a. Same-anchor split (sharp, last segment): where r8 and r7 select the SAME bank index their unrefined paths are bit-identical (GAP_unrefined = 0 asserted), so DIFF_EFFECT there is the refinement FUNCTION on identical prior paths

**EMA-15k | 2-seed mean | SAME anchor index in r8 and r7 | sharp | last segment** -- n_win 127, n_ep 37

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2711 [-0.4063, -0.1585] SEP |
| slope r8 REFINED emitted plan | -0.1432 [-0.2857, -0.0384] SEP |
| slope r7 UNREFINED selected anchor | -0.2711 [-0.4063, -0.1585] SEP |
| slope r7 REFINED emitted plan | -0.1146 [-0.2583, -0.0086] SEP |
| refinement effect r8 (refined - unrefined) | +0.1279 [+0.0710, +0.1893] SEP |
| refinement effect r7 (refined - unrefined) | +0.1565 [+0.0996, +0.2230] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0286 [-0.0486, -0.0123] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0000 [+0.0000, +0.0000] |
| GAP_refined (r8 - r7, emitted plans) | -0.0286 [-0.0486, -0.0123] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.472 [+0.249, +0.785] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.577 [+0.338, +0.952] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.106 [-0.217, -0.041] SEP |

**EMA-15k | 2-seed mean | DIFFERENT anchor index | sharp | last segment** -- n_win 208, n_ep 41

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2196 [-0.2894, -0.1494] SEP |
| slope r8 REFINED emitted plan | -0.0799 [-0.1427, -0.0231] SEP |
| slope r7 UNREFINED selected anchor | -0.2715 [-0.3400, -0.2024] SEP |
| slope r7 REFINED emitted plan | -0.0457 [-0.1014, +0.0074] |
| refinement effect r8 (refined - unrefined) | +0.1397 [+0.0909, +0.1902] SEP |
| refinement effect r7 (refined - unrefined) | +0.2258 [+0.1641, +0.3015] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0860 [-0.1264, -0.0510] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0519 [-0.0016, +0.0982] |
| GAP_refined (r8 - r7, emitted plans) | -0.0341 [-0.0752, -0.0017] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.636 [+0.440, +0.866] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.832 [+0.652, +1.028] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.195 [-0.313, -0.082] SEP |

**EMA-15k | seed 0 | SAME anchor index in r8 and r7 | sharp | last segment** -- n_win 157, n_ep 39

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2763 [-0.3875, -0.1770] SEP |
| slope r8 REFINED emitted plan | -0.1442 [-0.2630, -0.0523] SEP |
| slope r7 UNREFINED selected anchor | -0.2763 [-0.3875, -0.1770] SEP |
| slope r7 REFINED emitted plan | -0.1128 [-0.2304, -0.0213] SEP |
| refinement effect r8 (refined - unrefined) | +0.1321 [+0.0790, +0.1885] SEP |
| refinement effect r7 (refined - unrefined) | +0.1636 [+0.1085, +0.2260] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0315 [-0.0488, -0.0162] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0000 [+0.0000, +0.0000] |
| GAP_refined (r8 - r7, emitted plans) | -0.0315 [-0.0488, -0.0162] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.478 [+0.273, +0.749] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.592 [+0.372, +0.894] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.114 [-0.191, -0.055] SEP |

**EMA-15k | seed 0 | DIFFERENT anchor index | sharp | last segment** -- n_win 183, n_ep 41

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2109 [-0.2855, -0.1402] SEP |
| slope r8 REFINED emitted plan | -0.0754 [-0.1402, -0.0208] SEP |
| slope r7 UNREFINED selected anchor | -0.2817 [-0.3586, -0.2049] SEP |
| slope r7 REFINED emitted plan | -0.0483 [-0.1073, +0.0050] |
| refinement effect r8 (refined - unrefined) | +0.1355 [+0.0821, +0.1873] SEP |
| refinement effect r7 (refined - unrefined) | +0.2334 [+0.1602, +0.3141] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0979 [-0.1462, -0.0545] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0708 [+0.0031, +0.1286] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0271 [-0.0773, +0.0136] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.643 [+0.442, +0.870] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.829 [+0.640, +1.019] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.186 [-0.331, -0.056] SEP |

**EMA-15k | seed 1 | SAME anchor index in r8 and r7 | sharp | last segment** -- n_win 160, n_ep 38

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2571 [-0.3794, -0.1564] SEP |
| slope r8 REFINED emitted plan | -0.1300 [-0.2442, -0.0409] SEP |
| slope r7 UNREFINED selected anchor | -0.2571 [-0.3794, -0.1564] SEP |
| slope r7 REFINED emitted plan | -0.1001 [-0.2182, -0.0132] SEP |
| refinement effect r8 (refined - unrefined) | +0.1272 [+0.0779, +0.1818] SEP |
| refinement effect r7 (refined - unrefined) | +0.1570 [+0.1067, +0.2146] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0298 [-0.0509, -0.0132] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0000 [+0.0000, +0.0000] |
| GAP_refined (r8 - r7, emitted plans) | -0.0298 [-0.0509, -0.0132] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.495 [+0.292, +0.767] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.611 [+0.393, +0.926] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.116 [-0.226, -0.048] SEP |

**EMA-15k | seed 1 | DIFFERENT anchor index | sharp | last segment** -- n_win 187, n_ep 41

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2337 [-0.3114, -0.1571] SEP |
| slope r8 REFINED emitted plan | -0.0988 [-0.1746, -0.0317] SEP |
| slope r7 UNREFINED selected anchor | -0.2807 [-0.3493, -0.2117] SEP |
| slope r7 REFINED emitted plan | -0.0558 [-0.1258, +0.0080] |
| refinement effect r8 (refined - unrefined) | +0.1349 [+0.0755, +0.1902] SEP |
| refinement effect r7 (refined - unrefined) | +0.2249 [+0.1550, +0.3048] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0900 [-0.1383, -0.0492] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0470 [-0.0115, +0.1041] |
| GAP_refined (r8 - r7, emitted plans) | -0.0430 [-0.0893, -0.0042] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.577 [+0.346, +0.818] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.801 [+0.584, +1.032] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.224 [-0.347, -0.102] SEP |

**EMA-5k | 2-seed mean | SAME anchor index in r8 and r7 | sharp | last segment** -- n_win 166, n_ep 41

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2425 [-0.3580, -0.1477] SEP |
| slope r8 REFINED emitted plan | -0.1503 [-0.2663, -0.0633] SEP |
| slope r7 UNREFINED selected anchor | -0.2425 [-0.3580, -0.1477] SEP |
| slope r7 REFINED emitted plan | -0.1132 [-0.2303, -0.0276] SEP |
| refinement effect r8 (refined - unrefined) | +0.0922 [+0.0432, +0.1464] SEP |
| refinement effect r7 (refined - unrefined) | +0.1293 [+0.0803, +0.1834] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0372 [-0.0467, -0.0289] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0000 [+0.0000, +0.0000] |
| GAP_refined (r8 - r7, emitted plans) | -0.0372 [-0.0467, -0.0289] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.380 [+0.186, +0.630] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.533 [+0.318, +0.830] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.153 [-0.261, -0.092] SEP |

**EMA-5k | 2-seed mean | DIFFERENT anchor index | sharp | last segment** -- n_win 155, n_ep 42

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.1623 [-0.2318, -0.0922] SEP |
| slope r8 REFINED emitted plan | -0.0504 [-0.1034, -0.0078] SEP |
| slope r7 UNREFINED selected anchor | -0.2938 [-0.3670, -0.2226] SEP |
| slope r7 REFINED emitted plan | -0.0345 [-0.0887, +0.0111] |
| refinement effect r8 (refined - unrefined) | +0.1119 [+0.0545, +0.1686] SEP |
| refinement effect r7 (refined - unrefined) | +0.2593 [+0.2038, +0.3201] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.1474 [-0.1847, -0.1168] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.1315 [+0.0970, +0.1705] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0159 [-0.0363, +0.0043] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.690 [+0.429, +0.933] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.883 [+0.730, +1.044] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.193 [-0.340, -0.079] SEP |

**EMA-5k | seed 0 | SAME anchor index in r8 and r7 | sharp | last segment** -- n_win 203, n_ep 42

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2505 [-0.3602, -0.1532] SEP |
| slope r8 REFINED emitted plan | -0.1640 [-0.2823, -0.0712] SEP |
| slope r7 UNREFINED selected anchor | -0.2505 [-0.3602, -0.1532] SEP |
| slope r7 REFINED emitted plan | -0.1276 [-0.2484, -0.0341] SEP |
| refinement effect r8 (refined - unrefined) | +0.0865 [+0.0355, +0.1424] SEP |
| refinement effect r7 (refined - unrefined) | +0.1229 [+0.0725, +0.1791] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0364 [-0.0450, -0.0284] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0000 [+0.0000, +0.0000] |
| GAP_refined (r8 - r7, emitted plans) | -0.0364 [-0.0450, -0.0284] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.345 [+0.134, +0.604] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.491 [+0.266, +0.801] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.145 [-0.250, -0.088] SEP |

**EMA-5k | seed 0 | DIFFERENT anchor index | sharp | last segment** -- n_win 132, n_ep 37

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.1569 [-0.2366, -0.0804] SEP |
| slope r8 REFINED emitted plan | -0.0380 [-0.0968, +0.0098] |
| slope r7 UNREFINED selected anchor | -0.3257 [-0.4145, -0.2409] SEP |
| slope r7 REFINED emitted plan | -0.0201 [-0.0770, +0.0288] |
| refinement effect r8 (refined - unrefined) | +0.1189 [+0.0509, +0.1870] SEP |
| refinement effect r7 (refined - unrefined) | +0.3056 [+0.2360, +0.3855] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.1867 [-0.2381, -0.1427] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.1688 [+0.1215, +0.2192] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0179 [-0.0398, +0.0049] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.758 [+0.438, +1.086] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.938 [+0.789, +1.107] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.180 [-0.382, +0.011] |

**EMA-5k | seed 1 | SAME anchor index in r8 and r7 | sharp | last segment** -- n_win 209, n_ep 41

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.2476 [-0.3537, -0.1598] SEP |
| slope r8 REFINED emitted plan | -0.1404 [-0.2412, -0.0607] SEP |
| slope r7 UNREFINED selected anchor | -0.2476 [-0.3537, -0.1598] SEP |
| slope r7 REFINED emitted plan | -0.1038 [-0.2071, -0.0245] SEP |
| refinement effect r8 (refined - unrefined) | +0.1072 [+0.0563, +0.1633] SEP |
| refinement effect r7 (refined - unrefined) | +0.1438 [+0.0921, +0.1995] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0366 [-0.0464, -0.0277] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0000 [+0.0000, +0.0000] |
| GAP_refined (r8 - r7, emitted plans) | -0.0366 [-0.0464, -0.0277] SEP |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.433 [+0.231, +0.672] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.581 [+0.366, +0.859] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.148 [-0.233, -0.092] SEP |

**EMA-5k | seed 1 | DIFFERENT anchor index | sharp | last segment** -- n_win 135, n_ep 40

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | -0.1584 [-0.2262, -0.0931] SEP |
| slope r8 REFINED emitted plan | -0.0555 [-0.1075, -0.0125] SEP |
| slope r7 UNREFINED selected anchor | -0.3135 [-0.3690, -0.2553] SEP |
| slope r7 REFINED emitted plan | -0.0497 [-0.1090, +0.0080] |
| refinement effect r8 (refined - unrefined) | +0.1029 [+0.0458, +0.1648] SEP |
| refinement effect r7 (refined - unrefined) | +0.2638 [+0.2094, +0.3308] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.1609 [-0.1933, -0.1297] SEP |
| GAP_unrefined (r8 - r7, selected anchors) | +0.1551 [+0.1097, +0.2013] SEP |
| GAP_refined (r8 - r7, emitted plans) | -0.0058 [-0.0405, +0.0256] |
| recovery fraction r8 (1 - refined/unrefined slope) (DERIVED) | +0.650 [+0.376, +0.903] SEP |
| recovery fraction r7 (1 - refined/unrefined slope) (DERIVED) | +0.842 [+0.676, +1.026] SEP |
| recovery fraction r8 - r7 (DERIVED) | -0.192 [-0.346, -0.054] SEP |


### 1b. Same, on the 5.0->6.0 s terminal chord (SECONDARY; large angles)

_heading of the 5.0->6.0 s chord (model slots 6->7) minus GT's 5->6 s chord heading (ego frame at t0), same through-origin turn-aligned slope; needs GT not end-clamped and both chords >= 1 m; SECONDARY (large angles, wrap possible)_

**EMA-15k | 2-seed mean | sharp** -- n_win 243, n_ep 32, mean GT turn-aligned terminal heading 54.4 deg

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | +0.1786 [+0.0773, +0.3158] SEP |
| slope r8 REFINED emitted plan | -0.0591 [-0.1515, +0.0340] |
| slope r7 UNREFINED selected anchor | +0.1750 [+0.1052, +0.2849] SEP |
| slope r7 REFINED emitted plan | -0.0760 [-0.1363, +0.0027] |
| refinement effect r8 (refined - unrefined) | -0.2376 [-0.3334, -0.1773] SEP |
| refinement effect r7 (refined - unrefined) | -0.2510 [-0.3360, -0.1933] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | +0.0134 [-0.0233, +0.0526] |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0035 [-0.0812, +0.0738] |
| GAP_refined (r8 - r7, emitted plans) | +0.0169 [-0.0626, +0.0716] |

**EMA-5k | 2-seed mean | sharp** -- n_win 228, n_ep 32, mean GT turn-aligned terminal heading 53.8 deg

| term | est [95 % CI] |
|---|---|
| slope r8 UNREFINED selected anchor | +0.2427 [+0.1586, +0.3897] SEP |
| slope r8 REFINED emitted plan | -0.0185 [-0.0839, +0.0712] |
| slope r7 UNREFINED selected anchor | +0.1781 [+0.1041, +0.2982] SEP |
| slope r7 REFINED emitted plan | -0.0654 [-0.1270, +0.0189] |
| refinement effect r8 (refined - unrefined) | -0.2612 [-0.3589, -0.1996] SEP |
| refinement effect r7 (refined - unrefined) | -0.2434 [-0.3301, -0.1885] SEP |
| DIFF_EFFECT (effect r8 - effect r7) | -0.0178 [-0.0570, +0.0124] |
| GAP_unrefined (r8 - r7, selected anchors) | +0.0647 [+0.0316, +0.1154] SEP |
| GAP_refined (r8 - r7, emitted plans) | +0.0469 [+0.0290, +0.0733] SEP |


## 3. Fan-level coverage on sharp turns (terminal 5-6 s heading of every refined candidate vs the GT terminal heading)

GT terminal heading = heading of the GT 5.0->6.0 s chord (gt_future_ext idx 49->59, ego frame at t0, convention verified against panel g); candidate terminal heading = fan_term_heading_nav_true (chord of the REFINED candidate's slots 6->7 = 5.0->6.0 s); error = wrap(candidate - GT).  hit@tau = at least one candidate in the subset has |error| <= tau.  subsets: ALL = every emitted candidate (r8 149 = 117 bank + 32 allocator extras; r7 117), BANK117 = first 117 columns, SURV = candidates the S2 reach mask keeps.

n sharp 368; with valid 6 s GT chord (>= 1.0 m, not end-clamped) **262 windows / 32 episodes**. Values are 2-seed means of per-window indicators; per-seed rows follow.

### EMA-15k vs refcv7

**subset ALL**

| metric | n | r8 | r7 | r8 - r7 | seed 0 r8 - r7 | seed 1 r8 - r7 |
|---|---|---|---|---|---|---|
| hit@0.1806 rad (10.3 deg) | 262 | +0.992 [+0.982, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | -0.008 [-0.018, +0.000] | -0.008 [-0.018, +0.000] | -0.008 [-0.018, +0.000] |
| hit@0.35 rad (20.1 deg) | 262 | +0.996 [+0.989, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] |
| mean best |err| (deg) | 262 | +0.737 [+0.513, +1.014] SEP | +0.956 [+0.806, +1.137] SEP | -0.220 [-0.480, +0.050] | -0.187 [-0.478, +0.108] | -0.253 [-0.513, -0.008] SEP |
| mean best-candidate signed turn-aligned err (deg; <0 = best candidate still under-rotated) | 262 | -0.037 [-0.337, +0.166] | -0.026 [-0.143, +0.101] | -0.010 [-0.382, +0.262] | -0.036 [-0.431, +0.261] | +0.016 [-0.367, +0.321] |
| frac of candidates within 0.1806 rad | 262 | +0.146 [+0.129, +0.164] SEP | +0.090 [+0.078, +0.101] SEP | +0.056 [+0.047, +0.067] SEP | +0.056 [+0.047, +0.067] SEP | +0.056 [+0.047, +0.067] SEP |

**subset BANK117**

| metric | n | r8 | r7 | r8 - r7 | seed 0 r8 - r7 | seed 1 r8 - r7 |
|---|---|---|---|---|---|---|
| hit@0.1806 rad (10.3 deg) | 262 | +0.992 [+0.982, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | -0.008 [-0.018, +0.000] | -0.008 [-0.018, +0.000] | -0.008 [-0.018, +0.000] |
| hit@0.35 rad (20.1 deg) | 262 | +0.996 [+0.989, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] |
| mean best |err| (deg) | 262 | +0.891 [+0.657, +1.177] SEP | +0.956 [+0.806, +1.137] SEP | -0.065 [-0.310, +0.193] | -0.056 [-0.330, +0.235] | -0.074 [-0.321, +0.173] |
| mean best-candidate signed turn-aligned err (deg; <0 = best candidate still under-rotated) | 262 | -0.036 [-0.337, +0.149] | -0.026 [-0.143, +0.101] | -0.010 [-0.381, +0.253] | -0.036 [-0.432, +0.259] | +0.016 [-0.381, +0.339] |
| frac of candidates within 0.1806 rad | 262 | +0.141 [+0.116, +0.166] SEP | +0.090 [+0.078, +0.101] SEP | +0.051 [+0.036, +0.066] SEP | +0.051 [+0.036, +0.066] SEP | +0.051 [+0.036, +0.066] SEP |

**subset SURV**

| metric | n | r8 | r7 | r8 - r7 | seed 0 r8 - r7 | seed 1 r8 - r7 |
|---|---|---|---|---|---|---|
| hit@0.1806 rad (10.3 deg) | 262 | +0.992 [+0.982, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | -0.008 [-0.018, +0.000] | -0.008 [-0.018, +0.000] | -0.008 [-0.018, +0.000] |
| hit@0.35 rad (20.1 deg) | 262 | +0.996 [+0.989, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] |
| mean best |err| (deg) | 262 | +0.737 [+0.513, +1.014] SEP | +0.956 [+0.806, +1.137] SEP | -0.220 [-0.480, +0.050] | -0.187 [-0.478, +0.108] | -0.253 [-0.513, -0.008] SEP |
| mean best-candidate signed turn-aligned err (deg; <0 = best candidate still under-rotated) | 262 | -0.037 [-0.337, +0.166] | -0.026 [-0.143, +0.101] | -0.010 [-0.382, +0.262] | -0.036 [-0.431, +0.261] | +0.016 [-0.367, +0.321] |
| frac of candidates within 0.1806 rad | 262 | +0.146 [+0.129, +0.164] SEP | +0.090 [+0.078, +0.101] SEP | +0.056 [+0.047, +0.067] SEP | +0.056 [+0.047, +0.067] SEP | +0.056 [+0.047, +0.067] SEP |

### EMA-5k vs refcv7

**subset ALL**

| metric | n | r8 | r7 | r8 - r7 | seed 0 r8 - r7 | seed 1 r8 - r7 |
|---|---|---|---|---|---|---|
| hit@0.1806 rad (10.3 deg) | 262 | +1.000 [+1.000, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| hit@0.35 rad (20.1 deg) | 262 | +1.000 [+1.000, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mean best |err| (deg) | 262 | +0.814 [+0.661, +0.967] SEP | +0.956 [+0.806, +1.137] SEP | -0.142 [-0.300, -0.001] SEP | -0.082 [-0.274, +0.105] | -0.202 [-0.367, -0.063] SEP |
| mean best-candidate signed turn-aligned err (deg; <0 = best candidate still under-rotated) | 262 | -0.068 [-0.252, +0.089] | -0.026 [-0.143, +0.101] | -0.041 [-0.275, +0.163] | -0.072 [-0.359, +0.192] | -0.010 [-0.284, +0.234] |
| frac of candidates within 0.1806 rad | 262 | +0.109 [+0.098, +0.120] SEP | +0.090 [+0.078, +0.101] SEP | +0.019 [+0.014, +0.025] SEP | +0.020 [+0.015, +0.027] SEP | +0.018 [+0.013, +0.024] SEP |

**subset BANK117**

| metric | n | r8 | r7 | r8 - r7 | seed 0 r8 - r7 | seed 1 r8 - r7 |
|---|---|---|---|---|---|---|
| hit@0.1806 rad (10.3 deg) | 262 | +1.000 [+1.000, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| hit@0.35 rad (20.1 deg) | 262 | +1.000 [+1.000, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mean best |err| (deg) | 262 | +0.967 [+0.784, +1.162] SEP | +0.956 [+0.806, +1.137] SEP | +0.010 [-0.157, +0.183] | +0.027 [-0.163, +0.241] | -0.007 [-0.170, +0.162] |
| mean best-candidate signed turn-aligned err (deg; <0 = best candidate still under-rotated) | 262 | -0.067 [-0.253, +0.092] | -0.026 [-0.143, +0.101] | -0.040 [-0.269, +0.167] | -0.011 [-0.276, +0.224] | -0.069 [-0.355, +0.197] |
| frac of candidates within 0.1806 rad | 262 | +0.107 [+0.089, +0.125] SEP | +0.090 [+0.078, +0.101] SEP | +0.017 [+0.010, +0.024] SEP | +0.018 [+0.011, +0.025] SEP | +0.016 [+0.008, +0.024] SEP |

**subset SURV**

| metric | n | r8 | r7 | r8 - r7 | seed 0 r8 - r7 | seed 1 r8 - r7 |
|---|---|---|---|---|---|---|
| hit@0.1806 rad (10.3 deg) | 262 | +1.000 [+1.000, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| hit@0.35 rad (20.1 deg) | 262 | +1.000 [+1.000, +1.000] SEP | +1.000 [+1.000, +1.000] SEP | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mean best |err| (deg) | 262 | +0.814 [+0.661, +0.967] SEP | +0.956 [+0.806, +1.137] SEP | -0.142 [-0.300, -0.001] SEP | -0.082 [-0.274, +0.105] | -0.202 [-0.367, -0.063] SEP |
| mean best-candidate signed turn-aligned err (deg; <0 = best candidate still under-rotated) | 262 | -0.068 [-0.252, +0.089] | -0.026 [-0.143, +0.101] | -0.041 [-0.275, +0.163] | -0.072 [-0.359, +0.192] | -0.010 [-0.284, +0.234] |
| frac of candidates within 0.1806 rad | 262 | +0.109 [+0.098, +0.120] SEP | +0.090 [+0.078, +0.101] SEP | +0.019 [+0.014, +0.025] SEP | +0.020 [+0.015, +0.027] SEP | +0.018 [+0.013, +0.024] SEP |


runtime 4 s; free commit at start/end 6.89/5.90 GB.

(Reading note for section 3: hit@tau rates sit at 0.99-1.00 for both models, so the printed "SEP" on a rate whose CI is [1,1] only means the CI is narrow; the discriminating rows are `mean best |err|` and `frac of candidates within tau`.)

## 4. Feasibility only (NOTHING executed, no GPU, no model built): can the battery evaluate refcv8 with the allocator extras disabled?

Source = the refcv8 launch tree `C:/lgs/Lfb47a58376/stack` (launch-gate tree) and `E:/Projects/TanitAD/taniteval/tools/refcv3_arm.py`. Evidence class: MEASURED-FROM-SOURCE (read, not run).

- **There is no flag or config override for it today.** The dumps' manifest carries `eval_time_cfg_overrides: {}`; the only eval-time overrides in `refcv3_arm.py:1238-1262` are the two `seam_fail_patience` zeroings; a grep of the battery `code/` and of `refcv3_arm.py` for `n_alloc` / `r8_emit_live` finds nothing.
- **The extras come from `n_alloc`; they are SELECTABLE only through an emission switch.** `refc.py:2509` builds the 32 allocated candidates when `cfg.n_alloc > 0`; `refc.py:2578-2581` sets `emit = (kind == 0)` and adds the extras only `if bool(cfg.alloc_emit) and _live` where `_live = self.r8_emit_live`; `refc_v3.py:1047-1052` then does `rank.masked_fill(~r8_emit_keep, -inf)`, so a non-emitted candidate never wins. `r8_emit_live` is a plain Python attribute (`refc.py:2482`), set at load by `refcv8_train.apply_emit_schedule` (`refcv8_train.py:426-434`, called from `refcv7_loader.py:517-519` at the checkpoint step, which is why the extras are live at step 15000).
- **Route A (recommended, strict-safe by construction): emit-off.** After the model is built and loaded, set `model.core.decoder.r8_emit_live = False` and record it in `overrides` (`{"core.decoder.r8_emit_live": {"trained": True, "eval": False}}`), exactly like the existing `seam_fail_patience` neutralisation. About 4 lines plus a CLI flag in `refcv3_arm.py`. The state_dict is untouched (an attribute, not a Parameter or buffer) and the load has already finished, so strict loading is unaffected. The extras are still decoded (same cost), and `refc.py:2719-2724` documents that the base candidates are decoded in a separate pass with "no candidate-candidate interaction", so the base fan is expected to be unchanged and selection ranges over the 117 bank candidates only. That removes exactly the allocator-extras share of the GAP measured in FAN_VS_SELECTOR.md (about 36 % / 31 % by seed) and nothing else. (Expectation from source, not verified by a run.)
- **Route B: `n_alloc = 0` at build.** No `nn.Parameter` / buffer is sized by `n_alloc` (it is read only at `refc.py:2509` and in a telemetry min at `refc_v3.py:2092`; the dataclass field is `refcv8_conditioning.py:105`), so a model built that way would be EXPECTED to load strict (HYPOTHESIS: not built, not loaded). But the loader replays `train()` and its preflight REFUSES `--w-r8-sat > 0` with no allocated candidates (`refc_v3_train.py:1011-1013`; the checkpoint's argv carries `--w-r8-sat 0.1`), so the replay argv also needs `--w-r8-sat 0`; the config cross-check `param_breakdown_equal` would also have to be re-read. It additionally changes the candidate set and the sampler's random-draw shapes, so it is a different forward, not just a different selection. Not recommended over Route A.
- **Not answerable from the existing dumps:** only the selected plan is stored, so "re-select without extras" cannot be done post hoc.

## 5. Verdict

**Pre-registered question:** does refcv8's candidate-refinement step SHRINK tight turns (turn slope of the refined plan worse than that of the unrefined selected anchor, more in refcv8 than in refcv7)?

**MEASURED (this artifact; S2 panels; sharp turns, last 1.5->2.0 s chord; in-bank selections only, so n = 335 windows / 41 episodes instead of 363 / 42; one checkpoint per arm; CIs = episode-cluster bootstrap with models fixed; EMA-15k vs refcv7-50,400):**

1. **Controls PASS.** The banked slopes and CIs reproduce bit-exactly through the same code (EMA-15k os -0.1409, oracle r8 -0.0826 / r7 -0.0479, r7 os -0.0921; EMA-5k too; max diff 0.0). `plan_full_nav_true[:, :4] == os` exactly (frame and units: ego frame at t0, metres, model slots 0..3 = 0.5/1.0/1.5/2.0 s); the unrefined bank path of a given anchor index is identical across the two arms (max diff 0.0 on 2,654 / 2,638 windows); a mispaired unrefined path reads +0.018 against the paired -0.239, so the pairing is load-bearing.
2. **The refinement does NOT shrink the turn in either model; it ADDS rotation.** Unrefined selected anchors under-rotate heavily: r8 -0.2386 [-0.3155, -0.1647], r7 -0.2714 [-0.3499, -0.1998]. The refined plans read r8 -0.1032, r7 -0.0711. Refinement effect: r8 +0.1354 [+0.0889, +0.1847], r7 +0.2003 [+0.1465, +0.2651]. So "the refinement is the shrinker" is **REFUTED as stated**.
3. **What IS true: refcv8's refinement gives back less of the bank's under-rotation than refcv7's.** DIFF_EFFECT = **-0.0649 [-0.0923, -0.0398], separated**; it replicates on inference seed 0 (-0.0668) and seed 1 (-0.0622), on segments 2..K (-0.0685), on curved windows (-0.0642 / -0.0642) and at EMA-5k (-0.0882 [-0.1072, -0.0711]). Recovery fraction (1 - refined/unrefined slope, DERIVED): r8 0.567 [0.377, 0.783], r7 0.738 [0.542, 0.945], difference -0.171 [-0.248, -0.097].
4. **The unrefined anchors under-rotate in BOTH models, and r8's are not worse** (GAP_unrefined +0.0328 [-0.0010, +0.0630], not separated at the primary view, separated in four secondary views). So the second branch of the decision rule holds: the bank-level under-rotation is shared, and the extra refcv8 loss on in-bank windows (GAP_refined -0.0321 [-0.0615, -0.0089]) is entirely the refinement under-correcting, partly masked by r8's slightly better anchor choice. (The "share" rows exceed 1 for that reason: the two pieces have opposite signs.)
5. **Cleanest isolation (section 1a): where both models pick the SAME anchor index** the unrefined paths are identical (GAP_unrefined = 0, asserted), n = 127 windows / 37 episodes; refcv8's refinement still under-corrects by -0.0286 [-0.0486, -0.0123] (seed 0 -0.0315, seed 1 -0.0298; recovery 0.472 vs 0.577, diff -0.106 [-0.217, -0.041]). So about 45 % of DIFF_EFFECT is the refinement function alone on identical prior paths; the rest sits on windows where the two models pick different anchors (n = 208 / 41 ep: DIFF_EFFECT -0.0860 [-0.1264, -0.0510]; there r7's chosen anchors leave a larger deficit, -0.2715 vs -0.2196, for its refinement to repair).
6. **Trend (two checkpoints, one run each; no exponent quotable):** EMA-5k -> EMA-15k DIFF_EFFECT -0.0882 -> -0.0649 and r8 recovery 0.493 -> 0.567: the refinement is learning to correct more. But GAP_unrefined fell +0.0608 -> +0.0328 (r8's selected anchors became MORE under-rotated, -0.205 -> -0.239), so GAP_refined barely moved (-0.0273 -> -0.0321). The selector gave back what the refinement gained.
7. **At the 5-6 s terminal chord (SECONDARY, n = 243 windows / 32 episodes, a differently scoped subset)** the refinement REDUCES the terminal heading in both models equally (effect -0.2376 vs -0.2510; DIFF_EFFECT +0.0134 [-0.0233, +0.0526], not separated) and both refined plans are unbiased there (-0.059 / -0.076, CIs include 0). The r8 deficit is a <= 2 s phenomenon on these windows.
8. **Fan-level coverage is NOT the problem.** On 262 sharp windows / 32 episodes with a valid 6 s GT chord, a candidate within 0.1806 rad (10.3 deg) of the GT terminal heading exists in refcv7's fan on 100.0 % of windows and in refcv8's on 99.2 % (diff -0.008 [-0.018, +0.000], not separated; 2 of 262 windows), mean best |error| 0.74 deg vs 0.96 deg, and the best candidate's signed turn-aligned error is about 0 in both (-0.04 vs -0.03 deg). The coverage test is saturated (a 117-149 candidate fan contains every plausible terminal heading), so it cannot discriminate; the density of near-GT candidates is higher in refcv8 (0.146 vs 0.090, diff +0.056 [+0.047, +0.067]; bank-only +0.051 [+0.036, +0.066]). EMA-5k is the same (hit 1.000 vs 1.000).

**HYPOTHESIS (not tested here):**
- H-a. The shortfall is a gain error in the refinement/sampler at near horizons (it recovers about 57 % of the bank's 2 s under-rotation instead of about 74 %), rather than a shrink; refcv8's fine-tune objective (alloc-L1, sat, subscore, cons terms on top of the refcv7 init, `--w-r8-listwise 0`) has no term that pushes the refined near-horizon heading toward the GT on tight turns.
- H-b. The shared ~25 % under-rotation of the selected bank anchors at 1.5-2.0 s is the load the refinement has to repair; any refinement gain shortfall therefore appears as tight-turn shrinkage.

**Design lever it points to (largest measured term first):**
1. **A near-horizon (0.5-2.0 s) heading / curvature term on the REFINED candidates, weighted by GT |yaw| (tight turns upweighted)**, applied to the candidate assigned to `a_star` (and the base candidates), so the refinement is rewarded for recovering the turn. Measured size it targets: DIFF_EFFECT -0.065 (-0.029 on identical prior paths), i.e. about the whole in-bank gap. Cheapest discriminating arm: continue refcv8 from the 15k EMA checkpoint with that term, or try it on the v7-tiny rig first; success = recovery fraction >= refcv7's 0.74 with the same-anchor DIFF_EFFECT CI covering 0, read against a seed replicate (H-ESTIM-SEED-1).
2. **Bank coverage of the tight-turn region at 1.5-2.0 s** (anchor file shared and bit-identical, sha256 `51f930dc...`): selected anchors are -0.24 to -0.27 under-rotated in both models, so a bank that starts closer would cut the load on the refinement. This changes the parity key, so it is a PI-level change, not a free tweak.
3. **Do not spend effort on fan reach at the terminal heading** (saturated), **nor on the allocator extras for THIS defect** (the extras windows are excluded from this comparison; emit-off removes the separate ~1/3 share measured in FAN_VS_SELECTOR.md).

**Limits, stated once:** in-bank windows only (r8 sharp in-bank 93-96 %); one checkpoint per arm and one training run (H-ESTIM-SEED-1: nothing here says another training run would reproduce the sign or size); the effect is selection-conditional (each arm refines its own pick; section 1a is the control for that); the refinement "effect" mixes the sampler with whatever scene evidence it conditions on (the prior path is identical in the same-anchor subset, the scene features are not); EMA-10.5k and RAW-5k were not re-run.
