# KDRAW_E1 -- K-draw ensemble vs the echo on the ceiling-compliant S2 set (step48852ema_unc)

Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint seed sets A/B are the inference-run replicate; NOT another training run (single training seed).
Status: EXPLORATORY read of a banked milestone (EMA step in the tree name); the registered read is the FINAL checkpoint (48,852).
Bar of record: single-draw `os` (BAR-R8-E1); a K-draw arm is a NEW arm with inference cost up to xK.

## OVERALL: **VOID (controls failed: C-K1, C-K2)**  (per arm {'ENS-4': 'FAIL', 'MEDOID-4': 'FAIL'})

**VOID: controls failed -- ['C-K1', 'C-K2']**

## Seeds

| seed | available | dump |
|---|---|---|
| 0 | True | C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/dump_s0 |
| 1 | True | C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/dump_s1 |
| 2 | True | E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s2 |
| 3 | True | E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s3 |
| 4 | True | E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s4 |
| 5 | True | E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s5 |
| 6 | True | E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s6 |
| 7 | True | E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s7 |

## Controls

| control | result |
|---|---|
| C-MASK | PASS=True |
| C-PANEL | PASS=True |
| C-ID | PASS=True |
| C-DIST | PASS=True |
| C-CAP | PASS=True |
| C-DET | PASS=True |
| C-K1 | PASS=False |
| C-K2 | PASS=False |
| C-TRI | PASS=True |

Echo `ha0_ext` mean ADE (compliant) 0.2807 m.

## E1 cells: arm - echo, ADE 0-2 s, compliant windows

| arm | seeds | E1 delta [95% CI] | mean ADE | n win / ep |
|---|---|---|---|---|
| SINGLE | 0 | +0.0110 [-0.0065, +0.0292] | 0.2917 | 4159 / 138 |
| SINGLE | 1 | +0.0097 [-0.0063, +0.0265] | 0.2904 | 4159 / 138 |
| SINGLE | 2 | +0.0111 [-0.0067, +0.0300] | 0.2918 | 4159 / 138 |
| SINGLE | 3 | +0.0101 [-0.0065, +0.0281] | 0.2908 | 4159 / 138 |
| SINGLE | 4 | +0.0132 [-0.0041, +0.0315] | 0.2939 | 4159 / 138 |
| SINGLE | 5 | +0.0114 [-0.0053, +0.0295] | 0.2921 | 4159 / 138 |
| SINGLE | 6 | +0.0120 [-0.0059, +0.0305] | 0.2927 | 4159 / 138 |
| SINGLE | 7 | +0.0095 [-0.0078, +0.0280] | 0.2902 | 4159 / 138 |
| ENS-2 | [0, 1] | +0.0044 [-0.0122, +0.0219] | 0.2851 | 4159 / 138 |
| ENS-4A | [0, 1, 2, 3] | +0.0017 [-0.0151, +0.0195] | 0.2824 | 4159 / 138 |
| MEDOID-4A | [0, 1, 2, 3] | +0.0054 [-0.0113, +0.0233] | 0.2861 | 4159 / 138 |
| ENS-4B | [4, 5, 6, 7] | +0.0024 [-0.0148, +0.0205] | 0.2831 | 4159 / 138 |
| MEDOID-4B | [4, 5, 6, 7] | +0.0046 [-0.0127, +0.0228] | 0.2853 | 4159 / 138 |
| ENS-8 | [0, 1, 2, 3, 4, 5, 6, 7] | +0.0004 [-0.0166, +0.0184] | 0.2811 | 4159 / 138 |
| MEDOID-8 | [0, 1, 2, 3, 4, 5, 6, 7] | +0.0017 [-0.0155, +0.0200] | 0.2824 | 4159 / 138 |

## Verdicts (PREREG section 3)

* **ENS-4: FAIL**  cells A +0.0017 [-0.0151, +0.0195]  B +0.0024 [-0.0148, +0.0205]  detail: {'separated_by_set': [False, False], 'delta_negative_by_set': [False, False], 'within_2x_floor_by_set': [False, False], 'floor_arm_m': 0.0007097423076629639}
* **MEDOID-4: FAIL**  cells A +0.0054 [-0.0113, +0.0233]  B +0.0046 [-0.0127, +0.0228]  detail: {'separated_by_set': [False, False], 'delta_negative_by_set': [False, False], 'within_2x_floor_by_set': [False, False], 'floor_arm_m': 0.0008257031440734863}

## D-R8-E1-REF: the same arms against `ha0_ext_pose` (BESIDE the pre-registered read above; reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19))

Pose-echo mean ADE (compliant) 0.2858 m.  Controls PASS = **False** ({'V1_ha0_ext_rebuilt_vs_panel_max_abs_m': 0.0, 'R0926_pose_minus_echo_all_windows': {'got': '+0.0055 [+0.0036, +0.0076]', 'lit': (0.0055, 0.0036, 0.0076), 'ok': True, 'n_windows': 4754}, 'K1_pose_seed0': {'got': '+0.0059 [-0.0123, +0.0241]', 'lit': (-0.0136, -0.0286, 0.0018), 'ok': False}, 'K1_pose_seed1': {'got': '+0.0046 [-0.0121, +0.0215]', 'lit': (-0.0127, -0.0272, 0.0024), 'ok': False}, 'PASS': False}).

| arm | seeds | E1_pose delta [95% CI] |
|---|---|---|
| SINGLE | 0 | +0.0059 [-0.0123, +0.0241] |
| SINGLE | 1 | +0.0046 [-0.0121, +0.0215] |
| SINGLE | 2 | +0.0060 [-0.0120, +0.0250] |
| SINGLE | 3 | +0.0050 [-0.0122, +0.0231] |
| SINGLE | 4 | +0.0080 [-0.0094, +0.0264] |
| SINGLE | 5 | +0.0063 [-0.0108, +0.0243] |
| SINGLE | 6 | +0.0069 [-0.0115, +0.0258] |
| SINGLE | 7 | +0.0044 [-0.0134, +0.0231] |
| ENS-2 | [0, 1] | -0.0007 [-0.0178, +0.0171] |
| ENS-4A | [0, 1, 2, 3] | -0.0035 [-0.0210, +0.0147] |
| MEDOID-4A | [0, 1, 2, 3] | +0.0003 [-0.0170, +0.0184] |
| ENS-4B | [4, 5, 6, 7] | -0.0028 [-0.0205, +0.0153] |
| MEDOID-4B | [4, 5, 6, 7] | -0.0005 [-0.0180, +0.0177] |
| ENS-8 | [0, 1, 2, 3, 4, 5, 6, 7] | -0.0048 [-0.0225, +0.0134] |
| MEDOID-8 | [0, 1, 2, 3, 4, 5, 6, 7] | -0.0034 [-0.0208, +0.0147] |

* **ENS-4 (pose): PARTIAL**  cells A -0.0035 [-0.0210, +0.0147]  B -0.0028 [-0.0205, +0.0153]

* **MEDOID-4 (pose): FAIL**  cells A +0.0003 [-0.0170, +0.0184]  B -0.0005 [-0.0180, +0.0177]

## Mode disagreement (spread of the K 2-s endpoints > 1.0 m)

* K-set 4A: 5.0 % of compliant windows (n=208); curvature MAE all/disagree ENS 0.00407/0.00701, MEDOID 0.00424/0.00735; ENS-MEDOID curvature on disagree -0.000400 [-0.001800, +0.000700]; blend flag False
* K-set 4B: 5.0 % of compliant windows (n=207); curvature MAE all/disagree ENS 0.00407/0.00751, MEDOID 0.00419/0.00740; ENS-MEDOID curvature on disagree +0.000000 [-0.000300, +0.000400]; blend flag False
* K-set 8: 8.0 % of compliant windows (n=331); curvature MAE all/disagree ENS 0.00402/0.00683, MEDOID 0.00418/0.00694; ENS-MEDOID curvature on disagree -0.000100 [-0.000800, +0.000500]; blend flag False

## NR vs refcv7-50,400 (S2 headline metrics; OK = CI lower <= 0 at BOTH refcv7 seeds)

* ENS-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* MEDOID-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps, LON_along_abs_final_m', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps, LON_along_abs_final_m', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* MEDOID-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps, LON_along_abs_final_m', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-8: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* MEDOID-8: {'ADE': 'PASS', 'LONGITUDINAL': 'FAIL: LON_speed_mae_mps, LON_along_abs_final_m', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)

## Dose-response (nested sets, descriptive)

* K=1 (SINGLE-0): +0.0110 [-0.0065, +0.0292]  mean ADE 0.29167282581329346
* K=2 (ENS-2): +0.0044 [-0.0122, +0.0219]  mean ADE 0.2851370871067047
* K=4 (ENS-4A): +0.0017 [-0.0151, +0.0195]  mean ADE 0.28235045075416565
* K=8 (ENS-8): +0.0004 [-0.0166, +0.0184]  mean ADE 0.2810516059398651
