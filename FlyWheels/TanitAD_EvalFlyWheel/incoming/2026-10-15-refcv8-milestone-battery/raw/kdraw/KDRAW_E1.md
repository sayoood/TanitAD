# KDRAW_E1 -- K-draw ensemble vs the echo on the ceiling-compliant S2 set (step30000ema_unc)

Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint seed sets A/B are the inference-run replicate; NOT another training run (single training seed).
Status: EXPLORATORY read of a banked milestone (EMA step in the tree name); the registered read is the FINAL checkpoint (48,852).
Bar of record: single-draw `os` (BAR-R8-E1); a K-draw arm is a NEW arm with inference cost up to xK.

## OVERALL: **PASS**  (per arm {'ENS-4': 'PASS', 'MEDOID-4': 'PARTIAL'})

## Seeds

| seed | available | dump |
|---|---|---|
| 0 | True | C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc/dump_s0 |
| 1 | True | C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc/dump_s1 |
| 2 | True | E:/refcv8_eval_kit/kdraw/step30000ema_unc/dump_s2 |
| 3 | True | E:/refcv8_eval_kit/kdraw/step30000ema_unc/dump_s3 |
| 4 | True | E:/refcv8_eval_kit/kdraw/step30000ema_unc/dump_s4 |
| 5 | True | E:/refcv8_eval_kit/kdraw/step30000ema_unc/dump_s5 |
| 6 | True | E:/refcv8_eval_kit/kdraw/step30000ema_unc/dump_s6 |
| 7 | True | E:/refcv8_eval_kit/kdraw/step30000ema_unc/dump_s7 |

## Controls

| control | result |
|---|---|
| C-MASK | PASS=True |
| C-PANEL | PASS=True |
| C-ID | PASS=True |
| C-DIST | PASS=True |
| C-CAP | PASS=True |
| C-DET | PASS=True |
| C-K1 | PASS=True |
| C-K2 | PASS=True |
| C-TRI | PASS=True |

Echo `ha0_ext` mean ADE (compliant) 0.2807 m.

## E1 cells: arm - echo, ADE 0-2 s, compliant windows

| arm | seeds | E1 delta [95% CI] | mean ADE | n win / ep |
|---|---|---|---|---|
| SINGLE | 0 | -0.0085 [-0.0229, +0.0065] | 0.2722 | 4159 / 138 |
| SINGLE | 1 | -0.0076 [-0.0216, +0.0073] | 0.2731 | 4159 / 138 |
| SINGLE | 2 | -0.0101 [-0.0242, +0.0053] | 0.2706 | 4159 / 138 |
| SINGLE | 3 | -0.0090 [-0.0232, +0.0065] | 0.2716 | 4159 / 138 |
| SINGLE | 4 | -0.0078 [-0.0223, +0.0079] | 0.2729 | 4159 / 138 |
| SINGLE | 5 | -0.0071 [-0.0220, +0.0080] | 0.2736 | 4159 / 138 |
| SINGLE | 6 | -0.0074 [-0.0218, +0.0078] | 0.2733 | 4159 / 138 |
| SINGLE | 7 | -0.0080 [-0.0221, +0.0076] | 0.2727 | 4159 / 138 |
| ENS-2 | [0, 1] | -0.0138 [-0.0279, +0.0012] | 0.2669 | 4159 / 138 |
| ENS-4A | [0, 1, 2, 3] | -0.0174 [-0.0314, -0.0021] | 0.2633 | 4159 / 138 |
| MEDOID-4A | [0, 1, 2, 3] | -0.0149 [-0.0290, +0.0003] | 0.2658 | 4159 / 138 |
| ENS-4B | [4, 5, 6, 7] | -0.0158 [-0.0304, -0.0006] | 0.2648 | 4159 / 138 |
| MEDOID-4B | [4, 5, 6, 7] | -0.0130 [-0.0277, +0.0024] | 0.2677 | 4159 / 138 |
| ENS-8 | [0, 1, 2, 3, 4, 5, 6, 7] | -0.0180 [-0.0323, -0.0027] | 0.2627 | 4159 / 138 |
| MEDOID-8 | [0, 1, 2, 3, 4, 5, 6, 7] | -0.0162 [-0.0305, -0.0008] | 0.2645 | 4159 / 138 |

## Verdicts (PREREG section 3)

* **ENS-4: PASS**  cells A -0.0174 [-0.0314, -0.0021]  B -0.0158 [-0.0304, -0.0006]  detail: {'separated_by_set': [True, True], 'delta_negative_by_set': [True, True], 'within_2x_floor_by_set': [False, False], 'floor_arm_m': 0.0015491843223571777}
* **MEDOID-4: PARTIAL**  cells A -0.0149 [-0.0290, +0.0003]  B -0.0130 [-0.0277, +0.0024]  detail: {'n_sets_separated': 0, 'only_miss_is_not_proven_rule': False, 'separated_by_set': [False, False], 'delta_negative_by_set': [True, True], 'within_2x_floor_by_set': [False, False], 'floor_arm_m': 0.001929640769958496}

## D-R8-E1-REF: the same arms against `ha0_ext_pose` (BESIDE the pre-registered read above; reference ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19))

Pose-echo mean ADE (compliant) 0.2858 m.  Controls PASS = **True** ({'V1_ha0_ext_rebuilt_vs_panel_max_abs_m': 0.0, 'R0926_pose_minus_echo_all_windows': {'got': '+0.0055 [+0.0036, +0.0076]', 'lit': (0.0055, 0.0036, 0.0076), 'ok': True, 'n_windows': 4754}, 'K1_pose_seed0': {'got': '-0.0136 [-0.0286, +0.0018]', 'lit': (-0.0136, -0.0286, 0.0018), 'ok': True}, 'K1_pose_seed1': {'got': '-0.0127 [-0.0272, +0.0024]', 'lit': (-0.0127, -0.0272, 0.0024), 'ok': True}, 'PASS': True}).

| arm | seeds | E1_pose delta [95% CI] |
|---|---|---|
| SINGLE | 0 | -0.0136 [-0.0286, +0.0018] |
| SINGLE | 1 | -0.0127 [-0.0272, +0.0024] |
| SINGLE | 2 | -0.0152 [-0.0301, +0.0002] |
| SINGLE | 3 | -0.0142 [-0.0288, +0.0011] |
| SINGLE | 4 | -0.0129 [-0.0278, +0.0026] |
| SINGLE | 5 | -0.0123 [-0.0276, +0.0031] |
| SINGLE | 6 | -0.0125 [-0.0274, +0.0027] |
| SINGLE | 7 | -0.0131 [-0.0279, +0.0026] |
| ENS-2 | [0, 1] | -0.0189 [-0.0335, -0.0035] |
| ENS-4A | [0, 1, 2, 3] | -0.0225 [-0.0373, -0.0071] |
| MEDOID-4A | [0, 1, 2, 3] | -0.0200 [-0.0348, -0.0045] |
| ENS-4B | [4, 5, 6, 7] | -0.0210 [-0.0363, -0.0056] |
| MEDOID-4B | [4, 5, 6, 7] | -0.0181 [-0.0332, -0.0024] |
| ENS-8 | [0, 1, 2, 3, 4, 5, 6, 7] | -0.0232 [-0.0381, -0.0076] |
| MEDOID-8 | [0, 1, 2, 3, 4, 5, 6, 7] | -0.0213 [-0.0364, -0.0058] |

* **ENS-4 (pose): PASS**  cells A -0.0225 [-0.0373, -0.0071]  B -0.0210 [-0.0363, -0.0056]

* **MEDOID-4 (pose): PASS**  cells A -0.0200 [-0.0348, -0.0045]  B -0.0181 [-0.0332, -0.0024]

## Mode disagreement (spread of the K 2-s endpoints > 1.0 m)

* K-set 4A: 1.7 % of compliant windows (n=72); curvature MAE all/disagree ENS 0.00386/0.01215, MEDOID 0.00402/0.01358; ENS-MEDOID curvature on disagree -0.001400 [-0.003200, -0.000200]; blend flag False
* K-set 4B: 1.6 % of compliant windows (n=67); curvature MAE all/disagree ENS 0.00384/0.01186, MEDOID 0.00400/0.01234; ENS-MEDOID curvature on disagree -0.000500 [-0.001700, +0.000600]; blend flag False
* K-set 8: 3.4 % of compliant windows (n=142); curvature MAE all/disagree ENS 0.00381/0.01010, MEDOID 0.00398/0.01043; ENS-MEDOID curvature on disagree -0.000300 [-0.000900, +0.000200]; blend flag False

## NR vs refcv7-50,400 (S2 headline metrics; OK = CI lower <= 0 at BOTH refcv7 seeds)

* ENS-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* MEDOID-4A: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* MEDOID-4B: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* ENS-8: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)
* MEDOID-8: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}  (STRATEGIC: NOT APPLICABLE, n = 0)

## Dose-response (nested sets, descriptive)

* K=1 (SINGLE-0): -0.0085 [-0.0229, +0.0065]  mean ADE 0.2721710801124573
* K=2 (ENS-2): -0.0138 [-0.0279, +0.0012]  mean ADE 0.26692864298820496
* K=4 (ENS-4A): -0.0174 [-0.0314, -0.0021]  mean ADE 0.2632993161678314
* K=8 (ENS-8): -0.0180 [-0.0323, -0.0027]  mean ADE 0.2626606225967407
