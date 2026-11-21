# PAIRED_48852_WPLAN050 -- refcv8 step-PLAN-only-0.5 minus step-48,852, PAIRED on the same eval windows

**Generated:** 2026-10-21T20:09:43 (local).  **Evidence class:** MEASURED (ours): per-window arrays rebuilt with the battery's own functions from the banked `perframe_full` dumps/panels of both EMA checkpoints; script `paired_48852_wplan050.py` (this directory).  **Tier:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).

**Estimator / question:** paired episode-cluster bootstrap (`taniteval/ci.py`, B 2000, seed 0, 95 % percentile) over the episodes the set contains, models FIXED.  It answers *"would another draw of EPISODES say this?"* ONLY.  It does NOT answer *"would another TRAINING run say this?"* (one training seed: the two checkpoints are two points on ONE run, not two runs) and, beyond the two inference seeds printed per row, not *"would another INFERENCE run say this?"*.  A separated interval is necessary, not sufficient (H-ESTIM-SEED-1).

**Sign:** delta = PLAN-only-0.5 minus 48852.  For an ERROR metric (lower is better) a positive separated delta is WORSE at PLAN-only-0.5.  Headline set = ceiling-COMPLIANT windows, CAPPED plans (the landed `cap_plan_to_limit`, brake 2.5 m/s^2).  The ALL-WINDOW UNCAPPED form is printed beside.

## 0. Controls

| control | result |
|---|---|
| C1 reproduction (every banked instrument cell, both trees, to 0.001) | **PASS** -- PLAN-only-0.5: 464/464 cells, worst abs diff 0.00e+00; 48852: 464/464 cells, worst 0.00e+00; the published literals of the tree that has them (EMA-48,852: S6 ADE vs refcv7 +0.1725, E3 -0.7285, E1 -0.0069; the blend has none) reproduce: True |
| C2 identity (48852 minus itself) | **PASS** -- every headline cell reads delta 0, [0, 0], not separated |
| C3 window join | **PASS** -- key sets identical across trees and across inference seeds; compliant mask, episode labels and the model-free arms (logged v0, GT, echo, refcv7, ha0_ext_pose plans) bit-identical across trees; S2 4159 compliant / 4754 windows, 138 / 139 episodes; S6 3251 / 3668, 138 / 139 |
| C4 inference-seed floor | max over all rows of abs(delta seed0 - delta seed1) = 0.0290; headline ADE below |
| C5 mispaired (48852 rolled 35 windows) | **PASS** -- the interval widens 1.5x (S2) / 1.9x (S6) when the pairing is broken, point estimate unchanged |

## 1. Headline in plain words

Compliant windows, capped plans, PLAN-only-0.5 minus 48852, paired:

* S2 ADE (mean over the surface's instants): delta +0.1461 [+0.1217, +0.1699] seed 0, +0.1461 [+0.1215, +0.1695] seed 1 -> **SEPARATED WORSE** (seed 0) / **SEPARATED WORSE** (seed 1)
* S2 speed MAE: delta +0.1374 [+0.1121, +0.1617] seed 0, +0.1397 [+0.1152, +0.1634] seed 1 -> **SEPARATED WORSE** (seed 0) / **SEPARATED WORSE** (seed 1)
* S2 |along-track| at the final instant: delta +0.2723 [+0.2184, +0.3233] seed 0, +0.2768 [+0.2242, +0.3271] seed 1 -> **SEPARATED WORSE** (seed 0) / **SEPARATED WORSE** (seed 1)
* S2 heading MAE: delta +0.0166 [-0.0205, +0.0531] seed 0, +0.0090 [-0.0260, +0.0448] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S2 curvature MAE (masked): delta +0.0000 [-0.0002, +0.0003] seed 0, +0.0001 [-0.0001, +0.0003] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S2 yaw-rate MAE (masked): delta -0.0001 [-0.0008, +0.0007] seed 0, -0.0003 [-0.0009, +0.0004] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S6 ADE (mean over the surface's instants): delta +0.3009 [+0.1280, +0.4632] seed 0, +0.3130 [+0.1423, +0.4794] seed 1 -> **SEPARATED WORSE** (seed 0) / **SEPARATED WORSE** (seed 1)

Across ALL 28 (surface, metric) rows of `pair_read_r8.METRICS` (14 per surface x 2 surfaces) at BOTH inference seeds: separated WORSE 12: S2 ADE (mean over the surface's instants), S2 FDE (final instant), S2 speed MAE, S2 along-track MAE, S2 |along-track| at the final instant, S2 target-speed accuracy (+-0.5 m/s, fraction of steps), S2 accel MAE, S6 ADE (mean over the surface's instants), S6 speed MAE, S6 along-track MAE, S6 target-speed accuracy (+-0.5 m/s, fraction of steps), S6 curvature MAE (masked); separated BETTER 0: none; not separated 15: S2 heading MAE, S2 curvature MAE (masked), S2 yaw-rate MAE (masked), S2 cross-track MAE, S2 |cross-track| at the final instant, S2 trajectory-derived lateral decision correct, S2 trajectory-derived longitudinal decision correct, S6 FDE (final instant), S6 |along-track| at the final instant, S6 accel MAE, S6 heading MAE, S6 yaw-rate MAE (masked), S6 cross-track MAE, S6 |cross-track| at the final instant, S6 trajectory-derived lateral decision correct; seeds disagree 1: S6 trajectory-derived longitudinal decision correct (seed0 SEPARATED WORSE, seed1 not separated).

## 2. Headline metrics (compliant capped, all-window uncapped beside)

### ADE -- S2 (instants 0.5-2.0 s)

| set | seed | PLAN-only-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.4251 | 0.2790 | +0.1461 [+0.1217, +0.1699] | yes | SEPARATED WORSE | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.4271 | 0.2810 | +0.1461 [+0.1215, +0.1695] | yes | SEPARATED WORSE | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.4537 | 0.2956 | +0.1580 [+0.1360, +0.1794] | yes | SEPARATED WORSE | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.4558 | 0.2985 | +0.1573 [+0.1351, +0.1782] | yes | SEPARATED WORSE | 4754 / 139 |

### LON speed MAE -- S2 (instants 0.5-2.0 s)

| set | seed | PLAN-only-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.4057 | 0.2683 | +0.1374 [+0.1121, +0.1617] | yes | SEPARATED WORSE | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.4084 | 0.2688 | +0.1397 [+0.1152, +0.1634] | yes | SEPARATED WORSE | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.4369 | 0.2859 | +0.1510 [+0.1283, +0.1736] | yes | SEPARATED WORSE | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.4396 | 0.2876 | +0.1521 [+0.1294, +0.1740] | yes | SEPARATED WORSE | 4754 / 139 |

### LON |along-track| at the final instant -- S2 (instants 0.5-2.0 s)

| set | seed | PLAN-only-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.7930 | 0.5207 | +0.2723 [+0.2184, +0.3233] | yes | SEPARATED WORSE | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.7983 | 0.5215 | +0.2768 [+0.2242, +0.3271] | yes | SEPARATED WORSE | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.8524 | 0.5536 | +0.2988 [+0.2501, +0.3470] | yes | SEPARATED WORSE | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.8577 | 0.5571 | +0.3006 [+0.2531, +0.3474] | yes | SEPARATED WORSE | 4754 / 139 |

### LAT heading MAE -- S2 (instants 0.5-2.0 s)

| set | seed | PLAN-only-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 1.4218 | 1.4051 | +0.0166 [-0.0205, +0.0531] | no | not separated | 3932 / 138 |
| compliant, capped (HEADLINE) | 1 | 1.4245 | 1.4154 | +0.0090 [-0.0260, +0.0448] | no | not separated | 3933 / 138 |
| all-window, uncapped (beside) | 0 | 1.3551 | 1.3419 | +0.0132 [-0.0233, +0.0508] | no | not separated | 4519 / 139 |
| all-window, uncapped (beside) | 1 | 1.3613 | 1.3497 | +0.0116 [-0.0243, +0.0475] | no | not separated | 4520 / 139 |

### LAT curvature MAE (masked) -- S2 (instants 0.5-2.0 s)

| set | seed | PLAN-only-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.0038 | 0.0037 | +0.0000 [-0.0002, +0.0003] | no | not separated | 3899 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.0038 | 0.0037 | +0.0001 [-0.0001, +0.0003] | no | not separated | 3898 / 138 |
| all-window, uncapped (beside) | 0 | 0.0036 | 0.0036 | +0.0000 [-0.0002, +0.0002] | no | not separated | 4485 / 139 |
| all-window, uncapped (beside) | 1 | 0.0036 | 0.0035 | +0.0001 [-0.0001, +0.0003] | no | not separated | 4483 / 139 |

### LAT yaw-rate MAE (masked) -- S2 (instants 0.5-2.0 s)

| set | seed | PLAN-only-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.0175 | 0.0176 | -0.0001 [-0.0008, +0.0007] | no | not separated | 3899 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.0173 | 0.0176 | -0.0003 [-0.0009, +0.0004] | no | not separated | 3898 / 138 |
| all-window, uncapped (beside) | 0 | 0.0178 | 0.0180 | -0.0002 [-0.0009, +0.0005] | no | not separated | 4485 / 139 |
| all-window, uncapped (beside) | 1 | 0.0177 | 0.0180 | -0.0003 [-0.0009, +0.0004] | no | not separated | 4483 / 139 |

### ADE -- S6 (instants 1-6 s)

| set | seed | PLAN-only-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 2.9822 | 2.6813 | +0.3009 [+0.1280, +0.4632] | yes | SEPARATED WORSE | 3251 / 138 |
| compliant, capped (HEADLINE) | 1 | 2.9941 | 2.6810 | +0.3130 [+0.1423, +0.4794] | yes | SEPARATED WORSE | 3251 / 138 |
| all-window, uncapped (beside) | 0 | 3.1392 | 2.7813 | +0.3579 [+0.1972, +0.5142] | yes | SEPARATED WORSE | 3668 / 139 |
| all-window, uncapped (beside) | 1 | 3.1532 | 2.7977 | +0.3554 [+0.1962, +0.5072] | yes | SEPARATED WORSE | 3668 / 139 |

## 3. E1 / E3 forms (refcv8 minus `ha0_ext_pose`, the D-R8-E1-REF reference) and the paired difference of those E values

E value per tree = mean(ADE_refcv8 - ADE_ha0_ext_pose) on the set (negative = better than the prior).  The prior is model-free and identical for both trees, so the paired difference of E values equals the ADE delta above by construction (printed explicitly as the identity check).

| bar | set | seed | E(PLAN-only-0.5) [CI] | E(48852) [CI] | paired E(PLAN-only-0.5) - E(48852) [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| E1 (S2) | compliant, capped | 0 | +0.1393 [+0.1187, +0.1597] | -0.0069 [-0.0246, +0.0112] | +0.1461 [+0.1217, +0.1699] | yes | SEPARATED WORSE | 4159 / 138 |
| E1 (S2) | compliant, capped | 1 | +0.1413 [+0.1206, +0.1624] | -0.0048 [-0.0210, +0.0124] | +0.1461 [+0.1215, +0.1695] | yes | SEPARATED WORSE | 4159 / 138 |
| E1 (S2) | all-window, uncapped | 0 | +0.1595 [+0.1390, +0.1796] | +0.0015 [-0.0158, +0.0198] | +0.1580 [+0.1360, +0.1794] | yes | SEPARATED WORSE | 4754 / 139 |
| E1 (S2) | all-window, uncapped | 1 | +0.1617 [+0.1408, +0.1817] | +0.0044 [-0.0120, +0.0217] | +0.1573 [+0.1351, +0.1782] | yes | SEPARATED WORSE | 4754 / 139 |
| E3 (S6) | compliant, capped | 0 | -0.4276 [-0.7686, -0.1058] | -0.7285 [-1.0809, -0.3758] | +0.3009 [+0.1280, +0.4632] | yes | SEPARATED WORSE | 3251 / 138 |
| E3 (S6) | compliant, capped | 1 | -0.4157 [-0.7596, -0.0867] | -0.7287 [-1.0635, -0.3826] | +0.3130 [+0.1423, +0.4794] | yes | SEPARATED WORSE | 3251 / 138 |
| E3 (S6) | all-window, uncapped | 0 | -0.4372 [-0.7873, -0.0836] | -0.7951 [-1.1437, -0.4480] | +0.3579 [+0.1972, +0.5142] | yes | SEPARATED WORSE | 3668 / 139 |
| E3 (S6) | all-window, uncapped | 1 | -0.4232 [-0.7816, -0.0730] | -0.7787 [-1.1269, -0.4399] | +0.3554 [+0.1962, +0.5072] | yes | SEPARATED WORSE | 3668 / 139 |

Pre-ruling echo form (`ha0_ext`) beside: the paired difference is the same ADE delta (the echo is also model-free), so only the per-tree E values differ; see the JSON (`e_forms.*.ha0_ext`).

## 4. Every other metric of `pair_read_r8.METRICS`, compliant capped (headline set), both seeds

| surf | metric | seed | PLAN-only-0.5 | 48852 | delta [95 % CI] | verdict | n win / ep |
|---|---|---|---|---|---|---|---|
| S2 | FDE (final instant) (m) | 0 | 0.8629 | 0.6058 | +0.2571 [+0.2052, +0.3074] | SEPARATED WORSE | 4159 / 138 |
| S2 | FDE (final instant) (m) | 1 | 0.8671 | 0.6089 | +0.2582 [+0.2076, +0.3073] | SEPARATED WORSE | 4159 / 138 |
| S2 | along-track MAE (m) | 0 | 0.3912 | 0.2347 | +0.1565 [+0.1307, +0.1810] | SEPARATED WORSE | 4159 / 138 |
| S2 | along-track MAE (m) | 1 | 0.3937 | 0.2359 | +0.1577 [+0.1326, +0.1824] | SEPARATED WORSE | 4159 / 138 |
| S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0 | 0.7056 | 0.8439 | -0.1383 [-0.1634, -0.1127] | SEPARATED WORSE | 4159 / 138 |
| S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 1 | 0.7024 | 0.8427 | -0.1403 [-0.1647, -0.1155] | SEPARATED WORSE | 4159 / 138 |
| S2 | accel MAE (m/s2) | 0 | 0.4185 | 0.3837 | +0.0348 [+0.0150, +0.0538] | SEPARATED WORSE | 4159 / 138 |
| S2 | accel MAE (m/s2) | 1 | 0.4206 | 0.3822 | +0.0384 [+0.0198, +0.0564] | SEPARATED WORSE | 4159 / 138 |
| S2 | cross-track MAE (m) | 0 | 0.0956 | 0.0956 | +0.0001 [-0.0056, +0.0053] | not separated | 4159 / 138 |
| S2 | cross-track MAE (m) | 1 | 0.0957 | 0.0970 | -0.0013 [-0.0064, +0.0033] | not separated | 4159 / 138 |
| S2 | |cross-track| at the final instant (m) | 0 | 0.1935 | 0.1944 | -0.0009 [-0.0133, +0.0103] | not separated | 4159 / 138 |
| S2 | |cross-track| at the final instant (m) | 1 | 0.1932 | 0.1979 | -0.0047 [-0.0156, +0.0055] | not separated | 4159 / 138 |
| S2 | trajectory-derived lateral decision correct (frac) | 0 | 0.9632 | 0.9627 | +0.0005 [-0.0035, +0.0045] | not separated | 4159 / 138 |
| S2 | trajectory-derived lateral decision correct (frac) | 1 | 0.9635 | 0.9630 | +0.0005 [-0.0036, +0.0048] | not separated | 4159 / 138 |
| S2 | trajectory-derived longitudinal decision correct (frac) | 0 | 0.8173 | 0.8286 | -0.0113 [-0.0285, +0.0049] | not separated | 4159 / 138 |
| S2 | trajectory-derived longitudinal decision correct (frac) | 1 | 0.8113 | 0.8269 | -0.0156 [-0.0343, +0.0017] | not separated | 4159 / 138 |
| S6 | FDE (final instant) (m) | 0 | 6.9118 | 6.6100 | +0.3018 [-0.1035, +0.6919] | not separated | 3251 / 138 |
| S6 | FDE (final instant) (m) | 1 | 6.9368 | 6.6113 | +0.3255 [-0.0733, +0.7083] | not separated | 3251 / 138 |
| S6 | speed MAE (m/s) | 0 | 1.0713 | 0.9912 | +0.0801 [+0.0143, +0.1430] | SEPARATED WORSE | 3251 / 138 |
| S6 | speed MAE (m/s) | 1 | 1.0734 | 0.9895 | +0.0838 [+0.0188, +0.1476] | SEPARATED WORSE | 3251 / 138 |
| S6 | along-track MAE (m) | 0 | 2.6636 | 2.3374 | +0.3262 [+0.1536, +0.4940] | SEPARATED WORSE | 3251 / 138 |
| S6 | along-track MAE (m) | 1 | 2.6758 | 2.3334 | +0.3425 [+0.1701, +0.5113] | SEPARATED WORSE | 3251 / 138 |
| S6 | |along-track| at the final instant (m) | 0 | 6.0692 | 5.7282 | +0.3411 [-0.0593, +0.7304] | not separated | 3251 / 138 |
| S6 | |along-track| at the final instant (m) | 1 | 6.0936 | 5.7235 | +0.3701 [-0.0243, +0.7549] | not separated | 3251 / 138 |
| S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0 | 0.4105 | 0.5171 | -0.1066 [-0.1397, -0.0728] | SEPARATED WORSE | 3251 / 138 |
| S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 1 | 0.4054 | 0.5182 | -0.1128 [-0.1446, -0.0801] | SEPARATED WORSE | 3251 / 138 |
| S6 | accel MAE (m/s2) | 0 | 0.4374 | 0.4499 | -0.0125 [-0.0318, +0.0066] | not separated | 3251 / 138 |
| S6 | accel MAE (m/s2) | 1 | 0.4371 | 0.4476 | -0.0105 [-0.0300, +0.0081] | not separated | 3251 / 138 |
| S6 | heading MAE (deg) | 0 | 3.2219 | 3.1039 | +0.1180 [-0.0217, +0.2738] | not separated | 3131 / 138 |
| S6 | heading MAE (deg) | 1 | 3.2220 | 3.1161 | +0.1059 [-0.0262, +0.2557] | not separated | 3131 / 138 |
| S6 | curvature MAE (masked) (1/m) | 0 | 0.0055 | 0.0052 | +0.0003 [+0.0000, +0.0005] | SEPARATED WORSE | 3111 / 138 |
| S6 | curvature MAE (masked) (1/m) | 1 | 0.0055 | 0.0051 | +0.0003 [+0.0001, +0.0006] | SEPARATED WORSE | 3110 / 138 |
| S6 | yaw-rate MAE (masked) (rad/s) | 0 | 0.0280 | 0.0274 | +0.0006 [-0.0006, +0.0019] | not separated | 3111 / 138 |
| S6 | yaw-rate MAE (masked) (rad/s) | 1 | 0.0281 | 0.0274 | +0.0007 [-0.0004, +0.0020] | not separated | 3110 / 138 |
| S6 | cross-track MAE (m) | 0 | 0.7942 | 0.8209 | -0.0267 [-0.0796, +0.0219] | not separated | 3251 / 138 |
| S6 | cross-track MAE (m) | 1 | 0.7965 | 0.8286 | -0.0321 [-0.0811, +0.0141] | not separated | 3251 / 138 |
| S6 | |cross-track| at the final instant (m) | 0 | 2.0171 | 2.0857 | -0.0686 [-0.2057, +0.0615] | not separated | 3251 / 138 |
| S6 | |cross-track| at the final instant (m) | 1 | 2.0261 | 2.1003 | -0.0742 [-0.2042, +0.0540] | not separated | 3251 / 138 |
| S6 | trajectory-derived lateral decision correct (frac) | 0 | 0.8819 | 0.8896 | -0.0077 [-0.0194, +0.0047] | not separated | 3251 / 138 |
| S6 | trajectory-derived lateral decision correct (frac) | 1 | 0.8782 | 0.8902 | -0.0120 [-0.0248, +0.0003] | not separated | 3251 / 138 |
| S6 | trajectory-derived longitudinal decision correct (frac) | 0 | 0.5543 | 0.5930 | -0.0388 [-0.0705, -0.0058] | SEPARATED WORSE | 3251 / 138 |
| S6 | trajectory-derived longitudinal decision correct (frac) | 1 | 0.5564 | 0.5894 | -0.0329 [-0.0659, +0.0022] | not separated | 3251 / 138 |

## 5. Inference-seed floor (control C4)

| surface | set | delta seed 0 | delta seed 1 | abs(s0 - s1) of delta | inference floor PLAN-only-0.5 (abs mean-ADE s0 - s1) | floor 48852 | smaller abs(delta) / larger floor | clears 2x floor |
|---|---|---|---|---|---|---|---|---|
| S2 | compliant_capped | +0.1461 | +0.1461 | 0.0000 | 0.0020 | 0.0021 | 70.6x | yes |
| S2 | all_uncapped | +0.1580 | +0.1573 | 0.0007 | 0.0022 | 0.0029 | 54.0x | yes |
| S6 | compliant_capped | +0.3009 | +0.3130 | 0.0121 | 0.0119 | 0.0002 | 25.3x | yes |
| S6 | all_uncapped | +0.3579 | +0.3554 | 0.0025 | 0.0140 | 0.0164 | 21.7x | yes |

Within-tree inference replicate (seed 0 minus seed 1, same checkpoint, same windows, ADE, compliant capped):

* wplan050 S2: -0.0020 [-0.0050, +0.0010] separated=False
* wplan050 S6: -0.0119 [-0.0425, +0.0187] separated=False
* 48852 S2: -0.0021 [-0.0060, +0.0017] separated=False
* 48852 S6: +0.0002 [-0.0384, +0.0380] separated=False

## 6. Scope and caveats

* The two checkpoints are EMA snapshots of ONE training run (seed 0).  A separated checkpoint-to-checkpoint difference is a statement about THIS run's trajectory on THESE 139 eval episodes; it is not a training-seed effect size.  The paired bootstrap is structurally blind to training variance (`H-ESTIM-SEED-1`).
* Inference seeds: the two seeds give the same direction and separation on every headline cell and the delta moves by at most 0.0121 m between them (headline ADE rows), against deltas of ~0.011 (S2) and ~0.16-0.17 (S6).  Per window the two seeds are NOT identical (the within-tree seed0-minus-seed1 intervals are not zero-width), so agreement of the means is averaging, and two seeds are a thin floor: it does not rule out a larger inference variance.
* Reading E1: at 48852 the E1 interval sits below zero (separated better than the prior), at PLAN-only-0.5 it straddles zero.  That contrast is NOT itself the test (difference of significances); the test is the paired difference of the two E values, printed in section 3.
* 14 metrics x 2 surfaces x 2 seeds are tabulated; with 95 % intervals a few spurious 'separated' cells are expected by chance among the non-headline rows; the headline rows (S2/S6 ADE, S2 speed MAE, S2 |along-track| final) are the ones that decide.
* No clip ids appear in this document or the JSON; windows are identified only by position / sha12 join fingerprints.

## 7. Where the delta sits (POST-HOC, EXPLORATORY -- not a verdict, not pre-registered)

* (a) S2 seed 0: mean SIGNED final along-track error (plan minus human; + = farther/faster) PLAN-only-0.5 -0.5126 m vs 48852 +0.0875 m; paired delta -0.6000 [-0.6464, -0.5507] (separated=True) n 4159/138
* (a) S2 seed 1: mean SIGNED final along-track error (plan minus human; + = farther/faster) PLAN-only-0.5 -0.5185 m vs 48852 +0.0938 m; paired delta -0.6123 [-0.6576, -0.5645] (separated=True) n 4159/138
* (a) S6 seed 0: mean SIGNED final along-track error (plan minus human; + = farther/faster) PLAN-only-0.5 -2.1936 m vs 48852 +0.9779 m; paired delta -3.1715 [-3.5495, -2.7679] (separated=True) n 3251/138
* (a) S6 seed 1: mean SIGNED final along-track error (plan minus human; + = farther/faster) PLAN-only-0.5 -2.2297 m vs 48852 +1.0112 m; paired delta -3.2410 [-3.6111, -2.8398] (separated=True) n 3251/138
* (b) the landed cap binds on 0 compliant windows over all (surface, seed, tree) combinations, and the capped plan is bit-identical to the uncapped plan on the compliant set in 8 of 8 combinations: on the compliant set 'capped' and 'uncapped' are the same plans, so the headline is not a cap effect.  (The cap binds only on the non-compliant windows, ~5 % of all windows.)

(c) ADE delta by v0 tercile of the compliant set (terciles of the logged v0):

| split (ADE, compliant capped) | surf | seed | PLAN-only-0.5 ADE | 48852 ADE | delta [95 % CI] | verdict | n win / ep |
|---|---|---|---|---|---|---|---|
| v0_low (<= 7.03 m/s) | S2 | 0 | 0.3526 | 0.3411 | +0.0115 [-0.0081, +0.0306] | not separated | 1387 / 74 |
| v0_mid (7.03, 12.93] m/s | S2 | 0 | 0.4187 | 0.2547 | +0.1641 [+0.1388, +0.1866] | SEPARATED WORSE | 1386 / 82 |
| v0_high (> 12.93 m/s) | S2 | 0 | 0.5040 | 0.2411 | +0.2629 [+0.2224, +0.3029] | SEPARATED WORSE | 1386 / 59 |
| v0_low (<= 7.03 m/s) | S2 | 1 | 0.3556 | 0.3449 | +0.0107 [-0.0079, +0.0294] | not separated | 1387 / 74 |
| v0_mid (7.03, 12.93] m/s | S2 | 1 | 0.4163 | 0.2517 | +0.1646 [+0.1387, +0.1889] | SEPARATED WORSE | 1386 / 82 |
| v0_high (> 12.93 m/s) | S2 | 1 | 0.5094 | 0.2464 | +0.2630 [+0.2254, +0.3010] | SEPARATED WORSE | 1386 / 59 |
| v0_low (<= 7.04 m/s) | S6 | 0 | 2.8912 | 3.0549 | -0.1636 [-0.3309, +0.0121] | not separated | 1084 / 71 |
| v0_mid (7.04, 12.94] m/s | S6 | 0 | 2.7361 | 2.3771 | +0.3590 [+0.1430, +0.5485] | SEPARATED WORSE | 1083 / 79 |
| v0_high (> 12.94 m/s) | S6 | 0 | 3.3190 | 2.6117 | +0.7074 [+0.3137, +1.0844] | SEPARATED WORSE | 1084 / 56 |
| v0_low (<= 7.04 m/s) | S6 | 1 | 2.9111 | 3.0967 | -0.1856 [-0.3653, +0.0015] | not separated | 1084 / 71 |
| v0_mid (7.04, 12.94] m/s | S6 | 1 | 2.7068 | 2.3481 | +0.3587 [+0.1459, +0.5628] | SEPARATED WORSE | 1083 / 79 |
| v0_high (> 12.94 m/s) | S6 | 1 | 3.3640 | 2.5980 | +0.7660 [+0.3865, +1.1195] | SEPARATED WORSE | 1084 / 56 |

Direction control (sign of the signed along-track error agrees with the endpoint-range sign wherever the endpoints differ by > 0.5 m; need > 95 %): PASS.  The signed value's absolute equals the instrument's `LON_along_abs_final_m` exactly (asserted on every tree/seed/surface/variant).
