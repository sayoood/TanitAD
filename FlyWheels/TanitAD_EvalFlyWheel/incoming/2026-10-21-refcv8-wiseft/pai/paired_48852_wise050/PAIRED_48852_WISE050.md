# PAIRED_48852_WISE050 -- refcv8 step-blend-0.5 minus step-48,852, PAIRED on the same eval windows

**Generated:** 2026-10-21T16:48:25 (local).  **Evidence class:** MEASURED (ours): per-window arrays rebuilt with the battery's own functions from the banked `perframe_full` dumps/panels of both EMA checkpoints; script `paired_48852_wise050.py` (this directory).  **Tier:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).

**Estimator / question:** paired episode-cluster bootstrap (`taniteval/ci.py`, B 2000, seed 0, 95 % percentile) over the episodes the set contains, models FIXED.  It answers *"would another draw of EPISODES say this?"* ONLY.  It does NOT answer *"would another TRAINING run say this?"* (one training seed: the two checkpoints are two points on ONE run, not two runs) and, beyond the two inference seeds printed per row, not *"would another INFERENCE run say this?"*.  A separated interval is necessary, not sufficient (H-ESTIM-SEED-1).

**Sign:** delta = blend-0.5 minus 48852.  For an ERROR metric (lower is better) a positive separated delta is WORSE at blend-0.5.  Headline set = ceiling-COMPLIANT windows, CAPPED plans (the landed `cap_plan_to_limit`, brake 2.5 m/s^2).  The ALL-WINDOW UNCAPPED form is printed beside.

## 0. Controls

| control | result |
|---|---|
| C1 reproduction (every banked instrument cell, both trees, to 0.001) | **PASS** -- blend-0.5: 464/464 cells, worst abs diff 0.00e+00; 48852: 464/464 cells, worst 0.00e+00; the published literals of the tree that has them (EMA-48,852: S6 ADE vs refcv7 +0.1725, E3 -0.7285, E1 -0.0069; the blend has none) reproduce: True |
| C2 identity (48852 minus itself) | **PASS** -- every headline cell reads delta 0, [0, 0], not separated |
| C3 window join | **PASS** -- key sets identical across trees and across inference seeds; compliant mask, episode labels and the model-free arms (logged v0, GT, echo, refcv7, ha0_ext_pose plans) bit-identical across trees; S2 4159 compliant / 4754 windows, 138 / 139 episodes; S6 3251 / 3668, 138 / 139 |
| C4 inference-seed floor | max over all rows of abs(delta seed0 - delta seed1) = 0.0931; headline ADE below |
| C5 mispaired (48852 rolled 35 windows) | **PASS** -- the interval widens 2.5x (S2) / 2.2x (S6) when the pairing is broken, point estimate unchanged |

## 1. Headline in plain words

Compliant windows, capped plans, blend-0.5 minus 48852, paired:

* S2 ADE (mean over the surface's instants): delta +0.0431 [+0.0291, +0.0571] seed 0, +0.0420 [+0.0295, +0.0545] seed 1 -> **SEPARATED WORSE** (seed 0) / **SEPARATED WORSE** (seed 1)
* S2 speed MAE: delta +0.0355 [+0.0202, +0.0507] seed 0, +0.0371 [+0.0235, +0.0508] seed 1 -> **SEPARATED WORSE** (seed 0) / **SEPARATED WORSE** (seed 1)
* S2 |along-track| at the final instant: delta +0.0832 [+0.0504, +0.1162] seed 0, +0.0850 [+0.0560, +0.1141] seed 1 -> **SEPARATED WORSE** (seed 0) / **SEPARATED WORSE** (seed 1)
* S2 heading MAE: delta +0.0448 [+0.0062, +0.0851] seed 0, +0.0215 [-0.0127, +0.0589] seed 1 -> **SEPARATED WORSE** (seed 0) / **not separated** (seed 1)
* S2 curvature MAE (masked): delta +0.0002 [-0.0000, +0.0004] seed 0, +0.0001 [-0.0001, +0.0004] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S2 yaw-rate MAE (masked): delta +0.0005 [-0.0002, +0.0014] seed 0, +0.0001 [-0.0006, +0.0009] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S6 ADE (mean over the surface's instants): delta -0.0090 [-0.1511, +0.1263] seed 0, -0.0060 [-0.1362, +0.1226] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)

Across ALL 28 (surface, metric) rows of `pair_read_r8.METRICS` (14 per surface x 2 surfaces) at BOTH inference seeds: separated WORSE 9: S2 ADE (mean over the surface's instants), S2 FDE (final instant), S2 speed MAE, S2 along-track MAE, S2 |along-track| at the final instant, S2 target-speed accuracy (+-0.5 m/s, fraction of steps), S6 target-speed accuracy (+-0.5 m/s, fraction of steps), S6 heading MAE, S6 curvature MAE (masked); separated BETTER 2: S2 accel MAE, S6 accel MAE; not separated 16: S2 curvature MAE (masked), S2 yaw-rate MAE (masked), S2 cross-track MAE, S2 |cross-track| at the final instant, S2 trajectory-derived lateral decision correct, S2 trajectory-derived longitudinal decision correct, S6 ADE (mean over the surface's instants), S6 FDE (final instant), S6 speed MAE, S6 along-track MAE, S6 |along-track| at the final instant, S6 yaw-rate MAE (masked), S6 cross-track MAE, S6 |cross-track| at the final instant, S6 trajectory-derived lateral decision correct, S6 trajectory-derived longitudinal decision correct; seeds disagree 1: S2 heading MAE (seed0 SEPARATED WORSE, seed1 not separated).

## 2. Headline metrics (compliant capped, all-window uncapped beside)

### ADE -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.3221 | 0.2790 | +0.0431 [+0.0291, +0.0571] | yes | SEPARATED WORSE | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.3230 | 0.2810 | +0.0420 [+0.0295, +0.0545] | yes | SEPARATED WORSE | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.3467 | 0.2956 | +0.0511 [+0.0371, +0.0638] | yes | SEPARATED WORSE | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.3471 | 0.2985 | +0.0485 [+0.0364, +0.0600] | yes | SEPARATED WORSE | 4754 / 139 |

### LON speed MAE -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.3038 | 0.2683 | +0.0355 [+0.0202, +0.0507] | yes | SEPARATED WORSE | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.3059 | 0.2688 | +0.0371 [+0.0235, +0.0508] | yes | SEPARATED WORSE | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.3325 | 0.2859 | +0.0466 [+0.0315, +0.0611] | yes | SEPARATED WORSE | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.3339 | 0.2876 | +0.0463 [+0.0323, +0.0593] | yes | SEPARATED WORSE | 4754 / 139 |

### LON |along-track| at the final instant -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.6039 | 0.5207 | +0.0832 [+0.0504, +0.1162] | yes | SEPARATED WORSE | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.6065 | 0.5215 | +0.0850 [+0.0560, +0.1141] | yes | SEPARATED WORSE | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.6575 | 0.5536 | +0.1039 [+0.0721, +0.1346] | yes | SEPARATED WORSE | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.6586 | 0.5571 | +0.1015 [+0.0722, +0.1292] | yes | SEPARATED WORSE | 4754 / 139 |

### LAT heading MAE -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 1.5389 | 1.4941 | +0.0448 [+0.0062, +0.0851] | yes | SEPARATED WORSE | 3939 / 138 |
| compliant, capped (HEADLINE) | 1 | 1.5282 | 1.5067 | +0.0215 [-0.0127, +0.0589] | no | not separated | 3938 / 138 |
| all-window, uncapped (beside) | 0 | 1.4495 | 1.4194 | +0.0301 [-0.0081, +0.0713] | no | not separated | 4526 / 139 |
| all-window, uncapped (beside) | 1 | 1.4365 | 1.4292 | +0.0073 [-0.0295, +0.0456] | no | not separated | 4525 / 139 |

### LAT curvature MAE (masked) -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.0040 | 0.0038 | +0.0002 [-0.0000, +0.0004] | no | not separated | 3905 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.0039 | 0.0038 | +0.0001 [-0.0001, +0.0004] | no | not separated | 3907 / 138 |
| all-window, uncapped (beside) | 0 | 0.0038 | 0.0036 | +0.0001 [-0.0001, +0.0004] | no | not separated | 4491 / 139 |
| all-window, uncapped (beside) | 1 | 0.0037 | 0.0036 | +0.0001 [-0.0001, +0.0003] | no | not separated | 4492 / 139 |

### LAT yaw-rate MAE (masked) -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.0181 | 0.0176 | +0.0005 [-0.0002, +0.0014] | no | not separated | 3905 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.0179 | 0.0177 | +0.0001 [-0.0006, +0.0009] | no | not separated | 3907 / 138 |
| all-window, uncapped (beside) | 0 | 0.0182 | 0.0180 | +0.0002 [-0.0005, +0.0010] | no | not separated | 4491 / 139 |
| all-window, uncapped (beside) | 1 | 0.0179 | 0.0181 | -0.0002 [-0.0009, +0.0006] | no | not separated | 4492 / 139 |

### ADE -- S6 (instants 1-6 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 2.6723 | 2.6813 | -0.0090 [-0.1511, +0.1263] | no | not separated | 3251 / 138 |
| compliant, capped (HEADLINE) | 1 | 2.6751 | 2.6810 | -0.0060 [-0.1362, +0.1226] | no | not separated | 3251 / 138 |
| all-window, uncapped (beside) | 0 | 2.8385 | 2.7813 | +0.0571 [-0.0790, +0.1881] | no | not separated | 3668 / 139 |
| all-window, uncapped (beside) | 1 | 2.8409 | 2.7977 | +0.0432 [-0.0835, +0.1607] | no | not separated | 3668 / 139 |

## 3. E1 / E3 forms (refcv8 minus `ha0_ext_pose`, the D-R8-E1-REF reference) and the paired difference of those E values

E value per tree = mean(ADE_refcv8 - ADE_ha0_ext_pose) on the set (negative = better than the prior).  The prior is model-free and identical for both trees, so the paired difference of E values equals the ADE delta above by construction (printed explicitly as the identity check).

| bar | set | seed | E(blend-0.5) [CI] | E(48852) [CI] | paired E(blend-0.5) - E(48852) [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| E1 (S2) | compliant, capped | 0 | +0.0362 [+0.0215, +0.0510] | -0.0069 [-0.0246, +0.0112] | +0.0431 [+0.0291, +0.0571] | yes | SEPARATED WORSE | 4159 / 138 |
| E1 (S2) | compliant, capped | 1 | +0.0372 [+0.0224, +0.0519] | -0.0048 [-0.0210, +0.0124] | +0.0420 [+0.0295, +0.0545] | yes | SEPARATED WORSE | 4159 / 138 |
| E1 (S2) | all-window, uncapped | 0 | +0.0526 [+0.0375, +0.0685] | +0.0015 [-0.0158, +0.0198] | +0.0511 [+0.0371, +0.0638] | yes | SEPARATED WORSE | 4754 / 139 |
| E1 (S2) | all-window, uncapped | 1 | +0.0529 [+0.0371, +0.0694] | +0.0044 [-0.0120, +0.0217] | +0.0485 [+0.0364, +0.0600] | yes | SEPARATED WORSE | 4754 / 139 |
| E3 (S6) | compliant, capped | 0 | -0.7375 [-1.0588, -0.4253] | -0.7285 [-1.0809, -0.3758] | -0.0090 [-0.1511, +0.1263] | no | not separated | 3251 / 138 |
| E3 (S6) | compliant, capped | 1 | -0.7347 [-1.0586, -0.4263] | -0.7287 [-1.0635, -0.3826] | -0.0060 [-0.1362, +0.1226] | no | not separated | 3251 / 138 |
| E3 (S6) | all-window, uncapped | 0 | -0.7380 [-1.0670, -0.4151] | -0.7951 [-1.1437, -0.4480] | +0.0571 [-0.0790, +0.1881] | no | not separated | 3668 / 139 |
| E3 (S6) | all-window, uncapped | 1 | -0.7355 [-1.0713, -0.4115] | -0.7787 [-1.1269, -0.4399] | +0.0432 [-0.0835, +0.1607] | no | not separated | 3668 / 139 |

Pre-ruling echo form (`ha0_ext`) beside: the paired difference is the same ADE delta (the echo is also model-free), so only the per-tree E values differ; see the JSON (`e_forms.*.ha0_ext`).

## 4. Every other metric of `pair_read_r8.METRICS`, compliant capped (headline set), both seeds

| surf | metric | seed | blend-0.5 | 48852 | delta [95 % CI] | verdict | n win / ep |
|---|---|---|---|---|---|---|---|
| S2 | FDE (final instant) (m) | 0 | 0.6875 | 0.6058 | +0.0817 [+0.0497, +0.1133] | SEPARATED WORSE | 4159 / 138 |
| S2 | FDE (final instant) (m) | 1 | 0.6882 | 0.6089 | +0.0794 [+0.0507, +0.1071] | SEPARATED WORSE | 4159 / 138 |
| S2 | along-track MAE (m) | 0 | 0.2793 | 0.2347 | +0.0447 [+0.0300, +0.0592] | SEPARATED WORSE | 4159 / 138 |
| S2 | along-track MAE (m) | 1 | 0.2812 | 0.2359 | +0.0453 [+0.0325, +0.0583] | SEPARATED WORSE | 4159 / 138 |
| S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0 | 0.8113 | 0.8439 | -0.0326 [-0.0457, -0.0200] | SEPARATED WORSE | 4159 / 138 |
| S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 1 | 0.8086 | 0.8427 | -0.0341 [-0.0467, -0.0216] | SEPARATED WORSE | 4159 / 138 |
| S2 | accel MAE (m/s2) | 0 | 0.3652 | 0.3837 | -0.0185 [-0.0366, -0.0022] | SEPARATED BETTER | 4159 / 138 |
| S2 | accel MAE (m/s2) | 1 | 0.3663 | 0.3822 | -0.0158 [-0.0309, -0.0016] | SEPARATED BETTER | 4159 / 138 |
| S2 | cross-track MAE (m) | 0 | 0.0987 | 0.0956 | +0.0031 [-0.0020, +0.0079] | not separated | 4159 / 138 |
| S2 | cross-track MAE (m) | 1 | 0.0982 | 0.0970 | +0.0011 [-0.0037, +0.0057] | not separated | 4159 / 138 |
| S2 | |cross-track| at the final instant (m) | 0 | 0.2003 | 0.1944 | +0.0059 [-0.0053, +0.0164] | not separated | 4159 / 138 |
| S2 | |cross-track| at the final instant (m) | 1 | 0.1989 | 0.1979 | +0.0010 [-0.0098, +0.0113] | not separated | 4159 / 138 |
| S2 | trajectory-derived lateral decision correct (frac) | 0 | 0.9620 | 0.9627 | -0.0007 [-0.0049, +0.0032] | not separated | 4159 / 138 |
| S2 | trajectory-derived lateral decision correct (frac) | 1 | 0.9615 | 0.9630 | -0.0014 [-0.0055, +0.0029] | not separated | 4159 / 138 |
| S2 | trajectory-derived longitudinal decision correct (frac) | 0 | 0.8375 | 0.8286 | +0.0089 [-0.0039, +0.0217] | not separated | 4159 / 138 |
| S2 | trajectory-derived longitudinal decision correct (frac) | 1 | 0.8343 | 0.8269 | +0.0075 [-0.0044, +0.0200] | not separated | 4159 / 138 |
| S6 | FDE (final instant) (m) | 0 | 6.4157 | 6.6100 | -0.1943 [-0.5644, +0.1586] | not separated | 3251 / 138 |
| S6 | FDE (final instant) (m) | 1 | 6.4124 | 6.6113 | -0.1989 [-0.5374, +0.1282] | not separated | 3251 / 138 |
| S6 | speed MAE (m/s) | 0 | 0.9572 | 0.9912 | -0.0340 [-0.0934, +0.0232] | not separated | 3251 / 138 |
| S6 | speed MAE (m/s) | 1 | 0.9615 | 0.9895 | -0.0280 [-0.0844, +0.0258] | not separated | 3251 / 138 |
| S6 | along-track MAE (m) | 0 | 2.3288 | 2.3374 | -0.0086 [-0.1478, +0.1290] | not separated | 3251 / 138 |
| S6 | along-track MAE (m) | 1 | 2.3395 | 2.3334 | +0.0061 [-0.1212, +0.1305] | not separated | 3251 / 138 |
| S6 | |along-track| at the final instant (m) | 0 | 5.5244 | 5.7282 | -0.2038 [-0.5656, +0.1407] | not separated | 3251 / 138 |
| S6 | |along-track| at the final instant (m) | 1 | 5.5411 | 5.7235 | -0.1824 [-0.5138, +0.1346] | not separated | 3251 / 138 |
| S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0 | 0.4775 | 0.5171 | -0.0396 [-0.0676, -0.0124] | SEPARATED WORSE | 3251 / 138 |
| S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 1 | 0.4788 | 0.5182 | -0.0394 [-0.0677, -0.0132] | SEPARATED WORSE | 3251 / 138 |
| S6 | accel MAE (m/s2) | 0 | 0.4205 | 0.4499 | -0.0294 [-0.0505, -0.0098] | SEPARATED BETTER | 3251 / 138 |
| S6 | accel MAE (m/s2) | 1 | 0.4205 | 0.4476 | -0.0270 [-0.0458, -0.0086] | SEPARATED BETTER | 3251 / 138 |
| S6 | heading MAE (deg) | 0 | 3.3769 | 3.1498 | +0.2272 [+0.0545, +0.4155] | SEPARATED WORSE | 3137 / 138 |
| S6 | heading MAE (deg) | 1 | 3.3302 | 3.1681 | +0.1621 [+0.0049, +0.3301] | SEPARATED WORSE | 3135 / 138 |
| S6 | curvature MAE (masked) (1/m) | 0 | 0.0056 | 0.0052 | +0.0004 [+0.0001, +0.0007] | SEPARATED WORSE | 3112 / 138 |
| S6 | curvature MAE (masked) (1/m) | 1 | 0.0054 | 0.0051 | +0.0003 [+0.0000, +0.0006] | SEPARATED WORSE | 3110 / 138 |
| S6 | yaw-rate MAE (masked) (rad/s) | 0 | 0.0285 | 0.0273 | +0.0012 [-0.0002, +0.0026] | not separated | 3112 / 138 |
| S6 | yaw-rate MAE (masked) (rad/s) | 1 | 0.0281 | 0.0273 | +0.0008 [-0.0005, +0.0021] | not separated | 3110 / 138 |
| S6 | cross-track MAE (m) | 0 | 0.8076 | 0.8209 | -0.0133 [-0.0692, +0.0388] | not separated | 3251 / 138 |
| S6 | cross-track MAE (m) | 1 | 0.8046 | 0.8286 | -0.0240 [-0.0782, +0.0281] | not separated | 3251 / 138 |
| S6 | |cross-track| at the final instant (m) | 0 | 2.0508 | 2.0857 | -0.0349 [-0.1845, +0.1137] | not separated | 3251 / 138 |
| S6 | |cross-track| at the final instant (m) | 1 | 2.0389 | 2.1003 | -0.0614 [-0.2060, +0.0745] | not separated | 3251 / 138 |
| S6 | trajectory-derived lateral decision correct (frac) | 0 | 0.8791 | 0.8896 | -0.0105 [-0.0228, +0.0022] | not separated | 3251 / 138 |
| S6 | trajectory-derived lateral decision correct (frac) | 1 | 0.8822 | 0.8902 | -0.0080 [-0.0204, +0.0038] | not separated | 3251 / 138 |
| S6 | trajectory-derived longitudinal decision correct (frac) | 0 | 0.5764 | 0.5930 | -0.0166 [-0.0477, +0.0172] | not separated | 3251 / 138 |
| S6 | trajectory-derived longitudinal decision correct (frac) | 1 | 0.5749 | 0.5894 | -0.0145 [-0.0441, +0.0173] | not separated | 3251 / 138 |

## 5. Inference-seed floor (control C4)

| surface | set | delta seed 0 | delta seed 1 | abs(s0 - s1) of delta | inference floor blend-0.5 (abs mean-ADE s0 - s1) | floor 48852 | smaller abs(delta) / larger floor | clears 2x floor |
|---|---|---|---|---|---|---|---|---|
| S2 | compliant_capped | +0.0431 | +0.0420 | 0.0011 | 0.0010 | 0.0021 | 20.3x | yes |
| S2 | all_uncapped | +0.0511 | +0.0485 | 0.0026 | 0.0003 | 0.0029 | 16.7x | yes |
| S6 | compliant_capped | -0.0090 | -0.0060 | 0.0030 | 0.0028 | 0.0002 | 2.2x | yes |
| S6 | all_uncapped | +0.0571 | +0.0432 | 0.0139 | 0.0024 | 0.0164 | 2.6x | yes |

Within-tree inference replicate (seed 0 minus seed 1, same checkpoint, same windows, ADE, compliant capped):

* wise050 S2: -0.0010 [-0.0042, +0.0020] separated=False
* wise050 S6: -0.0028 [-0.0294, +0.0230] separated=False
* 48852 S2: -0.0021 [-0.0060, +0.0017] separated=False
* 48852 S6: +0.0002 [-0.0384, +0.0380] separated=False

## 6. Scope and caveats

* The two checkpoints are EMA snapshots of ONE training run (seed 0).  A separated checkpoint-to-checkpoint difference is a statement about THIS run's trajectory on THESE 139 eval episodes; it is not a training-seed effect size.  The paired bootstrap is structurally blind to training variance (`H-ESTIM-SEED-1`).
* Inference seeds: the two seeds give the same direction and separation on every headline cell and the delta moves by at most 0.0139 m between them (headline ADE rows), against deltas of ~0.011 (S2) and ~0.16-0.17 (S6).  Per window the two seeds are NOT identical (the within-tree seed0-minus-seed1 intervals are not zero-width), so agreement of the means is averaging, and two seeds are a thin floor: it does not rule out a larger inference variance.
* Reading E1: at 48852 the E1 interval sits below zero (separated better than the prior), at blend-0.5 it straddles zero.  That contrast is NOT itself the test (difference of significances); the test is the paired difference of the two E values, printed in section 3.
* 14 metrics x 2 surfaces x 2 seeds are tabulated; with 95 % intervals a few spurious 'separated' cells are expected by chance among the non-headline rows; the headline rows (S2/S6 ADE, S2 speed MAE, S2 |along-track| final) are the ones that decide.
* No clip ids appear in this document or the JSON; windows are identified only by position / sha12 join fingerprints.

## 7. Where the delta sits (POST-HOC, EXPLORATORY -- not a verdict, not pre-registered)

* (a) S2 seed 0: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 -0.1813 m vs 48852 +0.0875 m; paired delta -0.2687 [-0.3109, -0.2246] (separated=True) n 4159/138
* (a) S2 seed 1: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 -0.1855 m vs 48852 +0.0938 m; paired delta -0.2793 [-0.3191, -0.2394] (separated=True) n 4159/138
* (a) S6 seed 0: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 -1.1499 m vs 48852 +0.9779 m; paired delta -2.1278 [-2.5314, -1.6819] (separated=True) n 3251/138
* (a) S6 seed 1: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 -1.1557 m vs 48852 +1.0112 m; paired delta -2.1670 [-2.5363, -1.7531] (separated=True) n 3251/138
* (b) the landed cap binds on 0 compliant windows over all (surface, seed, tree) combinations, and the capped plan is bit-identical to the uncapped plan on the compliant set in 8 of 8 combinations: on the compliant set 'capped' and 'uncapped' are the same plans, so the headline is not a cap effect.  (The cap binds only on the non-compliant windows, ~5 % of all windows.)

(c) ADE delta by v0 tercile of the compliant set (terciles of the logged v0):

| split (ADE, compliant capped) | surf | seed | blend-0.5 ADE | 48852 ADE | delta [95 % CI] | verdict | n win / ep |
|---|---|---|---|---|---|---|---|
| v0_low (<= 7.03 m/s) | S2 | 0 | 0.3579 | 0.3411 | +0.0168 [+0.0022, +0.0315] | SEPARATED WORSE | 1387 / 74 |
| v0_mid (7.03, 12.93] m/s | S2 | 0 | 0.2785 | 0.2547 | +0.0238 [+0.0085, +0.0391] | SEPARATED WORSE | 1386 / 82 |
| v0_high (> 12.93 m/s) | S2 | 0 | 0.3298 | 0.2411 | +0.0887 [+0.0551, +0.1187] | SEPARATED WORSE | 1386 / 59 |
| v0_low (<= 7.03 m/s) | S2 | 1 | 0.3573 | 0.3449 | +0.0124 [-0.0017, +0.0277] | not separated | 1387 / 74 |
| v0_mid (7.03, 12.93] m/s | S2 | 1 | 0.2796 | 0.2517 | +0.0279 [+0.0122, +0.0435] | SEPARATED WORSE | 1386 / 82 |
| v0_high (> 12.93 m/s) | S2 | 1 | 0.3321 | 0.2464 | +0.0857 [+0.0584, +0.1116] | SEPARATED WORSE | 1386 / 59 |
| v0_low (<= 7.04 m/s) | S6 | 0 | 2.9302 | 3.0549 | -0.1246 [-0.3036, +0.0477] | not separated | 1084 / 71 |
| v0_mid (7.04, 12.94] m/s | S6 | 0 | 2.3691 | 2.3771 | -0.0079 [-0.1833, +0.1555] | not separated | 1083 / 79 |
| v0_high (> 12.94 m/s) | S6 | 0 | 2.7173 | 2.6117 | +0.1056 [-0.2470, +0.4191] | not separated | 1084 / 56 |
| v0_low (<= 7.04 m/s) | S6 | 1 | 2.9272 | 3.0967 | -0.1695 [-0.3405, -0.0058] | SEPARATED BETTER | 1084 / 71 |
| v0_mid (7.04, 12.94] m/s | S6 | 1 | 2.3603 | 2.3481 | +0.0122 [-0.1463, +0.1666] | not separated | 1083 / 79 |
| v0_high (> 12.94 m/s) | S6 | 1 | 2.7374 | 2.5980 | +0.1394 [-0.1586, +0.4106] | not separated | 1084 / 56 |

Direction control (sign of the signed along-track error agrees with the endpoint-range sign wherever the endpoints differ by > 0.5 m; need > 95 %): PASS.  The signed value's absolute equals the instrument's `LON_along_abs_final_m` exactly (asserted on every tree/seed/surface/variant).
