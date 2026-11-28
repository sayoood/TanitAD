# PAIRED_48852_WISE050 -- refcv8 step-blend-0.5 minus step-48,852, PAIRED on the same eval windows

**Generated:** 2026-10-21T14:25:19 (local).  **Evidence class:** MEASURED (ours): per-window arrays rebuilt with the battery's own functions from the banked `perframe_full` dumps/panels of both EMA checkpoints; script `paired_48852_wise050.py` (this directory).  **Tier:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).

**Estimator / question:** paired episode-cluster bootstrap (`taniteval/ci.py`, B 2000, seed 0, 95 % percentile) over the episodes the set contains, models FIXED.  It answers *"would another draw of EPISODES say this?"* ONLY.  It does NOT answer *"would another TRAINING run say this?"* (one training seed: the two checkpoints are two points on ONE run, not two runs) and, beyond the two inference seeds printed per row, not *"would another INFERENCE run say this?"*.  A separated interval is necessary, not sufficient (H-ESTIM-SEED-1).

**Sign:** delta = blend-0.5 minus 48852.  For an ERROR metric (lower is better) a positive separated delta is WORSE at blend-0.5.  Headline set = ceiling-COMPLIANT windows, CAPPED plans (the landed `cap_plan_to_limit`, brake 2.5 m/s^2).  The ALL-WINDOW UNCAPPED form is printed beside.

## 0. Controls

| control | result |
|---|---|
| C1 reproduction (every banked instrument cell, both trees, to 0.001) | **PASS** -- blend-0.5: 464/464 cells, worst abs diff 0.00e+00; 48852: 464/464 cells, worst 0.00e+00; the brief's named values (S6 ADE vs refcv7 +0.1725 / +0.0049, E3 -0.7285 / -0.8961, E1 -0.0069 / -0.0182) reproduce: True |
| C2 identity (48852 minus itself) | **PASS** -- every headline cell reads delta 0, [0, 0], not separated |
| C3 window join | **PASS** -- key sets identical across trees and across inference seeds; compliant mask, episode labels and the model-free arms (logged v0, GT, echo, refcv7, ha0_ext_pose plans) bit-identical across trees; S2 4159 compliant / 4754 windows, 138 / 139 episodes; S6 3251 / 3668, 138 / 139 |
| C4 inference-seed floor | max over all rows of abs(delta seed0 - delta seed1) = 0.0284; headline ADE below |
| C5 mispaired (48852 rolled 35 windows) | **PASS** -- the interval widens 5.2x (S2) / 4.6x (S6) when the pairing is broken, point estimate unchanged |

## 1. Headline in plain words

Compliant windows, capped plans, blend-0.5 minus 48852, paired:

* S2 ADE (mean over the surface's instants): delta -0.0114 [-0.0193, -0.0059] seed 0, -0.0102 [-0.0163, -0.0052] seed 1 -> **SEPARATED BETTER** (seed 0) / **SEPARATED BETTER** (seed 1)
* S2 speed MAE: delta -0.0158 [-0.0246, -0.0093] seed 0, -0.0132 [-0.0206, -0.0071] seed 1 -> **SEPARATED BETTER** (seed 0) / **SEPARATED BETTER** (seed 1)
* S2 |along-track| at the final instant: delta -0.0316 [-0.0498, -0.0184] seed 0, -0.0258 [-0.0407, -0.0131] seed 1 -> **SEPARATED BETTER** (seed 0) / **SEPARATED BETTER** (seed 1)
* S2 heading MAE: delta -0.0029 [-0.0176, +0.0102] seed 0, -0.0076 [-0.0206, +0.0061] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S2 curvature MAE (masked): delta -0.0000 [-0.0001, +0.0000] seed 0, -0.0000 [-0.0001, +0.0001] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S2 yaw-rate MAE (masked): delta -0.0002 [-0.0004, +0.0001] seed 0, -0.0002 [-0.0005, +0.0001] seed 1 -> **not separated** (seed 0) / **not separated** (seed 1)
* S6 ADE (mean over the surface's instants): delta -0.1676 [-0.2408, -0.1108] seed 0, -0.1593 [-0.2282, -0.1035] seed 1 -> **SEPARATED BETTER** (seed 0) / **SEPARATED BETTER** (seed 1)

Across ALL 28 (surface, metric) rows of `pair_read_r8.METRICS` (14 per surface x 2 surfaces) at BOTH inference seeds: separated WORSE 0: none; separated BETTER 19: S2 ADE (mean over the surface's instants), S2 FDE (final instant), S2 speed MAE, S2 along-track MAE, S2 |along-track| at the final instant, S2 target-speed accuracy (+-0.5 m/s, fraction of steps), S2 accel MAE, S2 trajectory-derived longitudinal decision correct, S6 ADE (mean over the surface's instants), S6 FDE (final instant), S6 speed MAE, S6 along-track MAE, S6 |along-track| at the final instant, S6 target-speed accuracy (+-0.5 m/s, fraction of steps), S6 accel MAE, S6 yaw-rate MAE (masked), S6 cross-track MAE, S6 |cross-track| at the final instant, S6 trajectory-derived longitudinal decision correct; not separated 7: S2 heading MAE, S2 curvature MAE (masked), S2 yaw-rate MAE (masked), S2 trajectory-derived lateral decision correct, S6 heading MAE, S6 curvature MAE (masked), S6 trajectory-derived lateral decision correct; seeds disagree 2: S2 cross-track MAE (seed0 not separated, seed1 SEPARATED BETTER), S2 |cross-track| at the final instant (seed0 not separated, seed1 SEPARATED BETTER).

## 2. Headline metrics (compliant capped, all-window uncapped beside)

### ADE -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.2676 | 0.2790 | -0.0114 [-0.0193, -0.0059] | yes | SEPARATED BETTER | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.2708 | 0.2810 | -0.0102 [-0.0163, -0.0052] | yes | SEPARATED BETTER | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.2842 | 0.2956 | -0.0115 [-0.0182, -0.0061] | yes | SEPARATED BETTER | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.2884 | 0.2985 | -0.0101 [-0.0155, -0.0052] | yes | SEPARATED BETTER | 4754 / 139 |

### LON speed MAE -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.2525 | 0.2683 | -0.0158 [-0.0246, -0.0093] | yes | SEPARATED BETTER | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.2556 | 0.2688 | -0.0132 [-0.0206, -0.0071] | yes | SEPARATED BETTER | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.2705 | 0.2859 | -0.0154 [-0.0230, -0.0092] | yes | SEPARATED BETTER | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.2746 | 0.2876 | -0.0129 [-0.0194, -0.0071] | yes | SEPARATED BETTER | 4754 / 139 |

### LON |along-track| at the final instant -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.4891 | 0.5207 | -0.0316 [-0.0498, -0.0184] | yes | SEPARATED BETTER | 4159 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.4957 | 0.5215 | -0.0258 [-0.0407, -0.0131] | yes | SEPARATED BETTER | 4159 / 138 |
| all-window, uncapped (beside) | 0 | 0.5227 | 0.5536 | -0.0309 [-0.0467, -0.0181] | yes | SEPARATED BETTER | 4754 / 139 |
| all-window, uncapped (beside) | 1 | 0.5315 | 0.5571 | -0.0256 [-0.0388, -0.0138] | yes | SEPARATED BETTER | 4754 / 139 |

### LAT heading MAE -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 1.4899 | 1.4929 | -0.0029 [-0.0176, +0.0102] | no | not separated | 3941 / 138 |
| compliant, capped (HEADLINE) | 1 | 1.4972 | 1.5047 | -0.0076 [-0.0206, +0.0061] | no | not separated | 3939 / 138 |
| all-window, uncapped (beside) | 0 | 1.4135 | 1.4184 | -0.0049 [-0.0165, +0.0063] | no | not separated | 4528 / 139 |
| all-window, uncapped (beside) | 1 | 1.4212 | 1.4275 | -0.0064 [-0.0182, +0.0062] | no | not separated | 4526 / 139 |

### LAT curvature MAE (masked) -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.0038 | 0.0038 | -0.0000 [-0.0001, +0.0000] | no | not separated | 3905 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.0038 | 0.0038 | -0.0000 [-0.0001, +0.0001] | no | not separated | 3908 / 138 |
| all-window, uncapped (beside) | 0 | 0.0036 | 0.0037 | -0.0001 [-0.0001, +0.0000] | no | not separated | 4491 / 139 |
| all-window, uncapped (beside) | 1 | 0.0036 | 0.0036 | -0.0000 [-0.0001, +0.0001] | no | not separated | 4493 / 139 |

### LAT yaw-rate MAE (masked) -- S2 (instants 0.5-2.0 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 0.0175 | 0.0176 | -0.0002 [-0.0004, +0.0001] | no | not separated | 3905 / 138 |
| compliant, capped (HEADLINE) | 1 | 0.0175 | 0.0177 | -0.0002 [-0.0005, +0.0001] | no | not separated | 3908 / 138 |
| all-window, uncapped (beside) | 0 | 0.0178 | 0.0181 | -0.0003 [-0.0005, -0.0001] | yes | SEPARATED BETTER | 4491 / 139 |
| all-window, uncapped (beside) | 1 | 0.0178 | 0.0180 | -0.0002 [-0.0005, +0.0001] | no | not separated | 4493 / 139 |

### ADE -- S6 (instants 1-6 s)

| set | seed | blend-0.5 mean | 48852 mean | delta [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|
| compliant, capped (HEADLINE) | 0 | 2.5137 | 2.6813 | -0.1676 [-0.2408, -0.1108] | yes | SEPARATED BETTER | 3251 / 138 |
| compliant, capped (HEADLINE) | 1 | 2.5218 | 2.6810 | -0.1593 [-0.2282, -0.1035] | yes | SEPARATED BETTER | 3251 / 138 |
| all-window, uncapped (beside) | 0 | 2.6227 | 2.7813 | -0.1586 [-0.2233, -0.1056] | yes | SEPARATED BETTER | 3668 / 139 |
| all-window, uncapped (beside) | 1 | 2.6463 | 2.7977 | -0.1515 [-0.2114, -0.0978] | yes | SEPARATED BETTER | 3668 / 139 |

## 3. E1 / E3 forms (refcv8 minus `ha0_ext_pose`, the D-R8-E1-REF reference) and the paired difference of those E values

E value per tree = mean(ADE_refcv8 - ADE_ha0_ext_pose) on the set (negative = better than the prior).  The prior is model-free and identical for both trees, so the paired difference of E values equals the ADE delta above by construction (printed explicitly as the identity check).

| bar | set | seed | E(blend-0.5) [CI] | E(48852) [CI] | paired E(blend-0.5) - E(48852) [95 % CI] | separated | verdict | n windows / episodes |
|---|---|---|---|---|---|---|---|---|
| E1 (S2) | compliant, capped | 0 | -0.0182 [-0.0332, -0.0030] | -0.0069 [-0.0246, +0.0112] | -0.0114 [-0.0193, -0.0059] | yes | SEPARATED BETTER | 4159 / 138 |
| E1 (S2) | compliant, capped | 1 | -0.0150 [-0.0297, +0.0001] | -0.0048 [-0.0210, +0.0124] | -0.0102 [-0.0163, -0.0052] | yes | SEPARATED BETTER | 4159 / 138 |
| E1 (S2) | all-window, uncapped | 0 | -0.0099 [-0.0245, +0.0057] | +0.0015 [-0.0158, +0.0198] | -0.0115 [-0.0182, -0.0061] | yes | SEPARATED BETTER | 4754 / 139 |
| E1 (S2) | all-window, uncapped | 1 | -0.0057 [-0.0205, +0.0095] | +0.0044 [-0.0120, +0.0217] | -0.0101 [-0.0155, -0.0052] | yes | SEPARATED BETTER | 4754 / 139 |
| E3 (S6) | compliant, capped | 0 | -0.8961 [-1.2239, -0.5789] | -0.7285 [-1.0809, -0.3758] | -0.1676 [-0.2408, -0.1108] | yes | SEPARATED BETTER | 3251 / 138 |
| E3 (S6) | compliant, capped | 1 | -0.8880 [-1.2074, -0.5699] | -0.7287 [-1.0635, -0.3826] | -0.1593 [-0.2282, -0.1035] | yes | SEPARATED BETTER | 3251 / 138 |
| E3 (S6) | all-window, uncapped | 0 | -0.9537 [-1.2852, -0.6355] | -0.7951 [-1.1437, -0.4480] | -0.1586 [-0.2233, -0.1056] | yes | SEPARATED BETTER | 3668 / 139 |
| E3 (S6) | all-window, uncapped | 1 | -0.9301 [-1.2550, -0.6090] | -0.7787 [-1.1269, -0.4399] | -0.1515 [-0.2114, -0.0978] | yes | SEPARATED BETTER | 3668 / 139 |

Pre-ruling echo form (`ha0_ext`) beside: the paired difference is the same ADE delta (the echo is also model-free), so only the per-tree E values differ; see the JSON (`e_forms.*.ha0_ext`).

## 4. Every other metric of `pair_read_r8.METRICS`, compliant capped (headline set), both seeds

| surf | metric | seed | blend-0.5 | 48852 | delta [95 % CI] | verdict | n win / ep |
|---|---|---|---|---|---|---|---|
| S2 | FDE (final instant) (m) | 0 | 0.5766 | 0.6058 | -0.0292 [-0.0484, -0.0159] | SEPARATED BETTER | 4159 / 138 |
| S2 | FDE (final instant) (m) | 1 | 0.5837 | 0.6089 | -0.0252 [-0.0401, -0.0126] | SEPARATED BETTER | 4159 / 138 |
| S2 | along-track MAE (m) | 0 | 0.2223 | 0.2347 | -0.0123 [-0.0197, -0.0069] | SEPARATED BETTER | 4159 / 138 |
| S2 | along-track MAE (m) | 1 | 0.2256 | 0.2359 | -0.0103 [-0.0161, -0.0052] | SEPARATED BETTER | 4159 / 138 |
| S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0 | 0.8554 | 0.8439 | +0.0115 [+0.0065, +0.0169] | SEPARATED BETTER | 4159 / 138 |
| S2 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 1 | 0.8532 | 0.8427 | +0.0105 [+0.0056, +0.0157] | SEPARATED BETTER | 4159 / 138 |
| S2 | accel MAE (m/s2) | 0 | 0.3601 | 0.3837 | -0.0236 [-0.0344, -0.0155] | SEPARATED BETTER | 4159 / 138 |
| S2 | accel MAE (m/s2) | 1 | 0.3614 | 0.3822 | -0.0207 [-0.0301, -0.0129] | SEPARATED BETTER | 4159 / 138 |
| S2 | cross-track MAE (m) | 0 | 0.0946 | 0.0956 | -0.0009 [-0.0035, +0.0010] | not separated | 4159 / 138 |
| S2 | cross-track MAE (m) | 1 | 0.0949 | 0.0970 | -0.0021 [-0.0041, -0.0003] | SEPARATED BETTER | 4159 / 138 |
| S2 | |cross-track| at the final instant (m) | 0 | 0.1920 | 0.1944 | -0.0024 [-0.0080, +0.0017] | not separated | 4159 / 138 |
| S2 | |cross-track| at the final instant (m) | 1 | 0.1934 | 0.1979 | -0.0045 [-0.0085, -0.0006] | SEPARATED BETTER | 4159 / 138 |
| S2 | trajectory-derived lateral decision correct (frac) | 0 | 0.9642 | 0.9627 | +0.0014 [-0.0007, +0.0037] | not separated | 4159 / 138 |
| S2 | trajectory-derived lateral decision correct (frac) | 1 | 0.9635 | 0.9630 | +0.0005 [-0.0019, +0.0029] | not separated | 4159 / 138 |
| S2 | trajectory-derived longitudinal decision correct (frac) | 0 | 0.8401 | 0.8286 | +0.0115 [+0.0048, +0.0191] | SEPARATED BETTER | 4159 / 138 |
| S2 | trajectory-derived longitudinal decision correct (frac) | 1 | 0.8396 | 0.8269 | +0.0127 [+0.0060, +0.0199] | SEPARATED BETTER | 4159 / 138 |
| S6 | FDE (final instant) (m) | 0 | 6.1781 | 6.6100 | -0.4319 [-0.6165, -0.2859] | SEPARATED BETTER | 3251 / 138 |
| S6 | FDE (final instant) (m) | 1 | 6.1970 | 6.6113 | -0.4143 [-0.5891, -0.2734] | SEPARATED BETTER | 3251 / 138 |
| S6 | speed MAE (m/s) | 0 | 0.9195 | 0.9912 | -0.0717 [-0.1013, -0.0477] | SEPARATED BETTER | 3251 / 138 |
| S6 | speed MAE (m/s) | 1 | 0.9208 | 0.9895 | -0.0688 [-0.0978, -0.0458] | SEPARATED BETTER | 3251 / 138 |
| S6 | along-track MAE (m) | 0 | 2.1690 | 2.3374 | -0.1684 [-0.2376, -0.1103] | SEPARATED BETTER | 3251 / 138 |
| S6 | along-track MAE (m) | 1 | 2.1771 | 2.3334 | -0.1563 [-0.2254, -0.1013] | SEPARATED BETTER | 3251 / 138 |
| S6 | |along-track| at the final instant (m) | 0 | 5.2969 | 5.7282 | -0.4312 [-0.6112, -0.2789] | SEPARATED BETTER | 3251 / 138 |
| S6 | |along-track| at the final instant (m) | 1 | 5.3208 | 5.7235 | -0.4028 [-0.5755, -0.2650] | SEPARATED BETTER | 3251 / 138 |
| S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0 | 0.5363 | 0.5171 | +0.0192 [+0.0124, +0.0258] | SEPARATED BETTER | 3251 / 138 |
| S6 | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 1 | 0.5354 | 0.5182 | +0.0172 [+0.0101, +0.0245] | SEPARATED BETTER | 3251 / 138 |
| S6 | accel MAE (m/s2) | 0 | 0.4251 | 0.4499 | -0.0248 [-0.0357, -0.0159] | SEPARATED BETTER | 3251 / 138 |
| S6 | accel MAE (m/s2) | 1 | 0.4235 | 0.4476 | -0.0241 [-0.0348, -0.0157] | SEPARATED BETTER | 3251 / 138 |
| S6 | heading MAE (deg) | 0 | 3.1079 | 3.1466 | -0.0387 [-0.0891, +0.0063] | not separated | 3151 / 138 |
| S6 | heading MAE (deg) | 1 | 3.1370 | 3.1735 | -0.0364 [-0.0984, +0.0208] | not separated | 3151 / 138 |
| S6 | curvature MAE (masked) (1/m) | 0 | 0.0051 | 0.0052 | -0.0001 [-0.0002, +0.0001] | not separated | 3123 / 138 |
| S6 | curvature MAE (masked) (1/m) | 1 | 0.0052 | 0.0052 | -0.0000 [-0.0001, +0.0001] | not separated | 3125 / 138 |
| S6 | yaw-rate MAE (masked) (rad/s) | 0 | 0.0270 | 0.0274 | -0.0004 [-0.0008, -0.0000] | SEPARATED BETTER | 3123 / 138 |
| S6 | yaw-rate MAE (masked) (rad/s) | 1 | 0.0270 | 0.0275 | -0.0005 [-0.0009, -0.0001] | SEPARATED BETTER | 3125 / 138 |
| S6 | cross-track MAE (m) | 0 | 0.8040 | 0.8209 | -0.0169 [-0.0330, -0.0031] | SEPARATED BETTER | 3251 / 138 |
| S6 | cross-track MAE (m) | 1 | 0.8066 | 0.8286 | -0.0220 [-0.0422, -0.0028] | SEPARATED BETTER | 3251 / 138 |
| S6 | |cross-track| at the final instant (m) | 0 | 2.0437 | 2.0857 | -0.0420 [-0.0818, -0.0063] | SEPARATED BETTER | 3251 / 138 |
| S6 | |cross-track| at the final instant (m) | 1 | 2.0400 | 2.1003 | -0.0602 [-0.1111, -0.0111] | SEPARATED BETTER | 3251 / 138 |
| S6 | trajectory-derived lateral decision correct (frac) | 0 | 0.8914 | 0.8896 | +0.0018 [-0.0031, +0.0066] | not separated | 3251 / 138 |
| S6 | trajectory-derived lateral decision correct (frac) | 1 | 0.8908 | 0.8902 | +0.0006 [-0.0052, +0.0064] | not separated | 3251 / 138 |
| S6 | trajectory-derived longitudinal decision correct (frac) | 0 | 0.6057 | 0.5930 | +0.0126 [+0.0037, +0.0218] | SEPARATED BETTER | 3251 / 138 |
| S6 | trajectory-derived longitudinal decision correct (frac) | 1 | 0.6044 | 0.5894 | +0.0151 [+0.0060, +0.0250] | SEPARATED BETTER | 3251 / 138 |

## 5. Inference-seed floor (control C4)

| surface | set | delta seed 0 | delta seed 1 | abs(s0 - s1) of delta | inference floor blend-0.5 (abs mean-ADE s0 - s1) | floor 48852 | smaller abs(delta) / larger floor | clears 2x floor |
|---|---|---|---|---|---|---|---|---|
| S2 | compliant_capped | -0.0114 | -0.0102 | 0.0012 | 0.0032 | 0.0021 | 3.2x | yes |
| S2 | all_uncapped | -0.0115 | -0.0101 | 0.0014 | 0.0042 | 0.0029 | 2.4x | yes |
| S6 | compliant_capped | -0.1676 | -0.1593 | 0.0083 | 0.0081 | 0.0002 | 19.7x | yes |
| S6 | all_uncapped | -0.1586 | -0.1515 | 0.0071 | 0.0235 | 0.0164 | 6.4x | yes |

Within-tree inference replicate (seed 0 minus seed 1, same checkpoint, same windows, ADE, compliant capped):

* wise050 S2: -0.0032 [-0.0065, +0.0000] separated=False
* wise050 S6: -0.0081 [-0.0380, +0.0227] separated=False
* 48852 S2: -0.0021 [-0.0060, +0.0017] separated=False
* 48852 S6: +0.0002 [-0.0384, +0.0380] separated=False

## 6. Scope and caveats

* The two checkpoints are EMA snapshots of ONE training run (seed 0).  A separated checkpoint-to-checkpoint difference is a statement about THIS run's trajectory on THESE 139 eval episodes; it is not a training-seed effect size.  The paired bootstrap is structurally blind to training variance (`H-ESTIM-SEED-1`).
* Inference seeds: the two seeds give the same direction and separation on every headline cell and the delta moves by at most 0.0083 m between them (headline ADE rows), against deltas of ~0.011 (S2) and ~0.16-0.17 (S6).  Per window the two seeds are NOT identical (the within-tree seed0-minus-seed1 intervals are not zero-width), so agreement of the means is averaging, and two seeds are a thin floor: it does not rule out a larger inference variance.
* Reading E1: at 48852 the E1 interval sits below zero (separated better than the prior), at blend-0.5 it straddles zero.  That contrast is NOT itself the test (difference of significances); the test is the paired difference of the two E values, printed in section 3.
* 14 metrics x 2 surfaces x 2 seeds are tabulated; with 95 % intervals a few spurious 'separated' cells are expected by chance among the non-headline rows; the headline rows (S2/S6 ADE, S2 speed MAE, S2 |along-track| final) are the ones that decide.
* No clip ids appear in this document or the JSON; windows are identified only by position / sha12 join fingerprints.

## 7. Where the delta sits (POST-HOC, EXPLORATORY -- not a verdict, not pre-registered)

* (a) S2 seed 0: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 +0.0949 m vs 48852 +0.0875 m; paired delta +0.0074 [-0.0113, +0.0301] (separated=False) n 4159/138
* (a) S2 seed 1: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 +0.0971 m vs 48852 +0.0938 m; paired delta +0.0033 [-0.0141, +0.0226] (separated=False) n 4159/138
* (a) S6 seed 0: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 +0.9301 m vs 48852 +0.9779 m; paired delta -0.0478 [-0.2451, +0.1815] (separated=False) n 3251/138
* (a) S6 seed 1: mean SIGNED final along-track error (plan minus human; + = farther/faster) blend-0.5 +0.9532 m vs 48852 +1.0112 m; paired delta -0.0580 [-0.2490, +0.1563] (separated=False) n 3251/138
* (b) the landed cap binds on 0 compliant windows over all (surface, seed, tree) combinations, and the capped plan is bit-identical to the uncapped plan on the compliant set in 8 of 8 combinations: on the compliant set 'capped' and 'uncapped' are the same plans, so the headline is not a cap effect.  (The cap binds only on the non-compliant windows, ~5 % of all windows.)

(c) ADE delta by v0 tercile of the compliant set (terciles of the logged v0):

| split (ADE, compliant capped) | surf | seed | blend-0.5 ADE | 48852 ADE | delta [95 % CI] | verdict | n win / ep |
|---|---|---|---|---|---|---|---|
| v0_low (<= 7.03 m/s) | S2 | 0 | 0.3374 | 0.3411 | -0.0037 [-0.0096, +0.0020] | not separated | 1387 / 74 |
| v0_mid (7.03, 12.93] m/s | S2 | 0 | 0.2485 | 0.2547 | -0.0062 [-0.0104, -0.0022] | SEPARATED BETTER | 1386 / 82 |
| v0_high (> 12.93 m/s) | S2 | 0 | 0.2169 | 0.2411 | -0.0242 [-0.0460, -0.0091] | SEPARATED BETTER | 1386 / 59 |
| v0_low (<= 7.03 m/s) | S2 | 1 | 0.3389 | 0.3449 | -0.0060 [-0.0135, +0.0006] | not separated | 1387 / 74 |
| v0_mid (7.03, 12.93] m/s | S2 | 1 | 0.2508 | 0.2517 | -0.0009 [-0.0053, +0.0035] | not separated | 1386 / 82 |
| v0_high (> 12.93 m/s) | S2 | 1 | 0.2226 | 0.2464 | -0.0238 [-0.0382, -0.0122] | SEPARATED BETTER | 1386 / 59 |
| v0_low (<= 7.04 m/s) | S6 | 0 | 2.9685 | 3.0549 | -0.0864 [-0.1578, -0.0191] | SEPARATED BETTER | 1084 / 71 |
| v0_mid (7.04, 12.94] m/s | S6 | 0 | 2.2597 | 2.3771 | -0.1174 [-0.1909, -0.0550] | SEPARATED BETTER | 1083 / 79 |
| v0_high (> 12.94 m/s) | S6 | 0 | 2.3127 | 2.6117 | -0.2989 [-0.4947, -0.1570] | SEPARATED BETTER | 1084 / 56 |
| v0_low (<= 7.04 m/s) | S6 | 1 | 2.9681 | 3.0967 | -0.1286 [-0.2185, -0.0558] | SEPARATED BETTER | 1084 / 71 |
| v0_mid (7.04, 12.94] m/s | S6 | 1 | 2.2957 | 2.3481 | -0.0524 [-0.1046, -0.0014] | SEPARATED BETTER | 1083 / 79 |
| v0_high (> 12.94 m/s) | S6 | 1 | 2.3014 | 2.5980 | -0.2966 [-0.4707, -0.1643] | SEPARATED BETTER | 1084 / 56 |

Direction control (sign of the signed along-track error agrees with the endpoint-range sign wherever the endpoints differ by > 0.5 m; need > 95 %): PASS.  The signed value's absolute equals the instrument's `LON_along_abs_final_m` exactly (asserted on every tree/seed/surface/variant).
