# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- G_s1_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.3276521499492995, "ceiling": -0.25915739560053375, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s1_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.26280685527208947, "ceiling": -0.25915739560053375, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s1_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.2639735039136133, "ceiling": -0.25915739560053375, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| G_s1_R1 | 471 | 6.23 | -0.139 [-0.299, -0.006] | +0.179 [+0.033, +0.327] | +0.318 [+0.187, +0.503] |
| G_s1_R3h | 471 | 6.23 | -0.084 [-0.220, +0.043] | +0.179 [+0.033, +0.327] | +0.262 [+0.148, +0.419] |
| G_s1_R3g | 471 | 6.23 | -0.078 [-0.213, +0.045] | +0.179 [+0.033, +0.327] | +0.257 [+0.144, +0.408] |

## Pairs (rollout skill, same windows)

- G s1 R3h minus G s1 R1: +0.055 [+0.026, +0.101] (separated above 0)
- G s1 R3g minus G s1 R1: +0.061 [+0.030, +0.109] (separated above 0)

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
