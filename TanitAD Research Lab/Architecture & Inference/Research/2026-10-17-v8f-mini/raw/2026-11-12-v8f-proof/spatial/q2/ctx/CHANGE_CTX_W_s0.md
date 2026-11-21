# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- W_s0_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.3239894926025355, "ceiling": -0.27584086250479967, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- W_s0_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.2013924844020556, "ceiling": -0.27584086250479967, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- W_s0_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.20925825501186157, "ceiling": -0.27584086250479967, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| W_s0_R1 | 471 | 6.23 | -0.219 [-0.498, -0.007] | -0.033 [-0.195, +0.144] | +0.186 [-0.025, +0.463] |
| W_s0_R3h | 471 | 6.23 | -0.126 [-0.334, +0.048] | -0.033 [-0.195, +0.144] | +0.093 [-0.089, +0.305] |
| W_s0_R3g | 471 | 6.23 | -0.118 [-0.325, +0.056] | -0.033 [-0.195, +0.144] | +0.086 [-0.098, +0.297] |

## Pairs (rollout skill, same windows)

- W s0 R3h minus W s0 R1: +0.093 [+0.039, +0.180] (separated above 0)
- W s0 R3g minus W s0 R1: +0.100 [+0.047, +0.189] (separated above 0)

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
