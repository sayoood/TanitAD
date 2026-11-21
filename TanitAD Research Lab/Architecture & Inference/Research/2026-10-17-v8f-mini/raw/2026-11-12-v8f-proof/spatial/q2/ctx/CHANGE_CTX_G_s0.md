# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- G_s0_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.23058268174772656, "ceiling": -0.25314325279400585, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s0_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.18088506884897204, "ceiling": -0.25314325279400585, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s0_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.18963199764414695, "ceiling": -0.25314325279400585, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| G_s0_R1 | 471 | 6.23 | -0.046 [-0.247, +0.100] | +0.136 [+0.009, +0.258] | +0.183 [+0.065, +0.337] |
| G_s0_R3h | 471 | 6.23 | -0.038 [-0.234, +0.103] | +0.136 [+0.009, +0.258] | +0.174 [+0.057, +0.321] |
| G_s0_R3g | 471 | 6.23 | -0.034 [-0.226, +0.109] | +0.136 [+0.009, +0.258] | +0.170 [+0.054, +0.317] |

## Pairs (rollout skill, same windows)

- G s0 R3h minus G s0 R1: +0.008 [-0.019, +0.042]
- G s0 R3g minus G s0 R1: +0.013 [-0.011, +0.045]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
