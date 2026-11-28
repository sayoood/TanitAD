# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- RR_s1_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.07412550859810896, "ceiling": -0.2626492363267301, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- RR_s1_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.07679238709311931, "ceiling": -0.2626492363267301, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- RR_s1_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.07604798948783231, "ceiling": -0.2626492363267301, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| RR_s1_R1 | 471 | 6.23 | +0.075 [-0.018, +0.173] | +0.187 [+0.031, +0.346] | +0.111 [+0.022, +0.203] |
| RR_s1_R3h | 471 | 6.23 | +0.073 [-0.021, +0.178] | +0.187 [+0.031, +0.346] | +0.114 [+0.018, +0.204] |
| RR_s1_R3g | 471 | 6.23 | +0.073 [-0.020, +0.179] | +0.187 [+0.031, +0.346] | +0.113 [+0.019, +0.204] |

## Pairs (rollout skill, same windows)

- RR s1 R3h minus RR s1 R1: -0.002 [-0.014, +0.014]
- RR s1 R3g minus RR s1 R1: -0.002 [-0.012, +0.012]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
