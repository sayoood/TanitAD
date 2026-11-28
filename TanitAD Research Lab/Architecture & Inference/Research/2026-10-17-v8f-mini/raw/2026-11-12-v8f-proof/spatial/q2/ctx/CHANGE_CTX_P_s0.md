# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- P_s0_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.1082305737711029, "ceiling": -0.2511310995936391, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- P_s0_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.07829028565262641, "ceiling": -0.2511310995936391, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- P_s0_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.08363582032874928, "ceiling": -0.2511310995936391, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| P_s0_R1 | 471 | 6.23 | +0.017 [-0.089, +0.133] | +0.193 [+0.066, +0.337] | +0.176 [+0.082, +0.290] |
| P_s0_R3h | 471 | 6.23 | +0.019 [-0.084, +0.145] | +0.193 [+0.066, +0.337] | +0.175 [+0.081, +0.268] |
| P_s0_R3g | 471 | 6.23 | +0.010 [-0.089, +0.129] | +0.193 [+0.066, +0.337] | +0.184 [+0.094, +0.275] |

## Pairs (rollout skill, same windows)

- P s0 R3h minus P s0 R1: +0.001 [-0.049, +0.071]
- P s0 R3g minus P s0 R1: -0.008 [-0.056, +0.062]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
