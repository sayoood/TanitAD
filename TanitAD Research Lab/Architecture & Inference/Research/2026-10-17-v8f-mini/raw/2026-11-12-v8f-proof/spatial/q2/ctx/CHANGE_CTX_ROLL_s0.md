# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- ROLL_s0_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.2980821252907595, "ceiling": -0.18402542010505796, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- ROLL_s0_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.09390876159543438, "ceiling": -0.18402542010505796, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- ROLL_s0_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.09528456612490199, "ceiling": -0.18402542010505796, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| ROLL_s0_R1 | 471 | 6.23 | -0.105 [-0.362, +0.087] | +0.106 [-0.004, +0.242] | +0.211 [+0.074, +0.415] |
| ROLL_s0_R3h | 471 | 6.23 | -0.035 [-0.185, +0.106] | +0.106 [-0.004, +0.242] | +0.141 [+0.049, +0.257] |
| ROLL_s0_R3g | 471 | 6.23 | -0.021 [-0.167, +0.117] | +0.106 [-0.004, +0.242] | +0.128 [+0.038, +0.237] |

## Pairs (rollout skill, same windows)

- ROLL s0 R3h minus ROLL s0 R1: +0.070 [-0.010, +0.192]
- ROLL s0 R3g minus ROLL s0 R1: +0.084 [+0.001, +0.215] (separated above 0)

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
