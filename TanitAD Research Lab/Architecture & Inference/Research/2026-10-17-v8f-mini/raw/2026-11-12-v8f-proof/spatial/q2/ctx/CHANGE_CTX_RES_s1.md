# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- RES_s1_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.058450252831474314, "ceiling": -0.22830757322386086, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- RES_s1_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.032354006407243174, "ceiling": -0.22830757322386086, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- RES_s1_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.034078979328403536, "ceiling": -0.22830757322386086, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| RES_s1_R1 | 471 | 6.23 | +0.068 [-0.007, +0.165] | +0.131 [-0.019, +0.279] | +0.063 [-0.046, +0.154] |
| RES_s1_R3h | 471 | 6.23 | +0.073 [+0.008, +0.156] | +0.131 [-0.019, +0.279] | +0.058 [-0.051, +0.147] |
| RES_s1_R3g | 471 | 6.23 | +0.075 [+0.010, +0.159] | +0.131 [-0.019, +0.279] | +0.056 [-0.054, +0.145] |

## Pairs (rollout skill, same windows)

- RES s1 R3h minus RES s1 R1: +0.004 [-0.015, +0.029]
- RES s1 R3g minus RES s1 R1: +0.007 [-0.011, +0.028]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
