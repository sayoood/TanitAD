# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- PF_s0: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.21114370154262518, "ceiling": -0.172815522511661, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- P_s0: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.11860950236761214, "ceiling": -0.25368881055912773, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| PF_s0 | 441 | 6.32 | -0.009 [-0.123, +0.090] | +0.189 [+0.074, +0.324] | +0.198 [+0.100, +0.339] |
| P_s0 | 441 | 6.32 | +0.007 [-0.103, +0.122] | +0.190 [+0.062, +0.330] | +0.183 [+0.090, +0.297] |

## Pairs (rollout skill, same windows)

- PF s0 minus P s0: -0.015 [-0.138, +0.094]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
