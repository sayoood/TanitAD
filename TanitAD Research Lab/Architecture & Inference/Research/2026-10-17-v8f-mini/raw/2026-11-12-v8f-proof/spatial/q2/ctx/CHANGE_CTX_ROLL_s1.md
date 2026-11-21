# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- ROLL_s1_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.36991089945364997, "ceiling": -0.2770786742127034, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- ROLL_s1_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.1568879079697043, "ceiling": -0.2770786742127034, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- ROLL_s1_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.15900196949618461, "ceiling": -0.2770786742127034, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| ROLL_s1_R1 | 471 | 6.23 | -0.052 [-0.240, +0.078] | +0.182 [+0.074, +0.296] | +0.235 [+0.105, +0.429] |
| ROLL_s1_R3h | 471 | 6.23 | -0.010 [-0.118, +0.104] | +0.182 [+0.074, +0.296] | +0.192 [+0.075, +0.318] |
| ROLL_s1_R3g | 471 | 6.23 | -0.005 [-0.113, +0.108] | +0.182 [+0.074, +0.296] | +0.188 [+0.071, +0.314] |

## Pairs (rollout skill, same windows)

- ROLL s1 R3h minus ROLL s1 R1: +0.042 [-0.040, +0.176]
- ROLL s1 R3g minus ROLL s1 R1: +0.047 [-0.033, +0.180]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
