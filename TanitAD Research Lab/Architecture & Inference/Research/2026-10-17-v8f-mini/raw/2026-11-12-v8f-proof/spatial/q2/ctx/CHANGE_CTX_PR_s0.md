# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- PR_s0_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.14720105127134198, "ceiling": -0.28615558729748436, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- PR_s0_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.05295835111033398, "ceiling": -0.28615558729748436, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- PR_s0_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.056780005196147565, "ceiling": -0.28615558729748436, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| PR_s0_R1 | 471 | 6.23 | +0.024 [-0.083, +0.137] | +0.164 [+0.029, +0.304] | +0.141 [+0.054, +0.225] |
| PR_s0_R3h | 471 | 6.23 | +0.031 [-0.035, +0.118] | +0.164 [+0.029, +0.304] | +0.133 [+0.039, +0.217] |
| PR_s0_R3g | 471 | 6.23 | +0.036 [-0.032, +0.129] | +0.164 [+0.029, +0.304] | +0.128 [+0.034, +0.207] |

## Pairs (rollout skill, same windows)

- PR s0 R3h minus PR s0 R1: +0.008 [-0.034, +0.058]
- PR s0 R3g minus PR s0 R1: +0.013 [-0.028, +0.065]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
