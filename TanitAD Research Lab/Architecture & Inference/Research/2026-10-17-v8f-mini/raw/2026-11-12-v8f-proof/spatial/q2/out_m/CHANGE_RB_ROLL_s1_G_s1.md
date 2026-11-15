# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- ROLL_s1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.3852390626721509, "ceiling": -0.2774205675738519, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.38506752302671154, "ceiling": -0.23509281150072736, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| ROLL_s1 | 441 | 6.32 | -0.053 [-0.239, +0.078] | +0.182 [+0.078, +0.296] | +0.235 [+0.109, +0.425] |
| G_s1 | 441 | 6.32 | -0.147 [-0.309, -0.012] | +0.180 [+0.035, +0.327] | +0.327 [+0.200, +0.512] |

## Pairs (rollout skill, same windows)

- ROLL s1 minus G s1: +0.094 [-0.082, +0.251]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
