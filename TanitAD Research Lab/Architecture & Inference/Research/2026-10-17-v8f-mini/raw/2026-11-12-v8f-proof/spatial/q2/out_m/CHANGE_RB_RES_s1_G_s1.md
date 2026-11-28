# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- RES_s1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.08204743796695113, "ceiling": -0.2659043994965202, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.38506752302671154, "ceiling": -0.23509281150072736, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| RES_s1 | 441 | 6.32 | +0.061 [-0.014, +0.154] | +0.132 [-0.018, +0.278] | +0.071 [-0.039, +0.158] |
| G_s1 | 441 | 6.32 | -0.147 [-0.309, -0.012] | +0.180 [+0.035, +0.327] | +0.327 [+0.200, +0.512] |

## Pairs (rollout skill, same windows)

- RES s1 minus G s1: +0.208 [+0.103, +0.363] (separated above 0)

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
