# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- RR_s2: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.09876397160961203, "ceiling": -0.25452329306101706, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s2: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.21796074357031703, "ceiling": -0.2603103070800985, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| RR_s2 | 441 | 6.32 | +0.091 [-0.004, +0.210] | +0.136 [-0.004, +0.276] | +0.045 [-0.073, +0.139] |
| G_s2 | 441 | 6.32 | -0.051 [-0.225, +0.063] | +0.158 [+0.028, +0.295] | +0.208 [+0.112, +0.375] |

## Pairs (rollout skill, same windows)

- RR s2 minus G s2: +0.142 [+0.045, +0.310] (separated above 0)

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
