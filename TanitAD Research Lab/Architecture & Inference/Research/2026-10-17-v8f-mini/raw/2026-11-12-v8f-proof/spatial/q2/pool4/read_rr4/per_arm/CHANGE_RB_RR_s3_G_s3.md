# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- RR_s3: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.10605444459904412, "ceiling": -0.27858215252997076, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- G_s3: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.21430151927334817, "ceiling": -0.31045449039229256, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| RR_s3 | 441 | 6.32 | +0.101 [+0.025, +0.205] | +0.191 [+0.053, +0.343] | +0.090 [-0.001, +0.168] |
| G_s3 | 441 | 6.32 | -0.023 [-0.137, +0.095] | +0.170 [+0.007, +0.331] | +0.192 [+0.062, +0.320] |

## Pairs (rollout skill, same windows)

- RR s3 minus G s3: +0.124 [+0.039, +0.240] (separated above 0)

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
