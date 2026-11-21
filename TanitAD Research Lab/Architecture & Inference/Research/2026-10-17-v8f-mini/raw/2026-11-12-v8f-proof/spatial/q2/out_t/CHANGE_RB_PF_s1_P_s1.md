# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- PF_s1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.3892739286396081, "ceiling": -0.31801650392622816, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- P_s1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.14317358018190274, "ceiling": -0.2777207302969009, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| PF_s1 | 441 | 6.32 | -0.167 [-0.342, -0.045] | +0.190 [+0.043, +0.329] | +0.357 [+0.201, +0.551] |
| P_s1 | 441 | 6.32 | +0.056 [-0.037, +0.162] | +0.205 [+0.085, +0.331] | +0.149 [+0.039, +0.266] |

## Pairs (rollout skill, same windows)

- PF s1 minus P s1: -0.222 [-0.406, -0.107] (separated below 0)

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
