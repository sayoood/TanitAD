# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- P_s1_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.12373847766158041, "ceiling": -0.2574900680621284, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- P_s1_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.06910006732607399, "ceiling": -0.2574900680621284, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- P_s1_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.07157601137474123, "ceiling": -0.2574900680621284, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| P_s1_R1 | 471 | 6.23 | +0.064 [-0.028, +0.166] | +0.204 [+0.084, +0.328] | +0.140 [+0.028, +0.256] |
| P_s1_R3h | 471 | 6.23 | +0.097 [+0.028, +0.189] | +0.204 [+0.084, +0.328] | +0.108 [+0.002, +0.206] |
| P_s1_R3g | 471 | 6.23 | +0.098 [+0.029, +0.196] | +0.204 [+0.084, +0.328] | +0.106 [-0.008, +0.207] |

## Pairs (rollout skill, same windows)

- P s1 R3h minus P s1 R1: +0.033 [-0.026, +0.106]
- P s1 R3g minus P s1 R1: +0.034 [-0.019, +0.110]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
