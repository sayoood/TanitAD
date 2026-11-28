# CHANGE form of the r-b read (note PREREG_V8F_PROOF_ADD16_N1.md)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Target: gap(t + 2.0 s) minus gap(t) on the windows with a lead at both times; skill over persistence (the copy reads 0); episode-cluster bootstrap, B 2000.
This read names nothing; the outcome names of the cards are decided on the level read.

## Controls (first)

- PE_s0_R1: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.09431961365647745, "ceiling": -0.3524419273174395, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- PE_s0_R3h: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.06891979398753123, "ceiling": -0.3524419273174395, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- PE_s0_R3g: k1_copy_is_0 PASS {"value": 0.0}; k2_shuffled_target PASS {"rollout": -0.07793818931910201, "ceiling": -0.3524419273174395, "bar": 0.02}; k3_identity_is_1 PASS {"value": 1.0}
- controls_pass: True

## Change skill

| arm | windows | target rms (m) | rollout | ceiling (true future state) | ceiling minus rollout |
|---|---|---|---|---|---|
| PE_s0_R1 | 471 | 6.23 | -0.060 [-0.180, +0.045] | +0.204 [+0.051, +0.356] | +0.265 [+0.111, +0.448] |
| PE_s0_R3h | 471 | 6.23 | -0.023 [-0.151, +0.076] | +0.204 [+0.051, +0.356] | +0.227 [+0.094, +0.378] |
| PE_s0_R3g | 471 | 6.23 | -0.040 [-0.166, +0.068] | +0.204 [+0.051, +0.356] | +0.245 [+0.098, +0.419] |

## Pairs (rollout skill, same windows)

- PE s0 R3h minus PE s0 R1: +0.038 [-0.016, +0.100]
- PE s0 R3g minus PE s0 R1: +0.020 [-0.030, +0.083]

Limits: one training run per arm per seed; the probes are those of the level read (fitted on levels, not on changes); 139 episodes.
