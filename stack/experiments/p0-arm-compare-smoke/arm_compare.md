# Phase-0 three-arm comparison

- git: `smoke-sample`  |  device: cuda
- common val episodes: **4** (92 windows), episode ids: `[100, 101, 102, 103]`
- arms: flagship (step 11), refa (step 5), refb (step 5)

## Trivial baselines (shared, model-free floor)

| baseline | ade@1s | ade@2s | ade_0_2s |
|---|---|---|---|
| constant_velocity | 0.820 | 2.254 | 2.254 |
| go_straight | 0.645 | 1.973 | 1.973 |
| constant_yaw_rate | 0.515 | 1.622 | 1.622 |

## Comparison table (arm x metric)

| metric | flagship | refa | refb |
|---|---|---|---|
| D1 decode ade_0_2s (parity, frozen probe) | 4.319 | 3.817 | 4.324 |
| D1 gate (camera <1.0m) | FAIL | FAIL | FAIL |
| best held-out ade_0_2s (ladder) | 3.186 | 3.169 | 3.186 |
| oracle-ceiling ade_0_2s (in-dist) | 1.367 | 1.054 | 1.365 |
| held-out / oracle ratio | 2.331 | 3.008 | 2.334 |
| grounded/native traj ade_0_2s | 11.132 | 11.860 | 11.496 |
| native beats CV (overall) | False | False | False |
| native beats CV (straight) | False | False | False |
| D2 direction-acc (imag usable) | 1.000 | 1.000 | N/A |
| D2 gate (>0.7) | PASS | PASS | N/A |
| D3 imagined/oracle ratio | 0.937 | 1.046 | N/A |
| D3 gate (<=1.5x) | BLOCKED | BLOCKED | N/A |

## Per-metric winner

| metric | winner | note |
|---|---|---|
| d1_decode_ade_0_2s | **refa** | frozen-probe parity metric (identical code path); lower is better |
| grounded_traj_ade_0_2s | **flagship** | per-arch mechanism, identical metric |
| d2_direction_acc | **flagship** | imagination usable for selection; higher is better (REF-B N/A) |

## Hierarchy-edge necessary conditions (flagship)

- `flagship_d1_ade_0_2s`: **4.318659424781799**
- `flagship_grounded_ade_0_2s`: **11.131806373596191**
- `flagship_beats_refs_on_d1_decode`: **False**
- `flagship_beats_refs_on_grounded_traj`: **True**
- `flagship_grounded_beats_cv_floor`: **False**
- `flagship_d1_gate`: **FAIL**
- `flagship_d2_gate`: **PASS**
- `flagship_d3_gate`: **BLOCKED**

## Doctrine

D1-D3 + open-loop grounded ADE are NECESSARY, not sufficient (arXiv 2512.24497). This verdict decides the decode/open-loop conditions ONLY; the closed-loop gates D4-D6 (interactive success, blocked-route, simple->complex slope) remain the arbiters of the hierarchy edge and are computed in sim, not here.

- [refb] REF-B is the pre-registered no-world-model reference: no imagination (D2/D3 N/A), no grounded rollout — its native trajectory is the BC waypoint head
