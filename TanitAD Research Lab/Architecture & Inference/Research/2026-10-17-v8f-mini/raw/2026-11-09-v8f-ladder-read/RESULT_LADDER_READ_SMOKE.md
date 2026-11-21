# RESULT_LADDER_READ (SMOKE: 19 reference draws, not the deciding run)

**Tier T0**: a world-model diagnostic (latent-space prediction of the encoded future of 141 eval clips). It is not driving performance, and no number below is.
Card: `theory/PREREG_V8F_LADDER.md` (Read, Committed outcomes). Implementation choices: `IMPL_CHOICES_LADDER_READ.md`. This file carries labels and tables only; the verdict is the Master Mind's.

## Status and controls

* Arms read: C0_s0, R35_s0. **R35b (the run-to-run floor): PENDING (still training on box 2; add `R35b_s0` to the arm list and rerun this report).**
* Control 1, M2b reproduction through this code path (six MRR values, chance, RankMe on IMAG1B's windows): PASS (`CONTROL_M2B.json`).
* Control 2, Copy-Last against itself, every level and reach of every arm: max |point, lo, hi| = 0.0 (must be exactly 0.0).
* Control 3, BUILT model equals argv (strides three ways, trained k, free-run anchors, context, history K, dilation, causal pad): all arms passed (a mismatch raises and the arm is not read).
* Control 4, deliberate regressions: `test_ladder_read.py` (`TEST_LADDER_READ.log`).

## Families (T0)

LONGITUDINAL and LATERAL appear only through the IA-2 targets (speed, lead gap, yaw rate) and the IA-3 gains below. TACTICAL: n = 0, no tactical target is read in this deliverable. **STRATEGIC: PAUSED** (PI item 25).

## Outcome per arm (the card's labels; both seeds required)

| arm | label | levels beating Copy-Last at the band end | levels not beating | HEALTHY L1 / L2 / L3 (all seeds) | note |
|---|---|---|---|---|---|
| C0 (seed [0] only) | **BAND-MET** | L1, L2, L3 | none | yes / yes / yes |  |
| R35 (seed [0] only) | **BAND-MET** | L1, L2, L3 | none | yes / yes / yes |  |

Per seed:

| run | label | L1 2.0 s skill [95 % CI] | L2 6.0 s | L3 12.0 s | healthy L1/L2/L3 |
|---|---|---|---|---|---|
| C0_s0 | BAND-MET | 0.064 [0.021, 0.104] BEATS | 0.278 [0.216, 0.333] BEATS | 0.359 [0.268, 0.433] BEATS | yes / yes / yes |
| R35_s0 | BAND-MET | 0.178 [0.145, 0.212] BEATS | 0.419 [0.363, 0.467] BEATS | 0.535 [0.469, 0.590] BEATS | yes / yes / yes |

Skill = 1 - sum|pred - oracle|^2 / sum|copy - oracle|^2 in the level's latent (0 = Copy-Last). Interval: episode-cluster bootstrap (B 2,000, seed 0). BEATS = lower bound above 0 AND point above the 97.5th percentile of the same statistic over 199 draws with the answer deranged inside the clip.

## Reach tables (IA-0 skill against Copy-Last; deciding rows marked *)

| run | level | requested s | actual s (k) | on grid | n anchors | clips excluded (<2 anchors) / masked at clip end | skill [95 % CI] | ref 97.5th | above | below |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_s0 | L1 | 1 | 1.0 (5) | True | 13532 | 0 / 705 | 0.070 [0.046, 0.094] | 0.079 | False | False |
| C0_s0 | L1 | 2* | 2.0 (10) | True | 12827 | 0 / 1410 | 0.064 [0.021, 0.104] | 0.044 | True | False |
| C0_s0 | L2 | 3 | 3.2 (8) | False | 6055 | 0 / 1128 | 0.272 [0.217, 0.323] | 0.106 | True | False |
| C0_s0 | L2 | 6* | 6.0 (15) | True | 5068 | 0 / 2115 | 0.278 [0.216, 0.333] | 0.065 | True | False |
| C0_s0 | L3 | 6 | 6.4 (8) | False | 2528 | 0 / 1128 | 0.414 [0.353, 0.466] | 0.143 | True | False |
| C0_s0 | L3 | 9 | 8.8 (11) | False | 2105 | 0 / 1551 | 0.406 [0.331, 0.466] | 0.198 | True | False |
| C0_s0 | L3 | 12* | 12.0 (15) | True | 1541 | 0 / 2115 | 0.359 [0.268, 0.433] | 0.267 | True | False |
| C0_s0 | L3 | 15 | 15.2 (19) | False | 977 | 0 / 2679 | 0.351 [0.243, 0.437] | 0.308 | True | False |
| C0_s0 | L3 | 18 | 18.4 (23) | False | 413 | 0 / 3243 | 0.314 [0.185, 0.415] | 0.295 | True | False |
| R35_s0 | L1 | 1 | 1.0 (5) | True | 13532 | 0 / 705 | 0.092 [0.069, 0.115] | 0.087 | True | False |
| R35_s0 | L1 | 2* | 2.0 (10) | True | 12827 | 0 / 1410 | 0.178 [0.145, 0.212] | 0.100 | True | False |
| R35_s0 | L2 | 3 | 3.0 (5) | True | 4091 | 0 / 705 | 0.316 [0.262, 0.364] | 0.079 | True | False |
| R35_s0 | L2 | 6* | 6.0 (10) | True | 3386 | 0 / 1410 | 0.419 [0.363, 0.467] | 0.113 | True | False |
| R35_s0 | L3 | 6 | 6.0 (2) | True | 705 | 0 / 282 | 0.468 [0.412, 0.515] | 0.086 | True | False |
| R35_s0 | L3 | 9 | 9.0 (3) | True | 564 | 0 / 423 | 0.508 [0.446, 0.557] | 0.220 | True | False |
| R35_s0 | L3 | 12* | 12.0 (4) | True | 423 | 0 / 564 | 0.535 [0.469, 0.590] | 0.357 | True | False |
| R35_s0 | L3 | 15 | 15.0 (5) | True | 282 | 0 / 705 | 0.545 [0.469, 0.607] | 0.460 | True | False |
| R35_s0 | L3 | 18 | 18.0 (6) | True | 0 | 141 / 846 | not readable (fewer than 2 anchors per clip) | | | |

n anchors = (clip, state) pairs scored; 'masked at clip end' = anchors whose target state lies past the end of the clip, counted and not scored (summed over clips).

## IA-0 health (k = 1: MRR of the encoded future among the clip's anchors, predicted latent as query)

| run | level | MRR | chance | excess [95 % CI] | ref 97.5th | separated (two-part) | RankMe pred / oracle |
|---|---|---|---|---|---|---|---|
| C0_s0 | L1 | 0.338 | 0.052 | 0.286 [0.279, 0.293] | -0.002 | True | 37.7 / 43.3 |
| C0_s0 | L2 | 0.336 | 0.090 | 0.246 [0.236, 0.257] | -0.001 | True | 52.6 / 61.9 |
| C0_s0 | L3 | 0.349 | 0.153 | 0.196 [0.182, 0.211] | -0.004 | True | 37.7 / 50.3 |
| R35_s0 | L1 | 0.336 | 0.052 | 0.284 [0.277, 0.292] | -0.002 | True | 34.3 / 44.1 |
| R35_s0 | L2 | 0.343 | 0.124 | 0.219 [0.207, 0.232] | -0.003 | True | 40.2 / 59.4 |
| R35_s0 | L3 | 0.501 | 0.408 | 0.092 [0.069, 0.117] | -0.005 | True | 31.1 / 46.9 |

## Lever effects against the R35b run-to-run floor (band-end skill)

**R35b is pending, so no difference between arms is called a lever effect yet.** The differences are printed for reference only.

| comparison (same seed) | level | skill A | skill B | A - B | exceeds floor |
|---|---|---|---|---|---|
| R35 - C0 s0 | L1 | 0.178 | 0.064 | 0.114 | R35b pending |
| R35 - C0 s0 | L2 | 0.419 | 0.278 | 0.141 | R35b pending |
| R35 - C0 s0 | L3 | 0.535 | 0.359 | 0.176 | R35b pending |

## Beside: planning-form context (context 1) band-end skill (decides nothing)

| run | L1 2.0 s | L2 6.0 s | L3 12.0 s |
|---|---|---|---|
| C0_s0 | -0.043 [-0.098, 0.008] | 0.235 [0.165, 0.295] above | 0.342 [0.247, 0.418] above |
| R35_s0 | -0.059 [-0.113, -0.008] below | 0.288 [0.218, 0.347] above | 0.512 [0.439, 0.572] above |

## Beside (no bar; ridge / logistic probes, bootstrap intervals only, no permutation reference)

### IA-1 capture (score = 1 - NMSE, 0 = the marginal mean) on the three IA-2 targets

| run | level | speed | lead gap | yaw rate |
|---|---|---|---|---|
| C0_s0 | L1 | 0.68 [0.59, 0.75] n=9596 | 0.11 [-0.21, 0.36] n=2489 | -0.13 [-0.33, 0.02] n=9596 |
| C0_s0 | L2 | 0.54 [0.41, 0.64] n=4798 | 0.10 [-0.24, 0.35] n=1232 | -0.08 [-0.35, 0.12] n=4798 |
| C0_s0 | L3 | 0.55 [0.38, 0.67] n=2399 | 0.18 [-0.03, 0.37] n=604 | 0.01 [-0.30, 0.22] n=2399 |
| R35_s0 | L1 | 0.68 [0.59, 0.74] n=9596 | 0.13 [-0.18, 0.37] n=2489 | -0.09 [-0.29, 0.06] n=9596 |
| R35_s0 | L2 | 0.67 [0.54, 0.76] n=3245 | 0.05 [-0.35, 0.38] n=819 | -0.12 [-0.40, 0.08] n=3245 |
| R35_s0 | L3 | 0.63 [0.52, 0.72] n=705 | 0.09 [-0.41, 0.42] n=152 | 0.07 [-0.24, 0.29] n=705 |

### IA-2 probe transfer at the band end: predicted minus Copy-Last score (positive = predicted decodes better)

| run | level | target | pred - copy [95 % CI] | oracle score | n |
|---|---|---|---|---|---|
| C0_s0 | L1 (2 s) | speed | 0.026 [-0.011, 0.062] | 0.683 | 8186 |
| C0_s0 | L1 (2 s) | lead_gap | 0.013 [-0.138, 0.195] | 0.123 | 2150 |
| C0_s0 | L1 (2 s) | yaw_rate | 0.115 [0.026, 0.222] | -0.101 | 8186 |
| C0_s0 | L2 (6 s) | speed | 0.119 [0.034, 0.221] | 0.549 | 2683 |
| C0_s0 | L2 (6 s) | lead_gap | 0.250 [-0.037, 0.586] | 0.015 | 684 |
| C0_s0 | L2 (6 s) | yaw_rate | 0.475 [0.306, 0.686] | -0.012 | 2683 |
| C0_s0 | L3 (12 s) | speed | 0.250 [0.147, 0.384] | 0.625 | 284 |
| C0_s0 | L3 (12 s) | lead_gap | 0.649 [0.011, 1.521] | -0.234 | 67 |
| C0_s0 | L3 (12 s) | yaw_rate | 0.283 [0.050, 0.497] | 0.253 | 284 |
| R35_s0 | L1 (2 s) | speed | 0.045 [0.002, 0.094] | 0.672 | 8186 |
| R35_s0 | L1 (2 s) | lead_gap | -0.020 [-0.142, 0.105] | 0.140 | 2150 |
| R35_s0 | L1 (2 s) | yaw_rate | 0.185 [0.090, 0.303] | -0.061 | 8186 |
| R35_s0 | L2 (6 s) | speed | 0.171 [0.104, 0.253] | 0.653 | 1835 |
| R35_s0 | L2 (6 s) | lead_gap | 0.206 [-0.042, 0.476] | 0.012 | 468 |
| R35_s0 | L2 (6 s) | yaw_rate | 0.566 [0.376, 0.828] | -0.129 | 1835 |
| R35_s0 | L3 (12 s) | speed | 0.241 [0.134, 0.389] | 0.650 | 141 |
| R35_s0 | L3 (12 s) | lead_gap | 1.606 [0.623, 3.084] | -0.486 | 33 |
| R35_s0 | L3 (12 s) | yaw_rate | 0.563 [0.276, 0.874] | 0.311 | 141 |

### IA-3 action audit at the band end (gain = decoded effect / physical effect, 1.0 = right size; cosine = imagined vs physical 2 x 2 Jacobian)

| run | level | k (s) | gain lon [CI] | gain lat [CI] | cosine [CI] | n |
|---|---|---|---|---|---|---|
| C0_s0 | L1 | 10 (2.0 s) | 0.300 [0.267, 0.333] | -0.074 [-0.086, -0.062] | 0.482 [0.325, 0.753] | 12827 |
| C0_s0 | L2 | 15 (6.0 s) | 0.330 [0.301, 0.356] | -0.030 [-0.052, -0.008] | 0.731 [0.580, 0.886] | 5068 |
| C0_s0 | L3 | 15 (12.0 s) | 0.296 [0.269, 0.323] | 0.000 [-0.021, 0.022] | 0.739 [0.606, 0.885] | 1541 |
| R35_s0 | L1 | 10 (2.0 s) | 0.384 [0.348, 0.419] | -0.038 [-0.049, -0.028] | 0.825 [0.606, 0.873] | 12827 |
| R35_s0 | L2 | 10 (6.0 s) | 0.427 [0.403, 0.450] | 0.142 [0.123, 0.163] | 0.902 [0.803, 0.973] | 3386 |
| R35_s0 | L3 | 4 (12.0 s) | 0.351 [0.333, 0.370] | 0.317 [0.294, 0.342] | 0.986 [0.946, 1.000] | 423 |

### TSI (mean of 1 - score over the fast targets minus over the slow ones, from IA-1) and cross-level slope

| run | TSI L1 | TSI L2 | TSI L3 | slope L1 vs L2 | slope L2 vs L3 |
|---|---|---|---|---|---|
| C0_s0 | 0.998 | 0.890 | 0.690 | 0.298 [0.194, 0.393] (6.0 s) | 0.570 [0.455, 0.681] (12.0 s) |
| R35_s0 | 1.063 | 0.902 | 0.585 | 0.546 [0.451, 0.646] (6.0 s) | 0.626 [0.518, 0.737] (6.0 s) |

TSI uses ten IA-1 targets; labels exist at the 9,596 bank frames only (lead-gap and lead-closing rows have fewer, n per row in the json). Slope: decoded speed change of the coarser level on the finer level over the same real horizon.

## Provenance

| run | argv sha256 (first 12) | adapter sha256 (first 12) | ckpt step | free GB at start | wall s |
|---|---|---|---|---|---|
| C0_s0 | aed786fae9a0 | 9c54ed2041e2 | 19164 | 16.08 | 90.1 |
| R35_s0 | 4ded69411500 | 9c54ed2041e2 | 19164 | 15.72 | 59.6 |
