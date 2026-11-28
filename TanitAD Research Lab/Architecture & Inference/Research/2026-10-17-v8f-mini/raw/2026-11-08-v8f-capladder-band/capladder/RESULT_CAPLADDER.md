# RESULT_CAPLADDER: where does the lead gap get lost between DINOv3 patch tokens and the world-model levels of M2b?

Card `../PREREG_V8F_CAPLADDER.md` (frozen prefix 7,830 bytes, sha256 `e17a4d90...`), `IMPL_CHOICES_CAPLADDER.md` (sha256 recorded in `IMPL_CHOICES_CAPLADDER.sha256`). Tier **T0**. STRATEGIC: PAUSED (PI item 25). Eval-141 frames; deciding target `lead_gap` on 2,489 frames of 56 episodes; 5 outer episode folds; dual ridge; bootstrap B 2,000 over 141 episodes (weights not checked (ModuleNotFoundError)). Written 2026-11-08 11:04:34.

## Banner: VALID

| control | result | detail |
|---|---|---|
| 1 calibration (20 deranged draws, at most 5 % separated per row kind) | pass | READABLE rows 0/480 (0.0 %), LOSS rows 4/400 (1.0 %) |
| 2 dimension (Gaussian n x d, seed 0, must not be READABLE) | pass | d 655,360: NMSE 1.001, bootstrap upper 1.003, reference 2.5th 1.007, not readable; d 163,840: NMSE 1.001, bootstrap upper 1.003, reference 2.5th 1.006, not readable; d 40,960: NMSE 1.001, bootstrap upper 1.003, reference 2.5th 1.007, not readable; d 3,072: NMSE 1.000, bootstrap upper 1.001, reference 2.5th 1.007, not readable; d 1,024: NMSE 1.001, bootstrap upper 1.002, reference 2.5th 1.007, not readable; d 384: NMSE 1.003, bootstrap upper 1.005, reference 2.5th 1.006, not readable; d 128: NMSE 1.004, bootstrap upper 1.007, reference 2.5th 1.008, not readable |
| 3 planted (synthetic a, b, c and the regression arms) | pass | `TEST_CAPLADDER.log`: 9 passed in 8.69s |
| 4 R0 identity (decides only R0) | ADMISSIBLE | 8x20: 100.00 % of 9,596 bank frames within 0.08 (median 0.0265, 99th 0.0268, max 0.0270); 4x10: 100.00 % (median 0.0265, max 0.0278); after the fp8 cast the pooled field equals the cache bit for bit on 100.0 % (8x20) and 100.0 % (4x10) of frames |

## Readability of `lead_gap` at every rung (NMSE; 1 = the fit-fold mean; READABLE = bootstrap upper bound < 1 AND point below the 2.5th percentile of 199 refit-under-permutation draws)

| rung | what | d | NMSE | 95 % CI | upper < 1 | reference 2.5th | label |
|---|---|---|---|---|---|---|---|
| R0 | patch tokens 16 x 40 | 655,360 | 0.515 | [0.430, 0.615] | yes | 1.001 | READABLE |
| R1 | pool 8 x 20 (cached) | 163,840 | 0.503 | [0.410, 0.614] | yes | 1.001 | READABLE |
| R2 | pool 4 x 10 (the model's input) | 40,960 | 0.501 | [0.401, 0.625] | yes | 0.995 | READABLE |
| R2m | mean token (beside) | 1,024 | 0.619 | [0.476, 0.821] | yes | 0.979 | READABLE |

| rung | what | d | seed | NMSE | 95 % CI | upper < 1 | reference 2.5th | label per seed | combined |
|---|---|---|---|---|---|---|---|---|---|
| R3 | readers S, T, O | 3,072 | s0 | 0.469 | [0.349, 0.629] | yes | 0.967 | READABLE | READABLE |
| R3 | readers S, T, O | 3,072 | s1 | 0.466 | [0.342, 0.632] | yes | 0.966 | READABLE |  |
| R4 | levels L1, L2, L3 | 384 | s0 | 0.586 | [0.449, 0.762] | yes | 0.989 | READABLE | READABLE |
| R4 | levels L1, L2, L3 | 384 | s1 | 0.545 | [0.429, 0.693] | yes | 0.987 | READABLE |  |
| R3s | read_s (beside) | 1,024 | s0 | 0.702 | [0.565, 0.873] | yes | 0.998 | READABLE | READABLE |
| R3s | read_s (beside) | 1,024 | s1 | 0.569 | [0.427, 0.755] | yes | 0.975 | READABLE |  |
| R3t | read_t (beside) | 1,024 | s0 | 0.477 | [0.353, 0.635] | yes | 0.968 | READABLE | READABLE |
| R3t | read_t (beside) | 1,024 | s1 | 0.460 | [0.332, 0.630] | yes | 0.952 | READABLE |  |
| R3o | read_o (beside) | 1,024 | s0 | 0.597 | [0.426, 0.836] | yes | 0.948 | READABLE | READABLE |
| R3o | read_o (beside) | 1,024 | s1 | 0.580 | [0.448, 0.750] | yes | 0.971 | READABLE |  |
| L1 | level 1 (beside) | 128 | s0 | 0.627 | [0.481, 0.812] | yes | 0.981 | READABLE | SEEDS-DISAGREE |
| L1 | level 1 (beside) | 128 | s1 | 0.734 | [0.532, 1.001] | no | 0.991 | UNREADABLE |  |
| L2 | level 2 (beside) | 128 | s0 | 0.635 | [0.483, 0.856] | yes | 0.993 | READABLE | READABLE |
| L2 | level 2 (beside) | 128 | s1 | 0.554 | [0.437, 0.703] | yes | 0.983 | READABLE |  |
| L3 | level 3 (beside) | 128 | s0 | 0.636 | [0.494, 0.821] | yes | 0.991 | READABLE | READABLE |
| L3 | level 3 (beside) | 128 | s1 | 0.703 | [0.547, 0.898] | yes | 0.978 | READABLE |  |
| R3 re-initialised (beside) | seed 100 | 3,072 | | 0.708 | [0.545, 0.923] | yes | 0.936 | READABLE | |
| R3 re-initialised (beside) | seed 101 | 3,072 | | 0.689 | [0.534, 0.897] | yes | 0.918 | READABLE | |
| R4 re-initialised (beside) | seed 100 | 384 | | 0.977 | [0.810, 1.170] | no | 1.007 | UNREADABLE | |
| R4 re-initialised (beside) | seed 101 | 384 | | 0.919 | [0.740, 1.131] | no | 0.967 | UNREADABLE | |

Figure: `fig/ladder_lead_gap.png` (NMSE per rung with the bootstrap band, both seeds).

## LOSS table per step (Delta NMSE = NMSE(after) - NMSE(before); LOSS = paired bootstrap lower bound > 0 AND point above the 97.5th percentile of the 199 reference Deltas)

| step | seed | Delta NMSE | 95 % paired CI | reference 97.5th | label | in an outcome |
|---|---|---|---|---|---|---|
| R1 -> R2 |  | -0.002 | [-0.034, 0.027] | 0.066 | no loss | yes |
| R0 -> R1 |  | -0.012 | [-0.041, 0.017] | 0.039 | no loss | yes |
| R2 -> R3 | s0 | -0.032 | [-0.135, 0.080] | 0.207 | no loss | yes |
| R2 -> R4 | s0 | 0.086 | [-0.024, 0.214] | 0.278 | no loss | yes |
| R3 -> R4 | s0 | 0.118 | [-0.023, 0.258] | 0.273 | no loss | beside |
| R2 -> R3s | s0 | 0.201 | [0.072, 0.342] | 0.188 | LOSS | beside |
| R2 -> R3t | s0 | -0.024 | [-0.125, 0.093] | 0.340 | no loss | beside |
| R2 -> R3o | s0 | 0.097 | [-0.056, 0.292] | 0.184 | no loss | beside |
| R2 -> L1 | s0 | 0.126 | [0.007, 0.273] | 0.288 | no loss | yes |
| L1 -> L2 | s0 | 0.008 | [-0.114, 0.150] | 0.181 | no loss | yes |
| L2 -> L3 | s0 | 0.002 | [-0.093, 0.079] | 0.128 | no loss | yes |
| R2 -> R3 | s1 | -0.034 | [-0.145, 0.093] | 0.212 | no loss | yes |
| R2 -> R4 | s1 | 0.044 | [-0.053, 0.148] | 0.201 | no loss | yes |
| R3 -> R4 | s1 | 0.078 | [-0.042, 0.191] | 0.160 | no loss | beside |
| R2 -> R3s | s1 | 0.068 | [-0.063, 0.225] | 0.229 | no loss | beside |
| R2 -> R3t | s1 | -0.041 | [-0.145, 0.076] | 0.247 | no loss | beside |
| R2 -> R3o | s1 | 0.079 | [-0.047, 0.216] | 0.159 | no loss | beside |
| R2 -> L1 | s1 | 0.234 | [0.073, 0.430] | 0.256 | no loss | yes |
| L1 -> L2 | s1 | -0.180 | [-0.366, -0.036] | 0.161 | no loss | yes |
| L2 -> L3 | s1 | 0.149 | [0.044, 0.264] | 0.202 | no loss | yes |

Seed agreement for the model steps: R2 -> R3: no loss; R2 -> R4: no loss. R3 -> R4 is not a step of the real dataflow (L1 reads R2; IMPL_CHOICES section 3) and is used in no outcome.

## Outcomes the card commits (read on `lead_gap`)

| outcome | holds | note |
|---|---|---|
| O-ENC | no | read at R0 |
| O-POOL8 | no | separated Delta NMSE -0.012 |
| O-POOL4 | no | separated Delta NMSE -0.002 |
| O-NET | no | per seed (LOSS of R2->R3 / R2->R4): s0: False / False; s1: False / False |
| O-DIFFUSE | no |  |
| O-NONE | yes |  |

**Triggered:** O-NONE. Ranked by the separated Delta NMSE (Rule Zero 5): no ranked step.

* O-NONE triggers: re-read IMAG1B's IA-2 gap row on frames before any arm.

## Beside rows (no bar; point and 95 % episode-cluster interval; deciding rungs and R2m only)

| rung | log(max(gap, 0.1 m)) NMSE | `lead_gap` <= 30 m NMSE | `lead_closing` NMSE | `ttc` NMSE | `lead_gap` MLP on PCA-256 NMSE | `lead_present` balanced accuracy - 0.5 |
|---|---|---|---|---|---|---|
| R0 | 0.447 [0.331, 0.591] | 0.500 [0.418, 0.590] | 0.988 [0.939, 1.051] | 1.066 [1.011, 1.144] | 0.695 [0.602, 0.800] | 0.218 [0.146, 0.283] (n 4978, balanced) |
| R1 | 0.405 [0.283, 0.575] | 0.463 [0.383, 0.553] | 0.989 [0.930, 1.062] | 1.071 [1.010, 1.162] | 0.662 [0.567, 0.771] | 0.213 [0.142, 0.277] (n 4978, balanced) |
| R2 | 0.382 [0.249, 0.569] | 0.435 [0.347, 0.539] | 0.997 [0.929, 1.092] | 1.089 [1.011, 1.205] | 0.665 [0.557, 0.786] | 0.202 [0.130, 0.270] (n 4978, balanced) |
| R2m | 0.546 [0.382, 0.771] | 0.570 [0.436, 0.748] | 1.047 [0.935, 1.220] | 1.049 [0.984, 1.136] | 0.738 [0.568, 0.955] | 0.203 [0.132, 0.269] (n 4978, balanced) |
| R3_s0 | 0.494 [0.345, 0.703] | 0.528 [0.377, 0.731] | 0.835 [0.717, 0.977] | 0.929 [0.778, 1.157] | 0.506 [0.381, 0.684] | 0.362 [0.324, 0.398] (n 4978, balanced) |
| R3_s1 | 0.456 [0.288, 0.706] | 0.546 [0.340, 0.858] | 0.876 [0.779, 0.994] | 0.938 [0.799, 1.155] | 0.479 [0.350, 0.654] | 0.340 [0.293, 0.382] (n 4978, balanced) |
| R4_s0 | 0.513 [0.339, 0.757] | 0.698 [0.503, 0.923] | 1.015 [0.963, 1.080] | 1.042 [0.981, 1.130] | 0.702 [0.485, 1.005] | 0.210 [0.143, 0.272] (n 4978, balanced) |
| R4_s1 | 0.502 [0.338, 0.725] | 0.690 [0.534, 0.865] | 1.069 [0.961, 1.249] | 1.110 [0.964, 1.339] | 0.550 [0.395, 0.752] | 0.223 [0.159, 0.280] (n 4978, balanced) |

## Disclosures
* 5 outer folds by `sha256(episode)[:8] % 5`: IMAG1B has only a two-fold split (IMPL_CHOICES section 1). `lead_present` is scored on a balanced 4,978-frame set at every rung, not on all frames. Beside targets on subsets use set A's standardisation. R3 carries the model's ego / nav conditioning as well as the scene tokens. Tier T0: a representation diagnostic, never a driving claim.
* A synthetic smoke before the statistics is disclosed in `IMPL_ADDENDUM_CAPLADDER.md`.
