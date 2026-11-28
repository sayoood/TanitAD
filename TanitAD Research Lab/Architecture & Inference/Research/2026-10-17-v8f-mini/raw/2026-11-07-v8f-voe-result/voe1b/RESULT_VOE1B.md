# RESULT_VOE1B: does each level notice an impossible future beyond what an untrained network does? (violation of expectation, calibrated reference; tier T0)
Card `../PREREG_V8F_VOE1B.md` (frozen prefix 5,785 bytes, sha256 `aa317fe3...`, unedited). Definitions: `IMPL_CHOICES_VOE1B.md` (sha256 `f7c2d624`, deltas) on top of VOE1's `IMPL_CHOICES_VOE1.md` (`848a2c8c`). Models: M2b s0 and s1, zero training; reference: 16 untrained world models per chassis (`torch.manual_seed` 100..115; every wm.* parameter re-initialised). 1,976 windows of eval-141, onset t0 = window frame 4, 1.6 s = 8 cache steps scored. **Tier T0** (a world-model diagnostic on cached latents, teacher-forced one-step prediction; never a driving claim). **STRATEGIC family: PAUSED (PI ruling, `PI_DECISION_QUEUE.md` item 25); nothing here is a strategic result.** Every number below is computed by `run_voe1b.py` into `RESULT_VOE1B.json` (run 2026-11-07 15:57:42 to 2026-11-07 16:06:10, 514 s).

## 0. Controls (checked before any row is read)
**Run status: VALID**

| control | rule (card) | result | pass |
|---|---|---|---|
| 1 real-vs-real, calibrated | count of separated cells among the 6 M2b cells (2 seeds x 3 levels; later copy 0.2 s later) must not exceed the 95th percentile of the same count over the R = 199 reference draws scored as if real | real count **0** of 6; reference-draw counts: mean 0.23, 95th percentile 1.00 (inverted-cdf 1), histogram {0: 156, 1: 40, 2: 3} | PASS |
| 2 untrained leave-one-out | each untrained seed in turn is 'the model' (rule (a) at that seed plus rule (c) against the other 15 of its chassis); over all (chassis, seed, manipulation, level) cells of FREEZE, ACTION-SWAP, ACTION-SWAP-M at most 10 % NOTICES | **9 of 288 = 3.12%** NOTICES (expected at most 1/16 = 6.25%); by chassis 5/144 and 4/144; by manipulation FREEZE 1/96, ACTION-SWAP 6/96, ACTION-SWAP-M 2/96 | PASS |
| 3 calibration | 20 extra permutation draws (1001-1020) scored as if real: at most 5 % of rows separated per family | ACC 283/12240 = 2.312%; NULL-ACC 9/720 = 1.250%; DIFF 21/720 = 2.917%; RVR 93/2040 = 4.559%; AP-CHANCE 6/360 = 1.667%; AP-NULL 3/360 = 0.833% | PASS |
| 4 synthetic test | `test_voe1b_synth.py`: planted toy NOTICES on FREEZE and TELEPORT (with rule (c)), action-blind toy BLIND on ACTION-SWAP, **a toy family biased in random directions gives INIT-LEVEL not NOTICES (added)**, plus VOE1's instrument tests; 22 tests | 22 passed; test file and core hash equal the log's: True | PASS |

Control 1 cells (accuracy of 'the later copy scores higher', two-part rule, either side):

| cell | n | accuracy [interval] | separated either side |
|---|---|---|---|
| M2b_s0/L1 | 1976 | 0.497 [0.489, 0.506] | False |
| M2b_s0/L2 | 1976 | 0.488 [0.461, 0.516] | False |
| M2b_s0/L3 | 1976 | 0.492 [0.467, 0.518] | False |
| M2b_s1/L1 | 1976 | 0.504 [0.495, 0.513] | False |
| M2b_s1/L2 | 1976 | 0.487 [0.460, 0.515] | False |
| M2b_s1/L3 | 1976 | 0.502 [0.475, 0.528] | False |

Beside (gates nothing): the untrained real-vs-real cells, 9 of 96 separated on either side (UT0_101, UT0_108, UT0_112, UT1_100, UT1_102, UT1_103, UT1_105, UT1_113, UT1_114).

Control 2 detail (NOTICES members of the leave-one-out, per chassis x manipulation x level; each cell has 16 members):

| chassis | manipulation | L1 | L2 | L3 |
|---|---|---|---|---|
| UT0 | FREEZE | 0 | 0 | 1 |
| UT0 | ACTION-SWAP | 1 | 1 | 1 |
| UT0 | ACTION-SWAP-M | 0 | 1 | 0 |
| UT1 | FREEZE | 0 | 0 | 0 |
| UT1 | ACTION-SWAP | 1 | 1 | 1 |
| UT1 | ACTION-SWAP-M | 0 | 1 | 0 |

Reproduction (MEASURED): the unmodified path (per-clip frame bank + adapter + wm.z1) vs `latents.pt` `Z1`, 1,976 windows: max abs difference 3.81e-06 (s0), 2.86e-06 (s1) on values up to 4.9; logged actions 0.0. Untrained family: 32 models, each re-initialised 2,007,874 of 2,007,874 wm.* parameters (133 tensors, none equal to the trained one); chassis (adapter, pos) trained.

## 1. Headline: the label of each (manipulation, level) over both seeds
NOTICES needs all three of (a) accuracy two-part separated above 0.5, (b) [TELEPORT, REVERSE, SPLICE] separated above the Copy-Last null, (c) accuracy above the MAXIMUM of the 16 untrained accuracies of that chassis, manipulation and level, at BOTH seeds. INIT-LEVEL = (a), (b) hold, (c) fails (an untrained network of the same architecture does this too). NULL-ONLY = (a) holds, (b) fails. BLIND = (a) fails.

| manipulation | n windows (episodes) | level 1 (0.2 s) | level 2 (0.4 s) | level 3 (0.8 s) |
|---|---|---|---|---|
| TELEPORT | 1,976 (141) | NULL-ONLY | NULL-ONLY | NULL-ONLY |
| FREEZE | 1,725 (137) | BLIND | BLIND | BLIND |
| REVERSE | 1,976 (141) | NULL-ONLY | NULL-ONLY | BLIND |
| SPLICE | 1,976 (141) | NULL-ONLY | NULL-ONLY | NULL-ONLY |
| ACTION-SWAP | 1,976 (141) | NOTICES | NOTICES | INIT-LEVEL |
| ACTION-SWAP-M | 530 (104) | NOTICES | NOTICES | NOTICES |

Per-seed labels (s0 / s1): TELEPORT: L1 NULL-ONLY/NULL-ONLY, L2 NULL-ONLY/NULL-ONLY, L3 NULL-ONLY/NULL-ONLY; FREEZE: L1 BLIND/BLIND, L2 BLIND/BLIND, L3 BLIND/BLIND; REVERSE: L1 NULL-ONLY/NULL-ONLY, L2 NULL-ONLY/NULL-ONLY, L3 BLIND/BLIND; SPLICE: L1 NULL-ONLY/NULL-ONLY, L2 NULL-ONLY/NULL-ONLY, L3 NULL-ONLY/NULL-ONLY; ACTION-SWAP: L1 NOTICES/NOTICES, L2 NOTICES/NOTICES, L3 INIT-LEVEL/INIT-LEVEL; ACTION-SWAP-M: L1 NOTICES/NOTICES, L2 NOTICES/NOTICES, L3 NOTICES/NOTICES

Per-seed conditions (a)(b)(c) of every cell are in the tables of section 2; `LABELS_VOE1B.json` holds them machine-readably.

## 2. Pairwise accuracy per manipulation x level x seed, beside the untrained reference
Accuracy = share of pairs where the manipulated copy's max normalised surprise over the 1.6 s after t0 is above its unmodified copy's (ties 0.5); interval = episode-cluster bootstrap (B 2,000, seed 0); flag S = two-part separated above (interval above the claim AND point above the 97.5th percentile of the R = 199 whole-episode label-swap reference), b = bootstrap interval only, - = neither. The untrained columns are the 16 accuracies (point values) of the untrained models on the SAME chassis, manipulation and level: min / median / max, and how many of them are themselves separated above 0.5. (a)(b)(c): T = holds, F = fails, . = not applicable. Copy-Last null accuracy and the paired model-minus-null difference (claim 0) are beside.

### TELEPORT  (n = 1,976 windows, 141 episodes)  [the Copy-Last null enters NOTICES]
| tag | level | n (episodes) | accuracy [interval] | untrained min / median / max (separated above 0.5) | (a)(b)(c) | Copy-Last null acc | model minus null [interval] | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | L1 | 1,976 (141) | 0.896 [0.878, 0.914] S | 0.480 / 0.516 / 0.551 (3 of 16) | TFT | 0.898 [0.879, 0.917] | -0.003 [-0.012, +0.007] - | NULL-ONLY |
| M2b_s0 | L2 | 1,976 (141) | 0.812 [0.788, 0.836] S | 0.494 / 0.533 / 0.566 (12 of 16) | TFT | 0.828 [0.807, 0.849] | -0.016 [-0.029, -0.002] S-below | NULL-ONLY |
| M2b_s0 | L3 | 1,976 (141) | 0.772 [0.751, 0.793] S | 0.473 / 0.499 / 0.520 (2 of 16) | TFT | 0.773 [0.750, 0.796] | -0.001 [-0.015, +0.012] - | NULL-ONLY |
| M2b_s1 | L1 | 1,976 (141) | 0.901 [0.882, 0.920] S | 0.475 / 0.507 / 0.543 (2 of 16) | TFT | 0.902 [0.884, 0.920] | -0.001 [-0.011, +0.009] - | NULL-ONLY |
| M2b_s1 | L2 | 1,976 (141) | 0.825 [0.803, 0.847] S | 0.506 / 0.538 / 0.575 (12 of 16) | TFT | 0.828 [0.805, 0.851] | -0.003 [-0.016, +0.010] - | NULL-ONLY |
| M2b_s1 | L3 | 1,976 (141) | 0.775 [0.754, 0.797] S | 0.479 / 0.503 / 0.529 (2 of 16) | TFT | 0.785 [0.763, 0.808] | -0.010 [-0.024, +0.004] - | NULL-ONLY |

### FREEZE  (n = 1,725 windows, 137 episodes)  [the null does not enter]
| tag | level | n (episodes) | accuracy [interval] | untrained min / median / max (separated above 0.5) | (a)(b)(c) | Copy-Last null acc | model minus null [interval] | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | L1 | 1,725 (137) | 0.315 [0.299, 0.331] S-below | 0.324 / 0.360 / 0.390 (0 of 16) | F.F | 0.254 [0.246, 0.261] | +0.061 [+0.045, +0.079] S | BLIND |
| M2b_s0 | L2 | 1,725 (137) | 0.322 [0.301, 0.342] S-below | 0.408 / 0.449 / 0.479 (0 of 16) | F.F | 0.245 [0.237, 0.252] | +0.077 [+0.057, +0.097] S | BLIND |
| M2b_s0 | L3 | 1,725 (137) | 0.294 [0.275, 0.316] S-below | 0.471 / 0.499 / 0.521 (2 of 16) | F.F | 0.252 [0.245, 0.259] | +0.042 [+0.022, +0.063] S | BLIND |
| M2b_s1 | L1 | 1,725 (137) | 0.298 [0.280, 0.317] S-below | 0.342 / 0.364 / 0.376 (0 of 16) | F.F | 0.246 [0.238, 0.252] | +0.052 [+0.034, +0.072] S | BLIND |
| M2b_s1 | L2 | 1,725 (137) | 0.299 [0.279, 0.318] S-below | 0.407 / 0.434 / 0.460 (0 of 16) | F.F | 0.248 [0.240, 0.255] | +0.051 [+0.032, +0.070] S | BLIND |
| M2b_s1 | L3 | 1,725 (137) | 0.286 [0.265, 0.308] S-below | 0.463 / 0.502 / 0.518 (0 of 16) | F.F | 0.239 [0.231, 0.247] | +0.047 [+0.026, +0.069] S | BLIND |

### REVERSE  (n = 1,976 windows, 141 episodes)  [the Copy-Last null enters NOTICES]
| tag | level | n (episodes) | accuracy [interval] | untrained min / median / max (separated above 0.5) | (a)(b)(c) | Copy-Last null acc | model minus null [interval] | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | L1 | 1,976 (141) | 0.904 [0.887, 0.921] S | 0.465 / 0.514 / 0.530 (1 of 16) | TFT | 0.908 [0.893, 0.922] | -0.004 [-0.015, +0.007] - | NULL-ONLY |
| M2b_s0 | L2 | 1,976 (141) | 0.720 [0.695, 0.746] S | 0.487 / 0.518 / 0.567 (6 of 16) | TFT | 0.730 [0.706, 0.755] | -0.010 [-0.025, +0.005] - | NULL-ONLY |
| M2b_s0 | L3 | 1,976 (141) | 0.521 [0.494, 0.548] - | 0.475 / 0.495 / 0.536 (2 of 16) | FTF | 0.477 [0.454, 0.499] | +0.044 [+0.020, +0.068] S | BLIND |
| M2b_s1 | L1 | 1,976 (141) | 0.913 [0.897, 0.929] S | 0.475 / 0.503 / 0.544 (4 of 16) | TFT | 0.908 [0.893, 0.923] | +0.006 [-0.007, +0.018] - | NULL-ONLY |
| M2b_s1 | L2 | 1,976 (141) | 0.735 [0.709, 0.763] S | 0.472 / 0.524 / 0.570 (7 of 16) | TFT | 0.742 [0.717, 0.769] | -0.007 [-0.021, +0.008] - | NULL-ONLY |
| M2b_s1 | L3 | 1,976 (141) | 0.520 [0.495, 0.546] - | 0.479 / 0.503 / 0.529 (2 of 16) | FTF | 0.480 [0.456, 0.503] | +0.040 [+0.018, +0.062] S | BLIND |

### SPLICE  (n = 1,976 windows, 141 episodes)  [the Copy-Last null enters NOTICES]
| tag | level | n (episodes) | accuracy [interval] | untrained min / median / max (separated above 0.5) | (a)(b)(c) | Copy-Last null acc | model minus null [interval] | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | L1 | 1,976 (141) | 1.000 [1.000, 1.000] S | 0.507 / 0.528 / 0.573 (1 of 16) | TFT | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] - | NULL-ONLY |
| M2b_s0 | L2 | 1,976 (141) | 1.000 [1.000, 1.000] S | 0.539 / 0.617 / 0.712 (15 of 16) | TFT | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] - | NULL-ONLY |
| M2b_s0 | L3 | 1,976 (141) | 0.996 [0.992, 0.999] S | 0.430 / 0.509 / 0.579 (1 of 16) | TFT | 0.996 [0.992, 0.999] | +0.000 [-0.003, +0.004] - | NULL-ONLY |
| M2b_s1 | L1 | 1,976 (141) | 1.000 [1.000, 1.000] S | 0.504 / 0.535 / 0.573 (1 of 16) | TFT | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] - | NULL-ONLY |
| M2b_s1 | L2 | 1,976 (141) | 1.000 [1.000, 1.000] S | 0.572 / 0.630 / 0.702 (15 of 16) | TFT | 0.999 [0.998, 1.000] | +0.001 [+0.000, +0.002] - | NULL-ONLY |
| M2b_s1 | L3 | 1,976 (141) | 0.996 [0.992, 0.998] S | 0.472 / 0.510 / 0.599 (1 of 16) | TFT | 0.992 [0.987, 0.996] | +0.004 [-0.001, +0.009] - | NULL-ONLY |

### ACTION-SWAP  (n = 1,976 windows, 141 episodes)  [the null does not enter]
| tag | level | n (episodes) | accuracy [interval] | untrained min / median / max (separated above 0.5) | (a)(b)(c) | Copy-Last null acc | model minus null [interval] | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | L1 | 1,976 (141) | 0.632 [0.606, 0.657] S | 0.434 / 0.479 / 0.611 (3 of 16) | T.T | 0.500 [0.500, 0.500] | +0.132 [+0.106, +0.157] S | NOTICES |
| M2b_s0 | L2 | 1,976 (141) | 0.679 [0.653, 0.705] S | 0.418 / 0.478 / 0.580 (2 of 16) | T.T | 0.500 [0.500, 0.500] | +0.179 [+0.153, +0.205] S | NOTICES |
| M2b_s0 | L3 | 1,976 (141) | 0.738 [0.716, 0.759] S | 0.322 / 0.509 / 0.759 (6 of 16) | T.F | 0.500 [0.500, 0.500] | +0.238 [+0.216, +0.259] S | INIT-LEVEL |
| M2b_s1 | L1 | 1,976 (141) | 0.604 [0.580, 0.629] S | 0.386 / 0.511 / 0.569 (5 of 16) | T.T | 0.500 [0.500, 0.500] | +0.104 [+0.080, +0.129] S | NOTICES |
| M2b_s1 | L2 | 1,976 (141) | 0.656 [0.628, 0.684] S | 0.388 / 0.490 / 0.612 (5 of 16) | T.T | 0.500 [0.500, 0.500] | +0.156 [+0.128, +0.184] S | NOTICES |
| M2b_s1 | L3 | 1,976 (141) | 0.733 [0.708, 0.759] S | 0.381 / 0.511 / 0.743 (6 of 16) | T.F | 0.500 [0.500, 0.500] | +0.233 [+0.208, +0.259] S | INIT-LEVEL |

### ACTION-SWAP-M  (n = 530 windows, 104 episodes)  [the null does not enter]
| tag | level | n (episodes) | accuracy [interval] | untrained min / median / max (separated above 0.5) | (a)(b)(c) | Copy-Last null acc | model minus null [interval] | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | L1 | 530 (104) | 0.583 [0.538, 0.630] S | 0.436 / 0.495 / 0.551 (0 of 16) | T.T | 0.500 [0.500, 0.500] | +0.083 [+0.038, +0.130] S | NOTICES |
| M2b_s0 | L2 | 530 (104) | 0.643 [0.601, 0.686] S | 0.425 / 0.489 / 0.575 (1 of 16) | T.T | 0.496 [0.491, 0.500] | +0.147 [+0.105, +0.190] S | NOTICES |
| M2b_s0 | L3 | 530 (104) | 0.668 [0.623, 0.712] S | 0.475 / 0.497 / 0.515 (0 of 16) | T.T | 0.500 [0.494, 0.506] | +0.168 [+0.122, +0.213] S | NOTICES |
| M2b_s1 | L1 | 530 (104) | 0.585 [0.541, 0.628] S | 0.457 / 0.501 / 0.558 (0 of 16) | T.T | 0.500 [0.500, 0.500] | +0.085 [+0.041, +0.128] S | NOTICES |
| M2b_s1 | L2 | 530 (104) | 0.611 [0.566, 0.653] S | 0.457 / 0.502 / 0.596 (2 of 16) | T.T | 0.504 [0.500, 0.510] | +0.108 [+0.062, +0.150] S | NOTICES |
| M2b_s1 | L3 | 530 (104) | 0.677 [0.629, 0.725] S | 0.485 / 0.502 / 0.532 (0 of 16) | T.T | 0.494 [0.480, 0.503] | +0.183 [+0.132, +0.236] S | NOTICES |

## 3. Event alignment (IA-6d, VOE1's definitions, unchanged; reported beside, no bar): do the surprise peaks sit within +-0.4 s of braking, turn and lead-change events?
Unmodified 16-frame windows, every (window, step) point of a level pooled; positive = an event onset within +-0.4 s (+-2 cache steps) of the step. Onsets in the 141 eval clips: brake 47, turn 100, lead 27. AP = average precision of ranking points by surprise; chance = the event rate; null AP = the same ranking by the Copy-Last distance of the level. S = two-part separated above 0 (bootstrap interval AND point above the 97.5th percentile of R = 199 label-permutation draws). The last columns give, for the 16 untrained models of the same chassis, the min / median / max of AP minus chance (no interval).

### brake

| level | points (positives) | seed | AP | chance | null AP | AP minus chance [interval] | AP minus null AP [interval] | untrained AP minus chance (min / median / max) | ALIGNS (both seeds) |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 29,640 (665) | s0 | 0.0211 | 0.0224 | 0.0209 | -0.0013 [-0.0039, +0.0026] - | +0.0002 [-0.0008, +0.0012] - | -0.0045 / +0.0017 / +0.0092 | no |
|  | 29,640 (665) | s1 | 0.0216 | 0.0224 | 0.0215 | -0.0009 [-0.0037, +0.0034] - | +0.0001 [-0.0010, +0.0010] - | -0.0075 / +0.0004 / +0.0068 |  |
| L2 | 13,832 (293) | s0 | 0.0201 | 0.0212 | 0.0194 | -0.0011 [-0.0039, +0.0036] - | +0.0007 [-0.0008, +0.0023] - | -0.0056 / +0.0004 / +0.0322 | no |
|  | 13,832 (293) | s1 | 0.0191 | 0.0212 | 0.0183 | -0.0020 [-0.0054, +0.0041] - | +0.0009 [-0.0004, +0.0032] - | -0.0054 / +0.0013 / +0.0136 |  |
| L3 | 5,928 (118) | s0 | 0.0177 | 0.0199 | 0.0178 | -0.0022 [-0.0049, +0.0022] - | -0.0001 [-0.0025, +0.0023] - | -0.0061 / -0.0000 / +0.0365 | no |
|  | 5,928 (118) | s1 | 0.0181 | 0.0199 | 0.0170 | -0.0018 [-0.0057, +0.0040] - | +0.0011 [-0.0010, +0.0042] - | -0.0077 / -0.0006 / +0.0095 |  |

### turn

| level | points (positives) | seed | AP | chance | null AP | AP minus chance [interval] | AP minus null AP [interval] | untrained AP minus chance (min / median / max) | ALIGNS (both seeds) |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 29,640 (1079) | s0 | 0.0416 | 0.0364 | 0.0391 | +0.0052 [-0.0007, +0.0162] - | +0.0025 [-0.0010, +0.0060] - | -0.0109 / -0.0001 / +0.0465 | no |
|  | 29,640 (1079) | s1 | 0.0391 | 0.0364 | 0.0380 | +0.0027 [-0.0022, +0.0114] - | +0.0011 [-0.0010, +0.0032] - | -0.0116 / +0.0064 / +0.0383 |  |
| L2 | 13,832 (509) | s0 | 0.0556 | 0.0368 | 0.0497 | +0.0188 [+0.0107, +0.0314] S | +0.0060 [+0.0004, +0.0129] S | -0.0092 / +0.0016 / +0.0480 | no |
|  | 13,832 (509) | s1 | 0.0553 | 0.0368 | 0.0500 | +0.0185 [+0.0092, +0.0386] S | +0.0053 [-0.0004, +0.0121] - | -0.0119 / +0.0089 / +0.0325 |  |
| L3 | 5,928 (215) | s0 | 0.0615 | 0.0363 | 0.0552 | +0.0253 [+0.0118, +0.0512] S | +0.0063 [-0.0051, +0.0234] - | -0.0118 / +0.0022 / +0.0196 | no |
|  | 5,928 (215) | s1 | 0.0746 | 0.0363 | 0.0689 | +0.0383 [+0.0155, +0.0818] S | +0.0056 [-0.0118, +0.0297] - | -0.0106 / +0.0001 / +0.0367 |  |

### lead

| level | points (positives) | seed | AP | chance | null AP | AP minus chance [interval] | AP minus null AP [interval] | untrained AP minus chance (min / median / max) | ALIGNS (both seeds) |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 29,640 (318) | s0 | 0.0102 | 0.0107 | 0.0104 | -0.0005 [-0.0026, +0.0021] - | -0.0002 [-0.0013, +0.0005] - | -0.0032 / +0.0002 / +0.0235 | no |
|  | 29,640 (318) | s1 | 0.0096 | 0.0107 | 0.0099 | -0.0012 [-0.0032, +0.0016] - | -0.0003 [-0.0013, +0.0002] - | -0.0022 / -0.0009 / +0.0021 |  |
| L2 | 13,832 (148) | s0 | 0.0109 | 0.0107 | 0.0114 | +0.0002 [-0.0021, +0.0055] - | -0.0005 [-0.0023, +0.0005] - | -0.0034 / +0.0013 / +0.0121 | no |
|  | 13,832 (148) | s1 | 0.0103 | 0.0107 | 0.0114 | -0.0004 [-0.0028, +0.0049] - | -0.0010 [-0.0061, -0.0000] - | -0.0028 / -0.0007 / +0.0162 |  |
| L3 | 5,928 (65) | s0 | 0.0083 | 0.0110 | 0.0084 | -0.0027 [-0.0044, +0.0002] - | -0.0001 [-0.0013, +0.0004] - | -0.0031 / -0.0002 / +0.0078 | no |
|  | 5,928 (65) | s1 | 0.0101 | 0.0110 | 0.0110 | -0.0009 [-0.0036, +0.0061] - | -0.0009 [-0.0043, +0.0001] - | -0.0035 / +0.0008 / +0.0448 |  |

**ALIGNS summary:** brake: no level; turn: no level; lead: no level.

**Hierarchy hypothesis (HYPOTHESIS, no bar):** (AP - chance at L3) minus (AP - chance at L1), episode-cluster interval from the shared bootstrap draws:
| event | seed | L1 AP-chance | L3 AP-chance | L3 minus L1 [interval] |
|---|---|---|---|---|
| brake | s0 | -0.0013 | -0.0022 | -0.0009 [-0.0043, +0.0031] |
| brake | s1 | -0.0009 | -0.0018 | -0.0009 [-0.0047, +0.0040] |
| turn | s0 | +0.0052 | +0.0253 | +0.0201 [+0.0067, +0.0424] |
| turn | s1 | +0.0027 | +0.0383 | +0.0356 [+0.0118, +0.0775] |
| lead | s0 | -0.0005 | -0.0027 | -0.0022 [-0.0046, +0.0004] |
| lead | s1 | -0.0012 | -0.0009 | +0.0003 [-0.0025, +0.0064] |

## 4. Triggers (the card's committed consequences, evaluated mechanically on the over-both-seeds labels)
| committed trigger | fires? | consequence if it fires (card) |
|---|---|---|
| ACTION-SWAP-M BLIND or INIT-LEVEL at every level | **no** (L1 NOTICES, L2 NOTICES, L3 NOTICES) | the error does not use the action beyond what an untrained network does; together with IMAG1's IA-3 (IMAG1 result file present (`../imag1/RESULT_IMAG1.md`): read it for IA-3 and IA-2) the lever is **E7 DIVRS** plus a counterfactual-target term |
| FREEZE BLIND or INIT-LEVEL at every level | **YES** (L1 BLIND, L2 BLIND, L3 BLIND) | the predictor does not carry motion beyond its architecture: **MRATE** (carded) |
| TELEPORT / SPLICE NULL-ONLY at every level | **YES** (TELEPORT L1 NULL-ONLY, L2 NULL-ONLY, L3 NULL-ONLY; SPLICE L1 NULL-ONLY, L2 NULL-ONLY, L3 NULL-ONLY) | surprise is a frame-change detector; IMAG1's IA-2 decides before any arm |
| NOTICES on FREEZE or ACTION-SWAP-M at any level | **YES** (ACTION-SWAP-M L1, ACTION-SWAP-M L2, ACTION-SWAP-M L3) | the first positive 'imagination is used' evidence on video; the per-level table and the surprise traces (`fig/`) go into the paper |

ORIGINAL ACTION-SWAP (VOE1's donor rule, beside; feeds no trigger): L1 NOTICES, L2 NOTICES, L3 INIT-LEVEL.

Event alignment (section 3) is VOE1's, reported beside with no bar.

## 5. Disclosures
* Windows: TELEPORT 1,976 (141 ep), FREEZE 1,725 (137 ep), REVERSE 1,976 (141 ep), SPLICE 1,976 (141 ep), ACTION-SWAP 1,976 (141 ep), ACTION-SWAP-M 530 (104 ep), RVR 1,976 (141 ep). ACTION-SWAP-M candidates per qualifying window: min 1, median 2. Neighbouring windows overlap by 12 of 16 frames and are 0.8 s apart; the episode-cluster bootstrap and the whole-episode permutation carry that.
* Estimator discipline: every interval is an episode-cluster bootstrap over the same episodes (it answers 'would another draw of EPISODES say this?'). The untrained reference adds initialisation variance of an untrained architecture (rule (c)); it is NOT a replicate of the training run (H-ESTIM-SEED-1): s0 and s1 are two trained seeds of one recipe and 'both seeds' is the only guard against training-seed variance here. A NOTICES label is a necessary, not a sufficient, condition for 'training taught the model to use it'.
* The untrained models of the two chassis share bit-identical wm.* weights per seed (same `torch.manual_seed`); only the tokens they read differ. Each trained seed is compared with the 16 untrained models on its own chassis. Rule (c) is strict (equal to the maximum is not above it) and uses point accuracies.
* ACTION-SWAP-M: magnitude within +-10 % on mean |lon| and mean |yaw|, opposite sign where both signed means reach 0.5 m/s^2 / 0.05 rad/s (the stricter reading; n under the looser reading 545, magnitude match alone 1,714; IMPL_CHOICES section 1).
* REVERSE is the 1.6 s segment played backwards (8-step onset jump then reversed motion); FREEZE is a pause; the original ACTION-SWAP donors differ by at least 1.5 m/s^2 or 0.15 rad/s mean action (VOE1 section 2).
* The models were trained on 16-frame windows with teacher forcing and context 3. Events: brake 47 onsets, turn 100, lead 27; a cell that is not separated says 'not shown', not 'absent'. STRATEGIC family: PAUSED (item 25), not reported as a pass.

## 6. Figures (`fig/`, PNG, no camera frame)
* `fig/trace_<MANIP>.png` (six): per level (columns) and model seed (rows) the M2b median (over the qualifying windows) of the normalised surprise against time from t0, manipulated and unmodified; the violet band is the min to max of the 16 untrained models' manipulated per-step medians on the same chassis (each normalised by its own unmodified median), the violet dashed line their median. (M2b interquartile bands are in the traces file, not drawn.)
