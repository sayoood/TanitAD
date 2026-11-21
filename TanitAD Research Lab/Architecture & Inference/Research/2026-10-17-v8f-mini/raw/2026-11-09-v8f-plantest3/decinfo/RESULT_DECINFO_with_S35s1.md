# RESULT_DECINFO: where on the vision chain are speed and yaw rate lost?
Tier **T0**, a readout diagnostic on the PLANTEST3 TUNE inner split (53 fit episodes, 17 held-out episodes, 238 held-out windows; salt `plantest2-v1`). No interval, as the card says: differences under 0.05 carry no weight. MEASURED: `decinfo/runs/*.json` (one file per rung and tag). Card: `../PREREG_V8F_DECINFO.md` (prefix sha256 `7e2e8aa6...`). Choices: `IMPL_CHOICES_DECINFO.md`.

## Controls (the read is INVALID if one fails)
* 1 shuffled targets (at or below 0.05): 104 of 104 reader rows pass. 2 planted signal (at or above 0.80, zeroed at or below 0.02): 23 of 26 rung/m inputs pass. 3 split: asserted in code and by `test_decinfo_controls.py` (see `TEST_DECINFO.log`). 4 reproduction of the PLANTEST3 gate within 0.01: 4 of 4. 5 chain identity bit for bit: 2 of 2.
* Violations: control 2: R4 M2b_s0 m=4 planted 0.332 zeroed -0.038; control 2: R4 M2b_s1 m=4 planted 0.568 zeroed -0.004; control 2: R6 encoder m=4 planted 0.743 zeroed -0.046
* Verdict on validity: **INVALID or incomplete (see violations / missing units)**

## Speed (m/s): held-out R2 per rung and reader (bar 0.70); shuffled-target control in brackets

| rung | input | tag | reader, m | R2 [shuffled] | fit samples | held-out samples | fit / held-out episodes |
|---|---|---|---|---|---|---|---|
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=1 | +0.271 [-0.318] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=1 | +0.420 [-0.194] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=4 | +0.317 [-0.567] | 9672 | 3094 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=4 | +0.406 [-0.086] | 9672 | 3094 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=1 | +0.314 [-0.124] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=1 | +0.387 [-0.198] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=4 | +0.299 [-0.117] | 9672 | 3094 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=4 | +0.355 [-0.020] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=1 | +0.408 [-0.232] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=1 | +0.353 [-0.192] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=4 | +0.480 [-0.501] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=4 | +0.445 [-0.040] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=1 | +0.390 [-0.218] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=1 | +0.421 [-0.124] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=4 | +0.437 [-0.337] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=4 | +0.476 [-0.043] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=1 | +0.369 [-0.231] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=1 | +0.376 [-0.085] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=4 | +0.412 [-0.415] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=4 | +0.383 [-0.175] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=1 | +0.173 [-0.132] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=1 | +0.409 [-0.261] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=4 | +0.146 [-0.125] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=4 | +0.445 [-0.075] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=1 | +0.364 [-0.237] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=1 | +0.386 [-0.264] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=4 | +0.416 [-0.471] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=4 | +0.355 [-0.214] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | ridge, m=1 | +0.119 [-0.137] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | mlp, m=1 | +0.418 [-0.238] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | ridge, m=4 | +0.062 [-0.155] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | mlp, m=4 | +0.444 [-0.071] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=1 | +0.473 [-0.274] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=1 | +0.245 [-0.093] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=4 | +0.498 [-0.956] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=4 | +0.346 [-0.169] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=1 | +0.365 [-0.125] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=1 | +0.232 [-0.138] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=4 | +0.269 [-0.636] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=4 | +0.391 [-0.315] | 9672 | 3094 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s0 | ridge, m=1 | +0.179 [-1.493] | 11904 | 3808 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s0 | conv, m=1 | +0.428 [+0.040] seeds +0.411, +0.441, +0.433 | 11904 | 3808 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s1 | ridge, m=1 | +0.153 [-1.396] | 11904 | 3808 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s1 | conv, m=1 | +0.397 [-0.039] seeds +0.409, +0.377, +0.405 | 11904 | 3808 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s0 | ridge, m=4 | +0.263 [-6.287] | 9672 | 3094 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s0 | conv, m=4 | +0.450 [+0.030] seeds +0.485, +0.443, +0.421 | 9672 | 3094 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s1 | ridge, m=4 | +0.197 [-4.079] | 9672 | 3094 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s1 | conv, m=4 | +0.464 [-0.103] seeds +0.396, +0.501, +0.496 | 9672 | 3094 | 53 / 17 |
| R5 | encoder grid, 1 frame (PCA 32) | encoder (added after the freeze) | ridge, m=1 | +0.224 [-0.934] | 11904 | 3808 | 53 / 17 |
| R5 | encoder grid, 1 frame (PCA 32) | encoder (added after the freeze) | conv, m=1 | +0.282 [-0.117] seeds +0.203, +0.300, +0.343 | 11904 | 3808 | 53 / 17 |
| R6 | encoder grid, 4 frames (PCA 32) | encoder (added after the freeze) | ridge, m=4 | +0.221 [-2.616] | 9672 | 3094 | 53 / 17 |
| R6 | encoder grid, 4 frames (PCA 32) | encoder (added after the freeze) | conv, m=4 | +0.355 [-0.149] seeds +0.416, +0.308, +0.340 | 9672 | 3094 | 53 / 17 |

## Yaw rate (rad/s): held-out R2 per rung and reader (bar 0.50); shuffled-target control in brackets

| rung | input | tag | reader, m | R2 [shuffled] | fit samples | held-out samples | fit / held-out episodes |
|---|---|---|---|---|---|---|---|
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=1 | -0.122 [-0.270] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=1 | +0.015 [-0.186] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=4 | -0.583 [-0.461] | 9672 | 3094 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=4 | +0.423 [-0.410] | 9672 | 3094 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=1 | +0.268 [-0.100] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=1 | +0.208 [-0.204] | 11904 | 3808 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=4 | -0.104 [-0.239] | 9672 | 3094 | 53 / 17 |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=4 | +0.001 [-0.528] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=1 | -0.198 [-0.322] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=1 | +0.398 [-0.352] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=4 | -0.675 [-0.630] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=4 | +0.418 [-0.249] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=1 | -0.055 [-0.326] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=1 | +0.427 [-0.193] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=4 | -0.476 [-0.608] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=4 | +0.488 [-0.394] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=1 | -0.186 [-0.339] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=1 | +0.350 [-0.097] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=4 | -0.625 [-0.653] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=4 | +0.523 [-0.431] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=1 | +0.253 [-0.198] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=1 | +0.068 [-0.316] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=4 | +0.123 [-0.433] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=4 | +0.079 [-0.364] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=1 | -0.157 [-0.351] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=1 | +0.388 [-0.093] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=4 | -0.531 [-0.663] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=4 | +0.527 [-0.417] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | ridge, m=1 | +0.224 [-0.164] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | mlp, m=1 | -0.051 [-0.269] | 11904 | 3808 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | ridge, m=4 | +0.086 [-0.327] | 9672 | 3094 | 53 / 17 |
| R1 | ladder z1 (128-D) | S35_s1 (added after the freeze) | mlp, m=4 | +0.166 [-0.314] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=1 | -0.551 [-0.455] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=1 | +0.417 [-0.167] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=4 | -1.159 [-0.822] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=4 | +0.502 [-0.220] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=1 | -0.646 [-0.408] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=1 | +0.329 [-0.225] | 11904 | 3808 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=4 | -1.647 [-0.784] | 9672 | 3094 | 53 / 17 |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=4 | +0.205 [-0.277] | 9672 | 3094 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s0 | ridge, m=1 | -0.796 [-0.297] | 11904 | 3808 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s0 | conv, m=1 | +0.377 [-0.221] seeds +0.427, +0.407, +0.298 | 11904 | 3808 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s1 | ridge, m=1 | -0.877 [-0.428] | 11904 | 3808 | 53 / 17 |
| R3 | adapter grid 4x10, 1 frame (PCA 32) | M2b_s1 | conv, m=1 | +0.442 [-0.222] seeds +0.521, +0.548, +0.257 | 11904 | 3808 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s0 | ridge, m=4 | -0.958 [-3.485] | 9672 | 3094 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s0 | conv, m=4 | +0.503 [-0.222] seeds +0.397, +0.583, +0.529 | 9672 | 3094 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s1 | ridge, m=4 | -0.608 [-2.734] | 9672 | 3094 | 53 / 17 |
| R4 | adapter grid 4x10, 4 frames (PCA 32) | M2b_s1 | conv, m=4 | +0.418 [-0.276] seeds +0.330, +0.454, +0.470 | 9672 | 3094 | 53 / 17 |
| R5 | encoder grid, 1 frame (PCA 32) | encoder (added after the freeze) | ridge, m=1 | -1.152 [-0.501] | 11904 | 3808 | 53 / 17 |
| R5 | encoder grid, 1 frame (PCA 32) | encoder (added after the freeze) | conv, m=1 | +0.148 [-0.133] seeds +0.262, +0.110, +0.071 | 11904 | 3808 | 53 / 17 |
| R6 | encoder grid, 4 frames (PCA 32) | encoder (added after the freeze) | ridge, m=4 | -1.097 [-1.165] | 9672 | 3094 | 53 / 17 |
| R6 | encoder grid, 4 frames (PCA 32) | encoder (added after the freeze) | conv, m=4 | +0.291 [-0.208] seeds +0.218, +0.325, +0.329 | 9672 | 3094 | 53 / 17 |

## Deciding quantity: best held-out R2 over the rung's readers (a maximum over a small fixed set, optimistic)

| rung | target | best per tag (reader) | bar | clears |
|---|---|---|---|---|
| R0 | speed | M2b_s0: +0.420 (mlp m=1); M2b_s1: +0.387 (mlp m=1) | 0.70 | False |
| R1 | speed | C0_s0: +0.480 (ridge m=4); R35H_s0: +0.476 (mlp m=4); R35_s0: +0.412 (ridge m=4); R35_s1: +0.445 (mlp m=4); S35_s0: +0.416 (ridge m=4); S35_s1: +0.444 (mlp m=4) | 0.70 | False |
| R2 | speed | M2b_s0: +0.498 (ridge m=4); M2b_s1: +0.391 (mlp m=4) | 0.70 | False |
| R3 | speed | M2b_s0: +0.428 (conv m=1); M2b_s1: +0.397 (conv m=1) | 0.70 | False |
| R4 | speed | M2b_s0: +0.450 (conv m=4); M2b_s1: +0.464 (conv m=4) | 0.70 | False |
| R5 | speed | encoder: +0.282 (conv m=1) | 0.70 | False |
| R6 | speed | encoder: +0.355 (conv m=4) | 0.70 | False |
| R0 | yaw | M2b_s0: +0.423 (mlp m=4); M2b_s1: +0.268 (ridge m=1) | 0.50 | False |
| R1 | yaw | C0_s0: +0.418 (mlp m=4); R35H_s0: +0.488 (mlp m=4); R35_s0: +0.523 (mlp m=4); R35_s1: +0.253 (ridge m=1); S35_s0: +0.527 (mlp m=4); S35_s1: +0.224 (ridge m=1) | 0.50 | True |
| R2 | yaw | M2b_s0: +0.502 (mlp m=4); M2b_s1: +0.329 (mlp m=1) | 0.50 | False |
| R3 | yaw | M2b_s0: +0.377 (conv m=1); M2b_s1: +0.442 (conv m=1) | 0.50 | False |
| R4 | yaw | M2b_s0: +0.503 (conv m=4); M2b_s1: +0.418 (conv m=4) | 0.50 | False |
| R5 | yaw | encoder: +0.148 (conv m=1) | 0.50 | False |
| R6 | yaw | encoder: +0.291 (conv m=4) | 0.50 | False |

## Committed outcomes, evaluated mechanically (per target)

* **speed: NOT-READ-FROM-VISION.** Committed trigger: the same probe at full n on box 2 (4,572 train clips cached only there; one job, after the ladder), before any conclusion about the encoder.
* **yaw: LADDER-READS.** Committed trigger: PLANTEST3 rerun on that checkpoint's z1 and readout.
* **Every outcome:** a card for measured ego state as part of the world-model state is written (speed and yaw rate are measured quantities, D-PI-YAWRATE-1, R-EGO); that arm does not depend on this read.

## Limits
* **n.** 53 fit episodes (744 windows; 11,904 one-frame samples) and 17 held-out episodes (238 windows; 3,808 one-frame samples; 3,094 four-frame samples). R2 has no interval here; one-seed readers differ by seed noise that this card does not measure, so differences under 0.05 carry no weight. Four frames reduce the sample count per window from 16 to 13.
* A rung's deciding value is a maximum over readers (ridge, MLP or conv, one or four frames), an optimistic read. A readout bound at this n does not show absence of the information (`A NEGATIVE FROM A LINEAR PROBE IS NOT A NEGATIVE ABOUT LEARNABILITY`).
* Grid tokens are read as a 4 x 10 grid with token index = row * 10 + column (assumed from the `pool4x10` cache name). Tier T0: never a driving result. STRATEGIC family: PAUSED (PI item 25); LONGITUDINAL, LATERAL, TACTICAL have no metric here for a stated reason (a readout R2 only).
