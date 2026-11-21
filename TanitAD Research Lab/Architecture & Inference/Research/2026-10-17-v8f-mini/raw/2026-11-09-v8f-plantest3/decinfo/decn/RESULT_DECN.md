# RESULT_DECN: is the weak speed and yaw-rate readout limited by the 53 fit episodes?
Tier **T0**, a readout diagnostic on the PLANTEST3 TUNE inner split: the same 17 held-out episodes (238 windows, 16 frames) at every N; fit sets N53 (53 TUNE fit episodes), N194, N1053 (train clips streamed read-only from box 2). No interval, as the card says: differences under 0.05 carry no weight. MEASURED: `decinfo/decn/runs/*.json` (one file per rung, tag and N). Card: `../../PREREG_V8F_DECN.md` (sha256 cfa5bb1fd...). Choices: `IMPL_CHOICES_DECN.md`.

## Validity first (the read is INVALID if one control fails)
* **Verdict on validity: INVALID or incomplete (see violations).** Stream: 1000 of 1,000 train clips with latents. N1053 complete.
* Control 1, target tie (cache `v` of the eval clips against the eval target on the 982 TUNE windows): R2 1.00000, mean absolute difference 0.00000 m/s over 15712 values (bars: R2 at or above 0.99, difference at or below 0.10): pass.
* Control 2, reproduction of DECINFO's R0 ridge m=1 at N53 (within 0.01, speed and yaw, both seeds): 4 of 4 pass.
* Control 3, shuffled targets (at or below 0.05): 206 of 216 reader rows pass (every N, input, reader, m, target).
* Control 4, planted signal (at or above 0.80, zeroed at or below 0.02): 54 of 54 unit/m inputs pass (the N1053 units included).
* Control 5, split: held-out block digest identical across all 27 units: True; 238 held-out windows, 17 episodes; guards asserted in code and in `test_decn_controls.py` (`TEST_DECN.log`).
* Control 6, clip integrity (md5 against box 2's manifest, mismatch stops the run): no mismatch recorded.
* Violations: control 3: R0 M2b_s0 N1053 mlp m=1 speed shuffled R2 +0.058; control 3: R0 M2b_s0 N1053 mlp m=4 speed shuffled R2 +0.102; control 3: R0 M2b_s0 N194 mlp m=1 yaw shuffled R2 +0.090; control 3: R0 M2b_s1 N194 mlp m=1 yaw shuffled R2 +0.112; control 3: R1 R35_s0 N1053 mlp m=4 speed shuffled R2 +0.061; control 3: R1 R35_s0 N194 mlp m=1 yaw shuffled R2 +0.122; control 3: R1 R35_s1 N194 mlp m=1 yaw shuffled R2 +0.158; control 3: R1 S35_s0 N194 mlp m=1 yaw shuffled R2 +0.101; control 3: R2 M2b_s0 N1053 mlp m=1 speed shuffled R2 +0.052; control 3: R2 M2b_s1 N194 mlp m=4 yaw shuffled R2 +0.068
* Logged, deciding nothing: train-clip cache `v` against the native-track speed at the same frames, maximum absolute difference over the streamed clips: 0.000000.

## Sample counts (episodes, windows, samples)

| rung | tag | N | fit episodes | fit windows | fit samples m=1 / m=4 (speed) | held-out episodes / windows | held-out samples m=1 / m=4 |
|---|---|---|---|---|---|---|---|
| R0 | M2b_s0 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R0 | M2b_s0 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R0 | M2b_s0 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R0 | M2b_s1 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R0 | M2b_s1 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R0 | M2b_s1 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R1 | C0_s0 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R1 | C0_s0 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R1 | C0_s0 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R1 | R35H_s0 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R1 | R35H_s0 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R1 | R35H_s0 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R1 | R35_s0 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R1 | R35_s0 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R1 | R35_s0 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R1 | R35_s1 (added after the freeze) | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R1 | R35_s1 (added after the freeze) | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R1 | R35_s1 (added after the freeze) | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R1 | S35_s0 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R1 | S35_s0 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R1 | S35_s0 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R2 | M2b_s0 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R2 | M2b_s0 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R2 | M2b_s0 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |
| R2 | M2b_s1 | N53 | 53 | 744 | 11904 / 9672 | 17 / 238 | 3808 / 3094 |
| R2 | M2b_s1 | N194 | 194 | 2577 | 41232 / 33501 | 17 / 238 | 3808 / 3094 |
| R2 | M2b_s1 | N1053 | 1053 | 13744 | 219904 / 178672 | 17 / 238 | 3808 / 3094 |

## Speed (m/s): held-out R2 per input, reader and N (bar 0.70); shuffled-target control in brackets

| rung | input | tag | reader, m | N53 | N194 | N1053 |
|---|---|---|---|---|---|---|
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=1 | +0.271 [-0.318] | +0.606 [-0.211] | +0.680 [-0.004] |
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=4 | +0.317 [-0.567] | +0.600 [-0.254] | +0.695 [-0.005] |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=1 | +0.420 [-0.194] (epochs 17/120) | +0.535 [-0.260] (epochs 29/60) | +0.634 [+0.058] (epochs 13/60) |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=4 | +0.406 [-0.086] (epochs 15/120) | +0.471 [-0.433] (epochs 47/60) | +0.650 [+0.102] (epochs 13/60) |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=1 | +0.314 [-0.124] | +0.622 [-0.269] | +0.647 [-0.026] |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=4 | +0.299 [-0.117] | +0.635 [-0.333] | +0.686 [-0.034] |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=1 | +0.387 [-0.198] (epochs 14/120) | +0.539 [-0.210] (epochs 25/60) | +0.676 [+0.048] (epochs 13/60) |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=4 | +0.355 [-0.020] (epochs 14/120) | +0.548 [-0.114] (epochs 14/60) | +0.623 [+0.024] (epochs 13/60) |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=1 | +0.408 [-0.232] | +0.621 [-0.261] | +0.690 [-0.007] |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=4 | +0.480 [-0.501] | +0.616 [-0.299] | +0.713 [-0.003] |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=1 | +0.353 [-0.192] (epochs 14/120) | +0.524 [-0.328] (epochs 29/60) | +0.667 [+0.020] (epochs 13/60) |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=4 | +0.445 [-0.040] (epochs 14/120) | +0.556 [-0.221] (epochs 29/60) | +0.669 [-0.084] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=1 | +0.390 [-0.218] | +0.636 [-0.192] | +0.700 [-0.017] |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=4 | +0.437 [-0.337] | +0.658 [-0.173] | +0.720 [-0.029] |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=1 | +0.421 [-0.124] (epochs 20/120) | +0.559 [-0.471] (epochs 21/60) | +0.703 [-0.030] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=4 | +0.476 [-0.043] (epochs 23/120) | +0.567 [-0.324] (epochs 45/60) | +0.710 [-0.055] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=1 | +0.369 [-0.231] | +0.646 [-0.156] | +0.706 [-0.004] |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=4 | +0.412 [-0.415] | +0.652 [-0.115] | +0.723 [-0.008] |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=1 | +0.376 [-0.085] (epochs 14/120) | +0.489 [-0.431] (epochs 25/60) | +0.674 [+0.002] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=4 | +0.383 [-0.175] (epochs 15/120) | +0.478 [-0.180] (epochs 29/60) | +0.662 [+0.061] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=1 | +0.173 [-0.132] | +0.637 [-0.181] | +0.686 [-0.050] |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=4 | +0.146 [-0.125] | +0.647 [-0.261] | +0.718 [-0.073] |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=1 | +0.409 [-0.261] (epochs 14/120) | +0.534 [-0.234] (epochs 15/60) | +0.699 [+0.011] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=4 | +0.445 [-0.075] (epochs 38/120) | +0.536 [-0.164] (epochs 16/60) | +0.701 [+0.034] (epochs 13/60) |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=1 | +0.364 [-0.237] | +0.642 [-0.155] | +0.694 [-0.005] |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=4 | +0.416 [-0.471] | +0.653 [-0.116] | +0.713 [-0.011] |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=1 | +0.386 [-0.264] (epochs 17/120) | +0.532 [-0.425] (epochs 21/60) | +0.681 [-0.007] (epochs 13/60) |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=4 | +0.355 [-0.214] (epochs 15/120) | +0.508 [-0.181] (epochs 44/60) | +0.637 [-0.030] (epochs 13/60) |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=1 | +0.473 [-0.274] | +0.558 [-0.369] | +0.680 [-0.032] |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=4 | +0.498 [-0.956] | +0.480 [-0.481] | +0.676 [-0.051] |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=1 | +0.245 [-0.093] (epochs 13/120) | +0.618 [-0.327] (epochs 15/60) | +0.713 [+0.052] (epochs 13/60) |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=4 | +0.346 [-0.169] (epochs 13/120) | +0.547 [-0.384] (epochs 13/60) | +0.721 [-0.024] (epochs 13/60) |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=1 | +0.365 [-0.125] | +0.648 [-0.407] | +0.708 [-0.065] |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=4 | +0.269 [-0.636] | +0.607 [-0.550] | +0.736 [-0.077] |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=1 | +0.232 [-0.138] (epochs 14/120) | +0.524 [-0.222] (epochs 14/60) | +0.686 [+0.043] (epochs 13/60) |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=4 | +0.391 [-0.315] (epochs 33/120) | +0.507 [-0.238] (epochs 13/60) | +0.713 [-0.009] (epochs 13/60) |

## Yaw rate (rad/s): held-out R2 per input, reader and N (bar 0.50); shuffled-target control in brackets

| rung | input | tag | reader, m | N53 | N194 | N1053 |
|---|---|---|---|---|---|---|
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=1 | -0.122 [-0.270] | -0.257 [-0.046] | -0.043 [-0.144] |
| R0 | M2b z1 (128-D) | M2b_s0 | ridge, m=4 | -0.583 [-0.461] | -0.341 [-0.074] | -0.058 [-0.172] |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=1 | +0.015 [-0.186] (epochs 15/120) | -0.413 [+0.090] (epochs 32/60) | +0.437 [-0.174] (epochs 14/60) |
| R0 | M2b z1 (128-D) | M2b_s0 | mlp, m=4 | +0.423 [-0.410] (epochs 19/120) | -0.452 [-0.119] (epochs 13/60) | +0.570 [-0.193] (epochs 20/60) |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=1 | +0.268 [-0.100] | -0.059 [-0.112] | +0.105 [-0.159] |
| R0 | M2b z1 (128-D) | M2b_s1 | ridge, m=4 | -0.104 [-0.239] | -0.104 [-0.243] | +0.109 [-0.174] |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=1 | +0.208 [-0.204] (epochs 23/120) | -0.192 [+0.112] (epochs 14/60) | +0.402 [-0.211] (epochs 13/60) |
| R0 | M2b z1 (128-D) | M2b_s1 | mlp, m=4 | +0.001 [-0.528] (epochs 19/120) | -0.783 [-0.090] (epochs 15/60) | +0.441 [-0.384] (epochs 13/60) |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=1 | -0.198 [-0.322] | -0.224 [-0.110] | -0.082 [-0.139] |
| R1 | ladder z1 (128-D) | C0_s0 | ridge, m=4 | -0.675 [-0.630] | -0.233 [-0.147] | -0.129 [-0.150] |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=1 | +0.398 [-0.352] (epochs 21/120) | -0.294 [-0.145] (epochs 16/60) | +0.414 [-0.142] (epochs 15/60) |
| R1 | ladder z1 (128-D) | C0_s0 | mlp, m=4 | +0.418 [-0.249] (epochs 30/120) | -0.151 [-0.237] (epochs 22/60) | +0.528 [-0.330] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=1 | -0.055 [-0.326] | -0.286 [-0.144] | -0.056 [-0.136] |
| R1 | ladder z1 (128-D) | R35H_s0 | ridge, m=4 | -0.476 [-0.608] | -0.312 [-0.180] | -0.069 [-0.131] |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=1 | +0.427 [-0.193] (epochs 15/120) | -0.233 [-0.087] (epochs 22/60) | +0.552 [-0.121] (epochs 15/60) |
| R1 | ladder z1 (128-D) | R35H_s0 | mlp, m=4 | +0.488 [-0.394] (epochs 19/120) | -0.386 [-0.155] (epochs 20/60) | +0.606 [-0.223] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=1 | -0.186 [-0.339] | -0.328 [-0.098] | -0.060 [-0.142] |
| R1 | ladder z1 (128-D) | R35_s0 | ridge, m=4 | -0.625 [-0.653] | -0.406 [-0.109] | -0.090 [-0.148] |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=1 | +0.350 [-0.097] (epochs 21/120) | -0.204 [+0.122] (epochs 20/60) | +0.407 [-0.138] (epochs 14/60) |
| R1 | ladder z1 (128-D) | R35_s0 | mlp, m=4 | +0.523 [-0.431] (epochs 19/120) | -0.523 [-0.186] (epochs 29/60) | +0.554 [-0.141] (epochs 13/60) |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=1 | +0.253 [-0.198] | -0.072 [-0.076] | +0.072 [-0.178] |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | ridge, m=4 | +0.123 [-0.433] | -0.112 [-0.178] | +0.057 [-0.207] |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=1 | +0.068 [-0.316] (epochs 21/120) | +0.156 [+0.158] (epochs 14/60) | +0.500 [-0.143] (epochs 14/60) |
| R1 | ladder z1 (128-D) | R35_s1 (added after the freeze) | mlp, m=4 | +0.079 [-0.364] (epochs 19/120) | -0.222 [-0.015] (epochs 16/60) | +0.388 [-0.375] (epochs 13/60) |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=1 | -0.157 [-0.351] | -0.392 [-0.111] | -0.074 [-0.140] |
| R1 | ladder z1 (128-D) | S35_s0 | ridge, m=4 | -0.531 [-0.663] | -0.472 [-0.137] | -0.104 [-0.151] |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=1 | +0.388 [-0.093] (epochs 19/120) | -0.278 [+0.101] (epochs 36/60) | +0.373 [-0.131] (epochs 13/60) |
| R1 | ladder z1 (128-D) | S35_s0 | mlp, m=4 | +0.527 [-0.417] (epochs 25/120) | -0.455 [-0.122] (epochs 21/60) | +0.535 [-0.114] (epochs 17/60) |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=1 | -0.551 [-0.455] | -0.283 [-0.107] | -0.166 [-0.214] |
| R2 | pooled adapter token (256-D) | M2b_s0 | ridge, m=4 | -1.159 [-0.822] | -0.249 [-0.238] | -0.220 [-0.262] |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=1 | +0.417 [-0.167] (epochs 16/120) | -0.370 [+0.039] (epochs 17/60) | +0.223 [-0.139] (epochs 13/60) |
| R2 | pooled adapter token (256-D) | M2b_s0 | mlp, m=4 | +0.502 [-0.220] (epochs 27/120) | -0.738 [-0.197] (epochs 21/60) | +0.451 [-0.114] (epochs 15/60) |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=1 | -0.646 [-0.408] | -0.327 [-0.232] | -0.057 [-0.237] |
| R2 | pooled adapter token (256-D) | M2b_s1 | ridge, m=4 | -1.647 [-0.784] | -0.464 [-0.477] | -0.110 [-0.250] |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=1 | +0.329 [-0.225] (epochs 19/120) | -0.108 [-0.163] (epochs 14/60) | +0.431 [-0.241] (epochs 13/60) |
| R2 | pooled adapter token (256-D) | M2b_s1 | mlp, m=4 | +0.205 [-0.277] (epochs 26/120) | -0.561 [+0.068] (epochs 14/60) | +0.507 [-0.264] (epochs 35/60) |

## Deciding quantity: best held-out R2 over the readers per input, N and target (a maximum over four readers, optimistic)

| target | input | tag | best N53 | best N194 | best N1053 | rise N1053 - N53 (cell value) | rise, same reader (max over readers) |
|---|---|---|---|---|---|---|---|
| speed | R0 | M2b_s0 | +0.420 (mlp m=1) | +0.606 (ridge m=1) | +0.695 (ridge m=4) | +0.274 | +0.409 (ridge m=1) |
| speed | R0 | M2b_s1 | +0.387 (mlp m=1) | +0.635 (ridge m=4) | +0.686 (ridge m=4) | +0.299 | +0.387 (ridge m=4) |
| speed | R1 | C0_s0 | +0.480 (ridge m=4) | +0.621 (ridge m=1) | +0.713 (ridge m=4) | +0.233 | +0.313 (mlp m=1) |
| speed | R1 | R35H_s0 | +0.476 (mlp m=4) | +0.658 (ridge m=4) | +0.720 (ridge m=4) | +0.244 | +0.309 (ridge m=1) |
| speed | R1 | R35_s0 | +0.412 (ridge m=4) | +0.652 (ridge m=4) | +0.723 (ridge m=4) | +0.311 | +0.337 (ridge m=1) |
| speed | R1 | R35_s1 | +0.445 (mlp m=4) | +0.647 (ridge m=4) | +0.718 (ridge m=4) | +0.273 | +0.572 (ridge m=4) |
| speed | R1 | S35_s0 | +0.416 (ridge m=4) | +0.653 (ridge m=4) | +0.713 (ridge m=4) | +0.297 | +0.330 (ridge m=1) |
| speed | R2 | M2b_s0 | +0.498 (ridge m=4) | +0.618 (mlp m=1) | +0.721 (mlp m=4) | +0.223 | +0.469 (mlp m=1) |
| speed | R2 | M2b_s1 | +0.391 (mlp m=4) | +0.648 (ridge m=1) | +0.736 (ridge m=4) | +0.345 | +0.467 (ridge m=4) |
| yaw | R0 | M2b_s0 | +0.423 (mlp m=4) | -0.257 (ridge m=1) | +0.570 (mlp m=4) | +0.147 | +0.525 (ridge m=4) |
| yaw | R0 | M2b_s1 | +0.268 (ridge m=1) | -0.059 (ridge m=1) | +0.441 (mlp m=4) | +0.173 | +0.440 (mlp m=4) |
| yaw | R1 | C0_s0 | +0.418 (mlp m=4) | -0.151 (mlp m=4) | +0.528 (mlp m=4) | +0.111 | +0.546 (ridge m=4) |
| yaw | R1 | R35H_s0 | +0.488 (mlp m=4) | -0.233 (mlp m=1) | +0.606 (mlp m=4) | +0.117 | +0.408 (ridge m=4) |
| yaw | R1 | R35_s0 | +0.523 (mlp m=4) | -0.204 (mlp m=1) | +0.554 (mlp m=4) | +0.031 | +0.535 (ridge m=4) |
| yaw | R1 | R35_s1 | +0.253 (ridge m=1) | +0.156 (mlp m=1) | +0.500 (mlp m=1) | +0.246 | +0.432 (mlp m=1) |
| yaw | R1 | S35_s0 | +0.527 (mlp m=4) | -0.278 (mlp m=1) | +0.535 (mlp m=4) | +0.008 | +0.427 (ridge m=4) |
| yaw | R2 | M2b_s0 | +0.502 (mlp m=4) | -0.249 (ridge m=4) | +0.451 (mlp m=4) | -0.051 | +0.938 (ridge m=4) |
| yaw | R2 | M2b_s1 | +0.329 (mlp m=1) | -0.108 (mlp m=1) | +0.507 (mlp m=4) | +0.178 | +1.537 (ridge m=4) |

## Committed outcomes, evaluated mechanically (per target; rules in `IMPL_CHOICES_DECN.md` section 7)

* **speed: UNDETERMINED** (controls not all valid). The tables above are the result. If a committed outcome held on the cells present it would be: R0: M2b_s0 +0.695 (rise +0.274), M2b_s1 +0.686 (rise +0.299); R2: M2b_s0 +0.721 (rise +0.223), M2b_s1 +0.736 (rise +0.345)
* **yaw: UNDETERMINED** (controls not all valid). The tables above are the result. If a committed outcome held on the cells present it would be: R0: M2b_s0 +0.570 (rise +0.147), M2b_s1 +0.441 (rise +0.173); R2: M2b_s0 +0.451 (rise -0.051), M2b_s1 +0.507 (rise +0.178)

## Limits
* **n.** N53 is 53 fit episodes (744 windows); N194 is 194; N1053 is 1,053. The held-out set is the same 17 episodes (238 windows; 3,808 one-frame, 3,094 four-frame samples) at every N. R2 has no interval here; one-seed readers differ by seed noise that this card does not measure, so differences under 0.05 carry no weight.
* A cell value is a maximum over four readers (ridge and MLP, one and four frames), an optimistic read. MLP epoch cap: 120 at N53, 60 at N194 and N1053; `epochs x/cap` is printed per cell, and a row at its cap may be under-trained. A readout bound at this n and reader family does not show absence of the information.
* Train windows are the 13 of 17 per clip whose 16 frames are all real. Train-clip speed is the cache `v`; its tie to the eval target is control 1. Ladder rows (R1) are read alongside and decide nothing; tags outside the frozen four are marked as added.
* Tier T0: a readout R2 only, never a driving result. LONGITUDINAL, LATERAL, TACTICAL: no metric here, for that reason; STRATEGIC: PAUSED (PI item 25), not a pass.
