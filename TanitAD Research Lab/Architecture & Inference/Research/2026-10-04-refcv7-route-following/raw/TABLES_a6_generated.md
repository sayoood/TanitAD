## controls

| control | measured | verdict |
|---|---|---|
| seed-0 / seed-1 / window-list windows identical | n=4634 | PASS |
| selection-time GT == captured GT | max abs diff 0.0e+00 m, validity mismatches 0 (n=4634) | PASS |
| GT-turn windows captured == selected | 2317 == 2317 | PASS |
| overlap identity seed0_vs_eval_s0g (214 windows) | fan max abs diff 0.0e+00, s_e9 0.0e+00, sel_idx mismatches 0 | PASS |
| overlap identity seed1_vs_eval_s1 (214 windows) | fan max abs diff 0.0e+00, s_e9 0.0e+00, sel_idx mismatches 0 | PASS |
| C1_capture_identity_a6_s0 | traj-vs-fan 0.0, E9 argmax mismatches 0, core 0, forwards/window 1 | PASS |
| C1_capture_identity_a6_s1 | traj-vs-fan 0.0, E9 argmax mismatches 0, core 0, forwards/window 1 | PASS |
| K0_join_bank_nav_eq_record_clip_token_a6_s0 | 4634/4634 | PASS |
| K0_join_bank_nav_eq_record_clip_token_a6_s1 | 4634/4634 | PASS |
| K1 clock vs replay bank | max abs diff 5.00e-05 s over 2059 windows | PASS |
| K2 nav_tl == clip token at t_rel ~ 0 (dense windows) | 11/11 | PASS |
| K2 supplementary (one t per eval clip) | 17/17 | PASS |

## capture

* 4634 windows over 139 episodes: **2317 GT-turn windows (every one of the 23772 eval windows with |terminal heading| >= 30 deg)** + 2317 seeded (seed 0) random others. Totals over all eval windows {'turnL': 849, 'turnR': 1468, 'straight': 12908, 'gentle': 2403, 'unclassified': 6144}; selected {'turnL': 849, 'turnR': 1468, 'straight': 1347, 'gentle': 276, 'unclassified': 694}.
* seed0: GPU forward 1804.8 s, loop 1945.7 s, wall 2072.8 s, peak 1.338 GiB (NVIDIA Thor)
* seed1: GPU forward 1795.1 s, loop 1934.1 s, wall 2061.0 s, peak 1.338 GiB (NVIDIA Thor)

## window counts (dense set)

| class | n | nav_ann active | nav_tl active | clip token active | nav_ann differs from nav_tl | nav_ann == GT dir | nav_ann opposite of GT |
|---|---|---|---|---|---|---|---|
| GT-turn | 2317 | 1021 | 1129 | 1125 | 108 | 1007 | 14 |
| GT-straight | 1347 | 21 | 22 | 448 | 1 | 0 | 0 |
| gentle | 276 | 57 | 61 | 115 | 4 | 54 | 2 |
| unclassified | 694 | 87 | 91 | 253 | 4 | 0 | 0 |
| every window | 4634 | 1186 | 1303 | 1941 | 117 | 1061 | 16 |

**THE REGISTERED SCORING: bar on the dense capture (paired episode-cluster bootstrap vs V0, B = 2000; seed 1 = sampler replicate)**

| arm | picks changed | turn dADE [CI] | turn dir-correct d [CI] | straight dADE [CI] | all dADE [CI] | seed 1: turn dADE [CI] | seed 1: turn dir-correct d [CI] | criteria 1/2/3/4 | bar |
|---|---|---|---|---|---|---|---|---|---|
| **T3a (REPORTED)** | 0.041 | -0.131 [-0.207, -0.062] | +0.064 [+0.034, +0.104] | +0.002 [-0.008, +0.012] | -0.076 [-0.116, -0.036] | -0.127 [-0.203, -0.057] | +0.065 [+0.032, +0.107] | Y/Y/Y/Y | **CLEARS** |
| T3 (foil: all entries) | 0.044 | -0.159 [-0.254, -0.080] | +0.070 [+0.040, +0.110] | +0.002 [-0.008, +0.012] | -0.092 [-0.144, -0.046] | -0.146 [-0.228, -0.072] | +0.070 [+0.037, +0.112] | Y/Y/Y/Y | **CLEARS** |
| T2a (hard filter, announced) | 0.072 | -0.241 [-0.461, -0.031] | +0.084 [+0.040, +0.135] | +0.009 [-0.000, +0.020] | -0.138 [-0.266, -0.015] | -0.239 [-0.453, -0.032] | +0.082 [+0.039, +0.132] | Y/Y/Y/Y | **CLEARS** |
| T3a-c CONTROL (derangement) | 0.021 | -0.011 [-0.085, +0.045] | -0.004 [-0.025, +0.015] | -0.000 [-0.008, +0.007] | -0.005 [-0.049, +0.030] | -0.005 [-0.058, +0.043] | -0.006 [-0.025, +0.009] | N/Y/Y/N | **FAILED** |
| B1t (label-side bound) | 0.092 | -0.486 [-0.748, -0.267] | +0.185 [+0.115, +0.270] | +0.000 [+0.000, +0.000] | -0.286 [-0.427, -0.152] | -0.478 [-0.727, -0.264] | +0.187 [+0.114, +0.274] | – | bound |
| ORACLE (label-side bound) | 0.676 | -2.072 [-2.448, -1.743] | +0.137 [+0.086, +0.198] | -0.946 [-1.117, -0.782] | -1.656 [-1.883, -1.420] | -2.045 [-2.414, -1.714] | +0.137 [+0.086, +0.197] | – | bound |

B1t fraction: dADE_turn(T3a) -0.1312 / B1t (recomputed on this capture) -0.4863 = **0.270** (per-draw ratio 95 % [0.15118415210501637, 0.4488345328476394]); against A5's registered -0.566: 0.232; T3 0.327, T2a 0.495.

**Reading rule: T3a PASSES, T3a-c FAILS** -- a deployable time-localised nav lifts route following at inference on refcv7 as trained (confirmed on denser windows of the SAME episodes); it ships as an opt-in inference rule and L2 enters refcv8

## four families (SPEC 2.1 format), dense set, sampler seed 0


**GT-turn windows (n=2317)**

| arm | ADE | FDE | LON |along| 6 s | LON along 6 s signed | LON speed MAE 0-2 s | LAT |cross| 6 s | LAT heading MAE 0-2 s deg | LAT curv MAE 0-2 s | LAT term. heading err deg | TAC dir correct | STR nav compliance (vs the CLIP token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 3.297 | 10.713 | 7.368 | 3.437 | 0.346 | 6.330 | 5.276 | 0.010 | 23.771 | 0.815 | 0.685 |
| T3a | 3.166 | 10.230 | 7.082 | 3.283 | 0.341 | 5.973 | 5.192 | 0.009 | 21.485 | 0.880 | 0.812 |
| T3 | 3.138 | 10.133 | 6.977 | 3.178 | 0.337 | 5.935 | 5.191 | 0.009 | 21.341 | 0.886 | 0.812 |
| T2a | 3.056 | 9.857 | 6.629 | 2.811 | 0.327 | 5.920 | 5.242 | 0.010 | 20.893 | 0.899 | 0.845 |
| T3a-c | 3.285 | 10.696 | 7.244 | 3.270 | 0.343 | 6.393 | 5.290 | 0.010 | 24.314 | 0.811 | 0.665 |
| B1t | 2.810 | 9.034 | 6.172 | 2.298 | 0.321 | 5.369 | 4.996 | 0.009 | 17.886 | 1.000 | 0.948 |
| ORACLE | 1.225 | 3.621 | 2.418 | 0.144 | 0.224 | 2.194 | 3.917 | 0.007 | 13.681 | 0.953 | 0.869 |

**GT-straight windows (n=1347)**

| arm | ADE | FDE | LON |along| 6 s | LON along 6 s signed | LON speed MAE 0-2 s | LAT |cross| 6 s | LAT heading MAE 0-2 s deg | LAT curv MAE 0-2 s | LAT term. heading err deg | TAC dir correct | STR nav compliance (vs the CLIP token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 1.678 | 5.353 | 4.995 | 0.224 | 0.252 | 1.048 | 0.455 | 0.002 | 2.289 | 0.963 | 0.029 |
| T3a | 1.680 | 5.359 | 4.997 | 0.225 | 0.252 | 1.052 | 0.457 | 0.002 | 2.303 | 0.960 | 0.038 |
| T3 | 1.680 | 5.359 | 4.997 | 0.225 | 0.252 | 1.052 | 0.457 | 0.002 | 2.303 | 0.960 | 0.038 |
| T2a | 1.687 | 5.377 | 5.000 | 0.167 | 0.253 | 1.084 | 0.462 | 0.002 | 2.400 | 0.955 | 0.054 |
| T3a-c | 1.678 | 5.358 | 4.981 | 0.235 | 0.251 | 1.068 | 0.461 | 0.002 | 2.380 | 0.958 | 0.036 |
| B1t | 1.678 | 5.353 | 4.995 | 0.224 | 0.252 | 1.048 | 0.455 | 0.002 | 2.289 | 0.963 | 0.029 |
| ORACLE | 0.733 | 2.228 | 1.882 | -0.250 | 0.158 | 0.812 | 0.461 | 0.002 | 2.490 | 0.941 | 0.029 |

**all classified windows (turn-enriched) (n=3940)**

| arm | ADE | FDE | LON |along| 6 s | LON along 6 s signed | LON speed MAE 0-2 s | LAT |cross| 6 s | LAT heading MAE 0-2 s deg | LAT curv MAE 0-2 s | LAT term. heading err deg | TAC dir correct | STR nav compliance (vs the CLIP token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 2.698 | 8.744 | 6.533 | 2.266 | 0.311 | 4.361 | 3.331 | 0.007 | 15.477 | 0.852 | 0.489 |
| T3a | 2.622 | 8.461 | 6.374 | 2.173 | 0.308 | 4.144 | 3.286 | 0.006 | 14.094 | 0.893 | 0.587 |
| T3 | 2.606 | 8.404 | 6.312 | 2.111 | 0.306 | 4.122 | 3.285 | 0.006 | 14.009 | 0.897 | 0.587 |
| T2a | 2.561 | 8.246 | 6.098 | 1.864 | 0.300 | 4.133 | 3.319 | 0.006 | 13.752 | 0.902 | 0.613 |
| T3a-c | 2.693 | 8.741 | 6.460 | 2.172 | 0.309 | 4.409 | 3.340 | 0.007 | 15.842 | 0.848 | 0.477 |
| B1t | 2.412 | 7.756 | 5.830 | 1.596 | 0.296 | 3.796 | 3.170 | 0.006 | 12.017 | 0.961 | 0.665 |
| ORACLE | 1.042 | 3.107 | 2.217 | -0.023 | 0.198 | 1.690 | 2.521 | 0.005 | 9.375 | 0.930 | 0.614 |

**TACTICAL (`four_families.tactical_from_trajectory`, 0-2 s, all dense windows)**

| arm | lateral accuracy / kappa | longitudinal accuracy / kappa |
|---|---|---|
| V0 | 0.9193 / 0.8511 | 0.7902 / 0.5205 |
| T3a | 0.9256 / 0.8627 | 0.7946 / 0.5312 |
| T3 | 0.9260 / 0.8635 | 0.7954 / 0.5337 |
| T2a | 0.9245 / 0.8606 | 0.8038 / 0.5609 |
| T3a-c | 0.9193 / 0.8511 | 0.7918 / 0.5242 |
| ORACLE | 0.9448 / 0.8997 | 0.9053 / 0.8001 |

*LONGITUDINAL distance keeping UNAVAILABLE (no lead tracks); STRATEGIC decision UNAVAILABLE (`--no-strategic`).*

**A5's EVAL-grid windows as a SUBSET of the dense capture (n=214; classes {'GT-turn': 107, 'GT-straight': 61, 'gentle': 11, 'unclassified': 35, 'every window': 214})**

| arm | picks changed | turn dADE [CI] | turn dir-correct d [CI] | straight dADE [CI] | all dADE [CI] | seed 1: turn dADE [CI] | seed 1: turn dir-correct d [CI] | criteria 1/2/3/4 | bar |
|---|---|---|---|---|---|---|---|---|---|
| **T3a (REPORTED)** | 0.041 | -0.107 [-0.214, -0.016] | +0.056 [+0.011, +0.114] | +0.027 [+0.000, +0.088] | -0.055 [-0.122, +0.000] | -0.097 [-0.197, -0.013] | +0.047 [+0.008, +0.101] | Y/Y/Y/Y | **CLEARS** |
| T3 (foil: all entries) | 0.044 | -0.159 [-0.312, -0.040] | +0.065 [+0.020, +0.124] | +0.027 [+0.000, +0.088] | -0.086 [-0.181, -0.011] | -0.143 [-0.286, -0.032] | +0.056 [+0.011, +0.113] | Y/Y/Y/Y | **CLEARS** |
| T2a (hard filter, announced) | 0.072 | -0.157 [-0.343, +0.034] | +0.075 [+0.019, +0.143] | +0.036 [+0.000, +0.098] | -0.082 [-0.202, +0.037] | -0.127 [-0.302, +0.060] | +0.028 [-0.010, +0.074] | N/Y/Y/N | **FAILED** |
| T3a-c CONTROL (derangement) | 0.021 | -0.085 [-0.221, +0.004] | +0.009 [-0.023, +0.042] | +0.000 [+0.000, +0.000] | -0.051 [-0.130, +0.002] | -0.077 [-0.197, +0.004] | +0.009 [-0.021, +0.041] | N/Y/Y/N | **FAILED** |
| B1t (label-side bound) | 0.092 | -0.566 [-0.986, -0.239] | +0.159 [+0.080, +0.258] | +0.000 [+0.000, +0.000] | -0.338 [-0.580, -0.137] | -0.546 [-0.945, -0.233] | +0.149 [+0.076, +0.245] | – | bound |
| ORACLE (label-side bound) | 0.676 | -1.995 [-2.537, -1.548] | +0.121 [+0.057, +0.202] | -0.944 [-1.362, -0.609] | -1.593 [-1.932, -1.266] | -1.986 [-2.550, -1.533] | +0.103 [+0.044, +0.175] | – | bound |

## POST-HOC (descriptive; NOT part of the bar or the reading rule)

Natural-frequency weights for the non-turn classes: {'straight': 9.583, 'gentle': 8.707} (turn windows weight 1: all of them are in the capture).

| arm | all-window dADE re-weighted to natural frequency, seed 0 | seed 1 | both draws pooled: turn | straight | all (turn-enriched) |
|---|---|---|---|---|---|
| T3a | -0.0150 [-0.0263, -0.0045] | -0.0136 [-0.0258, -0.0023] | -0.129 [-0.203, -0.061] | +0.003 [-0.006, +0.014] | -0.075 [-0.116, -0.035] |
| T3 | -0.0186 [-0.0323, -0.0067] | -0.0161 [-0.0291, -0.0039] | -0.152 [-0.239, -0.076] | +0.003 [-0.006, +0.014] | -0.088 [-0.137, -0.043] |
| T2a | -0.0237 [-0.0583, +0.0060] | -0.0279 [-0.0596, +0.0011] | -0.240 [-0.457, -0.031] | +0.008 [-0.001, +0.018] | -0.138 [-0.265, -0.015] |
| T3a-c | +0.0017 [-0.0110, +0.0152] | -0.0018 [-0.0095, +0.0055] | -0.008 [-0.071, +0.043] | -0.000 [-0.007, +0.006] | -0.004 [-0.042, +0.027] |

| arm | GT-turn windows | picks changed | improved | worse | episodes with a change | sum dADE over changed | sum of 5 largest gains |
|---|---|---|---|---|---|---|---|
| T3a | 2317 | 151 | 122 | 29 | 19 | -304.057 | -53.105 |
| T3 | 2317 | 165 | 136 | 29 | 20 | -368.618 | -53.105 |

**True announced series vs donor series, paired ADE (T3a - T3a-c; negative = the true series is better)**

| draw | turn | straight | all |
|---|---|---|---|
| seed0 | -0.120 [-0.230, -0.008] | +0.002 [-0.006, +0.011] | -0.071 [-0.134, -0.005] |
| seed1 | -0.123 [-0.219, -0.031] | +0.006 [-0.003, +0.018] | -0.070 [-0.124, -0.015] |

**Across episodes (seed 0, GT-turn windows)**

| arm | turn episodes | episodes dADE < 0 | > 0 | unchanged | leave-one-episode-out turn dADE (min, max) | largest single-episode share of the gain |
|---|---|---|---|---|---|---|
| T3a | 43 | 16 | 3 | 24 | [-0.1401, -0.1125] | 0.16 |
| T2a | 43 | 13 | 5 | 25 | [-0.3033, -0.1882] | 0.24 |

T3a vs T3 picks differ on 15 windows (14 GT-turn).
