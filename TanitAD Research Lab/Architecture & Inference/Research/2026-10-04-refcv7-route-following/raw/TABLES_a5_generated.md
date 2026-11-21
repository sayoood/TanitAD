## controls

| control | expected | measured | verdict |
|---|---|---|---|
| **K1** computed `t_now_raw` vs the replay bank's `t_label_s` | max abs diff <= 1e-3 s on every window | **5.00e-05 s** over 2059 windows / 12 clips (the bank rounds to 4 dp, so a correct clock reads <= 5e-5); W re-derived from the bank = 8 | PASS |
| **K2** nav_tl == clip token's side at t_rel in [-0.05, 0.05], `args.time_s` <= 6.0 (captured windows) | 100 % | **3/3** (EVAL grid n=0, TRAIN grid n=0, reel n=3) | PASS (the EVAL grid itself hits the band 0 times: VACUOUS there) |
| K2 supplementary: one integer t per eval clip at t_rel ~ 0 (not a captured window) | 100 % | **17/17** | PASS |
| K0 join: bank `nav` == record `nav_command` side, eval_s0g | 100 % | 1112/1112 | PASS |
| K0 join: bank `nav` == record `nav_command` side, eval_s1 | 100 % | 1112/1112 | PASS |
| K0 join: bank `nav` == record `nav_command` side, train_s0 | 100 % | 1112/1112 | PASS |
| K0 sidecar coverage | the 3 eval clips with no row are the config's 3 `tactical_excluded_sids`; 0 train-diag clips uncovered | 3 uncovered, sids equal = True; train-diag uncovered 0 | PASS |
| reproduction: V2_clip_token_vs_banked_V2|0.05 | max abs diff <= 2e-4 | 0.0e+00 | PASS |
| reproduction: V3_clip_token_vs_banked_V3|10 | max abs diff <= 2e-4 | 0.0e+00 | PASS |
| reproduction: X1_clip_token_refit_vs_banked_A2_X1 | max abs diff <= 2e-4 | 0.0e+00 | PASS |
| reproduction: B1t_vs_banked_A4_B1t | max abs diff <= 2e-4 | 0.0e+00 | PASS |
| reproduction: A2 X1 refit (alpha, 5-fold CV ADE) | same alpha, CV diff <= 5e-4 | alpha equal True, CV diff 0.0e+00 | PASS |
| recomputed navc term (clip token) == shipped term | <= 1e-9 | eval 0.0e+00 / train 0.0e+00 | PASS |

## token tally (139 eval clips) and window counts (EVAL grid, 1,112 windows)

* clip token (`nav_command`): {'left': 13, 'follow': 88, 'right': 38}; entry tokens (`nav_30s.entries`): {'NAV_FOLLOW_ROAD': 85, 'NAV_TURN_R': 52, 'NAV_TURN_L': 21}; entries per clip: {'1': 123, '2': 13, '3': 3}
* `nav_command` vs `nav_30s.entries[0]` disagree on 9 clips: {'NAV_FOLLOW_ROAD vs entries[0]=NAV_TURN_R': 6, 'NAV_TURN_R vs entries[0]=NAV_FOLLOW_ROAD': 2, 'NAV_TURN_L vs entries[0]=NAV_FOLLOW_ROAD': 1}
* window reason (all 1112 windows): {'ahead_within_H': 84, 'beyond_H': 280, 'follow_token': 680, 'no_entry_left': 25, 'under_way': 43}; t_rel range {'min': -6.12, 'max': 9.42, 'median': 1.517}; clock source {'nominal_dt': 24, 'sidecar': 1088}

| window class | n | nav_tl L / follow / R | clip token L / follow / R | nav_tl non-follow | clip-token non-follow | H=8 non-follow | clip->nav_tl (L->L, L->F, F->L, F->R, R->F, R->R) |
|---|---|---|---|---|---|---|---|
| turn | 107 | 12 / 55 / 40 | 13 / 57 / 37 | 52 | 50 | 52 | 11, 2, 0, 7, 3, 33 |
| turnL | 40 | 11 / 28 / 1 | 13 / 24 / 3 | 12 | 16 | 12 | 11, 2, 0, 1, 3, 0 |
| turnR | 67 | 1 / 27 / 39 | 0 / 33 / 34 | 40 | 34 | 40 | 0, 0, 0, 6, 0, 33 |
| straight | 588 | 4 / 572 / 12 | 52 / 395 / 141 | 16 | 193 | 33 | 4, 48, 0, 3, 132, 9 |
| gentle | 105 | 6 / 83 / 16 | 11 / 57 / 37 | 22 | 48 | 25 | 5, 6, 0, 1, 21, 15 |
| unclassified | 312 | 5 / 275 / 32 | 28 / 195 / 89 | 37 | 117 | 52 | 5, 23, 0, 4, 61, 28 |
| every_window | 1112 | 27 / 985 / 100 | 104 / 704 / 304 | 127 | 408 | 162 | 25, 79, 0, 15, 217, 85 |

nav_tl vs the geometric GT direction (descriptive; `straight` rows count follow == straight):

| class | n | nav_tl == GT dir | nav_tl follow | nav_tl opposite | clip token == GT dir | clip token follow | clip token opposite |
|---|---|---|---|---|---|---|---|
| turn | 107 | 50 | 55 | 2 | 47 | 57 | 3 |
| turnL | 40 | 11 | 28 | 1 | 13 | 24 | 3 |
| turnR | 67 | 39 | 27 | 1 | 34 | 33 | 0 |
| straight | 588 | 572 | 572 | 0 | 395 | 395 | 0 |
| gentle | 105 | 23 | 83 | 2 | 41 | 57 | 8 |

## bar (EVAL, paired episode-cluster bootstrap vs V0, B = 2000, the SPEC sec. 5 draws)

| arm | picks changed | turn dADE [CI] | turn dir-correct d [CI] | straight dADE [CI] | all dADE [CI] | seed 1: turn dADE [CI] | seed 1: turn dir-correct d [CI] | criteria 1/2/3/4 | bar |
|---|---|---|---|---|---|---|---|---|---|
| **T2 (REPORTED)** | 0.037 | -0.264 [-0.568, -0.000] | +0.093 [+0.031, +0.171] | +0.009 [+0.001, +0.020] | -0.043 [-0.089, -0.005] | -0.204 [-0.447, +0.028] | +0.047 [+0.000, +0.108] | Y/Y/Y/N | **FAILED** |
| T3 (a: term replaced) | 0.020 | -0.159 [-0.312, -0.040] | +0.065 [+0.020, +0.124] | +0.011 [+0.002, +0.025] | -0.014 [-0.040, +0.010] | -0.143 [-0.286, -0.032] | +0.056 [+0.011, +0.113] | Y/Y/Y/Y | **CLEARS** |
| T3b (b: term added, sensitivity) | 0.017 | -0.159 [-0.312, -0.040] | +0.065 [+0.020, +0.124] | +0.011 [+0.002, +0.025] | -0.014 [-0.040, +0.010] | -0.143 [-0.286, -0.032] | +0.056 [+0.011, +0.113] | Y/Y/Y/Y | **CLEARS** |
| T4 refit re-scorer | 0.053 | -0.006 [-0.171, +0.138] | +0.009 [-0.021, +0.045] | -0.034 [-0.076, +0.000] | -0.018 [-0.059, +0.019] | +0.013 [-0.120, +0.162] | +0.000 [-0.026, +0.027] | N/Y/Y/N | **FAILED** |
| T2c CONTROL (derangement) | 0.088 | -0.071 [-0.351, +0.135] | +0.009 [-0.035, +0.062] | +0.106 [+0.051, +0.166] | +0.089 [+0.037, +0.146] | -0.029 [-0.233, +0.150] | +0.000 [-0.041, +0.050] | N/N/N/N | **FAILED** |
| T2h8 H=8 (sensitivity) | 0.057 | -0.264 [-0.568, -0.000] | +0.093 [+0.031, +0.171] | +0.016 [-0.009, +0.043] | -0.039 [-0.082, -0.000] | -0.204 [-0.447, +0.028] | +0.047 [+0.000, +0.108] | Y/Y/Y/N | **FAILED** |
| B1t (label-side bound, not a lever) | 0.015 | -0.566 [-0.986, -0.239] | +0.159 [+0.080, +0.258] | +0.000 [+0.000, +0.000] | -0.076 [-0.132, -0.029] | -0.546 [-0.945, -0.233] | +0.149 [+0.076, +0.245] | – | bound |
| ORACLE (label-side bound, not a lever) | 0.597 | -1.995 [-2.537, -1.548] | +0.121 [+0.057, +0.202] | -0.925 [-1.102, -0.754] | -1.175 [-1.343, -1.007] | -1.986 [-2.550, -1.533] | +0.103 [+0.044, +0.175] | – | bound |

T2 fraction of the A4 B1t bound: dADE_turn(T2) -0.2643 / (-0.566) = **0.467** (B1t recomputed here -0.5658); per-draw ratio bootstrap 95 % [0.0010148931587815968, 0.85336217184577]

## four families (SPEC 2.1 format), EVAL seed 0 (eval_s0g)


**GT-turn windows (107)**

| arm | ADE | FDE | LON |along| 6 s | LON along 6 s signed | LON speed MAE 0-2 s | LAT |cross| 6 s | LAT heading MAE 0-2 s deg | LAT curv MAE 0-2 s | LAT term. heading err deg | TAC dir correct | STR nav compliance (vs the CLIP token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 3.186 | 10.504 | 7.418 | 3.272 | 0.312 | 6.182 | 4.006 | 0.010 | 22.369 | 0.841 | 0.740 |
| T2 | 2.922 | 9.623 | 6.426 | 2.415 | 0.290 | 5.964 | 3.928 | 0.009 | 20.074 | 0.935 | 0.900 |
| T3 | 3.028 | 9.928 | 6.937 | 2.901 | 0.302 | 5.882 | 3.947 | 0.009 | 20.607 | 0.906 | 0.860 |
| T4 | 3.180 | 10.498 | 7.187 | 2.955 | 0.314 | 6.455 | 4.024 | 0.009 | 23.164 | 0.851 | 0.780 |
| T2c | 3.115 | 10.296 | 7.192 | 2.725 | 0.302 | 6.089 | 3.992 | 0.010 | 22.878 | 0.851 | 0.740 |
| T2h8 | 2.922 | 9.623 | 6.426 | 2.415 | 0.290 | 5.964 | 3.928 | 0.009 | 20.074 | 0.935 | 0.900 |
| B1t | 2.620 | 8.629 | 6.233 | 2.189 | 0.289 | 5.019 | 3.576 | 0.008 | 17.217 | 1.000 | 0.940 |
| ORACLE | 1.191 | 3.598 | 2.194 | 0.266 | 0.217 | 2.402 | 3.405 | 0.007 | 13.404 | 0.963 | 0.880 |

**GT-straight windows (588)**

| arm | ADE | FDE | LON |along| 6 s | LON along 6 s signed | LON speed MAE 0-2 s | LAT |cross| 6 s | LAT heading MAE 0-2 s deg | LAT curv MAE 0-2 s | LAT term. heading err deg | TAC dir correct | STR nav compliance (vs the CLIP token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 1.656 | 5.220 | 4.824 | 0.426 | 0.246 | 1.079 | 0.833 | 0.002 | 2.348 | 0.964 | 0.021 |
| T2 | 1.666 | 5.242 | 4.820 | 0.413 | 0.246 | 1.125 | 0.848 | 0.002 | 2.453 | 0.954 | 0.047 |
| T3 | 1.667 | 5.261 | 4.849 | 0.403 | 0.247 | 1.105 | 0.841 | 0.002 | 2.479 | 0.956 | 0.047 |
| T4 | 1.622 | 5.108 | 4.724 | 0.433 | 0.243 | 1.042 | 0.810 | 0.002 | 2.255 | 0.959 | 0.036 |
| T2c | 1.762 | 5.627 | 4.944 | 0.102 | 0.245 | 1.540 | 0.940 | 0.002 | 3.656 | 0.918 | 0.062 |
| T2h8 | 1.672 | 5.263 | 4.784 | 0.310 | 0.246 | 1.212 | 0.850 | 0.002 | 2.774 | 0.937 | 0.093 |
| B1t | 1.656 | 5.220 | 4.824 | 0.426 | 0.246 | 1.079 | 0.833 | 0.002 | 2.348 | 0.964 | 0.021 |
| ORACLE | 0.731 | 2.124 | 1.796 | -0.156 | 0.162 | 0.757 | 0.802 | 0.002 | 2.228 | 0.954 | 0.026 |

**all classified windows (800)**

| arm | ADE | FDE | LON |along| 6 s | LON along 6 s signed | LON speed MAE 0-2 s | LAT |cross| 6 s | LAT heading MAE 0-2 s deg | LAT curv MAE 0-2 s | LAT term. heading err deg | TAC dir correct | STR nav compliance (vs the CLIP token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 2.004 | 6.440 | 5.457 | 0.951 | 0.262 | 2.211 | 1.498 | 0.003 | 6.220 | 0.891 | 0.203 |
| T2 | 1.961 | 6.281 | 5.289 | 0.767 | 0.259 | 2.183 | 1.491 | 0.003 | 5.796 | 0.906 | 0.275 |
| T3 | 1.990 | 6.393 | 5.412 | 0.892 | 0.261 | 2.196 | 1.496 | 0.003 | 6.050 | 0.900 | 0.261 |
| T4 | 1.986 | 6.382 | 5.390 | 0.921 | 0.262 | 2.213 | 1.483 | 0.003 | 6.230 | 0.894 | 0.234 |
| T2c | 2.094 | 6.791 | 5.544 | 0.578 | 0.262 | 2.618 | 1.592 | 0.004 | 7.648 | 0.854 | 0.227 |
| T2h8 | 1.966 | 6.296 | 5.259 | 0.686 | 0.259 | 2.249 | 1.493 | 0.003 | 6.029 | 0.892 | 0.309 |
| B1t | 1.929 | 6.189 | 5.298 | 0.806 | 0.259 | 2.056 | 1.440 | 0.003 | 5.531 | 0.912 | 0.237 |
| ORACLE | 0.830 | 2.478 | 1.916 | -0.103 | 0.173 | 1.131 | 1.170 | 0.003 | 4.390 | 0.919 | 0.237 |

**TACTICAL, `four_families.tactical_from_trajectory` (0-2 s, all 1,112 windows)**

| arm | lateral accuracy / kappa | longitudinal accuracy / kappa |
|---|---|---|
| V0 | 0.9568 / 0.8229 | 0.8273 / 0.5395 |
| T2 | 0.9577 / 0.8250 | 0.8309 / 0.5545 |
| T3 | 0.9577 / 0.8281 | 0.8318 / 0.5522 |
| T4 | 0.9586 / 0.8283 | 0.8345 / 0.5566 |
| T2c | 0.9568 / 0.8239 | 0.8309 / 0.5531 |
| T2h8 | 0.9577 / 0.8250 | 0.8300 / 0.5541 |
| ORACLE | 0.9640 / 0.8512 | 0.9415 / 0.8530 |

*Coverage: LONGITUDINAL distance keeping UNAVAILABLE (no lead tracks); STRATEGIC decision UNAVAILABLE (`--no-strategic`).*

## T4 fit (TRAIN, 5-fold episode-grouped CV; V0 TRAIN ADE in the row)

* T4 (nav_tl features): alpha 0.01, CV ADE by alpha {'0.001': 1.4736, '0.01': 1.465, '0.1': 1.4815, '1.0': 1.4838, '10.0': 1.4838}, in-sample 1.468, V0 1.4838; nav weights navc 0.0782, agree_nav_side 0.0782
* A2 X1 (clip token, banked): alpha 0.001, CV ADE by alpha {'0.001': 1.4693, '0.01': 1.4705, '0.1': 1.4903, '1.0': 1.4838, '10.0': 1.4838}, in-sample 1.4536

## POST-HOC (descriptive; NOT part of the bar or the reading rule)

**Both sampler draws pooled** (per-window dADE averaged over seeds 0 and 1):

| arm | turn | straight | all |
|---|---|---|---|
| T2 | -0.245 [-0.523, +0.006] | +0.008 [-0.003, +0.019] | -0.041 [-0.087, -0.003] |
| T3 | -0.156 [-0.309, -0.037] | +0.011 [+0.001, +0.027] | -0.014 [-0.039, +0.008] |
| T2c | -0.054 [-0.308, +0.142] | +0.110 [+0.055, +0.171] | +0.099 [+0.042, +0.159] |
| T2h8 | -0.245 [-0.523, +0.006] | +0.019 [-0.005, +0.044] | -0.033 [-0.074, +0.005] |

**True series vs donor series, paired ADE (T2 - T2c; negative = the true series is better)**

| draw | turn | straight | all |
|---|---|---|---|
| eval_s0g | -0.193 [-0.426, +0.016] | -0.097 [-0.158, -0.041] | -0.132 [-0.197, -0.072] |
| eval_s1 | -0.188 [-0.427, +0.016] | -0.108 [-0.173, -0.050] | -0.146 [-0.222, -0.079] |

**Where the GT-turn gain sits (seed 0)**

| arm | GT-turn windows | picks changed | improved | worse | episodes with a change | sum dADE over changed | sum of the 5 largest gains |
|---|---|---|---|---|---|---|---|
| T2 | 107 | 14 | 10 | 4 | 11 | -28.285 | -28.885 |
| T3 | 107 | 7 | 6 | 1 | 6 | -16.959 | -16.403 |

**How many GT-turn windows can a nav signal touch?** GT-turn 107: nav_tl active on 52, silent (follow) on 55; clip token active on 50.

| arm | nav_tl-active GT-turn (n=52) seed 0 | seed 1 | nav_tl-silent GT-turn (n=55) seed 0 | seed 1 |
|---|---|---|---|---|
| T2 dADE | -0.544 [-1.119, -0.001] | -0.464 [-0.958, +0.029] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| T3 dADE | -0.326 [-0.623, -0.089] | -0.315 [-0.611, -0.077] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| B1t dADE | -0.817 [-1.416, -0.306] | -0.802 [-1.361, -0.319] | -0.329 [-1.046, +0.007] | -0.324 [-1.094, +0.015] |

**Pure timing vs extra information: GT-turn windows (107) partitioned by the shipped clip token and nav_tl** (contribution = sum of per-window dADE / 107, i.e. what the part adds to the GT-turn mean)

| part | n | T2 seed 0: sum dADE / contribution | T2 seed 1 | T3 seed 0 | T3 seed 1 |
|---|---|---|---|---|---|
| clip_active_and_nav_tl_active | 45 | -12.94 / -0.121 | -10.60 / -0.099 | -11.42 / -0.107 | -10.83 / -0.101 |
| clip_FOLLOW_but_nav_tl_active_(extra_information_from_nav_30s) | 7 | -15.34 / -0.143 | -13.54 / -0.127 | -5.54 / -0.052 | -5.54 / -0.052 |
| clip_active_but_nav_tl_follow_(time-localisation_switches_it_off) | 5 | +0.00 / +0.000 | +0.00 / +0.000 | +0.00 / +0.000 | +0.00 / +0.000 |
| both_follow | 50 | +0.00 / +0.000 | +0.00 / +0.000 | +0.00 / +0.000 | +0.00 / +0.000 |

T2c vs T2 series: {'n_windows_side_differs': 218, 'n_windows_both_active': 22, 'n_windows_both_active_same_side': 14}

**Second sample for the RULE: the TRAIN-DIAG fans (139 other episodes; the checkpoint was TRAINED on them: NOT held-out, no bar verdict)** classes {'turn': 104, 'straight': 588, 'gentle': 115, 'all': 807}; nav_tl active on 164 of 1,112 windows

| arm | picks changed | turn dADE [CI] | turn dir-correct d [CI] | straight dADE [CI] | all dADE [CI] |
|---|---|---|---|---|---|
| T2 | 0.032 | +0.158 [-0.101, +0.477] | +0.038 [-0.018, +0.106] | +0.014 [+0.001, +0.031] | +0.031 [-0.004, +0.075] |
| T3 | 0.030 | -0.119 [-0.221, -0.042] | +0.096 [+0.047, +0.154] | +0.001 [-0.012, +0.015] | -0.013 [-0.032, +0.005] |
| T2c | 0.112 | +0.578 [+0.125, +1.059] | -0.086 [-0.162, -0.018] | +0.158 [+0.090, +0.246] | +0.245 [+0.143, +0.364] |
| T2h8 | 0.049 | +0.158 [-0.101, +0.477] | +0.029 [-0.038, +0.102] | +0.017 [+0.000, +0.037] | +0.035 [-0.002, +0.079] |
| B1t | 0.020 | -0.363 [-0.596, -0.171] | +0.211 [+0.138, +0.301] | +0.000 [+0.000, +0.000] | -0.047 [-0.080, -0.021] |
