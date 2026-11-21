### T1 Official two-stage EPDMS: what each zero term costs (A1 step 30,000; counterfactual, ESTIMATED upper bound)

| change applied to A1's per-scene sub-scores | official EPDMS | delta vs A1 0.2269 | A1 vs STOP 0.2985 after |
|---|---|---|---|
| DAC := STOP's DAC where A1 DAC=0 and STOP DAC=1 | 0.3504 | +0.1235 | +0.0519 |
| NC := STOP's NC where A1 NC=0 and STOP NC=1 | 0.2804 | +0.0535 | -0.0181 |
| DDC := STOP's DDC likewise | 0.2378 | +0.0109 | -0.0607 |
| TLC := STOP's TLC likewise | 0.2360 | +0.0091 | -0.0625 |
| DAC := 1 on every scene (ceiling) | 0.4165 | +0.1896 | +0.1180 |
| NC := 1 on every scene (ceiling) | 0.2929 | +0.0660 | -0.0056 |
| DAC := 1 on the 450 stage-1 rows only | 0.3245 | +0.0975 | +0.0259 |
| DAC := 1 on the 5,462 stage-2 rows only | 0.2744 | +0.0475 | -0.0241 |
| NC := 1 on stage-1 rows only | 0.2446 | +0.0177 | -0.0539 |
| NC := 1 on stage-2 rows only | 0.2692 | +0.0423 | -0.0293 |
| all four multipliers := 1 (ceiling) | 0.6690 | +0.4421 | +0.3705 |

### T2 Who else fails where refcv7 (A1 step 30k) fails -- zero terms, per stage

| stage | term | A1 zero n (rate) | STOP rate | PRIOR rate | A1 zero & STOP also zero | A1 zero & STOP passes | A1 zero & PRIOR also zero | A1 zero & seed-1 also zero | A1 zero & 5k also zero | seed Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DAC | 145 (32.2 %) | 6.9 % | 49.1 % | 20 | 125 | 111 | 130 | 106 | 0.84 |
| 1 | NC | 35 (7.8 %) | 0.2 % | 6.9 % | 1 | 34 | 17 | 33 | 22 | 0.82 |
| 1 | DDC | 15 (3.3 %) | 0.0 % | 14.4 % | 0 | 15 | 6 | 13 | 6 | 0.76 |
| 1 | TLC | 2 (0.4 %) | 0.2 % | 0.7 % | 0 | 2 | 2 | 2 | 2 | 0.67 |
| 2 | DAC | 1418 (26.0 %) | 13.7 % | 47.0 % | 604 | 814 | 1073 | 1312 | 1177 | 0.86 |
| 2 | NC | 854 (15.6 %) | 4.0 % | 16.3 % | 199 | 655 | 556 | 799 | 703 | 0.90 |
| 2 | DDC | 598 (10.9 %) | 1.8 % | 24.1 % | 96 | 502 | 486 | 555 | 497 | 0.87 |
| 2 | TLC | 92 (1.7 %) | 0.5 % | 1.2 % | 21 | 71 | 59 | 90 | 85 | 0.96 |

### T3 DAC-zero scenes (A1 30k): scene-level flag from the exact re-score (rescored 1563 of 1563)

| scene-level flag | stage 1 n (share) | stage 2 n (share) | both n (share) [95 % log-cluster CI] |
|---|---|---|---|
| INITIAL-STATE | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) [0.0, 0.0] |
| REF-ALSO-FAILS | 4 (2.8 %) | 423 (29.8 %) | 427 (27.3 %) [22.6, 31.8] |
| PLAN-INDUCED | 141 (97.2 %) | 995 (70.2 %) | 1136 (72.7 %) [68.2, 77.4] |

First non-drivable instant of the plan (all rescored DAC-zero scenes):

| time bin | stage 1 | stage 2 | both |
|---|---|---|---|
| t=0 s | 0 | 0 | 0 (0.0 %) |
| (0,1] s | 10 | 497 | 507 (32.4 %) |
| (1,2] s | 21 | 331 | 352 (22.5 %) |
| (2,3] s | 48 | 281 | 329 (21.0 %) |
| (3,4] s | 66 | 309 | 375 (24.0 %) |

median first non-drivable instant (both): 1.8 s

First, for the same rescored DAC-zero scenes: does a clean path exist (PDM-closed reference passes DAC), does doing nothing pass (STOP), does the kinematic prior pass (PRIOR)?

| reference | STOP | PRIOR | stage 1 n | stage 2 n | both n (share) |
|---|---|---|---|---|---|
| ref_clean | STOP_clean | PRIOR_FAILS | 92 | 394 | 486 (31.1 %) |
| ref_clean | STOP_clean | PRIOR_clean | 29 | 294 | 323 (20.7 %) |
| ref_clean | STOP_FAILS | PRIOR_FAILS | 17 | 298 | 315 (20.2 %) |
| ref_FAILS | STOP_FAILS | PRIOR_FAILS | 0 | 296 | 296 (18.9 %) |
| ref_FAILS | STOP_clean | PRIOR_FAILS | 2 | 85 | 87 (5.6 %) |
| ref_FAILS | STOP_clean | PRIOR_clean | 2 | 41 | 43 (2.8 %) |
| ref_clean | STOP_FAILS | PRIOR_clean | 3 | 9 | 12 (0.8 %) |
| ref_FAILS | STOP_FAILS | PRIOR_clean | 0 | 1 | 1 (0.1 %) |

seed-1 plan DAC-clean on 121 of 1563 (7.7 %); step-5,000 plan DAC-clean on 280 (17.9 %); reference clean AND none of STOP / PRIOR / seed-1 clean: 305 (19.5 %).

### T4 DAC-zero scenes by plan geometry class (route-referenced; literals in section 2)

| class | DAC-zero n (both) | share [95 % log CI] | PLAN-INDUCED subset n (share) | all scenes in class n | P(DAC=0 given class) A1 | STOP | PRIOR |
|---|---|---|---|---|---|---|---|
| LATERAL-DRIFT | 483 | 30.9 % [26.8, 35.2] | 346 (30.5 %) | 1526 | 31.7 % | 16.6 % | 60.9 % |
| ROUTE-FOLLOWING | 363 | 23.2 % [18.5, 28.1] | 256 (22.5 %) | 760 | 47.8 % | 18.0 % | 55.9 % |
| ON-ROUTE | 287 | 18.4 % [15.3, 21.5] | 243 (21.4 %) | 2032 | 14.1 % | 9.0 % | 36.2 % |
| OVER-STEER | 203 | 13.0 % [9.6, 16.2] | 170 (15.0 %) | 336 | 60.4 % | 16.7 % | 55.1 % |
| SPEED | 147 | 9.4 % [7.7, 11.1] | 66 (5.8 %) | 476 | 30.9 % | 22.7 % | 60.9 % |
| NO-RECOVERY | 42 | 2.7 % [1.6, 4.1] | 31 (2.7 %) | 382 | 11.0 % | 5.2 % | 45.5 % |
| WRONG-SIDE | 34 | 2.2 % [1.1, 3.3] | 24 (2.1 %) | 59 | 57.6 % | 30.5 % | 76.3 % |
| STOP-LIKE | 4 | 0.3 % [0.0, 0.6] | 0 (0.0 %) | 341 | 1.2 % | 1.2 % | 1.2 % |

### T5 Class mix of the DAC-zero scenes: step 5,000 vs 30,000 vs PRIOR vs seed-1 (geometry classes only; same literals)

| class | A1 5k n (share) | A1 30k n (share) | A1 30k seed-1 n | PRIOR n (share) |
|---|---|---|---|---|
| LATERAL-DRIFT | 804 (38.1 %) | 483 (30.9 %) | 481 | 1684 (60.4 %) |
| ROUTE-FOLLOWING | 626 (29.6 %) | 363 (23.2 %) | 360 | 484 (17.4 %) |
| ON-ROUTE | 283 (13.4 %) | 287 (18.4 %) | 303 | 133 (4.8 %) |
| OVER-STEER | 220 (10.4 %) | 203 (13.0 %) | 199 | 298 (10.7 %) |
| SPEED | 67 (3.2 %) | 147 (9.4 %) | 142 | 22 (0.8 %) |
| NO-RECOVERY | 39 (1.8 %) | 42 (2.7 %) | 38 | 21 (0.8 %) |
| WRONG-SIDE | 73 (3.5 %) | 34 (2.2 %) | 33 | 133 (4.8 %) |
| STOP-LIKE | 1 (0.0 %) | 4 (0.3 %) | 4 | 13 (0.5 %) |

DAC-zero totals: 5k 2113, 30k 1563, seed-1 1560, PRIOR 2788.

### T6 Command x route geometry in the horizon (all 5,912 scenes) and DAC-zero rates

| command | n | route in horizon: junction turn | curve turn | straight | A1 DAC0 | STOP DAC0 | PRIOR DAC0 |
|---|---|---|---|---|---|---|---|
| LEFT | 1058 | 488 | 181 | 389 | 39.1 % | 16.1 % | 47.2 % |
| RIGHT | 1254 | 769 | 150 | 335 | 40.0 % | 18.1 % | 50.7 % |
| STRAIGHT | 3586 | 280 | 36 | 3270 | 18.0 % | 10.6 % | 46.0 % |
| UNKNOWN | 14 | 0 | 0 | 14 | 28.6 % | 7.1 % | 28.6 % |

Plan complied with the command (LEFT: plan yaw at 4 s >= +10 deg; RIGHT: <= -10 deg; STRAIGHT: |yaw| < 20 deg) and DAC-zero rate, by command x route-in-horizon:

| command | route in horizon | n | plan complied | A1 DAC0 | STOP DAC0 | A1 DAC0 if complied (n) | A1 DAC0 if not complied (n) |
|---|---|---|---|---|---|---|---|
| LEFT | route turns in horizon (curve) | 181 | 90.6 % | 51.4 % | 27.6 % | 53.7 % (164) | 29.4 % (17) |
| LEFT | route turns in horizon (junction) | 488 | 85.5 % | 44.9 % | 15.8 % | 48.0 % (417) | 26.8 % (71) |
| LEFT | route straight in horizon | 389 | 59.6 % | 26.2 % | 11.1 % | 37.9 % (232) | 8.9 % (157) |
| RIGHT | route turns in horizon (curve) | 150 | 85.3 % | 55.3 % | 32.0 % | 61.7 % (128) | 18.2 % (22) |
| RIGHT | route turns in horizon (junction) | 769 | 76.7 % | 38.8 % | 14.3 % | 44.2 % (590) | 20.7 % (179) |
| RIGHT | route straight in horizon | 335 | 59.4 % | 35.8 % | 20.6 % | 43.2 % (199) | 25.0 % (136) |
| STRAIGHT | route turns in horizon (curve) | 36 | 58.3 % | 50.0 % | 8.3 % | 42.9 % (21) | 60.0 % (15) |
| STRAIGHT | route turns in horizon (junction) | 280 | 46.1 % | 25.7 % | 6.4 % | 18.6 % (129) | 31.8 % (151) |
| STRAIGHT | route straight in horizon | 3270 | 94.5 % | 16.9 % | 11.0 % | 15.9 % (3089) | 34.3 % (181) |

DAC-zero ROUTE-FOLLOWING / WRONG-SIDE / OVER-STEER / LATERAL-DRIFT scenes by command and by route kind (n per cell):

| class | DAC-zero n | LEFT | RIGHT | STRAIGHT | junction turn | curve turn | straight route | all scenes of class: LEFT/RIGHT/STRAIGHT | P(DAC0) in class: LEFT / RIGHT / STRAIGHT | all scenes of class: junction/curve/straight | P(DAC0) in class: junction / curve |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ROUTE-FOLLOWING | 363 | 118 | 192 | 53 | 273 | 90 | 0 | 211/406/143 | 55.9 % / 47.3 % / 37.1 % | 630/130/0 | 43.3 % / 69.2 % |
| WRONG-SIDE | 34 | 20 | 9 | 5 | 24 | 10 | 0 | 31/16/12 | 64.5 % / 56.2 % / 41.7 % | 40/19/0 | 60.0 % / 52.6 % |
| OVER-STEER | 203 | 106 | 85 | 12 | 144 | 59 | 0 | 172/137/27 | 61.6 % / 62.0 % / 44.4 % | 248/88/0 | 58.1 % / 67.0 % |
| LATERAL-DRIFT | 483 | 87 | 92 | 302 | 0 | 0 | 483 | 195/184/1140 | 44.6 % / 50.0 % / 26.5 % | 0/0/1526 | n/a / n/a |
| NO-RECOVERY | 42 | 3 | 4 | 35 | 0 | 0 | 42 | 34/20/326 | 8.8 % / 20.0 % / 10.7 % | 0/0/382 | n/a / n/a |
| SPEED | 147 | 39 | 39 | 69 | 65 | 11 | 71 | 89/90/297 | 43.8 % / 43.3 % / 23.2 % | 145/30/301 | 44.8 % / 36.7 % |
| ON-ROUTE | 287 | 40 | 78 | 168 | 82 | 23 | 182 | 253/306/1470 | 15.8 % / 25.5 % / 11.4 % | 347/80/1605 | 23.6 % / 28.7 % |
| STOP-LIKE | 4 | 1 | 2 | 0 | 1 | 1 | 2 | 73/95/171 | 1.4 % / 2.1 % / 0.0 % | 127/20/194 | 0.8 % / 5.0 % |

### T7 NC-zero scenes (A1 30k): first at-fault collision from the exact re-score (rescored 889 of 889)

| class | stage 1 n | stage 2 n | both n (share) [95 % log CI] | median time s | ego speed m/s | object speed m/s | rel. longitudinal m | also DAC-zero | STOP also NC0 | PRIOR also NC0 | seed-1 also NC0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACTIVE-FRONT-VEHICLE | 17 | 487 | 504 (56.7 %) [48.5, 65.1] | 2.2 | 6.1 | 1.7 | 6.2 | 91 | 94 | 337 | 483 |
| STOPPED-TRACK-VEHICLE/OBJECT | 13 | 294 | 307 (34.5 %) [26.2, 43.0] | 2.1 | 5.2 | 0.0 | 5.9 | 138 | 75 | 185 | 278 |
| LATERAL-AFTER-LANE-DEPARTURE | 2 | 25 | 27 (3.0 %) [1.2, 6.1] | 2.1 | 3.1 | 3.7 | 0.9 | 3 | 7 | 10 | 24 |
| ACTIVE-FRONT-VRU | 2 | 22 | 24 (2.7 %) [0.6, 5.7] | 2.5 | 5.3 | 1.4 | 4.3 | 3 | 4 | 18 | 22 |
| INITIAL-OVERLAP | 0 | 19 | 19 (2.1 %) [1.0, 3.4] | 0.1 | 5.3 | 0.0 | 6.0 | 6 | 19 | 19 | 19 |
| STOPPED-TRACK-VRU | 1 | 7 | 8 (0.9 %) [0.0, 2.6] | 3.1 | 3.7 | 0.0 | 4.3 | 8 | 1 | 4 | 6 |

plan 4-s speed minus PDM-closed reference 4-s speed (finite-difference), median: NC-zero +3.87 m/s vs all scenes -0.48 m/s; share with plan >= 3 m/s faster than the reference: NC-zero 64.8 % vs all 23.3 %.

NC-zero rate by ego speed at t0 (all 5,912 scenes):

| v0 band m/s | n | A1 NC0 | STOP NC0 | PRIOR NC0 |
|---|---|---|---|---|
| 0-2 | 1660 | 8.0 % | 1.1 % | 3.2 % |
| 2-5 | 1875 | 17.8 % | 4.4 % | 18.0 % |
| 5-8 | 1375 | 20.6 % | 5.7 % | 24.4 % |
| 8-12 | 842 | 15.0 % | 4.4 % | 22.0 % |
| >=12 | 160 | 8.8 % | 3.8 % | 6.9 % |

### T8 Max-speed input known vs unknown (D4 finding 1)

NavSim feeds `v_max_valid = 0` (all-zero row, 'unknown') on these shares; zero-rates are per-stratum, n per cell:

| split | stage | stratum | n (share of scenes) | A1 DAC0 | STOP DAC0 | PRIOR DAC0 | A1 NC0 | STOP NC0 | PRIOR NC0 | share of A1's DAC0 / NC0 |
|---|---|---|---|---|---|---|---|---|---|---|
| navhard | 1 | unknown | 206 (45.8 %) | 30.6 % | 3.9 % | 44.7 % | 2.9 % | 0.0 % | 4.4 % | 43.4 % / 17.1 % |
| navhard | 1 | known | 244 (54.2 %) | 33.6 % | 9.4 % | 52.9 % | 11.9 % | 0.4 % | 9.0 % | 56.6 % / 82.9 % |
| navhard | 2 | unknown | 2491 (45.6 %) | 24.8 % | 13.6 % | 46.2 % | 15.0 % | 4.1 % | 16.6 % | 43.6 % / 43.7 % |
| navhard | 2 | known | 2971 (54.4 %) | 26.9 % | 13.7 % | 47.7 % | 16.2 % | 4.0 % | 16.1 % | 56.4 % / 56.3 % |
| navhard | both | unknown | 2697 (45.6 %) | 25.3 % | 12.9 % | 46.1 % | 14.1 % | 3.7 % | 15.7 % | 43.6 % / 42.6 % |
| navhard | both | known | 3215 (54.4 %) | 27.4 % | 13.4 % | 48.1 % | 15.9 % | 3.8 % | 15.6 % | 56.4 % / 57.4 % |
| navtest | 1 | unknown | 5488 (45.2 %) | 13.6 % | 3.0 % | 28.2 % | 6.9 % | 2.3 % | 10.6 % | 51.2 % / 37.7 % |
| navtest | 1 | known | 6658 (54.8 %) | 10.7 % | 3.8 % | 24.1 % | 9.4 % | 2.8 % | 9.6 % | 48.8 % / 62.3 % |

Standardised (over stage x v0-band x command cells) unknown-minus-known zero-rate difference, percentage points; PRIOR is an input-free kinematic control, STOP a plan-free one:

| split | term | A1 | PRIOR (control) | STOP (control) |
|---|---|---|---|---|
| navhard | DAC | -0.31 | +1.84 | +0.79 |
| navtest | DAC | +4.06 | +4.44 | -0.56 |
| navhard | NC | -1.49 | +0.82 | +0.18 |
| navtest | NC | -3.65 | +0.36 | -0.23 |

Difference-in-differences [(A1 - reference arm)_unknown - (A1 - reference arm)_known], log-cluster 95 % CI:

| split | term | reference arm | DiD (pp) | 95 % CI (pp) | n unknown / known |
|---|---|---|---|---|---|
| navhard both | DAC | STOP (plan-free) | -1.68 | [-5.74, +1.70] | 2697 / 3215 |
| navhard both | NC | STOP (plan-free) | -1.79 | [-6.86, +2.90] | 2697 / 3215 |
| navhard stage 2 | DAC | STOP (plan-free) | -2.03 | [-6.15, +1.57] | 2491 / 2971 |
| navhard stage 2 | NC | STOP (plan-free) | -1.23 | [-6.51, +3.70] | 2491 / 2971 |
| navtest | DAC | STOP (plan-free) | +3.76 | [+1.08, +6.28] | 5488 / 6658 |
| navtest | NC | STOP (plan-free) | -1.96 | [-4.41, +0.78] | 5488 / 6658 |
| navhard both | DAC | PRIOR (input-free kinematic plan) | -0.15 | [-6.02, +5.00] | 2697 / 3215 |
| navhard both | NC | PRIOR (input-free kinematic plan) | -1.94 | [-6.37, +1.63] | 2697 / 3215 |
| navhard stage 2 | DAC | PRIOR (input-free kinematic plan) | -0.59 | [-7.01, +5.16] | 2491 / 2971 |
| navhard stage 2 | NC | PRIOR (input-free kinematic plan) | -1.75 | [-6.28, +2.06] | 2491 / 2971 |
| navtest | DAC | PRIOR (input-free kinematic plan) | -1.08 | [-5.08, +3.32] | 5488 / 6658 |
| navtest | NC | PRIOR (input-free kinematic plan) | -3.58 | [-5.70, -1.63] | 5488 / 6658 |

### T9 NAVTEST (single-stage PDMS v1, 12,146 tokens): DAC-zero classes against the HUMAN future

| class | A1 30k DAC0 n (share) | A1 5k n (share) | PRIOR n (share) | P(DAC0 given class) A1 30k | all scenes in class |
|---|---|---|---|---|---|
| ON-HUMAN | 599 (41.0 %) | 639 (34.1 %) | 657 (20.8 %) | 7.7 % | 7814 |
| ROUTE-FOLLOWING | 291 (19.9 %) | 501 (26.7 %) | 615 (19.5 %) | 27.2 % | 1068 |
| OVER-STEER | 285 (19.5 %) | 359 (19.1 %) | 556 (17.6 %) | 26.1 % | 1090 |
| LATERAL-DRIFT | 229 (15.7 %) | 263 (14.0 %) | 1230 (39.0 %) | 25.1 % | 912 |
| SPEED | 46 (3.2 %) | 75 (4.0 %) | 33 (1.0 %) | 5.2 % | 879 |
| WRONG-SIDE | 10 (0.7 %) | 39 (2.1 %) | 61 (1.9 %) | 34.5 % | 29 |
| STOP-LIKE | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) | 0.0 % | 354 |

navtest DAC-zero totals: 30k 1460 (12.0 %), 5k 1876 (15.4 %), PRIOR 3152 (26.0 %).
Among A1 30k's 1460 DAC-zero tokens: STOP also zero 219, STOP passes 1241, PRIOR also zero 895, seed-1 also zero 1247, 5k also zero 880, STOP and PRIOR both pass 502; seed Jaccard 0.74.

NAVTEST NC-zero (A1 30k) by plan class vs the human, and plan-vs-human distance:

| class | NC0 n (share) | P(NC0 given class) A1 | STOP | all scenes in class |
|---|---|---|---|---|
| SPEED | 464 (46.0 %) | 52.8 % | 1.8 % | 879 |
| ON-HUMAN | 312 (31.0 %) | 4.0 % | 2.7 % | 7814 |
| LATERAL-DRIFT | 125 (12.4 %) | 13.7 % | 1.8 % | 912 |
| OVER-STEER | 71 (7.0 %) | 6.5 % | 2.9 % | 1090 |
| ROUTE-FOLLOWING | 28 (2.8 %) | 2.6 % | 2.9 % | 1068 |
| STOP-LIKE | 6 (0.6 %) | 1.7 % | 1.7 % | 354 |
| WRONG-SIDE | 2 (0.2 %) | 6.9 % | 3.4 % | 29 |

plan 4-s distance / human 4-s distance, median: NC-zero 1.52 vs all 1.07; share with plan >= 4 m further than the human: NC-zero 57.0 % vs all 17.2 %.
NC-zero 1008 (8.3 %): STOP also 23, PRIOR also 434, seed-1 also 885, 5k also 680, also DAC-zero 77; seed Jaccard 0.76.

navtest command split:

| command | n | GT turn in 4 s (L/R/S) | A1 DAC0 | STOP DAC0 | PRIOR DAC0 | A1 NC0 | STOP NC0 | plan complied |
|---|---|---|---|---|---|---|---|---|
| LEFT | 2501 | 1891/4/606 | 21.6 % | 7.2 % | 37.7 % | 6.5 % | 3.0 % | 87.2 % |
| RIGHT | 1575 | 2/1070/503 | 20.8 % | 5.9 % | 43.0 % | 5.2 % | 2.9 % | 74.5 % |
| STRAIGHT | 8070 | 78/103/7889 | 7.4 % | 1.8 % | 19.0 % | 9.5 % | 2.4 % | 97.6 % |

### T10 Threshold sensitivity of the DAC-zero class shares (navhard, both stages; every literal x0.5 / x1.5)

| class | share at the literals | min over scenarios | max |
|---|---|---|---|
| LATERAL-DRIFT | 30.9 % | 22.1 % | 38.6 % |
| ROUTE-FOLLOWING | 23.2 % | 15.8 % | 32.3 % |
| ON-ROUTE | 18.4 % | 4.4 % | 37.9 % |
| OVER-STEER | 13.0 % | 5.8 % | 21.1 % |
| SPEED | 9.4 % | 3.4 % | 11.8 % |
| NO-RECOVERY | 2.7 % | 0.8 % | 12.4 % |
| WRONG-SIDE | 2.2 % | 1.8 % | 3.5 % |
| STOP-LIKE | 0.3 % | 0.3 % | 0.3 % |
