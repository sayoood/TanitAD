### P3-a SPEED oracle on navhard NC-zero scenes (n scored 889 of 889; the same geometric path driven at s x the planned speed; exact reactive devkit re-score)

| speed scale | NC cleared n (share [95 % log CI]) | cleared and DAC not worse | median 4-s distance m: cleared plans / all scaled plans (banked plans / PDM-closed reference) | cleared plans that still travel >= half the reference distance |
|---|---|---|---|---|
| x0.8 | 139 of 889 (15.6 % [12.6, 18.9]) | 138 | 17.3 / 17.0 (21.3 / 11.3) | 135 |
| x0.6 | 272 of 889 (30.6 % [26.0, 35.1]) | 269 | 13.2 / 12.9 (21.3 / 11.3) | 251 |
| x0.4 | 408 of 889 (45.9 % [40.3, 51.0]) | 402 | 8.6 / 8.7 (21.3 / 11.3) | 303 |
| any of the three (oracle) | 409 of 889 (46.0 % [40.4, 51.2]) | | | |

Not fixable by any speed (STOP also NC-zero: the initial-overlap scenes): 200; among the scenes where STOP passes, the oracle clears 404 of 689 (58.6 %).

By NC class (from the re-score of RESULT s5):

| class | n | cleared by any scale | x0.8 | x0.6 | x0.4 |
|---|---|---|---|---|---|
| ACTIVE-FRONT-VEHICLE | 504 | 240 | 82 | 168 | 239 |
| STOPPED-TRACK-VEHICLE/OBJECT | 307 | 137 | 50 | 87 | 137 |
| LATERAL-AFTER-LANE-DEPARTURE | 27 | 13 | 3 | 6 | 13 |
| ACTIVE-FRONT-VRU | 24 | 14 | 3 | 7 | 14 |
| INITIAL-OVERLAP | 19 | 0 | 0 | 0 | 0 |
| STOPPED-TRACK-VRU | 8 | 5 | 1 | 4 | 5 |

Official two-stage EPDMS if the per-token best of {banked, x0.8, x0.6, x0.4} were taken on the NC-zero scenes: 0.2269 -> 0.2880 (delta +0.0611; ESTIMATED upper bound, EC and stage-2 weights held; 889 scenes edited).

### P3-b LATERAL / SPEED oracle on navhard DAC-zero scenes (n scored 1563 of 1563; map-only DAC test of the edited plan; centreline = the cached PDM-Closed route centreline)

| fix | what it does | DAC cleared n (share [95 % log CI]) | in clean-path scenes | in reference-also-fails scenes | stage 1 | stage 2 | cleared and DDC ok |
|---|---|---|---|---|---|---|---|
| B075 | shift toward the centreline by <= 0.75 m | 204 of 1563 (13.1 % [10.6, 15.9]) | 195 of 1136 | 9 of 427 | 27 of 145 | 177 of 1418 | 149 |
| B150 | shift toward the centreline by <= 1.5 m | 270 of 1563 (17.3 % [14.5, 20.3]) | 267 of 1136 | 3 of 427 | 32 of 145 | 238 of 1418 | 197 |
| FULL | put every point ON the centreline at the plan's own along-route progress (route-follow oracle) | 803 of 1563 (51.4 % [46.4, 57.4]) | 795 of 1136 | 8 of 427 | 115 of 145 | 688 of 1418 | 746 |
| V80 | same path at 0.8 x speed | 231 of 1563 (14.8 % [13.2, 16.8]) | 219 of 1136 | 12 of 427 | 41 of 145 | 190 of 1418 | 189 |
| V60 | same path at 0.6 x speed | 425 of 1563 (27.2 % [24.2, 31.0]) | 405 of 1136 | 20 of 427 | 77 of 145 | 348 of 1418 | 370 |
| V40 | same path at 0.4 x speed | 594 of 1563 (38.0 % [34.2, 42.8]) | 559 of 1136 | 35 of 427 | 99 of 145 | 495 of 1418 | 536 |

Union (FULL snap or x0.6 or x0.4 speed): 934 of 1563 (59.8 %). Control: the identity plan stays DAC-zero on all scenes: True.

Official two-stage EPDMS if DAC were set to 1 on the scenes a fix clears (other sub-scores and weights held; ESTIMATED upper bound):

| fix | official EPDMS | delta vs A1 0.2269 |
|---|---|---|
| B075 | 0.2576 | +0.0307 |
| B150 | 0.2639 | +0.0369 |
| FULL | 0.3403 | +0.1134 |
| V60 | 0.2943 | +0.0674 |

### P3-c NAVTEST on the navsim-1.1 devkit (n scored 2491 of the DAC-zero 1460 + NC-zero 1008 + controls)

Controls: exact re-score reproduces the banked row on 100.0 % of scenes (max abs 0); the identity snap reproduces the banked DAC on 100.0 %; PASS = True.

DAC-zero first non-drivable instant (s): (3,4]: 697, (2,3]: 429, (1,2]: 271, (0,1]: 63; t = 0 violations 0; clean path exists (reference passes) 99.3 %; reference also fails 10.

NC-zero first at-fault event classes:

| class | n | share |
|---|---|---|
| STOPPED-TRACK-VEHICLE/OBJECT | 597 | 59.2 % |
| ACTIVE-FRONT-VEHICLE | 371 | 36.8 % |
| LATERAL-AFTER-LANE-DEPARTURE | 21 | 2.1 % |
| ACTIVE-FRONT-VRU | 12 | 1.2 % |
| STOPPED-TRACK-VRU | 7 | 0.7 % |

Front collisions (active-front + stopped-track): 97.9 % of NC-zero.

| oracle | clears | share [95 % log CI] |
|---|---|---|
| speed S0.8 on NC-zero | 517 of 1008 | 51.3 % [46.1, 56.9] |
| speed S0.6 on NC-zero | 819 of 1008 | 81.2 % [76.9, 85.7] |
| speed S0.4 on NC-zero | 941 of 1008 | 93.4 % [91.4, 95.4] |
| speed any of the three on NC-zero | 944 of 1008 | 93.7 % [91.8, 95.6] |
| B075 on DAC-zero | 350 of 1460 | 24.0 % [20.8, 27.3] |
| B150 on DAC-zero | 402 of 1460 | 27.5 % [24.2, 31.0] |
| FULL on DAC-zero | 1250 of 1460 | 85.6 % [81.0, 89.9] |
| V80 on DAC-zero | 463 of 1460 | 31.7 % [29.0, 34.5] |
| V60 on DAC-zero | 795 of 1460 | 54.5 % [50.8, 58.3] |
| V40 on DAC-zero | 980 of 1460 | 67.1 % [63.3, 70.9] |

PDMS x100 (ESTIMATED upper bounds; banked A1 71.88, STOP 61.82): oracle speed on NC-zero 78.82; DAC := 1 where the FULL snap clears 76.95.
