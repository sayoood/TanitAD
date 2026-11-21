### T1. Channel coverage per window

| channel | TRAIN n | TRAIN % | EVAL139 n | EVAL139 % |
|---|---|---|---|---|
| windows | 746,946 | 100.00 % | 23,772 | 100.00 % |
| joined to a v8 record | 746,946 | 100.00 % | 23,260 | 97.85 % |
| tactical lat/lon GT (in band) | 173,409 | 23.22 % | 5,404 | 22.73 % |
| >=1 goal token SCORED (w>0), as the workers saw it | 173,409 | 23.22 % | 5,404 | 22.73 % |
| >=1 goal token POSITIVE | 173,409 | 23.22 % | 5,404 | 22.73 % |
| nav token fed (nav_valid) | 746,946 | 100.00 % | 23,772 | 100.00 % |
| nav token = left or right | 276,434 | 37.01 % | 8,726 | 36.71 % |
| max-speed ceiling fed | 746,946 | 100.00 % | 23,772 | 100.00 % |
| agent boxes labelled | 719,739 | 96.36 % | 22,663 | 95.33 % |
| agent labelled AND >=1 agent | 702,773 | 94.09 % | 22,081 | 92.89 % |
| agent labelled and EMPTY (clear) | 16,966 | 2.27 % | 582 | 2.45 % |
| 3-D cuboid on >=1 agent | 702,773 | 94.09 % | 22,081 | 92.89 % |
| SAM3 10 cm map label (trainer census 'ok') | 746,946 | 100.00 % | 23,430 | 98.56 % |
| full 6 s of future poses in the clip | 576,848 | 77.23 % | 18,359 | 77.23 % |
| tactical GT AND full 6 s future | 173,409 | 23.22 % | 5,404 | 22.73 % |

EVAL139 episode-cluster bootstrap 95 % CI (139 clips, B=2000): tactical GT 22.1-23.2 %; agent labelled 92.5-97.9 %; map label 96.4-100.0 %; full6 77.2-77.3 %

TRAIN: 746,946 windows over 4,369 clips (windows per clip min/median/max 161/171/179); clips with >=1 tactical window 4,369; tactical windows per clip min/median/max 39/40/41. EVAL139: 23,772 windows over 139 clips; clips with >=1 tactical window 136.

### T2. Tactical lateral / longitudinal classes (windows carrying the class)

**lateral (`lat_v7`)**

| class | TRAIN windows | % of all | % of in-band | TRAIN clips | EVAL windows | % of all | % of in-band |
|---|---|---|---|---|---|---|---|
| LANE_KEEP | 112,092 | 15.01 % | 64.6 % | 2,824 | 3,616 | 15.21 % | 66.9 % |
| LANE_CHANGE_L | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| LANE_CHANGE_R | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| ABORT_LC | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| NUDGE_L | 18,409 | 2.46 % | 10.6 % | 464 | 517 | 2.17 % | 9.6 % |
| NUDGE_R | 22,377 | 3.00 % | 12.9 % | 564 | 794 | 3.34 % | 14.7 % |
| TURN_L | 10,716 | 1.43 % | 6.2 % | 270 | 160 | 0.67 % | 3.0 % |
| TURN_R | 9,815 | 1.31 % | 5.7 % | 247 | 317 | 1.33 % | 5.9 % |

**longitudinal (`lon_v7`)**

| class | TRAIN windows | % of all | % of in-band | TRAIN clips | EVAL windows | % of all | % of in-band |
|---|---|---|---|---|---|---|---|
| FOLLOW | 6,232 | 0.83 % | 3.6 % | 157 | 357 | 1.50 % | 6.6 % |
| CRUISE | 47,484 | 6.36 % | 27.4 % | 1,197 | 1,668 | 7.02 % | 30.9 % |
| YIELD_MERGE | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| BRAKE_TO | 34,568 | 4.63 % | 19.9 % | 871 | 876 | 3.69 % | 16.2 % |
| CREEP | 4,729 | 0.63 % | 2.7 % | 119 | 119 | 0.50 % | 2.2 % |
| HOLD | 4,007 | 0.54 % | 2.3 % | 101 | 80 | 0.34 % | 1.5 % |
| ADAPT_SPEED_FOR_CURVE | 37,721 | 5.05 % | 21.8 % | 950 | 874 | 3.68 % | 16.2 % |
| ACCELERATE | 38,668 | 5.18 % | 22.3 % | 974 | 1,430 | 6.02 % | 26.5 % |

### T3. The 22 goal tokens: positives and scored cells (windows)

| token | TRAIN positive windows | % of all | pos. clips | TRAIN scored (as the workers saw it) | % of all | TRAIN scored (train-blob census state) | EVAL positive | EVAL scored |
|---|---|---|---|---|---|---|---|---|
| FOLLOW_LANE | 138,161 | 18.497 % | 3,481 | 173,409 | 23.22 % | 173,409 | 4,408 | 5,404 |
| TURN_L | 10,716 | 1.435 % | 270 | 173,409 | 23.22 % | 173,409 | 160 | 5,404 |
| TURN_R | 9,815 | 1.314 % | 247 | 173,409 | 23.22 % | 173,409 | 317 | 5,404 |
| YIELD_FOR_TURN_L | 753 | 0.101 % | 19 | 173,409 | 23.22 % | 173,409 | 40 | 5,404 |
| YIELD_FOR_TURN_R | 753 | 0.101 % | 19 | 173,409 | 23.22 % | 173,409 | 0 | 5,404 |
| YIELD | 23,468 | 3.142 % | 591 | 23,468 | 3.14 % | 23,468 | 678 | 678 |
| STOP_POINT | 11,542 | 1.545 % | 291 | 173,409 | 23.22 % | 173,409 | 279 | 5,404 |
| SPEED_BAND | 173,409 | 23.216 % | 4,369 | 173,409 | 23.22 % | 173,409 | 5,404 | 5,404 |
| CORRIDOR_OFFSET | 32,532 | 4.355 % | 820 | 32,532 | 4.36 % | 32,532 | 913 | 913 |
| EVADE_IN_CORRIDOR | 9,414 | 1.260 % | 237 | 10,168 | 1.36 % | 10,168 | 239 | 279 |
| OVERTAKE_VEHICLE | 754 | 0.101 % | 19 | 139,152 | 18.63 % | 139,152 | 40 | 4,448 |
| MERGE | 3,097 | 0.415 % | 78 | 141,258 | 18.91 % | 141,258 | 200 | 4,608 |
| GAP_TARGET | 14,322 | 1.917 % | 361 | 14,322 | 1.92 % | 14,322 | 436 | 436 |
| REACT_ON_ONCOMING | 12,949 | 1.734 % | 326 | 12,949 | 1.73 % | 12,949 | 397 | 397 |
| TAKE_EXIT_L | 833 | 0.112 % | 21 | 14,897 | 1.99 % | 14,897 | 0 | 515 |
| TAKE_EXIT_R | 4,964 | 0.665 % | 125 | 16,276 | 2.18 % | 16,276 | 198 | 358 |
| TRAFFIC_LIGHT_REACT | 673 | 0.090 % | 17 | 29,701 | 3.98 % | 29,701 | 0 | 1,033 |
| TRAFFIC_LIGHT_REACT_RED | 14,022 | 1.877 % | 353 | 29,701 | 3.98 % | 29,701 | 317 | 1,033 |
| TRAFFIC_LIGHT_REACT_YELLOW | 876 | 0.117 % | 22 | 29,701 | 3.98 % | 29,701 | 120 | 1,033 |
| TRAFFIC_LIGHT_REACT_GREEN | 14,130 | 1.892 % | 356 | 29,701 | 3.98 % | 29,701 | 596 | 1,033 |
| LANE_CHANGE_L | 872 | 0.117 % | 22 | 173,409 | 23.22 % | 12,145 | 0 | 5,404 |
| LANE_CHANGE_R | 557 | 0.075 % | 14 | 11,244 | 1.51 % | 11,244 | 40 | 357 |

Two label-module states are tabulated for TRAIN (see FINDINGS): windows whose scored-cell count differs between them: 161,264; cells differing by token: {"LANE_CHANGE_L": 161264}.

### T4. In-band fraction as a function of t_now (1-s bins over the clip)

| t_now bin (s) | TRAIN windows | TRAIN in-band | TRAIN % | EVAL windows | EVAL in-band | EVAL % |
|---|---|---|---|---|---|---|
| [0,1) | 1,760 | 0 | 0.0 % | 58 | 0 | 0.0 % |
| [1,2) | 43,097 | 0 | 0.0 % | 1,374 | 0 | 0.0 % |
| [2,3) | 43,318 | 0 | 0.0 % | 1,380 | 0 | 0.0 % |
| [3,4) | 43,387 | 0 | 0.0 % | 1,378 | 0 | 0.0 % |
| [4,5) | 43,387 | 0 | 0.0 % | 1,376 | 0 | 0.0 % |
| [5,6) | 43,313 | 0 | 0.0 % | 1,381 | 0 | 0.0 % |
| [6,7) | 43,339 | 43,339 | 100.0 % | 1,382 | 1,352 | 97.8 % |
| [7,8) | 43,362 | 43,362 | 100.0 % | 1,381 | 1,351 | 97.8 % |
| [8,9) | 43,336 | 43,336 | 100.0 % | 1,384 | 1,354 | 97.8 % |
| [9,10) | 43,363 | 43,363 | 100.0 % | 1,377 | 1,347 | 97.8 % |
| [10,11) | 43,369 | 9 | 0.0 % | 1,382 | 0 | 0.0 % |
| [11,12) | 43,494 | 0 | 0.0 % | 1,382 | 0 | 0.0 % |
| [12,13) | 43,512 | 0 | 0.0 % | 1,381 | 0 | 0.0 % |
| [13,14) | 43,531 | 0 | 0.0 % | 1,386 | 0 | 0.0 % |
| [14,15) | 43,521 | 0 | 0.0 % | 1,383 | 0 | 0.0 % |
| [15,16) | 43,476 | 0 | 0.0 % | 1,385 | 0 | 0.0 % |
| [16,17) | 43,386 | 0 | 0.0 % | 1,381 | 0 | 0.0 % |
| [17,18) | 43,271 | 0 | 0.0 % | 1,378 | 0 | 0.0 % |
| [18,19) | 7,723 | 0 | 0.0 % | 243 | 0 | 0.0 % |
| [19,20) | 1 | 0 | 0.0 % | 0 | 0 | n/a % |

### T5. Turn windows (|dyaw over [0,6] s| >= 30 deg, windows with the full 6 s in the clip) and the tactical lateral label

| turn windows | TRAIN n | lat GT present | % present | side agrees where present | EVAL n | lat GT present | % present | side agrees where present |
|---|---|---|---|---|---|---|---|---|
| all_turn | 88,238 | 30,931 | 35.1 % | 65.2 % | 2,602 | 814 | 31.3 % | 53.7 % |
| left | 43,544 | 15,329 | 35.2 % | 65.5 % | 929 | 252 | 27.1 % | 49.6 % |
| right | 44,694 | 15,602 | 34.9 % | 64.9 % | 1,673 | 562 | 33.6 % | 55.5 % |

10-30 deg: TRAIN n 86,122, lat GT present 27.7 %; EVAL n 2,573, 27.9 %

< 10 deg: TRAIN n 402,488, lat GT present 29.5 %; EVAL n 13,184, 29.4 %

Route-following package definition (terminal heading of the slot-50 -> 60 segment, >= 30 deg, path >= 5 m) on ALL windows: TRAIN GT-turn 81,703 (L 40,109, R 41,594), lat IGNORE on 64.4 %; EVAL139 GT-turn 2,317 (L 849, R 1,468), lat IGNORE on 69.1 %.

### T6. The fed nav token against where a turn actually is

| quantity | TRAIN | EVAL139 |
|---|---|---|
| windows with nav token left / right / follow | 133,190 / 143,244 / 470,512 | 2,229 / 6,497 / 15,046 |
| clips with nav token left / right / follow | 779 / 838 / 2752 | 13 / 38 / 88 |
| L/R windows (denominator) | 276,434 | 8,726 |
| L/R windows, NO turn START within the next 6 s (labels) | 205,292 (74.3 %) | 6,996 (80.2 %) |
| ... and none IN PROGRESS either (6 s) | 172,389 (62.4 %) | 6,329 (72.5 %) |
| L/R windows, NO turn START within the next 10 s (labels) | 165,174 (59.8 %) | 5,620 (64.4 %) |
| ... and none IN PROGRESS either (10 s) | 134,509 (48.7 %) | 5,013 (57.4 %) |
| L/R windows with 6 s of future in the clip: NO realised >= 30 deg turn in 6 s (poses) | 153,466 of 213,494 (71.9 %) | 5,434 of 6,738 (80.6 %) |
| ... and not even 10 deg | 54.3 % | 62.6 % |
| realised >= 30 deg turn in the next 6 s: windows | 88,238 | 2,602 |
| ... of which the fed token is FOLLOW | 28,210 (32.0 %) | 1,298 (49.9 %) |
| ... of which the fed token matches the turn side | 53,866 (61.0 %) | 1,241 (47.7 %) |
| ... of which the fed token is the WRONG side | 6,162 | 63 |
| median / p5-p95 seconds to the next labelled turn start (L/R windows that have one) | 9.8 (1.0-27.7) | 11.7 (1.4-28.6) |

### T7. The per-clip speed ceiling against the window's own realised speed ([+2,+6] s ahead; windows with the full 6 s)

| fed ceiling | TRAIN windows | realised max EXCEEDS ceiling | realised bin BELOW fed bin | same bin | EVAL windows | exceeds | below | same |
|---|---|---|---|---|---|---|---|---|
| all fed bins | 576,848 | 57,535 (10.0 %) | 41,985 (7.3 %) | 488,458 (84.7 %) | 18,359 | 8.7 % | 7.5 % | 84.6 % |
| fed 30 km/h | 218,540 | 32,583 (14.9 %) | 0 (0.0 %) | 185,957 (85.1 %) | 6,210 | 15.7 % | 0.0 % | 84.3 % |
| fed 50 km/h | 204,826 | 12,801 (6.2 %) | 29,369 (14.3 %) | 162,656 (79.4 %) | 6,475 | 5.7 % | 13.8 % | 80.6 % |
| fed 100 km/h | 125,418 | 1,021 (0.8 %) | 10,961 (8.7 %) | 113,436 (90.4 %) | 4,752 | 2.3 % | 8.8 % | 88.8 % |
| fed 120 km/h | 28,064 | 11,130 (39.7 %) | 1,655 (5.9 %) | 26,409 (94.1 %) | 922 | 15.1 % | 7.3 % | 92.7 % |

Margin: realised max exceeds the fed ceiling by > 5 km/h on 4.9 % of TRAIN windows (28,400), by > 10 km/h on 2.1 % (11,893); it is more than 20 km/h BELOW the fed ceiling on 29.0 % (167,369); median realised-minus-ceiling -12.4 km/h (EVAL139: >5 km/h 3.4 %, >10 km/h 1.4 %).

In-band windows only (the span the ceiling's own definition describes): TRAIN n 173,409: exceeds 4.7 %, below 2.7 %, same 94.5 %; EVAL n 5,404: exceeds 3.5 %, below 4.0 %, same 93.2 %. Current speed v0 above the fed ceiling: TRAIN 7.6 %, EVAL 6.8 % of windows.


### T8. The per-clip constants against t_now (2-s bins; the nav token and the ceiling are ONE value per clip on every window)


| t_now bin | TRAIN windows | tactical GT | nav = L/R | L/R with a turn start <= 6 s (labels) | L/R with NO >=30 deg turn in 6 s (poses) | realised turns: token matches | realised turns: token FOLLOW | ceiling exceeded | realised bin below fed | same bin | EVAL L/R with turn start <= 6 s | EVAL ceiling exceeded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [0,2) | 44,857 | 0.0 % | 36.9 % | 0.0 % | 85.3 % | 26.9 % | 52.9 % | 15.6 % | 13.0 % | 73.2 % | 0.0 % | 14.2 % |
| [2,4) | 86,705 | 0.0 % | 37.0 % | 24.8 % | 79.6 % | 42.2 % | 43.2 % | 13.0 % | 11.5 % | 77.3 % | 10.6 % | 11.9 % |
| [4,6) | 86,700 | 0.0 % | 37.0 % | 36.4 % | 71.2 % | 62.7 % | 31.4 % | 8.8 % | 8.5 % | 84.6 % | 22.6 % | 8.7 % |
| [6,8) | 86,701 | 100.0 % | 37.0 % | 44.0 % | 64.5 % | 73.0 % | 25.2 % | 4.4 % | 3.2 % | 94.3 % | 30.4 % | 3.0 % |
| [8,10) | 86,699 | 100.0 % | 37.0 % | 27.7 % | 64.5 % | 70.4 % | 27.4 % | 5.1 % | 2.2 % | 94.7 % | 28.4 % | 3.8 % |
| [10,12) | 86,863 | 0.0 % | 37.0 % | 23.6 % | 70.1 % | 64.1 % | 30.7 % | 10.4 % | 5.9 % | 85.7 % | 21.7 % | 7.5 % |
| [12,14) | 87,043 | 0.0 % | 37.0 % | 22.0 % | 73.9 % | 61.2 % | 29.1 % | 14.4 % | 9.0 % | 78.5 % | 15.4 % | 13.2 % |
| [14,16) | 86,997 | 0.0 % | 37.0 % | 21.4 % | 76.0 % | 59.8 % | 29.8 % | 15.8 % | 10.5 % | 75.5 % | 16.4 % | 16.5 % |
| [16,18) | 86,657 | 0.0 % | 37.0 % | 19.8 % | n/a % | n/a % | n/a % | n/a % | n/a % | n/a % | 22.6 % | n/a % |


**In-band windows, lateral label against the window's own realised heading, by distance from the anchor**


| window distance from the 8.0 s anchor | TRAIN in-band | TRAIN turn windows | label = TURN_x | label = LANE_KEEP | label side == realised side | EVAL turn windows | EVAL side agree |
|---|---|---|---|---|---|---|---|
| abs(t_now-8) in [0,1] | 86,689 | 15,789 | 57.7 % | 34.6 % | 64.8 % | 410 | 53.9 % |
| abs(t_now-8) in [1,2] | 86,720 | 15,142 | 55.2 % | 32.0 % | 65.6 % | 404 | 53.5 % |