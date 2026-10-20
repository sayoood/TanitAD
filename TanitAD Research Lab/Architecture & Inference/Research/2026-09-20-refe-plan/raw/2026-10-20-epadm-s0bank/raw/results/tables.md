#### S1. THE THREE COMMITTED BARS (SIMULATED; primary NaN convention = real sigma; the fail-closed convention in brackets). Wilson 95 %. `MET` needs Base C AND Base X.

| geometry | base (seeds) | A1-r1 false refusal (bar: upper <= 5.0 %) | A2 (1.05) false refusal (bar: upper <= 5.0 %) | DAC power x1.25, margin 1.05 (bar: point >= 70 %) |
|---|---|---|---|---|
| GS41 | C carve-B clones (3) | 9.9 % [8.7, 11.3] (198/2000) | 2.5 % [1.9, 3.2] (49/2000) [37.5 %] | 7.6 % [6.6, 8.9] [7.6 %] |
| GS41 | X S0 stored G25 (2) | 2.4 % [1.8, 3.2] (48/2000) | 1.9 % [1.4, 2.7] (39/2000) [1.9 %] | 31.6 % [29.6, 33.6] [31.6 %] |
| GS117 | C carve-B clones (3) | 8.8 % [7.6, 10.1] (176/2000) | 2.9 % [2.3, 3.7] (58/2000) [8.8 %] | 13.8 % [12.4, 15.4] [13.8 %] |
| GS117 | X S0 stored G25 (2) | 2.1 % [1.5, 2.8] (41/2000) | 2.4 % [1.8, 3.1] (47/2000) [2.4 %] | 60.4 % [58.2, 62.5] [60.4 %] |

| geometry | A1-r1 FR | A2 (1.05) FR | DAC power x1.25 | all three MET | R-NAN (status differs between conventions) | R-GEN (T vs A, A1 FR, pp) | R-SEED (C vs X, pp: A1 FR / DAC power) |
|---|---|---|---|---|---|---|---|
| GS41 | MISSED | MET | MISSED | **False** | True | 9.9 | 7.5 / 23.9 |
| GS117 | MISSED | MET | MISSED | **False** | True | 8.8 | 6.7 / 46.6 |

#### S2. DESCRIPTIVE (not bars): A1 power, A2 power by head and margin, per-seed A1 false refusal, generator T cross-check

| geometry | base-gen | A1 power +0.135 | A1 power +0.05 | A2 power DAC x1.25 @1.00 / @1.05 / @1.10 | DAC x1.5 @1.05 | TTC x1.25 @1.05 | NC x1.25 @1.05 | NC x1.5 @1.05 | A1 FR per seed |
|---|---|---|---|---|---|---|---|---|---|
| GS41 | C-A | 100.0 % [99.6, 100.0] | 100.0 % [99.6, 100.0] | 10.0 % [8.7, 11.3] / 7.6 % [6.6, 8.9] / 5.7 % [4.7, 6.7] | 20.7 % [18.3, 23.3] | 6.1 % [4.8, 7.8] | 0.9 % [0.5, 1.7] | 1.0 % [0.5, 1.8] | 7.0 / 4.3 / 6.8 % |
| GS41 | C-T | n/a | n/a | 0.0 % [0.0, 0.4] / 0.0 % [0.0, 0.4] / 0.0 % [0.0, 0.4] | n/a | n/a | n/a | n/a | 0.0 / 0.0 / 0.0 % |
| GS41 | X-A | 100.0 % [99.6, 100.0] | 100.0 % [99.6, 100.0] | 47.9 % [45.7, 50.1] / 31.6 % [29.6, 33.6] / 18.4 % [16.7, 20.1] | 79.4 % [76.8, 81.8] | 24.1 % [21.6, 26.8] | 10.1 % [8.4, 12.1] | 34.9 % [32.0, 37.9] | 2.4 / 2.4 % |
| GS117 | C-A | 100.0 % [99.6, 100.0] | 100.0 % [99.6, 100.0] | 20.2 % [18.5, 22.1] / 13.8 % [12.4, 15.4] / 9.5 % [8.3, 10.9] | 43.7 % [40.7, 46.8] | 12.4 % [10.5, 14.6] | 1.5 % [0.9, 2.5] | 2.4 % [1.6, 3.5] | 6.2 / 4.3 / 6.0 % |
| GS117 | C-T | n/a | n/a | 1.0 % [0.5, 1.8] / 0.0 % [0.0, 0.4] / 0.0 % [0.0, 0.4] | n/a | n/a | n/a | n/a | 0.0 / 0.0 / 0.0 % |
| GS117 | X-A | 100.0 % [99.6, 100.0] | 100.0 % [99.6, 100.0] | 79.7 % [77.8, 81.4] / 60.4 % [58.2, 62.5] / 38.0 % [35.8, 40.1] | 97.7 % [96.6, 98.5] | 36.7 % [33.8, 39.7] | 20.8 % [18.4, 23.4] | 63.8 % [60.8, 66.7] | 1.9 / 2.1 % |

#### S3. CONTROLS and flags

* `C-GS117-A`: controls_ok **True**; PC1a True, PC1b True (max |obs/pred - 1| at the base = 8.881784197001252e-16), PC2 True, PC4 True, PC5 True, PC6 True; REPORTED A2 ratio means after the redraw {"DAC": [1.0173834908453763, 1.0198883191890258, 1.0177999542779281], "NC": [0.9680231844690058, 0.9680567626100179, 0.9680023034281265], "TTC": [0.9996729552172531, 0.9997169608300517, 0.9997476447198239]}
* `C-GS41-A`: controls_ok **True**; PC1a True, PC1b True (max |obs/pred - 1| at the base = 5.551115123125783e-16), PC2 True, PC4 True, PC5 True, PC6 True; REPORTED A2 ratio means after the redraw {"DAC": [1.0166445519844332, 1.0170280265243592, 1.0175415128948577], "NC": [0.637407915845133, 0.6374476845448245, 0.6374624803629827], "TTC": [1.0016359044441696, 1.0017762736236608, 1.0017705058630086]}
* `X-GS117-A`: controls_ok **True**; PC1a True, PC1b True (max |obs/pred - 1| at the base = 4.440892098500626e-16), PC2 True, PC4 True, PC5 True, PC6 True; REPORTED A2 ratio means after the redraw {"DAC": [1.0062170613171553, 1.0061652079731587], "NC": [1.0111952550225516, 1.0107328120677317], "TTC": [1.000935832713705, 1.0006198737747818]}
* `X-GS41-A`: controls_ok **True**; PC1a True, PC1b True (max |obs/pred - 1| at the base = 2.220446049250313e-16), PC2 True, PC4 True, PC5 True, PC6 True; REPORTED A2 ratio means after the redraw {"DAC": [1.003133276412334, 1.0031928871760856], "NC": [0.9989572300390357, 0.9990200411714524], "TTC": [0.9987556044841148, 0.9987771963507494]}
* PC3 (shortcut = `analyze_candidate`): True; PC8 (Base X code = the R1 simulator, bit for bit vs R2-CONFIRM): True; overall controls_ok **True**.
* R-EVENTS / R-DOSE (Base C pool, per seed; `C-GS117-A`): seed 0: unsafe picks NC 1 / DAC 10 / TTC 15, changed 108 (27.0 %); seed 1: unsafe picks NC 1 / DAC 11 / TTC 15, changed 104 (26.0 %); seed 2: unsafe picks NC 1 / DAC 10 / TTC 15, changed 106 (26.5 %)
* R-NAN counts (`C-GS117-A`, CAL, replicate-seed pairs whose base has no observed unsafe pick; n = 2000 x 3): {"DAC": 0, "NC": 363, "TTC": 0}
* R-NAN counts (`C-GS41-A`, CAL, replicate-seed pairs whose base has no observed unsafe pick; n = 2000 x 3): {"DAC": 0, "NC": 2181, "TTC": 0}
* R-NAN counts (`X-GS117-A`, CAL, replicate-seed pairs whose base has no observed unsafe pick; n = 2000 x 2): {"DAC": 0, "NC": 0, "TTC": 0}
* R-NAN counts (`X-GS41-A`, CAL, replicate-seed pairs whose base has no observed unsafe pick; n = 2000 x 2): {"DAC": 0, "NC": 0, "TTC": 0}
* R-DUP (mean distinct logs of N under generator A, 2,000 outer draws): GS41: 26.1, GS117: 74.2
