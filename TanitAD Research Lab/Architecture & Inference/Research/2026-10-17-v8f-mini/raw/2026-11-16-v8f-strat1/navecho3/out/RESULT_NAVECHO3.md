# RESULT_NAVECHO3 -- D-STRAT1-TACLAT-F1: cue missing or cue present on the labelled turns

Evidence class: MEASURED (ours; banked T1-kin dumps in `F:/TanitAD-scratch/strat1/read/t1`, zero training, CPU). Tier of every number: T1-kin tick 0 (the lateral head's argmax and the nav_in vector fed at the window's first tick) vs the v9 label lat_cls.
Estimator: episode-cluster bootstrap (taniteval.ci stat / paired stat), B=2000, seed 0, over the episodes holding the labelled turn windows; 'another draw of EPISODES' only (training-run and inference variance are not in it). Windows: 2,399 dump windows (141 episodes); the windows labelled TURN_L or TURN_R number **197 in 30 episodes** (TURN_L 47, TURN_R 150). Artifact: `F:/TanitAD-scratch/strat1/navecho3/out/RESULT_NAVECHO3.json`.

## Reading by the committed rules: **CUE-MISSING**

Rule as committed: CUE-MISSING: share of missed turns with no correct text turn at NOW while v9 names it >= 0.5 in every cell of S1 / N1 x s0 / s1; CUE-PRESENT-MODEL: share with the correct text turn at NOW >= 0.5 in every cell; else MIXED.

Share of each text arm's missed turns (head argmax is not the label) by the cue at NOW, point [95 % episode-cluster CI] (num/den):

| arm | n missed of 197 | no correct text turn while v9 names it (CUE-MISSING) | correct text turn at NOW (CUE-PRESENT) | neither feed names it |
|---|---|---|---|---|
| **S1_s0** (rule cell) | 87 | 0.759 [0.551, 0.910] (66/87) | 0.126 [0.037, 0.258] (11/87) | 0.115 [0.011, 0.294] (10/87) |
| **S1_s1** (rule cell) | 79 | 0.785 [0.579, 0.929] (62/79) | 0.089 [0.021, 0.189] (7/79) | 0.127 [0.012, 0.319] (10/79) |
| **N1_s0** (rule cell) | 90 | 0.733 [0.528, 0.886] (66/90) | 0.156 [0.057, 0.295] (14/90) | 0.111 [0.011, 0.281] (10/90) |
| **N1_s1** (rule cell) | 80 | 0.787 [0.588, 0.922] (63/80) | 0.087 [0.026, 0.174] (7/80) | 0.125 [0.012, 0.314] (10/80) |
| S1d_s0 (beside) | 88 | 0.750 [0.543, 0.901] (66/88) | 0.136 [0.045, 0.270] (12/88) | 0.114 [0.011, 0.286] (10/88) |
| S1d_s1 (beside) | 80 | 0.800 [0.613, 0.932] (64/80) | 0.087 [0.021, 0.183] (7/80) | 0.113 [0.012, 0.282] (9/80) |
| S1_s10 (beside) | 77 | 0.766 [0.556, 0.917] (59/77) | 0.104 [0.032, 0.213] (8/77) | 0.130 [0.013, 0.320] (10/77) |
| S1_s11 (beside) | 82 | 0.780 [0.568, 0.929] (64/82) | 0.098 [0.026, 0.208] (8/82) | 0.122 [0.012, 0.315] (10/82) |
| N1_s10 (beside) | 79 | 0.797 [0.595, 0.938] (63/79) | 0.076 [0.019, 0.162] (6/79) | 0.127 [0.013, 0.316] (10/79) |
| N1_s11 (beside) | 79 | 0.785 [0.581, 0.928] (62/79) | 0.089 [0.027, 0.182] (7/79) | 0.127 [0.012, 0.320] (10/79) |
| S1d_s10 (beside) | 81 | 0.778 [0.576, 0.919] (63/81) | 0.099 [0.031, 0.200] (8/81) | 0.123 [0.013, 0.307] (10/81) |
| S1d_s11 (beside) | 78 | 0.769 [0.567, 0.922] (60/78) | 0.103 [0.028, 0.216] (8/78) | 0.128 [0.013, 0.319] (10/78) |

The four rule cells read 0.733 to 0.787 for CUE-MISSING, each at or above 0.50, so the rule fires at both seeds for both text arms. All 8 beside arms (S1d s0 / s1 / s10 / s11, S1 and N1 at s10 / s11) read the same side of the bar. The lowest lower CI bound among the four cells is 0.528, above 0.50 in every cell as well.

Where the text feed names no correct turn, it names no turn at all, never the other side (the `names the other side` cells are 0 in every table below). Missed turns by the text feed's state at NOW (S1 s0): names the correct turn 11, names the other side 0, names no turn (FOLLOW or no row) 76.

Head recall on the turn windows split by the cue its OWN feed carries at NOW (point [CI] (num/den)):

| arm | recall when its feed names the correct turn | recall when it does not |
|---|---|---|
| C0_s0 (v9) | 0.865 [0.791, 0.919] (154/178) | 0.000 [0.000, 0.000] (0/19) |
| C0_s1 (v9) | 0.927 [0.879, 0.968] (165/178) | 0.000 [0.000, 0.000] (0/19) |
| S1_s0 (text) | 0.902 [0.826, 0.964] (101/112) | 0.106 [0.038, 0.184] (9/85) |
| S1_s1 (text) | 0.938 [0.882, 0.982] (105/112) | 0.153 [0.055, 0.247] (13/85) |
| N1_s0 (text) | 0.875 [0.798, 0.943] (98/112) | 0.106 [0.041, 0.167] (9/85) |
| N1_s1 (text) | 0.938 [0.893, 0.978] (105/112) | 0.141 [0.048, 0.229] (12/85) |

The text arms read the cue they are given: recall is near 0.9 when the text feed names the correct turn and near 0.1 to 0.2 when it does not.  C0 reads its own feed the same way: it recalls 0 of 19 turns its feed does not name at both seeds, and its feed names 178 of 197.

## Q1. Does the text nav name a turn at NOW, on the correct side? Does v9? (labelled turn windows, tick 0)

### TURN_L  (n 47 windows, 9 episodes)

2 x 2 of the correct-turn cue (rows text, columns v9); no cell holds an other-side token, so every non-correct cell is FOLLOW or no row:

| | v9 names the correct turn | v9 does not |
|---|---|---|
| **text names the correct turn** | 24 | 0 |
| **text does not** | 22 | 1 |

* text names the correct turn: 0.511 [0.145, 0.846] (24/47); v9: 0.979 [0.941, 1.000] (46/47); v9 minus text, paired over the same windows: +0.468 [+0.145, +0.816] sep.

### TURN_R  (n 150 windows, 22 episodes)

2 x 2 of the correct-turn cue (rows text, columns v9); no cell holds an other-side token, so every non-correct cell is FOLLOW or no row:

| | v9 names the correct turn | v9 does not |
|---|---|---|
| **text names the correct turn** | 79 | 9 |
| **text does not** | 53 | 9 |

* text names the correct turn: 0.587 [0.383, 0.758] (88/150); v9: 0.880 [0.712, 1.000] (132/150); v9 minus text, paired over the same windows: +0.293 [+0.065, +0.535] sep.

### BOTH_SIDES  (n 197 windows, 30 episodes)

2 x 2 of the correct-turn cue (rows text, columns v9); no cell holds an other-side token, so every non-correct cell is FOLLOW or no row:

| | v9 names the correct turn | v9 does not |
|---|---|---|
| **text names the correct turn** | 103 | 9 |
| **text does not** | 75 | 10 |

* text names the correct turn: 0.569 [0.393, 0.725] (112/197); v9: 0.904 [0.777, 0.995] (178/197); v9 minus text, paired over the same windows: +0.335 [+0.137, +0.533] sep.

Beside the table: the text feed names a turn on the wrong side on 0 of the 197 windows, and v9 on 0 of 197.

## Q2. Where both feeds name the correct turn: the distance channel and the label window

Distance = channel 3 x 100 m (the feed clips at 300 m; no window reads the clip). Time to the turn = distance / ego speed at NOW (dump `v0`, m/s); the label looks 2 s to 8 s ahead.

| side | n both (episodes) | text distance m, median (IQR) | v9 distance m, median (IQR) | text minus v9 m, median (IQR); text longer / shorter | mean diff m, paired [CI] | ego speed m/s, median |
|---|---|---|---|---|---|---|
| TURN_L | 24 (5) | 10.6 (4.1 to 19.4) | 4.5 (0.6 to 11.9) | 7.2 (3.1 to 8.8); 24 / 0 | +5.926 [+3.335, +8.608] sep | 4.8 |
| TURN_R | 79 (12) | 20.7 (11.8 to 33.6) | 13.2 (3.5 to 23.8) | 10.6 (4.3 to 12.5); 74 / 5 | +8.351 [+3.782, +11.844] sep | 7.0 |
| BOTH_SIDES | 103 (17) | 18.3 (8.9 to 32.2) | 11.2 (1.7 to 21.0) | 8.9 (4.1 to 12.4); 98 / 5 | +7.786 [+4.084, +10.634] sep | 6.3 |

| side | text time to turn s, median (IQR); below 2 s / 2-8 s / above 8 s | v9 time to turn s, median (IQR); below 2 s / 2-8 s / above 8 s | median text minus v9 s |
|---|---|---|---|
| TURN_L | 2.87 (1.9 to 4.4); 0.29 / 0.71 / 0.00 (n 21) | 1.32 (0.5 to 2.7); 0.62 / 0.38 / 0.00 (n 21) | 1.42 |
| TURN_R | 3.14 (2.3 to 4.3); 0.18 / 0.82 / 0.00 (n 78) | 2.06 (0.8 to 3.1); 0.50 / 0.50 / 0.00 (n 78) | 1.41 |
| BOTH_SIDES | 3.13 (2.1 to 4.4); 0.20 / 0.80 / 0.00 (n 99) | 1.77 (0.7 to 3.1); 0.53 / 0.47 / 0.00 (n 99) | 1.41 |

Each feed over ALL its correct-turn windows (not only the windows where both are correct):

| side | feed | n | median time to turn s (IQR) | share in 2-8 s | median distance m |
|---|---|---|---|---|---|
| TURN_L | text | 24 | 2.87 (1.9 to 4.4) | 0.71 | 10.6 |
| TURN_L | v9 | 46 | 1.60 (0.6 to 3.3) | 0.37 | 7.6 |
| TURN_R | text | 88 | 3.13 (2.2 to 4.3) | 0.80 | 20.8 |
| TURN_R | v9 | 132 | 2.31 (0.9 to 3.8) | 0.57 | 16.7 |
| BOTH_SIDES | text | 112 | 3.06 (2.1 to 4.3) | 0.79 | 19.0 |
| BOTH_SIDES | v9 | 178 | 2.20 (0.9 to 3.7) | 0.52 | 13.3 |

Reading: where both feeds name the correct turn, the text distance is longer than the v9 distance in 98 of 103 windows (median 8.9 m, about 1.4 s at the ego speed of the window). The text countdown lands 0.80 of the time inside the label's 2-8 s band; the v9 countdown 0.47, with 0.53 below 2 s (the turn is already about to start or running).

## Permutation importance per nav channel (beside)

Classifier: navecho2's (HistGradientBoosting, balanced weights, episode-grouped 5-fold out-of-fold). Importance = out-of-fold macro-F1 minus the macro-F1 with that channel permuted across the windows of each test fold (20 repeats, seed 0; mean with the min-max of the repeats; 'paired' = one repeat read through the paired episode-cluster bootstrap, B 2,000). One-hot channels are collinear, so a single permuted token channel makes impossible rows; the token and distance GROUP rows are the cleaner reading.

### ALL  (base macro-F1: v9 0.705, text 0.594)

| channel / group | v9 importance (min, max) | v9 paired | text importance (min, max) | text paired |
|---|---|---|---|---|
| ch0_FOLLOW | +0.106 (+0.086, +0.124) | +0.109 [+0.063, +0.159] sep | +0.095 (+0.083, +0.107) | +0.087 [+0.019, +0.146] sep |
| ch1_TURN_L | +0.069 (+0.048, +0.082) | +0.052 [-0.004, +0.105] not sep | +0.001 (-0.001, +0.007) | -0.000 [-0.014, +0.013] not sep |
| ch2_TURN_R | +0.177 (+0.163, +0.196) | +0.165 [+0.116, +0.215] sep | +0.167 (+0.154, +0.178) | +0.166 [+0.104, +0.218] sep |
| ch3_distance | +0.030 (-0.009, +0.064) | +0.051 [+0.003, +0.092] sep | +0.155 (+0.137, +0.180) | +0.165 [+0.071, +0.252] sep |
| ch4_distance_valid | +0.031 (-0.009, +0.056) | +0.032 [-0.001, +0.071] not sep | +0.002 (+0.001, +0.007) | +0.004 [-0.005, +0.015] not sep |
| token_group_ch0-2 | +0.324 (+0.289, +0.371) | +0.289 [+0.214, +0.348] sep | +0.217 (+0.197, +0.238) | +0.219 [+0.120, +0.298] sep |
| distance_group_ch3-4 | +0.213 (+0.164, +0.245) | +0.164 [+0.088, +0.230] sep | +0.111 (+0.084, +0.137) | +0.098 [+0.032, +0.161] sep |

### INF_ANY  (base macro-F1: v9 0.534, text 0.554)

| channel / group | v9 importance (min, max) | v9 paired | text importance (min, max) | text paired |
|---|---|---|---|---|
| ch0_FOLLOW | -0.027 (-0.056, -0.004) | - | +0.063 (+0.031, +0.095) | - |
| ch1_TURN_L | +0.016 (-0.019, +0.035) | - | -0.004 (-0.010, +0.006) | - |
| ch2_TURN_R | +0.092 (+0.074, +0.115) | - | +0.169 (+0.150, +0.190) | - |
| ch3_distance | +0.001 (-0.051, +0.053) | - | +0.146 (+0.110, +0.182) | - |
| ch4_distance_valid | -0.028 (-0.081, +0.024) | - | +0.014 (+0.008, +0.022) | - |
| token_group_ch0-2 | +0.232 (+0.174, +0.297) | - | +0.231 (+0.196, +0.270) | - |
| distance_group_ch3-4 | +0.131 (+0.078, +0.174) | - | +0.101 (+0.050, +0.135) | - |

The v9 classifier leans on the side tokens and on the distance pair; the text classifier leans on the TURN_R token, the FOLLOW token and the distance value, and its TURN_L token carries almost no importance (text TURN_L names are rare on the labelled windows). ALL-window importance is dominated by the TURN_R token because TURN_R is the larger turn class.

## Unweighted shuffled-nav control (beside; navecho2's weighted K-SHUFFLE band is not re-read)

Nav permuted across windows, out-of-fold, the navecho2 classifier without class weights, 10 permutations: v9 feed mean 0.318 (min 0.316, max 0.330); text feed mean 0.316 (min 0.316, max 0.317); majority-class score 0.317. Both feeds read within 0.014 of the majority-class score, so the unweighted classifier given noise reads the majority score; the weighted classifier reads below it because balanced weights spread its predictions over the rare classes. navecho2's weighted control read 0.230 (v9) and 0.145 (text) and its band stays as stated there (FAILED).

## Controls

* K-SELF: a feed's correct-turn indicator paired with itself reads 0.0 (pass True); the test also sends a mutant to RED.
* K-REACH: a token equal to the label reads CORRECT on every turn window: True.
* K-FEED (nav_in identical across every arm of a feed), K-LANDED (17 head macro-F1 values equal the landed `fam/out/RESULT_FAM_EXTRA.json` to 1e-12) and K-BASE (the unpermuted fold models reproduce navecho2.crossfit_hgb exactly): problems list [] (empty = pass).

## Limits

* The question lives on 197 windows in 30 episodes (TURN_L 47 windows in 9 episodes). Windows of one episode are consecutive, so the intervals are wide and the TURN_L cell is thin; the episode-cluster bootstrap carries that width.
* The estimator answers 'another draw of EPISODES' only; the shares agree across 4 training seeds and 3 text families (all beside rows), which is the training-run check available here, not a replicate arm.
* The text arms see the feed through the head, and the shares describe the feed at NOW; the dump holds no earlier-frame nav, so a cue named a few frames before NOW is not visible to this reading.
* The importance tables are a descriptive reading of the 5-channel classifier, not of the driving model.
