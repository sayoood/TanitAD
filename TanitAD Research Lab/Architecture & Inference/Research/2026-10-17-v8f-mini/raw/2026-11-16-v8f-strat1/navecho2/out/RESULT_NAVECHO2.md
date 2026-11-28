# RESULT_NAVECHO2 -- D-STRAT1-TACLAT-F1: flexible nav-only classifier

Evidence class: MEASURED (ours; banked T1-kin dumps in `F:/TanitAD-scratch/strat1/read/t1`, zero training of the driving model, CPU). Tier of every number: T1-kin tick 0 (lateral head argmax at the first tick of each window, dead-reckoned nav, logged scene) vs the v9 label lat_cls (5-way).
Classifier: sklearn HistGradientBoostingClassifier (max_iter 200, learning_rate 0.1, max_leaf_nodes 15, class_weight balanced, random_state 0, early_stopping off; fixed before scoring). Input = the 5 nav_in channels fed at the first tick (3 token one-hot, distance/100, distance-valid). The dump holds only the vector fed at NOW (`v8f_t1_eval.py:370`), so no earlier-frame nav history exists to use. Episode-grouped 5-fold cross-fitting (fold sizes 28/28/28/28/29 episodes), out-of-fold only, never tuned on the scored windows.
Estimator: episode-cluster bootstrap (`taniteval.ci`, paired for head minus classifier and for v9 minus text classifier), B 2,000, seed 0; answers "another draw of EPISODES" only, not training-run or inference variance (floor F = |C0_rep - C0_s0| = 0.0184 macro-F1 on ALL). Windows: 2,399 (2,107 labelled, 139 labelled episodes of 141). Artifact: `F:/TanitAD-scratch/strat1/navecho2/out/RESULT_NAVECHO2.json`.

## Reading by the committed rules: **FEED-INFO** (point-estimate rule met; one control missed its band)

| quantity (ALL labelled windows, n 2,107) | s0 | s1 |
|---|---|---|
| C0 head macro-F1 | 0.816 [0.752, 0.862] | 0.862 [0.809, 0.907] |
| flexible classifier on the v9 feed | 0.705 [0.636, 0.758] | same classifier |
| ratio classifier / head (ECHO-CONT needs >= 0.90) | **0.864** | **0.818** |
| head minus classifier, paired | +0.111 [+0.056, +0.168] sep | +0.157 [+0.112, +0.217] sep |
| flexible classifier on the text feed | 0.594 [0.490, 0.679] | same classifier |
| v9 minus text classifier, paired, same windows | +0.111 [-0.005, +0.226] not separated | same |
| C0 - S1 head gap | 0.140 | 0.150 |
| 2/3 of the gap (FEED-INFO bar) | 0.093 | 0.100 |

* **ECHO-CONT is not met.** The flexible classifier reads 86.4 % (s0) and 81.8 % (s1) of the C0 head, below 90 % at both seeds, and the head sits above it by a separated +0.111 / +0.157. The channels carry most of the head's score but the head also uses the scene beside the nav.
* **FEED-INFO is met on its stated point-estimate rule.** The v9 classifier beats the text classifier by +0.111, which is at least 2/3 of the C0 - S1 gap at s0 (0.093) and s1 (0.100). The same also holds at s10 (gap 0.100, bar 0.067) and s11 (gap 0.153, bar 0.102). Read it with its interval: the paired v9 - text difference has CI [-0.005, +0.226] and is not separated, so the data support "the v9 channels carry more lateral information than the text channels" as a point reading, not as a separated one.
* The two readings do not exhaust the gap: the head gap (0.14 to 0.15) exceeds the feed difference (0.111) by about 0.03 to 0.04, inside the seed floor F 0.018 to 0.067 depending on split.

## Splits (same classifier, same windows)

| split | labelled n / episodes | flex v9 | flex text | C0 s0 / s1 head | S1 s0 / s1 head |
|---|---|---|---|---|---|
| INF_ANY (a feed names a turn) | 371 / 41 | 0.534 | 0.554 | 0.739 / 0.804 | 0.585 / 0.628 |
| UNINF_BOTH | 1,736 / 131 | 0.329 | 0.351 | 0.332 / 0.332 | 0.332 / 0.332 |

* INF_ANY: the C0 head exceeds the v9 classifier by +0.205 / +0.270 (separated); the S1 head exceeds the text classifier by +0.031 / +0.074 (not separated), that is 88-95 % of its classifier ratio. The text-fed heads sit near what the text channels alone give, and the C0 head sits well above what the v9 channels alone give. The v9 - text classifier difference on INF_ANY is -0.020 [-0.192, +0.159].
* UNINF_BOTH holds 10 turn windows (TURN_L 1, TURN_R 9); every head and both classifiers read the majority-class score (0.33 to 0.35), so that split carries no discrimination, as in the landed navecho.

## Controls

* **K-REACH passes:** the label itself as the classifier input reads macro-F1 1.0 out-of-fold.
* **K-SHUFFLE misses its pre-stated band.** Nav permuted across windows (10 permutations, out-of-fold) reads 0.230 (v9 feed; range 0.171-0.311) and 0.145 (text feed; range 0.116-0.194), against the majority-class score 0.317 and the committed band |diff| <= 0.08. The control reads below the majority score, not above it: a balanced-weight classifier given noise spreads its predictions over the classes and scores about zero on the rare ones, so the majority score is the wrong reference for this classifier. The band is reported as written (FAILED); it is not moved. Direction matters for the reading: the control shows no leak that inflates the out-of-fold score, and the real classifiers sit 0.47 / 0.45 above their shuffles. An unweighted classifier was not run.
* K-FEED passes (nav_in identical across every arm of a feed); K-LANDED passes (17 head macro-F1 values equal the landed `fam/out/RESULT_FAM_EXTRA.json` to 1e-12). Grouping: no episode spans two folds (test pins it, and a mutant with window-level folds goes RED).

## What this says and does not say
The classifier on the 5 channels reaches 0.705 on v9 and 0.594 on text, so the channels (side token plus distance to the turn) carry real lateral information beyond the token-only map (0.562 v9, 0.547 text in the landed navecho). It stays below the C0 head by 0.11 to 0.16, so the head's score is not the nav alone. The classifier gap between the feeds (+0.111) is about 0.8 of the head gap; that sizes the route-cue content of the text feed as the lever, with the interval caveat above. A second seed-level floor, training-run variance, is covered only by repeating the reading at s10 / s11 (same sign, ratio 0.86 / 0.83). Frame-history nav channels are not in the dump and were not tested.
