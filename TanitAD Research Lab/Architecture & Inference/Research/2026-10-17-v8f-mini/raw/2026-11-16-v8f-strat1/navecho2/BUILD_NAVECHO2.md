# BUILD_NAVECHO2 -- flexible nav-only classifier for D-STRAT1-TACLAT-F1

Built 2026-11-17. CPU only (`CUDA_VISIBLE_DEVICES=-1`), no training of the driving model, nothing re-dumped, no landed file edited, nothing committed or pushed. Evidence class MEASURED, tier T1-kin tick 0.

## Files
* `navecho2.py` imports the landed `navecho.py` (and through it `fam/fam_extra.py`, `strat1_lib`, `taniteval.ci`). New: `episode_folds` (episodes dealt into 5 folds), `crossfit_hgb` (out-of-fold prediction, fit on the labelled windows of the other folds only), `label_onehot`, `shuffled_control`, `reading` (ECHO-CONT / FEED-INFO / MODEL), `run`.
* `test_navecho2.py`: 10 tests green (17 s). Grouping never spans an episode across folds; out-of-fold reads 1.0 with the label as input and near chance on noise; ignored windows excluded from fit and score; labels shifted in the scored fold only still get the rule learned elsewhere; mutations (fit on the scored fold, window-level folds, echo threshold, any-vs-all seeds, 2/3 factor) each turn a check RED; hyper-parameters pinned.
* Run: `python navecho2.py` (243 s at B 2,000). Output `out/RESULT_NAVECHO2.json`, `out/RESULT_NAVECHO2.md`.

## Layout facts
nav_in = `[N,5]` at the first tick: channels 0-2 one-hot (all-zero = no nav row), 3 = distance/100 clipped [0,3], 4 = distance valid. The T1 dump writes `batch["nav"][b, -1]` (`v8f_t1_eval.py:370`), the vector fed at NOW; the dump has no earlier-frame nav, so the frame history could not be used.

## Declared before scoring
HistGradientBoostingClassifier max_iter 200, learning_rate 0.1, max_leaf_nodes 15, class_weight balanced, random_state 0, early_stopping off; 5 folds, fold seed 0; reading rules as in the brief (FEED-INFO compares the paired v9 - text classifier difference with 2/3 of the C0 - S1 head gap at s0 and s1, gap positive; S1 is the text arm of the gap, N1 / S1d / s10 / s11 beside). K-SHUFFLE band 0.08 around the majority score, 10 permutations.

## Outcome and honest limits
Verdict FEED-INFO on the point-estimate rule; the paired v9 - text classifier difference (+0.111) has CI [-0.005, +0.226] and is not separated. K-SHUFFLE missed its band (reads 0.23 v9 / 0.15 text, below the 0.317 majority score, because the balanced weighting spreads noise predictions over classes); reported as FAILED, band unchanged, no leak direction. Not run: an unweighted classifier, frame-history channels. Training-run variance is covered only by the s10 / s11 repeat.
