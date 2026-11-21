# PREREG_RBDIST_ZT (written 2026-11-18 BEFORE any number on the eval target was computed)

Zero-training first step of the RB-DIST card named in `navrb2/PREREG_V8F_NAVRB2.md` section 7, row RB-READ-NO-GAIN. CPU only, `CUDA_VISIBLE_DEVICES=-1`, no driving model is trained, loaded or dumped. Evidence class MEASURED (ours, CPU, banked route-text table). Tier **T0, nav-only decodability: NOT driving performance** (nobody drives; a classifier reads route-text features and is scored against the label).

## 1. Question
Does the route text carry the roundabout exit SIDE? Can a nav-only classifier decode the executed lateral decision (the TAC-LAT target, the v9 `lat_cls` label, 5 classes) on roundabout windows from route-text features {state, exit number, countdown distance} better than from {state} alone?

## 2. Where the features live in the text (found from source, quoted)
* `navtok.py:32` rule `RB_TAKE_EXIT`: `At roundabout, take exit (?P<exit>\d+)`; the distance suffix is `_DIST = " in (?P<d>\d+)m"` (`navtok.py:35`), `parse_full` returns `exit_n` and `dist_m` (`navtok.py:76-92`, `NavTok(... dist_m ... exit_n)`).
* The pinned table `navtok_table.npz` (sha256 2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992) stores per cache step `{split}_cls_id`, `{split}_dist_m` (the countdown from the router text suffix), `{split}_flag`, `{split}_text_i` (index into `texts.json`, which holds the raw text and so the exit number). Row rule: latest router row with `ts <= now_s x 1e6 - 195,000 us` (`navtok/build_table.py:42-71`).
* MEASURED before this card (`navrb/out/RESULT_NAVRB.md`): RB_TAKE_EXIT rows carry the exit number always and the distance on 1,929 of 1,937 texts; RB_ENTER / RB_EXIT / ROTARY_EXIT rows carry a distance only; no roundabout text has a side or bearing word; on the grid: 235 windows in 16 episodes have a roundabout row at NOW, exit 1: 92, 2: 96, 3: 14, none: 33. So the exit number and the countdown distance both exist in the route text; the question is whether they decode the side.

## 3. Features (all computed from table rows of the clip at or before the window's NOW row r; nothing else)
`state` = 1 iff `flag[r] == 0` and the class at r is RB_ENTER, RB_EXIT, ROTARY_EXIT, RB_TAKE_EXIT or RB_NEXT_EXIT (the NAVRB2 `nav_rb` definition, `navrb.py is_state`). Roundabout windows (RB) are the windows with `state == 1`.
* `exit_cur` = exit number parsed from the text at r (0 if the text has none).
* `exit_last` = exit number of the latest row at or before r, in the same clip, with `flag == 0` and an exit number (`navrb.py latest_exit_causal`); 0 if none.
* `dist` = `dist_m[r] / 100` clipped to [0, 3] (the layout of nav channel 3), 0 when absent; `dist_valid` = 1 iff `dist_m[r]` is finite.
Feature sets: **S** = [state] (constant 1 on RB windows); **ED** (the full set) = [state, exit_cur, exit_last, dist, dist_valid]. Beside, no bar: **E** = [state, exit_cur, exit_last]; **D** = [state, dist, dist_valid].
Not used: ego state, track, speed, arc, join labels, the v9 nav, the situation classifier, any row after r, the row age.

## 4. Admissibility (binding)
Features are a pure function of the table arrays `(cls_id, flag, text_i, dist_m)`, the parsed-text arrays and the indices `(r, clip start o)`. A test replaces every table row after r by garbage and requires every RB feature unchanged; a mutant that reads row `r + 5` must turn the same test RED. The label never enters the feature function (its signature has no label argument; a test inspects it).

## 5. Windows, fit, score
* Eval windows: the 2,399 eval-141 grid windows of the banked dump (`read/t1/dump_STRAT1-C0_s0_m10_causal_heads`: `eid`, `clip_index`, `ws`); row = `eval_off[clip_index] + ws` (`strat1_lib.rows_of`). RB eval windows must number **235 in 16 episodes** (the run refuses on a mismatch). Scored: RB windows with a v9 label (`has` and `lat_cls >= 0`); n printed.
* Train windows: every row of the train-4572 split with `state == 1`, `has` and `lat_cls >= 0` (`ckpt120build/in/join_train.npz`, rows aligned to the table's `train_*` arrays, offsets verified equal). **Train clips and eval clips must be disjoint (asserted); no eval window or episode is in any fit.**
* Label: `lat = lat_cls` where `has` and `lat_cls >= 0`, else ignored (5-way: LANE_KEEP, TURN_L, TURN_R, LANE_CHANGE_L, LANE_CHANGE_R); this is the label `fam_extra.v9_rows` builds.
* Classifier: sklearn `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, class_weight=None, random_state=0, early_stopping=False)` (the `navecho2.py` HGB, with the class weighting removed because the primary metric is accuracy), fitted ONCE on the train RB windows per feature set; argmax prediction on the eval RB windows. Deterministic at a fixed seed.
* **Primary metric: 5-way accuracy** (the D1 quantity of NAVRB2, accuracy of TAC-LAT on RB). Beside, no bar: macro-F1 (5 classes), TURN recall and TURN side accuracy on the RB windows labelled TURN_L / TURN_R (the 43 windows / 6 episodes of NAVRB, descriptive).
* Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, per-window correctness indicators, B 2,000, seed 0, over the episodes of the RB windows (16 expected). It answers "would another draw of EPISODES say this?"; it does not answer a training-run question (the classifier is deterministic; the training-run question does not arise for it) and does not answer whether a driving model would use the features.

## 6. Committed rule and both outcomes
Gain = accuracy(ED) - accuracy(S), paired.
* **TEXT-CARRIES-SIDE**: gain >= **0.0383** (the NAVRB2 rig floor F2_rb at this stratum, `MM_READING_NAVRB2.md:18`; the smallest difference that rig could resolve) AND the CI is separated (lower bound > 0). Next lever: the **RB-DIST training arm card** (exit number + countdown-distance channels on the route-text nav, S1R recipe, gate and reads as NAVRB2), and its D1 bar is the margin above.
* **TEXT-SILENT**: anything else (including a point estimate at or above the margin whose CI is not separated, which is named SUGGESTIVE-UNSEPARATED inside this label). Next lever: the roundabout side must come from **vision** (roundabout geometry in the scene: exit direction, lane layout); the exit-number and distance channels are dropped for this gap and no RB-DIST training arm is built. The cheapest next experiment on that branch, zero-training, is a frozen-encoder probe of the exit side on the 43 RB-TURN windows (own card).
* Beside, no bar, same estimator: E - S and D - S (which feature carries it).

## 7. Controls and the deliberate-regression arms (each must read as stated, else the read is INVALID and no verdict stands)
* **K-CONST** (known value): the S classifier predicts one constant class, the train-RB majority class; its eval accuracy equals the eval share of that class **exactly**.
* **K-REACH** (known value): the label one-hot as the input, fitted on train, scored on eval, reads accuracy **1.0**.
* **R-SHUF-EVAL** (deliberate regression): the eval feature rows (exit_cur, exit_last, dist, dist_valid) of ED permuted across windows (so across episodes), 10 permutations (seeds 0-9), classifier fitted on the true train features: the mean gain over S across the 10 must be **< 0.0383**, and the count of permutations that would satisfy the TEXT-CARRIES-SIDE rule is printed.
* **R-SHUF-TRAIN** (deliberate regression): the same permutation applied to the train feature rows (5 permutations, seeds 0-4), true eval features: the mean gain over S must be **< 0.0383**.

## 8. Limits stated now
* Optimistic by construction: the route text on PhysicalAI is GPS matched to OSM routing along the drive's own trace (NAVRB2 card, section 3); every number is worded "with the route supplied along the driven path".
* 16 eval episodes: the interval is wide; RB-TURN has 6 episodes and is descriptive.
* The countdown here is the router text's `in <d>m`; the trained arms' channel 3 is dead-reckoned from the ego arc. This card measures what the text contains, not what the trained arms were fed.
* A positive here shows decodability from the text, not that a driving model would convert it into lateral skill (that is the RB-DIST arm).
