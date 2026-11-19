# PREREG_V8F_PROOF_ADD50 (GAPRATE10K): a closing speed from the change of the reader's own gap read, in the lead rule on the 10k checkpoints; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD49 and the T0 read beside it (section 1) and before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known (MEASURED; rows H-V8F-PROOF-104 and -105)

* Card ADD48 (T1-kin): at label quality the closing speed is the largest measured lever of the rule after the rule itself. `CLOSE10K-CLOSING-CUTS-UNSAFE-AT-NO-PRICE`: unsafe rate with against without the label closing speed -0.0320 [-0.0573, -0.0107]; levels 0.0463, 0.0302 against 0.0890, 0.0516; contact 0.0000, 0.0031 against 0.0327, 0.0156.
* Card ADD49: NOT NAMED. Its control h2 failed: the reader's trained closing output is no closing speed (Pearson r against the label closing speed 0.271, 0.284, bar 0.30).
* T0 read beside card ADD49 (`integ10k/read_closer/CLOSING_DIFF_T0.json`; clips of the training corpus held out from the reader; names nothing): the change of the reader's gap read over a baseline of 25 cache steps (5.0 s) follows the change of the label gap (r on the check clips 0.693, 0.721; error of the shrunk read 30 and 34 percent below that of the constant 0); over 5 to 15 steps it does not (r 0.25 to 0.57). The baseline 25 was chosen on the fit clips at both reader seeds; the least-squares slopes fitted on the fit clips are 0.7708154241032367 (reader seed 0) and 0.7831096226098508 (reader seed 1).
* A design input counted from the banked rule logs of `RQ50` (no outcome): the window starts lie between cache step 3 and 71; 74.5 percent of the 7,197 rule ticks per seed have the 29 earlier cache steps this read needs; of the reader-present ticks 75.6 and 75.9 percent.

## 2. Question

Does the reader-fed rule gain when its closing-speed input is the change of the reader's own gap read over the last 5 s, instead of 0? The T0 read says the change is readable; it does not say that a 5 s mean serves the rule. The arm `RQ50c` of card ADD48 (label closing speed at the tick) is the ceiling. Beside it, a replicate the level line did not have at level 0.50: the same form with the reader of the other seed.

## 3. Arms (no training, no new reader, no new map, no fit on the evaluation windows)

Models (`UE_s0`, `UE_s1`), reader arms, maps (level 0.50), presence threshold 0.5, floor form cap-only, rule parameters, evaluation cache, window grid and pinned tool as cards ADD46 to ADD49. Wrapper `integ10k/v8f_t1_eval_reader_rule_d.py` (written by `integ10k/make_gaprate10k.py` with asserted text edits from the wrapper of card ADD49).

The closing speed of the form `RRd` at a rule tick at cache step j of a window: the reader is read a second time on the logged camera tokens of the cache steps j-29 to j-25 of the window's own clip (frames before the tick only; the same token array the tick's own read is gathered from; no model weight, no label). If the reader is present at the tick and at step j-25 (probability at least 0.5, full 5-frame history at both), the fed closing speed is slope x (gap read at j - gap read at j-25) / 5.0 s, with the slope of section 1 for the reader seed and the gap reads before the map; else it is exactly 0, as in `RQ50`. Presence and gap are fed as in `RQ50`.

| run | model seed / reader seed | closing speed | role |
|---|---|---|---|
| `RQ50d` | 0 / 0 and 1 / 1 | the change of the reader's gap read over 25 steps, shrunk | deciding |
| `RQ50x` | 0 / 1 and 1 / 0 (reader and map of the other seed) | 0 | block X: what a second reader changes at level 0.50 |
| `RQ50Rd` | 0 / 0 | 0 | control d1: a rerun of `RQ50` at model seed 0 under the new wrapper |

Reused without a rerun: `R0`, `RLb`, `RLc`, `RQ50`, `RQ50c`. Five new rollouts. Before them the queue runs a smoke of the form on 96 windows of model seed 0; only the identities of `integ10k/check_prev_read.py` are read from it, no rate.

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample, so there is no inference variance to read). U on the 562 scored windows per seed, C and F on the 642 lead windows.

* S1 = U(`RQ50d`) - U(`RQ50`) (primary), S2 = C(`RQ50d`) - C(`RQ50`), S3 = F(`RQ50d`) - F(`RQ50`).
* Block X: U, C, F and the four families of `RQ50x` against `RQ50` at the same model seed. The reader floor on U is the larger absolute value of the two per-seed differences.
* Beside: `RQ50d` against `RQ50c` (the ceiling), against `R0` and against `RLc` on U, C and F; the four families for `RQ50d` against `RQ50`; the share of the reader-present ticks that carry a closing read.

## 5. Names (fixed here)

* `GAPRATE10K-CUTS-UNSAFE`: S1 below 0 at both model seeds and its pooled interval excludes 0.
* `GAPRATE10K-HARMS`: the pooled interval of S1 lies above 0.
* `GAPRATE10K-NO-U-CHANGE`: otherwise.
* To CUTS-UNSAFE and to HARMS one of two endings is added: `-ABOVE-READER-FLOOR` if the absolute pooled S1 is larger than the reader floor of block X, else `-WITHIN-READER-FLOOR`. A name with the second ending is not quoted as an effect of the lever.
* Secondary, named as secondary only: `-CUTS-CONTACT` if the interval of S2 lies below 0; `-LESS-FALLING-BEHIND` if the interval of S3 lies below 0, `-MORE-FALLING-BEHIND` if it lies above 0.
* Block X names nothing.

What follows, fixed now:
* CUTS-UNSAFE with `-ABOVE-READER-FLOOR`: the next arm is the replicate of the arm itself with the reader of the other seed (two rollouts, no training); if it keeps the sign at both model seeds the coordinator proposes this closing input for the 10k candidate; until then the closing input of the candidate stays 0 (decision D10).
* CUTS-UNSAFE with `-WITHIN-READER-FLOOR`, or NO-U-CHANGE with `-CUTS-CONTACT` and without `-MORE-FALLING-BEHIND`: the same replicate is run; D10 stands.
* NO-U-CHANGE otherwise, and HARMS: D10 stands. The levers left are a reader trained with a longer frame history and a closing target (a tool build and a probe training on this fork's pod, with its gate) and the reader fitted on the union (card ADD45, a go of the PI). After HARMS the closing read is first split at T0 by label gap band and by the change of the ego speed over the 5 s.
* A control fails: nothing is named. After d2 the lever is the reader with a longer history; after any other control the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named for `RQ50d`; block X is read if d1 and d7 to d10 pass)

* d1 (wrapper, known value): the dump of `RQ50Rd` at model seed 0 equals the banked dump of `RQ50` at model seed 0 array for array (`cmp_dumps.py`, 141 files, 0 differing).
* d2 (the fed value is a closing speed, T0): on the ticks where the label has a lead within 60 m (log of `RLc`) and a closing read was fed, the Pearson correlation of the fed closing speed (side log of `RQ50d`) with the label closing speed at the tick (log of `RLc`) is at least 0.3 at both model seeds, the bar of control h2 of card ADD49; and a closing read exists on at least 0.3 of the reader-present ticks at both model seeds. Reported beside: its mean absolute error and that of the constant 0.
* d3 (feed): in the rule log of `RQ50d` the closing-speed column equals the fed value of the side log within 1e-4 m/s at every tick; the fed value is exactly 0 where no closing read exists; no closing value came from the labels (the mix count is 0); the lead speed column equals max(0, ego speed column + closing-speed column) within 1e-4 m/s where the three are finite; the slope in the run record equals the slope of the T0 file for the reader seed and the baseline is 25.
* d4 (gather, known value in the run): the same gather with a baseline of 0 steps gives the tick's own read (probability and gap within 1e-4; equal history flags) at every tick.
* d5 (gather, independent path): `check_prev_read.py` recomputes the read at the tick and the earlier read from the cache file of the clip with the reader's own clip path (the path of the banked reader rows); probabilities and gaps agree within 1e-3, history flags are equal, at least one earlier read exists; the fed value equals the formula of section 3 from the logged reads.
* d6 (same read): at a model seed the reader's presence probability, read gap, fed gap and history flag of every tick are equal in the side logs of `RQ50` and `RQ50d`, and the tick index columns are equal.
* d7 (counts): 141 dump files, 2,399 windows and the same lead windows in every new dump.
* d8 (reused dumps): the result files of the five reused runs at both model seeds have the sha256 recorded before the run (`integ10k/GAPRATE10K_REUSED.json`).
* d9: the controls of the headway reader and of the family reader on the new pairs.
* d10 (model and reader): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed; the rule log of `RQ50x` at model seed k carries the reader sha256 and the map sha256 of `RQ50` at model seed 1-k and the tick index columns of `RQ50` at model seed k.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

56 episodes with a lead, read here for the fifth time (cards ADD46 to ADD49 before); no adjustment for the repeated reads is applied. A lead that does not react; two model seeds; two crossed runs make a coarse floor. The closing read is a mean over the last 5 s and lags a lead that has just started to brake. About a quarter of the ticks cannot have a closing read (the window starts less than 29 steps into its clip); there the arm equals `RQ50`. The baseline and the slopes were fitted on clips of the training corpus, against the change of the label gap over the same baseline, not against the closing speed at the tick; nothing is fitted on the evaluation windows. The ceiling arm `RQ50c` itself was not separated on the unsafe rate (card ADD48), so a large gain on the unsafe rate is not expected; the primary read is kept on the unsafe rate all the same.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument).
