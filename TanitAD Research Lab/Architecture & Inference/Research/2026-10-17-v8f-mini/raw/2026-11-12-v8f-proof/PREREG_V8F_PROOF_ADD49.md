# PREREG_V8F_PROOF_ADD49 (CLOSER10K): the reader's own closing-speed output in the lead rule on the 10k checkpoints; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD48 and its split (section 1) and before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known (MEASURED, T1-kin, read of card ADD48 in `integ10k/read_close/pod/READ_CLOSE10K.json`, splits in `integ10k/read_close/`)

* Card ADD48: `CLOSE10K-CLOSING-CUTS-UNSAFE-AT-NO-PRICE` (label-fed rule with the label closing speed against without: unsafe rate -0.0320 [-0.0573, -0.0107]; levels 0.0463, 0.0302 against 0.0890, 0.0516; contact 0.0000, 0.0031 against 0.0327, 0.0156) and `CLOSE10K-READER-NO-GAIN` (reader presence and gap with the label closing speed against closing 0: unsafe rate -0.0125 [-0.0307, +0.0021]; contact -0.0164 [-0.0416, -0.0016]; falling behind -0.0257 [-0.0496, -0.0050]).
* Its section 5 names, for this outcome, the reader fitted on the union (card ADD45) as the lever; that is a decision of the PI and is not taken here. The split it asked for is done: reader recall on in-range label ticks 0.795 and 0.854 (0.48 and 0.62 in the band 40 to 60 m), and about half of the windows that are unsafe under the reader-fed rule and safe under the label-fed rule have a reader that is on time and never more than 5 m above the label gap.
* Not used so far: the token reader of card ADD46 has a third, trained output, the closing speed (the token probe was trained on presence, gap and closing speed; scale 10 m/s). It reads tokens only, so it is admissible at inference.

## 2. Question

This card goes beyond the follow-up of card ADD48 and is stated as such: does the rule gain when its closing-speed input is the reader's own closing output, instead of 0? The arm `RQ50c` of card ADD48 (label closing speed) is its ceiling.

## 3. Arms (no training, no new reader, no new map)

Models, reader arms, maps (level 0.50), presence threshold 0.5, floor form cap-only, rule parameters, evaluation cache, window grid and pinned tool as cards ADD46 to ADD48. Wrapper `integ10k/v8f_t1_eval_reader_rule_h.py` (written by `integ10k/make_closer10k.py` with asserted text edits from the wrapper of card ADD48).

| run | presence and gap | closing speed | role |
|---|---|---|---|
| `RQ50h` | reader, map level 0.50 (model seed k with reader seed k) | the reader's third output times 10 m/s | deciding |
| `RQ50R` | reader, map level 0.50 | 0 | control h1: a rerun of `RQ50` at model seed 0 under the new wrapper |

Reused without a rerun: `R0`, `RLb`, `RQ50`, `RLc`, `RQ50c`. Three new rollouts.

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample). U on the 562 scored windows per seed, C and F on the 642 lead windows.

* R1 = U(`RQ50h`) - U(`RQ50`) (primary), R2 = C(`RQ50h`) - C(`RQ50`), R3 = F(`RQ50h`) - F(`RQ50`).
* Beside: `RQ50h` against `RQ50c` (the ceiling), against `R0` and against `RLc` on U, C and F; the four families for `RQ50h` against `RQ50`.

## 5. Names (fixed here)

* `CLOSER10K-READER-CLOSING-CUTS-UNSAFE`: R1 below 0 at both model seeds and its pooled interval excludes 0.
* `CLOSER10K-READER-CLOSING-HARMS`: the pooled interval of R1 lies above 0.
* `CLOSER10K-READER-CLOSING-NO-U-CHANGE`: otherwise.
* Secondary, named as secondary only: `-CUTS-CONTACT` if the interval of R2 lies below 0; `-LESS-FALLING-BEHIND` if the interval of R3 lies below 0, `-MORE-FALLING-BEHIND` if it lies above 0.

What follows, fixed now:
* CUTS-UNSAFE: the rule of the 10k candidate takes the reader's closing output.
* NO-U-CHANGE with `-CUTS-CONTACT` and without `-MORE-FALLING-BEHIND`: the candidate takes the reader's closing output for the contact rate, stated as a decision on secondary evidence. NO-U-CHANGE otherwise: the closing input stays 0; the lever is the reader on the union (card ADD45, a go of the PI).
* HARMS: the closing input stays 0; the reader's closing read is split by label gap band at T0 to say where it fails.

## 6. Controls (all must pass; else nothing is named)

* h1 (wrapper, known value): the dump of `RQ50R` at model seed 0 equals the banked dump of `RQ50` at model seed 0 array for array (`cmp_dumps.py`, 141 files, 0 differing).
* h2 (the closing read is a closing speed, T0): on the ticks where the label has a lead within 60 m (log of `RLc`) and the reader is present, the Pearson correlation of the reader's closing read (side log of `RQ50h`) with the label closing speed (log of `RLc`) is at least 0.3 at both model seeds. Reported beside: its mean absolute error and that of the constant 0.
* h3 (feed): in the rule log of `RQ50h` the closing-speed column equals the reader's closing read of the side log within 1e-4 m/s at every tick, and the lead speed column equals max(0, ego speed column + closing-speed column) within 1e-4 m/s at every tick where the three columns are finite.
* h4 (same read): at a model seed the reader's presence probability, read gap, fed gap and history flag of every tick are equal in the side logs of `RQ50` and `RQ50h`, and the tick index columns are equal.
* h5 (counts): 141 dump files, 2,399 windows and the same lead windows in every new dump.
* h6 (reused dumps): the result files of the five reused runs at both model seeds have the sha256 recorded before the run.
* h7: the controls of the headway reader and of the family reader on the new pairs.
* h8 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.

## 7. Limits

56 episodes with a lead, read here for the fourth time; a lead that does not react; two model seeds; the reader's closing output was trained on the corpus only. The ceiling arm itself was not separated on the unsafe rate, so a gain on the unsafe rate is not expected to be large; the primary read is kept on the unsafe rate all the same and the secondary names are marked as secondary.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument).

Cost: three rollouts of about five minutes each on this fork's pod; no training. The read runs on the pod (the dev box's memory is left to another session's read); the pod first reproduces the banked read of card ADD46 (control e1).
