# PREREG_V8F_PROOF_ADD48 (CLOSE10K): the lead rule with the closing speed on the 10k checkpoints; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD47 (section 1) and before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known (MEASURED, T1-kin, rows 101 and the read of card ADD47 in `integ10k/read_price/READ_PRICE10K.json`)

* Unsafe rate U per model seed (`UE_s0`, `UE_s1`): no rule 0.1459, 0.1139; the rule fed by labels WITHOUT the closing speed (`RLb`) 0.0890, 0.0516; the rule fed by the reader at map level 0.50 (`RQ50`) 0.0872, 0.0730; the rule with a planted gap of 0 m (`RR0`, it brakes at every tick with a reader presence) 0.0231, 0.0214.
* So the label-fed rule in its present form leaves 0.05 to 0.09 of unsafe windows, above the 0.02 of the planted run. The rule of cards ADD26/ADD33 has a second input, the closing speed (lead speed minus ego speed); cards ADD46 and ADD47 ran it with that input set to 0, because the token reader gives no closing speed.
* Card ADD47: name `PRICE10K-MEDIAN-KEEPS-CUT-PRICE-REMAINS`; the candidate takes map level 0.50.

## 2. Question

On the two 10k checkpoints: how much of the remaining unsafe rate of the label-fed rule is removed when the rule also gets the closing speed, and at what price in falling behind? And does the reader-fed rule gain the same when the closing speed comes from the labels (a ceiling for a reader-made closing speed; a label closing speed is NOT admissible at inference, this arm is a diagnostic)?

## 3. Arms (no training, no new reader, no new map)

Models, reader arms, maps (level 0.50 of card ADD47), presence threshold 0.5, floor form cap-only, rule parameters, evaluation cache, window grid and pinned tool are those of cards ADD46/ADD47. Only the closing-speed input of the rule changes. The wrapper `integ10k/v8f_t1_eval_reader_rule_c.py` is the wrapper of card ADD46 with two more forms (written by `integ10k/make_close10k.py` with asserted text edits; the mix planner of the distance-rule folder already has the closing-speed source `label`).

| run | presence and gap | closing speed | role |
|---|---|---|---|
| `RLc` | labels | labels | deciding |
| `RQ50c` | reader, map level 0.50 (model seed k with reader seed k) | labels where the label has a lead at the tick, else 0 (the missing-value rule of the mix planner; the count is recorded) | diagnostic, a ceiling |
| `RLbR` | labels | 0 | control c1: a rerun of `RLb` at model seed 0 under the new wrapper |

Reused without a rerun: the dumps `R0`, `RLb` (card ADD46) and `RQ50` (card ADD47). Five new rollouts (`RLc` and `RQ50c` at two model seeds, `RLbR` at seed 0).

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample). U on the 562 scored windows per seed, F on the 642 lead windows.

* Q1 = U(`RLc`) - U(`RLb`), Q2 = F(`RLc`) - F(`RLb`), Q3 = U(`RQ50c`) - U(`RQ50`), Q4 = F(`RQ50c`) - F(`RQ50`).
* Beside: U(`RLc`) - U(`R0`), the contact rate C for the same pairs, the levels of every run, the four families for `RLc` against `RLb` and for `RQ50c` against `RQ50`.

## 5. Names (fixed here)

* `CLOSE10K-CLOSING-CUTS-UNSAFE`: Q1 below 0 at both model seeds and its pooled interval excludes 0. It carries one of two endings:
  * `-AT-NO-PRICE`: the interval of Q2 includes 0 or lies below 0;
  * `-AT-A-PRICE`: the interval of Q2 lies above 0.
* `CLOSE10K-CLOSING-NO-CUT`: otherwise.
* For the diagnostic arm, a second name: `CLOSE10K-READER-GAINS` if Q3 is below 0 at both model seeds and its pooled interval excludes 0, else `CLOSE10K-READER-NO-GAIN`.

What follows, fixed now:
* CUTS-UNSAFE with READER-GAINS: the next arm is a closing speed made by the reader itself (the change of its fed gap between ticks, tokens only, no training), read against `RQ50c` as its ceiling, on its own card.
* CUTS-UNSAFE with READER-NO-GAIN: the errors of the reader's presence and gap hide the gain; the ticks where reader and label differ are split from the rule logs, and the reader fitted on the union (card ADD45, a go of the PI) is the lever.
* NO-CUT: the closing speed is not the lever. The windows that stay unsafe under `RLb` are split from the existing dumps and rule logs by cause (the lead is inside the rule's range at the first tick or enters later; the braking floor is reached; the replan interval), and the next arm follows the largest group.

## 6. Controls (all must pass; else nothing is named)

* c1 (wrapper, known value): the dump of `RLbR` at model seed 0 equals the banked dump of `RLb` at model seed 0 array for array (tool `cmp_dumps.py` of the distance-rule folder, 141 files, 0 differing).
* c2 (feed of the closing speed, from the rule logs): in `RLb` the closing-speed column is 0 at every tick; in `RLc` and in `RQ50c` it is finite at every tick the rule calls present (a missing label value is replaced by 0 by the mix planner, the count is recorded), its standard deviation over those ticks is above 0.1 m/s, and the lead speed column equals max(0, ego speed column + closing-speed column) within 1e-4 m/s at every tick where the three columns are finite.
* c3 (same read): at a model seed the reader's presence probability, read gap and fed gap of every tick are equal in the side logs of `RQ50` and `RQ50c`, and the tick index columns are equal in the logs of all runs of that seed.
* c4 (counts): 141 dump files, 2,399 windows and the same lead windows in every new dump.
* c5 (reused dumps): the result files of `R0`, `RLb` and `RQ50` have the sha256 recorded before the run.
* c6: the controls of the headway reader and of the family reader on the new pairs.
* c7 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.

## 7. Limits

56 episodes with a lead; a lead that does not react; two model seeds; the label closing speed comes from the label join and is exact only up to the tracker; `RQ50c` is a diagnostic and not a deployable form. The same lead episodes have now been read in three cards; only the pairs of section 4 can give a name here.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument).

Cost: five rollouts of about five minutes each on this fork's pod; no training.
