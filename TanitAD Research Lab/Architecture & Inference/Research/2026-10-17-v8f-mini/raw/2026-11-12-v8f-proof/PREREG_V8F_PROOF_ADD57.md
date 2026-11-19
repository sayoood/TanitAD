# PREREG_V8F_PROOF_ADD57 (BRAKE10K): the braking limit of the lead rule at the programme's 2.0 s cycle, 8.0 m/s2 in place of 6.0; no training

Card of the v8F proof study, written by the coordinator fork while the rollouts of cards ADD54 and ADD55 run and the queue of card ADD56 waits; none of the three was read when this card was frozen, and it uses none of them. Written before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Before the freeze only a tool check ran (section 6, b11; synthetic plans, no model, no data of any card). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.

## 1. What is known

* MEASURED, T1-kin (cards ADD46 to ADD48 and their rows): at the 2.0 s cycle the lead rule cuts the unsafe rate U of the two 10k models. Levels per model seed on 562 scored windows: no rule 0.1459, 0.1139; label-fed rule with the closing speed 0.0463, 0.0302 (26 and 17 windows); reader-fed rule at map level 0.50 0.0872, 0.0730 (49 and 41 windows).
* MEASURED, descriptive (`integ10k/read_gaprate/UNSAFE_CAUSE.json`, `TIMES_NOTE.md` line of 2026-11-19T16:30:01Z): under the label-fed rule ALL 26 and 17 windows that stay unsafe have the rule at its braking limit (6.0 m/s2) at the tick in force; all are of the time-gap kind, and 13 and 8 of them are shallow (never a gap under 2.0 m, smallest time gap at least 0.8 s). Under the reader-fed rule a part of the unsafe windows is at the limit and a part has the plan unchanged (the reader read no lead).
* MEASURED, T1-kin (card ADD51, row H-V8F-PROOF-107): a longer time gap of the rule does not cut U beyond the value at 1.2 s. (card ADD52, row 108): a shorter cycle cannot be read on these models, their own rollout is not stable at 1.0 s.
* So at label quality the one rule parameter every remaining unsafe window points to is the braking limit. 6.0 m/s2 was a design choice of the first rule card; a full brake on a dry road reaches about 8 to 10 m/s2 (textbook value, not measured here).

## 2. Question

At the 2.0 s cycle, does a braking limit of 8.0 m/s2 cut the unsafe rate of the rule-driven rollout further, with labels and with the reader, and what does it cost in falling behind and in the four families?

## 3. Arms (no training, no map fit, no limit on the model's plan, pinned readout)

Models `UE_s0`, `UE_s1`, reader arms, maps (level 0.50), rule parameters D 3 m, time gap 1.2 s, floor form cap-only, evaluation cache, window grid and pinned tool as cards ADD46 to ADD51; replan interval 10 steps. Wrapper `integ10k/v8f_t1_eval_reader_rule_bm.py` (by `integ10k/make_brake10k.py` from the wrapper of card ADD53; sha256 31da3c553eef1b8f...): one more parameter, `--b-max`, passed into the rule's keyword arguments and written into the rule log.

| run | form | braking limit (m/s2) | role |
|---|---|---|---|
| `RLcB8` | label-fed rule with the closing speed | 8.0 | the limit at label quality |
| `RQ50B8` | reader-fed rule at level 0.50, closing 0 | 8.0 | the limit with the reader |
| `RLcB10` | label-fed | 10.0 | dose; names nothing |
| `RLcB6`, model seed 0 | label-fed | 6.0 given as a parameter | control b1: must equal the banked `RLc` |

Each run except the control at both model seeds: seven rollouts, started after the queue of card ADD56 has ended. Reused without a rerun: the banked `R0`, `RLc`, `RQ50` (limit 6.0).

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (56 lead episodes, 642 lead windows, 562 scored for U; 141 episodes and 2,399 windows for the families).

* BR (per form) = U(limit 8.0) minus U(limit 6.0, banked); contact C and falling behind F of the same pair beside; all family cells beside (longitudinal, lateral, tactical).
* Dose, label-fed: limit 10.0 minus 6.0 and 10.0 minus 8.0.
* Beside: each 8.0 run against no rule; reader-fed minus label-fed, both at 8.0.
* Reported without a bar, from the rule logs: saturated executed steps (the cap would need more braking than the limit allows) at the first tick and over all ticks, per limit.

## 5. Names (fixed here)

Per form (LABEL, READER):

* `BRAKE10K-<form>-HARDER-BRAKING-CUTS-UNSAFE` if BR is below 0 at both model seeds and its pooled interval lies below 0.
* `BRAKE10K-<form>-HARDER-BRAKING-RAISES-UNSAFE` if BR is above 0 at both model seeds and its pooled interval lies above 0.
* `BRAKE10K-<form>-NO-U-CHANGE` otherwise.

What follows, fixed now:
* LABEL reads CUTS-UNSAFE: the braking limit is a live parameter of the safety layer; the value 8.0 goes into the specification of the integrated candidate as a proposal, with its price (F, families) beside it. Changing the specification of the safety layer is the PI's decision; this card changes nothing by itself. The split of the windows that still stay unsafe is repeated on the 8.0 runs (descriptive, convention of the `TIMES_NOTE.md` line of 2026-11-19T16:28:23Z, names nothing).
* LABEL reads NO-U-CHANGE: harder braking inside a 2.0 s reaction interval does not reach the windows at the limit. What is then left for them is the reaction interval itself, and that lever is held by the stability of the model at short cycles (cards ADD52 to ADD56); reported as written, with the dose arm beside.
* READER: reported as written. Whatever it reads, the part of its unsafe windows where the reader read no lead is a matter of reader quality (card ADD45, with the PI), not of this parameter.
* A name RAISES-UNSAFE for either form: reported as written; the split of the unsafe windows is repeated to see which windows were added.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* b1 (wrapper edit, known value): the dump of `RLcB6` at model seed 0 equals the banked dump of `RLc` at model seed 0 array for array (`cmp_dumps.py`, 141 files, 0 differing).
* b2 (parameter applied): the rule log of every new run records its braking limit, the same value in the rule's keyword arguments, the time gap 1.2 s and 10 steps; the banked rule logs record 6.0 and no such keyword.
* b3 (the first tick does not depend on the limit): against the banked run of the same form and model seed, the tick rows are the same and the rule inputs at the first tick (gap, closing speed, ego speed, lead speed) are equal (largest difference at most 1e-6).
* b4 (the reader reads logged frames only): the reader reads of `RQ50B8` equal those of the banked `RQ50` at every tick.
* b5 (the limit is live and acts in the known direction, deliberate check on identical inputs): at the first tick, where inputs and plans are those of the banked run, the number of saturated executed steps is above 0 at 6.0 and smaller in every run with 8.0 or 10.0; for the control run it is equal row by row.
* b6 (counts): 141 dump files in each of the seven new dumps, 2,399 windows, 642 lead windows.
* b7 (reused dumps): the result files of the banked runs have the sha256 recorded before the run (`integ10k/TAUCURVE10K_REUSED.json`, the ten files of card ADD51).
* b8: the controls of the headway reader and of the family reader on the new pairs.
* b9 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.
* b10: the wrapper of the runs has the sha256 of section 3.
* b11 (the rule takes the parameter, run before the freeze as a tool check): `integ10k/check_bmax.py` ends with its line True on the pod (`integ10k/read_brake/CHECK_BMAX.txt`): on 400 synthetic plans behind a slower lead the changed plans brake at most at the limit given (6.000, 8.000, 10.000 m/s2), the saturated executed steps fall from 1,479 to 1,363 to 1,286, and 6.0 given as a parameter equals the default.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

T1-kin: the lead does not react, the executor follows every plan, and no vehicle model says whether 8.0 or 10.0 m/s2 can be reached on the road of a window; the rule's limit is a parameter of the plan, not a measured capability of a car. Comfort is not scored by any family; hard braking has a price no number here shows. The unsafe count at label quality is small (26 and 17 windows at 6.0), so a cut can be real and still not separate; the dose arm is there to show the direction. 56 lead episodes are read again, with no adjustment for the repeated reads; two model seeds. The card was written after the split of the unsafe windows was known, which is why this parameter was chosen. The evaluation convention of the programme and its pinned tool are not changed.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument). The first tick's model outputs do not depend on the braking limit.
