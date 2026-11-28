# Card PREREG_V8F_PROOF_ADD65 (CAND10K): the lead rule under the causal readout without own acceleration, with and without the second look, against the programme's convention; no training

Card of the v8F proof study, written by the coordinator fork after the reads of cards ADD62 (row 118) and ADD63 and before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). It is the arm card ADD63 fixed for its outcome ("ZERO reads BETTER-THAN-PINNED for at least the LONGITUDINAL family with U not above pinned ... the next arm here is then the three rule forms under it at 2.0 s on their own card"). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known (MEASURED, T1-kin, the two 10k models `UE_s0`, `UE_s1`, 2.0 s cycle)

* Card ADD63 (read 2026-11-20T00:17Z, controls PASS): the family errors rise with the share of its own executed acceleration the model is handed at the tick (`ACCGAIN10K-LONGITUDINAL/LATERAL/TACTICAL-RISES-WITH-OWN-ACCELERATION`). With no own acceleration and the pinned path the longitudinal and tactical cells read better than with the pinned readout and the unsafe rate is not separated; taking the path from executed knots as well reads better again on the lateral and tactical cells (`ACCGAIN10K-PATH-LATERAL-BETTER`, `-PATH-TACTICAL-BETTER`). That readout (below: readout K) hands the model nothing from the plan it is about to discard and no acceleration from its own rollout. Without a rule, against the pinned readout, it reads better on all ten family cells (speed error -0.613 m/s [-0.722, -0.515], heading -2.83 deg, yaw rate -17.1 deg/s [-21.1, -13.7], tactical ADE -0.728 m [-0.948, -0.545]) with the unsafe rate not separated (-0.0294 [-0.0668, +0.0064]; levels 0.1121, 0.0890 against 0.1459, 0.1139).
* Card ADD62 (row 118): with the pinned readout the lead rule looking again after 1.0 s cuts the unsafe rate at label quality (-0.0089 [-0.0176, -0.0011]); reader-fed not separated.
* Row 114: under the backward study readout at 2.0 s the rule's cut was smaller than under the pinned readout (reader-fed -0.0196 [-0.0400, -0.0061]). How much of the rule's measured benefit under the pinned readout is the rule holding down the speed the pinned readout lets climb is not known.

## 2. Question

Under readout K: does the lead rule still cut the unsafe rate, does the second look still add, and how does the whole inference-side candidate (readout K, lead rule, second look) read against the programme's convention (pinned readout, rule at the tick) on the unsafe rate and on the families?

## 3. The mechanism (study wrapper; the pinned tool is not edited)

New wrapper `integ10k/v8f_t1_eval_reader_rule_rzgf.py`: the wrapper of card ADD62 (`v8f_t1_eval_reader_rule_rzf.py`, sha256 51b3e783536689f6; parameter `--rule-every`) with the readout edits of card ADD63 (`causal_g`, `--acc-gain`, `--path-exec`) applied by asserted text edits. Nothing new is added: readout K is `--readout causal_g --acc-gain 0 --path-exec`, and the second look is `--rule-every 5`. With the parent's parameters alone the wrapper must give the parent's result (control c2), and with the readout alone the result of the wrapper of card ADD63 (control c1).

## 4. Arms

Models `UE_s0`, `UE_s1`, reader arms, maps at level 0.50, rule parameters (D 3 m, time gap 1.2 s, braking limit 6.0, cap-only), evaluation cache, window grid and pinned tool as cards ADD46 to ADD63; replan interval of the model 10 steps; no limit on the plan.

| run | form | rule | readout | role |
|---|---|---|---|---|
| `RLcK` | label-fed with the closing speed | at the tick only | K | the rule under readout K at label quality |
| `RQ50K` | reader-fed at level 0.50, closing 0 | at the tick only | K | the same with the reader |
| `RLcKF5` | label-fed | every 5 steps | K | the whole candidate at label quality |
| `RQ50KF5` | reader-fed | every 5 steps | K | the whole candidate in its deployable form |
| `R0KID`, model seed 0 | none | none | K | control c1: must equal the banked `R0G0X` of card ADD63 |
| `RLcKIDp`, model seed 0 | label-fed | every 5 steps | pinned | control c2: must equal the banked `RLcF5` of card ADD62 |

Each run except the controls at both model seeds: ten rollouts. Reused without a rerun: `R0G0X` (no rule under K, card ADD63), the convention runs `R0`, `RLc`, `RQ50`, and `RLcF5`, `RQ50F5` (card ADD62).

## 5. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (56 lead episodes, 642 lead windows, 562 scored for U; 141 episodes and 2,399 windows for the families). The bootstrap answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample. Per form (LABEL, READER):

* RK = U(rule at the tick, K) minus U(no rule, K = `R0G0X`): the rule under readout K. C, F and all family cells beside.
* FK = U(rule every 5 steps, K) minus U(rule at the tick, K): the second look under readout K.
* CC = the whole candidate (rule every 5 steps, K) minus the programme's convention with the rule (the banked `RLc` or `RQ50`): U, C, F and all family cells, per family, never pooled into one score.
* Beside, naming nothing: (rule at the tick, K) minus the banked `RLc` / `RQ50` (the readout alone under the rule); (rule every 5 steps, K) minus the banked `RLcF5` / `RQ50F5`; reader-fed minus label-fed for the whole candidate; the whole candidate against no rule with the pinned readout (`R0`).
* Reported without a bar, from the logs: later looks that changed the plan and that reached the braking limit, per run; the saturated executed steps of the tick rows beside those of the banked runs; RS (share of the 2,399 windows ending more than 5 m/s above the logged speed) per run.

Family rule as card ADD63: named cells LONGITUDINAL speed error and along-track error, LATERAL heading error and yaw-rate error, TACTICAL tactical ADE; a family reads BETTER for a pair if every named cell reads BETTER by the family reader, WORSE if every named cell reads WORSE, NOT-SEPARATED if none reads BETTER or WORSE, else MIXED. For the READER form a pooled U difference of at most 0.0196 in absolute value is said to be inside the reader floor of row 106 wherever it is quoted.

## 6. Names (fixed here)

Per form:

* `CAND10K-<form>-RULE-CUTS-UNSAFE-UNDER-K` if RK is below 0 at both model seeds and its pooled interval lies below 0; else `CAND10K-<form>-RULE-NOT-SEPARATED-UNDER-K`.
* `CAND10K-<form>-SECOND-LOOK-CUTS-UNSAFE`, `-SECOND-LOOK-RAISES-UNSAFE` or `-SECOND-LOOK-NO-U-CHANGE` by FK (one sign at both model seeds and the pooled interval on that side of 0; else no change).
* `CAND10K-<form>-U-BELOW-CONVENTION`, `-U-ABOVE-CONVENTION` or `-U-NOT-SEPARATED-FROM-CONVENTION` by CC on U, the same way.
* `CAND10K-<form>-<FAMILY>-BETTER-THAN-CONVENTION`, `-WORSE-THAN-CONVENTION`, `-MIXED` or `-NOT-SEPARATED` by the family rule on CC, for LONGITUDINAL, LATERAL and TACTICAL.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument). The first tick's model outputs depend neither on the readout nor on the rule's rate.

What follows, fixed now:

* A form reads `RULE-CUTS-UNSAFE-UNDER-K`, U not above the convention, and no family WORSE-THAN-CONVENTION: the whole inference-side candidate (readout K, the lead rule, the second look) is put to the PI and the other session in the decision card as one proposal with these numbers, per form. Adopting any part of it changes what the planner reads at inference or the safety layer; that is their decision, and this card changes nothing by itself. The integrated 10k candidate of the common plan would then be read under the convention and under K side by side.
* A form reads `U-ABOVE-CONVENTION`: reported as written. The family gain of readout K then comes with a cost on the unsafe rate under the rule; the proposal of the decision card is the convention readout with the second look (row 118) and readout K stays a study device for that form.
* A form reads `RULE-NOT-SEPARATED-UNDER-K`: reported as the size of the rule under a causal readout; the rule's benefit measured under the pinned readout was then in part the rule holding down what the pinned readout lets climb, and the rows that quote it say so from then on.
* `SECOND-LOOK-NO-U-CHANGE` or `-RAISES-UNSAFE` under K: reported; the proposal of the second look then rests on the pinned readout alone (row 118).
* A family reads `WORSE-THAN-CONVENTION` or `MIXED`: reported with the table; nothing is proposed for that form beyond what holds cell by cell.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.
* The evaluation convention of the programme (2.0 s, pinned readout) and its pinned tool are not changed by this card.

## 7. Controls (all must pass; else nothing is named)

* e1: the pod read reproduces the banked dev-box read of card ADD46 leaf for leaf.
* c1 (wrapper edit, known value): the dump of `R0KID` (no rule, readout K, model seed 0) equals the banked dump of `R0G0X` at model seed 0 array for array (141 files, 0 differing). Marker name no earlier card used.
* c2 (wrapper edit, known value): the dump of `RLcKIDp` (label-fed, pinned readout, rule every 5 steps, model seed 0) equals the banked dump of `RLcF5` at model seed 0 array for array (141 files, 0 differing), with a marker name of its own.
* c3 (parameters applied): the readout log of every K run records `causal_g`, gain 0 and the executed path, the speed at the last executed knot equals the last executed step speed and the acceleration read is 0 (largest difference at most 1e-6); the rule logs record the form and the rule's rate.
* c4 (the first block does not depend on the readout, known value): the first 10 executed steps of `RLcK` and `RQ50K` equal those of the banked `RLc` and `RQ50` at the same model seed, and those of `RLcKF5` and `RQ50KF5` equal those of the banked `RLcF5` and `RQ50F5` (largest difference 0.0); the tick rows of the rule log at the first tick equal the banked ones in their inputs.
* c5 (cap only): in every later look of every run no step of the remaining plan is farther along the path than in the plan before the look (largest excess at most 1e-9 m).
* c6 (tool check before any run, on the pod CPU): with the combined wrapper the tool checks of card ADD62 (`check_rule_every.py`) and of card ADD63 (`check_acc_gain.py`) both end with their line True, each with its planted faults caught.
* c7 (counts): 141 dump files in each new dump, 2,399 windows, 642 lead windows; one tick row per window and tick; later-look rows as card ADD62.
* c8 (reused dumps), c9 (reader controls of the headway and family readers on the new pairs), c10 (checkpoint sha256 of each new run equals that of card ADD46; the wrapper of the runs has the sha256 the tool checks ran with).

## 8. Limits

T1-kin: no vehicle model, the executor follows every plan, the lead does not react. Readout K is a study device: it hands the model its executed speed held flat over the last frame and a path from executed knots only; a car measures an acceleration, and a model trained on its own rolled history may use it. The later looks use the logged frame of their moment. Two model seeds; 56 lead episodes and 141 episodes are read again with no adjustment for the repeated reads. The rule's parameters are those tuned under the pinned readout. The card was written knowing rows 110 to 118 and the read of card ADD63, which is why this readout was chosen.
