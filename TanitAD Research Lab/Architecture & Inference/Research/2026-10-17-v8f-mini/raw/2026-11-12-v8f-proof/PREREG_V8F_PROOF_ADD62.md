# Card PREREG_V8F_PROOF_ADD62 (RULEFAST10K): the lead rule applied twice per plan, the model still planning every 2.0 s; no training

Card of the v8F proof study, written by the coordinator fork after the reads of cards ADD54 to ADD57 and before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Cards ADD58, ADD59 and ADD61 were not read when this card was frozen, and it uses none of them. Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known, and the lever

* MEASURED, T1-kin, 2.0 s cycle (rows 101 to 103, 107, 113): the lead rule cuts the unsafe rate U of the two 10k models (no rule 0.1459, 0.1139; label-fed with the closing speed 0.0463, 0.0302; reader-fed at level 0.50 0.0872, 0.0730). Every window the label-fed rule leaves unsafe has the rule at its braking limit at the tick in force, and all are of the time-gap kind (`integ10k/read_gaprate/UNSAFE_CAUSE.json`). A longer time gap does not cut U further (row 107), a higher braking limit does not either (row 113).
* What is left for those windows is the time between two looks of the rule at the lead. Rows 108 and 110 to 112: the models are not stable when the MODEL replans every 1.0 s. That mixes two rates that need not be one: how often the model plans, and how often the rule checks the plan in force against the lead.
* This card separates them. The model plans every 2.0 s as in the programme's convention and with the pinned readout. The rule, which only ever lowers speed along the plan's own path (cap-only form), is applied again inside the block to the part of the plan that is still to be driven, with the lead observation of that later moment. No state readout is changed and no plan of the model is touched except by the rule.

## 2. The mechanism (study wrapper; the pinned tool is not edited)

New wrapper `integ10k/v8f_t1_eval_reader_rule_rzf.py`, from the wrapper of card ADD55 by asserted text edits: one more parameter, `--rule-every h` (executed steps; default 10 = the rule at the tick only, the behaviour of every earlier card).

For h below 10, at a tick at frame j with the rule's plan P1 (the model's plan after the rule at the tick, as before), for every later step q = h, 2h, ... below 10:

* the car is at knot q of the plan built so far (the executor follows the plan exactly in this tier);
* the rule's inputs are taken as a tick at frame j + q would take them: the label read at frame j + q (rows at or before that frame only) or the reader's read of the logged frame j + q; the logged speed and the logged arc at that step; the car's progress along the logged path at knot q; the car's speed = the chord speed of the last step before q;
* the same rule (same D, time gap, braking limit 6.0, cap-only) is applied to the remaining plan (knots after q, taken relative to knot q), with the executed-step count 10 - q; steps up to q are not changed.

The plan committed at the tick is the result, one commit of 10 steps as before. Applying the later looks when the plan is built is the same as applying them at their moment in this tier: the lead follows its logged track, the reader reads logged frames only, and the car's position at step q is fixed by the plan. The later looks are written to a log of their own (`ticks/rule_log_mid_<run>.npz`), so the rule log of the ticks keeps one row per window and tick.

## 3. Arms

Models `UE_s0`, `UE_s1`, reader arms, maps (level 0.50), rule parameters, evaluation cache, window grid and pinned tool as cards ADD46 to ADD57; replan interval of the model 10 steps; pinned readout; no limit on the model's plan.

| run | form | rule every | role |
|---|---|---|---|
| `RLcF5` | label-fed rule with the closing speed | 5 steps (1.0 s) | the rule twice per plan at label quality |
| `RQ50F5` | reader-fed rule at level 0.50, closing 0 | 5 steps | the same with the reader |
| `RLcF2` | label-fed | 2 steps (0.4 s) | dose; names nothing |
| `RLcFID`, model seed 0 | label-fed | 10 steps given as a parameter | control k1: must equal the banked `RLc` |

Each run except the control at both model seeds: seven rollouts, started after the queue of card ADD61 has ended. Reused without a rerun: the banked `R0`, `RLc`, `RQ50` (2.0 s, rule at the tick only) and, for control k3 only, the rule logs of the banked 5-step runs `RLcM5`, `RQ50M5` of card ADD52.

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (56 lead episodes, 642 lead windows, 562 scored for U; 141 episodes and 2,399 windows for the families).

* FR (per form) = U(rule every 5 steps) minus U(banked, rule at the tick only); contact C and falling behind F of the same pair beside; all family cells beside (LONGITUDINAL, LATERAL, TACTICAL), per family, never pooled into one score.
* Dose, label-fed: rule every 2 steps minus the banked run, and minus rule every 5 steps.
* Beside: each new run against no rule; reader-fed minus label-fed, both with the rule every 5 steps.
* Reported without a bar, from the logs: how many later looks changed the plan, how many reached the braking limit on a step still to be driven, per run; and the saturated executed steps of the tick rows beside those of the banked run.
* Committed now, descriptive, names nothing: the split of the windows that stay unsafe (convention of the `TIMES_NOTE.md` line of 2026-11-19T16:28:23Z) on the new runs, with the windows removed and added against the banked run of the same form.

## 5. Names (fixed here)

Per form (LABEL, READER):

* `RULEFAST10K-<form>-FASTER-RULE-CUTS-UNSAFE` if FR is below 0 at both model seeds and its pooled interval lies below 0.
* `RULEFAST10K-<form>-FASTER-RULE-RAISES-UNSAFE` if FR is above 0 at both model seeds and its pooled interval lies above 0.
* `RULEFAST10K-<form>-NO-U-CHANGE` otherwise.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument). The first tick's model outputs do not depend on the rule's rate.

What follows, fixed now:

* LABEL reads CUTS-UNSAFE: the rule's rate is a live parameter of the safety layer and it needs no change of the model's cycle or of the state readout. It goes into the specification of the integrated candidate as a proposal (the rule at 1.0 s over the 2.0 s plan), with its price (F, families) beside it; the dose arm says whether a faster rule still gains. Changing the specification of the safety layer is the PI's decision; this card changes nothing by itself.
* LABEL reads NO-U-CHANGE: the windows at the braking limit are not reached by a second look inside the block either. Reported as written with the split; the remaining lever on this line is then the model's own plan (the speed it asks for behind a lead), which is the lead supervision of the other session's line (cards ADD45, ADD60).
* READER: reported as written; the part of its unsafe windows where the reader read no lead stays a matter of reader quality.
* A name RAISES-UNSAFE: reported as written; the split shows which windows were added.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* e1: the pod read reproduces the banked dev-box read of card ADD46 leaf for leaf.
* k1 (wrapper edit, known value): the dump of `RLcFID` (rule every 10 steps given as a parameter, model seed 0) equals the banked dump of `RLc` at model seed 0 array for array (141 files, 0 differing).
* k2 (the later looks never change what was already driven, and the first look is the old rule): at the first tick, where the inputs are those of the banked run of the same form and model seed, the executed steps 1 to h of every window equal those of the banked run (largest difference 0.0), and the tick rows of the rule log equal the banked ones in their inputs.
* k3 (the later look reads what a real tick at that frame reads): at the first tick, the inputs of the look at step 5 (label read or reader read, logged speed, logged arc, the car's progress, the corrected gap) equal the inputs of the second tick (frame j + 5) of the banked 5-step run of the same form and model seed in every window (largest difference at most 1e-6). Both runs have driven the same first five steps when control k2 holds for the 5-step run's first block; the reader says for how many windows that premise holds and compares on those.
* k4 (cap only): in every later look of every run, no step of the remaining plan is farther along the path than in the plan before the look (largest excess at most 1e-9 m), and with the rule every 2 steps the plan is never ahead of the plan with the rule every 5 steps at the first tick's block.
* k5 (the rule's rate is live and acts in the known direction, deliberate check on identical inputs): tool check before any run, on synthetic plans behind a lead that brakes after the tick: with the rule every 5 steps the car ends the block farther from the lead than with the rule at the tick only, and with the lead holding its speed the two plans are equal. The same check must fail when the later look is fed the tick's own inputs (the planted fault).
* k6 (counts): 141 dump files in each new dump, 2,399 windows, 642 lead windows; one tick row per window and tick; later-look rows = windows x ticks x (10 / h - 1).
* k7 (reused dumps), k8 (reader controls of the headway and family readers on the new pairs), k9 (checkpoint sha256 of each new run equals that of card ADD46; the wrapper of the runs has the sha256 the tool check ran with).

## 7. Limits

T1-kin: the lead does not react, the executor follows every plan, no vehicle model. The later looks use the logged frame of their moment, which in this tier is what the camera shows at that moment; in a tier with re-perception they would see the rollout's own view. The rule only lowers speed; what a faster rule costs in comfort is not scored by any family. Two model seeds; 56 lead episodes are read again, with no adjustment for the repeated reads. The card was written knowing the split of the unsafe windows and the reads of the time-gap and braking-limit cards, which is why this parameter was chosen. The evaluation convention of the programme and its pinned tool are not changed; adopting a rule rate in the safety layer is the PI's and the other session's decision.
