# Card PREREG_V8F_PROOF_ADD69 (CYCLEBOUND10K): the 1.0 s planning cycle on a kinematic executor with a heading bound; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD67 (row 122) and before the read of card ADD68 (its rollouts are on the pod and unread), before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance.

## 1. What is known (MEASURED)

* On the executor without a heading bound, shortening the model's planning cycle from 2.0 s to 1.0 s raises the unsafe rate under every readout that was tried (rows 114 to 116). Coordinator decision D12 kept the 2.0 s cycle for every capability number on that basis.
* The executed heading of the pinned tool follows millimetres of plan jitter near a standstill, and every replan lays the next plan in the frame of that heading. A bound of 0.2 1/m on the heading lowers the convention's errors on all four families and its unsafe rate (row 122). The PI approved a heading bound for the T1-kin tier on 2026-11-20 (decision card D16, recommendation D).
* Not measured: the 1.0 s cycle on the bounded executor. A shorter cycle reads the unbounded heading twice as often, so the cost of rows 114 to 116 may be the heading's and not the cycle's.

## 2. Question

On the bounded executor (0.2 1/m), does the 1.0 s cycle read worse, equal or better than the 2.0 s cycle, family by family and on the unsafe rate: (a) in the convention without a rule, (b) under readout K, (c) with the lead rule under the pinned readout, (d) in the candidate (readout K with the rule at every plan, against the 2.0 s candidate whose rule takes a second look after 1.0 s, so that the rule is applied at the same rate in both)?

## 3. Mechanism

The wrapper of card ADD67, unchanged: `integ10k/v8f_t1_eval_reader_rule_rzgfh.py`, sha256 373ffe72f2655c35, with `--replan-steps 5` of the pinned tool. No new tool code for the rollouts. The pinned tool is not edited.

## 4. Arms

Models `UE_s0`, `UE_s1`, reader arms, maps, rule parameters, evaluation cache, window grid and pinned tool as cards ADD46 to ADD68; every new run at a cycle of 5 cache steps (1.0 s) with the bound 0.2 1/m.

| run | rule | readout | role |
|---|---|---|---|
| `R0M5Hb20` | none | pinned | (a) |
| `R0KM5Hb20` | none | K | (b) |
| `RLcM5Hb20` | label-fed, at every plan | pinned | (c) |
| `RQ50M5Hb20` | reader-fed, at every plan | pinned | (c), deployable form |
| `RLcKM5Hb20` | label-fed, at every plan | K | (d) |
| `RQ50KM5Hb20` | reader-fed, at every plan | K | (d), deployable form |
| `R0M5HbID`, model seed 0 | none | pinned, bound 1e9 | control y1: must equal the banked `R0M5` |

Each run except the control at both model seeds: thirteen rollouts. Reused without a rerun: `R0`, `RLc`, `RQ50`, `R0M5`, `RLcM5`, `RQ50M5` (banked, no bound), from card ADD67 `R0Hb20`, `R0KHb20`, `RLcKF5Hb20`, `RQ50KF5Hb20`, and from card ADD68 `RLcHb20`, `RQ50Hb20` (admitted only if the controls of card ADD68 pass; else the pairs of (c) are not named and the reason is stated).

## 5. Reads

Headway reader and family reader of card ADD46, as card ADD67 section 5 (same windows, estimators and family rule; named cells LONGITUDINAL speed and along-track error, LATERAL heading and yaw-rate error, TACTICAL tactical ADE; never pooled over families). STRATEGIC: the instrument of card ADD66 with Amendment 1, cell `pos` at 30, 50, 80 m at the tick 4.0 s into the window (cache step t0+20) in both cycles. In a 2.0 s run this is the third tick, as before; a 1.0 s run writes six ticks per window, of which the instrument reads those at t0, t0+10 and t0+20 (the strategic output at t0+20 was computed at t0+18 in both cycles: control y4).

Pairs that name (the 1.0 s run minus the 2.0 s run, both with the bound):

* CYC-PINNED = `R0M5Hb20` minus `R0Hb20`; CYC-K = `R0KM5Hb20` minus `R0KHb20`; CYC-LABEL-RULE = `RLcM5Hb20` minus `RLcHb20`; CYC-READER-RULE = `RQ50M5Hb20` minus `RQ50Hb20`; CYC-LABEL-CAND = `RLcKM5Hb20` minus `RLcKF5Hb20`; CYC-READER-CAND = `RQ50KM5Hb20` minus `RQ50KF5Hb20`.

Beside, naming nothing: the bound at a 1.0 s cycle (`R0M5Hb20` minus `R0M5`, `RLcM5Hb20` minus `RLcM5`, `RQ50M5Hb20` minus `RQ50M5`); the cycle without a bound re-read on the same reader (`R0M5` minus `R0`, `RLcM5` minus `RLc`, `RQ50M5` minus `RQ50`; the pairs of rows 114 to 116). Reported without a bar, as card ADD67: the share of executed frames at which the bound acted by speed bin and the heading offset at t0+20 by speed bin, per run.

## 6. Names (fixed here)

For each pair P of the six: `CYCLEBOUND10K-<P>-<FAMILY>-BETTER`, `-WORSE`, `-MIXED` or `-NOT-SEPARATED` for LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC, and `CYCLEBOUND10K-<P>-U-LOWER`, `-U-HIGHER` or `-U-NOT-SEPARATED` (one sign at both model seeds and the pooled interval on that side of 0).

What follows, fixed now, per pair:

* WORSE on at least one family or U higher, and BETTER on no family: the 1.0 s cycle costs on the bounded executor too in that setting; coordinator decision D12 (2.0 s for every capability number) stands there, now measured on the tier the PI approved.
* BETTER on at least one family, WORSE on none and U not higher: on the bounded executor the shorter cycle is a gain in that setting, and the cost of rows 114 to 116 was the heading's. The 1.0 s cycle then goes to the PI and the other session as a proposal for that setting; adopting it changes the evaluation convention and doubles the planner calls (the latency budget on the target hardware is theirs to weigh). Nothing is adopted by this card.
* All four families NOT-SEPARATED and U not separated: the cycle does not move the reads on the bounded executor in that setting; 2.0 s stays (half the planner calls).
* Anything else (MIXED, or better on one family with U higher): reported as written with the table; nothing is proposed for that setting.
* A control fails: nothing is named; the fault is logged in `TIMES_NOTE.md` before its repair and the runs are repeated under new run names.

## 7. Controls (all must pass; else nothing is named)

* e1: the pod read reproduces the banked dev-box read of card ADD46 leaf for leaf.
* y1 (known value): the dump of `R0M5HbID` equals the banked dump of `R0M5` at model seed 0 array for array (141 files, 0 differing); marker names no earlier card used (`cb1cmp`; the tool check `cb6check`).
* y2 (the bound holds and does nothing else): as control h3 of card ADD67 for every run; the readout log of every run records the readout, the gain, the path and the bound of the table, and its run record a cycle of 5 steps.
* y3 (tool checks before any run): the three tool checks of card ADD67 run again on the pod CPU against the same wrapper (planted faults caught); the wrapper of every run has the sha256 373ffe72f2655c35.
* y4 (the strategic cell, known values): on every 2.0 s run read by this card the selection of ticks by cache step gives exactly the three ticks the instrument read before, and the pair `R0KHb20` minus `R0G0X` of card ADD67 is reproduced to the last digit; in every 1.0 s run each window has six ticks at t0, t0+5, ..., t0+25 and the strategic output read at t0+10 and t0+20 has the ages 4 and 2 cache steps, as in the 2.0 s runs.
* y5 (the first block, known value): the first 5 executed positions of every run equal those of the banked 1.0 s run with the same rule form and model seed (`R0M5`, `RLcM5`, `RQ50M5`; largest difference 0.0).
* y6 (counts), y7 (reused dumps unchanged since they were read, including the runs of cards ADD67 and ADD68), y8 (reader controls of the headway and family readers and controls s1 to s9 of card ADD66 on the runs read), y9 (checkpoint sha256).

## 8. Limits

T1-kin: no vehicle model beyond the one bound; positions follow the plan's steps exactly; the lead does not react; logged camera frames. Two model seeds of one training recipe (the bootstrap answers another draw of episodes; the seeds are the only replicate for training variance; the rollout does not sample). The models were trained at one cycle; a 1.0 s cycle is a use outside that schedule only in how often the planner is called. In (d) the two candidates differ in more than the cycle: at 1.0 s the planner replans where at 2.0 s only the rule looks again. The heading and yaw-rate cells of the family reader mask windows without a finite value in both arms, so their differences do not add across pairs. The episodes are read again with no adjustment for the repeated reads. The STRATEGIC cell is the study instrument of card ADD66 adopted by the PI as the T1-kin strategic read (route goal; a decision between routes is not scored, n = 0). The card was written knowing rows 110 to 122 and nothing of card ADD68.
