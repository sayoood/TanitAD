# Card PREREG_V8F_PROOF_ADD68 (BOUNDTIER10K): the readout, the lead rule and the candidate on a kinematic executor with a heading bound; doses of the bound in the convention; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD67 (row 122), before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance.

## 1. What is known (MEASURED, rows 119 to 122)

* Without a bound, the family errors rise with the share of its own acceleration the planner reads at the replan tick; the causal readout with all of it (what a car would measure) is the worst and readout K (none of it, path from executed steps only) the best (row 119).
* The executed heading of the pinned tool follows the tangent of the executed path down to 1 mm per step. A bound of 0.2 1/m on the heading change per metre lowers the convention's errors on all four families and its unsafe rate, and still improves the strategic read under readout K and in the candidate; no family worse (row 122).
* Not measured: whether the readout term of row 119 stands once the executor has the bound; the convention with the lead rule on the bounded executor; the candidate against that; the bound at other values in the convention (one dose, 0.1, was run under readout K only).

## 2. Questions

(a) Does the bound help under the causal readout with all of the own acceleration, on the pinned path and on the path from executed steps only?
(b) On the bounded executor, does the planner still pay for reading its own acceleration (all of it against none of it, both on the executed path)?
(c) On the bounded executor, does readout K still read better than the pinned readout, and does the candidate of row 121 still read better than the convention with the rule?
(d) Where does the bound start to cut turns a car makes, and where does it stop removing the offset (doses in the convention)?

## 3. Mechanism

The wrapper of card ADD67, unchanged: `integ10k/v8f_t1_eval_reader_rule_rzgfh.py`, sha256 373ffe72f2655c35 (`--kappa-max K` applied to the heading column after the readout; without the parameter the parent's result bit for bit). No new tool code for the rollouts. The pinned tool is not edited.

## 4. Arms

Models `UE_s0`, `UE_s1`, reader arms, maps, rule parameters, evaluation cache, window grid and pinned tool as cards ADD46 to ADD67; 2.0 s cycle; no limit on the plan.

| run | rule | readout | bound | role |
|---|---|---|---|---|
| `R0C10Hb20` | none | causal, all of the own acceleration, pinned path (`--readout causal`) | 0.2 | (a) against the banked `R0C10` |
| `R0C10X` | none | causal, all of the own acceleration, path from executed steps only (`--readout causal_g --acc-gain 1 --path-exec`) | none | (a) reference, new |
| `R0C10XHb20` | none | the same | 0.2 | (a) against `R0C10X`; (b) against `R0KHb20` |
| `RLcHb20` | label-fed, once per plan | pinned | 0.2 | the convention with the rule on the bounded executor |
| `RQ50Hb20` | reader-fed, once per plan | pinned | 0.2 | the same in its deployable form |
| `R0Hb05`, `R0Hb10`, `R0Hb40` | none | pinned | 0.05, 0.1, 0.4 | (d) doses, name nothing |
| `R0C10HbID`, model seed 0 | none | causal, pinned path | 1e9 | control g1: must equal the banked `R0C10` |
| `RLcHbID`, model seed 0 | label-fed | pinned | 1e9 | control g2: must equal the banked `RLc` |

Each run except the controls at both model seeds: eighteen rollouts. Reused without a rerun: `R0`, `R0C10`, `RLc`, `RQ50` and, from card ADD67 (its controls passed), `R0Hb20`, `R0KHb20`, `RLcKF5Hb20`, `RQ50KF5Hb20`.

## 5. Reads

Headway reader and family reader of card ADD46 and the strategic instrument of card ADD66 with Amendment 1, exactly as card ADD67 section 5 (same windows, same estimators, same family rule: named cells LONGITUDINAL speed and along-track error, LATERAL heading and yaw-rate error, TACTICAL tactical ADE, STRATEGIC `pos` at 30, 50, 80 m at the third tick; BETTER only if every named cell reads better; never pooled over families).

Pairs that name (first run minus second run):

* The bound in a setting: CAUSAL = `R0C10Hb20` minus `R0C10`; CAUSALX = `R0C10XHb20` minus `R0C10X`; LABEL-RULE = `RLcHb20` minus `RLc`; READER-RULE = `RQ50Hb20` minus `RQ50`.
* On the bounded executor: ACC = `R0C10XHb20` minus `R0KHb20` (all of the own acceleration against none, both on the executed path); READOUT-K = `R0KHb20` minus `R0Hb20`; LABEL-CAND = `RLcKF5Hb20` minus `RLcHb20`; READER-CAND = `RQ50KF5Hb20` minus `RQ50Hb20`.

Beside, naming nothing: each dose minus `R0Hb20`; `R0C10X` minus `R0C10` (the path under the causal readout, no bound); `R0C10XHb20` minus `R0Hb20`. Reported without a bar, as card ADD67: the share of executed frames at which the bound acted by speed bin, and the heading offset at the third tick by speed bin, per run.

## 6. Names (fixed here)

For each pair P of the eight: `BOUNDTIER10K-<P>-<FAMILY>-BETTER`, `-WORSE`, `-MIXED` or `-NOT-SEPARATED` for LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC, and `BOUNDTIER10K-<P>-U-LOWER`, `-U-HIGHER` or `-U-NOT-SEPARATED` (one sign at both model seeds and the pooled interval on that side of 0).

What follows, fixed now:

* CAUSAL, CAUSALX, LABEL-RULE, READER-RULE, each by the rule of card ADD67 section 6: BETTER on at least one family, WORSE on none and U not higher means the unbounded heading is a measured term of the T1-kin errors in that setting too, and the setting joins the proposal for the definition of the tier (decision D of card D16). A family WORSE or U higher in a setting: reported as written, and the proposal is stated without that setting.
* ACC reads WORSE on at least one family and BETTER on none: the own-acceleration term of row 119 stands on the bounded executor; the training lever of the other session (decision C of card D16) stays the repair at the source and readout K stays a study device inside the candidate.
* ACC reads NOT-SEPARATED on all four families and U not higher: on the bounded executor the planner takes the state a car would measure at no measured cost; the readout term of row 119 was carried by the unbounded heading. Recommendation A of card D16 is then restated with the causal readout, and a candidate with it gets a card of its own before anything is proposed.
* ACC reads BETTER on a family, or MIXED: reported as written with the table.
* READOUT-K: BETTER on at least one family and WORSE on none means the gain of readout K holds on the bounded executor; all four NOT-SEPARATED means that gain was the heading's; reported either way.
* LABEL-CAND and READER-CAND: BETTER on at least one family, WORSE on none and U not higher means the candidate's reading against the convention with the rule holds when both have the bound, and recommendation A of card D16 stands under decision D. Otherwise it is reported as written and recommendation A is restated for the bounded executor.
* Doses: a dose that reads worse than 0.2 on a family is said as the point where the bound starts to cut turns a car makes (0.05, 0.1) or stops removing the offset (0.4). No value is proposed from the doses by this card.
* A control fails: nothing is named; the fault is logged in `TIMES_NOTE.md` before its repair and the runs are repeated under new run names.

## 7. Controls (all must pass; else nothing is named)

* e1: the pod read reproduces the banked dev-box read of card ADD46 leaf for leaf.
* g1, g2 (known values): the dumps of `R0C10HbID` and `RLcHbID` equal the banked dumps of `R0C10` and `RLc` at model seed 0 array for array (141 files, 0 differing each); marker names no earlier card used (`bt1cmp`, `bt2cmp`; the tool check `bt6check`).
* g3 (the bound holds and does nothing else): as control h3 of card ADD67 for every run with a bound; and the readout log of every run records the readout, the gain, the path and the bound of the table above.
* g4 (tool checks before any run): the three tool checks of card ADD67 run again on the pod CPU against the same wrapper (planted faults caught); the wrapper of every run has the sha256 373ffe72f2655c35.
* g5 (the first block, known value): the first 10 executed positions of every run equal those of a banked run with the same rule form and model seed (largest difference 0.0).
* g6 (counts), g7 (reused dumps unchanged since they were read, including the four runs of card ADD67), g8 (reader controls of the headway and family readers and controls s1 to s9 of card ADD66 on the runs read), g9 (checkpoint sha256).

## 8. Limits

T1-kin: no vehicle model beyond the one bound; positions follow the plan's steps exactly; the lead does not react; logged camera frames. Two model seeds of one training recipe (the bootstrap answers another draw of episodes; the seeds are the only replicate for training variance; the rollout does not sample). The heading and yaw-rate cells of the family reader mask windows without a finite value in both arms, so their differences do not add across pairs. The episodes are read again with no adjustment for the repeated reads. The STRATEGIC cell is the study instrument of card ADD66 (route goal; a decision between routes is not scored, n = 0). The card was written knowing rows 110 to 122.
