# Card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K): how much of its own acceleration the model is handed at a replan tick, as a dose, at the programme's 2.0 s cycle; no rule, no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD59 (row 114) and before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Cards ADD58, ADD61 and ADD62 were not read when this card was frozen, and it uses none of them. Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions); never driving performance.

## 1. What is known

All MEASURED, T1-kin, the two 10k models `UE_s0`, `UE_s1` without a rule, 2.0 s cycle, against the pinned readout (the banked `R0`):

* The pinned tool fills the frames of the executed steps by a spline with central tangents, so the speed at the last executed knot is the mean of the last executed step and the next step of the plan, which is never executed. The state the model reads at a replan tick therefore holds a part of the plan it is about to discard.
* Causal readout (`R0C10`, card ADD54, row 110; `integ10k/read_handover/pod/READ_HANDOVER10K.json`): the speed column from executed steps only, so the one-frame acceleration at the tick is the executed one. Against pinned: speed error +1.309 m/s [+0.986, +1.669], along-track +1.089 m, heading +5.00 deg, yaw rate +42.7 deg/s [+36.9, +49.3], tactical ADE +1.551 m [+1.264, +1.857], all worse; falling behind +0.2118 [+0.1582, +0.2686]; U not separated.
* Backward readout (`R0Bk10`, card ADD59, row 114): the pinned function given the executed knots only; it hands over an acceleration of the opposite sign and about a quarter of the size of the executed one, and a heading tangent from executed knots. Against pinned: speed error -0.483 m/s [-0.580, -0.397], along-track -0.297 m, cross-track -0.287 m, heading -2.36 deg, yaw rate -11.8 deg/s [-14.6, -9.5], tactical ADE -0.645 m [-0.862, -0.462], all better; U not separated.
* At the 1.0 s cycle (rows 110 to 112, 114): with the executed acceleration the model's speed climbs; with the acceleration held at 0 (ZERO) or with the backward readout it does not.

Three readouts, three sizes of the acceleration handed over, and the family errors are ordered the same way. That is a pattern from three points that also differ in other things (the backward readout changes the heading tangent too). This card measures the dose directly and separates the path part.

## 2. Hypothesis and the lever

HYPOTHESIS: at the programme's cycle the family errors of the rollout rise with the share of its own executed acceleration the model reads at the tick (the model continues the acceleration it reads; on its own rollout that feeds back). If it holds with a dose, the term is the model's response to its own acceleration, which is a matter of training (own rolled history) and, on the inference side, of what the planner is handed. Both outcomes are committed in section 6.

## 3. The mechanism (study wrapper; the pinned tool is not edited)

New wrapper `integ10k/v8f_t1_eval_reader_rule_rzg.py`, from the wrapper of card ADD55 (`v8f_t1_eval_reader_rule_rz.py`, sha256 fa7207d72311b5cf) by asserted text edits: one more readout, `causal_g`, with two parameters.

* `--acc-gain g`: positions and headings as pinned; the speed column from executed steps only: at the knot of executed step i the step speed s[i], at the half step before it s[i] - 0.5 x g x (s[i] - s[i-1]). The one-frame acceleration at every executed knot, and so at the tick, is g times the executed one. g = 1 is the causal readout of card ADD54 and g = 0 the ZERO readout of card ADD55, bit for bit (controls a1, a2).
* `--path-exec` (off by default): positions and headings from the pinned frames function given the executed knots only (as the backward readout does), the speed column as above. With it nothing the model reads at the tick comes from the unexecuted part of the old plan.

Every other readout of the parent stays bit for bit.

## 4. Arms

Models `UE_s0`, `UE_s1`, evaluation cache, window grid and pinned tool as cards ADD46 to ADD59; no rule (form `R0`); replan interval 10 steps (2.0 s); no limit on the plan.

| run | acceleration gain g | path | role |
|---|---|---|---|
| `R0G0` | 0 | pinned | no own acceleration at the programme's cycle |
| `R0G50` | 0.5 | pinned | half of the executed acceleration |
| `R0Gm25` | -0.25 | pinned | the size the backward readout hands over, without its path part |
| `R0G0X` | 0 | executed knots only | no own acceleration and no part of the discarded plan |
| `R0GID1`, model seed 0 | 1 | pinned | control a1: must equal the banked `R0C10` |
| `R0GID0`, model seed 0, replan interval 5 steps | 0 | pinned | control a2: must equal the banked `R0Z5` |

Each run except the controls at both model seeds: ten rollouts, started after the queue of card ADD62 has ended. Reused without a rerun: `R0` (pinned), `R0C10` (g = 1), `R0Bk10` (backward), and for control a2 `R0Z5`.

## 5. Reads

Family reader and headway reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (141 episodes and 2,399 windows for the families; 56 lead episodes, 642 lead windows, 562 scored for U). The bootstrap answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample.

Named cells per family (the error cells in physical units that moved in both known pairs of section 1): LONGITUDINAL speed error and along-track error; LATERAL heading error and yaw-rate error; TACTICAL tactical ADE. All other cells (cross-track, curvature, the three tactical error rates) are reported beside, per family, never pooled into one score.

* Dose: the two adjacent differences D1 = `R0G50` minus `R0G0` and D2 = `R0C10` minus `R0G50`, per cell; U, C, F beside.
* ZP = `R0G0` minus the banked pinned `R0` (what handing over no own acceleration does against the programme's convention): cells and U, C, F.
* PX = `R0G0X` minus `R0G0` (the path part alone): cells and U, C, F.
* Beside, naming nothing: `R0Gm25` minus `R0G0` (does a damping acceleration gain beyond zero), `R0Gm25` minus the banked `R0Bk10` (how much of the backward readout is its acceleration), `R0G0X` minus the banked pinned `R0`, and each new run against the banked `R0C10`.
* Reported without a bar, from the readout logs: the mean and the largest absolute acceleration read per run; and RS, the share of the 2,399 windows that end more than 5 m/s above the logged speed, per run and model seed, with the banked values beside.

A cell reads BETTER or WORSE by the family reader's own rule (one sign at both model seeds and a pooled interval on that side of 0); a family reads BETTER for a pair if every named cell of it reads BETTER, WORSE if every named cell reads WORSE, NOT-SEPARATED if no named cell reads BETTER or WORSE, else MIXED.

## 6. Names (fixed here)

* Dose, per family: `ACCGAIN10K-<FAMILY>-RISES-WITH-OWN-ACCELERATION` if the family reads WORSE for D1 and for D2; `ACCGAIN10K-<FAMILY>-FALLS-WITH-OWN-ACCELERATION` if it reads BETTER for both; else `ACCGAIN10K-<FAMILY>-NOT-MONOTONE`.
* Dose, U: `ACCGAIN10K-U-RISES-WITH-OWN-ACCELERATION` if U of D1 and of D2 is above 0 at both model seeds with both pooled intervals above 0; `-FALLS-` if both lie below 0 the same way; else `ACCGAIN10K-U-NOT-MONOTONE`.
* No own acceleration against the convention, per family: `ACCGAIN10K-ZERO-<FAMILY>-BETTER-THAN-PINNED-AT-2S`, `-WORSE-THAN-PINNED-AT-2S`, `-MIXED-AT-2S` or `-NOT-SEPARATED-FROM-PINNED-AT-2S` by the family rule on ZP; and `ACCGAIN10K-ZERO-U-BELOW-PINNED-AT-2S`, `-ABOVE-PINNED-AT-2S` or `-NOT-SEPARATED-FROM-PINNED-AT-2S` by the sign of U of ZP at both model seeds and its pooled interval.
* Path part, per family: `ACCGAIN10K-PATH-<FAMILY>-BETTER`, `-WORSE`, `-MIXED` or `-NOT-SEPARATED` by the family rule on PX.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick does not depend on the readout.

What follows, fixed now:

* LONGITUDINAL reads RISES-WITH-OWN-ACCELERATION: the model's response to the acceleration of its own rollout is a term of the T1-kin errors at the programme's cycle, with a dose, and the honest readout (g = 1, what a car would measure) is the worst of them. This is a property of models trained on logged histories and it goes to the other session as the measured case for the training lever (own rolled history at the tick). It is not an instrument fault that a readout repairs.
* ZERO reads BETTER-THAN-PINNED for at least the LONGITUDINAL family with U not above pinned: handing the planner its measured speed without the acceleration of its own rollout is a candidate for the specification of the integrated candidate, causal in the speed column, with its measured effect. The next arm here is then the three rule forms under it at 2.0 s on their own card. Adopting it changes what the planner reads at inference; that is the PI's and the other session's decision, and this card changes nothing by itself.
* ZERO reads WORSE or MIXED, or U above pinned: reported as written; then no readout of this family is proposed, the decision card lists pinned, backward, causal and ZERO with their measured effects side by side, and the training lever stands alone.
* A family reads NOT-MONOTONE or FALLS: reported as written with the table; the ordering of the three known readouts is then not a matter of the acceleration alone and the path read PX says what else moves.
* PATH reads BETTER or WORSE for a family: the heading tangent taken from the discarded plan is a second term of the handover, and a state rebuild has to treat both parts; PATH NOT-SEPARATED for all three: the path part of the handover is not a term at this cycle.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.
* The evaluation convention of the programme (2.0 s, pinned readout) and its pinned tool are not changed by this card.

## 7. Controls (all must pass; else nothing is named)

* e1: the pod read reproduces the banked dev-box read of card ADD46 leaf for leaf.
* a1 (wrapper edit, known value): the dump of `R0GID1` (g = 1, model seed 0) equals the banked dump of `R0C10` at model seed 0 array for array (141 files, 0 differing). The queue step uses a marker name no earlier card used.
* a2 (wrapper edit, known value): the dump of `R0GID0` (g = 0, 5 steps, model seed 0) equals the banked dump of `R0Z5` at model seed 0 array for array (141 files, 0 differing), with a marker name of its own.
* a3 (the gain is applied, from the readout log over all committed plans of each run): the speed at the last executed knot equals the last executed step speed (largest difference at most 1e-6), and the one-frame acceleration read equals g times the executed one (largest difference at most 1e-6 m/s2).
* a4 (the gain is live and acts in the known direction, tool check before any run on synthetic knots with a known speed change): the acceleration read is g times the executed one for g = -0.25, 0, 0.5, 1; with `--path-exec` the positions and headings equal those of the backward readout and the speed column equals that of the same gain without it; the check must fail when the gain is ignored and when the path is taken from the pinned call (two planted faults).
* a5 (the first block does not depend on the readout, known value): the first 10 executed steps of every new 10-step run equal those of the banked `R0` at the same model seed (largest difference 0.0), for the model and for the prior.
* a6 (counts): 141 dump files in each new dump, 2,399 windows, 642 lead windows.
* a7 (reused dumps): the result files of the reused runs have the sha256 recorded before the runs (`integ10k/ACCGAIN10K_REUSED.json`).
* a8: the controls of the family reader and of the headway reader on the new pairs.
* a9: the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed; the wrapper on the pod has the sha256 the tool check ran with.

## 8. Limits

T1-kin: no vehicle model, the executor follows every plan, other agents do not react. The gain scales the one-frame acceleration at executed knots and nothing else; the model also reads positions and headings of the executed frames, which carry the executed motion in another form. With the pinned path the heading at the last executed knot still comes from the central tangent; only `R0G0X` removes that. Every readout here is a study device and none is a capability number at any cycle other than through the comparison it names (coordinator decision D12 stands for the 1.0 s cycle). Two model seeds; 141 episodes read again with no adjustment for the repeated reads. The card was written knowing rows 110 to 114, which is why the dose was chosen; the named cells were picked from those rows.
