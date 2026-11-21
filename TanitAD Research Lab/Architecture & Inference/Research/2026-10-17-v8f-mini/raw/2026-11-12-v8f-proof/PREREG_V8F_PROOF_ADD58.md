# Card PREREG_V8F_PROOF_ADD58 (ACCWIN2S10K): the acceleration read at a replan taken over the last 2.0 s of executed motion

Card of the v8F proof study, written by the coordinator fork. Frozen before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). No training, no lead rule, no spend beyond the pod of this fork. Tier: T1-kin (kinematic rollout on the model's own actions); never driving performance.

## 1. What is known

* Card ADD55 (MEASURED, read 2026-11-19, `integ10k/read_accwin/pod/READ_ACCWIN10K.json`, controls z1 to z9 PASS): with the acceleration at a replan held at 0 the two 10k models are stable at the 1.0 s cycle (`ACCWIN10K-MODEL-ZERO-STABILISES-1S`: share of windows ending more than 5 m/s above the logged speed 0.1351, 0.1409 against the bars 0.1768, 0.1726); with the acceleration over the last 1.0 s of executed motion the climb is only slowed (`ACCWIN10K-MODEL-WINDOW-SLOWS-CLIMB`: 0.3331, 0.4173), and at the 2.0 s cycle that window costs (`ACCWIN10K-WINDOW-COSTS-AT-2S`).
* Card ADD55 fixed for this outcome: the next arm is the window length 2.0 s on its own card. This is that card.
* In the wrapper of card ADD55 the window never reaches back past the start of the current block (at most 1.0 s at the 1.0 s cycle).

## 2. Question

Does an acceleration estimate over 2.0 s of executed motion, reaching back across replans, steady the loop at the 1.0 s cycle, and what does it do at the programme's 2.0 s cycle?

## 3. Arms (no rule, no limit, no training)

Models `UE_s0`, `UE_s1`, evaluation cache, window grid and pinned tool as cards ADD46 to ADD57. Wrapper `integ10k/v8f_t1_eval_reader_rule_rv.py`, made by a generator from the wrapper of card ADD55 by asserted text edits (each old text exactly once); the pinned tool is not edited. One more readout:

* `causal_v` (WINDOW2): positions and headings of the pinned frames, the causal speed at every executed knot (the step speed into it), and at every executed step i of a block the one-frame acceleration (s_i - s_{i-w}) / (0.2 s x w), where s runs over the steps of the rollout's own track: the executed steps of the current block, before them the executed steps of the earlier blocks, and before the first block the logged steps the window started from; w = 10 where ten earlier steps exist in the track, else the number that exist (at least 1). The step speed is the distance between consecutive track knots (0.2 s apart) over 0.2 s, as in the wrapper of card ADD55.
* All other readouts of the parent wrapper stay bit for bit.

| run | readout | replan interval | role |
|---|---|---|---|
| `R0V5` | causal_v | 5 steps (1.0 s) | the acceleration over 2.0 s at the short cycle |
| `R0V10` | causal_v | 10 steps (2.0 s) | the same at the programme's cycle |
| `R0VID` | causal_w, model seed 0 | 5 steps | control v1: must equal the dump of `R0W5` at model seed 0 of card ADD55 |

`R0V5` and `R0V10` at both model seeds: five new rollouts. Reused without a rerun: `R0` (2.0 s, pinned), `R0M5` (1.0 s, pinned), `R0W5`, `R0Z5`, `R0W10` of card ADD55.

## 4. Reads

As card ADD55, with the one readout WINDOW2:

* SP = speed error (longitudinal family, m/s) of `R0V5` minus that of `R0M5`; all family cells and U, C, F of the pair beside it.
* RS (per run) = share of the 2,399 windows whose last-step speed lies more than 5 m/s above the logged one, model rollout and prior alone.
* GC = `R0V5` against the banked `R0` (2.0 s, pinned).
* PR = `R0V10` minus `R0`.
* Prior alone: RS and ADE6 at 5 steps, with the episode-cluster bootstrap interval of card ADD54 for the ADE6 difference against the prior of `R0M5`.
* Beside, naming nothing: `R0V5` minus `R0W5`, `R0V5` minus `R0Z5`, `R0V10` minus `R0W10`.
* Four families for every pair; STRATEGIC as the last line of this card.

## 5. Names (fixed here; the rules are those of card ADD55)

* Model: `ACCWIN2S10K-MODEL-STABILISES-1S` if SP is below 0 at both model seeds, its pooled interval lies below 0, and RS of `R0V5` is at most 0.1768 and 0.1726 at the two model seeds. `ACCWIN2S10K-MODEL-SLOWS-CLIMB` if the SP condition holds and the RS condition does not. `ACCWIN2S10K-MODEL-NO-EFFECT` otherwise.
* Prior alone: `ACCWIN2S10K-PRIOR-REMOVES-CLIMB` if its RS at 5 steps is at most 0.3014 and its ADE6 at most 3.443 m; `-REDUCES-CLIMB` if that does not hold, its RS is below 0.4185 and the interval of its ADE6 difference against the pinned prior at 5 steps lies below 0; else `-CLIMB-STAYS`.
* The programme's cycle: `ACCWIN2S10K-COSTS-AT-2S`, `-HELPS-AT-2S`, `-NEUTRAL-AT-2S` by the rule of card ADD54 applied to PR.

What follows, fixed now:
* The evaluation convention of the programme and its pinned tool are not changed by this card; a change of the state rebuild is the other session's and the PI's decision.
* `MODEL-STABILISES-1S` with `NEUTRAL-AT-2S` or `HELPS-AT-2S`: an acceleration over 2.0 s of executed motion is the readout this study proposes for short cycles in the decision card for the PI and the other session; the next arm here is the three forms of card ADD52 at 1.0 s under it, then the 0.4 s cycle.
* `MODEL-STABILISES-1S` with `COSTS-AT-2S`: it is a candidate for short cycles only, and the decision card says so.
* `MODEL-SLOWS-CLIMB` or `MODEL-NO-EFFECT`: no estimate of the executed acceleration up to 2.0 s steadies the loop; on the readout side only removing the acceleration does, and the lever is the training one (the model's own rolled history at a short cycle), which is the other session's. The study of the rule at the short cycle then rests on the study readouts (card ADD59).
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* v1 (wrapper edit, known value): the dump of `R0VID` at model seed 0 equals the dump of `R0W5` at model seed 0 of card ADD55 array for array (`cmp_dumps.py`, 141 files, 0 differing). The queue step that makes this comparison uses a marker name no earlier card used.
* v2 (parameters applied): the readout log of every new run records its readout, its interval and a call count above 0; the result files record 5 steps for the 5-step runs and 10 for `R0V10`.
* v3 (the readout does what it says, from the readout log over all committed plans): in every new run the speed at the last executed knot equals the last executed step speed; in the WINDOW2 runs the acceleration at the tick equals the speed change of the rollout's own track over the window, recomputed by the reader from the dump's executed track (largest difference at most 1e-6), and the mean window length is recorded (it must be above 5 steps in `R0V5`: the window reaches across replans).
* v4 (the first block does not depend on the readout, known value): the first 5 executed steps of `R0V5` equal those of the banked `R0M5` at the same model seed (largest difference 0.0), for the model and for the prior; the first 10 of `R0V10` equal those of `R0`.
* v5 (counts): 141 dump files in each of the five new dumps, 2,399 windows and the same lead windows.
* v6 (reused dumps and known values): the result files of the reused runs have the sha256 recorded before the run (`integ10k/ACCWIN2S10K_REUSED.json`); the RS computation of the reader and the ADE6 of the prior give the values of section 1 of card ADD54 on `R0` and `R0M5`.
* v7: the controls of the headway reader and of the family reader on the new pairs.
* v8 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.
* v9 (the function, run before the first rollout as a tool check on the pod): a check script ends with its line True with the wrapper of the runs: on straight plans with known step speeds the WINDOW2 acceleration equals the hand-computed value at every executed step, across two replans and with a logged past shorter and longer than ten steps; the readouts pinned, backward, causal, causal_w and causal_z equal the wrapper of card ADD55 bit for bit; the read checks that the wrapper on the pod has the sha256 the check ran with.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

No lead rule in this card. Two model seeds; 141 episodes read again, no adjustment for the repeated reads. The readout is a study device that changes the speed column of the rebuilt state only; the heading at the last executed knot still comes from the central tangent. In the first second of a window the 2.0 s reach lies partly in logged motion, so the estimate there is partly a measured one. The names of the prior rest on one rollout and thresholds chosen after the pinned values were known. Nothing at 1.0 s is a capability number (coordinator decision D12).

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick does not depend on the readout.
