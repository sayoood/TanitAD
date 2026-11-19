# PREREG_V8F_PROOF_ADD55 (ACCWIN10K): the acceleration the model reads from its own executed motion at the 1.0 s cycle, over a longer window or held at zero; no training

Card of the v8F proof study, written by the coordinator fork while the rollouts of cards ADD53 and ADD54 run and before either read; it uses neither. Written before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Before the freeze only a tool check ran (section 6, z9; no model, no data of any card). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; no lead rule in this card); never driving performance.

## 1. What is known

* MEASURED, T1-kin (card ADD52, row H-V8F-PROOF-108): at a 1.0 s replanning cycle the two 10k models climb in speed in their own loop; the kinematic prior alone climbs less; values in section 1 of card ADD54.
* Tool fact, MEASURED (card ADD54, section 1; `integ10k/read_handover/CHECK_TICK_READOUT.txt`): at a replan the pinned tool hands the model a speed and an acceleration that contain the plan's next, never executed step; with a flat executed speed and a next step of +1.0 m/s the acceleration read is +6.25 m/s2.
* HYPOTHESIS of the other session (its design note of 2026-10-22 and its message of 2026-11-19): the loop gain sits in how the plan answers its own acceleration state (the prior's acceleration and speed-times-acceleration features), not in the state rebuild.
* Card ADD54 (queued before this card, not read): the readouts `backward` and `causal`. The causal readout hands the model the executed step speed and the executed acceleration of the last step.

## 2. Question

With the speed read fixed at the executed step speed, does the climb at the 1.0 s cycle depend on the acceleration the model reads from its own executed motion? Two acceleration reads are compared with the one-step acceleration of the causal readout of card ADD54: the executed speed change over the last 1.0 s, and none at all.

## 3. Arms (no rule, no limit, no training)

Models `UE_s0`, `UE_s1`, evaluation cache, window grid and pinned tool as cards ADD46 to ADD54. Wrapper `integ10k/v8f_t1_eval_reader_rule_rz.py` (by `integ10k/make_accwin10k.py` from the wrapper of card ADD54): two more readouts. Both keep the positions and headings of the pinned frames and the causal speed at every executed knot (the step speed into it); they set the one-frame acceleration at every executed step i of a block:

* `causal_w` (WINDOW): (s_i - s_{i-w}) / (0.2 s x w) with w = min(i, 5): the executed speed change over the last 1.0 s (s_0 = the step into the block).
* `causal_z` (ZERO): 0. The model reads its executed speed and no acceleration from its own rollout. A diagnosis arm: it removes information the model was trained to use.

The readouts act on every committed plan, so the rollout of the kinematic prior alone in the same dump has the same readout.

| run | readout | replan interval | role |
|---|---|---|---|
| `R0W5` | causal_w | 5 steps (1.0 s) | the acceleration over 1.0 s |
| `R0Z5` | causal_z | 5 steps | no acceleration read |
| `R0W10` | causal_w | 10 steps (2.0 s) | what the window does at the programme's cycle |
| `R0ZID` | causal, model seed 0 | 5 steps | control z1: must equal the dump of `R0C5` of card ADD54 |

Each run except the control at both model seeds. Seven new rollouts, started after the queue of card ADD54 has ended. Reused without a rerun: `R0` (2.0 s) and `R0M5` (1.0 s), pinned readout; `R0C5` of card ADD54 for the control and for one pair beside.

## 4. Reads

As card ADD54, with the readouts WINDOW and ZERO in place of CAUSAL and BACKWARD:

* SP (per readout) = speed error (longitudinal family, m/s) of the 5-step run with the readout minus that of `R0M5`; all family cells and U, C, F of the pair beside it.
* RS (per run) = share of the 2,399 windows whose last-step speed lies more than 5 m/s above the logged one, model rollout and prior alone.
* GC (per readout) = the 5-step run against the banked `R0` (2.0 s, pinned).
* PR = `R0W10` minus `R0`.
* Prior alone: RS and ADE6 at 5 steps under each readout, with the episode-cluster bootstrap interval of card ADD54 for the ADE6 difference against the prior of `R0M5`.
* Beside: each readout against the causal readout of card ADD54 at 1.0 s (`R0W5` minus `R0C5`, `R0Z5` minus `R0C5`), and WINDOW minus ZERO.

## 5. Names (fixed here)

* Model, per readout: `ACCWIN10K-MODEL-<readout>-STABILISES-1S` if SP is below 0 at both model seeds, its pooled interval lies below 0, and RS of the 5-step run is at most twice the RS of the banked 2.0 s run at each model seed (bars 0.1768, 0.1726). `ACCWIN10K-MODEL-<readout>-SLOWS-CLIMB` if the SP condition holds and the RS condition does not. `ACCWIN10K-MODEL-<readout>-NO-EFFECT` otherwise.
* Prior alone, per readout: `ACCWIN10K-PRIOR-<readout>-REMOVES-CLIMB` if its RS at 5 steps is at most 0.3014 and its ADE6 at most 3.443 m; `ACCWIN10K-PRIOR-<readout>-REDUCES-CLIMB` if that does not hold, its RS is below 0.4185 and the interval of its ADE6 difference against the pinned prior at 5 steps lies below 0; else `ACCWIN10K-PRIOR-<readout>-CLIMB-STAYS`.
* The programme's cycle: `ACCWIN10K-WINDOW-COSTS-AT-2S`, `-HELPS-AT-2S`, `-NEUTRAL-AT-2S` by the rule of card ADD54 applied to PR.

What follows, fixed now:
* The evaluation convention of the programme and its pinned tool are not changed by this card; a change of the state rebuild is the other session's and the PI's decision. The coordinator sends the read to both with a recommendation.
* MODEL-ZERO reads STABILISES-1S and the causal readout of card ADD54 does not: the acceleration the model reads from its own rollout carries the loop gain; the other session's hypothesis is then measured on the model. If MODEL-WINDOW also reads STABILISES-1S, an acceleration over 1.0 s of executed motion is a candidate readout for short cycles; next arms here on their own card: the three forms of card ADD52 at 1.0 s under that readout, then the 0.4 s cycle.
* MODEL-ZERO reads STABILISES-1S and MODEL-WINDOW does not: the next arm here is the window length (2.0 s) on its own card, and the training lever (the model's own rolled history at a short cycle) goes to the other session with this read.
* MODEL-ZERO reads SLOWS-CLIMB or NO-EFFECT: the climb does not hang on the acceleration read; what is left on the readout side is the speed and the curvature state; reported as written, and the profile of the plan itself (its step-to-step speed changes) goes to the other session as the model-side term.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* z1 (wrapper edit, known value): the dump of `R0ZID` at model seed 0 equals the dump of `R0C5` at model seed 0 of card ADD54 array for array (`cmp_dumps.py`, 141 files, 0 differing).
* z2 (parameters applied): the readout log of every new run records its readout, its interval and a call count above 0; the result files record 5 steps for the 5-step runs and 10 for `R0W10`.
* z3 (the readout does what it says, from the readout log over all committed plans): in every new run the speed at the last executed knot equals the last executed step speed; WINDOW: the acceleration at the tick equals the executed speed change over the window; ZERO: the acceleration at the tick is 0; control run: the acceleration equals the executed one-step acceleration (largest differences at most 1e-6).
* z4 (the first block does not depend on the readout, known value): the first 5 executed steps of `R0W5` and `R0Z5` equal those of the banked `R0M5` at the same model seed (largest difference 0.0), for the model and for the prior; the first 10 of `R0W10` equal those of `R0`.
* z5 (counts): 141 dump files in each of the seven new dumps, 2,399 windows and the same lead windows.
* z6 (reused dumps and known values): the result files of `R0` and `R0M5` have the sha256 recorded before the run (`integ10k/HANDOVER10K_REUSED.json`, the file of card ADD54); the RS computation of the reader and the ADE6 of the prior give the values of section 1 of card ADD54 on them.
* z7: the controls of the headway reader and of the family reader on the new pairs.
* z8 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.
* z9 (the functions, run before the freeze as a tool check): `integ10k/check_tick_readout_rz.py` ends with its line True on the pod with the wrapper of the runs (both readouts at every executed step on five straight plans; pinned, backward and causal equal the wrapper of card ADD54 bit for bit); the read checks that the wrapper on the pod has the sha256 the check ran with.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

No lead rule in this card. Two model seeds; 141 episodes read again, no adjustment for the repeated reads. Both readouts are study devices that change the speed column of the rebuilt state only; the heading at the last executed knot still comes from the central tangent, so the curvature feature still looks one step ahead. ZERO is not a protocol: on logged state the acceleration is a measured quantity the model was trained with. The window of WINDOW reaches at most 1.0 s back and is shorter in the first steps of a block. How either readout reproduces human motion is not measured here. The names of the prior rest on one rollout and thresholds chosen after the pinned values were known, with an interval for the ADE6 difference only. The card was written after the unsafe rates and the speed profile of card ADD52 were known, and before cards ADD53 and ADD54 were read.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick does not depend on the readout.
