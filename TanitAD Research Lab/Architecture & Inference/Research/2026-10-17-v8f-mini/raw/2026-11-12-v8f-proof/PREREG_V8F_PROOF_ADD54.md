# PREREG_V8F_PROOF_ADD54 (HANDOVER10K): what the model reads as its own speed and acceleration at a replan, on the 10k checkpoints at the 1.0 s cycle; no training

Card of the v8F proof study, written by the coordinator fork while the rollouts of card ADD53 run and before its read; it does not use that read. Written before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Before the freeze only a tool check ran (section 1, last point; no model, no data of any card). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; no lead rule in this card); never driving performance.

## 1. What is known

* MEASURED, T1-kin (card ADD52, row H-V8F-PROOF-108): at a 1.0 s replanning cycle the two 10k models climb in speed in their own loop (speed error +11.01 m/s [+9.68, +12.30] against 2.0 s; share of windows whose last-step speed lies more than 5 m/s above the logged one 0.9075, 0.9825 against 0.0884, 0.0863); the kinematic prior alone climbs less (0.4185 against 0.3014 at 2.0 s; ADE6 5.407 m against 3.443 m; model 15.68, 18.76 m against 2.735, 2.691 m).
* INHERITED (other session, `raw/2026-10-22-t1/DESIGN_T1_V8F.md`, and its message of 2026-11-19): the instability of the prior at short replan periods is attributed to the prior's feedback on its own acceleration state, not to the state rebuild; its check for that executed the logged future through the same frames function (5,502 tick cells at 1 step: rebuilt against logged speed median 0.065 m/s, acceleration 0.055 m/s2, correlation 0.955). The other session confirms the tool fact below, holds that on human motion the central tangent is the true derivative, proposed the backward arm of this card for the prior alone and predicts that the runaway stays.
* Tool fact, MEASURED by a tool check with the pinned tool's own functions on straight plans (`integ10k/check_tick_readout.py`, output `integ10k/read_handover/CHECK_TICK_READOUT.txt`): the pinned tool fills the 10 Hz frames of the executed steps by a spline with central tangents, so the tangent at the last executed knot uses the plan's next knot, which is never executed. With step speeds s (0.2 s steps; s_m the last executed, s_m+1 the plan's next), the state the model reads at the tick is v0 = (s_m + s_m+1) / 2 and a0 = (0.625 (s_m+1 - s_m) - 0.125 (s_m - s_m-1)) / 0.1 s. Executed speed flat and the plan's next step +1.0 m/s: the tool reads v0 +0.5 m/s and a0 +6.25 m/s2 where the executed acceleration is 0; next step +0.35 m/s: a0 +2.19 m/s2. A last executed step of +1.0 m/s with a flat plan after it reads a0 -1.25 m/s2 where the executed acceleration is +5.0. On a smooth ramp the reading is the executed acceleration, which is why human motion reproduces. The model's plans are not smooth from step to step (mean speed of the first-tick plan at steps 1, 3, 5, 6, 8, 10: 11.62, 11.31, 11.72, 12.07, 11.86, 11.14 m/s), and a 5-step cycle hands over between steps 5 and 6.

## 2. Question

Does the climb at the 1.0 s cycle come from the acceleration and speed the tool hands to the model at a replan (the plan's own next step read back as measured state), or does it stay when the model reads only what was executed?

## 3. Arms (no rule, no limit, no training)

Models `UE_s0`, `UE_s1`, evaluation cache, window grid and pinned tool as cards ADD46 to ADD53. New wrapper `integ10k/v8f_t1_eval_reader_rule_h.py` (by `integ10k/make_handover10k.py` from the wrapper of card ADD53): one parameter, the readout of the rebuilt ego state after executed steps. The pinned file is not edited; the readout acts on every committed plan, so the rollout of the kinematic prior alone in the same dump has the same readout.

* `pinned`: the tool as it is (the frames function is passed through and counted).
* `backward`: the same frames function given the executed knots only, so the tangent at the last executed knot is the backward difference (the other session's proposal). On a straight path it reads v0 = s_m and a0 = -0.125 (s_m - s_m-1) / 0.1 s, the opposite sign of the executed acceleration at a quarter of its size.
* `causal`: positions and headings as pinned; the speed column from executed steps only: at a knot the step speed into it, at the half step the mean of that step and the step before. It reads v0 = s_m and a0 = (s_m - s_m-1) / 0.2 s, the executed values.

| run | readout | replan interval | role |
|---|---|---|---|
| `R0C5` | causal | 5 steps (1.0 s) | deciding arm |
| `R0B5` | backward | 5 steps | the other session's arm |
| `R0C10` | causal | 10 steps (2.0 s) | what the readout does at the programme's cycle |
| `R0HID` | pinned, model seed 0 | 10 steps | control h1: must equal the banked `R0` |

Each run except the control at both model seeds. Seven new rollouts, started after the queue of card ADD53 has ended. Reused without a rerun: `R0` (2.0 s) and `R0M5` (1.0 s), pinned readout.

## 4. Reads

Model: headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample).

* SP (per readout CAUSAL, BACKWARD) = speed error (longitudinal family, m/s) of the 5-step run with the readout minus that of `R0M5`; beside it all family cells and U, C, F of the same pair.
* RS (per run) = share of the 2,399 windows whose last-step speed lies more than 5 m/s above the logged one, for the model rollout and for the prior alone.
* GC (per readout) = the 5-step run with the readout against the banked `R0` (2.0 s, pinned): families and U, C, F.
* PR = `R0C10` minus `R0`: families and U, C, F.
* Prior alone (one rollout, it does not depend on the model seed): RS and ADE6 of the prior at 5 steps under each readout, with an episode-cluster bootstrap interval (141 episodes, 2,000 draws, fixed seed) for the ADE6 difference against the prior of `R0M5`.

## 5. Names (fixed here)

* Model, per readout: `HANDOVER10K-MODEL-<readout>-STABILISES-1S` if SP is below 0 at both model seeds, its pooled interval lies below 0, and RS of the 5-step run is at most twice the RS of the banked 2.0 s run at each model seed (bars 0.1768, 0.1726, as card ADD53). `HANDOVER10K-MODEL-<readout>-SLOWS-CLIMB` if the SP condition holds and the RS condition does not. `HANDOVER10K-MODEL-<readout>-NO-EFFECT` otherwise.
* Prior alone, per readout: `HANDOVER10K-PRIOR-<readout>-REMOVES-CLIMB` if its RS at 5 steps is at most 0.3014 and its ADE6 at most 3.443 m (the pinned values at 2.0 s); `HANDOVER10K-PRIOR-<readout>-REDUCES-CLIMB` if that does not hold, its RS is below 0.4185 and the interval of its ADE6 difference against the pinned prior at 5 steps lies below 0; else `HANDOVER10K-PRIOR-<readout>-CLIMB-STAYS`.
* The programme's cycle: `HANDOVER10K-CAUSAL-COSTS-AT-2S` if the pooled interval of PR on U, C or F lies above 0 or a family cell of that pair reads WORSE; `HANDOVER10K-CAUSAL-HELPS-AT-2S` if none of that holds and an interval of U, C or F lies below 0 or a family cell reads BETTER; else `HANDOVER10K-CAUSAL-NEUTRAL-AT-2S`.

What follows, fixed now:
* The evaluation convention of the programme and its pinned tool are not changed by this card. The pinned tool belongs to the other session; a change of its state rebuild is that session's and the PI's decision. The coordinator sends the read to both with a recommendation.
* MODEL-CAUSAL reads STABILISES-1S: the climb at the fast cycle is a term of the tick readout. Next arms here, on their own card: the three forms of card ADD52 at 1.0 s under the causal readout (the question of that card again), then the 0.4 s cycle.
* MODEL-CAUSAL reads SLOWS-CLIMB: both terms act. Next arm here: the causal readout together with the limit of card ADD53 at 1.0 s, if that card reads SLOWS-CLIMB or STABILISES for the no-rule form; else the same as the next point.
* MODEL-CAUSAL reads NO-EFFECT (the other session's prediction for the prior, carried to the model): the gain sits in how the plan answers its own state; no executor-side or readout-side lever is left on this line without training, and it goes to the other session as its training lever (the model's own rolled history at a short cycle), with the read of card ADD53 as the only executor-side result.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* h1 (wrapper edit, known value): the dump of `R0HID` at model seed 0 equals the banked dump of `R0` at model seed 0 array for array (`cmp_dumps.py`, 141 files, 0 differing).
* h2 (parameters applied): the readout log of every new run records its readout, its interval and a call count above 0; the result files record 5 steps for the 5-step runs and 10 for the others.
* h3 (the readout does what it says, from the readout log, over all committed plans of the run): causal: the speed at the last executed knot equals the last executed step speed and the one-frame acceleration equals the executed acceleration (largest differences at most 1e-6); backward: the speed at the last executed knot equals the last executed step speed (at most 1e-6); pinned: reported.
* h4 (the first block does not depend on the readout, known value): the first 5 executed steps of `R0C5` and of `R0B5` equal those of the banked `R0M5` at the same model seed (largest difference 0.0), for the model and for the prior; the first 10 of `R0C10` equal those of `R0`.
* h5 (counts): 141 dump files in each of the seven new dumps, 2,399 windows and the same lead windows.
* h6 (reused dumps and known values): the result files of the four reused runs have the sha256 recorded before the run (`integ10k/HANDOVER10K_REUSED.json`); the RS computation of the reader gives the values of section 1 on them; the ADE6 of the prior reads 3.443 and 5.407 m.
* h7: the controls of the headway reader and of the family reader on the new pairs.
* h8 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.
* h9 (the functions, run before the freeze as a tool check): `integ10k/check_tick_readout.py` ends with its formula line True on the pod with the wrapper of the runs; the read checks that the wrapper on the pod has the sha256 the check ran with.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

No lead rule in this card: the unsafe rates are those of the model alone. Two model seeds; 141 episodes read again, no adjustment for the repeated reads. The causal readout is a study device: it changes the speed column only (the heading at the last executed knot still comes from the central tangent, so the curvature feature still looks one step ahead); its speed is the mean over the last 0.2 s and its acceleration the mean over the last 0.4 s, a short lag against an instantaneous value; how it reproduces human motion (the other session's check for the pinned readout) is not measured here. The backward readout has a known sign error on the acceleration. The models were trained on logged state, where the acceleration is a measured one; neither readout is that. The bars of the prior's names are the pinned values at 2.0 s, chosen after they were known. The names of the prior rest on one rollout and thresholds, with an interval for the ADE6 difference only.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick does not depend on the readout.

## Amendment 1 (written before any run of the card; time in `TIMES_NOTE.md`; the hash of the card as first frozen is 929c9d7818102b90 in its first 16 characters, full value in the freeze line)

A file name only. The wrapper of this card was first written under the name `integ10k/v8f_t1_eval_reader_rule_h.py`, which is the name of the wrapper of card ADD49 (form RRh). The chain of this card stopped at its push step because the pod holds that older file (no queue was started, no rollout ran), and the local copy of the older file, written over by the generator, is put back from the repository (sha256 db05a707e6cacf7c in its first 16 characters, equal to the landed file). The wrapper of this card is `integ10k/v8f_t1_eval_reader_rule_ro.py`, its pod queue `integ10k/pod_handover10k_queue2.sh`; the generator now refuses to write over a file it did not write. The code of the wrapper is unchanged apart from its own name in the header; the tool check of control h9 was run again with the renamed wrapper (formula line True). Arms, readouts, reads, names and controls stand as frozen.
