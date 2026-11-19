# PREREG_V8F_PROOF_ADD53 (FEAS10K): the 1.0 s replanning cycle with an acceleration limit on the executed plan, on the 10k checkpoints; no training

Card of the v8F proof study, written by the coordinator fork after the reads of cards ADD52 and ADD51 and the descriptive floor read (section 1). Written before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Before the freeze only tool checks were run: the unit test of the new function (section 6, f10) and one check on banked files (section 6, r1). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known

* MEASURED, T1-kin (card ADD52, `integ10k/read_cycle/pod/READ_CYCLE10K.json`): at a 1.0 s replanning cycle (5 cache steps) instead of 2.0 s the unsafe rate rises for every form: no rule +0.7438 [+0.6709, +0.8132], label-fed rule +0.4484 [+0.3789, +0.5189], reader-fed rule +0.4956 [+0.4216, +0.5718]; speed error of the model alone +11.01 m/s [+9.68, +12.30]. The rule still cuts the unsafe rate at 1.0 s (label -0.3870, reader -0.2980).
* MEASURED, descriptive (`integ10k/read_cycle/CYCLE_PROFILE.txt`, `integ10k/read_cycle/FLOOR_VS_MODEL.txt`; made after the read of card ADD52, name nothing): at 1.0 s the mean executed speed climbs at every replan (11.6 to 41 m/s over 6 s against a logged 11.5 to 11.3 m/s); the share of windows whose last-step speed lies more than 5 m/s above the logged one is 0.9075 and 0.9825 for the model without a rule (0.0884 and 0.0863 at 2.0 s), 0.7078 and 0.7874 under the label-fed rule (0.0767, 0.0721), 0.6294 and 0.6886 under the reader-fed rule (0.0642, 0.0621). The kinematic prior alone reads 0.4185 at 1.0 s and 0.3014 at 2.0 s; its ADE6 3.443 m and 5.407 m equal the other session's table. The model is the better of the two at 2.0 s (ADE6 2.735, 2.691 m against 3.443) and the worse at 1.0 s (15.68, 18.76 m against 5.407).
* INHERITED (other session, `raw/2026-10-22-t1/DESIGN_T1_V8F.md` sections 8 and 9): the kinematic prior in its own loop is not stable at short replan periods; the operating points of the programme are 30 and 10 steps, 5 steps is a stress point; a feasibility-limited executor is listed there as an experiment to run if the model is also unstable, and a vehicle model in the evaluation convention is a design decision.
* Tool fact (pinned tool, read from its source): the executed track is a spline through the plan's knots with central tangents, so the ego speed measured at a tick is the mean of the last executed step and the plan's next step, which is never executed. On the banked run `RLcM5` (model seed 0) that speed differs from the last executed step speed by a median of 0.13, 0.33, 0.62 m/s at ticks 1, 2, 3 (99th percentile 2.4, 3.3, 6.1 m/s).

## 2. Question

Is the climb at the 1.0 s cycle a product of executing plans with no physical limit (every new plan may start with a speed step that no car can follow), or does the loop of model and state rebuild climb under a plausible acceleration limit as well?

## 3. Arms (no training, no new reader, no new map)

Models (`UE_s0`, `UE_s1`), reader arms, maps (level 0.50), presence threshold 0.5, floor form cap-only, rule parameters (D 3 m, time gap 1.2 s, braking limit 6 m/s2), evaluation cache, window grid and pinned tool as cards ADD46 to ADD52. New wrapper `integ10k/v8f_t1_eval_reader_rule_a.py` (by `integ10k/make_feas10k.py` from the wrapper of card ADD51): two parameters, an acceleration limit and a deceleration limit. Every plan of the model is retimed along its own path before anything else: from the ego speed measured at the tick, the speed along the path may rise by at most the acceleration limit times 0.2 s per step and fall by at most the deceleration limit times 0.2 s per step; a step that is not limited keeps the plan's own advance; the path is unchanged. A lead rule, if on, acts on the limited plan. The rollout of the kinematic prior alone (the floor arrays of the dump) is limited the same way. It is a study arm and no vehicle model: no jerk limit, no lateral limit, no speed-dependent limit.

Limits, fixed here before any run: acceleration 3.0 m/s2 (a strong acceleration of a passenger car at urban speed), deceleration 8.0 m/s2 (above the 6 m/s2 of the lead rule, so the rule's braking is not cut). Dose arm: acceleration 1.5 m/s2.

| run | form | replan interval | limits (m/s2) | role |
|---|---|---|---|---|
| `R0A5` | no rule | 5 steps (1.0 s) | 3.0 / 8.0 | deciding arm: the model alone |
| `RLcA5` | labels: presence, gap, closing speed | 5 steps | 3.0 / 8.0 | the rule at label quality |
| `RQ50A5` | reader presence and gap at level 0.50, closing 0 | 5 steps | 3.0 / 8.0 | the deployable form |
| `R0A10` | no rule | 10 steps (2.0 s) | 3.0 / 8.0 | what the limit does at the programme's cycle |
| `R0A5L` | no rule | 5 steps | 1.5 / 8.0 | dose arm, names nothing |
| `R0AID` | no rule, model seed 0 | 10 steps | out of reach (1e9) | control f1: must equal the banked `R0` |

Each run except the control at both model seeds (reader seed = model seed). Eleven new rollouts. Reused without a rerun: `R0`, `RLc`, `RQ50` (2.0 s, no limit) and `R0M5`, `RLcM5`, `RQ50M5` (1.0 s, no limit).

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample). U on the 562 scored windows per seed, C and F on the 642 lead windows, the four families on all 2,399 windows.

* SP (per form NORULE, LABEL, READER) = speed error (longitudinal family, m/s) of the A5 run minus that of the M5 run of the same form: the limit against no limit at 1.0 s. Beside it U, C, F and all family cells of the same pair.
* RS (per run) = share of the 2,399 windows whose last-step speed lies more than 5 m/s above the logged one (the quantity of section 1), for the model rollout and for the prior alone.
* GC (per form) = U of the A5 run minus U of the banked 2.0 s run of the same form, with C, F and the families beside it.
* PR = `R0A10` minus `R0`: U, C, F and the families (the limit at the programme's cycle).
* RU (per rule form) = U of the rule form minus U of no rule, both at 1.0 s with the limit.
* Dose, beside: `R0A5L` against `R0A5` and against `R0M5` (speed error, RS, U).

## 5. Names (fixed here)

* Per form: `FEAS10K-<form>-LIMIT-STABILISES-1S` if SP is below 0 at both model seeds, its pooled interval lies below 0, and RS of the A5 run is at most twice the RS of the banked 2.0 s run of the same form at each model seed (bars: no rule 0.1768, 0.1726; label rule 0.1534, 0.1442; reader rule 0.1284, 0.1242). `FEAS10K-<form>-LIMIT-SLOWS-CLIMB` if the SP condition holds and the RS condition does not. `FEAS10K-<form>-LIMIT-NO-EFFECT` otherwise.
* Per form: `FEAS10K-<form>-U-ABOVE-2S` if the pooled interval of GC lies above 0; `FEAS10K-<form>-U-BELOW-2S` if it lies below 0; else `FEAS10K-<form>-U-NOT-SEPARATED-FROM-2S` (not separated is no equivalence; for the READER form a pooled value of at most 0.0196 in size is said to be inside the reader floor of row 106).
* The programme's cycle: `FEAS10K-LIMIT-COSTS-AT-2S` if the pooled interval of PR on U, C or F lies above 0 or a family cell of that pair reads WORSE; `FEAS10K-LIMIT-HELPS-AT-2S` if none of that holds and an interval of U, C or F lies below 0 or a family cell reads BETTER; else `FEAS10K-LIMIT-NEUTRAL-AT-2S`.
* Per rule form: `FEAS10K-<form>-RULE-CUTS-UNSAFE-AT-1S-LIMITED` if RU is below 0 at both model seeds and its pooled interval excludes 0; else `FEAS10K-<form>-RULE-NOT-SEPARATED-AT-1S-LIMITED`.

What follows, fixed now:
* The evaluation convention of the programme (2.0 s, no limit) is not changed by this card. Whether an executor with a physical limit becomes part of any evaluation protocol is a decision of the PI; the coordinator puts the read to the PI and the other session with a recommendation.
* NORULE reads LIMIT-STABILISES-1S: the climb is a term of the unlimited executor. Next arm here: the 0.4 s cycle (2 steps) with the same limits on the three forms.
* NORULE reads LIMIT-SLOWS-CLIMB or LIMIT-NO-EFFECT: the loop of model and state rebuild climbs under a plausible limit too. Reported as written and passed to the other session as a finding on the model in its own loop (the lever on that side is training with the model's own rolled history at a short cycle). Next arm here, without training and on its own card: the handover of the plan (the speed measured at the tick taken from the last executed step alone, a state-rebuild arm), to separate the anticipating tangent of section 1 from the model's reaction.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* f1 (wrapper edit, known value): the dump of `R0AID` at model seed 0 equals the banked dump of `R0` at model seed 0 array for array (`cmp_dumps.py`, 141 files, 0 differing); the floor arrays are among them.
* f2 (parameters applied): the limit log of every new run records its two limits and its interval; the result files record 5 steps for the A5 and A5L runs and 10 for the A10 runs and the control.
* f3 (the limit holds, exact): in every limit log, for the model and for the prior, the speed along the path of every executed step lies within [max(0, v - 8.0 x 0.2), v + a x 0.2] of the step before it (the first step: of the ego speed measured at the tick), tolerance 1e-6 m/s, on every row with a finite ego speed.
* f4 (rows are windows): each limit log holds 2,399 rows per tick for the model and for the prior, and its first-tick ego speeds equal the ego speeds of the dump in file order within 1e-4 m/s.
* f5 (the executed track obeys the limit): inside each executed block of the model rollout of every A run, the share of steps whose chord acceleration lies above the acceleration limit + 0.25 m/s2 or below -(8.0 + 0.25) m/s2 is at most 0.01 (on the banked first-tick plans the function leaves 0.0004 and 0.0006: paths with a kink, where the chord is shorter than the path).
* f6 (counts): 141 dump files in each of the eleven new dumps, 2,399 windows and the same lead windows.
* f7 (reused dumps): the result files of the twelve reused runs have the sha256 recorded before the run (`integ10k/FEAS10K_REUSED.json`).
* f8: the controls of the headway reader and of the family reader on the new pairs.
* f9 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.
* f10 (the function, run before the freeze as a tool check): `integ10k/test_accel_limit.py` passes on the pod with the wrapper of the runs (18 of 18; step-by-step limits against a written-out recursion, the path kept, untouched rows bit-identical, out-of-reach limits return the input object, planted wrong limits and a wrong start speed are detected; on the 2,399 banked first-tick plans the speeds along the path obey the limits to 1e-6). The read checks that the wrapper on the pod has the sha256 the test ran with.
* r1 (reported, no bar): per A run the absolute difference between the ego speed measured at a tick and the last executed speed of the tick before (median, 99th percentile). The limit starts from the measured speed, so across a tick the executed speed may step by more than the limit; r1 says by how much.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

56 episodes with a lead, read again (cards ADD46 to ADD52 before); no adjustment for the repeated reads is applied. A lead that does not react; two model seeds. The limit is a study device with two constants chosen by plausibility, not fitted and not a vehicle model; it acts on the speed along the path only. The limits were fixed after the runaway of card ADD52 was seen and before any limited run. RS has a bar made from the banked 2.0 s runs (twice their value), chosen after those values were known. The models were trained and selected under the 2.0 s convention. With the limit the first plan of a window is changed in 0.26 of the windows (first-tick plans of the banked no-rule run, model seed 0), so the limited runs at 2.0 s are not the banked runs.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick's decision heads do not depend on the cycle or on the limit.

## Amendment 1 (written while the rollouts of the card run and before any of their results is pulled or opened; time in `TIMES_NOTE.md`; the hash of the card as first frozen is 79999907c1005a828d3739f430c36f60de37cb6b50e1b560f5913a5eeb1f5860)

Found while the reader was written: control f5 as first written covers every A run, the two rule forms included. In a rule form the lead rule acts after the limit, and when its cap lets go inside an executed block the speed returns to the plan's own speed in one step, so the executed track of a rule form is not bound by the acceleration limit by design. Control f5 therefore applies to the six no-rule A runs (`R0A5`, `R0A10`, `R0A5L` at both model seeds), with the bar unchanged; for the four rule-form A runs the same share is reported without a bar (r2). Control f3 (the limit holds in the log of the limiter, before the rule) covers every run as written. Nothing else changes: arms, limits, reads, names and all other controls stand. One addition to control f7, a known-value check of the RS computation of the reader: on the twelve reused runs it must give the values of section 1 (within 0.00005).
