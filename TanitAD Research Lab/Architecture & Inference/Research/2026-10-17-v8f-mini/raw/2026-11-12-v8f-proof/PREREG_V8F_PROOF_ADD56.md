# PREREG_V8F_PROOF_ADD56 (FEAS10K, the repeat): the acceleration limit of card ADD53 held on the executed knots; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD53 stopped at its controls, and before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Cards ADD54 and ADD55 were not read when this card was frozen. Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.

## 1. Why this card exists

* Card ADD53 was read on the pod (`integ10k/read_feas/pod/READ_FEAS10K.json`). Its control f5 did not pass: the share of in-block steps of the executed track whose chord acceleration lies beyond the limit by more than 0.25 m/s2 is 0.0170 and 0.0140 in the two runs `R0A10` (bar 0.01); `R0A5` reads 0.0099 and 0.0095, `R0A5L` 0.0075 and 0.0070. All other controls passed. By the rule of card ADD53 nothing is named, and the reader printed no name, no level and no difference. The pair results of those runs are banked and were not opened.
* The same rule of card ADD53: "the tool is repaired and the runs are repeated under new run names". This card is that repeat.
* Diagnosis, from instrument quantities only (`integ10k/read_feas/F5_DIAG.txt`, tool `integ10k/f5_diag.py`; no error against the log, no rate): 77 to 90 percent of the steps beyond the limit sit where the direction of the executed track changes by more than 0.2 rad from one step to the next. The limiter of card ADD53 bounds the speed along the plan's own path, and it does so exactly (control f3: largest excess 2e-14 m/s). On a bent path the chord between two knots is shorter than the arc between them. The executed track and the state rebuild of the pinned tool see the knots, so they see chords. Control f5 measured chords; the limiter bounded arcs.

## 2. The repair

Wrapper `integ10k/v8f_t1_eval_reader_rule_ac.py` (by `integ10k/make_feasc10k.py` from the wrapper of card ADD53; sha256 23a40394f8e63fc0...): one more parameter, `--alim-form path|chord`. `path` is the function of card ADD53, unchanged. `chord`: step i of the limited plan lies on the plan's own path, and its chord length from the knot before is kept in [max(0, v(i-1) - b dt) dt, (v(i-1) + a dt) dt], with v(i) the chord speed of step i and v(0) the ego speed at the tick (the speed feature the model reads). A step that needs no clip keeps the plan's own arc advance; a step that is not moved keeps its knot bit for bit. Limits out of reach return the plan object itself. As in card ADD53 the limit acts on the plan of the model and on the plan of the kinematic prior alone, and the lead rule, where a form has one, acts on the limited plan.

Tool check before the freeze (no run of this card, no data of card ADD53; path material = executed tracks of the banked no-rule run at 5 steps of card ADD52): `integ10k/test_accel_limit_chord.py` on the pod, 24 of 24 checks pass (`integ10k/read_feas/TEST_ACCEL_LIMIT_CHORD.txt`). On 800 bent tracks the chord form holds the limit on every chord speed to 1e-14 m/s, and the f5 quantity of the reader reads 0.0000 on its output and 0.047 to 0.054 on the output of the path form.

## 3. Arms

The arms of section 3 of card ADD53, with the form `chord` and new run names (E in place of A):

| run | form of the rule | replan interval | limits a / b (m/s2) | role |
|---|---|---|---|---|
| `R0E5`, `RLcE5`, `RQ50E5` | none / label-fed with closing speed / reader-fed at level 0.50 | 5 steps (1.0 s) | 3.0 / 8.0 | the limit at the 1.0 s cycle, per form |
| `R0E10` | none | 10 steps (2.0 s) | 3.0 / 8.0 | the limit at the programme's cycle |
| `R0E5L` | none | 5 steps | 1.5 / 8.0 | dose; names nothing |
| `R0EID` | none, model seed 0 | 10 steps | 1e9 / 1e9 | control f1: must equal the banked `R0` |

Each run except the control at both model seeds (`UE_s0`, `UE_s1`): eleven rollouts, started after the queue of card ADD55 has ended. Models, evaluation cache, window grid and pinned tool as cards ADD46 to ADD55. Reused without a rerun: the banked 2.0 s runs and the 5-step runs without a limit of card ADD52 (the twelve files of `integ10k/FEAS10K_REUSED.json`).

## 4. Reads, names, bars, what follows

Sections 4 and 5 of card ADD53 with its Amendment 1 govern, word for word, with the E runs in place of the A runs: the reads SP, RS, GC, PR, RU; the names `FEAS10K-<form>-LIMIT-STABILISES-1S`, `-SLOWS-CLIMB`, `-NO-EFFECT`; `FEAS10K-<form>-U-ABOVE-2S`, `-U-BELOW-2S`, `-U-NOT-SEPARATED-FROM-2S`; `FEAS10K-LIMIT-COSTS-AT-2S`, `-HELPS-AT-2S`, `-NEUTRAL-AT-2S`; `FEAS10K-<form>-RULE-CUTS-UNSAFE-AT-1S-LIMITED`, `-RULE-NOT-SEPARATED-AT-1S-LIMITED`; the RS bars (no rule 0.1768, 0.1726; label 0.1534, 0.1442; reader 0.1284, 0.1242); and the follow-ups fixed there. No bar and no name is changed by this card. Reader `integ10k/read_feasc10k.py` (by the same generator from the reader of card ADD53).

Beside, fixed now: after this card is read, the pair results of the path-form runs of card ADD53 are printed next to those of this card by a small tool, without a name: how much the form of the limit moves each read. They name nothing, whatever this card reads.

## 5. Controls (all must pass; else nothing is named)

f1 to f10 of card ADD53 with its Amendment 1, on the E runs, with these differences:

* f2 also asks that every limit log records the form `chord`.
* f3 is read on the chord speeds in the limit log (the quantity the limit now acts on): no executed step leaves the limit by more than 1e-6 m/s.
* f5 is unchanged: inside each executed block of the model rollout of the six no-rule E runs, the share of steps whose chord acceleration lies above a + 0.25 or below -(8.0 + 0.25) m/s2 is at most 0.01. The rule forms are reported without a bar (r2).
* f10: the test of section 2 ends with its PASS line on the pod, and the wrapper of the runs has the sha256 the test ran with.
* Reported without a bar, r3: the number of executed steps whose lower bound could not be reached on the path (a path that turns back on itself).

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

A control fails again: nothing is named, the read is reported as written, and there is no third form on this card; whether a limit on the executed plan steadies the 1.0 s cycle then stays open and is carried beside the reads of cards ADD54 and ADD55 with its instrument diagnosis.

## 6. Limits

The limits of card ADD53 hold. In addition: this card was written after the controls and the reported quantities r1 and r2 of card ADD53 were seen (not its names, levels or differences), so the form of the limit was chosen knowing how the first form failed its control. The chord form is a study device, no vehicle model: it bounds the speed change between knots 0.2 s apart and says nothing about lateral acceleration, which on the bent tracks of the 1.0 s cycle is far outside what a car can do. Where the path turns back on itself the lower bound may not be reachable (r3). The evaluation convention of the programme and its pinned tool are not changed; adopting any limit in a protocol is the PI's and the other session's decision. 141 episodes are read again, with no adjustment for the repeated reads.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick's plan is changed by the limit only where the plan asks for more than the limit.
