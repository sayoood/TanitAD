# Card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K): the causal readout together with the acceleration limit at the 1.0 s cycle; no training, no lead rule

Card of the v8F proof study, written by the coordinator fork after the reads of cards ADD54, ADD55 and ADD56, and before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Cards ADD57, ADD58 and ADD59 were not read when this card was frozen. Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; no lead rule in this card); never driving performance.

## 1. Why this card exists

* Card ADD54 fixed it before its data: "MODEL-CAUSAL reads SLOWS-CLIMB: both terms act. Next arm here: the causal readout together with the limit of card ADD53 at 1.0 s, if that card reads SLOWS-CLIMB or STABILISES for the no-rule form". Row 110: the causal readout reads `HANDOVER10K-MODEL-CAUSAL-SLOWS-CLIMB`. Row 112 (card ADD56, the repeat of card ADD53 with the limit held on the executed knots): `FEAS10K-NORULE-LIMIT-SLOWS-CLIMB`. Both conditions hold, so this arm runs.
* What it decides. The backward readout (row 110) and the readout with the acceleration at 0 (row 111) steady the 1.0 s cycle, and both are study devices: they hand the model an acceleration that is not the executed one. The causal readout and the chord limit are the two parts without such a device: the executed step speed, the executed one-step acceleration, and a bound on the speed change a plan may ask for. Each alone only slows the climb. This card reads the pair.
* With the causal readout the speed at a tick is the last executed step speed, and the chord limit bounds the first chord of the new plan against that speed. The pair is therefore one bound on the executed speed change across replans as well as inside a block; under the pinned readout it was a bound inside the block only.

## 2. Arms (no rule, no training)

Models `UE_s0`, `UE_s1`, evaluation cache, window grid and pinned tool as cards ADD46 to ADD59. New study wrapper `integ10k/v8f_t1_eval_reader_rule_rzc.py`: the wrapper of card ADD55 (`_rz`, readouts) with the chord form of the limit of card ADD56 added by the same asserted text edits (`--alim-form path|chord`); the pinned file is not edited. Readout and limit act on every committed plan, so the rollout of the kinematic prior alone in the same dump has both.

| run | readout | replan interval | limits a / b (m/s2), form | role |
|---|---|---|---|---|
| `R0CE5` | causal | 5 steps (1.0 s) | 3.0 / 8.0, chord | the pair at the 1.0 s cycle |
| `R0CE10` | causal | 10 steps (2.0 s) | 3.0 / 8.0, chord | the pair at the programme's cycle |
| `R0CEIDc` | causal, model seed 0 | 5 steps | 1e9 / 1e9, chord | control c1: must equal the banked `R0C5` of card ADD54 |
| `R0CEIDe` | pinned, model seed 0 | 5 steps | 3.0 / 8.0, chord | control c2: must equal the banked `R0E5` of card ADD56 |

`R0CE5` and `R0CE10` at both model seeds: six rollouts, started after the queue of card ADD58 has ended. Reused without a rerun: `R0` (2.0 s) and `R0M5` (1.0 s) with the pinned readout and no limit, `R0C5` (causal alone, card ADD54), `R0E5` and `R0E10` (limit alone, card ADD56).

## 3. Reads

Paired episode-cluster bootstrap over the 141 episodes, pooled over the two model seeds, per-seed values beside; 562 scored windows and 642 lead windows per seed; B as cards ADD53 to ADD56. Headway shares U (unsafe rate), C (contact), F (falling behind) and the family cells of the pinned reader for every pair.

* SP = `R0CE5` minus `R0M5` (the pair against the pinned 1.0 s run): speed error first.
* RS = share of the 2,399 windows whose last-step speed lies more than 5 m/s above the logged one, per model seed (bars 0.1768, 0.1726 = twice the banked 2.0 s value, as cards ADD53 to ADD56).
* PARTS = `R0CE5` minus `R0C5` (what the limit adds to the causal readout) and `R0CE5` minus `R0E5` (what the causal readout adds to the limit).
* GC = `R0CE5` minus the banked `R0` (2.0 s, pinned, no limit).
* PR = `R0CE10` minus `R0`; beside it `R0CE10` minus `R0E10`.
* Kinematic prior alone in the same dumps: RS and ADE6 with the episode-cluster interval of the ADE6 difference, reported.

## 4. Names (fixed before the data)

* `CAUSLIM10K-MODEL-STABILISES-1S` if SP on the speed error is below 0 at both model seeds, its pooled interval lies below 0, and RS of `R0CE5` is at most the bar at each model seed. `CAUSLIM10K-MODEL-SLOWS-CLIMB` if the SP condition holds and the RS condition does not. `CAUSLIM10K-MODEL-NO-EFFECT` otherwise.
* `CAUSLIM10K-BELOW-BOTH-PARTS` if in both PARTS pairs U is below 0 at both model seeds with the pooled interval below 0; else `CAUSLIM10K-NOT-BELOW-BOTH-PARTS`.
* `CAUSLIM10K-U-ABOVE-2S`, `-U-BELOW-2S`, `-U-NOT-SEPARATED-FROM-2S`: GC on U, the same sign at both model seeds and the pooled interval on that side of 0; else not separated.
* `CAUSLIM10K-COSTS-AT-2S` if the pooled interval of PR on U, C or F lies above 0 or a family cell of that pair reads WORSE; `CAUSLIM10K-HELPS-AT-2S` if none of that holds and an interval of U, C or F lies below 0 or a family cell reads BETTER; else `CAUSLIM10K-NEUTRAL-AT-2S` (the rule of card ADD53).

LONGITUDINAL, LATERAL and TACTICAL: the family cells of the pinned reader for every pair, per family, never pooled into one score. STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument).

## 5. Controls (all must pass; else nothing is named)

* e1: the pod read reproduces the banked dev-box read of card ADD46 leaf for leaf.
* c1: `R0CEIDc` (model seed 0) equals the banked `R0C5` of the same seed, 141 files, 0 differing. c2: `R0CEIDe` equals the banked `R0E5` of the same seed, 141 files, 0 differing. Together they say that the new wrapper is the wrapper of card ADD55 on the readout side and the wrapper of card ADD56 on the limit side.
* c3: every run records its readout, its interval, its limits and the form `chord` in its logs; the committed plan counts are those of the interval.
* c4: no chord speed in a limit log leaves the limit by more than 1e-6 m/s.
* c5: inside the executed blocks of `R0CE5` and `R0CE10` the share of steps whose chord acceleration lies above a + 0.25 or below -(8.0 + 0.25) m/s2 is at most 0.01 (control f5 of card ADD56).
* c6: in every run with the causal readout the speed at the tick equals the last executed step speed, and the acceleration at the tick equals the executed one-step acceleration (difference 0.0; the checks of cards ADD54 and ADD55).
* c7: 141 dumps per run, 2,399 rows per tick; the reused files and their known values reproduce (levels and RS of `R0`, `R0M5`, `R0C5`, `R0E5`, `R0E10` as rows 110 and 112 print them).
* c8: the reader controls of the pinned readers (headway and families); checkpoint hashes of the two models.
* c9: the unit test of the chord limit (24 checks of card ADD56) and the tick-readout check of card ADD55 pass on the pod with the new wrapper, and the runs carry that wrapper's sha256.
* Reported without a bar, r1: the share of steps at a block boundary (the first step of a block, its chord acceleration taken against the last step of the block before) that lie beyond the limit by more than 0.25 m/s2, for `R0CE5`, and for the banked `R0E5` beside it. It says whether the pair is one bound across replans, as section 1 expects; it gates nothing because the executed step speed and the chord speed are two quantities of the tool whose equality at the boundary this card has not shown.

A control fails: nothing is named and the read is reported as written; the tool fault is logged in `TIMES_NOTE.md` before its repair, and the runs are repeated under new run names.

## 6. What follows (committed before the data)

* `STABILISES-1S`: a state readout and an executor bound without a study device steady the 1.0 s cycle on these two models. It becomes the coordinator's candidate in the decision card on the state rebuild for short cycles that goes to the PI and the other session, with its cost at the 2.0 s cycle (PR) and the GC read beside; the lead rule under this pair is then read on its own card. Nothing is adopted by this card.
* `SLOWS-CLIMB` or `NO-EFFECT`: no pairing of executor bound and readout without a study device steadies the short cycle. This line ends here without training: the decision card states that the 2.0 s cycle stays the programme's cycle for capability numbers (coordinator decision D12 stands), that the backward readout and the acceleration at 0 stay study devices, and that the lever at the short cycle is the other session's training lever (the model's own rolled history at a short cycle). The reads of cards ADD58 and ADD59 are put beside it.
* `BELOW-BOTH-PARTS` or not: reported as written; it says whether the two parts add.

## 7. Limits

Two model seeds of one training recipe. The chord limit is a study device in the sense of card ADD56: it bounds the speed change along the track and says nothing about lateral acceleration. The causal readout hands the model the executed one-step acceleration, which it was not trained with at a short cycle. 141 episodes are read again, with no adjustment for the repeated reads. The lead does not react. The evaluation convention of the programme and its pinned tool are not changed; adopting a readout or a limit in a protocol is the PI's and the other session's decision.
