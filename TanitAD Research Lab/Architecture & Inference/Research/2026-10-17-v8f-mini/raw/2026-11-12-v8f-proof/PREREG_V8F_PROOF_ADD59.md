# Card PREREG_V8F_PROOF_ADD59 (RULE1S10K): the lead rule at the 1.0 s cycle once the loop is steady

Card of the v8F proof study, written by the coordinator fork. Frozen before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). No training, no spend beyond the pod of this fork. Tier: T1-kin (kinematic rollout on the model's own actions); never driving performance.

## 1. What is known

* Card ADD52 (MEASURED, row 107): at the 1.0 s cycle with the pinned state readout the unsafe rate rises for every form (`CYCLE10K-<form>-FASTER-CYCLE-RAISES-UNSAFE`), because the model's own speed climbs at that cycle; the rule still cuts the unsafe rate against no rule at 1.0 s. Whether a shorter reaction interval helps the rule could not be read.
* Card ADD54 (MEASURED, row 110): with the backward readout the loop of the two 10k models is stable at 1.0 s without a rule (unsafe rate 0.1530, 0.1512 against 0.1459, 0.1139 at 2.0 s with the pinned readout; the difference is not separated).
* Card ADD55 (MEASURED, read 2026-11-19): with the acceleration held at 0 (ZERO) the loop is stable at 1.0 s too (unsafe rate 0.2206, 0.2117; against the banked 2.0 s run separated worse by +0.0863).
* Banked at the 2.0 s cycle with the pinned readout: unsafe rate of the label-fed rule `RLc` 0.0463, 0.0302; the reader-fed rule `RQ50` 49 and 41 unsafe windows of 562.

## 2. Question

With a readout under which the loop is steady at 1.0 s: does the lead rule cut unsafe headway further at the 1.0 s cycle than at the 2.0 s cycle? Both readouts are study devices (coordinator decision D12: nothing at 1.0 s is a capability number); the card measures whether the short cycle is worth a state rebuild or the training lever.

## 3. Arms (no training)

Models `UE_s0`, `UE_s1`, evaluation cache, window grid, pinned tool, rule parameters, reader checkpoints and map level 0.50 as cards ADD46 to ADD57. Wrapper: `integ10k/v8f_t1_eval_reader_rule_rz.py` of card ADD55, unchanged (sha256 prefix fa7207d72311b5cf); no new wrapper. Forms: `R0` (no rule), `RLc` (labels: presence, gap, closing speed), `RQ50` (reader presence and gap at map level 0.50, closing 0).

Primary readout: backward (its run without a rule at 1.0 s is the one not separated from the banked 2.0 s run). Second readout, beside: ZERO.

| run | form | readout | replan interval | role |
|---|---|---|---|---|
| `RLcBk5` | labels | backward | 5 steps (1.0 s) | the rule at label quality, short cycle |
| `RQ50Bk5` | reader | backward | 5 steps | the deployable form, short cycle |
| `R0Bk10` | none | backward | 10 steps (2.0 s) | the same readout at the programme's cycle |
| `RLcBk10` | labels | backward | 10 steps | |
| `RQ50Bk10` | reader | backward | 10 steps | |
| `RLcZ5` | labels | causal_z | 5 steps | second readout, beside |
| `RQ50Z5` | reader | causal_z | 5 steps | second readout, beside |
| `R0BkID` | none, model seed 0 | backward | 5 steps | control r1: must equal the dump of `R0B5` at model seed 0 of card ADD54 |

Every run except the control at both model seeds (reader seed = model seed): fifteen new rollouts. Reused without a rerun: `R0B5` (card ADD54), `R0Z5` (card ADD55), and with the pinned readout `R0`, `RLc`, `RQ50` (2.0 s) and `R0M5`, `RLcM5`, `RQ50M5` (1.0 s, card ADD52).

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample). U on the 562 scored windows per seed, C and F on the 642 lead windows. Forms NORULE (`R0`), LABEL (`RLc`), READER (`RQ50`).

* CY (per form) = U at the 1.0 s cycle minus U at the 2.0 s cycle, both with the backward readout.
* RU (per rule form) = U of the rule form minus U of no rule, both at 1.0 s with the backward readout; beside it the same difference at 2.0 s with the backward readout.
* GB (per rule form) = U of the rule form at 1.0 s with the backward readout minus U of the banked run of the same form at 2.0 s with the pinned readout.
* RB (per form) = the form at 2.0 s with the backward readout minus the banked form at 2.0 s with the pinned readout (what the readout itself does at the programme's cycle).
* Beside, naming nothing: under ZERO, per rule form, the rule minus no rule at 1.0 s and the rule at 1.0 s minus the banked rule at 2.0 s (pinned); C and F for every pair; the four families for every pair.

## 5. Names (fixed here)

* Per form: `RULE1S10K-<form>-FASTER-CYCLE-CUTS-UNSAFE` if CY is below 0 at both model seeds and its pooled interval excludes 0; `RULE1S10K-<form>-FASTER-CYCLE-RAISES-UNSAFE` if the pooled interval lies above 0; else `RULE1S10K-<form>-NO-U-CHANGE`.
* Per rule form: `RULE1S10K-<form>-RULE-CUTS-UNSAFE-AT-1S` if RU is below 0 at both model seeds and its pooled interval excludes 0; else `RULE1S10K-<form>-RULE-NOT-SEPARATED-AT-1S`.
* Per rule form: `RULE1S10K-<form>-BELOW-BANKED-2S` if GB is below 0 at both model seeds and its pooled interval excludes 0; `RULE1S10K-<form>-ABOVE-BANKED-2S` if the pooled interval lies above 0; else `RULE1S10K-<form>-NOT-SEPARATED-FROM-BANKED-2S`.
* The programme's cycle: `RULE1S10K-<form>-BACKWARD-COSTS-AT-2S` if the pooled interval of RB on U lies above 0, `-HELPS-AT-2S` if it lies below 0, else `-NEUTRAL-AT-2S` (per form).
* For the READER form a difference inside the reader floor of row 106 (absolute pooled value at most 0.0196) is said to be inside that floor wherever it is quoted.

What follows, fixed now:
* The evaluation convention of the programme (2.0 s, pinned readout) is not changed by this card.
* LABEL reads `FASTER-CYCLE-CUTS-UNSAFE` or `BELOW-BANKED-2S`: at label quality the shorter reaction interval is worth something once the loop is steady. The size goes into the decision card for the PI and the other session as the measured gain of a short cycle, with the statement that the readout is a study device; the next arm here is the 0.4 s cycle under the same readout.
* LABEL reads `NO-U-CHANGE` and `NOT-SEPARATED-FROM-BANKED-2S` (or worse): the reaction interval is not the term at label quality even with a steady loop; the short cycle is not a lever of the distance line for now, and the levers that remain are the braking limit (card ADD57) and the lead read (card ADD45).
* READER differs in sign from LABEL: reported as written; the read of the lead is then the term at the short cycle (more ticks, more reads), which is card ADD45's question.
* NORULE CY and RB: reported as the loop's own cost of the cycle and of the readout.
* The two study readouts disagree in the sign of a separated difference: nothing about the short cycle is carried into the decision card beyond both reads side by side.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* r1 (queue, known value): the dump of `R0BkID` at model seed 0 equals the dump of `R0B5` at model seed 0 of card ADD54 array for array (`cmp_dumps.py`, 141 files, 0 differing). The queue step that makes this comparison uses a marker name no earlier card used.
* r2 (parameters applied): the readout log of every new run records its readout, its interval and a call count above 0; the result files record 5 steps for the 5-step runs and 10 for the 10-step runs; the rule logs of the rule runs record the same steps, and those of the 5-step rule runs hold the ticks 0 to 5.
* r3 (the readout does what it says, from the readout log over all committed plans): in every new run the speed at the last executed knot equals the last executed step speed (largest difference at most 1e-6); in the ZERO runs the acceleration at the tick is 0.
* r4 (the first block does not depend on the readout, known value): the first 5 executed steps of `RLcBk5` and `RLcZ5` equal those of the banked `RLcM5` at the same model seed, those of `RQ50Bk5` and `RQ50Z5` equal those of the banked `RQ50M5`; the first 10 of `R0Bk10`, `RLcBk10`, `RQ50Bk10` equal those of the banked `R0`, `RLc`, `RQ50` (largest difference 0.0), for the model and for the prior. At the first tick of each of the 2,399 windows the source gap, the closing speed, the ego speed and the lead speed in the rule log equal those of the banked run of the same form and cycle (within 1e-6).
* r5 (counts): 141 dump files in each of the fifteen new dumps, 2,399 windows and the same lead windows.
* r6 (reused dumps): the result files of the eight reused runs (per model seed) have the sha256 recorded before the run (`integ10k/RULE1S10K_REUSED.json`).
* r7: the controls of the headway reader and of the family reader on the new pairs.
* r8 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed; the reader checkpoint and the map of the reader-fed runs are those of the banked `RQ50` runs (sha256 in the run records).
* r9 (tool): the wrapper on the pod has the sha256 the tool check of card ADD55 ran with (fa7207d72311b5cf).

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

Two model seeds; 141 episodes read again, no adjustment for the repeated reads. Both readouts are study devices: the backward one hands over an acceleration of the opposite sign and a quarter of the size of the executed one, ZERO none; neither is a state rebuild, and the heading at the last executed knot still comes from the central tangent. GB compares across readout and cycle at once; CY and RB separate the two. The rule's parameters are those tuned at the 2.0 s cycle. Nothing at 1.0 s is a capability number (coordinator decision D12).

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick does not depend on the readout.
