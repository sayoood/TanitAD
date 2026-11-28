# PREREG_V8F_PROOF_ADD52 (CYCLE10K): the replanning cycle of model and lead rule at 1.0 s instead of 2.0 s on the 10k checkpoints; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD50 and the split of the unsafe windows (section 1), while the rollouts of card ADD51 ran and before the read of card ADD51; it does not use that read. Written before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known (MEASURED, T1-kin)

* Every rollout of the level line replans every 10 cache steps (2.0 s): the model gives a plan, the rule caps it, 10 steps are executed, three plans per 6 s window. The rule acts only at those ticks.
* Split of the windows that stay unsafe (`integ10k/read_gaprate/UNSAFE_CAUSE.json`; descriptive, convention in `TIMES_NOTE.md` before it ran): under the label-fed rule with the label closing speed (`RLc`, unsafe rate 0.0463, 0.0302) all 26 and 17 unsafe windows have the braking limit of the rule (6 m/s2) reached on an executed step of the plan in force; all are of the time-gap kind (no gap under 2.0 m), 13 and 8 of them shallow (smallest time gap at least 0.8 s). With label inputs and the closing speed, what stays unsafe is no perception miss: the rule acts too late or too weakly.
* Under the reader-fed rule (`RQ50`, 49 and 41 unsafe windows): the plan in force was left unchanged in 21 and 14, the braking limit was reached in 15 and 16, the reader saw no lead at the tick in 8 and 7 (all but one at the second tick), the plan was capped in 5 and 4.
* Card ADD50 (row 106): the reader floor on the unsafe rate at level 0.50 is 0.0196.

## 2. Question

How much of what stays unsafe is the 2.0 s reaction interval of the rollout protocol? At a 1.0 s cycle the rule sees the lead twice as often. Model and rule share the cycle, so the card reads the form without a rule at the same cycle as its own reference.

## 3. Arms (no training, no new reader, no new map)

Models (`UE_s0`, `UE_s1`), reader arms, maps (level 0.50), presence threshold 0.5, floor form cap-only, rule parameters (time gap 1.2 s), evaluation cache, window grid and pinned tool as cards ADD46 to ADD51; wrapper of card ADD51, unchanged. The one change: the pinned tool's own parameter for the replan interval, 5 steps instead of 10.

| run | form | replan interval | role |
|---|---|---|---|
| `R0M5` | no rule | 5 steps (1.0 s) | the model alone at the faster cycle |
| `RLcM5` | labels: presence, gap, closing speed | 5 steps | the rule at label quality |
| `RQ50M5` | reader presence and gap at level 0.50, closing 0 | 5 steps | the deployable form |
| `RLcM10` | labels, model seed 0 | 10 steps given as a parameter | control y1: must equal the banked `RLc` |

Each M5 run at both model seeds (reader seed = model seed). Reused without a rerun: `R0`, `RLc`, `RQ50` (the 2.0 s cycle). Seven new rollouts. The folder names of the new runs keep the ending the readers look for; the interval of a run is in its run record and is checked (control y2).

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap on the same windows (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample). U on the 562 scored windows per seed, C and F on the 642 lead windows.

* CY (per form NORULE, LABEL, READER) = U at the 1.0 s cycle minus U at the 2.0 s cycle, same form.
* RU (per rule form) = U of the rule form minus U of no rule, both at the 1.0 s cycle; beside it the same difference at 2.0 s.
* Beside: C and F for every pair; the four families for every form at 1.0 s against 2.0 s and for each rule form against no rule at 1.0 s; the split of section 1 repeated on the M5 runs (descriptive).

## 5. Names (fixed here)

* Per form: `CYCLE10K-<form>-FASTER-CYCLE-CUTS-UNSAFE` if CY is below 0 at both model seeds and its pooled interval excludes 0; `CYCLE10K-<form>-FASTER-CYCLE-RAISES-UNSAFE` if the pooled interval lies above 0; else `CYCLE10K-<form>-NO-U-CHANGE`.
* Per rule form: `CYCLE10K-<form>-RULE-CUTS-UNSAFE-AT-1S` if RU is below 0 at both model seeds and its pooled interval excludes 0; else `CYCLE10K-<form>-RULE-NOT-SEPARATED-AT-1S`.
* For the READER form a CY inside the reader floor of row 106 (absolute pooled value at most 0.0196) is said to be inside that floor wherever it is quoted.

What follows, fixed now:
* The evaluation convention of the programme (2.0 s) is not changed by this card: the shared readers and noise floors of both sessions rest on it. If a rule form reads FASTER-CYCLE-CUTS-UNSAFE, the coordinator proposes to the other session and the PI that the 10k candidate is read at both cycles, and the next arm here is the 0.4 s cycle (2 steps) on the same three forms.
* LABEL reads NO-U-CHANGE: the reaction interval is not the term; the next arm is the braking limit of the rule at label quality (6 against 8 m/s2, no training).
* NORULE reads RAISES-UNSAFE or its families get worse at the faster cycle: reported as written and passed to the other session as a finding on the model, not on the rule.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* y1 (queue edit, known value): the dump of `RLcM10` at model seed 0 equals the banked dump of `RLc` at model seed 0 array for array (`cmp_dumps.py`, 141 files, 0 differing).
* y2 (the interval is applied): the rule logs of the M5 rule runs record 5 steps, hold twice the rows of the banked logs and the ticks 0 to 5; the result files of all M5 runs record 5 steps, those of the banked runs 10.
* y3 (only the interval differs, known value): at the first tick of each of the 2,399 windows the source gap, the closing speed, the ego speed and the lead speed are equal between an M5 rule run and the banked run of the same form and model seed (within 1e-6).
* y4 (same read): at the cache steps both cycles share, the reader's presence probability, read gap, fed gap and history flag are equal in the side logs of `RQ50` and `RQ50M5` (the reader reads logged frames only); every tick of the banked run is among them.
* y5 (counts): 141 dump files in each of the seven new dumps, 2,399 windows and the same lead windows.
* y6 (reused dumps): the result files of the reused runs have the sha256 recorded before the run (`integ10k/CYCLE10K_REUSED.json`).
* y7: the controls of the headway reader and of the family reader on the new pairs.
* y8 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

56 episodes with a lead, read again (cards ADD46 to ADD51 before); no adjustment for the repeated reads is applied. A lead that does not react; two model seeds. Model and rule change their cycle together, so CY of a rule form mixes the faster rule with a model that replans more often; RU and the NORULE form are there to tell them apart, and they do so only in part. The models were trained and selected under the 2.0 s convention of the other session's reads; a 1.0 s cycle is a protocol the checkpoints were never read at. The split of section 1 was made after the unsafe rates were known and names nothing.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument). The first tick does not depend on the cycle.
