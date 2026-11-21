# PREREG_V8F_PROOF_ADD51 (TAUCURVE10K): the time gap of the lead rule on the 10k checkpoints, label-fed and reader-fed; no training

Card of the v8F proof study, written by the coordinator fork while the rollouts of card ADD50 ran and before the read of card ADD50; it does not use that read (its forms are those of cards ADD47 and ADD48). Written before any run of this card (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; the lead does not react and follows its logged track); never driving performance.

## 1. What is known (MEASURED, T1-kin; rows H-V8F-PROOF-102 and -104)

* Unsafe rate per model seed on the two 10k union checkpoints: no rule 0.1459, 0.1139; reader-fed rule at map level 0.50 with closing 0 (`RQ50`, the deployable form) 0.0872, 0.0730; label-fed rule with the label closing speed (`RLc`, the best measured form, not deployable) 0.0463, 0.0302. Falling behind: 0.0950, 0.1854; 0.2274, 0.2913; 0.1838, 0.2523.
* In every arm since cards ADD26 and ADD33 the rule asks for a time gap of 1.2 s (and at least 3 m). The unsafe step is scored at a time gap under 1.0 s or a gap under 2.0 m; the scoring does not read the rule's parameter. No arm on the 10k checkpoints varied the time gap.

## 2. Question

How do the unsafe rate, the contact rate and falling behind move with the time gap the rule asks for, at label quality and for the deployable reader-fed form? Beside it: what the reader-fed form reaches at a longer time gap against the label-fed form at 1.2 s, and at what price in falling behind.

## 3. Arms (no training, no new reader, no new map)

Models (`UE_s0`, `UE_s1`), reader arms, maps (level 0.50), presence threshold 0.5, floor form cap-only, the other rule parameters, evaluation cache, window grid and pinned tool as cards ADD46 to ADD50. Wrapper `integ10k/v8f_t1_eval_reader_rule_t.py` (written by `integ10k/make_taucurve10k.py` with asserted text edits from the wrapper of card ADD50): one more parameter, the time gap handed to the rule; without it nothing changes.

| run | form | time gap | role |
|---|---|---|---|
| `RLcT08`, `RLcT10`, `RLcT15`, `RLcT20` | labels: presence, gap, closing speed | 0.8, 1.0, 1.5, 2.0 s | curve at label quality |
| `RQ50T08`, `RQ50T10`, `RQ50T15`, `RQ50T20` | reader presence and gap at level 0.50, closing 0 | 0.8, 1.0, 1.5, 2.0 s | curve of the deployable form |
| `RLcT12` | labels, model seed 0 | 1.2 s given as a parameter | control t1: must equal the banked `RLc` |

Each curve run at both model seeds (reader seed = model seed). Reused without a rerun: `R0`, `RLc`, `RQ50` (the points at 1.2 s). Seventeen new rollouts.

## 4. Reads

Headway reader and family reader of card ADD46, pooled over the two model seeds, paired episode-cluster bootstrap (it answers another draw of episodes; the two model seeds are the only replicate for training variance; the rollout does not sample). U on the 562 scored windows per seed, C and F on the 642 lead windows.

* Per form and time gap: the levels of U, C, F per model seed, and the difference against the same form at 1.2 s.
* LONG (per form) = U at 2.0 s minus U at 1.2 s, with its F difference as the price. SHORT (per form) = U at 0.8 s minus U at 1.2 s, with its F difference.
* Beside, descriptive: the reader-fed form at 1.5 s and at 2.0 s against the label-fed form at 1.2 s (U, C, F); both forms at 2.0 s against no rule; the four families for 0.8 s and for 2.0 s against 1.2 s, both forms.

## 5. Names (fixed here; per form, LABEL or READER)

* `TAUCURVE10K-<form>-LONGER-GAP-CUTS-UNSAFE`: LONG below 0 at both model seeds and its pooled interval excludes 0; ending `-AT-A-PRICE` if the interval of the F difference lies above 0, else `-AT-NO-PRICE`.
* `TAUCURVE10K-<form>-NO-CUT-BEYOND-1.2`: otherwise.
* `TAUCURVE10K-<form>-SHORTER-GAP-COSTS-SAFETY`: SHORT above 0 at both model seeds and its pooled interval lies above 0; else `TAUCURVE10K-<form>-SHORTER-GAP-NO-SAFETY-COST`. Ending `-LESS-FALLING-BEHIND` if the interval of its F difference lies below 0.

What follows, fixed now:
* No time gap is chosen for the 10k candidate from this card. These evaluation windows have been read by several cards; a choice on them would be a fit on the evaluation data. The candidate keeps 1.2 s (decision D10).
* READER `-LONGER-GAP-CUTS-UNSAFE`: the coordinator writes a card for the choice of the time gap on clips outside the evaluation set (a rollout on held-out clips of the training corpus; a tool build) and puts the trade it shows to the PI.
* `-NO-CUT-BEYOND-1.2` at both forms: the time gap is no lever beyond 1.2 s; the windows that stay unsafe under the label-fed form with the closing speed are split by cause (descriptive, from the rule logs) as the next step.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 6. Controls (all must pass; else nothing is named)

* t1 (wrapper, known value): the dump of `RLcT12` at model seed 0 equals the banked dump of `RLc` at model seed 0 array for array (`cmp_dumps.py`, 141 files, 0 differing).
* t2 (the parameter is applied): the rule log of every curve run records its time gap, the banked logs of `RLc` and `RQ50` record 1.2 s.
* t3 (only the parameter differs, known value): at the first tick of every window the source gap, the closing speed, the ego speed and the lead speed are equal between a curve run and the banked run of the same form and model seed (within 1e-6), and the tick index columns are equal.
* t4 (same read): the reader's presence probability, read gap, fed gap and history flag of every tick are equal in the side logs of `RQ50` and of every `RQ50T` run of the model seed (the reader reads logged frames only).
* t5 (counts): 141 dump files in each of the seventeen new dumps, 2,399 windows and the same lead windows.
* t6 (reused dumps): the result files of the reused runs have the sha256 recorded before the run (`integ10k/TAUCURVE10K_REUSED.json`).
* t7: the controls of the headway reader and of the family reader on the new pairs.
* t8 (model): the checkpoint sha256 written by each new run equals that of card ADD46 for its model seed.

The pod read starts with control e1 (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf).

## 7. Limits

56 episodes with a lead, read again (cards ADD46 to ADD50 before); no adjustment for the repeated reads is applied. A lead that does not react; two model seeds. At 0.8 s the rule itself allows time gaps the scoring counts as unsafe (under 1.0 s), so a safety cost at 0.8 s is expected by construction and says little; the informative side is 1.5 s and 2.0 s. Falling behind is scored at 10 m at the last scored step and does not see how far behind; the family reads carry the size. The comparison of the reader-fed form at a longer time gap with the label-fed form at 1.2 s is descriptive and names nothing.

STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument).
