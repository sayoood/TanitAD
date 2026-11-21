# PREREG_V8F_PROOF_ADD30: MARGIN. Does a safety margin on the vision lead read, calibrated on training clips only, bring the vision-fed distance rule to the label-fed one? Zero training.

Frozen before any number of this card is computed, the calibration constants included. Addendum to `PREREG_V8F_PROOF.md`, to card ADD26 with its note N1 and to card ADD29. Tier T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react; a diagnostic, never driving performance). Evidence class of every result: MEASURED once run. LONGITUDINAL is the family measured (distance keeping and the four-family cost read); LATERAL and TACTICAL are read as cost cells; STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it

* MEASURED (card ADD26, row H-V8F-PROOF-54): the label-fed rule cuts the model-induced unsafe rate U by about 0.08 on the head checkpoints; the head-fed rule reaches, seed 0 / seed 1, a share of 0.55 / 0.47 (LS) and 0.22 / 0.17 (LSD) of that fall. `HEAD-RULE-REACHES-ORACLE` is not named.
* MEASURED (pod table of 2026-11-15, the ticks of the head-fed and label-fed rollouts matched on clip, start and tick; a description, no outcome): the head misses 15 to 20 percent of the ticks at which the labels have a lead, calls a lead at 20 to 29 percent of the ticks at which the labels have none, and its gap differs from the label gap by about 9 m RMSE with a 90th percentile of the over-read of +6.6 to +9.6 m.
* MEASURED (row H-V8F-PROOF-56, cards ADD27 and ADD28): neither a filter over the gap reads nor a probe on the pooled camera tokens gives a finer lead read.
* Card ADD29 (which part holds the rule back): its seed-0 rollouts are done and its seed-1 rollouts are running while this card is written; none of its distance or cost numbers has been read. This card does not depend on its outcome: the two constants below cover the presence part and the gap part, and the dose arms show which one moves U.
* Seen before the freeze: everything above. Not seen: any rollout with a threshold other than 0.5 or a margin other than 0, and any calibration constant.

## 2. The lever

The distance rule of card ADD26 with note N1 (keep 3.0 m plus 1.2 s, brake at most 6.0 m/s2, never move a plan ahead, cap-only), fed by the vision head only (presence, gap, closing speed all from the head, as the head-fed arms of card ADD26), with two fixed constants on the head read:

* presence threshold `P`: a lead is taken as present at head probability at least `P` (card ADD26: 0.5);
* gap margin `M`: the rule takes the head gap minus `M` metres, not below 0 (card ADD26: 0).

Tool: `distrule/v8f_t1_eval_rule_margin.py` (imports `v8f_t1_eval_rule_mix.py` and the wrapper of card ADD26 unedited; the pinned eval tool stays pinned).

## 3. Calibration (train probe clips only; no evaluation file is read)

Per checkpoint, from its banked head read on the 1,200 train probe clips (`spatial/leadsup/read/in/<ARM>_train_lead.npz`, rows with a full history) and the labels of the train pack (`spatial/leadsup/read/train_pack.npz`), rows matched by clip name and frame:

* near-lead rows = rows whose label has a lead with a finite gap of at most 40 m;
* `P*` = the largest value in {0.5, 0.4, 0.3, 0.2, 0.1, 0.05} at which the head calls at least 95 percent of the near-lead rows present; 0.05 if none does;
* over-read = head gap minus label gap on the near-lead rows; `M50`, `M80`, `M95` = its 50th, 80th and 95th percentile, each not below 0 and rounded up to the next 0.5 m.

The script (`distrule/margin_calib.py`) writes the constants, the row counts, the recall at 0.5 and at `P*`, the share of label-absent rows called present at 0.5 and at `P*`, and the sha256 of every file it read to `distrule/margin/CALIB.json`. The constants are fixed by that file before the first rollout and are not changed after.

## 4. Arms (checkpoints LS_s0, LS_s1, LSD_s0, LSD_s1; rollouts named `<BASE><code>_s<seed>`)

| code | P | M | role |
|---|---|---|---|
| MC | P* | M80 | **the calibrated rule: the only arm that is named** |
| MT | P* | 0 | threshold alone (reported) |
| MG | 0.5 | M80 | margin alone (reported) |
| ML | P* | M50 | lower dose (reported) |
| MH | P* | M95 | higher dose (reported) |
| MR | 0.95 | 0 | deliberate regression, LS_s0 and LS_s1 only (control g5) |

22 rollouts, one at a time, never as a fourth GPU job, memory check before each (as card ADD29's queue).

## 5. Read

`headway/headway_read.py` (U primary, contact, falling behind, gap bias), paired on windows, two-level bootstrap (seeds, then episodes), B 2,000, rng 0: each arm against no rule (`LSMC:LS`), against the head-fed rule of card ADD26 (`LSMC:LSRH`) and against the label-fed rule (`LSMC:LSRL`). `spatial/q3/q3_read.py`, four families, each arm against no rule, spread of G on six seeds. 642 lead windows of 56 episodes for the distance cells, 2,399 windows of 141 episodes for the cost cells.

Share reached = (U of no rule minus U of the arm) / (U of no rule minus U of the label-fed rule), per checkpoint.

## 6. Controls (a failed control is reported first and the outcomes it carries are not named)

* g1: the margin tool at `P` 0.5, `M` 0 on LS_s0 gives arrays equal to the banked head-fed dump LSRH_s0 (0 files differing).
* g2: no plan is moved ahead in any rule log of this card (lift 0).
* g3: at the first tick (identical states in both runs), on ticks the head calls present in both, the gap the rule used in each MC, MG, ML, MH run equals the gap of the head-fed run of card ADD26 minus `M`, not below 0, within 1e-6 m.
* g4: `CALIB.json` lists only train files (no path of an evaluation file), with their sha256.
* g5: the deliberate regression MR is read as worse than the head-fed rule on U for LS (difference above 0 in both seeds, two-level interval excluding 0). If the reader cannot see this, `MARGIN-CUTS-MORE` and `MARGIN-EQUAL` are not named.
* g6: the reader controls of every read.

## 7. Outcomes (per arm type LS and LSD, both seeds needed; arm MC only)

* `MARGIN-REACHES-ORACLE`: the share reached is at least 0.5 in both seeds and the two-level interval of U(MC) minus U(no rule) lies below 0. Otherwise not named.
* `MARGIN-CUTS-MORE`: U(MC) minus U(head-fed rule of card ADD26) is below 0 in both seeds with a two-level interval excluding 0. `MARGIN-WORSE`: above 0 in both seeds with an interval excluding 0. `MARGIN-EQUAL`: otherwise.
* Cost against no rule by the rule of card ADD11: `MARGIN-NO-COST` if no cell of the four families is named worse, else `MARGIN-COSTS` with the cells. Reported beside in every case: position error over 6 s, along-track error, share of windows more than 10 m behind, gap bias, and the same cells of the label-fed and head-fed rules of card ADD26 on the same checkpoints.
* The dose arms (MT, MG, ML, MH) are reported as a table of U, contact and cost; they name nothing.

A separated interval here answers the question about episodes and, through the two-level bootstrap over two seeds, only weakly the one about training runs; inference is deterministic, so a difference between two rollouts of one checkpoint is the constants alone.

## 8. What follows (committed before the data)

* `MARGIN-REACHES-ORACLE` with `MARGIN-NO-COST` for an arm type: the calibrated vision-fed rule is the form of the safety check for that arm type; the next card trains the margin into the head (a low-quantile gap output and a recall-weighted presence output on the banked states, so that the margin depends on the input) and reads whether the cost falls at equal U.
* `MARGIN-REACHES-ORACLE` with `MARGIN-COSTS`: the same next card, with the cost cells as its primary read.
* Not reached, and the dose arms show U falling with the dose: the read is too coarse for a fixed margin at an acceptable cost; the same next card (input-dependent margin) runs, and the finer gap read (unpooled tokens or a second encoder pass over the lead region; camera frames on the pod, a decision of the PI) is named as the blocker with the dose table as its evidence.
* Not reached, and the dose arms flat: the shortfall is not in presence recall or gap over-read; card ADD29's attribution decides the next lever.
