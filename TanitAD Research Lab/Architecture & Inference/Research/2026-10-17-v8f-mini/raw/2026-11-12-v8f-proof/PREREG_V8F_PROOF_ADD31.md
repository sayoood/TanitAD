# PREREG_V8F_PROOF_ADD31: QCAL. A calibrated lower quantile of the gap, as a function of the head's own gap read, in place of a fixed margin. Zero training of the model.

Frozen before any number of this card is computed, the calibration maps included, and before any rollout of card ADD30 has been read (its queue is waiting on the pod while this card is written; the constants of card ADD30 are known: P* 0.4 / 0.5 / 0.4 / 0.3 and M80 6.5 / 7.5 / 8.0 / 8.0 m for LS_s0, LS_s1, LSD_s0, LSD_s1). Addendum to `PREREG_V8F_PROOF.md`, to card ADD26 with its note N1 and to card ADD30. Tier T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react; a diagnostic, never driving performance). Evidence class of every result: MEASURED once run. LONGITUDINAL is the family measured; LATERAL and TACTICAL are read as cost cells; STRATEGIC: PAUSED (PI item 25).

## 1. Why

A fixed margin takes the same number of metres off every gap read. The head's error is not the same everywhere: on the train probe clips its over-read of near leads has a median of +0.7 to +3.1 m and a 95th percentile of +12.5 to +16.1 m (calibration file of card ADD30). A fixed margin of 6.5 to 8 m is then too large where the read is good and too small where it is poor. The lever here keeps the coverage of card ADD30 (the gap the rule uses lies above the label gap in about one row of five) and lets the amount depend on the read.

## 2. The lever

The vision-fed distance rule of card ADD26 with note N1 (all three parts from the head), presence threshold `P*` of card ADD30 for the checkpoint, and the gap replaced by

`g_used = min(g_head, Q(g_head))`

where `Q` is a piecewise-linear, non-decreasing map fitted per checkpoint on the train probe clips only (section 3). The rule never gets a gap larger than the head read. Tool: `distrule/v8f_t1_eval_rule_qcal.py` (imports `v8f_t1_eval_rule_mix.py` and the wrapper of card ADD26 unedited; the pinned eval tool stays pinned).

## 3. The map (train probe clips only; no evaluation file is read)

Rows: the banked head read of the checkpoint on the 1,200 train probe clips (`spatial/leadsup/read/in/<ARM>_train_lead.npz`), rows with a full history, a finite head gap, and a label lead with a finite gap (any distance), matched to the train pack by clip name and frame. Bins of the head gap with edges 0, 5, 10, 15, 20, 25, 30, 40, 50 m and above; a bin with fewer than 200 rows is merged into its lower neighbour. Per bin: knot position = the median head gap of the bin, knot value = the 20th percentile of the label gap of the bin. The knot values are then made non-decreasing in the knot position (pool adjacent violators, weights = row counts) and capped at the knot position. Between knots the map is linear; below the first knot it is the first knot's value scaled by `g_head / x_first`; above the last knot it is `g_head` minus the last knot's shift. The script (`distrule/qcal_fit.py`) writes the knots, the row counts, the share of train rows with `g_used` above the label gap, and the sha256 of every file it read to `distrule/qcal/<ARM>.json`. The maps are fixed by those files before the first rollout.

## 4. Arms

`<BASE>QC_s<seed>` on LS_s0, LS_s1, LSD_s0, LSD_s1: threshold `P*`, map `Q`. Four rollouts, after the queue of card ADD30, one at a time, never as a fourth GPU job, memory check before each. Control rollout `LSQ0_s0`: threshold 0.5, the identity map.

## 5. Read

As card ADD30 section 5: `headway_read.py` and `q3_read.py`, QC against no rule, against the head-fed rule of card ADD26 (`RH`), against the label-fed rule (`RL`) and against the fixed-margin arm of card ADD30 (`MC`), paired on windows, two-level bootstrap (seeds, then episodes), B 2,000, rng 0. Share reached as card ADD30.

## 6. Controls (a failed control is reported first and the outcomes it carries are not named)

* q1: the tool at threshold 0.5 with the identity map on LS_s0 gives arrays equal to the banked head-fed dump LSRH_s0 (0 files differing).
* q2: no plan is moved ahead in any rule log of this card (lift 0).
* q3: at the first tick, on ticks the head calls present in both runs, the gap the rule used in each QC run equals the map applied to the gap of the head-fed run of card ADD26, within 1e-4 m (the logs hold the gaps in single or double precision).
* q4: each map file lists only train files, with their sha256.
* q5: a unit check of the map in the tool: knots reproduce their values, the map is non-decreasing, and `g_used` is never above `g_head` (run before the queue; printed in the log).
* q6: the reader controls of every read; control g5 of card ADD30 (the deliberate regression is read as worse) carries the comparison outcomes of this card too.

## 7. Outcomes (per arm type LS and LSD, both seeds needed)

* `QCAL-REACHES-ORACLE`: the share reached is at least 0.5 in both seeds and the two-level interval of U(QC) minus U(no rule) lies below 0. Otherwise not named.
* Against the fixed margin, U: `QCAL-CUTS-MORE` if U(QC) minus U(MC) is below 0 in both seeds with a two-level interval excluding 0; `QCAL-CUTS-LESS` if above 0 in both seeds with an interval excluding 0; `QCAL-U-EQUAL` otherwise.
* Against the fixed margin, cost: position error over 6 s on all 2,399 windows, QC minus MC, paired: `QCAL-CHEAPER` if below 0 in both seeds with a two-level interval excluding 0; `QCAL-DEARER` if above 0 likewise; `QCAL-COST-EQUAL` otherwise.
* Cost against no rule by the rule of card ADD11: `QCAL-NO-COST` or `QCAL-COSTS` with the cells.

## 8. What follows (committed before the data)

* `QCAL-REACHES-ORACLE` with `QCAL-NO-COST`, or QC at least as good as MC on U and `QCAL-CHEAPER`: the calibrated-quantile form is taken forward; the next card trains it into the head (a low-quantile gap output on the model's states, all training clips) and reads the same cells.
* QC no better than MC on U and on cost: the dependence of the error on the read itself carries no gain; the next lever is then a finer lead read (the decision of the PI named in row H-V8F-PROOF-56) or, within the present authority, the trained low-quantile head with the state as input rather than the head's own read.
* In every case the dose table of card ADD30 and this card's row together give the trade between unsafe steps and position error for the vision-fed rule, which is the input of the training card with the rule in the loop.
