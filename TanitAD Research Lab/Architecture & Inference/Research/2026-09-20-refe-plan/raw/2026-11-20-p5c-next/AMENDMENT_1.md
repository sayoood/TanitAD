# PREREG_P5C_NEXT · AMENDMENT 1: the selection control is computed at the planner's precision (float32)

**Written 2026-11-20 ~09:55Z by the MM.** The amendment is frozen by its sha256 in the MM loop state BEFORE `propscore/S_RULE_READ.json`
(the as-written run) is opened. Only the per-arm control block of `full_C1_read.json` had been seen. That block holds CTRL-T, CTRL-P
and CTRL-S. A tail of that file printed by the runner also showed one descriptive fixed-grid number, which is not part of the ADOPT bar.

## What failed, as written (MEASURED, `propscore/full_{C1,C0,base}_read.json` controls; diagnosis `ctrls_diag.py`)
* CTRL-S ("the selection rule is the one we think") **FAILED as written on C1 and C0**:
  | arm | agreeing tokens | the one disagreeing token | gap between the planner's pick and the recomputed pick |
  |---|---|---|---|
  | C1 | 12,145 / 12,146 | `01e4bff700f15523` (pick 26, recompute 35) | 1.9e-7 |
  | C0 | 12,145 / 12,146 | `36e98388f7a35159` (pick 46, recompute 1) | 2.9e-8 |
  | base | 12,146 / 12,146 | none | none |
* CTRL-T and CTRL-P **PASS** on C1. Their maximum difference is 2.5e-11, below the 1e-9 bar.
* ⇒ **Per §3 as written, the S-RULE read is INVALID, with no verdict.** That run stays on disk unchanged, as
  `propscore/S_RULE_READ.json` / `.out`.

## Diagnosis
* The dumped logits are **float32**, and the planner selects in float32.
* Both read scripts cast the logits to float64 before the aggregate. At near-ties of 1e-7 and 3e-8, the extra precision flips the argmax.
* Recomputed in float32 with the same formula, the selection agrees with the dump on **12,146 / 12,146 tokens for all three arms**
  (36,438 tokens, 0 disagreements).
* ⇒ The selection rule IS navsim_v1. The control failed on the arithmetic dtype of the replication, not on the rule.

## The amendment (only this changes)
* `s_rule_read_a1.py` and `p5c_prop_read_a1.py` are the frozen scripts with one change each: the logits are cast to **float32** instead of
  float64 before the aggregate. The prop-read version also changes the matching cast inside `agg`.
* This makes the control **stricter in what it replicates**, because it now reproduces the planner's own arithmetic. It is not looser:
  CTRL-S still requires agreement on every token of every arm, with no tolerance.
* **Unchanged:** the TUNE/EVAL split, the 8-rule family, the choice rule and its tie-break, the estimator (paired log-cluster bootstrap,
  10,000 draws, seed 20260927), the three-part ADOPT bar, and the both-outcomes commitments of §3.

## Reporting, committed now
* **The verdict is the amended run's**, read from `propscore/S_RULE_READ_A1.json`.
* The as-written float64 run is reported beside it. It is labelled *CTRL-S failed, INVALID, no verdict*, and its Δ point and CI are
  shown for comparison.
* If the two runs would imply different verdicts, the report says so in its first line.
* The per-arm reads are re-run with `p5c_prop_read_a1.py` as `propscore/full_<arm>_read_a1.json`.
