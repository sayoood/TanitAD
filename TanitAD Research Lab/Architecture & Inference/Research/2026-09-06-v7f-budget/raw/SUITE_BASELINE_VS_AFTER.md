# The suite, as a CONTROLLED comparison

**2026-09-06 · ArchInf FlyWheel.** ⛔ A bare *"suite green"* is inadmissible when the baseline is
not green — **this baseline is not green** — so the comparison is run against a **pristine tree**,
not remembered, and the verdict is the **failure-ID SET DIFF**, not the counts.

## The two trees

| | tree | state |
|---|---|---|
| **BASELINE** | `C:\Users\Admin\tanitad-v7fbase` | a clean copy, md5-verified byte-identical to the branch worktree: `train_v6_staged.py` `d2ade650…` · `v6.py` `09cb5346…` · `predictor.py` `6656e296…`, and **no `_gradreach.py`** |
| **AFTER** | `C:\Users\Admin\tanitad-v7fbudget` | the same copy with this turn's edits |

Identical invocation in both: `cd stack && PYTHONPATH=<tree>/stack OMP_NUM_THREADS=4
PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider --continue-on-collection-errors`.

⚠️ **A separate, earlier "after" run was DISCARDED and re-run.** Its baseline had been started in
the *same* tree I then edited, so tests spawning subprocesses could have picked up new code
mid-run. A contaminated control is not a control.

## The result

| | failed | passed | skipped | xfailed | errors | wall |
|---|---:|---:|---:|---:|---:|---:|
| **BASELINE** | **65** | 6,357 | 121 | 2 | **51** | 768 s |
| **AFTER** | **65** | **6,374** | 121 | 2 | **51** | 725 s |

* ⭐ **Failure-ID set diff: EMPTY IN BOTH DIRECTIONS.** 116 distinct FAILED/ERROR ids in each run;
  `comm -13` (new failures) and `comm -23` (newly fixed) are both empty. **Zero regressions.**
* ⭐ **The +17 passed is accounted for EXACTLY**: `tests/test_grad_budget_honesty.py` collects
  **17** tests. `test_v6_ladder_edges.py` collects **33** in both trees and `test_v6_staged.py`
  **86** in both — so the two tests I rewrote were *modified, not added or dropped*, and no test
  was quietly deleted to make the numbers work.

⚠️ **These absolute numbers are NOT the same baseline the brief quotes** (`9 failed / 1187 passed
/ 38 skipped / 24 errors`). Both trees here are `stack/`-only copies without most repo-root data
(`Project Steering/`, artifacts, runbooks, `.git` hooks), so a large block of tests
(`test_runbook_commands`, `test_decision_check`, `test_render_quality_alignment_gate`,
`test_build_parity_guard`, `test_eval_contamination`, `test_secret_scan`, …) fails or errors for
**environment** reasons in *both* columns and cancels out of the diff. The comparison is
**tree-internal and like-for-like**, which is what the set diff needs; it is not a claim about the
repo's absolute suite state.

## ⛔ Five regressions were found by this comparison, and every one was FIXED, not reported

The first "after" run showed **+5 failures**. They were real, and they were all one mistake:
declaring `predictor_op.out_proj` grad-unreachable in `OperativePredictor.__init__` rather than in
`V6Stack.__init__`.

| test | what it caught |
|---|---|
| `test_train_flagship_v4.py::test_from_scratch_trunk_is_random_and_passes_not_frozen_gate` | ⛔ **`train_flagship_v4.py:1636`'s not-frozen gate REFUSED THE LAUNCH** — *"TRUNK FROZEN — Sayed's hard requirement is that the encoder AND predictor train jointly (NO frozen part)"*, `"trunk_tensors_frozen": 4` |
| `test_v5_trainer_v2_val.py` ×3 | the same `requires_grad`-as-proxy predicate, in the v5 val-provider gates |
| `test_v6_staged.py::test_stage_freeze_trains_exactly_the_declared_groups` | in scope: the same `requires_grad == (group in want)` predicate the ladder-edges test had |

⇒ the two declarations moved to `V6Stack.__init__` (the consumer whose budget was wrong), and the
two V6Stack-scoped tests had their **predicate** corrected — to *"in a trained group **AND** not
declared unreachable"* — each with a non-empty-declared-set assertion so the relaxation cannot go
vacuous, and one of them rewritten to **execute** `assert_resume_lineage` rather than pin a
coincidence that this change ended. ⭐ **Every headline number in `RESULT.md` was re-measured after
the move and is unchanged.**

⚠️ The flagship-v4 and v5 gates were left **untouched**: their intent is satisfied (a tensor no
gradient reaches was never training) but their predicate is this turn's defect in miniature, and
correcting a **PI hard requirement** is not a v7f-budget deliverable. It is logged as a work item
for their owners, with the one-line fix named.
