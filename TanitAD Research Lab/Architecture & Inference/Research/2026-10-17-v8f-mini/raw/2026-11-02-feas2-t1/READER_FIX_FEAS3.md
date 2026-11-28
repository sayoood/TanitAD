# READER_FIX_FEAS3: import-path fix of `code/feas3_t1_read.py` (2026-11-02, Master Mind)

**What failed.** The FEAS3 chain's 12 rollouts finished (nfail 0, 18:16:16); the read stopped at rc 1 at 18:18:57 inside
`disclosures()` (the card's pre-registered beside checks: M1 at 1 / 2 / 3 m/s margins and the hard-braking windows), with
`taniteval.stack_guard.StackShadowError`: `lever()` had put the arm tree's stack root (`F:/TanitAD-scratch/feas2_t1/tree_dag/stack`)
on `sys.path` and imported `tanitad.refs.feasible_decode` from it, so a second `tanitad` root preceded the pinned one
(`dagt1_tree_b43c7a9a/stack`) when `rr_lib` imported `taniteval`. The guard refused, as designed. `logs/read_feas3.out` holds the
traceback and the per-cell outcome lines the read had already printed before the stop (the Master Mind saw those lines before
this fix).

**The fix (path only).** `lever()` now loads `tree_dag/stack/tanitad/refs/feasible_decode.py` by file path
(`importlib.util.spec_from_file_location`) instead of adding the arm tree's stack root to `sys.path`. That file imports only
`math` and `torch`, and it is byte-identical to `dagt1_tree_b43c7a9a/stack/tanitad/refs/feasible_decode.py` (`diff -q`: no
difference), so the operator and every number it feeds are unchanged. Pre-fix reader kept at
`F:/TanitAD-scratch/feas2_t1/feas3_t1_read.pre_pathfix.py`.

**What this does not change.** No row, threshold, arm, control, outcome rule or bootstrap setting; the rollouts are not re-run.
The read is re-run once on the same outputs (`logs/read_feas3_rerun.out`); its per-cell lines must equal the first run's lines
(checked by `diff` of those lines, result appended below).

**Result (appended 2026-11-02T18:21:56).** The re-run read exited rc 0; its per-cell outcome lines are identical (12 of 12 per-cell lines) to the first run's; OUTCOME: every arm HARMS. The chain's `raw/FEAS3.fail` is kept as `raw/FEAS3.fail.first_read`; `raw/FEAS3.done` marks the re-run.
