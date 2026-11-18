# HANDOFF TFAR build (2026-11-18, build agent, ended at the context budget)

## State
* Card FROZEN, sha256 `816d4357...` verified at start. Tree: `F:/TanitAD-scratch/strat1/tfar/tree/` = `git archive aedf8716 stack taniteval` (LF) + my edits. Nothing committed, nothing launched, no GPU used.
* Built in the tree (see `BUILD_TFAR.md` for the manifest):
  model `stack/tanitad/models/v8f_mini.py` (e_ck_far, e_nav_s_t, set_t_far_eval real/zero/set/oracle), trainer `stack/scripts/v8f_mini_train.py` (7 flags, G-DVB `bad_tfar`, optimiser groups tfar/navst, `tfar_probe`, `param_names` in a lever-on checkpoint),
  `tfar_check.py` (token), `tfar_reach.py` (K-REACH-e), `v8f_mini_eval_tfar.py`, `v8f_t1_eval_tfar.py`, 4 test files + the aedf8716 module snapshot.
* Tests: `test_v8f_tfar.py`, `test_tfar_check.py`, `test_tfar_reach.py`, `test_tfar_eval.py`. The killed combined run `tmp/new3.log` printed 138 passes and no failure before it was stopped for the regression driver (box loaded, F: I/O slow).
* Pod kit staged in `tfar/pod/` (tfar_gsuite.sh, tfar_chain.sh, queue_tfar.sh), `bash -n` ok, COMMIT / NFILES / PRED_LOG placeholders unset on purpose.

## RUNNING (detached, started 2026-11-18 via WMI)
The regression driver (`tmp/regress.cmd` -> `tmp/regress.py`, 145 files, the 4 new test files first), WMI pid 33328. Driver log `tmp/regress_driver.log`; summary `tmp/regress_summary.json`; last line `ZZ REGRESS DONE n_regressions N new_file_failures M`. `tfar.patch` and `pod/tfar_tree.sha256` were already regenerated (12 files). `BUILD_TFAR.md` is written; section 5 awaits the regression result.

## NEXT (exact)
1. When `tmp/regress_driver.log` ends with `ZZ REGRESS DONE`, read `tmp/regress_summary.json`. Bar: n_regressions == 0 and new_file_failures == 0 (a failure that also fails on the aedf8716 base is a base failure, listed). A defect in the 4 new files shows after the first file: fix in `tree/` only, re-run `tmp/changed.py` and `tmp/mkpatch.py`, re-run the affected file with `tmp/run1.sh <log> tests/<file>`.
2. Put the result into `BUILD_TFAR.md` section 5 (and refresh the patch if anything changed).
5. Delete own temp by literal path and log it here: `tmp/bt`, `tmp/bt_*`, `tmp/t`, `tmp/pa`, `tmp/reg_base`, `tmp/junit_*.xml`, `tmp/regress_*.log` (keep `regress_summary.json`, `regress.py`, `changed.py`, `mkpatch.py`), `tmp/patch_model.py`, `tmp/patch_trainer.py`, `tmp/ref_*.py`, `tmp/v8f_mini_ref.py`, `tmp/new*.log`.

## Deletions so far
`F:/TanitAD-scratch/strat1/tfar/tar/t.tar` (465,776,640 B, a re-creatable git archive of aedf8716), logged in `tmp/deletions.log`.
