# PREREG_V8F_N1DH, Amendment 1 (FROZEN 2026-11-21 02:16Z by the MM, before any N1DH run, dump or number)

The card `PREREG_V8F_N1DH.md` (sha256 f1df05bc) stands. This amendment changes only the commit of record, by one test file.

## What was found (MEASURED, pod, about 02:15Z)
The lesson of the ROLLHIST G-SUITE stop the same night was to run every new test file on the pod tree before a pod launch. I ran N1DH's three new test files on `/root/strat1/n1dh/tree` (cc8c98a1):
* **1 failed**, 138 passed, 1 skipped.
* The failing test is `test_v8f_n1dh.py::test_every_stack_script_that_builds_the_v8f_config_from_a_checkpoint_is_in_the_loader_list`.
* `stack/scripts/n1dh_klim.py`, which is in the commit, builds `V8FMiniConfig(**ck["cfg"])` but is not in the test's `LOADERS`.

This is not a platform effect: the git grep at cc8c98a1 finds 12 matching scripts against 11 in `LOADERS`. The N1DH chain's G-SUITE (PASS needs 0 new-test failures) would have stopped on it after ROLLHIST, before any GPU step.

## The change
* **Commit f2e91e33fbe69586d7e2c464f3c1eda7ac7ddcfc**, branch `agent/v8f-n1dh-20261120`, parent cc8c98a1, pushed, with `ls-remote` equal.
* It changes exactly one file: `stack/tests/test_v8f_n1dh.py` gains `"n1dh_klim.py"` in `LOADERS` (new sha256 b8fd9a94..., git blob 3792022e). The parametrized round trip now covers that loader too.
* On a pod tree copy carrying the change:
  * the 3 new test files: **140 passed**, 1 skipped;
  * the 5 modified test files: **203 passed**, 1 skipped.
* **The code of record is now f2e91e33.** Model, trainer, checker and evaluators are byte-identical to cc8c98a1.

## Pod repoint (before any N1DH launch)
* `/root/strat1/n1dh/tree` gets the new test file.
* The `n1dh_tree.sha256` and `all_blobs.txt` lines for that file change.
* `COMMIT` changes in `n1dh_arm.sh` and `n1dh_gsuite.sh`.
* Copies of the old files are kept as `*.bcc8c`.
* Nothing in the arms, steps, dumps, reads or bars changes.
