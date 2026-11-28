# Full-suite comparison of the per-tick code: run 1 (invalid) and run 2 (the one quoted)

Method. Each `stack/tests/test_*.py` of the patched tree runs in its own pytest process (`runner/run_files.py new`, log
`files_NEW.log`). Then the UNMODIFIED tip runs exactly the files that failed on the patched tree (`runner/chain_tip_and_compare.py`,
log `files_TIP.log`), and `runner/compare_suites.py` writes `COMPARE.txt`. A failure that appears only on the patched tree is a
regression (NEW-ONLY); exit 0 means NEW-ONLY is empty. Both trees are raw `cat-file --batch` exports of `stack/` and `taniteval/`
only, so tests that need data, the dev-box secrets or other folders fail on both sides.

## Run 1 (`files_NEW_run1_basetemp_parent_missing.log`, runner `runner/run_files.py.run1_basetemp_parent_missing`): INVALID
* pytest's `--basetemp` pointed into `pt_files_new/`, which never existed, so every test using `tmp_path` errored with
  FileNotFoundError: 326 of 728 files errored before running a test.
* The tip tree was 76a0d73 (stale), and the tip pass never ran (the waiter timed out), so there was no `files_TIP.log`.
* Nothing from run 1 is quoted except the 396 files that exited rc 0 or 5 (passed, or nothing collected). Those were carried
  into run 2's log as seed lines, and the r9 test files were dropped from the seed so they re-ran on the new tip.

## Run 2 (`files_NEW.log`, `files_TIP.log`, `COMPARE.txt`, `run2.out`): the comparison quoted
* Fix: `run_files.py` creates the `--basetemp` parent before the first file runs. MEASURED: `test_anchor_meta.py` went from
  errors to 15 passed, and FileNotFoundError appears 0 times in `files_NEW.log`.
* Both trees were rebuilt on the tip fe18377591d22fe532eb0ac63f0ad2d70d09b91a. The patched tree equals the tip plus exactly the
  5 modified and 2 new files of this commit (tree diff checked before the run).
* Patched tree: 729 files (the tip's 728 plus `test_v8f_recover_pertick.py`), each exactly once: 12,917 passed, 266 failed,
  52 errors, 1,637 skipped. 34 files exited with rc not in (0, 5).
* Unmodified tip on those 34 files: 656 passed, 266 failed, 52 errors, the same counts as the patched tree on the same files.
* COMPARE.txt: NEW-ONLY 0, TIP-ONLY 0, BOTH (failing on both trees before this change) 318; exit 0. Run 2 started 01:34:15
  and ended 02:47:54 on 2026-11-05 (CHAIN-DONE 0).
* The changed and new test files on the patched tree: `test_v8f_recover_pertick` 32 passed, `test_v8f_recover_wiring` 53,
  `test_v8f_launch_intent` 36, `test_v8f_ego_dagger_compose` 36, `test_v8f_recover` 16, `test_v8f_fast_table` 21 + 1 skipped.
