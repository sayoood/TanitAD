# Full-suite comparison of lever (a) `--level-head` / `--w-level`: patched tree vs the tip

Method (the pattern of eccdabd). Each `stack/tests/test_*.py` of the patched tree runs in its own pytest process (`runner/run_files.py new`, log
`files_NEW.log`, basetemp parent created first). Then the UNMODIFIED tip runs exactly the files that failed on the patched tree
(`runner/chain_tip_and_compare.py`, log `files_TIP.log`), and `runner/compare_suites.py` writes `COMPARE.txt`. A failure only on the patched tree is a
regression (NEW-ONLY); exit 0 means NEW-ONLY is empty and both logs reached ALL-DONE. Both trees are raw `cat-file --batch` exports of `stack/` and
`taniteval/` only (tip b07982c0dc7ebce6815badde6ca02d9aff5f09c0), so tests that need data or other folders fail on both sides.

The patched tree equals the tip plus exactly these files (tree diff checked by sha256 before the run): edited `stack/tanitad/models/v8f_mini.py`,
`stack/scripts/v8f_mini_train.py`, `stack/tests/test_v8f_launch_intent.py`; new `stack/tests/test_v8f_level_head.py`, `stack/tests/v8f_mini_tip_snapshot_level.txt`.

Started 2026-11-05 13:00 on the dev box, CPU only, detached via WMI (`runner/run_suite.cmd`); about 75 to 85 minutes for the patched tree and a few minutes
more for the tip pass. Done marker: `SUITE_DONE.marker`. Read the result with `python runner/compare_suites.py` (NEW-ONLY must be empty).
