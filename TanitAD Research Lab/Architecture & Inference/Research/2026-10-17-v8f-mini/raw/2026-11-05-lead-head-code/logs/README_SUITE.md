Full-suite comparison for the lead head (patched tree `new` vs the unmodified tip 219f6c7), one pytest process per test file.
`files_NEW.log`: every test file of the patched tree; `files_TIP.log`: the SAME files that failed on NEW, run on the tip export; `COMPARE.txt`: NEW-ONLY must be empty (exit 0).
Runner: `runner/run_suite.cmd` (detached via WMI, CUDA_VISIBLE_DEVICES=-1, one process at a time); done when `SUITE_DONE.marker` exists.
