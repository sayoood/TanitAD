# IMPL_ADDENDUM2_CAPLADDER (Master Mind, 2026-11-08, before any result file exists)
Run 1 died at 22:24 on 2026-11-07 with `forrtl: error (200): program aborting due to window-CLOSE event` in its Gram stage; the dev box rebooted at 23:59:39. R0_A, R0_D and G_655360_A were written (rungs/, 21:55-22:02). No RESULT file and no statistic were read by anyone.
Resume changes, code only, no statistic or rule changes:
1. `run_capladder.py`: `have_r0` also holds when `rungs/R0_A.npz` exists. The job deletes `r0_fp16.bin` once R0's Grams are written, so without this a resume would list the finished R0 rung as NOT RUN.
2. `job_resume_capladder.cmd`: memory gate, then run, report, figures, cleanup. It skips the frame pull, the R0 encode and the features, whose outputs exist (`capladder_feat`, control 4 passed in run 1). Log `run_capladder.run2.log`. Launched with process flags 0x08000200 (no window, new process group).
