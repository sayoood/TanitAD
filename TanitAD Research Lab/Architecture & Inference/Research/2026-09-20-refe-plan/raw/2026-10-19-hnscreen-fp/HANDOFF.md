# HANDOFF -- HNSCREEN-FP-REQUIRED, sibling-test confirmation still running (2026-10-19)

State: the fix, the helper, the tests, the FULL before/after and the mutation REDs are DONE and staged (see RESULT.md). One item is open: the measured "0 new failures" over the 21 sibling test files that mention `launch_gate`.

Why it is a handoff: it needs > 1 h of CPU (`test_gate_suite_prerun.py` alone took 977 s on the baseline). A detached job is running it; no agent waits on it.

Job: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/par18/hnfp/suite_finish.py` (started 2026-10-19 ~08:45, log `par18/hnfp/out_suite_finish.log`). One child python at a time, `CUDA_VISIBLE_DEVICES=-1`, RAM-gated (>= 2 GB free). It is RESUMABLE: it skips any file already in `suite_after.json` / `suite_base.json`.
* runs all 21 files on the FIXED tree `par18/hnfp/work/refe` -> `suite_after.json`
* runs the ORIGINAL tree `par18/hnfp/base/refe` only for files that FAIL on the fixed tree (5 baselines already exist in `suite_base.json`) -> tells NEW from PRE-EXISTING
* then writes `par18/hnfp/suite_compare.md` and the marker `par18/hnfp/suite_finish.DONE`.

How to check: `ls C:/Users/Admin/par18/hnfp/suite_finish.DONE`; `cat C:/Users/Admin/par18/hnfp/out_suite_finish.log`. If the marker is absent and no `suite_finish.py` process is alive (`tasklist` / psutil on the cmdline, match the exact script name), re-run the same command: it resumes. If it died on a hang, run the missing file by hand: `cd par18/hnfp/work/refe && CUDA_VISIBLE_DEVICES=-1 python <file>`.

Exact next step (a fresh Sonnet agent, <= 15 min): read `suite_compare.md`; for each row with "after FAILS" read the `tail` in `suite_after.json` and decide NEW (cause: name the line) vs PRE-EXISTING; copy `suite_compare.md`, `suite_after.json`, `suite_base.json` into `raw/2026-10-19-hnscreen-fp/` (and add them to `paths.json`); replace section 5 of RESULT.md with the table and one sentence ("0 new failures" or each new failure with its cause); update the sha256 table. No git add / commit / push.

Constraints that still hold: do not touch any lock, the GPU, the pod, or `par18/{l1run,probe,r0eb}`; never run the full stack suite; at most 2 python processes.

Known non-passes to keep in the report (baseline, not caused by this change): `test_aux_g12_bf16.py` SKIPS (no CUDA: the bf16 G12 probe did not run); `test_gate_gpu_probes.py` prints `SKIP_CUDA: the GPU check did NOT run`.
