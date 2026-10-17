# HANDOFF -- SFT-4R readiness, written 2026-10-16 ~23:25 (token-budget rule: a job > 4 min => handoff, not a waiting agent)

## State
* DONE and MEASURED (see `READINESS.md`): runaway root-caused (`test_sft4r_gate.py:286` -> `launch_gate.py run` -> G-SUITE -> the same test; `REFE_GATE_NESTED` was set, never read), fixed in
  `gate_checks.check_suite` (refuses to nest) and in the test call (`--only`), pinned by `refe/test_suite_nesting.py` (88 checks, 5 bounded mutation arms RED), real `refe_sft4r` G-SUITE PASS in 59 s,
  patchers + `apply_all.py` verified byte-exact against a fresh `git archive` of tip `6c3d469c`, `AMENDMENT_3_DRAFT.md` written.
* RUNNING, detached (WMI, survives the session): `C:/s4r/tools/run_all.py --workers 4`, PID in `C:/s4r/queue_runner.pid` (32204), started 23:11:23. 23 tests x forms + the e2e `suite-prerun`, every run
  under `bounded_run.py` (timeout, <= 20 live descendants, kills by explicit PID). Banks one line per run to `C:/s4r/SUMMARY.txt`. ETA ~25-40 min from start (`test_launch_gate--full` is the long pole, timeout 50 min).
* WHEN IT FINISHES it writes `C:/s4r/TEST_TABLE.md` + `C:/s4r/queue_done.flag` ("<pairs> <not-clean>") and itself runs `tools/build_package.py --results-only`, which copies the logs into this package and
  FILLS the table into `READINESS.md` between `<!--TEST_TABLE_BEGIN/END-->`.

## Exact next step (a FRESH Sonnet agent, <= 100k context; do not wait on the queue yourself)
1. Wait by script, not by turns: Monitor / `until [ -f C:/s4r/queue_done.flag ]; do sleep 30; done`.
2. `python C:/s4r/tools/verify_post.py` must print `ZZVERIFY_POST_OK` (no test rewrote a deliverable in the tested tree; was OK at 23:17 before the queue's long tests ran).
   `cat C:/s4r/queue_done.flag` -> `<pairs> <not-clean>`. Read `C:/s4r/TEST_TABLE.md`. Every row must be PASS, PASS-ON-SERIAL-RERUN, or PRE-EXISTING.
   Known PRE-EXISTING already: `test_hf_egress` (the `driverl-eval` venv lacks `httpx`; passes 13/13 in the `tanitad` venv; `logs/manual/test_hf_egress_httpx_check.txt`).
3. Any **REGRESSION** or **NEW-TEST-FAILS** row: read `C:/s4r/logs/q/<test>.patched.<form>.txt`, fix in `C:/s4r/p/...` via a NEW `patch/` script (never hand-edit without one), rebuild
   (`python C:/s4r/tools/build_package.py`, then re-verify `apply_all.py` on a fresh `git archive` of the tip with `tools/treediff.py`: n_diff must be 0), re-run just that test:
   `python C:/s4r/tools/run_all.py --only <id>`. NOTE: `--only` rewrites `C:/s4r/SUMMARY.txt` and `TEST_TABLE.md`; copy the full ones aside first.
4. Confirm READINESS.md section 6 now holds the generated table (not the "RUNNING" text). If the queue died: `Get-CimInstance Win32_Process -Filter "ProcessId=<pid>"`; the lines already in SUMMARY.txt
   are valid; re-run the missing ids with `--only`.
5. Hand back to the Master Mind (<= 300 words): recursion root cause (file:line), the fix, the test table (pass/fail, pre-existing vs regression), and the package path (below).

## Rules that bind this work
* Kill ONLY by explicit PID (`taskkill //PID <pid> //T //F`); other agents' python processes (GPU battery, `p6_levers`, ...) run on this box. Never by image name or pattern.
* Never stage, commit or push. Never touch the pod. Do not print the pod address or any token. Do not edit `Project Steering/Mission Plan.md`.
* The package is NOT staged; the Master Mind lands it. Package: `E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-sft4r-readiness/`.
  Working copies: `C:/s4r/` (`p` = patched tested tree, `b` = unpatched tip export, `tools/`, `logs/`).
* The tip may move: `apply_all.py` refuses on a moved pre-image (use `--patchers-only`, then compare against `diff/`).
