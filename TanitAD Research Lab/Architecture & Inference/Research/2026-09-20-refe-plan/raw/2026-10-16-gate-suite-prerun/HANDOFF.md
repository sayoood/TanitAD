# HANDOFF: G-SUITE pre-run (suite-prerun / run --suite-evidence), 2026-10-16

PK = `C:/Users/Admin/AppData/Local/Temp/gpre/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan` (scratch extraction of tip 8f7fedf; SINGLE COPY until landed). Package dir = `PK/raw/2026-10-16-gate-suite-prerun/` (call it R).

## State
* **Code finished and tested** (nothing committed, nothing pushed, no `*_pod.sh` touched):
  * `PK/refe/gate_token.py` (+163 lines, 0 removed): evidence primitives. `PK/refe/launch_gate.py` (+87/-6): `suite_check`, `cmd_suite_prerun`, `--suite-evidence`. `gate_checks.py` unchanged.
  * `PK/refe/test_gate_suite_prerun.py` (new, in no profile's SUITE): 63 regression/control arms + 26-mutation source matrix. First complete runs on the dev copy: arms 63/63 as expected (92 asserts); matrix 26/26 RED (52 asserts).
* **Wiring = separate step:** `R/WIRING.md` (3-line change, per-chain start points, binding table, the pod RAM/CPU escalation).
* **A detached job (WMI, pid 29188 at launch) is producing the numbers.** Stages in `R/logs/finish.status`: A = the 13 suite scripts on the changed tree -> `R/baseline/scripts_after/` (BEFORE = `R/baseline/scripts_before/`, already complete: 12 green, `test_rl_s1b` rc 1); B = the new test in full + `R/arm_report.json`; C = real-profile measurement `R/measure_p2rep/`; D = `R/tables/*.txt` (`ALL.txt`). Expected ~30-40 min on this shared box. Markers: `R/DONE_GATE_SUITE_PRERUN` or `R/FAILED_GATE_SUITE_PRERUN`.
* **Pre-existing RED, not mine:** `test_rl_s1b.py` fails on the unmodified tip (stale anchor after S1c; its snippet occurs 0 times in `launch_gate.py`), so G-SUITE of `refe_rls1b`/`refe_rls1c` would FAIL today. Verified fix: `R/test_rl_s1b.anchor_fix.diff` (then `ZZTEST_PASS 18`, `R/logs/test_rl_s1b_with_anchor_fix.log`). Not applied.

## Next step (exact)
1. `cat R/logs/finish.status` and `ls R/DONE_GATE_SUITE_PRERUN`. If neither marker exists and `finish.status` last line is old, the job died: re-run `bash R/finish_all.sh` (detached via `Invoke-CimMethod Win32_Process Create`, see memory note devbox-drive-letter-and-detached-jobs).
2. Read `R/tables/ALL.txt`. Acceptance for "nothing that passes today may fail": every row of `tests_before_after.txt` says `same` (`test_rl_s1b` is `same` = RED both times); `regression_arms.txt` ends `expected != observed: none`; `mutation_matrix.txt` ends with a `ZZTEST_PASS` token; `files.txt` says the shipped tree is UNCHANGED.
3. Report to the Master Mind (headline = integration: land the 3 files, re-ship; then wire per `WIRING.md`): the P2-s1 anchor (22 of 40 idle minutes, MEASURED) -> ~21.8 min saved per P2-s1-class arm (ESTIMATED), `tables/savings_estimate.txt` for other profiles (ESTIMATED), `tables/measure_p2rep.txt` for the peak RSS/threads of the pre-run (the number to compare with the pod's free RAM before wiring).
4. NOT to do: no `*_pod.sh` edit, no commit/push (Operating Standard rule 1).

## Facts a fresh agent should not re-derive
* The evidence binds: signature, commit, tree digest, profile, suite entries, gate code (3 files), age <= 12 h, no regression arm, PASS/0 skips re-judged from rows, plus (added by me) the environment. A mismatch = the suite runs LIVE; the token's `checks["G-SUITE"]["suite_source"]` says which path and why.
* The harness blocks subagents from writing report `.md` files, so there is no REPORT.md: the report is the final message plus `R/tables/`.
