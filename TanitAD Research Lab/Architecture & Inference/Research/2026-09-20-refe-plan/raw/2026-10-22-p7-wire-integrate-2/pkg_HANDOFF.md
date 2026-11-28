# HANDOFF -- P7-WIRE-INTEGRATE-2 (<= 2k tokens; written 2026-10-22 ~23:45; the verification runs take ~30-40 min of CPU)

**State.** Code, tests, docs, probes and the RESULT narrative are COMPLETE and final in `C:/Users/Admin/par18/p7int2/work/<PK>/` (PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`). Nothing is committed/pushed/staged in any git tree (local override).
Single location: this directory. Tip used: `a8f2c89`. p7int's package (`C:/Users/Admin/par18/p7int/`) is untouched.

**Done and MEASURED (CPU).**
* Gap 1: M2 is NOT equivalent (probe: `raw/probes/probe_m2_original.out` vs `probe_m2_mutated_M2.out`); +23 assertions in `test_p7_integration.py` section I.
* Gap 2: `P7_TARGETS_SHA256` (profile `expect_schedule["targets_sha256_by_bank"]`) + `p7_refusals` + G-CLOCK content check + `p7_pins.py` third line + `p7_pins.py --check-run`; `test_p7_integration.py` `ZZP7INT_PASS 220`; static head test `ZZP7HEAD_PASS 182`.
* A TRIAL mutation run on the pre-final code: 25 mutants, ALL RED, control PASS, `ZZP7INT_MUTATIONS_OK` (`raw/mutations_try1.out`). The final code adds only section 0 and the try/except in `main()` (3 assertions): the run below re-proves it.
* Probes: `probe_digest_vs_trainer.out` (gate digest == train.run_identity's, ALL EQUAL), `probe_real_build_clock.out` (REACHABLE through the real BUILD), `probe_p6_identity_{tip,overlay}.out` (identical).

**Running (launched 23:31 via WMI, detached):** `jobs/{before_A,before_B,after_A,after_B,mutations}.bat` -> `raw/before/`, `raw/after/`, `raw/mutations.out`; markers `raw/before/_before_{A,B}_DONE`, `raw/after/_after_{A,B}_DONE`, `raw/_MUT_DONE`.
Trees under test: `before_full/<PK>` (pristine tip), `after_full/<PK>` (snapshot of `work/`, == the code to land).
The zero-token finaliser `finalize2.py` was started detached right after: it waits for the five markers, writes `raw/VERIFICATION.md`, builds `RESULT.md` from `RESULT.template.md`, runs `stage_code2.py` (-> `code/`, both diffs, `MANIFEST.txt`) and writes `raw/_FINISHED` (first line = the verdict).

**Exact next step (a fresh Sonnet agent is enough):**
1. `until [ -f /c/Users/Admin/par18/p7int2/raw/_FINISHED ]; do sleep 30; done; cat /c/Users/Admin/par18/p7int2/raw/_FINISHED` (a script, not a model loop). If the watcher died: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe C:/Users/Admin/par18/p7int2/finalize2.py` (waits itself; `--no-wait` if the markers exist).
2. The verdict must read GREEN. If NOT GREEN: compare `raw/after/after_<suite>.out` with `raw/before/before_<suite>.out`. Known environmental noise: other agents share the box (a timeout/lock flake in `test_p6_chain`); re-run ONE lane alone: `jobs/after_A.bat` via `cmd /c "<bat>" < NUL` (launch detached with WMI `Invoke-CimMethod Win32_Process Create`; a bash `nohup &` dies with its tool call).
   A mutant that is NOT RED means a missing arm in `test_p7_integration.py`: add the arm (never weaken the mutant) in `work/`, re-run `python make_jobs.py --snap`, relaunch `after_B` + `mutations`, then `finalize2.py`.
3. `code/MANIFEST.txt` must show `0` CR bytes for every file. Then `SubagentHandback` (<= 300 words): headline = `RESULT.md` HEADLINE items 1-6 + the manifest + "single location".

**Do not:** touch any git tree; edit `train.py`, `model.py`, `session_reserve.py`, `p6_pod.sh`, `p6_prep.py`, the `refe_p6` profile; register the three pins (PENDING the rebuilt 3-C bank, `PROCEDURE_P7_PINS.md`); touch the pod.
**Harness traps seen:** a bash heredoc halves backslashes (`\\n` -> a real newline inside the written file): write Python patch files with the Write tool or use Edit; `grep -c $'\r'` counts every line (count CR bytes with python); `nohup ... &` jobs die with the tool call.
