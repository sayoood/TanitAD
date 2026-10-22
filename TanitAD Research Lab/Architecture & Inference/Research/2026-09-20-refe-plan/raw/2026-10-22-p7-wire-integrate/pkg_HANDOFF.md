# HANDOFF -- P7-WIRE-INTEGRATE (<= 2k tokens; written 2026-10-22 ~22:50 because the verification runs take ~20-30 min of CPU)

**State.** Code, tests, docs and the pin procedure are COMPLETE and staged at repo paths under `C:/Users/Admin/par18/p7int/code/` (nothing committed/pushed/staged in any git tree; local override). Only the before/after suite runs and the mutation run are still going.
Package root: `C:/Users/Admin/par18/p7int/` (single location). Tip used: `a8f2c89`. Edit tree: `work/<PK>/refe/` (pristine: `base_full/`); authoritative output: `code/` + `code/P7_WIRE_INTEGRATE.diff` + `code/MANIFEST.txt`.

**Measured so far (CPU).** `test_p7_integration.py` `ZZP7INT_PASS 139` (~25 s); `test_p7_head_wiring.py --static` `ZZP7HEAD_PASS 181` (179 before); BEFORE `test_launch_gate` `ZZLG_STATIC_PASS 1168` rc 0; the defect (G-RESERVE FAILS a clean tree against the real module) in `raw/probes/probe_before.out`; the pin procedure's literal CLI commands in `raw/rehearsal/rehearse.out`.

**Running (started 22:31-22:36):** `run_suites_int.sh` lanes A (`test_p6_chain`) and B (`test_launch_gate`, `test_p6_levers --fast`, `test_session_reserve`, `test_p7_head_wiring --full`, `test_p7_integration`) for BEFORE (`raw/before/`) and AFTER (`raw/after/`), and `test_p7_integration.py --mutations` (`raw/mutations.out`, marker `raw/_MUT_DONE`). A watcher (`finalize_int.py`, started in the background) waits for all five markers, then writes `raw/VERIFICATION.md`, regenerates `RESULT.md` from `RESULT.template.md`, re-stages `code/`, and writes `raw/_FINISHED` (first line = the verdict).

**Exact next step (a fresh Sonnet agent is enough):**
1. `until [ -f C:/Users/Admin/par18/p7int/raw/_FINISHED ]; do sleep 30; done; cat C:/Users/Admin/par18/p7int/raw/_FINISHED` (a script, not a model loop). If the watcher died: `python C:/Users/Admin/par18/p7int/finalize_int.py`.
2. The verdict must read GREEN. If NOT GREEN: open `raw/after/after_<suite>.out` vs `raw/before/before_<suite>.out`. Known environmental noise: other agents share the box (a timeout/lock flake in `test_p6_chain`); re-run that one suite alone with `bash run_suites_int.sh <tree> raw/after after A`. A mutation that is NOT RED means a missing arm in `test_p7_integration.py`: add the arm (do NOT weaken the mutation), re-run `python test_p7_integration.py` and `--mutations`, then `python stage_code.py`.
3. `code/MANIFEST.txt` must show `0` CR-lines for every file. Then `SubagentHandback` (<= 300 words): headline = `RESULT.md` "HEADLINE" items 1-5 + the manifest + "single location".

**Do not:** touch any git tree; edit `train.py`, `session_reserve.py`, `p6_pod.sh`, `p6_prep.py`, the `refe_p6` profile; register the two pins (`P7_BANK_TUPLES`, `P7_AUX_STORE_PINS`): they are PENDING the rebuilt 3-C bank (`PROCEDURE_P7_PINS.md`); touch the pod.
**Backslashes in bash heredocs are halved by this harness**: write files with the Write tool, never `\n`-bearing heredoc patches.
