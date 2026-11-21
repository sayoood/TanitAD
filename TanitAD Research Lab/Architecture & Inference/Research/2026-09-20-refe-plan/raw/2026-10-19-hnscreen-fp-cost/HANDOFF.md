# SUPERSEDED 2026-10-20 ~05:10 -- the pending runs all finished and the finish pass is done: read `FINISH_NOTES.md` and `RESULT.md` (headline + section 11). The text below is the hand-off as written on 2026-10-19 ~23:35.

# HANDOFF -- HNSCREEN-FP-COST (b'): the confirmation runs are pending (written 2026-10-19 ~23:35)

**State.** DONE and MEASURED: call map (`CALLMAP.md`), the (b') implementation (gate_token / launch_gate / hnscreen_devbox / hn_screen_read), 38 gate arms green, 9 chain arms green, reader suite 36 green (5 new mutants RED), reads per screen 20 -> 4
(`logs/reads_per_screen.json`), R7 differential 83/83 identical vs the tip. `RESULT.md` is written (sections 4 and 6 hold generated blocks). **Pending (about 3 h CPU, one python at a time):** the R7 control's final run, the chain mutation matrix
(10 mutants), the gate-level matrix re-run (2 fixed mutants), then the sibling suites MINE then BASE (`list_mine.txt` 31 invocations, `list_base.txt` 29).

**Why a handoff.** Length, and `C:` has had 0.1-0.7 GB free since ~23:10 (other writers; `hnfp3` is only 100 MB). Every step is gated on free RAM >= 3 GB AND free `C:` >= 2 GB; a step whose gate times out (20 min) is SKIPPED and gets no `.DONE`.

**The job (detached, WMI).** `scripts/run_rest.cmd` (log `logs/run_rest.out`; started PID 58924). IDEMPOTENT: steps = `logs/r7_control_run3.log.DONE`, `logs/test_hnscreen_ctx_chain_matrix_run1.log.DONE`, `logs/test_hn_ctx_anchor_matrix_run2.log.DONE`,
`suite_mine.json.DONE`, `suite_base.json.DONE`; the end marker is `logs/run_rest.ALLDONE`. The suite runner is resumable and records nothing for a file skipped for lack of disk (prints `SKIPPED_NO_DISK`). Already in `suite_mine.json`: `test_launch_gate.py --static --no-banked` 1168, `eval/test_hn_screen_read.py` 36.

**How to check.** `ls C:/Users/Admin/par18/hnfp3/logs/*.DONE C:/Users/Admin/par18/hnfp3/*.DONE`; `tail logs/runner_mine.out`; free space `Get-PSDrive C`. If `ALLDONE` is absent and no process has a command line containing `par18\hnfp3\scripts\run_rest.cmd` (Get-CimInstance Win32_Process), relaunch it:
`Invoke-CimMethod Win32_Process Create` with `cmd /c "C:\Users\Admin\par18\hnfp3\scripts\run_rest.cmd" > ...\logs\run_rest.out 2>&1` (CreateFlags 0x08000200). If `C:` is still under ~2 GB, STOP and tell the Master Mind: the steps cannot be trusted at that level.

**Exact next step (a fresh Sonnet agent, <= 20 min), once ALLDONE:**
1. `C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/par18/hnfp3/scripts/finish_pkg.py` -> writes `suite_compare.md`, `MUTANTS.md`, fills the two generated blocks of `RESULT.md`, rebuilds `out/` + `out/paths.json` (prints the sha256 of the 8 code files).
2. Read: `logs/test_hnscreen_ctx_chain_matrix_run1.log` and `logs/test_hn_ctx_anchor_matrix_run2.log` must end `ZZTEST_PASS` (34 + 38 and 10 + 9); `logs/r7_control.json` must read `ZZR7_CONTROL_PASS` (unmutated identical; mutants A and B DIFFERENT); the last line of `suite_compare.md`: `NEW FAILURES` and `LOWER COUNTS` both `none`.
   A `FAILS ON BOTH` row: read its `tail` in `suite_*.json`: a missing `raw/` or `taniteval/` path -> export it into BOTH trees with `scripts/export_extra.py <tip> raw/<dir>`; `No space left` -> re-run that file in both. A NEW FAILURE is a real finding: root-cause it first.
3. In `RESULT.md`: replace the headline's "Open at hand-off" bullet and section 4's "not yet re-run" sentence with the final numbers; replace section 6's PENDING sentence; confirm "1168 vs 1167" against the base run. Re-run `finish_pkg.py`.
4. Hand back to the Master Mind (<= 300 words; `SubagentHandback`): the manifest of section 10, `out/paths.json`. `rm -rf C:/Users/Admin/par18/hnfp3/tmp/*`. Keep `base/` and `work/` until landed.

**Constraints that still hold.** CPU only (`CUDA_VISIBLE_DEVICES=-1`); no commit / push / `git add` / register edit; never a pod, a GPU, `devbox_gpu.lock`, `devbox_gpu.chain_active`; kill only my own PIDs (explicit, no `/T`, no `pkill`); <= 2 python at once; nothing on D:.
**The code files must not change after this hand-off.** If one does, re-run `test_hn_ctx_anchor.py`, `test_hnscreen_ctx_chain.py`, `eval/test_hn_screen_read.py`, `scripts/r7_identity_vs_tip.py` and the suites that touch it.
