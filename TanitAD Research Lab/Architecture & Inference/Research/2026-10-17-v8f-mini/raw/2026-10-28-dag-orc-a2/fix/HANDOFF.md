# HANDOFF (compose_fix) — written 2026-10-28 ~02:50

STATE. The fix, the new tests and the harness are DONE and verified on the pieces that were run by hand:
new tests GREEN on the fix (10/10; 3 passed in an interim run after one test-bug fix, then the full selection), RED on e7ea20d1 (10/10 failed, substantive numbers in NOTES.md), the flag-off golden test + the anchors test + the composition test GREEN.
What is still RUNNING (three WMI-parented pipelines, started ~02:36, survive this agent; ~60-80 min total):

| pipeline | what | end marker |
|---|---|---|
| A `tools/pipe_suites.sh` | RED-on-old rerun with the final test file, GREEN rerun, then the 7 suites one at a time (compose, ego_perturb, dagger_train, dagger, init_ckpt_o_interface, launch_gate_v8f, orc_chain) | `F:/TanitAD-scratch/compose_fix/logs/suites.done` ; per-file `logs/suite_<file>.log`, progress `logs/*.progress` |
| B `tools/pipe_mut.sh` | mutation sweep, 27 mutants (19 old + 8 new `M17a-h`) + 11 controls, on the copy `F:/TanitAD-scratch/compose_fix/mut` | `logs/mut.done` ; output `mutation.txt` (this dir) |
| C `tools/pipe_final.sh` | waits for A and B, then `assemble.py` writes `RESULTS.md`, refreshes `stage/`, `diff.patch`, `SHA256SUMS.txt`, `logs/` | **`FINISHED.txt` in this dir** |

NEXT STEP (a fresh Haiku-class agent). Wait for `FINISHED.txt` (poll with a script, not turns): read `RESULTS.md`. Required: section 2 shows red_on_old = 10 failed and green_new = 10 passed; section 3 every suite `passed` with 0 failed
(`test_v8f_ego_perturb` 2 skipped and `test_v8f_orc_chain` 32 skipped are by design, as for e7ea20d1: 80p/2s and 13p/32s); section 4 `SUMMARY: 27/27 mutants killed`, every CONTROL rc=0, and `M17a`, `M17b` killed by a BEHAVIOURAL test
(`equal_seeds_...`, `..._selection_uniforms_...`, `..._box2_shape...`). Then report to the Master Mind: land `stage/` (3 files) on the tip via the coordinated lander; the arm-A re-run needs the fixed trainer.

IF SOMETHING IS RED. Do not edit blindly. Work tree `F:/TanitAD-scratch/compose_fix/tree` (trainer `stack/scripts/v8f_mini_train.py`, data `stack/tanitad/data/v8f_mini_data.py`, tests `stack/tests/test_v8f_ego_dagger_compose.py`); pristine e7ea20d1 export `.../pristine`; old-code+new-tests `.../oldnew`; mutation copy `.../mut`
(its three files must equal `.../mutpristine/*`; after ANY edit of the tree refresh `mut` and `mutpristine`, then `python tmp/dry_anchors.py` must print `anchor problems: 0`).
Re-run one file: `bash F:/TanitAD-scratch/compose_fix/tmp/run_k.sh F:/TanitAD-scratch/compose_fix/tree <name> "<-k expr or empty>" <test file stem>` (waits for a pytest slot; writes `logs/<name>.log`). Rebuild the deliverable: `python -I F:/TanitAD-scratch/compose_fix/tmp/assemble.py`.

IF THE PIPELINES DIED (no `*.progress` update for >15 min and `pytest_runs.py` reads 0): relaunch with the PowerShell tool by WMI (`Invoke-CimMethod Win32_Process Create`, `cmd.exe /c ""C:\Program Files\Git\bin\bash.exe" F:/TanitAD-scratch/compose_fix/tmp/pipe_<x>.sh < NUL > ...\logs\pipe_<x>.sh.out 2>&1"`, CreateFlags 0x08000200); the scripts are idempotent.

RULES. pytest only when `C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/orcread/pytest_runs.py` < 2; `CUDA_VISIBLE_DEVICES=-1`; no git add/commit/push; never touch C:/Users/Admin/qland or box2_ship; box 2 is not touched by this package.
