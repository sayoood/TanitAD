# HANDOFF_LADDER_READ (agent, 2026-11-09 16:25; dev box CPU; nothing staged, committed or pushed; box 2 and `ladder_state` / `otband_state` untouched)

## State
* Tool built and tested: `lr_lib.py`, `run_ladder_read.py`, `report_ladder_read.py`, `control_m2b.py`, `test_ladder_read.py` (13 passed, `TEST_LADDER_READ.log`). IMPL in `IMPL_CHOICES_LADDER_READ.md` (sha256 `bdbf488a...`, file `.sha256`) + `IMPL_ADDENDUM_LADDER_READ.md`.
* Controls read: M2b reproduction through this code path PASS (`CONTROL_M2B.json`, MRR difference 0.0, tolerance 1e-3; wrong-seed regression FAILS as required, `CONTROL_M2B_WRONGSEED.json`). Copy-Last vs itself 0.0; built-vs-argv checked per arm.
* Smoke on C0 s0 and R35 s0 (19 reference draws): `RESULT_LADDER_READ_SMOKE.md` (decides nothing).
* **Deciding run launched detached 16:22** (WMI, `job_run_ladder_read.cmd`): 8 arms in sequence, 199 reference draws, about 3 to 5 min per arm (ESTIMATED from the smoke: 60 to 90 s at 19 draws). Log `run_ladder_read.log`; marker **`LADDER_READ_RUN.done`** (`rc=0` = all arms read; the job then ran the report). Per-arm results `results/arm_<name>.json` (resumable: a finished arm is skipped); a failed arm leaves `results/FAILED_<name>.txt`.

## Exact next steps
1. When `LADDER_READ_RUN.done` exists: `tail run_ladder_read.log`; `RESULT_LADDER_READ.md` / `.json` hold the labels (BAND-MET / PARTIAL / MISS per arm, both seeds) for the eight banked arms. R35b is pending and the report says so.
2. When `ladder_R35b_s0` is pulled into `C:/Users/Admin/t2v8f_pulls/` (box 2, about 19:00 ESTIMATED): `cmd /c env_ladder_read.cmd` then `python -u run_ladder_read.py R35b_s0` and `python report_ladder_read.py` (python = `C:\Users\Admin\venvs\tanitad\Scripts\python.exe`, cwd this folder). No code change. The report then prints the floor |skill(R35 s0) - skill(R35b s0)| per level and flags lever effects against it.
3. The Master Mind reads the tables and writes the verdict (this tool writes none).
4. Free the scratch afterwards (mine, re-creatable): `F:/TanitAD-scratch/ladder_read/` (code tree export about 38 MB, frames caches `cache/frames_s0.npz`, `frames_s1.npz` 14.7 MB each). Delete by literal path only when no job runs; keep `decn_scratch` and `plantest_tree`.

## Reading notes
* Deciding statistic = IA-0 skill against Copy-Last in the level's latent at its band end (L1 2.0, L2 6.0, L3 12.0 s), two-part rule with the answer deranged inside the clip (IMAG1B's IA-0 reference; no probe). Health = IMAG1B's MRR at k = 1. Beside (no bar): IA-1, IA-2, IA-3, TSI, cross-level slope.
* Anchors are the level's own states of the whole clip; L3 has few (7 states at strides 3,5), so L3 18 s is not readable (1 anchor per clip) and is beside anyway.
* C0 (2,2): L2 3.0 s and L3 6, 9, 15, 18 s are off the grid; the table states the actual horizon.
* One case the card does not name (all three levels beat Copy-Last but health fails) is printed as BAND-PARTIAL with a note.
