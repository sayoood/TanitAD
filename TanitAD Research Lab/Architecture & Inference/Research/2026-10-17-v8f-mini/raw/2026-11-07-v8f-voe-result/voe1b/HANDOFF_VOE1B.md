# HANDOFF_VOE1B (agent, 2026-11-07; nothing staged, committed or pushed)

Card `../PREREG_V8F_VOE1B.md`, frozen prefix 5,785 bytes `aa317fe3...` (re-hashed, unedited). `IMPL_CHOICES_VOE1B.md` frozen 15:40:13, sha256 `f7c2d624...` (`IMPL_CHOICES_VOE1B.sha256`), BEFORE any statistic. Tier T0. STRATEGIC family PAUSED (item 25).
Blindness kept: `voe1/RESULT_VOE1_INVALID_READ.md` and the M2b manipulation rows of `voe1/RESULT_VOE1.json` were never opened.

## State
* Bank rebuilt (F:, 113 s); reproduction vs `latents.pt` 3.81e-06 / 2.86e-06 (as VOE1). ACTION-SWAP-M qualifies **530 windows / 104 episodes** (labels only), not UNDERPOWERED. Tests: **22 passed** (`PYTEST_VOE1B.log`, hashes in `PYTEST_VOE1B.hashes`; VOE1's 13 + 9, incl. the added INIT-LEVEL family test and a deliberate-regression arm).
* **The real run was launched DETACHED 15:52:36 (WMI, PID 78896): `job_run_voe1b.cmd` = `run_voe1b.py --workers 6` -> (only if rc=0) `report_voe1b.py` -> `cleanup_voe1b.py`.** Expected about 7 to 9 min. Done marker `VOE1B_RUN.done` (`rc=<n>`); logs `run_voe1b.run1.log`, `report_voe1b.run1.log`, `cleanup_voe1b.run1.log`.
* Outputs when done: `RESULT_VOE1B.json` (raw), `traces_voe1b.npz`, `CONTROLS_VOE1B.json`, `RESULT_VOE1B.md` (controls 1 to 4 first; INVALID prints no label table and the cheapest next lever from control rows), `LABELS_VOE1B.json` (valid only), `fig/trace_<MANIP>.png` x6 (valid only).
* **No result has been read by this agent.** A smoke run (30 episodes, 3 untrained seeds per chassis, plumbing only; `--smoke-force-tables` flag exists for smoke only) was glanced at for layout; no definition changed afterwards.

## Next step (exact)
```
cat voe1b/VOE1B_RUN.done            # rc=0 expected
sed -n 1,60p voe1b/RESULT_VOE1B.md  # controls first
```
If the report did not run or crashed (run rc=0 but no `RESULT_VOE1B.md`), re-run it (no recompute):
```
export PYTHONDONTWRITEBYTECODE=1; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/voe1b && $PY report_voe1b.py
```
If the run failed (rc=1): read `run_voe1b.run1.log`; rebuild the bank first if it is gone: `export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 WMACT_CODE_ROOT=<tip_1246c8dd path as in job_run_voe1b.cmd>; $PY build_bank_voe1b.py` (113 s), then `cmd /c job_run_voe1b.cmd` via WMI (`< NUL`).
If a control is INVALID: report it as committed, then the lever the report prints; do NOT read the M2b rows to choose it.

## Readings I made where the card was ambiguous (all in IMPL_CHOICES, frozen before any statistic)
* ACTION-SWAP-M sign test: both the window's and the donor's signed mean must reach 0.5 m/s^2 / 0.05 rad/s (the stricter reading; looser reading 545 windows).
* "16 untrained models": 16 per trained chassis (32 models), each M2b seed compared with the 16 on its own chassis.
* Control 1 percentile: `numpy.percentile` linear (inverted-cdf printed beside). Control 2: pooled share over 288 cells (2 chassis x 16 x 3 manipulations x 3 levels), per chassis printed. Level label over both seeds: VOE1's rule extended (`BLIND/INIT-LEVEL` for one BLIND + one INIT-LEVEL).

## Post-freeze edits (none to a hashed definition)
`run_voe1b.py`, `report_voe1b.py`, tests, `cleanup_voe1b.py`, `job_run_voe1b.cmd` were written after the IMPL freeze (hashes of core/real/build_bank equal the IMPL values; run script and tests are hashed in `RESULT_VOE1B.json` meta). Report-only edits after the smoke: figure legend/title, no IQR fill, `--smoke-force-tables`.

## Deletions (own temp, re-creatable, none the only copy)
Automatic after rc=0, logged by `cleanup_voe1b.py` into `CLEANUP_VOE1B.log` (path, size): `F:/TanitAD-scratch/voe1b_bank/bank_voe1b.pt` (about 248 MB; rebuild `build_bank_voe1b.py`, 113 s), `F:/TanitAD-scratch/voe1b_smoke/` (plumbing test), `voe1b/__pycache__` if any. Check `CLEANUP_VOE1B.log`; if absent the job has not finished.
* Already deleted (logged in `CLEANUP_AGENT_SCRATCH.log`): three of my scratch scripts (1,309 + 1,095 + 12,704 bytes; toy exploration and a tests-addition draft, merged into `test_voe1b_synth.py`).
