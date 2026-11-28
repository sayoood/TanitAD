# STATE (agent FASTNEG-CENSUS) 2026-10-19 08:20. CPU only; stage never push.

## Done (in order, all before any label stratification)
1. Prereg sha256 verified `b6cece23...` (working copy == tip `3a0c038d`). Amendment 1 sealed 07:54: `PREREG_FASTNEG_CENSUS_AMENDMENT1.md` + `.sha256` (`f53ae097...`), in this dir.
2. Code (NEW files only) `code/`: fastneg_lib.py, fastneg_strata.py, fastneg_census.py (inputs|gate|fit), fastneg_analyze.py, strata_label_free_proof.py, test_fastneg.py, test_fastneg_more.py, test_fastneg_census.py. Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -W ignore -I`, env `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8`.
   Tests green: 5 + 15 + 2 (mutation arms 21/21 RED, reach, strata label-free, e2e plumbing).
3. strata.json written 08:04 (label-free): sha256 `67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6`; run/strata/. Label-free proof: stripped-copy run gives the identical content digest (run/strata/strata_label_free_proof.json). cuts c25/c50/c75 = 0.9076/1.0453/1.2228; S = 195 of 15,808 sets.
4. `inputs` OK (join proof 15,808 rows, 0 label mismatches). `gate` PASS 4/4 (run/gate_token.json).
5. Fits launched detached 08:16: `fit --groups primary,placebo,cold` (log run/fit_main.log) and `fit --groups null` (log run/fit_null.log). Resumable: finished tasks have run/fits/<id>.json + .npy and are skipped on restart.

## Next
- When both logs end with `FIT_GROUPS_DONE`: `python fastneg_census.py analyze --strata-sha 67f43e55...` (reads fits + HEADROW-PROBE OOF if present) -> run/results/census.json. Then write RESULT.md, paths.json, MANIFEST in out/ under the prereg's repo dir.
- DO NOT edit fastneg_lib.py / fastneg_census.py / fastneg_strata.py / test_*.py after the gate (the token binds their sha256). fastneg_analyze.py is NOT bound.
- Kill only by explicit PID (see logs). Do not read the probe's outcome files; only `raw/f10/oof_proposals_f10.npz` is read, by `analyze`.

## Update 08:20 -- HANDOFF.md written (read it first); finish watcher running
- Watcher: `code/finish.sh` (cmd 35648 -> bash 38444 -> bash 41504), log run/finish.log, markers run/FINISH_DONE.txt | run/FINISH_FAILED.txt.
- Fit processes: cmd 60980 (python 41452 -> 71024) = primary,placebo,cold; cmd 45152 (python 30884 -> 49928) = null. Kill ONLY by explicit PID.
- Resume launch (PowerShell, inline; CreateFlags 0x08000200): cmd /c "set CUDA_VISIBLE_DEVICES=-1&& set OMP_NUM_THREADS=8&& C:\Users\Admin\venvs\tanitad\Scripts\python.exe -W ignore -I C:\Users\Admin\par18\fastneg\code\fastneg_census.py fit --strata-sha 67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6 --groups <primary,placebo,cold | null> < NUL > <log> 2>&1"
- After FINISH_DONE: write RESULT.md, then `python code/build_package.py` (stages out/ + paths.json + MANIFEST.md).
