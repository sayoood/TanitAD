# HANDOFF -- H-HJ-COLLAPSE-1 (RA-0), stream S2 -- written 2026-10-17 ~14:55 Europe/Berlin

**DONE 2026-10-17 (write-up agent): 40/40 episodes, RESULT.md final (0 PENDING), `raw/GAC_ROW_UPDATE.md` written. Remaining for the orchestrator: paste that cell into `GOALS_AND_CLAIMS.md` row H-HJ-COLLAPSE-1 on the true tip; stage the package. Everything below is the historical hand-off.**


## State
The measurement code is written, shaken down on episode 00 and RUNNING DETACHED on the dev box (CPU only,
OMP_NUM_THREADS=6, RTX 4060 untouched). Relaunched ~14:48 with the final code (episode 00 recomputed); ~250-370 s per episode
=> **expected finish ~17:30-18:30 the same day** (see `run3.log` timings). A watcher (`code/post_run.py`) polls every 120 s, relaunches the
job if it dies (max 3x; resumable per episode) and, at 40/40, runs the read-out and renders the numbers.

| thing | where |
|---|---|
| work dir (per-episode JSON = commit marker, float64 moments npz, logs) | `C:/Users/Admin/hjcollapse_run1/` (`ep/`, `run3.log`, `post_run.log`, `meta.json`, `tripwire.json`) |
| repo mirror of per-episode JSON (small) | `raw/ep/<clip>.json`, `raw/meta.json`, `raw/tripwire.json` |
| measurement job | `code/hj_collapse_diag.py` (design + PREREG decision rule frozen in its docstring BEFORE the data) |
| read-out | `code/hj_collapse_readout.py` -> `raw/readout.json`, `raw/rho_moments_tokmean.npz` |
| tables | `code/render_numbers.py` -> `raw/RESULT_numbers.md` |
| launcher / watcher | `code/launch_detached.ps1`, `code/post_run.py` |
| `C:/Users/Admin/hjcollapse_work/` | STALE shake-down dir (old code, 1 episode). Ignore. |

## How to check
```
tail -3 C:/Users/Admin/hjcollapse_run1/run3.log          # "[done] NN/40 ..." lines; "[ALL-DONE]" at the end
tail -3 C:/Users/Admin/hjcollapse_run1/post_run.log       # "episodes done: n/40" ... "POST-RUN-DONE"
ls C:/Users/Admin/hjcollapse_run1/ep/*.json | wc -l       # 40 when done
```
Process check (do NOT use `pgrep -f`; match the executable): PowerShell
`Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -like '*hj_collapse_diag*' }`.
If the job is dead and < 40 episodes are done and the watcher has given up: re-run
`powershell -File code/launch_detached.ps1 -Work C:\Users\Admin\hjcollapse_run1 -Tag run3` (resumes).

## Exact next step (fresh Sonnet agent, <= 30 min of work, NO model load needed)
1. Confirm 40/40: `ls .../ep/*.json | wc -l`. If the watcher already ran, `raw/readout.json` and
   `raw/RESULT_numbers.md` exist; otherwise run (venv `C:/Users/Admin/venvs/tanitad`, `PYTHONPATH` = repo
   `stack;taniteval`, `PYTHONIOENCODING=utf-8`):
   `python code/hj_collapse_readout.py --work C:/Users/Admin/hjcollapse_run1 --out raw` then
   `python code/render_numbers.py raw/readout.json`.
2. Copy the small per-episode JSON to `raw/ep/` if the mirror is incomplete (the job writes it itself).
3. Write `RESULT.md` (a pre-filled draft with the verified facts and the interpretation scaffold is already
   there -- replace every `PENDING` with the read-out number, state the PREREG outcome from
   `readout.json["prereg"]`, the named next lever, and the post-hoc caveats listed in RESULT.md section 6).
4. Hand back <= 300 words + manifest. The register row (`Project Steering/GOALS_AND_CLAIMS.md`,
   H-HJ-COLLAPSE-1) must be updated by the orchestrating session (this stream may not modify tracked files).

## Do not
* Do not change the decision rule or the thresholds after reading the numbers (rule is in the docstring of
  `hj_collapse_diag.py`, committed before the data). Report the point-estimate outcome as written; post-hoc
  mechanism evidence goes in a separately labelled section.
* Do not run anything on the GPU.
