# HANDOFF: P2 full-navtest 64-proposal table (written 2026-10-16 00:08 Berlin; job (re)launched 00:03:25)

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. PACKAGE DIR (the job runs FROM it) = `C:/Users/Admin/AppData/Local/Temp/p2f/<PK>/raw/2026-10-16-p2-full-table/`. Budgets and gates: `PLAN.md`.

## State
ONE detached chain, launched via WMI (CreateFlags 0x08000200): cmd **33568** -> python shim **3112** -> **driver 32428** (the pid in `STATUS.json`; it holds the GPU lock body while dumping). Now: stage `dump`, **waiting for the GPU**. The probe (`scene_tap_probe.py`, pids 30716/28444, lock holder, ~00:30) must have EXITED, then `wait_idle.py`, then the O_EXCL lock. `refcv8-g0-step5000` has been queued via `gpu_lock.py acquire` since 21:47 (polls every 60 s), so it will probably run first; this job then waits for it. Machine kept awake by the driver (ES_SYSTEM_REQUIRED, released on exit). A first launch (23:54, pid 32436) was killed by explicit PID before it took any lock, because review found `take_lock` would have removed another job's lock that looked stale (`gpu_lock.py` writes an MSYS pid tasklist cannot see); the rule is now: another job's lock is NEVER removed, a stale-looking one is only reported in `STATUS.json` (`_looks_stale`). Its log: `driver_main_attempt0_killed_for_stale_lock_rule.log`.
Flow: 14 whole-log chunks (`chunks/chunks.json`, 502-993 tokens) -> per chunk dump (GPU) -> **same-pick gate BEFORE scoring** -> 64 harness columns on 6 workers, overlapped with the next dump -> table -> proof on all 12,146 -> anatomy -> 509 cross-check -> manifest. The lock is kept across chunks unless a `gpu_lock.py acquire` waiter exists.
**ETA = (when the lock is acquired) + 5 to 7 h** (central 6 h). Best case lock at ~00:35 -> done ~05:30-07:30 Berlin.

## Check (zero model turns)
```
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I "<package dir>/status_p2full.py"
ls "<package dir>"/P2FULL_DONE "<package dir>"/P2FULL_FAILED 2>/dev/null ; tail -3 "<package dir>/driver_main.log" | cut -c1-300
```
Heartbeat in `STATUS.json` every 60 s; > 180 s old with the pid alive = hang. `guard` in STATUS = paused on low D: space (< 3 GiB) or RAM (< 5,000 MiB available), not an error.

## Result
`P2FULL_DONE` (json) -> `same_pick_proof.json` (must say PASS), `anatomy.json`, `crosscheck_509.json`, `zero_subset_vs_landed.json`, `P2_MEASURED_from_full.json`, `gate_table.json`, `MANIFEST.sha256.json`.
Table: `D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz` (PRIMARY) + `C:/Users/Admin/p2full_work/final/table_p2_full.npz` (second copy); per-chunk dumps `D:/Projects/TanitAD-artifacts/p2_full_table/dumps/`. Neither is in git (~90 MB): the Master Mind decides where the table is banked.
Read: `np.load(table, allow_pickle=False)`: `pdms [12146,64]`, `sub [12146,64,6]` (order in `sub_names`), `logits [N,64,6]`, `proposals [N,64,8,3]`, `pick`, `token` (SORTED), `log`, `chunk`, `valid`.

## Resume / failure
Same command resumes (done chunks and done columns are skipped): `powershell -NoProfile -File "<package dir>/launch_p2full.ps1"` (add `-Workers N` to change the pool; `-FromStage final` redoes only table + proof + analysis).
`P2FULL_FAILED` holds the reason: **rc 2 = the dump does not reproduce P2's banked pick (gate): REPORT IT, DO NOT RE-RUN** (`state/gate_fidelity_cNN.json`, `same_pick_proof.json` stub); rc 1 dump/preflight; rc 3 harness columns (`score_failed_cols`, `logs/score_*`); rc 4 table G2/G3; rc 5 same-pick proof. A hard kill leaves `devbox_gpu.lock` with a dead pid: `gpu_lock.py` never breaks it, remove it by hand only after `tasklist` shows the pid gone.

## Caveats
- NOT verified: the GPU dump through this code path, the dump's RAM, the take-lock sequence against a FREE real lock (each piece is tested: CPU smoke, real `wait_idle` 47 s, scratch lock, mocks).
- Nothing is fitted on this table. It is the confirmation set of the planned DAC-aware selection-rule study: fit the rule on RL-val, pre-register, then read it here once.
- The package dir is under `%TEMP%`: copy it (or land it) before any temp cleanup.
