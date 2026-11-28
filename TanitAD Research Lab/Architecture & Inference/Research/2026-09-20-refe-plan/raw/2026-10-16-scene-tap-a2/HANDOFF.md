# HANDOFF: scene-tap Amendment A2 (arm U10k = V seed 0 at 10,000 updates) -- launched 2026-10-16 00:41 Berlin, dev box, 0 A40

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. PACKAGE DIR (this folder) = `C:/Users/Admin/AppData/Local/Temp/stp/<PK>/raw/2026-10-16-scene-tap-a2/`.
CONTRACT = `A2_CONTRACT_from_73db963.md` (the A2 section committed at 73db963). The banked run `raw/2026-10-15-scene-tap-probe/` is untouched. Python for everything: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -I`.

## State at hand-off
ONE detached chain (`chain_a2.py`), PIDs: **cmd 8732 -> shim 34300 -> chain 34060 -> grid shim 13756 -> grid python 31340** (31340 holds ~8 GB of cached ctx). Check: `tasklist //FI "PID eq 34060"`.
* Stage 1: `scene_tap_probe.py run --arms V --seeds 0 --updates 10000 --lease-s 600 --work D:/Projects/TanitAD-artifacts/scene_tap_probe_a2` = 5 units (V-s0-f0..f4), resumable. No init floors, no assembly.
* Stage 2: `a2_read.py` (identity test + zero control FIRST; either failing = no row), then H / I / Delta and exactly one of U1..U4. The chain then writes `A2_DONE`.
* At 00:45 the ctx was loaded and the grid was WAITING for the GPU: `devbox_gpu.lock` was held by `p2full_table` (pid 32428, job `p2full-main-c00`, since 00:19); another session (`refcv8-g0-step5000`, gpu_lock.py) also queues. The wait is unbounded in practice (max 48 h).
* The lease is ONE per unit (~19-20 min of GPU each; A1 timing 0.1135 s/update). ETA after the lock is won: ~5 x 19.5 min ~ 1.6 h; total depends on the P2 job.
* **The A2 copy never breaks a lock** (it only waits while a lock file exists; the banked `acquire_gpu_lock` would delete a lock whose pid Windows cannot see after 10 min -- shown by a test). A genuinely stale lock is therefore never cleared by this job: a human/Master Mind clears it.

## Watch (no model turns)
```
cd "<package dir>"; ls A2_DONE A2_FAILED 2>/dev/null; tail -3 chain_a2.log; tail -3 probe.log
until [ -f A2_DONE ] || [ -f A2_FAILED ]; do sleep 30; done; cat A2_DONE A2_FAILED 2>/dev/null     # run with run_in_background; wakes on the event
cat A2_RESULT.md      # the read, once A2_DONE exists (A2_RESULT.json has every number)
```
Markers: `A2_DONE` (json: row, H, I, delta) | `A2_FAILED` (stage, rc, log tail; rc 3 from a2_read = INSTRUMENT FAILED). Logs: `chain_a2.log`, `probe.log`, `a2_read.log`, `logs/grid.log`, `logs/a2_read.log`, `logs/chain_a2.out`.

## Resume (if the chain died without a marker)
`powershell -NoProfile -File launch_a2.ps1` -- finished units are skipped; a unit of another config is refused, never mixed; a second live chain exits at once (`chain_a2.lock`). Never `pkill -f`; kill by explicit PID.

## Verified before launch (logs/test_*.log)
* `test_scene_tap_probe.py`: **31 passed** (the 19 banked tests + 12 new: filter selects exactly V-s0's 5 folds; default = the 35-unit grid unchanged; 7 deliberate-regression arms of the filter -> all raise; `cmd_run` filtered runs only the 5 units / resumes / refuses another n_updates ("refusing to mix") / a replaced plan cannot widen the run; `cmd_run` default = 35 + 6 init + assembly; the 5 REAL banked units are refused at 10,000 and accepted at 2,000; config differs ONLY in n_updates, sha 222b588c... == PROBE_RESULT; `train_unit` bit-identical to the banked one on CPU; lease never breaks a lock + the banked acquire would).
* `test_a2_read.py`: **9 passed** (A2's table vs an independent oracle, boundaries and order; identity/zero-control green + 5 red arms; real-data identity).
* IDENTITY on the REAL banked V-s0 (2k) units (`A2_IDENTITY.json`): held-out IoU [0,40) **0.6348906684369943 == PROBE_RESULT.json exactly**, CI [0.6031610669441904, 0.6679655674359469], per-fold, in-sample 0.7024338659338062 and both SDF MAEs all exact; ZERO CONTROL Delta(V-s0 2k vs itself) = **0.0, CI [0.0, 0.0]**, not separated. V seed floor read from the file: 0.003552465097100832 (A2's text: 0.0036).

## NOT verified
* No real 10,000-update unit exists yet: the first real run of `train_unit` at 10k, the real loss-curve file and `a2_read` on U10k units are untested on real data (synthetic only).
* "Separated" = the probe's own rule (paired CI excludes 0 AND |Delta| > floor); A2's text says only "beyond V's seed floor". `A2_RESULT.json` carries `row_if_separated_meant_floor_only` / `row_depends_on_the_ci_clause`; read that flag before quoting a U2/U4 row.
* The banked 2k units carry no loss curve, so the 10k-vs-2k curve comparison is only the unit means / last-100 losses.

## Landing
Package files (small, incl. the two `.diff`s, `A2_IDENTITY.json`, `A2_RESULT.*`, the logs) go through the mirror lander. NOT in the repo (D: only, single copy): `D:/Projects/TanitAD-artifacts/scene_tap_probe_a2/units/*.npz` (5 x ~1 MB + loss curves).
