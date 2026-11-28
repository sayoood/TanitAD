# HANDOFF: WP YAW item 1, stream YF3 (written 2026-11-07 ~03:00 Berlin; next agent = a FRESH Sonnet agent, ANALYSE and continue)

Dir `$Y3 = E:/Projects/TanitAD-artifacts/yawrate_wp/YF3/`. Python `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`. Nothing in the repo, PK, YF2 or YT was edited; nothing staged. MM lands.
Frozen design: `YF2/PREREG_YAWFT.md` sha256 `d28a630f...5b74` (verified at start; the pipeline re-verifies in stage `pins`). `AMENDMENT_1.md` (sha in `AMENDMENT_1.sha256`) was written BEFORE any real trained arm was evaluated: operationalisation + 2 code-route fixes (K4 for arm C; adapter frame guard at n < 100). `YF3_PINS.sha256` pins every YF3 code file.

## State
* **FULL PIPELINE RUNNING (detached, WMI 0x08000200):** python PID in `run_full/status.json` (`pid`; launched 02:34:34, wrapper cmd 21664). It trains (CPU thread) and evaluates (GPU thread, YF2 lock, waits behind foreign holders, never forces) as checkpoints appear:
  R1 (B s0, B s1, C s0 + A0 re-run G6, G8, identity gates) -> `analysis/R1.json|.md` -> R2 (B,C x s0,s1) -> R3, chaining on NOT PROVEN / REFUTED(gates green); VOID or any failure stops.
* **Markers:** `markers/yf3_full.DONE` (json: verdicts) or `markers/yf3_full.FAILED` (json: traceback tail); progress `run_full/status.json`, `run_full/PIPELINE.log`; sub-logs `run_full/logs/*.log`, evals `run_full/eval/<arm>/`.
* **ETA (ESTIMATED; MEASURED so far: A0 full seam 2,400 s under CPU contention vs 1,222 s banked alone; R1 B arm train+ckpt+identity 1,350-1,670 s):** per eval ~50 min incl. lock + scoring; R1 verdict ~05:15, R2 ~08:30, R3 ~11:45 Berlin, longer if other agents hold the lock (YAX held it at 03:26) or RAM. An ADOPT stops the chain early (checkpoints kept in `run_full/ckpt/`).
* The W3 harness RAM guard (`W3_RAM_GUARD_ABORT`, floor 3000 MB, shared box) aborted A0's first scoring attempt; `yf3_eval_arm.run_score` retries after free RAM >= 4000 MB (stage2_sft.py protocol; 20 attempts for evals launched after 03:31, 6 for A0). Not a model fault.
* Disk: E: had 21 GB free at launch; composed 1.29 GB checkpoints of a rung that did not ADOPT are deleted after its analysis (reproducible from `arms/*/final.pt` with `build_yaw_ckpt_yf3.py`).

## Next commands
1. Poll by script, not turns: `until [ -f $Y3/markers/yf3_full.DONE ] || [ -f $Y3/markers/yf3_full.FAILED ]; do sleep 60; done`.
2. Read `run_full/analysis/R1.md` (and R2/R3): the verdict is mechanical (ADOPT / NOT PROVEN / REFUTED / VOID), gates, tier stamp, estimator, four families (TACTICAL n/a with reason, STRATEGIC PAUSED). `run_full/FINAL.md` has the chain verdict and next steps.
3. FAILED: read `status.json.why`; every stage is idempotent, so FIX THE CAUSE and relaunch the SAME command (detached, WMI): `yf3_pipeline.py --root $Y3/run_full --max-rung R3`. A VOID verdict = a red gate (see `analysis/R*.json gates_red`); that is an instrument fault, diagnose in the PREREG order (G7 shim frame, lr VAL curve in `arms/*/metrics.jsonl`, G2 sign).
4. ADOPT: log it, do NOT claim more than the PREREG allows (open-loop NAVSIM only); next = (a) full 12,146-token read (`eval/stage2_sft.py` pattern, shim launcher), (b) closed-loop A/B, (c) P6 final. After R3 non-ADOPT: PREREG elimination text + next levers are in `FINAL.md`.
5. Rule Zero: a verdict is not a finished turn.

## Smoke tail still queued (separate root `smoke_run2`, marker `markers/yf3_smoke.DONE|FAILED`)
Smoke R2 C-arm eval + R3 train/ckpt/identity are waiting on the GPU lock behind the full run; the live G7 refusal arm is NOT yet run: `yf3_smoke_reds.py --root smoke_run2 --live` (one-token seam, ~9 min of lock). Neither gates the full run.

## Known gaps (UNVERIFIED, named)
* G1 is YT's DB control; YF3 only asserts table coverage. G8 is measured on R1 and CARRIED to R2/R3. R3 was smoked train+ckpt+identity only (no eval); R2 smoke used ONE seed pair.
* `cross_*`/`along_*` secondary rows use ratio-of-sums per window (reproduces families6 means exactly, banked check); the CI of those rows is the log-cluster bootstrap, not families6's episode one. Decision uses only curvature MAE + PDMS + `fam_adverse` (families6's own intervals, unpaired, conservative).
* Pick-flip sensitivity: on the smoke (4 steps, |col4| ~ 1e-2) 15-22 % of tokens changed pick by > 1 cm: scorer ties among 64 proposals make tiny perturbations visible; F (seed floor) is the guard.
* The harness writes run folders under `<navsim root>/exp/w3_navtest_v1/runs/refe_yf3_{smoke,full}_*` (outside YF3).
