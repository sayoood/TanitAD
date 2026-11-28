# HANDOFF -- SHIP-R8 verdict pipeline (EvalFlyWheel), 2026-10-19 ~16:25 Berlin

PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery` (qland hold, not a git checkout; nothing git-added).

## State
* Scored arm = RETIME_DEPLOY8 (Amendment 2). Chain per tree: FORWARD_TRAINFIT exists (ALL_PASS) -> `deploy8_forward_r8.py` (sibling) -> `deploy_forward_r8.py` (per-surface, beside) -> `tac_retime_r8.py` -> `ship_r8_verdict.py`.
* WAITERS LIVE (launched 16:20:36): `sh code/ship_after_r8.sh step40000ema_unc` pid 1559273, `... step48852ema_unc` pid 1559274, both from `$PKG`, with `RETIME_STAGE_FILE=C:/lgs/retime_stage_scratch/stack/tanitad/retime_stage.py` in their environment.
  Log (opaque markers `ZZSHIP<NAME>ZZ`): `E:/refcv8_eval_kit/chain/ship_r8.log`; per-tree output `chain/ship_<tree>.out`. Terminal: `ZZSHIPDONEZZ` (line carries overall=PASS|FAIL|INCOMPLETE), then `ZZSHIPENDZZ`. Max wait 60 h.
  Check: `grep ZZSHIP /e/refcv8_eval_kit/chain/ship_r8.log | tail`; alive: `ps -ef | grep 'ship_after_r8[.]sh'`. Do not edit the .sh while it runs. Stop: `kill 1559273` / `kill 1559274` (by PID).
* RELAUNCH (if a waiter dies): `cd $PKG && RETIME_STAGE_FILE=C:/lgs/retime_stage_scratch/stack/tanitad/retime_stage.py nohup sh code/ship_after_r8.sh <tree> > /e/refcv8_eval_kit/chain/ship_waiter_<n>.out 2>&1 < /dev/null &`. It is resumable and re-runs all stages (artifacts are deleted before each stage so stale files cannot pass).

## Risks to read before trusting a 48,852 verdict
* `deploy8_forward_r8.py` needs `tanitad.retime_stage`; it currently resolves to the SCRATCH copy above (not landed). If that file moves, stage 2a logs `ZZSHIPDEP8FAILZZ` and the verdict reads INCOMPLETE (clauses 1-5). Land the stage, or set RETIME_STAGE_FILE / SHIP_PYPATH_PREFIX when relaunching.
* Clause 7 reads `raw/retime_deploy/PRIOR_PARITY.json` (sibling's `prior_parity_r8.py`; currently covers trees step30000ema_unc + step30000_unc). The waiter does NOT run it. Its CAVEATS (CPU float32 model-path, not an observed forward tensor; valid only on `_unc` pose-sync-off trees) are the sibling's, and are not part of the numeric clause.
* Clause (5) scope is stated as an interpretation: all of S2 and S6 x lat/lon x seeds 0/1 (8 cells), "separated worse" = CI upper bound < 0 (accuracy metrics).
* Clause (1) reads FORWARD_TRAINFIT controls (register wording) AND FORWARD_DEPLOY8 controls + a recorded gains hash (pinned b208c9c1...).

## Orientation read on EMA-30k (not the condition's tree)
`raw/ship_r8/SHIP_R8_step30000ema_unc.{json,md}`: clauses 1-5 PASS, 6 N/A, 7 PASS => overall PASS (ORIENTATION ONLY).
