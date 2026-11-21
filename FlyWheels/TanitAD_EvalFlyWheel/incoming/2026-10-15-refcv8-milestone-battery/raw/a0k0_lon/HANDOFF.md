# HANDOFF -- a0k0_lon (longitudinal-only residual scaling), EvalFlyWheel analyst, 2026-10-19 ~00:45 Berlin

State: PREREG_LON written BEFORE any real-data number of the new arms; instrument + pytest written; smoke passed (EMA-30k s0 S6 compliant: controls a/b/h/i
PASS -- the recomputed Amendment-2 RS_XF equals the banked cells); the FULL PIPELINE is running in the background (memory-gated, ONE compute).

## Files (all NEW; `a0k0_headroom_r8.py` and its test are untouched provenance)
* `raw/a0k0_lon/PREREG_LON.md` -- arms RS_LON_XF / GATED_RS_LON_XF / RS_LAT1_XF, bars LE-E1/E1P/E3/NR/OS, ACC-LAT (no LATERAL row of arm - os separated worse at EITHER seed, S6,
  compliant, EMA-30k), both-outcome readings, motivation disclosure, frozen-gains + forward-test definition.
* `code/a0k0_headroom_lon_r8.py` (copy + per-axis boxes; commands compute | render | freeze | forward --tree T | brief), `code/test_a0k0_headroom_lon_r8.py` (17 pass + 2 skip-until-deliverable).
* Pipeline: `bash <scratchpad>/lon_pipeline.sh` = gate -> compute -> freeze -> forward step30000_unc -> forward step30000ema_unc -> render -> brief -> pytest. Log `logs/full.log`
  (look for `ALL_DONE`, `STEP_RC ... = 0`, `PYTEST_RC=0`). Est. 14 min.

## If it did not finish
Check `logs/full.log` tail. Re-run only the missing steps with the same env (CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=4,
PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code'):
`python -X utf8 a0k0_headroom_lon_r8.py compute|freeze|forward --tree step30000_unc|forward --tree step30000ema_unc|render|brief` (gate: FreeVirtualMemory >= 6 GB; ONE at a time).
Outputs: `A0K0_LON.{json,md}`, `FROZEN_GAINS_LON_ema30k.json`, `FORWARD_LON_<tree>.json`, `RESULTS_BRIEF.md` (the hand-back cells in one file), `logs/`.

## Pending
* Forward test (Amendment-3 style): `python a0k0_headroom_lon_r8.py forward --tree <EMA 40k / 48,852 tree>` then `render`, `brief` (bars FZL-E1/E1P/E3/NR/LAT).
* Next lever if ACC-LAT FAILS (named in PREREG_LON, NOT run): scale the residual along the PRIOR's local path tangent instead of the t0 x-axis.
* Smoke observation (not the deliverable, s0 S6 compliant, EMA-30k): RS_LON_XF - os ADE -0.1482 vs RS_XF -0.1627; RS_LAT1_XF == RS_LON_XF on S6 (the cross constraint binds at 1.0).

## Update 00:49
Pipeline pid 1109472 (bash lon_pipeline.sh) running since 00:44:09; EMA-30k seed 0 done at ~270 s (log), ETA ALL_DONE ~00:59-01:00. Read `RESULTS_BRIEF.md` first when `ALL_DONE` and `PYTEST_RC=0` appear in `logs/full.log`; NOTHING is final until `A0K0_LON.json` exists. Do not start a second compute while it runs.
