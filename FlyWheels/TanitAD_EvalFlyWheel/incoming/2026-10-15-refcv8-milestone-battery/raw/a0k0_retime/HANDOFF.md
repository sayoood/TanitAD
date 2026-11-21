# UPDATE 2026-10-19 02:53 -- PIPELINE FINISHED (supersedes "WAITING AT THE MEMORY GATE" below)
`logs/full.log`: `ZZRTDONEZZ pipeline_rc=0`, every STEP_RC = 0, `PYTEST_RC=0` (23 passed); controls ALL_PASS = True. All outputs exist. Read `RESULTS_BRIEF.md` / `A0K0_RETIME.md`.
Headline (EMA-30k, ceiling-compliant, T1 open loop, paired episode-cluster bootstrap B 2000, single training seed; s0 | s1):
* S2 (n 4,159 w / 138 ep) RETIME_XF - ha0_ext -0.0269 [-0.0376,-0.0156] | -0.0265 [-0.0371,-0.0153]; GATED_RETIME_XF -0.0310 [-0.0419,-0.0201] | -0.0301 [-0.0402,-0.0191]. LE-E1 / E1P verdict form PASS (both).
* S6 (n 3,251 / 138) -ha0_ext: RETIME -0.9825 | -0.9808; GATED -0.9932 | -0.9907 (SEP). ADE vs os: RETIME -0.2290 [-0.3254,-0.1316], GATED -0.2396 (RS_XF -0.1627).
* ACC-LAT (S6, compliant, EMA-30k): HOLDS for both arms and not vacuous (ADE vs os separated better at both seeds). Heading vs os -0.037 [-0.105,+0.027] / -0.047 [-0.112,+0.015]; curvature, yaw rate, cross rows: none separated worse at either seed (GATED yaw rate separated BETTER). No LATERAL/any row separated worse than refcv7-50,400.
* Fitted g (EMA-30k, both seeds pooled, all windows) S2 [0.997,0.583,0.426,0.382]; S6 [0.604,0.421,0.464,0.508,0.530,0.555]; gated tables in FROZEN_GAINS_RETIME_ema30k.json.
CAVEATS (read before quoting): (1) ACC-LAT is defined on EMA-30k S6 compliant ONLY. BESIDE it, GATED_RETIME_XF on raw-30k ALL windows FAILS ACC-LAT (LAT_cross_mae_m separated worse vs os at both seeds, S6 all and S2 all). (2) Geometry: non-monotone progress windows 1.7 %, extrapolated points 11.0 %, fallback-direction windows 1.8 % (S6 compliant s0). (3) Gains are fit on eval episodes (cross-fitted): lever estimate, not a bar pass; the genuine forward test on EMA 40k/48,852 is NOT run. (4) Frozen forward on both 30k trees is a path check (in-sample / same run).

# HANDOFF -- a0k0_retime (arc-length re-timing), EvalFlyWheel analyst, 2026-10-19 ~01:25 Berlin

State: PREREG_RETIME.md written BEFORE any real-data number; instrument + pytest done (21 pass, 2 skip-until-deliverable; 4 mutations of the instrument each turn the suite RED).
The full pipeline (bash, background) is WAITING AT THE MEMORY GATE: FreeVirtualMemory was 5.6-5.8 GB (< 6 GB) because a SIBLING job holds 9.3 GB:
`force_s2_reroll_r8.py roll --tag full` (pid 23860, started 01:11). It is not mine; I did not touch it. The gate polls every 15 s for 40 min, then writes `STEP_RC ... = GATE_TIMEOUT`.

## Files (all NEW; `a0k0_headroom_lon_r8.py`, `a0k0_headroom_r8.py` and their tests are untouched)
* `raw/a0k0_retime/PREREG_RETIME.md` -- definition, arms, controls (a)-(e), bars LE-E1/E1P/E3/NR/OS, ACC-LAT (+ vacuity guard), both-outcome readings, the commitment: if ACC-LAT fails again the next step is the PI decision on a learned calibration (refcv8.x / refcv9), NOT another post-hoc arm.
* `code/a0k0_headroom_retime_r8.py` (copy of the LON script; commands compute | freeze | forward --tree T | render | brief), `code/test_a0k0_headroom_retime_r8.py`, `code/chain_retime_r8.sh` (the pipeline; env GATE_TRIES = number of 15 s gate polls).

## Pipeline order and markers
smoke (EMA s0 S6 compliant) -> compute -> freeze -> forward step30000_unc -> forward step30000ema_unc -> render -> brief -> pytest. Log `raw/a0k0_retime/logs/full.log`.
Poll for the marker `ZZRTDONEZZ pipeline_rc=N` (N=0 is success) -- never grep FAILED (bar verdicts contain it). Per-step lines: `STEP_RC <name> = <rc>`; `PYTEST_RC=0`.
Nothing is final until `A0K0_RETIME.json` exists and the marker reads pipeline_rc=0. Est. 20-30 min of compute once the gate opens.

## If it did not run
`GATE_TIMEOUT` in the log = the sibling still held the memory. Re-run when free: `GATE_TRIES=400 bash code/chain_retime_r8.sh` (the script smokes first; it is safe to re-run, outputs are overwritten atomically). Run exactly ONE compute at a time; check no `a0k0_headroom*` python process exists first.

## Outputs when done
`A0K0_RETIME.{json,md}`, `FROZEN_GAINS_RETIME_ema30k.json`, `FORWARD_RETIME_{step30000_unc,step30000ema_unc}.json`, `RESULTS_BRIEF.md` (the hand-back cells: E1-form s0/s1, S6 lateral rows vs os and vs refcv7, ACC-LAT label, fitted gains).
Pending (NOT run here): genuine forward test on an EMA 40k / 48,852 tree (`forward --tree <tree>` takes the frozen file unchanged).
