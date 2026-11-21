# HANDOFF - MOTIONPROBE (written 2026-11-05 12:12 local)

## State
* PREREG frozen before any real fit, then amended twice (still before any real fit, both from synthetic plumbing checks):
  v0 `PREREG_MOTIONPROBE_v0.md` -> v1 (K1 redefined) -> v2 = current `PREREG_MOTIONPROBE.md` (K2 "vanishes" = gK2 <= max(0.02, 0.10 g)).
  All shas: `FROZEN_SHA.txt`. Code: `code/motion_probe.py` (sha recorded in `raw/MOTIONPROBE.json`).
* Plumbing check on synthetic tokens (no real token): `selftest/n120_planted` (v: PRESENT, motion planted; D3 band R2 +0.956 vs F1 -1.31;
  dv ABSENT = power limit at 120 clips), `selftest/n120_null` (v: WEAK from band-R2 noise, all-windows gain +0.004; dv ABSENT);
  K1 within [-0.05, 0.03] everywhere. Older runs kept: `selftest/n40_*`, `n200_*_k1v0`, `n200_null_killed_k1v0`.
* REAL RUN LAUNCHED 12:11 local, detached (WMI cmd, pid 12124 at launch), 600 TRAIN clips, 5 folds by clip, eval-141 disjoint (asserted, 0 shared).
  Scratch `F:/TanitAD-scratch/motion_c/probe/real/` (about 13 GB). Log `logs/run.log` (script's own), `logs/stdout.txt`.
  ETA about 50-60 min (ingest ~6 min, bases+project ~5, ridge ~8, MLP ~20, grid2 ~5).
* Stages are resumable (`stage_*.done` in scratch; cells keyed by fold in `raw/cells_*.json`). To resume after a crash/reboot, re-run
  `powershell -File F:\TanitAD-scratch\motion_c\probe\launch_real.ps1` (same args; finished stages and folds are skipped).

## Completion markers
* `raw/PROBE.done`  -> finished: `RESULT.md` (auto-generated: verdict per committed rule, tables, K1/K2/K3, grid 2), `raw/MOTIONPROBE.json`,
  `raw/VERDICT.json`, `MANIFEST.txt` (path | bytes | sha256) all exist.
* `raw/PROBE.fail`  -> traceback inside; fix and re-launch (resume).

## Exact next step (fresh agent, Sonnet, <=100k context)
1. When `raw/PROBE.done` exists: read `RESULT.md` top to bottom. Check (a) the K1 table: every all-windows R2 in [-0.05, 0.03] (else the
   verdict printed is INVALID: diagnose, do not report a verdict); (b) the Verdict table against `raw/VERDICT.json`; (c) the noise-context
   table (band gain mean +/- SD over folds, all-windows gains); (d) the F1d and D3f sensitivity lines; (e) grid 2 section.
2. Add a short "Interpretation" paragraph to `RESULT.md` (what the verdict implies per PREREG section 10, including where the band-R2
   noise makes the verdict fragile) and re-run `write_manifest` (or regenerate MANIFEST with the same path|bytes|sha256 format).
3. grep RESULT.md for the avoid-list words; confirm LF only; hand back <= 300 words: verdict v and dv, key R2 with n, controls, manifest paths.
4. ESCALATE to the Master Mind: land `dagland/motion_c/probe/` (code, PREREG v0/v1/v2, FROZEN_SHA, raw, RESULT, MANIFEST, selftest) into the repo.

## Hazards
* RAM: box had ~4-6 GB free when launched (another CPU job runs); the script raises above 9 GB RSS and logs RSS per stage.
* Do not touch `dagland/motion_c/ri*` (another agent's).
* Band-R2 caveat: selected on v, so a no-signal model scores strongly negative band R2; the committed rule uses band R2 as written.
