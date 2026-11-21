# HANDOFF: scene-tap vs visual-tap probe (launched 2026-10-15 18:50 Berlin; ends on its own)

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. PACKAGE DIR (this folder) = `C:/Users/Admin/AppData/Local/Temp/stp/<PK>/raw/2026-10-15-scene-tap-probe/`. Code is the archive of `8e106f9`.

## State
ONE detached chain (`chain.py`) is running on the dev box (no A40, no ssh). **PIDs: 37012 (launcher shim) -> 24348 (the interpreter running `chain.py`)**; check `tasklist //FI "PID eq 24348"`.
* **Stage 1** (~1.0-1.6 h): `launch_extract.py` extracts `scene_ctx` fp32 [64, 256] for all 1,058 tokens into `D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_scene_ctx/` (8 done by the smoke), re-checking every token's
  same-forward proof (`visual_ctx` max |d| vs `c3_full_ctx` must be <= 1e-4; **STOPS and writes PROBE_FAILED otherwise**), then writes `scene_ctx_manifest.json` (sha256 per file) and `extract_summary.json`.
* **Stage 2** (~1.8-2 h): `scene_tap_probe.py run --out <this folder> --work D:/Projects/TanitAD-artifacts/scene_tap_probe --updates 2000 --folds-sha256 20dec3431663de2201190020f97c74f7975468a8e9e178f3db4d52fbacd5885f`.
  6 init floors, then 35 units (V-s0, S-s0, C-s0, V-s1, S-s1, V-s2, S-s2; 5 folds each). It takes the shared dev-box GPU lock (`devbox_gpu.lock`) in <= 20-min stretches and waits for an idle GPU (it may sit waiting for the PI's other jobs).
  Units are resumable: `D:/Projects/TanitAD-artifacts/scene_tap_probe/units/*.npz`.

## Markers (this folder)
`PROBE_RESULT.json` then `PROBE_DONE` on success; `PROBE_FAILED` (stage, rc, log tail) on any failure. Logs: `chain.log`, `probe.log`, `logs/extract_full.log`, `logs/grid.log`, `logs/chain.out|err`.

## Exact read command
```
cd "<package dir>"; ls PROBE_DONE PROBE_FAILED 2>/dev/null; tail -3 chain.log; tail -3 probe.log
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I -c "import json;r=json.load(open('PROBE_RESULT.json'));print(json.dumps(r['decision'],indent=1));print(r['insample_read']);print(r['prior_control']);print({k:(v['point'],v['ci95'],v['separated_beyond_floor']) for k,v in r['contrasts'].items()})"
```
`PROBE_RESULT.json` holds every number per arm/seed/fold and `decision.row` (computed mechanically by `decide` from PREREG section 4). The job interprets nothing beyond that row.
**If it died without a marker:** re-run `chain.py --updates 2000 --folds-sha256 <sha above>` with the interpreter `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I`; both stages resume. Never `pkill -f`; kill by explicit PID.

## What is NOT verified
* The full 1,058-token extraction and the 35 units have not run yet (only: 8-token same-forward smoke, bit-identical; a 20-update timing smoke; a 3-update unscored dry run of `run_unit` on the real universe; 19 tests incl. a synthetic 35-unit end-to-end).
* S1's distinct-frame / distinct-log counts (23,936-23,972 frames, 1,133-1,145 logs) are from an emulation of the trainer's draw, not from the pod's logs (PREREG 2.2). Frames seen = 400 x 64 = 25,600 is from the arms' `start` event.
* The S timing used a synthetic [8, 64, 256] ctx; the S estimate is a shape-based extrapolation.
* The bootstrap holds the 5 fold models fixed and the folds share 3/4 of their training data; the CI is "another draw of logs + a training seed", not "another corpus".

## Landing
Package files (small) go through the mirror lander. NOT in the repo (D: only, single copy): `c3_full_scene_ctx/` (1,058 x 64 x 256 fp32 + logits), `scene_tap_probe/units/*.npz`.
