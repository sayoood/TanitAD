# HANDOFF: aux-raster veto feasibility (2026-10-16). State: all four tasks done; nothing running.

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. This dir = `<PK>/raw/2026-10-16-auxveto-feasibility/` in the temp archive tree `C:/Users/Admin/AppData/Local/Temp/auxv/`. NOT committed, NOT pushed. The harness refused a `REPORT.md`; the report is in the hand-back text.

## Re-run (all CPU, no GPU; venv `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, set `TMP=C:/Users/Admin/tmpauxv`)
* `inventory.py` (14 s) -> INVENTORY.json. `test_geodac.py` (5 s) -> `ZZTEST_PASS 39 14`.
* `agreement.py` (15 s) -> AGREEMENT.json; needs the tracker shards for its section B.
* Tracker shards (regenerable, 175 MB, ONLY at `C:/Users/Admin/tmpauxv/tracked/`): in the NAVSIM venv `C:/Users/Admin/navsim-crun/venv/Scripts/python.exe` with `PYTHONPATH=D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896`, run `tracked_dump.py --stride 1 --shard i/4 --out C:/Users/Admin/tmpauxv/tracked` for i = 0..3 (about 6 min in parallel). Smoke gate: every token prints `d0==table 64/64`.
* `pred_geometry_check.py`, `crosscheck_feed.py` (torch), `cpu_timing.py` (about 1 min).

## Where things are single-copy
* `C:/Users/Admin/tmpauxv/tracked/*.npz` (regenerable). `C:/Users/Admin/AppData/Local/Temp/rls1dl/` (RL-val download, perishable Temp; HF repo `Sayood/tanitad-refe-rls1-c0`). `Temp/shf_scratch/navtest_head_pred_H1-s*.npz` (regenerable in ~15 min CPU).
* D: only: A2 / A3R / probe units, ctx caches, both label stores (INVENTORY.md sections 4, 5).

## Next steps (the Master Mind's; none was started)
1. Decide pose-only geoDAC vs tracked geoDAC (needs initial ego state at inference) and mask-logit vs SDF-sign for the predicted raster.
2. Pre-register, blind, the instrument-level read of the A2 predicted rasters vs NAVSIM DAC on the same 67,712 proposals (ceilings: GT 87.6 % pose-only, 94.7 % tracked), then the pick-level read. `agreement_core.run(fields=...)` takes any `{token: field}` dict and `geodac.PRED_GEOM` is the 2 m geometry.
3. Do not compute any pick-level number on a raster before (2). C3 is failure-enriched; re-weight to the 4.36 % stratum for a navtest rate.

## Do NOT
* read `c3_full/head_pred_H2-s0.npz` (untrained fixture); pool P2's 509-token table with u400's; treat the A3R `A2_RESULT.md` "U10k" label as 10k updates (it is 400).
