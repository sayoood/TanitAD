# RESULT — the REFe navtest evaluation path, bars E-0 / E-1 / E-2 (SPEC_NAVTEST.md)

**2026-09-24 02:10–03:05 Berlin, dev box. All three path bars PASS.** No model-quality number is
claimed here: the only REFe score below comes from a ~30-optimiser-step checkpoint on ONE log and exists
to prove the path, exactly as SPEC E-2 intends.

| bar | result | evidence |
|---|---|---|
| **E-0** harness live | ✅ **PASS** -- the HUMAN seam (W3's exported `human_future_poses`) scored through THIS package's driver reproduces W3's banked `HUMANseam_navtest` cells on the 200-token subset: **max abs diff 0.0 over 200 tokens x every score column**; guards C1-C3 PASS (200/200 successful, PDMS identity delta 0.0); subset PDMS 94.1195 | `raw/e0_human_sub200/` |
| **E-1** converter | ✅ **PASS** -- straight line 0.0 m; arcs R 20/60/8 m exactly at their chord-sagitta bound (2.499e-2 / 1.875e-2 / 9.998e-3 m); heading across +-pi 8.9e-16 rad (`refe_navtest_seam.py --selftest`, bar as AMENDED before any score) | `ZZCONVERTER_EXACTZZ` |
| **E-2** pipeline on an early checkpoint | ✅ **PASS** -- `snap_epoch001.pt` (md5 `d64af766…`, ~30 steps, epoch 0 = 7,536 scenes): **25/25 tokens** of log `2021.05.25.14.16.10_veh-35_02482_02649` produce a finite seam through `REFePlanner.infer` (4 cameras from the OpenScene test archive, route goal, per-sample rig), 0 misses; W3's scorer: 25/25 successful, C1 delta 0.0, guards PASS | `raw/e2_ep001_log1/` |

## What the path found on the way (each a defect that would otherwise have surfaced on day 7)

1. **Hydra cannot parse the package path** (`Architecture & Inference` has a space and an `&`): the devkit
   refused `+agent.seam_file=…` with `LexerNoViableAltException`. Seams and scorer scratch now live under
   `D:/Projects/TanitAD/data/refe_navtest/` and results are banked back here.
2. **`planner.py` could only import `augment_routes` when a runner put `code/` on sys.path** -- the first
   route goal raised `ModuleNotFoundError`. The planner now resolves its sibling dir itself (and gained the
   missing `import sys`). `diag_planner_holds.py` still PASSES 4/4 (`PLANNER_HOLDS_GUARDED`).
3. **My frame extractor wrote its zips' central directories only at the very end**, so no frame was readable
   until all 32 archives were done; E-2 ran on one log extracted with closed containers instead.
   *Fixed 2026-09-24 04:15: every zip is now closed after EACH archive and in `finally` on a crash (append
   mode re-reads the directory on the next archive); checked against the old mode, whose unclosed zip is
   unreadable. The full run itself finished `ZZNAVTEST_FRAMES_OK 48584 48584` (0 missing, 0 pairing misses).*
4. **My first frame control conflated two questions.** The log's future on NAVSIM's own 0.5 s grid matches
   W3's human future **exactly (max 7.3e-15 m, 25/25)** -- frame convention (rear axle, ego frame) and
   token -> scenario pairing are right. The same future on REFe's 0.2 s output grid, interpolated, differs by
   up to **0.123 m** (3/25 tokens, the odd half-seconds; median 0.0095 m) from uneven lidar-frame timing: an
   inherent conversion residual, now REPORTED per seam and not gated.

## The path's numbers (NOT a result)

`snap_epoch001` on the 25 tokens of one log: PDMS **29.20** (NC 52.0 · DAC 100.0 · EP 33.29 · TTC 24.0 ·
C 68.0 · DDC 96.0) -- a ~30-step model; the do-nothing STOP floor on full navtest is 61.82 (W3). Cost:
**~4.6 s/token** seam build on the shared dev box (scenario + route + 4 JPEG decodes + ViT-L fp32 on the
RTX 4060) -> the full 12,146 tokens is ~15 h single-process; learning-curve checkpoints use a fixed subset.

## Next (SPEC E-3 / E-4)

The full navtest frame extraction (`build_navtest_frames.py`, 48,584 frames, 0 pairing misses) is running;
then a fixed subset per checkpoint with its STOP floor and HUMAN ceiling (paired episode-cluster bootstrap)
plus the four families through `…/2026-09-23-refcv6-standard-tests/navsim/code/families6.py`.
