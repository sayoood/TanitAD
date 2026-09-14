# SPEC — SAM3 inference time on Thor for the front-camera map (pre-registered 2026-09-14, before any map-level result)

PI, 2026-09-14: *"We need to optimize the inference time of sam3 on thor, I need concrete approved measures."*

## What is being sped up
The per-frame SAM3 work of the delivered front-camera map **r** on Thor: v6 extractor `classify()` on CAM_FW (17 prompts),
the v6s stripe pass (2 prompts), the front-only re-lift. Downstream stages (refine, consensus, renderer v5m fields, compose
with the map-r options) run unchanged in every arm.

MEASURED before this SPEC (feasibility only, `sam3_profile.json`, `sam3_levers_probe.json`, `sam3_precision_probe.json` on Thor):
3.66 s/frame today; grounding 2.18 s (19 × 0.115 s), image encode 0.39 s (run twice), text encoder 0.16 s, mask host copy
0.30 s, classify CPU logic 0.26 s. Instance-level first look at half precision: fp16 on the fusion encoder is at the numerical
floor of a mathematically identical fp32 batched run; bf16 is not. **No map-level or raster-level result of any arm had been
seen when these bars were written.**

## Arms (tag · flags)
| tag | measures | class |
|---|---|---|
| `spd0` | MODE=ref — today's pipeline, stock processor | reference R0 |
| `spdA` | text cache + one image encode + GPU-side selection, fp32, batch 1 | exact |
| `spdF1` | spdA + fp16 autocast on the fusion encoder | numerical |
| `spdF2` | spdA + fp16 autocast on all grounding stages + 19 prompts in one forward | numerical |
| `spdREG` | spdA with the `curb` prompt dropped | deliberate regression — MUST FAIL |
Further numerical arms (fp16 backbone, compile, async CPU) may be added later; they are judged by the SAME bars below, unchanged.

## Controls
* **C-REPRO:** R0's raw frame files equal the historical `<c8>_v6sfraw` (written by the real extractor → stripe pass →
  make_front_only scripts) on every frame, bit for bit; R0's composed map equals the delivered `render5_<c8>_v65rf` cell for cell.
  If it fails, the delivered map is not reproducible today and R0 (not the delivered map) becomes the reference — reported.
* **C-REG:** `spdREG` must FAIL at least one numerical bar on at least one clip. If it passes, the bars are too loose to approve
  anything and no numerical arm is approved.

## Bars — both clips (night 73495082f98b, day 4fbd97b6a4b7), every bar, vs R0
**EXACT class (spdA, and any later exact measure):** every raw frame file bit-identical to R0 (`tok, T_world_rig, cls_CAM_FW,
evid_CAM_FW, xstripe_CAM_FW, pts_1..7, rng_1..7`) and the composed map identical cell for cell. Anything less = FAIL.

**NUMERICAL class:**
* **N1 map agreement** ≥ 0.990 over cells seen in either map (cls ≠ 255).
* **N2 per-class IoU** ≥ 0.95 for every class with ≥ 500 cells in R0's map.
* **N3 PI checks (roi block of `pi_checks.py`, same J and window as map r):** |Δ fragments_per_1000m2| ≤ 2.0 ·
  |Δ crosswalk_coloured_share_of_crossing_area| ≤ 0.02 · |Δ curb_recall_edge_within_1.0m| ≤ 0.01 ·
  |Δ edge_precision_within_0.6m| ≤ 0.01.
* **N4 frame rasters:** mean per-frame agreement of `cls_CAM_FW` over pixels labelled in either ≥ 0.990.
* **N5 finiteness:** no NaN/inf reached the outputs (a NaN score silently drops instances; checked via N4/N1 and explicit logs).
Reported, not a bar: LiDAR scores (`thor_run.py score`, SAM3_GT_worldmap MAP_A/A2/B/C/E).

## Timing protocol
s/frame = driver wall-clock per clip ÷ frames (load, SAM3, CPU logic, write), GPU otherwise idle; model build separate.
**Goal (committed):** ≥ 2.0× faster per front frame than R0 with every measure approved. Stretch: ≤ 1.2 s/frame.

## Outcomes committed in advance
* spdA identical → the three exact measures are APPROVED without qualification.
* A numerical arm passing N1–N5 on both clips while spdREG fails → APPROVED for GT production.
* A numerical arm failing any bar → NOT approved; report the failing bar and the next lever (partial precision, per-stage).
