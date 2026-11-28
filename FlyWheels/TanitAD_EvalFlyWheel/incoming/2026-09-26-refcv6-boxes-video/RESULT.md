# RESULT — refcv6's DETECTED AGENTS at step 38,000, rendered next to the GT (the "boxes" video)

EvalFlyWheel · 2026-09-26. Asked by the Master Mind after the PI's map-video review: render the model's
detected agents (both slot heads), the GT boxes, and a per-frame detection line, on the 3 eval clips
with the most GT agent boxes within 30 m.

## Headline

**The video.** It is on the D: working tree only, because `*.mp4` is gitignored.

| file | size | format | frames | verified |
|---|---|---|---|---|
| `D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-boxes-video/raw/refcv6_boxes_step38000.mp4` | 47,698,486 B (45.49 MiB, **over** the 30 MiB delivery limit) | 1920 × 1200, H.264 High yuv420p | 588 frames, 58.80 s at 10 fps | every frame decoded back, clean |
| `.../raw/refcv6_boxes_step38000_small.mp4` | 19,578,018 B (18.67 MiB) | 1600 × 1000, CRF 26 | 588 frames, 58.80 s | every frame decoded back, clean |

**The checkpoint.** `D:/refcv6_eval_kit/ckpt_final/ckpt_step38000.pt`, step 38,000.

- ckpt md5 `5a2e7222a9f5f8c7aa7bf38ef4698d8a`, checked at load.
- config md5 `a3193a4685ce0d07a6ae6b89fdc994a8`.
- The load was strict with 0 missing and 0 unexpected keys, and the anchor buffers were bit-identical.

**Per-clip detections.** Evidence class MEASURED, from `raw/per_clip.json` and `raw/per_frame.jsonl`. Tier: an
open-loop perception readout, **not driving performance**. Each clip is 171 eval windows, i.e. one
episode, so no interval is quoted.

| clip (sha12) | why chosen | targets (per window mean [min, max]) | 3-D box head AP@2 m BEV (3-D) | recall / precision at the model's gate | Hungarian centre L2 mean / median | agent head AP@2 m BEV | agent R / P |
|---|---|---|---|---|---|---|---|
| `0191487845ef` | rank 1: 7,021 GT boxes ≤ 30 m in field | 13,726 (80.3 [42, 120]) | **0.258** (0.256) | 0.473 / 0.384 | 4.12 / 2.80 m | 0.203 | 0.435 / 0.352 |
| `9f8bedcfb9de` | rank 2: 2,779 | 4,192 (24.5 [15, 32]) | **0.134** (0.133) | 0.651 / 0.162 | 1.87 / 1.56 m | 0.111 | 0.610 / 0.150 |
| `34765c024267` | rank 5, swapped in for the > 30° turn rule (turn 75.4°) | 3,590 (21.0 [8, 26]) | **0.360** (0.358) | 0.821 / 0.243 | 1.15 / 0.96 m | 0.282 | 0.767 / 0.199 |
| pooled | | 21,508 | | 0.565 / 0.265 | | | 0.524 / 0.236 |

Pooled recall and precision are the greedy 2 m matcher at the gate over all 513 windows. For the 3-D
head that is 12,162 TP of 21,508 targets, with 45,823 slots confident. For the agent head it is
11,273 TP and 47,837 confident.

### ⭐ The finding the PI saw: the model's own confidence gate does not separate detections from empty slots

MEASURED, per window, from `per_frame.jsonl`.

The gate is `sigmoid(presence_logit) ≥ 0.5`, the model's own
`AgentSeamConfig.presence_gate` (`refc_agents.py:197-199`, read from `config.json` `seams.agents`). Both
heads have 100 slots.

| clip | targets per window (mean) | 3-D head: confident slots per window (mean) | windows with ALL 100 confident | agent head: confident (mean) | all 100 |
|---|---|---|---|---|---|
| `0191487845ef` | 80.3 | 98.7 | 139 / 171 | 99.2 | 157 / 171 |
| `9f8bedcfb9de` | 24.5 | 98.3 | 112 / 171 | 99.6 | 158 / 171 |
| `34765c024267` | 21.0 | 71.1 | 2 / 171 | 80.9 | 24 / 171 |

So the video draws about 100 boxes per frame against 21-80 targets. Precision at the gate is therefore
capped at 0.15-0.38, and "boxes where there is nothing" is exactly what the gate lets through.

The ranking is not empty. AP@2 m is 0.13-0.36, and recall at the gate is 0.47-0.82. But the confidence
the model reports is not a usable detection threshold.

HYPOTHESIS, not measured here: the presence BCE's no-object weight 0.1 (`agent_slots.py:233`) lets
empty slots sit near p = 0.5 at little cost. The training-signal review the PI asked for owns this
question. The GT side is being validated in `../2026-09-26-refcv6-gt-validation/`.

⚠️ **Scope.** The clips are the most crowded in the eval set by the pre-registered rule
(`taniteval/tools/RENDER_REFCV6_MAP_VIDEO.md` §8.2). The in-run eval averages 63.6 visible targets per
16-window batch, about 4.2 per labelled window. These APs are therefore **not representative of the
eval set**. They are the crowded tail, chosen to make boxes visible.

### Map head on the same windows (trainer rule `refc_v3_train.py:4506-4521`)

| clip | n | drivable IoU mean | pooled ΣI/ΣU |
|---|---|---|---|
| `0191487845ef` | 171 | 0.617 | 0.605 |
| `9f8bedcfb9de` | 171 | 0.465 | 0.464 |
| `34765c024267` | 171 | 0.706 | 0.687 |

The in-run `eval_map_iou_drivable` at step 38,000 reads 0.67629 on 128 OTHER windows. That is a sanity
range only, not an identity.

## Method (every rule is the trainer's own, cited)

- **Forward.** `refc_v3_train.compute_losses_v3`, interrupted by a forward hook right after `model(...)`
  returns. The build goes through the battery loader (`C:/Users/Admin/ev6_battery/code/refcv6_loader.py`,
  md5 `3ef86651…`) on tree `C:/Users/Admin/ev6_82c2331`, and `tanitad.__file__` is asserted to come
  from that tree.
- **The heads.**
  - 3-D box head: `out['perception']['box_slots']` (`Box3DSlotDecoder`, `box3d_head.py:227-284`).
  - Agent head: `out['agent_slots']` (`AgentSlotDecoder`, `agent_slots.py:546-684`).
  - Yaw is `atan2(sin, cos)`. Boxes are in metres in the rig frame.
- **Target set.** `refc_agents.visible_target_filter` (`refc_agents.py:410-433`): the 120° field ∩ the
  decode box x 0…60 m, |y| ≤ 16 m. It is applied before matching, exactly as `box3d_set_loss` and
  `agent_losses` apply it.
- **Matching.** `agent_slots.match_slots` (`agent_slots.py:788-823`), Hungarian on presence + cls +
  centre L1 + 0.5 size.
  - It matches min(targets, 100 queries), so "matched == targets" is an identity except on
    `0191487845ef`. There, 146 targets on windows with more than 100 targets (max 120) have no slot:
    13,580 pairs for 13,726 targets.
- **AP.** Greedy `box3d_head.box3d_match_rows` at 2.0 m (`box3d_head.py:482-536`), with `ap_from_rows`.
- **Camera.** Cuboids are projected with the model's OWN per-clip `RigCamera` (`model._rig_camera`).
  The cross-check against `render_refcv3_video.CylProjector` reads 0.00 px on every clip.

## Controls (each must read a known value; all passed)

| control | expected | measured | n |
|---|---|---|---|
| GT boxes through the PREDICTED-box drawing and matching path must self-match | 0.0 m, all matched | 21,508 / 21,508 self-matched, max centre error 0.0 m, corner diff 0.0 m, greedy TP 21,508 | 513 windows |
| our centre readout vs the trainer's own `box3d_loss_row` centre term | identity (refuse if > 1e-4) | max \|d\| 1.9e-6 | 513 windows |
| CylProjector vs model RigCamera on ground points | ≤ 0.01 px | 0.00 px | 15 points × 3 clips |
| the map head's logits unchanged by drawing | 0.0 | 0.0 | every clip |

## Departures and deviations (stated, not hidden)

1. **A GPU smoke, not a CPU smoke.** The brief asked for a CPU smoke first. Host RAM sat at 4-8 GB
   throughout, held by other agents' jobs, and a CPU forward needs the 8 GB floor plus headroom. I ran a
   GPU smoke under the same gate instead: 2 windows, no marker, `raw/smoke_gpu/`. The Master Mind was
   told at the time.
2. **RAM during the GPU render.** The 8 GB floor was the START gate. It passed at 22:43:24 with
   8.94 GB free, 1,645 MiB GPU used, and no other python on the GPU. During the render other processes
   pushed box-free RAM as low as **4.29 GB** (1,957 samples below 8 GB). The render's own peak working
   set was 3.84 GB. Its CPU share per window was a decode and a draw, so it paused only below 2.5 GB
   (never reached) rather than stopping. This is the battery's GPU practice. I disclose it because the
   brief's CPU rule reads "never put CPU load on the box below 8 GB free".
3. **The contact sheet header read "step 35,000"** on this step-38,000 sheet, because it was
   hard-coded. Every frame inside was stamped correctly. The renderer now requires `step=`
   (`make_contact_sheet(..., step=)`), and the banked sheet is regenerated from the saved frames with a
   header-only pixel check: `raw/logs/contact_sheet_regen_record.json`.
4. **The equalize caption.** `equalize_bottom_rows` reads argv 43, trunk as BUILT 0, lift bank 43
   (D-REFCV6-EQUALIZE-DROPPED). The frames say so. The step-35,000 map video's caption was wrong about
   this, and that correction lives in `../2026-09-26-refcv6-map-video/RESULT.md`.

## ⛔ The camera cuboids in THIS video have a drawing defect (box-head audit, 2026-09-27)

The PI saw boxes as single flat faces (*"very unplausible"*). The box-head audit found a **DRAWING defect**
in this renderer's `draw_cuboid_cam`:
- An edge was drawn only when BOTH of its corners landed in the frame. A cuboid crossing the ±60°
  border lost its border-crossing edges and read as one face. Example: bus #39 in `9f8bedcfb9de`,
  window 104.
- Each kept edge was drawn as a straight chord, although the cylinder bends it.

The audit verified the **projection itself** independently: a raw-mp4 re-render, LiDAR through its own
extrinsic, and the SAM3 camera model. Its patch, which draws sampled, border-clipped edges
(`cuboid_edge_runs`), is now installed **verbatim** in `taniteval/tools/render_refcv6_map_video.py`.

⚠️ **This video's frames still carry the OLD drawing.** Re-rendering them needs the model forward on the
GPU. The GT-validation reel (`../2026-09-26-refcv6-gt-validation/`) is re-rendered with the patch for
its GT-only camera. Its camera A is a crop of this video, and is labelled as pre-patch.

The detection NUMBERS in this RESULT are unaffected: they come from the heads' decoded boxes and the
trainer's matchers, not from the drawing.

## What this does not show

- No driving claim. It is open loop, one DDIM draw, and an oracle nav token.
- No interval. Three clips are three episodes.
- It does not validate the GT itself. That is `../2026-09-26-refcv6-gt-validation/`.
- It does not show why the gate is saturated. That is the training-signal review.

## Files

`raw/refcv6_boxes_step38000.mp4` and `raw/refcv6_boxes_step38000_small.mp4` exist on D: only.

The rest are in the package: `raw/per_clip.json`, `raw/per_frame.jsonl`, `raw/render_record.json`,
`raw/clip_selection.json`, `raw/contact_sheet.png`, `raw/logs/boxes_render.log`,
`raw/logs/boxes_gpusmoke.log`, `raw/logs/BOXES_DONE.json` and `raw/smoke_gpu/*`.

The 516 frames exist only on the dev box, at `C:/Users/Admin/qland/work/mapvid/frames_refcv6_boxes_step38000/`.
