# render_refcv7_video.py — refcv7-r101-s0 at its FINAL checkpoint, frame by frame

The PI asked: *"Generate a long video showing the performance frame by frame including high quality
visualizations, overlays and bev representation (no static images)"*.

This tool renders every held-out eval window of 12 eval139 clips, at 10 fps (= real time, 0.1 s per
window), for refcv7-r101-s0 at step **50,400**. Each frame shows the camera, a metric BEV with the
predicted 10 cm map beside the SAM3 GT map, the model's decisions, this frame's metrics and running
per-clip time series.

⚠️ **What this reel is not.** It is OPEN-LOOP perception + planning on the held-out eval139 clips. The
emitted plan never drives the car; the next frame comes from the recording. T0 and T1 are both open
loop (`Project Steering/EVAL_DOCTRINE.md`), and nothing here is closed-loop driving. The per-frame and
per-clip numbers are a viewing aid on 12 clips, not a four-family eval result. No interval is quoted,
because one clip is one episode and its windows are not independent.

⚠️ **On this launch tree the speed ceiling does NOT reach the emitted plan (SPEC_REFCV7 §26.1).** The
decoder masks over-ceiling candidates in its own ranking, but the E9 goal selection re-ranks the fan
without that mask. Every frame states whether the emitted plan exceeds the fed ceiling, and the opening
card carries the stamp.

## 1. What runs (provenance)

| item | value | how it is checked |
|---|---|---|
| checkpoint | Thor `/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt` (1,172,814,977 B), the final rolling checkpoint | full md5 must equal `d5f104ee54ba6b2861e38030b4f6fcf1`; `ck["step"]` must equal 50,400; `summary.json` must say `done` at 50,400; the load is **strict** (0 missing / 0 unexpected); `param_breakdown` must equal `config.json`'s; the anchor file must equal the checkpoint's anchor buffers |
| config | the run's own `config.json` (full argv) | md5 recorded |
| code tree | the LAUNCH tree `/home/nvidia/refcv7_run/fec3a0dccf` (`PYTHONPATH=<tree>/stack`) | `tanitad.__file__` and `four_families.__file__` are asserted to lie under the tree |
| model build | `<tree>/stack/tanitad/eval/refcv7_loader.py::build_model` (the launch gate's G-EVAL loader: replays `train()` block by block, G-DVB, stamp checks) with `REFCV6_KIT=/home/nvidia` | the loader's own refusals, plus the checks above |
| eval windows | `refcv7_loader.build_eval_dataset(..., with_perception_targets=True)` over `refcv6-b1-416x1024-eval139` with the v8 eval labels, the 10 cm SAM3 GT (`sam3_gt_v3`), the agent join, the 3-D join and the VIS-1 sidecar | the trainer's own `V3Dataset` window contract |
| forward | `refc_v3_train.compute_losses_v3(model, batch, "cuda")` in eval mode, one window per batch, with a **non-raising** forward hook that captures `out` | the hook must fire; the loss path's own eval extras supply the map per-class counts and the detection packs |
| inference draw | `torch.manual_seed(0)` before every window: the plan is ONE DDIM draw | §5 (c) |

**Departures, all recorded in `render_record.json`:** the loader's own (`--trunk-compile` removed —
torch.compile wraps the backbone call only; Thor paths remapped to the kit, which on Thor IS
`/home/nvidia/data`; the tactical-goal pos_weight / mask from the run's stamp). Nothing else: the full
loss path runs, including the future-frame decode the LAW term reads.

## 2. The clip rule — fixed BEFORE rendering

The rule reads GT and labels only. It is applied in the forward process before the first model output
exists, and it is written to `clip_selection.json` (sha12s and counts only). A second full run recomputes
it and refuses if it differs.

> **Candidates:** the eval139 clips whose EVERY eval window has a 10 cm SAM3 GT frame
> (`perception_targets.require_map_coverage` on the dataset's own fine store, `n_ok == n_windows`), with a
> v7 nav token.
>
> **Per nav token** (`left`, `right`, `follow`): the token's candidates are split at their median
> clip-mean measured speed (v0 at each window's NOW) into LOW (`<` median) and HIGH (`>=` median).
>
> **In each half, two clips:**
> 1. the clip with the most **VRU windows** — a GT person / rider / stroller within 30 m inside the
>    camera's 120° field at the window's NOW;
> 2. of the rest, the clip with the most **lead-vehicle windows** — a GT vehicle-class box with
>    0 < x ≤ 60 m and |y| ≤ 1.75 m at NOW.
>
> **Ties, and a criterion no clip in the half satisfies, fall back to the smallest sha12.**
>
> 4 clips per token, 12 in all, in the order left / right / follow × low / high × VRU / lead. Every eval
> window of each clip is rendered in time order.

⚠️ **The ranking favours busy scenes by design** (the brief asks for VRU and lead-vehicle coverage), so the
12 clips are NOT a random sample of eval139 and their averages must not be read as the split's. The token
medians that split the speed halves were 7.49 m/s (left, 12 clips), 7.37 m/s (right, 37) and 11.88 m/s
(follow, 88) — the LOW and HIGH labels are relative to each token's own clips.

The tool refuses if the chosen set has no VRU clip or no lead-vehicle clip. The rule is a pure function
(`select_clips`) pinned by `taniteval/tests/test_render_refcv7_video.py` with literal expected picks.

## 3. What is on every frame (1920 × 1080)

The standing viz standard applies — camera projection, metric BEV and text overlay together on every
frame — and every frame is validated by `tanitad.viz_standard.check_frame`.

1. **FRONT CAMERA** (top left, 1024 × 416, native): the window's last input frame,
   `item["frames"][-1][-3:]`, the exact uint8 bytes the trunk is handed. Overlaid:
   * the GT future path (green, dots every 1 s) and the EMITTED plan `out["traj"]` (orange; filled =
     the 1 s slots, hollow = 0.5 / 1.5 s), through `render_refcv3_video.CylProjector` with the clip's
     MEASURED extrinsics;
   * the top-8 other candidates of the same fan by the E9 score `sel_score_v3`, faded cyan;
   * GT cuboids (green: solid = VIS-1 positive, thin = IGNORE) and the 3-D box head's cuboids at the
     display threshold (§4) in class colour, with class and score. Cuboids go through the model's own
     per-clip `RigCamera` and `render_refcv6_map_video.draw_cuboid_cam` (the box-head audit's
     edge-sampled drawing);
   * a horizon line predicted from the extrinsics (a self-check) and the bottom rows the trunk zeroes
     as trained and the lift treats as unobserved, captioned from the BUILT modules.
2. **BEV — PREDICTED 10 cm MAP** and **BEV — SAM3 GT MAP** (top right, side by side, 100 m ahead ×
   ±30 m, 6.73 px/m, forward is up, LEFT IS LEFT). The prediction is `map_head_hires.decide(logits,
   "prior_corrected", class_weight)`, the decision rule the run declared and trained with. On both
   panels: range rings every 10 m, lateral guides, the GT path with 1 s ticks, the plan with its slots,
   the faded fan, GT boxes and predicted boxes ≥ threshold with their scores, and a white link for each
   true-positive pair.
   * **Full brightness = a scored cell** (SAM3 saw it AND the lift reaches it now) — the SAME set on
     both panels. Prediction panel: dimmed where SAM3 has no label, dotted hatch where the lift cannot
     reach. GT panel: dimmed where the lift cannot reach, diagonal hatch where SAM3 never saw.
3. **INPUTS · DECISIONS** (middle left): ego speed v0 (measured at t0), the fed max speed (sidecar value →
   the set-speed/ceiling step the model's own `max_speed_1h_v6` makes of it), the nav token (a GIVEN
   INPUT, amber), the emitted plan's peak planned speed against the ceiling (red when it EXCEEDS), how many
   fan candidates are over the ceiling and whether the decoder's own pick obeys it; the refcv6 tactical
   decoder's lat / lon decision with probability against the v7 GT where labelled; the top-3 goal tokens;
   strategic UNAVAILABLE.
4. **THIS FRAME (vs GT)** (middle): ADE over the 8 slots, FDE at 6 s, speed / heading / curvature error
   on 0–2 s, the per-class map IoU of this frame (bars, with the union size), and the box matches.
5. **Running time series for the clip** (bottom, four charts, cursor = now): plan error (ADE, FDE); speed
   (measured v0, the plan's peak speed, the fed ceiling); map IoU per class (drivable, sidewalk, lane,
   crosswalk, edge, hatched); boxes (VIS-1 positives, detections ≥ threshold, true positives).
6. **Cards:** a 10 s opening card (model, step, ckpt md5, tier stamp, ceiling stamp, thresholds, how to
   read a frame, known weaknesses) and a 3.5 s title card per clip (sha12, nav token, why chosen, the
   clip's metric summary and its controls).

**Colour semantics, one meaning per colour:** green = ground truth (paths, boxes); orange `(249,115,22)` =
the model's emitted plan; faded cyan = other candidates; amber = a GIVEN input (text only; the orange is
> 90 L1 from it). No map or agent class colour sits within 60 of the green or the orange (asserted).

**UNAVAILABLE, declared on every frame:** `strategic` — `--no-strategic`: the strategic level is OFF in this
arm, and PhysicalAI-AV carries no strategic label. `tactical_gt` is declared UNAVAILABLE on a window
outside the label record's band (`lat_v7` / `lon_v7` = IGNORE).

## 4. Every number and where it comes from

| on screen | source |
|---|---|
| ADE / FDE | `‖out["traj"] − refb_labels.waypoint_targets(pose_last, future_poses_ext, horizons)‖`, mean over the VALID of the 8 slots / the 6 s slot only when valid |
| speed, heading, curvature error 0–2 s | `taniteval.four_families._seq_geometry` on the first 4 slots (a 0.5 s grid), plan vs GT, masked as that function masks |
| peak planned speed, ceiling | `refcv6_selection.planned_max_speed(out["traj"])` vs `refcv6_max_speed.limit_ms_of_bin(model.max_speed_1h_v6(v_max_ms, v_max_valid))` — the same functions the filter uses |
| map IoU per class | `map_head_hires.per_class_signal` inter / union under `prior_corrected` on seen & lift-valid cells. ⭐ **Identity-checked on every window against the trainer's own eval extras** (`map_hires_inter_*` / `map_hires_union_*`); the render refuses on a difference > 1e-6 |
| boxes | the trainer's own eval detection pack `_det_pack_box3d` (VIS-1 positives, IGNORE = DontCare), matched by `detection_metrics.greedy_rows` at 2 m — the A9 P0 rule. The pack's logits / centres / GT rows are identity-checked against `out["perception"]["box_slots"]` and the batch |
| tactical | argmax of `out["tacv6_lat_logits"]` / `out["tacv6_lon_logits"]` (the refcv6 tactical decoder the battery scores) vs `item["lat_v7"]` / `item["lon_v7"]` |

**The box display threshold.** The declared detection gate is `sigmoid(presence) ≥ 0.5`
(`slot_presence.DETECTION_GATE`). At step 50,400 it passes almost nothing: the in-run eval reads
`eval_box3d_conf_ratio` 0.0059 and `eval_box3d_rec@gate` 0.0059 (metrics.jsonl). Boxes are therefore drawn
at the run's OWN P = R operating point, `eval_box3d_calib_pr_gate` = **0.2589**, which
`detection_metrics.pr_equal_gate` measured on 256 TRAIN calibration windows. It is read from
`metrics.jsonl` at run time; it is not tuned on the rendered clips. Each frame also prints the counts at
the declared gate 0.5 and the frame's maximum presence score, and every drawn box carries its score.

## 5. Controls

* **(a) Identities (every window):** the map per-class counts recomputed by the module equal the trainer's
  extras; the supervised-cell count equals `n_map_hires_cells`; the detection pack's logits and centres
  equal `box_slots` exactly; the pack's GT rows equal the batch's valid rows; `out["traj"]` equals the fan
  member `anchor_traj[sel_idx]` exactly.
* **(b) Cameras (every clip):** `CylProjector` and the model's `RigCamera` agree on ground points to
  ≤ 0.01 px; a point 5 m LEFT projects left of centre and 5 m RIGHT right of it; the render stage rebuilds
  each `RigCamera` from the banked R / t / frame and must reproduce the model's own projections of four
  3-D probe points exactly.
* **(c) Draw independence (first window of each clip):** the forward is repeated with seed 1; the map
  logits and box presences must not move, and the plan movement is reported.
* **(d) Orientation:** the GT path's points must land on SAM3 road surface (drivable / lane / crosswalk /
  arrow) at least as often under the file contract as under a lateral mirror; reported per clip. The
  panel transform is pinned by unit tests with a lateral-mirror mutation that goes RED.
* **(e) Content:** every frame must carry a non-blank camera region and BEV panel, and the encoded reel
  is decoded back and its frame count asserted.

## 6. Usage (Thor)

```
T=/home/nvidia/refcv7_run/fec3a0dccf
PY=/home/nvidia/venvs/tanitad-train/bin/python
cd /home/nvidia/refcv7_post/video
# 1. forward (GPU) -- always under the shared lock
flock /home/nvidia/refcv7_post/thor_gpu.lock env OMP_NUM_THREADS=6 PYTHONPATH=$T/stack \
    $PY code/render_refcv7_video.py --stage forward          # --smoke: 1 clip x 6 windows
# 2. render + encode (CPU only, no lock)
env OMP_NUM_THREADS=1 PYTHONPATH=$T/stack $PY code/render_refcv7_video.py --stage render
```

Outputs: `bank/final/` (per-window npz + `rows.jsonl` + `forward_record.json`), `out/final/`
(`refcv7_final_50400_reel.mp4`, `refcv7_final_50400_preview.mp4`, `render_record.json`,
`clip_selection.json`, `keyframes/`). Thor has no ffmpeg binary; the encoder is libx264 through PyAV.

## 7. Results of the 2026-10-04 render

Every number below is MEASURED, from `Media/refcv7/render_record.json` (md5 `3977e2c9f4420faf062ec61c05300a23`)
and the Thor bank `/home/nvidia/refcv7_post/video/bank/final/forward_record.json`. Tier: OPEN-LOOP, held-out
eval139, one DDIM draw per window (seed 0). Estimators: plain means over a clip's windows; IoU pooled
(sum ∩ / sum ∪); no interval, because one clip is one episode.

**The video.** `refcv7_final_50400_reel.mp4`: 1920 × 1080, H.264 yuv420p, 10 fps, **2,579 frames = 257.9 s
(4 min 18 s)** — a 10 s opening card, 12 × 3.5 s title cards, and **2,059 window frames**. 155,934,453 B, md5
`9ce0ca03335fa0e4dd4116d8b406e47d`. ffprobe on the dev box reads 2,579 frames, and the tool's own decode-back
reads 2,579. The preview `refcv7_final_50400_preview.mp4` is 1280 × 720, the same 2,579 frames, 12,952,054 B
(≤ 15 MB), md5 `70bdf9f8c5ff8d57415f580d40731ca1`. Consecutive windows are 0.100–0.101 s apart on the label
clock, so 10 fps is real time.

**The clips the rule chose** (`clip_selection.json`): 137 of 139 eval clips have full 10 cm coverage; the 2
others have no GT file.

| # | sha12 | slot | n | ADE m | FDE 6 s m (n) | plan > ceiling | IoU drivable | sidewalk | lane | crosswalk | edge | hatched | box TP / det / VIS-1 pos @ 0.259 | R | P | det @ 0.5 | tac lat | tac lon | orientation file / mirror | seed 0→1 plan move m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `693335b810c3` | left/low/vru | 171 | 0.69 | 2.30 (132) | 0 | 0.59 | 0.08 | 0.01 | 0.03 | 0.000 | 0.000 | 714 / 2021 / 931 | 0.77 | 0.35 | 0 | 40/40 | 36/40 | 0.998 / 0.998 | 0.59 |
| 2 | `a78bf2711918` | left/low/lead | 171 | 3.96 | 13.57 (132) | 6 | 0.42 | 0.23 | 0.01 | 0.02 | 0.000 | 0.000 | 194 / 660 / 914 | 0.21 | 0.29 | 0 | 40/40 | 40/40 | 0.953 / 0.774 | 1.15 |
| 3 | `f6cfd0a20340` | left/high/vru | 177 | 3.10 | 11.07 (138) | 0 | 0.53 | 0.26 | 0.00 | 0.00 | 0.000 | — | 497 / 1586 / 1296 | 0.38 | 0.31 | 0 | 36/40 | 0/40 | 0.998 / 0.887 | 0.58 |
| 4 | `9855363e1ac1` | left/high/lead | 171 | 3.62 | 15.14 (132) | 0 | 0.49 | 0.55 | 0.06 | 0.08 | 0.000 | 0.000 | 135 / 341 / 334 | 0.40 | 0.40 | 42 | 0/40 | 15/40 | 0.999 / 0.829 | 2.10 |
| 5 | `0191487845ef` | right/low/vru | 171 | 2.34 | 7.93 (132) | 0 | 0.43 | 0.18 | 0.01 | 0.01 | 0.000 | 0.000 | 1543 / 5367 / 3700 | 0.42 | 0.29 | 0 | 0/40 | 40/40 | 0.969 / 0.889 | 1.38 |
| 6 | `9f8bedcfb9de` | right/low/lead | 171 | 1.36 | 4.72 (132) | 1 | 0.38 | 0.18 | 0.01 | 0.01 | 0.000 | 0.000 | 658 / 2117 / 1307 | 0.50 | 0.31 | 59 | 40/40 | 33/40 | 0.909 / 0.909 | 0.00 |
| 7 | `751cdd67811f` | right/high/vru | 171 | 1.94 | 7.59 (132) | **50** | 0.48 | 0.29 | 0.01 | 0.03 | 0.000 | 0.000 | 365 / 1134 / 1504 | 0.24 | 0.32 | 0 | 24/40 | 40/40 | 0.999 / 0.727 | **5.26** |
| 8 | `49f68312b36e` | right/high/lead | 171 | 0.99 | 3.49 (132) | **39** | 0.56 | 0.36 | 0.01 | 0.13 | 0.000 | — | 556 / 1769 / 1609 | 0.35 | 0.31 | 0 | 40/40 | 0/40 | 0.997 / 0.966 | 0.28 |
| 9 | `7f6670448ef6` | follow/low/vru | 172 | 1.62 | 6.06 (133) | 10 | 0.53 | 0.37 | 0.07 | 0.03 | 0.000 | 0.000 | 599 / 2073 / 1785 | 0.34 | 0.29 | 0 | 6/39 | 0/39 | 0.998 / 0.708 | 0.64 |
| 10 | `198bb4517e28` | follow/low/lead | 171 | 1.50 | 5.96 (132) | 0 | 0.59 | 0.32 | 0.00 | — | 0.000 | — | 773 / 2189 / 1344 | 0.58 | 0.35 | 56 | 40/40 | 34/40 | 1.000 / 1.000 | 0.67 |
| 11 | `e01bd05b8e43` | follow/high/vru | 171 | 0.87 | 2.80 (132) | 4 | 0.73 | 0.67 | 0.07 | 0.05 | 0.000 | 0.000 | 190 / 670 / 887 | 0.21 | 0.28 | 0 | 0/40 | 29/40 | 1.000 / 0.946 | 0.62 |
| 12 | `37616735658b` | follow/high/lead | 171 | 2.30 | 9.38 (132) | 0 | 0.72 | 0.07 | 0.03 | — | 0.000 | — | 155 / 400 / 878 | 0.18 | 0.39 | 0 | 22/40 | 16/40 | 1.000 / 1.000 | 0.51 |

(— = the class has no union cell in the clip. Tactical columns count windows inside the label record's band
only, ~40 per clip, one episode each — a viewing aid, not an accuracy.)

**What the viewer can see** (each checked by eye on extracted frames):
* **The ceiling defect is visible.** The emitted plan exceeds the fed ceiling on **110 of 2,059** windows
  (50 on clip 7, 39 on clip 8); the frame turns the line red and the speed chart shows the orange peak above
  the dashed ceiling.
* **The thin map classes are not predicted**: edge IoU 0.000 on every clip, hatched 0.000 or absent; lane
  ≤ 0.07. The GT panel shows lane lines and edges the prediction panel never draws.
* **Box confidence is low but the boxes are placed**: at the declared gate 0.5 only 3 clips produce any
  detection (42 / 59 / 56); at the P=R point 0.2589 the 3-D head reaches recall 0.18–0.77 and precision
  0.28–0.40 per clip, and on frames checked by eye the cuboids sit on the parked cars, the lead car and the
  pedestrian crowds.
* **Turn failures are visible**: on clip 3 (nav LEFT) the GT turns left into a side street at ~20 m while the
  emitted plan goes straight (window 109: ADE 9.93 m, FDE 29.52 m), the same on camera and BEV.
* **Inference variance is visible**: on clip 7 a second seed moved the plan 5.26 m (anchor 58 → 67).

**Controls.**
* Identities over all 2,059 windows: map per-class counts vs the trainer's extras, detection-pack logits /
  centres / GT rows vs `box_slots` and the batch, `out["traj"]` vs `anchor_traj[sel_idx]` — **max |Δ| 0.0
  on every one**. 2,059/2,059 windows carried an agent label and a detection pack.
* Cameras: `CylProjector` vs the model's `RigCamera` **0.0 px** on all 12 clips; the left/right control
  passed on all 12; the render stage's rebuilt `RigCamera` reproduced the model's 3-D probe projections
  exactly.
* Draw independence: map logits and box presences moved **0.0** under a second seed on all 12 clips; the
  plan moved 0.00–5.26 m.
* Orientation: the GT path lies on SAM3 road surface at least as often under the file contract as under a
  lateral mirror on 12/12 clips (strictly more on 8; equal on the straight/symmetric 4).

**Compute.** Forward on Thor: 1,609 s wall, **793 s of GPU forward** (0.69 s per window including the item
fetch), `torch.cuda.max_memory_allocated` 1.43 GiB, under `flock thor_gpu.lock`. Render + encode on Thor CPU
(8 workers, no GPU): 237 s. Thor has no ffmpeg; the encoder is libx264 through PyAV 18.1.0.

⚠️ **Two blobs of this file ran.** The forward ran with md5 `189fffcd5a134a199d638ef09aa2f422`; the final
render with `abca55c91c7408644483554b9f282787` (= the file in the repo). The difference was AST-compared:
only `render_frame` and `title_card` changed (text layout); `stage_forward`, every helper it calls and all
module-level constants are identical. Both blobs are kept on Thor under `/home/nvidia/refcv7_post/video/code/`.
