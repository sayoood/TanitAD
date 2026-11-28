# render_refcv7_replay.py — an interactive REPLAY TOOL and a TACTICAL video for refcv7-r101-s0 (step 50,400)

The PI, after the first reel (`RENDER_REFCV7_VIDEO.md`), verbatim: *"I need a replay tool where the different labels
can be selected and switched on/off. It is very hard to recognize what is GT and what is from the model. I need a video
where we can see the outputted tactical behavior and goals. The fan is sometimes completely messy."*

⚠️ **What this is not.** It is OPEN-LOOP perception + planning on the held-out eval139 clips. The emitted plan never
drives the car; the next frame comes from the recording (`Project Steering/EVAL_DOCTRINE.md`). The per-frame numbers are
a viewing aid on 12 clips, not a four-family eval result, and no interval is quoted (one clip is one episode).

⚠️ **On this launch tree the speed ceiling does NOT reach the emitted plan (SPEC_REFCV7 §26.1).** The decoder masks
over-ceiling candidates in its own ranking; the E9 goal selection re-ranks the fan without that mask. Every planner
overlay in the tool and the video carries the stamp *"speed ceiling not applied to the emitted plan (SPEC_REFCV7 26.1)"*
and turns red on a frame where the emitted plan exceeds the fed ceiling.

## 1. One colour rule, everywhere (the answer to "what is GT and what is the model")

| | colour | line / fill | tag |
|---|---|---|---|
| **GT** | green `(110,231,138)` | dashed or outline only; boxes are outlines | **GT** (green pill) |
| **MODEL** — emitted plan (E9 pick) | orange `(249,115,22)` | solid | **MODEL** (orange pill) |
| **MODEL** — decoder's own pick (when different), candidates, detections | blue / blue ramp | solid, detections FILLED with their score | **MODEL** |
| **MODEL** — tactical geometric goal `g_tac` | fuchsia diamond | filled | |
| GIVEN input (nav token, measured speed, fed ceiling) | amber | text only | |

A fixed legend sits on every screen, both families carry a tag on every panel, and the tool's **Both / GT only /
Model only** switch (keys `b g m`) hides one family completely. Asserted by `test_gt_green_is_not_close_to_any_model_colour`.
The fan ramp (light blue → indigo) contains no green and no orange.

## 2. What the extended capture reads (and what does not exist)

`--stage capture` is the reel's forward pass (same loader, same clip rule, same `compute_losses_v3` + non-raising forward
hook, same seed-0 DDIM draw) with read-only reads added. It **reproduces the reel exactly** (§6), so the replay tool and the
reel show the same plans.

| captured per window | source (launch tree `fec3a0dccf`) |
|---|---|
| lat / lon posteriors (8 + 8, softmax) | `out["tacv6_lat_logits" / "tacv6_lon_logits"]` — the refcv6 behaviour decoder (`refcv6_tactical.py::TacticalBehaviourDecoder`) |
| all 22 goal-token validities + confidences | `sigmoid(out["tacv6_goal_logits"])`, `sigmoid(out["tacv6_goal_conf"])` |
| GT: lat/lon label, goal positives, which tokens are scored | `item["lat_v7" / "lon_v7"]`, `item["tac_goal_y"]`, `item["tac_goal_w"] > 0` |
| **spatial tactical goals** `g_tac` (x, y, heading, speed at 2 / 4 / 6 s) + GT label | `out["g_tac"]` `[3, 4]` (the hierarchy's goal head, `refc_v3.py:1587`), `batch["goal_tac"]`, `batch["goal_tac_valid"]` |
| ALL 117 candidates `[117, 8, 2]` | `out["anchor_traj"]` (int16 centimetres in the data files) |
| E9 score / decoder's own score | `out["sel_score_v3"]` / `out["sel_score"]` |
| reach mask | `out["reach_keep"]` |
| ceiling mask | `planned_max_speed(fan) <= v_lim`, **identity-checked against `refcv6_selection.SpeedCeilingFilter`** (the filter the decoder calls) |
| nav-compliance flag per candidate | `refcv6_selection.nav_compliance_prior(fan, nav_cmd, tau_rad=navc_tau)` (τ = 0.18064 rad) |
| both picks | `out["sel_idx"]` (E9, the emitted plan) and `out["sel_idx_base"]` (the decoder's own pick) |
| distance of each candidate to the 2 s goal | `out["goal_dist"]` |

**Found, and stated on the page:**
* **The behaviour decoder emits NO spatial goal cells or points.** Its outputs are 22 validity logits, 22 confidence
  logits and the lat/lon posteriors (plus a cross-attention map, which the forward does not export). The only spatial
  goals are the `g_tac` points above, produced by the hierarchy's goal head; E9 selects by the distance of each candidate's
  2 s point to `g_tac[0]`. They are drawn as fuchsia diamonds with heading ticks, against the GT label as green rings.
* **Nav compliance is a gated score term, not a hard mask, on this tree** (`navc_gate` 0.1625 added into `sel_score`).
  The tool's "complies with nav" filter is therefore a *flag filter*, labelled as such; the hard masks are reach and ceiling.
* **`out["traj"]` is `anchor_traj[sel_idx]` exactly** (no refcv7 WTA candidates reach the pick on this run; `r7_candidates`
  never appears in `out`), so "117 candidates" is the whole fan.

## 3. The replay tool

`index.html` + `replay.css` + `replay.js` + `data/` — static, **opens from disk by double-click** (`file://`), no server, no
`fetch`/XHR. Data arrives as `<script src>` files (`data/manifest.js`, `data/clip_NN.js`) and `<img>` JPEGs.

* **Clip selector** (12 clips), timeline scrubber, play/pause at 10 fps real time (0.25× … 2×), step ±1, loop. Keys: `space`,
  `← →` (shift ±10), `[ ]` previous/next clip, `b g m`.
* **Decision strip** under the toolbar: MODEL and GT lat/lon decision for every window of the clip, a red cap wherever they
  differ, hatching where the GT is not labelled, two more rows for *plan vs tactical* and *plan vs nav*. Click to seek.
* **Camera canvas** and **BEV canvas(es)** (100 m ahead × ±30 m, forward up, left left), overlays drawn client-side from
  metric coordinates. Camera pixels come from a JavaScript port of the model's `RigCamera` cylindrical projection
  (`φ = atan2(x, z)`, `col = cx + f·φ`, `row = cy + f·y/ρ`).
* **Layers (each a checkbox or slider):** GT path · GT boxes (VIS-1 positives; IGNORE/hidden separately) · GT tactical goals ·
  GT map · emitted plan · decoder's pick · fan (top-k slider 1…117, rank by E9 or by the decoder's score, colour by rank or by
  score, "only candidates kept by" reach / ceiling / nav-compliance, rank labels) · detections with a score slider (default
  **0.2589**, the TRAIN P=R gate; min 0.10) · NMS toggle with a radius slider (default 2 m, BEV centre distance, class-agnostic) ·
  `g_tac` goals · predicted map. **Maps:** per-class chips (8), opacity slider, *side by side (MODEL | GT)* or *overlay (MODEL fill
  + GT outline)*, dimming of unscored cells.
* **Tactical panel:** lat and lon probability bars (all 8 classes each) with the GT label marked, all 22 goal tokens sorted by
  validity with GT positives marked (tooltip: validity and confidence; dimmed = an unscored token), the `g_tac` table (MODEL vs GT),
  the nav command, ego speed, fed max speed → ceiling step, the emitted plan's peak speed against the ceiling, and three
  agreement chips (§5).
* **Per-frame numbers:** ADE / FDE, speed / heading / curvature error (0–2 s, `four_families._seq_geometry`), map IoU per class,
  and box TP / FP / DontCare / recall / precision **recomputed live** for the current threshold and NMS (the greedy 2 m rule,
  ported to JS and checked against Python — the **self-test** button). The fan table lists the top candidates with E9 / decoder
  score, goal distance and the three flags, plus a "how messy" scalar: the mean pairwise distance of the 6 s endpoints over
  all candidates / reach-kept / also under the ceiling / the shown top-k.

### Data schema (compact window record, `data/clip_NN.js`)

`t, v0, nav, vmr, vmv, ck, vl` (time, speed, nav, fed max speed, ceiling step km/h, limit m/s) · `tr` plan `[8,2]` · `sel, core` picks ·
`fan` base64 int16 cm `[117,8,2]` · `e9, dc` scores · `rch, ceil, navc` bit strings · `nci` nav informative · `vmx` planned peak speed
per candidate · `gd` goal distance · `pvm, pex` plan peak speed / exceeds · `gtd` GT dense path (valid prefix) · `latp, lonp, latg, long_` ·
`gp, gcf, ggt, gsc` goal validity / confidence / GT positives / scored mask · `gtac, gtacg, gtacv` · `p3` plan agreement · `m` metrics ·
`bx, gb` model and GT boxes · `dt` Python match pairs at the gate. `mp[i] = [pred_png_b64, gt_png_b64]`: 8-bit grayscale PNG, pixel =
`class | 8·not-seen | 16·lift-invalid` (the browser reads the byte straight from `getImageData`; the maps travel as data URIs so the
canvas stays untainted on `file://`).

## 4. The tactical video — `refcv7_final_50400_tactical.mp4`

1920×1080, 10 fps, H.264, all 12 clips, every eval window, 10 s opening card + 3 s title card per clip. Camera: model plan (orange)
+ GT path (green, dashed, drawn wider so a path under the plan still shows) + the **top-3 fan only** (by the E9 score, rank 1 is the
plan) + NMS'd boxes at the P=R gate (GT outline green, model filled blue with score, `TP` where matched) + `g_tac` goals. BEV: same.
**Large tactical panel**: lat and lon bars with GT marked, the 22 goal tokens with probabilities, the agreement chips, and the
whole-clip decision strip with a moving cursor. Right column: inputs, ceiling, this frame's numbers, `g_tac` table, boxes, fan.

## 5. The agreement chips are DERIVED — and controlled

"Plan agrees with nav / with the tactical decoder" is not a programme output; this tool derives it:

* **plan → (lat3, lon3):** `refc_tactical.factor_from_kinematics` v2 rule at the label's own **2 s** horizon on the plan's first four
  slots: `dyaw` = heading of the last 0.5 s segment (stall < 0.05 m reads 0), `κ = dyaw / arc`, turn iff `|κ| ≥ 1/60`; `dv = v₁ − v₀`,
  accelerate iff `dv > +1`, brake_stop iff `dv < −1` or (`v₁ < 0.3` and `v₀ ≥ 1`). Class ids are the programme's
  (`lane_keep, turn_left, turn_right` / `brake_stop, steady, accelerate`). The numpy port is **identity-checked against the
  programme's function on every window**.
* **decoder action → lat3/lon3 comparison:** tables `LAT_TO_LAT3` / `LON_TO_LON3` (data, not logic). Nudges, lane changes and
  `ABORT_LC` have no lat3 counterpart → the chip reads *n/a*, never *disagree*. `HOLD` and `ADAPT_SPEED_FOR_CURVE` accept steady or
  brake_stop; `CREEP` steady or accelerate.
* **nav:** `left`/`right` → the programme's `nav_compliance_prior` predicate evaluated on the emitted path (terminal heading of the
  commanded sign, |·| ≥ τ = 0.18064 rad); `follow` → agrees iff the plan does not turn.
* **Known-value control:** the SAME rule applied to the GT path and scored against the GT v7 labels through the SAME tables (see
  §7 for the counts). It is coarse by construction; it is a reading aid, not a metric.

## 6. Controls (every one MEASURED; see §7)

* **Reproduction of the reel:** the capture's trajectories, `sel_idx`, `core_sel_idx` and predicted maps are compared with the reel's
  bank window by window.
* **Identities:** map per-class counts vs the trainer's extras; detection-pack logits/centres/GT rows vs `box_slots` and the batch;
  `out["traj"] == anchor_traj[sel_idx]`; the E9 pick and the decoder pick are re-derived from the exported scores
  (`argmax(E9 | reach)`, `argmax(decoder | reach & ceiling)`); the ceiling mask equals `SpeedCeilingFilter`'s; the numpy `plan_factor`
  and `greedy_match` equal the programme's functions.
* **Cameras:** `CylProjector` vs `RigCamera` ≤ 0.01 px; left/right control; the **JavaScript projection** reproduces the model's own
  `RigCamera` on 68 full-precision 3-D probe points per clip (self-test button; Python side regenerates the probes deterministically).
* **Draw independence** (first window per clip): a second seed leaves map logits and box presences unchanged.
* **Page wiring:** `test_every_layer_key_has_a_control_and_every_control_has_a_key`; browser test that every checkbox, slider, select,
  class chip and key changes the canvas.

## 7. Usage (Thor)

```
T=/home/nvidia/refcv7_run/fec3a0dccf ; PY=/home/nvidia/venvs/tanitad-train/bin/python
cd /home/nvidia/refcv7_post/replay          # code/ holds render_refcv7_replay.py, render_refcv7_video.py, refcv7_replay/
# 1. capture (GPU) -- always under the shared lock                         (--smoke: 1 clip x 6 windows)
flock /home/nvidia/refcv7_post/thor_gpu.lock env OMP_NUM_THREADS=6 PYTHONPATH=$T/stack $PY code/render_refcv7_replay.py --stage capture
# 2. the tool's data and the video (CPU only, no lock)
env OMP_NUM_THREADS=1 PYTHONPATH=$T/stack $PY code/render_refcv7_replay.py --stage assets --workers 8
env OMP_NUM_THREADS=1 PYTHONPATH=$T/stack $PY code/render_refcv7_replay.py --stage video  --workers 8
```
Outputs under `out/final/`: `tool/` (the static page + `data/`), `video/refcv7_final_50400_tactical.mp4`, `video/video_record.json`,
`bank/final/capture_record.json`. The 720p (≤ 15 MB) and 540p (≤ 5 MB) versions are two-pass `ffmpeg` re-encodes of the 1080p file.

## 8. Results of the 2026-10-04 run

Every number is MEASURED, from `Media/2026-10-04-refcv7-replay/REPLAY_CAPTURE_RECORD.json` (md5 `ccf73246f601e3e5ec4ab417d4b172b4`),
`REPLAY_ASSETS_RECORD.json`, `video_record.json` and the Thor bank `/home/nvidia/refcv7_post/replay/bank/final/`. Tier: OPEN-LOOP,
held-out eval139, one DDIM draw per window (seed 0). Estimator: counts and plain means; no interval (one clip is one episode).

**Compute.** Capture on Thor under `flock thor_gpu.lock`: 1,612 s wall, **792.5 s of GPU forward**, `max_memory_allocated` 1.426 GiB, 2,059
windows (the reel's 27 min). Assets: 37 s CPU. Video: 117 s CPU (8 workers, PyAV libx264). Nothing in the run dir was modified.

**Reproduction of the reel (the control that licenses reusing its numbers).** Over all **2,059** windows: trajectory max |Δ| **0.0 m** against the
reel's bank, `sel_idx` differs on **0**, `core_sel_idx` on **0**, predicted 10 cm maps differ on **0**. Identities, all max |Δ| 0.0:
map per-class counts vs the trainer's extras, detection logits/centres/GT rows vs `box_slots`, `traj` vs `anchor_traj[sel_idx]`. E9 pick
re-derived from the exported scores on 2,059/2,059, decoder pick on 2,059/2,059; numpy `plan_factor` equals `factor_from_kinematics` on
2,059/2,059; numpy `greedy_match` equals `detection_metrics.greedy_rows` on 2,059/2,059. No fan value was clipped by the int16 encoding.

**The ceiling mask has 31 documented exceptions.** `ceil_mask_equals_planned_speed` reads 2,028/2,059: the other 31 windows (all in clip 7, the first at window 129) are
the decoder's empty-survivor rows — NO candidate satisfies the 13.9 m/s ceiling, so `SpeedCeilingFilter` keeps the whole fan. The tool's ceiling
filter applies the same rule and says so on the page.

**What the data shows (descriptive; viewing aid, not an eval).**
* The 117-candidate fan is wide by construction: the mean pairwise distance of the 6 s endpoints is **35.7 m median (p90 57.1, max 76.4)**; over the
  reach-kept candidates under the ceiling 20.3 m (p90 42.5); the top-3 by E9 only **9.7 m (p90 11.6)**, the top-8 12.8 m. "Messy" is the full fan
  (or the top-k for large k); the top-3 is tight. The reach mask removes at most 8 of 117 candidates (min kept 109); the ceiling keeps a per-clip median of 71-117 of 117 (clip 4: all 117).
* **E9 re-ranks the decoder's pick in 217/2,059 windows (10.5 %)** — the two picks are different candidates there (clip 8: 46/171).
* The emitted plan exceeds the fed ceiling on **110/2,059** windows (clip 7: 50, clip 8: 39), as in the reel.
* Tactical decoder argmax vs the v7 GT, labelled windows only (479 = 12 × ~40): **lat 288/479, lon 283/479**. By clip the failures are whole-clip:
  e.g. lat 0/40 on clips 4, 5 and 11; lon 0/40 on clips 3 and 8 and 0/39 on clip 9 — visible as a red cap across the strip.
* Nav-given turns: with `nav = left/right` the emitted plan satisfies the compliance predicate on 0/171 (clips 1, 4, 6, 8), 17/171 (clip 5), 45/171 (clip 2),
  50/177 (clip 3), 80/171 (clip 7). With `follow` it does not turn on 172/172, 171/171, 171/171, 171/171.

**The agreement chips' known-value control.** The SAME plan -> (lat3, lon3) rule applied to the GT path agrees with the GT v7 labels on lat **285/320
(89 %)** and lon **258/479 (54 %)** labelled windows. Confusion (GT label × GT-path lon3): ACCELERATE -> accelerate 48 / steady 112; BRAKE_TO -> brake_stop 53 /
steady 61 / accelerate 6; ADAPT_SPEED_FOR_CURVE -> accelerate 42 / steady 38; CRUISE -> steady 119. Lateral: NUDGE_L/R -> lane_keep 39/120 (no lat3 counterpart,
shown as n/a), TURN_L -> lane_keep 33 / turn_left 7. So **the lon chip is coarse** (the v7 lon labels are finer than a 2 s kinematic rule) and is flagged as such on
the page and in the video's opening card; it is a reading aid, not a metric.

**The tool.** `Media/2026-10-04-refcv7-replay/`: **260.1 MB** of tool + data (103.7 MB camera JPEG q82 in 2,059 files, 156.3 MB of `clip_NN.js` holding 49.0 MB of map PNG
before base64, 12 clips): 2,059 JPEGs + 13 data scripts + 3 page files. ⚠️ D: is exFAT with 1 MiB clusters: the 2,059 small JPEGs occupy ~2.1 GB ON DISK although the bytes are 104 MB.
Self-test (button; run on all 12 clips in the Browser pane): JS projection vs the model's `RigCamera` on 68 probe points per clip **max |Δ| 2.3e-13 px**, 0 valid-flag
mismatches; JS greedy matcher vs Python on every agent-labelled window of all 12 clips: **0 windows with a different TP/FP/DontCare set** (1 window, clip 5 #117, lists
two detections with identical scores in the other order); E9 / decoder pick re-derived from the exported scores on all windows; scored-cell count of both map PNGs
equals the trainer's `n_supervised` exactly. Browser-pane tests (toggle suites on clips 3 and 7 and on the smoke clip, self-test on all 12): every checkbox, slider, select, class chip and key changes at least one canvas where its
layer has content (reach/ceiling/nav filters verified on windows where they remove a top candidate); 10 fps playback verified by counting frames (+19 in 2.0 s at 1x).

**The video.** `refcv7_final_50400_tactical.mp4`: 1920x1080 H.264 High yuv420p 10 fps, **2,519 frames = 251.9 s** (100 opening + 12 x 30 title + 2,059 windows), 120,570,003 B, md5
`6fced5ae9ea4c68ecdff281ab67b23a0` (ffprobe `-count_frames` reads 2,519; the tool's own decode-back reads 2,519). 720p `..._preview720p.mp4` 13,372,586 B (two-pass, 430 kb/s) and 540p
`..._540p.mp4` 4,256,946 B (two-pass, 135 kb/s), both 2,519 frames. Frames checked by eye: opening card, clip title card, clip 3 w109 (the nav-LEFT turn the plan misses), clip 7 w130 (plan over a
ceiling NO candidate satisfies), clip 10 w55 (a labelled window with GT goal tokens and a wrong lon decision).

⚠️ **Two blobs of this file ran.** The capture ran with md5 `82f250f8914957af9d39483eac8026e5` (recorded in the capture record); the assets and the video ran with the repo blob
`55f1d4c7a6f4fe9c8e74454711c144ae`. Between them, the capture stage changed only by moving the 68-point probe generator into `test_vector_probe` (identical RandomState draws; the assets
stage checks the regenerated probe against the captured one to 1e-6 m) and by the stage code added after it. The reel renderer the module imports is the reel's blob `abca55c91c7408644483554b9f282787`.
