# render_refcv6_map_video.py — refcv6's semantic-map head, rendered for examination

The PI asked on 2026-09-26: *"Can we visualize the sematic map output of the last checkpoint and render a
video to examine the results"*.

This tool renders a video of refcv6's SAM3 semantic-map head (`out["perception"]["map_logits"]`,
9 × 120 × 64 at 0.5 m) at the latest checkpoint available on this box. The map is shown next to its SAM3
ground truth, the camera frame, and the planner's own selected trajectory.

⚠️ **This is a perception diagnostic in open loop. It is not driving performance.** It says nothing
about closed-loop driving, and its IoU is **not** a four-family eval result.

## 1. What runs (provenance)

| item | value | how it is checked |
|---|---|---|
| checkpoint | `D:/refcv6_eval_kit/ckpt/ckpt_35000.pt` (1,177,897,033 B; the rolling `ckpt.pt` pulled at step 35,000) | Full md5 must equal `68a4ef3bbb0a3ec707b04f0fb2b6b86f` (`ckpt/MD5SUMS`). `ck["step"]` must equal 35000. The load is **strict**. |
| run state | post-switch A16 hybrid: resumed at step 34,500 on commit 82c2331 with the F3 cascade loss and the true label clock | the config below |
| config | `C:/Users/Admin/ev6_battery/raw/thor_reads/config_resume34500_20260926.json` | md5 must equal `a3193a4685ce0d07a6ae6b89fdc994a8`, the value `chain_final_v2.sh` names as `RESUME_CFG_MD5` |
| code tree | `C:/Users/Admin/ev6_82c2331` (`REFCV6_REPO`), as used by `gate_dryrun_35k.sh` and `chain_final_v2.sh` | `tanitad.__file__` is asserted to lie under `<tree>/stack` before any model code runs |
| clip clock | `D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl` (md5 `78466f99…`, equal to Thor's copy) | the loader's own `--clip-clock-sidecar` remap |
| model build | `C:/Users/Admin/ev6_battery/code/refcv6_loader.build_model`, the battery's replay of `train()` | Same object the battery scores. Strict load, and `param_breakdown` must equal the config. |
| eval windows | `refcv6_loader.build_eval_dataset(..., with_perception_targets=True)` over the kit's 139-clip eval cache | the trainer's own `V3Dataset` window contract |

**Departures from `compute_losses_v3`, all recorded in `render_record.json`:**

1. **Providers are filtered to the three rendered clips** before the dataset is built. This follows
   the Master Mind's `Q3_ONLY_EPISODE_SHA12` pattern (`q3_hooks_forward.py`). A full 139-episode build
   with perception targets exhausted this box's RAM five times.
2. **The agent join and the 3-D join are not attached.** Both are LOSS-only targets. The forward does
   not read them, because this arm is `--agents head` and `agent_gt` is used only on the oracle seam.
   The renderer asserts `agents.oracle` is off. The SAM3 map target **is** attached, through the
   trainer's own `enable_map_gt`.
3. **Future frames are not decoded.** Only the LAW loss reads them, and that loss never runs here.
4. **The forward is the trainer's own `compute_losses_v3`, stopped after the forward.** A forward
   hook on the model captures `out` and raises, so the whole input plumbing runs as written:
   `set_ego_window`, the per-clip lift geometry, `v_max_ms`/`v_max_valid`, nav, `ego_state`. Nothing
   after `model(...)` runs.
5. **The DDIM draw is seeded per window** (`torch.manual_seed(seed)` before each forward, seed 0 by
   default). The orange path is therefore one inference draw. The map head is checked separately to
   be independent of the draw (see §5).
6. `--trunk-compile` is dropped by the loader, because there is no Triton on Windows. This is the
   loader's own recorded departure.

## 2. The clip rule — fixed BEFORE rendering

> **Candidates:** the 139-clip eval set (the kit's `refcv6-b1-416x1024-eval139` cache and its v8 eval
> labels), restricted to clips with **full SAM3 map coverage**. A clip has full coverage when its
> SAM3 GT file exists, validates, and every one of its eval windows maps to an in-range GT frame. The
> test is `perception_targets.require_map_coverage` over the clip's own windows, requiring
> `n_ok == n_windows`.
>
> **One clip per v7 nav token** (`left`, `right`, `follow`). This is the clip's `nav_command` as the
> trainer resolves it (`V3Dataset.enable_nav_from_v7`, a per-clip constant).
>
> **Within each token, the clip with the smallest `sha12`** (`sha256(clip_id)[:12]`, compared as a
> lowercase hex string).
>
> **Every eval window of each chosen clip is rendered, in time order.** The video order is
> `left`, `right`, `follow`.

The selection is written to `clip_selection.json` with sha12s and counts only. A second run
recomputes it and refuses if it differs.

## 3. What is on every frame

The standing viz standard applies: camera projection, metric BEV and text overlay together, on every
frame. Every frame is validated by `tanitad.viz_standard.check_frame`. An element this reel does not
draw is declared UNAVAILABLE, and its reason is printed on the frame.

1. **FRONT CAMERA.** This is the window's last input frame: `item["frames"][-1][-3:]`, the exact
   uint8 bytes the trunk is handed. Two paths are projected into it:
   * the GT future path, in green (110, 231, 138);
   * refcv6's **selected** trajectory `out["traj"]`, in orange (255, 158, 61). This is the model's
     own output, never an oracle.

   The projection is `render_refcv3_video.CylProjector` (imported, not copied) with the clip's
   **measured** extrinsics from `refcv6_train_eval139_extrinsics.json`, loaded by
   `render_refcv3_video.load_extrinsics`. The rotation is cross-checked against
   `pai_extrinsics_table.quat_to_R`. A horizon line predicted from the extrinsics is drawn as a
   self-check. A dashed line marks the bottom 43 rows, which the trunk zeroes
   (`--equalize-bottom-rows 43`).
2. **BEV — PREDICTED MAP.** Argmax over `softmax(map_logits)`. The ego sits at the bottom centre, with
   metric range lines every 10 m. The GT path and the selected path are overlaid. Cells outside
   `map_seen & map_valid` are **not scored**, and are hatched in one of two ways:
   * diagonal lines: SAM3 never saw the cell (`~map_seen`);
   * dots: SAM3 saw the cell, but the lift's camera does not reach it at this instant
     (`map_seen & ~map_valid`).
3. **BEV — SAM3 GT MAP.** Argmax of `batch["map_frac"]`, with the same two hatches from the same
   masks and the same overlays.
4. **DRIVABLE AGREEMENT.** Scored cells are coloured TP, FP, FN or TN, with a legend. The panel also
   shows this frame's drivable IoU, the running clip mean, and the pooled clip IoU.
5. **HUD.** It carries:
   * `refcv6-r101-s0 · step 35,000 · hybrid: F3 + true label clock from step 34,500`;
   * `perception diagnostic, open loop — not driving performance`;
   * the clip sha12, t on the label clock, the measured v0 at t0, and the nav token (drawn as a GIVEN
     INPUT);
   * the 9-class palette legend and a per-frame IoU sparkline for the clip.

   No class colour reuses the GT green or the model orange; the renderer asserts this.

⚠️ **The class panels and the agreement panel use two different rules.** The class panels show the
**argmax**. The agreement panel and the IoU use the trainer's **≥ 0.5 threshold**. They differ on
"mixed" cells, where drivable is the argmax but its fraction is below 0.5. The count of such cells
is printed per frame and in `per_frame.jsonl`.

## 4. The IoU rule — the trainer's, cited

`C:/Users/Admin/ev6_82c2331/stack/scripts/refc_v3_train.py:4506-4521`, inside `compute_losses_v3`:

```
_sn = map_seen & map_valid            # only when model._map_lift_valid_mask (the run's default)
_gt = (map_frac[:, drivable] >= 0.5) & _sn
_pr = (softmax(map_logits, dim=1)[:, drivable] >= 0.5) & _sn
map_iou_drivable = |_gt & _pr| / |_gt | _pr|     (0.0 when the union is empty)
```

`drivable_iou_rule()` computes the same arithmetic on the same tensors, on the model's device, with
softmax in the logits' native dtype. The scored-cell count is cross-checked on every window against
the programme's own `refcv6_perception_branch.map_loss_row(..., lift_valid=map_valid)`
`n_map_cells`, and must be equal.

The in-run `eval_map_iou_drivable` (`refc_v3_train.py:8377-8401`) is the **mean over 8 batches of a
per-batch POOLED IoU** (16 windows per batch). This tool reports three per-clip numbers:

* **clip mean:** the mean over windows of the per-window IoU;
* **clip pooled:** Σ∩ / Σ∪ over the clip's windows, which is the trainer's batch rule with the whole
  clip as one pool;
* **n:** the number of windows.

⚠️ The windows of one clip come from ONE episode and are strongly correlated. **No interval is
quoted.** They are not independent samples.

## 5. Controls (results are filled in §7 after the render)

**(a) Known-value control.** The GT map is fed through the same prediction path as "logits".

* **Rendering check:** the logits are `K · one_hot(argmax map_frac)`, with K = 30. The predicted-map
  panel must equal the GT panel **pixel for pixel**, overlays included.
* **Scoring check:** here the literal one-hot is not enough. On mixed cells (drivable is the argmax
  but its fraction is < 0.5), a one-hot makes p(drivable) ≈ 1 while the GT threshold says "not
  drivable", so it is two rules. The **threshold-consistent** GT-as-logits keeps the one-hot
  everywhere except the mixed cells. On those, the drivable logit leads the other eight by exactly
  1, so the argmax is unchanged and p(drivable) = e/(e+8) = 0.2536 < 0.5. The drivable IoU must then
  read **exactly 1.0**, and its panel must equal the GT panel pixel for pixel.

  The literal one-hot's IoU is reported beside it, with the mixed-cell count that explains the gap.
* **The control must be able to fail.** Two mutation arms must go RED:
  * a lateral mirror of the control logits, which is the classic orientation defect;
  * the drivable channel read off by one.

**(b) Independent check.** Each clip's mean drivable IoU under the trainer's rule is reported beside
the run's own in-run `eval_map_iou_drivable` at step 35,000, read from the metrics file at run time.
That number covers a **different** set of 128 windows, so it is a sanity range, not an identity.

**(c) Draw independence, added.** On the first window of each clip, the forward is repeated with
seed 1. `map_logits` must not move beyond device float noise, and the thresholded drivable mask must
be identical. The orange path may move; that change is reported.

## 6. Usage

```
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
# 1. CPU smoke: 2 windows of the first clip, every panel, both controls, a tiny MP4
OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 $PY taniteval/tools/render_refcv6_map_video.py --mode smoke
# 2. full render: waits for the dev-box GPU gate (used < 4300 MiB, no other python on the GPU,
#    >= 8 GB free RAM) for up to 90 min; after that it renders on CPU only if >= 12 GB RAM is free,
#    and otherwise stops and names the blocker. It always writes the RENDER_DONE marker.
OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 $PY taniteval/tools/render_refcv6_map_video.py --mode render
```

Outputs are written to `--out-dir`:

* `refcv6_map_step35000.mp4`, at 10 fps, H.264 yuv420p, 1920 × 1200. It holds a title card per clip,
  then every window of that clip. `*.mp4` is gitignored.
* `refcv6_map_step35000_small.mp4`, when the full file is 30 MiB or larger (`verify_mp4.py`'s
  delivery limit). It is re-encoded from the frames, not from the mp4.
* `contact_sheet.png`, with 12 keyframes: 4 per clip, at quantiles 0.1, 0.37, 0.63 and 0.9 of the
  clip's windows.
* `per_frame.jsonl`, with one row per rendered window and no raw clip ids.
* `per_clip.json` and `render_record.json`.
* `clip_selection.json`.

Frames are written to `--work-dir` on local disk (C:), not on the exFAT D: drive.

## 7. Results of the 2026-09-26 render (step 35,000)

The full write-up is in `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-map-video/RESULT.md`.
Every number here is MEASURED, from that package's `raw/per_clip.json` and
`raw/orientation_control.json`.

**The clips the rule chose.** 137 of 139 eval clips have full coverage. The other 2 have no GT file.
The chosen clips are:
* left `693335b810c3`;
* right `0191487845ef`;
* follow `00213e99adca`.

Each has 171 windows, and all are rendered.

**Drivable IoU, trainer rule:**

| clip | nav | n | mean | pooled |
|---|---|---|---|---|
| `693335b810c3` | left | 171 | 0.703 | 0.703 |
| `0191487845ef` | right | 171 | 0.622 | 0.609 |
| `00213e99adca` | follow | 171 | 0.727 | 0.724 |

No interval is quoted, because one clip is one episode.

**Control (a), known value, over 171 + 171 + 171 windows:**

* **IoU.** The threshold-consistent GT-as-logits reads **exactly 1.0 on 513/513** windows.
* **Pixels.** Its predicted panel equals the GT panel pixel for pixel on **171/171** frames of clip 1.
* **The literal one-hot.** A one-hot of the argmax reads a mean IoU of 0.9955, 0.9906 and 0.9923.
  The whole gap is the "mixed" cells (1,033 / 4,613 / 4,131 over the clips), where argmax and
  threshold disagree. The literal one-hot's panel also equals the GT panel on 171/171 frames.
* **Mutations go RED on 513/513 windows**, for both the lateral mirror and the channel shift.
* **The cell set is the trainer's.** The scored-cell count equals `map_loss_row`'s `n_map_cells` on
  513/513 windows.

**Control (b), independent.** The clip means are 0.703, 0.622 and 0.727, and the pooled value over
all 513 windows is 0.677. The run's in-run `eval_map_iou_drivable` at step 35,000 is **0.67745**,
over 128 **other** windows. This is a sanity range, not an identity.

**Control (c), draw independence.** `map_logits` max |Δ| between seeds 0 and 1 is 0.0 on all three
clips. The selected path moves 0.35, 1.43 and 7.60 m.

⛔ **Correction (2026-09-26, after delivery): the camera caption of the step-35,000 video is wrong.**
It reads *"bottom 43 rows are ZEROED inside the trunk"*. The trainer's pin sets
`cfg.core.encoder.trunk_equalize_bottom_rows` at `refc_v3_train.py:381` as an **undeclared**
attribute. The `--image-hw` rebuild, `dataclasses.replace(enc, …)` at `:459`, carries only declared
fields, so the trunk zeroes **nothing** (D-REFCV6-EQUALIZE-DROPPED). Two things still hold:
* the loader replays the same pin, so the rendered model is the trained one;
* the BEV lift (`:6922-6925`, built from the argv) does treat those rows as unobserved.

Only the caption is wrong. From `--boxes` on, the caption is driven by what the built modules do:
`render_record.json` → `equalize_bottom_rows`.

**Orientation.** The ego's own GT path lands on SAM3-drivable cells:
* 0.969 / 1.000 / 0.995 under the file contract, against 0.889 / 0.964 / 0.964 under a lateral
  mirror;
* on the right clip's lateral windows only, 0.994 against 0.834.

## 8. `--boxes`: the detected-agents video (PI 2026-09-26, "the detected agent boxes by the model")

It uses the **step-38,000** checkpoint, the last valid one (the PI stopped the run at 38,211):
* checkpoint `D:/refcv6_eval_kit/ckpt_final/ckpt_step38000.pt`, md5 `5a2e7222a9f5f8c7aa7bf38ef4698d8a`;
* config `D:/refcv6_eval_kit/ckpt_final/config.json`, md5 `a3193a46…`, byte-identical to the
  post-switch resume config;
* the same 82c2331 tree and the same loader.

Every panel of §3 is kept. The additions are below.

### 8.1 What is detected, and how it is decoded (all file:line in 82c2331)

**The two detector heads.**

| head | output key | decoder |
|---|---|---|
| **3-D box head** (perception branch) | `out["perception"]["box_slots"]` | `Box3DSlotDecoder.decode`, `box3d_head.py:265-284` |
| **agent head** (the planner's agent tokens) | `out["agent_slots"]` | `AgentSlotDecoder.decode`, `agent_slots.py:656-684` |

Both emit:
* `box = (cx, cy, l, w)` in metres, rig frame;
* `yaw = atan2(sin, cos)`;
* `presence_logit`;
* `cls_logits` over the 10 `AGENT_CLASSES`.

The 3-D head adds `cz`, `h` and `box3d`.

**"Detected"** means `sigmoid(presence_logit) >= 0.5`. That is the model's own
`AgentSeamConfig.presence_gate` (`refc_agents.py:197-199`), read from `config.json`
`seams.agents.presence_gate` and asserted equal to the built config. It is **not tuned on these
frames**. Detected slots are counted inside the BEV range (x 0–60 m, |y| ≤ 16 m, the slot decode
box).

**GT.** The agent join, plus the 3-D join for z and h, as the trainer builds them per window:
`V3Dataset._agent_item(ep, t+w-1)`, `refc_v3_train.py:2764-2864`.

**Matching, by the trainer's own rule:**
1. `refc_agents.visible_target_filter` (`refc_agents.py:410-433`: the 120° field ∩ the decode box)
   picks the target set;
2. `agent_slots.match_slots` (`agent_slots.py:788-823`) runs the Hungarian match, with cost
   `presence + cls + centre + 0.5 size`.

This is exactly `box3d_set_loss`'s order (`box3d_head.py:376-411`). The Hungarian assigns a slot to
EVERY target, so `matched == targets` is an identity (`box_quality.py`), and the mean centre error
over those pairs includes forced assignments to slots the model does not believe in. For that
reason **both** of the following are reported:
* the trainer's pairs, with the count whose slot is confident;
* the programme's AP matcher, `box3d_head.box3d_match_rows` at 2 m (`box3d_head.py:482-536`),
  scored by presence. It gives TP, recall and precision at the gate, and AP@2 m over the clip.

**Identity check on every window.** The trainer's own `box3d_loss_row(...)["box3d_centre"]` must
equal our L1 centre mean over our pairs. If it differs by more than 1e-4 the render refuses.

**Drawing.**
* A **4th BEV panel, DETECTED AGENTS**, drawn over the predicted map dimmed:
  * GT boxes in green: solid for a trainer target, dashed for in range but outside the 120° field;
  * 3-D head boxes solid, in class colour;
  * agent-head boxes dashed, in class colour;
  * a white link for each Hungarian pair whose slot is confident.
* **Camera.** GT target cuboids in green, and the 3-D head's confident cuboids in class colour. They
  are projected by the **model's own per-clip `RigCamera`** (`model._rig_camera`,
  `rig_projection.py:166-320`). That camera is cross-checked against `CylProjector` on ground
  points, and the render refuses above 0.01 px.
  * ⛔ **Edges since 2026-09-27: `cuboid_edge_runs`, the box-head audit's patch, installed verbatim.**
    Each edge is projected as 32 sampled 3-D points. An edge that leaves the frame is bisected onto the
    border, and each in-frame run is drawn as a curve.
  * The previous `draw_cuboid_cam` drew an edge only when BOTH its corners were in the frame, as a
    straight chord. Cuboids crossing ±60° therefore read as single flat faces: the PI's "very
    unplausible" boxes.
  * The audit verified the PROJECTION independently (raw mp4, LiDAR, SAM3 camera); only the drawing
    was wrong.
  * ⚠️ **The step-38,000 boxes video was rendered BEFORE the patch.** Its frames carry the old drawing;
    its numbers do not depend on it. Package: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-boxes-video/RESULT.md`.

### 8.2 The clip rule (fixed BEFORE rendering)

> **Candidates:** eval clips with full SAM3 coverage (as §2) and at least one agent-labelled window.
>
> **Rank** by the number of GT agent boxes within **30 m** of the ego (`hypot(cx, cy) <= 30`) **and**
> inside the camera's 120° field (`|atan2(cy, cx)| <= 60°`, `cx >= 0`, the trainer's visible
> predicate). The boxes are counted at each eval window's NOW from the trainer's `_agent_item`, and
> summed over the clip's eval windows. Ties are broken by sha12.
>
> **Take the top 3.** If none of them has a window whose **GT path heading changes by more than 30°**
> within its valid 6 s future, the 3rd is replaced by the highest-ranked clip that does. The heading
> change is `max |wrap(yaw[now+k] - yaw[now])|`, for k = 1..60, over rows inside the clip.
>
> **Every eval window of each chosen clip is rendered in time order**, the clips in rank order.
> sha12 only.

The ranking (top 10), the turn swap if any, and the chosen clips are written to
`clip_selection.json` in the package.

### 8.3 Controls

* **Boxes known value.** On every window, the GT targets are written into the slot format
  (`gt_as_slots`) and sent through the SAME readout and drawing path. Every target must match
  **itself**, with centre error **exactly 0.0 m**, footprint corners identical, and greedy TP ==
  targets.
* **The map controls of §5 are kept unchanged.**

### 8.4 `--dump-map`: the map-class analysis dump (same process, same model load)

After the render, the map head's per-cell softmax, the GT fractions, `map_seen` and `map_valid` are
dumped for:
* the in-run eval's 128 fixed windows (`inrun_eval_perm`, seed 12345);
* the map video's 3 clips;
* the rendered clips;
* the in-run permutation continued until ≥ 1,000 unique windows.

The dump lives on local disk (`MapDumper`), and the analysis runs afterwards on CPU. The done-marker
is `C:/Users/Admin/qland/work/mapvid/BOXES_DONE`.
