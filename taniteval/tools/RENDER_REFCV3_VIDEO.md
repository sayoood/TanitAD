# RENDER_REFCV3_VIDEO — the five-panel refcv3 planner reel

**Instrument:** `taniteval/tools/render_refcv3_video.py` ·
**helper:** `taniteval/tools/pai_extrinsics_table.py` ·
**written:** 2026-09-03, on the PI's direct request ·
**⭐ DELIVERED REEL (2026-09-04):** `refcv3-b1-v72-30k/ckpt.pt` frozen as `ckpt_step40284_frozen.pt`
— **step 40 284, the FINAL checkpoint of the finished run**, `hier`, 107 082 365 tensor elements,
STRICT load **0 missing / 0 unexpected keys**, md5 `b1ed7075ff730d0993d2eaa3c86f6b56` ·
**superseded:** the 2026-09-03 `ckpt_30000.pt` reel (step 30 000) — kept only as the shape the PI
approved; ⛔ never quote it as "the model".

> ⛔ **READ `taniteval/tools/REFCV3_ARM.md` §2 FIRST.** It is the definition of what refcv3's arm
> **is**, derived with `file:line`, and every claim this renderer draws on a frame is bounded by it.
> This file draws pixels; it invents no number and it re-implements no part of the model path —
> `render_refcv3_video.py` **imports `refcv3_arm.py`** for the STRICT model rebuild, the eval-time
> seam neutralisation, the trainer's own window dataset with the v7.2 label join and the nav source,
> and the grid index-select. A renderer that rebuilt the model its own way could show a **different
> model** from the one the eval scores.

---

## 1. The five panels — what the PI asked for, and where each one is

| # | panel | what is drawn | source expression |
|---|---|---|---|
| **1** | **FRONT CAMERA** (top-left, 1280×512) | the GT future path (**green**) and refcv3's **selected** trajectory (**orange**) projected into the image, plus the `a_star` oracle (**violet**) under `--with-oracle` | `CylProjector(v2ep['frame'], per-clip sensor_extrinsics)` applied to `out["traj"]` and to `ego_future(poses, t0, 60)` |
| **2** | **METRIC BEV** (right, 604 px wide) | the **whole fan** — `anchor_traj` **[128, 8, 2]** — every anchor coloured by its **selection rank**, anchors killed by `reach_keep` in a distinct dull red, the selected path on top, GT in the same metric frame, `goal_point_tac` marked | `out["anchor_traj"]`, `out["sel_score_v3"]`, `out["reach_keep"]`, `out["traj"]`, `out["goal_point_tac"]` |
| **2b** | **SELECTION SURFACE** (right, below the BEV) | the same scores as a 1-D strip, one column per anchor **in anchor order**, with the selected column and the oracle's column marked | `out["sel_score_v3"]` |
| **3** | **TACTICAL** (left, below the camera) | the **factored** heads as **two separate 8-class distributions** — lateral and longitudinal — against the v7.2 labels; plus a **path-derived** row that is defined on every frame | `softmax(out["lat_logits_tac"])`, `softmax(out["lon_logits_tac"])` vs `v7_labels.tactical_class_ids`; `four_families.maneuver_kinematics` → `refc_tactical.factor_from_kinematics` |
| **4** | **STRATEGIC** (left, bottom) | `route_logits` against the **v2.1 per-window** route label; beside it, in its **own slot**, the v7.2 nav token that was FED; and `g_str`, the strategic goal bearing | `out["route_logits"]` vs `item["route_target"]/["route_valid"]`; `NAV_COMMANDS[item["nav_cmd"]]`; `out["g_str"]` |
| **5** | **HUD** (full width, bottom) | the measured `v0` at t0, the nav token, the **checkpoint step**, the frame's ADE and the clip mean, the legend, and the resolution caveat | — |

Every frame is validated through **`tanitad.viz_standard.check_frame`** before it is drawn, so a
silently missing panel is structurally impossible: an element that cannot be computed renders its
**reason**, never a blank. The seven declared elements are `camera`, `bev`, `tactical`, `strategic`,
`ade`, `strategic_input`, `tactical_gt`.

---

## 2. Colour semantics — one meaning per colour, in every panel

| colour | meaning |
|---|---|
| **green** `(110, 231, 138)` | **GROUND TRUTH** — everywhere, always. |
| **orange** `(255, 158, 61)` | **refcv3's own decision** — the selected trajectory, the predicted class, the `goal_point_tac` cross. |
| **violet** `(186, 128, 255)` | the **`a_star` ORACLE ceiling**. ⛔ Never the model's choice; the word ORACLE travels with it in the legend and on the selection strip. |
| **amber** `(240, 190, 90)` | a **GIVEN INPUT** — today only the nav token. |
| **dim slate → cyan** | the 128-anchor fan and the selection strip, ramped by **selection RANK**. |
| **dull red** `(58, 44, 52)` | an anchor **killed by `reach_keep`** before the argmax — drawn, never omitted. |
| **red** `(248, 113, 113)` / **green** | a head-vs-GT **MISMATCH** / **MATCH**. |

---

## 3. ⛔ The four things this renderer refuses to get wrong

**(a) The selection is `out["traj"]`, never `a_star`.** `a_star` is the anchor nearest the
**ground truth** (`refc_v3_train.py:460`) — an oracle. Drawing it as "the model's choice" would be a
fabricated result. The deployed path is the `sel_score_v3`-ranked choice
(`refc_v3.py:509-527`; `refc.py:1531-1534` on the flat arm), read straight off `out["traj"]`. The
oracle appears only under `--with-oracle`, in its own colour, labelled.

**(b) Frames are bridged locally (C79).** The overlay is drawn on `item["frames"][-1][-3:]` — the
exact uint8 tensor `frames_to_device` hands the encoder — upscaled with **NEAREST** so no pixel is
invented. Nothing is re-decoded or re-fetched between scoring and drawing.

**(c) The projection is the clip's own, and it is CYLINDRICAL.** The B1 cache is
**256×640, `f_ref` 305.5775, `projection="cylindrical"`**, read from each `*.v2ep.pt`'s own `frame`
field — never assumed. On an equidistant-azimuth raster the column is **linear in azimuth**;
`CylProjector` inverts `calib.cylindrical_rays` exactly (`u = cx + f·atan2(x, z)`,
`v = cy + f·y/hypot(x, z)` — note the **radial** denominator, not the forward one) and **refuses** a
projection it cannot express rather than approximating it. Applying the pinhole formula here is
wrong and looks entirely plausible: it implies a 92.6° field where the rig is
`camera_front_wide_120fov`.
The extrinsics are **per-clip and MEASURED** from PhysicalAI's own
`calibration/sensor_extrinsics`. ⚠️ Camera height on this corpus is **1.245–1.607 m** (median
1.306 m, 37 distinct values in 40 clips) and the three constants circulating in this repo — 1.22,
1.43, 1.5 — are **all wrong as a constant**; 1.22 is below the observed minimum. The camera also
sits **~2.0–2.1 m forward** of the vehicle origin with a per-clip pitch. Without `--extrinsics` the
camera overlay is **DISABLED** and says so on the frame; the BEV is calibration-independent and
carries the comparison either way.
⭐ **The frame carries its own falsifier:** the horizon row predicted from the clip's extrinsics is
drawn as a thin line. If it does not sit where the road meets the sky, the projection is wrong and
the viewer can see that without trusting a caption.

**(d) The tactical panel is never collapsed to one 5-way label.** `COLLAPSE_TABLE`
(`refc_tactical.py`) makes a turn absorb the longitudinal decision entirely, so a single 5-way bar
chart **cannot show a braking-into-a-turn error at all**. That defect is the reason the factored
head exists.

---

## 4. Two honesty notes that are drawn on the frame, not buried here

* **The v7.2 tactical GT exists only inside a ±2.0 s band** around each clip's own `t0_s`
  (`v7_labels.window_in_band`); outside it `tactical_class_ids` returns `IGNORE_ID` and is **never**
  clamped to a neutral class. The panel prints *"v7.2 GT: — outside the v7.2 record's ±2.0 s band"*
  on those frames. That is why the **path-derived** row is there beside it: it is defined on every
  frame, through the same instrument the four-families eval uses. ⚠️ It is a **many-to-one
  projection for METRICS, never a label source** — kin3 destroys exactly the distinctions (NUDGE vs
  LANE_CHANGE, CREEP vs BRAKE_TO) the v7 vocabulary exists to keep.
* **`future_poses_ext` is CLAMPED at the clip's last pose.** Admitting a window on all eight slots
  (6 s) would cost the last 6 s of every clip — on `ca11a2a2` that is the **entire left turn**. So a
  window is admitted on the **scored grid** (2 s), and GT is drawn only to the last genuinely valid
  slot with *"⚠ GT drawn to N s only"* printed on the BEV. An unmarked short GT would read as a stop
  the ego never made.

---

## 5. The clips, and why these three

Chosen from the **141-clip B1 EVAL set** by net |Δyaw| and speed range over the clip's own poses,
then filtered to daytime for legibility. All three are `train`-split clips in the v7.2 EVAL blob
(md5 `aa12c948f062181c3297265b51526ec5`, the same blob the run's `config.json` records for eval).

| order | clip | manoeuvre (MEASURED from `poses`) | nav token FED | v7.2 `a_tac` |
|---|---|---|---|---|
| 1 | `d85682b8-f612-4aa3-90a2-ee2bb86151e6` | **right turn while accelerating**: −88° over t 7→14 s, v 5.5 → 14.7 m/s. Urban, day, France. | `NAV_TURN_R` | lat `TURN_R`, lon `ADAPT_SPEED_FOR_CURVE` |
| 2 | `ca11a2a2-1021-475c-9907-e6dbe658ad12` | **left turn into a stop**: +96° over t 13→19 s, v 9.8 → 0.18 m/s. Intersection, day, Netherlands. | `NAV_TURN_L` | lat `LANE_KEEP`, lon `BRAKE_TO` |
| 3 | `6ed4ef7a-1a70-48c5-8dc1-3c6a48ec3ecc` | **sustained high-speed bend**: −200° over the whole clip at 13–18 m/s. Urban, day, Austria. | `NAV_FOLLOW_ROAD` | lat `LANE_KEEP`, lon `ADAPT_SPEED_FOR_CURVE` |

Between them they exercise **both lateral directions**, **both longitudinal signs** (accelerate and
brake-to-stop), and **all three nav tokens** — so no panel is showing a constant.

⛔ **This is a hand-picked reel and it must never be quoted as a representative one.** The selection
and its order are written into the sidecar JSON (`clip_selection`), and the per-clip ADE is reported
per clip. A representative read is `taniteval/tools/refcv3_arm.py` over the whole 141-clip set with
its paired episode-cluster bootstrap — not this file.

---

## 6. Running it

### 6.1 Build the per-clip camera extrinsics (once per clip set)

```bash
python taniteval/tools/pai_extrinsics_table.py \
  --root  <PhysicalAI root holding calibration/ and clip_index.parquet> \
  --clips d85682b8-...,ca11a2a2-...,6ed4ef7a-... \
  --out   <work>/extrinsics.json
```

PhysicalAI-AV is **gated**, so the parquet is not in this repo; this reads whatever local copy the
machine holds and writes a small JSON the renderer can consume without the dataset. A clip with no
entry renders with the camera overlay disabled and a message on the frame.

### 6.2 Render

```bash
OMP_NUM_THREADS=6 PYTHONPATH=<repo>/stack python taniteval/tools/render_refcv3_video.py \
  --ckpt       <run>/ckpt_30000.pt \
  --config     <run>/config.json \
  --episodes   <dir with the chosen *.v2ep.pt> \
  --labels     <...>/s2_labels_v7.2_eval.jsonl.gz \
  --extrinsics <work>/extrinsics.json \
  --with-oracle --stride 1 --fps 10 \
  --clips "d85682b8-f612-4aa3-90a2-ee2bb86151e6,ca11a2a2-1021-475c-9907-e6dbe658ad12,6ed4ef7a-1a70-48c5-8dc1-3c6a48ec3ecc" \
  --out        <out>/refcv3_five_panel_step30000.mp4
```

### 6.3 ⭐ Re-render against the FINAL checkpoint — the one-line change

⛔ **THERE IS NO `ckpt_40284_FINAL.pt`, AND NOTHING WILL EVER WRITE ONE.** MEASURED at source
(two probes): `refc_v3_train.py:103` sets `MILESTONES = (5000, 15000, 20000, 30000)`, and the only
milestone write is `ckpt_{step}.pt` gated on `step in MILESTONES` (`:1197-1199`). 40 284 is not a
milestone. **The final checkpoint is the rolling `ckpt.pt`** (`:1090` / `:1192`) — a file whose
**name never changes while its contents do**, which is exactly what makes a stale copy
indistinguishable from the real one. *(`REFCV3_ARM.md` §3.1/§5 still name the non-existent file;
reported here, not edited from this stream.)*

```bash
# on the pod, AFTER the run is done: freeze it under an immutable name and md5 it
cp /workspace/experiments/refcv3-b1-v72-30k/ckpt.pt \
   /workspace/experiments/refcv3-b1-v72-30k/ckpt_step40284_frozen.pt && md5sum …

# then the IDENTICAL render command, with two arguments changed:
  --ckpt <run>/ckpt_step40284_frozen.pt  --expect-step 40284 \
  --out  <out>/refcv3_five_panel_step40284.mp4
```

`--expect-step` reads `ck["step"]` and **refuses before any GPU is spent** if it disagrees.
Nothing else changes: the step is read from the checkpoint and burned into the banner and the
sidecar, so two reels can never be confused. ⛔ Confirm the supervisor has exited first
(`ps -eo args | grep -c 'sup[_]refcv3'` reads **0**) — a supervised run whose done-marker was never
written resurrects itself and overwrites `ckpt.pt`.

### 6.4 Notes

* ⛔ **Never run this on the training pod.** It needs a GPU; the dev box's RTX 4060 is the intended
  host. `scp` from the pod is fine.
* `PYTHONPATH=<clone>/stack` is **required** off the G: mount — the venv's editable `tanitad`
  install points at the Drive, and an import from an off-Drive clone dies with `Errno 22`.
* Runtime on the RTX 4060: **~0.35 s/frame** end-to-end (forward + PIL composition + PNG), so a
  3-clip stride-1 reel is a few minutes. `--stride N` subsamples; `--max-frames-per-ep` truncates.
* `--keep-frames` leaves the PNG sequence beside the mp4 (that is where the committed stills come
  from). `--stills a,b,c` writes named PNGs directly.
* Output is **1920×1122**, H.264 `-crf 18 -preset slow`, `+faststart`.

---

## 7. What is written beside the mp4

`<out>.json` carries the model provenance (ckpt, **step**, arm, anchor count, horizons, window,
decoder mode/steps, the **STRICT load report**), the grid, the nav source and its stats, the labels
md5 and join counts, the clip selection and order, the per-clip frame counts and mean/max ADE of the
**deployed selection**, the per-clip `frame` geometry and extrinsics actually used, the seven
`viz_standard` element records, and three plain-language guards: `what_is_drawn`,
`what_this_is_not`, `nav_is_an_oracle`.

⛔ `*.mp4` is **git-ignored**. The renderer, this README and the representative stills are committed;
the mp4 is delivered as a file.

---

## 8. ⭐ The delivered reel — step 40 284 (2026-09-04)

`taniteval/results/videos/refcv3_five_panel_step40284/`

| file | size | committed? |
|---|---|---|
| `refcv3_five_panel_step40284.mp4` | 39.2 MiB — full quality, `-crf 18 -preset slow` | ⛔ git-ignored, delivered as a file |
| `refcv3_five_panel_step40284_web.mp4` | **21.2 MiB** — `-crf 24`, same 1920×1122 / 10 fps / 51.90 s | ⛔ git-ignored, delivered as a file |
| `refcv3_five_panel_step40284.mp4.json` | the sidecar | ✅ |
| `still_0{1..5}_*.png` | five representative frames | ✅ |
| `extrinsics_used.json` | the per-clip extrinsics actually used | ✅ |

**⭐ TWO FILES, BECAUSE SIZE IS A HARD GATE.** The 30 000-step reel (38.74 MiB) was **REFUSED by
the delivery channel** and reached only the desktop app. Anything over **30 MiB** does not arrive.
The web copy is re-encoded **from the original PNG frames**, never transcoded from the mp4 — a
transcode stacks a second generation of loss on the first for no size benefit.

**⛔ VERIFY A RENDER BY DECODING IT BACK, NEVER BY `ffmpeg`'s EXIT CODE OR THE FILE SIZE.**
`taniteval/tools/verify_mp4.py` runs two probes that differ in path-binding: the container
metadata (`ffprobe -show_streams`) **and** a real decode of every packet
(`ffmpeg -i … -f null -` plus `-count_frames`). A file whose header claims 519 frames while its
decoder yields 300 is exactly the failure a size check cannot see. Both delivered files read:

```
1920x1122 · h264 High · yuv420p · avg_fps 10.000 · duration 51.90 s
DECODED 519 frames · decode_clean=True · n/fps 51.90 s == container 51.90 s
```

### 8.1 The per-clip ADE, and what it is NOT allowed to say

| clip | manoeuvre (MEASURED from `poses`) | nav FED | mean ADE(sel) @ 30 000 | @ **40 284** |
|---|---|---|---|---|
| `d85682b8` | right turn while accelerating, −88°, v 5.5 → 14.7 m/s | `NAV_TURN_R` | 0.544 m | **0.519 m** |
| `ca11a2a2` | left turn into a full stop, +96°, v 9.8 → 0.18 m/s | `NAV_TURN_L` | 0.647 m | **0.612 m** |
| `6ed4ef7a` | sustained high-speed bend, −200° at 13–18 m/s | `NAV_FOLLOW_ROAD` | 0.561 m | **0.615 m** |

⛔ **THAT LAST COLUMN IS NOT A STEP-30k-vs-40k RESULT AND MUST NEVER BE QUOTED AS ONE.** Three
clips picked for legibility are not a sample: the deltas run **both ways**, they carry no
estimator and no interval, and §5 already binds this reel as hand-picked. The representative read
is `taniteval/tools/refcv3_arm.py` over the whole 141-clip B1 EVAL set with its paired
episode-cluster bootstrap.

### 8.2 The nav caption was corrected on the frame — twice, and the second time matters more

The strategic panel used to say only that the nav token is an oracle input on which "both the route
head and the tactical heads are conditioned (E13)". True **of the wiring** — and misleading about
the **behaviour**: the route head is nav-**INSENSITIVE**, and stating only the conditioning credits
an oracle input for a decision it does not drive, which is the nav-echo defect read backwards.

**⭐ THE CLAIM IS ARCHITECTURAL; THE MEASUREMENT IS ONLY ITS WITNESS, AND THE WITNESS BELONGS TO A
DIFFERENT STEP.** `route_logits = route_head(pooled)`, and `pooled` is the encoder's output over
`frames` alone, while `nav_cmd` is one-hot'd into the **sibling** `measurement(...)` branch and
never reaches it (`stack/tanitad/refs/refc.py`; the v3 `nav_inject` path touches only `z_tac_raw`
and `ctx`). ⚠️ Line numbers for this differ between trees — Drive `HEAD` has 2 283 lines and the
eval clone 2 264 — so cite the symbol, not the line.

The witness is `paired_true_minus_shuffled_accuracy = 0.0 [0.0, 0.0]`, paired episode-cluster
bootstrap, **n = 3 622 windows / 128 episodes**, in
`taniteval/results/refcv3-30k-openloop-20260903-2004.json` — **⛔ MEASURED AT STEP 30 000, NOT at
40 284.** The first fix printed `+0.0000` bare on a step-40 284 frame, where it reads as *measured
here*. It was not. The step now travels with the number, and the architectural argument — which is
a `file:line`, and therefore stronger than any single run — is what carries to this checkpoint.

**⚠️ AND IT IS NOT A DEFECT — the primary artifact argues the reverse.** On the 1 736 changed-nav
windows the route head follows the **label** at **0.7414 [0.6777, 0.8047]** and the **wrong fed
nav** at only **0.2224 [0.1912, 0.2542]**: vision-derived route skill, the opposite of flagship
v1's 1.0000 echo of its own input. The **tactical** heads are a different story and *do* move with
nav (lat **+0.0233 [−0.0026, +0.0537]** not separated; lon **+0.0225 [+0.0009, +0.0417]**
separated), which is why the caption says INSENSITIVE and never "broken", and why the two families
are never quoted as one.

**⛔ AND THE CAPTION'S LINE CAP NOW REFUSES INSTEAD OF TRIMMING.** MEASURED in this same pass: the
rewritten caption wrapped to **seven** lines under a `[:6]` slice, and the line the slice ate was
*"MEASURED @ step 30 000"* — the exact qualifier the rewrite existed to add. A silent truncation
does not shorten a caption, it **deletes the bound on the claim**, and the frame still looks
finished. `render_refcv3_video.py` now exits with the over-long line quoted rather than cropping it.
