# A road map from SAM3 alone — combined prompts, pixel overlays, a BEV map, and what it took to make it right

**Master Mind, 2026-09-13.** PI request, verbatim: *"I want to try to create an own sam 3 map alone by smart precise
combined prompt: extract drivable road areas, extract non drivable road edges, extract drivable painted road markings
on the road (arrows, text, markings, lane markings, cross walks,...). Generate a rendered video both pixel overlays and
a bev map based on this idea."* Follow-ups the same day: *"It has some difficulties on painted markings/text and
crossings … can we play with prompt, confidence level for cross walks, can we extract longer videos with more
examples"* and *"In some cases the extracted map is not aligned with overlays in the front image … the cross walk
paintings have different extracted semantics, let review again the process to optimize it."*

⛔ **Labels, not an inference path.** SAM3 is the only SEMANTIC source; calibration, ego poses and the LiDAR ground
height only PLACE its pixels (labels may use LiDAR and ego). No driving tier applies. Clips appear by sha12 only:
**day 4fbd97b6a4b7**, **night 73495082f98b**.

---

## 1. The pipeline (as of v3)

| stage | what it does | code |
|---|---|---|
| views | PhysicalAI cameras reprojected exactly (rotation about each optical centre) into the virtual nuPlan rig: F0, L0, R0, L1, R1, L2, R2 (B0 skipped) | `2026-09-13-qwen-drive-usage-review/code/build_frames_v2.py` |
| segment | SAM3 image model per view, 18 prompts in 7 families, instance-level rules (§4) | `code/sam3map_extract_v3.py` (v1: `sam3map_extract.py`) |
| refine | ego bonnet (clip-level SAM3 "car hood"), ego-attached artefacts, thin edges — **exact**: a control re-lifts the stored rasters and must reproduce every BEV point bit for bit before a rule applies | `code/sam3map_refine_v3.py` (v2.1: `sam3map_refine.py`) |
| lift | every class pixel (stride 2) cast through its camera onto the LiDAR ground (2 m cells), stored in the clip's world frame **with its observation range** | extractor |
| fuse + render | camera overlays; BEV now = **past 2 s only**, range-weighted class vote, solid = seen within 10 m, faded = only seen from farther; clip map so far | `code/sam3map_render_v3.py` |
| score | LiDAR checks calibrated on Qwen-Drive's nuPlan demo against its true map, vs Qwen-Drive on the same frames | `code/sam3map_score.py` |

Runtime (MEASURED, Jetson Thor): v1 **15.7 s/frame** (7 views × 13 prompts; 1,507 s / 96 frames), v3 **~20.6 s/frame**
(18 prompts). Rendering, encoding (GStreamer `x264enc`) and scoring also run on Thor: the dev box sat at **97 %
committed memory (0.6–0.8 GB commit headroom)** under the BEV agent's training job. The LiDAR sweeps for scoring were
decoded on the dev box (DracoPy 2.0.0, one sweep at a time) and shipped (sha-verified listings); no download was made.

---

## 2. Is the map any good? LiDAR checks, full clips (96 frames each), head-to-head with Qwen-Drive

MAP_A2 tall static obstacles on drivable (true map 0.03 %, ↓) · MAP_B the car's own path on road (↑) · MAP_C GT
vehicle centres on road (true map 100 %, ↑; a visible-surface segmenter cannot see road under cars) · MAP_E predicted
edges on a LiDAR height step or obstacle (true map 74.7 %, ↑). Pooled over frames; MIRRORED = geometry control.
Raw: `raw/sam3map_score_{day,night}_{v1,v2}.json`.

| arm (v1 extraction) | day A2 | B | C | E | night A2 | B | C | E |
|---|---|---|---|---|---|---|---|---|
| SAM3 single frame | 0.0718 | 0.4065 | 0.2240 | 0.5111 | 0.0701 | 0.5089 | 0.1760 | 0.4457 |
| SAM3 fused ±1 s | 0.0998 | 0.4865 | 0.4903 | 0.5311 | 0.0566 | 0.5896 | 0.1797 | 0.5003 |
| **SAM3 clip-vote** | **0.0657** | **0.9762** | 0.5584 | **0.5734** | **0.0635** | **0.9689** | 0.4226 | **0.4652** |
| Qwen single | 0.1648 | 0.9029 | 0.8636 | 0.3742 | 0.3005 | 0.9909 | 0.8996 | 0.3894 |
| Qwen fused ±2 s | 0.1715 | 0.9565 | 0.8961 | 0.4125 | 0.2901 | 1.0000 | 0.9071 | 0.4358 |
| SAM3 mirrored | 0.4518 | 0.3151 | 0.3994 | 0.2090 | 0.3320 | 0.2096 | 0.2317 | 0.2383 |
| clip-vote mirrored | 0.4655 | 0.7212 | 0.5130 | 0.2260 | 0.3449 | 0.5602 | 0.2045 | 0.2479 |

⭐ **The pre-registered next lever PASSED on both clips.** SAM3_clipvote (whole-clip, visibility-normalised vote within
15 m; bar written in `sam3map_score.py` before its first run: MAP_B ≥ 0.95 AND MAP_A2 ≤ fused ±1 s + 0.05 AND its
mirrored control below it on MAP_B and MAP_E): day B 0.9762, A2 0.0657 ≤ 0.1498, mirrored 0.7212 / 0.2260 below;
night B 0.9689, A2 0.0635 ≤ 0.1066, mirrored 0.5602 / 0.2479 below. **Against Qwen-Drive, the SAM3 clip map puts
2.6× (day) and 4.6× (night) fewer tall obstacles on the road and more of its edges on real curbs; Qwen keeps the
road under vehicles (MAP_C), which SAM3 cannot see.**

⚠️ **v2's thin edges are not a LiDAR win.** Keeping only edge pixels on the road boundary cleaned the video but moved
MAP_E up on 2 of 6 arm × clip cells (day single +0.047, day fused +0.111) and down on 4 (day clip-vote −0.035, night
single −0.082, fused −0.055, clip-vote −0.060). The removed curb pixels were right more often than the look suggested.
Next lever: an adjacency radius sweep (1 / 3 / 5 px) scored the same way.

---

## 3. v2: the ego vehicle in its own map (bonnet, patch, reflection)

* **Bonnet (R1).** SAM3 labelled the bonnet as road. "car hood" found it in 8/8 sampled frames on both clips (score
  0.89–0.93); majority mask 7.5 % (day) / 6.3 % (night) of F0; 261,742 / 237,834 labelled bonnet pixels removed.
* **Anonymisation patch (R2).** Accepted as a lane marking. The first rule (marking in ≥ 20 % of frames) was **inert**:
  the patch is labelled in only 10 of 96 frames (40–42, 60, 65, 75, 86, 90, 91, 95). Revised to *recurs ≥ 3 times
  spread over ≥ 30 % of the clip, central 20 % of columns* → one 28 × 24 px zone, 10 components removed (day).
  ⛔ **On the night clip that rule removed REAL paint** — crosswalk stripes and a stop line in frames 83–90 whose large
  components merely touched the zone (38 components, 57,911 px). v2.1: remove a component only if ≥ 50 % of it lies
  inside the zone → night 1 component. v3 adds "the ego moved ≥ 8 m while the pixel stayed paint" for a reflection line
  that held frames 0–28.
* **Control:** every refine re-lifted the stored rasters and reproduced the extractor's points bit for bit —
  96/96 frames on both clips, every version.

---

## 4. Paint semantics: crosswalk vs hatched area vs line vs arrow/text

### 4.1 Confidence and prompts for crosswalks — the only true paint map we have (nuPlan demo, 4 frames × 8 cameras)
Prompt bank: 22 prompts, every instance at score ≥ 0.20, swept offline (`code/sam3_promptbank.py`,
`sam3_promptbank_sweep.py`; raw `raw/sweep_demo.json`). True cells: crosswalk 9,967, line 26,086; 0.45 m tolerance.

| | precision | recall | F1 |
|---|---|---|---|
| crosswalk v1 (crosswalk + zebra crossing, 0.50) | 0.814 | 0.331 | 0.471 |
| crosswalk best in-sample (4 prompts, 0.25) | 0.596 | 0.620 | 0.608 |
| **crosswalk 2-fold HELD-OUT** (select on 2 frames, score on 2) | 0.587 | 0.426 | **0.494** |
| crosswalk best arm, SHUFFLED frames (control) | 0.023 | 0.032 | 0.027 |
| line v1 (lane + road marking, 0.40) | 0.366 | 0.584 | 0.450 |
| line best in-sample ("lane marking", 0.60) | 0.425 | 0.557 | 0.482 |
| **line 2-fold HELD-OUT** | 0.233 | 0.227 | **0.230** — the in-sample winners do not transfer |

Where the other prompts land on the true map: **"text painted on road" puts 1,401 of 2,865 cells on true lane lines**
(the dash-read-as-text defect); "word on road surface" 1,192 on true crosswalks; "hatched road marking" everywhere
(1,800 crosswalk, 4,566 line, 4,037 neither); "chevron road marking" / "diagonal stripes on road" never on a true line
or crosswalk.

### 4.2 The demo-tuned crosswalk threshold FAILED on our streets — replay on 84 views of our clips
v3 as first written (4 crosswalk prompts at 0.25, symbols dropped when a line explains them) was replayed on the
prompt-bank instances of 12 frames × 7 views (`code/sam3_v3_marks.py`): crosswalk pixels ×3.8, yellow hatched zones,
dashes and an arrow became crosswalk, and all 58 symbols were dropped (the generic "road marking" covers arrows too).
**Measured on the failing views: SAM3 scores "crosswalk" 0.79–0.89 on hatched no-parking zones — as high as on a real
zebra (0.85) — so no crosswalk threshold separates them; "diagonal stripes on road" does: 0.53 / 0.46 on the hatched
zones, 0.49 on a painted gore triangle, 0.22 on the zebra, 0.30 on give-way dashes.**

### 4.3 v3 rules (each from a measured failure; `sam3map_extract_v3.py`)
* crosswalk / zebra crossing / pedestrian crossing at **0.50**; accepted if ≤ 15 % of the view's road pixels **or it
  holds ≥ 3 line stripes** (v1's area cap rejected the zebra right in front of the car); its stripes become crosswalk;
* a crosswalk region whose stripes are ≥ 30 % covered by "diagonal stripes on road" (≥ 0.40) is a **hatched area**
  (new class 6); a standalone diagonal-stripes instance ≥ 0.45 is hatched too;
* arrow / text at 0.45, **dropped when ≥ 60 % covered by "dashed line" (≥ 0.45) or "solid line" (≥ 0.85)** — measured
  on the bank: dash-like "text" instances carry dashed 0.43–0.83 or solid 0.91–0.94, the six real arrows dashed ≤ 0.31
  and solid ≤ 0.51 (in-sample, visual audit only) — or when the ground footprint is < 0.35 m wide or > 8 m long.
Replay of the crosswalk/hatch rules: hatched zones and the gore triangle → hatched, the night zebra → crosswalk, the
give-way scene unchanged. Night frame 89 after full v3 extraction: the zebra is 40,430 px of crosswalk (3 instances
accepted by stripes), no stripe left as line or text.

### 4.4 v3 on the whole night clip — and the new error it made (v3.1)
v3 night (96 frames, 1,790 s; refine control 96/96; LiDAR checks equal to v2.1 — clip-vote A2 0.0648, B 0.9689,
C 0.4226, E 0.4056): the phantom crosswalk the PI saw at t = 15.8 s is gone (2.8 m² faded, tentative), the zebra at
t = 17.6 s is one crosswalk. ⛔ **New error: zebras seen obliquely by the rear cameras became hatched areas.** Per
camera over the clip, crosswalk → hatched decisions: F0 **0**, L0 3, L1 3, L2 15, **R2 63** (R0, R1 0) — the
diagonal-stripes prompt is an image cue, and a zebra seen from behind-right looks diagonal.
**v3.1 decides it on the ground** (`code/stripe_geometry.py::decide2`): paint lifted to the ground, local directions in
1 m cells (hatch stripes that touch their boundary line merge into one component, so counting separate stripes failed
first: 106 of 118 decisions fell back), dominant direction vs the region's long axis — zebra ≥ 60°, hatched 20–55° or a
two-direction chevron, else ambiguous → front camera: the v3 prompt rule; other views: crosswalk. Replay on the failing
views: day rear-left hatched zone **hatched** (dominant share 0.71 at 25° to the band), painted gore triangle
**hatched** (two modes 45° apart), night front zebra **crosswalk** (74°), night rear-right oblique zebra **crosswalk**
(ambiguous → crosswalk; v3: hatched). Cost: a hatched zone with no crosswalk-family region seen from a side view is now
plain line paint (it was hatched through the prompt rule). v3.1 extraction of night frame 10: rear-right zebra 21,757 px
crosswalk, 0 hatched (`sam3map_extract_v31.py`).

---

## 5. "The map is not aligned with the front image" — what was wrong, including my own two wrong answers

1. **Poses are not the cause.** Nearest 100 Hz egomotion sample, |dt| 2.6 ms, full 3D rotation. The same paint seen
   1…10 frames apart agrees to a median **0.01 m** at every gap (`code/align_diag.py`).
2. ⛔ **RETRACTED (said in chat): "a systematic over-ranging, one cause for every camera: the ground estimate sits 2–6 cm
   low".** The ground estimator does not move the number (median signed offset at 8–16 m, day / night: 10th percentile
   +0.37 / +0.20, median +0.35 / +0.30, obstacle-free band +0.36 / +0.19, robust plane +0.37 / +0.19).
3. ⛔ **And the "over-ranging" itself was my estimator's artefact.** The reference paint was cut at the same radius
   that defined "far", so a far point whose true counterpart lay beyond the cut was paired with the cut edge — a
   positive offset by construction. With a 3 m margin (only test points whose counterpart must lie inside the reference
   disc): **median signed −0.001 m, median |offset| 1.7 cm (day) / 2.8 cm (night), n 1.63 M / 1.47 M; pitch fits
   0.00°** (`code/align_bias2.py`). The lifting geometry is self-consistent.
4. **The real cause is semantic spill at range, worst at night.** Share of far-lifted paint landing within 0.5 m of the
   same paint seen from < 8 m (margin-corrected, `code/align_class.py`):

   | | 4–8 m | 8–12 m | 12–16 m | 16–22 m |
   |---|---|---|---|---|
   | crosswalk, night | 90.9 % | **65.4 %** | **53.6 %** | 52.8 % |
   | crosswalk, day | 94.4 % | 91.3 % | 92.5 % | 87.1 % |
   | line, night | 66.0 % | 51.2 % | 43.7 % | 60.7 % |
   | line, day | 98.2 % | 91.9 % | 69.0 % | 65.9 % |
   | arrow/text, night | 96.2 % | 72.6 % | 51.5 % | — |

   A far SAM3 mask a few pixels too tall is metres too long on the ground (≈ 0.2 m per pixel at 20 m for F0), and the
   v1/v2 BEV fused ±1 s of frames at every range — and the future half of that window drew paint the front image had
   not yet labelled. **Fix (v3 renderer): past 2 s only; hits from ≤ 10 m weigh 1, 10–25 m 0.35; per-cell class =
   highest weighted score; solid only with a near hit, faded otherwise.**

---

## 6. More examples, longer videos
* **Mining.** Alpamayo2 reasoning text over 4,729 clips: crosswalk mentioned in 501, arrow 338, hatch/chevron/gore 85,
  painted markings 667 → 1,448 candidates; 400 have the front camera and ego poses on disk; the top 160 were checked by
  SAM3 on 3 front frames each (`code/sam3_mine_clips.py`, 992 s): 113 with ≥ 2 on-road crosswalk hits, 48 with arrows.
* **Only 4 clips have all 7 cameras on disk** (chunk 0768; the other 96 lack the front camera), so the long video uses
  the **front camera only** with the ground taken from the ego's own path (−3…+6 s, spread ±6 m; `build_front_seq.py`):
  8 mined clips, v3 rules, joined with title cards (`front_chain.sh`, `make_long.py`).

## 7. Next levers, ranked by measured effect
1. Range-aware fusion everywhere (the v3 renderer's rule) also in the clip-vote label map, then re-score.
2. Per-class, per-range reliability measured per clip (the §5 table is itself self-supervised) as the vote weight.
3. Edge adjacency radius sweep (§2 warning).
4. A human spot-check of 30–50 frames of paint classes — the only way to score arrows, text and hatched areas; the
   nuPlan demo has 4 frames and no hatch or symbol class.
5. Throughput for dataset scale (20.6 s/frame): paint prompts only on views that see paint ahead/behind, 2 Hz for the
   static map, TensorRT SAM3.

## 8. Deliverable manifest
| artifact | where |
|---|---|
| this report | `RESULT.md` (repo) |
| code (33 files) | `code/` (repo) — clip ids redacted to sha12; the chain scripts ran on Thor under /home/nvidia/sam3map |
| LiDAR scores (day v1/v2, night v1/v2/v3), refine counts, demo prompt sweep, replay totals, ground-estimator and per-class placement logs, mining counts (sha12 keys) | `raw/` (repo) |
| stills: v1/v2/v3 frames, replay contact sheets, patch-rule checks, hood prompt | `media/` (repo, JPG) |
| videos v1 / v2 / v2.1 / v3 (day, night), long front-only video, v3.1 videos | Thor `/home/nvidia/sam3map/render_*.mp4`, `long_v3.mp4`; copies sent to the PI; `*.mp4` is git-ignored |
| per-frame npz (class rasters + lifted points + ranges), prompt bank, decoded LiDAR sweeps | Thor `/home/nvidia/sam3map/{<clip>,<clip>_v2,_v3raw,_v3,_v31raw,_v31}/`, `promptbank/`, `data/sweeps/` (not banked: size) |

---

## 9. v5 → v6.1: native cameras, all seven of them, one class per painted region, and a ground-truth export (2026-09-13 evening)

**PI direction (verbatim):** *"I prefer to finalize the sam 3 map generation and then use those as gt for our new bev
head. So let achieve the best possible sam3 based semantic map, let me confirm the quality by considering the videos,
then augment the data sets the maps at each frame using all cameras."* — this section is the "best possible map" step;
corpus augmentation waits for the PI's confirmation of the videos (and for download permission per file).

Evidence class: **MEASURED** (ours; Thor `/home/nvidia/sam3map`) unless a row says otherwise. Tier: none — label artifact.
Clips: day `4fbd97b6a4b7`, night `73495082f98b` (96 tokens × 7 native cameras each, LiDAR ground).

### 9.1 v5: the native 120° camera instead of the 63.7° virtual view

The virtual pinhole view Qwen-Drive needs (fx 1545 @ 1920) is a crop of the native f-theta camera (focal 932.7 px): a
1.66× interpolated upsampling. v5 projects through the native f-theta model (`code/camera_model.py`, round trip
1.12e-3 m) onto a smooth LiDAR ground (`code/ground_surface.py`; the per-2 m-cell lookup put seams through arrows).
Long video: 8 mined clips, 848 frames, front camera only (refine control 96/96 on every clip). Sent to the PI.

### 9.2 v6: all seven native cameras, a sidewalk / verge class, per-camera ego masks

| step | what | control / measurement |
|---|---|---|
| extraction | 18 prompts × CAM_FW, CL, CR (120°), RL, RR (70°), RT, FT (30°); class 7 = sidewalk ∪ grass not drivable; a per-camera SAM3 **evidence bitfield** (diag, crosswalk, line, dashed, solid, symbol, curb, sidewalk) so later rules need no SAM3 re-run | adding the bitfield changed **0 class pixels on all 7 cameras** and 0 points (re-smoke of token 41); ~21 s/frame with a free GPU |
| refine v6 | ego body per camera from SAM3 "car hood" + border-touching "car" voted over 8 frames (the car rule only if the ego moved ≥ 15 m) | day: FW 2.72 %, RL 4.50 %, RR 3.54 % of pixels, others 0 (matches the contact sheet); relift control **96/96** day and night |
| scorer | coverage through `camera_model` (native f-theta) instead of the pinhole views | control: the new code reproduces the banked pinhole coverage on 6 tokens × 80,000 cells with **0 mismatches** — after restoring the banked 0.5 m depth cut (the first version differed on 12–17 cells) |
| renderer v4 patch | native camera names + class 7 | re-rendering a banked v5 clip: **96/96 PNGs byte-identical** |

**LiDAR checks, clip vote (pre-registered rule §2), v2 → v6 (MEASURED, 96 frames each):**

| clip | arm | MAP_A2 ↓ | MAP_B ↑ | MAP_C ↑ | MAP_E ↑ | camera coverage |
|---|---|---:|---:|---:|---:|---:|
| day | v2 clip vote (7 virtual views) | 0.0663 | 0.9760 | 0.5584 | 0.5384 | 0.958 |
| day | **v6 clip vote (7 native cameras)** | **0.0225** | **0.9841** | 0.5455 | 0.5409 | **0.983** |
| day | v6 mirrored control | 0.4336 | 0.7117 | 0.4838 | 0.2170 | |
| day | Qwen fused ±2 s | 0.1715 | 0.9565 | 0.8961 | 0.4125 | |
| night | v2 clip vote | 0.0650 | 0.9689 | 0.4226 | 0.4056 | 0.958 |
| night | **v6 clip vote** | **0.1429** | 0.9732 | **0.1140** | 0.4462 | 0.984 |
| night | v6 mirrored control | 0.3226 | 0.6030 | 0.1722 | 0.2291 | |
| night | Qwen fused ±2 s | 0.2901 | 1.0000 | 0.9071 | 0.4358 | |

The pre-registered clip-vote bar (MAP_B ≥ 0.95, MAP_A2 ≤ fused ±1 s + 0.05, mirrored control below on B and E) holds on
v6 day: B 0.9841, A2 0.0225 ≤ 0.0256 + 0.05, mirror B 0.7117 / E 0.2170. ⛔ **On the night clip v6 FAILS the same bar on A2: 0.1429 > 0.0664 + 0.05** (B 0.9732 passes, mirror below) — v2 passed it (0.0650). v6 is a regression at night on A2 (×2.2) and MAP_C (0.42 → 0.11); §9.9 attributes it.

### 9.3 The paint trim was eroding wide paint — metric ground layers (renderer v5)

The v4 renderer kept a thin-class pixel only if a **31 px** white top-hat found it brighter than the asphalt. On token 41
(`code/tophat_probe2.py`), the share of SAM3 thin-paint pixels kept by a 127 px top-hat but NOT by 31 px:

| camera | 0–4 m | 4–8 m | 8–12 m | 12–20 m |
|---|---:|---:|---:|---:|
| CAM_CR (120°) | 0.39 | 0.33 | 0.29 | 0.25 |
| CAM_RR (70°) | 0.18 | 0.20 | 0.12 | 0.04 |
| CAM_RT (30° tele) | — | — | **0.69** | **0.55** |

Hatched stripes and the yield triangle kept only their outlines. A pixel-sized element cannot fit focal lengths from 930
to ~3500 px at every range; a metre-sized one can. **Renderer v5** (`code/sam3map_render_v5.py`): every frame × camera
becomes a 0.10 m metric layer (cell → camera model → SAM3 class, luminance from the MIP level matching the cell's pixel
footprint), paint = 1.5 m-disc white top-hat on the layer. Timing trap found on the way: a 127 px ELLIPSE top-hat costs
0.64 s per 1920×1080 image on Thor, a RECT one 0.03 s.

### 9.4 R4: one class per painted region, decided from the whole clip (`code/sam3map_consensus.py`)

**The failure (contact sheet, token 41, against the raw images):** one white hatched gore area was *crosswalk* on
CAM_RL and CAM_RR and *lane lines* on CAM_RT; dashed lane-line segments on CAM_CR were *hatched*; on front clip
`1f1f05ca011d` a row of yield triangles was *hatched*. The v3.1 ground test could not run where SAM3's line prompt missed
the stripes ("too_few" → prompt fallback), and one oblique view rarely sees a whole region.

**Method:** (A) every metric layer accumulates in one world grid: observation weight (camera factor / (1 + (r/8 m)²)),
bright paint (from views ≤ 15 m only), SAM3 class votes and prompt evidence; (B) regions = cells whose area-paint votes
are ≥ 35 % of their observation weight; on the FUSED paint: yield-teeth row → line; thin fragments along the axis → line;
else stripe orientation by a **structure tensor** (the v3.1 1 m-cell point test returned "too_few" on all 4 regions of the
front clip: a stripe wider than ~0.5 m is not elongated inside a 1 m cell) with the v3.1 thresholds; ambiguous → class
votes, front cameras ×3; (C) every pixel of classes 1/2/3/6 is lifted to its world cell and relabelled; points re-lifted
after the relift control.

| clip | regions (crosswalk / hatched / line) | pixels changed | control |
|---|---|---:|---|
| front `1f1f05ca011d` | 4 (1 / 3 / 0) | 32,128 | 96/96 |
| day | 8 (7 / 1 / 0) — the gore area is hatched in every camera | 7,984,377 | 96/96 |
| night | 6 (5 / 1 / 0) | 1,992,808 | 96/96 |

⚠️ **UNSCORED:** PhysicalAI has no paint ground truth. The effect is shown on before/after contact sheets (the gore area
on RL / RT, frame 38 and 41) and in the videos; the LiDAR map metrics do not see paint semantics and moved as expected
(day clip vote A2 0.0225 → 0.0223, B/C/E unchanged). The yield-teeth rule did not fire on the front clip (to2 = 0).

### 9.5 The ground-truth world map and its export to the BEV head's grids

The non-causal world map (renderer v5): every world cell at 0.10 m keeps the class of the NEAREST labelled observation
over all frames and cameras (labels may use future frames; inference never reads it). **Exporter**
(`code/sam3map_export_gt.py`) on the head's own contract (`2026-09-13-bev-lidar-corpus-and-head`, `lidar_bev.py`,
`bev_gt_loader.py`): rig frame +y LEFT; cartesian x 0–60 m, y ±16 m at 0.5 m [120, 64]; polar [24, 20] and polar48
[48, 40] with col 0 LEFT; a fine [600, 320] code raster at 0.10 m. Coarse cells carry soft class fractions (9 channels:
background, 7 classes, not seen; 5 × 5 sub-samples; sum 255). Each export asserts its content:

| check | front clip | day | night |
|---|---|---|---|
| ego's next 3 s of poses on drivable (real / mirrored) | 1.000 / 0.882 | 1.000 / 0.810 | 1.000 / 0.779 |
| polar48 vs cartesian drivable MAE (same frame / frame + 40 control) | 0.031 / 0.431 | 0.026 / 0.404 | 0.034 / 0.409 |
| seen share of the cartesian window · size | 0.902 · 1.01 MB | 0.924 · 1.06 MB | 0.919 · 1.18 MB |
| re-render → world map | — | **byte-identical** (after a legend fix) | |

**LiDAR checks of the ground-truth world map itself** (new scorer arm, day clip, 96 frames):

| arm | MAP_A2 ↓ | MAP_B ↑ | MAP_C ↑ | MAP_E ↑ | vehicle centres on sidewalk / verge |
|---|---:|---:|---:|---:|---:|
| **GT world map (v6.1)** | 0.0533 | **1.000** | **0.7727** | 0.4798 | 2.27 % (n 308) |
| GT world map, mirrored | 0.5015 | 0.7426 | 0.5519 | 0.2669 | |
| clip vote (points) | 0.0223 | 0.9841 | 0.5455 | 0.5409 | |
| night GT world map (v6.1) | 0.0854 | 1.000 | 0.5019 | 0.4044 | **27.5 %** (n 807) |
| night GT world map, mirrored | 0.4201 | 0.6213 | 0.2367 | 0.2445 | |

The world map keeps the road under moving vehicles (C 0.77 vs 0.55 for the clip vote) and never takes the ego off its
road; its cost is A2: a far "road" label beats a near view of a pole or a wall.

### 9.6 ⛔ Three pre-registered compositing arms FAILED (surface vote)

Bars committed in `code/sam3map_render_v5.py` before each run, identical for all three: GT MAP_A2 ≤ 0.03 AND MAP_C ≥ 0.75
AND MAP_B ≥ 0.99 AND MAP_E ≥ 0.48, mirrored control below on B, C, E. Day clip:

| arm | rule | A2 | B | C | E | verdict |
|---|---|---:|---:|---:|---:|---|
| v6.1 nearest labelled | (delivered) | 0.0533 | 1.000 | 0.7727 | 0.4798 | reference |
| v6.2 surface_vote | surface = range-weighted vote incl. background; paint / edges nearest | 0.0317 | 0.9411 | 0.5877 | 0.4798 | **FAIL** |
| v6.3 + agent occluders | tracked boxes (obstacle.offline) projected per camera = not seen | 0.0291 | 0.8271 | 0.6266 | 0.506 | **FAIL** |
| v6.4 + refine-zeroed = not seen, background weight 0.5 | | 0.0297 | 0.8866 | 0.6364 | 0.506 | **FAIL** |

Mechanism (world-map crops at frame 33): a vote removes poles and walls, but wherever the near views are blocked — the
ego's own lane between a lead car and a tailgating van — background wins and the road is lost. The occluder masks
themselves are right (`code/occ_debug.py`: truck, van and distant cars covered in all 7 cameras), which also confirms
camera models and LiDAR boxes agree. **v6.1 stays the ground truth.** Next lever, not run: LiDAR tall obstacles written
into the label — admissible for labels, but MAP_A2 reads the same LiDAR, so it needs an independent check first.

### 9.7 Cross-camera registration: measured, corrections NOT applied

`code/cross_cam_registration.py` correlates each camera's bright paint with the fused paint of the other cameras at rig
shifts of ±0.4–0.8 m. ⛔ **Its first version was invalid**: it selected cells by "own paint OR the others' paint", i.e.
on the correlated variable; NCC at zero shift came out negative and every peak sat at the range edge (selection bias).
Corrected (selection by the camera's own SAM3 classes, reference over all observed cells), partial day clip:
FW (0, +0.1 m) 0.553 → 0.582, CR (0, −0.1 m) 0.452 → 0.475, CL (+0.3, −0.4 m) 0.407 → **0.577**, mirrored controls ≈ 0.

`code/calib_refine.py` then fitted small rotations + height per camera against CAM_FW. On the full day clip CR (yaw
+0.25°, roll −0.5°, dz +0.05) and RR (yaw +2.25°, dz +0.10) raise held-out NCC 0.42 → 0.58 and 0.24 → 0.48 — but fits on
the **temporally disjoint halves disagree** (RR yaw −0.25° vs +1.5° vs +2.25°; only CR yaw stays positive, +0.25…+0.75°),
and the even/odd held-out split was not independent (adjacent frames 200 ms apart). A range-split self-consistency mode
was **degenerate** (FW pitch +2°, CR yaw of the opposite sign; the near/far split moves with the candidate). ⇒ relative
misregistration exists (up to ~0.5 m at 12 m for CL), but these corrections are not identifiable from paint on one clip.
Next lever: a joint fit over several clips of the same vehicle, or a LiDAR-curb anchor.

### 9.8 Camera time offsets (measured, small here)

Per token, the nearest native frame differs from the reference time by a constant +10.7 ms (CL, CR, RT; day) and +4.1 ms
(night), 0 ms for FW, RL, RR, FT. At these clips' ≤ 8.1 m/s that is ≤ 0.09 m — below the 0.10 m cell; at 30 m/s it would be
0.32 m, so a corpus-scale build should use the pose at each camera's own timestamp.

### 9.9 The night regression, attributed per camera (`code/a2_by_camera.py`)

Same-frame evidence, every second frame: the share of ALL LiDAR tall-obstacle points of the frame whose 0.15 m cell a
single camera's lifted drivable points cover, by camera range (a lower bound per camera; comparable across cameras):

| clip | largest | next |
|---|---|---|
| night | **CAM_FW 8–15 m: 2.81 %**, CAM_FW 0–8 m: 1.26 % | CAM_RL 8–15 m 1.01 %, CAM_FT 8–15 m 0.92 %, CAM_CR 0–8 m 0.73 % |
| day | CAM_RL 8–15 m: 1.46 % | CAM_CR 8–15 m 0.67 %, CAM_RR 8–15 m 0.59 %; CAM_FW 8–15 m only 0.13 % |

At night the **front camera's road mask spills onto obstacles at 8–15 m** — 21× its day value. HYPOTHESIS (not yet tested):
SAM3 sees fewer pixels per metre of far road in the full 120° image than it did in v2's 63.7° crop, which acted as a zoom,
and night blur makes that worse. Night MAP_C is also confounded by
definition: many night vehicles stand in parking bays that SAM3 does not call road (27.5 % of vehicle centres on sidewalk
/ verge at night vs 2.3 % by day), while MAP_C was calibrated on nuPlan, where parking lots are drivable.

## 10. Next levers after v6.1 (ranked)

0. **Night far-range road masks** (§9.9): a zoomed centre tile of the native front camera for SAM3 (roughly 2× extraction
   time), pre-registered on night A2 with the day clip as a no-regression control.
1. **PI confirmation of the v6.1 videos**, then the corpus build: the B1 EVAL clips need all 7 camera streams (download
   permission per file, with size) — the exporter already writes the BEV head's grids.
2. **Paint accuracy is unscored** — a 30–50 frame human spot-check per class (arrows / text / hatched / crosswalk).
3. A2 of the world map (0.053): LiDAR tall obstacles in the label, with an independent validation.
4. Multi-clip joint camera calibration (§9.7) and per-camera timestamps (§9.8).
5. Throughput: ~21 s per frame × 7 cameras on Thor for extraction; consensus 4 min, render 3 min per clip.

### 10.1 Deliverable manifest (v5 → v6.1 additions)

| artifact | where |
|---|---|
| code (43 files, all new or changed since the banked `code/`) | `code/v6/` (repo) — clip ids redacted to sha12; run on Thor under /home/nvidia/sam3map |
| LiDAR scores v6 / v6.1 day and night, GT world-map arms incl. the three FAILED surface votes, consensus / refine / calibration / registration JSONs, controls | `raw/v6/` (repo) |
| ground-truth exports for the BEV head (both LiDAR clips) + reports | `gt_samples/*.sam3mapgt.npz` (repo, ~1.1 MB each) |
| contact sheets, probes, atlases, world-map crops, video stills | `media/v6/` (repo, JPG) |
| videos: v5 long front (8 clips), v6 and v6.1 day and night | Thor `/home/nvidia/sam3map/long_v5.mp4`, `sam3map_v6_<c8>.mp4`, `sam3map_v61_<c8>.mp4`; sent to the PI; `*.mp4` git-ignored |
| per-frame npz v6raw / v6 / v61, native 7-camera sequences, world maps | Thor `/home/nvidia/sam3map/<c8>_v6raw`, `_v6`, `_v61`, `native7/`, `render5_*/worldmap.npz` (not banked: size) |

---

## 11. The night regression, three pre-registered levers later (2026-09-13 night)

Evidence class **MEASURED** (Thor; LiDAR checks as §2, 96 frames per clip). Bars, identical for every arm below and
committed in each script before its run: **(a)** night clip-vote MAP_A2 ≤ that arm's own fused ±1 s + 0.05 (the bar v6
failed), **(b)** day clip-vote MAP_A2 rises by ≤ 0.01 over v6 (0.0225), **(c)** MAP_B ≥ 0.95 on both clips.

| arm | what changes | night A2 (bar) | night B | day A2 | day B | verdict |
|---|---|---:|---:|---:|---:|---|
| v6 | — | 0.1429 (0.1164) | 0.9732 | 0.0225 | 0.9841 | FAIL (a) |
| **v6z** zoom tile (`code/v6/sam3map_zoom_fw.py`) | SAM3 re-run on CAM_FW's central 960×540 upsampled 2× | 0.1430 (0.1161) | 0.9732 | 0.0225 | 0.9841 | **FAIL (a)** — changed ~0.3 % of CAM_FW pixels and no metric |
| **v6r** rectified front road (`code/v6/sam3map_rectified_road.py`) | CAM_FW drivable decision from the banked rectified v2 view of the same camera centre | 0.1331 (0.1124) | 0.9714 | 0.0224 | 0.9805 | **FAIL (a)** |
| **v6n** rectified road, all cameras + no tele drivable | FW/CL/CR/RL/RR drivable from their rectified v2 views; FT/RT drivable dropped | 0.1003 (0.1150) | 0.9673 | 0.0223 | 0.9762 | **PASS (a)(b)(c) on the clip vote — NOT adopted**, see below and §13 |

⛔ **v6n passed the bars it was given and is still not adopted.** The bars were written on the clip-vote arm; the
deliverable is the whole-clip world map, and the world map built from v6n's rasters is WORSE on the same night clip:
MAP_A2 0.1450 vs 0.0854 (v6.1), and on the robust form of §13 A2r_onroad **0.1096 vs 0.0502** (`raw/v6b/a2_robust2_night_v6b.txt`).
The bars measured the wrong artifact. v6r and v6z leave the world map unchanged (A2r_onroad 0.0503 / 0.0503).

**Controls that make the v6r / v6n mapping trustworthy:** every v2 virtual view has the SAME centre as its native camera
(F0←FW, L0/L1←CL, R0/R1←CR, L2←RL, R2←RR; translations identical, FW↔F0 rotation 0.79°), so native pixels map to virtual
pixels by ray alone; image intensity through that mapping correlates at **0.998 night / 0.997 day** vs **0.785 / 0.862**
for a 40 px shifted control.

**Where the night bleed lives** (`code/v6/a2_fov_split.py`; share of all tall-obstacle points covered by a camera part's
drivable, summed over cameras and ranges):

| clip | inside the v2 views | periphery + tele cameras | largest parts |
|---|---:|---:|---|
| night | 6.17 % | **3.62 %** | FW inside 8–15 m 1.73 % (v2 F0: 1.43 %), FW periphery 8–15 m 1.10 %, FT 8–15 m 0.92 %, RL inside 8–15 m 0.93 % (v2 L2: 0.66 %) |
| day | 3.45 % | 0.50 % | RL inside 8–15 m 1.45 %, CR inside 8–15 m 0.62 % |

⇒ at night the native fisheye masks spill 20–40 % more than the rectified ones **inside the same field of view**, and the
periphery and tele cameras add ~37 % of the total; by day the periphery is clean. Magnification alone (v6z) is refuted.
⚠️ **Read with §13:** these shares count mapq_core's tall points, whose minimum-z ground turns road-surface returns into
"obstacles" on this corpus (half of the night hits). Comparisons between camera parts on the same points stand; the absolute
night "bleed" does not.

⛔ **Incident, caught before any number was reported as a result:** the first v6r day control ran on **20 of 96 frames**.
The banked rectified-view extraction of the day clip (`<c8>_v3raw`) had been stopped after frame 19 earlier in the day;
the rectification crashed at frame 20, and refine, consensus and both LiDAR scores then ran on what existed (day "MAP_B
0.1078"). The chain logged `raw20` but gated nothing on it. All 15 outputs were renamed `INVALID20_*` on Thor (not
deleted), the missing 76 frames were extracted with the same extractor file (md5 `9eec3014`, the version the first 20
frames came from), and the day arms re-ran. ⇒ a control's first check is its INPUT COUNT; the chains now wait for all 96
frames before starting.

⇒ None of the three levers is adopted. §13 then shows that most of the night "bleed" these levers chased was the metric.

---

## 12. v6.5 — the PI's three questions answered on the map itself (2026-09-13 night)

The PI's review of the night comparison at t = 17.2 s: *"why is the exit of the roundabout noisy? why is the space between
the cross walk markings colored? ... the left curb in the roundabout was not always detected"* and *"can you prompt sam to
just color the markings belonging to cross walk, not the space between"*. Each question got a lever **and a measurement
that answers it** (`code/v65/pi_checks.py`, whole clip, cells within 30 m of the driven path; evidence class MEASURED,
Thor). The curb reference is LiDAR only (ground height steps 0.08–0.35 m, accumulated over 96 frames, components ≥ 2 m),
independent of SAM3; no map variant below uses LiDAR at label time.

| night clip, all 7 cameras | noise fragments / 1000 m² ↓ | crosswalk: coloured share of the crossing area (stripes only ≈ 0.5) | LiDAR curbs with an edge ≤ 0.6 m ↑ | map edges on a LiDAR curb / obstacle ↑ | lane-line area |
|---|---:|---:|---:|---:|---:|
| v6.1 nearest labelled view | 170.5 | 0.699 | 0.499 | 0.680 | 16.6 m² |
| v6.5s weighted majority + SAM3 "crosswalk stripe" prompt | 91.2 | 0.778 | 0.726 | 0.687 | 13.3 m² |
| v6.5c + tracked vehicle footprints drivable + 5×5 surface mode | 71.5 | 0.781 | 0.801 | 0.666 | 20.4 m² |
| **v6.5d** + adaptive paint threshold + stripes trimmed to bright paint | **69.7** | **0.507** | **0.801** | **0.666** | **31.4 m²** |
| front camera only, v6.1 → v6.5d | 152.8 → 105.5 | 0.908 → 0.606 | 0.549 → 0.788 | 0.732 → 0.617 | 0.7 m² (v6.5s) → 23.5 m² |

**The levers, each with the measurement that motivated it:**
* **Noise → weighted majority** (`sam3map_render_v5b.py`, composite `majority_v65`): 63 % of BEV cells receive two or more
  classes over the clip and the nearest view picked a minority label in 6–8 % of cells (`exit_edge_xwalk_diag.py`). Weight
  1/(1+(r/8 m)²) per labelled observation; background never votes.
* **Crosswalk gaps → stripe prompt** (`sam3map_xwalk_stripes.py`): on 9 crosswalk views "crosswalk stripe" (≥ 0.5) covers 72 %
  of the stripe paint and 6.5 % of the gaps (the area prompt "crosswalk": 70 % of the gaps); used inside the detected
  crossing only. The world map still filled gaps (0.78) until each camera layer's stripe cells were trimmed to paint brighter
  than the asphalt (v6.5d, 0.507).
* **Curb gaps → boundary edges**: along CAM_FW frames 76–92 the island's drivable boundary lay 23–50 px from any SAM3
  grass / curb mask, so the per-camera edge rule (≤ 3 px) never fired. Edges are now drawn where the consolidated
  sidewalk-verge meets the consolidated drivable surface (5×5 openings, components ≥ 1 m) plus cells labelled edge in ≥ 25 %
  of their near observations.
* **Night lane lines → adaptive paint threshold** (`night_paint_probe.py`): night CAM_FW line cells have a 1.5 m top-hat
  median of 2–4 grey levels (road 90th percentile 2–3, road MAD 0.2–1.0); the fixed floor of 10 kept **1–4 %** of them.
  Rule v6.5d: paint if the top-hat exceeds max(3, the layer's road 95th percentile).
* **Vehicles on sidewalk → footprints drivable**: 27.5 % of tracked vehicle centres sat on sidewalk-verge (parking lanes).
  Label-time boxes; MAP_B and MAP_C then read the same evidence and read 1.0 by construction.

**Display — the ground-truth map drawn into the cameras** (`gt_reproject.py`, `compare_gt_reproj.py`): every camera pixel's
ray meets the smooth LiDAR ground and shows the world-map class under it; tracked agent boxes, the ego body and LiDAR returns
standing in front of the ground are left unpainted. What the PI sees in a camera is now exactly the BEV label, not a
per-frame SAM3 mask.

⚠️ **Not a checked quantity: paint.** PhysicalAI's LiDAR intensity does not separate paint from asphalt (ground returns:
road p50 9 / p90 13, lines 11 / 15, crosswalk 11 / 15, sidewalk 13 / 31; `lidar_intensity_probe.py`), so the LiDAR paint
check in `pi_checks.py` is uninformative here and is not quoted. Paint quality rests on the PI's visual review.

---

## 13. ⛔ Correction: most of the night MAP_A2 was the metric, not road spill

**RETRACTED** (class: *a probe calibrated on one corpus quoted on another* — the `df` / Thor-`free` family): *"night road
spill"* as the diagnosis of the night clip's MAP_A2 (§11, and the PI messages of 2026-09-13). MEASURED on the night clip:

1. `mapq_core.split_points` takes the ground as the **minimum** z of a 2 m cell. 6.7 % of night 2 m cells (day 2.9 %) hold a
   return > 0.5 m below the cell's 10th percentile (`ground_ghost_probe.py`), so the "ground" drops and road-surface returns
   and car bodies count as tall obstacles.
2. The hit picture of the v6.5s map showed car-sized blobs in the middle of the roundabout lanes; in rig coordinates the top
   bins were a following car at x −7…−8 m and a lead car at x +10 m (`a2_rigpos.py`), and 40.5 % of hit cells were hit in a
   single frame.
3. **50–57 % of MAP_A2 hits on the all-camera maps are road-surface returns** (within 0.3 m of the 10th-percentile ground).

**Robust form A2r_onroad** (reported beside MAP_A2, never instead of it): ground = 10th percentile of the 2 m cell, agent boxes
grown 1.0 m, hits on any on-road class. Night clip:

| map | MAP_A2 | A2r_onroad | hits on road-surface returns |
|---|---:|---:|---:|
| v6.1 nearest, all 7 | 0.0854 | **0.0502** | 56.8 % |
| v6.5s majority + stripes | 0.1051 | 0.0708 | 53.9 % |
| v6.5c + agents + mode | 0.1119 | 0.0742 | 50.9 % |
| v6.5d + paint rule | 0.1149 | 0.0742 | 50.2 % |
| v6.1 nearest, front only | 0.0999 | 0.0519 | 38.4 % |

⚠️ **A real regression survives the correction: the majority composite adds +0.021 A2r_onroad** (0.0502 → 0.0708), and
88.6 % of the robust hits lie within 1 m of the road boundary (`a2_attrib.py ROBUST=1`). Hypothesis: several far "road"
views outvote one near sidewalk view, and near views of poles and walls never vote. Two pre-registered arms follow (§14).

---

## 14. Obstacles at the road border — pre-registered arms on top of v6.5d (night clip)

Bars, committed in `code/v65/sam3map_render_v5e.py` before any arm ran, identical for every arm: the night clip, all 7
cameras, **A2r_onroad ≤ 0.062 AND noise fragments ≤ 90 / 1000 m² AND LiDAR curb recall ≥ 0.75 AND crosswalk coloured share
≤ 0.60** (v6.5d: 0.0742 / 69.7 / 0.801 / 0.507; v6.1 nearest: 0.0502 / 170.5 / 0.499 / 0.699).

| arm | lever on top of v6.5d | A2r_onroad | fragments | curb recall | crosswalk share | verdict |
|---|---|---:|---:|---:|---:|---|
| e1 | every vote weighted 1/(1+(r/4 m)²)² — near views dominate | 0.0645 | 84.7 | 0.702 | 0.492 | **FAIL** (A2r, curb recall) |
| e2 | a "no class" observation within 10 m votes background; tracked agent boxes masked out of every camera | **0.0194** | 105.5 | 0.542 | 0.530 | **FAIL** (fragments, curb recall) |
| e3 | e2 + edges where near background meets the road + background-won specks < 0.5 m² reverted | 0.0194 | 99.1 | 0.744 | 0.530 | **FAIL** (fragments, curb recall) |
| e4 | e3 + every enclosed background / sidewalk speck < 0.5 m² takes its ring's majority surface | 0.0194 | 78.1 | 0.744 | 0.530 | **FAIL** (curb recall) |
| **e5** | e4 + near-background cells not opened before drawing edges | **0.0194** | **77.9** | **0.759** | **0.530** | **PASS** (all four) |

**What e2 teaches, measured:** near views of poles and walls voting against the road cut the robust obstacle rate to 0.0194 —
2.6× below even the nearest-view map (0.0502) — which confirms the §13 hypothesis about the majority's regression. It failed on
two things it broke, both traced before e3 was written: its extra fragments are **background specks** (class-0 fragments
123 → 293) and its lost curbs are borders where **background, not sidewalk, now meets the road** — the boundary-edge rule only
drew sidewalk/road borders (map edge cells in LiDAR coverage 3,312 → 2,233).

e3's background edges brought the curb recall back to 0.744 (edge cells 2,233 → 3,327) at edge precision 0.629, but its
specks were mostly cells with **no labelled vote at all** (background fragments 283; the background-won revert touched only
1,248 cells) — hence e4's enclosed-speck fill and, in parallel, e5's un-opened background edges.

**e5 passes all four bars on the night clip** — the robust obstacle rate falls **3.8× below v6.5d and 2.6× below the nearest-view
map**, while the noise, curb and crosswalk gains of §12 are kept. ⚠️ **Costs, reported beside the pass:** edges are less precise —
map edges on a LiDAR curb / obstacle 0.666 → 0.635 (`pi_checks.py`) and the scorer's per-frame MAP_E 0.5218 → 0.4710 — and the
masked agent boxes leave never-seen shadows behind parked cars in the BEV. Front camera only, e5: A2r_onroad 0.0547, fragments
120.0, curb recall 0.743. The day clip gets its own no-regression bars, committed before its run (`v65e5_day.sh`): A2r_onroad ≤ day v6.5d's, fragments ≤ 105.1, curb recall ≥ 0.667,
crosswalk share ≤ 0.60 — **e5 FAILS the day clip on curb recall (0.600)** while passing the other three (0.0084 / 90.3 / 0.536).

| arm | lever | night A2r_onroad / frag / curb / xw | day A2r_onroad / frag / curb / xw | verdict (both clips) |
|---|---|---|---|---|
| e6 | e5, but background may veto the ROAD only (a sidewalk-verge vote is never outvoted) | 0.0237 / 79.0 / 0.779 / 0.530 ✓ | 0.0084 / 91.4 / **0.626** / 0.536 | **FAIL** (day curb recall) |

Every background-vote arm loses 6–9 points of day curb recall; the night obstacle gain is robust across e2–e6.

---

## 15. The PI's question on the video: "why are there uncolored areas near to the objects?" — and a box-free map

**MEASURED** (`occluder_check.py`, day clip, 10 vehicle boxes within 20 m): the tracked boxes' z lies **0.43–1.28 m below the
smooth LiDAR ground under them (median 0.99 m)** while z + h matches the LiDAR roof (+0.19 m). The display's agent mask — and the
masking in arms e2–e6 — spanned z − h/2 − 0.2 … z + h + 0.2, i.e. down to ~2 m under the road, so it blanked the asphalt in front
of every vehicle (the PI's big rectangle under the van). The small squares were LiDAR returns marked in 8-px blocks grown by one
block; the grey square on the road is an object on the car inside the extraction's ego mask.

**PI, verbatim:** *"you don't need the boxes, scince sam 3 is seperating the roads from the rest very well"*. Applied twice:
* **Display v2** (`compare_gt_reproj_v2.py`): the colour is still the BEV ground truth; WHERE it is drawn now comes from that
  frame's own SAM3 raster (road-level classes, dilated 4 working px for thin edges). No boxes, no LiDAR occluders.
* **Box-free maps**: d0 = v6.5d without vehicle footprints; e6nb = e6 without any box masking (pre-registered in
  `sam3map_render_v5i.py`, same bars as e6, both clips).

| all 7 cameras | fragments / 1000 m² | crosswalk share | curb recall | edge precision | A2r_onroad | MAP_C vehicles on road | vehicles on sidewalk |
|---|---:|---:|---:|---:|---:|---:|---:|
| night v6.5d (vehicle boxes) | 69.7 | 0.507 | 0.801 | 0.666 | 0.0742 | 1.000 (by construction) | 0.0 |
| **night d0 box-free** | **67.0** | **0.507** | **0.814** | **0.668** | 0.0723 | 0.503 | 0.270 |
| night e6nb box-free | 39.5 | 0.509 | 0.816 | 0.705 | 0.0391 | **0.103** | 0.270 |
| day v6.5d (vehicle boxes) | 105.1 | 0.564 | 0.687 | 0.857 | 0.0094 | 1.000 (by construction) | 0.0 |
| **day d0 box-free** | **94.6** | **0.564** | **0.800** | **0.822** | 0.0093 | 0.776 | 0.020 |
| day e6nb box-free | 80.9 | 0.569 | 0.767 | 0.806 | 0.0076 | **0.679** | 0.020 |

* **Dropping the boxes improved the curbs on both clips** (day 0.687 → 0.800, night 0.801 → 0.814): the forced vehicle
  footprints had overwritten sidewalk and edges beside parked cars at the kerb.
* **e6nb PASSES all eight pre-registered bars** (night 0.0391 / 39.5 / 0.816 / 0.509; day 0.0076 / 80.9 / 0.767 / 0.569) — and
  is **NOT adopted**, on a quantity its bars did not contain: vehicles seen near now vote the road away under and behind them,
  MAP_C falls 0.503 → 0.103 at night and 0.776 → 0.679 by day, and the day BEV shows a background trail along the lane where a
  vehicle followed the ego. ⚠️ The gap is in the pre-registration: MAP_C was left out because the boxed arms read 1.0 by
  construction. The PASS is reported as written; the adoption decision rests on MAP_C and the BEV sheet
  (`media/v65/hard_frames_nobox_*.jpg`), stated here rather than folded into the bars after the fact.
* **Delivered: d0** — box-free, display v2. Open, with numbers: vehicles on sidewalk-verge 27 % at night (parking lanes seen
  only past the cars), the majority's border obstacles (A2r_onroad 0.072 vs 0.050 nearest), fragmented night stripes.


### Deliverables of sections 11-15 (where they live)

| artifact | location |
|---|---|
| v6z / v6r / v6n levers and chains | `code/v6b/` (repo) |
| v6.5 renderers (majority, stripes, agents, paint rule, border arms e1-e6nb, box-free), reprojection display v1/v2, PI checks, A2 probes | `code/v65/` (repo) |
| scores, PI checks, robust A2, render stats, chain logs, probe outputs | `raw/v6b/`, `raw/v65/` (repo) |
| hard-frame sheets, A2 hit maps, reprojection stills | `media/v65/` (repo, JPG) |
| world maps (`worldmap.npz`), comparison frames and MP4 videos | Thor `/home/nvidia/sam3map/render5_*`, `reproj*_*/`, `reproj*_*.mp4` (not in repo: *.mp4 is git-ignored) |

---

## 16. Filling what a vehicle hid — two pre-registered box-free arms (2026-09-14)

On the delivered box-free map d0, road that a vehicle covered for the whole clip stays "seen, no class": a grey strip where a
van followed the ego (day), and parking lanes seen only past parked cars (night: 27 % of tracked vehicle centres on sidewalk or
grey, MAP_C 0.503). Bars, committed in `code/v65/sam3map_render_v5j.py` before any run, identical for both arms, BOTH clips
(all 7 cameras): **MAP_C ≥ d0 + 0.03 AND A2r_onroad ≤ d0 + 0.005 AND LiDAR curb recall ≥ d0 − 0.02 AND fragments ≤ d0**.

| arm | lever on d0 | night C / A2r_onroad / curb / frag | day C / A2r_onroad / curb / frag | verdict |
|---|---|---|---|---|
| d0 (delivered) | — | 0.503 / 0.0723 / 0.814 / 67.0 | 0.776 / 0.0093 / 0.800 / 94.6 | reference |
| d1 | seen-no-class cells inside the 2.5 m closing of the road become road (strips with road on both sides) | 0.503 / 0.0725 / 0.814 / 66.4 | **0.893** / 0.0094 / 0.800 / 94.0 | **FAIL** (night C unchanged) |
| d2 | d1 + seen-no-class cells within 2.5 m of both road and sidewalk become road; fills before edges | 0.683 / **0.1679** / 0.804 / 64.8 | 0.896 / 0.0097 / 0.806 / 91.0 | **FAIL** (night A2r_onroad 2.3×) |

* d1 fills what it was built for — the day lane under a following vehicle (2,950 cells, MAP_C +0.117) — and nothing at night,
  because night parked cars have road on one side only. The day gain is not adopted: the pre-registered bar required both clips.
* d2's parking-lane fill painted 32,324 night cells (323 m²) as road, and the robust LiDAR check shows what stood there: tall
  obstacles on road 0.0723 → 0.1679. Between the road and the sidewalk there are more than parked cars.
* ⇒ **Delivered map unchanged: d0.** What a box-free map cannot know is what stands under a vehicle; the next lever needs
  evidence of the ground under it (LiDAR ground returns before or after the vehicle passes), not a geometric fill.

---

## 17. The PI's review of the box-free videos: hatched area in red, a noisy yellow/red border, stripes not separated (2026-09-14)

**PI, verbatim:** *"Why is the 'Sperrfläche' marked as red? Why is the separation between yellow and red so noisy? it was better in
the past. The stripes of the cross talk are also not well enough separated."* Evidence class MEASURED (Thor); every rule below is
**exploratory, not pre-registered** — the PI's visual review is the criterion, and each check was chosen while exploring.

**Instrument first.** The renderer of the delivered map d0 now saves its per-cell vote fields (`code/v65/sam3map_render_v5m.py`,
FIELDS=1, float32), and `code/v65/compose.py` rebuilds the map from them in seconds. Control: with no options the offline map
**equals d0 cell for cell on all four maps** (night / day × all 7 / front only). Every variant below differs only in composition.

**What each complaint was, from the fields** (`media/v65r/spot_*.jpg`):
* **Yellow inside the red kerb edge (day island):** no painted line lies along that kerb. SAM3 calls the bright kerb face a lane line
  in some views, and the adaptive paint threshold of v6.5d (max(3, road p95), adopted for night lane lines) keeps it. v6.5s, still on
  the old floor of 10, had a clean red edge there.
* **Red on the hatched area (night exit):** a white SUV stood on it at the crossing for most of the clip, so it is mostly "seen, no
  class"; a few views saw hatched paint, some saw sidewalk. Red boundary edges were drawn around those sidewalk fragments on the
  road, and observed-edge cells landed inside the road: **228 edge cells at night (553 by day) have no non-drivable cell within 0.2 m.**
* **Stripes not separated:** the delivered stripe cells sit on the paint but cover a small part of each crossing, as fragments.

**Composition rules adopted in map r** (`compose.py LINE_EDGE_GAP=1 LINE_MIN_LEN_M=1.0 WLK_ISLAND_M2=3 EDGE_OBS=near XWALK=dirclose XWALK_LEN=11`):
a lane-line cell touching sidewalk / no-class is road; lane-line pieces shorter than 1 m are road; sidewalk fragments < 3 m² with a
≥ 70 % road ring are road (before edges); observed edges only next to a non-drivable cell and never over paint; stripe cells joined
**along their own direction only** (local structure tensor, 1.1 m line element).

| all 7 cameras, d0 → r | night | day |
|---|---|---|
| edge cells with no non-drivable cell within 0.2 m (red inside the road) | 228 → **0** | 553 → **3** |
| lane-line / edge 4-neighbour contacts (yellow inside red) | 125 → **1** | 701 → **19** |
| noise fragments / 1000 m² | 67.0 → **25.4** | 94.6 → **27.8** |
| LiDAR curbs with an edge ≤ 0.6 m | 0.814 → 0.802 | 0.800 → 0.789 |
| map edges on a LiDAR curb / obstacle (pi_checks) · scorer MAP_E | 0.668 → 0.671 · 0.507 → 0.522 | 0.822 → 0.859 · 0.692 → 0.731 |
| A2r_onroad | 0.0723 → 0.0734 | 0.0093 → 0.0093 |
| crosswalk: image top-hat on bars / on gaps (single images) · bar share of crossing pixels | 1.544 → 1.517 · 0.071 → 0.087 | 2.405 → 2.387 · 0.039 → 0.052 |

⚠️ **Front camera only, night: curbs with an edge 0.792 → 0.717** (edge precision 0.614 → 0.594) — the one-camera arm loses the
observed edges it relied on. Day front only: 0.852 → 0.838 at precision 0.882 → 0.926.

### Stripes — five rules measured against the images, and why none of them solves it
The check (`code/v65/xwalk_reproj_contrast.py`): through the video reprojection, the image white top-hat on each map's crosswalk
pixels vs on the road pixels of the **same** crossing, every 2nd frame, front and cross cameras, single images — bars that sit on paint
read high, bars on asphalt read ~1.0. Full table: `raw/v65r/xwalk_contrast_all_variants.txt`.

| rule | night bar/gap · bar share | day bar/gap · bar share | verdict |
|---|---|---|---|
| d0 delivered (majority votes, stripe prompt, trimmed to bright paint) | 1.544 · 0.071 | 2.405 · 0.039 | on paint, fragmented |
| **periodic bar fit per crossing** (spectral peak, phase folding) | **1.070** · 0.295 | **0.952** · 0.325 | **looks clean, not on the paint** |
| periodic fit in 4.5 m windows | 1.121 · 0.169 | 1.308 · 0.257 | not on the paint |
| stripes from the image evidence (oriented smoothing + Otsu) | 1.511 · 0.157 | 2.115 · 0.071 | on paint, but mostly bands and blobs, not bars |
| … bar-shaped pieces only (≤ 0.9 m wide, ≤ 30° off) + votes | 1.536 · 0.077 | 2.346 · 0.043 | back to d0's coverage |
| paint votes from views ≤ 12 m only | 1.315 · 0.055 | 2.283 · 0.022 | worse |
| **along-stripe join 1.1 m (r)** | **1.517 · 0.087** | **2.387 · 0.052** | adopted: +23–33 % bars, still on paint |

* The zebra IS periodic in the fields (`stripe_fft_probe.py`: peaks of 0.8–1.6 m period at 8–43× the ring median, shuffled control
  2–3.6×) — and a fitted pattern still lands off the paint. ⇒ **a clean-looking stripe rendering is not evidence that it is right;
  only the single-image check separated the two.**
* **SAM3 at 2× zoom** (`stripe_zoom_test.py`, night front camera): at 15–25 m it finds no stripes on the full image and almost none
  on the crop; at 5–12 m the crop adds ~24 % stripe pixels at the same top-hat contrast. At night the limit is the paint's
  visibility beyond ~12 m, not the model's input resolution.

**Delivered: map r** (videos `reproj2_v65r_*.mp4` on Thor). Open, with numbers: stripe coverage 5–9 % of the crossing pixels; the
hatched area under the SUV stays largely unseen; front-only night curb recall −0.075.

---

## 18. Crosswalk stripes from zoomed SAM3 crops, softer trimming, one keyframe — three pre-registered arms, three FAILs (2026-09-14)

**PI, on the zoom test at night frame 84:** *"84 und zoom crop seem to be good, are we taking these?"* — they were not in the map
(a three-frame test). Three levers followed, each with bars written before its run and read against map r (§17) with the same
single-image check (`xwalk_reproj_contrast.py`: image white top-hat on bar pixels / on gap pixels of the same crossing, every 2nd
frame, front and cross cameras) plus the PI checks. Evidence class MEASURED (Thor); raw in `raw/v6sz/`.

| arm (all 7 cameras) | pre-registered bar | night: bar/gap · bar share | day: bar/gap · bar share | verdict |
|---|---|---|---|---|
| map r (delivered, §17) | — | 1.517 · 0.087 | 2.387 · 0.052 | reference |
| **zoomed SAM3 crops** — full-image masks OR crops around every crossing ≤ 20 m (`sam3map_xwalk_stripes_zoom.py`, 853 / 798 crops), full pipeline re-run | both clips: share ≥ 1.3× r, bar/gap ≥ r − 0.05, fragments ≤ r + 5, curb recall ≥ r − 0.02 | 1.537 · **0.091** | 2.351 · **0.062** | **FAIL** (share, both clips) |
| **softer trimming** — untrimmed stripe votes ≥ 0.15 of road weight with mean top-hat / threshold ≥ 0.6 (setting chosen on the night clip out of three) | day clip (the test): same four bars | 1.535 · 0.122 (tuned) | **2.310** · 0.073 | **FAIL** (day bar/gap < 2.337) |
| **one keyframe per crossing** — stripes from the frame / camera with the most stripe points 4–12 m, no fusion (`xwalk_keyframe.py`) | both clips: bar/gap ≥ r − 0.05 AND share ≥ 1.3× r | **1.451 · 0.033** | **1.901 · 0.040** | **FAIL** (both, both clips) |

Noise fragments and LiDAR curb recall stayed inside their bars for the zoomed and softer arms (night 25.4 / 28.3, curb 0.802; day
29.9 / 26.5, curb 0.789).

**What the three failures say together.**
* Zoomed crops add **3.7 % (night) / 5.2 % (day) stripe pixels over the whole clip** — the +24 % of frame 84 does not generalise:
  near views already resolve the bars, and at night far crossings show nothing to find.
* Softer trimming raises coverage 40 % but by day puts more stripe cells on asphalt than the bar allowed.
* ⭐ **The keyframe arm localises the defect.** Bars taken from ONE frame land clearly off the paint in the other frames (day bar/gap
  2.387 → 1.901): frames and cameras disagree on the ground position of the same bar by a sizeable fraction of a 0.5 m stripe.
  Fusion averages that disagreement and blurs neighbouring bars into bands. ⇒ **the stripe separation the PI asks for is
  limited by cross-frame / cross-camera registration, not by SAM3's masks or by the fusion rule** — the lever is
  H-SAM3-CALIB-1 (relative calibration and pose timing), which one clip could not identify.

Map r stays delivered; the softer-trimming videos (`reproj2_v65rs2_*.mp4`, Thor) are offered to the PI as a visual trade-off
(night better on both measures, day about 3 % more stripe cells on asphalt), not as a pass.

## 19. SAM3 inference time on Thor — approved measures

PI, 2026-09-14: *"We need to optimize the inference time of sam3 on thor, I need concret approved measures"*, and during the
work: *"Could quantization if the model help?"*

Scope: the per-frame SAM3 work of the delivered **front-camera map r** (v6 `classify()` on CAM_FW with 17 prompts, the v6s
stripe pass with 2 prompts, the front-only re-lift). Downstream stages (refine, consensus, renderer v5m fields, compose with the
map-r options) are unchanged in every arm. Thor: torch 2.13.0+cu130, SAM3 841 M parameters in fp32 (the C77 choice), TF32 already
enabled by the vendor builder, flash/mem-efficient SDPA enabled.

### 19.1 Where the time goes (MEASURED, `raw/speed/sam3_profile.json`, 6 warm front frames of both clips)

| stage, per front frame | s | note |
|---|---:|---|
| image encoder (`set_image`) | 0.392 | run **twice** today (classify, then again for the stripe prompts) |
| text encoder, 19 prompts | 0.158 | recomputed for every prompt of every frame, although prompts never change |
| grounding, 19 prompts | **2.182** | 0.115 s per prompt: fusion encoder 1.224 · decoder 0.531 · mask head 0.368 · prompt 0.067 |
| mask upsampling to 1920×1080 | 0.042 | for every query above the 0.25 processor threshold (142 per frame), not only the accepted ones |
| mask copy to host | 0.302 | same: all 142, as bytes |
| classify CPU logic | 0.263 | 0.06–0.56 depending on instance count |

### 19.2 Screen of levers (feasibility, instance level — NOT the approval; `sam3_levers_probe.json`, `sam3_precision_probe.json`)

Per prompt and query against the fp32 per-prompt path: max |Δp|, keep flips at the extractor thresholds, pixel disagreement of
masks kept by both. The yardstick is the **numerical floor**: 19 prompts batched in fp32 — mathematically the same computation,
different kernels.

| grounding variant (19 prompts) | s | max \|Δp\| | keep flips | mask px disagreement |
|---|---:|---:|---:|---:|
| fp32, per prompt (today) | 2.220 | 0 | 0/301 | 0 |
| text features cached | 2.156 | **0** | 0/158 | **0** |
| fp32 batched (floor) | 2.050 | 0.020 | 0/301 | 4.4e-5 |
| cudnn.benchmark | 2.156 | 0 | 0/158 | 0 (no speed-up) |
| **fp16 autocast, fusion encoder only** | **1.249** | 0.025 | **0/301** | 6.0e-5 |
| fp16 autocast, all stages | 1.127 | 0.066 | 1/301 | 2.5e-4 |
| **fp16 autocast, all stages, batched** | **0.841** | 0.234 | 1/301 | 2.6e-4 |
| bf16 autocast, fusion encoder only | 1.253 | 0.328 | 3/301 | 3.5e-4 |
| bf16 autocast, all stages, batched | 0.842 | 0.605 | 8/301 | 2.1e-3 |

Findings. (a) The fp32 fusion encoder is the single largest term, and half precision makes it **4.9× faster** (1.224 → 0.249 s);
in fp32 its attention cannot use the flash kernel. (b) **fp16 stays at the numerical floor, bf16 does not**: bf16's 7-bit
mantissa moves real detections (one "road marking" query 0.661 → 0.053 while another rises 0.079 → 0.68), fp16's only flip is
a "white stripe on road" query at 0.401 → 0.399 against a 0.4 threshold. (c) Batching in fp32 buys 5 % (the GPU is already
saturated per prompt); in fp16 it buys 25 %. (d) The mask head does not move with precision (0.31–0.37 s). (e) bf16 image backbone:
0.391 → 0.229 s but FPN features differ by up to 34 % and 16/158 instances flip — not carried.

### 19.3 Pre-registered approval (`code/speed/SPEC_sam3_speed.md`, banked on Thor 2026-09-14 15:30:16, md5 dfb9fc4c…, before any map-level result)

One driver (`code/speed/sam3map_front_fast.py`) writes the exact frame files refine reads. `MODE=ref` is today's pipeline
through the unmodified `classify()` and stock processor; `MODE=fast` runs **the same `classify` code object** with SAM3 results
injected, so its CPU logic cannot drift. Arms: `spd0` reference · `spdA` exact measures (text cache, one image encode, GPU-side
selection) · `spdF1` + fp16 fusion encoder · `spdF2` + fp16 all grounding stages, 19 prompts batched · `spdREG` spdA without the
`curb` prompt (deliberate regression, must FAIL) · later, same bars: `spdF2a` / `spdF3a` async CPU, `spdF3` + fp16 backbone.

Bars, both clips, vs `spd0`. EXACT: every frame file bit-identical and the composed map identical. NUMERICAL: N1 map agreement
≥ 0.990 over cells seen in either · N2 IoU ≥ 0.95 per class with ≥ 500 cells · N3 PI checks |Δ fragments/1000 m²| ≤ 2.0,
|Δ crosswalk coloured share| ≤ 0.02, |Δ curb recall 1.0 m| ≤ 0.01, |Δ edge precision 0.6 m| ≤ 0.01 · N4 per-frame raster
agreement ≥ 0.990. Controls: C-REPRO (`spd0` reproduces the historical extraction and the delivered map r) and C-REG.

### 19.4 Results, parts 1–2 (MEASURED, `raw/speed/spd_eval.json`; both clips, 96 front frames each; no eval tier applies — label-pipeline speed and output equivalence, not driving performance)

**Controls.** C-REPRO **PASS** on both clips: `spd0`'s 192 raw frame files are bit-identical to the historical extraction and its
composed map equals the delivered map r cell for cell — today's code reproduces the delivered map, so `spd0` IS map r. C-REG
**PASS**: dropping one prompt (`curb`) fails every numerical bar on both clips (N1 0.989 / 0.985, edge IoU 0.46 / 0.32, line IoU
0.55 / 0.54, fragments −2.9 / −14.4, N4 0.966 / 0.966) — the bars see a real quality loss.

| arm | s/frame night | s/frame day | vs `spd0` | verdict | the binding numbers |
|---|---:|---:|---:|---|---|
| `spd0` today | 3.417 | 3.601 | 1.00× | reference | — |
| `spdA` exact measures | 2.650 | 2.832 | 1.29× / 1.27× | **EXACT — APPROVED** | 192/192 frames bit-identical, both maps identical |
| `spdF1` + fp16 fusion encoder | 1.690 | 1.868 | 2.02× / 1.93× | **NUMERICAL-PASS — APPROVED** | N1 0.99986 / 0.99990 · lowest IoU 0.984 / 0.988 (edges) · N3 Δ ≤ 0.39 fragments, 0.003 edge precision · N4 0.99975 / 0.99987 · LiDAR \|Δ\| ≤ 0.0017 |
| `spdF2` + fp16 all grounding stages, batched | 1.286 | 1.470 | 2.66× / 2.45× | **FAIL (N2)** | edge IoU **0.943 / 0.931** < 0.95; N1 0.99933 / 0.99893, N3, N4 pass |
| `spdF3` = `spdF2` + fp16 backbone | 1.230* | 1.434* | — | **FAIL (N2)** | edge IoU **0.926 / 0.931** |
| `spdF2a` = `spdF2` + async CPU | 1.189* | 1.292* | — | **EXACT vs `spdF2`** (async approved; inherits F2's FAIL) | 192/192 frames and both maps identical to `spdF2` |
| `spdF3a` = `spdF3` + async CPU | 1.056* | 1.093* | — | **EXACT vs `spdF3`** (inherits F3's FAIL) | 192/192 frames and both maps identical to `spdF3` |

\* extraction overlapped part 1's downstream stages on the CPU; upper bounds, re-timed cleanly in part 3.

Reading. Everything that failed failed on ONE class: map **edges** (curb / boundary, ~3.4 k cells per clip), the class built from
mask boundaries of road, non-drivable and curb masks. Surfaces, lines, crosswalks and every PI check stayed inside their bars.
The pre-registered goal (≥ 2.0× with every measure approved) is **not met by part 2**: `spdF1` is the fastest approved arm and
reaches 2.02× by night but 1.93× by day.

### 19.5 Probes of the remaining levers (MEASURED, feasibility; `sam3_stage_probe.json`, `sam3_fp8_probe.json`)

* **Mask head** at fp16, 19 prompts batched: 0.197 s of the 0.749 s grounding is the pixel decoder (3×3 convs + GroupNorm up to
  288×288, prompt-dependent because the encoder output enters it); instance head 0.029, mask predictor 0.007, the unused semantic
  head 0.013. Nothing here moves with precision.
* **Image backbone in fp16:** 0.34 → 0.21 s/frame; FPN features within 1.3–5.4 %; 0/264 keep flips; mask disagreement ≤ 5e-4.
  Instance-level clean, but in `spdF3` it deepened the edge failure (0.943 → 0.926 night).
* **torch.compile** (inductor, static shapes, fp16, batched): 52.7 s to compile, then **slower** — 0.822 s vs 0.752 s eager, 0.916 s
  on a second frame. Not a lever on Thor.
* **Quantization — the PI's question — FP8 on the image backbone** (every ViT linear as float8_e4m3fn matmul through
  `torch._scaled_mm`, nothing installed): per-tensor scales **0.320 s vs fp16 0.206 s (slower)** with **37/264 kept instances
  flipped**, FPN features off by 56 %, mask disagreement 1.6e-2; row-wise scales 0.468 s (slower than fp32) with 44/264 flips.
  SAM3 is precision-sensitive (bf16 already moved detections that fp16 kept) and FP8's grid is coarser still: on this model and
  hardware quantization costs accuracy and buys no speed. Weight-only INT8/INT4 targets memory, and SAM3 peaks at 4.5–12.5 GB of
  Thor's 128 GB. **Not a lever; not carried.**

### 19.6 Why the fp16 decoder fails, and why no fp32 island rescues it (MEASURED, `sam3_decoder_island_probe.json`, 6 frames, 291 kept instances)

All variants: fusion encoder fp16, mask head fp32, 19 prompts batched; against the fp32 per-prompt reference.

| decoder | grounding s | decoder s | max \|Δp\| | flips | mask px disagreement |
|---|---:|---:|---:|---:|---:|
| all fp32 incl. encoder (floor) | 1.926 | 0.382 | 0.020 | 0/291 | 4.4e-5 |
| **fp32** (encoder fp16 = `spdF4a`) | **0.938** | 0.380 | 0.018 | 0/291 | **5.9e-5** |
| fp16 (as in the failed `spdF2`) | 0.741 | 0.183 | 0.063 | 1/291 | 2.4e-4 |
| fp16 + image cross-attention fp32 | 0.816 | 0.257 | 0.061 | 1/291 | 2.5e-4 |
| fp16 + box-RPB bias fp32 | 0.918 | 0.361 | 0.052 | 0/291 | 1.5e-4 |
| fp16 + box refinement fp32 | 0.743 | 0.183 | 0.286 | 1/291 | 2.4e-4 |
| fp16 + self / text attention fp32 | 0.754 | 0.184 | 0.090 | 1/291 | 2.7e-4 |
| fp16 + cross-attention AND RPB bias fp32 | 0.949 | 0.385 | 0.063 | 0/291 | 6.5e-5 |

The sensitive computation is the **box relative-position bias inside the image cross-attention**: only fp32 for both restores fp32
masks, and it then costs exactly what the fp32 decoder costs (0.385 vs 0.380 s). The RPB bias is a 19 × 8 × 201 × 5184 tensor per
layer — at fp16 its precision moves mask boundaries, which is what the edge class reads. **The decoder stays fp32; lever closed.**

### 19.7 Results, part 3 — decoder back to fp32 (MEASURED, `raw/speed/spd_eval.json`, `spd_eval_part3.txt`; timings clean: GPU otherwise idle)

Added after `spdF2` / `spdF3` failed, on the diagnosis of §19.2 (the extra mask deviation appears when the decoder joins fp16);
bars unchanged, as the SPEC commits for any further arm.

| arm | s/frame night | s/frame day | vs `spd0` | verdict | the binding numbers |
|---|---:|---:|---:|---|---|
| `spdF1a` = `spdF1` + async CPU | 1.646 | 1.548 | 2.08× / 2.33× | **EXACT vs `spdF1` — APPROVED** | 192/192 frames and both maps identical to `spdF1` |
| **`spdF4a`** fp16 fusion encoder · fp32 decoder + mask head · 19 prompts batched · async | **1.306** | **1.342** | **2.62× / 2.68×** | **NUMERICAL-PASS — APPROVED** | N1 0.99986 / 0.99967 · edge IoU **0.989 / 0.981**, all other classes ≥ 0.9987 · N3 \|Δ\| ≤ 0.002 · N4 0.99986 / 0.99978 · LiDAR \|Δ\| ≤ 0.001 |
| `spdF5a` = `spdF4a` + fp16 image backbone | 1.177 | 1.213 | 2.90× / 2.97× | **NUMERICAL-PASS — APPROVED, at the bar** | N1 0.99950 / 0.99926 · edge IoU **0.953 / 0.955** (bar 0.95) · line IoU 0.998 / 0.985 · N3 \|Δ\| ≤ 0.58 fragments, 0.004 edge precision · N4 0.99917 / 0.99958 |

Async CPU is exact on a third configuration. The pre-registered goal (≥ 2.0× with every measure approved) is **met** by `spdF1a`,
`spdF4a` and `spdF5a` on both clips; the stretch (≤ 1.2 s/frame) by `spdF5a` at night only.

**Recommendation for GT production: `spdF4a`.** `spdF5a` passes every committed bar and is approved by the SPEC's own rule, but its
edge class — the class the PI's reviews were about (kerb separation) — clears the bar by 0.003–0.005, where `spdF4a` clears it by
0.031–0.039. The backbone buys 0.13 s per frame (10 %); on two clips that margin is too thin to bet the corpus on without a third
clip. `spdF5a` stays available if Thor time becomes the binding constraint.

### 19.8 Pipeline-level measures (MEASURED, `raw/speed/front_stream_spdS.json`, `stream_equal_spdS.txt`)

`code/speed/sam3map_front_stream.py`: (1) SAM3 is built **once per process** — refine's ego-mask SAM3 calls (8 frames per clip) run on
the loaded model in fp32 with the fast path's fp16 wrapper suspended, where today every clip pays a second model build inside
refine; (2) refine (after its ego masks, `code/speed/refine_with_ego.py` = the unmodified refine with only its ego step fed),
consensus, renderer and compose run **unmodified as background processes** while the GPU extracts the next clip.

**EXACT** against the chain arm with the same per-frame flags (`spdF5a`), both clips: raw frames 96/96, refine output incl. ego
masks 96/96, consensus 96/96, composed maps identical, refine summaries equal.

| per 96-frame clip, `spdF5a` flags | every stage its own process | streaming |
|---|---:|---:|
| model build | 10.9 s (extractor) + ~11 s (refine) | 10.9 s once per process |
| SAM3 extraction | 115 s | 115–117 s (the second clip beside the first clip's CPU stages: 1.220 vs 1.213 s/frame — no measurable contention) |
| ego masks | inside refine | 5.2–5.4 s |
| refine / consensus / renderer / compose | 40 / 29 / 18 / 1 s, on the critical path | 16.5 / 30–32 / 19–21 / 0.6 s, in the background |
| **wall-clock per clip in steady state** | **≈ 214 s** | **≈ 120 s** (the CPU stages, 70 s, keep up with a 120 s GPU clip) |

### 19.9 The approved measures, and what they do to the augmentation estimate

| # | measure | class | approval | effect (MEASURED on Thor) |
|---|---|---|---|---|
| 1 | text features cached per prompt | exact | bit-identical, 192 frames + maps | part of 3.51 → 2.74 s/frame |
| 2 | one image encode per frame (classify + stripe prompts) | exact | same | 〃 |
| 3 | prompt threshold on the GPU before mask upsampling and host copy | exact | same | 〃; peak GPU memory 6.1 → 4.5 GB |
| 4 | CPU post-processing async (2 threads) | exact | bit-identical on 3 configurations | −0.04 to −0.34 s/frame |
| 5 | fp16 autocast on the fusion encoder | numerical | N1–N4 pass, both clips, edge IoU ≥ 0.981 | encoder 1.22 → 0.25 s/frame |
| 6 | the 19 prompts in one grounding forward (decoder, mask head fp32) | numerical | passes together with 5 (`spdF4a`) | −0.21 to −0.34 s/frame |
| 7 | one model build per process; refine's ego masks on the loaded model | exact | every intermediate bit-identical | −11 s per clip |
| 8 | refine / consensus / renderer / compose in the background | exact | same | ~70 s per clip moved off the GPU's critical path |
| 9 | fp16 image backbone (optional) | numerical | passes AT the bar (edge IoU 0.953 / 0.955) | −0.13 s/frame |
| ✗ | fp16 decoder / all stages · bf16 · FP8 quantization · torch.compile · cudnn.benchmark | — | FAIL or no speed-up | §19.4–19.6 |

**Per front frame: 3.417 / 3.601 s → 1.306 / 1.342 s with measures 1–6 (2.62× / 2.68×), 1.177 / 1.213 s with 9 (2.90× / 2.97×).**

Augmentation estimate, same assumptions as the estimate given to the PI the same day (20 s clips, 100 SAM3 frames each, fetch +
decode ≈ 1 min per clip — that one ESTIMATED, not measured; Thor's GPU dedicated), recomputed for the old pipeline with today's
measured numbers so both columns use one formula:

| scope | clips | today's pipeline (3.51 s/frame, stages in series: 514 s/clip) | approved, measures 1–8 (198 s/clip) | + backbone fp16 (185 s/clip) |
|---|---:|---:|---:|---:|
| front videos already on Thor | 160 | 22.8 h | **8.8 h** | 8.2 h |
| BEV-head clips (134 EVAL + 181 train) | 315 | 45 h (1.9 days) | **17.3 h** | 16.2 h |
| full v7 training corpus | 4,719 | 28 days | **10.8 days** | 10.1 days |

The fetch-and-decode minute is now a third of each clip; prefetching the next clip in parallel (not yet measured) would bring the
315 clips to about 12 h.

### 19.10 The last big lever — fewer prompts — FAILS (MEASURED; addenda pre-registered 20:17:21 and 20:39:51, before their results)

After precision, every grounding stage scales with the number of prompts. A screen over 192 frames (`prompt_redundancy.json`) found
five prompts whose accepted pixels are ≤ 2.1 % unique within their family (asphalt road surface 0.5 %, pedestrian crossing 0.8 %,
crosswalk stripe and zebra crossing 1.6 %, lane marking 2.1 %); `stop line` and `diagonal stripes on road` returned zero instances on
both clips and were excluded up front — the clips contain no stop line and no hatched area, which is not redundancy. Arms on
`spdF4a` flags, same bars:

| arm | dropped | s/frame | verdict | why |
|---|---|---:|---|---|
| `spdP3a` | asphalt road surface, zebra crossing, pedestrian crossing | 1.164 / 1.200 | **FAIL** | edge IoU **0.681 / 0.840**; night PI checks: fragments −4.03, curb recall +0.013, edge precision +0.025 |
| `spdP5a` | P3 + lane marking, crosswalk stripe | 1.062 / 1.088 | **FAIL** | night: line IoU 0.858, crosswalk 0.776, edge 0.679, crosswalk coloured share +0.098 |
| `spdP2a` (chosen after P3 failed, addendum 2) | zebra crossing, pedestrian crossing — both road prompts kept | 1.235 / 1.248 | **passes on the two test clips** | N1 0.99984 / 0.99943 · crosswalk IoU 0.9993 / 0.9971 · edge IoU 0.986 / **0.954** · N3 \|Δ\| ≤ 0.77 fragments · N4 0.99984 / 0.9997 |

Pixel redundancy does not predict map redundancy: a prompt's few unique pixels sit where the map is decided — the road boundary the
edge class is built from, the painted bars the stripe pass keeps — and the multi-frame vote makes a consistent shift permanent.
`spdP2a` is, as committed in advance, **not approved**: it buys 0.07–0.09 s per frame (≈ 6 %), the two clips hold four crossings,
and its day edge class clears the bar by 0.004. It needs the PI's sign-off and a validation on more crossings first.

### 19.11 Where this stands

**Done — the committed goal is cleared.** Every measure in rows 1–8 of §19.9 is approved against the map the PI confirmed:
**3.42 / 3.60 s → 1.31 / 1.34 s per front frame (2.62× / 2.68×)**, and with the streaming pipeline ≈ 214 s → ≈ 120 s per clip on
the GPU's path. For the 315 BEV-head clips the augmentation estimate falls from ≈ 45 h to ≈ 17 h.

Levers left, each with what decides it:
* **fp16 image backbone as default** (`spdF5a`, −0.13 s/frame): approved at the bar — the PI's call, or a third clip first.
* **Dropping the two crosswalk synonyms** (`spdP2a`, −0.08 s/frame): passes on two clips — PI sign-off + more crossings.
* **Prefetching the next clip's video** while the GPU works (≈ −1 min/clip, ESTIMATED): exact by construction, but it can only be
  measured on corpus videos, whose download still needs the PI's permission (the native test clips have nothing to fetch).
* **Fewer SAM3 frames per clip** (2.5 instead of 5 per second — halves the SAM3 GPU time, ≈ −1.1 min/clip): changes the map by
  design — the PI's call, as in the first estimate.
* Closed on evidence: fp16 / bf16 decoder, FP8 quantization, torch.compile, cudnn.benchmark, fp32 batching alone, pruning prompts
  that feed the road boundary or the stripe pass.
