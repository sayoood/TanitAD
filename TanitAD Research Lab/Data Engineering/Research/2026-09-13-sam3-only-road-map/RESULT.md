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
