# Input-pipeline design review — literature stream, AXIS 1 (resolution/aspect) + AXIS 4 (patch/tokenisation/pooling)

Date: 2026-08-30. Branch `agent/arch-inf-20260803`.
Evidence classes: MEASURED (ours) / PUBLISHED (primary, banked) / PUBLISHED-SECONDARY (inadmissible)
/ INHERITED / ESTIMATED (my arithmetic) / HYPOTHESIS.

All 22 papers below were banked as primaries via `tools/kb_add.py --tag input-pipeline`;
`--verify` reports **197 entries, 0 orphans, 0 problems** (content-hashed, not presence-checked).

---

## 0. The unit that makes this apples-to-apples: px/degree

Comparing "256×640" against "1600×900" is meaningless without FOV. Every claim below uses
**horizontal angular resolution in px/deg** = width / HFOV.

OUR value (ESTIMATED, arithmetic on INHERITED constants): 640 px / 120° = **5.333 px/deg**.
Cross-check: the inherited `f_ref = 305.577 px/rad` × π/180 = **5.333 px/deg**. The two inherited
constants are mutually consistent — a useful independent check that the corpus geometry is as stated.

Vertical (ESTIMATED, assumes the cylindrical remap uses the same f for elevation, i.e. perspective
in y): VFOV = 2·atan(128/305.577) = **45.4°**, so 256/45.4 = **5.64 px/deg** vertically.

| stack | input W×H | HFOV | **px/deg** | class |
|---|---|---|---|---|
| DriveVLA-W0 (NAVSIM SOTA) | 256×144 | ~64.4° | **~4.0** | PUBLISHED + ESTIMATED FOV |
| **TanitAD (ours)** | **640×256** | **120°** | **5.33** | INHERITED + ESTIMATED |
| TransFuser | 704×160 | 132° | **5.33** | PUBLISHED |
| Hydra-MDP / DiffusionDrive | 1024×256 (3 views) | ~120–150° | ~7–8.5 | PUBLISHED + ESTIMATED FOV |
| DrivingGPT / GAIA-1 | 512×288 | ~64.4° | ~8.0 | PUBLISHED |
| VAD-Tiny | 640×360 | 64.4° | 9.9 | PUBLISHED |
| BEVDet / SparseDrive-S / Fast-BEV | 704×256 | 64.4° | 10.9 | PUBLISHED |
| LAV primary cam | 480×288 | 40° | 12.0 | PUBLISHED |
| VAD-Base | 1280×720 | 64.4° | 19.9 | PUBLISHED |
| UniAD | 1600×900 | 64.4° | 24.8 | PUBLISHED |

nuScenes front-camera HFOV = 2·atan(800/1266) = 64.4° (ESTIMATED from the published f≈1266 px).

**Headline: TransFuser's angular resolution is 5.33 px/deg — identical to ours to four significant
figures**, and the current NAVSIM state of the art runs *below* us at ~4.0 px/deg.

---

## AXIS 1 — RESOLUTION AND ASPECT

### Q1. What resolution do camera-only E2E / world-model driving stacks actually use?

All PUBLISHED (primary, banked). W×H unless noted.

**Planners / E2E:**
- **UniAD** — 1600×900, ResNet-101, BEV 200×200 over `point_cloud_range [-51.2, 51.2]` = **0.512 m/cell**
  (verified from `projects/configs/stage2_e2e/base_e2e.py`: `img_scale=(1600,900)`, `bev_h_=bev_w_=200`).
- **VAD** — VAD-Base 1280×720, VAD-Tiny 640×360; BEV 200×200 (Base) / 100×100 (Tiny); perception
  range 60 m × 30 m ⇒ **0.30 m longitudinal / 0.15 m lateral per cell**. Closed-loop eval at 640×320.
- **PARA-Drive** — BEV 200×200 at **0.5 m/cell**; the paper states this "strikes a good balance
  between model performance and efficiency".
- **SparseDrive** — S: 256×704 / ResNet-50; B: 512×1408 / ResNet-101. **No dense BEV at all**
  (900 detection anchors, 100 map polylines, 600+33 temporal instances).
- **TransFuser** — 3 cameras at 960×480 each, "cropped to 320×160 to remove radial distortion at the
  edges", composited to **704×160**, composite FOV **132°**. LiDAR BEV 256×256 at **0.125 m/cell**.
- **InterFuser** — cameras 800×600 @ 100° HFOV; front scaled to 256 then **224×224 centre crop**;
  side views 128×128; focus view 128×128 uncropped. Output density map **20×20**.
- **LAV** — primary cam 480×288 @ 40°, three side cams 256×288 @ 64°; LiDAR pillars 0.25 m → 0.5 m
  output grid over 80 m × 80 m.
- **Hydra-MDP / Hydra-MDP++** — "the front-view image is concatenated with the center-cropped
  front-left-view and front-right-view images, yielding an input resolution of **256×1024** by
  default". Hydra-MDP-B uses 512×2048.
- **DiffusionDrive** — "three cropped and downscaled forward-facing camera images, concatenated as a
  **1024×256** image", ResNet-34, **88.1 PDMS** on navtest.
- **DriveVLA-W0** — NAVSIM models process **256×144** images from a **single front camera**;
  **93.0 PDMS** (NAVSIM v1, AR expert best-of-6), 88.4 single-model, 86.1 EPDMS on v2.
  (Verified with two independent probes of the primary; W×H, i.e. 144 px tall.)
- **EMMA** — 8 cameras concatenated into one 768×768 image. ⚠ **PUBLISHED-SECONDARY only** — I could
  not find this in the primary (2410.23262); the paper states only "surround-view camera videos" and
  a ≤4-frame temporal limit. **Do not quote the 768×768.**
- **Data Scaling Laws for E2E AD** — 7 cameras, native 3840×2160, **trained at 512×960**.

**World models:**
- **GAIA-1** — image tokenizer trained at **288×512**, spatial downsample **D=16** ⇒ **18×32 = 576
  discrete tokens/frame**, vocab 8192, tokenizer 0.3B params.
- **GAIA-2** — **448×960** per camera, video tokenizer **32× spatial / 8× temporal**, 5 cameras,
  latent 3×14×30, **12,600 tokens** for a 48-frame 5-camera sequence; 8.4B latent world model.
- **Vista** — **576×1024**; two-stage training 320×576 → 576×1024. SVD/latent-diffusion, not discrete tokens.
- **GenAD (predictive model, OpenDV-2K)** — "input frames are resized to 256×448 for training in both stages"; SDXL latent space.
- **DrivingGPT** — "Both nuPlan and NAVSIM record camera images of 900 pixels height and 1600 pixels
  width. We resize the original image to **288×512**"; VQ-VAE with spatial downsample **8 or 16**
  ⇒ "either **2304 or 576** image tokens for each front camera image"; image vocab 16384.
- **DriveWorld** — voxel 16×200×200; input resolution not stated in the primary. Planning L2 0.69 m
  avg, collision 0.19 %.

**VERDICT Q1: CONSISTENT.** 640×256 sits inside the field's planner band, and our 5.33 px/deg is
mid-to-low but not an outlier — it equals TransFuser exactly and beats DriveVLA-W0's NAVSIM SOTA.
The nuScenes-native stacks (UniAD 24.8, VAD-Base 19.9 px/deg) are 3.7–4.7× finer, but they are
perception-first architectures carrying dense detection/mapping/occupancy heads that need the pixels.

Note on aspect of the answer that *is* behind: our 640 px buys 120° of coverage from ONE camera.
The 10.9 px/deg stacks (BEVDet et al.) get their resolution by covering only 64° per camera and then
running **six** cameras. Per-camera we are 2.06× coarser; per-degree-of-360°-coverage we are cheaper.

---

### Q2. Is there a PUBLISHED CURVE of driving score vs input resolution?

**For DETECTION: yes, and it is clean.** (Labelled as detection, not planning.)

**Fast-BEV Table VI, "Ablation study of image resolution"** (nuScenes val, PUBLISHED primary):

| image resolution (H×W) | mAP | NDS |
|---|---|---|
| 256×448 | 0.280 | 0.419 |
| **256×704 (their default)** | **0.321** | **0.451** |
| 464×800 | 0.342 | 0.466 |
| 544×960 | 0.345 | 0.472 |
| 704×1208 | 0.358 | 0.478 |
| 832×1440 | 0.368 | 0.491 |
| 928×1600 | 0.369 | **0.488** |

Two things this curve says, both load-bearing:
1. **It saturates.** 832×1440 → 928×1600 is **+0.001 mAP and −0.003 NDS** — the last 1.15× of linear
   resolution buys nothing, and NDS goes *down*.
2. **The first step is the expensive-to-skip one.** 256×448 → 256×704 (width only, +57 %) is
   **+0.041 mAP**, larger than 704×1208 → 832×1440 (+0.010). Angular coverage/width mattered more
   than the later height increases.

**BEVDet (PUBLISHED primary), verbatim:** "The resolution of input image has a large impact on the
accuracy. For example, BEVDet with 1408×512 input size has a +4.5 % mAP superiority on that with
704×256 input size." So a clean **2× linear ⇒ +4.5 mAP** on detection.
Config points: 704×256 → 31.2 mAP / 39.2 NDS (Tiny); 1056×384 → 33.3 / 41.0; 1600×640 → 39.3 / 47.2.

**Fast-BEV Table VII, voxel/BEV-grid resolution ablation** (image fixed at 384×1056) — this is the
one that cuts *against* "finer is better":

| voxel resolution | mAP | NDS |
|---|---|---|
| 200×200×4 | 0.345 | 0.472 |
| **200×200×6 (default)** | **0.352** | **0.476** |
| 200×200×12 | 0.350 | 0.474 |
| 300×300×6 | 0.347 | 0.471 |
| 400×400×6 | **0.337** | 0.467 |
| 400×400×12 | 0.345 | 0.476 |

**Doubling the BEV grid makes it WORSE** (0.352 → 0.337 mAP). Spatial addressing has an optimum, and
200×200 is at it. This is the single strongest published datapoint against "more spatial resolution
in the readout is automatically better".

**For PLANNING: no clean curve exists.** I probed for one five ways and each came back negative:
- **Hydra-MDP** — I reproduced Table 2 row-by-row. Row 2: ViT-L(DepthAnything) @256×1024 → 89.9.
  Row 3: V2-99 @512×2048 → 90.3. Row 4: ViT-L(DepthAnything) @256×1024 → 91.0. **No two rows differ
  only in resolution** — V2-99 appears once. But the *bound* is informative: the 512×2048 arm (90.3)
  is **beaten by a 256×1024 arm (91.0)**. Whatever the resolution effect is, it is smaller than a
  backbone/pretraining swap.
- **Hydra-MDP++** at fixed 256×1024: ResNet-34 → PDMS 86.6; V2-99 → 91.0. **+4.4 PDMS from the
  backbone alone, at 256 px height.**
- **VAD** — no isolating ablation exists. VAD-Tiny changes resolution, BEV queries *and* layer count
  simultaneously. (See the confounded delta below.)
- **Vista** — no resolution ablation; the 320×576 → 576×1024 progression is a training-efficiency
  schedule, not an experimental variable.
- **DriveVLA-W0** and **Data Scaling Laws for E2E AD** — neither ablates resolution. The latter
  fits only a *data* power law: **Y = 0.6833 · X^−0.188, r = −0.963** for normalised ADE vs
  demonstrations over 10k/50k/0.7M/2M/4M.

**The closest thing to a planning-vs-resolution number, and it is confounded** (VAD, PUBLISHED):

| arm | input | BEV | layers | L2 avg | collision avg | FPS |
|---|---|---|---|---|---|---|
| VAD-Base | 1280×720 | 200×200 | 6 | **0.72 m** | **0.22 %** | 4.5 |
| VAD-Tiny | 640×360 | 100×100 | 3 | **0.78 m** | **0.38 %** | 16.8 |
| UniAD | 1600×900 | 200×200 | — | 1.03 m | 0.31 % | 1.8 |

⚠ **This is the most important row in the whole review, and it is a four-metric-families point.**
Halving linear resolution + halving the BEV grid + halving depth costs **+8.3 % on L2** but
**+73 % on collision rate** (0.22 → 0.38). The displacement metric barely moves; the safety metric
nearly doubles. Every "coarse is free" result in the literature is measured on ADE-family metrics —
which our own EVAL doctrine already says are the metrics least able to see a decision error.

**VERDICT Q2: UNKNOWN for planning / CONSISTENT-with-caveat for detection.** No paper publishes
driving-score-vs-resolution with everything else held fixed. The detection curves say the effect is
real but saturating, and the two planning proxies say resolution is dominated by backbone choice
(Hydra-MDP: +4.4 PDMS from backbone at fixed 256 px) — while the one safety-metric observation
available says coarsening hurts collisions ~9× more than it hurts ADE.

---

### Q3. Small-object detectability at 256 px height

**Our geometry** (ESTIMATED — my arithmetic, paraxial `pixel_height ≈ f·h/d` with the INHERITED
f = 305.577 px):

| object | h | 10 m | 20 m | 30 m | 50 m | 100 m |
|---|---|---|---|---|---|---|
| pedestrian | 1.7 m | 52.0 px | 26.0 px | **17.3 px** | 10.4 px | 5.2 px |
| car (height) | 1.5 m | 45.8 px | 22.9 px | 15.3 px | 9.2 px | 4.6 px |
| traffic-light housing | 0.9 m | 27.5 px | 13.8 px | 9.2 px | 5.5 px | 2.8 px |
| **single lamp** | 0.25 m | 7.6 px | 3.8 px | **2.5 px** | **1.5 px** | 0.8 px |

**The published threshold** (PUBLISHED primary, Zhang et al., *How Far are We from Solving Pedestrian
Detection?*, Fig. 12 — log-average miss rate by pedestrian pixel height, Checkerboards on Caltech):

| pedestrian height | miss rate |
|---|---|
| > 80 px | ~10 % |
| 50–80 px | ~18 % |
| 30–50 px | **~50 %** |

and the authors' own diagnosis, which is the quotable part: **"the small number of pixels is the true
source of difficulty"** — not blur, not contrast.

⇒ **On our frames a pedestrian exceeds 50 px only inside ~10 m, and is already inside the ~50 %-miss
band (30–50 px) by ~20 m; at 30 m it is 17 px, below the worst band the study measured.**
A single traffic-light lamp is **2.5 px at 30 m and 1.5 px at 50 m** — sub-detectable by any
published account, and the *state* (which colour) is a colour-blob decision at 2 px.

**Distance collapse in a real camera-only AV detector** (PUBLISHED primary, CramNet Table 4, Waymo
val, vehicle, 3D AP @ IoU 0.7 LEVEL_1, input downsampled to 640×960):

| method | overall | 0–30 m | 30–50 m | 50 m–∞ |
|---|---|---|---|---|
| CaDDN | 5.03 | 14.54 | 1.47 | 0.10 |
| CramNet-C | 4.14 | 15.46 | 1.20 | 0.15 |

**~10× per distance band, ~145× from the near band to beyond 50 m** — at roughly *double* our
angular resolution. This is the field admitting that camera-only far-field is broken even at 640×960.

**Fair comparison — the field is in the same hole:**
- UniAD @1600×900 (f≈1266): pedestrian at 30 m = **71.7 px** → the ~10–18 % miss band.
- BEVDet/Fast-BEV/SparseDrive-S @704×256 (f_eff ≈ 557): 30 m = **31.6 px** → right at the 30–50 px
  / ~50 % boundary.
- **Ours: 17.3 px** → ~1.8× worse than the most widely used BEV config, ~4.1× worse than UniAD.

**VERDICT Q3: BEHIND — and this is the one axis where the number is genuinely bad.** Not
catastrophically off the field (BEVDet's ubiquitous 704×256 is only 1.8× better and is itself in the
hard band), but our far-field and traffic-light budget is *structurally* absent, not merely degraded.
Two consequences worth stating plainly:
1. Any traffic-light-state capability is **unreachable at this geometry** — 2.5 px at 30 m is not a
   training problem. This is a corpus/geometry fact, and it compounds with the already-settled fact
   that PhysicalAI-AV carries **no traffic-light feature at all** (CLAUDE.md, 5 independent probes).
   So nothing is lost *today*; it is a ceiling on what this pipeline could ever be asked to do.
2. It argues for asymmetric spending: our deficit is concentrated in **angular resolution on small
   distant objects**, which is bought by width/focal length, not by taller frames.

---

### Q4. Aspect ratio — is 2.5:1 studied? Is vertical the binding constraint?

| stack | W×H | aspect |
|---|---|---|
| nuScenes native / UniAD / Vista / GAIA-1 / DriveVLA-W0 | — | **1.78:1 (16:9)** |
| GAIA-2 | 960×448 | 2.14:1 |
| **TanitAD (ours)** | **640×256** | **2.50:1** |
| BEVDet / Fast-BEV / SparseDrive-S | 704×256 | 2.75:1 |
| Hydra-MDP / DiffusionDrive | 1024×256 | 4.00:1 |
| TransFuser | 704×160 | 4.40:1 |

Clear pattern, PUBLISHED: **single-sensor and generative stacks keep the sensor's native 16:9;
every stack that feeds a *planner* goes wide-thin, 2.75:1 to 4.4:1.** Our 2.5:1 is at the narrow
end of the planner band. Nobody studies aspect ratio as a variable — but the field's *revealed
preference* is unambiguous.

**Direct evidence that vertical extent is deliberately sacrificed** (PUBLISHED primary, BEVDet's
image-view augmentation): the crop is random horizontally but **fixed vertically to the bottom of
the frame** — `(y1,y2) = (max(0, s·900 − H_in), y1 + H_in)`. At the standard s ≈ 0.44 that is rows
140–396 of a 396-row image: **the top 140 rows (35 %) are discarded, every sample, by construction.**
The field is explicitly saying the sky is not worth pixels.

⚠ **The consequence for us cuts the other way from how it looks.** Because BEVDet throws away 35 %
of the vertical FOV, its 256 rows cover only ~25° of elevation ⇒ **~10.2 px/deg vertically**. Our
256 rows are spread over ~45.4° ⇒ **5.64 px/deg**. So at the *same row count* we are **~1.8× coarser
vertically** — the same 1.8–2.06× deficit as horizontally. Our height is not "as good as BEVDet's"
just because the number 256 matches; it is spread over nearly twice the elevation.

**VERDICT Q4: CONSISTENT on aspect, WEAK-BY-DESIGN on vertical sampling.** 2.5:1 is normal for a
planner. But we are paying for elevation coverage we probably do not use — ~45° of vertical FOV on a
driving scene means a large fraction of rows are sky and near-field road surface. **The cheapest
resolution win available to us is not more pixels, it is cropping elevation** (see Option D below),
which raises px/deg vertically at *zero* token cost.

---

## AXIS 4 — PATCH SIZE / TOKENISATION / SPATIAL POOLING

### Q5. Patch sizes in driving world models and driving VLMs

All PUBLISHED. Note that driving world models overwhelmingly use a **VQ/VAE spatial downsample
factor**, not a ViT patch — but the quantity is the same thing: pixels per token.

| model | pixels/token (linear) | tokens/frame | resolution |
|---|---|---|---|
| Vista, GenAD (SDXL/SVD latent) | 8 | — (continuous latent) | 576×1024 / 256×448 |
| **DrivingGPT (alt config)** | **8** | **2304** | 288×512 |
| **GAIA-1** | **16** | **576** (18×32) | 288×512 |
| **DrivingGPT (main)** | **16** | **576** | 288×512 |
| **TanitAD (ours)** | **16** | **640** (16×40) | 256×640 |
| GAIA-2 | 32 | 3×14×30 latent | 448×960 |
| Triplane tokenizer (DINOv2-small / VQGAN baselines) | 14 / — | 160 per image | — |

**VERDICT Q5: CONSISTENT — patch/downsample 16 is the modal choice in driving world models, and our
640 tokens/frame is within 11 % of GAIA-1's and DrivingGPT's 576.** We are not coarse here. If
anything we are slightly *finer* than GAIA-2 (32×).

No driving paper ablates ViT patch size per se. The two AV-adjacent token-count ablations are Q6.

### Q6. Measured patch-size / token-count ablations

**ViT (PUBLISHED primary, Table 6, JFT-300M pretrain, 7 epochs):**

| model | ImageNet top-1 | exaFLOPs |
|---|---|---|
| ViT-B/32 | 80.73 % | 55 |
| ViT-B/16 | **84.15 %** | 224 |
| ViT-L/32 | 84.37 % | 196 |
| ViT-L/16 | **86.30 %** | 783 |

⇒ halving the patch: **+3.42 pp (B) / +1.93 pp (L) for ~4× FLOPs**. And the paper's own statement:
"the Transformer's sequence length is inversely proportional to the square of the patch size".
Note the diminishing return with scale — the bigger model gains *less* from finer patches.

**FlexiViT (PUBLISHED primary, banked):** trained at 240² so patch sizes tile exactly; token counts
25 (p=48) → 900 (p=8). Reference points: **ViT-B/8 = 85.6 % @ 156 GFLOPs vs ViT-B/32 = 79.1 % @ 8.6
GFLOPs** — **+6.5 pp for 18× the compute**. The paper's core claim is that randomising patch size at
train time yields one weight set that matches fixed-patch ViTs across /48…/8, i.e. **patch size is a
deployment-time knob, not an architectural commitment** — directly relevant if we ever want to trade
tokens for latency on Thor without retraining.
⚠ I could not extract the full per-patch numeric table (Appendix F) from either the HTML or the
banked PDF (no poppler on this box, so `Read` cannot rasterise it). The two anchor points above are
primary and quotable; **the full curve is banked but unread**.

**AV-specific #1 — Triplane tokenization for E2E driving (PUBLISHED primary, Table 2, 1B AR backbone):**

| tokenizer | tokens/image | minADE₆ @1s | @3s | @5s |
|---|---|---|---|---|
| VQGAN | 160 | 0.08 | 0.33 | 0.74 |
| DINOv2-small | 160 | 0.08 | 0.32 | 0.69 |
| triplane (8-8-8) | **45** | 0.08 | 0.32 | 0.72 |
| triplane (4-6-6) | 104 | 0.08 | **0.31** | **0.67** |

**A 3.6× token reduction (160 → 45) costs +0.03 m minADE at 5 s and nothing at 1 s and 3 s.**
Within the triplane family, 45 → 104 tokens (2.3×) buys 0.72 → 0.67 (−7 %).

**AV-specific #2 — DrivingGPT (PUBLISHED primary):** VQ-VAE downsample 8 vs 16 ⇒ 2304 vs 576 tokens
per front image; they **chose 16 (576 tokens) for the main experiments**, i.e. a 4× token saving was
judged worth taking.

**VERDICT Q6: CONSISTENT.** Measured patch ablations exist and are unambiguous — finer patches help,
sublinearly, at ~4× compute per halving, with the gain *shrinking as the model grows*. At our scale
(336.5M) the ViT-L datapoint (+1.9 pp for 4×) is the better predictor than the ViT-B one (+3.4 pp).
The AV-specific evidence is that aggressive token reduction is nearly free **on open-loop ADE**.

### Q7. Spatial pooling before the planner head — where the field actually sits

This is where I expected us to be embarrassed and we are not. The field **spans three orders of
magnitude**, and the coarse end is occupied by strong methods.

| stack | what the planner head actually receives | class |
|---|---|---|
| **TransFuser** | **a single 512-d vector** — "These feature maps are reduced to a dimension of 512 by average pooling"; spatial size immediately before pooling is **22×5** (image) and **8×8** (LiDAR BEV) | PUBLISHED |
| **TanitAD (ours)** | **4×8 = 32 tokens** | INHERITED |
| Triplane E2E | 45–104 tokens/image | PUBLISHED |
| InterFuser | **no pooling** — 10 waypoint queries + 400 density-map queries + 1 rule query cross-attend the multi-view features | PUBLISHED |
| VAD | ego queries attending agent/map queries (sparse); **the planner never touches the 200×200 BEV directly** | PUBLISHED |
| UniAD / PARA-Drive | 200×200 BEV @ 0.5 m/cell, attended by task queries | PUBLISHED |

**The angular comparison the brief asked for** (ESTIMATED, my arithmetic):
a 0.5 m BEV cell subtends 2·atan(0.25/30) = **0.955° at 30 m** and 0.573° at 50 m.

| readout | angular cell | vs ours @30 m |
|---|---|---|
| ours, 8 azimuth bins over 120° | **15.0°** | — |
| ours, if 4 bins (the disputed register row) | 30.0° | — |
| InterFuser 20×20 over 20 m × 20 m (1 m/cell) | 2.86° @20 m | we are 5.2× coarser |
| UniAD / PARA-Drive 0.5 m/cell | 0.955° @30 m | **we are 15.7× coarser** (26.2× @50 m) |
| VAD 0.15 m lateral cell | 0.286° @30 m | **we are 52× coarser** |
| TransFuser LiDAR BEV 0.125 m | 0.239° @30 m | 63× — **but its planner sees 1×1** |

⇒ **We are 1–2 orders of magnitude coarser than the BEV-grid lineage, and ~1 order of magnitude
*finer* than TransFuser's planner readout.** Both statements are true and the second one is the one
that matters, because TransFuser is the architecture that DiffusionDrive (88.1 PDMS) and the
Hydra-MDP line (91.0 PDMS) are built on. A 4×8 readout is not disqualifying.

⚠ **But note what makes TransFuser's 1×1 bottleneck survivable, because it is a design lesson we may
not be paying:** its pooled 512-d vector is trained under **dense auxiliary supervision** — 2D depth,
2D semantics, BEV semantics and 3D detection heads all hang off the pre-pooling feature maps. The
bottleneck is narrow but the *encoder* is forced to be spatially competent. Likewise VAD's ego query
is narrow but reads from *object and map queries* that were supervised as objects and maps. **In both
cases the narrow readout sits downstream of an explicitly spatially-supervised representation.**
Whether our 4×8 readout has that property is a question for the architecture stream, and it is the
right question — not "is 32 tokens too few".

**VERDICT Q7: CONSISTENT with the planner lineage, BEHIND the BEV lineage — and the axis is
mis-framed.** The field's variance on readout width is enormous and uncorrelated with score
(TransFuser's 1 token → DiffusionDrive 88.1 PDMS; UniAD's 200×200 → L2 1.03 m, worse than VAD-Tiny's
100×100 at 0.78 m). What correlates is **whether the representation feeding the readout was forced
to be spatial by auxiliary tasks.**

### Q8. Is there literature on the information bottleneck of pooling before planning?

**No paper measures this directly.** I probed for it four ways (VAD ablations, Fast-BEV, the triplane
paper, PARA-Drive's connectivity ablation) and the closest measurements are:

1. **Fast-BEV Table VII** — finer BEV grid is **worse**: 200×200×6 = 0.352 mAP, 400×400×6 = 0.337.
   There is an optimum and it is not "as fine as possible". (Detection.)
2. **Triplane** — 160 → 45 tokens costs +0.03 m minADE@5s, ~0 at 1–3 s. (Open-loop planning.)
3. **PARA-Drive** — 200×200 @ 0.5 m "strikes a good balance"; their ablations are about *module
   connectivity*, not spatial granularity, and they find removing edges can **improve** robustness.
4. **VAD-Tiny vs VAD-Base** — the confounded −8 % L2 / +73 % collision result above.

⚠ **The critical methodological gap, and it is ours to exploit rather than to inherit:** results 1–3
are all measured on **detection mAP or open-loop ADE/minADE**. Result 4 is the only one that reports
a *safety* metric, and it is the only one where coarsening looks expensive — by a factor of ~9
relative to the displacement metric. **The field's evidence that coarse spatial readout is free is
generated almost entirely by metrics that structurally cannot see a decision error**, which is
exactly the failure mode CLAUDE.md's four-metric-families rule exists to prevent, and exactly the
tier confusion EVAL_DOCTRINE forbids (all of 1–3 are T0-equivalent).

**VERDICT Q8: UNKNOWN — genuinely, and it is a publishable gap.** Nobody has published
readout-granularity vs collision/tactical-decision quality with everything else fixed. That is a
cheap discriminating experiment on the v7-tiny ladder (~19M): sweep the readout 4×8 / 8×16 / 16×40
with the encoder frozen and everything else held, and score the **tactical family**, not ADE.
It is exactly the shape of experiment CLAUDE.md rule 5 asks for, and the literature says the answer
is not known.

---

## What it would cost US to change — my arithmetic

Standard transformer forward FLOPs per layer, N tokens, width d:
`24·N·d²` (QKVO + 4× MLP) `+ 4·N²·d` (scores + AV).
At our N = 640 and d = 1024 (ESTIMATED — d not supplied; the conclusion is stable for d ∈ [768, 1280]):
linear term 1.611e10, quadratic term 1.678e9 ⇒ **the quadratic attention term is only 9.4 % of
encoder cost.**

⚠ **This corrects the framing in the brief.** "2× linear resolution is 4× tokens is ~4× attention
FLOPs" understates it and "16× because attention is quadratic" overstates it. Because we are
strongly in the *linear* regime at 640 tokens, 4× tokens multiplies the linear term by 4 and the
(small) quadratic term by 16, netting **~5.1×**, not 4× and not 16×.

| option | tokens | encoder FLOP × | notes |
|---|---|---|---|
| **current** 256×640, p16 | 640 (16×40) | 1.00× | 5.33 px/deg |
| **A.** 512×1280, p16 | 2560 (32×80) | **5.13×** | 10.67 px/deg; needs a **cache rebuild at 4× size** |
| **B.** 256×640, p8 | 2560 (32×80) | **5.13×** | same cost as A, **no new data, no cache rebuild** |
| **C.** 256×640, p32 | 160 (8×20) | **0.23×** | 4.3× cheaper; a real option for the tiny ladder |
| **D.** crop elevation 256→176, p16 | 440 (11×40) | **0.69×** | **+45 % vertical px/deg for FREE**; mirrors BEVDet's fixed sky crop |
| **E.** 384×640, p16 (taller) | 960 (24×40) | **1.57×** | vertical only |
| **F.** 512×640, p16 | 1280 (32×40) | **2.19×** | vertical only |
| **G.** readout 4×8 → 8×16 | 32 → 128 | ~free in the encoder | 4× on the (small) planner heads only; **15° → 7.5°/bin** |

**Non-FLOP costs of Option A that Option B avoids entirely** (INHERITED from CLAUDE.md DE-C152,
so re-measure before acting): the v7 `--v2-cache` is ~34 MB/ep × 2403 ep ≈ **161 GB** with "~4×
headroom" against the pod quota. **4× pixels ⇒ ~644 GB, which does not fit**, and the rebuild is
dominated by PNG *encode* (70 % of build time; current full build 3.4–4.6 h ⇒ **~14–18 h at 4×**).
The codec is load-bearing, not a speed knob — `v2_dataset.py:325` refuses to sub-frame a lossy cache.

**Ranking, given the evidence:**
1. **Option D first (0.69×, negative cost).** The Q4 finding is that we spend 256 rows over ~45° of
   elevation while the field spends 256 rows over ~25°. Cropping elevation buys ~1.45× vertical
   angular resolution while *reducing* tokens by 31 %. Nothing else in this review is free.
2. **Option G next (~free).** The readout is the axis with the least published support in either
   direction (Q8 = UNKNOWN), it is cheap to sweep, and 15°→7.5°/bin is a real change in addressing.
   Do it as a pre-registered tiny-ladder sweep scored on the **tactical family**, not ADE.
3. **Option B over Option A if resolution is pursued at all.** Identical FLOPs, identical token
   count, no 644 GB cache, no 18 h rebuild, no parity risk from re-deriving `HORIZON`/stride against
   a new cache. The only thing A buys over B is genuine new sensor detail — which matters *only* for
   Q3's far-field/small-object deficit, and that deficit is ~1.8–4× and would need the full 2× to
   half-close.
4. **Option C for the tiny ladder** if experiment throughput is the binding constraint.

---

## Retraction risks in this document

- **EMMA's 768×768** is PUBLISHED-SECONDARY. Not in the primary. Do not quote it.
- **FlexiViT's full per-patch table** is banked but unread (no poppler on this box). Only the
  ViT-B/8 85.6 % @156 GFLOPs and ViT-B/32 79.1 % @8.6 GFLOPs anchors are quotable from it.
- **All px/deg figures for nuPlan/NAVSIM-derived stacks** (DriveVLA-W0, Hydra-MDP, DiffusionDrive,
  DrivingGPT) carry an ESTIMATED FOV. DrivingGPT states NAVSIM/nuPlan images are 1600×900, so I
  applied the nuScenes-equivalent 64.4°; if the nuPlan front camera is wider, those rows move.
- **Our vertical FOV (45.4°)** assumes the cylindrical remap is perspective in elevation with the
  same f. That is the standard construction but it is my assumption, not a measured fact —
  and CLAUDE.md's own optics trap says to **state the projection before using any camera formula**.
  ⇒ **This one number should be verified against the remap source before it decides anything**, and
  Option D depends on it.
- **The v2-cache sizes** are INHERITED from CLAUDE.md, and that same entry is a retraction about
  pricing the wrong artifact. Re-price against the actual consumer loader before acting on Option A.

## Banked primaries (all `--tag input-pipeline`, all content-hash verified)

2205.15997 TransFuser · 2212.10156 UniAD · 2303.12077 VAD · 2405.19620 SparseDrive ·
2406.06978 Hydra-MDP · 2503.12820 Hydra-MDP++ · 2411.15139 DiffusionDrive · 2510.12796 DriveVLA-W0 ·
2412.02689 Data Scaling Laws E2E AD · 2207.14024 InterFuser · 2203.11934 LAV ·
2309.17080 GAIA-1 · 2503.20523 GAIA-2 · 2405.17398 Vista · 2403.09630 GenAD · 2412.18607 DrivingGPT ·
2405.04390 DriveWorld · 2301.12511 Fast-BEV · 2112.11790 BEVDet · 2210.09267 CramNet ·
2010.11929 ViT · 2212.08013 FlexiViT · 1602.01237 Pedestrian-detection-by-pixel-height ·
2506.12251 Triplane tokenization.

`python tools/kb_add.py --verify` ⇒ **197 entries, 0 orphans, 0 problems.**
Tool note: `kb_add.py` raised a Traceback once (2405.19620) under concurrent writes and succeeded on
immediate retry — the known concurrent-write race the `--reindex-orphans` flag exists for.
