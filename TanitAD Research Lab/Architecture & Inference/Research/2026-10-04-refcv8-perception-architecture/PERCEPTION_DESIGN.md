# PERCEPTION_DESIGN — refcv8 map and box heads (WP-D, R8-6)

*Architecture & Inference, WP-D, 2026-10-04 (Berlin). Answers the PI, verbatim: "We should optimize architecture,
training workflow to dramatically improve the map and box heads (do we need an initialization of the heads? or other
tricks?)". Evidence: `RESULT.md` (training dynamics, PROBE-0, the geometry bound). Probes: `PREREG_DRAFT_WPD_PROBES.md`.
Evidence classes: MEASURED (ours, artifact named) · ANALYTIC · PUBLISHED [LIB:<arXiv id>] = banked in
`TanitAD Research Lab/Library/` and the cited number READ in the banked PDF · ESTIMATED · INFERRED · UNVERIFIED.
⛔ This is a design and a probe plan: no trainer or model file was edited (WP-B owns them).*

## 0. The answer in five lines

1. **The box heads were cut off mid-climb, not maxed out** — AP@2 m +0.050 [+0.041, +0.058] in the last 20k steps at
   a falling LR, after ~1.08 epochs (MEASURED, RESULT §2.1). The cheapest large lever is to **keep training them**
   from refcv7-50,400 with a re-warmed LR and EMA weights.
2. **The box decoder needs the modern set-prediction recipe** — it produces 1.97 (box3d) / 3.12 (planner's agent
   head) boxes per detected object, ranks a 1.22 m slot above a 0.67 m one, and its anchors never moved (MEASURED,
   RESULT §2.2–2.6): **hybrid one-to-many + contrastive denoising + quality-aware presence + per-layer anchor
   refinement**, all training-only or zero-parameter.
3. **The planner reads the weaker detector** (AP@2 m 0.131 vs 0.248): **one detector, box3d → planner**, zero-gated.
4. **The map is not under-trained; it is under-resolved** — beyond ~40 m the image gives < 1 stride-8 row per 20 m
   band (ANALYTIC), so **spend 10 cm only on 0–40 m, add a stride-4 near tap and deeper-stage semantics there, train
   lines with a placement-tolerant target, and measure 40–100 m with a range-adaptive metric.**
5. **"Do we need an initialization of the heads?"** — not of the biases (the focal prior washed out, MEASURED); **yes
   of the anchors and the memory positions** (the 300 anchors are still their random init; the memory has no camera
   geometry) — fixed by per-layer refinement and a zero-initialised ray embedding; and the most valuable
   initialisation is refcv7's own heads (§1).

**Biggest blockers (named):** the probes need Thor GPU slots (queued behind A7 → R1 → WP-RL) and ~4.5–7.8 GB of
Thor disk for P-BOX; LiDAR depth (B4c) needs a data build and a PI download decision; temporal fusion needs an
admissibility ruling for past ego-motion. Nothing here needs a PI decision before P-GRAD / P-BOX / P-MAP run.

## 1. "Do we need an initialization of the heads?" — the answer

**Yes for the box head's query ANCHORS and for its POSITIONAL information; no for the biases; and the most valuable
"initialisation" of all is refcv7's own trained heads, which were not finished training.**

| what is initialised | what refcv7 did | MEASURED outcome | verdict for refcv8 |
|---|---|---|---|
| presence bias (focal prior 0.01, logit −4.595) | RetinaNet / Deformable-DETR / DETR3D prior | bias **−4.529** at 50,400 (`raw/refpts_by_step.json`); undoing the prior overshoots conf_ratio to 88.9 (diagnostics §2.4) — the prior washed out and is not the defect | **keep**; nothing to re-initialise |
| map class-logit bias | default conv init | trained biases −0.39 … +0.10 (same file); class balance is carried by the sqrt-MF weights | **keep**; new map paths start **zero-initialised** (the `NearLiftSkip` pattern) |
| box3d query **anchors** (`learned_ref`, 300 × (x, y)) | seed-0 uniform over 60 m × ±16 m, "learned" | **never moved: median 0.031 m, max 0.18 m in 50,400 steps** (random-walk size, lr·√N ≈ 0.02 m); 91 / 300 at \|y\| > 12 m; PROBE-0 Q6 58 % of queries never produce a confident TP; 16.5 % of TPs sit at the tanh limit; reach is fine (median 9 anchors per object) | **re-think** (INFERRED mechanism): the anchor is redundant with the per-query offset the decoder learns, so its gradient averages to zero. Make anchors PER-FRAME and PER-LAYER (B2's per-layer refinement = P-BOX arm PB4, or the already-built heatmap selection HQS = PB4h) instead of hoping a static table learns |
| memory positions (image + BEV tokens) | a learned table, 2,144 × 256 = 548,864 params, no camera geometry | localisation is the late-training gain, AP@0.5 m is 0.030 and the error is 2.5–3.4× longer along range (RESULT §2.1, §2.5) | **add geometry**: a PETR-style ray embedding, zero-initialised on top of the trained table (B4) |
| the whole detection head | ImageNet trunk → joint training, ~1.08 epochs | AP@2 m still rising **+0.023 per 10k steps at LR → 0** (RESULT §1.2) | **warm-start from refcv7-50,400 and keep training the heads at a re-warmed LR** (B1). This is the single largest "initialisation" lever and it costs nothing |
| new refcv8 paths (one-to-many groups, DN, quality branch, s4 tap, FPN) | — | — | groups/queries = **copies of the trained table**; additive paths **zero-initialised** so step 0 reproduces refcv7 exactly (W3) |

What a **detection pre-training** of the trunk (R7 in refcv7's deferred list; DD3D-style depth pre-training) would add
is a separate, PI-level question (B4c): the evidence for depth pre-training is strong for monocular 3-D
detection (DD3D: −5.3 / −10.7 Car BEV AP on KITTI when removed [LIB:2108.06417]) but it needs a LiDAR depth build.

## 2. The ranked changes

Ranked by the size of the MEASURED effect they act on (Rule Zero §5), then by cost. "Warm start" = fits a run that
starts from refcv7-50,400 with every unchanged module loaded and every new path zero-gated, so step 0 == refcv7.

### 2.0 Summary table

| rank | change | the MEASURED failure it acts on | published evidence (read in the banked PDF) | expected effect (class) | params · s/step on Thor (ESTIMATED vs the MEASURED 10.03 s/step) | planner risk | warm start |
|---|---|---|---|---|---|---|---|
| **1** | **B1 keep training the detection heads** — warm start from 50,400, re-warmed LR, EMA eval weights | AP@2 m +0.050 [+0.041, +0.058] in the last 20k steps at a falling LR, still rising at LR 0 after 1.08 epochs (RESULT §2.1) | DETR 500 / Deformable 50 epochs [LIB:2005.12872, 2010.04159]; DN parity at 50 % epochs [LIB:2203.01305]; EMA needs less LR decay [LIB:2411.18704] | +0.03 … +0.07 AP@2 m over a 30k-step run if the late rate (+0.024 / 10k, 1,061 windows) holds — ESTIMATED, a range not a forecast (1.5× the fitted window) | 0 · ~0 (EMA +0.4 GB) | none new | **is** the warm start |
| **2** | **B2 set-prediction recipe for both slot heads**: hybrid one-to-many (Group-DETR groups) + contrastive denoising + quality-aware presence + per-layer anchor refinement | 1.97 (box3d) / 3.12 (agent) boxes per detected object; top score on the nearest slot only 34.5 % / 23.0 %; anchors frozen, 16.5 % of TPs at tanh saturation, 58 % of queries never a TP (§2.2–2.6); decoder refines its loss 4 % over 3 layers | H-DETR +1.7 AP, H-PETRv2 +1.52 mAP; Group DETR +1.4 mAP / +3.0 NDS (PETR); DN +1.9 AP; DINO CDN; VFL +0.9–2.0 AP; Stable-DINO +1.1–1.4 AP; iterative refinement +1.6 AP | duplicates toward ~1 per object WITHOUT NMS; AP@2 m at or above the MEASURED NMS level (0.339) while AP@1 m rises instead of falling (NMS costs −0.019 at 1 m); same-box ORACLE ceiling for the ranking part (MEASURED, RESULT §2.4): re-ranking each object's own slots by placement + NMS r 2 m takes AP@1 m 0.130 → **0.263** and AP@2 m → 0.349 | 0 at inference (groups / DN training-only; refinement reuses `AnchorPosEmbed`) · +0.3 … +1.0 s (K-fold decoder + Hungarian; K chosen by P-BOX) | box3d: none (not read by the planner); agent: presence semantics change → only through B3 | yes: extra groups = copies of the trained query table + anchors; DN reuses the decoder; re-fit gates |
| **3** | **B3 one detector for the planner**: feed box3d's slots to `AgentTokenEmbed` as an extra token set (value projection zero-init), retire the stride-32 agent head once the four families say neutral-or-better | the planner reads AP@2 m 0.131 / 3.12 dups per object when 0.248 / 1.97 exists (RESULT §0.5, §2.2) | — (an internal routing decision) | the planner's agent tokens inherit every box gain (rank 1, 2, 5) instead of a second, weaker detector learning in parallel | −3.9 M params eventually · −0.1 … −0.2 s | moderate (input shift) → zero-gated; decided by the four-family battery, not by AP | yes |
| **4** | **M1 map: spend the resolution where the image has it** — (a) stride-4 near lift over 0–40 m; (b) deeper-stage (layer3) fusion into the stride-8 lift input; (c) 10 cm decode on 0–40 m only, the 0.25 m grid beyond; (d) a line-aware target for lane / edge | 40–100 m decoded from ~1.3 stride-8 rows; recall@1 m 0.60 → 0.01 over range at flat precision; 0–20 m edge recall@1 m only 0.60 with 21.5 stride-8 rows available (RESULT §3) | Simple-BEV: resolution dominates, deeper stage upsampled into the 1/8 map [LIB:2206.07959]; MapTRv2 aux / one-to-many [LIB:2308.05736]; clDice / boundary loss [LIB:2003.07311, 1812.07032] | the P-MAP bars: edge IoU@0.2 m 0–20 m 0.235 → ≥ 0.33, 20–40 m 0.118 → ≥ 0.18; lane 20–40 m 0.308 → ≥ 0.38 — UNVERIFIED until P-MAP | +~0.1 M · (c) −0.5 … −1.2 s, (a) +0.1 … +0.3 s → net ~0 or faster | (a)(c)(d) none (map-only, after the shared encoder); (b) shifts the shared 0.25 m BEV → zero-init | yes |
| **5** | **B4 geometry + depth for the longitudinal error**: (a) PETR-style ray embedding on the box memory's image tokens; (b) each query samples the 0.5 m BEV at its (refined) anchor; (c) LiDAR depth as an auxiliary trunk target (data-gated); (d) temporal BEV (deferred) | the box error is 2.5–3.4× larger along range than across it; 0–20 m longitudinal error ≈ 7 image rows (learnable, not optics) (RESULT §2.5) | PETR 2D → 3D PE NDS 0.208 → 0.356 [LIB:2203.05625]; DETR3D [LIB:2110.06922]; BEVDepth depth loss +2.2 mAP [LIB:2206.10092]; MapTRv2 depth +5.1 mAP; DD3D −5.3 / −10.7 BEV AP without depth pre-training [LIB:2108.06417]; SOLOFusion mATE 0.743 → 0.655 [LIB:2210.02443]; ⚠️ DualPathOcc: hard depth targets −0.96 mIoU [LIB:2609.06370] | localisation: AP@0.5–1 m and the longitudinal error; magnitude UNVERIFIED for one monocular camera (PETR's gain is multi-view) | (a)(b) ~0.1–0.2 M · ~0; (c) ~0 on the GPU, a data build of ~342 MB/clip | (a)(b) box3d only; (c) the trunk (all consumers) → aux loss weight from P-GRAD | yes: (a)(b) zero-init additive; (c) a new head |

Below rank 5, adopted at no cost: **W1–W6** (§2.6). Dropped: the IGNORE-exemption change (eliminated by PROBE-0
§2.2), a presence-prior re-init, 10 cm targets beyond 40 m (§4).

### 2.1 B1 — keep training the heads (rank 1)

*Mechanism.* refcv7's cosine schedule took both groups to LR ≈ 0 after ~1.08 epochs while the box heads were still
improving at every threshold and band (RESULT §2.1). A warm start that re-warms the head LR (1e-4) and the trunk LR
(5e-5) over ~1k steps and decays over the refcv8 run continues that curve; an EMA of the weights (decay ~0.9998,
evaluated, never trained) supplies the noise reduction the final LR decay was supplying, without stopping learning.
*What would falsify it:* P-BOX arm PB0 (continue on a frozen memory) staying flat — then the late gain was the trunk's,
not the decoder's, and only a full run (not a head branch) captures it.

### 2.2 B2 — the set-prediction recipe (rank 2), one variable per P-BOX arm

1. **Hybrid one-to-many** — K query groups (groups 1..K−1 initialised as copies of the trained table and anchors,
   jittered), one-to-one Hungarian inside each group, group-masked self-attention; inference reads group 0. Gives each
   object K matched queries per step instead of 1 → K× the positive presence signal (refcv7 sees ~3 positives per
   window against 300 queries; RESULT §1.2 train census 48.7 matched per 16-window batch).
2. **Contrastive denoising (CDN)** — noised GT boxes as extra queries (positives) plus larger-noise "no object"
   negatives, attention-masked, reconstructed with the same losses. Stabilises the assignment refcv7 MEASURED as
   unstable (0.4–6.5 % of targets keep their slot on a fixed frame, SPEC_REFCV7 A14) and teaches "a second query near
   an object is NOT that object" — the duplicate mechanism. ~3 positives per window ⇒ only tens of extra queries.
3. **Quality-aware presence** — the matched slot's target becomes q = exp(−d² / 2·(1 m)²) of its BEV centre distance
   (varifocal form); our AP is centre-distance AP, so q is the metric's own notion of quality. Acts on §2.3: the
   head ranks a 1.22 m slot above a 0.67 m one. ⚠️ Changes what σ(presence) means: gates are re-fitted, and the
   planner sees it only through B3's zero-gated path.
4. **Per-layer anchor refinement** — layer l+1's anchor = detach(layer l's decoded centre), its position re-embedded
   through the existing `AnchorPosEmbed` (0 new parameters). The static anchors never moved (RESULT §0.4); refinement
   makes the anchor per-frame and per-layer, and pulls the 16.5 % of TPs off the flat end of their tanh.
   (The already-built heatmap query selection, HQS, is the alternative per-frame anchor source; it is the arm to run
   if refinement alone does not move AP@1 m.)

### 2.3 B3 — one detector for the planner (rank 3)

The planner's `AgentTokenEmbed` consumes `{box, cls_logits, presence_logit, yaw_vec, rates}`; `Box3DSlotDecoder.decode`
returns every one of those keys with the same meaning (`box3d_head.py` contract, pinned by a test). So the box3d slots
can enter the planner as a second token set whose value projection starts at zero (step 0 == refcv7), with the agent
head kept until the four-family battery (route, longitudinal incl. distance keeping, lateral, tactical) reads the
switch as neutral-or-better; then the agent head is removed. **Cheapest probe:** R1 caches 300 agent slots per window
(R1 DESIGN §3); caching box3d's 300 slots too costs ~14 KB per window and lets R1's head-only planner arm read
box3d slots instead — a planner-side answer in minutes.

### 2.4 M1 — the map (rank 4)

* **(a) stride-4 near lift, 0–40 m.** `layer1` (256 ch, 104 × 256 at 416 × 1024) is computed inside the existing
  stride-8 tap pass (stem → layer2) and is free; project-first to `d_up`, sample at the 10 cm cells of 0–40 m with
  the lift geometry, ADD on those rows exactly like `NearLiftSkip` (projection zero-init). Doubles the image rows
  behind 20–40 m (2.2 → 4.4 feature rows) and halves the lateral footprint (0.49 → 0.24 m per column).
* **(b) deeper-stage fusion.** `s8_input += up×2(1×1(s16_fused))`, the 1×1 zero-initialised: the map currently sees
  ONLY layer2 of the newest frame (`map_hires.trunk_tap`, config), i.e. low-level features; curbs (the `edge` class:
  no paint, recall@1 m 0.60 even at 0–20 m) need context. The s16 map is already computed (box memory reads it).
* **(c) 10 cm only where it is supported.** Decode 0.1 m on x ∈ [0, 40) m; beyond, a 1×1 class head on the 0.25 m
  encoder grid (supervised by the SAM3 codes pooled to 0.25 m). The 10 cm branch is bandwidth-shaped (429 GF and
  14.5 GB touched per sample, step-cost package §2b, INHERITED); cropping the 10 cm decoder to 40 % of its area is the
  saving that pays for (a). ⚠️ The planner crop is 60 m: 40–60 m is then supervised at 0.25 m instead of 0.1 m — a
  small shift in the shared encoder's gradient, measured by P-MAP's drivable non-inferiority bar.
* **(d) line-aware target, 0–20 m.** Soft lateral targets for lane / edge (the diagnostics' F3) as the first arm;
  soft-clDice or a boundary (distance-transform) term as the alternative. The diagnostics measured the head FINDS
  edges near range but places them a few cells off: 0–20 m edge precision 0.07 on the exact cell, 0.35 within 0.2 m,
  0.72 within 1 m (M-b); a target that does not punish a 1–2-cell offset as hard as a miss is the cheap version of
  tolerance-aware training.
* **M5, the metric** (adopted now, no training): 0–40 m keeps IoU_0 / IoU@0.2 m; 40–100 m is reported with a
  range-adaptive tolerance — lateral placement of longitudinal lines within max(0.2 m, 2 px × x / f) and longitudinal
  within one image row of depth (x² / (f·h)) — beside recall@1 m, following Far3Det's distance-adaptive thresholds
  [LIB:2211.13858]. The same applies to boxes beyond 40 m. **This is a change to R8-6's stated target**: "edge IoU
  0.003 at 80–100 m" is an image-resolution bound (RESULT §3), not a defect a head can remove.

### 2.5 B4 — geometry and depth (rank 5)

* **(a) ray embedding.** For each image token, the clip's own cylindrical ray direction (per-clip extrinsics through
  `LiftGeometryBank`, already delivered to the forward) → 2-layer MLP, last layer zero-init, ADDED to the learned
  `mem_pos`. A learned 2-D table cannot represent camera height 1.20–1.69 m and per-clip pitch; at 20–40 m one image
  row is 1.27 m of range.
* **(b) BEV point sampling.** Each query reads the 0.5 m planner BEV (`bev_feats`, 96 × 120 × 64) bilinearly at its
  current (refined) anchor (3 × 3 neighbourhood → linear → d_model, zero-init) — today it sees only 2 m pooled tokens.
* **(c) LiDAR depth auxiliary.** The corpus has `lidar_top_360fov` (MEASURED, 2026-09-11 package: 10 Hz, 199 spins,
  same µs clock, median 30.2 ms from the camera frame). Target: per frame, min-depth of the projected points per
  stride-8 cell (uint16 cm, ~13 KB/frame), Gaussian-softened; head: a 1×1 on s8 / s16, L1 on log-depth. **Blocked on a
  data build** (~342 MB of zip-member streaming per clip, MEASURED 2026-09-11) → Data FlyWheel + a PI download
  decision; validated on v7-tiny first (P-DEPTH) because the evidence is two-sided.
* **(d) temporal BEV** — deferred (P-TEMP); needs an explicit admissibility ruling for past ego-motion warping.

### 2.6 Training-workflow items (adopted, no probe needed beyond the run's own gates)

| id | item | why (evidence) |
|---|---|---|
| W1 | re-warm both LR groups over ~1k steps from 50,400, then decay over the refcv8 run; keep `--opt dd` (head 1e-4, trunk 5e-5) | RESULT §1.1–1.2; DETR's 10× lower backbone LR is already approximated by dd's 0.5× [LIB:2005.12872] |
| W2 | EMA of all weights (decay ~0.9998) evaluated at every in-run eval and saved with each checkpoint | [LIB:2411.18704]; the late box gains arrived as the LR annealed |
| W3 | every new path zero-initialised or a copy of a trained module, so step 0 reproduces refcv7 exactly (pinned by a test: refcv8 step-0 forward == refcv7-50,400 forward on 8 windows) | the warm-start contract of PLAN §0.1 |
| W4 | NO perception-first stage to protect the trunk (aux already carries 99.4 % of its gradient norm, RESULT §1.4); OPTIONAL: planner LR 0 for the first ~2k steps while new perception paths (B2, M1) settle — a PI choice, ~5.6 h of Thor | MEASURED gradient shares |
| W5 | loss budget: run P-GRAD before changing any perception weight; the box `rates` term is the largest per-layer box term (~26 %) and its targets carry id-switch jumps (D3) — WP-C's clip lands first | RESULT §1.2 |
| W6 | every eval reports AP at 0.5 / 1 / 2 / 4 m raw AND after NMS r 2 m, boxes per detected object, and the map per band with M5's range-adaptive metric beyond 40 m | RESULT §2.2–2.4, §3 |

## 3. The literature, mapped to OUR setting

Monocular **cylindrical 120°** front camera at 416 × 1024 (f_ref 488.92), parameter-free **multi-height bilinear lift**
(not LSS: no depth distribution exists to supervise), **10 cm raster** map to 100 m, **300-query set prediction**
boxes over a 60 m × ±16 m BEV box, one training seed, ~1 epoch. Every number below was read in the banked PDF.

| topic | paper [LIB key] | the published number (setting) | what it means HERE | verdict |
|---|---|---|---|---|
| depth supervision of the view transform | BEVDepth [LIB:2206.10092] | depth loss **+2.2 mAP** (28.2 → 30.4), camera-aware depth −0.41 mATE, total +4.0 mAP / +4.0 NDS (nuScenes val, R50, 256×704, LSS) | our lift has no depth bins to supervise; depth enters as an AUX loss on the trunk features (MapTRv2 / DD3D style) or as depth-aware lift weights (FB-BEV) | probe (P-DEPTH) → B4c, data-gated |
| depth-aware backward projection | FB-BEV [LIB:2308.02236] | "each 3D coordinate on the ray is equally related to the same 2D coordinate … the distance prediction … along the longitudinal direction become[s] ambiguous" — exactly our lift; depth-aware projection +0.9 NDS on BEVFormer | names our lift's ray-smear; the fix needs a depth estimate | probe after P-DEPTH (B4c) |
| hard depth targets can hurt | DualPathOcc [LIB:2609.06370] | one-hot depth 35.96 vs none 36.92 mIoU; Gaussian 36.94 (Occ3D) | soften the target; a no-supervision control is mandatory | constrains P-DEPTH |
| depth pre-training | DD3D [LIB:2108.06417] | removing depth pre-training: −5.3 (DLA-34) / −10.7 (V2-99) Car BEV AP; depth > 2D-detection pre-training on the same 137k images | the trunk prior matters for monocular range | PI decision (R7) |
| localisation dominates monocular 3-D | Ma et al. [LIB:2103.16237] | "localization error is the vital factor"; distant objects "almost impossible" to localise and they mislead training | our AP@0.5 m 0.030 and the 2.5–3.4× longitudinal error (RESULT §2.5) fit this | supports B2 (refinement) and B4 |
| parameter-free lift, what matters | Simple-BEV [LIB:2206.07959] | the bilinear-sampling lift beats depth-based ones when tuned; resolution < 448 × 800 "drastically worsens"; batch 2 → 40: ~+14 IoU; ResNet-101 deeper-stage output upsampled + concatenated into the 1/8 map | our lift IS this design minus the deeper-stage fusion (we sample layer2 only); our batch is 16 | adopt deeper-stage fusion (M1b); batch: note only |
| denoising queries | DN-DETR [LIB:2203.01305], DINO [LIB:2203.03605] | DN **+1.9 AP**, baseline parity at **50 %** of the epochs; DINO (CDN + mixed query selection) **+6.0 AP @12 ep** over DN-DETR | refcv7 MEASURED unstable matching (0.4–6.5 % of targets keep their slot on one frame, SPEC_REFCV7 A14); DN is the published fix; training-only | adopt (B2) |
| hybrid / one-to-many | H-DETR [LIB:2207.13080], Group DETR [LIB:2207.13085], Co-DETR [LIB:2211.12860] | H-DETR: Deformable-DETR **+1.7 AP**; H-PETRv2 NDS 50.68 → 52.38, mAP +1.52 (memory 7.2 → 11.2 GB). Group DETR: PETR NDS 42.0 → 45.0, mAP 37.4 → 38.8; PETRv2 +1.0 NDS / +1.2 mAP | ~3 positives per window × 300 queries: presence sees few positives → flat scores; one-to-many multiplies positive supervision without changing inference | adopt (B2) |
| one-to-many for MAPS | MapTRv2 [LIB:2308.05736] | roadmap: + depth sup. **+5.1**, + BEV seg +1.3, + PV seg +0.6, + one-to-many **+3.9** mAP (nuScenes, R50, 24 ep) | the vector-map recipe's biggest levers are depth and one-to-many — both reusable in a raster world only partly | informs B4c and B2 |
| IoU-aware score | VarifocalNet [LIB:2008.13367], GFL [LIB:2006.04388] | VFL **+0.9 AP** (RetinaNet, FoveaBox, ATSS), +1.4 (RepPoints); VFNet +2.0 | presence carries no localisation quality (box-head audit: Spearman −0.016); our AP is centre-distance, so q = f(centre distance) | adopt (B2, quality target) |
| position-supervised score in DETR | Stable-DINO [LIB:2304.04742], Align-DETR [LIB:2304.07527] | Stable-DINO **+1.1–1.4 AP** over DINO (position-supervised loss alone +0.6); Align-DETR +0.6 AP on H-DETR | the DETR-native form of B2's quality target | adopt (B2) |
| dynamic anchors / refinement | DAB-DETR [LIB:2201.12329], Deformable DETR [LIB:2010.04159] | Deformable DETR + iterative box refinement **43.8 → 45.4 AP**; DAB updates anchors layer by layer | our anchors are static and never moved (§1); 3 layers refine the loss by 4 % | adopt (B2, refinement) |
| decoder depth & duplicates | DETR [LIB:2005.12872] | +8.2 AP from first to last of 6 layers; NMS helps the FIRST layer only (later layers deduplicate via self-attention); backbone lr 10× below the head | our 3-layer decoder behaves like an early DETR layer: 1.97 / 3.12 boxes per detected object, NMS r 2 m +0.090 AP@2 m (RESULT §2.2, §2.4) | B2 first; deeper decoder only behind the PARAM_BAND ruling |
| geometry in the memory | PETR [LIB:2203.05625], DETR3D [LIB:2110.06922], BEVFormer [LIB:2203.17270] | PETR 2D PE → 3D PE: NDS 0.208 → 0.356, mAP 0.069 → 0.305 (multi-view, so larger than ours will be) | our image tokens carry a learned 2-D table; per-clip camera height 1.20–1.69 m and pitch cannot be represented by one table | adopt (B4) |
| temporal fusion | SOLOFusion [LIB:2210.02443], BEVDet4D [LIB:2203.17054], StreamPETR [LIB:2303.11926] | SOLOFusion history 0 → 16 frames: mAP 0.307 → 0.377, mATE 0.743 → 0.655 | the trunk already computes the W window frames each step; perception reads only the newest | deferred (admissibility of past ego-motion warping to be ruled) |
| raster map heads | HDMapNet [LIB:2107.06307] | HDMapNet(Surr), 6 cameras, nuScenes Table I: divider 40.6 / ped crossing 18.7 / boundary 39.5 % IoU; LSS 38.3 / 14.9 / 39.3 | ours (1 front camera, 100 m, 10 cm, different rasterisation): lane 16.4 % (28.3 % at 0–20 m), crosswalk 10.5 % (23.6 %), edge 4.2 % (6.2 %) — CONTEXT only, not comparable | context |
| vector map heads | MapTR [LIB:2208.14437], MapTRv2 | permutation-equivalent point sets; real-time | a vector head sidesteps the 1-cell IoU problem for lines, but our SAM3 GT is a raster with no instance / vector labels → a GT vectorisation build first | deferred (needs a label build) |
| thin-structure losses | clDice [LIB:2003.07311], boundary loss [LIB:1812.07032], Lovász [LIB:1705.08790] | topology-preserving / distance-transform / IoU-surrogate losses for thin, unbalanced classes | targets lane / edge connectivity and placement at 0–20 m (where the image supports it) | probe (M1d = P-MAP arm PM3) |
| 3-D lanes from a front camera | PersFormer [LIB:2203.11089] | perspective transformer for 3-D lanes from ONE front camera | the closest published setting to our lane task | context / later |
| far field | Far3Det [LIB:2211.13858] | fixed 0.5–4 m thresholds are "too harsh for the far-field, while potentially too relaxed for the near-field"; proposes distance-adaptive thresholds (4 m at 50 m, linear / quadratic) | our 40–100 m map and 40–60 m boxes are image-bound (RESULT §3) | adopt as a METRIC (M5) |
| weight averaging | EMA [LIB:2411.18704] | EMA "requires less learning rate decay", better generalisation, calibration, consistency | our heads were still improving when the LR hit 0 → EMA of weights for every eval checkpoint | adopt (W2) |
| focal loss calibration | Mukhoti et al. [LIB:2002.09437], Focal loss [LIB:1708.02002] | focal loss learns models that are "already very well calibrated" (vs cross-entropy) | consistent with the diagnostics: the analytic focal inversion of our presence is calibrated (ECE 0.017) — the score is calibrated but blind to placement (RESULT §2.3), so the fix is a quality target, not a re-calibration | supports B2 (quality) |

## 4. What NOT to do (measured or argued against)

* **Re-initialise or re-tune the presence prior** — MEASURED irrelevant after training (§1).
* **Per-band decision thresholds for the map** — MEASURED worse pooled (diagnostics §1.2).
* **Chase 10 cm IoU beyond 40 m** — ANALYTIC: < 1 stride-8 feature row per 20 m band; 0.15 m lines are 0.8–1.5 px.
  Report it with a range-adaptive metric (M5); do not spend loss weight or decoder FLOPs there.
* **Rebalance perception against the planner** — MEASURED: the trajectory loss is < 1 % of the trunk gradient norm.
* **Feed a re-gated / recalibrated presence to the planner** — diagnostics F5: a distribution shift for a planner
  trained on flat scores; any score-semantics change (B2's quality target) goes to the planner only through B3's zero-gated path.
* **A bigger decoder first** — DETR shows depth helps, but the PARAM_BAND (2–4 M) is a ruled design guard; B2 adds
  supervision and refinement at 0 new parameters and comes first.
* **Change the IGNORE-radius presence exemption** — ELIMINATED by PROBE-0 (RESULT §2.2): objects next to an IGNORE row
  have fewer duplicates, and exempt slots are 3.8 % / 6.4 % of false positives.
