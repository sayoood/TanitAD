# LIT_DETECTION - 3D box detection: what comparable camera-only works did to train the head, and what that means for us

*TanitAD Research Lab - Architecture & Inference literature agent, 2026-10-27 (Berlin). Request (PI, verbatim):
"Find a clear solution to successfully improve. Review by focused research about comparable successful works regarding
training of bb object detection and bev map extraction. Review architecture and training recipes of the heads."
This file is the 3D-BOX half; the BEV-map half is a sibling agent's. **Literature only: 0 GPU-hours, no code or model
edited, nothing run on Thor.***

**Evidence classes.** PUB-PRIMARY = the number was read in the banked PDF (library key = arXiv id, table named) -
PUB-SECONDARY = read in a search summary only (none is used for a number here) - INHERITED = from our own earlier
documents, not re-verified here - INFERRED = my transfer argument - UNVERIFIED. All 34 cited primaries are banked in
`TanitAD Research Lab/Library/` (sha256 in `raw/lit_det_library_entries.json`); 19 of them were banked by this pass.

---

## 0. The answer in six lines

1. **The head family is NOT the lever.** At matched backbone, resolution and frames, dense-BEV and query heads sit within
   ~1-2 NDS of each other: SparseBEV beats SOLOFusion by +1.1 NDS / +0.5 mAP at R50 256x704 (SparseBEV Sec 4.3);
   BEVNeXt (a *dense* BEV model) beats the query model Sparse4Dv2 by +0.4 NDS and has 5.3 % lower mATE (BEVNeXt Sec 4.3).
   Re-writing our 100-query Hungarian slot head as a CenterPoint heatmap head would be a large change for an
   unmeasured, probably small gain. **Keep the slot head; change what trains it.**
2. **What the winners add is auxiliary supervision of DEPTH / 3D POSITION**, and our own diagnosis points at exactly that
   term: our box error is 2.5-3.4x larger along range than across it (INHERITED, `2026-10-04-refcv8-perception-architecture/
   RESULT.md` 2.5), and MonoDLE's oracle swap shows monocular 3D AP is **depth-limited, not size- or yaw-limited**
   (swap in GT depth: AP3D 11.12 -> 38.01; GT size 11.96; GT orientation 11.88; MonoDLE Tab.1, PUB-PRIMARY).
3. **Ranked by published effect and fit to a single front camera:** (a) finish training - the head was still rising at
   LR 0 (INHERITED); (b) **dense depth supervision on the BEV lift** (CaDDN, one camera, 3,712 images: +10.4 / +7.6 / +6.5
   AP3D from depth-distribution supervision alone; BEVDepth +2.2 mAP; Sparse4Dv2 -8.5 mAP when removed);
   (c) **a perspective-view 3D auxiliary head** (BEVFormer v2: +1.9 mAP / +2.5 NDS, and a control proves it is the
   *perspective* view, not "a second head"); (d) **per-frame anchors from a first stage** instead of 300 frozen random ones
   (Far3D +2.1 mAP; PETR: no anchors = no convergence); (e) denoising + a quality-aware score + training-time BEV-NMS
   (Sparse4Dv3 +0.8 / +0.8 mAP; DenseBEV tau 0.1 vs 0.8 = +1.7 mAP).
4. **A 4.6k-clip, one-camera setting is the KITTI regime, not the nuScenes one.** The strongest *single-camera,
   small-data* recipes (CaDDN, MonoDETR, MonoDLE, MonoFlex, PGD) all lean on depth supervision or depth-uncertainty
   modelling, train 140-250 epochs, and use almost no augmentation beyond flips.
5. **Three popular levers do NOT transfer:** BEVHeight's height-instead-of-depth lift is *worse* than BEVDepth on a
   ~1.5 m ego camera (mAP 0.291 vs 0.315, Tab.7); a DINOv2 backbone gave **no gain** on small mono-3D datasets
   (MonoCoP App. C.1); temporal fusion's big multi-camera gains rest on surround-view overlap and need an admissibility
   ruling on past ego-motion before they are even testable.
6. **Metric caveat that governs every number below.** Published figures are nuScenes **mAP / NDS** (class-mean, averaged
   over centre-distance thresholds; threshold set INHERITED from the nuScenes protocol, not re-read here), 6 cameras.
   Ours is **class-agnostic AP@2 m**, one camera. Gains below are therefore a **ranking of levers**, never forecasts.
   Do not add a published "+2 mAP" to our 0.21.

---

## 1. Our setting, and the deltas that decide transfer

| axis | ours | what the papers assume | consequence |
|---|---|---|---|
| cameras | **1** front-wide, 120 deg cylindrical (azimuth-linear column), 256x640 corpus, refcv8 at 416x1024 | 6 surround (nuScenes), 7 ring (AV2), or 1 pinhole (KITTI 3,712 train imgs; nuScenes-front 28,130 imgs, MonoCoP Sec 5.1) | no cross-view overlap -> no multi-view stereo; depth must come from monocular cues + supervision. State the cylindrical projection before using any pinhole formula. |
| data | ~4.6k clips; 746,946 train windows = 1.08 epochs at 50,400 x 16 (INHERITED) | nuScenes 700 train scenes / 28k keyframes; KITTI 3.7k images | scenes are KITTI-scale, frames are serially correlated -> expect early over-fitting once schedules lengthen (BEVDet Tab.4) |
| labels | `obstacle.offline`, 10 dynamic classes, ~87k cuboids, ground-standing | LiDAR-boxes + LiDAR depth in the dataset | **no per-pixel depth label today**; box-derived foreground depth is free, LiDAR depth is a data build (PI-gated, INHERITED) |
| head | 100-query Hungarian slot decoder over stride-16 tokens + BEV (avg-pooled), 3 layers, presence BCE + class + metre-L1 (`models/box3d_head.py`, `agent_slots.py`); 300 reference points that never moved (median 0.031 m, INHERITED) | dense heatmap (CenterPoint/BEVDet) or DETR3D/PETR/Sparse4D queries | the failure modes we measured (frozen anchors, 1.97 boxes per object, 58 % of queries never a TP) are *set-prediction* pathologies the literature has named fixes for |
| metric | class-agnostic AP@2 m, near band 0-20 m, EVAL-DIAG 1,061 windows | class-mean mAP/NDS | see 0.6. Class quality is **not scored** by our AP - a class collapse is invisible to it |
| deploy | ~100 M params total, Thor | 100-300 M, A100 | prefer **training-only** auxiliaries (aux heads vanish at inference) |

**The brief's "2026-09-21 class-output collapse in the box head" is INHERITED from the PI's request; I could not locate its
artifact in the repo in this pass, so no mechanism below is claimed as its cause.** What the literature does give is the
set of mechanisms that produce such a collapse in a query/Hungarian head (Sec 5), each cheap to test.

---

## 2. Ranked lever table

Rank = (published effect on the failure we MEASURED) x (fit to one camera / small data) / (cost). "Gain" is the paper's own
ablation delta, in the paper's metric, with its table. Cost: **L** = days of dev-box or <= 1 Thor-slot, no new data;
**M** = new module + a launch; **H** = data build or PI decision.

| # | lever | published gain (table) | class | transfer to 1 camera / ~4.6k clips | cost |
|---|---|---|---|---|---|
| 1 | **Train the head to convergence** (warm start refcv7-50,400, re-warmed LR, EMA) | our own: AP@2 m +0.050 [+0.041, +0.058] in the last 20k steps at falling LR (INHERITED). Literature schedules: Sparse4Dv3 100 ep, MonoDETR 195 ep, MonoCoP 250 ep, StreamPETR 24-60 ep; PETR "requires a longer schedule" (Fig 5a) | INHERITED + PUB-PRIMARY | **direct** - ours ran 1.08 epochs of windows | **L** (= 10-04 design B1) |
| 2 | **Dense depth supervision on the lift's depth head** (categorical LID bins, foreground-weighted focal/BCE; training-only) | CaDDN Sec 4.3: depth-distribution supervision **+10.40 / +7.60 / +6.54 AP3D** (E/M/H, KITTI val), fg-weighting alpha_fg 3.25 vs 0.25 **+0.67 / +1.07 / +0.91**, LID vs uniform bins **+3.17 / +1.21 / +1.09**. BEVDepth Tab.4: **+2.2 mAP** (0.282 -> 0.304), camera-aware DepthNet +1.0, refinement +0.8, total **+4.0 mAP/+4.0 NDS**; Tab.1: a frozen *random* depth costs only -3.7 mAP, i.e. un-supervised depth is barely used. Sparse4Dv2 Tab.3: removing dense depth supervision **mAP 0.439 -> 0.354, NDS -10.4, "gradient collapse"** | PUB-PRIMARY | **High.** CaDDN *is* a single-camera LSS-type lift trained on 3,712 images. MonoDLE Tab.1 + our 2.5-3.4x longitudinal error say depth is the bottleneck. Two label routes: (i) **box-derived foreground depth, no new data** (ray-cast each cuboid into the image; MonoDETR Tab.8: object-wise foreground depth beats dense), (ii) LiDAR-projected dense depth (data-gated). Caveat: DualPathOcc found *hard* depth targets -0.96 mIoU (INHERITED, 10-04 design) - use soft/categorical | **L** for (i), **H** for (ii) |
| 3 | **Perspective-view 3D auxiliary head** (FCOS3D/DD3D-style: projected centre, depth, size, yaw, centre-ness) on stride-8/16 trunk features, fed by 3D-box GT | BEVFormer v2 Tab.2 (R101, 48 ep): Perspective+BEV vs BEV-only **+1.9 mAP / +2.5 NDS** (0.355 -> 0.374; 0.426 -> 0.451), mATE .751 -> .730, mAOE .429 -> .379. **Control: "BEV & BEV" (second BEV head) = 0.350 / 0.428, i.e. nothing.** Tab.4 (R50, 24 ep): mAP 0.322 -> 0.351, converges faster. Stated: with it, depth-pretrained backbones "are no longer necessary" | PUB-PRIMARY | **High - it is the single-camera literature itself.** Every mono-3D detector (FCOS3D, PGD, MonoFlex, MonoDLE) is this head. Training-only if used purely as auxiliary | **M** |
| 4 | **Per-frame anchors / queries from a first stage** (2D-or-perspective proposals + depth, or dense BEV peaks), instead of learned fixed references | Far3D Tab.3 (AV2, 150 m): adaptive queries **+2.1 mAP / +1.5 CDS** (20.3 -> 22.4). PETR Tab.5c: with no anchor points the query head "fails to converge". DenseBEV: BEV cells as two-stage queries, pedestrian **+3.8 mAP** nuScenes, +8 LET-mAP Waymo (abstract) | PUB-PRIMARY | **High** - our 300 anchors are their random init (INHERITED). *Conflict, stated:* BEVFormer v2's BEV&BEV control shows a BEV first stage alone adds ~0; the gain came from the perspective stage. -> try perspective-first (lever 3) and BEV-cell second, one variable each | **M** |
| 5 | **Training-time de-duplication + quality-aware score** (BEV-NMS inside the loss; centre-ness / yaw-ness / IoU-aware presence) | DenseBEV Tab.5a: NMS tau 0.1 vs 0.8 **+1.7 mAP / +1.8 NDS**; Sparse4Dv3 Tab.5: centre-ness + yaw-ness **+0.8 mAP, -2.8 pt mATE, -1.9 mAVE**; text: "high-confidence box does not necessarily correspond to a closer GT" | PUB-PRIMARY | **High** - exactly our measured 1.97 boxes/object and "top score on the nearest slot 34.5 %" (INHERITED). Already in the 10-04 design (B2); this pass adds the *sizes* | **M** |
| 6 | **Query denoising** (single-frame, then temporal; range-modulated negatives) | Sparse4Dv3 Tab.5: single-frame DN **+0.8 mAP / +0.9 NDS**, temporal DN +0.4 / +0.6; Far3D Tab.3/5: range-modulated 3D DN **+1.0 mAP**, 2 negatives best (24.4 vs 23.4 with 0); SparseBEV Sec 4.1: "to stabilise training and speed up convergence" | PUB-PRIMARY | Medium-high; no multi-view dependence. Mainly convergence speed - which is our binding constraint | **M** |
| 7 | **Augmentation: BEV-space + image-space** | BEVDet Tab.4: no-aug baseline peaks at epoch 4 (mAP 23.0) and decays to 17.4 by epoch 20; **BEV aug +3.2 mAP at peak; image + BEV aug +8.6 mAP**; image aug alone *hurts* (20.5 < 23.0) | PUB-PRIMARY | **Conditional.** We have not shown over-fitting (AP rising at 1.08 epochs). **Trap:** BEV rotation/scale augmentation moves the BEV that the map head and planner also read - flip-only, or augment only a detached box branch. MonoFlex/MonoDLE: flip (+ 2D-only crop) is the whole small-data recipe | **L** to test, **after** the over-fit measurement in Sec 6 |
| 8 | **Loss shape**: x/y L1 weight 2.0; depth weight 0.2 -> 1.0 fine-tune; velocity weight 0.2; drop or down-weight GT beyond ~60 m | SparseBEV Sec 4.1 (xy 2.0, from StreamPETR); FCOS3D Tab.3: depth-loss in original space **+2.3 mAP**, depth weight 0.2 -> 1.0 fine-tune **+2.1 mAP**; CBGS: velocity 0.2; MonoDLE Tab.8: drop GT > 60 m **+0.7 Mod AP**, > 40 m *hurts* | PUB-PRIMARY | Cheap, but each is a hyper-parameter-sized effect (+0.7 .. +2.3); rank behind structural levers. Our matcher already uses metre-L1 | **L** |
| 9 | **Depth *uncertainty* / geometric depth ensemble** | MonoFlex Tab.4 (KITTI val): direct 12.60 -> +sigma **14.83** -> adaptive ensemble **17.51** Mod AP3D. PGD Tab.3: local consistency + probabilistic depth + geometric propagation **5.51 -> 16.60** Mod AP3D@0.7; Tab.5 nuScenes mAP .319 -> .348 | PUB-PRIMARY | Medium. Needs a keypoint/geometry branch (box corners) - bigger change; the sigma-weighting part is cheap | **M** |
| 10 | **Ray / 3D position embedding on image tokens** (zero-init) | PETR Tab.3: 2D PE -> 3D PE **mAP 0.069 -> 0.305** (NDS .208 -> .356) | PUB-PRIMARY | The size is for a baseline with *no* geometry across 6 views; our BEV tokens already carry metric geometry. Cheap, zero-init (10-04 design B4a). Expect small | **L** |
| 11 | **Temporal fusion** | StreamPETR Tab.5: 1 -> 8 training frames **mAP 31.7 -> 40.2**, mAVE .885 -> .316; SOLOFusion Tab.6: low-res long-term (16 f) **+6.5 mAP**, high-res short-term +2.2; BEVDet4D: NDS 39.2 -> 47.6, mAVE .909 -> .337 **but alignment alone ~+0.9 mAP (Tab.3 row B)** | PUB-PRIMARY | **Uncertain.** Gains are multi-view parallax; a forward-moving single camera has its epipole at image centre - weakest exactly for the lead object. Velocity part is real. **Needs a ruling on past ego-motion as an input** (INHERITED open item) | **H** |
| 12 | **Foundation backbone / depth prior** (DINOv2, Depth Anything V2, Metric3Dv2) | MonoDINO-DETR Tab.III (KITTI Car Mod AP3D): MonoDETR 17.52 -> +DINOv2+DPT depth head 18.69 -> 19.19 with a hierarchical fusion block. **MonoCoP App. C.1: DINOv2 frozen or fine-tuned = no gain** (small data, domain gap). Frozen DINOv2 + Metric3Dv2 *Giant* in LSS (BEV vehicle **segmentation**): +8.9 IoU, +7.4 with half the data; Small models do not beat the full-data baseline | PUB-PRIMARY | **Mixed; evidence is contradictory, so do not swap the trunk on this.** The robust part is a *metric-depth prior* feeding the lift, not the encoder. DD3D Tab.3/6: depth-pretraining -5.3 / -10.7 BEV AP when removed; depth beats 2D-det pretraining on the same 137k images, 21.8 vs 20.5 | **M-H** |
| 13 | **Resolution / batch size** | Simple-BEV (BEV vehicle seg): batch 2 -> 40 **~+14 IoU**; 448x800 -> 672x1200 **47.4 -> 49.3 IoU**. SOLOFusion Tab.7: R50 256x704 + 16 frames beats BEVDepth 512x1408 | PUB-PRIMARY (seg, not detection) | We are already 416x1024. **Check the batch we actually train the box head at** (A8 used batch 2; refcv7 used 16) | **L** |
| 14 | **Class balancing (CBGS, 4.5x resampling)** | CBGS: nuScenes mean 29.5 -> 52.8 vs PointPillars with all changes, bicycle 1.6 -> 22.3 (Tab.2/3); StreamPETR, Sparse4Dv3, SparseBEV all train **without** CBGS | PUB-PRIMARY | Irrelevant to class-*agnostic* AP@2 m; relevant only to a class collapse. Use class-balanced / logit-adjusted focal (INFERRED), not 4.5x epochs | **L** |
| 15 | **Height-instead-of-depth lift** (BEVHeight) | Tab.7 nuScenes ego camera: **mAP 0.291 vs BEVDepth 0.315 (-2.4)**; wins only on high roadside cameras | PUB-PRIMARY | **Negative.** Our camera is 1.43-1.56 m (INHERITED). Do not pursue | - |

---

## 3. Per-paper extraction (head | recipe | what each component bought | what transfers)

*Only the papers whose primaries were read are listed; numbers are the papers' own, table in brackets.*

**Dense BEV, CenterPoint-style head**
* **BEVDet** [2112.11790]. LSS lift -> ResNet BEV encoder -> CenterPoint head, 0.8 m cells in a 51.2 m ROI. AdamW 2e-4, batch 64, 20 ep,
  step LR; image aug (scale, rot +-5.4 deg, flip, crop) **and** BEV aug (flip, rot +-22.5 deg, scale .95-1.05); CBGS. Aug story: Tab.4 (see lever 7);
  BEV encoder +1.7 mAP; Scale-NMS +4.8 AP pedestrian / +7.5 traffic cone. *Transfers:* the over-fitting lesson and the BEV-encoder; *not:* rot-aug on a shared BEV.
* **BEVDet4D** [2203.17054] - alignment + velocity-offset target; velocity/orientation, not localisation, is what temporal buys first.
* **BEVDepth** [2206.10092]. Lever 2. Recipe: R50 256x704, AdamW 2e-4, batch 64, 24 ep (no CBGS in ablations), camera-aware DepthNet at stride 16.
  Depth quality: SILog 54.58 -> 27.62 with the depth loss (Tab.2). *Transfers fully* except LiDAR depth GT.
* **BEVNeXt** [2312.01696] - revives dense BEV with CRF depth + a second-stage object decoder; total **+3.1 mAP / +2.2 NDS** (Tab.5). Tab.4 caption:
  CRF modulation is minor once depth supervision is dense. Evidence that a *dense* model matches query models when its depth is supervised.
* **CenterPoint** [2006.11275] - K-channel Gaussian heatmap, focal loss, radius max(f(w,l), 2); +3.8-4.1 mAP over anchors on nuScenes (Tab.6, LiDAR).
  *Per-class sigmoid heatmaps cannot collapse into one softmax* (INFERRED, relevant to Sec 5). **CBGS** [1908.09492] - see lever 14.
* **Simple-BEV** [2206.07959]; **frozen foundation LSS** [2501.08118] - BEV *segmentation* evidence only; lever 12/13.

**Query / DETR-style**
* **PETR** [2203.05625] - 3D PE + learned queries from anchor points (Learned-3D best), LID 64 depth bins, lambda_cls 2.0, AdamW 2e-4 cosine, 24 ep, batch 8,
  GT rot +-22.5 deg. Tab.3 / 5c: levers 4 and 10.
* **StreamPETR** [2303.11926] - object-query propagation through a memory queue; AdamW 4e-4, batch 16, cosine, 24 ep ablation / 60 final, no CBGS; implicit motion-aware LN
  **+2.0 mAP** (Tab.6), explicit motion compensation did *not* help. Built on Focal-PETR's auxiliary 2D supervision. ViT-L trained only 24 ep "to prevent over-fitting".
* **SparseBEV** [2308.09244] - BEV-pillar queries (45.4 vs 44.0 mAP for 3D reference points, Tab.3), scale-adaptive self-attention (41.4 -> 45.4 mAP, Tab.4),
  Hungarian + focal + L1, xy weight 2.0, query denoising, AdamW 2e-4 batch 8 cosine, 24 ep no CBGS, 8 frames.
* **Sparse4Dv2** [2305.14018] - recurrent temporal fusion + **dense depth supervision (multi-scale, L1, at an "equivalent focal length", training-only)**; lever 2.
  **Sparse4Dv3** [2311.11722] - 6-layer decoder, 900 instances, 5 DN groups (3 temporal), AdamW **100 ep** no CBGS; Tab.5 chain 0.439 -> 0.469 mAP: DN +0.8, decoupled attention **+1.1**, temporal DN +0.4, quality +0.8.
* **Far3D** [2308.09616] - 2D proposals + depth prior -> 3D adaptive queries, perspective-aware aggregation, range-modulated DN; AdamW 2e-4, batch 8, 6 ep on AV2.
  *Transfers:* the adaptive-query idea is natural for ONE camera (the 2D proposal -> 3D point step is monocular by construction).
* **BEVFormer v2** [2211.10439] - lever 3. Perspective head = DD3D-style with camera-aware depth parameterisation; lambda_bev = lambda_pers = 1; AdamW 4e-4; 48 ep.
* **DenseBEV** [2512.16818] - every BEV cell is a candidate anchor; BEV-NMS inside training with an attention mask so gradients flow only through non-suppressed objects;
  grid 100x100 52.7 NDS -> 300x300 53.6 (Tab.5b) at +50 % train time. *Transfers:* the training-time NMS idea (our BEV is 120x64 cells).
* **CAM3DNet** [2604.17024], **StereoMV2D** [2512.17620], **RoPETR** [2504.12643], **GUPNet++** [2310.15624], **SMOKE** [2002.10111] - banked; abstract-level only, **no number used**.

**Single-camera / small-data (the closest analogues to us)**
* **CaDDN** [2103.01100] - categorical depth distribution + LID bins -> frustum -> BEV -> 3D head; **80 epochs, batch 4, Adam 1e-3 one-cycle, flip only**, lambda_depth 3.0 / cls 1.0 / reg 2.0 / dir 0.2. KITTI 3,712 train images. Lever 2.
* **MonoDETR** [2203.13310] - depth-guided transformer: foreground depth map (80 bins over 0-60 m, LID) supervised only on object regions; 50 queries; **195 ep, batch 16, AdamW 2e-4, LR x0.1 at 125/165**;
  discards training samples with depth > 65 m or < 2 m. Tab.5 (Car val AP3D E/M/H): 28.84 / **20.61** / 16.38; without the depth-guided transformer 19.69 / 15.15 / 13.93; without depth guidance 24.14 / 17.81 / 15.60.
* **MonoDLE** [2103.16237] - centre-heatmap head, 140 ep, batch 16, Adam 1.25e-3, **random crop/scale for the 2D branch only + flip** ("to avoid over-fitting"); error-analysis Tab.1 and sample-pruning Tab.8 (levers 2, 8).
* **MonoFlex** [2104.02323] - DLA-34 at 384x1280, 34k iterations, batch 7, AdamW 3e-4, **flip is the only augmentation**; edge-fusion module for truncated objects (relevant to a 120 deg lens);
  depth ensemble Tab.4 (lever 9). **PGD** [2107.14160], **FCOS3D** [2104.10956] (Tab.3 chain 22.7 -> 34.3 mAP: depth loss in original space, distance-based target assignment (the NDS lever), disentangled heads (mAOE .81 -> .64), DCN +1.5, depth-weight fine-tune; SGD 0.002, batch 32).
* **DD3D** [2108.06417] - lever 12. **MonoCoP** [2505.04594] - DINOv2 negative; 250 ep, lr 2e-4, median of 5 runs reported. **MonoDINO-DETR** [2502.00315] - 195 ep, batch 8; DPT/Depth-Anything-V2 depth branch. **BEVHeight** [2303.08498] - lever 15.

---

## 4. What does NOT transfer from surround-view / nuScenes-scale (and what that removes from the menu)

* **Multi-view overlap stereo** (SOLOFusion short-term, BEVStereo, DETR3D's cross-view sampling, PETR's multi-view PE) - one camera has no overlap.
* **Temporal fusion as a *localisation* lever** - the +6.5 / +8.5 mAP are on 6 cameras with 0.5 s keyframes; forward ego-motion gives a depth-ambiguous centre of the image. Keep as a *velocity* lever only, and only after the ego-motion admissibility ruling.
* **CBGS 4.5x** - the three strongest recent query models skip it.
* **Height-based lift** - negative at ego height (lever 15).
* **A frozen foundation encoder as a drop-in** - contradictory evidence (lever 12).
* **Published absolute numbers** - nuScenes mAP/NDS and Argoverse CDS are not comparable to class-agnostic AP@2 m (0.6).
* **Scale of recipes** - 8 x A100 for 24-100 epochs. Our compute is one Thor / A40; the *auxiliary* levers (2, 3, 5, 6) cost training time only and vanish at inference, which is why they rank above backbone swaps.

---

## 5. Mapping our MEASURED failures to published mechanisms

| our measurement (INHERITED unless noted) | published mechanism | lever |
|---|---|---|
| box error 2.5-3.4x larger along range than across (|dx| 0.96 vs |dy| 0.38 m at 0-20 m) | monocular 3D AP is depth-limited (MonoDLE Tab.1); LSS without depth supervision uses only partial depth (BEVDepth Sec 3.2-3.3) | 2, 3, 9 |
| 300 reference points never moved (0.031 m); 58 % of queries never a TP; 16.5 % of TPs at the tanh limit | learned fixed anchors give ~0 mean gradient; PETR with no anchors does not converge; Far3D / DenseBEV re-seed queries per frame | 4 |
| 1.97 boxes per object; top score is the nearest slot only 34.5 % of the time | one-to-one matching + confidence trained by classification does not rank by placement (Sparse4Dv3 Sec 4.4); training-time BEV-NMS (DenseBEV) | 5 |
| AP@2 m +0.050 in the last 20k steps at LR -> 0 after 1.08 epochs | query heads need long schedules (PETR Fig 5a; Sparse4Dv3 100 ep; MonoDETR 195 ep); denoising shortens them | 1, 6 |
| *class-output collapse (brief, artifact not located, UNVERIFIED)* | candidate mechanisms: (a) gradient collapse of a query head from ImageNet init without dense geometric supervision (Sparse4Dv2 Tab.3: -8.5 mAP and a failed run); (b) majority-class softmax under heavy imbalance (CBGS rationale); (c) Hungarian cost dominated by presence early on (DN fixes) | 2, 6, 14 |
| AP@1 m rising fastest late (+55 %) | localisation, not detection, is what is unfinished - again depth / 3D position | 2, 3 |

---

## 6. Recommendation

### 6.1 Head: keep the slot decoder, give it a first stage and a geometric teacher

**Decision: do NOT replace the 100-query Hungarian slot head by a CenterPoint heatmap head.** Reasons: (i) the
published head-family gap at matched inputs is ~1-2 NDS, far smaller than the depth-supervision effect (lever 2) and
than the 0.248 -> 0.349 (AP@2 m) *oracle* headroom we measured from ranking and de-duplication alone (INHERITED);
(ii) the planner seam (`AgentSlotDecoder` contract) is built on slots; (iii) no controlled head-vs-head experiment exists for
one camera at ~4.6k clips, so a swap would be an unmeasured bet. **One concession to the dense family:** a CenterPoint-style
per-class sigmoid Gaussian heatmap is the cleanest *first stage* (3x3 max-pool NMS, one peak per object, per-class channels that
cannot collapse into one softmax) - include it as arm A2 below, not as a replacement.

Proposed head ("PERSP-SLOT"), all additions zero-gated so step 0 reproduces the warm-start checkpoint:
1. **Perspective 3D auxiliary head** on the stride-8/16 trunk maps (FCOS3D/MonoFlex-style: projected-centre heatmap, depth, size, yaw as sin/cos, centre-ness), 3D-box GT projected through the **cylindrical** model (column linear in azimuth; verify the projected centre against the rig's 120 deg FOV, as the pinhole formula gives 92.6 deg). Supervised with **box-derived foreground depth** (cuboid ray-cast, categorical LID bins over 0-60 m near, soft) - lever 2(i) + 3.
2. **Dense depth supervision on the BEV lift's depth distribution** with the same labels, alpha_fg = 3.25 / alpha_bg = 0.25 (CaDDN), camera-aware depth input (BEVDepth).
3. **Per-frame queries**: top-K (K ~ 100) of the perspective head's (centre, depth) -> 3D points seed the slot anchors per frame; keep a minority of learned global queries (Far3D: adaptive + global). Replace the frozen 300-point table.
4. **Training-time quality + de-dup**: IoU/centre-ness-aware presence target, BEV-NMS tau ~ 0.1 inside the loss with the attention mask (DenseBEV), contrastive DN with range-modulated negatives (Far3D: 2 negatives).
5. **Classification as per-class sigmoid focal** with a class-balanced / logit-adjusted weight, and a **per-class readout** (never-predicted classes, per-class AP, prediction histogram) so a collapse is visible - our class-agnostic AP cannot see it.

### 6.2 Recipe

* **Warm start refcv7-50,400, EMA weights, re-warmed LR** (lever 1; the design B1 stands). AdamW, head lr ~ 2e-4 / encoder x0.5 (already ours), cosine, **>= 24 epochs-equivalent of *clips*, not windows** - the budget should be stated in clips seen, because our windows are serially correlated.
* **Loss**: focal presence/class; metre-L1 with x/y weight 2.0 (SparseBEV); depth/longitudinal term ramped 0.2 -> 1.0 (FCOS3D); velocity weight 0.2; optionally drop GT beyond ~60 m from the *detection* loss only (MonoDLE: 60 m helps, 40 m hurts).
* **Augmentation**: horizontal flip (with yaw/lateral label flip) as default - the whole small-data recipe (MonoDLE, MonoFlex, CaDDN). Image scale/crop only after the Sec 6.3 gap is measured. **BEV rotation/scale aug only on a detached box branch** (it perturbs the shared BEV the map head and planner read).
* **Batch**: audit the box head's effective batch (Simple-BEV: a large batch mattered most); A8 ran batch 2.
* **Not now:** temporal fusion, foundation-encoder swap, height lift, CBGS resampling, resolution increase (already 416x1024).

### 6.3 Cheapest discriminating experiments (Rule Zero: every one names the next step)

Each is one variable off a **B1 continue-training control** (same warm start, schedule, windows), read on the 1,061 EVAL-DIAG windows with the
episode-cluster paired bootstrap, **and against a replicate/seed floor** (H-ESTIM-SEED-1: a separated CI is necessary, not sufficient). Report AP@1/2/4 m,
near band 0-20 m, longitudinal |dx| and lateral |dy|, boxes per object, per-class AP and never-predicted classes. Tier T0 (labels only).

| arm | variable | SUPPORTED if | REFUTED if -> next lever |
|---|---|---|---|
| **E0** (0 GPU) | train-split vs held-out box AP on refcv7-50,400, equal window counts | train AP exceeds held-out by more than the bootstrap half-width -> over-fitting is live, lever 7 earns a slot | gap inside CI -> we are under-trained, not over-fit: skip aug, keep lever 1 |
| **E1** | box-derived foreground-depth supervision on the lift (lever 2i) | near-band longitudinal |dx| falls beyond the replicate floor, AP@1 m rises | no move -> depth was not the bottleneck at this stage: go to E2; or build LiDAR depth (PI item) |
| **E2** | perspective 3D aux head (lever 3), aux only | AP@2 m / mATE-analogue improve beyond floor; the *control* (a second BEV head, BEVFormer v2 style) does NOT | gain only in the control -> it was capacity, not view; stop |
| **E3** | per-frame anchors from E2's proposals + BEV-NMS tau 0.1 (levers 4, 5) | boxes/object -> ~1 without NMS, nearest-slot-is-top rises from 34.5 %, AP@1 m up | no move -> ranking is not the lever; next: DN (lever 6) |
| **E4** | per-class sigmoid focal + class-balanced weight | per-class AP for the rarest classes rises, no never-predicted class | no move -> the collapse is upstream (trunk/data), inspect label census |

**Order by effect/cost: E0 (free) -> E1 -> E2 -> E3 -> E4.** None needs a PI decision except LiDAR depth (lever 2ii) and the
ego-motion ruling (lever 11); both are named blockers, not reasons to wait.

---

## 7. Caveats and open items (named, not hidden)

* No published head-vs-head comparison for **one camera at ~4.6k clips** exists in what I read. The "keep the slot head" decision rests on the 1-2 NDS head-family gap at *multi-camera* matched inputs plus our own measured set-prediction pathologies; it is INFERRED for our setting.
* BEVFormer v2's BEV&BEV control (a BEV first stage adds nothing) and DenseBEV (BEV cells as anchors help) disagree on a BEV-first design; E3 runs both orders to settle it.
* MonoCoP (DINOv2: no gain) and MonoDINO-DETR (+1.17 Mod AP3D with a DPT depth branch) disagree on foundation backbones; unresolved, deliberately not in the plan.
* Two recipe papers (CAM3DNet, StereoMV2D) are 2025-26 and banked but their numbers are not used.
* The 2026-09-21 class-output collapse finding was not located; the class readout in E4 is how to turn it into a MEASURED artifact.
* Which augmentations refcv7/8 apply to the box branch today is UNVERIFIED (not read from the trainer in this pass).
