# LIT_BEVMAP — BEV map extraction: what the published work says and what to change

*TanitAD Research Lab literature agent, 2026-10-27. PI request: "Find a clear solution to successfully improve ... review architecture and training recipes of the heads, review size and representation of the map." Scope: the BEV MAP half (a sibling agent owns 3D detection). 0 GPU, 0 Thor, nothing written to G: except this file's landing path.*

## 0. Headline (read this if nothing else)

1. **The thin-class failure (edge 0.000, lane 0.164) is a representation x localisation problem, and none of the papers read trains what we train.** In the 38 papers read, no thin-class raster is supervised as 1-2 px (10 cm) hard lines under an 8-way softmax. They (a) draw thin classes as 3-5 px wide lines on a 0.15 m grid (about 0.5-0.75 m wide; BEVSegFormer §4.2), or (b) use 0.2-0.5 m cells plus a sub-cell offset (BEV-LaneDet Table 5: 0.05 m cells score 43.2 F-score vs 57.9 at 0.5 m; 0.5 m + offset 58.4), or (c) leave the raster and regress polylines scored at 0.5-1.5 m Chamfer tolerance (MapTR Table 1). Our own audit's analytic ceiling says the same thing (edge 0.04 at sigma = 0.2 m with hard 10 cm labels; INH, `Research/2026-09-26-map-signal-audit/RESULT.md` §2c).
2. **More CE steps will not move it.** The map-only fine-tune was flat because the objective is converged on a target the features cannot localise (stride-8 cell = 0.33 m laterally at 20 m, 0.65 m at 40 m; EST §2). The levers are in the target, the decoder and the data, not in the optimiser.
3. **Recommended change (details §9):** supervise thin classes on a **0.25 m grid (the lift's native cell) as dilated sigmoid heatmaps + sub-cell offset / signed-distance channels**, derive "non-drivable edge" from a drivable-area signed-distance channel instead of a 1-px class, keep 10 cm hard labels only for 0-10 m (the existing near-lift patch), keep area classes (drivable, sidewalk, crosswalk, hatched) as hard-label raster, add per-class focal+Dice(+boundary) on the thin group, add an explicit "unobservable" treatment for the blind near field, and **re-score thin classes under tolerance** (published protocols are tolerance protocols).
4. **Ranked by expected value per cost (full table §8):** (1) thin-class target+decoder redesign, (2) loss set + per-class thresholds on the thin group, (3) blind-zone / unobservable-cell handling (hits the measured 25 % sidewalk precision at 0-5 m), (4) label-ceiling audit (CPU, decides how far any head change can go), (5) image-plane auxiliary supervision, (6) temporal fusion (best-evidenced architecture lever, but needs a vision-only ruling and is the most expensive).
5. **Honest limits.** No paper read uses ONE front camera for vector map extraction (gap, §10). Thin-class raster IoU for a single front camera in the published record is 30-37 (BEVSegFormer Table 2) on a dilated GT, so "lane 0.164 on pixel-exact 10 cm GT" and "37 on 0.75 m-wide GT" are not comparable numbers; do not read the gap as a deficit until our metric is re-scored under the published protocol.

**Integration asks (for the Master Mind):** (i) bank the 38 cited PDFs (Appendix A has the exact `kb_add.py` loop; this agent could not run it because it writes into G:); (ii) PI ruling on past-frame ego-motion warp vs the vision-only rule (§9.5); (iii) decide whether the thin-class redesign is registered as a new arm (this document is the literature case, not a pre-registration).

---

## 1. Evidence legend and banking status

| tag | meaning |
|---|---|
| PUB-P | PUBLISHED, read in the primary PDF in this session (arXiv id; table / section named) |
| PUB-P(rep) | read in a primary, but the number belongs to ANOTHER method reported in that table (e.g. VectorMapNet in MapTR Table 1) |
| INH | INHERITED from the PI's brief or a repo document, not re-verified here |
| EST | ESTIMATED: my arithmetic, assumptions stated |
| HYP | HYPOTHESIS: inference, no published ablation |

**Banking.** `tools/kb_add.py` writes PDFs and `library.json` under `TanitAD Research Lab/Library/` on G:, which this agent may not write. All 38 cited PDFs are downloaded (arXiv, sha256 recorded) to `F:/c_overflow_2026-10-26/tmp/lit_map/<id>/<id>.pdf` with extracted text beside them. Appendix A lists id, bytes, pages, sha256 and the command to bank each. **Until banked, every number below is PUB-P (read) but NOT admissible for the registry or the paper.** arXiv's metadata API returned HTTP 429 during the session, so titles are taken from the PDFs' first pages.

## 2. Our setting, and the geometry that sets the ceiling

**Delta against the published record** (INH from the brief unless noted):

| item | ours | typical published map-seg / map-extraction setup |
|---|---|---|
| camera | ONE front-wide, 120 deg cylindrical, trained at 416x1024 (corpus 256x640, f_ref 305.577) | 6 surround cameras (nuScenes, AV2 7 cam); front-only appears only in BEVSegFormer Table 2/3, BEV-LaneDet, PON/TIIM/PanopticBEV/SkyEye (per-image monocular) |
| data | about 4.6k clips; map GT on 4,719 clips | nuScenes 28,130 train samples / 700 scenes at 2 Hz |
| map GT | SAM3-derived world map, multi-frame ("labels use every frame of the clip", `semantic_map_gt_fine.py` docstring), 8 classes at 10 cm, `not seen` = 255 ignored | human-annotated vector HD map rasterised at 0.15-0.5 m; thin classes 3-5 px wide |
| head | 0.25 m lift off the stride-8 trunk, bilinear x2.5 refine, 600x320 logits at 10 cm, hard-label CE | 0.15-0.5 m logits; sigmoid/focal/Dice (CVT, BEVFusion, BeMapNet) |
| class weights | brief says median-frequency; **source says sqrt(median-frequency), clipped at 25 (`sqrt_mf`, A8; `compute_map_class_weights.py` header)**: CONFLICT, resolve from the run's weights JSON `definition_id` | inverse-sqrt frequency (PON, PanopticBEV) |
| planner interface | planner BEV = 0.25 m features 2x2-avg-pooled to 0.5 m (`map_head_hires.py` docstring) | n/a |
| size | about 100 M params total | map heads are small: 1.2-1.7 M (FIN Table 3) |

**What is already in place and therefore NOT a lever** (verified in source this session): not-seen cells carry exactly zero loss (`ignore_index=255`, mutation-pinned); class weights are the inverse-sqrt family (PON/PanopticBEV precedent); a prior-corrected and a per-class-threshold decision rule exist; a tolerant-IoU monitor (2 cells) exists; a 0.1 m near-range lift and near refine block exist (NEW-2 R2/R3).

**Geometry of what the trunk can resolve** (EST; assumptions: cylindrical projection so image row v = f*h/rho, f = 1024/(120 deg in rad) = 488.9 px/rad at 1024 px, camera height h = 1.5 m (1.3-1.8 m changes depth rows by about +-15 %), zero pitch, stride-8 feature cell = 8 input px):

| range r | lateral size of one stride-8 cell | ground depth spanned by one stride-8 row | a 12 cm lane line is | 
|---|---|---|---|
| 5 m | 0.08 m | 0.27 m | 11.7 px = 1.5 cells |
| 10 m | 0.16 m | 1.09 m | 5.9 px = 0.73 cells |
| 20 m | 0.33 m | 4.4 m | 2.9 px = 0.37 cells |
| 40 m | 0.65 m | 17.5 m | 1.5 px = 0.18 cells |

Reading: a 0.25 m lift is oversampled laterally beyond about 12-15 m and a 10 cm output is supported by the stride-8 map only inside about 6-8 m. The audit's own stride-16 localisation estimate (0.35-1.25 m per feature, INH) is the same effect one stride up. The CNN decodes sub-cell position, so the lateral error is a fraction of the footprint, but that is exactly the sigma >= 0.2 m regime in which the audit's analytic ceiling for hard 10 cm labels collapses (edge 0.04, lane 0.68 with perfect information; INH).

## 3. Representation and size: raster vs vector, and what cell/range the record supports

### 3.1 Grids actually used (PUB-P unless noted)

| work (arXiv) | input | output cell / range | thin-class GT drawn as | loss |
|---|---|---|---|---|
| LSS (2008.05711) | 6 cam | 0.5 m, 100x100 m, 200x200 (§4) | lane: mask, BCE pos_weight 5.0 | BCE, 300k steps, Adam 1e-3 |
| CVT (2205.02833) | 6 cam | 0.5 m, 100x100 m (also 0.25 m, 100x50 m setting) | binary masks | focal, 30 ep, AdamW one-cycle, EfficientNet-B4, latent 25x25 upsampled x8 |
| BEVFusion (2205.13542) | 6 cam (+LiDAR) | eval [-50,50]^2 m | per-class binary | **multi-binary sigmoid + focal; per-class best-IoU threshold at eval** |
| PON (2003.13402) | 1 image | 0.25 m, 50 m ahead x +-25 m | binary | balanced BCE, alpha_c = sqrt(inverse class frequency), max-entropy loss on non-visible cells |
| TIIM (2110.00966) | 1 image | 100x100 px grid ("each pixel 0.5 m^2"), upsampled to 200x200 for evaluation (§IV) | binary | multi-scale Dice, Adam 5e-5, 40 ep |
| PanopticBEV (2108.03227) | 1 image 1408x768 / 768x448 | **0.074 / 0.077 m/px, 768x704 / 896x768 px (about 57x52 m / 69x59 m)** | area classes + things (no lane lines) | weighted CE + hard mining, weights = inverse sqrt frequency + distance taper |
| HDMapNet protocol, as stated in BEVSegFormer (2203.04050) §4.2 | 6 cam; front-only [0,60]x[-15,15] m | mask 400x200 (about 0.15 m) | **"line segments with 5-pixel width"** (Nullmax front-camera set: 3 px at 512x128 over 80x20 m) | weighted CE [1,15,15,15] |
| Busch 2026 (2602.18066) | 6 cam, 512x288 | 150x150 query grid over 30x60 m (0.2x0.4 m) | nuScenes map layers | CE |
| BEV-LaneDet (2210.06006) | 1 front cam | **0.5 m, 200x40 over x 3-103 m, y +-10 m** | confidence + embedding + offset + height | BCE (conf), MSE (offset) |
| MapTR / MapTRv2 (2208.14437, 2308.05736) | 6 cam | BEV feature 0.3 m, 60x30 m; 20 points/polyline | polylines (vector) | focal cls + point2point L1 + edge-direction; v2 adds dense aux |
| AugMapNet (2503.13430) | 6 cam | aux raster 100x50x3 (0.6 m) over 60x30 m | raster aux only | Dice (aux) |
| PointBeV (2312.00703) | 6 cam | 0.5 m, 100x100 m, **sparse**: trained on 5k of 40k cells | binary | seg + offset + centerness |

### 3.2 Which representation wins for thin classes, and why (PUB-P)

* **Vector instance queries score far above raster-then-vectorise on the same benchmark**: HDMapNet (camera) 23.0 mAP (30 ep) vs MapTR-tiny 50.3 (24 ep) and 58.7 (110 ep) (MapTR Table 1); MapTRv2 61.5 at 24 ep (MapTRv2 Table 3). Divider AP 21.7 -> 51.5/62.4, boundary 33.0 -> 53.1/62.4 (MapTR Table 1; MapTRv2 Table 4).
* **The authors' stated reasons**: raster maps carry no instance information and need heavy post-processing (MapTRv2 §1; PivotNet §1 lists bandwidth, grid-cell independence that "ignores the geometric relationship", and post-processing); permutation-equivalent point-set supervision puts the loss on the line itself (20 points) instead of on a few hundred mostly-background pixels.
* **Three caveats that bind our case.**
  1. *Tolerance.* Those AP values use Chamfer thresholds 0.5/1.0/1.5 m (MapTR §4). At 0.2/0.5/1.0 m MapTR-R50 24 ep drops to 27.3 mAP and MapTRv2 to 37.2 (MapQR 2402.17430 Table 1, PUB-P(rep)); BeMapNet's Bezier curves score 40.7 (EfficientNet-B0) / 41.3 (R50) / 43.7 (SwinT) at 30 ep under the same tight protocol (BeMapNet Table 1; MapQR Table 1) and a Bezier head beats a polyline head by 4.5 AP (BeMapNet Table 5).
  2. *Geography.* The original nuScenes split overlaps the validation set in more than 84 % of locations (StreamMapNet §4.1); on a geographically disjoint split MapTR drops to 20.9 mAP and StreamMapNet reaches 33.9 at 60x30 m (StreamMapNet Table 2); MapTracker 40.3 (MapTracker Table 3). StreamMapNet itself reports "approximately a 50 % performance decrease across all methods" on the new split (§4.5). Published vector numbers on the original split overstate generalisation.
  3. *Cameras.* Every vector result read is 6-camera. **No single-front-camera vector result exists in the papers read (gap).** The single-front-camera record is raster: BEVSegFormer front-only nuScenes divider 37.54 / ped-crossing 17.82 / boundary 34.25 / all 29.87 IoU (Table 2, 5,000 queries) vs 51.08/32.59/49.97/44.55 with six cameras (Table 1): about -13 to -16 IoU for the camera count; Nullmax highway front camera lane IoU 67.04 (Table 3); BEV-LaneDet OpenLane F-score 58.4 (Table 5).
* **Cell size for thin structures (PUB-P, BEV-LaneDet Table 5, single front camera, OpenLane).** F-score 43.2 at 0.05 m, 55.7 at 0.2 m, 57.9 at 0.5 m, 56.8 at 1 m; adding a sub-cell offset: 58.4 at 0.5 m (x-error near 0.429 -> 0.309, far 0.734 -> 0.659) and 57.7 at 1 m (near 0.607 -> 0.317). The authors' reason: a too-small cell unbalances positives vs negatives and starves the confidence branch, a too-large one overlaps embeddings; they recommend 0.5 m + offset. GFLOPs fall 735 -> 53 from 0.05 m to 0.5 m.
* **Dense hybrids** keep a coarse raster only as supervision/prior: AugMapNet's raster is 0.6 m cells and still adds +4.4 % mAP (Table 6: 36.7 -> 38.3 at 60x30 m; 28.9 vs 26.2 at 100x50 m); BeMapNet's dense semantic + instance aux takes the vector head from 20.0 to 43.7 mAP (Table 4).
* **Newest representation (2610.25284, Oct 2026).** The "Lane Field" densifies vector geometry into a 5-channel BEV tensor (direction, proximity, confidence); on OpenLane-V2 / LaneSegNet it adds +1.1 (attention guidance), +2.8 (BEV feature modulation), +4.0 mAP (both) (Table 2), and on nuScenes MapTRv2 first column 55.8 -> 57.7 (Table 4; column labels not recoverable from the text layer). Multi-camera; vector decoders downstream.

### 3.3 Range ("size") (PUB-P)

Accuracy falls fast with range, in every study that ablates it: StreamMapNet 33.9 (60x30 m) -> 23.0 (100x50 m) on the disjoint split (Table 2); AugMapNet baseline 33.8 / 23.2 / 6.9 and AugMapNet 38.3 / 28.9 / 9.8 at 60x30 / 100x50 / 150x75 m (Table 6). A single forward camera adds a physical reason (EST §2: one stride-8 row spans 4-17 m of ground at 20-40 m). The record therefore supports a **large extent at coarse far-field resolution**, not a large extent at uniform 10 cm.

## 4. Head architecture, losses and decoding (PUB-P)

| lever | evidence | delta |
|---|---|---|
| per-class sigmoid (overlapping classes) | BEVFusion §III: "different map categories may overlap ... multiple binary segmentation ... focal loss [following CVT]"; eval takes the best IoU over thresholds per class | camera-only BEVFusion divider 46.4 vs LSS 36.5, CVT 29.4; mean 56.6 vs 44.4 / 40.2 (Table III). The gain mixes backbone, depth/BEV-pool and loss: **no isolated loss ablation in that paper** |
| loss set | FIN (2510.00651) Table 4, nuScenes 6-class map seg, 0.5 m, ResNet-50, camera+radar, 24 ep: focal 46.3 -> +Dice 47.8 -> +Lovasz 49.8 -> +affinity 50.0 -> +boundary (raw) 52.4 -> +normalised boundary 53.5 mIoU | +7.2 mIoU. **Internal inconsistency in the primary:** §V-B text gives focal-only 49.6 and "worse by 2.8"; Table 4 gives 46.3. Quote the range +2.8 to +7.2. Camera-radar, not camera-only |
| normalised boundary loss | FIN eq. (11)-(12): distance-weighted, alpha 0.1 outside / 1.0 inside; Kervadec et al. (1812.07032) motivation: regional losses of very unbalanced classes differ by orders of magnitude, a contour-distance loss does not | +1.1 mIoU over raw boundary (FIN Table 4) |
| UNet-style head vs plain conv head | FIN Table 3: 1.7 M params 53.5 vs 1.2 M params 52.3 | +1.2 mIoU; head capacity is NOT a large lever; AugMapNet Table 5: gains saturate at 2 conv layers |
| Dice with CE | BeMapNet L = CE + Dice on semantic/instance aux ("Multi-task Auxiliary Loss" paragraph); TIIM multi-scale Dice; MapTracker focal + per-class Dice on its BEV aux; AugMapNet Dice aux | no isolated ablation of Dice alone in these |
| class-query mask decoder | BEVFormer (2203.17270) App. A.3: one learnable query per class, mask from attention maps; BEVFormer-S lane IoU 23.9 vs Lift-Splat* 18.3, jointly trained (§4.4) | encoder differences confounded with the head; 6 cameras |
| class weights | PON: alpha_c = sqrt(inverse frequency), "using inverse frequency directly leads to a tendency to overpredict on small classes"; PanopticBEV: inverse-sqrt frequency + linear taper of the weight within 20 px of an infrequent class "to prevent the infrequent class overflowing into the frequent class, resulting in fuzzy class boundaries"; LSS lane pos_weight 5; BEVSegFormer thin classes 15 | we already use the sqrt family; the taper is an unused detail |
| perspective sensitivity weight | PanopticBEV eq. (9), up-weights far BEV pixels: M6 -> M7 +0.98 PQ, +1.5 RQ | small |
| uncertainty on unobservable cells | PON eq. (2): max-entropy loss (weight 0.001) on cells outside the FoV or fully occluded, CE ignored there | no isolated ablation; prevents confident hallucination where there is no evidence |
| soft vs hard targets, thresholds | SAM-Road (2403.16051): plain BCE, no re-weighting, tuned thresholds vs 0.5 (Table 4 A vs H) | satellite domain; weak transfer |
| Lovasz-Softmax | Berman et al. (1705.08790): surrogate of the IoU; for dataset-mIoU use it after CE pre-training | consistent with FIN's ordering |
| clDice | Shit et al. (2003.07311): soft-skeleton topology loss for tubular structures, alpha in [0, 0.5], gains on Massachusetts Roads vs soft-Dice (Table 1) | aerial/medical domain; HYP for BEV lane lines |

## 5. Training recipes (PUB-P)

| work | epochs / steps | optimiser, lr | input | aug / pretrain |
|---|---|---|---|---|
| LSS | 300k steps | Adam 1e-3, wd 1e-7 | 128x352 resize+crop (§4), EfficientNet-B0 | camera dropout |
| CVT | 30 ep, 4/GPU | AdamW, one-cycle | EfficientNet-B4 | ImageNet backbone |
| Simple-BEV (2206.07959) | converges 1-3 days | lr 5e-4, one-cycle | 112x208 up to 896x1600; **below 448x800 "drastically worsens"; best 49.3 IoU at 672x1200 vs 47.4 at 448x800 (vehicle IoU, Fig. 3)** | **large effective batch matters** (paper's own finding); random crop/reference-camera/camera drop |
| PointBeV | up to 100 ep | Adam 3e-4, one-cycle | 224x480 / 448x800 | sparse point sampling |
| MapTR-tiny / v2 | 24 ep (110 ep long) | AdamW 6e-4, backbone x0.1, cosine, batch 32 | 0.5x / 0.3x of 1600x900 | colour jitter |
| StreamMapNet | 24 ep nuScenes | AdamW 5e-4, batch 32 | 480x800 | first 4 ep single-frame |
| BeMapNet | 30 / 110 ep | AdamW 2e-4, wd 1e-4, backbone x0.1 | EfficientNet-B0 / R50 / SwinT | multi-step schedule |
| PanopticBEV | 20 (KITTI-360) / 30 (nuScenes) ep | SGD 0.005 | 1408x768 / 768x448 | flips + colour |
| SkyEye | 20 + 20 ep | SGD 0.005 / 0.001 | 1408x384 | flips + colour |
| TIIM | 40 ep | Adam 5e-5, x0.99/ep | ResNet-50 FPN | none stated |
| FIN | 24 ep | AdamW 2e-4, lr x0.1 at the 5th-last and last epoch | 6 cam + radar | not stated |
| BEVSegFormer | 120 ep from scratch | AdamW 1e-5 / 1e-4 | 448x800 (front 384x640) | flips, colour |

Pretraining (MapNeXt 2401.07323, single-author arXiv, not peer-reviewed): MapTR-tiny, VoVNet-99, mAP 57.9 (ImageNet cls, frozen BN) / 59.3 (unfrozen) / 62.1 (nuScenes detection) / **63.9 (nuScenes map segmentation)** (Table 5); **nuImages-detection pretraining of R50 HURT map extraction: 50.3 -> 47.3**. The paper's rule: the pretraining task must be close to the map task. Our regime has 4.6k clips against nuScenes' 700 scenes, so the label-efficiency papers (SkyEye 1 %, LetsMap 1 %, Busch 50 %) address a scarcity we do not have.

## 6. Ablated gains, one place (PUB-P)

| lever | result | source |
|---|---|---|
| dense auxiliary supervision on a vector head | none 56.6 -> depth only 59.8 -> +PV seg 60.5 -> +BEV seg 61.0 -> all three 61.5 mAP; PV+BEV without depth 59.2 | MapTRv2 Table 8 |
| roadmap of one model | baseline 50.3 -> +aux depth 55.2 -> +aux BEV 56.5 -> +aux PV 57.1 -> +decoupled self-attn 57.6 -> +one-to-many 61.5 | MapTRv2 Table 3 |
| dense aux vs none (Bezier head) | none 20.0 -> semantic only 34.5 -> instance only 38.4 -> both 43.7 | BeMapNet Table 4 |
| global raster from queries | MapTRv2 61.5 -> +GRL 63.7 -> +GRG 65.0 (preprint under review) | MapGR (2510.06969) Table 3 |
| temporal fusion | HDMapNet IoU 32.9 / 35.8 / 36.4 for 1 / 2 / 4 frames; StreamMapNet 41.7 (single-frame) -> 46.1 with propagation + BEV fusion (+4.4); MapTracker 69.9 (no memory) -> 76.1 with strided memory fusion; Busch temporal loss +0.7 mIoU60 (crosswalk +2.2 from suppressing blind-spot artefacts) | HDMapNet Table IV; StreamMapNet Table 5; MapTracker Table 4; Busch Table I |
| depth supervision | LSS with oracle LiDAR depth: lane 19.58 -> 25.12 (1 scan) / 26.80 (multi-scan), drivable 70.81 -> 74.91 / 76.96 | LSS Table 5 |
| geometry-aware position encoding | IPM-PE +3.9 AP over sine PE, +5.7 over learned PE; shared FC +2.6 | BeMapNet Table 2 |
| image resolution | StreamMapNet 384x384 -> 608x608: 46.1 -> 51.2 mAP; Simple-BEV above | StreamMapNet Table 5 |
| finer feature strides | PanopticBEV strides 4-32 vs 8-64: 21.23 vs 20.55 PQ (+0.68) | PanopticBEV Table IV (M7 vs M8) |
| point-set query design | MapQR 43.3 vs MapTRv2 37.2 at 24 ep, tight protocol | MapQR Table 1 |
| sparse training | PointBeV "similar performance with 20 % of the points"; lane IoU 49.6 vs 44.8 (MatrixVT) | PointBeV Table 4, §4.2 |
| lane-structured dense field | +4.0 mAP (OpenLane-V2) | 2610.25284 Table 2 |
| domain-close backbone pretraining | +1.8 (map-seg vs det), +4.6 vs ImageNet cls | MapNeXt Table 5 |

## 7. Label sources like ours (SAM-/auto-labelled maps) (PUB-P)

* **SkyEye (2302.04233), single front camera, no BEV GT.** Pseudo-labels = front-view semantic labels lifted with a predicted depth, **accumulated over a window**, densified with morphological dilate/erode. Result matches full supervision (mIoU 30.49 vs 30.42; road 71.39, sidewalk 37.62, KITTI-360). Removing **frame accumulation** is the largest single loss: mIoU 30.49 -> 23.26 and sidewalk 37.62 -> 26.75 (Table 3). Our GT already accumulates the whole clip; the open question is whether class decisions are voted across frames.
* **WalkOCC (2606.19122, Jun 2026) uses SAM3 exactly as we do** (front camera, per-prompt masks merged with a deterministic priority overwrite, lifted by projection, then **majority vote inside world voxels (e.g. 0.5 m) to suppress semantic flicker**, App. D.3) and fine-tunes SAM3 on **6,000 hand-annotated images** because base SAM3 does not define curb/gutter/crosswalk (App. E). The SAM 3 paper itself: "struggles to generalize to fine-grained out-of-domain concepts ... adapt[s] quickly when fine-tuned on small quantities of human-annotated data" (2511.16719 Limitations, Tab. 2). WalkOCC's student beats its own pseudo-labels in noise and artefacts (Fig. 4) and the baselines on sidewalk 35.46 -> 38.11, crosswalk 24.37 -> 27.61 (Table 1).
* **Pseudo-labels are worst exactly on thin classes.** PseudoMapTrainer (2508.18788) Table 2: MapTRv2 trained on ground truth 25.4 mean AP (divider 19.5); on pseudo-labels from 2D segmentation + Gaussian-splat road surface: 6.7 (single trip) / 10.5 (multi trip), divider 3.2 / 2.6; with mask-aware assignment and loss 8.2 / 13.2, divider 3.8 / 4.1. The authors attribute the divider gap to differences between pseudo-labels and ground truth (§5.2). Mask-aware handling of unobserved cells gave "a significant gain" (Table 3); **we already ignore not-seen cells, so that part is done.**
* **Busch 2026 (2602.18066)**: BEV prediction differentiably re-projected into the image and supervised by Mask2Former 2D pseudo-labels. Alone it reaches only about 6-7 mIoU (supervised baseline 31.5 / 25.6 / 21.0 at 20 / 40 / 60 m), because pseudo-label policy differs from the GT's; as a pretraining stage plus 50 % of BEV labels it beats the fully supervised baseline: mIoU60 23.5 vs 21.0, lane 26.4 -> 28.1, boundary 25.1 -> 26.3, crosswalk 11.5 -> 16.0 (Table I).
* **Scale-ups**: MapRF (2511.19527) reaches about 75 % of the fully-supervised baseline from 2D image labels only; PseudoMapLabeler (2608.12600) reports +6.1 mAP from teacher-student pseudo-labels in a low-label regime. Both address label scarcity; we are label-rich. Low priority.

## 8. RANKED LEVER TABLE

Ranking = expected effect on the two measured failures (thin classes; planner-relevant area precision) x transferability to ONE front camera x cost. "Published gain" is the primary's own number on its own setup; it is an upper-bound prior, not a forecast (rig seed floor, `H-ESTIM-SEED-1`, applies to any single-seed read of these levers here).

| # | lever | published gain + reference | evidence | transferability to us | cost | note / risk |
|---|---|---|---|---|---|---|
| 1 | **Thin-class target + decoder redesign**: dilated sigmoid heatmap at the 0.25 m lift cell (width about the localisation sigma, 0.5 m) + sub-cell offset and/or signed-distance channels; edge from a drivable-area SDF | BEV-LaneDet Table 5: +0.5 / +0.9 F-score and near x-error -28 % / -48 % from offset at 0.5 / 1 m cells; 0.05 m cell -14.7 F-score vs 0.5 m; Lane Field Table 2 +2.8..+4.0 mAP; GT width precedent 3-5 px (BEVSegFormer §4.2) | PUB-P; no ablation of dilation width found (gap) | HIGH: single front camera, OpenLane, same failure mechanism as our audit's ceiling table | LOW-MED (label transform on the fly from `fine_codes`, +2-3 channels, loss) | the only lever that attacks the structural cause; testable as a head-only fine-tune |
| 2 | **Loss set on the thin group**: per-class sigmoid focal + Dice (+ Lovasz, normalised boundary), per-class best-IoU threshold; softmax CE kept on area classes | FIN Table 4: +2.8 (text) .. +7.2 (table) mIoU; head +1.2; BEVFusion/CVT convention | PUB-P (camera+radar; text/table conflict) | MED-HIGH | LOW | Dice/Lovasz move the operating point toward the minority class without an argmax contest; they do not fix localisation by themselves (audit ceiling) |
| 3 | **Unobservable / blind-zone handling**: explicit unknown channel or max-entropy loss on lift-invalid cells; check the 0-5 m band's lift-valid share | PON eq. (2); Busch: blind-spot crosswalk artefacts, temporal loss +2.2 crosswalk IoU | PUB-P; gain on sidewalk precision UNMEASURED | HIGH: a front camera does not see the ground in front of the bumper | LOW | explains the measured 25 % sidewalk precision at 0-5 m if those cells are unsupervised; diagnostic first (what fraction of 0-5 m is lift-invalid, what does the planner read there) |
| 4 | **Label-ceiling audit** (CPU): label sigma and thin-class precision/recall of the SAM3 world map against hand-verified frames; check cross-frame class voting | PseudoMapTrainer Table 2 (divider 4.1 vs 19.5 AP); SkyEye Table 3 (accumulation -7.2 mIoU); WalkOCC D.3/E | PUB-P | HIGH: same source family as ours | LOW (CPU, about 100-200 frames) | decides whether ANY head change can pass the audit's ceiling; run first |
| 5 | **Image-plane auxiliary supervision** (PV seg on SAM3 2D masks; depth if LiDAR depth is renderable) | MapTRv2 Table 8: +4.9 mAP all three; PV+BEV without depth +2.6; BeMapNet Table 4 | PUB-P (vector heads, 6 cam) | MED-HIGH: lines are 3 px at 20 m in the image and the SAM3 masks live in the image | MED (needs the 2D masks on disk; if absent, re-projecting BEV codes into the image loses the point) | check first that SAM3's per-frame masks were kept |
| 6 | **Temporal BEV fusion** (2-4 strided past frames, ego-motion-warped) | HDMapNet +3.5 IoU (4 frames); StreamMapNet +4.4 mAP; MapTracker +6.2 mAP | PUB-P, three independent papers | HIGH in principle (near-field blind zone, far-field refinement) | HIGH (trainer, memory, eval) | needs PI ruling: relative ego-motion over the last k frames vs the vision-only rule (v0/yaw-rate at t0 is already admissible; warp over k frames is the same sensor class, not the same ruling) |
| 7 | **Range-banded resolution** (10 cm to about 10 m, 0.25 m to the far end, 0.5 m for planner/far scoring) | none published; follows from EST §2 + BEV-LaneDet cell-size + range decay (§3.3) | HYP | HIGH | LOW-MED | EST arithmetic: today's 60 m x +-16 m = 100x320 + 200x128 = 57,600 cells vs 192,000 (3.3x fewer); A7 100 m x +-30 m = 100x600 + 120x240 + 120x120 = 103,200 vs 600,000 (5.8x fewer); keeps the maximal range the PI asked for |
| 8 | **Re-score under the published protocol** (GT dilated to about 0.75 m, per-class best threshold, Chamfer CDP/CDL) | HDMapNet metrics; BEVFusion best-IoU | PUB-P | HIGH | LOW | prerequisite for any claim of parity; not a model change |
| 9 | Vector (MapTR-style) lane/edge head in parallel | MapTRv2 divider 62.4 / boundary 62.4 AP at 0.5-1.5 m tolerance; 35.4 / 36.3 at 0.2-1.0 m | PUB-P | LOW-MED: 6-camera only, polyline GT must be extracted from the SAM3 raster, planner consumes raster | HIGH | keep as a shadow arm; strongest if the planner moves to lane-level inputs |
| 10 | Input resolution / finer-stride tap | StreamMapNet +5.1 mAP (384 -> 608); Simple-BEV +1.9 IoU (448x800 -> 672x1200); PanopticBEV +0.68 PQ (stride 4) | PUB-P | LOW: 416x1024 is already above the 448x800 knee | HIGH | not first |
| 11 | Domain-close trunk pretraining | MapNeXt Table 5 +1.8..+4.6 mAP; nuImages-det pretraining -3.0 | PUB-P (single author) | LOW: map-only fine-tune was flat | MED | the trunk is shared with the planner |
| 12 | Geometry-aware positional encoding | BeMapNet Table 2 +3.9..+5.7 AP | PUB-P | LOW: the lift is already camera-geometric | LOW | |
| 13 | Class-weight taper / sensitivity weighting | PanopticBEV +0.98 PQ | PUB-P | LOW | LOW | sqrt family already in place |
| 14 | Head capacity | FIN +1.2 mIoU; AugMapNet saturates at 2 layers | PUB-P | LOW | LOW | not the bottleneck |
| 15 | Semi-/self-supervised scale-up | PseudoMapLabeler +6.1 mAP (low-label); MapRF about 75 % from 2D labels | PUB-P | LOW: we are label-rich | MED-HIGH | |

**Excluded as non-transferable:** radar/LiDAR fusion (FIN, RESAR-BEV, HDMapNet fusion), SD-map / satellite priors (P-MapNet class; Lane Field's SD variant) unless the PI rules a map prior admissible, surround-view stitching tricks.

## 9. Recommended map representation, size, head and recipe for TanitAD

### 9.1 What to change about the 600x320 @ 10 cm raster fed by a 0.25 m lift

| today | change | why (evidence) |
|---|---|---|
| 600x320 logits at 10 cm everywhere, hard labels, 8-way softmax | **primary output on the lift's native 0.25 m grid (240x128 at 60 m x +-32 m); 10 cm only for 0-10 m (the existing NEW-2 near-lift patch)** | stride-8 cell is 0.16 m at 10 m and 0.33 m at 20 m (EST §2); BEV-LaneDet 0.05 m cell -14.7 F-score vs 0.5 m; our audit ceiling |
| lane / edge / arrow / hatched as 1-2 px hard classes in one softmax | **thin group as separate sigmoid heatmaps on targets dilated to about the localisation sigma (0.5 m wide), plus 2 offset channels (dx, dy in +-0.125 m) and an unsigned distance-to-line channel clipped at 1 m** | HDMapNet/BEVSegFormer 3-5 px lines; BEV-LaneDet offset Table 5; Lane Field proximity channel; audit §2c |
| "non-drivable edge" as a class (0.58 % of cells) | **derive from a signed-distance channel of the drivable area (clipped +-2 m); keep a dilated edge heatmap only as a secondary** | edge is by construction the boundary of an area that is predicted at about 0.9 IoU; boundary-loss literature (Kervadec; FIN eq. 12) puts the signal on the distance to the contour, not on a 1-px class. HYP for our head, building blocks PUB-P |
| CE on all 8 classes with sqrt_mf weights | **softmax CE (+Dice) on the area group {drivable, sidewalk, no-class, hatched, crosswalk}; sigmoid focal + Dice (+ Lovasz, normalised boundary as later arms) on the thin group {lane, arrow/text, edge}** | FIN Table 4; BEVFusion overlap argument (a lane line lies on drivable surface) |
| decision rule `prior_corrected` / `class_threshold` | keep `class_threshold`, **fit thresholds on a held-out fit split, per class and per range band** | BEVFusion/CVT best-threshold protocol; SAM-Road threshold study |
| not-seen cells ignored | keep, **plus an explicit unknown / max-entropy treatment for lift-invalid cells so the planner can read "no evidence"** | PON eq. (2); Busch blind-spot artefacts |
| single-frame lift of the current frame | see 9.5 | HDMapNet/StreamMapNet/MapTracker |

### 9.2 Size (extent)
Keep the maximal extent (A7: 100 m x +-30 m) **but supervise and score it in range bands**: 0-10 m at 10 cm, 10-40 m at 0.25 m, 40-100 m at 0.5 m. Arithmetic (EST): on today's 60 m x +-16 m, 100x320 (0-10 m) + 200x128 (10-60 m) = 57,600 cells vs 192,000; on A7 (100 m x +-30 m), 100x600 + 120x240 (10-40 m) + 120x120 (40-100 m) = 103,200 cells vs 600,000 at uniform 10 cm. Published accuracy falls 25-32 % relative from 60 to 100 m (StreamMapNet, AugMapNet) with six cameras; one forward camera has less far-field information, so far-field logits should not be allowed to dominate the loss (range-normalised weights, PanopticBEV's "sensitivity" idea goes the other way and is not recommended here).

### 9.3 Head
* Shared 0.25 m BEV encoder as today (the lift is already camera-geometric; BeMapNet's IPM-PE gain is therefore not expected).
* Two light branches (about 1-2 M params, FIN-sized): area branch (softmax) and thin branch (sigmoid + offset/SDF channels). No capacity increase.
* Auxiliary image-plane head on the stride-8 feature map: PV segmentation on the original SAM3 masks (and sparse-LiDAR depth if renderable), training-only and removable.

### 9.4 Recipe
Warm-start from the converged refcv8 trunk and fine-tune the map head only (the authorised map-only budget is enough for a head-only test), AdamW with a short one-cycle/cosine decay, loss = area softmax CE (sqrt_mf weights kept) + thin-group focal + Dice, loss weights set so initial magnitudes are comparable (FIN practice). Horizontal flip is NOT recommended without mirrored right-hand-traffic semantics (sidewalk side). Every arm is paired with a replicate seed (`H-ESTIM-SEED-1`).

### 9.5 Temporal fusion (the best-evidenced architecture lever)
Three independent papers (+3.5 IoU, +4.4 mAP, +6.2 mAP) and a front-only camera that must see the near field from earlier frames. Admissibility is the open question: the programme admits v0, acceleration and yaw-rate at t0 (IMU-measurable); relative ego-motion over the last 2-4 strided frames is derived from the same signals integrated over a short window, but that is a new ruling and this document does not assume it. If refused, the fallback is Busch-style temporal consistency as a training loss only (+0.7 mIoU60).

### 9.6 Proposed experiment ladder (cheapest first; pre-registration belongs to the Master Mind)
| step | what | cost | decision rule to commit in advance |
|---|---|---|---|
| E0 | label audit: thin-class precision/recall of the SAM3 world map at 0.2 / 0.5 m tolerance on 100-200 hand-checked frames; check cross-frame voting; check whether per-frame SAM3 masks exist | CPU, hours | if label precision at 0.5 m tolerance is below 0.7 for lane or edge, the label is the ceiling and head work stops at that ceiling |
| E1 | re-score the banked refcv8 map dumps under the published protocol (GT dilated to 0.75 m, per-class best threshold, Chamfer CDP/CDL, bands 0-10 / 10-20 / 20-40 / 40-60 m) and report 0-5 m lift-valid share | CPU | defines the baseline every later arm is compared to |
| E2 | head-only fine-tune, arms: control (same flags) + replicate; thin-group redesign (9.1); + PV aux | map-only budget | separated vs control AND exceeds the replicate spread on tolerance IoU of lane and edge, with no area-class regression beyond the replicate spread |
| E3 | loss-set arms on the thin group (Lovasz, normalised boundary) on the E2 winner | map-only budget | same rule |
| E4 | temporal memory (needs the ruling) | trunk-level | only if E2 passes and the 0-5 m / far-band deficit remains |

## 10. Gaps, conflicts and things not verified
* **Conflict:** the brief says median-frequency weights; the source (`compute_map_class_weights.py`, A8) registers sqrt(median-frequency), clip 25. Resolve from the running config's weights JSON.
* **Gap:** no single-front-camera vector map extraction result; no ablation of thin-class dilation width; no isolated loss ablation inside BEVFusion; no SAM3-label raster result on driving thin classes (WalkOCC is sidewalk-robot, occupancy).
* **FIN Table 4 vs text** disagree on the focal-only baseline (46.3 vs 49.6).
* **Not read to table level:** PivotNet (numbers only as reported in MapQR Table 1), VectorMapNet (as reported in MapTR Table 1), LetsMap (abstract and its 1 %-label claim), RESAR-BEV and Driver2Map (abstract), GeMap, VMA, 2409.00593, ViT-BEVSeg, BEVFusion-Liang (downloaded, not relied on; Appendix A.2).
* **INH numbers** (25 % sidewalk precision at 0-5 m, corridor precision 0.72-0.74, edge 0.000 argmax / lane 0.164 per-class thresholds at refcv7-50,400, incursions 17.9 % -> 1.7 % with the true map) come from the PI's brief; the audit ceilings come from `Research/2026-09-26-map-signal-audit/RESULT.md`. None re-verified.
* **Protocol warnings:** published thin-class IoUs are on dilated GT; published vector AP is tolerance-based; the nuScenes original split leaks by location (StreamMapNet). Do not compare our pixel-exact 10 cm IoU to them.
* **Geographic leakage on our side is unchecked:** confirm the map head's val clips are not revisits of train locations before reading any flat curve as convergence.

---

## Appendix A: banking manifest (kb_add.py could not run: it writes to G:)

Files: `F:/c_overflow_2026-10-26/tmp/lit_map/<id>/<id>.pdf` (+ `<id>.txt`). All PDFs begin `%PDF-`, end `%%EOF`, open with PyMuPDF at the page count shown. To bank on the dev box (arXiv re-download; compare the sha256 below; arXiv may re-render a PDF, so a mismatch is a re-render, not corruption, if page count and size agree):

```bash
cd "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
for id in 2008.05711 2205.02833 2203.17270 2205.13542 2206.07959 2312.00703 2003.13402 2110.00966 \
          2108.03227 2302.04233 2405.18852 2203.04050 2602.18066 2510.00651 2107.06307 2206.08920 \
          2208.14437 2308.05736 2308.12570 2403.15951 2306.09700 2308.16477 2402.17430 2503.13430 \
          2510.06969 2610.25284 2401.07323 2210.06006 2606.19122 2508.18788 2608.12600 2511.19527 \
          2511.16719 2403.16051 1708.02002 1705.08790 2003.07311 1812.07032; do
  python tools/kb_add.py "$id" --tag bev-map --cited-by "TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/LIT_BEVMAP.md" --note "see LIT_BEVMAP.md"
done
python tools/kb_add.py --verify
```

### A.1 Cited and read (38)

| id | short name | bytes | pages | sha256 |
|---|---|---|---|---|
| 2008.05711 | Lift, Splat, Shoot | 5319285 | 17 | b853e73904c1a61fdbfd843190dcff6c5b3a594ee947fab4939a68ff1617cd67 |
| 2205.02833 | Cross-view Transformers (CVT) | 5452825 | 10 | a600a4a50703b89a1ee4d44cc4b5b23dff46720db25b4b024b1b7bc94deff118 |
| 2203.17270 | BEVFormer | 4739748 | 20 | 641561d0e4c1d46d6883a3d38c68b475030de5bbbcdf747af268366f47f97a6b |
| 2205.13542 | BEVFusion (MIT) | 10244020 | 8 | 8b2d5d3d07fdafa38119e6876487226ceeaac1e02e664f4c4f3f2d77206b428f |
| 2206.07959 | Simple-BEV | 2899899 | 7 | 81a635343ff997a74190fc19a808edf4e46877c6d7be89226ccdc646409b3b20 |
| 2312.00703 | PointBeV | 3541351 | 16 | 215ea809f72ce71a97dd475a6956b6c72990aa430ef726feac5e6636ae41cfcb |
| 2003.13402 | Pyramid Occupancy Networks | 3706465 | 11 | 39747a82eb0290284c9baa2f8f5511d618cae963f6555d26900fcfe888e0fd53 |
| 2110.00966 | Translating Images into Maps | 1029536 | 7 | 23386ec2539387c78fd97c2e19c0723d280ccb7110fc7601f61a4c644d40f576 |
| 2108.03227 | PanopticBEV | 32824880 | 17 | 4d8f08a97e0380c371118eabf74e774b53e0a4738f1e36bcadd32ab19da540be |
| 2302.04233 | SkyEye | 17719319 | 14 | 2bd0c77a5669c0d1b0dd2192c94cc3831d851acdbf57565d67694bd5f5a2c875 |
| 2405.18852 | LetsMap (abstract-level) | 20060183 | 23 | c0226475ef937653fee9e15b2cc83819828bf73fbbc87e7e06de981b9253c173 |
| 2203.04050 | BEVSegFormer | 6066963 | 9 | 70437388b341399d99c10f260f15184309a25784e5837b9eaa2a330e4584c62e |
| 2602.18066 | Busch: Faster Training, Fewer Labels | 4064041 | 8 | c7632a6e542ffd4de4f5d035fac1030a66a00b764a3ba86ef0a1c50fbdfa185a |
| 2510.00651 | FIN | 6883014 | 14 | a6b710583015e33d5c895919117ca12203ddf27be3d9dff45a00f5e2bfe5699d |
| 2107.06307 | HDMapNet (7-page ICRA version) | 13630989 | 7 | 5f521e3749ec7b193bf6216a9d931b999a1307f0d848e17dc9a121c85588d2cb |
| 2206.08920 | VectorMapNet (via MapTR Table 1) | 13375900 | 18 | 7b7e5cca0a2103079cd404bfeb6c77a16431006cd370dd373339c097b5c7e1fe |
| 2208.14437 | MapTR | 3261101 | 18 | 919e4ce3de7a8046867086730ed7bc12aeb55550ae9a4d8242f742dd2b93c1b9 |
| 2308.05736 | MapTRv2 | 6901250 | 26 | fda81a85db6c7e534d7a34d4c268d6a970aeaadb9e8a4637210e9148b08c0f8a |
| 2308.12570 | StreamMapNet | 7478587 | 15 | 59d59b9bbcf577a57158e35784e617f5ec651d3894a17b4fc9524da849460196 |
| 2403.15951 | MapTracker | 16585597 | 33 | f195c01d7d2e79ef34ab72dee479893519b4746f8320aa70d9b6120f2b0624f7 |
| 2306.09700 | BeMapNet | 15702655 | 19 | 84fe91d0a261daae8a1647b8c77fdc84d50c639d58e2a40e86fe8c05d3f95c8f |
| 2308.16477 | PivotNet (via MapQR Table 1) | 6777456 | 17 | cc11c931c379ab8ca0b855ba18a411673a52cf6219a5dbf92b59361f4052c218 |
| 2402.17430 | MapQR | 11504596 | 23 | ec04eeeabc5d1fb2e9d533617f870d995142020ff332da4d4a6b81038bb5562e |
| 2503.13430 | AugMapNet | 9767823 | 18 | f2ab3623fb5a92c82a6109b6304b7ca2da29cfed1a18048ffdce983afd8451a0 |
| 2510.06969 | MapGR (under review) | 3773660 | 16 | 7b6395b13e702bf22355cdac49c354567ba4da8d6caa3c7a6492bb197fdb182b |
| 2610.25284 | Lane Field | 48848621 | 24 | 2f8851a2b28482f3c1a62fcce54233f6b65c3d2a9502bffd4d60834aea785425 |
| 2401.07323 | MapNeXt (single author) | 206756 | 13 | 746d2b03f70a4e58a5556e167bb69b5d804a7e8dc2b4b3fe32751fb125ff90d3 |
| 2210.06006 | BEV-LaneDet | 36349746 | 12 | 531e155d6959bcecb41562cf197cad207f523cbfd5974e5eed5cec7a6420d39b |
| 2606.19122 | WalkOCC (SAM3 pseudo-labels) | 36297645 | 26 | 2c7a03ffabb2dda7fa46698401487386581aff3bad4c636e6e633db434cb3502 |
| 2508.18788 | PseudoMapTrainer | 18245050 | 13 | 1c8ac7224cf4a942d3abc364065b11e6ed1ecf76df0bd9df2c3ad7111f90811a |
| 2608.12600 | PseudoMapLabeler | 1679159 | 17 | 1f9f02e06a7781189268486700ea48f17bc33f5872cbe8d4366ccce7b84e5a2d |
| 2511.19527 | MapRF | 2326745 | 8 | 5b983eb930163895a7f3b9afb06291e00e4caaa34a16705cf000eab7286e1985 |
| 2511.16719 | SAM 3 (limitations only) | 35521939 | 78 | 19e8a44e7fc33bc6cf237f6b4ac3948a79b33f7ffc62d09012fcac6fc9fb016f |
| 2403.16051 | SAM-Road | 19462163 | 11 | f0357f16ed477a1d47924035ffb51cdd1554a00c7adcab076fe821575431ea01 |
| 1708.02002 | Focal Loss | 1297358 | 10 | 2e6f45b97007009f32b82a3c19f35638d8da7022d8d4e4416b9078de91339f71 |
| 1705.08790 | Lovasz-Softmax | 4169265 | 14 | 82bdd01cacbab500abab94c78d6c27e2c1f2a7fafee16c9b302f8d1225c38b1e |
| 2003.07311 | clDice | 6744504 | 17 | 2e952692e01cb534b8851712e27846ab5d0dc06dfd170c9c64daf4e70d7f3170 |
| 1812.07032 | Boundary loss | 4157319 | 21 | 9746e5741c759ba10e05d9b18245f4d7f580f58ae8eb4812bfc73992dde1bf40 |

### A.2 Downloaded, NOT relied on (no claim above depends on them)

| id | short name | bytes | pages | sha256 |
|---|---|---|---|---|
| 2205.13790 | BEVFusion (Liang et al.) | 2542435 | 17 | b94f1b1c333afc20d8291cb9d122cc561bea248502b52ea1da2b4bd1edc08440 |
| 2205.15667 | ViT-BEVSeg | 3540857 | 7 | 1133f01636be791d7666b6e96ec5cb93415f25d9fe0018557bd232d2a1b71aac |
| 2304.09807 | VMA | 19790291 | 12 | 527672c5fa07d30d2499597aed1974432d37e675d96dd3aea87f6a4952b6fc63 |
| 2409.00593 | Online Temporal Fusion for Vectorized Maps | 7478743 | 8 | 9ccf45d00bd86e8f88dc03fd270b5d3c9cda4a30125ddfab0e4c49d6071d7da5 |
| 2505.06515 | RESAR-BEV | 4268054 | 12 | b728ecd6c8f7e119b22cc2409865099f2bbb84311536d3d972ccde8906cffa27 |
| 2608.01338 | Driver2Map | 3330049 | 13 | 714581ec301f4ea5f0c228e0e2cdec6f3acafa4eebb1ecabee250357e15a0d3a |
| 2312.03341 | GeMap | 11238229 | 24 | bca3a5e580d9636b083009762f232b4b164858551ae2d4a7aef8fea957efc63a |

## Appendix B: deliverable manifest

| artifact | where it lives | single copy? |
|---|---|---|
| `LIT_BEVMAP.md` | hold `C:/Users/Admin/qland/hold_lit_map/code/TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/LIT_BEVMAP.md` | yes, until landed |
| `expect.tsv`, `msg.txt` | `C:/Users/Admin/qland/hold_lit_map/` | yes |
| 45 PDFs + extracted text | `F:/c_overflow_2026-10-26/tmp/lit_map/<id>/` | yes, until banked (App. A loop) |
| download / extract helper scripts | `C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/` (`fetch.py`, `ext.py`, `manifest.py`) | scratch; not needed once banked |
