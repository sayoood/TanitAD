# RESULT: refcv6's box and agent heads against proven detection heads, and what refcv7's box head should change

Architecture & Inference, literature pass, 2026-09-27 (Berlin). Asked by the Master Mind after the PI, verbatim,
2026-09-27: *"Can you compare our box heads in tits architecture, wiring, training, size, training signal flow to
simialr sucessfull heads in proven research works..."*

**Scope.** The LITERATURE side plus a STATIC read of our code at tip `56ae4eb`
(`agent/arch-inf-20260803`), plus arithmetic on the refcv6 run's own `config.json`, `metrics.jsonl` and checkpoint
tensor shapes. The box-head AUDIT (`../2026-09-26-box-head-audit/`) owns every model measurement; it had landed only
partial raw files when this was written, and those are cited as the audit's, not re-measured.

**Evidence classes.** MEASURED (ours, with the artifact path) · STATIC (tip `file:line`) · ANALYTIC (a script in
`raw/`) · PUBLISHED `[LIB:<id>]` (banked) · PUBLISHED (arXiv:<id>, read online, NOT BANKED) · PUBLISHED-CODE (an
official repo file) · INFERRED · UNVERIFIED. Source keys `S1`-`S30` point into `raw/literature_factsheet.md`, which
carries every URL read, where each fact sits, and whether it was read once or twice.

---

## 0. Headline

**Our heads are DETR-shaped and DETR-sized, and they depart from every proven query-based detector on the same
five axes at once.** Size is NOT the problem: our heads have 3.82 M (agent) and 4.09 M (box3d) parameters, against
2.7-11.3 M for the published heads (ANALYTIC, `raw/head_param_counts.json`; our two totals reproduce the checkpoint
exactly). The departures are:

1. **Presence loss and decision rule.** Ours is BCE with the negatives down-weighted to 0.1 and averaged over all
   100 slots, read at a fixed 0.5 gate. Under that loss the gate fires at a match belief of only **π = 0.091**. All
   five camera-3-D heads whose configs were read (DETR3D, PETR, BEVFormer, StreamPETR, UniAD; the same values in
   all five) use a sigmoid **focal** loss (α 0.25, γ 2) with weight 2.0. It is normalised by the number of GT boxes: that is the
   mmdet DETR-head convention, read in DETR3D's and PETR's head code. Under that loss the same 0.5 gate needs
   **π = 0.75** (ANALYTIC, `raw/presence_optimum_by_loss.json`). None of them reads a fixed gate for its headline
   metric: they rank their top-300 boxes.
2. **No deep supervision, no refinement.** Our 3-layer decoder is supervised at its LAST layer only. The one
   controlled ablation at a 3-layer decoder costs **−11.5 AP** without the per-layer loss (39.8 → 28.3, S27).
3. **Query budget in crowds.** 100 queries face 80.3 targets per window (max 120) on the crowded clip. DETR's own
   appendix: a 100-query DETR finds every instance only up to about 50 (S1). The nuScenes heads run 900 queries
   (S5).
4. **Supervision set.** Our filter is horizontal field + decode box only. Everything it removes becomes a
   NEGATIVE for the presence loss, including visible objects just outside the box. The audit's partial z-buffer
   finds **46.2 %** of targets on the five validation clips with zero visible pixels. That is a LOWER bound on
   hidden, because walls, trees and poles are not in the z-buffer. The monocular and
   pedestrian literature trains with IGNORE regions and pixel-size, occlusion or depth cuts (S13, S18, S20, S22).
   The nuScenes and Waymo LiDAR-point rule would NOT catch our hidden targets: on the audit's LiDAR clip, 85.6 % of
   the under-10 %-visible targets still carry 3 or more LiDAR points.
5. **No geometry in the image-token positions.** Ours is a learned per-token table (548,864 parameters, 14 % of the
   decoder). The corpus has per-clip mount heights of 1.20-1.69 m. PETR's 3-D position embedding vs 2-D PE: NDS
   0.208 → 0.356 (S6, multi-view).

Around those five, the TRAINING differs as well:
- 0.81 epoch from ImageNet init, jointly with three other heads.
- The two detection losses are about **47-56 %** of the joint loss (the sum of the two medians per band; the agent
  part is a lower bound). The presence term is **0.9-1.2 %** of the box loss (MEASURED,
  `raw/refcv6_box_loss_share.json`).
- Not one detection metric was logged in 38,000 steps (MEASURED: 224 metric keys, 0 detection metrics).

### Escalations (integration and decisions needed; none of this is wired)

| # | what | who |
|---|---|---|
| E1 | **`SPEC_REFCV7.md` registers box3d as UNCHANGED** (A1 §6.2 item 2; A6 changes only its BEV input). Every recommendation below that touches the head needs an amendment, with G-DVB and G-LIVE entries, before the refcv7 launch. | Master Mind, then PI |
| E2 | **`PREREG_REFCV7.md` has no detection metric or bar**, and refcv6 logged none (MEASURED). Pre-registering them is required before ANY refcv7 box number exists (§5). | Master Mind |
| E3 | **M17 (`N_QUERIES_DEFAULT = 100`, zero-drop rule) is violated on the refcv6 corpus line.** It was ruled on the PARITY train join (max 94). The B1 eval clip reaches 120 targets per window, and 146 targets had no slot (video agent). It needs a re-ruling after the visibility rule lands (§4, R4). | Master Mind |
| E4 | **Bank 21 arXiv ids** (§8). | PI permission |
| E5 | **Timing.** R1, R2 and P0 are cheap enough for the refcv7 launch. R3 needs the audit's visibility rule plus a join rebuild. R5-R7 are ladder work: `TanitAD_ValidateAIDesign`, v7-tiny. Whether refcv7 waits for any of them is the PI's call. | PI |

---

## 1. Our two heads, as built and as trained

STATIC at tip `56ae4eb`; MEASURED from `D:/refcv6_eval_kit/ckpt_final/{config.json, metrics.jsonl,
ckpt_step38000.pt}` (refcv6-r101-s0, step 38,000). `Project Steering/MODEL_REGISTRY.md` carries the run's status
(STOPPED at 38,250; ckpt 38,000 md5 `5a2e7222…`) but no box-head row. Nothing here conflicts with it.

| | **box3d head** (perception branch) | **agent head** (planner seam) |
|---|---|---|
| module | `Box3DSlotDecoder(AgentSlotDecoder)` + `Box3DMemory` (`box3d_head.py:132-284`) | `AgentSlotDecoder` (`agent_slots.py:546-684`) via `build_agent_head` (`refc_agents.py:238-255`, built at `refc.py:3669`) |
| memory (keys/values) | 1,664 image tokens: stride-16 `fmap_s16`, 1024 ch, 26 × 64 → `Linear` 256 · plus 480 BEV tokens: 96-ch BEV features avg-pooled 120 × 64 → 30 × 16, i.e. **2 m cells** (`box3d_head.py:181-221`; config `fmap_s16_hw [26, 64]`) | 416 tokens: the stride-32 trunk map `fmap`, 2048 ch, 13 × 32 (`refc.py:4381-4382`; ckpt `mem_proj [256, 2048]`, `mem_pos [1, 416, 256]`) |
| positions | a learned per-token table `mem_pos` (**2,144 × 256 = 548,864 params**) + 2 learned source embeddings; no camera geometry (`agent_slots.py:602-605`, `box3d_head.py:186-187`) | a learned per-token table (416 × 256) |
| queries | **100** learned vectors, trunc-normal init, no reference points (`agent_slots.py:606-608`; `N_QUERIES_DEFAULT = 100`, `:212`) | 100 |
| decoder | `nn.TransformerDecoder`, **3 layers**, d 256, 8 heads, FFN 1024, GELU, pre-norm, dropout 0 (`:609-612`); dense self- and cross-attention | same |
| output | ONE `Linear` on the LAST layer (`:653`): presence, 10 class logits, cx, cy, l, w, yaw (sin, cos), 3 rates, occ, + cz, h | same, without cz, h |
| params | **4,094,743** (decoder 3,806,999 + memory 287,744; config `branch_params`) | **3,822,869** (ckpt) |
| presence | sigmoid logit, init prior **0.05** (`:573`, `:615-617`) | same |
| presence loss | BCE over ALL slots: matched → 1 (weight 1.0), unmatched → 0 (weight **`NO_OBJECT_W = 0.1`**), `reduction="mean"` over B·100 (`:233`, `:864-873`); term weight 1.0 | same |
| class loss | softmax CE on matched slots with **inverse-frequency weights**: automobile 0.004277, person 0.011534 … animal 5.553512; imbalance 1,298.6 : 1 (config `seams.agent_cls_weight`); normalised by Σw (`:897-918`) | same |
| box loss | per matched target: centre L1 (m) 1.0 · size L1 (m) 1.0 · yaw 1 − cos 1.0 · rates L1 0.5 · occ BCE 0.5 (`:226-229`) · + z, h L1 (m) 1.0 each (`box3d_head.py:126`) | same, minus z and h |
| matching | exact Hungarian (numpy, CPU) on 1.0·(−σ(presence)) + 1.0·(−p[class]) + 1.0·L1 centre (m) + 0.5·L1 size (m), BEV plane only (`agent_slots.py:216-221`, `:767-785`; `MATCH_INCLUDES_Z = False`, `box3d_head.py:114`). If targets > queries, the FARTHEST are dropped and counted (`:809-812`). | same |
| deep supervision / refinement / denoising | **none / none / none** | none / none / none |
| GT | `obstacle.offline` LiDAR cuboids → `build_obstacle_join.py` → `targets_from_join` → `visible_target_filter`: azimuth ≤ 60°, 0 ≤ x ≤ 60 m, abs(y) ≤ 16 m (`refc_agents.py:336-433`), applied BEFORE matching (`box3d_head.py:376-411`). Removed boxes are DELETED, so a slot firing on one is trained as a negative. No visibility, occlusion, min-size, min-point or per-class range rule. | same filter (`agent_losses`, `refc_agents.py:849-850`) |
| gradient path | box3d loss → decoder → memory → `fmap_s16` (trunk, attached) and → BEV encoder → lift → trunk (`refcv6_perception_branch.py:38-43, 441-464`) | agent loss → head → trunk `fmap` (attached). ALSO the planner and tactical losses reach the head through `AgentTokenEmbed`, which scales every token by σ(presence) (`refc_agents.py:317-334`; soft gate, `presence_gate 0.5`, `presence_hard False`) |
| weight in the loss | `--w-box3d 1.0` | `--w-agent 1.0` |

**Training (MEASURED, config + metrics).**
- Schedule: b16 × 38,000 steps = 608,000 window presentations = **0.81 epoch** of 746,946 train windows (719,739
  labelled). The planned run was 50,400 steps.
- Optimiser: lr 1e-4 peak, encoder ×0.5, warmup 2,000, cosine; weight decay 1e-4; ImageNet-init `resnet101.a1_in1k`
  with frozen BN.
- Joint with the planner, the tactical decoder v6 and the map head.

**Loss shares (MEASURED, `raw/refcv6_box_loss_share.json`, medians per step band).**

| band (steps) | box3d / total | agent / total (lower bound) | presence / box3d | regression / box3d | class / box3d |
|---|---|---|---|---|---|
| 0-1k | 0.283 | 0.190 | 0.009 | 0.728 | 0.184 |
| 15-25k | 0.325 | 0.234 | 0.011 | 0.739 | 0.171 |
| 34.5-38.3k | 0.271 | 0.205 | 0.012 | 0.724 | 0.164 |

- The agent share is a lower bound: its rates and occ terms are in its total but never logged.
- Visible targets per labelled training window: median **4.75-5.53** per band. Before the filter: 30.7-33.4.
- 0 query drops in any of the 765 logged training rows.
- The metrics carry 224 keys and **0 detection metrics**: no AP, precision, recall, AUROC or confident-slot count.
  The in-run eval logs only the loss terms. `box3d_ap` exists in code (`box3d_head.py:457-553`) but nothing wires it
  into the run.

**Symptoms this report tries to explain** (all MEASURED by others):
- **S-conf.** 98.3-98.7 of 100 box3d slots (99.2-99.6 agent) confident at σ ≥ 0.5 on the two crowded clips, 71.1
  (80.9) on the third. At the gate, matched and unmatched slots fire alike (TPR 0.98-1.00 vs FPR 0.95-0.98 on the
  crowded clips). Sources: `…/2026-09-26-refcv6-boxes-video/RESULT.md`; `…/2026-09-26-map-signal-audit/raw/box_presence_partial.md`.
- **S-prec.** Precision 0.16-0.38 and recall 0.47-0.82 at the gate.
- **S-rank.** AP@2 m 0.134-0.360 (box3d) on those three crowded clips.
- **S-loc.** Hungarian centre error, mean 1.15-4.12 m (median 0.96-2.80 m).
- **S-crowd.** 80.3 targets per window, max 120, on clip `0191487845ef`, where 146 targets had no slot.
- **S-gt.** 69.1 % of GT removed by the filter (GT validation). **46.2 % of the remaining targets have zero visible
  pixels** in the audit's GT-only z-buffer (`…/2026-09-26-box-head-audit/raw/visibility/vis_gtval.json`, partial).
  Visibility there is an upper bound, so this hidden share is a lower bound.
- **S-flow.** The presence share (1 %) and the detection share (about 47-56 %) above; no detection metric logged.
- **S-agent.** The agent head is more saturated than box3d on every clip.

---

## 2. The comparison tables

Rows are dimensions; columns are works. "=" means "as the column to its left". Loss weights are the official
configs' values (PUBLISHED-CODE) unless marked otherwise.

### 2a. Query-based 2-D and monocular detectors

| dimension | **OURS** (box3d / agent) | **DETR** S1 [LIB:2005.12872] | **Deformable DETR** S2 | **DN-DETR / DINO** S3, S4 | **MonoDETR** S13 |
|---|---|---|---|---|---|
| queries vs objects | 100 vs train mean ~5, crowd 80.3, max 120 | 100 vs COCO mean 7.7 (S30); finds all only up to ~50 instances, ~30 at 100 (App. A.5) | 300 | DINO 900 | 50 (KITTI, Car config) |
| decoder depth / d / FFN | 3 / 256 / 1024 | 6 / 256 / 2048 | 6 / 256 / 1024 | 6 / 256 | 3 / 256 / 256 |
| attention | dense | dense | deformable: 8 heads × 4 levels × 4 points around reference points | DN-DETR: on DAB-DETR (dense) and Deformable variants; DINO: deformable | depth cross-attention + visual cross-attention (deformable: UNVERIFIED) |
| feature access | stride 16 (26 × 64) + 2 m BEV (box3d); stride 32 (agent) | single scale (C5 or DC5) | 4 levels | DINO 4 levels | image features + a supervised foreground depth map (number of scales UNVERIFIED) |
| positions | learned per-token table | 2-D sine at every attention; none → −7.8 AP | 2-D sine + reference points | anchor boxes as queries | 2-D reference points + depth embeddings |
| presence / class loss | BCE, negatives 0.1, mean over all slots | softmax CE, no-object weight 0.1, weighted mean | sigmoid focal α 0.25 γ 2, weight 2, per GT | focal α 0.25 γ 2 (DINO cls 1.0) | focal α 0.25, cls coefficient 2 |
| box loss | L1 centre / size in metres 1.0; yaw 1.0; rates 0.5; z, h 1.0 | L1 5 + GIoU 2 (normalised coordinates) | L1 5 + GIoU 2 | L1 5 + GIoU 2 | L1 5 + GIoU 2 + 3-D centre 10 + depth / dim / angle 1 |
| matching cost | −σ(presence) 1 + −p(class) 1 + centre 1.0 / m + size 0.5 / m | class probability 1 + L1 5 + GIoU 2 | focal cost 2 + L1 5 + GIoU 2 | = | cls 2 + bbox 5 + GIoU 2 + 3-D centre 10 |
| per-layer (aux) loss | **no** | yes, shared heads; +8.2 AP layer 1 → 6 | yes (code) | yes (not re-read) | yes (config `aux_loss`) |
| denoising | no | no | no | DN (DN-DETR), contrastive DN with "no object" negatives (DINO) | no |
| iterative refinement | no | no | optional, +1.6 AP | yes, anchor updates per layer (not re-read) | not checked |
| class-score prior | 0.05 | none (softmax) | 0.01 (code) | not checked | not checked |
| schedule | 0.81 epoch (608k windows) | 300-500 ep × 118k = 35-59 M images | 50 ep = 5.9 M | 12 ep = 1.42 M → DINO 49.4 AP | 195 ep × 3,712 ≈ 724k |
| GT used in training | field + decode box | all COCO GT | = | = | drops objects deeper than 65 m or nearer than 2 m |
| eval / decision | none logged in-run; video: P / R at σ ≥ 0.5 | COCO AP, no threshold (panoptic demo 0.85) | top-100, no threshold | COCO AP | AP40; queries below 0.2 confidence dropped |
| head params (ANALYTIC) | 4.09 M / 3.82 M | 9.65 M | 6.43 M | — | ~2.7 M (levels assumed) |

### 2b. Camera 3-D detectors (nuScenes) and dense heads

DETR3D, PETR, BEVFormer, StreamPETR and UniAD share one mmdet3d recipe (PUBLISHED-CODE, all five configs read):
- loss: sigmoid focal (γ 2, α 0.25, weight 2.0) + L1 0.25;
- matching: FocalLossCost 2.0 + BBox3DL1Cost 0.25, with the centre terms in metres;
- output: top-300 with no threshold;
- data: `use_valid_flag=True`, range ±51.2 m.

Per-GT normalisation and a loss at every decoder layer were read in the DETR3D and PETR head code (and in mmdet's
DETRHead, S16). For BEVFormer, StreamPETR and UniAD they are INFERRED from the same code lineage. DETR3D's head also
initialises the class bias at a 0.01 prior.

| dimension | **OURS** box3d | **DETR3D** S5 | **PETR / PETRv2** S6, S7 [LIB] | **BEVFormer** S8 [LIB] | **StreamPETR** S9 | **Sparse4D / v3** S10 | **UniAD** S11 [LIB] | **FCOS3D** S12 | **CenterPoint** S14 |
|---|---|---|---|---|---|---|---|---|---|
| queries vs objects | 100 vs mean ~5, max 120 | 900 vs ~35 annotated per keyframe (S15; ANALYTIC 1.4 M / 40k); 100 → 900 queries lifts mAP 0.313 → 0.346 | 900 (config), 1500 in the paper: NDS 0.351 → 0.359 | 900, top-300 | 644 + 256 propagated | 900 (+600 temporal in v3) | 900 detection + track queries | dense | dense heatmap, max_objs 500 |
| depth / d / FFN | 3 / 256 / 1024 | 6 / 256 / 512 | 6 / 256 / 2048 | 6 / 256 / 512 | 6 / 256 / 2048 | 6 / 256 | 6 / 256 | conv | conv |
| attention | dense over 2,144 tokens | 3-D reference point projected into 6 cameras × 4 FPN levels, bilinear sampling | dense over 3-D-PE multi-view tokens (p4 = stride 16) | deformable into 200 × 200 BEV at 0.51 m | = PETR + temporal propagation | 13 keypoints × 4 scales × views × frames | = BEVFormer + track queries | per pixel, FPN P3-P7 | per BEV cell |
| positions | learned table | 3-D reference points | **3-D PE from camera frustum** (2-D PE only: NDS 0.208 vs 3-D 0.356) | BEV positions + reference points | 3-D PE + motion-aware | 4-D anchors | = BEVFormer | pixel grid | cell grid |
| presence | BCE 0.1-weighted, per slot | shared recipe | shared recipe | shared recipe | shared recipe | focal + depth-confidence reweight; v3 adds centerness + yawness quality | shared recipe; track gates 0.4 / 0.35 | focal + 3-D centre-ness | focal-type heatmap + IoU-guided score |
| box loss | L1 in metres, weight 1.0 | L1 0.25 (velocity × 0.2) | L1 0.25 | L1 0.25 | L1 0.25, centre × 2.0 | L1 + depth BCE | L1 0.25 | smooth L1; depth 0.2, velocity 0.05 | L1, weight 0.25 |
| matching | geometry-dominated (§3 D2) | shared recipe | shared recipe | shared recipe | shared recipe | Hungarian | shared recipe | location-based assignment | heatmap peak |
| aux / refine / DN | no / no / no | yes / yes (config) / no | yes / no: fixed reference points (code) / DN in v2 | yes (INFERRED) / yes (config) / no | yes (INFERRED) / not checked / **DN** | aux not checked / cascade refinement / **DN** (v3) | yes (INFERRED) / not checked / no | — | — |
| schedule / init | 0.81 ep, ImageNet init, joint with the planner | 24 ep × 28,130 ≈ 675k samples, FCOS3D-pretrained backbone | 24 ep | 24 ep, FCOS3D init | 24 / 60 ep, ImageNet or nuImages init | v1: 24-48 ep, FCOS3D init; v3: 100 ep | BEVFormer (24 ep) → 6 ep perception → 20 ep end-to-end | not checked | 20 ep, class-balanced GT paste |
| GT used in training | field + decode box | GT with ≥ 1 LiDAR or radar point, ±51.2 m, 10 classes | = | = | = | = | = | mono conversion: all visibility bins; boxes behind the camera or off-image dropped (S16) | zero-point boxes dropped (INFERRED from file name) |
| head params (ANALYTIC) | 4.09 M | 6.08 M | 11.26 M | 6.18 M | ~PETR | — | ~BEVFormer | — | — |

### 2c. GT filtering and evaluation protocols

| benchmark / practice | training GT rule | evaluation rule | decision on the score |
|---|---|---|---|
| **OURS** | 120° field + 0-60 m ahead + abs(y) ≤ 16 m; everything else DELETED (a slot on it is a negative); no occlusion / size / point / per-class rule | none in-run; video: greedy 2 m BEV matching, AP@2 m and P / R at σ ≥ 0.5 | fixed 0.5 (`presence_gate`), also used as the planner's soft gate |
| **nuScenes** S15 + devkit | objects annotated only if ≥ 1 LiDAR or radar point; mmdet3d drops 0-point GT (S16); ±51.2 m; no camera-visibility cut (FCOS3D keeps all 4 visibility bins) | per-class ranges: vehicles 50 m, pedestrian / 2-wheelers 40 m, cone / barrier 30 m; GT without points removed; ≤ 500 boxes per sample; centre distance 0.5 / 1 / 2 / 4 m; recall and precision below 10 % ignored; NDS | none prescribed; ranking only |
| **Waymo** S17 [LIB] | labels with 0 LiDAR points ignored | LEVEL_1 / LEVEL_2 (≤ 5 points or labeller-hard); AP / APH | ranking |
| **Waymo camera-only** S21 | — | LET-3D-AP: longitudinal tolerance 10 % of range; 3.1 → 23.0 for the same camera detector | ranking |
| **KITTI** S18 | the labels mark DontCare regions (how methods use them in training varies: INFERRED) | Easy ≥ 40 px, fully visible, ≤ 15 % truncation; Moderate ≥ 25 px, partly occluded, ≤ 30 %; Hard ≥ 25 px, ≤ 50 %; DontCare hits are not FPs; AP40 | ranking |
| **CityPersons** S22 | TRAINS only on "reasonable" pedestrians (≥ 50 px, ≤ 35 % occluded); other persons and ignore regions are never negatives | Reasonable MR-2 | — |
| **MonoDETR / MonoDLE** S13, S20 | drop objects > 65 m (MonoDETR); dropping > 60 m: Mod AP 12.97 → 13.66, but > 40 m: 11.25 (worse) (MonoDLE) | KITTI AP40 | MonoDETR drops queries below 0.2 |

**What that means for our pixels** (ANALYTIC, `raw/object_pixel_extent.json`; 416 × 1024 cylindrical, f 488.9 px/rad).
- A 1.66 m person is 25 px tall (KITTI's Moderate floor) at **32.5 m**, and 50 px (CityPersons' floor) at 16.2 m.
- At 40 m that person is 20 × 8 px, i.e. half a stride-16 token wide. At 60 m: 13.5 × 5.4 px.
- In the five-clip GT validation, 49 % of trainer targets lie beyond 30 m and 30 % beyond 40 m (from its range
  table). Those clips are crowd-weighted, not train-representative.

---

## 3. Each difference: the published evidence, and which of our symptoms it explains

### D1. Presence loss, its normalisation, and the fixed gate: S-conf, S-prec, S-flow
- **Ours.**
  - The per-slot optimum is p* = π / (π + 0.1(1 − π)), so logit p* = logit π + ln 10.
  - The 0.5 gate therefore fires at **π = 0.091**. At π = 0.5 the slot sits at p* = 0.909.
  - With exchangeable slots, K ≥ 10 targets per 100 slots already puts every slot above 0.5 (ANALYTIC; the map
    audit derived the same closed form).
- **DETR is the same.** Its no-object weight 0.1 gives the identical optimum (ANALYTIC). But DETR is never read at a
  fixed 0.5: COCO AP ranks, and its demo filters at 0.85 (S1).
- **Every camera-3-D head uses sigmoid focal.** α 0.25, γ 2, weight 2.0, in all five configs (S5, S6, S8, S9, S11).
  The normalisation is per GT: mmdet `avg_factor` = number of positives (S16), read in DETR3D's and PETR's head
  code, INFERRED for the others.
  - Focal's optimum is under-confident: it acts as a maximum-entropy regulariser (S26).
  - α 0.25 down-weights positives. The 0.5 gate therefore corresponds to **π = 0.75**; UniAD's 0.4 and 0.35 track
    gates to π ≈ 0.50 and 0.36 (ANALYTIC).
  - The same score scale that we read as "confident" at 0.5 would, under the literature's loss, need 8× the match
    belief.
- **Gradient share.**
  - Ours divides the presence BCE by B·100 slots, but the box terms by the number of matched targets. The presence
    term is **0.9-1.2 % of the box loss** (MEASURED); the regression terms are 72-74 %.
  - In the mmdet3d recipe classification and regression are both per-GT, and the classification weight is 8× the
    per-metre L1 weight.
- **Published evidence that the loss choice matters, not only the gate.**
  - Focal vs the best α-balanced CE: **34.0 vs 31.1 AP** (S19, dense detector).
  - DETR-DC5 + focal + 300 queries: 35.3 → 36.2 AP at 50 epochs (S2; confounded with the query count).
  - No clean focal-vs-CE ablation inside a DETR-style set head was found: UNVERIFIED for our setting.
- **What it explains.**
  - The saturation at the gate (S-conf, S-prec) is explained ANALYTICALLY, without any learning failure.
  - Whether the RANKING is also broken (S-rank) is the audit's AUROC question.

### D2. The confidence term's share of the matching cost: S-conf, duplicates
- **Ours.** Moving a slot from p = 0.1 to 0.9 on BOTH presence and class buys **1.6 m** of centre error in the cost.
- **The mmdet3d recipe.** Focal cost 2.0 against 0.25 per metre buys **14.9 m** (ANALYTIC,
  `raw/matching_cost_balance.json`; ratio 9.3×; 6.8× at 0.05 → 0.5).
- **So our assignment is geometry-dominated.** With centre errors of 1.5-4 m, the matcher picks the NEAREST slot
  largely regardless of its confidence.
- **Why that matters (S29).** A one-to-one detector matched on location alone keeps duplicate, medium-score
  predictions. Adding the classification cost removes them: RetinaNet one-to-one 33.6 → 37.5 AP, after which NMS
  adds +0.0 instead of +3.2.
- INFERRED for us. The audit's FP decomposition ("duplicate" share) will test it.

### D3. Deep supervision and iterative refinement: S-rank, S-loc, S-conf
- **Ours.** Last-layer loss only (`agent_slots.py:653`); no refinement.
- **Every other query-based head supervises every decoder layer.**
  - Read in text or code for DETR, Deformable DETR, DETR3D, PETR, MonoDETR (`aux_loss True`) and mmdet's DETRHead
    (S1, S2, S5, S6, S13, S16).
  - INFERRED by code lineage for BEVFormer, StreamPETR and UniAD (S8, S9, S11).
  - DETR states the aux loss helps the model output the right NUMBER of objects per class, which is S-conf.
- **Controlled evidence.**
  - Efficient DETR Table 1, read twice: at 1 encoder / 3 decoder layers, **39.8 AP with the per-layer loss vs 28.3
    without** (S27).
  - DETR: +8.2 AP from the first to the last layer (S1).
  - DETR3D: NDS 0.380 at layer 0 → 0.425 at layer 5, with most of the gain by layer 2 (0.420) (S5).
  - Deformable DETR's box refinement: +1.6 AP (S2).
- **Depth itself is fine.** DETR3D's layer 2 is within 0.005 NDS of its layer 5. What we lack is the per-layer
  supervision, not the layers.

### D4. Query budget vs crowd density: S-crowd, S-conf on crowded clips
- **Ours.** 100 queries. M17 ruled "≥ the observed max" on the PARITY train join (max 94). The B1 eval reaches 120.
  Training never dropped a target (0 of 765 logged rows).
- **The literature carries a 13-44× average headroom.**
  - DETR: 100 queries vs 7.7 objects per COCO image, and misses once more than ~50 instances are present (S1, App. A.5).
  - DETR3D: 900 queries vs ~35 annotated boxes per nuScenes keyframe; 100 → 900 queries lifts mAP 0.313 → 0.346,
    saturating at 900 (S5).
  - PETR: 900 → 1500 lifts NDS 0.351 → 0.359 (S6).
  - CrowdHuman DETRs (22.6 persons per image): 500-1000 queries (S28).
- **Our average headroom (~20×) is in family; the crowded tail (0.83×) is not.**
- **This rises or falls with D5.** The audit's partial visibility result takes the five-clip pool from **26.2 to
  9.0 targets per window at vis ≥ 0.30** (`vis_gtval.json`). The right N must be measured AFTER the visibility rule.

### D5. The supervision set: visibility, size, range and IGNORE semantics: S-gt, S-conf, S-prec
- **Ours, from the static read.**
  - `visible_target_filter` DELETES boxes before matching, so any slot that fires on a deleted box is an unmatched
    slot. It is trained toward 0.
  - This cuts both ways:
    - (a) camera-invisible targets inside the field stay POSITIVES, which teaches the head to fire on nothing
      visible;
    - (b) visible objects beyond 60 m or at abs(y) > 16 m become NEGATIVES.
- **The audit (partial, `raw/visibility/vis_gtval.json`).** Its visibility is an upper bound, so every hidden
  share below is a lower bound.
  - 46.2 % of targets have zero visible pixels and 56.2 % are under 10 % visible; persons 55.7 % hidden; the
    45-60 m band 64.7 % hidden.
  - On the LiDAR clip (`vis_lidar.json`), 85.6 % of the under-10 %-visible targets still have ≥ 3 LiDAR points
    (median 77).
- **The literature.**
  - **Camera-visibility cuts come from the monocular and pedestrian protocols, not from nuScenes.**
    - nuScenes and Waymo cut on LiDAR or radar POINTS (S15, S16, S17); on our data that cut would keep most hidden
      targets.
    - KITTI evaluates only ≥ 25 px, occlusion-bounded objects and never counts DontCare hits as false positives
      (S18).
    - CityPersons TRAINS only on ≥ 50 px, ≤ 35 %-occluded pedestrians and excludes everything else from negative
      sampling (S22).
    - Removing annotation errors from pedestrian training data: 28.63 → 23.87 % MR-2 (S23).
  - **Range cuts help only at the right threshold.**
    - MonoDETR drops objects beyond 65 m (S13).
    - MonoDLE: dropping beyond 60 m improves Moderate AP 12.97 → 13.66, but dropping beyond **40 m makes it worse
      (11.25)** (S20).
  - **Per-class ranges are standard in evaluation**: pedestrian 40 m, vehicles 50 m (S15).
- **So a range or size cut must be pre-registered and ablated, not assumed.**

### D6. Geometry in the feature positions: S-loc, S-rank
- **Ours.**
  - A learned per-token table for the image tokens: the same embedding for a given pixel in every clip.
  - Mount height spans 1.203-1.687 m and pitch −4.3° to +3.9° across the 4,508-clip extrinsics bank (config
    `seams.agent_rig_camera`), so a pixel row's ground range differs per clip. A fixed table cannot encode that.
  - The BEV tokens DO carry per-clip geometry, through the lift, but at 2 m cells.
- **The literature.**
  - PETR, 3-D PE vs 2-D PE: **NDS 0.208 → 0.356, mAP 0.069 → 0.305** (S6). Part of that gap is view disambiguation
    across 6 cameras, which a single camera does not need, so our expected gain is SMALLER (INFERRED).
  - DETR without spatial PE: −7.8 AP (S1).
  - DETR3D and Sparse4D sample features at projected 3-D reference points (S5, S10).
  - BEVFormer runs the Deformable DETR head on a 0.51 m BEV (S8).
  - Deformable sampling around reference points reaches 43.8 AP in 50 epochs where dense DETR-DC5 reaches 35.3
    (S2). That matters in our < 1-epoch regime.
- MonoDLE (S20): localisation, and depth above all, is the dominant error of monocular 3-D detection.

### D7. Denoising queries: S-conf (near-miss duplicates), convergence
- **DN-DETR.** Names matching instability as the cause of slow convergence. +1.9 AP, and parity with the baseline
  at half the epochs (S3).
- **DINO.** Contrastive DN trains near-miss queries to say "no object": 47.4 → 47.9 AP, and fewer duplicates (S4).
- **Adopted by StreamPETR and PETRv2; Sparse4D v3** gains +0.008 mAP / +0.009 NDS from it (S7, S9, S10).
- Gains are real but modest. Their value for us is convergence speed in a < 1-epoch budget.

### D8. Schedule, initialisation and staging: S-rank, S-loc, S-flow
- **Ours.** 0.81 epoch from ImageNet init, trained jointly with the planner, tactical and map heads from step 0.
  The two detection losses are about 47-56 % of the joint loss, and box3d's trunk gradient was 145× the map's at
  init (map audit, MEASURED).
- **nuScenes heads.** They see a similar number of sample presentations (24 ep × 28,130 ≈ 675k) but start from a
  detection-pretrained backbone (FCOS3D or nuImages, S5, S8, S9, S10).
- **UniAD.** Pre-trains perception (BEVFormer 24 ep), then 6 epochs of perception, then 20 end-to-end (S11).
- **Set heads converge slowly.** DETR-DC5 at 50 epochs reaches 35.3 AP vs 43.3 at 500 (S2).

### D9. Class weighting
- **Ours.** Full inverse frequency, 1,299 : 1, animal weight 5.55 vs automobile 0.0043.
- **The literature.**
  - None of the five camera-3-D configs sets a class weight.
  - CenterPoint resamples instead (class-balanced GT paste, S14; CBGS resampling, S24).
  - Class-Balanced Loss reports that inverse-frequency re-weighting usually performs poorly under high imbalance
    (S25).
- It touches presence only through the class cost in matching (D2). The audit's per-class confusion should decide
  its rank.

### D10. The planner's gradient into the agent head's presence: S-agent (hypothesis)
- **Ours.** `AgentTokenEmbed` multiplies every planner token by σ(presence) (`refc_agents.py:317-334`), so the
  planner and tactical losses can move presence for utility rather than for correctness.
- **The literature.** UniAD gates tracks with fixed thresholds on focal scores (S11). How its planner consumes
  query scores was not verified here: UNVERIFIED.
- This is INFERRED as a contributor to the agent head being more saturated than box3d.

---

## 4. Ranked recommendations for refcv7's box head

Ranking = strength of evidence × the symptom addressed ÷ cost. Every item changes a declared model or loss setting,
so every item needs a SPEC_REFCV7 amendment (E1) plus G-DVB and G-LIVE entries under the binding launch gate.
"PRE-REG" marks items whose rule must be registered before any refcv7 box number exists.

**P0 (prerequisite, not a head change): detection metrics and bars. PRE-REG.**
- **What.** Pre-register, and log in the in-run eval AND the battery:
  - mAP by centre distance at 0.5 / 1 / 2 / 4 m, per class, with per-class ranges (vehicles 50 m, persons 40 m), in
    range bands;
  - a LET-style longitudinal-tolerance AP for camera-only depth error;
  - presence AUROC (matched vs unmatched);
  - the confident-slots / visible-targets ratio;
  - P / R at a gate chosen on a named CALIBRATION split, never on eval.
- **Evidence.** S15, S21. Refcv6 logged none of these for 38,000 steps (MEASURED), the same failure class as the
  drivable-only map logging (SPEC A3).
- **Cost.** Eval code only; `box3d_ap` and `box3d_match_rows` already exist (`box3d_head.py:457-536`).

| rank | change | evidence (§3) | symptom | cost (params / compute / code) | PRE-REG | interaction with the audit |
|---|---|---|---|---|---|---|
| **R1** | **Presence package.** (a) sigmoid FOCAL presence loss, α 0.25, γ 2, weight 2.0, normalised by the number of matched targets; (b) the focal matching cost (weight 2.0) in place of −σ(presence); (c) presence prior 0.01; (d) a declared decision rule: the gate is chosen on a calibration split (P = R or best F1) and stamped, and the planner consumes a declared quantity (the calibrated probability or the logit). Keep the regression weights for now and ablate the L1 0.25 recipe separately (one variable per arm). | D1, D2: focal 34.0 vs 31.1 AP (S19); config consensus of 5 camera-3-D heads; the 0.5 gate at π 0.091 vs 0.75 (ANALYTIC); class cost removes duplicates, 33.6 → 37.5 AP (S29); the 0.01 prior is RetinaNet's, Deformable DETR's and DETR3D's (S19, S2, S5 code) | S-conf, S-prec, S-flow, duplicates | 0 params; negligible compute; ~40 lines in `slot_set_loss` / `_match_cost` + tests | YES: the loss form, the weights, the calibration-split protocol and a regression arm (today's BCE 0.1 at 0.5) | Pick ONE rule. The audit's tilt-correction (logit − ln 10) is the no-retrain fix for refcv6@38k readouts. It does NOT carry over to a focal-trained head (different optimum), and stacking both double-corrects. The planner's soft gate changes input scale under either, so G-DVB must name what it consumes. |
| **R2** | **Per-layer supervision.** Iterate `self.blocks.layers`, apply the shared `norm` + `head` after each of the 3 layers, re-match per layer and sum the losses (DETR-style, shared heads) | D3: 39.8 vs 28.3 AP at 3 decoder layers (S27); DETR's statement on object counts (S1); present in every query-based head read (S1, S2, S5, S6, S13; INFERRED for S8, S9, S11) | S-rank, S-loc, S-conf | 0 params (shared heads); head compute ×3, trivial vs the trunk; Hungarian calls ×3, CPU numpy, small K typical (measure the crowded K ≈ 100 case in G-LIVE) | YES, as a declared lever (G-DVB: on; G-LIVE: per-layer terms finite) | independent of the visibility rule; strengthens R1 because each layer's presence is supervised |
| **R3** | **Supervision set with IGNORE semantics.** Targets = the audit's visibility rule (e.g. vis_frac ≥ τ) ∧ a pixel-height floor ∧ per-class range. Boxes that fail the rule but are real (hidden, too small, beyond range, or visible beyond the decode box) become IGNORE: match against targets ∪ ignore, and a slot matched to an ignore box gets presence weight 0 (neither positive nor negative), as KITTI DontCare and CityPersons do. | D5: CityPersons trains on reasonable only, ignore never negative (S22); KITTI 25 px / occlusion + DontCare (S18); MonoDETR 65 m (S13); MonoDLE 60 m helps, 40 m hurts (S20); cleaning training annotations 28.63 → 23.87 MR-2 (S23); audit: 46.2 % hidden, LiDAR points do not catch it | S-gt, S-conf, S-prec | 0 params; per-box visibility precomputed into the join (Data Engineering: a z-buffer per window on CPU; the audit's `vis_zbuf.py` is the reference; its wall time, 193 s for 855 windows, extrapolates to ~47 single-process CPU-hours for the 746,946 train windows: INFERRED, parallelisable); ~80 lines in the filter, the matcher and the loss; the class-weight vector recounted on the new population (`load_cls_class_weight` REFUSES a mismatched population, correctly) | YES: τ, the pixel floor, the range per class; the arms are today's filter (regression), rule-as-delete and rule-as-ignore; choose the thresholds on TRAIN or a calibration split (MonoDLE shows the sign can flip) | this IS the audit's coming visibility rule; the recommendation is HOW to apply it (ignore, not delete). It also removes the "filtered-GT" false positives the audit's decomposition will count, some of which are the head being right. |
| **R4** | **Query count.** Re-rule M17 on the refcv7 corpus line AFTER R3: N ≥ 2 × the max visible targets per frame on TRAIN (DETR finds all only below ~50 % of N). This likely means 300. | D4: S1 App. A.5; DETR3D 100 → 900 queries, mAP 0.313 → 0.346 (S5); PETR 900 → 1500 (S6); CrowdHuman 500-1000 queries (S28) | S-crowd, S-conf on crowds | +51,200 params per head at N = 300 (box3d decoder 3.86 M, inside the 2-4 M band); self-attention ×9, cross-attention ×3 on the head only; Hungarian m = 300 | YES (a model shape; G-DVB) | ONLY together with R1: more empty slots under BCE 0.1 means more confident false positives. N depends on R3's target count. |
| **R5** | **Geometry-grounded feature access.** Either (a) a PETR-style ray / frustum position embedding for the image tokens, built from the per-clip rig (the `LiftGeometryBank` already delivers the per-clip geometry to the forward as `perception_grid`), replacing the 426k-param image part of `mem_pos`; or (b) a Deformable-DETR / BEVFormer-style head: one reference point per query, sampling the full-resolution per-clip-lifted BEV (0.5 m, not the 2 m pooled tokens) and the image at the projected point, refined per layer. | D6: PETR NDS 0.208 → 0.356 (S6, multi-view, so smaller for us); Deformable DETR 43.8 vs 35.3 AP at equal epochs (S2); DETR3D and Sparse4D reference points (S5, S10) | S-loc, S-rank | (a) ~0.1-0.3 M params replacing 0.43 M, moderate code; (b) a new cross-attention (grid-sample, no custom CUDA needed), several hundred lines | YES + a ladder run first (`TanitAD_ValidateAIDesign`, v7-tiny) | independent of R3; with R2, the per-layer reference points become the refinement |
| **R6** | **Denoising queries** (DN, then contrastive DN with "no object" negatives), training-only | D7: DN parity at 50 % epochs, +1.9 AP (S3); CDN +0.5 AP and fewer duplicates (S4); StreamPETR, PETRv2, Sparse4D v3 | S-conf (near-miss duplicates), convergence | 0 inference params; a few hundred lines (the attention mask and groups) | YES + ladder | needs R1's focal presence so the "no object" targets are learned the same way as the main set |
| **R7** | **Staging / initialisation.** A detection-only warm start of the perception branch, or a detection-pretrained trunk, before joint training | D8: FCOS3D / nuImages init (S5, S8, S9, S10); UniAD staging (S11); DETR 50 vs 500 epochs (S2) | S-rank, S-loc, S-flow | GPU-days; conflicts with SPEC_REFCV7 §4 "from ImageNet init" | PI decision | independent |
| **R8** | **Class weights.** Replace full inverse frequency with no weight, √-inverse or effective-number weights, or resampling | D9: S25; no class weight in the 5 camera-3-D configs; CBGS resampling (S14, S24) | class confusion (cls loss 1.04 train / 1.30 eval, MEASURED) | trivial | YES (the corpus and population guard already enforce scope) | recount on R3's population |
| **R9** | **Agent head: stop the planner gradient into presence** (detach σ(presence) in `AgentTokenEmbed`'s scaling) as a knockout ARM | D10 (hypothesis) | S-agent | one line + an arm | an arm, not a default | read after the audit's per-head AUROC |

**Not recommended as a lever: decoder depth or width.** Our head sits inside the published parameter range, and
DETR3D's gain beyond layer 2 is ≤ 0.005 NDS (S5). Multi-scale image tokens (stride 8: +1.7 / +1.5 AP for multi-scale
in S2; 2× up-sampling +3.74 MR-2 for small pedestrians in S22) are real, but only affordable with R5(b)'s sparse
sampling. Revisit after R5.

---

## 5. What must be registered BEFORE any refcv7 box number

1. **P0**: the metric set, the range bands and per-class ranges, the calibration split used for the gate, and the
   bars:
   - at least a separated gain over refcv6@38k on mAP at the same windows;
   - a confident-slots / visible-targets ratio ≤ 1.5 on train-density windows (the map audit's proposed check,
     `box_presence_partial.md` §4), with the regression arm BCE 0.1 at 0.5.
2. **R1**: the presence loss form and weight, the matching-cost form, the prior, the decision rule, and what the
   planner consumes.
3. **R3**: the visibility threshold, the pixel floor, the per-class ranges, the IGNORE semantics, and the class-weight
   population they imply.
4. **R4**: the N rule and its measured basis on the refcv7 corpus line (not parity, not eval).
5. **R2**: on or off, as a G-DVB lever.

R5-R7 are ladder candidates: registered as `TanitAD_ValidateAIDesign` specs with their own deliberate-regression arms,
not folded silently into the first refcv7 launch.

---

## 6. Interactions with the box-head audit

- **The presence fix.** The audit's tilt-corrected gate is the correct zero-training readout of refcv6@38k.
  - For refcv7 it is one of two mutually exclusive rules: keep BCE 0.1 and declare the ln 10 correction, OR adopt R1
    with a calibration-split gate. Never both.
  - The planner reads σ(presence) directly (soft gate), so either choice changes the planner's input distribution.
    That is a G-DVB item.
- **The visibility rule.**
  - Use the audit's per-box visibility as the TARGET predicate, and treat failures as IGNORE (R3).
  - Recount the class weights on that population.
  - Re-measure the per-frame maximum on TRAIN before setting N (R4).
  - The audit's partial numbers show the rule changes the crowded-clip density by about 3× (26.2 → 9.0 targets per
    window at vis ≥ 0.30), so R4 cannot be set before R3.
- **The audit's pending FP decomposition** (duplicate / mislocalised 2-5 m / filtered GT / out of field /
  hallucination) re-orders R1-R5:
  - a large duplicate share → R1(b) and R6;
  - a large filtered-GT share → R3's ignore semantics;
  - a large mislocalised share → R5;
  - a large hallucination share on hidden targets → R3.
- **The audit's presence AUROC.**
  - High AUROC with a saturated gate → the problem is mostly the decision rule (R1(d) + P0).
  - AUROC near 0.5 on crowded clips → the learning signal itself (R1(a)-(c), R2, R3).

---

## 7. What this does not show, and what is UNVERIFIED

- **No model was run.** Every model-side number here is from the run's own logs or checkpoint shapes, or is cited
  from the video agent, the GT validation, the map audit or the (partial) box-head audit.
- **The ranking of R1-R5 is an argument from published evidence plus our analytic optimum, not a measurement on our
  data.** Each needs its ladder or smoke arm.
- **Transfer caveats.**
  - The published ablations are on COCO, nuScenes and KITTI, with different sensors and densities.
  - PETR's PE gain is multi-view.
  - Efficient DETR's aux-loss ablation is on Deformable DETR at 3× schedule.
  - RetinaNet's focal gain is on a dense detector.
- **Single-read numbers** (marked 1x in `raw/literature_factsheet.md`) are published but were not cross-checked in
  this pass. One first read (S27) was wrong and was corrected by a second read, which is why the flag exists.
- **UNVERIFIED items.**
  - How UniAD's planner consumes query scores.
  - Whether CenterPoint's `filter_True` means zero-point filtering.
  - The head-branch layouts behind the DETR3D, BEVFormer and PETR parameter counts (±10 %; the transformer-layer
    counts come from the configs).
  - MonoDETR's number of feature levels.
- **Registry.** `MODEL_REGISTRY.md` has no box-head facts for refcv6. No conflict was found between it, the kit
  README and `config.json` on step, md5, trunk or resolution.
- **Density figures.** The 26.2 targets per window figure is the five-clip GT-validation pool (crowd-weighted). The
  training median is 4.75-5.53 (MEASURED here). The typical eval clip reads 2.65 (GT validation).
- **I did not do the model-side checks.** Presence AUROC, the per-slot distributions and the FP decomposition are the
  audit's.

---

## 8. arXiv ids to bank (the PI's permission is needed; nothing was downloaded or banked)

Already banked and cited by key: 2005.12872 (DETR), 2203.05625 (PETR), 2206.01256 (PETRv2), 2203.17270 (BEVFormer),
2212.10156 (UniAD), 1903.11027 (nuScenes), 1912.04838 (Waymo), 1602.01237 (pedestrian detection).

**Priority 1 (load-bearing for R1-R6):**
- 2104.01318 Efficient DETR
- 2010.04159 Deformable DETR
- 2110.06922 DETR3D
- 1708.02002 Focal loss / RetinaNet
- 2012.05780 OneNet
- 1702.05693 CityPersons
- 2103.16237 MonoDLE
- 2203.13310 MonoDETR
- 2203.01305 DN-DETR
- 2203.03605 DINO
- 2303.11926 StreamPETR
- 2311.11722 Sparse4D v3

**Priority 2:**
- 2211.10581 Sparse4D
- 2104.10956 FCOS3D
- 2006.11275 CenterPoint
- 2206.07705 LET-3D-AP
- 2203.07669 Progressive DETR in crowds
- 1901.05555 Class-Balanced Loss
- 1908.09492 CBGS
- 2002.09437 Focal-loss calibration
- 1405.0312 COCO

KITTI's benchmark definitions are a web page, not an arXiv paper.

---

## 9. Deliverable manifest

All files live in ONE place: the D: working tree, at
`D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-box-head-literature/`. They
are NOT committed and NOT staged; the Master Mind is the single committer. Scratch copies in the session scratchpad
are not a second home.

| artifact | where | only one place? |
|---|---|---|
| `RESULT.md` (this file) | repo working tree: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-box-head-literature/RESULT.md` | yes, uncommitted |
| `raw/literature_factsheet.md`: every source, its URL, the facts, 1x / 2x reads | same folder | yes |
| `raw/sources.json`: 31 sources, banked flag, URLs read, the to-bank list | same folder | yes |
| `raw/presence_optimum_by_loss.py` + `.json`: ANALYTIC p*(π) and π at each gate for BCE, ours, DETR and focal | same folder | yes |
| `raw/matching_cost_balance.py` + `.json`: ANALYTIC metre-equivalents of the confidence cost | same folder | yes |
| `raw/head_param_counts.py` + `.json`: ANALYTIC head sizes; self-check reproduces ours exactly | same folder | yes |
| `raw/object_pixel_extent.py` + `.json`: ANALYTIC pixel sizes and published cut-off ranges | same folder | yes |
| `raw/refcv6_box_loss_share.py` + `.json`: MEASURED loss shares, drops and logged-metric census from `metrics.jsonl` | same folder | yes |
| `raw/refcv6_ckpt_head_shapes.json`: MEASURED head tensor shapes from the checkpoint (mmap, shapes only) | same folder | yes |
| `LANDING_READY.txt` | same folder | yes |
