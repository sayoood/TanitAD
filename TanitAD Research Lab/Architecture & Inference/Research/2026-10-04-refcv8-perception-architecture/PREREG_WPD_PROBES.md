# PREREG_WPD_PROBES — REGISTERED by the Master Mind 2026-10-04 (hash + time in raw/SPEC_SHA256.txt), before any probe arm produced a number

Registration notes (binding): (1) Thor GPU order — P-GRAD first (it sets the refcv8 loss budget), then P-BOX capture + arms, then P-MAP, each job <= 40 min under flock /home/nvidia/refcv7_post/thor_gpu.lock, interleaved with R1 and WP-RL; P-BOX arms may run on dev-box CPU instead. (2) P-DEPTH stays BLOCKED on a LiDAR depth-target build (a Data FlyWheel job + a PI download decision). (3) P-TEMP stays DEFERRED pending the PI's admissibility ruling on past ego-motion for BEV warping. Nothing below is changed.

---

# PREREG DRAFT — WP-D perception probes for refcv8 (R8-6)

*WP-D, 2026-10-04. Arm names are prefixed PB (box) / PM (map) / D (depth) so they never collide with the design's change ids (B1–B4, M1, M5, W1–W6). **DRAFT, NOT BINDING.** It binds only when the Master Mind registers it (sha256 + UTC time in
`raw/` or the SPEC) BEFORE the first number of any arm below exists. Bars, arms, windows and step counts are written
now so they cannot move after the data. Design rationale: `PERCEPTION_DESIGN.md`; evidence: `RESULT.md`.*

## 0. Rules common to every probe

* **Tier:** OPEN-LOOP PERCEPTION DIAGNOSTIC. **The four driving families are N/A** (no plan is produced or scored);
  every result says so per family. Perception feeds the longitudinal family (lead-agent boxes) — that link is measured
  in refcv8's R5 battery, not here.
* **Eval windows:** EVAL-DIAG = the diagnostics package's 1,112 windows / 139 eval episodes (window-set sha256
  `92e36a1a74b8b699…`, `…/2026-10-04-refcv7-map-box-diagnostics/raw/run_eval_*.json`); 1,061 labelled windows / 137
  episodes for boxes. **Fit set** (thresholds, P = R gates): TRAIN-DIAG (1,112 windows / 139 train episodes, same
  package). Nothing is tuned on EVAL.
* **Decision rules held fixed across arms** so an architecture effect is never a decision-rule effect: map = F1
  (TRAIN-fitted per-class probability thresholds, re-fitted per arm on TRAIN-DIAG); box = the TRAIN P = R gate per arm
  for count metrics, and every AP is reported both raw and after centre-distance NMS r = 2 m (WP-C's F4b), the NMS
  radius fixed now.
* **Estimator:** paired episode-cluster bootstrap over the eval episodes, B = 1,000, 95 % percentile. **Plus a
  REPLICATE arm per probe family** (the control re-run with a different training seed): a lever's effect counts only
  if its paired CI excludes 0 **and** its point effect exceeds |replicate − control| (the rig's training-run noise
  floor, H-ESTIM-SEED-1). Name which question each interval answers.
* **Must-fail arms** read known values, or the probe is VOID. **No early stopping:** every arm runs its fixed step
  count; the reported checkpoint is the last.
* **GPU:** Thor only, under `flock /home/nvidia/refcv7_post/thor_gpu.lock`, ONE job at a time, each ≤ 40 min, queued
  after A7 capture → R1 → WP-RL (Master Mind order 2026-10-04). `OMP_NUM_THREADS=6`. No python process left alive.
  Disk: Thor had **56 GB free** (MEASURED 2026-10-04 13:2x, `df -h /home`: 937G / 834G used) before R1's 13.9 GB.
* **Privacy:** no raw clip ids in any artifact (sha12 only).

## P-GRAD — who trains the trunk, per TERM (≈ 5 min GPU, 0 disk) — run FIRST

Question: of the 99.4 % "aux" share of the trunk gradient (RESULT §1.4), how much is box3d, agent, map_hires and
tac_v6? Sizes any loss-weight change (design W5) before it is proposed.
* refcv7-50,400, train mode, frozen BN as trained; 8 TRAIN batches (b 8, seeded draw from TRAIN-DIAG windows).
  One backward per term (traj, box3d, agent, map_hires, tac_v6) at its loss weight, plus one backward of the total.
* Report: per term × trunk stage (stem, layer1–4, fuse) the gradient L2 norm and the cosine to every other term.
* **Control (must read known values):** Σ per-term gradients = total gradient, relative error ≤ 1e-4 per stage
  (linearity); a term with weight 0 reads exactly 0.
* No bar (a census); its numbers set the design W5 loss budget in the refcv8 SPEC.

## P-BOX — the slot decoders on a FROZEN box memory (cache ≈ 4.5 GB, ≈ 40 min capture + ≈ 5–7 min per arm)

**Cut point.** The output of `Box3DMemory` (`box3d_head.py:132`) — `[2144, 256]` = 1,664 stride-16 image tokens +
480 BEV tokens, after `img_proj` / `bev_proj` / `src_embed` — for box3d; for the agent arm (A-arms) the agent head's
memory `[416, 2048]` (stride-32 map tokens). Plus per window: the VIS-1 targets (POS / IGNORE, boxes, classes, rates,
z / h), the 0.5 m `bev_feats` `[96, 120, 64]` int8-affine (R1's format), and the clip's lift geometry id (the PETR-PE
arm builds ray directions from `LiftGeometryBank`, per clip, not per window).
**Windows.** EVAL-DIAG 1,112 + TRAIN 3,000 (600 train clips drawn with `numpy.default_rng(0)` from the 4,369, 5 windows
each at evenly spaced t; recorded in `selection.json` with sha12 + t). **Price** (computed from the tensors the arms
read): box_mem fp16 1.10 MB + bev int8 0.74 MB + targets ~0.05 MB ≈ **1.9 MB/window → 7.8 GB**; box_mem only ≈ 4.5 GB.
Capture: ESTIMATED 37–40 min (diagnostics eval pass MEASURED 9.1–10.9 min per 1,112 windows at b 8) — one GPU slot.
The cache may be moved to the dev box over the LAN (~63 MB/s MEASURED) and the arms trained there on CPU if Thor's
queue is full (ESTIMATED ~0.3 s/step at b 8; one arm ≈ 20 min).

**Training of every arm:** start from the refcv7-50,400 decoder weights (warm, the refcv8 default), 4,000 steps,
batch 16 windows, AdamW lr 1e-4 constant for steps 0–3,599 then cosine to 0 over the last 400 (A17's shape), wd 1e-4,
clip 10, seed 0; the launch loss (`box3d_loss_row`: focal presence α 0.25 γ 2 × 2.0, deep supervision, VIS-1 IGNORE
semantics, learned_ref anchors) unless the arm changes exactly one of these.

| arm | the ONE change vs PB0 | inference changed? |
|---|---|---|
| **PB0** | none (continue the refcv7 recipe on the cache) — the control | no |
| **PB0r** | PB0 with seed 1 — the replicate (noise floor) | no |
| **PB1 CDN** | contrastive denoising queries, training only: 3 groups × (noised positives: centre ± U(0, 0.4)·max(l, w), size × (1 ± 0.2), yaw ± 0.2 rad, labelled positive; noised negatives: centre noise U(0.4, 1.0)·max(l, w), labelled no-object), attention-masked from the matching queries and from other groups, reconstructed with the same box / presence / class losses (DINO CDN [LIB:2203.03605]) | no |
| **PB2 hybrid** | K = 4 query groups (groups 1–3 = copies of the trained query table + anchors jittered U(±1 m)), one-to-one Hungarian WITHIN each group, group-masked self-attention, losses averaged over groups; inference reads group 0 (Group DETR [LIB:2207.13085]) | no |
| **PB3 quality** | presence target for a matched slot = q = exp(−d² / 2·(1.0 m)²), d = BEV centre distance to its target (varifocal form: positives weighted by q, negatives focal as today) [LIB:2008.13367, 2304.04742] | score semantics |
| **PB4 refine** | per-layer anchor refinement: layer l+1's anchor = detach(layer l's decoded centre), its query position re-embedded through the existing `AnchorPosEmbed` (0 new params) [LIB:2201.12329, 2010.04159] | yes (anchors per layer) |
| ~~PB5 exempt~~ | ELIMINATED before registration by PROBE-0 (RESULT §2.2: objects with an IGNORE row within 2 m have FEWER duplicates, 1.22 vs 1.99; exempt slots are 3.8 % of FPs) — not run | — |
| **PB4h HQS** | the already-built heatmap query selection (`--slot-query-select heatmap`) instead of PB4's refinement — the per-frame anchor alternative | yes (anchors per frame) |
| **PB6 all** | PB1 + PB2 + PB3 + PB4 | — |
| **PB7 PETR-PE** | add to the image tokens a ray position embedding (per-token cylindrical ray direction from the clip's extrinsics → 2-layer MLP, last layer zero-init) [LIB:2203.05625] | yes |
| **PB8 BEV-sample** | each query additionally reads the 0.5 m `bev_feats` bilinearly at its current anchor (3×3 neighbourhood, linear → d_model, zero-init) — instead of only the 2 m pooled BEV tokens [LIB:2110.06922, 2203.17270] | yes |

**Must-fail (VOID if one passes):** `R-pres0` = PB0 with the presence loss weight 0 → AP@2 m (raw) must fall by ≥ 50 %
vs PB0; `R-memshuf` = PB0 with memory tokens permuted across the windows of each batch → AP@2 m ≤ 0.05; `R-qshuf` =
PB3 with q permuted among the matched slots of the batch → must NOT beat PB0 on AP@1 m (paired CI upper bound ≤ +0.01).

**Metrics** (EVAL-DIAG, class-agnostic unless stated): AP@{0.5, 1, 2, 4} m raw and after NMS r 2 m; per band 0–20 /
20–40 / 40–60 m; boxes per detected object at the TRAIN P = R gate (PROBE-0's definition); conf_ratio at that gate;
presence ECE; centre error split longitudinal / lateral by band (PROBE-0 Q5); per-class AP@2 m for automobile and
person.

**Bars (a lever PASSES only if all hold):**
1. ΔAP@2 m after NMS vs PB0 ≥ **+0.02**, paired CI > 0, and > |PB0r − PB0|; OR (for PB1, PB2, PB3) boxes per detected
   object reduced by ≥ **0.30** vs PB0 with ΔAP@2 m (after NMS) ≥ −0.005.
2. No band's AP@2 m (after NMS) drops by more than 0.02 vs PB0.
3. For PB3 / PB4 / PB7 / PB8 (ranking and localisation levers; PROBE-0b put the same-box ranking ceiling at AP@1 m +0.062): ΔAP@1 m (raw) ≥ **+0.02** with the same CI and floor rule.
**"Dramatic" (PB6, the candidate refcv8 recipe), committed now:** AP@2 m **raw (no NMS)** ≥ **0.339** (refcv7's
MEASURED NMS-r2 level, RESULT §2.4) AND AP@2 m after NMS ≥ PB0's after-NMS value + 0.08 AND AP@1 m raw ≥ **2 × PB0's** AND
boxes per detected object ≤ **1.3** (refcv7: 1.97) on EVAL-DIAG — ⚠️ on a FROZEN memory, so a pass says the decoder recipe is worth the
run, not that refcv8 will reach it.

## P-MAP — the 10 cm branch on a FROZEN trunk, near range, features recomputed on the fly (0 disk, ≈ 30–40 min per arm)

**Why on the fly:** a `map_hires_bev` cache is 12.3 MB/window fp16 (R1 DESIGN §3) and cannot fit Thor's disk at a
useful window count; the trunk's stem → layer3 forward with no grad is cheap next to the 10 cm branch.
**Setup:** refcv7-50,400 trunk FROZEN (BN already frozen), stem → layer3 run per step on the newest frame
(`s8_from_normalised` + the s16 fused map); the 10 cm branch starts from the refcv7 weights; the target and the
supervised region are CROPPED to x ∈ [0, 40) m × y ∈ ±30 m (400 × 600 cells at 0.1 m) — the region where the image
supports 10 cm (RESULT §3). Train windows streamed from the canonical train cache (seeded sampler); eval EVAL-DIAG.
3,000 steps, b 8, head lr 1e-4 constant then cosine over the final 300, seed 0. **Cost:** ESTIMATED 0.6–0.8 s/step
(the map-overfit harness MEASURED 0.584 s/step at b 4 with the trunk detached, INHERITED from
`2026-09-26-refcv7-map-hires/raw/gmo_early/…A15…json`) → 30–40 min per arm = one GPU slot each.

| arm | the ONE change vs PM0 | touches the planner's BEV? |
|---|---|---|
| **PM0** | none (continue the refcv7 branch, crop as above) — the control | — |
| **PM0r** | PM0, seed 1 — the replicate | — |
| **PM1 s4-near** | a stride-4 near lift: layer1 (256 ch, 104 × 256) projected to `d_up`, sampled at the 10 cm cells of x ∈ [0, 40) m with the lift geometry, ADDED on those rows like `NearLiftSkip` (projection zero-init); the existing s8 near lift stays | no (map-only) |
| **PM2 fpn** | the s8 lift input += upsample×2(1×1(s16 fused)), the 1×1 zero-init (Simple-BEV's deeper-stage fusion [LIB:2206.07959]) | YES (shared 0.25 m encoder) |
| **PM3 line** | lane and edge targets softened laterally: soft-CE with target mass 1.0 on the GT cell and exp(−k²/2) on ±1–2 cells across the line, renormalised (the F3 "tolerance-aware" target, diagnostics §3) | no |
| **PM4 all** | PM1 + PM2 + PM3 | yes |

**Must-fail:** `R-s4zero` = PM1 with the layer1 input replaced by zeros → must equal PM0 within |PM0r − PM0| on every
metric (a zero-fed zero-init skip carries nothing); `R-s4shuf` = PM1 with layer1 feature ROWS permuted within each
image → must NOT beat PM0 on 20–40 m IoU_2 (upper CI ≤ +0.01); `R-gtshift` = PM0 trained with the lane / edge GT
shifted +0.5 m laterally → its EVAL IoU_0 for lane at 0–20 m must fall below PM0's by ≥ 0.05.

**Metrics** (EVAL-DIAG, F1 thresholds re-fitted per arm on TRAIN-DIAG): per band 0–20 / 20–40 m and class (lane,
edge, crosswalk, drivable): IoU_0, IoU_2 (0.2 m Chebyshev tolerance), P@1 m and R@1 m (diagnostics' M-b definitions,
`code/diag_metrics.py`); the drivable-boundary-within-0.2 m-of-GT-edge share (M-f).

**Bars:**
1. PM1: edge or lane IoU_2 at **20–40 m** + **0.03** absolute vs PM0, CI > 0, > replicate floor.
2. PM2: edge R@1 m at **0–20 m** + **0.05**, same rule. ⚠️ PM2 shifts the planner's BEV input: it enters refcv8 only
   zero-gated (design §2.6 W3).
3. PM3: edge IoU_2 at 0–20 m + **0.03**, with drivable IoU_0 non-inferior (Δ ≥ −0.01).
4. **"Dramatic" (PM4), committed now:** edge IoU_2 0–20 m **0.235 → ≥ 0.33**; edge IoU_2 20–40 m **0.118 → ≥ 0.18**;
   lane IoU_2 20–40 m **0.308 → ≥ 0.38** (baselines: diagnostics M-b, thr_phat, EVAL). No bar is set beyond 40 m: the
   ANALYTIC bound (RESULT §3) says the image does not support 10 cm there; 40–100 m is reported with the range-adaptive
   metric of design §2.4 (M5), not barred.

## P-DEPTH — LiDAR depth as an auxiliary target (v7-tiny ladder; BLOCKED on a data build)

* **Blocker (named):** no projected front-camera depth target exists for any TRAIN clip. LiDAR exists in the corpus
  (`lidar_top_360fov`, 10 Hz, 199 spins/clip, same µs clock as the camera, median |camera − spin| 30.2 ms — MEASURED by
  `…/2026-09-11-lidar-bev-gt/RESULT.md`), but each clip costs **~342 MB** of zip-member streaming (same source) — the
  v7-tiny clip set ×342 MB is a Data-FlyWheel job and a download the PI must authorise (filename, source, size).
* **Target once built:** per frame, LiDAR points of the nearest spin (deskewed with the corrected sign, 09-13 P0)
  projected through the clip's extrinsics into the 416 × 1024 cylindrical frame, min-depth per stride-8 cell, stored
  uint16 (cm) — **~13 KB per frame** (52 × 128 × 2 B).
* **Arms (v7-tiny, the real trainer, ~17 min/arm on Thor, INHERITED from PLAN §6 R2):** D0 control, D0r replicate,
  D1 = + a depth head on the s8 / s16 features (L1 on log-depth over valid cells, weight to be set from P-GRAD),
  **D1-wrongclip** (must-fail: depth targets from a different clip → must not beat D0), D2 = D1 + depth-weighted lift
  samples (FB-BEV-style depth consistency [LIB:2308.02236]).
* **Evidence is two-sided and is reported as such:** depth supervision helped detection (BEVDepth +2.2 mAP
  [LIB:2206.10092]) and vectorised maps (MapTRv2 +5.1 mAP [LIB:2308.05736]); a hard one-hot depth target HURT
  occupancy (DualPathOcc 35.96 vs 36.92 mIoU without [LIB:2609.06370]). ⇒ a **no-supervision control is mandatory**
  (LAB_BACKLOG LR17-4) and the depth target is Gaussian-softened in D1.
* **Bar:** box AP@1 m (raw) + 0.02 AND edge IoU_2 0–20 m + 0.02 vs D0, CI > 0, > replicate floor; D1-wrongclip
  must fail.

## P-TEMP — temporal BEV fusion (deferred; registered so it is not forgotten)

Fuse an ego-warped BEV of an earlier window frame (the trunk already computes the W window frames each step; the
perception branch reads only the newest). ⚠️ **Admissibility to state before any arm:** warping uses PAST ego motion
(odometry); the PI ruled measured v0 at cycle time admissible (2026-09-02) — past ego poses for perception warping
need the same explicit ruling. Evidence: SOLOFusion 0 → 16 history frames mAP 0.307 → 0.377, mATE 0.743 → 0.655
[LIB:2210.02443]; BEVDet4D, StreamPETR [LIB:2203.17054, 2303.11926].

## Order and budget

| order | probe | GPU | disk | gate to refcv8 |
|---|---|---|---|---|
| 1 | P-GRAD | ~5 min | 0 | sets the W5 loss budget |
| 2 | P-BOX capture | ~40 min | 4.5–7.8 GB | — |
| 3 | P-BOX arms (13 incl. must-fail) | ~1.2–1.5 h on Thor in ≤ 40-min jobs, or CPU on the dev box | 0 | B-levers that PASS enter the refcv8 SPEC |
| 4 | P-MAP arms (8 incl. must-fail) | ~4–5 h in 30–40-min jobs | 0 | M-levers that PASS enter the SPEC |
| 5 | P-DEPTH | after the data build | ~13 KB/frame | PI decision |
