# ASIS_PERCEPTION: how refcv8's perception stack is built, from source (input to the refcv9 perception redesign)

Architecture & Inference, 2026-10-27 (Berlin). Brief: the PI's 2026-10-27 ruling. refcv9 becomes FULL training, with
perception included and no 48 h cap. It launches only after a research-backed redesign that "dramatically" improves
the 3D box head and the BEV map: *"Review architecture and training recipes of the heads, review size and
representation of the map."* This file is the **AS-IS** half of that redesign. Two sibling agents cover the
literature (detection, BEV map). A third measures the bottleneck: a frozen-feature probe, a LiDAR adjudication of
the map GT, and per-class box AP.

**What was read.** The code is `refs/heads/agent/arch-inf-20260803` @ `56b25a61`. It was read through plumbing
(`C:/Users/Admin/tanitad-push/.git`). Every `file:line` below refers to that ref.
- **Launch-tree check (MEASURED, blob comparison, both sides 40 chars):** these 9 perception files are byte-identical
  between refcv8's launch commit `fb47a58` and `56b25a61`: `bev_lift.py`, `map_head_hires.py`, `box3d_head.py`,
  `agent_slots.py`, `slot_query_select.py`, `slot_presence.py`, `refcv6_perception_branch.py`, `timm_trunk.py` and
  `semantic_map_gt_fine.py`.
- `refc.py` and the trainer differ from the launch tree. Their line numbers are at `56b25a61`, and the cited logic is
  the launch tree's. The exception is the refcv9 `--w-map-fp` term, which is new (§4.5).
- The launch argv is `stack/ops/runs.d/refcv8-r101-s0.argv.json` (210 tokens).

**Tier and four families.** This is a STATIC SOURCE REVIEW plus re-reads of banked OPEN-LOOP PERCEPTION diagnostics.
No plan is produced or scored, so the four metric families (longitudinal, lateral, tactical, strategic) are N/A for
this file, for that one reason.

**Evidence classes.**
- **MEASURED (source)** is a fact read from code at the ref above.
- **MEASURED (artifact)** is a number I re-read from the named file.
- **INHERITED** is another agent's banked number, not re-derived here.
- **ESTIMATED** is my arithmetic from stated inputs.

No GPU, no Thor, nothing executed beyond `git`/`grep`/`python` reads.

---

## 0. Headline: the eight findings that should steer the redesign

1. **The map head trains on features it has essentially no say in.** P-GRAD on refcv7-50,400 is refcv8's warm-start
   point. Every loss weight is 1.0 there.
   - The 10 cm map term pays for **0.014 %** of the whole-trunk update. Box3d pays for **35.7 %** and the agent head
     for **50.1 %**.
   - Inside the map's OWN 0.25 m BEV encoder, box3d pays for **97.3 %** and the map for **−0.4 %**. Box3d's gradient
     norm there is **15×** the map's.
   - MEASURED (artifact): `…/2026-10-04-refcv8-perception-architecture/raw/pgrad.json`, `summary.proj_share_on_total`.
   - ⇒ Today the "map head" is close to a probe on detector features.
2. **The map reads layer2 only, from the current frame only.** That is the stride-8 map: 512 ch, ImageNet layer2
   semantics, no FPN, no temporal fusion. MEASURED (source): `map_head_hires.py:849`, `timm_trunk.py:757`, `:803`.
3. **The whole BEV is a 64-channel, 538,112-parameter bottleneck.** That is 0.53 % of the 100.6 M model. This one
   tensor feeds the map, the planner's BEV, the box head's BEV tokens and the tactical decoder.
   - MEASURED (source): `map_head_hires.py:352-357`.
   - MEASURED (artifact): the group size `x_shared_bev_025m = 538,112` in `pgrad.json`. My arithmetic from the source
     gives the same number.
4. **The lift has a single geometry.** It samples the image along a flat road plane at 4 fixed heights.
   - No depth distribution and no depth supervision.
   - The pitch is the clip's static extrinsic, never the frame's.
   - ESTIMATED: a 1° pitch error at 40 m moves the sampled ground to 27–77 m. The box error is longitudinal (INHERITED).
5. **The 10 cm label runs 2.5× deeper than the image can support it, and it is supervised uniformly.** The label
   extends to 100 m. Beyond ~40 m one stride-8 feature row covers 271–903 label rows (INHERITED, analytic). The CE
   carries no range weighting and no tolerance/boundary term.
6. **The box head cannot be made bigger without a rule change.** A pre-registered 2–4 M parameter band is enforced at
   construction, and no flag reaches it.
   - MEASURED (source): `agent_slots.py:195`, `box3d_head.py:259-274`.
   - Today's decoder has ~3.86 M (INHERITED). One more decoder layer (~1.05 M, ESTIMATED) is REFUSED.
7. **The box head is a 3-layer DETR that does not de-duplicate.** It has a single-scale memory, learned per-token
   positions and no camera-ray encoding.
   - ~2.14 boxes per object, and the 0.5 gate is unusable (recall 0.013). INHERITED, R8-PERC8-1.
   - Its 300 learned anchors, parametrised in METRES at lr 1e-4, cannot move more than ~5 m in a whole run
     (ESTIMATED).
   - In refcv7 they moved a median 0.031 m (INHERITED).
8. **Nothing in the planner reads either improved head today.** The planner reads the *agent* head (stride 32, no
   BEV) and a pooled BEV, never the box3d slots (INHERITED, R8-PERC-WIRING-1). Injecting a near-perfect GT map was
   INERT (INHERITED, R8-P4-GTMAP).
   - ⇒ A perception redesign needs a **wiring** clause, or a "dramatically better" head will not move driving.

The 2026-09-21 **class collapse** was diagnosed as an objective failure. Its fix (`--agent-cls-weight`) **landed and
is ON in both refcv7 and refcv8**: commits `d014414`, `ff9a073` and `e152e40` are ancestors of `fec3a0d` and
`fb47a58` (MEASURED, `git merge-base --is-ancestor`). Its effect is partial: person, rider and bus now score AP, and
the rare classes are unresolved (INHERITED). See §3.6.

---

## 1. The perception data path

### 1.1 Input, projection, crop

| item | value | evidence |
|---|---|---|
| cache | `refcv8-b1-416x1024-train-4572`, PNG-coded v2ep, **built at the target geometry** (the loader has no resize, crop or interpolate) | argv `--v2-cache`; MEASURED (source): 0 hits for `interpolate`/`resize` in `v2_dataset.py` (only the docstring's word "crop") |
| geometry | **416 × 1024 cylindrical**, 120.0° HFOV, `f_ref` = 305.577 × 1024/640 = **488.92 px/rad**, VFOV 46.09° | argv `--image-hw 416 1024`; `trunk_shapes.py:73-84, 97`; the projection is the exact inverse of `calib.cylindrical_rays` (`bev_lift.py:17-28`) |
| source | the native f-theta fisheye, rectified onto the cylindrical ray fan at cache build (`calib.cylindrical_rectify`) | `bev_lift.py:19-24` |
| two rigs | cy ~543 / ~755. The per-clip extrinsic carries the pitch, and 57.73 % of clips carry a 26–43-row black strip | `bev_lift.py:24-26`; trainer help `refc_v3_train.py:11651-11658` (INHERITED %) |
| C26 equalisation | **the bottom 43 of 416 rows (10.3 %) are zeroed** on every frame, train and eval, and marked UNOBSERVED in the lift | argv `--equalize-bottom-rows 43`; `refc_v3_train.py:11651`; `refcv6_perception_branch.py:381-385` |
| ⇒ nearest visible ground | **4.31 m** at the 1.45 m median mount (3.6–5.0 m over mounts 1.21–1.67 m) | ESTIMATED: `r = f·h / (207.5 − 43)`; agrees with WP-D's 4.3 m (INHERITED, `…/2026-10-04-refcv8-perception-architecture/RESULT.md` §3) |
| frame stack | **K = 3** consecutive 10 Hz frames (0.2 s) per row; window W = 8 rows → 10 distinct frames per sample (dedup) | `--trunk-dedup-frames` help, `refc_v3_train.py:12305` ("refcv6 10, not 24") |
| normalisation | ImageNet mean/std after the equalisation | `timm_trunk.py:823-847` |

### 1.2 Augmentation: none, of any kind

| kind | state | evidence |
|---|---|---|
| image (flip, colour, crop, scale) | **OFF: does not exist** | MEASURED (source): `augment\|flip\|jitter` → 6 hits in `refc_v3_train.py`, **all unrelated** (e.g. "flipping 29.82 % of picks"); **0** in `v2_dataset.py`; 1 unrelated in `refc_v3.py`. Same-breath control: `equalize` → 29 hits in the same file |
| BEV (rotation, flip, translation with GT; the BEVDet "BDA" family) | **OFF: does not exist** | same grep; the lift geometry is per clip and fixed (`refcv6_perception_branch.py:356-385`) |
| extrinsic / pitch jitter | **OFF: does not exist** | same |
| temporal (frame drop, stride jitter) | **OFF** | same |

### 1.3 The trunk

| item | value | evidence |
|---|---|---|
| model | `timm` **`resnet101.a1_in1k`**, ImageNet weights (it REFUSES to build if they did not load); 42,500,160 params | argv `--trunk timm --trunk-name resnet101.a1_in1k`; `refc_v3_train.py:12254-12262` |
| taps | **s8 = layer2, 512 ch, 52×128**: the MAP only, current frame. **s16 = layer3, 1024 ch, 26×64**: box3d memory. **s32 = layer4, 2048 ch, 13×32 = 416 tokens**: the planner and the agent head. There is no FPN. | `timm_trunk.py:727-757`; `refc.py:4776-4785`; `refc.py:4329-4332, 5100-5101` |
| frames per tap | s16/s32: the K = 3 frames each run through the shared trunk, fused AFTER it by `TemporalFuse(concat1x1)`, identity-initialised on the newest frame. **s8: newest frame only, `s8_temporal_fusion: None`.** | argv `--trunk-mode shared --trunk-fuse concat1x1`; `timm_trunk.py:345-399, 757, 803` |
| window rows used by perception | **only the LAST row** of the 8-row window (`[:, -1]`); the other 7 rows feed the trunk but no perception loss | `refc.py:4776-4785` |
| BatchNorm | **frozen at ImageNet running stats** and folded into the conv; γ/β still train; BN re-calibration OFF | argv `--trunk-frozen-bn --trunk-fold-bn`, no `--trunk-bn-recalib`; `refc_v3_train.py:12283-12304, 12331` |
| speed levers | bf16 backbone, channels_last, `torch.compile`, chunked checkpointing N = 8 | argv; `refc_v3_train.py:12271` |

### 1.4 Optimiser, LR, epochs

| item | value | evidence |
|---|---|---|
| optimiser | `--opt dd`: AdamW, wd 1e-4, two groups: `core.encoder.*` (the trunk incl. TemporalFuse) at **lr × 0.5**, everything else (all heads, the BEV branch, the anchors) at lr | `refc_v3_train.py:8520-8536`; `timm_trunk.py:1316-1356`; argv `--lr 1e-4` (encoder-lr-mult default 0.5, `refc_v3_train.py:12961`) |
| peak LR | trunk **5e-5**, heads **1e-4** | same |
| schedule | linear warm-up 1,000 steps, then cosine to 0 at 48,852 | `refc_v3_train.py:10354-10356`; argv `--warmup 1000 --steps 48852` |
| batch / steps | 16 × 48,852 = 781,632 samples over **781,635** windows = **1.000 epoch** | MODEL_REGISTRY §16 (INHERITED window count from the trainer log) |
| prior training | warm start from refcv7-50,400, itself trained **from ImageNet** (no `--init-from`) for 50,400 × 16 over 746,946 windows = **1.08 epochs** | MEASURED (artifact): `stack/ops/runs.d/refcv7-r101-s0.argv.json` has no `--init-from`; 1.08 epochs INHERITED (WP-D §1.1) |
| ⇒ total perception exposure | **≈ 2.08 passes**. One supervised frame per window (the current one), on ~4.4–4.6 k distinct 20 s clips (171 windows/clip, overlapping ~95 %) | ESTIMATED |
| sampler | per-epoch `randperm`, no weighting or stratification | `refc_v3_train.py:8392`; INHERITED (`…/data_coverage_2026-10-12/RESULT.md` §2) |
| EMA | 0.9998 (eval reads EMA) | argv |

**Is the trunk LR "starved"?** No. 0.5× is DiffusionDrive's recipe and not small for a detection backbone. The trunk is
starved of **map gradient**, by loss balance (§5), not by LR.

---

## 2. The BEV lift(s)

### 2.1 Type

- **Parameter-free geometric sampling (Simple-BEV style), not LSS.**
  - Every BEV cell centre is projected through the clip's camera at **4 fixed heights above the road plane: 0, 0.5,
    1.5 and 2.5 m**.
  - The projection is cylindrical, exact and per clip. The stride-s feature map is bilinearly sampled at those
    points, and invalid samples are zeroed.
  - The heights are concatenated and passed through a 1×1 conv. A learned "unobserved" embedding fills cells with no
    valid height.
  - MEASURED (source): `bev_lift.py:69, 146-184, 226-270`.
- **No depth bins, no depth distribution, no depth supervision.** The corpus ships `lidar_top_360fov`. A projected
  depth target is "buildable, not absent" (INHERITED, WP-D §6). Nothing builds it.
- **Validity.** A cell is valid when it is in front of the camera, inside ±60°, inside the image rows, and not in the
  zeroed strip (`bev_lift.py:32-35, 90-105`). Validity is **not** an occlusion test.
- **Geometry is per clip, built once.** It uses the static sensor extrinsic (`refcv6_perception_branch.py:356-385`;
  "the per-clip pitch comes from the extrinsic", `bev_lift.py:26`). **No per-frame pitch, roll or road grade.** The
  egomotion is not used by the lift.

### 2.2 The three grids (only one lift is built in refcv8)

| grid | what | extent | cell | shape | status in refcv8 |
|---|---|---|---|---|---|
| **0.5 m main map** (refcv6) | stride-16 `BEVLift` (d 128) → `BEVEncoder` (96 ch, dilations 1-2-4-8) → 9-class soft-CE head | 60 m × ±16 m | 0.5 m | 120 × 64 | **NOT BUILT**: `--w-map 0` + `--bev-source map_hires_pool` (`refcv6_perception_branch.py:233-243, 472-488`; `bev_encoder.py:107-111`) |
| **0.25 m lift = THE shared BEV** | stride-8 `BEVLiftProjectFirst` (2048 → **64**) → `HiresBEVEncoder` (**64 ch**, 3×3 stem + 5 residual dilated blocks d = 1, 2, 4, 8, 16; RF ±15.75 m) | **100 m × ±30 m** | 0.25 m | **400 × 240** | ON. Feeds (i) the 10 cm map decoder and (ii) `PlannerBEVPool` (`map_head_hires.py:1-20, 352-357, 495-536, 571-597`) |
| **10 cm map** (`_map_hires`) | `HiresRefine`: 1×1 → 32 ch, bilinear ×2.5, two 3×3 convs (RF 0.5 m), 1×1 → 8 logits; plus the **near lift**: the stride-8 map lifted again at 0.1 m over x ∈ [0, 20) m (32 ch, zero-init), added before the convs, plus **1 near refine block** (dilations 2, 4) | 100 m × ±30 m (near lift 0–20 m, full width) | 0.1 m | **1000 × 600** logits | ON (`--map-hires-near-lift-m 20 --map-hires-near-refine-blocks 1`; `map_head_hires.py:600-639, 712-777`) |
| planner window | crop 60 m × ±16 m of the 0.25 m BEV (240 × 128) → 2×2 avg-pool → 1×1 + GN to 96 ch | 60 m × ±16 m | 0.5 m | 120 × 64, plus **30 × 16 tokens of 2 m** (adaptive avg-pool) | ON (`refcv6_perception_branch.py:293-353, 638`; argv `--bev-planner-crop-m 60 16`) |

Parameter count of the map branch (ESTIMATED, arithmetic from the module definitions):

| part | params |
|---|---|
| lift | 131,200 |
| encoder | 406,912 |
| 10 cm decoder | 20,936 |
| near lift | 65,600 |
| near refine | 18,496 |
| **total** | **≈ 643 k** |

The lift + encoder sum, **538,112**, equals P-GRAD's MEASURED group size `x_shared_bev_025m`. That match is the
cross-check.

### 2.3 Temporal fusion

- **Image level:** K = 3 frames over 0.2 s, at s16/s32 only (§1.3).
- **BEV level: none.** No ego-motion-aligned BEV history (BEVDet4D/BEVFormer-style). The map and the planner BEV come
  from one frame's layer2.

---

## 3. The box head (`box3d`) and its sibling (the agent head)

### 3.1 Architecture

| item | value | evidence |
|---|---|---|
| type | **query-based set prediction (DETR)**, not a dense heatmap. The HQS heatmap mode exists but is OFF | `box3d_head.py:227-301`; argv `--slot-query-select learned_ref` (`refcv6_perception_branch.py:519-521`) |
| memory | **s16 current-row tokens 26 × 64 = 1,664** (layer3, K-fused) + **480 BEV tokens** (the 120 × 64 planner BEV avg-pooled to 30 × 16, i.e. **2 m cells**), each linearly projected to 256 + a 2-way source embed | `box3d_head.py:132-221`; `refcv6_perception_branch.py:495-499, 673` |
| positional encoding | a **learned per-token table** `mem_pos` [1, 2,144, 256]. **No camera-ray / 3D position encoding** (PETR-style absent) | `agent_slots.py:636-639` |
| decoder | `nn.TransformerDecoder`, **3 layers**, d 256, 8 heads, FF 1024, pre-norm, dropout 0, global (non-deformable) cross-attention, single scale | `agent_slots.py:643-645`; `refcv6_perception_branch.py:500-503` |
| queries | **300** (`N_QUERIES_DEFAULT`), anchored: `learned_ref`. One trainable (x, y) per query, **in metres**, seeded uniform over 60 m × ±16 m; centre = anchor + tanh(raw) × **4 m**; the anchor also gives a sine position embedding at every layer | `agent_slots.py:224`; `slot_query_select.py:36, 147-183, 234-268` |
| outputs | presence, 10-class cls, (cx, cy, l, w) in m, yaw (sin, cos), relative velocity + yaw rate, occluded, **z, h** (appended) | `agent_slots.py:167-177`; `box3d_head.py:96-99, 277-300` |
| decode range | **x 0–60 m, y ±16 m** (`GRID_DEFAULT`) | `agent_slots.py:545-559` |
| size cap | **`PARAM_BAND` 2–4 M, enforced at construction; no argv reaches it.** Current: 3,858,199 (INHERITED from the `agent_slots.py:218-221` comment) | `agent_slots.py:195`; `box3d_head.py:259-274`; `refcv6_perception_branch.py:175-181` |

### 3.2 Targets and assignment

- **GT.** Targets are `obstacle.offline` cuboids (auto-labels) via the 3D join `b1_train_plus_eval_agents_3d.jsonl.xz`:
  cx, cy, cz, l, w, h, yaw and class. The rates are finite differences per track.
  - Id-switch frames are masked by `--join-defect-masks`, which is on in refcv8 (argv; `refc_v3_train.py:11991`).
- **Visible filter, ON by default and applied BEFORE matching.** Only boxes inside the ±60° field AND the decode box
  survive. That is **16.374 %** of raw join boxes (INHERITED, `box3d_head.py:350-356`; `refc_v3_train.py:12414-12425`).
- **VIS-1.** Targets are visibility-graded (POS / IGNORE with an exemption radius). Argv `--slot-vis1 --vis1-sidecar`.
- **Matcher.** Hungarian, one-to-one.
  - Cost = focal presence cost × 2.0 + cls × 1.0 + centre L1 (m) × 1.0 + size L1 (m) × 0.5.
  - z and h are NOT in the cost (`MATCH_INCLUDES_Z = False`).
  - Before the field cut the target set is the n_queries nearest (no-op at 300).
  - Evidence: `agent_slots.py:228-236, 847-892`; `slot_presence.py:64`; `box3d_head.py:114`.

### 3.3 Losses (per decoder layer, deep supervision on all 3)

| term | form | weight | evidence |
|---|---|---|---|
| presence | **sigmoid focal** α 0.25, γ 2, prior 0.01 | 2.0 | argv `--slot-presence-loss focal --slot-presence-prior 0.01`; `slot_presence.py:61-66` |
| cls | CE over 10 classes, **class-weighted (`b1`, inverse frequency, mean 1), weighted-mean denominator** | 1.0 | `agent_slots.py:238-241, 961-975`; `refc_v3_train.py:9196-9206` |
| centre | L1 in **metres** (not range-normalised) | 1.0 | `agent_slots.py:949` |
| size | L1 in metres (l, w; not log) | 1.0 | `agent_slots.py:951` |
| yaw | 1 − cos Δ | 1.0 | `agent_slots.py:953-959` |
| rates | L1 on v_rel x/y, yaw-rate | 0.5 | `agent_slots.py:238-241` |
| occluded | BCE | 0.5 | same |
| z, h | L1 (m) on matched rows with 3D labels | 1.0 each | `box3d_head.py:126, 425-445` |
| **head total** | added to the loss as **`--w-box3d 1.0` ×** the row | | `refc_v3_train.py:6618-6624` |

INHERITED (WP-D §1.2, refcv7 end): the rates term was the largest single box term (~26 % of a layer's loss) before
the id-switch masks.

### 3.4 Decode, NMS, gates

- **No NMS in the model or in the training loss.** One-to-one matching is meant to de-duplicate. `--det-nms` exists
  only as an opt-in eval report and is NOT in refcv8's argv (`refc_v3_train.py:12064-12070`).
- The declared detection gate is presence 0.5 (`slot_presence.py:68`). INHERITED (R8-PERC8-1): refcv8 EMA-48,852
  recall at 0.5 = **0.013**, i.e. unusable. 2.14 boxes per detected object at the P = R gate, 1.088 after a
  train-fitted 2.5 m NMS.

### 3.5 The AP@2 m evaluator

`box3d_ap` (`box3d_head.py:461-547`):
- **class-agnostic**;
- greedy score-ordered matching at a **3D centre-distance** threshold (2 m, with z; nuScenes-style distance, not IoU);
- scored by **presence**;
- all-point interpolated AP;
- over the VIS-1 / visible target set, inside the 60 m × ±16 m decode box.

Self-tests pin perfect → 1.0 and random → a closed-form base rate. A class-mean mAP is logged beside it. The planner
does not consume these boxes (§0.8).

### 3.6 The 2026-09-21 class collapse: cause, fix, status

| | |
|---|---|
| finding | `GOALS_AND_CLAIMS.md:13897-14100` (A8 `ckpt_5000`): box3d emitted **1 class of 10** on 2,000/2,000 slots; accuracy = the majority baseline 0.77935; 188 persons and 50 riders emitted as cars. The second (agent) head collapsed identically (320/320). Size was a global constant downstream of it. A linear readout separated VRU from vehicle within a scene (AUC 0.583), so this is an **objective** failure: an unweighted `cls` CE on a 577.5:1 target. (INHERITED) |
| fix | `--agent-cls-weight` (H-BOXCLS-1): class-weighted `cls` on BOTH heads. **`d014414`** (2026-09-22, plumbed to both loss paths), **`ff9a073`** (re-based onto the B1 corpus line), **`e152e40`** (2026-09-23, the in-field population vector). MEASURED: all three are ancestors of `fec3a0d` (refcv7) and `fb47a58` (refcv8). `--agent-cls-weight b1` is in **both** launch argvs |
| weights in force | `stack/tanitad/data/agent_cls_weights_b1.json`, `weights_inv_freq_mean1_visible`: automobile **0.0043**, person 0.0115, rider 0.097, heavy_truck 0.214, trailer 0.340, bus 0.480, protruding 0.522, stroller 1.245, other_vehicle 1.532, animal **5.554**. Visible imbalance **1,298.6 : 1**. Pure inverse frequency, no sqrt / effective-number tempering (MEASURED, artifact) |
| status | **PARTIALLY RESOLVED.** R8-PERC8-1 (`GOALS_AND_CLAIMS.md:17006`, INHERITED): class-mean AP@2 m 0.102 (refcv8) vs 0.070 (refcv7); "the gain is in automobile, person, rider and bus, and the rare classes are unresolved". A direct distinct-classes / confusion count on refcv7 or refcv8 was **not found** → the third agent's per-class AP answers it |

---

## 4. The map head(s)

### 4.1 Representation

**Classes.** There are 8 classes, plus 255 for "not seen". They are **mutually exclusive, one code per cell**:

| code | class |
|---|---|
| 0 | seen / no map class |
| 1 | drivable |
| 2 | lane / road line |
| 3 | crosswalk |
| 4 | arrow / text |
| 5 | non-drivable edge |
| 6 | hatched area |
| 7 | sidewalk / verge |

- Evidence: `semantic_map_gt.py:73-75`; `semantic_map_gt_fine.py:115-119`.
- The GT world map stores a **single** class per 0.1 m cell. The exporter does a nearest lookup
  (`…/2026-09-26-map-signal-audit/code/sam3map_export_gt_v3.py:79-83`).
- ⇒ lane / crosswalk / arrow cells are **not** drivable cells.

**Raster.** 1000 × 600 at 0.1 m over 100 m ahead × ±30 m. The extent was selected by the GT-coverage census, A7
(`semantic_map_gt_fine.py:291-300`; argv `--map-hires-x-max-m 100 --map-hires-y-half-m 30`).

**Which map the planner reads.** No map *decision* reaches the refcv8 planner. It reads the **pooled 0.25 m BEV
features** (60 m × ±16 m → 120 × 64 @ 0.5 m, plus 30 × 16 tokens) through the BEV coupling and the tactical decoder
(`--bev-source map_hires_pool --bev-coupling`). It never reads the 10 cm logits.
- ⇒ **68 %** of the BEV area the map is trained on (60 × 32 m kept of 100 × 60 m) is read by no consumer except the
  map loss (ESTIMATED).

### 4.2 Losses and decoding

**Loss.** `hires_map_ce` is **hard-label cross-entropy**. Its settings:
- `ignore_index` 255;
- mean over supervised cells, with the weights in the denominator;
- supervised cells narrowed to the lift-valid cells of this instant (`--map-lift-valid-mask`, default ON);
- **no boundary, Dice/Lovász, tolerance, range or occlusion term.**
- Evidence: `map_head_hires.py:917-951`; `refc_v3_train.py:5087-5160, 12426-12435`.

**Class weights.** `sqrt_mf`, i.e. the square root of median-frequency balancing:

| class | weight |
|---|---|
| no-class | 0.141 |
| drivable | 0.206 |
| lane | 0.939 |
| crosswalk | 1.074 |
| arrow | 2.957 |
| edge | 1.571 |
| hatched | 2.153 |
| sidewalk | 0.165 |

- Class frequencies: lane 1.02 %, edge 0.36 %, arrow 0.10 %.
- The weights were counted over **4,369** train clips (the refcv6 view): MEASURED (artifact)
  `…/2026-09-26-refcv7-map-hires/raw/map_hires_class_weights_train_100x30.json` + `.PROVENANCE.json`
  (`n_clips_used 4369`); argv `--map-hires-class-weights`.

**Decision rule.** `prior_corrected` is the declared rule: argmax(z − log w) (`map_head_hires.py:125, 144-187`; argv).
- It is used for the in-run IoU and the eval, never in the loss.
- INHERITED (REFCV7-MAP-RULE, `GOALS_AND_CLAIMS.md:16267`): this rule **hid** the thin classes (edge IoU ≤ 0.0008)
  while raw argmax read 0.025–0.031, and train-fitted per-class thresholds read lane 0.164 / edge 0.042.
- refcv8's argv still declares it. The thresholds file exists but is not passed.

### 4.3 How the SAM3 GT is built

| step | fact | evidence |
|---|---|---|
| source | **front camera only** (`CAM_F0`), native f-theta frames at **5 Hz**, SAM3 masks; **ground from the ego path, no LiDAR** | `…/Data Engineering/Research/2026-09-13-sam3-only-road-map/code/corpus/sam3map_prod.py:1-6, 24-25` |
| world map | per clip, composed at **0.1 m** (`world_map_res_m`); **non-causal**, built from every frame of the clip | `semantic_map_gt_fine.py:61-71, 120-121` |
| per-frame raster | on the v2ep 10 Hz axis, rig pose at each frame's exposure interpolated from egomotion; nearest lookup of the world map at each 10 cm rig-cell centre; "seen" = seen anywhere in the clip | `sam3map_prod.py:8-10`; `sam3map_export_gt_v3.py:79-83, 1-15` |
| /3 re-export | 100 m × ±30 m; byte-identical to /2 inside the old 60 × ±16 window (checked per frame) | `sam3map_export_gt_v3.py:1-20` |
| coverage of refcv8's 4,572 train clips | **4,369 have GT (95.6 %)**. **203 have none**: 58 "no validated map" (stationary / intersection, HOLD/CREEP-rich) and 145 "no agent join" (highway/fast) | INHERITED, `…/2026-10-04-refcv8-tactical-conditioning/data_coverage_2026-10-12/RESULT.md` §1, §5 (`sam3_gt_v3` holds 4,506 = 4,369 + 137 eval) |
| projection consistency | lift and GT share the per-clip extrinsic and the rig frame. They differ in ground model: GT uses the ego-path ground; the lift uses a **flat** rig plane with a **static per-clip pitch** | `bev_lift.py:26, 146-184` vs `sam3map_prod.py:3-4` |

### 4.4 Range vs image support

Every number here is INHERITED, analytic, from `…/2026-10-04-refcv8-perception-architecture/RESULT.md` §3 at h 1.45 m
and f 488.92. I re-derived the ground depth per image row at 30 m as 30²/(1.45 × 488.92) = **1.27 m**, which matches.

| band | stride-8 rows | 10 cm label rows per s8 row | 0.15 m line width in px |
|---|---|---|---|
| 0–20 m | 21.5 | 9 | 7.3 |
| 20–40 m | 2.2 | 90 | 2.4 |
| 40–60 m | 0.74 | 271 | 1.5 |
| 60–80 m | 0.37 | 542 | 1.05 |
| 80–100 m | 0.22 | 903 | 0.81 |

### 4.5 The new `--w-map-fp` term (`aeb5446`, 2026-10-27, refcv9 only)

**Definition.** L_fp is the mean, over **GT-drivable** cells of a near corridor, of **p_sidewalk + p_edge** (softmax
over the 8 classes).
- The corridor is x ∈ [0, 20) m, |y| < 3 m, i.e. rows 0–199 × cols 270–329 = 12,000 cells.
- The value is pooled over the labelled windows of the batch.
- It is added as w · L_fp to the 10 cm map term.
- It is **NOT narrowed by the lift's validity mask**, and that is deliberate.
- Default 0 means off and bit-identical. It is **not in refcv8**.
- Evidence: `stack/tanitad/train/map_precision.py:1-28, 84-102`; `refc_v3_train.py:12735-12738, 5143-5171`.

**Motivation (INHERITED, docstring).** At 0–5 m the own map's sidewalk flags are 0.261 precise, and 0.62 of them sit
on GT-drivable cells.

**Caveat (ESTIMATED).**
- The camera sees no ground closer than **~4.3 m** (§1.1). So corridor rows 0–42, **21.5 %** of the corridor, carry
  **no image evidence**.
- There the term can only learn a positional prior: "never sidewalk or edge in front of the ego". That prior also
  suppresses a REAL kerb the ego faces (T-junction, parking).
- It is a one-sided term: it only penalises sidewalk or edge on drivable cells, never the miss.
- See D16.

---

## 5. Loss balance in refcv8's argv, and who owns the trunk gradient

### 5.1 Weights as launched

**Perception.**

| term | weight | evidence |
|---|---|---|
| `--w-agent` (the planner-read agent head) | 1.0 | `refc_v3_train.py:6152-6158` |
| `--w-box3d` | 1.0 | |
| `--w-map-hires` | 1.0 | |
| `--w-map` | **0** (the 0.5 m map is removed) | |
| `--w-map-fp` | absent = 0 | |
| `--w-tac-v6` (the tactical decoder, reads BEV + agent tokens) | 1.0 | |
| `--bev-aux` | off | |

**Planner.**

| term | weight | evidence |
|---|---|---|
| TRAJ | 1.0 | `stack/scripts/refc_train.py:76-91`; `refc_v3_train.py:5794-5799` |
| ANCHOR_CLS | 1.0 | same |
| LAW | 0.5 | same |
| ROUTE | 0 (no-strategic) | same |
| LAT / LON tactical aux | 0.05 / 0.05 (halved) | same |
| `--w-r8-cons` | 0.05 | argv |
| `--w-r8-alloc-l1` | 1.0 | argv |
| `--w-r8-sat` | 0.1 | argv |
| `--w-r8-subscore` | 0.5 | argv |
| `--w-r8-v9-cons` | 0.05 | argv |
| `--w-r8-listwise` | 0 | argv |

Loss VALUES at refcv7's end (INHERITED, WP-D §1.4): map 0.91 of a ~36 total (~2.5 %), box3d 14.7 (~41 %).

### 5.2 Gradient share: what exists, and what is blocked

**(a) refcv8 in-run readings at 1,000 / 16,250 / 32,500 / 48,750: NOT FOUND in the repo or on this box. BLOCKED.**
- Probes I ran:
  - `git grep` for `gs_trunk_proj_*` / `gs_bev025_proj_*` over the whole ref. Only the 2026-10-11 derisk smoke's
    `metrics.jsonl` carries them. Positive control: the same grep finds the keys there.
  - `git ls-tree` for refcv8 `metrics*.jsonl`: only NavSim rows.
  - MODEL_REGISTRY §16 / §16.1 list the schedule only.
  - E:/F: (`/e/r8launch/gs_measure.json` = timing of one reading, 775 s).
- They live only in Thor `/home/nvidia/refcv8_run/runs/refcv8-r101-s0/metrics.jsonl` (registry §16 "where"). This
  brief forbids Thor. **Unblock:** one `grep '"gs_' metrics.jsonl` by an agent with Thor access.
- ⚠️ **Even that log cannot answer the brief's question.** The run passed `--grad-share-terms tac_v6,map_hires`, and
  `grad_share.py:21-26` folds every other term (box3d, agent, the planner) into `rest`. So it separates map vs
  everything-else, never box vs agent vs planner.

**(b) The per-term split that DOES exist.** P-GRAD on **refcv7-50,400**, refcv8's warm-start weights.
- Setup: weights 1.0, 64 windows, fp32, TF32 off.
- Controls: linearity 6.6e-6 ≤ 1e-4, PASS; zero-weight exactly 0, PASS.
- MEASURED (artifact, re-read): `…/2026-10-04-refcv8-perception-architecture/raw/pgrad.json` → `summary`.

Projection share of the total update:

| parameter group | traj | box3d | agent | map_hires | tac_v6 | rest (planner aux) |
|---|---|---|---|---|---|---|
| **whole trunk** | 0.0002 | **0.357** | **0.501** | **0.0001** | −0.0005 | 0.142 |
| layer2 (the map's own tap) | 0.0007 | 0.327 | 0.452 | **0.0010** | 0.0004 | 0.218 |
| layer3 | 0.0002 | 0.356 | 0.475 | 0 | 0.0002 | 0.169 |
| layer4 | 0.0006 | 0 | 0.632 | 0 | −0.0001 | 0.368 |
| TemporalFuse | 0.0003 | 0.569 | 0.332 | 0 | 0 | 0.099 |
| **shared 0.25 m BEV** (lift + encoder) | 0.0005 | **0.973** | 0 | **−0.004** | 0.0000 | 0.031 |
| planner BEV pool | 0.0015 | 0.915 | 0 | 0 | 0.0065 | 0.077 |

Gradient NORMS (same file):

| group | map_hires | box3d | agent |
|---|---|---|---|
| layer2 | 1.12 | 73.7 | 92.3 |
| shared BEV | 4.64 | 69.6 | — |

**(c) refcv8 full-size smoke at step 50.** fp32 replay, trunk group. Linearity 4.2e-4 FAILED its 1e-4 bar, so the
values are indicative only.

| agent | box3d | rest | traj | tac_v6 | map |
|---|---|---|---|---|---|
| 0.436 | 0.316 | 0.248 | 0.0010 | 0.00028 | −0.00063 |

INHERITED, `GOALS_AND_CLAIMS.md:16631` (G-SMOKE-L4).

**(d) Ladder rig (3,000-step arms, not the main run).** INHERITED, `GOALS_AND_CLAIMS.md:16705` (H-R8-MAPW-1).

| arm | `gs_bev025_proj_map_hires`, last-50 % median |
|---|---|
| V-R8 | 0.175 |
| `--w-map-hires 4` | 0.833 |

Box/agent AP got worse at weight 4. ⇒ Map weight alone buys the BEV but costs the boxes, because the BEV is shared.

**Reading.**
- The trunk is owned by the two detectors, 86 % between them. That makes the agent head, which the planner reads, the
  single largest trunk driver.
- The map is **two to three orders of magnitude** below both, even at its own layer2 tap.
- The trajectory loss is ~0.02 %.

---

## 6. Defects and suspicious choices (ranked by severity guess)

| # | finding | evidence (file:line / artifact) | sev. | cheapest check |
|---|---|---|---|---|
| D1 | **The map has ~0 trunk gradient and −0.4 % of its own BEV encoder's update; box3d drives the BEV (97 %)** | `pgrad.json` (MEASURED, §5.2b) | **HIGH** | P-GRAD replay on refcv8 EMA-48,852 (code exists: `…/code/pgrad.py`; minutes of GPU). Or read the Thor `gs_bev025_*` rows |
| D2 | **The map reads layer2 only, current frame only.** No FPN (layer3/4 semantics never reach the map), no temporal or BEV fusion, while the GT is a multi-frame product | `map_head_hires.py:849`; `timm_trunk.py:757, 803`; `refc.py:4776-4781` | **HIGH** | frozen-feature probe: s8 alone vs s8 + up-sampled s16/s32 on the same lift (fits the third agent's probe) |
| D3 | **The shared BEV is 64 ch / 538 k params** for map + planner + box tokens + tactical; the 10 cm decoder is 32 ch | `map_head_hires.py:352-357`; `pgrad.json` group size | **HIGH** | width arm 64 → 128/256 on the MAPFT kit (`FlyWheels/…/2026-10-26-refcv9-mapft/kit`) |
| D4 | **Flat-plane multi-height sampling, no depth, no depth supervision**; the residual box error is longitudinal (\|Δx\| 0.96 / 1.56 m vs \|Δy\| 0.38 / 0.46 m at 0–20 / 20–40 m, INHERITED WP-D §2.5) | `bev_lift.py:69, 146-184, 226-270` | **HIGH** (boxes), MED (map) | build the LiDAR-projected depth target on 50 clips; a depth-aux probe on frozen s16 |
| D5 | **Static per-clip pitch; no per-frame pitch, roll or grade** in the lift. A 1° error at 40 m → sampled ground 27–77 m (ESTIMATED, h 1.45, f 488.92, 1° = 8.53 px) | `bev_lift.py:26, 146-184`; `refcv6_perception_branch.py:356-385` | **HIGH** beyond ~30 m | CPU: per-frame egomotion pitch/roll minus clip mean over the 4,369 clips → implied range error per band |
| D6 | **The box head is size-capped** (2–4 M at construction, no flag); one more decoder layer is refused | `agent_slots.py:195, 653`; `box3d_head.py:259-274`; `refcv6_perception_branch.py:175-181` | **HIGH** (redesign blocker; the band is a "probe the latent" rule, not a detection rule) | a PI / Master-Mind ruling to retire the band for refcv9; no compute |
| D7 | **Box decoder: 3 layers, single-scale global attention, learned token positions (no ray / 3D PE), no iterative refinement, no NMS** → 2.14 boxes/object, objectness AUROC 0.890, 0.5 gate recall 0.013 | `agent_slots.py:636-645`; `box3d_head.py:205-221`; INHERITED R8-PERC8-1 | **HIGH** | already measured; NMS + train-fitted gate alone gives AP@2 m 0.447 (INHERITED) — the floor any redesign must beat |
| D8 | **Box and planner windows = 60 m × ±16 m.** 6 s anchors exceed 60 m above **10 m/s**; only 16.4 % of raw join boxes survive the field + decode-box cut | `agent_slots.py:545-559`; argv `--bev-planner-crop-m 60 16`; `box3d_head.py:350-356` (INHERITED %) | **HIGH** (highway) | CPU: fraction of train windows whose GT 6 s path or lead vehicle lies beyond 60 m / ±16 m |
| D9 | **The 10 cm CE is uniform over 0–100 m**: cells beyond 40 m with < 1 s8 row of support are supervised at full weight; no range weight, no tolerance or boundary loss → thin classes under-predicted (INHERITED R8-P2-MAPTOL) | `map_head_hires.py:917-951`; §4.4 | MED-HIGH | per-band loss share on one eval batch; a range-weighted / Lovász arm on the MAPFT kit |
| D10 | **Single-label softmax over overlapping semantics** (lane / crosswalk / arrow ⊂ drivable); "drivable" excludes its own markings | `semantic_map_gt.py:73-75`; `sam3map_export_gt_v3.py:79-83`; `map_head_hires.py:949` | MED | CPU on banked eval dumps: IoU(drivable ∪ markings) vs IoU(drivable) |
| D11 | **10 cm detail comes only from bilinear ×2.5 + two 3×3 convs (RF 0.5 m) + the 0–20 m near lift** | `map_head_hires.py:600-639, 712-777` | MED | covered by D2 / D3 arms |
| D12 | **Learned anchors in METRES at Adam lr 1e-4**: ≤ ~4.9 m total drift over 48,852 steps even with a constant gradient sign (Adam step ≈ lr; ESTIMATED); refcv7 measured 0.031 m (INHERITED) | `slot_query_select.py:147-157` | MED | CPU: refcv8 EMA `box_refpts.xy` vs the seed-0 init |
| D13 | **Pure inverse-frequency cls weights, 1,299 : 1** (car 0.0043): the cls gradient is dominated by the batch's rare classes; rare classes still unresolved | `agent_cls_weights_b1.json`; `agent_slots.py:961-975` | MED | per-class AP + confusion (third agent) |
| D14 | **Zero augmentation** (image, BEV, extrinsic) | grep, §1.2 | MED (rises with the no-cap multi-epoch run) | none needed: a known absence |
| D15 | **GT: front-camera SAM3, ego-path ground (no LiDAR), non-causal, no occlusion mask; 203/4,572 train clips (4.4 %) unlabelled** (stationary / intersection skew) | `sam3map_prod.py:1-6`; data-coverage RESULT §1 | MED | the third agent's LiDAR adjudication; occlusion share from the `obstacle.offline` boxes vs GT-seen cells |
| D16 | **`--w-map-fp` corridor not lift-masked**: 21.5 % of its rows (0–4.3 m) are camera-invisible; a one-sided term learns "no kerb ahead" | `map_precision.py:15-17, 84-102`; §1.1 | MED (safety-flavoured) | CPU: lift-valid fraction of the corridor per window; GT-edge recall in the corridor before/after the term |
| D17 | **The planner reads the weaker detector and no map decision** (agent AP@2 m 0.131 vs box3d 0.248 in refcv7; GT-map injection INERT) | `refc.py:4329-4332, 5100-5101`; INHERITED R8-PERC-WIRING-1, R8-P4-GTMAP | **HIGH** for the programme | none; this is a design clause for refcv9 |
| D18 | **Training length**: ≈ 2.08 passes total; box AP was still rising at LR → 0 in refcv7 (INHERITED WP-D §1.2) | §1.4 | MED (the cap is lifted) | none |
| D19 | The declared map decision rule `prior_corrected` is known to hide thin classes; it is still in refcv8's argv | argv; `GOALS_AND_CLAIMS.md:16267` | LOW (logs only) | pass `--map-hires-class-thresholds` |
| D20 | Frozen ImageNet BN with a zeroed 43-row strip; BN re-calibration off | `refc_v3_train.py:12283, 12331` | LOW | `--trunk-bn-recalib` exists |
| D21 | `TemporalFuse` is identity-initialised; UNVERIFIED whether it learned to use the older frames | `timm_trunk.py:370-399` | LOW | CPU: norm of the older-frame blocks of `fuse.proj.weight` in refcv8 EMA |

**What the AS-IS says to change first (my reading, for the redesign, not a result).**
1. Give the map its own trunk signal and capacity (D1–D3).
2. Put geometry into the lift and the box memory: depth supervision, per-frame pose, ray encoding (D4, D5, D7).
3. Retire the head-size band (D6).
4. Match the label resolution to image support by range (D9).
5. Wire the planner to the boxes and the map decisions (D17). Without it, none of the above can move driving.

---

## 7. Deviations, limits, UNVERIFIED

- **The refcv8 in-run gradient-share readings were not read.** They are on Thor, and so blocked (§5.2a). Even if read,
  they do not separate box from agent from planner.
- **Parameter counts for the map branch are arithmetic, not an instantiation.** They are cross-checked by one MEASURED
  group size (538,112). The box decoder's 3,858,199 is a source comment, INHERITED.
- **The geometry numbers assume flat ground and zero dynamic pitch.** Mount heights vary 1.21–1.67 m per clip.
- **Line numbers in `refc.py` / `refc_v3_train.py` are at `56b25a61`, not the launch tree.** The perception modules
  are byte-identical to `fb47a58`; those two files are not.
- **Every INHERITED number is another agent's.** The ones that would decide a GPU-day (P-GRAD) I re-read from the raw
  artifact.

## 8. Deliverable manifest

| artifact | where |
|---|---|
| this file | hold `C:/Users/Admin/qland/hold_asis_perc/code/TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/ASIS_PERCEPTION.md` → repo path of the same name (NEW, not staged, not committed) |
| read-only source exports (scratch, not deliverables) | `F:/c_overflow_2026-10-26/tmp/asis_perc/src/` |
