# refcv6 — what must be clear before any plan is approved

**Master Mind, 2026-09-15 (Europe/Berlin).** Answers to your nine points of this evening, in your order.

**Status:** CLARIFICATION, not a plan.
- Nothing is launched, and no model code is changed.
- The 09-15 design proposal (`REFCV6_DESIGN_GROUNDED.md`) stays **unapproved**. Two of its statements are corrected below (§9, C1–C2).

**Evidence classes:**
- **MEASURED** — a programme artifact, or a check run today.
- **PUBLISHED-PRIMARY** — paper page / table.
- **PUBLISHED-CODE** — released code, `file:line`.
- **ESTIMATED** — arithmetic on cited inputs.
- **HYPOTHESIS**.

Every DiffusionDrive fact was re-read today in the banked papers (V1 arXiv 2411.15139, V2 2512.07745) and the released code (`…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/`).

Your words, verbatim, are recorded in `Project Steering/Decisions/2026-09-15-pi-directives.md`.

---

## The short answers

| # | you asked | short answer |
|---|---|---|
| 1 | How is DiffusionDrive's diffusion planner linked to the ResNet trunk and to perception, in both papers, vs our plan? | **V1:** an ImageNet ResNet-34 per camera and LiDAR builds a **supervised BEV map** (segmentation + agent boxes, trained jointly). Each candidate trajectory **samples that map at its own 8 waypoints**, then attends to 30 detected agents and one ego query. **V2:** the same trunk and perception, **frozen**, plus RL post-training and an 800-candidate selector. **REF-C** attends to 160 image tokens by content, with **no map, no agents and no perception loss**. §1 |
| 2 | How do we prove we can extract boxes, semantic maps and occupancy? Can the planner link directly to the trunk? | **Not proven yet:** occupancy AP 0.435 vs a 0.60 bar; boxes and maps never probed on refcv5-v2. A proof needs **two halves**: a probe that beats zero-information controls (*extraction*), and a mask of the lead vehicle / road edge that moves the plan more than a random mask (*use*). **Direct link: yes**, REF-C already does it. A five-arm experiment decides direct vs BEV-mediated. Dev box, ≈ 5 days. §2 |
| 3 | Nav command mandatory for the tactical and operative layers, in selection and trajectory | Nav reaches both today, but refcv5-v2 **barely obeys it in the plan**: it follows a flipped junction command 20.5 % of the time. Its tactical turn head is **a pure nav echo** (turn recall 0.475 → 0.000 without nav). Proposed wiring + four acceptance tests (T-FLIP, T-ZERO, T-FLOOR, T-REG). §3 |
| 4 | Tactical layer learns max speed and all tactical behaviours from the GT labels | The labels exist: speed band on all 4,572 clips, lateral/longitudinal actions, 17 of 22 goal tokens trainable. ⚠️ **The shipped "max-speed input" is the same number as the label**, so your 09-01, 09-10 and 09-15 statements need one ruling. Default: **the tactical layer predicts it, and a user limit only clamps selection.** §4 |
| 5 | Is the diffusion implemented as in the paper? Hypotheses, denoising | **Inference: yes** (noise at t = 8, 2 DDIM steps, x0 prediction), with 117 speed-conditioned control anchors instead of 20 k-means paths. **Training: no**: no random-t denoising objective, no per-layer loss, timestep only at the input. Our noise grows with horizon, where DD's is flat at 0.9 m. Nine changes listed (F1–F9). §5 |
| 6 | How is the ResNet trained and used, vs DiffusionDrive and other AD works? | **DD:** ImageNet ResNet-34 (21.8 M) fine-tuned end-to-end at half the learning rate, with perception losses, 100 epochs; **frozen in V2**. **Ours:** a random-init 90.5 M trunk, **one epoch of 25.5 h**, no perception loss. Every compared work pretrains its trunk. Our frozen trunk is ≈ raw pixels on a BEV probe. Measured today: its tokens **do** carry their position, and training concentrated its token features ≈ 4× relative to random init (participation ratio 16 → 4). §6 |
| 7 | Prepare and validate the RL approach as in the paper | **In progress on the dev box.** The paper-exact spec, the objective as a tested library, the log-probability chain our decoder lacks, a reward proxy and a ≤ 3 GPU-h pre-registered validation. §7 is replaced when verified; no result is claimed yet. §7 |
| 8 | Your computer for preparation and final design; a pod for heavy work | **Dev box:** ≈ 20 GPU-h of proof tests + all code, default-off, with tests. **Pod:** ≈ 230–320 GPU-h (≈ 3 days on 4 × 48 GB) once you have ruled. §8 |
| 9 | refcv6 first, then refav1, then v7 | Recorded as binding; refav1 and v7 wait for your refcv6 rulings. §8.3 |

---

## Decisions for you (defaults apply only to preparation; nothing trains at 108 M before you approve)

| # | decision | options | default if silent |
|---|---|---|---|
| **D1** | **Max speed** (§4.2) | **R2** tactical predicts, user limit clamps · R1 separate input with validity bit · R3 input dropout | **R2** |
| **D2** | Tactical / nav details (§4.3 a–g) | per row | as listed there: T-ZERO binding; tactical → operative detached; strategic unchanged; nav args constant per clip; loss inside the maneuver budget; vocabulary gaps accepted for refcv6; traffic lights excluded until queue item 1 |
| **D3** | **Run the proof package** on the dev box (§2.4, ≈ 20 GPU-h) | go · amend | wait for your go |
| **D4** | SAM3 maps of the 315 LiDAR-GT clips onto the dev box (≈ 1 GB) | copy from Thor · download from your private HF corpus | copy from Thor after production is past them |
| **D5** | Default-off flags in `refc.py` (link arms b/c, mask hook, tactical wiring) | allow · not yet | written and tested **in a worktree only**; landed after your approval |
| **D6** | `timm` + ImageNet ResNet-34 weights (≈ 87 MB) for the prior test H1 (§6.4) | download · skip | skip until you say |
| **D7** | refcv6 diffusion | **paper-faithful** (F1–F6) with ours as the knockout · ours with the paper form as the knockout | paper-faithful |
| **D8** | Trunk | T-A keep ours (warm) · T-B ImageNet ResNet-34 (paper-faithful) · both, paired | decided by H1; T-A if H1 is not separated |
| **D9** | RL per the paper | put to you when §7 lands | — |
| **D10** | Pod | 4 × 48 GB · 2 × 80 GB | 4 × 48 GB, after D1–D9 |

---

## 1. DiffusionDrive — how the planner is linked to the ResNet trunk and to perception

**Sources (all banked):**
- V1 paper arXiv 2411.15139 (`TanitAD Research Lab/Library/papers/`, PUBLISHED-PRIMARY);
- V2 paper arXiv 2512.07745 (PUBLISHED-PRIMARY);
- released code under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/` (PUBLISHED-CODE).

Every page, table and `file:line` below was re-read on 2026-09-15.

### 1.1 DiffusionDrive V1 (NAVSIM)

**Inputs**

| input | detail | source |
|---|---|---|
| camera | 3 front cameras, cropped and concatenated into **1024×256** | p.7; `diffusiondrivev2_rl_config.py:40-41` |
| LiDAR | rasterised to 256×256 BEV, ±32 m | `rl_config.py:28-29,42-43` |
| status | **driving command (4) + velocity (2) + acceleration (2)** | `transfuser_features.py:45-49` |

**Trunk (TransFuser)**
- One ResNet-34 per sensor: `image_architecture` / `lidar_architecture` `"resnet34"`, weights `resnet34.a1_in1k` (`rl_config.py:16-18`).
- The two branches are fused by transformer blocks at 4 scales.
- p.12: *"We initialize the ResNet-34 backbone with ImageNet pre-trained weights"*.
- p.7: the model is *"trained on navtrain split from scratch for 100 epochs"* (batch 512, lr 6×10⁻⁴). "From scratch" refers to the driving model; the backbone starts from ImageNet.

**Two outputs of the trunk**
- **8×8 BEV tokens:** 512-ch, projected to 256-d, plus the status token, plus a **learned positional embedding** on all 65 tokens (`transfuser_model_v2.py:36,40,108-113`).
- **Up-scaled BEV feature map:** used by the segmentation head and, after fusion with the tokens, by the planner (`:104-123`).

**Perception, trained jointly with the planner (one summed loss)**
- **Scene decoder:** 1 ego query + 30 agent queries, a 3-layer transformer decoder over the 65 tokens (`:124-128`; `tf_num_layers 3`, `num_bounding_boxes 30`).
- **Agent head:** 2-D BEV boxes + class. The paper says "3D detection"; the code imports `BoundingBox2DIndex`. Class weight 10, box weight 1 (`rl_config.py:87-88`).
- **BEV semantic segmentation:** 7 classes, 128×256 at 0.25 m/px, weight 14 (`rl_config.py:89,109-113`; head `:43-66`). Labels are rasterised from the HD map and the boxes.

**The planner's links, one diffusion decoder layer** (`transfuser_model_v2.py:314-344`; `blocks.py:71-109`)

| # | step | what it reads |
|---|---|---|
| 1 | **map at the candidate's own waypoints** | bilinear `grid_sample` of the fused BEV map at the 8 waypoints, softmax-weighted, residual. ⚠️ The paper says "deformable"; **the code uses fixed waypoint positions, no learned offsets** (`blocks.py:84-104`) |
| 2 | **agents** | cross-attention to the 30 agent queries → norm |
| 3 | **ego** | cross-attention to the **one** ego query → norm. Over a single key this reduces to adding a learned projection of that query, identical for all 20 candidates |
| 4 | output | FFN → norm (no residual, `:335`) → timestep modulation (AdaLN) → offset + score head; offsets added to the noisy waypoints (`:341`) |

- There are 2 cascade layers (`:430`). Layer 2 samples the map at layer 1's **detached** output (`:379`).
- The planner has **no** direct link to image or LiDAR features. Everything it knows comes through the BEV map, the agent queries and the ego query.

**What the paper measured about these links** (Table 3, p.7, all six rows re-read)

| ID | decoder | params | PDMS |
|---|---|---|---|
| 1 | TransfuserTD: conditional **UNet** + ego query | 102 M | 85.7 |
| 2 | ego query only | 57 M | **55.1** (collapsed: NC 88.7, EP 43.3) |
| 3 | ego + **map at waypoints** | 58 M | 87.1 |
| 4 | ego + **agents** | 58 M | 85.1 |
| 5 | ego + map + agents | 59 M | 87.4 |
| 6 | ID 5 + 2-layer cascade | 60 M | **88.1** |

⇒ **Correct reading:**
- Without a scene link, this decoder collapses.
- Map-at-waypoints is the **stronger single link**: +2.0 PDMS over agents alone (ID 3 vs 4).
- Agents add +0.3 on top of the map (ID 5 vs 3).
- The cascade adds +0.7.

⛔ The earlier "+32 PDMS for the spatial link" (55.1 → 87.1) was measured against a collapsed baseline. It is true but overstates the effect, and it is withdrawn.

**The nav command in DiffusionDrive**
- The 4-way driving command is **one quarter of an 8-number status vector**. That vector is projected to a **single token** among the 65 the scene decoder reads.
- It reaches the planner **only through the ego query** (step 3 above): one learned projection, added identically to every candidate.
- The `status_encoding` argument is passed into the diffusion decoder layer but **never used there** (`:314-344`).
- Nothing forces the planner to obey the command.

```
3 front cams 1024×256 ─┐                               LiDAR BEV 256×256 (±32 m)
                       └─ TransFuser: ResNet-34 (ImageNet) ×2, fused at 4 scales
                                  │
     8×8 tokens (256-d + learned position) + status token [cmd·v·a] ──► 3-layer decoder
                                  │                                   ├─► ego query (1)
                                  │                                   └─► 30 agent queries ─► box head
     up-scaled BEV map ─► 7-class seg head
             │
             └─ fused BEV map ──┐
 20 k-means anchors + noise(t=8)│
                      ─► per layer: [sample map at own 8 waypoints] → [attend 30 agents] → [attend ego query]
                                    → FFN → time AdaLN → Δ(x,y,yaw) + score      (×2 cascade, ×2 DDIM steps)
                      ─► argmax score → 4 s trajectory
```

### 1.2 DiffusionDriveV2 — what changes

| | V2 | source |
|---|---|---|
| perception + trunk | **identical to V1 and FROZEN**: every parameter outside `_trajectory_head` gets `requires_grad = False` + `eval()` | `diffusiondrivev2_rl_agent.py:62-68` |
| decoder | **1** cascade layer instead of 2 | `diffusiondrivev2_model_rl.py:733`, `_sel.py:844` |
| RL post-training | 4 groups × 20 anchors = 80 chains | `rl_config.py:37`; details in §7 |
| mode selector | 10 groups × 20 anchors → 2 DDIM steps = 200 trajectories → **+3 randomly scaled copies of each** (per-trajectory ×(1 + N(0, σ)), σ ~ U(0.1, 0.3)) = **800 candidates** → Bézier smoothing → coarse scorer (NC/EP/DAC/TTC/C heads recombined by the PDMS formula) → **top 32** → fine scorer (3 layers) | `_sel.py:1399-1446,1270-1286,1461-1470,876-899` |
| command | same status-token path as V1 | `_sel.py:167,237-239` |

The paper describes the selector as coarse-to-fine, following DriveSuprim (§4.6).

### 1.3 REF-C today (refcv5-v2) — the "direct link" design

**Trunk**
- `ResNetEncoder`: ResNet-34-style but wider and deeper — base width 88, blocks (3, 6, 16, 6), **90,458,632 params** (MEASURED).
- **Random init** (§6).
- Input `in_channels 9` = **a 3-frame RGB stack** (≈300 ms of motion) from one front-wide camera, 256×640 cylindrical (`refc.py:312`).
- Stride 32 → **8×20 = 160 tokens** of 704 channels.

**The planner's link to the trunk**
- **Queries:** 117 anchor queries.
- **Decoder:** 4 layers, d = 384, 8 heads (`DecoderConfig`, `refc.py:409-411`; d = 384 is confirmed by the checkpoint's `offset_head` init bound 1/√384).
- **Each layer:** cross-attention to **all 160 tokens**, then an MLP modulated by FiLM with the condition [v0, nav, keep] (`refc.py:1316-1371`).
- ⚠️ **No positional encoding is added to the tokens** (`refc.py:2292`: `kv = feat_proj(fmap.flatten(2).transpose(1,2))`); DiffusionDrive adds a learned one (§1.1).
- ⭐ **MEASURED today:** position is nevertheless **linearly recoverable from the token content**.
  - Column: **86.2 %** exact, 95.3 % within one column (chance 5 %). Row: **97.5 %** (chance 12.5 %).
  - Test set: 31,360 tokens, 39 held-out clips, ridge probe with λ chosen on a validation split.
  - Controls: majority class and permuted labels read chance exactly.
  - Raw: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-refcv6-clarification/raw/h3_token_position_probe.{py,json}`.
  - So "where" is available to the planner. Whether the planner uses it is a separate question (§2.2).
- **Frames read:** only the **newest window step's** map. Earlier steps reach the model only as pooled vectors through the strategic-context GRU (`refc.py:3155-3165`).

**Perception supervision reaching the trunk in refcv5-v2: none.** `--agents off` is set and there is no BEV auxiliary head. Every loss is a planning or classification loss (MODEL_REGISTRY §4.7.0b, argv diff).

```
3-frame stack 256×640 ─► ResNet (90.5 M, random init) ─► 160 tokens (8×20, NO position encoding)
[v0, nav, keep] ─► MLP ─► cond ────────────────────────────┐ FiLM in every layer
117 v0-rolled control anchors + noise(t=8) ─► 4 layers: [attend all 160 tokens by content] → MLP·FiLM
                                            ─► DDIM ×2 ─► emitted fan ─► re-score + priors ─► argmax
```

### 1.4 Side by side

| link / piece | DiffusionDrive V1 | REF-C refcv5-v2 | consequence for us |
|---|---|---|---|
| sensors | 3 cams + LiDAR | 1 front cam | range must be inferred from the camera. Camera-only LTF scores 83.8 vs TransFuser 84.0 PDMS (DD Tab. 1, PUBLISHED-PRIMARY) |
| trunk init | ImageNet ResNet-34 ×2 | **random-init** 90.5 M ResNet | §6 |
| scene supervision of the trunk | BEV segmentation + agent boxes, jointly | **none** | the trunk learns only what planning losses reward |
| token geometry | learned position embedding | none added, but **position is linearly recoverable from content** (column 86 %, row 97 %, MEASURED) | a minor gap, not a blocker |
| **planner ↔ map** | ⭐ sample the BEV map **at each candidate's waypoints** | ⛔ **absent** | the strongest single link in DD's ablation |
| planner ↔ agents | attention to 30 supervised agent queries | none | +0.3 on top of the map link in DD |
| planner ↔ image | via BEV only | attention to 160 image tokens | more direct, but ungrounded |
| planner ↔ ego / command | 1 status token → ego query → 1 attention per layer | FiLM with [v0, nav, keep] in every layer + v0-rolled anchors | ours is structurally **stronger** than DD's, yet refcv5-v2 obeys nav only weakly (§3) |

⇒ **"Linking the planner directly to the trunk" is what REF-C does today, and it is not wrong in itself.**
- DiffusionDrive is also direct: it samples a trunk map.
- The two differences that matter:
  1. DD's map is **metric** (BEV) and **supervised** (segmentation + boxes).
  2. DD's planner reads the map **where its candidate would drive**.
- REF-C has neither.
- A third difference I had listed, "DD's tokens know where they are", **does not hold as a gap**. Our tokens carry their position in their content (MEASURED, §1.3).

---

## 2. Can we extract boxes, semantic maps and occupancy — and can the planner link directly to the trunk?

### 2.1 Where we stand: nothing is proven yet

| target | best measurement so far | bar | status |
|---|---|---|---|
| **Occupancy** (LiDAR BEV GT) | Frozen refcv5-v2 trunk + 2.28 M attention head, 181 extra train clips: **AP 0.4352** [0.3976, 0.4737]; zero-information (shuffled targets) 0.3801 on the same 7.93 M cells; 34 test clips. Stride-16 tokens, an azimuth prior and an unfrozen last stage did not move it (MEASURED, `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md:20,26,257`) | pre-registered **AP ≥ 0.60 and IoU (0–30 m) ≥ 0.45** (`PREREG_BEVHEAD_FROZEN_TRUNK.md:100-101`) | ⛔ **far below** (−0.165 AP). The information is thin in a trunk trained only on planning losses |
| **Bounding boxes** | **Never probed on refcv5-v2.** Nearest evidence: a 4 k-step REF-C trunk scores agent AP 0.0548 vs raw pixels 0.0438, **not separated** (`…/2026-09-07-wpd-bev-aux/PANEL_RESULT.md`). An agent head trained jointly at 17 M cost distance-keeping on 2 seeds (`…/2026-09-07-p1-agent-gate/RESULT.md`) | none registered yet | ❓ unknown |
| **Semantic map** (SAM3, 9 classes) | **No trunk has ever been probed.** The GT exists (quality measured on 2 clips × 96 frames; night weaker) | refcv6 proposal's MAP clause (unapproved) | ❓ unknown |

⚠️ The earlier real-trunk agent result of WP-A (AP 0.027–0.034) is retracted: its azimuth address was mirrored (`RETRACTION_LOG.md`, `R-2026-09-08-wpa-mirror`).

### 2.2 The proof has two halves, and both are needed

1. **Extraction:** the information is in the trunk. We show this with a probe that beats every control.
2. **Use:** the planner acts on it. We show this with a causal intervention that moves the plan in the right direction.

A probe that decodes the lead vehicle proves nothing about driving if masking that vehicle out of the image leaves the plan unchanged. That is exactly the failure behind v6's readout, which decoded no better than raw pixels.

#### Half 1 — extraction ladder (each rung earns the next)

| rung | where | what | controls that must read known values |
|---|---|---|---|
| **1. Frozen probe** | dev box, **tokens already cached** (31 GB: stride-32/16 for 139 eval + 181 train clips, `C:/Users/Admin/tanitad-caches/bevhead-20260913/`) | Three heads on the frozen refcv5-v2 trunk: box (heatmap + 40 DETR slots), 9-class map on seen cells, occupancy. Each is also run on the **planner's own decoder outputs** | constant (= prevalence exactly) · per-cell prior · **shuffled targets** (zero information) · **raw-pixel floor** · seed replicate · **mirrored address** (must lose) · for decoder outputs, an ego-only arm |
| **2. Tiny joint** | dev box, 17 M REF-C, 4 k steps ≈ 80 min/arm (1.1–1.2 s/step MEASURED) | planner-only vs planner + heads vs planner + heads with a **deranged join** (same task, zero information) | + gradient-conflict detector (+1 / 0 / −1 analytic controls, 30× mutation). ⚠️ **Not implemented yet** |
| **3. Full joint** | **pod**, 108 M, 12 k steps ≈ 13–16 GPU-h/arm | control, control replicate, + three heads, + shuffled heads | T1, four families, `n` 4,823 windows / 141 episodes |

Proposed bars (to be frozen before any data is seen):

| target | bar (0–30 m unless stated) |
|---|---|
| boxes | AP@2 m beats max(shuffled, prior, pixel) by ≥ 0.05, separated, ≥ 3× the seed floor, on 2 seeds; lead-presence AUC beats a within-clip shuffle by ≥ 0.10. Usefulness: recall@2 m ≥ 0.50 at precision ≥ 0.50, lead range MAE ≤ 2 m |
| map | drivable and edge IoU beat the prior by ≥ 0.05 (same statistics). Usefulness: drivable IoU ≥ 0.80, lane-line/edge IoU ≥ 0.30 |
| occupancy | the registered B1–B4 unchanged |

Scale reference: the Qwen-Drive teacher on our clips reaches R@2 m 0.38–0.48 (0–50 m), 0.54–0.75 in 0–20 m.

#### Half 2 — does the planner USE it? (T1 harness, 4,823 windows, ≈ 35 min per roll on the dev box)

| test | intervention | what must happen | known-value control |
|---|---|---|---|
| **T-A** | blank the frames (`frames_blind`); zero the ego inputs | plan degrades. On refcv4b, blanking moved ADE 0.30 → 1.05. **Never run on refcv5-v2** | model-free arms bit-identical |
| **T-B lead mask** ⭐ | project each lead vehicle (≤ 30 m) into all 3 stacked frames and grey it out | headway / time gap / TTC / planned deceleration change **more than under an area-matched random mask**, in the unsafe direction | zero-area mask → plan bit-identical |
| **T-C road-edge mask** ⭐ | grey out SAM3 road-edge cells lifted into the image | cross-track / heading / curvature change more than under a random mask | mirrored mask |
| T-D scene swap | donor frames matched on v0 and nav | plan follows the donor scene | self-swap bit-identical |
| T-E frozen motion | 3 copies of the current frame | longitudinal family changes in closing windows | identity stack bit-identical |
| T-F pasted agent | paste a lead at range r into a lead-free window | planned speed falls monotonically as r shrinks | equal-area road-texture patch |
| T-G attention | selected anchor's attention mass on the lead's token columns ÷ area share | ≫ 1 | uniform attention reads exactly 1.0 |
| T-H decision probe | regress planned 0–2 s acceleration on gap, closing rate, v0, a0 | gap coefficient matches the human's | hold-action's gap coefficient is exactly 0 |

**Pass:** T-B's effect on minimum time gap and T-C's effect on cross-track are each separated, ≥ 3× the inference-seed floor, in the unsafe direction.

**Fail:** a lead mask ≈ a random mask. The planner is then not reading the lead, whatever a probe decodes.

### 2.3 "Link the planner directly to the trunk" — yes, and we already do. The question is *which* link.

| link | who did it | status |
|---|---|---|
| pooled readout (4×4 or 4×8 cells) | v1 / v4 / v5f flagship, v7-tiny | ⛔ measured too coarse; v6's readout decoded no better than pixels |
| content attention over image tokens, **no position encoding** | **REF-C today** (refcv5-v2) | lateral works (curvature error 0.512× the straight-line floor); longitudinal loses to hold-action |
| **map sampled at each candidate's waypoints** | DiffusionDrive (BEV, supervised) | ⛔ not built for REF-C |

**The experiment that settles it** (each arm adds one thing; zero-init gates reproduce the previous arm exactly at step 0):

| arm | the planner additionally reads | its zero-information twin |
|---|---|---|
| a | today's content attention | seed replicate |
| b | + 2-D position encoding on the image tokens. ⚠️ Lower priority: the tokens already carry position linearly (column 86 %, row 97 %, MEASURED §1.3) | encoding permuted across positions |
| c | + image features **sampled at each anchor's waypoints** (direct, unsupervised, ≈ 0.27 M) | random sample locations; mirrored projection |
| d | c's sampling over a **supervised BEV** (lift + map/occupancy heads) | maps shuffled across the batch |
| e | d + agent slots addressed by waypoint | deranged agent join |

Reading:
- **c ≈ d, both beat b:** a direct waypoint-indexed link suffices, and no BEV is needed.
- **d beats c and its shuffled twin:** supervised BEV information pays, and the grounded design is right.
- **d ≈ its shuffled twin:** the auxiliary task helped, not the information.
- **Nothing beats a:** this is not a perception-access problem, so the next lever is RL / ≥GT.

Every arm also re-runs T-B/T-C: masking must move the plan *more* than before.

### 2.4 Minimal proof package before you approve refcv6 (dev box, ≈ 5 days, ≈ 20 GPU-h, strictly sequential)

| day | work | GPU-h |
|---|---|---|
| 1 | T-A on refcv5-v2 (VOID gate first), T-G, T-H; build the mask hook with its bit-identity test | ≈ 1.5 |
| 2 | T-B, T-C with random masks, T-D, T-E, inference-seed replicate | ≈ 4 |
| 2–3 | rung-1 **box** panel on the cached tokens; bank decoder outputs | ≈ 5–6 |
| 3–4 | rung-1 **map** panel; occupancy on decoder outputs | ≈ 3 |
| 4–5 | conflict detector + controls + mutation; tiny-rig joint arms | ≈ 6 |
| 5 | matrix: extraction (3 targets) × use (T-B/T-C) → which link arms go to the pod | — |

**Needs from you:**
1. **The SAM3 maps of the 315 LiDAR-GT clips on the dev box** (≈ 1 GB, ESTIMATED). They are your private HF corpus files, or can be copied from Thor, which is still producing. A download needs your go.
2. **Sign-off to land default-off flags in `refc.py`** for arms b/c and the mask hook. They can be written and tested in a worktree first; the refcv6g package records that model changes wait for you.

⚠️ **Found while checking:**
- The trainer has **no warm-start flag**. It only resumes its own `ckpt.pt` strictly, optimizer included (`refc_v3_train.py:4735-4741`; two probes).
- The proposal's "warm start from refcv5-v2 with an allow-list of new keys" is therefore **new code**, not a switch.

⛔ This package cannot prove refcv6 works. It decides whether refcv6 should bet on **reading** the trunk (arms c/e) or on **supervising** it (arm d), before ≥ 65 pod GPU-hours are spent.

---

## 3. The nav command — mandatory for the tactical and the operative layer

### 3.1 Where nav goes in refcv5-v2 today (code read on the `d4e8bc0` snapshot)

| layer | does nav reach it? | how | does the output matter? |
|---|---|---|---|
| **operative decoder** (117 anchors, DDIM) | ✅ directly | `[v0/10, nav one-hot, keep]` → MLP → FiLM in **all 4 layers, every denoising pass**; reaches both the confidence and the control heads (`refc.py:3265-3267, 2293, 1371, 2011-2022`) | yes: this is the trajectory |
| **tactical heads on z_tac** (lat8 / lon8, 22 goal tokens, goal point g_tac) | ✅ | nav → `nav_to_tac` → z_tac (`refc_v3.py:1128-1157`) | ⛔ **almost not.** lat8/lon8 feed nothing under v7 labels; the 22-token head was never trained (weights at init); only g_tac@2 s enters selection, through a gate measured at 0.0126 |
| **core lat3/lon3 heads** (these drive the anchor prior) | ⛔ **no** | pooled image only (`refc.py:2867-2870, 3279-3294`) | yes, via the H19 anchor prior |
| **selection** | ⚠️ only indirectly | FiLM-conditioned confidence + the E9 goal term; **no nav or route term**; the rule-based nav-compliance scorer exists but is not on the forward path (`refc_selector_targets.py:406-450`) | yes |
| strategic route head | ⛔ no (by design: anti-echo) | image only | nothing reads it |
| strategic goal g_str | ✅ wired (`nav_to_str`) | — | measured nav-blind (below) |

Nav is never dropped in training (no nav dropout exists). The ego channels are dropped at p = 0.5.

### 3.2 How much refcv5-v2 actually uses nav (MEASURED, T1)

T1: 4,823 windows / 141 episodes, paired episode-cluster bootstrap. Raw source: `…/2026-09-07-refcv5-v2-comparison/raw/refcv5-v2.json`, blocks `refcv3.strategic.nav_compliance` and `refcv3.tactical_declared`.

| question | refcv5-v2 | refcv4b | reading |
|---|---|---|---|
| ADE change when nav is zeroed (os − navzero) | −0.0059 m [−0.0125, +0.0005], seed 1 −0.0070 [−0.0137, −0.0001] | **−0.0961** [−0.1102, −0.0813] | refcv5-v2 lost refcv4b's reliance on nav |
| **at 352 junction windows** (29 episodes): plan complies with the true command | 0.617 | 0.617 | — |
| … drop when the command is shuffled | **+0.099** [0.061, 0.142] | +0.102 | nav steers the plan **somewhat** at junctions |
| **39 windows fed a different turn**: plan follows the fed command | **0.205** [0.031, 0.378] (a perfect follower would be ≈ 0.76) | 0.231 | the operative plan is mostly **vision/ego-driven**, weakly steerable by nav |
| trajectory-derived turn recall, true → nav zeroed | L 0.777 → 0.761 · R 0.884 → 0.891 | — | ⚠️ hold-current-action already scores 0.753 / 0.770; these windows are mostly *already turning*, so this metric cannot see nav use |
| **tactical z_tac turn head** recall, true / shuffled / zeroed | TURN_L **0.475 / 0.050 / 0.000** (n 40) · TURN_R 0.313 / 0.104 / 0.000 (n 67) | 0.625 / 0.050 / 0.075 | the tactical head's turn decisions exist **only because of nav**: it echoes nav and has learned no visual turn skill |
| strategic goal g_str complies with the command | **0.206**, identical under shuffle | 0.790 | nav-blind |

⇒ **Today the two layers behave in opposite ways.**
- The **operative** plan mostly ignores nav except at junctions.
- The **tactical** head is almost a pure copy of nav, and its outputs barely reach the plan.

Neither meets your directive.

### 3.3 What refcv6 must change

**Wiring** (proposal; each item is a one-variable, default-off change with a zero-init gate):
1. **Operative:** keep nav FiLM in every decoder layer (exists).
2. **Tactical:** the z_tac heads read **nav + vision + v0** (exists), and are actually **trained** (`--w-tac-goal` 0.05; lat/lon; speed §4).
3. **Tactical → operative:**
   - the tactical lat/lon posterior replaces the image-only lat3/lon3 as the **anchor prior** (new zero-init 8 → 117 layers);
   - the tactical speed ceiling becomes a **selection mask** (§4);
   - both paths are **detached**, so the planning loss cannot turn the tactical head back into a nav echo.
4. **Selection:** add the existing parameter-free **nav-compliance term** behind a zero-init gate.

**Acceptance tests.** Bars are proposed; they are frozen before any arm trains.

| test | what it proves | bar | today |
|---|---|---|---|
| **T-FLIP** (`refcv3_arm.py --with-navflip`, never run on refcv5-v2) | nav is **mandatory in the plan**: flip the command at junctions and the plan must follow it | plan follows a flipped command ≥ 0.50; true − shuffled compliance ≥ 0.38 (half the 0.76 ceiling) | ≈ 0.205 / +0.099 |
| **T-ZERO** | the tactical layer **learned behaviour from labels**, not just nav | with nav zeroed, tactical turn recall ≥ 0.20; os − navzero separated on ADE and compliance | 0.000 |
| **T-FLOOR** | every tactical class beats trivial predictors | beats the majority-class control and a nav-only predictor on F1; TURN_L/R recall ≥ 0.60 at precision ≥ 0.50 | 0.475 / 0.313 |
| **T-REG** | no regression elsewhere | os − hold-action (`ha0_ext`) separated below zero; no family separated worse than the control arm | +0.0205, separated worse |

---

## 4. The tactical layer learns max speed and the whole vocabulary from GT labels

### 4.1 What the labels can supervise

Measured by the audit agent on the v8.1 train labels, 4,572 clips (md5 `b45377a1…`).

| label | coverage | trainable? |
|---|---|---|
| **SPEED_BAND** (v_lo, v_hi) + 8-step speed bucket | **4,572 / 4,572 clips**. Bucket histogram 809 / 932 / 1593 / 620 / 165 / 229 / 138 / 86; 38 % of clips ≤ 30 km/h | ✅ as a **target** |
| lateral actions | LANE_KEEP 2958 · NUDGE_R 592 · NUDGE_L 488 · TURN_L 275 · TURN_R 259 · **LANE_CHANGE_L/R 0 · ABORT_LC 0** | ✅, ⚠️ labelled only within **±2 s of each clip's anchor** (1,157 of 4,823 eval windows) |
| longitudinal actions | CRUISE 1243 · ACCELERATE 998 · ADAPT_SPEED_FOR_CURVE 995 · BRAKE_TO 905 · FOLLOW 165 · CREEP 135 · HOLD 131 · **YIELD_MERGE 0** | ✅, same ±2 s band |
| 22 tactical goal tokens | FOLLOW_LANE 3629 · SPEED_BAND 4572 · STOP_POINT 327 · TURN_L/R 275/259 · traffic light RED 376 / GREEN 363 · GAP_TARGET 368 · REACT_ON_ONCOMING 333 · EVADE 240 · … · OVERTAKE 20 | ⚠️ **17 of 22 trainable**; 5 masked (no negatives); 10 under the n = 200 floor. Several tokens are VLM-derived (CORRIDOR_OFFSET 860 and YIELD 609 are all disputed) |
| traffic lights | RED / GREEN usable only as a teacher signal (queue item 1, pending) | ⚠️ |
| lane change | text-only, untimed, ≈ 2.3 % | ⛔ |

How much each label is already predictable from the nav token alone (mutual information / label entropy):

| label | share predictable from nav |
|---|---|
| tactical TURN goal | 42.7 % (no clip has NAV_FOLLOW_ROAD together with a tactical turn: 0 of 2,897) |
| strategic goal | 30.5 % |
| lateral action | 16.9 % |
| speed bucket | 5.5 % |
| longitudinal action | 5.2 % |

⇒ Turns are where a tactical layer can "succeed" by copying nav, which is why T-ZERO is needed. Speed and longitudinal behaviour must come from vision and ego state.

### 4.2 ⚠️ The max-speed conflict — needs your ruling

| date | your words | source |
|---|---|---|
| 2026-09-01 | *"at inference this data will be provided as input by the user like the nav command, thus these are not training labels, just input data"* | `…/2026-09-01-v8-tacsit-release/V8_MANIFEST.json:121` |
| 2026-09-10 | *"regarding to max speed as input, stick to the labels we created in the data set with the logic of minimal speed etc..."* | `PREREG_E16_MAX_SPEED_INPUT_REFCV6.md:18-19` |
| **2026-09-15** | *"The tactical layer is responsible to learn the max speed and all tactical behaviros from our vocabulary using the gt labels in our data set"* | today |

**The fact that forces a choice:**
- The shipped "max-speed input" is **numerically identical** to the tactical label: `speed_max_input.v_max_ms == SPEED_BAND.v_hi_ms` on 4,572 / 4,572 clips.
- Both are the **ego's own realised maximum speed over [t0 + 2 s, t0 + 6 s]**.
- A model that is fed it and also trained to predict it can score by copying.
- Fed alone, it is an oracle. Even the quantised 8-step bin plus v0 recovers the raw future ceiling at R² 0.9702, against 0.8789 for v0 alone (MEASURED 5-fold, clip-disjoint, `PREREG_E16_MAX_SPEED_INPUT_REFCV6.md:11-16`).

| option | at training | at inference | leak? |
|---|---|---|---|
| **R2 — tactical predicts, user limit is a clamp** ⭐ default | tactical head learns SPEED_BAND (L1) + bucket (CE) from labels; **no speed input anywhere** | candidates faster than min(predicted ceiling, user limit if given) are masked in selection, parameter-free, like the reach mask | none |
| R1 — two separate quantities | tactical predicts the band; an optional limit input with a validity bit; the band loss applies only on rows where the input is withheld | limit input optional | only if the withheld split leaks |
| R3 — input dropout | limit input dropped at random; report with-input and without-input separately | optional | with-input rows can copy |

⭐ **Why R2 is the default:** it satisfies all three statements at once.
- The **tactical layer learns** max speed from the labels (09-15), using the labels we created (09-10).
- A user/nav-system limit remains an **input at inference** (09-01), yet is **never a training label** and never the same number as the label.
- No oracle enters the network.

Acceptance:
- predicted v_hi adds R² over a v0-only control (0.8789), separated;
- over-speed fraction no worse than refcv5-v2;
- forcing a 30 km/h limit on windows whose GT exceeds 40 km/h keeps the planned maximum ≤ the limit on ≥ 99 %, with the ADE cost reported.

### 4.3 The tactical layer refcv6 would have

```
frames ─► trunk ─► z_tac ◄── nav (mandatory) ◄── v0 / ego (dropout 0.5)
                     ├─► lat8 / lon8 actions   (CE, v8.1 labels, ±2 s band)          ─┐ detached
                     ├─► 22 goal tokens        (BCE 0.05, 17 trainable)               │
                     ├─► SPEED_BAND v_lo/v_hi + bucket (L1 + CE)  ───────────────────┼─► speed ceiling ─► selection mask
                     └─► goal point g_tac@2/4/6 s (existing)                          │
                                                                                     └─► anchor prior (8→117, zero-init)
operative: nav FiLM every layer ─► 117-anchor DDIM fan ─► prior + nav-compliance term + speed mask ─► trajectory
```

**Rulings needed beyond R1/R2/R3.** The audit's list, with my proposed defaults:

| # | question | default |
|---|---|---|
| a | Is T-ZERO binding (tactical skill must survive without nav)? | **yes**, otherwise "tactical learns from labels" is unprovable |
| b | May tactical outputs drive the anchor prior and selection, detached? | **yes, detached**, one knockout arm |
| c | Nav in the strategic layer: keep g_str's nav input (09-05 ruling) and the vision-only route head? | **unchanged** |
| d | Nav arguments constant per clip (`t0_constant`) or decremented with ego progress? | **t0_constant** (no ego-state dependence) |
| e | Loss budget for the new heads: on top of or inside the maneuver weight (queue item 10)? | inside, 0.05 goal tokens as measured |
| f | Vocabulary gaps (LANE_CHANGE, ABORT_LC, YIELD_MERGE have 0 labels; 5 tokens masked) | accept for refcv6; open a corpus labelling item |
| g | Traffic-light targets (queue item 1) | excluded until item 1 is ruled |

⚠️ **Admissibility guard** (PI 2026-08-03): no situation classifier may feed a goal input. The tac_SIT / traffic-light / YIELD tokens may be **auxiliary targets only**, never inputs to g_tac, g_str or selection. The provenance-roles check (`refc_v3.py:1048`) must be extended to refuse it.

---

## 5. The diffusion in our planner vs the paper — hypotheses, denoising, training

### 5.1 DiffusionDrive's generative process

Source: PUBLISHED-CODE `v1/transfuser_model_v2.py`. The noise values are computed with `diffusers` 0.40.0 defaults.

| item | V1 | V2 |
|---|---|---|
| hypotheses | **20 anchors** = k-means of training trajectories, 8 waypoints (4 s at 0.5 s), metres (`:398,407-412`) | RL: 4 groups × 20 = 80 chains · selector: 800 candidates (200 denoised + 3 scaled copies each) |
| schedule | DDIM, T = 1000, `scaled_linear` β 1e-4 → 0.02, predicts the clean sample (`:400-404`) | same + `steps_offset=1` (`_rl.py:697-702`) |
| noise space | waypoints normalised to [−1, 1] (x span 56.9 m, y span 46 m, `:433-441`); **i.i.d. per waypoint**, same size at 0.5 s and at 4 s | ⚠️ **changed** to x/50, y/20 (`_rl.py:755-762`); still i.i.d. per waypoint |
| training noise | t ~ U[0, 50) (`:465-468`) → std up to **2.68 m (x) / 2.17 m (y)** | RL phase: §7 |
| inference noise | t = 8 (`:518`) → std **0.90 m (x) / 0.73 m (y)** per waypoint | t = 8 → **1.58 / 0.63 m** (ESTIMATED; D-DDV2-CODE-4). The V1 cold start was trained at 0.90 / 0.73 m and loaded `strict=False`, which drops V1's second cascade layer |
| denoising at inference | **2 DDIM steps** at labels [10, 0], η = 0 (`:505-554`). Step 10→9 keeps **95 %** of the gap to the noisy input, so pass 2 re-denoises nearly the same input under a different time label | 2 steps, 1 cascade layer |
| training loss | per cascade layer: L1 of the anchor nearest the GT (= the x0 reconstruction, weight 8) + focal classification (weight 10) (`multimodal_loss.py:122-163`, `:494-497`), the sum × `trajectory_weight` 12. There is no ε-MSE: `diff_loss_weight` 20 multiplies a `diffusion_loss` the V1 head never emits, so it contributes 0 (`transfuser_loss.py:31-37`) | adds RL (§7) |
| selection | argmax score (`:555-557`) | coarse-to-fine selector |

✔ **Paper ablations** (p.7):

| ablation | settings | PDMS |
|---|---|---|
| denoising steps (Tab. 4) | 1 / **2** / 3 | 87.9 / **88.1** / 88.1 |
| cascade stages (Tab. 5) | 1 / **2** / 4 | 87.4 / **88.1** / 88.2 |
| inference samples N (Tab. 6) | 10 / **20** / 40 | 84.9 / **88.1** / 88.2 |

### 5.2 Ours (refcv5-v2, code at `d4e8bc0`)

| item | refcv5-v2 | same as the paper? |
|---|---|---|
| hypotheses | **117** anchors = a 13 × 9 grid of constant (a_long, a_lat) controls rolled out from the measured v0, 6 s | different by design (control space, speed-conditioned) |
| schedule | DDIM, same β table; `refc_sampler.py` asserts equality against `diffusers` when importable (`:80-81,210-276`) | ✅ |
| noise space | normalised **control** space: 8 slots ending at 0.5 / 1 / 1.5 / 2 / 3 / 4 / 5 / 6 s, `control_norm` (4.0, 3.0), noise i.i.d. per slot. At t = 8 that is σ 0.126 m/s² along / 0.095 m/s² lateral. Position noise **grows with horizon**: ≈0.016 m at 0.5 s, **0.145 / 0.109 m at 2 s**, **0.86 / 0.65 m at 6 s** (ESTIMATED, linearised, v0 ≥ 4 m/s) | ⚠️ different: DD's is flat at 0.90 / 0.73 m from the first waypoint, ≈ 55–60× ours at 0.5 s |
| denoising | fresh noise at t = 8, labels [10, 0], η = 0, x0 prediction (`refc_sampler.py:160-201`) | ✅ labels; ⚠️ **our step is 10→0 and keeps 28 % of the gap; DD's is 10→9 and keeps 95 %** |
| **training** | ⛔ the network sees only the inference chain: one noise level, backprop through both steps, final output only | ⛔ **not DD's objective**. DD trains one denoising call at a random t ∈ [0, 50) with every cascade layer supervised |
| time conditioning | timestep embedding added once to the query | DD modulates after every layer (AdaLN) |
| regression loss | L1 in metres on the GT-nearest anchor **plus `--w-u0 0.5`**, a second x0 L1 in control units | DD has one |
| classification | softmax CE over 117 + goal-scorer CE | DD: focal loss per anchor |
| selection | extra decoder pass on the emitted fan + priors + reach mask + goal scorer | different |
| samples per anchor | 1; more is refused (`refc.py:2281-2291`) because three call sites would be mis-indexed | ✅ DD's default is also 1 per anchor (N 20 = 20 anchors). Doubling to 40 gave +0.1 PDMS (Tab. 6) |

**MEASURED on our hypotheses** (MODEL_REGISTRY §4.7, T1, 4,823 windows / 141 episodes):
- anchor accuracy **0.5271** [0.4858, 0.5685] = **61.7× chance**, indistinguishable from refcv4b (0.5275);
- only **62 of 117** anchors are ever selected (64 at seed 1; refcv4b 51);
- a single modal anchor (#67).

### 5.3 Verdict

* ✅ **At inference refcv5-v2 is a real truncated diffusion sampler.** refcv4b and earlier were refiners, not diffusion models (2026-09-05 audit).
* ⛔ **It is not trained the way DiffusionDrive is trained:** there is no random-t denoising objective, no per-layer supervision, and the timestep enters only at the input.
* ⚠️ **Noise and step semantics also differ.** Our noise grows with horizon where DD's is flat, and our second pass starts much closer to the first prediction.
* ⚠️ **Correction of our own record:** *"DiffusionDrive has no denoising loss, so `--w-u0` is our invention"* is misleading. DD's matched-anchor L1 **is** its x0 denoising loss; `--w-u0` duplicates it in control space.
* ⚠️ **The hypothesis set is used narrowly:** only 62 of our 117 anchors are ever selected, and one modal anchor recurs.

### 5.4 What would make ours paper-faithful (each a separate, measurable change)

| # | change | where it runs |
|---|---|---|
| F1 | random-t training: one decoder call at t ~ U[0, 50) with a per-layer loss (a diff exists in `…/2026-09-06-p4-p13-p14-validation/RESULT.md` §4.1) | retrain (pod) |
| F2 | DD step semantics t → t−1 | **inference-only A/B on the existing checkpoint (dev box)** |
| F3 | per-layer offset heads + per-layer loss + detach between layers | retrain |
| F4 | per-layer AdaLN timestep modulation | retrain |
| F5 | score with the emitting pass's own confidence + focal loss (removes one of four decoder calls) | retrain |
| F6 | `--w-u0 0`, the single paper loss | retrain, 2 seeds |
| F7 | several noise samples per anchor, after widening the three indexing sites. Low priority: +0.1 PDMS in DD Tab. 6 | code + dev-box eval |
| F8 | flat waypoint-space noise as in DD vs our control-space noise, as an explicit arm rather than an accident | retrain |
| F9 | keep the v0-conditioned vocabulary | — |

---

## 6. How the ResNet is trained and used — DiffusionDrive, other AD works, and ours

### 6.1 Side by side

| | **DiffusionDrive V1** | **DiffusionDriveV2** | **REF-C refcv5-v2** |
|---|---|---|---|
| image trunk | ResNet-34, **21.8 M** | same | BasicBlock ResNet, widths 88→704, blocks (3, 6, 16, 6): **90.5 M** = 4.25× a ResNet-34 built with the same code (MEASURED) |
| init | **ImageNet-1k** `resnet34.a1_in1k` (supp. §A p.12; `rl_config.py:16-18`); the LiDAR ResNet-34 starts from scratch | cold start from V1's checkpoint | **random**; the only weight loading is exact resume (`refc_v3_train.py:4735-4741`) |
| input | 3 cameras stitched into 1024×256, **one frame**, + LiDAR BEV | same | 1 front camera 256×640, **3 stacked frames** (9 channels) × 8 window positions |
| what trains it | end-to-end: trajectory + **BEV agent boxes + 7-class BEV segmentation** | ⛔ **nothing: the trunk is frozen** in RL and in selector training (`rl_agent.py:62-68`, `sel_agent.py:61-82`; the paper never says so) | trajectory 1.0 + anchor CE 1.0 + selection 1.0 + u0 0.5 + route 0.1 + four lat/lon heads at 0.025 each (0.10 total, `refc_train.py:76-91`, `refc_v3_train.py:2510-2513`) + LAW 0.5 (predict its own pooled latent 0.5 s ahead, **no EMA target**) · **no perception loss** |
| optimiser | AdamW 6e-4, weight decay 1e-4, **image encoder at 0.5× the learning rate** (`rl_config.py:124-128`), 3-epoch warm-up + cosine, fp16 | AdamW 2e-4, cosine | Adam 1e-4, **one rate for trunk and heads**, no weight decay, 2 k warm-up + cosine, clip 10 |
| augmentation | none in the released feature builder | trajectory noise for the selector | none |
| exposure | 100 epochs × 103 k samples at batch 512 (≈ 10.3 M samples), after ≈ 769 M ImageNet presentations (ESTIMATED: 600 epochs × 1.28 M) | 10 + 20 epochs | **one epoch**: 805,680 windows at batch 20 over 4,572 clips ≈ **25.5 h** of driving |
| use at inference | full trunk every frame; the planner reads a BEV map that carries **learned position embeddings** | same, frozen | full trunk every tick; the decoder reads the **newest** window position's 160 tokens with **no position encoding** |
| speed | whole model 60 M, **45 FPS on an RTX 4090** (Tab. 2) | not reported | not measured at 256×640 |

### 6.2 Other end-to-end driving works (PUBLISHED-PRIMARY; cells verified against the paper text)

| work | trunk · init | trained how | frames | perception supervision |
|---|---|---|---|---|
| TransFuser | RegNetY-3.2GF · **ImageNet** | fine-tuned, ±20° rotation augmentation | 1 | depth, semantics, BEV map, detection |
| UniAD | ResNet-101 · **BEVFormer checkpoint** | **frozen** in both stages | 5 | tracking, map, motion, occupancy |
| SparseDrive | ResNet-50/101 | fine-tuned at **0.1× the learning rate** | 3-frame queue | detection, tracking, map, motion, depth |
| Hydra-MDP / GTRS | V2-99 · **DD3D** (depth-pretrained) / ViT-L · DepthAnything | — | 1–4 | 3-D detection, BEV segmentation |
| Hydra-MDP++ | ResNet-34 / V2-99, "pretrained" | current frame fine-tuned, t−1 frozen | 2 | none (perception aux 86.6 → 86.1 PDMS) |
| DriveTransformer | ResNet-50 | single-stage fine-tune | 10 | detection, motion, map |
| DriveVLA-W0 | ResNet-34 / **DINOv3 ViT-7B** | fine-tuned | 1 | none (action-only) |
| **REF-C refcv5-v2** | 90.5 M · **random** | one learning rate, one epoch | 3-frame stack | **none** |

**What the literature supports:**
- **Pretraining.** Every stated image-trunk init is pretrained, and **none of these papers ablates ImageNet vs scratch**. REF-C is the outlier.
- **Supervision matters more than the prior.** DriveVLA-W0 shows that a pretrained trunk does not rescue action-only training: at 70 k frames, ResNet-34 ADE 2.59 and DINOv3-7B 2.58. Adding scene supervision moves results more:
  - LAW's latent loss: 77.5 → 84.6 PDMS;
  - TransFuser without BEV segmentation: 81.6 vs 83.3–84.4;
  - DriveTransformer planning-only: 54.2 vs 60.5.
  - Counterpoint: Hydra-MDP++ loses 0.5 PDMS with perception aux.
- **Data.** In NVIDIA's data-scaling study, closed-loop gains plateau around 256 h of driving. **Our 25.5 h is 10× below that knee.**

### 6.3 What the programme has measured about our trunk

| # | finding | evidence |
|---|---|---|
| 1 | The frozen refcv5-v2 trunk is **not separable from raw pixels** on LiDAR BEV: AP 0.4140 vs pixels 0.3849, Δ +0.029 [−0.004, +0.069]. All four bars fail | `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md:253-292` |
| 2 | Stride-16 tokens, azimuth attention and an unfrozen last stage do not help. Only 3.2× more training rows moves it | same, `:437-442` |
| 3 | **On our own frames a pretrained prior carries more scene content:** frozen DINOv3 beats a random-init encoder by +0.350 [+0.182, +0.523] and pixels by +0.182 on within-clip agent count | `…/2026-09-04-v7-seed-and-external-target/RESULT.md:97-106` |
| 4 | **Trunk size does not matter for planning:** REF-C v2.1 base vs XL ADE +0.0013 [−0.028, +0.032], not separated | MODEL_REGISTRY:2563 |
| 5 | ⛔ **An ImageNet-initialised REF-C arm was never run.** It was proposed on 2026-08-03 ("Pretrain the vision trunk. Do NOT scale it.") | `…/2026-08-03-refc-planner-vision-research/…:223,250-252` |
| 6 | Registry gap: the refcv3 / refcv4b / refcv5-v2 rows do not state the encoder init | sweep of MODEL_REGISTRY |

### 6.4 What this implies (hypotheses) and the cheap tests that decide them

| # | hypothesis | test | cost | needs |
|---|---|---|---|---|
| H1 | **The trunk lacks a pretrained prior** | frozen `resnet34.a1_in1k` (newest RGB frame) vs frozen refcv5-v2 vs random R34 on the banked agent-count panel (< 30 min) and the P4 BEV panel (2.5 h, 2 seeds) | ≈ 3 GPU-h | ⚠️ `timm` + 87 MB ImageNet weights: **a download, your go** |
| H2 | **Planning-only losses do not ground a trunk** | the same P4 probe on the first refcv6 arm that trains a perception head, vs refcv5-v2 | 35 min once the arm exists | — |
| H3 | ~~The decoder is position-blind~~ **REFUTED (MEASURED 2026-09-15)** | ridge probe on the cached refcv5-v2 tokens, clip-disjoint: column **86.2 %** (±1 col 95.3 %; chance 5 %), row **97.5 %** (chance 12.5 %); 31,360 test tokens / 39 clips; majority and permuted-label controls read chance exactly | done (CPU, seconds) | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-refcv6-clarification/raw/h3_token_position_probe.{py,json}` |
| H4 | **The trunk is data-limited** | P4 probe on refcv4b at step 9,500 vs 40,284 | ≈ 1.2 h | both checkpoints are **already on the dev box** (`navcomp/ckpt/ckpt_step9500.pt`; `refcv4b_final/ckpt_40284.pt`, 428,616,885 B = the HF file) |
| H5 | **LAW without an EMA target flattens the features** — **supported, not proven (MEASURED 2026-09-15)** | participation ratio (σ², the programme's collapse measure) on the same 200 eval frames. **Trained refcv5-v2:** tokens **4.2** (top direction 45 % of the energy), frame-pooled 2.9 · **random-init twin, scale-matched** (batch-statistic BN; max \|token\| 42 vs 37): tokens **16.1** (24 %), pooled 1.2 · random init with init-stat BN is degenerate (activations vanish, max 0.28) · controls: isotropic 689 ≈ d, rank-1 = 1.0 · G-RANK reference (frozen DINOv3) 8.56. ⇒ **Training concentrated the token field ≈ 4×.** Participation ratio cannot tell collapse from specialisation; the decider is a tiny-rig arm with LAW off or on an EMA target (≈ 2 h) | done (CPU) + ≈ 2 h tiny rig | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-refcv6-clarification/raw/h5_token_participation.*`, `TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-refcv6-clarification/raw/h5b_random_init_twin.*` |
| H6 | **The 9-channel stack is unnecessary** (it blocks direct ImageNet init; stack motion was shown not load-bearing, and DD is single-frame) | 3-ch vs 9-ch pixel arms in P4, then on the tiny rig with 2 seeds | ≈ 2.5 h | — |

### 6.5 The trunk decision refcv6 faces

| option | trunk | what it costs | what it risks |
|---|---|---|---|
| **T-A keep ours, warm start** (the 09-15 proposal) | refcv5-v2 trunk + perception supervision | 12 k steps per arm | inherits a trunk ≈ pixels on BEV; the prior question stays unanswered |
| **T-B paper-faithful** | ImageNet ResNet-34, single RGB frame, encoder at 0.5× lr, jointly supervised | planner **retrained from scratch**, 40 k steps. Per step it is cheaper than refcv5-v2, whose 90.5 M trunk runs on all 8 window positions with gradient; refcv5-v2 itself took ≈ 47 h, an upper bound (ESTIMATED) | loses refcv5-v2's lateral skill until retrained; stem and input change |
| T-C larger pretrained (DINOv3 / DD3D V2-99) | frozen or low-lr | more memory, slower tick | no evidence yet on our planner |

⭐ **Default:**
1. H3 and H5 ran today on the CPU (above).
2. H4 needs no download (≈ 1.2 GPU-h). It runs after the RL validation releases the GPU.
3. H1 runs once you approve the ~87 MB weights. The LAW-off / EMA tiny-rig arm for H5 joins the proof package.
4. Carry **T-A and T-B as a paired arm on the pod** only if H1 shows the ImageNet prior beats our trunk, separated. Otherwise T-A.

---

## 7. Reinforcement learning as in DiffusionDriveV2 — IN PROGRESS on the dev box

**Status at landing (2026-09-15, 22:50):** the preparation is running and **nothing in this section is a result yet**. This section is replaced when the work is verified.

**What is being prepared, in an isolated worktree** (not landed; nothing touches a live run):

| deliverable | purpose |
|---|---|
| `SPEC_DDV2_RL_PAPER.md` | the RL method **as the V2 paper states it**, each equation and hyper-parameter pinned to a page, beside what the released code actually runs (§1.2 found that the released code already departs from the paper on the frozen trunk, the decoder depth and the coordinate normalisation) |
| `ddv2_step_sensitivity.{py,txt}` | how the paper's per-step exploration noise behaves under DD's schedule |
| `stack/tanitad/rl/ddv2_rl.py` + tests | the paper's objective as a library: grouped rollouts per anchor, advantages, the constrained truncation, the imitation term |
| `stack/tanitad/rl/ddv2_refc_chain.py` + tests | the log-probability chain the objective needs on **our** control-space DDIM decoder, which today exposes none |
| `stack/tanitad/rl/pdm_proxy.py` + tests | a label-time reward proxy for DD's PDM score, because PhysicalAI has no simulator |
| `stack/scripts/ddv2_rl_refcv5.py` | the runner for a tiny pre-registered validation on refcv5-v2, with a held-out split |

**Validation budget:** ≤ 3 GPU-hours on the RTX 4060, sequential, pre-registered before any number is read.

**What this section will answer:**
1. What the paper's RL is, exactly.
2. What the released code runs instead.
3. What of it our data and decoder can carry.
4. Whether the tiny validation shows the objective improves the controls it must beat, **without** regressing the four families.
5. The decision for you (D9).

---

## 8. Your computer for preparation, a pod for heavy work

**Measured anchors for every cost below:**
- **Dev box:** RTX 4060, 8 GB, one GPU job at a time (a second job beside a trained arm crawled, MEASURED). One T1 evaluation roll of refcv5-v2 (4,823 windows) takes ≈ 35 min.
- **Tiny rig** (17 M REF-C): 1.1–1.2 s/step on the dev box, so ≈ 80 min per 4 k-step arm.
- **refcv5-v2 (108 M) training:**
  - on a 48 GB pod GPU: batch 20, 43.4 of 46 GB in use, ≈ 4.2 s/step, so 40,284 steps ≈ 47 h;
  - WP-D at 108 M on Thor: 4.2–4.9 s/step.
- **Thor** is busy producing SAM3 maps until ≈ 22 Sep.

### 8.1 On the dev box, now → approval (no plan launched, nothing trained at 108 M)

| # | work | answers PI point | GPU |
|---|---|---|---|
| 1 | Perception proof package (§2.4): planner-use tests T-A…T-H on refcv5-v2, rung-1 box / map / occupancy probes on cached tokens, tiny-rig joint arms, gradient-conflict detector | 2 | ≈ 20 GPU-h over ≈ 5 days |
| 2 | Nav battery on refcv5-v2: T-FLIP (never run), T-ZERO readout (§3.3) | 3 | 1–2 rolls |
| 3 | Diffusion inference A/Bs on the existing checkpoint: DD step semantics (F2), several noise samples per anchor (F7, after widening the three indexing sites) | 5 | ≈ 35 min per roll + code |
| 4 | RL per the paper: implementation, unit tests, tiny validation (§7) | 7 | ≤ 3 GPU-h (agent budget) |
| 5 | Code for the approved design, **every piece default-off**, with unit + mutation tests: tactical heads and wiring (§4.3), faithful diffusion training (F1, F3–F6), warm-start loader (**new code**: the trainer only resumes its own `ckpt.pt`), T-FLIP gate | 3, 4, 5 | CPU + tiny smokes |
| 6 | Pre-registration: bars, arms and controls frozen before the pod starts | all | — |

**Needs from you before items 1–2 can run:**
- The SAM3 maps of the 315 LiDAR-GT clips on the dev box (≈ 1 GB), from your private HF corpus or copied from Thor.
- Sign-off to land the default-off `refc.py` flags (worktree preparation needs none).

### 8.2 On the pod (after approval)

| block | arms | cost (ESTIMATED; re-measured by the launch gate) |
|---|---|---|
| link-type matrix (§2.3), frozen trunk | a–e + zero-information twins (≈ 9) | 9 × 6–8 GPU-h |
| link-type matrix, joint trunk | top 2 + replicates | 4 × 13–16 GPU-h |
| refcv6 system + knockouts, 12 k steps warm | control, full, replicate, 3–4 knockouts | 6–7 × 13–16 GPU-h |
| faithful-diffusion arms (F1, F3–F6 as one arm + F6 knockout) | 2–3 | 2–3 × 13–16 GPU-h |
| RL stage on the best arm | 1–2 | ≤ 1 day |

**Total ≈ 230–320 GPU-h on a 48 GB-class GPU** (≈ 10–13 days on one GPU, ≈ 2.5–3.5 days on 4).

**Suggested pod:** 4 × 48 GB (A40 / L40S / A6000), or 2 × 80 GB (A100 / H100). refcv5-v2 already fills 43 GB at batch 20, and the BEV branch adds memory. The first hour is the throughput gate, which re-states this table from measured s/step.

### 8.3 Order you set

refcv6 (this dossier → your rulings → proof package → pod) → **refav1** → **flagship v7**.

---

## 9. Corrections to our own records (found while answering)

Every row was re-checked on 2026-09-15 against the primary source named in its last column. Nothing is edited silently: each correction lands as its own commit with a RETRACTION_LOG line.

| # | where the stale statement lives | what it says | what is true | primary source |
|---|---|---|---|---|
| C1 | `Project Steering/REFCV6_DESIGN_GROUNDED.md:102` (my own proposal, `d4e8bc0`) | BEV sampling at the waypoints "plus P = 4 learned offsets each (deformable sampling, **DiffusionDrive's** trajectory-indexed spatial attention)" | DD's released code samples at the waypoints with **no learned offsets**. Offsets are **our extension**: label them so, and test them against the paper form | `ddv2_src/blocks.py:84-104` |
| C2 | `REFCV6_DESIGN_GROUNDED.md:126`; `Reports/2026-09-15-1445-refcv6-refav1-v7-status.md:40` | "DiffusionDrive has no denoising loss", "u0 0.5 (our invention)" | DD's matched-anchor L1 **is** its x0 denoising loss. `--w-u0` is a **second** copy of it in control space | `multimodal_loss.py:124-163`: the anchor nearest the GT is chosen (`:133-135`), then focal classification to it (`:148`) plus L1 of its prediction to the GT (`:159`); summed per cascade layer at `transfuser_model_v2.py:494-497` |
| C3 | `PREREG_REFCV6.md:49`; `REFCV6_ARCHITECTURE_REVIEW.md:49`; `GOALS_AND_CLAIMS.md:9059` | "Hungarian-matched **3D** box + class loss" | the code predicts **2-D BEV boxes** (x, y, heading, length, width). The paper's "3D detection" wording is looser than its code | `transfuser_model_v2.py:8,140-…` (`BoundingBox2DIndex`) |
| C4 | `PREREG_WPD_BEV_AUX.md:40` | "waypoint-indexed **deformable** cross-attention (DiffusionDrive coupling (1))" | fixed-point bilinear sampling at the waypoints, softmax-weighted | `blocks.py:71-109` |
| C5 | `stack/tanitad/refs/refc.py:2015-2017` (code comment) | WP-B's agent index "is exactly DiffusionDrive's coupling (1)" | DD's coupling (1) samples the **BEV map** at the waypoints. WP-B indexes **agent slots** by the path, which is closer to DD's agent attention with a geometric address | `transfuser_model_v2.py:324-325` |
| C6 | `stack/tanitad/refs/refc_sampler.py:38` (docstring) | timestep embedding "injected **per layer**" | added **once** to the query before the shared layers | `refc.py:2013-2014` |
| C7 | `refc_sampler.py:19-21` (docstring) | control noise gives "~2.3 m along-track and ~1.7 m lateral at the 6 s endpoint — comparable spread to DD's" | with the 8-slot i.i.d. noise the model actually uses: **≈0.86 / 0.65 m at 6 s, 0.145 / 0.109 m at 2 s** (ESTIMATED, linearised). 2.3 / 1.7 m holds only for a noise constant across slots | `refc_sampler.py:490-493`; horizons `refc_anchors_6s_v0cond_alat_117.pt.json` |
| C8 | my earlier chat summaries (not landed) | "the spatial link is worth +32 PDMS" | measured against a **collapsed** ego-only decoder (55.1). Over agents-only it is +2.0; agents add +0.3 on top of it | DD paper Table 3, p.7 |
| C9 | my draft of this dossier (caught before landing) | "the V2 selector sees 200 candidates, not 800" | **800 is right**: 200 denoised + 3 randomly scaled copies of each, exactly as `D-DDV2-CODE-4` says | `diffusiondrivev2_model_sel.py:1270-1286,1444` |

**Carried from an earlier review pass, not yet re-located today** (they are fixed only after being re-read):
- the scope of "V2's encoder is byte-identical to V1" (true of the RL model's import; not yet re-checked for the selector);
- literals in `2026-09-10-refcv6-build/code/verdict_refcv6.py` that may predate the 09-09 evaluation.
