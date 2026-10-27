# REFCV9_DESIGN: full training with a redesigned map and box head, wired into the planner (H-R9-FULL-1, design)

*TanitAD Research Lab, Architecture & Inference (refcv9 design agent), 2026-10-27 ~23:50 Berlin. This is a DESIGN, not a
pre-registration: §5 is the draft the Master Mind seals after E2 and before the full run fires. No GPU, no Thor, nothing
executed beyond `git` plumbing reads and file reads.*

**PI decision, 2026-10-27 ~23:00 (verbatim):** *"Do full training without 48 h cap. Find a clear solution to sucessfully
improve. Review by focused research about comparable scessful works regrading training of bb object detection and bev map
extraction. Review architecture and training recipes of the heads, review size and represenation of the map..."*
⇒ refcv9 = FULL training (perception trained), no 48 h cap, cost stated. The frozen-perception warm start
(`DRAFT_PREREG_REFCV9_WARM.md`) was NOT fired; its planner-side levers carry over where they still apply (§2.8).

| | |
|---|---|
| **Inputs read** | `ASIS_PERCEPTION.md` (our stack from source, D1-D21), `LIT_DETECTION.md`, `LIT_BEVMAP.md` (this package, ref `agent/arch-inf-20260803` @ `f837e28`); the perception-bottleneck package in the hold `C:/Users/Admin/qland/hold_perc_diag/` (`RESULT.md`, Q3 FINAL); `DRAFT_PREREG_REFCV9_WARM.md`; `PI_DECISION_QUEUE.md`; `GOALS_AND_CLAIMS.md` rows `H-R9-MAPFT-1`, `H-R9-ANTICHEAT-REF-1`; source of `refcv8_conditioning.py` (`DrivableCritic`), `refc_v3_train.py` (flags), `sam3map_prod.py` |
| **Evidence classes** | MEASURED (ours, artifact named) · INHERITED (another agent's number, not re-derived here) · PUB-P (read in the primary by a literature agent, table named in `LIT_*.md`) · PUB-SECONDARY (not banked) · ESTIMATED (my arithmetic, inputs stated) · HYPOTHESIS |
| **Banking status of the literature** | `LIT_DETECTION.md`: all 34 primaries banked (sha256 in `raw/lit_det_library_entries.json`). `LIT_BEVMAP.md`: 38 primaries downloaded to `F:/c_overflow_2026-10-26/tmp/lit_map/`, **NOT banked** (its agent could not write G:) ⇒ every `LIT_BEVMAP` number below is PUB-P read but **inadmissible for the registry or the paper until banked**. Two citations new in this file are PUB-SECONDARY (§7). |
| **Tier** | Design. Every bar in §5 names its tier; perception bars are T0 label-level reads, planner bars are **T1 self-action open loop on logged eval139 frames (not driving performance)**, as every refcv8/refcv9 registry row reads. |
| **Q1 (features or head?)** | **PENDING at writing.** `F:/perc_bottleneck/logs/post.log` had no `ZZPOST-done-0ZZ` at 23:45 (eval capture 326 of ~556 windows). Both branches are pre-committed in §3; the branch is picked mechanically by the sealed rule in `PREREG_PERC_BOTTLENECK.md`, never by taste. |

---

## 0. Headline

1. **The one measured driving lever this redesign exists for is the MAP INTO THE PLANNER.** refcv8 plans onto sidewalk in
   **17.9 %** of S6c windows (human 1.25 %); the TRUE map as a hard filter gives **1.7 %** with better ADE, as a soft cost on a
   progress-matched fan **2.2 %**; refcv8's OWN map converts **nothing** (cross-fitted weight 0 in 5/5 folds) (INHERITED,
   MAPSEL / MAPPATH `H-R9-MAPPATH-1`, `H-R9-MAPFT-1`). The gap is ~16 points of off-road planning, and it is locked behind map
   QUALITY (own sidewalk flags 25 % precise at 0-5 m; own edge IoU ~0.001) and WIRING (no map decision reaches the planner).
2. **Why the map is bad is measured, and it is not the head's optimiser:** the map term owns **0.00018 / −0.00114 / 0.00396 /
   0.00024** of the trunk update at refcv8 steps 1,000 / 16,250 / 32,500 / 48,750 and 0.0013-0.021 of its own 0.25 m BEV
   encoder (MEASURED, refcv8 `metrics.jsonl` on Thor, per the brief); box3d owns **97.3 %** of that encoder at the warm-start
   point (MEASURED, P-GRAD `pgrad.json`). The "map head" is a probe on detector features, and it is supervised as 1-2 px hard
   lines at 10 cm out to 100 m, which the image cannot support beyond ~6-8 m (ESTIMATED, `LIT_BEVMAP` §2).
3. **The box head's largest measured lever is training-time de-duplication:** a post-hoc 2.5 m NMS lifts AP@2 m
   **0.328 → 0.447** at zero training (INHERITED, PERC8), and the head emits **2.14 boxes per detected object**. The second is
   RANGE: AP@2 m **0.539 / 0.234 / 0.092** at 0-20 / 20-40 / 40-60 m, separated (MEASURED, Q3). The class output is NOT collapsed
   (TP class accuracy 0.822 vs majority 0.726, MEASURED Q3).
4. **The design (ranked by expected effect, §2):** (R1) a planner-facing map representation, range-banded, with the edge
   derived from a drivable signed-distance field and an explicit UNOBSERVED channel; (R2) wire it in: `DrivableCritic` ON,
   reading the own map, trained on GT-map labels, plus map decision channels into the planner BEV; (R3) give the map a
   gradient-share TARGET (≥ 0.05 of layer2's projected update) and its own BEV encoder so it stops fighting box3d for 64
   channels; (R4) box de-dup trained in (quality-aware presence, query denoising, 6 decoder layers); (R5) depth and range for
   boxes (box-derived foreground-depth supervision, a perspective 3D auxiliary head, per-frame queries, far-range s8 tokens);
   (R6) a WSD learning-rate schedule so the checkpoint is chosen on independent data, not on the test set.
5. **Cost (ESTIMATED, §4.5):** de-risking E1 + E2 ≈ **66 Thor-h**, the full run ≈ **214 Thor-h** (73,280 steps = 1.5
   window-epochs from refcv8 EMA-48,852 at ~10.1 s/step), an early decay read ≈ 17 Thor-h: **≈ 300 Thor-h, ~12.5 days of
   sequential Thor**, with two hard stop gates before most of it is spent. refcv8 cost ~121 Thor-h.
6. **Four PI decisions are needed** (§6): the compute envelope; retiring the 2-4 M box-head band; temporal BEV fusion with
   past ego-motion; LiDAR depth at train time. Two more are gate/protocol items.

---

## 1. The measured baseline this design builds on

| fact | value | class / source |
|---|---|---|
| refcv8 planner on sidewalk (S6c windows) | **17.9 %**; human 1.25 %; GT-map hard filter 1.7 %; GT-map soft cost on a progress-matched fan 2.2 %; own-map soft cost fitted to 0 weight 5/5 folds | INHERITED (MAPSEL, MAPPATH; memory `pi-planner-must-use-map-head`) |
| own-map corridor sidewalk+edge precision | 0-5 m **0.251**; 5-40 m 0.733 / 0.716 / 0.744 | INHERITED (`H-R9-MAPFT-1`, `nearfield_prec`) |
| own drivable boundary as the edge | recalls **29 %** of GT edges @0.3 m; edge class 0.2 %; GT-boundary ceiling 96 % | INHERITED (MAPEDGE, e457400) |
| map-only fine-tune | CE converged, map unchanged (train loss 0.886 → 0.904) | INHERITED (`H-R9-MAPFT-1`) |
| map gradient share, refcv8 run | trunk 0.00018 / −0.00114 / 0.00396 / 0.00024; 0.25 m BEV encoder 0.0013-0.021; `tac_v6` ≤ 0.0007; rest ~0.999 | MEASURED (brief; Thor `metrics.jsonl`) ⚠️ `rest` folds box3d + agent + planner (`grad_share.py:21-26`) |
| per-term split at the warm-start point | trunk: agent **0.501**, box3d **0.357**, map 0.0001; shared BEV: box3d **0.973**, map −0.004; norms at layer2 map 1.12 vs box3d 73.7 vs agent 92.3 | MEASURED (artifact `…/2026-10-04-refcv8-perception-architecture/raw/pgrad.json`, refcv7-50,400) |
| map weight alone (ladder) | `--w-map-hires 4`: BEV share 0.175 → 0.833, box/agent AP worse | INHERITED (`H-R8-MAPW-1`) |
| box AP@2 m (EMA-48,852, eval139) | **0.328** [0.301, 0.354]; 0-20 / 20-40 / 40-60 m 0.539 / 0.234 / 0.092 | MEASURED (Q3, hold `hold_perc_diag`, `raw/RESULT_Q3.json`) |
| box duplicates | 2.14 boxes / detected object at P=R, no NMS; 0.5 gate recall 0.013; NMS 2.5 m + gate → AP 0.447, 1.088 boxes/object | INHERITED (PERC8, R8-PERC8-1) |
| box localisation | \|Δx\| 0.96 / 1.56 m vs \|Δy\| 0.38 / 0.46 m at 0-20 / 20-40 m | INHERITED (WP-D §2.5) |
| box anchors | 300 learned anchors in metres at lr 1e-4; refcv7 median drift 0.031 m | INHERITED; ESTIMATED ≤ ~4.9 m possible |
| nearest visible ground | **~4.3 m** (bottom 43 rows zeroed by C26; flat plane, 1.45 m mount) | ESTIMATED (`ASIS` §1.1) |
| augmentation | none of any kind | MEASURED (source grep, `ASIS` §1.2) |
| training exposure | ≈ 2.08 window-passes total (refcv7 1.08 + refcv8 1.000); box AP still rising at LR 0 (+0.050 in the last 20k steps, refcv7) | ESTIMATED / INHERITED |
| refcv8 schedule hazard | EMA-40k → EMA-48,852 regressed 6 s ADE +0.168 m [+0.111, +0.241] at both inference seeds; choosing 40k on eval139 would select on the test set | INHERITED (`PI_DECISION_QUEUE.md` ~l.2370) |
| anti-cheat on refcv8 | W5 ego-only floor PASS (seeds 0, 1; S2 ADE margin only 2.8 cm); W4 camera-blind PASS (final) | INHERITED (`H-R9-ANTICHEAT-REF-1`; brief: both PASS) |
| cost | refcv8 full 8.94 s/step at batch 16 (~121 Thor-h for 48,852 steps); frozen-perception 4.34 s/step | MEASURED (brief / `refcv9_warm` kit smoke) |
| SAM3 per-frame 2D masks | **not kept**: per-clip working files are removed after export; the per-clip `worldmap.npz` and the per-frame BEV `sam3mapgt.npz` are kept | MEASURED (source, `sam3map_prod.py` docstring + l.239) |

---

## 2. Every change, ranked by expected effect (Rule Zero 5)

Rank = (size of the MEASURED effect the change attacks) × (published transfer to one front camera / ~4.6 k clips) ÷ cost.
Literature gains are each paper's own ablation in its own metric (nuScenes mAP/NDS, KITTI AP3D, mIoU); **they rank levers,
they are never forecasts of our class-agnostic AP@2 m or our 10 cm IoU** (`LIT_DETECTION` §0.6, `LIT_BEVMAP` §10).

| rank | change | measured motivation | literature (table, class) | E-arm |
|---|---|---|---|---|
| **R1** | **Planner-facing map representation**: range bands; drivable as a sigmoid SURFACE + signed-distance field; edge = SDF zero level; thin classes as dilated sigmoid heatmaps + offsets; UNOBSERVED channel (§2.1) | sidewalk 17.9 % vs 1.7 % GT ceiling; edge IoU ~0.001; own boundary recall 29 % vs 96 %; 0-5 m precision 0.251 | BEV-LaneDet T5: 0.05 m cell −14.7 F vs 0.5 m, offset +0.5/+0.9 F; GT lines 3-5 px wide (BEVSegFormer §4.2); PON eq. 2 max-entropy on unobservable cells (PUB-P, NOT banked) | M1 |
| **R2** | **Wire the map and boxes into the planner** (§2.7): `DrivableCritic` ON reading the OWN map, trained on GT-map labels; map decision channels into the planner BEV; box3d slots as planner tokens | GT-map ceiling −16 pts sidewalk; planner reads no map decision and the WEAKER detector (agent AP@2 m 0.131 vs box3d 0.248, refcv7, INHERITED); GT-map injection through the BEV was INERT (R8-P4-GTMAP) | none needed: it is the measured ceiling | E2 K2 (K3) |
| **R3** | **Map gradient-share target + its own BEV encoder** (§2.2) | map ≈ 0 of the trunk; box3d 97 % of the shared BEV; weight alone costs boxes (ladder) | FIN T4 loss set +2.8..+7.2 mIoU (PUB-P, NOT banked, camera+radar) | M2a, M2b, M3 |
| **R4** | **Box de-dup trained in** (§2.4): quality-aware presence, query denoising, training-time BEV-NMS, 3 → 6 decoder layers | NMS floor +0.119 AP@2 m; 2.14 boxes/object | Sparse4Dv3 T5 centre-/yaw-ness +0.8 mAP, DN +0.8; DenseBEV T5a τ 0.1 vs 0.8 +1.7 mAP (PUB-P, banked); DETR: NMS helps only the early decoder layers (PUB-SECONDARY) | B3, B4 |
| **R5** | **Box depth and range** (§2.3, §2.4): box-derived foreground-depth supervision; perspective 3D auxiliary head; per-frame queries; s8 tokens on the horizon band | AP 0.539 → 0.092 over 0-60 m; \|Δx\| 2.5-3.4× \|Δy\| | CaDDN §4.3 depth-distribution sup. +10.4/+7.6/+6.5 AP3D; Sparse4Dv2 T3 −8.5 mAP without dense depth; BEVFormer v2 T2 perspective head +1.9 mAP, BEV&BEV control ≈ 0; Far3D T3 adaptive queries +2.1 mAP; MonoDLE T1 GT depth 11.12 → 38.01 AP3D (PUB-P, banked) | B1, B2 |
| **R6** | **Schedule: WSD + 1.5 window-epochs**, checkpoint chosen on DEV, never on HOLD (§2.6) | 40k → 48,852 regression; AP still rising at LR 0 | WSD (PUB-SECONDARY); schedules 24-250 ep (PETR Fig 5a, Sparse4Dv3, MonoDETR; PUB-P) | full run |
| R7 | Map loss set: area softmax CE + Dice; thin sigmoid focal + Dice; L1 on SDF / offsets; band-normalised; per-class per-band thresholds fit on a FIT split | thin classes under-predicted; `prior_corrected` hid them | FIN T4; BEVFusion multi-binary + best-IoU threshold (PUB-P, NOT banked) | inside M1 |
| R8 | Blind zone 0-4.3 m: UNOBSERVED channel (all branches); temporal BEV warp (ruling D3) | 21.5 % of the near corridor is camera-invisible; `--w-map-fp` (D16) teaches "no kerb ahead" | HDMapNet +3.5 IoU (4 frames), StreamMapNet +4.4 mAP, MapTracker +6.2 mAP (PUB-P, NOT banked) | M1; T1 |
| R9 | Augmentation: photometric for all; horizontal flip for PERCEPTION losses only | none today; the run goes multi-epoch | BEVDet T4: no-aug peaks at epoch 4 then decays, image+BEV aug +8.6 mAP; MonoDLE/MonoFlex/CaDDN: flip is the small-data recipe (PUB-P, banked) | A1 (iff E0a) |
| R10 | Per-frame pitch: a vision-only pitch head supervised by egomotion pitch, correcting the lift | 1° at 40 m → ground sampled at 27-77 m (ESTIMATED) | none | iff E0c |
| R11 | Class weights tempered (sqrt inverse-frequency instead of 1,299:1) | class not collapsed; person→rider 18 % | PON / PanopticBEV sqrt rule (PUB-P) | inside B3 |
| R12 | Map tap: FPN (layer2 + up(layer3) + up(layer4)); image-plane auxiliary on the world map re-projected | Q1 branch (b2) only | MapTRv2 T8 PV+BEV aux +2.6..+4.9 mAP; MapNeXt T5 map-close pretraining +1.8..+4.6, detection pretraining −3.0 (PUB-P, NOT banked) | F1 |
| R13 | Trunk swap / foundation encoder | Q1 branch (b2), and only if R3 + R12 fail | MonoCoP C.1 DINOv2 no gain vs LSS-foundation +8.9 IoU: contradictory (PUB-P) | not in refcv9 |

### 2.1 Map representation and size

| item | today (refcv8) | refcv9 |
|---|---|---|
| extent | 100 m × ±30 m, uniform 10 cm, 1000 × 600 logits | **same extent, three bands**: 0-10 m at 0.10 m (100 × 600), 10-40 m at 0.25 m (120 × 240), 40-100 m at 0.50 m (120 × 120) = **103,200 cells vs 600,000** (ESTIMATED, `LIT_BEVMAP` §9.2). The 10 cm band keeps the existing near lift (narrowed 20 → 10 m); the 0.25 m band is the lift's native grid; the far band is 2×2-pooled. |
| why | — | one stride-8 feature row spans 0.27 / 1.09 / 4.4 / 17.5 m of ground at 5 / 10 / 20 / 40 m; one s8 column 0.08 / 0.16 / 0.33 / 0.65 m (ESTIMATED, cylindrical, 488.9 px/rad). 10 cm output is supported only inside ~6-8 m. |
| class structure | one 8-way softmax, lane / crosswalk / arrow cells are NOT drivable (D10) | **channels, not one code**: `drv` sigmoid = drivable SURFACE (codes 1, 2, 3, 4, 6, the PI's MUST tier) + `drv_sdf` (signed distance to the drv boundary, clipped ±2 m, L1); `sdw` sigmoid (code 7); `lane` dilated heatmap (code 2) + 2 sub-cell offsets; `edge_h` dilated heatmap (code 5, secondary); `xwalk`, `arrow`, `hatch` sigmoids |
| thin-class targets | 1-2 px hard labels at 10 cm | targets dilated to ~0.5 m wide (≈ the s8 localisation σ at 20 m) at every band; offsets carry the sub-cell position (BEV-LaneDet T5) |
| edge | a 0.36 %-frequency class at IoU ~0.001 | **the zero level set of `drv_sdf`** — the boundary of a class predicted at ~0.9 IoU; MAPEDGE measured that the own drivable boundary already recalls 29 % of GT edges (vs 0.2 % for the edge class) and the GT boundary 96 %. `edge_h` stays as a secondary channel. HYPOTHESIS for our head; building blocks PUB-P |
| blind zone | lift-invalid cells unsupervised, the head emits noise there; MAPSEL read them as seen | **`unobs`** = the lift's own validity mask for this frame (geometric, vision-only, not learned) is emitted WITH the map; losses ignored there plus a weak max-entropy term (PON eq. 2, weight 1e-3); every consumer (critic, planner channels, scorers) reads `unobs` as "no evidence", never as sidewalk |
| `--w-map-fp` | added for the warm start (aeb5446) | **retired** in refcv9: it is one-sided, not lift-masked, and 21.5 % of its corridor rows are camera-invisible (D16); `drv_sdf` + `sdw` + `unobs` replace it. Kept only as a fallback E1 arm if M1 misses at 5-20 m. |
| decision rule | `prior_corrected` (hides thin classes, D19) | per-channel, **per-band thresholds fitted on a FIT split of TRAIN clips** (`--map-hires-class-thresholds` machinery); never on eval |
| planner read | pooled 0.25 m BEV features only; 68 % of the mapped area read by no consumer (ESTIMATED) | the decision channels `drv`, `drv_sdf`, `sdw`, `lane`, `unobs` resampled to the planner's grid (§2.7) over the full 100 m × ±30 m (the 60 m × ±16 m crop is exceeded by 6 s anchors above 10 m/s, D8; E0b sizes it) |
| scoring | pixel-exact 10 cm IoU | **published protocol** beside the old one: GT dilated to 0.5 m for thin classes, per-class best threshold on FIT, recall/precision @0.3 m, bands 0-5 / 5-10 / 10-20 / 20-40 / 40-60 / 60-100 m; plus the planner-facing `nearfield_prec` corridor reads |

### 2.2 Map losses and the loss balance

* **Loss set (R7):** `drv`, `sdw`, `xwalk`, `arrow`, `hatch`: sigmoid focal (γ 2) + soft Dice; `lane`, `edge_h`: sigmoid
  focal + Dice on the dilated targets; `drv_sdf`, offsets: L1 on cells within 2 m of a boundary / line. Each band's loss is a
  mean over its own cells; band weights 1 / 1 / 0.5 (near / mid / far) so the far field (lowest image support) cannot
  dominate. Lovász and the normalised boundary loss are E1-follow-up arms on the M1 winner, not in M1 (FIN T4 orders them
  after Dice).
* **Map BEV capacity (M3):** today one 64-channel, 538,112-parameter BEV feeds the map, the planner BEV, the box tokens and
  the tactical decoder, and box3d owns 97 % of it. refcv9 gives the map **its own BEV encoder at 128 channels** reading the
  shared lift (lift output widened 64 → 128); the box head and the planner keep the existing 64-channel encoder. ESTIMATED
  +~1.6 M params, +0.3-0.6 s/step (§4.5). The shared parameters are then the trunk and the lift only — where the conflict
  the ladder measured (`H-R8-MAPW-1`: weight 4 bought the BEV and cost the boxes) can be read directly.
* **Gradient-share TARGET (the PI's ask "map weight such that the map actually shapes the trunk and lift"):**

  | group | target for the map term | refcv8 | floor kept for box3d |
  |---|---|---|---|
  | trunk layer2 (the map's own tap) | **≥ 0.05** projected share | 0.0010 (P-GRAD, refcv7-50,400) | ≥ 0.20 |
  | whole trunk | ≥ 0.02 | 0.00018-0.00396 (refcv8 run) | ≥ 0.20 |
  | lift (shared, 131,200 params ESTIMATED) | ≥ 0.25 | box3d 0.973 owns the lift + encoder | ≥ 0.25 |

  The 0.05 is a judgement (50× today at the tap, still a minority of the update), tested at two doses in E1 (M2a 0.05,
  M2b 0.15). HYPOTHESIS that this suffices; the arms read it.
* **How the weight is chosen (no training needed):** extend `pgrad.py` to store the per-group **Gram matrix** of the term
  gradients ⟨g_i, g_j⟩ (n_terms² inner products) at w = 1 on 64 windows of refcv8 EMA-48,852 with the NEW map loss. Then for
  any weight vector the projected share is exact: s_i(w) = w_i ⟨g_i, Σ_j w_j g_j⟩ / ‖Σ_j w_j g_j‖². Choose the SMALLEST
  `w_map` meeting the target with the box3d floor held; controls as P-GRAD (linearity ≤ 1e-4, zero-weight exactly 0).
  ⚠️ G-SMOKE-L4 failed linearity at 4.2e-4 on refcv8's step-50 replay — a failed linearity control voids the reading and
  the weight falls back to the E1 dose arms.
* **How it is measured in-run:** `--grad-share-every 6000 --grad-share-terms map_hires,box3d,agent,tac_v6,traj` (refcv8
  passed only `tac_v6,map_hires`, so box vs agent vs planner were folded into `rest`). One refcv8 reading cost 775 s
  (MEASURED, `/e/r8launch/gs_measure.json`); with 5 separated terms ≈ 7/4 × 775 ≈ 1,360 s (ESTIMATED), 12 readings ≈ 4.5 h.
  **Liveness rule for the gate profile:** if the median of two consecutive layer2 readings is below 0.25 × target, the
  supervisor raises an alarm row (not a kill); the reading is a monitor, not a controller — a dynamic weight (GradNorm-style)
  would make every later attribution conditional on a moving weight, and is held as a fallback arm only.

### 2.3 The lift: depth and pitch

* **Box-derived foreground depth (B1; no new data).** Ray-cast every `obstacle.offline` cuboid through the **cylindrical**
  model (column linear in azimuth, 120° HFOV — NOT the pinhole formula, which gives 92.6°) into the 416 × 1024 frame;
  per-pixel nearest-surface depth on object pixels, LID bins over 0-100 m, CaDDN weighting α_fg 3.25 / α_bg 0.25, soft
  (categorical) targets, applied to a training-only depth head on s8 and s16 (Sparse4Dv2's form: an auxiliary on the image
  features, removed at inference). It shapes the features the lift samples and the box memory reads. Analytic control
  before use: a cuboid placed at a known range must return that range at its centre pixel, and a frame with no box must
  produce zero foreground pixels (§4.1, E0g).
* **LiDAR projected depth (B1b; PI decision D4).** Dense ground + object depth from `lidar_top_360fov` would also supervise
  the ground the MAP lift samples (LSS T5 oracle depth: lane 19.58 → 25.12, drivable 70.81 → 74.91; PUB-P, NOT banked). It
  needs a data build; the raw sweeps are on Thor / HF and NOT on the dev box for eval139 (MEASURED, Q2 blocker). Labels only;
  inference stays vision-only.
* **No LSS rewrite.** The lift stays parameter-free geometric sampling at 4 heights; BEVHeight-style height lifts are worse at
  ego height (BEVHeight T7 −2.4 mAP; PUB-P). A depth-distribution lift is a refcv10 question once B1 reads.
* **Pitch (R10).** The lift uses the clip's static extrinsic pitch (D5). E0c measures the per-frame egomotion pitch spread;
  if its p95 exceeds 0.25°, a **vision-only pitch head** (regressing the frame's pitch offset from the s32 tokens,
  supervised by the egomotion pitch — labels may use ego) corrects the lift geometry. Feeding egomotion pitch at inference is
  NOT proposed (it is outside the admitted v0 / a0 / yaw-rate set and would need a ruling).

### 2.4 The box head

| item | refcv8 | refcv9 | motivation / literature |
|---|---|---|---|
| family | 300-query DETR, Hungarian, 3 layers | **keep** the slot decoder | head-family gap ~1-2 NDS at matched inputs (SparseBEV §4.3, BEVNeXt §4.3; PUB-P); the planner seam is slot-based |
| depth | 3 decoder layers | **6** | decoder depth is what lets set prediction drop duplicates (DETR, PUB-SECONDARY); Sparse4Dv3 uses 6. Needs D2 |
| presence target | binary focal | **quality-aware**: presence target = exp(−d_centre / 1 m) on the matched slot (centre-ness), yaw-ness term | Sparse4Dv3 T5 +0.8 mAP; "top score is the nearest slot 34.5 %" (INHERITED, refcv7) |
| denoising | none | 3 DN groups, 2 range-modulated negatives per GT, training-only | Sparse4Dv3 T5 DN +0.8 mAP; Far3D T3/5 +1.0 mAP |
| training-time NMS | none | BEV-NMS τ 0.1 inside the loss with the attention mask (DenseBEV) — **B3 tests it; if B3's DN + quality alone reach ≤ 1.25 boxes/object, the in-loss NMS is dropped** (fewer moving parts) | DenseBEV T5a +1.7 mAP |
| queries / anchors | 300 learned anchors in METRES at lr 1e-4 (moved 0.031 m) | **per-frame**: top-K = 200 (centre, depth) peaks of the perspective head (B2) lifted to 3D seed the slots; **100 learned global** queries with anchors in normalised [0, 1] coordinates at lr × 10 | Far3D T3 +2.1 mAP; PETR T5c no anchors = no convergence (PUB-P) |
| perspective 3D aux head (B2) | none | FCOS3D-style on s8 + s16: projected-centre heatmap, depth (LID), size, yaw (sin, cos), centre-ness; GT through the cylindrical model; training-only except as the query seeder | BEVFormer v2 T2 +1.9 mAP / +2.5 NDS, control BEV&BEV ≈ 0 (PUB-P) |
| memory | s16 tokens (1,664) + 480 BEV tokens (2 m cells), learned per-token PE | + **s8 tokens on the horizon band** (the 16 s8 rows around the rig's horizon row × 128 = 2,048 tokens: where 20-60 m objects sit); + camera-ray PE (zero-init) on image tokens | far AP 0.092; a car at 50 m is ~18 px wide (≈ 1 s16 token, 2 s8 tokens; ESTIMATED); PETR T3 ray PE (PUB-P; expected small, our BEV tokens already carry geometry) |
| decode range | x 0-60 m, \|y\| ≤ 16 m (16.4 % of raw join boxes survive) | **x 0-80 m, \|y\| ≤ 24 m** provisional; E0b's census of lead vehicles / 6 s paths beyond 60 m fixes it before sealing; GT beyond the range is IGNORE, never negative | D8; MonoDLE T8: drop GT > 60 m from the det. loss +0.7, > 40 m hurts (PUB-P) |
| loss shape | metre L1 centre / size | x/y weight 2.0; depth-direction error up-weighted (x weight 2.0 is along range) | SparseBEV §4.1; FCOS3D T3 (PUB-P) |
| class | CE, pure inverse-frequency (1,299 : 1) | CE with **sqrt** inverse-frequency (mean 1); per-class readout + confusion (person↔rider) every eval | class not collapsed (0.822 vs 0.726, MEASURED); PON's rule (PUB-P) |
| size cap | `PARAM_BAND` 2-4 M at construction (3,858,199 today) | **≤ 12 M** for box3d (ESTIMATED: 6 layers ≈ 7.0 M; perspective head ≈ 1.5 M; depth head ≈ 0.5 M) + Thor latency measured at the gate | D2 ruling |

### 2.5 Augmentation (R9, gated by E0a)

* **Photometric** (brightness / contrast / saturation ±0.2, hue ±0.02, light blur), applied to the whole sample; label-
  invariant, every loss keeps it.
* **Horizontal flip, perception losses only:** with p = 0.25 a sample is mirrored (image, boxes y → −y and yaw → −yaw, map
  columns, the cylindrical geometry is symmetric about the optical axis); on a flipped sample **every planner, tactical and
  goal loss is masked**, because a mirror turns right-hand traffic into left-hand traffic and swaps nav left/right
  (`LIT_BEVMAP` §9.4 warns of exactly this). ESTIMATED cost: 25 % fewer planner samples per step.
* **No BEV rotation / scale augmentation**: the BEV is shared with the planner and the map (`LIT_DETECTION` lever 7 trap).
* **Gate:** E0a measures the train-vs-held-out AP gap on refcv8 EMA-48,852 with equal window counts. If the gap is inside the
  bootstrap half-width, A1 is not run and only photometric (free, label-invariant) enters the full run, disclosed.

### 2.6 Trunk init, LR, BN, schedule

* **Init: refcv8 EMA-48,852 in BOTH Q1 branches** (md5 b4364029b68f29a33eeb4aebbbbd002b, INHERITED from the warm-start
  draft). Every new module is zero-gated / zero-init so step 0 reproduces refcv8's forward (a gate control, K-STEP0).
  Rationale for branch (b) too: the trunk has never been trained BY the map (0.014 % of its update at refcv7; at refcv8
  −0.11 % to +0.40 % over four readings, median ≈ 0.02 %), so a "feature ceiling" measured on these features is a property of the training mix, not of the R101. Re-initialising
  from ImageNet would discard ~2.08 passes of detection training while the box AP was still rising.
* **LR:** AdamW, heads 1e-4, trunk × 0.5 (5e-5; DiffusionDrive's recipe; MapTR / BeMapNet use × 0.1 — kept at 0.5 because the
  trunk must now move for the map). New modules (perspective / depth heads, map BEV encoder, DN) at the head LR; normalised
  anchors at × 10.
* **BN:** frozen at ImageNet stats and folded, γ/β trained (unchanged). Batch 16 is too small for live BN statistics; frozen BN
  is the detection norm. `--trunk-bn-recalib` stays off (D20, LOW).
* **Schedule (R6): warm-up 1,000 → WSD** (stable at peak, then a linear decay over the last 17 % of the run). The stable phase
  makes **decay branches** possible: the early read D1 decays from stable step 30,000 over 6,000 steps, and the registered
  checkpoint is the final decay end (73,280). Measured reason: refcv8's cosine left one pre-registered endpoint and a 40k
  checkpoint that scored better on the test set, which could not be chosen without selecting on eval139. ⚠️ WSD is
  PUB-SECONDARY (§7) and needs a trainer flag (code item C6).
* **Length:** 73,280 steps × 16 = 1.5 window-epochs (781,635 windows / epoch). Literature trains query heads 24-250 epochs of
  images; our windows are ~95 % overlapping, so exposure is better counted in clips: 1.5 epochs ≈ 256 supervised frames per
  clip (ESTIMATED). Option offered in D1: 1.0 epoch (48,852 steps, ≈ 143 Thor-h).

### 2.7 Planner wiring (PI 10-25: "the planner must use the map head")

| wire | what | admissibility |
|---|---|---|
| **W-MAP-1 critic** | `--r8-critic-drivable` ON (`refcv8_conditioning.py:717`, landed, left out of refcv8 only by the ladder stop): per-candidate critic on the OWN map, **detached**, trained by BCE on the SAM3 label of the candidate's footprint; its selection term is zero-init. **Changed input:** P(`drv`) and `drv_sdf` at the footprint corners from the refcv9 channels instead of an 8-way softmax sum, and corners on `unobs` cells read "no evidence" (today's code already reads out-of-range corners as (1, 1, 0)). | inference reads only the camera-derived map; the GT map is a LABEL (privileged labels admitted, PI 2026-08-03) |
| **W-MAP-2 channels** | the decision channels `drv`, `drv_sdf`, `sdw`, `lane`, `unobs` (detached, so the planner loss cannot bend the map toward trajectory convenience) resampled to 1 m over 100 m × ±30 m and concatenated into the planner BEV pool through a zero-init 1×1 projection | camera-derived; carries no situation-classifier output |
| **W-BOX** | top-32 box3d slots by quality-aware presence → tokens (slot embedding + MLP of x, y, l, w, yaw, v_rel, class) into the planner's and the tactical decoder's agent-token set, zero-gated beside the agent head's tokens (E2 K2). **K3 (optional):** the agent head is retired and the planner reads box3d only — it owns 50.1 % of the trunk update (refcv7 P-GRAD) while being the weaker detector | camera-derived |
| **W-COST (train only)** | the existing GT-constraint losses stay; the critic's BCE weight `--w-r8-drivable` set so the critic's BCE is ~0.1 × the trajectory loss at step 0 (fixed from the smoke, never from eval) | label-side only |

**Goal-input ruling (2026-08-03):** the goal path stays nav command + distance (no route checkpoint, PI 2026-10-11). The map and
box channels are scene perception, not the situation classifier's output; nothing above routes the tactical decoder's class
posterior, argmax or embedding into the goal. **Vision-only ruling:** at inference the planner sees the camera, the admitted t0
ego scalars (v0, a0, yaw rate), the nav command and the speed band — the band is **future-derived** (the ego's own max speed
2-6 s ahead, leak 0.224 vs the 0.05 bar) and every refcv9 number states it (PI 2026-10-27 07:25).
⚠️ **Camera-blind instrument consequence:** in W4 the donor window must supply ALL camera-derived inputs — trunk tokens, the
BEV, the map channels, the critic's map read and the box tokens. Swapping the trunk tokens alone would leave the map and box
channels as a camera leak in the "blind" arm. This is an instrument change (C9), tested by a mutation that swaps only the
trunk tokens and must read the blind arm BETTER than the correct instrument.

### 2.8 Carried over from the refcv9 warm-start work

| lever | status in refcv9-full | why |
|---|---|---|
| `--r9-alloc-force-schedule linear:1.0:0.0:0:36640` | **carried** (end = floor(73,280 / 2)) | `H-R9-ALLOC-ORACLE-1`, `H-R9-L2B-LOCUS-1`: the selector is trained only on pools that contain the right class. Not attributable inside the full run; REFCV9-CONT is not re-proposed (a 214 h attribution arm is not proportionate) — disclosed as "pool change not separately attributed" |
| `--r8-alloc-emit-start 0`, `--r8-tf-start 0.25 --r8-tf-end 0.25` | **carried** | warm-start hygiene from the draft: do not restart the emit delay or the teacher-forcing ratio |
| R9-PF stage 2 (70,299-param progress head) | **carried**, dev box, 0 Thor-h | refit on refcv9's OWN `z_tac` on the clean calibration clips (L1C 300 + FRESH2 GROW); never on the trunk's own clips (`refcv9-progress-head-needs-clean-split`) |
| FUSE decel decision (`H-R9-FRESH2-1`, F-1 PASS +0.0643 [+0.0423, +0.0847]) | **carried, re-read** | F-1 is re-read on CONFIRM with refcv9's LH before it is written in; disclosed: raw −a0 alone read 0.7341 vs FUSE 0.6981 there, the gain is ego dynamics, which is why W4 / W5 bind |
| `--r9-train-only` sets, `refcv9-warm` gate profile | **not used** | full training; a new `refcv9-full` profile is needed (D5) |
| `--w-map-fp` | **retired** (§2.1) | D16 |
| H2cn stage | reference only | refcv8 EMA-48,852 + H2cn is the comparison reference |

---

## 3. Q1 branches (pre-committed before the verdict is read)

Q1's sealed outcomes (`PREREG_PERC_BOTTLENECK.md`, sha256 `5489cf56ae6afdf8…`): **O1 HEAD-LIMITED** = Δ AP (sidewalk, best probe −
head) ≥ +0.05 with the CI above 0; **O2 FEATURE-LIMITED** otherwise; localisation `img` − `enc` ≥ +0.05 ⇒ the lift or BEV
encoder loses the information, else the image trunk is the ceiling; **A0 VISIBILITY-LIMITED** if > 50 % of the head's 0-5 m
sidewalk flags sit on lift-invalid cells.

| Q1 outcome | what it means | the map levers, in order | trunk init | E1 arms added / dropped |
|---|---|---|---|---|
| **(a) O1 HEAD-LIMITED** (probe ≫ head) | the information is in refcv8's features; the head/loss throws it away | M1 representation + loss (R1, R7) first; M2 gradient share second (still needed: the planner needs the map to shape the features over training, and the probe ceiling is a frozen-feature ceiling) | **refcv8 EMA-48,852**, trunk × 0.5 | + an **M1-FROZEN** arm on the map-only fine-tune rig (MAPFT kit, ~3.97 s/step MEASURED): M1 on frozen features, 4,000 steps, ≈ 4.4 Thor-h — the cheapest possible read, because in this branch the head is the lever; F1 dropped |
| **(b1) O2, `img` − `enc` ≥ +0.05** | the trunk features carry it; the 64-channel lift / BEV encoder loses it | M3 (own 128-ch map encoder, lift 64 → 128) first; M1; M2 | refcv8 EMA-48,852 | M3 promoted to the primary map arm; F1 dropped |
| **(b2) O2, trunk is the ceiling** | refcv8's features do not carry it | ranked by measured effect then literature: **(1) M2 gradient share** (the trunk was never trained by the map: ≤ 0.4 % of its update at any refcv8 reading); **(2) F1** FPN tap (layer3/4 semantics into the map) + image-plane auxiliary on the world map re-projected into the frame (MapTRv2 T8; per-frame SAM3 masks are NOT kept, so re-projection is the only image-plane target and loses some alignment — HYPOTHESIS); **(3) B1** depth supervision; **(4) T1** temporal (D3); **(5)** trunk re-init — MapNeXt T5 says detection pretraining HURTS map extraction (−3.0) and map-close pretraining helps (+1.8..+4.6), which argues against our detection-dominated trunk, but it costs the boxes and a full from-ImageNet run (~250+ Thor-h); **not in refcv9**, a refcv10 arm only if (1)-(3) fail; **(6)** foundation encoder — contradictory evidence (MonoCoP C.1 vs LSS-foundation), last | refcv8 EMA-48,852 (see §2.6) | F1 added (≈ 3.3 Thor-h) |
| **A0 VISIBILITY-LIMITED** (any branch) | 0-5 m flags are where the camera cannot see | `unobs` channel (all branches, structural); T1 temporal is the only lever that ADDS evidence there | — | T1 promoted to required — iff D3 is ruled YES; otherwise 0-5 m is reported as "no evidence by construction" and MAP-2's 0-5 m clause is read on lift-valid cells only (stated in the seal) |
| **Q1 WITHHELD** (C1 or C4 failed) | no verdict | run the union of (a) and (b1)'s arms (M1-FROZEN + M3), ≈ +4.4 Thor-h | refcv8 EMA-48,852 | as stated |

**Box head: no branch.** Q3 is final: not class-collapsed, range-limited, duplicate-limited. R4 and R5 run in every branch.

---

## 4. Staged plan

### 4.1 E0 — CPU / dev box, 0 Thor-h (each item names the decision it makes)

| id | what | decides |
|---|---|---|
| E0a | over-fit gap: box AP@2 m on train vs held-out windows (equal counts) on refcv8 EMA-48,852 | A1 runs or not |
| E0b | range census over train windows: share whose lead vehicle or GT 6 s path lies beyond 60 / 80 / 100 m or \|y\| > 16 / 24 m | the box decode range and the planner map extent |
| E0c | per-frame egomotion pitch / roll minus clip mean over the train clips | R10 pitch head in or out (p95 > 0.25°) |
| E0d | re-score refcv8's banked map rasters under the published protocol (thin GT dilated to 0.5 m, per-class threshold on FIT, @0.3 m, bands) | the baselines for MAP-3 |
| E0e | `drv_sdf` / dilated-target / offset label transform from `fine_codes`, with an analytic control (a straight GT kerb must give an SDF that is linear in lateral distance with slope 1; a lane line of known width must dilate to exactly 0.5 m) | M1 can be built |
| E0f | anchor drift on refcv8 EMA (`box_refpts.xy` vs init); `TemporalFuse` older-frame weight norms (D12, D21) | confirms the anchor lever is live in refcv8 too |
| E0g | box-derived depth target builder (cylindrical ray-cast) + the analytic control in §2.3 + a content assertion (non-zero foreground share per frame, printed mean) | B1 can be built |
| E0h | **map-quality dose-response**: on MAPSEL's banked fan, replace the GT raster by GT with calibrated false sidewalk flags at precision p ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95} per band, re-run MAPPATH's cross-fitted soft cost, read the sidewalk rate | **the map precision the planner actually needs** — it sets MAP-2's absolute value from the planner, not from taste |
| E0i | DEV / HOLD split of eval139 by sha256(clip_id) rank parity (even = DEV, odd = HOLD), sealed and hashed before any E1 read | the selection / confirmation protocol (§5.1) |
| E0j | Gram-matrix P-GRAD on refcv8 EMA-48,852 with the new map loss (dev box if 64 windows fp32 fit in 8.6 GB with accumulation; UNVERIFIED — else ~0.5 Thor-h) | `w_map` (§2.2) |

### 4.2 Code items (each lands with tests and a deliberate-regression arm per `TanitAD_ValidateAIDesign`)

C1 banded map head + channels + `unobs` + losses · C2 own map BEV encoder (128 ch) + lift width flag · C3 box-derived depth
targets + depth aux head · C4 perspective 3D aux head + per-frame query seeding + normalised global anchors · C5 decoder depth,
DN groups, quality-aware presence, optional in-loss BEV-NMS, s8 horizon tokens, ray PE · C6 WSD schedule + decay-branch resume ·
C7 perception-only E1 rig (newest row, K = 3 frames, planner / tactical not built) · C8 wiring W-MAP-1 / W-MAP-2 / W-BOX ·
C9 camera-blind instrument swaps ALL camera-derived inputs · C10 flip-with-planner-mask + photometric aug · C11 `pgrad.py` Gram
output · C12 `refcv9-full` launch-gate profile (D5) with G-LIVE rules for every new weight and the share-alarm row.
⚠️ C1 changes the map OUTPUT contract: MAPSEL / MAPPATH / `nearfield_prec` scorers need an adapter that reads `drv` / `sdw` /
`unobs` (and refcv8's 8-way raster through the same adapter, so both arms are scored by one code path).

### 4.3 E1 — perception rig on Thor (one variable off a control, read against a replicate)

Rig (C7): init refcv8 EMA-48,852, newest window row only (K = 3 frames instead of 10 distinct frames), planner / tactical off,
batch 16, warm-up 200, cosine to 0 over **4,000 steps**, EMA 0.999. Read on **DEV** (eval139 even half), T0 label level,
paired episode-cluster bootstrap B 2000 seed 0. Perception inference is deterministic, so the floor that matters is TRAINING
variance: **C1 = C0's flags with training seed 1**. Keep a lever iff its primary is separated in the good direction **and**
\|Δ\| > 2 × \|C1 − C0\| on that metric **and** no other perception primary is separated worse beyond 2 × its floor.

| arm | variable (off C0 unless stated) | primary metric (DEV) |
|---|---|---|
| C0 / C1 | refcv8 recipe on the rig, seeds 0 / 1 | the floor |
| M1 | §2.1 representation + R7 losses at w = 1 | corridor sidewalk+edge precision 5-20 m (lift-valid); beside: lane tolerance-IoU 0-20 m, boundary recall @0.3 m |
| M2a / M2b | M1 + `w_map` at the 0.05 / 0.15 layer2 share target | M1's primary vs M1; **box AP@2 m not worse** (the ladder's cost) |
| M3 | M2 winner + own 128-ch map encoder, lift 64 → 128 | M1's primary vs the M2 winner; box AP@2 m |
| B1 | box-derived depth aux | \|Δx\| on matched TPs 0-40 m; AP@1 m |
| B2 | perspective 3D aux head (aux only) | AP@2 m 20-60 m |
| B2c (iff B2 passes) | B2's control: a second BEV head instead of the perspective head (BEVFormer v2's BEV&BEV) — if B2c gains as much, the gain was capacity, not view | B2's primary |
| B3 | quality-aware presence + DN (+ in-loss BEV-NMS as B3') | boxes / detected object at P=R, no NMS; AP@2 m no NMS |
| B4 | B2 winner + per-frame queries + 6 layers + s8 horizon tokens (needs D2) | AP@2 m no NMS, by range |
| A1 (iff E0a) | §2.5 aug | AP@2 m DEV; map primary |
| F1 (iff Q1 = b2) | FPN map tap + re-projected image-plane aux | map primary |
| T1 (iff D3 = YES) | 2 past frames' BEV (t − 0.5 s, t − 1.0 s) warped by past ego-motion into the current frame | corridor precision 0-10 m incl. lift-invalid cells |
| M1-FROZEN (iff Q1 = a or WITHHELD) | M1 on the MAPFT frozen-feature rig | M1's primary |

### 4.4 E2 — full config screen on Thor, then the full run

* **E2** (init refcv8 EMA-48,852, full config incl. planner, 6,000 steps, WSD with decay over the last 1,000): **K0** refcv8's
  recipe continued; **K2** every E1 winner + W-MAP-1 / W-MAP-2 / W-BOX + the carried planner levers; **K3** (optional) K2 with
  the agent head retired. Read on DEV at inference seeds 0 and 1: **GO iff** K2 beats K0 in point estimate on the map primary,
  on box AP@2 m and on the planner's sidewalk rate (S6c-DEV), no four-family row is separated worse, and a partial W4 preview
  (K2 vs its camera-blind) is separated. NO-GO ⇒ diagnose and fix before the 214 h are spent (Rule Zero: the NO-GO names the
  failing clause and the next lever, it does not end the programme item).
* **E3, the full run:** a 50-step smoke (MEASURED s/step, `w_map` share check, G-LIVE), the `refcv9-full` gate, then 73,280
  steps. **D1 early branch** at stable step 30,000 decayed over 6,000 steps, read on DEV: continue iff D1's map primary and box
  AP@2 m are ≥ K2's and its sidewalk rate ≤ K2's; else stop the stable phase and diagnose. D1 is a monitor, never a deliverable.

### 4.5 Cost (ESTIMATED; every input stated)

**Step-cost model.** refcv8 full 8.94 s/step, frozen-perception 4.34 s/step (MEASURED) ⇒ perception training ≈ 4.60 s/step,
mostly the trunk backward over 10 distinct frames per sample. refcv9 adds: perspective + depth heads +0.2-0.4 s; decoder 6
layers + DN + s8 tokens +0.2-0.4 s; own 128-ch map encoder +0.3-0.6 s (≈ 4× the 64-ch encoder's ~3.9e10 MAC/sample on 400 ×
240 cells); FPN +0.1-0.2 s (branch b2 only); wiring +0.1 s; banded map decoder (103,200 vs 600,000 cells) −0.1-0.2 s ⇒
**≈ 9.6-10.5 s/step, central 10.1** (the smoke MEASURES it before launch). E1 rig: 3 of 10 frames through the trunk ⇒
≈ 3.0 s/step (ESTIMATED: trunk ≈ 6.6 s of fwd+bwd at 10 frames scaled × 0.3, + heads + overhead).

| stage | arithmetic | Thor-h |
|---|---|---|
| E0, C-items | CPU / dev box (E0j may need ~0.5) | 0-0.5 |
| E1 smoke | 50 steps per flag set + gate | ~0.5 |
| E1 core (C0, C1, M1, M2a, M2b, M3, B1, B2, B3, B4) | 10 × 4,000 × 3.0 s = 120,000 s | **33.3** |
| E1 conditional (A1, F1, B2c, M1-FROZEN at 3.97 s/step, T1 at ~6 s/step) | 3 × 3.3 + 4.4 + 6.7 | ≤ 21.0 |
| E2 K0 / K2 | 6,000 × 8.94 + 6,000 × 10.1 = 114,240 s | **31.7** |
| E2 K3 (optional) | 6,000 × 10.1 | 16.8 |
| E3 smoke + gate | — | ~2 |
| **E3 full run** | 73,280 × 10.1 = 740,128 s = 205.6 h, + in-run evals, saves and 12 gradient-share readings ≈ 8 h | **≈ 214 (203-222)** |
| D1 early branch | 6,000 × 10.1 | 16.8 |
| **total, core** | 0.5 + 0.5 + 33.3 + 31.7 + 2 + 214 + 16.8 | **≈ 299** |
| total with every conditional | + 21.0 + 16.8 | ≈ 337 |
| optional full-length seed-1 replicate (§5.3) | as E3 | +214 |

Wall-clock: ~12.5-14 days of sequential Thor (co-located GPU jobs are net-negative on Thor, MEASURED earlier). The dev-box
items (E0, the eval batteries, W4 / W5, R9-PF stage 2) run in parallel at 0 Thor-h. Cheaper option: a 1.0-epoch full run
(48,852 × 10.1 s + ~6 h ≈ 143 Thor-h) ⇒ total ≈ 228 Thor-h.

---

## 5. Pre-registered bars for the full refcv9 (DRAFT — sealed by the Master Mind after E2, before E3 fires)

### 5.1 Protocol

* **Reference:** refcv8 EMA-48,852 + the shipped H2cn stage; its banked registered table (`REGISTERED_PF_TABLE.sha256`) must
  reproduce (K-BASE) or the bars are void.
* **Scored weights:** refcv9 EMA at the final decay end (73,280). No other checkpoint is a candidate on HOLD.
* **Splits:** perception and map bars on **HOLD** (eval139 odd half, never read by E1 / E2 / D1); all of eval139 beside. The
  four families, W1-W5 on all of eval139 (comparability with refcv8's registered table) — E2 read them on DEV only, disclosed.
  Beside, never a bar: the four families on FRESH2 CONFIRM windows.
* **Tier:** perception = T0 label level; planner = T1 self-action open loop on logged eval139 frames (not driving performance).
* **Estimator:** paired episode-cluster bootstrap, B 2000, seed 0, over EPISODES — it answers *another draw of episodes*.
  **Inference seeds 0 and 1 must both hold** for every planner bar (the planner samples; `D-REFAV1-SEED-GOAL-MISMATCH` family).
* **Speed band:** future-derived (leak 0.224), stated on every number.

### 5.2 Bars

| id | bar | reference value | class of the reference |
|---|---|---|---|
| **MAP-1** | planner sidewalk rate on S6c-HOLD (in-model wiring, no post-hoc filter) **≤ 9.8 %** (≥ 50 % of the 17.9 → 1.7 gap) AND separated below refcv8 on HOLD | 17.9 % (all S6c) | INHERITED; refcv8 re-read on HOLD before sealing |
| **MAP-2** | corridor sidewalk+edge precision (`nearfield_prec`, adapter C1): 0-5 m ≥ 0.60 on lift-valid cells; 5-10 and 10-20 m ≥ the precision E0h says halves the sidewalk rate (provisional 0.85); 20-40 m separated above refcv8 | 0.251; 0.733 / 0.716 / 0.744 | INHERITED |
| **MAP-3** | lane tolerance-IoU (0.5 m) at 0-20 m ≥ 1.5 × refcv8 under the E0d protocol, separated; drivable-boundary recall of GT edges @0.3 m at 0-20 m ≥ 0.50 | E0d; 0.29 | E0d MEASURED before sealing; INHERITED |
| MAP-4 | `drv` IoU in every band not separated worse than refcv8's drivable-union IoU | E0d | — |
| MAP-5 | own sidewalk / edge flags on `unobs` cells = 0 — a **structural identity**, not an estimate (no seed changes it; the H-ECHO-4 rule) | — | — |
| **BOX-1** | AP@2 m, head output, NO post-hoc NMS, **≥ 0.45** (it must reach the zero-training NMS floor by training) AND separated above refcv8 | 0.328 | MEASURED |
| **BOX-2** | AP@2 m by range, each separated above refcv8: 0-20 m ≥ 0.62; 20-40 m ≥ 0.32; 40-60 m ≥ 0.15 | 0.539 / 0.234 / 0.092 | MEASURED |
| **BOX-3** | boxes per detected object at the P=R gate, no NMS, ≤ 1.25 | 2.14 | INHERITED |
| BOX-4 (beside) | TP class accuracy ≥ 0.822 − its CI half-width; person→rider confusion reported | 0.822 | MEASURED |
| **FAM-1** | no registered SHIP-R8 row (LONGITUDINAL, LATERAL, TACTICAL, `ade_m`) separated worse at seeds 0 and 1; STRATEGIC n = 0, stated with the reason (no route label) | refcv8 table | registered |
| **FAM-2** | the MAP-COMPLIANCE family (off-drivable, edge contact, lane-marking crossing vs GT) separated better; LONGITUDINAL distance-keeping (headway / time-gap / TTC) not worse | refcv8 | — |
| W1 / W2 | false decels separated lower / missed decels not separated worse (carried from the warm-start draft) | refcv8 + H2cn | — |
| **W4** | separated BETTER than CAMERA-BLIND (C9: every camera-derived input from the donor) on `ade_m` and each family's primary row | — | binding (PI) |
| **W5** | separated BETTER than the EGO-ONLY FLOOR (same instrument as `H-R9-ANTICHEAT-REF-1`) on `ade_m` and each family's primary row | refcv8 passes it | binding (PI) |

**PASS = every bold bar + W1, W2, MAP-4, MAP-5.** A missed bar is reported as FAILED, as written, then the committed next
lever below runs (Rule Zero).

### 5.3 Seed policy (`H-ESTIM-SEED-1`)

* E1: every lever read against the C0 / C1 training-replicate floor (§4.3).
* Full run: ONE training seed. Every refcv9 − refcv8 difference is reported with the E1 floor beside and labelled *"training
  variance at full length not estimated"*. A claim that a specific LEVER moved a metric needs its E1 read; the claim
  "refcv9 is better than refcv8" is a two-checkpoint comparison, and its training-variance answer is the full-length seed-1
  replicate (+214 Thor-h, D1b), proposed only after seed 0 passes.

### 5.4 Committed next steps per failure

| failure | reading | next lever (executed, not reported) |
|---|---|---|
| MAP-2 PASS, MAP-1 FAIL | the map is good; the selector does not use it | critic weight / candidate generation filtered by own `drv_sdf` at train time; dose from E0h |
| MAP-2 FAIL at 0-5 m only | visibility | T1 temporal (needs D3) |
| MAP-2 FAIL at 5-40 m | features | the next Q1-branch lever not yet in the run (§3) |
| BOX-1 PASS, BOX-2 FAIL at 40-60 m | far range | LiDAR depth (D4); long-range adaptive queries with range-modulated DN at 80-100 m |
| BOX-3 FAIL | de-dup not trained in | in-loss BEV-NMS (B3') if not in; more DN groups |
| W4 or W5 FAIL | the planner does not use the camera beyond the ego inputs | reported FIRST, no ship request |
| FAM-1 row worse | name the row | attribute with the trace-fixed method (MAPFT's K-PLAN-ID) and E2 K0 vs K2 |

---

## 6. PI decisions needed

| id | decision | recommendation |
|---|---|---|
| **D1** | compute envelope | **Approve ≈ 300 Thor-h (core), ≤ 337 with conditionals**, ~12.5-14 days sequential, with two hard stops: E2 GO and the D1 early read. Defer the full-length seed-1 replicate (+214 h, D1b) until seed 0 passes. Cheaper alternative: 1.0-epoch full run, ≈ 228 Thor-h total. |
| **D2** | box-head size cap | **Retire `PARAM_BAND` 2-4 M for box3d in refcv9** (it is a "probe the latent" rule, not a detection rule, D6) and replace it by **≤ 12 M for box3d + a Thor latency measurement in the gate**; keep the band on the agent slots while they exist. Without it B4 (6 layers, per-frame queries) is refused at construction. |
| **D3** | temporal BEV fusion: may past ego-motion (egomotion poses at t ≤ t0, IMU-class, causal) warp past BEV features / map decisions into the current frame at inference? | **YES, ruled now, built as E1 arm T1, joining the full run only if T1 clears.** The PI's 10-27 07:25 intent names "the history of ego data" as an input; v0 / a0 / yaw rate at t0 are already admitted; the K = 3 image stack is already temporal. Conditions: the warp uses past poses only, never future; W4 and W5 bind; a BEV-history-shuffled control must read worse. If NO: 0-5 m is "no evidence by construction" and MAP-2's 0-5 m clause is read on lift-valid cells. |
| **D4** | LiDAR depth supervision at train time (labels only; inference vision-only) | **YES in principle** (consistent with the 2026-09-11 LiDAR-supervised-BEV directive and the label ruling). refcv9 starts on box-derived depth (free); approve the LiDAR projected-depth build (sweeps on Thor / HF; size UNVERIFIED) — it joins as B1b only if built and read before E2. It also provides a per-frame ground-plane pitch label for R10. |
| D5 | a `refcv9-full` launch-gate profile (a gate edit beyond the approved liveness rules) | YES (the gate is binding; there is no profile for a perception-trained refcv9). |
| D6 | eval protocol: eval139 DEV / HOLD halves (E0i) — selection on DEV, bars on HOLD | YES; it removes the 40k-vs-48,852 test-set-selection trap at the cost of ~√2 wider CIs on the perception bars. |
| (MM, disclosed) | retire `--w-map-fp` (the PI approved it for the warm start); K3 agent-head retirement | retire; K3 decided by E2 |

---

## 7. Conflicts, limits, UNVERIFIED

* **`LIT_DETECTION` reads refcv7-era facts:** "100-query", "1.97 boxes per object", "AP@2 m 0.248 / 0.21", and it could not
  locate the 09-21 class-collapse artifact. AS-IS (source): 300 queries; PERC8 (refcv8): 2.14 boxes/object; Q3 settles the
  collapse: NOT collapsed in refcv8. Its lever ranking stands; its numbers about OUR head are superseded here.
* **Class-weight family:** the brief and a memory note say median-frequency; AS-IS read the artifact: `sqrt_mf` (MEASURED,
  `map_hires_class_weights_train_100x30.json`). AS-IS wins.
* **`LIT_BEVMAP` primaries are not banked** (38 PDFs on F:). Banking (`tools/kb_add.py`, writes G:) is an integration ask.
* **PUB-SECONDARY, not banked:** DETR (arXiv 2005.12872) — NMS helps the early decoder layers and not the last; WSD schedules
  (arXiv 2405.18392, 2404.06395). Both need `kb_add.py` before any registry or paper use.
* **Every step-cost and Thor-h number is ESTIMATED**; the E1 and E3 smokes measure them. The E1 rig's 3.0 s/step assumes the
  trunk dominates the perception cost (from 8.94 − 4.34 = 4.60 s MEASURED).
* **The gradient-share target 0.05 is a judgement**, tested at two doses; the Gram-matrix method assumes the 64-window
  gradients are representative and passes only with the linearity control.
* **Bars MAP-2 (absolute), BOX-1/2 thresholds are the MM's proposal, not forecasts**; MAP-2's 5-20 m value is replaced by
  E0h's planner-derived number before sealing.
* **E1 arms are one-variable screens at 4,000 steps**; interactions are read only in E2 K2. A lever whose effect needs a long
  schedule can be screened out falsely; B-arms that fail only on convergence speed are re-read in E2 rather than dropped.
* **Not read here:** the refcv8 in-run gradient-share rows themselves (Thor; numbers taken from the brief as MEASURED);
  W4's final numbers (brief: PASS).
* **Q1 / Q2 were not landed at writing** (§3 carries both branches).

## 8. Integration asks and deliverable manifest

**Asks (escalated, not "please merge"):** (1) the PI decisions D1-D6; (2) bank `LIT_BEVMAP`'s 38 PDFs + the three
PUB-SECONDARY items; (3) after Q1 lands, the MM marks the live branch in §3 (mechanical); (4) code items C1-C12 to a code agent
with this file as the spec; (5) E0 items to the dev box now (0 Thor-h, no decision needed).

| artifact | where | copies |
|---|---|---|
| this file | hold `C:/Users/Admin/qland/hold_refcv9_design/code/TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/REFCV9_DESIGN.md` → repo path of the same name (NEW; not staged, not committed) | ONE (the hold) until landed |
