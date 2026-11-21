# MAP_IN_PLANNER_DESIGN — how reference end-to-end planners consume map information, and the refcv9 wiring that follows

* Written 2026-10-25 by the Research Lab literature agent for the Master Mind. PI directive (2026-10-25): *the planner should use the map head; the trajectory must be in the drivable area, should avoid red (non-drivable-edge) cells, and should preferably stay within the lane.*
* **Evidence classes.** `PUBLISHED` = read from the banked PDF (section / table / equation given). `PUBLISHED-CODE` = read from `hustvl/DiffusionDrive` main HEAD `9b52ed0` (copies + sha256 in `F:/lit_map_planner/`). `MEASURED` = our artifact, path named. `INHERITED` = from the brief, not re-verified. `HYPOTHESIS` = my inference. No aggregator was quoted: every PDF number below was read from the banked PDF text or the rendered table image.
* **Banking.** Every cited paper was already in `TanitAD Research Lab/Library/` (16 PDFs, sha256 compared against `library.json`: 16/16 equal). Nothing new was downloaded. `F:/lit_map_planner/MANIFEST.json` lists them with sha256 and a one-line finding, plus four DiffusionDrive-v1 code files. `tools/kb_add.py` was not run; nothing was written to G:.
* **Mechanism legend** — (i) map as auxiliary training loss only · (ii) map features read by the planner · (iii) map-based constraint loss on the planned trajectory · (iv) map-based candidate scoring or selection.

---

## 0. Bottom line

1. **No reference planner gets its drivable-area compliance (DAC) from a map auxiliary loss alone.** Where (i) is ablated in isolation it is worth −0.5 to +2.3 PDMS and sits inside or near seed noise (§2.4). The measurable DAC gains come from **(iv) per-candidate sub-score heads whose labels come from the GT map** (Hydra-MDP++: DAC 92.0 → 96.0 from selection-time weighting alone, same trained model) and from **(ii) waypoint-anchored sampling of the BEV feature map** (DiffusionDrive v1).
2. **Not one paper applies a hard filter built from its own *predicted* map at inference.** The only hard-filter planner (PDM-Closed, DAC 98.8–99.8) reads the privileged GT map. Learned DAC heads (Hydra-MDP, GoalFlow, WoTE, DriveSuprim, GTRS, DDv2) take map-derived labels at training and predict the sub-score at inference. That is the shape `DrivableCritic` already has.
3. **Our measured situation favours selection first.** refcv8's 149-candidate fan has an oracle at 0.084 m ADE@2 s against its own pick at 0.299 m (`C:/Users/Admin/refcv8_pf/fanceil/RESULT.md`, T0): *the fan is good and the selector is weak.* On NAVSIM navtest (zero-shot, non-parity) refcv8 R8_LEGAL has **DAC 76.84** against refcv7 87.37, STOP 96.53, HUMAN 100 (§4.1). A map-aware selector attacks exactly that gap.
4. **P4 does not say "the map is worthless" — it says "that channel is inert".** Verdict `INCONCLUSIVE-INERT` (a wrong map injected the same way does not hurt). refcv8 reads the BEV only through a 30×16 average-pooled token grid; the DiffusionDrive planner instead point-samples a 0.25 m per-pixel feature map **at its own waypoint coordinates**, from the same features its BEV-semantic head decodes (§1, §5.3). (ii) is therefore worth a *gated* second stage, not a first one.

---

## 1. Per-paper table

`Y` = used, `–` = not used, `?` = not determinable from the paper. "Effect" is the number the paper itself reports for the map-related component.

| # | Paper (arXiv, library key) | Map representation | (i) aux | (ii) read by planner | (iii) constraint loss | (iv) scoring / selection | Reported effect of the map component (table) |
|---|---|---|---|---|---|---|---|
| 1 | **DiffusionDrive v1** (2411.15139) NAVSIM | 7-class raster BEV semantic: road (lane+intersection polygon), walkways, centerline (linestring), static objects, vehicles, pedestrians; 0.25 m px (`PUBLISHED-CODE` config) | Y — `bev_semantic_weight` 14.0 CE beside trajectory 12.0 / diffusion 20.0 (code) | **Y** — trajectory-coordinate spatial cross-attn into BEV (§3.4); no map-query cross-attn on NAVSIM because Transfuser has no vectorized map (§4.2). nuScenes variant takes SparseDrive map queries as input (App. A) | – | learned anchor-confidence head only; no map-derived score | **Table 3**: no spatial cross-attn + no agent/map CA (ID2) PDMS 55.1, DAC 83.2 → + spatial cross-attn (ID3) 87.1, DAC 95.4 → + agent CA (ID5) 87.4 → + cascade (ID6) 88.1, DAC 96.2. *Not map-isolated*: ID2 removes all scene interaction. |
| 2 | **DiffusionDrive v2** (2512.07745) | same BEV stack on NAVSIM; text also names map queries | inherited | **Y** in the selector: trajectory coords → deformable spatial cross-attn to BEV → cross-attn to agent **and map** queries → MLP score (§4.6) | – (RL reward is trajectory-level, §4.4–4.5; its definition is not stated in the paper text I read) | **Y** — two-stage coarse→fine scorer, BCE + margin-rank loss (Eq. 11) on GT sub-scores | **Table 9**: DD v1 with the v2 selector DAC 96.2 → 97.0, PDMS 88.1 → 89.1 (NC 98.2 → 97.2). Table 8: coarse-to-fine +0.2, rank loss +0.2 PDMS (89.7 → 90.1). |
| 3 | **TransFuser** (2205.15997, PAMI) | 3-channel 64×64 BEV mask: road, lane marking, other (§3.6 "Loss Functions" → Auxiliary Tasks → "HD Map") | **Y only** | – (BEV feature fusion; waypoint GRU reads pooled 512-d vector) | – | – | **Table 7** (CARLA Longest6, DS): none 44.29; no HD map 50.96; all losses 56.68 best seed / 49.49 worst seed. The worst-seed run is below the no-map row → inside seed noise (Table 6). |
| 4 | **NAVSIM / LTF** (2406.15349) | TransFuser's BEV semantic aux | **Y only** | – | – | DAC = ×0 hard penalty in the *metric* (§4.1) | **Table 2** E1 "No BEV segmentation": DAC 90.5, PDMS 81.6 vs default seeds A1–A3 83.3/84.0/84.4 (DAC 91.3/92.8/93.0; PDMS std 0.56 per text). One run, −2.3 vs seed mean. |
| 5 | **VAD** (2303.12077) nuScenes | **vectorized** polylines: lane divider, road boundary, ped crossing; 100 map queries × 20 pts (§3.1, §4.1) | Y (map loss ω1) | **Y** — ego query cross-attends map queries (§3.2, Eq. 2) | **Y** — ego-boundary overstepping hinge (Eq. 5, δ_bd = 1.0 m) + ego-lane directional angle loss (Eq. 6, δ_dir = 2.0 m); map confidence ε_m = 0.5; training only (Fig. 1 / §3.3). ω weights not published | – | **Table 2**: all on (ID7) L2 0.72 / CR 0.22; map-interaction off (ID1) 0.86 / 0.29; constraints off (ID3) 0.76 / 0.28; + overstep only (ID4) 0.80 / 0.24; + dir only (ID5) 0.75 / 0.25. **Table 3**: raster map, no constraints 0.74 / 0.39; vector, no constraints 0.77 / 0.26; vector + both 0.72 / 0.22. **No boundary-violation metric is reported.** |
| 6 | **VADv2** (2402.13243) | vectorized map tokens (MapTRv2 loss) | Y | **Y** — planning tokens interact with map tokens | **Y** — Conflict Loss (Eq. 6): vocabulary actions that conflict with agent futures **or road boundaries** are negatives | learned vocabulary distribution | **Table 7** (CARLA open-loop): no map token L2 0.086/0.191/0.332, col 0.005/0.034/0.070 vs full 0.082/0.169/0.290, 0.000/0.010/0.039; no conflict loss 0.086/0.173/0.291. Conflict loss covers agents + boundaries, not separable. |
| 7 | **SparseDrive** (2405.19620) | vectorized map instances (anchor polylines, §3.2) | Y (L_map) | **Y** — agent-map cross-attention in the parallel planner (§3.3) | – | collision-aware rescore is **agent-only** (§3.3) | agent-map cross-attention is **not ablated** (Table 4 covers PAL/EII/MTM/ATA/CAR). |
| 8 | **Hydra-MDP** (2406.06978) | Transfuser aux (3D det + BEV seg) for the student; **GT HD map only inside the offline simulator** that produces labels | Y (inherited) | – (no map input at inference) | – | **Y** — per-candidate sub-score heads (NC, **DAC**, TTC, C, EP), BCE distillation (Eq. 10), weighted log-sum selection (Eq. 11): 0.01 ≤ w_imi ≤ 0.1, 0.1 ≤ w_NC, w_DAC ≤ 1, 1 ≤ w_TTC/C/EP ≤ 10 (§2.4.1) | **Table 1** (navtest): V8192 DAC 91.7 / 83.0 → **-W** (weighted confidence at inference) DAC **96.1** / 85.7. Distilling the aggregate PDM score instead (-PDM) falls to DAC 88.9 / 80.2. |
| 9 | **Hydra-MDP++** (2503.12820) | same; adds lane-keeping (LK) and driving-direction (DDC) teachers from the lane graph | Y ("P") | – | – | **Y** (Eq. 7–9) | **Table 3** (R34): W off → on DAC **92.0 → 96.0**, PDMS 85.0 → 86.5. Aux perception **P**: DAC 96.0 → 95.6, PDMS 86.6 → 86.1 ("worsens"). **Table 5**: LK teacher raises LK 65.5 → 67.2 (R34) and 67.6 → 70.1 (V2-99). |
| 10 | **GoalFlow** (2503.05689) | HD-map-derived BEV aux supervision (§3.2.2); drivable **polygon** D for the DAC label | Y | goal scorer attends BEV (`F_bev`) | – | **Y** — goal-point DAC score: label 1 iff the 4 corners of a "shadow vehicle" at the goal pose lie in D (§3.2.3, Eq. 16 BCE, weight w5 = 0.005 vs distance w4 = 1.0); final log-sum with w1, w2 | **Table 2**: M0 DAC 94.2 → +distance goal (M1) 96.4 → +DAC score map (M2) 97.5, PDMS 89.4 → +trajectory scorer (M3) DAC 98.3, PDMS 90.3. |
| 11 | **WoTE** (2504.01941) | same 7-class BEV semantic as DD (road, walkways, centerline, static, vehicles, pedestrians) decoded for each predicted future BEV state (§3.4, Eq. 6, focal) | **Y** (future states) | reward model reads current + imagined BEV states | – | **Y** — reward = −w1 log r_imi + w2 log r_NC + w3 log r_DAC + w4 log(TTC/C/EP) (Eq. 4); r_sim by BCE vs simulator | **Table 4**: imitation-only DAC 92.5 / PDMS 83.5; sim-reward-only DAC 94.7 / 83.6; both 94.7 / 85.6. **Table 3**: no trajectory eval DAC 91.5 / 81.0 → eval on current state 92.9 / 83.2 → eval on imagined states 94.7 / 85.6. |
| 12 | **DriveSuprim** (2506.06659) | sim labels (map inside simulator) | – | – | – | **Y** — coarse→fine scorer, BCE on GT sub-scores (Eq. 8) | **Table 1** oracle over predicted ranking: top-1 DAC 98.7 / PDMS 91.9; best-of-top-16 DAC 99.8 / 96.1; top-256 100 / 98.7. |
| 13 | **GTRS** (2506.06664) | Transfuser BEV segmentation aux head on the diffusion generator (§2.1) | **Y only** | scorer attends image tokens only; no map | – | **Y** — scorer trained on sim sub-scores | map component not ablated. |
| 14 | **BEV-Planner** (2312.03031) — evaluation paper | raster boundary at **0.1 m** for the Curb Collision Rate (CCR) | – | – | – | – | **Table 3**: adding a map-perception aux task: L2 0.55 → 0.96, collision 0.59 → 0.89, **CCR 4.26 → 2.60**. **Table 1**: CCR VAD-Base 2.47–2.72, UniAD 1.59–1.93, Ego-MLP (no perception) 2.93. |
| 15 | **PDM-Closed** (2306.07962) + NAVSIM v2 (2506.04218) | centerline from lane graph, GT drivable polygon | – | GT map | – | **Y (rule)** — IDM proposals at 5 target speeds × 3 centerline offsets (−1, 0, +1 m), simulated and scored with a nuPlan-style scorer, argmax | CLS 92–93 vs learned 50–72 (Table 1); DAC 98.8 (pseudo-sim S1) / 99.8 (Hydra-MDP Table 1). EPDMS: NC, DAC, DDC, TLC are ×penalties; **LK is a weighted-average term (weight 2)**; a violation the human also commits is ignored (`filter_m`). |

---

## 2. Mechanism notes per paper (what actually enters the planner)

### 2.1 DiffusionDrive v1 — the reference the PI named
* **Where the map is (PUBLISHED §3.4, §4.2, App. A).** Each decoder layer first applies a deformable-style spatial cross-attention to BEV features **based on the trajectory coordinates**, then cross-attention to agent/map queries, then FFN, timestep modulation and an MLP that outputs a confidence and an offset. On NAVSIM only agent queries exist (§4.2: the Transfuser perception module has no vectorized map). The map therefore reaches the planner **only** as BEV pixels trained by the 7-class BEV-semantic head (App. A). On nuScenes the decoder consumes SparseDrive's map queries (App. A).
* **What the code does (PUBLISHED-CODE, `modules/blocks.py:42-110`, `transfuser_model_v2.py:105-127,276-290`).** `GridSampleCrossBEVAttention`: the 20 query features each produce a **softmax over the 8 waypoints** (zero-initialised linear → uniform at start); the 8 waypoint coordinates are normalised to the lidar window (±32 m) and used as a bilinear `grid_sample` into a 1×1-conv-projected per-pixel BEV feature map (zero padding); the weighted sum is added residually. **No learned sampling offsets.** The sampled map is `bev_proj(cat[status-fused tokens upsampled, bev_feature_upscale])`, and `bev_feature_upscale` is **the same tensor the `_bev_semantic_head` (3×3 conv → 1×1 conv → 7 classes) decodes**. So the semantic loss shapes exactly the features the planner point-samples at its own waypoints.
* **Loss weights (code, `transfuser_config.py`, `transfuser_loss.py`).** trajectory 12.0, diffusion 20.0, agent-class 10.0, agent-box 1.0, **bev-semantic 14.0**. No boundary / DAC / lane constraint loss. These coefficients multiply loss means of different scale and are not comparable across terms.
* **Effect (Table 3).** Spatial cross-attn is the single largest DAC mover (83.2 → 95.4). That is "access to scene features at the trajectory", not a map-only effect.

### 2.2 DiffusionDrive v2 — selection, not map input, is the new part
Selector (§4.6) = DD-v1-style waypoint-anchored BEV sampling, then agent + map cross-attention, then an MLP score; coarse (top-k) → fine scorer; BCE on the sub-score labels, margin-rank loss (Eq. 11) on continuous ones; training augments with multiplicative exploration noise (σ 0.1–0.2) and 1 % GTRS-vocabulary trajectories (§7). Table 9: the same selector bolted on DD v1 gives DAC 96.2 → 97.0 and EP 82.2 → 86.8 but NC 98.2 → 97.2. The RL stage (Tables 5–6) is trajectory-level and DAC enters only through whatever reward R is (not defined in the paper text I read; the lab's code-read note — INHERITED — says PDM score over a 4 s LQR rollout).

### 2.3 VAD — the only published **constraint losses** on the map
Eq. 5: for each future waypoint t, L_bd,t = δ_bd − d_t when d_t < δ_bd (distance to the nearest **predicted** boundary polyline with confidence ≥ ε_m = 0.5), else 0, averaged over the horizon; δ_bd = 1.0 m. Eq. 6: mean angular difference between the ego motion vector and the nearest predicted lane-divider direction within δ_dir = 2.0 m. Both act at training only (Fig. 1). The paper's evidence for them is L2 and agent-collision rate (Tables 2–3) — **there is no boundary-crossing or off-road metric in VAD**. The only independent boundary read (BEV-Planner Table 1, CCR) shows VAD-Base at 2.47–2.72, not better than UniAD (1.59–1.93) and no better than an ego-status-only MLP (2.93); that comparison crosses implementations and is confounded (HYPOTHESIS: the hinge is not visibly buying boundary clearance on nuScenes).

### 2.4 Map as auxiliary loss only — the pooled evidence
| source | arm | metric | effect | read |
|---|---|---|---|---|
| NAVSIM T2 E1 | no BEV segmentation | PDMS | 81.6 vs 83.3–84.4 (3 seeds) | −2.3 vs seed mean, one run, 4× the seed std |
| Hydra-MDP++ T3 | + aux perception "P" | PDMS | 86.6 → 86.1 (DAC 96.0 → 95.6) | the authors say it worsens planning |
| TransFuser T7 | no HD map | DS | 50.96 vs 56.68 / 49.49 (best / worst seed) | inside seed spread |
| BEV-Planner T3 | + map aux | CCR / L2 | CCR 4.26 → 2.60 but L2 0.55 → 0.96 | helps boundary, hurts imitation |

**Reading (HYPOTHESIS):** an aux map loss is a *representation* lever with a small, inconsistent sign on planning. It is not a substitute for a mechanism that makes the planner act on the map. refcv8 already has (i) and measures DAC 76.84, below refcv7's 87.37 (§4.1; not attributable — many arms differ).

### 2.5 Selection-time heads with map-derived labels — the strongest and most reproducible effect
| paper | arm | DAC before → after | note |
|---|---|---|---|
| Hydra-MDP++ T3 | weighted confidence at inference, **same model** | 92.0 → 96.0 | selection-time only |
| Hydra-MDP T1 | V8192 → V8192-W | 91.7 → 96.1 | |
| GoalFlow T2 | + goal-level DAC score | 96.4 → 97.5 | label = vehicle corners ⊂ drivable polygon |
| WoTE T4 | imitation-only → sim-reward-only | 92.5 → 94.7 | |
| WoTE T3 | eval on current vs imagined BEV state | 92.9 → 94.7 | predicted future map state helps |
| DDv2 T9 | DD v1 + learned selector | 96.2 → 97.0 | NC −1.0, EP +4.6 |
| DriveSuprim T1 | oracle top-16 over predicted ranking | DAC 99.8 | headroom for a good ranker |

The labels come from a simulator that reads the **GT map**; the student has no map at inference. Labels are per candidate and per sub-score, never the aggregated PDMS (Hydra-MDP "-PDM" row regresses to 80.2).

---

## 3. The PI's three tiers mapped onto mechanisms

| PI tier | mechanism | published precedent | form in refcv9 |
|---|---|---|---|
| **MUST stay in the drivable area** | **(iv)** learned DAC head + calibrated rejection; **(iii-select)** DAC-violating candidates become negatives in the anchor classifier | Hydra-MDP Eq. 10–11; Hydra-MDP++ T3 (+4.0 DAC); GoalFlow Eq. 16; WoTE Eq. 4; VADv2 Conflict Loss (boundary-conflicting actions are negatives); NAVSIM §4.1 (DAC = ×0) | DAC critic output + `reject if p_dac < τ unless all rejected`; τ calibrated so that ≥ 99 % of **human** GT trajectories pass (the HUMAN row is DAC 100); mask labels where the human also violates (EPDMS `filter_m`) |
| **AVOID edges** | **(iii)** boundary-distance hinge on the planned trajectory; **(iv)** clearance output beside the DAC head | VAD Eq. 5 (δ_bd 1.0 m, mean over horizon); BEV-Planner CCR as the measuring stick (0.1 m raster) | hinge on a **GT signed-distance field** at train time; clearance term in selection from the critic; **do not** use the predicted `edge` class — its measured 10 cm IoU is 0.001 (§4.2) |
| **PREFER the lane** | soft (iv) term, small weight; lane-direction (iii) only if lane direction becomes available | Hydra-MDP++ LK/DDC teachers (Table 5: LK +1.7 / +2.5, DDC +1.4 / +1.3); VAD Eq. 6 (needs vectorized lane direction); PDM-Closed centerline offsets {−1, 0, +1} m; EPDMS LK weight 2 | LK-style critic output from the GT lane raster, weight swept from 0; no direction loss on a raster `lane` class (paint pixels carry no direction; refcv8 lane IoU 0.199 at 0–20 m, 0.060 at 20–40 m) |

Why the order matters: DAC and edge clearance are **multiplicative / hard** in the target metric; lane preference is a weighted term at best (EPDMS LK, weight 2) and even PDM-Closed scores only ~66.7 on Hydra-MDP++'s LK definition (Table 2). Treat lane as a tie-breaker, never as a gate.

---

## 4. Our measured starting point (all `MEASURED`, artifacts named)

### 4.1 refcv8 on NAVSIM navtest (zero-shot PhysicalAI → nuPlan; non-parity; T1-family open loop; never closed loop; not a leaderboard submission)
`C:/Users/Admin/refcv8_pf/bars8fix/bars_rerun_navtest_only/NAVSIM_BARS_step48852_ema.md`, n = 12,146 tokens / 136 logs, log-cluster bootstrap (answers "another draw of logs" only):

| arm | NC | DAC | TTC | EP | DDC | PDMS | zeroed NC/DAC |
|---|---|---|---|---|---|---|---|
| R8_LEGAL | 85.31 | **76.84** | 77.15 | 52.07 | 91.36 | 57.84 | 4,027 |
| REFCV7_50400_LEGAL | 89.81 | 87.37 | 79.13 | 68.27 | 94.71 | 70.20 | 2,624 |
| STOP | 97.40 | 96.53 | 96.40 | 31.00 | 99.59 | 61.82 | 732 |
| HUMAN | 100 | 100 | 100 | 86.96 | 98.83 | 94.55 | 0 |

⚠️ **STOP scores DAC 96.53.** A DAC-optimising critic can win by preferring short or slow candidates. Every critic arm must report EP, stop-rate and the selected-speed distribution beside DAC, and be compared with STOP, not only with R8_LEGAL (same family as the existing "NavSim warmup rewards stopping" finding).
⚠️ The 20-point DAC gap to DD v1 (96.2) is **confounded by the zero-shot domain shift** — DD v1 trains on navtrain, refcv8 never saw nuPlan cameras. A critic that reads the *predicted* map inherits the same shift.

### 4.2 refcv8's 10 cm map head (EMA-48,852; 4,686 windows / 137 episodes; `C:/Users/Admin/refcv8_pf/perc8/RESULT.md` §3)
| class | IoU 0–20 m | 20–40 m | 40–60 m |
|---|---|---|---|
| drivable | 0.605 | 0.613 | 0.548 |
| lane / road line | 0.199 | 0.060 | 0.026 |
| crosswalk | 0.149 | 0.001 | 0.000 |
| arrow / text | 0.088 | 0.000 | 0.000 |
| non-drivable edge ("red") | **0.001** | 0.000 | 0.000 |
| hatched | 0.000 | 0.000 | 0.000 |
| sidewalk / verge | 0.459 | 0.529 | 0.456 |

⇒ **A hard filter on the predicted drivable mask is not admissible at IoU ≈ 0.6, and an edge-distance cost on the predicted `edge` class would read an empty channel.** Both facts push the training labels to the GT raster and make the predicted map an *input feature* to a learned critic, not a rule.

### 4.3 The fan and the channel
* **Fan** (`fanceil/RESULT.md`, T0 diagnostic, PhysicalAI eval139 = 4,754 windows / 139 episodes): 149 candidates (117 base + 32 extra); oracle-in-fan 0.0841 m [0.0755, 0.0960] vs own pick 0.2990 m [0.2747, 0.3266] at 2 s; own pick ranks median 13 of 149; the pick is the oracle in 8.9 % of windows.
* **Channel** (`gtmap_oracle/PREREG_P4.md` §2, `run/RESULT.md`): the planner reads `bev_tokens` = 30×16 average-pooled (zero params) from a 0.5 m BEV and `bev_feats` for decoder coupling; the only consumer of the map logits is `r8_drv`, **not built** in refcv8-r101-s0 (`refc_v3.py:2620-2630`). P4 verdict `INCONCLUSIVE-INERT`: TRUE − OWNMAP S6 ADE +0.0096 [−0.0213, +0.0418] (s0), +0.0079 [−0.0247, +0.0438] (s1), inference-seed floor 0.0170 m; the SHUF liveness arm (wrong map) is not separated (+0.0175 [−0.0196, +0.0564]). The planner **does** read the BEV (BEV_ZERO − BASE S6 ADE +1.15 m, per the prereg). So: the BEV matters, the map content does not travel through it.

### 4.4 What I could not verify
`stack/tanitad/refs/refcv8_conditioning.py` and commit `f49c46b` are **not present** in the checkouts I can reach (the G: tree on `agent/arch-inf-20260803` has no `refcv8_conditioning.py`; `git rev-parse f49c46b` fails). `DrivableCritic` is therefore described only as the brief gave it (per-candidate MLP on map-sampled features, zero-init selection weight, unbuilt) plus the `r8_drv` references in the P4 prereg. **Before wiring, check its actual input list against §5.2.**

---

## 5. refcv9 recommendation

### 5.1 Principle
Combine **train-time (iii)/(iii-select) with selection-time (iv)**, built on **GT-raster labels** (the SAM3 10 cm GT is already the training target of the map head), and treat the **predicted map as an input feature**, not as a rule. Put (ii) in as a **gated** second stage.

### 5.2 Components
**C1 — GT labels, no simulator.** For every candidate (all 149 plus the refined output) compute, on the SAM3 GT 10 cm raster:
* `dac_gt ∈ {1, 0, ignore}` = ego footprint (L × W from `vehicle_dimensions`; GoalFlow uses the 4 corners, add edge midpoints and sweep between waypoints so a 2 Hz gap cannot skip a curb) entirely inside the drivable **union**; `ignore` where any footprint cell is unlabelled (`nocls`, CE ignore index 255) **or** the human GT path itself violates (EPDMS `filter_m`);
* `clr_gt` = min signed distance from footprint to the nearest non-drivable cell, from a precomputed distance field of the GT mask;
* `lk_gt` (optional, small weight) = lateral offset to the nearest lane-line pixel vs the human's own offset.
Drivable union = {drivable, lane, crosswalk, arrow}. **Open PI/owner item:** `hatched` (painted neutral zone) default **excluded**; `sidewalk`, `edge`, `nocls` excluded; opposing-lane driving is *not* caught (no lane direction in the raster; DDC is a separate EPDMS penalty — out of scope for this wiring).

**C2 — `DrivableCritic` (iv).** Per candidate, input = (a) the 8-class predicted 10 cm map softmax sampled over the footprint points, (b) the 64-ch `map_hires_bev` feature bilinear-sampled at the same points (DD v1's mechanism, §2.1), (c) a coarse distance transform of the predicted drivable probability at each waypoint. Outputs `ŝ_dac` (BCE vs `dac_gt`, masked) and `ŝ_clr` (BCE or margin-rank vs thresholded `clr_gt`; DDv2 uses margin-rank for continuous scores). Selection: `score_i = log s_sel,i + λ_dac·log ŝ_dac,i + λ_clr·log ŝ_clr,i (+ λ_lk·log ŝ_lk,i)`, **λ = 0 at init (the existing zero-init weight)** and swept at inference — Hydra-MDP++ shows that sweep alone moved DAC 92.0 → 96.0. Grid from the published ranges (Hydra-MDP §2.4.1: imitation 0.01–0.1, DAC 0.1–1): λ_dac ∈ {0, 0.1, 0.3, 1, 3}. Calibrated rejection: drop candidates with `ŝ_dac < τ` unless none survive; τ set so ≥ 99 % of human GT trajectories pass on a held-out split.

**C3 — train-time constraint (iii).** On the refined plan: `L_bd = mean_t max(0, δ − d_t)` with `d_t` read by bilinear lookup in the **GT** distance field (VAD Eq. 5; δ_bd = 1.0 m was VAD's setting — start from half the ego width + clearance and sweep {0.5, 1.0, 1.5} m; weight sweep ×{0.05, 0.5, 5} of the trajectory L1, because published coefficients span 3 orders (GoalFlow DAC BCE 0.005 vs 1.0; DD v1 semantic CE 14.0 vs 12.0)). **C3b — conflict negatives (VADv2 Eq. 6 analogue):** anchors whose refined plan has `dac_gt = 0` are added as negatives to the anchor-classification BCE — zero extra labels, because C1 already computes them.

**C4 — keep (i)**, do not tune it as a lever (§2.4).

### 5.3 Is (ii) worth it, given P4?
* **P4 does not transfer** to waypoint-sampled access: it measured the pooled 30×16 path, and its own verdict is that the channel is inert. DD v1 samples a 0.25 m per-pixel map at the waypoints themselves, from the head-decoded features.
* **But (ii) is the only mechanism that lets the *generator* propose on-road candidates; (iv) can only *choose*.** The fan oracle (§4.3) says choosing has headroom first.
* **Decision:** build C1–C3 first (stage S1). Add **S2 = DD-style `GridSampleCrossBEV` on `map_hires_bev` plus footprint-sampled map logits in each cascade layer**, only if E0/E1 show the fan lacks DAC-pass candidates or the critic saturates below its GT-feature oracle. S2 must **pass the P4 liveness gate again**: TRUE-map − OWN-map above the inference-seed floor **and** SHUF must hurt, or the wiring still carries no map content.

### 5.4 Experiment ladder (cheapest, largest expected effect first; every arm pre-registered with both outcomes)
| id | what | cost | pass / fail, committed in advance |
|---|---|---|---|
| **E0** | **GT-DAC oracle on the banked refcv8 fans** (`fanceil` dumps): share of windows with ≥ 1 DAC-pass candidate; DAC of own pick vs oracle-by-DAC-then-imitation vs random pick; STOP-aware (report EP) | 0 training | share high **and** pick-vs-oracle gap large ⇒ E1 is worth running; share low ⇒ skip to S2. Thresholds sealed by the owner before reading. |
| **E1** | **Critic-only on frozen refcv8** (EMA-48,852): MLP on cached footprint-sampled features, λ swept at inference | tiny MLP, cached features | DAC of the selected candidate on eval139 and navtest, paired cluster bootstrap, separated **and** above the inference-seed floor (S6 0.017 m; PDMS ×100 floor 0.0772) **and** above a replicate-critic seed floor (H-ESTIM-SEED-1); EP / stop-rate not degraded vs STOP-controlled read. |
| controls in E1 | constant-only critic (reads the prior exactly); **oracle critic fed the GT map** (upper bound of map content); **SHUF-map critic (must hurt)**; raw-pixel/`map_hires_bev`-free floor | | an arm that does not beat the SHUF control has not used the map. |
| **E2** | refcv9 retrain with C3 + C3b + critic jointly vs a **replicate arm** (same flags, new seed) | one training run + replicate | the lever effect is read against the replicate's run-to-run spread, not against the CI alone. |
| **E3** | S2 decoder waypoint sampling + P4 liveness re-run | training | §5.3 gate. |

Report every arm on the four metric families (longitudinal, lateral incl. curvature and cross-track, tactical, strategic) **plus** NC / DAC / TTC / EP / C / DDC, tier-stamped; ADE alone is inadmissible.

### 5.5 Risks, stated up front
1. **Stop-gaming** (§4.1): DAC 96.53 for STOP.
2. **Domain shift:** the critic reads a predicted map on cameras the model never trained on; E1 must be read on both PhysicalAI eval139 (in-domain) and navtest (zero-shot) separately.
3. **Label-model circularity (the `ls-tree` family):** `dac_gt` and the critic's map features both come from SAM3-derived rasters; E1's oracle and SHUF controls exist to catch a critic that is merely echoing its own input.
4. **Aggregate over-blocking:** VAD never measured a boundary metric and BEV-Planner's CCR shows no VAD advantage; do not expect a constraint loss to move DAC by itself — C3 is a **secondary** lever behind C2.
5. **Single training seed** (one refcv8 run): no claim about a lever without a replicate arm.

### 5.6 Blocked / owner decisions
* `hatched` and opposite-lane handling in the drivable union (PI/owner).
* The `edge` class (IoU 0.001) needs its own fix before any predicted-edge cost is possible; until then edges enter only via GT labels (C1/C3) and the drivable-probability distance transform (C2c).
* `DrivableCritic`'s real input list (§4.4) — unverified by me.

---

## 6. Provenance
* Papers: `F:/lit_map_planner/MANIFEST.json` (sha256 of each banked PDF, compared against `library.json`: 16/16 equal).
* DiffusionDrive v1 code at `9b52ed0` (read as data, not executed): `F:/lit_map_planner/code_DiffusionDrive_9b52ed0/{modules_blocks,transfuser_config,transfuser_loss,transfuser_model_v2}.py`.
* Programme artifacts read (read-only): `refcv8_pf/{gtmap_oracle/PREREG_P4.md, gtmap_oracle/run/RESULT.md, perc8/RESULT.md, fanceil/RESULT.md, bars8fix/bars_rerun_navtest_only/NAVSIM_BARS_step48852_ema.md}`.
* Tables read as rendered images where the text layer lost check-marks: DD v1 Table 3, NAVSIM Table 2, VAD Tables 2–3, VADv2 Table 7, Hydra-MDP Table 1, WoTE Tables 3–4, BEV-Planner Table 1.
