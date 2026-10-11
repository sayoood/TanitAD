# DATA_SELECTION_REVIEW: training-data selection and corpus extension for REFe

**Author:** Research Lab agent, 2026-10-11. **Scope:** one deliverable, four questions. The RL question is a separate file owned by another agent (`RL_POSTTRAIN_REVIEW.md`); nothing here writes into it.
**Tier:** every PDMS below is NAVSIM v1.1 PDMS on navtest, a non-reactive pseudo-simulation of a 4 s open-loop plan. EPDMS rows (NAVSIM v2) are labelled EPDMS and are never compared with PDMS.
**Evidence classes:** MEASURED (ours, artifact named) | PUBLISHED-PRIMARY [library key, banked PDF] | PUBLISHED-SECONDARY (read through a summariser, not banked) | INHERITED (another doc, not re-verified) | ESTIMATED | HYPOTHESIS | UNVERIFIED.
**REFe state:** navtest PDMS 88.25 (SFT-4 arm B, u400, single seed) and the 94.55 bar are taken from the brief and from `REVIEW_7_GAP_TO_PAPER.md`. INHERITED: `MODEL_REGISTRY.md` section 14 does not contain the string "88.25" in this tree (grep, 2026-10-11), so the registry row is stale or not yet written. Gap = 6.30 PDMS.

---

## 0. Headline

1. **No published data-selection result is a controlled, same-compute gain for a frozen-ViT-L proposal-and-scorer planner at 100K-sample scale.** Everything published is a smaller model, a smaller or data-poor regime, or a different benchmark. The nearest controlled NAVSIM evidence (MOSAIC, Hydra-MDP, 460-clip base, 3 seeds) is **+1.5 to +1.6 EPDMS over random** for hard-first selection. That is real and primary, but it is an added-data result and cannot be read as a re-weighting gain on the whole of navtrain.
2. **Data is not where REFe's 6.30-point gap lives.** `REVIEW_7` MEASURED the gap as selection and label faithfulness (perfect NC/DAC/TTC heads would give 95.16; relabelling on executed plans is estimated +2 to +4). My ESTIMATE for all three data levers combined is **at most +1.5 PDMS, not additive**. They should run behind P1/P2, not in front of them. They are cheap because P1's relabel produces the hardness signal as a by-product.
3. **Extension beyond navtrain is mostly not new frames.** MEASURED here: navtrain is 1,192 logs / 103,288 tokens, **drive-disjoint** from every nuPlan test DB (0 of 162 drive ids shared with the 132 test drive ids). The OpenScene trainval pool outside navtrain is the filtered-out *trivial* frames of the same logs (about 550K to 630K, ESTIMATED) plus roughly 58 logs. NAVSIM filtered them out because a constant-velocity agent already solves them.
4. **SimScale (237K planner-based sim scenes) is the only extension with a published gain on this architecture family, and the gain is small on navtest PDMS:** DrivoR (the architecture REFe copies) 93.7 -> 94.0 at +65K and -> 94.6 at +134K sim scenes; DriveZero 94.8 -> 95.3 at +237K. Its large effect is on navhard EPDMS (DriveZero +5.6). Data and code are released (HF `OpenDriveLab/SimScale`, 3.6 TB history-only).
5. **PhysicalAI-AV cannot supply PDM targets (no map) and has no like-for-like B0 camera.** It can serve a map-free NC/TTC subset, an auxiliary drivable-area head from SAM3 labels, or representation. No published study adds a third-party corpus to a NAVSIM camera planner; the closest (DriveVLA-W0 Table 7) swings PDMS from -6.5 to +4.9 depending on supervision density. Verdict: not for the PDMS bar.
6. **Ranked levers (section 4):** L1 failure-aware soft re-weighting of navtrain; L2 trajectory-pattern re-balancing (label-free); L3 SimScale-lite (round 0, 65K scenes). Total head-only budget about 120 A40-hours for six arms.

**Integration and decisions needed (not a README request):**
- **Master Mind:** the pre-registrations in section 4 must be copied to `eval/PREREG_DATA_L1.md`, `_L2.md`, `_L3.md` and frozen before P1's relabel output is looked at for hardness. I did not write into `eval/` (another package).
- **PI decision (download, per Keys/permission rules):** L3 needs 609 GB pulled pod-side from public HF `OpenDriveLab/SimScale` (about 1.4 h at 118 MB/s). Licence is CC-BY-NC-SA 4.0 on top of the nuPlan non-commercial terms. I downloaded none of it.
- **Doc-vs-registry conflict (registry wins):** `TRAINING_TIME.md` and `POD_HANDOFF.md` price the A40 at **1.26 to 1.36 s per sample** (4060 measurement times an estimated multiplier). The registry (section 14.1) MEASURED the real run: **2,579,126 samples in 7.04 days = 0.236 s per sample**, 5.4 times cheaper. The planning docs overstate A40 cost; budgets below use the registry.

---

## 1. Q1: which selection methods are PROVEN for end-to-end and imitation planners

"Proven" here means a published, controlled gain on NAVSIM, nuPlan or a closed-loop benchmark, with a stated comparator. "Same compute" = same number of optimiser steps or epochs-times-samples as the comparator.

| # | method class | paper [library key] | benchmark and model | reported effect | same compute? | class | read-across to REFe |
|---|---|---|---|---|---|---|---|
| 1a | embedding-diversity / coreset (k-center, geometric) | MOSAIC baseline [2604.08366] Table 1B | NAVSIM navtrain pool (4,141 ten-second clips), base 460 clips, Hydra-MDP V2-99, EPDMS, 3 seeds | vs random at +100 / +400 / +1,600 clips: **+0.63 / +0.40 / +0.68** (random SD 0.60 / 0.20 / 0.22) | yes (epochs fixed per budget: 60/60/50) | PUBLISHED-PRIMARY | at best 1 to 2 SD. Not a lever |
| 1b | trajectory-entropy pruning | Liu et al. [2512.19270] Table I | nuPlan 1M samples, PLUTO-mini (**300K params**), Random14 closed loop | score 84.2 -> 84.5 at -40 % data; random pruning 78.0 at 40 % but 74.7 at 30 %, i.e. non-monotone | **no**: it is a compute saving, not a gain | PUBLISHED-PRIMARY, weak (single run, no variance, tiny model) | supports "~40 % of a redundant set can go"; says nothing about a gain |
| 1c | de-duplication of near-static frames | NAVSIM filter [2406.15349] section 3.1 | OpenScene -> navtrain | constant-velocity PDMS drops 79 -> 22 after the filter; keeps scenes where CV < 0.8 and human >= 0.8 | n/a (a split definition, not an ablation) | PUBLISHED-PRIMARY | **already applied** to navtrain; further dedup is only a compute saving |
| 2a | hard-first selection by the model's own metric | MOSAIC, "w/o Clustering" and full [2604.08366] Table 1B, Fig 6 | same as 1a | vs random: **+1.63 / +1.52 / +1.56** at +100 / +400 / +1,600 (MOSAIC SD 0.43 / 0.03 / 0.25). Ranking alone gives most of the low-budget (<=800) gain; clustering adds at high budget | yes | PUBLISHED-PRIMARY | the best primary evidence for L1. **Regime caveat:** added-data from a pool, 460-clip base, ranking by a model trained on that base |
| 2b | uncertainty (entropy of trajectory logits) | MOSAIC baseline [2604.08366] | same | vs random: **-0.16 / -0.62 / -0.87** | yes | PUBLISHED-PRIMARY | **negative**: do not use logit entropy as hardness |
| 2c | gradient-valued soft curriculum vs hard subset | Li et al. [2604.00388] | GameFormer, nuPlan, ADE (open loop), 3 seeds | TracIn-weighted full-data 1.704 +- 0.029 m vs uniform 1.772 +- 0.134 m (gap inside the uniform arm's SD); **top-20 % hard subset degrades about 2x** | yes | PUBLISHED-PRIMARY, weak | soft-weight, never prune hard; ADE is not PDMS |
| 2d | difficulty-prioritised sampling | Bronstein et al. [2212.01375] | Waymo fleet data + sim, imitation planner, **10 seeds** | a random 10 % subset equals the full set; prioritising difficult segments: **-15 % collisions, +14 % route adherence with 10 % of the data** | yes (less data) | PUBLISHED-PRIMARY | strongest controlled evidence that volume is not binding; data proprietary, not reproducible |
| 2e | failure-subset post-training | FactorDrive [2608.09591] | NAVSIM, VLM planner; 4,910 of 103,288 tokens with SFT PDMS < 0.80 | 88.4 -> 91.0 after RL on that subset | **unknown: no random-subset control**, RL+MCTS confounded | PUBLISHED-PRIMARY | **not selection evidence** (nothing isolates the subset); belongs to the RL review |
| 3a | scenario-type re-weighting | FlowDrive [2509.21961] Table 2a | nuPlan Val14(R) closed loop, flow planner | none 81.91 -> scenario-type **80.08 (-1.83)** | yes (sampler only) | PUBLISHED-PRIMARY, single run | **negative**: do not re-weight by scenario label |
| 3b | trajectory-pattern re-balancing (k-means K=20 on ego futures, inverse frequency) | FlowDrive [2509.21961] Tables 1, 2a | same; Test14-hard(R) too | Val14(R) 81.91 -> **85.37 (+3.46)**; Test14-hard(R) 69.46 -> **73.09 (+3.63)** | yes (sampler only) | PUBLISHED-PRIMARY, single run | the largest published sampler effect; different planner family and benchmark |
| 3c | targeted long-tail augmentation | Zheng et al. [2412.02689] | 4M demos (30,000 h, in-house), 23 scenario types, open and closed loop | open-loop error follows a power law in volume, **closed loop does not**; small additions of long-tail data beat indiscriminate scaling | no stated matching | PUBLISHED-PRIMARY | supports distribution over volume; not reproducible |
| 4 | curriculum ordering | 2212.01375, 2604.00388 (rows 2c, 2d) | none on a NAVSIM camera planner | | | | **absence** found in 3 web searches (2026-10-11) plus the banked library; not "does not exist" |
| 5 | semantic / VLM scenario retrieval | MOSAIC caption clusters [2604.08366] Fig 5 | Qwen2.5-VL-32B captions -> 6 clusters, navtrain | 61 % and 52 % fewer clips than random to match at 1,600 / 2,400 | yes | PUBLISHED-PRIMARY | VLM captions work as a clustering signal inside MOSAIC; no controlled PDMS gain for pure retrieval found (DriveVLM, nuPlan scenario SQL [2403.04133] are mining tools, not ablations) |
| 6 | data-scaling laws | Naumann et al. [2504.04338]; Baniodeh et al. [2506.08228]; Zheng [2412.02689]; DriveVLA-W0 [2510.12796]; SimScale Fig 4 [2511.23369] | in-house and public | see below | n/a | PUBLISHED-PRIMARY | see below |

**Scaling-law rows, one line each.** Naumann (NVIDIA, 16 to 8,192 h): scaling needs data and capacity together, dynamics differ per task, "open-loop performance does not directly translate to closed-loop". Baniodeh (Waymo, ~500K h): closed-loop metrics also improve with compute; optimal model size grows 1.5x as fast as data. Zheng: closed loop is not a power law in volume (conflicts with Baniodeh; not comparable: different sims and models). SimScale: a log-quadratic fit of navhard EPDMS vs sim volume, saturating earlier for recovery-based pseudo-experts. DrivoR (below) is the only series on REFe's own architecture.

**DrivoR sim-data series (REFe's architecture family, NAVSIM v1 navtest PDMS, ViT-S, single points) [2601.05083] Table 13:** navtrain 93.1; navtrain+navval 93.7; **+65K SimScale 94.0 (+0.3); +134K SimScale 94.6 (+0.9 over trainval)**. EP 90.0 -> 91.6, NC/DAC 98.9/98.9 -> 99.1/99.2. No seeds, no CI.

### What the evidence supports, in four statements

1. **Soft re-weighting beats hard pruning** (2c; 2d; FlowDrive; 1b is neutral at best). L1 below uses soft weights; hard prune is a single optional arm.
2. **Hardness must come from a metric-aligned signal, not model uncertainty** (2a positive, 2b negative).
3. **Re-weight by what the plan looks like, not by a scenario label** (3b positive +3.46, 3a negative -1.83).
4. **Nothing above is measured on a frozen foundation backbone at 100K scale.** MOSAIC's base is 460 clips, FlowDrive and PLUTO train small networks from scratch. The honest read-across discount is large; section 4 carries it into the ESTIMATES.

**Noise.** MOSAIC reports SDs of 0.03 to 0.60 EPDMS over three seeds; the 1a coreset gains sit inside that. REFe has no replicate yet, so its training-seed floor is UNKNOWN. By programme rule (`H-ESTIM-SEED-1`) every arm below carries a replicate (A0b).

---

## 2. Q2: extending the corpus beyond navtrain with PDM-labelable data

### 2.1 What REFe's labelling actually needs

REFe's student is supervised by teacher rollouts plus on-policy scorer labels, built from nuPlan DBs and maps (`navtrain_scenarios.load_split`, `index_dbs()`; see `POD_HANDOFF.md`). Cost per new scene is **about 38.7 shard-seconds**: INHERITED from `POD_HANDOFF.md` (805 + 303 shard-hours over 103,039 scenes, from 7.6 s per rollout and 6.2 s per scorer frame); the division is mine. So +65K scenes is about **700 shard-hours**, i.e. 2.9 days at 10 shards (about 64 vCPU) or 5.8 days at 5 shards. **CPU, not GPU, is the binding constraint on any labelled extension.**

### 2.2 Candidates

| # | corpus | what it is | size (MEASURED unless stated) | licence | download cost | PDM-labelable? | verdict |
|---|---|---|---|---|---|---|---|
| E1 | **OpenScene trainval, frames outside the navtrain filter** | same nuPlan logs; navtrain keeps scenes with CV-agent PDMS < 0.8 and human PDMS >= 0.8 | camera 1,242.21 GB in 200 shards; LiDAR 882.14 GB (not needed); metadata 7.05 GB. Frames: trainval ~650K to 735K (ESTIMATED from MOSAIC's 32,539 clips x 20 and from camera-GB share), so **~550K to 630K outside navtrain**. Logs: trainval 1,250 [2604.08366], navtrain 1,192 (MEASURED) -> ~58 extra logs | CC-BY-NC-SA 4.0 (HF card, MEASURED) + nuPlan non-commercial terms | pod 118 MB/s: **2.9 h**; dev box 1.2 MB/s: 287 h (12 d). Shards are all-camera tarballs, so a 4-camera subset does not shrink the transfer | yes (same DB/map stack as navtrain); 4-camera disk about 0.81 MB per frame (84 GB / 103K, INHERITED) -> ~450 GB for 560K frames | **no published gain; frames are the trivial ones by construction.** DriveVLA-W0 Table 7 shows unfiltered nuPlan pre-training *hurting* two of six arms by 6 PDMS. Hold back; use only as a hard-mined pool (a relaxed filter, not a bulk add) |
| E2 | **SimScale** (planner-based pdm, recovery) | 3DGS-rendered perturbed-ego scenes on navtrain blocks with pseudo-expert targets; released | see 2.3 | data CC-BY-NC-SA 4.0, code Apache-2.0 (repo README via WebFetch: PUBLISHED-SECONDARY; paper section F: MTGS Apache-2.0) | round 0: 609 GB = **1.4 h** pod; 141 h (5.9 d) dev box | yes in principle; **loader adapter UNVERIFIED** (sim tokens are not in any nuPlan DB) | **L3** |
| E3 | **navhard / NAVSIM v2 sets** | navhard_two_stage: 244 real + 4,164 synthetic (MTGS) observations, **a test split**; warmup_two_stage; private_test_hard | navhard curr 12.80 + hist 25.52 + pickles 0.21 GB; warmup 1.16 GB; private_test_hard 10.54 GB | as OpenScene | 50 GB total | not as training | **eval only.** Training on it destroys the secondary benchmark (`navhard EPDMS`) |
| E4 | **nuPlan logs without sensors** (1,282 h total; sensors for 128 h) [2403.04133] | annotations and maps for ~10x the camera hours | 200+ TB raw sensor; DBs only needed | nuPlan terms | DBs: cheap; **no pixels** | PDM yes; image no | usable only through rasterised annotations (RAP, section 2.4): a research build, not a lever for this week |
| E5 | OpenScene test frames outside navtest | other frames of the same test drives | 128 GB camera | | | | **excluded**: same drives as the benchmark |

### 2.3 SimScale in detail

**What it is** (PUBLISHED-PRIMARY [2511.23369]): MTGS 3D-Gaussian-splatting blocks reconstructed from navtrain logs (blocks with novel-view PSNR < 27 dropped). Ego end-points are perturbed over +-20 m longitudinal and +-2 m lateral (5 m and 0.5 m steps) from the 16,384-trajectory vocabulary subset with EPDMS >= 0.8 and bounded heading offset. Other agents run IDM reactively. Targets come from a **pseudo-expert**: *recovery-based* (nearest human manoeuvre) or *planner-based* (PDM-Closed from the perturbed state), kept only when all EPDMS sub-metrics pass (EP >= 0.5). Up to five non-overlapping sampling rounds. The paper's text states **237K planner-based** and **147K recovery-based** scenes; DriveZero's "237K" is the planner-based set.

**What is released** (HF API listing 2026-10-11, MEASURED, nothing downloaded; script `raw/data_selection/hfprobe.py`, output `raw/data_selection/hf_tree_sizes_2026-10-11.txt`):

| split | tokens (README, PUBLISHED-SECONDARY) | history-only blobs | meta |
|---|---|---|---|
| planner-based round 0 / 1 / 2 / 3 / 4 | 65K / 55K / 46K / 38K / 32K (=236K) | 609.09 / 518.28 / 430.58 / 356.55 / 298.46 GB (= **2,212.96 GB**) | 12.81 GB total |
| recovery-based round 0 to 4 | 45K / 36K / 28K / 22K / 17K (=148K) | 422.40 / 337.63 / 261.25 / 202.50 / 158.37 GB (= **1,382.16 GB**) | 8.41 GB total |
| GTRS reward pickles (16,384-vocab) | n/a | pdm 70.47 GB, recovery 43.74 GB (REFe does not need these: it scores its own proposals) | |

Clips are 6 s at 2 Hz in OpenScene format; the README says `sensor_blobs_hist` alone suffices for policy training. Transfer on a pod at 118 MB/s: round 0 **1.4 h**, rounds 0 to 1 **2.7 h**, all planner-based **5.2 h**, everything **8.5 h**.

**Cameras (UNVERIFIED, material for REFe).** The SimScale planners used **3 cameras** (front, front-left, front-right, 2048x512; paper Table 6). The README tree shows `CAM_B0`. DriveZero-Scale trained a **4-camera** model on SimScale, which implies B0 exists (possibly the original image). The 0.9 MB index file `simscale_pdm_v1.0-4_sensor_blobs_hist.json` would settle it; I did not download it (permission). **Probe ready, one file.**

**What it buys (PUBLISHED-PRIMARY):**

| result | model | navtest | navhard |
|---|---|---|---|
| DriveZero ViT-L, 100K -> +237K [2609.06055] Table 4/5 | closest to REFe | **PDMS 94.8 -> 95.3 (+0.5)** | **EPDMS 51.5 -> 57.1 (+5.6)** |
| DrivoR ViT-S, +65K / +134K [2601.05083] | REFe's architecture | PDMS +0.3 / +0.9 over trainval | EPDMS 52.3 / 54.6 |
| SimScale GTRS-Dense R34 / V2-99 [2511.23369] Table 2 | | **EPDMS** 82.3 -> 84.6 / 84.0 -> 84.6 | up to +8.6 |
| RAP recovery perturbation [2510.04333] Table 6 | | **PDMS 92.5 -> 92.5 (no effect)** | v2 two-stage 32.5 -> 36.9 |

Both RAP and SimScale agree: recovery/perturbation data **moves robustness (navhard) a lot and navtest PDMS a little or not at all**, and the gain shrinks as the baseline strengthens (+2.3 on R34, +0.8 on V2-99). For a bar defined on navtest PDMS this is a weak lever.

**Generation cost** of SimScale is not stated in the paper or appendix: UNVERIFIED. Regenerating more (MTGS is Apache-2.0) is not a one-A40-week item.

### 2.4 The raster route (E4), recorded for completeness

RAP [2510.04333] rasterises annotated primitives for **all 1,200 h** of OpenScene logs (about 120 h with real cameras), perturbs ego trajectories in a random 10 % subset, and aligns raster features to real-image features ("Raster-to-Real"). It reports the v1 null / v2 gain above. A frozen DINOv3 with a rank-32 LoRA is an unproven fit for the alignment loss. HYPOTHESIS; not recommended this week.

### 2.5 Can log-disjointness from navtest be guaranteed? Yes, and it is stronger than log-level.

MEASURED 2026-10-11 (`raw/data_selection/drive_disjoint_check_2026-10-11.txt`):

| check | result |
|---|---|
| navtrain logs / tokens / drive ids | 1,192 / 103,288 / 162 |
| nuPlan test DBs on the dev box / their drive ids | 1,349 / 132 |
| navtrain logs present in nuPlan test DBs | **0** |
| navtrain drive ids shared with test drive ids | **0** |
| navtrain logs present in the nuPlan val DBs held here | 214 (the other 978 are nuPlan train, held on the pod) |

navtest's 136 logs are a subset of the 1,349 test DBs (`eval/SPEC_NAVTEST.md`), so zero overlap with all test DBs implies zero overlap with navtest. A "drive id" is the log name without its two trailing frame-range fields; adjacent segments of one physical drive share it, so this also rules out adjacent-segment leakage.

**Guard to run before any extension (one line each):**
1. `assert set(ext_logs).isdisjoint(navtest_logs)` and the same on drive ids;
2. only draw extension logs from nuPlan **train/val** (never from the test split, even logs outside navtest's 136);
3. record both sets' sha256 in the run's `config.json`.

**What the guarantee does not cover:** the four nuPlan cities repeat the same roads across train and test drives, so "drive-disjoint" is the benchmark's standard, not scene novelty. State it that way in any claim.

---

## 3. Q3: combining PhysicalAI-AV with four cameras

### 3.1 What exists (two probes: HF dataset card and API, fetched 2026-10-11, MEASURED)

PhysicalAI-Autonomous-Vehicles: 306,152 clips x 20 s = **1,700 h**, 25 countries, 2,500+ cities, **133 TB**; multi-camera for all clips, LiDAR for 298,326, radar for 160,761. **Seven cameras**, 1080p mp4 at 30 fps: cross left 120, cross right 120, front wide 120, front tele 30, rear left 70, rear right 70, rear tele 30. The card states: *"we do not include open maps data"*. Licence: NVIDIA Autonomous Vehicle Dataset License (AV development "using NVIDIA technology", no redistribution, 12-month term; the card says commercial and non-commercial AV use). Recorded as a fact; the programme ruling is to use it.

### 3.2 Geometry against REFe's four nuPlan cameras

nuPlan values are MEASURED from `data/nuplan_cam_calib.json` (quaternion w,x,y,z, optical-axis yaw): fx = fy = 1545 at 1920x1080, pinhole HFOV 63.7 degrees, **true HFOV 72.25 degrees** after Caltech distortion (`REVIEW_5_CAMERAS.md`). The four-camera set covers 70.8 % of the horizon with a **52.7 degree** blind sector on each rear quarter.

| REFe camera | nuPlan yaw | closest PhysicalAI camera | PhysicalAI geometry | mismatch |
|---|---|---|---|---|
| CAM_F0 | -0.1 deg | `camera_front_wide_120fov` | cylindrical, HFOV **120 deg** (in-repo, `bev_aux.py:30`), 1080p | 1.66x lower native angular resolution. A centre crop to 72 deg is 60 % of the width and matches REFe's 13.3 px/deg at 960 px; the projection model still differs (cylindrical vs distorted pinhole) |
| CAM_L0 | **+55.2 deg** | `camera_cross_left_120fov` | 120 deg; **mounting yaw not probed: UNVERIFIED** | same projection issue; a crop around +55 deg is only possible if the cross camera's yaw is inside +-(55 +- 36) deg |
| CAM_R0 | **-55.0 deg** | `camera_cross_right_120fov` | as above | as above |
| CAM_B0 | 180.0 deg | `camera_rear_tele_30fov` (30 deg) or rear left/right 70 | tele is a narrow zoom | no like-for-like; B0 at 72 deg vs 30 deg is a 2.4x scale change |
| (none) | | `front_tele_30` | | no REFe analogue |

Also: PhysicalAI has two sensor rigs (cy about 543 vs about 755; INHERITED from a memory note, `physicalai-single-rig`), 30 fps video that must be decoded and sub-sampled to 2 Hz, and our episode build reads only the front-wide camera plus intrinsics, extrinsics, vehicle dimensions and egomotion (CLAUDE.md read-set table). **Extrinsics for the six non-front cameras were not probed**; that is the first thing to settle, from one small calibration file on the pod.

### 3.3 What labels are missing

The PDM score is a product of components that need different inputs:

| PDMS component | needs a map? | obtainable from PhysicalAI-AV |
|---|---|---|
| NC (no at-fault collision), TTC | no: agent boxes + ego rollout | **yes, partly:** `obstacle.offline` carries 3D tracks on **97.44 %** of the corpus (CLAUDE.md); proposals would need a map-free simulator |
| C (comfort) | no | yes (ego kinematics) |
| DAC, DDC, TLC | yes (drivable area, lane direction, signals) | **no**; SAM3 gives an image-space road mask only |
| EP (ego progress) | yes (centreline progress) | **no** |
| route / goal | yes | no route supplier except the ego's own future path |

So **no PDM/DAC/EP targets and no map-based teacher** (DriveRL-style rollouts need the map). The private SAM3 corpus (`Sayood/tanitad-v7-training-corpus`, 4,719 clips, `semantic_maps/` and `agents/obstacle_offline/`) and the v7.2 labels (traffic-light tactical goals on 779 of 4,572 records, MEASURED in CLAUDE.md) are the only extra supervision. Production of the SAM3 maps was launched 2026-09-15 for about 7.9 days; **completion is UNVERIFIED** here, and whether the corpus holds only front-wide frames is also UNVERIFIED.

### 3.4 Which objectives it can still serve

| objective | admissible under PI rulings? | value |
|---|---|---|
| O1 human-trajectory imitation (egomotion, 4 s at 2 Hz) | yes | **low.** REFe's student is trained on teacher rollouts, not on human logs; adding human targets re-introduces what DriveZero removes |
| O2 map-free NC/TTC scorer targets from `obstacle.offline` | yes (labels only) | medium, large engineering (a map-free proposal simulator); HYPOTHESIS |
| O3 auxiliary drivable-area head from SAM3 pseudo-masks, vision-only inference | **yes**: map-derived labels allowed for training, never as input | the only objective aimed at REFe's dominant defect (DAC, about 5.0 PDMS points, `REVIEW_7`); labels are pseudo and noisy; `REVIEW_7` P6 proposes the head already. HYPOTHESIS |
| O4 representation (LoRA adaptation on extra viewpoints) | yes | HYPOTHESIS; domain gap (25 countries vs 4 nuPlan cities) cuts both ways |

### 3.5 Published evidence on cross-dataset co-training for a NAVSIM planner

**Searches:** 3 web searches (2026-10-11) plus a library keyword pass over 606 entries. **Not found:** any study that adds Waymo, nuScenes or PhysicalAI to a NAVSIM camera planner and reports a PDMS delta. This is absence at limited locations, not proof.

**The nearest result, PUBLISHED-PRIMARY [2510.12796] Table 7** (DriveVLA-W0, 256x144 images, pre-train on unfiltered nuPlan 8K steps then fine-tune on NAVSIM 4K steps; single runs):

| model | NAVSIM from scratch | nuPlan pre-train + NAVSIM fine-tune | delta PDMS |
|---|---|---|---|
| TransFuser-50M | 79.2 | 83.6 | **+4.4** |
| TransFuser-7B | 77.9 | 71.6 | **-6.3** |
| VLA-VQ (action only) | 68.7 | 62.2 | **-6.5** |
| VLA-W0-VQ (+ world model) | 80.7 | 85.6 | **+4.9** |
| VLA-ViT | 70.3 | 70.6 | +0.3 |
| VLA-W0-ViT | 83.9 | 85.3 | +1.4 |

The authors attribute the two large drops to "overfitting to NuPlan's action distribution" under sparse action supervision, and the gains to dense visual supervision. Two readings that matter here: (i) **cross-corpus pre-training is a coin flip for action-only supervision**, (ii) it is the closest published analogue to E1. UniPlanner [2510.24166] is trajectory-only (no cameras, not NAVSIM) and is adjacent evidence only.

**Verdict:** do not co-train PhysicalAI-AV into the PDMS pipeline now. If a PhysicalAI arm is wanted, O3 is the one with a mechanism, and it should be a separate pre-registered auxiliary-head arm after L1 to L3.

---

## 4. Q4: ranked recommendation (three cheapest data levers on one A40 for 7 days)

### 4.1 Budget basis

MEASURED (`MODEL_REGISTRY.md` 14.1): the full REFe run is **2,579,126 samples, 10,075 steps, 7.04 days on one A40** = 0.236 s per sample including the on-policy dump. A head-and-scorer continuation of **300K sample-passes is about 19.7 A40-hours** (ESTIMATED: this assumes the continuation costs the same per sample as the full run; `REVIEW_7` independently estimates 10 to 20 A40-h for an SFT-4-style fine-tune). One week (168 h) therefore holds **six arms of 300K passes (about 120 h) plus evaluation**, or the equivalent of one full 25-epoch retrain. Labelling is CPU and runs in parallel (section 2.1).

All three levers start from the same checkpoint and take their signals from P1's relabel (per-token NAVSIM PDMS of the student's own pick on the banked on-policy sets; 190,696 sets, 127,385 at epoch 25). **Do not start before P1's relabel exists.**

### 4.2 Ranking

| rank | lever | label cost | expected effect on navtest PDMS | evidence class and basis | why this rank |
|---|---|---|---|---|---|
| 1 | **L1 failure-aware soft re-weighting of navtrain**, teacher-clean tokens only | none beyond P1 | **+0.5 [0, +1.5]** | ESTIMATED. Basis: MOSAIC hard-first +1.5 EPDMS (2a), Bronstein (2d), soft > hard (2c). Discount x0.3 to 1.0 for regime (whole-pool re-weighting, frozen ViT-L, PDMS not EPDMS). No REFe measurement | cheapest, aimed at the measured defect (72 % of zero picks sit in dirty-fan tokens), signal is free |
| 2 | **L2 trajectory-pattern re-balancing** (k-means K=20 on executed futures, inverse-frequency, capped) | none | **+0.3 [-0.5, +1.0]** | ESTIMATED. Basis: FlowDrive +3.46 / +3.63 (3b), discounted heavily: different planner, closed-loop nuPlan, single run. Scenario-type weighting is *negative* (3a) | label-free and decorrelated from L1; carries a published negative control |
| 3 | **L3 SimScale-lite** (planner-based round 0, 65K scenes, sim:real <= 0.63) | 609 GB pull + ~700 shard-h if teacher-relabelled + a loader adapter (UNVERIFIED) | **navtest +0.3 [0, +0.9]; navhard EPDMS +3 to +6** | navtest: PUBLISHED-PRIMARY series on DrivoR (+0.3 at 65K) and DriveZero (+0.5 at 237K). navhard: PUBLISHED-PRIMARY DriveZero +5.6 | the only extension with primary support on this architecture; costly and its navtest effect is within one SD |

**Combined, not additive:** ESTIMATED at most +1.5 PDMS against a 6.30 gap. `REVIEW_7`'s P1 (+2 to +4, ESTIMATED with V3r +2.00 [+0.53, +3.57] MEASURED) and P2 come first.

### 4.3 L1: pre-registration skeleton

**Hypothesis:** weighting each navtrain token by the student's own failure raises navtest PDMS because the dominant loss is zero picks on a minority of tokens.
**Weights:** `h_i = 1 - PDMS_i(pick)` (NAVSIM v1 formula) from the final-epoch on-policy sets; `w_i = 1 + 2 h_i`, capped at 4, normalised to mean 1; one value of the multiplier fixed now (no tuning). **Teacher-clean gate:** weight only tokens whose teacher target passes NAVSIM (DAC and NC), otherwise the scheme amplifies broken supervision. MEASURED: 4.97 % [1.30, 10.31] of held-out teacher targets leave NAVSIM's drivable area and 72 % of zero picks sit in dirty fans (`REVIEW_7`). This overlaps P5 (target cleaning); L1 uses P5's output if it lands first.

| arm | definition | purpose |
|---|---|---|
| A0 | uniform continuation, 300K passes | comparator |
| A0b | A0 with a different data-order seed | training-noise floor (`H-ESTIM-SEED-1`) |
| A1 | soft failure weights, teacher-clean gate | the lever |
| A1c | same weight multiset, permuted across tokens | separates "hardness signal" from "non-uniform sampling" |
| A2 (optional) | prune the easiest 40 % (h = 0, clean), re-invest passes | tests the published warning (2c): expected to be neutral or negative |

**Measurement:** PDMS on the registered held-out navtrain-log split with PDM truth first; navtest once, for the confirm run, with the decision frozen beforehand (navtest is already EXPLORATORY after many reads, `REVIEW_7`). Paired log-cluster bootstrap (136 navtest logs, 10,000 resamples). It answers "another draw of logs" only; the replicate answers "another training run". Report NC, DAC, EP, TTC, C separately (NAVSIM sub-scores); the strategic family stays PAUSED per PI item 25; the programme's longitudinal / lateral / tactical families are not separately computable on NAVSIM beyond these sub-scores, which is stated here rather than silently dropped.
**Committed outcomes:**
- SUCCESS iff (A1 - A0) >= +0.50 PDMS with lower CI > 0, **and** (A1 - A1c) > 0 with lower CI > 0, **and** |A0b - A0| <= half of (A1 - A0). Then adopt into the next full run's recipe.
- PARTIAL if A1 > A0 only on the held-out split: record, do not adopt, run L2.
- FAIL otherwise. Log the elimination (weighting on student failure does not move PDMS at this scale) and **the next arm is L2**. Committed prediction: A1 improves DAC and NC first; if EP falls by more than 0.5 the gain is a caution trade and is reported as such.

### 4.4 L2: pre-registration skeleton

**Hypothesis:** inverse-frequency sampling over trajectory-pattern clusters raises PDMS on turns and off-route goals (dirty fans: 22.6 % dirty at bearing >= 60 degrees vs 10.3 % straight; 40.6 % off-route, MEASURED in `REVIEW_7`).
**Arms:** A0 and A0b shared with L1; **B1** = k-means (K = 20) on normalised 4 s ego futures, per-cluster inverse-frequency weights, cap 5x; **B2 (negative control, optional)** = scenario-type re-weighting, published to be worse (3a); **B3** = B1 + A1, run only if A1 and B1 each reach PARTIAL or better.
**Bar:** (B1 - A0) >= +0.30 with lower CI > 0 and above the A0b floor. **If it fails:** record "trajectory-pattern balance is not binding at this scale", go to L3.

### 4.5 L3: pre-registration skeleton

**Gate zero (no GPU):** the one-file camera probe (section 2.3) and a loader-adapter feasibility read. If the teacher cannot consume OpenScene-format sim pickles, L3 is **BLOCKED on engineering**, named as such.
**Arms:** A0 (control) and **C0 = A0 plus the same number of extra *real* passes** (so the +65K sim samples are not credited for added steps); **C1** = +65K planner-based round 0 mixed at <= 0.63 sim:real, targets relabelled by REFe's teacher (authors' PDM-Closed trajectories are the fallback).
**Metrics:** navtest PDMS (primary) and navhard EPDMS (secondary, where the published effect lives).
**Bar:** navtest (C1 - C0) >= +0.30 with lower CI > 0, or navhard EPDMS (C1 - C0) >= +3.0. A navhard-only pass is recorded as a robustness gain, **not** as progress toward the 94.55 bar.
**Needs a PI decision:** the 609 GB pull.

### 4.6 What not to do, with the reason

| item | reason |
|---|---|
| train on navhard / NAVSIM v2 test sets | breaks the secondary benchmark (E3) |
| bulk-add unfiltered OpenScene frames (E1) | trivial by construction; DriveVLA-W0 Table 7 shows -6 PDMS on action-only models |
| logit-entropy "uncertainty" mining | published negative (2b) |
| scenario-type re-weighting | published negative (3a) |
| hard-prune to a small hard subset | published degradation about 2x (2c) |
| full MOSAIC (clusters + scaling pilots) | needs pilot trainings per cluster; the ranking half is L1 |
| PhysicalAI-AV in the PDMS pipeline | no map, no B0 analogue, coin-flip transfer (section 3) |

### 4.7 If everything fails (rule zero)

If L1, L2 and L3 all fail their bars, the committed conclusion is that **REFe's residual error is not data-limited at this scale**, and the lever list returns to P1/P2/P5 and the scorer. Blockers named: P1's relabel output (CPU, 85 worker-hours), the PI download decision for L3, and the pod's vCPU count (shards = vCPU / 6).

---

## 5. What I did not do, and what is UNVERIFIED

- **Not downloaded (permission):** the SimScale index JSON (camera set), any PhysicalAI calibration file, any sensor data. Both probes are one small file each.
- **UNVERIFIED:** SimScale B0 presence and generation cost; PhysicalAI cross/rear camera yaw; completion of SAM3 production and which cameras the private corpus holds; the loader adapter for sim tokens; the 1,250-log trainval count (published in MOSAIC, not recounted); trainval frame count (ESTIMATED 650K to 735K); pod disk quota for E1/E2.
- **Not read in full:** I read the abstract, setup and results sections of the cited papers, not every appendix. FlowDrive Table 2a and DriveVLA-W0 Table 7 were read from the PDF text; MOSAIC Table 1 from raw text, with the garbled column layout checked against the table row order.
- **R2SE** (hard-case refinement on NAVSIM) surfaced in search; I could not identify its arXiv id with certainty, so it is neither cited nor banked.
- The WebFetch tool summarised the NAVSIM docs and the SimScale/OpenScene READMEs: those rows are PUBLISHED-SECONDARY.
- No GPU was used; no pod was touched.

---

## 6. Library banking (CLAUDE.md research-banking rule)

Newly banked this pass (PDF + sha256 via `tools/kb_add.py`, tag `refe-data-selection`, cited by this file): **2509.21961** (FlowDrive), **2512.19270** (trajectory-entropy pruning), **2604.00388** (gradient-valued curriculum), **2212.01375** (zero-shot curricula), **2510.24166** (UniPlanner; its first note was a guess, the corrected note is appended). Already banked and relied on: 2604.08366, 2511.23369, 2609.06055, 2601.05083, 2510.04333, 2510.12796, 2412.02689, 2504.04338, 2506.08228, 2406.15349, 2403.04133, 2506.04218, 2608.09591, 2503.12552. `kb_add --verify` is recorded in the hand-back.

---

## 7. Deliverable manifest

| artifact | location | notes |
|---|---|---|
| this review | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-10-11-refe-data-and-rl/DATA_SELECTION_REVIEW.md` (staged) | only copy |
| HF listing script | `repo: .../2026-10-11-refe-data-and-rl/raw/data_selection/hfprobe.py` (staged) | metadata API only |
| HF tree sizes (OpenScene, NAVSIM v2, SimScale) | `repo: .../raw/data_selection/hf_tree_sizes_2026-10-11.txt` (staged) | MEASURED 2026-10-11 |
| drive-disjointness check | `repo: .../raw/data_selection/drive_disjoint_check_2026-10-11.txt` and `drive_disjoint_check.py` (staged) | MEASURED; the script reproduces the output from the repo root |
| 5 newly banked PDFs + index rows | `repo: TanitAD Research Lab/Library/papers/` (5 PDFs), `Library/library.json`, `Library/LIBRARY.md` (staged) | the two index files are shared with other agents' concurrent banking; staging them stages their rows too |
| extracted text used for reading | scratchpad only | not kept; reproducible with `pdftotext -raw` |
