# AV Dataset Landscape (D-012 standing duty — HuggingFace-first)

> One row per corpus. **License class** governs public claims (see DATA_STRATEGY §4 firewall):
> `public` = citable in demos/papers · `research/NC` = internal training only, no public claim ·
> `gated/confidential` = excluded from public claims entirely · `source-derived` = inherits YouTube
> video rights (gray).
> **Urban richness** = intersections / pedestrians / lights / night / weather density (comma2k19's
> highway commute is the low anchor). **Cost to first batch** = engineer-hours to a contract episode
> on our pipeline. **Status:** `loaded` (adapter + contract test) · `candidate` · `probe-only`.
> Maintained by the Data Engineering agent. Last sweep: **2026-07-15** (WorldModel-Synthetic pose gate CLOSED =
> pose-less; PandaSet loader shipped + geometry blocker; ZOD license corrected CC-BY-SA; Cosmos-Synthetic tracked).

## Tier 1 — in the Phase-0 pipeline

| Corpus | Size | Sensors | Actions | License class | Urban richness | Cost→batch | Status |
|---|---|---|---|---|---|---|---|
| **comma2k19** (`commaai/comma2k19`) | ~100 GB, 33 h, 10×8.7 GB chunks | front cam 20 fps + CAN + GNSS | **real CAN** (steer wheel, speed) | **public (MIT)** | low (highway commute only) | ~1–2 h (0 new code) | **loaded** (D-009) — public anchor |
| **Cosmos-Drive-Dreams** (`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`) | 5 843 RDS-HQ clips + **81 802** synth videos, 30 fps, 121-frame chunks | 7-cam rig (front_wide_120fov used), HD map, LiDAR, 4×4 poses | derived (ego 4×4 `vehicle_pose` → steer/accel) | **public (CC-BY-4.0)** | **high** (7 weathers: rain/snow/fog/night; intersections, VRUs) | ~2–3 h (**this run**) | **loaded** (D-014, this run) — public synthetic |
| **PhysicalAI-AV** (`nvidia/PhysicalAI-Autonomous-Vehicles`) | 1 727 h, 25 countries, 2 500+ cities | multi-cam + radar + lidar | egomotion (poses → yaw-rate/accel) | **gated/confidential** (NVIDIA AV licence, internal-dev-only, 12-mo) | **very high** | ~3–4 h | **loaded** (D-012) — `data:physicalai` tag, **no public claim** |
| **PhysicalAI-WorldModel-Synthetic-Scenarios** (`nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios`) | **264 k clips / ~1 467 h / 8.3 TB**, 7-cam @24 fps ~462 frames; families emergency/lanechange/nudging/pedestrian/weather_degradation | 7 camera mp4s (front_wide, front_tele, 3 fisheyes, rear L/R) + per-cam **Qwen2.5-7B caption** + `{weather,time_of_day,surface_type,region}` | **CONFIRMED POSE-LESS** (real bytes 2026-07-15) → **IDM (H7) / video-only / semantic-index** | **OpenMDW-1.1 — UNGATED** (proposed D-022, firewall held) | high (targeted long-tail) | pixels IDM-gated (H7, Phase-1); **captions/metadata usable NOW ($0)** | **candidate** — semantic-label index today (P1 2d + SC-02/05/06); action-loader = Phase-1 IDM |

## Tier 2 — real urban corpora (H4 arm-B / H7 scale-up / D8 OOD)

| Corpus | Size | Sensors | Actions | License class | Urban richness | Cost→batch | Status |
|---|---|---|---|---|---|---|---|
| **PandaSet** (`georghess/pandaset` mirror) | 103 seq × 8 s @10 Hz (~48 k imgs, ~44.5 GB) | 6-cam (front pinhole used) + lidar + GPS/IMU | derived (pose→steer/accel, no CAN) | **public (CC-BY-4.0)** — owned core | med (SF/El-Camino urban day, no night/weather) | ~2–3 h loader **DONE** (blocked on geom) | **loaded-blocked** (2026-07-15, intake) — front fx=1970 height-bound → f_eff 467≠266 + k1 distortion; needs **D-016 R1 pad-crop+undistort** |
| **BDD100K** | 100k clips 40 s, ~1.8 TB | front cam + GPS/IMU | weak (GPS/IMU; no CAN) → **needs IDM (H7)** | research/NC | **very high** (diverse US urban, weather, night) | ~6–8 h (IDM pseudo-labels) | candidate — H7 pseudo-label target |
| **nuScenes** | 1 000 scenes 20 s, 2 cities | 6-cam + radar + lidar | ego pose | research/NC | high | ~4 h | probe-only (D8 OOD, never trained) |
| **nuPlan** | 1 500 h, 4 cities | cam + lidar + tracks | ego + agent tracks (planning) | research/NC | high | ~6 h | candidate (NAVSIM/planning link, Bench&Eval) |
| **Argoverse 2** | 1 000 h sensor + 250k forecasting | 7-cam + 2 lidar | ego + tracks | research (CC-BY-NC 4.0) | high | ~6 h | candidate |
| **Waymo Open** | 2 030 seg 20 s + motion | 5-cam + 5 lidar | ego + tracks | research/NC (Waymo licence) | high | ~6 h | candidate |
| **Zenseact ZOD** | Frames/Sequences/Drives, ~1 473 h drives | cam (KB fisheye) + lidar + radar + CAN | **real CAN** (EU) | **CC-BY-SA-4.0** (corrected 2026-07-13 — was mis-tagged research/NC; owned-copyleft) | **high (14 EU countries, day/night, seasons/weather)** | ~5 h (fisheye→ftheta + D-016 R1 pad-crop) | **candidate #1** — owned real-urban PhysicalAI replacement (OWN_DATASET_PLAN §7) |
| **ONCE** | 1M scenes | cam + lidar | ego | research/NC | med (lidar-centric) | ~6 h | low priority |
| **CoVLA** | ~10 k clips 30 s | front cam | **actions + language captions** | research/NC | high | ~4 h | candidate — H12 command-conditioning link |

## Tier 3 — scale-up / uncontrolled diversity (Phase 1, H7 flywheel)

| Corpus | Size | Sensors | Actions | License class | Urban richness | Cost→batch | Status |
|---|---|---|---|---|---|---|---|
| **OpenDV-2K / OpenDV-YouTube** | ~2 000 h | front cam only | **none** → IDM (H7) | **source-derived** (YouTube rights, gray) | very high | ~8–10 h (IDM + focal canon on heterogeneous video) | Phase-1 candidate — the 1000× data thesis |
| **comma L2D / comma-steering-control** | GB-scale | front cam + steering | real steering | public (MIT-class — verify) | low–med | ~2 h | candidate — real-CAN augment |
| **Own GoPro / smartphone** | on demand | front cam | via IDM (H7) | own | controllable | focal-canon prototype (backlog #3) | Phase-1 |

## RANKED acquisition queue (added 2026-07-08 by the MVP orchestrator, per Sayed's directive:
## "leverage all high-quality possible data; PhysicalAI-AV in despite license — solve/replace later")

Score = edge value (urban richness × action quality × hypothesis/gate coverage) ÷ cost, license as
a *tag* not a blocker (public claims stay firewalled to comma+Cosmos regardless).

| # | Corpus | Why this rank | Action | Owner / when |
|---|---|---|---|---|
| 1 | **PhysicalAI-AV R1 (500→2,000 urban clips)** | richest real urban corpus we have; already loaded+tagged; Sayed-directed; feeds the 30k-follow-up training mix directly | run the R0 scorer at R1 scale on the pod (disk OK, 284 GB free) | DataEng Tue + MVP assists; **this week** |
| 2 | **PhysicalAI-WorldModel-Synthetic-Scenarios** | THE long-tail material (emergency/pedestrian/weather-degradation) for H6/H15/D9 + scenario DB data rows; loader mirrors cosmos (shared pose code) | verify HF card license → pilot 50 clips | DataEng Tue (backlog P1.4→P0) |
| 3 | **Zenseact ZOD** | real-CAN corpus #2 (EU/night/winter — everything comma lacks); H4 arm-B + D8 real-OOD probe | pilot loader on 5 drives (~5 h) | DataEng next runs |
| 4 | **Cosmos-Drive-Dreams expansion** | more shards → bigger weather-paired sets for SC-05/D8 + D-010 mix share | extract shards 001+ as needed (pod, streaming) | MVP loop, on demand |
| 5 | **BDD100K** | H7 flywheel proof-of-concept (IDM pseudo-labels on weak-action data); very high diversity | after 30k ckpt (needs trained inv-dyn) | DataEng, Phase 0→1 boundary |
| 6 | **nuScenes** | reserved D8 OOD probe (never trained — that's its value) | loader at gate-D8 time | Benchmarks+DataEng |
| 7 | **CoVLA** | language-conditioned driving (H12 bridge), Phase 1 | assess at Phase 1 kickoff | DataEng |
| 8 | **nuPlan / Argoverse 2 / Waymo Open** | benchmark-adjacent; value rises when closed-loop/forecasting comparisons start | defer to Phase 1 | DataEng |
| 9 | **OpenDV-2K (YouTube)** | the 1000× thesis at scale; needs robust heterogeneous-focal canon + IDM | Phase 1 flagship experiment | DataEng |
| 10 | comma L2D / steering-control | small real-CAN augment; cheap but low marginal value vs ZOD | opportunistic | DataEng |

## PhysicalAI-AV R1 status (measured 2026-07-09 — rank #1)

Scoring the **30 already-cached egomotion chunks** (offline, R0 scorer) yields **1,926 gate-passing urban
clips (67.6 % of 2,850)** — R1=2,000 is **74 short from cache**, needs ~1–2 more egomotion chunks. Camera
materialisation = the **same 30 chunks as R0**, ~60 GB (per-chunk cost) → **3.85× clips for R0's bandwidth**;
fetch-plan rule = extract ALL gate-passing clips per downloaded chunk. Selection: 23 countries, all 24 h,
mean 8.1 m/s. Artifacts: `.../physicalai/r1/{r1_selection.parquet,R1_REPORT.json}`; tool = intake pkg
`2026-07-09-physicalai-r1-selection/`. Next: 2-chunk top-up on pod → clear 2,000 → camera fetch.

## Sweep notes (2026-07-15)

- **WorldModel-Synthetic-Scenarios pose gate CLOSED (real bytes):** POSE-LESS — captions+metadata only, no
  vehicle_pose/CAN. Reclassified: pixels = Phase-1 IDM (H7) target; captions/metadata = **usable-now semantic
  index** (BACKLOG P1 2d + SC data rows). Its rank-#2 "mirror the cosmos loader" plan is retired.
- **PandaSet loader shipped (intake, 16✓) but geometry-BLOCKED:** real front fx=1970 on 1920×1080 → square-crop
  is height-bound → f_eff 467≠266 (1.75× scale mismatch) + ignored k1=−0.589 distortion. **Rule: fx>1122 px on a
  1080-tall frame is not square-croppable to 266.** Fix = D-016 R1 pad-crop+undistort in `calib.py` — now a
  **blocking prerequisite for the entire owned real-urban tier** (ZOD/Udacity hit it too), not a PandaSet detail.
- **ZOD license corrected** research/NC → **CC-BY-SA-4.0** (OWN_DATASET_PLAN §7, 3 independent fetches). Promotes
  ZOD to owned-copyleft candidate #1 for the real-urban PhysicalAI replacement.
- **New watch:** `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic` (card-only, no payload yet). `Newsflare`
  AV videos = stock-video copyright barrier (excluded). No new ungated real-AV video corpus → gap stays ZOD-shaped.

## Sweep notes (2026-07-09)

- **Loaded this run:** Cosmos-Drive-Dreams — the first *publicly-claimable* rich AV corpus (CC-BY-4.0),
  closing the gap left when the license review excluded real PhysicalAI-AV from public claims.
- **Adjacent watch (models, not datasets):** X-world (2026, controllable ego-centric multi-cam WM),
  EOT-WM / Trajectory-NWM (ego+other trajectory conditioning) — relevant to H2/H15, not ingestible corpora.
- **Next sweep priorities:** (1) verify PhysicalAI-WorldModel-Synthetic-Scenarios card license → if
  ungated, mirror the Cosmos loader (near-zero cost, shared pose/contract code); (2) Zenseact ZOD as a
  second real-CAN corpus for H4 arm-B (EU/night distribution comma2k19 lacks); (3) monthly HF `datasets`
  new-release sort for 2026 AV video drops.
- **Firewall reminder:** all public open-loop numbers stay **comma2k19 + Cosmos-Drive-Dreams**; every
  experiment row carries its `data:` tag so exposure is one `grep`.
