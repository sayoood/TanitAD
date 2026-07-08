# AV Dataset Landscape (D-012 standing duty — HuggingFace-first)

> One row per corpus. **License class** governs public claims (see DATA_STRATEGY §4 firewall):
> `public` = citable in demos/papers · `research/NC` = internal training only, no public claim ·
> `gated/confidential` = excluded from public claims entirely · `source-derived` = inherits YouTube
> video rights (gray).
> **Urban richness** = intersections / pedestrians / lights / night / weather density (comma2k19's
> highway commute is the low anchor). **Cost to first batch** = engineer-hours to a contract episode
> on our pipeline. **Status:** `loaded` (adapter + contract test) · `candidate` · `probe-only`.
> Maintained by the Data Engineering agent. Last sweep: **2026-07-14**.

## Tier 1 — in the Phase-0 pipeline

| Corpus | Size | Sensors | Actions | License class | Urban richness | Cost→batch | Status |
|---|---|---|---|---|---|---|---|
| **comma2k19** (`commaai/comma2k19`) | ~100 GB, 33 h, 10×8.7 GB chunks | front cam 20 fps + CAN + GNSS | **real CAN** (steer wheel, speed) | **public (MIT)** | low (highway commute only) | ~1–2 h (0 new code) | **loaded** (D-009) — public anchor |
| **Cosmos-Drive-Dreams** (`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`) | 5 843 RDS-HQ clips + **81 802** synth videos, 30 fps, 121-frame chunks | 7-cam rig (front_wide_120fov used), HD map, LiDAR, 4×4 poses | derived (ego 4×4 `vehicle_pose` → steer/accel) | **public (CC-BY-4.0)** | **high** (7 weathers: rain/snow/fog/night; intersections, VRUs) | ~2–3 h (**this run**) | **loaded** (D-014, this run) — public synthetic |
| **PhysicalAI-AV** (`nvidia/PhysicalAI-Autonomous-Vehicles`) | 1 727 h, 25 countries, 2 500+ cities | multi-cam + radar + lidar | egomotion (poses → yaw-rate/accel) | **gated/confidential** (NVIDIA AV licence, internal-dev-only, 12-mo) | **very high** | ~3–4 h | **loaded** (D-012) — `data:physicalai` tag, **no public claim** |
| **PhysicalAI-WorldModel-Synthetic-Scenarios** (`nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios`) | large; emergency / lanechange / nudging / pedestrian / weather_degradation | Omniverse multi-cam surround + per-cam VLM captions | scenario egomotion | **research/NC — verify card** | high (targeted long-tail) | ~3 h (mirror Cosmos loader) | **candidate** (D-014 backlog) — the H6/H15/D9 long-tail material |

## Tier 2 — real urban corpora (H4 arm-B / H7 scale-up / D8 OOD)

| Corpus | Size | Sensors | Actions | License class | Urban richness | Cost→batch | Status |
|---|---|---|---|---|---|---|---|
| **BDD100K** | 100k clips 40 s, ~1.8 TB | front cam + GPS/IMU | weak (GPS/IMU; no CAN) → **needs IDM (H7)** | research/NC | **very high** (diverse US urban, weather, night) | ~6–8 h (IDM pseudo-labels) | candidate — H7 pseudo-label target |
| **nuScenes** | 1 000 scenes 20 s, 2 cities | 6-cam + radar + lidar | ego pose | research/NC | high | ~4 h | probe-only (D8 OOD, never trained) |
| **nuPlan** | 1 500 h, 4 cities | cam + lidar + tracks | ego + agent tracks (planning) | research/NC | high | ~6 h | candidate (NAVSIM/planning link, Bench&Eval) |
| **Argoverse 2** | 1 000 h sensor + 250k forecasting | 7-cam + 2 lidar | ego + tracks | research (CC-BY-NC 4.0) | high | ~6 h | candidate |
| **Waymo Open** | 2 030 seg 20 s + motion | 5-cam + 5 lidar | ego + tracks | research/NC (Waymo licence) | high | ~6 h | candidate |
| **Zenseact ZOD** | Frames/Sequences/Drives, ~1 473 h drives | cam + lidar + radar + CAN | **real CAN** (EU) | research/NC | high (Nordic urban, night/winter) | ~5 h | candidate — real-CAN #2, strong for H4 |
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

## Sweep notes (2026-07-14)

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
