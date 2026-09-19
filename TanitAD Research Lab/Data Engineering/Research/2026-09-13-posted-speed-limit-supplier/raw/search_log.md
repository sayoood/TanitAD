# search log — 2026-09-13 posted-speed-limit-supplier

| # | query / probe | route | hits | used / result |
|---|---|---|---|---|
| S1 | `speed limit annotation autonomous driving dataset traffic sign map posted speed limit label` | WebSearch | 10 | TT100K (pl5–pl120 classes), GTSRB/GTSDB; patents on map-learned limits — context only |
| S2 | `Mapillary traffic sign dataset speed limit classes open license` | WebSearch | 9 | MTSD 100 k images / >300 classes (academic); **MVV `2508.02047`** → banked |
| S3 | `nuPlan map speed limit lane attribute` | WebSearch | 8 | lane speed limit used by nuPlan's speed-limit-compliance metric; `2403.04133` → banked (abstract-only) |
| S4 | `Argoverse 2 HD map speed limit field` | WebSearch | 7 | **EMPTY for a speed field** (probe 1) |
| S4b | AV2 `LaneSegment` schema | WebFetch argoverse.github.io/user-guide/api/hd_maps.html | 1 | **12 attributes, no speed limit** — EMPTY at a second, primary probe ⇒ absence stated |
| S5 | `OpenStreetMap maxspeed coverage percentage roads study` | WebSearch | 10 | ~12 % global, Sweden 84 % (RELAYED) |
| S6 | `OpenDRIVE xodr speed record lane speed element max unit` | WebSearch | 7 | `<type><speed max unit>` + lane-level speed (ASAM spec pages) |
| S7 | `Waymo Open Dataset map features speed limit mph lane` | WebSearch + WebFetch `map.proto` | 10 | `optional double speed_limit_mph = 1; // The speed limit for this lane.` (PUBLISHED-CODE) |
| S8 | `vision-based speed limit estimation implicit road context without signs deep learning dashcam` | WebSearch | 8 | patents (perception-based limit estimation); no primary paper with numbers — **EMPTY for a published vision-only implicit-limit benchmark** (probe 1) |
| S8b | ScienceDirect S0198971525001450 (road speed classes from OSM + Street View) | WebFetch | — | **HTTP 403** — unread; not cited as evidence |
| S9 | `NVIDIA PhysicalAI-Autonomous-Vehicles dataset 2026 update new features map speed limit` | WebSearch | 7 | no map/limit feature added (second probe on the published corpus: consistent with CLAUDE.md "no map") |
| S9b | HF card `nvidia/PhysicalAI-Autonomous-Vehicles-NCore` | WebFetch | 1 | ~1.1 k clips, calibration/egomotion/cuboids — **no map, no limit** |
| S10 | `nuPlan sensor data cameras speed_limit_mps lanes_polygons` | WebSearch | 10 | 8 cameras 10 Hz; polylines carry speed limit + type |
| S11 | `speed limit sign recognition dataset 2025 2026 arXiv benchmark "speed limit" VLM driving` | WebSearch | 7 | **TS-1M `2603.23034`** → banked (abstract-only); Drive-P2D, RoadSafe365 (scan only) |
| S12 | HF card `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` | WebFetch | 1 | **26.04 = 1,607 clips from PhysicalAI**; 26.01 = 918; licence internal AV dev, 12-month term; no map-production statement |
| S13 | HF tree API root of the NuRec repo (token from `Keys.txt`, read in place) | httpx | 6 entries | `clip_ratings_26.04.csv` found |
| S14 | `clip_ratings_26.04.csv` | httpx GET, HTTP 200, 88,411 B | 1,607 rows | header `clip_id, quality_score, time_of_day, ego_speed` → overlap census |
| S15 | local `map.xodr` | `…/2026-08-02-nurec-xodr-map/map.xodr` sha256 `7d26b5b1…` | 219 roads | census F3 |
| S16 | local clip-id sets | v7.2 train/eval jsonl.gz; `alpamayo_clip_ids.txt` | 4,572 / 147 / 4,729 | overlap 16 / 1 / 17 |
