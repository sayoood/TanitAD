# TanitDataSet v1 — AV Dataset Survey by SENSOR SUITE

**Author:** Data Engineering (deep-research session). **Date:** 2026-07-19. **Status:** research/planning only — no code, no pod, no git commit. **Scope:** external dataset survey through a **sensor-suite** lens (front cam / surround / LiDAR / radar / IMU / GPS) to inform **TanitDataSet v1**, a combined multi-source corpus for our camera-first, **action-conditioned** driving world model (our own training data is NVIDIA PhysicalAI-AV / Alpamayo).

> **Honesty caveat (engineer's read, not legal advice).** Licenses below were verified against official license pages / HF cards / dataset sites (URLs cited), 2026-07-19. Where a fact could not be confirmed it is marked **`verify`** rather than guessed. Because TanitDataSet v1 is intended to *ship the actual normalized episodes* (drop-in `[T,9,256,256]` cache, per `OWN_DATASET_PLAN.md`), the decisive axis is not "commercial use" but **"may we redistribute a normalized/augmented derivative"** — several datasets permit the former and forbid the latter. Anything that leaves the repo still needs Sayed / legal sign-off.

> **Companion internal docs:** `OWN_DATASET_PLAN.md` (license verdicts), `DATA_LAKE_ARCHITECTURE.md` (store/serve), `Research/DATASET_LANDSCAPE.md` (D-012 standing landscape). This survey **extends** them with the sensor lens and 2025-2026 releases, and issues three corrections (§6) — most importantly it surfaces **L2D (Yaak + HF LeRobot)**, which the internal landscape currently mis-attributes to comma.

---

## 0. TL;DR

- **The headline new find is `yaak-ai/L2D` ("Learning to Drive", Yaak + HuggingFace LeRobot).** **Apache-2.0** (commercial + redistribution, no ShareAlike virality), **6 surround HD cameras (360°)**, **real CAN continuous actions (steering / gas / brake)** + discrete (gear / turn-signal) + **cm-level RTK GPS + IMU**, **~5,000+ h / ~90+ TB / ~1M episodes**, 30 German cities, and an **expert (instructor) vs student (learner) policy split** — i.e. it *natively* carries the off-expert action-consequence signal we currently synthesize with CARLA. This is almost a bespoke fit for an action-conditioned camera-first world model, it is HF-native, and it is **absent from our internal landscape**. **Verify the current live episode count** (phased R1→R4 rollout) and run a privacy/anonymization check (German driving-school footage), but on paper this is the strongest single external addition available.
- **The permissive, genuinely-redistributable shortlist for v1** (can ship real episodes): **L2D (Apache-2.0) › comma2k19 (MIT, already loaded) › PandaSet (CC-BY-4.0) › ZOD (CC-BY-SA-4.0, copyleft) › Cosmos-Drive-Dreams (CC-BY-4.0, already loaded) › WorldModel-Synthetic (OpenMDW-1.1, pose-less) › Udacity (MIT)**, plus **CARLA** self-gen and a **ROVR** watch. This is the internal `OWN_DATASET_PLAN` set **+ L2D as the decisive real-urban+surround+action add**.
- **Biggest license gotchas** (look commercial-friendly, forbid a shippable derivative): **A2D2** (CC-BY-**ND** → NoDerivatives, so our normalization is illegal to redistribute despite "commercial OK"); **PhysicalAI-AV** ("commercial use" permitted for *internal AV dev* but **no derivatives / no hosting / no redistribution**); **Waymo** (non-commercial **and** "registered waymo.com/open recipients only"); **ONCE** (NC **and** explicitly bars redistribution of the data or modified versions); **Cityscapes** (NC, derivatives only as non-recoverable abstract representations). NC-but-negotiable: **nuScenes / nuPlan** (paid Motional commercial license path). SA-copyleft (owned but viral): **ZOD**.
- **Sensor-diversity note:** the richest *heterogeneous* sensor suites (LiDAR + radar + surround) sit almost entirely in the **non-redistributable** tier (nuScenes 5-radar ring, PhysicalAI-AV, MAN TruckScenes, OmniHD-Scenes). For a *shippable* v1 the diversity comes from **source heterogeneity** (real EU CAN via L2D/ZOD, real US highway CAN via comma2k19, real SF urban via PandaSet, synthetic weather/long-tail via Cosmos + WorldModel-Synth, off-expert via CARLA + L2D student policies), not from any single sensor-rich source.

---

## 1. Method, scoring axes, license legend

**Datasets surveyed (25):** nuScenes, Waymo Open, Argoverse 2, ZOD, PandaSet, nuPlan, KITTI, KITTI-360, A2D2, Lyft Level 5, ONCE, BDD100K, comma2k19, commaSteeringControl, commaVQ, Mapillary Vistas, Cityscapes, PhysicalAI-AV/Alpamayo, Cosmos-Drive-Dreams, WorldModel-Synthetic-Scenarios, Udacity, **L2D (Yaak/HF)**, MAN TruckScenes, OmniHD-Scenes, Open MARS — plus watch-list ROVR / AevaScenes / WayveScenes101 / CoVLA.

**Shortlist score = camera-richness × sensor-diversity × license-permissiveness × ego-motion.** For TanitDataSet v1 (a *redistributable* combined corpus) **license-permissiveness is a hard gate**, not a factor — an NC/ND/gated source scores ~0 for the *shippable* dataset regardless of how good its sensors are (it may still be an internal-only or negotiated add).

**License legend (redistribution of a normalized/augmented derivative):**
`YES` permissive (MIT / Apache-2.0 / CC-BY / OpenMDW) · `YES-SA` copyleft (CC-BY-SA — must relicense same, can't co-mingle into a closed file) · `NO-ND` NoDerivatives (verbatim re-host only) · `NO-NC` non-commercial · `NO-NC+reg` NC + registered-recipients-only · `NO-gated` confidential/internal-only · `NEGOTIABLE` paid commercial license may exist.

---

## 2. Master comparison tables

### Table A — Sensors & specs

| Dataset | Front cam / #cam / surround | LiDAR / radar / IMU / GPS | Camera res / FPS | Size | Geography + conditions | Annotations |
|---|---|---|---|---|---|---|
| **L2D** (Yaak/HF) | ✔ / **6 cam / 360° surround** (+ map view) | –/–/ IMU✔ / **RTK GPS✔ (cm)** | 1080×1920 / 10 Hz | **~5,000+ h / ~90+ TB / ~1M episodes** (phased; `verify` live count) | 30 German cities; weather/lighting/road-type labeled; day/night; **expert vs student policies** | state (speed/heading/GPS/accel/lane/road/weather/waypoints) + NL instructions; **no boxes** |
| **comma2k19** | ✔ / 1 front / no | –/–/ **9-axis IMU✔ / u-blox GNSS✔** | ~1164×874 `verify` / 20 Hz | 2,019 seg×1 min = **33 h / ~100 GB** | CA-280 highway (SF↔SJ); day/clear — **low diversity** | fused poses; parsed CAN (steer, wheel-speed, radar) — no perception labels |
| **PandaSet** | ✔ / 6 cam / ~360° | **2 LiDAR** (Pandar64 mech + PandarGT solid-state front) / – / IMU✔ / GPS✔ | 1920×1080 / 10 Hz | 103 scenes×8 s (~8,240 LiDAR frames, ~48k imgs) | Silicon Valley (SF/Palo Alto/San Mateo); day | 3D cuboids (28 cls) + point semseg (37 cls) |
| **ZOD** (Zenseact) | ✔ / **1 front** (8 MP) / no | **3 LiDAR** (VLS-128 + 2×VLP-16) / **radar✔** (added Jan-2025) / IMU✔ / GNSS✔ (OxTS RT3000) | 3848×2168 (8 MP), 120° / ~10 Hz | **100k Frames + 1,473 Seq×20 s + 29 Drives**; ~2 yr | **14 EU countries**; day/night, seasons, rain/snow | 3D+2D boxes (to 245 m), lanes, signs, road-condition |
| **Cosmos-Drive-Dreams** (NVIDIA, synthetic) | ✔ / 7-cam rig (front-wide 120° used) / surround | LiDAR✔ / – / – / – (sim) | 30 fps | 5,843 RDS-HQ clips + **81,802 synth videos** | synthetic; **7 weathers** (rain/snow/fog/night), VRUs, intersections | HD map, 4×4 ego poses |
| **WorldModel-Synthetic** (NVIDIA, synthetic) | ✔ / 7-cam @24 fps / surround | –/–/–/– (sim) | 4K / 24 fps | **264k clips / ~1,467 h / 8.3 TB** | synthetic long-tail: cut-in, ped, emergency, weather-deg | per-cam Qwen2.5 captions + {weather, ToD, surface, region}; **POSE-LESS** |
| **Udacity** CH2/CH3 | ✔ / front + L + R / partial | CH3 LiDAR✔ / – / IMU✔ / GPS✔ | ~ / — | ~10 h | Mountain View CA; day | real steer/throttle/brake/speed; some boxes |
| **PhysicalAI-AV / Alpamayo** (NVIDIA, **our data**) | ✔ / **7 cam** (fw120 + ftele30 + cross L/R 120 + rear L/R 70 + rtele30) / ~360° | **1 top-360° LiDAR** (298k clips) / **up to 10 radar** (161k clips) / — / — | 1080p / 30 fps | **1,700 h / 306,152 clips×20 s / 133 TB**; 25 countries, 2,500+ cities | very high; global; all conditions | egomotion, calibration, machine labels (rt + offline) |
| **A2D2** (Audi) | ✔ / 6 cam / 360° | **5×VLP-16 LiDAR** / – / IMU✔ / GPS✔ + **CAN** | 1928×1208 (2.3 MP) / ~30 Hz `verify` | 41,277 semseg frames; 12,497 w/ 3D boxes; 392k unlabeled | S. Germany (Ingolstadt/Munich); cloud/rain/sun | semseg (38 cls), instance, 3D boxes (front FOV), CAN |
| **nuScenes** | ✔ / **6 cam / 360°** | 1 LiDAR (32-beam) / **5 radar (360° ring)** / IMU✔ / GPS✔ | 1600×900 / 12 Hz | 1,000 scenes×20 s = **5.5 h annot** (~15 h raw); 40k keyframes | Boston + Singapore; **rain 19.4%, night 11.6%** | 1.4M 3D boxes (23 cls), 11-layer HD maps, lidarseg, panoptic, nuImages |
| **nuPlan** | ✔ / **8 cam / surround** | 5 LiDAR / – / IMU✔ / GNSS✔ | 10 Hz (res ~1920×1080 `verify`) | **~1,282 h logs** (~16 TB / ~128 h raw sensor = ~10%) | Vegas / Boston / Pittsburgh / Singapore | auto object tracks, 12-layer HD maps, traffic-light status, 73 scenario types |
| **Waymo Open** | ✔ / **5 cam** Perception (fwd+side, **no rear**); **8 cam 360°** in Motion/E2E | 5 LiDAR / **no radar** / IMU✔ (pose) / GPS✔ (fused) | 1920×1280 / 10 Hz | Perc 2,030 seg×20 s (390k frames); **Motion 103,354 seg / 574 h**; **E2E 5,000 seg** | 6 US cities (SF, Phoenix, LA, Detroit, Seattle, MtView) | 12.6M 3D + 11.8M 2D boxes, 3D semseg, panoptic, road-graph maps (Motion) |
| **Argoverse 2** | ✔ / **9 cam** (7 ring 360° + 2 front stereo) | 2×32-beam LiDAR (→20 Hz) / no radar / IMU/GNSS fused | 2048×1550 / 20 fps | Sensor 1,000 seq (~4.2 h); **Motion 250k scenarios×11 s**; LiDAR 20k seq | 6 US cities (Austin, Detroit, Miami, Pgh, Palo Alto, DC) | 3D cuboids (30 cls), per-log HD vector maps, map-change |
| **KITTI** | ✔ / 4 cam (2 gray+2 color, fwd stereo) / no | 1×HDL-64E / no radar / IMU✔ / GPS✔ (OXTS) | ~1382×512 / 10 Hz | ~6 h / ~50 seq (Karlsruhe) | Karlsruhe DE; day/clear | 3D+2D boxes, tracking, stereo/flow/depth, odometry |
| **KITTI-360** | ✔ / 4 cam (2 front stereo + 2 side fisheye) / 360° | HDL-64E + SICK 2D / no radar / OXTS✔ | 1408×376 / 1400×1400 / ~10 Hz | 320k imgs / 100k scans / 73.7 km / 9 seq | Karlsruhe DE suburbs; day | dense 2D+3D semseg (19 cls), 3D primitives, geoloc poses |
| **ONCE** (Huawei) | ✔ / 7 cam / 360° | 1×40-beam LiDAR / no radar / per-frame pose | ~1920×1020 `verify` / ~10 Hz `verify` | **1M LiDAR frames + 7M imgs / 144 h**; ~16k annot scenes / 417k boxes | China multi-city (~200 km²); day/night, sun/rain | 3D boxes (5 cls); ONCE-3DLanes extension |
| **Lyft Level 5** | ✔ / **7 cam 360°** (Perception) | **3 LiDAR** (Perception) / – / IMU✔ / GPS✔ | 1224×1024 & 2048×864 / 10 Hz | Perc 55k frames; **Pred 170k scenes×25 s = 1,000+ h** | Palo Alto CA (single route) — low geo diversity | 3D boxes, HD maps; agent+ego trajectories (Pred) |
| **BDD100K** | ✔ / 1 front dashcam / no | –/–/ IMU✔ (phone) / GPS✔ (phone) | 1280×720 / 30 fps (40 s clips) | **100k videos (~1,100 h)**, 100k annot key-frames | US (NY, SF Bay); **6 weather / 6 scene / 3 ToD** — high diversity | 2D boxes (10 cls), lanes, drivable area, semseg (10k), MOT/MOTS, pose |
| **commaSteeringControl** | ✘ **CAN-only, no camera** | –/–/–/– | N/A (tabular) | **~12,500 h / 45.5 GB** | global openpilot fleet `verify` | steer torque, vEgo, aEgo, lateral dynamics |
| **commaVQ** | ✔ / 1 front (VQ-tokenized) / no | –/–/–/– | 20 fps (tokens) | 100k seg (~1,600 h compressed) | openpilot fleet | none (video tokens for GPT-style prediction) |
| **Mapillary Vistas** | crowd stills (no fixed rig) | – / – / per-image GPS+compass | 2–22 MP stills | 25k annotated images (platform: 2B+ photos) | **global, 6 continents** — highest geo diversity | dense semseg+instance (v2.0: 124 cls) |
| **Cityscapes** | ✔ / front **stereo pair** / no | –/–/ odometry✔ / GPS✔ | 1024×2048 / ~17 fps snippets | 5k fine + 20k coarse annot images | 50 cities (DE+); **day, good weather only** | semseg (30 cls), instance, panoptic, disparity |
| **MAN TruckScenes** (2024) | ✔ / 4 cam / 360° | **6 LiDAR / 6 radar (360° 4D)** / 2 IMU / GNSS✔ | `verify` / ~ | 747 scenes×20 s | Germany `verify`; 3 seasons, multi-weather; **trucking** | 3D boxes (27 cls, 230 m), tracks, 34 scene tags |
| **OmniHD-Scenes** (TPAMI-2026) | ✔ / 6 cam / 360° | **128-beam LiDAR / 6× 4D radar** / IMU `verify` / GPS `verify` | `verify` | 1,501 clips×30 s (~450k frames) | China | 514k 3D boxes + static semseg |
| **Open MARS** (CVPR-2024) | ✔ / multi RGB (narrow + fisheye) / partial | **128-ch LiDAR** / – / IMU✔ / GPS✔ | 10 Hz | multi-traversal fleet corpus (May Mobility) | Ann Arbor MI; **multi-traversal** same-location, varied conditions | 3D + multi-traversal 3D reconstruction targets |

### Table B — Ego-motion, license, access, redistributability

| Dataset | Ego-motion (for action-conditioning) | License | Comm. use | **Redist. derivative?** | Access | **v1 (shippable)?** |
|---|---|---|---|---|---|---|
| **L2D** (Yaak/HF) | **Best-in-class:** real CAN **steering + gas + brake** + gear/turn-signal, **cm RTK GPS**, IMU | **Apache-2.0** | ✔ | **YES** | HF, **non-gated**, LeRobot/parquet | **✔ TOP** |
| **comma2k19** | **Excellent:** real CAN steer + wheel-speed, raw GNSS, 9-ax IMU, fused poses | **MIT** | ✔ | **YES** | GitHub / Torrents / **HF** | **✔ (loaded)** |
| **PandaSet** | GPS/IMU world pose (no CAN → steer via curvature) | **CC-BY-4.0** + Dataset Terms | ✔ | **YES** | official form; **HF `georghess/pandaset`** mirror | **✔** |
| **ZOD** | OxTS **RTK 100 Hz, 0.01 m** poses/vel/accel (motion, not steer) | **CC-BY-SA-4.0** (+privacy, no-military) | ✔ | **YES-SA** (viral copyleft) | official site + **HF `Zenseact/ZOD`**, non-gated | **✔ (separate SA shard)** |
| **Cosmos-Drive-Dreams** | 4×4 `vehicle_pose` → steer/accel (derived, synthetic) | **CC-BY-4.0** | ✔ | **YES** | **HF** (nvidia/…Cosmos-Drive-Dreams) | **✔ (loaded)** |
| **WorldModel-Synthetic** | **POSE-LESS** (confirmed) → needs IDM / video-only | **OpenMDW-1.1** | ✔ | **YES** (retain notice) | **HF**, ungated | **✔ (pose-less caveat)** |
| **Udacity** | real steer/throttle/brake/speed (weak pose) | **MIT** (datasets/ dir) | ✔ | **YES** | GitHub / torrents | **✔ (small)** |
| **ROVR Open** (2025) | **Not documented** (`verify`) | dual: CC-BY-NC-SA-4.0 / **CC-BY-4.0 commercial** | ✔ (comm tier) | **YES** (comm tier) | HF + **signed agreement** | **watch (ego-motion gap)** |
| **PhysicalAI-AV / Alpamayo** | egomotion + calibration + machine labels (rt+offline) | **NVIDIA AV Dataset License** | ✔ *internal AV dev* | **NO-gated** ("no derivative works… host… distribute") | **HF gated** | ✘ (our data; recipe-only) |
| **A2D2** | **full CAN:** steering/brake/throttle + IMU/GPS | **CC-BY-ND-4.0** | ✔ | **NO-ND** (verbatim re-host only) | site + **AWS Open Data** | ✘ (ND blocks our normalize) |
| **nuScenes** | **best sensor-rich:** CAN-bus expansion (steering, throttle, brake, wheel-speed) + fused pose | CC-BY-NC-SA-4.0 | NC (comm license path) | **NO-NC / NEGOTIABLE** | account/email; AWS | ✘ (unless Motional comm license) |
| **nuPlan** | **1,282 h ego trajectories** (GNSS/IMU); raw steer `verify` | CC-BY-NC-SA (Motional terms) | NC (comm license path) | **NO-NC / NEGOTIABLE** | **open S3 no account** | ✘ (unless comm license) |
| **Waymo Open** | pose + velocity + accel; **E2E adds 4 s trajectory + routing**; no raw CAN | Waymo NC Agreement (Mar-2025) | **NO** | **NO-NC+reg** (registered recipients only) | registration → GCS | ✘ |
| **Argoverse 2** | 6-DOF ego poses; velocity derivable; no raw CAN | **CC-BY-NC-SA-4.0** | NC | **NO-NC** (SA) | **open S3 no account** + HF subsets | ✘ |
| **KITTI** | OXTS GPS/IMU poses | CC-BY-NC-SA-3.0 | NC | **NO-NC** | cvlibs (email for raw) | ✘ |
| **KITTI-360** | high-accuracy geoloc poses | CC-BY-NC-SA-3.0 | NC | **NO-NC** | cvlibs (email) | ✘ |
| **ONCE** | relative pose only (no GPS/IMU/CAN) | CC-BY-NC-SA-4.0 + terms | NC | **NO-NC** (explicit no-redistribution) | GDrive / Baidu | ✘ |
| **Lyft Level 5** | AV-grade pose + velocity + agent/ego trajectories | ~CC-BY-NC-SA `verify` | NC | **NO-NC** | **host RETIRED** (l5kit archived Oct-2025); Kaggle/Archive mirrors | ✘ |
| **BDD100K** | phone-grade GPS/IMU (no CAN, no precise pose) | custom BAIR/UC-Berkeley | members-only | **research/NFP redist OK; comm = BAIR members** | register at bdd-data.berkeley.edu | ✘ (comm) / research-only |
| **commaSteeringControl** | real CAN steer + speed + lateral (no vision) | **MIT** | ✔ | **YES** | **HF**, non-gated | aux (no camera) |
| **commaVQ** | none (video tokens) | **MIT** | ✔ | **YES** | **HF** | aux (no ego-motion) |
| **Mapillary Vistas** | none usable (per-image GPS only) | Vistas CC-BY-NC-SA; **platform imagery CC-BY-SA-4.0** | mixed | **NO-NC (Vistas) / YES-SA (platform)** | account / API | ✘ (no ego-motion) |
| **Cityscapes** | odometry + GPS (yaw-rate, speed); no CAN | custom NC | **NO** | **NO-NC** (abstract representations only) | register | ✘ |
| **MAN TruckScenes** | GNSS + IMU (trucking) | **CC-BY-NC-SA-4.0** | NC | **NO-NC** (SA) | **AWS S3 `--no-sign-request`** | ✘ (NC; truck domain) |
| **OmniHD-Scenes** | `verify` | **research DUA** (email 2077ai) `verify` | `verify` (likely NC) | **NO / gated** `verify` | Alibaba OSS after signed DUA | ✘ (gated DUA) |
| **Open MARS** | ego-frame poses + multi-traversal trajectories | `verify` (HF `ai4ce-drive/MARS`) | `verify` | `verify` | **HF** | watch (verify license) |

---

## 3. Per-dataset notes (grouped by redistributability)

### 3.1 Permissive & redistributable — the TanitDataSet v1 candidate pool

- **L2D — `yaak-ai/L2D` (Yaak + HF LeRobot).** *The find of this survey.* Apache-2.0, 6 surround cams + full CAN continuous actions (steer/gas/brake) + discrete (gear/turn-signal) + cm-RTK GPS + IMU, ~5,000 h across 60 driving-school EVs in 30 German cities over 3 years, **expert-instructor vs student-learner** policies (native off-expert coverage), LeRobot v2.1/v3.0 parquet, HF non-gated. Matches our action contract directly (steer + accel from CAN) and adds surround + real off-expert. Weaknesses: Germany-only (geo-concentrated), **no LiDAR/radar** (camera+CAN only — fine for a camera-first WM), phased rollout so **`verify` the current live episode count / TB**, and it is real EU footage → **face/plate anonymization** required before re-host (same GDPR axis as ZOD/PandaSet). Apache-on-data carries the same "license text says software" caveat as MIT — treat as a full grant, get a one-line legal nod. Sources: [HF card](https://huggingface.co/datasets/yaak-ai/L2D), [L2D-v3](https://huggingface.co/datasets/yaak-ai/L2D-v3), [blog](https://huggingface.co/blog/lerobot-goes-to-driving-school).
- **comma2k19 — MIT.** Already our loaded anchor (D-009): 33 h CA-280 highway, real CAN steer + wheel-speed, raw u-blox GNSS, 9-axis IMU, fused poses. Cleanest inverse-dynamics regime; low scene diversity (highway/day). [LICENSE](https://github.com/commaai/comma2k19/blob/master/LICENSE), [HF](https://huggingface.co/datasets/commaai/comma2k19).
- **PandaSet — CC-BY-4.0 (+ Dataset Terms).** 6 cams + 2 LiDAR (mech + **solid-state front**) + GPS/IMU, 103×8 s SF-urban, 3D cuboids + point semseg. "First AV set for commercial use." Small; internal geometry blocker noted (front fx≈1970 not square-croppable to f_eff=266 → needs D-016 R1 pad-crop+undistort). Use HF mirror `georghess/pandaset` (Scale de-emphasized the official host; CC is irrevocable). [pandaset.org](https://pandaset.org/), [HF mirror](https://huggingface.co/datasets/georghess/pandaset).
- **ZOD — CC-BY-SA-4.0.** Confirmed commercial-OK + derivatives-OK **but ShareAlike** (viral; keep in a separate `tanitad-own-zod` shard, never co-mingled into a proprietary file). 1 front 8 MP fisheye + **3 LiDAR + radar (new Jan-2025) + OxTS RTK 0.01 m** ego-motion, **14 EU countries**, day/night/seasons — the best owned real-urban replacement for PhysicalAI's richness. Fisheye → `ftheta_crop_resize`. [license](https://zod.zenseact.com/license/), [HF `Zenseact/ZOD`](https://huggingface.co/datasets/Zenseact/ZOD), [radar update](https://zenseact.com/news/).
- **Cosmos-Drive-Dreams — CC-BY-4.0 (NVIDIA, synthetic).** Already loaded (D-014): 7-cam rig, 4×4 poses → steer/accel, HD map, LiDAR, **7 weathers + VRUs**. Synthetic pixels (domain gap) but publicly claimable rich AV. [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams).
- **WorldModel-Synthetic-Scenarios — OpenMDW-1.1 (NVIDIA, synthetic).** 264k clips / 1,467 h safety-critical long-tail (cut-in, ped, emergency, weather-deg), 7-cam, per-cam captions + metadata. **Confirmed POSE-LESS** → pixels need IDM (H7) or video-only; captions/metadata usable now. Massive owned scale. [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios), [OpenMDW](https://openmdw.ai/license/1-1/).
- **Udacity CH2/CH3 — MIT (datasets/ dir; repo code is GPLv3 — keep separate).** Front + L + R cams, real steer/throttle/brake/speed, ~10 h, CH3 adds LiDAR. Small, clean augment. [LICENSE](https://github.com/udacity/self-driving-car/blob/master/datasets/LICENSE.md).
- **commaSteeringControl / commaVQ — MIT (aux).** commaSteeringControl = ~12,500 h **CAN-only** steer+speed (no camera) → excellent for an IDM/inverse-dynamics prior; commaVQ = 100k VQ-tokenized front videos (no ego-motion) → representation pretraining. Both HF, permissive. [commaSteeringControl](https://huggingface.co/datasets/commaai/commaSteeringControl), [commaVQ](https://huggingface.co/datasets/commaai/commavq).
- **ROVR Open Dataset (2025) — dual CC-BY-NC-SA-4.0 / CC-BY-4.0 (commercial).** ADAS cameras + LiDAR (Omni-Quad variant = 4×360° LiDAR), global 50–90 countries, day/night/rain, first batch 1,363×30 s, currently monocular-depth GT (det/seg planned). Genuinely permissive commercial tier + global diversity, **but ego-motion is not documented and the camera config (mono vs surround) is unclear, and it needs a signed agreement** → **watch, verify ego-motion before ingest.** [GitHub](https://github.com/rovr-network/ROVR-Open-Dataset).

### 3.2 Commercial-use ≠ redistributable (the traps)

- **A2D2 (Audi) — CC-BY-ND-4.0.** *Confirmed.* Commercial use allowed and it has the **richest native bus signals of any open set** (steering/brake/throttle + IMU/GPS + 6-cam 360° + 5 LiDAR) — but **NoDerivatives** means our normalized/augmented `[T,9,256,256]` cache is an adaptation we may **not** redistribute. Verbatim re-host only. A bespoke Audi license is the only unblock. [site](https://a2d2-dataset.github.io/), [AWS](https://registry.opendata.aws/aev-a2d2/).
- **PhysicalAI-AV / Alpamayo (NVIDIA) — NVIDIA AV Dataset License.** *This is our own training data.* The HF card confirms: use permitted "for internal development of autonomous vehicles" (so "commercial use" is technically true) **but "You may not create derivative works… sell, rent, sublicense, transfer, distribute, embed, or host the Dataset."** → firewalled, recipe-only, never shipped (D-002). 7 cam + 360° LiDAR + up to 10 radar, 1,700 h / 306k clips / 25 countries / 133 TB — the richest real corpus we have, unusable as a shared asset. Alpamayo-1 (Jan-2026) is the 10B VLA trained on it. [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).

### 3.3 Non-commercial research benchmarks (surround/sensor-rich; reference or negotiated only)

- **nuScenes — CC-BY-NC-SA-4.0 (commercial license via Motional).** The reference multimodal set: 6-cam 360° + **5-radar ring (unique)** + LiDAR + IMU/GPS, **plus a CAN-bus expansion with real steering/throttle/brake/wheel-speed** — arguably the best *sensor-diverse* ego-motion of any surveyed set, but NC. Boston+Singapore, rain 19.4%/night 11.6%. Best D8-OOD probe (never trained). [terms](https://www.nuscenes.org/terms-of-use), [commercial](https://www.nuscenes.org/terms-of-use-commercial), [CAN docs](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md).
- **nuPlan — CC-BY-NC-SA (Motional; commercial path).** Largest real ego corpus (~1,282 h trajectories), 8-cam + 5-LiDAR (raw sensor ~10%), 4 cities, planning/closed-loop focus. Open S3 no account. NC. [paper](https://arxiv.org/html/2403.04133v1), [AWS](https://registry.opendata.aws/motional-nuplan/).
- **Waymo Open — Waymo NC Agreement (Mar-2025).** *Double-blocked for us:* non-commercial **and** redistribution only to registered waymo.com/open recipients. Perception 5-cam (no rear); **Motion/E2E use 8-cam 360°**; the **E2E Driving Dataset (5,000 seg, 2025)** adds trajectory + routing — relevant to action-conditioning but unusable to ship. [terms](https://waymo.com/open/terms/).
- **Argoverse 2 — CC-BY-NC-SA-4.0.** 9 cam (7 ring 360° + 2 stereo) + 2 LiDAR, 6 US cities, 250k motion-forecasting scenarios, open S3 no account, 6-DOF poses. NC+SA. (Internal landscape tags this "CC-BY-NC" — **correct to NC-SA**, §6.) [av2](https://www.argoverse.org/av2.html).
- **KITTI / KITTI-360 — CC-BY-NC-SA-3.0.** Legacy Karlsruhe benchmarks; excellent OXTS poses, low coverage (day/clear), NC. KITTI-360 adds side fisheyes → 360° + dense semseg. [KITTI](https://www.cvlibs.net/datasets/kitti/), [360](https://www.cvlibs.net/datasets/kitti-360/).
- **ONCE (Huawei) — CC-BY-NC-SA-4.0 + terms.** 1M LiDAR frames / 144 h, 7-cam 360°, China day/night/rain, **but only relative pose** (weak for action-conditioning) and terms **explicitly bar redistribution** of the data or modified versions. Double-blocked. [terms](https://once-for-auto-driving.github.io/terms_of_use.html).
- **Lyft Level 5 — ~CC-BY-NC-SA (`verify`).** 7-cam 360° + 3 LiDAR (Perception); 1,000+ h agent/ego trajectories (Prediction). **Access gotcha: l5kit archived Oct-2025, official portals down** — only Kaggle/Internet-Archive mirrors survive. NC + practically orphaned. [l5kit (archived)](https://github.com/woven-planet/l5kit).

### 3.4 Camera-first / segmentation / crowd (mixed)

- **BDD100K — custom BAIR/UC-Berkeley.** 100k dashcam videos (~1,100 h), highest US condition diversity (6 weather / 6 scene / 3 ToD), rich 2D labels — but **1 front cam, phone-grade GPS/IMU (no CAN, no precise pose)**. Free to modify+redistribute for research/not-for-profit; **commercial = BAIR Commons members only**. Good H7 pseudo-label (IDM) target, not a v1 ship. [license](https://doc.bdd100k.com/license.html).
- **Cityscapes — custom NC.** Front stereo, 50 cities (DE+), semseg gold standard, **day/good-weather only**, odometry+GPS ego. NC + no-recoverable-redistribution. [license](https://www.cityscapes-dataset.com/license/).
- **Mapillary — Vistas CC-BY-NC-SA (NC) / platform imagery CC-BY-SA-4.0.** Global crowd stills, dense semseg (124 cls), **no vehicle ego-motion** → not a world-model/control corpus. Platform imagery is CC-BY-SA (commercial-OK) but still lacks ego-motion. [Vistas](https://www.mapillary.com/dataset/vistas).

### 3.5 Notable 2025-2026 multi-sensor releases (mostly NC / gated; sensor reference)

- **MAN TruckScenes (NeurIPS-2024) — CC-BY-NC-SA-4.0.** First large autonomous-**trucking** set: 4 cam + **6 LiDAR + 6 radar (360° 4D, largest annotated radar set)** + 2 IMU + GNSS, 747×20 s, 3 seasons/weather, AWS `--no-sign-request`. NC + truck viewpoint (mount height differs from cars) → reference only. [AWS](https://registry.opendata.aws/man-truckscenes/), [paper](https://arxiv.org/abs/2407.07462).
- **OmniHD-Scenes (TPAMI-2026) — research DUA (`verify`).** 6 cam + 128-beam LiDAR + **6× 4D imaging radar**, 1,501×30 s, 514k 3D boxes, China. Gated behind an emailed Data-Use-Agreement (Alibaba OSS) → not shippable. [GitHub](https://github.com/TJRadarLab/OmniHD-Scenes), [arXiv](https://arxiv.org/abs/2412.10734).
- **Open MARS (CVPR-2024) — `verify` (HF `ai4ce-drive/MARS`).** May-Mobility fleet, Ann Arbor: 128-ch LiDAR + multi-RGB (narrow+fisheye) + IMU + GPS, **multi-traversal** (same locations, varied conditions/lighting/weather) — a strong signal for world-model consistency + memory. On HF; **verify license before considering ingest.** [GitHub](https://github.com/ai4ce/MARS), [HF](https://huggingface.co/datasets/ai4ce-drive/MARS).
- **AevaScenes (Sept-2025) — academic/NC.** FMCW **4D LiDAR (per-point velocity)** + camera. Novel modality, NC-only. Watch. [Aeva](https://www.aeva.com/press/).
- **WayveScenes101 — research (`verify`).** Multi-cam surround, 101 seq / 101k images (US+UK), COLMAP poses — a small NVS benchmark, not a scale corpus. [Wayve].
- **CoVLA (WACV-2025) — proprietary Turing NC + gated.** Forward cam + CAN + GNSS + IMU + **language captions**, ~10k clips / ~80 h, Tokyo — technically an ideal VLA fit, but the Turing license is "research preview, non-commercial only" and gated. Confirms the internal `nc-research` tag. Internal-only VLA view at best. [HF](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset).

---

## 4. License gotchas (the honest section)

**Forbid a shippable derivative (redistribution of our normalized cache) — hard NO:**
1. **A2D2 — CC-BY-ND.** Commercial-friendly *trap*: NoDerivatives forbids distributing our adapted version. Richest CAN signals, wasted for a shared asset.
2. **PhysicalAI-AV — NVIDIA AV License.** "Commercial use" ✔ but explicit "no derivative works / no host / no distribute." Our own data; stays recipe-only, firewalled.
3. **Waymo — NC + registered-recipients-only.** Cannot re-host publicly *or* privately to unregistered parties.
4. **ONCE — NC + explicit no-redistribution** of data or modified versions.
5. **Cityscapes — NC**, derivatives only as non-recoverable "abstract representations."
6. **Mapillary Vistas — NC** (the *platform* CC-BY-SA imagery is commercial-OK but has no ego-motion).

**Non-commercial but negotiable / research-internal:** nuScenes, nuPlan (paid Motional commercial license may grant redistribution — *confirm it does*), Argoverse 2 (NC-SA), KITTI/KITTI-360 (NC-SA), BDD100K (commercial = BAIR members), Lyft (NC + orphaned host), MAN TruckScenes / OmniHD-Scenes / CoVLA (NC / gated).

**Owned but copyleft (ShareAlike — viral):** **ZOD** and **Mapillary platform imagery** are CC-BY-SA — usable and commercial, but any file they are co-mingled into inherits SA (cannot be proprietary/closed). Keep ZOD in its own shard.

**"License text says *software*" caveat:** MIT (comma2k19, comma10k, Udacity, commaSteeringControl/VQ) and **Apache-2.0 (L2D)** are written for code; the intent for the data is a full grant, but get a one-line legal blessing to treat MIT/Apache *data* as fully redistributable.

**Privacy is a separate axis from copyright:** redistributing real EU footage (**L2D, ZOD, PandaSet**) triggers GDPR/personality obligations regardless of license → plan face/plate anonymization (ZOD ships de-identified with a notice that must travel; L2D/PandaSet need a `--blur` pass).

---

## 5. SHORTLIST — ranked for TanitDataSet v1

Ranked by **camera-richness × sensor-diversity × license-permissiveness × ego-motion**, gated on **redistributability** (v1 must ship real episodes). One-line rationale each.

| # | Source | License | Why (camera × sensor × license × ego-motion) |
|---|---|---|---|
| **1** | **L2D (Yaak/HF)** | Apache-2.0 | **6 surround cams + real CAN steer/gas/brake + RTK GPS/IMU + expert/student policies, ~5,000 h, HF-native** — the only permissive source that is simultaneously surround, real-action, huge, and off-expert. Verify live size + anonymize. |
| **2** | **comma2k19** | MIT | Cleanest real-CAN inverse-dynamics anchor (steer+speed+fused pose); already loaded; narrow (highway/day) but the action-fidelity backbone. |
| **3** | **ZOD** | CC-BY-SA-4.0 | Best owned **real-urban** richness — 14 EU countries, night/seasons, 3 LiDAR + radar + RTK pose; SA-copyleft so ship as a separate shard. |
| **4** | **PandaSet** | CC-BY-4.0 | Clean permissive real SF-urban, 6-cam + solid-state LiDAR + GPS/IMU; small; resolve the D-016 R1 geometry blocker. |
| **5** | **Cosmos-Drive-Dreams** | CC-BY-4.0 | Owned synthetic weather/night/VRU with 4×4 poses→actions; already loaded; publicly claimable long-tail. |
| **6** | **WorldModel-Synthetic** | OpenMDW-1.1 | Owned 264k-clip safety-critical long-tail at massive scale; **pose-less** → IDM/video-only, so a representation/long-tail booster not an action anchor. |
| **7** | **Udacity CH2/CH3** | MIT | Small clean real steer/throttle/speed augment; cheap; keep separate from GPLv3 repo code. |
| **8** | **CARLA (self-gen)** | MIT + CC-BY | Owned closed-loop **off-expert / counterfactual** action-consequence coverage real logs lack (already the D-014 arm). |
| *watch* | **ROVR Open (2025)** | CC-BY-4.0 (comm tier) | Permissive + global diversity, but **ego-motion undocumented + camera config unclear** — verify before ingest. |
| *watch* | **Open MARS** | `verify` | Multi-traversal same-location signal is uniquely valuable for WM consistency — **verify license** first. |
| *aux* | **commaSteeringControl** | MIT | 12,500 h CAN-only steer+speed → strong IDM/inverse-dynamics prior for pose-less sources (WorldModel-Synth, YouTube). |

**Fusion recommendation (extends `OWN_DATASET_PLAN` §6.2 with L2D):** real-CAN anchor **comma2k19** + real-surround-action **L2D** + real-EU-urban **ZOD (SA shard)** + real-SF **PandaSet/Udacity** + synthetic long-tail **Cosmos-DD + WorldModel-Synth** + off-expert **CARLA**. L2D is the single change that most improves v1 — it adds surround cameras, real off-expert actions, and EU-urban scale in one permissive, HF-native source, reducing reliance on the firewalled PhysicalAI-AV.

---

## 6. Corrections to the internal landscape (`DATASET_LANDSCAPE.md`)

1. **"comma L2D" is a misattribution.** `L2D` is **Yaak (Berlin) + HF LeRobot** (`yaak-ai/L2D`), **Apache-2.0, ~90 TB / 5,000 h, 6 surround cams, full CAN actions, expert/student** — not comma's, not "GB-scale front cam." It is the strongest permissive external add and should be promoted to a Tier-1 candidate. comma's actual sets are comma2k19, commaSteeringControl, commaVQ, commabody.
2. **Argoverse 2 license** is **CC-BY-NC-SA-4.0** (NC **+ ShareAlike**), not the "CC-BY-NC 4.0" tagged in the landscape — the SA clause matters for any derivative.
3. **ZOD gained radar (Jan-2025)** — it is now cam + **3 LiDAR + radar** + OxTS RTK, strengthening its sensor-diversity score; CC-BY-SA-4.0 confirmed (4th independent verification).
4. **New multi-sensor releases to log** (none shippable, sensor reference): MAN TruckScenes (NC-SA, 4D radar, trucking), OmniHD-Scenes (gated DUA, 6× 4D radar), Open MARS (verify, multi-traversal), AevaScenes (NC, FMCW 4D LiDAR), WayveScenes101 (verify), ROVR (CC-BY-4.0 comm tier, ego-motion `verify`).

---

### Provenance
External licenses/specs verified 2026-07-19 against the URLs cited inline (official license pages, HF dataset cards, dataset sites, papers, AWS Open Data registry). Internal grounding: `OWN_DATASET_PLAN.md`, `DATA_LAKE_ARCHITECTURE.md`, `Research/DATASET_LANDSCAPE.md`, contract `stack/tanitad/data/_contract.py`, decisions D-002/D-009/D-010/D-014/D-016/D-017/D-022. `verify` tags mark facts not confirmed to primary source in this pass (notably: L2D current live episode count; comma2k19 exact pixel resolution; ONCE camera res / LiDAR rate; A2D2 per-camera FPS; nuPlan camera res; Lyft exact license name + live mirror; ROVR/Open MARS/OmniHD-Scenes licenses & ego-motion). This is an engineer's read to route a decision, not legal advice; anything leaving the repo needs Sayed / legal sign-off.
