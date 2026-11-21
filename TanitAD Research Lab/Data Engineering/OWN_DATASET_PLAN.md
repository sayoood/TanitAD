# TanitAD-Own Dataset — Plan & Licensing Verdict

**Author:** Data Engineering (research/planning session). **Date:** 2026-07-13. **Status:** plan v1 — for Sayed
review. **Scope:** research + design only; no change to the running contract, no code committed to main, no pod
touched. Pipeline-code sketches (if built) live in a separate uncommitted dir.

> ⚠️ **Engineer's license read to route a decision — not legal advice.** Every source below was checked against its
> actual license page (URLs cited). Anything that leaves the repo publicly still needs Sayed / legal sign-off. The
> conservative default is the firewall we already run (comma2k19 + Cosmos-DD), widened only where a license is
> unambiguously permissive.

---

## 0. TL;DR — can we finally have "one dataset for all"?

**Yes — from a permissively-licensed subset, and this is the key result.** The reason every pod currently rebuilds
the cache is that our richest corpus (NVIDIA **PhysicalAI-AV real**) is under a license that is *internal-dev-only,
confidential, non-transferable, and destroy-on-termination* (D-002) — its derivatives may **never** be shipped to
HuggingFace, even privately (Sayed doctrine, BACKLOG P0). That forces the "recipe/manifest" workaround
(`Sayood/tanitad-realmix`, `stack/DATA_MANIFEST.json`, `scripts/rebuild_cache.py`): ship IDs + build params +
checksums, and make every pod re-materialise the episodes itself.

An **own dataset built only from license-clean sources removes that constraint**: we can ship the *actual normalized
episodes* (the drop-in `[T,9,256,256]` cache) to a private or public HF repo, and every training workflow pulls the
same bytes. The owned-safe sources are enough to reproduce PhysicalAI-AV's *value* (real urban + night/weather +
long-tail + off-expert action coverage) without its license:

| Owned-safe class | Sources | What they replace from PhysicalAI-AV |
|---|---|---|
| **Permissive (attribution-only) — the unrestricted core** | comma2k19 (MIT), Cosmos-Drive-Dreams (CC-BY-4.0), WorldModel-Synthetic-Scenarios (OpenMDW-1.1), PandaSet (CC-BY-4.0), Udacity (MIT), comma10k (MIT, aux), CARLA self-gen (MIT+CC-BY) | real-CAN grounding, synthetic urban/weather, 264k-clip safety-critical long-tail, real SF urban, off-expert action-consequence |
| **Copyleft (owned but must stay open+ShareAlike)** | Zenseact ZOD (CC BY-SA-4.0) | **real EU urban + night/winter/weather + real CAN** — the single best owned replacement for PhysicalAI's real urban richness |

**Headline recommendation:** build `tanitad-own` from the permissive core (shippable as real episodes, private→public,
even proprietary), and add **ZOD** as the flagship new real-urban ingest (as a separate CC-BY-SA shard). First new
targets in order: **ZOD → PandaSet → WorldModel-Synthetic-Scenarios (pose-probe first) → Udacity → CARLA off-expert**.
Everything famous and non-permissive (nuScenes, Waymo, Argoverse, KITTI, BDD100K, A2D2, ONCE, DDAD, Oxford, Lyft,
Honda, Cityscapes, Mapillary, CADC, OpenDV-YouTube) is **blocked** for a redistributable owned derivative — see §4.

---

## 1. The contract the own-dataset must fit (drop-in target)

Every adapter emits the identical episode contract, validated by `stack/tanitad/data/_contract.py :: assert_contract`
and fingerprinted by `CORPUS_META` (D-017 / I7). Own-data episodes must be **byte-compatible** so they coexist with or
replace the AV cache with zero trainer change.

| Field | Shape / dtype | Semantics | Source |
|---|---|---|---|
| `frames` | `[T, 9, 256, 256]` uint8 | D-015 3-frame RGB stack @10 Hz `[t−200ms, t−100ms, t]`, current frame last-3-channels | `comma2k19.py :: stack_frames` |
| `actions` | `[T, 2]` float32 | `(steer_road_rad, accel_mps2)`; steer = `atan(WHEELBASE·curvature)` or `wheel_deg·π/180 / steer_ratio`; accel = finite-diff of speed | `_contract.py :: finite_diff_accel` |
| `poses` | `[T, 4]` float32 | `(x_east_m, y_north_m, yaw_rad, v_mps)` in a segment-local ENU/clip-local frame | per-adapter |
| `episode_id` | int | stable hash of the source unit | per-adapter |

- **CORPUS_META fingerprint (must match across all sources):** `channels=9, image_size=256, f_eff_px=266.0, hz=10.0,
  actions=(steer_road_rad, accel_mps2), poses=(x_east_m, y_north_m, yaw_rad, v_mps)`.
- **Geometry (D-016, `stack/tanitad/data/calib.py`):** every camera is cropped so the **effective focal length is a
  shared `F_REF = 266 px`** at 256 px input. comma2k19 (f≈910 px pinhole) is the near-uncropped reference;
  `focal_crop_resize` handles pinhole cameras, `ftheta_crop_resize` handles f-theta fisheye (PhysicalAI/Cosmos
  front-wide; **also ZOD's Kannala-Brandt fisheye**). Because canonicalization is baked in at build time, shipped
  episodes are already geometry-normalized — intrinsics collapse to the `f_eff=266` constant, so the cache is truly
  drop-in. Extrinsic (mount height/pitch) normalization is the deferred D-016 R1 step; H17 (unified 120° canvas) is a
  proposed *superset* direction (§5.4).
- **On-disk episode format** (`stack/tanitad/data/mixing.py :: save_episode` → `stack/tanitad/data/epcache.py`):
  `torch.save({"frames_u8": uint8[T,9,256,256], "actions": [T,2], "poses": [T,4], "episode_id": int})` as
  `ep_%05d.pt` under `<cache_root>/<tag>-<key>/`, fault-tolerant (per-item skip markers), mmap-loaded. **This is the
  artifact the own-dataset ships.**

---

## 2. Why an own dataset (the problem, precisely)

1. **PhysicalAI-AV real is un-redistributable.** The NVIDIA Autonomous Vehicle Dataset License is internal-dev-only,
   confidential, non-transferable, non-sublicensable, 12-month expiry, destroy-on-termination — no publication/
   redistribution of the data *or derivatives* (D-002 review,
   `Data Engineering/Research/2026-07-07-physicalai-av-license-review.md`). So a PhysicalAI-derived episode cache can
   never be a shared asset; each pod must rebuild it from the gated origin, and it can never back a public claim.
2. **The current workaround is a recipe, not a dataset.** Sayed's 2026-07-11 doctrine (BACKLOG P0 `-2`/`-1`): ship a
   *recipe* (`Sayood/tanitad-realmix`: comma-by-reference + `r0_selection.parquet` + build params + split seed +
   per-episode SHA256 + rebuild instructions), never the PhysicalAI bytes. This kills the single-point-of-failure
   (throttled rsync off pod1 stalled the record run) but still makes every pod **rebuild**, and still can't be public.
3. **An own dataset from clean sources ships the bytes.** If the corpus contains only permissive/attribution data,
   the normalized episodes themselves are redistributable → one HF dataset, pulled identically by pod1/2/3 and by any
   public user, no rebuild, citable in the paper. That is the "one dataset for all" the program wants, and it directly
   reduces the PhysicalAI dependency by giving the trainers an owned real-urban corpus (ZOD + PandaSet) plus owned
   synthetic long-tail (Cosmos-DD + WorldModel-Synth) to lean on instead.

The own dataset does **not** delete PhysicalAI-AV from internal training — it makes the *shareable core* license-clean
so PhysicalAI becomes an *optional internal booster* (recipe-only), not the mandatory, un-shippable backbone.

---

## 3. Source survey (master table)

One row per corpus. **"Owned redist.?"** = can we build and re-host (private and/or public) a *normalized + augmented*
derivative? YES = permissive (attribution-only); YES* = copyleft (must stay CC-BY-SA, can't be proprietary); NO =
non-commercial / no-derivatives / gated / copyright-barrier; CONDITIONAL = needs a negotiated license or a
non-commercial-only scope. Signals key: **V** front video (not just keyframes) · **E** ego-motion/pose · **A** real
speed+steer/CAN · **K** camera calibration/intrinsics.

### Tier A — Owned-safe, permissive (the unrestricted core)

| Corpus | License (page) | Owned redist.? | Signals | Scale | Coverage | Fit vs contract |
|---|---|---|---|---|---|---|
| **comma2k19** | **MIT** ([repo LICENSE](https://github.com/commaai/comma2k19/blob/master/LICENSE) · [HF](https://huggingface.co/datasets/commaai/comma2k19)) | **YES** | V E A K | 33 h / 2019×1-min / 20 fps | CA-280 highway, day/clear (narrow) | **Excellent** — real CAN steer+speed, GNSS pose, intrinsics. Already the loaded anchor (D-009). |
| **Cosmos-Drive-Dreams** (`nvidia/…Cosmos-Drive-Dreams`) | **CC-BY-4.0** ([HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) | **YES** (attribution) | V E(4×4 pose→steer/accel) K | 5,843 RDS-HQ clips + 81,802 synth videos, 7 weathers, 30 fps | synthetic urban + rain/snow/fog/night + VRUs (high) | **High** — 120° front-wide, poses→actions; already loaded (D-014, `cosmos_drive.py`). Synthetic pixels ≠ off-expert. |
| **WorldModel-Synthetic-Scenarios** (`nvidia/…WorldModel-Synthetic-Autonomous-Driving-Scenarios`) | **OpenMDW-1.1** ([HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios) · [license](https://openmdw.ai/license/1-1/)) | **YES** (retain notices) | V K, **⚠ no ego-pose/actions on card** | 264k clips / 1,467 h / 8.3 TB / 4K@24 | safety-critical long-tail: cut-in 32.9%, veh-ped 21.1%, lanechange 12.9%, ped 12.4%, weather-deg 9.2%, emergency 2.7% | **Massive scale**, but pose-less → **IDM (H7) or video-only**. Pose probe is the gate (BACKLOG P1). |
| **PandaSet** (Hesai/Scale) | **CC-BY-4.0 + Dataset Terms** ([pandaset.org](https://pandaset.org/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [HF mirror](https://huggingface.co/datasets/georghess/pandaset)) | **YES** (attribution + privacy clause) | V E K (no CAN → steer from curvature) | 103 scenes × 8 s @10 Hz (~48k imgs, ~44.5 GB) | SF + El Camino, urban/suburban, day | **Good, small** — 6-cam incl. front, GPS/IMU pose, pinhole intrinsics. "First AV set for commercial use." Use HF mirror (Scale pulled official host; CC is irrevocable). |
| **Udacity self-driving-car** (CH2/CH3) | **MIT** (`datasets/` only) ([LICENSE.md](https://github.com/udacity/self-driving-car/blob/master/datasets/LICENSE.md)) | **YES** | V A K (E weak) | ~10 h; CH3 adds lidar | Mountain View / El Camino CA, day | **Good** — front+L+R cam, real steer/throttle/brake/speed. Keep separate from repo *code* (GPLv3). |
| **CARLA** (self-generated) | **MIT code + CC-BY assets** ([LICENSE](https://github.com/carla-simulator/carla/blob/master/LICENSE) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) | **YES** (attribute CARLA; stock assets only) | V E A K (all perfect, you set intrinsics) | unlimited | Towns 01–12, weather/time presets, off-expert | **Perfect labels + off-expert/counterfactual coverage**; synthetic gap. D-014 closed-loop arm (pod render being unblocked). |
| **comma10k** | **MIT** ([LICENSE](https://github.com/commaai/comma10k/blob/master/LICENSE)) | **YES** | images only (seg masks) | 10k PNG + 6-class masks | — | **Aux only** (not action-grounding) — MIT-clean segmentation/pretraining signal. |

### Tier B — Owned but copyleft (CC-BY-SA; must stay open + ShareAlike)

| Corpus | License (page) | Owned redist.? | Signals | Scale | Coverage | Fit vs contract |
|---|---|---|---|---|---|---|
| **Zenseact ZOD** | **CC BY-SA-4.0** ([license](https://zod.zenseact.com/license/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) | **YES\*** (ShareAlike + privacy notice + delete-on-request + no-military) | V E A K | Frames 100k · Sequences 1,473×20 s · Drives 29 (multi-min); OXTS+CAN @100 Hz | **14 European countries**, highway+urban+suburban, **day/night, seasons/weather** | **Excellent** — best owned real urban corpus; **real CAN steer+accel**, full calibration (Kannala-Brandt fisheye → `ftheta_crop_resize`). Fills comma's every gap. **Corrects the internal `DATASET_LANDSCAPE.md` "research/NC" mis-tag** (verified CC BY-SA by 3 independent fetches). |

### Tier C — NOT owned-redistributable (blocked for a shareable derivative)

| Corpus | License (page) | Why blocked | Note |
|---|---|---|---|
| **A2D2** (Audi) | CC **BY-ND**-4.0 ([site](https://a2d2-dataset.github.io/) · [BY-ND](https://creativecommons.org/licenses/by-nd/4.0/)) | **NoDerivatives** — a normalized/augmented cache is an adaptation → may not be distributed | Commercial-OK and **richest bus signals** (steer/speed/brake/throttle/accel). Only a separate Audi license unblocks it. Verbatim re-host is allowed; our derivative is not. |
| **nuScenes / nuImages / nuPlan** (Motional) | CC BY-NC-SA-4.0 (modified) ([terms](https://www.nuscenes.org/terms-of-use) · [commercial](https://www.nuscenes.org/terms-of-use-commercial)) | **Non-commercial** free tier; SA would force NC-SA on any derivative | **CONDITIONAL** via a paid Motional commercial license (terms private; must confirm it grants *redistribution*). Strong fit (CAN speed+steer). |
| **Waymo Open** (Perception+Motion) | Waymo Non-Commercial Agreement ([terms](https://waymo.com/open/terms/)) | NC **and** "distribution limited to those registered at waymo.com/open" → no public/private re-host | CONDITIONAL via negotiated commercial license. Motion subset has no imagery. |
| **Argoverse 1 & 2** | CC BY-NC-SA-4.0 ([terms](https://argoverse.github.io/user-guide/terms_and_conditions.html)) | **Non-commercial** | CONDITIONAL-YES only for a *non-commercial* CC-BY-NC-SA derivative. |
| **KITTI / KITTI-360** | CC BY-NC-SA-3.0 ([KITTI](https://www.cvlibs.net/datasets/kitti/) · [360](https://www.cvlibs.net/datasets/kitti-360/)) | **Non-commercial** + daytime/clear only | Excellent OXTS ego-motion, but NC and low coverage. |
| **BDD100K** | UC Regents custom ([dataset license](https://doc.bdd100k.com/license.html)) | Commercial rights **BAIR-Commons-members only**; free grant is "educational/research/not-for-profit" | **CONDITIONAL** — a non-profit/research re-host is arguably permitted (keep the notice); commercial needs a UC Berkeley OTL license. Video + phone GPS/IMU, **no intrinsics, no CAN**. Note: the repo `LICENSE` (BSD-3) covers **SDK code only**, not the data. |
| **ONCE** (Huawei) | CC BY-NC-SA-4.0 + terms ([terms](https://once-for-auto-driving.github.io/terms_of_use.html)) | NC **and** explicit *"prohibits You from distributing this dataset or modified versions"* | Double-blocked. Only abstract works (trained models) may be shared. |
| **DDAD** (Toyota TRI) | CC BY-NC-SA-4.0 ([LICENSE](https://github.com/TRI-ML/DDAD/blob/master/LICENSE.md)) | Non-commercial (code is MIT; **data** is NC-SA) | — |
| **Oxford RobotCar / Radar** | CC BY-NC-SA-4.0 ([site](https://robotcar-dataset.robots.ox.ac.uk/)) | Non-commercial + academic-email gate | — |
| **Lyft Level 5** (Woven) | CC BY-NC-SA-4.0 | Non-commercial + **official hosting offline** (DNS dead) | — |
| **Honda HDD** | Non-commercial, gated ([hdd](https://usa.honda-ri.com/hdd)) | NC + university-email gate, no redistribution grant | Strong CAN signals wasted by the license. |
| **Cityscapes** | Custom NC ([license](https://www.cityscapes-dataset.com/license/)) | NC + *"strictly not allowed to make … accessible to third parties … modified versions"* | Hard no-redistribution. |
| **Mapillary Vistas** | CC BY-NC-SA ([dataset](https://www.mapillary.com/dataset/vistas)) | Non-commercial | (Raw Mapillary street imagery is CC-BY-SA but lacks ego-motion.) |
| **CADC** (Canadian Adverse Driving) | CC BY-NC-4.0 ([terms](https://scale.com/legal/cadc-terms-of-use)) | Non-commercial | Adverse-weather value, but NC. |
| **comma.ai 2016 research** (`commaai/research`) | CC BY-NC-SA-3.0 ([repo](https://github.com/commaai/research)) | Non-commercial (unlike comma's MIT sets) | 7.25 h front cam + steer/speed/accel — great fit, killed by NC. |
| **OpenDV-YouTube** (OpenDriveLab) | YouTube ToS + per-video © ([README](https://github.com/OpenDriveLab/DriveAGI/blob/main/opendv/README.md)) | **Copyright barrier** — IDs only; each video is © its uploader; their README concedes *"cannot be directly redistributed"* | NO for a redistributable derivative. Only individually CC-BY-marked source videos could ever qualify (a small minority; ToS-acquisition still open). H7 keeps it *training-internal, source-derived/gray* (`YOUTUBE_DASHCAM_STRATEGY.md`). |
| **PhysicalAI-AV real** (`nvidia/PhysicalAI-Autonomous-Vehicles` +NCore/+NuRec) | NVIDIA AV Dataset License ([card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)) | Internal-dev-only, confidential, non-transferable, 12-mo, destroy-on-termination → **never** redistributable | The problem being solved (D-002/D-012). Stays recipe-only, internal. |
| **nuReality** | unclear/JS-blocked | Not naturalistic ego-driving (staged VR intersection study) | Drop — poor fit regardless of license. |

---

## 4. Licensing conclusion (the crux deliverable)

### 4.1 The owned-safe set
A **genuinely owned, redistributable** derivative can be built from, and only from:

- **Unrestricted core (attribution-only; may be private OR public, open OR proprietary, commercial):**
  **comma2k19 (MIT), Udacity (MIT), comma10k (MIT), Cosmos-Drive-Dreams (CC-BY-4.0), PandaSet (CC-BY-4.0),
  WorldModel-Synthetic-Scenarios (OpenMDW-1.1), CARLA self-generated (MIT+CC-BY).** Obligation = carry attribution /
  license notices in a `NOTICE` file; CARLA/PandaSet/Cosmos need attribution; PandaSet/ZOD carry a privacy clause.
  OpenMDW-1.1 verbatim grants *"permission … to deal in the Model Materials without restriction"* with only a
  retain-the-notice condition, and *"does not impose any restrictions … on any outputs"* — MIT-class for data.
- **Copyleft extension (owned, but the derivative must itself be CC-BY-SA-4.0 — cannot be proprietary/closed):**
  **Zenseact ZOD.** Great for a *public* owned dataset; incompatible with a *closed* one; **must not be co-mingled with
  non-SA sources in the same redistributed file** (ShareAlike is viral).

### 4.2 The nuances that decide "owned"
- **NC (non-commercial) ≠ owned.** nuScenes, Argoverse, KITTI, DDAD, Oxford, Lyft, CADC, Mapillary, ONCE, comma-2016
  all permit a *non-commercial* CC-BY-NC-SA re-host, but the derivative inherits NC → unusable by any commercial/
  proprietary workflow and not "one dataset for all." **Excluded** from the owned dataset.
- **ND (no-derivatives) = blocked.** A2D2 is commercial-friendly yet forbids distributing an adapted version — and our
  normalization+augmentation *is* an adaptation. Excluded (verbatim re-host only).
- **Members-only / registered-recipients-only = blocked.** BDD100K (commercial = BAIR members) and Waymo
  (recipients must be registered waymo.com/open users) cannot be openly re-hosted.
- **Copyright-barrier = blocked.** OpenDV-YouTube: the underlying videos are third-party copyrighted; only an ID list
  is distributable.
- **Private vs public is a red herring for the *blockers*.** A "private" HF repo shared with any third party is still
  *distribution* to them, so it does not rescue NC/ND/registered-only data. It only helps in that a permissive owned
  dataset can start **private** (internal-only) and flip to **public** on Sayed's call without a license change.
- **MIT-on-data caveat.** MIT is written for "the Software"; comma states it explicitly for the data, so intent is a
  full grant — low risk, but a lawyer should bless treating MIT *data* (comma2k19, comma10k, Udacity) as fully
  redistributable.
- **Privacy is a separate axis from copyright.** Redistributing real EU data (ZOD, PandaSet) carries GDPR/personality
  obligations regardless of license — plan face/plate anonymization; ZOD is already de-identified and *requires* its
  notice to travel with the data.

### 4.3 Verdict
**Build `tanitad-own` from the Tier-A permissive core (shippable as real episodes, private→public, even proprietary),
plus ZOD as a separate CC-BY-SA public shard.** This is the subset that genuinely permits an owned, redistributable
dataset and gives the trainers an owned substitute for PhysicalAI-AV's real-urban + synthetic-long-tail value.

---

## 5. Augmentation pipelines already in TanitAD, and how they plug in

| Pipeline | Contributes | Status | Owned-safe? |
|---|---|---|---|
| **NuRec counterfactual action-augmentation** | **action-grounding** (causal action→consequence; B1/H15) | **proposed** (design doc + queued feasibility probe; no code) | **Open question** — see below |
| **Degraded-visibility (SC-05 / D8)** | **OOD robustness + self-monitoring** (H11 AUROC>0.85 gate) | data-sourced + first paired measurement | **Yes** (sources are owned) |
| **CARLA (+ retired MetaDrive frontcam)** | **off-expert action-consequence + closed-loop** (D5/D6, occluders, collisions) | live-measured partial; pixel-eval pod-blocked | **Yes** (self-generated) |
| **D-016 focal canon / H17 unified-FOV** | **geometry normalization / scale** | D-016 implemented; H17 proposed | **Yes** (our code) |

### 5.1 NuRec counterfactual augmentation (`…/Research/COUNTERFACTUAL_ACTION_AUGMENTATION.md`)
Take real `(scene, action, ego-motion)`, neurally reconstruct the drive volume (3DGS/NuRec), **vary the action** (steer/
accel/lane-offset fan), re-render frames `F'(A')` and derive ego-motion `M'(A')` via bicycle-model rollout → a
self-supervised action→consequence signal a single human trajectory can't give. "**Golden sample**" = 1 comma highway
+ 1 urban scene × a pre-registered fan {nominal, lateral ±0.05/±0.15 rad, longitudinal −3/−1.5/+1.5 m/s², one
hazard-approach-then-recovery}. **Fidelity gate (falsifier):** the nominal branch must reproduce reality — SSIM/feature
match **AND** ego-motion **ADE < ~0.3 m** — else its counterfactuals are untrustworthy; kept a *minority* augmentation.
- **Licensing (open question for Sayed):** NVIDIA **NuRec** is part of the PhysicalAI family
  (`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, NVIDIA AV License). Renders of *PhysicalAI* scenes inherit the
  firewall (internal-only, un-shippable). Renders of *comma2k19/CARLA* geometry are of owned scenes, but whether the
  **NuRec tool's license** permits redistributing those renders is unconfirmed. **Clean-owned path:** run an
  **open 3DGS / neural-recon** engine (not NVIDIA NuRec) over comma/CARLA geometry → a fully owned counterfactual
  augmentation. Recommend the feasibility probe compares both.

### 5.2 Degraded-visibility (SC-05 / D8) — correction to the brief's framing
There is **no fog/rain/night corruption function in the codebase**. "Degraded visibility" is (a) **sourced** from
**Cosmos-DD weather renders** (Foggy/Rainy/Snowy/Night — owned CC-BY) + **WorldModel-Synth weather-deg 9.2%** (owned
OpenMDW), and (b) measured by **D8**, an OOD-detection AUROC gate (bar **>0.85**) on the model's free familiarity
signal (H11 self-monitoring; `stack/scripts/d8_preview.py`, `cosmos_pairs.py`). First matched-pairs result: 16/23
scenes higher imagination-error under degraded weather (p≈0.047).
- **Plug-in:** for the own dataset, add a **genuinely-own photometric degradation augmentation** (fog/rain/glare/
  low-light/motion-blur filters applied to clean-source frames) — cheap, 100% owned, and it makes the SC-05/D8 paired
  set arbitrarily large without needing more rendered weather clips. The Cosmos/WorldModel weather renders remain the
  *real-rendered* half of each pair.

### 5.3 CARLA / MetaDrive (D-014, D-010)
CARLA is the **off-expert + closed-loop** arm — scripted perturbation rollouts (steer bias, throttle/brake pulses),
scripted occluders (H15/D9), blocked-route topology (D5/D6), collisions, closed-loop eval — the action-consequence
coverage real logs never contain. Self-generated → MIT+CC-BY → fully owned. (MetaDrive retired as a corpus by D-014;
the front-cam RGB adapter with `PerturbConfig` is contract-clean and retained.) Pod pixel rendering is currently
blocked (compute-only GPUs); the turnkey graphics-pod recipe is the unblock.

### 5.4 Geometry: D-016 (now) → H17 (proposed)
D-016 canonicalizes `f_eff=266` by cropping every camera to the shared canonical half-angle — this is what makes any
new source **contract-drop-in** (pinhole via `focal_crop_resize`, fisheye via `ftheta_crop_resize`, incl. ZOD). **H17**
(`Architecture & Inference/Research/UNIFIED_FOV_FOVEATED_PATCHING.md`, open, Sayed 2026-07-12) *inverts* the crop-down:
pad narrow cameras (comma) **up** to a shared 120° angle-linear canvas with a masked "unobserved" periphery + foveated
patching, turning the periphery into a free H15 imagination target. **Own-dataset implication:** build at the current
`f_eff=266` contract now (drop-in); keep per-source native intrinsics in the manifest so a 120° unified-canvas variant
can be re-derived if H17 is adopted — don't discard the wide FOV of ZOD/PandaSet/Cosmos at ingest (store the wider crop
or the native frame reference).

---

## 6. TanitAD-Own Dataset spec

### 6.1 Format — a drop-in for the current trainers
- **Episode:** the exact contract (§1) — `frames [T,9,256,256] u8`, `actions [T,2]`, `poses [T,4]`, `episode_id`;
  `CORPUS_META` fingerprint identical to comma2k19 (D-017), so `MixedWindowDataset` admits own-data episodes alongside
  or in place of the AV cache with no trainer change.
- **On disk / on HF:** the `epcache` layout (`ep_%05d.pt` + `DONE`) plus:
  - `DATA_CARD.md` — per-source license + attribution + "changes made", CORPUS_META, split policy (route/clip-level
    I3), A8 consequence stats, semantic-coverage audit (intersections/ped/night/rain per source).
  - `MANIFEST.json` — per-episode `{source, source_unit_id, license, build_params_hash, split, sha256}` (extends
    Sayed's `DATA_MANIFEST.json` idea to the clean corpus → verify-without-rebuild AND reproducible).
  - `NOTICE` — MIT/CC-BY/OpenMDW attributions; privacy notices (ZOD, PandaSet).
- **Two repos to respect copyleft (do not co-mingle SA + non-SA):**
  1. **`tanitad-own-core`** — Tier-A permissive only → MIT/CC-BY/OpenMDW; may be **private → public**, even
     proprietary; single `NOTICE`.
  2. **`tanitad-own-zod`** — the ZOD-derived shard, licensed **CC-BY-SA-4.0**, carries the ZOD privacy notice. Mixed
     at train time via `MixedWindowDataset`, never merged into the core file.

### 6.2 Recommended source mix (license-cleanliness × fit × coverage)
Phase-0.5 clean-owned mix (all bake-off-gated per D-010 — a share stays only if real-held-out D1/D2 don't regress):

| Role | Source(s) | Weight (start) | Rationale |
|---|---|---|---|
| Real-CAN anchor | **comma2k19 (MIT)** | ~30% | cleanest inverse-dynamics regime; real steer+speed; public anchor |
| Real urban/night/weather | **ZOD (CC-BY-SA)** | ~25% | real EU urban + real CAN + night/winter — the owned PhysicalAI-urban replacement |
| Real urban (supplement) | **PandaSet (CC-BY)** + **Udacity (MIT)** | ~10% | SF urban / extra real steer; small but clean |
| Synthetic long-tail | **Cosmos-DD (CC-BY)** + **WorldModel-Synth (OpenMDW, video-only/IDM)** | ~20% | weather/VRU/urban + safety-critical (cut-in/emergency/ped) |
| Off-expert / counterfactual | **CARLA (MIT)** + NuRec/open-3DGS pilot | ~15% | action-consequence coverage logs lack; H15/D5/D6 |
| Aux (not in the 9-ch mix) | comma10k (MIT) | — | optional segmentation/pretraining |

Weights are a starting proposal; every change is one lever per run (D-004) and must clear the real-only-vs-mixed
bake-off (D-010).

### 6.3 Per-source signal/label extraction
- **Real CAN (comma2k19, ZOD, Udacity):** steer = `wheel_deg·π/180 / steer_ratio` (comma) or native road-wheel/CAN
  steer (ZOD); accel = finite-diff of CAN speed (or ZOD's native longitudinal accel). Highest action fidelity.
- **Pose-only (Cosmos-DD, PandaSet):** steer = `atan(WHEELBASE·κ)`, `κ = yaw_rate / v` (clipped at low speed —
  `cosmos_drive.py :: poses_to_signals`); accel = finite-diff of speed from the position derivative.
- **Pose-less (WorldModel-Synth, any YouTube):** **IDM (H7) inverse-dynamics head** (trained on the real-CAN corpora)
  or monocular **visual odometry**; cross-agreement is the per-clip action-quality score → low agreement ⇒ video-only
  (representation learning without action conditioning). Gate on WorldModel's pose-field probe first (BACKLOG P1).
- **Calibration:** pinhole → `focal_crop_resize`; f-theta/Kannala-Brandt fisheye (ZOD) → `ftheta_crop_resize`; CARLA →
  set FOV/intrinsics at render. All land `f_eff=266`.

### 6.4 Augmentation plan (own-able)
1. **Photometric degradation overlay** (own filters) → SC-05/D8 OOD/robustness, unlimited scale.
2. **CARLA off-expert episodes** → action-consequence + closed-loop (D-014).
3. **Counterfactual fan** via open-3DGS (or NuRec if licensing clears) on comma/CARLA geometry → B1/H15, fidelity-gated
   (SSIM + ADE<0.3 m), minority share.
All augmentation is domain-tagged and bake-off-gated; real gates are validated real-only.

---

## 7. Prioritized ingest order + first targets

| # | Target | Why first | License | Est. cost |
|---|---|---|---|---|
| 1 | **Zenseact ZOD** | The headline owned replacement for PhysicalAI's real urban richness — real CAN + calibration + ego-motion, 14 EU countries, night/winter/weather (everything comma lacks). First **correct** the `DATASET_LANDSCAPE.md` "research/NC" mis-tag → CC-BY-SA. Pilot 5 drives; fisheye via `ftheta_crop_resize`. | CC-BY-SA-4.0 | ~3–4 h loader |
| 2 | **PandaSet** | Cheapest clean real-urban add (CC-BY, ~44.5 GB HF mirror, pinhole, GPS/IMU pose). Reuses the cosmos pose→signals path. | CC-BY-4.0 | ~2–3 h |
| 3 | **WorldModel-Synthetic-Scenarios** | Owned 264k-clip safety-critical long-tail; **run the pose-field probe first** — if poses exist, near-zero cosmos-mirror; else video-only/IDM. | OpenMDW-1.1 | probe: minutes; loader: 2–3 h if posed |
| 4 | **Udacity CH2/CH3** | MIT real steer+speed, quick augment; keep separate from GPLv3 code. | MIT | ~2 h |
| 5 | **CARLA off-expert** | Owned action-consequence + closed-loop; already the D-014 arm (unblock pod render). | MIT+CC-BY | infra-bound |

Already owned + loaded: **comma2k19 (MIT)** and **Cosmos-Drive-Dreams (CC-BY)** — the own-dataset core exists today; ZOD
is the decisive new ingest.

---

## 8. Ingest pipeline design (source → normalize → augment → cache → HF)

```
 SOURCE (per-license fetch)                    NORMALIZE-TO-CONTRACT                 AUGMENT                 CACHE            PUBLISH
 comma2k19  HF tars (MIT)  ─┐        per-source adapter (comma2k19.py /        ┌ photometric degrade    epcache          tanitad-own-core (MIT/
 ZOD        open dl (BY-SA) ┤  ───▶  cosmos_drive.py pattern):                 │ (own SC-05/D8)          ep_%05d.pt   ──▶  CC-BY/OpenMDW) private→public
 PandaSet   HF mirror (BY)  ┤        • actions: CAN | pose→atan(wb·κ) | IDM    ┼ CARLA off-expert        collision-safe    + DATA_CARD + MANIFEST + NOTICE
 Cosmos-DD  HF (BY)         ┤        • poses: local ENU / clip-local           │ (D-014)                 keys, mmap,
 WorldModel HF (OpenMDW)    ┤        • D-015 3-frame/9-ch stack                └ counterfactual fan      fault-tolerant    tanitad-own-zod (CC-BY-SA)
 Udacity    torrents (MIT)  ┤        • D-016 focal canon → f_eff=266             (open-3DGS, gated)                          ← ZOD shard, kept separate
 CARLA      self-render     ┘        • assert_contract(channels=9) + I7                                                    PhysicalAI → recipe-only (never shipped)
```

- **Reproducibility:** `MANIFEST.json` (source unit IDs + build-params hash + per-episode SHA256) lets any consumer
  verify the shipped episodes match a deterministic rebuild — the paper's reproducibility statement, and the escape
  from "rebuild on every pod" for the clean core.
- **Firewall discipline preserved:** the PhysicalAI portion never enters these repos; it stays the internal
  recipe/manifest (`Sayood/tanitad-realmix` pattern). The own dataset is additive and un-gated; it *reduces* the
  PhysicalAI dependency by making the shareable backbone license-clean.
- **Privacy pass:** face/plate anonymization for redistributed real EU data (ZOD already anonymized + notice travels;
  apply to PandaSet); a `--blur` stage in the adapter before cache.
- **Contract guardrails:** `_check_contract` (mixing.py) + `assert_contract` + the I7 `CORPUS_META` equality test fail
  the build fast if any source drifts from `[T,9,256,256]` / f_eff=266.

---

## 9. Open licensing questions for Sayed (judgment calls)

1. **ZOD ShareAlike.** Accept **CC-BY-SA-4.0 on the public own-dataset** (forces open — the ZOD shard, and anything
   co-mingled with it, cannot be proprietary/closed)? If a *closed* dataset is ever wanted, ZOD stays a separate public
   shard and the closed product uses only the Tier-A permissive core. **Recommend: accept SA for a public ZOD shard.**
2. **MIT-on-data blessing.** OK to treat MIT-licensed *data* (comma2k19, comma10k, Udacity) as fully redistributable
   (MIT's text says "Software")? Low risk; a one-line legal nod closes it.
3. **WorldModel-Synth firewall (D-022, currently HOLD).** Confirm **OpenMDW-1.1 → public-claimable** so the 264k-clip
   long-tail can be *cited*, not only trained on. (OpenMDW is Linux-Foundation permissive; the hold is conservatism.)
4. **nuScenes / nuPlan commercial license.** Worth contacting Motional for a paid commercial license (would unlock the
   best real-urban CAN corpus)? Terms are private and must be confirmed to grant *redistribution* of a derivative.
5. **A2D2 (Audi).** Worth a direct license conversation? It has the richest native bus signals but ND blocks the
   derivative; only a bespoke Audi grant unblocks redistribution.
6. **NuRec tool license.** Does NVIDIA NuRec permit redistributing renders of *non-PhysicalAI* (comma/CARLA) geometry?
   If not / unclear, adopt an **open 3DGS** engine for a fully-owned counterfactual pipeline.
7. **Privacy/GDPR.** Confirm the anonymization bar (face/plate blur) required to redistribute real EU data (ZOD,
   PandaSet), independent of the copyright license.

---

### Provenance
Contract & plumbing: `stack/tanitad/data/{_contract,comma2k19,physicalai,cosmos_drive,calib,mixing,epcache}.py`.
Internal decisions: `DECISIONS.md` D-002/D-009/D-010/D-012/D-014/D-016/D-017/D-022; `DataEng/DATA_STRATEGY.md`;
`Data Engineering/Research/{DATASET_LANDSCAPE.md, 2026-07-07-physicalai-av-license-review.md, YOUTUBE_DASHCAM_STRATEGY.md,
COUNTERFACTUAL_ACTION_AUGMENTATION.md, STATE.md}`; `Data Engineering/BACKLOG.md` (Sayed 2026-07-11 recipe doctrine);
`Architecture & Inference/Research/UNIFIED_FOV_FOVEATED_PATCHING.md` (H17). External licenses: each linked in §3
(fetched & verified 2026-07-13). This is an engineer's license read to route a decision, not legal advice.
