# H2 — Substrate and Labeling Feasibility for Attention-Based Additional-Camera Usage

**Author:** Research engineer (substrate + labeling stream). **Date:** 2026-07-25.
**Status:** decision doc. **Companion:** `H2_DESIGN_FRAMING.md` (sibling agent, same folder) — this
document supplies the MEASURED substrate for its `L1` / `C-EFF` slots.
**Corpus:** NVIDIA PhysicalAI-Autonomous-Vehicles only (PI: *"stick to the AV dataset"*).

**Evidence classes on every fact:** `MEASURED` (ours + path/command) · `PUBLISHED` (cited) ·
`INHERITED` (another agent/doc, NOT re-verified here) · `ESTIMATED` · `HYPOTHESIS`.

> **Scope note from the PI clarifications folded in 2026-07-25:** cameras only (radar/lidar noted in
> passing, not designed for); the model input is **front-camera-only at decision time**; the
> efficiency claim is **primary**, not optional; and the corpus was **actively searched** for
> intersections/roundabouts rather than sampled and inferred.

---

## 1. VERDICT

# 🟡 GO-WITH-CAVEAT

**The substrate is a GO with no reservations. The label is a GO with one specific, named caveat that
must be closed by a pre-registered out-of-sample confirmation before any GPU is spent.**

| leg | verdict | the deciding measurement |
|---|---|---|
| **Cameras exist** | ✅ **GO — better than assumed** | **7 cameras, full 360° surround, on 100.00 % of our 3,000-clip corpus.** `camera_cross_left_120fov` — literally the "left camera" the PI's examples name — is present on **every** clip |
| **Per-camera calibration** | ✅ **GO** | Per-clip, per-camera **f-theta intrinsics** and **6-DoF extrinsics** on **100.00 %** of clips, 27 sensors. Frustum membership is exactly computable |
| **3D agent tracks** | ✅ **GO** | `obstacle.offline` on **96.90 %** of our corpus, `reference_frame = rig` — no transform chain needed |
| **Non-circular label buildable** | ✅ **GO** | L1 computes end-to-end. **It is derived from 3D world geometry + realised ego dynamics, and touches no model input** — it cannot repeat the `route_target = _NAV_TO_ROUTE[nav_cmd]` failure |
| **Label is decision-relevant** | 🟡 **CAVEAT — the crux risk** | Lift **2.22× [1.30, 3.14]** at the 3 m conflict gate, but the **sign flips to 0.43× [0.24, 0.71]** at 6 m. One threshold decides the direction of the result |
| **Statistical power** | ✅ **GO at corpus scale** | 21/80 episodes gate-positive → **≈624 of 2,376** parity episodes. Clears the n ≥ 40 episode-cluster bar by ~15× |
| **Efficiency claim** | ✅ **GO — strong** | **1.83 %** of frames need a 2nd camera (**6.85 %** with ±1 s hysteresis) → **84–85 % of multi-camera encoder compute saved** vs always-on-7 |
| **Front-camera decidability** | 🟡 **WEAK-POSITIVE** | ROC-AUC **0.650 / 0.685** (episode-grouped CV) from front-camera-derivable proxies vs a **0.46–0.56** shuffle band. Above chance, far from reliable — and this is a **lower bound** |
| **Situation coverage** | 🟡 **SPLIT** | Intersections **✅ 846 episodes** and lane changes **✅ 1,172 episodes** are richly powered. **Roundabouts are NOT: 19 episodes strict / 105 loose** — below the decision-grade bar |

**The one caveat, stated precisely.** The decision-relevance of L1 is **threshold-critical**: at a 3 m
counterfactual-conflict radius an unseen side agent raises P(ego yields) by 2.22×; at 6 m it *lowers*
it to 0.43×. Both CIs exclude 1.0. A physical mechanism explains the flip (below ~3 m two vehicle
centres are on a collision course; above ~3.5 m they are simply in **adjacent lanes**, which
correlates with free-flow cruising), so 3.0 m is defensible *a priori* — but it was chosen after
seeing the sweep. **Required before build: re-run the 3.0 m gate on held-out chunks not used here.**
This is ~2 CPU-hours, no GPU. Details in §6.4.

**What would flip this to BLOCKED:** the out-of-sample lift at 3.0 m failing to separate from 1.0.
**What would flip it to unconditional GO:** that same lift reproducing at ≥1.5× with a CI excluding 1.

---

## 2. SECTION A — CAMERAS

### A.1 What exists — MEASURED

**Source of truth:** `C:\Users\Admin\tanitad-data\physicalai\features.csv` (the dataset's own feature
manifest, 36 rows + header) and `metadata/feature_presence.parquet` (306,152 × 36 booleans).
**Command:** `scratchpad/probe_a.py`, `scratchpad/probe_geom.py`.

**PhysicalAI-AV ships SEVEN cameras. All seven are present on 100.00 % of both the full 306,152-clip
dataset and our 3,000-clip phase-0 selection.** (MEASURED)

| camera | boresight azimuth (rig) | full HFOV | full VFOV | native | rate | present, our corpus |
|---|---|---|---|---|---|---|
| `camera_front_wide_120fov` | **0.0°** | **120.5°** | 67.0° | 1920×1080 | 30.00 fps | **100.00 %** |
| `camera_front_tele_30fov` | +1.0° | 29.6° | 16.6° | 1920×1080 | 30.00 fps | **100.00 %** |
| **`camera_cross_left_120fov`** | **+67.1°** | **121.3°** | 67.2° | 1920×1080 | 30.00 fps | **100.00 %** |
| **`camera_cross_right_120fov`** | **−66.8°** | **120.8°** | 67.2° | 1920×1080 | 30.00 fps | **100.00 %** |
| `camera_rear_left_70fov` | +151.8° | 69.1° | 38.6° | 1920×1080 | 30.00 fps | **100.00 %** |
| `camera_rear_right_70fov` | −151.6° | 70.1° | 38.4° | 1920×1080 | 30.00 fps | **100.00 %** |
| `camera_rear_tele_30fov` | 178.9° | 29.7° | 16.6° | 1920×1080 | 30.00 fps | **100.00 %** |

*Azimuth convention: rig frame, 0° = vehicle-forward, positive = left/CCW. Boresight = `R(q)·[0,0,1]`
from `sensor_extrinsics`. HFOV/VFOV inverted from the **real f-theta radial polynomial** at the frame
edge, not from the nominal name — the `_120fov` label is accurate (120.5° measured) but
`_30fov`/`_70fov` are also accurate. 2,850 clips / 30 chunks.*

**Frame rate MEASURED** from `r0/camera_front_wide/*.timestamps.parquet` (500 files): 605 frames over
20.13 s = **30.00 fps**. Our cache resamples to 10 Hz (3 RGB frames channel-stacked at 100 ms — D-015).

**The union of the 7 frusta is a closed 360° ring.** `cross_left` spans **+6.5° … +127.7°**;
`front_wide` spans **−60.3° … +60.3°**. They overlap by ~54°, and there is no azimuthal gap anywhere
around the vehicle. **There is a left, a right, and a rear.**

### A.2 The finding that changes the design — the model does not see the front camera

`stack/tanitad/data/calib.py` canonicalizes every corpus to a shared effective focal `F_REF = 266`
at 256 px. The retained half-angle is `atan(128/266) = 25.697°`. (MEASURED, `calib.py:38,89-96`;
the crop is applied at `physicalai.py:358` and asserted in `build_pai_cache.py:61`.)

> **The encoder's actual field of view is 51.4°, not 120.5°.**
> **The canonical crop discards 57 % of the front camera's own horizontal field before the model
> ever sees it.**

This is not a defect — it is the D-016 cross-corpus geometry fix, and it is deliberate. `calib.py`'s
own docstring (lines 16–17) anticipated this exact workstream:

> *"the sacrificed wide periphery is precisely what **H2 modality steering** re-introduces later as
> dedicated side views."*

(MEASURED, `stack/tanitad/data/calib.py:16-17`. The name "H2" was reserved for this in the
codebase before this brief existed.)

**Design consequence:** there are now **two** distinct "unseen" regions, and the label must
distinguish them, because they have different remedies:

| region | azimuth | remedy |
|---|---|---|
| **Cropped-away** | 25.7°–60.3° | already in the front camera's pixels — a **wider crop** or a second front tap would recover it, no new sensor |
| **Genuinely off-front** | >60.3° | **only** `cross_left` / `cross_right` can see it — a true sensor request |

MEASURED split of agent-frames outside the model crop: **12.65 %** are cropped-away (in the raw front
camera), **11.69 %** are genuinely beyond the front camera entirely.

### A.3 The two-rig question — ANSWERED, and it is confined to the 120° cameras

The brief flagged that front-wide has two rigs (cy≈543 A / cy≈755 B) and asked whether the *other*
cameras vary by rig. **MEASURED** (`probe_geom.py`, 2,850 clips / 30 calibration chunks):

| camera | cy min / p50 / max | fraction cy < 650 | bimodal? |
|---|---|---|---|
| `camera_front_wide_120fov` | 534.0 / **750.8** / 761.9 | **29.1 %** | **YES** |
| `camera_cross_left_120fov` | 523.0 / **743.8** / 749.5 | **29.1 %** | **YES** |
| `camera_cross_right_120fov` | 529.2 / **742.8** / 749.2 | **29.1 %** | **YES** |
| `camera_front_tele_30fov` | 510.8 / 548.8 / 621.2 | 100.0 % | no |
| `camera_rear_left_70fov` | 529.6 / 543.1 / 583.9 | 100.0 % | no |
| `camera_rear_right_70fov` | 526.8 / 543.8 / 576.7 | 100.0 % | no |
| `camera_rear_tele_30fov` | 516.2 / 545.8 / 639.0 | 100.0 % | no |

**The rig split affects exactly the three 120° f-theta cameras, and it is the same 29.1 % of clips in
all three** — i.e. it is a genuine per-clip *rig* property, consistently stamped across the cameras
that have it, not per-camera noise. The 30°/70° cameras are single-mode.

**The EXTRINSICS also vary by rig — and differently.** MEASURED (same 2,850 clips, grouped by the
rig indicator `front_wide cy < 650`), median boresight and mount per rig:

| camera | Δ azimuth (A−B) | Δ elevation (A−B) | Δ mount height z (A−B) |
|---|---|---|---|
| `camera_cross_left_120fov` | **−0.01°** | +1.67° | +2.6 cm |
| `camera_cross_right_120fov` | **+0.18°** | +1.91° | +3.9 cm |
| `camera_front_wide_120fov` | +0.19° | −0.61° | +1.0 cm |
| `camera_front_tele_30fov` | −0.49° | −2.09° | +7.5 cm |
| `camera_rear_left_70fov` | −1.53° | −1.35° | +13.7 cm |
| `camera_rear_right_70fov` | +3.37° | −2.50° | +16.2 cm |
| `camera_rear_tele_30fov` | ~+1.19° (±180° wrap) | −0.26° | −1.9 cm |

**Reading:** the rigs differ mainly in **elevation (up to ~2.5°)** and **mount height (up to
~16 cm)** — rig A sits higher and pitched up. **Azimuth is nearly rig-invariant** (≤0.2° on the
cross cameras, ≤3.4° anywhere).

**Consequence for H2, which is favourable:** the L1 label depends on *azimuthal* frustum membership,
and azimuth is the one axis the rig split barely touches — so cross-camera frustum membership is
robust to rig even before per-clip correction. Any *vertical* or *metric-height* reasoning (ground
plane, horizon row, IPM) must use per-clip extrinsics. Since each clip carries its own `(cx, cy)`
**and** its own 6-DoF extrinsics, any computation that reads per-clip calibration is rig-safe by
construction. The rig split only bites code that hard-codes a *global* `cy` — which `calib.py`
already refuses to do (`per_clip=False` reverts to geometric-center with a `RuntimeWarning`,
`calib.py:244-253`). **Every measurement in this document uses per-clip intrinsics and extrinsics.**

*(Note: 29.1 % here vs the "23 %" in `calib.py:152`. Different samples — 2,850 clips / 30 chunks here
vs 500 R0-selected clips there. Not a contradiction; quote 29.1 % with this sample size.)*

---

## 3. SECTION B — THE 36-FEATURE QUESTION

### B.1 The complete schema — all 36, enumerated

**Source:** `C:\Users\Admin\tanitad-data\physicalai\features.csv` (the dataset's own manifest — the
tool that owns the fact) cross-checked against `metadata/feature_presence.parquet`. **Our ingest
reads exactly two** (`egomotion`, `camera_front_wide_120fov` — `stack/scripts/physicalai_r0.py:36-38`,
`stack/tanitad/data/physicalai.py`). The "2 of 36" figure is **CONFIRMED** (MEASURED).

| group | n | features | presence, all 306k | presence, our 3,000 |
|---|---|---|---|---|
| **camera** | **7** | front_wide_120, front_tele_30, **cross_left_120**, **cross_right_120**, rear_left_70, rear_right_70, rear_tele_30 | 100.00 % | **100.00 %** |
| **calibration** | **6** | `camera_intrinsics`, `camera_intrinsics.offline`, `lidar_intrinsics.offline`, **`sensor_extrinsics`**, `sensor_extrinsics.offline`, `vehicle_dimensions` | 100.00 % (online) / 97.44 % (offline) | **100.00 %** / 96.90 % |
| **labels** | **3** | `egomotion`, `egomotion.offline`, **`obstacle.offline`** | 100.00 % / 97.44 % / 97.44 % | 100.00 % / 96.90 % / **96.90 %** |
| **lidar** | **1** | `lidar_top_360fov` | 97.44 % | 96.90 % |
| **radar** | **19** | 4 corner-srr0, 4 corner-srr3, 2 side-srr0, 2 side-srr3, front-center srr0/mrr2/imaging-lrr1, rear-left srr0/mrr2, rear-right srr0/mrr2 | 28.48 % / 24.03 % / 7.45 % | 26.93 % / 14.43 % / 2.03 % |

**36 total. ✅ Arithmetic closes exactly.**

Per PI clarification #1, radar and lidar are **out of scope** for the H2 tool vocabulary. Noted for
the record only: `lidar_top_360fov` (96.90 %) is a *360° geometric* channel that could serve as an
independent verification of any camera-frustum computation, and the 19 radars are too sparse
(2–27 %) to be a reliable tool anyway.

### B.2 Features relevant to H2

| feature | coverage (ours) | why it matters here |
|---|---|---|
| **`obstacle.offline`** | **96.90 %** | 3D agent tracks. 16 cols; `reference_frame = rig`, `source = scene:obstacles:autolabels:v2`. **This is the label substrate** |
| **`sensor_extrinsics`** | **100.00 %** | 6-DoF pose per sensor per clip, **27 sensors**, `(qx,qy,qz,qw,x,y,z)`. Enables rig→camera transform |
| **`camera_intrinsics`** | **100.00 %** | Per-clip **per-camera** f-theta: `width,height,cx,cy,fw_poly_0..4,bw_poly_0..4`. Enables exact projection |
| `vehicle_dimensions` | 100.00 % | ego footprint — needed for a physically-grounded conflict radius |
| `egomotion` | 100.00 % | 200 Hz pose+velocity+curvature. **The ego-response substrate** |
| `blurred_boxes` (per camera) | 100.00 % | 2D anonymization boxes `(frame_index,x1,y1,x2,y2)`. **Not** agent annotation — faces/plates |

### B.3 ⚠️ "We have no HD map" — SECOND-PROBED, and it HOLDS

The brief correctly flagged that this claim had never been re-probed. It has now been probed at a
**third independent location**, and the answer is unchanged.

| leg | evidence | class |
|---|---|---|
| 1 | NVIDIA's card: *"we do not include open maps data"* | PUBLISHED (INHERITED via `DATA_STRATEGY_FOR_HIERARCHY.md §1`) |
| 2 | Keyword sweep of the 31,935-byte card: `HD map` 0 · `lane` 0 · `traffic light` 0 · `GPS` 0 · `turn signal` 0 | INHERITED (prior measurement, `DATA_STRATEGY §1`) |
| 3 | **NEW — the feature manifest itself.** All 36 features enumerated above. **There is no map, lane, lane-graph, drivable-area, traffic-light, traffic-sign, or junction feature.** The 36 are exhaustively 7 camera + 6 calibration + 3 label + 1 lidar + 19 radar | **MEASURED (ours, `features.csv`)** |

**Verdict: PhysicalAI-AV ships no HD map, no lane graph, no traffic-light and no sign annotation.**
This is now a three-leg absence including the schema that owns the fact. I consider the claim closed.

**Consequence for H2:** every situation label (roundabout / intersection / lane change) must be
derived **kinematically or from `obstacle.offline`**, never looked up from a map. This is a
constraint, but for H2 it is also a *protection*: it makes an L3-style rule-lookup label physically
unavailable, which removes the temptation that produced the `route_skill = 0.0` failure.

---

## 4. SECTION C — EXISTING LABELS WE CAN REUSE

Compiled by a dedicated sub-audit (code-read, MEASURED unless marked). Full inventory in that
agent's report; here is what H2 must know.

### C.1 ⚠️ The 5-way maneuver softmax — the documented defect is CONFIRMED

```
stack/tanitad/refs/refb.py:70-72
MANEUVER_CLASSES = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")
```
Classes 0/1/2 are **lateral**, 3/4 are **longitudinal**, in **one softmax** (`refc.py:87-92`,
`N_MANEUVERS = 5`). The collision is resolved by **priority, not factorization** —
`refb_labels.py:102-109`: turns overwrite accel/brake.

**Measured blast radius:** the label is lateral while a longitudinal mode is live on
**547/2,201 = 24.85 %** of windows (`label_v3_audit_val100.json`). Prediction side: `accelerate`
predicted **0/881** windows (INHERITED, `results/planfan_clips_tactical_head_val.json`).

> **Binding constraint on H2's design: the sensor-request head MUST NOT be a single softmax over
> mixed axes.** The joint output the PI wants — *"entering the roundabout, activating left camera
> necessary"* — decomposes into **three independent factors**: `SITUATION` (semantic class),
> `BEHAVIOUR` (tactical action) and `SENSOR_REQUEST` (a per-camera **multi-label sigmoid**, because
> left and right can both be needed simultaneously). Fusing these into one softmax would repeat the
> exact error that produced longitudinal blindness. A per-camera independent-Bernoulli head is the
> correct form.

### C.2 ⚠️ The circularity precedent — STILL LIVE AND STILL THE DEFAULT

```
stack/scripts/refb_labels.py:77-78, 172-175
_NAV_TO_ROUTE = {NAV_FOLLOW: ROUTE_STRAIGHT, NAV_LEFT: ROUTE_LEFT, NAV_RIGHT: ROUTE_RIGHT}
def route_target(nav_cmd: int) -> int:
    return _NAV_TO_ROUTE[nav_cmd]
```
Live at `refb_train.py:176-178`. Defaults keep it on (`config.py:237 v2_labels=False`;
`refc_train.py:557 --labels default="v1"`). Measured consequence: `route_acc_nav` **1.0000**,
`route_skill` **0.0000**, follow-histogram **L 0 / S 240 / R 0** (`HPP0_CONFOUND_AUDIT.md:148-152`).

**This is the failure H2 must not repeat, and §6 is written to be immune to it by construction.**

### C.3 Reusable labels — the short list

| label | coverage | usable for H2? |
|---|---|---|
| `LATMANEUVER` (7 kinematic lat tokens, factorized) | **100 %**, active 28.55 % | ✅ **yes** — already factorized, the right shape for the BEHAVIOUR factor |
| `LONMODE` (6 lon tokens, factorized) | **100 %**, active 25.27 % | ✅ **yes**, but `follow_lead`/`close_gap`/`open_gap` **never emit** (the `lead_state` stub) — `obstacle.offline` fixes this |
| `route_target_v21` (+`ROUTE_UNKNOWN` sentinel) | **80.43 % train / 79.35 % val** | 🟡 non-circular, but **model input** — usable as a covariate, **never** as the sensor-request target |
| v3 ROUTE token (9-token frozen) | 80.70 % / 79.58 % | ⚠️ **4 of 9 tokens never minted** (`straight`, `exit_left`, `exit_right`, `merge`); `u_turn` **roundabout-confounded on 24 windows** |
| `route_target` v1 | 27.2 % | ❌ **CIRCULAR** — do not touch |
| 5-way maneuver v1/v2 | 100 % | ❌ mixed-axis; see C.1 |
| VLM `road_geometry` (8-enum, incl. junction/roundabout) | **24 episodes / 30 windows in-repo** | 🟡 right *shape*, unusable *scale*; ROUTE direction measured **at chance** (0.4773, CI [0.268, 0.702]) |
| VLM `INTERACT` / `SIGNAL` | deleted | ❌ ~0 % informative — a forward camera cannot see a blinker |

**Net:** H2 should reuse the **factorized** `LATMANEUVER`/`LONMODE` for the BEHAVIOUR factor and mint
its own SITUATION and SENSOR_REQUEST factors. Nothing existing supplies a sensor-request label.

---

## 5. SECTION E — SITUATION FREQUENCY (the power question)

### E.1 ⚠️ Correcting the framing: the corpus is NOT US-highway-weighted

**MEASURED** (`r0/phase0_selection.parquet`, all 3,000 clips): the corpus is **deliberately
country-stratified at ~121 clips per country across ~25 countries** — Germany 121, France 121,
Belgium 121, Denmark 121, Austria 121, Finland 121, Greece 121, Hungary 121, Latvia 121,
Lithuania 121, … United States 118. Selection was by an **urban-interaction score** (moderate speed
band + stop fraction + yaw activity, with hard gates at 2–14 m/s — `physicalai_r0.py:99-108`).

**The PI is right and my brief's premise was wrong: this is a Europe-heavy, urban-weighted corpus,
not a US highway corpus.** That framing is retracted.

### E.2 Situation counts — MEASURED over the FULL corpus, episode AND window granularity

**Method:** kinematic detection on `egomotion` (10 Hz resample, `np.unwrap`ped yaw from quaternion),
**all 197 local chunks, all 3,000 phase-0 clips, 485,401 windows** (windows = 10 s, stride 0.5 s).
**Command:** `scratchpad/situ_full.py` → `scratchpad/situations_full.parquet`.

| situation | detector | **episodes** | ep % | **windows** | win % | est. of 2,376 parity eps |
|---|---|---|---|---|---|---|
| **turn** | \|Δψ\| ≥ 45° over 8 s | **1,746** | 58.20 % | 51,824 | 10.68 % | ~1,383 |
| **intersection / junction** | \|Δψ\| ≥ 45° over 6 s **and** turn radius ≤ 30 m (tight transient) | **846** | **28.20 %** | **8,103** | 1.67 % | **~670** |
| **lane change** | S-shaped yaw (both lobes ≥ 2°), net \|Δψ\| ≤ 3°, sustained lateral 2.5–5.0 m, v ≥ 8 m/s | **1,172** | **39.07 %** | **3,621** | 0.75 % | **~928** |
| **roundabout (loose)** | \|Δψ\| ≥ 135°, R ≤ 30 m, monotone sign > 0.85 | **105** | 3.50 % | 517 | 0.11 % | ~83 |
| **roundabout (strict)** | \|Δψ\| ≥ 180°, 6 ≤ R ≤ 25 m, monotone > 0.90, 3 ≤ v ≤ 11 m/s | **19** | **0.63 %** | **73** | 0.02 % | **~15** |

### E.3 Reading — two of three target situations are powered, one is not

- ✅ **Intersections: 846 episodes / 8,103 windows.** Richly powered. ~670 on the parity corpus,
  **16× the n ≥ 40 episode-cluster bar.**
- ✅ **Lane changes: 1,172 episodes / 3,621 windows.** Richly powered (~928 parity episodes).
- ❌ **Roundabouts: 19 episodes strict, 105 loose.** **BELOW the bar even at the loose threshold**
  once you account for the loose detector's confound. This corroborates an **independent prior
  measurement**: route-v3 minted `roundabout` on **8 of 2,201 windows, from ONE episode** (0.36 %,
  INHERITED, `DATA_STRATEGY §3.2`) — two unrelated methods agree that roundabouts are ~0.5 % of this
  corpus.

⚠️ **Honesty note on the loose roundabout detector:** it fires on any ≥135° sustained same-sign turn
at R ≤ 30 m, which also matches U-turns and tight slip roads. The program has the *mirror* of this
confound already on record (route-v3's `u_turn` is documented **roundabout-confounded**). I did not
stretch the definition to manufacture coverage; the strict number (19) is the one to plan against,
and even the loose number (105) is not decision-grade.

**Consequence:** H2 phase 1 should be scoped to **intersections and lane changes**, which are the two
situations the AV corpus actually supports. **Roundabouts cannot be trained or evaluated on
PhysicalAI-AV at decision grade.** Per PI clarification #5 this is not a blocker — it is a concrete,
quantified requirement to hand the sibling agent surveying external multi-camera datasets:
**we need ≥ 40 roundabout episodes with ≥ 2 calibrated cameras.**

---

## 6. SECTION D — THE CENTRAL QUESTION: A NON-CIRCULAR LABEL

### 6.1 Why the obvious label is forbidden — and why this corpus protects us

The failure to avoid is on the record: `route_target = _NAV_TO_ROUTE[nav_cmd]` made the strategic
brain predict its own input, so `route_skill` was **0.0 by construction** (§C.2). A label of the form
*"roundabout ⇒ activate left camera"* is **the identical error**: it is a deterministic function of a
situation class we also feed the model, so a head trained on it learns a lookup table and measures
nothing.

The recommended label below never touches a model input. Its three ingredients are (a) 3D agent
geometry from `obstacle.offline`, (b) per-camera calibration, and (c) the ego's **realised future
dynamics** — none of which is available to the model at decision time. **The model sees only the
front-camera crop and its own speed** (PI clarification #2).

### 6.2 ✅ RECOMMENDED LABEL — `L1-TanitAD`, written to be implementable

Notation: episode `e`; 10 Hz grid `t`; camera `X ∈ {cross_left, cross_right}`; agent tracks `a`
resampled per-track onto the grid with a **0.5 s max-gap guard** (mandatory — see §6.6).

```
CanonicalCrop(t)  := the c×c native-pixel box centred on the clip's per-clip (cx, cy) of
                     camera_front_wide_120fov, with
                        c = 2 · r_ftheta(atan(128/266))        [calib.ftheta_crop_size]
                     i.e. EXACTLY the pixels the encoder receives.  (±25.697° half-angle)

p_ego_CV(t+h)     := the ego's CONSTANT-SPEED, CONSTANT-HEADING continuation from t,
                     p_ego(t) + h · v(t) · [cos ψ(t), sin ψ(t)]          h ∈ (0, 4.0 s]

L1_gate(X, t) = 1  iff  ∃ agent a such that ALL of:
  (i)   OUT-OF-INPUT   proj_front_wide(a, t) ∉ CanonicalCrop(t)
  (ii)  IN-X           proj_X(a, t) ∈ frame(X)      [0≤u<W, 0≤v<H, θ < θ_max(X)]
  (iii) CONFLICT       min_{h ∈ (0,4s]} ‖ p_a(t+h) − p_ego_CV(t+h) ‖ ≤ 3.0 m
  (iv)  NON-REDUNDANT  no agent satisfying (iii) lies inside CanonicalCrop(t)

L1_label(X, t) = L1_gate(X, t)  AND  ( v(t + 4.0 s) − v(t) ≤ −1.0 m/s )     [ego yielded]
```

**Projection** is the real f-theta forward map, already implemented and tested in-repo:
`stack/tanitad/data/calib.py:273 ftheta_project_ray`. Rig→camera is
`p_cam = R(q)ᵀ · (p_rig − t)` from `sensor_extrinsics`. `obstacle.offline` is already in the rig
frame, so **no transform chain is needed** beyond this single step.

#### Why each clause is there

| clause | what it buys | what breaks without it |
|---|---|---|
| **(i)** | restricts to information the encoder genuinely lacks | you label things the model already sees |
| **(ii)** | makes it a *specific camera* request, not "something is out there" | the output is not actionable as a tool call |
| **(iii) counterfactual** | **decision-relevance instead of mere presence** | presence alone fires on **63.45 %** of frames (MEASURED) — a useless label |
| **(iii) uses `p_ego_CV`, not the realised path** | ⚠️ **a successful yield produces no close encounter.** Scoring conflict on the realised trajectory systematically deletes exactly the positives we want | the label is anti-correlated with the behaviour it should predict |
| **(iv)** | enforces *unique* information | you reward requesting a camera for an agent already visible in front |
| **(v)** in `L1_label` | confirms the agent actually constrained the ego | positives include harmless passers-by |

**Use `L1_gate` as the training target** (it is the *activation* decision and does not condition on
the future response); use `L1_label` as a **high-precision evaluation slice**.

### 6.3 Coverage and class balance — MEASURED

**Sample:** 80 episodes, 2 chunks (0036 US, 0170), **668,513 agent-frames**, 12,489 frames.
Limited by *local calibration availability* (30 chunks), not by data existence.
**Commands:** `scratchpad/crux.py`, `crux3.py`, `an1.py`, `an2.py`, `an4.py`, `an5.py`.

**Leg-by-leg attrition (agent-frames):**

| leg | rate | n |
|---|---|---|
| (i) outside the model's front crop | **74.00 %** | 360,289 |
| (i)+(ii) left-camera-only | 17.35 % | 84,453 |
| (i)+(ii) + within 40 m | 6.51 % | 31,685 |
| **(i)+(ii)+(iii) conflict course** | **0.01 %** | 69 |

**Frame-level class balance (the numbers to plan against):**

| quantity | value |
|---|---|
| **`L1_gate` positive frames** | **1.83 %** (228 / 12,489) |
| **`L1_label` positive frames** | **0.91 %** (114 / 12,489) |
| gate-positive, **left** camera | 0.50 % (62 frames, 9 episodes) |
| gate-positive, **right** camera | 1.46 % (182 frames, 16 episodes) |
| **episodes with ≥1 gate-positive frame** | **21 / 80 = 26.2 %** |
| **episodes with ≥1 label-positive frame** | 11 / 80 = 13.8 % |

**Power at corpus scale (ESTIMATED, linear extrapolation in episodes):**

| | est. of 2,376 parity episodes | vs the n ≥ 40 episode-cluster bar |
|---|---|---|
| gate-positive episodes | **≈ 624** | ✅ **15.6×** |
| label-positive episodes | **≈ 327** | ✅ **8.2×** |

**✅ The label is rare per-frame but abundant per-corpus. Power is not the problem.**

Sanity check passed: 100.0 % of "left-only" agents have **positive** (left) rig azimuth, median
68.5° — the geometry is not mirrored or sign-flipped.

### 6.4 ⚠️ THE CRUX CAVEAT — decision-relevance is threshold-critical

The question that decides the workstream: **does an unseen conflicting agent actually predict that
the ego yields?** Measured as a lift, `P(ego decelerates ≥1 m/s over 4 s | gate) / P(· | ¬gate)`,
with a **80-episode-cluster bootstrap** (2,000 resamples — the program's decision-grade estimator per
`CLAUDE.md`), sweeping only the conflict radius:

| conflict radius `d` | gate-positive | n+ | P(resp \| +) | P(resp \| −) | **lift** | **cluster-boot 95 % CI** |
|---|---|---|---|---|---|---|
| 2.0 m | 0.25 % | 31 | 38.71 % | 23.02 % | 1.68× | [0.00, 4.51] |
| **3.0 m** | **1.83 %** | **228** | **50.00 %** | **22.56 %** | **2.22×** | **[1.30, 3.14]** ✅ |
| 4.0 m | 5.89 % | 736 | 20.38 % | 23.23 % | 0.88× | [0.56, 1.30] — null |
| 5.0 m | 8.31 % | 1,038 | 14.35 % | 23.85 % | 0.60× | [0.34, 0.95] ❌ |
| 6.0 m | 10.18 % | 1,272 | 10.46 % | 24.49 % | 0.43× | [0.24, 0.71] ❌ |
| 8.0 m | 13.44 % | 1,679 | 13.52 % | 24.54 % | 0.55× | [0.32, 0.84] ❌ |

> **The sign of the effect flips with the threshold, and CIs on both sides exclude 1.0.**

This is the same class of error `CLAUDE.md` records for learning-curve exponents (*"the same log gives
−0.387 / −0.505 / −0.564 / −0.621 / −0.738 depending on the window"*). **I will not headline 2.22×.**

**There is a physical mechanism, which is why this is a caveat and not a refutation.** Measured agent
`size_y` is ~1.9–2.3 m, so two vehicle *centres* within 3.0 m are on a genuine collision course.
Between 3.5 m and 6 m the centres are one **lane width** apart (3.0–3.5 m) — i.e. ordinary
adjacent-lane traffic, which correlates with *free-flowing* driving and therefore *less*
deceleration. The flip is exactly what that mechanism predicts. **3.0 m is defensible a priori as the
vehicle-footprint radius** — but it was selected after seeing the sweep, so it is not yet admissible.

**The probe that closes this (required before any GPU spend):**

> Download `calibration/` for **10 chunks not in the local 30**, recompute `L1_gate` at
> **d = 3.0 m only** (pre-registered, no sweep), and test the lift on those held-out episodes.
> **PASS** = lift ≥ 1.5× with an episode-cluster CI excluding 1.0. **FAIL** = do not build L1;
> fall back to §6.7.
> **Cost: ~2 CPU-hours + ~1 GB download. No GPU. Both outcomes committed in advance.**

### 6.5 ✅ THE EFFICIENCY CLAIM (C-EFF) — promoted to primary, and it is strong

**What fraction of frames genuinely require ≥2 cameras under L1?** (MEASURED, `scratchpad/eff.py`)

| activation policy | left | right | **either** | **mean cameras/frame** |
|---|---|---|---|---|
| instantaneous (no hysteresis) | 0.50 % | 1.46 % | **1.83 %** | **1.020** |
| ±1.0 s hysteresis | 2.39 % | 5.22 % | **6.85 %** | **1.076** |
| ±2.0 s hysteresis | 4.23 % | 8.35 % | **11.18 %** | **1.126** |

Encoder cost is **linear in camera count** (one ViT forward per view; the 4-brain trunk runs once).

| policy | vs always-on **7 cameras** | vs always-on **3** (front+L+R) |
|---|---|---|
| instantaneous | **85.4 % saved** | 66.0 % saved |
| ±1 s hysteresis | **84.6 % saved** | 64.1 % saved |
| ±2 s hysteresis | **83.9 % saved** | 62.5 % saved |

> **The efficiency claim is MEASURED and strong: even with a conservative ±2 s hysteresis, selective
> activation costs 1.13 cameras/frame against 7 — an 84 % saving on multi-camera encoder compute.**
> The PI's hypothesis that "most frames need only the front camera" is **confirmed**: 88.8 % of frames
> need exactly one camera even under the most generous policy measured.

Per-camera cost anchor (INHERITED, from the frozen-WM audit): the flagship encoder+readout is
**87,121,280 params** of the 263 M total, and encoding 412 episodes took **296 s** on an A6000
(≈1.4 ep/s). A second camera therefore adds ~33 % to total model FLOPs when active — which, at 1.83 %
duty cycle, is **~0.6 % amortised**.

### 6.6 ⚠️ Two implementation traps, both already paid for once

1. **`obstacle.offline` rows are ASYNCHRONOUS, not per-frame.** Each row carries its own
   `timestamp_us`; naively `groupby("timestamp_us")` yields ~1 agent per "frame" and makes the labels
   look worthless. **You must resample per-track onto the 10 Hz grid with an explicit max-gap guard**
   (0.5 s used here). This trap cost the prior investigation one iteration; my `resample_tracks()`
   (`scratchpad/crux.py`) implements it correctly and is reusable.
2. **`source = scene:obstacles:autolabels:v2` — these are machine labels, not human GT.** Stamp them
   `prov: "autolabel"`, never `"human"`.

### 6.7 If the §6.4 probe FAILS — the closest computable alternative

**Do not fall back to a rule lookup.** The nearest non-circular alternative, in preference order:

1. **`L1-occlusion`** — extend clause (i) from *out-of-crop* to *out-of-crop **or occluded***. All 3D
   boxes are known, so occlusion is a painter's-algorithm depth sort **among the boxes themselves** —
   no new data, ~1 CPU-day. This is a strict superset of the current positives and should raise the
   rate materially. **⚠️ This is a real gap in the present measurement: my clause (i) is
   out-of-FOV/out-of-crop ONLY; I did not compute occlusion.** The brief asked for both.
2. **`L1-lateral`** — replace the deceleration response (v) with a *lateral* response (aborted or
   deferred lane change), detected by the S-shape detector in §E.2 failing to complete. Lane changes
   are 39 % of episodes, so this targets a well-populated stratum the speed-based response misses.
3. **`L2` oracle ablation** (per `H2_DESIGN_FRAMING.md`) — train with and without camera X and label
   by measured per-window improvement. This is the true ceiling and the ideal *validation* signal, but
   it costs n_models × n_cameras and is not a training target.

---

## 7. SECTION D-bis — CAN A FRONT-CAMERA-ONLY OBSERVER KNOW? (PI clarification #2)

**This is the question the PI's architectural constraint raises, and it deserves a direct answer.**

**Method:** predict `L1_gate` / `L1_label` from **front-camera-derivable observables only** —
ego speed, and the count/geometry of agents *inside the model's actual 256 px crop* (n visible, min
range, mean and max |azimuth|, azimuth spread, n within 30 m, n within 15 m). Gradient-boosted trees,
**5-fold episode-GROUPED CV** (no episode spans folds), plus a label-shuffle control.
**Command:** `scratchpad/decid.py`.

| target | base rate | **ROC-AUC** | per-fold | PR-AUC (lift) | shuffle control |
|---|---|---|---|---|---|
| `L1_gate` | 1.83 % | **0.650** | [0.742, 0.601, 0.638, 0.700, 0.570] | 0.0311 (**1.70×**) | 0.564 |
| `L1_label` | 0.91 % | **0.685** | [0.591, 0.638, 0.769, 0.865, 0.560] | 0.0155 (**1.69×**) | 0.462 |

**Answer: YES, there is front-camera-visible evidence — but it is weak.** AUC 0.65–0.69 against a
shuffle band of 0.46–0.56. The need is **partially** anticipatable, not reliably so. Per-fold variance
is large (0.56–0.87), so at n = 80 episodes this is **suggestive, not decision-grade**.

**Three qualifications that all point the same way — this is a LOWER BOUND:**

1. My features are **proxies for front-camera content** (agents inside the crop, derived from 3D
   geometry), **not encoder features**. The real encoder additionally sees road geometry, junction
   layout, lane markings, kerb lines, signage and other cars' poses — all strong junction cues that
   my feature set entirely lacks.
2. No temporal context: single-frame features only, while the model gets a 0.8 s window.
3. 80 episodes.

**This is the single most important open question in H2, and it is cheap to settle** — see U-1 in §9.
If the true (encoder-feature) AUC is ≥0.75, the workstream is well-founded. If it sits at ~0.65, the
model can only react, not anticipate, and the design should be re-scoped to a *reactive* sensor
request — which is still useful and still buys the efficiency claim, but is a materially weaker
scientific claim than "estimates the semantic need."

**I want to state the negative case plainly, per the coordinator's instruction:** if the need for a
side camera is driven by agents that are genuinely invisible and unhinted from the front view, then
no front-camera-only model can predict it, and the honest result is that the *request* must be driven
by **situation class** (junction/lane-change, which IS front-visible) rather than by **agent
presence**. That would be a real and publishable negative result about the limits of the design, and
the measurement above does not yet exclude it.

---

## 8. SECTION F — THE FROZEN-WM INTERFACE

Established by a dedicated sub-audit against `MODEL_REGISTRY.md` and committed source. **flagship-v1
is `flagship4b-speedjerk-30k`** (`action_dim=3`, ADE@2s 0.4522 heldout) — **not**
`flagship4b-phase0-30k`, which is the no-speed ablation control (2.9176).

### F.1 Tap points a tactical head can consume (MEASURED, code-read)

| tap | tensor | where | how |
|---|---|---|---|
| **T1** encoder token grid | `[B, 256, 768]` (16×16) | `models/encoder.py:55-68` | `model.encode_tokens(frames)` |
| **T2** compact state | `[B, 2048]` (4×4×128) | `models/readout.py:32-37` | `model.encode(frames)` |
| **T2b** spatial re-expand | `[B, 16, 128]`/frame | `flagship_v15.py:97-99` | `states.reshape(B, W*16, 128)` |
| **T3** ⭐ **temporal window** | **`[B, 8, 2048]`** = 0.8 s | `fourbrain.py:473-477` | `model.encode_window(frames)` |
| **T6** rollout latents | `list[(z_prev, z_hat)] × K` | `metric_dynamics.py:247-266` | `rollout_transitions(...)` |
| **T9** strategic ctx | `{ctx:[B,256], route_logits:[B,3]}` | `fourbrain.py:73-86` | `model.strategic_policy(...)` |
| **T10** tactical policy | `maneuver_logits [B,5]`, `intent [B,256]` | `fourbrain.py:328-361` | `model.tactical_policy(states, ctx)` |

**Attach the H2 head at T3 `[B, 8, 2048]`** — the tap every existing brain uses; the policy brains are
explicitly state-dim-agnostic (`fourbrain.py:53, 293`).

**Loader** (dev-box safe): `stack/scripts/v15_prep.py:56 load_frozen_v1(ckpt, device)` — strict load,
`requires_grad_(False)`, and it **raises `SystemExit` if `act_emb.0.weight.shape[1] != 3`**, i.e. it
fails loud on the phase0/speedjerk inversion. Use it.
*(Do not use `taniteval/loaders.py` — it hard-codes `/root/TanitAD/stack`.)*

### F.2 ⚠️ Imagination at >1 step — the "σ dies by k=4" claim is TRUE but about a DIFFERENT module

This distinction is load-bearing for H2's anticipation design.

- **H15 `ImaginationField`** (tokens fed back on themselves, no actions, no re-observation):
  fidelity **dies at k=3** on the deployed lineage (cos 0.232 → 0.016 ≈ chance at k=3, −0.035 at k=4)
  **while σ falls** — false confidence. Program rule: **cap every deployed use of the variance field
  at 1 step.** `freeze-1` (run once at k=1, hold) stays flat and beats persistence 7×.
- **Operative predictor rollout under supplied actions** (`rollout_transitions` over the 2048-d
  state): **fully usable to K = 20** (2 s). `op_abs_long_by_wp = [0.065, 0.203, 0.480, 0.836] m` at
  {0.5, 1, 1.5, 2} s; ADE 0.4271 vs CV 0.8377.

> **For H2: anticipation via the action-conditioned rollout is available to 2 s. Anticipation via the
> H15 variance field is NOT available beyond 1 step.** If the design wants to foresee a situation
> before it arrives, use the former.

⚠️ Both σ-dissipation runs are at step 6,500 and 19k; **the 30k re-run has no artifact in the repo.**

### F.3 The decisive precedent — and its warning

**`FlagshipV15Head`** (`stack/tanitad/models/flagship_v15.py` + `train_flagship_v15.py`) is a
committed, frozen-trunk head — the closest existing analogue to H2's head. Its `imagine_probes()`
(`:504-538`) is **the shipped way to feed multi-step consequences into a new head**: 8 FPS-sampled
action probes × 20 steps, latents read at (5,10,15,20), `v0` held at the observed value.

> ⚠️ **The warning H2 must heed.** In the frozen-WM planner bank: a head that regresses geometry
> **directly off the static latent** measured **3.649 m** — reproducing the REF-A frozen-encoder
> ceiling. The *same* head routed **through the frozen dynamics** measured **0.599 m**, statistically
> indistinguishable from the WM's own oracle-action ceiling (0.4045). **Route through T6/T7, not off
> T2.**

**Cost (MEASURED):** encode 400 episodes ≈ **296 s** on one A6000 (~0.2 GB cache); train a 3.8 M head
on 8,803 windows ≈ **853 s**. **The cheapest experiment class in the program.**

**Keep the encoder frozen.** The one in-family fine-tune (v1.6, 4 blocks + predictor) cost
**+144 % on the WM canary** (0.452 → 1.1022) for an ADE delta of **+0.0104 m, CI [−0.0888, +0.1147] —
NOT separated**. The `feat_cos 0.966` safety result is **REF-C's ResNet encoder, not flagship's ViT**,
and even there it was not a net win.

---

## 9. WHAT I COULD NOT DETERMINE — and the exact probe that settles each

| # | Unknown | Exact probe | Cost |
|---|---|---|---|
| **U-1** ⭐ | **Whether the frozen v1 latent can actually predict `L1_gate`.** §7 used geometric proxies, not encoder features. This is the highest-value open question in H2. | Encode 40 val episodes with `v15_prep.py` (~0.2 GB), fit a ridge/MLP probe from `z [N,2048]` → `L1_gate`. Report **PR-AUC vs base rate** with an episode-cluster bootstrap. | ~30 GPU-min, **no training** |
| **U-2** ⭐ | **Out-of-sample confirmation of the 3.0 m lift** (§6.4). The one thing standing between GO-WITH-CAVEAT and GO. | 10 held-out chunks' `calibration/`, recompute at d = 3.0 m **only**, pre-registered. | ~2 CPU-h, ~1 GB, **no GPU** |
| **U-3** | **Occlusion.** Clause (i) is out-of-FOV/out-of-crop only; I did **not** compute occluded-but-in-FOV. The brief asked for both. | Painter's-algorithm depth sort among the known 3D boxes in each camera frame. No new data. | ~1 CPU-day |
| **U-4** | **Corpus-wide L1 rate.** 80 episodes / 2 chunks / 2 countries. Density is known to vary ~2× by region (45.2 % US / 53.4 % DE / 28.0 % SK lead presence). | Pull `calibration/` + `obstacle.offline` for the remaining phase-0 chunks; rerun `crux3.py`. | ~12 GB, ~1 CPU-day |
| ~~U-5~~ | ~~Whether the rig split affects cross-camera *extrinsics*.~~ **CLOSED** — measured, see §A.3: extrinsics differ by rig in elevation (≤2.5°) and mount height (≤16 cm), but azimuth is nearly rig-invariant (≤0.2° on the cross cameras). L1's azimuthal frustum test is robust to rig. | — | done |
| **U-6** | **True-future latent fidelity per k** for `flagship-30k`. The published `zcos_by_step` is cos(ẑ_k, z_t) — a *commitment* metric, not fidelity. No `latent_fidelity` artifact exists. | `taniteval/imagination.py` already emits `latent_fidelity`; extend it to a per-k series. | one eval pass |
| **U-7** | Camera **video** decode cost per view (I measured presence, calibration and timestamps, not decode throughput). Relevant to the embedded efficiency claim end-to-end. | Decode 10 clips of `cross_left` and time it. | ~20 CPU-min |

---

## 10. RECOMMENDATION

1. **Run U-2 and U-1 before anything else.** Together ~30 GPU-min + 2 CPU-hours, and they decide
   whether H2 is a strong or a weak claim. Pre-register both outcomes.
2. **Scope phase 1 to intersections + lane changes.** Both are richly powered (846 / 1,172 episodes).
   **Roundabouts are not available at decision grade on this corpus (19–105 episodes)** — hand that
   requirement to the external-dataset survey rather than stretching a definition.
3. **Build the head as three independent factors** — `SITUATION` (softmax) × `BEHAVIOUR` (reuse the
   factorized `LATMANEUVER`/`LONMODE`) × `SENSOR_REQUEST` (**per-camera independent Bernoulli**).
   Never one softmax over mixed axes (§C.1).
4. **Attach at T3, route through the dynamics (T6/T7), keep the encoder frozen** (§F.3).
5. **Lead with the efficiency claim.** It is MEASURED, strong (84 % saving), and — unlike the
   decision-relevance lift — not threshold-fragile.

---

### Provenance

All scratch scripts and intermediate parquets:
`C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\`
— `probe_a.py` · `probe_calib.py` · `probe_geom.py` · `crux.py` · `crux2.py` · `crux3.py` ·
`an1.py` · `an2.py` · `an3.py` · `an4.py` · `an5.py` · `decid.py` · `eff.py` · `situ.py` ·
`situ2.py` · `situ3.py` · `situ_full.py`; outputs `vis.parquet`, `crux2.parquet`, `crux3.parquet`,
`situations_full.parquet`, `feature_presence_phase0.csv`.

Data read (read-only, dev box, **no pod touched, no GPU used**):
`C:\Users\Admin\tanitad-data\physicalai\` — `features.csv`, `metadata/feature_presence.parquet`,
`r0/phase0_selection.parquet`, `calibration/camera_intrinsics/*.parquet` (30 chunks),
`calibration/sensor_extrinsics/*.parquet` (30 chunks), `labels/obstacle.offline/*.zip` (2 chunks
usable), `labels/egomotion/*.zip` (197 chunks), `r0/camera_front_wide/*.timestamps.parquet`.

**Not staged.** Per instruction, no `git add` was performed.
