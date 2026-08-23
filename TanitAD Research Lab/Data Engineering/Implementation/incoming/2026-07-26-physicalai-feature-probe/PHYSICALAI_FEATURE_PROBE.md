# PhysicalAI-AV — the 34 features we do not read

**Date:** 2026-07-26 · **Author:** Data Engineering · **Status:** probe complete, files written, **NOT staged** (per brief).
**Repo probed:** `nvidia/PhysicalAI-Autonomous-Vehicles`, revision `main` @ `b719eea7f0a63619ef51ec7f54178af0937ef050`, `gated=auto`.
**Cost:** ~34 MB of metadata + 2 label chunks (65 MB). **No pod touched. No GPU. No bulk download.**
**Handling:** PhysicalAI-AV is `gated-confidential`. Nothing in this directory contains clip content — only
schemas, enum names, counts and aggregates. Gated bytes were cached **outside the repo**
(`%TEMP%\claude\pai_probe_cache`).

---

## 0. TL;DR — the answer, and it is a clean NO on the thing that mattered

**There is no map, lane geometry, lane graph, routable topology, junction annotation, roundabout label,
traffic-light feature, or route/goal signal anywhere in PhysicalAI-AV.** This is now established at
**five independent locations**, not one:

| # | Probe | Result | Evidence |
|---|---|---|---|
| 1 | The dataset's **own feature manifest** `features.csv` | **exactly 36 rows.** 7 camera + 6 calibration + 3 label + 1 lidar + 19 radar. No map/lane/light/junction row | **MEASURED** — `pai_features.csv` |
| 2 | **HF tree API**, L1→L3, every directory | 7 top-level groups, 36 feature dirs, no others | **MEASURED** — `pai_tree_l{1,2,3_sample}.json` |
| 3 | The **dataset card, verbatim** | *"we do not include open maps data. Scenes are not compatible with CARLA unless the user generates their own XODR data for now."* | **PUBLISHED** — `pai_card.md:303` |
| 4 | **`obstacle.offline` class enum**, measured on 12 clips / 87,481 rows | **10 classes, all dynamic agents**: `automobile, person, heavy_truck, trailer, bus, rider, other_vehicle, protruding_object, stroller, animal`. No traffic light, no sign, no static infrastructure | **MEASURED** — `pai_label_schemas.json` |
| 5 | **Both tagged revisions** (`25.10`, `26.03`) | identical top-level layout; nothing was removed that we could recover | **MEASURED** — `pai_sizes_and_revs.json` |

⚠️ **A sixth finding that closes a standing open lead.** The NC-ingest survey listed
*"OpenStreetMap map-matching onto our ego GPS traces"* as one of three candidate commercially-clean
lane-graph paths. **That path is dead for this corpus.** `labels/egomotion` carries
`timestamp, qx,qy,qz,qw, x,y,z, vx,vy,vz, ax,ay,az, curvature` and **nothing else** — coordinates are
**clip-local metres** with origin at t=0 (measured max |x| 240 m, |y| 254 m over a 20 s clip). There is
**no latitude, no longitude, no GNSS, no city** anywhere in the corpus; `data_collection.parquet` stops
at **country**. Without a global position there is nothing to map-match. *(MEASURED, this session;
card §Labels corroborates: "local coordinate frame … origin located at the ego vehicle's position at
timestamp 0".)*

**But four things we do not read are genuinely valuable**, and one of them is nearly free — see §5.

---

## 1. The complete 36-feature table

Per-clip coverage is the **exact** count from `metadata/feature_presence.parquet` (306,152 × 36 booleans).
Size = one measured chunk × the measured chunk count; **ESTIMATED** at the corpus level (the sum comes to
150.7 TB vs the card's PUBLISHED 133 TB, so read sizes as ±15 %, ordering-accurate).
`✅ read` = consumed by the production pipeline. Per-clip ≈ chunk ÷ ~97 clips.

### Group A — labels (3)

| # | Feature | Schema (MEASURED) | Coverage | Chunk | Corpus | Read? |
|--:|---|---|---:|---:|---:|:--:|
| 1 | `egomotion` | `timestamp, qx,qy,qz,qw, x,y,z, vx,vy,vz, ax,ay,az, curvature` — 15 cols, ~2,224 rows/clip (~111 Hz), clip-local | **100.00 %** | 36.8 MB | 0.12 TB | ✅ |
| 2 | `egomotion.offline` | `timestamp, qx,qy,qz,qw, x,y,z` — **8 cols only** (pose, no velocity/accel/curvature), ~202 rows/clip | 97.44 % | 1.6 MB | 0.005 TB | ❌ |
| 3 | `obstacle.offline` | `timestamp_us, source, track_id, center_{x,y,z}, size_{x,y,z}, orientation_{x,y,z,w}, label_class, reference_frame, reference_frame_timestamp_us` — **3D tracked cuboids**, `source='scene:obstacles:autolabels:v2'`, `reference_frame='rig'`, ~7,290 rows & ~39 tracks per clip | 97.44 % | 63.7 MB | 0.20 TB | ❌ **⭐** |

### Group B — calibration (6)

| # | Feature | Schema (MEASURED) | Coverage | Chunk | Corpus | Read? |
|--:|---|---|---:|---:|---:|:--:|
| 4 | `camera_intrinsics` | per-clip × 7 cams, f-theta `poly`, `cx`, `cy`, `width`, `height` | **100.00 %** | 0.04 MB | ~0 | ✅ |
| 5 | `camera_intrinsics.offline` | `clip_id, camera_name, model_type='ftheta', model_parameters (JSON), ` **`ego_mask_image_png`** | 97.44 % | 0.2 MB | 0.001 TB | ❌ **⭐** |
| 6 | `lidar_intrinsics.offline` | `clip_id, lidar_name, model_type='row-offset-spinning', model_parameters` (128 rows × 3600 cols, 10 Hz, per-row elevations) | 97.44 % | 5.2 MB | 0.016 TB | ❌ |
| 7 | `sensor_extrinsics` | per-clip × 8 sensors, `qx..qw, x,y,z` | **100.00 %** | 0.1 MB | ~0 | ✅ |
| 8 | `sensor_extrinsics.offline` | same, 8 sensors (7 cam + lidar) | 97.44 % | 0.1 MB | ~0 | ❌ |
| 9 | `vehicle_dimensions` | `length, width, height, rear_axle_to_bbox_center, ` **`wheelbase`** `, track_width` — only 3 distinct vehicles | **100.00 %** | 0.03 MB | ~0 | ⚠️ probe only |

### Group C — camera (7) · all **100.00 %**, ~6–8 TB each

| # | Feature | Contents | Chunk | Corpus | Read? |
|--:|---|---|---:|---:|:--:|
| 10 | `camera_front_wide_120fov` | mp4 1080p30 + `timestamps.parquet` + `blurred_boxes.parquet` | 2,051 MB | 6.45 TB | ✅ |
| 11 | `camera_front_tele_30fov` | idem | 1,627 MB | 5.12 TB | ❌ |
| 12 | `camera_cross_left_120fov` | idem | 2,261 MB | 7.11 TB | ❌ |
| 13 | `camera_cross_right_120fov` | idem | 2,521 MB | 7.93 TB | ❌ |
| 14 | `camera_rear_left_70fov` | idem | 2,130 MB | 6.70 TB | ❌ |
| 15 | `camera_rear_right_70fov` | idem | 2,392 MB | 7.53 TB | ❌ |
| 16 | `camera_rear_tele_30fov` | idem | 2,001 MB | 6.29 TB | ❌ |

⭐ Note the **third per-camera artefact nobody has mentioned: `blurred_boxes.parquet`** — the
anonymisation boxes (faces/plates), shipped alongside every camera in every chunk. Free 2D
person/vehicle evidence, already inside the zips we download.

### Group D — lidar (1)

| # | Feature | Contents | Coverage | Chunk | Corpus | Read? |
|--:|---|---|---:|---:|---:|:--:|
| 17 | `lidar_top_360fov` | Draco-compressed point clouds, ~200 spins/clip @10 Hz | 97.44 % | **32,340 MB** | **101.7 TB** | ❌ |

**The lidar is 68 % of the entire dataset by bytes.** Any "download more of PhysicalAI" plan must
exclude it explicitly.

### Group E — radar (19) — the bucket nobody had opened, now decoded

The 19 radars are **three mutually-exclusive rig configurations**, and the arithmetic closes exactly
against `data_collection.radar_config`:

| config | clips | radars present | which |
|---|---:|---:|---|
| `low` | 87,197 (28.48 %) | **9** | all nine `*_srr_0` |
| `med` | 50,747 (16.58 %) | **8** | 4× corner `srr_3` + `front_center_{imaging_lrr_1, mrr_2}` + `rear_{left,right}_mrr_2` |
| `high` | 22,817 (7.45 %) | **10** | the `med` 8 **+** `side_{left,right}_srr_3` |
| `nan` | 145,391 (47.49 %) | 0 | no radar |

Check: 87,197 + 50,747 + 22,817 = **160,761** = the card's PUBLISHED radar-clip count. ✔
This is what "up to 10 radar" means. All 19 share one schema: `{scans: <clip>.parquet}` 3D radar point
clouds. Coverage per feature is therefore **28.48 %** (the nine `srr_0`), **24.03 %** (the eight
`med`+`high`), or **7.45 %** (the two side `srr_3`). Corpus size **1.46 TB total across all 19** —
the whole radar suite is cheaper than one camera.

| # | Feature | Coverage | Chunk | Read? |
|--:|---|---:|---:|:--:|
| 18–26 | `radar_{corner_front_left, corner_front_right, corner_rear_left, corner_rear_right, front_center, rear_left, rear_right, side_left, side_right}_srr_0` | 28.48 % | 56–78 MB | ❌ |
| 27–30 | `radar_corner_{front_left, front_right, rear_left, rear_right}_srr_3` | 24.03 % | 68–75 MB | ❌ |
| 31 | `radar_front_center_imaging_lrr_1` | 24.03 % | 215 MB | ❌ |
| 32 | `radar_front_center_mrr_2` | 24.03 % | 133 MB | ❌ |
| 33–34 | `radar_{rear_left, rear_right}_mrr_2` | 24.03 % | 120–122 MB | ❌ |
| 35–36 | `radar_side_{left, right}_srr_3` | 7.45 % | 14 MB | ❌ |

### ⭐ Group F — FOUR corpus tables that are **not in the 36 at all**

The "36 features" framing hid these. They are not in `features.csv`, they are not under a feature
directory, and **no audit has enumerated them**. Combined: **34 MB for the whole corpus.**

| Table | Rows | Schema (MEASURED) | Read? |
|---|---:|---|:--:|
| `clip_index.parquet` | 306,152 | `clip_id, clip_is_valid (all True), chunk, ` **`split`** ` (train 153,625 / val 90,928 / test 61,599)` | ✅ (r0) |
| `metadata/data_collection.parquet` | 306,152 | `clip_id, country (25), month, hour_of_day, platform_class (hyperion_8/8.1), radar_config` | ✅ (r0) |
| `metadata/feature_presence.parquet` | 306,152 × 36 | the coverage matrix used throughout this report | ⚠️ audit only |
| **`reasoning/ood_reasoning.parquet`** | **1,740** | `clip_id, feature (always front_wide), ` **`event_cluster`** ` (9), ` **`events`** ` (JSON: frame, timestamp, free-text Chain-of-Causation), split` | ❌ **⭐ NEW** |

`reasoning/` exists **only on `main`** — it is absent from both the `26.03` and `25.10` tags, i.e. it
postdates every previous audit of this repo. It is **human-verified** (card §Reasoning Labels), which
makes it the only non-machine label in the corpus.

---

## 2. Per-question verdicts

### 2.1 Map / lane geometry / lane graph / routable topology — 🔴 **DOES NOT EXIST. CLOSED.**

**Verdict: NO, decisively, and the program should stop re-asking.** Five independent probes (§0) plus the
authors' own published statement that they *"do not include open maps data"*. The card adds that XODR is a
**future** intention — so this is not a hidden feature, it is a known gap on NVIDIA's own roadmap.

**Second and third probes, per `CLAUDE.md`:** I did not stop at the tree. I checked (a) the feature
manifest, (b) `obstacle.offline`'s value space for any lane/road/topology string — **zero hits** on a
regex covering `lane|map|road|junction|intersect|roundab|traffic|light|signal|sign|route|nav|goal|
topolog|centerline|crossing|stop`; (c) both tagged revisions; (d) the free-text reasoning labels;
(e) the egomotion schema for a global position that could support OSM map-matching. All negative.

**Consequence:** the commercially-clean lane-graph path is **not** in our own corpus, and it is **not**
reachable via OSM on our own corpus either (no GPS). Of the three leads in `NC_INGEST_REPORT` §5, lead
(a) *PhysicalAI's 34 unread features* is now **falsified**, and lead (c) *OSM map-matching on our ego
traces* is **falsified for PhysicalAI specifically** (it survives for comma2k19, which has u-blox GNSS,
and L2D, which has RTK GPS — both permissive).

### 2.2 Roundabouts — 🟡 **the card is honest, but the label does not exist — and the corpus is NOT roundabout-rich**

This is the reconciliation the brief asked for, and **the answer goes against the hypothesis.**

**Where the card's roundabout mention lives.** `pai_card.md:354-360`, section *"Dataset Diversity:
Environmental and Traffic"*: a six-bullet prose list (traffic density, road types, weather, surface,
time-of-day, *"Infrastructure elements such as tunnels, bridges, roundabouts, railway crossings, toll
booths, inclines, and more"*). **It is a description of what the drives contain, not an annotation
schema.** The proof: the same section promises weather, surface condition, road type and traffic
density — and `data_collection.parquet` ships **none of them**. It has exactly six columns:
`clip_id, country, month, hour_of_day, platform_class, radar_config`. **There is no queryable
roundabout tag, and no queryable weather or road-type tag either.** *(MEASURED.)*

**Now the harder question: is the corpus roundabout-rich, with our screen as the limiting factor?**
**No — two independent instruments converge on ~0.63 %.**

| Instrument | What it measures | Rate | Wilson 95 % |
|---|---|---:|---|
| Our kinematic screen, **strict** (`situ_full.py`, 3,000 phase-0 clips, 485,401 windows) | \|Δψ\|≥180°, 6≤R≤25 m, monotone>0.90, 3≤v≤11 m/s | **0.633 %** (19) | [0.406, 0.987] |
| NVIDIA's **human-verified CoC text** (1,740 OOD clips) | free-text mentions of `roundabout\|traffic circle\|rotary` | **0.632 %** (11) | [0.353, 1.129] |
| Our kinematic screen, **loose** | \|Δψ\|≥135°, R≤30 m, monotone>0.85 | 3.500 % (105) | [2.900, 4.219] |

Two instruments that share **no code, no author and no failure mode** — a yaw/radius geometry detector
and a human-written English sentence — land **0.633 % vs 0.632 %**. That is not proof (neither sample is
a random geometric census, and the reasoning subset is OOD-curated at 49 % work zones, so the direction
of its bias is **not established**), but it is a strong argument that **our screen was measuring the
corpus correctly, not clipping it.**

⚠️ **Two corrections to the numbers as briefed.** (1) The screen ran over the **3,000-clip phase-0
selection**, not 2,376 — quoting 19/2,376 inflates the rate by 26 %. (2) Scaled to the 2,376-clip
parity corpus the screen implies **~15 strict / ~83 loose**, not 19/105.

**The geography sub-finding, which is the genuinely new part.** In the reasoning labels the roundabout
mention rate is **EU 1.16 % vs US 0.00 %** (0 of the US clips). And our own corpus — measured for the
first time this session by joining `r0_selection.parquet` to `data_collection.parquet` — is
**91.6 % European** against a corpus-wide **49.3 %**. So our selection is *already* roundabout-enriched
relative to a random PhysicalAI draw; a corpus-wide scale-up would roughly **halve** the density.
**Scaling up inside PhysicalAI buys volume, not density.** To reach ~200 strict roundabout clips at the
measured 0.633 % rate you would need to ingest on the order of **32,000 EU-weighted clips**
(**~0.68 TB** of front-wide camera alone, at a measured **21.1 MB/clip**) — feasible, but it must be
sold as a volume play, not a targeted one.

### 2.3 Junction / intersection annotation — 🔴 **DOES NOT EXIST.** H2's proxy cannot be replaced.

**Verdict: NO.** Nothing topological. `obstacle.offline` is agents only. The nearest thing in the entire
corpus is free text on 1,740 clips (5.86 % mention an intersection/junction).

**So H2's junction stratum keeps its kinematic proxy** — and this probe surfaces a *different* problem
with it that matters more than the missing topology. **"Junction" currently means two incompatible
kinematic things in this program:**

- H2 label-v2 / E0-E1 (`l2_build.py:138-162`, `h2e_build.py:53-81`): **|Δψ| ≥ 45° over 6 s AND R ≤ 30 m**
- the gate stratum (`taniteval/corridor.py:158-170`): **|Δψ| ≥ 10° over the first 2 s**

Both explicitly disclaim topology in their own docstrings, so neither is *wrong* — but the **0.45×
[0.00, 1.40] junction NULL** was measured under the 45°/6 s definition while the gate reports a
different population under 10°/2 s. Before anyone concludes "junctions destroy the association", the
cheap discriminating check is to **re-run the H2 stratum under the gate's definition** and see whether
the NULL survives. That is a CPU-minutes re-slice of an existing frame table, and it does not need a
map. **This is the single most actionable item this probe produced for H2.**

### 2.4 Traffic-light presence or state — 🔴 **DOES NOT EXIST. The standing claim is CONFIRMED.**

**Verdict: NO, confirmed at three locations.** (1) No feature row. (2) `obstacle.offline`'s enum is 10
classes, all dynamic agents — **no `traffic_light`, no `traffic_sign`, no static class at all**
(MEASURED on 87,481 cuboids). (3) The card documents no light state anywhere. The only traffic-light
information in the entire 133 TB corpus is **free text on 81 of 1,740 reasoning clips (4.66 %)** — and
even that is a *mention*, never a state.

This closes the question the way the brief hoped it might not: **G5 cannot be filled in-house.**

### 2.5 Route / navigation / goal — 🔴 **DOES NOT EXIST.**

No route, nav command, goal, destination or intent field in any of the 36 features or the 4 corpus
tables. The reasoning `events[].coc` free text is *action-grounded* (`"Steer right to exit the
construction zone"`), which is closer to a **tactical** annotation than a strategic route, and it exists
on 0.57 % of clips.

### 2.6 Anything else decision-relevant — 🟢 **YES, four things, and one is nearly free**

| Asset | Why it matters | Coverage | Cost for our 2,376 clips |
|---|---|---:|---:|
| ⭐ **`obstacle.offline`** — 3D tracked cuboids, 10 classes, `track_id`, ~39 tracks/clip | lead vehicle, gap acceptance, yield partners — **T1/T2/T3's `Y_outcome` inputs without a simulator** | 97.44 % | **~1.5 GB** |
| ⭐ **`camera_intrinsics.offline.ego_mask_image_png`** | a per-camera **ego-vehicle mask**. We currently crop the hood blind | 97.44 % | **~5 MB** |
| ⭐ **`vehicle_dimensions.wheelbase`** | **2.85 m (90 %) / 3.165 m (10 %)** — we hard-code `WHEELBASE = 2.9` in `physicalai.py:51`, and it feeds `steer = atan(WHEELBASE · curvature)` on **every action label in the program** | 100 % | **~0.1 MB** |
| ⭐ **`reasoning/ood_reasoning.parquet`** | human-verified event clusters + CoC text; `COMPLEX_INTERSECTION_INTERACTION`, `yield` 12.87 % | 1,740 clips | **0.15 MB (already pulled)** |
| `data_collection` country / hour_of_day | domain-shift strata we have never used | 100 % | already local |
| `blurred_boxes.parquet` | 2D anonymisation boxes, already inside camera zips | 100 % | free |

**No weather, no road-type, no traffic-density, no surface-condition field exists** despite the card
describing all four. Anyone planning a weather-stratified eval on PhysicalAI must label it themselves.

---

## 3. Reconciling the three standing claims

### 3.1 *"No map/lane/traffic-light feature exists"* — ✅ **CONFIRMED, and now much better evidenced**

Previously a one-location claim from `features.csv`. Now five locations plus the authors' own words
(§0). The claim was **right**. It should be quoted with the card citation (`pai_card.md:303`) from now
on, because a published author statement outranks our enumeration.

### 3.2 *"We read 2 of 36"* — ⚠️ **STALE. We read 4 of 36, and touch 5.**

The "2" is true of `physicalai_r0.py` **alone** (the selection/download stage:
`labels/egomotion` + `camera/camera_front_wide_120fov`). But the **episode-build** stage reads two more:

- `stack/tanitad/data/physicalai.py:153` `_CALIB_INTR = "camera_intrinsics"` — resolved per clip
- `stack/tanitad/data/physicalai.py:154` `_CALIB_EXTR = "sensor_extrinsics"` — resolved per clip

Both landed with the **D-016 R1 two-rig fix** and both are load-bearing: without a per-clip `cy` the
crop reverts to geometric-centre and the rig-A/rig-B horizon inconsistency returns. The local cache
confirms ~30 downloaded chunks of each. `pai_calib_probe.py` additionally reads `vehicle_dimensions`.

**Corrected statement: the production ingest reads 4 of 36 features (11.1 %); a 5th is touched by a
probe script; and 2 of the 4 non-feature corpus tables are read by `physicalai_r0.py`.** The count
"2 of 36" appears in `CLAUDE.md:132` and at least seven documents; it predates D-016 R1 and should be
updated. *(This does not weaken the headline — 32 features are still unread — it just makes the number
honest.)*

### 3.3 *"4 of 9 v3 ROUTE tokens are never minted, each because it asserts a MAP fact"* — ✅ **CONFIRMED, and the justification is now airtight**

The 9 tokens are `follow, straight, turn_left, turn_right, exit_left, exit_right, merge, u_turn,
roundabout` (`stack/tanitad/lake/vocab.py:40-41`, mirrored `refb_labels.py:803-805`, pinned by
`test_refb_labels_v3.py:195-197`). The 4 never minted are **`straight`, `exit_left`, `exit_right`,
`merge`** — `refb_labels.py:997-999` states `straight` is never emitted because it *"asserts a junction
exists = a MAP fact"*. *(Note the earlier "6 of 9" was already retracted as class C4 →
`RETRACTION_LOG.md:42`; **4 of 9 is the correct figure** and the brief used it correctly.)*

**This probe upgrades the justification from an assertion to a proof.** The refusal rested on "there is
no map" — which was a single-location claim. It now rests on five probes *and* NVIDIA's published
statement. **The refusal is correct and should not be revisited on this corpus.**

One honest caveat in the other direction: `merge` and `exit_*` have a *weak* text source that did not
exist when the refusal was written — the reasoning labels mention merging on 5.06 % and exiting on
0.23 % of 1,740 clips. That is **~88 and ~4 clips**. Far too thin to mint a token, but it means the
statement "there is no signal at all" is now marginally too strong; the accurate statement is
"there is no signal at usable scale, and none that is queryable".

### 3.4 Bonus reconciliation — the `$0` AlpaSim roundabout gate is **not executable as written**

The 4-brain program (`DATA_STRATEGY.md:346`, `4BRAIN_DOMINANCE_PROGRAM.md:276`) asks for *"category
frequencies over the **356 already-banked screened labels**"*. **356 is the number of keyframes
screened; only 38 were banked** (`scaled_suite_labels.json`, MEASURED), and they were banked as a
**deliberately balanced suite** — `roundabout 8, highway 8, straight_other 8, traffic_light 7,
intersection 7`.

⚠️ **A balanced sample cannot estimate a pool frequency.** Reading 8/38 = 21 % and projecting it onto
the 1,606-scene pool gives ~338 scenes and is **inadmissible** — it measures the sampling design.
The only usable statistic is the screening proportion recorded in
`scenario_stratified_scaled_NOTE.md:14-15` (*"356 candidate keyframes … Roundabouts are rare (~2.5 %)"*).

**Gate result, run:** 8/356 = 2.25 %, Wilson 95 % [1.14 %, 4.37 %] → on 1,606 scenes:
**36 roundabout scenes, 95 % CI [18, 70], against a bar of 40.**

> **VERDICT: the interval straddles the bar. The gate can be neither passed nor failed on banked
> evidence.** S4 is *plausibly* powerable on AlpaSim and *plausibly* not. Resolving it costs one
> classification pass over more of the 1,606-scene pool — still $0, still no download. It does **not**
> justify a corpus acquisition either way, and the "~36 scenes" point estimate already in
> `DATA_STRATEGY.md:343` is the right number; it was simply missing its interval.

---

## 4. GO / NO-GO on each blocked problem

| Problem | Blocked on | Does PhysicalAI unblock it? | Verdict |
|---|---|---|---|
| **S1** branch selection at multi-option junctions | map successors as the option set | **NO.** No topology, no successors, and no GPS to derive them | 🔴 **NO-GO in-house. Needs an external map corpus or AlpaSim.** |
| **S2** lane selection | parallel-lane set + `succ(lane)` | **NO.** No lane geometry at all | 🔴 **NO-GO in-house.** |
| **S4** roundabout exit ordinal | ordered exit polygons + ≥40 roundabout scenes | **NO** for the option set (no exit polygons). **Partly** for volume: ~0.63 % strict → ~1,900 clips corpus-wide, but at **~15 in our parity corpus** | 🔴 **NO-GO in-house.** AlpaSim gate is **INDETERMINATE** [18, 70] vs bar 40 |
| **H2 junction stratum** | "junction" is kinematic, not topological | **NO** topology. **But** the probe found a **definitional split** (45°/6 s vs 10°/2 s) that is a cheaper explanation of the NULL than the missing map | 🟡 **PARTIAL — re-slice under the gate's definition first (CPU-minutes) before blaming topology** |
| **T1/T2/T3** yield / gap / overtake | agent state | **YES, substantially** — `obstacle.offline` is 3D tracked cuboids on 97.44 % of clips, ~1.5 GB for our corpus | 🟢 **GO. Highest-value ingest in this report.** |
| **G5** traffic lights | any light annotation | **NO.** 10 dynamic classes, zero static | 🔴 **NO-GO in-house, permanently.** |

### Do we still need an external corpus?

**Yes — for S1, S2, S4's option sets and G5, unavoidably.** This probe removes the last hope that our
own corpus could supply them, and it does so cleanly enough that the program should now commit rather
than re-survey. The strategic-brain proof needs either (a) an external map corpus, accepting that
`nc-research` + ShareAlike means **the proof cannot ship**, or (b) **AlpaSim**, which is in-house and
has the topology — which is exactly why the *VectorMap connectivity probe* is ranked #1 on the 4-brain
"start today" list and should now be treated as **the** unblocking action, not one of several.

**But the premise that ranked this probe #0 was wrong in an informative way.** The survey ranked it
first on the hypothesis that a queryable roundabout tag might exist. It does not, and the corpus is not
roundabout-dense either. **The probe's actual payoff is different and still large:** `obstacle.offline`
at 1.5 GB, a wheelbase error touching every action label in the program, and a plausible non-map
explanation for the H2 junction NULL.

---

## 5. Recommended ingest — what to actually do

Priority order. None of these needs a training pod; all are dev-box or eval-pod work.

**P0 — `vehicle_dimensions.wheelbase` (minutes, ~0.1 MB).** `physicalai.py:51` hard-codes
`WHEELBASE = 2.9`; the true values are **2.85 m (90 % of clips) and 3.165 m (10 %)** — so the constant
is wrong for **100 % of clips**. It feeds `steer = atan(WHEELBASE · curvature)`, i.e. **every action
label in every arm**. MEASURED bias, near-constant across the curvature range: the 10 % at 3.165 m have
steer **under-estimated by ~8.4 %**; the 90 % at 2.85 m have it **over-estimated by ~1.8 %**. Small, but
systematic, sign-flipped between two populations, and free to fix. ⚠️ **This is a parity-affecting
change**: re-deriving actions changes the training signal, so it needs a *measured* comparison against
the current labels and a deliberate decision, not a silent fix. Adapter: one per-clip join in
`physicalai.py`, exactly like the existing intrinsics path.

**P1 — `obstacle.offline` for our 2,376 clips (~1.5 GB, hours).** 3D tracked cuboids, 10 classes,
`track_id`, `reference_frame='rig'`, ~39 tracks/clip at 97.44 % coverage. Unblocks lead-vehicle state,
gap acceptance and yield partners **without a simulator**, and it is the corpus-side half of T1/T2/T3.
The adapter is small: per-chunk zip → per-clip parquet, already the exact shape of
`load_egomotion()`. **This is the single highest-value thing in the report.**

**P2 — `camera_intrinsics.offline.ego_mask_image_png` (~5 MB, hours).** Per-camera ego-vehicle mask.
Feeds the f-theta crop so the hood is masked rather than cropped blind.

**P3 — `reasoning/ood_reasoning.parquet` (already pulled, 0.15 MB).** 1,740 human-verified clips with
event clusters and CoC text. Two uses: a **held-out semantic eval set** (it is the only human label in
the corpus, and it ships its own train/val split), and a **validation set for our VLM labeler** — our
`enums.json` already has `road_geometry ∈ {junction, roundabout, …}` and `scenario_tag ∈
{traffic_light_stop, unprotected_turn, yield_merge}`, so the CoC text can score the labeler on clips it
did not write. ⚠️ 1,740 clips is small and OOD-curated; it is an eval asset, not a training corpus.

**P4 — re-slice the H2 junction stratum under `corridor.py`'s 10°/2 s definition (CPU-minutes).**
Not an ingest; the cheapest discriminating test of whether the 0.45× NULL is a topology problem or a
definition mismatch. Both outcomes are informative and it should be pre-registered.

**Explicitly NOT recommended:** the 6 unread cameras (**40.7 TB**), the lidar (**101.7 TB — 67.5 % of
the dataset**), and the radar suite (1.46 TB across 19 features, ≤28.5 % coverage, and no unblocked
problem depends on it). If surround cameras are wanted for H2 later, cost them per-clip —
**21.1 MB/clip/camera**, MEASURED — not per-corpus: our 2,376 clips × 6 cameras ≈ **301 GB**, which is a
very different conversation from 40 TB.

---

## 6. Probes run (so this question can be closed rather than re-asked)

| # | Probe | Command / artefact | Result |
|--:|---|---|---|
| 1 | HF tree API L1 | `probe_pai_features.py` → `pai_tree_l1.json` | 7 dirs + `features.csv` + `clip_index.parquet` + LICENSE + README |
| 2 | HF tree API L2, all 7 groups | `pai_tree_l2.json` | 6+7+3+1+19 = 36 feature dirs; `metadata` 2 files; `reasoning` 1 file |
| 3 | HF tree API L3, all 36 dirs | `pai_tree_l3_sample.json` | chunked layout confirmed; radar chunk ranges differ by config |
| 4 | The dataset's own `features.csv` | `pai_features.csv` | **exactly 36 rows**; split verified exactly |
| 5 | Coverage matrix | `metadata/feature_presence.parquet` → `pai_metadata_summary.json` | 306,152 × 36 booleans; five distinct coverage tiers |
| 6 | Dataset card | `pai_card.md` | *"we do not include open maps data"* (line 303) |
| 7 | `obstacle.offline` class enum | `probe_obstacle_schema.py` → `pai_label_schemas.json` | **10 classes, all dynamic**; regex for map/traffic semantics → **NONE** |
| 8 | `egomotion` global position | measured chunk_0000 | **no lat/lon/GNSS**; clip-local metres → **OSM map-matching impossible** |
| 9 | Both tagged revisions | `pai_sizes_and_revs.json` | `25.10` and `26.03` identical at top level; `reasoning/` is `main`-only |
| 10 | Per-feature sizes | `get_paths_info` × 36 | 150.7 TB extrapolated vs 133 TB published; lidar = 68 % |
| 11 | Our corpus vs corpus metadata | `profile_our_corpus.py` → `our_corpus_profile.json` | **91.6 % EU** vs 49.3 %; all sampled clips in official `train` |
| 12 | Roundabout / junction / light rates | `roundabout_arithmetic.py` → `roundabout_arithmetic.json` | 0.632 % vs our 0.633 % — two independent instruments converge |
| 13 | AlpaSim `$0` gate | `scaled_suite_labels.json` (38, balanced) + screening note | **36 scenes, CI [18, 70], bar 40 → INDETERMINATE** |
| 14 | NVlabs wiki | `github.com/NVlabs/physical_ai_av/wiki` | **"Machine Labels" page is literally "(Coming Soon)"** — the obstacle schema is undocumented upstream; probe 7 is the only source |

**Not probed / open:** the `physical_ai_av` pip package may embed a schema constant; the
**Cosmos Dataset Search** preview (~41 k clips, `build.nvidia.com`) offers *semantic video search*
and could in principle retrieve roundabouts by text query — that is a **service**, not a feature, it
covers 13 % of the corpus, and it returns clips rather than labels, but it is the one remaining place a
roundabout *retrieval* capability could live. It needs a browser login, so it is a human action.

---

## 7. Deliverable manifest

All under `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-26-physicalai-feature-probe/`
in the working tree. **NOT `git add`ed** (per brief).

| File | What |
|---|---|
| `PHYSICALAI_FEATURE_PROBE.md` | this report |
| `probe_pai_features.py` | HF tree-API walk (L1→L3) + card + repo info |
| `pull_pai_metadata.py` | pulls + describes the 4 corpus tables (~34 MB) |
| `probe_pai_sizes_and_revs.py` | per-feature sizes via `get_paths_info`; revision comparison |
| `probe_obstacle_schema.py` | measures label/calibration schemas + the map-semantics regex hunt |
| `profile_our_corpus.py` | joins our `r0_selection` to the corpus metadata |
| `roundabout_arithmetic.py` | prevalence rates, Wilson intervals, the AlpaSim `$0` gate |
| `scrub_gated_samples.py` | **confidentiality pass** — see below |
| `pai_features.csv` | the dataset's own 36-row manifest (copy) |
| `pai_card.md` | the dataset card as fetched |
| `pai_tree_l1.json` · `pai_tree_l2.json` · `pai_tree_l3_sample.json` | raw tree output |
| `pai_repo_info.json` · `pai_sizes_and_revs.json` | repo metadata, sizes, revisions |
| `pai_metadata_summary.json` · `pai_label_schemas.json` | derived schemas + coverage |
| `our_corpus_profile.json` · `roundabout_arithmetic.json` | the two analyses |

Gated bytes cached **outside the repo** at `%TEMP%\claude\pai_probe_cache` (~110 MB) and not staged.
Total written here: **554 KB**, all schema and aggregates.

⚠️ **Confidentiality pass, and it caught something.** The schema-describing helpers sample real values
per column, which pulled three kinds of gated *content* into the derived JSON: **7 clip UUIDs**, **6 raw
`ego_mask_image_png` byte blobs**, and per-clip calibration JSON. `scrub_gated_samples.py` replaced
UUIDs with salted 8-hex tags, binary blobs with length-only descriptors, and long calibration blobs with
their key names — preserving every dtype, cardinality, enum and count the report depends on.
`pai_label_schemas.json` fell 525 KB → 23 KB (the PNG bytes were almost all of it). **Re-verified: zero
UUIDs and zero PNG blobs remain in any file.** Anyone re-running the probe scripts must re-run the
scrubber before the output goes anywhere — the raw probe output is *not* safe to stage.

## 8. Escalations (integration, not a note in a README)

1. **`CLAUDE.md:132` and ≥7 documents say "we read 2 of 36". It is 4 of 36** since D-016 R1 (§3.2).
2. **`WHEELBASE = 2.9` in `physicalai.py:51` is wrong for 100 % of clips** (true: 2.85 / 3.165). Parity-affecting — needs an owner and a measured comparison, not a silent fix.
3. **The `$0` roundabout gate in `DATA_STRATEGY.md:346` cannot be run as written** (356 screened ≠ 38 banked, and the 38 are balanced). Run as corrected in §3.4 the answer is **INDETERMINATE**, not the implied go/no-go.
4. **"Junction" has two incompatible kinematic definitions** (45°/6 s vs 10°/2 s) across H2 and the gate (§2.3). This is a cheaper candidate explanation for the 0.45× NULL than missing topology and should be tested first.
5. **The roundabout screen ran on 3,000 clips, not 2,376** — several docs quote 19/2,376.
