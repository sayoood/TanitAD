# Data Strategy for the Three-Layer Hierarchy — a decision document

**Author:** Data Engineering agent. **Date:** 2026-07-21. **Status:** decision doc — for Sayed / orchestrator review.
**Question it answers:** Sayed's *"part of strategy must be the data handling — we need the right data to train our
three layers."* The v4+ architecture (strategic / tactical / operative, each planning through imagination) is blocked
by **data, not design**. This document ranks the ways through, with costs and falsifiers.

> **Companion docs:** `TANITDATASET_V1_STRATEGY.md` (rev-4, the two-tier lake doctrine) ·
> `Research/2026-07-19-tanitdataset-v1-sensor-survey.md` (the 25-dataset licence survey — this doc does not repeat it) ·
> `DATA_LAKE_ARCHITECTURE.md` · `../Architecture & Inference/V3_GOAL_VOCABULARY_V1.md` (the frozen slot vocabulary
> everything below must fill) · `Project Steering/MODEL_REGISTRY.md` (the only quotable source for model facts).

**Evidence classes used throughout — every number carries one.**
`MEASURED (ours)` = we ran it, this session or a cited prior run · `PUBLISHED` = an official card/paper/licence, cited ·
`ESTIMATED` = our arithmetic or judgement, with the reasoning shown · `UNVERIFIED` = could not confirm; do not act on it.

---

## 0. TL;DR — the decision

**The single most important finding: two of the three blockers are already solved by data we can obtain for
≈13 GB of download and zero episode re-selection.**

1. ❌ **PhysicalAI-AV ships NO geo.** Not partial, not hidden — the dataset card says it outright, and both pose
   variants are clip-local. The map-matching path (GPS → HMM road-snapping → OSM topology) **does not exist for our
   canonical corpus and cannot be made to exist**. Close that thread.
2. ✅ **PhysicalAI-AV *does* ship full 3D agent tracks — `obstacle.offline` — on 96.90 % of our corpus, and we never
   ingested them.** Track ids, class, 3D box, orientation, in the ego rig frame. That is blocker #2 (no agent state)
   solved **on the parity corpus itself**, for a **12.4 GB** download and **no re-selection whatsoever** — we
   re-derive labels, we do not re-pick episodes. Parity is untouched.
3. ✅ **L2D (`yaak-ai/L2D`, Apache-2.0, already registered in our lake as tier `ship`) solves the other two blockers.**
   It carries OSM-snapped waypoints, road class, lane count and speed limit *per frame*; the **ego's own turn-signal
   from CAN**; natural-language strategic instructions *with distances* ("In 100 m when you have the right of way
   enter the roundabout"); and — the finding that matters most — its 30 s episodes are **sliding windows at a
   ~13.8 s stride over continuous drives that reconstruct exactly**, giving drives up to **35.4 minutes**.
   The 20 s ceiling is a packaging artefact, and L2D's packaging is invertible.
4. 🔻 **CARLA is the only source on earth that gives *other agents'* indicator state — and we cannot render it on our
   pods.** MEASURED (ours, 2026-07-09): RunPod GPUs expose `compute,utility` only, so Vulkan/EGL has no ICD and CARLA
   is `-nullrhi` (no pixels) on pod1/2/3/eval. Keep CARLA for label-structure and closed-loop, not for camera data,
   until a graphics-capable pod exists.
5. ⛔ **Two licence exclusions to encode now.** **Waymax** §2.e forbids using it, directly or indirectly, to train or
   improve *any* AI foundation model or anything distilled from one. **Waymo Open / WOD-E2E** goes further: its terms
   follow the *trained weights* into vehicle operation and production systems — the only licence in this survey that
   reaches our end product, so even an internal-only tier is unsafe. Neither may enter the lake.
6. 🟢 **A freebie we already own:** `comma2k19` (MIT, already in our lake) carries raw CAN from which
   `leftBlinker`/`rightBlinker` are decodable. It is the only *commercially-clean* ego-indicator corpus in the
   landscape and it costs zero download.

> 🔴 **CORRECTION 2026-07-21 — the `obstacle.offline` ingest recommendation below is SUPERSEDED by
> measurement.** The pre-registered lead-state gate (`Research/2026-07-21-lead-state-gate.md`,
> `stack/scripts/lead_state_gate.py`) tested the premise this recommendation rests on and **falsified it**:
> adding lead state (gap / closing speed / TTC) to an ego-state regressor reduces the held-out error of the
> ego's future 2 s **longitudinal** displacement by **+1.16 % [−0.92, +3.19]** (paired episode-cluster
> bootstrap, 126 held-out episodes, 21,546 windows) — inside the pre-registered **FAIL** band, and ≤ +1.83 %
> at every horizon out to **6 s**. Shuffle control negative in every cell; ridge and GBM agree.
> **Do not run the 197-chunk ingest on this argument.** What survives: `lead_state` is *measurable*
> (headway/TTC metrics + lead-presence stratification are unblocked), and a **lead-conditioned specialist**
> gains ~8–10 % on the 38.5 % of windows that have a lead — a post-hoc HYPOTHESIS needing its own
> pre-registration, not a pass. Corpus-wide lead presence, now measured on **26 chunks / 25 countries /
> 614 clips**: **38.51 %** of windows, **66.1 %** of clips (supersedes the 3-chunk estimate in §2).

**Recommended sequence:** ~~ingest `obstacle.offline` this week~~ *(see the correction above — the premise
failed its gate)* → pilot L2D next (unblocks *strategic* + *horizon*, new arm, never mixed into the parity
set) → treat synthetic pixel generation as **out of reach on our hardware** and CARLA as a *label-schema
and closed-loop* asset only.

**The composition, in one line:** *agents from PhysicalAI · map + horizon + indicator from L2D · schema from nuPlan ·
a commercially-clean indicator cross-check from comma2k19 · pixels from nobody synthetic.*

---

## 1. Question 1 — does the raw PhysicalAI-AV source carry geo, and did we drop it on ingest?

### Verdict: **NO — and it was never there.** High confidence, three independent legs.

| leg | evidence | class |
|---|---|---|
| The card says so | NVIDIA's own dataset card: *"we do not include open maps data. Scenes are not compatible with CARLA unless the user generates their own XODR data for now."* | PUBLISHED |
| Keyword sweep of the card | `GPS` 0 hits · `latitude` 0 · `longitude` 0 · `location` 0 · `HD map` 0 · `lane` 0 · `traffic light` 0 · `turn signal` 0, over the full 31,935-byte card | MEASURED (ours) |
| The pose data itself | Both pose products start at the origin: card §Labels — *"a local coordinate frame … with the origin located at the ego vehicle's position at timestamp 0, oriented such that there is 0 yaw at timestamp 0"*. Row 0 of `egomotion.offline.chunk_0036` is literally `x=y=z=0, q=(0,0,0,1)` | MEASURED (ours) + PUBLISHED |
| The clip metadata | `metadata/data_collection.parquet` (306,152 rows) has exactly five columns: `country`, `month`, `hour_of_day`, `platform_class`, `radar_config`. `clip_index.parquet` has three: `clip_is_valid`, `chunk`, `split`. **There is no finer location field than *country*.** | MEASURED (ours) |

**Interpretation.** The coarsening to *country × month × hour* is deliberate privacy engineering, not an omission. It
also explains a second property we measured: **chunks are not drive-sessions.** Over all 3,146 chunks the largest
`(country, month, hour, platform)` group inside a chunk is a median of **5.1 %** of the chunk (MEASURED, ours) — the
data is shuffled at ingest, so temporally adjacent 20 s clips are scattered across the 133 TB archive. Our 2,376-episode
corpus is 0.78 % of the dataset; the probability that it contains a usable adjacent pair is negligible.

**Consequences, stated plainly.**
- The Newson–Krumm HMM map-matching path, the Overpass/OSM junction-topology extraction, roundabout exit ordering,
  lane counts and speed limits from a road graph — **all of it is inapplicable to the parity corpus.** There is no
  coordinate to snap. No amount of engineering recovers it.
- The GPS-accuracy-vs-lane-width question ("do we get lane-level or road-level truth") **does not arise for
  PhysicalAI**. It arises for L2D, where it is answered in §3.2.
- **Forward-looking, PUBLISHED:** NVIDIA states *"we are looking to add XODR to enable simulation for CARLA, AlpaSim,
  and others in the future."* If that lands, the map blocker closes on the parity corpus at zero data cost. This is
  worth a standing watch on the dataset's Version History (currently `26.03`); it is **not** something to plan around.

### But the ingest *did* drop something — and it is the bigger prize

`metadata/feature_presence.parquet` (which our own `physicalai_r0.py` already downloads and joins) lists **36
features**. Our ingest reads exactly two: `egomotion` and `camera_front_wide_120fov`. Present and unread:

| feature | presence, all 306,152 clips | presence, our 3,000-clip phase0 selection | what it is |
|---|---|---|---|
| **`obstacle.offline`** | **97.44 %** (298,326) | **96.90 %** (2,907 / 3,000) | **3D agent tracks** — see §2 |
| `egomotion.offline` | 97.44 % | 96.90 % | 10 Hz smoothed pose (vs 200 Hz online); still clip-local |
| `lidar_top_360fov` | 97.44 % | 96.90 % | ~200 Draco-compressed spins/clip |
| radar (10 sensors) | 52.5 % of all clips | — | short/medium/long range scans |
| `reasoning/ood_reasoning.parquet` | 1,740 clips total | **43 clips overlap ours** | human-verified Chain-of-Causation — see §2.3 |

All MEASURED (ours) from the local `metadata/` parquets and the live HF repo.

**Why we missed it, honestly:** dataset version `26.03` *added* `egomotion.offline` / `obstacle.offline` for 97 % of
clips (PUBLISHED, Version History). Our local metadata snapshot is dated 2026-07-06 and already carries the 26.03
schema, while the corpus was built 2026-07-12 — so **the labels existed at ingest and our loader simply never looked
at them.** The parent hypothesis was right about the location of the gap ("the schema gap is in the PhysicalAI ingest,
not the lake design") — it was just the *agent* labels, not geo.

---

## 2. What `obstacle.offline` actually gives us — measured

Downloaded `labels/obstacle.offline/obstacle.offline.chunk_0036.zip` (62.7 MB, 98 clips — chunk 36 is in our corpus)
and analysed it. All numbers MEASURED (ours), 2026-07-21.

**Schema** (16 columns): `timestamp_us, source, track_id, center_{x,y,z}, size_{x,y,z}, orientation_{x,y,z,w},
label_class, reference_frame, reference_frame_timestamp_us`. `source = scene:obstacles:autolabels:v2`,
`reference_frame = rig` — boxes are already in the **ego rig frame at that timestamp**, so no transform chain is
needed to get relative geometry.

**Classes** (346,503 rows over 40 clips): `automobile` 84.4 %, `person` 10.6 %, `heavy_truck` 1.9 %, `trailer` 1.0 %,
`rider` 0.8 %, `bus` 0.5 %, `protruding_object` 0.45 %, `other_vehicle` 0.24 %, `stroller` 0.05 %, `animal` 0.02 %.

**Density** (98 clips, resampled to a 10 Hz grid with a 0.5 s association gap guard):

| quantity | value |
|---|---|
| tracks per 20 s clip | median **134**, mean 152, max 417 |
| simultaneously tracked agents per frame | mean **38.9**, p50 35.7, p90 77.1 |
| **frames with a lead vehicle** (2–80 m ahead, \|lateral\| < 2 m) | mean **45.2 %** of frames |
| clips with a lead vehicle at some point | **87 %** |
| clips with a lead ≥ 20 % / ≥ 50 % of frames | **66 %** / **43 %** |
| frames with a VRU in the near field (−5…50 m, \|lat\| < 8 m) | mean 18.3 %; 41 % of clips have one |

**Does it generalise across regions?** I repeated the measurement on two more corpus chunks (MEASURED, ours):

| chunk | n clips | country | lead-vehicle present (mean) | agents/frame (mean) |
|---|---|---|---|---|
| 0036 | 98 | United States | **45.2 %** | 38.9 |
| 1807 | 76 | Germany | **53.4 %** | 50.9 |
| 2490 | 97 | Slovakia | **28.0 %** | 17.0 |

Density varies ~2× by region — dense EU urban at the top, sparse Slovakia at the bottom — but **every region has a
usable lead signal**, and the corpus-wide figure will land in the 30–50 % band. That variation is itself useful: it
gives a natural stratification axis for the curation sampler.

⚠️ **Sampling caveat:** 271 clips across 3 of 197 chunks. Enough to decide *whether to ingest*; not a corpus
statistic. Recompute over all 197 phase-0 chunks during the ingest and record the corpus-wide number.

⚠️ **The labels are asynchronous, not per-frame.** Each row carries its own `timestamp_us`; a track is sampled at
irregular intervals (median inter-row Δt across all tracks 1.58 ms because rows interleave). You **must** resample
per-track onto the 10 Hz episode grid with an explicit max-gap guard, exactly as the measurement above does. Naïvely
grouping by `timestamp_us` produces ~1 agent per "frame" and will make the labels look worthless — that trap cost this
investigation one iteration.

⚠️ `source = autolabels:v2` — these are **machine labels, not human GT.** They are the same class of artefact as our
kinematic labels, and the provenance stamp must say `prov: "map"`→ no; the correct stamp is a **new
`prov: "autolabel"` value or reuse of `"engineered"`**, decided at ingest. Do not stamp them `human`.

### 2.1 It plugs into infrastructure we already built

`stack/tanitad/lake/enrich.py:vlm_pending_lead_state()` already defines the target shape
(`{present, gap_m, closing_speed_ms, ttc_s}`), and `stack/tanitad/lake/curation.py` already consumes it — TTC-gated
safety events, closing-speed weakness strata, the `lead_constrained` VSOURCE path. The module comment says it exactly:
*"fires only with … lead_state … then they light up with no code change."* This is a **fill-the-stub** job, not a new
subsystem. The one change: the stub assumes a VLM will fill it; `obstacle.offline` fills it with geometry, which is
strictly better and needs no GPU.

### 2.2 Which v3 vocabulary slots become mintable

| slot | today | with `obstacle.offline` |
|---|---|---|
| `HEADWAY` | unknown | **exact** — real gap in metres, real closing speed |
| `LONMODE` lead modes (`follow_lead`, `approach_lead`, …) | unknown | **exact** |
| `VSOURCE = lead_constrained` | unknown | **exact** |
| `INTERACT` (cut-in, yield-to, merge-in-front) | unknown | **derivable** from track lateral motion into our corridor |
| `TACPOINT` (the point to act at) | unknown | **derivable** (nearest constraining agent / stop point) |
| `SIGNAL` (other agents' indicators) | unknown | ❌ **still unknown** — no light state in the schema |
| `LIGHTSTATE` (traffic lights) | unknown | ❌ **still unknown** — no traffic-light class |

That is the honest scope: `obstacle.offline` unblocks **agent-relational tactics**, not **signalling**.

### 2.3 The reasoning labels — small, but the right shape

`reasoning/ood_reasoning.parquet` (1,740 clips; train 1,450 / val 290) carries human-verified Chain-of-Causation
events keyed by clip UUID: `{event_start_frame, event_start_timestamp, coc}` — e.g. *"Decelerate to maintain a safe
distance from the lead vehicle with oversize load ahead."*, *"Steer right to exit the construction zone."*
(MEASURED, ours). Clusters: work-zones 49 %, pedestrian-density 22 %, uncommon-vehicle-behaviour 15 %, cyclists 4.5 %,
complex-intersection 2.9 %, plus long-tail.

**Overlap with our corpus: 43 clips** (MEASURED, ours, against the r0 ∪ r1 ∪ phase0 union of 4,613 clip ids). Too few
to train on. Its value is as a **gold evaluation slice for the CoC / language head** and as a **prompt-design
reference** for our own VLM pass — 1,740 human-refined examples of exactly the sentence form we want the strategic
layer to consume. It costs 1 file to hold; take it, do not build on it.

---

## 3. L2D — the strategic and horizon unblocker

`yaak-ai/L2D`, **Apache-2.0**, LeRobot v3.0, last modified 2026-05-26, 60,024 downloads (PUBLISHED, HF API).
Already registered in our lake: `stack/tanitad/lake/schema.py:68` — `SourceLicense("owned-safe", "Apache-2.0",
share_alike=False, is_synthetic=False)` → **tier `ship`, `commercial_ok = True`, HF-publishable.** No schema change
needed to admit it.

### 3.1 What is in it — verified against the data, not the card

I downloaded `meta/info.json`, `meta/episodes/chunk-000/file-000.parquet` (all 100,000 episodes' metadata) and
`data/chunk-000/file-000.parquet` (5,307 episodes, 1,392,787 frames of state). All below MEASURED (ours) unless marked.

| field | content |
|---|---|
| `observation.state.vehicle[8]` | speed, heading, **heading_error**, **latitude**, **longitude**, altitude, a_x, a_y |
| `observation.state.waypoints[10,2]` | 10 future (lon, lat) waypoints **already snapped to the OpenStreetMap graph** (PUBLISHED, card) |
| `observation.state.road` | OSM highway class — measured distribution: secondary 24 %, tertiary 21 %, residential 14 %, primary 14 %, NA 9.8 %, trunk 5.0 %, unclassified 4.4 %, motorway 3.4 %, service 1.9 %, + `*_link` ramps |
| `observation.state.lanes` | lane count — 2: 43 %, 1: 38 %, 3: 7.6 %, 4: 1.3 %, 5–6: 0.14 %, NA 9.8 % |
| `observation.state.max_speed` | **posted speed limit** per frame (e.g. 70.0) |
| `observation.state.{precipitation,conditions,lighting}` | Clouds 59 % / Clear 31 % / Rain 10 %; this file is all Day |
| `action.continuous[3]` | gas, brake, **steering angle** — real CAN |
| `action.discrete[2]` | gear, **turn_signal** |
| `task.policy` | **EXPERT 86.2 % / STUDENT 13.8 %** — native off-expert data |
| `task.instructions` | a natural-language strategic instruction per episode, **with a distance** |
| `observation.state.timestamp` | **unix epoch nanoseconds** — an absolute clock |
| cameras | 6 × 1080×1920 @10 Hz + a rendered BEV `map` channel at 360×640 |

`extrinsic_RDF.yaml` confirms **`cam_front_left` is the reference camera** (identity rotation, zero translation) —
i.e. `observation.images.front_left` is our forward camera. Coordinate system `{x: right, y: down, z: front}`.

### 3.2 The three blockers, against L2D

**Blocker #1 — map/topology: SOLVED, and the map-matching is already done for us.**
Yaak snapped the waypoints to OSM themselves; we get the *result*, not the HMM problem. Instruction coverage
**over all 100,000 released episodes** (MEASURED, ours, from `meta/episodes`; 28,949 distinct instruction strings):

| concept | episodes | share | | concept | episodes | share |
|---|---|---|---|---|---|---|
| **roundabout** | **3,532** | **3.53 %** | | pedestrian crossing | 18,268 | 18.27 % |
| turn left | 14,104 | 14.10 % | | pedestrian (any) | 17,706 | 17.71 % |
| turn right | 14,076 | 14.08 % | | traffic light | 6,299 | 6.30 % |
| lane reference | 13,480 | 13.48 % | | **right of way** | 6,728 | 6.73 % |
| exit | 3,833 | 3.83 % | | **unprotected** turn | 2,080 | 2.08 % |
| highway | 7,023 | 7.02 % | | railway/train crossing | 1,581 | 1.58 % |
| residential | 13,080 | 13.08 % | | tram | 1,355 | 1.35 % |
| u-turn | 745 | 0.74 % | | merge | 611 | 0.61 % |
| **posted speed limit** | **69,985** | **69.98 %** | | **carries a metric distance** | — | **96.74 %** |

> **Compare to what we have today:** route-v3 mints `roundabout` on **8 of 2,201 windows, from ONE episode**
> (0.36 %, MEASURED, ours, prior) with 24 u-turn confusions. L2D gives **3,532 roundabout episodes** — map-derived
> rather than kinematically guessed — and it *names the exit* ("take the highway exit on the slight right using the
> slight right lane"). That is roughly a **10× density improvement and a change of evidence class**: from an
> inference off our own ego track to a label off a road graph. Note also **6.73 % right-of-way** and **2.08 %
> unprotected-turn** episodes — the `RULECTX` slot, which we currently cannot mint at all.

Real instruction examples (MEASURED, ours, verbatim from the parquet):
- *"In 100 m when you have the right of way enter the roundabout"*
- *"In 200 m take the highway exit on the slight right using the slight right lane, observe the speed limit of 100 km/h"*
- *"In 150 m turn left at the intersection through the uncontrolled pedestrian crossing"*
- *"Turn left at the intersection following the right before left rule"*
- *"go straight on the tertiary road for 200 m, observe the speed limit of 30 km/h through the marked pedestrian crossing"*

**These are `⟨ROUTE, ROUTEDIST, SPEEDPOLICY, RULECTX⟩` in one sentence.** Note especially that **96.74 % of them
carry a metric distance** — which is exactly the argument made in `vocab.py`'s v1.1 candidate block
(*"`roundabout` is a description, `roundabout, exit in 40 m` is a command"*). L2D is the corpus that makes the
`ROUTEDIST` slot enrollable with real evidence instead of kinematic banding. **Escalation: this is a concrete
argument for the v1.1 vocabulary bump Sayed has been holding.**

**Accuracy limit, stated honestly.** The card claims cm-level RTK; I could **not verify a per-fix accuracy estimate
in the released data — UNVERIFIED.** The relevant question is *lane-level vs road-level truth*: a German lane is
~3.0–3.5 m, so road-level topology (which road, which junction, which exit) is safe under any plausible GNSS error,
while *lane assignment* is only safe if the RTK claim holds and the OSM geometry is lane-accurate — OSM usually is
not. **Treat road/junction/exit topology as trustworthy and lane index as suspect** until measured. Note that
`observation.state.lanes` gives a lane *count*, not the ego's lane *index*, which is consistent with that caution.

**Blocker #2 — agent state: NOT SOLVED.** L2D ships no boxes, no tracks, no depth, no perception labels at all.
This is the clean division of labour: **PhysicalAI has agents and no map; L2D has map and no agents.**

**Blocker #3 — the 20 s ceiling: SOLVED, and mechanically.** This is the finding I would most like reviewed.

L2D episodes are 30 s (max 300 frames @ 10 Hz; p50 300, mean 265). At face value that is barely better than our 20 s.
But the episodes are **sliding windows over continuous drives**:

| measurement (all 100,000 released episodes, from `meta/episodes`) | value |
|---|---|
| consecutive episodes that **overlap** in time | **90.8 %** |
| median stride between episode starts | **13.76 s** (so ~16 s of every 30 s episode is a duplicate of its neighbour) |
| shared frames per overlapping pair (spot check, 400 pairs) | **150 frames = 15.0 s**, in 368/400 pairs |
| **GPS disagreement on those shared frames** | **0.0000 m** — median, p95 and max. Byte-identical. |

The last row is the proof: overlapping episodes are **literally the same drive re-windowed**, so de-duplication by
unix timestamp is *exact* — not a matching heuristic, not a research problem, a `groupby`.

Reconstructing drives by chaining overlapping episodes (MEASURED, ours, over all 100,000):

| reconstructed continuous drives | value |
|---|---|
| number of drives | **9,217** |
| total unique drive time | **424 h** (vs 735 h of episode-time — the difference is the overlap) |
| duration p50 / p90 / p99 / max | **99 s / 346 s / 897 s / 2,121 s (35.4 min)** |
| share of episodes inside a drive ≥ 60 s | **94.7 %** |
| ≥ 120 s | **78.6 %** |
| ≥ 300 s | **44.9 %** |
| ≥ 600 s | **16.6 %** |

> **Put next to our measured ceiling** — PhysicalAI supervisable window starts: 5 s 74.3 %, 10 s 48.2 %, 15 s 22.0 %,
> **20 s 0.0 %** — L2D supports a **20 s** horizon on 100 % of episodes and a **120 s** horizon on 78.6 %.
> A roundabout traversal is ~10–15 s; a genuine strategic decision ("which exit, then which lane, then the next
> junction") is 60–120 s. **L2D is the only source in this document that reaches that horizon with real pixels.**

⚠️ **Two honesty notes on L2D scale.** (a) The card's *"5,000+ hours"* is Yaak's **full collection**; the **released**
L2D v3.0 tranche is **735 h of episodes / 424 h unique** (MEASURED, ours, from `meta/episodes` + `meta/info.json`
`total_frames = 19,042,712`). Quote 424 h, not 5,000 h. (b) The overlap means naive training on episodes double-counts
~50 % of frames — a silent 2× duplication that would corrupt any held-out split. **De-duplicate by timestamp before
splitting, and split on *drives*, never on episodes.** This is a data-leak trap of exactly the kind that made REF-A's
I-JEPA val number unusable.

**Blocker #4 — indicators: SOLVED for the ego, not for others.**
`action.discrete[:,1]` is the ego's own turn-signal from CAN. MEASURED (ours) over 1,392,787 frames:
**off 81.6 %, left 9.7 %, right 8.6 % → non-zero on 18.36 % of frames.** That is a dense, free, exact tactical label
for the one thing a forward camera fundamentally cannot see.

**I tested whether it is actually a *predictive* label**, since Sayed's example is *"wait for the vehicles to pass,
indicate, change lane"* — the ordering is the point. MEASURED (ours), 741 signal-onset events with ≥15 s of future:

| | LEFT (n=388) | RIGHT (n=353) |
|---|---|---|
| median net heading change over the next 15 s | −4.0° | +10.4° |
| onsets followed by a >20° heading change (a *turn*) | 45 % | 52 % |
| **median lead time from onset to a 20° heading change** | **6.7 s** | **5.5 s** |

**Direction predicted correctly, among the 358 onsets followed by a real (>20°) maneuver: 79.9 %.**

Two things follow. (a) The signal is a genuine **5–7 second advance** declaration of intent — precisely the horizon a
tactical layer plans over, and precisely the ordering *"indicate → then maneuver"*. (b) About **half** of signal
onsets are *not* followed by a large heading change: those are lane changes and merges, i.e. the signal separates
"turn" from "lane change" for free — a distinction our kinematic labeler must currently guess from net-yaw thresholds.

> **The comparison that decides this:** Cosmos-Reason2 reads turn direction from our frames at **57.1 %,
> CI [0.400, 0.745] — chance** (MEASURED, ours, prior). The CAN channel reads it at **79.9 %, 5–7 s early, at zero
> inference cost.** No VLM budget can buy this label; a CAN bus gives it away.

For **other agents'** indicators, L2D offers nothing — see §5.

### 3.3 Cost to acquire

MEASURED (ours) from the HF file listing: `data/` = 20 parquets ≈ **1.25 GB** (all 19 M frames of state, all 100 k
episodes) · `meta/episodes` = 98 MB · front camera = **1,492 × ~511 MB ≈ 763 GB** for everything ·
BEV `map` render = 42 × ~522 MB ≈ 22 GB.

| slice | size | ESTIMATED reasoning |
|---|---|---|
| **All state + metadata, zero video** | **~1.35 GB** | exact file sizes |
| ~2,000 episodes, front camera only | **~15 GB** | 100 k episodes / 1,492 files ≈ 67 ep/file → 30 files × 511 MB |
| ~10,000 episodes, front camera only | **~77 GB** | same arithmetic |
| Full front camera | 763 GB | exact |

**The state layer is 1.35 GB.** Every map, speed-limit, lane-count, turn-signal, instruction, waypoint and
policy-label measurement in this section came out of two files. **Any analysis of L2D's label quality can be done for
under 2 GB and zero GPU before a single frame of video is downloaded.** That is unusually cheap for a decision this
size.

### 3.4 Risks — the ones that would kill it

| risk | severity | how to retire it |
|---|---|---|
| **Camera intrinsics are NOT shipped.** MEASURED (ours): the repo's full 9,651-file listing contains exactly one calibration artefact, `extrinsic_RDF.yaml` (**extrinsics only**); `yaak-ai/L2D-v3` (977 files) has none at all. Our pipeline canonicalizes to `f_eff = F_REF = 266` and *asserts* it (`build_pai_cache.py`) | **HIGH** — without intrinsics we cannot prove the crop, and a wrong zoom ships silently (this is exactly the D-016 failure mode: the nominal path was ~434 px against a 266 target) | Estimate `f` from the horizon/vanishing point or a known-width object and validate against the extrinsics + measured ego speed; run the existing geometry falsifier (§9.5). If it cannot be pinned, admit L2D for the **strategic** head only, where absolute scale matters least — and say so in the arm's card |
| **GDPR / anonymization.** Real German driving-school footage; no anonymization statement on the card | **HIGH for redistribution**, low for internal training | Never re-host L2D frames until a face/plate check is done. Apache-2.0 grants the copyright; it does not grant the data-protection right |
| **Geographic concentration.** 30 German cities, one vehicle model (KIA Niro EV 2023), one rig | MEDIUM | Keep it as an *arm*, not a replacement. Our OOD numbers (comma 17.5 %, cosmos 29.4 % win-rate vs 49.7 % in-distribution, MEASURED, ours, prior) say cross-source transfer is not free |
| **Overlap-induced leakage** (§3.2) | **HIGH if missed** | De-dup by timestamp; split on reconstructed drives |
| **"Apache-2.0 on data"** — the licence text is written for software | LOW | Already the standing position for comma2k19 (MIT); one legal nod, per the sensor survey |
| **No loader exists.** `stack/tanitad/data/` has comma2k19 / cosmos_drive / physicalai / metadrive — no L2D | MEDIUM (cost, not risk) | ~2–3 eng-days, LeRobot v3 parquet + mp4 is a well-documented format |

### 3.5 The rest of the real-dataset field — why L2D wins it on *our* constraints

The 25-dataset licence and sensor survey already exists and is not repeated here
(`Research/2026-07-19-tanitdataset-v1-sensor-survey.md`, 2026-07-19). What changes with the hierarchy requirement is
the *ranking criterion*: we now need **map + agents + long horizon + a front camera + a licence we can use**, and
almost every rich dataset fails on the licence or the horizon rather than on the sensors.

| dataset | HD map / lane graph | agent tracks | **ego indicator** | scenario length | front camera | licence → our tier | verdict for the hierarchy |
|---|---|---|---|---|---|---|---|
| **L2D** | OSM topology per frame (class, lanes, limit, waypoints, NL instruction) | ❌ none | ✅ **CAN, 18.4 %** | 30 s → **reconstructs to 35 min** | ✅ 1080p @10 Hz | Apache-2.0 → **`ship`** | ✅ **the strategic + horizon source** |
| **nuPlan** | ✅ 19-layer map, lane connectors typed `STRAIGHT/LEFT/RIGHT/UTURN`, **traffic-light status**, `route_roadblock_ids`, 73 scenario types. ⚠️ **no roundabout primitive, no speed-limit layer** | ✅ auto tracks + ids | ❌ | **logs are continuous, ~4.7–8 min**; the 15 s "scenario" is a config knob | ✅ `CAM_F0` 2000×1200 **@10 Hz** — an exact rig match | CC-BY-NC-SA + Motional NC agreement; **an active commercial counterparty exists** → **`nc`** | 🥇 **the strongest real dataset on our criteria** — and the only NC one we could ever negotiate. ⚠️ camera covers only **128 h of 1,282 h**. **Go direct, not via OpenScene/NAVSIM (both 2 Hz — a 5× rig mismatch)** |
| **nuScenes** (+map, **+CAN bus**) | ✅ 11 layers | ✅ 1.4 M boxes | ✅ **`vehicle_monitor.left_signal`/`right_signal` @2 Hz** — plus a **`route`** polyline | 20 s scenes | ✅ 1600×900 @12 Hz | CC-BY-NC-SA → **`nc`** | 🟡 the only source pairing **ego blinker + HD map + tracks + route + front camera** — but NC and the same 20 s ceiling |
| **Argoverse 2** | ✅ per-log vector maps (`successors`, neighbours, `is_intersection`); no stop lines/lights/limits | ✅ 30 classes + ids | ❌ | Sensor **15 s**; TbV avg **54 s** but **no tracks** | ✅ 2048×1550 @20 Hz | CC-BY-NC-SA → **`nc`**. ⚠️ **Argo AI dissolved — the NC is permanent, no counterparty** | ❌ shorter than what we already have; best download ergonomics (`s5cmd --no-sign-request`) |
| **Waymo Open** (Perc/Motion/**E2E**) | ✅ road-graph; **E2E ships `intent` = GO_STRAIGHT/LEFT/RIGHT** | ✅ 12.6 M boxes | ❌ | 20 s / 9 s / 20 s | ✅ (Motion ships only VQ-GAN tokens) | ⛔ NC **+ registered-recipients-only**, and the terms **follow the trained weights into vehicle operation** | ⛔ **refuse** — see the escalation below |
| **ZOD** (Frames / Sequences / **Drives**) | ❌ **none** — image-space lane paint + signs only; roundabouts **not representable** | ✅ boxes to 245 m, ⚠️ **no track ids**, Sequences annotated on **one keyframe**, **Drives unannotated** | ✅ **`EgoVehicleControls.turn_indicator` @100 Hz** (Sequences + Drives only) | Sequences 20 s; **Drives ≈ minutes** | ✅ **single front 120° @10.1 Hz** — a literal rig match | CC-BY-**SA-4.0** → **`ship-sa`**; ⚠️ *are weights "Adapted Material"?* **legally untested** | 🟡 **the only commercially-usable blinker source with a matching camera** — but you supply the map and the tracks yourself |
| **comma2k19** *(already ours)* | ❌ | ❌ | 🟡 **raw CAN → decodable** (`leftBlinker`/`rightBlinker` exist in openpilot `CarState`); not pre-extracted | **1 min segments** | ✅ front | **MIT** → `ship` | 🟢 **the only MIT ego-blinker source, and we already hold it** — highway-only, map-free, but free to mine |
| **OpenDV / GenAD** | ❌ | ❌ | ❌ | ~49 min YouTube video @10 Hz | ✅ | ⛔ frames carry **no licence grant**; not redistributable and not reproducible (link rot) | ❌ |
| **PandaSet** | ❌ | ✅ cuboids | ❌ | 8 s | ✅ @10 Hz | CC-BY-4.0 claim **untraceable to a licensor doc**; official site 403 | ❌ too short, and the licence claim is now shaky |
| **KITTI-360** | ❌ (OSM for viz only) | ✅ persistent ids | ❌ | **11 seqs ≈12.5 min each** | ✅ | CC-BY-NC-SA **3.0** | 🟡 the only other real *long* front-camera set; no map, no route, NC |
| **INTERACTION** | ✅ Lanelet2, **explicitly includes roundabouts** | ✅ trajectories | ❌ | 16.5 h drone-view | ⛔ **no vehicle camera** | NC | ❌ cannot feed our encoder — noted only because it is the one real corpus built *around* roundabouts |
| **Bench2Drive** (CARLA-derived) | ✅ sim topology | ✅ ids + **`light_state`** (bit meanings **UNVERIFIED**) | ✅ (sim) | ~14.7 s @10 Hz | ✅ | ⚠️ **self-contradictory: repo LICENSE says CC-BY-NC-ND, HF card + paper say Apache-2.0** | ⚠️ **do not build on it until the licence is in writing** — ND would block publishing any derived label |
| ⛔ **Lyft L5 / Woven Planet** | ✅ (richest map of all — even `roundabout_circulation_sign`) | ✅ | ❌ | 25 s | ⛔ **no camera at all** | ⛔ **dataset is GONE** — hosts 404/NXDOMAIN, `l5kit` archived Oct-2025 | ⛔ remove from all plans |
| **PhysicalAI-AV** *(our corpus)* | ❌ (§1) | ✅ **`obstacle.offline`** | ❌ | 20 s hard | ✅ | NVIDIA AV licence → **`firewalled`**, internal-dev-only | ✅ **the agent source** |
| ⛔ **Waymax** | ✅ | ✅ | ❌ | 9 s | ⛔ none | ⛔ **§2.e bars foundation-model training** | ⛔ **refuse** |

**The synthesis.** **No real dataset satisfies all five of map + agents + long horizon + front camera + permissive
licence — the field has zero exceptions.** Everything with an HD map *and* agent tracks is CC-BY-NC-SA or stricter;
the only permissively-licensed member (ZOD) has neither a map nor track ids. Our answer is therefore a
**multi-source composition, not a single-source choice**:

- **agents** ← PhysicalAI `obstacle.offline`, on the parity corpus itself;
- **map + horizon + ego indicator** ← **L2D**, the only Apache-2.0 source with topology and multi-minute drives;
- **schema** ← **nuPlan**, whose typed lane connectors and `route_roadblock_ids` are the best published donor for our
  strategic/tactical vocabulary — and reading a published taxonomy needs no licence at all;
- **a second, commercially-clean blinker source** ← **comma2k19**, which **we already hold under MIT** and have never
  mined for `leftBlinker`/`rightBlinker`.

**Two findings worth their own line.**
1. ⚠️ **Correction to an earlier internal claim:** nuScenes' CAN-bus expansion *does* carry the ego blinker
   (`left_signal`, `right_signal`) and a route polyline. It is NC, so it changes nothing operationally — but the
   sensor survey's "no blinker" reading of nuScenes should be corrected.
2. 🟢 **Roundabout topology and the OSM licence.** *No AV dataset ships a roundabout primitive* — nuPlan, AV2,
   nuScenes and Lyft all leave it implicit as a lane-graph cycle. The only explicit source is **OpenStreetMap**
   (`junction=roundabout`). Licence-wise this is *good* news: **ODbL's ShareAlike binds Derivative Databases, not
   Produced Works**, and model weights are arguably a Produced Work requiring only attribution — making OSM **more
   weight-friendly than CC-BY-SA**. That is the licence L2D's map labels ultimately descend from, and it strengthens
   rather than weakens the L2D path. *(Confirm with counsel before relying on it.)*

⚠️ **Licence discipline reminder.** Every row above is already encoded in `SOURCE_REGISTRY`
(`stack/tanitad/lake/schema.py`), which raises `PermissionError` if a `gated-confidential` source is fed to the lake.
Waymax should be added there as a hard-fail. **PhysicalAI-AV imagery remains internal-dev-only by Sayed's ruling** —
note for the record that NVIDIA's own card says *"ready for commercial/non-commercial AV use per the license
agreement"*, which is *more* permissive than our internal rule; the internal rule stands and is not overridden by
this document.

---

## 4. Synthetic data — the ambitious answer, and why it is smaller than it looks

The instruction was to be ambitious here because a simulator owning the scene graph solves all three blockers at once.
That reasoning is correct and I want to preserve it — **but it applies to *simulators*, not to *generators*, and our
hardware excludes the one simulator that would deliver.**

### 4.1 The structural finding

**No video generator emits semantic ground truth — every one of them consumes it.** Cosmos-Transfer1,
Cosmos-Drive-Dreams, Vista, MagicDrive-V2, OpenDWM, OmniDreams and GAIA all take HD-map / 3D-box / trajectory
conditioning *in* and emit pixels *out* (PUBLISHED, per-model cards). A generator is an **appearance randomiser
bolted on top of labels you must already possess.** It therefore cannot solve blockers #1 or #2 by construction. It
can only improve robustness of a model whose labels came from somewhere else.

### 4.2 The hardware wall

- **Cosmos-Transfer1-7B is H100-class.** CARLA's own integration doc states it *"requires high-performance datacenter
  GPUs such as the NVIDIA H100"* (PUBLISHED). Our fleet is A40/A6000.
- Generation is also **length-capped at 121 frames ≈ 4.03 s** at 30 fps (PUBLISHED, model card). It does not fix the
  horizon blocker even if we had the GPUs.
- ESTIMATED throughput on an A40 with offloading: **1.5–4 clips (≈6–16 s of video) per GPU-hour** — reasoning: no
  FP8/Hopper attention path, 48 GB against a ~66 GB checkpoint forces offload, ~3–5× the H100 wall time. Compare
  CARLA at an ESTIMATED 500–1,000 twenty-second clips per GPU-hour. **Two to three orders of magnitude.**
- **AlpaSim** (`NVlabs/alpasim`, Apache-2.0) is the credible open successor to DRIVE Sim and emits route, lane graph,
  agent boxes and traffic-light state — but its NuRec scenes are **~900 reconstructions of 20 s each** (PUBLISHED), so
  it *inherits our exact ceiling*, has no indicator labels, and its throughput is UNVERIFIED. **Scope it as an
  evaluation platform, not a data factory.** (An intake already exists:
  `Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/`.)

### 4.3 CARLA — the best label emitter we own, and we cannot photograph it

CARLA 0.9.16 is installed locally (`C:/Users/Admin/carla`, with HDMaps) and natively emits, exactly and free:
OpenDRIVE lane topology (`Map.get_topology()`, `Junction.get_waypoints()` → **roundabout exit enumeration is exact**),
`RoadOption` per route step (LEFT/RIGHT/STRAIGHT/LANEFOLLOW/CHANGELANELEFT/CHANGELANERIGHT — *already a tactical
vocabulary*), agent boxes with persistent ids, traffic-light state, exact TTC, ego route — and **unbounded episode
length**. Licence: code MIT, assets CC-BY (UE4.26 EULA also attaches — one legal read before anything ships).

**And it gives other vehicles' blinkers** — `carla.VehicleLightState` bitflags
(`RightBlinker 0x10`, `LeftBlinker 0x20`, `Brake 0x8`), readable per actor via `Vehicle.get_light_state()`
(PUBLISHED). Two traps that must be written down now:
1. The Traffic Manager **does not drive vehicle lights by default** — *"vehicle lights … are never updated"* unless
   you call `tm.update_vehicle_lights(actor, True)` **per vehicle** (PUBLISHED). Miss it and the whole corpus has
   blinkers permanently off while the labels say otherwise.
2. **Not all blueprints have lights.** Only Generation-2 and special vehicles do; the catalogue lists
   `"Has lights": True/False` per vehicle (PUBLISHED). Filter the blueprint library on `has_lights` before spawning
   traffic, or a large fraction of agents will be *physically incapable* of signalling while labelled as signalling.
   Silent label/pixel divergence.

🔴 **The blocker: we cannot render CARLA on our pods.** MEASURED (ours, 2026-07-09,
`Tools&DevEnv/Research/2026-07-09-carla-render-blocker-and-testsuite-io-cost.md`): RunPod launches containers with
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so no Vulkan ICD / EGL device exists in-container; `nvidia-smi` works,
`vulkaninfo` returns NULL. This is host-side and **cannot be fixed from inside the container**. CARLA runs
`-nullrhi` (state only, ~1,400 ticks/s) on pod1/2/3/eval. A turnkey recovery recipe exists in that note (a template
with `NVIDIA_DRIVER_CAPABILITIES=all`, gated on a single `vulkaninfo` probe, plus Xvfb for the UE4-Vulkan-offscreen
bug), but it needs a **new pod type and Sayed's supervised go/no-go**.

The local RTX-4060 box *can* render, at an ESTIMATED small fraction of an A40's rate and with the Google-Drive I/O tax
— fine for a **hundred-clip proof**, not for a corpus.

> **Also relevant, MEASURED (ours):** the ESTIMATED 250 px risk. At our image class an amber blinker on a vehicle
> 20–30 m ahead is ~1–3 pixels. The *label* would be perfect; the *evidence* may be below the encoder's resolution.
> Any indicator-reading objective must first measure detectability versus range, or we will train a head on a signal
> that is not in the pixels.

### 4.4 ⛔ Waymax — licence exclusion, escalate

Waymax's *License Agreement for Non-Commercial Use* §2.e states verbatim that you *"may not use the Waymax source
code or Documentation (or any derivative thereof) to train or otherwise develop or improve (directly or indirectly)
an artificial intelligence foundation model … or any model distilled or fine-tuned therefrom"* (PUBLISHED). §2.b
additionally bars commercial scenario simulation and use in production systems. **TanitAD is squarely inside the
prohibition. Waymax must not enter the pipeline in any form.** Add it to the lake's refusal list next to PhysicalAI-AV.

---

## 5. Tactical signals specifically — indicators, and what is actually knowable

Sayed named turn signals. The honest decomposition:

| the signal | is it knowable? | from where |
|---|---|---|
| **Ego's own indicator** | **YES, free, dense, exact, and 5–7 s predictive** — but *never* from a forward camera | **Four verified sources:** ① **L2D** `action.discrete[:,1]` — Apache-2.0, 18.36 % non-zero, 79.9 % direction accuracy at a 5.5–6.7 s lead (MEASURED, ours, §3.2). ② **ZOD** `EgoVehicleControls.turn_indicator` @100 Hz — CC-BY-SA, Sequences+Drives only. ③ **nuScenes CAN** `left_signal`/`right_signal` @2 Hz — NC. ④ **comma2k19** raw CAN, decodable via opendbc (`leftBlinker`/`rightBlinker` in openpilot `CarState`) — **MIT, and already in our lake** |
| **Other agents' indicators, real data** | ❌ **No AV dataset annotates it.** Verified against the actual label schemas, not blogs: nuScenes' annotator instructions list exactly four attributes and no light field; AV2's 30-class taxonomy has none; Waymo has none in any of its three sets; ZOD's object attributes are `unclear`/`occlusion_ratio`/`relative_position`/`with_rider` | nowhere — this is a real gap in the public landscape, not a search failure |
| **Other agents' indicators, synthetic** | **YES, exact** | **CARLA only** (`VehicleLightState`) — and we cannot render it (§4.3) |
| **Other agents' indicators, from pixels** | a genuine perception problem: small, blinking, often occluded, amber-on-amber | would have to be learned; see below |
| **Brake lights** (the easier cousin) | plausibly readable — large, red, high-contrast, and *causally redundant* with the lead's decel we can already compute from `obstacle.offline` | derivable |

**Be sceptical about the VLM route here.** MEASURED (ours, prior): Cosmos-Reason2 is at **chance on turn direction —
57.1 %, CI [0.400, 0.745]** — and the enum-order probe proved the left bias belongs to the *model*, not the prompt. A
model that cannot say which way the *ego* is turning will not read a 2-pixel blinker on a car 30 m away. **Do not
task the VLM with indicator reading.**

**The practical conclusion.** Split the requirement:
- *"indicate, then change lane"* as an **ego action** → **L2D, today, free.** This is most of what Sayed's sentence
  actually asks for: the planner must learn to *emit* an indicator before a lane change, and L2D supervises exactly
  that. 🟢 **And there is a zero-download second source: `comma2k19` is already in our lake under MIT and carries raw
  CAN frames from which `leftBlinker`/`rightBlinker` are decodable via `opendbc`.** We have never mined it. Highway
  lane changes with a real blinker label, commercially clean, for the cost of a decoder — worth an afternoon.
- *"wait for the vehicles to pass"* → **`obstacle.offline`, today, free.** Gap, closing speed, TTC, and the
  occupancy of the target lane are all geometry, and geometry we now have.
- *"read the other car's blinker"* → **defer.** It is the only piece with no credible cheap source, and it is also
  the least load-bearing: a tactical planner that reasons over *observed motion* of neighbours (which we now have)
  captures most of the decision value. Revisit only if a graphics pod appears.

---

## 6. VLM extraction — scope it down, decisively

A Cosmos-Reason2 production run is live on pod3 (`vlm-production`, PID 53351, do not disturb). Established, all
MEASURED (ours, prior): **0 parse failures**, ~99 % token adherence, ~257 clips/GPU-hr — the engine is sound. And:
it **cannot mint ROUTE** (chance on direction), and it **fabricated VTARGET band edges on 48 % of pilot clips**.

**Decision, per slot:**

| slot | VLM? | why |
|---|---|---|
| Scene tags (weather, time-of-day, road type, surface, traffic density) | ✅ **yes** | non-geometric, non-metric, verifiable by eye |
| Caption / Chain-of-Causation / risk | ✅ **yes** | its actual strength; the 1,740 human CoC examples (§2.3) are the style reference |
| Speed-limit **sign reading** | 🟡 **pilot it, don't assume it** | plausible (OCR of a large high-contrast disc) but it is *reading a number*, and this model fabricates numbers. **Gate it on a labelled 100-sign precision test before any use.** Note L2D supplies speed limits from OSM for free, which is the better source where available |
| ROUTE / turn direction | ❌ **never** | measured at chance; the bias is the model's |
| Any metric quantity (gap, TTC, VTARGET, distances) | ❌ **never** | 48 % fabrication; and `obstacle.offline` now measures these exactly |
| **Location names** | ❌ **drop it — decided** | Three reasons. (1) **Topology is what a planner consumes; names are not.** "Roundabout, second exit, 40 m" is actionable; "Munich" is not. (2) It is unverifiable — with no geo in the corpus (§1) there is nothing to check a guessed name against, so a wrong name is undetectable, which is the worst property a label can have. (3) It is a **privacy regression**: NVIDIA deliberately coarsened location to *country*; re-deriving finer location from imagery and storing it re-creates what they removed. **Do not ask the VLM where it is.** |
| Other agents' indicators | ❌ **never** | §5 |

**Net:** the VLM's job shrinks to *scene semantics and language* — which is what it is good at — and the geometric
and topological slots move to `obstacle.offline` and L2D respectively, where they are exact.

---

## 7. The ranked decision table

Cost columns: **eng-days** = our engineering time · **GPU-days** = A40-class · **$** = marginal cloud spend beyond
pods we already pay for. Parity impact is judged against `physicalai-train-e438721ae894` (2,376 eps, skip-hash
`f09e44db`).

| # | option | eng-days | GPU-days | $ | unblocks **map** | unblocks **agents** | unblocks **horizon** | parity impact | licence risk | evidence class |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 🔴 **Ingest `obstacle.offline`** into the existing lake stubs — **GATE FAILED 2026-07-21, do not run on the tactical-prediction argument** (`Research/2026-07-21-lead-state-gate.md`) | **2–3** | **0** | ~0 (12.4 GB egress) | ❌ | ⚠️ tracks/classes/geometry are exact, but adding them to an ego-state regressor moves the 2 s longitudinal held-out error by **+1.16 % [−0.92, +3.19]** — pre-registered FAIL | ❌ | ✅ **none** — labels re-derived on the *same* episodes; no re-selection | ✅ none (already-licensed source, internal-dev-only, no new terms) | MEASURED (ours) |
| **2** | **L2D pilot** — state layer + ~2 k episodes front camera, as a **new arm** | **3–5** | ~0.5 (feature extraction) | ~0 (≈16 GB) | ✅ **fully** — OSM class/lanes/limits/waypoints + NL instructions with distances | ❌ | ✅ **fully** — drives to 35.4 min, 78.6 % ≥ 120 s | ✅ none *if* kept as a separate arm/tier | 🟡 low — Apache-2.0, `ship` tier already; **GDPR check before any re-host** | MEASURED (ours) |
| **3** | **L2D scale-up** — 10 k+ episodes, drive-level splits, joint arm | 5–8 | 2–4 | ~0 (≈77 GB) | ✅ | ❌ | ✅ | ✅ new arm only | 🟡 same | ESTIMATED |
| 4 | **Enrol `ROUTEDIST`/`TACDIST` (vocab v1.1)** using L2D's distances | 1–2 | 0 | 0 | ✅ makes the map label *actionable* | ❌ | — | ⚠️ vocabulary version bump — Sayed's call, migration note required | ✅ none | MEASURED (ours) — the tokens already exist in `vocab.py` |
| 5 | **Take the 1,740 CoC reasoning labels** as a gold eval slice | 0.5 | 0 | 0 | ❌ | ❌ | ❌ | ✅ none (43-clip overlap; eval only) | ✅ none | MEASURED (ours) |
| **5b** | **Mine `leftBlinker`/`rightBlinker` out of `comma2k19`'s raw CAN** (opendbc) — we already hold it | **0.5–1** | **0** | **0 — zero download** | ❌ | ❌ | 🟡 1-min segments | ✅ existing lake source | ✅ **MIT** — the only commercially-clean blinker source | PUBLISHED (openpilot `CarState` schema) |
| 6 | **CARLA state-only corpus** (`-nullrhi`, no pixels) for label-structure and closed-loop | 3–6 | 1–2 | 0 | ✅ (perfect, synthetic) | ✅ (perfect, synthetic) | ✅ (unbounded) | ✅ new arm | ✅ MIT/CC-BY (+UE EULA read) | MEASURED (ours) — render blocker; ESTIMATED throughput |
| 7 | **CARLA with pixels** on a graphics-capable pod | 4–7 + a **supervised pod op** | 2–5 | new pod hourly | ✅ | ✅ | ✅ | ✅ new arm | ✅ | ESTIMATED; gated on ONE `vulkaninfo` probe |
| 8 | **VLM scene/language pass** (already running) | in flight | in flight | 0 | ❌ (proven: cannot mint ROUTE) | ❌ | ❌ | ✅ | ✅ | MEASURED (ours) |
| 9 | **AlpaSim + NuRec** as a closed-loop *eval* platform | 5–10 | UNVERIFIED | 0 | ✅ | ✅ | ❌ (20 s scenes) | ✅ eval only | ✅ Apache-2.0 | PUBLISHED |
| 10 | **Generative video (Cosmos-Transfer / OpenDWM / Vista)** | 10+ | H100-class | rented H100 | ❌ **by construction** | ❌ **by construction** | ❌ (121-frame cap) | ✅ | ✅ (CC-BY/NVIDIA OML) | PUBLISHED |
| 11 | **nuPlan commercial-licence conversation with Motional** (the only NC dataset with a live counterparty) | 0 eng — a business ask | — | UNVERIFIED $ | ✅ best real map + route | ✅ | ✅ 4.7–8 min logs | ✅ new arm | 🟡 NC today; negotiable | PUBLISHED |
| ⛔ | **Waymo Open / WOD-E2E** | — | — | — | — | — | — | — | ⛔ terms follow the **trained weights** into vehicle operation | PUBLISHED |
| ⛔ | **Waymax** | — | — | — | — | — | — | — | ⛔ **§2.e forbids training foundation models** | PUBLISHED |
| ⛔ | **Lyft L5 / Woven Planet** | — | — | — | — | — | — | — | ⛔ **dataset no longer exists** (hosts 404/NXDOMAIN, `l5kit` archived) | PUBLISHED |
| ⛔ | **Map-matching PhysicalAI to OSM** | — | — | — | ❌ **impossible** | — | — | — | — | MEASURED (ours) — no geo exists (§1) |

**Read of the table.** Options 1 and 2 are both *cheap*, both *parity-safe*, and between them they cover all three
blockers. Nothing below rank 3 changes that conclusion. Options 6–7 (CARLA) are the only route to *other agents'
indicators* and to *unbounded* horizon, and both are gated on hardware we do not currently have.

---

## 8. What each option does **not** fix — stated explicitly

| option | map/topology | agent state | 20 s horizon | other-agents' indicators | traffic-light state |
|---|---|---|---|---|---|
| `obstacle.offline` | ❌ **no** | ✅ yes | ❌ **no** | ❌ no | ❌ no |
| L2D | ✅ yes (road-level; lane-level UNVERIFIED) | ❌ **no** | ✅ yes (to 35 min) | ❌ no (**ego's own: yes**) | ❌ no (only "traffic light" as an instruction word) |
| CARLA (if renderable) | ✅ yes | ✅ yes | ✅ yes (unbounded) | ✅ **yes — uniquely** | ✅ yes |
| Cosmos-Drive-Dreams (held clips) | ✅ HD-map ships with them | ✅ 4D tracks ship | ❌ no | ❌ no | ❌ no (map geometry only) |
| AlpaSim/NuRec | ✅ yes | ✅ yes | ❌ **no — 20 s scenes** | ❌ no | ✅ yes |
| Generative video | ❌ **consumes labels** | ❌ **consumes labels** | ❌ 121-frame cap | ❌ no | ❌ no |
| VLM pass | ❌ **proven no** | ❌ no (metric slots fabricate) | ❌ no | ❌ no | 🟡 maybe, unmeasured |

**The one gap nothing cheap closes: other agents' indicators.** Accept it, and design the tactical layer to reason
over *observed neighbour motion* (which `obstacle.offline` supplies exactly) rather than over signalled intent.

---

## 9. Staged plan

### This week (≈3 eng-days, 0 GPU-days, ~13 GB)

1. **`obstacle.offline` ingest.** Download the 197 phase-0 chunks (12.4 GB), build a per-episode agent-track sidecar
   aligned to the existing 10 Hz episode grid, and fill `enrich.vlm_pending_lead_state()` from geometry instead of
   from a VLM. Recompute the §2 density statistics corpus-wide and record them. **Non-negotiable constraints:** write
   a *sidecar keyed by `episode_id`*, never a new `_epcache`; do not touch `physicalai-train-e438721ae894`; stamp
   provenance as an autolabel, not `human`.
2. **Run the falsifier in §10 before writing any head.** It is CPU-only and answers whether the labels carry
   decision-relevant information at all.
3. **Pull the 1,740 CoC reasoning labels** (one file) into the hub as a gold eval slice; note the 43-clip overlap.
4. **Encode the refusal list** in `SOURCE_REGISTRY` — Waymax, Waymo Open/WOD-E2E, OpenDV frames, Lyft L5 — with
   citations (§11.4).
5. **Half-day freebie: decode `leftBlinker`/`rightBlinker` from `comma2k19`'s raw CAN** via `opendbc`. Zero download
   (we hold it), MIT licence, and it gives a second, commercially-clean ego-indicator corpus to cross-check L2D's
   79.9 %/5–7 s numbers on a completely different vehicle, country and road type.

### This month (≈5 eng-days, ~1 GPU-day, ~16 GB)

5. **L2D pilot.** State layer first (1.35 GB, zero GPU) → verify instruction-vocabulary coverage corpus-wide, build
   the drive-reconstruction index, and measure turn-signal→maneuver lead time. Then ~2,000 episodes of
   `front_left` video (≈15 GB) and the **geometry gate**: can we canonicalize to `f_eff = F_REF = 266`? The existing
   ZOD/pinhole falsifier (`Research/2026-07-18-zod-loader-and-geometry-falsifier.md`) is the template. If intrinsics
   are genuinely absent, that is a HIGH-severity finding and must be reported before any training use.
6. **Write `stack/tanitad/data/l2d.py`** to the same episode contract as `physicalai.py`, with a **drive-level split**
   and timestamp de-duplication built in (not bolted on).
7. **Price the `ROUTEDIST` v1.1 vocabulary bump** with L2D's real distances as the evidence, and put it to Sayed.
8. **Decide the graphics-pod question.** The CARLA recipe is turnkey and gated on one `vulkaninfo` probe. It costs one
   pod-hour to answer. It is the only path to other-agent indicators and to unbounded episodes.

### Genuine research bets (do not schedule; propose)

9. **Self-supervised lane-graph / topology inference from the camera.** If a head can infer junction topology from
   pixels, the map dependency weakens for *every* source including our parity corpus. High value, unproven.
10. **Few-label strategic learning.** The strategic layer plausibly needs **far fewer** labelled windows than the
    operative one — it emits a token every few seconds, not a trajectory every 100 ms. If ~2,000 map-labelled L2D
    episodes suffice to train the strategic head while the operative head trains on the full unlabelled corpus, the
    whole data problem shrinks by an order of magnitude. **This is the cheapest possible framing of the entire
    strategy and should be tested early** — it is a label-efficiency curve, not a new corpus.
11. **Ego-trajectory clustering across repeated traversals** (the lake's `geo_cell` multi-traversal signal, ~100 m
    grid, `GEO_CELL_DEG = 0.001`). ❌ **Inapplicable to PhysicalAI** (no geo, §1) — but **directly applicable to L2D**,
    which has lat/lon on every frame and 9,217 drives across 30 cities. Repeated traversals of the same junction give
    a free consistency check on the OSM labels and a way to *infer* topology where OSM is wrong.
12. **Clip stitching for PhysicalAI.** ❌ **Assessed and rejected.** No absolute clock, no geo, and chunks are not
    sessions (max intra-chunk metadata group = 5.1 %, MEASURED, ours). Adjacency could only be recovered by
    image-retrieval over all 306,152 clips, which needs the full 133 TB. Not viable. *(L2D needed none of this — its
    packaging is invertible by timestamp.)*

---

## 10. The cheapest falsifying experiment for the top option

**Top option = ingest `obstacle.offline`.** Its premise is: *"other-agent state carries decision-relevant information
that our current label set does not."* If that is false, the tactical layer does not need agents and the whole ranking
changes.

### The experiment — the lead-state longitudinal information test

**Why this one.** Our measured dominant residual is **longitudinal**: 83 % of the flagship's 2 s error is along-track
at 30 k, with a +0.66 m/s speed over-prediction (MEASURED, ours, prior — `flagship-longitudinal-lever`). Longitudinal
error is *exactly* what a lead vehicle causes. If lead geometry does not explain a material share of it, agent state
is not our bottleneck and we should not spend engineering on it.

**Protocol.**
1. Download **one** obstacle chunk that is in the corpus — **62.7 MB** (chunk 0036 is already downloaded).
2. Resample tracks to the 10 Hz episode grid (the §2 code path) and compute per-frame `{lead_present, gap_m,
   closing_speed_ms, ttc_s, target_lane_occupied}`.
3. Fit two small regressors for the ego's **future 2 s longitudinal displacement**:
   **(A)** ego state only — current `v`, `a`, recent yaw rate (this is what the model effectively has today);
   **(B)** ego state **+ lead state**.
4. Read: the **paired reduction in held-out longitudinal error**, with an **episode-cluster bootstrap** interval
   (`taniteval/ci.py` — never the `overlapping_holdout_se` block, which is 1.28–2.06× too narrow).

**Pre-registered decision rule.**
- **≥ 15 % paired reduction, interval excluding 0** → the premise holds; ingest all 197 chunks and wire the tactical
  slots. *(15 % because the longitudinal share of total error is ~83 %, so a 15 % longitudinal gain is ~12 % of total
  2 s ADE — larger than any single lever we have measured this month.)*
- **≤ 5 %, or an interval spanning 0** → **the premise is falsified.** Agent state is not the tactical bottleneck;
  stop at the 62.7 MB already spent, re-rank toward L2D-only, and report the negative result.
- **In between** → quantify and report; do not round to either story.

**Cost: ~1 eng-day, 0 GPU-hours, 62.7 MB, $0.** Runs on the dev box. It is decisive because it tests the *premise*
(does agent state inform the ego's decision) rather than an *implementation* (does our head learn it) — a negative
result cannot be blamed on a training bug.

**A second falsifier worth pre-registering, for the strategic path (option 2).** Before building a strategic head at
all: take L2D's 1.35 GB state layer, and ask whether the OSM-derived route token at t+H is predictable **above the
marginal** from a *frozen* DINOv2 embedding of the current frame (linear probe, ~2,000 episodes, ~1 GPU-hour). If the
answer is no, the strategic layer has no visual evidence to plan on and the problem is representational, not
data-scarcity — and no corpus fixes it. **Cost: ~1 eng-day + ~1 GPU-hour + 16 GB.**

---

## 11. Escalations — decisions that need Sayed

1. 🔴 **Vocabulary v1.1 (`ROUTEDIST`/`TACDIST`).** L2D's instructions ship the distance natively
   (*"In 100 m … enter the roundabout"*). `vocab.py` already holds the tokens and the banding function, pinned by
   tests, waiting on his call. This is the first time we have had **real** evidence for those bands rather than
   kinematic guesses. Enrolling is a version bump with a migration note.
2. 🔴 **The graphics-capable pod.** One `vulkaninfo` probe on a new pod type decides whether *other agents'
   indicators*, *traffic-light state* and *unbounded episode length* are available to this program at all. Cheap to
   answer, and it has been pending since 2026-07-09.
3. 🟡 **L2D licence sign-off** — Apache-2.0 is clean for training; **re-hosting real German driving-school footage is
   a data-protection question the licence does not answer.** Internal training needs no decision; publication does.
4. 🟡 **Licence refusals to encode in `SOURCE_REGISTRY` as hard-fails**, so an ingestor cannot admit them by accident:
   **Waymax** (§2.e bars foundation-model training), **Waymo Open / WOD-E2E** (terms follow the *trained weights*
   into vehicle operation — the one licence here that reaches our end product, so even the internal `nc` tier is
   unsafe), **OpenDV frames** (no licence grant from anyone), and **Lyft L5** (the dataset no longer exists; any
   archive copy is unlicensed).
5. 🟡 **Two licence texts a human must read before either source is tiered.** (a) The nuScenes/nuPlan terms pages are
   JS-rendered and could not be fetched — the licence is CC-BY-NC-SA-4.0 *"with modifications outlined in
   nuscenes.org/terms-of-use"* and **those modifications are UNVERIFIED**. (b) **Bench2Drive's licence is
   self-contradictory** — its repo `LICENSE` and README say **CC-BY-NC-ND-4.0** (NoDerivatives would block publishing
   *any* derived label) while its HF card and paper say **Apache-2.0**. Do not build on it until that is in writing.
6. 🟢 **Corrections to the record.** (a) The 2026-07-19 sensor survey lists nuScenes as having no blinker; it **does**
   — `vehicle_monitor.left_signal`/`right_signal` in the CAN-bus expansion, plus a `route` polyline. (b) PandaSet's
   "CC-BY-4.0 / commercial-OK" claim traces only to a third-party HF tag; the official site returns 403. Both belong
   in the survey's next revision.
7. 🟢 **Watch item, no action:** NVIDIA has stated an intent to add XODR maps to PhysicalAI-AV. If that ships, the map
   blocker closes on the parity corpus for free. Check the Version History table monthly.

---

## 12. Measurement log — provenance for every "MEASURED (ours)" number above

All run 2026-07-21 on the dev box (`C:/Users/Admin/venvs/tanitad`), read-only, **no GPU used, no pod disturbed**
(pod3's `vlm-production` and pod1's `exp-a` untouched; pod2 was probed read-only over ssh).

| # | claim | source artefact |
|---|---|---|
| 1 | no geo in PhysicalAI metadata | `C:/Users/Admin/tanitad-data/physicalai/{clip_index.parquet, metadata/data_collection.parquet}` — column lists |
| 2 | card keyword sweep (0 GPS/lat/lon/location) | HF `nvidia/PhysicalAI-Autonomous-Vehicles/README.md`, 31,935 B |
| 3 | pose is clip-local | `labels/egomotion.offline/egomotion.offline.chunk_0036.zip`, row 0 = origin/identity |
| 4 | feature presence 97.44 % / 96.90 % | `metadata/feature_presence.parquet` (306,152 × 36) joined to `r0/phase0_selection.parquet` |
| 5 | chunks are not sessions (median max-group 5.1 %) | groupby over `clip_index ⋈ data_collection`, 3,146 chunks |
| 6 | obstacle schema, classes, density, lead-frac 45.2 % | `obstacle.offline.chunk_0036.zip` (62.7 MB, 98 clips), 10 Hz resample w/ 0.5 s gap guard |
| 6b | cross-region generalisation (US 45.2 % / DE 53.4 % / SK 28.0 %) | chunks 0036 / 1807 / 2490, 271 clips, same resample |
| 7 | reasoning labels, clusters, 43-clip overlap | `reasoning/ood_reasoning.parquet` (1,740 × 4) |
| 8 | L2D schema, turn-signal 18.36 %, road/lane/limit distributions, policy split | `yaak-ai/L2D` `data/chunk-000/file-000.parquet` (5,307 eps / 1,392,787 frames) |
| 9 | L2D drive reconstruction (9,217 drives, 424 h, max 35.4 min) **and** the corpus-wide instruction table (100,000 eps, 28,949 distinct strings) | `yaak-ai/L2D` `meta/episodes/chunk-000/file-000.parquet` (all 100,000 episodes) |
| 10 | overlapping episodes are byte-identical (0.0000 m GPS) | 400-pair timestamp join on the same data file |
| 11 | L2D file sizes / download slices | `HfApi.get_paths_info` + `list_repo_files` (9,651 files) |
| 12 | `cam_front_left` is the reference camera; **no intrinsics anywhere in either repo** | `yaak-ai/L2D/extrinsic_RDF.yaml` + full file listings (L2D 9,651 files, L2D-v3 977 files) |
| 12b | turn-signal → maneuver: 79.9 % direction accuracy, 5.5–6.7 s lead, 741 onset events | `yaak-ai/L2D` `data/chunk-000/file-000.parquet`, heading unwrap over 15 s post-onset |
| 13 | CARLA render blocker on RunPod | prior: `Tools&DevEnv/Research/2026-07-09-carla-render-blocker-and-testsuite-io-cost.md` |
| 14 | `l2d` already in `SOURCE_REGISTRY` as `ship` | `stack/tanitad/lake/schema.py:68` |
| 15 | `lead_state` stubs already wired | `stack/tanitad/lake/enrich.py:61`, `stack/tanitad/lake/curation.py:77-170` |

**Prior-run numbers quoted but not re-measured here** (traceable to the loop's own records, flagged as such where
used): route-v3 roundabout 8/2201 windows · window-start horizon table (5 s 74.3 % … 20 s 0.0 %) · Reason2 turn
direction 57.1 % CI [0.400, 0.745] · VTARGET fabrication 48 % · OOD win-rates 17.5 % / 29.4 % / 49.7 % · longitudinal
error share 83 % / +0.66 m/s.
