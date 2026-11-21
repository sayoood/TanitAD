# v2 corpus — 50 h maneuver-balanced selection design (Phase 1)

**Date:** 2026-07-24 · **Discipline:** Data Engineering ·
**Slug:** `2026-07-24-v2-corpus-50h-balanced`
**Scope:** SELECTION DESIGN ONLY — cheap egomotion, **no camera downloaded**.
Phase 2 (camera fetch + epcache build) is gated on the budget in §7.

**New corpus key:** `physicalai-v2bal-4b7eeeac222d` (sha1 of the sorted 9,000
clip_ids + target + K). This is a **NEW canonical corpus that BREAKS PARITY** with
`physicalai-train-e438721ae894` by design (Sayed's ruling). The parity set and its
key are **untouched**; the from-scratch flagship on the 13 h parity set is not
affected.

> **One-line answer for Sayed:** 50 h is comfortably reachable — the 197 egomotion
> chunks we already hold locally contain **18,731 moving clips (104.6 h)**, drawn
> from the *same* country-stratified chunk universe the parity set came from. A
> greedy water-filling selection of **9,000 clips (50.25 h)** hits an explicit
> balanced target **exactly**: **turns 14.25 % → 28.0 %** (L/R balanced), lane-keep
> **59.6 % → 45.0 %**, while *keeping* the speed balance (highway held at 38 %).
> Per-clip junction presence rises **37.7 % → 61.3 %**, stop presence to 27.3 %.
> **The one gate: the 256 px feature cache is ~982 GB, 2.1× the ~466 GB MooseFS
> pod quota** — Phase 2 needs storage provisioning (§7). Kinematic balancing
> **cannot** create semantic coverage (lights/roundabouts/pedestrians) — that stays
> a separate VLM/map effort (§8).

---

## 1. Feasibility — 50 h is reachable from LOCAL egomotion alone ✅ (MEASURED)

| Quantity | Value | Source |
|---|---|---|
| PhysicalAI-AV total clips | 306,152 | `clip_index.parquet` |
| valid & train | 153,625 (~858 h) | idx.split==train |
| Total chunks | 3,146 (~97 clips/chunk) | idx.chunk |
| **Egomotion chunks already local** | **197** (~7.8 GB) | `labels/egomotion/*.zip` |
| valid&train clips in local chunks | 18,988 | scored |
| **Moving clips (parked/degenerate excluded)** | **18,731 = 104.6 h** | §3 filter |
| Parked fraction (mean_v<1) | 1.4 % | MEASURED |

**We do not need to download any more egomotion.** The 197 local chunks are exactly
the chunks the parity selection was drawn from (§6), span all **25 countries**
(~5 %/country — the original pick was country-stratified), and give **2× headroom**
over the 9,000 clips (50 h) the design needs. Evidence class: **MEASURED** —
`score_v2_pool.py` on the dev box (venv `tanitad`, HF reachable via
`truststore.inject_into_ssl()`; not needed here since egomotion was already local).

## 2. Scoring window — the camera-aligned first 20 s (MEASURED, faithful)

Each clip_id has an egomotion parquet spanning the **whole recording** (~114 s mean,
~35 Hz native) but only **one front_wide mp4 of ~20.1 s** (605 frames). Measured on
40 clip pairs: the camera excerpt sits at the **START** of the recording —
`cam_start_frac = (cam_t0−ego_t0)/ego_span ∈ [0.000, 0.008]`. So the training
episode is the **first 20.1 s** of the egomotion, which we reproduce with **no
camera**: `t_query = linspace(ego_t0, ego_t0+20.1 s, 201)`, then `poses[2:]` (the
`build_episode` stacking drop). Poses use `tanitad.data.physicalai.signals_at`
verbatim (x, y, quaternion-yaw, `hypot(vx,vy)`); maneuvers use `refb_labels`
verbatim — **no re-implementation**, so labels equal what the trainer will see.

**Cross-check (comparability proof):** the aggregate over all 18,988 valid&train
clips reproduces the parity profile — turns **17.2 %** (parity 14.25 %), speed
regime **7.1 / 51 / 42** (parity 7.8 / 46 / 46). The pool is naturally a touch
richer in turns/city than the parity subset, as expected.

## 3. Candidate pool scoring (MEASURED — `v2_pool_scored.parquet`, 18,988 rows)

Per clip: v1 + v2 maneuver histograms, turn/junction/stop presence, net & cumulative
heading, speed regime, curvature, country/hour. **Parked/degenerate excluded** by a
*gentle* driving gate (keeps highway — we are NOT filtering to urban like the old R0
scorer): `mean_v ≥ 1 m/s AND stop_frac < 0.9 AND distance ≥ 20 m` → **18,731 moving
clips**. Pool structure that bounds the design:

| Property | Pool value |
|---|---|
| clips with ≥1 turn timestep | 50.3 % |
| clips with a junction (v2.1 tight-transient) | 42.8 % |
| **clips with a full stop (v<0.5 m/s)** | **19.3 % ← scarcest** |
| clips with a brake_stop label | 60.3 % |
| **turn ceiling** — top-9,000 by turn-count | **36.0 % turns** (highway collapses to 25 %) |

The **36 % turn ceiling** is what makes a 28 % target feasible *with room to spare*
for holding the speed balance; stops are the true scarcity (max ~19 % clip-level).

## 4. Target vs current vs projected — the headline table (MEASURED)

Per-timestep maneuver distribution (v1 kinematic labeler, 2 s horizon):

| Class | Current parity (2,376 / 13.13 h) | **v2 TARGET** | **v2 ACHIEVED (9,000 / 50.25 h)** |
|---|---|---|---|
| lane_keep | 59.64 % | 45 % | **45.0 %** |
| turn_left | 6.86 % | 14 % | **14.0 %** |
| turn_right | 7.39 % | 14 % | **14.0 %** |
| accelerate | 13.23 % | 13 % | **13.0 %** |
| brake_stop | 12.88 % | 14 % | **14.0 %** |
| **turns (L+R)** | **14.25 %** | **28 %** | **28.0 %** |

Speed regime (kept balanced on purpose — this axis was already good):

| Regime | Current parity | v2 target | **v2 achieved** |
|---|---|---|---|
| stopped (<1 m/s) | 7.76 % | 10 % | **9.76 %** |
| city (1–12 m/s) | 45.92 % | 52 % | **52.24 %** |
| highway (>12 m/s) | 46.32 % | 38 % | **38.0 %** |

Per-clip scenario *presence* (the "turn/stop-poor at clip level" gap the profile
flagged) improves sharply:

| Event ≥1× in clip | Current parity | **v2 achieved** |
|---|---|---|
| any turn | 42.6 % | **76.4 %** |
| junction-scale turn | 37.7 % | **61.3 %** |
| any full stop (v<0.5) | ~thin | **27.3 %** |
| any brake_stop | 57.5 % | **73.4 %** |
| net heading >45° | 25.0 % | **40.4 %** |
| net heading >90° (sharp/roundabout geom.) | 10.4 % | **13.8 %** |

**Design choice — why 28 % turns and not the 36 % ceiling:** 28 % doubles the turn
mass and halves lane-keep dominance while keeping highway substantial (38 %). Pushing
to 30 % is available (highway would fall to ~36 %); the max is 36 % (highway ~25 %,
lane-keep 35 %) — an over-correction that would starve highway. Change one line
(`TMAN`) in `select_v2_corpus.py` to re-target; the run is 2 s.

## 5. Selection method — greedy water-filling (deterministic, reproducible)

Maintain running class totals; at each of K steps add the remaining clip that
minimizes `1.0·L1(maneuver, TMAN) + 0.5·L1(speed, TSPD)`. Water-filling keeps the
running distribution *on target throughout*, so at K=9,000 the aggregate equals the
target (feasible here). **No RNG — pure argmin**; re-running reproduces the identical
key `physicalai-v2bal-4b7eeeac222d` (verified). No country cap needed (pool already
~5 %/country; achieved max share **5.5 %**, United States). Runtime **2 s**.

## 6. Parity re-selection — the whole corpus is re-balanced, not a blob on a base

The candidate pool is drawn from the **same 197 chunks** the parity set came from
(2,964 / 3,000 phase0 clips are in the scored pool; the rest are parked). Every clip
competes equally under the objective. Result: of the original 3,000 phase0 clips,
**1,311 (43.7 %) are re-selected** into v2 and the other **7,689 (85.4 %) of v2 are
new** clips. This satisfies Sayed's directive — the full 50 h is balanced, **not** a
balanced new blob stacked on the skewed 2,376.

## 7. Phase-2 budget — APPROVE BEFORE THE BIG DOWNLOAD (MEASURED unit costs)

Unit costs measured on the 500 local front_wide mp4s (**12.71 MB/clip**) and the
local epcache (`ep_*.pt` = **111.78 MB/episode**, uint8 `[199,9,256,256]` + poses/
actions/maneuvers).

| Item | Value | Note |
|---|---|---|
| Camera chunks to fetch | **200** | mean 45 clips/chunk selected; 3 chunks (3116/3120/3121) carry the 5 clips whose egomotion was bundled in a neighbour zip — see remap note below |
| Camera **download** (transient) | **~242 GB** | full chunk zips; `fetch-camera` extracts selected clips then **deletes each zip** (peak +1 zip ~1.3 GB) |
| Camera **kept** (9,000 mp4s) | **~112 GB** | extracted front_wide only |
| **Feature cache @ 256 px** | **~982 GB** | **2.1× the ~466 GB MooseFS pod quota — THE GATE** |
| — parity 2,376-clip ref | ~259 GB | for scale |
| Build decode compute | **~2.5 h** | multi-worker pod, decode-bound (parity 2,376 ≈ 40 min) |

**The gate is disk, not download or compute.** A 50 h / 9,000-clip epcache at the
parity-matched 256 px is ~982 GB and will not fit one pod's ~466 GB MooseFS quota
(verify with a real `dd` write, never `df` — `df` shows the 965 TB cluster). Options,
in recommended order:

1. **Keep 256 px, provision ~1.1 TB** — resize a RunPod volume or **shard** the cache
   across the two training pods' volumes (~491 GB each). Preserves comparability with
   the 256 px flagship. *Recommended.*
2. **Stream-decode from mp4** (no pre-built uint8 cache) — disk drops to the ~112 GB
   of mp4s, at the cost of per-step decode compute. An architecture change to the
   dataloader.
3. **128 px cache = 246 GB (fits)** — but breaks 256 px parity/comparability; only if
   a smaller-input model is intended. (224 px = 753 GB and 192 px = 553 GB both still
   exceed the quota.)
4. **Fallback smaller corpus** — ~4,200 clips @ 256 px ≈ 459 GB fits one quota, but
   that is ~23 h, not 50 h. Last resort.

`r0_selection_v2.parquet` is drop-in for the existing pipeline: `fetch-camera` and
`discover_r0_clips` read only `clip_id` + `chunk`, both present. **Do NOT run
`physicalai_r0.py select`** against this root — it would overwrite the selection;
Phase 2 should point `fetch-camera` at `r0_selection_v2.parquet` (or copy it to the
`r0/` root as the active selection on a v2-dedicated data root).

**Chunk-remap correctness note (latent bug avoided):** 5 of the 18,988 pool clips
(0.026 %) have their egomotion bundled in a *neighbour* chunk's zip while the
camera lives in the catalog's canonical chunk (e.g. clip `32ad1a3a…` egomotion in
zip 1573, camera in chunk 3117). The selector maps `chunk` from `clip_index.parquet`
(unique clip_id index), so `fetch-camera` hits the right camera zip. Without this,
Phase 2 would download the wrong chunk and silently drop those clips.

## 8. Honest gap — this balances KINEMATICS, not SEMANTICS

Every label here is derived from **ego poses only** (yaw + speed). The v2 corpus
corrects the *kinematic* scenario skew (turns, junctions-by-geometry, stops, speed).
It **cannot** create — and does not claim — semantic coverage:

- Traffic lights / signs, roundabouts *as a class*, pedestrians/cyclists, lane
  merges/splits, right-of-way, weather/lighting remain **0 % labeled**.
- "Junction presence" is the **kinematic** proxy (a tight transient heading change);
  it **misses every intersection the ego crosses straight through on green** — likely
  the majority. So 61.3 % junction presence is a floor on *turning* junctions, not a
  count of intersections.

Raising semantic coverage needs the **VLM/map-label track** (a separate effort). The
v2 selection is the right substrate for it — richer in turns/junctions/stops means
more *events* for a semantic labeler to annotate — but the two must not be conflated.
Evidence class for this section: **MEASURED absence** (label coverage) +
**HYPOTHESIS** (that straight-through intersections dominate).

---

## Deliverable manifest

| Artifact | Location | Notes |
|---|---|---|
| `V2_CORPUS_DESIGN.md` (this file) | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-24-v2-corpus-50h-balanced/` | staged |
| `r0_selection_v2.parquet` | same dir | **9,000 clips** + per-clip scores; drop-in for `fetch-camera`/`discover_r0_clips` |
| `v2_pool_scored.parquet` | same dir | full 18,988-clip scored pool (provenance / re-selection at other K) |
| `v2_selection_meta.json` | same dir | machine-readable: key, target vs achieved, Phase-2 budget, parity overlap |
| `score_v2_pool.py` | same dir | egomotion pool scorer (first-20 s window) — ~3.5 min, dev box |
| `select_v2_corpus.py` | same dir | greedy water-filling selector — deterministic, 2 s |

All artifacts are **staged in the repo working tree** (no commit/push per the agent
standard). Nothing lives only on a pod or in scratch.

**Escalation / integration:** Phase 2 (camera fetch + epcache build) is **gated on
the §7 storage decision** — it needs Sayed/orchestrator approval for ~242 GB of
download and, above all, **~982 GB of feature-cache disk** (or one of the §7
alternatives). That is the single decision blocking the build.

**Evidence classes:** feasibility, pool structure, target-vs-achieved, per-clip
presence, and unit costs (mp4 MB/clip, ep MB/clip) are all **MEASURED** (artifacts
above). Camera *download* total (~242 GB) and *build* time (~2.5 h) are **ESTIMATED**
from the measured per-clip mp4 size and the parity build time. The semantic-gap
dominance claim is **HYPOTHESIS**.
