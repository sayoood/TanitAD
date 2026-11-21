# The NuRec `map.xodr` — TanitAD has a real lane graph, a route signal, and WGS84 coordinates

**Stream B, 2026-08-02.** Scene `00040136-e651-4abd-991d-0655ccda9430`, NuRec sample set 26.04.
All numbers below are **MEASURED** by `xodr_probe.py` (this directory) unless labelled otherwise.
Reproduce with:

```bash
python3 xodr_probe.py --usdz <scene>.usdz --out <workdir> --clipgt
```

---

## Headline

The programme's standing conclusion — *"no map, lane graph, junction annotation, traffic-light
feature or route/goal signal; the strategic brain's topology must come from AlpaSim or an external
corpus"* — was correct **about PhysicalAI-AV** and is **not** true of the NuRec scenes.

This single 1.96 GB usdz contains a **georeferenced OpenDRIVE HD map** (219 roads, 356 driving
lanes, 26 junctions) **plus** a second independent annotation source (`clipgt/`, 19 parquet tables)
carrying per-lane manoeuvre direction, intersection typology and stop-line semantics.

Three claims that decide work, each MEASURED:

1. **A lane graph is derivable.** 356 driving-lane nodes, 340 directed edges, built from
   `<road><link>` + `<lane><link>` + `<junction><connection><laneLink>`.
2. **A route/goal signal is derivable.** The ego trajectory snaps to lane centrelines at
   **median 0.390 m, max 0.916 m, 100 % under 2 m** across the 202-pose scene window, yielding an
   ordered route of 14 roads / 15 lane hops.
3. **The scene is georeferenced to WGS84.** `+proj=tmerc +lat_0=59.3487393155161
   +lon_0=17.957061097803336 +ellps=WGS84`. The scene is in **Stockholm, Sweden**. OSM map-matching
   is possible. **This differs from PhysicalAI-AV, whose `egomotion` is clip-local metres with no
   lat/lon.**

⚠️ **Two traps found, both of which would silently corrupt a LONGITUDINAL metric — see §6.**

---

## 1. Summary table of counts

### 1.1 OpenDRIVE (`map.xodr`, 530 746 B, md5 `17dd87c56f4d701cebc5e5493e563c06`)

| quantity | count | note |
|---|---:|---|
| `<road>` | **219** | total length **4 175.53 m**; min 0.25 m, mean 19.07 m, max 46.67 m |
| roads inside a junction | 86 | `junction != "-1"` |
| `<junction>` | **26** | all untyped (no `type` attribute) |
| `<junction><connection>` | 96 | |
| `<junction>…<laneLink>` | 120 | |
| `<laneSection>` | 219 | **exactly one per road** — 0 roads have multiple |
| `<lane>` total | **578** | |
| — `driving` | **356** | 329 right + 27 left |
| — `none` (centre) | 219 | the centre lane of each road |
| — `restricted` | 3 | |
| — sidewalk / shoulder / parking / biking / border / curb / median | **0** | **driving-only lane model** |
| `<lane><link>` entries | 516 | |
| `<signal>` | **5** | all `name="stopline"`, all `dynamic="no"` |
| `<controller>` | **0** | |
| `<junctionGroup>` | **0** | |
| `<signalReference>` | **0** | |
| `<object>` | **0** | 219 empty `<objects>` containers |
| roads with `<elevationProfile>` | **219 / 219** | full 3D |
| roads with `<lateralProfile>` | **219 / 219** | superelevation/shape |
| road `<type>` = `town` | 170 | |
| road `<type>` = `motorway` | 49 | |

### 1.2 Geometry primitives — **only two are used**

| primitive | count |
|---|---:|
| `arc` | **275** |
| `line` | **18** |
| `spiral` (clothoid) | **0** |
| `poly3` | **0** |
| `paramPoly3` | **0** |

293 `<geometry>` records total. A consumer needs to implement **arc and line only**. (`xodr_probe.py`
implements all five regardless, so it will not silently mis-sample another NuRec scene.)

### 1.3 Speed limits — declared

| level | values | counts |
|---|---|---|
| road `<type><speed>` | 40 / 50 / 70, `unit="mph"` | 54 / 116 / 49 |
| lane `<speed>` | 40 / 50 / 70, `unit="mph"` | 106 / 158 / 92 |

⚠️ **The `unit="mph"` attribute is wrong — see §6.1.**

### 1.4 Lane graph (derived)

| quantity | value |
|---|---:|
| nodes (driving lanes) | **356** |
| directed edges | **340** |
| sources (in-degree 0) | 51 |
| sinks (out-degree 0) | 44 |
| isolated (degree 0) | 20 |
| max out-degree | 3 |
| weakly connected components | **28** |
| largest component | **229 nodes (64.3 %)** |
| second / third | 71 / 12 |
| unresolved links | 2 (`junction_predecessor_unmapped`) |

### 1.5 `clipgt/` — the second, independent annotation source (19 parquet tables)

| table | rows | empty placeholder? |
|---|---:|---|
| `obstacle` | 3 287 | no |
| `association` | 2 728 | no |
| `road_boundary` | 246 | no |
| **`lane`** | **235** | no |
| `egomotion_estimate` | 202 | no |
| `lane_line` | 79 | no |
| **`wait_line`** | **49** | no |
| `road_island` | 15 | no |
| **`intersection_area`** | **4** | no |
| `clip`, `calibration_estimate`, `mads_trace` | 1 | no (genuinely single-row) |
| **`traffic_light`** | 1 | **YES — empty** |
| **`traffic_sign`** | 1 | **YES — empty** |
| `crosswalk`, `pole`, `road_marking`, `gore_area`, `buffer_zone` | 1 | **YES — empty** |

*Empty-placeholder detection is not a row count:* the marker is 1 row whose every field is null
**and whose Arrow type is inferred as `null`**. `clip`/`calibration_estimate`/`mads_trace` also have
1 row but carry real typed data, and are correctly not flagged.

---

## 2. Can a LANE GRAPH be derived? — **Yes**, with one caveat

**Present.** Every ingredient for a directed lane graph:

- `<road><link><predecessor|successor elementType="road|junction" elementId=…>` — 150 road-to-road
  successors, 39 road-to-junction, 30 none (map-boundary dangles); predecessors 150 / 40 / 29.
- `<lane><link><predecessor|successor id=…>` — 516 entries.
- `<junction><connection incomingRoad connectingRoad contactPoint><laneLink from to>` — 96
  connections, 120 lane links.
- Full lane **width** polynomials, so lane **centrelines** are recoverable (`lane_centerlines.json`,
  356 driving-lane polylines). Their correctness is proven independently by the 0.390 m ego snap.

**Absent.** No lateral (lane-change) edges exist in OpenDRIVE by construction — a lane's left/right
neighbour must be inferred from lane-id adjacency within a `laneSection`. My graph is
**longitudinal-only**; that is an instrument limitation, not a map gap.

**Caveat — 28 weakly connected components.** The largest holds 229 of 356 lanes (64 %). Fragmentation
is expected for a clipped map tile (51 sources / 44 sinks are boundary dangles), but 20 fully
isolated lanes is more than boundary effects alone explain. **Not diagnosed. Do not assume the graph
is globally routable.** It *is* connected along the driven corridor (§3).

---

## 3. Can a ROUTE / GOAL signal be derived? — **Yes**

Method (no frame assumption; see §5): take the 299 WGS84 poses in `pose_record.json`, project them
through the map's own `+proj=tmerc` into map-local metres, snap each to the nearest driving-lane
centreline.

| metric | scene window (202 poses, 20.0 s) | full record (299 poses, 29.8 s) |
|---|---:|---:|
| snap distance median | **0.390 m** | 0.430 m |
| snap p90 | 0.652 m | 0.779 m |
| snap max | **0.916 m** | 3.374 m |
| fraction < 2 m | **100 %** | 99.0 % |
| distinct roads traversed | **14** | 26 |
| distinct lanes traversed | — | 27 |
| junction-internal roads traversed | — | 9 |

**Sub-metre lane-level agreement over the entire scene.** The derived route for the scene window:

```
101:-1 → 205:-1 → 100:-1 → 99:-1 → 98:-1 → 97:-1 → 96:-1 → 95:-1 → 94:-1
      → 93:-2 → 67:-2 → 66:-1 → 65:-1 → (71:-1) → 65:-1
```

This is a genuine **strategic route/goal signal**: an ordered lane sequence with a downstream goal
lane, exactly the input the programme concluded had to come from AlpaSim or an external corpus.

**Honest limit — 11 of 14 scene-window hops are edges in the derived graph (16/28 over the full
record).** The three misses are *instrument* artefacts, not map defects:

- `23:-1 → 23:-2` is a **within-road lane change**, which by construction cannot be a longitudinal
  edge (no lateral edges in my graph).
- `65 → 71 → 65` and the `64 → 166 → 169 → 107 → 89 → 64` excursion are **nearest-neighbour snap
  ambiguity** among near-coincident short junction roads (mean road length is only 19 m and
  junction-internal roads overlap). Each such road captures 1–2 samples.

> **Hypothesis formed and REFUTED.** I first read the excursion as the ego passing *under* an
> overpass (roads 107 and 153 are `type="motorway"`), with a 2D snap grabbing the elevated
> carriageway. **Measurement refutes it:** at those samples the ego's altitude minus the snapped
> road's elevation is −1.06 to +0.16 m — same grade, not an overpass. The correct fix is
> heading-gated / HMM map-matching instead of pure nearest-neighbour, not a 3D snap.

**Fix is cheap and known:** standard HMM map-matching with a heading term collapses these.

---

## 4. Traffic lights, signals and stop lines — **no traffic lights, at three probes**

Per the *"absence at one location is not absence"* rule, probed at three independent locations:

| probe | result |
|---|---|
| 1. `map.xodr` `<signal>` elements | 5, **all** `name="stopline"`, **all** `dynamic="no"` |
| 2. `map.xodr` `<controller>` / `<junctionGroup>` / `<signalReference>` | **0 / 0 / 0** |
| 3. `clipgt/traffic_light.parquet` | 1 row, **every field null, Arrow type `null`** → empty |

⇒ **This scene contains no traffic lights.** Corroborated by a fourth signal: one of the four
`intersection_area` rows is categorised `FOUR_WAY_UNCONTROLLED_ASYMMETRICAL`.

`traffic_sign.parquet` is likewise empty, so **there is no sign/light state to condition on in this
scene.** This is a property of *this scene*, not of NuRec — other scenes must be re-probed.

**What does exist** — richer than the xodr alone:

- `wait_line` (49): `category` STOP 6 / UNKNOWN 43; `intersection_subtype` **ENTRY 27 / EXIT 22**;
  `is_implicit` True 43 / False 6. The 6 explicit stop lines correspond to the 5 xodr `stopline`
  signals.
- `intersection_area` (4): `FOUR_WAY`, `T_JUNCTION`, `T_JUNCTION_ASYMMETRICAL`,
  `FOUR_WAY_UNCONTROLLED_ASYMMETRICAL`. All `is_complete=False` (clipped by the scene).

### 4.1 `clipgt/lane` carries a per-lane MANOEUVRE label

Directly relevant to the programme's largest known defect (the 5-way softmax that mixes lat+lon):

| `lane_direction` | count |
|---|---:|
| STRAIGHT | 173 |
| STRAIGHT_TURN | 18 |
| RIGHT_TURN | 14 |
| LEFT_TURN | 13 |
| BRANCH_STRAIGHT | 6 |
| BRANCH_LEFT | 6 |
| U_TURN | 2 |
| BRANCH_RIGHT | 3 |

Plus per-lane `left/right_rail` 3D polylines, `left/right_edge_styles` (VIRTUAL 2103,
LONG_DASHED_SINGLE 857, SOLID_SINGLE 658, TALL_CURB 650, BARRIER 194, ROAD_BOUNDARY 107,
SHORT_DASHED_SINGLE 94, SOLID_GROUP 70, LONG_DASHED_GROUP 15), `edge_colors` (WHITE 898,
UNKNOWN 1476), `map_end` (NONE 228 / BACK 4 / FRONT 3), `vehicle_types` (CAR 235/235).

**These are `autolabels:v0`** (`label_class_id: "minimap:lanes:autolabels:v0"`), not human GT.

---

## 5. Georeferencing — **yes, WGS84, and this differs from PhysicalAI-AV**

`<header>` `revMajor=1 revMinor=4`, `vendor="DeepMap, Inc."`, `name`/`date`/`version` all empty.
No `north`/`south`/`east`/`west` bbox attributes.

```
+proj=tmerc +lon_0=17.957061097803336 +lat_0=59.3487393155161 +=alt_0=0
            +ellps=WGS84 +units=m +geoidgrids=egm96_15.gtx
```

| item | value |
|---|---|
| projection | transverse Mercator, k₀ = 1, WGS84 ellipsoid, metres |
| origin | 59.3487393155161 °N, 17.957061097803336 °E — **Stockholm, Sweden** |
| vertical datum | EGM96 geoid (`egm96_15.gtx`) |
| map bbox (local) | x ∈ [−127.30, +127.24] m, y ∈ [−174.40, +632.56] m |
| map bbox (WGS84) | SW 59.34717 / 17.95482 · NE 59.35442 / 17.95930 |
| TM forward↔inverse round-trip error | **0.60 mm** (max over 5 corners) |

**Three independent confirmations that the map frame and the trajectory frame are the same:**

1. **Exact origin identity.** The xodr's `lat_0`/`lon_0` are **bit-identical to all 16 digits** with
   `pose_record.json → record[0].alignment_world_pose.lat_lng_alt`
   (59.3487393155161 / 17.957061097803336). Not a coincidence — the map origin *is* the scene's
   first alignment pose.
2. **Horizontal.** Projecting the 202 scene poses through the map's own proj4 lands them on lane
   centrelines at median 0.390 m (§3).
3. **Vertical.** Ego altitude minus snapped-road `<elevation>`: mean **−0.318 m**, median −0.275 m,
   σ 0.230 m, range [−1.132, +0.170] over all 299 poses. Sub-metre in Z as well. (The systematic
   −0.3 m is consistent with a rig-height / geoid offset; not diagnosed.)

**⇒ OSM map-matching is possible for NuRec scenes.** The programme's finding that
`egomotion` in PhysicalAI-AV carries no lat/lon and cannot be OSM-matched **stands and is
unaffected** — these are different corpora. NuRec adds the capability PhysicalAI-AV lacks.

**⚠️ Parser trap:** the proj4 string contains the malformed token **`+=alt_0=0`** (vendor typo for
`+alt_0=0`). A strict proj4/pyproj parse can reject the whole string. `xodr_probe.py` records it in
`geoReference_malformed_tokens` rather than hiding it.

### 5.1 The road `name` field is a DeepMap id — **not** OSM

219 roads, 219 distinct names, 0 empty: **185 positive integers** in [15 356 506, 68 465 560] and
**34 negative integers** in [−184, −8]. No street-name strings anywhere in the file.

- **They join to `clipgt`:** 138 of the 219 xodr road names appear as `clipgt/lane` `key.map_id`.
  The two sources **share a DeepMap feature-id space** — so clipgt lane attributes (manoeuvre
  direction, edge styles, speed) can be joined onto xodr roads. Used in §6.1.
- **They are NOT OSM way ids** (checked): OSM way `15356506` is *"West Avenue A"*, `highway=residential`,
  `tiger:county=Milam, TX` — **Texas, USA**, not Stockholm. Corroborated by the field being named
  `map_id` with a `map_id_version` of `17476`.

⇒ **External linkage must go through geometry + WGS84 coordinates, not through ids.** That path is
open (§5).

### 5.2 Which pose source to use — `pose_record.json`, **not** `T_rig_worlds`

`rig_trajectories.json → T_rig_worlds` is **not** a planar rotation of the geodetic track. Best-fit
rotation about the origin gives **RMS residual 30.16 m** (max 44.51 m) over the 202 compared poses;
the inverted reading is worse (45.07 m). For reference `record[0].axis_angle.angle = 103.019°` and the
best-fit is 107.600°.

Contributing, MEASURED: the two series are on **different time grids** — `T_rig_world_timestamps_us`
is 202 samples spanning exactly 20.0 s (99.502 ms steps, matching `data_info`), while `pose_record`
is 202 samples over 20.100011 s (exactly 100.000 ms steps). That alone cannot explain 30 m.

**Root cause NOT established. Use `pose_record.json` for anything map-related.** Note also that
`clipgt` geometry is in the *egomotion* frame (`gtaas:egomotion_deepmap:v0`, ego-start origin,
x-forward), **not** the xodr tmerc frame — `intersection_area` sits at x≈274, y≈−41, outside the
xodr's x range entirely. **Relating clipgt geometry to the xodr frame is unsolved here** and is the
main open item.

Separately, `T_world_base` is an **ECEF** matrix, but it anchors a *different* origin: converting its
translation back to geodetic gives 59.3496153 / 17.9567238, i.e. **97.5 m north and 19.1 m west** of
the alignment origin. Do not use it as the scene origin.

---

## 6. ⚠️ Two traps that would corrupt a LONGITUDINAL metric

Per the binding four-metric-families rule, LONGITUDINAL requires target-speed accuracy. Both traps
below sit directly on that input.

### 6.1 `unit="mph"` in the xodr is WRONG — the values are km/h

Settled by joining the two sources on the shared DeepMap feature id (§5.1), **138 roads, zero
exceptions**:

| xodr `<speed max unit>` | `clipgt/lane.speed_limit` | × 1.609344 | roads |
|---|---|---:|---:|
| `40 mph` | `24.854801` | **39.9999 km/h** | 32 |
| `50 mph` | `31.0685` | **49.9999 km/h** | 71 |
| `70 mph` | `43.495903` | **69.9999 km/h** | 35 |

`clipgt` stores the limit genuinely in mph; multiplying by 1.609344 reproduces the xodr's number
**exactly** on every joined road. ⇒ the xodr exporter wrote the **km/h value** and labelled it `mph`.

Corroborating: every `<signal>` carries `country="USA"` while the geoReference places the scene in
Sweden — the same wrong template. And 40/50/70 km/h are exactly the standard Swedish limits.

**Consuming the xodr number as mph overstates every speed limit by 1.609×.**

### 6.2 The limit VALUE is also inconsistent with the observed driving — do not use it as a target

MEASURED ego speed from the geodetic track: **min 43.4, mean 59.7, max 74.8 km/h**; the first ~15 s
sit steadily at **70–73 km/h**. Yet **293 of 299 samples snap to roads both sources label 40 km/h**.

Under **neither** unit reading is this consistent:

| reading | limit | ego mean 59.7 km/h | ego first-15 s ≈ 71 km/h |
|---|---|---|---|
| km/h (correct per §6.1) | 40 km/h | **+49 %** | **+78 %** |
| mph (as declared) | 64.4 km/h | −7 % | +10 % |

**This is an implausible magnitude and I am reporting it as a finding, not caveating it away.**
Both sources are `autolabels:v0`, so the most likely reading is an autolabel error on these roads —
but that is a **HYPOTHESIS, not established.**

⇒ **The speed-limit field in this scene must not be used as a target-speed label for the
LONGITUDINAL family without external verification.** The ego's own speed profile is trustworthy
(it derives from the geodetic track that snaps to lanes at 0.39 m); the *posted limit* is not.

---

## 7. Coverage cross-check — the map covers the whole scene, with margin

| quantity | value |
|---|---:|
| map bbox | **254.54 m (E–W) × 806.96 m (N–S)** = 0.2054 km² |
| total road length in map | **4 175.53 m** |
| ego path, 20.0 s scene window | **341.30 m** |
| ego path, full 29.8 s record | 493.80 m |
| map road length ÷ ego scene path | **12.23 ×** |
| ego bbox | 99.12 m × 463.04 m |
| **ego fully inside map bbox** | **TRUE** |
| margin beyond ego: x−/x+ | **80.32 m / 75.09 m** |
| margin beyond ego: y−/y+ | **174.40 m / 169.52 m** |

**Verdict: not a fragment.** The map covers the full 20 s scene *and* the longer 29.8 s pose record,
with 75–174 m of margin on every side and 12× more road length than the ego drives.

**But it is a narrow corridor, not a neighbourhood.** Lateral margin is only ~75–80 m, so roads and
junctions visible in the camera beyond ~80 m to the side are **not** in the map. For a strategic
planner needing alternatives at a junction, that is the binding limit.

**Incidental finding:** `pose_record.json` holds **299 poses over 29.800 s**, while `data_info.json`
declares the rendered scene as **202 poses over 20.0 s**. The pose record extends **~49 % beyond the
rendered scene.** Anyone iterating `record[]` without truncating to 202 will run 9.8 s past the end
of the renderable scene.

---

## 8. What this means for TanitAD

1. **The strategic brain has a topology source that is not AlpaSim.** Lane graph + ordered route +
   junction typology + per-lane manoeuvre labels, all from an artifact already on Thor.
2. **The tactical manoeuvre label problem has an external reference.** `lane_direction` gives 8
   classes that separate turn from branch and keep U_TURN distinct — directly usable to audit the
   5-way softmax that mixes lateral and longitudinal.
3. **Route/goal conditioning becomes testable.** The programme noted REF-C evaluates with
   `nav_cmd=None`, confounding the "strategic choice is a ~2 % lever" refusal. A real route signal
   now exists to condition on.
4. **NuRec scenes are OSM-matchable**; PhysicalAI-AV is not. If external map priors are wanted, the
   NuRec corpus is the entry point.
5. **Scale is the open question.** All of the above is **one 20 s scene**. Whether the sample set's
   other scenes carry the same `map.xodr` + populated `clipgt` is **not measured here** and is the
   first thing to check before anything is designed on it.

### Next actions, cheapest first

1. **Probe every scene in the NuRec sample set** with `xodr_probe.py --clipgt` — do all carry a map?
   Are `traffic_light`/`traffic_sign` empty everywhere? (0 GPU, minutes.)
2. **Solve the clipgt ↔ xodr frame relation** (§5.2). Without it the rich clipgt attributes cannot be
   fused with the xodr topology. Likely a rigid transform recoverable from `egomotion_estimate`
   (202 rows) against the geodetic track.
3. **Replace nearest-neighbour snapping with heading-gated HMM map-matching** to close the 3 of 14
   route hops (§3).
4. **Do NOT wire the xodr speed limit into any LONGITUDINAL target-speed metric** until §6.2 is
   resolved externally.

---

## 9. Evidence classes

| claim | class |
|---|---|
| all counts in §1, §3, §4, §5, §7 | **MEASURED** — `xodr_probe.py` → `xodr_probe_output.json` |
| xodr `unit="mph"` is really km/h (§6.1) | **MEASURED** — 138-road join, 0 exceptions |
| ego speed 43.4/59.7/74.8 km/h (§6.2) | **MEASURED** — geodetic track, `ego_track_map_frame.json` |
| road ids are not OSM way ids (§5.1) | **MEASURED** — OSM API way 15356506 = Milam County, TX |
| "no traffic lights in this scene" (§4) | **MEASURED** — three independent probes |
| overpass explanation for snap excursions | **REFUTED** — elevation Δ −1.06…+0.16 m (§3) |
| the 40 km/h limit is an autolabel error (§6.2) | **HYPOTHESIS** — not established |
| the −0.3 m vertical offset is rig height / geoid | **HYPOTHESIS** — not diagnosed |
| `T_rig_worlds` 30 m residual root cause (§5.2) | **UNRESOLVED** — stated as open |
| other NuRec scenes carry the same map | **UNTESTED** — explicitly not claimed |

---

## 10. Deliverable manifest

Repo — `TanitAD Research Hub/Architecture & Inference/Research/2026-08-02-nurec-xodr-map/`:

| file | bytes | what |
|---|---:|---|
| `XODR_MAP.md` | — | this report |
| `xodr_probe.py` | 51 804 | the probe (stdlib + numpy; `--clipgt` needs pyarrow) |
| `xodr_probe_output.json` | 25 166 | every count above, machine-readable |
| `lane_centerlines.json` | 151 282 | 356 driving-lane centreline polylines, map frame |
| `lane_graph_edges.json` | 10 413 | 356 nodes + 340 directed edges |
| `ego_track_map_frame.json` | 68 941 | 299 ego poses in map frame + snapped lane + snap distance |
| `map.xodr` | 530 746 | the extracted OpenDRIVE (md5 `17dd87c56f4d701cebc5e5493e563c06`) |
| `pose_record.json` | 82 494 | 299 WGS84 poses — the georeferencing anchor |
| `data_info.json` | 2 018 | scene window definition (202 poses / 20.0 s) |

⚠️ `map.xodr`, `pose_record.json`, `data_info.json` are **third-party NVIDIA NuRec sample data**
extracted verbatim. Staged for provenance and because stranding is the larger risk; **flagging them
for the orchestrator** in case policy prefers they be referenced rather than committed.

Thor — `tanitad-thor:/home/nvidia/xodr_work/`:

| path | what |
|---|---|
| `xodr_work/*.json`, `map.xodr`, `xodr_probe.py` | same set as above |
| `xodr_work/clipgt/*.parquet` | 19 extracted annotation tables (~700 KB) |
| `xodr_work/rig_trajectories.json` | 2 155 480 B — too large to bank; regenerate by extraction |
| `/tmp/pqvenv` | **throwaway** venv with pyarrow 25.0.0 + numpy, created for the parquet probe |

**Venv note:** per the two-venv rule, pyarrow was installed into a **throwaway `/tmp/pqvenv`**, never
into `tanitad-edge`. The xodr probe itself runs on `~/venvs/tanitad-edge/bin/python` (stdlib + numpy);
only `--clipgt` needs the throwaway.

Source scene (unmodified): `tanitad-thor:/home/nvidia/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430/`
