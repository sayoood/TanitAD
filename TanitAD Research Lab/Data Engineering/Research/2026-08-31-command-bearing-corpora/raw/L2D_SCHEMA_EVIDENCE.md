<title>raw/L2D_SCHEMA_EVIDENCE — the descriptor fields, with retrieval paths</title>

# raw/L2D_SCHEMA_EVIDENCE.md

`Evidence for every L2D fact quoted in RESULT.md. Two tiers, kept apart on purpose.`

---

## TIER A — `MEASURED`, from the dataset's OWN descriptor, in this repo

**Path:** `TanitAD Research Lab/Data Engineering/l2d-adapter-2026-07-22/l2d_info.json`
**Provenance of that file:** `meta/info.json` of `yaak-ai/L2D`, pulled 2026-07-22 by this
programme's Data Engineering agent; it is the LeRobot v3.0 dataset descriptor, not a summary.
**Re-read by me on 2026-08-31** (`json.load`, fields printed verbatim below).

```
observation.state.vehicle   float32 [8]
  axes: speed, heading, heading_error, hp_loc_latitude, hp_loc_longitude,
        hp_loc_altitude, acceleration_x, acceleration_y
observation.state.waypoints float32 [10, 2]     names: ["way", "points"]
action.continuous           float32 [3]
  axes: gas_pedal_normalized, brake_pedal_normalized, steering_angle_normalized
action.discrete             int32   [2]
  axes: gear, turn_signal
```

Descriptor top-level keys present: `codebase_version, robot_type, total_episodes,
total_frames, total_tasks, chunks_size, data_files_size_in_mb, video_files_size_in_mb, fps,
splits, data_path, video_path, features`.

⭐ **The distinction that matters and is visible in the descriptor itself:** the realised
state (`observation.state.*` — speed, heading, GPS, IMU, waypoints) and the actuation
(`action.*` — pedals, steering, gear, turn signal) are **separate feature groups**, not two
views of one array.

### Also MEASURED by this programme on 2026-07-22 (same note, `2026-07-22-l2d-adapter-schema-and-slice.md`)

| fact | value | how it was measured |
|---|---|---|
| scale of the released tranche | **100,000 episodes · 26,466,954 frames @ 10 fps = 735 h** | `meta/info.json` |
| robot | `KIA Niro EV 2023` | `meta/info.json` |
| distinct drives | **1,103** | `groupby(session_id)` over `meta/episodes/…parquet` |
| overlap | a known-overlapping consecutive pair shares **150 frames at 0.000000 m GPS disagreement** | corpus-wide |
| de-duplicated size | **100,000 → 46,473 episodes (53.5 % dropped)** | `session_id` + non-overlapping unix-time tiling |
| policy split | **EXPERT 86.2 % / STUDENT 13.8 %** | `task.policy` |
| ⛔ intrinsics | **NONE ship** — only `extrinsic_RDF.yaml` | real file tree |
| unit trap | `vehicle[0]` "speed" is **km/h, not m/s** (reads 65–85 on a road posted 70) | cross-check against `max_speed` |
| licence at that date | Apache-2.0, `gated=False` | HF README front-matter + card body + HF tag |

## TIER B — `PUBLISHED`, from the official card, retrieved 2026-08-31

**Source:** `huggingface.co/datasets/yaak-ai/L2D` and its `raw/main/README.md`.

> `"gas_pedal_normalized, brake_pedal_normalized, steering_angle_normalized"` — action continuous
> `"gear, turn_signal"` — action discrete
> `"Speed/Heading/GPS/IMU"` — observation state
> `"Lane count, Road type (highway|residential), Road surface (asphalt, cobbled, sett), Max speed limit"`
> `"Precipitation, Conditions (Snow, Clear, Rain), Lighting (Dawn, Day, Dusk)"`
> `"Future waypoints snapped to OpenStreetMap graph, aditionally rendered in birds-eye-view"` *(sic)*
> `"Expert (driving instructors) and student (learner drivers) policies"`
> `"5000+ hours of driving"`, 30 German cities, `90+ TeraBytes`
> `"apache-2.0"`

Cameras named on the card: `left_forward, front_left, right_forward, left_backward, rear,
right_backward` at 1080×1920, plus a map view at 360×640.

⚠️ **TWO THINGS THE CARD DOES NOT SAY, checked for and absent:**
1. **Where the action signals come from.** No mention of CAN, of a vehicle bus, or of the
   measurement point. ⇒ command provenance is **UNVERIFIED at the primary**.
2. **Field of view or intrinsics.** Absent from the card, and absent from the file tree
   (Tier A). ⇒ the camera geometry is **UNKNOWN**, not merely unstated.

⚠️ **A number that must not be conflated:** the card's *"5000+ hours"* describes the
collection programme; the **released tranche measured in Tier A is 735 h**. Quoting 5,000 h
as available data would be wrong.

## TIER C — the gate's own text, for the claim being checked

**Source:** `Project Steering/V7_LAUNCH_GATE.md`, read 2026-08-31.

> "⛔ **TWO BLOCKERS ON THE ONLY CORPUS THAT COULD TEST IT:** 1. **GEOMETRY.** comma2k19's
> entire field is **65.203°**; our frame is **120°** … 2. **DATA.** comma2k19 is **not on
> Thor** — only `extract_comma2k19.py` and its tests."

> "once the data is present, correlate **CAN steering against the curvature-derived proxy on
> the same clips**. … ⚠️ Data-only: no training, no GPU, and it decides whether to spend either."

## TIER D — the absence probes, and ⛔ WHAT MY FIRST PROBE GOT WRONG

⛔ **My first probe produced a false absence and I nearly wrote it up.** Pattern `L2D|yaak`
over three files returned one hit, which I was about to describe as "no L2D reference".
A second, exact-case probe over the whole directory shows that reading was wrong, and the
correction changes the claim this package can make.

**Probe 1 (too narrow).** `Select-String -Pattern 'L2D|yaak'` over `GOALS_AND_CLAIMS.md`,
`V7_LAUNCH_GATE.md`, `BACKLOG.md` → 1 hit, at `GOALS_AND_CLAIMS.md:1489`. **That hit is
`adaptive_avg_pool2d`** — `pool2d` contains the substring `l2d`, case-insensitively. A real
false positive inside an apparent absence.

**Probe 2 (case-sensitive, whole directory).** `Select-String -Pattern 'L2D' -CaseSensitive`
over `Project Steering\*.md` → **9 hits in 3 files**:

| file:line | what it says |
|---|---|
| `ROADMAP.md:77` | X2 lists L2D inside a semantic/strategic-label dataset survey |
| `LOOP_STATE.md:994` | S4 — *"**L2D adapter (~2-3 eng-d) = the real unblock**"* for TanitDataSet C+R |
| `LOOP_STATE.md:998` | S7 — ✅ **DONE**: *"schema verified (LeRobot v3.0, 100k eps/26.5M frames/735h), drive-dedup PROVEN … Apache-2.0 re-verified, 1-drive slice built end-to-end"* |
| `LOOP_STATE.md:1024` | *"L2D ships a metric distance-to-maneuver natively"* |
| ⛔ `LOOP_STATE.md:1025` | 🔴 **GDPR gate — see Tier E. This is the item a narrow probe would have hidden.** |
| `REPO_TRIAGE_2026-07-20.md:86` | L2D semantic-label survey, *"Apache-2.0, 4 219 nav cmds"* |

**Probe 3.** `yaak` over `Project Steering\*.md` → **zero hits** (the corpus is referred to
only by its short name).

⇒ **CORRECTED FINDING.** L2D is **not unknown** to the steering layer — it is surveyed,
adapted, sliced and roadmapped. What is absent is the **JOIN**: no document connects L2D's
`action.*` group to P2(b)'s command question, and the P2(b) scoping written the same day
names only comma2k19 and PhysicalAI. ⭐ **That is a weaker and truer claim than "I found a new
corpus", and it is still decision-changing** — the gate's *"the ONLY corpus that could test
it"* is incorrect as written.

## TIER E — ⛔ THE BLOCKER MY NARROW PROBE WOULD HAVE MISSED

**`Project Steering/LOOP_STATE.md:1025`, verbatim:**

> 🔴 **GDPR gate on L2D** — real German dashcam footage, NO anonymization statement in the
> card. Apache-2.0 grants copyright, not the data-protection right → **face/plate check
> REQUIRED before any L2D frame is re-hosted / published**. Blocks L2D entering a shippable
> corpus.

⚠️ **Scope it exactly, in both directions.**
- **It binds** any L2D **frame** that would be re-hosted or published, and therefore binds a
  vision arm and any dataset release. That is a real blocker and it is unresolved.
- **It does not bind** a read-only correlation over the **state parquet** — no frames, no
  re-hosting, no publication. The triage test in RESULT §6 is deliberately on that side of
  the line, and that is not a convenient reading: it is why the state-only design was chosen.
- ⛔ **It is not mine to resolve, and this package does not treat it as resolved.**
