<title>L2D adapter — verified schema, drive-dedup proof, intrinsics + licence decision (2026-07-22)</title>

# L2D → TanitDataSet adapter: the schema unblock + one-drive slice

**Date:** 2026-07-22 (local, Europe/Berlin). **Author:** Data Engineering agent (L2D adapter task).
**Status:** schema **VERIFIED against the real files** (not the card); drive-level dedup **PROVEN**;
adapter code **staged**; one-drive slice **built**. Full ingest is the follow-on (~2–3 eng-days).

Companion: `DATA_STRATEGY_FOR_HIERARCHY.md` §3 (the prior L2D survey — its numbers were INHERITED
until this note; re-verified below, two corrected), `TANITDATASET_TIER_INTEGRATION_2026-07-21.md`,
the build report `2026-07-22-tanitdataset-C-build-and-push-stage.md`.

Every number is tagged **MEASURED** (ours, this session, artifact path) · **PUBLISHED** (card/HF) ·
**INHERITED** (a prior doc/agent, re-verified or not). Source files pulled to
`C:/Users/Admin/tanitad-data/l2d/` (state + meta only, ~155 MB, zero video for the schema work).

---

## 0. Headline

- **`yaak-ai/L2D` is LeRobot v3.0**, 100,000 episodes, **26,466,954 frames @ 10 fps = 735 h**
  (MEASURED, `meta/info.json`). Robot `KIA Niro EV 2023`, 6 cameras (1080×1920) + a rendered BEV
  `map` channel (360×640). One `train` split of 0:100000 — **we make our own drive-disjoint split.**
- **Drive identity ships natively.** Every episode carries `session_id` (UUID) and `canonical_name`
  (`Niro110-HQ/2023-06-02--19-01-31`). **Drive-level dedup is `groupby(session_id)`** — not the
  timestamp-chaining heuristic the prior survey assumed. **1,103 drives** (MEASURED).
- **The overlap trap is real and now retired with an EXACT test.** A known-overlapping consecutive
  pair shares **150 frames at 0.000000 m GPS disagreement** (median/p95/max) — byte-identical
  re-windowing. De-dup by `session_id` + non-overlapping unix-time tiling collapses the corpus
  **100,000 → 46,473 episodes (53.5 % dropped as duplicate overlap)**. (MEASURED, corpus-wide)
- **One-drive slice BUILT end-to-end with real front-camera pixels** (14 records → 2 SA-segregated
  shards → catalog, sha256-verified, 0 corrupt, tier `ship`). The pipeline is unforked. (MEASURED)
- **Licence RE-VERIFIED: Apache-2.0.** README front-matter `license: apache-2.0`, card body
  `**License:** apache-2.0`, HF tag `license:apache-2.0`, `gated=False`. (MEASURED, HF)
- **Intrinsics: NONE ship** (confirmed on the real file tree — only `extrinsic_RDF.yaml`).
  **Decision: L2D enters strategic/tactical/operative on STATE, and on the camera path only with an
  ESTIMATED, flagged focal — never our asserted `f_eff=266`.** See §5.

---

## 1. Repository layout (MEASURED, HF tree API — NOT `list_repo_files`, which hangs)

⚠️ The HF tree API default-pages at 50 entries; a video chunk dir has ~1000 files, not 50.

```
yaak-ai/L2D  (dataset, apache-2.0, gated=False, 60,024 downloads, lastModified 2026-05-26)
├─ README.md                       # card; license: apache-2.0
├─ extrinsic_RDF.yaml              # EXTRINSICS ONLY (6 cams); NO intrinsics anywhere
├─ .gitattributes
├─ meta/
│  ├─ info.json                    # LeRobot v3 dataset descriptor (the schema, §2)
│  ├─ stats.json                   # global feature stats
│  ├─ tasks.parquet   (135 KB)     # 4,219 distinct task/instruction rows
│  └─ episodes/chunk-000/file-000.parquet  (93 MB)  # 100,000 rows × 140 cols (§3) — the map to everything
├─ data/chunk-000/
│  └─ file-000.parquet … file-019.parquet  (~60 MB each; file-019 = 117 KB tail)  # per-frame STATE
└─ videos/
   ├─ observation.images.front_left/chunk-{000,001,…}/file-{000…}.mp4   (~500 MB each)  # REFERENCE fwd cam
   ├─ observation.images.{left_forward,right_forward,left_backward,right_backward,rear}/…
   └─ observation.images.map/…      # rendered BEV (360×640), intrinsics-free
```

- `data_path  = data/chunk-{chunk:03d}/file-{file:03d}.parquet`
- `video_path = videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4`
- `chunks_size = 1000` files/chunk; target sizes `data 100 MB`, `video 200 MB` (actual video ~500 MB).

## 2. `meta/info.json` — the per-frame feature schema (MEASURED)

```
codebase_version v3.0 · robot KIA Niro EV 2023 · fps 10
total_episodes 100000 · total_frames 26,466,954 · total_tasks 4219 · splits {train: 0:100000}
```

| feature | dtype / shape | axes / content |
|---|---|---|
| `observation.state.vehicle` | f32 [8] | **[speed, heading, heading_error, latitude, longitude, altitude, acc_x, acc_y]** |
| `observation.state.waypoints` | f32 [10,2] | 10 future **(lon, lat)** absolute GPS, OSM-snapped |
| `observation.state.road` | str | OSM highway class (secondary/tertiary/…/motorway/`*_link`/NA) |
| `observation.state.lanes` | str | lane **count** "1".."6"/NA (not ego lane index) |
| `observation.state.max_speed` | str | **posted speed limit**, km/h as "50.0"/"0.0"/"NA" |
| `observation.state.surface` | str | asphalt 83.5 % / sett / concrete / … |
| `observation.state.{precipitation,conditions,lighting}` | str | Clouds 59 %/Clear 31 %/Rain 10 %; this tranche 100 % Day |
| `observation.state.timestamp` | i64 | **unix epoch NANOSECONDS** (absolute clock — the dedup key) |
| `task.policy` | str | **EXPERT 86.2 % / STUDENT 13.8 %** (native off-expert data) |
| `task.instructions` | str | NL strategic instruction, usually with a metric distance |
| `action.continuous` | f32 [3] | [gas_norm, brake_norm, **steering_norm**] — steering is NORMALIZED, not rad |
| `action.discrete` | i32 [2] | [gear, **turn_signal**] |
| `timestamp` | f32 | seconds within episode (0..29.9) |
| `frame_index / episode_index / index / task_index` | i64 | LeRobot indices |
| `observation.images.{6 cams}` | video [3,1080,1920] | `front_left` = reference (identity extrinsics) |
| `observation.images.map` | video [3,360,640] | rendered BEV; **no camera intrinsics involved** |

**Unit trap (MEASURED, must handle):** `vehicle[0]` "speed" reads **65–85 on a road posted 70** →
it is **km/h, not m/s**. Divide by 3.6 for our `poses[:,3]` (v in m/s). `heading` is degrees.
`steering_norm` (≈±0.02) is **not radians** and ships no wheel-angle scale → derive our
`actions[:,0]` steer kinematically, not from this channel.

## 3. `meta/episodes/…parquet` — 140 cols, the map to everything (MEASURED)

The per-episode index. Columns that matter for the adapter:

- `episode_index`, `length`, `tasks` (list<str>), `canonical_name`, **`session_id`**, `visualization`.
- **Data slice:** `data/chunk_index`, `data/file_index`, `dataset_from_index`, `dataset_to_index`
  → episode's contiguous **row range** in the data parquet (ep 0 = rows 0:300).
- **Video slice, per camera key:** `videos/{key}/{chunk_index, file_index, from_timestamp, to_timestamp}`
  → the mp4 file and **[from,to] second range** to seek+decode (ep 0 front_left = file 691,
  881.5→911.5 s). This is exactly how one episode's pixels are recovered from a bundled mp4.
- **Per-episode stats** (`stats/<feat>/{min,max,mean,std,q01..q99}`) incl.
  `stats/observation.state.timestamp/{min,max}` → each episode's absolute unix-ns span **without
  touching the data parquet** (used for drive reconstruction below).

## 4. Drive reconstruction + the dedup PROOF (MEASURED — `l2d/drive_dedup_summary.json`)

`groupby(session_id)` over all 100,000 episode-meta rows:

| quantity | MEASURED | prior INHERITED (`DATA_STRATEGY §3.2`) |
|---|---|---|
| distinct drives | **1,103** (`session_id`) | 9,217 (overlap-chained sub-segments — see note) |
| episodes/drive | p50 **72**, p90 197, max 242 | — |
| drive duration | p50 **1048 s**, p90 3023 s, max **3788 s (63 min)** | p50 99 s, max 2121 s (35 min) |
| unique drive-time | **419 h** | 424 h ✓ |
| episode-time (with overlap) | **735 h** | 735 h ✓ |
| consecutive pairs overlapping | **93.9 %** | 90.8 % |
| median stride between starts | **15.0 s** | 13.76 s |

> **Reconciliation of the drive count.** `session_id` is L2D's native recording-session id; the
> prior 9,217 came from chaining episodes that share frames, which SPLITS a session at any
> recording gap. Both partition the same ~420 h. **`session_id` is the correct, native split unit
> and dedup key** — coarser is safer against train/val leakage; within a session we tile by unix
> time, which absorbs gaps for free. No heuristic chaining is needed.

**The exact overlap test (MEASURED).** Session `186e287f-…`, consecutive windows ep 0 and ep 22:

- time overlap 14.90 s; ep A rows[0:300], ep B rows[6260:6560].
- frames sharing an **identical unix-ns timestamp: 150**.
- **GPS disagreement on those 150 shared frames: median 0.000000 m, p95 0.000000 m, max 0.000000 m.**
- ⇒ overlapping episodes are the SAME drive re-windowed; **dedup on `session_id` is EXACT — a
  `groupby`, not a matching heuristic** (retires trap #2).
- That drive: **34 raw episodes → 17 non-overlapping tiles kept (50 % dropped as duplicate overlap).**
  Ingesting episodes naively double-counts ~half the frames and would corrupt any held-out split.

## 5. ⚠️ Intrinsics decision — DOCUMENTED, not silently assumed (trap #1)

**Fact (MEASURED):** the only calibration artifact in the entire repo is `extrinsic_RDF.yaml`
(rotation+translation of 6 cameras; `cam_front_left` = reference, identity R, zero t, frame RDF
{x:right,y:down,z:front}). **No focal length, principal point, or distortion model ships anywhere.**
Our pipeline canonicalizes every camera to `f_eff = F_REF = 266` and *asserts* it
(`data/calib.py`); we cannot prove that crop for L2D.

**Decision (this note):**
1. **State layers (strategic/tactical/operative labels, poses, actions, vocab) are admitted in full**
   — they need no camera geometry. This is where L2D's unique value is (map + horizon + ego indicator).
2. **Camera-pixel path is admitted only with an ESTIMATED, flagged focal**, never `f_eff=266` as
   truth. The record carries `intrinsics_native={..., "estimated": true}` and `camera_model="pinhole"`;
   the arm card must state "camera scale UNVERIFIED — strategic/tactical head safe, absolute-metric
   camera geometry not asserted." Estimation path (follow-on): vanishing-point / known-width object,
   cross-checked against ego speed + the extrinsics, then the geometry falsifier. If it cannot be
   pinned, L2D camera frames feed only heads where absolute scale does not bind.
3. **The BEV `map` render is intrinsics-free** and is the zero-risk pixel option for a BEV-only arm.

The adapter therefore takes a `frame_source ∈ {none, front_camera, bev_map}` knob; the ingest
default is **state-only (`none`)** so nothing depends on an unproven focal by accident.

## 6. Licence RE-VERIFICATION (was INHERITED)

**Apache-2.0 — CONFIRMED (MEASURED):** README YAML front-matter `license: apache-2.0`; card body
`**License:** apache-2.0`; HF `cardData.license=apache-2.0`; tag `license:apache-2.0`; `gated=False`,
`private=False`. There is **no standalone `LICENSE` file** — the licence is declared via the HF card
metadata + README front-matter (the canonical HF-dataset mechanism). Registry entry
`schema.py:l2d = SourceLicense("owned-safe","Apache-2.0",share_alike=False)` → tier **`ship`,
commercial_ok=True** is correct. **GDPR:** real EU (German) driving-school footage; **no
anonymization statement anywhere in the card** (grep: 0 hits for anonymiz/face/plate/gdpr). Apache-2.0
grants the copyright, **not** the data-protection right → face/plate check REQUIRED before any
re-hosting of L2D frames. Flag for the card; do not solve tonight.

## 7. Vocabulary-slot mapping (L2D native → frozen v3 slots)

L2D fills more slots, and with higher evidence class (`map`/`human` vs `kinematic`), than any source
we have. Provenance uses the **frozen** `PROVENANCE` axis (a new value would break `validate_goal`);
the native-vs-derived distinction is recorded in a separate `slot_source` annotation, not in `prov`.

| frozen slot | L2D source field | mapping | `prov` | native? |
|---|---|---|---|---|
| **SIGNAL** (tactical) | `action.discrete[1]` turn_signal | 0→none, 1→indicator_left, 2→indicator_right | `human` | native |
| **VTARGET** (tactical) | `max_speed` (km/h) | ÷3.6 → `vtarget_band()` | `map` | derived |
| **VSOURCE** (tactical) | `max_speed` present | → `sign_limit` | `map` | native |
| **SPEEDPOLICY** (strategic) | `max_speed` magnitude | 0/NA→unknown, ≤30→cap_low, ≤60→cap_med, else cap_high (banded) | `map` | derived |
| **ROUTE** (strategic) | `task.instructions` keywords | straight/turn_left/right/roundabout/exit_*/merge/u_turn | `map` | derived |
| **ROUTEDIST** ⚠️ *v1.1 candidate, NOT frozen* | instruction metric distance | "150 m"/"0.7 km" → `routedist_band()` | `map` | derived |

⚠️ **ROUTEDIST is a v1.1 vocabulary CANDIDATE** (`vocab.py` V11 block), not in `GOAL_SLOTS`. L2D is
the concrete evidence for the bump Sayed has been holding: **96.74 % of instructions carry a metric
distance** (INHERITED; re-verify on the full task set). The adapter mints ROUTEDIST into a **sidecar
extension field**, NOT the validated goal tuple (inserting it would break the frozen 18-slot / 114-token
counts). **Escalation: enrol ROUTEDIST + TACDIST as vocab v1.1** so this label becomes first-class.

`road` (OSM class) and `lanes` (count) have **no direct frozen slot** — stored as native metadata
(they inform `VSOURCE=road_class_default` and future LANEOBJ), never force-fit. `precipitation/
conditions/lighting` fill the VLM-pending `scene_tags` skeleton with `prov=map` for free.

## 8. The one-drive slice — end-to-end proof (MEASURED, `l2d/slice_ingest_report.json`)

`L2DIngestor(frame_source="front_camera")` run through the REAL `ingest_source` path on the proof
drive's locally-available tiles (one front mp4, file-692, 434 MB):

- **14 records built** (drive-disjoint split put the single drive in `val`), **0 skipped, 0 corrupt**;
  each `frames [T,9,256,256]` uint8 real decoded pixels, `assert_contract(channels=9)` passes.
- Written to **2 SA-segregated shards** `shards/owned-safe/l2d/val/shard-000{00,01}.tar` + a 14-row
  Parquet catalog; every member **sha256-verified**; `two_pass_dedup` → 4 perceptual exemplars.
- **tier `ship`, commercial_ok=True** for all 14 — the licence firewall admits L2D exactly as designed.
- **Speed-unit CONFIRMED km/h** (not inferred): ep 721 `vehicle[0]` mean 59.0 → /3.6 = 16.39 m/s vs
  GPS-track-derived 16.17 m/s (matches within 1.4 %). The adapter divides by 3.6.
- **Vocab mapping on real state** (ep 721, *"go straight … observe the speed limit of 50 km/h …"*):
  ROUTE `straight`·map · VTARGET `v(12.5-15]`·map · VSOURCE `sign_limit`·map · SPEEDPOLICY `cap_med`·map
  · SIGNAL `none`·human · ROUTEDIST candidate `d_200_plus` (600 m). All pass `validate_goal`.

The pipeline is **unforked** — L2D records flow through the identical
dedup→tier→shard→catalog machinery the comma2k19 path uses.

## 9. What is built vs. what remains

**Built + staged (this session):** verified schema (this note); drive-reconstruction + GPS dedup proof
(`drive_dedup_summary.json`); the adapter `stack/tanitad/data/l2d.py` (drive discovery, non-overlapping
tiling, drive-disjoint split, state→poses/actions with the km/h fix + kinematic steer, instruction+speed
parsers, native vocab-slot minting, video-seek decode with the estimated-focal flag); `L2DIngestor` in
`stack/tanitad/lake/ingest.py`; `stack/tests/test_l2d.py` (9 tests, green); the end-to-end one-drive
slice (§8); provenance JSONs under `l2d-adapter-2026-07-22/`.

**Remaining (follow-on, ~2–3 eng-d):** full ingest of all 20 data parquets → drive-disjoint split;
the ESTIMATED-focal camera path + geometry falsifier; ROUTEDIST vocab-v1.1 enrolment; face/plate
GDPR check before any frame re-host; wire the L2D goal slots into `enrich.py` alongside the kinematic
minter.
