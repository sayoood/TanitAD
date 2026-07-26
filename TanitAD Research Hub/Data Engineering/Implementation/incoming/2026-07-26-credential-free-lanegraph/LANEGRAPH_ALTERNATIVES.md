# Credential-free routable lane-graph corpus — survey, access proof, verdict

**Date:** 2026-07-26 · **Agent:** data-engineering / lane-graph-alternatives
**Trigger:** `…/2026-07-26-nc-ingest/PI_DECISION_nuscenes.md` — the PI decided *"ingest freely"* on
nuScenes and **our whole side is built and green** (adapter `stack/tanitad/data/nuscenes.py`, 22 tests;
driver `stack/scripts/ingest_nuscenes.py`; corrected `SOURCE_REGISTRY`; 8 SA-shard regression tests).
**But nuScenes gates the bytes behind a human account + Terms click, and the PI has no computer access
for an indefinite period.** So the path is blocked on a human action that cannot happen.

⛔ **Hard constraint honoured throughout:** no routing around a Terms gate. The ~12 third-party
HF/GitHub **mirrors** of nuScenes, torrents and scraped copies were deliberately **not** touched.
A mirror of a gated corpus is a rejected candidate *by construction*, and is recorded as such below.

---

## 0. PRE-REGISTRATION — written before any probe was run

*(Committed here first so the verdict cannot be retro-fitted to whatever the probes happened to find.
Both outcomes are specified with their consequence.)*

**Decision rule.** A candidate **PASSES** only if all three hold, each MEASURED or PUBLISHED:

| # | criterion | what counts as satisfied | what does NOT count |
|---|---|---|---|
| P1 | **Credential-free** | an anonymous, unauthenticated HTTP/S3 request from this dev box returns HTTP 200 for *both* a listing and an actual data object | a login, an API token, an accepted EULA/Terms click, an email-gated link, a "request access" form, **or a third-party mirror of a gated corpus** |
| P2 | **Routable lane graph** | lane centrelines **plus an explicit successor/predecessor (or equivalent edge) relation**, byte-verified in a real downloaded file | a lane *polygon* or drivable-area raster with no edge relation; a BEV segmentation mask; "has HD map" in a paper |
| P3 | **Licence read as a document** | the licensor's own licence/terms page fetched and quoted, **data licence separated from code/devkit licence** | a short name from a README, a HF tag, a papers-with-code badge, or my own recall |

**Outcome A (pre-registered):** at least one candidate passes P1+P2+P3 → name it, publish the raw
access proof, give the `SOURCE_REGISTRY` entry with **conservative** flags, and give the adapter delta
against `stack/tanitad/data/nuscenes.py`. **Consequence:** S1/S2/HP-4 proceed without the PI.

**Outcome B (pre-registered):** every candidate is gated, or has a map with no connectivity → say so
plainly, name the specific failing criterion per candidate, and **stop**. **Consequence:** the
strategic-brain proof depends on AlpaSim's `trajdata.VectorMap` alone (being stood up on pod3) or waits
for the PI. **A clean negative is a full deliverable — a partial match will NOT be stretched into a PASS.**

**Pre-registered failure modes I am explicitly guarding against** (each has cost this program before):

- **Licence-from-short-name.** Made **twice** (ZOD; then nuScenes recorded `CC-BY-NC-4.0` when it is
  `CC-BY-NC-SA-4.0`, copyleft). Mitigation: fetch the licensor's page, save the bytes, quote it. Where
  the authoritative page is unreachable, record the **conservative branch and mark it a floor**.
- **Absence-at-one-location.** Mitigation: every "X is gated" / "X has no lane graph" claim is probed at
  ≥2 paths/names before it is written.
- **Devkit ≠ data.** Recorded as two separate columns, always.
- **"Has an HD map" ≠ routable.** P2 requires the *edge relation*, byte-verified.

---

## 1. VERDICT — **Outcome A. Argoverse 2 passes all three criteria.**

**Argoverse 2 is a genuinely credential-free corpus with a real routable lane graph, and it
substitutes for nuScenes on S1, S2 and HP-4.** No account, no Terms click, no token, no mirror.
I listed the bucket and downloaded 85 real map files from this dev box with `curl` and nothing else.

**One thing it does NOT fix, stated up front so it is not rediscovered:** AV2's data licence is
**`CC-BY-NC-SA-4.0` — the *same* licence as nuScenes**, non-commercial *and* copyleft. So AV2 removes
the **access** blocker, not the **publishability** one. A proof built on AV2 is still NC+SA and still
cannot ship commercially. The good news is that the plumbing this requires **already exists and is
already green**: the SA-shard routing and the C-tier refusal built for nuScenes apply to AV2 unchanged.

⚖️ **Precision on "credential-free", because the distinction is the whole point of this brief.**
Argoverse has a **Terms of Use** and it binds us — *"By using or downloading Argoverse, you … are
agreeing to comply with the terms of use"* (browse-wrap). What it does **not** have is an
**access-control gate**: no account to create, no checkbox for a human to tick, no credential to
issue. Downloading it accepts its terms in the ordinary way; **nothing is circumvented, and no gate
is routed around.** That is categorically different from nuScenes, where a human must register and
click before any byte is served.

---

## 2. Ranked candidate table

| # | corpus | access mechanism (MEASURED unless noted) | data licence — **as a document** | code/devkit licence | lane **connectivity**? | verdict |
|---|---|---|---|---|---|---|
| **1** | **Argoverse 2** (sensor · motion-forecasting · lidar · TbV) | **Anonymous `s3://argoverse` works.** ListBucket **HTTP 200** unsigned; object GET **HTTP 200** unsigned (2 files, 62 993 + 127 571 B). Publisher's own command is `s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*"`, and the user guide states *"an AWS account is not required to download the datasets"* | **`CC-BY-NC-SA-4.0`** — *"provided free of charge under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License"* (argoverse.org/about.html, fetched + saved). **Plus non-CC Terms of Use** (marketing, re-identification ban, unilateral termination) ⇒ registry entry is a **FLOOR** | **MIT** — *"Argoverse code and APIs are licensed under the MIT license"*; `av2-api/LICENSE` = MIT, © 2022 Argo AI | ✅ **YES, byte-verified.** `successors` + `predecessors` + `left_neighbor_id` + `right_neighbor_id` + `is_intersection` on **every** lane segment (7 692 segments across 85 maps, 100 % field presence) | ✅ **PASS — recommended** |
| 2 | **Argoverse 1 / 1.1** | same public bucket (`datasets/av1/`, `datasets/av1.1/` listed anonymously, HTTP 200) | same Argoverse ToU, `CC-BY-NC-SA-4.0`, © 2018-2019 Argo AI | MIT | PUBLISHED (devkit `get_lane_successor_ids`); **not byte-verified by me** | ⚪ superseded by AV2 — kept only as a second confirmation that the bucket is open |
| 3 | **OpenLane-V2** (OpenDriveLab) | ❌ **no official ungated host.** `OpenDriveLab/OpenLane-V2` returns **401 on both** the HF *datasets* and *models* namespaces, and the org's dataset listing (HTTP 200, n=11) **does not contain it**. Official bytes are **Google Drive / Baidu Yun / OpenDataLab** links in the repo's `data/README.md` | repo `LICENSE` is **Apache-2.0** — but that is the **code** badge; the annotations are derived from nuScenes (subset_A) and AV2 (subset_B) and **inherit those licences**, incl. nuScenes' NC-SA | Apache-2.0 | ✅ yes — `topology_lsls` lane-lane connectivity **and** traffic elements (lights/signs), which AV2 lacks | ⚠️ **REJECT as a corpus** — subset_A redistributes nuScenes imagery outside the Terms gate = the forbidden mirror case. **subset_B (AV2-derived) annotations are a legitimate *add-on*** if paired with imagery we pull ourselves; needs its own licence determination. See §6 |
| 4 | **Zenseact ZOD** | ❌ **gated by human application.** Publisher's own README: *"Prerequisites are that you have applied for access and received a download link."* `zod download --url="<download-link>"` — the URL **is** the credential | **`CC-BY-SA-4.0`** (share-alike, **but commercial-OK**) — README §License, © 2023 Zenseact AB. *(Registry already carries this correctly after the 2026-07-13 correction.)* | **MIT** (`zenseact/zod/LICENSE`, © 2021-2022 Zenseact AB) | not established — moot, blocked at P1 | ❌ **FAIL P1.** Painful: it is the only *commercially usable* candidate here. **Worth a PI ask** — one access application, and it is the publishable twin |
| 5 | **Waymo Open / WOMD** | ❌ **gated.** Anonymous GCS JSON list **401** on `waymo_open_dataset_v_1_4_3` *and* `waymo_open_dataset_motion_v_1_2_0`; XML endpoint **403**. Requires Google account + terms | Waymo Dataset License | — | (WOMD has `entry_lanes`/`exit_lanes`, PUBLISHED) | ❌❌ **FAIL P1 — and already `refuse` in our `SOURCE_REGISTRY`**: its terms follow the trained **weights**, so no tier can hold it. Do not revisit |
| 6 | **Lyft Level 5 / Woven** | ❌ **dead + gated.** `level-5.global` **does not resolve (NXDOMAIN)**; `s3://level5-avdata` **NoSuchBucket**; `s3://lyft-l5-datasets-public` **exists but 403 AccessDenied**; `woven.toyota/en/prediction-dataset/` **404** | not retrievable ⇒ **conservative floor: treat as NC + no-redistribution** | — | (semantic map had `lanes_ahead`, PUBLISHED) | ❌ **FAIL P1** — 4 independent probes, no live channel |
| 7 | **KITTI-360** | ❌ **login-gated.** `download.php` body executes `location.href = 'user_login.php'`; page chrome shows *"Logout \| Edit Account"* | `CC-BY-NC-SA-3.0` (registry, INHERITED — not re-fetched, blocked at P1) | — | ❌ no lane graph (semantic 3D + 2D/3D labels, no lane successor relation) | ❌ **FAIL P1 and P2** |
| 8 | **nuPlan** | ❌ **same Motional account gate as nuScenes** (nuscenes.org/nuplan; page is a JS app returning no static text — 2 fetches) | inherits the nuScenes ToU | Apache-2.0 devkit | ✅ rich (nuPlan maps have full lane connectivity, PUBLISHED) | ❌ **FAIL P1** — identical blocker to the one we are routing around; no gain |
| 9 | **nuScenes** *(baseline)* | ❌ human account + Terms click | `CC-BY-NC-SA-4.0` (+ modifications) | Apache-2.0 | ✅ `lane_connector`, `is_intersection`, YIELD stop-lines | ❌ blocked — **this is what we are replacing** |

**The ~12 third-party HuggingFace/GitHub mirrors of nuScenes were not probed, not listed and not used.**
They are excluded by construction, not by ranking.

---

## 3. Empirical access proof (Outcome A evidence)

**Class: MEASURED — from this dev box, `curl` only, no credential of any kind, 2026-07-26.**
Raw artefacts: `evidence/` (see manifest §8).

```
# 1. anonymous ListBucket — no signature, no AWS account
curl "https://s3.amazonaws.com/argoverse?list-type=2&delimiter=/&max-keys=100"
  -> HTTP 200 · CommonPrefixes: assets/ datasets/ tasks/
  -> datasets/av2/{sensor,lidar,motion-forecasting,tbv,tars,demo,archive}/

# 2. anonymous object GET — a real routable lane graph
curl "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/val/\
00010486-9a07-48ae-b493-cf4545855937/log_map_archive_00010486-….json"
  -> HTTP 200 · 62 993 bytes
curl "https://s3.amazonaws.com/argoverse/datasets/av2/sensor/val/\
02678d04-cc9f-3148-9f95-1ba66347dff9/map/log_map_archive_02678d04-…____PIT_city_71109.json"
  -> HTTP 200 · 127 571 bytes
```

⚠️ **Dev-box-specific gotcha, worth recording:** bare `curl` fails here with
`schannel: CRYPT_E_NO_REVOCATION_CHECK` behind the TLS proxy — it looks exactly like a network
outage (`HTTP=000`). **`curl --ssl-no-revoke` fixes it.** This is the `curl` sibling of the known
Python `truststore.inject_into_ssl()` workaround; without it every probe in this report reads as
"host unreachable" and the whole survey would have produced a false negative.

**Publisher's own documented mechanism** (PUBLISHED, `argoverse.github.io/user-guide/getting_started.html`):
`s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR`, with the
explicit statement *"an AWS account is not required to download the datasets."*
⇒ Two independent confirmations that access is unauthenticated: **our own unsigned HTTP 200s**, and
**the licensor's published `--no-sign-request` instruction.**

**Total bytes pulled for this entire investigation: ~7.7 MB** (85 map JSONs + HTML/licence pages).
**No bulk download was performed.** See §5 for the sizes and the exact command that would be needed,
which is **held for approval**.

---

## 4. Lane-connectivity proof — MEASURED, not "it has an HD map"

Criterion P2 demands the **edge relation**, byte-verified. Both AV2 splits were sampled independently.
Raw: `evidence/av2_lanegraph_stats.json`, `evidence/av2_sensor_lanegraph_stats.json`.

### 4a. Motion-forecasting `val` — 60 scenario maps, 4 201 lane segments

| quantity | value | what it buys |
|---|---|---|
| lane segments with ≥1 **`successor`** | **4 161 / 4 201 = 99.0 %** | the graph is *dense*, not decorative |
| lane segments with ≥1 **`predecessor`** | 4 152 / 4 201 = 98.8 % | reverse traversal works |
| directed **successor edges** | **4 854** | |
| **branch points** (out-degree ≥ 2) | **514** (2→371, 3→115, 4→20, 5→8) | **S1 — branch selection** |
| **merge points** (in-degree ≥ 2) | 513 | |
| maps containing **≥1 branch** | **60 / 60 = 100 %** | every scenario has an S1 decision |
| `is_intersection` lane segments | **1 421 / 4 201 = 33.8 %** | **HP-4 — junction topology** |
| maps containing ≥1 intersection lane | **60 / 60 = 100 %** | |
| `left_neighbor_id` present | 2 693 (64.1 %) | **S2 — lane selection** |
| `right_neighbor_id` present | 1 602 (38.1 %) | |
| successor refs resolving **inside** the local map | 4 270 / 4 854 = **88.0 %** | see caveat ↓ |
| `lane_type` | VEHICLE 3 576 · BIKE 599 · BUS 26 | |
| explicit **`centerline`** field | **present (4 201/4 201)** | free — no midpoint computation |

### 4b. Sensor `val` (the split that carries **imagery**) — 25 log maps, 3 846 lane segments

| quantity | value |
|---|---|
| directed successor edges | **4 492** |
| branch points (out-degree ≥ 2) | **538** (2→393, 3→127, 4→9, 5→9) |
| maps with ≥1 branch / ≥1 intersection | **25 / 25** and **25 / 25** |
| `is_intersection` | **1 395 / 3 846 = 36.3 %** |
| `left` / `right` neighbour present | 2 374 (61.7 %) / 969 (25.2 %) |
| successor refs resolving locally | 4 126 / 4 492 = **91.9 %** |
| logs whose successor graph contains a **directed cycle** | **9 / 25 = 36 %** ← loop/roundabout proxy |
| cities in the sample | MIA 11 · PIT 7 · WDC 5 · DTW 2 |
| explicit **`centerline`** field | ❌ **0 / 3 846 — ABSENT** (adapter delta, §7) |
| `ground_height` raster + `img_Sim2_city` present | 25 / 25 |

**Caveat, stated because it is a real limitation and not a rounding error:** 8–12 % of successor
references point to lane ids **outside** the local map. This is the **crop boundary**, not corruption —
AV2 ships a *log-local / scenario-local* map, so edges leaving the region dangle. Consequence for us:
route/branch labels must be computed **inside** the local map and any traversal must treat a dangling
successor as a terminal, not as an error. The sensor split (whole-log maps) dangles less (8.1 %) than
the scenario-cropped motion-forecasting split (12.0 %), exactly as that explanation predicts.

### 4c. What AV2 does **NOT** have — established at two independent probes

Probe 1 (MEASURED): across **all 85 downloaded maps**, the complete set of top-level layers is exactly
`{drivable_areas, lane_segments, pedestrian_crossings}`, and the complete set of lane-segment fields is
the 12 listed above. A name-scan for `stop|traffic|light|signal|roundabout|junction|route|goal`
returned **zero matches**.
Probe 2 (PUBLISHED): the MIT-licensed devkit's own schema — `av2/map/map_api.py` `ArgoverseStaticMap`
carries `vector_drivable_areas`, `vector_lane_segments`, `vector_pedestrian_crossings`,
`raster_ground_height_layer`, `raster_drivable_area_layer`, `raster_roi_layer` — **and nothing else**.

⇒ **No stop-lines. No traffic-light state or position. No roundabout label. No route/goal signal.**
- vs nuScenes: nuScenes' **YIELD stop-lines** are lost. Roundabouts are **not labelled** in either, but
  AV2 gives a usable *derivable* proxy — the 36 % of sensor logs with a directed cycle.
- Traffic-light **state** was already known to be absent from nuScenes too (PI decision §4), so this is
  **not a regression** against the plan of record. It remains OpenLane-V2 territory (§6).

---

## 5. Sizes and the exact command — HELD FOR APPROVAL, nothing large downloaded

MEASURED from the anonymous listing of `datasets/av2/tars/` (44 objects), `evidence/av2_tars_sizes.json`:

| bundle | tar objects | size |
|---|---|---|
| `sensor` (7 ring + 2 stereo cameras, lidar, 3D tracks, **maps**) | 20 | **1 051.26 GiB** |
| `motion-forecasting` (agent tracks + **maps**, *no imagery*) | 3 | **57.45 GiB** |
| `tbv` (Trust-but-Verify map-change, imagery + maps) | 21 | **818.03 GiB** |
| **total** | 44 | **1 926.75 GiB** |

**⛔ Not run. Requires approval before anything large moves:**
```
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/*"  <TARGET>   # ~1.03 TiB
```

**Exact split sizes** (MEASURED, full anonymous pagination — `evidence/av2_split_counts.json`):

| split | units | | split | units |
|---|---|---|---|---|
| `sensor/train` | **700** logs | | `motion-forecasting/train` | **199 908** scenarios |
| `sensor/val` | **150** logs | | `motion-forecasting/val` | **24 988** |
| `sensor/test` | **150** logs | | `motion-forecasting/test` | **24 984** |
| `lidar/{train,val,test}` | 16 000 / 2 000 / 2 000 | | `tbv` | **1 043** logs |

### 🔑 The cheap path — this is the recommendation

The lane graphs are **individually addressable S3 objects**; they do **not** require the tars.
Mean map size MEASURED at **154 353 B** (sensor, n=25) and **107 941 B** (motion-forecasting, n=60):

| pull | scope | size |
|---|---|---|
| **all 1 000 AV2 sensor lane graphs** | the split that has imagery | **≈ 147 MiB** *(ESTIMATED from n=25)* |
| all 249 880 motion-forecasting lane graphs | no imagery, but 250 k junction topologies | ≈ 25.1 GiB *(ESTIMATED from n=60)* |
| *(for contrast)* sensor imagery tars | | 1 051 GiB |

⇒ **The entire strategic-brain ground truth for the imagery split costs ~147 MiB** — a rounding error —
and can be pulled and evaluated **before** committing to a single image byte. That is the same
metadata-first discipline `nuscenes.py` was built around, and it survives the corpus swap intact.
**This is the pull I recommend approving first**, and it is small enough that "approve" is nearly free.

---

## 7. Adapter delta — what a sibling `argoverse2.py` costs against `nuscenes.py`

`stack/tanitad/data/nuscenes.py` (34 264 B, **22 tests green**) is the template. Its structure carries
over almost unchanged, because AV2 is *simpler*: no global token graph, and per-log directories instead
of monolithic tables. **Class: MEASURED for every AV2-side fact (schema read off downloaded bytes);
ESTIMATED for the effort figure.**

| `nuscenes.py` element | AV2 equivalent | delta |
|---|---|---|
| `load_tables` — 13 JSON tables + `NuScenesTermsError` | per-log dirs; `city_SE3_egovehicle.feather`, `calibration/*.feather`, `annotations.feather` | **rewrite** (feather/parquet, not JSON). ⚠️ the `NuScenesTermsError` guard **has no analogue and must not be copied** — AV2 needs no terms gate; a copied guard would be a lie in the code |
| `NuScenesIndex` (token-join graph: sample→sample_data→ego_pose→calibrated_sensor) | directory + timestamp join | **much simpler** — AV2 has no token indirection |
| `ego_track`, `actions_from_track` | `city_SE3_egovehicle` poses | **near-verbatim reuse** (both are SE(3) + timestamps) |
| `quat_to_rotmat` / `quat_to_yaw` / `wrap_pi` | identical convention (`qw,qx,qy,qz`) | **verbatim reuse** |
| `agent_tracks` (instance ids) | `annotations.feather` `track_uuid` | near-verbatim |
| `camera_intrinsics_of` → `PinholeIntrinsics`; `calib.pinhole_rectify` | AV2 ships pinhole intrinsics **already rectified** | **simpler** — the `fx 1266.4 → 266.0` rectification step is **not needed** |
| `camera_visibility` (cross-camera frustum projection) | 7 ring cameras (vs 6) + 2 stereo | **reuse; change the channel list.** AV2 ring is a *superset* — strictly better for the camera-attention workstream |
| `build_episode` / `load_keyframe_images` / `canonicalize_frames` | `sensors/cameras/<cam>/<ts>.jpg` | reuse; AV2 is **20 Hz** vs nuScenes **2 Hz keyframes** — a real gain, and it removes the keyframe-interpolation hack |
| `split_unit_of` / `discover_scenes` | `train`/`val`/`test` are **directories** | **simpler** — no `splits.json` |
| *(no analogue)* | **`load_lane_graph`** — the new thing | **NEW, ~120 lines.** Parse `log_map_archive_*.json` → `{id: LaneSegment}`; build successor/predecessor adjacency; expose `branch_points()`, `is_intersection`, lateral neighbours |

**Three AV2-specific traps to encode in the adapter — each MEASURED here, each will otherwise cost a day:**

1. **Sensor maps have NO `centerline` field** (0 / 3 846 measured), motion-forecasting maps **do**
   (4 201 / 4 201). The devkit computes it from the two boundaries. An adapter that assumes `centerline`
   will work perfectly on motion-forecasting and **fail on every sensor log** — i.e. on exactly the split
   that has the images. Compute the midpoint line for sensor.
2. **8–12 % of successor ids point outside the local map** (crop boundary, §4). Traversal must treat a
   dangling successor as terminal, not raise.
3. **Lane ids are unique only *within* one local map** — the devkit says so verbatim: *"guaranteed to be
   unique only within this local map"*. Never build a cross-log lane index on the raw id.

**ESTIMATED effort: ~0.5–1 day**, dominated by the feather/parquet IO swap and `load_lane_graph`.
The geometry, pose, action, visibility and episode-assembly code is reusable as-is.

---

## 8. Proposed `SOURCE_REGISTRY` entry — **it already exists, and it is CORRECT**

`stack/tanitad/lake/schema.py:104-105` already carries:

```python
"argoverse2": SourceLicense("nc-research", "CC-BY-NC-SA-4.0",
                            share_alike=True, is_synthetic=False),
```

**I verified this independently against the licensor's own document** (not against a short name, not
against another doc) and it is right on every field:

| field | value | why — MEASURED against argoverse.org/about.html |
|---|---|---|
| `license_class` | `nc-research` | *"We license Argoverse data and documents to you for **non-commercial use only**"* |
| `license_name` | `CC-BY-NC-SA-4.0` | verbatim: *"Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License"* |
| `share_alike` | `True` | the **SA** in the licence name; the ToU adds *"you may be required to confirm that your changes are subject to CC BY-NC-SA 4.0"* |
| `is_synthetic` | `False` | real camera + lidar |
| `commercial_ok` (derived) | **`False`** | derived property = `owned-safe AND NOT share_alike` → False on both counts |

**No change is proposed. The correct action is to record that it was verified**, so the next agent does
not re-derive it. Two riders:

- ⚠️ **This entry is a FLOOR, not a ceiling** — same rider the corrected nuScenes entry carries. The
  Argoverse **Terms of Use add obligations beyond the CC grant**: a trademark/marketing restriction, a
  ban on re-identifying individuals or combining with another dataset to do so, and a clause letting
  Argo *"decline to grant a license, and/or elect to terminate a license"* including where use
  *"has the potential to negatively impact Argo's reputation."* None of these fit a CC short name.
- ✅ **Routing is already correct and already tested.** `share_alike=True` + `nc-research` sends AV2 to
  the segregated copyleft shard and makes it refuse entry to `TanitDataSet-C`. **The 8 SA-shard
  regression tests written for nuScenes cover AV2 unchanged** — same class, same two flags.

**Counterpart correction I did *not* find any need for:** ZOD's entry
(`owned-safe`, `CC-BY-SA-4.0`, `share_alike=True`) also matches the licensor's README verbatim
(*"licensed under CC BY-SA 4.0"*, devkit MIT). That one is right too.

---

## 9. What this substitutes for — plainly

| goal | nuScenes would have given | **Argoverse 2 gives** | verdict |
|---|---|---|---|
| **S1 — branch selection** | `lane_connector` graph | **514 branch points in 60 val maps; 100 % of maps contain ≥1 branch**; out-degree up to 5 | ✅ **full substitute** |
| **S2 — lane selection** | lateral lane adjacency | `left_neighbor_id` 64.1 % / `right_neighbor_id` 38.1 %, plus per-side `LaneMarkType` (10 distinct types measured) telling you whether the change is *legal* | ✅ **full substitute — arguably richer** (nuScenes has no lane-mark type) |
| **HP-4 — unseen junction topologies** | `is_intersection` | **33.8 % of lane segments** `is_intersection`; 100 % of maps contain one; **6 cities MEASURED** over 145 sensor logs — PIT 53 · MIA 49 · DTW 20 · WDC 17 · ATX 4 · PAO 2; **249 880 motion-forecasting scenarios** = a very large topology bank for held-out-**city** and held-out-junction splits | ✅ **full substitute — materially better** (nuScenes is 2 cities) |
| Roundabouts | unpublished/unknown | **not labelled**, but derivable: **9 / 25 sensor logs (36 %) contain a directed cycle** | 🟡 **parity — both need derivation** |
| Stop lines | **YIELD stop-lines** | ❌ **absent** | ❌ **regression — the one real loss** |
| Traffic-light state | ❌ absent (PI decision §4) | ❌ absent | ⚪ no change |
| Commercial / publishable proof | ❌ NC+SA | ❌ **NC+SA — identical** | ⚪ **no change — AV2 fixes access, not publishability** |
| Imagery for a vision world model | 6 cam @ 2 Hz keyframes, 1 000 scenes | **7 ring + 2 stereo @ 20 Hz, 1 000 logs** | ✅ **better** |

**Bottom line: AV2 substitutes for nuScenes on S1, S2 and HP-4 — the three things the PI decision said
nuScenes was for — and beats it on imagery, cities and scale. It loses stop-lines, and it does not
improve publishability.**

---

## 10. Recommendation + ESCALATION

**Recommended:** proceed on **Argoverse 2** instead of waiting for the PI's computer.

1. **Approve the ~147 MiB pull** of all 1 000 AV2 sensor lane graphs (§5). Metadata-first, no imagery,
   answers every remaining value question. *(Not run — held for approval per the brief.)*
2. **Build `stack/tanitad/data/argoverse2.py`** as a sibling of `nuscenes.py` (§7, ~0.5–1 day).
3. **Keep `stack/tanitad/data/nuscenes.py` exactly as it is.** It is 22-tests green and costs nothing
   to hold; the moment the PI accepts the Terms it lights up, and the two corpora are the same licence
   class so they can share the SA shard.

🔴 **ESCALATION 1 — the publishable twin is still open, and ZOD is now the whole answer.**
AV2 does **not** make the strategic-brain proof shippable. Of everything surveyed, **Zenseact ZOD is the
only commercially usable candidate** (`CC-BY-SA-4.0` — share-alike but **not** non-commercial), and it is
blocked by **one human action: an access application**. That is a *smaller* PI ask than the nuScenes
Terms flow and it buys strictly more. **Recommend putting it to the PI as its own decision.**

🔴 **ESCALATION 2 — `argoverse2` is in `SOURCE_REGISTRY` but there is no adapter.** The registry entry has
been sitting there correct and unused. Nothing in `stack/tanitad/data/` reads AV2. **This report is the
integration request** — it is not written into a README to be found later.

🟢 **Relation to the AlpaSim fallback (commit `a6dd9de`, landed while this ran).** The
`trajdata.VectorMap` corridor instrument reported its own Outcome A — 51/51 scenes, lane width
3.359 m, ego-in-lane containment 0.9837. **The two are complementary, not redundant:** that work
verifies *lane geometry* (edges, widths, corridor departure); this work supplies the *lane graph*
(successor/predecessor edges, branch points, `is_intersection`) at **~5 000× the scale** — 249 880
scenarios and 1 000 imagery logs vs 51 scenes. **The strategic-brain proof no longer depends on
either one alone.** *(Its 51-scene VectorMap is nuPlan-derived, so its licence lineage differs from
AV2's — flagged, not assessed here.)*

⚪ **Noted, not recommended:** MetaDrive is already wired into our stack
(`stack/tanitad/data/metadrive_env.py`, `metadrive_frontcam.py`) but is used only with the `"S"`
straight-block map and **is not installed on this box** — so its procedural junction topologies are a
**HYPOTHESIS-class** complement for HP-4, not a verified candidate. I did not probe it further.

---

## 11. Deliverable manifest

**Everything below is in the repo and `git add`-staged. Nothing lives only in a scratchpad, only on a
pod, or only in this agent's context. No pod was touched. Nothing was committed or pushed.**

| # | artifact | where it lives | only one place? |
|---|---|---|---|
| 1 | `LANEGRAPH_ALTERNATIVES.md` — pre-registration, ranked table, access proof, connectivity proof, adapter delta, recommendation | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-26-credential-free-lanegraph/` | no — staged |
| 2 | `evidence/access_probes.json` — every probe: URL, HTTP code, verdict, per corpus | same dir | no — staged |
| 3 | `evidence/license_determinations.json` — every licence as a **document**, verbatim quotes, data-vs-code split, FLOOR markers | same dir | no — staged |
| 4 | `evidence/av2_lanegraph_stats.json` — MEASURED connectivity, 60 motion-forecasting val maps (+ per-file S3 keys and sizes) | same dir | no — staged |
| 5 | `evidence/av2_sensor_lanegraph_stats.json` — MEASURED connectivity, 25 sensor val log maps | same dir | no — staged |
| 6 | `evidence/av2_tars_sizes.json` — all 44 tar objects with exact byte sizes | same dir | no — staged |
| 7 | `evidence/av2_split_counts.json` — exact per-split log/scenario counts (full pagination) | same dir | no — staged |
| 8 | `evidence/av2_cities_measured.json` — 6-city distribution over 120 additional sensor logs | same dir | no — staged |
| 9 | `evidence/argoverse_terms_of_use_2026-07-26.txt` — the licensor's ToU page, text-extracted, as fetched | same dir | no — staged |
| 10 | `evidence/av2_sample.py`, `evidence/av2_sensor_sample.py` — the two measurement scripts (rerunnable, anonymous, no credential) | same dir | no — staged |

**Deliberately NOT staged — and why:**

- **The 85 raw AV2 map JSONs** (~10.3 MB) stay in the scratchpad. AV2 is `CC-BY-NC-SA-4.0`; committing
  them would be **redistribution of source bytes**, which our policy forbids (*"pointers + derived
  features, never source bytes"*). The staged JSONs carry **pointers (S3 keys) + derived statistics**
  only. They are fully reproducible by rerunning the two staged scripts.
  *Scratchpad location, for this session only:*
  `…/scratchpad/av2_maps/*.json`, `…/scratchpad/av2_sensor_maps/*.json`.
- **No PhysicalAI-AV content of any kind** appears in any artifact here — no clip UUIDs, no raw content.
- **`Keys.txt` was never read, printed, copied or passed in argv.** No credential was needed: every
  probe in this report is anonymous, which is the entire finding.

## 12. Claims ledger — evidence class on everything load-bearing

| claim | class | where |
|---|---|---|
| AV2 is anonymously downloadable (list + object GET, HTTP 200, no credential) | **MEASURED** | `evidence/access_probes.json` probes 1-6 |
| Publisher states no AWS account is required; `s5cmd --no-sign-request` | **PUBLISHED** | argoverse.github.io user guide |
| AV2 data licence is `CC-BY-NC-SA-4.0`; code is MIT | **PUBLISHED** (licensor's page, saved) | `evidence/license_determinations.json`, `evidence/argoverse_terms_of_use_2026-07-26.txt` |
| AV2 lane segments carry `successors`/`predecessors`/neighbours/`is_intersection` on 100 % of 7 692 segments | **MEASURED** | the two `*_lanegraph_stats.json` |
| 514 + 538 branch points; 100 % of maps contain a branch and an intersection | **MEASURED** | same |
| Sensor maps have **no** `centerline` field (0/3 846) | **MEASURED** | `av2_sensor_lanegraph_stats.json` |
| AV2 has **no** stop-lines / traffic lights / roundabout label / route-goal | **MEASURED** (85 maps) **+ PUBLISHED** (MIT devkit `ArgoverseStaticMap` schema) | §4c — two independent probes |
| 9/25 sensor logs contain a directed cycle (roundabout proxy) | **MEASURED** | `av2_sensor_lanegraph_stats.json` |
| Tar sizes 1 051.26 / 57.45 / 818.03 GiB; split counts 700/150/150, 199 908/24 988/24 984, 1 043 | **MEASURED** | `av2_tars_sizes.json`, `av2_split_counts.json` |
| 6 cities: PIT 53 · MIA 49 · DTW 20 · WDC 17 · ATX 4 · PAO 2 (145 logs) | **MEASURED** | `av2_cities_measured.json` + sensor stats |
| "All sensor lane graphs ≈ 147 MiB" and "all MF ≈ 25.1 GiB" | **ESTIMATED** (extrapolated from n=25 / n=60 means) | §5 — marked as estimates |
| Waymo / Lyft / ZOD / KITTI-360 / OpenLane-V2 gates | **MEASURED** (≥2 probes each) | `evidence/access_probes.json` |
| nuPlan gate | **INHERITED** — site is client-rendered, could not re-verify by fetch | flagged in `access_probes.json`; decides nothing here |
| nuScenes gate | **INHERITED** from `…/2026-07-26-nc-ingest/PI_DECISION_nuscenes.md` (the brief's premise) | not re-probed |
| Adapter effort ~0.5-1 day | **ESTIMATED** | §7 |
| MetaDrive as an HP-4 complement | **HYPOTHESIS** — not installed here, not probed | §10 |


---

## 6. Traffic elements — the one honest gap, and how it could be closed

AV2 has **no traffic-light or stop-line layer** (§4c). The only candidate that supplies them is
**OpenLane-V2**, and it is **rejected as a corpus** (§2 row 3) because its `subset_A` redistributes
nuScenes imagery outside the Terms gate — precisely the mirror case this brief forbids.

**What is *not* rejected, and is worth a follow-up:** OpenLane-V2 **`subset_B` is derived from
Argoverse 2**. Taking only its **annotations** (lane-lane `topology_lsls`, and traffic elements) while
pulling the **imagery ourselves from the public AV2 bucket** is a composition, not a mirror. It would
add traffic-light/sign elements on top of a corpus we can already access legitimately.
**UNVERIFIED and out of scope here** — it needs its own licence determination (the repo's Apache-2.0
badge is the **code** licence; the annotations inherit AV2's NC-SA), and its bytes still come from
Google Drive / Baidu / OpenDataLab. **Flagged, not recommended, pending that check.**

---

