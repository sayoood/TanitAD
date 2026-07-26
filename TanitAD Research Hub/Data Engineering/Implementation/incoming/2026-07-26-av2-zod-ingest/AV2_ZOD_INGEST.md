# AV2 lane-graph pull · adapter · ZOD requirements · new alternatives

**Date:** 2026-07-26 · **Agent:** data-engineering / av2-zod-ingest
**Continues:** `…/2026-07-26-credential-free-lanegraph/LANEGRAPH_ALTERNATIVES.md`
**Executes:** the PI's three decisions — pull AV2 (approved), pursue ZOD, keep hunting.
**Compute:** dev box only. **No pod was touched** (pod1 training, pod2 H2 classifier, eval pod Bar-A).

---

## 0. HEADLINE — with its tier

> **The AV2 pull is 1000/1000 — DECISION-GRADE.** Every one of the 1 000 sensor lane graphs
> downloaded anonymously, byte-exact against the bucket's own `<Size>`, **MD5-exact against its
> `ETag`**, and parsed. Zero failures. 161 255 215 B (153.8 MiB) in 115 s.
> **The adapter is built, in the repo, and 46/46 green — including 4 tests that run over all 1 000
> real maps.** Full suite `1099 passed, 7 skipped`.

Three findings that change what we should do next, in descending order of consequence:

1. 🔴 **ZOD does NOT have a lane graph.** Not a weak one — *none*. Its lane annotation is
   **2-D image-space polygons with no connectivity**, established at **three** independent probes.
   So ZOD's commercial licence **does not buy the publishable strategic-brain twin.** The plan of
   record for ZOD was wrong about what ZOD contains. §4.
2. 🟢 **A commercially-usable routable graph exists, and we can already reach it:
   Overture Maps transportation.** Anonymous, no gate (2 endpoints, HTTP 200), **byte-verified
   connectivity** (20 000/20 000 sampled segments carry ≥2 connectors), and it carries **turn
   restrictions and signposted destinations — which AV2 does not have.** §5.
3. ⚠️ **Counting AV2 branch points on raw successors overstates them by 8.9 %** — 22 606 raw vs
   **20 591 real**; 2 015 are "branches" only because one option leaves the local crop. Anything
   that sizes the S1 decision set must use the resolved count. §2.3.

**One thing the PI must decide (§4c):** ZOD is *also* published by Zenseact themselves on
**Academic Torrents**. The brief forbids torrents outright; the brief's *rationale* is "never route
around a licensor's gate", which a licensor-published channel does not do. **I did not touch it.**
This is flagged as a PI decision, not acted on.

---

## 1. TASK 1 — the AV2 pull. Verified yield: **1000/1000**

**Class: MEASURED. Tier: DECISION-GRADE** (whole population, not a sample; two independent code
paths agree — §2.4).

### 1.1 What "verified" means here

The brief demanded completeness, not presence — *"this program has been burned by a `find` hit on a
file that was 48.9 MB short."* So a file counts as OK only if it passes **four** gates:

| gate | mechanism | result |
|---|---|---|
| **byte count** | local `os.path.getsize` vs the bucket's own `<Size>` from ListObjectsV2 | **1000 / 1000** |
| **integrity** | local MD5 vs the object's `<ETag>` (== MD5 for these single-part objects) | **1000 / 1000** |
| **parse** | `json.loads` on the full file | **1000 / 1000** |
| **semantic** | `lane_segments` non-empty AND ≥1 successor edge | **1000 / 1000** |

**Yield: 1000/1000 = 100.0 %. Failures: 0.** Raw per-file record for all 1 000 (S3 key, remote
size, ETag, local size, all four gate results, derived counts):
`evidence/av2_pull_manifest_1000.json`. Summary: `evidence/av2_pull_summary_1000.json`.

*Reported as the brief required: this is a genuine 1000/1000, not a rounded 940/1000. Had it been
940 it would say 940.*

### 1.2 Access — no credential of any kind

```
GET https://s3.amazonaws.com/argoverse?list-type=2&prefix=datasets/av2/sensor/{split}/&delimiter=/
GET https://s3.amazonaws.com/argoverse/datasets/av2/sensor/{split}/<log>/map/log_map_archive_*.json
  -> HTTP 200, unsigned, anonymous
```
No account, no token, no Terms click, **no mirror**. Two independent confirmations, as before: our
own unsigned HTTP 200s, and the licensor's published `s5cmd --no-sign-request` instruction.

⚠️ **The dev-box trap held exactly as briefed.** Bare `curl` here returns `HTTP=000`
(`CRYPT_E_NO_REVOCATION_CHECK`), indistinguishable from an outage. **`--ssl-no-revoke` is in every
request in this report.** The Python probes use the sibling workaround
`truststore.inject_into_ssl()`. Without either, every number here would have been a false negative.

### 1.3 Cost and what was NOT pulled

| | |
|---|---|
| pulled | **161 255 215 B = 153.8 MiB**, 1 000 files, 115.1 s, 14 workers |
| vs the estimate it replaces | 147 MiB ESTIMATED from n=25 → **153.8 MiB MEASURED** (+4.6 %) |
| ⛔ **not touched** | the 1 051.26 GiB sensor tars, the 818.03 GiB TbV tars, the 57.45 GiB MF tars |

### 1.4 What the 1 000 maps contain

MEASURED over the full population (`evidence/av2_pull_summary_1000.json`,
reproduced independently by the adapter in `evidence/av2_adapter_corpus_stats_1000.json`):

| quantity | value | what it buys |
|---|---|---|
| lane segments | **163 698** | |
| successor edges (raw / resolved) | 191 770 / **175 629** | |
| **branch points (resolved)** | **20 591** | **S1 — branch selection** |
| merge points | 20 561 | |
| `is_intersection` segments | **57 415** (35.1 %) | **HP-4 — junction topology** |
| left / right neighbour | 104 924 (64.1 %) / 44 788 (27.4 %) | **S2 — lane selection** |
| maps with ≥1 branch | **998 / 1000** | 2 maps have none |
| maps with ≥1 intersection | **997 / 1000** | 3 maps have none |
| maps containing a directed cycle | **368 / 1000 = 36.8 %** | roundabout/loop proxy |
| lane types | VEHICLE 149 645 · BIKE 12 370 · BUS 1 683 | |
| cities | MIA 354 · PIT 350 · WDC 126 · DTW 117 · ATX 31 · PAO 22 | 6 cities for held-out-city splits |
| top-level layers | `{drivable_areas, lane_segments, pedestrian_crossings}` on **1000/1000** | |
| explicit `centerline` | **0 / 163 698** | **the trap — §2.1** |

**Two prior sampled numbers now confirmed at full population:** the 36 % directed-cycle rate
(9/25 sample → **36.8 %** over 1 000) and the ~8 % dangling rate (**8.4 %**: 16 141 / 191 770).
Both held.

---

## 2. TASK 1b — the adapter, and the trap it exists to prevent

**In the repo:** `stack/tanitad/data/argoverse2.py` · **tests:** `stack/tests/test_argoverse2.py`
**46 passed** (42 fixture + **4 over all 1 000 real maps**). Full suite: **1099 passed, 7 skipped**.

### 2.1 THE documented trap, confirmed at full scale, and how the tests catch it

> **AV2 SENSOR maps have NO `centerline` field.** MEASURED BY ME **0 / 163 698** segments across
> all 1 000 sensor logs.
> *(The motion-forecasting counterpart — present on 4 201 / 4 201 — is **INHERITED** from the
> previous agent's n=60 sample and was **not** re-measured here. It decides nothing in this report:
> the adapter handles both flavours and the explicit-centerline path is covered by a fixture test,
> not by that number.)*

An adapter written against motion-forecasting therefore **passes every test it is given and then
fails on every log that has images.** The tests are built so that cannot happen:

| test | what it would catch |
|---|---|
| `test_sensor_fixture_is_faithful_no_centerline_field` | someone "helpfully" adding a centerline to the sensor fixture, which would silently disarm every test below |
| `test_naive_centerline_access_raises_on_sensor_map` | **asserts the bug is real** — `seg["centerline"]` raises `KeyError` on a sensor-shaped record |
| `test_centerline_is_derived_on_sensor_map` | the adapter deriving a wrong midpoint, or not deriving one |
| `test_centerline_is_explicit_on_motion_forecasting_map` | the adapter **ignoring** a shipped centerline. The fixture's explicit line sits at x=1.0 while the boundary midpoint is x=2.0, so a silent recompute fails loudly |
| `test_every_sensor_lane_yields_a_usable_centerline` | a per-lane failure hiding behind a passing lane-1 test |
| **`test_real_av2_sensor_split_has_zero_explicit_centerlines`** | **the claim itself, on all 1 000 real maps** |

The API makes the trap unhittable: there is no way to read a raw centerline. Every consumer goes
through `LaneGraph.centerline(id)`, which returns the explicit polyline when present and otherwise
derives the midpoint — and `centerline_source(id)` reports **which**, so a derived centerline is
never silently quoted as ground truth.

### 2.2 A second sensor-only trap, found while building (not in the brief)

**Left and right lane boundaries have DIFFERENT lengths on 49.0 % of segments** (MEASURED,
n=19 713). A naive elementwise mean of the two boundaries is therefore wrong about half the time —
and in numpy it does not even fail cleanly, it raises a shape error only when the lengths differ by
more than broadcasting tolerates. `midpoint_line()` arc-length **resamples** both polylines onto a
common parameterisation first. Arc length, not index: AV2 boundary vertices are not equally spaced,
so index interpolation would bunch the derived centerline toward the denser boundary and bias every
downstream heading. Covered by `test_midpoint_line_handles_mismatched_boundary_lengths`,
`test_naive_elementwise_mean_would_have_failed`, `test_interp_arc_is_uniform_in_arclength_not_index`.

### 2.3 The third trap — and a correction it forces on the S1 numbers

**8.4 % of successor ids point outside the local map** (16 141 / 191 770 — the log-local crop
boundary, not corruption). Traversal treats a dangling successor as **terminal**, never as an error.

⚠️ **This is not only a traversal concern — it changes a headline count.** Branch points computed on
**raw** successors: **22 606**. Computed on **resolved** successors: **20 591**.

> **2 015 apparent branch points (8.9 %) are "branches" only because one of their options leaves the
> local crop.** An option we cannot represent cannot be a training label. Any sizing of the S1
> decision set must use **20 591**, not 22 606.

The adapter defaults to resolved and exposes the raw count explicitly. `LaneGraph.routes_from()` is
also cycle-safe — necessary, since **36.8 %** of real logs contain a directed cycle; a naive DFS
hangs on them.

### 2.4 An independent cross-check, not a re-report

The puller (`av2_pull_sensor_lane_graphs.py`, plain-dict parsing) and the adapter
(`argoverse2.py`, dataclass graph) are **separate implementations**. Run over the same 1 000 files
they agree exactly on lane segments (163 698), raw successor edges (191 770), dangling refs
(16 141), `is_intersection` (57 415), neighbours (104 924 / 44 788), maps-with-a-branch (998),
maps-with-an-intersection (997), and explicit centerlines (**0**). The only divergence is the
branch count, and it is the *intended* raw-vs-resolved difference documented above. This is why the
headline is tiered **DECISION-GRADE** rather than CONFIRMED.

### 2.5 What the adapter reuses vs adds

Reused **verbatim by import, not copied**: `quat_to_rotmat`, `quat_to_yaw`, `wrap_pi` from
`nuscenes.py` (both corpora use `[w,x,y,z]` in a right-handed frame — divergent copies of a
quaternion convention is exactly how a silent sign error enters a pose pipeline;
`test_quaternion_helpers_are_the_same_objects_as_nuscenes` pins this). Also reused: the ego-track
contract, the pose-derived action formulation, `finite_diff_accel`, and the split-unit discipline.

New: `LaneGraph` / `LaneSegment` / `midpoint_line` / `interp_arc` / `routes_from` / `branch_options`,
the feather readers (`city_SE3_egovehicle.feather`, `calibration/intrinsics.feather`), and discovery.

Two genuine improvements over the nuScenes path:
- **AV2 ships radial distortion coefficients** (`k1,k2,k3`), so `PinholeIntrinsics.dist` is real and
  `pinhole_rectify` can actually undistort, instead of degrading to its pad-crop half as it must for
  nuScenes' bare 3×3.
- **20 Hz** rather than 2 Hz keyframes — the keyframe-interpolation hack is not needed at all.

### 2.6 ⚠️ The terms-gate guard was deliberately NOT copied

`nuscenes.py` raises `NuScenesTermsError` because a human must register and accept before any byte
is served. **AV2 has no access-control gate.** Copying that guard would be a lie in the code, so
`Argoverse2MapError` instead says *how to fetch the bytes anonymously*. Two tests pin this:
`test_missing_file_error_does_not_invent_a_terms_gate` (the message must contain `--no-sign-request`
and must **not** contain "sign-up"/"register") and `test_adapter_exposes_no_terms_error_symbol`.

### 2.7 A footgun the tests caught in my own first draft

`discover_logs(root, split)` originally fell back to `root` when the split directory was missing.
The test suite caught it returning `['train','val','test']` **as if the three splits were three
logs** — a silent corpus-wide mis-ingest. It now raises and names the available splits.
Regression-pinned by `test_discover_logs_missing_split_fails_loud_not_silently`.

### 2.8 `SOURCE_REGISTRY` — re-verified against the licence DOCUMENT

Re-fetched `argoverse.org/about.html` myself (57 615 B, HTTP 200, saved). The existing entry
`"argoverse2": SourceLicense("nc-research", "CC-BY-NC-SA-4.0", share_alike=True, is_synthetic=False)`
is **CORRECT on every field. No change proposed.** Verbatim support in
`evidence/license_determinations_2026-07-26.json`.

**Routing and refusal now covered BY NAME**, not by inference from nuScenes' row — four new tests in
`stack/tests/test_lake.py`: `test_argoverse2_registered_correctly_against_the_licence_DOCUMENT`,
`test_argoverse2_routes_to_segregated_copyleft_shard` (→ `shards/nc-research/sharealike/argoverse2/`),
`test_argoverse2_record_refused_from_commercial_tier_C`, `test_argoverse2_tier_is_nc`.
*(C-tier refusal was already covered generically by `test_every_nc_research_source_refused_from_commercial_tier_C`, which iterates all `nc-research` sources; the new test pins AV2 by name.)*

⚠️ **The entry is a FLOOR.** The Terms of Use add obligations no CC short name carries — and
**Prohibited-Use Examples 3 and 4 name training a model for a product explicitly**. An AV2-trained
model cannot ship regardless of which lake tier the records land in. That is stronger than a bare
"NC" reading and is quoted in full in the evidence JSON.

---

## 3. Known limitations of this pull — stated, not buried

1. **Sensor split only.** The 249 880 motion-forecasting scenario maps (~25 GiB) were not pulled.
   They are the larger junction-topology bank but carry **no imagery**.
2. **Our flat file naming drops the city token.** The puller stores archives as `<log_id>.json`, so
   `LaneGraph.city` is empty when loading from our pull; the city is preserved in
   `av2_pull_manifest_1000.json` (parsed from the S3 key). The adapter parses city correctly from
   the *original* AV2 filename — `test_load_lane_graph_round_trip` covers that path.
3. **The raw 154 MiB of map JSON is NOT staged.** AV2 is `CC-BY-NC-SA-4.0`; committing the archives
   would be redistribution of source bytes, which our policy forbids (*pointers + derived features,
   never source bytes*). Staged artifacts carry S3 keys + derived statistics only, and the puller
   reproduces the corpus exactly in ~2 minutes.
4. **No pose/calibration data is on disk.** Those live in the 1 051 GiB tars. The feather readers
   are tested against synthetic tables of the documented schema, **not** against real AV2 feather
   bytes — marked UNVERIFIED against real bytes.
5. **`is_intersection` 35.1 %** here vs 33.8 % / 36.3 % in the earlier samples — consistent, and the
   full-population figure supersedes both.

---

## 4. TASK 2 — ZOD: the exact requirements, and the exact human step

### 4a. 🔴 The finding that changes the plan: **ZOD has no lane graph**

**Class: MEASURED/PUBLISHED at THREE independent probes. Tier: CONFIRMED.**

| # | probe | result |
|---|---|---|
| 1 | licensor's own `/annotations/` page (42 388 B, saved) | lane annotations are **lane markings** (`lm_solid`, `lm_dashed`, `lm_botts_dot`, `lm_shaded`) + **road paintings** (arrow/pictogram/text/crosswalk). Stated to be *"an image-level annotation, i.e., it only conveys information in 2D"* |
| 2 | the MIT devkit's own source, `zod/anno/lane.py` | `LaneMarkingAnnotation` carries only `uuid`, `geometry` (2-D polygon), `type`, `colored`, `instance_id`, `cardinality`. **No successor, predecessor, neighbour or topology field on any class.** The devkit's annotation modules are `ego_road.py`, `lane.py`, `object.py`, `road_condition.py`, `tsr/` — **no map or graph module at all** |
| 3 | the ICCV-2023 paper / arXiv 2305.02008 | annotations listed as 2D/3D objects, road segmentation, traffic-sign recognition, road classification. **No HD map, lane graph or topology** |

> **A lane polygon set with no successor relation is not a routable graph — and ZOD does not even
> have lane polygons, it has lane *marking* polylines in image space.** ZOD therefore **fails P2**.

**Consequence, stated plainly as the brief asked:** ZOD's publishability advantage **does not buy
what we need**. Acquiring ZOD would give a commercially-usable *driving corpus*; it would **not**
give a commercially-usable *lane graph*. The ZOD-as-publishable-twin plan, as written, does not work.

**It is not worthless** — see §5c, where ZOD becomes useful again in a different role.

### 4b. Licence — verified as a DOCUMENT, and it carries terms the short name does not

Fetched and saved `zod.zenseact.com/license/` (7 376 B, HTTP 200) plus the README.

- ✅ **`CC-BY-SA-4.0` CONFIRMED. It is NOT non-commercial** — no NonCommercial term exists in
  CC BY-SA. The licensor states it directly: *"ZOD is currently the only AD dataset released under
  the permissive `CC BY-SA 4.0` license, allowing for both commercial and non-commercial use."*
- ✅ Devkit is **MIT**, separate from the data licence.
- ✅ Registry entry `"zod": SourceLicense("owned-safe", "CC-BY-SA-4.0", share_alike=True, …)` →
  tier `ship-sa`. **CORRECT, unchanged.** The 2026-07-13 correction held.
- ⚠️ **Version subtlety worth recording:** the `/license/` *prose* says only **"CC BY-SA"** with **no
  version number**. The `4.0` comes from the hyperlink target
  (`creativecommons.org/licenses/by-sa/4.0/`) and the README. Reading the prose alone leaves the
  version unresolved.
- ⚠️ **Two obligations the short name does NOT carry:**
  - a **mandatory notice** that must be reproduced on *"any public use, distribution, or display"* —
    the PII-removal statement quoted verbatim in the evidence JSON. This binds any paper or artifact
    we publish using ZOD.
  - a **field-of-use statement**: *"ZOD is not intended for military use."*

### 4c. The access requirement, and **exactly what the PI must click**

The licensor's own `/download/` page and README agree (2 independent paths):
*"Prerequisites are that you have applied for access and received a download link."*
**The download URL is the credential** — `zod download --url="<download-link>"`.

**There is no web form, no account to create, and no Terms checkbox. It is a single email.**

> #### The complete human step — one email, ~4 lines
>
> **To:** `opendataset@zenseact.com`
> **Body** (the licensor's own template):
> ```
> Dear Zenseact,
> I am interested in using the Zenseact Open Dataset (ZOD) for my research and I
> would like to request access to the dataset.
> Please find the requested information below:
> Name:         <name>
> Affiliation:  <affiliation>
> Email:        <email address connected to a Dropbox account>
> Intended use: <short description>
> Best regards,
> ```

| question the brief asked | answer |
|---|---|
| Account needed? | **No.** No registration, no login, no Terms checkbox. |
| **Institutional address needed?** | **Not stated as mandatory.** The page asks for "your affiliation" and does not say it must be institutional. **UNVERIFIED** whether a non-institutional affiliation is accepted — the licensor does not say, and I did not ask on the PI's behalf. |
| What *is* strictly required | name · affiliation · **an email address connected to a Dropbox account** · a short intended-use description. The Dropbox linkage is the one non-obvious hard requirement. |
| Expected turnaround | **Not published.** The page says only *"we will review it and get back to you as soon as possible."* No SLA. **Do not plan against a date.** |
| Reviewed by a human? | Yes — *"we will review it"*. |

#### ⚠️ **Does the PI need to be at a computer? — Answer: NO, and that is the operationally useful part**

**The application is an email.** It can be sent from a phone. It does not require his desktop, a
browser session, an account, or a Terms click. This is materially **smaller than the nuScenes ask**
(which needs registration + an explicit Terms acceptance in a browser).

**What still needs a machine, eventually:** a Dropbox-linked email address must exist, and the actual
download (`pip install zod[cli]; zod download --url=…`) needs a machine with disk — but **that can be
a pod or this dev box, and it does not have to be his.** Only the *reply with the link* has to reach him.

**Can anything proceed meanwhile? Yes:** §5 is entirely credential-free and does not wait on ZOD, and
AV2 is already fully in hand.

⛔ **What I did NOT do, and will not do:** I did not create an account, register, submit any form,
send any email, or accept any Terms on the PI's behalf. Those are hard boundaries regardless of
approval. The email above is drafted **for him to send**, not sent.

### 4d. 🟡 **A PI decision I am flagging, not taking: ZOD's second official channel**

The licensor's own download page states:

> *"We are now also hosting the dataset via **Academic Torrents**. You can find the Frames,
> Sequences and Drives subsets there."*

This is **the licensor publishing a second channel themselves** — not a third-party mirror, not a
scraped copy. If that channel is real, ZOD's bytes are obtainable **with no human application at all**.

**I did not touch it, probe it, or download from it.** Reason: the brief's hard constraint says
*"no torrents"* flatly. Its stated *rationale* is *"never route around a licensor's access gate"* —
which a licensor-operated channel does not do. **Letter and rationale diverge here, so the call is
the PI's, not mine.**

Two caveats if he considers it: (i) I could not verify the Academic Torrents listing itself — the
site returned a bot-check page, so *"Zenseact published these torrents"* is **PUBLISHED (licensor's
own statement), not MEASURED**; (ii) it changes nothing about §4a — ZOD still has no lane graph.

---

## 5. TASK 3 — new alternatives. **One real find.**

Probes: `evidence/access_probes_2026-07-26.json` (17 probes, **≥2 per candidate**).
Licences: `evidence/license_determinations_2026-07-26.json`.

### 5a. 🟢 **Overture Maps — transportation theme.** The scarce combination, found.

**Access — MEASURED, tier CONFIRMED (two independent endpoints):**
```
https://overturemaps-us-west-2.s3.amazonaws.com?list-type=2&prefix=release/   -> HTTP 200
https://s3.us-west-2.amazonaws.com/overturemaps-us-west-2?...                 -> HTTP 200
releases visible: 2026-06-17.0, 2026-07-22.0
```
No account, no token, no Terms click, **no mirror** — an official foundation-operated bucket, also
listed on the AWS Registry of Open Data.

**Connectivity — MEASURED and byte-verified, not "the docs say routable":**
read via parquet **HTTP-range** requests — **3.3 MB fetched in 7 requests**, not the ~14 GB theme
(`evidence/overture_connectivity_probe.py` / `.json`).

| | |
|---|---|
| segment schema | 21 columns incl. **`connectors: list<struct<connector_id: string, at: double>>`** |
| rows in the probed file | 2 732 314 segments (one of 5+ parts) |
| connector features | **13 044 867** |
| sampled rows | 20 000 |
| **rows with ≥1 connector** | **20 000 / 20 000 = 100 %** |
| **rows with ≥2 connectors** | **20 000 / 20 000 = 100 %** |
| total connector refs / max on one segment | 48 913 / **29** |
| sampled classes | residential 7 330 · service 3 793 · track 3 487 · tertiary 1 335 · secondary 1 098 · primary 973 · … |

**Three things it has that AV2 does NOT** (all present in the verified schema):
- **`prohibited_transitions`** — legal turn restrictions, as `sequence: list<struct<connector_id,
  segment_id>>`. This is *"you may not turn left here"* as data.
- **`destinations`** — **signposted destinations** with `from_connector_id` / `to_segment_id` /
  `final_heading`. This is the closest thing to a **route/goal signal** anything in our corpus set has.
  Recall the settled finding that PhysicalAI-AV has *no* route/goal signal at all.
- **`speed_limits`**, **`access_restrictions`**, **`routes`**.

**Licence — read as a document, and it contains a trap:**
- Transportation theme is **`ODbL-1.0`**, because it is OSM-derived. **Commercially usable
  (no NC term), but share-alike.**
- ⚠️ **THE TRAP:** Overture is widely described as **CDLA-Permissive-2.0**. That is true of the
  *foundation's default* and **false of the transportation theme**. Taking the headline short name
  would have recorded a permissive non-copyleft licence where the real one is **copyleft** — the
  same root-cause class as the ZOD and nuScenes short-name errors, caught here **before** it entered
  the registry.
- Required attribution: *"© OpenStreetMap contributors. Available under the Open Database License"*.

**Honest limitations — this is a road graph, not a lane graph:**
- **No lane-level geometry and no `lanes` column** in the 21-column segment schema (MEASURED).
  Segments are ways, connectors are junctions. **It does not substitute for AV2 on S2 (lane
  selection).** For **S1** and **HP-4** it is arguably better: global scale, plus turn restrictions.
- **No imagery.** It is a map, not a driving corpus — it must be **paired** with a corpus carrying
  geo-coordinates (§5c).
- ⚖️ **OPEN LEGAL QUESTION, not a determination:** ODbL share-alike attaches to the *database* and
  to *Derivative Databases*. Whether a **model trained on it** is a Derivative Database or a Produced
  Work is a real legal question **I am not competent to settle**. Flagged for the PI.

**Proposed registry entry — written, NOT applied:**
```python
"overture_transportation": SourceLicense("owned-safe", "ODbL-1.0",
                                         share_alike=True, is_synthetic=False)
#   -> derived tier `ship-sa`, same segregated copyleft shard class as ZOD
```
Not added, because nothing consumes it yet and an unused registry entry is the exact
*"sitting there correct and unused"* state the previous report flagged. Without an entry an ingest
attempt fails loudly, which is the safe default. **Escalated in §7, not written into a README.**

### 5b. Other candidates probed

| corpus | access (≥2 probes) | licence as document | lane connectivity | verdict |
|---|---|---|---|---|
| **MetaDrive** | ✅ none (pip). **Not installed here → nothing byte-verified** | **Apache-2.0** (file fetched) — commercial-OK **and not share-alike**: the *most permissive licence in the sweep* → tier `ship` | ⚠️ **road-level, not lane-level.** `node_road_network.py`: `graph[start_node][end_node] = [lanes]` — successors between **nodes**, lanes merely grouped in a list | 🟡 **PROVISIONAL.** Real value for **HP-4** (procedural junction topologies, commercially clean). Prior report called it HYPOTHESIS; licence now PUBLISHED, connectivity still unverified |
| **OpenStreetMap** (direct / Geofabrik) | ✅ ungated, HTTP 200 both paths | **ODbL-1.0** | road-level | ⚪ **superseded by Overture**, which is the same graph better normalised (turn restrictions, destinations) |
| **levelXdata** (highD/inD/rounD/exiD) | ❌ **human form, manually reviewed** — *"each request is checked manually"*; requires full name, official address, org type | **non-commercial**: *"you may not use the highD dataset or any derivative work for commercial purposes"*; commercial needs a paid licence | page does not even mention a lanelet2/OpenDRIVE map | ❌ **REJECT** — fails P1 **and** fails the publishability goal |
| **PandaSet** | HF `georghess/pandaset`: `gated:false`, `license:cc-by-4.0` (HTTP 200). Original `scale.com` channel **404 — gone** | `CC-BY-4.0` (already in registry, tier `ship`) | ❌ no HD map (INHERITED, not re-probed) | ⚪ **not a lane-graph candidate.** ⚖️ *Mirror analysis, because it is easy to get backwards:* the HF copy is third-party, **but CC-BY-4.0 grants redistribution explicitly**, so it is a permitted redistribution — **not** the forbidden mirror-of-a-gated-corpus case, which is defined by terms that *forbid* redistribution (nuScenes imagery in OpenLane-V2 `subset_A`) |
| **INTERACTION** | ⚠️ 2 probes, **both HTTP 200 but unreadable** (JS app, static text = page title only). Devkit is BSD-3 (**code**, not data) | not established | lanelet2 maps reported, **not confirmed by me** | ⚠️ **UNVERIFIED — left open, decides nothing.** Two probes returned no readable content, and per the standing rule an unread page is not evidence. **I did not write it up as "gated."** |

### 5c. 🟢 The composition this opens — and where ZOD becomes useful again

ZOD has no lane graph (§4a) — but **ZOD's devkit `EgoMotion` carries `origin_lat_lon: Tuple[float,
float]`** (PUBLISHED, read from devkit source) alongside local poses. That means ZOD traces can in
principle be **georeferenced**, and therefore **map-matched onto Overture/OSM**.

> **The publishable twin may be reachable after all — not as "ZOD's lane graph" (there is none), but
> as ZOD imagery + Overture's routable graph, both commercially usable.**
> ZOD `CC-BY-SA-4.0` + Overture `ODbL-1.0` — both `owned-safe`, both share-alike → **both already
> route to the same `ship-sa` segregated copyleft shard our SA plumbing handles today.**

**Marked HYPOTHESIS, deliberately:**
- the local-frame convention (ENU alignment relative to `origin_lat_lon`) is **UNVERIFIED**;
- a single origin per sequence + local poses gives *derived* global coordinates, whose accuracy for
  map-matching is unmeasured;
- **it still requires the ZOD access step** (§4c).

**A cheaper discriminating test that needs nobody:** `comma2k19` is already in the lake as
`owned-safe / MIT / tier ship` — **the most permissive corpus we have** — and it carries **real
GNSS**. Map-matching comma2k19 onto Overture would test the whole composition **with no application,
no gate, and no PI action at all**, and if it works the result is `ship`-tier, not even `ship-sa`.
**Recommended as the next cheap discriminating experiment.** Not run here (out of scope).

---

## 6. Claims ledger — evidence class **and tier** on everything load-bearing

| claim | class | tier | where |
|---|---|---|---|
| AV2 pull yield **1000/1000**, size+MD5+parse+semantic verified | MEASURED | **DECISION-GRADE** | `av2_pull_summary_1000.json`, `av2_pull_manifest_1000.json` |
| 161 255 215 B / 153.8 MiB, 115.1 s | MEASURED | DECISION-GRADE | same |
| 163 698 lane segments · 191 770 raw / 175 629 resolved edges · 57 415 intersection · 104 924/44 788 neighbours | MEASURED | **DECISION-GRADE** (2 independent implementations agree) | `av2_pull_summary_1000.json` + `av2_adapter_corpus_stats_1000.json` |
| **0 / 163 698 sensor segments have `centerline`** | MEASURED | **DECISION-GRADE** (full population, 2 implementations) | same + `test_real_av2_sensor_split_has_zero_explicit_centerlines` |
| Branch points **20 591 resolved** vs 22 606 raw (8.9 % overstatement) | MEASURED | DECISION-GRADE | `av2_adapter_corpus_stats_1000.json` |
| 36.8 % of maps contain a directed cycle | MEASURED | CONFIRMED (36 % at n=25 → 36.8 % at n=1000) | same |
| Left/right boundary lengths differ on 49.0 % (n=19 713) | MEASURED | CONFIRMED | §2.2, reproducible from the pull |
| AV2 licence `CC-BY-NC-SA-4.0` / code MIT; registry entry correct | PUBLISHED | **CONFIRMED** (2 independent fetches of the licensor doc) | `argoverse_about_2026-07-26.html`, `license_determinations_2026-07-26.json` |
| AV2 Terms prohibit training a model for a product (Examples 3, 4) | PUBLISHED | CONFIRMED | same |
| Adapter 46/46 green incl. 4 real-corpus tests; full suite 1099 passed / 7 skipped | MEASURED | DECISION-GRADE | `stack/tests/test_argoverse2.py` |
| **ZOD has NO lane connectivity** | MEASURED (devkit source) + PUBLISHED (annotations page, paper) | **CONFIRMED — 3 independent probes** | §4a, `zod_annotations_2026-07-26.html` |
| ZOD is `CC-BY-SA-4.0`, **commercial-OK**, not NC; devkit MIT | PUBLISHED | **CONFIRMED** (licensor page + README) | `zod_license_2026-07-26.html` |
| ZOD carries extra terms (mandatory PII notice; "not intended for military use") | PUBLISHED | CONFIRMED | same |
| ZOD access = one email to `opendataset@zenseact.com`; no account, no form, no Terms click | PUBLISHED | **CONFIRMED** (2 licensor paths) | `zod_download_2026-07-26.html` + README |
| ZOD turnaround time | **NOT PUBLISHED** | — | no SLA stated; do not plan against a date |
| ZOD accepts a non-institutional affiliation | **UNVERIFIED** | — | licensor does not say; I did not ask on the PI's behalf |
| Zenseact also publishes ZOD on Academic Torrents | PUBLISHED (licensor's own statement) | **PROVISIONAL** — the AT listing itself was **not** verified (bot-check) | §4d |
| ZOD `EgoMotion.origin_lat_lon` exists → georeferencing possible in principle | PUBLISHED (devkit source) | PROVISIONAL | §5c |
| Overture anonymous access | MEASURED | **CONFIRMED** (2 independent endpoints) | `access_probes_2026-07-26.json` |
| Overture connectivity: 20 000/20 000 segments ≥2 connectors; `prohibited_transitions`, `destinations` in schema | MEASURED | **CONFIRMED** (schema + 20 000 real rows) | `overture_connectivity_probe.json` |
| Overture transportation is **ODbL-1.0**, not CDLA-Permissive | PUBLISHED | PROVISIONAL — read via `docs.overturemaps.org`; the ODbL legal text was not separately fetched | `license_determinations_2026-07-26.json` |
| Overture has **no** lane-level geometry / no `lanes` column | MEASURED (21-column schema) | CONFIRMED | `overture_connectivity_probe.json` |
| Whether an ODbL-trained model is a Derivative Database | **OPEN LEGAL QUESTION** | — | not for an agent to settle |
| MetaDrive is Apache-2.0 | PUBLISHED | CONFIRMED | licence file fetched |
| MetaDrive connectivity is road-level, not lane-level | PUBLISHED (source read) | **PROVISIONAL** — not installed, nothing byte-verified | §5b |
| levelXdata: manual form + non-commercial | PUBLISHED | PROVISIONAL (1 page read in full) | §5b |
| INTERACTION access mechanism | **UNVERIFIED** | — | 2 probes, both unreadable. Not claimed as gated |
| PandaSet has no HD map | INHERITED | — | not re-probed; decides nothing |
| comma2k19 + Overture as the cheap discriminating test | HYPOTHESIS | — | §5c |

---

## 7. 🔴 ESCALATIONS — integration decisions, in the headline, not in a README

1. **`argoverse2.py` is new code that nothing calls yet.** The adapter and its 46 tests are staged,
   but no ingest driver reads AV2. The nuScenes path has `stack/scripts/ingest_nuscenes.py`; **AV2
   has no sibling driver.** That is the next integration step and it is small. *(This report is the
   request — the previous cycle's identical request sat unread for 10 days.)*
2. **The ZOD plan of record is wrong and needs re-deciding.** ZOD was designated the publishable
   twin because it is commercially usable. **It has no lane graph** (§4a, 3 probes). The PI should
   decide whether ZOD is still worth the application **for imagery/geo alone** (§5c) or dropped.
3. **Overture is the actual answer to the publishability question — and needs a PI decision**, on
   (a) adding the proposed `SOURCE_REGISTRY` entry, and (b) the ODbL Derivative-Database question,
   which is a legal call and not mine.
4. **The ZOD torrent question is the PI's** (§4d): the brief forbids torrents flatly; the channel is
   the licensor's own. I did not act. If he says yes, ZOD needs **no** human application at all.
5. **Recommended next cheap experiment, needing nobody:** map-match **comma2k19** (`ship` tier, real
   GNSS, already in the lake) onto **Overture**. It tests the whole composition with no gate and no
   PI action, and a success is `ship`-tier — better than anything ZOD could produce.

---

## 8. Deliverable manifest

**Everything is in the repo and `git add`-staged. Nothing lives only in a scratchpad, only on a pod,
or only in this agent's context. No pod was touched. Nothing was committed or pushed.**

| # | artifact | where it lives | only one place? |
|---|---|---|---|
| 1 | `AV2_ZOD_INGEST.md` — this report | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-26-av2-zod-ingest/` | no — staged |
| 2 | **`argoverse2.py` — the adapter** | `repo:stack/tanitad/data/argoverse2.py` | no — staged |
| 3 | **`test_argoverse2.py` — 46 tests** (42 fixture + 4 real-corpus) | `repo:stack/tests/test_argoverse2.py` | no — staged |
| 4 | 4 new AV2 registry/routing/refusal tests | `repo:stack/tests/test_lake.py` (edited) | no — staged |
| 5 | `av2_pull_sensor_lane_graphs.py` — the puller (rerunnable, anonymous, verifies size+MD5+parse) | `…/2026-07-26-av2-zod-ingest/evidence/` | no — staged |
| 6 | `av2_pull_summary_1000.json` — the verified yield | same | no — staged |
| 7 | `av2_pull_manifest_1000.json` — per-file record for all 1 000 (key, size, ETag, gates, counts) | same | no — staged |
| 8 | `av2_adapter_corpus_stats_1000.json` — adapter's independent aggregate + per-map stats | same | no — staged |
| 9 | `overture_connectivity_probe.py` / `.json` — byte-verified routable graph via HTTP range | same | no — staged |
| 10 | `access_probes_2026-07-26.py` / `.json` — 17 probes, ≥2 per candidate | same | no — staged |
| 11 | `license_determinations_2026-07-26.json` — every licence as a **document**, verbatim, data-vs-code split, FLOOR markers | same | no — staged |
| 12 | `argoverse_about_2026-07-26.html` — licensor's ToU/licence page as fetched (57 615 B) | same | no — staged |
| 13 | `zod_{license,download,annotations,frames,sequences,drives}_2026-07-26.html` + headers — ZOD licensor pages as fetched | same | no — staged |

**Deliberately NOT staged — and why:**

- **The 1 000 raw AV2 map JSONs (153.8 MiB).** AV2 is `CC-BY-NC-SA-4.0`; committing them is
  redistribution of source bytes, which our policy forbids (*pointers + derived features, never
  source bytes*). Staged artifacts carry **S3 keys + derived statistics only** and are fully
  reproducible by rerunning the staged puller (~2 min).
  *Scratchpad location, this session only:* `…/scratchpad/av2_sensor_lane_graphs/{train,val,test}/*.json`.
  ⚠️ **This is the one thing that exists in only one place** — but it is regenerable-by-script from a
  public bucket, so it is not stranded work.
- **No `SOURCE_REGISTRY` change was made.** The AV2 entry needed none; the Overture entry is
  *proposed* (§5a) and escalated rather than applied.
- **No PhysicalAI-AV content** of any kind — no clip UUIDs, no raw content.
- **`Keys.txt` was never read, printed, copied, or passed in argv.** Every probe in this report is
  anonymous; no credential was needed anywhere.
- **Nothing gated was routed around.** No third-party mirror of a gated corpus, no scraped copy, no
  torrent. No account created, no form submitted, no Terms accepted, no email sent.
