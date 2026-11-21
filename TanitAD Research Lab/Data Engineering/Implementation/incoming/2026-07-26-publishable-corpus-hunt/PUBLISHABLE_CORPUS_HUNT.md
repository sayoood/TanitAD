# The publishable corpus hunt — commercial × credential-free × lane connectivity

**Date:** 2026-07-26 · **Agent:** data-engineering / publishable-corpus-hunt
**Continues:** `…/2026-07-26-credential-free-lanegraph/LANEGRAPH_ALTERNATIVES.md` and
`…/2026-07-26-av2-zod-ingest/AV2_ZOD_INGEST.md`
**Compute:** dev box only, CPU + network. **No pod was touched** (pod1 training, pod2 H2 classifier,
eval pod Bar-A).
**PI instruction honoured verbatim:** *"look for alternatives and skip Zod until im at home."*
**ZOD was not probed at all in this cycle** — no email, no application, no torrent, no fetch. §7.

---

## 0. HEADLINE — with its tier

> ### 🟢 The three-way combination EXISTS, and we can already reach it.
> **DLR's ASAM OpenDRIVE HD maps on Zenodo are `CC-BY-4.0` (commercial-OK, *not* share-alike),
> anonymous, and carry byte-verified LANE-LEVEL connectivity.**
> Five maps pulled with `curl` and nothing else: **2 921 roads · 343 junctions · 139.96 km ·
> 24 536 driving lanes · 86 200 explicit `<lane><link><successor>` elements · 4 837 junction
> `<laneLink from→to>` lane-to-lane turn edges · 583 branch points** (out-degree up to 5).
> **Tier: DECISION-GRADE** — whole-population parse of every downloaded file, licence read as the
> shipped legal document, access proven by anonymous HTTP 200.

**And it closes AV2's one real gap.** AV2 has no stop-lines, no traffic lights, no yield markings.
These maps carry **372 positioned traffic lights** (`dynamic="yes"`, `name="TrafficLight"`),
**160 YIELD signs** (StVO 205), **23 STOP signs** (StVO 206) and **73 speed-limit signs** (StVO 274),
each with `s/t/zOffset/orientation` geometry. MEASURED, §4.3.

Three findings that change what we should do, in descending order of consequence:

1. 🟢 **A `ship`-tier lane graph exists.** `CC-BY-4.0` is strictly better than everything else in the
   program's map inventory: better than AV2 (`CC-BY-NC-SA`, non-commercial **and** copyleft) and
   better than Overture (`ODbL`, copyleft with an unresolved Derivative-Database question). This is
   the **first** map asset that would route to tier **`ship`**, not `ship-sa`, not `nc-research`. §4.
2. 🔴 **Overture's ceiling is road-level. Settled at four independent probes — stop asking.**
   The authoritative JSON-Schema has **no lane property**, the transportation theme contains **only**
   `segment` and `connector`, the repo has **no lane entity in any of its 6 themes**, and a *different*
   parquet part from a *newer* release than the last agent read has **21 columns, zero of them lane
   anything, and zero nested types mentioning lane**. §3.
3. ⚠️ **The scarce thing is now IMAGERY, not the lane graph.** Every candidate that satisfies all
   three criteria is a **map with no camera data**. No corpus surveyed is simultaneously commercial,
   credential-free, lane-connected **and** carries imagery. That is the honest residual gap and it is
   stated as a clean negative, not stretched. §6.

⚠️ **One per-record trap that would have cost us the whole finding if taken on the publisher's name:**
the *same author*, for the *same test bed*, publishes the **412 KB sample under `CC-BY-4.0`** and the
**81 MB full A2/A391/A39/L295 map under `CC-BY-NC-SA-4.0`**. Licence is **per record DOI**, never per
publisher. §4.4.

---

## 1. Decision rule — adopted before probing, unchanged from the standing pre-registration

I adopted P1–P3 verbatim from `LANEGRAPH_ALTERNATIVES.md` §0 so the results stay comparable across
the three cycles, and added **P4**, which is the whole point of this brief.

| # | criterion | satisfied by | NOT satisfied by |
|---|---|---|---|
| **P1** | **Credential-free** | an anonymous, unauthenticated request from this dev box returns HTTP 200 for **both** a listing and a real data object | a login, token, accepted EULA click, email-gated link, "request access" form, **or a third-party mirror of a gated corpus** |
| **P2** | **Routable lane graph** | lane centrelines **plus an explicit successor/predecessor (or equivalent edge) relation**, byte-verified in a real downloaded file | a lane *polygon*, a marking polyline, a drivable-area raster, a BEV mask, or "has HD map" in a paper |
| **P3** | **Licence read as a document** | the licensor's own licence text fetched and quoted, **data licence separated from code/devkit licence** | a short name from a README, an API metadata tag, an HF badge, or recall |
| **P4** | **Commercially usable** | the licence document contains **no NonCommercial term**; share-alike recorded separately because it is a *different* obstacle | an NC term; "research only"; a field-of-use restriction naming products |

**Pre-registered outcomes, both committed in advance:**
**Outcome A** — something satisfies P1+P2+P3+P4 → name it, publish the raw proof, give the registry
entry and the recommendation. **Outcome B** — nothing does → say so plainly, name the failing
criterion per candidate, name the closest miss, and stop.
**Result: Outcome A**, with an explicit rider that the winner is a *map* and not a *driving corpus* (§6).

**Failure modes explicitly guarded against, each of which has cost this program before:**
- *Licence-from-short-name* (made 3× already: ZOD, nuScenes, and nearly Overture-as-CDLA). Mitigation:
  the shipped legal text was fetched and machine-checked for the strings `NonCommercial` and
  `ShareAlike`, not eyeballed. §4.4.
- *Absence-at-one-location.* Every "gated" / "no lane graph" claim below rests on **≥2 probes**.
- *Devkit ≠ data.* Two separate columns, always.
- *The ZOD lesson — a rejected candidate carries no verified attributes, only a verified defect.*
  Nothing here is promoted on its licence alone; **every** PASS row has byte-verified connectivity.

---

## 2. RANKED CANDIDATE TABLE — the three-way combination

Legend: ✅ satisfied · ❌ failed · ⚠️ satisfied with a stated caveat.
"Access proof" is what *I* measured this cycle unless marked INHERITED.

| # | candidate | P1 credential-free (access proof) | P4 commercial (licence **as a document**) | P2 lane connectivity (byte-verified) | verdict |
|---|---|---|---|---|---|
| **1** | **DLR OpenDRIVE HD maps** — Schwarzer Berg (Brunswick), ViVre (Brunswick), Brunswick inner ring road, 5G Living Lab (Wolfsburg), Test Bed Lower Saxony *sample* | ✅ **Zenodo, anonymous.** Record API **200**; object GETs **200**: 26 807 654 · 5 240 081 · 3 893 972 · 3 667 360 · 412 133 B. No account, no click, no mirror | ✅ **`CC-BY-4.0`.** The **full CC BY 4.0 legal text ships inside every record** as `LICENSE.txt` (18 655–18 657 B). Machine-checked: `NonCommercial` **absent**, `ShareAlike` **absent** in all 4 zips. README verbatim: *"licenced under the terms of Creative Commons Attribution 4.0 International"*. **Not share-alike → tier `ship`** | ✅ **YES.** 86 200 `<lane><link><successor>` elements; **4 837 `<laneLink from→to>`** turn edges inside `<junction><connection>`; **583 branch points**, out-degree hist `{1:3577, 2:504, 3:69, 4:8, 5:2}`; 343 junctions | ✅ **PASS — all four. RECOMMENDED** |
| **2** | **CARLA** `TownBig.xodr` (synthetic) | ✅ raw.githubusercontent **200**, 15 536 627 B, in-repo, anonymous | ⚠️ commercial-OK either way, but **ambiguous which licence covers the file**: root `LICENSE` is **MIT**; README says *"CARLA specific assets are distributed under the CC-BY License"* — **with no version number**. Neither reading is NC or SA | ✅ **YES.** 1 923 `<road>`, 192 `<junction>`, **2 416 `<laneLink>`** | ⚠️ **PASS on all four, but SYNTHETIC.** A procedurally-authored town, not surveyed road. Useful for HP-4 topology, worthless as evidence about real junction distributions |
| **3** | **esmini** `resources/xodr/` + `Unittest/xodr/` (~82 maps) | ✅ in-repo, anonymous, GitHub tree **200** | ⚠️ **MPL-2.0** (16 726 B, fetched). Commercial-OK; file-level weak copyleft → conservatively `ship-sa` | ⚠️ OpenDRIVE, so the relation exists by format — **but I did not byte-verify these specific files**, and most are 20 m–3 km synthetic test roads (`straight_20m`, `highway_split`, `fabriksgatan`) | 🟡 **PROVISIONAL.** Tiny. Real value only as unit-test fixtures for an OpenDRIVE adapter |
| **4** | **CommonRoad scenarios** (TUM) | ✅ public GitLab, anonymous API **200**; 12 scenario XMLs fetched **200** | ⚠️ **BSD-3-Clause** is the repo's only licence (1 517 B, fetched verbatim, © 2019 TU München). **But** it says *"SOFTWARE"* throughout on a data-only repo, GitLab's own licence field is `null`, and **the only sub-collection with a dense lane graph is plausibly OSM-derived → would inherit ODbL**. Unresolved | ⚠️ **UNEVEN.** `recorded/hand-crafted`: **0 successors in 5 of 6** sampled files (2–4 lanelets each). `scenario-factory`: **116 `<successor>` + 116 `<predecessor>` + 94 `<adjacentLeft>` + 8 `<intersection>` per file** | 🟡 **PROVISIONAL — do not promote on the BSD file.** This is the OpenLane-V2 pattern: a permissive repo badge over data whose upstream licence is unestablished |
| 5 | **Overture Maps** transportation | ✅ INHERITED + re-confirmed: unsigned ListObjectsV2 **200**, unsigned ranged parquet GETs **200/206** | ⚠️ `ODbL-1.0` — commercial-OK but **copyleft**, plus an **open legal question** (is a trained model a Derivative Database?). INHERITED, tier PROVISIONAL | ❌ **ROAD-level only. Settled at 4 probes** (§3). Has `prohibited_transitions` + `destinations`, which no lane-level candidate here has | ⚠️ **2 of 3.** Global scale + turn restrictions + signposted destinations. **Complement, not substitute** |
| 6 | **Argoverse 2** *(incumbent)* | ✅ anonymous S3, INHERITED — 1000/1000 already pulled and verified | ❌ **`CC-BY-NC-SA-4.0`** + Terms whose Prohibited-Use Examples 3 & 4 **name training a model for a product**. NC **and** copyleft | ✅ 163 698 lane segments, 20 591 resolved branch points, `is_intersection` 35.1 % | ⚠️ **2 of 3.** The best lane graph we hold; **cannot ship**. Keep for research |
| 7 | **KITScenes-Multimodal** (KIT-MRT) — *new candidate, rejected* | ❌ **GATED.** HF API `gated:"auto"`; anonymous object fetch **401**; anonymous card fetch **401** (*"Access … is restricted. You must have access to it and be authenticated"*) | ❌ **`cc-by-nc-4.0`** (API tag). **Non-commercial.** ⚠️ tier PROVISIONAL only — the card itself is behind the gate, so the licence **could not be read as a document** | (Lanelet2, "full topological connectivity" — **PUBLISHED claim, unverifiable behind the gate**) | ❌ **DOUBLE FAIL — P1 and P4.** Painful: 62 km² Lanelet2 with lanes+signs+lights is exactly the shape we want |
| 8 | **Baidu Apollo** HD maps | ✅ GitHub raw, anonymous **200** | ✅ Apache-2.0 (repo) | ❌ **`modules/map/data` holds only 2 maps**; the shipped text map `demo/base_map.txt` (343 487 B) has **0 `successor_id` and 0 `predecessor_id`**. The *proto* defines them; the *data* does not populate them | ❌ **REJECT.** Classic "the format supports it" trap. No corpus value regardless |
| 9 | **levelXdata**, **Waymo**, **Lyft L5**, **KITTI-360**, **nuPlan**, **nuScenes**, **OpenLane-V2** | ❌ each already rejected at ≥2 probes in the two prior cycles | — | — | ❌ **not re-probed** — nothing found this cycle would change any of them |
| — | **ZOD** | ⛔ **NOT PROBED — parked by explicit PI instruction.** Also parked on the merits (4 prior probes: 2-D marking polylines, no topology field, no map module) | — | — | ⛔ **out of scope this cycle** |

**Satisfying only two, explicitly:**
- **Overture** = commercial ✅ + credential-free ✅, **lane connectivity ❌** (road-level).
- **Argoverse 2** = credential-free ✅ + lane connectivity ✅, **commercial ❌**.
- **KITScenes** = lane connectivity (claimed) ✅, **credential-free ❌ + commercial ❌**.
- **CommonRoad / esmini / CARLA** satisfy all three *on the evidence gathered*, but each carries a
  caveat that stops it being the recommendation: unresolved upstream licence, triviality, syntheticity.

**Nothing in this table is a mirror of a gated corpus.** No such mirror was fetched, listed or ranked.

---

## 3. THE OVERTURE LANE-LEVEL VERDICT — road-level is the ceiling

**Question asked by the brief:** does any Overture theme, extension or companion dataset carry
lane-level detail? **Answer: no. Class MEASURED. Tier: CONFIRMED at four independent probes**
(the brief required two).

| # | probe | result |
|---|---|---|
| 1 | **The authoritative JSON-Schema.** `raw.githubusercontent.com/OvertureMaps/schema/main/schema/transportation/segment.yaml`, HTTP 200, 29 908 B | **No lane property exists.** Road-segment properties are exactly `class`, `destinations`, `prohibited_transitions`, `road_surface`, `road_flags`, `speed_limits`, `width_rules`, `subclass`, plus the common `subclass_rules`, `access_restrictions`, `level`, `level_rules`, `connectors`, `routes`, `names`. The **only** occurrence of the substring "lane" in the whole file is the enum comment *"Inclined plane / cliff railway"* |
| 2 | **The whole schema repo.** GitHub tree API, HTTP 200, **1 167 paths** | Overture has exactly **6 themes** — `addresses`, `base`, `buildings`, `divisions`, `places`, `transportation`. `transportation/` contains exactly **two** entities: `segment.yaml` and `connector.yaml`. There is **no lane entity, no lane theme, no lane extension, no companion schema.** The only repo paths containing "lane" are `examples/…/road-oneway-no-lanes.yaml` — which, fetched (470 B), is a **one-way *access* example** with `access_restrictions: [{access_type: denied, when: {heading: forward}}]`. It has nothing to do with lanes |
| 3 | **The real bytes, independent of the last agent.** `release/2026-07-22.0/…/type=segment/part-00042-…parquet` (0.684 GB) read by unsigned HTTP range — **3.28 MB in 69 requests** | **21 columns**: `id, names, subtype, class, subclass, subclass_rules, connectors, road_surface, road_flags, rail_flags, width_rules, level_rules, access_restrictions, speed_limits, prohibited_transitions, routes, destinations, sources, geometry, version, bbox`. **Columns whose name contains "lane": 0. Nested arrow types anywhere containing "lane": 0.** |
| 4 | **The bucket listing.** unsigned ListObjectsV2, HTTP 200 | Releases `2026-06-17.0` and `2026-07-22.0`. Newest transportation theme = **68 segment parts (~38 GB) + 32 connector parts (~25 GB)**. Access is still ungated |

**Why probe 3 matters and is not a re-report:** the previous cycle read *one* part and reported
"21 columns, no `lanes`". I read a **different part** from a **newer release** and dumped the **full**
arrow schema plus a deep scan of every nested type. Two independent reads, same answer.
*(Small correction to the prior report while I was there: it described the theme as "~14 GB". The
2026-07-22.0 transportation theme is **~63 GB** across both entity types.)*

> **Verdict: Overture is a ROAD graph — segments are ways, connectors are junctions. Lane-level is
> not a gap in the data, it is absent from the model. It cannot substitute for a lane graph on S2
> (lane selection), and it never will without a schema change.**

**What Overture still uniquely buys, and no lane-level candidate here has:** `prohibited_transitions`
(legal turn restrictions as data), `destinations` (**signposted** destinations with
`from_connector_id`/`to_segment_id`/`final_heading` — the closest thing to a route/goal signal
anywhere in our corpus set), `speed_limits`, `access_restrictions`, `routes`, at **planetary** scale.
**Keep it. Use it as the strategic/road-level layer above a lane-level map.** Complement, not rival.

---

## 4. THE FIND, IN DETAIL — DLR's OpenDRIVE HD maps

### 4.1 What they are

Lane-detailed ASAM OpenDRIVE HD maps of real German road, surveyed by mobile mapping and published
by **DLR** (German Aerospace Center; Schwarzer Berg jointly with **iMAR Navigation GmbH**) on
**Zenodo**, each with a DOI, a CITATION.cff, a CHANGELOG and the licence text.
Publisher's stated accuracy: *"absolute coordinate error is expected to be less than 20 cm for road
elements within the drivable surface"*.

| record (version DOI) | map | licence | wire bytes | `.xodr` bytes | OpenDRIVE rev |
|---|---|---|---|---|---|
| `10.5281/zenodo.17434825` (v1.1.0) | **Schwarzer Berg, Brunswick** | **CC-BY-4.0** | 26 807 654 | 26 807 654 | 1.6 |
| `10.5281/zenodo.4043193` (v1.0.0) | **Brunswick inner ring road** | **CC-BY-4.0** | 5 240 081 (zip) | 46 720 948 | 1.4 |
| `10.5281/zenodo.7071846` | **ViVre research track, Brunswick** | **CC-BY-4.0** | 3 667 360 (zip) | 28 148 272 | 1.5 |
| `10.5281/zenodo.7072631` | **5G Living Lab, Wolfsburg** | **CC-BY-4.0** | 3 893 972 (zip) | 16 373 618 | 1.5 |
| `10.5281/zenodo.7056722` | **Test Bed Lower Saxony — SAMPLE** | **CC-BY-4.0** | 412 133 (zip) | 1 767 841 | 1.4 |
| `10.5281/zenodo.18507692` | Test Bed Lower Saxony — **FULL** (A2/A391/A39/L295) | ⚠️ **CC-BY-NC-SA-4.0** | *not pulled* | 81 236 104 | — |

**The five distinct maps cost 40 021 200 B ≈ 38.2 MiB on the wire.** Nothing large was downloaded and
the sizes are reported above as the brief required. *(Survey-wide traffic was higher — ≈ 94 MB — because
Schwarzer Berg was pulled at **both** versions to settle §4.4's version question, plus range probes and
CARLA's 15.5 MB map. Itemised in §7.)*

### 4.2 Lane connectivity — MEASURED over every downloaded file, not sampled

Raw: `evidence/dlr_opendrive_lanegraph_stats.json` (per-map, with md5 of each parsed file).
Probe: `evidence/opendrive_lanegraph_probe.py` (rerunnable, anonymous, no credential).

OpenDRIVE expresses the routable lane graph two ways and **both were checked**:
1. `<road><lanes><laneSection><left|center|right><lane id><link><successor id=/>` — lane continuation;
2. `<junction><connection incomingRoad connectingRoad contactPoint><laneLink from=".." to=".."/>` —
   **explicit lane-to-lane turn connectivity through an intersection**. This *is* the S1 structure.

| map | roads | junctions | km | lane nodes | driving lanes | `<lane><link><successor>` | junction `<laneLink>` | **branch pts** | out-deg hist |
|---|---|---|---|---|---|---|---|---|---|
| Schwarzer Berg | 257 | 31 | 11.26 | 24 146 | 4 197 | 19 413 | 577 | **79** | `{1:395, 2:63, 3:10, 4:4, 5:2}` |
| Inner ring road | 1 230 | 97 | 57.28 | 57 072 | 9 014 | 40 505 | 2 279 | **256** | `{1:1732, 2:224, 3:32}` |
| ViVre | 397 | 56 | 18.13 | 26 660 | 4 864 | 21 183 | 903 | **125** | `{1:632, 2:106, 3:17, 4:2}` |
| 5G Living Lab | 998 | 150 | 49.25 | 8 440 | 6 109 | 4 688 | 1 021 | **122** | `{1:763, 2:110, 3:10, 4:2}` |
| TFNds sample | 39 | 9 | 4.04 | 654 | 352 | 411 | 57 | **1** | `{1:55, 2:1}` |
| **TOTAL** | **2 921** | **343** | **139.96** | **116 972** | **24 536** | **86 200** | **4 837** | **583** | up to out-degree **5** |

Road-level link topology is also explicit and typed — e.g. Schwarzer Berg:
`predecessor:road 192 · successor:road 192 · successor:junction 54 · predecessor:junction 52`, so a
traversal knows whether the next element is a road or a junction without guessing.
Largest single junction: **18 connections** (5G Living Lab).

**Verbatim publisher confirmation that the driving lanes are routable** — from the Schwarzer Berg
README, and it is the *negative* form that makes it credible:
> *"bicycle and sidewalk lanes … are only modelled for visual completeness, **without logical routing
> capability**"* and *"tram lanes are only included for visual completeness, without logical routing
> capability"*.

The publisher distinguishes lanes that route from lanes that do not — which is precisely the
distinction ZOD could not make, because ZOD had no topology at all.

### 4.3 What it buys that AV2 does NOT have — the traffic-element gap, closed

AV2's measured absences (85 maps + the MIT devkit schema, two probes, prior cycle): **no stop-lines,
no traffic-light state or position, no roundabout label, no route/goal signal.**

MEASURED here across the three signalised maps:

| element | Schwarzer Berg | ViVre | 5G Living Lab | total | vs AV2 |
|---|---|---|---|---|---|
| **traffic lights** (`dynamic="yes"`, `name="TrafficLight"`, positioned `s/t/zOffset/orientation`) | 46 | 108 | 218 | **372** | AV2: **0** |
| **YIELD** (StVO 205) | 26 | 50 | 84 | **160** | AV2: 0 — this is the *nuScenes* YIELD stop-line capability, recovered |
| **STOP** (StVO 206) | 0 | 3 | 20 | **23** | AV2: 0 |
| **speed limit** (StVO 274) | 0 | 37 | 36 | **73** | AV2: 0 |
| all signal elements | 689 | 1 210 | 1 772 | **4 006** (+ 222 inner ring, 113 TFNds) | AV2: 0 |
| road objects | 5 116 | 3 111 | 4 000 | — | — |
| lane speed records | 4 258 | 4 874 | 3 763 | — | — |

Sample, verbatim from `bs-schwarzer-berg.xodr`:
```xml
<signal s="1.1987482007e+02" t="1.4760" id="5000600" name="TrafficLight" dynamic="yes"
        orientation="+" zOffset="2.3788" country="DEU" type="1000001" subtype="-1"
        hOffset="0.188821267414" width="0.26" height="0.78">
```

**Also: every map is georeferenced.** `<geoReference>` carries a full PROJ string —
`+proj=tmerc +lat_0=0 +lon_0=9 +k=0.9996 …` (Brunswick, ETRS89/DREF91/2016 UTM 32N + EGM2008 height,
with the WKT shipped as a separate file) and `+proj=utm +zone=32 +ellps=GRS80 …` (Wolfsburg).
So these maps can be **map-matched to any georeferenced trace and to Overture/OSM**, which the
log-local AV2 maps cannot be.

### 4.4 ⚠️ The licence, as a DOCUMENT — and the per-record trap

**Class: PUBLISHED. Tier: CONFIRMED.** Not from a badge, not from an API tag.

1. **The full CC BY 4.0 legal text ships inside every record** — `LICENSE.txt`, 18 657 B in the
   Schwarzer Berg record and 18 655 B inside each of the four zips. First line verbatim:
   `Attribution 4.0 International`.
2. **Machine-checked, not eyeballed:** the string `NonCommercial` is **absent** and the string
   `ShareAlike` is **absent** from all four shipped licence texts. It is plain BY.
3. **Publisher's prose** (README): *"This HD road network dataset is licenced under the terms of
   Creative Commons Attribution 4.0 International (LICENSE.txt)."*
4. **Zenodo record metadata** independently reports `license.id = "cc-by-4.0"`.

**Required attribution string, given by the publisher:**
> *OpenDRIVE dataset Schwarzer Berg © DLR and iMAR Navigation GmbH, CC BY 4.0, 2024*

**One obligation beyond the short name, recorded so it is not later mistaken for an NC term:** the
publisher disclaims correctness — *"a prototypic and simplified modelling … intended to be used in
research and development only. Neither the completeness nor the correctness of the data can be
guaranteed."* That is a **warranty disclaimer**, not a field-of-use restriction, and it does not
narrow the CC BY grant. It **is** a reason not to treat these maps as survey-grade ground truth.

> ### ⚠️ THE TRAP: the licence is PER RECORD, not per publisher.
> `10.5281/zenodo.7056722` — Test Bed Lower Saxony **sample**, 412 KB — is **`CC-BY-4.0`**.
> `10.5281/zenodo.18507692` — Test Bed Lower Saxony **full** A2/A391/A39/L295, 81 MB — is
> **`CC-BY-NC-SA-4.0`**.
> Same author. Same test bed. Opposite tier. An ingest that reads the licence once per publisher
> would silently pull an NC-SA motorway map into a `ship` shard. **Read the licence per version DOI.**

**A second version trap, measured:** Schwarzer Berg v1.0.0 (`…15395840`) and v1.1.0 (`…17434825`) have
**identical byte size (26 807 654)** but the **first 1 MiB md5 differs**
(`600138fc…` vs `cda7f278…`). The CHANGELOG attributes it to a precisified CRS definition. Size
equality is not identity — **cite the version DOI, never the concept DOI**.

### 4.5 Honest limitations of this find — stated, not buried

1. **NO IMAGERY.** These are maps. There is no camera, no lidar, no ego trajectory. This is the
   central limitation and it is why §6 is a partial negative.
2. **SMALL.** 139.96 km across 2 German cities, vs AV2's 1 000 logs / 6 US cities and Overture's
   planet. As a *topology bank* it is ~5 orders of magnitude below Overture.
3. **Geographically unpairable with our permissive corpora.** The obvious pairing candidate,
   `comma2k19` (`MIT`, tier `ship`, real GNSS), was driven in **California**. These maps are
   **Brunswick and Wolfsburg**. Map-matching one to the other is impossible. Stated because the
   previous cycle's recommended "comma2k19 + Overture" experiment does *not* transfer here.
4. **Merges were NOT measured.** My probe keys turn options on `(connectingRoad, to-lane)`, which is
   unique **by construction** — so its merge count is a structural zero, not a finding. Resolving a
   true merge needs the connecting road's own `<link><successor>` followed through. Recorded as
   **UNVERIFIED** in the JSON and in the code, so a zero is never read as "no merges exist".
5. **Roundabouts: not labelled**, same as AV2. All 343 junctions are `type="default"`; there is no
   roundabout flag. Derivable from cycles, not measured here.
6. **`is_intersection` has no direct analogue** — OpenDRIVE marks junction membership via the
   *connecting road* belonging to a `<junction>`, not a per-lane boolean. An adapter must derive it.
7. **CRS varies across the family** (tmerc with different false eastings; UTM32N GRS80; one map ships
   an explicit offset variant). An ingest must read `<geoReference>` per file, not assume.
8. **Two of the five are old** — the inner ring road header is dated `2015-03-25` (OpenDRIVE 1.4).
   Road layouts change. Do not treat as current.

---

## 5. Proposed `SOURCE_REGISTRY` entry — WRITTEN, deliberately NOT APPLIED

```python
# stack/tanitad/lake/schema.py
"dlr_opendrive": SourceLicense("owned-safe", "CC-BY-4.0",
                               share_alike=False, is_synthetic=False),
#   -> derived tier `ship` — the FIRST map asset in the program that is not `nc-research`
#      and not `ship-sa`. It may enter TanitDataSet-C.
```

**Not applied**, for the reason the previous cycle established: an unused registry entry is exactly
the *"sitting there correct and unused"* state that was flagged, and without an entry an ingest
attempt fails loudly, which is the safe default. **Escalated in §8, not written into a README.**

⚠️ **If applied, it must be keyed per record**, because of §4.4. A single `dlr_opendrive` key that
also admitted `zenodo.18507692` would route an **NC-SA** map into a **`ship`** shard. Either restrict
the key to the five CC-BY DOIs, or add a second key
`"dlr_opendrive_nc": SourceLicense("nc-research", "CC-BY-NC-SA-4.0", share_alike=True, …)`.

---

## 6. 🔴 THE RESIDUAL GAP — the clean negative, stated as a complete finding

The brief asked for a **publishable, credential-free corpus with a routable lane graph**. Split into
its parts, the answer is now asymmetric and worth stating precisely:

| what we need | status after this cycle |
|---|---|
| a **lane graph** that is commercial + credential-free + routable | ✅ **SOLVED** — DLR OpenDRIVE, `CC-BY-4.0`, tier `ship` |
| a **road graph** at global scale, with turn restrictions and route/goal signal | ✅ solved-with-copyleft — Overture, `ODbL` |
| **imagery + ego trajectories** that are commercial + credential-free **and** carry a lane graph | ❌ **NOT SOLVED. Nothing surveyed satisfies this.** |

> **No corpus in this survey is simultaneously commercially usable, credential-free, lane-connected
> AND carries camera data.** Every all-four-criteria PASS above is a map with no sensor stream.
> The closest misses, named as the brief required:
> - **Argoverse 2** — has everything except a shippable licence. One criterion away, and it is the
>   criterion that cannot be worked around.
> - **KITScenes-Multimodal** — 62 km² Lanelet2 with lanes, signs and lights over Karlsruhe/Frankfurt/
>   Sindelfingen, i.e. exactly the shape we want, but **gated *and* non-commercial**. Two away.
> - **comma2k19** — already ours at tier `ship` (MIT) with real GNSS, but **no lane graph**, and it is
>   in California so it cannot be matched to the German maps above.

**The consequence, stated plainly:** a *fully* publishable end-to-end proof still has no single-corpus
path. What is now possible that was not before is a **publishable MAP-side proof** — the strategic /
tactical branch-selection instrument can be built, trained and evaluated on a `ship`-tier lane graph
with real traffic lights and yield signs, without any NC contamination. Whether that is sufficient
for the paper is a PI call, not mine.

---

## 7. ⛔ Constraint compliance — what I did NOT do

- **ZOD: not touched.** Not probed, not fetched, not searched, no email drafted, no torrent, no
  Academic Torrents lookup. Parked by the PI's verbatim instruction until he is at his machine.
  *(It is also parked on the merits — 4 prior probes established 2-D marking polylines, no successor/
  neighbour/topology field, no map module. It never was the publishable strategic twin.)*
- **No licensor access gate was routed around.** No third-party mirror of any gated corpus was
  fetched, listed or ranked. When KITScenes returned 401 I recorded the 401 and stopped — I did not
  look for a copy of it elsewhere.
- **No account created, no form submitted, no Terms checkbox accepted, no email sent.**
- **No bulk download.** Total wire traffic for the entire survey ≈ **94 MB**, itemised:
  Schwarzer Berg v1.0.0 26 807 654 + v1.1.0 26 807 654 (both, to settle the version question in §4.4)
  + range probes 1 500 001 + 4 807 654 + 1 048 576 + inner ring 5 240 081 + ViVre 3 667 360
  + 5G Living Lab 3 893 972 + TFNds sample 412 133 + CARLA `TownBig.xodr` 15 536 627
  + Overture parquet range reads ≈ 3 280 000 + schema/tree/HTML/JSON ≈ 1 MB.
  The **63 GB** Overture theme, the **81 MB** NC-SA motorway map and the **1 926 GiB** AV2 tars were
  **not** pulled.
- **No pod was touched.** Dev box, CPU and network only.
- **`Keys.txt` was never read, printed, copied or passed in argv.** No credential was needed anywhere
  — that is the entire finding.

⚠️ **The dev-box trap held exactly as briefed.** Bare `curl` here returns `HTTP=000`
(`CRYPT_E_NO_REVOCATION_CHECK`), indistinguishable from an outage. **`--ssl-no-revoke` is in every
request in this report**; the Python parquet probe uses the sibling `truststore.inject_into_ssl()`.
Without either, this survey would have returned a false negative on every single candidate.

---

## 8. 🔴 ESCALATIONS — decisions, in the headline, not in a README

1. **Add `dlr_opendrive` to `SOURCE_REGISTRY` as tier `ship` — but keyed PER RECORD DOI.** §5. This is
   the first map asset that could legitimately enter `TanitDataSet-C`. It is *not* applied; the
   per-record hazard in §4.4 means a careless single key would route an NC-SA map into a commercial
   shard. **This report is the integration request.**
2. **An OpenDRIVE adapter does not exist and is now the blocking piece.** `stack/tanitad/data/` has
   `nuscenes.py` and the new `argoverse2.py`; there is **no `.xodr` reader**. The parse is
   straightforward (`evidence/opendrive_lanegraph_probe.py` already builds the graph in ~200 lines)
   and esmini's ~82 maps are ready-made unit-test fixtures. **ESTIMATED ~0.5 day.**
3. **Stop asking whether Overture has lanes.** Four probes, one verdict: road-level is the ceiling,
   and it is a model limitation, not a coverage gap. §3. Keep Overture as the **global road/route
   layer above** a lane-level map, for `prohibited_transitions` and `destinations` — the only
   route/goal signal anywhere in our inventory.
4. **The residual gap is IMAGERY, not the lane graph** (§6). If the PI wants a fully publishable
   end-to-end proof, that is now the question to aim the next cycle at — and the honest options are
   (a) accept a map-side-only publishable proof, (b) collect our own imagery in a mapped area, or
   (c) revisit whether a `ship`-tier imagery corpus with GNSS can be map-matched to Overture at
   road level. **Note that the previous cycle's recommended "comma2k19 + Overture" experiment is
   unaffected by this find and remains the cheapest discriminating test** — but it cannot use the
   DLR maps, because comma2k19 is Californian and the DLR maps are German.
5. **CommonRoad needs a licence determination it did not get here.** BSD-3 is the repo's only licence
   file, but the only sub-collection with a dense lane graph is plausibly OSM-derived and would
   inherit ODbL. **Do not promote it on the BSD file** — that is the OpenLane-V2 pattern. One probe
   of the scenario-factory provenance would settle it.

---

## 9. Claims ledger — evidence class **and** tier on everything load-bearing

| claim | class | tier | where |
|---|---|---|---|
| DLR maps are anonymously downloadable (record API + object GET, HTTP 200, no credential) | **MEASURED** | **DECISION-GRADE** | `evidence/access_probes_pch.json` probes 1–5 |
| Data licence is `CC-BY-4.0`; full legal text ships in-record; `NonCommercial` and `ShareAlike` absent from all 4 shipped texts | **PUBLISHED** (document fetched + machine-checked) | **CONFIRMED** | `evidence/license_determinations_pch.json`, `evidence/dlr_CC-BY-4.0_LICENSE_as_shipped.txt` |
| 2 921 roads · 343 junctions · 139.96 km · 116 972 lane nodes · 24 536 driving lanes | **MEASURED** | **DECISION-GRADE** (full parse of every file pulled) | `evidence/dlr_opendrive_lanegraph_stats.json` |
| 86 200 `<lane><link><successor>` elements; 4 837 junction `<laneLink>` turn edges; **583 branch points**, out-degree up to 5 | **MEASURED** | **DECISION-GRADE** | same |
| 372 traffic lights (`dynamic="yes"`), 160 YIELD (205), 23 STOP (206), 73 speed-limit (274), 4 006 signal elements | **MEASURED** | **DECISION-GRADE** | §4.3, reproducible from the staged probe |
| Every map is georeferenced with a full PROJ string | **MEASURED** | CONFIRMED | `dlr_opendrive_lanegraph_stats.json` `header.geoReference` |
| Publisher states bicycle/sidewalk/tram lanes are "without logical routing capability" (⇒ driving lanes route) | **PUBLISHED** | CONFIRMED | `evidence/dlr_schwarzer_berg_README_v1.1.0.md` |
| **Per-record licence divergence**: TFNds *sample* is CC-BY-4.0, TFNds *full* is CC-BY-NC-SA-4.0 | **MEASURED** (Zenodo API, both records) | **CONFIRMED** | `access_probes_pch.json` probe 5 |
| Schwarzer Berg v1.0.0 vs v1.1.0: identical size, **different** first-1 MiB md5 | **MEASURED** | CONFIRMED | `access_probes_pch.json` probe 3 |
| **Overture has no lane-level detail in any theme** | **MEASURED** (schema + repo tree + parquet, 3 paths) **+ PUBLISHED** (authoritative JSON-Schema) | **CONFIRMED — 4 independent probes** | §3, `evidence/overture_lane_probe_part00042.json`, `evidence/overture_segment_schema_main_2026-07-26.yaml` |
| Overture 2026-07-22.0 transportation is 68 segment + 32 connector parts (~63 GB) | **MEASURED** | CONFIRMED | §3 probe 4 |
| Overture licence is `ODbL-1.0`, not CDLA-Permissive | **INHERITED** from the prior cycle — **not re-derived here** | PROVISIONAL | `license_determinations_pch.json` |
| KITScenes is gated (401 on both a data object and the card) | **MEASURED** | **CONFIRMED — 3 probes** | `access_probes_pch.json` |
| KITScenes is `cc-by-nc-4.0` | **PUBLISHED (API metadata tag only — the card is behind the gate and could NOT be read as a document)** | **PROVISIONAL** | same |
| CARLA: code MIT, assets "the CC-BY License" (**no version stated**); `TownBig.xodr` has 2 416 `<laneLink>` | **PUBLISHED** + **MEASURED** | CONFIRMED for the statement; PROVISIONAL for which licence covers the file | §2 row 2 |
| CommonRoad: BSD-3 is the repo's only licence; connectivity real but uneven (0 successors in 5/6 hand-crafted; 116 in scenario-factory) | **PUBLISHED** + **MEASURED** (12 files) | **PROVISIONAL** — upstream (likely OSM/ODbL) inheritance unresolved | `evidence/commonroad_lanelet_probe.json` |
| esmini is MPL-2.0 with ~82 `.xodr`; **those maps not byte-verified by me** | **PUBLISHED** | PROVISIONAL | §2 row 3 |
| Apollo's shipped `demo/base_map.txt` has **0** `successor_id` / `predecessor_id` | **MEASURED** | CONFIRMED (1 of only 2 maps in repo; the other is binary proto — **not** parsed) | `access_probes_pch.json` |
| **Merges in the DLR maps** | **UNVERIFIED** — probe key is unique by construction; a structural zero, not a finding | — | §4.5 item 4, and a note in the probe source |
| AV2 licence / AV2 connectivity / Waymo / Lyft / KITTI-360 / nuPlan / nuScenes / OpenLane-V2 / levelXdata gates | **INHERITED** from the two prior cycles — not re-probed | — | decides nothing new here |
| OpenDRIVE adapter effort ~0.5 day | **ESTIMATED** | — | §8 item 2 |

---

## 10. Deliverable manifest

**Everything below is in the repo and `git add`-staged. Nothing lives only in a scratchpad, only on a
pod, or only in this agent's context. No pod was touched. Nothing was committed or pushed.**

| # | artifact | where it lives | only one place? |
|---|---|---|---|
| 1 | `PUBLISHABLE_CORPUS_HUNT.md` — this report: decision rule, ranked three-way table, Overture verdict, the find, residual gap, escalations, claims ledger | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-26-publishable-corpus-hunt/` | no — staged |
| 2 | `evidence/opendrive_lanegraph_probe.py` — the OpenDRIVE lane-graph analyser (rerunnable, anonymous, md5s every file it parses; carries the UNVERIFIED merge note in-source) | same dir | no — staged |
| 3 | `evidence/dlr_opendrive_lanegraph_stats.json` — per-map + aggregate connectivity, headers, geoReference, signal type histograms, md5s | same dir | no — staged |
| 4 | `evidence/overture_lane_probe.py` — independent parquet-over-HTTP-range schema reader | same dir | no — staged |
| 5 | `evidence/overture_lane_probe_part00042.json` — full 21-column arrow schema, deep lane scan, sample rows, bytes/requests | same dir | no — staged |
| 6 | `evidence/overture_segment_schema_main_2026-07-26.yaml` — the authoritative JSON-Schema as fetched (29 908 B) | same dir | no — staged |
| 7 | `evidence/overture_connector_schema_main_2026-07-26.yaml` — as fetched (658 B) | same dir | no — staged |
| 8 | `evidence/access_probes_pch.json` — every access probe with URL, method, HTTP code, bytes, finding; ≥2 per candidate; plus an explicit *not-probed-and-why* section | same dir | no — staged |
| 9 | `evidence/license_determinations_pch.json` — every licence as a **document**, verbatim, data-vs-code split, PROVISIONAL/CONFIRMED tier per determination, per-record warning | same dir | no — staged |
| 10 | `evidence/dlr_CC-BY-4.0_LICENSE_as_shipped.txt` — the CC BY 4.0 legal text exactly as the licensor ships it (18 657 B) | same dir | no — staged |
| 11 | `evidence/dlr_schwarzer_berg_README_v1.1.0.md` — the publisher's own README (routing-capability statement, CRS, attribution string) | same dir | no — staged |
| 12 | `evidence/commonroad_lanelet_probe.json` — 12 real CommonRoad scenario XMLs, per-file lanelet/successor/predecessor/adjacent/intersection counts | same dir | no — staged |
| 13 | `evidence/zenodo_records_dlr_opendrive.json` — the six Zenodo record metadata blobs (DOI, concept DOI, version, licence id, file list with sizes) | same dir | no — staged |

**Deliberately NOT staged — and why:**

- **The 5 raw `.xodr` files (119 818 333 B = 114.3 MiB uncompressed).** Our policy is *pointers + derived features,
  never source bytes*. ⚖️ **Worth recording precisely, because it is easy to get backwards:** unlike
  AV2, `CC-BY-4.0` **does** permit redistribution, so staging these would be *legally* fine. It is a
  **policy** choice about repo hygiene, not a licence constraint. They are fully reproducible in
  ~40 s by rerunning the staged probe against the DOIs in `zenodo_records_dlr_opendrive.json`.
  *Scratchpad location, this session only:* `…/scratchpad/pch/dlr/*.xodr`.
- **`carla_TownBig.xodr` (15.5 MB)** — same reasoning; it is one `curl` from a public repo.
- **No `SOURCE_REGISTRY` change was made.** The `dlr_opendrive` entry is *proposed* (§5) and escalated
  (§8), not applied — deliberately, because of the per-record hazard in §4.4.
- **No PhysicalAI-AV content of any kind** — no clip UUIDs, no raw content.
- **`Keys.txt` was never read, printed, copied or passed in argv.**
- **Nothing gated was routed around.** No third-party mirror, no scraped copy, no torrent, no account,
  no form, no Terms accepted, no email sent. **ZOD was not touched.**
