# P5 — TanitScena — SPEC

`Product P5 of TANITAD_PROGRAMME.md §1. Owner: TanitAD_DataFlyWheel (P2 P3 P5 P8).`
`Status: DRAFT for PI review. Written 2026-08-23. No prior P5 spec existed.`

> **PI definition (verbatim, `Project Steering/TANITAD_PROGRAMME.md:40`):**
> *"P5 | **TanitScena** | scenario DB: descriptions, dataset links, vector search over
> embeddings; code API + high-quality UI (semantic search)"*

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED`
(another agent/doc, not re-verified) · `ESTIMATED` (arithmetic shown) · `HYPOTHESIS` ·
`UNVERIFIED` (probe named). Facts carrying a `file:line` were read directly this session by me or
by a survey agent working in this worktree; the `file:line` **is** the re-verification handle.
§12 lists everything this session could **not** verify and how to close it.

---

## 0. TL;DR for the PI

**TanitScena exists and works — over the wrong population.** It is a polished, tested,
single-port app with real local semantic search over the **14 hand-authored opponent-weakness
scenario *classes*** (`SC-01…SC-14`). Parser, vector index, dataset-link cards, UI: all real code
with 253 lines of tests.

**What is missing is the database half.** Nothing in a record links a scenario to a clip, an
episode or a time window. That blocks P8, blocks `/TanitAD_SearchScenarios`, and is why the
product reads "partly exists".

**Three findings decide the design, and all three are good news:**

1. **The text corpus to embed already exists and is free.**
   `…/2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy_per_clip.jsonl` holds **4,729 rows**
   keyed on `clip_id`, each with a short natural-language sentence (`cot`, 100 % coverage, median
   49 chars). Plus **801** templated `scenario_description` one-liners in the fused PH1 records.
   **No GPU pass and no VLM run is needed to reach a working instance index.**
   ⚠️ **But it is mostly NOT the parity corpus** — MEASURED today: only **201 of 4,729** Alpamayo
   clips are in the parity train set, and **6 are in the canonical val40 deployment**. See §3.7.
2. **The licence registry already exists in code** — `stack/tanitad/lake/schema.py:44` defines
   `LICENSE_CLASSES`, and `SOURCE_REGISTRY` (`:68-148`) covers **19 sources**. TanitScena must
   **import** it, not re-invent it. (An earlier draft of this spec invented a licence vocabulary;
   that was wrong and is corrected here.)
3. **The parity join key is `clip_id`, not an episode index** — `stack/tanitad/data/parity.py`
   gives the constants, the digest function and the membership oracle. And **`physicalai_av` is
   `gated-confidential`**, so the *publishable* form is `sha256(clip_id)`, never the raw id.

**Compute: every core path is CPU-only.** No GPU is required for v1.

---

## 1. Purpose & scope

### 1.1 Purpose

TanitScena is the programme's **single searchable index of driving scenarios**: what a scenario
is, which data exemplifies it, under which licence, and where each example lives in our corpora.
It exists so a natural-language question — *"stopped lead vehicle in my lane"* — returns, in one
step, the curated scenario class **and** the concrete clips, with provenance and licence attached.

Three consumers justify it:

1. **P8 TanitDataSetCreator** builds datasets *by querying TanitScena*
   (`Project Steering/SKILLS_SPECS.md:45`: *"query TanitScena (P5) → select sources → licence
   class check … → register in TanitScena"*).
2. **`/TanitAD_SearchScenarios`** is a named programme skill whose entire procedure is TanitScena
   (`SKILLS_SPECS.md:48-51`: *"embed query → vector search over scenario/dataset embeddings →
   return scenarios with dataset links, counts, provenance"*).
3. **The excellence programme.** `SCENARIO_DATABASE.md:3-4` states the goal: *"prove TanitAD
   excels at every scenario in this database"*. That requires per-scenario data to exist and be
   findable — today it is prose.

### 1.2 In scope

- Scenario **class** records (curated, authored, order 10-100).
- Scenario **instance** records (derived, order 10^3-10^6), each a clip or window in a named
  corpus, keyed on the parity clip identity.
- **Dataset link** records: corpus, URL, **licence class (imported from `lake/schema.py`)**,
  reachability, size, train/eval-lane role.
- A **text embedding space**, exact-cosine retrieval, and structured filters applied pre-top-k.
- A **code API** (Python + CLI) and an **HTTP API**.
- A **UI** with semantic search, faceting, detail views, and set export.

### 1.3 Out of scope (explicitly)

- **Running or scoring scenarios.** That is `stack/tanitad/eval/scenarios/` + P7. TanitScena
  stores a *pointer* to a registry key; it never scores.
- **Building datasets.** Materialising frames/features is P8. TanitScena emits a **ScenarioSet**
  (keys + provenance), never bytes.
- **Authoring opponent evidence.** Owned by the Opponent Analyzer (`SCENARIO_DATABASE.md:9-12`).
- **Deriving labels.** TanitScena consumes label artifacts; it never invents a label (§4.5).
- **Any metered compute** (§9).

---

## 2. Boundaries

### 2.1 vs P8 — TanitDataSetCreator (sits ON TOP)

| | TanitScena (P5) | TanitDataSetCreator (P8) |
|---|---|---|
| answers | *"which scenarios / clips match this description?"* | *"build me a dataset with this mix"* |
| output | a **ScenarioSet**: ids + keys + provenance + licence, **no pixels** | a materialised dataset + manifest + skip-hash |
| owns | descriptions, embeddings, links, licence class | mixing ratios, sampling, caching, build |
| direction | P8 **queries** P5 and **registers the result back** as a `DatasetLink` (`SKILLS_SPECS.md:45-46`) | — |

**The dividing line is materialisation.** The moment bytes are copied or features computed, it is
P8. `products/P8-datasetcreator/` exists (empty, sibling agent) — the two specs must agree on the
ScenarioSet manifest shape (§6.4). **Escalated as an integration item.**

### 2.2 vs P7 — TanitEval's strata (RELATED, SEPARATE — do not conflate)

This is now precise, because the strata are defined in exactly one place.

**MEASURED** `taniteval/taniteval/corridor.py:231-243` — `strata(head_deg_2s, speed, junction_deg)`
returns **four boolean numpy masks** over a window array: `overall`, `junction`, `longitudinal`,
`other`. `junction` is `|head_deg_2s| >= JUNCTION_DEG` (`JUNCTION_DEG = 10.0`, `corridor.py:135`,
marked PROPOSED); `longitudinal` is `(~junction) & (speed >= np.median(speed))`.

Four consequences, each of which separates a stratum from a scenario record:

1. **A stratum is an ephemeral mask, not a record.** It has no id, no text, no provenance, and it
   is recomputed at scoring time.
2. **A stratum is relative to the arm being scored** — `longitudinal` uses the **median of the
   current window set**, so the same window can change stratum when the window set changes.
3. **Strata partition; scenarios do not.** Every window lands in exactly one of
   `junction/longitudinal/other`. A clip may match zero, one, or several scenarios; most match none.
4. **A stratum is kinematic by construction and says so.** `corridor.py:223-227` is explicit that
   it is *"a kinematic signature, never a topology"* and that there is no map, lane graph or
   junction annotation in this corpus. A scenario record, by contrast, is an authored claim about
   a situation.

There are **three** stratifications in the repo, which is itself a reason not to add a fourth:
`corridor.py:231-243` (the canonical four), `taniteval/taniteval/bench.py:239-252` (curvature
buckets + speed **terciles**, `:290-293`), and `taniteval/_planner_extra.py:55-62` (five ad-hoc
masks). Closed-loop rebuilds the canonical four at `taniteval/taniteval/clhorizon.py:820-821`.

**Binding rule:** TanitScena must never be the source of an eval stratification, and a stratum
definition must never be imported as a scenario record. If a scenario is promoted to an eval
stratum, the stratum is defined in P7, frozen there, and TanitScena stores only a pointer.

⚠️ **Name collision, three ways — name it or it will be overloaded a fourth time.**
`…/2026-07-21-vlm-production-semantic/scenario_strata_val.jsonl` is **neither** a scenario record
**nor** a corridor stratum: it is 30 per-`(episode, t)` VLM semantic-label rows with 36 typed
fields, produced on the **known-leaky** val build `physicalai-val-f1b378f295ae`
(`stack/tanitad/data/parity.py:24-27`) and carrying `*_passB_CONTAMINATED` variants.
⛔ **It must not be ingested as instances**; it may be read only as a design reference for facet
field names.

### 2.3 vs `stack/tanitad/eval/scenarios/` — the executable scenarios

**MEASURED** — the registry exists and is small:
`stack/tanitad/eval/scenarios/registry.py:46-53` defines
`@dataclass(frozen=True) class ScenarioEntry(name, make, policies, simulate, score, headline)`;
`SCENARIO_REGISTRY` (`:58`) has **exactly three keys**: `work_zone_phantom` (headline `OKRI`),
`traffic_light_red` and `traffic_light_green` (both `TLC`). The suite entry point is
`run_registered_suite(names=None) -> {scenario_key: {policy: metric_dict}}` (`:86`, `:95-96`).
Adding a scenario means **editing `registry.py`** — there is no file-driven manifest.

⇒ **TanitScena's `ScenarioType` carries `eval_registry_key` — a pointer, nothing more.** Of the
14 catalogued `SC-xx`, only **SC-01** and **SC-14** have executable counterparts today.

⚠️ **Stranding escalation.** Three more scenario modules of the same shape exist only under
`TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/` and **cannot be imported from
`stack/`**: `2026-07-24-stop-arm-gate-scenario/stop_arm_gate.py` (SC-04),
`2026-07-31-stationary-lead-scenario/stationary_lead.py` (SC-13),
`2026-08-07-emergency-scene-scenario/emergency_scene.py` (SC-06). Their SC entries claim
"awaiting orchestrator triage" (`SCENARIO_DATABASE.md:94-96, :170-172, :360-363`).
**Reported, not written into a doc: these need promotion into the registry.** (Backlog `P5-12`.)

### 2.4 vs the `SCENARIO_DATABASE.md` markdown

The 14 `SC-xx` entries are hand-authored Markdown owned jointly by four agents on a weekly
cadence (`SCENARIO_DATABASE.md:9-12`). TanitScena parses it today. The spec keeps that ingestion
path — prose is the right medium for curated opponent evidence — but adds a **sha256 of the
source document** to every ingested record, so a record always traces to the exact revision it
came from. Direction-of-truth is PI question Q4 (§13).

---

## 3. Current state — MEASURED

| # | PI-required capability | state | one-line evidence |
|---|---|---|---|
| 1 | scenario **descriptions** | **PARTIAL** | 14 class records, free text + structure; **zero** instance descriptions in the store (though the raw text exists — §3.7) |
| 2 | **dataset links** | **PARTIAL** | free-text refs + heuristic landing pages; **no licence class**, no reachability, no size — while a full licence registry sits unused in `lake/schema.py` |
| 3 | **vector DB over embeddings** | **PARTIAL** | exact cosine over **one** text space, n=14; MiniLM path present but **not installed on any venv** |
| 4 | **code API** | **PARTIAL** | parse + index + HTTP; **no query CLI, no structured filters, no export** |
| 5 | **UI, semantic search** | **EXISTS** | working single-port SPA with live semantic search, facet chips, detail views |

### 3.1 Descriptions — PARTIAL

- **MEASURED** `stack/tanitad/scena/parse.py:322-344` — the record emitted per entry:
  `id, title, w_code, family, stars, headline, is_new, tags, evidence_label, evidence_label_raw,
  opponent_evidence, description, correct_behavior, mechanism, data_sources, metric_hooks,
  metric_hooks_text, lifecycle_stage, status_text, evidence_links, parse_warnings`.
  The description is **both** free text and structured.
- **MEASURED** `parse.py:39-46` — the lifecycle ladder is a fixed 6-tuple:
  `catalogued, spec-drafted, data-sourced, oracle-tested, live-measured, excellence-proven`.
- **MEASURED** `parse.py:51` — evidence labels `FACT | CLAIM | INFER`; compound labels collapse to
  the primary token (`parse.py:84-90`).
- **MEASURED** `parse.py:286-293` — `correct_behavior` is **regex-extracted from prose**, not an
  authored field, and fails soft into `parse_warnings`. ⇒ the single most decision-relevant
  sentence per record is derived by a regex. The schema must mark it as such (§4.1
  `description.authored`).
- **MEASURED** — `SCENARIO_DATABASE.md` is 417 lines, `SC-01 … SC-14`;
  `stack/tests/test_scena.py:51` pins `>= 14`. **No machine-readable sibling exists** in
  `Opponent Analyzer/` (the other entries are `BACKLOG.md`, `Implementation/`, `Research/`).

### 3.2 Dataset links — PARTIAL, and the fix is already in the repo

- **MEASURED** `parse.py:209-234` — a data source parses to `{kind, ref, status, link, replay}`;
  `kind` is regex-detected from a fixed list (`parse.py:55-63`:
  `nuScenes, Cosmos, NuRec, comma2k19, HF, CARLA, dashcam`, else `other`).
- **MEASURED** `parse.py:108-130` — `_dataset_link()` returns an explicit URL if the clause has
  one, else a hard-coded landing page, else for HF/Cosmos a HuggingFace **dataset-search** URL.
  The docstring states the intent: *"never a fabricated repo id"*. Honest — and it means the links
  are **not machine-resolvable dataset identifiers**.
- ⭐ **GAP, with the fix already written.** The emitted dict has **no licence field**
  (`parse.py:227-233`), yet a machine-readable licence registry exists:
  - **MEASURED** `stack/tanitad/lake/schema.py:44` —
    `LICENSE_CLASSES = ("owned-safe", "nc-research", "gated-confidential", "refuse")`.
    `refuse` is worse than gated: terms follow the **trained weights**, so no tier can contain
    them (`schema.py:37-43`).
  - **MEASURED** `schema.py:47-61` — `@dataclass(frozen=True) class SourceLicense(license_class,
    license_name, share_alike, is_synthetic)` with
    `commercial_ok == (license_class == "owned-safe" and not share_alike)`.
  - **MEASURED** `schema.py:68-148` — `SOURCE_REGISTRY`, **19 sources**. `comma2k19` MIT ·
    `cosmos_dd` CC-BY-4.0 synthetic · `l2d` Apache-2.0 · `pandaset`, `dlr_opendrive` CC-BY-4.0 ·
    `udacity` MIT · `worldmodel_synth` **OpenMDW-1.1** (`schema.py:81-82`) · `zod`,
    `dlr_opendrive_nc` share-alike · `nuscenes, bdd100k, a2d2, argoverse2, kitti360, once, drama,
    rank2tell` = `nc-research` · **`waymo`, `waymax` = `refuse`** ·
    **`physicalai_av` = `gated-confidential`**.
  - Enforcement already exists: `lake/filtering.py:29 tier_of(...)`,
    `lake/license_guard.py:23 verify_license_scope(...)`, shard layout
    `shards/<license_class>[/sharealike]/<source>/<split>/…` (`lake/shards.py:20,39`), and catalog
    columns `license_class, commercial_ok, share_alike, split, source` (`lake/catalog.py:112`).
  ⇒ **TanitScena imports this. It does not define a licence vocabulary.**
- **⚠️ Vocabulary correction, worth stating once.** `TANITAD_PROGRAMME.md:18-21` uses the prose
  terms `research-OK` / `commercial-OK`. Those strings **do not exist in code** (probed
  `--include=*.py` over `stack/ tools/ taniteval/`, and `--include=*.md` over `stack/ tools/
  DataEng/`). The code field is `license_class` with the four values above plus the derived
  `commercial_ok` boolean. **The registry wins over prose** (house rule). TanitScena stores the
  code vocabulary and renders the prose term in the UI.
- **GAP — reachability and size.** `status` is whatever sat in the clause's last parenthesis
  (`parse.py:224-225`); no probe, no date, no counts.

### 3.3 Vector DB over embeddings — PARTIAL

- **MEASURED** `stack/tanitad/scena/vector.py:34-41` — `doc_text()` embeds
  `title + description + correct_behavior + tags`. `opponent_evidence`, `mechanism`,
  `metric_hooks` and `status_text` are **not** embedded.
- **MEASURED** `vector.py:212-222` — exact cosine over an in-memory unit-row `float32` matrix
  (`sims = self.matrix @ q`, `argsort`, top-k). No ANN index, and none is needed at n=14.
- **MEASURED** `vector.py:155-188` — two embedder paths: `minilm`
  (`sentence-transformers all-MiniLM-L6-v2`, lazily imported at `vector.py:124`) and a
  deterministic pure-numpy hashing TF-IDF (`vector.py:48-107`; md5-bucketed for cross-process
  determinism, `:76-77`). `prefer="auto"` picks MiniLM iff importable.
- **MEASURED, dev box, 2026-08-23** — project venv `C:\Users\Admin\venvs\tanitad` (Python 3.13.5):
  `numpy ✅ fastapi ✅ uvicorn ✅ pytest ✅ sklearn ✅ torch ✅ transformers ✅ onnxruntime ✅
  httpx ✅ · **sentence_transformers ❌** · faiss ❌ · chromadb ❌ · hnswlib ❌`.
  A survey agent extended the probe to **all five interpreters on the box**
  (`venvs/tanitad`, `venvs/carla312`, `venvs/sam3run`, `venvs/colab`, `C:\Python314`):
  **`sentence_transformers` is importable on none of them.**
  ⇒ **the live embedder today is `hashing-tfidf`.**
- **MEASURED** — the repo has **exactly one** dependency file, `stack/pyproject.toml`:
  `dependencies = ["torch>=2.4", "numpy>=1.26"]`; extras `dev`, `sim`, `net`, `real`. No
  `requirements*.txt`, no `environment.yml`, no lockfile. Adding MiniLM means **one extras entry**.
- **MEASURED, dev box, 2026-08-23** — GPU: `NVIDIA GeForce RTX 4060, 8188 MiB, driver 591.86`
  (`nvidia-smi`). Unmetered.
- **MEASURED** `vector.py:225-248` — the index persists to one `vectors.npz`
  (`ids`, `matrix`, `meta` JSON, `idf`) and reloads without re-embedding.

**DEFECT D1 — stale vectors survive a text edit.** `MEASURED` by reading
`stack/scripts/scena_app.py:83-113`: `_ensure_index(force=False)` reuses the cached `vectors.npz`
whenever `set(idx.ids) == cur_ids` (`:96-102`), and `_reload()` (`:70-80`) only sets
`st.index = None` without deleting the npz. ⇒ **edit a description, add or remove nothing,
restart the server — the served ranking comes from the OLD text.** `/api/reindex` (`:194-197`) is
the only escape. The cache key must be a content hash, not an id-set.

**DEFECT D2 — the hashing IDF is corpus-global.** `MEASURED` `vector.py:93-102`: IDF is fit over
the whole corpus at build time, so adding one record changes **every** vector. Incremental
re-embedding is sound for MiniLM only; the hashing path needs a full refit. Any incremental design
must state this rather than discover it.

**GAP — one space only.** No visual embedding, no second query space.

### 3.4 Code API — PARTIAL

- **MEASURED** `stack/tanitad/scena/__init__.py:20-32` — the entire importable surface:
  `STAGES, parse_file, parse_scenarios, to_index, write_scenarios_json, HashingTfidf,
  VectorIndex, doc_text, static_dir`.
- **MEASURED** `parse.py:419-436` — the only CLI is the parser:
  `python -m tanitad.scena.parse --db-md <path> --out scenarios.json`.
  **No search CLI, no filter CLI, no export CLI.**
- **MEASURED HTTP surface** `scena_app.py:148-202`: `GET /` · `GET /static/*` · `GET /api/meta` ·
  `GET /api/scenarios` · `GET /api/scenario/{sid}` (guarded `^SC-\d+$`, `:46,174`) ·
  `GET /api/search?q=&k=` · `POST|GET /api/reindex`.
- **DEFECT D3 — no server-side structured filters.** `MEASURED` `scena_app.py:178-192`: the search
  endpoint's whole signature is `search(q: str = "", k: int = 8)`. The facet chips documented at
  `scena/README.md:70-72` are **client-side filtering of an already-returned list**. A
  programmatic consumer cannot ask for *"licence in {owned-safe} AND corpus=comma2k19"* — which is
  exactly what `SKILLS_SPECS.md:50` requires (*"filters (source, licence, size)"*).
- **DEFECT D4 — no export.** No endpoint or function emits a selected set.
- `build_app(db_md, static, cache_dir, prefer)` (`scena_app.py:116-204`) is a clean factory and a
  good extension point. Keep it.

### 3.5 UI with semantic search — EXISTS

- **MEASURED** `stack/tanitad/scena/static/` — `index.html` (815 B), `app.js` (40,963 B),
  `style.css` (15,208 B). `index.html` is a no-build, no-CDN SPA shell: `TanitScena` wordmark,
  `#app` root, one `<script src="/static/app.js">`.
- **MEASURED** `scena/README.md:68-89` documents the two implemented views: a search-first home
  (live `/api/search`, ranked cards with score bars, filter chips over lifecycle stage / evidence
  label / stars / data-source kind) and a detail view (opponent evidence, description, highlighted
  correct-behavior callout, mechanism, a hand-authored schematic BEV canvas per scenario family
  **with an on-canvas legend**, metric hooks, lifecycle stepper, dataset-link cards with a replay
  affordance). URL-hash routing (`#/s/<id>`, `#/q/<query>`) makes any view shareable.
- **This is the strongest of the five capabilities. Do not rewrite it.** The work is new views
  over new record types, not a new front end.
  *(`app.js` internals are `UNVERIFIED` — the filesystem refused to read it, §12. The claims above
  come from README.md and from the endpoints the server actually exposes.)*

### 3.6 Tests that pin today's behaviour

`stack/tests/test_scena.py` (253 lines), CPU-only and offline-safe, pins against the **real**
`SCENARIO_DATABASE.md`: parser coverage and uniqueness (`:49-54`); SC-01's FACT/Waymo evidence,
3 stars, `W-01`, a resolvable data-source link and the `closure_incursion_m` hook (`:57-73`);
evidence labels and lifecycle stages (`:75-84`); fail-soft on malformed input (`:95-100`);
hashing-path top-1 for cone / stop-arm / fog / red-light queries (`:104-110`, `:129-134`);
byte-for-byte determinism (`:112-117`); npz round-trip (`:119-127`); server list / detail / 404 /
traversal guard / search / reindex / meta (`:167-232`); and portability — no absolute path in
`scenarios.json` (`:240-253`).

**Evidence class for the retrieval results: `INHERITED`** — pinned in-repo, **not re-run this
session** (§12).

### 3.7 ⭐ The instance text corpus already exists (and it is free)

This is the finding that makes §8 SC-B reachable without a GPU.

| artifact | rows | key | text | in git? |
|---|---|---|---|---|
| `…/2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy_per_clip.jsonl` | **4,729** | `clip_id` | ⭐ `cot` — a natural-language sentence, **100 % coverage**, median **49 chars**, **1,103 distinct** (23.3 % distinct ⇒ heavily templated). Example record: `{"clip_id":"923e09b1-…","longitudinal":"Gentle Deceleration","lateral":"Steer Right","lane":"Lane Keep","cot":"Slow down due to the lead vehicle ahead."}` | **yes** |
| fused PH1 records, top-level `scenario_description` | **801** | `clip_id` | a templated one-liner, e.g. `"night, clear, urban 2-lane; ego 4.1 m/s braking/turning_right; no agents"` | ⚠️ **no — 204 files (4.26 MB) live on HF `Sayood/tanitad-ph0-aug120/fused_aug120/`; only 60 samples in repo** |
| `…/2026-08-16-s2-v1-labels/review/labels_v2/s2_labels_{aug120,w120val}.jsonl` | **201 / 596** | `clip_id` | tokens only, no free text | yes |
| `…/2026-07-21-vlm-production-semantic/{val_full,legacy_pod3_passA}.jsonl` | 30 / 400 | `episode`,`t` | VLM `pass_A.parsed.route_evidence`, 1-2 sentences | yes, **but on the leaky val build** (§2.2) |

**MEASURED absence, two probes:** there is **no VLM free-form per-clip caption corpus** — glob
`**/*caption*` returns nothing and `grep caption` over `stack/scripts` returns 0 hits. What exists
is (a) 4,729 templated `cot` sentences, (b) ~430 VLM `route_evidence` sentences on val windows,
(c) 801 templated `scenario_description` lines.

#### ⭐ MEASURED 2026-08-23 — the Alpamayo/parity overlap (this closes an open question)

Probe (mine, this session): read all 4,729 rows, take `sha256(clip_id)` per row, intersect with
`stack/tanitad/data/parity_train_clip_digests.json` (`clip_id_digests`, **2,400** entries) and with
`stack/tanitad/data/deployed_val40_clip_digests.json` (**40** entries).

| quantity | value |
|---|---|
| Alpamayo rows / distinct `clip_id` | **4,729 / 4,729** (no duplicates) |
| `cot` non-empty / distinct | **4,729 / 1,103** (100 % coverage, 23.3 % distinct) |
| **Alpamayo ∩ parity-train** | **201** (4.25 % of Alpamayo, 8.4 % of the parity corpus) |
| Alpamayo **not** in parity-train | **4,528** |
| parity-train **not** in Alpamayo | **2,199** |
| ⚠️ **Alpamayo ∩ canonical val40** | **6** |

**Two consequences, both load-bearing.**

1. ⚠️ **95.7 % of the v1 instance corpus sits OUTSIDE the parity corpus.** That is fine for
   *finding* scenarios — retrieval does not care — but it means an Alpamayo query is **not** a
   parity-corpus query. `corpus.in_parity` is therefore not a formality: it must be resolved per
   record at ingest with `clips_in_parity_train()` (`parity.py:1951-1959`), stored, exposed as a
   filter, and **shown in every result card**. Non-parity clips are stored with
   `split: "out-of-parity"` — **never silently dropped and never silently merged.**
2. ⛔ **6 Alpamayo clips are in the canonical val40 deployment set** — the episode set behind the
   published open-loop statistic. An unfiltered Alpamayo -> dataset path is therefore a **live
   contamination hazard**. `export_set` must refuse a training-role export containing a val40 clip,
   and that guard needs its own deliberate-regression arm (§8 SC-F).

`HYPOTHESIS` worth one probe, not asserted: **201 is exactly the `aug120` s2-label count**
(§3.7 table), which suggests the in-parity subset *is* the aug120 set. Probe: intersect the two
`clip_id` lists directly. It matters only for provenance bookkeeping, not for the design.

---

## 4. The data model

Three record types. `ScenarioType` re-specifies what exists; `DatasetLink` and `ScenarioInstance`
are new and are the substance of this spec.

### 4.0 Identity rules (binding)

- Every id is **stable and immutable**. Renaming never changes an id.
- `ScenarioType`: **human-assigned**, `SC-<nn>` for the opponent catalogue, `SG-<slug>` for
  generic classes that are not opponent-weakness entries — so `SC-` keeps its meaning.
- `DatasetLink`: `DS-<source_key>` where `<source_key>` is **the key in
  `lake/schema.py:SOURCE_REGISTRY`** when one exists (e.g. `DS-comma2k19`, `DS-physicalai_av`).
- `ScenarioInstance`: **deterministic and content-derived**, so any machine produces the same id:
  `SI-<source_key>-<clip_sha8>-<t0>-<t1>`, `t0`/`t1` integer frame indices at 10 Hz.
  ⛔ **`clip_sha8`, never the raw `clip_id`** — see §4.4.
- Every record carries `schema_version: int`. A reader meeting a newer version **fails loud**.

### 4.1 `ScenarioType` — the curated class record

```jsonc
{
  "schema_version": 1,
  "kind": "scenario_type",
  "id": "SC-13",
  "title": "Stationary-object / same-lane lead response",
  "aliases": ["stopped lead", "stationary object braking"],

  "description": {
    "free_text": "a stopped/slow lead vehicle or a stationary object in the ego lane ...",
    "correct_behavior": "early, smooth deceleration to a safe following distance.",
    "mechanism": "H15 imagination predicts the consequence of the closing gap ...",
    "authored": false            // false => regex-derived (parse.py:286-293), weaker evidence
  },

  "taxonomy": {
    "w_code": "W-08", "family": false, "stars": 2,
    "situation_classes": [],     // stack/tanitad/data/situations.py  (2 emitted, see 4.5)
    "tactical_tokens":   [],     // v6.py:217-223 / HIERARCHY_VOCABULARY.md:79-99
    "strategic_tokens":  []      // s2_derive.py:52-60
  },

  "evidence": {
    "label": "FACT",             // FACT | CLAIM | INFER          (parse.py:51)
    "label_raw": "FACT",
    "opponent_evidence": "NHTSA ODI opened an investigation (2026-05-08) into Avride ...",
    "links": ["https://techcrunch.com/2026/05/08/..."]
  },

  "lifecycle_stage": "live-measured",     // parse.py:39-46, or null
  "status_text": "...",
  "metric_hooks": ["OKRI", "LAL", "min-TTC"],
  "eval_registry_key": null,              // -> stack/tanitad/eval/scenarios/registry.py:58, or null
  "dataset_links": ["DS-comma2k19"],      // ids, not inline blobs

  "instance_selector": {                  // how instances are FOUND, never SELECTED
    "kind": "derived_label",              // derived_label | manual | none
    "predicate": {"longitudinal": ["Gentle Deceleration", "Strong Deceleration"]},
    "note": "a selector FILTERS a corpus; it never re-selects or reorders the parity corpus"
  },

  "embeddings": {
    "text": {"embedder_id": "minilm/all-MiniLM-L6-v2@<rev>/384/l2", "doc_sha256": "<sha256>"}
  },

  "provenance": {
    "source_doc": "SCENARIO_DATABASE.md",  // basename only (portability, parse.py:390-396)
    "source_doc_sha256": "<sha256 of the file as ingested>",
    "source_heading_line": 278,
    "ingested_at": "2026-08-23T11:04:00Z",
    "parse_warnings": []
  }
}
```

**Changes vs today:** `description.authored`, `taxonomy.*`, `eval_registry_key`, `dataset_links`
as ids, `instance_selector`, `embeddings.embedder_id + doc_sha256`, `provenance.source_doc_sha256`.
Everything else carries over from `parse.py:322-344` without a rename.

### 4.2 `DatasetLink` — licence-bearing, imported not invented

```jsonc
{
  "schema_version": 1,
  "kind": "dataset_link",
  "id": "DS-comma2k19",
  "source_key": "comma2k19",          // MUST be a key of lake/schema.py:SOURCE_REGISTRY (68-148)
  "name": "comma2k19",
  "url": "https://github.com/commaai/comma2k19",
  "hf_repo": null,                    // exact repo id when one exists; never guessed

  "licence": {                        // MIRRORED from SourceLicense (schema.py:47-61) - not authored
    "license_class": "owned-safe",    // owned-safe | nc-research | gated-confidential | refuse
    "license_name": "MIT",
    "share_alike": false,
    "is_synthetic": false,
    "commercial_ok": true,            // derived property, schema.py:58-61
    "tier": "ship",                   // lake/filtering.py:29 tier_of(...)
    "resolved_from": "tanitad.lake.schema.SOURCE_REGISTRY",
    "resolved_at": "2026-08-23"
  },

  "reachability": {"status": "not-probed", "probed_on": null,
                   "probe": "byte-pull of one shard with the Sayood token"},
  "size": {"n_episodes": null, "n_clips": null, "hours": null, "provenance": null},
  "parity_role": "neither",           // train | val | eval-lane | neither
  "publication": {"ids_publishable": true}   // false for gated-confidential - see 4.4
}
```

**Binding rules.**

1. `licence` is **mirrored from `SOURCE_REGISTRY`, never authored in TanitScena.** A source that
   is not in the registry gets `license_class: null` and is **not exportable**.
2. ⛔ **`license_class == "refuse"` sources (`waymo`, `waymax`) may be catalogued as opponent
   evidence but may never appear in a `DatasetLink` used by an export.** `schema.py:37-43`: their
   terms follow the trained weights, so no tier can contain them.
3. ⛔ **`gated-confidential` (`physicalai_av`) sets `ids_publishable: false`**, which forces the
   id-redaction rule of §4.4 everywhere that source's instances are rendered or exported.

### 4.3 `ScenarioInstance` — the new record type

```jsonc
{
  "schema_version": 1,
  "kind": "scenario_instance",
  "id": "SI-physicalai_av-923e09b1-000240-000296",   // <source_key>-<clip_sha8>-<t0>-<t1>

  "scenario_type": "SC-13",          // nullable: an unclassified instance is still indexable
  "match": {"how": "derived_label", "confidence": null,
            "deriver": "tac_a1_alpamayo_taxonomy.py@<sha>"},

  "corpus": {
    "dataset_link": "DS-physicalai_av",
    "corpus_key": "physicalai-train-e438721ae894",   // parity.py:78
    "skip_hash": "f09e44db",                          // parity.py:80
    "split": "train",                                 // train | val | eval-lane | out-of-parity
    "in_parity": true                                 // clips_in_parity_train(), parity.py:1951-1959
  },

  "locator": {
    "clip_sha8": "923e09b1",         // sha256(clip_id)[:8]  <- THE published join key
    "clip_digest": "923e09b1...",    // sha256(clip_id) full, parity.py:1858-1864
    "clip_id": null,                 // populated ONLY when dataset_link.ids_publishable is true
    "episode_uid": "ep_00417.pt",    // OPTIONAL, positional, meaningless without corpus_key
    "t0": 240, "t1": 296,            // half-open [t0, t1), integer frames
    "hz": 10.0,                      // situations.py:77
    "t0_s": 24.0, "t1_s": 29.6       // derived, for humans; never a join key
  },

  "labels": {
    "alpamayo": {"longitudinal": "Gentle Deceleration", "lateral": "Steer Right",
                 "lane": "Lane Keep"},
    "situation": [],                 // lane_change | intersection  (only 2 are emitted, 4.5)
    "tactical":  {"a_tac_lat": null, "a_tac_lon": null},   // low-confidence, see 4.5 warning
    "strategic": {"g_str": null, "a_str": null},
    "label_provenance": [
      {"deriver": "…/tac_a1_alpamayo_taxonomy.py", "deriver_sha": "<sha>",
       "artifact": "raw/a1_alpamayo_taxonomy_per_clip.jsonl", "n_rows": 4729,
       "inputs_used": ["records.parquet:meta_action"], "derived_at": "2026-08-16"}
    ]
  },

  "description": {
    "text": "Slow down due to the lead vehicle ahead.",
    "source": "alpamayo_cot",        // alpamayo_cot | scenario_description | template | vlm | human
    "model": null,
    "generated_at": "2026-08-16"
  },

  "admissibility": {
    "label_uses_ego": true,          // ALWAYS allowed - labels are built offline
    "inference_inputs": [],          // MUST stay [] or vision-only for any deployable use
    "note": "BINDING: labels may use ego; inference is vision-only (CLAUDE.md)."
  },

  "embeddings": {"text": {"embedder_id": "...", "doc_sha256": "..."}, "visual": null},
  "media": {"preview": null, "resim_bundle": null}
}
```

### 4.4 The parity join key — `clip_id`, published as `clip_sha8`

**MEASURED** `stack/tanitad/data/parity.py`. Two uid spaces exist, and only one of them joins
across caches:

| space | form | constant | note |
|---|---|---|---|
| raw epcache | `ep_%05d.pt` basename | `EPCACHE_UID_KIND = "epcache_basename"` (`:105`), `EPISODE_GLOB = "ep_*.pt"` (`:97`) | **positional**; `parity.py:34-37` calls the index *"the canonical, stable identity of an episode **within a build key**"* — meaningless without `corpus_key`, and **absent from v2 caches** |
| v2 compressed | `<clip_id>.v2ep.pt` | `V2_UID_KIND = "v2ep_clipid"` (`:104`), `V2_SUFFIX = ".v2ep.pt"` (`:103`) | `clip_id` is a UUID string — the identity that survives every cache boundary |

Corpus constants (`parity.py:78-82`): `PARITY_TRAIN_KEY = "physicalai-train-e438721ae894"` ·
`PARITY_VAL_KEY = "physicalai-val-0c5f7dac3b11"` · `PARITY_SKIP_HASH = "f09e44db"` (24-corrupt-clip
skipset marker) · `PARITY_TRAIN_EPISODES = 2376` · `PARITY_VAL_EPISODES = 600`.

**⇒ `ScenarioInstance.locator` keys on `clip_id`.** Ranked, with the functions to call:

1. **`clip_digest(clip_id) = sha256(clip_id).hexdigest()`** (`parity.py:1858-1864`) and its short
   form **`clip_sha8`** (first 8 hex — the field name already banked per-episode in
   `stack/tanitad/data/deployed_val40_clip_digests.json`). Membership oracle:
   **`clips_in_parity_train(clip_ids)`** (`parity.py:1951-1959`).
2. **`corpus_key`** — mandatory alongside, because `episode_uid` is positional.
3. `episode_uid` — optional convenience only.
4. `t0`/`t1` frame indices at `HZ = 10.0` (`situations.py:77`).

**Canonical digest of a set:** `uid_digest(uids) = sha256("\n".join(sorted(uids)))`
(`parity.py:127-137`), whose docstring states *"Canonical serialization is fixed here and nowhere
else; changing it invalidates every committed manifest."* **TanitScena calls it; it does not
re-implement it.** Parity assertion for a set of eids: `assert_eids_parity(...)` (`parity.py:529`).

**Manifests** (all under `stack/tanitad/data/`): `parity_manifest.json`
(`schema: "tanitad.parity_manifest/1"`; per-corpus `episode_uid_sha256`, `episode_uids[]`,
`skip_indices[]`, and a `clip_membership` block with `n_clips: 2400`, `clip_id_sha256_sorted`,
`discovered_clips: 3000`, `val_clips: 600`, `decode_failures: 24`) ·
`parity_train_clip_digests.json` (**digests only** — the ids are gated-confidential) ·
`deployed_val40_clip_digests.json` (`n_clips: 40`, the canonical TanitEval deployment).
Generators: `stack/scripts/make_parity_manifest.py` (modes `--from-profile-csv` `:96`,
`--record` `:195`, `--verify` `:252`; cross-check constants `:75-79`) and
`make_parity_clip_digests.py`.

⛔ **THE REDACTION RULE (binding).** `physicalai_av` is `license_class: "gated-confidential"`
(`schema.py:68-148`), and `parity_train_clip_digests.json` states the ids are gated-confidential
and banks **digests only**. ⇒ **TanitScena stores and renders `clip_sha8`/`clip_digest`. A raw
`clip_id` may be materialised in the store only when the source's
`DatasetLink.publication.ids_publishable` is true, and may NEVER appear in an export, an API
response, a UI view, or a log line for a gated-confidential source.** `verify` (§7.4) scans for
UUID-shaped strings on gated sources and fails.

⚠️ **The leaky val build.** `physicalai-val-f1b378f295ae` is flagged known-leaky at
`parity.py:24-27`; the canonical val key is `physicalai-val-0c5f7dac3b11` (`:79`). Ingest refuses
the leaky key outright.

### 4.5 Label vocabularies — consumed, never invented, and honestly weighted

TanitScena imports four vocabularies and stores tokens verbatim with the deriver's identity. It
**never** invents a label. An unknown token is a **hard ingest error**.

**(a) Situation classes — `stack/tanitad/data/situations.py`.**
Detectors: `detect_lane_change` (`:269`), `detect_roundabout` (`:318`), `detect_intersection`
(`:392`, whose turn half is `detect_turns` `:352`), `detect_curves` (`:373`).
Input is the ego pose track only — `P = [x, y, yaw, v]` at 10 Hz through `kinematics(...)` (`:159`).
⚠️ **Only TWO labels are emitted: `lane_change` and `intersection`.** `roundabout` is computed but
**not emitted** (26 held-out clusters = unpowered, PI deferred) and `curve` is a **control
population** for PRE-REG §6.2, not a situation label. Thresholds (`:77-117`) are **frozen by
PRE_REGISTRATION.md §2 and must not be swept** — TanitScena must never re-derive with different
thresholds. The target is anticipation, not recognition: `anticipation_target(...)` (`:409`).

⛔ **DEFECT D5 — the emitted situation labels cannot be joined.** `MEASURED`
`stack/scripts/emit_situation_labels.py:63`: the NPZ's `episode` column is written as
`np.full(T, i, np.int32)` where `i` is **the enumeration index of `sorted(glob(...))`** — not a
uid, not a clip id. The output (`np.savez_compressed`, `:70-73`) carries
`episode, t, y_lane_change, valid_lane_change, y_intersection, valid_intersection, lead_s, hz`
and a `_summary.json` sidecar. ⇒ **the situation labels cannot be joined back to a clip without
re-deriving the identical sorted file list.** This is the same class as the `eid` normalisation
defect the branch already fixed. **Blocking for situation facets; backlog `P5-10`.**

**(b) Strategic — `stack/scripts/s2_derive.py`.** ⚠️ It is a **library, not a CLI** (no `argparse`,
no `__main__`). Public API: `derive_g_str(...)` (`:277`), `derive_a_str(...)` (`:436`),
`lane_change_requirement(...)` (`:219`), `check_action_geometry(...)` (`:586`),
`check_vocab_drift()` (`:106`).
- `STRATEGIC_GOAL_TOKENS` (11, `:52-56`): `KEEP_CORRIDOR, LANE_TARGET, EXIT_RIGHT, EXIT_LEFT,
  TURN_LEFT, TURN_RIGHT, STRAIGHT_THROUGH, ROUTE_TO, STOP_AT, FOLLOW_MAIN_ROAD, NONE_ABSTAIN`
- `STRATEGIC_ACTION_TOKENS` (6, `:57-60`): `PREPARE_LANE_CHANGE, HOLD_CORRIDOR, REDUCE_TO,
  PREPARE_EXIT, PREPARE_STOP, RESUME_CRUISE`
- `GOAL_ARG_NAMES` (8 float slots, `:61-62`); `PROVENANCE_CLASSES = ("path","signage","vlm-fused")`
  (`:65`)
- ⛔ **Three tokens are in the vocabulary but never emitted — a facet UI must not offer them:**
  `LANE_TARGET` removed from `g_str` entirely (`:171-216`, `:284-287`; PI adjudicated 14/18 wrong);
  `ROUTE_TO` gated and remapped, and the validator **refuses** it
  (`colab/s2_schema.py:239-242`); `PREPARE_LANE_CHANGE` needs `lane_context`, which is `None` on
  **801/801** clips (`:211-216`, `:515-523`).
- Output schema is authoritative in `colab/s2_schema.py` (`SCHEMA_VERSION = "s2-strategic-v1"`,
  `:46`; `build_record()` `:144-172`): record keys `schema_version, clip_id, t0_s, g_str, a_str,
  valid_window_s, disjointness, _provenance`; each block carries
  `token, token_id, args[8], arg_mask[8], provenance, sources[], confidence, corroboration`.
  `T0_S_DEFAULT = 8.0`, `VALID_WINDOW_S_DEFAULT = (-2.0, 2.0)` (`:68-69`).
- ⛔ **The canonical label directory is `review/labels_v2/`, NOT `labels/`** — `labels/`
  carries a `SUPERSEDED.json` redirect, pinned as `S2_CANONICAL_LABELS_REL` at
  `stack/scripts/s2_labels.py:186-188`. Counts: **201** (`aug120`) + **596** (`w120val`).
- **Reuse the existing loader**: `load_s2_labels(path, role="train")` (`s2_labels.py:528`),
  `S2Row` (`:328-349`), `S2LabelSet` (`:352`), superseded-refusal (`:494`).

**(c) Tactical — `HIERARCHY_VOCABULARY.md` + `v6.py`.**
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-07-hierarchical-wm-redesign/HIERARCHY_VOCABULARY.md:79-99`.
9 `g_tac` goal tokens (`:82-92`): `ANCHOR_GOAL, CORRIDOR_OFFSET, GAP_TARGET, SPEED_BAND, YIELD_AT,
STOP_POINT, WAIT_FOR_ONCOMING, EVADE_IN_CORRIDOR, TRAFFIC_LIGHT_REACT`.
`a_tac` (`:95-96`): LAT `LANE_KEEP · LANE_CHANGE_L/R(within_m) · ABORT_LC · NUDGE_L/R(lat_m)`;
LON `FOLLOW(time_gap_s) · CRUISE(v) · YIELD/MERGE(gap_slot) · BRAKE_TO(v, within_m) · CREEP ·
HOLD`. The code splits the goals into two disjoint tuples at `stack/tanitad/models/v6.py:217-223`
(`TACTICAL_GOAL_TOKENS_LAT` = 4, `TACTICAL_GOAL_TOKENS_LON` = 6). Band: `TAC_BAND_S = (2.0, 6.0)`
(`v6.py:136-140`; the doc's "2-8 s" heading is superseded at `HIERARCHY_VOCABULARY.md:113-114`).
The emitter is the fuser `stack/scripts/ph1_fuse.py` (`SCHEMA = "ph1-fused-v1"`, `:55`), which
emits `a_tac_lat`/`a_tac_lon` by block vote (`:772-787`) and emits `g_tac_lat`/`g_tac_lon`
**empty with an `unavailable_reason`** (`:253-262`, `:794`).

⛔ **Load-bearing caveat — the tactical facet is EXPERIMENTAL, and the SPEC says so on the record.**
`…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md` reports that tactical labels are **not
demonstrated buildable at the band the architecture uses**: **LON κ 0.1428 [0.0540, 0.2250] n=201**
and **LAT κ 0.1777 [0.0658, 0.2953] n=193** over `TAC_BAND` (2-6 s). ⇒ TanitScena stores tactical
tokens but **marks the facet `experimental: true` in `/api/facets` and shows the κ and n in the
UI**. It must never be presented as a reliable filter. *(There is also no tactical loss term in
the trainer — `train_v6_staged.py:151-217`.)*

**(d) Alpamayo `meta_action` — the highest-coverage vocabulary we have.**
Produced by `…/2026-08-16-tactical-labels/code/tac_a1_alpamayo_taxonomy.py`
(CLI `--records <records.parquet> --out <json>`, `:28-29`, `:60-63`). Three **independent 7-value
axes**, n = **4,729**, **0 unparsable**:
- **longitudinal**: Gentle Deceleration 33.7 % · Maintain Speed 25.9 % · Gentle Acceleration
  24.3 % · Stop 6.4 % · Strong Deceleration 5.6 % · Strong Acceleration 3.8 % · Reverse 0.13 %
- **lateral**: Go Straight 53.0 % · Steer Right 21.6 % · Steer Left 13.7 % · Sharp Steer Right
  2.9 % · Sharp Steer Left 2.4 % · Reverse Right · Reverse Left
- **lane**: Lane Keep 85.3 % · Turn Right 2.1 % · Turn Left 1.8 % · Right Lane Change 1.7 % ·
  Slightly Shift Left 1.5 % · Slightly Shift Right 0.66 % · Left Lane Change 0.47 %

⇒ **this is the v1 facet set for instances**, because it is the only vocabulary with full coverage
over a real clip population and it ships with its own text field (`cot`).

**Generated check:** `verify` re-resolves every stored token against its owning source, so an
upstream rename is caught at verify time rather than silently splitting the index.
`VOCABULARY.md` governs naming (`TANITAD_PROGRAMME.md:142-149`).

⚠️ **Situation labels are derived from ego dynamics.** TanitScena is a retrieval system, not a
classifier, so the ego-leak rule does not bind it directly — **but any instance later used to
train or evaluate a head must carry `admissibility.inference_inputs`** so the leak test can be
run. That is why the field exists in §4.3.

### 4.6 Deliberately NOT in the schema

- **No pixels, features or tensors.** Keys and text only. (⚠️ `data/` is git-ignored repo-wide
  (`.gitignore:16`), so a store directory must **never** be named `data/`.)
- **No metric values.** Results live in `MODEL_REGISTRY.md` and raw JSON; a record may hold a
  pointer, never a number. A copied number is the failure mode `CLAUDE.md` opens with.
- **No eval strata** (§2.2).
- **No raw `clip_id` for gated-confidential sources** (§4.4).

---

## 5. Embedding & vector-search design

### 5.1 What gets embedded

Two document builders, one per record type, **both TEXT**.

```
doc_text(ScenarioType)     = title + aliases + description.free_text
                           + description.correct_behavior + description.mechanism
                           + taxonomy tokens + metric_hooks + dataset names
doc_text(ScenarioInstance) = description.text                       # the Alpamayo `cot` sentence
                           + alpamayo{longitudinal, lateral, lane}  # full-coverage facets
                           + situation/strategic tokens when present
                           + scenario_type title when matched
                           + corpus name
```

The type builder **widens** today's (`vector.py:34-41` embeds only
`title + description + correct_behavior + tags`) to include `mechanism`, `metric_hooks` and the
dataset names — the terms users actually type. **Widening changes rankings, so it ships WITH the
retrieval eval of §8 SC-C, never before it.**

### 5.2 Which model

**Primary: `sentence-transformers/all-MiniLM-L6-v2`** — 384-d, ~80 MB, CPU inference. It is
already the declared upgrade path in the code (`vector.py:118-127`) and already handled end to end
(build, persist, query-encode), so adopting it costs **one extras entry in
`stack/pyproject.toml`** and no new code. It is small enough for a free HF CPU tier and needs no
GPU. **MEASURED: it is not installed on any interpreter on this box** (§3.3) — installing it is
backlog `P5-9`.

**Fallback and control arm: the existing deterministic hashing TF-IDF** (`vector.py:48-107`). It
stays for two reasons: offline CI (no download), **and as the anti-false-positive control** — if
MiniLM does not beat it on the frozen query set, the neural embedder is not earning its keep and
we keep the cheaper one. Both outcomes are pre-registered in §8 SC-C.

**Not chosen, and why:** no hosted embedding API (network at query time is excluded by design,
`scena/README.md:6-8`, and it is metered); no large embedder (§9 ceiling); no cross-encoder
re-ranker in v1 — a re-ranker is only worth adding once first-stage recall is **measured**.

### 5.3 Where the index lives — and why no vector DB

**Keep exact cosine over a numpy `float32` matrix. Do not add faiss / chroma / hnswlib / lancedb.**
(MEASURED: none of them is in `stack/pyproject.toml` or importable anywhere on the box, §3.3.)

`ESTIMATED` (arithmetic shown): a 384-d float32 vector is 384 x 4 = **1,536 B**.

| n records | matrix | one query |
|---|---|---|
| 10^2 | 154 KB | 38 K MACs |
| **4,729** (today's Alpamayo corpus) | **7.3 MB** | **1.8 M MACs** |
| 10^5 | 154 MB | 38 M MACs |
| 10^6 | 1.54 GB | 384 M MACs |

A 384 M-MAC BLAS `matmul` is sub-second on any modern CPU, and the matrix is RAM-resident to
~10^6 records. **Exact search is correct and sufficient to ~10^6 instances** — far above any
corpus we have. An ANN index buys approximation error plus a dependency, in exchange for nothing.
Re-open at n > 10^6, **or** when p99 query latency measured on the serving box exceeds 200 ms —
whichever comes first, and only with the measurement in hand.

**Layout** — one portable directory, no absolute paths (the invariant already tested at
`test_scena.py:240-253`):

```
<store>/                       # NOT named data/  -  .gitignore:16 would swallow it
  types.jsonl
  datasets.jsonl
  instances.jsonl              # append-mostly; fine to ~10^6 rows
  index/
    text.npz                   # ids + float32 matrix + meta + idf   (today's vectors.npz format)
    text.manifest.json         # {record_id: {doc_sha256, embedder_id}} + corpus_state_sha
  STORE.json                   # schema_version, counts, source hashes, created/updated
```

If `instances.jsonl` stops scaling, the migration is to SQLite with the **same record shape**, not
to a different model. Stating that now makes the migration boring.

### 5.4 A visual space — phase 2, and honestly

Text-to-image retrieval needs a **jointly trained** text/image space (CLIP/SigLIP class). Our
cached DINO features are **not** text-aligned and cannot serve a natural-language query. Two
admissible routes:

- **(a) caption-then-embed** — a short caption per clip enters the **existing text space**. One
  query space, no new index, human-auditable. ⭐ **We already have 4,729 such sentences for free**
  (§3.7), so route (a) is *already done* for v1 without any model run.
- **(b) a genuine second index** — a small CLIP/SigLIP pair in its **own** space, reported
  **separately**.

**Route (a) is the recommendation. Route (b) is deferred** until the retrieval eval shows text
retrieval is the bottleneck.

⛔ **Never fuse two spaces into one score** — the same "a single composite hides the trade-off"
failure the four-metric-families rule exists to prevent. If both exist, results are reported per
space.

### 5.5 Staying consistent as the corpus grows (fixes D1 and D2)

**Content-addressed index, not id-set-addressed.**

- `embedder_id = "<family>/<model>@<revision>/<dim>/<normalisation>"` — a full identity, so a
  model, revision or normalisation change invalidates the index.
- `doc_sha256` = sha256 of the exact embedded document string.
- `text.manifest.json` maps `record_id -> {doc_sha256, embedder_id}`.
- `refresh()` re-embeds exactly the records whose `doc_sha256` or `embedder_id` changed and
  appends new ids. **This fixes D1**: a text edit now invalidates its own vector with or without
  an id-set change.
- **The hashing embedder cannot refresh incrementally (D2** — corpus-global IDF, `vector.py:93-102`).
  Its manifest therefore also stores `corpus_state_sha` = sha256 over the sorted
  `(id, doc_sha256)` list; any change forces a **full refit**. Cheap at our n, and stated rather
  than discovered.
- `GET /api/meta` reports freshness explicitly:
  `{embedder_id, n_indexed, n_stale, n_missing, built_at}`. A UI serving a stale index says so.

---

## 6. The code API

Package stays at **`stack/tanitad/scena/`** (§13 Q6). New modules alongside `parse.py`/`vector.py`.

### 6.1 Records and store

```python
# stack/tanitad/scena/model.py
SCHEMA_VERSION: int = 1

@dataclass(frozen=True)
class ScenarioType:     ...   # §4.1;  .to_dict() / .from_dict(d) / .doc_text()
@dataclass(frozen=True)
class DatasetLink:      ...   # §4.2;  .from_source_key(key) mirrors lake.schema.SOURCE_REGISTRY
@dataclass(frozen=True)
class ScenarioInstance: ...   # §4.3

def validate(record: dict) -> list[str]:
    """Schema violations; empty list means valid. Never raises."""
```

```python
# stack/tanitad/scena/store.py
class ScenaStore:
    @classmethod
    def open(cls, root: str | Path, *, create: bool = False) -> "ScenaStore": ...
    def upsert_type(self, t: ScenarioType) -> str: ...
    def upsert_dataset(self, d: DatasetLink) -> str: ...
    def upsert_instances(self, rows: Iterable[ScenarioInstance], *, batch: int = 1000) -> int: ...
    def get(self, rid: str) -> dict | None: ...
    def iter_records(self, kind: str | None = None) -> Iterator[dict]: ...
    def counts(self) -> dict[str, int]: ...
    def close(self) -> None: ...
```

### 6.2 Query — semantic + structured in one call

```python
# stack/tanitad/scena/query.py
@dataclass(frozen=True)
class Filters:
    kind:            str | None = None            # "scenario_type" | "scenario_instance"
    scenario_type:   Sequence[str] | None = None
    source_key:      Sequence[str] | None = None  # lake.schema.SOURCE_REGISTRY keys
    corpus_key:      Sequence[str] | None = None
    split:           Sequence[str] | None = None  # train | val | eval-lane | out-of-parity
    in_parity:       bool | None = None
    license_class:   Sequence[str] | None = None  # owned-safe | nc-research | gated-confidential
    commercial_ok:   bool | None = None
    lifecycle_stage: Sequence[str] | None = None
    evidence_label:  Sequence[str] | None = None  # FACT | CLAIM | INFER
    alp_longitudinal: Sequence[str] | None = None # the 7-value Alpamayo axes (§4.5d)
    alp_lateral:      Sequence[str] | None = None
    alp_lane:         Sequence[str] | None = None
    situation:       Sequence[str] | None = None  # lane_change | intersection  (only 2 emitted)
    strategic:       Sequence[str] | None = None  # emitted tokens only - never the 3 refuted ones
    tactical:        Sequence[str] | None = None  # EXPERIMENTAL - carries kappa + n (§4.5c)
    min_stars:       int | None = None
    def as_mask(self, store: "ScenaStore") -> "np.ndarray": ...

@dataclass(frozen=True)
class Hit:
    id: str; kind: str; rank: int; score: float
    record: dict
    why: dict          # {"embedder_id":..., "matched_filters":[...], "top_terms":[...]}

def search(store, query: str, *, k: int = 20, filters: Filters | None = None,
           space: str = "text", embedder: str = "auto",
           explain: bool = False) -> list[Hit]: ...

def lookup(store, ids: Sequence[str]) -> list[dict]: ...
def neighbours(store, rid: str, *, k: int = 10, filters=None) -> list[Hit]: ...

def facets(store, filters: Filters | None = None) -> dict[str, dict]:
    """value -> count per facet under the current filter. Answers the 'counts' half of the
    /TanitAD_SearchScenarios contract (SKILLS_SPECS.md:51). Experimental facets carry
    {'experimental': True, 'kappa': ..., 'n': ...}."""
```

⚠️ **Filters are applied as a mask BEFORE top-k**, never as a post-filter of a fixed-k result — a
post-filter silently returns fewer than `k` and is the classic retrieval bug.

### 6.3 Index

```python
# stack/tanitad/scena/index.py
class ScenaIndex:
    @classmethod
    def build(cls, store, *, space: str = "text", embedder: str = "auto") -> "ScenaIndex": ...
    def refresh(self, store) -> dict:
        """Content-addressed incremental re-embed (§5.5).
        -> {'n_reembedded','n_added','n_removed','full_refit'}"""
    def freshness(self, store) -> dict: ...           # {n_indexed,n_stale,n_missing,built_at}
    def search_vec(self, qvec, k: int, mask=None) -> list[tuple[str, float]]: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "ScenaIndex": ...
```

`VectorIndex` (`vector.py:134`) is kept and becomes `ScenaIndex`'s single-space engine — the
cosine core is not rewritten.

### 6.4 Export — the P8 handoff

```python
# stack/tanitad/scena/export.py
class ScenaExportError(RuntimeError): ...

def export_set(store, hits_or_ids, out: str | Path, *,
               name: str,
               fmt: str = "jsonl",                                   # jsonl | csv
               allow_license: Sequence[str] = ("owned-safe", "nc-research"),
               allow_share_alike: bool = True,
               parity_check: bool = True,
               redact_ids: bool = True) -> Path:
    """Emit a ScenarioSet: <out> (rows) + <out>.manifest.json (provenance).

    Raises ScenaExportError when:
      - any row's license_class is null, "refuse", or outside `allow_license`;
      - `redact_ids` and a gated-confidential row would emit a raw clip_id (§4.4);
      - `parity_check` and the implied episode selection differs from the parity manifest
        (uid_digest / assert_eids_parity, parity.py:127-137 / :529);
      - any row's schema_version exceeds this reader's SCHEMA_VERSION.
    """
```

The manifest carries: query string, `Filters`, `embedder_id`, index `built_at`, `STORE.json`
hashes, per-row `license_class` + `tier`, `corpus_key` + `skip_hash`, the `uid_digest` of the
selection, and the tool version. **P8 consumes the manifest, not the rows alone.**

### 6.5 CLI

```
python -m tanitad.scena ingest-types      --db-md <SCENARIO_DATABASE.md> --store <dir>
python -m tanitad.scena ingest-datasets   --from-lake-registry           --store <dir>
python -m tanitad.scena ingest-instances  --alpamayo <a1_alpamayo_taxonomy_per_clip.jsonl>
                                          --source-key physicalai_av
                                          --corpus-key physicalai-train-e438721ae894 --store <dir>
python -m tanitad.scena ingest-instances  --s2-labels <review/labels_v2/s2_labels_aug120.jsonl> ...
python -m tanitad.scena index             --store <dir> [--embedder auto|minilm|hashing] [--refresh]
python -m tanitad.scena search  "<query>" --store <dir> [-k 10] [--filter license_class=owned-safe]
                                          [--filter alp_longitudinal="Gentle Deceleration"]
                                          [--json] [--explain]
python -m tanitad.scena facets            --store <dir> [--filter ...]
python -m tanitad.scena export            --store <dir> --query "<q>" -k 500 --name <set> --out <p>
python -m tanitad.scena verify            --store <dir>
```

`search --json` prints one JSON object per line so a skill or a shell pipeline consumes it without
a parser.

### 6.6 HTTP API (extends `scena_app.py:116-204`)

```
GET  /                        SPA
GET  /static/*                SPA assets
GET  /api/meta                embedder_id, freshness, counts, warnings, schema_version
GET  /api/facets?<filters>    facet -> value -> count (+ experimental flags)
GET  /api/search?q=&k=&<filters>&explain=      ranked hits, filters applied PRE-top-k
GET  /api/record/{id}         any record (guard: ^(SC|SG|DS|SI)-[A-Za-z0-9._-]+$)
GET  /api/datasets            DatasetLink list with license_class + tier
POST /api/export              {query, filters, k, name} -> ScenarioSet (licence + parity gated)
POST /api/reindex             full re-parse + refresh
GET  /api/scenarios           KEPT for backward compatibility
GET  /api/scenario/{sid}      KEPT, guard stays ^SC-\d+$   (scena_app.py:46)
```

Backward compatibility is a requirement, not a courtesy: `test_scena.py:167-232` drives those two
endpoints and must stay green.

---

## 7. The UI

### 7.1 Stack — keep what exists

**Vanilla JS + FastAPI + uvicorn, one plain-HTTP port, no build step, no CDN, no network at query
time.** Already implemented (`scena_app.py`, `scena/static/`) and correct here: proxy-friendly
(a single RunPod/HF-Space port), no toolchain to rot, and it shares TanitResim's design language
(`scena/README.md:6-10`). **MEASURED:** `fastapi`, `uvicorn`, `httpx` are all importable in the
project venv (§3.3) — the server runs on the dev box today with no install.

⛔ **Do not introduce React/Vite/Tailwind.** A build step means a `node_modules` that is not in the
repo, which means the UI is not servable from a fresh checkout — the exact stranding failure
`CLAUDE.md` §"Finish before you start" is about.

### 7.2 What the UI must do

1. **Search-first home** — the existing query box and ranked score-bar cards, **plus** a
   **server-driven facet rail** (`/api/facets`) over corpus · licence class · split · Alpamayo
   longitudinal/lateral/lane · situation · strategic · lifecycle stage · evidence label · stars.
   Facets show **counts** — half of the `/TanitAD_SearchScenarios` contract
   (`SKILLS_SPECS.md:51`).
2. **Two result modes**, toggled: **Types** (today's curated cards) and **Instances** (new:
   corpus, `clip_sha8`, window, labels, matched type, a "why" chip).
3. **Type detail** — unchanged, plus an "instances of this type" panel with counts per corpus and
   licence, and a link to `eval_registry_key` when the scenario is executable (§2.3).
4. **Instance detail** — locator, labels with their deriver and artifact, description with its
   `source` badge (`alpamayo_cot` / `scenario_description` / `template` / `vlm` / `human`), and a
   replay affordance when a TanitResim `.rrd` exists.
5. **Export basket** — accumulate hits, name the set, `POST /api/export`. **The licence gate is
   visible**: a basket containing a `refuse` or null-licence source cannot be exported, and the UI
   names the blocking record.
6. **Freshness banner** — when `/api/meta` reports `n_stale > 0`, say so in the header. A silent
   stale index is D1 in a new costume.
7. **Explain-this-hit** — with `explain=1`, show `embedder_id`, matched filters, top contributing
   terms. Retrieval that cannot be interrogated cannot be debugged.
8. ⛔ **Experimental facets are labelled in the UI** with their κ and n (§4.5c). A filter that is
   near-chance must not look like a filter that works.

### 7.3 Being realistic about what this repo can serve

- **No video.** `*.mp4` is git-ignored (`.gitignore:22`). Degrade to the schematic BEV canvas +
  metadata whenever media is absent; never render a broken thumbnail.
- **No dataset bytes.** Dataset links open the source's landing page; TanitScena never proxies a
  download.
- ⛔ **No raw ids for gated-confidential sources** — the UI renders `clip_sha8` (§4.4).
- **Deployment target is open** (§13 Q1). The app is a plain ASGI app on one port, so it runs on
  the dev box, on a pod behind the RunPod HTTP proxy (`scena/README.md:91-109`), or in an HF
  **CPU** Space. ⛔ Never `@spaces.GPU`.

---

## 8. Success criteria — measurable

The charter bar is *"TanitScena answers a semantic query end-to-end"* (`INHERITED` from the
briefing — see §10.3 item 1). Made concrete, with the test that proves each. **Both outcomes are
pre-registered where a criterion can fail.**

| id | criterion | pass condition | test |
|---|---|---|---|
| **SC-A** | **the named end-to-end query, types** | `python -m tanitad.scena search "school bus stop arm and an occluded child crossing" --store <s> -k 5 --json` returns **`SC-04` at rank 1** on **both** embedder paths, **no network**, exit 0, valid JSON. ⭐ **The retrieval half is already MEASURED (2026-08-23): SC-04 at rank 1, score 0.4715, hashing path** (§12). What is missing is only the CLI surface | `tests/test_scena.py::test_cli_search_end_to_end` (new). The library-level equivalent exists today at `test_scena.py:131` |
| **SC-B** | **the named end-to-end query, instances** | `search "slow down for the vehicle stopped ahead" --filter kind=scenario_instance -k 20` returns **>= 20** hits over the **4,729-row** Alpamayo corpus; **every** hit carries a `clip_sha8` **and** an `in_parity` verdict from `clips_in_parity_train()`; **no raw `clip_id` appears in the output**; and — because only **201/4,729** are in-parity (§3.7) — the result payload states `n_in_parity` and `n_out_of_parity` explicitly rather than leaving the caller to assume | `tests/test_scena_instances.py::test_instance_search_end_to_end` |
| **SC-C** | **the embedder earns its keep** (control arm) | on a **frozen query set of >= 30** `query -> relevant-id` pairs, report **Recall@5** and **MRR** for `minilm` and `hashing-tfidf`, **n printed**. **Both outcomes pre-registered:** if MiniLM's Recall@5 exceeds hashing's, MiniLM becomes the default; **if it does not, we keep hashing-tfidf and record that** — no neural embedder adopted on vibes | `tests/test_scena_retrieval_eval.py` + banked `raw/retrieval_eval.json` |
| **SC-D** | **no stale index** (fixes D1) | edit one description, restart the server **without** `/api/reindex`; the served vector for that record **changes**, and `/api/meta` reported `n_stale >= 1` before the refresh | `tests/test_scena.py::test_content_addressed_index_invalidates` |
| **SC-E** | **licence travels** | `export_set` **raises** when a row's `license_class` is null or `"refuse"` or outside `allow_license`; the manifest carries `license_class` + `tier` for **every** row; the values are the ones `lake/schema.py` returns, not authored strings | `tests/test_scena_export.py::test_export_refuses_unlicensed` |
| **SC-F** | **parity is sacred, and val40 is not contaminable** | `export_set(..., parity_check=True)` **raises** when the implied selection differs from `parity_manifest.json`, **and raises when a training-role export contains any of the 40 canonical val40 clips** — MEASURED: 6 Alpamayo clips are in val40 (§3.7), so this is a live hazard, not a hypothetical. **Two deliberate-corruption arms prove both checks CAN fail** | `tests/test_scena_export.py::{test_parity_guard_can_fail, test_val40_contamination_guard_can_fail}` |
| **SC-G** | **no gated id leaks** | for a `gated-confidential` source, **no** UUID-shaped string appears in any API response, export row, manifest, or log; a deliberate-injection arm proves the scanner can fail | `tests/test_scena_export.py::test_gated_id_redaction_can_fail` |
| **SC-H** | **latency** | warm `/api/search` p99 **< 200 ms** at the store's current n, on the serving box, with n and hardware printed | `tools/` bench script + banked JSON |
| **SC-I** | **offline** | the whole suite passes with **no network** (the hashing path needs only numpy) | already the standard, `test_scena.py:1-17` |

**SC-F and SC-G carry deliberate-regression arms on purpose** — `TANITAD_PROGRAMME.md:161-163`:
a guard must be shown ABLE TO FAIL before its PASS means anything.

**Milestone definition of done: SC-A + SC-B + SC-E + SC-F + SC-G green.**
SC-C decides the embedder; SC-D and SC-H are correctness/perf hygiene.

---

## 9. Compute budget and ceilings

- ⛔ **HF Pro quota is a HARD CEILING** (`TANITAD_PROGRAMME.md:24-28`, PI 2026-08-22: *"no chance
  to exceed it, I would never approve it"*). Verify remaining quota **before** any metered job; if
  it is unknown, **do not start — ask**.
- **Default compute is local and unmetered:** dev box **RTX 4060, 8188 MiB, driver 591.86**
  (MEASURED 2026-08-23) and its CPU.
- **Every core TanitScena path is CPU-only** — parsing, hashing TF-IDF, MiniLM over ~10^4 short
  documents, exact cosine, the server, the UI.
- ⭐ **v1 needs no model run at all.** The instance text is the existing 4,729 `cot` sentences
  (§3.7). A GPU pass is a *later option*, not a prerequisite.
- ⛔ **No `@spaces.GPU` in any TanitScena code.** If deployed to an HF Space, it is a **CPU** Space.
- **UNVERIFIED:** the briefing cites an HF Space with "16 CPU, ~97 GiB RAM". The only in-repo
  statement about our Space is `TANITAD_PROGRAMME.md:30` — *"private Space `Sayood/TanitAD` —
  Gradio, zero-a10g ZeroGPU, RUNNING"* — which is a **ZeroGPU** Space, and **ZeroGPU is metered**.
  Probe before deploying: the Space hardware settings page, or
  `huggingface_hub.get_space_runtime("Sayood/TanitAD")`. **Do not assume the CPU allowance.**

---

## 10. Interfaces

### 10.1 Consumes

| from | what | how |
|---|---|---|
| `TanitAD Research Hub/Opponent Analyzer/SCENARIO_DATABASE.md` | the 14 `SC-xx` records | `parse_file()` (exists) + a source-doc sha256 |
| `stack/tanitad/lake/schema.py` | **`SOURCE_REGISTRY` + `SourceLicense`** | `DatasetLink.from_source_key()` mirrors it; never authored |
| `stack/tanitad/data/parity.py` | corpus keys, `clip_digest`, `uid_digest`, `clips_in_parity_train`, `assert_eids_parity` | the join key and the `verify`/`export` guards |
| `stack/tanitad/data/parity_manifest.json` + `parity_train_clip_digests.json` | episode/clip membership | parity check on export |
| `…/2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy_per_clip.jsonl` | **4,729** instances + `cot` text + 3x7 facets | the v1 instance corpus |
| `…/2026-08-16-s2-v1-labels/review/labels_v2/s2_labels_*.jsonl` | strategic tokens (201 + 596), keyed on `clip_id` | via `load_s2_labels()` (`s2_labels.py:528`) |
| `stack/scripts/ph1_fuse.py` output (`ph1-fused-v1`) | `a_tac_lat`/`a_tac_lon` + `scenario_description` | experimental facet + a second text source ⚠️ **not in git**, §10.3 item 4 |
| `stack/tanitad/data/situations.py` + `emit_situation_labels.py` | `lane_change`, `intersection` | ⛔ **blocked by defect D5** (§4.5a) |
| `stack/tanitad/eval/scenarios/registry.py` | `SCENARIO_REGISTRY` keys | `ScenarioType.eval_registry_key` pointer |
| P3 TanitDataReconstruction | new corpora | a new `DatasetLink` + instances |

### 10.2 Feeds

| to | what |
|---|---|
| **P8 TanitDataSetCreator** | a **ScenarioSet** (rows + manifest) from `export_set`; P8 registers the built dataset **back** as a `DatasetLink` (`SKILLS_SPECS.md:45-46`) |
| **`/TanitAD_SearchScenarios`** | `search --json` and `facets` — the skill's whole procedure (`SKILLS_SPECS.md:48-51`) |
| **`/TanitAD_DesignDataSet`** | the licence check and the scenario mix (`SKILLS_SPECS.md:44-46`) |
| **P7 TanitEval** | *pointers only* — a scenario id a stratum may reference. Never a stratum definition (§2.2) |
| **Opponent Analyzer** | coverage reporting: which `SC-xx` have **zero** instances in any licensed corpus. That gap list currently lives in prose (`SCENARIO_DATABASE.md:274`) |

### 10.3 Integration escalations (reported here, per the operating standard — not written into a doc and left)

1. **`Project Steering/AGENT_CHARTERS.md` does not exist in this worktree.** Probed two ways: a
   full listing of `Project Steering/` (69 entries, no charters file) and
   `git ls-files | grep -i charter` (no match, taken before git became unreadable, §12). The brief
   cites its §0 and §2 as required reading, and the success bar quoted in §8 comes from the brief.
   **Either the file is unmerged on another branch, or the citation is stale.** This SPEC is
   written against `TANITAD_PROGRAMME.md` + `SKILLS_SPECS.md` instead.
2. **P5 is absent from the live register.** MEASURED: neither `Project Steering/BACKLOG.md`
   (18,296 chars) nor `Project Steering/GOALS_AND_CLAIMS.md` (5,380 chars) contains `scena`,
   `scenario`, `P5`, `semantic search` or `vector`. The only P5 mention in `Project Steering/`
   outside the programme doc is `VOCABULARY.md:15`. **P5 work items need to enter the register.**
3. **Three scenario modules are stranded** outside `stack/` and cannot be imported (§2.3):
   `stop_arm_gate.py` (SC-04), `stationary_lead.py` (SC-13), `emergency_scene.py` (SC-06). They
   carry passing offline test counts in their INTAKE notes and have been "awaiting orchestrator
   triage" since 2026-07-24 / 07-31 / 08-07.
4. ⚠️ **The fused PH1 corpus is not in git.** Per
   `…/2026-08-15-aug120-fusion/MANIFEST.md`, the 201 `aug120` fused records (+600 `w120val`) live
   on **HF `Sayood/tanitad-ph0-aug120/fused_aug120/`** (204 files, 4.26 MB); only aggregate JSON
   plus **60** sample records are in-repo. Its `scenario_description` field is a text source
   TanitScena wants. **Banking it is a P2/P3 item, flagged here because P5 depends on it.**
5. **`l2d` (`yaak-ai/L2D`, Apache-2.0) was surveyed and never integrated.**
   `…/2026-07-11-semantic-label-survey` measured 4,219 distinct compositional nav commands over
   100 k episodes / 26.5 M frames and recommended it; its orchestrator verdict block is **still
   unfilled**. It is already in `SOURCE_REGISTRY` as `owned-safe`/Apache-2.0. A strong future
   `DatasetLink`.
6. **`/TanitAD_SearchScenarios` is spec'd but not built** (`SKILLS_SPECS.md:48-51`); no
   `.claude/skills/TanitAD_SearchScenarios/` exists. Backlog `P5-13`.
7. **The P8 spec is being written concurrently** (`products/P8-datasetcreator/`, empty as of
   13:08 today). **The two specs must agree on the ScenarioSet manifest shape (§6.4).**

---

## 11. Invariants (binding)

1. ⛔ **Parity is sacred.** An instance is a **view** over the corpus: it stores a key into the
   manifest and **never re-selects, re-orders, filters-into or re-weights** the canonical corpus
   `physicalai-train-e438721ae894` / skip-hash `f09e44db`. `export` runs the parity check and
   **refuses to write** on a mismatch (SC-F). TanitScena calls `parity.py`'s digest and membership
   functions; it never re-implements them.
2. ⛔ **Licence class travels with every linked dataset, and is IMPORTED.** Every `DatasetLink`
   mirrors `lake/schema.py:SOURCE_REGISTRY`. A source absent from the registry has
   `license_class: null` and is **not exportable**. `refuse`-class sources are never exportable at
   any tier (`schema.py:37-43`).
3. ⛔ **Gated-confidential ids are never published.** `clip_sha8`/`clip_digest` only, everywhere
   (SC-G).
4. **Vocabularies are imported, never invented** (§4.5) — including the licence vocabulary. An
   unknown token is a hard ingest error. Refuted tokens (`LANE_TARGET`, `ROUTE_TO`,
   `PREPARE_LANE_CHANGE`) are never offered as filters.
5. **Weak evidence is labelled as weak.** Regex-derived `correct_behavior` carries
   `authored: false`; the tactical facet carries `experimental: true` with its κ and n.
6. **No number is copied into a record.** Results live in `MODEL_REGISTRY.md` and raw JSON; a
   record may hold a pointer, never a value.
7. **Two spaces are never fused into one score** (§5.4).
8. **Everything is offline-safe.** No network at query time; the deterministic hashing path keeps
   CI green with numpy alone.
9. **Portability.** No absolute filesystem path is ever written into a store or export (already
   tested at `test_scena.py:240-253`). The store directory must **not** be named `data/`
   (`.gitignore:16`).
10. **Backward compatibility.** `GET /api/scenarios` and `GET /api/scenario/{sid}` keep their
    contracts; the existing tests stay green.
11. ⛔ **No metered compute** (§9). CPU-only serving; no `@spaces.GPU`.
12. **`admissibility.inference_inputs` is recorded on every instance** so the vision-only leak test
    can be run downstream.
13. ⛔ **The known-leaky val build `physicalai-val-f1b378f295ae` (`parity.py:24-27`) is refused at
    ingest.**
14. ⛔ **val40 is never contaminable.** MEASURED (§3.7): **6 of 4,729** Alpamayo clips are in the
    canonical val40 deployment set. A training-role export containing a val40 clip is **refused**,
    with a deliberate-regression arm proving the refusal fires (SC-F). Every instance record
    carries `corpus.in_parity` and `corpus.split`, and every result payload states
    `n_in_parity` / `n_out_of_parity` — **a caller must never have to assume parity membership.**

---

## 12. Verification status of this SPEC

**What could not be verified this session, and why.** The Google Drive-backed filesystem hosting
this worktree failed reads intermittently, then persistently. MEASURED symptoms: `cp` of
`parse.py` (16,783 B) and `vector.py` (10,275 B) produced **0-byte** files after 15 retries each,
while a 1,391 B file copied on attempt 2 (so it is size/streaming-related, not permissions);
Python raised `OSError: [Errno 22] Invalid argument` on directory-visible files; `ripgrep`
reported `Invalid request code`; and `git` failed with
`fatal: failed to read .git/worktrees/interesting-tharp-463cf3/commondir: Invalid argument` on
8 consecutive attempts. A survey agent hit the same class independently and reported
`[System.IO.File]::ReadAllText()` in a retry loop as the only reliable read path.

**The filesystem recovered late in the session and three of the open items were closed.**

| claim | class | evidence / probe |
|---|---|---|
| **the scena suite passes today** | ✅ **MEASURED 2026-08-23** | `& C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest tests/test_scena.py -q` from `stack/` with `PYTHONPATH=<repo>/stack` -> **21 passed, 1 skipped, 19.64 s**. The skip is the MiniLM test (`test_scena.py:148-149`), skipped because `sentence-transformers` is absent — consistent with §3.3. |
| **the SC-A query behaves as specified** | ✅ **MEASURED 2026-08-23** | live index build (`prefer="hashing"`, n=14) then `search(q, k=3)`: *"school bus stop arm and an occluded child crossing"* -> **SC-04 0.4715**, SC-02 0.2174, SC-03 0.2153. Two more: *"slow down for the vehicle stopped ahead"* -> **SC-13 0.2192**; *"emergency vehicle blocking the road"* -> **SC-06 0.3507**. All rank-1 correct on the **hashing** path, no network. |
| **the Alpamayo/parity overlap** | ✅ **MEASURED 2026-08-23** | see §3.7: 201 in-parity, 4,528 out, **6 in val40**. This changed the design's emphasis — see the two consequences there. |
| retrieval rankings for the remaining pinned queries (SC-01/05/14) | **MEASURED, indirectly** | they are assertions inside the suite that just passed (`test_scena.py:104-134`) |
| `app.js` internals | **UNVERIFIED** | read `stack/tanitad/scena/static/app.js` (40,963 B) |
| `stack/tanitad/models/v6.py` tactical tuples | **INHERITED** — the survey agent could not read `v6.py` after 25 retries and cited two mirrors (`TACTICAL_LABEL_VALIDATION.md`) instead | read `v6.py:217-223` and `:136-140` directly |
| `cot` retrieval quality | **UNVERIFIED** — coverage, distinctness and length are measured; *usefulness* is not | the SC-C retrieval eval (`P5-10`) |
| MiniLM > hashing on our queries | **UNVERIFIED, and deliberately so** | `P5-10`. Note the three probes above all succeeded on the **hashing** path, so MiniLM is an optimisation, not a prerequisite |
| HF Space CPU allowance | **UNVERIFIED** | `huggingface_hub.get_space_runtime("Sayood/TanitAD")` |

**None of the remaining items change the design.** The one that *did* change the emphasis — the
Alpamayo/parity overlap — has been measured and folded into §3.7, §8 SC-B/SC-F and invariant 14.

---

## 13. Open questions for the PI

**Q1 — Where does TanitScena get served?**
Dev box only, a pod behind the RunPod proxy, or an HF Space? If a Space: it must be a **CPU**
Space, because the known Space `Sayood/TanitAD` is ZeroGPU and ZeroGPU is metered
(`TANITAD_PROGRAMME.md:30`). *Recommendation: dev box + pod first; a Space only when there is an
audience.*

**Q2 — Do we ever spend a GPU pass on VLM captions?**
⭐ **Not needed for v1** — the 4,729 Alpamayo `cot` sentences are already in the repo (§3.7). But
they are **templated** (1,103 distinct over 4,729 rows, 23.3 %), which caps retrieval
discrimination. *Recommendation: ship on `cot`, measure with SC-C, and only quote a captioning
budget if SC-C shows the text is the bottleneck — on the local 4060, batched, never metered.*

**Q3 — How wide is the instance corpus in v1?**
Alpamayo/PhysicalAI only (4,729 clips, already in repo), or also comma2k19 / Cosmos / nuScenes /
`l2d`? Each extra corpus is a `DatasetLink` plus a label-derivation run.
*Recommendation: Alpamayo first; `l2d` second (Apache-2.0, `owned-safe`, already surveyed and
never integrated — §10.3 item 5).*

**Q4 — Direction of truth for the `SC-xx` catalogue.**
`SCENARIO_DATABASE.md` is hand-authored on a weekly four-agent cadence and TanitScena parses it.
Does it stay authoritative, or does TanitScena become the store and the markdown a rendered view?
*Recommendation: keep the markdown authoritative for curated prose; TanitScena is authoritative
only for instances, links and embeddings.*

**Q5 — Is a public TanitScena in scope?**
The records quote named opponents' incidents with sourced URLs, and the corpus behind the
instances is **gated-confidential**. A public UI is a narrative artifact about other companies
sitting on top of a licence-restricted corpus. ⛔ **No public deployment without explicit PI
approval**, and never with raw ids.

**Q6 — Where does the code live?**
`stack/tanitad/scena/` (today, with tests and the app wired to it) or `products/P5-tanitscena/`?
*Recommendation: code stays in `stack/` — moving it breaks `test_scena.py`, `scena_app.py` and the
pod deploy for no benefit — and `products/P5-tanitscena/` holds the product-level SPEC, BACKLOG
and result artifacts.* **This SPEC assumes that answer; say so if it is wrong.**

**Q7 — Does `Project Steering/AGENT_CHARTERS.md` exist?**
Cited by the briefing, absent from this worktree (§10.3 item 1). If it exists on another branch,
re-check this SPEC against its §0 and §2.

**Q8 — Do the three stranded scenario modules get promoted?**
`stop_arm_gate.py`, `stationary_lead.py`, `emergency_scene.py` have sat outside `stack/` since
July/August (§10.3 item 3). Promoting them into `SCENARIO_REGISTRY` would take `SC-04`, `SC-13`
and `SC-06` from "prose only" to "executable + linked", which is the single cheapest lift to the
excellence programme's coverage. **It is P7/orchestrator work, not P5's — but P5 is where the gap
becomes visible.**
