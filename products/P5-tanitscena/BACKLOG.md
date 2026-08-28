# P5 — TanitScena — BACKLOG

`Companion to products/P5-tanitscena/SPEC.md. Written 2026-08-23. Owner: TanitAD_DataFlyWheel.`

**Ranking rule:** ordered by *fastest path to the charter bar — "TanitScena answers a semantic
query end-to-end" — without violating the compute ceiling.* Guards that must exist **before** a
given payload lands are ranked ahead of that payload, not after it.

**Effort scale:** `XS` < 1 h · `S` ~ half a day · `M` 1-2 days · `L` 3-5 days.
**Compute:** every item below is **CPU-only and unmetered** unless the row says otherwise.
⛔ No item here requires HF Pro quota. `P5-21` is the only GPU-shaped item and it is DEFERRED
behind a PI decision.

**Milestone M1 (the charter bar, instances included) = `P5-1 … P5-9` green.**
**Milestone M2 (the product is demoable) = M1 + `P5-10`, `P5-11`, `P5-17`.**

---

## Ranked list

### P5-1 — Search CLI over the existing type index
**Ships the charter bar for scenario types, today, with no schema work.**

- **Why.** The bar is *"answers a semantic query end-to-end"*. The retrieval already works and is
  tested; what is missing is a **callable surface** a human, a skill, or a shell pipeline can use.
  This is the single cheapest item that turns an existing capability into a delivered one.
- **Measured evidence of the gap.** `stack/tanitad/scena/parse.py:419-436` is the **only** CLI in
  the package (`python -m tanitad.scena.parse --db-md … --out …`). There is no search entry point.
  ⭐ **The retrieval it would expose is MEASURED WORKING today (2026-08-23, mine):** a live index
  build over the real DB (n=14, `hashing-tfidf`) returns **`SC-04` at rank 1, score 0.4715** for
  the exact SC-A query *"school bus stop arm and an occluded child crossing"*; also
  **`SC-13` 0.2192** for *"slow down for the vehicle stopped ahead"* and **`SC-06` 0.3507** for
  *"emergency vehicle blocking the road"*. And the suite is green: `pytest tests/test_scena.py -q`
  -> **21 passed, 1 skipped (MiniLM), 19.64 s**. ⇒ **the only thing standing between the repo and
  the charter bar is a CLI entry point.**
- **Effort.** `XS`.
- **Dependencies.** None.
- **Definition of done.** `python -m tanitad.scena search "<q>" --db-md <path> -k 5 [--json]`
  prints ranked `(id, score, title)`; `--json` emits one JSON object per line; exit 0; no network.
  New test `tests/test_scena.py::test_cli_search_end_to_end` asserts **SC-04 at rank 1** for
  *"school bus stop arm and an occluded child crossing"* on **both** embedder paths (MiniLM
  skipped when unavailable, as the file already does at `:148-149`). **SC-A green.**

---

### P5-2 — Record schema + `ScenaStore`
**The foundation every later item builds on.**

- **Why.** TanitScena has no persistent record model — it re-parses a markdown file into dicts on
  every boot. Instances, dataset links, provenance and content hashes all need a store.
- **Measured evidence of the gap.** `stack/scripts/scena_app.py:70-80` — `_reload()` re-parses the
  whole DB and holds `st.scenarios` / `st.by_id` in memory; the only persisted artifact is
  `scenarios.json` (`parse.py:410-416`), which is a dump, not a store. There is **no
  `schema_version` anywhere** in the package.
- **Effort.** `M`.
- **Dependencies.** None.
- **Definition of done.** `stack/tanitad/scena/model.py` (`SCHEMA_VERSION`, the three frozen
  dataclasses of SPEC §4.1-4.3, `validate()`) and `store.py` (`ScenaStore.open/upsert_*/get/
  iter_records/counts`). Store layout exactly as SPEC §5.3, **not** under a directory named
  `data/` (`.gitignore:16` would swallow it). Round-trip test: write 3 types + 2 datasets +
  100 instances, reopen, byte-identical `to_dict()`. A reader meeting a newer `schema_version`
  **raises**, and a test proves it raises.

---

### P5-3 — `DatasetLink` ingest from the existing licence registry
**Licence class stops being prose and becomes a field — by import, not by authorship.**

- **Why.** SPEC invariant 2 and `SKILLS_SPECS.md:50` (*"filters (source, licence, size)"*). It is
  also the item that prevents the most likely error: someone hand-typing a licence string.
- **Measured evidence of the gap.** The parsed data-source dict has **no licence field**
  (`stack/tanitad/scena/parse.py:227-233`), while a complete machine-readable registry already
  exists and is unused by TanitScena: `stack/tanitad/lake/schema.py:44`
  (`LICENSE_CLASSES = ("owned-safe","nc-research","gated-confidential","refuse")`), `:47-61`
  (`SourceLicense` + `commercial_ok`), `:68-148` (`SOURCE_REGISTRY`, **19 sources**), with
  `tier_of()` at `lake/filtering.py:29`.
  ⚠️ Also measured: the prose terms `research-OK` / `commercial-OK` used in
  `TANITAD_PROGRAMME.md:18-21` **do not exist as strings in any `.py` under `stack/`, `tools/`,
  `taniteval/`** — the code vocabulary is different, and the registry wins.
- **Effort.** `S`.
- **Dependencies.** `P5-2`.
- **Definition of done.** `python -m tanitad.scena ingest-datasets --from-lake-registry --store <d>`
  writes one `DatasetLink` per `SOURCE_REGISTRY` key with `licence` mirrored (never authored) and
  `tier` from `tier_of()`. Test asserts: `physicalai_av` -> `gated-confidential` with
  `ids_publishable: false`; `waymo`/`waymax` -> `refuse`; `comma2k19` -> `owned-safe`,
  `commercial_ok true`; `worldmodel_synth` -> `OpenMDW-1.1`; a source absent from the registry
  gets `license_class: null`. Test **fails** if any licence string in the store is not
  byte-identical to the registry's.

---

### P5-4 — Gated-id redaction guard + `verify`
**Must land BEFORE any PhysicalAI clip id can enter the store.**

- **Why.** Ordered ahead of `P5-5` deliberately: once gated ids are in a store, an index, a log
  and an export, redaction becomes archaeology. This is the leak-prevention item.
- **Measured evidence of the gap.** `stack/tanitad/lake/schema.py:68-148` classes `physicalai_av`
  as `gated-confidential`; `stack/tanitad/data/parity_train_clip_digests.json` banks **digests
  only** and records the ids as gated-confidential; `deployed_val40_clip_digests.json` uses
  `clip_sha8` as the published short form. TanitScena today has **no notion of a confidential
  field** — `parse.py`'s records are emitted whole.
- **Effort.** `S`.
- **Dependencies.** `P5-2`, `P5-3`.
- **Definition of done.** `clip_digest()` / `clip_sha8` computed via
  `stack/tanitad/data/parity.py:1858-1864` (**called, never re-implemented**). A scanner rejects
  UUID-shaped strings in any record, API response, export row, manifest or log line whose source
  has `ids_publishable: false`. `python -m tanitad.scena verify --store <d>` runs it and exits
  non-zero on a violation. ⭐ **A deliberate-injection arm proves the scanner CAN fail**
  (`TANITAD_PROGRAMME.md:161-163`). **SC-G green.**

---

### P5-5 — Alpamayo instance ingest (4,729 rows) + parity membership
**This is the corpus. It is already in the repo and costs no compute.**

- **Why.** It is the *only* artifact that gives instance-level text, full-coverage facets and a
  `clip_id` join key in one file. Without it there is nothing to search.
- **Measured evidence of the gap / of the opportunity.**
  `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy_per_clip.jsonl`
  holds **4,729 rows**, keys `clip_id, longitudinal, lateral, lane, cot`; `cot` has **100 %
  coverage**, median **49 chars**, **1,103 distinct** values (23.3 %). The 3x7 vocabulary census
  (n=4,729, **0 unparsable**) is in `raw/a1_alpamayo_taxonomy.json`, produced by
  `…/code/tac_a1_alpamayo_taxonomy.py` (CLI at `:28-29`, `:60-63`).
  TanitScena stores **zero** instance records today.
  ⭐ **The parity overlap is now MEASURED (2026-08-23, mine)** — and it is the reason this item is
  `M` and not `S`:

  | quantity | value |
  |---|---|
  | rows / distinct `clip_id` | 4,729 / 4,729 |
  | `cot` non-empty / distinct | 4,729 / 1,103 |
  | **∩ parity-train** (2,400 digests) | **201** (4.25 %) |
  | **not** in parity-train | **4,528** |
  | ⛔ **∩ canonical val40** (40 digests) | **6** |

  ⇒ **95.7 % of this corpus is outside the parity corpus, and 6 clips sit in the val40
  deployment set.** An unfiltered ingest-then-export path is a **live contamination hazard**, not a
  hypothetical one.
- **Effort.** `M`.
- **Dependencies.** `P5-2`, `P5-3`, `P5-4`.
- **Definition of done.**
  `python -m tanitad.scena ingest-instances --alpamayo <jsonl> --source-key physicalai_av
  --corpus-key physicalai-train-e438721ae894 --store <d>` writes 4,729 `ScenarioInstance` records.
  Every record carries `clip_sha8`, `corpus.in_parity` resolved by
  `clips_in_parity_train()` (`parity.py:1951-1959`), `corpus.split`, and
  `labels.label_provenance` naming the artifact and its row count. **Out-of-parity clips are
  stored with `split: "out-of-parity"` — never silently dropped and never silently merged.**
  ⛔ The ingest **refuses** the known-leaky val key `physicalai-val-f1b378f295ae`
  (`parity.py:24-27`), and **tags the 6 val40 clips with `split: "val40"`** so `P5-9`'s guard can
  see them. A test asserts the counts above **exactly** (201 / 4,528 / 6) — if the corpus or the
  manifest changes, the test tells us instead of the number silently drifting.

---

### P5-6 — Content-addressed index (fixes defects D1 and D2)
**Without this, every later ranking claim is potentially computed from stale text.**

- **Why.** A search product that serves stale vectors after an edit is not a database. It also
  makes every retrieval measurement (`P5-10`) un-trustworthy.
- **Measured evidence of the gap.**
  **D1:** `stack/scripts/scena_app.py:96-102` reuses the cached `vectors.npz` whenever
  `set(idx.ids) == cur_ids`, and `_reload()` (`:70-80`) only sets `st.index = None` without
  deleting the npz ⇒ **edit a description, change no ids, restart — the served ranking comes from
  the OLD text.** `/api/reindex` (`:194-197`) is the only escape.
  **D2:** `stack/tanitad/scena/vector.py:93-102` fits IDF over the whole corpus, so adding one
  record changes **every** hashing vector — incremental refresh is unsound on that path.
- **Effort.** `M`.
- **Dependencies.** `P5-2`.
- **Definition of done.** `ScenaIndex` (SPEC §6.3) with
  `embedder_id = "<family>/<model>@<rev>/<dim>/<norm>"`, per-record `doc_sha256`, and
  `text.manifest.json`. `refresh()` re-embeds exactly the changed records; on the hashing path it
  detects a changed `corpus_state_sha` and does a **full refit**, reporting `full_refit: true`.
  `/api/meta` reports `{embedder_id, n_indexed, n_stale, n_missing, built_at}`. Test: edit one
  description, restart **without** `/api/reindex`, assert the vector changed and that
  `n_stale >= 1` was reported first. **SC-D green.**

---

### P5-7 — Query API: structured filters applied pre-top-k, plus facets (fixes D3)
**Turns "a search box" into "a queryable database".**

- **Why.** `SKILLS_SPECS.md:50-51` requires filters *(source, licence, size)* and **counts**. P8
  cannot select data without them.
- **Measured evidence of the gap.** `stack/scripts/scena_app.py:178-192` — the search endpoint's
  entire signature is `search(q: str = "", k: int = 8)`. The facet chips documented at
  `stack/tanitad/scena/README.md:70-72` are **client-side filtering of an already-returned list**,
  so a programmatic consumer cannot filter at all.
- **Effort.** `M`.
- **Dependencies.** `P5-5`, `P5-6`.
- **Definition of done.** `Filters` + `Hit` + `search/lookup/neighbours/facets` per SPEC §6.2.
  ⚠️ **Filters are a boolean mask applied BEFORE top-k**, and a test proves it: a filtered query
  that would return fewer than `k` under post-filtering returns exactly `k` here. `facets()`
  returns value->count per facet, and experimental facets carry
  `{"experimental": true, "kappa": …, "n": …}`. Refuted tokens (`LANE_TARGET`, `ROUTE_TO`,
  `PREPARE_LANE_CHANGE`) are **never offered** as filter values, with a test naming each.

---

### P5-8 — Instance search end-to-end (CLI + HTTP)
**The charter bar, over the real corpus. This is the milestone item.**

- **Why.** `P5-1` ships the bar for 14 curated classes. This ships it for thousands of real clips,
  which is what P8 and `/TanitAD_SearchScenarios` actually need.
- **Measured evidence of the gap.** No endpoint or CLI accepts a `kind` or a corpus filter today
  (`scena_app.py:168-192`), and no instance records exist to return.
- **Effort.** `S`.
- **Dependencies.** `P5-5`, `P5-6`, `P5-7`.
- **Definition of done.** `GET /api/search?q=&k=&kind=scenario_instance&<filters>` and
  `python -m tanitad.scena search "<q>" --filter kind=scenario_instance --json` both work.
  Test: *"slow down for the vehicle stopped ahead"* returns **>= 20** hits; every hit carries a
  `clip_sha8` and an `in_parity` verdict; **no raw `clip_id` appears anywhere in the output**
  (re-uses the `P5-4` scanner). `GET /api/scenarios` and `GET /api/scenario/{sid}` still pass
  `test_scena.py:167-232` unchanged. **SC-B green.**

---

### P5-9 — `export_set` with the licence gate and the parity guard
**The P8 handoff, and the two guards that make it safe.**

- **Why.** P8 consumes a ScenarioSet (`SKILLS_SPECS.md:45-46`). Without the guards, an export can
  ship a `refuse`-class source or imply a re-selection of the parity corpus — the two errors the
  programme's invariants exist to prevent.
- **Measured evidence of the gap.** No export function or endpoint exists anywhere in the package
  (`stack/tanitad/scena/__init__.py:20-32` lists the whole surface). Meanwhile
  `stack/tanitad/lake/schema.py:37-43` records that `refuse`-class terms follow the **trained
  weights**, so no tier can contain them, and `stack/tanitad/data/parity.py:131-132` states the
  canonical serialization *"is fixed here and nowhere else; changing it invalidates every
  committed manifest."*
- **Effort.** `M`.
- **Dependencies.** `P5-3`, `P5-4`, `P5-8`.
- **Definition of done.** `export_set(...)` per SPEC §6.4 plus `POST /api/export`. Manifest
  carries query, filters, `embedder_id`, `STORE.json` hashes, per-row `license_class` + `tier`,
  `corpus_key` + `skip_hash`, and the `uid_digest` of the selection (via `parity.py:127-137`).
  ⭐ **All three guards ship with a deliberate-regression arm**: a corrupted selection makes the
  parity check raise; an injected `refuse` row makes the licence check raise; and a training-role
  export containing one of the **6 measured val40 clips** (`P5-5`) makes the contamination check
  raise. **SC-E and SC-F green.**
  ⚠️ **Integration:** the manifest shape is agreed with `products/P8-datasetcreator/SPEC.md`
  before this merges — escalated, not written into a doc and left.

---

### P5-10 — Install MiniLM and run the retrieval eval (the control arm)
**Decides the embedder with a measurement instead of a preference.**

- **Why.** SPEC §5.2. The programme's anti-false-positive rule
  (`TANITAD_PROGRAMME.md:156-166`) requires a control before a claim; "MiniLM is better" is
  currently an assumption.
- **Measured evidence of the gap.** `sentence_transformers` is **not importable on any of the five
  interpreters on this box** (`venvs/tanitad`, `venvs/carla312`, `venvs/sam3run`, `venvs/colab`,
  `C:\Python314`), so `stack/tanitad/scena/vector.py:114-127`'s MiniLM branch has **never
  executed here**. The repo's only dependency file, `stack/pyproject.toml`, lists
  `["torch>=2.4", "numpy>=1.26"]` plus `dev/sim/net/real` extras — no embedding library.
  ⇒ every ranking result in `test_scena.py` today is a **hashing-TF-IDF** result.
- **Effort.** `M`. CPU-only; the model is a one-time ~80 MB download.
- **Dependencies.** `P5-6`, `P5-8`.
- **Definition of done.** A `scena` extra in `stack/pyproject.toml`
  (`sentence-transformers>=3.0`). A **frozen** query set of **>= 30** `query -> relevant-id` pairs
  banked at `products/P5-tanitscena/raw/queries.jsonl`, authored **before** the run.
  `raw/retrieval_eval.json` reports **Recall@5 and MRR with n printed** for `minilm` and
  `hashing-tfidf`. ⭐ **Both outcomes pre-registered:** MiniLM wins -> it becomes the default;
  MiniLM does not win -> **we keep `hashing-tfidf` and record that in the SPEC.**
  The widened `doc_text` of SPEC §5.1 ships **in this item**, so the widening is measured, not
  assumed. **SC-C green.**

---

### P5-11 — UI: instance mode, facet rail, freshness banner, export basket
**Makes the product demoable. Kept out of M1 on purpose — the API is the contract.**

- **Why.** The PI's definition names *"high-quality UI (semantic search)"*. The UI is the
  strongest existing capability and needs extending, not replacing.
- **Measured evidence of the gap.** The SPA serves only type cards and a type detail view
  (`stack/tanitad/scena/README.md:68-89`); filter chips are client-side (`:70-72`); there is no
  instance view, no server-driven facet rail, no freshness indicator and no export affordance.
- **Effort.** `L`.
- **Dependencies.** `P5-7`, `P5-8`, `P5-9`.
- **Definition of done.** SPEC §7.2 items 1-8 implemented in the existing vanilla-JS SPA.
  ⛔ **No build step, no CDN, no `node_modules`** — a fresh checkout serves the UI with
  `python scripts/scena_app.py` and nothing else. An export basket containing a blocked source
  **names the blocking record on screen**. Experimental facets are labelled with their κ and n.
  A FastAPI `TestClient` test asserts each new endpoint the UI depends on.

---

### P5-12 — Register P5 in the live claims register
**Cheap, parallelisable, and required by the operating standard.**

- **Why.** `TANITAD_PROGRAMME.md:129-133`: *"Every session that asserts or refutes a claim UPDATES
  the register in the same turn."* P5 currently asserts a product exists and is not in the
  register at all.
- **Measured evidence of the gap.** Neither `Project Steering/BACKLOG.md` (18,296 chars) nor
  `Project Steering/GOALS_AND_CLAIMS.md` (5,380 chars) contains the strings `scena`, `scenario`,
  `P5`, `semantic search` or `vector`. The only P5 mention in `Project Steering/` outside the
  programme doc is `VOCABULARY.md:15`.
- **Effort.** `XS`.
- **Dependencies.** None. **Runs in parallel with anything.**
- **Definition of done.** A `P5` block in `GOALS_AND_CLAIMS.md` carrying the SPEC's success
  criteria as OPEN claims (`SC-A … SC-I`), and P5 rows in `Project Steering/BACKLOG.md` pointing
  at this file. `VOCABULARY.md` entries for `ScenarioType`, `ScenarioInstance`, `DatasetLink`,
  `ScenarioSet` so the terms do not drift.
  ⚠️ **Owned by the orchestrator, not by an agent editing `Project Steering/` unilaterally** —
  raise it, do not merge it.

---

### P5-13 — Strategic (`s2`) label ingest
**Adds a second, independent label axis to the instance facets.**

- **Why.** Strategic tokens are the programme's own vocabulary and are keyed on `clip_id`, so the
  join is free.
- **Measured evidence of the gap.** The labels exist and TanitScena reads none of them:
  `…/2026-08-16-s2-v1-labels/review/labels_v2/s2_labels_aug120.jsonl` (**201** records) and
  `s2_labels_w120val.jsonl` (**596**), schema `s2-strategic-v1` (`colab/s2_schema.py:46`;
  `build_record()` `:144-172`), record keys `schema_version, clip_id, t0_s, g_str, a_str,
  valid_window_s, disjointness, _provenance`.
  ⛔ **The canonical directory is `review/labels_v2/`, NOT `labels/`** — `labels/` carries a
  `SUPERSEDED.json` redirect pinned as `S2_CANONICAL_LABELS_REL` at
  `stack/scripts/s2_labels.py:186-188`. Reading the wrong directory is a silent correctness bug.
- **Effort.** `S`.
- **Dependencies.** `P5-5`.
- **Definition of done.** Ingest via the **existing loader** `load_s2_labels(path, role=…)`
  (`s2_labels.py:528`, `S2Row` `:328-349`) — not a re-implementation — so the superseded-refusal
  at `:494` is inherited. `strategic` becomes a facet. Test asserts the three
  never-emitted tokens are absent from the facet values and that reading `labels/` instead of
  `review/labels_v2/` **raises**: `LANE_TARGET` (refuted, `s2_derive.py:171-216`, `:284-287`),
  `ROUTE_TO` (validator-refused, `colab/s2_schema.py:239-242`), `PREPARE_LANE_CHANGE`
  (`lane_context` is `None` on **801/801** clips, `s2_derive.py:211-216`, `:515-523`).

---

### P5-14 — Close the situation-label join defect (D5) — UPSTREAM
**A one-line defect that makes an entire label family unusable to any consumer.**

- **Why.** `lane_change` and `intersection` are the only two situation labels the programme emits,
  and they cannot currently be joined to a clip. That blocks a facet TanitScena wants and it
  blocks every other consumer too.
- **Measured evidence of the gap.** `stack/scripts/emit_situation_labels.py:63` writes
  `eid.append(np.full(T, i, np.int32))` where `i` is **the enumeration index of
  `sorted(glob(...))`** — not a uid, not a `clip_id`. The NPZ (`np.savez_compressed`, `:70-73`)
  carries `episode, t, y_lane_change, valid_lane_change, y_intersection, valid_intersection,
  lead_s, hz`. ⇒ **the labels cannot be joined back without re-deriving the identical sorted file
  list.** Same class as the `eid` normalisation defect this branch already fixed (commit
  `3b51214`: *"the defect cannot be written again"*).
- **Effort.** `S`.
- **Dependencies.** None for the fix; `P5-5` to consume it.
- **Definition of done.** The emitter writes the episode **uid** (and, where available, the
  `clip_sha8`) alongside the positional index; a regression test asserts a join succeeds against
  the parity manifest without re-globbing. ⛔ **The frozen thresholds
  (`stack/tanitad/data/situations.py:77-117`, frozen by `PRE_REGISTRATION.md` §2) are NOT touched**
  — this is an identity fix only. ⚠️ **This file is outside P5's ownership. Escalate to the owning
  agent; do not edit it inside a P5 work package.**

---

### P5-15 — Link `ScenarioType` to the executable scenario registry + coverage report
**Turns "which scenarios can we actually run?" from prose into a query.**

- **Why.** The excellence programme's goal (`SCENARIO_DATABASE.md:3-4`) is per-scenario proof.
  Knowing which entries are executable and which have zero licensed data is the first thing anyone
  needs to plan that.
- **Measured evidence of the gap.** `stack/tanitad/eval/scenarios/registry.py:58` —
  `SCENARIO_REGISTRY` has **exactly three keys** (`work_zone_phantom`, `traffic_light_red`,
  `traffic_light_green`), entry shape `ScenarioEntry(name, make, policies, simulate, score,
  headline)` at `:46-53`, runner `run_registered_suite()` at `:86`. So **2 of 14** catalogued
  `SC-xx` are executable. Nothing joins the catalogue to the registry.
- **Effort.** `S`.
- **Dependencies.** `P5-2`, `P5-5`.
- **Definition of done.** `ScenarioType.eval_registry_key` populated where a counterpart exists;
  `python -m tanitad.scena verify --store <d>` emits a coverage table:
  per `SC-xx`, `executable? / n_instances / n_licensed_instances / licence classes present`.
  A test asserts every non-null `eval_registry_key` is a live key of `SCENARIO_REGISTRY`.

---

### P5-16 — Tactical facet, shipped with its own falsification
**Store the tokens; refuse to let them look reliable.**

- **Why.** SPEC invariant 5 (weak evidence is labelled as weak). A near-chance filter that looks
  like a working filter is worse than no filter.
- **Measured evidence of the gap.** The vocabulary exists
  (`HIERARCHY_VOCABULARY.md:79-99`; code split `stack/tanitad/models/v6.py:217-223`;
  band `TAC_BAND_S = (2.0, 6.0)` at `v6.py:136-140`) and an emitter exists
  (`stack/scripts/ph1_fuse.py`, `SCHEMA = "ph1-fused-v1"` `:55`, `a_tac_lat`/`a_tac_lon` by block
  vote `:772-787`) — **but `g_tac_lat`/`g_tac_lon` are emitted EMPTY with an `unavailable_reason`
  (`:253-262`, `:794`)**, and `…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md` reports
  the labels are **not demonstrated buildable at the band**: **LON κ 0.1428 [0.0540, 0.2250]
  n=201**, **LAT κ 0.1777 [0.0658, 0.2953] n=193**.
- **Effort.** `S`.
- **Dependencies.** `P5-7`, `P5-19` (the text/label source is partly out of git).
- **Definition of done.** `tactical` is a facet whose `facets()` entry carries
  `{"experimental": true, "kappa": 0.1428, "ci": [0.0540, 0.2250], "n": 201}` per axis, and the UI
  renders that beside the filter. `g_tac_*` is **absent**, not empty-but-offered. A test asserts
  the experimental flag is present and that removing it fails the test.

---

### P5-17 — Build the `/TanitAD_SearchScenarios` skill
**Conservation: a validated procedure becomes a skill so it never strands in a transcript.**

- **Why.** `TANITAD_PROGRAMME.md:185-198` lists it as one of seven programme skills, and
  `SKILLS_SPECS.md:1-7` states skills are the conservation mechanism.
- **Measured evidence of the gap.** `SKILLS_SPECS.md:48-51` specifies the skill in full
  (*"embed query -> vector search over scenario/dataset embeddings -> return scenarios with
  dataset links, counts, provenance; optionally hand off to /TanitAD_DesignDataSet"*), and no
  `.claude/skills/TanitAD_SearchScenarios/` exists.
- **Effort.** `S`.
- **Dependencies.** `P5-8`, `P5-9`.
- **Definition of done.** `.claude/skills/TanitAD_SearchScenarios/SKILL.md` that runs
  `python -m tanitad.scena search … --json` and `facets`, returns scenarios **with dataset links,
  counts and provenance** (all three, per the spec sentence), and offers the
  `/TanitAD_DesignDataSet` hand-off. The skill states the licence class of every returned source.

---

### P5-18 — Latency bench
- **Why.** SPEC §5.3 defers an ANN index on an **arithmetic** argument. That argument needs a
  measurement attached before anyone is tempted to add faiss.
- **Measured evidence of the gap.** No timing artifact exists for `/api/search`; the current n is
  14 (`test_scena.py:51`), so nothing has ever been measured at scale.
- **Effort.** `XS`.
- **Dependencies.** `P5-8`.
- **Definition of done.** A bench script + banked JSON reporting p50/p95/p99 warm `/api/search`
  latency at the store's current n, with **n and hardware printed**. **SC-H green.** The SPEC's
  "re-open the ANN question at n > 10^6 or p99 > 200 ms" rule now has a number behind it.

---

### P5-19 — Bank the fused PH1 corpus (ESCALATION — not P5 work)
- **Why.** Its `scenario_description` field is a second instance-text source, and P5 depends on it
  for `P5-16`.
- **Measured evidence of the gap.** Per `…/2026-08-15-aug120-fusion/MANIFEST.md`, the 201 `aug120`
  fused records (+600 `w120val`) live on **HF `Sayood/tanitad-ph0-aug120/fused_aug120/`**
  (204 files, 4.26 MB, far-side verified); only aggregate JSON plus **60** sample records are
  in-repo (`…/2026-08-16-s2-strategic-gap/raw/sample_fused_{aug120,w120val}/`). Example field
  value: `"night, clear, urban 2-lane; ego 4.1 m/s braking/turning_right; no agents"`.
- **Effort.** `S` (a pull + a commit), **owned by P2/P3**.
- **Dependencies.** None.
- **Definition of done.** The 801 fused records are in the repo with their manifest, or an explicit
  PI decision that they stay on HF and P5 reads them from there. ⚠️ **Raise it; do not merge it
  from a P5 package.**

---

### P5-20 — `l2d` as a second corpus
- **Why.** The largest licence-clean semantic corpus the programme has surveyed, and it is already
  in `SOURCE_REGISTRY` as `owned-safe` / Apache-2.0 (`stack/tanitad/lake/schema.py:68-148`).
- **Measured evidence of the gap.** `…/2026-07-11-semantic-label-survey` measured **4,219 distinct
  compositional navigation commands over 100 k episodes / 26.5 M frames**
  (`l2d_taxonomy_result.json`) and recommended `yaak-ai/L2D`; its **orchestrator verdict block is
  still unfilled** and it was never integrated.
- **Effort.** `M`.
- **Dependencies.** `P5-3`, `P5-5`. **PI decision Q3.**
- **Definition of done.** A `DS-l2d` `DatasetLink` with reachability probed and dated, plus
  instances ingested with their nav-command text. ⛔ **Parity is untouched — `l2d` instances carry
  `split: "out-of-parity"`.**

---

### P5-21 — A visual embedding space (DEFERRED — the only GPU-shaped item)
- **Why.** Text retrieval over templated sentences has a discrimination ceiling
  (**1,103 distinct `cot` strings over 4,729 rows = 23.3 %**). A visual space, or richer captions,
  is the way past it.
- **Measured evidence of the gap.** No visual embedding exists; `stack/tanitad/scena/vector.py`
  has a single text space. Our cached DINO features are **not text-aligned**, so they cannot serve
  a natural-language query directly.
- **Effort.** `L`. ⛔ **GPU-shaped. DEFERRED behind PI decision Q2 (SPEC §13).**
- **Dependencies.** `P5-10` must run first — **do not spend compute until SC-C shows the text
  space is the bottleneck.**
- **Definition of done.** Either (a) richer captions entering the **existing** text space, or
  (b) a genuine second index reported **separately**. ⛔ **Never fused into one score.**
  Any compute is local-4060 and batched, or it asks first.

---

### P5-22 — Promote the three stranded scenario modules (ESCALATION — not P5 work)
- **Why.** It is the cheapest lift to the excellence programme's coverage: it would take `SC-04`,
  `SC-13` and `SC-06` from prose-only to executable-and-linked, which `P5-15` would then surface
  automatically.
- **Measured evidence of the gap.** Three modules of the same shape as the registered ones live
  only under `TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/` and **cannot be
  imported from `stack/`**: `2026-07-24-stop-arm-gate-scenario/stop_arm_gate.py`,
  `2026-07-31-stationary-lead-scenario/stationary_lead.py`,
  `2026-08-07-emergency-scene-scenario/emergency_scene.py`. Their catalogue entries record
  passing offline test counts and "awaiting orchestrator triage"
  (`SCENARIO_DATABASE.md:94-96`, `:170-172`, `:360-363`) — since **2026-07-24**.
- **Effort.** `M`, **owned by the orchestrator / P7**.
- **Dependencies.** None.
- **Definition of done.** The three modules are importable from `stack/tanitad/eval/scenarios/`
  and registered in `SCENARIO_REGISTRY`; `test_scenario_suite_wiring.py:53-59`'s registry contract
  covers them automatically. ⚠️ **Raise it; do not merge it from a P5 package.**

---

## Dependency graph (M1 critical path)

```
P5-1 (XS, standalone -- ships SC-A immediately)

P5-2 --> P5-3 --> P5-4 --> P5-5 --> P5-7 --> P5-8 --> P5-9   <-- M1
              \          \      /
               \          P5-6 /
                \-------------/

P5-12 (XS) runs in parallel with everything.
P5-14, P5-19, P5-22 are UPSTREAM/ESCALATION items -- raise, do not merge from P5.
P5-21 is gated on P5-10's measurement AND a PI decision.
```

**If only one day is available:** `P5-1` (SC-A green in an hour) then `P5-12`. Both are XS, both
ship something durable, and neither depends on anything.

**Environment note.** The Drive-backed filesystem failed reads for most of this session (SPEC §12)
and then recovered; the suite ran green and all three probes above were taken after recovery. If it
degrades again, every item here is still specifiable and testable, but **nothing can be staged** —
`git` itself becomes unreadable. Verify a read path before planning around it, and never trust a
`git add` exit code: `git ls-files --stage <path>` is the evidence.
