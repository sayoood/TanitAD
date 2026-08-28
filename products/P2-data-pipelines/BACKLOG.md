# P2 — Data pipelines — BACKLOG

`Companion to products/P2-data-pipelines/SPEC.md. Ranked by leverage on the DATA MOAT
(TANITAD_PROGRAMME.md:91-93 — "best results with least data effort, maximal automation").
Every item's "evidence" row is a gap MEASURED in this worktree on 2026-08-23 — not a guess.
Items with no measured gap are not in this list; §"Explicitly NOT on this list" says why.`

**Ranking principle, stated so it can be argued with.** Two facts set the order.

1. `DataEng/DATA_STRATEGY.md:48` and `:63-67`: we are data-limited by ~2 orders of magnitude, and
   **only two levers close that** — a larger corpus under a **new, declared** parity key, or a
   frozen pretrained encoder. Curation, filtering, dedup and near-duplicate pruning are explicitly
   **single-digit factors**.
2. ⛔ **The corpus we already have is at ONE COPY** and has **no committed recipe to rebuild it**
   (P2-2). A moat you cannot reproduce is not a moat.

⇒ **Tier A = corpus mass, reach and survival.** **Tier B = the integrity that makes any P2
measurement admissible.** **Tier C = structure and hygiene.** An agent pulling top-down is
optimising the right term; one starting with the tidy refactors is not.

**Effort scale.** XS < 1 h · S ≈ half a day · M = 1–3 days · L > 3 days or GPU-gated.

---

## Tier A — moat-defining

### P2-1 · Mint-a-declared-parity-key workflow (make corpus enlargement one safe operation)

- **Why it matters.** Corpus enlargement is the **only** P2-side lever with a 100×-class payoff
  (`DATA_STRATEGY.md:63-67`). It is also the operation most likely to void every cross-arm number
  in the programme, because "enlargement" and "silent widening" differ only by whether someone
  remembered to declare a key and re-record a manifest.
- **Evidence it is needed (MEASURED).** The pieces exist and are excellent — `parity.build_entry:272`,
  `register_geometry_sibling:806`, `register_v2_geometry_sibling:1577`, `guard_corpus_build`
  (`parity.py:2216+`), `scripts/make_parity_manifest.py` (298 L), `make_parity_clip_digests.py`
  (314 L). **They are separate entry points with no single driver**, and the manifest shows what
  that costs: `physicalai-val-0c5f7dac3b11` carries `"uid_source": "count-only-unrecorded"` with an
  explicit `"todo"` that was never executed — a step that depended on someone running a second
  command, and it did not happen.
- **Effort.** M. **Dependencies.** none to build. ⛔ The go/no-go on the **4,472-clip build** is
  **PI Q1** (`DATA_STRATEGY.md:879`, §12 row 2) — build the workflow regardless; a gated decision
  does not gate its own instrument.
- **Definition of done.** One command takes (clip selection, geometry, role) → runs the ingest gate
  → builds → records count + `episode_uid_sha256` + `clip_membership` digest + skip set +
  `provenance{derived_from, derived_on, evidence_class, cross_checks}` → **stages the manifest
  diff**. A **deliberate-regression arm** (a selection containing one deployed-val clip) must make
  it REFUSE; a guard not shown able to fail proves nothing (`TANITAD_PROGRAMME.md` §6.2).

### P2-2 · ⛔ The corpus is at SINGLE_COPY and has no rebuild recipe — fix both

- **Why it matters.** This is the **existential** item. `DATA_LAKE_ARCHITECTURE.md:65-67` states
  invariant **I-D, "the recipe never dies"**, precisely so a corpus is *rebuildable, not only
  copyable* — and its stated trigger incident is a throttled rsync that **stalled a record run on
  2026-07-11 ~21:30**. Neither half is currently satisfied.
- **Evidence it is needed (MEASURED, then INHERITED for the freshness).**
  `…/incoming/2026-08-03-corpus-durability/CORPUS_DURABILITY_CENSUS.json` (`min_copies: 2`) returns
  `verdict: "SINGLE_COPY"` for the raw parity **TRAIN** epcache (`:9-10, :71`) **and** the raw
  parity **VAL** epcache (`:78-79, :118`); the two w120 caches read `copies 2 / durable 1` with
  `volatility_warning: "… one termination from SINGLE_COPY"` (`:205-206, :249-250`). More recent
  and independent: `DATA_STRATEGY.md:393-394` — the unified 201-clip perception corpus is
  `pushed_to_hf: false`; `:706-707` — the 476-clip / **18.33 GB** w120 pilot exists **only on
  Thor**. And the rebuild half: `stack/DATA_MANIFEST.json` and `scripts/rebuild_cache.py` are
  specified (`DATA_LAKE_ARCHITECTURE.md:45`; `Data Engineering/BACKLOG.md:80-88`) and ⚠️
  **UNVERIFIED to exist** — I could not locate either. Settling probe:
  `git ls-files | grep -E "DATA_MANIFEST|rebuild_cache"`, retried.
- **Effort.** S for the manifest + rebuild script (**0 GPU, and it is the half that does not need a
  disk**); M–L for actually placing a second durable copy.
- **Dependencies.** ⚠️ The second-copy half collides with **PI Q2** (the licence conflict) for
  PhysicalAI-derived bytes — which is exactly why the **recipe** half is the one to do first:
  `Data Engineering/BACKLOG.md:72-79` already designs it as `Sayood/tanitad-realmix`, a
  **recipe-only** HF dataset (clip ids + build params + split seed + per-episode sha256 +
  rebuild instructions), on the stated doctrine *"PhysicalAI-derived data: NEVER to HF, even
  privately."* **That ships without touching Q2.**
- **Definition of done.** `stack/DATA_MANIFEST.json` committed (ids + build-params hash + counts +
  per-episode sha256); `scripts/rebuild_cache.py` reproduces a parity-passing cache on a fresh host
  **verified by content, not by count**; `tools/corpus_census.py` re-run and its verdict recorded;
  and if the verdict is still SINGLE_COPY, that fact is escalated as a PI decision rather than
  filed.

### P2-3 · Give `l2d.py` a real pixel path (the largest commercially-clean corpus we hold)

- **Why it matters.** L2D is **Apache-2.0, tier `ship`, `commercial_ok`** (`lake/schema.py:75-76`)
  — HF-publishable and **inside** the public firewall, unlike everything the programme currently
  trains on (`DATA_STRATEGY.md:777`). Its header records 100,000 episodes / 26,466,954 frames. It
  is the corpus-mass lever and the publishability lever in one asset. It is also the shortest path
  to **Data-Eng goal G1**, whose deadline has already passed (below).
- **Evidence it is needed (MEASURED).** `l2d.py:478-496`: `build_episode`'s default
  `frame_source="none"` **synthesises a constant mid-grey (128) frame stack** purely to satisfy the
  tensor contract, self-labelled *"do NOT ship"*. `stack/tests/test_l2d.py` is 8 tests over **pure
  functions only** — `build_episode:473`, `read_episode_index:89`, `read_state_rows:335` and
  `_decode_window:423` are **untested**. Separately, `Data Engineering/GOALS.md:6-22` (**G1**)
  required **≥2 licence-clean owned real-urban corpora verified ON REAL BYTES by 2026-08-15**;
  last recorded status is 2026-07-17 (*"real-bytes verification is the last mile"*), and
  `Research/DATASET_LANDSCAPE.md` shows the other candidate, **ZOD, still ACCESS-blocked**
  (`Data Engineering/BACKLOG.md:28`). ⇒ **G1 is overdue with no recorded close.**
- **Effort.** M. **Dependencies.** L2D data reachable locally (unmetered). ⚠️
  `L2D_FRONT_HFOV_DEG_ASSUMED = 60.0` and `estimated_front_focal_px` are **ESTIMATED**
  (`l2d.py:420, :464`) — that estimate must be **stamped on every episode's provenance**, not
  silently baked in.
- **Definition of done.** `build_episode(frame_source="front_camera")` produces contract-valid
  episodes verified **on real bytes** (`assert_contract` + a decoded-frame variance check that
  would catch a grey-filler regression); an end-to-end test exists; the focal assumption appears in
  the episode/lake provenance as `ESTIMATED`; **G1 is closed or re-scoped in `GOALS.md` with a
  date**.

### P2-4 · Wire the curation stage into a real training sampler

- **Why it matters.** Stage 9 of the pipeline is fully implemented and its output reaches nothing.
  Until a `curation_weight` changes what a model sees, `lake/curation.py` is worth exactly zero to
  a result, and SC-9 fails by definition.
- **Evidence it is needed (MEASURED).** Repo-wide grep for `curation_weight | weakness_boost |
  curate_corpus | is_eval_holdout | run_enrichment | enrich_corpus` across `stack/`, `taniteval/`,
  `tools/`: **every hit outside `stack/tanitad/lake/` is in `test_lake_curation.py` or
  `test_lake_enrich.py`.** Zero trainers, zero evaluators, zero build scripts. Same one level up:
  `LakeWindowDataset` (`lake/view.py:151`) has **one** consumer repo-wide,
  `stack/scripts/lake_byteproof.py` — the acceptance gate that proves it byte-identical. **The lake
  proved it can replace the episode path and was then never pointed at.**
- **Effort.** M. **Dependencies.** The sampler seam exists: `StratifiedEpisodeSampler`
  (`stack/tanitad/models/v6.py:1251`) and `InteractionSampler`, constructed at
  `train_v6_staged.py:3865/3901/3959`. ⛔ **Coordinate with the TrainingFlyWheel (P4) — do not edit
  a live trainer unilaterally.**
- **Definition of done.** A training run whose sampler demonstrably consumes P2-produced weights,
  **verified by grepping the flags out of the RUNNING process** (never by reading a supervisor
  manifest — CLAUDE.md §Traps), plus a test that fails when the weight is ignored.

### P2-5 · SC-4: the curated-vs-random measurement, pre-registered with both outcomes

- **Why it matters.** The experiment that converts *"data moat"* from a slogan into a number. ⭐
  **Its most valuable outcome may be the negative one**: `DATA_STRATEGY.md:63-67` predicts curation
  is worth a single-digit factor, so a null result would correctly **de-prioritise a whole P2
  workstream** rather than embarrass it. Committing both outcomes in advance is what makes that
  acceptable.
- **Evidence it is needed (MEASURED, three probes).** (a) `curated (set|subset|corpus).*(random|
  baseline)` + `data.?efficien` → **0 hits**; (b) `sample.?efficien | subset ablation | corpus size
  ablation | matched.?size | data.?moat` → 8 hits, all literature notes or **HP-5** (*"structure
  substitutes for data"* — a **model**-side hypothesis at data fractions, not a curated-vs-random
  **data** comparison); (c) `curate_corpus` consumers → tests only. The programme's own read is
  explicit and cuts both ways: `V6_DATA_REQUIREMENT.md:101-105` — *"LIMO and s1 show tiny curated
  sets beating large ones — but for reasoning fine-tuning … ⚠️ **Do not read them as 'we can learn
  driving physics from 13 hours'**"*, and `:92-98` records **deduplication as untried** while
  *"'repeated data is worth less' is the strongest consistent finding in the curation literature"*.
  `DATA_STRATEGY.md:798-800` closes the flywheel section *"Unchanged and still unstarted."*
- **Effort.** L (GPU). **Dependencies.** P2-4. ⛔ **Gated on PI Q3** for the compute.
- **Definition of done.** A `SPEC.md` under the `TANITAD_PROGRAMME.md` §3 schema with **both
  outcomes committed before launch**; matched N, fixed model, fixed seed budget, fixed eval; result
  reported with the **paired episode-cluster bootstrap** (`taniteval/ci.py`) — never
  `overlapping_holdout_se` — and **all four metric families**, never ADE alone.

---

## Tier B — integrity (these decide whether any P2 number is admissible)

### P2-6 · Record the VAL corpus content digest

- **Why it matters.** Every published open-loop number is computed on a subset of
  `physicalai-val-0c5f7dac3b11`, and that corpus is **count-checked only**. A substituted or
  re-selected val set of the right size is undetectable today — exactly the failure
  `parity.py:20-23` was written to prevent, left open on the split that matters most.
- **Evidence it is needed (MEASURED).** `stack/tanitad/data/parity_manifest.json`, entry
  `physicalai-val-0c5f7dac3b11`: `"episode_uid_sha256": null`, `"uid_source":
  "count-only-unrecorded"`, `"evidence_class": "MEASURED (count) / UNRECORDED (uid set)"`, plus a
  literal `"todo": "run --record --split val on a pod against a cache that compute_skipset.py has
  just verified, then stage the diff"`. The train entry carries `9877bef6…3ac7386`.
  Related and still open: `DATA_STRATEGY.md:892` (§12 row 15) — **no parity-VAL 600-clip oracle
  exists**, against a 40-clip deployed-val oracle that does.
- **Effort.** S (`make_parity_manifest.py --record --split val` exists). **Dependencies.** a host
  holding a verified val cache. ⚠️ **UNVERIFIED that any host currently does** — and P2-2 says the
  last census called it SINGLE_COPY. Settling probe: `python tools/corpus_census.py --json`.
- **Definition of done.** `episode_uid_sha256` non-null and staged; `assert_val_cache:631`
  content-checks it; a deliberate-substitution arm makes it refuse.

### P2-7 · One conformance test over every adapter's window contract — and fix `Comma2k19Dataset`

- **Why it matters.** The point of the episode contract is that *"a model trained on one adapter's
  episodes consumes any other's without a code change"* (`_contract.py:106-108`). That claim is
  currently false, and the guard meant to catch it cannot.
- **Evidence it is needed (MEASURED, verified directly).**
  `_contract.EpisodeWindowDataset.__getitem__` (`_contract.py:126-139`) emits **7 keys including
  `future_actions`**. `Comma2k19Dataset.__getitem__` (`comma2k19.py:741-754`) emits **6 and omits
  `future_actions`**, while its docstring (`comma2k19.py:702`) asserts the contract is *"identical
  to the toy/MetaDrive datasets"*. `mixing.MixedWindowDataset._check_contract` (`mixing.py:103-112`)
  compares only `frames / actions / future_frames` — structurally unable to see it. comma2k19 is
  the corpus **all public numbers are supposed to come from** (`DATA_STRATEGY.md:74, :777`), which
  is what makes this more than tidiness.
- **Effort.** S. **Dependencies.** none.
- **Definition of done.** One parameterised test over every adapter that emits windows, asserting
  the exact key set / shapes / dtypes; **it must FAIL on a deliberately dropped key** before its
  PASS means anything; `Comma2k19Dataset` emits `future_actions`; the stale docstring is corrected
  or the divergence documented as intentional with a reason.

### P2-8 · Licence class on every corpus manifest entry

- **Why it matters.** SC-2. A dataset without a machine-readable licence class cannot be safely
  exported, mixed or published, and the guard that would refuse a violation
  (`license_guard.verify_license_scope:22`) reads a field the parity corpora do not have.
- **Evidence it is needed (MEASURED).** In `parity_manifest.json`, the **only** case-insensitive
  match for `licen` in the entire file is the word *"license"* used as a **verb** inside an evidence
  note. Meanwhile `lake/schema.py:68-148` registers ~20 sources with `license_class`,
  `license_name`, `share_alike`, `is_synthetic` — and the lake **cannot** contain PhysicalAI by
  construction (`physicalai_av` → `gated-confidential`, `schema.py:146`). ⇒ the corpus every arm
  trains on has no licence field anywhere in the system.
- **Effort.** S. **Dependencies.** ⚠️ the *value* for PhysicalAI/Alpamayo is **PI Q2**. Ship the
  **field and the test** now; the value is `gated-confidential` until Q2 resolves — the
  conservative reading, and already what `SOURCE_REGISTRY` says.
- **Definition of done.** Every `parity_manifest.json` entry resolves to a `SOURCE_REGISTRY` key; a
  test asserts it for all entries and fails on an unregistered source.

### P2-9 · Make the corpus census verify BY CONTENT

- **Why it matters.** The census is the programme's answer to *"did the copy count silently reach
  1?"* — and P2-2 shows that answer is currently **yes**. As built it answers *"do N files named
  like this exist here?"*, which cannot distinguish a complete copy from a same-sized substituted or
  truncated-and-renamed one.
- **Evidence it is needed (MEASURED).** `tools/corpus_census.py` counts members by
  `Artifact.pattern` glob (`:131`, `count_hf_members:450`, `probe_repo:526`) against an expected
  `members` integer. The **only** `sha256` in the module is read from HuggingFace **LFS metadata**
  (`:405`, `:442`) — never computed locally, never compared across hosts. Its own header (`:24-46`)
  lists the four rules it encodes; content-verification is not among them. Same class as C110's
  `SUFFIXES` census, which `DATA_STRATEGY.md:836-846` records as *"a real number answering a
  narrower question than the one asked"* — it missed **46 of 102** stranded files.
- **Effort.** M. **Design note:** a full re-hash of 278.78 GB is not the design — reuse
  `parity.uid_digest:127` for a manifest-digest comparison plus sampled member hashes.
  **Dependencies.** P2-6 (a recorded val digest gives the census something to compare against);
  P2-2 (a corpus manifest is the natural digest source).
- **Definition of done.** The census reports, per artifact per host, a **digest matching the
  committed manifest** or a named mismatch; two independent runs on the same fleet agree; a
  deliberate-regression arm (one member renamed, one truncated) is **detected**.

### P2-10 · Fix the dataset card the exporter writes

- **Why it matters.** The licence and provenance apparatus is correct all the way to the last step
  and then publishes something unreadable. A dataset nobody can read the terms of is not a
  publishable asset — and the card is where SC-1, SC-2 and SC-3 become visible to anyone outside
  the programme.
- **Evidence it is needed (MEASURED).** `…/incoming/2026-07-25-tanitdataset-hf-push/NOTE.md:28-33`:
  *"The remote C repo has NO dataset card. HF renders `README.md`; the exporter wrote
  `DATA_CARD.md`, which HF ignores. The repo currently shows a blank card"* — and *"an invisible
  second card without the honest-limits section is worse than none."* `:34-37`: the staged bundle
  was **missing the Parquet catalog**, so *"the card even tells consumers to 'use the catalog's
  curation weights', which they could not."* `:252` names the code site: *"`hf_export.py` writes the
  wrong card filename."* Independent second instance of the same class:
  `DATA_STRATEGY.md:288-290` — the A2 dataset card **understates its completeness hole by 356×**
  (claims one missing task row; measured **356 of 24,000**).
- **Effort.** S. **Dependencies.** none for the filename + catalog inclusion. ⛔ **Any actual push is
  a publish action and needs explicit human approval** — `NOTE.md:21-27` records the last attempt
  being **denied by the permission classifier and stopped, not worked around**: *"an agent brief is
  not user consent for a public-publish."* Keep it that way.
- **Definition of done.** `export_hf` writes `README.md` with the YAML front-matter HF needs; the
  staged bundle contains the Parquet catalog it references; a test opens the staged bundle and
  asserts **every artifact the card names is present**; and a card claim that is false of the bytes
  fails the test.

### P2-11 · `validate_pool()` over every corpus-selection artifact

- **Why it matters.** A selection parquet is the *input* to parity — a wrong column semantic there
  produces a corpus that passes every downstream digest check and is still the wrong corpus.
- **Evidence it is needed (MEASURED).** `Data Engineering/BACKLOG.md:20-23`: *"`validate_pool()`
  over EVERY corpus-selection parquet we hold (`r0_selection.parquet`, `phase0_selection.parquet`,
  `r0_selection_v2.parquet`, the TanitDataSet catalogs). **The A3 defect was a missing contract, and
  the same class of column ships in all of them.**"* The contract already exists and works — the v2
  clean-val pool is machine-checked at **34/34 on 18,988 rows**
  (`…/incoming/2026-08-02-v2-clean-val-selector/`). And these artifacts are load-bearing inside
  parity itself: `parity_manifest.json`'s w120 provenance cites
  `"selection_parquet": "…/r0/r0_selection.parquet"`.
- **Effort.** S (~2 h, 0 GPU per the source item). **Dependencies.** none.
- **Definition of done.** Every held selection parquet passes the column-semantics contract, or its
  failure is recorded with the count and the offending column; run in `pytest`.

### P2-12 · Quantify (and then decide about) the `randperm` parity-val draw

- **Why it matters.** If the val split is distributionally offset from train, **every headline
  number carries that offset** and nobody has measured it. This is upstream of P7's entire
  leaderboard.
- **Evidence it is needed (MEASURED).** `Data Engineering/BACKLOG.md:24-27`: *"Balance the PARITY
  val the same way … **it was drawn by `randperm`, not matched** … every headline number carries a
  distributional offset nobody has quantified."* Falsifier pre-registered: **max |d| < 0.10**. The
  method is proven on the v2 line — `Project Steering/BACKLOG.md:18` (A3): cell-quota matching
  **does NOT** balance (max |d| **0.3997**, 10/13 axes over bar) while greedy covariate balancing
  reaches **0.0094**; shipped n=400 at max |d| **0.0409**.
- **Effort.** S (0 GPU — it is a statistic over the existing pool). **Dependencies.** ⛔ **Measure
  only. Do NOT re-draw the parity val** — that is a re-selection and is refused by invariant 1.
  Report the offset; the remedy is a **PI decision**.
- **Definition of done.** Per-axis standardised differences train-vs-val reported with n; the
  pre-registered falsifier adjudicated either way; if max |d| ≥ 0.10, escalated as a PI decision
  rather than acted on.

### P2-13 · Test `stack/scripts/physicalai_r0.py`

- **Why it matters.** This script decides **which clips exist** in every PhysicalAI cache — the
  urban-scoring heuristic, the chunk pick, the selection parquet. It is upstream of parity, upstream
  of every arm, and a silent change to it changes the corpus without changing any parity key.
- **Evidence it is needed (MEASURED).** No test anywhere in `stack/tests/` or `tools/tests/`
  references it. Its selection stage writes `r0_selection.parquet` (`:163`) + `R0_REPORT.json`
  (`:172`), and `parity_manifest.json`'s w120 provenance already cites that parquet by path — the
  untested artifact is load-bearing inside the parity record. ⚠️ Its absence has already bitten:
  `DATA_STRATEGY.md:711-721` — the w120 build's first launch **downloaded 536 MB and died with zero
  clips built** because nothing creates `<root>/r0/r0_selection.parquet`, and without it
  `intrinsics_for_clip` warns **once** and falls back to the corpus-median principal point ⇒ **a
  silently mis-cropped corpus**. *"The crash was the good outcome."*
- **Effort.** S–M. Copy the `make_fake_r0` fixture pattern at `stack/tests/test_physicalai.py:19`.
  **Dependencies.** none.
- **Definition of done.** Selection is deterministic and pinned given a fixed catalog fixture; the
  urban score and chunk pick are unit-tested; **a missing `r0_selection.parquet` fails loudly rather
  than falling back to a median principal point**.

---

## Tier C — structure, reach and hygiene

### P2-14 · Move the production window `Dataset` into `tanitad.data`

- **Why it matters.** P2 does not currently own its own last stage, so the P2→P4 interface is a
  contract about a file P4 edits.
- **Evidence it is needed (MEASURED).** Repo-wide `^class .*WindowDataset|^class .*Dataset\(` →
  **22 classes**; only **6** are in `stack/tanitad/data/` and **1** in `lake/`. The production one
  is `FlagshipWindowDataset` at `stack/scripts/train_flagship4b.py:106`, and the current v6 trainer
  imports it **from another trainer** (`train_v6_staged.py:3595-3597`, constructed `:3735-3739`).
  `parity.py:16-18` names the cost: *"the codebase already carries a 4× copy-pasted window class."*
- **Effort.** M. **Dependencies.** ⛔ **must be additive** — `train_v6_staged.py` is a live run and
  strict resume is tensor-level. Move by re-export first (`tanitad.data` becomes the canonical
  import path, the trainer keeps working); never delete the trainer-side class in the same change.
  Coordinate with P4.
- **Definition of done.** `from tanitad.data import FlagshipWindowDataset` works; a test asserts the
  trainer-side name is an alias of the library one, so the two cannot drift.

### P2-15 · Fix the `stack/tanitad/data/__init__.py` export surface and pin it

- **Why it matters.** The package's declared public API is 4 modules; the package has ~20. Every
  consumer reaches past the surface, so there is no place to state or evolve a P2 API.
- **Evidence it is needed (MEASURED, verified directly).** `data/__init__.py` re-exports only
  `toy_driving`, `metadrive_env`, `comma2k19`, `stats`. Absent from `__all__`: `physicalai`,
  `nuscenes`, `argoverse2`, `l2d`, `cosmos_drive`, `metadrive_frontcam`, `lan`, `mixing`,
  `_contract`, `calib`, `parity`, `v2_dataset`, `epcache`, `bev_raster`, `situations`,
  `anchor_goal`.
- **Effort.** S. ⚠️ **Import-cost care**: the existing file deliberately comments *why* each import
  is safe with no sim/codec deps. Preserve that discipline — lazy where the dep is heavy.
- **Definition of done.** `__all__` reflects the real surface; a test pins it and fails when a new
  public module is added without being declared.

### P2-16 · A map / road-topology ingest path (the strategic-brain data gap)

- **Why it matters.** ⛔ **Settled at five probes and it must stop being re-asked**
  (`DATA_STRATEGY.md:81-85`): PhysicalAI-AV ships **no map, lane graph, junction annotation,
  roundabout label, traffic-light feature or route/goal signal** — the card says verbatim *"we do
  not include open maps data"* — and `egomotion` carries **no lat/lon/GNSS**, so **OSM map-matching
  on our traces is impossible**. The strategic brain's topology must be *ingested*, and P2 owns
  that ingest.
- **Evidence it is needed (MEASURED).** The consumers exist and the supply does not:
  `lan.LaneCorridor.from_json` (`lan.py:502`) reads `lane_centerlines.json` /
  `lane_graph_edges.json`; `argoverse2.LaneGraph` (`:305`) reads AV2 maps but AV2 is `nc-research` +
  share-alike (`schema.py:128`) so it can never enter a shippable tier —
  `…/2026-07-26-publishable-corpus-hunt/PUBLISHABLE_CORPUS_HUNT.md:93` records the verdict *"the
  best lane graph we hold; cannot ship"*. Meanwhile `lake/schema.py:83-106` registers **DLR ASAM
  OpenDRIVE** as *"the FIRST map asset in the programme that is neither `nc-research` nor
  `ship-sa`"* — CC-BY-4.0, commercial-OK, with **86,200** lane-link elements, **4,837** junction
  turn edges, **583** branch points, **372** positioned traffic lights, and a **full PROJ string**
  so unlike AV2 it **can** be map-matched. ⚠️ **UNVERIFIED whether a DLR OpenDRIVE *adapter*
  exists** — only `stack/tests/test_dlr_opendrive_registry.py` (a registry-entry test) was found,
  and my adapter probe timed out on the flaky mount. **Settling probe:**
  `grep -rn "dlr_opendrive\|xodr" stack/tanitad stack/scripts`, retried.
- **Effort.** M–L. **Dependencies.** P2-15; `stack/experiments/nurec-gsplat/xodr_map.py` is prior
  art to **promote, not re-derive**. ⛔ The per-record-DOI rule is mandatory:
  `schema.dlr_opendrive_source_key:170` **refuses unknown DOIs rather than defaulting**, because
  *"picking the permissive side turns an unrecognised record into a licence violation."*
- **Definition of done.** A registered adapter turning a licensed map source into the
  `lane_centerlines.json` / `lane_graph_edges.json` shape `lan.py` already consumes, with the
  DOI-level licence rule enforced, and a real `LaneCorridor.snap` round-trip test.

### P2-17 · Train one arm end-to-end through the lake

- **Why it matters.** 4,036 lines of licence-clean, tiered, deduped, export-guarded pipeline that
  has **never produced a training result**. Until an arm runs through it, its correctness is a
  byte-proof and not a capability, and the licence/tier apparatus is untested against the thing it
  exists for.
- **Evidence it is needed (MEASURED).** `LakeWindowDataset` (`lake/view.py:151`) has exactly one
  repo-wide consumer, `stack/scripts/lake_byteproof.py` — the acceptance gate whose docstring says
  its job is to *"prove LakeWindowDataset == EpisodeWindowDataset (byte-for-byte)"*. It succeeded,
  and the path was then never used.
- **Effort.** L (GPU, but **local/unmetered** — `build_tanitdataset.py:52-54` names
  `C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f` as the real MIT cache present on
  this box). **Dependencies.** P2-4 (so the run also exercises curation).
- **Definition of done.** A comma2k19-only arm trained through `LakeWindowDataset`, its provenance
  manifest + licence class + tier recorded, and its result **quotable inside the public firewall**
  (`DATA_STRATEGY.md:777`) — which no current arm is.

### P2-18 · Implement the `data:physicalai` tag-audit script

- **Why it matters.** `DATA_STRATEGY.md:775-776` makes exposure *"auditable in one grep"* the whole
  basis of the licence firewall. Without the audit, that is a claim about a grep nobody runs.
- **Evidence it is needed (MEASURED / UNVERIFIED).** `Data Engineering/BACKLOG.md:163-164` specifies
  it — *"grep ledger/leaderboard/paper for untagged PhysicalAI-AV numbers; one-command audit script
  committed to Implementation/. **Expected: 0 violations**"* — and ⚠️ **no such script was located**.
  Settling probe: `git ls-files | grep -iE "tag_audit|data_tag"`, retried, plus a grep for the
  expected output filename.
- **Effort.** S (0 GPU). **Dependencies.** none.
- **Definition of done.** One command produces a violations list with n; it is wired into
  `tools/ci_gate.py` alongside the other guards; and it is **shown able to fail** on a deliberately
  untagged row before its `0 violations` means anything.

### P2-19 · Retire `metadrive_frontcam.py` (do not implement it)

- **Why it matters.** The honest action is deletion, not completion. A stub whose test **pins the
  stub as intended behaviour** manufactures the appearance of coverage.
- **Evidence it is needed (MEASURED, verified directly).** `metadrive_frontcam.py:295-297` raises
  `NotImplementedError` for `scripted_occluder` and `blocked_route` — the two scenarios the module
  exists to provide. `stack/tests/test_metadrive_frontcam.py:166`
  (`test_populate_scene_cruise_is_noop_others_flagged`) **asserts the raise**, so the suite will not
  notice if it is fixed and will break if it is. The corpus row is already struck through:
  `DATA_STRATEGY.md:79` — *"~~CARLA on RunPod~~ · ~~MetaDrive~~ — superseded by AlpaSim / retired
  per D-014."*
- **Effort.** XS–S. **Dependencies.** confirm nothing imports it. `metadrive_env.py` is a **separate
  module and is NOT covered by this item**.
- **Definition of done.** Module removed or moved to `stack/experiments/` with a deprecation note
  naming D-014; `data/__init__.py` and tests updated; **no capability claim anywhere still cites
  it**.

### P2-20 · State comma2k19's licence in `comma2k19.py`

- **Why it matters.** comma2k19 is the corpus **all public claims rest on** (`DATA_STRATEGY.md:74,
  :777`). Its licence being asserted from a *different file* is exactly the drift
  `lake/schema.py:49-51` was designed to prevent (*"never inferred, so it cannot drift"*).
- **Evidence it is needed (MEASURED).** `comma2k19.py` has no licence block; MIT is asserted in
  `cosmos_drive.py:9` and `lake/schema.py:69`. Every other adapter states its own —
  `nuscenes.py:12-40`, `argoverse2.py:16-38`, `l2d.py:3`, `cosmos_drive.py:78`.
- **Effort.** XS. **Definition of done.** Licence block in-file citing
  `SOURCE_REGISTRY["comma2k19"]` as the single source, plus a test that the two agree.

### P2-21 · Refresh `DATASET_LANDSCAPE.md` and make the monthly source sweep a real duty

- **Why it matters.** *"Identifies every accessible data source and makes it usable"* is half the
  DataFlyWheel's charter (`TANITAD_PROGRAMME.md:91-93`). A landscape that is not swept is a moat
  that is not being widened.
- **Evidence it is needed (MEASURED).** `Research/DATASET_LANDSCAPE.md:10` records the last sweep as
  **2026-07-15** — ~5 weeks stale — against a **standing monthly HF `datasets` sweep duty (D-012)**
  at `Data Engineering/BACKLOG.md:170-174`. Two named unswept leads sit in that same backlog:
  arXiv **2604.01044** (*"A global dataset of continuous urban dashcam driving"*, `:125-129`) and a
  semantic/strategic-label dataset survey (`:131-141`). The staleness is not cosmetic: the ZOD
  licence was **corrected 2026-07-13** and nuScenes' `share_alike` was **corrected 2026-07-26** —
  licence facts in this space move.
- **Effort.** S per sweep. **Dependencies.** none (HF listing is unmetered).
- **Definition of done.** A refreshed landscape with a sweep date and, per new row, **licence class
  + actions availability + calibration/FOV + scale**; every determination banked via
  `tools/kb_add.py`; the sweep becomes a scheduled item rather than a paragraph.

---

## Escalations (⛔ NOT actionable inside this work package — reported, not written into a doc)

Per the agent standard: *"escalate integration, don't write 'please merge' into a doc."* These
require edits **outside `products/P2-data-pipelines/`** and are reported to the Master Mind.

### P2-E1 · `DataEng/DATA_STRATEGY.md` §11.1 and §12 row 14 are STALE — the credential scanner exists

- `DATA_STRATEGY.md:825-830` states in bold, at three probes (C117), **"NO CREDENTIAL SCANNER EXISTS
  IN THIS REPO"**; `:891` lists implementing one as open and **unassigned**.
- **MEASURED 2026-08-23:** `tools/secret_scan.py` exists (**51,751 bytes**) and is tracked;
  `stack/tests/test_secret_scan.py` is tracked; and `tools/ci_gate.py:88` already carries
  `"tests/test_secret_scan.py": 60`, so it runs in the mandated gate. The module's header states
  C117's claim is wrong and **explains why all three probes were structurally blind** — they
  searched script/test **names** and the **third-party** tool names
  `detect-secrets`/`trufflehog`/`gitleaks`, while the real scan lived inside `tools/safe_commit.py`
  from 2026-07-25. It ships `--install-hook` so the scan binds on every commit.
- ⚠️ **The other half of row 14 is NOT closed:** `DATA_STRATEGY.md:831-832` says the exposed HF
  token **is still in plaintext on Thor**. Rotation is PI-owned and time-sensitive (SPEC §8 Q6).
- **Ask:** the DATA_STRATEGY owner corrects §11.1 and §12 row 14; the correction is logged in
  `RETRACTION_LOG.md` by **root-cause class** — *"an absence claim from probes structurally unable
  to see the thing"* (operating-standard rule #2; same class as the Vulkan-ICD and
  `obstacle.offline` cases).

### P2-E2 · `Project Steering/AGENT_CHARTERS.md` does not exist in this worktree

- Three probes: `git ls-files | grep -i charter` (empty), `git log --all --diff-filter=A --
  "*AGENT_CHARTERS*"` (empty), repo-wide grep for the literal string across `*.md` (empty).
- SPEC §6 was therefore built from the DataFlyWheel brief + `TANITAD_PROGRAMME.md` §2/§6.
  **Ask:** confirm whether the charter is being written in another checkout; if so, reconcile
  SPEC §6 before this spec is cited.

### P2-E3 · `TanitAD Research Hub/` → `TanitAD Research Lab/` rename in flight

- `TANITAD_PROGRAMME.md:62` establishes **TanitAD Research Lab** as superseding the Research Hub
  rotation; the directory here is still `TanitAD Research Hub/`. Both P2 documents cite paths under
  it, and **`tools/corpus_census.py:223-245` hard-codes two `TanitAD Research Hub/...` artifact
  paths** that will resolve to `ABSENT` after the rename.
- **Ask:** whoever lands the rename updates `corpus_census.py`'s `ARTIFACTS` candidates in the same
  change, or the census will report the anchor sets as missing — a false SINGLE_COPY on top of a
  real one.

### P2-E4 · `Data Engineering/GOALS.md` G1 is overdue with no recorded close

- `GOALS.md:6-22`: **G1** required ≥2 licence-clean owned real-urban corpora with `drop_in=True` and
  episode-contract PASS **on real bytes**, deadline **2026-08-15 (Phase-0 data close)**. Last status
  is **2026-07-17**. **G2** (`:24-31`, the H7 IDM data-efficiency loop) reads *"head not yet
  built"*.
- **Ask:** the Data-Eng goal owner re-dates or closes G1 and G2. P2-3 is the concrete route to G1;
  G2 belongs to **P3**, not P2, and should be re-homed there rather than left in a P2-adjacent
  document.

### P2-E5 · Four `Implementation/incoming/` packages have no `RESULT.md` at all

- **MEASURED:** `2026-07-20-pod-corpus-build` (7 build scripts), `2026-07-20-vlm-labeling-pilot`
  (3 scripts), `2026-07-20-vtarget-validation` (15 result JSONs), `2026-07-21-lead-state-gate`
  (3 result artifacts) are **code/artifact-only**.
- ⚠️ **This item went stale inside the session that wrote it, which is the point.** At 13:02 today
  `2026-08-23-alpamayo-mapping` was an empty directory and `2026-08-23-sam3-completion-audit` had
  raw JSON with no RESULT. **Re-probed at 13:34:** the sam3 package now carries
  `SAM3_COMPLETION_AUDIT.md` (16,451 B, staged by a concurrent stream), and `alpamayo-mapping` has
  grown `code/` and `raw/` but **still has no top-level `.md`**. ⇒ the four 2026-07 packages are
  the durable finding; the two 2026-08-23 rows are **live work, not neglect**. Verify before
  acting on this item — `DATA_STRATEGY.md` v3.0 lasted 24 h for the same reason.
- This violates `TANITAD_PROGRAMME.md` §3 (`RESULT.md` is a required member of the work-package
  schema) and is the *"a banked script that cannot produce its banked result is worse than a missing
  one, because the pair looks like provenance"* hazard (`DATA_STRATEGY.md:850-853`) in its weaker
  form: artifacts with no stated provenance at all.
- **Ask:** the Data-Eng owner back-fills a RESULT for the four, or marks them ARCHIVED with a
  reason. **Do not** back-fill by reading the artifacts and inferring what was run — that
  manufactures provenance.

---

## Explicitly NOT on this list, and why

- **"Add more adapters."** Not a measured gap. Six adapters already emit contract-valid episodes;
  the measured problems are that one ships grey frames (P2-3), one drops a contract key (P2-7), and
  the corpus we *have* is 2 orders of magnitude too small (P2-1). Breadth is not the bottleneck.
- **"Refactor `parity.py`."** 2,446 lines, and it is the **strongest asset in P2** — count +
  content digest + leaky-split refusal + two uid spaces + a build-time ingest gate whose door set
  is **derived from source** by `stack/tests/test_build_parity_guard.py` rather than hand-listed
  (34 mention · 11 write · 6 gated · 0 unclassified), and which **pins its own blind spot** rather
  than hiding it. It earns its size. Leave it alone.
- **"Implement `metadrive_frontcam`."** See P2-19 — the honest action is retirement.
- **"Re-draw the parity val to balance it."** That is a re-selection. P2-12 measures and escalates;
  it does not act.
- **Anything requiring metered compute.** No HF GPU, no `@spaces.GPU`. The Tier-A GPU items (P2-5,
  P2-17) are scoped to local/unmetered hardware or are explicitly PI-gated.
- **The P3 items living in P2's tree** — `stack/tanitad/data/calib.py`, `stack/scripts/idm_head.py`,
  `run_idm_*.py`, and Data-Eng goal **G2**. They do P3's job (calibration estimation, IDM action
  reconstruction). SPEC §8 Q4 asks the PI to assign them; **P2 does not claim them unilaterally.**
