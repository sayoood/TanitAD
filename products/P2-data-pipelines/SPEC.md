# P2 — Data pipelines — SPEC

| | |
|---|---|
| **Product** | P2 (`TANITAD_PROGRAMME.md` §1) — *curation, filtering, dataset generation: models, algorithms, workflows* |
| **Owner** | TanitAD_DataFlyWheel (§2) |
| **Status** | v1.0 — first spec. Written 2026-08-23. |
| **Governs** | `stack/tanitad/data/`, `stack/tanitad/lake/`, the corpus build/parity/label-join scripts in `stack/scripts/`, `tools/corpus_census.py`, `DataEng/DATA_STRATEGY.md` |
| **Schema** | `TANITAD_PROGRAMME.md` §3 (work-package schema) and §6 (quality). Every number below carries an evidence class. |

> ⚠️ **Source-of-truth rule.** No model fact is quoted here. Corpus facts are quoted from
> **source files, the committed manifests, and `MODEL_REGISTRY.md`** only. Where this spec
> repeats a strategy claim it names `DataEng/DATA_STRATEGY.md` as an *index*, not as the
> measurer — that file states the same rule about itself (`DATA_STRATEGY.md:946`).

> ⚠️ **`Project Steering/AGENT_CHARTERS.md` DOES NOT EXIST in this worktree** — three probes:
> `git ls-files | grep -i charter` (empty), `git log --all --diff-filter=A -- "*AGENT_CHARTERS*"`
> (empty), and a repo-wide grep for the string `AGENT_CHARTERS` across `*.md` (empty). The
> charter criteria this spec adopts (§6) are taken from the DataFlyWheel brief and from
> `TANITAD_PROGRAMME.md` §2/§6. **If the charter exists in another checkout, reconcile §6
> against it before this spec is cited.**

> ⚠️ **Method note — the instrument, so the reader can judge the evidence.** This worktree lives
> on a Google Drive stream mount. During this survey, reads of individual files intermittently
> failed with `EISDIR` / `Invalid request code`, **and `git` itself intermittently failed to read
> `.git`** — including one sweep in which `git ls-files tools/` returned *nothing* and would have
> been read as "these files are untracked". Every negative in this document was therefore taken
> from a probe that was **retried until it succeeded**, never from a first empty result. This is
> the `df`-reports-the-cluster class (CLAUDE.md §Traps): an instrument that fails silently in the
> shape of an answer.

---

## 1. Purpose & scope

P2 turns **raw driving corpora into trainable, comparable, licence-clean, provenance-stamped
datasets** — and does it as *automation*, not as a sequence of remembered commands.

The DataFlyWheel's mandate (`TANITAD_PROGRAMME.md:91-93`) is *"best results with least data
effort, maximal automation"* plus *"identifies every accessible data source and makes it usable."*
P2 is the half of that mandate that operates on data **we can already reach**; P3 is the half that
creates reach where there was none.

P2 owns, end to end:

1. **Corpus adapters** — one module per source that emits the single episode contract.
2. **Selection & parity** — which clips are in a corpus, provably, forever.
3. **Cache & window materialisation** — the on-disk forms a trainer reads.
4. **Label derivation & join** — turning privileged signals (ego, agents, maps, VLM/SAM) into
   supervision, with the admissibility rules enforced at the join.
5. **Filtering, dedup, curation** — deciding what the model sees, and how often.
6. **Licence & provenance** — the class, the manifest, the export guard, the census.

⭐ **The strategic frame P2 exists inside, and it is not negotiable by cleverness:**
`DATA_STRATEGY.md:48` — *"We are data-limited by roughly two orders of magnitude, and
compute-limited by none."* `DATA_STRATEGY.md:63-67` states that only **two** levers close that
gap — a larger corpus under a **new, declared** parity key, or a frozen pretrained encoder — and
that *every* other data lever, curation included, is worth **single-digit factors**. ⇒ **P2's
headline job is corpus ENLARGEMENT under declared parity; curation and filtering are the second
order.** A P2 backlog that inverts that ranking is optimising the wrong term.

## 2. What P2 is explicitly NOT

| boundary | P2 stops at | the other product owns |
|---|---|---|
| **P3 — TanitAD_DataReconstruction** | corpora that already carry actions/poses/calibration | dashcam / phone / YouTube video → usable data; **automatic calibration estimation**; **IDM reconstructing actions from observation**. P2 *consumes* P3's output as one more source adapter. `stack/tanitad/data/calib.py` (66 kB) and `stack/scripts/idm_head.py` / `run_idm_*.py` sit on this seam — see §5.3. |
| **P5 — TanitScena** | producing the per-episode enrichment/tags that a scenario DB indexes | the scenario DB itself, embeddings, vector search, the semantic-search UI (`stack/tanitad/scena/{parse,vector}.py`) |
| **P8 — TanitDataSetCreator** | the *library* that builds, filters and exports a dataset | the **clickable + CLI configurator** on top of it. `stack/scripts/build_tanitdataset.py` is today a P2 library batch job with a CLI; P8 is the product it becomes. The chain is already written down: `Project Steering/SKILLS_SPECS.md:45` — *"query TanitScena (P5) → select sources → **licence class check** → build via TanitDataSetCreator (P8) config → **skip-hash + provenance manifest** → episode-count and content checks (verify by content) → register in TanitScena."* **P2 owns the middle three arrows.** |
| **P4 — Training** | handing over episode providers + a window `Dataset` + labels | the loop, the sampler wiring, the losses. ⚠️ **This boundary is currently violated** — see §4 stage 6. |
| **P7 — TanitEval** | the val corpus, its parity guard and its contamination oracle | metrics, estimators, leaderboard (`taniteval/`) |
| **not P2 at all** | — | model weights, architecture, deployment. P2 never publishes a model number. |

---

## 3. Current state — MEASURED from the repo

### 3.0 The documents that already govern P2, and their health

| doc | what it is | health |
|---|---|---|
| `TanitAD Research Hub/Data Engineering/DATA_LAKE_ARCHITECTURE.md` (545 L) | **the P2 architecture spec** (2026-07-13). Declares invariants **I-A** strict superset of the episode contract · **I-B** geometry baked at ingest · **I-C** licence is a first-class *structural* axis (physical partitioning + export guard, not a filter) · **I-D** *"the recipe never dies"* — source ids + build-params hash + per-episode `sha256` | ⚠️ `:3-6` self-describes as *"design / planning only. No infra stood up."* The **lake code shipped**; the R2 store and the `MANIFEST.json`-per-shard half did not. |
| `DataEng/DATA_STRATEGY.md` v4.0 (952 L, 2026-08-18) | the strategy INDEX + 16 open decisions | ⚠️ **structurally stale by design** — `:946-951` records that v2.0 lasted 48 h and v3.0 **24 h** because §12 row 12 (*"own the maintenance contract"*) is **unassigned**. Its own "newest package" pointer (`:949-951`) is already wrong: two 2026-08-23 packages exist. |
| `TanitAD Research Hub/Data Engineering/GOALS.md` (46 L) | the three standing Data-Eng goals | ⛔ **G1's deadline has passed with no recorded close.** G1 = ≥2 licence-clean owned real-urban corpora verified **on real bytes**, due **2026-08-15**; last status 2026-07-17 *"real-bytes verification is the last mile"*. **G2** (IDM head pseudo-labelling, action-agreement r ≥ 0.6 vs real CAN) — *"head not yet built"*. **G3** owned real-urban ≈35 % of the mix. |
| `TanitAD Research Hub/Data Engineering/Research/DATASET_LANDSCAPE.md` (97 L) | the corpus census by licence class | ⚠️ last sweep **2026-07-15** — ~5 weeks stale against a standing monthly HF sweep duty (D-012) |
| `TanitAD Research Hub/Data Engineering/Implementation/incoming/` | **55 work-package directories** | 🟡 mixed. Strong recent RESULTs (`2026-08-18-alpamayo-parity-exclusion`, `2026-08-18-build-parity-guard`, `2026-08-17-*`). ⛔ **Four are code/artifact-only with no `.md` at all** — `2026-07-20-pod-corpus-build`, `2026-07-20-vlm-labeling-pilot`, `2026-07-20-vtarget-validation`, `2026-07-21-lead-state-gate`. Two 2026-08-23 packages are **live work in this same session** and moved while this spec was being written (`2026-08-23-sam3-completion-audit` gained its RESULT at ~13:18; `2026-08-23-alpamayo-mapping` gained `code/`+`raw/` and still has no top-level `.md`) — do not read them as neglect. |

⚠️ **Note the rename in flight.** `TANITAD_PROGRAMME.md:62` establishes **TanitAD Research Lab** as superseding the Research Hub rotation; in THIS worktree the directory is still `TanitAD Research Hub/`. Every path above breaks at the rename — including two hard-coded candidates inside `tools/corpus_census.py:223-245`.

### 3.1 The stage map

| # | stage | implementing modules | state | evidence |
|---|---|---|---|---|
| 1 | **Source acquisition** | `stack/scripts/physicalai_r0.py` (2-stage HF CLI); per-source `_ACQUIRE_MSG` blocks | **PARTIAL** | complete for PhysicalAI; other sources are documented-manual. **No test anywhere** covers `physicalai_r0.py` (P2-13). |
| 2 | **Ingest / episode build** | `stack/tanitad/data/{physicalai,comma2k19,nuscenes,cosmos_drive,l2d,metadrive_*,toy_driving}.py` → `_contract.assemble_episode` | **PARTIAL** | 6 of 10 adapters emit a real `ToyEpisode`; 2 are not episode adapters at all; 1 hard stub; 1 filler-frame path. §3.2 |
| 3 | **Episode cache** | `stack/tanitad/data/epcache.py` (162 L), `mixing.save_episode/load_episode` | **EXISTS** | resumable per-source build with `skip_%05d` markers, `epcache.py:132-156`; collision-safe key `epcache.py:62-67` |
| 4 | **Parity selection + guard** | `stack/tanitad/data/parity.py` (2,446 L) + `parity_manifest.json` + `parity_train_clip_digests.json` + `deployed_val40_clip_digests.json` | **EXISTS — the strongest asset in P2** | count + content digest + leaky-split refusal + two uid spaces + a build-time **ingest gate** (`epcache.py:112-117` → `parity.guard_corpus_build`, `parity.py:2216+`) |
| 5 | **Compressed / geometry re-cache (v2)** | `stack/scripts/v2_compressed.py`, `stack/tanitad/data/v2_dataset.py` (568 L) | **EXISTS** | `build_v2_providers` `v2_dataset.py:462`, `LazyV2Episode` `:186`, `stable_episode_id` `:69`, manifest v3 `:60-61` |
| 6 | **Window / dataset export** | `_contract.EpisodeWindowDataset:104`; **but the production one is `FlagshipWindowDataset` in `stack/scripts/train_flagship4b.py:106`** | **PARTIAL — boundary violation** | the trainer imports its window class from another trainer: `train_v6_staged.py:3595-3597`, used at `:3735-3739`. §4 stage 6. |
| 7 | **Label derivation** | `data/situations.py` (430 L), `data/anchor_goal.py`, `scripts/{emit_situation_labels,s2_derive,v4_labels,refb_labels}.py`, `scripts/ph0_sam3.py` (1,567 L), `scripts/ph1_fuse.py`, `scripts/vlm_*.py` | **EXISTS, fragmented** | admissibility is enforced *in the code*: `s2_labels.py` re-asserts goal/situation disjointness **on the bytes it consumes**; `s2_derive.py` reads Engine A through an `ENGINE_A_ALLOWED` allowlist so `situations` is structurally unreadable |
| 8 | **Label join** | `stack/scripts/build_obstacle_join.py` (899 L), `s2_labels.py` (836 L) | **EXISTS** | join schema pinned to its real consumer + a roundtrip test (`build_obstacle_join.py:1-30`); S2 join is **stable-id only**, legacy 16-bit ids REFUSED because 69/2400 train + 7/600 val collide (`s2_labels.py` header) |
| 9 | **Filtering / dedup / curation** | `stack/tanitad/lake/{filtering,dedup,curation,enrich}.py` (1,263 L) | ⛔ **EXISTS BUT UNCONSUMED** | see §3.3 — this is the single largest measured gap |
| 10 | **Licence & tiering** | `lake/schema.py:44` `LICENSE_CLASSES`, `:68-148` `SOURCE_REGISTRY`, `lake/filtering.py:29` `tier_of`, `lake/license_guard.py` | **EXISTS (lake only)** | 20 sources registered with per-source class, share-alike and synthetic flags; DOI-level allowlist for DLR OpenDRIVE (`schema.py:157-167`) |
| 11 | **Export / publication** | `lake/hf_export.py:101 export_hf` + `_data_card:26`; `lake/license_guard.verify_license_scope:22` | **EXISTS, ungated by CI** | export refuses on the FIRST out-of-scope row; `gated-confidential` and `refuse` can never be in a scope (`license_guard.py:41-48`) |
| 12 | **Durability census** | `tools/corpus_census.py` (763 L) + `tools/tests/test_corpus_census.py` | **PARTIAL** | probes N hosts × M candidate paths, distinguishes `PRESENT/ABSENT/PARTIAL/UNKNOWN/UNREACHABLE` (`:82-93`), durable-vs-volatile hosts (`:109-110`). ⛔ **Counts members by glob; it does not verify by CONTENT** — see §3.4 |

### 3.2 Corpus adapters — the measured inventory

*(MEASURED — delegated file:line survey of `stack/tanitad/data/`; the three claims this spec
leans on hardest were re-read by me directly and are marked ✓.)*

| adapter | lines | corpus | licence stated in-file | emits `ToyEpisode`? | dedicated test |
|---|---|---|---|---|---|
| `physicalai.py` | 757 | PhysicalAI-AV R0 (gated) | gated/confidential note `:147-160` | ✅ `build_episode:679` | ✅ ×4 |
| `comma2k19.py` | 754 | comma2k19 | ⛔ **none in-file** (MIT asserted only from `cosmos_drive.py:9`) | ✅ `:649` + `Comma2k19Dataset:701` | ✅ ×3 |
| `nuscenes.py` | 733 | nuScenes v1.0 | CC-BY-NC-SA-4.0 `:12-40` | ✅ `:683` | ✅ 22 tests |
| `cosmos_drive.py` | 320 | Cosmos-Drive-Dreams | `LICENSE = "CC-BY-4.0"` `:78` | ✅ `:225` + dataset factory `:292` | ✅ 13 tests |
| `l2d.py` | 513 | yaak-ai L2D | Apache-2.0 `:3` | ⚠️ **frames are a constant grey filler** on the default path (`:478-496`) | ✅ 8 tests — **none touch `build_episode`** |
| `argoverse2.py` | 750 | Argoverse 2 | CC-BY-NC-SA-4.0 `:16-38` | ⛔ **no** — lane-graph reader only, 0 `ToyEpisode` refs | ✅ 46 tests |
| `lan.py` | 617 | *n/a* | — | ⛔ **no** — route/goal feature encoder, not an adapter | ✅ 34 tests |
| `metadrive_env.py` | 207 | MetaDrive BEV | none | ✅ + `MetaDriveDataset:115` | ✅ 8 tests |
| `metadrive_frontcam.py` | 363 | MetaDrive front RGB | none | ⛔ **stub** — `raise NotImplementedError` `:295-297` ✓ | ✅ 18 tests — **the test PINS the stub** |
| `toy_driving.py` | 159 | synthetic | — | ✅ (contract root) | ✅ 3 tests |

Three findings that belong in the spec rather than a backlog footnote:

- ✓ **CONTRACT DIVERGENCE, verified directly.** `_contract.EpisodeWindowDataset.__getitem__`
  (`_contract.py:126-139`) emits **`future_actions`**. `Comma2k19Dataset.__getitem__`
  (`comma2k19.py:741-754`) emits **six keys and not that one**, while its docstring (`:702`)
  claims the contract is *"identical to the toy/MetaDrive datasets"*. `mixing.MixedWindowDataset
  ._check_contract` (`mixing.py:103-112`) compares only `frames/actions/future_frames`, so the
  guard that exists is structurally unable to see it. → **P2-7**.
- ✓ **The package surface is stale.** `stack/tanitad/data/__init__.py` re-exports **4** modules
  (`toy_driving`, `metadrive_env`, `comma2k19`, `stats`). `physicalai`, `nuscenes`, `argoverse2`,
  `l2d`, `cosmos_drive`, `parity`, `epcache`, `v2_dataset`, `calib`, `situations`, `anchor_goal`,
  `bev_raster`, `mixing`, `_contract`, `lan`, `metadrive_frontcam` are **not** in `__all__`.
  Callers reach past the package surface everywhere. → **P2-15**.
- **Two of the ten files in the "adapter" folder are not adapters.** `argoverse2.py` (lane
  graphs) and `lan.py` (route encoding) are complete and well-tested for their real roles, but
  neither can hand a trainer an episode. Any inventory that counts folder membership as adapter
  count is wrong by 20 %.

### 3.3 ⛔ The largest measured gap: filtering/curation is implemented and consumed by nothing

`stack/tanitad/lake/` is 4,036 lines across 17 modules and implements the full rev-3 pipeline —
Stage 1 enrich, Stage 2 filter + dedup, Stage 3 stratified curation, Stage 4 kinematic goal
minting (`lake/__init__.py:48-58`). Curation implements exactly what the strategy asks for:
inverse-frequency clamped stratum weights (`curation.py:38-67`), a 2.5× boost on the five known
weakness strata (`:69-116`), kinematic safety-event mining (`:134-178`), and a hash-pinned
per-tier eval holdout at `split_unit_id` granularity (`:180-206`).

**MEASURED 2026-08-23** — repo-wide grep for `curation_weight | weakness_boost | curate_corpus |
is_eval_holdout | run_enrichment | enrich_corpus` across `stack/`, `taniteval/`, `tools/`:

> **every hit outside `stack/tanitad/lake/` is a test.** `test_lake_curation.py`,
> `test_lake_enrich.py`. **Zero trainers. Zero evaluators. Zero build scripts.**

The same holds one level up: `LakeWindowDataset` (`lake/view.py:151`) — the drop-in the lake was
built to be — has **exactly one consumer in the repo**, `stack/scripts/lake_byteproof.py`, which
is the acceptance test that proves it is byte-identical. **The lake proved it can replace the
episode path and was then never pointed at.**

⇒ This is the C108 class named in `parity.py:2216+`: *"a rule that depends on the next operator
having read a report is doctrine that never runs."* Curation here is not doctrine — it is
working code — but nothing invokes it, which has the same downstream value: zero. → **P2-4**.

### 3.4 Provenance, licence and census — where each criterion actually stands

**Provenance manifest.** `stack/tanitad/data/parity_manifest.json` (schema
`tanitad.parity_manifest/1`) carries four corpora, each with `episode_count`, `uid_kind`,
`uid_source`, `episode_uid_sha256`, `skip_indices` / `skip_count`, `clip_membership`, and a
`provenance` block with **`evidence_class` and named cross-checks** (e.g. the train entry's
`sum_T_out_vs_corpus_profile_total_frames: "472627 == 472627 MATCH (different pod, different
path)"`). This is a genuinely good manifest. Two holes:

- ⛔ **The VAL corpus has no content digest.** `physicalai-val-0c5f7dac3b11` carries
  `"episode_uid_sha256": null`, `"uid_source": "count-only-unrecorded"`, `"evidence_class":
  "MEASURED (count) / UNRECORDED (uid set)"` and an explicit `"todo"` to record it. **Every
  published open-loop number in this programme is computed on a subset of that corpus**, and it
  is count-checked only. → **P2-6**.
- ⛔ **The manifest carries no licence class.** MEASURED: the only case-insensitive match for
  `licen` in the whole file is the word *"license"* used as a verb inside an evidence note. The
  licence axis exists **only in the lake** (`lake/schema.py:68-148`) and the lake **cannot
  contain PhysicalAI** by construction (`physicalai_av` is `gated-confidential`, `schema.py:146`).
  ⇒ the corpus every arm trains on has no machine-readable licence field anywhere. → **P2-8**.

**Skip-hash.** `PARITY_SKIP_HASH = "f09e44db"` is a named constant (`parity.py:80`) and
`skip_indices` (24 entries) are in the manifest. ✅ criterion met for the train corpus.

**Census reproducible by content.** ⛔ **Not met.** `tools/corpus_census.py` counts members by
**glob pattern** per host (`Artifact.pattern`, `:131`) and compares to an expected `members`
count. The only `sha256` in the module is read from **HuggingFace LFS metadata**
(`:405`, `:442`) — never computed locally, never compared across hosts. A census therefore
proves *"N files named like this exist here"*, which is the same shape as C110's
`SUFFIXES = (".py", ".sh")` failure that `DATA_STRATEGY.md:836-840` records: a real number
answering a narrower question than the one asked. The module's own header even carries the rule
it does not yet satisfy. → **P2-9**.

⛔ **Durability: the canonical corpus is at ONE COPY.** The last recorded census
(`TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-03-corpus-durability/
CORPUS_DURABILITY_CENSUS.json`, `min_copies: 2`) returns `verdict: "SINGLE_COPY"` for **both** the
raw parity TRAIN epcache (`:9-10, :71`) and the raw parity VAL epcache (`:78-79, :118`); the two
w120 caches read `copies 2 / durable 1` with `volatility_warning: "only 1 durable copy … the rest
are on rented pods ['pod5'] — one termination from SINGLE_COPY"` (`:205-206, :249-250`). ⚠️
**INHERITED — that census is dated 2026-08-03 and I did not re-run it** (no fleet access asserted;
settling probe is `python tools/corpus_census.py --json`). Independently and more recently:
`DATA_STRATEGY.md:393-394` records the unified 201-clip perception corpus as `pushed_to_hf:
false`, and `:706-707` the 476-clip / 18.33 GB w120 pilot as existing **only on Thor**. ⇒ **the
single largest existential risk to the data moat is not curation quality, it is that the corpus
has one copy and no committed recipe to rebuild it.** → **P2-2**.

⛔ **The dataset card the exporter writes is not the card HF renders.** MEASURED at
`…/incoming/2026-07-25-tanitdataset-hf-push/NOTE.md:28-33`: *"The remote C repo has NO dataset
card. HF renders `README.md`; the exporter wrote `DATA_CARD.md`, which HF ignores. The repo
currently shows a blank card."* And `:34-37`: the staged bundle was **missing the Parquet
catalog**, so *"the card even tells consumers to 'use the catalog's curation weights', which they
could not."* ⇒ the licence/provenance apparatus is correct all the way to the last step and then
publishes nothing legible. → **P2-10**.

**Data-efficiency MEASURED.** ⛔ **Not met, and nothing in the repo attempts it.** Three probes,
all repo-wide: (a) `curated (set|subset|corpus).*(random|baseline)` + `data.?efficien` → **0
hits**; (b) `sample.?efficien | subset ablation | corpus size ablation | matched.?size |
data.?moat` → 8 hits, **all** either literature notes or `HP-5` (*"structure substitutes for
data"*, matched-param learning curves at data fractions — a **model**-side hypothesis, not a
curated-vs-random **data** comparison); (c) `curate_corpus` consumers → tests only. The one
sentence in the strategy that would need it, `DATA_STRATEGY.md:798-800`, ends *"Unchanged and
still unstarted."* → **P2-5**, and it is the criterion that decides whether the data moat is
real.

### 3.5 A live doc/code contradiction found while measuring — the credential scanner

`DATA_STRATEGY.md:825-830` states, in bold and at three probes (C117), **"NO CREDENTIAL SCANNER
EXISTS IN THIS REPO"**, and `DATA_STRATEGY.md:891` (§12 row 14) lists *"implement the credential
scan for bulk imports"* as an open decision with the scanner **unassigned**.

**MEASURED 2026-08-23:** `tools/secret_scan.py` exists — **51,751 bytes**, tracked
(`git ls-files` confirms `tools/secret_scan.py` **and** `stack/tests/test_secret_scan.py`). Its
own header states the case explicitly: *"C117 then recorded, at three probes, that no scanner
exists. That absence claim is WRONG and this module's first job is to say so"* — and it names
`tools/safe_commit.py` as having carried a six-pattern content scan since 2026-07-25, invisible
to all three C117 probes because they searched **names** and **third-party tool names**. It ships
`--install-hook` (a `pre-commit` hook so the scan binds) and a test that fails when the hook is
missing.

⇒ **The doc is stale, not the code.** This is a documentation-correction item, and it is exactly
the failure mode the operating standard's rule #2 exists for — an absence claim from probes that
were structurally unable to see the thing. → **P2-E1**. ⚠️ The *other* half of that row — **rotate
the exposed HF token, which `DATA_STRATEGY.md:831-832` says is still in plaintext on Thor** —
is **PI-owned and NOT closed by this finding**.

---

## 4. Architecture — the pipeline as it actually is

```
   ┌─ 1 SOURCE ─────────────────────────────────────────────────────── PARTIAL
   │  physicalai_r0.py  (HF gated: select -> r0_selection.parquet -> fetch-camera)
   │  manual/documented for comma2k19, nuScenes, Cosmos-DD, L2D, AV2, NuRec
   ▼
   ┌─ 2 INGEST ──────────────────────────────────────────────────────── PARTIAL
   │  data/<corpus>.py :: build_episode  ->  ToyEpisode
   │  contract: frames[T,C,H,W] u8 · actions[T,2] · poses[T,4] · episode_id
   │  pinned by _contract.assert_contract:59
   ▼
   ┌─ 3 EPISODE CACHE ─────────────────────────────────────────────────  EXISTS
   │  epcache.build_episodes_cached:80  ->  <root>/<tag>-<key>/ep_%05d.pt
   │  fault-tolerant (skip_%05d), resumable, mmap-backed, collision-safe key
   │  ⭐ THE INGEST GATE FIRES HERE: epcache.py:112-117 -> parity.guard_corpus_build
   ▼
   ┌─ 4 PARITY ────────────────────────────────────────────────────────  EXISTS
   │  parity.py: count + sha256(sorted uids) + leaky-split refusal
   │  two uid spaces: epcache_basename | v2ep_clipid  (parity.py:104-105)
   │  oracles: parity_train_clip_digests.json · deployed_val40_clip_digests.json
   ▼
   ┌─ 5 v2 GEOMETRY RE-CACHE ──────────────────────────────────────────  EXISTS
   │  scripts/v2_compressed.py  ->  <clip_id>.v2ep.pt (JPEG, cylindrical crop)
   │  v2_dataset.build_v2_providers:462 -> [LazyV2Episode] (same attr surface)
   ▼
   ┌─ 6 WINDOWS ────────────────────────────────── PARTIAL / BOUNDARY VIOLATION
   │  production path: FlagshipWindowDataset  @ scripts/train_flagship4b.py:106
   │  library path:    EpisodeWindowDataset   @ data/_contract.py:104
   ▼
   ┌─ 7/8 LABELS + JOIN ───────────────────────────────────  EXISTS, FRAGMENTED
   │  situations.py · anchor_goal.py · s2_derive/s2_labels · build_obstacle_join
   │  ph0_sam3 (pixels) · ph1_fuse · vlm_* (symbols) · v4_labels · refb_labels
   ▼
   ┌─ 9 FILTER / DEDUP / CURATE ──────────────── EXISTS BUT CONSUMED BY NOBODY
   │  lake/{filtering,dedup,curation,enrich}.py   -> curation_weight, holdout
   ▼
   ┌─ 10/11 LICENCE + EXPORT ──────────────────────────── EXISTS (lake only)
   │  lake/schema.SOURCE_REGISTRY -> tier_of -> shards/catalog partitioned
   │  license_guard.verify_license_scope -> hf_export.export_hf (+ data card)
   ▼
   ┌─ 12 CENSUS ────────────────────────────────────────────────────── PARTIAL
      tools/corpus_census.py — copy count over DISTINCT machines, by GLOB not hash
```

**Stage 6, stated plainly, because it is an architecture defect and not a style complaint.**
The window class the flagship actually trains on lives in a **trainer script**, and the current
v6 trainer imports it *from another trainer*: `train_v6_staged.py:3595-3597` imports
`build_train_episodes` from `train_v58f_unicycle_head` and `FlagshipWindowDataset` from
`train_flagship4b`, then constructs the dataset at `:3735-3739`. MEASURED: a repo-wide grep for
`^class .*WindowDataset|^class .*Dataset\(` returns **22 classes**, of which **6** are in
`stack/tanitad/data/` and **1** in `lake/`; the rest are in `stack/scripts/` and
`stack/experiments/`. `parity.py:16-18` already names the cost: *"the codebase already carries a
4× copy-pasted window class."* ⇒ **P2 does not currently own its own last stage**, and any P2/P4
contract written today is a contract about a file P4 edits. → **P2-14**.

**Two parallel worlds, and they do not meet.** The *PhysicalAI/parity/v2* path (stages 1-8) is
what every trained arm uses. The *lake* path (stages 9-11) is licence-clean, tiered, curated,
export-guarded — and by construction can **never** contain PhysicalAI (`schema.py:146`, `refuse`
and `gated-confidential` raise on ingest). They share only `mixing.load_episode` and the episode
contract. **Neither is wrong; the gap is that no arm has ever been trained through the second
one**, so the curation, dedup, licence and export machinery has never affected a result. That
single sentence is the honest summary of P2's current state.

---

## 5. Interfaces

### 5.1 P2 → P4 (Training)

**The contract as it is today (MEASURED):**

| handed over | type | produced by |
|---|---|---|
| episode providers | `list[ToyEpisode]` or `list[LazyV2Episode]` — `.frames`, `.actions`, `.poses`, `.episode_id`, optional `.maneuvers` | `epcache.build_episodes_cached` / `v2_dataset.build_v2_providers:462` |
| the frozen contract | `frames[T,C,H,W] uint8` · `actions[T,2]` · `poses[T,4] = [x,y,yaw,v]` · `episode_id` | asserted by `_contract.assert_contract:59` |
| parity verdict | `parity=True/False` + a loud refusal, **before any GPU allocation** | `parity.assert_parity_corpus:467` / `assert_v2_parity_cache:1647` |
| episode identity | `stable_episode_id(clip_id)` — blake2b>>1, 63-bit | `v2_dataset.py:69` |
| labels | per-window batch keys (`g_str_id … s2_valid`, maneuvers, anchors, goals) | `s2_labels.py`, `v4_labels.py`, `anchor_goal.py`, `build_obstacle_join.py` |

**The contract as it must become** (see P2-14): P2 owns the window `Dataset` and P4 imports it
from `tanitad.data`, not from a sibling trainer. **P2 must also start handing over a sampling
weight** (`curation_weight`) — today the curation stage computes one and nobody receives it.

### 5.2 P2 → P7 (Eval)

- The **val corpus** and its resolution: `parity.resolve_val_dir:587`, `assert_val_cache:631`,
  `val_deployments:577` — with registered admissible subset counts (600 built / 40 canonical),
  because *"a strict ==600 check would refuse every eval this program has run, and a bare
  '<=600' check catches nothing"* (manifest note).
- The **contamination oracles**: `assert_eval_clips_disjoint_from_parity_train:1962` and
  `assert_train_clips_disjoint_from_deployed_val:2115`, backed by committed per-clip digests.
  MEASURED on the committed oracles (`parity.py:2216+` §10c header): parity-TRAIN (2,400) ∩
  deployed-VAL (40) = **0**.
- The **eval holdout rule** for lake-tier corpora: `curation.is_eval_holdout:192` — hash-pinned
  at `split_unit_id`, so a route's windows never straddle the split. *(Currently unconsumed.)*

### 5.3 P2 ← consumes

- **P3** — reconstructed corpora (dashcam/phone/YouTube), estimated calibration, IDM-derived
  actions. Today the seam is ambiguous: `stack/tanitad/data/calib.py` (66 kB) and
  `stack/scripts/{idm_head,run_idm_*}.py` are in P2's tree but do P3's job. **Assign in the
  charter reconciliation; do not let both products claim them.**
- **P5/P8** — a dataset *spec* (which tiers, which strata, which licence scope) that P2 resolves
  into a concrete view. `lake/catalog.resolve_view` + `LakeView` is the resolution primitive
  that already exists for this.
- **Sources** — HF (gated PhysicalAI, ungated comma2k19 mirror, Cosmos-DD, L2D), Zenodo (DLR
  OpenDRIVE, DOI-level), local disk, NuRec/AlpaSim scenes.

---

## 6. Success criteria (measurable, pre-registered)

Each criterion states the instrument and what a FAIL looks like. A criterion with no instrument
is a work item (`TANITAD_PROGRAMME.md` §6.3), not an excuse.

| # | criterion | instrument | today |
|---|---|---|---|
| **SC-1** | **Every dataset ships a provenance manifest.** Corpus key, split, count, uid kind, uid digest, skip set, build params, derived-from, evidence class. | a test that walks every registered corpus key and asserts all fields non-null | 🟡 **PARTIAL** — schema exists and is good; `physicalai-val-*.episode_uid_sha256 is null` |
| **SC-2** | **Every dataset ships a licence class.** One of `owned-safe / nc-research / gated-confidential / refuse`, plus `share_alike`, plus `commercial_ok`, set as a CONSTANT and never inferred. | `lake/schema.SOURCE_REGISTRY` + a test that every manifest entry resolves to a registry key | 🔴 **FAIL for the parity corpora** — no licence field in `parity_manifest.json` (§3.4) |
| **SC-3** | **Every dataset ships a skip-hash.** | `parity.PARITY_SKIP_HASH` + `skip_indices` in the manifest | 🟢 **PASS** (train); val has `skip_count: 0` recorded |
| **SC-4** | ⭐ **Data efficiency is MEASURED, not asserted.** A curated set of size N must beat a random set of size N **on a fixed model, fixed seed budget, fixed eval**, reported with a **paired episode-cluster bootstrap** CI and **all four metric families**. | a pre-registered A/B under `TANITAD_PROGRAMME.md` §3 with both outcomes committed in advance | 🔴 **NOT ATTEMPTED** — 3 probes, 0 hits (§3.4) |
| **SC-5** | **Corpus census is reproducible BY CONTENT.** Two independent runs on the same fleet agree; a member set that is renamed, truncated or substituted is detected. | `tools/corpus_census.py` extended to a per-member digest + a deliberate-regression arm that must FAIL | 🔴 **FAIL** — glob counts only (§3.4) |
| **SC-6** | **Parity is enforced by the BUILD, not by the operator.** Every corpus-materialising entry point calls the ingest gate; the gate's population is DERIVED from source, never hand-listed. | `stack/tests/test_build_parity_guard.py` (568 L) derives the door set | 🟢 **PASS** — and it already documents its one known hole (`rebuild_pai_rolling.py` calls `save_episode` directly and carries its own gate) |
| **SC-7** | **No inference-time input contains anything its own label was derived from.** | per-head admissibility test; `s2_labels` already re-asserts disjointness **on the consumed bytes**, and `s2_derive` uses a structural allowlist | 🟡 **PARTIAL** — enforced for S2 goals; **not enforced corpus-wide** (§7) |
| **SC-8** | **A licence-scope violation is impossible, not merely discouraged.** No export of any kind can emit a row outside its declared scope. | `license_guard.verify_license_scope` raises on the FIRST violation; `gated-confidential`/`refuse` can never be in a scope | 🟡 **PARTIAL** — the mechanism is correct; it is not wired into CI or into any non-lake export path |
| **SC-9** | **Curation reaches the model.** A `curation_weight` produced by P2 is observable in the sampler of a real training run. | grep the running process + a trainer test | 🔴 **FAIL** — zero non-test consumers (§3.3) |
| **SC-10** | **Every adapter emits the identical window contract.** | one shared conformance test parameterised over every adapter; must FAIL on a deliberately dropped key | 🔴 **FAIL** — `Comma2k19Dataset` drops `future_actions` and the existing guard cannot see it (§3.2) |
| **SC-11** | **No load-bearing corpus is at fewer than 2 DURABLE copies.** "Durable" excludes rented pods (`corpus_census.is_durable_host:109`). | `tools/corpus_census.py`, exit code non-zero on a single copy — run on a schedule, not on demand | 🔴 **FAIL as last measured** — parity TRAIN and VAL both `SINGLE_COPY` (§3.4). ⚠️ INHERITED (2026-08-03), not re-run. |
| **SC-12** | ⭐ **Every corpus is REBUILDABLE, not merely copyable.** A committed manifest (source ids + build-params hash + per-episode `sha256`) lets any host reproduce the exact corpus from origin. | `DATA_LAKE_ARCHITECTURE.md:65-67` **I-D**; the artifact would be `stack/DATA_MANIFEST.json` + `scripts/rebuild_cache.py` | 🔴 **FAIL** — both are referenced (`DATA_LAKE_ARCHITECTURE.md:45`, `Data Engineering/BACKLOG.md:80-88`) and ⚠️ **UNVERIFIED to exist**; I could not locate either. Settling probe: `git ls-files \| grep -E "DATA_MANIFEST\|rebuild_cache"`. |
| **SC-13** | **A published dataset renders a card a human can read, and the card's claims are true of the bytes shipped.** | a post-push fetch that reads the rendered card and cross-checks its counts against the staged manifest | 🔴 **FAIL** — the exporter writes a filename HF ignores; the last push rendered blank (§3.4) |
| **SC-14** | **Every corpus-selection artifact passes a column-semantics contract before it is used.** | `validate_pool()` over every selection parquet (`r0_selection.parquet`, `phase0_selection.parquet`, `r0_selection_v2.parquet`, the TanitDataSet catalogs) | 🟡 **PARTIAL** — the contract exists and is machine-checked for the v2 clean-val pool (34/34 on 18,988 rows); `Data Engineering/BACKLOG.md:20-23` records that *"the same class of column ships in all of them"* and is unapplied |

---

## 7. Invariants (binding — a P2 change that breaks one is refused, not reviewed)

1. ⛔ **Parity is sacred.** The canonical train corpus is `physicalai-train-e438721ae894`
   (2,376 episodes) with skip-hash `f09e44db` (`parity.py:78-82`). **Anything that re-selects
   episodes must be refused.** Corpus enlargement is a **new, declared parity key** with the
   evaluation set unchanged — never a silent widening (`DATA_STRATEGY.md:759-761`).
2. ⛔ **Labels may use ego and other privileged signals; INFERENCE IS VISION-ONLY.**
   (PI, 2026-08-03.) `situations.py:62-69` states the enforcement: `x`, `y`, `yaw` and every
   derived curvature are *privileged geometry*, never a model input. The generalisation is the
   binding one: **for any head, ask whether its inference inputs contain something its label was
   derived from.**
3. ⛔ **Goal signals stay information-disjoint from the situation classifier.**
   (PI, 2026-08-03.) Enforced structurally where it has been implemented: `s2_derive.py` reads
   Engine A through `ENGINE_A_ALLOWED` so `situations` is *unreadable*, and `s2_labels.py`
   re-asserts the stamp **on the bytes it consumes** rather than trusting the builder — and
   deliberately scans the goal **payload only**, because scanning the whole record would match
   the record's own `disjointness` stamp key.
4. ⛔ **`gated-confidential` and `refuse` never enter the lake and never enter an export scope**
   (`license_guard.py:41-48`; `schema.py:141-147`). PhysicalAI-AV is `gated-confidential`; Waymo
   and Waymax are `refuse` because their terms follow the trained **weights**, not just the data.
5. ⛔ **Licence class is a CONSTANT per source, never inferred** (`schema.py:49-51`), and it is
   **per record DOI where the publisher is not the licensor** — the DLR OpenDRIVE case, where the
   same authors' same test bed ships CC-BY-4.0 and CC-BY-NC-SA-4.0 under different DOIs
   (`schema.py:92-106`).
6. ⛔ **A derivative inherits the STRICTEST input tier.**
   `tier(derivative) = strictest( tier(source_record), tier(generator_model), tier(conditioning_labels) )`
   (`Data Engineering/TANITDATASET_TIER_INTEGRATION_2026-07-21.md:67-75`). Concretely: **a
   Cosmos-Drive-Dreams render of a PhysicalAI clip is still `gated`**, and the A2 augmentation set
   is **inside** the public firewall, not outside it (`DATA_STRATEGY.md:780-782`) — publishing it
   on HF does not make it publicly claimable. This is the invariant most likely to be violated by
   a well-meaning augmentation pipeline.
7. ⛔ **`Keys.txt` is never committed**; tokens are read in place, never copied into args or logs.
   ⚠️ C111 proved the weaker half is the real risk: **a token in a run log is protected by
   nothing.** Bulk imports are credential-scanned before staging (`tools/secret_scan.py`).
8. **Verify by CONTENT, never by exit code, file count, filename or listing.** A listing sees a
   *missing* file, never a *short* one. Standing check: a conservation count `n_out == n_in` per
   stage per batch (`DATA_STRATEGY.md:857-860`).
9. **Absence found at one location is not absence.** Two probes, two names, and the tool that
   owns the fact — before any "X does not exist" enters a P2 document. §3.5 is this session's
   example of the rule paying for itself.
10. **Never add load to a training pod; never eval on a training pod.** P2 builds are heavy and
    this has already killed a flagship mid-checkpoint.
11. **Two-key quotes.** Any number in a P2 report names its artifact path + evidence class.
12. **State the instrument's inclusion rule with every count.** Three instrument-scope errors in
    one week are named as a family at `DATA_STRATEGY.md:848-849` — C113's contamination
    denominator (catalogue vs buildable set: **4.25 % vs 78.21 %**), C110's `SUFFIXES = (".py",
    ".sh")` census (missed **46 of 102** stranded files), and C87's cropper. All three are *"a real
    number answering a narrower question than the one asked."* ⇒ **this is a required schema field
    on a P2 count, not a convention.**

---

## 8. Open questions for the PI

| # | question | why it needs the PI, not an agent | blocks |
|---|---|---|---|
| **Q1** | **Does the 4,472-clip w120 build go ahead, and under what parity key?** `DATA_STRATEGY.md:879` (§12 row 2) is unchanged and sharpened: the build would pull **15 % of the deployed val set into training**, and the 201 already-built clips are *inside* the train corpus. | it re-scopes the sacred corpus; only the PI may declare a new parity key | the single biggest planned data job — and P0, the only lever that closes the 443× gap |
| **Q2** | **The licence conflict on PhysicalAI/Alpamayo** — `TANITDATASET_V1_STRATEGY.md` reads *no-derivatives → firewalled, recipe-only*; `ALPAMAYO2_SUPER_ANALYSIS.md` cites an OpenMDW-1.1 derivative permission (`DATA_STRATEGY.md:783-788`). Recorded rather than resolved-by-whoever-read-last. | a licence reading, with legal exposure | every public claim; SC-2 for the parity corpora |
| **Q3** | **Do we spend a GPU-day on SC-4 (curated vs random, same size)?** It is the only experiment that converts "data moat" from a slogan into a measurement — and `DATA_STRATEGY.md:63-67` predicts the answer is a **single-digit factor**, i.e. it may be worth doing precisely to *stop* over-investing in curation. | it costs metered/local GPU and pre-registers a result that could de-prioritise a whole FlyWheel workstream | P2-4, P2-5, and the ranking of the entire P2 backlog |
| **Q4** | **Who owns `calib.py` and the IDM scripts — P2 or P3?** | a product-boundary call | P3's spec, and whether P2-13 covers them |
| **Q5** | **Is `AGENT_CHARTERS.md` real?** It does not exist in this worktree at three probes. If it exists elsewhere, §6 must be reconciled against its criteria before this spec is cited. | only the PI/Master Mind knows which checkout is canonical | citability of this spec |
| **Q6** | **Rotate the exposed HF token.** `DATA_STRATEGY.md:831-832`: *"the token is still in plaintext on Thor"*. The scanner half of that row is now **closed** (§3.5); the rotation half is not, and it is time-sensitive. | credential rotation is PI-only | nothing technical — but it is live exposure |

---

## 9. How this document was measured

Every `path:line` above was read from source in this worktree on **2026-08-23**, with the retry
discipline described in the method note. The corpus-adapter table in §3.2 is a delegated
file:line survey; its three load-bearing claims (`Comma2k19Dataset` missing `future_actions`,
`metadrive_frontcam.py:295` `NotImplementedError`, `data/__init__.py` exporting 4 modules) were
**re-read directly by me** and are marked ✓. Nothing in this document is quoted from a summary,
a weekly report or a changelog.

The `TanitAD Research Hub/Data Engineering/` tree (§3.0) is a second delegated file:line survey —
55 `Implementation/incoming/` packages, `GOALS.md`, `DATA_LAKE_ARCHITECTURE.md`,
`DATASET_LANDSCAPE.md`, `Data Engineering/BACKLOG.md`. Its one claim I re-verified myself is the
credential-scanner contradiction (§3.5), which I found independently before the survey returned it.

**Deliberately NOT verified here (state them as UNVERIFIED if cited):**

- ⚠️ **~25 documents in that tree could not be read after up to 80 retries** on the Drive mount —
  including `TANITDATASET_V1_STRATEGY.md` (the doc `DATA_STRATEGY.md:784` cites for the licence
  conflict), `OWN_DATASET_PLAN.md`, `DATA_STRATEGY_FOR_HIERARCHY.md`, `S2_STRATEGIC_GAP.md`,
  `PERCEPTION_FLOOR_UNIFY.md`, `SAM3_EXTRACTION_V2.md`, and all of
  `Research/2026-08-19-alpamayo-screening/` (which contains a file named
  `WINDOW_ALIGNMENT_DEFECT.md` and is **referenced nowhere in `DATA_STRATEGY.md` v4.0**). **Their
  absence from this spec is an artifact of the mount, not evidence about their content.** The
  settling probe is a retry loop from a host with a materialised checkout.
- Whether **`stack/DATA_MANIFEST.json` and `scripts/rebuild_cache.py` exist**. Both are referenced
  (`DATA_LAKE_ARCHITECTURE.md:45`; `Data Engineering/BACKLOG.md:80-88`) and neither was located.
  Settling probe: `git ls-files | grep -E "DATA_MANIFEST|rebuild_cache"`, retried.
- Whether the **`data:physicalai` tag-audit script** exists (`Data Engineering/BACKLOG.md:163-164`
  specifies it; it was not found). This is the *enforcement* half of the licence firewall
  (`DATA_STRATEGY.md:775-776`), so its absence would mean exposure is auditable in principle only.
- Whether a **DLR OpenDRIVE adapter** exists (the `SOURCE_REGISTRY` entry does). Settling probe:
  `grep -rn "dlr_opendrive\|xodr" stack/tanitad stack/scripts`, retried — mine timed out.
- Whether the **on-disk corpora** are present on Thor or any pod **today**. The durability numbers
  in §3.4 are **INHERITED from a 2026-08-03 census**. Settling probe:
  `python tools/corpus_census.py --json` from a host with fleet ssh; not run.
- The **v5/v6 live training run's** current data flags. The settling probe is grepping the flags
  out of the RUNNING process, never reading the supervisor manifest (CLAUDE.md §Traps).
