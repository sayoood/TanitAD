# P8 — TanitDataSetCreator — SPEC

`STATUS: DRAFT v1, 2026-08-23. Owner: TanitAD_DataFlyWheel. Authority: Project
Steering/TANITAD_PROGRAMME.md §1 (P8) · §3 (work-package schema) · §6 (quality) ·
§8 (/TanitAD_DesignDataSet) · §9 (what binds).`
`This SPEC is written BEFORE the code, per §3 "Spec before code. Tests before results.
Both-outcomes before launch." Nothing here is implemented yet unless §2 says EXISTS.`

**Evidence-class key** — `MEASURED` (ours, artifact path given) · `PUBLISHED` (cited) ·
`INHERITED` (another agent/doc, not re-verified here) · `ESTIMATED` · `HYPOTHESIS`.
Every capability claim in §2 carries `file:line`. Nothing in this document quotes a
number that is not traceable to source.

⚠️ **Filesystem caveat on re-verification.** This worktree is on a Google Drive mount
whose files are cloud placeholders. `Read`, ripgrep, `cat` and `Select-String` fail
intermittently with `EISDIR` / `Invalid request code` / `Unzulässige Funktion` on a
non-hydrated file — and ripgrep returns *"No matches found"* rather than an error, which
reads exactly like absence. Every citation below was extracted with a retry loop. **If you
re-verify a line number and get an I/O error or an empty result, retry before concluding
the file is absent.** (Same family as the `df`-reports-the-cluster trap: a probe that
fails in the shape of an answer.)

---

## 1. Purpose & scope

### 1.1 What P8 is

**TanitDataSetCreator turns the sentence *"I want a dataset with these properties"* into a
reproducible, parity-classified, provenance-stamped dataset — and refuses to produce one
whose comparability status is unclear.**

Per `Project Steering/TANITAD_PROGRAMME.md:43`, P8 is a *"clickable + CLI dataset
configurator on top of TanitScena / public corpora"*. The configurator half is the easy
half. The load-bearing half is the second clause of §1.2.

### 1.2 The constraint that shapes the whole design

`CLAUDE.md §Invariants`, verbatim:

> **Parity is sacred:** the canonical train corpus is `physicalai-train-e438721ae894`
> (2376 episodes) with skip-hash `f09e44db`. Anything that re-selects episodes breaks
> cross-arm comparability and must be refused.

**A dataset configurator is, by definition, an episode-re-selection machine.** P8 is
therefore the single product in the programme whose core function is the thing the
programme's most sacred rule forbids. This is not a footnote to work around; it is the
design centre. §5 is the answer, and it is a mechanism, not a promise.

The reconciliation already exists in PI-sanctioned prose —
`DataEng/DATA_STRATEGY.md:761`:

> "Corpus enlargement (P0) is therefore a **new, declared parity key**, never a silent
> widening, with the evaluation set unchanged."

⇒ **P8 does not weaken parity. P8 makes the declaration mechanical.** Re-selection is
permitted, always produces a *different* corpus identity, and is *structurally* barred
from being compared against parity arms. What P8 removes is the human step where somebody
was supposed to remember to declare it — the exact defect class named in
`TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-18-build-parity-guard/BUILD_PARITY_GUARD.md:10-14`,
where C113 ended with *"whoever runs that build must call `parity.filter_train_clips()`
first"* and the report's own verdict was **"That sentence was the defect."**

### 1.3 Scope boundaries

| Boundary | P8 does | P8 does NOT do |
|---|---|---|
| **vs P5 TanitScena** | consumes a scenario/situation index and compiles its tags into selection predicates | own, build, or serve the index |
| **vs P2 Data pipelines** | emits a **spec** and a **resolved selection**; declares the parity relation; stamps provenance | execute the ingest/decode/encode work — P2's builders do that (`lake/ingest.py`, `epcache.build_episodes_cached`, `scripts/v2_compressed.py`) |
| **vs P4 Training** | hands a spec whose digests the trainer records | choose an architecture, a schedule, or a loss |
| **vs P7 TanitEval** | hands a dataset carrying a `comparability_key` | compute metrics or decide a gate |
| **vs P3 DataReconstruction** | can *select over* reconstructed data once it carries a licence class and a source key | reconstruct video, estimate calibration, or run IDM |

⛔ **The one-sentence rule:** *P8 decides WHICH data. It never decides HOW the data is
made, and it never decides WHAT is trained on it.*

⚠️ **A boundary that is currently mis-stated in the programme and must be corrected.**
The brief for this work package assumed *"P5 TanitScena supplies the searchable scenario
index; P8 composes datasets from it."* **That is not true of the code today.** MEASURED:
`stack/tanitad/scena/` indexes **prose**, not clips — `stack/tanitad/scena/README.md:1-6`
says it turns the Opponent-Analyzer's `SCENARIO_DATABASE.md` **SC-01..SC-14** catalogue
into a searchable app; `vector.py:163` builds the index over `s["id"]`, i.e. the strings
`SC-xx`; `vector.py:37-41` embeds only `title`/`description`/`correct_behavior`/`tags`.
The corpus is **14 markdown documents**. There is **no path from TanitScena to an
`ep_*.pt`**, and `parse.py:108-130` states that the `data_sources[].link` fields resolve
to public landing pages, explicitly *not* real repo ids.
⇒ **The P5→P8 interface described in §9.1 is a contract to be built, not a wire to be
connected.** P8 must ship useful without it (§3.4 predicate families 2–6 need no P5).

---

## 2. What already exists — survey before design

Every row is MEASURED from source in this worktree unless marked otherwise.

### 2.1 Parity machinery — EXISTS, and it is strong

| Capability | Status | Evidence |
|---|---|---|
| Corpus identity constants | **EXISTS** | `stack/tanitad/data/parity.py:78-82` — `PARITY_TRAIN_KEY = "physicalai-train-e438721ae894"`, `PARITY_VAL_KEY = "physicalai-val-0c5f7dac3b11"`, `PARITY_SKIP_HASH = "f09e44db"`, `PARITY_TRAIN_EPISODES = 2376`, `PARITY_VAL_EPISODES = 600` |
| A refusal type that kills a run | **EXISTS** | `parity.py:108-111` `class ParityViolation(SystemExit)` — subclasses `SystemExit` deliberately so a pod supervisor sees a non-zero exit |
| Set-identity digest | **EXISTS** | `parity.py:127-137` `uid_digest(uids)` = `sha256("\n".join(sorted(uids)))`; raises `ValueError` on duplicate uids |
| Per-element membership oracle | **EXISTS** | `parity.py:1858-1864` `clip_digest(clip_id)` = `sha256(clip_id.encode("utf-8"))`; the committed set is `parity_train_clip_digests.json` (2400 entries) |
| Episode-set check, strict + subset modes | **EXISTS** | `parity.py:325` `check_uids(uids, *, corpus_key, label, cache_dir, mode, manifest_path, subset_note)`; count check `:370`, digest check `:394-402` |
| Trainer entry point | **EXISTS** | `parity.py:467` `assert_parity_corpus(...)`; refusal text at `:491-495` |
| **The ingest gate** | **EXISTS** | `parity.py:2278-2283` `guard_corpus_build(clip_ids, *, label, role="", mode="refuse", path=None, val_path=None, sanctioned_audit=None) -> tuple[list[str], dict]` |
| Gate oracle preflight | **EXISTS** | `parity.py:2430` `require_ingest_gate(where)` |
| Roles and modes | **EXISTS** | `parity.py:2268-2275` — `CORPUS_ROLES_SUPERVISION = ("train","augmentation")`, `CORPUS_ROLES_HELDOUT = ("val","eval")`, `+ ("audit",)`; `CORPUS_GUARD_MODES = ("refuse","exclude")` |
| Contamination checks, both directions | **EXISTS** | `parity.py:1962` `assert_eval_clips_disjoint_from_parity_train`, `:2037` `filter_eval_clips`, `:2115` `assert_train_clips_disjoint_from_deployed_val`, `:2177` `filter_train_clips` |
| Reader-side firewall | **EXISTS** | `parity.py:1789-1801` `assert_not_parity(*paths, label)` — turns a docstring claim into an assertion |
| Geometry-sibling registration | **EXISTS** | `parity.py:806` `register_geometry_sibling`, `:1577` `register_v2_geometry_sibling` |
| Registered-subset ("deployment") pattern | **EXISTS** | `stack/scripts/make_parity_clip_digests.py:155-229` `build_deployment(...)`; the committed instance is `stack/tanitad/data/deployed_val40_clip_digests.json` (`is_full_corpus: false`, `n_clips: 40`, `cross_check_episodes: 40`) |
| Admissible / inadmissible deployment registry | **EXISTS** | `parity_manifest.json` → `corpora["physicalai-val-0c5f7dac3b11"].known_deployments` (600, 40) and `.deployments_seen_but_NOT_admissible` (12, with `why_not`); enforced at `parity.py:692-698` |
| No env-var kill switch | **EXISTS (by design)** | `parity.py:62-64` — *"There is deliberately no environment variable that disables the content check"* |
| Gate wired at the corpus writers | **EXISTS, 6 doors** | `epcache.py:114-117`, `scripts/rebuild_pai_rolling.py:130-131`, `scripts/v2_to_pilot.py:95-96`, `scripts/v2_compressed.py:416,418`, `scripts/aug120_pipeline.py:79-80`, `scripts/slice_v2_cache.py:121-122` |
| The guard's own guard | **EXISTS** | `stack/tests/test_build_parity_guard.py` — 568 lines, 24 tests; the door population is **derived by AST walk** (`:153` `derive_corpus_writers`), not hand-listed; a neuter matrix of 8 deliberate-regression cases is banked all-RED at `…/2026-08-18-build-parity-guard/raw/neuter_matrix.txt` |

**The parity key derivation — the fact P8's whole guard rests on.**
`stack/tanitad/data/epcache.py:62-67`:

```python
def cache_key(sources: list, params: dict) -> str:
    """12-hex cache key over the ordered source identities + build params."""
    ids = [_source_id(s) for s in sources]
    return hashlib.sha1(
        json.dumps({"ids": ids, "params": params}, sort_keys=True,
                   default=str).encode()).hexdigest()[:12]
```

⭐ **`e438721ae894` IS a selection digest.** It is `sha1` over the *ordered source
identities plus the build params*, truncated to 12 hex. ⇒ **A configurator that changes
the selection cannot produce that key; it produces a different one, by arithmetic.** The
parity guard is therefore not a policy layered on top of P8 — it is a consequence of the
existing key derivation that P8's job is to *expose and enforce*, never to hide.

### 2.2 The three MEASURED holes P8 must not inherit

1. ⛔ **The skip-hash is STAMPED, NOT CHECKED.** `PARITY_SKIP_HASH` appears five times in
   `parity.py` and **not one is a comparison**: the definition (`:80`), two record stamps
   (`:364`, `:1755`), two print statements (`:406`, `:1782`). A cache with zero skip
   markers still gets `"skip_hash": "f09e44db"` written into its provenance record and
   printed in the VERIFIED banner. The only real check lives pod-side in
   `stack/scripts/pod_ops/compute_skipset.py:86-91`, which no trainer invokes. Note also
   that `f09e44db` is the **first 8 hex of a full sha256** (`compute_skipset.py:25-26`)
   over a **comma-joined** canonical form (`compute_skipset.py:3-4`,
   `stack/scripts/parity_skipset.sh:34-36`) — *different* from `uid_digest`'s newline-join.
   ⇒ **§4.4: if a P8 manifest carries a skip-hash it MUST compute it, and a test must
   prove the computation is not a copy.** A copied constant is a promise wearing an
   enforcement costume.
2. ⚠️ **The parity VAL split has no content digest.** `parity_manifest.json` →
   `corpora["physicalai-val-0c5f7dac3b11"].episode_uid_sha256` is `null`,
   `uid_source: "count-only-unrecorded"`, and `check_uids` degrades to the COUNT-ONLY
   branch (`parity.py:384-391`). A *substituted* val set of the right size passes.
   ⇒ **§5.6: P8 may not claim a val-side selection is pinned.** Backlog `P8-11`.
3. ⚠️ **Corpus identity is a path SUBSTRING, not a content hash.** `parity.py:238-241`
   resolves a corpus key by longest-first substring match on the resolved POSIX path.
   Naming any directory so it contains `e438721ae894` makes it "the parity corpus" as far
   as key resolution goes. **P8 lets a user name the output directory** ⇒ this is P8's
   single largest new attack surface, and §5.2 Rule G1 closes it.

### 2.3 Licence machinery — EXISTS, and it is the strongest part of the lake

| Capability | Status | Evidence |
|---|---|---|
| Licence classes | **EXISTS** | `stack/tanitad/lake/schema.py:44` `LICENSE_CLASSES = ("owned-safe", "nc-research", "gated-confidential", "refuse")` |
| Per-source constant registry (20 sources) | **EXISTS** | `schema.py:68-148` `SOURCE_REGISTRY: dict[str, SourceLicense]`; `schema.py:19` — *"set by the ingestor from a per-source CONSTANT … never inferred, so it cannot drift"* |
| Licence record shape | **EXISTS** | `schema.py:47-61` `SourceLicense(license_class, license_name, share_alike, is_synthetic)` + derived `commercial_ok` property |
| Ingest-time refusal | **EXISTS** | `schema.py:296-307` `assemble_lake_record` raises `PermissionError` on `gated-confidential` and on `refuse` |
| Export-scope refusal | **EXISTS** | `lake/license_guard.py:18` `class LicenseScopeError(PermissionError)`; `:22-25` `verify_license_scope(rows, allowed_classes, require_commercial_ok=False, forbid_share_alike=False, context="export")`; `:41-48` refuses to even *construct* a scope containing `gated-confidential` or `refuse` |
| Physical partitioning | **EXISTS** | `lake/catalog.py:29` `PARTITION_COLS = ["license_class", "source", "split"]`; `lake/shards.py:39-46` inserts a literal `sharealike` path component |
| Tier derivation | **EXISTS** | `lake/filtering.py:26` `TIERS = ("ship", "ship-sa", "nc", "firewalled")`; `:29` `tier_of(license_class, share_alike, commercial_ok)`, raises on `refuse` at `:39-41` |
| PhysicalAI is firewalled from the lake | **EXISTS** | `schema.py:146` `"physicalai_av": SourceLicense("gated-confidential", "NVIDIA-AV-internal", …)`; `schema.py:66-67` — *"listed ONLY so an ingestor that tries to admit it fails loudly"* |
| Licence tracking under `DataEng/` | **MISSING** | `DataEng/` holds 2 files; licence handling there is prose only (`DATA_STRATEGY.md:773-795`) |

⚠️ **A live régime split P8 must handle explicitly, not paper over.** `SOURCE_REGISTRY`
classes `physicalai_av` as `gated-confidential` — never ingestible into the lake
(`schema.py:146`) — while **every trained arm to date runs on PhysicalAI-AV**
(`DATA_STRATEGY.md:757`). The lake and the epcache/parity path are **two different
régimes**, and the lake's licence firewall does not cover the parity path at all.
⇒ P8 spans both and must carry the régime in the spec (`sources[].regime`, §3.3).

### 2.4 Selection / curation primitives — EXISTS as libraries, MISSING as a product

| Capability | Status | Evidence |
|---|---|---|
| Queryable Parquet catalog | **EXISTS** | `lake/catalog.py:86` `resolve_view(lake_root, filter_expr=None, columns=None, sort_by="episode_id") -> pa.Table`; `:105` `resolve_members(...) -> list[dict]` projecting exactly `["episode_id","shard_key","member_key","sha256","license_class","commercial_ok","share_alike","split","source","split_unit_id"]` (`catalog.py:111-113`) |
| Quality predicates (banded, not binary) | **EXISTS** | `lake/filtering.py` — `blur_band` `:184` (`BLUR_BANDS` `:137`), `exposure_band` `:196`, `truncation_frac` `:214`, `detect_corrupt` `:108`, `egomotion_sane` `:275`, `assign_rig` `:249`, bundled by `assess_quality` `:348` → `QualityVerdict` `:314-328` |
| Curation (distribution shaping) | **EXISTS** | `lake/curation.py:222` `curate_corpus(records, clamp=WEIGHT_CLAMP, holdout_frac=HOLDOUT_FRAC)`; `:55` `inverse_frequency_weights`; `:110` `weakness_boost`; `:150` `safety_event`; `:192` `is_eval_holdout` |
| Deterministic frozen holdout | **EXISTS** | `curation.py:180-181` `HOLDOUT_FRAC = 0.1`, `HOLDOUT_SALT = "taniteval-holdout-v1"`; `:184` `stable_unit_frac` uses `blake2b`, salted, unit-keyed |
| Two-pass dedup | **EXISTS** | `lake/dedup.py:216` `two_pass_dedup(items, radius=NEAR_DUP_HAMMING)` — pHash+LSH within source, then GPS-cell×time across sources; multi-traversal kept and tagged, not collapsed |
| Deterministic corpus mixing | **EXISTS** | `stack/tanitad/data/mixing.py:86` `MixedWindowDataset(sources: list[tuple[Dataset, float]], length=None, seed=0)`; contract check `:103`; `mix_report` `:123` |
| A **view** = a selection | **PARTIAL** | `lake/view.py:38-64` `@dataclass LakeView(lake_root, name, filter_expr, scope)`; `:58` `signature()` = `sha1(sorted((episode_id, sha256)))[:12]` — **content-addressed** |
| A view as a **portable, recorded artifact** | ⛔ **MISSING** | `view.signature()` is used only as a hydrate-cache tag (`view.py:168`). Nothing writes a view manifest; the only persisted file is `DONE` = `{"episodes","hydrated","reused"}` (`view.py:143-144`). `filter_expr` is a live `pyarrow.dataset.Expression` — **a `LakeView` is not serialisable.** A selection is reproducible *if you re-run the same code*; it is not a file you can hand to someone. **This is precisely the P8-shaped hole.** |
| A CLI that resolves + curates + emits | ⛔ **MISSING** | Every lake CLI is ingest or proof: `scripts/lake_ingest.py`, `scripts/lake_byteproof.py`, `scripts/build_tanitdataset.py`, `scripts/ingest_nuscenes.py`. Repo-wide, `curation`, `verify_license_scope` and `enrich_lake` have **zero** script consumers |
| A dataset **config file format** | ⛔ **MISSING** | Zero `.yaml`/`.yml` under `stack/`; the only TOML is `stack/pyproject.toml`. No `dataset_config` / `DatasetConfig` / `corpus_config` symbol anywhere. Today's substitutes are shell `TRAIN_CMD=` strings (`stack/scripts/flagship_phase0.run.env:31-32`, `stack/ops/runs.d/flagship-v5f-w120-30k.env:10`), the committed `parity_manifest.json`, and code-level `LakeView` |
| Public package surface | **PARTIAL** | `lake/__init__.py:60-75` exports **11 names**; `curation`, `view`, `catalog`, `license_guard`, `proof`, `dedup`, `shards`, `vocab`, `hf_export` are **not** in `__all__` |

**The nearest existing prototype** is `stack/scripts/build_tanitdataset.py` — a genuine P8
ancestor. It already carries two tiers (`TanitDataSet-C` = permissive/`ship`,
`TanitDataSet-R` = C ∪ NC/`nc`, `:13-18`), a deterministic split
(`split_map_by_id(by_id, val_every=5)` over content-stable ids, `:72-80`), a
`params_hash`, and two structural refusals stated in its own header (`:20-23`): it can
never admit the gated parity corpus, and *"it NEVER pushes"*. **What it lacks is exactly
what P8 adds: a config file, arbitrary predicates, a parity relation, and a provenance
manifest.**

### 2.5 Census, scenario index, and the situation labels

| Capability | Status | Evidence |
|---|---|---|
| Corpus **durability** census | **EXISTS** | `tools/corpus_census.py` — 9 hardcoded artifacts (`:134-279`), CLI `--json --no-hf --hosts --timeout` (`:704-710`), exit `2`/`1`/`0` (`:672-678`) |
| Census as a **content index** | ⛔ **MISSING** | It counts copies over machines, not contents. No per-episode or per-clip data reaches the output. It *reads* LFS sha256 at `:440-443` and then **discards it** — `count_hf_members()` (`:450-467`) consumes only `meta["size"]` |
| Census parity assertion | ⚠️ **DECLARATIVE ONLY** | `Artifact.parity` is a hand-written string, e.g. `corpus_census.py:140` `parity="physicalai-train-e438721ae894 / skip-hash f09e44db / 2376 eps"`, copied verbatim to JSON at `:633` and **never verified**. Same defect class as hole #1 in §2.2 |
| Census wired into CI | ⛔ **MISSING** | `tools/ci_gate.py` has no `corpus_census` reference; flagged open at `…/2026-08-03-corpus-durability/CORPUS_DURABILITY.md:316` |
| Semantic search over **scenarios** | **EXISTS** | `scena/vector.py:134` `VectorIndex`, `:212` `search(query, k=5)`, persisted `vectors.npz` `:225`; server `scripts/scena_app.py:116` `build_app(...)`, routes `/api/search` `:178` |
| Semantic search over **clips/episodes** | ⛔ **MISSING** | index ids are `SC-xx` strings (`vector.py:163`); corpus = 14 markdown entries |
| Situation classes | **EXISTS (2 of 3 powered)** | `data/situations.py:269` `detect_lane_change`, `:392` `detect_intersection`, `:352` `detect_turns`, `:373` `detect_curves`; `:318` `detect_roundabout` is **UNPOWERED** (26 clusters, `:18`) and PI-deferred |
| Situation label provenance | **EXISTS, ego-only** | `situations.py:34-39` and `:62-66` — every label is a pure deterministic function of `P = [x, y, yaw, v]`; `scripts/emit_situation_labels.py:53-58` reads only `d["poses"]` and calls `detect_intersection(K)` with `cross` defaulting to `None`. **No pixels, no maps, no annotations enter the label.** |

### 2.6 What a "dataset" concretely IS on disk — three incompatible things

P8 must speak all three, and say which one it emitted.

1. **epcache** — `<cache_root>/<tag>-<sha1[:12]>/{ep_%05d.pt, skip_%05d, DONE}`
   (`epcache.py:85-90`). Each `.pt` is `{frames_u8, actions, poses, episode_id[, maneuvers]}`
   (`mixing.py:53-61`). Read with `mmap=True`.
2. **v2 compressed cache** — `<dir>/{*.v2ep.pt, _v2manifest.pt}`, JPEG/PNG-encoded frames
   decoded per slice (`v2_dataset.py:3-8`, `:63-64` `MANIFEST_NAME`/`MANIFEST_VERSION = 3`).
   Chosen because the decoded corpus is ~1 TB and cannot be held in RAM.
3. **lake shards** — WebDataset-convention `.tar` (stdlib `tarfile`, `mtime=0` for
   reproducible bytes, `shards.py:110`) + a Hive-partitioned Parquet catalog
   (`catalog.py:28-29`).

Plus sidecar label archives (`.npz` from `emit_situation_labels.py:70`) and the committed
identity manifests (`parity_manifest.json`, `parity_train_clip_digests.json`,
`deployed_val40_clip_digests.json`).

The window contract is shared: `_contract.py:8-11` — `frames [T,1,H,W]` float32 in [0,1],
`actions [T,2]` (steer rad, accel m/s²), `poses [T,4]` (x, y, yaw, v), `episode_id: int`;
windowing arithmetic `t_max = T - window - max_horizon` (`_contract.py:119-121`).

### 2.7 How training currently selects data — the gap P8 closes at the P4 seam

⛔ **There is no `--corpus` flag on any trainer and no `--parity-key` flag anywhere in the
repo.** Corpora are selected by filesystem path and identity is *inferred* from that path
by substring match.

- `stack/scripts/train_flagship_v4.py:1800` — `--train-cache` / `--val-cache`, **no help text**
- `:1808` `--v2-train-cache` (`nargs="+"`), `:1813` `--v2-val-cache`
- `:1835` `--require-parity`, `:1851` `--parity-off-reason` (a reason, deliberately not a boolean)
- `stack/tanitad/train/train_worldmodel.py:482` — `--data {toy,comma2k19,physicalai,realmix,mix,cached}` is the **only symbolic corpus selector in the repo**
- `stack/scripts/v2_to_pilot.py:161,169,173,175` — `--corpus`, `--corpus-role`, `--exclude-parity-overlap`, `--sanctioned-audit`: the closest thing to a declared-role selector, and the model P8's CLI follows

---

## 3. The configuration model

### 3.1 What a dataset config IS

**A `DatasetSpec` is a single JSON file, `<name>.tanitds.json`, schema
`tanitad.datasetspec/1`.** It is the *complete* declaration of a selection: sources,
licence scope, parity relation, predicates, curation, split policy, window/horizon,
labels, seed, output. It is human-editable, diffable, and reviewable, and it is the object
the clickable UI writes.

**Why JSON, not YAML.** The repo has **zero** `.yaml`/`.yml` files under `stack/` and the
entire provenance stack (`parity_manifest.json`, clip digests, run reports) is JSON.
Adding a YAML parser dependency for one file buys comments and costs a dependency plus a
second serialisation with different canonicalisation rules — and canonicalisation is
load-bearing here (§4.2). Comments are carried by an explicit `notes` field.

⚠️ **Where the file may live.** `.gitignore` ignores `data/` and `*.parquet` wholesale.
Specs are **program artifacts, not data**: they live under
`products/P8-datasetcreator/specs/` (committed) or beside their work package per
`TANITAD_PROGRAMME.md §3`. A spec written into `data/` would be silently un-committable.

### 3.2 The schema

```jsonc
{
  "schema": "tanitad.datasetspec/1",

  // ---- identity ---------------------------------------------------------
  "dataset_id": "tanitds-lowspeed-stopgo-v1",   // slug; becomes the corpus tag
  "title":      "Stop-and-go longitudinal set, comma2k19 + L2D",
  "notes":      "free text; excluded from spec_sha256 (see §4.2)",
  "created_on": "2026-08-23",                   // excluded from spec_sha256
  "author":     "…",                            // excluded from spec_sha256

  // ---- sources ----------------------------------------------------------
  "sources": [
    { "source": "comma2k19",                    // MUST be a key of lake.schema.SOURCE_REGISTRY
      "regime": "lake",                         // "lake" | "epcache" | "v2"   (see §2.3)
      "roots":  ["C:/…/comma2k19-val-61c46fca8f7f"],
      "expect_license_class": "owned-safe" }    // ASSERTED against SOURCE_REGISTRY, never inferred
  ],

  // ---- licence ----------------------------------------------------------
  "licence": {
    "scope": ["owned-safe"],                    // subset of LICENSE_CLASSES minus the two hard-fail
    "require_commercial_ok": true,
    "forbid_share_alike": false,
    "release": "private"                        // "private" | "internal" | "public"  (see §10 Q1)
  },

  // ---- parity relation --------------------------------------------------
  "parity": {
    "relation": "non-parity",                   // identity | geometry-sibling | recorded-view | non-parity
    "of_corpus_key": null,                      // required unless relation == "non-parity"
    "corpus_role": "train",                     // parity.CORPUS_ROLES
    "overlap_mode": "refuse",                   // parity.CORPUS_GUARD_MODES
    "sanctioned_audit": null                    // required iff corpus_role == "audit"
  },

  // ---- selection predicates --------------------------------------------
  "select": {
    "scenario":  { "tags_any": [], "tags_all": [], "scena_ids": [] },
    "situation": { "classes_any": ["stop_and_go"], "min_events": 1,
                   "lead_s": 3.0 },
    "quality":   { "corrupt": "drop",
                   "blur_band_in":       ["sharp", "soft"],
                   "exposure_band_in":   ["ok", "dim", "bright"],
                   "truncation_band_in": ["clear", "partial"],
                   "egomotion_sane": true },
    "dynamics":  { "v_ms": [0.0, 15.0], "abs_accel_ms2_max": 8.0,
                   "abs_kappa_max": null, "min_frames": 80 },
    "geometry":  { "rig_in": null, "camera_model_in": ["pinhole"],
                   "image_size": 256, "hz": 10.0 },
    "dedup":     { "policy": "exemplar_only",   // keep_all | exemplar_only
                   "keep_multi_traversal": true },
    "modality":  { "require": ["has_poses"], "forbid": [] }
  },

  // ---- curation ---------------------------------------------------------
  "curate": {
    "strategy": "none",                         // none | inverse_frequency | weakness_boost | safety_mining
    "clamp": 10.0,                              // curation.WEIGHT_CLAMP
    "weakness_boost": 2.5,                      // curation.WEAKNESS_BOOST
    "target_episodes": null,                    // null = keep all; N = resample to N
    "emit_weights": true                        // weights ride in the catalog, never applied silently
  },

  // ---- split ------------------------------------------------------------
  "split": {
    "policy": "hash_pinned",                    // hash_pinned | every_nth | declared
    "unit": "split_unit_id",                    // the disjointness unit (route/drive/scene)
    "holdout_frac": 0.1,                        // curation.HOLDOUT_FRAC
    "salt": "taniteval-holdout-v1",             // curation.HOLDOUT_SALT — CHANGING THIS MOVES THE HOLDOUT
    "every_nth": null,                          // used iff policy == "every_nth"
    "declared": null                            // used iff policy == "declared": {"train":[…],"val":[…]}
  },

  // ---- windows / frame / labels ----------------------------------------
  "windows": { "window": 8, "max_horizon": 16, "stride": 8, "maneuver_h": null },
  "frame":   { "height": 256, "width": 256, "channels": 9,
               "projection": "pinhole", "f_ref": null, "hz": 10.0 },
  "labels":  ["goal_v3_kinematic", "situations"],

  // ---- determinism + output --------------------------------------------
  "seed": 0,
  "output": {
    "format": "lake",                           // lake | epcache | v2 | selection-only
    "root": "C:/…/tanitad-data/tanitds/…",      // excluded from spec_sha256
    "emit": ["shards", "catalog", "PROVENANCE.json", "DATA_CARD.md", "SELECTION.json"]
  }
}
```

### 3.3 Field rules that are enforced, not documented

| Rule | Enforced by |
|---|---|
| `sources[].source` must be a key of `SOURCE_REGISTRY` | lint refuses an unknown source — a licence class is **never inferred** (`schema.py:19`) |
| `sources[].expect_license_class` must equal the registry value | lint refuses on mismatch; catches a stale spec after a registry correction |
| `licence.scope` may not contain `gated-confidential` or `refuse` | delegated to `verify_license_scope` (`license_guard.py:41-48`) — P8 does not re-implement the refusal |
| `sources[].regime == "lake"` ⇒ the source may not be `physicalai_av` | `assemble_lake_record` raises `PermissionError` (`schema.py:296-307`) |
| `parity.corpus_role == "audit"` ⇒ `sanctioned_audit` non-empty | delegated to `guard_corpus_build` (`parity.py:2313-2319`) |
| A typo in `corpus_role` or `overlap_mode` **refuses**, never degrades | delegated to `parity.py:2302-2312` |
| `parity.relation != "non-parity"` ⇒ `of_corpus_key` required and registered | §5 |
| `split.salt` change ⇒ loud warning + a distinct `selection_sha256` | §4.3 |

⚠️ **Delegation is deliberate.** Every refusal above is *delegated to the module that
already owns it*. P8 re-implementing a licence or parity check would create a second
source of truth that can drift from the first — the failure class the programme calls
*"an instrument structurally unable to report the answer it is cited for"*. P8's job is to
make the existing gates **unavoidable**, not to duplicate them.

### 3.4 The six predicate families and their readiness

| # | Family | Backed by | Status |
|---|---|---|---|
| 1 | `scenario` (P5 tags) | TanitScena | ⛔ **BLOCKED** — P5 indexes prose, not clips (§1.3). Spec field exists; lint refuses a non-empty `scenario` block until `P8-04` lands |
| 2 | `situation` | `data/situations.py:269,392` + `emit_situation_labels.py` `.npz` | **READY (2 of 3 classes)**; `roundabout` refused as UNPOWERED (`situations.py:18`) |
| 3 | `quality` | `lake/filtering.py:108,184,196,214,275` | **READY**; `truncation_frac` is self-labelled a **proxy** (`filtering.py:220-222`) and the data card must say so |
| 4 | `dynamics` | `situations.py:159` `kinematics(P)` | **READY** |
| 5 | `geometry` | `lake/filtering.py:249` `assign_rig`, `RIG_CLUSTERS` `:243-246` | **READY** for the two known rigs |
| 6 | `dedup` | `lake/dedup.py:216` | **READY** |

### 3.5 ⛔ Selection admissibility — the check that generalises the two BINDING rules

`CLAUDE.md` binds two admissibility rules that both reduce to one question:
***does an input at inference contain something the thing being measured also produces?***
Selection is a third place that question must be asked, and it has not been asked before.

**The situation predicates are derived from ego poses ONLY** — MEASURED,
`situations.py:34-39` and `emit_situation_labels.py:53-58`. That is *admissible* under the
PI ruling (*"for ground truth data of scenario classification you can use both ego and
other label, for inference only vision"*), because selection happens offline, like a
label. **But it has a consequence that must be disclosed, not assumed away:**

> A dataset selected by an ego-derived predicate is **not a random sample of driving**. Any
> number measured on it is *conditional on that predicate*. If the same predicate shapes
> both the train pool and the eval pool of a vision-only model, the reported number is
> conditional on a signal the deployed model will not have.

⇒ **P8 emits `selection_conditioning` in every provenance manifest**: the list of
predicate families that were non-trivially applied, each with the privileged channels it
read (`ego_poses`, `maps`, `future_poses`, `agent_tracks`, `vision_only`, `metadata_only`).
⇒ **Rule A1:** if `selection_conditioning` includes a privileged channel **and** the
dataset's declared `split.policy` produces the eval side too, the data card carries the
disclosure verbatim and `comparability_key` (§5.4) records it. This costs nothing and
makes an otherwise-invisible confound visible at the point it is created — which is the
whole lesson of the C6 confound and the REF-A I-JEPA leak.

---

## 4. The determinism & provenance contract

### 4.1 The contract, stated as an obligation

> **The same `DatasetSpec` resolved twice, on any host, in any process order, produces a
> byte-identical selection — and every output ships a provenance manifest, a licence
> class, and a computed skip-hash.**

### 4.2 The four digests

| Digest | Definition | Reuses |
|---|---|---|
| `spec_sha256` | `sha256(json.dumps(spec_minus_volatile, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode("utf-8"))` | — |
| `selection_sha256` | `parity.uid_digest(sorted(member_uids))` = `sha256("\n".join(sorted(uids)))` | `parity.py:127-137` |
| `content_sha256` | `parity.uid_digest(sorted(f"{uid}:{sha256}"))` over per-member content hashes | `parity.py:127-137` + `lake/schema.py:267` `frames_sha256`; the same idea as `view.signature()` (`view.py:62-64`), at full width because this is a provenance record, not a cache tag |
| `skip_hash` | `sha256(",".join(sorted(skipped_ids)))[:8]` — **comma-joined**, matching `parity_skipset.sh:34-36` and `compute_skipset.py:3-4` | ⚠️ deliberately NOT `uid_digest` |

**`volatile` is an explicit, pinned list**: `notes`, `created_on`, `author`,
`output.root`, `sources[].roots`. Everything else digests. The list is pinned by a test
(§4.5 T4) so nobody can quietly add a field to it and make two different specs collide.

⚠️ **Why source *roots* are volatile but source *keys* are not.** The same corpus lives at
different absolute paths on the dev box, Thor and a pod. Digesting the path would make the
same selection produce three digests. Digesting the `source` key plus the resolved member
uids captures identity without capturing the mount point. **The membership check is what
catches a wrong root** — a mis-pointed root resolves to different members and the
`selection_sha256` differs, loudly.

### 4.3 Determinism rules

1. **Sort before you permute.** Every resolution sorts its member set before any
   randomised step. Precedent: `lake/ingest.py:570-572` sorts then `randperm`s, and
   `parity.uid_digest` sorts internally (`parity.py:129`).
2. **One seed, explicitly threaded.** `spec.seed` is the only entropy source. No use of
   the `random` module; every generator is `torch.Generator().manual_seed(seed)`. The lake
   already meets this — MEASURED: no unseeded randomness anywhere in `stack/tanitad/lake/`.
3. **The split is hash-pinned, not seeded, by default.** `split.policy = "hash_pinned"`
   uses `curation.stable_unit_frac` (`curation.py:184`, `blake2b`, salted, keyed on
   `split_unit_id`) so **adding data does not move existing units across the split**. A
   seeded permutation reshuffles everything when the pool grows; that has silently
   invalidated held-out comparisons before.
4. **Changing `split.salt` is a corpus-identity change.** It is inside `spec_sha256` and
   produces a new `selection_sha256`. Never edit it to "rebalance" a split.
5. **Float predicates are compared at declared precision.** Thresholds in `select` are
   round-tripped through the canonical JSON before use, so a spec and its digest can never
   disagree about `0.1`.

### 4.4 The provenance manifest

Every P8 output ships `PROVENANCE.json`, schema `tanitad.datasetprov/1`:

```jsonc
{
  "schema": "tanitad.datasetprov/1",
  "dataset_id": "…",
  "built_on": "2026-08-23", "built_by": "…", "built_host": "…",
  "tanitds_version": "…", "git_commit": "…",

  "spec": { … },                      // the spec VERBATIM, embedded
  "spec_sha256": "…",
  "spec_volatile_fields": ["notes","created_on","author","output.root","sources[].roots"],

  "selection": {
    "n_members": 1234,
    "selection_sha256": "…",
    "content_sha256":   "…",
    "uid_kind": "epcache_basename",   // epcache_basename | v2ep_clipid | lake_episode_id
    "skipped": { "n": 7, "skip_hash": "…", "reasons": {"all_black": 3, "nonfinite_poses": 4} },
    "per_predicate_attrition": [      // n in / n out, PER FAMILY, in application order
      {"family":"quality","n_in":2000,"n_out":1880,"dropped_by":{"blur_band":90,"corrupt":30}}
    ]
  },

  // ---- the parity block: the load-bearing part ---------------------------
  "parity": {
    "relation": "non-parity",
    "of_corpus_key": null,
    "is_parity": false,
    "comparability_key": "nonparity:tanitds-lowspeed-stopgo-v1:<selection_sha256[:16]>",
    "relation_proof": { … },          // §5.3 — the verification record, or the refusal
    "ingest_gate": { … },             // guard_corpus_build's `rec` VERBATIM
    "naming_firewall": {"out_dir_corpus_key": null, "checked": true}
  },

  "licence": {
    "scope": ["owned-safe"], "require_commercial_ok": true, "forbid_share_alike": false,
    "verified_rows": 1234,            // verify_license_scope return value
    "by_class": {"owned-safe": 1234},
    "by_source": {"comma2k19": 1234},
    "tiers": {"ship": 1234},
    "release": "private", "notice_file": "NOTICE"
  },

  "selection_conditioning": {         // §3.5
    "families_applied": ["quality","dynamics","situation"],
    "privileged_channels": ["ego_poses"],
    "disclosure": "Situation and dynamics predicates read ego poses. Any number measured on this set is conditional on that selection."
  },

  "evidence_class": "MEASURED (live resolution over the declared sources)"
}
```

⭐ **`per_predicate_attrition` is not a nicety.** `DATA_STRATEGY.md:834-847` states the
rule this field exists to satisfy: *"before quoting a count, read the instrument's
inclusion rule"* — a census *"is a claim about the FILTER until proven otherwise"*. A
dataset that reports its size without reporting what removed the rest is exactly that
unqualified claim.

⭐ **`ingest_gate` is embedded verbatim** because `parity.py:2298-2300` requires it:
*"`record` is meant to be written into the build's own manifest. A filtered build whose
manifest does not say what was filtered reports a clip count that no longer matches the
selection it names."*

### 4.5 How the contract is VERIFIED — a test, not a promise

`stack/tests/test_p8_determinism.py` (must exist before the first dataset is built):

| # | Test | Asserts |
|---|---|---|
| T1 | `test_same_spec_same_selection_across_processes` | two **separate subprocesses** resolve the same spec → identical `selection_sha256` and `content_sha256`. Same-process reuse would pass on a cached result and prove nothing |
| T2 | `test_selection_is_source_enumeration_order_independent` | shuffle the input enumeration order → digest unchanged |
| T3 | ⭐ `test_a_changed_predicate_changes_the_digest` | **the CAN-FAIL arm.** Perturb one threshold by one ULP-ish step that changes ≥1 member → digest MUST change. A determinism test with no failing arm proves only that the function is constant (programme §6.2) |
| T4 | `test_volatile_field_list_is_pinned` | the excluded set equals the literal list in §4.2; adding a field fails the test |
| T5 | `test_seed_moves_the_split_but_not_the_selection` | separates the two determinism axes |
| T6 | ⭐ `test_skip_hash_is_computed_not_copied` | mutate the skipped set → emitted `skip_hash` changes. **Closes hole #1 of §2.2 for P8's own output** |
| T7 | `test_skip_hash_uses_the_comma_joined_serialisation` | pin the canonical form against `compute_skipset.py:3-4`; a newline-join must FAIL |
| T8 | `test_provenance_manifest_schema_is_complete` | every key in §4.4 present; an absent `parity` or `licence` block refuses |
| T9 | `test_attrition_sums_to_the_input_count` | `n_in − Σdropped == n_out` per family, and the chain composes |
| T10 | `test_a_spec_that_lints_clean_but_resolves_empty_refuses` | an empty selection is a **refusal**, never a zero-row dataset |
| T11 | `test_hash_pinned_split_is_stable_under_pool_growth` | add 20 % more units → **no existing unit changes side** |
| T12 | `test_two_specs_differing_only_in_volatile_fields_share_a_digest` | the converse of T4 |

⛔ **`tanitds verify <dir>` recomputes every digest from the artifacts on disk and exits
non-zero on any mismatch.** It is the runtime form of T1–T9 and is a required step in
`/TanitAD_DesignDataSet`.

---

## 5. THE PARITY GUARD

*This is the section the product exists for.*

### 5.1 The threat model, stated precisely

A dataset configurator can damage parity in exactly four ways. Naming them is what lets
the guard be checked rather than believed.

| # | Threat | Why the existing machinery does not already stop it |
|---|---|---|
| **T-A** | **Identity theft.** P8 writes into a directory whose path contains `e438721ae894`; `corpus_key_of` (`parity.py:238-241`) then resolves it *as* the parity corpus | Corpus identity is a **path substring** (§2.2 hole #3). P8 is the first tool that lets a user *choose* that path |
| **T-B** | **Silent re-selection.** A filtered subset is trained on and its result compared to a parity arm | Nothing today attaches a comparability identity to a *result*; the comparison happens in a report, downstream of every guard |
| **T-C** | **Contamination.** A P8 set built for eval contains parity-train clips, or a train set swallows the deployed val | `guard_corpus_build` exists (`parity.py:2278`) but only fires at the 6 wired doors. A new door — and P8 *is* a new door — has no gate until it is wired |
| **T-D** | **False provenance.** P8 emits `skip_hash: f09e44db` by copying the constant, as `parity.py:364` and `corpus_census.py:140` both already do | The pattern is *established practice in this repo*. Inheriting it would be the default |

### 5.2 Rule G1 — the naming firewall (closes T-A)

**Before resolving a single member, P8 calls `parity.corpus_key_of(spec.output.root)`.**
If it returns a registered key, P8 **refuses**, unless `parity.relation` is `identity` or
`geometry-sibling` *and* the §5.3 proof has passed.

This is `parity.assert_not_parity` (`parity.py:1789-1801`) inverted. That function is the
**reader-side** firewall — it stops a side model from *reading* the parity corpus. **No
writer-side equivalent exists**, and P8 is a writer. The refusal text names the resolved
path, the key it matched, and the two ways out (rename the output, or declare and prove a
relation). It prints **no clip ids** — every parity refusal in this repo prints counts
only (`test_build_parity_guard.py:434`, `parity.py:2201`), and P8 keeps that.

### 5.3 Rule G2 — four relations, each verified, default `non-parity`

Every spec declares exactly one `parity.relation`. **The declaration is not trusted; it is
proved, and the proof is the generation** — the doctrine at
`make_parity_clip_digests.py:125-127`: *"The generation IS the proof: a digest set minted
from anything but the registered clip split would silently authorise the wrong
exclusions."*

| relation | meaning | verification (all pre-existing) | `is_parity` |
|---|---|---|---|
| `identity` | the selection resolves to exactly the registered set of `of_corpus_key` | `parity.check_uids(..., mode="strict")` must reproduce `episode_uid_sha256`; for a v2 corpus, `parity.verify_v2_membership` (`parity.py:1385`) must reproduce `clip_id_sha256_sorted` with `membership_identical: true` | **true** |
| `geometry-sibling` | identical membership, different canonical frame or container | `parity.register_v2_geometry_sibling` (`parity.py:1577`) — the committed precedent is `physicalai-train-e438721ae894-w120-256x640cyl`, whose `provenance.membership_proof` records `clips_expected 2400 / clips_built 2400 / missing_count 0 / extra_count 0 / membership_identical true` | **true** |
| `recorded-view` | a **strict, enumerated, digested SUBSET** of one registered corpus | the `build_deployment` pattern (`make_parity_clip_digests.py:155-229`): a subset **cannot** reproduce the whole-corpus digest, so the proof is a **second-source cross-check** — the committed instance is `deployed_val40_clip_digests.json` with `cross_check_episodes: 40`, `is_full_corpus: false` | **false** |
| `non-parity` | anything else: re-selection, filtering, curation, mixing, new corpora, augmentation | `parity.guard_corpus_build` must pass at the declared role | **false** |

⭐ **`recorded-view` is not new machinery — it is the existing 40-episode val deployment,
generalised.** `parity_manifest.json` already distinguishes `known_deployments`
(admissible: 600, 40) from `deployments_seen_but_NOT_admissible` (the 12-episode pod1
partial, with a `why_not` that reads *"a decision-grade eval against it silently reports a
different benchmark. It already blocked a decision-grade run once."*). Today that
admissibility judgement is **hand-maintained prose enforced only at the count level**
(`parity.py:692-698` compares `n` against `ok_counts`). **P8 promotes it to a digest-level,
machine-written registry.**

⛔ **The default is `non-parity`, and an absent `relation` field is `non-parity`, not an
error and not a pass.** A guard whose safe state requires the user to type something is
the C113 defect again.

### 5.4 Rule G3 — `comparability_key`, the structural bar (closes T-B)

Declaring a set NON-PARITY is worthless unless something downstream *cannot* compare it.
The mechanism:

> **Every P8 dataset carries a `comparability_key`. Two results may be compared only if
> their `comparability_key`s are byte-equal. P7 refuses a paired bootstrap across
> differing keys; the refusal names both keys and both dataset ids.**

| relation | `comparability_key` |
|---|---|
| `identity`, `geometry-sibling` | the registered corpus key, e.g. `physicalai-train-e438721ae894` |
| `recorded-view` | `view:<of_corpus_key>:<selection_sha256[:16]>` |
| `non-parity` | `nonparity:<dataset_id>:<selection_sha256[:16]>` |

Consequences, all intended:

- A curated subset of parity train is **never** comparable to a parity arm — its key
  differs by construction. It **is** comparable to another arm on the *same* view, because
  those keys match.
- A geometry sibling **is** comparable to its parent — which is already how
  `-w120-256x640cyl` is treated, now machine-enforced instead of remembered.
- ⭐ **The key is a function of the selection digest, so it cannot be spoofed by renaming.**
  This is the property `corpus_key_of`'s substring match lacks (§2.2 hole #3).
- ⚠️ **`selection_conditioning.privileged_channels` (§3.5) is appended to the key when
  non-empty**, so a set selected on ego-derived predicates cannot be silently compared
  against one that was not.

⚠️ **P8 cannot enforce this alone** — the comparison happens in P7. §9.4 states the
contract, and `P8-08` is the escalation. **Until P7 implements the refusal, the bar is a
convention, and this SPEC says so rather than claiming enforcement it does not have.**

### 5.5 Rule G4 — the ingest gate runs first, and it runs early (closes T-C)

Before resolving a single member, and **before the first byte is read or written**, P8
calls:

```python
parity.require_ingest_gate("P8.resolve")                       # parity.py:2430
kept, rec = parity.guard_corpus_build(
    clip_ids, label=f"P8 {spec.dataset_id}",
    role=spec.parity.corpus_role,                              # parity.py:2268-2275
    mode=spec.parity.overlap_mode,
    sanctioned_audit=spec.parity.sanctioned_audit)
```

`rec` goes into `PROVENANCE.json` verbatim (§4.4). Why *early* rather than merely
*present*: `parity.py:2289-2290` — *"C112's own launch-path defect died after paying for a
536 MB download; a gate that runs late is a gate that costs money to trip."*

**Pinned by test, following the two patterns that already work:**
- `test_p8_gate_refuses_before_any_write` — mirrors `test_build_parity_guard.py:479`,
  which asserts both that `v2_compressed.build` raises **and** `not os.path.exists(a.out)`.
- `test_p8_gate_precedes_the_expensive_step` — mirrors `test_build_parity_guard.py:529`,
  an **AST line-number** comparison `min(gate_lines) < min(spend_lines)`.

⭐ **P8 must appear in the derived door population.** `test_build_parity_guard.py:153`
`derive_corpus_writers` walks the AST for corpus-artifact writes and requires every writer
to be `gated` or explicitly classified. **P8's builder must be *derived* into that
population, not added to `KNOWN_DOORS` by hand** — the whole reason that test derives
instead of listing is that "the builder is one place" was false. Backlog `P8-03`.

### 5.6 Rule G5 — compute what you stamp (closes T-D)

- P8 **computes** its `skip_hash` from its own skipped set, comma-joined
  (`compute_skipset.py:3-4`), pinned by tests T6 + T7 (§4.5).
- P8 **never** copies `parity.PARITY_SKIP_HASH` into its own output. If a P8 dataset is a
  proved `identity` of the parity corpus, it may *quote* the constant — and then it must
  also *reproduce* it, or the relation proof failed.
- ⛔ **P8 may not claim a val-side selection is pinned**, because the parity val split
  carries `episode_uid_sha256: null` (§2.2 hole #2). A `recorded-view` on the val split
  emits `content_check: "COUNT-ONLY — the parent corpus carries no uid digest"` and the
  data card says so.

### 5.7 Rule G6 — the guard must be shown able to fail

Per `TANITAD_PROGRAMME.md:161-162`: *"A guard must be shown ABLE TO FAIL (the
deliberate-regression arm) before its PASS means anything."*

P8 ships a **neuter matrix** in the shape of
`…/2026-08-18-build-parity-guard/raw/neuter_matrix.txt` (8 cases, all banked RED). P8's
matrix has **7 cases**, and the deliverable is the banked RED result, not the intention:

| # | Neuter | Must go RED |
|---|---|---|
| N1 | G1 naming firewall → no-op | the T-A test |
| N2 | G2 `identity` verification → always-true | the identity test |
| N3 | G2 `geometry-sibling` verification → always-true | the sibling test |
| N4 | G2 `recorded-view` cross-check → skipped | the subset test |
| N5 | G3 `comparability_key` → constant | the cross-key comparison test |
| N6 | G4 `guard_corpus_build` call → removed | the contamination + gate-order tests |
| N7 | G5 `skip_hash` → copy the constant | T6 + T7 |

⚠️ **Restore by byte-compare, not by `try/finally`.** The banked matrix header records
that an earlier attempt *"was killed by a 2-minute tool timeout MID-CASE and left
`aug120_pipeline.py` neutered on disk — a `try/finally` is no protection against SIGKILL.
That is why the byte-compare exists."* P8's runner md5s every touched file afterwards.

### 5.8 The oracles P8 depends on, and what happens when they are missing

`require_ingest_gate` refuses if either oracle is empty (`parity.py:2442-2445`), and
`load_clip_digests` refuses a self-inconsistent digest file (`parity.py:1928-1941`). MEASURED
oracle sizes: `parity_train_clip_digests.json` **2400 clips**,
`deployed_val40_clip_digests.json` **40 clips**, intersection **0**
(`test_build_parity_guard.py:400`, `:458`).

⛔ **A missing oracle is a REFUSAL, never a skip.** `test_build_parity_guard.py:38-39`
states why: *"A skipped leak test is the absent check that produced C112, wearing a green
suite."* P8 inherits that: no oracle, no build.

---

## 6. CLI surface

**One entry point, subcommands** — `stack/scripts/tanitds.py`.

⚠️ **Name collision, flagged rather than assumed away.** `stack/scripts/p8_bev_reel.py`
and `stack/scripts/train_p8_occupancy.py` already use `p8` as a *probe* number, alongside
`probe_saliency_p9.py`. A `p8_*.py` CLI prefix would collide with an established namespace.
`tanitds` is proposed; §10 Q2 asks the PI to confirm, and it needs a `VOCABULARY.md` entry
per `TANITAD_PROGRAMME.md §5`.

```
tanitds spec new     --template <name> --out <spec.tanitds.json>
tanitds spec lint    <spec>                        # schema, registry, licence, role — no I/O
tanitds spec diff    <specA> <specB>               # field diff + which digests move

tanitds parity-check <spec>                        # relation classification ONLY. no build, no writes.
                                                   # prints relation, proof verdict, comparability_key,
                                                   # ingest-gate record. Exit 0 pass / 3 refuse.

tanitds resolve      <spec> [--out-json SELECTION.json] [--limit N]
                                                   # resolve members, run G1+G4, emit the digests.
                                                   # WRITES NOTHING but the selection file.

tanitds build        <spec> --out <dir> [--yes] [--format lake|epcache|v2|selection-only]
                                                   # resolve -> guards -> hand to the P2 builder ->
                                                   # emit PROVENANCE.json / DATA_CARD.md / SELECTION.json

tanitds verify       <dir>                         # recompute EVERY digest from the artifacts on disk.
                                                   # exit non-zero on any mismatch. (§4.5)

tanitds card         <dir> [--out DATA_CARD.md] [--html]
tanitds register     <dir>                         # write the dataset into the P8 registry (+ P5 when P8-04 lands)
tanitds ls           [--registry <path>] [--relation …] [--licence …]
```

**Signatures that are load-bearing:**

| Command | Guarantee |
|---|---|
| `spec lint` | **pure** — no filesystem reads beyond the spec, no network. Safe in CI |
| `parity-check` | **read-only** and cheap. This is what `/TanitAD_DesignDataSet` runs before proposing a build, and what a reviewer runs on a PR |
| `resolve` | runs G1 + G4 and **refuses before any output directory is created** |
| `build` | requires `--yes` when `relation != "identity"`, so a re-selection is always an explicit act |
| `verify` | the only command whose exit code is a *claim about the artifacts*; every other command claims about the spec |

**Exit codes**, following `corpus_census.py:672-678` and `compute_skipset.py:91`:
`0` OK · `1` warnings (attrition above a declared threshold; a `recorded-view` on a
count-only parent) · `2` bad usage / lint failure · `3` **refusal** (parity, licence, or
gate). A refusal is distinguishable from a crash, because a supervisor must tell them apart.

⛔ **No `--force`.** Following `train_flagship_v4.py:1847-1851`, the escape hatch is a
**reason**, not a boolean: `--parity-off-reason "<why>"`, printed, stamped into
`PROVENANCE.json`, and it sets `decision_grade: false`. Same contract as
`--sanctioned-audit` (`parity.py:2313-2319`).

---

## 7. The clickable UI — realistic scope

**MEASURED:** there is **no** Gradio or Streamlit anywhere in this repo (two probes: `grep
-rl gradio --include=*.py` and a second over `*.md`/`*.txt` — both empty). There **is** a
working FastAPI + vanilla-JS SPA pattern: `stack/scripts/scena_app.py:116` `build_app(...)`
with routes `/`, `/api/meta`, `/api/scenarios`, `/api/scenario/{sid}`, `/api/search`,
`/api/reindex`, serving `stack/tanitad/scena/static/{index.html,app.js,style.css}` — three
files, **no build step, no CDN**.
**INHERITED (not re-verified here):** `TANITAD_PROGRAMME.md:29-30` records a private Space
`Sayood/TanitAD` — Gradio, zero-a10g ZeroGPU, RUNNING.

### 7.1 ⭐ The one design decision that makes the UI safe

> **The UI writes CONFIGS. It never writes DATASETS.**

The clickable surface is a **spec editor with a live preview**. It resolves counts and
distributions; it emits a `.tanitds.json`; the build stays in the CLI. ⇒ **the clickable
path cannot bypass a single guard**, because it never reaches the code that would need
one. Any design where a button starts a build re-opens every threat in §5.1 behind a
surface nobody reviews.

### 7.2 Three tiers, in shipping order

| Tier | What | Deps | Compute |
|---|---|---|---|
| **T0 — static card** | `tanitds card --html` emits a self-contained HTML page per dataset: parity verdict banner, the four digests, licence/tier breakdown, per-predicate attrition waterfall, stratum histogram, `selection_conditioning` disclosure. Inline CSS/SVG, no JS required | **none** | none |
| **T1 — local spec editor** | Reuse the `scena_app.py` FastAPI + static-SPA pattern against the Parquet catalog. Predicate controls; **live resolved count + stratum distribution + attrition**, recomputed on change; a *Parity* panel calling `parity-check`; **Download spec**. Read-only over the catalog | `fastapi`, `pyarrow` (already used) | local, unmetered |
| **T2 — hosted read-only** | The same app on the existing HF Space, over a *published* catalog only | as T1 | ⛔ **gated on the HF quota rule** (`TANITAD_PROGRAMME.md:22-28`): verify remaining quota BEFORE launching; if unknown, DO NOT START |

### 7.3 Out of scope, explicitly

- Any button that starts a build, an ingest, or a push.
- Multi-user accounts, or write access to the dataset registry from the browser.
- ⛔ **Displaying gated clip ids.** The confidentiality rule is absolute
  (`parity.py:1829-1834`; `test_build_parity_guard.py:434` asserts no refusal ever prints
  one). The UI shows **counts and digests only** for any `gated-confidential` source.
- Rendering frames from a `gated-confidential` corpus in a hosted context.

---

## 8. Success criteria — measurable

### 8.1 Gate A — the product works (binary, cheap, no GPU)

| # | Criterion | Measured by |
|---|---|---|
| A1 | A spec resolved in two separate processes yields identical `selection_sha256` and `content_sha256` | T1 |
| A2 | The determinism suite has a **failing arm** that goes RED on a perturbed predicate | T3 |
| A3 | Every output carries `PROVENANCE.json` with all §4.4 keys; `tanitds verify` exits 0 | T8, `verify` |
| A4 | `skip_hash` is computed; copying the constant fails the suite | T6, T7 |
| A5 | All **7** neuter cases banked RED, with an md5 byte-compare showing every file restored | §5.7 |
| A6 | P8's builder appears in `derive_corpus_writers`' **derived** population as `gated` | `test_build_parity_guard.py:153,261` |
| A7 | A spec whose output path contains a parity key is refused, printing no clip ids | G1 test |
| A8 | A `non-parity` dataset's `comparability_key` differs from every registered parity key | G3 test |
| A9 | An eval-role spec containing parity-train clips is refused **before any write** | §5.5 test |
| A10 | An empty selection refuses rather than emitting a zero-row dataset | T10 |

### 8.2 Gate B — ⭐ data efficiency, MEASURED not asserted

`TANITAD_PROGRAMME.md §6.1` requires pre-registered success criteria with **both outcomes
committed in advance**. This is that pre-registration.

---

#### **PREREG-P8-CUR1 — does curation beat random at equal size?**

**Claim under test (H-P8-1).** A P8-*curated* set of N episodes beats a *random* set of N
episodes drawn from the **same pool**, when trained on a **fixed model** with a fixed
budget, on the four metric families.

⚠️ **Status: `curate.strategy` defaults to `"none"` until this reports.** MEASURED: a
repo-wide search for a curated-vs-random comparison (`curated.*random`, `random.*curated`,
`vs.random`, `baseline.*random` over `stack/**/*.py` and `stack/**/*.md`) returns **zero
matches**. `lake/curation.py` computes weights and stores them; **nothing in this repo
consumes them for a comparative claim**, and every curation test asserts only that the
weights are computed correctly (`test_lake_curation.py`), never that they help. **The
programme's data-curation value proposition is currently unmeasured.**

**Design.**

| | |
|---|---|
| **Pool** | a declared `non-parity` pool (it must be — curation re-selects). The lake `owned-safe` view, so no gated data and no parity entanglement |
| **Arm C (curated)** | `curate.strategy = "inverse_frequency"` + `weakness_boost`, `target_episodes = N` |
| **Arm R (random)** | identical spec, `curate.strategy = "none"`, `target_episodes = N`, **≥3 seeds**, so one lucky draw cannot decide it |
| **Fixed model** | one architecture, one step budget, identical everything else, named in the prereg before launch (§10 Q5) |
| **Eval set** | ONE fixed held-out set, **identical for both arms**, disjoint from both pools — checked with `guard_corpus_build(role="eval")`, not assumed |
| **Statistic** | **paired episode-cluster bootstrap** (`taniteval/ci.py`), per family, ⛔ **never pooled**. `overlapping_holdout_se` is inadmissible |
| **Reported** | all **four families** — longitudinal, lateral, tactical, strategic — plus ADE. ⛔ An ADE-only table is not a result |
| **Printed** | n, d, the number of random seeds, and the spread across them |

**Primary metrics** (declared in advance, so the outcome cannot be chosen after the fact):
**longitudinal** (target-speed accuracy, headway/TTC) and **tactical** (manoeuvre-decision
quality). Rationale, from `CLAUDE.md`: 88.7 % of the oracle gap is longitudinal, and the
lat+lon-mixing 5-way manoeuvre softmax is the single largest known defect. Lateral and
strategic are **secondary guardrails** — a win bought by regressing them is not a win.

**Controls, required before the result may be quoted** (`TANITAD_PROGRAMME.md §6.2`):
- **C1 constant-only control** — a "curation" assigning uniform weights must land on Arm R's
  number. If it does not, the harness is confounded and nothing may be concluded.
- **C2 raw-input floor** — the no-information value, printed.
- **C3 deliberate-regression arm** — an *anti*-curated set (up-sample the majority stratum)
  must score **worse**. ⛔ **If it does not, the instrument cannot rule and must say so
  loudly** rather than reporting a null.

**Pre-flight that costs zero GPU** — run first, and abort if it fails: verify Arms C and R
actually differ in the intended way (stratum entropy; share of weakness strata; speed
histogram). **If the two selections do not measurably differ, the experiment cannot rule
and must not be launched.**

##### Both outcomes, committed in advance

| Verdict | Declared condition | What we do, committed now |
|---|---|---|
| ⭐ **CURATION HELPS** | Arm C beats the **median** random seed on **both** primaries with a paired 95 % CI excluding 0, **and** neither lateral nor strategic regresses beyond a margin declared before launch | `curate.strategy` gains a non-`none` default. `MODEL_REGISTRY.md` records the effect size, CI, n, seeds. `P8-09` (curation tuning) is unblocked |
| ⛔ **CURATION DOES NOT HELP** | the advantage CI **includes 0** on both primaries, **or** Arm C falls inside the spread of the random seeds | **P8 ships as a reproducibility + provenance + licence tool only.** `curate.strategy` stays `"none"` **permanently by default**. `lake/curation.py` is stamped **NOT-SHOWN-TO-HELP** in this SPEC, in `GOALS_AND_CLAIMS.md` and in the registry. The DataFlyWheel's *"best results with least data effort"* goal is re-scoped to **automation and provenance**, and we say so in the paper |
| ⛔ **CURATION HARMS** | Arm R beats Arm C with a CI excluding 0 on **any** family | `RETRACTION_LOG.md` entry with the root-cause class. `curate.*` is removed from the schema, not merely defaulted off |
| ⚠️ **CANNOT RULE** | C3 fails, or the pre-flight shows the arms do not differ | **Report exactly this.** Do not report a null as evidence of no effect. Fix the instrument, re-register, re-run |

⭐ **We commit to reporting the result at equal prominence whichever way it lands**, in
`products/P8-datasetcreator/RESULT.md`, in `GOALS_AND_CLAIMS.md` (H-P8-1), and in the
programme report. A negative result here is a **durable asset**: it retires an unmeasured
assumption that currently sits under a whole FlyWheel's stated goal.

⚠️ **Compute constraint, binding.** Local compute (Thor, RTX 4060) is unmetered and is the
default; the HF Pro quota is a **hard ceiling** and no metered job starts without a
verified remaining quota (`TANITAD_PROGRAMME.md:22-28`). The fixed model and step budget in
§10 Q5 must be chosen to fit local compute, or the experiment does not get to exist.

### 8.3 Gate C — adoption (the "finish before you start" criterion)

| # | Criterion |
|---|---|
| C1 | ≥1 real training run launched from a `DatasetSpec`, with `spec_sha256` + `selection_sha256` + `comparability_key` in its `config.json` and its `MODEL_REGISTRY.md` row |
| C2 | The 40-episode deployed val is expressible as a `recorded-view` spec whose `selection_sha256` reproduces the committed `deployed_val40_clip_digests.json` `digest_of_digests` — **a round-trip against a committed artifact, not a synthetic fixture** |
| C3 | `/TanitAD_DesignDataSet` runs end-to-end and its parity/provenance step is `tanitds parity-check` + `tanitds verify` |
| C4 | The dataset registry lists every P8 dataset with its relation and comparability key, and `tanitds ls` renders it |

---

## 9. Interfaces

### 9.1 ← P5 TanitScena (consumes) — **CONTRACT TO BE BUILT**

Today P5 indexes 14 markdown scenarios, not clips (§1.3). The contract P8 needs:

```
scena.resolve_tags(tags: list[str]) -> list[ClipRef]     # ClipRef = {source, clip_id | episode_uid}
```

Until it exists, `select.scenario` is **lint-refused** and families 2–6 (§3.4) carry the
product. ⇒ escalation `P8-04`.
⛔ **When it lands, `ClipRef` must carry `source`, so a tag match in one corpus can never
be silently applied to another** — that is the C112 shape (an overlap assumed from
provenance rather than computed from ids).

### 9.2 ← P2 Data pipelines (consumes)

P8 resolves and declares; **P2 executes.** The handoff is `SELECTION.json` + the spec's
`output` block, passed to the appropriate builder:

| `output.format` | Builder | Gate already present |
|---|---|---|
| `epcache` | `epcache.build_episodes_cached(..., corpus_role=, sanctioned_audit=)` (`epcache.py:80-83`) | ✅ `epcache.py:114-117` |
| `v2` | `scripts/v2_compressed.py` | ✅ `:416,418` |
| `lake` | `lake/ingest.ingest_source(...)` (`ingest.py:104`) | ✅ licence at `schema.py:296-307` |
| `selection-only` | none — emits the manifest and the id list, nothing else | n/a |

⭐ **`selection-only` is the format that makes P8 useful on day one**, before any builder
integration: it produces a proved, digested, parity-classified selection that an existing
pipeline can consume by path.

### 9.3 → P4 Training (hands configs)

P8 adds `--dataset-spec <file>` to the trainers. The trainer:
1. resolves the spec (or reads a committed `SELECTION.json`),
2. records `spec_sha256`, `selection_sha256`, `comparability_key`, `parity.relation` into
   its `config.json`,
3. carries them into its `MODEL_REGISTRY.md` row.

⭐ **This closes a real measured gap.** Today `train_flagship_v4.py:1800` takes
`--train-cache` with no help text and corpus identity is inferred from a path substring
(§2.7). After this, an arm's row *states* which selection it trained on and whether that
selection is comparable to any other — instead of a reader inferring it from a directory
name.

### 9.4 → P7 TanitEval (hands datasets)

- Every dataset carries `comparability_key`; every eval result inherits it.
- ⛔ **P7 refuses a paired comparison across differing keys**, naming both.
- P7 reports all four families (`CLAUDE.md`); P8's data card states **which families the
  set can support and which it cannot, per family with the reason and the n** — e.g. no
  lead agent in frame ⇒ no headway/TTC. Silently dropping a family is forbidden.
- `selection_conditioning` (§3.5) travels with the result, so a conditional number is
  never read as unconditional.

⚠️ **This is an INTEGRATION ESCALATION, not a note in a doc.** `P8-08`. The programme has
measured this failure twice: an orthogonality instrument sat unmerged for **10 days** and a
LAL-v2 implementation for **12**, both because the request lived in a README nobody
re-read.

### 9.5 ↔ P3 DataReconstruction

Reconstructed data becomes selectable once it has a `SOURCE_REGISTRY` entry with an
explicit licence class and a stable `split_unit_id`. Until then P8 lint-refuses it —
**not because it is untrusted, but because a source with no licence constant cannot be
tiered, and `tier_of` raises rather than guessing** (`filtering.py:39-41`).

---

## 10. Open questions for the PI

| # | Question | Why it blocks | Default if unanswered |
|---|---|---|---|
| **Q1** | **The licence conflict is unresolved and already escalated.** `DATA_STRATEGY.md:783-788` records that `TANITDATASET_V1_STRATEGY.md` classes PhysicalAI/Alpamayo as *"commercial-OK for internal AV dev but no-derivatives → firewalled, recipe-only"* while `ALPAMAYO2_SUPER_ANALYSIS.md` cites an **OpenMDW-1.1 derivative permission** — *"These are not obviously the same reading"* (escalated §12 row 1, `:878`). Which reading binds? | `licence.release` cannot be specified correctly; it decides whether a P8 dataset over PhysicalAI can exist outside the firewall at all | the **stricter** reading: `release: "private"`, firewalled, recipe-only |
| **Q2** | CLI name. `p8_*` collides with an existing probe namespace (`p8_bev_reel.py`, `train_p8_occupancy.py`, `probe_saliency_p9.py`). Confirm `tanitds`, and add a `VOCABULARY.md` entry | naming is a §5 governance item; renaming later costs every doc that cites a command | `tanitds` |
| **Q3** | **May a `recorded-view` over the parity TRAIN split exist at all?** It is a re-selection by definition. Proposal: allowed, permanently NON-PARITY, never comparable to parity arms. The alternative — refuse outright — is simpler and defensible | decides whether P8 can serve ablations over the parity corpus | **allowed, permanently non-parity** |
| **Q4** | The parity **val** split has no uid digest (`uid_source: "count-only-unrecorded"`). Minting it needs a host holding the 600 val clip ids. Authorised? On which machine? | until then no val-side selection can be content-pinned (§5.6); a substituted val of the right size passes today | leave unminted; emit COUNT-ONLY and say so |
| **Q5** | **PREREG-P8-CUR1**: which fixed model, which step budget, which N, how many random seeds? Must fit **local compute** — the HF Pro quota is a hard ceiling and no metered job may start without a verified quota | the experiment cannot be pre-registered without them, and §8.2 is the product's central claim | do not launch; ship Gate A and report the gap |
| **Q6** | Confirm `curate.strategy` defaults to `"none"` until PREREG-P8-CUR1 reports, and that a null result is published at equal prominence | this is the anti-false-positive commitment; it is worth an explicit yes | yes |
| **Q7** | Should `tanitds register` write into P5 (per `SKILLS_SPECS.md:45` *"register in TanitScena"*) or into a separate P8 dataset registry? P5's index is prose over 14 documents and is not shaped for this | decides where the dataset registry lives and who owns its schema | a separate `products/P8-datasetcreator/REGISTRY.json`, with a P5 bridge when `P8-04` lands |

---

## 11. What this SPEC does NOT claim

Stated so it is not over-read — the discipline `parity.py:1843-1848` models.

- **P8 does not make a re-selected dataset comparable to a parity arm.** It makes the
  incomparability *explicit, machine-readable, and hard to lose*. Nothing here weakens
  `CLAUDE.md §Invariants`.
- **P8 does not verify episode CONTENT bytes against the parity corpus.** The manifest
  pins *slots*, not bytes (`parity_manifest.json` `notes.limitations`; `parity.py:40-41`).
  P8's `content_sha256` pins the bytes **of its own output**, which is a different and
  weaker claim than "these are the same bytes the flagship trained on".
- **P8 does not close §2.2 hole #3 for the rest of the repo.** G1 stops *P8* from stealing
  an identity. `corpus_key_of`'s substring match remains what it is everywhere else.
- **P8 does not enforce §5.4 by itself.** The comparability bar lands in P7. Until `P8-08`
  is integrated, it is a convention, and this SPEC does not pretend otherwise.
- **No claim is made that curation improves anything.** §8.2 exists precisely because that
  claim is currently **unmeasured**, and the DataFlyWheel's stated goal rests on it.
