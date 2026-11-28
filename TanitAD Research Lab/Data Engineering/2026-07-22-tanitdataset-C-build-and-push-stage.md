<title>TanitDataSet-C — first real build + staged HF push (2026-07-22)</title>

# TanitDataSet — the build: design → records → staged push

**Date:** 2026-07-22 (local, Europe/Berlin). **Author:** Data Engineering agent.
**Status:** C **BUILT** (real records) and **staged for push, NOT pushed** — a public
dataset is irreversible, so the final `huggingface-cli upload` waits on Sayed.

Turns the two-tier design (`TANITDATASET_V1_STRATEGY.md` rev-4 +
`TANITDATASET_TIER_INTEGRATION_2026-07-21.md`) from **0 records** into actual sharded
records with a provenance-stamped, license-verified push set.

---

## Headline

- **TanitDataSet-C (commercial tier): 90 real records built** — all `comma2k19`
  (MIT), canonical `[T,9,256,256]` uint8 + `actions[T,2]` + `poses[T,4]`. 72 train /
  18 val, 14 tar shards, Parquet catalog. **MEASURED** (`build_report_C.json`).
- **License verification on the push set: PASSED.** All 90 records
  `owned-safe ∧ commercial_ok ∧ ¬share_alike`; 0 rows outside scope; 0 gated; 0
  refuse. sha256 re-verified over every frame blob in the staged bundle.
- **TanitDataSet-R (research superset): = C tonight (90 records, all `ship`).** No
  `nc-research` source is downloaded on this box, so R adds no records yet. The nc
  extension is pending download + adapters; PhysicalAI-AV stays firewalled/recipe-only.
- **New enforcement `refuse` class encoded + tested** (Waymax, Waymo Open): ingest
  raises exactly as `gated-confidential` does.
- **Two real bugs found and fixed in the export scaffold** (below): a Windows
  cp1252 card-write crash, and a **shard-collision that silently dropped 3 of 14
  shards** and mixed splits. Both would have corrupted any real push.

## What is NOT done (fail loud)

- **L2D produced 0 records** — no loader exists (LeRobot-v3 parquet+mp4 → 9-ch stack
  is a ~2–3 eng-day adapter) and no L2D video is on this box. It is correctly
  registered `ship`; it cannot enter the corpus until the adapter lands. **This is
  the top integration escalation** — L2D is the source that makes C complete across
  strategic + tactical + operative layers.
- **R's NC value-add (nuScenes/AV2/BDD/A2D2/KITTI360/ONCE/DRAMA/Rank2Tell) = 0
  records** — large NC downloads + adapters, none on this box, none HF-publishable.

---

## 1. `refuse` — the fourth license class (design step 1)

`nc` means "research-only, not shippable." Two sources are worse: their terms follow
the **trained weights** into the model/product, so contamination survives training and
no tier can contain them (`TANITDATASET_TIER_INTEGRATION` §2).

| source | why | encoded as |
|---|---|---|
| **Waymax** | §2.e bars training *any* foundation model | `license_class="refuse"` |
| **Waymo Open / WOD-E2E** | terms reach vehicle operation / production weights | `license_class="refuse"` |

Encoded in `stack/tanitad/lake/schema.py`: `LICENSE_CLASSES` gains `"refuse"`;
`waymo` moved nc→refuse and `waymax` added; `assemble_lake_record` **and**
`ingest_source` raise `PermissionError` on it; `PERMISSIVE_SOURCES` excludes it;
`license_guard` refuses it in any export scope; `filtering.tier_of` gives it no tier.
5 new tests, all green (`test_lake.py`).

## 2. TanitDataSet-C — the build (design step 2)

**Source of records:** the only license-clean, canonical-resolution real cache on
this box — `C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f`, **90
distinct** comma2k19 episodes at 256px (the 128px `_epcache` is a half-resolution
older build; using it would ship off-spec frames and mix resolutions — excluded).

**Pipeline (all real):** `CachedEpisodeIngestor` → `assemble_lake_record` (schema +
license stamp) → SA-segregated tar shards + Parquet catalog → two-pass dedup →
tier stamp. Driver: `stack/scripts/build_tanitdataset.py` (committable, reusable,
ready for L2D/cosmos_dd when adapters exist).

| metric | value |
|---|---|
| records built | **90** (72 train, 18 val) |
| per-source | comma2k19 (MIT) 90 |
| tier | `ship` (all) — `commercial_ok=True`, `share_alike=False` |
| shards | 14 tar (11 train + 3 val), catalog 90 rows |
| skipped / errors | 0 |
| lake root (off-Drive) | `C:/Users/Admin/tanitad-data/tanitdataset/lake` |

### The dedup finding (reported, not applied to the corpus)

Two-pass dedup collapsed 90 → **23 pHash exemplars** (67 flagged near-dup). This is
**not** duplication: the 90 have distinct ids AND distinct exact-frame hashes;
near-dup pairs sit at mid-keyframe L1 ≈ 0.10 vs 0.15 for random pairs — genuinely
different highway scenes. The 8×8 single-keyframe aHash + transitive union-find
**over-collapses homogeneous highway footage** (comma2k19 is one commute route
re-driven) into a 30-member smear. Per the design ("filtering removes what is
unusable; curation decides what we see how often"), **all 90 records ship**; the
exemplar flag is a sampling hint. Root cause + fix are escalated below.

## 3. TanitDataSet-R — the superset (design step 3)

R = C ∪ NC over one schema; the machinery is live (a view with scope
`{owned-safe, nc-research}` auto-admits nc records when ingested). **Tonight R holds
90 records, all `ship` — identical to C**, because no nc source is downloaded here.
- **May leave (ship):** 90. **May NOT leave (nc):** 0 tonight; any future nc record
  is internal-only and never enters a push.
- **PhysicalAI-AV:** `gated-confidential` → firewalled, recipe-only, structurally
  unable to become a lake record or a push member (the `PermissionError` held —
  verified: full catalog is 90/90 owned-safe, 0 gated).

## 4. The staged HF push (design step 4) — PUBLISHING ACTION, awaiting Sayed

Staged via the guarded exporter (`tanitad.lake.hf_export.export_hf`,
`require_commercial_ok=True`). The layer-3 guard passed: every row `owned-safe`,
`commercial_ok`, SA-free. **Nothing was pushed.**

- **Staged bundle:** `C:/Users/Admin/tanitad-data/tanitdataset/hf_stage_C` (~15 GB)
  — `shards/owned-safe/comma2k19/{train,val}/shard-*.tar` (14, split-preserved),
  `DATA_CARD.md`, `MANIFEST.json`, `NOTICE`.
- **Public manifest (what would go public):** 90 comma2k19 (MIT) episodes; **no**
  ZOD/share-alike, **no** nc, **no** PhysicalAI/gated. `repo_id: Sayood/TanitDataSet-C`.

### The exact push command — run only after Sayed's go, on a **private** repo first

```bash
# token from Keys.txt (WRITE to Sayood); never inline it in argv
export HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' 'Keys.txt' | head -1)"
huggingface-cli repo create Sayood/TanitDataSet-C --repo-type dataset --private -y
huggingface-cli upload-large-folder Sayood/TanitDataSet-C \
  "C:/Users/Admin/tanitad-data/tanitdataset/hf_stage_C" --repo-type dataset
# review privately, then flip visibility to public deliberately.
```

The `export_hf` push path is intentionally left unimplemented (raises on
`push=True`) — publishing stays a human, out-of-band act.

### Pre-publish checklist for Sayed (each a deliberate call)

1. **Anonymization.** comma2k19 is already public US-highway MIT footage; the design's
   blanket real-footage face/plate gate still says run a check before *re*-distribution.
   Low PII risk (forward highway), but it is a human call, not a code gate.
2. **"MIT-on-data" legal nod** (already the standing position for comma2k19).
3. **ZOD (`ship-sa`, CC-BY-SA):** stays OUT of this push by construction (share_alike
   fails `commercial_ok`); ships only as a *separate* segregated bundle if ever chosen.

## 5. Escalations (integration — do not let these sit)

1. **Build the L2D adapter (~2–3 eng-days).** Top lever: it is the only Apache-2.0
   source with map + horizon + ego-indicator. Traps recorded: **no camera intrinsics
   ship** (risk to `f_eff=266` — estimate from vanishing point / known-width object and
   run the geometry falsifier, else admit L2D for the strategic head only), and the
   sliding-window overlap **double-counts ~50%** unless de-duplicated by timestamp and
   **split on reconstructed drives, not episodes**.
2. **Dedup over-collapse on homogeneous sources.** Single-keyframe 8×8 aHash + Hamming≤6
   + transitive union-find chains distinct highway scenes into one cluster. Fix:
   multi-keyframe pHash, a diameter cap on clusters (no transitive chaining), and
   **surface comma2k19 GPS** (it has GNSS/ECEF in `global_pose`; the bare cache dropped
   it) so pass-2 protects same-road-different-time as the wanted `multi_traversal`
   signal instead of pass-1 collapsing it.
3. **Split provenance.** This cache lost route ids, so the split is episode-level;
   rebuild-from-raw preserves route ids for a strictly route-disjoint split.

## 6. Bugs fixed in the export scaffold (would have corrupted a real push)

- **Shard-name collision (data loss).** `export_hf` flattened shards to basenames;
  train and val both number from `shard-00000`, so 3 val shards **overwrote** 3 train
  shards — 11 of 14 survived, splits mixed. Fixed to mirror the partition layout
  (`shards/<class>/<source>/<split>/…`); all 14 now stage, verified 90/90 episodes.
- **cp1252 crash.** `Path.write_text` on Windows can't encode the card's `≈`/`×`;
  forced `encoding="utf-8"` on card + manifest + sidecars; NOTICE header made ASCII.

## Deliverable manifest

| artifact | where it lives | notes |
|---|---|---|
| `refuse` class + firewall | `repo:stack/tanitad/lake/schema.py`, `ingest.py`, `license_guard.py`, `filtering.py` | staged |
| export-scaffold fixes | `repo:stack/tanitad/lake/hf_export.py`, `ingest.py` | staged |
| tests (+5 refuse, shard-mirror assert) | `repo:stack/tests/test_lake.py` | staged; 73 lake tests green |
| build driver | `repo:stack/scripts/build_tanitdataset.py` | staged; reusable |
| this report | `repo:TanitAD Research Hub/Data Engineering/2026-07-22-tanitdataset-C-build-and-push-stage.md` | staged |
| provenance (card, manifests, build report, NOTICE) | `repo:TanitAD Research Hub/Data Engineering/tanitdataset-build-2026-07-22/` | staged (KB-sized) |
| **built lake (14 shards + catalog)** | `local(off-Drive):C:/Users/Admin/tanitad-data/tanitdataset/lake` | **single location** — data tier, not committed |
| **staged push bundle (~15 GB)** | `local(off-Drive):C:/Users/Admin/tanitad-data/tanitdataset/hf_stage_C` | **single location** — destined for HF |

**Not staged (unrelated pre-existing):** `test_flagship_v4.py::test_seam_clamp…` fails
only in full-suite order (passes in isolation) — another agent's WIP v4 model
(`flagship_v4.py`/`test_flagship_v4.py` are staged `A`/`AM`, the S2 stream), untouched
by this work. All 73 lake tests pass.
