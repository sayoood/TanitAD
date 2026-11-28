---
license: mit
task_categories:
- robotics
tags:
- autonomous-driving
- world-model
- tanitad
- ego-driving
- camera
- webdataset
size_categories:
- n<1K
extra_gated_prompt: >-
  TanitDataSet-C is the commercially-clean tier of TanitDataSet. Its contents are
  redistributable under their upstream licenses (today: 100% comma2k19, MIT).
  Access is request-gated so the maintainer can see who is using it and notify
  consumers of corrections; the gate is an access log, NOT an additional license
  restriction — your rights are the upstream ones.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
---

# TanitDataSet-C — the commercially-clean tier

**Seed release · 90 episodes · 15.93 GB · 14 WebDataset shards**

TanitDataSet-C is the **commercially-clean, redistributable** tier of TanitDataSet,
the camera-first autonomous-driving corpus behind the [TanitAD](https://huggingface.co/Sayood)
sub-300M hierarchical latent world model. Every record is `owned-safe` **and**
`commercial_ok`: a permissive upstream license, **no** share-alike, **no** gated,
non-commercial, or `refuse`-class source.

The tier is a **per-record stamp derived structurally** from a per-source license
CONSTANT (`SOURCE_REGISTRY`), never inferred from prose, and a hard export guard
refuses egress if a single row falls outside that scope.

> **Read the [Honest limits](#honest-limits--read-this-before-you-plan-around-it)
> section before planning around this dataset.** This is a *seed-scale* release
> (90 episodes, one source, one road type), not a training corpus.

---

## Contents

| | |
|---|---|
| **Episodes** | **90** (train 72 · val 18) |
| **Sources** | `comma2k19` (MIT) × 90 — **100 %** |
| **License classes present** | `owned-safe` × 90 — no `nc-research`, no `gated-confidential`, no `refuse` |
| **Share-alike rows** | 0 |
| **Shards** | 14 tar (11 train + 3 val), ~1.24 GB each |
| **Total size** | 15.93 GB |
| **Frame format** | `uint8 [T, 9, 256, 256]` — 100 % of records |
| **Catalog** | Hive-partitioned Parquet, 90 rows |

### Sources & licenses

| source | license | class | `commercial_ok` | `share_alike` | episodes |
|---|---|---|---|---|---|
| [`comma2k19`](https://github.com/commaai/comma2k19) | MIT | `owned-safe` | ✅ | ❌ | 90 |

The **C tier admits** any permissive source (MIT / Apache-2.0 / CC-BY-4.0 /
OpenMDW-1.1). Today exactly one of them is built — see *Honest limits*.

---

## Record schema — the world-model contract

Each episode is the byte-identical contract every TanitAD adapter emits:

- **`frames`** — `uint8 [T, 9, 256, 256]` — a 3-frame RGB stack (9 = 3×RGB),
  canonicalized to `f_eff ≈ 266 px` (the TanitAD D-016 geometry canon).
- **`actions`** — `f32 [T, 2]` — `(steer, accel)`, the action applied between
  *t* and *t+1*.
- **`poses`** — `f32 [T, 4]` — `(x, y, yaw, v)` ego trajectory.
- **per-episode metadata** — `source`, `license_class`, `license_name`,
  `commercial_ok`, `share_alike`, `split`, `sha256` of the frame blob,
  `build_params_hash`, native intrinsics, modality flags.

### Shard layout

```
shards/<license_class>/<source>/<split>/shard-XXXXX.tar
└── shards/owned-safe/comma2k19/train/shard-00000.tar … shard-00010.tar   (72 eps)
└── shards/owned-safe/comma2k19/val/shard-00000.tar   … shard-00002.tar   (18 eps)
```

Partitioning by `license_class` is **layer 1 of the license firewall** — a
share-alike source would live under a segregated `sharealike/` prefix and could
never share a tar with non-SA data. There is none in this release.

Each tar holds three members per episode (WebDataset convention):

```
{episode_id}.frames.npy    uint8 [T, 9, 256, 256]   the canonical blob
{episode_id}.motion.npz    actions / poses / timestamps
{episode_id}.meta.json     the full catalog row + provenance
```

### Loading

A shard is a plain tar — no `webdataset` package required.

```python
import io, json, tarfile, hashlib, numpy as np

with tarfile.open("shards/owned-safe/comma2k19/val/shard-00000.tar") as tf:
    blobs, metas = {}, {}
    for ti in tf:
        key, _, ext = ti.name.partition(".")
        data = tf.extractfile(ti).read()
        if ext == "frames.npy": blobs[key] = data
        elif ext == "meta.json": metas[key] = json.loads(data)

    for key, meta in metas.items():
        frames = np.load(io.BytesIO(blobs[key]), allow_pickle=False)  # [T,9,256,256]
        assert hashlib.sha256(frames.tobytes()).hexdigest() == meta["sha256"]
```

The `catalog/` Parquet index carries one row per episode (everything except the
frame blob) Hive-partitioned by `license_class / source / split`, so you can plan
a subset with a predicate before touching a single byte of video:

```python
import pyarrow.dataset as pads
cat = pads.dataset("catalog", partitioning="hive")
rows = cat.to_table(filter=(pads.field("split") == "val")).to_pylist()
```

> **Note:** no `configs:` auto-loader block is declared. `datasets`'
> WebDataset builder has no decoder for the `.npz` motion member, so an
> auto-config would silently drop actions and poses. Use the snippet above.

---

## Provenance & verification

Every episode carries a **`sha256` of its exact frame bytes** and a
`build_params_hash`, so a consumer can verify any shard member **without
rebuilding it** — a rotted shard fails loudly instead of training on garbage.

Verified on **2026-07-25** immediately before this release, over the actual
payload bytes (not the metadata claim):

| check | result |
|---|---|
| shards present | **14 / 14** |
| episodes in payload | **90** (train 72 · val 18) |
| `sha256` re-verified over `frames.npy` bytes | **90 / 90 PASS**, 0 fail |
| catalog ↔ payload episode-id bijection | ✅ exact |
| catalog ↔ payload `sha256` agreement | ✅ exact |
| frame shape / dtype uniformity | 90 / 90 `[T,9,256,256]` `uint8` |
| distinct source corpora in payload | `{comma2k19: 90}` — **only** |
| distinct license classes in payload | `{owned-safe: 90}` — **only** |
| duplicate episode ids across shards | **0** |
| train/val episode-id overlap | **0** |
| shard-path ↔ metadata split mismatches | **0** |
| gated / `refuse` / NC / share-alike rows | **0 / 0 / 0 / 0** |

Machine-readable: **`LICENSE_VERIFICATION.json`** (both legs — the repo's own
`license_guard` and the independent payload audit), **`build_report_C.json`**,
**`MANIFEST.json`**, **`BUILD_MANIFEST.json`**, **`NOTICE`**.

---

## Honest limits — read this before you plan around it

This is a **seed release**. We would rather publish the gaps than let the size of
the repo imply a corpus that does not exist.

1. **90 episodes is seed-scale, not training-scale.** This is a working proof of
   the schema, the license firewall and the shard/catalog contract — with real
   records attached. It is **not** enough data to train a driving world model.
   TanitAD's own flagship trains on a different, larger, internal corpus.

2. **One source, one road type.** All 90 episodes are comma2k19: US highway,
   forward camera, largely free-flow. There is **no** urban, no intersection, no
   VRU-dense, no night/adverse-weather coverage in this release, and no surround
   camera, LiDAR, map or route annotation.

3. **L2D contributed 0 records — no adapter exists yet.** [L2D](https://huggingface.co/datasets/yaak-ai/L2D)
   (Apache-2.0) is the source that would make this tier complete across the
   strategic (map / speed limit / route), tactical (CAN turn-indicator) and
   operative (ego trajectory) layers. It is correctly registered as shippable, but
   the LeRobot-v3 `parquet+mp4` → 9-channel-stack adapter is ~2–3 engineering days
   of work that has **not** been done. Until it lands, L2D cannot enter the corpus.
   Two known traps are already recorded for whoever builds it: **L2D ships no
   camera intrinsics** (a risk to the `f_eff ≈ 266` canon), and its sliding-window
   episodes **double-count ~50 %** unless de-duplicated by timestamp and split on
   reconstructed drives rather than episodes.

4. **PhysicalAI-AV is deliberately excluded and always will be.** TanitAD's main
   internal training corpus is NVIDIA's gated PhysicalAI-AV. It is
   `gated-confidential`: **not redistributable**, firewalled, recipe-only. It is
   structurally unable to become a record in this lake — the ingestor raises
   `PermissionError` — and it will never appear in this dataset or in
   TanitDataSet-R. Nothing here is derived from it.

5. **Waymo Open / WOD-E2E and Waymax are refused outright**, not merely excluded.
   Their terms follow the *trained weights* into the model and vehicle operation,
   so the contamination would survive training and no tier could contain them.
   They are encoded as a distinct `refuse` license class that raises on ingest.

6. **The split is episode-level, not route-disjoint.** The cache this build read
   had already lost comma2k19's route ids, so train/val were split per episode.
   comma2k19 is one commute route re-driven, so **train and val episodes can share
   road segments.** Do not report a generalization number from this split without
   saying so; rebuild from the comma2k19 origin if you need a strictly
   route-disjoint split.

7. **Near-duplicates are kept on purpose, and the near-dup detector over-collapses
   here.** A two-pass perceptual dedup flagged 67 of 90 as near-duplicates of 23
   exemplars. That is a **detector artifact**, not duplication: the 90 have
   distinct ids and distinct exact-frame hashes, and near-dup pairs sit at
   mid-keyframe L1 ≈ 0.10 vs 0.15 for random pairs — genuinely different highway
   scenes. A single-keyframe 8×8 aHash with transitive union-find chains
   homogeneous highway footage into one smear. **All 90 records ship**; the
   exemplar flag is a sampling hint, and the repeats are wanted multi-traversal
   signal. Control frequency by sampling weight, not by deletion.

8. **No semantic / VLM labels in this release.** The v3 goal vocabulary
   (`VTARGET` / `LONMODE` / `HEADWAY` / lead-state / scene tags) and the
   Chain-of-Causation traces are designed and piloted but are **not** in these
   records. Records carry frames, actions, poses and provenance only.

9. **Anonymization is inherited, not re-applied.** comma2k19 is already publicly
   distributed under MIT as forward-facing US-highway dashcam footage, where PII
   exposure is low; **no additional face/plate blurring was applied by us**, and
   nothing was added that is not already in the upstream release. If your
   jurisdiction requires a face/plate pass before *your* redistribution, run it.

10. **Numbers here are measured on this bundle only.** No TanitAD model result is
    quoted on this card; model facts live in the program's model registry, and
    quoting them from a data card is exactly the error class this program logs.

---

## Relationship to TanitDataSet-R

`R = C ∪ NC` over one schema. **As of this release, [TanitDataSet-R](https://huggingface.co/datasets/Sayood/TanitDataSet-R)
contains exactly the same 90 records as C** — no non-commercial source is
ingested yet, so R currently adds nothing. If you want the commercial tier, this
repo is the one to use.

## Citation / attribution

This dataset is a re-packaging, into the TanitAD canonical world-model contract,
of publicly released data. **Cite the upstream source:**

```bibtex
@article{schafer2018commute,
  title  = {A Commute in Data: The comma2k19 Dataset},
  author = {Schafer, Harald and Santana, Eder and Haden, Andrew and Biasini, Riccardo},
  journal= {arXiv preprint arXiv:1812.05752},
  year   = {2018}
}
```

Attribution and per-source license text ship in **`NOTICE`**.

_Built by the TanitAD Phase-A lake pipeline. Provenance travels with the data._
