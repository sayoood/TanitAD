# TanitAD Data Lake — Architecture Spec

**Author:** Data Engineering (design session). **Date:** 2026-07-13. **Status:** design v1 — for Sayed review.
**Scope:** design / planning only. No infra stood up, no pod / training / cache touched, no change to the running
episode contract (`stack/tanitad/data/_contract.py`). All code in this doc is an illustrative *sketch* (marked as
such); if prototyped it lives on a branch, never merged into the running contract.

> Companion doc: `Data Engineering/OWN_DATASET_PLAN.md` (the source survey + per-license verdicts). This spec is the
> *architecture* that operationalizes that survey's "one dataset for all" conclusion. Where the plan answers **what we
> may ship**, this answers **how it is stored once and served as views**.

---

## 0. TL;DR — the recommendation in one breath

Build the lake on **Cloudflare R2** (S3-compatible, **zero egress** — decisive because the whole point is re-reading
written-once data from many off-AWS RunPod pods without re-processing). Layout is **two tiers**: a **Parquet catalog**
(one row per episode: every scalar field + language/VLA annotations + modality-availability flags + native intrinsics +
`sha256` + `license_class`) over **WebDataset tar shards** holding the canonical `uint8` frame blobs, physically
partitioned by `license_class` and ShareAlike-segregated for ZOD. **Per-source ingestors wrap the existing
`build_episode` adapters** (`comma2k19.py` / `cosmos_drive.py` / `physicalai.py`) and add metadata + language
extraction; each runs **once**. A **view** = Parquet predicate-pushdown + column projection → a shard list → a
`LakeWindowDataset` that streams shards, **caches once locally** (mmap — replacing `build_episodes_cached`'s *decode*
with a *download*), and yields the **byte-identical `EpisodeWindowDataset` window contract**, so it is a drop-in for the
trainers. **PhysicalAI-AV never enters the lake** (recipe-only, or a self-hosted gated tier if legal clears) — the
firewall becomes structural. The owned-safe view exports to `tanitad-own-core` (permissive, private→public) and
`tanitad-own-zod` (CC-BY-SA) via a license-scope-guarded HF exporter.

**Two decisions gate rollout (see §9):** (1) the confidential-AV legal read (may AV bytes ever leave the licensed pod
onto *any* third-party or self-hosted object store?), and (2) the R2-vs-self-host storage-backend call.

---

## 1. Problem, and why a lake

**Today every pod rebuilds the episode cache.** Two independent causes:

1. **RunPod network volumes are single-pod-attach.** A cache built on pod1 cannot be mounted read-only by pod2/pod3;
   each re-materializes its own `<data_root>/_epcache/<tag>-<key>/ep_*.pt` via
   `epcache.build_episodes_cached` (comma2k19 from public HF tars — decode ~40 min for ~1000 videos; PhysicalAI from
   gated chunks + `stack/scripts/build_pai_cache.py`).
2. **The richest corpus can't be shared.** PhysicalAI-AV real is under the NVIDIA AV license — *internal-dev-only,
   confidential, non-transferable, 12-mo expiry, destroy-on-termination* (D-002,
   `Research/2026-07-07-physicalai-av-license-review.md`). Its derivatives may **never** ship to HF, even privately, so
   the program's fallback is a *recipe* (`Sayood/tanitad-realmix`, `stack/DATA_MANIFEST.json`, `scripts/rebuild_cache.py`
   — Sayed doctrine, `BACKLOG.md` P0 `-2`/`-1`): ship IDs + build params + checksums; every pod rebuilds. A throttled
   rsync off the training pod already stalled a record run (2026-07-11 ~21:30) — the incident that motivated
   "rebuildable, not only copyable."

**The lake dissolves both** for the license-clean corpus: normalize each source **once** into a superset schema on a
private object store; every use case is a filtered **view** streamed and cached-once. Pods download normalized bytes
(no decode); the owned subset is publishable; AV stays firewalled. This is the natural home for the survey's
`tanitad-own` conclusion — the survey defines the *shippable set*; the lake is the *store + serving layer* around it.

**Design invariants (must hold):**

- **I-A. Strict superset of the current contract.** The core projection `{frames[T,9,256,256] u8, actions[T,2],
  poses[T,4], episode_id}` is recoverable byte-for-byte, so `assert_contract` and the I7 `CORPUS_META` identity
  (D-017) still pass and existing trainers run unchanged.
- **I-B. Geometry is baked at ingest.** Frames are already D-016-canonicalized (`f_eff = F_REF = 266 px`) via
  `calib.focal_crop_resize` (pinhole) / `ftheta_crop_resize` (f-theta / Kannala-Brandt) — intrinsics collapse to a
  constant, the cache is truly drop-in. Native intrinsics are **retained in metadata** so a future H17 unified-FOV
  canvas can be re-derived without re-ingest (survey §5.4).
- **I-C. License is a first-class, structural axis** — physical partitioning + an export guard, not just a filter.
- **I-D. The recipe never dies.** `MANIFEST.json` (source IDs + build-params hash + per-episode `sha256`) ships with
  every shard, so any consumer can verify-without-rebuild AND rebuild-from-origin if the store is lost — the ultimate
  anti-lock-in fallback.

---

## 2. Canonical SUPERSET schema

**One record = one episode** (an independent recording unit — a comma2k19 segment, a Cosmos/PhysicalAI clip, a ZOD
sequence). Frames and per-frame signals are `T`-indexed *inside* the record. A source that lacks a modality sets the
field **NULL** *and* flips its `modality_flags.*` bit — consumers filter on the cheap flag and never touch the null
column.

### 2.1 The core projection (unchanged current contract)

These four fields are the **exact** contract emitted by every adapter today (`_contract.assemble_episode`,
`comma2k19.stack_frames`, `physicalai.build_episode`). NOT NULL for any world-model-trainable row.

| Field | Arrow / on-disk type | Shape | Semantics | Null? |
|---|---|---|---|---|
| `episode_id` | int64 | — | stable hash of the source unit | NOT NULL |
| `frames` | uint8 tensor (WebDataset `.npy` member) | `[T, 9, 256, 256]` | D-015 3-frame RGB stack @10 Hz `[t−200ms, t−100ms, t]`, current frame in last 3 ch; D-016 `f_eff=266` | NOT NULL |
| `actions` | float32 tensor | `[T, 2]` | `(steer_road_rad, accel_mps2)` | **nullable** (see 2.3) |
| `poses` | float32 tensor | `[T, 4]` | `(x_east_m, y_north_m, yaw_rad, v_mps)`, clip-local ENU | **nullable** (see 2.3) |

> **Storage refinement (recommended, §8 risk 3):** store frames **unstacked** as `[T, 3, 256, 256]` uint8 + an
> `n_stack=3` field, and apply `comma2k19.stack_frames` in the loader. This removes the 3× channel-redundancy of the
> on-disk 9-ch stack (~3× storage saving) and lets `n_stack`/`window` change without re-ingest. It does **not** change
> `_contract.py`: the loader reconstructs the identical `[w,9,256,256]` window (oldest-first, current-last). Flagged as
> a lake-internal optimization; the naive path stores the 9-ch blob as-is.

### 2.2 Core metadata (always present)

| Field | Type | Semantics | Null? |
|---|---|---|---|
| `source` | string (enum) | `comma2k19 \| cosmos_dd \| worldmodel_synth \| pandaset \| udacity \| carla \| zod \| physicalai_av \| covla \| bddx \| drivelm \| …` | NOT NULL |
| `license_class` | string (enum) | `owned-safe \| nc-research \| gated-confidential` (§6) | NOT NULL |
| `share_alike` | bool | copyleft virality flag (ZOD `true`); owned-safe but cannot co-mingle in a proprietary/closed file | NOT NULL |
| `commercial_ok` | bool (derived) | `license_class=='owned-safe' AND NOT share_alike` — the commercial-clean gate | NOT NULL |
| `is_synthetic` | bool | rendered vs real camera (D-010 real/sim role split) | NOT NULL |
| `T` | int32 | timestep count | NOT NULL |
| `hz` | float32 | sample rate (10.0) | NOT NULL |
| `f_eff_px` | float32 | achieved canonical focal (266.0; from `*.last_f_eff`) | NOT NULL |
| `timestamps` | float64 `[T]` | source-clock seconds, re-origined to frame 0 | nullable (synthetic may omit) |
| `split_unit_id` | string | route/clip id — the **I3** split unit (never split one unit's windows across sets) | NOT NULL |
| `split` | string (enum) | `train \| val \| test \| unassigned` (assigned in the manifest, seeded) | nullable |
| `build_params_hash` | string | hash of `{size, n_stack, stride, hz, adapter_version}` (extends `epcache.cache_key`) | NOT NULL |
| `sha256` | string | digest of the frames blob (verify-without-rebuild) | NOT NULL |
| `attribution_id` | string | key into the shard's `NOTICE` (MIT/CC-BY/OpenMDW/ShareAlike/privacy) | NOT NULL |

### 2.3 Motion / control (nullable per source)

| Field | Type | Shape | Semantics | Null when |
|---|---|---|---|---|
| `actions` | float32 | `[T,2]` | `(steer_road_rad, accel_mps2)` — see core | pose-less source with no IDM/VO fill |
| `action_source` | string (enum) | — | `can \| pose_derived \| idm \| vo \| none` — **how** actions were derived | NOT NULL (`none` when unavailable) |
| `poses` | float32 | `[T,4]` | `(x,y,yaw,v)` clip-local ENU — see core | pose-less (e.g. WorldModel-Synth on-card has none) |
| `velocity` | float32 | `[T]` | speed (== `poses[:,3]` when posed; standalone for pose-less speed estimates) | no speed signal |
| `curvature` | float32 | `[T]` | path curvature `κ` (pre-`atan(WHEELBASE·κ)`), provenance for steer | not derivable |
| `has_can` | bool | — | real CAN steer+speed present (vs pose-derived) | NOT NULL |

**Derivation reuses the existing code paths verbatim** (survey §6.3): real CAN → `comma2k19.actions_and_poses` /
`physicalai.signals_at` (`steer = atan(WHEELBASE·curvature)`); 4×4 pose → `cosmos_drive.poses_to_signals`; pose-less →
IDM (H7) or VO, tagged `action_source`, with a per-clip action-quality score gating admission. `accel` is always
`_contract.finite_diff_accel`.

### 2.4 Language / VLA (nullable — only VLA sets)

All NULL for a pure world-model source; populated for CoVLA / BDD-X / DriveLM / future owned captions. Stored as a
JSON member (`{episode_id}.lang.json`) and denormalized into Parquet list-of-struct columns for query.

| Field | Type | Semantics | Populated by |
|---|---|---|---|
| `captions` | list<struct{t_start:f32, t_end:f32, text:string}> | free-text scene/behavior description over a time span | CoVLA, WorldModel-Synth per-cam VLM captions |
| `qa_pairs` | list<struct{q:string, a:string, qa_type:string, t_ref:f32}> | perception/prediction/planning QA | DriveLM (graph-QA), LingoQA, nuScenes-QA |
| `instructions` | list<struct{t_start:f32, t_end:f32, text:string}> | nav command / imperative ("turn left at the lights") | CoVLA, L2D, CARLA scripted |
| `rationales` | list<struct{t_ref:f32, text:string}> | action *explanation* ("slowing because pedestrian ahead") | BDD-X |
| `maneuver_labels` | list<struct{t_start:f32, t_end:f32, label:string}> | discrete taxonomy (turn/lane-change/stop/yield) | CoVLA, BDD-X, derived |
| `language_source` | string (enum) | `none \| covla \| bddx \| drivelm \| worldmodel_synth \| owned_caption \| …` | NOT NULL |

> **VLA license nuance (important).** `license_class` is the **most restrictive of `{frames, language}`**. DriveLM's
> QA annotations are permissively licensed, but they are bound to **nuScenes** imagery (CC-BY-NC-SA) → the joined record
> is **`nc-research`**. CoVLA and BDD-X are likewise `nc-research` (survey / `DATASET_LANDSCAPE.md`). Consequence: the
> **VLA view is primarily an internal-research view.** An **owned-safe VLA** row needs owned frames + owned/permissive
> language — e.g. auto-captioning comma2k19/Cosmos or CARLA scripted instructions (a Phase-1 augmentation, analogous to
> the survey's own-photometric-degradation §5.2). The schema supports both; the license axis keeps them apart.

### 2.5 Filtering metadata (mostly nullable)

| Field | Type | Values | Null? |
|---|---|---|---|
| `scenario` | string (enum) | `highway \| urban \| intersection \| cut_in \| ped \| veh_ped \| lane_change \| emergency_veh \| weather_deg \| nudging \| off_expert \| unknown` | nullable |
| `geography` | struct<country:string, city:string, region:string> | e.g. `{US, San Francisco, CA}` / `{SE, …, Nordic}` | nullable |
| `weather` | string (enum) | `clear \| rain \| snow \| fog \| overcast \| unknown` | nullable |
| `time_of_day` | string (enum) | `day \| night \| dawn \| dusk \| golden_hour \| unknown` | nullable |
| `intrinsics_native` | struct<model:string, params:list<f32>, cx:f32, cy:f32, width:i32, height:i32> | RAW pre-canon intrinsics (pinhole `fx,fy` / f-theta `fw_poly_0..4`); enables H17 re-derive | nullable (→ corpus-median fallback used, per `physicalai.intrinsics_for_clip`) |
| `camera_model` | string (enum) | `pinhole \| ftheta \| kannala_brandt` | NOT NULL |
| `crop_provenance` | struct<crop_side:i32, retained_hfov_deg:f32, f_eff_before:f32> | data-card audit (from `calib.ftheta_feff_report`) | nullable |
| `modality_flags` | struct<has_actions, has_poses, has_can, has_language, has_qa, has_maneuver, has_intrinsics : bool> | **the availability bitmap** — the primary view filter | NOT NULL |

### 2.6 How a source that lacks a modality is represented — worked examples

| Source | frames | actions / `action_source` | poses | language | `modality_flags` (key bits) |
|---|---|---|---|---|---|
| **comma2k19** (real CAN) | `[T,9,256,256]` | `[T,2]` / `can` | `[T,4]` | NULL / `none` | has_actions✔ has_poses✔ has_can✔ has_language✗ |
| **Cosmos-DD** (4×4 pose) | `[T,9,256,256]` | `[T,2]` / `pose_derived` | `[T,4]` | NULL / `none` | has_actions✔ has_poses✔ has_can✗ |
| **WorldModel-Synth** (pose-less on card) | `[T,9,256,256]` | NULL or IDM / `idm`\|`none` | NULL | captions? / `worldmodel_synth` | has_actions✗* has_poses✗ has_language✔ |
| **DriveLM** (QA on nuScenes) | `[T,9,256,256]` | pose-derived (nuScenes ego) | `[T,4]` | qa_pairs / `drivelm` | has_qa✔ has_poses✔ (license `nc-research`) |
| **BDD-X** (rationales) | `[T,9,256,256]` | weak (GPS/IMU→IDM) / `idm` | `[T,4]`* | rationales+maneuver / `bddx` | has_language✔ has_can✗ |

`*` decided by the pose-field probe / IDM agreement score at ingest. A consumer requesting the world-model view filters
`has_poses AND has_actions`; WorldModel-Synth video-only rows are simply absent from that view but present in a
representation-learning (frames-only) view.

---

## 3. Storage format + backend

### 3.1 Format: WebDataset vs Parquet/Arrow vs HF-datasets

| Criterion | **WebDataset** (tar shards) | **Parquet / Arrow** | **HF `datasets`** |
|---|---|---|---|
| Large binary frame blobs (10–170 MB/episode) | **excellent** — arbitrary members, sequential streaming | poor — multi-MB blobs bloat row groups, hurt scan | inherits Arrow; Hub-hosted |
| Column projection (read only captions, skip frames) | **no** — whole sample read | **yes** — predicate pushdown + projection | yes (Arrow) |
| Metadata filtering to build a view | needs a sidecar index | **native** (DuckDB/PyArrow) | via `filter` (materializes) |
| Streaming from S3-compatible store to GPU dataloader | **native** (URL pipeline, `IterableDataset`) | possible, not the sweet spot | streaming mode exists |
| Random access within shard | sequential only | row-group random | row random |
| Mutation / append a source | append new shard (immutable) | rewrite file / new part | new revision |
| Confidential hosting (self / non-HF) | **any S3 store** | **any S3 store** | **HF-coupled** (can't host AV) |

**Neither alone fits a streaming multimodal lake.** Frames want WebDataset; metadata + language + view-planning want
Parquet; and HF-datasets can't host the confidential tier (the core problem). → **hybrid.**

**RECOMMENDATION — two-tier "catalog + shards":**

- **Metadata tier — Parquet.** One row per episode carrying *everything except the frame blob*: all §2.2–2.5 scalars,
  language/VLA list-columns, `modality_flags`, `intrinsics_native`, `sha256`, `license_class`, `split`, plus
  `shard_url` + `member_key`. This is the **catalog**; a view is a Parquet query returning a filtered shard/member list
  — **no frame I/O to plan a view.** Partitioned (Hive-style) by `license_class / source / split`.
- **Frame/heavy tier — WebDataset tar shards.** ~1–2 GB shards, one sample per episode:
  `{episode_id}.frames.npy` (uint8 `[T,3or9,256,256]`), `{episode_id}.motion.npz` (actions/poses/velocity/timestamps),
  `{episode_id}.lang.json` (if any), `{episode_id}.meta.json` (full record — the single source of truth). Sharded by
  `(license_class, source, split)`; **ZOD (ShareAlike) never shares a tar with non-SA data** (§6).

The tar `meta.json`/`lang.json` are the source of truth; the Parquet catalog is a **derived, rebuildable index** over
them (denormalized for query). This mirrors what the code already does — it is essentially the current
`torch.save({...})` dict (`mixing.save_episode`) split into a streamable blob + a queryable row, plus the survey's
`MANIFEST.json`/`DATA_CARD.md`/`NOTICE` (§6.1) as shard-level sidecars.

### 3.2 Backend: Cloudflare R2 vs AWS S3 vs Backblaze B2

Pods run on **RunPod (off-AWS)** and re-read the lake **repeatedly** (multi-pod, multi-epoch, cold cache on new pods).
The dominant cost is therefore **egress**, not storage.

| | **Cloudflare R2** | **AWS S3** (Standard) | **Backblaze B2** |
|---|---|---|---|
| Storage (indicative $/GB-mo) | ~0.015 | ~0.023 | **~0.006** |
| **Egress $/GB** | **0.00 (free)** | ~0.09 (to internet) | ~0.01 (free ≤3× stored/day; free via Cloudflare) |
| Read ops (indicative) | ~$0.36 / M (Class B) | ~$0.40 / M GET | low |
| S3-compatible API | yes | native | yes |
| Throughput / latency | high | **highest** + ecosystem | lower (cold tier) |
| Fit for repeated off-cloud reads | **best (zero egress)** | worst (egress dominates) | cheap-at-rest backup |

**Cost intuition** (2 TB permissive lake, indicative): R2 storage ~$30/mo, **re-reads free**. S3 storage ~$46/mo **+
one full read to a pod = 2 TB × $0.09 ≈ $180**, ×(pods × cold epochs) → hundreds–thousands/mo. That single line is the
quantitative case for R2.

**RECOMMENDATION — Cloudflare R2** as the primary store for the permissive + NC tiers; **B2 as an optional cheap
cold/backup tier**; **S3 only if compute later moves into AWS**. (All pricing indicative — re-verify at procurement.)

### 3.3 The confidential-AV split (and the legal flag)

PhysicalAI-AV is *confidential, non-transferable, destroy-on-termination* (D-002). Uploading its derivatives to **any**
third-party object store (R2 included — a US cloud sub-processor) is plausibly a prohibited "transfer / making
available," independent of bucket privacy (survey §4.2: "a *private* HF repo shared with any third party is still
*distribution*").

**RECOMMENDATION — three physically separated tiers:**

- **Tier P (owned-safe / permissive)** → **R2** bucket/prefix `owned-safe/…` (ZOD under `owned-safe/sharealike/`).
  Private by default; flips to public export without a license change.
- **Tier N (nc-research)** → **R2** prefix `nc-research/…`, **never** included in any public/HF export path.
- **Tier G (gated-confidential = PhysicalAI-AV)** → **NOT on R2.** Default = **status quo, recipe-only**: AV stays the
  `DATA_MANIFEST.json` + `build_pai_cache.py` re-materialization on each licensed pod; the lake merely *references* it
  by manifest, never stores its bytes. *If* legal clears a single-tenant private store, an alternative is a
  **self-hosted MinIO** on a TanitAD-controlled volume, access-scoped to licensed pods, with destroy-on-termination
  hooks — but this needs sign-off first.

> **LEGAL FLAG (blocking for Tier G only; §9.1).** Does the NVIDIA AV license permit AV-derived episodes to leave the
> licensed pod onto (a) a third-party object store like R2 — almost certainly **no**; or (b) a self-hosted,
> single-tenant, access-controlled store you fully own — **unclear**, needs a read. **Nothing in Tiers P/N is blocked
> by this** — the permissive lake proceeds regardless; only AV's storage placement waits on the answer. Default if
> unanswered: AV recipe-only (no change from today).

---

## 4. Write-once ingestion

### 4.1 The per-source ingestor (one per source, run once)

An ingestor **wraps the existing adapter** and adds metadata/language + record assembly + shard write + license tag.
The heavy lifting (decode → geometry-canon → action/pose derivation) is the **unchanged** `build_episode` of
`comma2k19.py` / `cosmos_drive.py` / `physicalai.py`. Sketch (illustrative — not committed to `stack/`):

```python
# SKETCH — lives on a branch, not in the running contract.
class SourceIngestor(Protocol):
    source: str                 # "zod"
    license_class: str          # "owned-safe" | "nc-research" | "gated-confidential"
    share_alike: bool           # ZOD -> True
    def discover(self, root) -> list[Unit]: ...             # reuse existing discover_* (segments/clips)
    def build_core(self, unit) -> ToyEpisode: ...           # reuse existing build_episode -> geometry-normalized frames/actions/poses
    def extract_metadata(self, unit) -> dict: ...           # scenario/weather/geo/time/native-intrinsics (per source)
    def extract_language(self, unit) -> dict | None: ...    # VLA sets only; else None
    def action_provenance(self, unit) -> str: ...           # "can" | "pose_derived" | "idm" | "vo" | "none"

def ingest_source(ing: SourceIngestor, root, store):        # RUN ONCE per source
    units = ing.discover(root)
    train, val = split_units(units, seed=0)                 # I3 route/clip-level, reuse split_by_route/split_clips
    for split, us in (("train", train), ("val", val)):
        with store.shard_writer(ing.license_class, ing.source, split,
                                sharealike=ing.share_alike) as w:   # SA-segregated tar
            for u in us:
                try:
                    ep   = ing.build_core(u)                # decode + D-016 canon + CAN|pose|IDM actions
                    meta = ing.extract_metadata(u)
                    lang = ing.extract_language(u)
                    rec  = assemble_lake_record(ep, meta, lang, ing, split)  # NULL-fills absent modalities + flags
                    validate_superset(rec)                  # assert_contract(core) + flag<->field consistency
                    w.write(rec)                            # frames.npy + motion.npz + lang.json + meta.json
                    catalog_append(rec)                     # one Parquet row (index over meta/lang)
                except Exception as e:
                    skip_log(u, e)                          # F-6 fault tolerance: one bad unit never kills the run
```

### 4.2 Pipeline stages (what `build_core` + assembly do)

```
 SOURCE (per-license fetch)      NORMALIZE-TO-CONTRACT (existing code)     ENRICH            RECORD            SHARD + TAG
 comma2k19  HF tars (MIT)  ─┐    build_episode():                       ┌ metadata:         assemble_        WebDataset tar
 ZOD        dl (CC-BY-SA)  ─┤ ─▶ • decode video (PyAV)                  │  scenario/weather/  lake_record  ─▶ owned-safe/<src>/<split>
 Cosmos-DD  HF (CC-BY)     ─┤    • D-016 focal canon -> f_eff=266       │  geo/time/native-   (NULL-fill      (+ ZOD -> .../sharealike/)
 PandaSet   HF (CC-BY)     ─┤      (focal_crop_resize | ftheta_crop_    │  intrinsics         absent mods,   + Parquet catalog row
 Udacity    torrents (MIT) ─┤       resize | kannala_brandt)            ┼ language (VLA):      set flags,    + MANIFEST/NOTICE/DATA_CARD
 WorldModel HF (OpenMDW)   ─┤    • actions: CAN | atan(wb·κ) | IDM      │  captions/QA/        sha256)       ── (Tier P/N -> R2)
 CARLA      self-render    ─┘    • poses: clip-local ENU                └  instr/rationale                    PhysicalAI-AV -> NOT here
                                 • assert_contract(channels=9) + I7                                            (recipe-only / gated tier)
```

**Idempotent + reproducible.** Re-running an ingestor is a no-op for units whose `(episode_id, build_params_hash,
sha256)` already exist (same guarantee as `epcache`'s resume-by-`ep_%05d.pt`). The privacy pass (face/plate blur for
redistributed real EU data — ZOD, PandaSet; survey §8) is a `--blur` stage before shard write.

### 4.3 Adding a new source (worked example: ZOD)

1. Implement the 4 methods. `discover` = walk ZOD sequences; `build_core` reuses `ftheta_crop_resize` (ZOD is
   **Kannala-Brandt fisheye** — already supported, survey §1) + native CAN steer/accel; `extract_metadata` reads
   country/night/season from ZOD frame metadata + stores the KB polynomial as `intrinsics_native`; `extract_language` =
   `None`.
2. Register `ZodIngestor(source="zod", license_class="owned-safe", share_alike=True)`.
3. `ingest_source(ZodIngestor(), zod_root, r2_store)` **once** → shards land under `owned-safe/sharealike/zod/…`, rows
   appear in the catalog → ZOD is instantly available to every view (SA-gated). ~3–4 h loader (survey §7).

VLA add (CoVLA/BDD-X/DriveLM): same shape, but `extract_language` parses the annotation files into
captions/QA/rationale/maneuver structs, and `license_class="nc-research"` (§2.4 nuance).

---

## 5. Multi-view read layer

**A view = column projection × metadata predicate over the Parquet catalog** → a filtered `(shard_url, member_key)`
list → the streaming loader pulls only those samples, caches once, and presents the **standard window contract**. No
source is re-processed to serve a new view.

### 5.1 The drop-in loader

Must match `EpisodeWindowDataset.__getitem__` **exactly** (this is what the trainer builds in every real path — see
`train_worldmodel.py` `_build_datasets`, which wraps comma/physicalai/cached episodes in `EpisodeWindowDataset`):

```
{ "frames":         float [w, 9, 256, 256],   # to_float_frames applied
  "actions":        float [w, 2],
  "future_frames":  float [H, 9, 256, 256],
  "future_actions": float [H, 2],
  "future_poses":   float [H, 4],
  "pose_last":      float [4],
  "episode_id":     int }                       # (+ "domain" when mixed, via MixedWindowDataset)
```

Sketch:

```python
# SKETCH — drop-in for EpisodeWindowDataset; not committed to stack/.
class LakeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, view: LakeView, window=8, max_horizon=16, cache_dir=...):
        shards = view.resolve()                       # Parquet predicate -> shard/member list (once)
        eps = hydrate_cached(shards, cache_dir)       # download-once -> local ep_*.pt (mmap), else load
        self._inner = EpisodeWindowDataset(eps, window=window, max_horizon=max_horizon)  # REUSE
    def __len__(self):  return len(self._inner)
    def __getitem__(self, i): return self._inner[i]   # byte-identical window contract
```

`hydrate_cached` is the crux: it **replaces `build_episodes_cached`'s *decode* with a *download*** — same on-disk
`ep_%05d.pt` layout, same `mmap=True` load (F-7), same fault-tolerant per-item skips (F-6), same cgroup-bounded working
set. The **second epoch and every subsequent pod read locally** — the per-pod *rebuild* becomes a per-pod *first-touch
download* of already-normalized bytes (no PyAV, no geometry math).

**Integration is trivial** because the trainer already has a `data="cached"` path (`train_worldmodel.py` L121–146)
that reads pre-built `ep_*.pt` from a root. Two options, both backward-compatible:
- **Zero-code:** point `--data cached --data-root <lake_cache_dir>` at the lake-hydrated directory (it *is* `ep_*.pt`).
- **Thin branch:** add `data="lake"` → `LakeWindowDataset(view)`; ~15 lines, additive, nothing else changes.

### 5.2 The standard views

| View | Projection | Predicate | Consumer |
|---|---|---|---|
| **World-model** | frames + actions + poses | `has_actions AND has_poses AND license_class IN scope` | flagship / REF-* trainers |
| **VLA** | frames + language (+ actions) | `has_language` (typically `nc-research`) | Phase-1 VLA / command-conditioning (H12) |
| **Eval / gate** | frames + poses + actions | `split='test'` + scenario/weather slices | D1/D2/D3 + spectral gate slicing |
| **Per-experiment** | any | any predicate (e.g. `source='comma2k19' AND weather='clear'`; `is_synthetic=false`) | one-lever bake-offs (D-004/D-010) |
| **Owned-safe (commercial)** | core (+lang if owned) | `license_class='owned-safe' AND commercial_ok` | commercial-clean model + HF publish |
| **All-internal research** | any | `license_class IN ('owned-safe','nc-research')` | max-data internal training |

Mixing across sources/licenses is the **existing** `MixedWindowDataset` (deterministic ratio, `domain` tag, contract
check) — the lake's owned-safe view and the (internal) AV recipe-cache mix exactly as comma+physicalai do today
(`realmix`), so `sim_frac`/weights and the D-010 bake-off are unchanged.

**Real-only gate integrity (I-invariant preserved):** the eval view filters `is_synthetic=false` so all public D1–D3
numbers stay real-held-out (D-010), and the `data:<source>` exposure tag survives as the `source`/`license_class`
columns — exposure is one Parquet predicate instead of one `grep`.

---

## 6. License-view enforcement

`license_class` is set by the ingestor from a **per-source constant** (never inferred), so it cannot drift. Enforcement
is **structural**, three layers:

1. **Physical partitioning.** R2 prefixes `owned-safe/`, `nc-research/`; **AV is not present** at all. A view cannot
   select what isn't in its tier's buckets.
2. **ShareAlike segregation.** ZOD (owned-safe, `share_alike=true`) lives under `owned-safe/sharealike/` and is **never
   written into the same tar shard** as non-SA data — this satisfies the survey's hard constraint that ShareAlike data
   "must not be co-mingled with non-SA sources in the same redistributed file."
3. **Export guard.** `verify_license_scope(view)` runs before any egress/HF export and **asserts** every row's class is
   within the view's declared scope; a commercial view additionally asserts `commercial_ok` (excludes ZOD). A scope
   violation is a hard failure, not a warning.

### 6.1 owned-safe (commercial-clean) vs all-internal research

- **owned-safe view** (`commercial_ok=true`): permissive core only (comma2k19 MIT, Cosmos-DD CC-BY, WorldModel-Synth
  OpenMDW, PandaSet CC-BY, Udacity MIT, CARLA MIT+CC-BY). Feeds a **commercial-clean model** and is **HF-publishable**.
- **owned-safe + ShareAlike** (`share_alike=true`, ZOD): publishable **only** as CC-BY-SA; excluded from a
  proprietary/closed product; kept in its own shards.
- **all-internal research view** (`owned-safe ∪ nc-research`): adds nuScenes/Waymo/CoVLA/DriveLM/BDD-X etc. for
  internal training only — **never** exported. AV (gated-confidential) is structurally outside even this view.

### 6.2 HF export of the owned subset

Two views → two repos (the survey §6.1 plan, mechanized):

- `export_hf(view=owned_safe, commercial_ok=true)` → **`tanitad-own-core`** (MIT/CC-BY/OpenMDW; private→public) +
  `DATA_CARD.md` + `MANIFEST.json` + `NOTICE`.
- `export_hf(view=owned_safe, share_alike=true, only='zod')` → **`tanitad-own-zod`** (CC-BY-SA) + ZOD privacy notice.
- The exporter **cannot** select gated-confidential (not in the lake) or nc-research (guard blocks it). The AV firewall
  is now an invariant of the store topology, not a discipline to remember.

---

## 7. Migration plan (no pod touched now — this is the plan)

**Current state (grounded).** Three architecture arms (Steering STATE 2026-W31): **main** (flagship 4B, comma+physicalai
`realmix`, ~25.75k→30k), **REF-A** (frozen-DINO, 30k complete, comma-only), **REF-B** (budget arm). Each pod rebuilds
`<data_root>/_epcache/<tag>-<key>/ep_*.pt` via `build_episodes_cached`; pod2 already trains from pre-built caches via
`data="cached"` (has caches, not raw video). AV is recipe-only (`DATA_MANIFEST.json` + `build_pai_cache.py`).

**Phase A — stand up the permissive lake offline (dev box; zero pod/training impact).** Ingest the sources that already
have loaders — **comma2k19 + Cosmos-DD** — into R2 `owned-safe/`. This is copy/normalize of *already-built* episodes;
no decode on any pod. Write catalog + MANIFEST + NOTICE. **Validation = byte-equivalence:** assert
`LakeWindowDataset(view=comma)` yields windows byte-identical to `EpisodeWindowDataset` over the same episodes + seed.

**Phase B — AV as INTERNAL tier (never HF), per the legal decision (§9.1).** Default: AV stays recipe-only; the lake
*references* it via manifest. If legal clears a self-hosted single-tenant store, push AV episodes to
`gated-confidential/` (MinIO) reachable only by licensed pods. Either way AV never touches R2/HF. `realmix` becomes
`lake_view(owned-safe: comma+cosmos) ⊕ AV(recipe|gated)` — same `MixedWindowDataset`, same ratio, same numbers.

**Phase C — cut the running arms over one at a time, only at a checkpoint/restart boundary (never mid-run).** Because
`hydrate_cached` produces the same `ep_*.pt`, a pod switches by pointing `--data cached --data-root <lake_cache>` (or
`--data lake`) at the lake. **First cutover validates equivalence** (same episodes/windows/seed → identical resumed
loss). Order by risk: **REF-B → REF-A → main**, each at its next scheduled restart — the live 30k is never interrupted.

**Phase D — incremental source add (survey ingest order).** **ZOD → PandaSet → WorldModel-Synth (pose-probe first) →
Udacity → CARLA**; each runs its ingestor once → shards appear in views automatically → **bake-off gated** (D-010: a new
share stays only if real held-out D1/D2 don't regress; D-004 one-lever). VLA sets (CoVLA/BDD-X/DriveLM) added as
language rows for the VLA view when the Phase-1 VLA workstream starts.

**Backward-compat guarantees.** (a) `_contract.py`, `assert_contract`, `CORPUS_META`, every `build_episode` adapter —
**unchanged** (the lake wraps them). (b) The window contract is byte-identical. (c) `data="cached"` still works; the
lake is **additive**. (d) The AV firewall (`data:physicalai` tag, no-HF) is preserved and made structural. (e)
`epcache`/`build_episodes_cached` remains the local materializer — the lake changes the *source* (download vs decode),
not the cache format. Rollback = point `--data-root` back at the origin caches.

---

## 8. Cost + effort, risks, recommended stack

### 8.1 Storage + cost (indicative)

Episodes are 256px `uint8` (far smaller than raw). Permissive lake sizing: comma2k19 (full 33 h) + Cosmos-DD shards +
ZOD pilot + PandaSet + Udacity ≈ **0.5–2 TB**; WorldModel-Synth normalized episodes could add ~1–2 TB (264k clips, but
256px not 4K). On **R2**: ~$0.015/GB-mo → **~$8–60/mo** at rest, **re-reads free** across all pods. The unstacked-frame
refinement (§2.1) cuts frame storage ~3×. B2 cold backup ~$0.006/GB-mo. (S3 rejected: one 2 TB pod read ≈ $180 egress.)

### 8.2 Effort

| Work | Estimate |
|---|---|
| Lake framework (schema, Parquet catalog, WebDataset writer/reader, `LakeWindowDataset` drop-in, view resolver, license guard, HF exporter) | **1.5–2.5 eng-weeks** |
| Ingestors: comma2k19 + Cosmos-DD (wrap existing) | ~0.5 day each |
| Ingestors: ZOD / PandaSet / Udacity (loaders exist or survey-costed) | ~2–4 h each |
| WorldModel-Synth (pose-probe + loader) | ~1 day |
| VLA ingestors (CoVLA / BDD-X / DriveLM — language parse + token schema) | ~2–3 days each |
| Migration validation + arm cutover (equivalence tests) | ~2–3 days |
| **To Phase-C (permissive lake + AV internal + arms on lake)** | **~3–4 eng-weeks** |

### 8.3 Risks + mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | **AV transfer legality** (Tier G) | Default recipe-only (status quo); §9.1 legal read gates any AV store move. **Tiers P/N unblocked.** |
| 2 | **ShareAlike contamination** (ZOD virality) | Physical shard segregation + `verify_license_scope` export guard (§6). |
| 3 | **Frame-blob size / streaming throughput** | Store unstacked `[T,3,256,256]` + `stack_frames` in loader (~3× saving; window-size agnostic). |
| 4 | **Pose/action absence** (WorldModel-Synth) | `modality_flags` make it explicit; IDM/VO `action_source` tag; D-010 bake-off gates admission. |
| 5 | **R2 durability / lock-in** | B2 cold backup + `MANIFEST.json`/`sha256` → rebuild-from-origin is the ultimate fallback (recipe never dies). |
| 6 | **Privacy / GDPR** (real EU: ZOD, PandaSet) | `--blur` face/plate stage in the ingestor before shard write (survey §8); ZOD notice travels with the data. |
| 7 | **Catalog drift** vs shards | Parquet is a *derived* index; a `rebuild_catalog` job regenerates it from tar `meta.json` (single source of truth). |

### 8.4 Recommended stack (one paragraph)

**Cloudflare R2** (zero-egress, S3-compatible) as the private object store, with a **two-tier layout**: a **Parquet
catalog** (one row per episode — every scalar field + language/VLA annotations + modality-availability flags + native
intrinsics + `sha256` + `license_class`) over **WebDataset tar shards** of the canonical `uint8` frame blobs,
partitioned by `license_class` and ShareAlike-segregated for ZOD. **Per-source ingestors wrap the existing
`build_episode` adapters** (`comma2k19` / `cosmos_drive` / `physicalai`), adding metadata + language extraction, run
**once** each. A **view** = Parquet predicate-pushdown + column projection → a `LakeWindowDataset` that streams shards,
**caches once locally** (mmap `ep_*.pt` — download replacing decode in `build_episodes_cached`), and yields the
**byte-identical `EpisodeWindowDataset` window contract**, a drop-in for the trainers via the existing `data="cached"`
path. **PhysicalAI-AV never enters the lake** (recipe-only, or a self-hosted gated tier if legal clears), so the
firewall is structural; the **owned-safe view exports to `tanitad-own-core`** (permissive, private→public) and
**`tanitad-own-zod`** (CC-BY-SA) through a license-scope-guarded HF exporter.

---

## 9. Open decisions for Sayed

1. **[BLOCKING for AV tier only] Confidential-AV legal read.** May AV-derived episodes ever leave the licensed pod onto
   (a) a third-party store like R2 — likely **no**; (b) a self-hosted single-tenant store you fully own — **unclear**?
   Default if unanswered: **AV recipe-only** (no change). Tiers P/N proceed regardless.
2. **Storage backend.** Approve **Cloudflare R2** (zero egress) as the primary store, B2 as cold backup? (Alternative:
   self-host MinIO for full control — higher ops, needed anyway for AV if 1(b) clears.)
3. **ZOD ShareAlike** (carried from survey §9.1). Accept **CC-BY-SA-4.0** on the public ZOD shard (forces open for
   anything co-mingled)? Recommend: yes, as a **separate** `tanitad-own-zod` shard; keep the commercial view SA-free.
4. **MIT-on-data blessing** (survey §9.2). One-line legal nod that MIT-licensed *data* (comma2k19, comma10k, Udacity) is
   fully redistributable (MIT's text says "Software").
5. **WorldModel-Synth firewall** (D-022, currently HOLD). Confirm **OpenMDW-1.1 → public-claimable** so the 264k-clip
   long-tail can be cited, not only trained on. Until then it ingests as `owned-safe` for training but is export-held.
6. **VLA scope.** The VLA sets (CoVLA/BDD-X/DriveLM) are **`nc-research`** (NC frames) → internal VLA view only. Fund an
   **owned VLA** path (auto-caption owned frames / CARLA scripted instructions) for a commercial-clean VLA view?
7. **Unstacked-frame storage** (§2.1). Approve storing `[T,3,256,256]` + loader-side `stack_frames` (~3× storage
   saving, window-agnostic) vs the naive `[T,9,256,256]` blob?
8. **Privacy bar** (survey §9.7). Confirm the face/plate anonymization required to redistribute real EU data (ZOD,
   PandaSet), independent of copyright.

### Provenance

Current contract & plumbing: `stack/tanitad/data/{_contract,comma2k19,physicalai,cosmos_drive,calib,mixing,epcache}.py`;
trainer integration `stack/tanitad/train/train_worldmodel.py` (`_build_datasets`, `data="cached"` path). Source survey +
license verdicts: `Data Engineering/OWN_DATASET_PLAN.md`; landscape `Research/DATASET_LANDSCAPE.md`; recipe doctrine
`BACKLOG.md` P0 `-2`/`-1` (Sayed 2026-07-11). Decisions: D-002 (AV license), D-004 (one lever), D-009 (comma first),
D-010 (real/sim bake-off), D-012 (landscape duty), D-014 (Cosmos sim arm), D-015 (3-frame 9-ch stack), D-016 (focal
canon), D-017/I7 (`CORPUS_META` identity), D-022 (WorldModel-Synth firewall hold); invariants I3 (split unit), F-6/F-7
(fault-tolerant + mmap cache). This is an engineer's design to route decisions, not legal advice.
