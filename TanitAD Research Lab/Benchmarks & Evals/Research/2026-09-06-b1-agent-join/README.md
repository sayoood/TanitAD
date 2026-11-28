# B1 EVAL agent join — `obstacle.offline` scoped to the corpus v7 actually evaluates

**Date** 2026-09-06 · **Agent** B1 agent-join · **Evidence class** MEASURED (ours) ·
**Tier** N/A (a label artifact, not an eval result) · **Compute** dev-box CPU, 52.7 s, **zero GPU**

WP-6 (agent conditioning) was blocked in three separate reports on a B1-scoped
`obstacle.offline` join. This package is that join, plus the proof it lands on the
right frames.

---

## 1. The blocker, as a fraction with both numbers and the corpus named

⛔ A percentage alone is inadmissible here — two different fractions round to the same one.

| join | B1 clips it covers | scope |
|---|---|---|
| `train2400_agents.jsonl.xz` | **193 / 4,719** | TRAIN corpus (`physicalai-train-e438721ae894`) |
| `val40_agents.jsonl` | **6 / 4,719** | val40 parity deployment |
| `lead130_agents.jsonl` | **12 / 4,719** | a subset *of* the train join |
| **union of all three** | **199 / 4,719 = 4.22 %** | — |

And on the set that actually matters — the **141-clip B1 EVAL split every v7 T1/val
number is computed over** — the union covered **11 / 141**. So **130 of 141** eval
clips had no agent labels at all. That is the blocker, stated as a fraction.

**This package: 139 / 141 B1 EVAL clips joined.**

## 2. What B1 is (the arithmetic that reconciles 4,719 / 4,713 / 147 / 141)

| number | what it is |
|---|---|
| 4,719 | B1 clips in the v7 r0 selection (`physicalai-b1/r0/r0_selection.parquet`) |
| 4,713 | actually built — `parity.require_ingest_gate` drops 6 val40 clips |
| 4,572 | v7.2 TRAIN split |
| **147** | v7.2 EVAL split — what the B1 EVAL lead block covers |
| **141** | the 147 that have pixels inside the built B1 epcache — **what refav1 scores on** |

⚠️ **"Two corpora, one phrase."** B1 and the parity train corpus are largely disjoint:
`|TRAIN2308 ∩ B1| = 193`. A claim about "our corpus" that does not name which one is
inadmissible — this exact confusion has already produced one false absence claim on
this topic.

## 3. ⛔ The index-space trap, and why this is a separate builder

`build_obstacle_join.py` emits `frame_idx` in **EPISODE index space** — the
post-`n_stack`-trim provider index `i − (n_stack − 1)`. The B1 EVAL lead block emits
`frame` in the **RAW v2ep index** `i ∈ [0, n_target)`, because the refav1 loader reads
RAW poses with no trim. With `n_stack = 3` the two spaces differ by **2 frames ≈ 0.2 s**
— about **2.7 m of lead displacement at 13.6 m/s**. They are not interchangeable.

⭐ **This artifact emits BOTH, because the corpus has two consumers that disagree:**

| key | index space | consumer |
|---|---|---|
| `frame` | RAW v2ep index | refav1 adapter / `b1_eval_lead_block.npz` |
| `frame_idx` | post-`n_stack`-trim | `train_p8_occupancy.JoinFileReader`, which `refc_v3_train.py --agent-join` uses |

A reader ignores keys it does not know (pinned by `build_obstacle_join`'s roundtrip
test), so one artifact serves both and **neither consumer has to guess**. `frame_idx`
is negative for the first `n_stack − 1` frames; those keys are never looked up, and
dropping the rows instead would lose raw frames the lead-block consumer wants.

## 4. Alignment is PROVEN, not asserted

⛔ **The gate is the SPEED agreement, not the time agreement.** Both this join and the
lead block carry the ego speed at the same frame index (`poses[:, 3]` and `speeds`),
both egomotion-interpolated there — so their agreement proves the join hit the right
clip **and** the right frame, and it is **independent of the registration fit**.

| statistic | value |
|---|---|
| TRUE `max\|Δv\|` | **1.472e-04 m/s** |
| tolerance | 1.0e-03 m/s — adopted verbatim from `refav1_arm.py:238` `LEAD_SPEED_TOL_MPS` |
| mis-join(+1) control, `min\|Δv\|` | **2.129e-02 m/s** |
| **separation** | **144.6×** |

**Reported, not gated** — time agreement: TRUE `max|Δt|` 1.040e-02 s, mis-join(+1)
1.071e-01 s, over 138 non-circular clips.

⚠️ **A first cut GATED on `|Δt| ≤ 5e-3 s` and refused the whole build on ONE clip**
(`fe764229`, 1.04e-02 s). Diagnosis: that clip registers on **23 probes** where healthy
clips use 96, so its fitted time *offset* is less precise — a smooth ~7 ms bias across
all frames, **not** a ~0.1 s frame step. Its speed agreement is 2.25e-05 m/s, **44×
inside** the proven tolerance, and `lead_source`'s own documented accuracy is *"worst
25.9 ms over 500 clips"*. **The 5e-3 s figure was invented here and was measuring
registration PRECISION while the question asked was KEY CORRECTNESS.** The gate was
moved to the correct statistic, not loosened on the one it was failing — and the
control now sits on the same statistic as the gate.

⚠️ **Honest limit of one control.** Key *existence* against the block cannot
discriminate: the block covers every frame densely, so a +1-shifted key set still
intersects 26,328 / 26,394. The discriminating control is the speed gate above.

### Guards proven by MUTATION, not inspection

| mutation | result |
|---|---|
| M1 empty join | **REFUSED** (zeros artifact) |
| M2 duplicate `(clip_id, frame)` | **REFUSED** (ambiguous join) |
| M3 frames shifted by +1 | **REFUSED** by the gate — TRUE 1.096e-05 vs shifted 1.453e-01 m/s |
| C1 valid 2-line join (control) | **PASSED** — so M1/M2 are not refusing for the wrong reason |

## 5. The artifact

`raw/b1eval_agents.jsonl.xz` — md5 **`3ddb42ecbd3926066795a94587af2aed`** (scope:
**compressed**), **10,012,564 B**; sidecar `raw/b1eval_agents.jsonl.xz.meta.json`.

⚠️ **Corrected 2026-09-06 (B1 TRAIN join agent).** This line, and the claims register
row that copied it, read **9,960,084 B** — a transcription error, low by exactly
**52,480**. The md5 and the file were always right; only the byte count was wrong, which
is why nothing downstream caught it. Re-MEASURED by reading the artifact:
`len(open(...,'rb').read())` = **10,012,564**, md5 unchanged. *(The class is the one this
programme keeps paying for — a number quoted without re-reading its artifact. The digest
was verifiable and verified; the size was neither.)*

| | |
|---|---|
| clips joined | **139 / 141** offered (B1 EVAL) |
| lines (labelled frames) | **26,394** — unique keys 26,394, no duplicates |
| agent boxes | **905,512** |
| visible (in-field) boxes | 369,310 — `visible_frac` **0.4078** |
| labelled frames per clip | min 5, max 201 |
| `max_agents_per_frame` | 395 |
| mean \|cx\|, \|cy\| | 49.23 m, 23.42 m (non-degenerate — the zeros-file assertion) |

⭐ Independent sanity check: `visible_frac` **0.4078** here vs **0.4106** on the
train2400 join — two separately built joins over disjoint clip sets agreeing to
0.003 is a real cross-check, not a restatement.

**Class histogram** (905,512 boxes): automobile 676,084 · person 164,275 · rider 22,977 ·
heavy_truck 18,300 · bus 8,440 · trailer 7,514 · protruding_object 3,534 ·
other_vehicle 3,136 · stroller 723 · animal 460 · train_or_tram_car 69.
⚠️ That is **11** distinct labels, not the "10 classes" carried in the brief;
`protruding_object` is also not obviously a *dynamic agent*. Flagged, not resolved.

### The 2 clips not joined — and the 1 that was rescued

* `204b15b4-…`, `38fe80c1-…` — **no `obstacle.offline` member exists** for them. These
  are the **same 2 clips the lead block itself reports** as label-less, so this is a
  corpus fact, not a build failure. NO_LABEL, never "road clear".
* `af6f5964-…` — content registration is **impossible**: *"only 0 of 201 poses are
  moving (> 0.3 m between neighbours)"*. `register_poses_to_time` identifies time by
  POSITION, and a clip that does not move carries no positional signal. **Recovered**
  by taking the time base from the lead block (which derives it from the timestamp
  grid, not from position). Marked `time_source: lead_block_t0_s`, and **excluded from
  the reported Δt** because that comparison is then circular. Its speed gate is not
  circular and still applies.

## 6. Verified through the REAL consumers

Not by inspecting the file — by loading it with the code that will consume it.

| consumer | result |
|---|---|
| `train_p8_occupancy.JoinFileReader` (what `refc_v3_train.py --agent-join` uses) | `n_records=26394  n_clips=139  has_occlusion_flags=True  has_classes=True  max_agents_per_frame=395`; `lookup(eid, 10)` **HIT** (13 agents), `lookup(eid, 100000)` **None** (the must-miss control) |
| refav1 / lead-block key `(clip_id, RAW frame)` | **26,394 / 26,394 = 100.0 %** of keys present in the banked block |

## 7. Conventions (inherited, not reinvented)

Geometry, visibility and the NO_LABEL rule are **imported** from
`build_obstacle_join` so the two joins cannot drift.

* `cx/cy/yaw` — per-frame **EGO** frame, +x fwd, +y **LEFT**; `l = size_x`, `w = size_y`.
* composition — rig@sample → world at the sample's **own** timestamp → ego@frame.
* `occ` — 0 = agent centre inside the 120° front-camera field, 1 = outside while the
  track continues. ⛔ It **IS** `bev_raster.fov_mask` at agent-centre granularity
  (`P4_PREDICATE_IDENTITY`) — a field mask, **not** an independent occlusion label.
* **NO_LABEL vs labelled-clear** — an absent `(clip, frame)` line is NO_LABEL; an
  **empty `agents` list IS a label** meaning clear road. Collapsing them trains the
  head that an unlabelled frame is an empty road.

## 8. Reproduce

```
set PYTHONPATH=<repo>/stack;<repo>/taniteval
python stack/scripts/build_b1_agent_join.py ^
    --eps-dir      C:/Users/Admin/tanitad-data/refav1-eval141/eps ^
    --ego-dir      C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo ^
    --obstacle-dir C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1eval ^
    --lead-block   "<repo>/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz" ^
    --out          <out>/b1eval_agents.jsonl.xz
```

⚠️ **Zero download was needed.** The B1-eval obstacle parquets were **already
extracted** at `.../labels/obstacle_offline_b1eval` (145 files, 73.2 MB) by the
lead-block build. A first plan sized an **8.6 GB** chunk-zip pull (110 obstacle +
92 egomotion chunks) before probing a second location. *Absence found at one
location is not absence* — and here it was worth 8.6 GB.

## 9. What this does NOT cover — the named next lever

This join is **B1 EVAL** (141 clips). **Agent conditioning during TRAINING needs the
B1 TRAIN split — 4,572 clips — and that join does not exist.**

Cost, priced by opening the **consumer's** loader (`build_b1_agent_join.read_raw_poses`
for poses, `pandas.read_parquet` for labels), not by measuring a file found on disk:

* obstacle parquets for 4,572 clips are **not** extracted; B1 spans **1,411 chunk zips**
  against **58 obstacle / 197 egomotion** held locally.
* MEASURED locally: obstacle chunks average **44.2 MB** (2.57 GB / 58), egomotion
  **41.5 MB** (8.18 GB / 197) ⇒ a full-chunk pull is **≈ 121 GB** for ~3.3 B1 clips per
  chunk. C: has 237.6 GB free, so it fits, but it is wasteful by ~an order of magnitude.
* ⭐ **The cheap route already exists and should be used instead**:
  `taniteval/tools/build_lead_block_b1.py` carries an **HTTP range-request reader for
  remote zips** (`pull_obstacle_offline`) that pulls *individual per-clip members* out
  of a chunk zip without downloading it. That is how these 145 parquets were obtained.
  Re-pointing it at the 4,572 TRAIN clips is the next lever, and it is **I/O-bound, not
  GPU-bound** — it does not contend with the live refcv5 run.

⛔ Not attempted here: that pull is a large network job and the build itself would then
be ~30 min of CPU (52.7 s for 141 clips ⇒ ~28 min for 4,572, linear). Named as the next
lever rather than started, because it is a multi-GB network spend.

---

### Deliverable manifest

| artifact | where it lives |
|---|---|
| builder | `stack/scripts/build_b1_agent_join.py` (repo) |
| join | `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-06-b1-agent-join/raw/b1eval_agents.jsonl.xz` (repo) |
| sidecar | same dir, `…​.meta.json` — per-clip stats, alignment proof, provenance |
| this doc | same dir, `README.md` |
| guard mutation test | `stack/tests/test_b1_agent_join.py` (repo) |
| working copies | `C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\` (dev box) |
