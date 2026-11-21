# v2 corpus QA — every one of the 9,000 built clips, opened and re-labelled

**Date:** 2026-07-25 · **Discipline:** Data Engineering ·
**Slug:** `2026-07-25-v2-corpus-qa`
**Corpus:** `physicalai-v2bal-4b7eeeac222d` — 9,000 clips,
`tanitad-pod`:4,953 + `tanitad-pod3`:4,047, `*.v2ep.pt` under
`/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d/`
**Scope:** READ-ONLY verification before any model trains on it.

---

## VERDICT — **GO on the data. NO-GO on launching today** (two code/ops gates)

| | |
|---|---|
| **P1 integrity** | ✅ **9,000 / 9,000 loadable. Zero corrupt, zero truncated, zero malformed.** 1,808,710 JPEG frames decoded individually — **0 failures**. 0 non-finite pose/action values. |
| **P2 distribution** | ✅ **The balancing worked.** Turns **28.04 %** against a 28.0 % target (parity: 14.25 %). Every class within **0.07 pp** of target. |
| **P3 consolidation** | ✅ Shards **disjoint (0 overlap)**, union **exactly the 9,000** selected, and the corpus key **recomputes to `4b7eeeac222d`** from the built files. |
| **P4 train-readiness** | ⚠️ The staged loader works and the **12-key window contract is intact** — but **neither pod's checkout can run it** (see gates below). |

**The two gates before a v2 flagship launch** (neither is a data problem):

1. **The pods cannot run `--v2-cache`.** MEASURED 2026-07-25: `grep -c v2-cache
   stack/scripts/train_flagship4b.py` returns **0** on `tanitad-pod` *and*
   `tanitad-pod3`, and `stack/tanitad/data/v2_dataset.py` **does not exist** on
   either. Both pods sit at `0f93b98` (dirty) with a stale `refb_labels.py`
   (pod1 **474** lines, pod3 **745**, repo **1300** — pod1's has no `route_from_future_v21`
   at all). A launch today dies on an unrecognised flag, then on `ModuleNotFoundError`.
   **Fix: sync `stack/` to the repo.** This QA proved a repo-synced shadow stack
   (`/workspace/tmp/qa_stack`) runs the corpus end-to-end.
2. **No single node can mount all 9,000 clips.** The shards live on two
   single-attach volumes. `--v2-cache` accepts multiple dirs *visible to one
   host*, which these are not. Consolidate (pod3's shard is 10,660,387,873 B = 9.93 GiB; pod1 has 79 GB
   free local NVMe) or run on a node that holds both.

Everything below is **MEASURED** on the built bytes unless marked otherwise;
artifacts are in the manifest. Times are UTC (pods); Sayed's local is UTC+2.

---

## P1 — Integrity: every clip opened, every frame decoded

Method: `v2_corpus_qa_scan.py` loads each `*.v2ep.pt`, checks the
`build_compressed` key set, decodes **every** JPEG individually, channel-stacks
them exactly as `load_compressed` does, and validates every derived tensor.
Not a sample — all 9,000.

| Check | Result |
|---|---|
| Clips scanned / loadable | **9,000 / 9,000** |
| Corrupt · truncated · malformed | **0 · 0 · 0** — `bad_clip_ids: []` |
| JPEG frames decoded / failed | **1,808,710 / 0** |
| Non-finite values in `poses` / `actions` | **0 / 0** |
| Near-constant (black/flat) frames, 1-in-20 sampled | **0** |
| `image_size` · `n_stack` · JPEG `quality` | **256 · 3 · 90**, uniform across all 9,000 |
| `poses`/`actions`/frame-count alignment | exact on all 9,000 (`[T,4]` / `[T,2]` / T JPEGs) |
| Stacked length `T_out` | 189–205, mean **198.968** (8,531 clips at exactly 199) |
| Total stacked frames · label steps | **1,790,710** · **1,610,710** |
| **Trainable hours** | **49.742 h** stacked · 50.242 h raw (pre-stack drop) |
| Cache size | **22.32 GB**, 2.663 MB/clip (1.09–5.38 MB) |
| Scan wall-clock | pod1 **91.7 s** (16 workers, local NVMe) · pod3 **1,285.3 s** (12 workers, MooseFS-bound at 2.1 clip/s) |

**On "50.25 h":** both numbers are right, they measure different things. 50.242 h
is the raw 10 Hz resample of the ~20.1 s camera spans; **49.742 h** is what a
trainer actually sees after the D-015 3-frame stacking drops 2 frames per clip.
Quote **49.742 h** for training compute.

**Independent build-log cross-check (rule 2 — absence needs a second probe):**
`grep FAILED` over all ten `worker_*.log` on both pods returns **zero lines**;
the five pod3 workers report `DONE built=` summing to exactly 4,047. The 9,000
file count and the logs agree.

### The strongest integrity evidence: the v2 build reproduces the parity build exactly

571 clips are in **both** the trusted 13.13 h parity epcache and pod1's v2 shard.
Matching them on the **full clip_id** (resolved via the stored `episode_id`'s
*unique* 4-char prefix among the 3,000 discovered parity clips — the `ep_NNNNN`
file index is **not** the discovery index; it is offset by 2 on this cache, which
is why a first naive attempt looked like a mismatch):

| | |
|---|---|
| Clips compared | **571** |
| `T` equal | **571 / 571** |
| **Poses BIT-IDENTICAL** | **571 / 571** (max abs Δ = **0.0** on x, y, yaw, v) |
| Maneuver histograms identical | **571 / 571** |

Two independent code paths — `physicalai.build_episode` and
`v2_compressed._resampled` — produce the *same trajectory to the last bit*. Every
kinematic label in this corpus therefore rests on the same pose pipeline the
parity flagship was trained on.

### Outliers, all benign (named, not hidden)

- **5 clips** travel **18.9–19.6 m** on the built poses, just under the selector's
  20 m gate — which was evaluated on the egomotion resample (20.0–21.1 m for the
  same clips). Sub-metre disagreement at the boundary, not degenerate data; all
  five are slow, stop-heavy urban clips (`stop_frac` 0.41–0.81), i.e. the scarcest
  and most wanted class. Clip ids in `v2_corpus_qa.json`.
- **3 clips** have `T_out = 189` (191 raw frames — a slightly shorter source mp4).
  All are normal driving (7.2–21.9 m/s). Well above the 28-frame window+horizon.
- **33 clips** at `T_out = 205`. No clip is too short to window.
- Country max share **5.51 %** (United States) — the design's ~5 %/country claim holds.

---

## P2 — Distribution: did the balancing actually land?

**Yes.** Recomputed with the repo-authoritative `refb_labels.maneuver_labels`
(v1 kinematic, `LABEL_HORIZON=20`, md5 `6632348b…`) over **all 1,610,710
non-sentinel timesteps** of the built corpus. The parity column is **not
inherited from the design doc** — it is re-measured from the parity epcache with
the identical labeler.

### Per-timestep maneuver distribution

| Class | Parity 13.13 h (MEASURED) | **v2 TARGET** | Selection projection | **v2 ACHIEVED (MEASURED)** | Δ vs target |
|---|---|---|---|---|---|
| lane_keep | 59.64 % | 45.0 % | 45.00 % | **44.93 %** | −0.07 pp |
| turn_left | 6.86 % | 14.0 % | 14.00 % | **14.01 %** | +0.01 pp |
| turn_right | 7.39 % | 14.0 % | 14.00 % | **14.03 %** | +0.03 pp |
| accelerate | 13.23 % | 13.0 % | 13.00 % | **13.03 %** | +0.03 pp |
| brake_stop | 12.88 % | 14.0 % | 14.00 % | **14.00 %** | 0.00 pp |
| **turns (L+R)** | **14.25 %** | **28.0 %** | **28.00 %** | **28.04 %** | **+0.04 pp** |

Turn mass **1.97×** the parity corpus; lane-keep dominance cut by 14.7 pp.
L/R balance is 14.01 / 14.03 — symmetric.

### Speed regime (step-weighted)

| Regime | Parity (MEASURED) | Target | Projection | **v2 ACHIEVED** |
|---|---|---|---|---|
| stopped (<1 m/s) | 7.76 % | 10 % | 9.74 % | **9.72 %** |
| city (1–12 m/s) | 45.93 % | 52 % | 52.26 % | **52.28 %** |
| highway (>12 m/s) | 46.32 % | 38 % | 38.00 % | **38.01 %** |

Highway is held at 38 % as designed — the turn gain did not come out of highway.

### Per-clip scenario presence

| Event ≥1× in clip | Parity (MEASURED) | Projection | **v2 ACHIEVED** |
|---|---|---|---|
| any turn | 42.59 % | 76.41 % | **76.46 %** |
| junction-scale (v2.1 tight-transient) | 37.84 % | 61.34 % | **61.38 %** |
| net heading > 45° | 25.00 % | 40.36 % | **40.26 %** |
| net heading > 90° | 10.44 % | 13.88 % | **13.91 %** |
| any full stop (v < 0.5 m/s) | 22.98 % | 27.30 % | **27.31 %** |
| any brake_stop label | 57.49 % | 73.37 % | **73.43 %** |

### The selection-time proxy was near-perfect (per clip, n = 9,000)

The selector scored clips from **egomotion alone** on a synthetic
`linspace(t0, t0+20.1 s, 201)` window; the build used the camera's **real**
timestamps. That substitution could have silently moved the distribution. It did
not:

| | |
|---|---|
| Pearson r, per-clip turn fraction (projected vs built) | **0.9997** |
| Mean abs per-clip turn-fraction error | **0.36 pp** |
| Junction-flag agreement | **99.72 %** |
| `has_turn` agreement | **99.89 %** |
| `T_out` exactly as projected | 8,531 / 9,000 |
| Mean-speed mean abs error | 0.033 m/s |

Per-shard (a sanity check that neither pod drifted): turns **27.35 %** (pod1) /
**28.89 %** (pod3) — the water-filling selector was ordered by clip_id, so the
two halves differ slightly and only the union is on target. **Do not train on one
shard alone and expect 28 %.**

### One honest caveat — the 28 % is a **v1-labeler** number

Under the curvature-gated **v2** labeler (`maneuver_labels_v2`, which calls a
gentle highway sweep lane-keeping) the same corpus reads **18.83 % turns**
(9.70 L / 9.13 R) vs parity's 11.56 % — a **1.63×** gain instead of 1.97×. The
flagship config measured in P4 has **`labels_v2 = False`**, so a default run sees
28.04 %. If a future run flips `--v2-labels`, its effective turn share is 18.83 %,
not 28 %. Both numbers are in `v2_corpus_qa.json`.

---

## P3 — Disjointness, completeness, consolidation readiness

| Check | Result |
|---|---|
| pod1 built · pod3 built · union | 4,953 · 4,047 · **9,000** |
| **clip-id overlap between shards** | **0** |
| Selected clips missing from both shards | **0** |
| Built clips not in the selection | **0** |
| **Corpus key recomputed from the BUILT union** | **`4b7eeeac222d`** = expected ✅ |
| Double-built with differing bytes | **0** — impossible: zero overlap, and within a shard the filename *is* the clip_id |
| Chunks whose clips span both pods | 177 / 200 (the split was by clip-id, not chunk — harmless, the cache is clip-id-keyed) |

Recomputing the sha1 over the sorted **built** clip_ids + target + K and getting
`4b7eeeac222d` back is the end-to-end proof that what is on disk *is* the
designed corpus — not merely 9,000 files that happen to number 9,000.

### FINDING (medium, pre-existing, not a training blocker): `episode_id` is not unique

`build_compressed` sets `episode_id = int.from_bytes(clip_id.encode()[:4])` — the
**first 4 hex characters** of the UUID, i.e. 16 bits of entropy.

| | v2 corpus | Parity corpus |
|---|---|---|
| Clips | 9,000 | 2,376 |
| Distinct `episode_id` | **8,391** | 2,342 |
| **Collisions** | **609 (6.8 %)** | 34 (1.4 %) |
| Max multiplicity | 4 | — |

The flagship trainer emits `episode_id` in every window but does not consume it,
so **this does not affect a training run**. It matters for anything that groups
*by episode*: episode-disjoint splitting, and the decision-grade
**episode-cluster bootstrap** (`taniteval/ci.py`) — on this corpus 609 pairs of
genuinely different clips would be silently clustered together, narrowing the
interval. It is **not a v2 regression** (parity has the same defect at 1.4 %),
but it scales with corpus size and should be fixed before v2 is ever used for
clustered inference. Cheap fix: hash the full clip_id.

---

## P4 — Train-readiness smoke (staged `--v2-cache` path)

Ran the **exact** lines `train_flagship4b.py` executes under `--v2-cache`
(`build_v2_providers` → `_wrap` → `DataLoader`) at the real `flagship4b` config,
against pod1's **whole 4,953-clip shard** (not a sample), from a repo-synced
shadow stack.

**A · The builder's reader and the trainer's reader agree, byte for byte.**
On 24 randomly sampled **real** built clips,
`v2_compressed.load_compressed` (build path, pulls pandas/pyav) vs
`v2_dataset.decode_full_episode` (training path, torchvision only):

| frames byte-identical | poses identical | actions identical | mismatches |
|---|---|---|---|
| **24 / 24** | **24 / 24** | **24 / 24** | **0** |

**B · Providers over the whole shard.** 4,953 lazy providers; frames
`[199, 9, 256, 256]` uint8, poses `[199,4]`, actions `[199,2]`; manifest build
**2.3 s** (0.44 s warm), **0.82 GB** RSS, 26.6 MB `_v2manifest.pt` sidecar.
Windows: **846,854** on this shard; **≈1,547,710 on the full 9,000** (window 8,
max_horizon 20).

**C · The 12-key window contract is intact.** A real window from the full corpus
carries exactly:

`actions [8,2] f32` · `frames [8,9,256,256] f32` · `future_actions [20,2] f32` ·
`future_frames [20,9,256,256] f32` · `future_poses [20,4] f32` ·
`pose_last [4] f32` · `pose_prev [4] f32` · `maneuver_label [] i64` ·
`nav_cmd [] i64` · `nav_valid [] bool` · `route_target [] i64` · `episode_id int`

Batched at 16: `frames [16,8,9,256,256]`, `future_frames [16,20,9,256,256]`.

**D · Throughput and RAM** (batch 16, `--v2-lru 64`, pod1 while the YouTube
harvest ran; subtree-only RSS):

| workers | windows/s | frame bytes/s | subtree peak RSS |
|---|---|---|---|
| 4 | 20.0 | 1.3 GB/s | **4.4 GB** |
| 8 | 37.6 | 2.5 GB/s | **16.8 GB** |
| 16 | **49.4** | 3.3 GB/s | **30.4 GB** |

**Read the RAM number before choosing `--workers`.** ~1.9 GB/worker at 16, and it
is **not** the LRU (64 clips ≈ 173 MB): it is `future_frames`, **66.1 MB of
float32 per window**, in flight ×2 prefetch per worker. On pod1's 503 GB this is
free; **on a ~55 GB cgroup (pod2) 16 workers would OOM** — that box is where the
flagship has been OOM-killed before. Budget ≈2 GB/worker.

**E · No mutation.** `find <cache> -name '*.v2ep.pt' -newermt 2026-07-24T22:50`
→ **0**. The only file this QA created inside the corpus dir is
`_v2manifest.pt` — the loader's own index sidecar, additive and auto-rebuilt when
the clip set changes (i.e. it will rebuild itself on consolidation).

---

## What I could not verify, and why

- **A single-node batch over all 9,000 clips.** The shards sit on two
  single-attach volumes; the honest smoke is pod1's 4,953 (55 % of the corpus,
  846,854 windows). Nothing in the loader is shard-size-dependent, but this is
  *stated as unverified*, not assumed away.
- **Pixel-level fidelity of the f-theta crop vs the parity pipeline.** Verified
  at Phase 2 against a re-decoded mp4 (INHERITED, `V2_PHASE2_BUILD.md` §3); I
  re-verified only that the JPEGs decode and that **poses** are bit-identical to
  the parity build (571/571 — see P1). Re-checking pixels needs an mp4 re-decode
  the corpus no longer stores.
- **Semantic coverage** (lights, roundabouts-as-class, pedestrians, merges) is
  still **0 % labelled** — unchanged from the design's §8. This QA measured
  kinematics, which is all the corpus claims.

---

## Deliverable manifest

All artifacts are **staged in the repo working tree** (no commit, no push).

| Artifact | Location | What it is |
|---|---|---|
| `V2_CORPUS_QA.md` (this file) | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-25-v2-corpus-qa/` | the report + GO/NO-GO |
| `v2_corpus_qa.json` | same dir | machine-readable: every number above, plus per-shard splits, outlier clip ids, pod code state |
| `v2_corpus_qa_scan.py` | same dir | the scanner — full-decode integrity + label recompute; reusable on any v2 cache |
| `qa_full_pod1.{json,csv}` · `qa_full_pod3.{json,csv}` | same dir | **per-clip raw evidence**, 9,000 rows (41 columns each) |
| `parity_profile.py` · `parity_profile.{json,csv}` | same dir | the parity 13.13 h profile, re-MEASURED (2,376 rows) |
| `crossbuild_poses3.py` · `crossbuild_poses3.json` | same dir | the 571-clip bit-identity proof vs the parity epcache |
| `p3_disjoint.py` · `p3_disjoint.json` | same dir | disjointness, completeness, corpus-key reproduction |
| `p4_train_readiness.py` · `p4b_readiness.py` · `p4_pod1.json` · `p4b_pod1.json` | same dir | the `--v2-cache` smoke, contract dump, worker scaling |
| `merge_qa.py` · `outliers.py` · `qa_launch.sh` | same dir | aggregation + the detached, niced pod launcher |
| The corpus itself | `tanitad-pod:` and `tanitad-pod3:/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d/` | **unchanged** (write audit above) |
| QA working copies | `tanitad-pod:/workspace/tmp/{v2_corpus_qa_scan.py,qa_lib/,qa_stack/,*.json,*.csv}`, same on `tanitad-pod3` | scratch; everything of value is staged in the repo |

## Escalation — two owners, not a note in a doc

1. **Sync `stack/` to both pods before any v2 launch** (`v2_dataset.py` missing,
   `train_flagship4b.py` has no `--v2-cache`, `refb_labels.py` stale on both).
   This is a file-copy, and the shadow-stack run proves it is sufficient.
   Recommend folding it into the launch runbook so it cannot be forgotten again.
2. **Consolidate the two shards onto one node** (pod3's shard is 10,660,387,873 B = 9.93 GiB; pod1 has
   79 GB free). Until then no run sees the balanced 28 % — pod1 alone is 27.35 %,
   pod3 alone 28.89 %.
3. *(Lower priority, separate owner)* **`episode_id` collides for 609 of 9,000
   clips.** Harmless for training, wrong for episode-clustered inference. Fix by
   hashing the full clip_id in `v2_compressed.build_compressed`.

**Evidence classes.** P1, P2, P3, the parity profile, the cross-build bit-identity,
the pod code state and the P4 throughput/contract are **MEASURED** (this scan;
artifacts above). The full-corpus window count (1,547,710) is **MEASURED** from
per-clip `T_out` via the exact index formula. Pixel-level f-theta fidelity is
**INHERITED** (`V2_PHASE2_BUILD.md` §3) and flagged as such.
