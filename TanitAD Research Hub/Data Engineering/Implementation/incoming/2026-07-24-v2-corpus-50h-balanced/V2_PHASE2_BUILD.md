# v2 corpus — Phase 2 build (FIT-THE-QUOTA compressed cache)

**Date:** 2026-07-24 · **Pod:** `tanitad-pod` (pod1) · **Corpus:**
`physicalai-v2bal-4b7eeeac222d` (9,000 clips / 50.25 h, the Phase-1 selection).
**Storage option (Sayed):** *FIT THE QUOTA — no new disk.* Delivered by JPEG-
compressing the f-theta-cropped 256 px frames instead of storing raw uint8.

> **One-line status:** the compressed v2 cache is **building on TWO pods**
> (pod1 = clip-id bottom half, pod3 = top half; disjoint) and banking incrementally
> at **~26 clips/min combined → ETA ~5-8 h**, into
> `…/epcache-physicalai-v2bal-4b7eeeac222d/` on each. At a **MEASURED ~2.9 MB/
> episode** the full cache is **~25 GB** (vs 982 GB raw). Frames are **BIT-IDENTICAL**
> to the parity build pipeline (256 px, per-clip f-theta crop); only JPEG (q90)
> differs. `load_compressed()` returns a real `ToyEpisode(frames[T,9,256,256] u8,
> poses, maneuvers)` — verified. The two shards consolidate by clip-id when pod2
> frees (§5b).

---

## 1. Disk gate — pod1 is LOCAL NVMe, not MooseFS (MEASURED)

The brief warned of a ~466 GB MooseFS quota invisible to `df`. **On pod1 that does
not apply:** `/workspace` is a local block device `/dev/nvme1n1` (`mfsgetquota`
absent), so **`df` is authoritative here**: **500 GB total, 409 GB used, 92 GB
free** at start; a real `dd` wrote 3 GB at 2.6 GB/s. The existing 409 GB is the
parity build (`physicalai_phase0/_epcache` 260 GB + mp4s 34 GB) + cosmos (41 GB) —
**not touched.** The ~25 GB compressed cache + ~2 GB transient-per-chunk fits the
92 GB headroom comfortably. **GO** decision was gated on this + the measured
compressed size, not assumed.

## 2. Why it fits: compressed, un-stacked, crop-only-kept (MEASURED 2.9 MB/clip)

Raw epcache = **111.8 MB/episode** (`[199,9,256,256]` uint8) → 982 GB for 9,000.
The v2 cache stores the **same** f-theta-cropped 256 px frames but:
- **JPEG-encoded** (torchvision, q90) — ~15-25× smaller, near-lossless;
- **un-stacked** — the 3× D-015 channel-stacking is reproduced at load, not stored;
- **only the ~201 kept frames** (10 Hz), not all ~605 decoded.

Result **~2.9 MB/episode → ~25 GB** total. `load_compressed()` decodes the JPEGs,
re-applies `stack_frames` and `maneuvers_for_poses`, and returns the identical
`ToyEpisode` contract the trainer already consumes.

## 3. Faithfulness — frames are BIT-IDENTICAL to the parity pipeline (MEASURED)

The build reuses `signals_at` (poses), `intrinsics_for_clip` + `ftheta_crop_resize`
(the exact per-clip fisheye crop, `ftheta_v2` calibration), and `maneuver_labels`
verbatim. The one optimization — cropping **only the kept frames** instead of all
605 then subsampling — is provably output-identical (per-frame crop independence)
and was **verified `torch.equal == True`** against `_decode_mp4(...)[frame_idx]` on
a real clip. The **only** deviation from a raw build is JPEG lossiness on pixels
(q90). Verified load on a freshly-built episode: `frames (199,9,256,256) uint8,
poses (199,4), maneuvers (199,)`, maneuver classes present, pixel range [14,255].

## 4. Build pipeline (`v2_compressed.py build`, sharded, resumable)

Per chunk, per worker: **ensure egomotion** (fetch if missing) → **download the
camera chunk zip** (curl, resumable, stall-abort `--speed-limit`) → **extract only
selected clips' mp4+timestamps** → **delete the zip** → **decode+crop+JPEG each
clip** (atomic `.pt` write) → **delete the mp4s**. Peak transient disk ≈ 1 zip
(~1.3 GB) + 1 chunk's mp4s (~0.6 GB). Banks per clip; **resumable** (skips built
clip_ids), so a kill yields the fetched+cached portion. 200 camera chunks total
(~242 GB transient download, deleted as it goes).

## 5. Parallelism + the traps hit and fixed (MEASURED, honest)

Tuned to pod1's **128 cores but ~57.7 GB cgroup RAM cap** (the cap that OOM-killed
the flagship here before). The journey, logged because it cost real time:

| Symptom | Root cause | Fix |
|---|---|---|
| worker OOM-killed at K=12 (54 GB RSS) | ~4.9 GB/worker × 12 > 57.7 GB cap | drop workers |
| K=7 peaked **52.4 GB (91 %)** | raw decode held all 605 cropped frames | **K=5** + CUDA off |
| ~1-2 clips/min, ETA 180 h | f-theta crop of **all 605** frames dominated + contended | **crop only ~201 kept → 4.84× faster** |
| curl stalls at 51 KB/s, 10 h ETA | no stall timeout | `--speed-limit 1MB/s --speed-time 25` + retry |

Final config: **K=5 workers**, `V2_TORCH_THREADS=20` (5×20<128, no oversubscribe),
`CUDA_VISIBLE_DEVICES=""`, decode-batch 16. **Steady state: 26 clips/min, peak RSS
well under the cap, load ~62/128.** ETA ~5.6 h. (`df` is fine here; the OOM cap was
verified by the kill itself + per-process RSS, not `df`.)

## 5b. Two-pod parallelization (2026-07-24, for runway margin)

pod1-solo sustained ~220 clips/h → ~40 h ETA, tight vs the ~2-day runway before
pod2 frees. **pod3 (idle A40) was added on a clip-id-DISJOINT shard:**

- **Split point (deterministic):** `r0_selection_v2` sorted by `clip_id`, index
  4500. **pod1 → bottom half** (0-4499, `sel_bottom.parquet`); **pod3 → top half**
  (4500-8999). Overlap = **0** (verified). pod1 had already built 453 top-half
  clips before the split — pod3 builds `sel_top` **minus** those (`sel_top_remaining`,
  4047 clips) so nothing is double-built; the 453 orphans live in pod1's cache and
  consolidate by clip-id.
- **pod3 config:** `/workspace/venv/bin/python` (torch 2.8 — has av/torchvision;
  `/usr/bin/python3` lacks av/cv2/pandas), MooseFS `/workspace` (13 GB `dd`
  verified; shard ≈12 GB), HF token from `Keys.txt`, K=5 tuned to pod3's **~46.6 GB
  cgroup cap** (lower than pod1's 57.7). Same `v2_compressed` JPEG format, same OUT
  basename → clean consolidation. Verified: 5/5 alive, RSS 5.2 GB/46.6, 0 errors.
- **Revised combined ETA:** ~26 clips/min sustained across both pods → **~5-8 h**
  (well within the runway; instantaneous bursts hit ~38/min). Both shards are
  resumable/clip-id-keyed and bank incrementally.
- **Consolidation:** when pod2 frees (or via HF relay), merge
  `pod1:…/epcache-…/` ∪ `pod3:…/epcache-…/` into one dir — clip-id filenames make
  it a plain union (any duplicate is byte-for-byte the same episode).

## 6. Status & how to finish / monitor

The build runs **detached** (setsid+nohup, survives ssh logout); PIDs in
`…/logs/worker_pids.txt`. It will complete in ~6 h and each worker exits when its
40 chunks are done.

- **Progress:** `ls …/epcache-physicalai-v2bal-4b7eeeac222d/*.v2ep.pt | wc -l` (target 9000)
- **Done when:** count ≈ 9000 and all `worker_pids` exited; `logs/worker_*.log` end in `DONE built=…`.
- **Relaunch if needed** (resumable): `bash /workspace/data/physicalai_v2/build_v2_launch.sh`.
- **Kill** (if ever): by explicit PID from `worker_pids.txt` — **never `pkill -f`** (self-matches).
- A handful of clips may fail (corrupt mp4) → logged `FAILED`; expect ≥ ~8,990/9,000.

## 7. INTEGRATION REQUIRED (escalation) — the v2 trainer needs the compressed loader

The v2 cache is a **new on-disk format**: JPEG dicts loaded by
`v2_compressed.load_compressed(path) -> ToyEpisode`, **not** the raw
`load_episode()` the current epcache uses. **A v2 training run must read episodes
via `load_compressed` (a thin `Dataset` wrapper), decoding per `__getitem__`.**
Decode cost is ~1-2 ms/frame; for training throughput, wrap it in the dataloader
workers (as the existing pipeline already does for windowing). This is the one
wiring step before v2 training — flagged here so it is not discovered later.

## 8. Deliverable manifest

| Artifact | Location | Notes |
|---|---|---|
| v2 compressed cache (~25 GB, building) | `tanitad-pod:/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d/` | the corpus; lives on the pod (training data), fully reproducible from the staged scripts + selection |
| `v2_compressed.py` | `repo:…/2026-07-24-v2-corpus-50h-balanced/` + `tanitad-pod:/workspace/TanitAD/stack/scripts/` | build + `load_compressed`; the loader the v2 trainer imports |
| `build_v2_launch.sh` | same repo dir + `tanitad-pod:/workspace/data/physicalai_v2/` | K=5 detached launcher (tuned for the 57.7 GB cap) |
| `V2_PHASE2_BUILD.md` (this file) | same repo dir | build report + manifest |
| build logs | `tanitad-pod:/workspace/data/physicalai_v2/logs/worker_*.log` | per-worker progress; not staged (pod-side ops logs) |

**Evidence classes:** disk headroom, compressed size, bit-identity, throughput,
per-worker RSS are **MEASURED** (pod1, artifacts above). ETA (~5.6 h) and final
cache size (~25 GB) are **ESTIMATED** from the measured rate/size. Storage choice
follows Sayed's "fit the quota" ruling.
