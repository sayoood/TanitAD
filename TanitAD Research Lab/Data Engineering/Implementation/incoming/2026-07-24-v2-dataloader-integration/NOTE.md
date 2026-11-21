# v2 compressed-cache dataloader integration (2026-07-24)

Closes the single wiring gap flagged in `V2_PHASE2_BUILD.md` (lines 119-124):

> A v2 training run must read episodes via `load_compressed` (a thin `Dataset`
> wrapper), decoding per `__getitem__`. ... wrap it in the dataloader ... wiring
> step before v2 training — flagged here so it is not discovered later.

The v2 corpus (`physicalai-v2bal-4b7eeeac222d`, ~9,000 clips / 50 h) is a
JPEG-compressed on-disk cache (~2.9 MB/clip; the *decoded* corpus is ~1 TB, so it
CANNOT be held in RAM the way the raw epcache path does). This delivers a lazy,
memory-bounded, **contract-identical** dataloader and wires it into the flagship
trainer behind a parity-safe flag.

## What ships

| file | canonical home | role |
|---|---|---|
| `v2_dataset.py` | `stack/tanitad/data/v2_dataset.py` | the lazy loader: `build_v2_providers`, `LazyV2Episode`, `_V2FramesProxy`, `V2CompressedCache`, `decode_full_episode` |
| `train_flagship4b.py` | `stack/scripts/train_flagship4b.py` | wired: adds `--v2-cache` / `--v2-lru`; **no flag == byte-identical to today** |
| `test_v2_dataset.py` | `stack/tests/test_v2_dataset.py` | unit test (6 cases): partial-decode byte-identity, window value-identity vs a materialised episode (v1+v2 labels), raw-schema match, LRU bound, manifest cache, flag threading |

## How it works (why it is contract-identical)

`build_v2_providers(cache_dir)` returns a `list[LazyV2Episode]` that is a drop-in
replacement for the raw episode list. Each provider quacks like a `ToyEpisode`:

- `.poses` / `.actions` — the small **float32** `[T,4]`/`[T,2]` tensors, read once
  per clip at index time via a **metadata-only `mmap` scan** that never pages in
  the JPEG buffer, then kept resident (whole-corpus poses+actions ~= 45 MB).
- `.frames` — a `_V2FramesProxy`: `.shape` is O(1); a **slice decodes ONLY the
  JPEGs that slice needs** (window+horizon ~= 30 of ~200 frames), stacking them
  exactly as `load_compressed` does (`stack_frames`, D-015). Compressed clip
  payloads live in a **bounded per-worker LRU** so RAM stays flat at any corpus
  size.

Because the providers feed the **UNCHANGED `FlagshipWindowDataset`** (via the
existing `_wrap`), every emitted window — keys, shapes, dtypes, nav/maneuver
labels, `pose_prev` — is produced by byte-for-byte the same code as the raw path.
`--v2-cache` changes ONLY the frame *source*. A per-clip metadata sidecar
`_v2manifest.pt` is written into the cache dir on first use (one-time scan;
instant on subsequent starts).

Faithfulness to the canonical `scripts/v2_compressed.load_compressed` (MEASURED
on 20 real `*.v2ep.pt`, eval pod, 2026-07-24): frames/poses/actions/episode_id
are **byte-identical**; partial decode of any window slice == the full-clip decode
sliced. The loader reuses `tanitad.data.comma2k19.stack_frames` +
`torchvision.io.decode_jpeg` and deliberately does NOT import `v2_compressed.py`
(whose *build* path pulls pandas/pyav), keeping the training + CI import surface
minimal.

## Launch command — v2 flagship on the consolidated cache

Consolidate the two build halves into ONE dir on the training pod (clip-id
filenames are unique, so `pod1:…/epcache-…/` ∪ `pod3:…/epcache-…/` merge cleanly
— `V2_PHASE2_BUILD.md` line 101), then:

```bash
cd /workspace/TanitAD/stack
PYTHONPATH=/workspace/TanitAD/stack python scripts/train_flagship4b.py \
  --v2-cache /workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d \
  --config flagship4b --v2 \
  --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint \
  --workers 8 --v2-lru 64 \
  --out /workspace/experiments/flagship-v2-30k
```

Notes:
- `--v2-cache` (data source) and `--v2` (model levers: ego->planners, speed-input,
  curvature-relative **v2 labels**, anchored tactical, rollout-k 12, ...) are
  independent and compose. The command above is the intended **flagship-v2** run
  (v2 corpus + v2 levers). Drop `--v2` to train the v1 model on the v2 corpus
  (data-only ablation); add neither and it is today's raw run untouched.
- Multiple cache dirs are allowed if you do NOT merge:
  `--v2-cache <dir_bottom> <dir_top>` (providers are concatenated; each dir keeps
  its own manifest + LRU).
- `--v2-lru N` bounds RAM at ~`workers * N * 2.9 MB` of compressed payloads.
  Default 64.
- First launch runs a one-time metadata-only manifest scan over all clips
  (`mmap`, never decodes JPEG; MEASURED sub-second for 20 clips; ESTIMATED
  ~1-2 min for 9,000, cached to `_v2manifest.pt`). Subsequent starts are instant.

## MEASURED (eval pod `tanitad-eval`, A40, 20 real clips, 2026-07-24)

- **Contract identity:** 20/20 clips byte-identical to `load_compressed`
  (frames `[T,9,256,256]` u8, poses/actions `f32`); v2 windows through the real
  `FlagshipWindowDataset` are **value-for-value identical** to windowing a
  materialised `ToyEpisode`, for BOTH v1 and v2 label regimes.
- **RAM bounded:** `lru=8` over 1,200 random fetches (3,420 windows): RSS
  1229 -> 1212 MB (**growth -18 MB, flat**), max LRU length 8/8. `lru=64`: max
  LRU 20 (capped by the 20-clip sample), RSS +18 MB. Resident RAM is
  `O(workers * lru_size * ~2.9 MB)`, **independent of corpus size** — the whole
  point.
- **Throughput** (CONTENDED eval pod, load ~24, shared — a pessimistic floor):
  workers=0 **16.9** win/s, workers=4 **29.5**, workers=8 **46.7** (2.92 batch/s
  @ bs16). The flagship step (~8.6 s for the 286 M 4-brain) consumes 64
  windows/step (bs16 x accum4) = **~1.4 s of data at 8 workers, fully hidden**
  under compute with prefetch. Each window decodes only ~32 of ~200 frames;
  per-window cost is dominated by the 2.9 MB payload `torch.load` on an LRU miss
  (an `mmap` byte-range decode is a future lever if data ever gates on the pod's
  MooseFS I/O).

See the deliverable manifest in the agent report for pod paths and the full
measurement log.
