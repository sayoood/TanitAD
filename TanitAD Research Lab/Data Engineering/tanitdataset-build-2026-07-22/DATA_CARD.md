---
license: other
task_categories: [robotics]
tags: [autonomous-driving, world-model, tanitad, ego-driving, camera]
---

# Sayood/TanitDataSet-C — TanitDataSet-C (commercial-clean tier)

The **commercially-clean, HF-publishable** tier of TanitDataSet: a camera-first
autonomous-driving corpus under one schema. Every record is `owned-safe` and
`commercial_ok` (permissive license, **no** share-alike, **no** gated/firewalled
or `refuse`-class source) — the tier is a per-record stamp derived structurally
from a per-source license CONSTANT, never inferred, and a hard export guard
refuses egress if a single row falls outside that scope.

**Episodes:** 90  (train: 72, val: 18)

## Sources & licenses

| source | license | class | episodes |
|---|---|---|---|
| `comma2k19` | MIT | `owned-safe` | 90 |

## Record schema (the world-model contract, D-015/D-016)

Each episode is the byte-identical contract every TanitAD adapter emits:

- `frames`  — `uint8 [T, 9, 256, 256]` — a 3-frame RGB stack (9 = 3×RGB),
  canonicalized to `f_eff ≈ 266 px` (D-016 geometry canon).
- `actions` — `f32 [T, 2]` — `(steer, accel)`, the action applied between t, t+1.
- `poses`   — `f32 [T, 4]` — `(x, y, yaw, v)` ego trajectory.
- per-episode metadata: `source`, `license_*`, `commercial_ok`, `sha256` of the
  frame blob, `build_params_hash`, native intrinsics, modality flags.

Stored as WebDataset tar shards (`{id}.frames.npy` / `.motion.npz` /
`.meta.json`); the reader re-verifies each frame blob's `sha256` on load
(verify-without-rebuild), so a rotted shard fails loudly instead of training on
garbage.

## Provenance & reproducibility

Every episode carries a `sha256` of its exact frame bytes and a
`build_params_hash` — a consumer can verify a shard member without rebuilding it,
and the build recipe travels with the data (`MANIFEST.json` / `NOTICE`).

## Notes for consumers

- **Split granularity.** Where a source's route/clip id is preserved the split is
  route-disjoint; for caches where only the episode survives, the split is
  **episode-level** — rebuild from origin if you need strictly route-disjoint
  train/val.
- **Near-duplicates & multi-traversal.** Records are NOT dropped for perceptual
  similarity. Homogeneous highway sources (e.g. comma2k19 — one commute route
  re-driven) contain wanted **multi-traversal** repeats; use the catalog's
  curation weights to control sampling frequency rather than deleting records.
- **Anonymization.** Real camera footage may contain faces/plates. Sources here
  are already publicly distributed under their stated licenses; still, run a
  face/plate check appropriate to your jurisdiction before any re-distribution.

_Staged by the TanitAD Phase-A HF exporter. Attribution/NOTICE ships alongside._
