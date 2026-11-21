# 408 × 1024 validated on the 139 eval clips — 16/16, the geometry the PI chose

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: n/a** (a data-and-geometry validation)

## The ruling this verifies

> *"You can take 408x1024 …"* — PI, 2026-09-17

`408 × 1024` supersedes `256 × 1024` for every future training. This is the same real-data suite
that validated 256 × 1024 (`ec56eb9`), pointed at the 408 cache.

## Result — 16 passed, 0 failed, 5:29

`TANITAD_V2EP_DIR` → `v2ep-eval139-408x1024cyl` · `TANITAD_ORIENT_CLIPS=0` (score **every** scorable
clip, which is what makes the orientation guard powered rather than convenient).

Every check that passed at 256 × 1024 passes at 408 × 1024, including the four that matter most:

| check | at 408 × 1024 |
|---|---|
| **orientation guard, mirror goes red** | PASS |
| **the guard itself fails when the lift is mirrored** | PASS (mutation) |
| the whole map branch runs on real frames and the loss reaches the trunk | PASS |
| the 3-D join reaches the height targets · covers every agent-frame it ships | PASS · PASS |
| a map renamed to another clip is refused · an off-by-`n_stack` goes red · a 2-D obstacle table is refused | PASS (3 mutations) |

⭐ **The orientation pair is the one that had to be re-run.** Changing the image height moves what
each feature cell subtends, and the lift's azimuth mapping is the thing a geometry change could
silently break. The guard passes on the real lift **and goes RED on the mirrored one**, at the new
geometry, over every scorable clip.

⭐ **Nothing in the code changed to make this work.** The geometry came from the payload
(`stored_frame_of`) and the channel counts from `timm.feature_info`, which is what the
mutation-proven `test_refcv6_geometry_agnostic.py` exists to enforce. A geometry switch the PI can
make in one sentence is the return on that rule.

## What the ruling buys and costs, MEASURED

| | 256 × 1024 | **408 × 1024** |
|---|---|---|
| VFOV | 29.341° | **45.296°** (99.6 % of today's) |
| **nearest visible road ahead** | 5.02 m | **3.15 m** |
| corpus cache (4,713 clips) | 273.6 GB | **386.5 GB** (+112.9) |
| stride-16 / stride-32 tokens | 1024 / 256 | **1664 / 416** (1.63×) |
| resnet101 K=3 activations | 1033.5 MB | **1662.9 MB** |

⇒ the **1.88 m blind strip is not taken**.

## ⛔ What this does NOT validate

The **data and geometry** path only. `lift_orientation.lift_image_features` average-pools the image
by stride as a trunk stand-in, so **no timm trunk, no tactical layer and no planner ran here** — the
same scope limit the 256 × 1024 note carries.

⚠️ **Two obligations that follow the ruling and are NOT discharged by this note:**
1. ⛔ **HF quota.** +112.9 GB against a hard ceiling. Check **before** the corpus rebuild is pushed.
2. ⚠️ **Token count rises 1.63× at both strides**, so every per-step cost measured at 256 × 1024 is
   now low. **Re-measure rather than scale** — including the conflict detector's overhead, which was
   +71.7 % to +104.2 % on the smaller grid.

## Files

| path | what |
|---|---|
| `raw/pytest_realdata_408x1024.log` | the full run, verbatim |
