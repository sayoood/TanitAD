"""What the slice COSTS on the path training actually uses — measured honestly.

⚠️ WHY THIS EXISTS. A first pass timed ``load_compressed(parent)`` then
``load_compressed(sub)`` on the same file and read the sub as 22 % FASTER. That
number is CONFOUNDED: the second call runs against a warm page cache. Anything
that loads the parent first and the sub second measures the file system, not the
slice.

So: interleave the arms, alternate which goes first, discard a warm-up rep, and
report both the median and the spread. The path measured is
``V2CompressedCache.decode_stacked_range`` over a WINDOW-SIZED range — what the
DataLoader worker calls — not a whole-clip convenience loader.

Usage:
  PYTHONPATH=<stack> python3 time_window_decode.py --cache <dir> --n 6 \
      --reps 5 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stack", default=os.environ.get(
        "TANITAD_STACK", "/workspace/TanitAD/stack"))
    ap.add_argument("--n", type=int, default=6, help="clips")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--window", type=int, default=24,
                    help="stacked rows per fetch (window+horizon sized)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    import tanitad.data.v2_dataset as V2
    from tanitad.data.calib import (PHYSICALAI_RIG_CLEAN_128x576,
                                    PHYSICALAI_RIG_CLEAN_176x624)

    files = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(a.cache, "*.v2ep.pt")))[:a.n]
    assert files, a.cache
    arms = {"parent_256x640": None,
            "sub_176x624": PHYSICALAI_RIG_CLEAN_176x624,
            "sub_128x576": PHYSICALAI_RIG_CLEAN_128x576}
    caches = {}
    for name, fr in arms.items():
        c = V2.V2CompressedCache(a.cache, lru_size=1, frame=fr)
        c.files = files
        caches[name] = c

    # warm the page cache for every file, equally, before any timing
    for i in range(len(files)):
        caches["parent_256x640"].decode_stacked_range(i, 0, 2)
    for c in caches.values():
        c._lru = None                       # cold LRU, warm page cache

    samples: dict[str, list[float]] = {k: [] for k in arms}
    shapes: dict[str, list] = {}
    order = list(arms)
    for rep in range(a.reps + 1):           # rep 0 is discarded
        order = order[1:] + order[:1]       # rotate: no arm is always first
        for name in order:
            c = caches[name]
            c._lru = None                   # every rep pays a fresh payload load
            t0 = time.time()
            for i in range(len(files)):
                out = c.decode_stacked_range(i, 0, a.window)
            dt = time.time() - t0
            if rep:
                samples[name].append(dt / len(files))
            shapes[name] = list(out.shape)

    res = {"cache": a.cache, "n_clips": len(files), "reps": a.reps,
           "window_rows": a.window, "shapes": shapes,
           "note": ("per-clip seconds for one payload load + a decode of "
                    f"{a.window}+2 raw frames; arms INTERLEAVED and rotated, "
                    "page cache pre-warmed for all, one rep discarded."),
           "arms": {}}
    base = st.median(samples["parent_256x640"])
    for name, xs in samples.items():
        res["arms"][name] = {
            "median_s_per_clip": round(st.median(xs), 4),
            "min_s": round(min(xs), 4), "max_s": round(max(xs), 4),
            "ratio_vs_parent": round(st.median(xs) / base, 4),
            "samples": [round(x, 4) for x in xs]}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
    print(json.dumps(res["arms"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
