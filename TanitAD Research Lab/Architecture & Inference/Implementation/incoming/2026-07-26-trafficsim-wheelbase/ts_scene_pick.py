#!/usr/bin/env python3
"""Pick the highest-traffic scenes for the full-runtime reactivity test.

Reads `clipgt/obstacle.parquet` straight out of each USDZ archive (a zip) with pyarrow --
the same raw-archive path gate 1's probe B used, so it does not depend on trajdata or on any
AlpaSim service. Counts DYNAMIC agents (those whose logged track actually moves), because a
reaction test on parked cars is powerless: gate 2's v1 harness diluted 2-3 interacting agents
into 75 objects of which 62 were parked, and that defect is what its v2 fixed.

Output: artifacts/ts_scene_pick.json  (per scene: n_objects, n_dynamic, span, path length)
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pyarrow.parquet as pq

USDZ_DIR = "/workspace/alpa-invest/alpasim/data/nre-artifacts/all-usdzs"
OUT = "/workspace/ts_scene_pick.json"
MOVE_TH_M = 2.0  # a logged track must move at least this far to count as dynamic


def scene_stats(path: str) -> dict:
    rec: dict = {"scene": os.path.basename(path).replace(".usdz", "")}
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("obstacle.parquet")]
            if not names:
                rec["error"] = "no obstacle.parquet"
                return rec
            tbl = pq.read_table(io.BytesIO(z.read(names[0])))
            rec["columns"] = sorted(tbl.column_names)
            # MEASURED schema (2026-07-26): key:struct<clip_id,timestamp_micros,
            # label_class_id,label_id>, obstacle:struct<trackline_id,center{x,y,z},
            # size,orientation,category,...>, version:uint64.
            d = tbl.to_pydict()
            keys, obs = d["key"], d["obstacle"]
            ids = np.array([o["trackline_id"] for o in obs])
            xs = np.array([o["center"]["x"] for o in obs], dtype=float)
            ys = np.array([o["center"]["y"] for o in obs], dtype=float)
            ts_all = np.array([k["timestamp_micros"] for k in keys], dtype=float)
            cats = np.array([o["category"] for o in obs])
            rec["categories"] = {c: int((cats == c).sum()) for c in np.unique(cats)}
            uniq = np.unique(ids)
            n_dyn = n_long = 0
            spans = []
            for u in uniq:
                m = ids == u
                if m.sum() < 2:
                    continue
                px, py, pt = xs[m], ys[m], ts_all[m]
                span = float(np.hypot(px.max() - px.min(), py.max() - py.min()))
                spans.append(span)
                if span >= MOVE_TH_M:
                    n_dyn += 1
                    # the runtime drops actors present < min_traffic_duration_us (3 s)
                    if (pt.max() - pt.min()) >= 3_000_000:
                        n_long += 1
            rec["n_objects"] = int(len(uniq))
            rec["n_dynamic"] = int(n_dyn)
            rec["n_dynamic_ge3s"] = int(n_long)
            rec["n_rows"] = int(tbl.num_rows)
            rec["median_span_m"] = round(float(np.median(spans)), 2) if spans else None
            rec["max_span_m"] = round(float(np.max(spans)), 2) if spans else None
            rec["duration_s"] = round(float((ts_all.max() - ts_all.min()) / 1e6), 2)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> None:
    files = sorted(f for f in os.listdir(USDZ_DIR) if f.endswith(".usdz"))
    out = []
    for i, f in enumerate(files):
        r = scene_stats(os.path.join(USDZ_DIR, f))
        out.append(r)
        print(f"[{i+1}/{len(files)}] {r.get('scene')[:8]} "
              f"obj={r.get('n_objects')} dyn={r.get('n_dynamic')} {r.get('error','')}",
              flush=True)
    ok = [r for r in out if r.get("n_dynamic") is not None]
    ok.sort(key=lambda r: -r["n_dynamic_ge3s"])
    payload = {
        "usdz_dir": USDZ_DIR,
        "move_threshold_m": MOVE_TH_M,
        "n_scenes": len(files),
        "n_parsed": len(ok),
        "top10_by_dynamic": ok[:10],
        "all": out,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("\n=== TOP 10 by dynamic agents ===")
    for r in ok[:10]:
        print(f"  {r['scene']}  dyn={r["n_dynamic"]:3d} dyn3s={r["n_dynamic_ge3s"]:3d} obj={r["n_objects"]:3d} "
              f"dur={r.get('duration_s')}s")
    print("WROTE", OUT)


if __name__ == "__main__":
    sys.exit(main())
