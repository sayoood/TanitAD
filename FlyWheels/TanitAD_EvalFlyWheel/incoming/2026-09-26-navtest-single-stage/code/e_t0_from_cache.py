#!/usr/bin/env python3
"""(NAVSIM VENV) E-T0: the tokens whose metric cache holds an agent box intersecting the ego footprint at t0.

    <navsim venv python> code/e_t0_from_cache.py --cache <metric cache dir> --out raw/e_t0_<name>.json

W8, 2026-09-26 — the ONLY exception set of PREREG.md §3 (C-NC). Computed from the CACHE CONTENT, never
from a scorer output, with the SAME expression the v2 non-reactive policy uses to remove those agents
(``navsim/traffic_agents_policies/log_replay_traffic_agents.py:35-46`` @0a380a9):

    ego_box = metric_cache.ego_state.car_footprint.oriented_box.geometry
    detections_tracks = metric_cache.observation.detections_tracks[:num_poses]
    colliding = {a.metadata.track_token for a in detections_tracks[0].tracked_objects.tracked_objects
                 if a.box.geometry.intersects(ego_box)}

v1.1 keeps those agents in its observation with ``collided_track_ids = []`` (v1 pdm_observation.py:258),
so on these tokens the v2 NC may only be >= the v1 NC; everywhere else PREREG requires equality.
"""
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    t0 = time.time()
    paths = sorted(a.cache.glob("*/*/*/metric_cache.pkl"))
    et0, n, fails, n_objs0 = {}, 0, [], 0
    for p in paths:
        tok = p.parts[-2]
        try:
            mc = pickle.loads(lzma.decompress(p.read_bytes()))
            ego_box = mc.ego_state.car_footprint.oriented_box.geometry
            first = mc.observation.detections_tracks[0]
            objs = first.tracked_objects.tracked_objects
            n_objs0 += len(objs)
            hit = sorted({o.metadata.track_token for o in objs if o.box.geometry.intersects(ego_box)})
            if hit:
                et0[tok] = {"log": p.parts[-4], "n_agents_at_t0_intersecting_ego": len(hit),
                            "types": sorted({str(o.tracked_object_type) for o in objs
                                             if o.metadata.track_token in hit})}
            n += 1
        except Exception as e:                                           # noqa: BLE001
            fails.append({"file": str(p), "exception": f"{type(e).__name__}: {e}"[:200]})
    rec = {"schema": "w8-e-t0/1", "cache": str(a.cache).replace("\\", "/"), "n_entries": len(paths), "n_read": n,
           "n_read_failures": len(fails), "read_failures": fails[:20], "n_objects_at_t0_total": n_objs0,
           "n_E_T0": len(et0), "E_T0": et0,
           "expression": "log_replay_traffic_agents.py:35-46 @0a380a9 (detections_tracks[0] vs ego footprint)",
           "wall_s": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("E_T0", "read_failures")}))
    return 0 if (n == len(paths) and n > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
