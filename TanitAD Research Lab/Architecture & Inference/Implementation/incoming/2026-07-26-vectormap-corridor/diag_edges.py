#!/usr/bin/env python3
"""Diagnostic: why do the ring test and the edge-profile bounds disagree?

Two independent constructions of "inside the lane" disagree on 12.3 % of ego
steps (``corridor_verdict.json`` cross-check). One of them is wrong. This
isolates which, by testing the assumption the edge-profile construction rests on:
**that a lane's ``left_edge`` / ``right_edge`` bracket its own centreline.**

If ``lat_left(s) > 0 > lat_right(s)`` fails, the edges are not that lane's edges
(most likely they are shared ROAD boundaries spanning several lanes), and every
``d_left`` / ``d_right`` derived from them is void.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vectormap_corridor import SSROOT, edge_profile, frenet  # noqa: E402

SUSPECT = ["00097de1", "27068a85", "03376794", "1c9774e9", "001564ce",
           "00040136", "000548db", "bfb44da0"]


def main():
    from alpasim_runtime.scene_loader import ArtifactSceneProvider
    from trajdata.maps.vec_map_elements import MapElementType
    scenes = {}
    for sd in sorted(d for d in glob.glob(SSROOT + "/*") if os.path.isdir(d)):
        try:
            prov = ArtifactSceneProvider.from_path(sd, smooth_trajectories=True)
        except Exception:
            continue
        for sid in sorted(prov.scene_ids):
            scenes.setdefault(sid, prov)

    out = {}
    for sid in sorted(scenes):
        short = sid[7:15] if sid.startswith("clipgt-") else sid[:8]
        if short not in SUSPECT:
            continue
        ds = scenes[sid].get_data_source(sid)
        lanes = ds.map.elements[MapElementType.ROAD_LANE]
        n_ok = n_bad = 0
        widths, npts = [], []
        bad_examples = []
        for k, L in lanes.items():
            c = np.asarray(L.center.points, dtype=np.float64)[:, :2]
            le, re_ = getattr(L, "left_edge", None), getattr(L, "right_edge", None)
            if le is None or re_ is None or c.shape[0] < 2:
                continue
            le = np.asarray(le.points, dtype=np.float64)[:, :2]
            re_ = np.asarray(re_.points, dtype=np.float64)[:, :2]
            if le.shape[0] < 2 or re_.shape[0] < 2:
                continue
            sc, fl, fr = edge_profile(c, le, re_)
            npts.append((len(c), len(le), len(re_)))
            # THE ASSUMPTION: left edge is to the LEFT (+) and right to the RIGHT (-)
            brackets = float(((fl > 0) & (fr < 0)).mean())
            w = float(np.median(fl - fr))
            widths.append(w)
            if brackets > 0.95:
                n_ok += 1
            else:
                n_bad += 1
                if len(bad_examples) < 3:
                    bad_examples.append({
                        "lane": str(k), "frac_bracketed": round(brackets, 3),
                        "median_lat_left": round(float(np.median(fl)), 3),
                        "median_lat_right": round(float(np.median(fr)), 3),
                        "median_width": round(w, 3),
                        "n_centre_pts": len(c), "n_left_pts": len(le),
                        "n_right_pts": len(re_)})
        out[short] = {
            "n_lanes_edges_bracket_centreline": n_ok,
            "n_lanes_edges_DO_NOT_bracket": n_bad,
            "frac_bad": round(n_bad / max(n_ok + n_bad, 1), 4),
            "median_width_m": round(float(np.median(widths)), 3) if widths else None,
            "p90_width_m": round(float(np.percentile(widths, 90)), 3) if widths else None,
            "median_pts_centre_left_right": [
                int(np.median([x[0] for x in npts])),
                int(np.median([x[1] for x in npts])),
                int(np.median([x[2] for x in npts]))] if npts else None,
            "bad_examples": bad_examples}
        print(short, json.dumps(out[short])[:400], flush=True)
        try:
            ds.clear_cache()
        except Exception:
            pass

    json.dump(out, open("/workspace/diag_edges.json", "w"), indent=1)
    print("WROTE /workspace/diag_edges.json")


if __name__ == "__main__":
    main()
