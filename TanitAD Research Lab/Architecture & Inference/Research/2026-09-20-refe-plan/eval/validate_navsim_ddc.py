#!/usr/bin/env python3
"""Does refe/navsim_lane.py reproduce NAVSIM's OWN driving-direction verdict? (gate for the SFT-2 labeller)

For N navtest tokens of raw/2026-10-04-lane-discipline/lane_census.jsonl (whose `ddc` column is NAVSIM v1.1's scorer,
run on the token's metric cache for PDM-Closed + the 64 hypotheses + the human), re-simulate the same 66 trajectories with
the LABELLER's path (navsim_dac.simulated_states, the vendored LQR + bicycle) and compute DDC with navsim_lane:
  * on-route lanes from the metric cache's own `route_lane_ids` (NAVSIM's), polygons from the map API by id;
  * a second read with the lanes REBUILT from roadblocks (navsim_lane.route_lane_polygons over the roadblocks of those
    lanes) -- what the navtrain labeller does with the corrected scenario route.
Prints the confusion matrices and ZZNAVSIMDDC_AGREE <pct_cache_lanes> <pct_rebuilt>.
    python eval/validate_navsim_ddc.py [--n 300]      (re-launches itself in the navsim v1.1 venv)
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import lzma
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DRV = os.environ.get("REFE_DRIVE", "E:")
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = f"{DRV}/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
CACHE = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
MAPS = f"{DRV}/Archive/devbox-C/navsim/data/maps"
CENSUS = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "lane_census.jsonl")
PROPS = f"{DRV}/Projects/TanitAD/data/refe_navtest/proptable/navtest_final/proposals.npz"


def main() -> int:
    if os.environ.get("REFE_DDCV_CHILD") != "1":
        env = dict(os.environ, REFE_DDCV_CHILD="1", PYTHONPATH=f"{V11};{os.path.join(PKG, 'refe')}",
                   PYTHONIOENCODING="utf-8", NUPLAN_MAPS_ROOT=MAPS, NUPLAN_MAP_VERSION="nuplan-maps-v1.0")
        return subprocess.call([NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    import navsim_dac as ND
    import navsim_lane as NL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    P = np.load(PROPS)
    idx = {str(t): i for i, t in enumerate(P["token"])}
    rows = []
    for line in open(CENSUS, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
        if len(rows) >= a.n:
            break
    rows.sort(key=lambda r: E[r["token"]]["map_name"])
    maps, conf_a, conf_b = {}, collections.Counter(), collections.Counter()
    bad = []
    for k, r in enumerate(rows):
        tok = r["token"]
        mn = E[tok]["map_name"]
        if mn not in maps:
            maps.clear()
            maps[mn] = get_maps_api(MAPS, "nuplan-maps-v1.0", mn)
        mp = maps[mn]
        mc = pickle.load(lzma.open(f"{CACHE}/{r['log']}/unknown/{tok}/metric_cache.pkl", "rb"))
        ego = mc.ego_state
        props = P["proposals"][idx[tok]].astype(np.float64)
        hum = np.asarray(E[tok]["human_future_poses"], np.float64)
        trajs = np.concatenate([props, hum[None]], 0)                         # 64 hypotheses + human (PDM-Closed skipped:
        sim = ND.simulated_states(trajs, ego)                                 # it is not an 8-pose plan)
        cen = NL.centres_from_states(sim)
        flat = cen.reshape(-1, 2)

        def ddc_with(polys):
            on = np.zeros(len(flat), bool)
            for poly in polys:
                on |= shapely.contains_xy(poly, flat[:, 0], flat[:, 1])
            return NL.ddc_from_centres(cen, on.reshape(cen.shape[:2]))

        lanes = []
        rb = set()
        for lid in mc.route_lane_ids:
            o = mp.get_map_object(str(lid), L.LANE) or mp.get_map_object(str(lid), L.LANE_CONNECTOR)
            if o is not None:
                lanes.append(o.polygon)
                rb.add(o.get_roadblock_id())
        d_a = ddc_with(lanes)
        d_b = ddc_with(NL.route_lane_polygons(mp, sorted(rb)))
        ref = np.array(r["ddc"][1:66], dtype=np.float64)                       # census order: 1..64 hypotheses, 65 human
        for x, y in zip(ref, d_a):
            conf_a[(x, y)] += 1
        for x, y in zip(ref, d_b):
            conf_b[(x, y)] += 1
        if np.any(ref != d_a):
            bad.append({"token": tok, "n_disagree": int((ref != d_a).sum())})
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(rows)}", flush=True)
    tot = sum(conf_a.values())
    pa = 100.0 * sum(v for (x, y), v in conf_a.items() if x == y) / tot
    pb = 100.0 * sum(v for (x, y), v in conf_b.items() if x == y) / tot
    out = {"tokens": len(rows), "trajectories": tot,
           "confusion_cache_lanes": {f"navsim={x} ours={y}": v for (x, y), v in sorted(conf_a.items())},
           "confusion_rebuilt_lanes": {f"navsim={x} ours={y}": v for (x, y), v in sorted(conf_b.items())},
           "agree_pct_cache_lanes": round(pa, 3), "agree_pct_rebuilt_lanes": round(pb, 3), "disagreeing_tokens": bad[:40]}
    dst = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "validate_navsim_ddc.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreeing_tokens"}, indent=1))
    print(f"ZZNAVSIMDDC_AGREE {pa:.2f} {pb:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
