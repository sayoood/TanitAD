#!/usr/bin/env python3
"""LANE-1 instrument fix (declared departure, see PREREG_LANE1.md §"Departure"): lane keeping measured against the map's
LANE CENTRELINES, because both registered lane proxies failed their controls on the first 1,180 tokens:
  * NAVSIM's MULTIPLE_LANES footprint test (strad_s >= 1 s) fires on the HUMAN 25 % and PDM-Closed 37 % -- a 2.3 m-wide
    footprint in a ~3 m nuPlan lane polygon straddles under normal driving;
  * distance to the metric cache's single ROUTE centreline reads > 1 m on 15 % of PDM-Closed drives even where the human
    keeps that lane (PDM-Closed's own +-1 m offset proposals, and parallel lanes).
WHAT. Every trajectory of lane_census.py (0 = PDM-Closed, 1..64 hypotheses, 65 = human) is re-simulated identically
(NAVSIM v1.1 PDMSimulator, 41 steps at 0.1 s), and for steps 1..40 the vehicle CENTRE is located on the nuPlan map:
  * lane step   the centre lies in >= 1 LANE polygon and in no INTERSECTION polygon (lane connectors in junctions excluded)
  * d           distance from the centre to the BASELINE (centreline) of the containing lane (min over containing lanes)
  * lk10_s      longest run (s) of lane steps with d > 1.0 m   -- with a ~3.5 m lane and a 2.3 m car the wheels are on or
                over the marking; a lane CHANGE also produces a run (~0.5-1 s), hence the human/PDM floors below
  * lk15_s      the same with d > 1.5 m (half the car in the neighbouring lane)
  * offl_n      steps whose centre is in no lane, no lane connector and no intersection
Thresholds are fixed here, before any hypothesis-level lane-keeping number is read; the floors come from the human and
PDM-Closed, which are reported beside every model number.
    python lane_census2.py [--workers 3]      (re-launches itself in the navsim v1.1 venv)
"""
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import pickle
import random
import subprocess
import sys
import time
import traceback

import numpy as np

import lane_census as LC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lane_census2.jsonl")
MAPS = f"{LC.DRV}/Archive/devbox-C/navsim/data/maps"
RADIUS = 90.0
_W = {}


def runs(mask_row):
    return LC.longest_run(mask_row) * 0.1


def _init():
    LC._init()
    _W.update(LC._W)
    _W["maps"] = {}


def get_map(name):
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    m = _W["maps"]
    if name not in m:
        m.clear()                                    # tokens arrive grouped by city: hold one map per worker
        m[name] = get_maps_api(MAPS, "nuplan-maps-v1.0", name)
    return m[name]


def lane_token(args):
    tok, log, map_name, hum = args
    try:
        import shapely
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.abstract_map import SemanticMapLayer as L
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
        if not _W:
            _init()
        i = _W["idx"][tok]
        props = _W["P"]["proposals"][i].astype(np.float64)
        hum = np.asarray(hum, dtype=np.float64)
        mc = pickle.load(lzma.open(f"{LC.CACHE}/{log}/unknown/{tok}/metric_cache.pkl", "rb"))
        ego = mc.ego_state
        samp8 = TrajectorySampling(num_poses=8, interval_length=0.5)
        states = [get_trajectory_as_array(mc.trajectory, _W["samp"], ego.time_point)]
        for traj in list(props) + [hum]:
            it = transform_trajectory(Trajectory(traj.astype(np.float32), samp8), ego)
            states.append(get_trajectory_as_array(it, _W["samp"], ego.time_point))
        sim = _W["sim"].simulate_proposals(np.stack(states, 0), ego)              # (66, 41, S)
        r2c = ego.car_footprint.vehicle_parameters.rear_axle_to_center
        h = sim[:, 1:, StateIndex.HEADING]
        cx = sim[:, 1:, StateIndex.X] + r2c * np.cos(h)
        cy = sim[:, 1:, StateIndex.Y] + r2c * np.sin(h)
        n, T = cx.shape
        X, Y = cx.ravel(), cy.ravel()
        mp = get_map(map_name)
        objs = mp.get_proximal_map_objects(Point2D(ego.center.x, ego.center.y), RADIUS,
                                           [L.LANE, L.LANE_CONNECTOR, L.INTERSECTION])
        d = np.full(X.shape, np.inf)
        in_lane = np.zeros(X.shape, bool)
        for ln in objs[L.LANE]:
            m = shapely.contains_xy(ln.polygon, X, Y)
            if not m.any():
                continue
            in_lane |= m
            bl = np.array([[s.x, s.y] for s in ln.baseline_path.discrete_path])
            dd = LC.point_polyline_dist(np.c_[X[m], Y[m]], bl)
            d[m] = np.minimum(d[m], dd)
        in_conn = np.zeros(X.shape, bool)
        for lc in objs[L.LANE_CONNECTOR]:
            in_conn |= shapely.contains_xy(lc.polygon, X, Y)
        in_int = np.zeros(X.shape, bool)
        for it_ in objs[L.INTERSECTION]:
            in_int |= shapely.contains_xy(it_.polygon, X, Y)
        lane_step = (in_lane & ~in_int).reshape(n, T)
        D = np.where(lane_step, d.reshape(n, T), np.nan)
        lk10 = lane_step & (D > 1.0)
        lk15 = lane_step & (D > 1.5)
        offl = (~in_lane & ~in_conn & ~in_int).reshape(n, T)
        with np.errstate(all="ignore"):
            dmed = np.nanmedian(np.where(lane_step, D, np.nan), axis=1)
        r3 = lambda a: [None if not np.isfinite(x) else round(float(x), 3) for x in a]   # noqa: E731
        return {"token": tok, "log": log, "map": map_name, "pick": int(_W["P"]["pick"][i]),
                "order": "0=pdm_closed,1..64=hypotheses,65=human",
                "lk10_s": [round(runs(r), 1) for r in lk10], "lk15_s": [round(runs(r), 1) for r in lk15],
                "nlane": [int(x) for x in lane_step.sum(1)], "offl_n": [int(x) for x in offl.sum(1)],
                "dmed": r3(dmed), "n_lanes_near": len(objs[L.LANE])}
    except Exception as e:                                                         # noqa: BLE001
        return {"token": tok, "log": log, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def main() -> int:
    if os.environ.get("REFE_LANE_CHILD") != "1":
        env = dict(os.environ, REFE_LANE_CHILD="1", PYTHONPATH=f"{LC.V11};{HERE}", PYTHONIOENCODING="utf-8",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUPLAN_MAPS_ROOT=MAPS, NUPLAN_MAP_VERSION="nuplan-maps-v1.0")
        return subprocess.call([LC.NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    E = json.load(gzip.open(LC.EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = sorted(E)
    random.Random(LC.SEED).shuffle(toks)
    if a.limit:
        toks = toks[: a.limit]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add(r["token"])
    todo = sorted((t for t in toks if t not in done), key=lambda t: E[t]["map_name"])   # stable: city, then shuffle order
    jobs = [(t, E[t]["log_name"], E[t]["map_name"], E[t]["human_future_poses"]) for t in todo]
    print(f"census2: {len(done)} done, {len(jobs)} to do, {a.workers} workers", flush=True)
    t0 = time.time()
    import multiprocessing as mp
    with open(OUT, "a", encoding="utf-8") as f, mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(lane_token, jobs, chunksize=8), 1):
            f.write(json.dumps(r) + "\n")
            if k % 200 == 0:
                f.flush()
                el = time.time() - t0
                print(f"{k}/{len(jobs)}  {el / k:.2f} s/token  eta {(len(jobs) - k) * el / k / 60:.0f} min", flush=True)
    print("ZZLANE_CENSUS2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
