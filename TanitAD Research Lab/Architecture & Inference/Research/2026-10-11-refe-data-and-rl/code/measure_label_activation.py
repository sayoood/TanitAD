"""How often is each REFe RL-reward label ACTIVE on real nuPlan navtrain frames?  (0 GPU, CPU only)

WHY THIS EXISTS. The programme's own RL history (GOALS_AND_CLAIMS D-RL-COLL-SPARSE-1) measured that a
GRPO advantage is identically zero in a window where every candidate gets the same value -- 92 % of
windows for the collision channel.  A reward term that is active in 1 % of frames cannot move a
64-way ranking, whatever its weight.  So before any lane-change / lane-consistency / drivable-area term
is pre-registered, this script measures, on real map-matched frames, how often its LABEL fires.

It is also the field check the brief asks for: every attribute used below is read from the nuPlan map
API / tracked objects on real data, so an AttributeError here is a finding, not a crash to hide.

Run (dev box, eval venv; nuPlan devkit + 4 city maps are local):
  PYTHONPATH=<refe-dir> NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0 \
  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe measure_label_activation.py --out <json> \
      --logs 14 --per-log 24 --seed 20261011

Definitions (REGISTERED HERE before reading any rate; the numbers come out of the run, not out of tuning):
  human_lane_change   the human's next 4 s (8 poses @ 0.5 s) map-match to two DIFFERENT LANE objects of
                      the SAME roadblock (a lane connector, i.e. an intersection, never counts), AND the
                      final pose lies > 2.0 m laterally from the START lane's baseline path.
  slow_lead           nearest tracked VEHICLE ahead (ego-frame x in [2, 40] m, |y| < 1.6 m) with
                      speed < 0.7 * ego speed, ego speed > 4 m/s.
  adjacent_free       the adjacent LANE (left or right, from Lane.adjacent_edges) exists, and no tracked
                      VEHICLE centre lies inside its polygon with ego-frame x in [-12, +35] m.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sqlite3
import sys
import time

import numpy as np

os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")

DB_GLOB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval/*.db"


def _lane_at(map_api, p, SemanticMapLayer):
    """('L'|'C', id, roadblock_id, obj) for the first LANE (else LANE_CONNECTOR) containing p."""
    for layer, tag in ((SemanticMapLayer.LANE, "L"), (SemanticMapLayer.LANE_CONNECTOR, "C")):
        objs = map_api.get_all_map_objects(p, layer)
        if objs:
            o = objs[0]
            return tag, str(o.id), str(o.get_roadblock_id()), o
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", type=int, default=14)
    ap.add_argument("--per-log", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261011)
    a = ap.parse_args()

    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from shapely.geometry import LineString, Point
    from navtrain_scenarios import build_scenarios_for_log

    rng = random.Random(a.seed)
    dbs = sorted(glob.glob(DB_GLOB))
    rng.shuffle(dbs)
    rec = []
    errs = {}
    t0 = time.time()
    used_logs = 0
    for db in dbs:
        if used_logs >= a.logs:
            break
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            toks = [r[0] for r in con.execute(
                "SELECT lower(hex(token)) FROM lidar_pc ORDER BY timestamp").fetchall()]
        finally:
            con.close()
        if len(toks) < 400:
            continue
        # evenly spread, away from both ends; the builder applies the exact history/future guard itself
        cand = toks[100:-100]
        step = max(len(cand) // a.per_log, 1)
        pick = cand[::step][:a.per_log]
        n_before = len(rec)
        try:
            for sc in build_scenarios_for_log(db, pick):
                try:
                    rec.append(_one(sc, Point2D, SemanticMapLayer, TrackedObjectType, LineString, Point))
                except Exception as e:              # a failed frame is COUNTED, never silently dropped
                    k = type(e).__name__ + ":" + str(e)[:80]
                    errs[k] = errs.get(k, 0) + 1
        except Exception as e:
            k = "BUILD:" + type(e).__name__ + ":" + str(e)[:80]
            errs[k] = errs.get(k, 0) + 1
        if len(rec) > n_before:
            used_logs += 1
        print(f"[{time.time() - t0:6.1f}s] logs={used_logs} frames={len(rec)} errs={sum(errs.values())}",
              flush=True)

    out = _summarise(rec)
    out["errors"] = errs
    out["n_frames"] = len(rec)
    out["n_logs"] = used_logs
    out["seed"] = a.seed
    out["definitions"] = __doc__.split("Definitions", 1)[1]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"summary": out, "frames": rec}, f, indent=1)
    print(json.dumps(out, indent=1)[:3000])


def _one(sc, Point2D, SemanticMapLayer, TrackedObjectType, LineString, Point):
    ego0 = sc.initial_ego_state
    ex, ey, eyaw = ego0.rear_axle.x, ego0.rear_axle.y, ego0.rear_axle.heading
    v_e = float(ego0.dynamic_car_state.speed)
    cs, sn = math.cos(eyaw), math.sin(eyaw)
    fut = list(sc.get_ego_future_trajectory(0, 4.0, 8))
    poses = [(s.rear_axle.x, s.rear_axle.y) for s in fut]
    m = sc.map_api
    lane0 = _lane_at(m, Point2D(ex, ey), SemanticMapLayer)
    seq = [lane0] + [_lane_at(m, Point2D(x, y), SemanticMapLayer) for x, y in poses]
    # human lane change: two different LANE objects of the same roadblock + > 2 m off the START lane path
    lc, off_end, lat_dev_max = False, 0.0, 0.0
    if lane0 is not None and lane0[0] == "L":
        path = LineString([(p.x, p.y) for p in lane0[3].baseline_path.discrete_path])
        offs = [path.distance(Point(x, y)) for x, y in poses]
        off_end, lat_dev_max = offs[-1], max(offs)
        same_rb_other_lane = any(s is not None and s[0] == "L" and s[1] != lane0[1] and s[2] == lane0[2]
                                 for s in seq[1:])
        lc = bool(same_rb_other_lane and off_end > 2.0)
    in_conn = any(s is not None and s[0] == "C" for s in seq)
    # tracked vehicles in the ego frame
    veh = sc.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
    objs = []
    for o in veh:
        dx, dy = o.center.x - ex, o.center.y - ey
        lx, ly = dx * cs + dy * sn, -dx * sn + dy * cs
        objs.append((lx, ly, float(math.hypot(o.velocity.x, o.velocity.y)), o))
    lead = [(lx, v) for lx, ly, v, _ in objs if 2.0 <= lx <= 40.0 and abs(ly) < 1.6]
    lead_gap, lead_v = (min(lead)[0], min(lead)[1]) if lead else (None, None)
    slow_lead = bool(lead and v_e > 4.0 and lead_v < 0.7 * v_e)
    # adjacent lane free
    adj = {"left": None, "right": None}
    if lane0 is not None and lane0[0] == "L":
        left, right = lane0[3].adjacent_edges
        for side, lane in (("left", left), ("right", right)):
            if lane is None:
                continue
            poly = lane.polygon
            occ = any(poly.contains(Point(o.center.x, o.center.y)) and -12.0 <= lx <= 35.0
                      for lx, ly, v, o in objs)
            adj[side] = (not occ)
    adjacent_exists = any(v is not None for v in adj.values())
    adjacent_free = any(v is True for v in adj.values())
    return dict(token=str(sc.token), log=str(sc.log_name), v_ego=v_e, in_connector=in_conn,
                human_lane_change=lc, off_end_m=off_end, lat_dev_max_m=lat_dev_max,
                lead_gap=lead_gap, lead_v=lead_v, slow_lead=slow_lead,
                adjacent_exists=adjacent_exists, adjacent_free=adjacent_free,
                n_vehicles=len(objs), lane0=None if lane0 is None else lane0[0])


def _summarise(rec):
    n = max(len(rec), 1)

    def frac(f):
        return round(sum(1 for r in rec if f(r)) / n, 4)

    lc = [r for r in rec if r["human_lane_change"]]
    sl = [r for r in rec if r["slow_lead"]]
    return dict(
        p_human_lane_change=frac(lambda r: r["human_lane_change"]),
        n_human_lane_change=len(lc),
        p_slow_lead=frac(lambda r: r["slow_lead"]),
        n_slow_lead=len(sl),
        p_lc_given_slow_lead=(round(sum(r["human_lane_change"] for r in sl) / len(sl), 4) if sl else None),
        p_slow_lead_given_lc=(round(sum(r["slow_lead"] for r in lc) / len(lc), 4) if lc else None),
        p_adjacent_exists=frac(lambda r: r["adjacent_exists"]),
        p_adjacent_free=frac(lambda r: r["adjacent_free"]),
        p_slow_lead_and_adjacent_free=frac(lambda r: r["slow_lead"] and r["adjacent_free"]),
        p_in_connector_within_4s=frac(lambda r: r["in_connector"]),
        p_lat_dev_over_0p5m=frac(lambda r: r["lat_dev_max_m"] > 0.5),
        p_lane0_is_lane=frac(lambda r: r["lane0"] == "L"),
        median_v_ego=float(np.median([r["v_ego"] for r in rec])) if rec else None,
    )


if __name__ == "__main__":
    main()
