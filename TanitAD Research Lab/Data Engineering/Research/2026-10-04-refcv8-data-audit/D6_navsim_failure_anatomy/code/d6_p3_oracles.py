"""D6 P3 (navhard) -- zero-training PERFECT-FIX ORACLES on the banked refcv7 step-30,000 plans.   NAVSIM VENV, CPU only.

    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" OMP_NUM_THREADS=2 \
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe d6_p3_oracles.py --mode {speed,snap,controls} --hooks <A1 30k hooks.json> \
        --tokens <file> --out <jsonl> [--shard i/n]

Each oracle edits ONLY the banked emitted plan and re-scores it with the devkit's own scorer (exact path, the same objects the
official run used; the re-score reproduces the banked sub-scores bit-exactly, see d6_rescore.py).  The question each answers is an
UPPER BOUND: "if a fix of this KIND were perfect, what share of the failing scenes would clear?"

SPEED  (full reactive scorer, NC needs the IDM environment):  the SAME geometric path driven at s x the planned speed, i.e. the
       polyline through the planned poses re-sampled at arclength s * a_i (heading interpolated), s in {0.8, 0.6, 0.4}, plus s = 0.0
       (the STOP plan, the known-value control) and s = 1.0 (identity, must reproduce the banked row).
SNAP   (map-only DAC / DDC, no environment):  lateral fixes toward the route centreline, keeping the plan's own progress along the route:
       B075 / B150 = shift each point toward the centreline by at most 0.75 / 1.5 m (heading unchanged); FULL = every point put ON the
       centreline at the plan's own along-route position (heading = centreline tangent); and speed-only DAC variants V80 / V60 / V40.
Literals fixed here BEFORE any oracle number was read.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import pickle
import sys
import time

import numpy as np
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_rescore as R                                              # noqa: E402  (sets env defaults, imports navsim)
from navsim.common.dataclasses import Trajectory                     # noqa: E402
from navsim.evaluate.pdm_score import pdm_score                      # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling   # noqa: E402

SPEED_SCALES = (0.8, 0.6, 0.4)
SNAP_BOUNDS = (("B075", 0.75), ("B150", 1.5))
V_VARIANTS = (("V80", 0.8), ("V60", 0.6), ("V40", 0.4))
SUBS = R.SUBS


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def scale_speed(plan: np.ndarray, s: float) -> np.ndarray:
    """re-sample the plan's polyline at arclength s * a_i (same path, s x the speed)."""
    p = np.asarray(plan, dtype=np.float64)
    if s <= 0.0:
        return np.zeros_like(p)
    xy = np.vstack([[0.0, 0.0], p[:, :2]])
    yaw = np.concatenate([[0.0], np.unwrap(p[:, 2])])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    a = np.concatenate([[0.0], np.cumsum(seg)])
    tgt = s * a[1:]
    out = np.zeros_like(p)
    out[:, 0] = np.interp(tgt, a, xy[:, 0])
    out[:, 1] = np.interp(tgt, a, xy[:, 1])
    out[:, 2] = np.interp(tgt, a, yaw)
    return out


def centerline_frame(mc, plan):
    """per plan point: along-route s, signed lateral offset (left +), centreline point + tangent heading (global)."""
    ini = mc.ego_state
    x0, y0, h0 = ini.rear_axle.x, ini.rear_axle.y, ini.rear_axle.heading
    c, sn = math.cos(h0), math.sin(h0)
    ls = mc.centerline.linestring
    rows = []
    for x, y, _h in np.asarray(plan, dtype=np.float64):
        gx, gy = x0 + c * x - sn * y, y0 + sn * x + c * y
        s = float(ls.project(Point(gx, gy)))
        q = ls.interpolate(s)
        qa, qb = ls.interpolate(min(s + 1.0, ls.length)), ls.interpolate(max(s - 1.0, 0.0))
        tx, ty = qa.x - qb.x, qa.y - qb.y
        n = math.hypot(tx, ty) or 1.0
        tx, ty = tx / n, ty / n
        lat = (-ty) * (gx - q.x) + tx * (gy - q.y)
        rows.append((gx, gy, s, lat, q.x, q.y, tx, ty))
    return rows, (x0, y0, h0)


def snap_plan(mc, plan, mode, bound=None):
    rows, (x0, y0, h0) = centerline_frame(mc, plan)
    plan = np.asarray(plan, dtype=np.float64)
    out = plan.copy()
    ls = mc.centerline.linestring
    s_prev = -1e9
    for i, (gx, gy, s, lat, qx, qy, tx, ty) in enumerate(rows):
        if mode == "bound":
            lat2 = lat - math.copysign(min(abs(lat), bound), lat)
            ng = (-ty, tx)
            nx, ny = gx - (lat - lat2) * ng[0], gy - (lat - lat2) * ng[1]
            hh = plan[i, 2]
        else:                                                    # FULL: on the centreline at the plan's own progress
            s = max(s, s_prev)
            s_prev = s
            q = ls.interpolate(s)
            qa, qb = ls.interpolate(min(s + 1.0, ls.length)), ls.interpolate(max(s - 1.0, 0.0))
            nx, ny = q.x, q.y
            hh_g = math.atan2(qa.y - qb.y, qa.x - qb.x)
            hh = wrap(hh_g - h0)
        e = R.to_ego([[nx, ny, 0.0]], x0, y0, h0)[0]
        out[i, 0], out[i, 1] = e[0], e[1]
        out[i, 2] = hh
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("speed", "snap", "controls"))
    ap.add_argument("--hooks", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    toks = [l.strip() for l in open(a.tokens) if l.strip()]
    toks = [t for i, t in enumerate(toks) if i % n == k]
    if a.limit:
        toks = toks[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["token"])
            except Exception:                                     # noqa: BLE001
                pass
    hooks = {c["token"]: c for c in json.load(open(a.hooks, encoding="utf-8"))["pdm_score_calls"]}
    ps, simulator, scorer, policy = R.build_objects()
    samp = TrajectorySampling(time_horizon=4, interval_length=0.5)
    cnt = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok in toks:
            if tok in done:
                continue
            t0 = time.time()
            h = hooks[tok]
            rec = {"token": tok, "scene_type": h["scene_type"], "mode": a.mode}
            try:
                with lzma.open(R.cache_path(h["log_name"], tok), "rb") as f:
                    mc = pickle.load(f)
                plan = np.asarray(h["agent_poses"], dtype=np.float64)
                if a.mode in ("speed", "controls"):
                    scales = SPEED_SCALES if a.mode == "speed" else (1.0, 0.0)
                    rec["variants"] = {}
                    for s in scales:
                        pl = scale_speed(plan, s) if s != 1.0 else plan
                        traj = Trajectory(poses=pl.astype(np.float32), trajectory_sampling=samp)
                        row, _sim = pdm_score(mc, traj, ps, simulator, scorer, policy)
                        rec["variants"][f"S{s}"] = {kk: float(row[kk].iloc[0]) for kk in SUBS}
                        rec["variants"][f"S{s}"]["end_dist"] = float(np.hypot(*pl[-1, :2]))
                    rec["banked"] = {kk: float(h["row"][kk]) for kk in SUBS}
                else:                                              # snap (map-only)
                    rec["variants"] = {}
                    plans = {"ID": plan}
                    for name, b in SNAP_BOUNDS:
                        plans[name] = snap_plan(mc, plan, "bound", b)
                    plans["FULL"] = snap_plan(mc, plan, "full")
                    for name, s in V_VARIANTS:
                        plans[name] = scale_speed(plan, s)
                    for name, pl in plans.items():
                        traj = Trajectory(poses=pl.astype(np.float32), trajectory_sampling=samp)
                        res, _ = R.score_states_only(scorer, simulator, mc, traj, ps)
                        rec["variants"][name] = {"dac": res["dac"], "ddc": res["ddc"], "nd_first": res["nd_first"]}
                    rec["banked_dac"] = float(h["row"]["drivable_area_compliance"])
                rec["status"] = "OK"
            except Exception as e:                                 # noqa: BLE001
                rec["status"] = "ERR"
                rec["err"] = repr(e)[:300]
            rec["wall_s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            cnt += 1
    print("DONE", cnt, "->", a.out)


if __name__ == "__main__":
    main()
