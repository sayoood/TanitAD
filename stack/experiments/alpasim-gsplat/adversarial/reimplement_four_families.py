#!/usr/bin/env python3
"""Independent re-implementation of ADE / LONGITUDINAL / LATERAL from raw dumps.

Deliberately does NOT import cl_metrics — the formulas are re-written from the
definitions so a shared bug cannot hide.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/cutin/rollouts")
PANEL = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/cutin/CUTIN_PANEL.json")
sys.path.insert(0, r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

DT = 0.1
WP = (5, 10, 15, 20)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ego_frame(d, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def cross_track(p, xy):
    seg = xy[1:] - xy[:-1]
    L2 = (seg ** 2).sum(1).clip(1e-9)
    t = (((p[None, :2] - xy[:-1]) * seg).sum(1) / L2).clip(0, 1)
    proj = xy[:-1] + t[:, None] * seg
    dd = np.linalg.norm(proj - p[None, :2], axis=1)
    i = int(np.argmin(dd))
    n = seg[i] / math.sqrt(L2[i])
    r = p[:2] - proj[i]
    return float(n[0] * r[1] - n[1] * r[0])


def plan_poses(plan, v0, n=21):
    knots = np.vstack([[0.0, 0.0], np.asarray(plan, float)])
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    tq = np.arange(n) * DT
    x = np.interp(tq, ts, knots[:, 0])
    y = np.interp(tq, ts, knots[:, 1])
    d = np.diff(np.stack([x, y], 1), axis=0, prepend=np.zeros((1, 2)))
    yaw = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
    return np.stack([x, y, yaw], 1)


def rows(arm, cond):
    d = json.loads((ROOT / f"rollouts_{arm}_{cond}.json").read_text())
    gt = d["gt"]
    xy = np.array([[g["x"], g["y"]] for g in gt], float)
    gyaw = np.array([g["yaw"] for g in gt], float)
    gv = np.zeros(len(gt))
    dd = np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT
    gv[:-1] = dd
    gv[-1] = dd[-1]
    N = len(gt)
    out = {}
    for r in d["rollouts"]:
        for k, s in enumerate(r["steps"]):
            i = int(s["i_gt"])
            plan = np.array(s["plan"], float)
            gtc = np.stack([ego_frame(xy[min(i + h, N - 1)] - xy[i], gyaw[i]) for h in WP])
            de = np.linalg.norm(plan - gtc, axis=1)
            lon = np.abs(plan[:, 0] - gtc[:, 0])
            lat = np.abs(plan[:, 1] - gtc[:, 1])
            ct = cross_track(np.array(s["ego"][:2]), xy)
            j = min(i + WP[-1], N - 1)
            dyaw_gt = wrap(gyaw[j] - gyaw[i])
            arc_gt = float(np.linalg.norm(np.diff(xy[i:j + 1], axis=0), axis=1).sum())
            pp = plan_poses(plan, s["v"])
            dyaw_pl = wrap(float(pp[-1, 2]))
            arc_pl = float(np.linalg.norm(np.diff(pp[:, :2], axis=0), axis=1).sum())
            yr_exec = s["v"] / 2.7 * math.tan(s["steer"])
            yr_gt = wrap(gyaw[min(i + 1, N - 1)] - gyaw[i]) / DT
            out[(r["start_frame"], k)] = dict(
                ade=float(de.mean()), lon_ade=float(lon.mean()), lat_ade=float(lat.mean()),
                cross=abs(ct), dist_to_gt=abs(ct),
                sperr=abs(float(s["v_target"] - gv[i])),
                head=abs(wrap(dyaw_pl - dyaw_gt)),
                curv=abs(dyaw_pl / max(arc_pl, 1.0) - dyaw_gt / max(arc_gt, 1.0)),
                yawr=abs(yr_exec - yr_gt),
                corridor=float(abs(ct) > 2.0),
            )
    return out


panel = json.loads(PANEL.read_text())
MAP = {"ade": "ade_0_2s", "sperr": "abs_target_speed_err_ms", "lon_ade": "along_track_ade_m",
       "head": "heading_err_rad", "curv": "curvature_err_1pm", "yawr": "yawrate_err_rads",
       "cross": "cross_track_abs_m", "lat_ade": "lateral_ade_m",
       "corridor": "route_corridor_departure_rate", "dist_to_gt": "dist_to_gt_traj_m"}

print(f"{'arm/cond':22s} {'metric':26s} {'MINE':>28s}   {'PANEL':>28s}  MATCH")
bad = 0
for arm in ("refc-base", "flagship-v1"):
    E = rows(arm, "empty")
    for cond in ("lead25", "lead15", "lead8", "cutin"):
        C = rows(arm, cond)
        common = sorted(set(C) & set(E))
        eid = [c[0] for c in common]
        pv = panel["arms"][arm][cond]["paired_vs_empty"]
        for mk, pk in MAP.items():
            a = [C[c][mk] for c in common]
            b = [E[c][mk] for c in common]
            r = paired_episode_cluster_bootstrap(a, b, eid)
            p = pv[pk]
            mine = f"{r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"
            theirs = f"{p['delta']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}]"
            ok = (abs(r["delta"] - p["delta"]) < 1e-4 and abs(r["lo"] - p["lo"]) < 1e-3
                  and abs(r["hi"] - p["hi"]) < 1e-3 and r["separated"] == p["separated"])
            if not ok:
                bad += 1
            if cond in ("lead25", "lead8") or not ok:
                print(f"{arm+'/'+cond:22s} {pk:26s} {mine:>28s} vs {theirs:>28s}  "
                      f"{'OK' if ok else '**MISMATCH**'} sep_mine={r['separated']} sep_panel={p['separated']}")
print()
print("total mismatches:", bad)
