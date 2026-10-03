#!/usr/bin/env python3
"""CPU only, NO model. Does the PRIVILEGED PLANNER's route fix REFe's off-route goal?

NAVSIM's PDM-Closed repairs the route at iteration 0 (navsim pdm_closed_planner.py:73 -> route_utils.py:96
`route_roadblock_correction`: when the ego's roadblock is not on the route, a roadblock-graph BFS connects it), and its
`centerline` is stored in the navtest metric cache -- the SAME object NAVSIM's scorer measures EP and DDC against. Here the
nav goal is re-derived on that centerline exactly as REFe derives it (2 points at arc length s/2 and s ahead of the ego's
projection, s = max(|v0|, 5 m/s) * 12 s) and every goal estimate is scored as a PATH against the human's own 4 s future:
the [0, p1, p2] polyline at the human's travelled arc length, vs the human's 4 s endpoint (metres). Report-only.

Estimators: CURRENT (the census goal on the raw route) / PDM (route-corrected centerline) / STRAIGHT (along the heading) /
ARC (constant curvature omega/v from the last two ego headings; map-free).
    python pdm_route_goal_probe.py            (run in the navsim v1.1 venv; see main)
"""
from __future__ import annotations

import gzip
import json
import lzma
import math
import os
import pickle
import random
import subprocess
import sys

import numpy as np

DRV = os.environ.get("REFE_DRIVE", "D:")
HERE = os.path.dirname(os.path.abspath(__file__))
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = f"{DRV}/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
CACHE = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
HOR, MINV = 12.0, 5.0


def walk(P, s_targets):
    """points at arc lengths s_targets along polyline P (n,2), starting at P[0]; clamps at the end"""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.stack([np.interp(s_targets, cum, P[:, 0]), np.interp(s_targets, cum, P[:, 1])], 1)


def project_and_cut(C):
    """the centerline from the ego's projection onward (ego = origin), plus the ego's distance to it"""
    a, b = C[:-1], C[1:]
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    q = a + t[:, None] * ab
    d = np.linalg.norm(q, axis=1)
    i = int(np.argmin(d))
    return np.vstack([q[i], C[i + 1:]]), float(d[i])


def path_err(goal4, hum):
    """[0,p1,p2] at the human's travelled arc length vs the human's 4 s endpoint"""
    P = np.vstack([[0.0, 0.0], np.asarray(goal4, float).reshape(2, 2)])
    L = float(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], hum]), axis=0), axis=1).sum())
    return float(np.linalg.norm(walk(P, [L])[0] - hum[-1]))


def main() -> int:
    if os.environ.get("REFE_PDMG_CHILD") != "1":
        env = dict(os.environ, REFE_PDMG_CHILD="1", PYTHONPATH=V11, PYTHONIOENCODING="utf-8",
                   NUPLAN_MAPS_ROOT=f"{DRV}/Archive/devbox-C/navsim/data/maps", NUPLAN_MAP_VERSION="nuplan-maps-v1.0")
        return subprocess.call([NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    C = json.load(open(os.path.join(HERE, "goal_census_navtest_full.json"), encoding="utf-8"))["rows"]
    T = json.load(open(os.path.join(HERE, "goal_trigger_analysis.json"), encoding="utf-8"))
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    old = [t for t, r in C.items() if r.get("ego_to_route_m", 0) > 20.0]
    new_only = [r["token"] for r in T["new_only"]]
    rest = sorted(t for t, r in C.items() if "goal" in r and r.get("ego_to_route_m", 0) <= 20.0 and t not in set(new_only))
    random.Random(20260927).shuffle(rest)
    groups = {"old_trigger": old, "new_only": new_only, "control_untriggered_300": rest[:300]}
    out = {"what": __doc__.split("\n\n")[0], "groups": {}}
    rows = {}
    for gname, toks in groups.items():
        for t in toks:
            lg = C[t]["log"]
            p = os.path.join(CACHE, lg, "unknown", t, "metric_cache.pkl")
            if not os.path.exists(p):
                rows[t] = {"group": gname, "error": "no cache file"}
                continue
            mc = pickle.loads(lzma.decompress(open(p, "rb").read()))
            ra = mc.ego_state.rear_axle
            c, s = math.cos(-ra.heading), math.sin(-ra.heading)
            G = np.array([[st.x, st.y] for st in mc.centerline.discrete_path], dtype=np.float64)
            G = np.stack([(G[:, 0] - ra.x) * c - (G[:, 1] - ra.y) * s, (G[:, 0] - ra.x) * s + (G[:, 1] - ra.y) * c], 1)
            Pc, d_pdm = project_and_cut(G)
            v = math.hypot(*C[t]["v"])
            sl = max(v, MINV) * HOR
            g_pdm = walk(Pc, [sl / 2, sl]).reshape(-1)
            st_ = E[t]["ego_statuses"]
            h = [x["ego_pose"][2] for x in st_[-2:]]
            om = ((h[1] - h[0] + math.pi) % (2 * math.pi) - math.pi) / 0.5
            k = om / max(v, 1.0)                                   # curvature; capped speed floor 1 m/s
            arc = lambda L: np.array([math.sin(k * L) / k, (1 - math.cos(k * L)) / k]) if abs(k) > 1e-6 else np.array([L, 0.0])
            g_arc = np.concatenate([arc(sl / 2), arc(sl)])
            g_str = np.array([sl / 2, 0.0, sl, 0.0])
            hum = np.asarray(E[t]["human_future_poses"], dtype=np.float64)[:, :2]
            r = {"group": gname, "d_route_raw_m": C[t].get("ego_to_route_m"), "d_route_pdm_m": d_pdm,
                 "cmd": int(np.argmax(st_[-1]["driving_command"])), "v": v,
                 "goal_current": C[t]["goal"], "goal_pdm": [round(float(x), 3) for x in g_pdm],
                 "pdm_centerline_len_m": float(np.linalg.norm(np.diff(Pc, axis=0), axis=1).sum()),
                 "err": {"current": path_err(C[t]["goal"], hum), "pdm": path_err(g_pdm, hum),
                         "straight": path_err(g_str, hum), "arc": path_err(g_arc, hum)}}
            r["goal_current_vs_pdm_m"] = float(np.linalg.norm(np.asarray(C[t]["goal"]) - g_pdm))
            rows[t] = r
        ok = [rows[t] for t in toks if "err" in rows.get(t, {})]
        if not ok:
            continue
        summ = {"n": len(ok), "n_missing_cache": len(toks) - len(ok)}
        for k_ in ("current", "pdm", "straight", "arc"):
            e = np.array([r["err"][k_] for r in ok])
            summ[f"path_err_{k_}"] = {"mean": float(e.mean()), "median": float(np.median(e)), "p90": float(np.percentile(e, 90))}
        dp = np.array([r["d_route_pdm_m"] for r in ok])
        summ["ego_to_pdm_centerline_m"] = {"median": float(np.median(dp)), "p90": float(np.percentile(dp, 90)),
                                           "max": float(dp.max()), "n_over_20m": int((dp > 20).sum())}
        gd = np.array([r["goal_current_vs_pdm_m"] for r in ok])
        summ["goal_current_vs_pdm_m"] = {"median": float(np.median(gd)), "p90": float(np.percentile(gd, 90))}
        out["groups"][gname] = summ
        print(gname, json.dumps(summ), flush=True)
    out["rows"] = rows
    json.dump(out, open(os.path.join(HERE, "pdm_route_goal_probe.json"), "w", encoding="utf-8", newline="\n"), indent=0)
    print("ZZPDMG_DONE", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
