#!/usr/bin/env python3
"""The off-route defect AND its fix, for one token, in the ego frame (x forward, y left). CPU, no model.
Left: the wide view -- the goal REFe is given today (on the scenario route that does not contain the ego's road) vs the
route NAVSIM's privileged planner (PDM-Closed) actually uses after its iteration-0 route correction (metric-cache
`centerline`). Right: the near view -- the corrected-route goal, the human's own 4 s future, the straight / arc fallbacks.
    python offroute_fix_figure.py --token 134a3394c9b756d9      (navsim v1.1 venv, re-launches itself)
"""
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DRV = os.environ.get("REFE_DRIVE", "D:")
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = f"{DRV}/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
CACHE = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default="134a3394c9b756d9")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if os.environ.get("REFE_OFF_CHILD") != "1":
        env = dict(os.environ, REFE_OFF_CHILD="1", PYTHONPATH=V11, PYTHONIOENCODING="utf-8")
        return subprocess.call([NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a.token
    C = json.load(open(HERE / "goal_census_navtest_full.json", encoding="utf-8"))["rows"][t]
    P = json.load(open(HERE / "pdm_route_goal_probe.json", encoding="utf-8"))["rows"][t]
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"][t]
    mc = pickle.loads(lzma.decompress(open(os.path.join(CACHE, C["log"], "unknown", t, "metric_cache.pkl"), "rb").read()))
    ra = mc.ego_state.rear_axle
    c, s = math.cos(-ra.heading), math.sin(-ra.heading)
    G = np.array([[st.x, st.y] for st in mc.centerline.discrete_path], dtype=np.float64)
    G = np.stack([(G[:, 0] - ra.x) * c - (G[:, 1] - ra.y) * s, (G[:, 0] - ra.x) * s + (G[:, 1] - ra.y) * c], 1)
    T = mc.trajectory
    pdm_traj = []
    for st in T.get_sampled_trajectory():
        x, y = st.rear_axle.x - ra.x, st.rear_axle.y - ra.y
        pdm_traj.append([x * c - y * s, x * s + y * c])
    pdm_traj = np.asarray(pdm_traj)
    hum = np.asarray(E["human_future_poses"], dtype=np.float64)[:, :2]
    gc = np.asarray(C["goal"]).reshape(2, 2)
    gp = np.asarray(P["goal_pdm"]).reshape(2, 2)
    v = math.hypot(*C["v"])
    sl = max(v, 5.0) * 12.0
    cmd = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")[int(np.argmax(E["ego_statuses"][-1]["driving_command"]))]
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5))
    for k, A_ in enumerate(ax):
        A_.plot(G[:, 1], G[:, 0], "-", color="#2a9d8f", lw=2.5, label="PDM-Closed's CORRECTED route (metric-cache centerline)")
        A_.plot(pdm_traj[:, 1], pdm_traj[:, 0], "-", color="#264653", lw=1.5, alpha=0.8, label="PDM-Closed's own 4 s plan (privileged, GT agents)")
        A_.plot(hum[:, 1], hum[:, 0], "o-", color="#e76f51", ms=4, lw=2, label="human's logged 4 s future")
        A_.plot([0, gp[0, 1], gp[1, 1]], [0, gp[0, 0], gp[1, 0]], "s--", color="#2a9d8f", ms=9, lw=1.2,
                label=f"nav goal on the corrected route ({sl / 2:.0f} m / {sl:.0f} m ahead)")
        A_.plot([0, gc[0, 1], gc[1, 1]], [0, gc[0, 0], gc[1, 0]], "X--", color="#d62828", ms=11, lw=1.2,
                label=f"nav goal REFe is given today (on the raw route, {np.linalg.norm(gc[1]):.0f} m away)")
        A_.plot(0, 0, marker=(3, 0, 0), color="k", ms=16)
        A_.set_aspect("equal")
        A_.grid(alpha=0.3)
        A_.set_xlabel("y (m, left +)")
        A_.set_ylabel("x (m, forward +)")
        A_.invert_xaxis()
        if k == 0:
            pts = np.vstack([gc, gp, [[0, 0]], hum])
            lo, hi = pts.min(0) - 25, pts.max(0) + 25
            A_.set_xlim(hi[1], lo[1]); A_.set_ylim(lo[0], hi[0])
            A_.set_title(f"token {t}: wide view -- ego {C['ego_to_route_m']:.0f} m from its scenario route")
        else:
            A_.set_xlim(40, -40); A_.set_ylim(-10, 75)
            A_.set_title(f"near view -- {v * 3.6:.0f} km/h, nav command {cmd}")
    ax[0].legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.suptitle("Why the ego's road segment is 'missing': the scenario route skips the ego's roadblock; NAVSIM's privileged "
                 "planner repairs the route at t=0, REFe's goal path does not", fontsize=11)
    fig.tight_layout()
    out = a.out or str(HERE / f"offroute_fix_{t}.png")
    fig.savefig(out, dpi=110)
    print("wrote", out, "| path error vs human 4 s endpoint (m): current", round(P["err"]["current"], 2),
          "pdm", round(P["err"]["pdm"], 2), "straight", round(P["err"]["straight"], 2), "arc", round(P["err"]["arc"], 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
