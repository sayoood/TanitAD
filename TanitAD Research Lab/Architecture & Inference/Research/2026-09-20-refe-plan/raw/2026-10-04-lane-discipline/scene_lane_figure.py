#!/usr/bin/env python3
"""Top-down MAP view of consecutive plans of one drive: the nuPlan lanes (polygons + centrelines), lane connectors and
intersections, with REFe's 64 hypotheses (coloured by the planner's predicted score), the selected plan, the human's
future and PDM-Closed -- all as NAVSIM simulates them (41 steps, 0.1 s), drawn in the ego frame of each plan.

Why: the video projects plans onto the front camera with a FLAT-GROUND model; on a sloped curve that displaces far points,
so the camera overlay alone cannot say whether a plan leaves its lane. This view can (map geometry, no projection).
Visualisation only -- the map is never a model input (map ruling).
    python scene_lane_figure.py --log <log> --first 18 --count 9 --out fig.png      (re-launches in the navsim venv)
"""
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import pickle
import subprocess
import sys

import numpy as np

import lane_census as LC
import lane_census2 as LC2

HERE = os.path.dirname(os.path.abspath(__file__))


def run_tokens(E, log):
    lst = sorted((v["timestamps_us"][-1], t) for t, v in E.items() if v["log_name"] == log)
    runs, cur = [], [lst[0]]
    for a, b in zip(lst, lst[1:]):
        if abs((b[0] - a[0]) / 1e6 - 0.5) < 0.02:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return [x[1] for x in max(runs, key=len)]


def main() -> int:
    if os.environ.get("REFE_LANE_CHILD") != "1":
        env = dict(os.environ, REFE_LANE_CHILD="1", PYTHONPATH=f"{LC.V11};{HERE}", PYTHONIOENCODING="utf-8",
                   NUPLAN_MAPS_ROOT=LC2.MAPS, NUPLAN_MAP_VERSION="nuplan-maps-v1.0")
        return subprocess.call([LC.NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer as L
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--first", type=int, default=0, help="1-based plan number in the drive (as the video's 'plan #')")
    ap.add_argument("--count", type=int, default=9)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = json.load(gzip.open(LC.EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = run_tokens(E, a.log)
    LC2._init()
    W = LC2._W
    sel = toks[a.first - 1: a.first - 1 + a.count]
    rows = (len(sel) + a.cols - 1) // a.cols
    fig, axes = plt.subplots(rows, a.cols, figsize=(5.2 * a.cols, 6.4 * rows), squeeze=False)
    samp8 = TrajectorySampling(num_poses=8, interval_length=0.5)
    info = []
    for k, (tok, ax) in enumerate(zip(sel, axes.ravel())):
        i = W["idx"][tok]
        props = W["P"]["proposals"][i].astype(np.float64)
        agg = LC.agg_v1(W["P"]["logits"][i])
        pick = int(W["P"]["pick"][i])
        hum = np.asarray(E[tok]["human_future_poses"], np.float64)
        mc = pickle.load(lzma.open(f"{LC.CACHE}/{a.log}/unknown/{tok}/metric_cache.pkl", "rb"))
        ego = mc.ego_state
        st = [get_trajectory_as_array(mc.trajectory, W["samp"], ego.time_point)]
        for tr in list(props) + [hum]:
            st.append(get_trajectory_as_array(transform_trajectory(Trajectory(tr.astype(np.float32), samp8), ego),
                                              W["samp"], ego.time_point))
        sim = W["sim"].simulate_proposals(np.stack(st, 0), ego)
        x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        c, s = np.cos(-h0), np.sin(-h0)

        def to_ego(xy):
            d = np.asarray(xy) - [x0, y0]
            return np.c_[d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c]

        def plot_xy(xy, **kw):                       # ego frame -> axes: forward up, left to the left
            e = to_ego(xy)
            ax.plot(-e[:, 1], e[:, 0], **kw)

        mp = LC2.get_map(E[tok]["map_name"])
        objs = mp.get_proximal_map_objects(Point2D(x0, y0), 70.0, [L.LANE, L.LANE_CONNECTOR, L.INTERSECTION])
        for it in objs[L.INTERSECTION]:
            e = to_ego(np.array(it.polygon.exterior.coords))
            ax.fill(-e[:, 1], e[:, 0], color="#f3e3c3", zorder=0)
        route = set(str(x) for x in mc.route_lane_ids)
        for ln in objs[L.LANE] + objs[L.LANE_CONNECTOR]:
            e = to_ego(np.array(ln.polygon.exterior.coords))
            bl = np.array([[q.x, q.y] for q in ln.baseline_path.discrete_path])
            hd = np.array([q.heading for q in ln.baseline_path.discrete_path])
            j = int(np.argmin(np.linalg.norm(bl - [x0, y0], axis=1)))
            same = abs((hd[j] - h0 + np.pi) % (2 * np.pi) - np.pi) < np.pi / 2
            on_route = str(ln.id) in route
            ax.fill(-e[:, 1], e[:, 0], color=("#d9e7f7" if on_route else "#e6e6e6") if same else "#f6d5d5", zorder=1)
            ax.plot(-e[:, 1], e[:, 0], color="#9a9a9a", lw=0.6, zorder=2)
            be = to_ego(bl)
            ax.plot(-be[:, 1], be[:, 0], color="#a0a0a0", lw=0.7, ls=(0, (3, 3)), zorder=2)
            if len(be) > 3:                          # travel direction arrow at the centreline midpoint
                m = len(be) // 2
                ax.annotate("", xy=(-be[m + 1, 1], be[m + 1, 0]), xytext=(-be[m - 1, 1], be[m - 1, 0]),
                            arrowprops=dict(arrowstyle="-|>", color="#808080", lw=0.8), zorder=2)
        nrm = (agg - agg.min()) / max(agg.max() - agg.min(), 1e-9)
        for j in np.argsort(agg):
            if j != pick:
                plot_xy(sim[j + 1, :, :2], color=cm.turbo(0.15 + 0.85 * nrm[j]), lw=0.8, alpha=0.8, zorder=3)
        plot_xy(sim[0, :, :2], color="#2b6cf0", lw=1.6, ls="--", zorder=4, label="PDM-Closed (privileged)")
        plot_xy(sim[65, :, :2], color="black", lw=2.2, zorder=5, label="human future")
        plot_xy(sim[pick + 1, :, :2], color="#1fae3a", lw=2.6, zorder=6, label="selected plan")
        ax.add_patch(plt.Rectangle((-1.15, -1.13), 2.3, 5.18, color="#1fae3a", zorder=7))
        ax.set_xlim(-16, 16)
        ax.set_ylim(-6, 42)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"plan #{a.first + k}  {tok}\nselected #{pick}  pred {100 * agg[pick]:.1f}", fontsize=9)
        info.append({"plan": a.first + k, "token": tok, "pick": pick})
    for ax in axes.ravel()[len(sel):]:
        ax.axis("off")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"{a.log}  ({E[sel[0]]['map_name']}): lanes blue = on the (corrected) route, grey = same direction off "
                 "route, red = ONCOMING; dashed = centreline, arrow = travel direction; junctions tan.\n"
                 "Hypotheses coloured by predicted score (blue low -> red high). Map is drawn for inspection only, never a "
                 "model input.", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
