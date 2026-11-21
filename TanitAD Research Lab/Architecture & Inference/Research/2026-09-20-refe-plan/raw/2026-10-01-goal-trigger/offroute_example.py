#!/usr/bin/env python3
"""One picture of the defect: a navtest token whose scenario ROUTE does not contain the ego's own road segment.
Draws, in the ego frame (x forward, y left), the route polyline the planner builds, its nearest point to the ego, the
goal points REFe is given, the ego's logged future, and the front camera.   (driverl venv, CPU, no model)
    python offroute_example.py --token <tok> --out offroute_<tok>.png
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
sys.path.insert(0, str(PKG / "refe"))
sys.path.insert(0, str(PKG / "code"))
import eval_checkpoint as EC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("REFE_OX_CHILD") != "1":
        env = EC.env_driverl()
        env["REFE_OX_CHILD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(PKG / "refe"), str(PKG / "eval"), str(PKG / "code")])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), "--token", a.token, "--out", str(Path(a.out).resolve())],
                               cwd=str(PKG / "eval"), env=env)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    import torch
    import augment_routes as A
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    from driverl.datatypes.goal_position_utils import route_goal_positions
    from nuplan.planning.script import driverl_runtime_map_features as M  # noqa: F401
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    ex = exp[a.token]
    log = ex["log_name"]
    sc = next(s for s in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{log}.db"), [a.token], history_rows=1, future_rows=80)
              if s._initial_lidar_token == a.token)
    ego = sc.get_ego_state_at_iteration(0)
    ids = list(sc.get_route_roadblock_ids() or [])
    poly, _ = A._route_with_lane_rank(sc.map_api, ids, A._anchor_from_ego(ego), 0)
    poly = np.asarray(poly, float)
    v = ego.dynamic_car_state.rear_axle_velocity_2d
    rp = torch.as_tensor(poly[None], dtype=torch.float32)
    mask = torch.as_tensor((np.abs(poly).sum(-1) > 0)[None])
    g = route_goal_positions(torch.zeros(1, 1, 2), torch.tensor([[[float(v.x), float(v.y)]]], dtype=torch.float32), rp, mask,
                             horizon_s=12.0, min_speed_mps=5.0, num_goal_positions=2).reshape(-1).tolist()
    a_, b_ = poly[:-1], poly[1:]
    ab = b_ - a_
    t = np.clip(-(a_ * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    pts = a_ + t[:, None] * ab
    k = int(np.argmin(np.linalg.norm(pts, axis=1)))
    near = pts[k]
    dmin = float(np.linalg.norm(near))
    hum = np.array(ex["human_future_poses"])[:, :2]
    s = max(float(np.hypot(v.x, v.y)), 5.0) * 12.0
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_axes([0.04, 0.10, 0.40, 0.78])
    ax.plot(-poly[:, 1], poly[:, 0], "-", color="#1f9bd7", lw=3, label=f"route polyline the planner builds ({len(poly)} pts)")
    ax.plot(-near[1], near[0], "o", color="#1f9bd7", ms=9)
    ax.annotate(f"closest route point\n{dmin:.0f} m from the ego", (-near[1], near[0]), textcoords="offset points", xytext=(12, 8), fontsize=10)
    ax.plot(0, 0, "s", color="k", ms=10, label="ego car (origin)")
    ax.plot(-hum[:, 1], hum[:, 0], "-", color="#2ca02c", lw=3, label="where the human actually drove (4 s)")
    gp = np.array(g).reshape(-1, 2)
    ax.plot(-gp[:, 1], gp[:, 0], "x", color="#d62728", ms=13, mew=3, label=f"goal points REFe is given ({np.hypot(*gp[-1]):.0f} m away)")
    th = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ax.plot(-s * np.sin(th), s * np.cos(th), ":", color="#999999", lw=1.5, label=f"farthest a valid goal can be: {s:.0f} m (speed x 12 s)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    lim = max(float(np.abs(poly).max()), 40.0) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.15, lim)
    ax.set_title("Ego frame (x forward, metres): the route does not pass through the ego's road segment", fontsize=11)
    ax.legend(loc="lower center", fontsize=8.5, framealpha=0.9)
    ax2 = fig.add_axes([0.46, 0.10, 0.52, 0.78])
    zf = zipfile.ZipFile(f"{SEAM.DEFAULT_FRAMES if hasattr(SEAM, 'DEFAULT_FRAMES') else 'D:/Projects/TanitAD/data/refe_navtest/frames'}/{log}_CAM_F0.zip")
    name = os.path.basename(ex["cams"]["cam_f0"][-1])
    im = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
    ax2.imshow(im[:, :, ::-1])
    ax2.axis("off")
    ax2.set_title(f"front camera, same scene (speed {np.hypot(v.x, v.y) * 3.6:.0f} km/h, command "
                  f"{['LEFT', 'STRAIGHT', 'RIGHT', 'UNKNOWN'][int(np.argmax(ex['ego_statuses'][-1]['driving_command']))]})", fontsize=11)
    fig.suptitle(f"token {a.token}   log {log[:10]}..{log[-11:]}", fontsize=10)
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out, "d_route", round(dmin, 1), "goal", [round(x, 1) for x in g])
    return 0


if __name__ == "__main__":
    sys.exit(main())
