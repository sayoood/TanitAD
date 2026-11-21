#!/usr/bin/env python3
"""P1(b) — DECIDE the rig-frame convention by experiment, not by assumption.

`build_lead_tracks.lead_track_in_window` composes rig -> world as

    L_w = ego_xy + R(yaw) @ [center_x, center_y]

which ASSERTS x = forward, y = left, z = up and the same yaw convention as
`quaternion_yaw`. Nothing in the dataset card proves that, and getting the
handedness wrong silently mirrors every lead track about the ego's axis — a
defect that would look like plausible numbers.

THE DISCRIMINATING TEST. A parked car is stationary in the WORLD. Under the
correct convention its world track collapses to a point; under a mirrored or
swapped convention it sweeps an arc whenever the ego yaws. So: map every track
to world under each candidate convention and count how many tracks are
"world-static" (world position std < 0.5 m over >= 2 s of life).

Four candidates:
  xf_yl : x fwd, y left   (the deployed assumption)
  xf_yr : x fwd, y right  (mirrored lateral)
  xl_yf : x left, y fwd   (axes swapped)
  xr_yf : x right, y fwd
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack" / "scripts"))
from lead_state_gate import quaternion_yaw  # noqa: E402

ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
OUT = Path(__file__).with_name("p1_frame_convention.json")

CANDIDATES = {
    "xf_yl": lambda cx, cy: (cx, cy),
    "xf_yr": lambda cx, cy: (cx, -cy),
    "xl_yf": lambda cx, cy: (-cy, cx),
    "xr_yf": lambda cx, cy: (cy, -cx),
}
STATIC_STD_M = 0.5
MIN_LIFE_S = 2.0
MIN_SAMPLES = 15


def ego_poses(ego):
    t = ego["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    g = lambda c: ego[c].to_numpy(np.float64)[o]  # noqa: E731
    return {"t": t[o], "x": g("x"), "y": g("y"),
            "yaw": np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"), g("qw"))),
            "v": np.hypot(g("vx"), g("vy"))}


def main():
    cs = sorted({int(p.name.split("_")[-1].split(".")[0])
                 for p in (ROOT / "labels" / "obstacle.offline").glob("*.zip")}
                & {int(p.name.split("_")[-1].split(".")[0])
                   for p in (ROOT / "labels" / "egomotion").glob("*.zip")})
    tally = {k: {"static": 0, "std_sum": 0.0} for k in CANDIDATES}
    n_tracks = n_clips = 0
    ego_yaw_span = []
    for c in cs[:8]:
        with zipfile.ZipFile(ROOT / "labels" / "obstacle.offline" /
                             f"obstacle.offline.chunk_{c:04d}.zip") as oz, \
             zipfile.ZipFile(ROOT / "labels" / "egomotion" /
                             f"egomotion.chunk_{c:04d}.zip") as ez:
            om = {n.split("/")[-1].split(".")[0]: n for n in oz.namelist()
                  if n.endswith(".parquet")}
            em = {n.split("/")[-1].split(".")[0]: n for n in ez.namelist()
                  if n.endswith(".parquet")}
            for cid in sorted(set(om) & set(em))[:6]:
                obs = pd.read_parquet(io.BytesIO(oz.read(om[cid])))
                ego = pd.read_parquet(io.BytesIO(ez.read(em[cid])))
                p = ego_poses(ego)
                n_clips += 1
                for _tid, g in obs.groupby("track_id"):
                    t = g["timestamp_us"].to_numpy(np.float64) / 1e6
                    o = np.argsort(t)
                    t = t[o]
                    if t.size < MIN_SAMPLES or (t[-1] - t[0]) < MIN_LIFE_S:
                        continue
                    cx = g["center_x"].to_numpy(np.float64)[o]
                    cy = g["center_y"].to_numpy(np.float64)[o]
                    ex = np.interp(t, p["t"], p["x"])
                    ey = np.interp(t, p["t"], p["y"])
                    ya = np.interp(t, p["t"], p["yaw"])
                    ego_yaw_span.append(float(ya.max() - ya.min()))
                    n_tracks += 1
                    ce, se = np.cos(ya), np.sin(ya)
                    for k, f in CANDIDATES.items():
                        ax, ay = f(cx, cy)
                        wx = ex + ax * ce - ay * se
                        wy = ey + ax * se + ay * ce
                        sd = float(np.hypot(wx.std(), wy.std()))
                        tally[k]["std_sum"] += sd
                        if sd < STATIC_STD_M:
                            tally[k]["static"] += 1
    res = {
        "_what": ("which rig-frame convention makes PARKED cars actually parked. "
                  "A static object's WORLD position is constant only under the "
                  "correct convention."),
        "n_clips": n_clips, "n_tracks_eligible": n_tracks,
        "criterion": {"static_std_m": STATIC_STD_M, "min_life_s": MIN_LIFE_S,
                      "min_samples": MIN_SAMPLES},
        "ego_yaw_span_rad_median": round(float(np.median(ego_yaw_span)), 4)
        if ego_yaw_span else None,
        "candidates": {k: {"n_world_static": v["static"],
                           "frac_world_static": round(v["static"] / max(n_tracks, 1), 4),
                           "mean_world_pos_std_m": round(v["std_sum"] / max(n_tracks, 1), 3)}
                       for k, v in tally.items()},
    }
    best = max(res["candidates"], key=lambda k: res["candidates"][k]["n_world_static"])
    res["VERDICT"] = best
    res["deployed_assumption"] = "xf_yl"
    res["agrees_with_deployed"] = (best == "xf_yl")
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
