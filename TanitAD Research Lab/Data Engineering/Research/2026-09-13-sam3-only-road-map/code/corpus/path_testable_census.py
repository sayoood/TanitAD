"""How many corpus clips can the export's path-on-drivable check test at all? Emulates export_v2ep's geometry on the v2ep grid
(camera timestamps -> linspace at 10 Hz, rig pose at each frame by the same interpolation, the next 30 frames' positions in the rig
frame, forward >= 2 m, inside the fine grid 0-60 m x +-16 m). A clip with zero such points gets real = None and fails the gate
whatever its map looks like. (The export also needs those cells to be seen; this census cannot know that, so it is an upper bound on
testability.)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
SP = Path(r"<scratchpad>")
EGO = Path(r"C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo"); CAM = Path(r"C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
order = [l.strip() for l in open(SP / "corpus/production_order.txt") if l.strip()]
rows = []
for k, clip in enumerate(order):
    e = pd.read_parquet(EGO / f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
    t_cam = pd.read_parquet(CAM / f"{clip}.timestamps.parquet")["timestamp"].to_numpy(np.float64)
    n_target = max(int((t_cam[-1] - t_cam[0]) / 1e6 * 10.0), 4)
    tq = np.linspace(t_cam[0], t_cam[-1], n_target); t_img = t_cam[np.searchsorted(t_cam, tq).clip(0, len(t_cam) - 1)]
    ts = e["timestamp"].to_numpy(np.float64); i = np.clip(np.searchsorted(ts, t_img), 1, len(ts) - 1)
    a = ((t_img - ts[i - 1]) / np.maximum(ts[i] - ts[i - 1], 1.0)).clip(0, 1)
    P = e[["x", "y"]].to_numpy(np.float64); p = P[i - 1] * (1 - a)[:, None] + P[i] * a[:, None]
    Q = e[["qw", "qx", "qy", "qz"]].to_numpy(np.float64); q0, q1 = Q[i - 1], Q[i]; q1 = np.where((np.sum(q0 * q1, axis=1) < 0)[:, None], -q1, q1)
    q = q0 * (1 - a)[:, None] + q1 * a[:, None]; q /= np.linalg.norm(q, axis=1, keepdims=True)
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    n_pts = 0
    for n in range(len(p) - 1):
        d = p[n + 1: n + 31] - p[n]; c, s = np.cos(yaw[n]), np.sin(yaw[n])
        fx = d[:, 0] * c + d[:, 1] * s; fy = -d[:, 0] * s + d[:, 1] * c
        n_pts += int(((fx >= 2.0) & (fx < 60.0) & (np.abs(fy) < 16.0)).sum())
    path_m = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    rows.append({"clip": clip, "frames": len(p), "testable_points": n_pts, "path_m_camera_span": round(path_m, 2)})
    if k % 500 == 0:
        print(k, flush=True)
df = pd.DataFrame(rows); df.to_parquet(SP / "corpus/path_testable_census.parquet", index=False)
z = df[df["testable_points"] == 0]
print(f"clips {len(df)} | path check untestable (0 points) {len(z)} = {100 * len(z) / len(df):.2f} % | <30 points {int((df['testable_points'] < 30).sum())}")
print("path length over the camera span for the untestable clips: median", z["path_m_camera_span"].median(), "max", z["path_m_camera_span"].max())
print("is 081b986f8888's clip among them:", any(c.startswith("081b986f8888") for c in z["clip"]))
first315 = df.iloc[:315]; print("in the 315 BEV-head clips (first in the order):", int((first315["testable_points"] == 0).sum()))
