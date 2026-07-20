"""Scan cosmos-val clips: net yaw + max lateral offset of the ego path, to find
curves for the projection-verification test."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
from taniteval.cam_overlay import ego_future_path  # noqa
from tanitad.data.mixing import load_episode        # noqa

ROOT = "/root/valdata/cosmos-val-e8f3cef4976b"
files = sorted(Path(ROOT).glob("ep_*.pt"))
rows = []
for i, f in enumerate(files):
    ep = load_episode(str(f), mmap=True)
    p = ep.poses.float()
    T = p.shape[0]
    net_yaw = float((p[-1, 2] - p[0, 2]) * 57.2958)
    # max lateral offset over any 2s future window (ego frame)
    maxlat = 0.0
    for t in range(0, T - 21, 5):
        fp = ego_future_path(p, t, 20)
        maxlat = max(maxlat, float(fp[:, 1].abs().max()))
    rows.append((i, T, float(p[:, 3].mean()), net_yaw, maxlat))
rows.sort(key=lambda r: -abs(r[3]))
print("idx  T   v0    net_yaw  maxlat_m  (sorted by |net_yaw|)")
for i, T, v0, ny, ml in rows[:16]:
    print(f"{i:3d} {T:3d} {v0:5.1f}  {ny:7.1f}  {ml:6.2f}")
