"""Small non-road islands in a world map: connected components of non-drivable cells (seen-no-class, edge, sidewalk-verge) of
1-30 m^2 whose 1-cell ring is >= 80 % drivable (incl. paint), within 25 m of the path; printed with their rig position at the
nearest frame and their class mix. Usage: find_islands.py <npz dir> <render dir>"""
import sys
from pathlib import Path
import numpy as np
import cv2
nd, rd = Path(sys.argv[1]), Path(sys.argv[2])
wm = np.load(rd / "worldmap.npz", allow_pickle=True); cls = wm["cls"]; (x0, y0), res = wm["origin"], float(wm["res"])
frames = sorted(nd.glob("[0-9][0-9][0-9].npz")); Ts = [np.load(f, allow_pickle=True)["T_world_rig"] for f in frames]
P = np.array([T[:2, 3] for T in Ts])
non = np.isin(cls, (0, 5, 7)).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(non, connectivity=8)
drv = np.isin(cls, (1, 2, 3, 4, 6))
rows = []
for k in range(1, n):
    a = st[k, cv2.CC_STAT_AREA]
    if a < 100 or a > 3000:
        continue
    xx, yy, ww, hh = st[k, 0], st[k, 1], st[k, 2], st[k, 3]
    i0, i1, j0, j1 = max(yy - 1, 0), min(yy + hh + 1, cls.shape[0]), max(xx - 1, 0), min(xx + ww + 1, cls.shape[1])
    comp = lab[i0:i1, j0:j1] == k
    ring = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~comp
    share = float(drv[i0:i1, j0:j1][ring].mean())
    if share < 0.8:
        continue
    ci, cj = cen[k][1], cen[k][0]
    w = np.array([x0 + (ci + 0.5) * res, y0 + (cj + 0.5) * res])
    dd = np.hypot(*(P - w).T); m = int(dd.argmin())
    if dd[m] > 25:
        continue
    q = (w - Ts[m][:2, 3]) @ Ts[m][:2, :2]
    sub = cls[i0:i1, j0:j1][comp]
    mix = {c: int((sub == c).sum()) for c in (0, 5, 7)}
    rows.append((a * res * res, m, q, share, mix, w))
for a, m, q, share, mix, w in sorted(rows, key=lambda r: r[1]):
    print(f"island {a:5.1f} m2  nearest frame {m:3d}  rig x {q[0]:6.1f} y {q[1]:6.1f}  ring drivable {share:.2f}  cells bg/edge/sidewalk {mix[0]}/{mix[5]}/{mix[7]}  world {w.round(1)}")
print("ZZISLANDS-DONEZZ")
