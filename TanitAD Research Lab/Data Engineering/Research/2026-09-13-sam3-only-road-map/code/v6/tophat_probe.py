"""Probe: how much SAM3 paint does the renderer's image-space refine (white top-hat 31 px) keep, per camera and class,
and how does that depend on RANGE (near paint is wide in pixels, wider than the 31 px structuring element)?
Reads one extraction npz + the native images + the LiDAR ground; prints kept share per class in range bins."""
import json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import sam3_paint as P  # noqa: E402
import camera_model as CM  # noqa: E402
import ground_surface as GS  # noqa: E402

npz, seq = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(npz, allow_pickle=True)
fd = seq / str(d["tok"])
c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
TOPHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
BINS = (0, 4, 8, 12, 20, 60)
for cam in [k[4:] for k in d.files if k.startswith("cls_")]:
    img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
    cls = cv2.resize(d[f"cls_{cam}"], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    Y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, TOPHAT)
    road = cls == 1
    mad = float(np.median(np.abs(th[road] - np.median(th[road])))) if road.any() else 3.0
    thr = max(10.0, 2.5 * mad)
    C = CM.Camera.from_calib(c, fr["cam_order"].index(cam), img.shape[1], img.shape[0])
    row = {}
    for k in (2, 3, 4, 6):
        m = cls == k
        if m.sum() < 200:
            continue
        vv, uu = np.nonzero(m[::4, ::4]); u = uu * 4.0; v = vv * 4.0
        ray = C.rays_rig(u, v); ok = np.isfinite(ray).all(axis=1) & (ray[:, 2] < -1e-3)
        s = (float(np.median(sg)) - C.t[2]) / np.where(ok, ray[:, 2], -1.0)
        xy = C.t[:2] + ray[:, :2] * s[:, None]; rng = np.where(ok, np.hypot(xy[:, 0] - C.t[0], xy[:, 1] - C.t[1]), np.inf)
        kept = th[vv * 4, uu * 4] > thr
        row[k] = {f"{a}-{b}m": (int(((rng >= a) & (rng < b)).sum()), round(float(kept[(rng >= a) & (rng < b)].mean()), 2) if ((rng >= a) & (rng < b)).any() else None)
                  for a, b in zip(BINS[:-1], BINS[1:])}
    print(cam, "thr %.1f" % thr, json.dumps(row), flush=True)
print("ZZTOPHAT-DONEZZ")
