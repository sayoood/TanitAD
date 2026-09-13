"""Probe 2: is the renderer's 31 px image-space top-hat ERODING wide paint? Inside SAM3 thin-class masks (2 line, 4 symbol,
6 hatched), compare what a 31 px and a 127 px white top-hat keep, by ground range. Pixels kept by 127 px but not by 31 px
are the interior of paint wider than the small structuring element. Saves a visual: image | 31 px keep | 127 px keep."""
import json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import sam3_paint as P  # noqa: E402
import camera_model as CM  # noqa: E402
import ground_surface as GS  # noqa: E402

npz, seq, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
d = np.load(npz, allow_pickle=True)
fd = seq / str(d["tok"])
c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
BINS = (0, 4, 8, 12, 20, 60)
panels = []
for cam in [k[4:] for k in d.files if k.startswith("cls_")]:
    img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
    cls = cv2.resize(d[f"cls_{cam}"], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    Y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    road = cls == 1
    keep = {}
    for se in (31, 127):
        th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se)))
        mad = float(np.median(np.abs(th[road] - np.median(th[road])))) if road.any() else 3.0
        keep[se] = th > max(10.0, 2.5 * mad)
    thin = np.isin(cls, (2, 4, 6))
    if thin.sum() < 200:
        continue
    C = CM.Camera.from_calib(c, fr["cam_order"].index(cam), img.shape[1], img.shape[0])
    vv, uu = np.nonzero(thin[::2, ::2]); u = uu * 2.0; v = vv * 2.0
    ray = C.rays_rig(u, v); ok = np.isfinite(ray).all(axis=1) & (ray[:, 2] < -1e-3)
    s = (float(np.median(sg)) - C.t[2]) / np.where(ok, ray[:, 2], -1.0)
    xy = C.t[:2] + ray[:, :2] * s[:, None]; rng = np.where(ok, np.hypot(xy[:, 0] - C.t[0], xy[:, 1] - C.t[1]), np.inf)
    k31 = keep[31][vv * 2, uu * 2]; k127 = keep[127][vv * 2, uu * 2]
    row = {}
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (rng >= a) & (rng < b)
        if m.sum() >= 50:
            row[f"{a}-{b}m"] = {"n": int(m.sum()), "keep31": round(float(k31[m].mean()), 2), "keep127": round(float(k127[m].mean()), 2),
                                "only127": round(float((k127 & ~k31)[m].mean()), 2)}
    print(cam, json.dumps(row), flush=True)
    vis = img.copy()
    vis[thin & keep[31]] = (255, 214, 0); vis[thin & keep[127] & ~keep[31]] = (255, 40, 40); vis[thin & ~keep[127] & ~keep[31]] = (60, 60, 255)
    panels.append((cam, Image.fromarray(vis).resize((960, 540))))
sheet = Image.new("RGB", (1920, 540 * ((len(panels) + 1) // 2)), (0, 0, 0))
for q, (cam, p) in enumerate(panels):
    sheet.paste(p, ((q % 2) * 960, (q // 2) * 540))
sheet.save(out)
print("ZZTOPHAT2-DONEZZ  yellow = kept by 31 px, red = kept only by 127 px (eroded by 31 px), blue = SAM3 thin mask kept by neither")
