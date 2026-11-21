"""Do the crosswalk bars of a map land on the white bars in the camera images? Per frame (every 2nd) and camera (FW, CL, CR):
the image's white top-hat (127 px rectangle, grey levels) on the pixels of each crossing where the map says BAR (crosswalk) and
where it says GAP (road inside the same crossing), through the same ground reprojection as the videos, restricted to pixels the
frame's SAM3 raster calls road-level. A map whose bars sit on the paint shows bar >> gap; hallucinated or shifted bars do not.
The crossing region is the same for every map (1.5 m closing of cells with any crosswalk vote, on road, from the vote fields), so
maps are compared on identical pixels. Single images, not the multi-frame averaged fields the stripe fit used.
Usage: xwalk_reproj_contrast.py <c8> <npz dir> <fields render dir> <label=render dir> [...]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import gt_reproject as GR

ROOT = Path("/home/nvidia/sam3map/native7")
c8, nd, fdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
variants = [(a.split("=", 1)[0], Path(a.split("=", 1)[1])) for a in sys.argv[4:]]
F = np.load(fdir / "worldmap_fields.npz", allow_pickle=True)
vk = {k: F[f"vk{k}"].astype(np.float32) for k in (1, 2, 3, 4, 6, 7)}
dw = vk[1] + vk[2] + vk[3] + vk[4] + vk[6]
drv = (dw > 0) & (dw >= vk[7])
x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
cross = ((cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv).astype(np.uint8)
origin, res = F["origin"], float(F["res"])
maps = {l: np.load(d / "worldmap.npz", allow_pickle=True)["cls"] for l, d in variants}
acc = {l: {"bar_sum": 0.0, "bar_n": 0, "gap_sum": 0.0, "gap_n": 0} for l in maps}
RECT = cv2.getStructuringElement(cv2.MORPH_RECT, (127, 127))
for f in sorted(nd.glob("[0-9][0-9][0-9].npz"))[::2]:
    z = np.load(f, allow_pickle=True); tok = str(z["tok"]); fd = ROOT / f"seq_{c8}" / tok; T = z["T_world_rig"]
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    lid = np.load(fd / "lidar.npy"); grid, fb = P.ground_grid(lid.astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    for cam in ("CAM_FW", "CAM_CL", "CAM_CR"):
        img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
        C = CM.Camera.from_calib(c, fr["cam_order"].index(cam), img.shape[1], img.shape[0])
        wxy, keep, hw = GR.camera_ground(C, sg, T, 2, lidar=None, ego_small=z[f"ego_{cam}"] if f"ego_{cam}" in z.files else None)
        g = (z[f"cls_{cam}"] >= 1) & (z[f"cls_{cam}"] <= 7)
        keep &= cv2.resize(g.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).ravel() > 0
        inx = GR.lookup(wxy, keep, hw, cross, origin, res).ravel() > 0
        if inx.sum() < 200:
            continue
        Y = cv2.cvtColor(cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (63, 63))).ravel()
        for l, cls in maps.items():
            code = GR.lookup(wxy, keep, hw, cls, origin, res).ravel()
            bar = inx & (code == 3); gap = inx & (code == 1)
            a = acc[l]; a["bar_sum"] += float(th[bar].sum()); a["bar_n"] += int(bar.sum()); a["gap_sum"] += float(th[gap].sum()); a["gap_n"] += int(gap.sum())
rep = {}
for l, a in acc.items():
    b = a["bar_sum"] / max(a["bar_n"], 1); gp = a["gap_sum"] / max(a["gap_n"], 1)
    rep[l] = {"bar_tophat": round(b, 2), "gap_tophat": round(gp, 2), "bar_over_gap": round(b / max(gp, 1e-6), 3), "bar_px": a["bar_n"], "gap_px": a["gap_n"],
              "bar_share_of_crossing_px": round(a["bar_n"] / max(a["bar_n"] + a["gap_n"], 1), 3)}
    print(l, rep[l], flush=True)
Path(f"/home/nvidia/sam3map/xwalk_reproj_contrast_{c8}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print("ZZXRC-DONEZZ")
