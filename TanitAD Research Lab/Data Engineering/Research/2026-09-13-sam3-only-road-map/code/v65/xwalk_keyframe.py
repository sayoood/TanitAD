"""Crosswalk stripes from ONE keyframe per crossing instead of the fusion of all frames (PI 2026-09-14 on the zoom test at frame 84).
MEASURED motivation: in a single near frame SAM3 separates the bars (frame 84: ~6 bars), but the world map fuses every frame and
camera, and the bars land a little differently each time, so they merge into a band along the crossing edge (map r and every
fusion rule tried: stripes cover 5-12 % of the crossing pixels, as bands and fragments).
Rule: the crossings are the 1.5 m closing of cells with any crosswalk vote, on road (the delivered map's vote fields). For every
frame and camera, that camera's stored stripe mask (xstripe, 540 x 960, ego body removed) is lifted onto the smooth LiDAR ground
exactly like the extraction (camera_model.lift, every 2nd full-resolution pixel); points 4-12 m from the camera count. Per crossing
the keyframe is the (frame, camera) with the most stripe points inside it. In the output map (a copy of the base map) that crossing's
crosswalk cells become road and the keyframe's stripe cells (3 x 3 closing) on road become crosswalk. Crossings without any
keyframe point keep the base map.
PRE-REGISTERED 2026-09-14 before the first run: the keyframe map WORKS if, on BOTH clips (all 7 cameras), against map r, the crosswalk
bar/gap image top-hat over all frames (xwalk_reproj_contrast.py, same pixels) is >= r - 0.05 (night >= 1.467, day >= 2.337) AND the
bar share of crossing pixels is >= 1.3 x r (night >= 0.113, day >= 0.068). Anything else is a FAIL.
Usage: xwalk_keyframe.py <c8> <npz dir> <fields render dir> <base render dir> <out render dir>"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
import sam3_paint as P
import camera_model as CM
import ground_surface as GS

ROOT = Path("/home/nvidia/sam3map/native7")
c8, nd, fdir, bdir, odir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]); odir.mkdir(parents=True, exist_ok=True)
F = np.load(fdir / "worldmap_fields.npz", allow_pickle=True)
vk = {k: F[f"vk{k}"].astype(np.float32) for k in (1, 2, 3, 4, 6, 7)}
dw = vk[1] + vk[2] + vk[3] + vk[4] + vk[6]; drv = (dw > 0) & (dw >= vk[7])
x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
cross = (cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv
n_x, lab_x, st_x, _ = cv2.connectedComponentsWithStats(cross.astype(np.uint8), connectivity=8)
big = set(int(k) for k in np.flatnonzero(st_x[:, cv2.CC_STAT_AREA] >= 200)[1:])
(ox, oy), res = F["origin"], float(F["res"])
WW, WH = lab_x.shape
best = {}                                                   # crossing -> (count, frame, cam, cells i, cells j, mean range)
for f in sorted(nd.glob("[0-9][0-9][0-9].npz")):
    z = np.load(f, allow_pickle=True); tok = str(z["tok"]); fd = ROOT / f"seq_{c8}" / tok; T = z["T_world_rig"]
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    for key in [k for k in z.files if k.startswith("xstripe_")]:
        cam = key[len("xstripe_"):]
        m = z[key].astype(bool)
        if f"ego_{cam}" in z.files:
            m = m & ~z[f"ego_{cam}"].astype(bool)
        if m.sum() < 20:
            continue
        C = CM.Camera.from_calib(c, fr["cam_order"].index(cam))
        full = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)
        xy, rng = C.lift(full, sg, stride=2, max_xy=(40.0, 40.0))
        keep = (rng >= 4.0) & (rng <= 12.0)
        if keep.sum() < 20:
            continue
        w = xy[keep] @ T[:2, :2].T + T[:2, 3]
        ci = np.floor((w[:, 0] - ox) / res).astype(np.int64); cj = np.floor((w[:, 1] - oy) / res).astype(np.int64)
        ok = (ci >= 0) & (ci < WW) & (cj >= 0) & (cj < WH)
        ci, cj, rr = ci[ok], cj[ok], rng[keep][ok]
        labs = lab_x[ci, cj]
        for k in set(labs.tolist()) & big:
            sel = labs == k
            cnt = int(sel.sum())
            if k not in best or cnt > best[k][0]:
                best[k] = (cnt, f.stem, cam, ci[sel], cj[sel], float(rr[sel].mean()))
base = np.load(bdir / "worldmap.npz", allow_pickle=True)
out = base["cls"].copy()
rows = []
for k, (cnt, fr_n, cam, ci, cj, mr) in sorted(best.items()):
    comp = lab_x == k
    out[comp & (out == 3)] = 1
    mk = np.zeros(out.shape, np.uint8); mk[ci, cj] = 1
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    put = mk & comp & (out == 1)
    out[put] = 3
    rows.append({"crossing": k, "area_m2": round(float(comp.sum()) * res * res, 1), "keyframe": fr_n, "camera": cam, "stripe_points": cnt,
                 "mean_range_m": round(mr, 1), "stripe_cells": int(put.sum())})
np.savez_compressed(odir / "worldmap.npz", cls=out, rng=base["rng"], origin=base["origin"], res=base["res"], clip_sha12=base["clip_sha12"], ver=base["ver"],
                    composite=str(base["composite"]) + "+xwalk_keyframe")
stats = {"crossings_with_keyframe": len(rows), "crossings_total": len(big), "per_crossing": rows}
(odir / "render_v5_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in stats.items() if k != "per_crossing"}), rows[:8]); print("ZZKEYFRAME-DONEZZ")
