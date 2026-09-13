"""SAM3 road paint -> BEV, fused with the Qwen-Drive map.

LIFT: every SAM3 mask pixel is cast as a ray through the camera's exact calibration and intersected with the
LOCAL ground surface measured by LiDAR (2 m cells, lower envelope; iterated twice), not a flat-world plane.
Labels may use LiDAR; nothing here is an inference path.

FUSION (single frame on the demo; the same rule accumulates over +-2 s on our clips):
  paint  := SAM3 {lane marking, road marking} lifted, kept only within 0.6 m of (Qwen driveable|line|crosswalk)
  zebra  := SAM3 {crosswalk, zebra crossing} lifted, same gate
  Qwen road_line / crosswalk are KEPT only where no camera observed the ground (outside every frustum or
  beyond 30 m); where SAM3 looked, SAM3 decides the paint -- the texture problem is its strength.

SCORE (demo only, where the TRUE nuPlan map exists): paint classes {road_line, crosswalk} as one set,
precision = predicted paint cells within 0.45 m of true paint; recall = true paint cells (inside the
observed 30 m window) within 0.45 m of predicted paint. Arms: QWEN (its own prediction), SAM3, FUSED,
plus SAM3-MIRRORED and SAM3-SHUFFLED controls that must collapse.
"""
import json, math, sys, time, glob
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor")
sys.path.insert(0, "/home/nvidia/qwendrive/v2/mapq")
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import sam3_smoke as S
import mapq_core as mc

PAINT_PROMPTS = ["lane marking", "road marking"]
ZEBRA_PROMPTS = ["crosswalk", "zebra crossing"]
SCORE_MIN = 0.30
TOL = 3                                          # cells = 0.45 m


def ground_grid(pts):
    """2 m lower-envelope ground height from LiDAR (ego frame): 10th percentile of the lowest returns."""
    p = pts[np.hypot(pts[:, 0], pts[:, 1]) < 45]
    key = (np.floor((p[:, 0] + 50) / 2).astype(int) * 50 + np.floor((p[:, 1] + 50) / 2).astype(int))
    order = np.argsort(key, kind="stable"); k, z = key[order], p[order, 2]
    grid = np.full(2500, np.nan)
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    for s, e in zip(starts, np.r_[starts[1:], len(k)]):
        if e - s >= 5:
            grid[k[s]] = np.percentile(z[s:e], 10)
    fallback = float(np.nanmedian(grid)) if np.isfinite(grid).any() else -0.3
    return grid, fallback


def lift(mask, K, R, t, grid, fallback, stride=2):
    """mask (H, W) bool in native pixels -> (N, 2) ego-frame ground points."""
    v, u = np.nonzero(mask[::stride, ::stride]); u = u * stride + stride / 2; v = v * stride + stride / 2
    if not len(u):
        return np.zeros((0, 2))
    rays = np.c_[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))] @ R.T
    ok = rays[:, 2] < -1e-3
    rays = rays[ok]
    z = np.full(len(rays), fallback)
    for _ in range(3):
        s = (z - t[2]) / rays[:, 2]
        xy = t[:2] + rays[:, :2] * s[:, None]
        ci = np.floor((xy[:, 0] + 50) / 2).astype(int); cj = np.floor((xy[:, 1] + 50) / 2).astype(int)
        inb = (ci >= 0) & (ci < 50) & (cj >= 0) & (cj < 50)
        zz = np.full(len(rays), fallback); idx = ci[inb] * 50 + cj[inb]
        zz[inb] = np.where(np.isfinite(grid[idx]), grid[idx], fallback)
        z = zz
    keep = (s > 0) & (np.abs(xy[:, 0]) <= 30) & (np.abs(xy[:, 1]) <= 15)
    return xy[keep]


def raster(xy):
    m = np.zeros((200, 400), bool)
    i = np.floor((xy[:, 1] + 15) / 0.15).astype(int); j = np.floor((xy[:, 0] + 30) / 0.15).astype(int)
    k = (i >= 0) & (i < 200) & (j >= 0) & (j < 400); m[i[k], j[k]] = True
    return m


def observed_ground(Ks, Rs, ts, sizes, grid, fallback):
    """Map cells whose ground point projects inside at least one camera image (the frusta)."""
    ii, jj = np.meshgrid(np.arange(200), np.arange(400), indexing="ij")
    x = -30 + (jj.ravel() + 0.5) * 0.15; y = -15 + (ii.ravel() + 0.5) * 0.15
    ci = np.clip(np.floor((x + 50) / 2).astype(int), 0, 49); cj = np.clip(np.floor((y + 50) / 2).astype(int), 0, 49)
    z = np.where(np.isfinite(grid[ci * 50 + cj]), grid[ci * 50 + cj], fallback)
    P = np.c_[x, y, z]
    seen = np.zeros(len(P), bool)
    for K, R, t, (w, h) in zip(Ks, Rs, ts, sizes):
        c = (P - t) @ R                                          # ego -> camera (R is cam->ego)
        front = c[:, 2] > 0.5
        uv = (c[front, :2] / c[front, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.flatnonzero(front)[inside]; seen[idx] = True
    return seen.reshape(200, 400) & (np.hypot(x, y).reshape(200, 400) <= 30)


def sam3_masks(proc, img, prompts):
    state = proc.set_image(img)
    union = np.zeros((img.height, img.width), bool); n = 0
    for pr in prompts:
        out = proc.set_text_prompt(state=state, prompt=pr)
        if out.get("scores") is None:
            continue
        sc = out["scores"].float().cpu().numpy().reshape(-1)
        ms = out["masks"].cpu().numpy().reshape(len(sc), img.height, img.width) if len(sc) else None
        for k in np.flatnonzero(sc >= SCORE_MIN):
            union |= ms[k]; n += 1
    return union, n


def fuse(qmap, paint, zebra, observed):
    fused = qmap.copy()
    roadish = ndimage.binary_dilation(np.isin(qmap, (1, 2, 4)), iterations=4)
    q_paint_unobserved = np.isin(qmap, (2, 4)) & ~observed
    fused[np.isin(fused, (2, 4)) & observed] = 1                 # where cameras looked, SAM3 decides paint
    fused[zebra & roadish] = 4
    fused[paint & roadish] = 2
    fused[q_paint_unobserved] = qmap[q_paint_unobserved]
    return fused


def pr(pred_paint, true_paint, observed):
    near_true = ndimage.binary_dilation(true_paint, iterations=TOL)
    near_pred = ndimage.binary_dilation(pred_paint, iterations=TOL)
    pp = pred_paint & observed; tp = true_paint & observed
    return int((pp & near_true).sum()), int(pp.sum()), int((tp & near_pred).sum()), int(tp.sum())


def run_demo(proc):
    D = Path("/home/nvidia/qwendrive/qwen-drive/data/demo/perception"); A0 = Path("/home/nvidia/qwendrive/ctrl/A0_out")
    arms = {k: [0, 0, 0, 0] for k in ("QWEN", "SAM3", "FUSED", "SAM3_MIRRORED", "SAM3_SHUFFLED")}
    frames, keep = [], {}
    for fd in sorted(p for p in D.iterdir() if p.is_dir()):
        fr = json.loads((fd / "frame.json").read_text())
        if fr["dataset_type"] != "nuplan":
            continue
        c = np.load(fd / "calib.npz"); g = np.load(fd / "gt.npz"); q = np.load(A0 / f"{fd.name}.npz")["map"].astype(int)
        pts = np.load(fd / "lidar.npy").astype(np.float64)           # lidar == ego on nuPlan
        grid, fb = ground_grid(pts)
        paint = np.zeros((200, 400), bool); zebra = np.zeros((200, 400), bool); npaint = nzebra = 0
        Ks, Rs, ts, sizes = [], [], [], []
        for i, cam in enumerate(fr["cam_order"]):
            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
            Ks.append(K); Rs.append(R); ts.append(t); sizes.append(img.size)
            m, n = sam3_masks(proc, img, PAINT_PROMPTS); npaint += n; paint |= raster(lift(m, K, R, t, grid, fb))
            m, n = sam3_masks(proc, img, ZEBRA_PROMPTS); nzebra += n; zebra |= raster(lift(m, K, R, t, grid, fb))
        observed = observed_ground(Ks, Rs, ts, sizes, grid, fb)
        paint = ndimage.binary_closing(paint, iterations=1); zebra = ndimage.binary_closing(zebra, iterations=2)
        frames.append((fd.name, g["map"].astype(int), q, paint, zebra, observed))
        print(f"  {fd.name[:8]}: sam3 paint instances {npaint}, zebra {nzebra}, observed cells {int(observed.sum())}", flush=True)
    for idx, (tok, gm, q, paint, zebra, observed) in enumerate(frames):
        true_paint = np.isin(gm, (2, 4))
        fused = fuse(q, paint, zebra, observed)
        sh = frames[(idx + 1) % len(frames)]
        res = {"QWEN": np.isin(q, (2, 4)), "SAM3": paint | zebra, "FUSED": np.isin(fused, (2, 4)),
               "SAM3_MIRRORED": (paint | zebra)[::-1, :], "SAM3_SHUFFLED": sh[3] | sh[4]}
        for arm, pm in res.items():
            a = pr(pm, true_paint, observed)
            for k in range(4):
                arms[arm][k] += a[k]
        keep[tok] = {"fused_paint_cells": int(np.isin(fused, (2, 4)).sum()), "true_paint_cells_observed": int((true_paint & observed).sum())}
    out = {arm: {"precision": round(v[0] / max(v[1], 1), 4), "recall": round(v[2] / max(v[3], 1), 4), "n_pred": v[1], "n_true": v[3]} for arm, v in arms.items()}
    for arm, d in out.items():
        f1 = 2 * d["precision"] * d["recall"] / max(d["precision"] + d["recall"], 1e-9)
        d["f1"] = round(f1, 4)
        print(f"  {arm:14s} P {d['precision']:.3f}  R {d['recall']:.3f}  F1 {f1:.3f}  (pred cells {d['n_pred']}, true cells {d['n_true']})")
    Path("sam3_demo_paint.json").write_text(json.dumps({"arms": out, "frames": keep, "score_min": SCORE_MIN, "tol_m": TOL * 0.15}, indent=1))
    return frames


if __name__ == "__main__":
    t0 = time.time()
    proc, _ = S.build(conf=0.25)
    print("built %.1fs" % (time.time() - t0), flush=True)
    run_demo(proc)
    print("ZZSAM3-DEMO-DONEZZ %.0fs" % (time.time() - t0))
