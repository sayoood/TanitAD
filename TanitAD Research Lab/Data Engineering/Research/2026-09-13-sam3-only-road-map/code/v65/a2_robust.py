"""MAP_A2 decomposed, and a ROBUST variant A2r (reported beside A2, never instead of it).
A2 (mapq_core): tall = 1.2-3 m above the MINIMUM z of the 2 m cell, outside agent boxes grown 0.5 m, on predicted drivable.
MEASURED 2026-09-13 (ground_ghost_probe.py): 6.7 % of night 2 m cells (day 2.9 %) have a minimum > 0.5 m below their 10th
percentile (returns below the road surface), and 5.3 % of night tall points (day 3.1 %) are road-surface returns.
A2r: ground = 10th percentile of the 2 m cell (>= 20 returns, else 6 m cell p10); tall = 1.2-3 m above it; agent boxes grown
1.0 m; ego footprint removed as A2. Decomposition of A2 hits: road-surface return (|z - p10| <= 0.3 m), within 1.5 m of an
agent box, and the rest; plus the share of A2r hits whose 0.5 m world cell is hit in >= 3 frames (persistent = static).
Usage: a2_robust.py <c8> <npz dir> <label=render dir> [...]"""
import json, sys
from pathlib import Path
HERE = Path("/home/nvidia/sam3map/eval")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "stub"))
import numpy as np
V2 = Path("/home/nvidia/qwendrive/v2"); D = Path("/home/nvidia/sam3map/data")
import mapq_ours as mo
import mapq_core as mc
mo.V2, mo.PRED, mo.LIDAR = V2, D / "preds", D
mo.EGO_ZIP, mo.EXT = D / "egomotion.chunk_0768.zip", D / "sensor_extrinsics.chunk_0768.parquet"
import sam3map_score as S

c8, npz_dir = sys.argv[1], Path(sys.argv[2])
variants = [(a.split("=", 1)[0], Path(a.split("=", 1)[1])) for a in sys.argv[3:]]
variants = [(l, d) for l, d in variants if (d / "worldmap.npz").exists()]
clip = mo.Clip(c8)
W = {l: dict(np.load(d / "worldmap.npz", allow_pickle=True)) for l, d in variants}


def in_boxes(p, boxes, grow):
    m = np.zeros(len(p), bool)
    for b in boxes:
        c, s = np.cos(-b[6]), np.sin(-b[6]); dx, dy = p[:, 0] - b[0], p[:, 1] - b[1]
        lx, ly = dx * c - dy * s, dx * s + dy * c
        m |= (np.abs(lx) <= b[3] / 2 + grow) & (np.abs(ly) <= b[4] / 2 + grow)
    return m


acc = {l: {"A2_hits": 0, "A2_n": 0, "surface": 0, "near_agent": 0, "A2r_hits": 0, "A2r_n": 0, "cells": {}} for l in W}
for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz")):
    z = np.load(f, allow_pickle=True); tok = str(z["tok"]); T = z["T_world_rig"]
    meta = json.loads((V2 / f"seq_{c8}" / tok / "meta.json").read_text(encoding="utf-8"))
    pts = np.load(D / "sweeps" / c8 / f"{int(np.abs(clip.mid - meta['t_ref_us']).argmin()):04d}.npy")
    g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz"); boxes = g["boxes"]
    tall = mc.split_points(pts, boxes)[4]
    p = pts[np.hypot(pts[:, 0], pts[:, 1]) <= mc.R_MAX + 2.0]
    key = (np.floor((p[:, 0] + 64) / 2).astype(int) * 64 + np.floor((p[:, 1] + 64) / 2).astype(int))
    key6 = (np.floor((p[:, 0] + 66) / 6).astype(int) * 23 + np.floor((p[:, 1] + 66) / 6).astype(int))

    def pct(keys, nbins):
        out = np.full(nbins, np.nan); cnt = np.bincount(keys, minlength=nbins)
        order = np.argsort(keys, kind="stable"); ks, zs = keys[order], p[order, 2]
        st = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]; en = np.r_[st[1:], len(ks)]
        for a, b in zip(st, en):
            out[ks[a]] = np.percentile(zs[a:b], 10)
        return out, cnt
    p10, cnt = pct(key, 64 * 64 + 64); p10_6, _ = pct(key6, 23 * 23 + 23)
    gr = np.where(cnt[key] >= 20, p10[key], p10_6[key6])
    ego = (p[:, 0] > -2.0) & (p[:, 0] < 5.0) & (np.abs(p[:, 1]) < 1.5)
    tall_r = p[(~in_boxes(p, boxes, 1.0)) & (~ego) & (p[:, 2] > gr + 1.2) & (p[:, 2] < gr + 3.0)]
    kt = (np.floor((tall[:, 0] + 64) / 2).astype(int) * 64 + np.floor((tall[:, 1] + 64) / 2).astype(int))
    k6t = (np.floor((tall[:, 0] + 66) / 6).astype(int) * 23 + np.floor((tall[:, 1] + 66) / 6).astype(int))
    gt_ = np.where(cnt[kt] >= 20, p10[kt], p10_6[k6t])
    surface = np.abs(tall[:, 2] - gt_) <= 0.3
    near_ag = in_boxes(tall, boxes, 1.5)
    for l, wm in W.items():
        gtm = S.worldmap_raster(wm, T)
        c_, k_ = mc.map_cls(gtm, tall[:, :2]); on = c_ == 1
        a = acc[l]; a["A2_hits"] += int(on.sum()); a["A2_n"] += len(c_)
        a["surface"] += int((surface[k_][on]).sum()); a["near_agent"] += int((near_ag[k_][on] & ~surface[k_][on]).sum())
        c2, k2 = mc.map_cls(gtm, tall_r[:, :2]); on2 = c2 == 1
        a["A2r_hits"] += int(on2.sum()); a["A2r_n"] += len(c2)
        w = tall_r[k2][on2][:, :2] @ T[:2, :2].T + T[:2, 3]
        for cell in set(zip(np.floor(w[:, 0] / 0.5).astype(int).tolist(), np.floor(w[:, 1] / 0.5).astype(int).tolist())):
            a["cells"][cell] = a["cells"].get(cell, 0) + 1
rep = {}
for l, a in acc.items():
    v = np.array(list(a["cells"].values())) if a["cells"] else np.zeros(0)
    rep[l] = {"A2": round(a["A2_hits"] / max(a["A2_n"], 1), 4), "A2_n": a["A2_n"],
              "A2_hits_share_road_surface_returns": round(a["surface"] / max(a["A2_hits"], 1), 3),
              "A2_hits_share_near_agent_box": round(a["near_agent"] / max(a["A2_hits"], 1), 3),
              "A2r": round(a["A2r_hits"] / max(a["A2r_n"], 1), 4), "A2r_n": a["A2r_n"],
              "A2r_hit_cells_persistent_ge3_frames": round(float((v >= 3).mean()), 3) if len(v) else None}
    print(l, rep[l], flush=True)
Path(f"/home/nvidia/sam3map/a2_robust_{c8}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print("ZZA2R-DONEZZ")
