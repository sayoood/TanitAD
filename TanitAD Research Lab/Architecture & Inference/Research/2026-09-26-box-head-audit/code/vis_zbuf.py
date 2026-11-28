#!/usr/bin/env python3
"""vis_zbuf.py -- box-head audit task 0b: how much of each GT target can the FRONT CAMERA actually see?

An EXACT per-pixel z-buffer in the clip's own canonical camera (416x1024 CYLINDRICAL, f_ref 488.924, the
per-clip extrinsics the renderer and the trainer's RigCamera use): every pixel's ray is intersected with every
GT cuboid (ray / oriented-box slab test), the nearest hit owns the pixel. Per GT box:

  n_full   pixels its cuboid covers on an EXTENDED cylinder (azimuth to +-120 deg, rows -300..H+300), i.e. its
           whole silhouette whether or not the camera frame contains it
  n_img    of those, inside the real 416x1024 frame AND on an OBSERVED pixel (the clip's own f-theta sensor
           reaches it: canonical ray -> fw_poly -> inside the 1920x1080 native frame)
  n_vis    of those, pixels where THIS box is the nearest cuboid
  vis_frac = n_vis / n_full   (the PI's question: visible AND inside the image)
  unocc    = n_vis / n_img    (visible share of the in-image part)
  vis_rows = vertical extent (px) of its visible pixels

Occluders = every valid GT row (targets AND rows the trainer's filter removed). ⚠️ An UPPER bound on
visibility: walls, trees, poles, the ego hood and unlabelled objects are not in the GT.

Modes:
  --mode gtval     the 5 GT-validation clips, every eval window (the trainer's V3Dataset._agent_item block)
  --mode clipgrid  the bha_dump clipgrid set (139 eval clips x 4 labelled windows), rows from the dump
  --mode lidar     the LiDAR clip (--lidar-c8, sha12 73495082f98b): visibility beside the LiDAR returns in each box
sha12 only in every output.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

W, H, F = 1024, 416, 488.92398517830253
PAD_U, PAD_V = 512, 300
CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person", "rider", "stroller",
           "animal", "protruding_object")


def sha12(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def R_from_quat(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def observed_mask(poly, icx, icy, iw, ih):
    """[H, W] bool: canonical pixels whose ray lands inside the native f-theta frame."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    phi = (uu - (W - 1) / 2.0) / F
    yn = (vv - (H - 1) / 2.0) / F
    x, y, z = np.sin(phi), yn, np.cos(phi)
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    r = np.zeros_like(th)
    for c in reversed(poly):
        r = r * th + c
    k = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
    u, v = icx + x * k, icy + y * k
    return (u >= 0) & (u <= iw - 1) & (v >= 0) & (v <= ih - 1) & (z > 0)


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def corners(cx, cy, cz, l, w, h, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    xy = loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    return np.concatenate([np.c_[xy, np.full(4, cz - h / 2)], np.c_[xy, np.full(4, cz + h / 2)]])


def zbuffer(boxes, R, t, obs):
    """boxes: [M, 7] (cx, cy, cz, l, w, h, yaw) rig frame. Returns per-box n_full, n_img, n_vis, vis_rows."""
    M = len(boxes)
    Hx, Wx = H + 2 * PAD_V, W + 2 * PAD_U
    zb = np.full((Hx, Wx), np.inf)
    idb = np.full((Hx, Wx), -1, np.int64)
    n_full = np.zeros(M, np.int64)
    n_img = np.zeros(M, np.int64)
    inimg = np.zeros((Hx, Wx), bool)
    inimg[PAD_V:PAD_V + H, PAD_U:PAD_U + W] = obs
    for i in range(M):
        cx, cy, cz, l, w, h, yaw = [float(v) for v in boxes[i]]
        if not (l > 0 and w > 0 and h > 0):
            continue
        cor = corners(cx, cy, cz, l, w, h, yaw)
        samp = np.concatenate([cor[a][None] + (cor[b] - cor[a])[None] * np.linspace(0, 1, 17)[:, None]
                               for a, b in EDGES])
        pc = (samp - t) @ R
        rho = np.hypot(pc[:, 0], pc[:, 2])
        phi = np.arctan2(pc[:, 0], pc[:, 2])
        cen = (np.array([cx, cy, cz]) - t) @ R
        if abs(math.atan2(cen[0], cen[2])) > math.radians(150) or (np.abs(phi) > math.radians(170)).any():
            continue                                    # behind the camera: not in any front image
        u = (W - 1) / 2.0 + F * phi + PAD_U
        v = (H - 1) / 2.0 + F * pc[:, 1] / np.maximum(rho, 1e-6) + PAD_V
        u0, u1 = int(max(0, math.floor(u.min()) - 1)), int(min(Wx - 1, math.ceil(u.max()) + 1))
        v0, v1 = int(max(0, math.floor(v.min()) - 1)), int(min(Hx - 1, math.ceil(v.max()) + 1))
        if u1 < u0 or v1 < v0:
            continue
        uu, vv = np.meshgrid(np.arange(u0, u1 + 1, dtype=np.float64), np.arange(v0, v1 + 1, dtype=np.float64))
        ph = (uu - PAD_U - (W - 1) / 2.0) / F
        yn = (vv - PAD_V - (H - 1) / 2.0) / F
        d = np.stack([np.sin(ph), yn, np.cos(ph)], -1) @ R.T          # rig-frame ray directions
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        o = t - np.array([cx, cy, cz])
        ob = np.array([cyaw * o[0] + syaw * o[1], -syaw * o[0] + cyaw * o[1], o[2]])
        db = np.stack([cyaw * d[..., 0] + syaw * d[..., 1], -syaw * d[..., 0] + cyaw * d[..., 1], d[..., 2]], -1)
        hs = np.array([l / 2, w / 2, h / 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-hs - ob) / db
            t2 = (hs - ob) / db
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = tmax >= np.maximum(tmin, 0.0)
        dep = np.maximum(tmin, 0.0) * np.linalg.norm(d, axis=-1)
        n_full[i] = int(hit.sum())
        sub_in = inimg[v0:v1 + 1, u0:u1 + 1]
        n_img[i] = int((hit & sub_in).sum())
        zs = zb[v0:v1 + 1, u0:u1 + 1]
        ids = idb[v0:v1 + 1, u0:u1 + 1]
        upd = hit & (dep < zs)
        zs[upd] = dep[upd]
        ids[upd] = i
    own = idb[inimg]
    n_vis = np.bincount(own[own >= 0], minlength=M)[:M] if M else np.zeros(0, np.int64)
    rows = np.zeros(M, np.int64)
    ys = np.nonzero(inimg)[0]
    for i in np.unique(own[own >= 0]):
        yy = ys[own == i]
        rows[i] = int(yy.max() - yy.min() + 1)
    return n_full, n_img, n_vis, rows, idb


def summarise(rows):
    """rows: list of dicts with vis_frac etc. for TARGETS."""
    def dist(v):
        v = np.asarray([x for x in v if x == x], float)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
                "frac_lt_0.10": float((v < 0.10).mean()), "frac_lt_0.30": float((v < 0.30).mean()),
                "frac_lt_0.50": float((v < 0.50).mean()), "frac_zero": float((v == 0).mean())}
    out = {"all_targets": dist([r["vis_frac"] for r in rows]),
           "all_targets_unocc_in_image": dist([r["unocc"] for r in rows]),
           "by_class": {c: dist([r["vis_frac"] for r in rows if r["cls"] == c])
                        for c in sorted(set(r["cls"] for r in rows))},
           "by_range": {}}
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90)):
        out["by_range"][f"{lo}-{hi}m"] = dist([r["vis_frac"] for r in rows if lo <= r["range"] < hi])
    return out


def rule_effect(rows_by_window, rules):
    """Targets per window after each candidate visibility rule."""
    res = {}
    n_w = len(rows_by_window)
    base = [sum(1 for r in rw if r["target"]) for rw in rows_by_window]
    res["baseline_targets_per_window"] = float(np.mean(base)) if n_w else None
    for name, fn in rules.items():
        kept = [sum(1 for r in rw if r["target"] and fn(r)) for rw in rows_by_window]
        res[name] = {"targets_per_window": float(np.mean(kept)) if n_w else None,
                     "frac_of_targets_kept": (float(np.sum(kept)) / max(1, float(np.sum(base))))}
    return res


RULES = {
    "vis_frac>=0.10": lambda r: r["vis_frac"] >= 0.10,
    "vis_frac>=0.30": lambda r: r["vis_frac"] >= 0.30,
    "vis_frac>=0.50": lambda r: r["vis_frac"] >= 0.50,
    "R1: vis_frac>=0.30 & vis_rows>=8": lambda r: r["vis_frac"] >= 0.30 and r["vis_rows"] >= 8,
    "R2: n_vis>=150px & vis_frac>=0.25": lambda r: r["n_vis"] >= 150 and r["vis_frac"] >= 0.25,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("gtval", "clipgrid", "lidar"), required=True)
    ap.add_argument("--tree", default="/home/nvidia/bha_2325/tree")
    ap.add_argument("--code", default="/home/nvidia/bha_2325/code")
    ap.add_argument("--calib-dir", default="/home/nvidia/bha_2325/ov/calib_ship")
    ap.add_argument("--dump-dir", default="/home/nvidia/bha_2325/out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", default="0191487845ef,9f8bedcfb9de,34765c024267,22d47ae7e052,59d98b34ae8b")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--dump-set", default="clipgrid", help="--mode clipgrid: which bha_dump set (clipgrid | train | inrun)")
    ap.add_argument("--lidar-c8", default=None,
                    help="--mode lidar: the 8-hex file prefix of the LiDAR clip ON THOR (runtime only)")
    a = ap.parse_args(argv)
    t0 = time.time()
    import pandas as pd
    D = "/home/nvidia/data"
    table = json.load(open(f"{D}/refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
    by12 = {sha12(c): c for c in table}

    def camera(cid):
        e = table[cid]
        R = R_from_quat(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"]))
        t = np.array([float(e["x"]), float(e["y"]), float(e["z"])])
        ch = int(e.get("chunk", -1))
        ip = glob.glob(f"{a.calib_dir}/camera_intrinsics/camera_intrinsics.chunk_{ch:04d}.parquet") or \
            glob.glob(f"/home/nvidia/sam3map/data/calib/camera_intrinsics.chunk_{ch:04d}.parquet")
        obs, why = np.ones((H, W), bool), "no intrinsics file: image bounds only"
        if ip:
            it = pd.read_parquet(ip[0]).reset_index()
            it = it[(it["clip_id"] == cid) & (it["camera_name"] == "camera_front_wide_120fov")]
            if len(it):
                r = it.iloc[0]
                obs = observed_mask(tuple(float(r[f"fw_poly_{i}"]) for i in range(5)), float(r.cx), float(r.cy),
                                    int(r.width), int(r.height))
                why = "f-theta observed mask"
        return R, t, obs, why

    windows = []           # list of (sha12, window key, [gt rows dict])
    if a.mode == "clipgrid":
        import torch
        rows = []
        for f in sorted(glob.glob(f"{a.dump_dir}/{a.dump_set}_chunk*.pt")):
            rows += torch.load(f, weights_only=False)
        for r in rows:
            if not r["agent_label"]:
                continue
            gt = []
            for j in range(int(r["agent_box"].shape[0])):
                b = r["agent_box"][j].tolist()
                gt.append({"cx": b[0], "cy": b[1], "l": b[2], "w": b[3], "yaw": float(r["agent_yaw"][j]),
                           "cz": float(r["agent_cz"][j]), "h": float(r["agent_h"][j]),
                           "zh": bool(r["agent_zh_mask"][j]), "cls_i": int(r["agent_cls"][j])})
            windows.append((r["sha12"], int(r["t"]), gt))
    elif a.mode == "gtval":
        os.environ["REFCV6_REPO"] = a.tree
        sys.path.insert(0, a.code)
        import refcv6_loader as L
        L.bootstrap()
        import torch  # noqa: F401
        from tanitad.data import v2_dataset as v2d
        tr = L.trainer()
        config = L.load_config("/home/nvidia/refcv6_run/runs/refcv6-r101-s0/config.json")
        model, cfg, targs, _ = L.build_model(config, "/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt.pt",
                                             device="cpu", remap={})
        del model
        want12 = set(a.clips.split(","))
        keep = {by12[s] for s in want12 if s in by12}
        _orig = v2d.build_v2_providers

        def clip_of(ep):
            fp = ep.frames
            from pathlib import Path
            return Path(fp._cache.files[fp._clip]).name.split(".")[0]

        def _only(*aa, **kk):
            return [e for e in _orig(*aa, **kk) if clip_of(e) in keep]
        v2d.build_v2_providers = _only
        try:
            e_ds, e_eps, _ = L.build_eval_dataset(None, cfg, targs, config, with_perception_targets=True)
        finally:
            v2d.build_v2_providers = _orig
        Wn = int(cfg.core.window)
        for wi, (e_i, t) in enumerate(e_ds.index):
            ep = e_eps[e_i]
            it = e_ds._agent_item(ep, t + Wn - 1)
            if not bool(it["agent_label"]):
                continue
            v = it["agent_valid"].numpy().astype(bool)
            gt = []
            for j in np.nonzero(v)[0]:
                b = it["agent_box"][j].tolist()
                gt.append({"cx": b[0], "cy": b[1], "l": b[2], "w": b[3], "yaw": float(it["agent_yaw"][j]),
                           "cz": float(it["agent_cz"][j]), "h": float(it["agent_h"][j]),
                           "zh": bool(it["agent_zh_mask"][j]), "cls_i": int(it["agent_cls"][j])})
            windows.append((sha12(clip_of(ep)), int(t), gt))
    else:
        import lzma
        sys.path.insert(0, os.path.join(a.tree, "stack"))
        from tanitad.data import agent_cuboid_gt as ACG
        lid = sorted(glob.glob(f"/home/nvidia/sam3map/data/{a.lidar_c8}-*.lidar_top_360fov.parquet"))[0]
        cid = os.path.basename(lid).split(".")[0]
        boxes_ov = json.load(open("/home/nvidia/bha_2325/ov/out/ov_boxes_73495082f98b.json"))
        frames = sorted(set(b["f"] for b in boxes_ov))
        recs = {}
        with lzma.open(f"{D}/joins/b1_train_plus_eval_agents.jsonl.xz", "rt", encoding="utf-8") as fh:
            for ln in fh:
                if cid in ln:
                    r = json.loads(ln)
                    if r["clip_id"] == cid and int(r["frame_idx"]) in frames:
                        recs[int(r["frame_idx"])] = r
        j3 = ACG.open_join3d(f"{D}/join3d/b1_train_plus_eval_agents_3d.jsonl.xz", clips={cid})
        nin = {}
        for b in boxes_ov:
            nin.setdefault(b["f"], []).append(b)
        for f in frames:
            r = recs[f]
            tids = [str(d.get("track_id", "")) for d in r["agents"]]
            cz, hh, zm = ACG.zh_for_frame(cid, f + 2, tids, join3d=j3)
            gt = []
            for j, d in enumerate(r["agents"]):
                gt.append({"cx": float(d["cx"]), "cy": float(d["cy"]), "l": float(d["l"]), "w": float(d["w"]),
                           "yaw": float(d["yaw"]), "cz": float(cz[j]), "h": float(hh[j]), "zh": bool(zm[j]),
                           "cls": str(d.get("cls", "")), "n_in_lidar": nin[f][j]["n_in"]})
            windows.append((sha12(cid), f, gt))
        by12[sha12(cid)] = cid
    if a.max_windows:
        windows = windows[: a.max_windows]
    print(f"[vis] {a.mode}: {len(windows)} labelled windows ({time.time() - t0:.0f} s)", flush=True)

    cams = {}
    per_box, per_window = [], []
    for k, (s12, wkey, gt) in enumerate(windows):
        if s12 not in cams:
            cams[s12] = camera(by12[s12])
        R, t, obs, why = cams[s12]
        gt_z = [g for g in gt if g["zh"]]
        bx = np.array([[g["cx"], g["cy"], g["cz"], g["l"], g["w"], g["h"], g["yaw"]] for g in gt_z]) \
            if gt_z else np.zeros((0, 7))
        nf, ni, nv, vr, _ = zbuffer(bx, R, t, obs)
        rw = []
        for i, g in enumerate(gt_z):
            cx, cy = g["cx"], g["cy"]
            tgt = (0 <= cx <= 60) and abs(cy) <= 16 and math.atan2(abs(cy), cx) <= math.radians(60)
            cls = g.get("cls") or (CLASSES[g["cls_i"]] if 0 <= g.get("cls_i", -1) < len(CLASSES) else "unknown")
            row = {"sha12": s12, "w": wkey, "cls": cls, "range": math.hypot(cx, cy), "target": bool(tgt),
                   "n_full": int(nf[i]), "n_img": int(ni[i]), "n_vis": int(nv[i]), "vis_rows": int(vr[i]),
                   "vis_frac": (nv[i] / nf[i]) if nf[i] else float("nan"),
                   "in_img": (ni[i] / nf[i]) if nf[i] else float("nan"),
                   "unocc": (nv[i] / ni[i]) if ni[i] else float("nan")}
            if "n_in_lidar" in g:
                row["n_in_lidar"] = int(g["n_in_lidar"])
            rw.append(row)
        per_box += rw
        per_window.append(rw)
        if k % 100 == 0:
            print(f"[vis] {k}/{len(windows)} windows ({time.time() - t0:.0f} s)", flush=True)
    tg = [r for r in per_box if r["target"]]
    out = {"mode": a.mode, "n_windows": len(per_window), "n_gt_rows_3d": len(per_box), "n_targets": len(tg),
           "n_gt_rows_without_3d_label": int(sum(1 for _, _, gt in windows for g in gt if not g["zh"])),
           "observed_mask_source": sorted(set(v[3] for v in cams.values())),
           "method": "exact per-pixel ray/oriented-box z-buffer, canonical 416x1024 cylinder, per-clip extrinsics "
                     "(the renderer's table) and the clip's f-theta observed mask; occluders = every valid GT row "
                     "with a 3-D label; UPPER bound (static occluders absent)",
           "targets": summarise(tg),
           "targets_per_window_rules": rule_effect(per_window, RULES),
           "wall_s": round(time.time() - t0, 1)}
    if a.mode == "lidar":
        lv = [r for r in per_box if r["target"] and r["vis_frac"] == r["vis_frac"]]
        out["lidar_vs_visibility"] = {
            f"vis<{hi}" if lo == 0 else f"{lo}<=vis<{hi}": {
                "n": sum(1 for r in lv if lo <= r["vis_frac"] < hi),
                "frac_lidar_ge3": (float(np.mean([r["n_in_lidar"] >= 3 for r in lv if lo <= r["vis_frac"] < hi]))
                                   if any(lo <= r["vis_frac"] < hi for r in lv) else None),
                "median_lidar_pts": (float(np.median([r["n_in_lidar"] for r in lv if lo <= r["vis_frac"] < hi]))
                                     if any(lo <= r["vis_frac"] < hi for r in lv) else None)}
            for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01))}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    json.dump(per_box, open(a.out.replace(".json", "_boxes.json"), "w"), default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "targets"}, indent=1, default=float)[:4000], flush=True)
    print(json.dumps(out["targets"]["all_targets"], default=float), flush=True)


if __name__ == "__main__":
    main()
