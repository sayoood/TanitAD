"""v3 refinement of the SAM3-only map, derived EXACTLY from the v3 extractor's per-view class rasters.

Same contract as sam3map_refine.py (v2.1): re-lifting the stored 960x540 rasters must reproduce the extractor's points
AND observation ranges bit for bit on every frame before any rule is applied; a frame failing that CONTROL is not
refined. v3 classes: 1 drivable, 2 line, 3 crosswalk, 4 arrow/text, 5 edge, 6 hatched area; drivable points are lifted
from classes {1, 2, 3, 4, 6} (edges are not drivable).
  R1 ego bonnet     as v2.1 (SAM3 "car hood", 8 frames, majority, dilated 5 px)
  R2 ego artefact   as v2.1 (component >= 50 % inside the patch zone) with ONE addition: a band pixel also joins the
                    zone when the ego MOVED >= 8 m between its first and last marking frame -- ground paint cannot stay
                    at one pixel while the car travels 8 m (night clip: a reflection line held frames 0-28 and escaped
                    the 30 %-of-clip spread rule at 29/96)
  R3 thin edges     edge pixels survive only if 4-adjacent to a drivable-or-paint pixel
Output: /home/nvidia/sam3map/<c8>_v3/<jjj>.npz (+ refine_v3_<c8>.json)
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image
import sam3_paint as P
from sam3map_refine import bonnet_mask, K_FRAMES, HOOD_SCORE, HOOD_MIN_PX, HOOD_BOTTOM, DILATE_PX, BAND_PX, PATCH_MIN_FRAMES, PATCH_SPREAD, PATCH_COLS, PATCH_INSIDE

import os
V2 = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/qwendrive/v2"))
VIEWS = os.environ.get("SAM3MAP_VIEWS", "CAM_F0,CAM_L0,CAM_R0,CAM_L1,CAM_R1,CAM_L2,CAM_R2").split(",")
PAINT = (2, 3, 4, 6)
DRIVE = (1, 2, 3, 4, 6)
MOVE_M = 8.0


def relift(cls_small, K, R, t, grid, fb, T):
    full = np.repeat(np.repeat(cls_small, 2, axis=0), 2, axis=1)
    pts, rng = {}, {}
    for k in range(1, 7):
        m = np.isin(full, DRIVE) if k == 1 else (full == k)
        if m.any():
            xy = P.lift(m, K, R, t, grid, fb, stride=2)
            if len(xy):
                pts[k] = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32)
                rng[k] = np.hypot(xy[:, 0] - t[0], xy[:, 1] - t[1]).astype(np.float16)
    return pts, rng


def cat(lst, shape, dtype):
    return np.concatenate(lst) if lst else np.zeros(shape, dtype)


def main():
    c8 = sys.argv[1]
    tag = os.environ.get("SAM3MAP_TAG", "v3")                  # v3.1 extractions live in <c8>_v31raw
    src = Path(f"/home/nvidia/sam3map/{c8}_{tag}raw"); dst = Path(f"/home/nvidia/sam3map/{c8}_{tag}"); dst.mkdir(exist_ok=True)
    sd = V2 / f"seq_{c8}"
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    files = sorted(src.glob("[0-9][0-9][0-9].npz"))
    t0 = time.time()
    import sam3map_refine as R2mod
    R2mod.V2 = V2                                              # bonnet_mask reads the sequence root from its own module
    bonnet, found = bonnet_mask(c8, toks, sd)
    print(f"bonnet: {int(bonnet.sum())} px ({100 * bonnet.mean():.1f} % of F0); sampled (j, score, px): {found}", flush=True)
    cols = np.nonzero(bonnet.any(axis=0))[0]
    band = np.zeros_like(bonnet)
    for x in cols:
        top = int(np.argmax(bonnet[:, x])); band[max(0, top - BAND_PX): top, x] = True
    band[:, : int(PATCH_COLS[0] * 960)] = False; band[:, int(PATCH_COLS[1] * 960):] = False
    N = len(files)
    pos = np.zeros((N, 2))
    cnt = np.zeros((540, 960), np.int32); first = np.full((540, 960), N, np.int32); last = np.full((540, 960), -1, np.int32)
    for n, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        pos[n] = d["T_world_rig"][:2, 3]
        m = np.isin(d["cls_CAM_F0"], PAINT) & ~bonnet
        cnt += m; first[m & (first == N)] = n; last[m] = n
    spread = np.where(cnt > 0, last - first, 0)
    fi, la = first.clip(0, N - 1), last.clip(0, N - 1)
    moved = np.hypot(pos[la, 0] - pos[fi, 0], pos[la, 1] - pos[fi, 1]) * (cnt > 0)
    zone = band & (cnt >= PATCH_MIN_FRAMES) & ((spread >= PATCH_SPREAD * N) | (moved >= MOVE_M))
    zone_d = cv2.dilate(zone.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    print(f"patch zone: {int(zone.sum())} px (spread rule {int((band & (cnt >= PATCH_MIN_FRAMES) & (spread >= PATCH_SPREAD * N)).sum())}, "
          f"moved rule {int((band & (cnt >= PATCH_MIN_FRAMES) & (moved >= MOVE_M)).sum())})", flush=True)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    tot = {"control_identical": 0, "control_failed": 0, "R1_px": 0, "R2_components": 0, "R2_px": 0, "R3_px": 0}
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        tok = str(d["tok"]); fd = sd / tok; T = d["T_world_rig"]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        lp = fd / "lidar.npy"
        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        cams = {cam: (c["cam_intrinsic"][fr["cam_order"].index(cam)], c["sensor2lidar_rotation"][fr["cam_order"].index(cam)],
                      c["sensor2lidar_translation"][fr["cam_order"].index(cam)]) for cam in VIEWS}
        accp, accr = {k: [] for k in range(1, 7)}, {k: [] for k in range(1, 7)}
        for cam in VIEWS:
            p_, r_ = relift(d[f"cls_{cam}"], *cams[cam], grid, fb, T)
            for k in p_:
                accp[k].append(p_[k]); accr[k].append(r_[k])
        same = all(np.array_equal(cat(accp[k], (0, 2), np.float32), d[f"pts_{k}"]) and np.array_equal(cat(accr[k], (0,), np.float16), d[f"rng_{k}"])
                   for k in range(1, 7))
        if not same:
            tot["control_failed"] += 1
            print(f"  CONTROL FAILED on {f.name}: not refined", flush=True)
            continue
        tot["control_identical"] += 1
        rec = {}; newp, newr = {k: [] for k in range(1, 7)}, {k: [] for k in range(1, 7)}
        for cam in VIEWS:
            cl = d[f"cls_{cam}"].copy()
            r1 = r2c = r2 = 0
            if cam == "CAM_F0":
                r1 = int((cl[bonnet] > 0).sum()); cl[bonnet] = 0
                mk = np.isin(cl, PAINT).astype(np.uint8)
                n, lab = cv2.connectedComponents(mk, connectivity=8)
                size = np.bincount(lab.ravel(), minlength=n); inside = np.bincount(lab[zone_d].ravel(), minlength=n)
                hit = np.nonzero((inside >= PATCH_INSIDE * size) & (np.arange(n) > 0))[0]
                if len(hit):
                    kill = np.isin(lab, hit); r2c = int(len(hit)); r2 = int(kill.sum()); cl[kill] = 0
            e = cl == 5
            keep = e & (cv2.dilate(np.isin(cl, DRIVE).astype(np.uint8), cross) > 0)
            r3 = int((e & ~keep).sum()); cl[e & ~keep] = 0
            d[f"cls_{cam}"] = cl
            rec[cam] = {"R1_px": r1, "R2_components": r2c, "R2_px": r2, "R3_px": r3}
            for q, v in (("R1_px", r1), ("R2_components", r2c), ("R2_px", r2), ("R3_px", r3)):
                tot[q] += v
            p_, r_ = relift(cl, *cams[cam], grid, fb, T)
            for k in p_:
                newp[k].append(p_[k]); newr[k].append(r_[k])
        st = json.loads(str(d["stats"])); st["_refine_v2"] = rec
        d["stats"] = json.dumps(st)
        for k in range(1, 7):
            d[f"pts_{k}"] = cat(newp[k], (0, 2), np.float32); d[f"rng_{k}"] = cat(newr[k], (0,), np.float16)
        np.savez_compressed(dst / f.name, **d)
    json.dump({"bonnet_px_960x540": int(bonnet.sum()), "bonnet_sampled": found, "patch_zone_px": int(zone.sum()), **tot,
               "rules": {"MOVE_M": MOVE_M, "PATCH_INSIDE": PATCH_INSIDE, "PATCH_SPREAD": PATCH_SPREAD, "BAND_PX": BAND_PX}},
              open(dst.parent / f"refine_{tag}_{c8}.json", "w"), indent=1)
    print(f"totals {tot}  {time.time() - t0:.0f}s", flush=True)
    print("ZZREFINE3-DONEZZ")


if __name__ == "__main__":
    main()
