"""Qwen-Drive boxes against OUR obstacle.offline GT: v1 packing vs v2 packing, same instants.

Both packings put their predictions in the RIG frame's BEV (v1: lidar2ego = identity;
v2: lidar2ego is a pure z-shift), and both GT sets carry the same cuboids, so a BEV
centre-distance match compares like with like. Class-agnostic greedy match at 2.0 m,
score >= THR, GT and predictions within R_MAX of the ego.

Also reported, because a single recall hides where the teacher is blind:
  * per class (vehicle / pedestrian+bicycle), and per SECTOR (front |az|<=45,
    side, rear |az|>=135) -- the rear sector is where v2's B0 is ~18 % observed;
  * a LiDAR-SUPPORTED GT subset (>= MIN_PTS points of the frame's lidar.npy inside the
    box) where LiDAR exists -- obstacle.offline also carries occluded tracks, which no
    camera teacher can be asked to see.

⚠️ T0-style perception scoring on 14 frames / 6 clips (legacy) and ~95 frames / clip
(sequences): these are sensitivity readings, not a benchmark. No interval is claimed.
"""
import json, math, sys
from pathlib import Path
import numpy as np

THR = 0.25
R_MAX = 50.0
DIST = 2.0
MIN_PTS = 3
OCC_R = 40.0
VEH, VRU = {0}, {2, 4}


def match(pred_xy, pred_s, gt_xy, dist=DIST):
    used = np.zeros(len(gt_xy), bool)
    tp_pred = np.zeros(len(pred_xy), bool)
    for i in np.argsort(-pred_s):
        if len(gt_xy) == 0:
            break
        d = np.hypot(*(gt_xy - pred_xy[i]).T)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= dist:
            used[j] = True
            tp_pred[i] = True
    return used, tp_pred


def lidar_support(frame_dir: Path, boxes: np.ndarray, z_shift: float) -> np.ndarray | None:
    p = frame_dir / "lidar.npy"
    if not p.exists():
        return None
    pts = np.load(p).astype(np.float64)
    n = np.zeros(len(boxes), int)
    for k, b in enumerate(boxes):
        c, s = math.cos(-b[6]), math.sin(-b[6])
        dx, dy = pts[:, 0] - b[0], pts[:, 1] - b[1]
        lx, ly = dx * c - dy * s, dx * s + dy * c
        z = pts[:, 2] - z_shift                        # rig -> ego
        n[k] = int(((np.abs(lx) <= b[3] / 2 + 0.3) & (np.abs(ly) <= b[4] / 2 + 0.3)
                    & (z >= b[2] - 0.2) & (z <= b[2] + b[5] + 0.3)).sum())
    return n


def score(frames: Path, preds: Path, tokens):
    agg = {"tp": 0, "fp": 0, "fn": 0, "gt": 0,
           "veh_gt": 0, "veh_tp": 0, "vru_gt": 0, "vru_tp": 0,
           "sec": {s: [0, 0] for s in ("front", "side", "rear")},
           "sup_gt": 0, "sup_tp": 0, "sup_frames": 0, "frames": 0,
           "band": {b: [0, 0] for b in ("0-20", "20-35", "35-50")}, "tp4": 0,
           "occ_iou_any": [], "occ_iou_obst": []}
    per = {}
    for tok in tokens:
        fd, pp = frames / tok, preds / f"{tok}.npz"
        if not (fd / "gt.npz").exists() or not pp.exists():
            continue
        g, r = np.load(fd / "gt.npz"), np.load(pp)
        gb, gl = g["boxes"], g["labels"]
        zs = float(np.load(fd / "calib.npz")["lidar2ego"][2, 3]) * -1.0
        gm = np.hypot(gb[:, 0], gb[:, 1]) <= R_MAX
        gb, gl = gb[gm], gl[gm]
        pb, ps = r["boxes"], r["scores"]
        pm = (ps >= THR) & (np.hypot(pb[:, 0], pb[:, 1]) <= R_MAX)
        used, tpp = match(pb[pm, :2], ps[pm], gb[:, :2])
        tp = int(used.sum())
        used4, _ = match(pb[pm, :2], ps[pm], gb[:, :2], dist=4.0)
        agg["tp4"] += int(used4.sum())
        rg = np.hypot(gb[:, 0], gb[:, 1])
        for b, (lo, hi) in {"0-20": (0, 20), "20-35": (20, 35), "35-50": (35, 50.001)}.items():
            m = (rg >= lo) & (rg < hi)
            agg["band"][b][0] += int(m.sum()); agg["band"][b][1] += int((used & m).sum())
        agg["tp"] += tp; agg["fp"] += int(pm.sum()) - tp; agg["fn"] += len(gb) - tp; agg["gt"] += len(gb)
        agg["frames"] += 1
        isv = np.isin(gl, list(VEH)); isu = np.isin(gl, list(VRU))
        agg["veh_gt"] += int(isv.sum()); agg["veh_tp"] += int((used & isv).sum())
        agg["vru_gt"] += int(isu.sum()); agg["vru_tp"] += int((used & isu).sum())
        az = np.degrees(np.arctan2(gb[:, 1], gb[:, 0]))
        sec = np.where(np.abs(az) <= 45, "front", np.where(np.abs(az) >= 135, "rear", "side"))
        for s in ("front", "side", "rear"):
            agg["sec"][s][0] += int((sec == s).sum()); agg["sec"][s][1] += int((used & (sec == s)).sum())
        sup = lidar_support(fd, gb, zs)
        if sup is not None:
            agg["sup_frames"] += 1
            agg["sup_gt"] += int((sup >= MIN_PTS).sum()); agg["sup_tp"] += int((used & (sup >= MIN_PTS)).sum())
        if "occ" in g.files and "occ" in r.files and (g["occ"] != 9).any():
            # BEV columns only: v1's grid has its z origin on the road, v2's 0.325 m above it,
            # so a 3D comparison would score the z-shift, not the scene. Ground counts as
            # occupied in BOTH (LiDAR ground = 'driveable' here, road = 'driveable' upstream).
            go = g["occ"].reshape(200, 200, 16); po = r["occ"].reshape(200, 200, 16)
            xx = (np.arange(200) + 0.5) * 0.5 - 50.0
            near = np.hypot(xx[:, None], xx[None, :]) <= OCC_R
            ga, pa = (go != 9).any(axis=2) & near, (po != 9).any(axis=2) & near
            gob = np.isin(go, [0, 1, 2, 3, 4, 5, 6, 8]).any(axis=2) & near
            pob = np.isin(po, [0, 1, 2, 3, 4, 5, 6, 8]).any(axis=2) & near
            agg["occ_iou_any"].append(float((ga & pa).sum() / max((ga | pa).sum(), 1)))
            agg["occ_iou_obst"].append(float((gob & pob).sum() / max((gob | pob).sum(), 1)))
        per[tok] = {"gt": len(gb), "tp": tp, "fp": int(pm.sum()) - tp}
    a = agg
    out = {"frames": a["frames"], "gt_within_50m": a["gt"], "tp": a["tp"], "fp": a["fp"], "fn": a["fn"],
           "precision": round(a["tp"] / max(a["tp"] + a["fp"], 1), 3),
           "recall": round(a["tp"] / max(a["gt"], 1), 3),
           "precision_4m": round(a["tp4"] / max(a["tp"] + a["fp"], 1), 3),
           "recall_4m": round(a["tp4"] / max(a["gt"], 1), 3),
           "recall_by_range_m": {b: {"n": v[0], "recall": round(v[1] / max(v[0], 1), 3)} for b, v in a["band"].items()},
           "recall_vehicle": round(a["veh_tp"] / max(a["veh_gt"], 1), 3), "n_vehicle": a["veh_gt"],
           "recall_vru": round(a["vru_tp"] / max(a["vru_gt"], 1), 3), "n_vru": a["vru_gt"],
           "recall_by_sector": {s: {"n": v[0], "recall": round(v[1] / max(v[0], 1), 3)} for s, v in a["sec"].items()},
           "lidar_supported": {"frames": a["sup_frames"], "n": a["sup_gt"],
                               "recall": round(a["sup_tp"] / max(a["sup_gt"], 1), 3)},
           "occ_vs_lidar_bev_iou": {"frames": len(a["occ_iou_any"]), "radius_m": OCC_R,
                                    "any_mean": round(float(np.mean(a["occ_iou_any"])), 4) if a["occ_iou_any"] else None,
                                    "obstacle_mean": round(float(np.mean(a["occ_iou_obst"])), 4) if a["occ_iou_obst"] else None,
                                    "note": "LiDAR pseudo-occupancy is ONE sweep, not semantic GT; a visual-reference agreement, not an occupancy score"},
           "per_frame": per}
    return out


if __name__ == "__main__":
    Q = Path("/home/nvidia/qwendrive")
    res = {"thr": THR, "r_max_m": R_MAX, "match_dist_m": DIST}
    legacy = sorted(p.name for p in (Q / "v2" / "frames_legacy").iterdir() if p.is_dir())
    # ⭐ v1 predictions are scored against the v2 frame dirs' GT: same cuboids, same
    # instants, same order (identical builder logic and 60 ms tolerance), BEV identical
    # (v2 only shifts z) -- so both packings meet the IDENTICAL GT set and LiDAR support.
    res["legacy_v1_packing"] = score(Q / "v2" / "frames_legacy", Q / "out", legacy)
    res["legacy_v2_packing"] = score(Q / "v2" / "frames_legacy", Q / "v2" / "out_legacy", legacy)
    for sd in sorted((Q / "v2").glob("seq_*")):
        if sd.is_dir():
            toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
            res[f"{sd.name}_v2"] = score(sd, Q / "v2" / f"out_{sd.name}", toks)
    (Q / "v2" / "ours_metrics.json").write_text(json.dumps(res, indent=1))
    hdr = f"{'set':22s} {'frames':>6s} {'GT':>5s} {'P':>6s} {'R':>6s} {'R_veh':>6s} {'R_vru':>6s} {'R_front':>7s} {'R_side':>6s} {'R_rear':>6s} {'R_lidar':>7s} {'R@4m':>6s} {'R0-20':>6s} {'R20-35':>6s} {'R35-50':>6s} {'occIoU':>6s} {'obsIoU':>6s}"
    print(hdr)
    for k, v in res.items():
        if not isinstance(v, dict):
            continue
        s = v["recall_by_sector"]
        print(f"{k:22s} {v['frames']:6d} {v['gt_within_50m']:5d} {v['precision']:6.3f} {v['recall']:6.3f} "
              f"{v['recall_vehicle']:6.3f} {v['recall_vru']:6.3f} {s['front']['recall']:7.3f} {s['side']['recall']:6.3f} "
              f"{s['rear']['recall']:6.3f} {v['lidar_supported']['recall']:7.3f} {v['recall_4m']:6.3f} "
              f"{v['recall_by_range_m']['0-20']['recall']:6.3f} {v['recall_by_range_m']['20-35']['recall']:6.3f} {v['recall_by_range_m']['35-50']['recall']:6.3f} "
              f"{(v['occ_vs_lidar_bev_iou']['any_mean'] or float('nan')):6.3f} {(v['occ_vs_lidar_bev_iou']['obstacle_mean'] or float('nan')):6.3f}")
    print("ZZOURS-METRICS-DONEZZ")
