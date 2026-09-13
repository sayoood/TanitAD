"""Score Qwen-Drive predictions against the packed ground truth, per variant.

Detection: class-agnostic BEV centre-distance matching (greedy by score) at 2.0 m, in
the EGO frame (predictions come back in the lidar frame and are converted through
lidar2ego exactly as the upstream visualizer does). GT and predictions restricted
to the 51.2 m detection square.
Map: per-class IoU over the 5 foreground classes. Occupancy: geometric IoU
(non-empty vs empty) and per-class IoU over the 9 non-empty classes.

⚠️ n is 4 nuPlan frames + 2 nuScenes frames. These are sensitivity readings for
effects expected to be LARGE (a 2.3x image-scale change), not benchmark numbers.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/nvidia/qwendrive/qwen-drive/src")
from qwen_drive_perception import geometry
from qwen_drive_perception.configuration_perception import MAP_CLASS_NAMES, OCC_CLASS_NAMES

CTRL = Path(sys.argv[1])
THR = (0.25, 0.35)
EMPTY = len(OCC_CLASS_NAMES) - 1


def det_match(pred_xy, pred_s, gt_xy, dist=2.0):
    order = np.argsort(-pred_s)
    used = np.zeros(len(gt_xy), bool)
    tp = 0
    for i in order:
        if len(gt_xy) == 0:
            break
        d = np.hypot(*(gt_xy - pred_xy[i]).T)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= dist:
            used[j] = True
            tp += 1
    return tp, len(pred_xy) - tp, int((~used).sum())


rows = {}
for var in [v for v in ("A0", "A0r", "A1", "A2", "A3", "A4", "A5", "A6") if (CTRL / v).is_dir()]:
    out_dir = CTRL / f"{var}_out"
    for fd in sorted(p for p in (CTRL / var).iterdir() if p.is_dir()):
        pp = out_dir / f"{fd.name}.npz"
        if not pp.exists():
            continue
        r = np.load(pp)
        gt = np.load(fd / "gt.npz")
        l2e = np.load(fd / "calib.npz")["lidar2ego"]
        typ = json.loads((fd / "frame.json").read_text())["dataset_type"]
        rec = {"type": typ, "n_cam": len(json.loads((fd / "frame.json").read_text())["cam_order"])}
        pb = geometry.lidar_to_ego_boxes(torch.as_tensor(r["boxes"], dtype=torch.float32),
                                         torch.as_tensor(l2e, dtype=torch.float32)).numpy()
        gb = gt["boxes"]
        ing = (np.abs(gb[:, 0]) <= 51.2) & (np.abs(gb[:, 1]) <= 51.2)
        for t in THR:
            k = (r["scores"] >= t) & (np.abs(pb[:, 0]) <= 51.2) & (np.abs(pb[:, 1]) <= 51.2)
            tp, fp, fn = det_match(pb[k, :2], r["scores"][k], gb[ing, :2])
            rec[f"det@{t}"] = {"tp": tp, "fp": fp, "fn": fn, "n_gt": int(ing.sum())}
        pm, gm = r["map"].astype(int), gt["map"].astype(int)
        rec["map_iou"] = {}
        for c in range(1, len(MAP_CLASS_NAMES)):
            u = ((pm == c) | (gm == c)).sum()
            if (gm == c).sum() > 0:
                rec["map_iou"][MAP_CLASS_NAMES[c]] = round(float(((pm == c) & (gm == c)).sum() / max(u, 1)), 4)
        po, go = r["occ"].astype(int).reshape(200, 200, 16), gt["occ"].astype(int).reshape(200, 200, 16)
        u = ((po != EMPTY) | (go != EMPTY)).sum()
        rec["occ_geo_iou"] = round(float(((po != EMPTY) & (go != EMPTY)).sum() / max(u, 1)), 4)
        rec["occ_iou"] = {}
        for c in range(EMPTY):
            if (go == c).sum() > 0:
                u = ((po == c) | (go == c)).sum()
                rec["occ_iou"][OCC_CLASS_NAMES[c]] = round(float(((po == c) & (go == c)).sum() / max(u, 1)), 4)
        rec["n_pred@0.25"] = int((r["scores"] >= 0.25).sum())
        rec["pred_occ_nonempty_frac"] = round(float((po != EMPTY).mean()), 5)
        rec["gt_occ_nonempty_frac"] = round(float((go != EMPTY).mean()), 5)
        rows.setdefault(var, {})[fd.name] = rec

# ---- aggregate over the nuPlan frames, per variant --------------------------------
summ = {}
for var, frs in rows.items():
    nu = {k: v for k, v in frs.items() if v["type"] == "nuplan"}
    if not nu:
        continue
    s = {"n_frames": len(nu)}
    for t in THR:
        tp = sum(v[f"det@{t}"]["tp"] for v in nu.values())
        fp = sum(v[f"det@{t}"]["fp"] for v in nu.values())
        fn = sum(v[f"det@{t}"]["fn"] for v in nu.values())
        s[f"det@{t}"] = {"tp": tp, "fp": fp, "fn": fn,
                         "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
    for key in ("map_iou", "occ_iou"):
        cls = sorted({c for v in nu.values() for c in v[key]})
        s[key] = {c: round(float(np.mean([v[key][c] for v in nu.values() if c in v[key]])), 4) for c in cls}
        s[key + "_mean"] = round(float(np.mean(list(s[key].values()))), 4) if s[key] else None
    s["occ_geo_iou"] = round(float(np.mean([v["occ_geo_iou"] for v in nu.values()])), 4)
    summ[var] = s

(CTRL / "metrics.json").write_text(json.dumps({"per_frame": rows, "nuplan_summary": summ}, indent=1))
print(f"{'variant':6s} {'P@.25':>6s} {'R@.25':>6s} {'tp/fp/fn':>12s} {'mapIoU':>7s} {'occGeo':>7s} {'occmIoU':>8s}")
for var, s in summ.items():
    d = s["det@0.25"]
    print(f"{var:6s} {d['precision']:6.3f} {d['recall']:6.3f} {str(d['tp'])+'/'+str(d['fp'])+'/'+str(d['fn']):>12s} "
          f"{s['map_iou_mean']:7.4f} {s['occ_geo_iou']:7.4f} {s['occ_iou_mean']:8.4f}")
for var, frs in rows.items():
    for k, v in frs.items():
        if v["type"] == "nuscenes":
            d = v["det@0.25"]
            print(f"nuscenes {var} {k[:8]}: tp/fp/fn {d['tp']}/{d['fp']}/{d['fn']}  map {np.mean(list(v['map_iou'].values())):.4f}  occGeo {v['occ_geo_iou']:.4f}")
print("ZZMETRICS-DONEZZ")
