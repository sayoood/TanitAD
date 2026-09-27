#!/usr/bin/env python3
"""gt_zcensus.py -- box-head audit: are large-vehicle cuboids sunk, measured against the SAME frame's own
ground-standing agents (no rig-plane or LiDAR assumption)?

For every GT row with a 3-D label in the bha_dump sets: bottom = cz - h/2 (rig frame). The reference ground of
a row is the MEDIAN bottom of the frame's ground-standing reference agents (automobile, person, rider; the row
itself excluded) within 15 m (BEV) of it, when at least 2 exist. Slope and pitch move a box and its neighbours
together, so d = bottom - reference is a per-class LABEL offset, not terrain.

Also: the proposed target-side correction for refcv7 ("re-seat": cz' = cz - d for classes whose median |d| >
0.25 m), and its effect on the z targets (how many targets move, by how much, their share of all targets).
sha12 only in the output.
"""
import argparse
import glob
import json
import math

import numpy as np

CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person", "rider", "stroller",
           "animal", "protruding_object")
REF = (0, 5, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/home/nvidia/bha_2325/out")
    ap.add_argument("--sets", default="inrun,clipgrid,train")
    ap.add_argument("--radius-m", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    res = {"radius_m": a.radius_m, "reference_classes": [CLASSES[i] for i in REF], "sets": {}}
    for s in a.sets.split(","):
        rows = []
        for f in sorted(glob.glob(f"{a.dump_dir}/{s}_chunk*.pt")):
            rows += torch.load(f, weights_only=False)
        per = {}
        tgt_count = {c: 0 for c in CLASSES}
        n_targets = 0
        for r in rows:
            if not r["agent_label"] or "agent_cz" not in r:
                continue
            b = r["agent_box"].double().numpy()
            cz = r["agent_cz"].double().numpy()
            h = r["agent_h"].double().numpy()
            zm = r["agent_zh_mask"].numpy().astype(bool)
            cl = r["agent_cls"].numpy().astype(int)
            bot = cz - h / 2
            vis = (b[:, 0] >= 0) & (b[:, 0] <= 60) & (np.abs(b[:, 1]) <= 16) & \
                (np.arctan2(np.abs(b[:, 1]), b[:, 0]) <= math.radians(60))
            for j in np.nonzero(vis)[0]:
                if 0 <= cl[j] < 10:
                    tgt_count[CLASSES[cl[j]]] += 1
                    n_targets += 1
            ref = zm & np.isin(cl, REF)
            for j in np.nonzero(zm)[0]:
                if not (0 <= cl[j] < 10):
                    continue
                d = np.hypot(b[:, 0] - b[j, 0], b[:, 1] - b[j, 1])
                nb = ref & (d <= a.radius_m)
                nb[j] = False
                if nb.sum() < 2:
                    continue
                rg = float(np.median(bot[nb]))
                rr = math.hypot(b[j, 0], b[j, 1])
                per.setdefault(CLASSES[cl[j]], []).append((bot[j] - rg, bot[j], cz[j] + h[j] / 2, h[j], rr,
                                                           bool(vis[j])))
        out = {"n_windows": sum(1 for r in rows if r["agent_label"]), "n_targets": n_targets,
               "target_class_counts": tgt_count, "by_class": {}}
        for c, v in sorted(per.items()):
            v = np.asarray(v, dtype=np.float64)
            d = v[:, 0]
            row = {"n": int(len(d)), "n_targets": int(v[:, 5].sum()), "median_d_m": float(np.median(d)),
                   "p10_d_m": float(np.percentile(d, 10)), "p90_d_m": float(np.percentile(d, 90)),
                   "median_bottom_rig_m": float(np.median(v[:, 1])), "median_top_rig_m": float(np.median(v[:, 2])),
                   "median_h_m": float(np.median(v[:, 3])),
                   "by_range": {}}
            for lo, hi in ((0, 20), (20, 40), (40, 200)):
                m = (v[:, 4] >= lo) & (v[:, 4] < hi)
                if m.any():
                    row["by_range"][f"{lo}-{hi}m"] = {"n": int(m.sum()), "median_d_m": float(np.median(d[m]))}
            out["by_class"][c] = row
        # the proposed re-seat: classes whose median |d| > 0.25 m on >= 30 rows
        fix = {c: v["median_d_m"] for c, v in out["by_class"].items() if v["n"] >= 30 and abs(v["median_d_m"]) > 0.25}
        n_fix_t = sum(out["by_class"][c]["n_targets"] for c in fix)
        out["reseat_proposal"] = {"classes": fix,
                                  "targets_moved": n_fix_t,
                                  "share_of_targets_with_reference": (n_fix_t / max(1, sum(v["n_targets"] for v in
                                                                                             out["by_class"].values())))}
        res["sets"][s] = out
        print(s, json.dumps({k: v for k, v in out.items() if k != "by_class"}), flush=True)
        for c, v in out["by_class"].items():
            print("   ", c, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "by_range"},
                  v["by_range"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
