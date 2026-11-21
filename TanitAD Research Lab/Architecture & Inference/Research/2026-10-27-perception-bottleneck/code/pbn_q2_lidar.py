"""pbn_q2_lidar.py -- PERC-BOTTLENECK Q2 (PREREG_PERC_BOTTLENECK.md): are refcv8's own-sidewalk flags on GT-drivable cells
GT errors? CPU only. Population: eval capture cells (cap_eval, 0.25 m centres, |y| < 3 m) at x 0-20 m with the head's
DECLARED code == sidewalk (7) and SAM3 GT in the drivable set {1,2,3,4,6}.

Instrument: `bev-lidar-gt-b1eval-20260913` (`tanitad.lidar_bev_gt/2`): 0.5 m cartesian grid, x 0-60 m rows, col0 = y -16 m
(RIGHT), +y LEFT; axis 0 = RAW v2ep frame index; a window's current frame = stacked row (t + W - 1) + 2.
⛔ The artifact has NO ground-height channel (occupancy is z 0.30-3.0 m): a curb step is NOT readable here (blocker, stated).
What is read: OCCUPIED (`cart_occ`, the lidar_bev default rule) and observed, for
  FP  = own-sidewalk on GT-drivable,
  AGR = own-drivable on GT-drivable (agreement control, same windows/band),
  MIS = the FP cells read 20 raw frames away (mispaired-time control; +20, else -20).
Committed reading: FP occupancy >= 2x AGR AND > MIS -> an obstacle/GT-error share of the "false positives" exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DRV = (1, 2, 3, 4, 6)
EV_R = np.arange(0, 160, 2)
EV_C_ABS = np.arange(108, 132, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-eval", required=True)
    ap.add_argument("--bev", default="E:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stack", type=int, default=3)
    ap.add_argument("--shift", type=int, default=20)
    a = ap.parse_args()
    cap = Path(a.cap_eval)
    meta = json.loads((cap / "capture.json").read_text(encoding="utf-8"))
    x = (EV_R + 0.5) * 0.25
    y = 30.0 - (EV_C_ABS + 0.5) * 0.25
    br = np.floor(x / 0.5).astype(int)                         # bevgt row
    bc = np.floor((y + 16.0) / 0.5).astype(int)                # bevgt col (col0 = RIGHT)
    RR, CC = np.meshgrid(br, bc, indexing="ij")
    XX = np.broadcast_to(x[:, None], RR.shape)
    acc = {k: {"n": 0, "occ": 0, "obs": 0} for k in ("FP", "AGR", "MIS")}
    per_dist = {}
    n_win = n_nobev = n_nolabel = 0
    cache = {}
    for w in meta["windows"]:
        f = next(cap.glob(f"w_{w['n']:05d}_*.npz"))
        bp = Path(a.bev) / f"{w['sha12']}.bevgt.npz"
        if not bp.exists():
            n_nobev += 1
            continue
        if w["sha12"] not in cache:
            cache.clear()
            z = np.load(bp)
            cache[w["sha12"]] = {k: z[k] for k in ("raw_frame", "label_valid", "cart_occ", "cart_observed")}
        B = cache[w["sha12"]]
        raw = int(w["t"]) + int(w["W"]) - 1 + (a.n_stack - 1)
        pos = np.nonzero(B["raw_frame"] == raw)[0]
        if len(pos) != 1 or not bool(B["label_valid"][pos[0]]):
            n_nolabel += 1
            continue
        i = int(pos[0])
        z = np.load(f)
        dec = z["dec"][EV_R][:, EV_C_ABS - 104]
        gt = z["gt"][EV_R][:, EV_C_ABS - 104]
        band = (XX < 20) & (gt != 255)
        fp = band & (dec == 7) & np.isin(gt, DRV)
        agr = band & np.isin(dec, DRV) & np.isin(gt, DRV)
        occ, obs = B["cart_occ"][i][RR, CC], B["cart_observed"][i][RR, CC]
        j = i + a.shift if i + a.shift < len(B["raw_frame"]) else i - a.shift
        occ_m = B["cart_occ"][j][RR, CC] if (0 <= j < len(B["raw_frame"]) and bool(B["label_valid"][j])) else None
        for k, m, o in (("FP", fp, occ), ("AGR", agr, occ)):
            acc[k]["n"] += int(m.sum()); acc[k]["occ"] += int((o & m).sum()); acc[k]["obs"] += int((obs & m).sum())
        if occ_m is not None:
            acc["MIS"]["n"] += int(fp.sum()); acc["MIS"]["occ"] += int((occ_m & fp).sum())
        for lo, hi in ((0, 5), (5, 10), (10, 20)):
            mm = fp & (XX >= lo) & (XX < hi)
            d = per_dist.setdefault(f"{lo}_{hi}", {"n": 0, "occ": 0})
            d["n"] += int(mm.sum()); d["occ"] += int((occ & mm).sum())
        n_win += 1
    rate = {k: (v["occ"] / v["n"] if v["n"] else None) for k, v in acc.items()}
    obs_rate = {k: (v["obs"] / v["n"] if v["n"] else None) for k, v in acc.items() if k != "MIS"}
    reading = bool(rate["FP"] is not None and rate["AGR"] and rate["MIS"] is not None
                   and rate["FP"] >= 2 * rate["AGR"] and rate["FP"] > rate["MIS"])
    out = {"tool": "pbn_q2_lidar.py", "n_windows_read": n_win, "n_windows_no_bevgt": n_nobev, "n_windows_no_label": n_nolabel,
           "counts": acc, "occupied_rate": rate, "observed_rate": obs_rate,
           "fp_occupied_by_distance": {k: {**v, "rate": (v["occ"] / v["n"] if v["n"] else None)} for k, v in per_dist.items()},
           "OBSTACLE_SHARE_EXISTS": reading,
           "fp_share_explained_upper_bound": (rate["FP"] - rate["AGR"]) if (rate["FP"] is not None and rate["AGR"] is not None) else None,
           "curb_height_test": "BLOCKED: the artifact has no ground-height channel (occupancy band starts at 0.30 m); raw sweeps on the "
                               "dev box cover 2 clips, neither in eval139 (checked by uuid and sha12 prefix)"}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
