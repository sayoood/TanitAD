"""Does ONE baked camera calibration hold across navtrain's vehicles?

REFe bakes K / distortion / extrinsics as constants in REFeConfig (provenance: 64 driverl_val14
logs). navtrain spans four cities and many vehicles. For every local DB that navtrain lists, read
the `camera` table for the four REFe channels and measure the deviation from the baked values:
translation (m), rotation (deg, geodesic between quaternions), focal/principal point (px),
distortion coefficients. Deviation matters because the frustum lifts image features into the ego
frame with these numbers.
"""
import glob
import json
import math
import os
import pickle
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
                   "2026-09-20-refe-plan/refe")
CAL = json.load(open("D:/Projects/TanitAD/data/nuplan_cam_calib.json", encoding="utf-8"))
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
import navtrain_scenarios as NS  # noqa: E402

per_log, rep = NS.resolve()
dbs = NS.index_dbs()
logs = sorted(l for l in per_log if l in dbs)
print(f"navtrain logs held locally: {len(logs)}")


def unp(b):
    if b is None:
        return None
    try:
        return np.asarray(pickle.loads(b), dtype=float)
    except Exception:
        return None


def qang(q1, q2):
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    q2 = np.asarray(q2, float) / np.linalg.norm(q2)
    d = min(1.0, abs(float(np.dot(q1, q2))))
    return math.degrees(2 * math.acos(d))


dev = defaultdict(list)
veh_of = {}
distinct = defaultdict(set)
for lg in logs:
    c = sqlite3.connect(dbs[lg])
    cols = [r[1] for r in c.execute("PRAGMA table_info(camera)").fetchall()]
    for ch in CAMS:
        row = c.execute("SELECT * FROM camera WHERE channel=?", (ch,)).fetchone()
        if row is None:
            continue
        r = dict(zip(cols, row))
        t = unp(r.get("translation"))
        q = unp(r.get("rotation"))
        K = unp(r.get("intrinsic"))
        dist = unp(r.get("distortion"))
        b = CAL[ch]
        if t is not None:
            dev[(ch, "t_m")].append(float(np.linalg.norm(t[:3] - np.asarray(b["t"][:3]))))
        if q is not None:
            dev[(ch, "rot_deg")].append(qang(q, b["q"]))
        if K is not None:
            Kb = np.asarray(b["K"], float)
            dev[(ch, "K_px")].append(float(np.abs(K.reshape(3, 3) - Kb).max()))
        if dist is not None:
            dev[(ch, "dist")].append(float(np.abs(dist.ravel()[:5] - np.asarray(b["distortion"])).max()))
        distinct[ch].add((tuple(np.round(t, 3)) if t is not None else None,
                          tuple(np.round(q, 4)) if q is not None else None))
    c.close()
    veh_of[lg] = lg.split("_")[1] if "_" in lg else "?"

vehs = sorted(set(veh_of.values()))
print(f"vehicles: {len(vehs)} ({', '.join(vehs[:12])}{' ...' if len(vehs) > 12 else ''})")
print(f"{'camera':7s} {'quantity':8s} {'n':>4s} {'median':>9s} {'p95':>9s} {'max':>9s}")
worst = {}
for (ch, k), v in sorted(dev.items()):
    a = np.asarray(v)
    print(f"{ch:7s} {k:8s} {len(a):4d} {np.median(a):9.4f} {np.percentile(a, 95):9.4f} {a.max():9.4f}")
    worst[k] = max(worst.get(k, 0.0), float(a.max()))
for ch in CAMS:
    print(f"  {ch}: {len(distinct[ch])} distinct (translation, rotation) calibrations over {len(logs)} logs")
print("ZZCALIB_" + "_".join(f"{k}{worst[k]:.4f}" for k in sorted(worst)) + "ZZ")
