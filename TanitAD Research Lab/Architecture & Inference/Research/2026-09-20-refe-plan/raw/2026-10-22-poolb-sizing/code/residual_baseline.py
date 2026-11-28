#!/usr/bin/env python3
"""Baseline for the mirror-frame pairing gate: per-camera |image timestamp - token lidar timestamp| of the frame the DB rule picks, on the 7 DB-local pool-B logs
(8,192 frame x camera pairs, MEASURED from the nuPlan DB `image` / `lidar_pc` tables). A mirror whose nearest-row residual distribution differs from this is not the same frame set."""
import os, sys, json, gzip, sqlite3
import numpy as np
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
P = json.load(gzip.open(os.path.join(RAW, "poolB_frames.json.gz"), "rt"))
ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval/"
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
res = {c: [] for c in CAMS}; img_dt = []
for log in P:
    p = ROOT + log + ".db"
    if not os.path.exists(p): continue
    c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    lp = {r[0]: r[1] for r in c.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc")}
    per = {}
    for ch in CAMS:
        tok = c.execute("SELECT token FROM camera WHERE channel=?", (ch,)).fetchone()[0]
        ts = np.array([r[0] for r in c.execute("SELECT timestamp FROM image WHERE camera_token=? ORDER BY timestamp", (tok,))], dtype=np.int64)
        per[ch] = ts; img_dt.extend(np.diff(ts) / 1000.0)
    for f in P[log]["frames"]:
        t = lp[f["token"]]
        for ch in CAMS:
            res[ch].append(float(np.min(np.abs(per[ch] - t))) / 1000.0)
    c.close()
def q(a):
    a = np.asarray(a); return {"n": int(a.size), "mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "p99_ms": round(float(np.percentile(a, 99)), 2), "max_ms": round(float(a.max()), 2)}
out = {"per_camera_residual": {c: q(v) for c, v in res.items()}, "all4_residual": q(sum(res.values(), [])), "image_row_spacing_ms": q(img_dt)}
json.dump(out, open(os.path.join(RAW, "residual_baseline.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
