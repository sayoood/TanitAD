"""Extent statistics of the banked REFe rank-0 targets (ego frame, 20 poses @5 Hz = 4 s) to choose the label raster extent.
Reads a rank-0 jsonl; prints x/y percentiles of the 4-s endpoint and of ALL points. Evidence class: MEASURED."""
import json, sys, numpy as np
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
xs, ys, ex, ey, v0, logs = [], [], [], [], [], set()
with open(path, encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= n: break
        r = json.loads(l)
        t = np.asarray(r["traj"], float)
        xs.append(t[:, 0]); ys.append(t[:, 1]); ex.append(t[-1, 0]); ey.append(t[-1, 1]); v0.append(r["ego"][-1]); logs.add(r["log_name"])
xs, ys = np.concatenate(xs), np.concatenate(ys); ex, ey = np.array(ex), np.array(ey)
q = [0, 1, 50, 90, 99, 99.9, 100]
print(f"rows={len(ex)} logs={len(logs)} file={path}")
print("endpoint x pct", dict(zip(q, np.round(np.percentile(ex, q), 2))))
print("endpoint y pct", dict(zip(q, np.round(np.percentile(ey, q), 2))))
print("all-pts  x pct", dict(zip(q, np.round(np.percentile(xs, q), 2))))
print("all-pts |y| pct", dict(zip(q, np.round(np.percentile(np.abs(ys), q), 2))))
print("v0 pct", dict(zip(q, np.round(np.percentile(v0, q), 2))))
