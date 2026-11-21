"""Clock check: reproduce the cache's 10 Hz speed from the 100 Hz egomotion LOG at the trainer's clock
(grid_start_s + (row + 2) * dt_s), and at +-0.05 s / +-0.10 s shifts.  A correct clock has a sharp minimum at 0."""
import json, sys
import numpy as np, pandas as pd, d2_lib as L
man, out = sys.argv[1], sys.argv[2]
side = L.load_clock_sidecar("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
trs = L.tracks_from_manifest(man, side)
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
res = {s: [] for s in (-0.1, -0.05, 0.0, 0.05, 0.1)}
n = 0
src = {}
for tr in trs:
    try:
        df = pd.read_parquet(EGO.format(tr.clip_id), columns=["timestamp", "vx", "vy", "vz"])
    except Exception:
        continue
    ts = df.timestamp.to_numpy(dtype=np.float64) / 1e6
    ts = ts - ts[0]
    v = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64), axis=1)
    for s in res:
        res[s].append(float(np.abs(np.interp(tr.t + s, ts, v) - tr.v).mean()))
    src.setdefault(tr.clock_src, []).append(float(res[0.0][-1]))
    n += 1
R = {"n_clips": n, "mean_abs_speed_err_ms_by_shift": {str(s): {"median": round(float(np.median(v)), 5), "p95": round(float(np.percentile(v, 95)), 5), "max": round(float(np.max(v)), 5)} for s, v in res.items()},
     "shift0_by_clock_source": {k: {"n": len(v), "median": round(float(np.median(v)), 5), "max": round(float(np.max(v)), 5)} for k, v in src.items()}}
json.dump(R, open(out, "w"), indent=1)
print(json.dumps(R, indent=1))
