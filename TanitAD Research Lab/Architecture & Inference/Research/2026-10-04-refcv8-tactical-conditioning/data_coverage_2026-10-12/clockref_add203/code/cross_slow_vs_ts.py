"""Slow-regime q4c (ego-log position inversion) vs the camera-timestamp route, on the stationary-class clips
(33 ADD + 12 old).  The two are DIFFERENT measurements (ego-log kinematics vs camera timestamps), so this is the
independence check on exactly the population the standard q4c could not reach.  Dev box (camera timestamps live here)."""
import hashlib, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "C:/Users/Admin/tanitad-data/add203/repo/stack")
from tanitad.data import pose_sync as PS
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/"
ids = [l.strip() for l in open("raw/ADD_203_clip_ids.txt") if l.strip()] + [l.strip() for l in open("C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_train_clips.txt") if l.strip()]
byk = {sha12(c): c for c in ids}
d = json.load(open(sys.argv[1]))
rows = d["splits"]["train"]["per_clip"]
out, ndt = [], 0
for r in rows:
    c = byk[r["clip"]]
    ts = pd.read_parquet(CAM + c + ".timestamps.parquet")
    tf = ts[next(x for x in ts.columns if "time" in x.lower())].to_numpy(np.float64)
    te = np.sort(pd.read_parquet(EGO + c + ".parquet")["timestamp"].to_numpy(np.float64))
    tq, fi, unit = PS.resample_grid(tf); dd, dt, n = PS.grid_delta(tf)
    g0 = (tq[0] - te[0]) / unit
    # compare the implied time of the LAST window row too (the gate's t_true at row r, ns=3, r up to ~200): the dt lever arm
    j = np.array([2 + 0, 2 + 100, 2 + 199])
    t_ts = g0 + j * dt; t_q = r["grid_start_on_label_timeline_s"] + j * r["dt_s"]
    out.append({"clip": r["clip"], "grp": r["grp"], "n_moving_rows": r["n_moving_rows"], "resid_max_ms": r["fit_resid_max_ms"],
                "d_g0_s": r["grid_start_on_label_timeline_s"] - g0, "d_dt_s": r["dt_s"] - dt, "max_abs_d_t_over_window_s": float(np.max(np.abs(t_q - t_ts)))})
res = {"n": len(out), "max_abs_d_g0_s": max(abs(o["d_g0_s"]) for o in out) if out else None,
       "max_abs_d_dt_s": max(abs(o["d_dt_s"]) for o in out) if out else None,
       "max_abs_d_t_over_window_s": max(o["max_abs_d_t_over_window_s"] for o in out) if out else None, "per_clip": out}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_clip"})
for o in sorted(out, key=lambda o: -o["max_abs_d_t_over_window_s"])[:50]:
    print(o["grp"], o["clip"], "n %d resid %.4f ms  d_g0 %+.3e s  d_dt %+.3e  max|d_t| window %.3e s" % (o["n_moving_rows"], o["resid_max_ms"], o["d_g0_s"], o["d_dt_s"], o["max_abs_d_t_over_window_s"]))
