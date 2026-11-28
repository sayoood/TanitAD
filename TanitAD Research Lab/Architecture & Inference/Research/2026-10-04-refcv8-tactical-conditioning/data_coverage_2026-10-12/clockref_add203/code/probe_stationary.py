"""Probe: how much kinematic content do the 33 stationary ADD clips (and the 12 old non-fit) carry?
Per clip: max cache speed, rows above 0.1/0.5/2 m/s, cache-path length, whole-log max speed / path length."""
import hashlib, json, sys
import numpy as np, pandas as pd, torch
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
m = torch.load("/home/nvidia/data/refcv8-b1-416x1024-train-4572/_v2manifest.pt", map_location="cpu", weights_only=False)
ADD = [l.strip() for l in open("/home/nvidia/clockref203/code/ADD_203_clip_ids.txt") if l.strip()]
A = json.load(open("/home/nvidia/clockref203/work/q4c_grid_vs_egolog_ALLTRAIN.json"))["splits"]["train"]["per_clip"]
ref = {r["clip"]: r for r in A}
new = {r["clip"]: r for r in json.load(open("/home/nvidia/clockref203/work/q4c_grid_vs_egolog_ADD203.json"))["splits"]["train"]["per_clip"]}
EGO = "/home/nvidia/data/obstacle/egomotion_alpamayo/"
rows = []
for i, c in enumerate(m["clip_id"]):
    k = sha12(c)
    grp = "add" if c in set(ADD) else "old"
    fit = (k in new) if grp == "add" else (k in ref)
    hi = (ref[k]["fit_resid_max_ms"] >= 5.0) if (grp == "old" and k in ref) else False
    if fit and not hi:
        continue
    p = m["poses"][i].double().numpy()
    eg = pd.read_parquet(EGO + c + ".parquet")
    t = eg["timestamp"].to_numpy(np.float64); o = np.argsort(t); t = t[o]
    x = eg["x"].to_numpy(np.float64)[o]; y = eg["y"].to_numpy(np.float64)[o]
    v = np.hypot(eg["vx"].to_numpy(np.float64)[o], eg["vy"].to_numpy(np.float64)[o])
    path_log = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    cache_path = float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))
    sp = p[:, 3]
    rows.append({"clip": k, "grp": grp, "why_unfit": ("resid>=5ms" if hi else "stationary(<20 rows >2m/s)"),
                 "n_rows": len(p), "cache_v_max": float(sp.max()), "n_gt_0p1": int((sp > .1).sum()), "n_gt_0p5": int((sp > .5).sum()),
                 "n_gt_2": int((sp > 2).sum()), "cache_path_m": cache_path, "log_v_max": float(v.max()), "log_path_m": path_log,
                 "log_span_s": float((t[-1] - t[0]) / (1e6 if (t[-1]-t[0])/1e6 > 1 else 1)), "fit_resid_old_ms": ref.get(k, {}).get("fit_resid_max_ms")})
json.dump(rows, open("/home/nvidia/clockref203/raw/probe_stationary.json", "w"), indent=1)
import collections
print("n unfit/high-resid:", len(rows), collections.Counter((r["grp"], r["why_unfit"]) for r in rows))
for r in sorted(rows, key=lambda r: (r["grp"], r["cache_path_m"])):
    print(r["grp"], r["clip"], r["why_unfit"][:4], "vmax %.2f n>0.1 %d n>0.5 %d n>2 %d cachepath %.2f m logvmax %.2f logpath %.1f" % (r["cache_v_max"], r["n_gt_0p1"], r["n_gt_0p5"], r["n_gt_2"], r["cache_path_m"], r["log_v_max"], r["log_path_m"]))
