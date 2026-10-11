"""Step 3: q4c (position inversion, ADD203 file) vs the ts33 clip-clock sidecar vs the camera-timestamp route,
on the 203 ADD clips. Dev box (the camera timestamp parquets live here). Output is sha12-keyed (no raw UUIDs)."""
import hashlib, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "C:/Users/Admin/tanitad-data/add203/repo/stack")
from tanitad.data.v2_dataset import stable_episode_id
from tanitad.data import pose_sync as PS
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/"
SC = "C:/Users/Admin/tanitad-data/add203/out/refcv8_clip_clock_sidecar_ts33.jsonl"
ids = [l.strip() for l in open("raw/ADD_203_clip_ids.txt") if l.strip()]
assert len(ids) == 203
side = {}
for l in open(SC, encoding="utf-8"):
    if l.strip():
        r = json.loads(l); side[r["sid"]] = r
q4 = {r["clip"]: r for r in json.load(open("raw/q4c_grid_vs_egolog_ADD203.json"))["splits"]["train"]["per_clip"]}
def ts_route(cid):
    ts = pd.read_parquet(CAM + cid + ".timestamps.parquet")
    tf = ts[next(c for c in ts.columns if "time" in c.lower())].to_numpy(np.float64)
    te = np.sort(pd.read_parquet(EGO + cid + ".parquet")["timestamp"].to_numpy(np.float64))
    t_query, fi, unit = PS.resample_grid(tf)
    d, dt, n = PS.grid_delta(tf)
    return (t_query[0] - te[0]) / unit, dt
rows = []
for cid in ids:
    s = stable_episode_id(cid); k = sha12(cid)
    g0, dt = ts_route(cid)
    sr = side.get(s)
    rows.append({"clip": k, "in_q4c": k in q4, "sidecar_src": (sr or {}).get("clock_src", "position-inversion" if sr else None),
                 "ts_g0": g0, "ts_dt": dt,
                 "sc_g0": (sr or {}).get("grid_start_s"), "sc_dt": (sr or {}).get("dt_s"),
                 "q_g0": q4[k]["grid_start_on_label_timeline_s"] if k in q4 else None,
                 "q_dt": q4[k]["dt_s"] if k in q4 else None})
def mx(a, b, sel):
    v = [abs(r[a] - r[b]) for r in rows if sel(r) and r[a] is not None and r[b] is not None]
    return (max(v), len(v)) if v else (None, 0)
mov = lambda r: r["in_q4c"]; stat = lambda r: not r["in_q4c"]
out = {"n_add": len(rows), "n_in_q4c": sum(r["in_q4c"] for r in rows), "n_stationary_not_in_q4c": sum(not r["in_q4c"] for r in rows),
       "sidecar_src_counts": {str(k): sum(1 for r in rows if r["sidecar_src"] == k) for k in {r["sidecar_src"] for r in rows}},
       "moving170_q4c_vs_sidecar_pos_inversion": {"grid_start_s": mx("q_g0", "sc_g0", mov), "dt_s": mx("q_dt", "sc_dt", mov),
           "note": "SAME algorithm (build_clip_clock_sidecar = position inversion): agreement is determinism, NOT independence"},
       "moving170_q4c_vs_camera_timestamp_route": {"grid_start_s": mx("q_g0", "ts_g0", mov), "dt_s": mx("q_dt", "ts_dt", mov),
           "note": "two DIFFERENT measurements (ego-log position inversion vs camera timestamps via pose_sync)"},
       "moving170_sidecar_vs_camera_timestamp_route": {"grid_start_s": mx("sc_g0", "ts_g0", mov), "dt_s": mx("sc_dt", "ts_dt", mov)},
       "stationary33_sidecar_vs_camera_timestamp_route": {"grid_start_s": mx("sc_g0", "ts_g0", stat), "dt_s": mx("sc_dt", "ts_dt", stat),
           "note": "the sidecar rows ARE the camera-timestamp route (same code): determinism only"},
       "stationary33_ts_g0_range": [min(r["ts_g0"] for r in rows if stat(r)), max(r["ts_g0"] for r in rows if stat(r))],
       "per_clip": rows}
json.dump(out, open("raw/xcheck_q4c_vs_ts33.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "per_clip"})
