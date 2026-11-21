#!/usr/bin/env python
"""r9clk residual: the SIGNED time error of the obstacle sample the join actually labels with.
world_agents_at (build_obstacle_join.py) takes, per track, the sample nearest t_s within tol_s=0.06 s. Mirror that rule on the raw
obstacle parquets and report (t_sample - t_s): a bias would be a real label lag; a symmetric spread is sampling jitter."""
import json, sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
D = "/home/nvidia/data"
src = json.load(open(sys.argv[1]))
EGO, OBS = f"{D}/obstacle/egomotion_alpamayo", f"{D}/obstacle/obstacle_offline_b1train"
import lzma, hashlib
# clips: first 60 clips of the join in file order, t_s from the banked join itself
want, recs = [], {}
with lzma.open(f"{D}/joins/b1_train_plus_eval_agents.jsonl.xz", "rt") as f:
    for line in f:
        r = json.loads(line)
        c = r["clip_id"]
        if c not in recs:
            if len(recs) >= int(sys.argv[2]):
                break
            recs[c] = []
        recs[c].append((r["frame"], r["t_s"], {a["track_id"] for a in r["agents"]}))
off, per, n_tr = [], [], 0
for c, rs in recs.items():
    if not os.path.isfile(f"{OBS}/{c}.parquet"):
        continue
    o = pd.read_parquet(f"{OBS}/{c}.parquet")
    t = o["timestamp_us"].to_numpy(np.float64) / 1e6
    tid = o["track_id"].astype(str).to_numpy()
    tracks = {k: np.sort(t[tid == k]) for k in np.unique(tid)}
    for k, tt in tracks.items():
        if len(tt) > 3:
            per.append(float(np.median(np.diff(tt))))
    for fr, ts, ids in rs:
        for k in ids:
            tt = tracks.get(k)
            if tt is None:
                continue
            j = int(np.argmin(np.abs(tt - ts)))
            off.append(float(tt[j] - ts)); n_tr += 1
off, per = np.array(off), np.array(per)
q = lambda v: {k: round(float(x), 5) for k, x in zip(("min", "p05", "p25", "med", "p75", "p95", "max"), np.quantile(v, [0, .05, .25, .5, .75, .95, 1]))}
out = {"n_clips": len(recs), "n_agent_labels": int(len(off)), "t_sample_minus_t_s_s": q(off), "mean_s": float(off.mean()),
       "frac_abs_gt_0.05": float((np.abs(off) > 0.05).mean()), "track_sample_period_s": q(per), "n_tracks": int(len(per))}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
