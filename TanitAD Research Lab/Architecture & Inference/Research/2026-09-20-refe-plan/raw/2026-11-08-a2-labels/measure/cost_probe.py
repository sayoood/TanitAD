#!/usr/bin/env python3
"""Cost of the extension: per scene, warm, 3x OFF and 3x ON calls of score_plans; medians.  usage: cost_probe.py <scenes_dir> [n_scenes]"""
import importlib.util, json, os, statistics as st, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
NAVSIM = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
OV = "F:/a2_scratch/merged_refe"
sys.path[:0] = [NAVSIM, OV]
import numpy as np
import navsim_pdm_targets as PT, online_labeller as OL
PT._setup()
S = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
states = json.load(open(f"{S}/states.json")); rows = []
for k, s in enumerate(states):
    p = f"{S}/parts/{s['log']}/{s['token']}.pkl.xz"
    if not os.path.isfile(p): continue
    parts = OL.load_parts(p); pl = PT.executed_plans8(np.load(f"{S}/fan_{k:02d}.npy"), repair=True)
    PT.score_plans(parts, pl, behaviour=True)                         # warm
    off, on = [], []
    for _ in range(3):
        t = time.perf_counter(); PT.score_plans(parts, pl); off.append(time.perf_counter() - t)
        t = time.perf_counter(); PT.score_plans(parts, pl, behaviour=True); on.append(time.perf_counter() - t)
    rows.append((st.median(off), st.median(on)))
    if len(rows) >= N: break
o, n = [r[0] for r in rows], [r[1] for r in rows]
res = dict(n=len(rows), off_median_s=st.median(o), on_median_s=st.median(n), extra_median_s=st.median([b - a for a, b in rows]), extra_p90_s=sorted(b - a for a, b in rows)[int(0.9 * (len(rows) - 1))])
print(json.dumps(res)); json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_probe_result.json"), "w"), indent=1)
