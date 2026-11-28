#!/usr/bin/env python3
"""Hot-path profile of the P6 online labeller body: executed_plans8 + score_plans on M-LAB's cached navtest parts (the cached design).

Run in the NAVSIM venv (C:/Users/Admin/navsim-crun/venv), pinned to ONE logical CPU, env as M-LAB (REFE_SIM_HZ=10, 1 BLAS thread).
    python prof_score.py --n 60 --pin 4 --out prof_score.json [--threads N]
cProfile inflates python-level call overhead, so the numbers are RANKINGS (share of cumulative time), not absolute costs; the un-profiled
CPU per set is reported next to them (same sets, same process).
"""
import argparse, cProfile, io, json, lzma, os, pickle, pstats, sys, time
from pathlib import Path
import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "refe")); sys.path.insert(0, str(PKG / "code"))
MLAB = PKG / "raw" / "2026-10-11-m-lab"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--pin", type=int, default=-1)
ap.add_argument("--plan", default="cached_r1")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.pin >= 0:
    psutil.Process().cpu_affinity([a.pin])
import navsim_pdm_targets as PT
PT._setup()
plan = json.load(open(MLAB / "plan" / f"{a.plan}.json", encoding="utf-8"))
npz = np.load(MLAB / "plan" / f"{a.plan}.npz")
toks = [str(t) for t in npz["token"]][3:3 + a.n]
row = {str(t): i for i, t in enumerate(npz["token"])}
props = npz["proposals"]


def upsample20(p8):
    t8 = 0.5 * np.arange(1, 9); t20 = 0.2 * np.arange(1, 21)
    out = np.zeros((p8.shape[0], 20, 3))
    for c in range(3):
        for m in range(p8.shape[0]):
            out[m, :, c] = np.interp(t20, t8, p8[m, :, c])
    return out


def load(t):
    mc = pickle.loads(lzma.decompress(open(plan["paths"][t], "rb").read()))
    return (mc.trajectory, mc.ego_state, mc.observation, mc.centerline, list(mc.route_lane_ids), mc.drivable_area_map)


parts = {t: load(t) for t in toks}
p20 = {t: upsample20(props[row[t]]) for t in toks}
for t in toks[:3]:                                       # warm-up (untimed)
    PT.score_plans(parts[t], PT.executed_plans8(p20[t], repair=True))
P = psutil.Process()
c0 = P.cpu_times(); w0 = time.perf_counter()
for t in toks:
    PT.score_plans(parts[t], PT.executed_plans8(p20[t], repair=True))
c1 = P.cpu_times(); w1 = time.perf_counter()
plain = ((c1.user + c1.system) - (c0.user + c0.system)) / len(toks)
pr = cProfile.Profile()
pr.enable()
for t in toks:
    PT.score_plans(parts[t], PT.executed_plans8(p20[t], repair=True))
pr.disable()
s = io.StringIO()
st = pstats.Stats(pr, stream=s).strip_dirs()
total = st.total_tt
rows = []
for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
    rows.append({"func": f"{fn}:{ln}({name})", "ncalls": nc, "tottime_s": tt, "cumtime_s": ct})
by_tot = sorted(rows, key=lambda r: -r["tottime_s"])[:25]
by_cum = sorted(rows, key=lambda r: -r["cumtime_s"])[:30]
res = {"n_sets": len(toks), "plain_cpu_per_set_s": plain, "plain_wall_per_set_s": (w1 - w0) / len(toks), "profiled_total_s": total,
       "profiled_per_set_s": total / len(toks), "top_by_tottime": by_tot, "top_by_cumtime": by_cum,
       "env": {k: os.environ.get(k) for k in ("REFE_SIM_HZ", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}, "pin": a.pin,
       "numpy": np.__version__, "affinity": P.cpu_affinity()}
json.dump(res, open(a.out, "w"), indent=1)
print(f"plain cpu/set {plain:.4f}  profiled/set {total / len(toks):.4f}")
for r in by_tot[:18]:
    print(f"{r['tottime_s'] / total * 100:5.1f}% tot  {r['cumtime_s'] / total * 100:5.1f}% cum  {r['ncalls']:>7}  {r['func']}")
