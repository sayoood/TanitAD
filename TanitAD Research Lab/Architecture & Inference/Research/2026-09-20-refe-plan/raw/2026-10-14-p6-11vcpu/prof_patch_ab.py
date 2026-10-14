#!/usr/bin/env python3
"""A/B of ONE exact-by-construction hot-path change: NAVSIM's `StateIndex.size()` rebuilds its answer from `dir(cls)` on EVERY call (2,665 calls per 64-proposal set).
Arm B replaces it with a constant computed ONCE from the original method. Interleaved ABAB, same sets, pinned to one logical CPU; outputs must be BIT-identical.
    python prof_patch_ab.py --n 40 --reps 3 --pin 4 --out prof_patch_ab.json
"""
import argparse, json, lzma, pickle, sys, os, time
from pathlib import Path
import numpy as np, psutil
HERE = Path(__file__).resolve().parent; PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "refe")); sys.path.insert(0, str(PKG / "code"))
MLAB = PKG / "raw" / "2026-10-11-m-lab"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--pin", type=int, default=4); ap.add_argument("--out", required=True)
a = ap.parse_args()
P = psutil.Process(); P.cpu_affinity([a.pin])
import navsim_pdm_targets as PT; PT._setup()
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
orig = StateIndex.__dict__["size"]
SIZE = StateIndex.size()
plan = json.load(open(MLAB / "plan" / "cached_r1.json")); npz = np.load(MLAB / "plan" / "cached_r1.npz")
toks = [str(t) for t in npz["token"]][3:3 + a.n]; row = {str(t): i for i, t in enumerate(npz["token"])}
def up(p8):
    t8 = 0.5 * np.arange(1, 9); t20 = 0.2 * np.arange(1, 21); o = np.zeros((p8.shape[0], 20, 3))
    for c in range(3):
        for m in range(p8.shape[0]): o[m, :, c] = np.interp(t20, t8, p8[m, :, c])
    return o
def load(t):
    mc = pickle.loads(lzma.decompress(open(plan["paths"][t], "rb").read()))
    return (mc.trajectory, mc.ego_state, mc.observation, mc.centerline, list(mc.route_lane_ids), mc.drivable_area_map)
parts = {t: load(t) for t in toks}; p20 = {t: up(npz["proposals"][row[t]]) for t in toks}
def arm(patched):
    StateIndex.size = classmethod(lambda cls: SIZE) if patched else orig
    outs = []
    c0 = P.cpu_times(); c0 = c0.user + c0.system; w0 = time.perf_counter()
    for t in toks:
        outs.append(PT.score_plans(parts[t], PT.executed_plans8(p20[t], repair=True)))
    c1 = P.cpu_times(); c1 = c1.user + c1.system
    return (c1 - c0) / len(toks), (time.perf_counter() - w0) / len(toks), outs
arm(False)  # warm
res = {"A_orig": [], "B_cached_size": [], "n_sets": len(toks), "state_index_size": SIZE}
ref = None; identical = True
for r in range(a.reps):
    for name, flag in (("A_orig", False), ("B_cached_size", True), ("A_orig", False), ("B_cached_size", True)):
        c, w, outs = arm(flag)
        res[name].append({"cpu_per_set": c, "wall_per_set": w})
        if ref is None: ref = outs
        else:
            for o1, o2 in zip(ref, outs):
                for k in o1:
                    if not np.array_equal(o1[k], o2[k]): identical = False
StateIndex.size = orig
mean = lambda L: float(np.mean([x["cpu_per_set"] for x in L]))
res["A_mean_cpu_per_set"] = mean(res["A_orig"]); res["B_mean_cpu_per_set"] = mean(res["B_cached_size"])
res["B_over_A"] = res["B_mean_cpu_per_set"] / res["A_mean_cpu_per_set"]
res["bit_identical_all_arrays"] = identical
res["env"] = {k: os.environ.get(k) for k in ("REFE_SIM_HZ", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("A_mean_cpu_per_set", "B_mean_cpu_per_set", "B_over_A", "bit_identical_all_arrays", "n_sets")}))
