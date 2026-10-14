import cProfile, io, json, lzma, pickle, pstats, sys, os
from pathlib import Path
import numpy as np, psutil
HERE = Path(__file__).resolve().parent; PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "refe")); sys.path.insert(0, str(PKG / "code"))
MLAB = PKG / "raw" / "2026-10-11-m-lab"
psutil.Process().cpu_affinity([4])
import navsim_pdm_targets as PT; PT._setup()
plan = json.load(open(MLAB / "plan" / "cached_r1.json")); npz = np.load(MLAB / "plan" / "cached_r1.npz")
toks = [str(t) for t in npz["token"]][3:23]; row = {str(t): i for i, t in enumerate(npz["token"])}
def up(p8):
    t8 = 0.5 * np.arange(1, 9); t20 = 0.2 * np.arange(1, 21); o = np.zeros((p8.shape[0], 20, 3))
    for c in range(3):
        for m in range(p8.shape[0]): o[m, :, c] = np.interp(t20, t8, p8[m, :, c])
    return o
def load(t):
    mc = pickle.loads(lzma.decompress(open(plan["paths"][t], "rb").read()))
    return (mc.trajectory, mc.ego_state, mc.observation, mc.centerline, list(mc.route_lane_ids), mc.drivable_area_map)
parts = {t: load(t) for t in toks}; p20 = {t: up(npz["proposals"][row[t]]) for t in toks}
PT.score_plans(parts[toks[0]], PT.executed_plans8(p20[toks[0]]))
pr = cProfile.Profile(); pr.enable()
for t in toks: PT.score_plans(parts[t], PT.executed_plans8(p20[t]))
pr.disable()
st = pstats.Stats(pr).strip_dirs()
for pat in ("svd", "contains_xy", "ego_state_to_state_array", "c_einsum"):
    print("=== callers of", pat)
    st.print_callers(pat)
