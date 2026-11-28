import sys, os, json
import numpy as np
PK = "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, PK + "/refe"); sys.path.insert(0, PK + "/eval/instruments")
import transp_score as TSC
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
d = np.load("E:/Projects/TanitAD-artifacts/trkscore/work/smoke/trk_smoke_c1.npz", allow_pickle=False)
tlog = TSC.load_token_log("F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
names = ["x", "y", "heading", "vx", "vy", "ax", "ay", "steer", "steer_rate", "ang_vel", "ang_acc"]
mx = np.zeros(11); rel = np.zeros(11)
for i, t in enumerate(d["token"]):
    ego = TSC.load_cache_parts("F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest", tlog[str(t)], str(t))[1]
    s = ego_state_to_state_array(ego)
    dd = np.abs(s - d["s0"][i]); mx = np.maximum(mx, dd)
    rel = np.maximum(rel, dd / np.maximum(np.abs(s), 1e-12))
    if i == 0:
        print("tp cache", ego.time_point.time_us, "x", repr(s[0]), repr(d["s0"][i][0]), "vx", repr(s[3]), repr(d["s0"][i][3]))
print(json.dumps({n: [float(a), float(r)] for n, a, r in zip(names, mx, rel)}))
