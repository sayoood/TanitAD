"""One-off: freeze the refav1 trunk-probe eval grid (141 episodes, 2,399 windows) and the 141 episodes' kinematics.

Reads the LOCAL v2ep episodes (poses [201,4] = x,y,yaw,v at 10 Hz; actions [201,2], channel 0 = road-wheel STEER angle)
and the banked probe features' window starts (``feat_s4/ep*.npz::ws``), and VERIFIES, for every one of the 2,399 windows,
that (v0, a0, steer0) recomputed from poses/actions equals the banked ``kin`` column to 0.0 -- which also proves that
episode ``i`` of the probe is the i-th v2ep file in sorted order.

Writes (next to this package, raw/):  eval141_kin.npz  (clip order, per-clip frame counts n, concatenated poses/actions)   and   eval_grid_windows.json.
"""
import glob, json, os, sys
import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
EPS = "C:/Users/Admin/tanitad-data/refav1-eval141/eps"
FEAT = "C:/Users/Admin/refav1_probe/feat_s4"

fs = sorted(glob.glob(f"{EPS}/*.v2ep.pt"))
fe = sorted(glob.glob(f"{FEAT}/ep*.npz"))
assert len(fs) == len(fe) == 141
order, P, A, windows, worst = [], [], [], {}, 0.0
for f, g in zip(fs, fe):
    o = torch.load(f, weights_only=False)
    cid = o["clip_id"]
    pose, act = o["poses"].numpy().astype(np.float32), o["actions"].numpy().astype(np.float32)
    z = np.load(g)
    ws = z["ws"].astype(int)
    k = 2 * ws
    kin = np.stack([pose[k, 3], (pose[k, 3] - pose[k - 2, 3]) / 0.2, act[k, 0]], 1)
    worst = max(worst, float(np.abs(kin - z["kin"]).max()))
    order.append(cid); P.append(pose); A.append(act); windows[cid] = [int(t) for t in ws]
assert worst == 0.0, f"kin mismatch {worst}"
n = np.array([len(x) for x in P])
np.savez_compressed(f"{OUT}/eval141_kin.npz", clip=np.array(order), n=n, poses=np.concatenate(P), actions=np.concatenate(A))
json.dump({"order": order, "windows": windows, "n_windows": sum(len(v) for v in windows.values()),
           "source": "refav1 trunk-probe feat_s4 ws; stride 4, t = 3 mod 4; episode i = i-th sorted v2ep",
           "kin_max_abs_diff_vs_banked": worst}, open(f"{OUT}/eval_grid_windows.json", "w"))
print("OK", len(order), sum(len(v) for v in windows.values()), "kin max abs diff", worst)
