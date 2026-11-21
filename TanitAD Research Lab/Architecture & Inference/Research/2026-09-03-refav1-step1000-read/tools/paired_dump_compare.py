"""Paired-checkpoint trajectory comparison (MEASURED 2026-09-03 05:45 Berlin, 0 GPU).
Loads the two T1 dump dirs written by `refav1_arm.py` for the two step-1,000 checkpoints (the retired fp32
incumbent and the clean epoch under EMA + bf16 + TF32) over the SAME 140 windows, asserts the window grid and the
ground truth match, and counts per arm how many windows are bit-identical between the two reads."""
import glob

import numpy as np

A = sorted(glob.glob("t1_dump/ep0*.npz"))
B = sorted(glob.glob("t1_dump_ep2/ep0*.npz"))
arms = ["cl", "ha", "ol", "cl_navshuf", "cl_oraclegoal"]
diff = {a: [0, 0, 0.0] for a in arms}
for fa, fb in zip(A, B):
    za, zb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    assert np.array_equal(za["g"], zb["g"]) and np.array_equal(za["ws"], zb["ws"]), (fa, "window mismatch")
    for a in arms:
        d = np.abs(za[a] - zb[a])
        for i in range(d.shape[0]):
            diff[a][0 if d[i].max() < 1e-9 else 1] += 1
            diff[a][2] = max(diff[a][2], float(d[i].max()))
for a in arms:
    i, n, m = diff[a]
    print(f"{a:14s} identical {i:3d}/{i + n}  differing {n:3d}  max|delta| {m:.6f} m")
