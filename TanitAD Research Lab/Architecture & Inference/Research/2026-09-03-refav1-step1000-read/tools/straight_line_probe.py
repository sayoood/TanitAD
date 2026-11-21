"""Straight-line probe over the banked T1 dumps (MEASURED 2026-09-03 03:55 Berlin, 0 GPU).
For every arm: how many of the 140 windows are an exactly straight (y == 0) constant-speed line; the crude lateral
error |y_arm - y_gt| in the t0 frame, split by whether the human's own path is straight (max |y_gt| < 0.3 m)."""
import glob
import numpy as np

files = sorted(glob.glob("raw_dump_or_slice/t1_dump/ep*.npz"))   # run from the package dir with the dump linked in
arms = ["cl", "ha", "ol", "cl_navshuf", "cl_oraclegoal"]
for f in files:
    z = np.load(f, allow_pickle=True)
    g = z["g"]
    for a in arms:
        t = z[a]
        for i in range(g.shape[0]):
            straight = np.max(np.abs(t[i, :, 1])) < 1e-6
            steps = np.linalg.norm(np.diff(np.vstack([[0, 0], t[i]]), axis=0), axis=1)
            const_speed = np.ptp(steps) < 1e-4
            print(f, a, i, straight, const_speed, float(np.mean(np.abs(t[i, :, 1] - g[i, :, 1]))))
