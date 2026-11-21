"""make_rehearsal_arrays: inputs for the SUBSET rehearsal of tswap_result.py. The pinned extractor needs all 141 episodes, so the subset rollouts (10 episodes) cannot be extracted; the rehearsal therefore uses the
first 170 windows of the BANKED RIC0_s0 arrays as 'S_OWN' and the same arrays with M1 shifted by -0.02 on the windows where it is finite as 'S_TSWAP' (a known answer: the M1 contrast reads -0.02 exactly). Not a result."""
import sys
import numpy as np
src, dst = "E:/Projects/TanitAD-artifacts/dagland/motion_c/ri_read/raw/arrays/RIC0_s0.npz", sys.argv[1]
z = np.load(src, allow_pickle=True)
n = 170
out = {k: (z[k][:n] if (z[k].ndim >= 1 and len(z[k]) == len(z["eid"])) else z[k]) for k in z.files}
import os
os.makedirs(dst, exist_ok=True)
np.savez(f"{dst}/S_OWN.npz", **out)
t = dict(out)
t["a_18"] = out["a_18"] - 0.02
np.savez(f"{dst}/S_TSWAP.npz", **t)
print("wrote", dst, "n", n, "episodes", len(np.unique(out["eid"])))
