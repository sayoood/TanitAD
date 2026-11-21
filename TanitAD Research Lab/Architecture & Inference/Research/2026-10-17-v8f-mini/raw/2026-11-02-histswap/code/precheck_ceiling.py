import sys, os, glob, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagreread/code")
import numpy as np
import rr_lib as R
VMIN = 5.2
out = {}
for tag in ("DAGOACC_s0", "DAGOACC_s1", "SOUP_s0", "SOUP_s1"):
    d = f"F:/TanitAD-scratch/kappa_t1/out/SREP_{tag}/dump_SREP_{tag}"
    z = [np.load(f) for f in sorted(glob.glob(d + "/ep*.npz"))]
    cl = np.concatenate([x["cl"] for x in z]); fl = np.concatenate([x["fl"] for x in z]); g = np.concatenate([x["g"] for x in z])
    rec = {}
    for nm, X in (("Y", cl), ("P10", fl)):
        pa = R.pair_err(X, g)
        slow = pa["fast"] & (pa["vX"] < VMIN)
        rec[nm] = {"fast": int(pa["fast"].sum()), "slow": int(slow.sum()), "slow_by_block": [int(slow[:, a:b].sum()) for a, b in ((0, 10), (10, 20), (20, 30))],
                   "pair_cols": int(slow.shape[1])}
    out[tag] = rec
    ex = [rec["Y"]["slow_by_block"][i] - rec["P10"]["slow_by_block"][i] for i in range(3)]
    tot = sum(ex)
    print(tag, "pairs/window", rec["Y"]["pair_cols"], "Y slow", rec["Y"]["slow"], "P10 slow", rec["P10"]["slow"], "fast", rec["Y"]["fast"], "excess by block", ex, "tick0-block share of excess %.3f" % (ex[0] / tot if tot else float("nan")))
json.dump(out, open("F:/TanitAD-scratch/histswap/pre_ceiling.json", "w"), indent=1)
