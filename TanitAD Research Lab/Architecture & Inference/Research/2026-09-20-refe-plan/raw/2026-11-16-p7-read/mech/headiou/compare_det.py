"""Run-twice determinism on 2 tokens: head_pred npz (logit, sdf_m) and the cached ctx + scorer logits must be bit-identical between run A and run B.
usage: compare_det.py detA detB ctxA ctxB out.json"""
import glob
import json
import os
import sys

import numpy as np

A, B, CA, CB, out = sys.argv[1:6]
res = {"head_pred": {}, "cache": {}}
ok = True
for f in sorted(glob.glob(A + "/head_pred_*.npz")):
    g = os.path.join(B, os.path.basename(f))
    za, zb = np.load(f), np.load(g)
    r = {k: bool(np.array_equal(za[k], zb[k])) for k in ("log", "token", "logit", "sdf_m")}
    r["n"] = int(len(za["token"]))
    res["head_pred"][os.path.basename(f)] = r
    ok &= all(v for k, v in r.items() if k != "n")
for f in sorted(glob.glob(CA + "/*.npy")):
    g = os.path.join(CB, os.path.basename(f))
    same = os.path.exists(g) and bool(np.array_equal(np.load(f), np.load(g)))
    res["cache"][os.path.basename(f)] = same
    ok &= same
res["n_cache_files"] = len(res["cache"])
res["bit_identical"] = bool(ok and res["n_cache_files"] >= 4 and res["head_pred"])
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w", encoding="utf-8"), indent=1, sort_keys=True)
print("DETERMINISM", res["bit_identical"], flush=True)
sys.exit(0 if res["bit_identical"] else 3)
