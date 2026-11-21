#!/usr/bin/env python3
"""RL-2 stage 0 (BEFORE the PREREG is hashed): OUTCOME-FREE facts needed to fix the clip grid.
Reads ONLY the logit dumps (np.load of the npz); never loads a label file, never calls rlh.load_labels.
Outputs: sha256 of the three step-0 base dumps, key-alignment checks, quantiles of |lane delta| = |B_lane - base0_lane| pooled over all sets x proposals."""
import hashlib, json, sys
import numpy as np
DL0 = "C:/Users/Admin/rls1b_dl"
DLD = "C:/Users/Admin/par18/rl/r0d_dl"
PATHS = {"r0": (DL0 + "/r0/final", 1000), "r0b": (DL0 + "/r0b/final", 500), "r0d": (DLD + "/r0d/final", 1000)}
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
out = {"sha256_step0_base": {}, "alignment": {}, "lane_absdelta_quantiles": {}, "lane_absdelta_stats": {}}
Q = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
base = {}
for arm, (d, u) in PATHS.items():
    pb = d + "/scores_step0_base.npz"
    out["sha256_step0_base"][arm] = sha(pb)
    zb = np.load(pb, allow_pickle=False)
    zB = np.load(f"{d}/scores_update{u}_B.npz", allow_pickle=False)
    kb = [tuple(map(str, r)) for r in zip(zb["log"], zb["token"], zb["step"], zb["rank"])]
    kB = [tuple(map(str, r)) for r in zip(zB["log"], zB["token"], zB["step"], zB["rank"])]
    out["alignment"][arm] = {"n_base": len(kb), "n_B": len(kB), "keys_identical_order": kb == kB, "shape": list(zb["scores"].shape), "dtype": str(zb["scores"].dtype)}
    d6 = np.abs(np.asarray(zB["scores"], np.float64)[..., 6] - np.asarray(zb["scores"], np.float64)[..., 6]).ravel()
    out["lane_absdelta_quantiles"][arm] = {str(q): float(np.quantile(d6, q)) for q in Q}
    out["lane_absdelta_stats"][arm] = {"mean": float(d6.mean()), "max": float(d6.max()), "frac_zero": float((d6 == 0).mean())}
    base[arm] = np.asarray(zb["scores"], np.float64)
out["base0_arrays_equal"] = {"r0_vs_r0b": bool(np.array_equal(base["r0"], base["r0b"])), "r0_vs_r0d": bool(np.array_equal(base["r0"], base["r0d"]))}
out["base0_maxabs_diff"] = {"r0_vs_r0b": float(np.abs(base["r0"] - base["r0b"]).max()) if base["r0"].shape == base["r0b"].shape else None,
                            "r0_vs_r0d": float(np.abs(base["r0"] - base["r0d"]).max()) if base["r0"].shape == base["r0d"].shape else None}
json.dump(out, open("stage0_taus.json", "w"), indent=1)
print(json.dumps(out, indent=1))
