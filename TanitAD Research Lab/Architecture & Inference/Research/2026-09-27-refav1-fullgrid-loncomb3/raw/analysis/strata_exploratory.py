"""EXPLORATORY (2026-09-27): the full-grid planner vs damped floors inside GT-only strata (turn / straight / stop /
accelerate), ADE @ 2 s, paired episode-cluster bootstrap. Strata definitions = the trunk probe's SPEC A2-prime."""
import sys, glob, json
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refav1_probe"); sys.path.insert(0, "C:/Users/Admin/tipsnap/b3f7ea6f/taniteval")
import probe as P
from taniteval.ci import paired_episode_cluster_bootstrap as pb

R = "C:/Users/Admin/refav1_fullgrid"


def load(d):
    Z = [np.load(f) for f in sorted(glob.glob(d + "/ep*.npz"))]
    return ({k: np.concatenate([z[k] for z in Z]) for k in ("g", "v0", "cl", "ha0", "ha0_ext")},
            np.concatenate([np.full(len(z["ws"]), i) for i, z in enumerate(Z)]))


A, ep = load(f"{R}/dump_loncomb3_s0"); B, _ = load(f"{R}/dump_loncomb3_s1")
g = A["g"].astype(np.float64); v0 = A["v0"].astype(np.float64)
floors = {"damp50": 0.5 * A["ha0"] + 0.5 * A["ha0_ext"], "kd_x": P.kdx(A["ha0"], A["ha0_ext"]), "ha0_ext": A["ha0_ext"]}
d = g[:, -1] - g[:, -2]; head = np.degrees(np.abs(np.arctan2(d[:, 1], d[:, 0]))); vt = np.linalg.norm(d, axis=-1) / 0.2
S = {"turn": head > 15, "straight": head < 5, "stop": (v0 > 3) & (vt < 1), "accelerate": (vt - v0) > 1.5}
out = {}
for name, m in S.items():
    ncl = int(len(np.unique(ep[m])))
    row = {"n_windows": int(m.sum()), "n_clusters": ncl}
    if ncl >= 10:
        for lab, arm in (("A1", A["cl"]), ("A2", B["cl"])):
            for fn, fl in floors.items():
                row[f"{lab}-{fn}"] = pb(P.ade(arm[m].astype(np.float64), g[m]), P.ade(fl[m].astype(np.float64), g[m]), ep[m])
    else:
        row["status"] = "UNDERPOWERED"
    out[name] = row
json.dump(out, open(f"{R}/analysis/strata_exploratory.json", "w"), indent=1)
print({k: (v["n_windows"], v["n_clusters"]) for k, v in out.items()})
