"""EXPLORATORY (2026-09-27, not pre-registered): do the damped kinematic floors keep their lead over ha0_ext at 6 s?
Same 2,399 stride-4 eval windows / 141 episodes as the trunk probe; floors built by refav1_arm's own builders at
K=30 x 0.2 s; paired episode-cluster bootstrap (taniteval.ci), n_boot 2000. T1 (t0 state only)."""
import sys, json
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refav1_probe"); sys.path.insert(0, "C:/Users/Admin/tipsnap/b3f7ea6f/taniteval")
import probe as P
from taniteval.ci import paired_episode_cluster_bootstrap as pb

D = P.load(); ok = D["g6_ok"].astype(bool); g6 = D["g6"][ok].astype(np.float64); ep = D["ep"][ok]
h0 = D["ha0_6"][ok].astype(np.float64); hx = D["ha0_ext_6"][ok].astype(np.float64)
arms = {"ha0_6": h0, "ha0_ext_6": hx, "damp50_6": 0.5 * h0 + 0.5 * hx, "kd_x_6": P.kdx(D["ha0_6"], D["ha0_ext_6"])[ok]}
ade = {k: P.ade(v, g6) for k, v in arms.items()}
fde = {k: np.linalg.norm(v[:, -1] - g6[:, -1], axis=-1) for k, v in arms.items()}
out = {"n_windows": int(ok.sum()), "n_clusters": int(len(np.unique(ep))),
       "means": {k: {"ade6": float(ade[k].mean()), "fde6": float(fde[k].mean())} for k in arms}, "pairs": {}}
for a, b in (("damp50_6", "ha0_ext_6"), ("kd_x_6", "ha0_ext_6"), ("kd_x_6", "damp50_6"), ("ha0_6", "ha0_ext_6")):
    out["pairs"][f"{a}-{b}"] = {"ade6": pb(ade[a], ade[b], ep), "fde6": pb(fde[a], fde[b], ep)}
json.dump(out, open("C:/Users/Admin/refav1_probe/floors_6s_exploratory.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}))
for k, v in out["pairs"].items():
    print(k, "ADE6", v["ade6"]["delta"], [v["ade6"]["lo"], v["ade6"]["hi"]], "| FDE6", v["fde6"]["delta"], [v["fde6"]["lo"], v["fde6"]["hi"]])
