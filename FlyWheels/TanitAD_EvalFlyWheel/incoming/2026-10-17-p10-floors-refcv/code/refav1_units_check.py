"""S1 side-finding: on the refav1 grid (feat_s4, 141 eps / 2,399 windows) the banked ha0_ext_6 integrates the recorded
STEER ANGLE as a curvature (extract.py calls paths_from_controls with its default action_units="kappa"; actions[:,0] is
arctan(L*kappa), ~2.9x kappa).  Rebuild the floors with the CORRECT unit and compare 6 s ADE.  MEASURED, CPU, seconds."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s1_lib as L
from taniteval import floors_kin as fk
from taniteval.ci import paired_episode_cluster_bootstrap as pb
sys.path.insert(0, "C:/Users/Admin/refav1_probe"); import probe as P
E = P.load(); ok = E["g6_ok"].astype(bool); ep = E["ep"][ok]
g6 = E["g6"][ok].astype(np.float64); kin = E["kin"][ok].astype(np.float64); v0 = kin[:, 0]
h0 = fk.ha0_paths(v0, 0.2, 30)
arms = {}
for units in ("kappa", "steer"):
    he = fk.ha0_ext_paths(kin, 0.2, 30, action_units=units)
    arms[f"ha0_ext[{units}]"] = he
    arms[f"damp50[{units}]"] = fk.damp50(h0, he)
    arms[f"kd_x[{units}]"] = fk.kd_x(h0, he).astype(np.float64)
bank = E["ha0_ext_6"][ok].astype(np.float64)
res = {"n_windows": int(ok.sum()), "n_episodes": int(len(np.unique(ep))),
       "rebuilt_kappa_vs_banked_ha0_ext_6_max_abs": float(np.abs(arms["ha0_ext[kappa]"] - bank).max()),
       "rebuilt_steer_vs_banked_ha0_ext_6_max_abs": float(np.abs(arms["ha0_ext[steer]"] - bank).max()),
       "mean_abs_steer_rad": float(np.abs(kin[:, 2]).mean()),
       "ade6": {k: float(fk.ade(v, g6).mean()) for k, v in arms.items()}, "paired": {}}
for a, b in (("damp50[kappa]", "ha0_ext[kappa]"), ("damp50[steer]", "ha0_ext[steer]"), ("ha0_ext[steer]", "ha0_ext[kappa]"),
             ("kd_x[steer]", "kd_x[kappa]")):
    res["paired"][f"{a}-{b}"] = pb(fk.ade(arms[a], g6), fk.ade(arms[b], g6), ep, n_boot=2000, seed=0)
json.dump(res, open(os.path.join(HERE, "..", "raw", "refav1_units_check.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "paired"}, indent=1))
for k, v in res["paired"].items(): print(k, v["delta"], v["lo"], v["hi"])
