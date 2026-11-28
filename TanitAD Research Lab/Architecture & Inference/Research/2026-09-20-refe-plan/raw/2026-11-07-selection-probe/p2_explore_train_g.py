#!/usr/bin/env python3
"""TRAIN-only exploration of EGO-GATED variants: the rule acts only where the CURRENT ego state reads cruise-like
(v0 > 5 m/s and |ax0| < 0.3 m/s^2 -- current measured ego only, no future), elsewhere the banked pick is kept."""
import json

import numpy as np

import sel_common as sc
import sel_rules as sr

D = sc.load()
split = json.load(open(sc.OUT + "/split.json"))
tr = np.isin(D["log"], split["train_logs"])
assert not np.isin(D["log"][tr], split["heldout_logs"]).any()
idx = np.where(tr)[0]
L, pdms, sub, v0, v4 = D["logits"][idx], D["pdms"][idx], D["sub"][idx], D["v0"][idx], D["v4_all"][idx]
strat, cru, vgt, gv, ax0 = D["stratum"][idx], D["cruise"][idx], D["v4_gt"][idx], D["gt_valid"][idx], D["ax0"][idx]
s = sr.speed_ratio(v4, v0)
r = np.arange(len(idx))
G = (v0 > 5.0) & (np.abs(ax0) < 0.3)
W = {"zero": 529 / 12146, "random": 11617 / 12146}
mix = lambda v: 100 * (W["zero"] * v[strat == "zero"].mean() + W["random"] * v[strat == "random"].mean())
base = D["pick"][idx]
print("gate share all", G.mean(), "gate share cruise", G[cru].mean(), "gate&~cruise", int((G & ~cru).sum()), "n", len(idx))
out = {}
def rep(name, pk):
    pk = np.where(G, pk, base)
    ratio = v4[r, pk] / np.maximum(v0, 1e-6)
    sl = ratio < 0.85
    o = {"dmix": mix(pdms[r, pk]) - mix(pdms[r, base]), "drand": 100 * (pdms[r, pk] - pdms[r, base])[strat == "random"].mean(),
         "dzero": 100 * (pdms[r, pk] - pdms[r, base])[strat == "zero"].mean(), "nc_fail_rand": float((sub[r, pk, 0] < 1)[strat == "random"].mean()),
         "slow_cruise": float(sl[cru].mean()), "slow_gate": float(sl[G].mean()), "bias_cruise": float((v4[r, pk] - vgt)[cru & gv].mean()),
         "bias_gate": float((v4[r, pk] - vgt)[G & gv].mean()), "changed": float((pk != base).mean())}
    out[name] = o
    print(name.ljust(10), " ".join("%s=%.3f" % (k, v) for k, v in o.items()))
for lam in (0.5, 1, 2, 4, 8):
    rep("Sg(%s)" % lam, sr.pick_S(L, s, lam))
for d in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
    rep("Bg(%s)" % d, sr.pick_B(L, s, d))
for t in (0.8, 0.85, 0.9):
    rep("Vg(%s)" % t, sr.pick_V(L, s, t))
json.dump(out, open(sc.OUT + "/p2_explore_train_g.json", "w"), indent=1)
