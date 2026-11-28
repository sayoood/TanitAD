#!/usr/bin/env python3
"""TRAIN-only exploration of rule B(delta) and per-stratum slow shares (adds to p2_explore_train.json's families)."""
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
strat, cru, vgt, gv = D["stratum"][idx], D["cruise"][idx], D["v4_gt"][idx], D["gt_valid"][idx]
s = sr.speed_ratio(v4, v0)
r = np.arange(len(idx))
W = {"zero": 529 / 12146, "random": 11617 / 12146}
mix = lambda v: 100 * (W["zero"] * v[strat == "zero"].mean() + W["random"] * v[strat == "random"].mean())
base = D["pick"][idx]
out = {}
for d in (0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
    pk = sr.pick_B(L, s, d)
    ratio = v4[r, pk] / np.maximum(v0, 1e-6)
    sl = ratio < 0.85
    out[str(d)] = {"mix_pdms_x100": mix(pdms[r, pk]), "dmix": mix(pdms[r, pk]) - mix(pdms[r, base]),
                   "rand_pdms_x100": 100 * pdms[r, pk][strat == "random"].mean(), "zero_pdms_x100": 100 * pdms[r, pk][strat == "zero"].mean(),
                   "nc_fail_rand": float((sub[r, pk, 0] < 1)[strat == "random"].mean()),
                   "slow_cruise": float(sl[cru].mean()), "slow_cruise_rand": float(sl[cru & (strat == "random")].mean()),
                   "slow_cruise_zero": float(sl[cru & (strat == "zero")].mean()), "bias_cruise": float((v4[r, pk] - vgt)[cru & gv].mean()),
                   "changed": float((pk != base).mean())}
    print("B(%s)" % d, " ".join("%s=%.3f" % (k, v) for k, v in out[str(d)].items()))
print("n cruise rand/zero", int((cru & (strat == "random")).sum()), int((cru & (strat == "zero")).sum()))
json.dump(out, open(sc.OUT + "/p2_explore_train_b.json", "w"), indent=1)
