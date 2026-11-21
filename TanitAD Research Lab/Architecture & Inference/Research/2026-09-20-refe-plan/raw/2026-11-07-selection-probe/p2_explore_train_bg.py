#!/usr/bin/env python3
"""TRAIN-only exploration of the guarded band rule BG(delta, eps), ungated and ego-gated (v0 > 5, |ax0| < 0.3)."""
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
out = {}
for gated in (False, True):
    for d in (0.05, 0.1, 0.2, 0.3):
        for eps in (0.0, 0.01, 0.02, 0.05):
            pk = sr.pick_BG(L, s, d, eps)
            if gated:
                pk = np.where(G, pk, base)
            sl = v4[r, pk] / np.maximum(v0, 1e-6) < 0.85
            o = {"dmix": mix(pdms[r, pk]) - mix(pdms[r, base]), "drand": 100 * (pdms[r, pk] - pdms[r, base])[strat == "random"].mean(),
                 "n_rand_worse": int(((pdms[r, pk] < pdms[r, base] - 1e-9) & (strat == "random")).sum()),
                 "n_rand_better": int(((pdms[r, pk] > pdms[r, base] + 1e-9) & (strat == "random")).sum()),
                 "slow_cruise": float(sl[cru].mean()), "bias_cruise": float((v4[r, pk] - vgt)[cru & gv].mean()), "changed": float((pk != base).mean())}
            nm = ("g" if gated else "u") + "BG(%s,%s)" % (d, eps)
            out[nm] = o
            print(nm.ljust(16), " ".join("%s=%.3f" % (k, v) for k, v in o.items()))
json.dump(out, open(sc.OUT + "/p2_explore_train_bg.json", "w"), indent=1)
