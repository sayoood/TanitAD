#!/usr/bin/env python3
"""P2 exploration on the TRAIN logs ONLY (split.json). Every PDMS read here is masked to TRAIN before it is computed.
Writes p2_explore_train.json. The held-out logs are never touched by this script (asserted)."""
import json

import numpy as np

import sel_common as sc
import sel_rules as sr

D = sc.load()
split = json.load(open(sc.OUT + "/split.json"))
tr = np.isin(D["log"], split["train_logs"])
assert not np.isin(D["log"][tr], split["heldout_logs"]).any()
idx = np.where(tr)[0]
L = D["logits"][idx]
pdms = D["pdms"][idx]
sub = D["sub"][idx]
v0 = D["v0"][idx]
v4 = D["v4_all"][idx]
log = D["log"][idx]
strat = D["stratum"][idx]
cru = D["cruise"][idx]
vgt = D["v4_gt"][idx]
gv = D["gt_valid"][idx]
s = sr.speed_ratio(v4, v0)
n = len(idx)
r = np.arange(n)
W = {"zero": 529 / 12146, "random": 11617 / 12146}
ul, li, draws = sc.log_boot_idx(log, 2000, 20260927)


def mix(vals):
    pz = vals[strat == "zero"].mean()
    pr = vals[strat == "random"].mean()
    return 100 * (W["zero"] * pz + W["random"] * pr)


def summarize(pk, base):
    pd = pdms[r, pk]
    ratio = v4[r, pk] / np.maximum(v0, 1e-6)
    return {"mix_pdms_x100": mix(pd), "dmix_vs_base_x100": mix(pd) - mix(pdms[r, base]),
            "random_stratum_pdms_x100": 100 * pd[strat == "random"].mean(), "zero_stratum_pdms_x100": 100 * pd[strat == "zero"].mean(),
            "nc_fail_share_random": float((sub[r, pk, 0] < 1)[strat == "random"].mean()),
            "cruise_slow_share": float((ratio < 0.85)[cru].mean()),
            "cruise_bias_v4_minus_gt": float((v4[r, pk] - vgt)[cru & gv].mean()),
            "all_bias_v4_minus_gt": float((v4[r, pk] - vgt)[gv].mean()),
            "pick_changed_share": float((pk != base).mean())}


base = sr.pick_S(L, s, 0)
assert (base == D["pick"][idx]).all()
out = {"n_train_tokens": n, "n_train_logs": len(split["train_logs"]), "base": summarize(base, base), "S": {}, "L": {}, "E": {}, "V": {}}
for lam in (0, 0.5, 1, 2, 3, 4, 6, 8, 12):
    out["S"][str(lam)] = summarize(sr.pick_S(L, s, lam), base)
for lam in (0, 0.25, 0.5, 0.75, 1.0):
    out["L"][str(lam)] = summarize(sr.pick_L(L, lam), base)
for lam in (0, 0.5, 1, 2, 4, 8):
    out["E"][str(lam)] = summarize(sr.pick_E(L, lam), base)
for tau in (0, 0.7, 0.8, 0.85, 0.9):
    out["V"][str(tau)] = summarize(sr.pick_V(L, s, tau), base)
json.dump(out, open(sc.OUT + "/p2_explore_train.json", "w"), indent=1)
keys = ("mix_pdms_x100", "dmix_vs_base_x100", "random_stratum_pdms_x100", "zero_stratum_pdms_x100", "nc_fail_share_random",
        "cruise_slow_share", "cruise_bias_v4_minus_gt", "all_bias_v4_minus_gt", "pick_changed_share")
print("rule  " + " ".join(k[:14] for k in keys))
for fam in ("S", "L", "E", "V"):
    for k, v in out[fam].items():
        print(f"{fam}({k})".ljust(8), " ".join("%8.3f" % v[q] for q in keys))
