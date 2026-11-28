#!/usr/bin/env python3
"""RL-2 post-verdict SENSITIVITY (not part of the pre-registration, does not change any row): Monte-Carlo precision of the HELD interval of the
pre-selected c* (R0 u1000 L0.25; R0d u1000 L0.25).  Re-draws the bootstrap with 10 other seeds on the SAME HELD sets; no config is added, none is selected."""
import json, sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlh
from rlh import pick, compose
SPLIT_SEED, N_FIT = 20261019, 38
kv, ck, s, logs, st = rlh.load_labels(); N = len(kv); ar = np.arange(N)
ul = sorted(set(logs.tolist())); pl = np.random.default_rng(SPLIT_SEED).permutation(len(ul)); fs = set(ul[i] for i in pl[:N_FIT])
mh = ~np.asarray([l in fs for l in logs.tolist()])
base0 = rlh.load_dump(f"{rlh.DL}/r0/final/scores_step0_base.npz", kv, ck); ib0 = pick(base0)
out = {}
for arm, dl in (("r0", rlh.DL), ("r0d", "C:/Users/Admin/par18/rl/r0d_dl")):
    B = rlh.load_dump(f"{dl}/{arm}/final/scores_update1000_B.npz", kv, ck)
    x = compose(B, base0, (2, 4)); x[..., 6] = 0.75 * base0[..., 6] + 0.25 * B[..., 6]
    ip = pick(x)
    rows = []
    for sd in [20260927] + list(range(1, 11)):
        bt = rlh.Boot(logs[mh], seed=sd)
        r = {q: bt.ci((100.0 * (s[q][ar, ip] - s[q][ar, ib0]))[mh]) for q in ("pdms", "nc", "ttc")}
        rows.append({"seed": sd, "pdms_lb": r["pdms"][1], "nc_ub": r["nc"][2], "ttc_ub": r["ttc"][2], "pdms_pt": r["pdms"][0]})
    lbs = [r["pdms_lb"] for r in rows[1:]]
    nc_exact = float(np.mean((100.0 * (s["nc"][ar, ip] - s["nc"][ar, ib0]))[mh]))
    out[arm] = {"rows": rows, "pdms_lb_other_seeds": {"min": min(lbs), "max": max(lbs), "frac_gt0": float(np.mean(np.asarray(lbs) > 0))},
                "nc_ub_other_seeds_max": max(r["nc_ub"] for r in rows[1:]), "nc_ub_other_seeds_min": min(r["nc_ub"] for r in rows[1:]),
                "n_held_sets": int(mh.sum()), "n_picks_differ_from_base0_held": int((ip != ib0)[mh].sum()),
                "n_nc_worse_sets": int(((s["nc"][ar, ip] < s["nc"][ar, ib0]) & mh).sum()), "n_nc_better_sets": int(((s["nc"][ar, ip] > s["nc"][ar, ib0]) & mh).sum())}
    print(arm, json.dumps({k: v for k, v in out[arm].items() if k != "rows"}), flush=True)
json.dump(out, open("../out/raw/sensitivity_heldseed.json", "w"), indent=1)
