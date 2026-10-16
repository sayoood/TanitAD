#!/usr/bin/env python3
"""per-set evaluation arrays of the BANKED closed-loop full-arm states (closed_P_{0.0,1.0}.pt, K=25/50) -> full_ref.pkl (the f denominator and the A control)."""
import os, pickle, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
SF, a, holder, cfg, tr, ev = PC.setup(int(os.environ.get("PH_THREADS", "4")))
th0 = PC.theta0_of(tr)
out = {"theta0": {pn: e.evaluate(th0) for pn, e in ev.items()}, "logs": {pn: e.logs for pn, e in ev.items()}}
for lam in ("0.0", "1.0"):
    z = torch.load(f"{PC.SCR}/closed/closed_P_{lam}.pt", weights_only=False)
    for k in (25, 50):
        th = [x + d for x, d in zip(th0, z["D"][k])]
        out[(float(lam), k)] = {pn: e.evaluate(th) for pn, e in ev.items()}
        out[(float(lam), k, "disp_norm")] = float(torch.cat([d.reshape(-1) for d in z["D"][k]]).norm())
pickle.dump(out, open(os.path.join(HERE, "full_ref.pkl"), "wb"))
print("ZZFULLREF_OK")
