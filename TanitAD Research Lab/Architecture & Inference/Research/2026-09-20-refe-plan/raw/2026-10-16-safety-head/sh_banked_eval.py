#!/usr/bin/env python3
"""SAFEHEAD regime-P controls from the BANKED full-arm states (closed_P_{0.0,1.0}.pt, K = 25 / 50; the same states phead's full_ref.pkl evaluates).
  FULL lam 0 (= arm A) state, view 'own'  -> the matched control of FULL, SH_A, SH_SG   (cross-checked against full_ref.pkl: must be equal)
  FULL lam 0 (= arm A) state, view 'SH_F' -> SH_F's matched control (A's progress rows + base safety rows)
  FULL lam 1 state, views 'own' / 'SH_F'  -> FULL (cross-check vs full_ref.pkl) and, informational, FULL's B read through the base safety rows
-> out/banked_P.pkl"""
import os, pickle, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
PC = SH.PC
t0 = time.time()
SF, a, holder, cfg, tr, ev = SH.setup(int(os.environ.get("SH_THREADS", "6")))
th0 = [p.detach().clone() for p in tr.base.parameters()]
FR = pickle.load(open(os.path.join(SH.PHD, "full_ref.pkl"), "rb"))
out = {"logs": {pn: e.logs for pn, e in ev.items()}, "xcheck_vs_full_ref": {}}
for lam in (0.0, 1.0):
    z = torch.load(f"{PC.SCR}/closed/closed_P_{lam}.pt", weights_only=False)
    for k in (25, 50):
        th = [x + d for x, d in zip(th0, z["D"][k])]
        out[(lam, k)] = {pn: SH.eval_views(e, tr, th, None, ["own", "SH_F"]) for pn, e in ev.items()}
        out["xcheck_vs_full_ref"][f"lam{lam}_K{k}"] = {pn: float(np.abs(out[(lam, k)][pn]["own"]["J"] - FR[(lam, k)][pn]["J"]).max()) for pn in ev}
        print(lam, k, out["xcheck_vs_full_ref"][f"lam{lam}_K{k}"], f"{time.time()-t0:.0f}s", flush=True)
pickle.dump(out, open(os.path.join(HERE, "out", "banked_P.pkl"), "wb"))
print("ZZBANKED_OK", f"{time.time()-t0:.0f}s", flush=True)
