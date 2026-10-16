#!/usr/bin/env python3
"""Reachability of the A1 lever from its caller (CLAUDE.md 'built, tested, unreachable'): through the SAME patch sh_ttcpen.py applies, on one micro-batch at a moved state,
  lam_ttc 0    -> ep and arm B's gradient EQUAL the unpatched trainer closure's (==, 0.0)
  lam_ttc 0.25 -> ep and the gradient DIFFER (the penalty reaches the loss)
  the Evaluator's R (what J is measured with) is unchanged by the patch.   -> ttcpen_reach.json"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
PC = SH.PC
SF, a, holder, cfg, tr, ev = SH.setup(2)
fn = tr.fn
th0 = [p.detach().clone() for p in tr.base.parameters()]
D = torch.load(f"{PC.SCR}/closed/closed_P_1.0.pt", weights_only=False)["D"][25]
SH._set(tr.armB, [x + d for x, d in zip(th0, D)])
groups = torch.load(os.path.join(PC.BANK_DIR, "bank_s0.pt"), weights_only=False, mmap=True)["groups"]
ids = [int(i) for i in groups[0]]
cxy, ctg, sctx, _ = tr.pool.micro(ids)
sx0 = tr.sx0(ids, cxy, sctx)
pB = list(tr.armB.parameters())
R_eval_before = ev["val"].R.clone()


def ep_grad():
    sxB = tr.armB(cxy, sctx)
    ep = fn["epo_term"](sx0, sxB, ctg, ids, 0, 1.0)
    return float(ep), torch.autograd.grad(ep, pB)


g = fn["epo_term"].__globals__
base_reward = g["epo_reward"]
e0, g0 = ep_grad()
out = {}
for lam in (0.0, 0.25):
    def pen(tg6, recs, rcfg, lam=lam):
        return base_reward(tg6, recs, rcfg) - lam * (1.0 - np.asarray(tg6, np.float64)[..., 3])
    g["epo_reward"] = pen
    try:
        e1, g1 = ep_grad()
    finally:
        g["epo_reward"] = base_reward
    out[f"lam{lam}"] = {"ep": e1, "ep_unpatched": e0, "ep_equal": e1 == e0, "grad_max_abs_diff": max(float((x - y).abs().max()) for x, y in zip(g1, g0))}
out["evaluator_R_unchanged"] = bool(torch.equal(R_eval_before, ev["val"].R))
out["PASS"] = bool(out["lam0.0"]["ep_equal"] and out["lam0.0"]["grad_max_abs_diff"] == 0.0 and (not out["lam0.25"]["ep_equal"]) and out["lam0.25"]["grad_max_abs_diff"] > 0 and out["evaluator_R_unchanged"])
json.dump(out, open(os.path.join(HERE, "ttcpen_reach.json"), "w"), indent=1)
print("ZZREACH", out)
