#!/usr/bin/env python3
"""SAFEHEAD identity controls (PREREG_SAFEHEAD.md s.9).  Must all pass before any safety-head J number is read.  -> controls.json
  (e)  the view evaluator ('own') == phead_common.Evaluator.evaluate, every per-set array, exactly
  (a') sh_losses with sep OFF == the trainer's train_losses closure on one micro-batch (lB, ep values and arm B's gradient, exactly), at a moved state
  (b)  EPO's gradient on the safety source == 0.0 EXACTLY; arm B's EP row and score_dec != 0 (same breath), per design
  (c)  SAFEHEAD_MUTATE_NO_SEP=1 deletes the separation line: (b) must turn RED (non-zero) for every design
  (a)  K = 3 closed loop, P, lam 1: every design with sep OFF reproduces FULL's arm B bit for bit (max |diff| == 0.0); with sep ON it must differ (the separation is not inert)
    python sh_controls.py [--threads 6] [--K 3]"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
PC = SH.PC


def mx(ts):
    return max((float(t.abs().max()) if t is not None else 0.0) for t in ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--K", type=int, default=3)
    q = ap.parse_args()
    t0 = time.time()
    SF, a, holder, cfg, tr, ev = SH.setup(q.threads)
    fn = tr.fn
    out = {"tool": "sh_controls/1"}
    th0 = [p.detach().clone() for p in tr.base.parameters()]
    zB = torch.load(f"{PC.SCR}/closed/closed_P_1.0.pt", weights_only=False)["D"][25]
    zA = torch.load(f"{PC.SCR}/closed/closed_P_0.0.pt", weights_only=False)["D"][25]
    thB = [x + d for x, d in zip(th0, zB)]                     # a MOVED state: FULL lam 1 at K = 25 (banked)
    thA = [x + d for x, d in zip(th0, zA)]                     # arm A = FULL lam 0 at K = 25 (banked)

    # (e) evaluator identity on the val pool (62 sets) at the moved state
    e = ev["val"]
    ref = e.evaluate(thB)
    mine = SH.eval_views(e, tr, thB, None, ["own"])["own"]
    out["e_eval_identity"] = {k: bool(np.array_equal(ref[k], mine[k])) for k in ref}
    out["e_pass"] = all(out["e_eval_identity"].values())
    print("(e)", out["e_pass"], f"{time.time()-t0:.0f}s", flush=True)

    # one micro-batch
    groups = torch.load(os.path.join(PC.BANK_DIR, "bank_s0.pt"), weights_only=False, mmap=True)["groups"]
    ids = [int(i) for i in groups[0]]
    cxy, ctg, sctx, _ = tr.pool.micro(ids)
    SH._set(tr.armB, thB); SH._set(tr.armA, thA)
    for p in tr.armB.parameters():
        p.requires_grad_(True)
    for p in tr.armA.parameters():
        p.requires_grad_(True)
    pB = list(tr.armB.parameters())
    namesB = [n for n, _ in tr.armB.named_parameters()]
    iw, ib = namesB.index("score_head.weight"), namesB.index("score_head.bias")
    idec = [i for i, n in enumerate(namesB) if n.startswith("score_dec.")]
    sx0 = tr.sx0(ids, cxy, sctx)

    # (a') loss assembly identity, sep OFF, every design, and the trainer's closure
    a_prime = {}
    sxB = tr.armB(cxy, sctx); sxA = tr.armA(cxy, sctx)
    lA0, lB0, ln0, ep0 = fn["train_losses"](sx0, sxA, sxB, ctg, ids, 0, lam_pg=1.0)
    g0 = torch.autograd.grad(lB0, pB)
    for d in ("SH_F", "SH_A", "SH_SG"):
        sxB = tr.armB(cxy, sctx); sxA = tr.armA(cxy, sctx)
        lA, lB, ln, ep = SH.sh_losses(fn, a, d, False, sx0, sxA, sxB, ctg, ids, 0, lam_pg=1.0)
        g = torch.autograd.grad(lB, pB)
        a_prime[d] = {"lB_equal": bool(float(lB) == float(lB0)), "ep_equal": bool(float(ep) == float(ep0)), "lA_equal": bool(float(lA) == float(lA0)),
                      "gradB_max_abs_diff": max(float((x - y).abs().max()) for x, y in zip(g, g0))}
    out["a_prime"] = a_prime
    out["a_prime_pass"] = all(v["lB_equal"] and v["ep_equal"] and v["lA_equal"] and v["gradB_max_abs_diff"] == 0.0 for v in a_prime.values())
    print("(a')", out["a_prime_pass"], a_prime, flush=True)

    # (b) and (c): EPO gradient on the safety source, live forward, lam_pg 1.0
    probe = SF.ScorerCopy(holder)                                # a live, grad-enabled copy of the base = SH_F's safety source for the probe
    for p in probe.parameters():
        p.requires_grad_(True)
    pP, pA = list(probe.parameters()), list(tr.armA.parameters())

    def probe_design(d):
        sxB = tr.armB(cxy, sctx)
        S = {"SH_F": probe(cxy, sctx), "SH_A": tr.armA(cxy, sctx), "SH_SG": sxB}[d]
        comb = SH.combine(d, sxB, S, True)
        ep = fn["epo_term"](sx0, comb, ctg, ids, 0, 1.0)
        src = {"SH_F": pP, "SH_A": pA, "SH_SG": []}[d]
        gs = torch.autograd.grad(ep, src + pB, allow_unused=True)
        gsrc, gB = gs[:len(src)], gs[len(src):]
        none_src = sum(1 for x in gsrc if x is None)
        if d == "SH_SG":
            safe = [gB[iw][list(SH.SAFE)], gB[ib][list(SH.SAFE)]]
        else:
            safe = list(gsrc)
        return {"safety_grad_max_abs": mx(safe), "safety_tensors": len(safe), "safety_grads_None_not_in_graph": none_src if d != "SH_SG" else 0,
                "ctrl_B_EP_row_grad_max_abs": float(gB[iw][2].abs().max()), "ctrl_B_score_dec_grad_l2": float(torch.cat([gB[i].reshape(-1) for i in idec]).norm()),
                "ctrl_B_safe_rows_grad_max_abs": float(torch.cat([gB[iw][list(SH.SAFE)].reshape(-1), gB[ib][list(SH.SAFE)]]).abs().max()), "ep": float(ep)}

    res_b, res_c = {}, {}
    for d in ("SH_F", "SH_A", "SH_SG"):
        res_b[d] = probe_design(d)
    os.environ["SAFEHEAD_MUTATE_NO_SEP"] = "1"
    try:
        for d in ("SH_F", "SH_A", "SH_SG"):
            res_c[d] = probe_design(d)
    finally:
        os.environ.pop("SAFEHEAD_MUTATE_NO_SEP", None)
    out["b"], out["c"] = res_b, res_c
    out["b_pass"] = all(v["safety_grad_max_abs"] == 0.0 and v["ctrl_B_EP_row_grad_max_abs"] > 0 and v["ctrl_B_score_dec_grad_l2"] > 0 for v in res_b.values())
    out["c_red"] = all(v["safety_grad_max_abs"] > 0 for v in res_c.values())
    print("(b)", out["b_pass"], {d: v["safety_grad_max_abs"] for d, v in res_b.items()}, "(c) RED", out["c_red"], {d: v["safety_grad_max_abs"] for d, v in res_c.items()}, f"{time.time()-t0:.0f}s", flush=True)

    # (a) K-step closed-loop identity: sep OFF == FULL bit for bit; sep ON must differ
    _, _, thF, _ = SH.run_closed(SF, a, tr, ev, "FULL", 1.0, "P", q.K, set(), 0, name="FULL")
    res_a = {}
    for d in ("SH_F", "SH_A", "SH_SG"):
        _, _, th_off, _ = SH.run_closed(SF, a, tr, ev, d, 1.0, "P", q.K, set(), 0, sep=False, name=f"{d}_off")
        _, _, th_on, _ = SH.run_closed(SF, a, tr, ev, d, 1.0, "P", q.K, set(), 0, sep=True, name=f"{d}_on")
        res_a[d] = {"sep_off_vs_FULL_max_abs": max(float((x - y).abs().max()) for x, y in zip(th_off, thF)),
                    "sep_on_vs_FULL_max_abs": max(float((x - y).abs().max()) for x, y in zip(th_on, thF))}
        print("(a)", d, res_a[d], f"{time.time()-t0:.0f}s", flush=True)
    out["a"] = res_a
    out["a_pass"] = all(v["sep_off_vs_FULL_max_abs"] == 0.0 for v in res_a.values())
    out["a_sep_not_inert"] = all(v["sep_on_vs_FULL_max_abs"] > 0.0 for v in res_a.values())
    out["ALL_PASS"] = bool(out["e_pass"] and out["a_prime_pass"] and out["b_pass"] and out["c_red"] and out["a_pass"] and out["a_sep_not_inert"])
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "controls.json"), "w"), indent=1)
    print("ZZCONTROLS", "ALL_PASS" if out["ALL_PASS"] else "FAIL", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
