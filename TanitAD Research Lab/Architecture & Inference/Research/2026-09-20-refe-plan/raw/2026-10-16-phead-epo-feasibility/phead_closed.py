#!/usr/bin/env python3
"""PHEAD closed-loop arms (PREREG_PHEAD.md s.4): adam_closed.py's real trainer step (the trainer's own train_losses, per-micro backward / accum, clip, lr, AdamW) on moving theta,
with the restriction applied.  kind R: the group's entries are the ONLY trainable ones (everything else requires_grad False and is not in the optimiser or the clip).
kind E: the EPO gradient is masked to the group, BCE moves everything.  One process = one (stream, regime, group, kind, lam, lr_mult); K updates; snapshots evaluated on both pools.
    python phead_closed.py --group PH_C_HEAD --kind R --lam 1.0 --regime P --K 50 --out-dir <dir> [--stream 0] [--lr-mult 1] [--identity-vs-banked]
"""
import argparse, os, pickle, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
from adam_sim import draws_for                     # noqa: E402
import adam_common as AC                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--kind", default="R", choices=("R", "E"))
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--regime", default="P", choices=("P", "T"))
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--snaps", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--identity-vs-banked", action="store_true")
    ap.add_argument("--tag", default="")
    q = ap.parse_args()
    t0 = time.time()
    SF, a, holder, cfg, tr, ev = PC.setup(q.threads)
    a.epo_lam = float(q.lam)
    th0 = PC.theta0_of(tr)
    names = tr.names
    shapes = [tuple(p.shape) for p in th0]
    sl = PC.group_slices(q.group, names, shapes)
    bank = torch.load(os.path.join(PC.BANK_DIR, f"bank_s{q.stream}.pt"), weights_only=False, mmap=True)
    groups = bank["groups"]
    draws = draws_for(q.stream, q.K, groups.shape[0])
    full = [(i, None) for i in range(len(th0))]
    R = PC.Restricted(th0, sl if q.kind == "R" else full, lr=a.lr * q.lr_mult)
    trainable = {i for i, _ in R.sl}
    for i, p in enumerate(tr.params):
        p.requires_grad_(i in trainable)
    mask = PC.flat_mask(sl, shapes) if q.kind == "E" else None
    name = f"closed_s{q.stream}_{q.regime}_{q.group}_{q.kind}_lr{q.lr_mult:g}_lam{q.lam:g}{q.tag}"
    log, snaps = [], {}
    for t in range(q.K):
        tot_bce = tot_ep = tot_J = 0.0
        gacc = None
        for j in draws[t]:
            ids = [int(i) for i in groups[j]]
            cxy, ctg, sctx, _ = tr.pool.micro(ids)
            sx0 = tr.sx0(ids, cxy, sctx)
            sxB = tr.forward(ids, cxy, sctx)
            lam_pg = None if q.regime == "T" else float(q.lam)
            lA, lB, ln, ep = tr.fn["train_losses"](sx0, sxB.detach(), sxB, ctg, ids, t, lam_pg=lam_pg)
            if q.kind == "R" and not trainable:                 # EMPTY group: nothing may move, nothing is back-propagated
                gl = [None] * len(tr.params)
            elif q.kind == "R":
                (lB / a.accum).backward()
                g = [p.grad if p.grad is not None else None for p in tr.params]
                gl = [(x.detach().clone() if x is not None else None) for x in g]
                for p in tr.params:
                    p.grad = None
            else:
                tp = [p for p in tr.params]
                bce = lB - ep
                gB = torch.autograd.grad(bce / a.accum, tp, retain_graph=True)
                gE = torch.autograd.grad(ep / a.accum, tp, allow_unused=True)
                gBf = torch.cat([x.reshape(-1) for x in gB])
                gEf = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(gE, tp)]) * mask
                gl = PC.unflat_t((gBf + gEf).detach(), shapes)
            gacc = gl if gacc is None else [(x if y is None else (y if x is None else x + y)) for x, y in zip(gacc, gl)]
            tot_bce += float(lB - ep) / a.accum
            tot_ep += float(ep) / a.accum
            tot_J += float(tr.epo_last.get("J", 0.0)) / a.accum
        lr = (AC.lr_at(t, a) if q.regime == "T" else a.lr) * q.lr_mult
        n = R.step(gacc, lr)
        th = R.theta()
        with torch.no_grad():
            for p, x in zip(tr.params, th):
                p.copy_(x)
        log.append({"t": t, "grad_norm_pre_clip": n, "bce": tot_bce, "epo": tot_ep, "J_train_batch": tot_J, "lr": lr})
        if t % 5 == 0:
            print(f"{name} step {t} gn {n:.3f} bce {tot_bce:.4f} epo {tot_ep:.4f} J {tot_J:.4f} {time.time()-t0:.0f}s", flush=True)
        if (t + 1) in q.snaps:
            snaps[t + 1] = {"theta": th, "eval": {pn: e.evaluate(th) for pn, e in ev.items()},
                            "outside": PC.outside_group_displacement(th, th0, sl) if q.kind == "R" else None,
                            "disp_norm": float(torch.cat([(x - y).reshape(-1) for x, y in zip(th, th0)]).norm())}
            for p in tr.params:
                p.grad = None
    out = {"name": name, "args": vars(q), "log": log, "snaps": {k: {kk: vv for kk, vv in v.items() if kk != "theta"} for k, v in snaps.items()}}
    if q.identity_vs_banked:
        z = torch.load(f"{PC.SCR}/closed/closed_{q.regime}_{q.lam}.pt", weights_only=False)
        idn = {}
        for k, s in snaps.items():
            if k in z["D"]:
                d = [(x - y) - Dk for x, y, Dk in zip(s["theta"], th0, z["D"][k])]
                dn = float(torch.cat([x.reshape(-1) for x in d]).norm())
                bn = float(torch.cat([x.reshape(-1) for x in z["D"][k]]).norm())
                mx = max(float(x.abs().max()) for x in d)
                idn[k] = {"abs_l2_diff": dn, "banked_l2_norm": bn, "rel_l2_diff": dn / bn, "max_abs_diff": mx}
        out["identity_vs_banked"] = idn
    pickle.dump(out, open(os.path.join(q.out_dir, name + ".pkl"), "wb"))
    print("ZZCLOSED_OK", name, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
