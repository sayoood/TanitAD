#!/usr/bin/env python3
"""PHEAD closed loop for HEAD-ONLY groups on CACHED decoder features.  With score_q_mlp and score_dec frozen (kind R, groups PH_A / PH_B / PH_C), the scoring decoder's output s (the input of score_head)
does not depend on theta, so it is computed ONCE per set with the real modules (REFe.score_trajectories lines 706-708: q_mlp -> 4 CrossBlocks) and only `score_head(s)` is re-run per update.  Everything else is the
real closed-loop step of phead_closed.py: the trainer's own train_losses closure, per-micro backward / accum, Restricted clip + AdamW, stream draws.  Validated against the real-pipeline PH_C runs (check_cached_vs_real.py).
    python phead_closed_cached.py --group PH_A_EPROW --lam 1.0 --regime P --K 50 --stream 0 --lr-mult 1 --out-dir closed_cached
"""
import argparse, os, pickle, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
from adam_sim import draws_for                     # noqa: E402
import adam_common as AC                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--regime", default="P", choices=("P", "T"))
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--streams", type=int, nargs="+", default=[0])
    ap.add_argument("--lr-mults", type=float, nargs="+", default=[1.0])
    ap.add_argument("--snaps", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    q = ap.parse_args()
    t0 = time.time()
    SF, a, holder, cfg, tr, ev = PC.setup(q.threads)
    th0 = PC.theta0_of(tr)
    names = tr.names
    shapes = [tuple(p.shape) for p in th0]
    pool = tr.pool
    # --- cache the decoder features s (theta-independent when q_mlp and score_dec are frozen) with the real modules
    S = []
    with torch.no_grad():
        for i in range(0, len(pool.keys), 8):
            ids = list(range(i, min(i + 8, len(pool.keys))))
            cxy, ctg, sctx, _ = pool.micro(ids)
            s = tr.armB.score_q_mlp(cxy.flatten(2))
            for blk in tr.armB.score_dec:
                s = blk(s, sctx)
            S.append(s.float())
    S = torch.cat(S)                                                   # [N, 64, 256]
    ev["train"].evaluate(th0)                                          # fills the base-logit cache (ev.sx0)
    SX0 = ev["train"].sx0
    head = tr.armB.score_head
    ih = [names.index("score_head.weight"), names.index("score_head.bias")]
    for p in tr.params:
        p.requires_grad_(False)
    for i in ih:
        tr.params[i].requires_grad_(True)
    # equivalence of the cached path at theta_0: head(S) == the full forward at theta_0 (bitwise or ~1e-6)
    with torch.no_grad():
        ids = list(range(8)); cxy, _c, sctx, _ = pool.micro(ids)
        full0 = tr.armB(cxy, sctx).float(); cached0 = head(S[ids]).float()
    eq0 = float((full0 - cached0).abs().max())
    print("cached-vs-full forward at theta_0 max|diff|", eq0, flush=True)
    for stream in q.streams:
        bank = torch.load(os.path.join(PC.BANK_DIR, f"bank_s{stream}.pt"), weights_only=False, mmap=True)
        groups = bank["groups"]
        draws = draws_for(stream, q.K, groups.shape[0])
        for g in q.groups:
            sl = PC.group_slices(g, names, shapes)
            for lrm in q.lr_mults:
                for lam in q.lams:
                    a.epo_lam = float(lam)
                    with torch.no_grad():
                        for p, t in zip(tr.params, th0):
                            p.copy_(t)
                    R = PC.Restricted(th0, sl, lr=a.lr * lrm)
                    log, snaps = [], {}
                    for t in range(q.K):
                        gacc = None
                        tot_ep = tot_J = 0.0
                        for j in draws[t]:
                            ids = [int(i) for i in groups[j]]
                            ctg = torch.from_numpy(pool.ctg[ids])
                            sxB = head(S[ids]).float()
                            lam_pg = None if q.regime == "T" else float(lam)
                            lA, lB, ln, ep = tr.fn["train_losses"](SX0[ids], sxB.detach(), sxB, ctg, ids, t, lam_pg=lam_pg)
                            (lB / a.accum).backward()
                            gl = [None] * len(tr.params)
                            for i in ih:
                                gl[i] = tr.params[i].grad.detach().clone()
                                tr.params[i].grad = None
                            gacc = gl if gacc is None else [(x if y is None else x + y) for x, y in zip(gacc, gl)]
                            tot_ep += float(ep) / a.accum
                            tot_J += float(tr.epo_last.get("J", 0.0)) / a.accum
                        lr = (AC.lr_at(t, a) if q.regime == "T" else a.lr) * lrm
                        n = R.step(gacc, lr)
                        th = R.theta()
                        with torch.no_grad():
                            for i in ih:
                                tr.params[i].copy_(th[i])
                        log.append({"t": t, "grad_norm_pre_clip": n, "epo": tot_ep, "J_train_batch": tot_J, "lr": lr})
                        if (t + 1) in q.snaps:
                            snaps[t + 1] = {"eval": {pn: e.evaluate(th) for pn, e in ev.items()}, "outside": PC.outside_group_displacement(th, th0, sl),
                                            "disp_norm": float(torch.cat([(x - y).reshape(-1) for x, y in zip(th, th0)]).norm())}
                            for i in ih:
                                tr.params[i].requires_grad_(True)
                    name = f"closed_s{stream}_{q.regime}_{g}_R_lr{lrm:g}_lam{lam:g}{q.tag}"
                    pickle.dump({"name": name, "args": vars(q), "cached_vs_full_theta0_maxdiff": eq0, "log": log, "snaps": snaps}, open(os.path.join(q.out_dir, name + ".pkl"), "wb"))
                    print(f"ZZCACHED_OK {name} J(K={max(snaps)}) train {snaps[max(snaps)]['eval']['train']['J'].mean():.5f} val {snaps[max(snaps)]['eval']['val']['J'].mean():.5f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
