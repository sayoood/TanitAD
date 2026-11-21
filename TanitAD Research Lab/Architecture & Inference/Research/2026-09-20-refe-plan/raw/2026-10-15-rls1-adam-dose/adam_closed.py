#!/usr/bin/env python3
"""H-ADAM section 3.2, CLOSED-LOOP check: the real trainer step (the trainer's own train_losses, per-micro backward / accum, clip, lr, AdamW), gradients recomputed at the moving theta.
One process = one lam (a.epo_lam = lam, so regime T uses the trainer's OWN ramp via lam_pg=None); both regimes; K = 50 updates of 8 micro-batches of 8; draws = stream 0's.
Saves D(lam) at 25 / 50 to <out-dir>/closed_{regime}_{lam}.pt ; closed_stats.py combines the three lams.

    python adam_closed.py --lam 0.25 --out-dir <scratch> --bank-dir <scratch/bank> [--threads 8] [--K 50]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adam_common as AC  # noqa: E402
from adam_sim import draws_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--regimes", default="TP")
    ap.add_argument("--beta", type=float, default=None, help="override a.epo_beta (ablation of the KL anchor); default = R0's 0.1")
    ap.add_argument("--tag", default="")
    q = ap.parse_args()
    if q.threads:
        torch.set_num_threads(q.threads)
    SF = AC.load_trainer()
    a, _ = AC.r0_args(SF)
    if q.beta is not None:
        a.epo_beta = float(q.beta)
    a.epo_lam = float(q.lam)                                   # the closures read a.epo_lam / a.epo_ramp (scorer_finetune.py epo_term)
    holder, cfg = AC.build_scorer(SF, a)
    pool = AC.Pool(AC.prepare_pool(SF, a, cfg, "train")).load_visual_all()
    tr = AC.Trainer(SF, a, holder, pool)
    bank = torch.load(os.path.join(q.bank_dir, "bank_s0.pt"), weights_only=False)
    groups = bank["groups"]
    draws = draws_for(0, q.K, groups.shape[0])
    theta0 = [p.detach().clone() for p in tr.params]
    for regime in q.regimes:
        for p, t0 in zip(tr.params, theta0):
            p.data.copy_(t0)
        opt = torch.optim.AdamW(tr.params, lr=a.lr, weight_decay=0.01)
        D, log, t_start = {}, [], time.time()
        for t in range(q.K):
            tot_bce = tot_ep = 0.0
            for j in draws[t]:
                ids = [int(i) for i in groups[j]]
                cxy, ctg, sctx, _ = pool.micro(ids)
                sx0 = tr.sx0(ids, cxy, sctx)
                sxB = tr.forward(ids, cxy, sctx)
                lam_pg = None if regime == "T" else float(q.lam)
                lA, lB, ln, ep = tr.fn["train_losses"](sx0, sxB.detach(), sxB, ctg, ids, t, lam_pg=lam_pg)
                ((lA + lB) / a.accum).backward()
                tot_bce += float(lB - ep) / a.accum
                tot_ep += float(ep) / a.accum
            n = float(torch.nn.utils.clip_grad_norm_(tr.params, 1.0))
            lr = AC.lr_at(t, a) if regime == "T" else a.lr
            for gr in opt.param_groups:
                gr["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            log.append({"t": t, "grad_norm_pre_clip": n, "bce": tot_bce, "epo": tot_ep, "lr": lr})
            if (t + 1) in (25, q.K):
                D[t + 1] = [(p.detach() - t0).clone() for p, t0 in zip(tr.params, theta0)]
            if t % 5 == 0:
                print(f"lam {q.lam} regime {regime} step {t} gn {n:.3f} bce {tot_bce:.4f} epo {tot_ep:.4f} {time.time()-t_start:.0f}s", flush=True)
        torch.save({"D": D, "log": log, "names": tr.names, "lam": q.lam, "regime": regime, "K": q.K}, os.path.join(q.out_dir, f"closed_{regime}_{q.lam}{q.tag}.pt"))
        print(f"ZZCLOSED_DONE lam {q.lam} regime {regime} {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
