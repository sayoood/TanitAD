#!/usr/bin/env python3
"""H-ADAM controls that need the trainer's closures (PREREG_H_ADAM.md section 4): C1, C5, C6, C7, C8.   python controls_trainer.py --out controls_trainer.json"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adam_common as AC  # noqa: E402


def maxabs(xs, ys):
    return float(max((x - y).abs().max() for x, y in zip(xs, ys)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    q = ap.parse_args()
    SF = AC.load_trainer()
    a, _ = AC.r0_args(SF)
    holder, cfg = AC.build_scorer(SF, a)
    pool = AC.Pool(AC.prepare_pool(SF, a, cfg, "train", max_n=16)).load_visual_all()
    tr = AC.Trainer(SF, a, holder, pool)
    ids = list(range(8))
    cxy, ctg, sctx, _ = pool.micro(ids)
    res = {"tool": "controls_trainer/1", "C6": "PASS (strict load in build_scorer; reached here)", "n_params": int(sum(p.numel() for p in tr.params))}
    sx0 = tr.sx0(ids, cxy, sctx)

    # ---- C1: lam = 0 through the trainer's own train_losses == the BCE-only arm A, gradients AND one AdamW step, bit for bit
    sxA = tr.forward(ids, cxy, sctx, arm=tr.armA)
    sxB = tr.forward(ids, cxy, sctx)
    out = {}
    for tag, lam_pg in (("explicit_lam0_graph_live", 0.0), ("scheduled_upd0_short_circuit", None)):
        for p in tr.armA.parameters():
            p.grad = None
        for p in tr.armB.parameters():
            p.grad = None
        sxA = tr.forward(ids, cxy, sctx, arm=tr.armA)
        sxB = tr.forward(ids, cxy, sctx)
        lA, lB, ln, ep = tr.fn["train_losses"](sx0, sxA, sxB, ctg, ids, 0, lam_pg=lam_pg)
        ((lA + lB) / a.accum).backward()
        gA = [p.grad.clone() for p in tr.armA.parameters()]
        gB = [p.grad.clone() for p in tr.armB.parameters()]
        # one AdamW step on copies
        pa = [torch.nn.Parameter(p.detach().clone()) for p in tr.armA.parameters()]
        pb = [torch.nn.Parameter(p.detach().clone()) for p in tr.armB.parameters()]
        for P, G in ((pa, gA), (pb, gB)):
            for p, g in zip(P, G):
                p.grad = g.clone()
            torch.nn.utils.clip_grad_norm_(P, 1.0)
            o = torch.optim.AdamW(P, lr=a.lr, weight_decay=0.01)
            for gr in o.param_groups:
                gr["lr"] = AC.lr_at(0, a)
            o.step()
        out[tag] = {"loss_A": float(lA), "loss_B": float(lB), "epo_term": float(ep), "max_abs_grad_diff": maxabs(gA, gB), "max_abs_update_diff": maxabs(pa, pb),
                    "bit_identical": bool(all(torch.equal(x, y) for x, y in zip(gA, gB)) and all(torch.equal(x, y) for x, y in zip(pa, pb)))}
    res["C1_lam0_equals_BCE_only"] = out

    # ---- C5: the extracted closures vs the module-level functions and a hand-written BCE (gradients, one micro-batch)
    gBc, gEc, info = tr.micro_grads(ids, cxy, ctg, sctx)
    sxB = tr.forward(ids, cxy, sctx)
    rcfg = SF.rl_reward.RewardCfg(lam_d=a.epo_lam_d, w_prog=a.epo_w_prog, guard=True, lam_k=a.epo_lam_k, use_k=a.epo_k, lam_l=a.epo_lam_l, use_l=a.epo_l)
    recs = [pool.terms[pool.keys[i]] for i in ids]
    R = torch.as_tensor(SF.epo_reward(ctg[..., :6].numpy(), recs, rcfg), dtype=torch.float32)
    sel = lambda p: SF.agg_v1(p[..., :6]) * p[..., 6]  # noqa: E731
    ep_h = SF.epo_loss(sxB, sx0, R, sel, a.epo_temp, a.epo_beta, 1.0)
    m = ctg[..., 6] >= 0
    bce_h = F.binary_cross_entropy_with_logits(sxB[..., :6], ctg[..., :6]) + a.lane_weight * F.binary_cross_entropy_with_logits(sxB[..., 6][m], ctg[..., 6][m])
    gBh = torch.autograd.grad(bce_h / a.accum, tr.params, retain_graph=True)
    gEh = torch.autograd.grad(ep_h / a.accum, tr.params)
    rel = lambda xs, ys: float(torch.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(xs, ys)) / sum((y ** 2).sum() for y in ys)))  # noqa: E731
    res["C5_closures_vs_module_level"] = {"rel_diff_gB": rel(gBc, gBh), "rel_diff_gE": rel(gEc, gEh), "bce_closure": info["bce"], "bce_hand": float(bce_h), "epo_closure": info["epo_unit"], "epo_hand": float(ep_h),
                                         "pass_le_1e-5": bool(rel(gBc, gBh) <= 1e-5 and rel(gEc, gEh) <= 1e-5), "closure_ast_hashes": tr.closure_hashes}

    # ---- C7: the dose is exact at the logit level BEFORE Adam: d epo_loss / d logits at lam_pg 0.25 == 0.25 x at 1.0
    g = {}
    for lam in (0.25, 1.0):
        s = tr.forward(ids, cxy, sctx)
        e = SF.epo_loss(s, sx0, R, sel, a.epo_temp, a.epo_beta, lam)
        g[lam] = torch.autograd.grad(e, s)[0]
    res["C7_logit_level_dose_exact"] = {"max_abs(g025 - 0.25*g1)": float((g[0.25] - 0.25 * g[1.0]).abs().max()), "max_abs_g1": float(g[1.0].abs().max()), "exact": bool(torch.equal(g[0.25], 0.25 * g[1.0]))}

    # ---- C8: fp32 vs CPU bf16 autocast, per-tensor r_p = |gE|/|gB| on this micro-batch
    sxb = tr.forward(ids, cxy, sctx, autocast=True)
    lA, lB, ln, ep = tr.fn["train_losses"](sx0, sxb.detach(), sxb, ctg, ids, 0, lam_pg=1.0)
    gBa = torch.autograd.grad((lB - ep) / a.accum, tr.params, retain_graph=True)
    gEa = torch.autograd.grad(ep / a.accum, tr.params)
    dev = []
    for n, b32, e32, b16, e16 in zip(tr.names, gBc, gEc, gBa, gEa):
        r32 = float(e32.norm() / (b32.norm() + 1e-30))
        r16 = float(e16.norm() / (b16.norm() + 1e-30))
        dev.append((n, r32, r16, abs(r16 / r32 - 1.0) if r32 > 0 else None))
    worst = sorted([d for d in dev if d[3] is not None], key=lambda d: -d[3])[:5]
    big = [d for d in dev if d[3] is not None]
    res["C8_fp32_vs_bf16_autocast"] = {"median_rel_dev_of_r_p": float(np.median([d[3] for d in big])), "max_rel_dev_of_r_p": float(max(d[3] for d in big)),
                                       "share_tensors_within_15pct": float(np.mean([d[3] <= 0.15 for d in big])), "worst5": [list(d) for d in worst]}
    json.dump(res, open(q.out, "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "worst5"}) for k, v in res.items()}, indent=1)[:3500])
    print("ZZCTRL_OK")


if __name__ == "__main__":
    main()
