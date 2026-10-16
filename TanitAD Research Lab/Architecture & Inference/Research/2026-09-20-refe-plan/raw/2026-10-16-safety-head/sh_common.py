#!/usr/bin/env python3
"""SAFEHEAD instrument (PREREG_SAFEHEAD.md): shared machinery.  CPU ONLY.

REUSES the PHEAD harness unchanged (raw/2026-10-16-phead-epo-feasibility/phead_common.py), which reuses the H-ADAM instrument (raw/2026-10-15-rls1-adam-dose/adam_common.py):
the trainer's own closures extracted verbatim from scorer_finetune.py's AST (losses, epo_term, train_losses, sel_score), the real OnPolicyBank pools, the gradient bank's draws,
torch.optim.AdamW + clip_grad_norm_ through phead_common.Restricted.  What is NEW here:

  * `combine`      -- the safety-head selection logits: S on the SAFE columns (NC 0, DAC 1, TTC 3), arm B elsewhere.  THE SEPARATION LINE detaches S for the EPO input.
  * `sh_losses`    -- train_losses with ONE change: the EPO term reads `combine(...)` instead of sxB.  With sep off it is the trainer's assembly exactly (control a').
  * `run_closed`   -- the closed-loop step of phead_closed.py (kind R, all tensors), plus arm A co-trained by its own BCE for SH_A (the trainer's optA, :1359-1361).
  * `eval_views`   -- J and the pick / policy proxies of several SELECTION VIEWS of one state (the phead Evaluator's arithmetic, per view).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHD = os.path.normpath(os.path.join(HERE, "..", "2026-10-16-phead-epo-feasibility"))
sys.path.insert(0, PHD)
import phead_common as PC  # noqa: E402
import adam_common as AC  # noqa: E402
from adam_sim import draws_for  # noqa: E402

SAFE = (0, 1, 3)                     # NC, DAC, TTC = rl_reward.safe_mask's three components (rl_reward.py:87-89)
DESIGNS = ("FULL", "SH_F", "SH_A", "SH_SG")
SH_SCR = f"{PC.SCR}/safehead"
os.makedirs(SH_SCR, exist_ok=True)


def safe_mask_cols(n, device=None):
    m = torch.zeros(n, dtype=torch.bool, device=device)
    m[list(SAFE)] = True
    return m


def combine(design, sxB, sxS, sep=True):
    """The selection logits of a design.  FULL (or sep off) = sxB itself.  Otherwise comb[..., c] = S[..., c] for c in SAFE, sxB[..., c] else.
    torch.where routes the gradient of a SAFE column to S only, so sxB's SAFE columns get exactly 0 from anything computed on comb."""
    if design == "FULL" or not sep:
        return sxB
    if os.environ.get("SAFEHEAD_MUTATE_NO_SEP") == "1":
        S_epo = sxS                                   # <- MUTATION (control c): the separation line is deleted
    else:
        S_epo = sxS.detach()                          # <- THE SEPARATION LINE: EPO's gradient can never reach the safety source
    return torch.where(safe_mask_cols(sxB.shape[-1], sxB.device), S_epo, sxB)


def safety_source(design, sx0, sxA, sxB):
    return {"SH_F": sx0, "SH_A": sxA, "SH_SG": sxB}.get(design)


def sh_losses(fn, a, design, sep, sx0, sxA, sxB, tg, rows, upd, lam_pg=None):
    """= the trainer's train_losses (scorer_finetune.py:792-800 at d06546a with --epo-ref base; :792-799 at af1e2a8) with ONE change: epo_term reads combine(...)."""
    lA, lB, ln = fn["losses"](sxA, sxB, tg, a.lam)
    ep = torch.zeros((), device=sxB.device)
    if a.epo:
        comb = combine(design, sxB, safety_source(design, sx0, sxA, sxB), sep)
        ep = fn["epo_term"](sx0, comb, tg, rows, upd, lam_pg)
        lB = lB + ep
    return lA, lB, ln, ep


# --------------------------------------------------------------------------------------------------------------------------- views
VIEW_FN = {
    "own":       lambda sxB, sxA, sx0: sxB,                                             # arm B's own selection (FULL; SH_SG at eval)
    "SH_F":      lambda sxB, sxA, sx0: torch.where(safe_mask_cols(sxB.shape[-1]), sx0, sxB),
    "SH_A":      lambda sxB, sxA, sx0: torch.where(safe_mask_cols(sxB.shape[-1]), sxA, sxB),
    "A_own":     lambda sxB, sxA, sx0: sxA,                                             # arm A = FULL lam 0 = the matched control of FULL, SH_A, SH_SG
    "SH_F_ctrl": lambda sxB, sxA, sx0: torch.where(safe_mask_cols(sxA.shape[-1]), sx0, sxA),   # SH_F's matched control: A's progress rows + base safety rows
}


def _set(module, theta):
    with torch.no_grad():
        for p, t in zip(module.parameters(), theta):
            p.copy_(t)


@torch.no_grad()
def eval_views(ev, tr, thB, thA, views):
    """-> {view: per-set arrays} on ONE pool (the phead Evaluator.evaluate arithmetic, verbatim, applied to each view's logits)."""
    SF, a, sel = ev.SF, ev.a, ev.sel
    _set(tr.armB, thB)
    if thA is not None:
        _set(tr.armA, thA)
    if ev.sx0 is None:
        ev.sx0 = torch.cat([ev._forward(tr.base, ids) for ids in ev.batches])
    keys = ("J", "kl", "pi_comp", "pick_comp", "pick_R", "pick_pdms", "pick_idx", "pi_pdms")
    out = {v: {k: [] for k in keys} for v in views}
    for ids in ev.batches:
        cxy, _ctg, sctx, _ = ev.pool.micro(ids)
        sxB = tr.armB(cxy, sctx).float()
        sxA = tr.armA(cxy, sctx).float() if thA is not None else None
        sx0 = ev.sx0[ids]
        logpi0 = SF.epo_policy(sx0, sel, a.epo_temp)
        R, tg6 = ev.R[ids], ev.tg6[ids]
        pdms_all = SF.agg_v1(tg6)
        ar = torch.arange(len(ids))
        for v in views:
            sx = VIEW_FN[v](sxB, sxA, sx0)
            logpi = SF.epo_policy(sx, sel, a.epo_temp)
            pi = logpi.exp()
            pick = sel(torch.sigmoid(sx)).argmax(1)
            o = out[v]
            o["J"].append((pi * R).sum(1))
            o["kl"].append((logpi0.exp() * (logpi0 - logpi)).sum(1))
            o["pi_comp"].append((pi[..., None] * tg6).sum(1))
            o["pick_comp"].append(tg6[ar, pick])
            o["pick_R"].append(R[ar, pick])
            o["pick_pdms"].append(pdms_all[ar, pick])
            o["pick_idx"].append(pick.float())
            o["pi_pdms"].append((pi * pdms_all).sum(1))
    return {v: {k: torch.cat(x).numpy().astype(np.float64) for k, x in o.items()} for v, o in out.items()}


def views_for(design, co_train_A):
    if co_train_A:
        return ["own", "SH_A", "A_own", "SH_F_ctrl"] + (["SH_F"] if design != "SH_A" else [])
    return {"FULL": ["own"], "SH_SG": ["own"], "SH_F": ["SH_F", "own"]}[design]


# --------------------------------------------------------------------------------------------------------------------------- the closed loop
def run_closed(SF, a, tr, ev, design, lam, regime, K, snaps, stream=0, sep=True, co_train_A=None, lr_mult=1.0, keep_theta=False, log_every=5, name=""):
    """K closed-loop updates of `design` (phead_closed.py's step, kind R, every tensor trainable).  co_train_A (default: design == SH_A) also trains arm A by ITS OWN loss lA
    in the same backward, as the trainer does ((lA + lB)/accum, :1381), with its own clip + AdamW (:1390-1395).  Returns (log, snaps, thetaB, thetaA)."""
    assert design in DESIGNS
    co = (design == "SH_A") if co_train_A is None else co_train_A
    a.epo_lam = float(lam)
    fn = tr.fn
    with torch.no_grad():                                   # every run starts from theta_0 (the base) for both arms
        for p, q in zip(tr.armB.parameters(), tr.base.parameters()):
            p.copy_(q)
        for p, q in zip(tr.armA.parameters(), tr.base.parameters()):
            p.copy_(q)
    thB0 = [p.detach().clone() for p in tr.armB.parameters()]
    thA0 = [p.detach().clone() for p in tr.armA.parameters()]
    allB = [(i, None) for i in range(len(thB0))]
    RB = PC.Restricted(thB0, allB, lr=a.lr * lr_mult)
    RA = PC.Restricted(thA0, allB, lr=a.lr * lr_mult) if co else None
    for p in tr.armB.parameters():
        p.requires_grad_(True)
    for p in tr.armA.parameters():
        p.requires_grad_(bool(co))
    bank = torch.load(os.path.join(PC.BANK_DIR, f"bank_s{stream}.pt"), weights_only=False, mmap=True)
    groups = bank["groups"]
    draws = draws_for(stream, K, groups.shape[0])
    pB, pA = list(tr.armB.parameters()), list(tr.armA.parameters())
    log, out_snaps, t0 = [], {}, time.time()
    thA = thA0
    for t in range(K):
        gB = gA = None
        tot = {"bceB": 0.0, "bceA": 0.0, "epo": 0.0, "J": 0.0}
        for j in draws[t]:
            ids = [int(i) for i in groups[j]]
            cxy, ctg, sctx, _ = tr.pool.micro(ids)
            sx0 = tr.sx0(ids, cxy, sctx)
            sxB = tr.forward(ids, cxy, sctx)
            sxA = tr.armA(cxy, sctx) if co else sxB.detach()          # phead_closed passes sxB.detach() as arm A when A is not trained
            lam_pg = None if regime == "T" else float(lam)
            if design == "FULL":
                lA, lB, ln, ep = fn["train_losses"](sx0, sxA, sxB, ctg, ids, t, lam_pg=lam_pg)
            else:
                lA, lB, ln, ep = sh_losses(fn, a, design, sep, sx0, sxA, sxB, ctg, ids, t, lam_pg=lam_pg)
            loss = ((lA + lB) if co else lB) / a.accum
            loss.backward()
            g = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in pB]
            gB = g if gB is None else [x + y for x, y in zip(gB, g)]
            for p in pB:
                p.grad = None
            if co:
                g = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in pA]
                gA = g if gA is None else [x + y for x, y in zip(gA, g)]
                for p in pA:
                    p.grad = None
            tot["bceB"] += float(lB - ep) / a.accum
            tot["bceA"] += float(lA) / a.accum
            tot["epo"] += float(ep) / a.accum
            tot["J"] += float(tr.epo_last.get("J", 0.0)) / a.accum
        lr = (AC.lr_at(t, a) if regime == "T" else a.lr) * lr_mult
        nB = RB.step(gB, lr)
        thB = RB.theta()
        _set(tr.armB, thB)
        nA = None
        if co:
            nA = RA.step(gA, lr)
            thA = RA.theta()
            _set(tr.armA, thA)
        log.append({"t": t, "gnB": nB, "gnA": nA, **tot, "lr": lr})
        if log_every and t % log_every == 0:
            print(f"{name} step {t} gnB {nB:.3f} bceB {tot['bceB']:.4f} epo {tot['epo']:.4f} J {tot['J']:.4f} {time.time()-t0:.0f}s", flush=True)
        if (t + 1) in snaps:
            vws = views_for(design, co)
            s = {"eval": {pn: eval_views(e, tr, thB, thA if co else None, vws) for pn, e in ev.items()},
                 "disp_B": float(torch.cat([(x - y).reshape(-1) for x, y in zip(thB, thB0)]).norm()),
                 "disp_A": float(torch.cat([(x - y).reshape(-1) for x, y in zip(thA, thA0)]).norm()) if co else None}
            if keep_theta:
                s["thetaB"] = [x.clone() for x in thB]
                s["thetaA"] = [x.clone() for x in thA] if co else None
            out_snaps[t + 1] = s
            _set(tr.armB, thB)
            if co:
                _set(tr.armA, thA)
    return log, out_snaps, thB, (thA if co else None)


def setup(threads):
    return PC.setup(threads)
