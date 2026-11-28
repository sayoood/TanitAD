#!/usr/bin/env python3
"""SCORER-FT: reachability + analytic proofs for refe/rank_loss.py and its wiring in refe/train.py (DESIGN_SCORER_FT.md 2.2, 2.4, 3.4; CLAUDE.md "built, tested and unreachable").

    python test_rank_loss.py --fast        # unit + tiny-trainer checks (self-made synthetic bank, ViT-S, 1 camera, CPU); no mutations. G-SUITE runs this.
    python test_rank_loss.py               # --fast, then every mutation: (a) a LINE of the source is edited in a temp copy of refe/*.py, (b) a REFE_RANK_MUTATE arm is set;
                                           # one named check is run there and MUST exit != 0
    python test_rank_loss.py --check NAME  # one check (the mutation harness's entry point)
    python test_rank_loss.py --pristine DIR   # + the OFF-path identity against an unpatched refe dir DIR (the pre-change trainer): metrics rows and final weights bit-identical
Prints `ZZRANK_PASS <n>`.  A skip is not a pass: nothing here skips.  The first line of defence is a check that CAN go red: every check below has a mutation that turns it red.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rank_loss as RK  # noqa: E402
import test_p6_levers as P6  # noqa: E402  (the self-made bank, the tiny-trainer argv, the temp-dir registry that cannot leak)

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")           # float(L) of a graph-carrying loss is deliberate here

N = 0


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


# ------------------------------------------------------------------------------------------------ an independent numpy reference (no torch, loops)
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logsig(x):
    return -math.log1p(math.exp(-x)) if x > -30 else x


def _softplus(x):
    return math.log1p(math.exp(x)) if x < 30 else x


def _v1(p):
    return p[0] * p[1] * (5.0 * p[2] + 5.0 * p[3] + 2.0 * p[4]) / 12.0


def ref_term(s, y, msk, batch, heads, gap, k):
    """The spec 2.2 definition, written as loops.  Returns (L, {head: set of (b, i, j)})"""
    s, y = np.asarray(s, np.float64), np.asarray(y, np.float64)
    B, M, _ = s.shape
    pairs = {h: set() for h in heads}
    tot = 0.0
    for b in range(B):
        safe = [(y[b, m, 0] > 0.5) and (y[b, m, 1] > 0.5) for m in range(M)]
        pdms = [_v1(y[b, m]) for m in range(M)]
        agg = [_v1([_sig(v) for v in s[b, m, :5]]) for m in range(M)]
        top = set(np.argsort(-np.array(agg), kind="stable")[:k].tolist())
        pick = int(np.argmax(np.array(agg)))                                                       # the first maximal index (the planner's argmax)
        for h in heads:
            num = den = 0.0
            for i in range(M):
                for j in range(M):
                    if h == "ep" and safe[i] and safe[j] and y[b, i, 2] - y[b, j, 2] >= gap:
                        a, z = 5.0 / 12.0 * (y[b, i, 2] - y[b, j, 2]), s[b, i, 2] - s[b, j, 2]
                    elif h == "gate" and safe[i] and not safe[j] and j in top and pdms[i] > pdms[j]:
                        a = pdms[i] - pdms[j]
                        z = (_logsig(s[b, i, 0]) + _logsig(s[b, i, 1])) - (_logsig(s[b, j, 0]) + _logsig(s[b, j, 1]))
                    elif h == "ttc" and safe[i] and safe[j] and y[b, i, 3] == 1.0 and y[b, j, 3] == 0.0:
                        a, z = 5.0 / 12.0, s[b, i, 3] - s[b, j, 3]
                    elif h == "pick" and j == pick and pdms[i] > pdms[j]:
                        a, z = pdms[i] - pdms[j], agg[i] - agg[j]
                    else:
                        continue
                    pairs[h].add((b, i, j))
                    num += a * _softplus(-z)
                    den += a
            if den > 0 and msk[b] > 0:
                tot += msk[b] * num / den
    return tot / batch, pairs


def rand_case(seed, B=2, M=64, dtype=torch.float64):
    g = np.random.default_rng(seed)
    s = g.normal(0, 2.0, (B, M, 6))
    y = np.zeros((B, M, 6))
    y[..., 0] = (g.random((B, M)) > 0.35)
    y[..., 1] = (g.random((B, M)) > 0.35)
    y[..., 2] = np.where(g.random((B, M)) > 0.7, 1.0, np.round(g.random((B, M)), 3))
    y[..., 3] = (g.random((B, M)) > 0.4)
    y[..., 4] = (g.random((B, M)) > 0.3)
    y[..., 5] = (g.random((B, M)) > 0.2)
    y[..., 0] = np.where(g.random((B, M)) > 0.9, 0.5, y[..., 0])                           # NC = 0.5 (the half-penalty) is NOT safe: the boundary of `> 0.5`
    msk = np.ones(B)
    return torch.tensor(s, dtype=dtype, requires_grad=True), torch.tensor(y, dtype=dtype), torch.tensor(msk, dtype=dtype), s, y, msk


# ------------------------------------------------------------------------------------------------ unit checks
def chk_analytic():
    """A hand-computed case.  Set 0 (labelled): 4 proposals; set 1 (UNlabelled, garbage labels): contributes exactly 0; denominator = batch = 2.
       p0 safe EP .9 | p1 safe EP .5 | p2 safe EP .49 | p3 unsafe (NC 0).   EP logits z = [0, 1, -1, *].  gap 0.02.
       EP pairs: (0,1) d .40, (0,2) d .41; (1,2) d .01 < gap EXCLUDED.  L_ep = (.40 sp(1) + .41 sp(-1)) / .81 with sp(x) = log(1+e^x)."""
    sp = lambda x: math.log1p(math.exp(x))                                                          # noqa: E731
    hand_ep = (0.40 * sp(1.0) + 0.41 * sp(-1.0)) / 0.81
    ok(abs(hand_ep - 0.8070888) < 1e-6, hand_ep)                                                    # the literal, computed by hand: .40*1.31326 + .41*.31326 = .653742 ; / .81 = .8070888
    M = 4
    s = torch.zeros(2, M, 6, dtype=torch.float64)
    s[0, :, 2] = torch.tensor([0.0, 1.0, -1.0, 0.0])
    y = torch.zeros(2, M, 6, dtype=torch.float64)
    y[0, :3, 0] = 1.0
    y[0, :3, 1] = 1.0
    y[0, 3, 0], y[0, 3, 1] = 0.0, 1.0
    y[0, :, 2] = torch.tensor([0.9, 0.5, 0.49, 0.3], dtype=torch.float64)
    y[1, :, 0], y[1, :, 1] = 1.0, 1.0
    y[1, :, 2] = torch.tensor([0.9, 0.1, 0.5, 0.2], dtype=torch.float64)                             # garbage WITH pairs on the unlabelled set (logits 0 => loss ln 2): must contribute exactly 0
    msk = torch.tensor([1.0, 0.0], dtype=torch.float64)
    L = RK.RankLoss(("ep",), ep_gap=0.02, gate_topk=M).term(score=s, labels=y, msk=msk, batch=2)
    ok(abs(float(L) - hand_ep / 2.0) < 1e-12, (float(L), hand_ep / 2.0))
    # the gate: z = logsig(sNC) + logsig(sDAC).  safe i in {0,1,2} vs unsafe j = 3 (pdms 0): weight pdms_i = (5 EP_i + 5 TTC + 2 C) / 12 (TTC = C = 0 here => 5 EP_i / 12)
    s2 = torch.zeros(2, M, 6, dtype=torch.float64)
    s2[0, :, 0] = torch.tensor([2.0, 0.0, -1.0, 0.5])
    s2[0, :, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    zz = [(-sp(-s2[0, m, 0].item())) + (-sp(-s2[0, m, 1].item())) for m in range(M)]               # logsigmoid(x) = -softplus(-x)
    w = [5.0 * y[0, m, 2].item() / 12.0 for m in range(3)]
    hand_gate = sum(w[i] * sp(-(zz[i] - zz[3])) for i in range(3)) / sum(w)
    Lg = RK.RankLoss(("gate",), gate_topk=M).term(score=s2, labels=y, msk=msk, batch=2)
    ok(abs(float(Lg) - hand_gate / 2.0) < 1e-12, (float(Lg), hand_gate / 2.0))
    # a set with NO pair of a head contributes exactly 0.0 (and is counted), not NaN
    yy = y.clone()
    yy[0, :, 2] = 0.5
    r = RK.RankLoss(("ep",), ep_gap=0.02, gate_topk=M)
    Lz = r.term(score=s.clone().requires_grad_(True), labels=yy, msk=msk, batch=2)
    ok(float(Lz) == 0.0 and r.last["pairs"]["ep"] == 0 and r.last["sets"]["ep"] == 0 and r.last["sets_labelled"] == 1, (float(Lz), r.last))
    # the unlabelled set's pairs are not COUNTED either (counts are over labelled sets)
    r2 = RK.RankLoss(("ep",), ep_gap=0.02, gate_topk=M)
    r2.term(score=s, labels=y, msk=msk, batch=2)
    ok(r2.last["pairs"]["ep"] == 2 and r2.last["sets"]["ep"] == 1, r2.last)


def chk_reference_audit():
    """L4 of the gate, here: the pair SETS and the value against the loop reference, for every head, on random sets (incl. NC = 0.5, EP ties, TTC)."""
    for seed in range(3):
        for heads in (("ep",), ("gate",), ("ttc",), ("ep", "gate"), ("ep", "gate", "ttc")):
            st, yt, mt, s, y, m = rand_case(seed)
            r = RK.RankLoss(heads, ep_gap=0.02, gate_topk=8)
            L = r.term(score=st, labels=yt, msk=mt, batch=2)
            Lr, prs = ref_term(s, y, m, 2, heads, 0.02, 8)
            P = r.pairs(st, yt)
            for h in heads:
                mine = {tuple(int(v) for v in idx) for idx in P[h][0].nonzero().tolist()}
                ok(mine == prs[h] and len(mine) > 0, (seed, h, len(mine), len(prs[h])))
            ok(abs(float(L) - Lr) <= 1e-12 * max(1.0, abs(Lr)), (seed, heads, float(L), Lr))


def chk_analytic_identities():
    """topk = M reproduces the unrestricted gate (a bar below everything); a unit-level gap of 1.5 gives zero EP pairs, L exactly 0.0 and gradient exactly 0.0
       (a bar above everything); gap 0.02 is partial; topk < M is partial and strictly fewer gate pairs."""
    st, yt, mt, s, y, m = rand_case(11)
    full = RK.RankLoss(("gate",), gate_topk=64)
    L = full.term(score=st, labels=yt, msk=mt, batch=2)
    Lr, prs = ref_term(s, y, m, 2, ("gate",), 0.02, 64)
    ok(abs(float(L) - Lr) <= 1e-12 and full.last["pairs"]["gate"] == len(prs["gate"]) > 0, (float(L), Lr))
    # unrestricted by construction: every (safe i, unsafe j, pdms_i > pdms_j)
    safe = (y[..., 0] > 0.5) & (y[..., 1] > 0.5)
    pd = np.array([[_v1(y[b, k]) for k in range(64)] for b in range(2)])
    n_unres = sum(1 for b in range(2) for i in range(64) for j in range(64) if safe[b, i] and not safe[b, j] and pd[b, i] > pd[b, j])
    ok(full.last["pairs"]["gate"] == n_unres, (full.last, n_unres))
    part = RK.RankLoss(("gate",), gate_topk=1)
    part.term(score=st, labels=yt, msk=mt, batch=2)
    ok(0 < part.last["pairs"]["gate"] < n_unres, (part.last, n_unres))
    # gap above everything
    hi = RK.RankLoss(("ep",), ep_gap=0.02)
    hi.ep_gap = 1.5                                                                                  # unit level: the constructor (0, 1] would refuse it
    st.grad = None
    Lh = hi.term(score=st, labels=yt, msk=mt, batch=2)
    Lh.backward()
    ok(float(Lh) == 0.0 and hi.last["pairs"]["ep"] == 0 and float(st.grad.abs().sum()) == 0.0, (float(Lh), hi.last))
    lo = RK.RankLoss(("ep",), ep_gap=0.02)
    st.grad = None
    Ll = lo.term(score=st, labels=yt, msk=mt, batch=2)
    Ll.backward()
    n_all_ep = sum(1 for b in range(2) for i in range(64) for j in range(64) if safe[b, i] and safe[b, j] and y[b, i, 2] > y[b, j, 2])
    ok(float(Ll) > 0.0 and float(st.grad.abs().sum()) > 0.0 and 0 < lo.last["pairs"]["ep"] < n_all_ep, (lo.last, n_all_ep))


def chk_pick_identity():
    """O3: the loss's own pick IS the inference-time pick, bit for bit.  (1) `RK.navsim_v1(sigmoid(s))` is EXACTLY `REFePlanner.aggregate(rule navsim_v1)` (the planner's own method, called through the same shim
       onpolicy_label_v4.v1_aggregate uses) on float32 logits, and `RK.pick_index` == the planner's argmax (`_pick`: aggregate(score)[0].argmax()); (2) the pair mask of head `pick` is the column of THAT pick on
       every set; (3) a constructed near-tie (two proposals whose float32 aggregates are EQUAL while their float64 aggregates differ) goes to the FIRST index exactly as the float32 planner does, and a float64 argmax
       would have gone to the other one (the check CAN go red)."""
    import types
    from planner import REFePlanner
    shim = types.SimpleNamespace(rule="navsim_v1", V1_W=REFePlanner.V1_W, PDM_W=REFePlanner.PDM_W)
    g = np.random.default_rng(77)
    for seed in range(6):
        sc = torch.tensor(g.normal(0, 2.0, (3, 64, 6)), dtype=torch.float32)
        sc[:, 7] = sc[:, 3]                                                                          # an EXACT duplicate: a tie the planner resolves to the first index
        a_pl = REFePlanner.aggregate(shim, sc)
        a_rk = RK.navsim_v1(torch.sigmoid(sc))
        ok(torch.equal(a_pl, a_rk), "RK.navsim_v1 must equal REFePlanner.aggregate bit for bit")
        pk_pl = a_pl.argmax(1)
        ok(torch.equal(RK.pick_index(sc), pk_pl), (seed, RK.pick_index(sc), pk_pl))
        y = torch.tensor(g.random((3, 64, 6)), dtype=torch.float32).round(decimals=2)
        P = RK.RankLoss(("pick",)).pairs(sc, y)
        m = P["pick"][0]                                                                             # [B, i, k]
        for b in range(3):
            cols = m[b].any(0).nonzero().flatten().tolist()
            ok(cols in ([int(pk_pl[b])], []), (seed, b, cols, int(pk_pl[b])))                          # pairs only on the planner's pick column (none if the pick is the oracle)
    # (3) the near-tie: logit EP of proposal 5 is 1e-7 above proposal 2's, all else equal
    found = 0
    for base in (0.31, -0.7, 1.9, 0.05):
        sc = torch.zeros(1, 8, 6, dtype=torch.float32)
        sc[0, :, EP_] = -3.0
        sc[0, 2, EP_] = base
        sc[0, 5, EP_] = float(np.float32(base) + np.float32(1e-7))
        a32 = RK.navsim_v1(torch.sigmoid(sc))
        a64 = RK.navsim_v1(torch.sigmoid(sc.double()))
        if int(a64.argmax(1)) != int(a32.argmax(1)):
            found += 1
            ok(int(RK.pick_index(sc)) == int(REFePlanner.aggregate(shim, sc).argmax(1)) == int(a32.argmax(1)) == 2 and int(a64.argmax(1)) == 5, (base, int(a32.argmax(1)), int(a64.argmax(1))))
            y = torch.zeros(1, 8, 6, dtype=torch.float32)
            y[0, :, 0], y[0, :, 1], y[0, 6, 2] = 1.0, 1.0, 1.0
            P = RK.RankLoss(("pick",)).pairs(sc, y)
            ok(sorted(P["pick"][0][0].any(0).nonzero().flatten().tolist()) == [2], "the pairs anchor on the FLOAT32 pick (index 2), not the float64 one (5)")
    ok(found >= 1, "no constructed near-tie separated float32 from float64: the sensitivity of this check is unproved")


EP_ = 2


def chk_pick_analytic():
    """Hand-computed.  Set 0 (labelled), 4 proposals, ALL NC = DAC = 1 logits 0 and TTC = C logits 0 (p = .5), EP logits z = [0, 2, -1, 1]:  aggregate_m = .25 (5 sigma(z_m) + 3.5) / 12  =>  pick = 1.
       labels pdms: p0 (EP .9, TTC 1, C 1) 11.5/12 | p1 (EP .1, TTC 1, C 1) 7.5/12 | p2 (EP 1, TTC 1, C 1) 12/12 | p3 unsafe 0.   Pairs: (0 vs pick) d 4/12, (2 vs pick) d 4.5/12; p3 is NOT above the pick.
       L(b) = (4 softplus(agg1 - agg0) + 4.5 softplus(agg1 - agg2)) / 8.5 ; set 1 is UNlabelled (garbage labels with pairs) and contributes exactly 0; denominator = batch = 2.  The CONSTANT denominator: the labelled set's
       value is halved, not renormalised over the sets that have pairs."""
    sp = lambda x: math.log1p(math.exp(x))                                                          # noqa: E731
    sg = lambda x: 1.0 / (1.0 + math.exp(-x))                                                       # noqa: E731
    M = 4
    z = [0.0, 2.0, -1.0, 1.0]
    agg = [0.25 * (5.0 * sg(v) + 3.5) / 12.0 for v in z]
    ok(max(range(M), key=lambda m: agg[m]) == 1, agg)
    hand = (4.0 * sp(agg[1] - agg[0]) + 4.5 * sp(agg[1] - agg[2])) / 8.5
    s = torch.zeros(2, M, 6, dtype=torch.float64)
    s[0, :, 2] = torch.tensor(z, dtype=torch.float64)
    s[1, :, 2] = torch.tensor(z, dtype=torch.float64)
    y = torch.zeros(2, M, 6, dtype=torch.float64)
    y[0, :, 0] = 1.0
    y[0, :, 1] = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    y[0, :, 2] = torch.tensor([0.9, 0.1, 1.0, 0.0], dtype=torch.float64)
    y[0, :, 3] = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    y[0, :, 4] = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    y[1] = y[0]
    y[1, 0, 2], y[1, 2, 2] = 1.0, 0.0                                                                # garbage on the unlabelled set (still has pairs)
    msk = torch.tensor([1.0, 0.0], dtype=torch.float64)
    r = RK.RankLoss(("pick",))
    sv = s.clone().requires_grad_(True)
    L = r.term(score=sv, labels=y, msk=msk, batch=2)
    ok(abs(float(L) - hand / 2.0) < 1e-12, (float(L), hand / 2.0))
    ok(r.last["pairs"]["pick"] == 2 and r.last["sets"]["pick"] == 1 and r.last["sets_labelled"] == 1, r.last)
    ok(RK.RankLoss(("pick",)).pairs(s, y)["pick"][0][0].nonzero().tolist() == [[0, 1], [2, 1]], "pairs are (i better, k = the pick)")
    L.backward()
    gr = sv.grad.abs().sum((0, 1))
    ok(float(gr[EP_]) > 0.0 and float(sv.grad[1].abs().sum()) == 0.0, "the unlabelled set has exactly 0 gradient")
    ok(float(gr[5]) == 0.0, "DDC exactly 0.0")                                                       # the aggregate ignores DDC
    # no pair when the pick is the label-best: L exactly 0.0 and gradient exactly 0.0 (a bar above everything)
    y2 = y.clone()
    y2[0, :, 2] = torch.tensor([0.1, 1.0, 0.5, 0.0], dtype=torch.float64)                            # p1 (the pick) is now the oracle
    sv2 = s.clone().requires_grad_(True)
    r2 = RK.RankLoss(("pick",))
    L2 = r2.term(score=sv2, labels=y2, msk=msk, batch=2)
    L2.backward()
    ok(float(L2) == 0.0 and r2.last["pairs"]["pick"] == 0 and float(sv2.grad.abs().sum()) == 0.0, (float(L2), r2.last))
    # weight 0 of the trainer's contribution: (term * 0) reads EXACTLY 0.0 on every logit
    sv3 = s.clone().requires_grad_(True)
    (RK.RankLoss(("pick",)).term(score=sv3, labels=y, msk=msk, batch=2) * 0.0).backward()
    ok(float(sv3.grad.abs().sum()) == 0.0, "the zero control must read exactly 0.0")


def _pick_term_unrouted(s, y, msk, batch):
    """The pick term with NO routing, written independently of rank_loss.RankLoss.pairs' agg line: same pair mask and weights (label-only), z = the plain aggregate navsim_v1(sigmoid(s)) with every factor live."""
    P = RK.RankLoss(("pick",)).pairs(s, y)
    m, a, _ = P["pick"]
    agg = RK.navsim_v1(torch.sigmoid(s.double()))
    dz = agg.unsqueeze(2) - agg.unsqueeze(1)
    w = a * m.double()
    den = w.sum((1, 2))
    num = (w * torch.nn.functional.softplus(-dz)).sum((1, 2))
    Lh = torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(den))
    return ((Lh * msk.detach().double()).sum() / float(batch)).to(s.dtype), m, dz


def chk_pick_routing():
    """LABEL-CONSISTENT ROUTING of the pick term (MM design decision 10).  (a) the FORWARD is bit-identical to the unrouted form (loss value, pair mask, logit differences) on random float64 AND float32 sets (the trainer's dtype);
       and the routed gradient equals the unrouted gradient EXACTLY where an independent consistency mask (pick: y < p, better proposal: y > p) holds and is EXACTLY 0.0 elsewhere.  Hand-built set (4 proposals, p = .5 on
       NC / DAC / TTC / C, EP logits [0, 5, 1.5, -1] => pick = 1; EP labels [.9, .1, .8, 0], p2 EP y .8 < p .818):
       (b) the pick's NC label is 1 (> p): its NC and DAC gradients from the term are EXACTLY 0.0, as are TTC and C (labels 1 > .5);  (c) with the pick's NC label 0 (< p) the NC gradient is non-zero and LOWERS p_NC (loss increases
       with the logit); DAC stays exactly 0.0;  (d) better j0 (EP label .9 > p .5): the EP gradient RAISES p_EP (loss falls as the logit grows); better j2 (EP label .8 < p .818): EP gradient EXACTLY 0.0 though j2 is better;
       DDC exactly 0.0 throughout."""
    for dtype in (torch.float64, torch.float32):
        for seed in range(4):
            st, yt, mt, s_, y_, m_ = rand_case(seed + 50, dtype=dtype)
            su = st.detach().clone().requires_grad_(True)
            Lr = RK.RankLoss(("pick",)).term(score=st, labels=yt, msk=mt, batch=2)
            Lu, mu, dzu = _pick_term_unrouted(su, yt, mt, 2)
            Pr = RK.RankLoss(("pick",)).pairs(st, yt)["pick"]
            ok(torch.equal(Lr, Lu) and torch.equal(Pr[0], mu) and torch.equal(Pr[2], dzu), (dtype, seed, float(Lr), float(Lu)))       # forward: bit for bit
            Lr.backward()
            Lu.backward()
            with torch.no_grad():
                pk = RK.pick_index(su.detach())
                p = torch.sigmoid(st.detach().double())
                at = torch.zeros(p.shape[:2], dtype=torch.bool)
                at.scatter_(1, pk.unsqueeze(1), True)
                cons = torch.where(at.unsqueeze(2), yt.double() < p, yt.double() > p)
                exp = torch.where(cons, su.grad.double(), torch.zeros_like(su.grad.double()))
            err = float((st.grad.double() - exp).abs().max())
            ok(err <= 1e-12 * max(1e-30, float(su.grad.abs().max())) and bool(((st.grad == 0) | cons).all()), (dtype, seed, err))   # exact zeros where inconsistent
            ok(int((~cons & (su.grad != 0)).sum()) > 0, "the case must actually mask something (else the test proves nothing)")
    M = 4
    z = torch.zeros(1, M, 6, dtype=torch.float64)
    z[0, :, 2] = torch.tensor([0.0, 5.0, 1.5, -1.0], dtype=torch.float64)
    y = torch.zeros(1, M, 6, dtype=torch.float64)
    y[0, :3, 0] = 1.0
    y[0, :3, 1] = 1.0
    y[0, :3, 3] = 1.0
    y[0, :3, 4] = 1.0
    y[0, :, 2] = torch.tensor([0.9, 0.1, 0.8, 0.0], dtype=torch.float64)
    msk = torch.ones(1, dtype=torch.float64)

    def grad_of(yy):
        sv = z.clone().requires_grad_(True)
        r = RK.RankLoss(("pick",))
        r.term(score=sv, labels=yy, msk=msk, batch=1).backward()
        ok(sorted(r.last and RK.RankLoss(("pick",)).pairs(z, yy)["pick"][0][0].any(0).nonzero().flatten().tolist()) == [1], "the pick is proposal 1")
        return sv.grad[0]
    g = grad_of(y)
    for col in (0, 1, 3, 4):
        ok(float(g[1, col]) == 0.0, ("(b) the pick's label is 1 > p: exactly 0.0", col, float(g[1, col])))
    ok(float(g[1, 2]) > 0.0, "the pick's EP label .1 < p .993: flows, loss rises with the logit, descent LOWERS p_EP")
    ok(all(float(g[0, c]) < 0.0 for c in (0, 1, 2, 3, 4)) and all(float(g[2, c]) < 0.0 for c in (0, 1, 3, 4)), "(d) a better proposal's labels exceed p: loss falls with the logit, descent RAISES p")
    ok(float(g[2, 2]) == 0.0, ("(d) better j2: EP label .8 < p .818 -> exactly 0.0", float(g[2, 2])))
    ok(float(g[:, 5].abs().sum()) == 0.0 and float(g[3].abs().sum()) == 0.0, "DDC and the not-better p3 exactly 0.0")
    # card section 1.1 / Q-P16: STRICT inequality.  Pick live iff p > y; better j live iff p < y; an exact tie p == y is NOT live.
    #   (i) pick NC label 0.5, p_NC 0.7: live, gradient > 0 (lowers p);  (ii) the same pick with p_NC 0.3: exactly 0.0;  (iii) EP tie p == y: exactly 0.0 (pick and better j).
    def lg(p):
        return math.log(p / (1.0 - p))
    for p_nc, live in ((0.7, True), (0.3, False)):
        zz = z.clone()
        zz[0, :, 0] = lg(p_nc)                                                                       # every proposal has p_NC = p_nc, so the pick stays proposal 1
        yy = y.clone()
        yy[0, 1, 0] = 0.5                                                                            # NC label .5 (pdms .5 x ...; the half penalty): proposals 0 / 2 (NC 1) are above it
        sv = zz.clone().requires_grad_(True)
        r = RK.RankLoss(("pick",))
        r.term(score=sv, labels=yy, msk=msk, batch=1).backward()
        ok(RK.pick_index(zz).tolist() == [1], "the pick is still proposal 1 (EP logit 5)")
        gnc = float(sv.grad[0, 1, 0])
        ok((gnc > 0.0) if live else (gnc == 0.0), ("(i)/(ii) NC label .5 vs p", p_nc, gnc))
    # (iii) an EP tie p == y: sigmoid(0.0) == 0.5 exactly, label .5.  (A) the tie on a BETTER j; (B) the tie on the PICK.  Each is exactly 0.0, with a live control beside it.
    for case in ("A", "B"):
        zt = z.clone()
        yt = y.clone()
        if case == "A":
            zt[0, :, 2] = torch.tensor([4.0, 0.0, 0.0, -1.0], dtype=torch.float64)                   # pick = proposal 0 (EP p .982)
            yt[0, :, 2] = torch.tensor([0.05, 0.95, 0.5, 0.0], dtype=torch.float64)                  # j1 live (.95 > .5); j2 TIE (.5 == .5); pick .05 < .982 live
            tie, ctrl, sgn = 2, 1, -1.0
        else:
            zt[0, :, 2] = torch.tensor([0.0, -3.0, -3.0, -1.0], dtype=torch.float64)                 # pick = proposal 0 (EP p .5)
            yt[0, :, 2] = torch.tensor([0.5, 0.9, 0.8, 0.0], dtype=torch.float64)                    # the pick: p .5 == label .5 TIE; j1 live (.9 > .047)
            tie, ctrl, sgn = 0, 1, -1.0
        sv = zt.clone().requires_grad_(True)
        rt = RK.RankLoss(("pick",))
        rt.term(score=sv, labels=yt, msk=msk, batch=1).backward()
        ok(RK.pick_index(zt).tolist() == [0] and rt.last["pairs"]["pick"] >= 1, (case, RK.pick_index(zt).tolist(), rt.last))
        ok(float(sv.grad[0, tie, 2]) == 0.0, ("(iii) EP tie p == y: exactly 0.0", case, float(sv.grad[0, tie, 2])))
        ok(float(sv.grad[0, ctrl, 2]) * sgn > 0.0, ("control: the live better j", case, float(sv.grad[0, ctrl, 2])))
    # card 4.4(a): a pick with DAC label 0 gets a non-zero DAC gradient that LOWERS it; a better j with DAC label 1 gets one that RAISES it; label-0 on a better j is exactly 0.0
    yd = y.clone()
    yd[0, 1, 1] = 0.0                                                                                # the pick's DAC label 0 < p .5
    gd = grad_of(yd)
    ok(float(gd[1, 1]) > 0.0 and float(gd[0, 1]) < 0.0 and float(gd[2, 1]) < 0.0, ("DAC: pick lowered, better j raised", gd[:, 1].tolist()))
    yj = y.clone()
    yj[0, 2, 3] = 0.0                                                                                # better j2 with TTC label 0 (< p .5): never gets gradient (pdms still above the pick)
    gj = grad_of(yj)
    ok(float(gj[2, 3]) == 0.0 and float(gj[0, 3]) < 0.0, ("a label-0 head on a better j: exactly 0.0", gj[:, 3].tolist()))
    y2 = y.clone()
    y2[0, 1, 0] = 0.0                                                                                # (c) the pick's NC label 0 < p .5: its pdms is 0, so every proposal with pdms > 0 is above it
    g2 = grad_of(y2)
    ok(float(g2[1, 0]) > 0.0, ("(c) NC label 0 < p: flows, descent LOWERS p_NC", float(g2[1, 0])))
    ok(float(g2[1, 1]) == 0.0, "DAC label 1 > p: still exactly 0.0")


def chk_pick_semantics():
    """The loop reference (ref_term, with its own pick) against the term on random sets, value and pair set, alone and combined with ep / gate; STRICT PDMS inequality (an equal-PDMS proposal is no pair); an UNSAFE pick
       (pdms 0) pairs every proposal with positive PDMS; the gradient rows: NC, DAC, EP, TTC and C > 0, DDC EXACTLY 0.0 (and EP / TTC / C-only: the aggregate is what is read); accum identity (B 2 x accum 2 == B 4)."""
    for seed in range(3):
        for heads in (("pick",), ("ep", "gate", "pick"), ("pick", "ttc")):
            st, yt, mt, s, y, m = rand_case(seed + 30)
            r = RK.RankLoss(heads, ep_gap=0.02, gate_topk=8)
            L = r.term(score=st, labels=yt, msk=mt, batch=2)
            Lr, prs = ref_term(s, y, m, 2, heads, 0.02, 8)
            P = r.pairs(st, yt)
            for h in heads:
                mine = {tuple(int(v) for v in idx) for idx in P[h][0].nonzero().tolist()}
                ok(mine == prs[h] and len(mine) > 0, (seed, h, len(mine), len(prs[h])))
            ok(abs(float(L) - Lr) <= 1e-12 * max(1.0, abs(Lr)), (seed, heads, float(L), Lr))
    # strict inequality: two proposals with EXACTLY the pick's pdms, one above
    M = 4
    y = torch.zeros(1, M, 6, dtype=torch.float64)
    y[0, :, 0], y[0, :, 1] = 1.0, 1.0
    y[0, :, 2] = torch.tensor([0.5, 0.5, 0.75, 0.1], dtype=torch.float64)
    s = torch.zeros(1, M, 6, dtype=torch.float64)
    s[0, 1, 2] = 3.0                                                                                 # p1 is the pick (pdms 5*.5/12); p0 ties it, p2 is above, p3 below
    r = RK.RankLoss(("pick",))
    P = r.pairs(s, y)
    ok(sorted(P["pick"][0][0].nonzero().tolist()) == [[2, 1]], P["pick"][0][0].nonzero().tolist())
    # an unsafe pick pairs everything with pdms > 0
    y3 = y.clone()
    y3[0, 1, 0] = 0.0                                                                                # the pick p1 is unsafe: pdms 0
    P3 = RK.RankLoss(("pick",)).pairs(s, y3)
    ok(sorted(P3["pick"][0][0].nonzero().tolist()) == [[0, 1], [2, 1], [3, 1]], P3["pick"][0][0].nonzero().tolist())
    # gradient rows on bare logits
    st, yt, mt, *_ = rand_case(6)
    RK.RankLoss(("pick",)).term(score=st, labels=yt, msk=mt, batch=2).backward()
    g = st.grad.abs().sum((0, 1))
    ok(all(float(g[k]) > 0.0 for k in (0, 1, 2, 3, 4)) and float(g[5]) == 0.0, g.tolist())
    # accum identity
    st, yt, mt, s_, y_, m_ = rand_case(8, B=4)
    mt[1] = 0.0
    r = RK.RankLoss(("pick",))
    full = 0.5 * r.term(score=st, labels=yt, msk=mt, batch=4)
    full.backward()
    g_full = st.grad.clone()
    s2 = torch.tensor(s_, dtype=torch.float64, requires_grad=True)
    tot = 0.0
    for sl in (slice(0, 2), slice(2, 4)):
        t = 0.5 * r.term(score=s2[sl], labels=yt[sl], msk=mt[sl], batch=2) / 2.0
        t.backward()
        tot = tot + float(t)
    ok(abs(tot - float(full)) <= 1e-15 * max(1.0, abs(float(full))) and float((s2.grad - g_full).abs().max()) <= 1e-15 * float(g_full.abs().max()), (tot, float(full)))


def chk_accum_identity():
    """B 2 x accum 2 == B 4 x accum 1.  The trainer's contribution of a micro-batch is rank_w * term(batch = --batch) / accum, summed over the window; with a CONSTANT
       denominator that is the full-batch value.  A per-micro own-count denominator breaks it (train.py:1858-1871)."""
    st, yt, mt, s, y, m = rand_case(5, B=4)
    mt[1] = 0.0                                                                                      # a poorly covered window: one set unlabelled, so own-count denominators would differ per micro-batch
    w = 0.37
    r = RK.RankLoss(("ep", "gate"), gate_topk=8)
    full = w * r.term(score=st, labels=yt, msk=mt, batch=4) / 1.0
    full.backward()
    g_full = st.grad.clone()
    s2 = torch.tensor(s, dtype=torch.float64, requires_grad=True)
    tot = 0.0
    for sl in (slice(0, 2), slice(2, 4)):
        t = w * r.term(score=s2[sl], labels=yt[sl], msk=mt[sl], batch=2) / 2.0
        t.backward()
        tot = tot + float(t)
    ok(abs(tot - float(full)) <= 1e-15 * max(1.0, abs(float(full))), (tot, float(full)))
    ok(torch.equal(s2.grad, g_full) or float((s2.grad - g_full).abs().max()) <= 1e-15 * float(g_full.abs().max()), float((s2.grad - g_full).abs().max()))
    # an unlabelled micro-batch (msk 0) contributes LESS, not the same: the denominator does not renormalise
    mt0 = mt.clone()
    mt0[2:] = 0.0
    a = r.term(score=st.detach(), labels=yt, msk=mt0, batch=4)
    b = r.term(score=st.detach()[:2], labels=yt[:2], msk=mt[:2], batch=4)
    ok(float(a) == float(b), (float(a), float(b)))


def chk_boundary_semantics():
    """NC = 0.5 is not safe (strict >); EP gap is inclusive (>=); TTC pairs need exactly 1 / 0; the gate needs a STRICT pdms gap and j in the top-k."""
    M = 3
    y = torch.zeros(1, M, 6, dtype=torch.float64)
    y[0, :, 0] = torch.tensor([1.0, 1.0, 0.5])
    y[0, :, 1] = 1.0
    y[0, :, 2] = torch.tensor([0.75, 0.5, 0.1])
    s = torch.zeros(1, M, 6, dtype=torch.float64, requires_grad=True)
    r = RK.RankLoss(("ep", "gate"), ep_gap=0.25, gate_topk=3)
    P = r.pairs(s, y)
    ep = {tuple(i) for i in P["ep"][0].nonzero().tolist()}
    ok(ep == {(0, 0, 1)}, ep)                                                                        # 0.75 - 0.5 = 0.25 >= 0.25 is a pair (inclusive, exact in binary); p2 (NC 0.5) is unsafe -> no EP pair with it
    gt = {tuple(i) for i in P["gate"][0].nonzero().tolist()}
    ok(gt == {(0, 0, 2), (0, 1, 2)}, gt)                                                             # the unsafe p2 sits below both safe ones
    ok(P["safe"].tolist() == [[True, True, False]], P["safe"])


def chk_gradient_rows():
    """On bare logits: heads {ep, gate}: rows NC, DAC, EP non-zero; TTC, C, DDC EXACTLY 0.0.  Adding `ttc` turns the TTC row on.  The zero control: the term x 0 reads 0 everywhere."""
    st, yt, mt, *_ = rand_case(2)
    r = RK.RankLoss(("ep", "gate"), gate_topk=8)
    r.term(score=st, labels=yt, msk=mt, batch=2).backward()
    g = st.grad.abs().sum((0, 1))
    ok(all(float(g[k]) > 0.0 for k in (0, 1, 2)) and all(float(g[k]) == 0.0 for k in (3, 4, 5)), g.tolist())
    st.grad = None
    RK.RankLoss(("ep", "gate", "ttc"), gate_topk=8).term(score=st, labels=yt, msk=mt, batch=2).backward()
    g = st.grad.abs().sum((0, 1))
    ok(all(float(g[k]) > 0.0 for k in (0, 1, 2, 3)) and all(float(g[k]) == 0.0 for k in (4, 5)), g.tolist())
    st.grad = None
    (RK.RankLoss(("ep", "gate"), gate_topk=8).term(score=st, labels=yt, msk=mt, batch=2) * 0.0).backward()
    ok(float(st.grad.abs().sum()) == 0.0, "the zero control must read exactly 0.0")


def chk_model_reach():
    """The REAL model's default forward (trajectory detached): L_rank alone => non-zero on score_head rows NC/DAC/EP, score_dec, score_q_mlp, scene_proj (the trunk, because
       detach_scorer_context is False); EXACTLY 0.0 on traj_head, dec, queries and on score_head rows TTC/C/DDC.  Control: the term x 0 reads exactly 0.0 everywhere."""
    m = P6._tiny_model("command")
    img = torch.randn(1, 1, 3, 512, 960) * 0.1
    ego = torch.randn(1, 7) * 0.1
    cmd = torch.tensor([[1.0, 0, 0, 0]])
    g = np.random.default_rng(3)
    M = m.cfg.n_proposals
    y = np.zeros((1, M, 6))
    y[..., 0] = g.random((1, M)) > 0.4
    y[..., 1] = g.random((1, M)) > 0.4
    y[..., 2] = np.round(g.random((1, M)), 2)
    y[..., 3] = g.random((1, M)) > 0.5
    yt, mt = torch.tensor(y, dtype=torch.float32), torch.ones(1)
    out = {}
    for tag, scale in (("zero", 0.0), ("real", 1.0)):
        m.zero_grad(set_to_none=True)
        traj, score = m(img, ego, cmd)
        L = RK.RankLoss(("ep", "gate"), gate_topk=8).term(score=score.float(), labels=yt, msk=mt, batch=1)
        (L * scale).backward()
        grp = {k: float(sum(p.grad.double().abs().sum() for n, p in m.named_parameters() if n.startswith(k) and p.grad is not None)) for k in
               ("traj_head", "dec.", "queries", "score_dec", "score_q_mlp", "scene_proj", "score_head")}
        rows = m.score_head.weight.grad.double().abs().sum(1).tolist() if m.score_head.weight.grad is not None else [0.0] * 6
        out[tag] = (grp, rows)
        if tag == "real":
            ok(float(L) > 0.0, float(L))
    grp, rows = out["zero"]
    ok(all(v == 0.0 for v in grp.values()) and all(v == 0.0 for v in rows), out["zero"])
    grp, rows = out["real"]
    ok(grp["traj_head"] == 0.0 and grp["dec."] == 0.0 and grp["queries"] == 0.0, grp)                # the detach at model.py:779: EXACTLY 0.0
    ok(all(grp[k] > 0.0 for k in ("score_dec", "score_q_mlp", "scene_proj", "score_head")), grp)
    ok(all(rows[k] > 0.0 for k in (0, 1, 2)) and all(rows[k] == 0.0 for k in (3, 4, 5)), rows)


def chk_refusals():
    import argparse
    import train as T
    ap = T.build_parser()
    base = ["--backbone", "vits16", "--labeller", "online"]

    def refs(*extra, labeller="online"):
        a = ap.parse_args(["--backbone", "vits16", "--labeller", labeller, *extra])
        return T.rank_refusals(a)
    ok(refs() == [] and refs("--rank-w", "0.0") == [], "the defaults must be clean")
    ok(refs("--rank-w", "0.5", "--rank-heads", "ep,gate", "--rank-ep-gap", "0.05", "--rank-gate-topk", "4") == [])
    ok(refs("--rank-w", "0.5", "--rank-heads", "ttc") == [])
    ok(refs("--rank-w", "0.5", "--rank-heads", "pick") == [] and refs("--rank-w", "0.5", "--rank-heads", "ep,pick") == [])
    ok(any("named without `ep`" in r for r in refs("--rank-w", "0.5", "--rank-heads", "pick", "--rank-ep-gap", "0.1")) and any("named without `gate`" in r for r in refs("--rank-w", "0.5", "--rank-heads", "pick", "--rank-gate-topk", "4")),
       "pick alone carries no gap / top-k")
    for extra, frag in ((("--rank-w", "-0.1"), "finite number >= 0"), (("--rank-w", "nan"), "finite number >= 0"), (("--rank-w", "inf"), "finite number >= 0"),
                        (("--rank-w", "0.5"), "needs --rank-heads"), (("--rank-w", "0.5", "--rank-heads", "ep,ep"), "duplicate"),
                        (("--rank-w", "0.5", "--rank-heads", "ep,foo"), "unknown rank head"), (("--rank-w", "0.5", "--rank-heads", ""), "needs --rank-heads"),
                        (("--rank-heads", "ep"), "configured and never trained"), (("--rank-w", "0", "--rank-heads", "ep"), "configured and never trained"),
                        (("--rank-ep-gap", "0.1"), "named with --rank-w 0"), (("--rank-gate-topk", "4"), "named with --rank-w 0"),
                        (("--rank-w", "0.5", "--rank-heads", "ep", "--rank-ep-gap", "0.0"), "(0, 1]"), (("--rank-w", "0.5", "--rank-heads", "ep", "--rank-ep-gap", "1.5"), "(0, 1]"),
                        (("--rank-w", "0.5", "--rank-heads", "ep", "--rank-gate-topk", "0"), "[1, 64]"), (("--rank-w", "0.5", "--rank-heads", "ep", "--rank-gate-topk", "65"), "[1, 64]"),
                        (("--rank-w", "0.5", "--rank-heads", "ep", "--rank-ep-gap", "0.1", "--rank-gate-topk", "4"), "named without `gate`"),
                        (("--rank-w", "0.5", "--rank-heads", "gate", "--rank-ep-gap", "0.1"), "named without `ep`")):
        rs = refs(*extra)
        ok(rs and any(frag in r for r in rs), (extra, frag, rs))
    rs = refs("--rank-w", "0.5", "--rank-heads", "ep", labeller="bank")
    ok(rs and any("--labeller online" in r for r in rs), rs)
    # and the trainer itself exits 4 before any build
    pr = subprocess.run([sys.executable, str(HERE / "train.py"), "--backbone", "vits16", "--synthetic", "--cpu", "--rank-w", "0.5", "--rank-heads", "ep"], cwd=str(HERE), capture_output=True,
                        text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    ok(pr.returncode == 4 and "REFUSING" in pr.stdout and "--labeller online" in pr.stdout, (pr.returncode, pr.stdout[-300:], pr.stderr[-300:]))


def chk_identity_args():
    """a resume under another rank configuration is refused; an OLD checkpoint (no rank keys) reads as rank-off and resumes under a rank-off trainer."""
    import train as T
    for k, v in (("rank_w", 0.0), ("rank_heads", ""), ("rank_ep_gap", 0.02), ("rank_gate_topk", 8)):
        ok(k in T.IDENTITY_ARGS and T.IDENTITY_DEFAULTS[k] == v, k)
    keys = ("n_tuples", "n_scenes", "per_rank", "targets_sha256", "scorer_sha256", "calib_sha256", "frozen_sha256")
    new = {"args": {k: T.IDENTITY_DEFAULTS.get(k, 1) for k in T.IDENTITY_ARGS}}
    old = {"args": {k: v for k, v in new["args"].items() if not k.startswith("rank_")}}
    for d in (old, new):
        d.update({k: None for k in keys})
    ok(T._identity_diff(old, new) == [], T._identity_diff(old, new))
    for k, v in (("rank_w", 0.5), ("rank_heads", "ep"), ("rank_ep_gap", 0.1), ("rank_gate_topk", 4)):
        n2 = copy.deepcopy(new)
        n2["args"][k] = v
        ok(any("--" + k.replace("_", "-") in d for d in T._identity_diff(old, n2)), (k, T._identity_diff(old, n2)))


# ------------------------------------------------------------------------------------------------ trainer-level (the REAL train.run on a self-made tiny bank)
def _rich_labels():
    """a mixed safe / unsafe, spread-EP, TTC-0/1 label function keyed on the proposal's INDEX (the synthetic backend's smooth labels are nearly constant on an untrained
       model, which would leave every head with zero pairs: 'structurally dead' is exactly the failure class this file exists to catch)."""
    def f(props, salt=0):
        M = props.shape[0]
        m = np.arange(M)
        out = np.stack([(m % 3 != 0), (m % 5 != 0), ((m * 37) % 100) / 100.0, (m % 2 == 0), np.ones(M), np.ones(M)], -1).astype(np.float32)
        return out
    return f


def _run(td, bank, tag, extra=(), env=None, score_only=False, zero_bce=False):
    """One tiny trainer run in this process (gate_checks.instrumented_run).  score_only: the trajectory term's gradient is removed (wta on detached proposals); zero_bce: the BCE
       term is zeroed, so every gradient observed is L_rank's.  Returns cap."""
    import gate_checks as C
    import online_labeller as OL
    import torch.nn.functional as F
    import train as T
    real_wta, real_bce, real_syn = T.wta_loss, F.binary_cross_entropy_with_logits, OL.synthetic_labels

    def wta_det(traj, target, *a, **k):
        l, idx, lp, ly = real_wta(traj.detach(), target, *a, **{**k, "parts": True})
        return (l, idx, lp, ly) if k.get("parts") else (l, idx)
    old_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    OL.synthetic_labels = _rich_labels()
    if score_only:
        T.wta_loss = wta_det
    if zero_bce:
        F.binary_cross_entropy_with_logits = lambda x, *a, **k: x * 0.0
    try:
        cap = C.instrumented_run(P6.argv_p6(bank, str(td / f"o_{tag}"), extra=extra), mode="smoke", work=td / f"w_{tag}")
    finally:
        T.wta_loss, F.binary_cross_entropy_with_logits, OL.synthetic_labels = real_wta, real_bce, real_syn
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ok(cap.rc == 0, (tag, cap.rc, cap.error))
    return cap


def _grp(cap, rec=0):
    return {g: cap.grads[rec][g]["abs_sum"] for g in cap.grads[rec]}


def chk_trainer_reach():
    """argv -> parser -> RankLoss built -> term in the deferred backward -> .backward(): with the BCE zeroed and the trajectory term's gradient removed, every gradient observed is
       L_rank's: non-zero on score_head / score_dec / score_q_mlp and the trunk (--scorer-sees-trunk); EXACTLY 0.0 on traj_head and dec (candidates detached).  The control (no
       --rank-w) reads EXACTLY 0.0 on the same groups.  The flag VALUE reaches the loss: w = 1.0 doubles the gradient of w = 0.5."""
    td = P6._mkdtemp()
    bank = P6.make_bank(td)
    h = ("--rank-heads", "ep,gate")
    a = _grp(_run(td, bank, "w05", ("--rank-w", "0.5", *h), score_only=True, zero_bce=True))
    for g in ("score_head", "score_dec", "score_q_mlp", "backbone.lora", "scene_proj"):
        ok(a[g] > 0.0, (g, a))
    ok(a["traj_head"] == 0.0 and a["dec"] == 0.0, a)
    b = _grp(_run(td, bank, "w10", ("--rank-w", "1.0", *h), score_only=True, zero_bce=True))
    for g in ("score_head", "score_dec", "score_q_mlp"):
        ok(abs(b[g] / a[g] - 2.0) < 1e-4, (g, b[g] / a[g]))
    c = _grp(_run(td, bank, "off", (), score_only=True, zero_bce=True))
    ok(all(c[g] == 0.0 for g in ("score_head", "score_dec", "score_q_mlp", "backbone.lora", "traj_head", "dec")), c)
    shutil.rmtree(td, ignore_errors=True)


def chk_trainer_pick_reach():
    """O3, reachability from the ACTUAL CALLER: argv `--rank-w W --rank-heads pick` -> parser (train.py:2348-2355) -> RK.RankLoss built (train.py:1591) -> `rank.term(...)` (train.py:2011) -> `tot_ + a.rank_w * l_rank`
       (train.py:2014) -> `.backward()`.  With the BCE zeroed and the trajectory term's gradient removed, every gradient observed is L_rank(pick)'s: non-zero on score_head / score_dec / score_q_mlp / the trunk, EXACTLY 0.0 on
       traj_head and dec; the flag VALUE reaches the loss (w = 0.002 doubles w = 0.001; small weights, because the trainer clips the global grad norm at 1.0 (train.py:2040) and the ROUTED pick gradient is large enough that w = 0.5 / 1.0 both sit on the clip); the control (no --rank-w) reads EXACTLY 0.0; the logged rows carry rank_L > 0 and rank_pairs_pick > 0."""
    td = P6._mkdtemp()
    bank = P6.make_bank(td)
    h = ("--rank-heads", "pick")
    cap_a = _run(td, bank, "pw05", ("--rank-w", "0.001", *h), score_only=True, zero_bce=True)
    a = _grp(cap_a)
    for g in ("score_head", "score_dec", "score_q_mlp", "backbone.lora", "scene_proj"):
        ok(a[g] > 0.0, (g, a))
    ok(a["traj_head"] == 0.0 and a["dec"] == 0.0, a)
    b = _grp(_run(td, bank, "pw10", ("--rank-w", "0.002", *h), score_only=True, zero_bce=True))
    for g in ("score_head", "score_dec", "score_q_mlp"):
        ok(abs(b[g] / a[g] - 2.0) < 1e-4, (g, b[g] / a[g]))
    c = _grp(_run(td, bank, "poff", (), score_only=True, zero_bce=True))
    ok(all(c[g] == 0.0 for g in ("score_head", "score_dec", "score_q_mlp", "backbone.lora", "traj_head", "dec")), c)
    cap = _run(td, bank, "pon", ("--rank-w", "0.5", *h))
    ev = [r for r in cap.metrics if r.get("event") == "rank"]
    ok(len(ev) == 1 and ev[0]["heads"] == ["pick"] and ev[0]["aggregate"] == "navsim_v1" and ev[0]["mutate"] is None and ev[0]["rank_w"] == 0.5, ev)
    steps = [r for r in cap.metrics if "traj_L1" in r]
    # a step whose scorer pick IS the label-best has no pick pair (L exactly 0.0, counted): so finiteness on every row, and pairs > 0 with rank_L > 0 on at least one row (smoke step 0 of this bank)
    ok(len(steps) == 2 and all(math.isfinite(r["rank_L"]) and r["rank_L"] >= 0.0 and "rank_pairs_ep" not in r and (r["rank_pairs_pick"] > 0) == (r["rank_L"] > 0.0) for r in steps)
       and any(r["rank_pairs_pick"] > 0 and r["rank_sets_pick"] > 0 and r["rank_L"] > 0.0 for r in steps), steps)
    shutil.rmtree(td, ignore_errors=True)


def chk_trainer_mutation_arms():
    """the env arms that reach the trainer: `drop` computes L_rank and does not add it (the score_head reads exactly 0.0 where the unmutated run reads > 0); `traj_leak` gives the
       trajectory decoder a gradient (the unmutated run reads exactly 0.0).  Run as ONE check so the mutation harness can name it; the env is set by the CALLER (a mutation arm
       sets it and this check must go RED) -- here the unmutated expectations are asserted with the env as given."""
    td = P6._mkdtemp()
    bank = P6.make_bank(td)
    g = _grp(_run(td, bank, "m", ("--rank-w", "0.5", "--rank-heads", "ep,gate"), score_only=True, zero_bce=True))
    ok(g["score_head"] > 0.0 and g["traj_head"] == 0.0 and g["dec"] == 0.0, g)
    shutil.rmtree(td, ignore_errors=True)


def chk_trainer_row_fields():
    """a normal run (all terms): the `rank` start event carries the BUILT facts; every logged row carries rank_L finite, rank_w, rank_pairs_* > 0 and rank_sets_* > 0 for BOTH heads
       (a head with no pairs is structurally dead); pick_v1/best_v1 are logged; the loss stays finite; the six-term live set is unchanged.  A run WITHOUT --rank-w has none of
       these."""
    td = P6._mkdtemp()
    bank = P6.make_bank(td)
    cap = _run(td, bank, "on", ("--rank-w", "0.5", "--rank-heads", "ep,gate", "--rank-ep-gap", "0.05", "--rank-gate-topk", "4"))
    ev = [r for r in cap.metrics if r.get("event") == "rank"]
    ok(len(ev) == 1 and ev[0]["heads"] == ["ep", "gate"] and ev[0]["ep_gap"] == 0.05 and ev[0]["gate_topk"] == 4 and ev[0]["rank_w"] == 0.5 and ev[0]["aggregate"] == "navsim_v1"
       and ev[0]["reduce"] == "set-wmean/const-B" and ev[0]["mutate"] is None, ev)
    steps = [r for r in cap.metrics if "traj_L1" in r]
    ok(len(steps) == 2)
    for r in steps:
        ok(math.isfinite(r["rank_L"]) and r["rank_L"] > 0.0 and r["rank_w"] == 0.5 and r["rank_pairs_ep"] > 0 and r["rank_pairs_gate"] > 0 and r["rank_sets_ep"] > 0 and r["rank_sets_gate"] > 0
           and "rank_pairs_ttc" not in r, r)
        ok(all(math.isfinite(r[t]) for t in ("traj_L1", "score", "pos_L1", "yaw_L1", "bce_nc", "bce_dac", "bce_ep", "bce_ttc", "bce_c", "bce_ddc")), r)
    op = [r for r in cap.metrics if r.get("event") == "onpolicy"]
    ok(op and all(("pick_v1" in r and "best_v1" in r and "skill_v1" in r and r["best_v1"] >= r["pick_v1"] - 1e-12) for r in op if r["sets"]), op[:1])
    off = _run(td, bank, "none", ())
    ok(not any(r.get("event") == "rank" for r in off.metrics) and not any(any(k.startswith("rank_") for k in r) for r in off.metrics)
       and not any("pick_v1" in r for r in off.metrics), "an arm without --rank-w must carry no rank field and no rank event")
    shutil.rmtree(td, ignore_errors=True)


def _digest(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def _sub_run(refe_dir: Path, bank, out, extra=()):
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    e.pop("TANITAD_GATE_ARM", None)
    argv = P6.argv_p6(bank, out, steps=3, accum=2, extra=extra)
    pr = subprocess.run([sys.executable, str(refe_dir / "train.py")] + argv, cwd=str(refe_dir), env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert pr.returncode == 0, pr.stdout[-600:] + pr.stderr[-600:]
    rows = [json.loads(l) for l in (Path(out) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    keep = ("traj_L1", "score", "pos_L1", "yaw_L1", "bce_nc", "bce_dac", "bce_ep", "bce_ttc", "bce_c", "bce_ddc")
    return [{k: r[k] for k in keep} for r in rows if "traj_L1" in r], _digest(Path(out) / "model_final.pt")


def chk_off_path_identity(pristine: str | None = None):
    """--rank-w 0 (named) == the flags OMITTED == (with --pristine) the PRE-CHANGE trainer: the logged losses (6 decimals as logged = full float repr) and the final weights are
       bit-identical.  The in-repo form compares omitted against explicit 0.0 and asserts no rank object exists; the pre-change comparison needs the unpatched tree."""
    td = P6._mkdtemp()
    bank = P6.make_bank(td)
    rows_a, dig_a = _sub_run(HERE, bank, str(td / "a"))
    rows_b, dig_b = _sub_run(HERE, bank, str(td / "b"), ("--rank-w", "0.0"))
    ok(rows_a == rows_b and dig_a == dig_b, "explicit --rank-w 0.0 must equal the flags omitted")
    if pristine:
        rows_p, dig_p = _sub_run(Path(pristine), bank, str(td / "p"))
        ok(rows_a == rows_p and dig_a == dig_p, ("OFF path differs from the pre-change trainer", rows_a[:1], rows_p[:1], dig_a, dig_p))
        print(f"OFF-PATH IDENTITY vs pre-change trainer: {len(rows_a)} rows identical, final weights sha256 {dig_a[:16]} == {dig_p[:16]}")
    shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ the weight rule (weight_rule_rank.py)
def chk_weight_rule_decide():
    """the committed rule on hand-made r(w), rho(w): the smallest w with r >= 1 (inclusive); K2 when r(4) < 1; K2b when rho(w*) > 1; ERROR on a non-finite value."""
    import weight_rule_rank as WR
    ws = WR.CANDIDATES
    ok(ws == tuple(2.0 ** k for k in range(-6, 3)) and ws[0] == 1 / 64 and ws[-1] == 4.0 and len(ws) == 9, ws)
    lin = lambda r1: {w: w * r1 for w in ws}                                                          # noqa: E731
    d = WR.decide(lin(0.3), lin(0.1))
    ok(d["status"] == "SELECT" and d["rank_w"] == 4.0, d)                                              # r(2) = 0.6 < 1 <= r(4) = 1.2
    d = WR.decide(lin(1.0), lin(0.5))
    ok(d["status"] == "SELECT" and d["rank_w"] == 1.0, d)                                              # r(1) = 1.0 exactly qualifies (>=)
    d = WR.decide(lin(64.0), lin(0.01))
    ok(d["status"] == "SELECT" and d["rank_w"] == 1 / 64, d)
    d = WR.decide(lin(0.2), lin(0.1))
    ok(d["status"] == "K2" and d["rank_w"] is None, d)                                                 # r(4) = 0.8 < 1
    d = WR.decide(lin(1.0), lin(1.5))
    ok(d["status"] == "K2b" and d["rank_w"] is None and d["w_star"] == 1.0, d)                         # rho(1) = 1.5 > 1
    d = WR.decide(lin(1.0), lin(1.0))
    ok(d["status"] == "SELECT", d)                                                                     # rho(w*) == 1 is not above it
    d = WR.decide({**lin(1.0), 4.0: float("nan")}, lin(0.1))
    ok(d["status"] == "ERROR", d)
    try:
        WR.decide({1.0: 1.0}, {1.0: 1.0})
        ok(False)
    except ValueError:
        ok(True)


def chk_weight_rule_smoke():
    """the tool end to end on a self-made tiny bank with the SYNTHETIC labeller (stamped NOT A DECISION): it builds through the REAL train.run, collects labels from the arm's own
       labeller, differentiates L_rank, l_score and l_traj on the scorer and trunk groups, reports non-zero pair counts for BOTH heads, and r / rho are linear in w."""
    import io
    import contextlib
    import weight_rule_rank as WR
    import online_labeller as OL
    td = P6._mkdtemp()
    bank = P6.make_bank(td, 6)                                                                         # 6 scenes + 2 rank-1 twins = the 8 rows the rule draws at --batch 1
    args = " ".join(P6.argv_p6(bank, str(td / "wr_out"), extra=("--rank-heads", "ep,gate")))
    real = OL.synthetic_labels
    OL.synthetic_labels = _rich_labels()
    real_grad = torch.autograd.grad
    grad_calls = []                                   # A3: every torch.autograd.grad the tool takes, with its retain_graph / create_graph

    def rec_grad(*a_, **k_):
        grad_calls.append((bool(k_.get("retain_graph")), bool(k_.get("create_graph"))))
        return real_grad(*a_, **k_)
    buf = io.StringIO()
    try:
        torch.autograd.grad = rec_grad
        with contextlib.redirect_stdout(buf):
            rc = WR.main(["--args", args, "--refe", str(HERE), "--out", str(td / "wr.json")])
    finally:
        torch.autograd.grad = real_grad
        OL.synthetic_labels = real
    res = json.loads((td / "wr.json").read_text(encoding="utf-8"))
    ok(rc in (0, 5) and res["NOT_A_DECISION"] is True and res["label_backend"] == "synthetic" and res["heads"] == ["ep", "gate"], (rc, buf.getvalue()[-400:]))
    ok(res["donated_buffer"] is False, "the measurement must run with donated buffers OFF: three retain_graph grads on one --compile graph (pod 2026-11-21 16:43Z)")
    ok(len(grad_calls) == WR.GRAD_PASSES * WR.N_MICRO and not any(r or c for r, c in grad_calls) and res["grad_passes"] == 3 and res["retained_graph"] is False,
       ("A3: one forward + one backward per gradient, NEVER a retained graph: a retained --compile graph cannot be backpropagated twice (pod 2026-11-22 21:39Z, in-place "
        "version 18 vs 0)", len(grad_calls), grad_calls[:4]))
    ok(res["replay_max_abs_diff"]["score"] == 0.0 and res["replay_max_abs_diff"]["traj"] == 0.0,
       ("A3: the three passes of one micro-batch must be the SAME graph (replayed input and RNG state); on CPU the replay is exact", res["replay_max_abs_diff"]))
    ok(res["pairs_by_head"]["ep"] > 0 and res["pairs_by_head"]["gate"] > 0 and res["dead_heads"] == [], res["pairs_by_head"])
    ok(res["labelled_sets"] == 8 and res["scorer_tensors"] > 0 and res["trunk_tensors"] > 0, res)
    p = res["pooled"]
    ok(all(math.isfinite(p[k]) and p[k] > 0.0 for k in ("norm_grad_S_l_score", "norm_grad_S_rank_w1", "norm_grad_T_l_traj", "norm_grad_T_rank_w1", "r1", "rho1")), p)
    for w in WR.CANDIDATES:
        ok(res["r_by_w"][repr(w)] == w * p["r1"] and res["rho_by_w"][repr(w)] == w * p["rho1"], w)
    ok(res["rule"]["status"] == WR.decide({w: res["r_by_w"][repr(w)] for w in WR.CANDIDATES}, {w: res["rho_by_w"][repr(w)] for w in WR.CANDIDATES})["status"])
    ok(res["argv_sha256"] == hashlib.sha256("\x00".join(args.split()).encode()).hexdigest(), "the argv hash is of the argv AS GIVEN (no --rank-w was added to it)")
    shutil.rmtree(td, ignore_errors=True)


def chk_weight_rule_pick_smoke():
    """weight_rule_rank.py accepts the `pick` head: the tool end to end on a self-made tiny bank (SYNTHETIC labels, NOT A DECISION) builds through the REAL train.run with --rank-heads pick (and NO gap / top-k), measures
       r / rho from the pick term's gradients, and reports pairs > 0 for `pick` (no dead head)."""
    import io
    import contextlib
    import weight_rule_rank as WR
    import online_labeller as OL
    td = P6._mkdtemp()
    bank = P6.make_bank(td, 6)
    args = " ".join(P6.argv_p6(bank, str(td / "wr_out"), extra=("--rank-heads", "pick")))
    real = OL.synthetic_labels
    OL.synthetic_labels = _rich_labels()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = WR.main(["--args", args, "--refe", str(HERE), "--out", str(td / "wr.json")])
    finally:
        OL.synthetic_labels = real
    res = json.loads((td / "wr.json").read_text(encoding="utf-8"))
    ok(rc in (0, 5) and res["NOT_A_DECISION"] is True and res["heads"] == ["pick"], (rc, buf.getvalue()[-400:]))
    ok(res["pairs_by_head"]["pick"] > 0 and res["sets_with_pairs_by_head"]["pick"] > 0 and res["dead_heads"] == [], res["pairs_by_head"])
    p = res["pooled"]
    ok(all(math.isfinite(p[k]) and p[k] > 0.0 for k in ("norm_grad_S_l_score", "norm_grad_S_rank_w1", "norm_grad_T_l_traj", "norm_grad_T_rank_w1", "r1", "rho1")), p)
    for w in WR.CANDIDATES:
        ok(res["r_by_w"][repr(w)] == w * p["r1"], w)
    shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ the check table and the mutations
UNIT = ("analytic", "reference_audit", "analytic_identities", "pick_identity", "pick_analytic", "pick_routing", "pick_semantics", "accum_identity", "boundary_semantics", "gradient_rows", "model_reach", "refusals", "identity_args",
        "weight_rule_decide")
TRAINER = ("trainer_reach", "trainer_pick_reach", "trainer_mutation_arms", "trainer_row_fields", "off_path_identity", "weight_rule_smoke", "weight_rule_pick_smoke")
CHECKS = {n: globals()[f"chk_{n}"] for n in UNIT + TRAINER}

# (name, file, old, new, the check that must go RED).  Edits are exact single-occurrence string replacements in a COPY of refe/*.py.
LINE_MUTATIONS = [
    ("train: delete the line that ADDS the term", "train.py", "                            tot_ = tot_ + a.rank_w * l_rank", "                            pass", "trainer_reach"),
    ("train: delete the line that ADDS the term (pick arm)", "train.py", "                            tot_ = tot_ + a.rank_w * l_rank", "                            pass", "trainer_pick_reach"),
    ("train: the weight never reaches the loss (constant 1) (pick arm)", "train.py", "tot_ = tot_ + a.rank_w * l_rank", "tot_ = tot_ + 1.0 * l_rank", "trainer_pick_reach"),
    ("train: --rank-heads not passed to the built term (pick arm)", "train.py", "RK.RankLoss(a.rank_heads, ep_gap=a.rank_ep_gap,", "RK.RankLoss('ep', ep_gap=a.rank_ep_gap,", "trainer_pick_reach"),
    ("train: the weight never reaches the loss (constant 1)", "train.py", "tot_ = tot_ + a.rank_w * l_rank", "tot_ = tot_ + 1.0 * l_rank", "trainer_reach"),
    ("train: the term is never built (rank = None)", "train.py", "rank = RK.RankLoss(a.rank_heads, ep_gap=a.rank_ep_gap, gate_topk=a.rank_gate_topk) if a.rank_w > 0.0 else None", "rank = None", "trainer_row_fields"),
    ("train: --rank-heads not passed to the built term", "train.py", "RK.RankLoss(a.rank_heads, ep_gap=a.rank_ep_gap,", "RK.RankLoss('ep', ep_gap=a.rank_ep_gap,", "trainer_row_fields"),
    ("train: --rank-gate-topk not passed to the built term", "train.py", "gate_topk=a.rank_gate_topk) if", "gate_topk=8) if", "trainer_row_fields"),
    ("train: --rank-ep-gap not passed to the built term", "train.py", "ep_gap=a.rank_ep_gap, gate_topk", "ep_gap=0.02, gate_topk", "trainer_row_fields"),
    ("train: the term scores the DETACHED logits", "train.py", "l_rank = rank.term(score=P[\"score_rk\"] if P[\"score_rk\"] is not None else P[\"score\"],", "l_rank = rank.term(score=(P[\"score_rk\"] if P[\"score_rk\"] is not None else P[\"score\"]).detach(),", "trainer_reach"),
    ("train: the refusals are not called", "train.py", "    bad_rank = rank_refusals(a, cfg.n_proposals)", "    bad_rank = []", "refusals"),
    ("train: rank keys dropped from the resume identity", "train.py", '"rank_w", "rank_heads", "rank_ep_gap", "rank_gate_topk") + AUX', ') + AUX', "identity_args"),
    ("train: the OFF path builds the term anyway", "train.py", "if a.rank_w > 0.0 else None\n    if rank is not None:", "if a.rank_w >= 0.0 else None\n    if rank is not None:", "off_path_identity"),
    ("rank_loss: the pick head's pair mask is emptied (the pick term is never added)", "rank_loss.py", 'out["pick"] = (m, d, agg.unsqueeze(2) - agg.unsqueeze(1))', 'out["pick"] = (torch.zeros_like(m), d, agg.unsqueeze(2) - agg.unsqueeze(1))', "pick_analytic"),
    ("rank_loss: the pick term's routing is removed (prob stays whole)", "rank_loss.py", "            prob = torch.where(cons, prob, prob.detach())", "            prob = prob", "pick_routing"),
    ("rank_loss: the routing direction is flipped on the pick (y > p)", "rank_loss.py", "cons = torch.where(at.unsqueeze(2), y < prob.detach(), y > prob.detach())", "cons = torch.where(at.unsqueeze(2), y > prob.detach(), y > prob.detach())", "pick_routing"),
    ("rank_loss: the routing direction is flipped on a better proposal (y < p)", "rank_loss.py", "cons = torch.where(at.unsqueeze(2), y < prob.detach(), y > prob.detach())", "cons = torch.where(at.unsqueeze(2), y < prob.detach(), y < prob.detach())", "pick_routing"),
    ("rank_loss: pick pairs weight 1 instead of the PDMS gap", "rank_loss.py", 'out["pick"] = (m, d, agg.unsqueeze(2) - agg.unsqueeze(1))', 'out["pick"] = (m, torch.ones_like(d), agg.unsqueeze(2) - agg.unsqueeze(1))', "pick_analytic"),
    ("rank_loss: pick pairs include equal PDMS (>= instead of >)", "rank_loss.py", "m = at.unsqueeze(1) & (d > 0)", "m = at.unsqueeze(1) & (d >= 0)", "pick_semantics"),
    ("rank_loss: pick z reads the EP logit, not the aggregate", "rank_loss.py", "            agg = navsim_v1(prob)                                                            # [B, M] float64 WITH grad: the aggregate the rule argmaxes", "            agg = s[..., EP]", "pick_semantics"),
    ("rank_loss: the pick is argmaxed in float64 (a near-tie flips)", "rank_loss.py", "        return navsim_v1(torch.sigmoid(score.detach())).argmax(1)", "        return navsim_v1(torch.sigmoid(score.detach().double())).argmax(1)", "pick_identity"),
    ("rank_loss: the pick tie-break goes to the LAST index (the planner takes the first)", "rank_loss.py", "        return navsim_v1(torch.sigmoid(score.detach())).argmax(1)", "        return score.shape[1] - 1 - navsim_v1(torch.sigmoid(score.detach())).flip(1).argmax(1)", "pick_identity"),
    ("rank_loss: the aggregate drops the TTC term", "rank_loss.py", "    return p[..., NC] * p[..., DAC] * (p[..., EP] * w1[0] + p[..., TTC] * w1[1] + p[..., C] * w1[2]) / w1.sum()", "    return p[..., NC] * p[..., DAC] * (p[..., EP] * w1[0] + p[..., C] * w1[2]) / w1.sum()", "pick_identity"),
    ("weight rule: pick pair counts not reported", "weight_rule_rank.py", "            pairs[hh] += rank.last[\"pairs\"][hh]", "            pairs[hh] += 0", "weight_rule_pick_smoke"),
    ("rank_loss: EP pairs lose the safe condition", "rank_loss.py", "else (safe.unsqueeze(2) & safe.unsqueeze(1))", "else torch.ones_like(d, dtype=torch.bool)", "reference_audit"),
    ("rank_loss: gate pairs lose the top-k restriction", "rank_loss.py", "& top.unsqueeze(1) & (d > 0)", "& (d > 0)", "reference_audit"),
    ("rank_loss: gate weight = 1 instead of the PDMS gap", "rank_loss.py", 'out["gate"] = (m, d, z.unsqueeze(2) - z.unsqueeze(1))', 'out["gate"] = (m, torch.ones_like(d), z.unsqueeze(2) - z.unsqueeze(1))', "analytic"),
    ("rank_loss: the logits are detached", "rank_loss.py", "        s = score.double()", "        s = score.detach().double()", "gradient_rows"),
    ("rank_loss: per-set own-count denominator instead of the constant batch", "rank_loss.py", "return (total / float(batch)).to(score.dtype)", "return (total / float(max(int(has.sum()), 1))).to(score.dtype)", "accum_identity"),
    ("rank_loss: the labelled-set mask is dropped", "rank_loss.py", "total = total + (Lh * mw).sum()", "total = total + Lh.sum()", "analytic"),
    ("rank_loss: EP gap exclusive", "rank_loss.py", "m = both & (d >= self.ep_gap)", "m = both & (d > self.ep_gap)", "boundary_semantics"),
    ("rank_loss: safe uses >= 0.5", "rank_loss.py", "safe = (y[..., NC] > 0.5) & (y[..., DAC] > 0.5)", "safe = (y[..., NC] >= 0.5) & (y[..., DAC] > 0.5)", "boundary_semantics"),
    ("rank_loss: the gate reads the EP logit", "rank_loss.py", "z = F.logsigmoid(s[..., NC]) + F.logsigmoid(z_second)", "z = F.logsigmoid(s[..., NC]) + F.logsigmoid(s[..., EP])", "gradient_rows"),
    ("rank_loss: C joins the EP term's logit", "rank_loss.py", "            z = s[..., EP]\n", "            z = s[..., EP] + 1e-3 * s[..., C]\n", "gradient_rows"),
    ("weight rule: the K2 threshold is 0", "weight_rule_rank.py", "R_TARGET, RHO_MAX = 1.0, 1.0", "R_TARGET, RHO_MAX = 0.0, 1.0", "weight_rule_decide"),
    ("weight rule: the smallest-w scan starts from the largest", "weight_rule_rank.py", "w = next(w for w in ws if r_by_w[w] >= R_TARGET)", "w = next(w for w in reversed(ws) if r_by_w[w] >= R_TARGET)", "weight_rule_decide"),
    ("weight rule: the L_rank gradient is taken on detached logits", "weight_rule_rank.py", "        traj, score = traj.float(), score.float()", "        traj, score = traj.float(), score.float().detach()", "weight_rule_smoke"),
    ("weight rule: labels never collected from the arm's labeller (all-ones)", "weight_rule_rank.py", "lab_t, msk = torch.from_numpy(labels_np).to(dev), torch.from_numpy(mask_np).to(dev)", "lab_t, msk = torch.ones(1, cfg.n_proposals, 6).to(dev), torch.ones(1).to(dev)", "weight_rule_smoke"),
    ("weight rule: donated buffers left on (retain_graph on a compiled graph)", "weight_rule_rank.py", "    FC.donated_buffer = False", "    pass", "weight_rule_smoke"),
    ("weight rule: a retained graph again (A3: a compiled backward reuses saved buffers)", "weight_rule_rank.py",
     "gs = torch.autograd.grad(l_score, S + Tr, retain_graph=False, allow_unused=True)", "gs = torch.autograd.grad(l_score, S + Tr, retain_graph=True, allow_unused=True)", "weight_rule_smoke"),
    ("weight rule: pass 2 is not a replay of pass 1 (A3: L_rank on another graph)", "weight_rule_rank.py",
     "        traj2, score2 = forward(img, ego, goal, cal, seed_k)", "        traj2, score2 = forward(img * 0.5, ego, goal, cal, seed_k)", "weight_rule_smoke"),
    ("rank_loss: the loss is a sum, not a weighted mean", "rank_loss.py", "num / torch.where(den > 0, den, torch.ones_like(den))", "num", "analytic"),
]
# (name, env, the check that must go RED).  REFE_RANK_MUTATE arms: the env is set for the child; the named check must exit != 0.
ENV_MUTATIONS = [
    ("REFE_RANK_MUTATE=drop", {"REFE_RANK_MUTATE": "drop"}, "trainer_mutation_arms"),
    ("REFE_RANK_MUTATE=unsafe_ep", {"REFE_RANK_MUTATE": "unsafe_ep"}, "reference_audit"),
    ("REFE_RANK_MUTATE=shuffle", {"REFE_RANK_MUTATE": "shuffle"}, "reference_audit"),
    ("REFE_RANK_MUTATE=head_swap", {"REFE_RANK_MUTATE": "head_swap"}, "reference_audit"),
    ("REFE_RANK_MUTATE=traj_leak", {"REFE_RANK_MUTATE": "traj_leak"}, "trainer_mutation_arms"),
    ("REFE_RANK_MUTATE=<typo>", {"REFE_RANK_MUTATE": "dorp"}, "trainer_mutation_arms"),
    ("REFE_RANK_MUTATE=pick_oracle", {"REFE_RANK_MUTATE": "pick_oracle"}, "pick_analytic"),
    ("REFE_RANK_MUTATE=pick_noroute", {"REFE_RANK_MUTATE": "pick_noroute"}, "pick_routing"),
]


def run_fast(names):
    for n in names:
        CHECKS[n]()
    return N


def run_mutations():
    n = 0
    for name, fname, old, new, check in LINE_MUTATIONS:
        src = (HERE / fname).read_text(encoding="utf-8")
        assert src.count(old) == 1, f"mutation '{name}': the target text occurs {src.count(old)} times in {fname}"
        td = Path(tempfile.mkdtemp(prefix="rankmut_"))
        P6._TMP_DIRS.append(td)
        for f in HERE.glob("*.py"):
            shutil.copy2(f, td / f.name)
        for sub in ("navsim_pdm",):
            if (HERE / sub).is_dir():
                shutil.copytree(HERE / sub, td / sub, dirs_exist_ok=True)
        (td / fname).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
        pr = subprocess.run([sys.executable, str(td / "test_rank_loss.py"), "--check", check], cwd=str(td), capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        assert pr.returncode != 0, f"MUTATION STAYED GREEN: '{name}' ({fname}) must turn check '{check}' RED"
        print(f"RED  {name:70s} -> {check}")
        shutil.rmtree(td, ignore_errors=True)
        n += 1
    for name, env, check in ENV_MUTATIONS:
        pr = subprocess.run([sys.executable, str(HERE / "test_rank_loss.py"), "--check", check], cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8", **env))
        assert pr.returncode != 0, f"MUTATION STAYED GREEN: {name} must turn check '{check}' RED"
        print(f"RED  {name:70s} -> {check}")
        n += 1
    return n


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pristine = None
    if "--pristine" in argv:
        i = argv.index("--pristine")
        pristine = argv[i + 1]
        del argv[i:i + 2]
    if "--check" in argv:
        name = argv[argv.index("--check") + 1]
        if name == "off_path_identity":
            chk_off_path_identity(pristine)
        else:
            CHECKS[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    names = list(UNIT + TRAINER)
    for nme in names:
        if nme == "off_path_identity":
            chk_off_path_identity(pristine)
        else:
            CHECKS[nme]()
        print(f"ok   {nme}")
    total = N
    if "--fast" in argv:
        print(f"ZZRANK_PASS {total}")
        return 0
    m = run_mutations()
    print(f"ZZRANK_PASS {total}  ({len(names)} checks green + {m} RED mutations: {len(LINE_MUTATIONS)} source lines, {len(ENV_MUTATIONS)} REFE_RANK_MUTATE arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
