#!/usr/bin/env python3
"""SEL-RANK Deliverable 2 support: a REACH PROXY for the cheapest candidate arm, fitted ONLY on the navtrain held-out (DEV) table with grouped-by-log cross-validation.

Nothing here touches navtest.  Nothing here is the arm: it is a measurement of how much of the non-zero-pick pool a CPU-only post-hoc recalibration of the six scorer
logits can recover on SIBLING scorers (SFT-4 u400 + 400 updates, arms N-s0/1/2) -- NOT on P2's scorer, which has no held-out table.  The reach is therefore a proxy, and the
cross-scorer transfer rows say how far it can be trusted.

Candidate family CAL12 (12 parameters, identity at init = the planner's rule):
    p'_h = sigmoid(a_h * z_h + b_h)  for h in nc, dac, ep, ttc, c                       (10)
    score = p'_nc^alpha * p'_dac^beta * (wE p'_ep + wT p'_ttc + wC p'_c)               (alpha, beta; wE, wT, wC with wE+wT+wC = 12 -> 12 fixed ... see below)
  parameters: a[5], b[5], log-alpha, log-beta, and the 3 weights through a softmax x 12 (2 free)  -> 14 free parameters; identity: a=1, b=0, alpha=beta=1, w=(5,5,2).
Family W5 (5 parameters): identity heads, only alpha, beta and the three weights are learned.
Objective (fixed in advance, never tuned on a test fold): minimise  -mean_sets sum_j softmax(log score_j / TAU)_j * PDMS_j  +  L2 * ||theta - theta0||^2,
TAU = 0.1, L2 = 1e-3, Adam lr 0.02 for 400 steps.  Evaluation: HARD argmax of the fitted score on the held-OUT fold; gain = mean PDMS at that pick - mean PDMS at the
planner's navsim_v1 pick, 4 folds grouped by LOG (24 logs), log-cluster bootstrap over the 24 logs for the pooled out-of-fold gain.
Controls: (N) the same fit on within-token SHUFFLED PDMS labels must give an out-of-fold gain <= 0 (a fit that manufactures gain from noise is a broken probe);
          (I) the identity parameters must reproduce the planner's pick on 100 % of sets (exactly 0.0 gain).
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe selrank_cal_proxy.py --out ../raw/cal_proxy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zero_tail import Boot, B_BOOT, SEED  # noqa: E402
from selrank_attr import agg_v1_np, sigmoid64, ratio  # noqa: E402
from selrank_dev import load_dev  # noqa: E402

TAU, L2, LR, STEPS, N_FOLDS = 0.1, 1e-3, 0.02, 400, 4      # TAU is overridden by --tau (the first run at 0.1 was 40x wider than the median top1-top8 log-score margin 0.025: see RESULT.md)
torch.set_num_threads(4)


class Cal(torch.nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.a = torch.nn.Parameter(torch.ones(5), requires_grad=(family == "CAL14"))
        self.b = torch.nn.Parameter(torch.zeros(5), requires_grad=(family == "CAL14"))
        self.la = torch.nn.Parameter(torch.zeros(2))                                   # log alpha, log beta
        self.w = torch.nn.Parameter(torch.log(torch.tensor([5.0, 5.0, 2.0]) / 12.0))   # softmax logits of the 3 weights
        self.theta0 = [p.detach().clone() for p in self.parameters()]

    def log_score(self, Z):                                                            # Z [n,64,5] raw logits (nc,dac,ep,ttc,c)
        x = self.a * Z + self.b
        lp = F.logsigmoid(x)                                                           # log p'
        p = torch.exp(lp)
        w = torch.softmax(self.w, 0) * 12.0
        mix = w[0] * p[..., 2] + w[1] * p[..., 3] + w[2] * p[..., 4]
        return torch.exp(self.la[0]) * lp[..., 0] + torch.exp(self.la[1]) * lp[..., 1] + torch.log(mix.clamp_min(1e-12)) - np.log(12.0)

    def reg(self):
        return sum(((p - q) ** 2).sum() for p, q in zip(self.parameters(), self.theta0))


def fit(Z, Y, family, seed=0):
    torch.manual_seed(seed)
    m = Cal(family)
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        ls = m.log_score(Z)
        loss = -(torch.softmax(ls / TAU, dim=1) * Y).sum(1).mean() + L2 * m.reg()
        loss.backward()
        opt.step()
    return m


def hard_pick(m, Z):
    with torch.no_grad():
        return m.log_score(Z).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw", "cal_proxy.json"))
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--arms", default="N-s0,N-s1,N-s2")
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    global TAU, STEPS
    TAU, STEPS = a.tau, a.steps
    keys, log, sub, pdms, S = load_dev(tuple(a.arms.split(",")))
    N, M, _ = sub.shape
    ulogs, log_idx = np.unique(log, return_inverse=True)
    B = Boot(log_idx, len(ulogs))
    ar = np.arange(N)
    rng = np.random.default_rng(SEED)
    fold_of_log = np.zeros(len(ulogs), int)
    perm = rng.permutation(len(ulogs))
    for i, j in enumerate(perm):
        fold_of_log[j] = i % N_FOLDS
    fold = fold_of_log[log_idx]
    Y = torch.from_numpy(pdms)
    R = {"_what": "SEL-RANK reach PROXY on the navtrain held-out (dev) table. Grouped-by-log CV, 4 folds. SIBLING scorers, NOT P2. Nothing here touches navtest.",
         "fixed_in_advance": {"TAU(set by --tau)": a.tau, "L2": L2, "lr": LR, "steps": STEPS, "folds": N_FOLDS, "fold_assignment_seed": SEED, "families": ["W5", "CAL14"]},
         "estimator": "log-cluster bootstrap over the 24 held-out logs of the POOLED out-of-fold gain (the fits are not re-run per resample)", "arms": {}}
    for arm, sc7 in S.items():
        Z = torch.from_numpy(sc7[..., :5].copy()).float()
        base_pick = agg_v1_np(sigmoid64(sc7[..., :6])).argmax(1)
        pp = pdms[ar, base_pick]
        orc = pdms.max(1)
        nz = pp > 0
        pool = ratio(B, (orc - pp) * nz, scale=100)
        rec = {"baseline_pdms_x100": ratio(B, pp, scale=100), "pool_x100": pool, "families": {}}
        for fam in ("W5", "CAL14"):
            # (I) identity control: untrained module reproduces the planner pick
            m0 = Cal(fam)
            ident = int((hard_pick(m0, Z) == base_pick).sum())
            oof = np.full(N, -1)
            ins = []
            for f in range(N_FOLDS):
                tr, te = fold != f, fold == f
                m = fit(Z[tr], Y[tr], fam, seed=f)
                oof[te] = hard_pick(m, Z[te])
                ins.append(float((pdms[tr][np.arange(tr.sum()), hard_pick(m, Z[tr])] - pp[tr]).mean() * 100))
            d = pdms[ar, oof] - pp
            # (N) negative control: shuffled labels within token
            Ysh = pdms.copy()
            for i in range(N):
                Ysh[i] = Ysh[i, rng.permutation(M)]
            oofn = np.full(N, -1)
            for f in range(N_FOLDS):
                tr, te = fold != f, fold == f
                m = fit(Z[tr], torch.from_numpy(Ysh[tr]), fam, seed=f)
                oofn[te] = hard_pick(m, Z[te])
            dn = pdms[ar, oofn] - pp
            rec["families"][fam] = {
                "identity_control_pick_matches_planner": {"match": ident, "of": N},
                "out_of_fold_gain_x100": ratio(B, d, scale=100),
                "out_of_fold_gain_from_nonzero_pick_x100": ratio(B, d * nz, scale=100),
                "out_of_fold_share_of_pool": ratio(B, d * nz, (orc - pp) * nz),
                "in_sample_gain_x100_mean_over_folds": float(np.mean(ins)),
                "tokens_improved_worsened": [int((d > 1e-12).sum()), int((d < -1e-12).sum())],
                "negative_control_shuffled_labels_oof_gain_x100": ratio(B, dn, scale=100)}
            print(arm, fam, "oof gain", round(rec["families"][fam]["out_of_fold_gain_x100"]["point"], 3), "share", round(rec["families"][fam]["out_of_fold_share_of_pool"]["point"], 3),
                  "insample", round(float(np.mean(ins)), 3), "neg", round(rec["families"][fam]["negative_control_shuffled_labels_oof_gain_x100"]["point"], 3), flush=True)
        # cross-scorer transfer: fit on THIS scorer's full table, apply to the OTHER two (a seed-transfer proxy for 'fit on one scorer, deploy on another')
        mfull = fit(Z, Y, "CAL14", seed=99)
        tr = {}
        for arm2, sc2 in S.items():
            if arm2 == arm:
                continue
            Z2 = torch.from_numpy(sc2[..., :5].copy()).float()
            bp2 = agg_v1_np(sigmoid64(sc2[..., :6])).argmax(1)
            d2 = pdms[ar, hard_pick(mfull, Z2)] - pdms[ar, bp2]
            tr[arm2] = ratio(B, d2, scale=100)
        rec["cross_scorer_transfer_CAL14_fit_on_this_applied_to"] = tr
        rec["cross_scorer_transfer_NOTE"] = ("fitted on ALL 3,137 sets of one scorer and applied to the OTHER scorers on the SAME sets: in-sample over SETS (only the scorer differs), "
                                             "so it is NOT an out-of-sample reach and is not quoted as one; the out-of-fold rows above are")
        R["arms"][arm] = rec
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
