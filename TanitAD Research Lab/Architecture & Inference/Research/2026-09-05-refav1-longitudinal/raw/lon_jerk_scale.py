# -*- coding: utf-8 -*-
"""P6 - IS THE SEAM PAIR WORTH TWO GPU HOURS? Price the jerk term BEFORE running.

Both streams have now established that `--jerk-seam` is inert at `W_JERK = 0`.
The obvious successor is a pair at the SHIPPED `W_JERK = 0.02`. But `W_KAPPA` in
the same triple is **15.11245**, i.e. 756x larger, so the jerk term may be
negligible against the terms it trades with -- in which case the pair would be
"technically live, practically inert", which is the same trap one notch weaker
and would cost two more GPU hours to discover.

Computed from the BANKED `wk15` / `lonshift` plans, exactly as `_cost_chunk`
does, at the weights the proposed arms would use. Zero GPU, ASCII only.
"""
import glob
import sys

import numpy as np

P = "C:/Users/Admin/refav1_margin/p4out"
DT = 0.2
W_JERK, W_KAPPA = 0.02, 15.11245


def load(tag):
    C, PC, BCV = [], [], []
    for f in sorted(glob.glob(P + "/dump_" + tag + "/decisions/*.npz")):
        with np.load(f) as z:
            C.append(z["cl_controls"])
            PC.append(z["plan_cost_cl"])
            BCV.append(z["basecost_cv_cl"])
    if not C:
        return None
    return np.concatenate(C), np.concatenate(PC), np.concatenate(BCV)


print("# The three cost terms at the PROPOSED seam-arm weights")
print("# (W_JERK %.4f, W_KAPPA %.5f), computed from the banked plans exactly as"
      % (W_JERK, W_KAPPA))
print("# `_cost_chunk` does. `plan_cost` is what the arm actually minimised at")
print("# ITS OWN weights (W_JERK = 0), so the goal term is recovered as")
print("# plan_cost - W_KAPPA*mean(kappa^2), which is exact for those arms.")
print()
hdr = ("%-10s %5s %13s %13s %13s %13s"
       % ("arm", "n", "jerk term", "SEAM delta", "kappa term", "goal term"))
print(hdr)
print("-" * len(hdr))
rows = {}
for tag in ("wk15", "lonshift", "lonseam"):
    r = load(tag)
    if r is None:
        print("%-10s -- DUMP ABSENT" % tag)
        continue
    C, PC, BCV = r
    a = C[..., 0]
    kap = C[..., 1]
    jerk = np.diff(a, axis=1) / DT
    t_jerk = W_JERK * (jerk ** 2).mean(1)
    t_kap = W_KAPPA * (kap ** 2).mean(1)
    t_goal = PC - t_kap                       # the arms ran with W_JERK = 0
    # the SEAM's own contribution: prepend a0 (unknown here) -- bound it by the
    # LARGEST possible first-step jerk, |a[0] - a0| <= |a[0]| + max|a0|.
    a0 = []
    for f in sorted(glob.glob(P + "/dump_" + tag + "/decisions/*.npz")):
        with np.load(f) as z:
            a0.append(z["ha0_ext_controls"][:, 0, 0])
    a0 = np.concatenate(a0)
    seam = ((a[:, 0] - a0) / DT) ** 2
    # adding one term to a mean over H-1 -> H terms
    H = a.shape[1]
    t_jerk_seam = W_JERK * (((jerk ** 2).sum(1) + seam) / H)
    rows[tag] = (t_jerk, t_jerk_seam, t_kap, t_goal)
    print("%-10s %5d %13.3e %13.3e %13.3e %13.3e"
          % (tag, len(a), t_jerk.mean(), (t_jerk_seam - t_jerk).mean(),
             t_kap.mean(), t_goal.mean()))

print()
print("== THE QUESTION: can the seam term MOVE a decision? ==")
for tag, (tj, tjs, tk, tg) in rows.items():
    d_seam = np.abs(tjs - tj)
    print("  %-10s  |seam contribution| median %.3e   max %.3e" % (tag, np.median(d_seam), d_seam.max()))
    print("             vs |goal term| median %.3e   vs |kappa term| median %.3e"
          % (np.median(np.abs(tg)), np.median(np.abs(tk))))
    denom = np.median(np.abs(tg)) + np.median(np.abs(tk))
    print("             seam / (goal + kappa)  median ratio  %.3e" % (np.median(d_seam) / max(denom, 1e-30)))
    # the decision scale: how much does the objective SPREAD across candidates?
    print()
print("# A term whose magnitude is orders below the spread of the terms it trades")
print("# against cannot re-rank candidates, and an arm built on it would be")
print("# 'technically live, practically inert' -- the same trap one notch weaker.")
