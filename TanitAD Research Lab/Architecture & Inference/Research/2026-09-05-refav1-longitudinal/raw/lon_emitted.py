# -*- coding: utf-8 -*-
"""P2b - WHAT THE PLANNER ACTUALLY EMITS ON THE LONGITUDINAL CHANNEL, per arm.

The lateral half of this question is already banked (`kappa_quantisation.txt`:
the winning plan's curvature series is EXACTLY CONSTANT on 31/40 windows, and
reads exactly 0.000000 or 0.080000 - the decoded token's canonical profile
verbatim). This is the same reading on channel 0, and it is the diagnostic that
says in advance whether a GOAL change can move anything: if the emitted `a` is
already the goal's `a` on most windows, changing the goal changes the plan; if
the emitted `a` is zero regardless of the goal, the blocker is the COST.

Zero GPU. ASCII only.
"""
import glob
import os
import sys

import numpy as np

from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)
from tanitad.refs import refa_v1 as R

LAT_V70, LON_V70 = list(TACTICAL_LAT_ACTIONS_V7), list(TACTICAL_LON_ACTIONS_V7)
DT, K, OP = 0.2, 10, 30
P = "C:/Users/Admin/refav1_margin/p4out"
SRC = ["cem", "baseline:cv", "baseline:hold_v0", "baseline:proposal",
       "baseline:decel_1.5", "baseline:decel"]

print("# Emitted LONGITUDINAL control, per arm. `a_goal` is the decoded token's")
print("# own canonical a[0] at that window's v0 -- so `a_emit == a_goal` means")
print("# the plan IS the goal profile and a goal change would move it.")
print()
hdr = ("%-14s %4s %9s %9s %9s %9s %9s %9s"
       % ("arm", "n", "mean|a|", "frac a==0", "const-a", "==a_goal",
          "cem frac", "cost~cv"))
print(hdr)
print("-" * len(hdr))
for a in sys.argv[1:]:
    fs = sorted(glob.glob(os.path.join(P, "dump_" + a, "decisions", "*.npz")))
    if not fs:
        print("%-14s -- DUMP ABSENT" % a)
        continue
    C, LON, LAT, V0, SRCI, PC, BC = [], [], [], [], [], [], []
    for f in fs:
        with np.load(f) as z:
            C.append(z["cl_controls"])
            LON.append(z["goal_lon_cl"])
            LAT.append(z["goal_lat_cl"])
            SRCI.append(z["plan_source_cl"])
            PC.append(z["plan_cost_cl"])
            BC.append(z["basecost_cv_cl"])
        e = f.replace(os.sep + "decisions" + os.sep, os.sep)
        with np.load(e) as z2:
            V0.append(z2["v0"])
    C = np.concatenate(C); LON = np.concatenate(LON); LAT = np.concatenate(LAT)
    V0 = np.concatenate(V0); SRCI = np.concatenate(SRCI)
    PC = np.concatenate(PC); BC = np.concatenate(BC)
    acc = C[..., 0]                                       # [n, K]
    n = acc.shape[0]
    a_goal = np.array([float(R.canonical_controls(LAT_V70[LAT[i]],
                                                  LON_V70[LON[i]],
                                                  float(V0[i]), OP, DT)[0, 0])
                       for i in range(n)])
    const = np.array([len(np.unique(acc[i])) == 1 for i in range(n)])
    eqgoal = np.abs(acc[:, 0] - a_goal) < 1e-6
    near = np.isclose(PC, BC, rtol=1e-6, atol=1e-12)
    print("%-14s %4d %9.5f %9.3f %9.3f %9.3f %9.3f %9.3f"
          % (a, n, np.abs(acc).mean(), (np.abs(acc).max(1) < 1e-9).mean(),
             const.mean(), eqgoal.mean(),
             (SRCI == SRC.index("cem")).mean(), near.mean()))
print()
print("# distinct emitted a[0] values (the longitudinal quantisation):")
for a in sys.argv[1:]:
    fs = sorted(glob.glob(os.path.join(P, "dump_" + a, "decisions", "*.npz")))
    if not fs:
        continue
    C = np.concatenate([np.load(f)["cl_controls"] for f in fs])
    u, c = np.unique(np.round(C[:, 0, 0], 6), return_counts=True)
    pairs = sorted(zip(u.tolist(), c.tolist()), key=lambda x: -x[1])[:8]
    print("  %-14s %d distinct; top: %s"
          % (a, len(u), ", ".join("%+.6f x%d" % p for p in pairs)))
