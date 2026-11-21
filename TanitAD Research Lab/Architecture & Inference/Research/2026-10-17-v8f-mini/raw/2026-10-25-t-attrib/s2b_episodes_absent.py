"""s2b_episodes_absent.py -- Q1 add-ons (EXPLORATORY, MEASURED on eval-141): the worst episodes by name, and the ABSENT-slot values for completeness
(R63: T's absent-slot values are O's input, not error; shown only so 'present slot AND pooled' is complete). Also the turn/nav cell share used in the ceilings."""
import json
import numpy as np
from tcommon import *

T = load()
grid = json.load(open(S + "raw_in/eval_grid_windows.json"))
order = grid["order"]
eid, has = T["ci"], T["has"]
G = T["gt_std"]
for arm in ("M2b_s0", "M2b_s1"):
    E, P = sq(T, arm)
    E[~has], P[~has] = 0.0, False
    TOT = E.sum()
    ue = np.unique(eid)
    es = np.array([E[eid == e].sum() for e in ue])
    ec = np.array([P[eid == e].sum() for e in ue])
    o = np.argsort(-es)[:10]
    print(f"\n{arm}: worst 10 episodes by squared error (share of total SSE; n_windows; dominant cell by SSE; mean cell MSE)")
    for e in o:
        m = eid == e
        c = int(np.argmax(E[m].sum(0)))
        print(f"   ep {e:3d} {order[e][:8]}  share {es[e] / TOT:.3f}  n_win {m.sum():3d}  cell MSE {es[e] / ec[e]:.3f}  dominant {CONS_NAMES[c]} ({E[m][:, c].sum() / es[e]:.2f} of the episode)")
    print(f"   top-10 episodes together: {es[o].sum() / TOT:.3f} of SSE, holding {ec[o].sum() / P.sum():.3f} of present cells; median episode share {np.median(es) / TOT:.4f}")
    # absent slots (T's own values, GT absent) -- NOT error (R63)
    absent = (~np.isfinite(G)) & has[:, None]
    Tf = T[f"T_{arm}"]
    print(f"   absent-slot T values (std units; the target O is teacher-forced on is 0): RMS by group: " +
          ", ".join(f"{g} {np.sqrt(np.mean(Tf[:, idx][absent[:, idx]] ** 2)):.3f} (n {int(absent[:, idx].sum())})" for g, idx in GROUPS.items()) +
          f";  pooled over present+absent, with absent scored against 0: latcons-like 'would-be' SSE share absent = " +
          ", ".join(f"{g} {np.sum(Tf[:, idx][absent[:, idx]] ** 2) / (np.sum(Tf[:, idx][absent[:, idx]] ** 2) + E[:, idx].sum()):.2f}" for g, idx in GROUPS.items()))
    turn = np.isfinite(G[:, 1]) & has
    nav0 = T["nav_tok"] == 0
    cols = list(range(1, 10))
    print(f"   turn cells 1-9: share of total SSE {E[:, cols].sum() / TOT:.3f}; in route-silent (nav FOLLOW) turn windows {E[turn & nav0][:, cols].sum() / TOT:.3f}; in route-speaking turn windows {E[turn & ~nav0][:, cols].sum() / TOT:.3f}")
    m = turn & nav0
    print(f"   route-silent turn windows: n {m.sum()}; their turn_t_start GT mean {G[m, 1].mean() * T['sd'][1] + T['mu'][1]:.2f} s (route-speaking: {G[turn & ~nav0, 1].mean() * T['sd'][1] + T['mu'][1]:.2f} s); nav_d valid? ")
