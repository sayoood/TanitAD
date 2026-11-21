"""pg_policy.py -- the PROBEGATE / RANDGATE forcing policy (PREREG_PROBEGATE.md section 3). Pure numpy, no I/O, no probe fitting.

Definitions are the L2 ones (code/l2_common.py): LON6 = (FOLLOW, CRUISE, BRAKE_TO, CREEP, HOLD, ACCELERATE) at v7 ids (0, 1, 3, 4, 5, 7);
decel group = LON6 idx 2,3,4 (v7 3,4,5); cruise group = LON6 idx 0,1 (v7 0,1); accel = LON6 idx 5 (v7 7).  v7 columns 2 and 6 are NOT LON6 hypotheses
and are ignored by the head view (a distractor in the unit test).
"""
from __future__ import annotations

import numpy as np

LON6_V7_IDS = (0, 1, 3, 4, 5, 7)
G_DECEL, G_CRUISE, G_ACCEL = (2, 3, 4), (0, 1), (5,)
NONE = -100                                   # the oracle-format "leave this window NATIVE" value


def head_view(raw8):
    """raw8 [n, 8] (v7 column order) -> (group 'D'/'C'/'A' of the argmax LON6 class, v7 id of the best DECEL-group class, v7 id of the best CRUISE-group class)."""
    l6 = np.asarray(raw8, np.float64)[:, list(LON6_V7_IDS)]
    am = l6.argmax(1)
    grp = np.where(np.isin(am, G_DECEL), "D", np.where(np.isin(am, G_CRUISE), "C", "A"))
    v7 = np.asarray(LON6_V7_IDS)
    dpick = v7[np.asarray(G_DECEL)[l6[:, list(G_DECEL)].argmax(1)]]
    cpick = v7[np.asarray(G_CRUISE)[l6[:, list(G_CRUISE)].argmax(1)]]
    return grp, dpick, cpick


def top_k_mask(score, k):
    """Boolean mask of the k highest scores; ties broken by window index (stable sort of -score)."""
    s = np.asarray(score, np.float64)
    order = np.argsort(-s, kind="stable")
    m = np.zeros(len(s), bool)
    m[order[:int(k)]] = True
    return m


def labels_from_flips(grp, dpick, cpick, up, down):
    lab = np.full(len(grp), NONE, np.int64)
    lab[up] = dpick[up]
    lab[down] = cpick[down]
    return lab


def probegate(score, raw8):
    """tau is label-free: the probe's decel set is the top-k scores with k = the number of windows whose HEAD argmax group is decel (equal decel rates)."""
    grp, dpick, cpick = head_view(raw8)
    k = int((grp == "D").sum())
    pdec = top_k_mask(score, k)
    up = pdec & (grp != "D")                  # probe says decel, head's group is not decel  -> force best DECEL-group class
    down = (~pdec) & (grp == "D")             # probe says not-decel, head's group is decel   -> force best CRUISE-group class
    tau = float(np.sort(np.asarray(score, np.float64))[::-1][k - 1]) if k > 0 else float("inf")
    return labels_from_flips(grp, dpick, cpick, up, down), up, down, {"k": k, "tau": tau, "n": int(len(grp))}


def randgate(raw8, n_up, n_down, seed=0):
    """Same flip counts per direction, windows drawn at random from the eligible sets (up: head group != D; down: head group == D), then the same forced-class rule."""
    grp, dpick, cpick = head_view(raw8)
    rng = np.random.default_rng(seed)
    ue, de = np.flatnonzero(grp != "D"), np.flatnonzero(grp == "D")
    ru = rng.choice(ue, int(n_up), replace=False)         # draw order is part of the contract: UP first, then DOWN, from one generator
    rd = rng.choice(de, int(n_down), replace=False)
    up = np.zeros(len(grp), bool)
    down = np.zeros(len(grp), bool)
    up[ru] = True
    down[rd] = True
    return labels_from_flips(grp, dpick, cpick, up, down), up, down, {"eligible_up": int(len(ue)), "eligible_down": int(len(de))}
