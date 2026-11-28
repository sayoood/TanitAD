#!/usr/bin/env python
"""td_closing.py -- T0, descriptive (names nothing): can the temporal difference of the lead head's own gap reads stand in for its closing-speed output?
Reader dumps of card ADD36 (windows every 0.8 s per episode). For a span of 0.8 s and 1.6 s: c_td = (head gap at t - head gap at t - span) / span, set against the
label closing speed (LS slope of the trailing label run, at least 3 frames) and used in the 2.0 s forecast head gap + c * 2.0. No fit on these windows.
Written by the lead, 2026-11-19. Usage: td_closing.py <join npz> <dump npz>...
"""
import sys

import numpy as np

DT = 0.2
J = np.load(sys.argv[1], allow_pickle=True)
off, n, valid = J["off"], J["n"], J["lead_valid"]
gap = J["lead_gap"].astype(float)
cidx = {str(c): i for i, c in enumerate(J["clip"])}


def ok(r):
    g = gap[r]
    return bool(valid[r] == 1 and np.isfinite(g) and 0 <= g <= 60)


def closing(ci, j):
    o = int(off[ci])
    r = o + j
    idx = [r]
    f = r - 1
    while f >= o and len(idx) < 5 and ok(f) and abs(gap[f + 1] - gap[f]) <= 3.0:
        idx.append(f)
        f -= 1
    if len(idx) < 3:
        return None
    idx = idx[::-1]
    G = gap[idx]
    t = np.arange(len(idx)) * DT
    tb = t.mean()
    return float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())


print("T0, descriptive; STRATEGIC: NOT COMPUTED (n = 0; read of the lead closing speed, no strategic instrument)")
for dp in sys.argv[2:]:
    d = np.load(dp, allow_pickle=True)
    eid = d["eid"].astype(str)
    g0 = d["g0"].astype(int)
    hg, hc = d["present"][:, 1].astype(float), d["present"][:, 2].astype(float)
    gt, g2 = d["gap_t"].astype(float), d["gap_t2"].astype(float)
    key = {(e, int(j)): i for i, (e, j) in enumerate(zip(eid, g0))}
    for span in (4, 8):
        rows = []
        for i, (e, j) in enumerate(zip(eid, g0)):
            k = key.get((e, int(j) - span))
            if k is None or e not in cidx:
                continue
            if not all(np.isfinite(x) for x in (gt[i], g2[i], hg[i], hg[k], gt[k])):
                continue
            c = closing(cidx[e], int(j))
            if c is None:
                continue
            rows.append((gt[i], g2[i], c, hg[i], hc[i], (hg[i] - hg[k]) / (span * DT), hg[i] - gt[i], hg[k] - gt[k]))
        gt_, g2_, c, hg_, hc_, ctd, e1, e0 = np.array(rows).T
        print("%s span %.1f s n %d | corr with the label closing: head output %.3f, temporal difference %.3f (sd %.2f m/s) | corr of the head gap errors at t and t - span %.3f"
              % (dp.replace("\\", "/").split("/")[-1], span * DT, len(rows), np.corrcoef(hc_, c)[0, 1], np.corrcoef(ctd, c)[0, 1], ctd.std(), np.corrcoef(e1, e0)[0, 1]))
        print("    2.0 s forecast MAE m: head gap held %.3f | + head closing %.3f | + temporal difference %.3f | + half of it %.3f | + label closing %.3f"
              % (np.abs(hg_ - g2_).mean(), np.abs(hg_ + 2 * hc_ - g2_).mean(), np.abs(hg_ + 2 * ctd - g2_).mean(), np.abs(hg_ + ctd - g2_).mean(), np.abs(hg_ + 2 * c - g2_).mean()))
print("ZZ TD-CLOSING DONE")
