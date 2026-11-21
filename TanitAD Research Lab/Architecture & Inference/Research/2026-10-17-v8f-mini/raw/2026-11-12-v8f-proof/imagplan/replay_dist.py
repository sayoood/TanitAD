#!/usr/bin/env python
"""replay_dist.py -- note ADD40_N4 diagnostic (written by the lead): is the weak least-squares response of the candidates at 2.0 s a property of most windows or of a few?

Per active window at tick 0: the imagined gap of the plan (d = 0) and of the candidates, beside the imagined gap under the logged actions (p4/p5 log).
Prints quantiles of the imagined gaps, the per-pair ratio (imagined gap change / distance behind the plan) as quantiles, and the least-squares slope with the windows
whose plan-based imagined gap at 2.0 s lies outside [-5, 120] m left out. Reads no unsafe rate.
"""
import argparse
import json

import numpy as np

Q = (1, 5, 25, 50, 75, 95, 99)


def q(x):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    return [round(float(v), 2) for v in np.percentile(x, Q)] if len(x) else []


def slope0(d, e):
    d, e = np.asarray(d, np.float64).ravel(), np.asarray(e, np.float64).ravel()
    ok = np.isfinite(d) & np.isfinite(e)
    return round(float((d[ok] * e[ok]).sum() / (e[ok] * e[ok]).sum()), 3) if ok.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p45", required=True)
    ap.add_argument("--imag-log", required=True)
    a = ap.parse_args()
    P = np.load(a.p45, allow_pickle=True)
    L = np.load(a.imag_log, allow_pickle=True)
    k0 = np.asarray(L["k"]).astype(int) == 0
    key0 = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(L["ci"])[k0], np.asarray(L["t0"])[k0]))}
    rows0 = np.array([key0[(int(c), int(t))] for c, t in zip(P["ci"], P["t0"])])
    act = np.asarray(L["active"]).astype(bool)[k0][rows0]
    cs = np.asarray(L["cand_set"], np.float64)
    i0 = int(np.flatnonzero(np.isclose(cs, 0.0))[0])
    gi = np.asarray(L["imag_raw"], np.float64)[k0][rows0][act]            # [n, C, 2]
    ah = np.asarray(L["a_h"], np.float64)[k0][rows0][act]
    gs = np.asarray(L["g_src"], np.float64)[k0][rows0][act]
    v0 = np.asarray(L["v0"], np.float64)[k0][rows0][act]
    nyl = np.asarray(L["n_yaw_lim"], np.float64)[k0][rows0][act]
    gl = np.asarray(P["imag_gap_logged"], np.float64)[act]
    go = np.asarray(P["imag_gap_logged_off"], np.float64)[act]
    print("percentiles", list(Q), "n active windows", int(act.sum()))
    print("present gap (head)            ", q(gs))
    for hi, hn in ((0, "1.0 s"), (1, "2.0 s")):
        print("imagined gap, logged actions  ", hn, q(gl[:, hi]))
        print("imagined gap, plan (d = 0)    ", hn, q(gi[:, i0, hi]))
        print("plan minus logged             ", hn, q(gi[:, i0, hi] - gl[:, hi]))
    da = ah[:, [i0], :] - ah
    dg = gi - gi[:, [i0], :]
    for hi, hn, lo, hi_ in ((0, "1.0 s", 0.25, 1.0), (1, "2.0 s", 1.0, 4.0)):
        u = (da[:, :, hi] >= lo) & (da[:, :, hi] <= hi_) & np.isfinite(dg[:, :, hi])
        u[:, i0] = False
        r = dg[:, :, hi][u] / da[:, :, hi][u]
        print("candidates", hn, "LS slope", slope0(dg[:, :, hi][u], da[:, :, hi][u]), "ratio quantiles", q(r), "share ratio in [0.5, 1.5]", round(float(((r >= 0.5) & (r <= 1.5)).mean()), 3))
        e = np.stack([-0.5 * d * (hi + 1.0) ** 2 for d in (-2.0, -1.0)])
        rT = (go[:, :2, hi] - gl[:, None, hi]) / e[None, :]
        print("logged+offset", hn, "LS slope", slope0(go[:, :2, hi] - gl[:, None, hi], np.broadcast_to(e, rT.shape)), "ratio quantiles", q(rT), "share ratio in [0.5, 1.5]",
              round(float(((rT >= 0.5) & (rT <= 1.5)).mean()), 3))
    hi = 1
    inr = (gi[:, i0, hi] >= -5.0) & (gi[:, i0, hi] <= 120.0)
    u = (da[:, :, hi] >= 1.0) & (da[:, :, hi] <= 4.0) & np.isfinite(dg[:, :, hi]) & inr[:, None]
    u[:, i0] = False
    print("2.0 s, windows with the plan's imagined gap inside [-5, 120] m:", int(inr.sum()), "of", len(inr), "LS slope", slope0(dg[:, :, hi][u], da[:, :, hi][u]))
    # by what the plan's imagined gap differs from the one under the logged actions
    dev = np.abs(gi[:, i0, hi] - gl[:, hi])
    for lo_, hi_ in ((0, 1), (1, 3), (3, 10), (10, 1e9)):
        m = (dev >= lo_) & (dev < hi_)
        u = (da[:, :, hi] >= 1.0) & (da[:, :, hi] <= 4.0) & np.isfinite(dg[:, :, hi]) & m[:, None]
        u[:, i0] = False
        r = dg[:, :, hi][u] / da[:, :, hi][u]
        print("  |plan - logged| at 2.0 s in [%g, %g) m: windows %d, LS slope %s, median ratio %s, median v0 %.1f, mean yaw-limited steps %.1f" % (
            lo_, hi_, int(m.sum()), slope0(dg[:, :, hi][u], da[:, :, hi][u]), (round(float(np.median(r)), 3) if len(r) else None), float(np.median(v0[m])) if m.any() else float("nan"),
            float(nyl[m].mean()) if m.any() else float("nan")))


if __name__ == "__main__":
    main()
