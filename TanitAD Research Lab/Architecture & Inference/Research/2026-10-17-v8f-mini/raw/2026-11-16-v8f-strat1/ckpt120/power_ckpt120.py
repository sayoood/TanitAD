"""power_ckpt120.py -- design-time power of the S_CKPT constant contrasts on the eval-141 grid (labels only, no model output).

A planted follower is correct on each scorable window with probability p (independent per window, or episode-clustered:
per-episode accuracy ~ Beta with mean p and concentration 4). For each p it reports P(separated) over N_SIM simulated
followers of
  RAW  : mean(follower) - mean(best constant)  (paired episode-cluster bootstrap, the STRAT1 form)
  BAL  : balanced level (mean of the per-side levels) - 0.5  (a constant reads exactly 0.5; episode-cluster bootstrap)
on three window sets: the 120 m scorable set, its nav-informed stratum (token on), and the 80 m set (the STRAT1 deciding set).
Evidence class of the output: ESTIMATED (simulation on MEASURED window structure).  B 500 per simulation (design only;
the read itself uses B 2,000).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

COMPLETE_M = 20.0


def boot_sep(diff_fn, eid, rng, B=500):
    uniq = np.unique(eid)
    idx = {u: np.nonzero(eid == u)[0] for u in uniq}
    pt = diff_fn(np.arange(len(eid)))
    d = np.empty(B)
    for b in range(B):
        sel = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        d[b] = diff_fn(sel)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return pt, bool(lo > 0 or hi < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sim", type=int, default=200)
    a = ap.parse_args()
    z = np.load(a.npz)
    arcs = list(z["arcs"])
    side, d, on, ok, eid = z["side"], z["d_text"], z["token_on"], z["rc_ok"], z["eid"]
    sets = {}
    for name, L, strat in (("arc120_all", 120.0, None), ("arc120_nav_informed", 120.0, True), ("arc120_nav_silent", 120.0, False), ("arc80_all", 80.0, None)):
        k = arcs.index(L)
        sc = ok[:, k] & (side != 0) & np.isfinite(d) & (d + COMPLETE_M <= L)
        if strat is not None:
            sc = sc & (on == strat)
        sets[name] = sc
    rep = {"tool": "power_ckpt120.py", "evidence_class": "ESTIMATED (simulation on the MEASURED eval-141 window and episode structure)",
           "n_sim": a.n_sim, "B_per_sim": 500, "sets": {}}
    for name, sc in sets.items():
        s, e = side[sc], eid[sc]
        nL, nR = int((s == 1).sum()), int((s == -1).sum())
        maj = 1 if nL >= nR else -1
        bc = (s == maj).astype(float)
        out = {"n_windows": int(sc.sum()), "n_episodes": int(len(np.unique(e))), "n_left": nL, "n_right": nR, "best_constant": round(float(bc.mean()), 4), "p": {}}
        for p in (0.80, 0.85, 0.90, 0.95, 1.00):
            res = {}
            for mode in ("independent", "episode_clustered"):
                rng = np.random.default_rng(12345)
                sr = sb = 0
                for _ in range(a.n_sim):
                    if mode == "independent" or p >= 1.0:
                        f = (rng.random(len(s)) < p).astype(float)
                    else:
                        ue = np.unique(e)
                        pe = dict(zip(ue, rng.beta(4 * p, 4 * (1 - p), len(ue))))
                        f = (rng.random(len(s)) < np.array([pe[x] for x in e])).astype(float)

                    def raw(sel, f=f):
                        return f[sel].mean() - bc[sel].mean()

                    def bal(sel, f=f):
                        ss = s[sel]
                        m = [f[sel][ss == q].mean() for q in (1, -1) if (ss == q).any()]
                        return float(np.mean(m)) - 0.5 if len(m) == 2 else np.nan

                    sr += boot_sep(raw, e, rng)[1]
                    sb += boot_sep(bal, e, rng)[1]
                res[mode] = {"P_sep_raw": round(sr / a.n_sim, 3), "P_sep_balanced": round(sb / a.n_sim, 3)}
            out["p"][str(p)] = res
        rep["sets"][name] = out
        print(name, json.dumps(out), flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
