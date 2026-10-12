"""POST-HOC lever ranking (not a verdict): WHERE is V-R8's lateral gap to V0? Strata by the fed nav command (V0 gets
v7's per-clip nav, V-R8 gets v9's per-frame nav: `--r8-nav-from-v9`) and by the GT plan's class (ladder_score.gt_class).
Paired over the SAME windows, episode-cluster bootstrap (ladder_score's draws), both sampler seeds, batch-4 dev-box dumps.
usage: python nav_strata.py <eval_b4 dir> [metric ...]"""
import sys
from pathlib import Path

import numpy as np

import ladder_score as LS

B = Path(sys.argv[1])
METRICS = sys.argv[2:] or ["cross_abs_6s", "ade_all", "yaw_mae_0_2s", "dir_correct_turn"]
NAV = {0: "follow", 1: "left", 2: "right"}


def load(arm, s):
    z = LS.load_pass(B / arm / f"base_s{s}.npz")
    return z, LS.window_metrics(z)


for s in LS.SEEDS:
    z0, w0 = load("V0", s)
    z8, w8 = load("V-R8", s)
    zd, wd = load("V-R8d", s)
    n7, n9 = np.asarray(z0["nav"]).astype(int), np.asarray(z8["nav"]).astype(int)
    gtc, _th = LS.gt_class(np.asarray(z0["gt"]), np.asarray(z0["gt_valid"]))
    gtc = np.asarray(gtc).astype(str)
    eps, inv = LS.episodes(z0["win_sha12"])
    W = LS.counts_of(LS.make_draws(len(eps)), len(eps))[:, inv]      # (B, n_windows) episode-cluster weights
    strata = {
        "ALL": np.ones(len(n7), bool),
        "nav agree": n7 == n9,
        "nav DISAGREE": n7 != n9,
        "v7 turn -> v9 follow": (n7 != 0) & (n9 == 0),
        "v7 follow -> v9 turn": (n7 == 0) & (n9 != 0),
    }
    for c in sorted(set(gtc.tolist())):
        strata[f"GT class {c}"] = gtc == c
        strata[f"GT class {c} & DISAGREE"] = (gtc == c) & (n7 != n9)
    print(f"=== sampler seed {s}: nav v7 {np.bincount(n7, minlength=3)} v9 {np.bincount(n9, minlength=3)}; "
          f"GT classes {dict(zip(*np.unique(gtc, return_counts=True)))}")
    for m in METRICS:
        print(f"  -- {m}  (V0 / V-R8 / V-R8d mean; d = V-R8 - V0 with 95% episode-cluster CI; share of the ALL-window "
              f"gap sum)")
        a0, m0 = w0[m]
        a8, m8 = w8[m]
        ad, md = wd[m]
        both = m0.astype(bool) & m8.astype(bool) & md.astype(bool)
        tot_gap = np.nansum((a8 - a0)[both])
        for nm, st in strata.items():
            k = both & st
            if k.sum() < 5:
                continue
            diff = np.where(k, a8 - a0, 0.0)
            # paired episode-cluster bootstrap of the mean difference over the stratum's windows
            num, den = (W[:, k] * diff[k]).sum(1), W[:, k].sum(1)
            bs = num[den > 0] / den[den > 0]
            lo, hi = np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
            diff = a8 - a0
            share = np.nansum(diff[k]) / tot_gap if abs(tot_gap) > 1e-12 else np.nan
            print(f"     {nm:28s} n={int(k.sum()):4d}  {np.mean(a0[k]):7.3f} {np.mean(a8[k]):7.3f} {np.mean(ad[k]):7.3f}"
                  f"   d={np.mean(diff[k]):+7.3f} [{lo:+.3f},{hi:+.3f}]  share={share:5.2f}")
