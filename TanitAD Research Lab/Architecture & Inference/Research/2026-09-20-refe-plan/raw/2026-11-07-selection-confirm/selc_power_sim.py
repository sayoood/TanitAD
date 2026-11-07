#!/usr/bin/env python3
"""Heavy-tail check of POWER.md: simulate the confirmation read on the ACTUAL fresh allocation (75 logs, their token counts) with per-token deltas
resampled from C3's measured per-token deltas, instead of the normal approximation. CPU, banked data only. Writes power_sim.json.

PDMS: d_i ~ empirical random-stratum C3 deltas (529, all logs) [+ shift], + a log effect N(0, s2b) (s2b = the held-out estimate, the larger one).
CRUISE slow clause: n_cruise_j ~ Binomial(n_j, 0.119) per log (C3 random-stratum CRUISE share); slow diffs resampled from the 63 random-stratum CRUISE
values, then recentred to a target mean (pooled -0.238, TRAIN -0.067, held-out -0.394). Bootstrap = the prereg's log-cluster percentile CI (1,000 draws here).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
YSEL = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
sys.path.insert(0, YSEL)
sys.path.insert(0, HERE)
import sel_common as sc  # noqa: E402
import selc_rule as R  # noqa: E402

W_R = 11617 / 12146
NSIM, NB = 400, 1000


def boot_ci(vals, li, nl, rng):
    s = np.bincount(li, weights=vals, minlength=nl)
    c = np.bincount(li, minlength=nl).astype(float)
    dr = rng.integers(0, nl, (NB, nl))
    S, C = s[dr].sum(1), c[dr].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S / C
    r = r[np.isfinite(r)]
    return s.sum() / c.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def main():
    D = sc.load()
    r = np.arange(len(D["token"]))
    pr = R.rule_vec(D["logits"], D["v4_all"], D["v0"], D["ax0"])
    rnd = D["stratum"] == "random"
    d = 100 * (D["pdms"][r, pr] - D["pdms"][r, D["pick"]])[rnd]
    ratio_b = D["v4_all"][r, D["pick"]] / np.maximum(D["v0"], 1e-6)
    ratio_r = D["v4_all"][r, pr] / np.maximum(D["v0"], 1e-6)
    m = rnd & D["cruise"]
    ds = ((ratio_r < 0.85).astype(float) - (ratio_b < 0.85).astype(float))[m]
    doc = json.load(open(os.path.join(HERE, "tokens_confirm.json")))
    _, nj = np.unique([doc["token_log"][t] for t in doc["tokens"]], return_counts=True)
    pw = json.load(open(os.path.join(HERE, "power.json")))
    s2b = pw["variance_components"]["random_heldout"]["s2b"]
    s2b_slow = pw["cruise_slow_nonzeroed"]["random_all"]["vc"]["s2b"]
    rng = np.random.default_rng(20261107)
    nl = len(nj)
    li = np.repeat(np.arange(nl), nj)
    out = {"n_tokens": int(nj.sum()), "n_logs": int(nl), "nsim": NSIM, "nboot": NB, "s2b_pdms": s2b, "s2b_slow": s2b_slow, "pdms": {}, "slow": {}}
    for name, shift in (("empirical_mean_as_measured(+%.3f)" % d.mean(), 0.0), ("true_0", -d.mean()), ("true_heldout_+0.317", 0.3167 - d.mean())):
        ok, los, hws = 0, [], []
        for _ in range(NSIM):
            b = rng.normal(0, np.sqrt(s2b), nl)
            v = rng.choice(d, nj.sum()) + shift + b[li]
            pt, lo, hi = boot_ci(v, li, nl, rng)
            ok += int(W_R * lo > -0.5)
            los.append(W_R * lo)
            hws.append(W_R * (hi - lo) / 2)
        out["pdms"][name] = {"power_lowerCI_gt_-0.5": ok / NSIM, "median_lower_mix": float(np.median(los)), "median_halfwidth_mix": float(np.median(hws))}
    for name, target in (("pooled_-0.238", -0.238), ("train_-0.067", -0.0667), ("heldout_-0.394", -0.394)):
        ok, pts = 0, []
        for _ in range(NSIM):
            nc = rng.binomial(nj, 0.119)
            lic = np.repeat(np.arange(nl), nc)
            b = rng.normal(0, np.sqrt(s2b_slow), nl)
            v = rng.choice(ds, nc.sum()) - ds.mean() + target + b[lic]
            pt, lo, hi = boot_ci(v, lic, nl, rng)
            ok += int(pt <= -0.10 and hi < 0)
            pts.append(pt)
        out["slow"][name] = {"power_cond_ii": ok / NSIM, "median_n_cruise": int(np.median(rng.binomial(nj.sum(), 0.119, 100))), "median_point": float(np.median(pts))}
    json.dump(out, open(os.path.join(HERE, "power_sim.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
