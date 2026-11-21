"""Paired, stratified comparison of two eval dumps on the SAME windows (ladder_score metrics, episode-cluster bootstrap,
both sampler seeds). Strata: ALL, GT turn (turnL|turnR), GT straight.
usage: python strata_pair.py <dirT> <rowT> <dirB> <rowB> [metric ...]     (d = T - B; METRICS orientation shown)"""
import sys
from pathlib import Path

import numpy as np

import ladder_score as LS

dT, rT, dB, rB = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
METRICS = sys.argv[5:] or ["cross_abs_6s", "ade_all", "yaw_mae_0_2s", "curv_mae_0_2s", "dir_correct_turn"]
print(f"T = {dT.name}:{rT}   B = {dB.name}:{rB}")
for s in LS.SEEDS:
    zt, zb = LS.load_pass(dT / f"{rT}_s{s}.npz"), LS.load_pass(dB / f"{rB}_s{s}.npz")
    if zt["_json"]["window_sha12_t_sha256"] != zb["_json"]["window_sha12_t_sha256"]:
        raise SystemExit("windows differ")
    wt, wb = LS.window_metrics(zt), LS.window_metrics(zb)
    gtc, _ = LS.gt_class(np.asarray(zb["gt"]), np.asarray(zb["gt_valid"]))
    gtc = np.asarray(gtc).astype(str)
    eps, inv = LS.episodes(zb["win_sha12"])
    W = LS.counts_of(LS.make_draws(len(eps)), len(eps))[:, inv]
    strata = {"ALL": np.ones(len(gtc), bool), "GT turn": np.isin(gtc, ["turnL", "turnR"]), "GT straight": gtc == "straight"}
    print(f"=== sampler seed {s}")
    for m in METRICS:
        at, mt = wt[m]
        ab, mb = wb[m]
        better = LS.METRICS[m][1]          # +1 = higher is better, -1 = lower is better (ladder_score's table)
        for nm, st in strata.items():
            k = st & mt.astype(bool) & mb.astype(bool)
            if k.sum() < 5:
                continue
            diff = at - ab
            num, den = (W[:, k] * diff[k]).sum(1), W[:, k].sum(1)
            bs = num[den > 0] / den[den > 0]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sep = "ns"
            if lo > 0 or hi < 0:
                good = (hi < 0) if better < 0 else (lo > 0)
                sep = "T BETTER" if good else "T WORSE"
            print(f"  {m:18s} {nm:12s} n={int(k.sum()):4d}  T {np.mean(at[k]):7.3f}  B {np.mean(ab[k]):7.3f}  "
                  f"d={np.mean(diff[k]):+7.3f} [{lo:+.3f},{hi:+.3f}] {sep}")
