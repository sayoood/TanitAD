#!/usr/bin/env python3
"""SHUF permutation null for PROBE-N1D-FEAT (beside; moves no committed band). Reuses n1dwprobe_probe.fit_one / folds / labels unchanged.
Per (arm, probe, fset) the SHUF fit is repeated with NPERM permutation seeds (2..NPERM+1; the committed cells used seeds 0 and 1, so the null draws are new).
The probe seed does not change a fit (TREE-SEED-IDENTITY, measured in the probe), so the committed seed-0 and seed-1 cells of one (arm,probe,fset) are placed against the same null: 40 cells, 20 nulls."""
import argparse, json, os, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
PD = "/root/strat1/n1dwprobe/code"
sys.path.insert(0, PD)
import numpy as np
import n1dwprobe_probe as P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/root/strat1/n1dwprobe/out/feat1")
    ap.add_argument("--probe-result", default="/root/strat1/n1dwprobe/out/result/PROBE_RESULT.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nperm", type=int, default=50)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    t_all = time.time()
    paths = {arm: f"{a.feats}/feat_{arm}.npz" for arm in P.ARMS}
    F = {arm: P.load_feat(p) for arm, p in paths.items()}
    ref = F[P.ARMS[0]]
    eid, g = ref["ci"].astype(np.int64), ref["g"]
    y = P.kin_labels(g)
    fold = P.fold_ids(eid)
    N = len(y)
    pr = json.load(open(a.probe_result))
    tasks = [(arm, fs, probe, s, True, k) for arm in P.ARMS for probe in P.PROBES for fs in P.FSETS for s in range(2, a.nperm + 2) for k in range(P.NFOLD)]
    w = {"F-PLAN": 0, "F-PLAN+TEXT": 1, "F-NAV": 2, "F-VIS": 3, "F-TEXT": 4}
    tasks.sort(key=lambda t: (w[t[1]], t[2] != "tree"))
    print(len(tasks), "fits", a.workers, "workers", flush=True)
    pred = {}
    n = 0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=P._init, initargs=(paths, y, fold)) as ex:
        for task, idx, p, dt in ex.map(P.fit_one, tasks, chunksize=1):
            pred.setdefault(task[:5], np.full(N, -1, np.int64))[idx] = p
            n += 1
            if n % 100 == 0 or n == len(tasks):
                print(f"fits {n}/{len(tasks)} {time.time()-t_all:.0f}s", flush=True)
    assert all((v >= 0).all() for v in pred.values())
    null = {}
    for (arm, fs, probe, s, _), v in pred.items():
        null.setdefault(f"{arm}|{probe}|{fs}", {})[s] = float((v == y).mean())
    cells, below, above = {}, 0, 0
    for arm in P.ARMS:
        for probe in P.PROBES:
            for seed in P.SEEDS:
                for fs in P.FSETS:
                    d = np.array([null[f"{arm}|{probe}|{fs}"][s] for s in sorted(null[f"{arm}|{probe}|{fs}"])])
                    c = pr["controls"]["SHUF"]["cells"][f"{arm}|{probe}|seed{seed}|{fs}"]["accuracy"]
                    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                    cells[f"{arm}|{probe}|seed{seed}|{fs}"] = {"committed_shuf": c, "null_mean": float(d.mean()), "null_sd": float(d.std(ddof=1)), "null_p2.5": lo, "null_p97.5": hi,
                        "null_min": float(d.min()), "null_max": float(d.max()), "n_perm": int(len(d)), "inside_95": bool(lo <= c <= hi),
                        "rank_frac_null_le_committed": float((d <= c).mean()), "committed_minus_null_mean": float(c - d.mean())}
    # class-prior predictor, fold-wise (train-fold majority class, scored on the held-out fold)
    cp = {}
    for k in range(P.NFOLD):
        pri = np.bincount(y[fold != k], minlength=3) / (fold != k).sum()
        m = int(pri.argmax())
        cp[k] = {"train_prior": pri.tolist(), "predicted_class": m, "test_acc": float((y[fold == k] == m).mean()), "n_test": int((fold == k).sum())}
    pooled = float(sum(v["test_acc"] * v["n_test"] for v in cp.values()) / N)
    res = {"what": "SHUF permutation null for PROBE-N1D-FEAT; beside, moves no committed band", "n_perm_per_null": a.nperm, "perm_seeds": [2, a.nperm + 1],
           "perm_rule": "labels of the TRAINING fold permuted with RandomState(1000*s+k+17) (probe's own rule); committed cells used s=0,1",
           "n_nulls": len(null), "n_cells": len(cells), "cells": cells,
           "cells_outside_95": [k for k, v in cells.items() if not v["inside_95"]],
           "foldwise_class_prior_predictor": {"folds": cp, "pooled_accuracy": pooled, "majority_share_all": float(np.bincount(y, minlength=3).max() / N)},
           "null_accuracy_all": {k: [v[s] for s in sorted(v)] for k, v in null.items()}, "elapsed_s": time.time() - t_all}
    json.dump(res, open(a.out, "w"), indent=1)
    print("ZZ N1DWT0T1 DONE ZZ" if False else "shuf wrote", a.out, flush=True)

if __name__ == "__main__":
    main()
