#!/usr/bin/env python3
"""c2_band.py -- POST-FREEZE (raw/POSTFREEZE_LOG.md): the frozen C2 (episode-shuffled labels, seed 2) failed its 'CI covers 0.5' clause on Z_nn+E / P-STOP (fold-mean 0.454 [0.418, 0.491]).  The frozen reading stands (FAILED-CONTROL);
this script measures the NULL BAND of the same pipeline instead of a single draw: C2 repeated with donor seeds 3..12 on E and on each new arm, fold-mean AUC of the full pipeline (PCA already fixed per fold, C by inner CV).
The true arm is then read against its own band.  The mechanism (hypothesis, tested by the band): the sample is class-balanced with a fixed total of positives, so permuting labels across clips makes the folds' prevalences anti-correlated,
and an arm with enough clip-identifying capacity (Z, the speed-limit one-hot) fits clip-level noise in the training folds and is anti-correlated on the held-out fold -> a below-chance null.  python -I c2_band.py --pop pstop|pmove [--srcs E Znn V4 Vdiff]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np

import common2 as C  # noqa: E402

L = C.L
SEEDS = tuple(range(3, 13))


def feat(src, d, pop):
    if src == "E":
        return lambda f: d["E"]
    if src == "V+E":
        S = np.load(f"{C.OLDRAW}/pca_V_{pop}.npz")["S"]
    elif src == "Z+E":
        S = np.load(f"{C.OLDRAW}/pca_Z_{pop}.npz")["S"]
    else:
        S = np.load(f"{C.RAW}/pca2_{src}_{pop}.npz")["S"]
    return lambda f: np.concatenate([S[f].astype(np.float64), d["E"]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, choices=C.POPS)
    ap.add_argument("--srcs", nargs="+", default=["E", "V+E", "Z+E", "Znn", "Znl"])
    a = ap.parse_args()
    d = C.load_pop(a.pop)
    y, clip, outer, inner = d["y"], d["clip"], d["outer"], d["inner"]
    path = f"{C.RAW}/c2_band_{a.pop}.json"
    out = json.load(open(path)) if os.path.exists(path) else {"pop": a.pop, "seeds": list(SEEDS), "arms": {}}
    for src in a.srcs:
        if src in out["arms"]:
            continue
        if src not in ("E", "V+E", "Z+E") and not os.path.exists(f"{C.RAW}/pca2_{src}_{a.pop}.npz"):
            print("SKIP", src, "(PCA not ready)")
            continue
        fn = feat(src, d, a.pop)
        vals = []
        for sd in SEEDS:
            yd = L.donor_labels(y, clip, sd)
            o, _ = L.cv_oof(fn, yd, outer, inner, "linear")
            vals.append(L.FoldWAuc(o, yd, outer).auc())
        out["arms"][src if src in ("E", "V+E", "Z+E") else f"{src}+E"] = {"aucs": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "min": float(np.min(vals)), "max": float(np.max(vals))}
        json.dump(out, open(path, "w"), indent=1)
        print(a.pop, src, "C2 null band fold-mean AUC: mean %.4f sd %.4f [%.4f, %.4f]" % (np.mean(vals), np.std(vals, ddof=1), np.min(vals), np.max(vals)), flush=True)


if __name__ == "__main__":
    main()
