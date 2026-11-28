#!/usr/bin/env python3
"""postfreeze_g.py -- POST-FREEZE sensitivity arms (NOT in PREREG_FROZEN.md; clearly marked in RESULTS; they CANNOT move the verdict).  Motivation: in the frozen run the inner CV picked the SMALLEST C of the grid in most folds
(the optimum sits on the grid edge), and one C regularises the 8 ego numbers together with the 640 / 2,560 image features.  Cheap questions this answers (linear, TRAIN, same folds / resamples):
  PF-A  C grid extended downward {1e-6 .. 1}                                  : arms G4+E, G8+E          (does the edge matter?)
  PF-B  equal width: the first 4 PCA components of every 8x20 token (640 features) vs the 16-component G4 (640 features)   : arms G8k4+E, G4k4+E
  PF-D  two-stage low-rank readout (probe 1's idea at token level): PCA-64 of the whole 640 / 2,560-d token-PCA feature vector (centred, not scaled, fitted on the outer-train rows), + E : arms G4w64+E, G8w64+E
  PF-C  E nearly unpenalised: the 8 ego columns replicated 64x (the L2 on their total weight falls 64x)                    : arms G4+E*64, G8+E*64
All use the extended grid.   python -I postfreeze_g.py --pop pstop|pmove     banks raw/oofg_<pop>_pf_linear.npz / raw/resg_<pop>_pf_linear.json (resumable)."""
from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import g_lib as G  # noqa: E402
import numpy as np

L = G.L
PF_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def build(d, S):
    E = d["E"]

    def fn(src, k=16, erep=1):
        def f(fold):
            X = np.asarray(S[src][fold], np.float64)
            if k < 16:
                X = X.reshape(len(X), -1, 16)[:, :, :k].reshape(len(X), -1)
            return np.concatenate([X] + [E] * erep, 1)
        return f

    def w64(src):
        def f(fold):
            X = np.asarray(S[src][fold], np.float32)
            tr = d["outer"] != fold
            mu = X[tr].mean(0)
            Xc = X[tr] - mu
            w, V = np.linalg.eigh((Xc.T @ Xc).astype(np.float64))
            comp = V[:, np.argsort(w)[::-1][:64]].astype(np.float32)
            return np.concatenate([((X - mu) @ comp).astype(np.float64), E], 1)
        return f

    return {"PF:G4w64+E": (w64("G4"), None), "PF:G8w64+E": (w64("G8"), None), "PF:G4+E": (fn("G4"), None), "PF:G8+E": (fn("G8"), None), "PF:G8k4+E": (fn("G8", 4), None), "PF:G4k4+E": (fn("G4", 4), None),
            "PF:G4+Ex64": (fn("G4", 16, 64), None), "PF:G8+Ex64": (fn("G8", 16, 64), None)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, choices=G.POPS)
    a = ap.parse_args()
    L.C_GRID = PF_GRID
    d = G.load_pop(a.pop)
    S = {"G4": np.load(f"{G.SCR}/S4_{a.pop}.npy", mmap_mode="r"), "G8": np.load(f"{G.SCR}/S8_{a.pop}.npy", mmap_mode="r")}
    G.run_arms(f"{a.pop}_pf", d, "linear", build(d, S))
    print("stage done", a.pop, flush=True)
