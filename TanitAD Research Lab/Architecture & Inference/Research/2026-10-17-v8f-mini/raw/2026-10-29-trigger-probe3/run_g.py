#!/usr/bin/env python3
"""run_g.py -- the probe stages of PREREG 3 and 6 for the G4 / G8 readouts (one population, one family).  Resumable: every arm is banked as it finishes (raw/oofg_*, raw/resg_*).
    python -I run_g.py --pop pstop|pmove --family linear|mlp [--mutate] [--only NAME[,NAME]]
Priority order (PREREG 4 then 6):  G4+E, G8+E, G4, G8, C1 shuffle, C3 leak, C2 donor (seed 2).  MLP: G4+E, G8+E, G4, G8, then the MLP C1 shuffles LAST.
E (and probe 1's V+E) are NOT refitted: their out-of-fold scores are probe 1's banked ones (g_lib.banked).
FIX of the probe-1 trap (run_probe.py:108 `if src == "V"` skipped the controls for any other source name): the controls here are built for every source in ``SRC`` (G4, G8) by the same loop.
--mutate (C5b): the G8 features come from raw/pcaG_mut_<pop> (nearest-neighbour upsample of G4); arms are named G8mut+E; only the P-STOP linear G8mut+E arm is required.
CUDA_VISIBLE_DEVICES=-1."""
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
SRC = ("G4", "G8")


def feats(pop, mut=False):
    S = {"G4": np.load(f"{G.SCR}/S4_{pop}.npy", mmap_mode="r"), "G8": np.load(f"{G.SCR}/{'mut_' if mut else ''}S8_{pop}.npy", mmap_mode="r")}
    assert S["G8"].shape[2] == 2560 and S["G4"].shape[2] == 640, (S["G8"].shape, S["G4"].shape)          # reachability: the G8 arm must be 160 x 16 wide, the G4 arm 40 x 16
    return S


def build_arms(d, S, fam, mut=False):
    E, y, clip = d["E"], d["y"], d["clip"]

    def fn(src, with_e, leak=None):
        def f(fold):
            X = np.asarray(S[src][fold], np.float64)
            if with_e:
                X = np.concatenate([X, E], 1)
            if leak is not None:
                X = np.concatenate([X, leak[:, None].astype(np.float64)], 1)
            return X
        return f

    arms = {}
    if mut:
        arms["G8mut+E"] = (fn("G8", True), None)
        return arms
    for src in SRC:
        arms[f"{src}+E"] = (fn(src, True), None)
    for src in SRC:
        arms[src] = (fn(src, False), None)
    if fam == "linear":
        ys, yd = L.shuffle_labels(y, 1), L.donor_labels(y, clip, 2)
        for src in SRC[::-1]:                                                        # C1 label shuffle (seed 1)
            arms[f"C1:shuffle:{src}+E"] = (fn(src, True), ys)
        for src in SRC[::-1]:                                                        # C3 deliberate leak: the GT label appended
            arms[f"C3:leak:{src}+E"] = (fn(src, True, leak=y), None)
        for src in SRC[::-1]:                                                        # C2 episode-shuffled (donor) labels (seed 2)
            arms[f"C2:donor:{src}+E"] = (fn(src, True), yd)
        for seed in (3, 4, 5, 6):                                                    # conditional C2 band (PREREG 6): run only with --c2-band
            for src in SRC[::-1]:
                arms[f"C2:donor{seed}:{src}+E"] = (fn(src, True), L.donor_labels(y, clip, seed))
    else:
        ys = L.shuffle_labels(y, 1)
        for src in SRC[::-1]:
            arms[f"C1:shuffle:{src}+E"] = (fn(src, True), ys)
    return arms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, choices=G.POPS)
    ap.add_argument("--family", required=True, choices=("linear", "mlp"))
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--c2-band", action="store_true", help="also run the conditional C2 seeds 3-6 (only for arms whose seed-2 CI misses 0.5)")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    d = G.load_pop(a.pop)
    bk = G.banked(a.pop, "linear")
    assert np.array_equal(bk["y"], d["y"]) and np.array_equal(bk["clip"], d["clip"]), "row order differs from probe 1's banked OOF"
    S = feats(a.pop, a.mutate)
    arms = build_arms(d, S, a.family, a.mutate)
    if not a.c2_band:
        arms = {k: v for k, v in arms.items() if not k.startswith("C2:donor") or k.startswith("C2:donor:")}
    if a.only:
        arms = {k: v for k, v in arms.items() if k in a.only.split(",")}
    tag = f"{a.pop}_mut" if a.mutate else a.pop
    G.run_arms(tag, d, a.family, arms)
    print("stage done", a.pop, a.family, flush=True)
