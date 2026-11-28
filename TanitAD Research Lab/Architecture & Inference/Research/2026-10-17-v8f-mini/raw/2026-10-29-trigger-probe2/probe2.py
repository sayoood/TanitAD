#!/usr/bin/env python3
"""probe2.py -- the probe stages for the new feature sources (PREREG_FROZEN.md 3, 5, 6, 8).  Resumable and time-boxed.
    python -I probe2.py --src Znn|Znl|V4|Vdiff --pop pstop|pmove --stage pca|lin|mlp|pair
src Znn / Znl : h_O with the donor nav (Znn, PRE-REGISTERED primary) / donor nav + limit (Znl, POST-FREEZE) -> scratch/Z{nn,nl}_<pop>.npy   [12000, 1024] fp32
src V4        : the 4 frames t-3..t concatenated (163,840 dims) from scratch/Vm{3,2,1} + the previous package's V_<pop>.npy
src Vdiff     : V_t - V_{t-3} (40,960 dims)
stage pca  : PCA-64 per OUTER fold on its training rows (+ one on all rows for the eval confirmation)  -> raw/pca2_<src>_<pop>.npz, scratch/pcacomp2_<src>_<pop>_all.npz   (per-fold files in scratch/, resumable)
stage lin  : linear arms <src>, <src>+E, controls C1 (shuffle) / C2 (episode-donor labels) / C3 (leak) on <src>+E  -> raw/res2_<src>_<pop>_linear.json, raw/oof2_...
stage mlp  : MLP arm <src>+E
stage pair : paired deltas against the banked E / V+E / Z+E (and Znn vs Z, V4 vs Vdiff)  -> raw/results2_<src>_<pop>.json
Primary metric = mean of per-fold AUCs ('fold'); pooled beside.  CUDA_VISIBLE_DEVICES=-1."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np

import common2 as C  # noqa: E402

L = C.L
K_PCA = 64
DIMS = {"Znn": 1024, "Znl": 1024, "V4": 4 * 40960, "Vdiff": 40960}
CHUNK = {"Znn": 500, "Znl": 500, "V4": 250, "Vdiff": 500}


def reader(src, pop):
    """read(rows) -> float32 [len(rows), D] without loading the whole matrix."""
    if src in ("Znn", "Znl"):
        X = np.load(f"{C.SCR}/{src}_{pop}.npy", mmap_mode="r")
        return (lambda r: np.asarray(X[r], np.float32)), X.shape
    Vt = np.load(f"{C.OLDSCR}/V_{pop}.npy", mmap_mode="r")
    Vm = {l: np.load(f"{C.SCR}/Vm{l}_{pop}.npy", mmap_mode="r") for l in (1, 2, 3)}
    if src == "Vdiff":
        return (lambda r: np.asarray(Vt[r], np.float32) - np.asarray(Vm[3][r], np.float32)), Vt.shape
    if src == "V4":
        return (lambda r: np.concatenate([np.asarray(Vm[3][r], np.float32), np.asarray(Vm[2][r], np.float32), np.asarray(Vm[1][r], np.float32), np.asarray(Vt[r], np.float32)], 1)), (Vt.shape[0], 4 * Vt.shape[1])
    raise KeyError(src)


def stage_pca(src, pop, budget_s=190.0):
    t0 = time.time()
    C.require_ram(3000 if src in ("V4", "Vdiff") else 1500, f"pca {src} {pop}")
    d = C.load_pop(pop)
    read, (n, D) = reader(src, pop)
    assert n == len(d["y"]) and D == DIMS[src], (n, D)
    for f in range(6):
        fp = f"{C.SCR}/pcafold2_{src}_{pop}_{f}.npz"
        if os.path.exists(fp):
            continue
        rows = np.flatnonzero(d["outer"] != f) if f < 5 else np.arange(n)
        p = L.StreamPCA(k=K_PCA, seed=0, chunk=CHUNK[src]).fit(read, rows, D)
        S = p.transform(read, np.arange(n))
        np.savez(fp, S=S, sv=p.S, mu=p.mu, comp=p.comp)
        print(f"pca {src} {pop} fold {f} rows {len(rows)} {time.time() - t0:.0f}s freeMB {C.free_mb()}", flush=True)
        if time.time() - t0 > budget_s and f < 5:
            print("time box: banked, re-invoke to continue", flush=True)
            return False
    parts = [np.load(f"{C.SCR}/pcafold2_{src}_{pop}_{f}.npz") for f in range(6)]
    np.savez(f"{C.RAW}/pca2_{src}_{pop}.npz", S=np.stack([p["S"] for p in parts]), sv=np.stack([p["sv"] for p in parts]))
    np.savez(f"{C.SCR}/pcacomp2_{src}_{pop}_all.npz", mu=parts[5]["mu"], comp=parts[5]["comp"], S=parts[5]["sv"])
    print(f"pca {src} {pop} complete {time.time() - t0:.0f}s", flush=True)
    return True


def arms_for(src, d, S, fam):
    E, y = d["E"], d["y"]
    Sf = lambda f: S[f].astype(np.float64)
    SE = lambda f: np.concatenate([S[f].astype(np.float64), E], 1)
    arms = {f"{src}+E": (SE, None)}
    if fam == "linear":
        arms = {src: (Sf, None), f"{src}+E": (SE, None)}
        arms[f"C1:shuffle:{src}+E"] = (SE, L.shuffle_labels(y, 1))
        arms[f"C3:leak:{src}+E"] = ((lambda f: np.concatenate([S[f].astype(np.float64), E, y[:, None].astype(np.float64)], 1)), None)
        if src in ("Znn", "V4"):
            arms[f"C2:donor:{src}+E"] = (SE, L.donor_labels(y, d["clip"], 2))
    return arms


def stage_fit(src, pop, fam):
    d = C.load_pop(pop)
    S = np.load(f"{C.RAW}/pca2_{src}_{pop}.npz")["S"]
    return C.run_arms(f"{src}_{pop}", d, fam, arms_for(src, d, S, fam))


def stage_pair(src, pop):
    d = C.load_pop(pop)
    y, clip, outer = d["y"], d["clip"], d["outer"]
    counts = L.boot_counts(clip)
    out = {"src": src, "pop": pop, "n": int(len(y)), "note": "'fold' = mean of per-fold AUCs (primary), 'pooled' = pooled OOF AUC; paired on the same 12,000 windows / folds / bootstrap as the banked arms", "arms": {}, "paired": {}}
    for fam in ("linear", "mlp"):
        bk = C.banked(pop, fam)
        if fam == "linear":
            assert np.array_equal(bk["y"], y) and np.array_equal(bk["clip"], clip)
        resf = f"{C.RAW}/res2_{src}_{pop}_{fam}.json"
        if not os.path.exists(resf):
            continue
        oof = dict(np.load(f"{C.RAW}/oof2_{src}_{pop}_{fam}.npz"))
        out["arms"][fam] = json.load(open(resf))["arms"]
        out["arms"][fam].update({f"banked:{k}": C.rec(bk[k], y, clip, outer, counts) for k in ("E", "V+E", "Z+E") if k in bk})
        P = {}
        a = oof.get(f"{src}+E")
        if a is None:
            out["paired"][fam] = P
            continue
        P[f"{src}+E_minus_E"] = C.paired(a, bk["E"], y, clip, outer, counts)
        P[f"{src}+E_minus_V+E"] = C.paired(a, bk["V+E"], y, clip, outer, counts)
        if src in ("Znn", "Znl"):
            P[f"Z+E_minus_{src}+E"] = C.paired(bk["Z+E"], a, y, clip, outer, counts)
        if src in ("V4", "Vdiff"):
            other = "Vdiff" if src == "V4" else "V4"
            of = f"{C.RAW}/oof2_{other}_{pop}_{fam}.npz"
            if os.path.exists(of) and f"{other}+E" in np.load(of).files:
                P[f"{src}+E_minus_{other}+E"] = C.paired(a, np.load(of)[f"{other}+E"], y, clip, outer, counts)
        if fam == "linear" and src in oof:
            P[f"{src}_minus_E"] = C.paired(oof[src], bk["E"], y, clip, outer, counts)
        out["paired"][fam] = P
    json.dump(out, open(f"{C.RAW}/results2_{src}_{pop}.json", "w"), indent=1, default=float)
    for fam, P in out["paired"].items():
        for k, v in P.items():
            print(src, pop, fam, k, f"fold {v['fold']['delta']:+.4f} [{v['fold']['lo']:+.4f},{v['fold']['hi']:+.4f}]  pooled {v['pooled']['delta']:+.4f} [{v['pooled']['lo']:+.4f},{v['pooled']['hi']:+.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, choices=tuple(DIMS))
    ap.add_argument("--pop", required=True, choices=C.POPS)
    ap.add_argument("--stage", required=True, choices=("pca", "lin", "mlp", "pair"))
    ap.add_argument("--budget-s", type=float, default=190.0)
    a = ap.parse_args()
    print("free MB", C.require_ram(3000 if (a.src in ("V4", "Vdiff") and a.stage == "pca") else 1000, f"probe2 {a.src} {a.pop} {a.stage}"), flush=True)      # only the V4 / Vdiff PCA readers are heavy (3 GB rule); every lin / mlp / pair stage reads 64-d PCA scores (< 0.4 GB RSS) and runs with a 1 GB floor
    if a.stage == "pca":
        stage_pca(a.src, a.pop, a.budget_s)
    elif a.stage == "pair":
        stage_pair(a.src, a.pop)
    else:
        ok = stage_fit(a.src, a.pop, "linear" if a.stage == "lin" else "mlp")
        print("complete" if ok else "INCOMPLETE")
