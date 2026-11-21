#!/usr/bin/env python3
"""run_probe.py -- the probe stages (PREREG_FROZEN.md 4-5, 8).  Each stage banks to raw/ and is resumable.
    python -I run_probe.py --pop pstop|pmove --src V|Z --stage pca|lin|mlp
src V = the frozen pool4x10 features (scratch/V_<pop>.npy, [n, 40960] fp16); src Z = h_O of the DAG-OACC s0 checkpoint (scratch/Z_<pop>.npy, [n, 1024] fp32).
stage pca : PCA-64 fitted per OUTER fold on the training rows (+ one on all rows for the eval confirmation) -> raw/pca_<src>_<pop>.npz
stage lin : linear probes (E, E_H, src, src+E, GO24 arms, controls C1-C3 for src V) -> raw/res_<pop>_<src>_linear.json + raw/oof_<pop>_<src>_linear.npz
stage mlp : the same arms with the 1x64 MLP (E, src, src+E) -> raw/res_<pop>_<src>_mlp.json
Evidence class MEASURED; estimator = episode-cluster bootstrap B=2000 seed 0 (blind to training variance: H-ESTIM-SEED-1).  CUDA_VISIBLE_DEVICES=-1."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW, SCR = f"{HERE}/raw", f"{HERE}/scratch"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402

K_PCA = 64


def free_mb():
    import subprocess
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def load_pop(pop):
    A = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    idx = np.load(f"{RAW}/sample_{pop}.npz")["idx"]
    clip = A["clip_i"][idx].astype(np.int64)
    sha = A["episode_sha12"][clip]
    d = {"idx": idx, "y": A["go"][idx].astype(int), "y24": A["go24"][idx].astype(int), "clip": clip, "sha": sha, "outer": L.outer_fold(sha), "inner": L.inner_fold(sha),
         "E": A["E"][idx].astype(np.float64), "EH": A["EH"][idx].astype(np.float64), "v0": A["v0"][idx], "t": A["t"][idx]}
    for k in ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart"):
        d[k] = A[k][idx]
    return d


def src_reader(pop, src):
    p = f"{SCR}/{src}_{pop}.npy"
    X = np.load(p, mmap_mode="r")
    return (lambda r: np.asarray(X[r], np.float32)), X.shape


def stage_pca(pop, src):
    d = load_pop(pop)
    read, (n, D) = src_reader(pop, src)
    assert n == len(d["y"])
    S = np.zeros((6, n, K_PCA), np.float32)
    sv = []
    t0 = time.time()
    for f in range(6):
        rows = np.flatnonzero(d["outer"] != f) if f < 5 else np.arange(n)
        p = L.StreamPCA(k=K_PCA, seed=0).fit(read, rows, D)
        S[f] = p.transform(read, np.arange(n))
        sv.append(p.S)
        if f == 5:
            np.savez(f"{SCR}/pcacomp_{src}_{pop}_all.npz", mu=p.mu, comp=p.comp, S=p.S)
        print(f"pca {pop} {src} fold {f} rows {len(rows)} {time.time() - t0:.0f}s freeMB {free_mb()}", flush=True)
    np.savez(f"{RAW}/pca_{src}_{pop}.npz", S=S, sv=np.array(sv))


def arm_feats(d, S, name, src):
    E, EH = d["E"], d["EH"]
    if name == "E":
        return lambda f: E
    if name == "E_H":
        return lambda f: EH
    if name == src:
        return lambda f: S[f].astype(np.float64)
    if name == f"{src}+E":
        return lambda f: np.concatenate([S[f].astype(np.float64), E], 1)
    if name == f"{src}+E_H":
        return lambda f: np.concatenate([S[f].astype(np.float64), EH], 1)
    raise KeyError(name)


def stage_lin(pop, src):
    d = load_pop(pop)
    S = np.load(f"{RAW}/pca_{src}_{pop}.npz")["S"]
    y, y24, clip, outer, inner = d["y"], d["y24"], d["clip"], d["outer"], d["inner"]
    counts = L.boot_counts(clip)
    oof, res = {}, {"pop": pop, "src": src, "family": "linear", "n": int(len(y)), "n_go": int(y.sum()), "n_clips": int(len(np.unique(clip))), "free_mb_start": free_mb()}
    t0 = time.time()

    def run(name, X_fn, labels, tag=None):
        key = tag or name
        o, hps = L.cv_oof(X_fn, labels, outer, inner, "linear")
        oof[key] = o
        res.setdefault("arms", {})[key] = {**L.auc_ci(o, labels, clip, counts=counts), "foldmean": L.auc_ci_fold(o, labels, clip, outer, counts=counts), "hp_per_fold": hps}
        a_ = res["arms"][key]
        print(f"{pop} {src} lin {key} pooled {a_['auc']:.4f} [{a_['lo']:.4f},{a_['hi']:.4f}] foldmean {a_['foldmean']['auc']:.4f} [{a_['foldmean']['lo']:.4f},{a_['foldmean']['hi']:.4f}] {time.time() - t0:.0f}s", flush=True)

    for nm in ("E", "E_H", src, f"{src}+E", f"{src}+E_H"):
        run(nm, arm_feats(d, S, nm, src), y)
    for nm in ("E", src, f"{src}+E"):                                   # secondary label GO24
        run(nm, arm_feats(d, S, nm, src), y24, tag=f"GO24:{nm}")
    if src == "V":                                                      # controls (PREREG 8)
        ys = L.shuffle_labels(y, 1)
        run("C1:shuffle:V+E", arm_feats(d, S, "V+E", src), ys, tag="C1:shuffle:V+E")
        run("C1:shuffle:E", arm_feats(d, S, "E", src), ys, tag="C1:shuffle:E")
        yd = L.donor_labels(y, clip, 2)
        run("C2:donor:E", arm_feats(d, S, "E", src), yd, tag="C2:donor:E")
        for nm in ("E", "V+E"):
            base = arm_feats(d, S, nm, src)
            run(nm, (lambda f, b=base: np.concatenate([b(f), y[:, None].astype(np.float64)], 1)), y, tag=f"C3:leak:{nm}")
    res["paired"] = {}
    P = res["paired"]
    def both(a, b, lab):
        return {**L.dauc_ci(a, b, lab, clip, counts=counts), "foldmean": L.dauc_ci_fold(a, b, lab, clip, outer, counts=counts)}
    P[f"{src}+E_minus_E"] = both(oof[f"{src}+E"], oof["E"], y)
    P[f"{src}+E_minus_E_H"] = both(oof[f"{src}+E"], oof["E_H"], y)
    P[f"{src}_minus_E"] = both(oof[src], oof["E"], y)
    P[f"{src}+E_minus_{src}"] = both(oof[f"{src}+E"], oof[src], y)
    P[f"GO24:{src}+E_minus_E"] = both(oof[f"GO24:{src}+E"], oof["GO24:E"], y24)
    res["seconds"] = time.time() - t0
    np.savez(f"{RAW}/oof_{pop}_{src}_linear.npz", **{k.replace(":", "__"): v for k, v in oof.items()}, y=y, y24=y24, clip=clip)
    json.dump(res, open(f"{RAW}/res_{pop}_{src}_linear.json", "w"), indent=1)


def stage_mlp(pop, src):
    d = load_pop(pop)
    S = np.load(f"{RAW}/pca_{src}_{pop}.npz")["S"]
    y, clip, outer, inner = d["y"], d["clip"], d["outer"], d["inner"]
    counts = L.boot_counts(clip)
    path = f"{RAW}/res_{pop}_{src}_mlp.json"
    res = json.load(open(path)) if os.path.exists(path) else {"pop": pop, "src": src, "family": "mlp", "n": int(len(y)), "arms": {}}
    oofp = f"{RAW}/oof_{pop}_{src}_mlp.npz"
    oof = dict(np.load(oofp)) if os.path.exists(oofp) else {}
    t0 = time.time()
    for nm in ("E", src, f"{src}+E"):
        if nm.replace(":", "__") in oof:
            continue
        o, hps = L.cv_oof(arm_feats(d, S, nm, src), y, outer, inner, "mlp")
        oof[nm.replace(":", "__")] = o
        res["arms"][nm] = {**L.auc_ci(o, y, clip, counts=counts), "foldmean": L.auc_ci_fold(o, y, clip, outer, counts=counts), "hp_per_fold": hps}
        np.savez(oofp, **oof)
        json.dump(res, open(path, "w"), indent=1)
        print(f"{pop} {src} mlp {nm} auc {res['arms'][nm]['auc']:.4f} [{res['arms'][nm]['lo']:.4f},{res['arms'][nm]['hi']:.4f}] {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > 170:                                    # one arm per call near the 4 min wall: bank and stop; re-invoke to continue
            print("time box: banked, re-invoke to continue", flush=True)
            return
    if all(k in oof for k in ("E", src, f"{src}+E")):
        def both(a, b):
            return {**L.dauc_ci(a, b, y, clip, counts=counts), "foldmean": L.dauc_ci_fold(a, b, y, clip, outer, counts=counts)}
        res["paired"] = {f"{src}+E_minus_E": both(oof[f"{src}+E"], oof["E"]), f"{src}_minus_E": both(oof[src], oof["E"]), f"{src}+E_minus_{src}": both(oof[f"{src}+E"], oof[src])}
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, choices=("pstop", "pmove"))
    ap.add_argument("--src", default="V", choices=("V", "Z"))
    ap.add_argument("--stage", required=True, choices=("pca", "lin", "mlp"))
    a = ap.parse_args()
    fm = free_mb()
    print("free MB at start", fm, flush=True)
    {"pca": stage_pca, "lin": stage_lin, "mlp": stage_mlp}[a.stage](a.pop, a.src)
