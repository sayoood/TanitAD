#!/usr/bin/env python3
"""n1dwprobe_probe.py -- PROBE-N1D-FEAT (PREREG_V8F_N1DW.md sec 10): probes, controls, paired bootstrap. CPU only. Reads the feature files of n1dwprobe_extract.py.

Probes (card): (1) multinomial logistic regression, standardised, L2, C = 1.0 (never tuned); (2) HistGradientBoostingClassifier with RBJK's settings (rbdist_zt.HGB: max_iter 200, lr 0.1,
max_leaf_nodes 15, class_weight None, early_stopping False, random_state = the probe seed). Folds: 5-fold CV grouped by episode (ci), fold assignment seed 0 = the unique episodes shuffled by
RandomState(0), episode k-th in that order goes to fold k % 5. Probe seeds 0 and 1 (the tree and the logistic fit have no sampling without early stopping, so the seeds give identical probes;
that is MEASURED and printed as the control TREE-SEED-IDENTITY; the seed does drive the SHUF permutations).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over the episodes (answers 'another draw of EPISODES' only).
Label: read_otband.labels (3-class kinematic lat) of the GT 6 s track g, the function ast-extracted from the banked read_otband.py (sha pinned) and executed in the pod tree's namespace.
Controls: SHUF, K-CONST, K-DET, regression arm (+ additional informational K-LABEL / K-PARITY against the banked NAVRBD dump, K-SEED-IDENTITY)."""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("N1DWPROBE_TREE", "/root/strat1/navrbd/tree")
for p in (HERE, TREE + "/stack", TREE + "/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)
PYLIB = os.environ.get("N1DWPROBE_PYLIB", os.path.join(HERE, "..", "pylib"))
if os.path.isdir(PYLIB):
    sys.path.append(os.path.abspath(PYLIB))

import numpy as np  # noqa: E402

import n1dwprobe_outcome as OC  # noqa: E402

ARMS = ("N1D_s0", "N1D_s1")
FSETS = ("F-TEXT", "F-NAV", "F-PLAN", "F-VIS", "F-PLAN+TEXT")
KEY = {"F-TEXT": "F_TEXT", "F-NAV": "F_NAV", "F-PLAN": "F_PLAN", "F-VIS": "F_VIS"}
NFOLD, FOLD_SEED, NB, BOOT_SEED = 5, 0, 2000, 0
BAR = OC.BAR
MAJ_SHARE = 0.338215
HGB = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, class_weight=None, early_stopping=False)   # + random_state = the probe seed (rbdist_zt.HGB)
LOGIT_C = 1.0
SEEDS = (0, 1)
PROBES = ("tree", "logit")
READ_OTBAND = os.environ.get("N1DWPROBE_READ_OTBAND", os.path.join(HERE, "..", "lib", "read_otband.py"))
READ_OTBAND_SHA = "18afede9720159f3944046ceaecfe88e352247e8c711c4331b48797610d53ac5"
CONTRASTS = {"NAV-TEXT": ("F-NAV", "F-TEXT"), "PLAN-NAV": ("F-PLAN", "F-NAV"), "PLAN-TEXT": ("F-PLAN", "F-TEXT"), "PLANTEXT-PLAN": ("F-PLAN+TEXT", "F-PLAN"),
             "VIS-TEXT": ("F-VIS", "F-TEXT"), "PLANTEXT-TEXT": ("F-PLAN+TEXT", "F-TEXT"), "PLAN-VIS": ("F-PLAN", "F-VIS")}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ label
def kin_labels(g):
    """read_otband.labels, ast-extracted from the banked file (sha pinned) and run in the pod tree's namespace"""
    if sha256_file(READ_OTBAND) != READ_OTBAND_SHA:
        raise SystemExit("REFUSED: read_otband.py differs from its pin")
    import torch
    from taniteval import four_families as FF
    src = open(READ_OTBAND, encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "labels")
    ns = {"np": np, "torch": torch, "FF": FF, "DT": 0.2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), READ_OTBAND, "exec"), ns)
    return ns["labels"](g)[0]


# ------------------------------------------------------------------------------------------------ folds / features
def fold_ids(eid, k=NFOLD, seed=FOLD_SEED):
    u = np.unique(eid)
    perm = np.random.RandomState(seed).permutation(len(u))
    fold_of = {int(u[i]): int(r % k) for r, i in enumerate(perm)}
    return np.array([fold_of[int(e)] for e in eid], np.int64)


def load_feat(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def feature_matrix(F, name):
    if name == "F-PLAN+TEXT":
        return np.concatenate([F["F_PLAN"], F["F_TEXT"]], 1).astype(np.float64)
    if name == "F-TEXT-REG":                                  # the regression arm: c0 and the exit columns c1..c12 zeroed (c13 dist, c14 dist_valid kept)
        X = F["F_TEXT"].astype(np.float64).copy()
        X[:, :13] = 0.0
        return X
    if name == "F-TEXT-REG-ALL":                              # INFORMATIONAL (not a card control): all 15 columns zeroed = a constant feature, the floor the regression arm approaches
        return np.zeros_like(F["F_TEXT"], dtype=np.float64)
    return F[KEY[name]].astype(np.float64)


# ------------------------------------------------------------------------------------------------ one fit (runs in a worker)
_G = {}


def _init(feat_paths, y, fold):
    _G["F"] = {a: load_feat(p) for a, p in feat_paths.items()}
    _G["y"], _G["fold"] = np.asarray(y), np.asarray(fold)


def fit_one(task):
    arm, fs, probe, seed, shuf, k = task
    F, y, fold = _G["F"][arm], _G["y"], _G["fold"]
    X = feature_matrix(F, fs)
    tr, te = fold != k, fold == k
    ytr = y[tr].copy()
    if shuf:
        ytr = ytr[np.random.RandomState(1000 * seed + k + 17).permutation(len(ytr))]    # permuted within the training fold
    t0 = time.time()
    if probe == "tree":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(**HGB, random_state=seed).fit(X[tr], ytr)
        pred = clf.classes_[clf.predict_proba(X[te]).argmax(1)]
    else:
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(C=LOGIT_C, max_iter=2000, random_state=seed).fit(sc.transform(X[tr]), ytr)
        pred = clf.predict(sc.transform(X[te]))
    return task, te.nonzero()[0], np.asarray(pred, np.int64), time.time() - t0


# ------------------------------------------------------------------------------------------------ stats
def load_ci():
    from taniteval import ci as CI
    return CI


def paired(CI, a, b, eid):
    r = CI.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid), n_boot=NB, seed=BOOT_SEED)
    return {"delta": float(np.mean(a) - np.mean(b)), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(r["n_windows"]), "n_episodes": int(r["n_episodes"])}


def level(CI, a, eid):
    r = CI.episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(eid), reduce="mean", n_boot=NB, seed=BOOT_SEED)
    return {"point": float(np.mean(a)), "lo": float(r["lo"]), "hi": float(r["hi"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="dir holding feat_N1D_s0.npz and feat_N1D_s1.npz")
    ap.add_argument("--det-feats", default="", help="dir holding the SECOND extraction (feat_N1D_s0.npz, feat_N1D_s1.npz) for K-DET")
    ap.add_argument("--dump-dir", default="", help="dir holding the banked NAVRBD dumps dump_NAVRBD-HO-<arm>_m10_causal_heads (informational K-LABEL / K-PARITY)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="a code check on a reduced window set: the card's counts / 0.338215 are not asserted; never a result")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    CI = load_ci()
    paths = {arm: f"{a.feats}/feat_{arm}.npz" for arm in ARMS}
    F = {arm: load_feat(p) for arm, p in paths.items()}
    ref = F[ARMS[0]]
    for arm in ARMS:                                            # both checkpoints see the SAME windows in the same order
        assert np.array_equal(F[arm]["ci"], ref["ci"]) and np.array_equal(F[arm]["t0"], ref["t0"]) and np.array_equal(F[arm]["g"], ref["g"]), "window lists differ between arms"
    eid, g = ref["ci"].astype(np.int64), ref["g"]
    y = kin_labels(g)
    assert y.min() >= 0 and y.max() <= 2
    fold = fold_ids(eid)
    N = len(y)
    shares = np.bincount(y, minlength=3) / N
    res = {"card": "PREREG_V8F_N1DW.md sec 10 (sha256 59f2b7f919af958fd2da9c21f559c0f68db64add84e6fcb70cd5d6c7905c2301)", "tier": "T0 decodability of frozen features, never driving performance",
           "evidence_class": "MEASURED (ours; banked N1D checkpoints, D1-HO grid, CPU, zero training)", "smoke": bool(a.smoke),
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over the held-out episodes; answers 'another draw of EPISODES' only",
           "n_windows": int(N), "n_episodes": int(len(np.unique(eid))), "class_counts": np.bincount(y, minlength=3).tolist(), "majority_share": float(shares.max()),
           "bar": BAR, "folds": {"k": NFOLD, "seed": FOLD_SEED, "rule": "unique episodes shuffled by RandomState(0); k-th in that order -> fold k % 5", "windows_per_fold": np.bincount(fold, minlength=NFOLD).tolist()},
           "hgb": {**HGB, "random_state": "probe seed"}, "logit": {"C": LOGIT_C, "penalty": "l2", "standardised": True},
           "gt_ok_false": int((~ref["gt_ok"]).sum()), "feat_dims": {k: list(ref[k].shape[1:]) for k in ("F_TEXT", "F_NAV", "F_PLAN", "F_VIS")},
           "feat_sha256": {arm: sha256_file(p) for arm, p in paths.items()}}
    # ---- tasks
    fsets_all = list(FSETS) + ["F-TEXT-REG", "F-TEXT-REG-ALL"]
    tasks = []
    for arm in ARMS:
        for fs in fsets_all:
            for probe in PROBES:
                for seed in SEEDS:
                    for shuf in (False, True):
                        if fs in ("F-TEXT-REG", "F-TEXT-REG-ALL") and (shuf or probe != "tree"):
                            continue                                                    # the regression arm is a tree-probe arm
                        for k in range(NFOLD):
                            tasks.append((arm, fs, probe, seed, shuf, k))
    weight = {"F-PLAN": 0, "F-PLAN+TEXT": 1, "F-NAV": 2, "F-VIS": 3, "F-TEXT": 4, "F-TEXT-REG": 5, "F-TEXT-REG-ALL": 6}
    tasks.sort(key=lambda t: (weight[t[1]], t[2] != "tree"))
    print(f"{len(tasks)} fits, {a.workers} workers", flush=True)
    pred = {}
    done_n = 0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(paths, y, fold)) as ex:
        for task, idx, p, dt in ex.map(fit_one, tasks, chunksize=1):
            key = task[:5]
            pred.setdefault(key, np.full(N, -1, np.int64))[idx] = p
            done_n += 1
            if done_n % 20 == 0 or done_n == len(tasks):
                print(f"fits {done_n}/{len(tasks)}  {time.time() - t_all:.0f}s", flush=True)
    assert all((v >= 0).all() for v in pred.values()), "a window was never out-of-fold"
    corr = {k: (v == y).astype(np.float64) for k, v in pred.items()}
    np.savez_compressed(f"{a.out}/oof_correct.npz", **{"|".join(map(str, k)): v.astype(np.int8) for k, v in corr.items()}, y=y, eid=eid, fold=fold)
    # ---- arms
    arms = {}
    for (arm, fs, probe, seed, shuf), c in corr.items():
        nm = ("SHUF:" if shuf else "") + fs
        arms.setdefault(arm, {}).setdefault(probe, {}).setdefault(str(seed), {})[nm] = {**level(CI, c, eid), "pred_counts": np.bincount(pred[(arm, fs, probe, seed, shuf)], minlength=3).tolist()}
    res["accuracy"] = arms
    # ---- contrasts (real feature sets, paired on the same windows)
    cons = {}
    for arm in ARMS:
        for probe in PROBES:
            for seed in SEEDS:
                u = f"{arm}|{probe}|seed{seed}"
                cons[u] = {n: paired(CI, corr[(arm, A, probe, seed, False)], corr[(arm, B, probe, seed, False)], eid) for n, (A, B) in CONTRASTS.items()}
    res["contrasts"] = cons
    # ---- outcome (tree decides; the logistic probe is printed beside)
    for probe in PROBES:
        units = {u: c for u, c in cons.items() if u.split("|")[1] == probe}
        res["outcome_" + probe] = {"combine": OC.combine(units), "headline": OC.headline(OC.combine(units)), "deciding": probe == "tree"}
    # ---- controls
    ctrl = {}
    maj_pred_acc = {}
    for k in range(NFOLD):
        trm = fold != k
        m = int(np.bincount(y[trm], minlength=3).argmax())
        maj_pred_acc[k] = float((y[fold == k] == m).mean())
    cc = {"majority_share_live": float(shares.max()), "majority_share_card": MAJ_SHARE, "foldwise_train_majority_accuracy_pooled": float(np.sum([maj_pred_acc[k] * (fold == k).sum() for k in range(NFOLD)]) / N)}
    cc["pass"] = bool(round(cc["majority_share_live"], 6) == MAJ_SHARE) if not a.smoke else None
    ctrl["K-CONST"] = cc
    shuf = {}
    for arm in ARMS:
        for probe in PROBES:
            for seed in SEEDS:
                for fs in FSETS:
                    acc = float(corr[(arm, fs, probe, seed, True)].mean())
                    shuf[f"{arm}|{probe}|seed{seed}|{fs}"] = {"accuracy": acc, "abs_diff_from_majority_share": abs(acc - MAJ_SHARE), "within_0.02": bool(abs(acc - MAJ_SHARE) <= 0.02)}
    ctrl["SHUF"] = {"cells": shuf, "pass": bool(all(v["within_0.02"] for v in shuf.values())) if not a.smoke else None, "max_abs_diff": float(max(v["abs_diff_from_majority_share"] for v in shuf.values()))}
    reg = {}
    for arm in ARMS:
        for seed in SEEDS:
            c = paired(CI, corr[(arm, "F-TEXT-REG", "tree", seed, False)], corr[(arm, "F-TEXT", "tree", seed, False)], eid)
            reg[f"{arm}|tree|seed{seed}"] = {**c, "separated_below": bool(c["separated"] and c["delta"] < 0)}
    ctrl["REGRESSION-ARM"] = {"definition": "F-TEXT with c0 and the exit columns c1..c12 zeroed (c13 dist, c14 dist_valid kept), tree probe, minus F-TEXT: must be separated below 0",
                              "cells": reg, "pass": bool(all(v["separated_below"] for v in reg.values()))}
    rega = {}
    for arm in ARMS:
        for seed in SEEDS:
            c = paired(CI, corr[(arm, "F-TEXT-REG-ALL", "tree", seed, False)], corr[(arm, "F-TEXT", "tree", seed, False)], eid)
            rega[f"{arm}|tree|seed{seed}"] = {**c, "separated_below": bool(c["separated"] and c["delta"] < 0)}
    ctrl["REGRESSION-ARM-ALL-ZERO (informational)"] = {"definition": "F-TEXT with ALL 15 columns zeroed (constant feature), tree probe, minus F-TEXT; not a card control", "cells": rega}
    seedid = {}
    for arm in ARMS:
        for probe in PROBES:
            for fs in FSETS:
                seedid[f"{arm}|{probe}|{fs}"] = bool(np.array_equal(pred[(arm, fs, probe, 0, False)], pred[(arm, fs, probe, 1, False)]))
    ctrl["TREE-SEED-IDENTITY"] = {"cells_identical_between_probe_seeds_0_and_1": seedid, "all_identical": bool(all(seedid.values())),
                                  "note": "informational: without early stopping or subsampling the probes have no random draw, so probe seeds 0 and 1 give the same predictions; the card's four-unit rule still reads four units"}
    # K-DET
    if a.det_feats:
        det = {}
        for arm in ARMS:
            Z = load_feat(f"{a.det_feats}/feat_{arm}.npz")
            det[arm] = {k: bool(np.array_equal(Z[k], F[arm][k])) for k in ("F_TEXT", "F_NAV", "F_PLAN", "F_VIS", "plan0", "g", "ci", "t0")}
        ctrl["K-DET"] = {"cells": det, "pass": bool(all(all(v.values()) for v in det.values())), "second_extraction": a.det_feats}
    else:
        ctrl["K-DET"] = {"pass": None, "note": "no second extraction given"}
    # K-LABEL / K-PARITY against the banked NAVRBD dump (informational)
    if a.dump_dir:
        lab = {}
        for arm in ARMS:
            d = f"{a.dump_dir}/dump_NAVRBD-HO-{arm}_m10_causal_heads"
            fs_ = sorted(glob.glob(d + "/ep*.npz"))
            if not fs_:
                lab[arm] = {"note": "dump not found: " + d}
                continue
            Zs = [np.load(f) for f in fs_]
            dg = {(int(c), int(w)): (gg, oo) for z in Zs for c, w, gg, oo in zip(z["clip_index"], z["ws"], z["g"], z["ol"])}
            hit = [(i, dg.get((int(c), int(t)))) for i, (c, t) in enumerate(zip(ref["ci"], ref["t0"]))]
            have = [(i, v) for i, v in hit if v is not None]
            gd = np.array([v[0] for _, v in have]); od = np.array([v[1] for _, v in have]); ii = np.array([i for i, _ in have])
            lab[arm] = {"windows_matched": int(len(have)), "windows_mine": int(N), "g_max_abs_diff_m": float(np.abs(gd - g[ii]).max()), "label_agree": int((kin_labels(gd) == y[ii]).sum()),
                        "tick0_plan_max_abs_diff_m_vs_dump_ol": float(np.abs(od - F[arm]["plan0"][ii]).max()), "dump_dir": d}
        ctrl["K-LABEL/K-PARITY (informational)"] = lab
    res["controls"] = ctrl
    res["elapsed_s"] = float(time.time() - t_all)
    json.dump(res, open(f"{a.out}/PROBE_RESULT.json", "w"), indent=1)
    print("wrote", f"{a.out}/PROBE_RESULT.json", f"{res['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
