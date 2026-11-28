#!/usr/bin/env python3
"""n1dwceil_probe_run.py -- PREREG_N1D_CEIL (sha256 092bb25c...): tree probe (+ logistic beside) on F-TEXT, F-EGO, F-TE, F-PLAN, F-ALL, identity arm, regression arm.
fit settings, folds, label, paired/level are the parent's (n1dwprobe_probe.py, unchanged). Probe seed 0 only (the tree makes no random draw: parent TREE-SEED-IDENTITY).
F-TEXT / F-EGO / F-TE / identity / regression do not depend on the N1D seed (asserted) and are fitted once, used for both units."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import n1dwprobe_probe as P  # noqa: E402

ARMS = P.ARMS
BAR = 0.0383
SEED = 0
COMMON = ("F-TEXT", "F-EGO", "F-TE")
TREE_ONLY_COMMON = ("F-TE-EGOCONST", "F-TE-TEXTZERO")
PERARM = ("F-PLAN", "F-ALL")
_G = {}


def feature_matrix(F, name):
    T, E, Pl = F["F_TEXT"].astype(np.float64), F["F_EGO"].astype(np.float64), F["F_PLAN"].astype(np.float64)
    if name == "F-TEXT":
        return T
    if name == "F-EGO":
        return E
    if name == "F-TE":
        return np.concatenate([T, E], 1)
    if name == "F-PLAN":
        return Pl
    if name == "F-ALL":
        return np.concatenate([T, E, Pl], 1)
    if name == "F-TE-EGOCONST":                      # identity arm: the 8 ego columns a constant
        return np.concatenate([T, np.zeros_like(E)], 1)
    if name == "F-TE-TEXTZERO":                      # regression arm: the 15 text columns zeroed
        return np.concatenate([np.zeros_like(T), E], 1)
    raise KeyError(name)


def _init(Fs, y, fold):
    _G["F"], _G["y"], _G["fold"] = Fs, np.asarray(y), np.asarray(fold)


def fit_one(task):
    arm, fs, probe, k = task
    F, y, fold = _G["F"][arm], _G["y"], _G["fold"]
    X = feature_matrix(F, fs)
    tr, te = fold != k, fold == k
    t0 = time.time()
    if probe == "tree":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(**P.HGB, random_state=SEED).fit(X[tr], y[tr])
        pred = clf.classes_[clf.predict_proba(X[te]).argmax(1)]
    else:
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(C=P.LOGIT_C, max_iter=2000, random_state=SEED).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
    return task, te.nonzero()[0], np.asarray(pred, np.int64), time.time() - t0


def above(c, bar=BAR):
    return bool(c["delta"] >= bar and c["lo"] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banked-feats", required=True, help="TCODE feat1 dir (F_TEXT, F_PLAN, g, ci, t0)")
    ap.add_argument("--ego1", required=True)
    ap.add_argument("--ego2", required=True, help="second extraction (K-EGO-DET)")
    ap.add_argument("--banked-oof", default="", help="TCODE oof_correct.npz (K-PARITY on the correctness vectors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    CI = P.load_ci()
    B = {arm: P.load_feat(f"{a.banked_feats}/feat_{arm}.npz") for arm in ARMS}
    E1 = {arm: P.load_feat(f"{a.ego1}/feat_{arm}.npz") for arm in ARMS}
    E2 = {arm: P.load_feat(f"{a.ego2}/feat_{arm}.npz") for arm in ARMS}
    # ---- join by window: same order, ci, t0 (smoke: the ego extraction is a subsample, joined by key)
    F, join = {}, {}
    for arm in ARMS:
        key_b = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(B[arm]["ci"], B[arm]["t0"]))}
        ii = np.array([key_b.get((int(c), int(t)), -1) for c, t in zip(E1[arm]["ci"], E1[arm]["t0"])])
        assert (ii >= 0).all(), "an ego window is absent from the banked features"
        same_order = bool(np.array_equal(ii, np.arange(len(ii))) and len(ii) == len(B[arm]["ci"]))
        assert same_order or a.smoke, "window order differs from the banked features"
        join[arm] = {"same_order_and_count": same_order, "n_ego": int(len(ii)), "n_banked": int(len(B[arm]["ci"])),
                     "ci_equal": bool(np.array_equal(E1[arm]["ci"], B[arm]["ci"][ii])), "t0_equal": bool(np.array_equal(E1[arm]["t0"], B[arm]["t0"][ii])),
                     "g_equal": bool(np.array_equal(E1[arm]["g"], B[arm]["g"][ii])), "F_TEXT_equal": bool(np.array_equal(E1[arm]["F_TEXT"], B[arm]["F_TEXT"][ii])),
                     "F_PLAN_equal": bool(np.array_equal(E1[arm]["F_PLAN"], B[arm]["F_PLAN"][ii])),
                     "F_PLAN_max_abs_diff": float(np.abs(E1[arm]["F_PLAN"].astype(np.float64) - B[arm]["F_PLAN"][ii].astype(np.float64)).max())}
        assert join[arm]["ci_equal"] and join[arm]["t0_equal"] and join[arm]["g_equal"], "ci / t0 / g differ from the banked features"
        F[arm] = {"F_TEXT": B[arm]["F_TEXT"][ii], "F_PLAN": B[arm]["F_PLAN"][ii], "F_EGO": E1[arm]["F_EGO"], "g": B[arm]["g"][ii], "ci": B[arm]["ci"][ii], "t0": B[arm]["t0"][ii]}
    ref = F[ARMS[0]]
    for arm in ARMS:
        assert np.array_equal(F[arm]["ci"], ref["ci"]) and np.array_equal(F[arm]["t0"], ref["t0"]) and np.array_equal(F[arm]["g"], ref["g"]), "window lists differ between arms"
    assert np.array_equal(F[ARMS[0]]["F_TEXT"], F[ARMS[1]]["F_TEXT"]), "F-TEXT differs between the two N1D arms: common-fit reuse is invalid"
    assert np.array_equal(F[ARMS[0]]["F_EGO"], F[ARMS[1]]["F_EGO"]), "F-EGO differs between the two N1D arms: K-EGO-DET fails; common-fit reuse is invalid"
    eid, g = ref["ci"].astype(np.int64), ref["g"]
    y = P.kin_labels(g)
    fold = P.fold_ids(eid)
    N = len(y)
    shares = np.bincount(y, minlength=3) / N
    res = {"card": "PREREG_N1D_CEIL.md (sha256 092bb25c...)", "tier": "T0 decodability of frozen features / inputs, never driving performance", "smoke": bool(a.smoke),
           "evidence_class": "MEASURED (ours; banked N1D checkpoints, D1-HO grid, CPU, zero training)",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over the held-out episodes; answers another draw of EPISODES only (training variance: the two N1D seeds)",
           "n_windows": int(N), "n_episodes": int(len(np.unique(eid))), "class_counts": np.bincount(y, minlength=3).tolist(), "majority_share": float(shares.max()), "bar": BAR,
           "probe_seed": SEED, "feat_dims": {"F_TEXT": list(ref["F_TEXT"].shape[1:]), "F_EGO": list(ref["F_EGO"].shape[1:]), "F_PLAN": list(ref["F_PLAN"].shape[1:])},
           "ego_columns": ["v", "a", "k", "v^2", "v*a", "v*k", "v^2*k", "a*k"], "join": join, "windows_per_fold": np.bincount(fold, minlength=P.NFOLD).tolist(),
           "gt_ok_false": int((~B[ARMS[0]]["gt_ok"]).sum())}
    tasks = []
    for fs in COMMON:
        for probe in P.PROBES:
            for k in range(P.NFOLD):
                tasks.append(("N1D_s0", fs, probe, k))
    for fs in TREE_ONLY_COMMON:
        for k in range(P.NFOLD):
            tasks.append(("N1D_s0", fs, "tree", k))
    for arm in ARMS:
        for fs in PERARM:
            for probe in P.PROBES:
                for k in range(P.NFOLD):
                    tasks.append((arm, fs, probe, k))
    tasks.sort(key=lambda t: (t[2] != "tree", t[1] in PERARM))
    print(f"{len(tasks)} fits, {a.workers} workers", flush=True)
    pred, done_n = {}, 0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(F, y, fold)) as ex:
        for task, idx, p, dt in ex.map(fit_one, tasks, chunksize=1):
            pred.setdefault(task[:3], np.full(N, -1, np.int64))[idx] = p
            done_n += 1
            if done_n % 5 == 0 or done_n == len(tasks):
                print(f"fits {done_n}/{len(tasks)}  {time.time() - t_all:.0f}s", flush=True)
    assert all((v >= 0).all() for v in pred.values()), "a window was never out-of-fold"

    def P_(arm, fs, probe):                                   # prediction vector for a unit (common sets are shared)
        return pred[("N1D_s0" if fs in COMMON + TREE_ONLY_COMMON else arm, fs, probe)]
    corr = {}
    for arm in ARMS:
        for fs in COMMON + TREE_ONLY_COMMON + PERARM:
            for probe in P.PROBES:
                if fs in TREE_ONLY_COMMON and probe != "tree":
                    continue
                corr[(arm, fs, probe)] = (P_(arm, fs, probe) == y).astype(np.float64)
    np.savez_compressed(f"{a.out}/oof_correct.npz", **{"|".join(k): v.astype(np.int8) for k, v in corr.items()}, y=y, eid=eid, fold=fold,
                        **{"pred|" + "|".join(k): v.astype(np.int8) for k, v in pred.items()})
    res["accuracy"] = {}
    for (arm, fs, probe), c in corr.items():
        res["accuracy"].setdefault(arm, {}).setdefault(probe, {})[fs] = {**P.level(CI, c, eid), "pred_counts": np.bincount(P_(arm, fs, probe), minlength=3).tolist()}
    CONTR = {"TE-PLAN": ("F-TE", "F-PLAN"), "TE-TEXT": ("F-TE", "F-TEXT"), "ALL-PLAN": ("F-ALL", "F-PLAN"), "EGO-TEXT": ("F-EGO", "F-TEXT"), "TE-EGO": ("F-TE", "F-EGO"),
             "ALL-TE": ("F-ALL", "F-TE"), "PLAN-TE": ("F-PLAN", "F-TE")}
    cons = {}
    for arm in ARMS:
        for probe in P.PROBES:
            cons[f"{arm}|{probe}"] = {n: P.paired(CI, corr[(arm, A, probe)], corr[(arm, Bn, probe)], eid) for n, (A, Bn) in CONTR.items()}
    res["contrasts"] = cons
    for probe in P.PROBES:
        per_unit = {}
        for arm in ARMS:
            c = cons[f"{arm}|{probe}"]["TE-PLAN"]
            per_unit[arm] = "CEIL-ABOVE-PLAN" if above(c) else "CEIL-AT-PLAN"
        labs = {}
        for lab in ("CEIL-AT-PLAN", "CEIL-ABOVE-PLAN"):
            hold = [u for u, l in per_unit.items() if l == lab]
            labs[lab] = "HOLDS" if len(hold) == len(ARMS) else ("ONE-SEED" if hold else "NOT-HELD")
        holds = [l for l, s in labs.items() if s == "HOLDS"]
        res["outcome_" + probe] = {"per_unit": per_unit, "labels": labs, "headline": holds[0] if holds else "NEITHER", "deciding": probe == "tree"}
    ctrl = {"K-CONST": {"majority_share_live": float(shares.max()), "majority_share_card": P.MAJ_SHARE,
                        "pass": bool(round(float(shares.max()), 6) == P.MAJ_SHARE) if not a.smoke else None}}
    # K-PARITY: banked TCODE correctness vectors bit-compared (F-TEXT tree, F-PLAN tree) + the accuracies the card names
    par = {"cells": {}}
    if a.banked_oof and os.path.exists(a.banked_oof) and not a.smoke:
        Z = np.load(a.banked_oof)
        for arm in ARMS:
            for fs in ("F-TEXT", "F-PLAN"):
                kk = f"{arm}|{fs}|tree|0|False"
                if kk in Z.files:
                    par["cells"][f"{arm}|{fs}"] = {"identical_correct_vector": bool(np.array_equal(Z[kk].astype(np.float64), corr[(arm, fs, "tree")])),
                                                   "accuracy": float(corr[(arm, fs, "tree")].mean()), "banked_accuracy": float(Z[kk].mean())}
    par["card_accuracies"] = {"F-TEXT": 0.6044, "F-PLAN": [0.7028, 0.6937]}
    par["live_accuracies_4dp"] = {"F-TEXT": round(float(corr[("N1D_s0", "F-TEXT", "tree")].mean()), 4), "F-PLAN": [round(float(corr[(arm, "F-PLAN", "tree")].mean()), 4) for arm in ARMS]}
    par["pass"] = None if a.smoke else bool(len(par["cells"]) == 4 and all(v["identical_correct_vector"] for v in par["cells"].values())
                                            and par["live_accuracies_4dp"]["F-TEXT"] == 0.6044 and par["live_accuracies_4dp"]["F-PLAN"] == [0.7028, 0.6937])
    par["join_ok"] = bool(all(v["ci_equal"] and v["t0_equal"] and v["g_equal"] and v["F_TEXT_equal"] and v["F_PLAN_equal"] for v in join.values()))
    ctrl["K-PARITY"] = par
    det = {arm: {k: bool(np.array_equal(E2[arm][k], E1[arm][k])) for k in ("F_EGO", "F_TEXT", "F_PLAN", "g", "ci", "t0")} for arm in ARMS}
    f0, f1, g0, g1 = E1[ARMS[0]]["F_EGO"], E1[ARMS[1]]["F_EGO"], E2[ARMS[0]]["F_EGO"], E2[ARMS[1]]["F_EGO"]
    det_cross = bool(np.array_equal(f0, f1) and np.array_equal(f0, g0) and np.array_equal(f0, g1))
    ctrl["K-EGO-DET"] = {"second_extraction_vs_first": det, "F_EGO_bit_identical_across_all_four": det_cross, "pass": bool(det_cross and all(det[arm]["F_EGO"] for arm in ARMS))}
    idp, tp = P_("N1D_s0", "F-TE-EGOCONST", "tree"), P_("N1D_s0", "F-TEXT", "tree")
    ctrl["IDENTITY-ARM"] = {"definition": "F-TE with the 8 ego columns constant: out-of-fold predictions identical to F-TEXT's on every window",
                            "n_differing_windows": int((idp != tp).sum()), "n_windows": int(N), "pass": bool((idp == tp).all())}
    reg = {}
    for arm in ARMS:
        z = corr[(arm, "F-TE-TEXTZERO", "tree")]
        te_clears_ego = above(cons[f"{arm}|tree"]["TE-EGO"])
        vs_te = P.paired(CI, z, corr[(arm, "F-TE", "tree")], eid)
        vs_ego = P.paired(CI, z, corr[(arm, "F-EGO", "tree")], eid)
        sep_below = bool(vs_te["separated"] and vs_te["delta"] < 0)
        reg[arm] = {"F-TE_clears_F-EGO": bool(te_clears_ego), "TEXTZERO_minus_F-TE": vs_te, "TEXTZERO_minus_F-EGO": vs_ego,
                    "n_windows_TEXTZERO_pred_differs_from_F-EGO_pred": int((P_(arm, "F-TE-TEXTZERO", "tree") != P_(arm, "F-EGO", "tree")).sum()),
                    "separated_below_F-TE": sep_below, "status": ("PASS" if sep_below else "FAIL") if te_clears_ego else "PRINTED-AS-SUCH (F-TE does not clear F-EGO)"}
    ctrl["REGRESSION-ARM"] = {"definition": "F-TE with the 15 text columns zeroed reads separated below F-TE, unless F-TE does not clear F-EGO (then printed as such)",
                              "cells": reg, "pass": bool(all(v["status"] != "FAIL" for v in reg.values()))}
    res["controls"] = ctrl
    res["elapsed_s"] = float(time.time() - t_all)
    json.dump(res, open(f"{a.out}/RESULT_CEIL.json", "w"), indent=1)
    print("wrote", f"{a.out}/RESULT_CEIL.json", f"{res['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
