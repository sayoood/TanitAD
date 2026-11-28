"""FOV crop audit — PART 2 step 3: fit, score and ADJUDICATE the FOV x resolution x aspect sweep.

Probe: the situation classifier's image-only arm, re-fit per arm. PCA rank **16** (the measured
dose-response optimum; raw-2048 concat is 17-25 % worse — INHERITED, sibling stream, replicated in
its own CV) fitted on TRAIN rows only, then a closed-form ridge. Every arm gets the SAME clips, the
SAME frames, the SAME labels and the SAME chunk-grouped split; only the crop geometry and the input
shape differ.

⚠️ Chance is `taniteval.rank_metrics.chance_ap` and every AP uses `ties="collapse"`. The repaired
module is used deliberately: the stable-argsort comparator that this program shipped scored
**1.726x chance** on an all-tied score, biasing every above-chance test toward "not separated".
`assert_chance_comparator` is called on the constant comparator and is unwaivable.

The verdict is COMPUTED by `verdict()` from the rules fixed in `FOV_CROP_AUDIT.md` Sec 1.3 before
any number existed, and it can return `REFUSED` (vacuity guard) and `INSTRUMENT-BLIND` (power
guard) as well as CONFIRM / PARTIAL / REFUTE.

usage:  python fov_eval.py <feats_dir> <labels_dir> <out_json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get(
    "TANITEVAL_DIR", r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\taniteval"))
from taniteval.ci import _draws, episode_index                       # noqa: E402
from taniteval.rank_metrics import (assert_chance_comparator,        # noqa: E402
                                    average_precision, chance_ap)

RANK = 16
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
B_BOOT = 2000
SEED = 0
SITS = ["lane_change", "intersection", "roundabout"]
MIN_CLUSTERS = 40


def fit_pca(X, rank):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    # economy SVD on the (n x d) matrix; n >> d = 2048 so the Gram route is cheaper
    C = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    idx = np.argsort(w)[::-1][:rank]
    return mu, V[:, idx].astype(np.float32), float(w[idx].sum() / max(w.sum(), 1e-12))


def ridge_fit(Z, y, lam):
    Z1 = np.hstack([Z, np.ones((len(Z), 1), np.float64)])
    A = Z1.T @ Z1 + lam * np.eye(Z1.shape[1])
    A[-1, -1] -= lam                                    # do not penalise the intercept
    return np.linalg.solve(A, Z1.T @ y.astype(np.float64))


def ridge_pred(Z, w):
    return np.hstack([Z, np.ones((len(Z), 1), np.float64)]) @ w


def ap_ci(y, s, eid, n_boot=B_BOOT, seed=SEED):
    uniq, idx = episode_index(eid)
    pt = average_precision(y, s)
    b = []
    for sel in _draws(uniq, idx, n_boot, seed):
        yy = y[sel]
        if yy.sum() == 0:
            continue
        b.append(average_precision(yy, s[sel]))
    b = np.asarray(b)
    lo, hi = np.percentile(b, [2.5, 97.5])
    base = chance_ap(y)
    return {"AP": round(float(pt), 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "base_rate": round(base, 6), "AP_over_base": round(float(pt) / max(base, 1e-12), 4),
            "above_chance": bool(lo > base), "n_boot": int(len(b))}


def paired_ap_delta(y, sa, sb, eid, n_boot=B_BOOT, seed=SEED):
    uniq, idx = episode_index(eid)
    pt = average_precision(y, sa) - average_precision(y, sb)
    d = []
    for sel in _draws(uniq, idx, n_boot, seed):
        yy = y[sel]
        if yy.sum() == 0:
            continue
        d.append(average_precision(yy, sa[sel]) - average_precision(yy, sb[sel]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(pt), 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "separated": bool(lo > 0 or hi < 0), "favours_a": bool(lo > 0),
            "n_boot": int(len(d)), "estimator": "paired_clip_cluster_bootstrap"}


def verdict(res, sit, wide_arms, base_arm="A_51_256"):
    """The pre-registered rule, executed. Can return REFUSED / INSTRUMENT-BLIND / CONFIRM /
    PARTIAL / REFUTE — every branch reachable, which is the point."""
    p = res["power"][sit]
    if p["n_positive_clip_clusters"] < MIN_CLUSTERS:
        return {"verdict": "REFUSED", "why": f"C-POW {p['n_positive_clip_clusters']} < "
                f"{MIN_CLUSTERS} held-out positive clip clusters — no verdict is emitted"}
    A = res["arms"][sit]
    if not A[base_arm]["above_chance"]:
        return {"verdict": "REFUSED", "why": "the baseline arm (today's 51.4 deg / 256 px) does "
                "not clear chance in this universe; a sweep whose baseline cannot see the target "
                "cannot rank crops"}
    blind = res["contrasts"][sit].get("R_blur100_vs_A_51_256")
    if blind is None or not (blind["separated"] and blind["lo"] < 0):
        return {"verdict": "INSTRUMENT-BLIND", "why": "the known-worse control R_blur100 (today's "
                "field degraded to the 100 deg arm's angular resolution, ZERO information added) "
                "is not detected as worse; the sweep has no power to rank crops",
                "control": blind}
    wins_matched, wins_base = [], []
    for a in wide_arms:
        m = res["contrasts"][sit].get(f"{a}_vs_M_match100")
        b = res["contrasts"][sit].get(f"{a}_vs_{base_arm}")
        if m and m["separated"] and m["favours_a"]:
            wins_matched.append(a)
        if b and b["delta"] > 0:
            wins_base.append(a)
    both = [a for a in wins_matched if a in wins_base]
    if both:
        return {"verdict": "CONFIRM", "why": "wider beats its own matched-degradation control AND "
                "the undegraded baseline", "arms": both}
    if wins_matched:
        return {"verdict": "PARTIAL", "why": "wider beats its matched-degradation control (the "
                "information IS out there) but not the undegraded baseline — the resolution/"
                "padding cost eats the gain; the remedy is more pixels, not a wider F_REF",
                "arms": wins_matched}
    return {"verdict": "REFUTE", "why": "no wide arm beats its matched-degradation control and no "
            "wide arm beats the baseline on the point estimate"}


def main():
    feats_dir, labels_dir, out_json = sys.argv[1:4]
    L = np.load(os.path.join(labels_dir, "fov_labels.npz"))
    meta = {m["k"]: m for m in json.load(open(os.path.join(labels_dir, "fov_meta.json")))}
    files = sorted(f for f in os.listdir(feats_dir) if f.startswith("clip_"))
    ks = [int(f[5:10]) for f in files]
    arms = list(np.load(os.path.join(feats_dir, files[0])).keys())
    arms = [a for a in arms if a != "t"]
    print(f"[eval] {len(ks)} clips, {len(arms)} arms", flush=True)

    X = {a: [] for a in arms}
    Y = {s: [] for s in SITS}
    V = {s: [] for s in SITS}
    EID, SIDE = [], []
    for f, k in zip(files, ks):
        z = np.load(os.path.join(feats_dir, f))
        t = z["t"]
        for a in arms:
            X[a].append(z[a].astype(np.float32))
        for s in SITS:
            Y[s].append(L[f"c{k}_y_{s}"][t])
            V[s].append(L[f"c{k}_valid_{s}"][t])
        EID.append(np.full(len(t), k))
        SIDE.append(np.full(len(t), meta[k]["side"] == "HELDOUT"))
    for a in arms:
        X[a] = np.concatenate(X[a])
    for s in SITS:
        Y[s] = np.concatenate(Y[s]).astype(np.float64)
        V[s] = np.concatenate(V[s]).astype(bool)
    EID = np.concatenate(EID)
    HELD = np.concatenate(SIDE)

    res = {"n_clips": len(ks), "n_rows": int(len(EID)), "arms_list": arms,
           "rank": RANK, "n_boot": B_BOOT, "MIN_CLUSTERS": MIN_CLUSTERS,
           "estimator": "paired clip-cluster bootstrap (taniteval.ci._draws)",
           "rank_metrics": "taniteval.rank_metrics, ties='collapse' (repaired comparator)",
           "power": {}, "arms": {}, "contrasts": {}, "controls": {}, "verdicts": {}}

    for s in SITS:
        m_tr = V[s] & ~HELD
        m_he = V[s] & HELD
        pos_clusters = len(np.unique(EID[m_he & (Y[s] > 0)]))
        res["power"][s] = {"n_train_rows": int(m_tr.sum()), "n_heldout_rows": int(m_he.sum()),
                           "n_heldout_pos": int(Y[s][m_he].sum()),
                           "n_positive_clip_clusters": int(pos_clusters),
                           "base_rate": round(float(Y[s][m_he].mean()), 6),
                           "C_POW": "OK" if pos_clusters >= MIN_CLUSTERS else "UNDERPOWERED"}
        res["arms"][s], res["contrasts"][s], scores = {}, {}, {}
        if m_tr.sum() < 100 or Y[s][m_tr].sum() < 5 or m_he.sum() < 100:
            continue
        for a in arms:
            mu, W, evr = fit_pca(X[a][m_tr], RANK)
            Ztr = (X[a][m_tr] - mu) @ W
            Zhe = (X[a][m_he] - mu) @ W
            sd = Ztr.std(0, keepdims=True) + 1e-6
            Ztr, Zhe = Ztr / sd, Zhe / sd
            # lambda chosen on TRAIN by out-of-fold AP over chunk-grouped folds -> no held-out peek
            chunks = np.array([meta[int(k)]["chunk"] for k in EID[m_tr]])
            uch = np.unique(chunks)
            folds = {c: i % 5 for i, c in enumerate(uch)}
            fid = np.array([folds[c] for c in chunks])
            best, best_ap = LAMBDAS[0], -1.0
            for lam in LAMBDAS:
                oof = np.zeros(len(Ztr))
                for f in range(5):
                    tr, va = fid != f, fid == f
                    if Y[s][m_tr][tr].sum() < 2 or Y[s][m_tr][va].sum() < 1:
                        continue
                    oof[va] = ridge_pred(Ztr[va], ridge_fit(Ztr[tr], Y[s][m_tr][tr], lam))
                apv = average_precision(Y[s][m_tr], oof)
                if apv > best_ap:
                    best, best_ap = lam, apv
            w = ridge_fit(Ztr, Y[s][m_tr], best)
            sc = ridge_pred(Zhe, w)
            scores[a] = sc
            res["arms"][s][a] = {**ap_ci(Y[s][m_he], sc, EID[m_he]),
                                 "lambda": best, "cv_AP_train_oof": round(float(best_ap), 6),
                                 "pca_explained_var": round(evr, 5)}
        # ---- controls, both directions ----
        rng = np.random.default_rng(SEED)
        base_a = "A_51_256" if "A_51_256" in scores else arms[0]
        Xs = X[base_a].copy()
        for j in range(Xs.shape[1]):                       # C-NEG: destroy every row association
            Xs[:, j] = Xs[rng.permutation(len(Xs)), j]
        mu, W, _ = fit_pca(Xs[m_tr], RANK)
        Ztr, Zhe = (Xs[m_tr] - mu) @ W, (Xs[m_he] - mu) @ W
        sd = Ztr.std(0, keepdims=True) + 1e-6
        sc_neg = ridge_pred(Zhe / sd, ridge_fit(Ztr / sd, Y[s][m_tr], 100.0))
        const = np.zeros(int(m_he.sum()))
        res["controls"][s] = {
            "C_NEG_shuffled": ap_ci(Y[s][m_he], sc_neg, EID[m_he], n_boot=400),
            "chance_comparator_audit": assert_chance_comparator(
                Y[s][m_he], const, name="constant-score chance comparator")}
        for a in arms:
            for ref in ("M_match100", base_a):
                if a == ref or ref not in scores or a not in scores:
                    continue
                res["contrasts"][s][f"{a}_vs_{ref}"] = paired_ap_delta(
                    Y[s][m_he], scores[a], scores[ref], EID[m_he])

    wide = [a for a in arms if a.startswith("F")]
    for s in SITS:
        if s in res["arms"] and res["arms"][s]:
            res["verdicts"][s] = verdict(res, s, wide)
        else:
            res["verdicts"][s] = {"verdict": "REFUSED", "why": "no fitted arm"}
    json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps({"power": res["power"], "verdicts": res["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
