"""RESOLUTION-GAIN — step 3: fit, score and ADJUDICATE the downward angular-resolution ladder.

Two probes, both read off the SAME frozen features, both adjudicated PER SITUATION and NEVER pooled:

  P1 (PRIMARY)   the situation classifier's image-only arm — PCA-16 + closed-form ridge, AP on the
                 held-out (chunk-grouped) side, lambda chosen by out-of-fold AP on TRAIN only.
  P2 (SECONDARY, and the SENSITIVITY instrument)
                 image-only ridge regression of ego SPEED and |YAW RATE| from the same frozen state,
                 full-rank, out-of-sample R^2. Continuous ⇒ far more power per clip than a
                 rare-event AP, and it reads the 3-frame stack's apparent motion, which is exactly
                 what a low-pass degrades.

The verdict is COMPUTED by `verdict()` from the rules fixed in `PRE_REGISTRATION.md` Sec 4 BEFORE
any score existed. Every branch is reachable: UNPOWERED / INSTRUMENT-BLIND / GAIN / NO GAIN /
PARTIAL.

⛔ Estimator: paired clip-cluster bootstrap (`taniteval.ci._draws`, B = 2000, seed 0), unit = clip.
   `overlapping_holdout_se` is not used and is not importable from here.

usage:  python res_eval.py <feats_dir> <labels_dir> <out_json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

TANITEVAL = os.environ.get("TANITEVAL_DIR",
                           r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\taniteval")
SC_SCRIPTS = os.environ.get(
    "SC_SCRIPTS",
    r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
    r"\Architecture & Inference\Implementation\incoming\2026-07-26-situation-classifier\scripts")
sys.path.insert(0, TANITEVAL)
sys.path.insert(0, SC_SCRIPTS)

from sc_situations import kinematics                                  # noqa: E402
from taniteval.ci import _draws, episode_index                        # noqa: E402
from taniteval.rank_metrics import (assert_chance_comparator,         # noqa: E402
                                    average_precision, chance_ap)

RANK = 16
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
LAMBDAS_REG = (10.0, 100.0, 1000.0, 10000.0, 100000.0)
B_BOOT = 2000
SEED = 0
SITS = ["lane_change", "intersection", "roundabout"]
MIN_CLUSTERS = 40
BASE_ARM = "V5_640"
LADDER_A = ["V5_640", "D_today", "D_1p5", "D_2", "D_3", "D_6"]
LADDER_B = ["B_today", "B_D2", "B_D6"]
DEMO_ARM = "D_6"                 # the registered S-DEMO rung for ladder A
DEMO_ARM_B = "B_D6"
PRIMARY_ARM = "D_1p5"            # the exact mirror of the 384x960 step
SECOND_ARM = "D_2"
UP_ARM = "U_960"
MIN_DR = 2.0                     # registered: DR = |delta(D_6)| / halfwidth(CI[delta(D_1p5)])


# ------------------------------------------------------------------ fitting helpers
def fit_pca(X, rank):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    idx = np.argsort(w)[::-1][:rank]
    return mu, V[:, idx].astype(np.float32), float(w[idx].sum() / max(w.sum(), 1e-12))


def ridge_fit(Z, y, lam):
    Z1 = np.hstack([Z, np.ones((len(Z), 1), np.float64)])
    A = Z1.T @ Z1 + lam * np.eye(Z1.shape[1])
    A[-1, -1] -= lam                                     # never penalise the intercept
    return np.linalg.solve(A, Z1.T @ y.astype(np.float64))


def ridge_pred(Z, w):
    return np.hstack([Z, np.ones((len(Z), 1), np.float64)]) @ w


# ------------------------------------------------------------------ interval helpers
def ap_ci(y, s, eid, n_boot=B_BOOT, seed=SEED):
    uniq, idx = episode_index(eid)
    pt = average_precision(y, s)
    b = [average_precision(y[sel], s[sel]) for sel in _draws(uniq, idx, n_boot, seed)
         if y[sel].sum() > 0]
    b = np.asarray(b)
    lo, hi = np.percentile(b, [2.5, 97.5])
    base = chance_ap(y)
    return {"AP": round(float(pt), 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "base_rate": round(base, 6), "AP_over_base": round(float(pt) / max(base, 1e-12), 4),
            "above_chance": bool(lo > base), "n_boot": int(len(b))}


def paired_delta(y, sa, sb, eid, fn, n_boot=B_BOOT, seed=SEED):
    """Paired clip-cluster bootstrap of `fn(y, a) - fn(y, b)` on the SAME resampled rows."""
    uniq, idx = episode_index(eid)
    pt = fn(y, sa) - fn(y, sb)
    d = []
    for sel in _draws(uniq, idx, n_boot, seed):
        yy = y[sel]
        if fn is average_precision and yy.sum() == 0:
            continue
        d.append(fn(yy, sa[sel]) - fn(yy, sb[sel]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(pt), 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "halfwidth": round(float(hi - lo) / 2.0, 6),
            "separated": bool(lo > 0 or hi < 0), "favours_a": bool(lo > 0),
            "worse": bool(hi < 0), "n_boot": int(len(d)),
            "estimator": "paired_clip_cluster_bootstrap"}


def r2(y, p):
    sse = float(((y - p) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


# ------------------------------------------------------------------ the registered rule
def verdict(res, probe, key):
    """`PRE_REGISTRATION.md` Sec 4, executed. Returns one of
    UNPOWERED / INSTRUMENT-BLIND / GAIN / NO GAIN / PARTIAL."""
    P = res[probe]["power"].get(key, {})
    if P.get("n_positive_clip_clusters", 10 ** 9) < MIN_CLUSTERS:
        return {"verdict": "UNPOWERED",
                "why": f"{P.get('n_positive_clip_clusters')} < {MIN_CLUSTERS} held-out positive "
                       "clip clusters — no verdict is emitted, and this is NOT 'no effect'"}
    C = res[probe]["contrasts"].get(key, {})
    demo = C.get(f"{DEMO_ARM}_vs_{BASE_ARM}")
    prim = C.get(f"{PRIMARY_ARM}_vs_{BASE_ARM}")
    seco = C.get(f"{SECOND_ARM}_vs_{BASE_ARM}")
    up = C.get(f"{UP_ARM}_vs_{BASE_ARM}")
    if demo is None or prim is None:
        return {"verdict": "UNPOWERED", "why": "a registered contrast is missing"}
    if not (demo["separated"] and demo["worse"]):
        return {"verdict": "INSTRUMENT-BLIND",
                "why": f"S-DEMO FAILED: the extreme rung {DEMO_ARM} (a 6x angular-resolution loss "
                       "to 0.889 px/deg) is NOT detected as worse; a probe that cannot see that "
                       "cannot rank resolutions, so no verdict is emitted about any milder rung",
                "S_DEMO": demo}
    dr = abs(demo["delta"]) / max(prim["halfwidth"], 1e-12)
    if dr < MIN_DR:
        return {"verdict": "UNPOWERED",
                "why": f"DR = |delta({DEMO_ARM})| / halfwidth(CI[delta({PRIMARY_ARM})]) = "
                       f"{dr:.2f} < {MIN_DR}: the instrument's demonstrated dynamic range is not "
                       "large against its own resolution, so a null here is UNPOWERED, not NO GAIN",
                "DR": round(dr, 3)}
    if up is not None and up["separated"] and up["favours_a"]:
        return {"verdict": "GAIN", "why": "the DIRECT upward arm U_960 (8.0 px/deg, 1440 tokens) "
                "beats the v5 frame with a separated CI DESPITE its declared one-directional "
                "handicap — strong evidence, and it overrides the ladder", "DR": round(dr, 3),
                "driver": "U_960", "U_960": up}
    if prim["separated"] and prim["worse"]:
        return {"verdict": "GAIN",
                "why": "the 1.5x angular-resolution LOSS that exactly mirrors the 384x960 step "
                       "costs materially (separated) ⇒ the model is sensitive at this scale ⇒ "
                       "there is headroom; recommend the costed upward training test",
                "DR": round(dr, 3), "driver": PRIMARY_ARM, "primary": prim}
    if seco is not None and seco["separated"] and seco["worse"]:
        return {"verdict": "PARTIAL",
                "why": "a 1.5x loss is free but a 2x loss is not: the knee lies BELOW the v5 frame "
                       "(between 2.667 and 3.556 px/deg). For the UPWARD question this is still "
                       "NO GAIN, with the knee located", "DR": round(dr, 3),
                "primary": prim, "secondary": seco}
    return {"verdict": "NO GAIN",
            "why": "the probe demonstrably detects a 6x resolution loss, yet BOTH a 1.5x and a 2x "
                   "loss are free (CIs contain 0) ⇒ the model is insensitive to angular resolution "
                   "over the range straddling the v5 frame ⇒ 256x640 is the right stopping point "
                   "and 384x960 is not worth several GPU-weeks",
            "DR": round(dr, 3), "primary": prim, "secondary": seco}


# ------------------------------------------------------------------ main
def main():
    feats_dir, labels_dir, out_json = sys.argv[1:4]
    L = np.load(os.path.join(labels_dir, "fov_labels.npz"))
    meta = {m["k"]: m for m in json.load(open(os.path.join(labels_dir, "fov_meta.json")))}
    files = sorted(f for f in os.listdir(feats_dir) if f.startswith("clip_"))
    ks = [int(f[5:10]) for f in files]
    z0 = np.load(os.path.join(feats_dir, files[0]))
    arms = [a for a in z0.keys() if a != "t"]
    print(f"[eval] {len(ks)} clips, {len(arms)} arms", flush=True)

    X = {a: [] for a in arms}
    Y = {s: [] for s in SITS}
    V = {s: [] for s in SITS}
    ONG = {s: [] for s in SITS}
    KIN = {"speed": [], "absyaw": []}
    EID, SIDE = [], []
    for f, k in zip(files, ks):
        z = np.load(os.path.join(feats_dir, f))
        t = z["t"]
        if any(a not in z for a in arms):
            continue
        for a in arms:
            X[a].append(z[a].astype(np.float32))
        for s in SITS:
            Y[s].append(L[f"c{k}_y_{s}"][t])
            V[s].append(L[f"c{k}_valid_{s}"][t])
            ONG[s].append(L[f"c{k}_ongoing_{s}"][t])
        P = L[f"c{k}_poses"]
        K = kinematics(P.astype(np.float64))
        KIN["speed"].append(K["v"][t])
        KIN["absyaw"].append(np.abs(K["omega"])[t])
        EID.append(np.full(len(t), k))
        SIDE.append(np.full(len(t), meta[k]["side"] == "HELDOUT"))
    for a in arms:
        X[a] = np.concatenate(X[a])
    for s in SITS:
        Y[s] = np.concatenate(Y[s]).astype(np.float64)
        V[s] = np.concatenate(V[s]).astype(bool)
        ONG[s] = np.concatenate(ONG[s]).astype(bool)
    for t_ in KIN:
        KIN[t_] = np.concatenate(KIN[t_]).astype(np.float64)
    EID = np.concatenate(EID)
    HELD = np.concatenate(SIDE)

    res = {"n_clips": len(ks), "n_rows": int(len(EID)), "arms_list": arms,
           "rank_P1": RANK, "n_boot": B_BOOT, "MIN_CLUSTERS": MIN_CLUSTERS,
           "BASE_ARM": BASE_ARM, "MIN_DR": MIN_DR,
           "estimator": "paired clip-cluster bootstrap (taniteval.ci._draws), unit = clip",
           "rank_metrics": "taniteval.rank_metrics, ties='collapse' (repaired comparator)",
           "P1": {"power": {}, "arms": {}, "contrasts": {}, "controls": {}, "verdicts": {}},
           "P2": {"power": {}, "arms": {}, "contrasts": {}, "controls": {}, "verdicts": {}},
           "P3": {"power": {}, "arms": {}, "contrasts": {}, "controls": {}, "verdicts": {}}}

    # ================================================================= P1 — situations
    for s in SITS:
        m_tr, m_he = V[s] & ~HELD, V[s] & HELD
        pos_cl = len(np.unique(EID[m_he & (Y[s] > 0)]))
        res["P1"]["power"][s] = {
            "n_train_rows": int(m_tr.sum()), "n_heldout_rows": int(m_he.sum()),
            "n_heldout_pos": int(Y[s][m_he].sum()), "n_positive_clip_clusters": int(pos_cl),
            "base_rate": round(float(Y[s][m_he].mean()) if m_he.sum() else 0.0, 6),
            "C_POW": "OK" if pos_cl >= MIN_CLUSTERS else "UNDERPOWERED"}
        res["P1"]["arms"][s], res["P1"]["contrasts"][s] = {}, {}
        if m_tr.sum() < 100 or Y[s][m_tr].sum() < 5 or m_he.sum() < 100:
            continue
        chunks = np.array([meta[int(k)]["chunk"] for k in EID[m_tr]])
        uch = np.unique(chunks)
        fid = np.array([{c: i % 5 for i, c in enumerate(uch)}[c] for c in chunks])
        scores = {}
        for a in arms:
            mu, W, evr = fit_pca(X[a][m_tr], RANK)
            Ztr, Zhe = (X[a][m_tr] - mu) @ W, (X[a][m_he] - mu) @ W
            sd = Ztr.std(0, keepdims=True) + 1e-6
            Ztr, Zhe = Ztr / sd, Zhe / sd
            best, best_ap = LAMBDAS[0], -1.0
            for lam in LAMBDAS:
                oof = np.zeros(len(Ztr))
                for fo in range(5):
                    tr, va = fid != fo, fid == fo
                    if Y[s][m_tr][tr].sum() < 2 or Y[s][m_tr][va].sum() < 1:
                        continue
                    oof[va] = ridge_pred(Ztr[va], ridge_fit(Ztr[tr], Y[s][m_tr][tr], lam))
                apv = average_precision(Y[s][m_tr], oof)
                if apv > best_ap:
                    best, best_ap = lam, apv
            sc = ridge_pred(Zhe, ridge_fit(Ztr, Y[s][m_tr], best))
            scores[a] = sc
            res["P1"]["arms"][s][a] = {**ap_ci(Y[s][m_he], sc, EID[m_he]), "lambda": best,
                                       "cv_AP_train_oof": round(float(best_ap), 6),
                                       "pca_explained_var": round(evr, 5)}
        # ---- C-NEG + chance audit ----
        rng = np.random.default_rng(SEED)
        Xs = X[BASE_ARM].copy()
        for j in range(Xs.shape[1]):
            Xs[:, j] = Xs[rng.permutation(len(Xs)), j]
        mu, W, _ = fit_pca(Xs[m_tr], RANK)
        Ztr, Zhe = (Xs[m_tr] - mu) @ W, (Xs[m_he] - mu) @ W
        sd = Ztr.std(0, keepdims=True) + 1e-6
        sc_neg = ridge_pred(Zhe / sd, ridge_fit(Ztr / sd, Y[s][m_tr], 100.0))
        res["P1"]["controls"][s] = {
            "C_NEG_shuffled": ap_ci(Y[s][m_he], sc_neg, EID[m_he], n_boot=400),
            "chance_comparator_audit": assert_chance_comparator(
                Y[s][m_he], np.zeros(int(m_he.sum())), name="constant-score chance comparator")}
        for a in arms:
            if a == BASE_ARM:
                continue
            res["P1"]["contrasts"][s][f"{a}_vs_{BASE_ARM}"] = paired_delta(
                Y[s][m_he], scores[a], scores[BASE_ARM], EID[m_he], average_precision)
        # ladder B is adjudicated against ITS OWN baseline (a different frame and token count)
        if "B_today" in scores:
            for a in LADDER_B[1:]:
                if a in scores:
                    res["P1"]["contrasts"][s][f"{a}_vs_B_today"] = paired_delta(
                        Y[s][m_he], scores[a], scores["B_today"], EID[m_he], average_precision)

    # ================================================================= P2 — kinematics
    m_tr, m_he = ~HELD, HELD
    chunks = np.array([meta[int(k)]["chunk"] for k in EID[m_tr]])
    uch = np.unique(chunks)
    fid = np.array([{c: i % 5 for i, c in enumerate(uch)}[c] for c in chunks])
    for tgt, yall in KIN.items():
        res["P2"]["power"][tgt] = {
            "n_train_rows": int(m_tr.sum()), "n_heldout_rows": int(m_he.sum()),
            "n_positive_clip_clusters": int(len(np.unique(EID[m_he]))),
            "target_mean": round(float(yall[m_he].mean()), 5),
            "target_sd": round(float(yall[m_he].std()), 5), "C_POW": "OK"}
        res["P2"]["arms"][tgt], res["P2"]["contrasts"][tgt] = {}, {}
        preds = {}
        ytr, yhe = yall[m_tr], yall[m_he]
        for a in arms:
            Xtr, Xhe = X[a][m_tr].astype(np.float64), X[a][m_he].astype(np.float64)
            mu = Xtr.mean(0, keepdims=True)
            sd = Xtr.std(0, keepdims=True) + 1e-6
            Ztr, Zhe = (Xtr - mu) / sd, (Xhe - mu) / sd
            best, best_r2 = LAMBDAS_REG[0], -1e18
            for lam in LAMBDAS_REG:
                oof = np.zeros(len(Ztr))
                for fo in range(5):
                    tr, va = fid != fo, fid == fo
                    oof[va] = ridge_pred(Ztr[va], ridge_fit(Ztr[tr], ytr[tr], lam))
                v = r2(ytr, oof)
                if v > best_r2:
                    best, best_r2 = lam, v
            p = ridge_pred(Zhe, ridge_fit(Ztr, ytr, best))
            preds[a] = p
            uniq, idx = episode_index(EID[m_he])
            b = np.asarray([r2(yhe[sel], p[sel]) for sel in _draws(uniq, idx, 400, SEED)])
            res["P2"]["arms"][tgt][a] = {
                "R2": round(r2(yhe, p), 6), "lo": round(float(np.percentile(b, 2.5)), 6),
                "hi": round(float(np.percentile(b, 97.5)), 6), "lambda": best,
                "cv_R2_train_oof": round(float(best_r2), 6)}
        # C-NEG for the regression probe
        rng = np.random.default_rng(SEED)
        Xs = X[BASE_ARM].astype(np.float64).copy()
        for j in range(Xs.shape[1]):
            Xs[:, j] = Xs[rng.permutation(len(Xs)), j]
        mu, sd = Xs[m_tr].mean(0, keepdims=True), Xs[m_tr].std(0, keepdims=True) + 1e-6
        pn = ridge_pred((Xs[m_he] - mu) / sd,
                        ridge_fit((Xs[m_tr] - mu) / sd, ytr, 1000.0))
        res["P2"]["controls"][tgt] = {"C_NEG_shuffled_R2": round(r2(yhe, pn), 6)}
        for a in arms:
            if a == BASE_ARM:
                continue
            res["P2"]["contrasts"][tgt][f"{a}_vs_{BASE_ARM}"] = paired_delta(
                yhe, preds[a], preds[BASE_ARM], EID[m_he], r2)
        if "B_today" in preds:
            for a in LADDER_B[1:]:
                if a in preds:
                    res["P2"]["contrasts"][tgt][f"{a}_vs_B_today"] = paired_delta(
                        yhe, preds[a], preds["B_today"], EID[m_he], r2)
        # ---- P3 (amendment A1): the SAME heads, evaluated PER SITUATION on that situation's own
        # ongoing frames. Adds no fit and no free parameter — it re-slices the held-out side — and
        # it is what makes a per-situation resolution read possible even if P1's rare-event AP
        # turns out to be blind. Registered in PRE_REGISTRATION.md amendment A1.
        for s in SITS:
            sel = ONG[s][m_he]
            if sel.sum() < 200:
                res["P3"]["power"][f"{tgt}|{s}"] = {
                    "n_heldout_rows": int(sel.sum()), "n_positive_clip_clusters": 0,
                    "C_POW": "UNDERPOWERED"}
                res["P3"]["contrasts"][f"{tgt}|{s}"] = {}
                continue
            eid_s = EID[m_he][sel]
            res["P3"]["power"][f"{tgt}|{s}"] = {
                "n_heldout_rows": int(sel.sum()),
                "n_positive_clip_clusters": int(len(np.unique(eid_s))),
                "target_mean": round(float(yhe[sel].mean()), 5),
                "target_sd": round(float(yhe[sel].std()), 5),
                "C_POW": "OK" if len(np.unique(eid_s)) >= MIN_CLUSTERS else "UNDERPOWERED"}
            res["P3"]["arms"][f"{tgt}|{s}"] = {
                a: {"R2": round(r2(yhe[sel], preds[a][sel]), 6)} for a in arms}
            res["P3"]["contrasts"][f"{tgt}|{s}"] = {
                f"{a}_vs_{BASE_ARM}": paired_delta(
                    yhe[sel], preds[a][sel], preds[BASE_ARM][sel], eid_s, r2)
                for a in arms if a != BASE_ARM}
            if "B_today" in preds:
                for a in LADDER_B[1:]:
                    res["P3"]["contrasts"][f"{tgt}|{s}"][f"{a}_vs_B_today"] = paired_delta(
                        yhe[sel], preds[a][sel], preds["B_today"][sel], eid_s, r2)

    for key in res["P3"]["power"]:
        res["P3"]["verdicts"][key] = (verdict(res, "P3", key)
                                      if res["P3"]["contrasts"].get(key)
                                      else {"verdict": "UNPOWERED", "why": "too few rows"})

    for s in SITS:
        res["P1"]["verdicts"][s] = (verdict(res, "P1", s) if res["P1"]["arms"].get(s)
                                    else {"verdict": "UNPOWERED", "why": "no fitted arm"})
    for tgt in KIN:
        res["P2"]["verdicts"][tgt] = verdict(res, "P2", tgt)

    json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps({"P1_power": res["P1"]["power"], "P1_verdicts": res["P1"]["verdicts"],
                      "P2_verdicts": res["P2"]["verdicts"],
                      "P3_power": res["P3"]["power"],
                      "P3_verdicts": res["P3"]["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
