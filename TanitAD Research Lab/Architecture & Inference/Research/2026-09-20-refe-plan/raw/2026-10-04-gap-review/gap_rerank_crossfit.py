#!/usr/bin/env python3
"""REVIEW 7 -- DIAGNOSTIC: how much of the selection gap is recoverable with FAITHFUL (NAVSIM) labels and NO new
visual information? (CPU only; navsim venv python; OMP_NUM_THREADS capped by the caller.)

⛔ This trains on NAVTEST labels, so its number is NEVER a reportable navtest score. It is cross-fitted by LOG
(two halves by a fixed hash; fit on one half, select on the other, both directions pooled), so every scored token's
re-ranker never saw its log. It answers one question: is the information needed to choose well ALREADY present in
the shipped model's per-proposal outputs + the proposal's own geometry, once the label is NAVSIM's own? If yes, the
binding constraint is LABEL FIDELITY / CALIBRATION, and the deployable version is the same re-ranker (or the scorer
itself) trained on navtrain proposals labelled by NAVSIM's PDM scorer.

Features per proposal (all available at inference, vision-only + ego + the model's own outputs):
  the 6 stored logits; path length, its within-set rank and ratio to the set max; endpoint x, y; max |y|;
  heading change; max curvature proxy and max lateral-acceleration proxy; the 8 per-step distances (speed profile);
  ego v0; set context: per-logit set mean and the proposal's endpoint distance to the set medoid.
Labels: NAVSIM GT nc==1, dac==1, ttc==1 (binary classifiers) and GT EP on SAFE proposals only (regressor; avoids the
EP leak documented in gap_heads.py). Selection: navsim_v1 aggregate of the predicted components (comfort = model's).
"""
import hashlib
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import gap_selection as GS

OUT = GS.PKG + "/raw/2026-10-04-gap-review/gap_rerank_crossfit.json"
HOOKS = GS.ROOT + "/data/refe_navtest/score/refe_navtest_final/refe_navtest_final_hooks.json"


def main():
    z = np.load(GS.PROPS, allow_pickle=True)
    tok = z["token"]
    props = z["proposals"].astype(np.float32)
    logit = z["logits"].astype(np.float32)
    pick = z["pick"]
    N = len(tok)
    pos = {t: i for i, t in enumerate(tok)}
    G = {k: np.full((N, 64), np.nan, np.float32) for k in ("nc", "dac", "ttc", "c", "pdms", "ep")}
    logs = np.empty(N, dtype=object)
    for line in open(GS.CENSUS):
        r = json.loads(line)
        i = pos[r["token"]]
        logs[i] = r["log"]
        for k in ("nc", "dac", "ttc", "c", "pdms"):
            G[k][i] = np.asarray(r[k][1:65], np.float32)
        m = G["nc"][i] * G["dac"][i]
        with np.errstate(divide="ignore", invalid="ignore"):
            ep = np.where(m > 0, (12 * G["pdms"][i] / np.where(m > 0, m, 1) - 5 * G["ttc"][i] - 2 * G["c"][i]) / 5, 0)
        G["ep"][i] = np.clip(ep, 0, 1)
    logs = logs.astype(str)
    v0 = np.zeros(N, np.float32)
    for h in json.load(open(HOOKS))["pdm_score_calls"]:
        v0[pos[h["token"]]] = h["v0_mps"]

    # ---- features
    xy = props[..., :2]
    full = np.concatenate([np.zeros((N, 64, 1, 2), np.float32), xy], axis=2)
    d = np.linalg.norm(np.diff(full, axis=2), axis=-1)            # [N,64,8] per-0.5 s distances
    L = d.sum(-1)
    Lmax = L.max(1, keepdims=True)
    rankL = L.argsort(1).argsort(1).astype(np.float32) / 63.0
    yaw = props[..., 2]
    dyaw = np.abs(np.diff(np.concatenate([np.zeros((N, 64, 1), np.float32), yaw[..., :-1]], -1), axis=-1))
    # last pose heading is repaired at execution (t=4.0 s := t=3.5 s in 8-pose space is approximated by step 6)
    curv = dyaw / np.maximum(d[..., :7], 0.05)
    spd = d / 0.5
    alat = (spd[..., :7] ** 2) * curv
    end = xy[:, :, -1, :]
    med = np.median(end, axis=1, keepdims=True)
    dmed = np.linalg.norm(end - med, axis=-1)
    lg_mean = logit.mean(1, keepdims=True).repeat(64, 1)
    F = np.concatenate([
        logit, lg_mean,
        L[..., None], (L / np.maximum(Lmax, 1e-3))[..., None], rankL[..., None],
        end, np.abs(xy[..., 1]).max(-1)[..., None],
        np.abs(yaw[..., 6] - 0)[..., None], curv.max(-1)[..., None], alat.max(-1)[..., None],
        d, np.repeat(v0[:, None, None], 64, 1), dmed[..., None]], axis=-1).astype(np.float32)
    nfeat = F.shape[-1]
    half = np.array([int(hashlib.sha256(l.encode()).hexdigest(), 16) % 2 for l in logs])
    rows = np.arange(N)
    base = G["pdms"][rows, pick].astype(np.float64)
    P = 1 / (1 + np.exp(-logit.astype(np.float64)))
    pred = {k: np.zeros((N, 64)) for k in ("nc", "dac", "ttc", "ep")}
    fitinfo = {}
    for h in (0, 1):
        tr = half != h
        te = half == h
        Xtr = F[tr].reshape(-1, nfeat)
        Xte = F[te].reshape(-1, nfeat)
        for k in ("nc", "dac", "ttc"):
            y = (G[k][tr].reshape(-1) >= 1.0).astype(np.int8)
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                 l2_regularization=1.0, random_state=0)
            clf.fit(Xtr, y)
            pred[k][te] = clf.predict_proba(Xte)[:, 1].reshape(te.sum(), 64)
        safe_tr = ((G["nc"][tr] * G["dac"][tr]) == 1.0).reshape(-1)
        reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                            l2_regularization=1.0, random_state=0)
        reg.fit(Xtr[safe_tr], G["ep"][tr].reshape(-1)[safe_tr])
        pred["ep"][te] = np.clip(reg.predict(Xte).reshape(te.sum(), 64), 0, 1)
        fitinfo[str(h)] = {"train_props": int(Xtr.shape[0]), "train_tokens": int(tr.sum())}

    pc = P[..., 4]
    arms = {
        "R1_reranker_all_heads_faithful_labels": GS.agg(pred["nc"], pred["dac"], pred["ep"], pred["ttc"], pc),
        "R2_reranker_NC_DAC_TTC_only_model_EP": GS.agg(pred["nc"], pred["dac"], P[..., 2], pred["ttc"], pc),
        "R3_reranker_DAC_only": GS.agg(P[..., 0], pred["dac"], P[..., 2], P[..., 3], pc),
        "R4_reranker_TTC_only": GS.agg(P[..., 0], P[..., 1], P[..., 2], pred["ttc"], pc),
        "R5_reranker_NC_only": GS.agg(pred["nc"], P[..., 1], P[..., 2], P[..., 3], pc),
    }
    out = {"n_tokens": N, "n_features": nfeat, "fit": fitinfo,
           "warning": "DIAGNOSTIC ONLY -- trained on navtest labels, cross-fitted by log; never a reportable score",
           "pick": GS.boot_paired(base, base, logs)}
    for k, a in arms.items():
        sel = a.argmax(1)
        sc = G["pdms"][rows, sel].astype(np.float64)
        res = GS.boot_paired(sc, base, logs)
        res["picks_scoring_zero"] = int((sc == 0).sum())
        msk = (G["nc"][rows, sel] * G["dac"][rows, sel]) > 0
        res["EP_x100_of_pick_among_M>0"] = float(G["ep"][rows, sel][msk].mean() * 100)
        res["mean_path_len_pick_m"] = float(L[rows, sel].mean())
        out[k] = res
    out["shipped_pick_EP_x100_among_M>0"] = float(G["ep"][rows, pick][(G["nc"][rows, pick] * G["dac"][rows, pick]) > 0].mean() * 100)
    out["shipped_pick_mean_path_len_m"] = float(L[rows, pick].mean())
    # within-set AUC of the re-ranker vs the shipped heads, DAC
    def auc_within(p, lab):
        vals = []
        for a, b in zip(p, lab):
            pos_ = a[b >= 1]
            neg_ = a[b < 1]
            if len(pos_) == 0 or len(neg_) == 0:
                continue
            vals.append(float((pos_[:, None] > neg_[None, :]).mean() + 0.5 * (pos_[:, None] == neg_[None, :]).mean()))
        return float(np.mean(vals)), len(vals)
    out["within_set_AUC_DAC_shipped_vs_reranker"] = [auc_within(P[..., 1], G["dac"]), auc_within(pred["dac"], G["dac"])]
    out["within_set_AUC_NC_shipped_vs_reranker"] = [auc_within(P[..., 0], G["nc"]), auc_within(pred["nc"], G["nc"])]
    out["within_set_AUC_TTC_shipped_vs_reranker"] = [auc_within(P[..., 3], G["ttc"]), auc_within(pred["ttc"], G["ttc"])]
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
