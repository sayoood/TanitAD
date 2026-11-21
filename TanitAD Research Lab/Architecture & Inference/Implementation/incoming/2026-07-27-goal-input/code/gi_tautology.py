#!/usr/bin/env python3
"""S1 -- THE TAUTOLOGY TEST (priority 1).

    Is predicting a 2-D endpoint genuinely EASIER than picking among 256
    eight-dimensional trajectories?

Both quantities are put on ONE axis: the 2 s ENDPOINT L2 error in metres.

    e_sel   ||fan[w, sel_w, -1] - gt[w, -1]||     the as-trained selector's
                                                  trajectory-picking error,
                                                  expressed on the goal axis
    e_head  ||ghat_w - gt[w, -1]||                a learned goal head, OUT-OF-FOLD

If e_head >= e_sel the decomposition buys NOTHING and the 88 % is an artifact of
handing over ground truth.

Every head input is observation-only and is asserted against the oracle
whitelist in `gi_common.assert_no_oracle`. `head_deg` (future net heading
change), `a_gt`, `v_target`/`vt_*` and `route*` are inadmissible and are never
read.

Folds: 5 EPISODE-DISJOINT folds; ridge alpha chosen by an inner GroupKFold over
train-fold EPISODES, so no episode is ever in both fit and score.

Usage:  python gi_tautology.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from gi_common import (OUT, assert_no_oracle, ci_paired, ci_single, eid_str,
                       endpoint_err, episode_folds, load_eh2, load_refc_fan, r4,
                       SEED)

warnings.filterwarnings("ignore")

ALPHAS = np.logspace(-3, 4, 22)
N_PCA = 32


# --------------------------------------------------------------------------- #
# features -- observation-only by construction, audited by name                #
# --------------------------------------------------------------------------- #
def build_features(d: dict, eh: dict, variant: str = "neutral") -> dict:
    """-> dict of named blocks. NOTHING future-derived may enter here."""
    pre = "neutral|" if variant == "neutral" else ""
    used = ["lat", "lon", "dist", "refined_pre", "prior",   # eh2 (latent)
            "v0", "cv", "fan", "logits", "sel"]             # fan dump
    assert_no_oracle(used)

    W = len(d["gt"])
    fan = d["fan"].numpy().astype(np.float64)
    cv = d["cv"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    logits = d["logits"].numpy().astype(np.float64)
    sel = d["sel"].numpy()

    F_lat = np.concatenate([eh[pre + "lat"].numpy(), eh[pre + "lon"].numpy(),
                            eh[pre + "dist"].numpy()], 1).astype(np.float64)
    F_ego = np.concatenate([v0[:, None], cv.reshape(W, 8)], 1)

    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    fan_end = fan[:, :, -1, :]                                  # [W,N,2]
    F_ans = np.concatenate([fan[np.arange(W), sel][:, -1, :],    # sel endpoint
                            (p[:, :, None] * fan_end).sum(1)], 1)
    F_conf_raw = np.concatenate([eh[pre + "refined_pre"].numpy(),
                                 eh[pre + "prior"].numpy()], 1).astype(np.float64)
    return {"F_lat": F_lat, "F_ego": F_ego, "F_ans": F_ans,
            "F_conf_raw": F_conf_raw,
            "cv_end": cv[:, -1, :], "sel_end": fan[np.arange(W), sel][:, -1, :]}


FEATSETS = {"lat_ego": ("F_lat", "F_ego"),
            "all": ("F_lat", "F_ego", "F_ans", "F_conf_pca")}


def assemble(B: dict, name: str, tr, te):
    """Fold-safe assembly: the PCA of the 512 confidences is fitted on the TRAIN
    fold only, then applied to both."""
    blocks_tr, blocks_te = [], []
    for k in FEATSETS[name]:
        if k == "F_conf_pca":
            pca = PCA(n_components=N_PCA, random_state=SEED).fit(B["F_conf_raw"][tr])
            blocks_tr.append(pca.transform(B["F_conf_raw"][tr]))
            blocks_te.append(pca.transform(B["F_conf_raw"][te]))
        else:
            blocks_tr.append(B[k][tr])
            blocks_te.append(B[k][te])
    Xtr, Xte = np.concatenate(blocks_tr, 1), np.concatenate(blocks_te, 1)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


# --------------------------------------------------------------------------- #
# models                                                                       #
# --------------------------------------------------------------------------- #
def fit_ridge(Xtr, Ytr, gtr):
    """alpha by inner GroupKFold over TRAIN-fold episodes -- never an episode
    that will be scored."""
    ng = len(np.unique(gtr))
    kf = GroupKFold(n_splits=min(4, ng))
    best, best_a = np.inf, ALPHAS[0]
    for a in ALPHAS:
        errs = []
        for itr, ite in kf.split(Xtr, Ytr, groups=gtr):
            m = Ridge(alpha=a).fit(Xtr[itr], Ytr[itr])
            errs.append(np.linalg.norm(m.predict(Xtr[ite]) - Ytr[ite],
                                       axis=-1).mean())
        e = float(np.mean(errs))
        if e < best:
            best, best_a = e, a
    return Ridge(alpha=best_a).fit(Xtr, Ytr), best_a


def fit_mlp(Xtr, Ytr, _g):
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000,
                        early_stopping=True, n_iter_no_change=30,
                        validation_fraction=0.2, random_state=SEED,
                        learning_rate_init=3e-3).fit(Xtr, Ytr), None


def fit_gbr(Xtr, Ytr, _g):
    ms = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                        max_depth=4, l2_regularization=1.0,
                                        early_stopping=True,
                                        validation_fraction=0.2,
                                        random_state=SEED).fit(Xtr, Ytr[:, j])
          for j in range(2)]
    return ms, None


def predict(model, X):
    if isinstance(model, list):
        return np.stack([m.predict(X) for m in model], 1)
    return model.predict(X)


MODELS = {"ridge": fit_ridge, "mlp": fit_mlp, "gbr": fit_gbr}
#: target parameterisations. `resid_cv`/`resid_sel` shrink toward a KINEMATIC or
#: the SELECTOR's own answer instead of toward the marginal mean -- deliberately
#: generous to the head, because the test is designed to be able to REFUTE.
TARGETS = ("raw", "resid_cv", "resid_sel")


def offset(B, mode):
    if mode == "raw":
        return np.zeros_like(B["cv_end"])
    return B["cv_end"] if mode == "resid_cv" else B["sel_end"]


# --------------------------------------------------------------------------- #
def run_oof(B, g_end, eid, featset, model, target, insample=False):
    W = len(g_end)
    off = offset(B, target)
    Y = g_end - off
    pred = np.zeros((W, 2))
    alphas = []
    if insample:
        idx = np.arange(W)
        Xtr, Xte = assemble(B, featset, idx, idx)
        m, a = MODELS[model](Xtr, Y, eid)
        pred = predict(m, Xte)
        alphas.append(a)
    else:
        for tr, te in episode_folds(eid):
            Xtr, Xte = assemble(B, featset, tr, te)
            m, a = MODELS[model](Xtr, Y[tr], eid[tr])
            pred[te] = predict(m, Xte)
            alphas.append(a)
    return pred + off, alphas


def main() -> None:
    d = load_refc_fan("xl")
    eh = load_eh2()
    gt = d["gt"].numpy().astype(np.float64)
    fan = d["fan"].numpy().astype(np.float64)
    sel = d["sel"].numpy()
    eid = eid_str(d)
    W = len(gt)
    g_end = gt[:, -1]
    B = build_features(d, eh, "neutral")

    err4 = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
    a0 = err4[np.arange(W), sel]

    # ---- zero-training reference arms on the endpoint axis -----------------
    e_sel = endpoint_err(B["sel_end"], g_end)
    e_cv = endpoint_err(B["cv_end"], g_end)
    e_const = np.zeros(W)
    for tr, te in episode_folds(eid):
        e_const[te] = endpoint_err(np.repeat(g_end[tr].mean(0)[None],
                                             te.sum(), 0), g_end[te])

    res = {"_stream": "2026-07-27-goal-input", "_stage": "S1 tautology test",
           "_axis": "2 s endpoint L2 error, metres",
           "_estimator": "paired_episode_cluster_bootstrap B=2000, unit=episode",
           "_folds": "5 episode-disjoint, seed 20260727",
           "_features": {k: list(np.shape(B[k])) for k in
                         ("F_lat", "F_ego", "F_ans", "F_conf_raw")},
           "_oracle_fields_excluded": ["head_deg", "a_gt", "v_target",
                                       "vt_valid", "vt_lookahead", "gt",
                                       "route", "route_graded"],
           "as_trained_ade_0_2s": r4(a0.mean()),
           "arms": {}, "preds": {}}

    def record(name, e, pred=None, extra=None):
        blk = {"endpoint_err_m": ci_single(e, eid),
               "vs_e_sel": ci_paired(e, e_sel, eid),
               "err_along_m": r4(np.abs((pred[:, 0] - g_end[:, 0])).mean())
               if pred is not None else None,
               "err_cross_m": r4(np.abs((pred[:, 1] - g_end[:, 1])).mean())
               if pred is not None else None}
        if extra:
            blk.update(extra)
        res["arms"][name] = blk
        return blk

    record("H0_const", e_const, np.repeat(g_end.mean(0)[None], W, 0))
    record("H1_cv", e_cv, B["cv_end"])
    record("H2_sel", e_sel, B["sel_end"])

    print("=== S1 TAUTOLOGY TEST -- axis: 2 s endpoint L2, metres ===")
    print(f"{'arm':30s} {'e (m)':>8s} {'CI':>18s} {'vs e_sel':>9s} sep")
    for k in ("H0_const", "H1_cv", "H2_sel"):
        a = res["arms"][k]
        print(f"{k:30s} {a['endpoint_err_m']['mean']:8.4f} "
              f"[{a['endpoint_err_m']['lo']:.3f},{a['endpoint_err_m']['hi']:.3f}] "
              f"{a['vs_e_sel']['delta']:+9.4f} "
              f"{'SEP' if a['vs_e_sel']['separated'] else '-'}")

    # ---- the learned grid --------------------------------------------------
    best = (np.inf, None, None)
    for fs in FEATSETS:
        for mdl in MODELS:
            for tgt in TARGETS:
                pred, alphas = run_oof(B, g_end, eid, fs, mdl, tgt)
                e = endpoint_err(pred, g_end)
                nm = f"H_{mdl}_{fs}_{tgt}"
                record(nm, e, pred, {"ridge_alphas": [float(a) for a in alphas]
                                     if mdl == "ridge" else None})
                res["preds"][nm] = pred.tolist()
                a = res["arms"][nm]
                print(f"{nm:30s} {a['endpoint_err_m']['mean']:8.4f} "
                      f"[{a['endpoint_err_m']['lo']:.3f},"
                      f"{a['endpoint_err_m']['hi']:.3f}] "
                      f"{a['vs_e_sel']['delta']:+9.4f} "
                      f"{'SEP' if a['vs_e_sel']['separated'] else '-'}")
                if e.mean() < best[0]:
                    best = (e.mean(), nm, pred)

    res["_best_oof"] = {"arm": best[1], "endpoint_err_m": r4(best[0])}
    print(f"\nBEST OOF HEAD: {best[1]}  e_head = {best[0]:.4f} m "
          f"vs e_sel = {e_sel.mean():.4f} m")

    # ---- H7 in-sample bound (the "more data would fix it" discharge) -------
    mdl, fs, tgt = best[1].split("_")[1], "_".join(best[1].split("_")[2:-1]), \
        best[1].split("_")[-1]
    pred_is, _ = run_oof(B, g_end, eid, fs, mdl, tgt, insample=True)
    e_is = endpoint_err(pred_is, g_end)
    record("H7_insample", e_is, pred_is)
    res["preds"]["H7_insample"] = pred_is.tolist()
    print(f"H7_insample (bound):           {e_is.mean():8.4f}")

    # ---- NC3 shuffled features (must collapse to H0_const) ----------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(W)
    Bs = {k: (v[perm] if k.startswith("F_") else v) for k, v in B.items()}
    pred_nc, _ = run_oof(Bs, g_end, eid, fs, mdl, "raw")
    e_nc = endpoint_err(pred_nc, g_end)
    record("NC3_shuffled_feats", e_nc, pred_nc)
    print(f"NC3_shuffled_feats:            {e_nc.mean():8.4f}  "
          f"(H0_const = {e_const.mean():.4f})")

    # ---- THE VERDICT -------------------------------------------------------
    d_taut = ci_paired(endpoint_err(best[2], g_end), e_sel, eid)
    res["TAUTOLOGY"] = {
        "e_head_m": r4(best[0]), "e_sel_m": r4(e_sel.mean()),
        "delta_head_minus_sel": d_taut,
        "ratio_head_over_sel": r4(best[0] / e_sel.mean()),
        "verdict": ("T-PASS" if (d_taut["delta"] < 0 and d_taut["separated"])
                    else "T-REFUTE" if (d_taut["delta"] >= 0)
                    else "T-UNPOWERED"),
        "_note": ("T-PASS is NECESSARY, NOT SUFFICIENT: a shrunk goal can have "
                  "lower L2 and still be a worse reference. The operative test "
                  "is S3."),
    }
    np.savez_compressed(OUT / "gi_head_preds.npz",
                        best=best[2], insample=pred_is, nc3=pred_nc,
                        gt_end=g_end, sel_end=B["sel_end"], cv_end=B["cv_end"],
                        eid=eid, arm=np.array([best[1]]))
    (OUT / "gi_tautology.json").write_text(json.dumps(res, indent=2))
    print(f"\n*** TAUTOLOGY VERDICT: {res['TAUTOLOGY']['verdict']} *** "
          f"e_head {best[0]:.4f} vs e_sel {e_sel.mean():.4f}  "
          f"delta {d_taut['delta']:+.4f} [{d_taut['lo']:+.4f},{d_taut['hi']:+.4f}] "
          f"sep={d_taut['separated']}")
    print("wrote", OUT / "gi_tautology.json")


if __name__ == "__main__":
    main()
