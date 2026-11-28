"""E-EXP-2 P1 — is the per-window optimal radial scale λ* FINDABLE from REF-C's latents?

Pre-registration: ../PREREG_E_EXP2.md  (blob 7d09ee5f8e9d71b1d465a0205c196b5ca7b6757f)
0 GPU here — reads the banked fan and the banked latent dump. Estimator: taniteval
paired episode-cluster bootstrap, unit = episode, n_boot=2000, seed=0.

E-EXP-1 measured the CEILING with an ORACLE λ (+0.1590 base / +0.1651 XL). This measures
how much of it a PREDICTED λ recovers, and whether that beats the `v0` echo floor
(MEASURED: corr(2 s along-track endpoint, v0) = 0.9973 — v0 nearly determines the very
quantity λ corrects, so a latent that merely encodes speed scores above zero while
carrying nothing).

Protocol, fixed in advance (§5 of the pre-registration):
  * target      grid λ* over HAD's published {0.92,0.96,1.00,1.04,1.08};
                regression target λ_ls = <f,g>/<f,f>, the closed-form LS optimum
  * score       REALISED ADE after applying the predicted λ, clipped to [0.92, 1.08]
  * protocol    leave-one-episode-out over the 40 val episodes; α by inner 5-fold
                GroupKFold over episodes, minimising REALISED ADE on the inner folds.
                Standardisation statistics are fitted on the training episodes ONLY.
  * ⛔ nothing about the held-out episode enters its own fit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                                    "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

LAMBDAS = np.array([0.92, 0.96, 1.00, 1.04, 1.08], dtype=np.float64)
LAM_LO, LAM_HI = float(LAMBDAS[0]), float(LAMBDAS[-1])
ALPHAS = np.array([1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5])
KNN_KS = (5, 10, 20, 40)
N_BOOT, SEED = 2000, 0
N_INNER = 5
#: the DEAD threshold, committed in the pre-registration: K1's measured upper bound on the
#: ENTIRE re-ranking class = 8.4 % x 0.3075 m. A lever recovering less than the bound on a
#: class we already killed does not earn a parameter.
THETA_M = 0.0258


def ade(traj, gt):
    """(...,T,2) vs (W,T,2) -> mean-over-T L2, broadcasting on leading dims."""
    return np.linalg.norm(traj - gt, axis=-1).mean(axis=-1)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _score_feats(lg):
    p = _softmax(lg.astype(np.float64))
    ent = -(p * np.log(p + 1e-12)).sum(1)
    s = np.sort(lg, axis=1)
    return np.stack([ent, s[:, -1] - s[:, -2], lg.std(1), s[:, -1], lg.mean(1)], 1)


def build_features(fan, gt, sel, v0, logits, emitted, prefinal, lat, eids, dt=0.5):
    """-> dict of feature matrices. The list of tensors read is CLOSED by the prereg."""
    W = len(gt)
    fsel = fan[np.arange(W), sel]                       # (W, T, 2)
    v0 = v0.astype(np.float64)

    F1 = np.stack([v0, v0 ** 2, v0 ** 3, 1.0 / (1.0 + v0)], 1)

    pts = np.concatenate([np.zeros((W, 1, 2)), fsel], 1)          # prepend origin
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=-1)           # (W, T)
    arc = seg.sum(1)
    head = np.arctan2(fsel[:, -1, 1], fsel[:, -1, 0])
    rng = np.linalg.norm(fsel[:, -1], axis=-1)
    t_end = fsel.shape[1] * dt
    geom = np.concatenate([
        fsel[:, :, 0], fsel[:, :, 1], seg,
        np.stack([arc, head, head / np.maximum(arc, 1e-6), rng,
                  # ⭐ the single most natural λ feature that is NOT a latent: how far a
                  # constant-velocity ego would travel, over how far the fan actually goes
                  (v0 * t_end) / np.maximum(arc, 1e-6),
                  v0 * t_end - arc], 1)], 1)
    F2 = np.concatenate([F1, geom], 1)
    F3 = np.concatenate([F2, _score_feats(logits), _score_feats(emitted),
                         _score_feats(prefinal)], 1)

    pooled = lat["pooled"]
    pseq = lat["pooled_seq"]
    ctx = lat["ctx"]
    meas = lat["measurement"]
    F5 = np.concatenate([pseq[:, -1], pseq.mean(1), pseq[:, -1] - pseq[:, 0]], 1)
    F7 = np.concatenate([pooled, ctx, meas], 1)

    rng_g = np.random.default_rng(SEED)
    perm = rng_g.permutation(W)
    perm_ep = np.arange(W)
    for e in sorted(set(eids)):
        m = np.flatnonzero(np.asarray(eids) == e)
        perm_ep[m] = rng_g.permutation(m)

    return {
        "F0_const": np.zeros((W, 0)),
        "F1_v0": F1,
        "F2_geom": F2,
        "F3_score": F3,
        "F4_pooled": pooled,
        "F5_pooled_seq": F5,
        "F6_ctx": ctx,
        "F7_all_latent": F7,
        "F8_latent_v0": np.concatenate([F7, F1], 1),
        "C_shuf": pooled[perm],
        "C_shuf_ep": pooled[perm_ep],
    }, fsel


# ---------------------------------------------------------------------------
# ridge, dual form — one eigendecomposition serves every alpha
# ---------------------------------------------------------------------------

def _standardize(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    keep = sd > 1e-8                       # a constant column carries no information and
    sd = np.where(keep, sd, 1.0)           # would divide by ~0; zero it instead of dropping
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    return A * keep, B * keep


def _ridge_from_gram(G, K, Y, alphas):
    """Dual ridge from a PRE-COMPUTED Gram. -> (n_alpha, n_test, n_target).

    ⭐ ONE eigendecomposition serves every α AND every target column. The naive version
    (one call per α per target) recomputed it 5x for the ADE-surface rule and dominated the
    runtime; this is the same estimator, not an approximation.
    """
    Y = np.atleast_2d(Y.T).T if Y.ndim > 1 else Y[:, None]
    ym = Y.mean(0)
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, 0.0)
    VtY = V.T @ (Y - ym)                        # (n, k)
    KV = K @ V                                  # (n_test, n)
    out = np.empty((len(alphas), K.shape[0], Y.shape[1]))
    for i, al in enumerate(alphas):
        out[i] = KV @ (VtY / (w + al)[:, None]) + ym
    return out


class _FoldEngine:
    """Standardise ONCE on the outer-training episodes and build its Gram ONCE; every
    inner fold is then a submatrix of it.

    ⛔ Why this is not a leak: the standardisation and the Gram are built from the OUTER
    TRAINING SET, which excludes the held-out episode entirely. Only the inner α/k selection
    sees the (still test-free) outer-train scaling. Nothing about the held-out episode enters
    its own fit.
    """

    def __init__(self, X, tr_mask, te_mask):
        self.p = X.shape[1]
        self.n_te = int(te_mask.sum())
        if self.p == 0:
            self.G = self.K = None
            return
        A, B = _standardize(X[tr_mask], X[te_mask])
        self.A = A
        self.G = A @ A.T
        self.K = B @ A.T
        self.sq = (A ** 2).sum(1)
        self.sq_te = (B ** 2).sum(1)

    def ridge(self, it, iv, Y, alphas):
        """it/iv: index arrays LOCAL to the outer-training block; iv=None -> the test set."""
        if self.p == 0:
            Y2 = Y[it] if Y.ndim > 1 else Y[it][:, None]
            m = Y2.mean(0)
            n = self.n_te if iv is None else len(iv)
            return np.repeat(m[None, None], len(alphas), 0).repeat(n, 1)
        G = self.G[np.ix_(it, it)]
        K = self.K[:, it] if iv is None else self.G[np.ix_(iv, it)]
        return _ridge_from_gram(G, K, Y[it], alphas)

    def knn(self, it, iv, y, ks):
        if self.p == 0:
            n = self.n_te if iv is None else len(iv)
            return np.repeat(y[it].mean(), n)[None].repeat(len(ks), 0)
        if iv is None:
            d = self.sq_te[:, None] + self.sq[None, it] - 2 * self.K[:, it]
        else:
            d = self.sq[iv][:, None] + self.sq[None, it] - 2 * self.G[np.ix_(iv, it)]
        order = np.argsort(d, axis=1)
        yt = y[it]
        return np.stack([yt[order[:, :k]].mean(1) for k in ks])


def _realized(lam, fsel, gt, idx):
    return ade(fsel[idx] * np.clip(lam, LAM_LO, LAM_HI)[:, None, None], gt[idx])


def _group_folds(eps_tr, n_folds, seed=SEED):
    r = np.random.default_rng(seed)
    e = np.array(sorted(eps_tr))
    r.shuffle(e)
    return [set(x.tolist()) for x in np.array_split(e, n_folds)]


def loeo_all(X, y, err, fsel, gt, eids):
    """Every rule in ONE leave-one-episode-out sweep, sharing the fold engine.

    -> dict with `ridge` / `knn` (both predict λ_ls, the pre-registered PRIMARY class) and
    `surface` (⚠️ POST-HOC: regress the whole ADE(λ) surface and take the argmin — a
    strictly more powerful rule, included so a DEAD verdict is not an artefact of a weak
    read. ⛔ A pass there with a failing primary requires a FRESH pre-registration, never a
    changed verdict.)
    """
    eids = np.asarray(eids)
    W = len(y)
    out = {k: np.empty(W) for k in ("ridge", "knn", "surface")}
    picked = {"ridge": {}, "knn": {}, "surface": {}}
    KS = np.asarray(KNN_KS, dtype=float)
    for ep in sorted(set(eids.tolist())):
        te = eids == ep
        tr = ~te
        tr_idx = np.flatnonzero(tr)
        te_idx = np.flatnonzero(te)
        eng = _FoldEngine(X, tr, te)
        y_tr = y[tr]
        err_tr = err[:, tr].T                        # (n_tr, 5) target matrix
        eps_tr = sorted(set(eids[tr].tolist()))
        c_r = np.zeros(len(ALPHAS))
        c_k = np.zeros(len(KS))
        c_s = np.zeros(len(ALPHAS))
        for fold in _group_folds(eps_tr, N_INNER):
            loc_v = np.flatnonzero(np.isin(eids[tr], list(fold)))
            loc_t = np.flatnonzero(~np.isin(eids[tr], list(fold)))
            if not len(loc_v) or not len(loc_t):
                continue
            g_v = tr_idx[loc_v]
            P = eng.ridge(loc_t, loc_v, y_tr, ALPHAS)[:, :, 0]
            S = eng.ridge(loc_t, loc_v, err_tr, ALPHAS)          # (nA, n_v, 5)
            Q = eng.knn(loc_t, loc_v, y_tr, [int(k) for k in KS])
            for i in range(len(ALPHAS)):
                c_r[i] += _realized(P[i], fsel, gt, g_v).sum()
                c_s[i] += err[S[i].argmin(1), g_v].sum()
            for i in range(len(KS)):
                c_k[i] += _realized(Q[i], fsel, gt, g_v).sum()
        al_r, al_s = ALPHAS[int(c_r.argmin())], ALPHAS[int(c_s.argmin())]
        kk = int(KS[int(c_k.argmin())])
        picked["ridge"][str(ep)] = float(al_r)
        picked["surface"][str(ep)] = float(al_s)
        picked["knn"][str(ep)] = float(kk)
        loc_all = np.arange(len(tr_idx))
        out["ridge"][te] = eng.ridge(loc_all, None, y_tr, np.array([al_r]))[0, :, 0]
        out["knn"][te] = eng.knn(loc_all, None, y_tr, [kk])[0]
        S = eng.ridge(loc_all, None, err_tr, np.array([al_s]))[0]   # (n_te, 5)
        out["surface"][te] = LAMBDAS[S.argmin(1)]
        del te_idx
    for k in ("ridge", "knn"):
        out[k] = np.clip(out[k], LAM_LO, LAM_HI)
    return out, picked


def loeo(X, y, fsel, gt, eids, mode="ridge"):
    """Single-rule wrapper — used by the P2 four-families pass, which needs only λ̂."""
    o, p = loeo_all(X, y, np.stack([ade(fsel * l, gt) for l in LAMBDAS]), fsel, gt, eids)
    return o[mode], p[mode]


# ---------------------------------------------------------------------------

def run(fan_path, lat_path):
    d = torch.load(fan_path, map_location="cpu", weights_only=False)
    L = torch.load(lat_path, map_location="cpu", weights_only=False)
    fan = d["fan"].double().numpy()
    gt = d["gt"].double().numpy()
    sel = d["sel"].numpy()
    eids = list(d["eid"])
    v0 = d["v0"].double().numpy()
    W = len(gt)

    fails = list(L.get("instrument_fail", []))
    for k in ("fan", "gt", "v0"):
        if not torch.equal(d[k].float(), L[k].float()):
            fails.append(f"latent dump's `{k}` is not bit-identical to the fan bank")
    if not torch.equal(d["sel"], L["sel"]):
        fails.append("latent dump's `sel` is not bit-identical to the fan bank")
    if list(L["eid"]) != eids:
        fails.append("latent dump's `eid` differs from the fan bank")
    if W != 881 or len(set(eids)) != 40:
        fails.append(f"counts {W}/{len(set(eids))} != 881/40")

    lat = {k: L[k].double().numpy() for k in
           ("pooled", "pooled_seq", "ctx", "measurement")}
    feats, fsel = build_features(
        fan, gt, sel, v0, d["logits"].double().numpy(),
        d["emitted_logits"].double().numpy(), d["prefinal_logits"].double().numpy(),
        lat, eids)

    # ---- targets ---------------------------------------------------------
    err = np.stack([ade(fsel * L_, gt) for L_ in LAMBDAS])          # (5, W)
    ade_sel = err[int(np.flatnonzero(LAMBDAS == 1.00)[0])]
    lam_star = LAMBDAS[err.argmin(0)]
    ceiling = err.min(0)
    num = (fsel * gt).sum(axis=(1, 2))
    den = (fsel * fsel).sum(axis=(1, 2))
    lam_ls = num / np.maximum(den, 1e-12)

    # ⭐ the echo floor's premise, re-measured here rather than inherited
    echo = float(np.corrcoef(gt[:, -1, 0], v0)[0, 1])

    B = lambda a, b: paired_episode_cluster_bootstrap(a, b, eids, n_boot=N_BOOT,
                                                     seed=SEED)
    out = {
        "arm": os.path.basename(fan_path), "n_windows": W,
        "n_episodes": len(set(eids)), "n_anchors": int(fan.shape[1]),
        "ckpt_step": int(d["ckpt_step"]), "instrument_fail": fails,
        "latent_dims": {k: list(v.shape[1:]) for k, v in lat.items()},
        "feature_dims": {k: int(v.shape[1]) for k, v in feats.items()},
        "echo_corr_gt2s_along_vs_v0": echo,
        "theta_m": THETA_M,
        "point": {
            "ade_sel": float(ade_sel.mean()),
            "ceiling_oracle_lambda": float(ceiling.mean()),
            "ceiling_recovery": float(ade_sel.mean() - ceiling.mean()),
            "lambda_star_hist": {str(l): int((lam_star == l).sum()) for l in LAMBDAS},
            "lambda_ls_mean": float(lam_ls.mean()),
            "lambda_ls_std": float(lam_ls.std()),
            "corr_lambda_ls_v0": float(np.corrcoef(lam_ls, v0)[0, 1]),
        },
        "arms": {},
    }
    out["point"]["ceiling_ci"] = B(ade_sel, ceiling)

    realized, lamhat = {}, {}
    allw = np.arange(W)
    for name, X in feats.items():
        row = {"n_features": int(X.shape[1])}
        pred, picked = loeo_all(X, lam_ls, err, fsel, gt, eids)
        for mode in ("ridge", "knn"):
            lam = pred[mode]
            r_cont = _realized(lam, fsel, gt, allw)
            snap = LAMBDAS[np.abs(lam[:, None] - LAMBDAS[None]).argmin(1)]
            r_snap = _realized(snap, fsel, gt, allw)
            row[mode] = {
                "ade_continuous": float(r_cont.mean()),
                "recovery_continuous": B(ade_sel, r_cont),
                "ade_snapped": float(r_snap.mean()),
                "recovery_snapped": B(ade_sel, r_snap),
                "acc_vs_lambda_star": float((snap == lam_star).mean()),
                "lambda_hat_mean": float(lam.mean()),
                "lambda_hat_std": float(lam.std()),
                "hyper_by_fold": picked[mode],
            }
            if mode == "ridge":
                realized[name] = r_cont
                lamhat[name] = lam
        # ⚠️ POST-HOC, labelled: the strictly more powerful ADE-surface rule
        r_s = _realized(pred["surface"], fsel, gt, allw)
        row["POSTHOC_ade_surface"] = {
            "ade": float(r_s.mean()), "recovery": B(ade_sel, r_s),
            "acc_vs_lambda_star": float((pred["surface"] == lam_star).mean()),
            "alpha_by_fold": picked["surface"],
            "NOT_PREREGISTERED": ("decides nothing; a pass here with a failing primary "
                                  "requires a FRESH pre-registration, not a changed "
                                  "verdict"),
        }
        out["arms"][name] = row
        print(f"    [{out['arm'][:28]}] {name:<16} ridge "
              f"{row['ridge']['recovery_continuous']['delta']:+.4f} | knn "
              f"{row['knn']['recovery_continuous']['delta']:+.4f} | surface "
              f"{row['POSTHOC_ade_surface']['recovery']['delta']:+.4f}", flush=True)

    # ---- the two PRIMARY contrasts --------------------------------------
    out["PRIMARY"] = {
        "R_latent_F4_pooled": out["arms"]["F4_pooled"]["ridge"]["recovery_continuous"],
        "R_v0_F1": out["arms"]["F1_v0"]["ridge"]["recovery_continuous"],
        "R_latent_minus_R_v0": B(realized["F1_v0"], realized["F4_pooled"]),
        "share_of_ceiling_F4": (
            out["arms"]["F4_pooled"]["ridge"]["recovery_continuous"]["delta"]
            / out["point"]["ceiling_recovery"]),
    }
    best = max((k for k in ("F4_pooled", "F5_pooled_seq", "F6_ctx", "F7_all_latent",
                            "F8_latent_v0")),
               key=lambda k: out["arms"][k]["ridge"]["recovery_continuous"]["delta"])
    out["SECONDARY_best_latent"] = {
        "arm": best,
        "recovery": out["arms"][best]["ridge"]["recovery_continuous"],
        "vs_v0": B(realized["F1_v0"], realized[best]),
        "multiplicity_caveat": ("best-of-5 latent arms; the pre-declared PRIMARY is "
                                "F4_pooled alone"),
    }
    out["CONTROLS"] = {
        "C_shuf": out["arms"]["C_shuf"]["ridge"]["recovery_continuous"],
        "C_shuf_ep": out["arms"]["C_shuf_ep"]["ridge"]["recovery_continuous"],
    }
    return out, realized, lamhat, ade_sel, lam_star, err, eids


def verdict(o):
    """The pre-registered three-sided table, evaluated mechanically."""
    if o["instrument_fail"]:
        return "INSTRUMENT-FAIL", o["instrument_fail"]
    for k, c in o["CONTROLS"].items():
        if c["separated"] and c["delta"] > 0:
            return "INSTRUMENT-FAIL", [f"{k} recovers separably ABOVE 0 "
                                       f"({c['delta']:+.4f}) — the pipeline leaks"]
    R = o["PRIMARY"]["R_latent_F4_pooled"]
    D = o["PRIMARY"]["R_latent_minus_R_v0"]
    why = [f"R_latent {R['delta']:+.4f} [{R['lo']:+.4f}, {R['hi']:+.4f}] "
           f"sep={R['separated']}",
           f"R_latent-R_v0 {D['delta']:+.4f} [{D['lo']:+.4f}, {D['hi']:+.4f}] "
           f"sep={D['separated']}",
           f"theta={THETA_M}"]
    if not R["separated"] or R["delta"] <= 0 or not D["separated"] or D["delta"] <= 0:
        return "DEAD", why
    return ("FINDABLE-WORTH-A-HEAD" if R["delta"] > THETA_M
            else "FINDABLE-TOO-SMALL-TO-FUND"), why


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="fan.pt:latents.pt pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-prefix", default=None,
                    help="write per-window λ̂ / realised ADE for the P2 family pass")
    a = ap.parse_args()
    res = []
    for pair in a.pairs:
        fp, lp = pair.split("::")
        o, realized, lamhat, ade_sel, lam_star, err, eids = run(fp, lp)
        o["VERDICT"], o["VERDICT_WHY"] = verdict(o)
        res.append(o)
        if a.dump_prefix:
            stem = os.path.basename(fp).replace("fan_emitted_", "").replace(".pt", "")
            np.savez(f"{a.dump_prefix}{stem}.npz",
                     ade_sel=ade_sel, lam_star=lam_star, err=err,
                     eid=np.array(eids, dtype=object).astype(str),
                     **{f"realized_{k}": v for k, v in realized.items()},
                     **{f"lamhat_{k}": v for k, v in lamhat.items()})
        print(f"\n=== {o['arm']}  N={o['n_anchors']}  "
              f"fail={o['instrument_fail'] or 'none'}")
        p = o["point"]
        print(f"  ade_sel {p['ade_sel']:.4f} -> oracle-λ ceiling "
              f"{p['ceiling_oracle_lambda']:.4f}  (recovery {p['ceiling_recovery']:+.4f})")
        print(f"  echo corr(gt 2s along, v0) = {o['echo_corr_gt2s_along_vs_v0']:.4f}")
        hdr = f"  {'arm':<16}{'nfeat':>6}{'ade':>9}{'recovery':>10}  CI"
        print(hdr)
        for k, v in o["arms"].items():
            r = v["ridge"]["recovery_continuous"]
            print(f"  {k:<16}{v['n_features']:>6}{v['ridge']['ade_continuous']:>9.4f}"
                  f"{r['delta']:>+10.4f}  [{r['lo']:+.4f},{r['hi']:+.4f}] "
                  f"sep={r['separated']}")
        print(f"  VERDICT: {o['VERDICT']}  :: " + " | ".join(o["VERDICT_WHY"]))
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
