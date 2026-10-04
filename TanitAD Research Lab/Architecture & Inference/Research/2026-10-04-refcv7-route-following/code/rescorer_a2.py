"""SPEC_ADDENDUM_A2: a TRAIN-fitted linear conditional-logit re-scorer over the model's own selection terms.

Fit on train_s0 ONLY (5-fold episode-grouped CV for the L2 strength), scored on eval_s0g (seed 0) and eval_s1 (the
inference replicate). Reuses analyze_route's derive / plan_metrics / bar_check so the metrics are byte-identical in
definition to SPEC §5's lever table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_route as A  # noqa: E402
import route_metrics as rm  # noqa: E402

FEATS = ["sampler_conf", "tac8_lat", "tac8_lon", "behaviour", "navc", "e9_graft", "s_e9_z", "agree_tac_side",
         "p_side", "agree_nav_side", "abs_theta", "progress_ratio", "alat_peak", "ade8_to_prior", "dist_to_gtac",
         "is_e9_pick"]
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ARMS = {"X1": FEATS, "X2_no_nav": [f for f in FEATS if f not in ("navc", "agree_nav_side")],
        "X3_no_goalhead": [f for f in FEATS if f != "dist_to_gtac"]}
SLOT_T = np.concatenate([[0.0], rm.SLOT_T_S])


def alat_peak(F):
    """max_t v_t^2 |kappa_t| on the 8-slot geometry (origin prepended); joints with a step < 0.05 m excluded."""
    P = np.concatenate([np.zeros(F.shape[:-2] + (1, 2)), F], axis=-2)
    dP = np.diff(P, axis=-2)                               # [..., 8, 2]
    ds = np.linalg.norm(dP, axis=-1)
    dt = np.diff(SLOT_T)
    v = ds / dt
    hd = np.arctan2(dP[..., 1], dP[..., 0])
    dh = rm.wrap(hd[..., 1:] - hd[..., :-1])
    dsm = 0.5 * (ds[..., 1:] + ds[..., :-1])
    kap = np.abs(dh) / np.maximum(dsm, 1e-6)
    vj = 0.5 * (v[..., 1:] + v[..., :-1])
    ok = (ds[..., 1:] > 0.05) & (ds[..., :-1] > 0.05)
    return np.where(ok, vj ** 2 * kap, 0.0).max(-1)


def features(z, d):
    W, N = z["s_e9"].shape
    reach = d["reach"]
    f = {}
    f["sampler_conf"] = z["refined_base"]
    f["tac8_lat"] = z["tac8_lat"]
    f["tac8_lon"] = z["tac8_lon"]
    f["behaviour"] = d["beh_term"]
    f["navc"] = d["navc_term"]
    f["e9_graft"] = z["s_e9"] - z["s_core"]
    mu = np.array([z["s_e9"][w][reach[w]].mean() for w in range(W)])
    sd = np.array([max(z["s_e9"][w][reach[w]].std(), 1e-6) for w in range(W)])
    f["s_e9_z"] = (z["s_e9"] - mu[:, None]) / sd[:, None]
    f["agree_tac_side"] = (d["dir_c"] == d["tac_side"][:, None]).astype(float)
    f["p_side"] = A.pside(z, d)
    f["agree_nav_side"] = ((d["dir_c"] == d["nav_side"][:, None]) & (d["nav_side"][:, None] != 0)).astype(float)
    f["abs_theta"] = np.abs(d["th_c"])
    prog = np.linalg.norm(z["fan"][:, :, 7, :], axis=-1)
    f["progress_ratio"] = prog / np.maximum(6.0 * z["v0"][:, None], 1.0)
    f["alat_peak"] = alat_peak(z["fan"].astype(np.float64))
    pp = z["prior_path"].astype(np.float64)
    f["ade8_to_prior"] = np.linalg.norm(z["fan"] - pp[:, None], axis=-1).mean(-1)
    g = z["g_tac"]
    f["dist_to_gtac"] = sum(np.linalg.norm(z["fan"][:, :, si, :] - g[:, None, ti, :2], axis=-1)
                            for ti, si in ((0, 3), (1, 5), (2, 7)))
    isp = np.zeros((W, N))
    isp[np.arange(W), z["sel_idx"]] = 1.0
    f["is_e9_pick"] = isp
    return np.stack([f[k] for k in FEATS], -1).astype(np.float64)       # [W, N, 16]


def oracle_reach(d):
    a = np.where(d["reach"] & np.isfinite(d["ade_c"]), d["ade_c"], np.inf)
    o = a.argmin(1)
    ok = np.isfinite(a.min(1))
    return o, ok


class CLogit:
    def __init__(self, cols, alpha):
        self.cols, self.alpha = cols, alpha

    def fit(self, X, keep, y, ok, mu, sd):
        Xs = (X[..., self.cols] - mu[self.cols]) / sd[self.cols]
        Xs = Xs[ok]
        K = keep[ok]
        yy = y[ok]
        n = Xs.shape[0]

        def fg(th):
            u = Xs @ th
            u = np.where(K, u, -np.inf)
            m = u.max(1, keepdims=True)
            e = np.exp(u - m)
            Z = e.sum(1, keepdims=True)
            p = e / Z
            lse = (m + np.log(Z))[:, 0]
            uo = u[np.arange(n), yy]
            loss = -(uo - lse).mean() + self.alpha * th @ th
            exp_phi = np.einsum("wc,wcf->wf", np.where(K, p, 0.0), np.where(K[..., None], Xs, 0.0))
            grad = -(Xs[np.arange(n), yy] - exp_phi).mean(0) + 2 * self.alpha * th
            return loss, grad
        r = minimize(fg, np.zeros(len(self.cols)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.theta, self.mu, self.sd, self.opt = r.x, mu, sd, {"success": bool(r.success), "nit": int(r.nit),
                                                              "loss": float(r.fun)}
        return self

    def pick(self, X, keep):
        u = ((X[..., self.cols] - self.mu[self.cols]) / self.sd[self.cols]) @ self.theta
        return A.masked_argmax(u, keep, keep)[0]


def standardise(X, keep):
    flat = X[keep]
    return flat.mean(0), np.maximum(flat.std(0), 1e-9)


def mean_ade(d, idx, mask=None):
    a = d["ade_c"][np.arange(d["W"]), idx]
    m = np.isfinite(a) if mask is None else (mask & np.isfinite(a))
    return float(a[m].mean())


def fit_arm(cols, Xt, dt, y, ok, mu, sd, seed=0):
    eps = np.unique(dt["ep"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    fold_of = {eps[p]: i % 5 for i, p in enumerate(perm)}
    folds = np.array([fold_of[e] for e in dt["ep"]])
    cv = {}
    for a in ALPHAS:
        picks = np.zeros(dt["W"], np.int64)
        for k in range(5):
            tr = folds != k
            m = CLogit(cols, a).fit(Xt[tr], dt["reach"][tr], y[tr], ok[tr], mu, sd)
            picks[~tr] = m.pick(Xt[~tr], dt["reach"][~tr])
        cv[a] = mean_ade(dt, picks)
    a_star = min(cv, key=lambda a: (cv[a], -a))
    model = CLogit(cols, a_star).fit(Xt, dt["reach"], y, ok, mu, sd)
    return model, {"cv_ade_by_alpha": {str(k): round(v, 4) for k, v in cv.items()}, "alpha": a_star,
                   "train_in_sample_ade": round(mean_ade(dt, model.pick(Xt, dt["reach"])), 4),
                   "train_V0_ade": round(mean_ade(dt, dt["_z"]["sel_idx"]), 4),
                   "theta": {FEATS[c]: round(float(t), 4) for c, t in zip(cols, model.theta)}, "opt": model.opt}


def table_for(z, d, picks, draws):
    """the lever_table row format for custom picks (V0 + arms + oracle)."""
    ar = np.arange(d["W"])
    base = A.plan_metrics(z["fan"][ar, z["sel_idx"]].astype(np.float64), z, d)
    out = {}
    for name, idx in picks.items():
        m = A.plan_metrics(z["fan"][ar, idx].astype(np.float64), z, d)
        row = {"rule": name, "param": None, "pick_changed_frac": round(float((idx != z["sel_idx"]).mean()), 4)}
        for cn in ("turn", "straight", "gentle", "all"):
            msk = A.cls_mask(d, cn)
            blk = {"n": int(msk.sum())}
            for k, v in m.items():
                pt, bs, _ = rm.boot_mean(v, d["ep"], msk, draws=draws)
                ent = {"mean": None if not np.isfinite(pt) else round(float(pt), 4), "ci95": rm.ci95(bs)}
                if name != "V0":
                    pt2, bs2, _ = rm.boot_mean(v - base[k], d["ep"], msk, draws=draws)
                    ent["delta_vs_V0"] = None if not np.isfinite(pt2) else round(float(pt2), 4)
                    ent["delta_ci95"] = rm.ci95(bs2)
                blk[k] = ent
            row[cn] = blk
        out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.data)
    zt = A.load(D / "train_s0.npz")
    dt = A.derive(zt)
    dt["_z"] = zt
    Xt = features(zt, dt)
    y, ok = oracle_reach(dt)
    mu, sd = standardise(Xt, dt["reach"])
    res = {"features": FEATS, "alphas": list(ALPHAS), "n_train_windows_fitted": int(ok.sum())}
    models = {}
    for arm, names in ARMS.items():
        cols = [FEATS.index(n) for n in names]
        m, info = fit_arm(cols, Xt, dt, y, ok, mu, sd)
        models[arm] = m
        res[f"fit_{arm}"] = info
        print(f"[A2] {arm}: {json.dumps({k: v for k, v in info.items() if k != 'theta'})}", flush=True)
    # shuffled-target control (X1 columns, X1's alpha)
    rng = np.random.default_rng(7)
    ysh = np.array([rng.choice(np.nonzero(dt["reach"][w])[0]) for w in range(dt["W"])])
    cols1 = [FEATS.index(n) for n in ARMS["X1"]]
    msh = CLogit(cols1, res["fit_X1"]["alpha"]).fit(Xt, dt["reach"], ysh, ok, mu, sd)
    models["CTRL_shuffled_target"] = msh
    res["fit_CTRL_shuffled_target"] = {"theta": {FEATS[c]: round(float(t), 4) for c, t in zip(cols1, msh.theta)},
                                      "train_in_sample_ade": round(mean_ade(dt, msh.pick(Xt, dt["reach"])), 4)}
    for tag in ("eval_s0g", "eval_s1"):
        p = D / f"{tag}.npz"
        if not p.exists():
            continue
        ze = A.load(p)
        de = A.derive(ze)
        if "g_tac" not in ze or "prior_path" not in ze:
            raise SystemExit(f"{tag} lacks the g_tac / prior capture")
        draws = rm.make_draws(de["ep"], B=A.B, seed=A.SEED)
        Xe = features(ze, de)
        picks = {"V0": ze["sel_idx"]}
        for arm, m in models.items():
            picks[arm] = m.pick(Xe, de["reach"])
        picks["ORACLE"] = de["oracle"]
        lt = table_for(ze, de, picks, draws)
        res[f"levers_{tag}"] = lt
        res[f"bar_{tag}"] = {arm: A.bar_check(lt, arm) for arm in picks if arm not in ("V0", "ORACLE")}
        print(f"[A2] {tag}: " + " | ".join(
            f"{k} all {lt[k]['all']['ade']['mean']} turn {lt[k]['turn']['ade']['mean']} "
            f"str {lt[k]['straight']['ade']['mean']}" for k in lt), flush=True)
    if "levers_eval_s0g" in res and "levers_eval_s1" in res:
        res["bar_X1_with_replicate"] = A.bar_check(res["levers_eval_s0g"], "X1", res["levers_eval_s1"])
    Path(a.out).write_text(json.dumps(A.sanitize(res), indent=1, allow_nan=False), encoding="utf-8")
    print("[A2] wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
