"""SPEC_ADDENDUM_A3: the nonlinear (gradient-boosted) pointwise re-scorer. Fit on train_s0 only; scored on eval_s0g
and eval_s1 with the SAME metrics / bar as SPEC §5 (via rescorer_a2.table_for and analyze_route.bar_check)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_route as A  # noqa: E402
import rescorer_a2 as A2  # noqa: E402
import route_metrics as rm  # noqa: E402

EXTRA = ["v0", "rank_s_e9", "rank_sampler_conf"]
NAMES = A2.FEATS + EXTRA
ARMS = {"Y1": NAMES, "Y2_no_nav": [n for n in NAMES if n not in ("navc", "agree_nav_side")]}
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False,
           random_state=0)


def feats(z, d):
    X = A2.features(z, d)
    W, N, _ = X.shape
    v0 = np.repeat(z["v0"][:, None], N, 1)
    r1 = np.argsort(np.argsort(-np.where(d["reach"], z["s_e9"], -np.inf), 1), 1).astype(float)
    r2 = np.argsort(np.argsort(-np.where(d["reach"], z["refined_base"], -np.inf), 1), 1).astype(float)
    return np.concatenate([X, v0[..., None], r1[..., None], r2[..., None]], -1)


def rows(X, d, ok):
    keep = d["reach"] & ok[:, None] & np.isfinite(d["ade_c"])
    tgt = d["ade_c"] - np.where(keep, d["ade_c"], np.inf).min(1, keepdims=True)
    return X[keep], tgt[keep]


def pick(model, X, d, cols):
    W, N, F = X.shape
    pred = model.predict(X[..., cols].reshape(-1, len(cols))).reshape(W, N)
    return A.masked_argmax(-pred, d["reach"], d["reach"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.data)
    zt = A.load(D / "train_s0.npz")
    dt = A.derive(zt)
    Xt = feats(zt, dt)
    ok = np.isfinite(np.where(dt["reach"], dt["ade_c"], np.nan)).any(1)
    res = {"features": NAMES, "hgb": HGB}
    models = {}
    eps = np.unique(dt["ep"])
    perm = np.random.default_rng(0).permutation(len(eps))
    fold_of = {eps[p]: i % 5 for i, p in enumerate(perm)}
    folds = np.array([fold_of[e] for e in dt["ep"]])
    for arm, names in ARMS.items():
        cols = [NAMES.index(n) for n in names]
        cvp = np.zeros(dt["W"], np.int64)
        for k in range(5):
            tr = folds != k
            sub = {kk: (v[tr] if isinstance(v, np.ndarray) and v.shape[:1] == (dt["W"],) else v) for kk, v in dt.items()}
            Xr, yr = rows(Xt[tr][..., cols], sub, ok[tr])
            m = HistGradientBoostingRegressor(**HGB).fit(Xr, yr)
            subk = {kk: (v[~tr] if isinstance(v, np.ndarray) and v.shape[:1] == (dt["W"],) else v)
                    for kk, v in dt.items()}
            cvp[~tr] = pick(m, Xt[~tr], subk, cols)
        Xr, yr = rows(Xt[..., cols], dt, ok)
        m = HistGradientBoostingRegressor(**HGB).fit(Xr, yr)
        models[arm] = (m, cols)
        res[f"fit_{arm}"] = {"cv_ade": round(A2.mean_ade(dt, cvp), 4),
                             "train_in_sample_ade": round(A2.mean_ade(dt, pick(m, Xt, dt, cols)), 4),
                             "train_V0_ade": round(A2.mean_ade(dt, zt["sel_idx"]), 4), "n_rows": int(len(yr))}
        print(f"[A3] {arm}: {res[f'fit_{arm}']}", flush=True)
    for tag in ("eval_s0g", "eval_s1"):
        p = D / f"{tag}.npz"
        if not p.exists():
            continue
        ze = A.load(p)
        de = A.derive(ze)
        draws = rm.make_draws(de["ep"], B=A.B, seed=A.SEED)
        Xe = feats(ze, de)
        picks = {"V0": ze["sel_idx"]}
        for arm, (m, cols) in models.items():
            picks[arm] = pick(m, Xe, de, cols)
        picks["ORACLE"] = de["oracle"]
        lt = A2.table_for(ze, de, picks, draws)
        res[f"levers_{tag}"] = lt
        res[f"bar_{tag}"] = {arm: A.bar_check(lt, arm) for arm in picks if arm not in ("V0", "ORACLE")}
        print(f"[A3] {tag}: " + " | ".join(
            f"{k} all {lt[k]['all']['ade']['mean']} turn {lt[k]['turn']['ade']['mean']} "
            f"str {lt[k]['straight']['ade']['mean']}" for k in lt), flush=True)
    if "levers_eval_s0g" in res and "levers_eval_s1" in res:
        res["bar_Y1_with_replicate"] = A.bar_check(res["levers_eval_s0g"], "Y1", res["levers_eval_s1"])
    Path(a.out).write_text(json.dumps(A.sanitize(res), indent=1, allow_nan=False), encoding="utf-8")
    print("[A3] wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
