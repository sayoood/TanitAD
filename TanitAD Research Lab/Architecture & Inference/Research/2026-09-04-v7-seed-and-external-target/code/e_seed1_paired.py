"""E-SEED-1 paired deltas — the estimator the claims actually need.

Per-arm CIs answer "does this arm beat zero". The claims are DIFFERENCES
("the repaired seed beats the as-wired seed", "the as-wired seed does not beat
the random floor"), and for two arms scored on the SAME windows the estimator is
the PAIRED episode-cluster bootstrap, never a combination in quadrature
(CLAUDE.md, `taniteval/ci.py` doctrine).

Both arms' predictions are recomputed on the identical scored rows, then the
SAME resampled episode set is applied to both inside every draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE / "eseed1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "clone_scripts"))
import e_seed1_v2 as V2                                       # noqa: E402

PAIRS = [
    ("seed_imnet", "seed_asis", "does repairing the input normalisation help?"),
    ("seed_asis", "scratch", "is the seed, AS WIRED, better than random init?"),
    ("seed_imnet", "scratch", "is the REPAIRED seed better than random init?"),
    ("dino_hf", "seed_imnet", "how much does the repaired seed still lose?"),
    ("dino_hf", "scratch", "INSTRUMENT VALIDITY: can the probe see a good rep?"),
    ("seed_imnet", "seed_imnet_pos0", "does the random `pos` table help or hurt?"),
]
N_BOOT = 2000


def preds(arm, pool, y, ep, fit, sc):
    F = np.load(BANK / (f"feat_{arm}.npy" if pool == "global"
                        else f"featsp_{arm}.npy"))
    Z0 = F[0::2]
    Ffit, Fsc = Z0[fit], Z0[sc]
    mu = Ffit.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ffit - mu, full_matrices=False)
    V = Vt[:V2.PCA_K].T
    Zf, Zs = (Ffit - mu) @ V, (Fsc - mu) @ V
    sd = Zf.std(0, keepdims=True) + 1e-8
    Zf, Zs = (Zf / sd).astype(np.float64), (Zs / sd).astype(np.float64)
    ym, ys = y[fit].mean(), y[fit].std() + 1e-8
    yf = ((y[fit] - ym) / ys).astype(np.float64)
    eps = np.unique(ep[fit])
    folds = np.array_split(eps, 5)
    best, lam = -np.inf, None
    for L in V2.LAMBDAS:
        s = []
        for f in folds:
            m = np.isin(ep[fit], f)
            W = np.linalg.solve(Zf[~m].T @ Zf[~m] + L * np.eye(Zf.shape[1]),
                                Zf[~m].T @ yf[~m])
            s.append(float(V2.r2(yf[m], Zf[m] @ W)))
        if np.mean(s) > best:
            best, lam = float(np.mean(s)), L
    W = np.linalg.solve(Zf.T @ Zf + lam * np.eye(Zf.shape[1]), Zf.T @ yf)
    return (Zs @ W) * ys + ym, lam


def main() -> int:
    L = np.load(BANK / "labels.npy")
    ep = np.load(BANK / "rows.npy")[:, 0]
    fit, sc = ep < V2.N_FIT_EP, ep >= V2.N_FIT_EP
    y = L[:, 0]                                   # speed -- the clean read
    ysc, gsc = y[sc], ep[sc]
    ue = np.unique(gsc)
    idx = {e: np.where(gsc == e)[0] for e in ue}
    out = {"target": "v_mps (speed)", "estimator":
           "PAIRED episode-cluster bootstrap, 2000 draws, 30 scored episodes",
           "evidence_class": "MEASURED (ours)", "pairs": {}}
    cache = {}
    for pool in ("global", "spatial4x4"):
        for a, b, q in PAIRS:
            for arm in (a, b):
                if (arm, pool) not in cache:
                    cache[(arm, pool)] = preds(arm, pool, y, ep, fit, sc)
            ya, la = cache[(a, pool)]
            yb, lb = cache[(b, pool)]
            ra, rb = float(V2.r2(ysc, ya)), float(V2.r2(ysc, yb))
            rng = np.random.default_rng(0)
            d = np.empty(N_BOOT)
            for k in range(N_BOOT):
                ii = np.concatenate([idx[e]
                                     for e in rng.choice(ue, len(ue), True)])
                d[k] = V2.r2(ysc[ii], ya[ii]) - V2.r2(ysc[ii], yb[ii])
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            out["pairs"][f"{pool}::{a}-{b}"] = {
                "question": q, "r2_a": ra, "r2_b": rb, "delta": ra - rb,
                "delta_ci95": [lo, hi], "separated": bool(lo > 0 or hi < 0),
                "lambda_a": la, "lambda_b": lb,
                "n_scored": int(sc.sum()), "n_clusters": int(len(ue))}
            print(f"{pool:11s} {a:16s} - {b:16s} "
                  f"{ra:+.4f} - {rb:+.4f} = {ra-rb:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] "
                  f"{'SEPARATED' if (lo>0 or hi<0) else 'not separated'}",
                  flush=True)
    (BANK / "eseed1_paired.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", BANK / "eseed1_paired.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
