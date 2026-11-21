"""s7b_panel.py -- which cell kinds' T error does O's ADE follow? A within-window panel across the 7 M2b-family arms (natural variation in T's error, EXPLORATORY).
For each window w and arm a: y = ADE6_a(w) - mean_a ADE6_a(w);  x_k = SSE_k,a(w) - mean_a SSE_k,a(w) (k = cell kind, present cells, SD^2 summed in the window).
Pooled OLS without intercept, episode-clustered bootstrap (B=2000) for the interval. CONFOUNDED: the arms differ in O too, so this is an association (not the
causal oracle read); it is a cross-check on the NNLS read of s4/s7, nothing more. The 'seed' row (s0,s1 only) is the cleanest pair (same recipe).
"""
import numpy as np

from tcommon import *
from s3_tail import KIND


def fit(Y, X, eid, ue, nb=2000):
    # Y [n], X [n,k] already demeaned within window; cluster bootstrap over episodes
    b0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    idx_by = [np.where(eid == e)[0] for e in ue]
    r = np.random.default_rng(0)
    bs = np.empty((nb, X.shape[1]))
    for i in range(nb):
        k = r.integers(0, len(ue), len(ue))
        ii = np.concatenate([idx_by[j] for j in k])
        bs[i] = np.linalg.lstsq(X[ii], Y[ii], rcond=None)[0]
    return b0, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def main():
    T = load()
    has = T["has"]
    eid = T["ci"]
    kinds = sorted(set(KIND.values()))
    for label, arms in (("7 M2b-family arms", M2B_FAMILY), ("seed pair s0,s1", ["M2b_s0", "M2b_s1"])):
        ade = np.stack([T[f"ade6_{a}"] for a in arms])                        # [A, N]
        X = []
        for k in kinds:
            idx = [i for i in KIND if KIND[i] == k]
            X.append(np.stack([sq(T, a)[0][:, idx].sum(1) for a in arms]))   # [A, N]
        X = np.stack(X, -1)                                                   # [A, N, K]
        Yd = ade - ade.mean(0, keepdims=True)
        Xd = X - X.mean(0, keepdims=True)
        m = np.broadcast_to(has[None, :], Yd.shape)
        Y = Yd[m]
        XX = Xd[m]
        e = np.broadcast_to(eid[None, :], Yd.shape)[m]
        b, lo, hi = fit(Y, XX, e, np.unique(eid))
        # standardised effect: coefficient x the cross-arm SD of the regressor (m of ADE per 1 cross-arm SD of that kind's SSE)
        sdx = XX.std(0)
        print(f"\n[{label}] n={len(Y)} (arm,window) rows; ADE6 within-window SD across arms {Y.std():.3f} m")
        print(f"   {'kind':40s} {'coef m/SD^2':>11s} {'[CI95]':>18s} {'x 1 cross-arm SD of the regressor (m)':>40s}")
        pred = XX @ b
        for j, k in enumerate(kinds):
            print(f"   {k:40s} {b[j]:11.4f} [{lo[j]:7.4f},{hi[j]:7.4f}] {b[j] * sdx[j]:40.4f}")
        print(f"   R2 of within-window ADE variation explained by the six kinds' SSE: {1 - ((Y - pred) ** 2).sum() / (Y ** 2).sum():.3f}")


if __name__ == "__main__":
    main()
