"""Unit tests for wmceil.py's pure statistics (no checkpoints, CPU). Run: PYTHONPATH=<tree>/stack python -m pytest -q test_wmceil.py --basetemp=<existing dir>/bt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmceil as W


def _ar1(n_ep, rho, D, T, seed):
    rng = np.random.default_rng(seed)
    z = np.zeros((n_ep, T, D)); z[:, 0] = rng.standard_normal((n_ep, D))
    for t in range(1, T):
        z[:, t] = rho * z[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal((n_ep, D))
    return z


def test_ar1_ceiling_matches_closed_form_on_synthetic_ar1():
    rho = 0.7
    z = _ar1(6000, rho, 8, 2, 0)                      # lag-1 pairs of a stationary AR(1)
    z0, z1 = z[:, 0], z[:, 1]
    r = W.lag_corr(z0, z1).mean()
    assert abs(r - rho) < 0.03
    eid = np.arange(len(z0)); fold = W.fold_ids(eid)
    pred, _ = W.cross_fit(z0, z1 - z0, eid.astype(str), fold)
    sk = W.skill_from_err(W.sqerr(z0 + pred, z1), W.sqerr(z0, z1))
    assert abs(sk - W.ar1_ceiling(rho)) < 0.02, (sk, W.ar1_ceiling(rho))      # (1 - 0.7) / 2 = 0.15
    assert abs(W.ar1_ceiling(r) - 0.15) < 0.02


def test_ridge_recovers_known_linear_map():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4000, 12)); Wt = rng.standard_normal((12, 5)); b = rng.standard_normal(5)
    Y = X @ Wt + b + 0.01 * rng.standard_normal((4000, 5))
    eid = np.arange(4000).astype(str); fold = W.fold_ids(eid)
    pred, lam = W.cross_fit(X, Y, eid, fold)
    r2 = 1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    assert r2 > 0.999, r2
    rp = W.RidgePath(X, Y)
    assert np.abs(rp.predict(X, 1e-6) - Y).max() < 0.1


def test_deliberate_regression_shuffled_targets_read_zero_not_positive():
    rng = np.random.default_rng(2)
    z0 = rng.standard_normal((3000, 10))
    zk = z0 * 0.8 + 0.6 * rng.standard_normal((3000, 10))
    eid = np.arange(3000).astype(str); fold = W.fold_ids(eid)
    p, _ = W.cross_fit(z0, zk - z0, eid, fold)
    real = W.skill_from_err(W.sqerr(z0 + p, zk), W.sqerr(z0, zk))
    assert real > 0.08                                                          # the real pairing carries predictable change
    d = zk - z0
    zs = z0 + d[rng.permutation(3000)]                                          # the CHANGE shuffled across windows: no relation to z0 remains
    ps, _ = W.cross_fit(z0, zs - z0, eid, fold)
    sh = W.skill_from_err(W.sqerr(z0 + ps, zs), W.sqerr(z0, zs))
    assert -0.02 < sh < 0.01, sh                                                # near 0, never positive beyond sampling noise


def test_lambda_never_chosen_on_scored_fold():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((600, 6)); Y = rng.standard_normal((600, 3))
    eid = np.arange(600).astype(str); fold = W.fold_ids(eid)
    p1, l1 = W.cross_fit(X, Y, eid, fold)
    Y2 = Y.copy(); Y2[fold == 0] += 100.0                                        # corrupt only fold 0's targets
    p2, l2 = W.cross_fit(X, Y2, eid, fold)
    assert np.allclose(p1[fold == 0], p2[fold == 0])                             # fold 0's predictions come from the other folds only
