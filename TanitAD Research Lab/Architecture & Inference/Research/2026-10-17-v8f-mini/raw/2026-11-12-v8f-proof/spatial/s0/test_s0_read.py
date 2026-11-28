"""Unit tests of s0_read.py on SYNTHETIC data only (no real token, no real label)."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

import s0_read as S

D_TOK, NC, K_CELL, N_MEAN = 64, 40, 4, 8


def synth_tokens(n=400, planted_cell=7, signal=1.0, seed=0):
    """Tokens [n, 40, D] of noise; a cell-local object (a fixed direction, amplitude `signal` * sigma * sqrt(D)) in `planted_cell` for half of the rows."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, NC, D_TOK)).astype(np.float32)
    v = rng.standard_normal(D_TOK).astype(np.float32)
    v = v / np.linalg.norm(v) * np.sqrt(D_TOK) * signal
    flag = (rng.random(n) < 0.5).astype(np.float32)
    X[:, planted_cell, :] += flag[:, None] * v[None, :]
    return X, flag


def episodes(n, n_ep=20):
    return np.array([f"e{i % n_ep}" for i in range(n)])


def states(X, tr):
    M = S.mean_token(X)
    return S.grid_state(X, tr, k=K_CELL), S.mean_state(M, tr, N_MEAN)


def oof_auc(Sx, y, ep, folds):
    p = S.fold_cv_probe(Sx, y, np.ones(len(y), bool), ep, folds)
    return S.RP.auc_pooled(y, p)


def test_planted_cell_object_found_by_grid_lost_by_mean():
    X, flag = synth_tokens(signal=0.5)
    ep = episodes(len(X)); folds = S.RP.make_folds(ep)
    ag, am = [], []
    # per-fold states (PCA on training folds only), pooled out-of-fold AUROC
    pg, pm = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for f in range(5):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        Sg, Sm = states(X, tr)
        epi = np.array([int(e[1:]) for e in ep])
        pg[te] = S.fit_probe(Sg[tr], flag[tr].astype(float), epi[tr], f)(Sg[te])
        pm[te] = S.fit_probe(Sm[tr], flag[tr].astype(float), epi[tr], f)(Sm[te])
    assert S.RP.auc_pooled(flag, pg) > 0.95
    assert S.RP.auc_pooled(flag, pm) <= 0.60


def test_horizon_zero_returns_present_state_exactly():
    rng = np.random.default_rng(1)
    Z0 = rng.standard_normal((120, 12)); A = rng.standard_normal((120, S.NACT)); ep = np.array([i % 10 for i in range(120)])
    fn = S.fit_state_predictor(Z0, Z0, A, ep, steps=0)
    Zq = Z0[:30].astype(np.float32)
    assert float(np.abs(fn(Zq, A[:30]) - Zq).max()) == 0.0


def test_predictor_uses_actions():
    """With a change that is a linear function of the actions the ridge predictor beats the copy."""
    rng = np.random.default_rng(2)
    n = 400
    Z0 = rng.standard_normal((n, 6)); A = rng.standard_normal((n, S.NACT)); ep = np.array([i % 20 for i in range(n)])
    Zk = Z0 + A[:, :6] * 0.8
    fn = S.fit_state_predictor(Z0[:300], Zk[:300], A[:300], ep[:300])
    err_pred = ((fn(Z0[300:], A[300:]) - Zk[300:]) ** 2).sum()
    err_copy = ((Z0[300:] - Zk[300:]) ** 2).sum()
    assert err_pred < 0.1 * err_copy


def test_control_s1_fires_on_planted_violation():
    """A label that is NOT permuted (a planted signal the probe can read) must trip the s1 bar."""
    rng = np.random.default_rng(3)
    n = 300
    X = rng.standard_normal((n, 10)); ep = np.array([i % 20 for i in range(n)]); folds = S.RP.make_folds(np.array([f"e{e}" for e in ep]))
    y_true = X[:, 0] * 2.0 + 0.1 * rng.standard_normal(n)
    r2_bad = S.RP.r2_pooled(y_true, S.fold_cv_probe(X, y_true, np.ones(n, bool), ep, folds))
    y_perm = S.permute_within_fold(y_true, folds, 1)
    r2_ok = S.RP.r2_pooled(y_perm, S.fold_cv_probe(X, y_perm, np.ones(n, bool), ep, folds))
    assert not S.check_s1({"unpermuted": r2_bad})["pass"]
    assert S.check_s1({"permuted": r2_ok})["pass"]
    assert sorted(y_perm) == sorted(y_true)


def test_control_s2_s3_s4_fire_on_violations():
    assert S.check_s2(0.99, 0.50)["pass"]
    assert not S.check_s2(0.99, 0.90)["pass"]           # the mean state reads the object
    assert not S.check_s2(0.80, 0.50)["pass"]           # the grid does not
    ok = S.check_s3({"g": 0.3}, {"g": 0.2}, {"g": 0.5}, 0.14)
    assert ok["pass"]
    assert not S.check_s3({"g": 0.3}, {"g": 0.2}, {"g": 0.25}, 0.14)["pass"]   # ceiling below (i)
    assert not S.check_s3({"g": 0.3}, {"g": 0.2}, {"g": 0.5}, 0.20)["pass"]    # ego floor off FULLRES E by more than 0.03
    assert S.check_s4({"S0g": 0.0, "S0m": 0.0})["pass"]
    assert not S.check_s4({"S0g": 1e-7, "S0m": 0.0})["pass"]


def test_outcome_names_follow_the_card():
    sep = {"separated_pos": True}; no = {"separated_pos": False}
    o = S.outcome_names({"g": sep, "m": no}, {"g": sep, "m": sep}, {"g": sep, "m": no}, sep)
    assert o["S0g"] == {"lead": "S0-LEAD-KEPT", "imagine": "S0-IMAGINE-LEAD"}
    assert o["S0m"] == {"lead": "S0-LEAD-LOST", "imagine": "S0-IMAGINE-EGO"}
    assert o["headline"] == "GRID-ADDS"
    o = S.outcome_names({"g": no, "m": no}, {"g": no, "m": no}, {"g": no, "m": no}, no)
    assert o["S0g"]["imagine"] == "S0-IMAGINE-FLAT" and o["headline"] == "GRID-EQUAL"


def pca_is_fold_restricted() -> bool:
    """True when the basis of a fold does not depend on the test rows (R5): perturb the test rows wildly, the training-fold state basis must not move."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 4, 16)).astype(np.float32)
    tr = np.arange(0, 150); te = np.arange(150, 200)
    a = S.cell_block(X, tr, 1, 3)[tr]
    X2 = X.copy(); X2[te] = X2[te] * 50.0 + 7.0
    b = S.cell_block(X2, tr, 1, 3)[tr]
    return bool(np.array_equal(a, b))


def test_pca_fit_is_restricted_to_training_folds():
    assert pca_is_fold_restricted()


def test_mutation_removing_fold_restriction_is_caught(monkeypatch):
    monkeypatch.setattr(S, "LEAK_PCA_FIT", True)
    assert not pca_is_fold_restricted()                  # the guard test goes red under the leak


def test_probe_ridge_and_krr_selectable():
    rng = np.random.default_rng(5)
    n = 500
    X = rng.standard_normal((n, 5)); ep = np.array([i % 25 for i in range(n)])
    y = np.sin(2.5 * X[:, 0]) * 2 + 0.05 * rng.standard_normal(n)                  # nonlinear: kernel ridge should win
    p = S.fit_probe(X[:400], y[:400], ep[:400])(X[400:])
    assert S.RP.r2_pooled(y[400:], p) > 0.5


def test_grid_state_cell_layout():
    X, _ = synth_tokens(n=60)
    tr = np.arange(40)
    g = S.grid_state(X, tr, k=K_CELL)
    assert g.shape == (60, NC * K_CELL)
    assert np.allclose(g[:, 3 * K_CELL:4 * K_CELL], S.cell_block(X, tr, 3, K_CELL))
