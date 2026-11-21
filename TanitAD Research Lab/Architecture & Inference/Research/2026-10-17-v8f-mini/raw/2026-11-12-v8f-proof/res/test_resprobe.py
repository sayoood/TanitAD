import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resprobe as R


def test_folds_no_episode_in_two_folds():
    eid = np.repeat([f"e{i:03d}" for i in range(139)], 7)
    f = R.make_folds(eid)
    for e in set(eid):
        assert len(set(f[eid == e])) == 1
    assert set(f) == set(range(5))
    sizes = [len(set(eid[f == k])) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert (R.make_folds(eid) == f).all()          # fixed seed


def test_bootstrap_recovers_known_difference():
    rng = np.random.default_rng(1)
    n_ep, per = 60, 40
    ep = np.repeat(np.arange(n_ep), per)
    y = rng.standard_normal(len(ep)) * 2
    pa = y + rng.standard_normal(len(ep)) * 0.5      # R2 = 1 - .25/4 = 0.9375
    pb = y + rng.standard_normal(len(ep)) * 1.5      # R2 = 1 - 2.25/4 = 0.4375
    C = R.boot_counts(n_ep, 2000)
    A = R.boot_r2(R.ep_stats_r2(ep, y, pa, n_ep), C); B = R.boot_r2(R.ep_stats_r2(ep, y, pb, n_ep), C)
    row = R.diff_row(A, B, R.r2_pooled(y, pa), R.r2_pooled(y, pb))
    assert abs(row["diff"] - 0.5) < 0.06 and row["separated_pos"] and row["ci95"][0] < 0.5 < row["ci95"][1] + 0.1
    same = R.diff_row(A, A, 0.9, 0.9)
    assert same["ci95"] == [0.0, 0.0] and not same["separated_pos"]
    # AUROC bootstrap agrees with the pooled AUROC at the identity draw
    yb = (rng.random(len(ep)) < 0.4).astype(float); s = yb * 1.0 + rng.standard_normal(len(ep))
    Cid = np.ones((1, n_ep))
    assert abs(R.boot_auc(ep, yb, s, Cid)[0] - R.auc_pooled(yb, s)) < 1e-9


def test_control_k2_synthetic_linear_target_reads_high():
    rng = np.random.default_rng(3)
    n, d = 600, 300
    spec = 1.0 / (1 + np.arange(d)) ** 1.2
    X = rng.standard_normal((n, d)) * np.sqrt(spec)[None]
    K = X @ X.T
    a = rng.standard_normal(n); y = K @ a                       # fixed linear function of the tokens
    ep = np.repeat(np.arange(30), 20).astype(str)
    f = R.make_folds(ep)
    oof = np.full(n, np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        proj, _ = R.kpca_fit(K, tr, rank=64)
        fit = R.fit_p1(proj(tr), y[tr], ep[tr]); oof[te] = fit(proj(te))
    assert R.r2_pooled(y, oof) >= 0.95
    # shuffled-target control reads ~0 on the same pipeline
    ysh = y.copy()
    for k in range(5):
        te = np.flatnonzero(f == k); ysh[te] = y[np.random.default_rng(k).permutation(te)]
    oof2 = np.full(n, np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        proj, _ = R.kpca_fit(K, tr, rank=64)
        oof2[te] = R.fit_p1(proj(tr), ysh[tr], ep[tr])(proj(te))
    assert R.r2_pooled(ysh, oof2) <= 0.02
