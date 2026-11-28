"""Fast unit tests for rescurve.py on synthetic data (no cache). Run: python -m pytest -q test_rescurve.py -p no:cacheprovider"""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rescurve as W
R = W.R


def _data(n=90, d=40, n_ep=18, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    ep = np.repeat(np.arange(n_ep), n // n_ep)
    y = X[:, 0] * 2 + 0.3 * rng.standard_normal(n)
    return X, y, ep


def test_full_kernel_equals_rank_n_ridge():
    X, y, ep = _data()
    K = X @ X.T
    n = len(K)
    Kc, cm, gm = W._centre(K)
    lam, V = np.linalg.eigh(Kc)
    keep = lam > 1e-9 * lam.max()                     # all n-1 nonzero components = the full rank
    Z = V[:, keep] * np.sqrt(lam[keep])
    d = Z.shape[1]
    pred_k, vm_k, vv_k = W.krr_fit(K, y, ep, d=d)
    pred_p, vm_p, vv_p = W.fit_p1x(Z, y, ep)
    assert abs(vm_k - vm_p) < 1e-8 * max(1.0, abs(vm_p)) and abs(vv_k - vv_p) < 1e-12
    Xt = np.random.default_rng(5).standard_normal((20, X.shape[1]))
    Kx = Xt @ X.T
    Zt = W._centre_x(Kx, cm, gm) @ V[:, keep] / np.sqrt(lam[keep])
    assert np.allclose(pred_k(Kx), pred_p(Zt), atol=1e-6)


def test_ego_block_in_kernel_equals_concatenated_features():
    X, y, ep = _data(seed=2)
    E = np.random.default_rng(3).standard_normal((len(X), 4))
    sv = 2.5
    Eb = E / E.std(0) * np.sqrt(sv)
    K = X @ X.T + Eb @ Eb.T
    Kc, cm, gm = W._centre(X @ X.T)
    lam, V = np.linalg.eigh(Kc)
    keep = lam > 1e-9 * lam.max()
    Z = np.concatenate([V[:, keep] * np.sqrt(lam[keep]), Eb - Eb.mean(0)], 1)
    pred_k, _, _ = W.krr_fit(K, y, ep, d=Z.shape[1])
    pred_p, _, _ = W.fit_p1x(Z, y, ep)
    Zt = Z[:15]
    # the kernel prediction on training rows equals the primal prediction on training rows
    assert np.allclose(pred_k(K[:15]), pred_p(Zt), atol=1e-6)


def test_fit_p1x_matches_resprobe_fit_p1():
    X, y, ep = _data(seed=4)
    a = W.fit_p1x(X, y, ep)[0](X)
    b = R.fit_p1(X, y, ep)(X)
    assert np.array_equal(a, b)


def test_nested_subsets_nested_and_disjoint_from_heldout():
    eps = np.arange(100)
    folds = {e: e % 5 for e in eps}
    for f in range(5):
        tr_eps = np.array([e for e in eps if folds[e] != f])
        sub = W.nested_subsets(tr_eps, W.FRACS + (1.0,), seed=11 + f)
        sizes = [len(sub[x]) for x in sorted(sub)]
        assert sizes == sorted(sizes) and sizes[-1] == len(tr_eps)
        order = sorted(sub)
        for a, b in zip(order[:-1], order[1:]):
            assert set(sub[a]) <= set(sub[b])
        for v in sub.values():
            assert not (set(v) & {e for e in eps if folds[e] == f})
        assert len(sub[0.125]) == round(0.125 * len(tr_eps))
    # a different draw gives a different subset
    assert not np.array_equal(W.nested_subsets(tr_eps, (0.25,), 1)[0.25], W.nested_subsets(tr_eps, (0.25,), 2)[0.25])


def test_selection_budget_whole_episodes_label_free():
    rng = np.random.default_rng(0)
    n_ep = 30
    ep = np.repeat(np.arange(n_ep), 7)
    Z = rng.standard_normal((len(ep), 6)) + (ep % 3)[:, None]
    for rule in ("random", "kcentre", "leverage"):
        sel = W.select_episodes(rule, Z, ep, 8, seed=3)
        assert len(sel) == 8 and len(set(sel)) == 8 and set(sel) <= set(range(n_ep))
        tr = np.isin(ep, sel)
        assert tr.sum() == 8 * 7                                    # whole episodes: every frame of a chosen episode, none of any other
        assert np.array_equal(sel, W.select_episodes(rule, Z, ep, 8, seed=3))  # deterministic
    # labels cannot enter: the signature takes none, and the output is invariant to anything but Z, ep, budget, seed
    import inspect
    assert [p for p in inspect.signature(W.select_episodes).parameters] == ["rule", "Z", "ep_tr", "budget", "seed"]
    # k-centre spreads: with 3 well separated clusters and budget 3 it picks one episode per cluster
    sel = W.select_episodes("kcentre", Z * 0 + (ep % 3)[:, None] * 50.0 + 0.01 * rng.standard_normal(Z.shape), ep, 3)
    assert sorted(set(sel % 3)) == [0, 1, 2]
    # leverage: an episode with an extreme direction ranks first
    Z2 = rng.standard_normal((len(ep), 6)); Z2[ep == 5] *= 6.0
    assert 5 in W.select_episodes("leverage", Z2, ep, 1)


def test_last_doubling_gain_is_pairwise_on_same_frames():
    rng = np.random.default_rng(7)
    n = 300
    epi = np.repeat(np.arange(30), 10)
    y = rng.standard_normal(n)
    p_full = y + 0.5 * rng.standard_normal(n)
    halves = [y + 0.9 * rng.standard_normal(n) for _ in range(5)]
    Cb = R.boot_counts(30, 400)
    g = W.last_doubling_gain("r2", epi, y, p_full, halves, Cb, 30)
    manual = np.mean([R.r2_pooled(y, p_full) - R.r2_pooled(y, h) for h in halves])
    assert abs(g["diff"] - manual) < 1e-12
    # bootstrap interval from identical resamples: shifting every half prediction toward the truth lowers the gain
    g2 = W.last_doubling_gain("r2", epi, y, p_full, [p_full] * 5, Cb, 30)
    assert g2["diff"] == 0.0 and g2["ci95"] == [0.0, 0.0]
    assert g["ci95"][0] < g["diff"] < g["ci95"][1]
    ga = W.last_doubling_gain("auc", epi, (y > 0).astype(float), p_full, halves, Cb, 30)
    assert abs(ga["diff"] - np.mean([R.auc_pooled((y > 0).astype(float), p_full) - R.auc_pooled((y > 0).astype(float), h) for h in halves])) < 1e-12
