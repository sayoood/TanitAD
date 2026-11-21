"""Unit tests for tp_lib (PREREG_FROZEN.md section 8, C4).  Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q test_trigger_probe.py   (from this directory; needs numpy + sklearn)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_lib as L  # noqa: E402


def test_weighted_auc_equals_sklearn_at_unit_weights_and_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + 0.7 * y, 1)                     # heavy ties
    assert abs(L.WAuc(s, y).auc() - roc_auc_score(y, s)) < 1e-12
    w = rng.integers(0, 4, 500).astype(float)                             # integer multiplicities == replicated rows
    rep = np.repeat(np.arange(500), w.astype(int))
    assert abs(L.WAuc(s, y).auc(w) - roc_auc_score(y[rep], s[rep])) < 1e-12


def test_go_label_matches_mlaunch_python_loop():
    rng = np.random.default_rng(1)
    g = np.cumsum(rng.normal(0.4, 0.3, (200, 30, 2)), 1)
    v0 = rng.uniform(0, 8, 200)
    got = L.go_label(v0, g, 29)
    ref = []                                                              # mlaunch.launch_subset_loop's spelling, verbatim
    for i in range(200):
        dx, dy = float(g[i][29][0]) - float(g[i][28][0]), float(g[i][29][1]) - float(g[i][28][1])
        v6 = math.sqrt(dx * dx + dy * dy) / 0.2
        v = float(v0[i])
        ref.append(bool(v == v and v < 6.0 and v6 >= v + 1.0))
    assert np.array_equal(got, np.array(ref))
    assert got.sum() > 5 and (~got).sum() > 5                             # the test is not vacuous
    g2 = g.copy()
    g2[:, 29] = g2[:, 28]                                                 # a standstill end: nobody launches
    assert not L.go_label(v0, g2, 29).any()


def test_folds_never_split_a_clip_and_are_stable():
    shas = [f"{i:012x}" for i in range(0, 5000, 7)]
    f = L.outer_fold(shas)
    assert set(f.tolist()) == {0, 1, 2, 3, 4}
    assert np.array_equal(f, L.outer_fold(shas))                         # a pure function of the clip's own sha12
    assert set(L.inner_fold(shas).tolist()) == {0, 1, 2}


def test_stream_pca_subspace_matches_dense_pca():
    rng = np.random.default_rng(2)
    N, D, k = 400, 300, 5
    Z = rng.normal(size=(N, k)) * np.array([10, 8, 6, 4, 3.0])
    X = (Z @ rng.normal(size=(k, D)) + 0.05 * rng.normal(size=(N, D))).astype(np.float32)
    rd = lambda r: X[r]
    p = L.StreamPCA(k=k, oversample=10, n_iter=2, seed=0, chunk=64).fit(rd, np.arange(N), D)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    # principal angles between the two k-subspaces: all cosines ~ 1
    cs = np.linalg.svd(p.comp.astype(np.float64) @ Vt[:k].T, compute_uv=False)
    assert cs.min() > 0.999, cs
    assert np.abs(p.mu - X.mean(0)).max() < 1e-4


def _planted(n=1500, K=60, seed=0, signal=1.2):
    rng = np.random.default_rng(seed)
    clip = rng.integers(0, K, n)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 6)) + signal * y[:, None] * np.array([1, .5, 0, 0, 0, 0])
    return X, y, clip


def _cv(X, y, clip, family="linear"):
    shas = np.array([f"{c * 7919 + 13:012x}" for c in clip])
    outer, inner = L.outer_fold(shas), L.inner_fold(shas)
    oof, _ = L.cv_oof(lambda f: X, y, outer, inner, family)
    return oof, shas


def test_planted_signal_is_seen_and_a_null_is_not():
    X, y, clip = _planted()
    oof, _ = _cv(X, y, clip)
    r = L.auc_ci(oof, y, clip, n_boot=300)
    assert r["auc"] > 0.80 and r["lo"] > 0.7
    X0, y0, clip0 = _planted(signal=0.0, seed=3)
    oof0, _ = _cv(X0, y0, clip0)
    r0 = L.auc_ci(oof0, y0, clip0, n_boot=300)
    assert r0["lo"] <= 0.5 <= r0["hi"]


def test_leak_control_reads_one_and_shuffle_reads_chance():
    X, y, clip = _planted(signal=0.0, seed=4)
    leak = np.concatenate([X, y[:, None].astype(float)], 1)
    oof, _ = _cv(leak, y, clip)
    assert L.WAuc(oof, y).auc() >= 0.99
    ys = L.shuffle_labels(y, 1)
    oof2, _ = _cv(X, ys, clip)
    r = L.auc_ci(oof2, ys, clip, n_boot=300)
    assert r["lo"] <= 0.5 <= r["hi"]


def test_donor_labels_come_from_another_clip():
    clip = np.repeat(np.arange(10), 20)
    y = (clip % 2 == 0).astype(int)                                       # label is a pure function of the clip
    yd = L.donor_labels(y, clip, 2)
    donor_parity = {c: set(yd[clip == c].tolist()) for c in range(10)}
    assert all(len(v) == 1 for v in donor_parity.values())                # one donor clip per clip
    assert any(donor_parity[c] != {int(c % 2 == 0)} for c in range(10))   # and not (all) its own labels


def test_paired_dauc_is_zero_for_identical_scores():
    X, y, clip = _planted()
    oof, _ = _cv(X, y, clip)
    r = L.dauc_ci(oof, oof, y, clip, n_boot=200)
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["excludes_0"]


def test_foldmean_metric_unbiased_on_prevalence_shifted_null():
    """The POST-FREEZE metric: on clip-clustered null labels the pooled OOF AUC can sit below 0.5 (fold intercepts), the fold-mean AUC does not."""
    rng = np.random.default_rng(5)
    K, per = 120, 30
    clip = np.repeat(np.arange(K), per)
    rate = rng.uniform(0.05, 0.95, K)                                     # clip-level prevalence varies a lot
    y = (rng.uniform(size=K * per) < rate[clip]).astype(int)
    X = rng.normal(size=(K * per, 5))                                     # no signal
    shas = np.array([f"{c * 104729 + 7:012x}" for c in clip])
    outer, inner = L.outer_fold(shas), L.inner_fold(shas)
    oof, _ = L.cv_oof(lambda f: X, y, outer, inner, "linear")
    pooled = L.WAuc(oof, y).auc()
    fm = L.FoldWAuc(oof, y, outer).auc()
    assert abs(fm - 0.5) < 0.05
    assert fm >= pooled - 1e-9                                            # the pooled number is the lower one under prevalence shift
