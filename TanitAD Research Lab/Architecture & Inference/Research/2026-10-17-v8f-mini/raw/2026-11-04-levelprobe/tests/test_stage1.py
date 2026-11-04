"""Known-answer tests of Stage 1's pieces on synthetic data (planted signal recovered, null stays null, the pooling equals an independent loop, a deliberate leak is read).
python -I -m pytest -q -p no:cacheprovider tests/test_stage1.py"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import numpy as np  # noqa: E402
import stage1 as S  # noqa: E402


def test_pool_equals_independent_loop():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, S.D_IN)).astype(np.float16)
    p = S.pool_g2x4(x)
    assert p.shape == (3, S.D_OUT)
    for i in range(3):
        assert np.abs(p[i] - S.pool_reference(x[i])).max() <= 1e-4


def test_pool_block_membership():
    """A unit planted in token (r, c) lands in block (r // 4, c // 5) and nowhere else, with weight 1/20."""
    x = np.zeros((1, 8, 20, 1024), np.float32)
    x[0, 5, 17, 7] = 20.0
    p = S.pool_g2x4(x.reshape(1, -1)).reshape(2, 4, 1024)
    assert p[1, 3, 7] == 1.0 and p.sum() == 1.0


def _synthetic(n=900, d=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = np.zeros(d, np.float32)
    w[:5] = 1.0
    y = X @ w + 0.1 * rng.standard_normal(n)
    return X, y


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def test_ridge_recovers_planted_signal_and_null_stays_null():
    X, y = _synthetic()
    tr, te = np.arange(600), np.arange(600, 900)
    R = S.Ridge(X, None, tr, base_trace_cols=64)
    best = max(_r2(y[te], p) for p in R.path(y[tr], X[te]).values())
    assert best > 0.95
    ysh = np.random.default_rng(1).permutation(y)
    R2 = S.Ridge(X, None, tr, base_trace_cols=64)
    null = max(_r2(ysh[te], p) for p in R2.path(ysh[tr], X[te]).values())
    assert null < 0.05, null


def test_leak_column_is_read_through_the_same_path():
    X, y = _synthetic()
    rng = np.random.default_rng(2)
    X = rng.standard_normal(X.shape).astype(np.float32)           # features carry NO signal
    leak = ((y - y.mean()) / y.std()).astype(np.float32)[:, None] * 8.0
    tr, te = np.arange(600), np.arange(600, 900)
    R = S.Ridge(X, leak, tr, base_trace_cols=64)
    best = max(_r2(y[te], p) for p in R.path(y[tr], X[te], leak[te]).values())
    assert best >= 0.99, best


def test_metrics_known_values():
    m = S.metrics([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1.0
    m = S.metrics([0.0, 2.0, 4.0], [2.0, 2.0, 2.0])
    assert abs(m["r2"]) < 1e-12 and abs(m["mae"] - 4 / 3) < 1e-12
