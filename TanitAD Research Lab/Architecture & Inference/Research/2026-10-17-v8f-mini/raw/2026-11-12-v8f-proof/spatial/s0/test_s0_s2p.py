"""Unit test of control s2p (S0_DEFS R18) on SYNTHETIC data only: the POSITION of a planted object is read from the grid state and not from the mean state, at a large amplitude."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import s0_read as S


def _run(amplitude):
    rng = np.random.default_rng(0)
    n, nc, dt, k = 600, 40, 64, 4
    X = rng.standard_normal((n, nc, dt)).astype(np.float32)
    v = rng.standard_normal(dt).astype(np.float32)
    v = v / np.linalg.norm(v) * amplitude
    side = (rng.random(n) < 0.5).astype(np.float32)
    ep = np.array([f"e{i % 30}" for i in range(n)])
    idx = np.arange(n)
    tr, te = idx[idx % 30 < 24], idx[idx % 30 >= 24]
    a, b = 13, 16                                            # a cell and its mirror cell of the same grid row
    Sg = S.grid_state(X, tr, k)
    Sg[:, a * k:(a + 1) * k] = S.cell_block(X, tr, a, k, add=(1.0 - side)[:, None] * v[None, :])
    Sg[:, b * k:(b + 1) * k] = S.cell_block(X, tr, b, k, add=side[:, None] * v[None, :])
    Sm = S.mean_state(S.mean_token(X) + v[None, :] / nc, tr, 8)
    out = {}
    for name, Sx in (("g", Sg), ("m", Sm)):
        fit = S.fit_probe(Sx[tr], side[tr].astype(np.float64), ep[tr], 0)
        out[name] = float(S.RP.auc_pooled(side[te], fit(Sx[te])))
    return out


def test_position_read_by_grid_not_by_mean_at_large_amplitude():
    out = _run(amplitude=8.0 * np.sqrt(64))                 # 8 standard deviations per coordinate: the mean state would read PRESENCE at this size
    assert out["g"] > 0.95, out
    assert out["m"] <= 0.60, out
    assert S.check_s2(out["g"], out["m"])["pass"]


def test_position_bar_fires_when_the_mean_state_is_given_the_side():
    # mutation of the control itself: a state that carries the side must fail the bar
    assert not S.check_s2(0.99, 0.99)["pass"]
