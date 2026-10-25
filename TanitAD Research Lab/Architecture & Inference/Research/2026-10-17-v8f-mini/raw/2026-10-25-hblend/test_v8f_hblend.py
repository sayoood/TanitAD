"""Self-tests of the HBLEND arithmetic (analytic identities + a deliberate-regression arm). CPU, numpy only, tiny.

  python -m pytest -q test_v8f_hblend.py --basetemp C:/Users/Admin/pt_hblend -p no:cacheprovider
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_hblend as H  # noqa: E402


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n, 30, 2)) * 10
    r = rng.normal(size=(n, 30, 2)) * np.linspace(0.05, 2.0, 30)[None, :, None]
    return P, r


def test_pav_matches_bruteforce_and_is_monotone():
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = rng.uniform(0, 1, size=7)
        z = H.pav_nondecreasing(y)
        assert (np.diff(z) >= -1e-12).all()
        # optimality: no other non-decreasing step function on a grid of candidate levels has a smaller squared error
        levels = np.unique(np.round(np.concatenate([y, z]), 6))
        best = min(((np.array(c) - y) ** 2).sum() for c in itertools.combinations_with_replacement(levels, 7)
                   if all(np.diff(c) >= 0)) if len(levels) <= 9 else None
        if best is not None:
            assert ((z - y) ** 2).sum() <= best + 1e-9


def test_closed_form_identities():
    P, r = _data()
    tau = P + r
    # y = tau  -> w == 1 everywhere ; y = P -> w == 0 ; y = P + 0.4 r -> w == 0.4 (flat, so isotonic leaves it)
    assert np.allclose(H.fit_weights(tau, P, tau)["w"], 1.0)
    assert np.allclose(H.fit_weights(tau, P, P)["w"], 0.0)
    assert np.allclose(H.fit_weights(tau, P, P + 0.4 * r)["w"], 0.4)
    # y = P + 3 r -> raw 3 -> clip 1 ; y = P - r -> raw -1 -> clip 0
    assert np.allclose(H.fit_weights(tau, P, P + 3 * r)["w"], 1.0)
    assert np.allclose(H.fit_weights(tau, P, P - r)["w"], 0.0)


def test_clip_then_isotonic_order_and_range():
    P, r = _data()
    rng = np.random.default_rng(2)
    y = P + r * rng.uniform(-0.5, 1.5, size=(1, 30, 2)) + rng.normal(size=r.shape) * 0.3
    R = H.fit_weights(P + r, P, y)
    assert ((R["w"] >= 0) & (R["w"] <= 1)).all() and (np.diff(R["w"], axis=0) >= -1e-12).all()
    assert np.allclose(R["clipped"], np.clip(R["raw"], 0, 1))
    # the PREREG order is clip THEN isotonic: re-deriving it from `clipped` reproduces w exactly
    for a in range(2):
        assert np.allclose(R["w"][:, a], H.pav_nondecreasing(R["clipped"][:, a]))


def test_apply_identities():
    P, r = _data()
    tau = P + r
    assert np.allclose(H.apply_w(P, tau, 1.0), tau)
    assert np.allclose(H.apply_w(P, tau, 0.0), P)
    w = np.linspace(0, 1, 30)[:, None] * np.ones((1, 2))
    assert np.allclose(H.apply_w(P, tau, w), P + w * r)


def test_deliberate_regression_wrong_target_is_caught():
    """If the GT were silently swapped for the PRIOR (a defect: the fit would read no signal) the weights collapse to 0 -- the shrinkage then
    reproduces P10 exactly, which the w == 0 identity control on the real eval would flag as 'H == P10', never as a gain."""
    P, r = _data()
    w_bad = H.fit_weights(P + r, P, P)["w"]
    assert np.allclose(w_bad, 0.0)
    assert np.allclose(H.apply_w(P, P + r, w_bad), P)


def test_midpoints_are_derived_not_fitted():
    w = np.stack([np.linspace(0.1, 0.9, 30), np.linspace(0.2, 1.0, 30)], 1)
    w60 = H.w_midpoints(w)
    assert w60.shape == (60, 2) and np.allclose(w60[1::2], w)
    assert (np.diff(w60, axis=0) >= -1e-12).all()


def test_banked_weights_are_valid():
    import json
    for s in H.SEEDS:
        p = os.path.join(H.OUT, f"weights_{s}.json")
        if not os.path.exists(p):
            pytest.skip("weights not fitted yet")
        W = json.load(open(p))
        w = np.stack([W["w_along_by_step"], W["w_cross_by_step"]], 1)
        assert w.shape == (30, 2) and ((w >= 0) & (w <= 1)).all() and (np.diff(w, axis=0) >= -1e-12).all()
        assert W["n_clips"] == 141 and W["k"] == 32
