"""p1_lead_transforms — the probes must find what is planted and miss what isn't."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p1_lead_transforms as T                                       # noqa: E402


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


def _folds(n, k=5):
    return np.arange(n) % k


def test_ridge_finds_linear_signal(rng):
    X = rng.normal(size=(400, 16))
    y = X @ rng.normal(size=16) + 0.1 * rng.normal(size=400)
    assert T.ridge_oof(X, y, _folds(400)) > 0.9


def test_ridge_reads_nothing_from_noise(rng):
    X = rng.normal(size=(400, 16))
    y = rng.normal(size=400)
    assert T.ridge_oof(X, y, _folds(400)) < 0.1


def test_mlp_reads_nonlinear_signal_linear_does_not(rng):
    # y depends on ||x|| — invisible to a linear probe, easy for the MLP
    X = rng.normal(size=(600, 8))
    y = np.linalg.norm(X, axis=1) + 0.05 * rng.normal(size=600)
    assert T.ridge_oof(X, y, _folds(600)) < 0.25
    assert T.mlp_oof(X, y, _folds(600)) > 0.5


def test_transforms_shapes_and_monotonicity():
    gap = np.array([0.0, 4.0, 20.0, 79.0])
    v = np.array([1.0, 10.0, 10.0, 30.0])
    t = T.transforms_of(gap, v)
    assert set(t) == {"gap_m", "log1p_gap", "inv_gap", "ttc_proxy"}
    assert np.all(np.diff(t["log1p_gap"]) > 0)      # monotone in gap
    assert np.all(np.diff(t["inv_gap"]) < 0)        # anti-monotone
    assert t["ttc_proxy"][0] == 0.0 and t["ttc_proxy"][1] == pytest.approx(0.4)
