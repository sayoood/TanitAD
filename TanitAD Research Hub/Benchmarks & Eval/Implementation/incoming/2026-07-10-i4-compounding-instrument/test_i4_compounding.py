"""Analytic-ground-truth sanity tests for the hardened I4/D3 compounding instrument (G-B2).

Every assertion has a KNOWN closed-form answer — no model, no checkpoint, no randomness that
isn't seeded. These lock in the audit verdicts as regressions.
"""
from __future__ import annotations

import numpy as np
import pytest

from i4_compounding import (
    compounding_ratio, compounds, cr_from_predictions, cr_from_rel, rel_triplet,
)


def _unit(rng, n, d):
    e = rng.normal(size=(n, d))
    return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)


def test_rel_triplet_exposes_numerator_and_denominator():
    # z_t=0, true=(3,4) -> drift=5; pred off by exactly (0.5,0) from true -> abs_err=0.5.
    z = np.zeros((1, 2))
    true = np.array([[3.0, 4.0]])
    pred = true + np.array([[0.5, 0.0]])
    t = rel_triplet(pred, true, z)
    assert t["drift_median"] == pytest.approx(5.0, abs=1e-9)
    assert t["abs_err_median"] == pytest.approx(0.5, abs=1e-9)
    assert t["rel_k"] == pytest.approx(0.1, abs=1e-9)   # 0.5 / 5


def test_falling_rel_despite_genuine_superlinear_compounding():
    """THE artifact regression. Absolute error compounds superlinearly (err ~ k^1.3) yet rel_k
    FALLS with k because drift ~ k^1.5 grows faster. compounds() (abs-curve) must still say
    'compounds' and 'superlinear' — proving the rel-slope must NOT be used for that call."""
    rng = np.random.default_rng(20260710)
    N, d = 300, 512
    z0 = rng.normal(size=(N, d))
    v = rng.normal(scale=1.0 / np.sqrt(d), size=(N, d))
    rel_by_k, abs_by_k = {}, {}
    for k in (1, 2, 4):
        true_k = z0 + (k ** 1.5) * v                    # drift ~ k^1.5
        pred_k = true_k + (0.30 * k ** 1.3) * _unit(rng, N, d)  # abs err ~ k^1.3 (compounds)
        t = rel_triplet(pred_k, true_k, z0)
        rel_by_k[k], abs_by_k[k] = t["rel_k"], t["abs_err_median"]
    # rel FALLS with k (the artifact)
    assert rel_by_k[1] > rel_by_k[2] > rel_by_k[4]
    # but absolute error RISES with k (the truth) and the verdict is compounding+superlinear
    assert abs_by_k[1] < abs_by_k[2] < abs_by_k[4]
    v = compounds(abs_by_k)
    assert v["verdict"] == "compounds"
    assert v["superlinear"] is True
    assert v["loglog_growth_exponent"] == pytest.approx(1.3, abs=0.05)


def test_cr_is_denominator_free_equals_abs_error_ratio():
    """recursive and direct share the target true_k and z_t. CR must equal the abs-error ratio
    exactly, and reconstructing it from the two rel_k values must give the same number."""
    rng = np.random.default_rng(7)
    N, d = 400, 256
    z0 = rng.normal(size=(N, d))
    true_k = z0 + rng.normal(scale=0.5, size=(N, d))
    direct = true_k + 0.20 * _unit(rng, N, d)          # abs err 0.20
    rollout = true_k + 0.80 * _unit(rng, N, d)         # abs err 0.80 -> CR should be ~4
    cr = cr_from_predictions(rollout, direct, true_k)
    assert cr == pytest.approx(4.0, rel=0.05)
    # reconstruct CR from the two rel_k (shared denominator) -> identical
    rel_roll = rel_triplet(rollout, true_k, z0)["rel_k"]
    rel_dir = rel_triplet(direct, true_k, z0)["rel_k"]
    assert cr_from_rel(rel_roll, rel_dir) == pytest.approx(cr, rel=0.05)


def test_reported_d3_recursion_ratios_reproduce_as_cr():
    """The step-14k reported numbers: CR = recursive/direct_k4 with shared denominator."""
    assert cr_from_rel(20.513, 5.123) == pytest.approx(4.004, abs=0.01)   # comma2k19
    assert cr_from_rel(9.473, 2.550) == pytest.approx(3.715, abs=0.01)    # physicalai
    # within-arm CR at step-10500: base compounds, K-step arm does not (CR<1)
    assert cr_from_rel(14.495, 3.712) == pytest.approx(3.906, abs=0.01)   # base arm
    assert cr_from_rel(1.113, 2.891) == pytest.approx(0.385, abs=0.01)    # K-step arm


def test_collapse_masquerade_inflates_rel_at_equal_abs_error():
    """Two models, IDENTICAL absolute error, encoder drift halved -> rel_k ~2x worse.
    Proves cross-model rel_k comparison is confounded; CR (within-model) is not."""
    rng = np.random.default_rng(123)
    N, d = 300, 512
    for scale, key in ((1.0, "normal"), (0.5, "compressed")):
        z0 = rng.normal(size=(N, d))
        true = z0 + rng.normal(scale=scale / np.sqrt(d), size=(N, d))
        pred = true + 0.30 * _unit(rng, N, d)          # SAME abs err 0.30
        t = rel_triplet(pred, true, z0)
        if key == "normal":
            rel_normal, drift_normal = t["rel_k"], t["drift_median"]
        else:
            rel_comp, drift_comp = t["rel_k"], t["drift_median"]
    assert drift_normal == pytest.approx(2 * drift_comp, rel=0.1)
    # halving drift roughly doubles rel_k at fixed absolute error
    assert rel_comp / rel_normal == pytest.approx(2.0, rel=0.15)


def test_cr_near_one_when_rollout_tracks_direct():
    rng = np.random.default_rng(99)
    N, d = 300, 128
    true_k = rng.normal(size=(N, d))
    err = 0.25 * _unit(rng, N, d)
    assert compounding_ratio(np.linalg.norm(err, axis=-1),
                             np.linalg.norm(err, axis=-1)) == pytest.approx(1.0, abs=1e-6)


def test_compounds_flat_error_is_not_superlinear():
    v = compounds({1: 0.30, 2: 0.30, 4: 0.30})
    assert v["superlinear"] is False
    assert v["loglog_growth_exponent"] == pytest.approx(0.0, abs=1e-6)
