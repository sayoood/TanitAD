"""The horizon estimator must recover a KNOWN horizon, and refuse when it cannot.

These tests are synthetic on purpose: the estimator's whole claim is that
``v_h`` falls out of the geometry without a known lane width, so the way to test
it is to build data from a horizon we chose and check it comes back.

⚠️ THE REFUSAL CASES CARRY AS MUCH WEIGHT AS THE RECOVERY CASE. The sibling
failure this module exists to prevent (`R-2026-09-20-horizon472`) was a metric
that returned a confident answer where it was blind, so an estimator that cannot
say "I don't know" would reproduce it in a new place.
"""
from __future__ import annotations

import numpy as np
import pytest

from tanitad.data.trajrecon.horizon_calib import horizon_from_lane_width


def _obs(v_h=463.0, w_over_h=2.30, rows=(540.0, 810.0), n=400, noise=0.0, seed=0):
    """Rows and pixel widths for a lane of constant metric width."""
    rng = np.random.default_rng(seed)
    v = np.linspace(rows[0], rows[1], n)
    w = w_over_h * (v - v_h)
    if noise:
        w = w + rng.normal(0.0, noise, size=n)
    return v, w


def test_recovers_a_known_horizon_exactly_when_clean():
    v, w = _obs(v_h=463.0, w_over_h=2.30)
    fit = horizon_from_lane_width(v, w)
    assert fit.ok, fit.reason
    assert fit.horizon_row == pytest.approx(463.0, abs=0.05)
    assert fit.lane_width_over_height == pytest.approx(2.30, rel=1e-6)


def test_lane_width_needs_a_height_and_is_not_free():
    """The slope is W/h; metres only appear once a height is supplied."""
    v, w = _obs(v_h=463.0, w_over_h=3.60 / 1.586)
    fit = horizon_from_lane_width(v, w)
    assert fit.ok
    assert fit.lane_width_m(1.586) == pytest.approx(3.60, rel=1e-6)
    # the SAME pixels with a different height give a different width -- that is
    # the f*h degeneracy, and this estimator does not touch it.
    assert fit.lane_width_m(1.40) == pytest.approx(3.60 * 1.40 / 1.586, rel=1e-6)


def test_survives_noise_and_recovers_the_horizon():
    v, w = _obs(v_h=463.0, noise=6.0, n=800, seed=3)
    fit = horizon_from_lane_width(v, w)
    assert fit.ok, fit.reason
    assert fit.horizon_row == pytest.approx(463.0, abs=2.0)


def test_outliers_with_leverage_do_not_drag_the_intercept():
    """A few far-row misdetections must not move the answer.

    A wrong paint pairing at a FAR row is the dangerous case: it sits where the
    lever is longest, so a plain least squares moves the intercept a long way.
    """
    v, w = _obs(v_h=463.0, noise=3.0, n=600, seed=7)
    w = w.copy()
    w[:25] *= 2.2                      # 25 far-row detections twice too wide
    fit = horizon_from_lane_width(v, w)
    assert fit.ok, fit.reason
    assert fit.horizon_row == pytest.approx(463.0, abs=4.0)
    plain = np.polyfit(v, w, 1)        # what an untrimmed fit would have said
    naive_vh = -plain[1] / plain[0]
    assert abs(naive_vh - 463.0) > abs(fit.horizon_row - 463.0)


def test_refuses_a_row_span_too_short_to_extrapolate():
    v, w = _obs(v_h=463.0, rows=(700.0, 730.0), n=200)
    fit = horizon_from_lane_width(v, w)
    assert not fit.ok
    assert "row span" in fit.reason


def test_refuses_too_few_observations():
    v, w = _obs(n=10)
    fit = horizon_from_lane_width(v, w)
    assert not fit.ok
    assert "usable observations" in fit.reason


def test_refuses_a_negative_slope():
    """Lane width must GROW toward the bottom of the image."""
    v, w = _obs(v_h=463.0)
    fit = horizon_from_lane_width(v, w[::-1])
    assert not fit.ok
    assert "positive" in fit.reason or "R^2" in fit.reason


def test_refuses_when_width_is_not_linear_in_row():
    """No single horizon describes these data, so none is returned."""
    rng = np.random.default_rng(11)
    v = np.linspace(540.0, 810.0, 500)
    w = 200.0 + rng.normal(0.0, 60.0, size=v.size)      # flat + noise, no trend
    fit = horizon_from_lane_width(v, w)
    assert not fit.ok
    assert "R^2" in fit.reason or "positive" in fit.reason


def test_refuses_an_answer_outside_the_admissible_window():
    v, w = _obs(v_h=300.0)
    fit = horizon_from_lane_width(v, w, row_bounds=(430.0, 520.0))
    assert not fit.ok
    assert "admissible window" in fit.reason
    assert fit.horizon_row == pytest.approx(300.0, abs=1.0)   # reported, not used


def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError):
        horizon_from_lane_width([1.0, 2.0, 3.0], [1.0, 2.0])


def test_a_nine_pixel_error_is_the_flare_this_module_exists_to_prevent():
    """Pin the magnitude that made a 9 px horizon error visible on screen.

    Drawn width is wrong by (v - v_h_true)/(v - v_h_used); with v_h_true = 463
    and v_h_used = 472 this is ~3.7 % at 10 m and ~11 % at 30 m for the
    2026-08-08 geometry (f = 1533, h = 1.586).
    """
    f_h = 1533.0 * 1.586
    for rng_m, want in ((10.0, 1.037), (30.0, 1.111)):
        v = 472.0 + f_h / rng_m                     # row for that range at v_h=472
        assert (v - 463.0) / (v - 472.0) == pytest.approx(want, abs=0.002)
