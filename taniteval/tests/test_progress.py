"""EGO PROGRESS — pinned on the properties that make it admissible, not on a golden number.

Every rule below is also driven with the value that makes it **FAIL**, per the suite's standard.

What is pinned:

1. **GT scores exactly 1.0 under ``human_dir``** — the property the whole discrimination control
   rests on. If it ever stops holding, ``progress_error`` is no longer an error with a known
   optimum and no GT-vs-arm control means anything.
2. **The curvature confound in the published ``t0_axis`` reading is REAL and is measured, not
   asserted** — on a curved window GT scores ``cos(theta) < 1`` against itself.
3. **The threshold is pseudosim's, not a second one.** A future edit that forks the constant fails
   here rather than silently creating two definitions of "the human did not move".
4. **A stopped human is EXCLUDED, never clamped to a perfect score.**
5. **dt-invariance** — the ratio must not move when the same geometry is resampled onto a different
   grid, which is what makes it immune to the 2026-08-03 sparse-grid dt defect.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from taniteval import progress as P


def _straight(n=4, H=4, dist=10.0):
    """A human driving straight ahead: endpoints at (dist, 0)."""
    t = np.linspace(1.0 / H, 1.0, H)
    g = np.zeros((n, H, 2))
    g[:, :, 0] = dist * t
    return g


def _curved(n=1, H=4, dist=10.0, theta_deg=30.0):
    """A human whose CHORD sits at ``theta`` from the t0 forward axis."""
    th = math.radians(theta_deg)
    t = np.linspace(1.0 / H, 1.0, H)
    g = np.zeros((n, H, 2))
    g[:, :, 0] = dist * t * math.cos(th)
    g[:, :, 1] = dist * t * math.sin(th)
    return g


# --------------------------------------------------------------------------- #
# 1. the property the discrimination control rests on                           #
# --------------------------------------------------------------------------- #
def test_gt_scores_exactly_one_under_human_dir_on_straight_and_curved():
    for g in (_straight(), _curved(theta_deg=40.0)):
        out = P.progress(g, g)
        assert out["status"] == "OK"
        assert out["progress_ratio_mean"] == pytest.approx(1.0, abs=1e-9)
        assert out["progress_error_mean"] == pytest.approx(0.0, abs=1e-9)


def test_the_failing_case_a_shortened_arm_does_NOT_score_one():
    g = _straight(dist=10.0)
    out = P.progress(g * 0.8, g)
    assert out["progress_ratio_mean"] == pytest.approx(0.8, abs=1e-9)
    assert out["under_progress_rate"] == 1.0


def test_over_driving_is_visible_and_signed_not_folded():
    g = _straight(dist=10.0)
    over = P.progress(g * 1.25, g)
    under = P.progress(g * 0.75, g)
    # ⛔ both are errors of 0.25 — an unsigned-only report could not tell them apart
    assert over["progress_error_mean"] == pytest.approx(0.25, abs=1e-9)
    assert under["progress_error_mean"] == pytest.approx(0.25, abs=1e-9)
    assert over["under_progress_rate"] == 0.0
    assert under["under_progress_rate"] == 1.0


# --------------------------------------------------------------------------- #
# 2. the curvature confound in the published t0-axis reading                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.0, 15.0, 30.0, 45.0])
def test_t0_axis_charges_the_HUMAN_for_curvature_and_the_size_is_cos_theta(theta):
    g = _curved(theta_deg=theta)
    r = P.progress_per_window(g, g, "t0_axis")["ratio"]
    assert float(r[0]) == pytest.approx(math.cos(math.radians(theta)), abs=1e-9)
    # and the default convention does NOT have that defect
    r2 = P.progress_per_window(g, g, "human_dir")["ratio"]
    assert float(r2[0]) == pytest.approx(1.0, abs=1e-9)


def test_the_audit_field_reports_the_confound_rather_than_hiding_it():
    g = _curved(theta_deg=30.0)
    out = P.progress(g, g)
    assert out["t0_axis_gt_self_ratio"] == pytest.approx(math.cos(math.radians(30.0)), abs=1e-4)


def test_t0_axis_rewards_an_arm_for_driving_STRAIGHTER_than_the_human():
    """The consequence that makes the confound decision-relevant, driven to failure."""
    g = _curved(theta_deg=30.0, dist=10.0)
    straight_arm = _straight(n=1, dist=10.0)          # same distance, no turn
    r_t0 = P.progress_per_window(straight_arm, g, "t0_axis")["ratio"][0]
    r_hd = P.progress_per_window(straight_arm, g, "human_dir")["ratio"][0]
    human_self_t0 = P.progress_per_window(g, g, "t0_axis")["ratio"][0]
    # ⛔ the arm that IGNORED the turn scores a perfect 1.0 while the HUMAN scores cos(30 deg).
    assert r_t0 == pytest.approx(1.0, abs=1e-9)
    assert human_self_t0 < r_t0                        # the human loses to a wrong path
    assert r_hd < 1.0                                  # human_dir charges it correctly
    assert r_t0 > r_hd


# --------------------------------------------------------------------------- #
# 3. one constant, not two                                                      #
# --------------------------------------------------------------------------- #
def test_threshold_is_pseudosims_published_constant_not_a_second_one():
    from taniteval import pseudosim as PS
    assert P.PROGRESS_HUMAN_MIN_M == PS.PROGRESS_HUMAN_MIN_M


# --------------------------------------------------------------------------- #
# 4. the stopped human is excluded, never clamped                                #
# --------------------------------------------------------------------------- #
def test_a_stopped_human_is_excluded_and_counted_not_scored_as_perfect():
    g = np.zeros((3, 4, 2))
    g[0, :, 0] = np.linspace(2.5, 10.0, 4)            # one moving window
    out = P.progress(g, g)
    assert out["status"] == "OK"
    assert out["n"] == 1
    assert out["n_excluded_low_gt_progress"] == 2
    assert out["n_windows"] == 3


def test_all_stopped_is_UNAVAILABLE_with_a_reason_never_a_pass():
    g = np.zeros((3, 4, 2)) + 0.01
    out = P.progress(g, g)
    assert out["status"] == "UNAVAILABLE"
    assert out["n"] == 0
    assert "clamped" in out["reason"]


def test_excluded_entries_are_nan_not_a_filled_value():
    g = np.zeros((2, 4, 2))
    g[1, :, 0] = np.linspace(2.5, 10.0, 4)
    w = P.progress_per_window(g, g)
    assert np.isnan(w["ratio"][0]) and not np.isnan(w["ratio"][1])


# --------------------------------------------------------------------------- #
# 5. dt-invariance — immunity to the sparse-grid defect                          #
# --------------------------------------------------------------------------- #
def test_ratio_is_dt_invariant_across_a_sparse_and_a_dense_grid():
    dense_g = _straight(n=1, H=20, dist=10.0)
    dense_p = dense_g * 0.9
    sparse_g, sparse_p = dense_g[:, 4::5], dense_p[:, 4::5]      # 4-waypoint view
    a = P.progress(dense_p, dense_g)["progress_ratio_mean"]
    b = P.progress(sparse_p, sparse_g)["progress_ratio_mean"]
    assert a == pytest.approx(b, abs=1e-9)


# --------------------------------------------------------------------------- #
# 6. contracts refuse rather than guess                                          #
# --------------------------------------------------------------------------- #
def test_unknown_convention_raises_instead_of_defaulting():
    g = _straight()
    with pytest.raises(ValueError, match="unknown convention"):
        P.progress_per_window(g, g, "along_track")


def test_shape_mismatch_and_wrong_rank_are_refused():
    g = _straight()
    with pytest.raises(ValueError):
        P.progress_per_window(g, g[:, :2])
    with pytest.raises(ValueError):
        P.progress_per_window(g[:, :, 0], g[:, :, 0])


def test_a_reversing_arm_keeps_its_negative_sign():
    g = _straight(n=1, dist=10.0)
    w = P.progress_per_window(-g, g)
    assert float(w["ratio"][0]) == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 7. wiring: the LONGITUDINAL family carries it                                  #
# --------------------------------------------------------------------------- #
def test_longitudinal_family_reports_ego_progress():
    import torch

    from taniteval import four_families as FF
    g = torch.as_tensor(_straight(n=5, dist=10.0)).float()
    out = FF.longitudinal(g * 0.8, g, dt=0.5)
    assert out["ego_progress"]["status"] == "OK"
    assert out["ego_progress"]["progress_ratio_mean"] == pytest.approx(0.8, abs=1e-4)
