"""Analytic sanity tests for the Traffic-Light Compliance (TLC) metric (SC-14).

Every case is a hand-built synthetic fixture whose TLC value is derived by hand in the comment
next to the assertion, so a reviewer can recompute it. No trained model, no simulator — the point
is that the *math* is trustworthy the moment real signalized-intersection telemetry exists.

TLC = red_entry_gate * stop_quality * green_flow, in [0, 1], higher better; 0 = ran a red light.
ego_s here is the cumulative distance ``_cum(v) = cumsum(v)*dt`` (distance through the end of each
step) so the halt position and line-crossing are exact and hand-checkable.
"""

import numpy as np
import pytest

from tanitad.eval.metrics import (
    SIGNAL_GREEN,
    SIGNAL_RED,
    TLC_COMFORT_DECEL,
    TLC_MARGIN_COMFORT_M,
    TLC_MARGIN_SCALE_M,
    compute_tlc,
    tlc_report,
)


def _cum(v, dt=1.0):
    """Cumulative down-route distance: distance through the end of each step (monotonic)."""
    return np.cumsum(np.asarray(v, dtype=float)) * dt


# --------------------------------------------------------------------------- #
# 1. Clean comfortable stop on red -> TLC = 1 (the pass case)                   #
# --------------------------------------------------------------------------- #
def test_clean_stop_on_red_is_one():
    # v=[10,8,6,4,2,0,0,0] dt=1: interior gradient = -2 -> peak_decel=2 (< comfort 2.5 -> smooth=1).
    # s=[10,18,24,28,30,30,30,30] halts at s=30; stopline=33 -> margin=3 (<=5 -> margin_factor=1).
    # never crosses (max s=30<33) -> red_entry_gate=1; all RED -> must_stop, green N/A -> TLC=1.
    v = [10, 8, 6, 4, 2, 0, 0, 0]
    s = _cum(v)
    sig = np.full(len(v), SIGNAL_RED)
    rep = tlc_report(v, s, sig, stopline_s=33.0, dt=1.0)
    assert rep["red_entry_gate"] == 1.0
    assert rep["entered_on_red"] is False
    assert rep["stopped_before_line"] is True
    assert rep["stop_margin_m"] == pytest.approx(3.0)
    assert rep["peak_decel_mps2"] <= TLC_COMFORT_DECEL
    assert rep["stop_quality"] == pytest.approx(1.0)
    assert rep["TLC"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. Ran the red -> TLC = 0 (the hard-fail case, the Dallas failure)            #
# --------------------------------------------------------------------------- #
def test_ran_the_red_is_zero():
    # v=6 constant; s=[6,12,18,24,30] crosses stopline=10 at idx 1 (s=12) while RED at v=6>creep
    # -> entered_on_red=True -> red_entry_gate=0 -> TLC=0 regardless of anything else.
    v = np.full(5, 6.0)
    s = _cum(v)
    sig = np.full(5, SIGNAL_RED)
    rep = tlc_report(v, s, sig, stopline_s=10.0, dt=1.0)
    assert rep["entered_on_red"] is True
    assert rep["red_entry_gate"] == 0.0
    assert rep["TLC"] == 0.0
    assert compute_tlc(v, s, sig, stopline_s=10.0, dt=1.0) == 0.0


# --------------------------------------------------------------------------- #
# 3. Smooth proceed on green -> TLC = 1 (no phantom braking)                    #
# --------------------------------------------------------------------------- #
def test_smooth_proceed_on_green_is_one():
    # all GREEN, speed held -> must_stop False; green_flow: v_ref=min_v=10 -> severity 0 -> 1.
    v = np.full(6, 10.0)
    s = _cum(v)
    sig = np.full(6, SIGNAL_GREEN)
    rep = tlc_report(v, s, sig, stopline_s=100.0, dt=1.0)
    assert rep["must_stop"] is False
    assert rep["green_flow"] == pytest.approx(1.0)
    assert rep["TLC"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 4. Phantom braking on green -> TLC < 1 (analytic value)                       #
# --------------------------------------------------------------------------- #
def test_phantom_brake_on_green_penalized():
    # all GREEN, v dips 10->4 then recovers. v_ref=max(first 30% of 6 steps)=10, min_v=4 ->
    # severity=0.6; green_flow = 1 - (0.6 - 0.05)/(1 - 0.05) = 1 - 0.55/0.95 = 0.421053.
    v = [10, 10, 4, 4, 10, 10]
    s = _cum(v)
    sig = np.full(6, SIGNAL_GREEN)
    rep = tlc_report(v, s, sig, stopline_s=100.0, dt=1.0)
    assert rep["green_drop_frac"] == pytest.approx(0.6, abs=1e-9)
    assert rep["TLC"] == pytest.approx(1.0 - 0.55 / 0.95, abs=1e-4)
    assert rep["TLC"] < 1.0


# --------------------------------------------------------------------------- #
# 5. Over-cautious far-back stop on red -> margin flow-cost penalty (analytic)  #
# --------------------------------------------------------------------------- #
def test_overcautious_far_stop_penalized():
    # v=[10,8,6,4,2,0,0,0] dt=1 (peak_decel=2 -> smooth=1); s halts at 30; stopline=43 -> margin=13.
    # margin_factor = exp(-(13 - 5)/8) = exp(-1); smooth=1 -> stop_quality = exp(-1).
    v = [10, 8, 6, 4, 2, 0, 0, 0]
    s = _cum(v)
    sig = np.full(len(v), SIGNAL_RED)
    rep = tlc_report(v, s, sig, stopline_s=43.0, dt=1.0)
    assert rep["stop_margin_m"] == pytest.approx(13.0)
    expected = np.exp(-(13.0 - TLC_MARGIN_COMFORT_M) / TLC_MARGIN_SCALE_M)  # exp(-1)
    assert rep["TLC"] == pytest.approx(expected, abs=1e-4)
    assert rep["TLC"] < 1.0                       # over-caution is a (mild) flow cost, not a legal fail


# --------------------------------------------------------------------------- #
# 6. Harsh emergency slam on red -> smoothness penalty (analytic)               #
# --------------------------------------------------------------------------- #
def test_harsh_slam_stop_penalized():
    # v=[12,6,0,0,0,0] dt=1: interior central-diff gradient min = -6 -> peak_decel=6 (> comfort 2.5).
    # s=[12,18,18,18,18,18] halts at s=18; stopline=20 -> margin=2 (comfort -> margin_factor=1).
    # smooth_factor = 1/(1 + 0.5*(6 - 2.5)) = 1/(1 + 1.75) = 1/2.75; stop_quality = that.
    v = [12, 6, 0, 0, 0, 0]
    s = _cum(v)
    sig = np.full(6, SIGNAL_RED)
    rep = tlc_report(v, s, sig, stopline_s=20.0, dt=1.0)
    assert rep["stopped_before_line"] is True
    assert rep["stop_margin_m"] == pytest.approx(2.0)
    assert rep["peak_decel_mps2"] == pytest.approx(6.0)   # > comfort 2.5
    expected = 1.0 / (1.0 + 0.5 * (6.0 - TLC_COMFORT_DECEL))  # 1/2.75
    assert rep["stop_quality"] == pytest.approx(expected, abs=1e-4)
    assert rep["TLC"] == pytest.approx(expected, abs=1e-4)
    assert rep["TLC"] < 1.0


# --------------------------------------------------------------------------- #
# 7. Crossing on green is not a violation                                      #
# --------------------------------------------------------------------------- #
def test_crossing_on_green_is_not_a_violation():
    # v=10 const crosses stopline=15 at speed, but the signal is GREEN -> no violation.
    v = np.full(6, 10.0)
    s = _cum(v)
    sig = np.full(6, SIGNAL_GREEN)
    rep = tlc_report(v, s, sig, stopline_s=15.0, dt=1.0)
    assert rep["entered_on_red"] is False
    assert rep["red_entry_gate"] == 1.0
    assert rep["TLC"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 8. Direction/ordering: clean stop > slam > ran-the-red                       #
# --------------------------------------------------------------------------- #
def test_ordering_clean_gt_slam_gt_ranred():
    sig_r = lambda n: np.full(n, SIGNAL_RED)
    clean = tlc_report([10, 8, 6, 4, 2, 0, 0, 0], _cum([10, 8, 6, 4, 2, 0, 0, 0]),
                       sig_r(8), 33.0, dt=1.0)["TLC"]
    slam = tlc_report([12, 6, 0, 0, 0, 0], _cum([12, 6, 0, 0, 0, 0]), sig_r(6), 20.0, dt=1.0)["TLC"]
    ranred = tlc_report(np.full(5, 6.0), _cum(np.full(5, 6.0)), sig_r(5), 10.0, dt=1.0)["TLC"]
    assert clean > slam > ranred == 0.0
