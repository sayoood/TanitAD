"""Guards for the reference<->rig frame-alignment estimator and its refusal rules.

Every test here is driven with input designed to make the estimator FAIL, because the
defect this module exists to prevent (`R-2026-08-03-k`) was an estimator that could not
fail: a bare ``max(d, key=score)`` returns an offset on a monotone curve, on a flat
curve, and on pure noise, and each of those was read as an answer at least once.

Three of the cases below are MEASURED failures, not hypotheticals:

* ``test_refuses_at_the_scan_boundary`` -- an earlier +-3 scan stopped at its boundary
  still rising and reported ">= +3".  A boundary argmax is the WINDOW's answer.
* ``test_refuses_a_monotone_curve`` -- the GPU-free cross-correlation on scene
  ``7c72937c`` rose monotonically from -15 to +15 and its argmax was +15.  The ego is
  stationary for the first 9 frames, so there is no signal to correlate against.
* ``test_leader_pad_refuses_when_the_ego_is_stationary`` -- the banked
  ``ALIGNMENT_DIRECTION_GPUFREE.json`` reports ``static_head_block_frames = 5`` for that
  same scene, whose own rig speed field reads ``0.0`` for the first 9 frames.  A frozen
  head block is evidence of a synthetic leader ONLY if the camera was moving.

And the case the task brief demanded explicitly: an estimator that always returns an
argmax will return one on a scene whose true offset is 0.  ``test_recovers_zero_offset``
and ``test_recovers_an_injected_shift`` are the known-answer controls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_EXP = Path(__file__).resolve().parents[1] / "experiments" / "alpasim-gsplat"
sys.path.insert(0, str(_EXP))

from frame_align import (AlignEstimate, adjudicate, bootstrap_offset,  # noqa: E402
                         consensus, count_delta, leader_pad, motion_lag)


# --------------------------------------------------------------------------------- #
# synthetic series with a KNOWN lag                                                  #
# --------------------------------------------------------------------------------- #
def _series(true_lag: int, n: int = 400, noise: float = 0.05, seed: int = 3):
    """Ego speed with structure, and an image-motion series lagged by `true_lag`.

    Convention, identical to the instrument: ``mp4_index = rig_index + true_lag``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ego = 1.0 + 0.6 * np.sin(t / 17.0) + 0.3 * np.sin(t / 5.0) + noise * rng.normal(size=n)
    ego[0] = np.nan
    img = np.full(n + abs(true_lag) + 4, np.nan)
    for f in range(1, n):
        i = f + true_lag
        if 0 <= i < len(img):
            img[i] = ego[f] * 2.0 + noise * rng.normal()
    return img, ego


# --------------------------------------------------------------------------------- #
# known-answer controls                                                              #
# --------------------------------------------------------------------------------- #
@pytest.mark.parametrize("true_lag", [0, 1, 5, 6, -4])
def test_recovers_a_known_lag_including_zero(true_lag):
    """The control the brief demands: a search that always returns an argmax will return
    one on a scene whose true answer is 0. This one must return 0 there."""
    img, ego = _series(true_lag)
    e = motion_lag(img, ego, max_lag=15)
    assert not e.refused, e.reason
    assert e.offset == true_lag


def test_recovers_an_injected_shift():
    """Re-index the image series by a known d -> the recovered offset moves by exactly d."""
    img, ego = _series(6)
    for d in (-3, -1, 2, 4):
        shifted = (np.r_[np.full(d, np.nan), img[:-d]] if d > 0
                   else np.r_[img[-d:], np.full(-d, np.nan)])
        e = motion_lag(shifted, ego, max_lag=15)
        assert not e.refused, (d, e.reason)
        assert e.offset == 6 + d, (d, e.offset)


def test_subframe_peak_is_near_integer_for_a_pure_index_error():
    """A whole-frame index error peaks at a near-integer position; a sub-frame timing
    error would not. Reported so the two are distinguishable rather than merged."""
    img, ego = _series(6)
    e = motion_lag(img, ego, max_lag=15)
    assert abs(e.subframe_offset - 6.0) < 0.5


# --------------------------------------------------------------------------------- #
# the four refusal rules                                                             #
# --------------------------------------------------------------------------------- #
def test_refuses_at_the_scan_boundary():
    """MEASURED failure: a +-3 scan stopped at its edge still rising and reported '>= +3'."""
    img, ego = _series(6)
    e = motion_lag(img, ego, max_lag=3)          # truth is outside the window
    assert e.refused
    assert e.reason in ("boundary", "no_turnover", "not_separated"), str(e)
    assert e.offset is None


def test_refuses_a_monotone_curve():
    """MEASURED failure: `7c72937c`'s cross-correlation rose monotonically -15 -> +15."""
    curve = {d: 0.30 + 0.02 * d for d in range(-15, 16)}   # prominent, so not "flat"
    e = adjudicate(curve, "synthetic_monotone")
    assert e.refused and e.reason == "boundary", str(e)


def test_refuses_a_plateau_that_never_turns_over():
    """A tie at the top is not a maximum: the argmax is then chosen by iteration order,
    which is the tie-break equivalent of the boundary failure."""
    curve = {d: 0.30 + 0.02 * d for d in range(-15, 11)}
    curve[11] = curve[10]          # plateau: argmax's right neighbour is not lower
    curve[12] = 0.10
    e = adjudicate(curve, "plateau")
    assert e.refused and e.reason == "no_turnover", str(e)


def test_refuses_pure_noise():
    rng = np.random.default_rng(11)
    e = motion_lag(rng.normal(size=500), rng.normal(size=500), max_lag=15)
    assert e.refused, str(e)


def test_refuses_a_flat_curve_as_not_separated():
    curve = {d: 0.50 for d in range(-5, 6)}
    curve[0] = 0.5001                 # a real but meaningless argmax
    curve[-1] = curve[1] = 0.4999     # give it a turnover so `not_separated` is reached
    e = adjudicate(curve, "flat")
    assert e.refused and e.reason == "not_separated"


def test_refuses_a_weak_peak():
    curve = {d: 0.0 for d in range(-5, 6)}
    curve[0] = 0.03                   # peaked and separated, but far below any real match
    e = adjudicate(curve, "weak", min_prominence=0.001)
    assert e.refused and e.reason == "weak"


def test_a_flat_curve_peaking_at_the_edge_is_uninformative_not_off_window():
    """MEASURED false alarm: `7c72937c` frame 60 is a STATIONARY segment; its whole
    +-10 curve spans 0.4003-0.4023 and its max sits at the scan edge. Reported as
    `boundary` it reads as "the residual is off-window" and HALTS a correct run; it is
    really "this frame carries no alignment information". Signal strength is adjudicated
    before window position, and this is the test that pins the order."""
    curve = {d: 0.4023 - 0.0002 * (d + 3) for d in range(-3, 4)}   # flat, max at -3
    e = adjudicate(curve, "flat_at_edge")
    assert e.refused and e.reason == "not_separated", str(e)


def test_a_clean_peak_is_accepted():
    curve = {d: 0.30 - 0.02 * abs(d - 6) for d in range(-5, 12)}
    e = adjudicate(curve, "clean")
    assert not e.refused and e.offset == 6
    assert e.prominence == pytest.approx(0.04, abs=1e-9)


def test_window_too_small_is_refused_not_answered():
    assert adjudicate({0: 1.0, 1: 2.0}, "tiny").refused
    assert adjudicate({}, "empty").refused


# --------------------------------------------------------------------------------- #
# the counting estimator                                                             #
# --------------------------------------------------------------------------------- #
def test_count_delta_matches_the_two_measured_scenes():
    """MEASURED: `00040136` mp4 605 / rig 599 -> +6; `7c72937c` mp4 604 / rig 599 -> +5."""
    assert count_delta(605, 599).offset == 6
    assert count_delta(604, 599).offset == 5


def test_count_delta_refuses_a_shorter_video():
    e = count_delta(590, 599)
    assert e.refused and e.reason == "negative_delta" and e.offset is None


def test_count_delta_is_zero_when_the_counts_agree():
    e = count_delta(599, 599)
    assert not e.refused and e.offset == 0


# --------------------------------------------------------------------------------- #
# the leader-pad estimator                                                           #
# --------------------------------------------------------------------------------- #
def test_leader_pad_finds_the_pad_on_a_moving_ego():
    head = [0.07, 0.02, 0.01, 0.01, 0.37, 0.02, 5.3, 5.6, 6.6, 8.1]   # 00040136, measured
    e = leader_pad(head, [20.5] * 9)
    assert not e.refused and e.offset == 6
    assert e.extra["static_head_block_frames"] == 7


def test_leader_pad_refuses_when_the_ego_is_stationary():
    """MEASURED: `7c72937c` has rig speed 0.0 m/s for its first 9 frames, so a frozen
    head block is not identifiable as a synthetic leader from the video alone."""
    head = [0.07, 0.02, 0.01, 0.01, 0.78, 0.71, 0.08, 0.09, 0.08, 0.83]
    e = leader_pad(head, [0.0] * 9)
    assert e.refused and e.reason == "ego_stationary_unidentifiable"
    assert e.offset is None


def test_leader_pad_refuses_a_head_that_never_moves():
    e = leader_pad([0.01] * 10, [20.0] * 9)
    assert e.refused and e.reason == "no_motion_in_head"


# --------------------------------------------------------------------------------- #
# consensus + uncertainty                                                            #
# --------------------------------------------------------------------------------- #
def test_consensus_reports_conflict_instead_of_averaging():
    a = AlignEstimate("count_delta", 6, False, "ok")
    b = AlignEstimate("motion_lag", 5, False, "ok")
    c = consensus([a, b])
    assert not c["agree"] and c["offset"] is None and c["conflict"] == [5, 6]


def test_consensus_ignores_refusals_but_records_them():
    a = AlignEstimate("count_delta", 6, False, "ok")
    b = AlignEstimate("motion_lag", None, True, "no_turnover")
    c = consensus([a, b])
    assert c["agree"] and c["offset"] == 6
    assert c["refusals"] == {"motion_lag": "no_turnover"}
    assert c["n_admissible"] == 1


def test_bootstrap_mass_concentrates_on_the_true_offset():
    rng = np.random.default_rng(5)
    curves = [{d: 0.30 - 0.02 * abs(d - 6) + 0.004 * rng.normal() for d in range(-5, 12)}
              for _ in range(12)]
    out = bootstrap_offset(curves, b=400, seed=1)
    assert out["point"] == 6 and not out["refused"]
    assert out["mass"].get(6, 0.0) > 0.8
    assert out["n_units"] == 12


def test_bootstrap_reports_spread_when_the_frames_disagree():
    """Two populations of frames disagreeing by one -> the mass must NOT be a point."""
    curves = ([{d: 0.30 - 0.02 * abs(d - 6) for d in range(-5, 12)}] * 6 +
              [{d: 0.30 - 0.02 * abs(d - 5) for d in range(-5, 12)}] * 6)
    out = bootstrap_offset(curves, b=400, seed=2)
    assert out["modal_mass"] < 0.999
    assert set(out["mass"]) <= {5, 6}
