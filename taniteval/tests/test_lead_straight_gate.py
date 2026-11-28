"""The straight-driving gate on lead selection (Sayed, 2026-08-26).

⛔ THE DEFECT THESE PIN. `select_lead_causal` searches a STRAIGHT ``|lat| < 2 m`` band in the
ego frame at t0, and it was applied through curves. On a bend the true lead is around the
corner, falls outside the band, and the window was recorded ``NO_LEAD`` -- which this module
defines as *"labels present, road genuinely clear"*. ⇒ **A curve manufactured an empty road**,
the same bias the ``NO_LABEL`` rule exists to prevent, arriving through a different door. Since
distance-keeping is half the family carrying 88.7 % of the oracle gap, a fabricated free-flow
window is not a rounding error -- it is a safe-looking number with nothing behind it.

⚠️ ``test_a_curve_is_never_recorded_as_clear_road`` is the load-bearing one; the rest guard the
gate against over-reach (it must not throw away windows whose claim IS supported).
"""
from __future__ import annotations

import numpy as np
import pytest

from taniteval.lead_source import (
    LEAD,
    LEAD_LAT_M,
    NO_LABEL,
    NO_LEAD,
    NOT_STRAIGHT,
    ego_lateral_departure,
    lead_block,
)

TS_REL = np.array([0.5, 1.0, 1.5, 2.0])
V = 10.0
LEAD_SIZE_X = 4.5


def _ego(curvature: float = 0.0, t_end: float = 30.0, dt: float = 0.1) -> dict:
    """A constant-speed ego track of constant curvature (0 = dead straight)."""
    t = np.arange(0.0, t_end, dt)
    yaw = curvature * V * t                      # unwrapped, as the module requires
    x = np.concatenate([[0.0], np.cumsum(np.cos(yaw[:-1]) * V * dt)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(yaw[:-1]) * V * dt)])
    return {"t": t, "x": x, "y": y, "yaw": yaw, "v": np.full_like(t, V)}


def _obs(gap_m: float | None, ego: dict) -> dict | None:
    """One vehicle held at a constant ``gap_m`` dead ahead, in the ego frame at each t."""
    t = ego["t"]
    if gap_m is None:                            # labels exist, road really is clear
        return {"t": t, "track": np.array(["none"] * t.size), "center_x": np.full(t.size, 200.0),
                "center_y": np.zeros(t.size), "size_x": np.full(t.size, LEAD_SIZE_X),
                "is_vehicle": np.ones(t.size, bool)}
    return {"t": t, "track": np.array(["lead"] * t.size),
            "center_x": np.full(t.size, gap_m + LEAD_SIZE_X / 2.0),
            "center_y": np.zeros(t.size), "size_x": np.full(t.size, LEAD_SIZE_X),
            "is_vehicle": np.ones(t.size, bool)}


def _state(ego: dict, obs, t0: float = 5.0) -> str:
    return lead_block([t0], TS_REL, obs, ego)["state"][0]


# --------------------------------------------------------------------------- #
# the defect                                                                    #
# --------------------------------------------------------------------------- #
def test_a_curve_is_never_recorded_as_clear_road():
    """⛔ THE LOAD-BEARING TEST. A curved window with no in-band vehicle must be
    NOT_STRAIGHT, never NO_LEAD -- 'clear road' is a claim the geometry cannot support."""
    # radius 500 m: lateral departure over the 80 m claim range is ~6.4 m, far outside
    # the 2 m corridor, so a straight band says nothing about the lane at 80 m.
    assert _state(_ego(curvature=1 / 500.0), _obs(None, _ego(curvature=1 / 500.0))) == NOT_STRAIGHT


def test_the_same_geometry_driven_straight_is_still_clear_road():
    """The gate must not simply relabel everything: straight + genuinely empty stays NO_LEAD."""
    ego = _ego(curvature=0.0)
    assert _state(ego, _obs(None, ego)) == NO_LEAD


# --------------------------------------------------------------------------- #
# the gate must not over-reach                                                  #
# --------------------------------------------------------------------------- #
def test_a_near_lead_survives_a_curve_that_would_kill_a_far_claim():
    """⭐ THE ASYMMETRY. With a lead at 15 m only the path out to 15 m must be straight.
    The identical curve that invalidates an 80 m 'clear' claim leaves this window usable."""
    ego = _ego(curvature=1 / 500.0)               # ~0.22 m departure at 15 m, ~6.4 m at 80 m
    assert _state(ego, _obs(15.0, ego)) == LEAD


def test_a_straight_road_with_a_lead_is_unchanged():
    ego = _ego(curvature=0.0)
    blk = lead_block([5.0], TS_REL, _obs(25.0, ego), ego)
    assert blk["state"][0] == LEAD
    assert blk["has_lead"][0]
    assert blk["gap0_m"][0] == pytest.approx(25.0, abs=0.5)


def test_no_label_still_outranks_the_straightness_gate():
    """A window with no labels at all is NO_LABEL whatever the geometry -- the gate may
    never convert missing data into a geometric verdict."""
    assert _state(_ego(curvature=1 / 500.0), None) == NO_LABEL


# --------------------------------------------------------------------------- #
# the reported denominator                                                      #
# --------------------------------------------------------------------------- #
def test_every_state_is_counted_and_the_counts_are_exhaustive():
    ego = _ego(curvature=0.0)
    blk = lead_block([5.0, 6.0, 7.0], TS_REL, _obs(25.0, ego), ego)
    assert sum(blk["counts"].values()) == 3
    assert NOT_STRAIGHT in blk["counts"]
    assert NOT_STRAIGHT in blk["conventions"]


def test_the_checked_distance_is_reported_not_absorbed():
    """A short check is weaker evidence; it must be visible per window."""
    ego = _ego(curvature=0.0)
    blk = lead_block([5.0], TS_REL, _obs(25.0, ego), ego)
    assert np.isfinite(blk["straight_checked_m"][0])
    # with a lead at 25 m the gate only needs to look ~25 m ahead, not the full 80
    assert blk["straight_checked_m"][0] <= 30.0


# --------------------------------------------------------------------------- #
# the helper itself                                                             #
# --------------------------------------------------------------------------- #
def test_departure_is_zero_on_a_straight_path_and_grows_with_curvature():
    ego = _ego(curvature=0.0)
    dep_straight, _ = ego_lateral_departure(5.0, ego["t"], ego["x"], ego["y"], ego["yaw"], 80.0)
    assert dep_straight == pytest.approx(0.0, abs=1e-6)

    curved = _ego(curvature=1 / 500.0)
    dep_curved, checked = ego_lateral_departure(
        5.0, curved["t"], curved["x"], curved["y"], curved["yaw"], 80.0)
    assert dep_curved > LEAD_LAT_M
    assert checked == pytest.approx(80.0, abs=1.0)


def test_departure_over_a_short_range_is_small_on_the_same_curve():
    """The gate is a function of the RANGE CLAIMED, not of the road alone."""
    curved = _ego(curvature=1 / 500.0)
    dep_near, checked = ego_lateral_departure(
        5.0, curved["t"], curved["x"], curved["y"], curved["yaw"], 15.0)
    assert dep_near < LEAD_LAT_M
    assert checked == pytest.approx(15.0, abs=1.0)


def test_no_future_track_reports_nan_rather_than_a_false_pass():
    ego = _ego(curvature=0.0, t_end=5.0)
    dep, checked = ego_lateral_departure(
        99.0, ego["t"], ego["x"], ego["y"], ego["yaw"], 80.0)
    assert np.isnan(dep)
    assert checked == 0.0
