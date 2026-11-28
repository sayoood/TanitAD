"""Tests for `tanitad.eval.constraints`.

⛔ THE GOVERNING RULE HERE: a guard needs MUTATION, not inspection. Every test
that asserts a constraint is SATISFIED is paired with one that reintroduces the
violation and asserts the metric FAILS it. A gate that cannot fail a knowingly
bad arm says nothing when it passes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tanitad.eval.constraints import (
    A_LAT_MAX_MS2,
    CLEARANCE_DATUM,
    D0_STANDOFF_M,
    TAU_HEADWAY_S,
    KAPPA_MAX_1_M,
    LEAD_LAT_M,
    clearance_speed_ceiling,
    combined_ceiling,
    curvature_from_xy,
    kinematic_speed_ceiling,
    lead_gap,
    point_to_box_clearance,
    speed_envelope_report,
    trajectory_clearance,
)


# --- lead gap: the corridor test that the GT control forced ------------------

def test_lead_gap_uses_the_lead_source_convention():
    """gap = cx - l/2, rig origin to the box's REAR face."""
    assert lead_gap([(20.0, 0.0, 0.0, 4.0, 2.0)]) == pytest.approx(18.0)


def test_an_agent_BEHIND_the_ego_is_not_a_lead():
    assert lead_gap([(-20.0, 0.0, 0.0, 4.0, 2.0)]) is None


def test_an_agent_BESIDE_the_ego_is_not_a_lead():
    """⛔ THE DEFECT THIS PINS. Taking the min distance to ANY agent made the
    GT trajectory violate its own speed envelope on 71.2 % of steps, because an
    adjacent-lane or parked car 2 m to the side was priced as a speed
    constraint. Only what is AHEAD and IN CORRIDOR bounds speed."""
    beside = [(1.0, LEAD_LAT_M + 0.5, 0.0, 4.0, 2.0)]
    assert lead_gap(beside) is None
    # ...while the proximity family still sees it, because it is genuinely close
    r = trajectory_clearance(np.zeros((1, 2)), beside)
    assert not r.censored and r.min_clearance_m < 2.0


def test_lead_gap_takes_the_NEAREST_in_corridor_agent():
    boxes = [(40.0, 0.0, 0.0, 4.0, 2.0), (15.0, 0.5, 0.0, 4.0, 2.0)]
    assert lead_gap(boxes) == pytest.approx(13.0)


def test_lead_gap_is_None_on_an_empty_corridor_never_zero():
    """None means CENSOR. Zero would mean 'a lead is touching us'."""
    assert lead_gap([]) is None


def test_lead_gap_never_returns_negative():
    assert lead_gap([(1.0, 0.0, 0.0, 6.0, 2.0)]) == 0.0


# --- the curvature clamp -----------------------------------------------------

def test_absurd_curvature_is_CLAMPED_not_propagated():
    """⛔ MEASURED: an unclamped double difference gave a ceiling minimum of
    0.279 m/s = kappa 43.7 1/m = a 2.3 cm turning radius. Pose jitter amplified
    twice, read as a constraint."""
    v_absurd, ok = kinematic_speed_ceiling([50.0])
    v_at_cap, _ = kinematic_speed_ceiling([KAPPA_MAX_1_M])
    assert ok[0]
    assert v_absurd[0] == pytest.approx(v_at_cap[0])
    assert v_absurd[0] > 4.0        # no longer below walking pace


def test_the_clamp_does_not_touch_realistic_curvature():
    v, _ = kinematic_speed_ceiling([0.01])
    assert v[0] == pytest.approx(math.sqrt(A_LAT_MAX_MS2 / 0.01))


# --- kinematic ceiling -------------------------------------------------------

def test_kinematic_ceiling_matches_the_closed_form():
    # a 100 m radius curve is kappa = 0.01 1/m
    v, ok = kinematic_speed_ceiling([0.01])
    assert ok[0]
    assert v[0] == pytest.approx(math.sqrt(A_LAT_MAX_MS2 / 0.01))


def test_kinematic_ceiling_is_sign_invariant():
    left, _ = kinematic_speed_ceiling([+0.02])
    right, _ = kinematic_speed_ceiling([-0.02])
    assert left[0] == pytest.approx(right[0])


def test_a_straight_path_is_CENSORED_not_given_an_enormous_ceiling():
    """⛔ The trap this pins: handing a straight path v_max = 1e6 would make
    every straight window trivially 'satisfied' and silently inflate the pass
    rate. Straight must be CENSORED for this family."""
    v, ok = kinematic_speed_ceiling([0.0, 1e-9, 1e-4])
    assert not ok.any()
    assert np.isnan(v).all()


def test_tighter_curve_gives_a_LOWER_ceiling():
    v, _ = kinematic_speed_ceiling([0.005, 0.02, 0.08])
    assert v[0] > v[1] > v[2]


# --- clearance ceiling -------------------------------------------------------

def test_clearance_ceiling_inverts_the_distance_keeping_cost():
    gap = 20.0
    v, ok = clearance_speed_ceiling([gap])
    assert ok[0]
    assert v[0] == pytest.approx((gap - D0_STANDOFF_M) / TAU_HEADWAY_S)
    # the banked cost's own predicate must hold at the ceiling
    assert D0_STANDOFF_M + TAU_HEADWAY_S * v[0] == pytest.approx(gap)


def test_a_gap_inside_the_standoff_gives_a_ZERO_ceiling_not_a_negative_one():
    v, ok = clearance_speed_ceiling([1.0, 0.0])
    assert ok.all()
    assert (v == 0.0).all()


# --- geometry ----------------------------------------------------------------

def test_point_to_box_clearance_on_an_axis_aligned_box():
    box = (10.0, 0.0, 0.0, 4.0, 2.0)          # rear face at x=8, sides at |y|=1
    assert point_to_box_clearance(0.0, 0.0, box) == pytest.approx(8.0)
    assert point_to_box_clearance(10.0, 5.0, box) == pytest.approx(4.0)


def test_a_point_INSIDE_the_box_reads_zero_never_negative():
    """A penetration is a collision. A negative depth would let a deep
    penetration average away against slack elsewhere."""
    box = (10.0, 0.0, 0.0, 4.0, 2.0)
    assert point_to_box_clearance(10.0, 0.0, box) == 0.0


def test_rotating_the_box_changes_the_clearance():
    """Inspection-proof: if yaw were ignored the two would be equal."""
    p = (0.0, 0.0)
    flat = point_to_box_clearance(*p, (10.0, 0.0, 0.0, 8.0, 2.0))
    turned = point_to_box_clearance(*p, (10.0, 0.0, math.pi / 2, 8.0, 2.0))
    assert flat == pytest.approx(6.0)
    assert turned == pytest.approx(9.0)
    assert abs(flat - turned) > 1.0


def test_curvature_of_a_known_circle():
    R = 50.0
    th = np.linspace(0, 0.6, 40)
    xy = np.column_stack([R * np.sin(th), R * (1 - np.cos(th))])
    k, good = curvature_from_xy(xy)
    assert good[5:-5].all()
    assert np.abs(k[5:-5]).mean() == pytest.approx(1.0 / R, rel=0.05)


def test_curvature_of_a_straight_line_is_zero():
    xy = np.column_stack([np.arange(20.0), np.zeros(20)])
    k, _ = curvature_from_xy(xy)
    assert np.abs(k).max() < 1e-9


# --- trajectory clearance + CENSORING ---------------------------------------

def _straight(n=10, dx=2.0):
    return np.column_stack([np.arange(n) * dx, np.zeros(n)])


def test_trajectory_clearance_finds_the_nearest_box():
    boxes = [(40.0, 0.0, 0.0, 4.0, 2.0), (12.0, 3.0, 0.0, 4.0, 2.0)]
    r = trajectory_clearance(_straight(), boxes)
    assert not r.censored
    assert r.n_boxes_considered == 2
    assert r.min_clearance_m == pytest.approx(2.0)   # |3| - 2/2 at x=12


def test_an_empty_scene_is_CENSORED_not_scored_as_infinite_clearance():
    """⛔ The failure this pins: scoring 'no agent in range' as a perfect
    clearance rewards blindness -- an arm that perceives nothing would win."""
    r = trajectory_clearance(_straight(), [])
    assert r.censored
    assert math.isnan(r.min_clearance_m)
    assert "no agent box" in r.reason


def test_boxes_beyond_max_range_are_excluded_and_that_censors_the_window():
    r = trajectory_clearance(_straight(), [(500.0, 0.0, 0.0, 4.0, 2.0)],
                             max_range_m=60.0)
    assert r.censored and r.n_boxes_considered == 0


def test_ego_footprint_inflation_reduces_clearance_and_never_goes_negative():
    boxes = [(12.0, 3.0, 0.0, 4.0, 2.0)]
    base = trajectory_clearance(_straight(), boxes).min_clearance_m
    infl = trajectory_clearance(_straight(), boxes, ego_half_w=1.0).min_clearance_m
    assert infl < base
    huge = trajectory_clearance(_straight(), boxes, ego_half_w=50.0).min_clearance_m
    assert huge == 0.0


def test_clearance_datum_names_the_convention():
    """A number without its datum is not quotable -- two rival gap conventions
    exist in this repo and differ by EGO_LEN = 4.7 m."""
    d = CLEARANCE_DATUM.lower()
    assert "rig-origin" in d, "the datum must name which origin it measures from"
    assert "not subtracted" in d, "it must say the ego footprint is not removed"
    # and it must name the RIVAL convention, or the two get mixed silently
    assert "ego_len" in d and "4.7" in d


# --- the two-sided envelope --------------------------------------------------

def test_a_compliant_arm_reads_zero_on_BOTH_sides():
    v = np.full(10, 12.0)
    c = np.full(10, 14.0)          # ceiling high => situation allows
    r = speed_envelope_report(v, v_ceiling=c, ceiling_valid=np.ones(10, bool))
    assert r["status"] == "OK"
    assert r["frac_over_ceiling"] == 0.0
    assert r["frac_under_when_allowed"] == 0.0


def test_DELIBERATE_REGRESSION_an_overdriving_arm_FAILS_the_over_side():
    """Mutation, not inspection: reintroduce the violation and require it fires."""
    v = np.full(10, 25.0)
    c = np.full(10, 14.0)
    r = speed_envelope_report(v, v_ceiling=c, ceiling_valid=np.ones(10, bool))
    assert r["frac_over_ceiling"] == 1.0
    assert r["mean_overshoot_ms"] == pytest.approx(11.0)


def test_DELIBERATE_REGRESSION_a_CREEPING_arm_FAILS_the_under_side():
    """⭐ THE PI'S TWO-SIDED ASK, pinned. An arm that always crawls satisfies
    every ceiling; without this half of the metric it would score perfectly."""
    v = np.full(10, 1.0)
    c = np.full(10, 20.0)          # situation plainly allows speed
    r = speed_envelope_report(v, v_ceiling=c, ceiling_valid=np.ones(10, bool))
    assert r["frac_over_ceiling"] == 0.0          # it never speeds
    assert r["frac_under_when_allowed"] == 1.0    # and it is caught anyway
    assert r["mean_shortfall_ms"] == pytest.approx(19.0)


def test_slowness_is_NOT_penalised_where_the_ceiling_is_low():
    """Penalising slowness behind a lead would make the metric reward
    recklessness. The under side is scored only where the situation allows."""
    v = np.full(10, 1.0)
    c = np.full(10, 2.0)           # low ceiling: obstructed
    r = speed_envelope_report(v, v_ceiling=c, ceiling_valid=np.ones(10, bool))
    assert r["n_situation_allows"] == 0
    assert r["frac_under_when_allowed"] is None


def test_the_two_sides_are_never_pooled_into_one_score():
    """A single scalar would let an over-driving win buy an under-driving loss."""
    v = np.array([25.0] * 5 + [1.0] * 5)
    c = np.full(10, 14.0)
    r = speed_envelope_report(v, v_ceiling=c, ceiling_valid=np.ones(10, bool))
    assert r["frac_over_ceiling"] == 0.5
    assert r["frac_under_when_allowed"] == 0.5
    assert not any(k.endswith("_score") for k in r)


def test_censored_steps_are_counted_not_dropped():
    v = np.full(10, 12.0)
    c = np.full(10, 14.0)
    ok = np.array([True] * 6 + [False] * 4)
    r = speed_envelope_report(v, v_ceiling=c, ceiling_valid=ok)
    assert r["n_scored"] == 6 and r["n_censored"] == 4


def test_a_fully_censored_window_reports_CENSORED_not_a_perfect_score():
    r = speed_envelope_report(np.full(5, 12.0), v_ceiling=np.full(5, np.nan),
                              ceiling_valid=np.zeros(5, bool))
    assert r["status"] == "CENSORED" and r["n_scored"] == 0


def test_the_vacuity_gate_carries_the_manoeuvre_rate():
    """⛔ MEASURED 2026-09-06: a friction-circle zero was bought by a turn_left
    recall of exactly 0.0000. A safety zero means nothing without the rate."""
    v, c = np.full(10, 12.0), np.full(10, 20.0)
    r = speed_envelope_report(v, v_ceiling=c, ceiling_valid=np.ones(10, bool),
                              manoeuvre_flag=np.zeros(10, bool))
    assert r["manoeuvre_rate"] == 0.0
    assert "declining" in r["_vacuity_warning"]


# --- combined ceiling --------------------------------------------------------

def test_combined_ceiling_takes_the_binding_constraint():
    out, ok = combined_ceiling([20.0, 5.0], [True, True], [8.0, 30.0],
                               [True, True])
    assert ok.all()
    assert out[0] == pytest.approx(8.0)     # clearance binds
    assert out[1] == pytest.approx(5.0)     # curvature binds


def test_one_valid_ceiling_is_enough_to_score_a_step():
    out, ok = combined_ceiling([np.nan], [False], [9.0], [True])
    assert ok[0] and out[0] == pytest.approx(9.0)


def test_a_step_with_NEITHER_ceiling_is_censored():
    out, ok = combined_ceiling([np.nan], [False], [np.nan], [False])
    assert not ok[0] and np.isnan(out[0])
