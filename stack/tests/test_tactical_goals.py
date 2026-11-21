"""Tests for hindsight tactical-goal derivation.

⭐ TWO ARMS PIN DEFECTS THE PI FOUND BY LOOKING AT FRAMES, and both would pass a
naive test suite:

  * ``test_a_late_turn_still_yields_a_tactical_goal`` — a turn beginning at
    t+6.1 s reads `lane_keep` on the 2 s `a_tac` window while the frames plainly
    show a right turn. `g_tac` is defined over the 2-6 s band, so it must not
    inherit that blindness.
  * ``test_an_unambiguous_turn_is_never_a_strategic_abstain`` — clip 5aef0388
    executes an 89 deg turn at R = 12.2 m and the pipeline emitted
    NONE_ABSTAIN. Abstention is for ambiguous geometry; this is not that.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tanitad.data import ego_manoeuvre as EM
from tanitad.data import tactical_goals as TG

HZ = 10.0


def _drive(segments, hz=HZ):
    """segments = [(speed, radius_or_None, seconds), ...]; None radius = straight."""
    xs, ys, yaws, vs = [], [], [], []
    x = y = yaw = 0.0
    for v, radius, secs in segments:
        for _ in range(int(round(secs * hz))):
            if radius:
                yaw += (v / hz) / radius
            x += v / hz * math.cos(yaw)
            y += v / hz * math.sin(yaw)
            xs.append(x); ys.append(y); yaws.append(yaw); vs.append(v)
    return np.stack([xs, ys, yaws, vs], axis=1)


def _profile(speeds, hz=HZ):
    """Straight-line poses with a given speed profile."""
    x = np.cumsum(np.asarray(speeds, float) / hz)
    return np.stack([x, np.zeros_like(x), np.zeros_like(x),
                     np.asarray(speeds, float)], axis=1)


def _with_lateral(poses, offset_fn, hz=HZ):
    """Add a lateral profile AND re-derive yaw from the resulting path.

    ⚠️ Shifting y alone leaves yaw at 0, so the classifier sees a straight road
    and no lateral token can ever fire. The heading must follow the path.
    """
    p = poses.copy()
    t = np.arange(len(p)) / hz
    p[:, 1] = p[:, 1] + offset_fn(t)
    dx = np.gradient(p[:, 0]); dy = np.gradient(p[:, 1])
    p[:, 2] = np.arctan2(dy, dx)
    return p


def test_anchor_goal_is_the_default_not_an_abstain():
    g = TG.derive(_drive([(10.0, None, 10.0)]))
    assert g.lat_token == "ANCHOR_GOAL"
    assert g.lat_args["t_reach_s"] == pytest.approx(6.0, abs=0.2)
    assert g.lat_args["goal_x_m"] > 30


def test_a_held_speed_yields_a_speed_band():
    g = TG.derive(_drive([(9.0, None, 10.0)]))
    assert g.lon_token == "SPEED_BAND"
    assert g.lon_args["v_lo_ms"] < 9.0 < g.lon_args["v_hi_ms"]


def test_a_stop_in_the_band_yields_a_stop_point():
    poses = _drive([(8.0, None, 2.5)] + [(v, None, 0.1) for v in
                                         np.linspace(8, 0, 15)]
                   + [(0.0, None, 3.0)])
    g = TG.derive(poses)
    assert g.lon_token == "STOP_POINT"
    assert g.lon_args["reason"] == "unknown"


def test_the_vlm_may_refine_a_stop_reason_but_never_invent_the_stop():
    poses = _drive([(8.0, None, 2.5)] + [(v, None, 0.1) for v in
                                         np.linspace(8, 0, 15)]
                   + [(0.0, None, 3.0)])
    with_reason = TG.derive(poses, stop_reason="light")
    assert with_reason.lon_token == "STOP_POINT"
    assert with_reason.lon_args["reason"] == "light"
    assert "vlm(light)" in with_reason.lon_provenance
    # ...and on a clip that never stops, the same reason creates nothing
    cruising = TG.derive(_drive([(10.0, None, 10.0)]), stop_reason="light")
    assert cruising.lon_token != "STOP_POINT"


def test_a_late_turn_still_yields_a_tactical_goal():
    """The 2 s a_tac window says lane_keep; the 2-6 s g_tac band must not."""
    poses = _drive([(6.0, None, 4.0), (6.0, 12.0, 4.0), (6.0, None, 4.0)])
    g = TG.derive(poses)
    assert g.lat_token == "ANCHOR_GOAL"
    # the goal point must actually reflect the turn, not sit straight ahead
    assert abs(g.lat_args["goal_y_m"]) > 3.0, g.lat_args


def test_band_is_2_to_6_seconds_not_the_2_second_action_window():
    g = TG.derive(_drive([(10.0, None, 10.0)]))
    assert tuple(g.band_s) == (2.0, 6.0)


def test_truncation_is_reported_not_hidden():
    g = TG.derive(_drive([(10.0, None, 3.0)]))
    assert g.truncated is True


# -- the strategic abstain defect -------------------------------------------
def test_an_unambiguous_turn_is_never_a_strategic_abstain():
    for radius, want in [(12.0, "TURN_LEFT"), (-12.0, "TURN_RIGHT")]:
        poses = _drive([(5.0, radius, 4.0), (5.0, None, 4.0)])
        man = EM.analyse(poses)
        tok, why = TG.strategic_from_geometry(man)
        assert tok == want, (tok, why, man.lateral_class)
        assert tok != "NONE_ABSTAIN"


def test_a_straight_road_is_follow_main_road_not_abstain():
    man = EM.analyse(_drive([(12.0, None, 10.0)]))
    assert TG.strategic_from_geometry(man)[0] == "FOLLOW_MAIN_ROAD"


def test_a_bend_is_follow_main_road_not_a_turn():
    man = EM.analyse(_drive([(14.0, 220.0, 8.0)]))
    tok, _ = TG.strategic_from_geometry(man)
    assert tok == "FOLLOW_MAIN_ROAD", man.lateral_class


def test_every_emitted_token_is_in_the_v6_vocabulary():
    """The tuples size live embedding tables; an unknown token is a crash.

    ⚠️ The LON side is checked against v6.1 because `ADAPT_SPEED_FOR_CURVE` is
    a v6.1 append — and the emitter must only produce it when v6.1 is selected,
    which `test_the_curve_token_is_absent_under_v60` pins separately.
    """
    from tanitad.models import v6
    for t in TG.LAT_TOKENS:
        assert t in v6.TACTICAL_GOAL_TOKENS_LAT
    v61 = set(v6.tactical_lon_goals("v6.1"))
    for t in TG.LON_TOKENS:
        assert t in v61, t
    # everything except the v6.1 append must ALSO exist in v6.0
    v60 = set(v6.tactical_lon_goals("v6.0"))
    for t in set(TG.LON_TOKENS) - {"ADAPT_SPEED_FOR_CURVE"}:
        assert t in v60, t


def test_goal_args_are_expressed_in_the_ego_frame_at_the_key():
    """A goal point must be relative to where the planner sits, not the origin."""
    poses = _drive([(10.0, None, 12.0)])
    g0 = TG.derive(poses, key=0)
    g40 = TG.derive(poses, key=40)
    # same geometry, different anchor -> both goal points ~6 s ahead of THEIR key
    assert g0.lat_args["goal_x_m"] == pytest.approx(g40.lat_args["goal_x_m"],
                                                    rel=0.05)


# -- defects found by emitting at corpus scale (801 clips) ------------------
def test_evade_needs_a_floor_not_only_a_ceiling():
    """59.8% of EVADE emissions were lane-keeping jitter before this."""
    # a tiny drift must NOT be an evasion
    tiny = _drive([(8.0, 4000.0, 8.0)])
    g = TG.derive(tiny)
    assert g.lat_token != "EVADE_IN_CORRIDOR", g.lat_args


def test_a_real_evasion_still_fires():
    poses = _with_lateral(_drive([(8.0, None, 9.0)]),
                          lambda t: 1.8 * np.exp(-((t - 4.0) ** 2) / 1.2))
    g = TG.derive(poses)
    assert g.lat_token in ("EVADE_IN_CORRIDOR", "CORRIDOR_OFFSET"), g.lat_token


def test_a_stationary_ego_does_not_get_a_goal_point_on_top_of_itself():
    """4 goals were BEHIND the car and 22 within 2 m before this."""
    g = TG.derive(_profile([0.0] * 100))
    assert g.lat_token == "LAT_UNCONSTRAINED"
    assert "near-stationary" in g.lat_provenance


def test_a_moving_ego_still_gets_its_anchor_goal():
    g = TG.derive(_drive([(10.0, None, 10.0)]))
    assert g.lat_token == "ANCHOR_GOAL"
    assert g.lat_args["goal_x_m"] > TG.ANCHOR_MIN_M


def test_evade_requires_an_out_and_back_not_a_lateral_shift():
    """0/40 emissions had the return signature before this: all lane shifts."""
    poses = _with_lateral(_drive([(8.0, None, 9.0)]),
                          lambda t: 2.0 / (1.0 + np.exp(-(t - 4.0) * 3.0)))
    g = TG.derive(poses)
    assert g.lat_token != "EVADE_IN_CORRIDOR", (g.lat_token, g.lat_args)
    assert g.lat_token == "CORRIDOR_OFFSET"


def test_a_true_out_and_back_is_still_an_evasion():
    poses = _with_lateral(_drive([(8.0, None, 9.0)]),
                          lambda t: 2.0 * np.exp(-((t - 4.0) ** 2) / 0.6))
    g = TG.derive(poses)
    assert g.lat_token == "EVADE_IN_CORRIDOR", (g.lat_token, g.lat_args)


# -- ADAPT_SPEED_FOR_CURVE (PI 2026-08-23) ---------------------------------
def test_slowing_into_a_curve_is_not_a_speed_band():
    """The longitudinal partner of a turn: speed governed by curvature."""
    poses = _drive([(12.0, None, 2.0), (11.0, 20.0, 1.0), (7.0, 12.0, 3.0),
                    (7.0, None, 3.0)])
    g = TG.derive(poses, lon_vocab="v6.1")
    assert g.lon_token == "ADAPT_SPEED_FOR_CURVE", (g.lon_token, g.lon_args)
    assert g.lon_args["v_apex_ms"] < g.lon_args["v_approach_ms"]


def test_a_turn_at_a_held_speed_is_STILL_governed_by_the_curve():
    """PI clarification 2026-08-23: the trigger is the TURNING MANOEUVRE, not a
    deceleration and not the word "curve". A turn taken at an already-suitable
    speed is still speed governed by that curve — `dv_ms` records how much the
    ego actually slowed, so the two cases stay distinguishable without a second
    token."""
    poses = _drive([(8.0, 30.0, 9.0)])
    g = TG.derive(poses, lon_vocab="v6.1")
    assert g.lon_token == "ADAPT_SPEED_FOR_CURVE", (g.lon_token, g.lon_args)
    assert abs(g.lon_args["dv_ms"]) < 1.0, "held speed should show ~no dv"


def test_a_straight_road_at_a_held_speed_is_a_speed_band():
    """The negative control: no arc, no curve token."""
    g = TG.derive(_drive([(9.0, None, 9.0)]), lon_vocab="v6.1")
    assert g.lon_token == "SPEED_BAND", g.lon_args


def test_decelerating_to_a_stop_is_a_stop_point_not_a_curve():
    """A red-light approach also decelerates; it must not steal this token."""
    poses = _drive([(9.0, 15.0, 2.0)] + [(v, 15.0, 0.1) for v in
                                         np.linspace(9, 0, 20)]
                   + [(0.0, None, 3.0)])
    assert TG.derive(poses, lon_vocab="v6.1").lon_token == "STOP_POINT"


def test_a_straight_deceleration_is_not_a_curve():
    poses = _drive([(12.0, None, 3.0), (6.0, None, 6.0)])
    assert TG.derive(poses, lon_vocab="v6.1").lon_token != "ADAPT_SPEED_FOR_CURVE"


def test_the_curve_token_is_absent_under_v60():
    """v6.0 has no such token; the emitter must not invent it."""
    poses = _drive([(12.0, None, 2.0), (11.0, 20.0, 1.0), (7.0, 12.0, 3.0),
                    (7.0, None, 3.0)])
    g = TG.derive(poses, lon_vocab="v6.0")
    assert g.lon_token != "ADAPT_SPEED_FOR_CURVE"


def test_every_lon_token_is_in_the_v61_vocabulary():
    from tanitad.models import v6
    v61 = set(v6.tactical_lon_goals("v6.1"))
    for t in TG.LON_TOKENS:
        assert t in v61, t
