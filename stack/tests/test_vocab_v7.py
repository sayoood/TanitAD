"""v7 vocabulary and emitter — the PI redesign of 2026-08-23.

Each arm pins one decision from the review, and several pin a defect that a
naive suite would pass over.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from tanitad.data import cot_tokens_v7 as COT
from tanitad.models import vocab_v7 as V7

HZ = 10.0


def _emitter():
    p = Path(__file__).resolve().parents[1] / "scripts" / "s2_geom_emit_v7.py"
    spec = importlib.util.spec_from_file_location("s2v7", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _drive(segments, hz=HZ):
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


# -- vocabulary shape -------------------------------------------------------
def test_strategic_tokens_all_carry_the_follow_route_meaning():
    """PI: the strategic layer is always 'the overnext manoeuvre TO follow the
    route', and the name must say so."""
    for t in V7.STRATEGIC_GOAL_TOKENS_V7:
        assert t == "FOLLOW_ROUTE" or t.endswith("_FOLLOW_ROUTE"), t
    for t in V7.STRATEGIC_ACTION_TOKENS_V7:
        assert t == "HOLD_MAIN_ROAD" or t.endswith("_FOLLOW_ROUTE"), t


def test_hold_main_road_replaced_hold_corridor_strategically():
    assert "HOLD_MAIN_ROAD" in V7.STRATEGIC_ACTION_TOKENS_V7
    assert "HOLD_CORRIDOR" not in V7.STRATEGIC_ACTION_TOKENS_V7


def test_tactical_goals_have_no_abstain_and_no_anchor_goal():
    """PI: remove ABSTAIN; the anchor is the ARG FRAME, not a goal."""
    assert not any("ABSTAIN" in t for t in V7.TACTICAL_GOAL_TOKENS_V7)
    assert "ANCHOR_GOAL" not in V7.TACTICAL_GOAL_TOKENS_V7
    assert V7.ANCHOR_ARG_SLOTS == ("goal_x_m", "goal_y_m", "t_reach_s")


def test_traffic_light_goals_carry_the_colour():
    for c in ("RED", "YELLOW", "GREEN"):
        assert f"TRAFFIC_LIGHT_REACT_{c}" in V7.TACTICAL_GOAL_TOKENS_V7
    assert "TRAFFIC_LIGHT_REACT" in V7.TACTICAL_GOAL_TOKENS_V7, \
        "the colourless fallback must remain"


def test_lon_actions_gained_accelerate_and_kept_the_curve_name():
    assert "ACCELERATE" in V7.TACTICAL_LON_ACTIONS_V7
    assert "ADAPT_SPEED_FOR_CURVE" in V7.TACTICAL_LON_ACTIONS_V7
    assert "ADAPT_SPEED_FOR_TURNING" not in V7.TACTICAL_LON_ACTIONS_V7


def test_no_v6_tuple_is_mutated_by_importing_v7():
    """The tensor contract: v7 adds names, it never edits v6."""
    from tanitad.models import v6
    assert v6.TACTICAL_LAT_ACTIONS == ("LANE_KEEP", "LANE_CHANGE_L",
                                       "LANE_CHANGE_R", "ABORT_LC",
                                       "NUDGE_L", "NUDGE_R")
    assert len(v6.STRATEGIC_ACTION_TOKENS) == 6


# -- the exclusion matrix ---------------------------------------------------
def test_a_valid_multi_goal_set_passes():
    assert V7.validate_goal_set(
        {"YIELD_FOR_TURN_L", "TURN_L", "TRAFFIC_LIGHT_REACT_RED"}) == []


@pytest.mark.parametrize("pair", [
    ("TURN_L", "TURN_R"), ("TURN_L", "FOLLOW_LANE"),
    ("TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_GREEN"),
    ("YIELD_FOR_TURN_L", "TURN_R"),
])
def test_forbidden_pairs_are_rejected(pair):
    assert V7.validate_goal_set(set(pair)), f"{pair} should be rejected"


def test_an_empty_goal_set_is_a_violation_not_an_abstain():
    assert V7.validate_goal_set(set()), "v7 has no ABSTAIN; must emit FOLLOW_LANE"


def test_geometry_may_not_emit_perception_tokens():
    allowed = V7.geometry_emittable(V7.TACTICAL_GOAL_TOKENS_V7)
    assert not (allowed & V7.TACTICAL_GOAL_NEEDS_PERCEPTION)
    assert "TRAFFIC_LIGHT_REACT_RED" not in allowed
    assert "TURN_L" in allowed


# -- CoT extraction ---------------------------------------------------------
@pytest.mark.parametrize("cot,colour", [
    ("Stop for the red traffic light.", "RED"),
    ("Slow down because of yellow traffic light ahead.", "YELLOW"),
    ("Keep speed through the intersection since the traffic light is green.", "GREEN"),
    ("Proceed at the traffic light.", "UNKNOWN"),
])
def test_traffic_light_colour_is_extracted(cot, colour):
    assert COT.extract(cot).traffic_light == colour


def test_react_on_oncoming_fires_on_any_oncoming_mention():
    """PI: react on oncoming — do NOT require the word 'wait'. Requiring it
    yielded 0 clips while 142 mention oncoming."""
    for cot in ("Nudge right due to an oncoming vehicle",
                "Keep lane because an oncoming vehicle is approaching",
                "Wait for the oncoming vehicle before turning"):
        assert "REACT_ON_ONCOMING" in COT.goals_from_cot(cot), cot
    assert "REACT_ON_ONCOMING" not in COT.goals_from_cot("Stop for the red light.")


def test_overtake_is_a_MOVING_vehicle_and_evade_is_static():
    """PI's distinction: overtake = passing a slower MOVING vehicle in front;
    evade = a lateral move for a static obstacle or a VRU. They share the verb,
    so only the OBJECT separates them — and 419 parked-car cases must not be
    filed as overtakes."""
    assert COT.extract("Overtake the slower truck in the right lane").overtake
    assert COT.extract("Nudge left to pass the truck ahead in the same lane").overtake
    ev = COT.extract("Nudge left to pass the parked car on the right")
    assert ev.overtake is False and ev.evade_obj == "PARKED"
    assert COT.extract("Nudge left to pass the cyclist in the same lane").evade_obj == "CYCLIST"


def test_yield_is_extracted_and_is_the_largest_semantic_class():
    """MEASURED 400/4729 = 8.5 % — larger than everything except the light."""
    assert COT.extract("Yield due to pedestrians in the crosswalk").yield_ == "HAZARD"
    assert COT.extract(
        "Slow down for the roundabout because of the yield sign ahead.").yield_ == "SIGN"
    assert "YIELD" in COT.goals_from_cot("yield due to a pedestrian in the crosswalk")


def test_merge_and_gap_are_extracted():
    assert "MERGE" in COT.goals_from_cot(
        "Slow down to create a gap in the adjacent left lane and prepare to merge left")
    assert "GAP_TARGET" in COT.goals_from_cot(
        "Slow down to create a gap in the adjacent left lane for a lane change")


def test_exit_comes_from_terms_not_geometry():
    """PI: extract exit from the CoT terms rather than from geometry."""
    assert "TAKE_EXIT_R" in COT.goals_from_cot(
        "Split to the right to take the off-ramp because the lane divides ahead.")
    assert "TAKE_EXIT_L" in COT.goals_from_cot(
        "Lane change to the left due to the current lane becoming an exit-only lane")


# -- goal <-> action linkage (PI: document which actions serve which goals) --
def test_every_goal_has_an_admissible_action_set():
    for g in V7.TACTICAL_GOAL_TOKENS_V7:
        assert g in V7.GOAL_ADMISSIBLE_LAT, g
        assert g in V7.GOAL_ADMISSIBLE_LON, g
        assert V7.GOAL_ADMISSIBLE_LAT[g] and V7.GOAL_ADMISSIBLE_LON[g]


def test_admissible_actions_are_real_tokens():
    for m, vocab in ((V7.GOAL_ADMISSIBLE_LAT, V7.TACTICAL_LAT_ACTIONS_V7),
                     (V7.GOAL_ADMISSIBLE_LON, V7.TACTICAL_LON_ACTIONS_V7)):
        for g, acts in m.items():
            for a in acts:
                assert a in vocab, (g, a)


def test_the_contradiction_shape_is_detected():
    """`TURN_LEFT` goal with a hold-speed action serves the goal laterally but
    NOT longitudinally — the shape the PI found on 0e56dae2."""
    bad = V7.action_serves_goals("LANE_KEEP", "CRUISE", {"TURN_L"})
    assert bad["lon_serves"] is False
    good = V7.action_serves_goals("TURN_L", "ADAPT_SPEED_FOR_CURVE", {"TURN_L"})
    assert good["lat_serves"] and good["lon_serves"]


def test_speed_band_is_back_in_the_vocabulary():
    """PI: 'the target speed band was there before, it disappeared.'"""
    assert "SPEED_BAND" in V7.TACTICAL_GOAL_TOKENS_V7


# -- the emitter: ordinal layering -----------------------------------------
def test_one_manoeuvre_gives_a_tactical_turn_and_follow_route_strategically():
    """`01b24287`'s shape: a single turn ⇒ nothing overnext."""
    m = _emitter()
    poses = _drive([(8.0, None, 3.0), (6.0, 10.0, 4.0), (8.0, None, 25.0)])
    seq = m.manoeuvre_sequence(poses, 0)
    assert len(seq) == 1, seq
    g, a = m.strategic(poses, 0, seq)
    assert g["token"] == "FOLLOW_ROUTE"
    assert a["token"] == "HOLD_MAIN_ROAD"
    goals, anchor, viol = m.tactical_goals(poses, 0, seq, None)
    assert "TURN_L" in goals and not viol


def test_two_manoeuvres_put_the_second_on_the_strategic_layer_with_args():
    """`01bee851`'s shape: the overnext manoeuvre needs distance and time."""
    m = _emitter()
    poses = _drive([(8.0, None, 2.0), (6.0, 10.0, 3.0), (8.0, None, 8.0),
                    (6.0, -10.0, 3.0), (8.0, None, 15.0)])
    seq = m.manoeuvre_sequence(poses, 0)
    assert len(seq) >= 2, seq
    g, a = m.strategic(poses, 0, seq)
    assert g["token"].endswith("_FOLLOW_ROUTE") and g["token"] != "FOLLOW_ROUTE"
    assert g["args"]["within_m"] > 0 and g["args"]["by_time_s"] > 0, g["args"]
    assert a["token"].startswith("PREPARE_TURN_")


def test_the_anchor_is_always_present_as_args():
    m = _emitter()
    poses = _drive([(9.0, None, 12.0)])
    _, anchor, _ = m.tactical_goals(poses, 0, [], None)
    assert set(anchor) == set(V7.ANCHOR_ARG_SLOTS)


def test_acceleration_is_named_not_filed_as_cruise():
    m = _emitter()
    poses = _drive([(2.0, None, 1.0), (6.0, None, 2.0), (12.0, None, 4.0)])
    assert m.tactical_actions(poses, 0, [])["lon"] == "ACCELERATE"


def test_a_turn_sets_the_curve_action():
    m = _emitter()
    poses = _drive([(8.0, None, 1.0), (6.0, 9.0, 4.0), (7.0, None, 3.0)])
    assert m.tactical_actions(poses, 0, [])["lon"] == "ADAPT_SPEED_FOR_CURVE"


def test_nav_command_is_marked_oracle():
    """⛔ derived from the ego future ⇒ training input only, never a
    vision-only eval input."""
    m = _emitter()
    poses = _drive([(8.0, None, 3.0), (6.0, 10.0, 4.0), (8.0, None, 10.0)])
    nav = m.nav_command(poses, 0, m.manoeuvre_sequence(poses, 0))
    assert nav["oracle"] is True
    assert nav["provenance"] == "ego-future"
    assert nav["token"] in V7.NAV_COMMAND_TOKENS
    assert set(nav["args"]) == set(V7.NAV_ARG_SLOTS)


def test_nav_follow_road_when_nothing_ahead():
    m = _emitter()
    poses = _drive([(10.0, None, 20.0)])
    assert m.nav_command(poses, 0, [])["token"] == "NAV_FOLLOW_ROAD"
