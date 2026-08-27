"""Pin the three defects the PI's MERGE question exposed (2026-08-27).

He asked why `59b57590` emitted MERGE when its CoT never mentions one. It
doesn't — and answering that found two larger problems next to it.
"""
from tanitad.data import cot_tokens_v7 as COT

#: The exact text of `59b57590`, which the PI questioned.
CLIP = ("Change lanes to the left due to the right lane being constrained by "
        "parked vehicles and a curbside parking bay while the left lane is open "
        "and the traffic light ahead is green.  "
        "They are adjacent to the right lane and create potential "
        "door-opening/merge hazards; keeping extra clearance can motivate "
        "moving left.")


def test_a_hazard_class_is_not_a_merge_event():
    """⛔ "potential door-opening/merge hazards" is a RISK, not a manoeuvre."""
    g = COT.goals_from_cot(CLIP)
    assert "MERGE" not in g, f"phantom MERGE from a hazard noun phrase: {g}"


def test_the_lane_change_that_was_actually_stated_is_emitted():
    """⭐ 174 clips stated a lane change and NOT ONE produced a token."""
    g = COT.goals_from_cot(CLIP)
    assert "LANE_CHANGE_L" in g, f"the stated manoeuvre is missing: {g}"


def test_the_green_light_still_survives():
    assert "TRAFFIC_LIGHT_REACT_GREEN" in COT.goals_from_cot(CLIP)


def test_a_real_merge_still_fires():
    """The guard must not cost us genuine merges."""
    g = COT.goals_from_cot("Yield to the vehicle merging from the right ramp.")
    assert "MERGE" in g, g


def test_merge_into_a_named_lane_is_a_lane_change_not_a_merge():
    """"merge into the left lane" describes a lane change; the more specific
    token wins so the two cannot both fire on one phrase."""
    g = COT.goals_from_cot("Merge into the left lane ahead of the slower truck.")
    assert "LANE_CHANGE_L" in g and "MERGE" not in g, g


def test_both_sides_extract():
    assert "LANE_CHANGE_R" in COT.goals_from_cot("Change lanes to the right now.")
    assert "LANE_CHANGE_L" in COT.goals_from_cot("Move into the left lane.")


def test_a_negated_lane_change_does_not_fire():
    """Negation scoping still applies to the new extractor."""
    g = COT.goals_from_cot("Do not change lanes to the left; stay in this lane.")
    assert not any(k.startswith("LANE_CHANGE") for k in g), g
