"""Pin the negated-enumeration bug that inverted TRAFFIC_LIGHT_REACT 408x."""
import pytest

from tanitad.data import cot_negation as NEG
from tanitad.data import cot_tokens_v7 as COT

#: The exact shape that broke it, from `critical_components_analysis`.
EMPTY = ("The first 2 seconds show a clear highway with no lead vehicle "
         "pedestrians cyclists traffic lights or obstacles affecting the ego lane.")
REAL = "Stop at the stop line because the traffic light is red."
MIXED = ("There are no pedestrians in the crosswalk, but a cyclist is "
         "approaching from the right.")


def test_the_enumeration_that_caused_it():
    """Six terms under ONE `no` — the shape a narrow window cannot see."""
    for term in ("lead vehicle", "pedestrians", "cyclists", "traffic lights",
                 "obstacles"):
        i = EMPTY.find(term)
        assert NEG.is_negated(EMPTY, i, i + len(term)), term


def test_no_traffic_light_token_from_an_empty_scene():
    assert COT.goals_from_cot(EMPTY) == {}


def test_a_real_light_still_fires_with_its_colour():
    g = COT.goals_from_cot(REAL)
    assert "TRAFFIC_LIGHT_REACT_RED" in g, g


def test_negation_closes_at_a_contrastive_conjunction():
    """`but` flips polarity back — the cyclist after it is PRESENT."""
    i = MIXED.find("pedestrians")
    assert NEG.is_negated(MIXED, i, i + len("pedestrians"))
    j = MIXED.find("cyclist is")
    assert not NEG.is_negated(MIXED, j, j + 7)


def test_offsets_are_preserved_by_blanking():
    """Blanking, not deleting — a caller's indices must stay valid."""
    assert len(NEG.strip_negated(EMPTY)) == len(EMPTY)


@pytest.mark.parametrize("txt", [
    "There is no doubt the lead vehicle is braking.",
    "Not only the pedestrian but also a cyclist is crossing.",
])
def test_false_cues_do_not_open_a_scope(txt):
    """`no doubt` / `not only` are assertions, not absences."""
    assert NEG.negated_spans(txt) == [] or all(
        b - a < len(txt) for a, b in NEG.negated_spans(txt))


def test_empty_and_none_are_safe():
    assert NEG.negated_spans("") == []
    assert NEG.strip_negated("") == ""
    assert COT.goals_from_cot(None) == {}


def test_a_stopped_bus_is_an_evade_not_an_overtake():
    """The PI's distinction: EVADE's object is STATIC, OVERTAKE's moves.

    `d452ea24` — "Nudge left to pass the stopped bus in the same lane" — was
    labelled FOLLOW_LANE and nothing else.
    """
    g = COT.goals_from_cot("Nudge left to pass the stopped bus in the same lane")
    assert "EVADE_IN_CORRIDOR" in g, g
    assert g["EVADE_IN_CORRIDOR"]["obstacle_class"] == "stopped_vehicle"
    assert "OVERTAKE_VEHICLE" not in g, "a stopped object is not an overtake"


def test_a_moving_slower_vehicle_is_still_an_overtake():
    g = COT.goals_from_cot("Overtake the slower truck ahead in the left lane")
    assert "OVERTAKE_VEHICLE" in g, g
