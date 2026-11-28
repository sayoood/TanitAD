"""Pin the CORRIDOR_OFFSET CoT extraction (PI-designed, 2026-08-28).

The token was declared unreachable for want of a calibrated GEOMETRIC
threshold. The PI's route needs none: extract the side from CoT TERMS, with
exactly one constraint (left|right) and no invented magnitude.

⛔ The load-bearing subtlety: OBJECT-SIDE phrases invert. "clearance to the van
ON THE RIGHT" means the ego offsets LEFT. Reading object-side as ego-side
silently flips ~2/3 of the corpus signal (the object-side class dominates).
"""
import pytest

from tanitad.data import cot_tokens_v7 as COT


def side(txt):
    return (COT.goals_from_cot(txt).get("CORRIDOR_OFFSET") or {}).get("side")


@pytest.mark.parametrize("txt,want", [
    # DIRECT: the text states the ego's own direction
    ("Keep right and maintain safe clearance to the oncoming truck.", "right"),
    ("position left before the roundabout exit", "left"),
    ("slight left offset within the lane to pass a parked van", "left"),
    ("shift slightly right within the lane", "right"),
    # OBJECT-SIDE: the obstacle's side — ego goes the OPPOSITE way
    ("increase clearance to the stopped van on the right", "left"),
    ("pedestrians standing on the left edge of the lane", "right"),
    ("parked cars on the right constrain the lane", "left"),
])
def test_side_resolution(txt, want):
    assert side(txt) == want


def test_conflicting_votes_yield_no_token():
    """A tie is ambiguity — the honest output is neither side."""
    assert side("keep left; also parked cars on the left require clearance") is None


def test_negation_still_guards():
    assert side("Do not shift left; hold the center of the lane.") is None


def test_no_claim_no_token():
    assert side("The road ahead is clear.") is None


def test_only_constraint_is_the_side():
    """PI: 'just give it left or right value as constraint' — nothing else."""
    g = COT.goals_from_cot("slight left offset to pass a parked van")
    assert set(g["CORRIDOR_OFFSET"]) == {"side"}


def test_both_classes_agreeing_reinforce():
    g = side("shift left within the lane; parked truck on the right")
    assert g == "left"
