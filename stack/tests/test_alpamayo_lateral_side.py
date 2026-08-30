"""Alpamayo's stated nudge DIRECTION must survive tokenization.

⛔ THE DEFECT (PI 2026-08-30: *"leverage more the cot and reasoning of
Alpamayo"*). `_LAT_RULES` maps BOTH ``nudge to the left`` and ``nudge to the
right`` onto the single unsided token ``EVADE_IN_CORRIDOR``, so a direction
Alpamayo stated explicitly was thrown away on 130 segments (91 left / 39 right).
Every other sided type kept its side — `turn right`, `split to the right`,
`change lane to the left` — nudge was the one that lost it.

⚠️ The v7 vocabulary is FROZEN, so the fix must NOT mint EVADE_IN_CORRIDOR_L/R.
These tests pin BOTH halves: the token stays exactly as it was, AND the side is
recoverable beside it.
"""
import pytest

from tanitad.data import alpamayo_structured as AST


def seg(raw_type: str, motion: str = "") -> AST.MotionSegment:
    return AST.MotionSegment(index=1, raw_type=raw_type, t0_alpamayo=0.0,
                             t1_alpamayo=4.0, motion=motion or raw_type)


@pytest.mark.parametrize("raw,side", [
    ("nudge to the left", "left"),
    ("nudge to the right", "right"),
    ("split to the right", "right"),
    ("split to the left", "left"),
    ("turn right", "right"),
    ("turn left", "left"),
    ("change lane to the left", "left"),
    ("change lane to the right", "right"),
])
def test_a_stated_direction_survives(raw, side):
    assert seg(raw).lateral_side == side


@pytest.mark.parametrize("raw", ["nudge to the left", "nudge to the right"])
def test_the_frozen_token_is_unchanged(raw):
    """⛔ The vocabulary must NOT gain a sided EVADE token."""
    lat, _ = seg(raw).tokens()
    assert lat == "EVADE_IN_CORRIDOR"


def test_keep_lane_is_straight_not_none():
    """A lane-keep is a POSITIVE claim of 'no lateral', not an absence of one."""
    assert seg("keep lane").lateral_side == "straight"


def test_an_unmapped_type_makes_no_claim():
    """Absence of a match is not a claim of absence — it must read None."""
    assert seg("stop and wait").lateral_side is None


def test_the_old_behaviour_would_fail_this():
    """⭐ THE GUARD IS SHOWN TO FAIL against the pre-fix logic: mapping through
    the token alone loses the nudge side, which is exactly the defect."""
    def old_side(raw):                       # what the tokenizer alone can say
        lat = next((v for k, v in AST._LAT_RULES if k in raw), None)
        if not lat:
            return None
        return "left" if lat.endswith("_L") else (
            "right" if lat.endswith("_R") else "straight")

    assert old_side("nudge to the left") == "straight"       # the bug
    assert seg("nudge to the left").lateral_side == "left"   # the fix
    assert old_side("turn left") == seg("turn left").lateral_side  # unaffected
