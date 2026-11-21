"""Pin the substring collision that read every straight clip as a left turn."""
import pytest

from tanitad.data import alpamayo_fusion as AF


def test_follow_lane_is_straight_not_left():
    """⛔ THE BUG: `"_L" in "FOLLOW_LANE"` is True.

    It reported 24.9 % lateral agreement where the truth is 69.9 %.
    """
    assert "_L" in "FOLLOW_LANE"                 # the collision is real
    assert AF.side_of("FOLLOW_LANE") == "straight"


@pytest.mark.parametrize("cls,want", [
    ("TURN_L", "left"), ("TURN_R", "right"),
    ("JUNCTION_TURN_L", "left"), ("JUNCTION_TURN_R", "right"),
    ("YIELD_FOR_TURN_L", "left"), ("YIELD_FOR_TURN_R", "right"),
    ("FOLLOW_LANE", "straight"), ("SPEED_BAND", "straight"),
    ("LANE_CHANGE_L", "left"), ("NUDGE_R", "right"),
    ("STOP_POINT", "straight"), ("CORRIDOR_OFFSET", "straight"),
])
def test_side_of(cls, want):
    assert AF.side_of(cls) == want


def test_fuse_lateral_refuses_a_raw_token():
    """A caller passing a token gets an error, not a silent wrong answer."""
    with pytest.raises(ValueError, match="geom_side must be one of"):
        AF.fuse_lateral("whatever", "FOLLOW_LANE")
