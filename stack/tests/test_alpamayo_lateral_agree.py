"""`alpamayo.lateral.agree` must compare LATERAL CLASSES, including NUDGE.

⛔ THE DEFECT THIS PINS (MEASURED 2026-08-30 on the released blob
`s2_labels_v7.jsonl.gz` md5 ee44875916ae7c0ac002c6716b9658ea, 4,416 records
carrying both fields): the emitter derived the geometry side from a scan of the
GOAL tokens for a `TURN_`/`YIELD_FOR_TURN_` prefix, defaulting to "straight".
NUDGE_L/NUDGE_R are ACTIONS, never turn goals, so every non-turn clip was
compared as "straight" and the flag INVERTED on 992 of 4,416 records (22.5 %) —
in BOTH directions:

    side=right   + NUDGE_R  -> agree False, but they genuinely AGREE     (353)
    side=left    + NUDGE_L  -> agree False, but they genuinely AGREE     (269)
    side=straight+ NUDGE_R  -> agree True,  but they genuinely DIFFER    (221)
    side=straight+ NUDGE_L  -> agree True,  but they genuinely DIFFER    (149)

622 real agreements were discarded as noise; 370 real contradictions were
hidden. The shipped 41.7 % disagreement rate was an artefact of the flag; the
direction-consistent rate is 36.0 %.

⚠️ `side_of` was NOT the cause. It handles the NUDGE suffix correctly and was
simply never called with a NUDGE class — "fixing" it would have been a no-op.
The bug was the CALLER passing a goal-derived side instead of the lateral class,
so this file pins BOTH halves: `side_of` stays directional for NUDGE, AND the
comparison is done on the class.
"""
import pytest

from tanitad.data import alpamayo_fusion as AF

#: the four cells the shipped corpus got backwards, with their record counts
INVERTED_CELLS = [
    ("right", "NUDGE_R", True, 353),
    ("left", "NUDGE_L", True, 269),
    ("straight", "NUDGE_R", False, 221),
    ("straight", "NUDGE_L", False, 149),
]
#: cells that were already right — they must STAY right
CORRECT_CELLS = [
    ("straight", "LANE_KEEP", True), ("right", "TURN_R", True),
    ("left", "TURN_L", True), ("right", "LANE_KEEP", False),
    ("left", "LANE_KEEP", False), ("right", "TURN_L", False),
    ("left", "TURN_R", False), ("straight", "TURN_L", False),
]


@pytest.mark.parametrize("alp_side,lat_cls,expect,_n", INVERTED_CELLS)
def test_the_four_inverted_cells(alp_side, lat_cls, expect, _n):
    """Each of these was the OPPOSITE of `expect` in the shipped corpus."""
    assert (AF.side_of(lat_cls) == alp_side) is expect


@pytest.mark.parametrize("alp_side,lat_cls,expect", CORRECT_CELLS)
def test_the_cells_that_were_already_right_stay_right(alp_side, lat_cls, expect):
    assert (AF.side_of(lat_cls) == alp_side) is expect


def test_nudge_is_directional_exactly_like_turn():
    """The one-line statement of the defect: NUDGE carries a side, like TURN."""
    assert AF.side_of("NUDGE_L") == AF.side_of("TURN_L") == "left"
    assert AF.side_of("NUDGE_R") == AF.side_of("TURN_R") == "right"
    assert AF.side_of("LANE_KEEP") == "straight"


def test_the_old_goal_scan_reproduces_the_inversion():
    """⭐ THE GUARD IS SHOWN TO FAIL. A test that has never distinguished the
    broken implementation from the fixed one proves nothing — this corpus was
    wrong end to end without anyone noticing, so the old logic is reconstructed
    here and asserted to DISAGREE with the truth on exactly the four cells."""
    def old_side(goals):                      # the shipped implementation
        t = next((k for k in goals if k.startswith(("TURN_", "YIELD_FOR_TURN_"))),
                 None)
        return AF.side_of(t) if t else "straight"

    wrong = 0
    for alp_side, lat_cls, truth, _n in INVERTED_CELLS:
        goals = [lat_cls] if lat_cls.startswith("TURN_") else []
        if (old_side(goals) == alp_side) is not truth:
            wrong += 1
    assert wrong == len(INVERTED_CELLS), (
        "the old goal-scan logic no longer reproduces the inversion — this "
        "test can no longer tell the broken version from the fixed one")


def test_fuse_lateral_still_refuses_a_raw_token():
    """The older trap stays closed: `"_L" in "FOLLOW_LANE"` is True."""
    with pytest.raises(ValueError):
        AF.fuse_lateral("whatever", "FOLLOW_LANE")
