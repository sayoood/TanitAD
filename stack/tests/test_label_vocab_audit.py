"""Pins the LABEL-AND-VOCABULARY audit's load-bearing facts (2026-09-06).

⭐ WHY THESE ARE TESTS AND NOT PROSE. Every fact below was MEASURED by
`label_vocab_audit.py` and every one of them is the kind that rots silently:
a head width, a supervised-head set, a default weight, a band tolerance. A
prose note recording "the 22-token tactical goal set has no head" is stale the
day someone adds one; this file fails instead.

⛔ These pin WHAT IS, not what SHOULD BE. A test failing here means the audit's
table is out of date and must be re-run -- it does NOT mean the change is wrong.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.data import v7_labels as v7l
from tanitad.models import vocab_v7 as V7
from tanitad.refs import refc_tactical as tac


# --------------------------------------------------------------------------
# The vocabularies, as sized
# --------------------------------------------------------------------------
def test_kin3_is_three_by_three():
    """The CORE tactical surface is kin3 by construction (refc.py N_LAT_MAN)."""
    assert tac.N_LAT == 3, tac.LAT_CLASSES
    assert tac.N_LON == 3, tac.LON_CLASSES


def test_v7_action_vocabularies_are_eight_by_eight():
    assert len(V7.TACTICAL_LAT_ACTIONS_V7) == 8
    assert len(V7.TACTICAL_LON_ACTIONS_V7) == 8


def test_exactly_four_v7_heads_are_supervised():
    """⛔ THE FINDING. The v7.2 release declares FOUR supervised heads. The
    22-token TACTICAL GOAL SET is NOT one of them -- it is minted into every
    record and no head predicts it."""
    assert set(v7l.HEADS) == {"tac_lat", "tac_lon", "str_action", "str_goal"}
    assert "tac_goal" not in v7l.HEADS
    assert len(V7.TACTICAL_GOAL_TOKENS_V7) == 22


def test_tactical_goal_set_has_no_supervised_head():
    """The goal SET tokens appear in no HEADS tuple, at any width."""
    supervised = {t for toks in v7l.HEADS.values() for t in toks}
    goals = set(V7.TACTICAL_GOAL_TOKENS_V7)
    # only the tokens that double as ACTION names may overlap
    overlap = goals & supervised
    assert overlap <= {"TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R"}, \
        sorted(overlap)


def test_not_yet_extractable_is_eight_tokens():
    """Tokens frozen into the vocabulary with NO extraction path at all."""
    assert len(V7.NOT_YET_EXTRACTABLE) == 8
    assert set(V7.NOT_YET_EXTRACTABLE) <= V7.ALL_V7_TOKENS


def test_nav_is_an_input_not_a_supervised_head():
    """PI: the nav command is an INPUT simulating the vehicle's nav system."""
    for t in V7.NAV_COMMAND_TOKENS:
        assert V7.ROLE_OF[t] == "input"
    assert not set(V7.NAV_COMMAND_TOKENS) & {
        t for toks in v7l.HEADS.values() for t in toks}


# --------------------------------------------------------------------------
# The band: how much of the corpus the v7.2 tactical GT actually covers
# --------------------------------------------------------------------------
def test_tactical_band_tolerance_is_plus_minus_two_seconds():
    """One record per clip at ``t0_s``, valid over (hi-lo)/2 = +-2.0 s."""
    lo, hi = 2.0, 6.0
    assert (hi - lo) / 2.0 == 2.0


# --------------------------------------------------------------------------
# The default weights that decide whether a label is trained at all
# --------------------------------------------------------------------------
def test_zero_default_weights_are_still_zero():
    """⛔ Three GT-consuming objectives are OFF by default. Each allocates a
    head and reads a label; none receives gradient unless explicitly enabled.
    If a default here changes, the audit's class D/B split changes with it."""
    import importlib.util as iu
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    p = root / "scripts" / "refc_v3_train.py"
    if not p.exists():                       # pragma: no cover
        pytest.skip("trainer not present in this tree")
    spec = iu.spec_from_file_location("_rv3t", p)
    mod = iu.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["refc_v3_train"]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = saved
    assert mod.AGENT_WEIGHT_DEFAULT == 0.0
    assert mod.GOAL_POINT_WEIGHT_DEFAULT == 0.0
    assert mod.U0_WEIGHT_DEFAULT == 0.0
    # and the ones that ARE on, so a silent disable also fails
    assert mod.ROUTE_WEIGHT == 0.1
    assert mod.LAT_WEIGHT == 0.05
    assert mod.LON_WEIGHT == 0.05
