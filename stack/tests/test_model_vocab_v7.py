"""The MANDATORY v7 vocabulary wired into the model (PI, 2026-08-27).

⛔ THE TWO LOAD-BEARING TESTS:
  * ``test_new_builds_carry_the_flywheel_shapes`` — a default-config stack's six
    vocab tensors are sized by the FROZEN FlyWheel tuples (8/7/22/8/8 + the
    derived LON-goal split), i.e. the mandate actually reaches the tensors.
  * ``test_v6_shapes_are_unchanged_under_their_recorded_version`` — an old run's
    recorded ``tac_vocab_version`` still produces the old shapes exactly, so
    every existing checkpoint stays loadable. The v7 strategic/goal tuples are
    RESTRUCTURES; without this property the mandate would strand the fleet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))

from tanitad.models import vocab_v7 as V7  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS, TACTICAL_GOAL_TOKENS,
    TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS, V6Config,
    strategic_action_tokens, strategic_goal_tokens, tactical_goal_tokens,
    tactical_lat_actions, tactical_lon_actions_v, tactical_lon_goals,
)


# --------------------------------------------------------------------------- #
# resolution                                                                    #
# --------------------------------------------------------------------------- #
def test_v7_resolves_to_the_frozen_flywheel_tuples():
    assert strategic_goal_tokens("v7.0") == V7.STRATEGIC_GOAL_TOKENS_V7
    assert strategic_action_tokens("v7.0") == V7.STRATEGIC_ACTION_TOKENS_V7
    assert tactical_goal_tokens("v7.0") == V7.TACTICAL_GOAL_TOKENS_V7
    assert tactical_lat_actions("v7.0") == V7.TACTICAL_LAT_ACTIONS_V7
    assert tactical_lon_actions_v("v7.0") == V7.TACTICAL_LON_ACTIONS_V7


def test_v7_sizes_are_the_frozen_ones():
    """8 / 7 / 22 / 8 / 8 — the RESULT.md freeze record, pinned as numbers so a
    silent tuple edit fails here even if both sides drift together."""
    assert len(strategic_goal_tokens("v7.0")) == 8
    assert len(strategic_action_tokens("v7.0")) == 7
    assert len(tactical_goal_tokens("v7.0")) == 22
    assert len(tactical_lat_actions("v7.0")) == 8
    assert len(tactical_lon_actions_v("v7.0")) == 8


def test_the_lon_goal_split_is_derived_from_the_flywheel_map():
    """The v7 LON-goal split = GOAL_ADMISSIBLE_LON's keys in goal-tuple order —
    the FlyWheel's own admissibility map, never a hand-made list here."""
    lg = tactical_lon_goals("v7.0")
    assert len(lg) > 0
    assert set(lg) <= set(V7.TACTICAL_GOAL_TOKENS_V7)
    assert set(lg) == set(V7.TACTICAL_GOAL_TOKENS_V7) & set(V7.GOAL_ADMISSIBLE_LON)
    assert list(lg) == [t for t in V7.TACTICAL_GOAL_TOKENS_V7 if t in set(lg)]


def test_v6_shapes_are_unchanged_under_their_recorded_version():
    assert strategic_goal_tokens("v6.0") == STRATEGIC_GOAL_TOKENS
    assert strategic_action_tokens("v6.0") == STRATEGIC_ACTION_TOKENS
    assert tactical_goal_tokens("v6.0") == TACTICAL_GOAL_TOKENS
    assert tactical_lat_actions("v6.0") == TACTICAL_LAT_ACTIONS
    assert tactical_lon_actions_v("v6.0") == TACTICAL_LON_ACTIONS


def test_unknown_versions_refuse_loudly():
    for fn in (strategic_goal_tokens, strategic_action_tokens,
               tactical_goal_tokens, tactical_lon_actions_v):
        with pytest.raises(ValueError, match="unknown"):
            fn("v9.9")


def test_every_v7_token_passes_the_freeze_guard():
    for t in (strategic_goal_tokens("v7.0") + strategic_action_tokens("v7.0")
              + tactical_goal_tokens("v7.0") + tactical_lat_actions("v7.0")
              + tactical_lon_actions_v("v7.0")):
        assert V7.assert_frozen(t, where="test_model_vocab_v7") == t


# --------------------------------------------------------------------------- #
# the tensors                                                                   #
# --------------------------------------------------------------------------- #
def _tiny(vv):
    cfg = V6Config()
    for f, v in (("tac_vocab_version", vv),):
        setattr(cfg, f, v)
    return cfg


def test_new_builds_carry_the_flywheel_shapes():
    from tanitad.models.v6 import V6Stack
    cfg = V6Config()
    assert cfg.tac_vocab_version == "v7.0", "the mandate: new builds default v7"
    st = V6Stack(cfg)
    d = cfg.d_goal_embed
    assert st.vocab_str.table.weight.shape == (8, d)
    assert st.vocab_tac.table.weight.shape == (22, d)
    assert st.vocab_a_str.table.weight.shape == (7, d)
    assert st.vocab_a_lat.table.weight.shape == (8, d)
    assert st.vocab_a_lon.table.weight.shape == (8, d)


def test_v6_builds_reproduce_the_old_tensor_shapes():
    from tanitad.models.v6 import V6Stack
    cfg = _tiny("v6.0")
    st = V6Stack(cfg)
    d = cfg.d_goal_embed
    assert st.vocab_str.table.weight.shape == (len(STRATEGIC_GOAL_TOKENS), d)
    assert st.vocab_tac.table.weight.shape == (len(TACTICAL_GOAL_TOKENS), d)
    assert st.vocab_a_str.table.weight.shape == (len(STRATEGIC_ACTION_TOKENS), d)
    assert st.vocab_a_lat.table.weight.shape == (len(TACTICAL_LAT_ACTIONS), d)
    assert st.vocab_a_lon.table.weight.shape == (len(TACTICAL_LON_ACTIONS), d)


def test_recorded_args_without_the_field_mean_v6():
    """THE LOADER PROPERTY (found the hard way: the k4_30k drift read failed on
    v7-shaped heads): build_stack_from_args on a namespace LACKING the field —
    i.e. any pre-mandate run's recorded args — must produce v6 shapes; with
    v7.0 it must produce the FlyWheel shapes."""
    sys.path.insert(0, str(_STACK / "scripts"))
    from train_v6_staged import build_parser, build_stack_from_args
    base = ["--out", "UNUSED", "--stage", "S-W", "--frame-h", "64",
            "--frame-w", "160", "--enc-dim", "32", "--enc-depth", "1",
            "--enc-heads", "2", "--readout-grid", "2", "--readout-grid-w", "4",
            "--readout-dim", "16", "--pred-dim", "32", "--pred-depth", "1",
            "--pred-heads", "2", "--window", "4", "--d-tac", "16",
            "--d-str", "8"]
    a_new = build_parser().parse_args(base)
    assert a_new.tac_vocab_version == "v7.0"
    st_new = build_stack_from_args(a_new)
    assert st_new.vocab_a_str.table.weight.shape[0] == 7      # v7
    a_old = build_parser().parse_args(base)
    del a_old.tac_vocab_version                                # pre-field args
    st_old = build_stack_from_args(a_old)
    assert st_old.vocab_a_str.table.weight.shape[0] == 6      # v6.0
    assert st_old.vocab_str.table.weight.shape[0] == 11       # v6.0
