"""v6.1 tactical lateral vocabulary — TURN_L / TURN_R, prepared for post-30k.

⛔ THE PROPERTY THE LIVE RUN DEPENDS ON. `TACTICAL_LAT_ACTIONS` sizes real
tensors: the v6F S-W 30k checkpoint on Thor holds `vocab_a_lat.table.weight
(6, 128)` and `act_head_lat.type_head.weight (6, 256)` under a TENSOR-STRICT
resume contract. Editing the tuple in place would not disturb the running
process — it would brick its AUTO-RESUME hours later, silently, as a shape
mismatch on a 4.6-day run. Every test below exists to keep that from happening.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.models.v6 import (TACTICAL_LAT_ACTIONS, TACTICAL_LAT_ACTIONS_V61,
                               TACTICAL_LAT_MIN_N_FOR_METRIC,
                               TACTICAL_LAT_UNDERPOWERED,
                               TACTICAL_LON_ACTIONS, TACTICAL_VOCAB_VERSIONS,
                               V6Config, tactical_lat_actions)


# --------------------------------------------------- inert at default (the point) --
def test_the_DEFAULT_is_still_v6_0_so_the_live_run_is_untouched():
    assert V6Config().tac_vocab_version == "v6.0"
    assert tactical_lat_actions() is TACTICAL_LAT_ACTIONS
    assert len(tactical_lat_actions()) == 6


def test_the_v6_0_tuple_is_UNCHANGED_member_for_member():
    """The live checkpoint's six rows mean these six things, in this order."""
    assert TACTICAL_LAT_ACTIONS == ("LANE_KEEP", "LANE_CHANGE_L",
                                    "LANE_CHANGE_R", "ABORT_LC",
                                    "NUDGE_L", "NUDGE_R")


def test_the_longitudinal_axis_was_NOT_touched():
    assert TACTICAL_LON_ACTIONS == ("FOLLOW", "CRUISE", "YIELD_MERGE",
                                    "BRAKE_TO", "CREEP", "HOLD")
    assert len(TACTICAL_LON_ACTIONS) == 6


# ------------------------------------------------------------------ append-only --
def test_v6_1_APPENDS_and_never_reorders():
    """Indices 0-5 must survive verbatim, or every banked label, dump and
    artifact silently changes meaning and a 6-wide head cannot be padded."""
    assert TACTICAL_LAT_ACTIONS_V61[:6] == TACTICAL_LAT_ACTIONS
    assert TACTICAL_LAT_ACTIONS_V61[6:] == ("TURN_L", "TURN_R")
    assert len(TACTICAL_LAT_ACTIONS_V61) == 8


def test_index_of_every_v6_0_token_is_identical_in_v6_1():
    for i, tok in enumerate(TACTICAL_LAT_ACTIONS):
        assert TACTICAL_LAT_ACTIONS_V61.index(tok) == i, tok


def test_no_duplicate_tokens_in_either_version():
    for v, toks in TACTICAL_VOCAB_VERSIONS.items():
        assert len(set(toks)) == len(toks), v


def test_an_unknown_version_is_REFUSED_not_silently_defaulted():
    with pytest.raises(ValueError, match="unknown tactical vocabulary"):
        tactical_lat_actions("v7")


# ----------------------------------------------- representable, NOT scoreable --
def test_the_new_tokens_are_declared_underpowered():
    """n = 85 (TURN_L) / 101 (TURN_R) in the Alpamayo corpus; ~2 per class on a
    40-episode val split. They may be emitted and supervised; a per-class metric
    on them must be refused, exactly as cost_fidelity refuses below n = 200."""
    assert TACTICAL_LAT_UNDERPOWERED == {"TURN_L", "TURN_R"}
    assert TACTICAL_LAT_MIN_N_FOR_METRIC == 200
    for tok in TACTICAL_LAT_UNDERPOWERED:
        assert tok not in TACTICAL_LAT_ACTIONS, \
            "an underpowered token must not be in the v6.0 (scoreable) set"


def test_the_underpowered_set_names_only_tokens_that_exist():
    assert TACTICAL_LAT_UNDERPOWERED <= set(TACTICAL_LAT_ACTIONS_V61)


# ------------------------------------------------- the shapes, actually built --
def _vocab(version: str):
    from tanitad.models.v6 import GoalVocabulary
    return GoalVocabulary(tactical_lat_actions(version), 128)


def test_the_default_build_has_the_LIVE_CHECKPOINTS_shape():
    v = _vocab("v6.0")
    assert tuple(v.table.weight.shape) == (6, 128)


def test_v6_1_builds_the_widened_shape():
    v = _vocab("v6.1")
    assert tuple(v.table.weight.shape) == (8, 128)


def test_a_v6_0_checkpoint_can_be_PADDED_into_v6_1_without_retraining():
    """⭐ The migration, demonstrated rather than asserted: rows 0-5 copy
    verbatim, rows 6-7 are new. A 6-wide head becomes 8-wide by padding, which
    is why appending (not inserting) was the whole design constraint."""
    old, new = _vocab("v6.0"), _vocab("v6.1")
    with torch.no_grad():
        new.table.weight[:6].copy_(old.table.weight)
    assert torch.equal(new.table.weight[:6], old.table.weight)
    assert new.table.weight.shape[0] - old.table.weight.shape[0] == 2


def test_padding_a_head_with_a_very_negative_bias_keeps_it_behaviourally_inert():
    """The new classes must start effectively unreachable, so the first step
    after the widening behaves like the step before it."""
    logits6 = torch.randn(4, 6)
    logits8 = torch.cat([logits6, torch.full((4, 2), -20.0)], dim=-1)
    p6, p8 = logits6.softmax(-1), logits8.softmax(-1)
    assert torch.allclose(p6, p8[:, :6], atol=1e-6)
    assert float(p8[:, 6:].max()) < 1e-6
