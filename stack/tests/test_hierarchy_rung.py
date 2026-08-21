"""The shared hierarchy rung must BE v6's rung, not merely resemble it.

⛔ WHY THESE TESTS. The PI's instruction is "the same and best hierarchy
architecture in all our designs". A shared component that silently drifts from
v6 would give every arm a DIFFERENT hierarchy while claiming one — which is the
"second vocabulary" failure the programme has already paid for once.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.models import v6 as V
from tanitad.models.hierarchy import (HierarchyRung, HierarchyRungConfig,
                                      assert_matches_v6, rung_param_count)


def _rungs(stack, c):
    tac = HierarchyRung(
        HierarchyRungConfig(d_in=c.d_op, d_layer=c.d_tac,
                            adapter_hidden=c.adapter_hidden,
                            d_goal_embed=c.d_goal_embed,
                            f_hidden=c.f_hidden_tac, f_blocks=c.f_blocks,
                            d_goal_pred=2 * c.d_goal_embed),
        vocab_goal=stack.vocab_tac,
        vocab_actions=(stack.vocab_a_lat, stack.vocab_a_lon),
        vocab_above=stack.vocab_str)
    stg = HierarchyRung(
        HierarchyRungConfig(d_in=c.d_tac, d_layer=c.d_str,
                            adapter_hidden=c.adapter_hidden,
                            d_goal_embed=c.d_goal_embed,
                            f_hidden=c.f_hidden_str, f_blocks=c.f_blocks,
                            d_goal_pred=c.d_goal_embed),
        vocab_goal=stack.vocab_str, vocab_actions=(stack.vocab_a_str,),
        vocab_above=None)
    return tac, stg


@pytest.fixture(scope="module")
def built():
    c = V.V6Config()
    stack = V.V6Stack(c)
    return stack, c, *_rungs(stack, c)


def test_the_rung_reproduces_v6_component_by_component(built):
    """⭐ THE LOAD-BEARING TEST. Not a total — every shared component."""
    stack, _, tac, stg = built
    rep = assert_matches_v6(stack, tactical=tac, strategic=stg)
    assert rep["tactical"]["total"] == 5_767_981
    assert rep["strategic"]["total"] == 4_152_993


def test_assert_matches_v6_CAN_FAIL(built):
    """⛔ A guard that cannot fail is not a guard (the C13 class)."""
    stack, c, _, stg = built
    wrong = HierarchyRung(
        HierarchyRungConfig(d_in=c.d_op, d_layer=c.d_tac,
                            adapter_hidden=c.adapter_hidden,
                            d_goal_embed=c.d_goal_embed,
                            f_hidden=c.f_hidden_tac,
                            f_blocks=c.f_blocks + 1,        # <- one block too many
                            d_goal_pred=2 * c.d_goal_embed),
        vocab_goal=stack.vocab_tac,
        vocab_actions=(stack.vocab_a_lat, stack.vocab_a_lon),
        vocab_above=stack.vocab_str)
    with pytest.raises(AssertionError, match="predictor"):
        assert_matches_v6(stack, tactical=wrong, strategic=stg)


def test_every_rung_has_its_OWN_predictor(built):
    """The axis on which v6 is alone today, and the reason this exists."""
    _, _, tac, stg = built
    for r in (tac, stg):
        assert sum(p.numel() for p in r.predictor.parameters()) > 3_000_000


def test_the_top_rung_has_no_conditioner(built):
    """Nothing hands the strategic layer a goal; a zeroed conditioner would
    still carry params and would break state_dict parity with v6."""
    _, _, tac, stg = built
    assert stg.cond is None and tac.cond is not None
    assert "cond" not in dict(stg.named_children())


def test_vocabulary_is_SHARED_not_copied(built):
    """§5: one vocabulary, two views. `id()` identity, not equality."""
    stack, _, tac, _ = built
    assert tac.cond.vocab is stack.vocab_str
    assert tac.goal_head.vocab is stack.vocab_tac


def test_a_rung_without_an_action_head_is_refused(built):
    stack, c, _, _ = built
    with pytest.raises(ValueError, match="cannot express a decision"):
        HierarchyRung(HierarchyRungConfig(), vocab_goal=stack.vocab_tac,
                      vocab_actions=(), vocab_above=stack.vocab_str)


def test_the_three_operations_run(built):
    _, c, tac, stg = built
    z = tac.uplink(torch.randn(2, c.d_op))
    assert tuple(z.shape) == (2, c.d_tac)
    assert tuple(stg.uplink(z).shape) == (2, c.d_str)
    assert stg.condition(None) is None
    out = tac.imagine(z, torch.randn(2, 2 * c.d_goal_embed))
    assert tuple(out.shape) == (2, c.d_tac)


def test_v6_itself_is_UNTOUCHED_by_this_module():
    """⛔ v6 is training under tensor-strict resume. This module is ADDITIVE."""
    import tanitad.models.hierarchy  # noqa: F401
    stack = V.V6Stack(V.V6Config())
    assert sum(p.numel() for p in stack.parameters()) == 87_893_449
    assert len(stack.state_dict()) == 405
