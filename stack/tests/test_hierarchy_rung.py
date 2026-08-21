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


# --------------------------------------------------------------------------- #
# The ALIGNED tier (PI 2026-08-21): REF-A v1 / REF-D tactical CAPACITY carrying
# v6's OWN predictor, heads and vocabulary — in every arm.
# --------------------------------------------------------------------------- #
from tanitad.models.hierarchy import (ALIGNED_STRATEGIC, ALIGNED_TACTICAL,  # noqa: E402
                                      V6_STRATEGIC, V6_TACTICAL,
                                      aligned_body_matches_refa)


@pytest.fixture(scope="module")
def aligned(built):
    stack, _, _, _ = built
    tac = HierarchyRung(ALIGNED_TACTICAL, vocab_goal=stack.vocab_tac,
                        vocab_actions=(stack.vocab_a_lat, stack.vocab_a_lon),
                        vocab_above=stack.vocab_str)
    stg = HierarchyRung(ALIGNED_STRATEGIC, vocab_goal=stack.vocab_str,
                        vocab_actions=(stack.vocab_a_str,), vocab_above=None)
    return tac, stg


def test_aligned_body_EXACTLY_matches_refa_v1_tactical_blocks(aligned):
    """⭐ THE ALIGNMENT CLAIM. REF-A v1's TokenFieldPredictor.blocks MEASURE
    50,384,896 at d_state=1024 / tac_layers=4. The aligned tier must reproduce
    that number, not approximate it — 'same size' has to mean the same size."""
    tac, _ = aligned
    assert aligned_body_matches_refa(tac) == 50_384_896


def test_aligned_rungs_still_carry_v6s_OWN_predictor(aligned):
    """Capacity comes from the body; the PREDICTOR stays v6's FTac in both."""
    tac, stg = aligned
    from tanitad.models.tactical import FTac
    assert isinstance(tac.predictor, FTac) and isinstance(stg.predictor, FTac)
    assert sum(p.numel() for p in stg.predictor.parameters()) == 3_481_856


def test_the_canonical_v6_geometries_are_UNCHANGED_by_the_body(built):
    """⛔ Adding the body must not redefine v6's rung — body_layers=0 is v6."""
    stack, _, _, _ = built
    tac = HierarchyRung(V6_TACTICAL, vocab_goal=stack.vocab_tac,
                        vocab_actions=(stack.vocab_a_lat, stack.vocab_a_lon),
                        vocab_above=stack.vocab_str)
    stg = HierarchyRung(V6_STRATEGIC, vocab_goal=stack.vocab_str,
                        vocab_actions=(stack.vocab_a_str,), vocab_above=None)
    assert_matches_v6(stack, tactical=tac, strategic=stg)
    assert len(tac.body) == 0 and len(stg.body) == 0


def test_aligned_rungs_share_v6s_vocabulary_objects(aligned, built):
    """Same vocab tables as v6 — `id()` identity across ALL arms is the point."""
    stack, _, _, _ = built
    tac, stg = aligned
    assert tac.goal_head.vocab is stack.vocab_tac
    assert tac.cond.vocab is stack.vocab_str
    assert stg.goal_head.vocab is stack.vocab_str
    assert tac.act_heads[0].vocab is stack.vocab_a_lat


def test_aligned_cascade_runs_end_to_end(aligned):
    tac, stg = aligned
    z_tac = tac.uplink(torch.randn(2, 2048))
    assert tuple(z_tac.shape) == (2, 1024)
    z_str = stg.uplink(z_tac)
    assert tuple(z_str.shape) == (2, 256)
    assert tuple(stg.imagine(z_str, torch.randn(2, 128)).shape) == (2, 256)
    assert tuple(tac.imagine(z_tac, torch.randn(2, 256)).shape) == (2, 1024)
