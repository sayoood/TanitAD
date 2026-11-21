"""⛔ THE FROZEN-EXTERNAL GUARD — pinned in BOTH directions.

WHY THIS FILE EXISTS. `apply_stage_freeze` sets ``requires_grad`` from the
MODULE_GROUPS map. An external-encoder arm (E-XENC-1) installs a foreign
backbone under the ``encoder`` group — which S-W TRAINS — so the freeze
**un-freezes the foreign backbone**. MEASURED at build time
(`…/Research/2026-08-18-encoder-experiments/raw/eenc1_param_delta.json`):
**86,580,480 foreign parameters** would have trained while the run called itself
*"frozen external encoder"*, and nothing in the ladder would have said so.

⛔ AND WHY IT IS PINNED IN BOTH DIRECTIONS. C95/C97: this programme shipped a
rejects-everything guard and a passes-everything guard **within one day**. A
guard that only asserts "the foreign backbone is frozen" is satisfied by
freezing the WHOLE model. So:

  * **direction A** — the guard RAISES when a declared subtree is trainable;
  * **direction B** — the guard RAISES when a declared-trainable group has no
    trainable NATIVE parameter left, i.e. it cannot be satisfied by freezing
    everything;
  * **it does NOT fire** on the incumbent stack (no declaration), on a stage
    that legitimately freezes the encoder, or after the documented repair.

NO GPU, NO CORPUS, NO CHECKPOINT — synthetic tiny stack, same wiring.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    FROZEN_EXTERNAL_FLAG, FrozenExternalViolation, MODULE_GROUPS, V6Config,
    V6Stack, apply_stage_freeze, assert_frozen_external,
    declare_frozen_external, frozen_external_prefixes,
    reassert_frozen_external, stage_trainable_groups)


def tiny_cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


@pytest.fixture()
def stack() -> V6Stack:
    torch.manual_seed(0)
    return V6Stack(tiny_cfg())


class _FakeBackbone(nn.Module):
    """Stands in for `facebook/dinov2-base` — the ONLY thing the guard cares
    about is that it is a submodule carrying parameters inside a group."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList(nn.Linear(d, d) for _ in range(2))


def _install_external(stack: V6Stack) -> tuple[nn.Module, int]:
    """Attach a declared-frozen foreign backbone under `encoder.*`, exactly as
    E-XENC-1's adapter would. Returns (module, n_params)."""
    bb = declare_frozen_external(
        _FakeBackbone(stack.cfg.encoder.d_model),
        "facebook/dinov2-base stand-in — E-XENC-1 frozen external backbone")
    stack.encoder.add_module("backbone", bb)
    return bb, sum(p.numel() for p in bb.parameters())


# ---------------------------------------------------------------------------
# the incumbent stack: NO declaration -> the guard must be silent
# ---------------------------------------------------------------------------
def test_guard_is_silent_on_the_incumbent_stack(stack):
    for st in ("S-W", "S-T", "S-S", "S-J"):
        apply_stage_freeze(stack, st)
        rep = assert_frozen_external(stack, st)
        assert rep["n_declared_subtrees"] == 0
        assert rep["n_external_params"] == 0
        assert rep["n_trainable"] > 0
        # every declared-trainable group really does hold trainable natives
        for g in stage_trainable_groups(st):
            assert rep["native_trainable_per_group"][g] > 0, (st, g)


def test_declaration_is_greppable_and_freezes_on_the_spot():
    bb = _FakeBackbone()
    assert not hasattr(bb, FROZEN_EXTERNAL_FLAG)
    declare_frozen_external(bb, "because")
    assert getattr(bb, FROZEN_EXTERNAL_FLAG) == "because"
    assert all(not p.requires_grad for p in bb.parameters())


# ---------------------------------------------------------------------------
# ⛔ DIRECTION A — it must CATCH the un-freeze
# ---------------------------------------------------------------------------
def test_S_W_unfreezes_the_foreign_backbone_and_the_guard_catches_it(stack):
    bb, n_ext = _install_external(stack)
    assert n_ext > 0
    assert frozen_external_prefixes(stack) == {
        "encoder.backbone": getattr(bb, FROZEN_EXTERNAL_FLAG)}

    audit = apply_stage_freeze(stack, "S-W")
    # the trap, reproduced: the group map made the FOREIGN weights trainable
    assert all(p.requires_grad for p in bb.parameters()), (
        "the premise of this guard no longer holds — apply_stage_freeze no "
        "longer un-freezes an encoder-group submodule; re-derive the guard")
    assert audit["per_group"]["encoder"]["trainable"] >= n_ext

    with pytest.raises(FrozenExternalViolation, match="TRAINABLE"):
        assert_frozen_external(stack, "S-W")


def test_the_documented_repair_makes_the_guard_pass_without_starving_S_W(stack):
    bb, n_ext = _install_external(stack)
    apply_stage_freeze(stack, "S-W")
    rep = reassert_frozen_external(stack)
    assert rep["n_params_refrozen"] == n_ext
    assert rep["declared_subtrees"] == {"encoder.backbone":
                                        getattr(bb, FROZEN_EXTERNAL_FLAG)}

    out = assert_frozen_external(stack, "S-W")
    assert out["n_external_params"] == n_ext
    assert out["external_per_group"] == {"encoder": n_ext}
    # ⭐ the native encoder still trains — the repair froze the FOREIGN subtree
    # only. A guard satisfied by freezing everything would pass without this.
    assert out["native_trainable_per_group"]["encoder"] > 0
    for g in stage_trainable_groups("S-W"):
        assert out["native_trainable_per_group"][g] > 0, g
    assert all(not p.requires_grad for p in bb.parameters())


def test_a_stage_that_does_not_train_the_encoder_needs_no_repair(stack):
    _install_external(stack)
    apply_stage_freeze(stack, "S-T")          # encoder is NOT in S-T's groups
    out = assert_frozen_external(stack, "S-T")
    assert out["native_trainable_per_group"]["encoder"] == 0
    assert out["n_external_params"] > 0


# ---------------------------------------------------------------------------
# ⛔ DIRECTION B — it must NOT pass a stack that trains nothing
# ---------------------------------------------------------------------------
def test_freezing_everything_does_NOT_satisfy_the_guard(stack):
    _install_external(stack)
    apply_stage_freeze(stack, "S-W")
    stack.requires_grad_(False)               # the passes-everything failure
    with pytest.raises(FrozenExternalViolation, match="ZERO trainable native"):
        assert_frozen_external(stack, "S-W")


def test_a_declaration_that_swallows_a_whole_group_is_refused(stack):
    """If someone declares the ENCODER ITSELF frozen-external, S-W trains no
    encoder at all — direction A is satisfied and the arm is still a lie."""
    declare_frozen_external(stack.encoder, "over-broad declaration")
    apply_stage_freeze(stack, "S-W")
    reassert_frozen_external(stack)
    with pytest.raises(FrozenExternalViolation) as e:
        assert_frozen_external(stack, "S-W")
    assert "encoder" in str(e.value)


def test_expected_trainable_count_is_a_parameter_not_a_retyped_number(stack):
    _install_external(stack)
    apply_stage_freeze(stack, "S-W")
    reassert_frozen_external(stack)
    n = assert_frozen_external(stack, "S-W")["n_trainable"]
    assert assert_frozen_external(stack, "S-W", expect_n_trainable=n)
    with pytest.raises(FrozenExternalViolation, match="not the arm it claims"):
        assert_frozen_external(stack, "S-W", expect_n_trainable=n + 1)


def test_every_group_is_still_partitioned_with_an_external_subtree(stack):
    """The guard must not disturb the group partition `apply_stage_freeze`
    depends on — a foreign submodule under `encoder.` maps to `encoder`."""
    _install_external(stack)
    audit = apply_stage_freeze(stack, "S-J")
    assert audit["n_trainable"] + audit["n_frozen"] == sum(
        p.numel() for p in stack.parameters())
    for n, _ in stack.named_parameters():
        assert stack.group_of(n) in MODULE_GROUPS
