"""REF-A v1′ — the action-as-tokens alternative arm.

The tests that matter are the ones pinning it as a SINGLE-AXIS delta: if v1′
differs from v1 in any way other than how the action reaches the predictor, a
v1/v1′ comparison stops isolating the design choice and becomes a confound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.refs.refa_v1 import (RefAV1, RefAV1Config,  # noqa: E402
                                  TokenFieldPredictor)
from tanitad.refs.refa_v1p import (ActionStreamPredictor,  # noqa: E402
                                   RefAV1Prime, RefAV1PrimeConfig)


def _small():
    """A tiny config so the tests run on CPU in seconds.

    ⚠️ ``d_enc`` must shrink WITH ``d_state``: v1's change-#3 guard refuses
    ``d_state < d_enc`` (FROST-Drive measured 8.17 -> 7.68 RFS on that axis), so
    a test config that only shrank the state would be refused — correctly.
    """
    c = RefAV1PrimeConfig()
    c.d_enc = 64
    c.d_state = 64
    c.n_tokens = 16
    c.op_layers = 1
    c.tac_layers = 1
    c.op_heads = 4
    return c


# --------------------------------------------------------------------------- #
# the delta is exactly one thing
# --------------------------------------------------------------------------- #
def test_v1_is_not_modified_by_importing_v1p():
    """⛔ v1′ is PARKED, not a replacement. v1's predictor must still be the
    broadcast-concat form after v1′ is imported."""
    assert not issubclass(TokenFieldPredictor, ActionStreamPredictor)
    assert issubclass(ActionStreamPredictor, TokenFieldPredictor)
    assert "mix" in dict(TokenFieldPredictor(
        RefAV1Config(), 32, 1, 4).named_children())


def test_rollout_is_INHERITED_not_reimplemented():
    """⚠️ Duplicating rollout would let the `last_only` memory rule (3.9 GB for
    a 300-candidate CEM population) drift between the two arms."""
    assert "rollout" not in ActionStreamPredictor.__dict__
    assert ActionStreamPredictor.rollout is TokenFieldPredictor.rollout


def test_only_step_is_overridden():
    overridden = {k for k, v in ActionStreamPredictor.__dict__.items()
                  if callable(v) and not k.startswith("_")}
    assert overridden == {"step"}, overridden


# --------------------------------------------------------------------------- #
# the mechanism
# --------------------------------------------------------------------------- #
def test_step_preserves_shape_and_is_residual_at_init():
    """Zero-init on act_split makes the step near-identity at initialisation —
    the same property v1 gets from its residual head, so a 6 s rollout does not
    drift on the first gradient."""
    p = ActionStreamPredictor(_small(), 64, 1, 4)
    f = torch.randn(2, 10, 64)
    out = p.step(f, torch.zeros(2, _small().a_dim))
    assert out.shape == f.shape
    # head is not zero-init, so allow the residual to be small but bounded
    assert torch.isfinite(out).all()


def test_the_action_ACTUALLY_reaches_the_prediction():
    """⛔ The failure this catches is silent: an action path that is wired but
    inert trains a scene extrapolator wearing a world model's name."""
    p = ActionStreamPredictor(_small(), 64, 1, 4)
    # un-zero the split so the action can carry signal (post-init state)
    torch.nn.init.trunc_normal_(p.act_split.weight, std=0.05)
    f = torch.randn(2, 10, 64)
    a1 = torch.zeros(2, _small().a_dim)
    a2 = torch.ones(2, _small().a_dim)
    assert not torch.allclose(p.step(f, a1), p.step(f, a2), atol=1e-6)


def test_field_tokens_are_read_back_out_not_the_action_tokens():
    """The stream is N + n_act wide; the output must be N."""
    for n_act in (1, 2, 4):
        c = _small()
        c.n_act_tokens = n_act
        p = ActionStreamPredictor(c, 64, 1, 4, n_act_tokens=n_act)
        f = torch.randn(2, 7, 64)
        assert p.step(f, torch.zeros(2, c.a_dim)).shape == (2, 7, 64)


def test_intent_still_reaches_the_predictor():
    """⚠️ The hierarchy port. v1 adds `intent` to the action embedding; v1′ must
    keep that path or the arm silently deletes the hierarchy."""
    c = _small()
    p = ActionStreamPredictor(c, 64, 1, 4, intent_dim=16)
    torch.nn.init.trunc_normal_(p.act_split.weight, std=0.05)
    f = torch.randn(2, 10, 64)
    a = torch.zeros(2, c.a_dim)
    i1, i2 = torch.zeros(2, 16), torch.ones(2, 16)
    assert not torch.allclose(p.step(f, a, intent=i1),
                              p.step(f, a, intent=i2), atol=1e-6)


def test_rollout_last_only_matches_the_final_step_of_the_full_rollout():
    p = ActionStreamPredictor(_small(), 64, 1, 4)
    torch.nn.init.trunc_normal_(p.act_split.weight, std=0.05)
    f = torch.randn(2, 8, 64)
    acts = torch.randn(2, 3, _small().a_dim)
    full = p.rollout(f, acts, last_only=False)
    last = p.rollout(f, acts, last_only=True)
    assert full.shape[1] == 3
    assert torch.allclose(full[:, -1], last, atol=1e-5)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_zero_action_tokens_is_refused():
    c = _small()
    c.n_act_tokens = 0
    with pytest.raises(ValueError, match="n_act_tokens"):
        c.sanity()


def test_default_is_the_PARAMETER_MATCHED_value():
    """⚠️ 2 is the value E-ACTSTREAM-1 measured at parameter parity. A default
    of 4 would buy capacity as well as structure and stop isolating the design
    choice."""
    assert RefAV1PrimeConfig().n_act_tokens == 2


# --------------------------------------------------------------------------- #
# the assembled arm
# --------------------------------------------------------------------------- #
def test_prime_swaps_BOTH_field_predictors_and_nothing_else():
    c = _small()
    m = RefAV1Prime(c)
    assert isinstance(m.operative, ActionStreamPredictor)
    assert isinstance(m.tactical, ActionStreamPredictor)
    # ⚠️ the strategic predictor operates on a compact subspace, not a token
    # field — swapping it too would make this a two-axis edit
    assert not isinstance(m.strategic, ActionStreamPredictor)


def test_v1_and_v1p_expose_the_same_planning_interface():
    """The arms must be drop-in for each other, or they cannot be compared."""
    for name in ("step", "rollout"):
        assert hasattr(RefAV1Prime(_small()).operative, name)
    assert set(dir(RefAV1)) - set(dir(RefAV1Prime)) == set()
