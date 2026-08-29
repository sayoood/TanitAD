"""Analytic pins for the critic-free advantage estimators.

The estimator is where a silent no-op hides best: a centred advantage over a
group of one is identically zero, the loss is finite, the optimiser steps, and
nothing changes. Several tests here exist purely to prove the library REFUSES
that instead of reporting it as a successful step.
"""

from __future__ import annotations

import pytest
import torch

from tanitad.rl import advantage as A


# ---------------------------------------------------------------------------
# Intra-group (GRPO)
# ---------------------------------------------------------------------------

def test_grpo_advantage_is_mean_zero_and_analytic():
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    adv = A.grpo_advantage(r)
    assert torch.allclose(adv, torch.tensor([[-1.5, -0.5, 0.5, 1.5]]))
    assert adv.mean().abs() < 1e-6


def test_grpo_default_does_NOT_divide_by_std():
    """Dr. GRPO: dividing by the group std is a bias, not a normalisation."""
    r = torch.tensor([[0.0, 10.0]])
    centred = A.grpo_advantage(r, normalize="none")
    scaled = A.grpo_advantage(r, normalize="std")
    assert torch.allclose(centred, torch.tensor([[-5.0, 5.0]]))
    assert not torch.allclose(centred, scaled)
    # std of [0,10] (unbiased) is 7.071 -> +-0.7071
    assert torch.allclose(scaled, torch.tensor([[-0.7071, 0.7071]]), atol=1e-3)


def test_grpo_refuses_a_group_of_one():
    """⛔ The silent-no-op refusal."""
    with pytest.raises(ValueError, match="group of >=2"):
        A.grpo_advantage(torch.tensor([[1.0]]))


def test_grpo_rejects_a_bad_normalize_name():
    with pytest.raises(ValueError, match="normalize"):
        A.grpo_advantage(torch.zeros(1, 4), normalize="zscore")


def test_identical_rewards_give_zero_advantage():
    """No spread -> no signal. Correct, and worth pinning as intended."""
    adv = A.grpo_advantage(torch.full((2, 5), 3.0))
    assert torch.allclose(adv, torch.zeros_like(adv))


# ---------------------------------------------------------------------------
# Inter-anchor truncation
# ---------------------------------------------------------------------------

def test_negatives_are_truncated_to_zero():
    r = torch.tensor([[0.0, 1.0, 2.0, 3.0]])          # mean 1.5
    adv = A.truncated_inter_anchor_advantage(r)
    assert torch.allclose(adv, torch.tensor([[0.0, 0.0, 0.5, 1.5]]))
    assert (adv >= 0).all()


def test_collisions_are_pinned_to_minus_one_even_when_above_mean():
    """A colliding candidate must be punished however good it otherwise looks."""
    r = torch.tensor([[0.0, 1.0, 2.0, 9.0]])
    collided = torch.tensor([[False, False, False, True]])
    adv = A.truncated_inter_anchor_advantage(r, collided=collided)
    assert adv[0, 3].item() == pytest.approx(-1.0)
    assert adv[0, 0].item() == pytest.approx(0.0)


def test_collided_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="must match"):
        A.truncated_inter_anchor_advantage(torch.zeros(1, 4),
                                           collided=torch.zeros(1, 3, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_composite_returns_both_halves_separately():
    """An aggregate that hides which half moved makes an ablation unattributable."""
    r = torch.rand(2, 3, 4)
    out = A.composite_advantage(r)
    assert set(out) == {"intra", "inter", "total", "per_anchor_reward"}
    assert out["intra"].shape == (2, 3, 4)
    assert out["inter"].shape == (2, 3)
    assert out["total"].shape == (2, 3, 4)
    assert torch.allclose(out["intra"].mean(dim=-1),
                          torch.zeros(2, 3), atol=1e-6)


def test_composite_weights_are_honoured():
    r = torch.rand(1, 2, 4)
    only_intra = A.composite_advantage(r, w_intra=1.0, w_inter=0.0)
    parts = A.composite_advantage(r)
    assert torch.allclose(only_intra["total"], parts["intra"])


def test_composite_refuses_a_flat_tensor():
    with pytest.raises(ValueError, match=r"\[\.\.\., N, G\]"):
        A.composite_advantage(torch.rand(4))


# ---------------------------------------------------------------------------
# The surrogate loss
# ---------------------------------------------------------------------------

def test_policy_gradient_detaches_the_advantage():
    """An advantage carrying gradient is not a policy gradient — silently."""
    logp = torch.zeros(1, 4, requires_grad=True)
    adv = torch.ones(1, 4, requires_grad=True)
    out = A.policy_gradient_loss(logp, adv)
    out["loss"].backward()
    assert adv.grad is None, "advantage must be detached inside the surrogate"
    assert logp.grad is not None


def test_pg_loss_sign_pushes_up_positive_advantage():
    """Higher logp on a positive-advantage sample must LOWER the loss."""
    adv = torch.tensor([[1.0, -1.0]])
    lo = A.policy_gradient_loss(torch.tensor([[0.0, 0.0]]), adv)["loss"]
    hi = A.policy_gradient_loss(torch.tensor([[1.0, 0.0]]), adv)["loss"]
    assert hi < lo


def test_kl_is_off_by_default_and_nonnegative_when_on():
    logp = torch.zeros(1, 4, requires_grad=True)
    adv = torch.ones(1, 4)
    assert "kl" not in A.policy_gradient_loss(logp, adv)

    ref = torch.full((1, 4), 0.3)
    out = A.policy_gradient_loss(logp, adv, logp_ref=ref, kl_coef=0.1)
    assert out["kl"].item() >= 0.0            # k3 estimator is non-negative


def test_kl_without_reference_is_refused():
    with pytest.raises(ValueError, match="logp_ref"):
        A.policy_gradient_loss(torch.zeros(1, 2), torch.ones(1, 2), kl_coef=0.1)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="must match"):
        A.policy_gradient_loss(torch.zeros(1, 4), torch.ones(1, 3))
