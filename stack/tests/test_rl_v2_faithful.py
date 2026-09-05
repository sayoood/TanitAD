"""The two V2-faithful ingredients added for the REF-C RL re-scope (2026-09-05).

Both come from DiffusionDriveV2's RELEASED code (`hustvl/DiffusionDriveV2@1cd12a1`,
read in full by `…/Architecture & Inference/Research/2026-09-05-diffusiondrive-v2-analysis/`),
not from its paper:

1. the >=GT POSITIVE MASK — positive inter-anchor advantage only for candidates whose
   reward is at least the human trajectory's own reward on that window
   (`diffusiondrivev2_model_rl.py:891-893`);
2. the TWO-SCALAR exploration policy — one longitudinal and one lateral scalar per
   trajectory, broadcast over the waypoints (`:646-654`; the additive term is ×0).

Each test pins a property that a silent regression would break while the loop keeps
exiting 0 — the false-green class these files exist to refuse.
"""

from __future__ import annotations

import pytest
import torch

from tanitad.rl import PostTrainConfig, RewardSpec
from tanitad.rl import advantage as A
from tanitad.rl import rewards as R
from tanitad.rl.posttrain import apply_exploration_noise, rl_objective
from tanitad.rl.refcv3_adapter import sample_offsets


# --------------------------------------------------------------------------- #
# 1. the >=GT bar                                                               #
# --------------------------------------------------------------------------- #
def test_bar_masks_positive_advantage_below_the_human_score():
    r = torch.tensor([[0.2, 0.6, 0.9, 1.0]])            # [B=1, N=4], mean 0.675
    plain = A.truncated_inter_anchor_advantage(r)
    assert torch.allclose(plain, torch.tensor([[0.0, 0.0, 0.225, 0.325]]))
    barred = A.truncated_inter_anchor_advantage(r, gt_bar=torch.tensor([0.95]))
    # 0.9 is above the mean but BELOW the human's 0.95 -> no positive credit
    assert torch.allclose(barred, torch.tensor([[0.0, 0.0, 0.0, 0.325]]))


def test_bar_is_a_threshold_not_a_target():
    """Two candidates equally far above the bar get identical credit whatever their
    geometry — the bar reads the SCORE, never the shape (the anti-echo property)."""
    r = torch.tensor([[0.1, 0.8, 0.8]])
    adv = A.truncated_inter_anchor_advantage(r, gt_bar=torch.tensor([0.5]))
    assert adv[0, 1] == adv[0, 2] and adv[0, 1] > 0


def test_bar_accepts_the_1e6_slack_of_the_released_code():
    r = torch.tensor([[0.0, 1.0]])
    adv = A.truncated_inter_anchor_advantage(r, gt_bar=torch.tensor([1.0]))
    assert adv[0, 1] > 0, "a candidate EQUAL to the human must clear the bar (>= with 1e-6 slack)"


def test_veto_pins_after_the_bar():
    r = torch.tensor([[0.0, 1.0]])
    adv = A.truncated_inter_anchor_advantage(r, gt_bar=torch.tensor([0.5]),
                                             veto=torch.tensor([[False, True]]))
    assert float(adv[0, 1]) == -1.0


def test_bar_shape_is_refused_when_not_per_window():
    r = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="per-window"):
        A.truncated_inter_anchor_advantage(r, gt_bar=torch.zeros(2, 3))


def test_composite_reports_frac_above_bar_and_leaves_intra_untouched():
    torch.manual_seed(0)
    r = torch.rand(2, 5, 4)
    bar = torch.tensor([0.0, 10.0])          # window 0: everyone clears; window 1: nobody
    parts = A.composite_advantage(r, gt_bar=bar)
    plain = A.composite_advantage(r)
    assert torch.equal(parts["intra"], plain["intra"])
    assert abs(float(parts["frac_above_bar"]) - 0.5) < 1e-6
    assert torch.all(parts["inter"][1] == 0.0), "a bar nobody clears zeroes the inter half"


def test_objective_refuses_a_configured_bar_without_a_value():
    cfg = PostTrainConfig(use_gt_bar=True, group_size=2, steps=1)
    spec = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    traj = torch.zeros(1, 2, 2, 5, 2)
    traj[..., :, 0] = torch.linspace(0, 8, 5)
    logp = torch.zeros(1, 2, 2, requires_grad=True)
    with pytest.raises(ValueError, match="use_gt_bar=True but no gt_bar"):
        rl_objective(traj, logp, {"dt": 0.5, "v0": 4.0}, cfg, spec)


def test_objective_refuses_a_bar_the_config_did_not_declare():
    cfg = PostTrainConfig(use_gt_bar=False, group_size=2, steps=1)
    spec = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    traj = torch.zeros(1, 2, 2, 5, 2)
    traj[..., :, 0] = torch.linspace(0, 8, 5)
    logp = torch.zeros(1, 2, 2, requires_grad=True)
    with pytest.raises(ValueError, match="use_gt_bar is False"):
        rl_objective(traj, logp, {"dt": 0.5, "v0": 4.0}, cfg, spec,
                     gt_bar=torch.tensor([0.5]))


def test_objective_records_veto_rate_and_frac_above_bar():
    cfg = PostTrainConfig(use_gt_bar=True, group_size=2, steps=1)
    spec = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    traj = torch.zeros(1, 3, 2, 5, 2)
    traj[..., :, 0] = torch.linspace(0, 8, 5)
    traj[:, 2] *= 0.5                                    # a slower anchor
    logp = torch.zeros(1, 3, 2, requires_grad=True)
    out = rl_objective(traj, logp, {"dt": 0.5, "v0": 4.0}, cfg, spec,
                       gt_bar=torch.tensor([0.9]))
    assert "veto_rate" in out and float(out["veto_rate"]) == 0.0
    assert 0.0 < float(out["frac_above_bar"]) < 1.0


# --------------------------------------------------------------------------- #
# 2. the two-scalar exploration policy                                          #
# --------------------------------------------------------------------------- #
def test_two_scalar_ratio_is_constant_along_the_trajectory():
    """V2 draws ONE scalar per axis per trajectory: (sample - mean) / (sigma*|mean|)
    must be the same number at every waypoint of a given axis."""
    torch.manual_seed(1)
    cfg = PostTrainConfig(noise_mode="two_scalar", noise_scale=0.1, group_size=4)
    off = torch.randn(2, 3, 8, 2) * 5.0 + 3.0            # |mean| >> min_scale
    sample, logp = sample_offsets(off, cfg)
    mean = off.unsqueeze(2).expand_as(sample)
    ratio = (sample - mean) / (mean.abs() * 0.1)         # [B, N, G, S, 2]
    spread_along_s = ratio.std(dim=-2)                   # over the S axis
    assert float(spread_along_s.max()) < 1e-4, "the per-axis scalar must not vary along S"
    # and the two axes are DIFFERENT draws
    assert float((ratio[..., 0] - ratio[..., 1]).abs().mean()) > 0.1
    assert logp.shape == (2, 3, 4)


def test_multiplicative_ratio_varies_along_the_trajectory():
    """The control: per-coordinate noise is NOT constant along S."""
    torch.manual_seed(1)
    cfg = PostTrainConfig(noise_mode="multiplicative", noise_scale=0.1, group_size=4)
    off = torch.randn(2, 3, 8, 2) * 5.0 + 3.0
    sample, _ = sample_offsets(off, cfg)
    mean = off.unsqueeze(2).expand_as(sample)
    ratio = (sample - mean) / (mean.abs() * 0.1)
    assert float(ratio.std(dim=-2).mean()) > 0.5


def test_two_scalar_gradient_points_toward_good_samples():
    """The directional pin (the P-RC21 collapse class), on the new sampler."""
    torch.manual_seed(0)
    m = torch.nn.Parameter(torch.full((1, 1, 4, 2), 0.5))
    opt = torch.optim.SGD([m], lr=0.05)
    cfg = PostTrainConfig(group_size=8, noise_mode="two_scalar", noise_scale=0.3)
    for _ in range(400):
        sample, logp = sample_offsets(m, cfg)
        reward = -(sample.detach() - 3.0).pow(2).sum(dim=(-1, -2))
        adv = reward - reward.mean(dim=-1, keepdim=True)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(m.mean()) > 2.0, f"mean stuck at {float(m.mean()):+.3f}"


def test_apply_exploration_noise_two_scalar_broadcasts_over_waypoints():
    torch.manual_seed(2)
    cfg = PostTrainConfig(noise_mode="two_scalar", noise_scale=0.1)
    off = torch.ones(2, 3, 8, 2) * 4.0
    noisy = apply_exploration_noise(off, cfg)
    ratio = noisy / off
    assert float(ratio.std(dim=-2).max()) < 1e-6
    assert float((ratio[..., 0] - ratio[..., 1]).abs().mean()) > 0.01


def test_config_accepts_two_scalar_and_still_refuses_junk():
    PostTrainConfig(noise_mode="two_scalar").validate()
    with pytest.raises(ValueError, match="noise_mode"):
        PostTrainConfig(noise_mode="three_scalar").validate()


def test_gt_bar_is_the_human_score_under_the_same_spec():
    """The bar is computed by scoring the human path with the SAME RewardSpec the
    candidates see — sanity: on a lead-free window the human's progress reward is
    exactly along/(v0*horizon)."""
    spec = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    human = torch.zeros(1, 1, 1, 5, 2)
    human[..., :, 0] = torch.linspace(0, 6, 5)           # 6 m in 2 s
    ctx = {"dt": 0.5, "v0": torch.tensor([[[4.0]]])}     # v0*2 s = 8 m reference
    bar = spec(human, ctx).reshape(1)
    assert abs(float(bar) - 0.75) < 1e-6
    assert R.COMPONENTS["progress"].lo <= float(bar) <= R.COMPONENTS["progress"].hi
