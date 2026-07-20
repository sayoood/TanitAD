"""Sanity tests for the D1 ADE power audit (G-B2: analytic/synthetic ground truth).

Run: pytest "TanitAD Research Hub/Benchmarks & Eval/Implementation/d1_power_audit/tests"
The tests exercise the pure statistics (bootstrap CI, ego rotation, per-episode
aggregation) with KNOWN ground truth — no checkpoint / CUDA required.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1_ade_power_audit import _ego, bootstrap_val, per_episode_ade  # noqa: E402


class _ConstProbe:
    """A fake probe whose per-window residual equals a fixed offset (predict=Y+off)."""
    def __init__(self, offset):
        self.offset = offset

    def predict(self, S):
        return S * 0.0 + self.offset          # residual vs Y handled by caller


def _synthetic(ep_means, w=25):
    """Build per-episode residual-norm tensors with an exact per-episode mean.

    d_by_ep[e] is a length-w tensor of residual norms all equal to ep_means[e], so
    the pooled window-mean of n equal-size episodes == the mean of the drawn ep means.
    """
    return {e: torch.full((w,), float(m)) for e, m in enumerate(ep_means)}


def test_bootstrap_ci_halfwidth_shrinks_with_n():
    """More val episodes -> tighter CI (the whole point of the audit)."""
    torch.manual_seed(0)
    ep_means = (2.0 + 6.0 * torch.rand(40)).tolist()   # heterogeneous routes
    d_by_ep = _synthetic(ep_means)
    gen = torch.Generator().manual_seed(1)
    hw4 = bootstrap_val(d_by_ep, 4, 2000, gen)["ci95_halfwidth"]
    hw9 = bootstrap_val(d_by_ep, 9, 2000, gen)["ci95_halfwidth"]
    hw20 = bootstrap_val(d_by_ep, 20, 2000, gen)["ci95_halfwidth"]
    assert hw4 > hw9 > hw20 > 0.0


def test_bootstrap_sd_matches_analytic_standard_error():
    """Equal-window episodes -> bootstrap SD of pooled mean ~= pop_sd / sqrt(n)."""
    ep_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    d_by_ep = _synthetic(ep_means, w=10)
    pop = torch.tensor(ep_means)
    # population SD with replacement sampling (bootstrap draws WITH replacement)
    pop_sd = float(pop.std(unbiased=False))
    for n in (4, 9):
        gen = torch.Generator().manual_seed(7)
        sd = bootstrap_val(d_by_ep, n, 8000, gen)["sd"]
        analytic = pop_sd / math.sqrt(n)
        assert abs(sd - analytic) / analytic < 0.12, (n, sd, analytic)


def test_ego_rotation_identity_and_90deg():
    dxy = torch.tensor([[3.0, 4.0]])
    # yaw=0 -> unchanged
    assert torch.allclose(_ego(dxy, torch.tensor([0.0])), dxy, atol=1e-5)
    # yaw=+90deg -> a forward displacement rotates into the ego frame deterministically
    r = _ego(dxy, torch.tensor([math.pi / 2]))
    assert torch.allclose(r, torch.tensor([[4.0, -3.0]]), atol=1e-4)
    # rotation preserves length
    assert abs(float(r.norm()) - 5.0) < 1e-4


def test_per_episode_ade_counts_and_values():
    # 3 episodes, 4 windows each; residual norm == |offset| per window (S all zeros)
    N = 12
    S = torch.zeros(N, 2)
    Y = torch.zeros(N, 2)
    E = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    probe = _ConstProbe(torch.tensor([0.0, 3.0]))     # residual norm = 3.0 everywhere
    ep_ade, d = per_episode_ade(probe, S, Y, E, [0, 1, 2])
    for e in (0, 1, 2):
        mean_e, cnt = ep_ade[e]
        assert cnt == 4
        assert abs(mean_e - 3.0) < 1e-5
    assert d.shape == (N,)
