"""The v5 guard's scoring math, held by tests BEFORE it adjudicates a relaunch.

The guard decides goal_dropout for v5-flagship — a wrong classifier or a κ that rewards chance
would adjudicate a training decision on an instrument bug (the exact class GATE_PROTOCOL §0.7
exists for). So the math is pinned on synthetic cases with known answers.
"""
import math

import torch

from scripts.v5_guard import (R_LEFT, R_RIGHT, R_STRAIGHT, classify_route,
                              cohens_kappa, mean_speed)


def _traj(heading_deg: float, n: int = 20, step: float = 1.0) -> torch.Tensor:
    th = math.radians(heading_deg)
    xs = torch.arange(1, n + 1, dtype=torch.float32) * step
    return torch.stack([xs * math.cos(th), xs * math.sin(th)], dim=-1)


def test_classify_route_known_headings():
    t = torch.stack([_traj(0.0), _traj(20.0), _traj(-20.0), _traj(4.0)])
    assert classify_route(t).tolist() == [R_STRAIGHT, R_LEFT, R_RIGHT, R_STRAIGHT]


def test_kappa_perfect_and_chance():
    a = torch.tensor([0, 1, 2, 0, 1, 2, 1, 1])
    assert cohens_kappa(a, a.clone()) == 1.0
    # systematic disagreement must land at/below zero, never look like skill
    b = (a + 1) % 3
    assert cohens_kappa(a, b) <= 0.0
    assert math.isnan(cohens_kappa(torch.tensor([]), torch.tensor([])))


def test_mean_speed_constant_velocity():
    t = _traj(0.0, n=20, step=1.0)[None]          # 1 m per 0.1 s = 10 m/s
    assert abs(float(mean_speed(t)) - 10.0) < 1e-4


def test_collapse_detector_logic():
    """follow_true ≈ follow_shuffled ⇒ no route signal — the delta, not the level."""
    g = torch.Generator().manual_seed(0)
    cmd = torch.randint(0, 3, (400,), generator=g)
    # a COLLAPSED head: output class independent of the command
    out = torch.randint(0, 3, (400,), generator=g)
    perm = torch.randperm(400, generator=g)
    follow_true = float((out == cmd).float().mean())
    follow_shuf = float((out == cmd[perm]).float().mean())
    assert abs(follow_true - follow_shuf) < 0.08
    # a LIVE head: follows the command 90 % of the time
    live = torch.where(torch.rand(400, generator=g) < 0.9, cmd,
                       torch.randint(0, 3, (400,), generator=g))
    assert float((live == cmd).float().mean()) - \
        float((live == cmd[perm]).float().mean()) > 0.3
