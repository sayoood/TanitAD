"""⛔ The conflict detector's controls must WAIT for a readable step -- and refuse a permanent zero.

⭐⭐ MEASURED 2026-09-23 on refcv6's launch configuration (Thor, 416x1024, the real `train()`):
the run refused to start because `cos(g,g)` read NaN. The trajectory loss reached all 316 trunk
tensors and every gradient was EXACTLY zero -- `control_head` is zero-initialised by design (the
first refinement pass is the identity) -- and after ONE optimizer update all 316 were non-zero.
A control that is 0/0 on step 1 is UNDEFINED, not missed. The detector now defers on an exactly-
zero plan gradient and bounds the deferral, because a detach is also exactly zero -- forever.

Pinned on the detector's own two-head rig, with the three cases that must stay distinct:
(1) a zero-init trajectory head DEFERS on step 1 and reads +1 / -1 / 0 exactly after one update;
(2) a DETACHED trajectory head (a real "planner not trained through the trunk") is REFUSED once
    the bound is exceeded;
(3) a NaN plan gradient is NOT a deferral -- it still FAILS, on the first check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tanitad.train.grad_conflict import (  # noqa: E402
    ConflictConfig, ControlFailure, GradientConflictDetector)


class _Net(nn.Module):
    def __init__(self, *, zero_traj: bool = False, detach_traj: bool = False):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(8, 8, 3, stride=2, padding=1))
        self.traj_head = nn.Linear(8, 4)
        self.aux_head = nn.Linear(8, 3)
        if zero_traj:
            nn.init.zeros_(self.traj_head.weight)
            nn.init.zeros_(self.traj_head.bias)
        self.detach_traj = detach_traj

    def forward(self, x):
        f = self.encoder(x).mean(dim=(2, 3))
        return self.traj_head(f.detach() if self.detach_traj else f), self.aux_head(f)


def _batch():
    g = torch.Generator().manual_seed(1)
    return (torch.randn(4, 3, 16, 16, generator=g), torch.randn(4, 4, generator=g),
            torch.randn(4, 3, generator=g))


def _losses(net):
    x, yt, ya = _batch()
    pt, pa = net(x)
    return ((pt - yt) ** 2).mean(), ((pa - ya) ** 2).mean()


def _det(net, **kw):
    return GradientConflictDetector(net.named_parameters(),
                                    ConflictConfig(enabled=True, trunk_prefixes=("encoder.",),
                                                   **kw))


def test_a_ZERO_INIT_head_defers_on_step_1_then_the_controls_read_EXACTLY():
    net = _Net(zero_traj=True)
    det = _det(net)
    lt, la = _losses(net)
    r1 = det.self_check(lt, la)                     # must NOT raise
    assert r1.deferred is True and r1.ok is False
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    opt.zero_grad()
    (lt + la).backward()
    opt.step()                                      # the zero-init head leaves zero
    lt, la = _losses(net)
    r2 = det.self_check(lt, la)
    assert r2.deferred is False and r2.ok is True
    assert (r2.self_cos, r2.negated_cos, r2.detached_conflict) == (1.0, -1.0, 0.0)


def test_a_DETACHED_trajectory_head_is_REFUSED_after_the_bound():
    """⛔ The deferral must not become a way to run a planner the trunk never trains."""
    net = _Net(detach_traj=True)
    det = _det(net, max_deferred_steps=3)
    for _ in range(3):
        lt, la = _losses(net)
        assert det.self_check(lt, la).deferred is True
    lt, la = _losses(net)
    with pytest.raises(ControlFailure, match="EXACTLY ZERO"):
        det.self_check(lt, la)


def test_a_NAN_plan_gradient_is_NOT_deferred_it_FAILS():
    """⛔ `norm == 0.0` is the ONLY deferral; NaN != 0.0 and must still miss."""
    net = _Net()
    det = _det(net)
    lt, la = _losses(net)
    with pytest.raises(ControlFailure):
        det.self_check(lt * float("nan"), la)


def test_the_bound_is_STAMPED_and_validated():
    assert ConflictConfig(enabled=True).as_dict()["max_deferred_steps"] == 200
    with pytest.raises(ValueError):
        ConflictConfig(enabled=True, max_deferred_steps=0)


def test_the_trainer_takes_NO_reading_on_a_deferred_step():
    """The call site: a deferred check must not be followed by `measure` on that step."""
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    assert '_cd_row = {"cd_deferred": 1}' in src
    assert "if (_cd_checked and step % _cd_cfg.every == 0" in src
    assert "and _cd_la is not None and _cd_checked" in src
