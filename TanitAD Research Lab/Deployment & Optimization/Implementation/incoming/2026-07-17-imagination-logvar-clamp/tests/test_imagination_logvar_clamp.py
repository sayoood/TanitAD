"""Standalone tests for the bounded-logvar numerics guard (models review #3).

Run: pytest "TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-07-17-imagination-logvar-clamp/tests" -q

The fix (`bounded_logvar.py`) is self-contained (torch only), so the core tests
run without the tanitad package. Two witness tests import the CURRENT stack
`imagination.py` to document the live overflow; they SKIP if tanitad is not on
the path (run them with `PYTHONPATH=stack` to see the bug reproduced).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_logvar import (LOGVAR_MAX, LOGVAR_MIN, clamp_logvar,  # noqa: E402
                            imagination_nll_safe)

B, N, D = 2, 16, 8


def _case(logvar_value: float):
    """A hidden/visible imagination batch with a uniform logvar value."""
    torch.manual_seed(0)
    pred = torch.randn(B, N, D)
    target = torch.randn(B, N, D)
    vis = torch.ones(B, N)
    vis[:, : N // 2] = 0.0                     # half hidden (the imagination cells)
    logvar = torch.full((B, N), float(logvar_value))
    return pred, target, logvar, vis


# --- the safe function: finite everywhere, unchanged in the healthy range ----

@pytest.mark.parametrize("lv", [-100.0, -1e4, -88.8, 50.0, 1e4])
def test_safe_nll_finite_on_pathological_logvar(lv):
    pred, target, logvar, vis = _case(lv)
    loss = imagination_nll_safe(pred, target, logvar, vis)
    assert torch.isfinite(loss), f"safe NLL non-finite at logvar={lv}: {loss}"


@pytest.mark.parametrize("lv", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_safe_matches_reference_in_healthy_range(lv):
    """Inside [LOGVAR_MIN, LOGVAR_MAX] the clamp is a no-op, so the safe NLL is
    bit-for-bit the stack formula → a converged checkpoint is unchanged."""
    pred, target, logvar, vis = _case(lv)
    # reference = the exact stack formula, no clamp (healthy range => no overflow)
    err2 = (pred - target).pow(2).mean(dim=-1)
    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)
    w = (1.0 - vis) + 0.1 * vis
    ref = (w * nll).sum() / w.sum().clamp_min(1e-8)
    got = imagination_nll_safe(pred, target, logvar, vis)
    assert torch.allclose(got, ref, atol=1e-6), f"drift at logvar={lv}"


def test_clamp_bounds_and_noop_in_range():
    lv = torch.tensor([-1e9, -10.0, -2.0, 0.0, 5.0, 10.0, 1e9])
    out = clamp_logvar(lv)
    assert float(out.min()) >= LOGVAR_MIN and float(out.max()) <= LOGVAR_MAX
    # in-range values are untouched
    in_range = torch.tensor([-2.0, 0.0, 5.0])
    assert torch.equal(clamp_logvar(in_range), in_range)


def test_clamp_gradient_zeroed_at_bound_flows_in_range():
    lv = torch.tensor([-50.0, 0.0], requires_grad=True)   # one clamped, one free
    clamp_logvar(lv).sum().backward()
    assert lv.grad.tolist() == [0.0, 1.0]                 # zeroed at bound, 1 in-range


def test_safe_nll_is_differentiable():
    pred, target, logvar, vis = _case(-100.0)
    logvar = logvar.clone().requires_grad_(True)
    imagination_nll_safe(pred, target, logvar, vis).backward()
    assert torch.isfinite(logvar.grad).all()


def test_std_export_path_finite_after_clamp():
    """replay/arms.py:284 exports uncertainty as (0.5*logvar).exp(); an unbounded
    positive logvar overflows the std. Clamped logvar keeps it finite."""
    raw = torch.tensor([1e4, 200.0, 0.0])
    std_raw = (0.5 * raw).exp()
    std_clamped = (0.5 * clamp_logvar(raw)).exp()
    assert not torch.isfinite(std_raw).all()              # documents the hazard
    assert torch.isfinite(std_clamped).all()


def test_field_forward_would_be_bounded_by_head_clamp():
    """The proposed fix clamps at the logvar_head output in ImaginationField.
    Force the head over-confident and show clamp bounds the emitted logvar."""
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 1))
    nn.init.constant_(head[-1].bias, -500.0)              # pathological over-confidence
    x = torch.randn(B, N, D)
    raw_logvar = head(x).squeeze(-1).detach()
    assert float(raw_logvar.max()) < LOGVAR_MIN           # head really is out of range
    bounded = clamp_logvar(raw_logvar)
    assert float(bounded.min()) >= LOGVAR_MIN and float(bounded.max()) <= LOGVAR_MAX
    assert torch.isfinite(torch.exp(-bounded)).all()      # the nll exp is now safe


# --- witness: the CURRENT stack code overflows (skips without tanitad) --------

def _stack_imagination_nll():
    try:
        from tanitad.models.imagination import imagination_nll
        return imagination_nll
    except Exception:
        pytest.skip("tanitad not importable (run with PYTHONPATH=stack to witness)")


def test_stack_imagination_nll_overflows_witness():
    fn = _stack_imagination_nll()
    pred, target, logvar, vis = _case(-100.0)
    loss = fn(pred, target, logvar, vis)
    assert not torch.isfinite(loss), (
        "expected the UNPATCHED stack imagination_nll to overflow to inf at "
        "logvar=-100; if this now passes, the clamp has already been integrated")


def test_stack_matches_safe_in_healthy_range_witness():
    fn = _stack_imagination_nll()
    pred, target, logvar, vis = _case(-2.0)
    assert torch.allclose(fn(pred, target, logvar, vis),
                          imagination_nll_safe(pred, target, logvar, vis), atol=1e-6)
