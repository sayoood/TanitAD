"""Witness test for the NaN-class sweep (fleet review 2026-07-18).

kamm_circle_violation used `(a_lon^2 + a_lat^2).sqrt()` feeding a relu. At zero
total acceleration (a fully-feasible, coasting control) sqrt'(0)=inf and relu'=0
multiply to 0*inf=NaN on the backward pass — a silent run-death trap (F-5/6/7).
The fix clamps the sqrt argument; these tests pin that the gradient is finite at
and around the zero-accel point, and the forward value is unchanged."""
import torch

from tanitad.models.kinematic import kamm_circle_violation


def test_grad_finite_at_zero_accel():
    # steer=0, accel=0 -> total accel exactly 0 -> the sqrt-relu trap point.
    controls = torch.zeros(2, 4, 2, requires_grad=True)
    v = torch.zeros(2)
    loss = kamm_circle_violation(controls, v)
    loss.backward()
    assert torch.isfinite(loss), "forward NaN at zero accel"
    assert controls.grad is not None
    assert torch.isfinite(controls.grad).all(), "NaN/inf gradient at zero accel"


def test_grad_finite_small_accel_sweep():
    for eps in (0.0, 1e-8, 1e-4, 1e-2):
        c = torch.full((1, 4, 2), float(eps), requires_grad=True)
        loss = kamm_circle_violation(c, torch.tensor([eps]))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(c.grad).all(), eps


def test_forward_value_unchanged_in_feasible_range():
    # far from zero the clamp is a no-op: a coasting control is feasible (0).
    controls = torch.zeros(1, 4, 2)
    controls[..., 0] = 1.0                      # 1 m/s^2 longitudinal << mu*g
    out = kamm_circle_violation(controls, torch.tensor([5.0]))
    assert float(out) == 0.0                    # below the mu*g threshold


def test_violation_positive_when_infeasible():
    controls = torch.zeros(1, 4, 2)
    controls[..., 0] = 12.0                      # 12 m/s^2 > mu*g (7.85)
    out = kamm_circle_violation(controls, torch.tensor([0.0]))
    assert float(out) > 0.0
