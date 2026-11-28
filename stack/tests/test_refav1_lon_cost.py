#!/usr/bin/env python3
"""Pins for refav1's distance-keeping cost term (``D-REFAV1-DK-COST``).

The properties tested here are the ones the term's admissibility rests on, not a
coverage exercise:

* the shipped path is **exactly** zero (bit-identical A/B),
* the integrator matches ``rollout_unicycle`` *numerically*, so this cost's gap
  is the gap ``lead_metrics`` scores and not a second geometry,
* the term is one-sided and saturating (it can never reward accelerating into a
  large gap, nor reward braking past adequacy),
* the steady-state closure leaves a constant-velocity plan's gap EXACTLY at
  ``gap0`` -- the property that makes the term fire only on a real shortfall.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tanitad.models.kinematic import rollout_unicycle  # noqa: E402
from tanitad.refs.refav1_lon_cost import (  # noqa: E402
    DK_D0_M, DK_TAU_TARGET_S, LEAD_SPEED_STATIC, LEAD_SPEED_STEADY,
    DistanceKeepingSpec, desired_gap, distance_keeping_cost, gap_violation,
    predicted_along, predicted_gap, predicted_speeds,
)

DT = 0.2
H = 10


def _ctrl(accels, kappa=0.0):
    a = torch.tensor(accels, dtype=torch.float64)
    if a.ndim == 1:
        a = a[None]
    k = torch.full_like(a, float(kappa))
    return torch.stack([a, k], dim=-1)


# --------------------------------------------------------------------------- #
# A. the shipped path is EXACTLY zero                                          #
# --------------------------------------------------------------------------- #
def test_a_unarmed_is_exactly_zero():
    c = _ctrl([[-2.0] * H, [1.0] * H])
    out = distance_keeping_cost(c, v0=10.0, gap0_m=1.0, dt=DT)   # default w_dk = 0
    assert out.shape == (2,)
    assert torch.equal(out, torch.zeros(2, dtype=out.dtype))


@pytest.mark.parametrize("gap0", [None, float("nan"), float("inf"), float("-inf")])
def test_a_no_lead_is_exactly_zero(gap0):
    spec = DistanceKeepingSpec(w_dk=100.0, gap_source="unit-test")
    c = _ctrl([[0.0] * H])
    out = distance_keeping_cost(c, v0=10.0, gap0_m=gap0, dt=DT, spec=spec)
    assert torch.equal(out, torch.zeros(1, dtype=out.dtype))


def test_a_empty_horizon_is_exactly_zero():
    spec = DistanceKeepingSpec(w_dk=100.0, gap_source="unit-test")
    c = torch.zeros(3, 0, 2, dtype=torch.float64)
    out = distance_keeping_cost(c, v0=10.0, gap0_m=1.0, dt=DT, spec=spec)
    assert torch.equal(out, torch.zeros(3, dtype=out.dtype))


# --------------------------------------------------------------------------- #
# B. the integrator IS rollout_unicycle's                                      #
# --------------------------------------------------------------------------- #
def test_b_speeds_and_along_match_rollout_unicycle():
    """Straight-line candidates: the module's ``s[k]`` must equal the integrator's
    own ``x[k]`` to float64 round-off, and ``v[k]`` its START-of-step speed."""
    accels = [[0.0] * H, [1.5] * H, [-3.0] * H, [2.0, -2.0] * (H // 2)]
    c = _ctrl(accels, kappa=0.0)
    v0 = 8.0
    state0 = torch.zeros(len(accels), 4, dtype=torch.float64)
    state0[:, 3] = v0
    states = rollout_unicycle(state0, c, dt=DT)

    v = predicted_speeds(c[..., 0], v0, DT)
    s = predicted_along(v, DT)
    # x after step k
    assert torch.allclose(s, states[..., 0], atol=1e-12)
    # v after step k is the module's step-start speed SHIFTED by one
    assert torch.allclose(v[:, 1:], states[:, :-1, 3], atol=1e-12)
    assert torch.allclose(v[:, 0], torch.full((len(accels),), v0, dtype=torch.float64))


def test_b_clamp_stops_a_braking_candidate():
    """A candidate that brakes to zero STAYS at zero; a closed-form cumsum would
    credit it with driving backwards."""
    c = _ctrl([[-10.0] * H])
    v = predicted_speeds(c[..., 0], 5.0, DT)
    assert float(v.min()) >= 0.0
    s = predicted_along(v, DT)
    assert torch.all(s[:, 1:] >= s[:, :-1] - 1e-12)      # monotone non-decreasing


def test_b_last_accel_never_moves_the_horizon():
    """``a[H-1]`` updates v only AFTER the final displacement -- so it cannot
    change any gap inside the horizon."""
    a1 = [0.5] * H
    a2 = list(a1)
    a2[-1] = -9.0
    c = _ctrl([a1, a2])
    g, _ = predicted_gap(c[..., 0], v0=9.0, gap0_m=20.0, dt=DT)
    assert torch.allclose(g[0], g[1], atol=1e-12)


# --------------------------------------------------------------------------- #
# C. the steady-state closure                                                  #
# --------------------------------------------------------------------------- #
def test_c_constant_velocity_holds_the_gap_exactly():
    """The property the whole term rests on: under LEAD_SPEED_STEADY the all-zero
    control's predicted gap is EXACTLY gap0 at every step."""
    c = _ctrl([[0.0] * H])
    g, _ = predicted_gap(c[..., 0], v0=13.6, gap0_m=18.0, dt=DT,
                         lead_speed_closure=LEAD_SPEED_STEADY)
    assert torch.allclose(g, torch.full_like(g, 18.0), atol=1e-12)


def test_c_static_closure_closes_on_a_stationary_lead():
    c = _ctrl([[0.0] * H])
    g, _ = predicted_gap(c[..., 0], v0=10.0, gap0_m=30.0, dt=DT,
                         lead_speed_closure=LEAD_SPEED_STATIC)
    # gap falls by v0 * t
    want = 30.0 - 10.0 * torch.arange(1, H + 1, dtype=torch.float64) * DT
    assert torch.allclose(g, want, atol=1e-12)


def test_c_accelerating_closes_the_gap_decelerating_opens_it():
    accel, hold, brake = _ctrl([[1.0] * H]), _ctrl([[0.0] * H]), _ctrl([[-1.0] * H])
    kw = dict(v0=12.0, gap0_m=20.0, dt=DT)
    ga, _ = predicted_gap(accel[..., 0], **kw)
    gh, _ = predicted_gap(hold[..., 0], **kw)
    gb, _ = predicted_gap(brake[..., 0], **kw)
    assert float(ga[0, -1]) < float(gh[0, -1]) < float(gb[0, -1])


def test_c_explicit_lead_speed_overrides_the_closure():
    c = _ctrl([[0.0] * H])
    g, _ = predicted_gap(c[..., 0], v0=10.0, gap0_m=25.0, dt=DT,
                         lead_speed_mps=12.0)
    want = 25.0 + 2.0 * torch.arange(1, H + 1, dtype=torch.float64) * DT
    assert torch.allclose(g, want, atol=1e-12)


# --------------------------------------------------------------------------- #
# D. one-sided, saturating, regime-aware                                       #
# --------------------------------------------------------------------------- #
def test_d_adequate_gap_costs_exactly_zero():
    spec = DistanceKeepingSpec(w_dk=1.0, gap_source="unit-test")
    v0 = 10.0
    safe = DK_D0_M + DK_TAU_TARGET_S * v0 + 50.0
    out = distance_keeping_cost(_ctrl([[0.0] * H]), v0=v0, gap0_m=safe, dt=DT,
                                spec=spec)
    assert torch.equal(out, torch.zeros(1, dtype=out.dtype))


def test_d_short_gap_makes_holding_speed_cost_more_than_braking():
    """THE DEFECT THIS TERM EXISTS FOR. With no longitudinal cost the all-zero
    control is the joint minimiser; with a real shortfall it must not be."""
    spec = DistanceKeepingSpec(w_dk=1.0, gap_source="unit-test")
    v0 = 12.0
    short = 6.0                                   # s*(12) = 5 + 1.5*12 = 23 m
    c = _ctrl([[0.0] * H, [-2.0] * H, [1.0] * H])
    out = distance_keeping_cost(c, v0=v0, gap0_m=short, dt=DT, spec=spec)
    hold, brake, accel = (float(x) for x in out)
    assert brake < hold < accel
    assert hold > 0.0


def test_d_braking_saturates_and_never_goes_negative():
    spec = DistanceKeepingSpec(w_dk=1.0, gap_source="unit-test")
    hard = distance_keeping_cost(_ctrl([[-6.0] * H]), v0=12.0, gap0_m=6.0,
                                 dt=DT, spec=spec)
    harder = distance_keeping_cost(_ctrl([[-12.0] * H]), v0=12.0, gap0_m=6.0,
                                   dt=DT, spec=spec)
    assert float(harder) <= float(hard)
    assert float(harder) >= 0.0


def test_d_desired_gap_is_affine_and_covers_both_regimes():
    v = torch.tensor([0.0, 0.5, 13.6], dtype=torch.float64)
    s = desired_gap(v)
    assert float(s[0]) == pytest.approx(DK_D0_M)            # crawl: d0 alone
    assert float(s[2]) == pytest.approx(DK_D0_M + DK_TAU_TARGET_S * 13.6)
    # a 1 m gap at 0.5 m/s is a violation even though gap/v = 2 s > tau
    viol = gap_violation(_ctrl([[0.0] * H])[..., 0], v0=0.5, gap0_m=1.0, dt=DT)
    assert float(viol.min()) > 0.0


def test_d_cost_scales_linearly_in_w():
    kw = dict(v0=12.0, gap0_m=6.0, dt=DT)
    c = _ctrl([[0.0] * H])
    a = distance_keeping_cost(c, spec=DistanceKeepingSpec(w_dk=1.0, gap_source="unit-test"), **kw)
    b = distance_keeping_cost(c, spec=DistanceKeepingSpec(w_dk=3.0, gap_source="unit-test"), **kw)
    assert float(b) == pytest.approx(3.0 * float(a), rel=1e-12)


def test_d_curvature_channel_is_ignored():
    spec = DistanceKeepingSpec(w_dk=1.0, gap_source="unit-test")
    kw = dict(v0=12.0, gap0_m=6.0, dt=DT, spec=spec)
    a = distance_keeping_cost(_ctrl([[0.0] * H], kappa=0.0), **kw)
    b = distance_keeping_cost(_ctrl([[0.0] * H], kappa=0.08), **kw)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# E. the declaration travels                                                   #
# --------------------------------------------------------------------------- #
def test_e_spec_records_the_closure_and_convention():
    r = DistanceKeepingSpec(w_dk=1.0, gap_source="oracle_label").record()
    assert r["lead_speed_closure"] == LEAD_SPEED_STEADY
    assert r["gap_source"] == "oracle_label"
    assert "rear face" in r["gap_convention"]
    assert "UNOBSERVABLE" in r["closure_note"]


def test_e_bad_closure_and_weights_refuse():
    with pytest.raises(ValueError):
        DistanceKeepingSpec(lead_speed_closure="guess")
    with pytest.raises(ValueError):
        DistanceKeepingSpec(w_dk=-1.0)
    with pytest.raises(ValueError):
        DistanceKeepingSpec(tau_target_s=-0.1)


def test_e_bad_control_shape_refuses():
    with pytest.raises(ValueError):
        distance_keeping_cost(torch.zeros(4, 5), v0=1.0, gap0_m=1.0, dt=DT,
                              spec=DistanceKeepingSpec(w_dk=1.0, gap_source="unit-test"))
    with pytest.raises(ValueError):
        predicted_speeds(torch.zeros(4, 5, 2), 1.0, DT)


def test_e_gradients_flow():
    """iCEM does not need them, but a differentiable term keeps a gradient-based
    refinement reachable without a second implementation."""
    a = torch.zeros(1, H, dtype=torch.float64, requires_grad=True)
    c = torch.stack([a, torch.zeros_like(a)], dim=-1)
    out = distance_keeping_cost(c, v0=12.0, gap0_m=6.0, dt=DT,
                                spec=DistanceKeepingSpec(w_dk=1.0, gap_source="unit-test"))
    out.sum().backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0.0
    assert math.isfinite(float(a.grad.abs().sum()))
