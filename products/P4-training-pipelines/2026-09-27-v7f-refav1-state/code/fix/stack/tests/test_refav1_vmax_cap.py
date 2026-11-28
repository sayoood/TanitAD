"""R1 (PI directive 2026-09-27): refav1's planner must take the max speed as an input and NEVER exceed it.

`PlanConfig.v_max` (None = OFF, bit-identical to every earlier arm) caps each step's acceleration so the speed
that step CLOSES at is <= v_max, on EVERY candidate: the CEM population and seed pool through `_clip`, the
injected baselines through `_baseline_controls`. Every expectation below is a hand-computed LITERAL, and the
end-to-end test carries a discriminating control: the same planner WITHOUT the cap must exceed the limit, or
the test proves nothing.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs.refa_v1_plan import (PlanConfig, _baseline_controls, _cap_speed,
                                       _clip, icem_plan)

DT = 0.2


def _closing_speeds(a: torch.Tensor, v0: float, dt: float = DT) -> torch.Tensor:
    return v0 + torch.cumsum(a, dim=-1) * dt


def test_the_cap_is_OFF_by_default():
    assert PlanConfig().v_max is None


def test_a_non_binding_cap_is_bit_identical_to_no_cap():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 10, 2, generator=g) * 3.0            # beyond both actuator clamps
    off = _clip(x, PlanConfig(), v0=10.0)
    huge = _clip(x, PlanConfig(v_max=1.0e9), v0=10.0)
    assert torch.equal(off, huge)
    # and OFF equals the plain actuator box (kamm_mu is None by default)
    ref = torch.stack([x[..., 0].clamp(-4.0, 4.0), x[..., 1].clamp(-0.2, 0.2)], dim=-1)
    assert torch.equal(off, ref)


def test_the_cap_LITERAL_profile_accelerating_into_the_limit():
    # v0 = 10, a = +2 for 10 steps, v_max = 11:
    # step0 lim (11-10)/0.2 = 5 -> 2 (v 10.4); step1 lim 3 -> 2 (v 10.8); step2 lim 1 -> 1 (v 11.0); then 0.
    a = torch.full((1, 10), 2.0)
    got = _cap_speed(a, 10.0, 11.0, DT, 4.0)[0]
    want = torch.tensor([2.0, 2.0, 1.0, 0, 0, 0, 0, 0, 0, 0])
    assert torch.allclose(got, want, atol=1e-5)
    assert float(_closing_speeds(got, 10.0).max()) == pytest.approx(11.0, abs=1e-5)


def test_v0_ABOVE_the_limit_brakes_at_a_max_until_under_it():
    # v0 = 15, v_max = 11, a_max = 4: demanded decel is floored at -4 -> 15, 14.2, 13.4, 12.6, 11.8, 11.0, then hold.
    a = torch.zeros(1, 10)
    got = _cap_speed(a, 15.0, 11.0, DT, 4.0)[0]
    want = torch.tensor([-4.0, -4.0, -4.0, -4.0, -4.0, 0, 0, 0, 0, 0])
    assert torch.allclose(got, want, atol=1e-5)


def test_the_INJECTED_BASELINES_are_capped_when_v0_exceeds_the_limit():
    cfg = PlanConfig(horizon=10, v_max=11.0)
    out = _baseline_controls(cfg, 15.0, "cpu", None)
    want = torch.tensor([-4.0, -4.0, -4.0, -4.0, -4.0, 0, 0, 0, 0, 0])
    for name in ("cv", "hold_v0"):
        assert torch.allclose(out[name][:, 0], want, atol=1e-5), name
        assert torch.equal(out[name][:, 1], torch.zeros(10)), name


def test_the_injected_baselines_are_UNCHANGED_below_the_limit():
    off = _baseline_controls(PlanConfig(horizon=10), 10.0, "cpu", None)
    on = _baseline_controls(PlanConfig(horizon=10, v_max=20.0), 10.0, "cpu", None)
    assert set(off) == set(on)
    for k in off:
        assert torch.equal(off[k], on[k]), k


def _speed_seeking_cost(controls: torch.Tensor) -> torch.Tensor:
    """A cost that REWARDS acceleration, so an uncapped planner drives the speed up as far as it can."""
    return -controls[..., 0].sum(dim=-1)


def test_icem_plan_NEVER_exceeds_v_max_and_the_uncapped_control_DOES():
    # ⛔ The limit sits BELOW what the uncapped planner reaches. MEASURED on the first run (seed 0): with
    # v_max = 11 the uncapped plan topped out at 10.53 m/s, so "capped <= 11" would have passed VACUOUSLY.
    v0, vmax = 10.0, 10.2
    base = dict(horizon=10, dt=DT, n_samples=64, n_iters=4, n_elites=8, seed=0)
    uncapped = icem_plan(_speed_seeking_cost, v0=v0, cfg=PlanConfig(**base), device="cpu")
    capped = icem_plan(_speed_seeking_cost, v0=v0, cfg=PlanConfig(v_max=vmax, **base), device="cpu")
    top_uncapped = float(_closing_speeds(uncapped.controls[:, 0], v0).max())
    top_capped = float(_closing_speeds(capped.controls[:, 0], v0).max())
    # the discriminating control: without the cap this cost MUST push past the limit
    assert top_uncapped > vmax + 0.1, top_uncapped
    # with it, never past the limit, whichever candidate won
    assert top_capped <= vmax + 1e-4, (top_capped, capped.source)


def test_sanity_rejects_a_non_positive_limit():
    with pytest.raises(ValueError, match="v_max"):
        PlanConfig(v_max=0.0).sanity()
    PlanConfig(v_max=13.89).sanity()
