"""Tests for ``tanitad.rl.control_space`` — the CONTROL-space two-scalar policy.

⭐ THE LOAD-BEARING TEST is :func:`test_control_space_is_flyable_BY_CONSTRUCTION`
paired with :func:`test_metre_space_regression_DOES_violate`: the first shows the
hypothesis arm reads envelope violation **exactly 0.0** under adversarial noise,
the second shows the pre-registered deliberate-regression arm does **not** — so
the flyability claim is discriminating rather than vacuous. A property that both
arms satisfy proves nothing.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.rl import control_space as CS
from tanitad.rl import rewards as R


def _controls(B=3, N=5, K=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(B, N, K, generator=g) - 0.5) * 2 * (R.A_MAX_MPS2 * 0.6)
    k = (torch.rand(B, N, K, generator=g) - 0.5) * 2 * (R.KAPPA_MAX_1PM * 0.6)
    return torch.stack((a, k), dim=-1)


def _state0(B=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(B, 4)
    s[:, 3] = 5.0 + 10.0 * torch.rand(B, generator=g)      # v in [5, 15) m/s
    return s


# ---------------------------------------------------------------------------
# ⭐⭐ Flyability by construction — and its discriminating counterpart
# ---------------------------------------------------------------------------

def test_control_space_is_flyable_BY_CONSTRUCTION():
    """Even at 25x the published sigma, every sample stays inside the envelope."""
    c, s0 = _controls(), _state0()
    gen = torch.Generator().manual_seed(7)
    _, _, ctl = CS.sample_control_space(s0, c, group=4, dt=0.5, sigma=1.0,
                                        generator=gen)
    v = CS.envelope_violation(ctl)
    assert float(v.max()) == pytest.approx(0.0, abs=1e-12), \
        "a control-space sample must be flyable EXACTLY, not approximately"


def test_metre_space_regression_DOES_violate():
    """⛔ THE DISCRIMINATING CONTROL. Scaling WAYPOINTS (the pre-registered
    `reg_metre` arm) produces controls outside the envelope, so the flyability
    property above is a real distinction and not a tautology."""
    c, s0 = _controls(), _state0()
    base = CS.roll_controls(s0[:, None, :].expand(3, 5, 4), c, dt=0.5)
    stretched = base * 3.0                                  # metre-space scaling
    kin = R.kinematics(stretched.reshape(-1, base.shape[-2], 2), dt=0.5)
    over = torch.maximum(kin.accel.abs().amax(-1) / R.A_MAX_MPS2,
                         kin.kappa.abs().amax(-1) / R.KAPPA_MAX_1PM)
    assert float(over.max()) > 1.0, \
        "if metre-space noise never violated, `reg_metre` would be a vacuous arm"


def test_envelope_violation_reads_zero_on_a_compliant_input_and_positive_above():
    ok = torch.zeros(1, 3, 2)
    ok[..., 0] = R.A_MAX_MPS2
    ok[..., 1] = R.KAPPA_MAX_1PM
    assert float(CS.envelope_violation(ok).max()) == pytest.approx(0.0, abs=1e-9)
    bad = ok.clone()
    bad[0, 1, 0] = 2.0 * R.A_MAX_MPS2
    assert float(CS.envelope_violation(bad)[0]) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# The two scalars — the published mechanism, not a per-waypoint field
# ---------------------------------------------------------------------------

def test_exactly_TWO_scalars_move_per_candidate():
    """⭐ The published family is 2-parameter. If the noise were per-waypoint the
    ratio ctl/base would differ across K, and the mechanism would not be DD-v2's."""
    c, s0 = _controls(B=2, N=3, K=9, seed=3), _state0(B=2)
    gen = torch.Generator().manual_seed(11)
    _, _, ctl = CS.sample_control_space(s0, c, group=3, dt=0.5, sigma=0.05,
                                        generator=gen)
    base = c.unsqueeze(2)
    for axis in (0, 1):
        ratio = ctl[..., axis] / base[..., axis]
        spread = ratio.amax(dim=-1) - ratio.amin(dim=-1)     # over K
        assert float(spread.abs().max()) < 1e-5, \
            f"axis {axis} scale varies across waypoints: not a two-scalar family"


def test_the_two_axes_are_INDEPENDENT():
    s_lon, s_lat, _ = CS.sample_control_scales(
        (4000,), sigma=0.2, generator=torch.Generator().manual_seed(5))
    r = torch.corrcoef(torch.stack((s_lon, s_lat)))[0, 1]
    assert abs(float(r)) < 0.08, f"lon/lat scales correlated at {float(r):.3f}"


def test_scales_are_centred_on_ONE_with_the_published_sigma():
    s_lon, s_lat, _ = CS.sample_control_scales(
        (20000,), sigma=CS.SIGMA_EXPLORE_FLOOR,
        generator=torch.Generator().manual_seed(6))
    assert float(s_lon.mean()) == pytest.approx(1.0, abs=0.002)
    assert float(s_lon.std()) == pytest.approx(CS.SIGMA_EXPLORE_FLOOR, rel=0.05)
    assert float(s_lat.std()) == pytest.approx(CS.SIGMA_EXPLORE_FLOOR, rel=0.05)


def test_sigma_zero_limit_reproduces_the_BASE_exactly():
    """⭐ CONTROL WITH A KNOWN VALUE: with no exploration the sample must be the
    unperturbed candidate, bit for bit."""
    c, s0 = _controls(B=2, N=2, K=6, seed=8), _state0(B=2)
    gen = torch.Generator().manual_seed(2)
    _, _, ctl = CS.sample_control_space(s0, c, group=2, dt=0.5, sigma=1e-12,
                                        generator=gen)
    assert torch.allclose(ctl, c.unsqueeze(2).expand_as(ctl), atol=1e-6)


# ---------------------------------------------------------------------------
# The likelihood, stated for what it is
# ---------------------------------------------------------------------------

def test_logp_uses_the_LIKELIHOOD_sigma_not_the_exploration_sigma():
    """⛔ They are DIFFERENT numbers in the published code (0.10 vs 0.04) and
    collapsing them silently rescales the gradient."""
    assert CS.SIGMA_LIKELIHOOD_FLOOR != CS.SIGMA_EXPLORE_FLOOR
    a = CS.sample_control_scales((512,), sigma=0.04, likelihood_sigma=0.10,
                                 generator=torch.Generator().manual_seed(4))[2]
    b = CS.sample_control_scales((512,), sigma=0.04, likelihood_sigma=0.20,
                                 generator=torch.Generator().manual_seed(4))[2]
    assert not torch.allclose(a, b)


def test_logp_is_the_density_of_TWO_scalars_not_of_2K_coordinates():
    """At eps = 0 the log-density is exactly 2 * (-log s - 0.5 log 2pi)."""
    _, _, logp = CS.sample_control_scales((3,), sigma=1e-12, likelihood_sigma=0.10)
    want = 2.0 * (-math.log(0.10) - 0.5 * math.log(2 * math.pi))
    assert float(logp[0]) == pytest.approx(want, rel=1e-5)


def test_logp_is_finite_and_shaped_per_group_member():
    c, s0 = _controls(B=2, N=4, K=7, seed=9), _state0(B=2)
    traj, logp, ctl = CS.sample_control_space(
        s0, c, group=4, dt=0.5, generator=torch.Generator().manual_seed(1))
    assert traj.shape == (2, 4, 4, 7, 2)
    assert logp.shape == (2, 4, 4)
    assert ctl.shape == (2, 4, 4, 7, 2)
    assert torch.isfinite(logp).all() and torch.isfinite(traj).all()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_group_of_one_is_REFUSED():
    """A group-relative advantage over one sample is identically zero — a silent
    no-op arm, which this programme has already paid a GPU-hour for."""
    with pytest.raises(CS.ControlSpaceError, match="group"):
        CS.sample_control_space(_state0(B=1), _controls(B=1), group=1, dt=0.5)


def test_bad_shapes_are_loud():
    with pytest.raises(CS.ControlSpaceError):
        CS.sample_control_space(torch.zeros(2, 3), _controls(B=2), group=2, dt=0.5)
    with pytest.raises(CS.ControlSpaceError):
        CS.sample_control_space(_state0(B=2), torch.zeros(2, 3, 4, 5),
                                group=2, dt=0.5)
    with pytest.raises(CS.ControlSpaceError):
        CS.sample_control_space(_state0(B=3), _controls(B=2), group=2, dt=0.5)


def test_non_finite_controls_are_refused():
    c = _controls(B=1, N=2, K=4)
    c[0, 0, 0, 0] = float("nan")
    with pytest.raises(CS.ControlSpaceError):
        CS.sample_control_space(_state0(B=1), c, group=2, dt=0.5)


def test_negative_sigma_is_refused():
    with pytest.raises(CS.ControlSpaceError):
        CS.sample_control_scales((4,), sigma=-0.1)


def test_rollout_is_reproducible_under_a_seeded_generator():
    c, s0 = _controls(B=2, N=3, K=5, seed=12), _state0(B=2)
    a = CS.sample_control_space(s0, c, group=3, dt=0.5,
                                generator=torch.Generator().manual_seed(99))[0]
    b = CS.sample_control_space(s0, c, group=3, dt=0.5,
                                generator=torch.Generator().manual_seed(99))[0]
    assert torch.equal(a, b)
