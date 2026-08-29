"""KNOWN-VALUE controls for the RL reward components.

⛔ WHY THESE ARE ANALYTIC AND NOT REGRESSION SNAPSHOTS
------------------------------------------------------
A snapshot test pins whatever the code did on the day it was written, including
its bugs. Every test here computes the value BY HAND from the geometry and
asserts the component reproduces it. The commission names the canonical case —
*"a straight-line constant-speed trajectory scores its analytic value"* — and it
is the first test below.

The programme's measured lesson behind this: four separate estimator failures in
one afternoon each produced a confident, publishable-looking number, and each was
caught ONLY because a control read a value that was known in advance.
"""

from __future__ import annotations

import math

import pytest
import torch

from tanitad.rl import rewards as R


DT = 0.1


def straight(n=21, v=10.0, dt=DT, lateral=0.0):
    t = torch.arange(n, dtype=torch.float32)
    return torch.stack([v * dt * t, torch.full_like(t, lateral)], dim=-1)


# ---------------------------------------------------------------------------
# Kinematics — the analytic core everything else is built on
# ---------------------------------------------------------------------------

def test_straight_constant_speed_has_analytic_kinematics():
    """THE named control: straight line at constant v -> exact known values."""
    v = 10.0
    n = 21
    kin = R.kinematics(straight(n, v), DT)

    assert torch.allclose(kin.speed, torch.full_like(kin.speed, v), atol=1e-4)
    assert torch.allclose(kin.accel, torch.zeros_like(kin.accel), atol=1e-3)
    assert torch.allclose(kin.jerk, torch.zeros_like(kin.jerk), atol=1e-3)
    assert torch.allclose(kin.yaw_rate, torch.zeros_like(kin.yaw_rate), atol=1e-5)
    assert torch.allclose(kin.kappa, torch.zeros_like(kin.kappa), atol=1e-5)
    assert torch.allclose(kin.lat_acc, torch.zeros_like(kin.lat_acc), atol=1e-4)
    # arc length and along-track = v * T exactly
    expected = v * DT * (n - 1)
    assert kin.arc_len.item() == pytest.approx(expected, abs=1e-3)
    assert kin.along.item() == pytest.approx(expected, abs=1e-3)


def test_constant_acceleration_reads_its_analytic_accel():
    """x(t) = 0.5 a t^2  ->  accel == a, jerk == 0."""
    a = 2.0
    n = 21
    t = torch.arange(n, dtype=torch.float32) * DT
    traj = torch.stack([0.5 * a * t ** 2, torch.zeros_like(t)], dim=-1)
    kin = R.kinematics(traj, DT)
    # finite differences of a quadratic are exact up to the half-step offset
    assert torch.allclose(kin.accel, torch.full_like(kin.accel, a), atol=1e-3)
    assert torch.allclose(kin.jerk, torch.zeros_like(kin.jerk), atol=1e-2)


def test_circular_arc_reads_its_analytic_curvature():
    """A circle of radius Rm has kappa == 1/Rm everywhere."""
    radius, v = 20.0, 8.0
    omega = v / radius
    n = 31
    t = torch.arange(n, dtype=torch.float32) * DT
    traj = torch.stack([radius * torch.sin(omega * t),
                        radius * (1.0 - torch.cos(omega * t))], dim=-1)
    kin = R.kinematics(traj, DT)
    assert torch.allclose(kin.kappa, torch.full_like(kin.kappa, 1.0 / radius),
                          atol=2e-3)
    assert torch.allclose(kin.lat_acc,
                          torch.full_like(kin.lat_acc, v ** 2 / radius), atol=0.1)


def test_kinematics_refuses_bad_shapes():
    with pytest.raises(ValueError, match=r"\[\.\.\., S, 2\]"):
        R.kinematics(torch.zeros(5, 3), DT)
    with pytest.raises(ValueError, match="jerk"):
        R.kinematics(torch.zeros(3, 2), DT)


def test_stationary_path_reads_zero_curvature_not_infinity():
    """The MIN_SPEED guard: kappa is undefined at v=0 and must not explode."""
    kin = R.kinematics(torch.zeros(21, 2), DT)
    assert torch.isfinite(kin.kappa).all()
    assert torch.allclose(kin.kappa, torch.zeros_like(kin.kappa), atol=1e-6)


# ---------------------------------------------------------------------------
# Components — each against a hand-computed value
# ---------------------------------------------------------------------------

def test_progress_is_along_track_over_reference():
    v, n, ref = 10.0, 21, 30.0
    r = R.COMPONENTS["progress"](straight(n, v), {"progress_ref_m": ref})
    # clamped at hi=1.5; v*T = 20 m, 20/30 = 0.6667
    assert float(r) == pytest.approx(20.0 / ref, abs=1e-3)


def test_collision_fires_only_when_within_radius():
    traj = straight(21, 10.0)                       # runs along +x to x=20
    # obstacle sitting squarely on the path
    ctx_hit = {"obstacles": torch.tensor([[10.0, 0.0]]),
               "ego_radius_m": 1.0, "obs_radius_m": 1.0}
    assert float(R.COMPONENTS["collision"](traj, ctx_hit)) == pytest.approx(-1.0)
    # same obstacle moved 5 m laterally -> no contact
    ctx_miss = {**ctx_hit, "obstacles": torch.tensor([[10.0, 5.0]])}
    assert float(R.COMPONENTS["collision"](traj, ctx_miss)) == pytest.approx(0.0)


def test_collision_with_no_obstacles_is_zero_for_everyone():
    """No obstacles -> 0, which is `collision`'s NEUTRAL (and its best).

    Absence means no constraint, therefore no penalty. It is NOT a claim of
    safety: `audit.report_component_coverage` flags the component as not-fired
    so the absence stays visible instead of being read as a safeguard.
    """
    r = R.COMPONENTS["collision"](straight(), {})
    assert float(r) == pytest.approx(0.0)


def _headway_at(gap_time_s, t_star=2.0, v=10.0, n=21, lead_len=4.5):
    """Build a scene whose worst-step time gap is exactly `gap_time_s`."""
    ego = straight(n, v)
    lead = ego + torch.tensor([gap_time_s * v + lead_len, 0.0])
    return float(R.COMPONENTS["headway"](
        ego, {"lead_path": lead, "target_time_gap_s": t_star,
              "lead_len_m": lead_len}))


def test_graded_headway_peaks_exactly_at_the_target_gap():
    """⛔ The A0 fix. The reward's ARGMAX is the target gap itself."""
    assert _headway_at(2.0) == pytest.approx(1.0, abs=1e-3)
    assert _headway_at(1.0) < 1.0 and _headway_at(3.0) < 1.0


def test_graded_headway_analytic_values_below_target():
    """Below T*: (t/T*)**k with k=2. Steep, and hand-computable."""
    assert _headway_at(1.0) == pytest.approx(0.25, abs=1e-2)    # (0.5)^2
    assert _headway_at(1.5) == pytest.approx(0.5625, abs=2e-2)  # (0.75)^2
    assert _headway_at(0.0) == pytest.approx(0.0, abs=1e-3)


def test_graded_headway_analytic_values_above_target():
    """Above T*: mild linear decline, saturating at FAR=2x."""
    assert _headway_at(4.0) == pytest.approx(0.75, abs=1e-2)    # 1 - dawdle
    assert _headway_at(8.0) == pytest.approx(0.75, abs=1e-2)    # saturated


def test_headway_is_ASYMMETRIC_tailgating_hurts_exactly_3x_dawdling():
    """⭐ The design ruling: dawdling at 4 s is wrong; 1 s following is worse.

    The ratio is not a vague inequality — it is ANALYTIC and worth pinning as a
    number, so that changing `headway_tailgate_exp` or `headway_dawdle_penalty`
    has to be a deliberate decision rather than a silent drift:

        at t = T*/2  ->  (0.5)**2      = 0.25  ->  loss 0.75
        at t = 2*T*  ->  1 - dawdle    = 0.75  ->  loss 0.25
        ratio                                        = 3.0 exactly
    """
    tailgate_loss = 1.0 - _headway_at(1.0)     # half the target gap
    dawdle_loss = 1.0 - _headway_at(4.0)       # double the target gap
    assert tailgate_loss / dawdle_loss == pytest.approx(3.0, abs=0.15), (
        f"tailgating loss {tailgate_loss:.3f} vs dawdling {dawdle_loss:.3f}")
    assert tailgate_loss > dawdle_loss


def test_graded_headway_RANKS_on_both_sides_of_the_target():
    """⛔ THE PROPERTY A0 MEASURED MISSING: spread across candidates.

    The old saturating term read a median spread of 0.0000 across the fan and
    was therefore INERT in a group-relative advantage.
    """
    vals = [_headway_at(t) for t in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]
    assert len(set(round(v, 4) for v in vals)) >= 5, (
        f"headway must rank distinct gaps, got {vals}")
    assert max(vals) - min(vals) > 0.5


# ---------------------------------------------------------------------------
# The TTC veto — a CONSTRAINT, deliberately not a reward component
# ---------------------------------------------------------------------------

def test_ttc_veto_fires_when_closing_fast_on_a_static_lead():
    ego = straight(21, 20.0)                       # 20 m/s
    lead = torch.zeros(21, 2) + torch.tensor([15.0, 0.0])   # static, 15 m ahead
    assert bool(R.ttc_violation(ego, {"lead_path": lead, "ttc_min_s": 1.5,
                                      "lead_len_m": 0.0}))


def test_ttc_veto_is_quiet_when_matching_the_lead_speed():
    """Matching speed at a short gap is stable, not imminent — TTC uses the
    CLOSING speed, not the ego speed."""
    ego = straight(21, 20.0)
    lead = ego + torch.tensor([15.0, 0.0])         # same speed, constant gap
    assert not bool(R.ttc_violation(ego, {"lead_path": lead, "ttc_min_s": 1.5,
                                          "lead_len_m": 0.0}))


def test_ttc_veto_is_quiet_with_no_lead():
    assert not bool(R.ttc_violation(straight(21, 10.0), {}))


def test_ttc_veto_is_NOT_a_reward_component():
    """The separation is the point: constraints pin, ranking signals order."""
    assert "ttc" not in R.COMPONENTS
    assert not any("ttc" in n for n in R.DEFAULT_WEIGHTS)


def test_feasibility_is_one_inside_the_envelope_and_decays_outside():
    inside = R.COMPONENTS["feasibility"](straight(21, 10.0), {})
    assert float(inside) == pytest.approx(1.0, abs=1e-4)

    n = 21
    t = torch.arange(n, dtype=torch.float32) * DT
    violent = torch.stack([0.5 * 40.0 * t ** 2, torch.zeros_like(t)], dim=-1)
    out = R.COMPONENTS["feasibility"](violent, {})
    assert 0.0 <= float(out) < 0.2


def test_comfort_is_one_for_a_smooth_path():
    assert float(R.COMPONENTS["comfort"](straight(21, 10.0), {})) == pytest.approx(
        1.0, abs=1e-3)


def test_gt_similarity_is_one_at_zero_ade_and_decays():
    traj = straight(21, 10.0)
    exact = R.COMPONENTS["gt_similarity"](traj, {"gt_traj": traj})
    assert float(exact) == pytest.approx(1.0, abs=1e-6)
    off = R.COMPONENTS["gt_similarity"](
        traj, {"gt_traj": traj + torch.tensor([0.0, 2.0]), "ade_scale_m": 2.0})
    assert float(off) == pytest.approx(math.exp(-1.0), abs=1e-3)


def test_every_component_is_bounded_as_declared():
    """Bounds are the anti-amplifier: one unbounded term dominates a sum."""
    panel = [straight(21, 0.0), straight(21, 30.0), straight(21, 10.0, lateral=3.0)]
    ctx = {"obstacles": torch.tensor([[5.0, 0.0]]), "gt_traj": straight(21, 10.0),
           "lead_path": straight(21, 10.0) + torch.tensor([25.0, 0.0])}
    for name, comp in R.COMPONENTS.items():
        for traj in panel:
            v = float(comp(traj, ctx))
            assert comp.lo - 1e-6 <= v <= comp.hi + 1e-6, (name, v)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_reward_spec_rejects_unknown_components():
    with pytest.raises(KeyError, match="unknown reward components"):
        R.RewardSpec(weights={"not_a_component": 1.0})


def test_reward_spec_rejects_empty():
    with pytest.raises(ValueError, match="no components"):
        R.RewardSpec(weights={})


def test_default_weights_reference_only_real_components():
    assert set(R.DEFAULT_WEIGHTS) <= set(R.COMPONENTS)
    assert set(R.HACKABLE_WEIGHTS) <= set(R.COMPONENTS)


def test_spec_is_the_weighted_sum_of_its_parts():
    spec = R.RewardSpec()
    traj = straight(21, 10.0)
    ctx = {"gt_traj": traj}
    parts = spec.per_component(traj, ctx)
    manual = sum(spec.weights[k] * float(v) for k, v in parts.items())
    assert float(spec(traj, ctx)) == pytest.approx(manual, abs=1e-5)


def test_batched_shapes_are_preserved():
    """[B, N, G, S, 2] in -> [B, N, G] out, for every component."""
    traj = straight(21, 10.0).expand(2, 3, 4, 21, 2).contiguous()
    spec = R.RewardSpec()
    out = spec(traj, {"gt_traj": straight(21, 10.0)})
    assert out.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# ABSENCE SEMANTICS + the imitation double-count (peer review, 2026-08-29)
# ---------------------------------------------------------------------------

def test_absence_means_no_penalty_for_EVERY_component():
    """⛔ The E.3 fix.

    Before this, `collision` absent read its BEST value while `headway` and
    `gt_similarity` absent read their WORST — the same underlying fact ("no
    such constraint in this scene") scored in opposite directions. On a corpus
    where most windows have no lead vehicle, that silently penalised ordinary
    open road.
    """
    traj = straight(21, 10.0)
    for name, comp in R.COMPONENTS.items():
        if name == "progress":
            continue          # progress needs no scene fact; it is always defined
        v = float(comp(traj, {}))
        assert v == pytest.approx(comp.neutral, abs=1e-6), (
            f"{name} with an absent scene fact read {v}, not its declared "
            f"neutral {comp.neutral}")
        assert v == pytest.approx(comp.hi, abs=1e-6), (
            f"{name}'s neutral must be its BEST value — absence means no "
            f"constraint, therefore no penalty (got {v}, hi={comp.hi})")


def test_no_lead_is_not_punished():
    """The concrete case: open road must not score worse than car-following."""
    traj = straight(21, 10.0)
    assert float(R.COMPONENTS["headway"](traj, {})) == pytest.approx(1.0)


def test_gt_similarity_is_NOT_in_the_default_reward():
    """⛔ The E.2 fix — an imitation term inside a group-relative advantage is
    a fan-collapse objective."""
    assert "gt_similarity" not in R.DEFAULT_WEIGHTS
    assert "gt_similarity" in R.COMPONENTS, (
        "the component stays available as a diagnostic; only the DEFAULT drops it")
