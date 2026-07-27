"""Tests for ``tanitad.ego_plan`` — the ego/action seam and the plan geometry.

The load-bearing tests are the IDENTITY ones: the re-timing instrument is only
admissible because composing a plan's own shape with its own schedule returns
the plan bit-exactly, and composing a straight shape with a constant-velocity
schedule returns ``cv_holdv0`` bit-exactly. Two of the four cells of the
2026-07-28 factorial are already-published arms, so the construction validates
itself in BOTH directions rather than being asserted.
"""

from __future__ import annotations

import math

import pytest
import torch

from tanitad.ego_plan import (SPEED_SCALE, arc_length, attach_ego_input,
                              constant_speed_schedule, ego_vector, resample_by_arclength,
                              retime, straight_plan, terminal_tangent)

DT = 0.1


# --------------------------------------------------------------------------- #
# arc length                                                                   #
# --------------------------------------------------------------------------- #
def test_arc_length_starts_at_zero_and_counts_from_the_ego_origin():
    # a plan is [n, H, 2] where traj[:, 0] is step 1 -- the ego's own (0, 0) at
    # step 0 is IMPLICIT and must be included or every length is short by the
    # first segment.
    traj = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    s = arc_length(traj)
    assert s.shape == (1, 4)
    assert s[0, 0] == 0.0
    torch.testing.assert_close(s[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_arc_length_of_an_L_shape_is_the_sum_of_the_legs():
    traj = torch.tensor([[[3.0, 0.0], [3.0, 4.0]]])
    s = arc_length(traj)
    torch.testing.assert_close(s[0], torch.tensor([0.0, 3.0, 7.0]))


def test_arc_length_of_a_standing_still_plan_is_zero():
    traj = torch.zeros(1, 5, 2)
    assert float(arc_length(traj)[0, -1]) == 0.0


def test_arc_length_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match=r"\[n, H, 2\]"):
        arc_length(torch.zeros(4, 3))


# --------------------------------------------------------------------------- #
# THE IDENTITY: shape o its own schedule == the plan                           #
# --------------------------------------------------------------------------- #
def test_retiming_a_plan_to_its_OWN_arc_length_returns_it_exactly():
    """⭐ The instrument is only admissible if this holds. If re-timing a plan to
    the schedule it already had changed it, every downstream delta would be
    contaminated by the transform instead of by the intervention."""
    g = torch.Generator().manual_seed(0)
    traj = torch.cumsum(torch.rand(64, 20, 2, generator=g) * 0.5 + 0.1, dim=1)
    own = arc_length(traj)[:, 1:]            # drop the leading 0 -> [n, H]
    out = retime(traj, own)
    torch.testing.assert_close(out["traj"], traj, rtol=1e-5, atol=1e-5)
    assert not bool(out["extrapolated"].any())
    assert not bool(out["degenerate"].any())


def test_straight_shape_with_a_constant_speed_schedule_IS_cv_holdv0():
    """⭐ The other self-validating cell. ``cv_holdv0`` is defined in the panel
    driver as ``x = v0 * t, y = 0`` with ``t = arange(1, H+1) * DT`` -- exactly
    ``straight_plan(constant_speed_schedule(...))``. Bit-exact or the factorial
    is not measuring what it says."""
    v0 = torch.tensor([0.0, 3.7, 12.4, 31.9])
    horizon = 20
    got = straight_plan(constant_speed_schedule(v0, horizon, DT))
    t = torch.arange(1, horizon + 1, dtype=torch.float32) * DT
    want = torch.stack([v0[:, None] * t[None], torch.zeros(4, horizon)], dim=-1)
    torch.testing.assert_close(got, want)


def test_retiming_a_straight_plan_to_a_new_speed_is_that_speed():
    v0_old, v0_new = torch.tensor([10.0]), torch.tensor([5.0])
    old = straight_plan(constant_speed_schedule(v0_old, 20, DT))
    out = retime(old, constant_speed_schedule(v0_new, 20, DT))
    want = straight_plan(constant_speed_schedule(v0_new, 20, DT))
    torch.testing.assert_close(out["traj"], want, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# the shape is PRESERVED -- only the schedule moves                            #
# --------------------------------------------------------------------------- #
def test_retiming_keeps_every_new_point_on_the_original_curve():
    """A re-timed plan must trace the SAME curve. Measured as: every re-timed
    point lies (to tolerance) on the original polyline."""
    g = torch.Generator().manual_seed(7)
    traj = torch.cumsum(torch.rand(16, 20, 2, generator=g) * 0.6 + 0.05, dim=1)
    total = arc_length(traj)[:, -1]
    # a schedule strictly INSIDE the path so nothing extrapolates
    frac = torch.linspace(0.05, 0.95, 20)[None, :]
    out = retime(traj, total[:, None] * frac)
    poly = torch.cat([torch.zeros(16, 1, 2), traj], dim=1)     # [n, 21, 2]
    a, b = poly[:, :-1], poly[:, 1:]                            # segments
    p = out["traj"]
    # distance from each new point to the nearest segment
    ab = (b - a)[:, None]                                       # [n,1,H,2]
    ap = p[:, :, None] - a[:, None]                             # [n,H,H,2]
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    proj = a[:, None] + t[..., None] * ab
    d = (p[:, :, None] - proj).norm(dim=-1).amin(dim=-1)
    assert float(d.max()) < 1e-4, f"re-timed point left the curve by {d.max()}"


def test_retiming_does_not_change_the_total_path_shape_length_ordering():
    g = torch.Generator().manual_seed(3)
    traj = torch.cumsum(torch.rand(8, 20, 2, generator=g) + 0.1, dim=1)
    out = retime(traj, constant_speed_schedule(torch.full((8,), 5.0), 20, DT))
    s = arc_length(out["traj"])[:, 1:]
    assert bool((s.diff(dim=-1) >= -1e-4).all()), "schedule must be monotone"


# --------------------------------------------------------------------------- #
# extrapolation past the end of the path                                       #
# --------------------------------------------------------------------------- #
def test_past_the_end_it_extends_along_the_terminal_tangent():
    traj = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])        # heading +x, len 2
    out = retime(traj, torch.tensor([[1.0, 5.0]]))
    torch.testing.assert_close(out["traj"][0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(out["traj"][0, 1], torch.tensor([5.0, 0.0]))
    assert bool(out["extrapolated"][0, 1])
    assert not bool(out["extrapolated"][0, 0])


def test_terminal_tangent_uses_the_last_NON_degenerate_segment():
    # a plan that stops moving at the end still has a direction
    traj = torch.tensor([[[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]]])
    tan, dead = terminal_tangent(traj)
    torch.testing.assert_close(tan[0], torch.tensor([0.0, 1.0]))
    assert not bool(dead[0])


def test_a_plan_that_never_moves_is_flagged_and_never_invented():
    """⛔ A silently-invented tangent is how a degenerate arm sneaks a score.
    The standing-still adversary must come back flagged, not extended."""
    traj = torch.zeros(2, 5, 2)
    out = retime(traj, constant_speed_schedule(torch.tensor([10.0, 10.0]), 5, DT))
    assert bool(out["degenerate"].all())
    torch.testing.assert_close(out["traj"], torch.zeros(2, 5, 2))
    assert not bool(out["extrapolated"].any())


def test_frac_extrapolated_is_reported():
    traj = torch.tensor([[[1.0, 0.0]], [[100.0, 0.0]]])
    out = retime(traj, torch.tensor([[10.0], [10.0]]))
    assert float(out["frac_extrapolated"]) == 0.5


def test_negative_schedule_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        retime(torch.ones(1, 3, 2), -torch.ones(1, 3))


def test_schedule_shape_is_checked():
    with pytest.raises(ValueError, match=r"s_new must be"):
        retime(torch.ones(2, 3, 2), torch.ones(2, 4))


# --------------------------------------------------------------------------- #
# DIRECTION OF THE CONTROL -- the trap that cost a sibling a whole result       #
# --------------------------------------------------------------------------- #
def test_the_half_speed_control_travels_exactly_half_on_a_straight_path():
    """⛔ A degradation that makes the composite go UP is the named trap on this
    surface. The degradation itself must at least be a real degradation. On a
    straight path chord == arc, so the factor is EXACT."""
    traj = straight_plan(constant_speed_schedule(torch.full((8,), 12.0), 20, DT))
    v0 = torch.linspace(1.0, 20.0, 8)
    full = retime(traj, constant_speed_schedule(v0, 20, DT, scale=1.0))["traj"]
    half = retime(traj, constant_speed_schedule(v0, 20, DT, scale=0.5))["traj"]
    torch.testing.assert_close(arc_length(half)[:, -1],
                               arc_length(full)[:, -1] * 0.5,
                               rtol=1e-5, atol=1e-5)


def test_the_half_speed_control_always_travels_strictly_less_when_curved():
    """On a CURVED path the re-timed polyline's chordal length is slightly under
    its schedule (the chords cut the corners), so the ratio is not exactly 0.5 —
    but the ORDERING, which is all the control needs, is strict."""
    g = torch.Generator().manual_seed(11)
    traj = torch.cumsum(torch.rand(32, 20, 2, generator=g) + 0.2, dim=1)
    v0 = torch.rand(32, generator=g) * 20 + 1
    full = retime(traj, constant_speed_schedule(v0, 20, DT, scale=1.0))["traj"]
    half = retime(traj, constant_speed_schedule(v0, 20, DT, scale=0.5))["traj"]
    s_full, s_half = arc_length(full)[:, -1], arc_length(half)[:, -1]
    assert bool((s_half < s_full).all())
    ratio = (s_half / s_full.clamp_min(1e-6))
    assert bool(((ratio > 0.40) & (ratio < 0.55)).all()), f"ratio {ratio}"
    # and the ENDPOINT the metric actually reads (along-track x) also drops
    assert bool((half[:, -1, 0] < full[:, -1, 0]).all())


# --------------------------------------------------------------------------- #
# the ego vector contract                                                      #
# --------------------------------------------------------------------------- #
def test_ego_vector_matches_the_trainer_contract():
    v0 = torch.tensor([12.0, 0.0])
    yr = torch.tensor([0.3, -0.2])
    got = ego_vector(v0, yr, pose_scale=4.0)
    torch.testing.assert_close(got, torch.tensor([[3.0, 0.3], [0.0, -0.2]]))


def test_speed_scale_is_not_the_pose_scale():
    """⚠️ The operative ACTION channel divides by 10.0; the planner EGO vector
    divides by pose_scale. Swapping them decodes garbage (registry §1.2)."""
    assert SPEED_SCALE == 10.0
    v0 = torch.tensor([20.0])
    assert float(ego_vector(v0, torch.zeros(1), pose_scale=4.0)[0, 0]) != \
        float(v0 / SPEED_SCALE)


def test_ego_vector_rejects_a_zero_scale():
    with pytest.raises(ValueError):
        ego_vector(torch.ones(1), torch.ones(1), pose_scale=0.0)


# --------------------------------------------------------------------------- #
# the graft                                                                    #
# --------------------------------------------------------------------------- #
def _tactical(ego_input: bool = False):
    from tanitad.config import TacticalPolicyConfig
    from tanitad.models.fourbrain import TacticalPolicy
    cfg = TacticalPolicyConfig(d_model=32, depth=1, n_heads=2, d_intent=8,
                               waypoint_horizons=[5, 10])
    torch.manual_seed(0)
    return TacticalPolicy(cfg, state_dim=16, window=4, d_cond=12,
                          ego_input=ego_input)


def test_a_stock_tactical_policy_has_no_ego_embedding():
    """⭐ THE PREMISE, pinned as a test so it cannot rot. The parameter EXISTS in
    the signature but the module is None unless v2_ego_to_planners was on."""
    pol = _tactical(ego_input=False)
    assert pol.ego_emb is None
    import inspect
    assert "ego" in inspect.signature(pol.forward).parameters


def test_passing_ego_to_a_stock_policy_is_SILENTLY_IGNORED():
    """⛔ The second gap, pinned. Handing a stock policy an ego vector changes
    NOTHING and raises NOTHING -- which is why an ego-trained checkpoint
    evaluated through the shipped harness is silently ego-blind."""
    pol = _tactical(ego_input=False).eval()
    st = torch.randn(3, 4, 16)
    ctx = torch.randn(3, 12)
    with torch.no_grad():
        a = pol(st, ctx)["waypoints"][5]
        b = pol(st, ctx, ego=torch.randn(3, 2) * 100)["waypoints"][5]
    torch.testing.assert_close(a, b)


def test_graft_is_numerically_identical_until_it_is_trained():
    """The graft must be a NO-OP at init, so the ego/no-ego contrast is a paired
    within-checkpoint comparison whose control direction is guaranteed."""
    pol = _tactical(ego_input=False).eval()
    st = torch.randn(5, 4, 16)
    ctx = torch.randn(5, 12)
    with torch.no_grad():
        before = pol(st, ctx)["waypoints"][10]
    attach_ego_input(pol, d_cond=12)
    assert pol.ego_emb is not None
    with torch.no_grad():
        after = pol(st, ctx, ego=torch.randn(5, 2) * 50)["waypoints"][10]
    torch.testing.assert_close(before, after)


def _wake_the_film(pol):
    """A FRESHLY BUILT policy ignores ``ctx`` entirely: ``FiLM.to_scale_shift`` is
    ZERO-INITIALISED (``tanitad/models/predictor.py:25-26``), so the whole cond
    path — ctx AND any ego graft — has zero effect and zero gradient at init. A
    TRAINED policy does not, which is the case that matters. Give the FiLM
    non-zero weights so these tests exercise the trained regime."""
    with torch.no_grad():
        for blk in pol.blocks:
            blk.film.to_scale_shift.weight.normal_(0.0, 0.05)


def test_a_fresh_policy_ignores_ctx_entirely_because_film_is_zero_init():
    """⚠️ Pinned because it changes how a v5 ego graft must be trained: on a
    from-scratch policy the ego seam has NO gradient until the FiLM moves."""
    pol = _tactical(ego_input=False).eval()
    st = torch.randn(4, 4, 16)
    with torch.no_grad():
        a = pol(st, torch.zeros(4, 12))["waypoints"][10]
        b = pol(st, torch.randn(4, 12) * 10)["waypoints"][10]
    torch.testing.assert_close(a, b)


def test_grafted_policy_responds_to_ego_once_the_graft_is_nonzero():
    """VALIDATION IN THE OTHER DIRECTION: the seam must be able to matter, or
    the identity test above proves nothing but that it is dead."""
    pol = _tactical(ego_input=False).eval()
    _wake_the_film(pol)
    attach_ego_input(pol, d_cond=12)
    with torch.no_grad():
        pol.ego_emb.weight.fill_(0.5)
    st = torch.randn(5, 4, 16)
    ctx = torch.randn(5, 12)
    with torch.no_grad():
        a = pol(st, ctx, ego=torch.zeros(5, 2))["waypoints"][10]
        b = pol(st, ctx, ego=torch.full((5, 2), 3.0))["waypoints"][10]
    assert float((a - b).abs().max()) > 1e-4


def test_graft_matches_the_shipped_lever_shape():
    """A grafted policy must be state-dict-compatible with a ``--v2`` run, or the
    graft is a different architecture wearing the same name."""
    stock = _tactical(ego_input=False)
    lever = _tactical(ego_input=True)
    attach_ego_input(stock, d_cond=12)
    assert set(stock.state_dict()) == set(lever.state_dict())
    assert stock.ego_emb.weight.shape == lever.ego_emb.weight.shape


def test_graft_refuses_to_overwrite_a_trained_ego_embedding():
    lever = _tactical(ego_input=True)
    with pytest.raises(ValueError, match="already has"):
        attach_ego_input(lever, d_cond=12)


def test_graft_refuses_a_module_without_the_slot():
    with pytest.raises(TypeError, match="no ego_emb slot"):
        attach_ego_input(torch.nn.Linear(2, 2), d_cond=4)


def test_strategic_policy_has_the_same_gap_and_the_same_graft():
    from tanitad.config import StrategicPolicyConfig
    from tanitad.models.fourbrain import StrategicPolicy
    cfg = StrategicPolicyConfig(d_model=32, depth=1, n_heads=2, d_cmd=12,
                                d_ctx=16)
    torch.manual_seed(0)
    pol = StrategicPolicy(cfg, state_dim=16, window=4).eval()
    assert pol.ego_emb is None                     # the same gap
    st, nav = torch.randn(3, 4, 16), torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        before = pol(st, nav)["ctx"]
    attach_ego_input(pol)                          # d_cond inferred from d_cmd
    assert pol.ego_emb.out_features == 12
    with torch.no_grad():
        after = pol(st, nav, ego=torch.randn(3, 2))["ctx"]
    torch.testing.assert_close(before, after)


# --------------------------------------------------------------------------- #
# the numbers the report quotes, recomputable without a GPU                    #
# --------------------------------------------------------------------------- #
def test_a_speed_blind_head_cannot_calibrate_distance_but_a_schedule_can():
    """The whole argument in one assertion: two windows at very different speeds
    get the SAME plan from a speed-blind head, and re-timing separates them."""
    shape = torch.tensor([[1.0, 0.0], [2.0, 0.05], [3.0, 0.2]])[None]
    plan = shape.expand(2, 3, 2).contiguous()           # identical plans
    v0 = torch.tensor([5.0, 25.0])                      # very different speeds
    assert float((arc_length(plan)[0, -1] - arc_length(plan)[1, -1]).abs()) == 0.0
    out = retime(plan, constant_speed_schedule(v0, 3, dt=1.0))
    d = arc_length(out["traj"])[:, -1]
    assert float(d[1] / d[0]) == pytest.approx(5.0, rel=1e-3)


def test_horizon_and_dt_match_the_pseudosim_grid():
    """The instrument's grid is 20 steps at 10 Hz = 2 s. If this drifts the
    factorial silently compares different horizons."""
    v0 = torch.tensor([10.0])
    sched = constant_speed_schedule(v0, 20, 0.1)
    assert sched.shape == (1, 20)
    assert float(sched[0, -1]) == pytest.approx(10.0 * 2.0)
    assert math.isclose(20 * 0.1, 2.0)
