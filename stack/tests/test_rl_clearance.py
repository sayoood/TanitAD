"""Tests for ``rewards.clearance`` — the single definition of "how close".

⭐ WHY THIS FILE EXISTS. ``clearance`` was extracted from ``_proximity`` so the
barrier and the pilot readout's R5 threshold the SAME quantity. Two independent
implementations of "close" is how a report ends up comparing a barrier's notion of
close with a readout's and calls the difference a result.

⛔ THE LOAD-BEARING TEST IS ``test_no_obstacles_is_nan_not_clear``. Every other
property here is arithmetic; that one encodes a lesson that has cost this program
real retractions (TRAIN-C2: a statistic pooled over windows where the quantity is
undefined is a DIFFERENT measurement; the E-DETECT-1 all-zero floor: an absence
that scores as success manufactures false positives). A thin obstacle join must
report thin, never safe.
"""
import pytest
import torch

from tanitad.rl import clearance
from tanitad.rl import rewards as RW


def _ctx(obs, **kw):
    d = {"obstacles": torch.tensor(obs, dtype=torch.float32)}
    d.update(kw)
    return d


def test_matches_hand_computed_distance():
    # straight path along +x; one obstacle 10 m ahead, 3 m to the side.
    traj = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    ctx = _ctx([[[10.0, 3.0]]])
    # nearest waypoint is (2,0): hypot(8,3) = 8.544, minus (1+1) = 6.544
    assert clearance(traj, ctx).item() == pytest.approx(8.544003 - 2.0, abs=1e-4)


def test_takes_the_worst_step_and_the_worst_obstacle():
    traj = torch.tensor([[[0.0, 0.0], [5.0, 0.0], [50.0, 0.0]]])
    far = _ctx([[[100.0, 0.0]]])
    near = _ctx([[[100.0, 0.0], [5.0, 1.0]]])
    assert clearance(traj, far).item() > clearance(traj, near).item()
    # the near obstacle sits 1 m off waypoint 2 ⇒ inside the 2 m radius ⇒ floored
    assert clearance(traj, near).item() == pytest.approx(0.0)


def test_penetration_floors_at_zero_never_negative():
    traj = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    assert clearance(traj, _ctx([[[1.0, 0.0]]])).item() == 0.0


def test_radii_are_read_from_ctx():
    traj = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    obs = [[[11.0, 0.0]]]
    base = clearance(traj, _ctx(obs)).item()                       # r = 2
    wide = clearance(traj, _ctx(obs, ego_radius_m=3.0, obs_radius_m=2.0)).item()
    assert base == pytest.approx(8.0)
    assert wide == pytest.approx(5.0)


def test_no_obstacles_is_nan_not_clear():
    """⛔ THE ONE THAT MATTERS. Absence must be UNDEFINED, never 'safe'.

    If this returned 0.0 the path would read as colliding everywhere; if it
    returned +inf (or a large number) a window with no join coverage would read
    as perfectly clear and DILUTE any violation rate toward zero. A thin join
    would then look like a safe policy.
    """
    traj = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    for ctx in ({}, {"obstacles": torch.zeros(0, 2)}):
        out = clearance(traj, ctx)
        assert out.shape == (1,)
        assert torch.isnan(out).all(), f"absence must be nan, got {out}"


def test_batch_and_candidate_shapes_are_preserved():
    """The documented shape: obstacles need an axis for the N candidates.

    ⚠️ This test was WRITTEN WRONG first (``[B, K, 2]``) and the failure is the
    reason the shape contract is now spelled out in the docstring. ``[B, K, 2]``
    broadcasts fine at ``B == 1`` and dies at ``B > 1``, so the wrong shape
    survives a single-window smoke test and fails later.
    """
    traj = torch.randn(2, 7, 4, 2)                       # [B, N, S, 2]
    ctx = {"obstacles": torch.randn(2, 1, 3, 2)}         # [B, 1, K, 2]
    assert clearance(traj, ctx).shape == (2, 7)


def test_wrong_obstacle_shape_fails_loudly_at_batch_gt_1():
    """It must not silently produce a differently-shaped answer."""
    traj = torch.randn(2, 7, 4, 2)
    with pytest.raises(RuntimeError):
        clearance(traj, {"obstacles": torch.randn(2, 3, 2)})


def test_proximity_barrier_agrees_with_clearance():
    """The barrier must be a FUNCTION of this quantity, not a second opinion."""
    traj = torch.randn(1, 5, 4, 2) * 3.0
    ctx = _ctx([[[6.0, 0.0], [12.0, 4.0]]])
    clr = clearance(traj, ctx)
    d_safe, p = 5.0, 2.0
    expect = -((1.0 - clr / d_safe).clamp(0.0, 1.0) ** p)
    assert torch.allclose(RW.COMPONENTS["proximity"](traj, ctx), expect, atol=1e-6)


def test_proximity_still_neutral_without_obstacles():
    """⚠️ REGRESSION: routing through a nan-returning helper must NOT leak nan
    into the reward. ``_proximity``'s documented contract is 0.0 (no
    information) when obstacles are absent, and the early return that provides
    it has to survive the refactor — a nan here would poison the whole composed
    reward and every gradient taken through it."""
    traj = torch.randn(1, 5, 4, 2)
    out = RW.COMPONENTS["proximity"](traj, {})
    assert torch.equal(out, torch.zeros(1, 5))
    assert not torch.isnan(out).any()
