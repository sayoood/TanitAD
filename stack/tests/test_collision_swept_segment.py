"""`_collision` must test the SWEPT SEGMENT, not the sampled waypoints.

⛔ **The hole has a SIZE, and it is metres.** Waypoints are `v*dt` apart; with a
combined radius `r = ego_r + obs_r` an obstacle more than `r` from BOTH endpoints
of a segment it sits inside is invisible to a point test. The undetected corridor
per segment is `v*dt - 2r` — at 15 m/s on a 0.5 s grid that is 7.5 - 4.0 = **3.5 m**.

Raised by the refcv5 build stream as its escalation #1 and confirmed at source:
the pre-fix static branch was `(traj.unsqueeze(-2) - obs.unsqueeze(-3)).norm(-1)`,
pure point-to-point.

⚠️ This is a STRICTER instrument, not a corrected one. Numbers banked under the
point test (the RL panel's `sel_contact` 0.0000 / 0.0007) are LOWER BOUNDS; a
swept re-read is not a regression against them and must not be reported as one.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.rl.rewards import _collision, _swept_hit, segment_point_distance  # noqa: E402


def _ctx(obs, ego_r=1.0, obs_r=1.0):
    return {"obstacles": torch.tensor(obs, dtype=torch.float32),
            "ego_radius_m": ego_r, "obs_radius_m": obs_r}


def _point_hit(traj, obs, r):
    """The PRE-FIX expression, verbatim, so the control is the real thing."""
    d = (traj.unsqueeze(-2) - obs.unsqueeze(-3)).norm(dim=-1)
    return (d < r).any(dim=-1).any(dim=-1)


# ------------------------------------------------ the deliberate-regression control


def test_the_straddling_candidate_the_point_test_waved_through():
    """⛔ Without this the suite proves nothing. Both waypoints sit 3.0 m from the
    obstacle (r = 2.0, so both are clear) while the segment passes through it."""
    traj = torch.tensor([[[-3.0, 0.0], [3.0, 0.0]]])      # [1, 2, 2]
    obs = torch.tensor([[[0.0, 0.0]]])                    # [1, 1, 2]
    r = 2.0

    assert not bool(_point_hit(traj, obs, r)), \
        "the control must reproduce the DEFECT — if the point test catches this, " \
        "the fixture is wrong and the test below proves nothing"
    assert bool(_swept_hit(traj, obs, r)), "the swept test must catch it"
    assert float(_collision(traj, {"obstacles": obs, "ego_radius_m": 1.0,
                                   "obs_radius_m": 1.0})) == -1.0


def test_the_corridor_the_old_test_left_open_is_metres_not_rounding():
    """Size the hole, so the fix carries a number rather than an assertion."""
    r = 2.0
    for v, dt in ((15.0, 0.5), (30.0, 0.5), (10.0, 0.2)):
        spacing = v * dt
        corridor = spacing - 2 * r
        if corridor <= 0:
            continue
        # an obstacle at the segment midpoint, offset just inside r
        traj = torch.tensor([[[0.0, 0.0], [spacing, 0.0]]])
        obs = torch.tensor([[[spacing / 2.0, 0.0]]])
        assert not bool(_point_hit(traj, obs, r)), \
            f"v={v} dt={dt}: spacing {spacing} m leaves a {corridor} m corridor"
        assert bool(_swept_hit(traj, obs, r))


# ------------------------------------------------------------- it never loses one


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_swept_never_detects_less_than_the_point_test(seed):
    """The property that makes banked numbers a LOWER BOUND rather than
    incomparable: the swept form is a superset of the point form, always."""
    g = torch.Generator().manual_seed(seed)
    traj = torch.randn(16, 12, 2, generator=g) * 5.0
    obs = torch.randn(16, 4, 2, generator=g) * 5.0
    r = 2.0
    pt = _point_hit(traj, obs, r)
    sw = _swept_hit(traj, obs, r)
    assert bool((sw | pt).eq(sw).all()), "swept dropped a detection the point test made"


# ----------------------------------------------------------------- the degenerate


def test_a_stopped_ego_falls_back_to_the_point_test_and_does_not_divide_by_zero():
    traj = torch.zeros(1, 6, 2)                      # every waypoint identical
    near = torch.tensor([[[1.5, 0.0]]])
    far = torch.tensor([[[9.0, 0.0]]])
    assert bool(_swept_hit(traj, near, 2.0))
    assert not bool(_swept_hit(traj, far, 2.0))
    assert torch.isfinite(segment_point_distance(traj[:, :-1], traj[:, 1:], far)).all()


# ------------------------------------------------------ the time-aligned lead guard


def test_a_competent_follower_with_a_time_gap_is_still_not_flagged():
    """⛔ The H-RL-THRESH-1 failure the moving branch's docstring exists to
    prevent: holding the lead STATIC turns every competent follower into a
    collision. Sweeping in the RELATIVE frame must not re-introduce it."""
    steps = 12
    t = torch.arange(steps, dtype=torch.float32) * 0.5
    ego = torch.stack([10.0 * t, torch.zeros(steps)], dim=-1)[None]
    lead = torch.stack([10.0 * t + 20.0, torch.zeros(steps)], dim=-1)[None]
    out = _collision(ego, {"lead_path": lead, "ego_radius_m": 1.0, "obs_radius_m": 1.0})
    assert float(out) == 0.0, "a 2 s-gap follower was flagged as a collision"


def test_a_closing_lead_is_flagged_so_the_branch_is_not_simply_inert():
    """The same-breath positive control: the branch must be able to fire."""
    steps = 12
    t = torch.arange(steps, dtype=torch.float32) * 0.5
    ego = torch.stack([10.0 * t, torch.zeros(steps)], dim=-1)[None]
    lead = torch.stack([20.0 - 0.0 * t, torch.zeros(steps)], dim=-1)[None]  # stopped
    out = _collision(ego, {"lead_path": lead, "ego_radius_m": 1.0, "obs_radius_m": 1.0})
    assert float(out) == -1.0, "the ego drives into a stopped lead and is not flagged"


def test_absent_obstacles_are_still_zero_not_a_free_pass_disguised_as_safety():
    traj = torch.zeros(3, 8, 2)
    assert torch.equal(_collision(traj, {}), torch.zeros(3))
