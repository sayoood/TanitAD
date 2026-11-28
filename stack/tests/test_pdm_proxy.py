"""Pins ``tanitad.rl.pdm_proxy`` on hand-built scenes whose answers are known in advance,
and requires each deliberate regression (a plausible port error) to change an answer."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tanitad.refs import refc_sampler as rs
from tanitad.rl import pdm_proxy as P

CFG = P.PROXY
T0 = CFG.n_ticks + 1


def _straight(v0, accel=0.0, m=1, ticks=T0):
    """Rear-axle states along +x from the origin under constant accel (floored at v = 0)."""
    st = torch.zeros(m, ticks, 4)
    x, v = 0.0, float(v0)
    for k in range(ticks):
        st[:, k, 0], st[:, k, 3] = x, v
        x += v * CFG.dt
        v = max(0.0, v + accel * CFG.dt)
    return st


def _tracks(xy_fn, n=T0 + 9, lw=(4.5, 2.0), static=False, yaw=0.0):
    frames = [[{"track_id": "a", "cx": xy_fn(k)[0], "cy": xy_fn(k)[1], "yaw": yaw,
                "l": lw[0], "w": lw[1], "cls": "protruding_object" if static else "automobile"}]
              for k in range(n)]
    poses = torch.zeros(n, 4)            # ego frame at every tick == world == ego@t0
    return P.AgentTracks.from_frames(frames, poses)


# --------------------------------------------------------------------------- #
def test_boxes_overlap_cases():
    a = P.box_corners(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), 2.0, 2.0)
    b = P.box_corners(torch.tensor(1.5), torch.tensor(0.0), torch.tensor(0.0), 2.0, 2.0)
    c = P.box_corners(torch.tensor(3.1), torch.tensor(0.0), torch.tensor(0.0), 2.0, 2.0)
    d = P.box_corners(torch.tensor(2.3), torch.tensor(0.0), torch.tensor(math.pi / 4), 2.0, 2.0)
    assert bool(P.boxes_overlap(a, b)) and not bool(P.boxes_overlap(a, c))
    # diamond centred 2.3 m away reaches 2.3 - sqrt(2) = 0.886 < 1 -> overlaps
    assert bool(P.boxes_overlap(a, d))
    far = P.box_corners(torch.tensor(2.5), torch.tensor(0.0), torch.tensor(math.pi / 4), 2.0, 2.0)
    assert not bool(P.boxes_overlap(a, far))                # 2.5 - 1.414 = 1.086 > 1


def test_controls_states_agree_with_roll_controls_at_the_slots():
    hz = (5, 10, 15, 20, 30, 40, 50, 60)
    g = torch.Generator().manual_seed(0)
    u = torch.stack([torch.randn(2, 6, 8, generator=g) * 1.5,
                     torch.randn(2, 6, 8, generator=g) * 1.0], dim=-1)
    v0 = torch.tensor([9.0, 2.0])
    cfg60 = P.ProxyConfig(n_ticks=60)
    st = P.ego_states_from_controls(u, v0, hz, cfg=cfg60)
    ref = rs.roll_controls(u, v0, hz, control_units="alat")
    got = st[:, :, torch.tensor(hz), :2]                      # tick k is index k (t0 prepended)
    assert torch.equal(got, ref)


def test_human_states_are_in_the_t0_ego_frame():
    th = 0.7
    k = torch.arange(1, 41, dtype=torch.float32)
    fut = torch.stack([3 + 10 * 0.1 * k * math.cos(th), -2 + 10 * 0.1 * k * math.sin(th),
                       torch.full_like(k, th), torch.full_like(k, 10.0)], dim=-1)[None]
    st = P.ego_states_from_poses(torch.tensor([[3.0, -2.0, th, 10.0]]), fut)
    assert torch.allclose(st[0, :, 1], torch.zeros(T0), atol=1e-4)
    assert torch.allclose(st[0, -1, 0], torch.tensor(40.0), atol=1e-3)
    assert torch.allclose(st[0, :, 2], torch.zeros(T0), atol=1e-5)


def test_agent_frames_compose_ego_at_k_to_ego_at_t0():
    poses = torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 5.0, math.pi / 2, 0.0]])
    frames = [[], [{"track_id": 7, "cx": 2.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0,
                    "cls": "automobile"}]]
    tr = P.AgentTracks.from_frames(frames, poses)
    assert torch.allclose(tr.xy[1, 0], torch.tensor([10.0, 7.0]), atol=1e-5)
    assert abs(float(tr.yaw[1, 0]) - math.pi / 2) < 1e-5
    assert bool(tr.valid[1, 0]) and not bool(tr.valid[0, 0])


# --------------------------------------------------------------------------- #
# NC
# --------------------------------------------------------------------------- #
def test_nc_drive_into_a_stopped_car_is_at_fault_zero():
    ego = _straight(10.0)
    tr = _tracks(lambda k: (25.0, 0.0))                       # parked 25 m ahead
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 0.0


def test_nc_static_object_is_half():
    ego = _straight(10.0)
    tr = _tracks(lambda k: (25.0, 0.0), static=True)
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 0.5


def test_nc_rear_ended_while_stopped_is_not_at_fault():
    ego = _straight(0.0)
    tr = _tracks(lambda k: (-20.0 + 6.0 * 0.1 * k, 0.0))      # closes from behind at 6 m/s
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 1.0


def test_nc_initial_overlap_is_ignored():
    ego = _straight(0.0)
    tr = _tracks(lambda k: (1.0, 0.0))
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 1.0


def test_nc_a_track_already_in_contact_at_t0_is_ignored_even_while_moving():
    """NAVSIM seeds its ignore list with the observation's initial collisions. A car overlapping
    the ego's front at t0 and moving with it would otherwise be an at-fault front contact at
    tick 0. (Added after mutant P04 survived the 2026-09-15 sweep.)"""
    front = CFG.rear_axle_to_center + CFG.ego_length / 2
    ego = _straight(10.0)
    tr = _tracks(lambda k: (front + 2.0 + 10.0 * 0.1 * k, 0.0))   # rear edge 0.25 m inside
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 1.0


def test_nc_moving_rear_contact_from_behind_while_ego_moves_is_not_at_fault():
    ego = _straight(5.0)
    tr = _tracks(lambda k: (-15.0 + 12.0 * 0.1 * k, 0.0))      # faster car hits ego's rear
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 1.0


def test_REGRESSION_any_overlap_counts_turns_a_rear_end_into_a_fault(monkeypatch):
    ego = _straight(5.0)
    tr = _tracks(lambda k: (-15.0 + 12.0 * 0.1 * k, 0.0))
    real = P._rel_in_ego
    monkeypatch.setattr(P, "_rel_in_ego", lambda s, p: real(s, p).abs())   # 'behind' never true
    monkeypatch.setattr(P, "_segment_hits_box", lambda a, b, box: torch.ones(
        torch.broadcast_shapes(a.shape[:-1], box.shape[:-2]), dtype=torch.bool))
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 0.0


def test_REGRESSION_without_the_rear_axle_offset_a_near_miss_disappears():
    ego = _straight(0.1)                                      # creeps 0.4 m in 4 s
    rear_edge = 4.3                                           # agent rear 4.3 m ahead of origin
    tr = _tracks(lambda k: (rear_edge + 2.25, 0.0))           # 4.5 m long car, stopped
    with_offset = float(P.no_at_fault_collision(ego, tr)[0])
    no_offset = float(P.no_at_fault_collision(ego, tr, P.ProxyConfig(rear_axle_to_center=0.0))[0])
    assert with_offset == 0.0 and no_offset == 1.0


# --------------------------------------------------------------------------- #
# TTC
# --------------------------------------------------------------------------- #
def test_ttc_hard_stop_short_of_a_car_still_fails_ttc_but_not_nc():
    ego = _straight(6.0, accel=-9.0)                         # stops after ~2 m
    front = CFG.rear_axle_to_center + CFG.ego_length / 2
    tr = _tracks(lambda k: (front + 3.0 + 2.25, 0.0))        # 3 m gap, stopped car
    assert float(P.no_at_fault_collision(ego, tr)[0]) == 1.0
    assert float(P.ttc_within_bound(ego, tr)[0]) == 0.0


def test_ttc_ignores_a_track_outside_the_forward_cone():
    """A parked bus lying ACROSS the adjacent area: its box reaches the ego's projected footprint,
    but its centre sits >= 36.9 deg off the heading from every ego tick, so it is not 'ahead'.
    (Added after mutant P05 — the cone dropped — survived the 2026-09-15 sweep.)"""
    ego = _straight(10.0)
    frames_n = T0 + 9
    tr = _tracks(lambda k: (6.0, 4.5), n=frames_n, lw=(12.0, 2.0), static=False, yaw=math.pi / 2)
    assert float(P.ttc_within_bound(ego, tr)[0]) == 1.0
    no_cone = P.ProxyConfig(ahead_cone_deg=90.0)                  # cos(90) = 0: everything ahead
    assert float(P.ttc_within_bound(ego, tr, no_cone)[0]) == 0.0


def test_ttc_same_speed_follow_passes_and_a_stopped_ego_passes():
    front = CFG.rear_axle_to_center + CFG.ego_length / 2
    ego = _straight(10.0)
    tr = _tracks(lambda k: (front + 5.0 + 2.25 + 10.0 * 0.1 * k, 0.0))
    assert float(P.ttc_within_bound(ego, tr)[0]) == 1.0
    tr2 = _tracks(lambda k: (front + 1.0 + 2.25, 0.0))
    assert float(P.ttc_within_bound(_straight(0.0), tr2)[0]) == 1.0


# --------------------------------------------------------------------------- #
# comfort
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order,deriv", [(2, 1), (3, 2), (2, 2)])
def test_poly_deriv_is_scipy_savgol_with_a_full_window(order, deriv):
    from scipy.signal import savgol_filter
    rng = np.random.default_rng(1)
    y = rng.normal(size=(3, 41)).cumsum(axis=1)
    ref = savgol_filter(y, window_length=41, polyorder=order, deriv=deriv, delta=0.1, axis=-1)
    got = P._poly_deriv(torch.tensor(y), 0.1, order, deriv).numpy()
    assert np.allclose(got, ref, atol=1e-8, rtol=1e-6)


def test_comfort_constant_speed_passes_hard_brake_fails():
    assert float(P.comfort(_straight(12.0))[0]) == 1.0
    assert float(P.comfort(_straight(20.0, accel=-6.0))[0]) == 0.0


def test_REGRESSION_raw_finite_difference_jerk_would_fail_a_slot_boundary():
    """Piecewise-constant slot controls step the acceleration; NAVSIM's full-window polynomial
    smooths that step, a 0.1 s finite difference turns it into a jerk spike."""
    hz = (5, 10, 15, 20, 30, 40, 50, 60)
    u = torch.zeros(1, 1, 8, 2)
    u[0, 0, 1, 0] = 1.0                                       # +1 m/s^2 in slot 2 only
    st = P.ego_states_from_controls(u, torch.tensor([10.0]), hz)[0]
    v = st[0, :, 3]
    fd_jerk = ((v[2:] - 2 * v[1:-1] + v[:-2]) / 0.1 ** 2).abs().max()
    assert float(fd_jerk) > CFG.max_abs_lon_jerk
    assert float(P.comfort(st)[0]) == 1.0


# --------------------------------------------------------------------------- #
# EP, PDMS, DAC
# --------------------------------------------------------------------------- #
def _route(length=80.0):
    return torch.stack([torch.linspace(0, length, 161), torch.zeros(161)], dim=-1)


def test_ep_pairwise_against_the_human():
    far = _tracks(lambda k: (500.0, 50.0))                    # no interactions
    human = _straight(10.0)[0]                                # 40 m
    cands = torch.cat([_straight(5.0), _straight(11.25), _straight(0.5)])
    out = P.score_candidates(cands, human, far, _route())
    ep = out["ep"]
    assert abs(float(ep[0]) - 0.5) < 1e-3                     # 20 m / 40 m
    assert float(ep[1]) == pytest.approx(1.0, abs=1e-4)       # 45 m >= human
    assert abs(float(ep[2]) - 2.0 / 40.0) < 1e-3              # 2 m vs 40 m: max > 5 -> ratio
    assert out["human"]["ep"] == pytest.approx(1.0, abs=1e-4)
    both_short = P.score_candidates(_straight(0.5), _straight(0.25)[0], far, _route())
    assert float(both_short["ep"][0]) == 1.0                  # max <= 5 m and no gate -> 1


def test_REGRESSION_normalising_ep_by_the_best_candidate_changes_it():
    far = _tracks(lambda k: (500.0, 50.0))
    human = _straight(10.0)[0]
    cands = torch.cat([_straight(5.0), _straight(15.0)])
    raw = P.ego_progress(cands, _route())
    mutant = raw / raw.max()                                   # best-candidate reference
    ours = P.score_candidates(cands, human, far, _route())["ep"]
    assert not torch.allclose(mutant[0], ours[0], atol=1e-2)


def test_collision_gates_pdms_and_flags_the_constraint():
    human = _straight(10.0)[0]
    tr = _tracks(lambda k: (25.0, 3.5))                       # parked in the next lane
    cands = torch.stack([_straight(10.0)[0],
                         torch.cat([_straight(10.0)[0, :, :1],
                                    torch.full((T0, 1), 3.5), torch.zeros(T0, 1),
                                    torch.full((T0, 1), 10.0)], dim=-1)])
    out = P.score_candidates(cands, human, tr, _route())
    assert float(out["nc"][0]) == 1.0 and float(out["nc"][1]) == 0.0
    assert float(out["pdms"][1]) == 0.0 and bool(out["constraint_fail"][1])
    assert not bool(out["constraint_fail"][0])


def test_pdms_formula_literal():
    one = torch.ones(1)
    assert float(P.pdms(one, one, one * 0.5, one, one * 0)) == pytest.approx((2.5 + 5) / 12)


def test_dac_from_drivable_grid():
    ego = _straight(10.0)
    frac = torch.ones(120, 64)
    seen = torch.ones(120, 64, dtype=torch.bool)
    assert float(P.dac_from_drivable(ego, frac, seen)[0]) == 1.0
    # front-left corner: x = rear axle + 1.461 + 2.588, y = +1.1485 -> at tick 26 it sits in
    # cell ix = floor(30.049 / 0.5) = 60, iy = floor((1.1485 + 16) / 0.5) = 34
    bad = frac.clone()
    bad[60, 34] = 0.0
    assert float(P.dac_from_drivable(ego, bad, seen)[0]) == 0.0
    unseen = seen.clone()
    unseen[60, 34] = False
    assert float(P.dac_from_drivable(ego, bad, unseen)[0]) == 1.0
    # ⚠️ CORNER-ONLY (the NAVSIM ego-area rule as read; SPEC §13.6 UNVERIFIED): a non-drivable
    # cell under the centreline, between the corners, is NOT seen -> an upper bound on compliance
    centre = frac.clone()
    centre[60, 32] = 0.0
    assert float(P.dac_from_drivable(ego, centre, seen)[0]) == 1.0
