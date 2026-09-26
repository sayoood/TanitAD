"""refcv7 oracle — every expectation is a LITERAL derived by hand from the geometry,
never an expression over the code under test."""
import torch

from tanitad.refs.refcv7_oracle import (OracleConfig, aggregate, oracle_subscores,
                                        path_kinematics)

SLOT_T = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])


def straight(v: float, y: float = 0.0) -> torch.Tensor:
    """Constant-speed straight path along +x at lateral offset y: [1, 1, 8, 2]."""
    x = SLOT_T * v
    return torch.stack([x, torch.full_like(x, y)], -1)[None, None]


def full_raster(drivable_rows=None, drivable_cols=None):
    d = torch.ones(1, 120, 64)
    if drivable_cols is not None:
        d = torch.zeros(1, 120, 64)
        d[:, :, drivable_cols[0]:drivable_cols[1]] = 1.0
    return d, torch.ones(1, 120, 64, dtype=torch.bool)


def run(traj, **kw):
    v0 = torch.tensor([10.0])
    gt = straight(10.0)[:, 0]
    return oracle_subscores(traj, SLOT_T, v0, gt, **kw)


def test_kinematics_of_a_constant_speed_line_are_exact():
    k = path_kinematics(straight(10.0)[0], SLOT_T, torch.tensor([10.0]))
    assert torch.allclose(k["v"], torch.full((1, 8), 10.0))
    assert torch.allclose(k["a_lon"], torch.zeros(1, 8), atol=1e-5)
    assert torch.allclose(k["yaw_rate"], torch.zeros(1, 8), atol=1e-6)


def test_clear_road_straight_line_passes_every_gate():
    d, s = full_raster()
    o = run(straight(10.0), drivable=d, seen=s, max_speed_ms=torch.tensor([13.89]))
    for k in ("nc", "dac", "ttc", "ep", "comf", "spd"):
        assert float(o[k]) == 1.0, k
    assert not bool(o["nc_mask"])       # no agents -> NC abstains, it does not pass silently


def test_agent_on_the_path_is_a_collision():
    # ego at slot 4 (t = 3 s) is at x = 30; a 4.5 x 2 car parked there, static
    ag = torch.tensor([30.0, 0.0, 4.5, 2.0, 0.0]).expand(1, 1, 8, 5).clone()
    o = run(straight(10.0), agents=ag, agents_valid=torch.ones(1, 1, 8, dtype=torch.bool))
    assert float(o["nc"]) == 0.0 and bool(o["nc_mask"])


def test_agent_ten_metres_to_the_side_is_not_a_collision():
    ag = torch.tensor([30.0, 10.0, 4.5, 2.0, 0.0]).expand(1, 1, 8, 5).clone()
    o = run(straight(10.0), agents=ag, agents_valid=torch.ones(1, 1, 8, dtype=torch.bool))
    assert float(o["nc"]) == 1.0 and float(o["ttc"]) == 1.0


def test_agent_just_ahead_fails_ttc_but_not_nc():
    # final ego slot x = 60 at 10 m/s; ego front at 62.45; TTC extends it by 10 m.
    # an agent centred at x = 67 (rear at 64.75) is clear of the box but inside the 1 s reach
    ag = torch.zeros(1, 1, 8, 5)
    ag[..., 0] = 67.0; ag[..., 2] = 4.5; ag[..., 3] = 2.0
    valid = torch.zeros(1, 1, 8, dtype=torch.bool); valid[..., -1] = True
    o = run(straight(10.0), agents=ag, agents_valid=valid)
    assert float(o["nc"]) == 1.0
    assert float(o["ttc"]) == 0.0


def test_invalid_agent_slots_are_ignored():
    ag = torch.tensor([30.0, 0.0, 4.5, 2.0, 0.0]).expand(1, 1, 8, 5).clone()
    o = run(straight(10.0), agents=ag, agents_valid=torch.zeros(1, 1, 8, dtype=torch.bool))
    assert float(o["nc"]) == 1.0


def test_dac_lane_corridor():
    # drivable only for y in [-4, 4): columns (y + 16) / 0.5 -> 24 .. 40
    d, s = full_raster(drivable_cols=(24, 40))
    assert float(run(straight(8.0, y=0.0), drivable=d, seen=s)["dac"]) == 1.0
    assert float(run(straight(8.0, y=6.0), drivable=d, seen=s)["dac"]) == 0.0


def test_dac_abstains_on_unseen_ground_and_outside_the_raster():
    d, _ = full_raster(drivable_cols=(24, 40))
    unseen = torch.zeros(1, 120, 64, dtype=torch.bool)
    o = run(straight(8.0, y=6.0), drivable=d, seen=unseen)
    assert float(o["dac"]) == 1.0 and not bool(o["dac_mask"])   # abstain, never a pass


def test_progress_is_the_length_ratio_to_the_human():
    o = run(straight(5.0))               # half the human's 10 m/s
    assert abs(float(o["ep"]) - 0.5) < 1e-5


def test_comfort_fails_a_hard_brake():
    # 10 m/s then stop within 0.5 s: -20 m/s^2 << MIN_LON_ACCEL -4.05
    traj = torch.zeros(1, 1, 8, 2)
    traj[..., 0] = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    assert float(run(traj)["comf"]) == 0.0


def test_set_speed_compliance():
    assert float(run(straight(15.0), max_speed_ms=torch.tensor([13.89]))["spd"]) == 0.0
    assert float(run(straight(15.0), max_speed_ms=torch.tensor([27.78]))["spd"]) == 1.0


def test_aggregate_zeroes_on_any_gate_and_is_pdms_weighted():
    one = torch.ones(1, 1)
    sub = {"nc": one, "dac": one, "spd": one, "ttc": one, "ep": one * 0.5, "comf": one}
    assert abs(float(aggregate(sub)) - (5 + 2.5 + 2) / 12) < 1e-6
    sub["dac"] = one * 0
    assert float(aggregate(sub)) == 0.0


def _left_curve():
    th = SLOT_T.clone() * 10.0 / 50.0              # R = 50 m at 10 m/s: clearly left
    return torch.stack([50 * torch.sin(th), 50 * (1 - torch.cos(th))], -1)[None, None]


def test_nav_target_is_timing_aware_not_side_only():
    """Told LEFT: turning is right only if the HUMAN turned within the horizon."""
    from tanitad.refs.refb import NAV_COMMANDS
    L = torch.tensor([NAV_COMMANDS.index("left")])
    v0 = torch.tensor([10.0])
    left, straight_p = _left_curve(), straight(10.0)

    def nav(cand, gt):
        return float(oracle_subscores(cand, SLOT_T, v0, gt[:, 0], nav_cmd=L,
                                      nav_tau_rad=0.2)["nav"])
    assert nav(left, left) == 1.0            # due, and turns
    assert nav(straight_p, straight_p) == 1.0  # not yet due, and does NOT turn early
    assert nav(left, straight_p) == 0.0      # turns EARLY: the defect a side-only label rewards
    assert nav(straight_p, left) == 0.0      # due, and misses the turn


def test_nav_abstains_on_follow():
    from tanitad.refs.refb import NAV_COMMANDS
    F = torch.tensor([NAV_COMMANDS.index("follow")])
    o = run(_left_curve(), nav_cmd=F, nav_tau_rad=0.2)
    assert not bool(o["nav_mask"])


def test_aggregate_nav_gate_only_where_informative():
    one = torch.ones(1, 1)
    sub = {"nc": one, "dac": one, "spd": one, "ttc": one, "ep": one, "comf": one,
           "nav": one * 0}
    assert float(aggregate(sub, nav_informative=torch.tensor([False]))) == 1.0
    assert float(aggregate(sub, nav_informative=torch.tensor([True]))) == 0.0
