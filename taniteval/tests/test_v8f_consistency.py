"""v8F-mini CONSISTENCY / nav-only baseline / macro-F1: hand-made cases with known answers, boundary values
included. Pure numpy; no data. Run: PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q taniteval/tests/test_v8f_consistency.py

Every rule has (i) a case that must fire, (ii) a control that must NOT fire, (iii) the boundary on both sides.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taniteval import v8f_consistency as C  # noqa: E402

DT, K = C.DT, C.K


def straight(v, n=1):
    """[n, 30, 2] straight constant-speed plan."""
    x = v * DT * np.arange(1, K + 1)
    return np.tile(np.stack([x, np.zeros(K)], -1), (n, 1, 1))


def from_speed_profile(vprof):
    """Straight plan from per-interval speeds [30] -> [1, 30, 2]."""
    x = np.cumsum(np.asarray(vprof, float) * DT)
    return np.stack([x, np.zeros(K)], -1)[None]


def no_c(n=1):
    nan = np.full(n, np.nan)
    return {"limit_ms": nan.copy(), "v_hi_ms": nan.copy(), "stop_active": np.zeros(n, bool), "d_stop": nan.copy(),
            "t_stop": nan.copy(), "turn_side": np.zeros(n, np.int64), "turn_t_start": nan.copy(),
            "turn_t_end": nan.copy(), "lead_tg_min": nan.copy(), "lead_gap_h": np.full((n, K), np.nan),
            "s_h": np.zeros((n, K))}


# ----------------------------------------------------------------------------------------------- SPEED
def test_speed_fires_and_boundary():
    plan = straight(10.0)
    vmax = C.plan_kinematics(plan)["v"][:, 10:].max()
    c = no_c()
    c["limit_ms"][:] = 12.0
    c["v_hi_ms"][:] = 9.0                                   # cap = min(12, 9) = 9
    V = C.violations(plan, c)
    assert V["SPEED"]["active"][0] and V["SPEED"]["violated"][0]
    c["v_hi_ms"][:] = vmax - 0.5                            # exactly on the tolerance: NOT a violation
    assert not C.violations(plan, c)["SPEED"]["violated"][0]
    c["v_hi_ms"][:] = vmax - 0.5 - 1e-6                     # just past it
    assert C.violations(plan, c)["SPEED"]["violated"][0]


def test_speed_cap_is_the_min_of_limit_and_band():
    plan = straight(10.0)
    c = no_c()
    c["limit_ms"][:] = 8.0                                  # limit alone binds
    c["v_hi_ms"][:] = 30.0
    assert C.violations(plan, c)["SPEED"]["violated"][0]
    c["limit_ms"][:] = 30.0                                 # band alone binds
    c["v_hi_ms"][:] = 8.0
    assert C.violations(plan, c)["SPEED"]["violated"][0]
    c["limit_ms"][:] = 30.0                                 # neither binds
    c["v_hi_ms"][:] = 30.0
    assert not C.violations(plan, c)["SPEED"]["violated"][0]


def test_speed_inactive_without_cap():
    V = C.violations(straight(30.0), no_c())
    assert not V["SPEED"]["active"][0] and not V["SPEED"]["violated"][0]
    assert not V["any_active"][0] and not V["any_violated"][0]


def test_speed_is_read_on_the_band_overlap_only():
    """I2: a braking plan that starts fast must not violate a cap defined on [NOW+2, NOW+8]."""
    prof = np.where(np.arange(K) < 10, 15.0, 5.0)           # fast for the first 2 s, then slow
    c = no_c()
    c["limit_ms"][:] = 8.0
    assert not C.violations(from_speed_profile(prof), c)["SPEED"]["violated"][0]
    prof2 = np.where(np.arange(K) < 10, 5.0, 15.0)          # control: fast AFTER 2 s
    assert C.violations(from_speed_profile(prof2), c)["SPEED"]["violated"][0]


# ----------------------------------------------------------------------------------------------- STOP
def stop_c(d, t=4.0):
    c = no_c()
    c["stop_active"][:] = True
    c["d_stop"][:] = d
    c["t_stop"][:] = t
    return c


def test_stop_fires_when_still_moving_at_the_stop_distance():
    plan = straight(5.0)                                    # 5 m/s the whole way, d_stop = 10 m
    V = C.violations(plan, stop_c(10.0))
    assert V["STOP"]["active"][0] and V["STOP"]["violated"][0]
    assert V["STOP"]["v_at_stop"][0] == pytest.approx(5.0)


def test_stop_ok_when_the_plan_stops_there():
    # decelerate to 0 exactly at arc 10 m, then stay
    prof = np.concatenate([np.array([6, 5, 4, 3, 2, 1, .8, .6, .4, .2, 0.0]), np.zeros(19)])
    plan = from_speed_profile(prof)
    d = float(C.plan_kinematics(plan)["s"][0, -1])          # the plan's own total arc
    V = C.violations(plan, stop_c(d - 0.2))                 # stop point 0.2 m before its end: speed there ~0, pass < 2
    assert V["STOP"]["active"][0] and not V["STOP"]["violated"][0]


def test_stop_pass_boundary_2m():
    prof = np.zeros(K)
    prof[:5] = 2.0                                          # 5 intervals at 2 m/s then stopped: total arc 2.0 m
    plan = from_speed_profile(prof)
    total = float(C.plan_kinematics(plan)["s"][0, -1])
    assert total == pytest.approx(2.0)
    # stop point 2.0 m behind the plan end -> passes by exactly 2.0: NOT a violation of the pass clause
    # (speed at d_stop = the interval holding arc 0 -> 2 m/s is > 1.0 -> that clause would fire; use d > total - eps)
    c = stop_c(0.0 + 1e-9)
    # d_stop ~ 0 : speed at that arc is the first interval (2 m/s) -> violation by the SPEED clause
    assert C.violations(plan, c)["STOP"]["violated"][0]
    # now a stopped-at-start plan (v=0 everywhere, total 0): passes by 0 -> fine
    z = from_speed_profile(np.zeros(K))
    assert not C.violations(z, stop_c(0.0))["STOP"]["violated"][0]
    # plan that creeps 0.5 m/s: total 3.0 m; d_stop = 1.0 -> passes by 2.0 exactly (not > 2.0), speed 0.5 <= 1.0
    cr = straight(0.5)
    assert float(C.plan_kinematics(cr)["s"][0, -1]) == pytest.approx(3.0)
    assert not C.violations(cr, stop_c(1.0))["STOP"]["violated"][0]
    assert C.violations(cr, stop_c(1.0 - 1e-3))["STOP"]["violated"][0]       # passes by 2.001 > 2.0


def test_stop_inactive_beyond_horizon_or_short():
    plan = straight(5.0)
    assert not C.violations(plan, stop_c(10.0, t=6.5))["STOP"]["active"][0]  # stop is after the plan ends (I3)
    assert C.violations(plan, stop_c(10.0, t=6.0))["STOP"]["active"][0]      # boundary: <= 6.0 is active
    # plan stops short of d_stop: never reaches -> no speed violation
    short = straight(1.0)                                    # total arc 6 m
    V = C.violations(short, stop_c(20.0))
    assert V["STOP"]["active"][0] and not V["STOP"]["violated"][0]


def test_stop_speed_boundary_1ms():
    # isolate the speed clause: d_stop close to plan end so pass clause cannot fire
    p1, p2 = straight(1.0), straight(1.0 + 1e-3)
    assert not C.violations(p1, stop_c(5.9))["STOP"]["violated"][0]
    assert C.violations(p2, stop_c(5.9))["STOP"]["violated"][0]


# ----------------------------------------------------------------------------------------------- TURN
def arc_plan(v, yaw_rate):
    """Constant-speed constant-yaw-rate plan (left +)."""
    th = np.cumsum(np.full(K, yaw_rate * DT))
    x = np.cumsum(v * DT * np.cos(th))
    y = np.cumsum(v * DT * np.sin(th))
    return np.stack([x, y], -1)[None]


def turn_c(side, t0=1.0, t1=5.0):
    c = no_c()
    c["turn_side"][:] = side
    c["turn_t_start"][:] = t0
    c["turn_t_end"][:] = t1
    return c


def test_turn_sign_fires_and_controls():
    left = arc_plan(8.0, 0.3)
    assert not C.violations(left, turn_c(+1))["TURN"]["violated"][0]
    assert C.violations(left, turn_c(-1))["TURN"]["violated"][0]
    right = arc_plan(8.0, -0.3)
    assert not C.violations(right, turn_c(-1))["TURN"]["violated"][0]
    assert C.violations(right, turn_c(+1))["TURN"]["violated"][0]
    V = C.violations(straight(8.0), turn_c(+1))              # no turn at all during an active TURN goal -> violation
    assert V["TURN"]["active"][0] and V["TURN"]["violated"][0]


def test_turn_deadband_boundary_1deg():
    # heading change over [1, 5] s = yaw_rate * 4 s ; psi is read at the INTERVAL tangent, so compute it
    for dpsi_deg, expect_violation in ((0.9, True), (1.1, False)):
        w = math.radians(dpsi_deg) / 4.0
        V = C.violations(arc_plan(8.0, w), turn_c(+1))
        assert V["TURN"]["dpsi_deg"][0] == pytest.approx(dpsi_deg, abs=0.05)
        assert V["TURN"]["violated"][0] == expect_violation


def test_turn_overlap_rule():
    left = arc_plan(8.0, 0.3)
    assert not C.violations(left, turn_c(-1, t0=5.5, t1=8.0))["TURN"]["active"][0]      # 0.5 s overlap < 1 s
    assert C.violations(left, turn_c(-1, t0=5.0, t1=8.0))["TURN"]["active"][0]          # exactly 1.0 s overlap
    assert C.violations(left, turn_c(-1, t0=-3.0, t1=2.0))["TURN"]["active"][0]          # in progress at NOW
    assert not C.violations(left, no_c())["TURN"]["active"][0]                          # no goal -> inactive


# ----------------------------------------------------------------------------------------------- LEAD
def lead_c(plan_like, gap0=20.0, tg_min=2.0):
    c = no_c()
    c["lead_tg_min"][:] = tg_min
    c["lead_gap_h"][:] = gap0                                # lead 20 m ahead of the HUMAN at every knot
    kin = C.plan_kinematics(plan_like)
    c["s_h"][:] = kin["s"][:, 1:]                            # human arc == the reference plan's arc
    return c


def test_lead_human_plan_is_consistent_and_a_faster_plan_is_not():
    human = straight(10.0)
    c = lead_c(human, gap0=20.0, tg_min=2.0)                 # tg = 20/10 = 2.0 s everywhere
    assert not C.violations(human, c)["LEAD"]["violated"][0]
    V = C.violations(human, c)
    assert V["LEAD"]["tg_min_plan"][0] == pytest.approx(2.0)
    assert C.violations(straight(12.0), c)["LEAD"]["violated"][0]   # closes the gap -> tg drops below 1.8 s


def test_lead_tolerance_boundary_0p2s():
    human = straight(10.0)
    # the SAME plan, but a constraint 0.2 s above its own min time gap: exactly on the tolerance -> not a violation
    c = lead_c(human, gap0=20.0, tg_min=2.2)
    assert not C.violations(human, c)["LEAD"]["violated"][0]
    c["lead_tg_min"][:] = 2.2 + 1e-6
    assert C.violations(human, c)["LEAD"]["violated"][0]


def test_lead_inactive_without_data():
    c = lead_c(straight(10.0))
    c["lead_gap_h"][:] = np.nan
    assert not C.violations(straight(30.0), c)["LEAD"]["active"][0]
    c2 = lead_c(straight(10.0))
    c2["lead_tg_min"][:] = np.nan
    assert not C.violations(straight(30.0), c2)["LEAD"]["active"][0]


# ----------------------------------------------------------------------------------------------- aggregate
def test_consistency_rates_and_gt_control():
    n = 4
    plans = np.concatenate([straight(10.0, 2), straight(15.0, 2)])
    c = no_c(n)
    c["limit_ms"][:] = 12.0
    res = C.consistency(plans, c)
    assert res["any"]["n_violated"] == 2 and res["any"]["rate_all"] == pytest.approx(0.5)
    assert res["SPEED"]["rate_over_active"] == pytest.approx(0.5)
    assert res["TURN"]["n_active"] == 0 and res["TURN"]["rate_over_active"] == 0.0
    assert res["any"]["n_active"] == 4
    # per-window flags are exposed for the paired bootstrap
    assert res["_per_window"]["any_violated"].tolist() == [False, False, True, True]


def test_consistency_with_cluster_ci():
    plans = np.concatenate([straight(10.0, 6), straight(15.0, 6)])
    c = no_c(12)
    c["limit_ms"][:] = 12.0
    eid = np.repeat(np.arange(6), 2)
    res = C.consistency(plans, c, eid=eid, n_boot=200)
    assert "ci" in res["any"] and res["any"]["ci"]["mean"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------------- nav-only baseline
def test_nav_baseline_straight_extrapolates_the_p10_path():
    ck = C.nav_only_checkpoints(straight(10.0), np.array([0]), np.array([np.nan]))
    assert ck.shape == (1, 3, 3)
    np.testing.assert_allclose(ck[0, :, 0], [30.0, 50.0, 80.0], atol=1e-9)    # 30 m is INSIDE the 60 m path; 80 beyond
    np.testing.assert_allclose(ck[0, :, 1:], 0.0, atol=1e-9)


def test_nav_baseline_curved_p10_keeps_its_tangent_beyond_the_end():
    plan = arc_plan(10.0, 0.1)
    ck = C.nav_only_checkpoints(plan, np.array([0]), np.array([np.nan]))
    end, th_end = plan[0, -1], math.atan2(*(plan[0, -1] - plan[0, -2])[::-1])
    assert ck[0, 2, 2] == pytest.approx(th_end, abs=1e-9)
    s_end = float(C.plan_kinematics(plan)["s"][0, -1])
    exp = end + (80.0 - s_end) * np.array([math.cos(th_end), math.sin(th_end)])
    np.testing.assert_allclose(ck[0, 2, :2], exp, atol=1e-6)


def test_nav_baseline_turn_left_and_right_known_values():
    R, d = 12.0, 40.0
    for tok, sg in ((1, +1.0), (2, -1.0)):
        ck = C.nav_only_checkpoints(straight(10.0), np.array([tok]), np.array([d]))[0]
        np.testing.assert_allclose(ck[0], [30.0, 0.0, 0.0], atol=1e-9)                  # before the turn: straight
        th = (50.0 - d) / R
        np.testing.assert_allclose(ck[1], [d + R * math.sin(th), sg * R * (1 - math.cos(th)), sg * th], atol=1e-9)
        # 80 m: u = 40 > arc length 18.85 -> after the exit, straight along the new heading (90 deg)
        np.testing.assert_allclose(ck[2], [d + R, sg * (R + (40.0 - R * math.pi / 2)), sg * math.pi / 2], atol=1e-6)


def test_nav_baseline_unknown_distance_injects_no_turn():
    ck = C.nav_only_checkpoints(straight(10.0), np.array([1]), np.array([np.nan]))
    np.testing.assert_allclose(ck[0, :, 1], 0.0, atol=1e-9)


def test_checkpoint_errors_mask_and_wrap():
    pred = np.array([[[3.0, 4.0, math.radians(179.0)]]])
    gt = np.array([[[0.0, 0.0, math.radians(-179.0)]]])
    e = C.checkpoint_errors(pred, gt, np.array([[True]]))
    assert e["pos_err_m"][0, 0] == pytest.approx(5.0) and e["head_err_deg"][0, 0] == pytest.approx(2.0, abs=1e-6)
    assert np.isnan(C.checkpoint_errors(pred, gt, np.array([[False]]))["pos_err_m"][0, 0])


# ----------------------------------------------------------------------------------------------- macro-F1
def test_macro_f1_hand_case_and_ignore():
    yt = np.array([0, 0, 1, 1, 2, -100])
    yp = np.array([0, 1, 1, 1, 2, 0])
    r = C.macro_f1(yt, yp, 4)
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 1.0) / 3)               # class 3 absent -> excluded
    assert r["n"] == 5 and r["confusion"][0, 1] == 1 and np.isnan(r["per_class_f1"][3])


def test_majority_baseline():
    y = np.array([0, 0, 0, 1, -100])
    r = C.majority_baseline(y, 2)
    assert r["majority_class"] == 0 and r["macro_f1"] == pytest.approx((2 * 3 / (2 * 3 + 1 + 0) + 0.0) / 2)


def test_constraints_from_v9_bits():
    g = C.GOAL22
    y = (1 << g.index("TURN_L")) | (1 << g.index("STOP_POINT"))
    w = y | (1 << g.index("TURN_R"))
    L = {"goal_y": np.array([y]), "goal_w": np.array([w]), "v_hi_ms": np.array([9.0]),
         "lon_d_reach_m": np.array([12.0]), "lon_t_reach_s": np.array([4.0]), "turn_t_start_s": np.array([1.0]),
         "turn_t_end_s": np.array([5.0]), "lead_valid": np.array([0]), "lead_tg_min_s": np.array([2.0])}
    c = C.constraints_from_v9(L, np.array([13.9]), np.full((1, K), np.nan), np.zeros((1, K)))
    assert c["turn_side"][0] == 1 and c["stop_active"][0] and np.isnan(c["lead_tg_min"][0])
    # a bit with y=1 but w=0 (IGNORE) is not a goal
    L2 = dict(L, goal_w=np.array([1 << g.index("TURN_R")]))
    c2 = C.constraints_from_v9(L2, np.array([13.9]), np.full((1, K), np.nan), np.zeros((1, K)))
    assert c2["turn_side"][0] == 0 and not c2["stop_active"][0]
