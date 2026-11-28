"""WP-A v9 builder: analytic controls (SPEC §9.V V4) and one MUTATION per field that must go RED (V6).

Every expectation is a LITERAL from geometry (a circle's radius, a constant deceleration's stopping distance), never
an expression over the code under test. Synthetic 100 Hz logs enter through ``log_from_arrays``.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
B = pytest.importorskip("build_v9_labels")

DT = 0.01


def make_log(v_fn, omega_fn, t_end=40.0, reverse_fn=None):
    """Integrate speed v(t) [m/s] and yaw rate omega(t) [rad/s] at 100 Hz from the origin heading +x."""
    t = np.arange(0.0, t_end + 1e-9, DT)
    v = np.array([v_fn(a) for a in t], np.float64)
    w = np.array([omega_fn(a) for a in t], np.float64)
    yaw = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * DT)])
    sgn = np.ones_like(t) if reverse_fn is None else np.array([-1.0 if reverse_fn(a) else 1.0 for a in t])
    x = np.concatenate([[0.0], np.cumsum(sgn[1:] * 0.5 * (v[1:] + v[:-1]) * np.cos(yaw[1:]) * DT)])
    y = np.concatenate([[0.0], np.cumsum(sgn[1:] * 0.5 * (v[1:] + v[:-1]) * np.sin(yaw[1:]) * DT)])
    rev = ((v > 0.3) & (sgn < 0)).astype(np.float64)
    return B.log_from_arrays(t, x, y, yaw, v, rev=rev)


def labels(log, nows, lead=None, lc=None, lc_labels=True):
    G = B.clip_geo(log, None)
    k = np.round(np.asarray(nows, np.float64) / 0.1).astype(int)       # a clock with grid_start 0 and dt 0.1 s
    return G, B.clip_labels(G, np.asarray(nows, np.float64), k, None, lead, lc, lc_labels=lc_labels)


def circle_left(R=15.0, v=5.0, t_on=13.0):
    """straight at v, then a left circle of radius R through 90 deg starting at t_on, then straight."""
    T90 = (math.pi / 2) * R / v
    return make_log(lambda t: v, lambda t: (v / R) if t_on <= t < t_on + T90 else 0.0), T90


# ------------------------------------------------------------------------------------------- V4 analytic controls
def test_circle_reads_turn_left_with_its_radius_and_timing():
    log, T90 = circle_left()
    _, F = labels(log, [10.0])                      # band [12, 18] contains the whole 90 deg left turn at 13 s
    assert F["lat_cls_a"][0] == 1 and F["lat_cls_b"][0] == 1               # TURN_L, both variants
    assert abs(F["turn_r_arc_m"][0] - 15.0) <= 0.3 * 2                     # R_arc of the detected segment
    assert abs(F["turn_dyaw_deg"][0] - 90.0) <= 1.5
    assert abs(F["turn_t_start_s"][0] - 3.0) <= 0.15                       # 13.0 - 10.0
    assert F["turn_is_turn"][0] == 1


def test_circle_mirror_reads_turn_right():
    v, R = 5.0, 15.0
    T90 = (math.pi / 2) * R / v
    log = make_log(lambda t: v, lambda t: -(v / R) if 13.0 <= t < 13.0 + T90 else 0.0)
    _, F = labels(log, [10.0])
    assert F["lat_cls_a"][0] == 2 and F["lat_cls_b"][0] == 2


def test_straight_reads_lane_keep_and_keep():
    log = make_log(lambda t: 14.0, lambda t: 0.0)
    _, F = labels(log, [10.0], lead={})           # agent data present but empty: FOLLOW decidable (no lead)
    assert F["lat_cls_a"][0] == 0
    assert F["lon_cls_computed"][0] == 6                                   # KEEP


def test_constant_deceleration_reads_stop_at_the_right_time_and_distance():
    # 10 m/s until NOW + 2.5 s, then -2 m/s^2: v first <= 0.5 m/s at NOW + 2.5 + 4.75 = 7.25 s (rest at 7.5 s);
    # distance to that point = 25 + (10^2 - 0.5^2) / (2 * 2) = 49.94 m
    now = 10.0
    log = make_log(lambda t: 10.0 if t < now + 2.5 else max(0.0, 10.0 - 2.0 * (t - now - 2.5)), lambda t: 0.0)
    _, F = labels(log, [now])
    assert F["lon_cls"][0] == 2                                            # STOP
    assert abs(F["lon_t_reach_s"][0] - 7.25) <= 0.1                       # 0.1 s grid
    assert abs(F["lon_d_reach_m"][0] - 49.94) <= 0.2
    assert F["lon_v7id"][0] == B.LON7.index("BRAKE_TO")


def test_acceleration_ramp_reads_accelerate():
    log = make_log(lambda t: 8.0 + 0.5 * max(0.0, t - 12.0), lambda t: 0.0)
    _, F = labels(log, [10.0], lead={})
    assert F["lon_cls_computed"][0] == 5


def test_rc_on_a_circle_with_no_smoothing_is_the_chord_point():
    log, _ = circle_left(R=15.0, v=5.0, t_on=13.0)
    tr = B.track10(log)
    p0 = B.smooth_path(tr, 0.0)
    now = 10.0                                    # 15 m before the arc
    x0, y0, psi0, _ = (float(a) for a in B.sample(log, now))
    s_now = float(B.s_at(tr, now))
    x, y, _, ok = B.rc_point(p0, s_now + 25.0, x0, y0, psi0)               # 10 m into the arc
    th = 10.0 / 15.0
    assert ok and abs(x - (15.0 + 15.0 * math.sin(th))) < 0.05 and abs(y - 15.0 * (1 - math.cos(th))) < 0.05


def test_nav_reads_the_exact_distance_to_a_known_turn():
    log, _ = circle_left(R=15.0, v=5.0, t_on=13.0)
    G = B.clip_geo(log, None)
    now = 9.0                                     # 20 m before the turn starts (v = 5 m/s)
    nav = B.nav_fields(G.turns, float(B.s_at(G.tr, now)), now, 5.0, G.s_max)
    assert nav["nav_side_next"] == 1 and nav["nav_args_valid"] == 1
    assert abs(nav["nav_d_next_m"] - 20.0) <= 0.6                        # the 6 deg/s onset sits a hair after 13.0 s
    assert nav["nav_token"] == B.NAV_TURN_L                                # 20 m <= max(30 m, 6 s * 5 m/s)


def test_lead_at_two_seconds_reads_follow():
    log = make_log(lambda t: 10.0, lambda t: 0.0)
    tr = B.track10(log)
    p0 = B.smooth_path(tr, 0.0)
    # lead centre 27.5 m ahead, length 4.0 m -> bumper gap 27.5 - 2.0 - 3.6 = 21.9 m, time gap 2.19 s
    L, gap, tg, s, v = B.lead_for_frame(log, tr, p0, 10.0, [{"cls": "automobile", "cx": 27.5, "cy": 0.3, "l": 4.0}])
    assert L and abs(gap - 21.9) <= 0.2 and abs(tg - 2.19) <= 0.05
    lead = {k: (True, True, 21.9, 2.19, 27.5, 10.0) for k in range(0, 400)}
    _, F = labels(log, [10.0], lead=lead)
    assert F["lon_cls"][0] == 3                                            # FOLLOW


def test_ego_footprint_box_is_never_a_lead():
    log = make_log(lambda t: 10.0, lambda t: 0.0)
    tr = B.track10(log)
    L, *_ = B.lead_for_frame(log, tr, B.smooth_path(tr, 0.0), 10.0, [{"cls": "automobile", "cx": 1.4, "cy": 0.0, "l": 4.6}])
    assert not L


def _lc_table(now0, side=+1, t_cross=14.0, n=300, dt=0.1):
    """lane offsets for a 3.5 m lane: the ego drifts toward the line and crosses it at t_cross (left: d_L -> 0)."""
    k = np.arange(n)
    t = k * dt
    pos = np.clip((t - (t_cross - 2.0)) / 4.0, 0.0, 1.0) * 3.5            # lateral progress across one lane
    yl = 1.75 - side * pos                                                   # left line offset in a lane-centred frame
    dl = np.mod(yl, 3.5)
    dl = np.where(dl < 1e-6, 3.5, dl)
    dr = 3.5 - dl
    return {"k": k, "dl": dl, "dr": dr}


def test_synthetic_left_lane_change_reads_lane_change_left():
    log = make_log(lambda t: 14.0, lambda t: 0.0)
    _, F = labels(log, [10.0], lc=_lc_table(10.0, side=+1))
    assert F["lc_measurable"][0] == 1
    assert F["lat_cls_a"][0] == 3 and F["lc_side"][0] == 1


def test_lc_labels_off_keeps_the_measurement_but_never_labels_a_lane_change():
    """The release default after V5 failed: the crossing stays a diagnostic, the class takes the partial path."""
    log = make_log(lambda t: 14.0, lambda t: 0.0)
    _, F = labels(log, [10.0], lc=_lc_table(10.0, side=+1), lc_labels=False)
    assert F["lc_side"][0] == 1 and F["lc_measurable_raw"][0] == 1 and F["lc_measurable"][0] == 0
    assert F["lat_cls_a"][0] != 3


def test_reversing_window_is_masked():
    log = make_log(lambda t: 2.0, lambda t: 0.0, reverse_fn=lambda t: 13.0 <= t < 16.0)
    _, F = labels(log, [10.0])
    assert F["reversing"][0] == 1 and F["lat_allowed_a"][0] == 0 and F["lon_allowed"][0] == 0


# ------------------------------------------------------------------------------------------- V6 mutations (must go RED)
def test_mutation_yaw_sign_flip_flips_the_turn_side():
    v, R = 5.0, 15.0
    T90 = (math.pi / 2) * R / v
    good = make_log(lambda t: v, lambda t: (v / R) if 13.0 <= t < 13.0 + T90 else 0.0)
    bad = B.log_from_arrays(good.ts, good.x, -good.y, -good.yaw, good.v)
    assert labels(good, [10.0])[1]["lat_cls_a"][0] == 1
    assert labels(bad, [10.0])[1]["lat_cls_a"][0] == 2                    # the mutation is visible


def test_mutation_band_shift_loses_the_turn():
    log, _ = circle_left(t_on=13.0)
    assert labels(log, [10.0])[1]["lat_cls_b"][0] == 1
    assert labels(log, [16.0 + 6.0])[1]["lat_cls_b"][0] == 0               # band [24, 30]: the turn ended at ~17.7 s


def test_mutation_speed_scale_moves_the_stop_distance():
    now = 10.0
    f = lambda s: (lambda t: s * (10.0 if t < now + 2.5 else max(0.0, 10.0 - 2.0 * (t - now - 2.5))))
    d1 = labels(make_log(f(1.0), lambda t: 0.0), [now])[1]["lon_d_reach_m"][0]
    d2 = labels(make_log(f(1.5), lambda t: 0.0), [now])[1]["lon_d_reach_m"][0]
    assert abs(d2 - 1.5 * d1) <= 1.0 and abs(d2 - d1) > 10.0


def test_mutation_mirrored_map_flips_the_lane_change_side():
    log = make_log(lambda t: 14.0, lambda t: 0.0)
    lc = _lc_table(10.0, side=+1)
    mir = {"k": lc["k"], "dl": lc["dr"], "dr": lc["dl"]}
    assert labels(log, [10.0], lc=lc)[1]["lc_side"][0] == 1
    assert labels(log, [10.0], lc=mir)[1]["lc_side"][0] == -1


def test_mutation_removing_the_lead_removes_follow():
    log = make_log(lambda t: 10.0, lambda t: 0.0)
    lead = {k: (True, True, 21.9, 2.19, 27.5, 10.0) for k in range(0, 400)}
    nolead = {k: (True, False, np.nan, np.nan, np.nan, 10.0) for k in range(0, 400)}
    assert labels(log, [10.0], lead=lead)[1]["lon_cls"][0] == 3
    assert labels(log, [10.0], lead=nolead)[1]["lon_cls"][0] != 3


def test_mutation_nav_arc_from_the_clip_start_is_wrong():
    log, _ = circle_left(t_on=13.0)
    G = B.clip_geo(log, None)
    now = 9.0
    good = B.nav_fields(G.turns, float(B.s_at(G.tr, now)), now, 5.0, G.s_max)["nav_d_next_m"]
    bad = B.nav_fields(G.turns, 0.0, now, 5.0, G.s_max)["nav_d_next_m"]   # D4's mutation: arc origin at the clip start
    assert abs(good - 20.0) <= 0.6 and abs(bad - good) > 40.0


def test_mutation_rc_y_flip_flips_the_checkpoint_side():
    log, _ = circle_left(t_on=13.0)
    G = B.clip_geo(log, None)
    rc = B.rc_fields(G.tr, G.log, G.paths, G.turns, 10.0, G.s_nat)
    flipped = B.log_from_arrays(log.ts, log.x, -log.y, -log.yaw, log.v)
    G2 = B.clip_geo(flipped, None)
    rc2 = B.rc_fields(G2.tr, G2.log, G2.paths, G2.turns, 10.0, G2.s_nat)
    assert rc["rc_A50_y"] > 1.0 and rc2["rc_A50_y"] < -1.0


def test_builder_holds_no_mutable_module_state_between_clips():
    """Two clips built in sequence give the same labels as each built alone (the D1 F2 failure family)."""
    a, _ = circle_left(t_on=13.0)
    b = make_log(lambda t: 14.0, lambda t: 0.0)
    alone = labels(a, [10.0])[1]["lat_cls_a"][0]
    labels(b, [10.0])
    again = labels(a, [10.0])[1]["lat_cls_a"][0]
    assert alone == again == 1
