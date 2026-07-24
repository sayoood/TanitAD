"""Tests for the v3 label set (scripts/refb_labels.py): the vocabulary-complete
ROUTE tokens with DISTANCE-TO-MANEUVER, and the FACTORIZED tactical labeler.

Two defects are pinned here, both instances of the same disease — the frozen
vocabulary (`tanitad.lake.vocab`) is richer than anything that mints it:

  ROUTE     vocab freezes 9 tokens; the labeler emitted 3 + a sentinel, so
            `exit_left`, `exit_right`, `merge`, `u_turn`, `roundabout` (and
            `straight`) had never been minted by anything.
  TACTICAL  `N_MANEUVERS = 5` packs 3 LATERAL and 2 LONGITUDINAL classes into
            one softmax and resolves the collision by PRIORITY (turn > brake >
            accel), while vocab already models them as independent slots
            LATMANEUVER (9) x LONMODE (9) x TACPOINT (5).

Pins:
  (A) ADDITIVE — v1 / v2 / v2.1 are untouched, and v3's 3-class `route`/`valid`
      are byte-identical to v2.1 on every synthetic track (so switching a run to
      v3 cannot regress the 26.0 % -> 81.9 % coverage fix).
  (B) VOCAB<->CODE PIN — the labeler's token tuples are exactly the frozen vocab
      slots (or documented subsets), and the frozen counts still hold.
  (C) ROUNDABOUT is minted from the ego track alone when its full signature is
      present, and is NOT minted (staying at v2.1's defensible turn_*) when the
      exit is not observed.
  (D) U-TURN is minted; a roundabout is not mistaken for one.
  (E) EXIT / MERGE are minted only on the diverge-and-straighten signature, and
      a plain road curve / a lane change do NOT false-positive into them.
  (F) DISTANCE-TO-MANEUVER is in metres, monotone, and keeps `d_none`
      ("looked far enough, nothing there") distinct from `d_unknown`.
  (G) FACTORIZATION — a braking TURN carries BOTH a lateral class and a live
      longitudinal mode, which the 5-way label cannot express (`collapsed`).
  (H) LONMODE lead-referenced tokens are NEVER emitted (no lead state exists),
      and the TACPOINT token stays `unknown` while its DISTANCE is minted.

CPU-only, synthetic trajectories.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels as R                                          # noqa: E402
from tanitad.lake import vocab as V                              # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic trajectory builders (10 Hz, +y = left, CCW-positive yaw)           #
# --------------------------------------------------------------------------- #
def _drive(spec, dt=0.1):
    """spec = [(n_steps, speed, yaw_rate), ...] -> poses [T, 4]."""
    rows, x, y, yaw = [], 0.0, 0.0, 0.0
    for n, v, yr in spec:
        for _ in range(int(n)):
            rows.append([x, y, yaw, v])
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += yr * dt
    rows.append([x, y, yaw, rows[-1][3]])
    return torch.tensor(rows, dtype=torch.float32)


def _arc(v, radius, deg, sign=1.0, dt=0.1):
    """(n_steps, v, yaw_rate) sweeping ``deg`` degrees at ``radius``."""
    yr = sign * v / radius
    n = int(round(math.radians(deg) / (abs(yr) * dt)))
    return (n, v, yr)


def _roundabout(approach_m=60.0, v_app=10.0, v_ring=6.0, radius=12.0,
                sweep_deg=200.0, exit_deg=25.0, sign=1.0):
    """Straight approach -> a constant-radius ring -> the EXIT counter-deflection
    -> straight away. The exit reversal is what makes it a roundabout and not
    'a very tight long turn'."""
    n_app = int(round(approach_m / (v_app * 0.1)))
    return _drive([(n_app, v_app, 0.0),
                   _arc(v_ring, radius, sweep_deg, sign),
                   _arc(v_ring, radius, exit_deg, -sign),
                   (80, v_app, 0.0)])


def _u_turn(approach_m=60.0, v_app=9.0, v_turn=5.0, radius=8.0, sign=1.0):
    n_app = int(round(approach_m / (v_app * 0.1)))
    return _drive([(n_app, v_app, 0.0), _arc(v_turn, radius, 180.0, sign),
                   (100, v_app, 0.0)])


def _junction_turn(approach_m=60.0, v_app=10.0, v_turn=7.0, radius=15.0,
                   deg=90.0, sign=1.0):
    n_app = int(round(approach_m / (v_app * 0.1)))
    return _drive([(n_app, v_app, 0.0), _arc(v_turn, radius, deg, sign),
                   (100, v_app, 0.0)])


def _offramp(v0=26.0, v1=18.0, deflect_deg=14.0, sign=-1.0, run=120):
    """Diverge off the through-line at a small angle, straighten, DECELERATE —
    the exit-ramp signature: the offset accumulates, the heading returns."""
    return _drive([(40, v0, 0.0),
                   _arc(v0, 220.0, deflect_deg, sign),
                   (run, (v0 + v1) / 2, 0.0),
                   _arc(v1, 220.0, deflect_deg, -sign),
                   (60, v1, 0.0)])


def _onramp(v0=14.0, v1=26.0, deflect_deg=14.0, sign=1.0, run=120):
    """Same shape, ACCELERATING onto the through road = merge."""
    return _drive([(40, v0, 0.0),
                   _arc(v0, 220.0, deflect_deg, sign),
                   (run, (v0 + v1) / 2, 0.0),
                   _arc(v1, 220.0, deflect_deg, -sign),
                   (60, v1, 0.0)])


def _lane_change(v=18.0, lat_m=3.5, secs=4.0):
    """A real lane change: ~one lane of offset, net heading ~0."""
    deg = math.degrees(math.atan(2 * lat_m / (v * secs)))
    return _drive([(40, v, 0.0), _arc(v, 400.0, deg, 1.0),
                   _arc(v, 400.0, deg, -1.0), (60, v, 0.0)])


def _road_curve(v=28.0, radius=479.0, deg=48.0):
    """The ep_00069 val case: a wide sustained sweep. Must NOT read as an exit."""
    return _drive([(20, v, 0.0), _arc(v, radius, deg, 1.0), (20, v, 0.0)])


def _straight(v=12.0, n=250):
    return _drive([(n, v, 0.0)])


def _brake_to_stop(v0=2.6, decel=1.3, cruise=40, yaw_rate=0.0):
    """The ep19 clip's shape: rolling at 2.6 m/s and decelerating to a full stop
    (a pedestrian crossing), then holding. Optionally while turning."""
    rows, x, y, yaw, v, dt = [], 0.0, 0.0, 0.0, v0, 0.1
    for _ in range(cruise):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
    while v > 0.0:
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v - decel * dt)
    for _ in range(60):
        rows.append([x, y, yaw, 0.0])
    return torch.tensor(rows, dtype=torch.float32)


ALL_TRACKS = {
    "roundabout_l": _roundabout(sign=1.0), "roundabout_r": _roundabout(sign=-1.0),
    "u_turn": _u_turn(), "junction_l": _junction_turn(sign=1.0),
    "junction_r": _junction_turn(sign=-1.0), "offramp": _offramp(),
    "onramp": _onramp(), "lane_change": _lane_change(),
    "road_curve": _road_curve(), "straight": _straight(),
    "brake_stop": _brake_to_stop(),
}


# ---------- (A) additive: v1/v2/v2.1 untouched, v3 == v2.1 on the CE ---------
def test_v3_does_not_change_the_v21_decision_anywhere():
    """v3 only ever UPGRADES the token. `route`, `valid` and `ambiguous` — the
    fields every shipped consumer reads — must be identical to v2.1 on every
    window of every synthetic track."""
    n = 0
    for name, poses in ALL_TRACKS.items():
        for t in range(0, poses.shape[0] - 1, 7):
            a = R.route_from_future_v21(poses, t)
            b = R.route_from_future_v3(poses, t)
            assert b["route"] == a["route"], (name, t)
            assert b["valid"] == a["valid"] and b["ambiguous"] == a["ambiguous"]
            assert b["reason"] == a["reason"], (name, t)   # v2.1 reason intact
            for k in ("net_dyaw", "mean_curv", "peak_kappa", "arc_m"):
                assert b[k] == a[k], (name, t, k)
            n += 1
    assert n > 200


def test_v1_and_v2_paths_still_byte_identical():
    curve = _road_curve()
    r2 = R.route_from_future(curve, 0)
    assert set(r2) == {"route", "valid", "ambiguous", "net_dyaw", "signed_curv",
                       "peak_kappa", "concentration"}
    j = _junction_turn()
    assert R.nav_command(j, 0)[0] in (R.NAV_LEFT, R.NAV_RIGHT, R.NAV_FOLLOW)
    assert int(R.maneuver_labels(j)[0]) in range(5)
    assert int(R.maneuver_labels_v2(j)[0]) in range(5)


# ---------- (B) the vocab <-> code pin ---------------------------------------
def test_route_v3_tokens_are_exactly_the_frozen_vocab_slot():
    assert R.ROUTE_V3_TOKENS == V.STRATEGIC_TOKENS["ROUTE"]
    assert len(R.ROUTE_V3_TOKENS) == 9


def test_tactical_token_tuples_are_subsets_of_the_frozen_slots():
    lon = set(V.TACTICAL_TOKENS["LONMODE"])
    lat = set(V.TACTICAL_TOKENS["LATMANEUVER"])
    assert set(R.LON_KINEMATIC_TOKENS) | set(R.LON_LEAD_TOKENS) == lon
    assert set(R.LON_KINEMATIC_TOKENS) & set(R.LON_LEAD_TOKENS) == set()
    assert set(R.LAT_KINEMATIC_TOKENS) | set(R.LAT_CONTEXT_TOKENS) == lat
    assert len(lon) == 9 and len(lat) == 9
    assert len(V.TACTICAL_TOKENS["TACPOINT"]) == 5


def test_frozen_vocabulary_counts_are_unchanged_by_the_v11_candidate():
    """The distance slot is a CANDIDATE: it must not touch the freeze."""
    assert sum(len(v) for v in V.STRATEGIC_TOKENS.values()) == 34
    assert sum(len(v) for v in V.TACTICAL_TOKENS.values()) == 80
    assert len(V.GOAL_SLOTS) == 18
    assert set(V.empty_goal()) == set(V.GOAL_SLOTS)
    assert not set(V.V11_CANDIDATE_TOKENS) & set(V.GOAL_SLOTS)
    assert len(V.ROUTEDIST_TOKENS) == 8
    assert R.DIST_BAND_TOKENS == V.ROUTEDIST_TOKENS       # one banding, two homes


def test_dist_band_matches_the_vocab_banding_function():
    for d in (None, 0.0, 5.0, 9.99, 10.0, 24.9, 25.0, 60.0, 150.0, 999.0):
        for arc in (0.0, 50.0, 500.0):
            assert R.dist_band(d, arc) == V.routedist_band(d, arc), (d, arc)


def test_kinematic_constants_agree_with_goal_labels():
    """goal_labels re-declares three of these; drift would silently give the
    lake and the trainer different lateral labels."""
    from tanitad.lake import goal_labels as G
    assert R.CREEP_CEIL_MS == G.CREEP_CEIL_MS
    assert R.LANE_HALF_M == G.LANE_HALF_M
    assert R.LC_NET_YAW_MAX == G.LC_NET_YAW_MAX


# ---------- (C) roundabout ----------------------------------------------------
def test_roundabout_is_minted_from_the_ego_track_alone():
    for sign, expect_v21 in ((1.0, R.ROUTE_LEFT), (-1.0, R.ROUTE_RIGHT)):
        poses = _roundabout(sign=sign)
        r = R.route_from_future_v3(poses, 0)
        assert r["token"] == "roundabout", (sign, r["token"], r["v3_rule"])
        assert r["upgraded"] and r["v3_rule"] == "roundabout"
        assert r["token_v21"] == ("turn_left" if sign > 0 else "turn_right")
        assert r["route"] == expect_v21 and r["valid"]      # CE target unchanged
        assert abs(r["maneuver_dyaw"]) > math.radians(135.0)
        # the approach is 60 m of straight road; the ring starts about there
        assert 40.0 <= r["dist_m"] <= 80.0, r["dist_m"]
        assert r["dist_band"] in ("d_50_100", "d_25_50")


def test_roundabout_without_an_observed_exit_stays_a_turn():
    """A clip that ends mid-circulation must NOT be promoted: v2.1's turn_* is
    defensible, `roundabout` there would be an invention (R3)."""
    # the same approach + ring, cut the moment the ring ends: no exit observed
    cut = _drive([(60, 10.0, 0.0), _arc(6.0, 12.0, 200.0, 1.0)])
    r = R.route_from_future_v3(cut, 0)
    assert r["token"] == "turn_left", r["token"]
    assert not r["upgraded"] and r["roundabout_candidate"]


def test_a_plain_junction_turn_is_not_a_roundabout():
    for sign in (1.0, -1.0):
        r = R.route_from_future_v3(_junction_turn(sign=sign), 0)
        assert r["token"] == ("turn_left" if sign > 0 else "turn_right")
        assert not r["upgraded"] and not r["roundabout_candidate"]


# ---------- (D) u-turn --------------------------------------------------------
def test_u_turn_is_minted_and_is_not_confused_with_a_roundabout():
    r = R.route_from_future_v3(_u_turn(), 0)
    assert r["token"] == "u_turn" and r["v3_rule"] == "u_turn"
    assert abs(abs(r["maneuver_dyaw"]) - math.pi) < math.radians(35.0)
    # the roundabout track sweeps past 180 too, but shows the exit reversal
    assert R.route_from_future_v3(_roundabout(), 0)["token"] == "roundabout"


# ---------- (E) exit / merge, and their false-positive guards -----------------
def test_offramp_mints_exit_and_onramp_mints_merge():
    e = R.route_from_future_v3(_offramp(sign=-1.0), 0)
    assert e["token"] == "exit_right" and e["v3_rule"] == "exit"
    el = R.route_from_future_v3(_offramp(sign=1.0), 0)
    assert el["token"] == "exit_left"
    m = R.route_from_future_v3(_onramp(), 0)
    assert m["token"] == "merge" and m["v3_rule"] == "merge"


def test_road_curve_and_lane_change_do_not_false_positive_as_exits():
    for name, poses in (("road_curve", _road_curve()),
                        ("lane_change", _lane_change()),
                        ("straight", _straight())):
        r = R.route_from_future_v3(poses, 0)
        assert r["token"] not in ("exit_left", "exit_right", "merge",
                                  "roundabout", "u_turn"), (name, r["token"])


def test_straight_is_never_minted():
    """`straight` asserts a junction exists = a MAP fact, not a trajectory one."""
    for poses in ALL_TRACKS.values():
        for t in range(0, poses.shape[0] - 1, 11):
            assert R.route_from_future_v3(poses, t)["token"] != "straight"


# ---------- (F) distance-to-maneuver -----------------------------------------
def test_distance_to_maneuver_is_metres_and_monotone_as_we_approach():
    poses = _junction_turn(approach_m=120.0, v_app=12.0)
    ds = [R.route_from_future_v3(poses, t)["dist_m"] for t in (0, 20, 40, 60)]
    assert all(d is not None for d in ds), ds
    assert ds == sorted(ds, reverse=True), ds        # shrinks as we drive at it
    assert abs(ds[0] - 120.0) < 25.0, ds[0]          # metres, not steps/seconds
    # speed-invariance: the SAME geometry at half the speed = the same metres
    slow = _junction_turn(approach_m=120.0, v_app=6.0)
    assert abs(R.route_from_future_v3(slow, 0)["dist_m"] - ds[0]) < 12.0


def test_d_none_and_d_unknown_are_distinct():
    """The v2.1 lesson (a silent fallback conflating 'cannot judge' with a real
    class poisoned the route prior) applied to the distance axis."""
    far = R.route_from_future_v3(_straight(v=12.0, n=250), 0)
    assert far["dist_m"] is None and far["dist_band"] == "d_none"
    short = R.route_from_future_v3(_straight(v=2.0, n=40), 0)   # ~8 m of road
    assert short["dist_band"] == "d_unknown"
    assert R.dist_band(None, 0.0) == "d_unknown"
    assert R.dist_band(None, 500.0) == "d_none"


def test_dist_band_edges():
    assert R.dist_band(0.0, 500) == "d_now"
    assert R.dist_band(9.99, 500) == "d_now"
    assert R.dist_band(10.0, 500) == "d_10_25"
    assert R.dist_band(24.99, 500) == "d_10_25"
    assert R.dist_band(25.0, 500) == "d_25_50"
    assert R.dist_band(99.9, 500) == "d_50_100"
    assert R.dist_band(100.0, 500) == "d_100_200"
    assert R.dist_band(2000.0, 500) == "d_200_plus"


# ---------- (G) the factorization ---------------------------------------------
def test_stop_at_a_crossing_is_lane_keep_AND_stop_at_point_with_a_distance():
    """The ep19 clip: rolling at 2.6 m/s, decelerating to a full stop at a
    pedestrian crossing. The overlay could only say `lane keep`."""
    poses = _brake_to_stop(v0=2.6, decel=1.3)
    d = R.tactical_from_future_v3(poses, 0)
    assert d["lat"]["token"] == "lane_keep" and d["lat"]["valid"]
    assert d["lon"]["token"] == "stop_at_point" and d["lon"]["active"]
    assert d["lon"]["stop_dist_m"] is not None
    # 4 s of rolling at 2.6 m/s (10.4 m) + 2.6 m of braking distance
    assert 10.0 < d["lon"]["stop_dist_m"] < 16.0, d["lon"]["stop_dist_m"]
    assert d["lon"]["stop_dist_band"] == "d_10_25"
    # the POSITION is minted; the NAME is not (crossing vs stop line vs queue)
    assert d["tacpoint"]["token"] == "unknown" and not d["tacpoint"]["valid"]


def test_a_stop_beyond_the_2s_head_horizon_is_still_seen():
    """A stop ~30 m away at 13 m/s needs ~5 s of lookahead — structurally
    invisible to the 2 s 5-way head, which is the point of STOP_SEARCH_STEPS."""
    poses = _brake_to_stop(v0=13.0, decel=2.2, cruise=5)
    d = R.tactical_from_future_v3(poses, 0)
    assert d["lon"]["token"] == "stop_at_point"
    assert 30.0 < d["lon"]["stop_dist_m"] < 60.0, d["lon"]["stop_dist_m"]
    assert d["lon"]["stop_dist_band"] == "d_25_50"
    assert d["man5"] == R.BRAKE_STOP


def test_braking_turn_carries_both_axes_and_the_5way_collapses_it():
    """A turn taken while braking to a stop. `classify_maneuver_v2` resolves the
    collision by PRIORITY (turn > brake), so the 5-way label emits TURN and the
    longitudinal decision is destroyed. The factorized labeler keeps both."""
    poses = _brake_to_stop(v0=8.0, decel=1.6, cruise=10, yaw_rate=8.0 / 14.0)
    d = R.tactical_from_future_v3(poses, 0)
    assert d["man5"] == R.TURN_LEFT                       # priority collapse
    assert d["lon"]["token"] == "stop_at_point" and d["lon"]["active"]
    assert d["collapsed"] is True


def test_lateral_and_longitudinal_are_independent():
    lc = _lane_change()
    d = R.tactical_from_future_v3(lc, 40)      # the change starts at step 40
    assert d["lat"]["token"] in ("lc_left", "lc_right", "nudge_left",
                                 "nudge_right")
    assert d["lon"]["token"] == "free_cruise" and not d["lon"]["active"]
    st = R.tactical_from_future_v3(_straight(), 0)
    assert st["lat"]["token"] == "lane_keep"
    assert st["lon"]["token"] == "free_cruise"
    assert st["collapsed"] is False


def test_hold_stop_launch_and_creep():
    held = torch.tensor([[0.0, 0.0, 0.0, 0.0]] * 60, dtype=torch.float32)
    assert R.lonmode_from_future(held, 0)["token"] == "hold_stop"
    launch = _drive([(5, 0.0, 0.0)] + [(3, v, 0.0) for v in
                                       (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5)])
    assert R.lonmode_from_future(launch, 0)["token"] == "launch"
    creep = _drive([(80, 1.2, 0.0)])
    assert R.lonmode_from_future(creep, 0)["token"] == "creep"


# ---------- (H) the honest gaps ----------------------------------------------
def test_lead_referenced_lonmodes_are_never_emitted():
    """`lead_state` is a None stub: follow_lead / close_gap / open_gap are not
    kinematically separable and must never be faked."""
    seen = set()
    for poses in ALL_TRACKS.values():
        for t in range(0, poses.shape[0] - 1, 5):
            seen.add(R.lonmode_from_future(poses, t)["token"])
    assert not seen & set(R.LON_LEAD_TOKENS), seen
    assert seen <= set(R.LON_KINEMATIC_TOKENS) | {R.TOKEN_UNKNOWN}


def test_context_lateral_tokens_are_never_emitted():
    seen = set()
    for poses in ALL_TRACKS.values():
        for t in range(0, poses.shape[0] - 1, 5):
            seen.add(R.latmaneuver_from_future(poses, t)["token"])
    assert not seen & set(R.LAT_CONTEXT_TOKENS), seen
    assert seen <= set(R.LAT_KINEMATIC_TOKENS) | {R.TOKEN_UNKNOWN}


def test_unjudgeable_windows_stay_unknown_not_a_guess():
    tiny = _straight(v=8.0, n=3)
    r = R.route_from_future_v3(tiny, 1)
    assert r["token"] == "unknown" and not r["valid"]
    assert r["dist_band"] == "d_unknown"
    assert R.lonmode_from_future(tiny, 2)["token"] == "unknown"
    assert R.latmaneuver_from_future(tiny, 1)["token"] == "unknown"


def test_route_target_v3_triple():
    tok, band, valid = R.route_target_v3(_roundabout(), 0)
    assert tok == "roundabout" and valid and band.startswith("d_")
