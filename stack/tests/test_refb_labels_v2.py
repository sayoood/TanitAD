"""Tests for the v2 curvature-relative REF-B/flagship labels + the label-quality
harness (scripts/refb_labels.py v2 additions, scripts/validate_refb_labels.py).

Pins:
  (A) v1 is UNCHANGED by the v2 addition (regression guard on the shipped path).
  (B) path_curvature == 1/R on a known-radius arc; standstill -> 0 (no NaN).
  (C) v2 maneuver is CURVATURE-GATED: a gentle highway curve that v1 calls a
      turn stays lane_keep under v2; tight turns still turn; accel/brake/stop
      logic is preserved; window==episode.
  (D) v2 route separates ROAD-FOLLOWING from JUNCTION turns: a sustained gentle
      curve -> straight(valid); a tight discrete turn -> turn(valid, right sign,
      wrap-robust); a gray-zone fork -> AMBIGUOUS(valid=False); too-short ->
      invalid. Direction from peak curvature is robust to yaw wrap (>180 turn).
  (E) circularity BREAK: route_target_v2 is a function of the FUTURE PATH, not
      of any fed nav command (unlike v1's route_target(nav_cmd)).
  (F) harness: on the synthetic GT corpus v2 strictly reduces the road-curve-vs-
      junction conflation vs v1, keeps sane class balance, is deterministic; the
      circularity section reports the command-echo null.
CPU-only, synthetic trajectories.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels as R  # noqa: E402
import validate_refb_labels as V  # noqa: E402


def _poses(T, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.0, yaw0=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, yaw0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


# ---------- (A) v1 unchanged -------------------------------------------------

def test_v1_labels_unchanged_regression():
    H = R.LABEL_HORIZON
    assert (R.maneuver_labels(_poses(40), H) == R.LANE_KEEP).all()
    assert (R.maneuver_labels(_poses(40, yaw_rate=0.2), H) == R.TURN_LEFT).all()
    # v1 nav still thresholds NET heading over time (the behavior we improve on)
    left = _poses(400, yaw_rate=0.06)
    assert R.nav_command(left, 0) == (R.NAV_LEFT, True)
    assert R.route_target(R.NAV_LEFT) == R.ROUTE_LEFT
    # v1 route_target IS the fed-command derivation (circular, by construction)
    assert R.route_target(R.NAV_FOLLOW) == R.ROUTE_STRAIGHT


# ---------- (B) path curvature ----------------------------------------------

def test_path_curvature_equals_inverse_radius():
    for R_m, v in ((300.0, 30.0), (60.0, 12.0), (15.0, 8.0)):
        p = _poses(120, v0=v, yaw_rate=v / R_m)          # constant-radius arc
        k = R.path_curvature(p)
        assert torch.isfinite(k).all()
        # left turn (yaw_rate>0) -> +kappa; magnitude ~ 1/R
        assert (k > 0).float().mean() > 0.9
        assert abs(float(k.median()) - 1.0 / R_m) < 0.1 / R_m   # within 10 %


def test_path_curvature_standstill_zero_no_nan():
    st = _poses(60, v0=0.0)                               # never moves
    k = R.path_curvature(st)
    assert torch.isfinite(k).all()
    assert float(k.abs().max()) == 0.0


# ---------- (C) v2 maneuver is curvature-gated ------------------------------

def test_v2_maneuver_gentle_highway_curve_is_lane_keep():
    H = R.LABEL_HORIZON
    sweep = _poses(60, v0=30.0, yaw_rate=0.1)             # R=300 m, road-follow
    m1 = R.maneuver_labels(sweep, H)
    m2 = R.maneuver_labels_v2(sweep, H)
    assert (m1 == R.TURN_LEFT).all()                     # v1 mislabels a turn
    assert (m2 == R.LANE_KEEP).all()                     # v2: lane-keeping


def test_v2_maneuver_tight_turn_and_straight_and_accel_brake():
    H = R.LABEL_HORIZON
    tight = _poses(60, v0=8.0, yaw_rate=8.0 / 15.0)      # R=15 m junction turn
    assert (R.maneuver_labels_v2(tight, H) == R.TURN_LEFT).all()
    tight_r = _poses(60, v0=8.0, yaw_rate=-8.0 / 15.0)
    assert (R.maneuver_labels_v2(tight_r, H) == R.TURN_RIGHT).all()
    assert (R.maneuver_labels_v2(_poses(40), H) == R.LANE_KEEP).all()
    # accel/brake/stop preserved from v1 (same thresholds/priority)
    assert (R.maneuver_labels_v2(_poses(40, accel=1.0), H) == R.ACCELERATE).all()
    assert (R.maneuver_labels_v2(_poses(40, accel=-1.0), H) == R.BRAKE_STOP).all()
    stop = R.maneuver_labels_v2(_poses(40, v0=1.2, accel=-0.48), H)
    assert stop[0] == R.BRAKE_STOP


def test_v2_window_maneuver_matches_episode():
    H = R.LABEL_HORIZON
    p = _poses(60, v0=8.0, yaw_rate=8.0 / 15.0)
    ep = R.maneuver_labels_v2(p, H)
    win = R.window_maneuver_labels_v2(p[:1], p[1:1 + H].unsqueeze(0), H)
    assert int(win[0]) == int(ep[0])


# ---------- (D) v2 route: road-follow vs junction vs ambiguous --------------

def test_v2_route_sustained_curve_is_straight_not_turn():
    curve = _poses(400, v0=28.0, yaw_rate=28.0 / 250.0)  # R=250 m, sweeps >45deg
    # v1 calls it a route TURN (net heading over 25 s exceeds 45 deg)...
    assert R.nav_command(curve, 0)[0] == R.NAV_LEFT
    # ...v2 recognizes road-following (large radius) -> straight, still valid
    r = R.route_from_future(curve, 0)
    assert r["route"] == R.ROUTE_STRAIGHT and r["valid"] and not r["ambiguous"]
    assert R.nav_command_v2(curve, 0) == (R.NAV_FOLLOW, True)


def _discrete_junction(T=400, sign=1.0, t0=150, dur=30, v_turn=8.0, R_m=15.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, 12.0
    for t in range(T):
        rows.append([x, y, yaw, v])
        turning = t0 <= t < t0 + dur
        yr = sign * v_turn / R_m if turning else 0.0
        v = v_turn if turning else 12.0
        x += v * math.cos(yaw) * 0.1
        y += v * math.sin(yaw) * 0.1
        yaw += yr * 0.1
    return torch.tensor(rows, dtype=torch.float32)


def test_v2_route_discrete_junction_is_turn_correct_direction():
    left = R.route_from_future(_discrete_junction(sign=1.0), 0)
    right = R.route_from_future(_discrete_junction(sign=-1.0), 0)
    assert left["route"] == R.ROUTE_LEFT and left["valid"]
    assert right["route"] == R.ROUTE_RIGHT and right["valid"]
    assert left["peak_kappa"] > R.CURV_TURN_PER_M
    assert left["concentration"] >= R.CONCENTRATION_MIN   # transient event


def test_v2_route_direction_robust_to_yaw_wrap():
    # sustained tight RIGHT turn > 180 deg: net_dyaw WRAPS positive, but the
    # peak-curvature sign keeps the direction correct (regression on the sign
    # bug where a >180 deg right roundabout was labeled left).
    loop_r = _poses(400, v0=8.0, yaw_rate=-8.0 / 20.0)   # R=20 m, ~-16 rad net
    assert R.route_target_v2(loop_r, 0) == R.ROUTE_RIGHT
    loop_l = _poses(400, v0=8.0, yaw_rate=8.0 / 20.0)
    assert R.route_target_v2(loop_l, 0) == R.ROUTE_LEFT


def test_v2_route_gray_zone_fork_is_ambiguous_invalid():
    # large-radius divergence (R~100 m, between R_ROAD 150 and R_TURN 60):
    # a real route decision NOT separable from a road curve -> flagged invalid.
    fork = _discrete_junction(T=400, sign=1.0, t0=150, dur=110, v_turn=24.0,
                              R_m=100.0)
    r = R.route_from_future(fork, 0)
    assert r["ambiguous"] and not r["valid"]
    assert r["route"] == R.ROUTE_STRAIGHT                 # interface stability
    # and it lands in the gray band by construction
    assert R.CURV_ROAD_PER_M < r["peak_kappa"] < R.CURV_TURN_PER_M


def test_v2_route_too_short_is_invalid_but_not_ambiguous():
    short = _poses(40)                                    # < NAV_MIN_STEPS future
    r = R.route_from_future(short, 0)
    assert not r["valid"] and not r["ambiguous"]
    assert R.nav_command_v2(short, 0) == (R.NAV_FOLLOW, False)


# ---------- (E) circularity break -------------------------------------------

def test_v2_route_target_is_future_derived_not_command_echo():
    # v1: route_target(cmd) is a pure function of the FED command (circular).
    # v2: route_target_v2(poses, t) depends on the FUTURE PATH only — feeding a
    # different "command" cannot change it (it takes no command).
    curve = _poses(400, v0=28.0, yaw_rate=28.0 / 250.0)
    junction = _discrete_junction(sign=-1.0)
    assert R.route_target_v2(curve, 0) == R.ROUTE_STRAIGHT
    assert R.route_target_v2(junction, 0) == R.ROUTE_RIGHT
    # signature carries no nav_cmd argument (structural guarantee)
    import inspect
    assert "nav_cmd" not in inspect.signature(R.route_target_v2).parameters
    assert "nav_cmd" in inspect.signature(R.route_target).parameters


# ---------- (F) harness ------------------------------------------------------

def test_harness_v2_reduces_conflation_on_gt_corpus():
    eps = V.synth_corpus(seed=0, n_episodes=80)
    sc = V.build_scorecard(eps)
    conf = sc["2_conflation"]
    assert conf["has_ground_truth"]
    v1, v2 = conf["v1"], conf["v2"]
    # v2 strictly reduces BOTH error modes and by a wide margin
    assert v2["false_turn_rate_on_road_following"] < \
        0.3 * v1["false_turn_rate_on_road_following"]
    assert v2["missed_turn_rate_on_junctions"] < \
        0.3 * v1["missed_turn_rate_on_junctions"]
    assert v1["false_turn_rate_on_road_following"] > 0.1   # v1 really is bad
    # v2 flags the honest-ceiling forks instead of forcing a hard label
    assert v2["ambiguous_forks_flagged_invalid"] > 0.8
    assert v2["road_following_kept_valid"] > 0.9           # keeps real straights
    # sane maneuver class balance (lane_keep dominant, all classes present-ish)
    man2 = sc["1_distributions"]["ALL"]["maneuver_v2"]
    assert 0.6 < man2["lane_keep"] < 0.98


def test_harness_circularity_and_determinism():
    eps = V.synth_corpus(seed=1, n_episodes=40)
    a = V.build_scorecard(eps)
    b = V.build_scorecard(V.synth_corpus(seed=1, n_episodes=40))
    import json
    assert json.dumps(a["2_conflation"]) == json.dumps(b["2_conflation"])
    circ = a["4_circularity_leakage"]
    assert circ["target_is_fed_command_derivation"] is True
    assert circ["command_echo_route_acc"] == 1.0
    assert circ["route_skill_vs_chance_of_command_echo"] == 0.0
    # the base rate is the route-from-vision null accuracy
    assert circ["route_from_vision_null_acc"] == circ["route_base_rate_majority"]


def test_harness_threshold_sensitivity_v2_more_stable():
    eps = V.synth_corpus(seed=2, n_episodes=60)
    s3 = V.build_scorecard(eps)["3_threshold_sensitivity"]
    v1t = [b["turn_frac"] for b in s3["v1_vs_turn_threshold"].values()]
    v2t = [b["turn_frac"] for b in s3["v2_vs_radii"].values()]
    # v2 turn-fraction varies far less across its knob than v1 across the degree
    # cut (curvature is physical -> junctions/road-curves are well-separated)
    assert (max(v2t) - min(v2t)) < 0.5 * (max(v1t) - min(v1t))
