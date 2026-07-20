"""Tests for the v2.1 adaptive-horizon route labels (scripts/refb_labels.py).

The v2 labeler was measured on 80 PhysicalAI val episodes and had three
defects; this file pins the fix for each and guards the v2 path against
regression.

Pins:
  (A) v2 and v1 are UNCHANGED by the v2.1 addition (shipped runs stay
      reproducible).
  (B) D1 COVERAGE — a window with only a few seconds of future, which v2
      discarded via NAV_MIN_STEPS=150, is now JUDGED on the arc it actually
      travelled.
  (C) D2 NEVER-STRAIGHT — every unjudgeable window returns ROUTE_UNKNOWN, never
      ROUTE_STRAIGHT; ROUTE_UNKNOWN <-> valid=False is an iff; the sentinel sits
      OUTSIDE the 3-class CE range so an unmasked loss fails loudly.
  (D) D3 net_dyaw IN THE DECISION — the 479 m-radius / 48 deg val case
      (ep_00069) that v2 labelled straight+valid becomes a turn, and reverts
      when the rule is switched off.
  (E) GRADED TARGET — mean_curv is horizon- and speed-invariant and is defined
      exactly where the discrete label is masked.
  (F) transience is arc-anchored and only gates when it can discriminate;
      direction stays wrap-robust; tight junctions still classify.
CPU-only, synthetic trajectories.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels as R  # noqa: E402


def _poses(T, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.0, yaw0=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, yaw0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _junction(T=400, sign=1.0, t0=150, dur=30, v_turn=8.0, R_m=15.0, v_cruise=12.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v_cruise
    for t in range(T):
        rows.append([x, y, yaw, v])
        turning = t0 <= t < t0 + dur
        yr = sign * v_turn / R_m if turning else 0.0
        v = v_turn if turning else v_cruise
        x += v * math.cos(yaw) * 0.1
        y += v * math.sin(yaw) * 0.1
        yaw += yr * 0.1
    return torch.tensor(rows, dtype=torch.float32)


# ep_00069's signature: R=479 m sweeping 48 deg of net heading over the 25 s
# route horizon. v2 calls this `road_following` -> ROUTE_STRAIGHT, valid=True.
def _wide_drift(T=251, R_m=479.0, deg=48.0, secs=25.0):
    yaw_rate = math.radians(deg) / secs
    return _poses(T, v0=yaw_rate * R_m, yaw_rate=yaw_rate)


# ---------- (A) v2 / v1 unchanged -------------------------------------------

def test_v2_and_v1_paths_unchanged_by_v21():
    curve = _poses(400, v0=28.0, yaw_rate=28.0 / 250.0)
    r2 = R.route_from_future(curve, 0)
    assert r2["route"] == R.ROUTE_STRAIGHT and r2["valid"]
    assert set(r2) == {"route", "valid", "ambiguous", "net_dyaw", "signed_curv",
                       "peak_kappa", "concentration"}          # no new keys
    short = _poses(40)
    assert R.route_from_future(short, 0)["route"] == R.ROUTE_STRAIGHT   # the bug
    assert R.nav_command_v2(short, 0) == (R.NAV_FOLLOW, False)
    assert R.route_target_v2(_junction(sign=-1.0), 0) == R.ROUTE_RIGHT
    assert R.nav_command(_poses(400, yaw_rate=0.06), 0) == (R.NAV_LEFT, True)


# ---------- (B) D1 coverage --------------------------------------------------

def test_short_window_is_judged_not_discarded():
    # 4 s of future at 8 m/s = 32 m of road: far below v2's 15 s floor, plenty
    # of arc to see that the road is straight.
    short = _poses(41, v0=8.0)
    assert not R.route_from_future(short, 0)["valid"]           # v2 discards
    r = R.route_from_future_v21(short, 0)
    assert r["valid"] and r["route"] == R.ROUTE_STRAIGHT
    assert r["reason"] == "road_following"
    assert r["h_steps"] == 40 and 30.0 < r["arc_m"] < 34.0


def test_late_clip_window_of_a_200_frame_clip_is_judged():
    # Sayed's screenshot case: f113 of a ~199-frame PhysicalAI clip. v2 needs
    # 150 steps of future and only 85 exist -> unlabeled -> silently "straight".
    clip = _junction(T=199, sign=-1.0, t0=120, dur=30, v_turn=8.0, R_m=15.0)
    assert not R.route_from_future(clip, 113)["valid"]
    r = R.route_from_future_v21(clip, 113)
    assert r["valid"] and r["route"] == R.ROUTE_RIGHT


def test_coverage_rises_across_a_short_clip():
    clip = _junction(T=199, sign=1.0, t0=60, dur=30)
    ts = range(0, 190, 5)
    cov2 = sum(R.route_from_future(clip, t)["valid"] for t in ts)
    cov21 = sum(R.route_from_future_v21(clip, t)["valid"] for t in ts)
    assert cov2 / len(list(ts)) < 0.35          # v2: only the clip head
    assert cov21 / len(list(ts)) > 0.9          # v2.1: essentially everything


# ---------- (C) D2 never default to straight --------------------------------

def test_unjudgeable_returns_unknown_never_straight():
    stopped = _poses(300, v0=0.0)                       # no arc at all
    r = R.route_from_future_v21(stopped, 0)
    assert r["route"] == R.ROUTE_UNKNOWN and not r["valid"]
    assert r["reason"] == "no_arc" and not r["ambiguous"]
    tail = _poses(60, v0=10.0)
    r_tail = R.route_from_future_v21(tail, 59)          # literally no future
    assert r_tail["route"] == R.ROUTE_UNKNOWN and r_tail["reason"] == "no_future"
    # ... and v2 answered "straight" to both of these
    assert R.route_from_future(stopped, 0)["route"] == R.ROUTE_STRAIGHT
    assert R.route_from_future(tail, 59)["route"] == R.ROUTE_STRAIGHT


def test_unknown_iff_invalid_and_sentinel_is_outside_ce_range():
    from tanitad.refs.refb import ROUTE_CLASSES
    assert R.ROUTE_UNKNOWN == len(ROUTE_CLASSES)        # out of CE range: loud
    cases = [_poses(300, v0=0.0), _poses(41), _wide_drift(),
             _junction(sign=1.0), _junction(sign=-1.0),
             _poses(400, v0=28.0, yaw_rate=28.0 / 250.0),
             _junction(T=400, t0=150, dur=30, v_turn=24.0, R_m=100.0)]
    seen_unknown = seen_valid = 0
    for p in cases:
        for t in range(0, p.shape[0], 13):
            r = R.route_from_future_v21(p, t)
            assert (r["route"] == R.ROUTE_UNKNOWN) == (not r["valid"])
            assert r["route"] in (R.ROUTE_LEFT, R.ROUTE_STRAIGHT,
                                  R.ROUTE_RIGHT, R.ROUTE_UNKNOWN)
            seen_unknown += r["route"] == R.ROUTE_UNKNOWN
            seen_valid += r["valid"]
    assert seen_unknown > 0 and seen_valid > 0          # both branches exercised


def test_route_target_v21_returns_target_and_mask_together():
    tgt, valid = R.route_target_v21(_poses(300, v0=0.0), 0)
    assert tgt == R.ROUTE_UNKNOWN and valid is False
    tgt2, valid2 = R.route_target_v21(_junction(sign=1.0), 0)
    assert tgt2 == R.ROUTE_LEFT and valid2 is True
    # nav_cmd (a model INPUT) stays inside NAV_COMMANDS even when unknown
    nav, nv = R.nav_command_v21(_poses(300, v0=0.0), 0)
    assert nav == R.NAV_FOLLOW and nv is False


# ---------- (D) D3 net_dyaw in the decision ---------------------------------

def test_wide_radius_drift_is_a_turn_under_v21():
    drift = _wide_drift()                               # R=479 m, 48 deg net
    r2 = R.route_from_future(drift, 0)
    assert r2["route"] == R.ROUTE_STRAIGHT and r2["valid"]      # v2: the bug
    r = R.route_from_future_v21(drift, 0)
    assert r["route"] == R.ROUTE_LEFT and r["valid"]
    assert r["reason"] == "net_heading"
    assert math.degrees(abs(r["net_dyaw"])) > 45.0
    assert r["peak_kappa"] < R.CURV_ROAD_PER_M          # genuinely gentle radius
    right = R.route_from_future_v21(_poses(251, v0=16.0, yaw_rate=-0.0335), 0)
    assert right["route"] == R.ROUTE_RIGHT


def test_net_heading_rule_is_switchable_back_to_v2_semantics():
    drift = _wide_drift()
    r = R.route_from_future_v21(drift, 0, use_net_dyaw=False)
    assert r["route"] == R.ROUTE_STRAIGHT and r["reason"] == "road_following"


def test_gentle_curve_below_the_net_threshold_stays_straight():
    # 20 deg of net heading at road radius is still road-following.
    gentle = _poses(251, v0=16.0, yaw_rate=math.radians(20.0) / 25.0)
    r = R.route_from_future_v21(gentle, 0)
    assert r["route"] == R.ROUTE_STRAIGHT and r["valid"]
    assert r["reason"] == "road_following"


# ---------- (E) graded target ------------------------------------------------

def test_mean_curv_is_horizon_and_speed_invariant():
    # 1/R recovered from 5 s to 25 s of future, at 6 to 30 m/s. Also covers the
    # >180 deg cases (R=60 @250 steps sweeps 5 rad, R=25 sweeps 6 rad): the
    # cumulative net_dyaw must NOT wrap, which the endpoint difference does.
    for R_m, v in ((300.0, 30.0), (60.0, 12.0), (25.0, 6.0)):
        p = _poses(300, v0=v, yaw_rate=v / R_m)
        vals = [R.route_from_future_v21(p, 0, horizon_steps=h)["mean_curv"]
                for h in (50, 100, 175, 250)]
        for mc in vals:
            assert abs(mc - 1.0 / R_m) < 0.12 / R_m         # within 12 %
        assert max(vals) - min(vals) < 0.1 / R_m            # horizon-invariant


def test_graded_target_honest_floor_below_min_arc():
    # The graded target is NOT defined below the arc floor: mean_curv =
    # net_dyaw/arc explodes as arc -> 0, so a near-stationary window reports
    # UNKNOWN with a zeroed soft target rather than a huge fabricated curvature.
    crawl = _poses(300, v0=1.5, yaw_rate=1.5 / 25.0)        # 4.5 m over 3 s
    r = R.route_from_future_v21(crawl, 0, horizon_steps=30)
    assert r["reason"] == "no_arc" and r["arc_m"] < R.MIN_ARC_ROUTE_M
    assert r["mean_curv"] == 0.0 and r["graded_route"] == 0.0
    # give it enough road and the same trajectory is judged
    r2 = R.route_from_future_v21(crawl, 0, horizon_steps=250)
    assert r2["arc_m"] > R.MIN_ARC_ROUTE_M and r2["valid"]


def test_graded_target_defined_where_discrete_label_is_masked():
    # gray-zone window: CE target is UNKNOWN/masked, the soft target still
    # carries the signed tightness (this is what R4 buys).
    fork = _junction(T=400, t0=150, dur=30, v_turn=24.0, R_m=100.0)
    r = R.route_from_future_v21(fork, 0)
    assert r["route"] == R.ROUTE_UNKNOWN and r["ambiguous"]
    assert r["reason"] == "gray_zone"
    mc, nd = R.route_graded_target(fork, 0)
    assert mc > 0.0 and nd > 0.0                        # left-ish, quantified
    assert 0.0 < r["graded_route"] < 1.0
    assert abs(r["graded_route"] - math.tanh(mc / R.CURV_TURN_PER_M)) < 1e-6


def test_graded_route_sign_matches_direction():
    left = R.route_from_future_v21(_junction(sign=1.0), 0)
    right = R.route_from_future_v21(_junction(sign=-1.0), 0)
    assert left["graded_route"] > 0.05 and right["graded_route"] < -0.05


# ---------- (F) transience, direction, tight junctions ----------------------

def test_tight_junction_and_road_sweep_still_separate():
    j = R.route_from_future_v21(_junction(sign=1.0), 0)
    assert j["route"] == R.ROUTE_LEFT and j["reason"] == "tight_transient"
    assert j["peak_kappa"] > R.CURV_TURN_PER_M
    sweep = R.route_from_future_v21(
        _poses(400, v0=28.0, yaw_rate=28.0 / 900.0), 0)     # R=900 m, 44 deg
    assert sweep["route"] == R.ROUTE_STRAIGHT


def test_transience_gate_only_applies_when_measurable():
    long_arc = R.route_from_future_v21(_junction(sign=1.0), 0)
    assert long_arc["transience_measurable"]
    assert long_arc["concentration"] > R.CONCENTRATION_MIN
    # a 3 s window at 8 m/s = 24 m of road: no sustained alternative can fit,
    # so the gate is skipped and tightness alone decides.
    tight_short = _poses(31, v0=8.0, yaw_rate=8.0 / 15.0)
    r = R.route_from_future_v21(tight_short, 0)
    assert not r["transience_measurable"]
    assert r["route"] == R.ROUTE_LEFT and r["reason"] == "tight_transient"


def test_direction_robust_to_yaw_wrap_over_180_degrees():
    loop_r = _poses(400, v0=8.0, yaw_rate=-8.0 / 20.0)      # R=20 m roundabout
    assert R.route_from_future_v21(loop_r, 0)["route"] == R.ROUTE_RIGHT
    loop_l = _poses(400, v0=8.0, yaw_rate=8.0 / 20.0)
    assert R.route_from_future_v21(loop_l, 0)["route"] == R.ROUTE_LEFT


def test_arc_concentration_separates_junction_from_sustained_sweep():
    j = R.route_from_future_v21(_junction(sign=1.0), 0)
    sweep = R.route_from_future_v21(_poses(400, v0=28.0, yaw_rate=28.0 / 250.0), 0)
    assert j["concentration"] > 2.0 * sweep["concentration"]


def test_determinism():
    p = _junction(sign=-1.0)
    a = R.route_from_future_v21(p, 40)
    b = R.route_from_future_v21(p, 40)
    assert a == b
