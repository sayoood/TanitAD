"""Tests for E4.1: tactical hindsight goals + 3-axis severity manoeuvres
(scripts/refb_labels.py, E4.1 section).

Pins:
  (1) constant-velocity straight line -> g_tac == (v*tau*DT, 0, 0, v) exactly
      (1e-4), all valid; maneuver3 == (STRAIGHT, KEEP, LANE keep) everywhere.
  (2) short episode -> out-of-range taus are valid=False, rows CLAMP to the
      last available pose, never NaN.
  (3) R=20 m left junction turn -> LAT SHARP_LEFT and g_tac heading positive.
  (4) large-radius highway curve at speed -> NOT sharp (the speed-invariance
      point): R=100 m lands GENTLE, R=200 m (beyond R_ROAD_M=150) STRAIGHT.
  (5) hard-brake profile -> LON HARD_BRAKE (plus the soft/accel rungs).
  (6) 3.5 m lateral lane change (heading returns ~0, low curvature) ->
      LANE change toward the correct side, LAT not sharp.
  (7) goal_tac_labels (vectorised) == per-t goal_tac_targets on a random
      episode, values and valid masks both.
CPU-only, synthetic trajectories.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels as R  # noqa: E402

DT = R.DT_DEFAULT


def _poses(T, dt=DT, v0=8.0, yaw_rate=0.0, accel=0.0, yaw0=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, yaw0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _poses_from_xy(x, y, dt=DT):
    """Poses [T, 4] from position arrays: yaw/v via finite differences."""
    x, y = torch.as_tensor(x, dtype=torch.float32), \
        torch.as_tensor(y, dtype=torch.float32)
    dx, dy = x[1:] - x[:-1], y[1:] - y[:-1]
    yaw = torch.atan2(dy, dx)
    yaw = torch.cat([yaw, yaw[-1:]])
    v = torch.sqrt(dx * dx + dy * dy) / dt
    v = torch.cat([v, v[-1:]])
    return torch.stack([x, y, yaw, v], dim=-1)


# ---------- (1) constant-velocity straight line ------------------------------

def test_straight_line_goals_exact_and_all_valid():
    v0 = 8.0
    p = _poses(100, v0=v0)
    g, valid = R.goal_tac_targets(p, 0)
    assert valid.all()
    for k, tau in enumerate(R.GOAL_TAC_TAUS_STEPS):
        assert abs(float(g[k, 0]) - v0 * tau * DT) < 1e-4   # x = v * tau * DT
        assert abs(float(g[k, 1])) < 1e-4                   # y = 0
        assert abs(float(g[k, 2])) < 1e-4                   # heading = 0
        assert abs(float(g[k, 3]) - v0) < 1e-4              # speed = v


def test_straight_line_maneuver3_all_neutral():
    m = R.maneuver3_labels(_poses(100))
    assert m.shape == (100, 3) and m.dtype == torch.long
    assert (m[:, 0] == R.LAT3_STRAIGHT).all()
    assert (m[:, 1] == R.LON3_KEEP).all()
    assert (m[:, 2] == R.LANE3_KEEP).all()


# ---------- (2) short episode: clamp, mask, no NaN ---------------------------

def test_short_episode_clamps_and_masks_no_nan():
    T = 30
    p = _poses(T, v0=6.0)
    g, valid = R.goal_tac_labels(p)                        # taus (20, 40, 60)
    assert g.shape == (T, 3, 4) and valid.shape == (T, 3)
    assert torch.isfinite(g).all()                          # NEVER NaN
    for t in range(T):
        for k, tau in enumerate(R.GOAL_TAC_TAUS_STEPS):
            assert bool(valid[t, k]) == (t + tau <= T - 1)
    # invalid rows clamp to the LAST pose: same ego-frame target as the last
    # reachable pose, computed from first principles
    t = 5
    exp_xy = R.ego_frame(p[-1, :2] - p[t, :2], p[t, 2])
    for k in (1, 2):                                        # taus 40, 60 invalid
        assert not bool(valid[t, k])
        assert torch.allclose(g[t, k, :2], exp_xy, atol=1e-5)
        assert abs(float(g[t, k, 3]) - float(p[-1, 3])) < 1e-5
    # maneuver3 on a short episode is also finite/defined for every row
    m = R.maneuver3_labels(p)
    assert m.shape == (T, 3)


# ---------- (3) tight junction turn ------------------------------------------

def test_junction_turn_sharp_left_and_heading_positive():
    p = _poses(100, v0=8.0, yaw_rate=8.0 / 20.0)           # R = 20 m, left
    m = R.maneuver3_labels(p)
    assert int(m[0, 0]) == R.LAT3_SHARP_LEFT
    g, valid = R.goal_tac_targets(p, 0)
    assert valid.all()
    assert (g[:, 2] > 0).all()                              # heading swings left
    # right turn mirrors
    pr = _poses(100, v0=8.0, yaw_rate=-8.0 / 20.0)
    assert int(R.maneuver3_labels(pr)[0, 0]) == R.LAT3_SHARP_RIGHT


# ---------- (4) highway curve at speed is NOT sharp (speed invariance) -------

def test_highway_curve_at_speed_is_not_sharp():
    # R = 100 m at 25 m/s: kappa = 0.01 in [1/150, 1/60) -> GENTLE. The same
    # net heading over time would read as a "turn" to a time-thresholded label
    # (the v2-header failure); curvature is speed-invariant so it stays gentle.
    p100 = _poses(100, v0=25.0, yaw_rate=25.0 / 100.0)
    lat100 = int(R.maneuver3_labels(p100)[0, 0])
    assert lat100 == R.LAT3_GENTLE_LEFT
    # R = 200 m at 30 m/s: beyond R_ROAD_M = 150 -> road-following, STRAIGHT
    p200 = _poses(100, v0=30.0, yaw_rate=30.0 / 200.0)
    lat200 = int(R.maneuver3_labels(p200)[0, 0])
    assert lat200 == R.LAT3_STRAIGHT
    for lat in (lat100, lat200):
        assert lat not in (R.LAT3_SHARP_LEFT, R.LAT3_SHARP_RIGHT)


# ---------- (5) longitudinal severity rungs ----------------------------------

def test_lon_severity_rungs():
    # hard brake: -3 m/s^2 sustained over the 4 s horizon
    assert int(R.maneuver3_labels(_poses(60, v0=16.0, accel=-3.0))[0, 1]) \
        == R.LON3_HARD_BRAKE
    # soft brake / soft accel / hard accel
    assert int(R.maneuver3_labels(_poses(60, v0=16.0, accel=-1.0))[0, 1]) \
        == R.LON3_BRAKE
    assert int(R.maneuver3_labels(_poses(60, v0=5.0, accel=1.0))[0, 1]) \
        == R.LON3_ACCEL
    assert int(R.maneuver3_labels(_poses(60, v0=5.0, accel=3.0))[0, 1]) \
        == R.LON3_HARD_ACCEL


# ---------- (6) lane change --------------------------------------------------

def _lane_change_poses(T=100, lat_m=3.5, v=15.0, t0=10, dur=30):
    """Smoothstep lateral shift of `lat_m` over `dur` steps at speed `v`:
    heading returns to ~0, curvature stays sub-junction."""
    ts = torch.arange(T, dtype=torch.float32)
    x = v * ts * DT
    u = ((ts - t0) / dur).clamp(0.0, 1.0)
    y = lat_m * (3 * u ** 2 - 2 * u ** 3)                  # smoothstep
    return _poses_from_xy(x, y)


def test_lane_change_toward_correct_side_lat_not_sharp():
    left = _lane_change_poses(lat_m=3.5)
    m = R.maneuver3_labels(left)
    assert int(m[0, 2]) == R.LANE3_CHANGE_LEFT             # +y = left
    assert int(m[0, 0]) not in (R.LAT3_SHARP_LEFT, R.LAT3_SHARP_RIGHT)
    right = _lane_change_poses(lat_m=-3.5)
    mr = R.maneuver3_labels(right)
    assert int(mr[0, 2]) == R.LANE3_CHANGE_RIGHT
    assert int(mr[0, 0]) not in (R.LAT3_SHARP_LEFT, R.LAT3_SHARP_RIGHT)
    # and a junction turn is NOT a lane change even though it displaces
    turn = _poses(100, v0=8.0, yaw_rate=8.0 / 20.0)
    assert int(R.maneuver3_labels(turn)[0, 2]) == R.LANE3_KEEP


# ---------- (7) vectorised == per-t ------------------------------------------

def test_goal_tac_labels_matches_per_t_targets():
    torch.manual_seed(0)
    T = 80
    xy = torch.cumsum(torch.randn(T, 2), dim=0)
    yaw = (torch.rand(T) - 0.5) * 4 * math.pi              # exercises wrapping
    v = torch.rand(T) * 20.0
    p = torch.cat([xy, yaw[:, None], v[:, None]], dim=1)
    g_all, valid_all = R.goal_tac_labels(p)
    for t in range(T):
        g_t, valid_t = R.goal_tac_targets(p, t)
        assert torch.equal(valid_all[t], valid_t)
        assert torch.allclose(g_all[t], g_t, atol=1e-5), f"mismatch at t={t}"
