"""Regression tests for the D-016 R1 egomotion SIGNAL upgrade (physicalai.py).

Pins the three signal-source fixes so a future refactor cannot silently revert
them (the pre-fix pipeline used d/dt(v) accel + atan2(vy,vx) yaw):
  (a) longitudinal accel == the dataset's OWN `ax` column, NOT a finite
      difference of speed (which differentiates interpolation noise and lags);
  (b) yaw == the ORIENTATION-quaternion heading (qx,qy,qz,qw), which stays
      correct at a standstill where atan2(vy,vx) is the direction of a ~zero
      vector (pure noise);
  (c) steer == atan(WHEELBASE * curvature) (unchanged, pinned for completeness);
and that per-timestep maneuver labels are derived and aligned to the poses.
CPU-only, synthetic egomotion (no dataset access).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels  # noqa: E402  (scripts/refb_labels.py — label vocabulary)
from tanitad.data.physicalai import (WHEELBASE, maneuvers_for_poses,  # noqa: E402
                                     quaternion_yaw, signals_at)


def _quat_level_yaw(psi: np.ndarray) -> tuple[np.ndarray, ...]:
    """Unit quaternion (qx,qy,qz,qw) for a LEVEL heading psi about +z."""
    return (np.zeros_like(psi), np.zeros_like(psi),
            np.sin(psi / 2.0), np.cos(psi / 2.0))


def _ego(n: int = 60, hz: float = 100.0, *, psi=None, vx=None, vy=None,
         ax=None, curv=None) -> pd.DataFrame:
    t = (np.arange(n) * (1e6 / hz)).astype(np.int64)          # microseconds
    if psi is None:
        psi = np.linspace(0.0, 0.4, n)
    if vx is None:
        vx = np.full(n, 10.0)
    if vy is None:
        vy = np.zeros(n)
    if ax is None:
        ax = np.linspace(-2.0, 2.0, n)                        # distinct ramp
    if curv is None:
        curv = np.full(n, 0.01)
    qx, qy, qz, qw = _quat_level_yaw(np.asarray(psi, float))
    return pd.DataFrame({
        "timestamp": t, "qx": qx, "qy": qy, "qz": qz, "qw": qw,
        "x": np.cumsum(vx) * 0.01, "y": np.cumsum(vy) * 0.01, "z": np.zeros(n),
        "vx": vx, "vy": vy, "vz": np.zeros(n),
        "ax": ax, "ay": np.zeros(n), "az": np.zeros(n),
        "curvature": curv})


def test_quaternion_yaw_matches_heading():
    psi = np.linspace(-3.0, 3.0, 41)
    qx, qy, qz, qw = _quat_level_yaw(psi)
    got = quaternion_yaw(qx, qy, qz, qw)
    assert np.allclose(np.unwrap(got), psi, atol=1e-9)


def test_accel_is_ax_not_finite_diff_of_v():
    # v is engineered to VARY so that d/dt(v) is clearly non-zero and clearly
    # unequal to the (independent) ax ramp -> proves the source is ax.
    n = 60
    vx = 10.0 + 3.0 * np.sin(np.linspace(0, 6, n))
    ax = np.linspace(-2.0, 2.0, n)
    ego = _ego(n=n, vx=vx, ax=ax)
    t = ego["timestamp"].to_numpy(np.float64)
    actions, _ = signals_at(ego, t)                           # query at samples
    accel = actions[:, 1]
    assert np.allclose(accel, ax, atol=1e-4)                  # == ax exactly
    # and it is NOT the finite difference of speed:
    v = np.hypot(vx, ego["vy"].to_numpy())
    dvdt = np.gradient(v, t * 1e-6)
    assert np.max(np.abs(accel - dvdt)) > 1.0                 # sources differ


def test_yaw_is_quaternion_and_standstill_robust():
    # Standstill: vx=vy=0 (so atan2(vy,vx) is meaningless) but the vehicle is
    # oriented at a KNOWN non-zero heading via the quaternion.
    n = 40
    psi = np.full(n, 0.7)
    ego = _ego(n=n, psi=psi, vx=np.zeros(n), vy=np.zeros(n))
    t = ego["timestamp"].to_numpy(np.float64)
    _, poses = signals_at(ego, t)
    yaw = poses[:, 2]
    assert np.allclose(yaw, 0.7, atol=1e-4)                   # quaternion heading
    # the legacy atan2(vy,vx) would have collapsed to 0 here:
    assert abs(np.arctan2(0.0, 0.0) - 0.7) > 0.5


def test_yaw_matches_velocity_direction_when_moving():
    # When genuinely moving, quaternion yaw and velocity direction agree.
    n = 40
    psi = np.linspace(0.0, 0.6, n)
    vx, vy = np.cos(psi) * 8.0, np.sin(psi) * 8.0
    ego = _ego(n=n, psi=psi, vx=vx, vy=vy)
    t = ego["timestamp"].to_numpy(np.float64)
    _, poses = signals_at(ego, t)
    assert np.max(np.abs(poses[:, 2] - psi)) < 1e-3


def test_steer_is_atan_wheelbase_curvature():
    ego = _ego(curv=np.full(60, 0.05))
    t = ego["timestamp"].to_numpy(np.float64)
    actions, _ = signals_at(ego, t)
    assert np.allclose(actions[:, 0], math.atan(WHEELBASE * 0.05), atol=1e-5)


def test_no_nan_inf_and_shapes():
    ego = _ego()
    t = ego["timestamp"].to_numpy(np.float64)
    actions, poses = signals_at(ego, t)
    assert actions.shape == (len(t), 2) and poses.shape == (len(t), 4)
    assert np.isfinite(actions).all() and np.isfinite(poses).all()


def test_maneuver_labels_aligned_and_tail_sentinel():
    H = refb_labels.LABEL_HORIZON
    T = H + 30
    # a clean left turn: yaw ramps up well past the turn threshold over H steps.
    yaw = np.linspace(0.0, 1.2, T)
    poses = np.zeros((T, 4), np.float32)
    poses[:, 2] = yaw
    poses[:, 3] = 10.0                                        # steady speed
    man = maneuvers_for_poses(torch.from_numpy(poses))
    assert man.shape == (T,) and man.dtype == torch.long
    assert (man[T - H:] == -1).all()                          # unlabelable tail
    # early steps see the turn within the horizon -> turn_left
    assert (man[:T - H] == refb_labels.TURN_LEFT).all()
    # identical to the trainer's on-the-fly window label at t=0
    win = refb_labels.window_maneuver_labels(
        torch.from_numpy(poses[:1]), torch.from_numpy(poses[None, 1:H + 1]))
    assert int(win[0]) == int(man[0])
