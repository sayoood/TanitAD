"""Known-answer tests for e0bc_census.py (tier 2).  Analytic targets: a constant-curvature path, straight lines at 30 and 15 m/s, a pitch that alternates 0 / 0.5 deg."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0bc_census as C  # noqa: E402


def _traj(v, R=None, T=12.0, hz=100):
    t = np.arange(0, T, 1.0 / hz)
    if R is None:
        return t * 1e6, v * t, np.zeros_like(t), np.zeros_like(t)
    w = v / R
    return t * 1e6, R * np.sin(w * t), R * (1 - np.cos(w * t)), np.rad2deg(w * t)


def test_straight_30ms_ends_at_180m_and_15ms_at_90m():
    for v, xe in ((30.0, 180.0), (15.0, 90.0)):
        t, x, y, yaw = _traj(v)
        w = C.window_paths(t, x, y, yaw)
        assert abs(w[0][0] - xe) < 0.5 and w[0][1] < 1e-9


def test_circle_radius_100_at_20ms_matches_closed_form():
    t, x, y, yaw = _traj(20.0, R=100.0)
    ws = C.window_paths(t, x, y, yaw)
    x0, y0 = ws[0]
    xe, ym = ws[3]                       # a window that starts mid-curve: the ego frame must make it identical to the first
    assert abs(xe - x0) < 0.2 and abs(ym - y0) < 0.2
    assert abs(x0 - 100 * np.sin(20 * 6 / 100)) < 0.3 and abs(y0 - 100 * (1 - np.cos(20 * 6 / 100))) < 0.3     # literal: R sin(vT/R), R (1 - cos(vT/R))


def test_pitch_alternating_0_and_half_degree_gives_quarter_degree_and_ignores_yaw_and_roll():
    from scipy.spatial.transform import Rotation as R
    n = 400
    pitch = np.where(np.arange(n) % 2 == 0, 0.0, 0.5)
    yaw = np.full(n, 37.0)               # an arbitrary heading must not leak into pitch
    roll = np.full(n, 0.2)
    q = R.from_euler("ZYX", np.stack([yaw, pitch, roll], 1), degrees=True).as_quat()
    dp, dr = C.pitch_roll_dev(q)
    assert np.allclose(np.abs(dp), 0.25, atol=1e-6) and np.allclose(dr, 0.0, atol=1e-6)


def test_frame_tilt_times_yaw_fools_the_clip_mean_deviation_but_not_the_3s_highpass():
    """A rig that only YAWS (10 deg/s, 20 s), expressed in a world frame tilted by 1.7 deg: the true pitch deviation is exactly 0, but ZYX pitch swings +-1.7 deg with the yaw."""
    from scipy.spatial.transform import Rotation as R
    t = np.arange(0, 20, 0.01)
    yaw = R.from_euler("z", (10.0 * t)[:, None], degrees=True)
    q = (R.from_euler("x", 1.7, degrees=True) * yaw).as_quat()
    dp, _ = C.pitch_roll_dev(q)
    hp, _ = C.pitch_roll_dev_hp(q)
    assert np.quantile(np.abs(dp), 0.95) > 1.0         # the clip-mean statistic reads a spurious ~1.6 deg (literal bound: > 1 deg)
    assert np.quantile(np.abs(hp[200:-200]), 0.95) < 0.15   # the 3 s high-pass removes it (literal bound: < 0.15 deg, interior samples)
