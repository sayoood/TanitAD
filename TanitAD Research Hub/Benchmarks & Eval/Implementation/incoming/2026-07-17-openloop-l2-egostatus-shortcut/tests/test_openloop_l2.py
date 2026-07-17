"""Analytic ground-truth sanity tests (G-B2) for the open-loop L2 protocol,
the kinematic baselines-to-3s, the ego-status shortcut regressor, and the
collision proxy. Every case has a hand-computed expected value.

Run standalone:  pytest tests/  (from the package dir)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openloop_l2 as ol  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers: build synthetic ego pose sequences with known motion               #
# --------------------------------------------------------------------------- #
def cv_sequence(T=60, dt=0.1, speed=20.0, yaw=0.3):
    """Exact constant-velocity straight motion at heading `yaw`."""
    x = np.zeros(T); y = np.zeros(T)
    for t in range(1, T):
        x[t] = x[t - 1] + speed * np.cos(yaw) * dt
        y[t] = y[t - 1] + speed * np.sin(yaw) * dt
    poses = np.stack([x, y, np.full(T, yaw), np.full(T, speed)], axis=1)
    return poses


# --------------------------------------------------------------------------- #
# horizon_step                                                                 #
# --------------------------------------------------------------------------- #
def test_horizon_step():
    assert ol.horizon_step(0.1, 1.0) == 10
    assert ol.horizon_step(0.1, 3.0) == 30
    assert ol.horizon_step(0.5, 2.0) == 4


# --------------------------------------------------------------------------- #
# l2_metrics: pointwise vs cumulative                                          #
# --------------------------------------------------------------------------- #
def test_l2_constant_per_step_error_conventions_agree():
    # every step off by exactly 2 m -> both conventions == 2.0 at every horizon
    K, dt = 30, 0.1
    gt = np.zeros((K, 2))
    pred = np.zeros((K, 2)); pred[:, 1] = 2.0
    m = ol.l2_metrics(pred, gt, dt, horizons=(1.0, 2.0, 3.0))
    for h in (1.0, 2.0, 3.0):
        assert m["pointwise"][h] == pytest.approx(2.0)
        assert m["cumulative"][h] == pytest.approx(2.0)
    assert m["avg_pointwise"] == pytest.approx(2.0)


def test_l2_growing_error_cumulative_below_pointwise():
    # error grows linearly with step -> cumulative (mean) < pointwise (endpoint)
    K, dt = 30, 0.1
    gt = np.zeros((K, 2))
    pred = np.zeros((K, 2))
    pred[:, 0] = np.arange(1, K + 1) * 0.1  # step k off by 0.1*k metres
    m = ol.l2_metrics(pred, gt, dt, horizons=(1.0, 2.0, 3.0))
    # pointwise@1s = 0.1*10 = 1.0 ; cumulative@1s = mean(0.1..1.0) = 0.55
    assert m["pointwise"][1.0] == pytest.approx(1.0)
    assert m["cumulative"][1.0] == pytest.approx(0.55)
    assert m["cumulative"][3.0] < m["pointwise"][3.0]


# --------------------------------------------------------------------------- #
# kinematic baselines to 3s: perfect on true CV motion, stop is not           #
# --------------------------------------------------------------------------- #
def test_kinematic_baselines_perfect_on_cv_motion():
    dt = 0.1
    poses = cv_sequence(T=60, dt=dt, speed=20.0, yaw=0.3)
    t, kmax = 20, 30
    preds = ol.kinematic_preds_full(poses, t, dt, kmax)
    ks = np.arange(1, kmax + 1)
    gt = ol.bp.gt_future_ego(poses[:, 0], poses[:, 1],
                             np.unwrap(poses[:, 2]), t, ks)
    for name in ("cv", "go_straight", "ctrv"):
        err = ol.l2_metrics(preds[name], gt, dt)["avg_pointwise"]
        assert err < 1e-6, f"{name} should be ~perfect on CV motion, got {err}"
    # stop predicts staying put -> error == forward displacement, large
    stop_err = ol.l2_metrics(preds["stop"], gt, dt)["pointwise"][1.0]
    assert stop_err == pytest.approx(20.0 * 1.0, rel=1e-6)  # 20 m/s * 1 s


# --------------------------------------------------------------------------- #
# ridge trajectory head: recovers an exact linear map                          #
# --------------------------------------------------------------------------- #
def test_ridge_recovers_linear_map():
    rng = np.random.RandomState(0)
    N, F, K = 400, 6, 5
    X = rng.randn(N, F)
    Wtrue = rng.randn(F, 2 * K)
    Y = X @ Wtrue
    head = ol.RidgeTrajectoryHead(lam=1e-6).fit(X, Y)
    pred = head.predict(X)
    assert pred.shape == (N, K, 2)
    assert np.allclose(pred.reshape(N, -1), Y, atol=1e-3)


def test_ego_status_shortcut_learns_cv():
    # a dataset of pure-CV sequences: the no-vision shortcut must predict the
    # future ego waypoints ~perfectly (ego status fully determines CV motion).
    dt, kmax = 0.1, 30
    seqs = [cv_sequence(T=60, dt=dt, speed=s, yaw=yw)
            for s in (8.0, 15.0, 25.0) for yw in (-0.2, 0.0, 0.2)]
    X, Y = [], []
    for poses in seqs:
        yawu = np.unwrap(poses[:, 2])
        for t in range(6, len(poses) - kmax):
            X.append(ol.ego_status_features(poses, t, dt))
            gt = ol.bp.gt_future_ego(poses[:, 0], poses[:, 1], yawu, t,
                                     np.arange(1, kmax + 1))
            Y.append(gt.reshape(-1))
    X, Y = np.asarray(X), np.asarray(Y)
    head = ol.RidgeTrajectoryHead(lam=1e-4).fit(X, Y)
    pred = head.predict(X).reshape(len(X), kmax, 2)
    errs = [ol.l2_metrics(pred[i], Y[i].reshape(kmax, 2), dt)["avg_pointwise"]
            for i in range(len(X))]
    assert np.median(errs) < 0.5, f"shortcut should nail CV, med={np.median(errs)}"


# --------------------------------------------------------------------------- #
# collision proxy                                                              #
# --------------------------------------------------------------------------- #
def test_collision_rate_hit_and_miss():
    traj = np.stack([np.arange(1, 11) * 2.0, np.zeros(10)], axis=1)  # 2..20 m fwd
    # a box straddling forward=10 m, on the path -> exactly one step (k=5, 10 m)
    box_hit = np.array([[10.0, 0.0, 0.5, 0.5]])
    r = ol.collision_rate(traj, box_hit, ego_half=(2.35, 0.95))
    assert r > 0.0
    # a box far to the left -> no collision
    box_miss = np.array([[10.0, 50.0, 0.5, 0.5]])
    assert ol.collision_rate(traj, box_miss) == 0.0
    # no boxes -> 0
    assert ol.collision_rate(traj, np.empty((0, 4))) == 0.0


# --------------------------------------------------------------------------- #
# skill_score                                                                  #
# --------------------------------------------------------------------------- #
def test_skill_score():
    assert ol.skill_score(2.0, 1.0) == pytest.approx(2.0)
    assert ol.skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert ol.skill_score(1.0, 0.0) == float("inf")
