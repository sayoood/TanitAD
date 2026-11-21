"""Analytic ground-truth sanity tests for the kinematic-baseline floor (G-B2).

Every case has a closed-form answer so the metric is validated without any real
data or model. Run: `pytest test_baseline_predictors.py -q`.
"""
import numpy as np
import pytest

import baseline_predictors as bp


DT = 0.1  # 10 Hz, matches comma2k19 (hz=10) and strided Cosmos (30fps/3)


def _straight_poses(v=25.0, n=60, yaw0=0.7):
    """Constant-speed straight line at a fixed world heading yaw0."""
    t = np.arange(n) * DT
    x = v * np.cos(yaw0) * t
    y = v * np.sin(yaw0) * t
    yaw = np.full(n, yaw0)
    return np.stack([x, y, yaw, np.full(n, v)], axis=1)


def _arc_poses(v=15.0, omega=0.3, n=60, yaw0=0.0):
    """Constant turn-rate + speed circular arc (the CTRV generative model)."""
    t = np.arange(n) * DT
    yaw = yaw0 + omega * t
    r = v / omega
    # integrate: position along an arc starting at origin, heading yaw0
    x = r * (np.sin(yaw) - np.sin(yaw0))
    y = -r * (np.cos(yaw) - np.cos(yaw0))
    return np.stack([x, y, yaw, np.full(n, v)], axis=1)


# --------------------------------------------------------------------------- #
# 1. Straight line -> all three baselines are exact                            #
# --------------------------------------------------------------------------- #
def test_straight_line_all_baselines_exact():
    poses = _straight_poses()
    recs = bp.evaluate_sequence(poses, DT)
    assert recs, "no anchors produced"
    for r in recs:
        for name in ("cv", "go_straight", "ctrv"):
            assert r[f"ade_{name}_1s"] < 1e-6, (name, r[f"ade_{name}_1s"])
            assert r[f"ade_{name}_2s"] < 1e-6
        assert r["curv_stratum"] == "straight"


# --------------------------------------------------------------------------- #
# 2. Circular arc -> CTRV exact; CV & go-straight fail predictably (chord<arc)  #
# --------------------------------------------------------------------------- #
def test_arc_ctrv_recovers_cv_fails():
    poses = _arc_poses(v=15.0, omega=0.3)
    recs = bp.evaluate_sequence(poses, DT)
    assert recs
    mid = recs[len(recs) // 2]
    # CTRV is the generative model -> essentially zero error
    assert mid["ade_ctrv_2s"] < 1e-3, mid["ade_ctrv_2s"]
    # CV and go-straight cannot follow the curve -> real error, and it grows
    assert mid["ade_cv_2s"] > 0.3
    assert mid["ade_go_straight_2s"] > mid["ade_ctrv_2s"]
    # longer horizon strictly worse than shorter for the wrong-model baselines
    assert mid["ade_go_straight_2s"] > mid["ade_go_straight_1s"]


def test_arc_is_stratified_as_turning():
    # omega=0.3 rad/s -> ~17 deg over 1s -> "sharp"
    poses = _arc_poses(v=15.0, omega=0.3)
    recs = bp.evaluate_sequence(poses, DT)
    assert all(r["curv_stratum"] == "sharp" for r in recs)
    # a gentle arc (~0.06 rad/s -> ~3.4 deg/s) lands in "gentle"
    recs_g = bp.evaluate_sequence(_arc_poses(v=15.0, omega=0.06), DT)
    assert all(r["curv_stratum"] == "gentle" for r in recs_g)


# --------------------------------------------------------------------------- #
# 2b. Standstill yaw-jitter is NOT mislabelled "sharp" (speed-gated curvature)  #
# --------------------------------------------------------------------------- #
def test_standstill_yaw_jitter_gated():
    # near-zero motion with a large jittering heading (the comma-val artifact):
    # curvature = yaw_rate / v is singular at v->0. Must land in "standstill",
    # not "sharp".
    rng = np.random.default_rng(0)
    n = 60
    x = np.cumsum(rng.normal(0, 0.001, n))      # ~stationary
    y = np.cumsum(rng.normal(0, 0.001, n))
    yaw = np.cumsum(rng.normal(0, 0.5, n))      # wild heading jitter
    v = np.hypot(np.gradient(x, DT), np.gradient(y, DT))
    poses = np.stack([x, y, yaw, v], axis=1)
    recs = bp.evaluate_sequence(poses, DT, min_speed=2.0)
    assert recs
    assert all(r["curv_stratum"] == "standstill" for r in recs), \
        {r["curv_stratum"] for r in recs}
    # a genuine high-speed arc is unaffected by the gate
    recs_arc = bp.evaluate_sequence(_arc_poses(v=15.0, omega=0.3), DT, min_speed=2.0)
    assert all(r["curv_stratum"] == "sharp" for r in recs_arc)


# --------------------------------------------------------------------------- #
# 3. Ego-frame transform matches a hand computation                            #
# --------------------------------------------------------------------------- #
def test_ego_frame_transform():
    # heading 90 deg (pointing +y world); a point 5 m ahead in world +y
    # should read as (forward=5, left=0) in the ego frame.
    ego = bp._ego_frame(np.array([0.0]), np.array([5.0]), np.pi / 2)[0]
    assert ego[0] == pytest.approx(5.0, abs=1e-9)
    assert ego[1] == pytest.approx(0.0, abs=1e-9)
    # a point 3 m to world +x with heading 90 deg is 3 m to the ego RIGHT (-left)
    ego2 = bp._ego_frame(np.array([3.0]), np.array([0.0]), np.pi / 2)[0]
    assert ego2[0] == pytest.approx(0.0, abs=1e-9)
    assert ego2[1] == pytest.approx(-3.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. skill_score arithmetic                                                    #
# --------------------------------------------------------------------------- #
def test_skill_score():
    base = {"cv": 0.5, "go_straight": 2.0, "ctrv": 0.8}
    # model equal to the best baseline (cv=0.5) -> skill 1.0
    assert bp.skill_score(0.5, base) == pytest.approx(1.0)
    # model 3x the best baseline -> 3.0
    assert bp.skill_score(1.5, base) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# 5. CV captures a lateral velocity component (not just forward speed)          #
# --------------------------------------------------------------------------- #
def test_cv_uses_full_velocity_vector():
    # crab motion: heading 0 but velocity has a +y (left) component.
    n = 40
    vx, vy = 20.0, 3.0
    t = np.arange(n) * DT
    x, y = vx * t, vy * t
    poses = np.stack([x, y, np.zeros(n), np.full(n, np.hypot(vx, vy))], axis=1)
    recs = bp.evaluate_sequence(poses, DT)
    mid = recs[len(recs) // 2]
    # CV models the sideways drift exactly (straight-line world motion) -> ~0
    assert mid["ade_cv_1s"] < 1e-6
    # go-straight ignores the lateral component -> nonzero error
    assert mid["ade_go_straight_1s"] > 0.2


# --------------------------------------------------------------------------- #
# 6. Determinism / shape contract                                              #
# --------------------------------------------------------------------------- #
def test_record_contract():
    recs = bp.evaluate_sequence(_straight_poses(), DT)
    r = recs[0]
    for key in ("t", "speed", "future_turn_deg", "curv_stratum", "speed_stratum",
                "ade_cv_1s", "fde_cv_1s", "ade_ctrv_2s", "ade_go_straight_2s"):
        assert key in r
