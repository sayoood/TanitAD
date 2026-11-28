"""Community-standard open-loop L2 protocol + the ego-status "shortcut" ceiling.

Why this exists (Benchmarks & Eval mission: own the leaderboard + honest
placement). The Phase-0 driving diagnostic
(`Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md`) reports our D1 ADE in a
**camera-frame** target space and against a **best-of-3 kinematic floor**
(`../2026-07-15-baseline-floor/`). Neither is directly comparable to the number
the outside world ranks driving models on: **nuScenes-style open-loop L2**
(metric BEV, ego frame, waypoints at 0.5 s, reported at 1/2/3 s), which UniAD /
VAD / ST-P3 / AD-MLP all quote. This module closes three gaps at once:

1. **Protocol.** Implements L2 in the ego frame in **metres** under BOTH reported
   averaging conventions (they disagree by ~2x and papers rarely say which):
     * `pointwise`  (UniAD): L2 at *exactly* t seconds.
     * `cumulative` (ST-P3 / VAD): mean of the per-step L2 up to t seconds.
   plus the `avg` over {1,2,3} s. Reporting both is itself the honest fix for a
   known benchmark ambiguity.

2. **The ego-status shortcut (AD-MLP, arXiv 2312.03031 "Is Ego Status All You
   Need").** ~74 % of nuScenes is straight cruising, so a model that sees *no
   pixels* and regresses future waypoints from **ego status alone** (speed /
   accel / yaw-rate history) scores L2 ~= 0.29 m and beats most sensor models.
   Any open-loop L2 that is not reported *beside* this no-vision ceiling is
   uninterpretable. We fit that shortcut (ridge, closed-form, held-out by clip)
   on our own corpora so every future TanitAD L2 has its shortcut denominator.
   The kinematic floor (CV/CTRV) and the learned ego-status shortcut and NAVSIM
   v2's constant-velocity triviality filter are the SAME idea from three
   communities; this module makes them one number.

3. **skill_score.** `model_L2 / shortcut_L2` (and `/ kinematic_floor`) -- the
   scale-free, distribution-aware denominator the D1 gate should report instead
   of a raw metre value. Re-exported from the floor package.

Everything is metric-BEV ego frame (FLU: +x forward, +y left), causal (a
predictor at t uses only samples <= t), pure numpy. No torch, no model weights.

Frame / pose convention (shared `stack` contract): `poses[:, :4] =
[x_world, y_world, yaw_flu, speed]` at a fixed rate 1/dt.
"""
from __future__ import annotations

import numpy as np

# Kinematic baselines live in the floor package; vendored alongside for a
# standalone `pytest`. Source of truth: ../2026-07-15-baseline-floor/.
import baseline_predictors as bp

DEFAULT_HORIZONS = (1.0, 2.0, 3.0)


# --------------------------------------------------------------------------- #
# L2 under both community conventions                                          #
# --------------------------------------------------------------------------- #
def horizon_step(dt: float, h: float) -> int:
    """1-indexed step count for a horizon of `h` seconds at rate 1/dt."""
    return int(round(h / dt))


def l2_metrics(pred_full: np.ndarray, gt_full: np.ndarray, dt: float,
               horizons=DEFAULT_HORIZONS) -> dict:
    """L2 at each horizon under both conventions, from full per-step traj.

    `pred_full`, `gt_full`: [K, 2] ego-frame waypoints for steps 1..K at rate
    1/dt (K must cover max(horizons)). Returns:
      pointwise[h]  = ||pred_k - gt_k||  at k = round(h/dt)           (UniAD)
      cumulative[h] = mean_{j<=k} ||pred_j - gt_j||                   (ST-P3/VAD)
      avg_pointwise / avg_cumulative = mean over horizons.
    """
    step_l2 = np.linalg.norm(pred_full - gt_full, axis=-1)  # [K]
    point, cumul = {}, {}
    for h in horizons:
        k = horizon_step(dt, h)
        idx = k - 1  # 1-indexed horizon -> 0-indexed array
        point[h] = float(step_l2[idx])
        cumul[h] = float(np.mean(step_l2[:k]))
    return {
        "pointwise": point,
        "cumulative": cumul,
        "avg_pointwise": float(np.mean(list(point.values()))),
        "avg_cumulative": float(np.mean(list(cumul.values()))),
    }


# --------------------------------------------------------------------------- #
# Full-horizon kinematic baseline trajectories (to max horizon)               #
# --------------------------------------------------------------------------- #
def kinematic_preds_full(poses: np.ndarray, t: int, dt: float, kmax: int) -> dict:
    """Each trivial baseline's full [kmax, 2] ego-frame prediction at anchor t.

    Adds `stop` (predict staying put -> the zero trajectory), the degenerate
    null that bounds the metric from above.
    """
    x = poses[:, 0].astype(np.float64)
    y = poses[:, 1].astype(np.float64)
    yaw = np.unwrap(poses[:, 2].astype(np.float64))
    ks = np.arange(1, kmax + 1)
    vx = (x[t] - x[t - 1]) / dt
    vy = (y[t] - y[t - 1]) / dt
    speed_t = float(np.hypot(vx, vy))
    omega_t = float((yaw[t] - yaw[t - 1]) / dt)
    return {
        "stop": np.zeros((kmax, 2)),
        "go_straight": bp.predict_go_straight(speed_t, dt, ks),
        "cv": bp.predict_cv_ego(x, y, yaw, t, dt, ks),
        "ctrv": bp.predict_ctrv(speed_t, omega_t, dt, ks),
    }


# --------------------------------------------------------------------------- #
# The ego-status shortcut (AD-MLP repro): a NO-VISION learned predictor        #
# --------------------------------------------------------------------------- #
def ego_status_features(poses: np.ndarray, t: int, dt: float, hist: int = 5) -> np.ndarray:
    """Ego-status-only feature vector at anchor t (NO position, NO pixels).

    The information an AD-MLP-style shortcut is allowed: recent speed, forward
    acceleration, and yaw-rate over the last `hist` steps, expressed as motion
    quantities that are translation/heading invariant (so the model cannot cheat
    by memorising a world location). Everything derived causally from poses.
    """
    yaw = np.unwrap(poses[:, 2].astype(np.float64))
    x = poses[:, 0].astype(np.float64)
    y = poses[:, 1].astype(np.float64)
    feats = [1.0]  # bias
    for j in range(hist):
        tt = t - j
        if tt - 1 < 0:
            feats += [0.0, 0.0]
            continue
        vx = (x[tt] - x[tt - 1]) / dt
        vy = (y[tt] - y[tt - 1]) / dt
        speed = float(np.hypot(vx, vy))
        omega = float((yaw[tt] - yaw[tt - 1]) / dt)
        feats += [speed, omega]
    # forward acceleration over the last step
    if t - 2 >= 0:
        s0 = float(np.hypot((x[t - 1] - x[t - 2]) / dt, (y[t - 1] - y[t - 2]) / dt))
        s1 = float(np.hypot((x[t] - x[t - 1]) / dt, (y[t] - y[t - 1]) / dt))
        feats.append((s1 - s0) / dt)
    else:
        feats.append(0.0)
    return np.asarray(feats, dtype=np.float64)


class RidgeTrajectoryHead:
    """Closed-form ridge from ego-status features -> flattened ego waypoints.

    Deterministic (no torch, no seed). `fit` solves (X'X + lam I) W = X'Y.
    """

    def __init__(self, lam: float = 10.0):
        self.lam = float(lam)
        self.W: np.ndarray | None = None
        self.kmax: int | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeTrajectoryHead":
        F = X.shape[1]
        A = X.T @ X + self.lam * np.eye(F)
        self.W = np.linalg.solve(A, X.T @ Y)
        self.kmax = Y.shape[1] // 2
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """[N, F] -> [N, kmax, 2] ego-frame waypoints."""
        assert self.W is not None, "fit first"
        yhat = X @ self.W
        return yhat.reshape(yhat.shape[0], self.kmax, 2)


# --------------------------------------------------------------------------- #
# Collision proxy (tested primitive; used where agent boxes exist)            #
# --------------------------------------------------------------------------- #
def collision_rate(traj_ego: np.ndarray, obstacle_boxes: np.ndarray,
                   ego_half=(2.35, 0.95)) -> float:
    """Fraction of horizon steps whose ego footprint overlaps any obstacle box.

    `traj_ego`: [K, 2] ego-frame (forward, left) waypoints (heading assumed
    aligned with motion -> axis-aligned ego footprint, a standard open-loop
    collision proxy). `obstacle_boxes`: [M, 4] = (cx, cy, half_fwd, half_lat) in
    the SAME ego frame at the anchor. `ego_half`: (half_length, half_width) m.
    Returns collisions / K in [0, 1]. Axis-aligned overlap (a conservative,
    orientation-free proxy; the closed-loop CARLA path measures true OBB).
    """
    if obstacle_boxes is None or len(obstacle_boxes) == 0:
        return 0.0
    ehf, ehl = float(ego_half[0]), float(ego_half[1])
    hits = 0
    for k in range(traj_ego.shape[0]):
        fx, lx = float(traj_ego[k, 0]), float(traj_ego[k, 1])
        for (cx, cy, hf, hl) in obstacle_boxes:
            if abs(fx - cx) <= (ehf + hf) and abs(lx - cy) <= (ehl + hl):
                hits += 1
                break
    return hits / traj_ego.shape[0]


# --------------------------------------------------------------------------- #
# skill_score (re-exported from the floor package for one import site)         #
# --------------------------------------------------------------------------- #
def skill_score(model_l2: float, denom_l2: float) -> float:
    """model_L2 / denominator_L2. >1 = worse than the trivial/no-vision null."""
    return float(model_l2 / denom_l2) if denom_l2 > 0 else float("inf")
