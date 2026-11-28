"""Trivial kinematic-baseline trajectory floors for open-loop ego-motion eval.

The "honest denominator" for the Phase-0 driving-capability diagnostic
(`Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md`). The diagnostic's headline
verdict -- "the model is 10-15x worse than constant-velocity everywhere" -- rests
on a SINGLE trivial baseline (constant-velocity) measured on comma highway. This
module makes that denominator (a) reusable, (b) tested on analytic ground truth,
and (c) *stratified*, because the community standard NAVSIM v2 already uses a
constant-velocity agent as a triviality FILTER (it removes frames a CV agent
solves with PDMS>0.8, arXiv 2506.04218). If CV's dominance is corpus- and
curvature-dependent, then the diagnostic's skill denominator must be stratified,
not a single scalar.

Three canonical kinematic baselines (the standard simple predictors in the
trajectory-prediction literature -- CV / CA / CTRV, e.g. arXiv 2503.03262):

  * constant-velocity (CV)  -- world velocity vector held constant (captures any
                               lateral drift in the instantaneous heading).
  * go-straight            -- constant heading + constant speed: pure forward
                               motion along the current heading (the "keep the
                               wheel straight" null).
  * CTRV                    -- constant turn-rate + velocity: a circular arc;
                               the correct null on curved roads (arXiv 2503.03262
                               "suitable for predicting paths on curved roads").

All predictions and ground truth are expressed in the **ego frame at the anchor
time** (FLU: +x forward, +y left), so an ADE is a metric displacement error in
metres. Everything is causal: a predictor at time t uses only samples <= t.

Pose convention (shared `stack` contract, `cosmos_drive.poses_to_signals`):
`poses[:, :4] = [x_world, y_world, yaw_flu, speed]`, sampled at a fixed rate.

Pure numpy, no torch / no model -- runs on CPU in milliseconds.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #
def _ego_frame(dx: np.ndarray, dy: np.ndarray, yaw: float) -> np.ndarray:
    """World displacement (dx,dy) -> ego frame at heading `yaw` (FLU x-forward).

    Returns [...,2] = (forward, left).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = c * dx + s * dy
    lat = -s * dx + c * dy
    return np.stack([fwd, lat], axis=-1)


def gt_future_ego(x: np.ndarray, y: np.ndarray, yaw: np.ndarray,
                  t: int, ks: np.ndarray) -> np.ndarray:
    """Ground-truth future waypoints in the ego frame at anchor t. [K,2] metres."""
    dx = x[t + ks] - x[t]
    dy = y[t + ks] - y[t]
    return _ego_frame(dx, dy, float(yaw[t]))


# --------------------------------------------------------------------------- #
# Baseline predictors (all return [K,2] ego-frame forward/left, metres)        #
# --------------------------------------------------------------------------- #
def predict_cv_ego(x, y, yaw, t, dt, ks) -> np.ndarray:
    """Constant world-velocity, expressed in the ego frame at t.

    Backward-diff velocity at t held constant; captures any lateral component of
    the instantaneous heading (velocity direction need not equal yaw).
    """
    vx = (x[t] - x[t - 1]) / dt
    vy = (y[t] - y[t - 1]) / dt
    ev = _ego_frame(np.array([vx]), np.array([vy]), float(yaw[t]))[0]  # (vf, vl)
    tau = (ks * dt)[:, None]
    return ev[None, :] * tau


def predict_go_straight(speed_t, dt, ks) -> np.ndarray:
    """Constant heading + constant speed: pure forward motion."""
    tau = ks * dt
    fwd = speed_t * tau
    lat = np.zeros_like(fwd)
    return np.stack([fwd, lat], axis=-1)


def predict_ctrv(speed_t, omega_t, dt, ks, eps: float = 1e-4) -> np.ndarray:
    """Constant turn-rate + velocity -> circular arc in the ego frame at t."""
    tau = ks * dt
    if abs(omega_t) < eps:
        fwd = speed_t * tau
        lat = np.zeros_like(fwd)
    else:
        dtheta = omega_t * tau
        r = speed_t / omega_t
        fwd = r * np.sin(dtheta)
        lat = r * (1.0 - np.cos(dtheta))
    return np.stack([fwd, lat], axis=-1)


# --------------------------------------------------------------------------- #
# Errors                                                                        #
# --------------------------------------------------------------------------- #
def ade(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """Average displacement error over the horizon (mean L2 across steps)."""
    return float(np.mean(np.linalg.norm(pred_xy - gt_xy, axis=-1)))


def fde(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """Final displacement error (L2 at the last horizon step)."""
    return float(np.linalg.norm(pred_xy[-1] - gt_xy[-1]))


def skill_score(model_ade: float, baseline_ades: dict[str, float]) -> float:
    """model_ade / best trivial-baseline ADE. >1 = worse than the best null."""
    best = min(baseline_ades.values())
    return float(model_ade / best) if best > 0 else float("inf")


# --------------------------------------------------------------------------- #
# Stratification                                                                #
# --------------------------------------------------------------------------- #
def curvature_stratum(future_turn_deg: float,
                      gentle: float = 2.0, sharp: float = 10.0) -> str:
    """Classify an anchor by |heading change| over the 1 s lookahead (degrees)."""
    a = abs(future_turn_deg)
    if a < gentle:
        return "straight"
    if a < sharp:
        return "gentle"
    return "sharp"


def speed_stratum(v: float, lo: float = 10.0, hi: float = 20.0) -> str:
    if v < lo:
        return "low"
    if v < hi:
        return "mid"
    return "high"


# --------------------------------------------------------------------------- #
# Sequence evaluation                                                           #
# --------------------------------------------------------------------------- #
def evaluate_sequence(poses: np.ndarray, dt: float,
                      horizons_s=(1.0, 2.0), min_speed: float = 2.0) -> list[dict]:
    """Per-anchor baseline ADE/FDE for one ego pose sequence.

    `poses[:, :4] = [x, y, yaw, v]` at rate 1/dt Hz. Returns one record per valid
    anchor with all three baselines' ADE/FDE at each horizon, plus the anchor's
    speed and 1 s future heading change (for stratification). Causal: uses only
    the backward difference at t, never future samples, to form the prediction.

    `min_speed` (m/s): anchors slower than this are tagged `curv_stratum =
    "standstill"` regardless of heading change. Curvature = yaw_rate / speed is
    singular as v -> 0 (the same singularity `poses_to_signals` clips for
    steering): at a standstill, GNSS/INS heading jitter produces spurious huge
    yaw-rates with ~0 displacement, which would otherwise pollute the "sharp"
    stratum with trivially-predictable near-zero-motion anchors. Speed-gating the
    curvature stratum is REQUIRED for an honest per-curvature floor.
    """
    x = poses[:, 0].astype(np.float64)
    y = poses[:, 1].astype(np.float64)
    yaw = np.unwrap(poses[:, 2].astype(np.float64))
    T = len(x)
    hsteps = {h: int(round(h / dt)) for h in horizons_s}
    hmax = max(hsteps.values())
    h1 = hsteps[min(horizons_s)]  # shortest horizon defines curvature lookahead
    out = []
    for t in range(1, T - hmax):
        vx = (x[t] - x[t - 1]) / dt
        vy = (y[t] - y[t - 1]) / dt
        speed_t = float(np.hypot(vx, vy))
        omega_t = float((yaw[t] - yaw[t - 1]) / dt)
        future_turn_deg = float(np.degrees(yaw[t + h1] - yaw[t]))
        curv = ("standstill" if speed_t < min_speed
                else curvature_stratum(future_turn_deg))
        rec = {
            "t": t,
            "speed": speed_t,
            "future_turn_deg": future_turn_deg,
            "curv_stratum": curv,
            "speed_stratum": speed_stratum(speed_t),
        }
        for h, k in hsteps.items():
            ks = np.arange(1, k + 1)
            gt = gt_future_ego(x, y, yaw, t, ks)
            preds = {
                "cv": predict_cv_ego(x, y, yaw, t, dt, ks),
                "go_straight": predict_go_straight(speed_t, dt, ks),
                "ctrv": predict_ctrv(speed_t, omega_t, dt, ks),
            }
            for name, p in preds.items():
                rec[f"ade_{name}_{h:g}s"] = ade(p, gt)
                rec[f"fde_{name}_{h:g}s"] = fde(p, gt)
        out.append(rec)
    return out
