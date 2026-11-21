"""TanitAD custom evaluation metric suite (Benchmarks & Eval, backlog #1).

WHY THIS EXISTS
---------------
Standard open-loop metrics (ADE/FDE) and even NAVSIM's safety-aware EPDMS do not
isolate what TanitAD's world-model *edge* is supposed to buy: acting on an
occluded hazard's inferred continuation **before** it is visible, staying smooth
under partial observability, and doing so inside a hard edge-compute budget. The
2026 cross-benchmark study (arXiv 2605.00066) is blunt about this: displacement
metrics show *no reliable correlation* with closed-loop driving score, and even
NAVSIM PDMS correlates *non-monotonically* with Bench2Drive DS (ranking
inversions). So we need metrics purpose-built to measure the edges the recognized
KPIs miss. These five are the "Deep Think 14" definitions
(``Ressources/Deep Think Analysis/Deep Think 14.md``); each formula is reproduced
in its function docstring.

    LAL   Latent Anticipation Latency        t_LoS - t_anticipation  (s, >0 proactive)
    TMS   Tactical Maneuver Stability         1/(1 + a*integral|jerk| + b*integral|steer_rate|)
    OKRI  Occluded Kinematic Risk Integral    integral( (1/2 m v^2) / (d_blind+eps) ) dt  (lower safer)
    CNCE  Compute-Normalized Causal Efficacy  D_progress / (tau_infer * P_billions) * exp(-lambda*collisions)
    LOPS  Latent Object Permanence Score      mean_{t in occ} exp(-gamma * ||p_wm - p_gt||)

WHAT THIS IS AND IS NOT (honesty, P8)
-------------------------------------
LAL/OKRI/LOPS require *closed-loop scenario telemetry* (per-step ego kinematics,
occlusion geometry, world-model hidden-agent estimate, ground-truth hidden-agent
position). That telemetry comes from the scripted MetaDrive occluder scenarios
(Ghost Cut-Through / Blind Creep / Choke Weave) which are still gated on the
supervised MetaDrive source install (PROJECT_STATE W2). This module is the
*computation* those scenarios will call; it ships now with synthetic-ground-truth
sanity tests (G-B2) so the math is trustworthy the moment live logs exist. **No
metric is claimed on a real TanitAD run in this package** — the numbers here are
on synthetic fixtures with analytically known answers.

SEAM WITH THE GATE RUNNER (Architecture, 2026-07-14)
----------------------------------------------------
The D1-D3 gate runner (``eval/gates.py``, intake 2026-07-14) exposes an
``extra_metrics={name: callable}`` hook where each callable is ``(pred_xy,
true_xy) -> float`` over trajectory tensors. ``trajectory_extra_metrics()`` here
returns exactly that dict (rmse / miss-rate / n-eff) so the custom suite plugs
into the decode gates without either module importing the other. The five headline
metrics above operate on *telemetry*, not trajectory tensors — a different domain
— so they are exposed via ``run_scenario_suite`` / ``ScenarioTelemetry``, not the
seam. Proposed target on integration: ``stack/tanitad/eval/metrics.py`` (same
``eval/`` package the gate runner lands in).

Dependency-light on purpose: numpy only (already a torch transitive dep). Torch is
imported lazily *only* to accept torch tensors on the trajectory seam; the
telemetry metrics are pure numpy. Trapezoid integration is implemented locally to
avoid the ``np.trapz`` (<2.0) / ``np.trapezoid`` (>=2.0) rename.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Named constants (Deep Think 14 defaults; no magic numbers in the body).      #
# --------------------------------------------------------------------------- #
JERK_BRAKE_THRESHOLD = -1.5     # m/s^3 : prophylactic longitudinal braking onset
LAL_NO_REACTION = -999.0        # sentinel: hazard reached LoS but ego never braked
TMS_ALPHA = 1.0                 # weight on integral of |longitudinal jerk|
TMS_BETA = 1.5                  # weight on integral of |steering rate|
OKRI_APPROACH_M = 30.0          # only integrate risk while within this of a blind spot
OKRI_EPS = 0.1                  # metres, guards the 1/d_blind singularity
OKRI_SCALE = 1000.0             # report in kJ/m-scale (Deep Think 14 reference)
CNCE_LAMBDA = 2.0               # harsh collision penalty exponent
LOPS_GAMMA = 0.5               # spatial decay (1/m) of the permanence kernel
MISS_RATE_M = 2.0               # final-point miss threshold (metres)


# --------------------------------------------------------------------------- #
# Small numeric helpers                                                        #
# --------------------------------------------------------------------------- #
def _trapz(y: np.ndarray, t: np.ndarray | None = None, dt: float = 1.0) -> float:
    """Trapezoidal integral of ``y`` over ``t`` (or uniform ``dt``).

    Local implementation so the module is agnostic to the numpy-2.0
    ``trapz -> trapezoid`` rename. Returns 0.0 for fewer than 2 samples.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    if t is not None:
        t = np.asarray(t, dtype=float)
        return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(t)))
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * dt))


def _to_numpy(x) -> np.ndarray:
    """Accept torch tensors (from the gate-runner seam) or array-likes."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# --------------------------------------------------------------------------- #
# Scenario telemetry container                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioTelemetry:
    """Per-timestep log of one scenario clip + scalar outcome.

    One clip = one safety-critical scenario (the Bench2Drive convention: one
    scenario per short route), so clip-global reductions like "first braking
    onset" attribute to *this* hazard. Segment multi-hazard drives upstream.

    Time series (all length T, uniform ``dt`` unless ``timestamp_s`` given):
      timestamp_s       optional explicit timestamps (s); else uniform dt
      ego_v             ego longitudinal speed (m/s)
      ego_jerk          ego longitudinal jerk (m/s^3)
      steer_rate        steering angular rate |d(delta)/dt| driver (rad/s)
      latency_ms        per-decision inference latency (ms)
      hazard_los_flag   bool: the hazard is deterministically visible (LoS)
      dist_to_blind_spot distance to nearest dynamic-occlusion cone edge (m)
      is_occluded_flag  bool: a tracked agent is 100% occluded from the FoV
      wm_hazard_xy      [T,2] world-model estimate of the hidden agent (NaN if none)
      gt_hazard_xy      [T,2] ground-truth hidden-agent position

    Scalars:
      collisions        collision count over the clip
      ego_mass_kg       vehicle mass for kinetic-energy terms
      params_billions   active parameters (billions) for CNCE
    """
    ego_v: np.ndarray
    ego_jerk: np.ndarray
    steer_rate: np.ndarray
    latency_ms: np.ndarray
    hazard_los_flag: np.ndarray
    dist_to_blind_spot: np.ndarray
    is_occluded_flag: np.ndarray
    wm_hazard_xy: np.ndarray
    gt_hazard_xy: np.ndarray
    timestamp_s: np.ndarray | None = None
    dt: float = 0.1
    collisions: int = 0
    ego_mass_kg: float = 1500.0
    params_billions: float = 4.0

    def __post_init__(self):
        for name in ("ego_v", "ego_jerk", "steer_rate", "latency_ms",
                     "dist_to_blind_spot"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        for name in ("hazard_los_flag", "is_occluded_flag"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=bool))
        self.wm_hazard_xy = np.asarray(self.wm_hazard_xy, dtype=float)
        self.gt_hazard_xy = np.asarray(self.gt_hazard_xy, dtype=float)
        if self.timestamp_s is not None:
            self.timestamp_s = np.asarray(self.timestamp_s, dtype=float)

    def _t(self, i: int) -> float:
        """Timestamp of index ``i`` (from timestamp_s or uniform dt)."""
        if self.timestamp_s is not None:
            return float(self.timestamp_s[i])
        return float(i) * self.dt


# --------------------------------------------------------------------------- #
# 1. Latent Anticipation Latency (LAL)                                         #
# --------------------------------------------------------------------------- #
def compute_lal(tel: ScenarioTelemetry,
                jerk_threshold: float = JERK_BRAKE_THRESHOLD) -> float:
    """LAL = t_LoS - t_anticipation  (seconds).

    t_LoS         first time the hazard is deterministically visible (LoS flag).
    t_anticipation first braking onset (ego_jerk < jerk_threshold) in the clip.

    A pure reactive policy brakes only after the hazard's pixels appear
    (t_anticipation >= t_LoS) -> LAL <= 0. A world model that infers the occluded
    agent's continuation brakes *before* line-of-sight -> LAL > 0. Returns 0.0 if
    the hazard never reaches LoS (nothing to anticipate); ``LAL_NO_REACTION`` if
    LoS occurs but the ego never brakes (worst case, sortable).
    """
    los_idx = np.flatnonzero(tel.hazard_los_flag)
    if los_idx.size == 0:
        return 0.0
    t_los = tel._t(int(los_idx[0]))
    brake_idx = np.flatnonzero(tel.ego_jerk < jerk_threshold)
    if brake_idx.size == 0:
        return LAL_NO_REACTION
    t_anticipation = tel._t(int(brake_idx[0]))
    return float(t_los - t_anticipation)


# --------------------------------------------------------------------------- #
# 2. Tactical Maneuver Stability (TMS)                                         #
# --------------------------------------------------------------------------- #
def compute_tms(tel: ScenarioTelemetry,
                alpha: float = TMS_ALPHA, beta: float = TMS_BETA) -> float:
    """TMS = 1 / (1 + alpha*∫|jerk|dt + beta*∫|steer_rate|dt).

    Inverse kinematic-entropy: penalizes the longitudinal-jerk and steering-rate
    "flicker" of frame-by-frame reactive control. Perfectly smooth (zero jerk,
    zero steer rate) -> 1.0; noisier control -> toward 0. In (0, 1].
    """
    t = tel.timestamp_s
    jerk_i = _trapz(np.abs(tel.ego_jerk), t, tel.dt)
    steer_i = _trapz(np.abs(tel.steer_rate), t, tel.dt)
    return float(1.0 / (1.0 + alpha * jerk_i + beta * steer_i))


# --------------------------------------------------------------------------- #
# 3. Occluded Kinematic Risk Integral (OKRI)                                   #
# --------------------------------------------------------------------------- #
def compute_okri(tel: ScenarioTelemetry,
                 approach_m: float = OKRI_APPROACH_M,
                 eps: float = OKRI_EPS, scale: float = OKRI_SCALE) -> float:
    """OKRI = (1/scale) * ∫ [ (1/2 m v^2) / (d_blind + eps) ] dt, over d_blind < approach_m.

    Kinetic energy carried into a blind region, weighted by proximity to the
    occlusion edge. A world model throttles on the *epistemic* uncertainty of the
    blind spot, lowering the integral; a pixel-reactive policy that keeps speed on
    "empty visible pixels" scores higher (worse). Lower is safer. 0.0 if the ego
    never comes within ``approach_m`` of a blind spot.
    """
    mask = tel.dist_to_blind_spot < approach_m
    if not mask.any():
        return 0.0
    ke = 0.5 * tel.ego_mass_kg * tel.ego_v ** 2
    risk = ke / (tel.dist_to_blind_spot + eps)
    risk = np.where(mask, risk, 0.0)
    t = tel.timestamp_s
    return float(_trapz(risk, t, tel.dt) / scale)


# --------------------------------------------------------------------------- #
# 4. Compute-Normalized Causal Efficacy (CNCE)                                 #
# --------------------------------------------------------------------------- #
def compute_cnce(tel: ScenarioTelemetry, lam: float = CNCE_LAMBDA) -> float:
    """CNCE = D_progress / (mean_latency_s * params_billions) * exp(-lambda*collisions).

    Metres of safe progress bought per unit (latency x active-parameters), with a
    harsh exponential collision penalty. Binds tactical safety to edge-hardware
    cost: a 15B multi-camera transformer pays for its bloat in the denominator;
    the minimal 4B model buys more safe metres per compute cycle. Higher is better.
    """
    dist = _trapz(tel.ego_v, tel.timestamp_s, tel.dt)   # D_progress = ∫ v dt
    mean_lat_s = float(np.mean(tel.latency_ms)) / 1000.0
    denom = max(mean_lat_s * tel.params_billions, 1e-9)
    return float((dist / denom) * np.exp(-lam * tel.collisions))


# --------------------------------------------------------------------------- #
# 5. Latent Object Permanence Score (LOPS)                                     #
# --------------------------------------------------------------------------- #
def compute_lops(tel: ScenarioTelemetry, gamma: float = LOPS_GAMMA) -> float:
    """LOPS = mean_{t in occ} exp(-gamma * ||p_wm(t) - p_gt(t)||_2).

    Averaged over timesteps where the agent is fully occluded *and* the world
    model holds an estimate (non-NaN). Perfect latent tracking of the hidden agent
    -> 1.0; large tracking error -> toward 0. A model with no explicit latent
    tracking (standard E2E) produces no estimate under occlusion -> 0.0 baseline.
    In [0, 1].
    """
    occ = tel.is_occluded_flag & np.isfinite(tel.wm_hazard_xy).all(axis=1)
    if not occ.any():
        return 0.0
    err = np.linalg.norm(tel.wm_hazard_xy[occ] - tel.gt_hazard_xy[occ], axis=1)
    return float(np.mean(np.exp(-gamma * err)))


# --------------------------------------------------------------------------- #
# Assembly                                                                      #
# --------------------------------------------------------------------------- #
def run_scenario_suite(tel: ScenarioTelemetry, model_name: str = "TanitAD") -> dict:
    """Compute all five custom metrics for one scenario clip.

    Keys carry the direction-of-goodness so a report can never invert them:
    LAL >0 proactive, TMS ->1 smooth, OKRI lower safer, CNCE higher efficient,
    LOPS ->1 tracks-hidden.
    """
    return {
        "model": model_name,
        "LAL_s": round(compute_lal(tel), 4),
        "TMS": round(compute_tms(tel), 4),
        "OKRI": round(compute_okri(tel), 4),
        "CNCE": round(compute_cnce(tel), 4),
        "LOPS": round(compute_lops(tel), 4),
        "_directions": {"LAL_s": ">0 proactive", "TMS": "->1 smooth",
                        "OKRI": "lower safer", "CNCE": "higher efficient",
                        "LOPS": "->1 tracks hidden"},
    }


# --------------------------------------------------------------------------- #
# Trajectory-domain metrics — the gate-runner extra_metrics seam               #
# --------------------------------------------------------------------------- #
# These are (pred_xy, true_xy) -> float, matching what eval/gates.py merges into
# each gate's ``metrics`` block. They complement (do not duplicate) the runner's
# own ``ade_fde`` tuple: recognizable add-ons the decode gates can report.
def _as_traj(xy: np.ndarray) -> np.ndarray:
    """[N,2] -> [N,1,2]; [N,H,2] unchanged."""
    xy = _to_numpy(xy)
    if xy.ndim == 2:
        return xy[:, None, :]
    assert xy.ndim == 3 and xy.shape[-1] == 2, f"bad traj shape {xy.shape}"
    return xy


def ade(pred_xy, true_xy) -> float:
    """Average displacement error (m). Accepts [N,2] or [N,H,2]."""
    p, t = _as_traj(pred_xy), _as_traj(true_xy)
    return float(np.linalg.norm(p - t, axis=-1).mean())


def fde(pred_xy, true_xy) -> float:
    """Final displacement error (m): displacement at the last horizon step."""
    p, t = _as_traj(pred_xy), _as_traj(true_xy)
    return float(np.linalg.norm(p[:, -1] - t[:, -1], axis=-1).mean())


def rmse_xy(pred_xy, true_xy) -> float:
    """Root-mean-square displacement error (m)."""
    p, t = _as_traj(pred_xy), _as_traj(true_xy)
    return float(np.sqrt((np.linalg.norm(p - t, axis=-1) ** 2).mean()))


def miss_rate(pred_xy, true_xy, thresh_m: float = MISS_RATE_M) -> float:
    """Fraction of samples whose FINAL point misses ground truth by > thresh_m."""
    p, t = _as_traj(pred_xy), _as_traj(true_xy)
    final_err = np.linalg.norm(p[:, -1] - t[:, -1], axis=-1)
    return float((final_err > thresh_m).mean())


def trajectory_extra_metrics(
    include: Sequence[str] = ("rmse", "miss_rate"),
) -> Mapping[str, Callable]:
    """Return an ``extra_metrics`` dict for the D1-D3 gate runner's seam.

    Defaults to rmse + miss_rate (ade/fde are already the runner's own primary
    numbers). Usage:  ``run_d1(..., extra_metrics=trajectory_extra_metrics())``.
    """
    table: dict[str, Callable] = {
        "ade": ade, "fde": fde, "rmse": rmse_xy, "miss_rate": miss_rate,
    }
    return {name: table[name] for name in include}
