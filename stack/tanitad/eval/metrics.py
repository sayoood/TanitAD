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
# 1b. LAL-v2 — anticipatory deceleration lead (integrated 2026-07-09, intake   #
#     `2026-07-09-lal-v2-anticipation`). LAL-v1 measures reaction HARDNESS     #
#     (jerk threshold) and is blind to comfort-bounded anticipatory slowing —  #
#     proven non-discriminative on the first live SC-01 run (both policies     #
#     -0.7 s). LAL-v2 detects sustained speed drop vs the free-cruise          #
#     reference; >0 = slowed BEFORE line-of-sight (the H15 edge). Assumes the  #
#     clip STARTS in free cruise (true for the SC-01 contract).                #
# --------------------------------------------------------------------------- #
LAL2_DROP_FRAC = 0.15   # speed must fall >=15% below free-cruise to count
LAL2_REF_FRAC = 0.30    # free-cruise ref = max speed over first 30% of clip
LAL2_HOLD = 3           # steps the drop must persist (reject one-sample dips)


def decel_onset_index(ego_v,
                      drop_frac: float = LAL2_DROP_FRAC,
                      ref_frac: float = LAL2_REF_FRAC,
                      hold: int = LAL2_HOLD):
    """Index of the first sustained deceleration onset, or None."""
    v = np.asarray(ego_v, dtype=float)
    T = v.size
    if T == 0:
        return None
    ref_n = max(1, int(round(ref_frac * T)))
    v_ref = float(np.max(v[:ref_n]))
    if v_ref <= 0.0:
        return None
    thresh = (1.0 - drop_frac) * v_ref
    below = v <= thresh
    for i in np.flatnonzero(below):
        j = min(int(i) + hold, T - 1)
        if v[j] <= thresh + 1e-9:       # sustained, not a transient dip
            return int(i)
    return None


def compute_lal_v2(ego_v, hazard_los_flag, dt: float = 0.1,
                   timestamp_s=None,
                   drop_frac: float = LAL2_DROP_FRAC,
                   ref_frac: float = LAL2_REF_FRAC,
                   hold: int = LAL2_HOLD) -> float:
    """LAL_v2 = t_LoS - t_decel_onset (s; >0 => slowed before line-of-sight).

    Latent, pre-LoS generalization of the TTB/TTC family: credits braking that
    begins before the hazard is detectable at all. Sentinels shared with v1:
    0.0 if the hazard never reaches LoS; LAL_NO_REACTION if never decelerates.
    """
    los = np.asarray(hazard_los_flag, dtype=bool)
    v = np.asarray(ego_v, dtype=float)
    t = (np.asarray(timestamp_s, dtype=float) if timestamp_s is not None
         else np.arange(v.size, dtype=float) * dt)
    los_idx = np.flatnonzero(los)
    if los_idx.size == 0:
        return 0.0
    t_los = float(t[int(los_idx[0])])
    onset = decel_onset_index(v, drop_frac, ref_frac, hold)
    if onset is None:
        return LAL_NO_REACTION
    return float(t_los - t[onset])


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
# 6. Traffic-Light Compliance (TLC) — SC-14 signal-phase compliance            #
#    (added 2026-07-24, intake `2026-07-24-traffic-light-scenario-metric`).    #
#                                                                              #
# WHY: none of LAL/TMS/OKRI/CNCE/LOPS scores a discrete legal signal barrier.  #
# The Dallas red-light-running failure (SCENARIO_DATABASE SC-14, W-03 family)  #
# is a HARD rule — running a red is not a soft trade-off against an apparently #
# clear intersection. TLC is the traffic-light-handling metric: one score in   #
# [0, 1] (higher better) that (a) hard-fails a red-light entry, (b) rewards a   #
# smooth stop at/before the line on red, and (c) penalizes phantom braking on  #
# a green the ego should just proceed through.                                 #
# --------------------------------------------------------------------------- #
# Signal-phase encoding (this module owns it; the scenario imports these).
SIGNAL_RED = 0
SIGNAL_YELLOW = 1
SIGNAL_GREEN = 2

# TLC named constants (no magic numbers in the body; direction documented inline).
TLC_CREEP_SPEED = 0.5         # m/s : at/below this the ego is "stopped", not "entering" the line
TLC_MARGIN_COMFORT_M = 5.0    # m   : halting anywhere within this before the line is fully compliant
TLC_MARGIN_SCALE_M = 8.0      # m   : decay scale penalizing over-cautious far-back stops (flow cost)
TLC_COMFORT_DECEL = 2.5       # m/s^2 : comfortable deceleration ceiling for a smooth stop
TLC_DECEL_K = 0.5             # (m/s^2)^-1 : smoothness sensitivity to braking harder than comfort
TLC_REF_FRAC = 0.30           # free-cruise reference = max speed over the first 30% of the clip
TLC_PHANTOM_DEADBAND = 0.05   # frac : green-phase speed dips below this are natural, not phantom braking
TLC_PHANTOM_DROP_FRAC = 0.15  # frac : a green-phase speed drop beyond this is flagged phantom braking


def tlc_report(ego_v, ego_s, signal_state, stopline_s,
               dt: float = 0.1, timestamp_s=None,
               creep_speed: float = TLC_CREEP_SPEED,
               margin_comfort_m: float = TLC_MARGIN_COMFORT_M,
               margin_scale_m: float = TLC_MARGIN_SCALE_M,
               comfort_decel: float = TLC_COMFORT_DECEL,
               decel_k: float = TLC_DECEL_K,
               ref_frac: float = TLC_REF_FRAC,
               phantom_deadband: float = TLC_PHANTOM_DEADBAND) -> dict:
    """Full breakdown of the Traffic-Light Compliance score for one approach clip.

    TLC = red_entry_gate * stop_quality * green_flow          (in [0, 1], higher better)

      red_entry_gate  {0,1}   0 iff the ego crosses the stop line while the signal is RED
                              above ``creep_speed`` (running the red — the discrete legal
                              barrier; a single violation zeroes the whole score). 1 otherwise.
      stop_quality    [0,1]   applies only when a stop is required (a RED phase is faced):
                              margin_factor * smooth_factor.
                                margin_factor = 1 if the ego halts in [0, margin_comfort_m]
                                  before the line; exp(-(margin-comfort)/scale) if it stops
                                  further back (over-cautious, a flow cost); 0 if it never
                                  halts before the line or halts past it.
                                smooth_factor = 1/(1 + decel_k * max(0, peak_decel - comfort_decel));
                                  a stop within the comfortable deceleration ceiling -> 1, an
                                  emergency slam -> < 1.
                              1.0 (N/A) when no stop is required.
      green_flow      [0,1]   applies only on a genuine proceed-on-green (no RED faced):
                              1 - (severity - deadband)/(1 - deadband), where severity is the
                              fractional speed drop below the free-cruise reference during the
                              green approach. Holding speed -> 1; phantom-braking to a crawl -> ~0.
                              1.0 (N/A) when a stop is required (slowing for the red is legitimate).

    One intersection per clip (the ScenarioTelemetry / Bench2Drive convention). ``ego_s`` is the
    ego's cumulative down-route distance (m); ``signal_state`` is per-step {SIGNAL_RED, _YELLOW,
    _GREEN}. Returns every component so a report can never invert the direction.
    """
    v = np.asarray(ego_v, dtype=float)
    s = np.asarray(ego_s, dtype=float)
    sig = np.asarray(signal_state).astype(int)
    T = v.size
    t = (np.asarray(timestamp_s, dtype=float) if timestamp_s is not None
         else np.arange(T, dtype=float) * dt)

    # ---- red-entry gate (the hard legal barrier) --------------------------------------- #
    crossed = np.flatnonzero(s >= stopline_s)
    cross_idx = int(crossed[0]) if crossed.size else None
    entered_on_red = bool(cross_idx is not None
                          and sig[cross_idx] == SIGNAL_RED
                          and v[cross_idx] > creep_speed)
    red_entry_gate = 0.0 if entered_on_red else 1.0

    # A stop is required iff the ego faces a RED phase in this clip (one intersection/clip).
    must_stop = bool(np.any(sig == SIGNAL_RED))

    # ---- stop quality (only when a stop is required) ----------------------------------- #
    stopped_before_line = False
    stop_margin_m = float("nan")
    peak_decel = 0.0
    if must_stop:
        halted = np.flatnonzero((v <= creep_speed) & (s <= stopline_s + 0.1))
        if halted.size:
            halt_idx = int(halted[0])
            stopped_before_line = True
            stop_margin_m = float(stopline_s - s[halt_idx])
        else:
            halt_idx = T - 1
        accel = np.gradient(v, t) if T >= 2 else np.zeros(T)
        peak_decel = float(max(0.0, -np.min(accel[:halt_idx + 1]))) if T >= 2 else 0.0
        if not stopped_before_line or stop_margin_m < 0.0:
            margin_factor = 0.0
        elif stop_margin_m <= margin_comfort_m:
            margin_factor = 1.0
        else:
            margin_factor = float(np.exp(-(stop_margin_m - margin_comfort_m) / margin_scale_m))
        smooth_factor = float(1.0 / (1.0 + decel_k * max(0.0, peak_decel - comfort_decel)))
        stop_quality = margin_factor * smooth_factor
    else:
        margin_factor = smooth_factor = stop_quality = 1.0

    # ---- green flow (only on a genuine proceed-on-green) ------------------------------- #
    green_drop_frac = 0.0
    if (not must_stop) and np.any(sig == SIGNAL_GREEN):
        ref_n = max(1, int(round(ref_frac * T)))
        v_ref = float(np.max(v[:ref_n]))
        consider = (sig == SIGNAL_GREEN) & (s <= stopline_s + 0.1)
        min_v = float(np.min(v[consider])) if consider.any() else v_ref
        green_drop_frac = float(max(0.0, (v_ref - min_v) / v_ref)) if v_ref > 0 else 0.0
        if green_drop_frac <= phantom_deadband:
            green_flow = 1.0
        else:
            green_flow = float(np.clip(
                1.0 - (green_drop_frac - phantom_deadband) / (1.0 - phantom_deadband), 0.0, 1.0))
    else:
        green_flow = 1.0

    tlc = float(np.clip(red_entry_gate * stop_quality * green_flow, 0.0, 1.0))
    return {
        "TLC": round(tlc, 4),
        "red_entry_gate": red_entry_gate,
        "entered_on_red": entered_on_red,
        "must_stop": must_stop,
        "stopped_before_line": stopped_before_line,
        "stop_margin_m": (round(stop_margin_m, 3) if stop_margin_m == stop_margin_m else None),
        "peak_decel_mps2": round(peak_decel, 3),
        "stop_quality": round(float(stop_quality), 4),
        "green_flow": round(float(green_flow), 4),
        "green_drop_frac": round(green_drop_frac, 4),
        "_direction": "TLC ->1 compliant; 0 = ran a red light",
    }


def compute_tlc(ego_v, ego_s, signal_state, stopline_s,
                dt: float = 0.1, timestamp_s=None, **kw) -> float:
    """TLC scalar in [0, 1] (higher better; 0 = ran a red light). See ``tlc_report``."""
    return tlc_report(ego_v, ego_s, signal_state, stopline_s,
                      dt=dt, timestamp_s=timestamp_s, **kw)["TLC"]


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
        "LAL_v2_s": round(compute_lal_v2(tel.ego_v, tel.hazard_los_flag,
                                         tel.dt, tel.timestamp_s), 4),
        "TMS": round(compute_tms(tel), 4),
        "OKRI": round(compute_okri(tel), 4),
        "CNCE": round(compute_cnce(tel), 4),
        "LOPS": round(compute_lops(tel), 4),
        "_directions": {"LAL_s": ">0 proactive (reaction-onset, hard-brake)",
                        "LAL_v2_s": ">0 anticipation lead (decel-onset)",
                        "TMS": "->1 smooth",
                        "OKRI": "lower safer", "CNCE": "higher efficient",
                        "LOPS": "->1 tracks hidden"},
    }


def traffic_light_metrics(tel: ScenarioTelemetry, ego_s, signal_state, stopline_s,
                          model_name: str = "TanitAD") -> dict:
    """Score a signalized-intersection (SC-14) clip: the base suite + the TLC metric.

    ``tel`` is the standard telemetry (so LAL/TMS/OKRI/CNCE/LOPS still report — OKRI
    here measures kinetic energy carried toward the stop-line barrier), and TLC adds
    the traffic-light-handling verdict. ``ego_s`` / ``signal_state`` / ``stopline_s``
    carry the signal-phase geometry that the fixed ScenarioTelemetry contract does not.
    """
    base = run_scenario_suite(tel, model_name=model_name)
    tlc = tlc_report(tel.ego_v, ego_s, signal_state, stopline_s,
                     dt=tel.dt, timestamp_s=tel.timestamp_s)
    base.update({k: v for k, v in tlc.items() if k != "_direction"})
    base["_directions"]["TLC"] = tlc["_direction"]
    return base


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
