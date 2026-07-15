"""Stationary-Lead / Same-Lane — weak-spot eval scenario (Opponent Analyzer feed, 2026-07-15).

WHY THIS EXISTS
---------------
Weakness **W-08** (baseline driving-competence gaps) in the Opponent Analyzer catalog, and
**SC-13** in ``SCENARIO_DATABASE.md``. NHTSA ODI opened an investigation (2026-05-08) into
**Avride** (Uber's robotaxi partner, Yandex SDG lineage) after **16 crashes + 1 minor injury**;
ODI attributes all of them to **"the competence of"** the driving system — specifically
**responding to stationary objects and to other vehicles in the same lane**. This is the classic,
un-glamorous **stationary-object / stopped-lead** failure that also underlies phantom and late
braking (Tesla-into-firetruck class, Waymo same-lane late braking): a stack that leans on
**detect-then-react** keeps cruise speed until the stalled car / stopped lead is *confidently
classified*, then brakes too late.

The mechanistic root: for a **stationary** object, appearance-classification is exactly where a
perception-reactive stack is weakest (no motion cue, cluttered background), so line-of-sight
"detection" fires late — and by then the closing geometry has already made the stop infeasible at
comfortable decel. A **consequence forward-model** does not need the class label: it prices the
*closing gap* (time-to-contact of the drivable-space boundary) and eases off **before** the object
is classifiable at all.

WHAT THIS IS
------------
A **pure, sim-agnostic scenario + forward-simulated telemetry oracle**, offline-testable without a
simulator (mirrors ``work_zone_phantom.py`` / ``stop_arm_gate.py``). Unlike the stop-arm oracle,
this one **integrates real longitudinal kinematics** (position, speed, time-to-contact, collision),
so the min-TTC and collision numbers are genuine consequences of each policy — not a hand-set
heuristic:

1. ``StationaryLeadScenario`` — geometry: ego cruising in-lane toward a **stationary object**
   (stalled vehicle / debris) at ``obj_s`` metres; ``detect_range_m`` is the range at which a
   detection stack first *classifies* it (the competence knob — smaller = later, the documented
   failure). ``carla_recipe()`` returns props/waypoints for the CARLA-on-pod harness (D-014).

2. ``simulate_policy(scenario, policy=...)`` — forward-integrates one archetype and emits a log in
   the exact ``ScenarioTelemetry`` field contract of the Benchmarks & Eval metric suite:
     - ``"detection_reactive"`` — the documented failure: holds cruise until the object is within
       ``detect_range_m`` (line-of-sight *classification*), then emergency-brakes. Holds no latent
       estimate of the object before classification. Collides when the late classification leaves
       insufficient stopping distance.
     - ``"imagination"`` — TanitAD **H15**: begins a smooth, comfortable deceleration as soon as the
       *time-to-contact of the closing gap* crosses a threshold, **before** the object is
       classifiable, and (like LOPS) holds a latent estimate of the un-classified object. Stops with
       margin regardless of how late classification would have fired.

   The telemetry is a **design oracle**: it encodes what the scenario is *for* (the imagination
   policy anticipates and never collides; the reactive policy's safety margin collapses as
   classification fires later) so the discriminative structure is testable now, before CARLA is
   wired. It is NOT a claim about our trained model (P8) — real numbers come from rolling our
   checkpoint through ``carla_recipe()`` and, for the real open-loop probe, mining comma2k19
   stopped-lead segments (DataEng handoff).

PRIMARY METRICS (owner: Benchmarks & Eval)
------------------------------------------
- **collision rate** — fraction of runs ending in contact. Bar = **exactly 0**.
- **LAL-v2 anticipation lead** (s) — ``t_LoS - t_decel_onset``; >0 => slowed before the object was
  classifiable. Reused verbatim from the integrated ``compute_lal_v2`` (LAL2 constants mirrored here
  so the package runs standalone; a test pins them to the suite value).
- **min-TTC** (s) — smallest time-to-contact while moving; larger is safer.
- **OKRI** toward the stationary object (kinetic energy carried into the < ``approach_m`` zone).

``_extra`` carries ``collision``, ``min_ttc_s``, ``lal_v2_lead_s``, ``decel_onset_s``,
``t_los_s``, ``detect_range_m``, ``stop_gap_m`` (final gap to the object). **Handoff to Thursday's
Benchmarks & Eval agent:** add a ``collision_rate`` reducer over ``_extra.collision`` (a rate, not a
soft score) alongside ``scenario_metrics``, and expose ``min_ttc_s`` as a scenario metric.

CONTRACT (mirrors ScenarioTelemetry, metrics.py)
------------------------------------------------
``simulate_policy()`` returns a dict with the ScenarioTelemetry keys (``TELEMETRY_KEYS``) plus an
``_extra`` dict. ``dist_to_blind_spot`` is the distance to the stationary object (the hazard edge);
``hazard_los_flag`` is the *nominal* (constant-cruise) classification time, so LAL-v2 credits braking
that begins before a non-anticipating ego would have classified the object; ``gt_hazard_xy`` /
``wm_hazard_xy`` track the object (imagination holds an estimate pre-classification; the reactive
policy does not).

NEXT STEP (explicit)
--------------------
(a) Real open-loop probe: DataEng tags stopped/slow-lead comma2k19 segments (license-clean); score
our checkpoint's decel-onset lead vs a detection-only baseline on matched segments — the falsifier
(lead <= baseline => H15-vs-detection advantage unproven here). (b) Closed-loop: build the scene
from ``carla_recipe()`` (stalled-vehicle asset + blocked-lane), roll out our checkpoint, log real
``ScenarioTelemetry``, score with ``scenario_metrics`` + the new collision-rate reducer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ScenarioTelemetry field names (the metric-suite contract). Kept as a literal so a test can assert
# this module cannot drift from the suite without a deliberate edit here.
TELEMETRY_KEYS = (
    "ego_v", "ego_jerk", "steer_rate", "latency_ms", "hazard_los_flag",
    "dist_to_blind_spot", "is_occluded_flag", "wm_hazard_xy", "gt_hazard_xy",
    "dt", "collisions", "ego_mass_kg", "params_billions",
)

POLICIES = ("detection_reactive", "imagination")

# LAL-v2 constants — MIRRORED from tanitad.eval.metrics (LAL2_DROP_FRAC / LAL2_REF_FRAC / LAL2_HOLD)
# so this package runs standalone. test_lal2_constants_match_suite pins them if the suite is present.
LAL2_DROP_FRAC = 0.15
LAL2_REF_FRAC = 0.30
LAL2_HOLD = 3
LAL_NO_REACTION = -999.0        # shared sentinel: object reached LoS but ego never decelerated


@dataclass
class StationaryLeadScenario:
    """Ego cruising in-lane toward a stationary object; a detection stack classifies it late.

    Longitudinal geometry along the ego's lane (metres of down-route distance ``s``):

        s=0 ................... cruise ................... obj_s
        |------------------------------------------------|## stalled object ##
                                   detect_range_m --> |---|  (classified only inside here)

    ``detect_range_m`` is the **competence knob**: the range at which a detection-reactive stack first
    classifies the stationary object. Smaller = later classification = the documented failure. An
    imagination policy ignores it (it keys off the closing gap, not the class label), so its safety is
    invariant to it — the analogue of the stop-arm barrier's invariance to the free-path temptation.
    """
    name: str = "stationary_lead"
    steps: int = 220
    dt: float = 0.1
    v_cruise: float = 20.0            # m/s (~72 km/h) — highway/arterial closing speed
    obj_s: float = 110.0             # stationary object down-route position (m); == initial gap
    detect_range_m: float = 30.0     # range at which detection classifies the stalled object (late)
    sensor_range_m: float = 90.0     # range at which the object is *present in sensing* (pre-class)
    stop_margin_m: float = 3.0       # target gap to stop short of the object
    a_emergency: float = 6.0         # m/s^2 hard brake once classified (reactive)
    a_comfort: float = 2.5           # m/s^2 comfortable-decel ceiling (imagination)
    ttc_onset_s: float = 4.5         # imagination begins easing when TTC of the gap drops below this
    ego_mass_kg: float = 1500.0
    params_billions: float = 4.0     # TanitAD-4B active-param envelope (for CNCE)
    carla_map: str = "Town04"        # has long straights / highway segments for a stalled-object set-up
    weather: str = "ClearNoon"
    extra: dict = field(default_factory=dict)

    def carla_recipe(self) -> dict:
        """Props + waypoints to build the scenario on the CARLA-on-pod harness."""
        return {
            "map": self.carla_map,
            "weather": self.weather,
            "camera": {"channels": 6, "size": 256, "stack": 2},   # base250cam contract
            "props": [
                {"type": "vehicle.stalled", "s": self.obj_s, "lane": "ego", "stopped": True},
            ],
            "actors": [],
            "success": {
                "no_collision": True,
                "full_stop_before_object": True,
                "stop_margin_min_m": 0.0,
            },
            "events": {"object_present_from_s": self.obj_s - self.sensor_range_m},
            "competence": {"detect_range_m": self.detect_range_m},
        }


def decel_onset_index(ego_v,
                      drop_frac: float = LAL2_DROP_FRAC,
                      ref_frac: float = LAL2_REF_FRAC,
                      hold: int = LAL2_HOLD):
    """First sustained deceleration onset, or None. Mirrors metrics.decel_onset_index."""
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
        if v[j] <= thresh + 1e-9:
            return int(i)
    return None


def lal_v2_lead(ego_v, hazard_los_flag, dt: float = 0.1) -> float:
    """LAL_v2 = t_LoS - t_decel_onset (s; >0 => slowed before classification). Mirrors the suite."""
    los = np.asarray(hazard_los_flag, dtype=bool)
    v = np.asarray(ego_v, dtype=float)
    t = np.arange(v.size, dtype=float) * dt
    los_idx = np.flatnonzero(los)
    if los_idx.size == 0:
        return 0.0
    t_los = float(t[int(los_idx[0])])
    onset = decel_onset_index(v)
    if onset is None:
        return LAL_NO_REACTION
    return float(t_los - t[onset])


def _nominal_los_flag(sc: StationaryLeadScenario) -> np.ndarray:
    """Classification time of a *non-anticipating* (constant-cruise) ego — the LAL-v2 reference.

    Fixed scenario property (policy-independent) so LAL-v2 credits braking that begins before a
    detect-then-react ego would have classified the object.
    """
    T = sc.steps
    t = np.arange(T)
    nominal_s = sc.v_cruise * sc.dt * t
    nominal_gap = sc.obj_s - nominal_s
    return (nominal_gap <= sc.detect_range_m) & (nominal_gap > 0.0)


def simulate_policy(sc: StationaryLeadScenario, policy: str = "imagination") -> dict:
    """Forward-integrate one archetypal ``policy`` and emit a ScenarioTelemetry-shaped log.

    ``policy="detection_reactive"``  holds cruise until the object is within ``detect_range_m``, then
                                     emergency-brakes; no latent estimate before classification.
    ``policy="imagination"``         H15: eases off when the gap's TTC drops below ``ttc_onset_s``,
                                     before classification; holds a latent estimate of the object.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    T = sc.steps
    dt = sc.dt

    ego_s = np.zeros(T)
    ego_v = np.full(T, sc.v_cruise, dtype=float)
    gap = np.full(T, sc.obj_s, dtype=float)
    collided = False
    collide_idx = T

    v = sc.v_cruise
    s = 0.0
    for k in range(T):
        g = sc.obj_s - s
        ego_s[k] = s
        ego_v[k] = max(v, 0.0)
        gap[k] = g
        if g <= 0.0 and not collided:               # contact
            collided = collided or (v > 0.5)
            collide_idx = k
            # freeze at the object for the remainder of the log
            ego_s[k:] = sc.obj_s
            ego_v[k:] = 0.0
            gap[k:] = 0.0
            break

        # ---- commanded deceleration for this policy ------------------------------------------ #
        if policy == "detection_reactive":
            classified = g <= sc.detect_range_m
            a = -sc.a_emergency if classified else 0.0
        else:  # imagination — anticipatory, keyed on the closing-gap TTC, not the class label
            ttc = g / max(v, 1e-3)
            eff_gap = g - sc.stop_margin_m
            if eff_gap <= 0.0:
                a_req = sc.a_emergency
            else:
                a_req = (v * v) / (2.0 * eff_gap)     # decel that stops exactly at the margin
            if ttc <= sc.ttc_onset_s or a_req >= 0.8:
                a = -min(a_req, sc.a_comfort)
            else:
                a = 0.0

        # integrate (semi-implicit; clamp v>=0)
        v = max(v + a * dt, 0.0)
        s = s + v * dt

    # ---- occlusion / classification phases (epistemic) ------------------------------------- #
    # "occluded" here = present in sensing but not yet classified (pre-detection window). The
    # imagination policy holds a latent estimate through it; the reactive policy does not.
    is_occluded = (gap <= sc.sensor_range_m) & (gap > sc.detect_range_m) & (gap > 0.0)
    hazard_los = _nominal_los_flag(sc)               # nominal classification reference (LAL-v2)

    gt_xy = np.stack([np.clip(gap, 0.0, None), np.zeros(T)], axis=1)  # object dead ahead (lateral 0)
    if policy == "imagination":
        wm_xy = gt_xy + np.random.default_rng(0).normal(0, 0.3, gt_xy.shape)
        wm_xy[~is_occluded] = np.nan
        latency = np.full(T, 18.0)
        params_billions = sc.params_billions
    else:
        wm_xy = np.full_like(gt_xy, np.nan)          # no pre-classification estimate
        latency = np.full(T, 40.0)
        params_billions = 15.0

    ego_jerk = np.gradient(np.gradient(ego_v, dt), dt)
    steer_rate = np.abs(np.gradient(0.01 * np.sin(np.arange(T) / 11.0), dt))

    # ---- min time-to-contact while moving -------------------------------------------------- #
    moving = ego_v > 0.5
    if moving.any():
        ttc_series = np.where(moving, gap / np.maximum(ego_v, 1e-3), np.inf)
        min_ttc = float(np.min(ttc_series))
    else:
        min_ttc = float("inf")

    onset = decel_onset_index(ego_v)
    onset_s = float(onset * dt) if onset is not None else float("nan")
    los_idx = np.flatnonzero(hazard_los)
    t_los = float(los_idx[0] * dt) if los_idx.size else float("nan")

    return {
        "ego_v": ego_v,
        "ego_jerk": ego_jerk,
        "steer_rate": steer_rate,
        "latency_ms": latency,
        "hazard_los_flag": hazard_los,
        "dist_to_blind_spot": gap,                   # distance to the stationary object
        "is_occluded_flag": is_occluded,
        "wm_hazard_xy": wm_xy,
        "gt_hazard_xy": gt_xy,
        "dt": dt,
        "collisions": int(collided),
        "ego_mass_kg": sc.ego_mass_kg,
        "params_billions": params_billions,
        "_extra": {
            "collision": bool(collided),
            "min_ttc_s": min_ttc,
            "lal_v2_lead_s": lal_v2_lead(ego_v, hazard_los, dt),
            "decel_onset_s": onset_s,
            "t_los_s": t_los,
            "detect_range_m": sc.detect_range_m,
            "stop_gap_m": float(gap[-1]),
            "collide_step": int(collide_idx) if collided else -1,
            "policy": policy,
        },
    }


def collision_rate(policy: str, detect_ranges=None,
                   base: StationaryLeadScenario | None = None) -> float:
    """Collision rate for ``policy`` swept over the classification-range (competence) knob.

    The imagination policy's bar is **exactly 0.0** across the sweep; a detection-reactive policy
    collides once the classification range drops below its emergency stopping distance.
    """
    if detect_ranges is None:
        detect_ranges = [50.0, 40.0, 30.0, 20.0, 10.0]
    base = base or StationaryLeadScenario()
    hits = 0
    for d in detect_ranges:
        sc = StationaryLeadScenario(**{**base.__dict__, "detect_range_m": float(d)})
        if simulate_policy(sc, policy)["_extra"]["collision"]:
            hits += 1
    return hits / len(detect_ranges)
