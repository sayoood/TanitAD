"""Stationary-Lead — weak-spot eval scenario (Opponent Analyzer scenario feed, 2026-07-31 / run #3).

WHY THIS EXISTS
---------------
Weakness **W-08** (baseline driving-competence gaps) in the Opponent Analyzer catalog, and
**SC-13** in ``SCENARIO_DATABASE.md``. Two FACT-grade opponent failures name exactly this axis:

  * **Avride** (Uber robotaxi partner, Yandex SDG lineage) — NHTSA ODI opened an investigation
    (2026-05-08) after **16 crashes + 1 minor injury**; ODI attributes all to **"the competence of"**
    the system, listing **responding to stationary objects** and **responding to other vehicles in
    the same lane**. — techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-...
  * **Tesla FSD** — NHTSA Engineering Analysis EA26002 (degradation-detection) found that in *each*
    of the degraded-visibility crashes it reviewed, "FSD **also lost track of or never detected a
    lead vehicle in its path.**" — nhtsa.gov ODI EA26002.

The failure class is generic and *mundane* (which is what makes it a broad, damning surface): a
**detection-then-react** stack brakes late on a stopped/slow lead or a stationary object because it
waits for the object to be *classified* with confidence. Under any ambiguity (a stationary object
whose class prior is weak, or a degraded sensing channel) the reaction slips later and later, and in
the limit the lead is dropped entirely → late/no braking → collision. That is exactly wrong: the
*consequence* of the closing gap is knowable **before** the object is classified.

WHAT THIS IS
------------
A **pure, sim-agnostic scenario specification + synthetic-telemetry generator**, offline-testable
without a simulator (mirrors ``work_zone_phantom.py`` and ``stop_arm_gate.py``):

1. ``StationaryLeadScenario`` — a dataclass describing the geometry (ego cruising toward a
   **stationary lead object** in the same lane) plus a ``classification_ambiguity`` in [0, 1] — how
   hard the object is to *classify* (stalled car vs debris vs shadow; or a degraded sensing channel).
   ``carla_recipe()`` returns props/waypoints to build it on the **CARLA-on-pod** harness (D-014).

2. ``simulate_policy(scenario, policy=...)`` — emits a ``ScenarioTelemetry``-shaped log for two
   archetypes:
     - ``"detection_reactive"`` — the documented failure: brakes only when time-to-contact crosses a
       reaction threshold, and that threshold **slips later as ``classification_ambiguity`` rises**;
       past a drop threshold it **loses the lead** (no latent estimate) and brakes far too late.
     - ``"imagination_forward"`` — TanitAD-style **H15**: forward-models the time-to-contact of the
       *closing gap* and begins a smooth deceleration early, **independent of classification** — no
       detection/class prior to be wrong about (and **A9** holds a latent lead estimate throughout).

   The synthetic telemetry is a **design oracle**: it encodes what the scenario is *for* so the
   discriminative structure (reactive degrades and eventually collides as ambiguity rises;
   imagination is invariant and never collides) is testable now, before CARLA is wired. It is NOT a
   claim about our real model (P8) — real numbers come from running our checkpoint through this
   scenario on the pod, or an open-loop probe on comma2k19 stopped-lead segments.

PRIMARY METRICS (owner: Benchmarks & Eval)
------------------------------------------
**Braking-onset lead time** (LAL-v2 anticipation, per 2026-07-09) and **min time-to-contact**
(min-TTC), with a hard **collisions == 0** bar. Secondary: OKRI toward the lead object; min following
gap. ``_extra`` carries ``brake_onset_lead_time_s``, ``min_ttc_s``, ``min_gap_m``, ``collided``,
``lead_dropped``, ``classification_ambiguity``. Handoff to Thursday's Benchmarks & Eval agent: add a
``min_ttc`` reducer + a ``collision_rate`` reducer over ``_extra``, and reuse the LAL-v2 lead-time
metric over ``_extra.brake_onset_lead_time_s``.

CONTRACT (mirrors ScenarioTelemetry, tanitad_metrics.py)
--------------------------------------------------------
``simulate_policy()`` returns a dict with the ScenarioTelemetry keys (see ``TELEMETRY_KEYS``) plus a
scenario-specific ``_extra`` dict. Here ``dist_to_blind_spot`` is the *actual* gap to the stationary
lead (so an OKRI-style reducer prices kinetic energy carried toward it), and ``gt_hazard_xy`` /
``wm_hazard_xy`` track the lead object (the reactive policy's ``wm`` goes NaN when it drops the lead).

FALSIFIER (pre-registered, backlog P0)
--------------------------------------
If, on matched real stopped-lead segments (comma2k19) with our trained checkpoint, the
imagination-error braking-onset lead time is **≤** a detection-only baseline, the H15-vs-detection
advantage is *unproven here* — record it as a negative result (P8), do not claim SC-13 excellence.

NEXT STEP (explicit)
--------------------
(a) DataEng: tag stopped/slow-lead segments in comma2k19 (real, license-clean) for an open-loop lead
-time probe. (b) Benchmarks & Eval: add the min-TTC + collision-rate reducers and wire SC-13 into the
eval set. (c) On CARLA-on-pod: build from ``carla_recipe()`` (stalled vehicle / debris prop in-lane),
roll out our trained checkpoint, log real ``ScenarioTelemetry``, and score. This module's
``simulate_policy`` is then replaced by the real rollout; the geometry and ``_extra`` stay.
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

POLICIES = ("detection_reactive", "imagination_forward")

_A_COMFORT = 2.5   # m/s^2 comfortable deceleration (imagination's planned brake)
_A_HARD = 6.0      # m/s^2 hard/emergency deceleration (reactive's late brake)
_COLLISION_GAP_M = 1.0   # front bumper within 1 m of the stationary lead == contact
_SENSOR_RANGE_M = 90.0   # the lead is physically visible from here in (the point: visible != acted-on)


@dataclass
class StationaryLeadScenario:
    """A stationary lead object in the ego lane; ego cruises toward it.

    Longitudinal geometry along the ego's approach (metres of down-route distance ``s``):

        s=0                                   lead_s
        |------------- free cruise ------------|##| stationary object (stalled car / debris)

    ``classification_ambiguity`` in [0, 1] is how hard the object is to *classify* (weak class prior,
    or a degraded sensing channel a la the Tesla EA lead-loss finding). A detection-reactive policy's
    braking-onset slips later as it rises; a forward-model policy is invariant to it because it prices
    the closing gap, not the class. All fields are pure numbers; nothing imports a sim.
    """
    name: str = "stationary_lead"
    steps: int = 200
    dt: float = 0.1
    v_cruise: float = 15.0             # m/s approach (~54 km/h)
    lead_s: float = 100.0             # stationary lead object down-route (m)
    classification_ambiguity: float = 0.0   # [0,1] weak class prior / degraded channel
    safe_gap_m: float = 5.0           # target following gap to a stopped lead
    ego_mass_kg: float = 1500.0
    params_billions: float = 4.0      # TanitAD-4B active-param envelope (for CNCE)
    # CARLA build hints (used by carla_recipe; not needed for the offline telemetry oracle)
    carla_map: str = "Town10HD"
    weather: str = "ClearNoon"
    extra: dict = field(default_factory=dict)

    def carla_recipe(self) -> dict:
        """Props + waypoints to build the scenario on the CARLA-on-pod harness."""
        return {
            "map": self.carla_map,
            "weather": self.weather,
            "camera": {"channels": 6, "size": 256, "stack": 2},  # base250cam contract
            "props": [
                # a stalled vehicle (or debris) stopped in the ego lane
                {"type": "vehicle.stalled", "s": self.lead_s, "lane": "ego", "stopped": True},
            ],
            "actors": [],
            "ego": {"spawn_s": 0.0, "v_cruise": self.v_cruise},
            "success": {
                # correct behaviour: smooth early decel to a safe gap; never contact the lead
                "no_contact": True,
                "min_gap_m": self.safe_gap_m * 0.5,
                "max_decel_ms2": _A_HARD,
            },
            "ambiguity": {"classification_ambiguity": self.classification_ambiguity},
        }


def _integrate(sc: StationaryLeadScenario, policy: str):
    """Step-integrate ego kinematics toward the stationary lead for one archetypal policy.

    Returns per-step arrays (ego_v, ego_s, gap) plus the brake-onset index and a lead_dropped flag.
    """
    T, dt, v0 = sc.steps, sc.dt, sc.v_cruise
    a = float(np.clip(sc.classification_ambiguity, 0.0, 1.0))

    ego_v = np.empty(T)
    ego_s = np.empty(T)
    gap = np.empty(T)
    v = v0
    s = 0.0
    collided = False
    lead_dropped = False
    brake_onset = -1
    latched = False        # once a policy commits to braking it holds it until stopped

    if policy == "imagination_forward":
        # Forward-model: begin a comfortable brake as soon as the *consequence* of the closing gap
        # is unsafe, i.e. once the gap falls to the comfortable stopping distance + safe margin.
        # Independent of classification_ambiguity (no class prior to be wrong about).
        d_trigger = v0 * v0 / (2 * _A_COMFORT) + sc.safe_gap_m + 5.0   # forward-looking margin
        decel = _A_COMFORT
    else:  # detection_reactive
        # React on a TTC threshold that *slips later* as ambiguity rises; past a drop threshold the
        # lead is lost entirely and only a panic brake fires (far too late).
        lead_dropped = a >= 0.75
        ttc_thresh = 2.0 * (1.0 - 0.5 * a)         # 2.0 s -> 1.0 s across the ambiguity range
        decel = _A_HARD
        d_trigger = None                            # computed per-step from TTC

    for i in range(T):
        g = sc.lead_s - s
        gap[i] = max(g, 0.0)
        ego_v[i] = v
        ego_s[i] = s

        if not collided and g <= _COLLISION_GAP_M and v > 0.1:
            collided = True

        # decide braking (latched: once a policy commits, it brakes until stopped)
        if collided:
            v = 0.0                                 # crash stop
        else:
            if not latched:
                if policy == "imagination_forward":
                    trigger = g <= d_trigger
                elif lead_dropped:
                    # dropped the lead: only a panic brake at a very short gap (too late to stop)
                    trigger = g <= (v0 * v0 / (2 * _A_HARD)) * 0.45
                else:
                    trigger = (g / max(v, 1e-3)) <= ttc_thresh
                if trigger:
                    latched = True
                    brake_onset = i
            if latched:
                v = max(v - decel * dt, 0.0)
        # advance
        s = s + v * dt

    return ego_v, ego_s, gap, brake_onset, collided, lead_dropped


def simulate_policy(sc: StationaryLeadScenario, policy: str = "imagination_forward") -> dict:
    """Emit a ScenarioTelemetry-shaped design-oracle log for one archetypal ``policy``.

    ``policy="detection_reactive"``   brakes on a TTC threshold that slips later with ambiguity, and
                                      drops the lead past a threshold (the documented failure).
    ``policy="imagination_forward"``  H15 forward-model; smooth early brake, invariant to ambiguity,
                                      holds a latent lead estimate (A9), never contacts the lead.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    T, dt, v0 = sc.steps, sc.dt, sc.v_cruise
    t = np.arange(T)
    ego_v, ego_s, gap, brake_onset, collided, lead_dropped = _integrate(sc, policy)

    # ---- derived kinematics ---------------------------------------------------------------- #
    ego_jerk = np.gradient(np.gradient(ego_v, dt), dt)
    steer_rate = np.abs(np.gradient(0.01 * np.sin(t / 13.0), dt))   # lane-keeping only (longitudinal scene)

    # policy-independent flags off the *nominal* (unbraked) approach: the object is physically
    # visible from _SENSOR_RANGE_M, and "ambiguous to classify" within that band. The damning point
    # is that hazard_los is True (visible) yet the reactive stack still acts late.
    nominal_s = np.clip(v0 * dt * t, 0, sc.lead_s)
    nominal_gap = sc.lead_s - nominal_s
    hazard_los = nominal_gap <= _SENSOR_RANGE_M
    is_occluded = (nominal_gap <= _SENSOR_RANGE_M) & (nominal_gap > _COLLISION_GAP_M)  # "ambiguous band"

    # ground-truth lead position (constant); wm estimate tracks it unless the reactive policy drops it
    gt_xy = np.stack([np.full(T, sc.lead_s), np.zeros(T)], axis=1)
    in_range = nominal_gap <= _SENSOR_RANGE_M
    wm_xy = gt_xy + np.random.default_rng(0).normal(0, 0.3, gt_xy.shape)
    wm_xy[~in_range] = np.nan
    if policy == "detection_reactive" and lead_dropped:
        wm_xy[in_range] = np.nan            # lost the lead exactly where it mattered

    # ---- outcome signals (the H15-vs-detection verdict) ------------------------------------ #
    approaching = (ego_v > 0.5) & (gap > _COLLISION_GAP_M)
    if approaching.any():
        ttc_series = gap[approaching] / np.maximum(ego_v[approaching], 1e-3)
        min_ttc = float(np.min(ttc_series))
    else:
        min_ttc = float("inf")
    min_ttc = 0.0 if collided else min_ttc
    min_gap = float(np.min(gap))

    # braking-onset lead time (LAL-v2 anticipation): time between a fixed reference point (when a
    # reasonable anticipatory brake *should* start, at ttc_ref) and when this policy actually starts
    # braking. Positive => anticipatory (braked before the reference); negative => late/reactive.
    ttc_ref = 2.5
    nominal_ttc = nominal_gap / max(v0, 1e-3)
    ref_hits = np.flatnonzero(nominal_ttc <= ttc_ref)
    ref_idx = int(ref_hits[0]) if ref_hits.size else T
    onset_idx = brake_onset if brake_onset >= 0 else T
    brake_onset_lead_time = float((ref_idx - onset_idx) * dt)

    latency = np.full(T, 18.0 if policy == "imagination_forward" else 40.0)
    params_billions = sc.params_billions if policy == "imagination_forward" else 15.0

    return {
        "ego_v": ego_v,
        "ego_jerk": ego_jerk,
        "steer_rate": steer_rate,
        "latency_ms": latency,
        "hazard_los_flag": hazard_los,
        "dist_to_blind_spot": gap,                 # actual gap to the stationary lead
        "is_occluded_flag": is_occluded,
        "wm_hazard_xy": wm_xy,
        "gt_hazard_xy": gt_xy,
        "dt": dt,
        "collisions": int(collided),
        "ego_mass_kg": sc.ego_mass_kg,
        "params_billions": params_billions,
        "_extra": {
            "brake_onset_lead_time_s": brake_onset_lead_time,
            "min_ttc_s": min_ttc,
            "min_gap_m": min_gap,
            "collided": bool(collided),
            "lead_dropped": bool(lead_dropped),
            "classification_ambiguity": float(sc.classification_ambiguity),
            "policy": policy,
        },
    }


def _sweep(policy: str, ambiguities=None, base: StationaryLeadScenario | None = None):
    if ambiguities is None:
        ambiguities = [0.0, 0.25, 0.5, 0.75, 1.0]
    base = base or StationaryLeadScenario()
    out = []
    for a in ambiguities:
        sc = StationaryLeadScenario(**{**base.__dict__, "classification_ambiguity": float(a)})
        out.append(simulate_policy(sc, policy)["_extra"])
    return out


def collision_rate(policy: str, ambiguities=None, base: StationaryLeadScenario | None = None) -> float:
    """Fraction of runs that contact the stationary lead across the ambiguity sweep.

    The forward-model bar is **exactly 0.0**; a detection-reactive policy's rate rises as ambiguity
    (weak class prior / degraded channel) grows and it drops the lead.
    """
    ex = _sweep(policy, ambiguities, base)
    return sum(e["collided"] for e in ex) / len(ex)


def mean_lead_time(policy: str, ambiguities=None, base: StationaryLeadScenario | None = None) -> float:
    """Mean braking-onset lead time (LAL-v2) across the ambiguity sweep (higher = earlier braking)."""
    ex = _sweep(policy, ambiguities, base)
    return float(np.mean([e["brake_onset_lead_time_s"] for e in ex]))
