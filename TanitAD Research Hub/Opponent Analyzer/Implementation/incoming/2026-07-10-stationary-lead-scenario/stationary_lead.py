"""Stationary-object / same-lane lead response — weak-spot eval scenario
(Opponent Analyzer scenario feed, 2026-07-10).

WHY THIS EXISTS
---------------
Weakness **W-08** (baseline driving-competence gaps) in the Opponent Analyzer catalog, and
**SC-13** in ``SCENARIO_DATABASE.md``. NHTSA's ODI opened a preliminary evaluation (2026-05-06,
PE published 05-08) into **Avride** (Uber's robotaxi partner, Yandex SDG lineage) after **16
crashes + 1 minor injury** in Dallas/Austin. The regulator's own words describe the failure class
verbatim: the vehicles **"did not brake for slow-moving or stopped vehicles, and struck stationary
objects partially blocking the roadway"** (most below 20 mph); the probe examines **"conflict
avoidance, driving behavior competence and assertiveness."**
— https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/

This is the *cheapest, broadest* failure surface there is: not an exotic edge case but the mundane
longitudinal task of slowing for a stopped lead / stationary object. It is the same mechanism that
underlies "phantom" and *late* braking across the field (radar stationary-object filtering,
classification-gated braking). This module turns that documented failure into a repeatable eval
scenario.

THE MECHANISM WE ARE MODELLING (and why H15 beats it)
-----------------------------------------------------
A **detection-then-react** stack brakes only once a stopped object is *confidently classified*.
A stationary object against road clutter is exactly the ambiguous case a classifier resolves
**late** (short range). By then the braking distance (∝ v²) may exceed the remaining gap → a
crash or a violent near-miss. The class label is on the critical path.

TanitAD's **H15** imagination forward-models the *consequence* of the closing gap — time-to-contact
from range + range-rate — which is available continuously and needs **no class label**. It begins a
gentle, comfort-bounded deceleration as soon as the forward-modelled TTC drops below a threshold,
long before any classifier would commit. The advantage is therefore *structural*: it comes from
acting on geometry the competitor has, but ignores until classification.

**Honesty (P8):** the advantage is *specifically* about acting-before-classification. If the
competitor's perception classified the object early (large ``detect_range_m``), the safety gap
would close — see ``brake_lead_time`` and ``test_advantage_vanishes_with_early_classification``.
That falsifier is built into the oracle on purpose.

WHAT THIS IS
------------
A **pure, sim-agnostic scenario specification + synthetic-telemetry generator**, offline-testable
without a simulator (mirrors ``stop_arm_gate.py`` / ``work_zone_phantom.py``):

1. ``StationaryLeadScenario`` — a dataclass describing the geometry (ego cruising at ``v_cruise``
   toward a stationary/slow lead at initial range ``gap0`` that the competitor only classifies
   within ``detect_range_m``). ``carla_recipe()`` returns props/waypoints to build it on the
   **CARLA-on-pod** harness (D-014, blocked_route/stationary-object family, W31-32).

2. ``simulate_policy(scenario, policy=...)`` — emits a telemetry log in the exact
   ``ScenarioTelemetry`` field contract of the Benchmarks & Eval metric suite, for two archetypes:
     - ``"classifier_react"`` — the documented failure: brakes hard only once the object is within
       classification range; holds no latent estimate of the lead before that. Late, high-energy,
       collides when the closing speed makes the fixed detection range insufficient.
     - ``"imagination_forward"`` — TanitAD-style **H15**: forward-models TTC from range/range-rate,
       begins a comfort-bounded deceleration as soon as TTC < ``comfort_ttc_s``, and holds a latent
       estimate of the lead from the moment it is in line-of-sight (no class needed).

   The synthetic telemetry is a **design oracle**: it encodes what the scenario is *for* so the
   discriminative structure is testable now, before CARLA is wired. It is NOT a claim about our
   real model (P8) — real numbers come from rolling our checkpoint through ``carla_recipe()`` on
   the pod, and from the real-comma2k19 open-loop probe (DataEng handoff below).

PRIMARY METRICS (owner: Benchmarks & Eval)
------------------------------------------
- **collision rate** over the approach-speed sweep — bar is **exactly 0** for H15 imagination.
- **braking-onset lead time (LAL-v2 style, 2026-07-09)** — seconds by which imagination begins
  decelerating ahead of the classifier-react baseline.
- **min-TTC distribution** and **OKRI** toward the lead (kinetic energy carried into the closing
  gap). Secondary: comfort (peak |jerk|). ``_extra`` carries ``collision``, ``brake_onset_s``,
  ``min_ttc_s``, ``min_range_m``, ``okri_proxy``, ``peak_jerk``, ``v_cruise``, ``detect_range_m``.
  **Handoff to Thursday's Benchmarks & Eval agent:** add a ``collision_rate`` reducer over
  ``_extra.collision`` and reuse ``compute_lal`` (v2) on ``ego_v`` for the anticipation lead.

REAL-DATA HANDOFF (owner: Data Engineering, Tuesday)
----------------------------------------------------
SC-13 is the one scenario with abundant *real, license-clean* support: **comma2k19** has plentiful
lead-vehicle following. Tag slow/stopped-lead segments (radar/vision lead range from the log, or
low-throttle deceleration events) → an **open-loop probe**: does our imagination-error / predicted
TTC lead the actual brake-onset by more than a detection-only baseline on matched segments? Falsifier
(pre-registered): if lead time ≤ the detection baseline on matched stopped-lead segments, the
H15-vs-detection advantage is unproven on real data.

CONTRACT (mirrors ScenarioTelemetry, tanitad_metrics.py)
--------------------------------------------------------
``simulate_policy()`` returns a dict with the ScenarioTelemetry keys (see ``TELEMETRY_KEYS``) plus a
scenario-specific ``_extra`` dict. Field remap for this scenario (documented so a test pins it):
``dist_to_blind_spot`` = range to the lead object (the hazard edge); ``gt_hazard_xy`` = the lead
object position; ``wm_hazard_xy`` = the policy's *latent* estimate of the lead (finite once held);
``is_occluded_flag`` = the **pre-classification (ambiguous) phase** — the object is visible but not
yet confidently classified (there is no literal occlusion here; this is the class-ambiguity analog);
``hazard_los_flag`` = the lead is within sensor line-of-sight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ScenarioTelemetry field names (the metric-suite contract). Kept as a literal so a test can
# assert this module cannot drift from the suite without a deliberate edit here.
TELEMETRY_KEYS = (
    "ego_v", "ego_jerk", "steer_rate", "latency_ms", "hazard_los_flag",
    "dist_to_blind_spot", "is_occluded_flag", "wm_hazard_xy", "gt_hazard_xy",
    "dt", "collisions", "ego_mass_kg", "params_billions",
)

POLICIES = ("classifier_react", "imagination_forward")

APPROACH_M = 30.0          # matches OKRI_APPROACH_M in the metric suite


@dataclass
class StationaryLeadScenario:
    """A stationary / slow lead object in the ego lane.

    Longitudinal geometry along a straight lane (metres of down-route distance):

        ego(s=0) ───────── cruise ──────────►         [lead object @ gap0, speed v_lead]
                                              |◄── detect_range_m ──►|
                                              classifier only commits inside detect_range_m;
                                              range + range-rate (⇒ TTC) are available the whole time.

    The lead is fully visible from the start (``hazard_los``); what is *late* is the confident
    **classification** (``detect_range_m``). A detection-gated policy waits for the class label; an
    imagination policy acts on TTC. Defaults are tuned so imagination is collision-free across the
    whole approach-speed sweep while the classifier-react baseline collides once closing speed makes
    the fixed detection range insufficient (braking distance ∝ v²).
    """
    name: str = "stationary_lead"
    steps: int = 200
    dt: float = 0.1
    v_cruise: float = 15.0             # m/s approach (~54 km/h urban arterial)
    v_lead: float = 0.0                # lead speed (0 = fully stationary object / stalled vehicle)
    gap0: float = 110.0               # initial range to the lead (m)
    detect_range_m: float = 20.0      # range at which the competitor confidently classifies (m)
    comfort_ttc_s: float = 4.5        # TTC threshold at which H15 begins a comfort-bounded stop (s)
    a_comfort: float = 3.0            # comfort deceleration for imagination_forward (m/s^2)
    a_hard: float = 6.0              # hard/emergency deceleration for classifier_react (m/s^2)
    sensor_range_m: float = 130.0    # line-of-sight range (object visible within this)
    ego_mass_kg: float = 1500.0
    params_billions: float = 4.0     # TanitAD-4B active-param envelope (for CNCE)
    # CARLA build hints (used by carla_recipe; not needed for the offline telemetry oracle)
    carla_map: str = "Town10HD"
    weather: str = "ClearNoon"
    extra: dict = field(default_factory=dict)

    def carla_recipe(self) -> dict:
        """Props + waypoints to build the scenario on the CARLA-on-pod harness (blocked_route family)."""
        stalled = self.v_lead <= 1e-6
        return {
            "map": self.carla_map,
            "weather": self.weather,
            "camera": {"channels": 6, "size": 256, "stack": 2},  # base250cam contract
            "props": [
                # a stalled/disabled vehicle or debris partially blocking the ego lane
                {"type": "vehicle.stalled" if stalled else "vehicle.slow_lead",
                 "s": self.gap0, "lane": "ego", "speed_mps": self.v_lead,
                 "partial_block": True},
            ],
            "actors": [],
            "success": {
                # correct behaviour: smooth early deceleration to a safe following distance, no contact
                "no_collision": True,
                "min_ttc_s_min": 2.0,
                "comfort_bounded": True,          # peak |jerk| under a comfort bound
            },
            "events": {"lead_speed_mps": self.v_lead},
            "perception": {"classify_within_m": self.detect_range_m,
                           "los_within_m": self.sensor_range_m},
        }


def _rollout(sc: StationaryLeadScenario, policy: str):
    """Forward-integrate the ego under one archetypal ``policy``. Returns per-step arrays.

    Deterministic Euler integration. The lead moves at constant ``v_lead`` (0 = stationary).
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    T = sc.steps
    dt = sc.dt
    s_ego = 0.0
    v = float(sc.v_cruise)
    braking = False
    brake_onset_idx = -1
    collided = False

    ego_v = np.empty(T)
    rng = np.empty(T)              # range to the lead
    ttc = np.empty(T)
    wm_finite = np.zeros(T, dtype=bool)   # is the latent lead-estimate held this step?

    for i in range(T):
        s_lead = sc.gap0 + sc.v_lead * (i * dt)
        r = s_lead - s_ego
        closing = v - sc.v_lead
        t_ttc = r / closing if closing > 1e-6 else np.inf

        ego_v[i] = v
        rng[i] = r
        ttc[i] = t_ttc

        in_los = r <= sc.sensor_range_m
        classified = r <= sc.detect_range_m

        # ---- the two archetypes -------------------------------------------------------------- #
        if policy == "imagination_forward":
            # H15: hold a latent estimate of the lead from the moment it is in line-of-sight
            # (no class label needed), and begin a comfort stop when forward-modelled TTC is low.
            wm_finite[i] = in_los
            if in_los and t_ttc <= sc.comfort_ttc_s:
                braking = True
            accel = -sc.a_comfort if braking else 0.0
        else:  # classifier_react
            # brakes hard only once the object is confidently classified (short range); holds no
            # latent estimate before classification.
            wm_finite[i] = classified
            if classified:
                braking = True
            accel = -sc.a_hard if braking else 0.0

        if braking and brake_onset_idx < 0:
            brake_onset_idx = i

        # collision check: contact with the (possibly moving) lead while still closing
        if r <= 0.0 and v > sc.v_lead + 1e-6:
            collided = True

        # integrate; never reverse past the lead speed (came to rest / matched lead)
        v = max(v + accel * dt, sc.v_lead)
        s_ego += v * dt

    return ego_v, rng, ttc, wm_finite, brake_onset_idx, collided


def simulate_policy(sc: StationaryLeadScenario, policy: str = "imagination_forward") -> dict:
    """Emit a ScenarioTelemetry-shaped design-oracle log for one archetypal ``policy``."""
    T = sc.steps
    ego_v, rng, ttc, wm_finite, brake_onset_idx, collided = _rollout(sc, policy)

    # ground-truth lead position (x = down-route position of the lead, y = lane offset ~0)
    t = np.arange(T)
    s_lead = sc.gap0 + sc.v_lead * (t * sc.dt)
    gt_xy = np.stack([s_lead, np.zeros(T)], axis=1)
    # latent estimate: the lead's position where held, NaN otherwise (small tracking noise for H15)
    wm_xy = gt_xy + np.random.default_rng(0).normal(0.0, 0.3, gt_xy.shape)
    wm_xy[~wm_finite] = np.nan

    hazard_los = rng <= sc.sensor_range_m
    # class-ambiguity phase: visible but not yet confidently classified (the SC-13 analog of occlusion)
    is_ambiguous = hazard_los & (rng > sc.detect_range_m)

    ego_jerk = np.gradient(np.gradient(ego_v, sc.dt), sc.dt)
    steer_rate = np.abs(np.gradient(0.01 * np.sin(t / 11.0), sc.dt))
    latency = np.full(T, 18.0 if policy == "imagination_forward" else 40.0)
    params_billions = sc.params_billions if policy == "imagination_forward" else 15.0

    # ---- derived scalar metrics ------------------------------------------------------------ #
    brake_onset_s = float(brake_onset_idx * sc.dt) if brake_onset_idx >= 0 else float("nan")
    closing = ego_v - sc.v_lead
    ttc_closing = np.where(closing > 1e-6, ttc, np.inf)
    min_ttc_s = float(np.min(ttc_closing))
    min_range_m = float(np.min(rng))
    # OKRI proxy: kinetic energy carried into the closing gap over the approach band (lower = safer)
    mask = (rng < APPROACH_M) & (rng > -1.0)
    ke = 0.5 * sc.ego_mass_kg * ego_v ** 2
    okri_proxy = float(np.sum(np.where(mask, ke / (np.abs(rng) + 0.1), 0.0)) * sc.dt)
    peak_jerk = float(np.max(np.abs(ego_jerk)))

    return {
        "ego_v": ego_v,
        "ego_jerk": ego_jerk,
        "steer_rate": steer_rate,
        "latency_ms": latency,
        "hazard_los_flag": hazard_los,
        "dist_to_blind_spot": rng,                    # range to the lead object (hazard edge)
        "is_occluded_flag": is_ambiguous,             # SC-13: pre-classification ambiguity phase
        "wm_hazard_xy": wm_xy,
        "gt_hazard_xy": gt_xy,
        "dt": sc.dt,
        "collisions": int(collided),
        "ego_mass_kg": sc.ego_mass_kg,
        "params_billions": params_billions,
        "_extra": {
            "collision": bool(collided),
            "brake_onset_s": brake_onset_s,
            "min_ttc_s": min_ttc_s,
            "min_range_m": min_range_m,
            "okri_proxy": okri_proxy,
            "peak_jerk": peak_jerk,
            "v_cruise": sc.v_cruise,
            "detect_range_m": sc.detect_range_m,
            "policy": policy,
        },
    }


def collision_rate(policy: str, speeds=None, base: StationaryLeadScenario | None = None) -> float:
    """Collision rate for ``policy`` swept over a set of approach speeds (m/s).

    Returns the fraction of runs in which the ego contacts the stationary lead. The H15-imagination
    bar is **exactly 0.0** across the realistic sweep; the classifier-react baseline collides once
    the closing speed makes its fixed detection range insufficient (braking distance ∝ v²).
    """
    if speeds is None:
        speeds = [8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 25.0]
    base = base or StationaryLeadScenario()
    hits = 0
    for vv in speeds:
        sc = StationaryLeadScenario(**{**base.__dict__, "v_cruise": float(vv)})
        if simulate_policy(sc, policy)["_extra"]["collision"]:
            hits += 1
    return hits / len(speeds)


def brake_lead_time(base: StationaryLeadScenario | None = None) -> float:
    """Seconds by which imagination_forward begins braking ahead of classifier_react.

    Positive = imagination anticipates. This is the ``LAL``-style anticipation lead the scenario is
    built to expose. It *shrinks toward zero* when ``detect_range_m`` is large (the competitor
    classifies early) — that is the honest falsifier, exercised in the tests.
    """
    base = base or StationaryLeadScenario()
    react = simulate_policy(base, "classifier_react")["_extra"]["brake_onset_s"]
    imag = simulate_policy(base, "imagination_forward")["_extra"]["brake_onset_s"]
    if np.isnan(react) or np.isnan(imag):
        return float("nan")
    return float(react - imag)
