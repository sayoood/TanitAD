"""Work-Zone Phantom — weak-spot eval scenario (Opponent Analyzer monthly feed, 2026-07-17).

WHY THIS EXISTS
---------------
Weakness **W-01** in the Opponent Analyzer catalog: on 2026-06-18 Waymo recalled 3,871
robotaxis after 13 incidents driving into freeway **construction zones** — failing to
recognize ramp-closure signs (Phoenix) and driving *between* lane-closure cones (SF Bay).
The mechanism (INFER) is a divergence between the operator's *prior* road topology and the
*posterior* reality of a closed/altered drivable area, compounded by an **occluded actor**
behind the cone taper. This module turns that public failure into a repeatable eval scenario.

WHAT THIS IS
------------
A **pure, sim-agnostic scenario specification + synthetic-telemetry generator**. It does two
things, both offline-testable without a simulator:

1. ``WorkZonePhantomScenario`` — a dataclass describing the work-zone geometry (cone taper,
   ramp-closure sign, closed lane, occluded merging actor). ``carla_recipe()`` returns the
   props/waypoints needed to build it on the **CARLA-on-pod** harness (D-014: MetaDrive is
   retired; the sim arm is CARLA-in-Docker on RunPod, W31-32).

2. ``simulate_policy(scenario, policy=...)`` — emits a telemetry log **in the exact
   ``ScenarioTelemetry`` field contract** of the Benchmarks & Eval metric suite
   (`Benchmarks & Eval/Implementation/incoming/2026-07-16-eval-metric-suite/tanitad_metrics.py`),
   for two archetypal policies:
     - ``"reactive"``  — pixel-reactive baseline (standard E2E): brakes only once the hazard
       reaches line-of-sight, holds speed toward the blind cone edge, holds NO latent estimate
       of the occluded actor.
     - ``"world_model"`` — TanitAD-style: imagines the changed drivable area + the occluded
       actor, brakes *before* line-of-sight, throttles on the blind-spot's epistemic sigma, and
       holds a latent estimate under occlusion.
   The synthetic telemetry is a **design oracle**: it encodes what the scenario is *for* so the
   discriminative structure (reactive worse than world_model on LAL/OKRI/LOPS) is testable now,
   before CARLA is wired. It is NOT a claim about our real model (P8) — the real numbers come
   from running our checkpoint through this scenario on the pod.

CONTRACT (mirrors ScenarioTelemetry, tanitad_metrics.py L103-141)
-----------------------------------------------------------------
``telemetry()`` returns a dict with keys: ego_v, ego_jerk, steer_rate, latency_ms,
hazard_los_flag, dist_to_blind_spot, is_occluded_flag, wm_hazard_xy, gt_hazard_xy, dt,
collisions, ego_mass_kg, params_billions  — plus a scenario-specific ``_extra`` dict with
``closure_incursion_m`` (metres the ego drove into the closed lane past the cone taper), the
seed of a future H9 rule-compliance / violation-rate signal (owner: Benchmarks & Eval).

NEXT STEP (explicit)
--------------------
On the CARLA-on-pod harness: build the scene from ``carla_recipe()``, roll out our trained
checkpoint, log real telemetry into ``ScenarioTelemetry``, and score with
``tanitad_metrics.scenario_metrics``. This module's ``simulate_policy`` is replaced by the real
rollout; the scenario geometry + the ``_extra`` compliance signal stay.
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

POLICIES = ("reactive", "world_model")


@dataclass
class WorkZonePhantomScenario:
    """A construction-zone scenario: ramp-closure sign + cone taper + occluded merging actor.

    Geometry is longitudinal along the ego's approach (metres of down-route distance ``s``):

        s=0                sign_s        taper_s      los_s        end
        |--- free drive ---|--- sign ----|== cones ==|== actor visible ==|
                                          ^ blind edge (dist_to_blind_spot -> 0)
                                          the occluded actor merges from behind the taper

    All fields are pure numbers; nothing here imports a simulator.
    """
    name: str = "work_zone_phantom"
    steps: int = 200
    dt: float = 0.1
    v_cruise: float = 16.0           # m/s approach speed (~58 km/h)
    sign_s: float = 60.0             # ramp-closure sign appears (m down-route)
    taper_s: float = 90.0            # cone taper / blind edge begins (m)
    los_s: float = 110.0             # occluded actor reaches line-of-sight (m)
    closed_lane_len: float = 40.0    # length of the closed lane past the taper (m)
    ego_mass_kg: float = 1500.0
    params_billions: float = 4.0     # TanitAD-4B active-param envelope (for CNCE)
    # CARLA build hints (used by carla_recipe; not needed for the offline telemetry oracle)
    carla_map: str = "Town04"        # has freeway + ramps
    weather: str = "ClearNoon"
    extra: dict = field(default_factory=dict)

    # ---- pure geometry helpers ------------------------------------------------------------ #
    def route_s(self, ego_v: np.ndarray) -> np.ndarray:
        """Cumulative down-route distance from a speed profile (trapezoidal integral)."""
        s = np.cumsum(ego_v * self.dt)
        return s - ego_v[0] * self.dt  # start at ~0

    def carla_recipe(self) -> dict:
        """Props + waypoints to build the scenario on the CARLA-on-pod harness.

        Returned dict is a plain spec (no CARLA import): the pod-side builder maps
        ``props`` to CARLA blueprints and ``events`` to actor triggers keyed on the ego's
        down-route distance ``s``.
        """
        return {
            "map": self.carla_map,
            "weather": self.weather,
            "camera": {"channels": 6, "size": 256, "stack": 2},  # base250cam contract
            "props": [
                {"type": "warning_sign.ramp_closure", "s": self.sign_s, "lane": "ego"},
                {"type": "constructioncone.taper", "s0": self.taper_s,
                 "s1": self.taper_s + self.closed_lane_len, "into_lane": "ego"},
                {"type": "static.workzone_barrier", "s": self.taper_s + 2.0},
            ],
            "actors": [
                # merging worker/vehicle occluded behind the taper until los_s
                {"type": "walker.worker", "occluded_until_s": self.los_s,
                 "spawn_s": self.taper_s + 6.0, "merge_from": "closed_lane"},
            ],
            "success": {
                # correct behaviour: do NOT enter the closed lane; slow before the blind edge
                "no_closure_incursion": True,
                "min_clearance_m": 2.0,
                "brake_before_los": True,
            },
            "events": {"actor_merge_trigger_s": self.los_s},
        }


def _base_profiles(sc: WorkZonePhantomScenario):
    """Shared per-step scaffolding: distance-to-blind-edge and occlusion/LoS flags.

    Built from a nominal *constant-cruise* route so both policies see the same geometry;
    each policy then perturbs its own speed/braking on top.
    """
    T = sc.steps
    t = np.arange(T)
    nominal_s = sc.v_cruise * sc.dt * t                          # constant-cruise route
    dist_to_blind = np.abs(sc.taper_s - nominal_s)              # 0 at the cone edge
    # actor is fully occluded from when the ego nears the taper until line-of-sight at los_s
    is_occluded = (nominal_s >= sc.taper_s - 20.0) & (nominal_s < sc.los_s)
    hazard_los = nominal_s >= sc.los_s
    # ground-truth hidden actor position (down-route, in the closed lane), simple merge track
    gt_xy = np.stack([np.clip(nominal_s, 0, sc.los_s + 20.0),
                      np.full(T, -1.75)], axis=1)                # ~one lane to the side
    return t, nominal_s, dist_to_blind, is_occluded, hazard_los, gt_xy


def simulate_policy(sc: WorkZonePhantomScenario, policy: str = "world_model") -> dict:
    """Emit a ScenarioTelemetry-shaped design-oracle log for one archetypal ``policy``.

    ``policy="reactive"``      pixel-reactive baseline (brakes at LoS, no latent tracking).
    ``policy="world_model"``   imagines the closure + occluded actor (brakes early, tracks it).
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    T = sc.steps
    t, nominal_s, dist_to_blind, is_occluded, hazard_los, gt_xy = _base_profiles(sc)

    los_idx = int(np.flatnonzero(hazard_los)[0]) if hazard_los.any() else T - 1
    taper_idx = int(np.argmin(dist_to_blind))

    ego_v = np.full(T, sc.v_cruise, dtype=float)
    if policy == "world_model":
        # brake BEFORE line-of-sight: start slowing ~1.5 s ahead of the blind edge
        brake_start = max(0, taper_idx - 15)
        ramp = np.clip((t - brake_start) / 12.0, 0.0, 1.0)
        ego_v = sc.v_cruise * (1.0 - 0.75 * ramp)              # slow to ~25% before the edge
        # holds a latent estimate of the occluded actor (small tracking error)
        wm_xy = gt_xy + np.random.default_rng(0).normal(0, 0.3, gt_xy.shape)
        wm_xy[~is_occluded] = np.nan                            # only meaningful under occlusion
        closure_incursion = 0.0                                 # respects the closure
        latency = np.full(T, 18.0)                              # ms, small model
        params_billions = sc.params_billions
        collisions = 0
    else:  # reactive
        # holds cruise speed until the hazard's pixels appear, then brakes hard after a
        # ~0.2 s reaction latency (a reactive policy responds strictly *after* line-of-sight)
        brake_start = min(T - 1, los_idx + 2)
        ramp = np.clip((t - brake_start) / 6.0, 0.0, 1.0)
        ego_v = sc.v_cruise * (1.0 - 0.9 * ramp)               # late, hard braking
        wm_xy = np.full_like(gt_xy, np.nan)                    # no latent estimate at all
        # drives into the closed lane past the taper before reacting
        closure_incursion = float(max(0.0, (los_idx - taper_idx)) * sc.v_cruise * sc.dt)
        latency = np.full(T, 40.0)                              # ms, larger reactive stack
        params_billions = 15.0
        collisions = 0

    # kinematics derived from the speed profile
    ego_jerk = np.gradient(np.gradient(ego_v, sc.dt), sc.dt)
    steer_rate = np.abs(np.gradient(
        0.02 * np.sin(t / 9.0) + (0.0 if policy == "world_model" else 0.05 * (t >= brake_start)),
        sc.dt))

    return {
        "ego_v": ego_v,
        "ego_jerk": ego_jerk,
        "steer_rate": steer_rate,
        "latency_ms": latency,
        "hazard_los_flag": hazard_los,
        "dist_to_blind_spot": dist_to_blind,
        "is_occluded_flag": is_occluded,
        "wm_hazard_xy": wm_xy,
        "gt_hazard_xy": gt_xy,
        "dt": sc.dt,
        "collisions": collisions,
        "ego_mass_kg": sc.ego_mass_kg,
        "params_billions": params_billions,
        "_extra": {"closure_incursion_m": closure_incursion, "policy": policy},
    }
