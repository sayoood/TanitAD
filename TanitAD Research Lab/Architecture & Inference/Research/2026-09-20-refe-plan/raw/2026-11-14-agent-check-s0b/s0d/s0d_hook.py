"""S0-D hook: modify the scenario's FUTURE tracked objects BEFORE NAVSIM builds its observation.

THE HOOK POINT (narrowest).  NAVSIM's `MetricCacheProcessor._interpolate_gt_observation(scenario)` reads the ground-truth
detections ONLY through `scenario.get_tracked_objects_at_iteration(iteration=...)`, for gt_indices = 0, 10, ..., 100 (0.5 s steps):
    navsim @ 3e8291b  navsim/planning/metric_caching/metric_cache_processor.py:110-113
It also reads `scenario.database_interval` (line 105).  `HookedScenario` below is a proxy that supplies exactly those two members
(everything else is delegated), so the MODIFIED list flows through NAVSIM's OWN interpolation (StateInterpolator, lines 115-206),
PDMObservation (line 200-206) and PDMScorer.  The ego, the map, PDM-Closed's planner input and the proposals are untouched: the hook
is passed ONLY to `_interpolate_gt_observation`, never to `_get_planner_inputs`.

Arms (all are functions of the TRUE tracks of the 11 iterations; statics are NEVER modified in any arm):
  true   identity: the very same objects are returned (no copy).
  t0     every dynamic agent present at iteration 0 is placed, at EVERY iteration, at its t0 box (pose, size, heading), velocity 0.
         Dynamic agents absent at iteration 0 do not exist (single-frame perception).
  t0v    pose and box frozen at t0 exactly as in t0, but each agent keeps its TRUE t0 velocity on the object (so is_track_stopped classifies
         it as on the real scene).  Secondary arm: separates pose from the stopped-track classification.
  cv     every dynamic agent present at iteration 0 is placed at t0 centre + (vx, vy)_t0 * t, heading, size fixed, velocity (vx, vy)_t0.
         (vx, vy) is the nuPlan Agent.velocity, i.e. the lidar_box vx, vy in the GLOBAL frame, so the propagation is global.
  none   all dynamic agents removed at every iteration.
Dynamic = tracked_object_type in NAVSIM's AGENT_TYPES (vehicle, pedestrian, bicycle; EGO never occurs in a detection list).
"""
from __future__ import annotations

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

MODES = ("true", "t0", "cv", "none", "t0v")


def is_dynamic(obj) -> bool:
    return obj.tracked_object_type in AGENT_TYPES


def _agent_at(proto, x, y, heading, vx, vy):
    """A new Agent with proto's token / type / size / metadata at the given pose and velocity (global frame)."""
    box = OrientedBox(StateSE2(float(x), float(y), float(heading)), proto.box.length, proto.box.width, proto.box.height)
    return Agent(tracked_object_type=proto.tracked_object_type, oriented_box=box, velocity=StateVector2D(float(vx), float(vy)),
                 metadata=proto.metadata, angular_velocity=proto.angular_velocity)


def transform_one(objs, t0_dyn, t_s, mode):
    """Objects of ONE iteration -> modified object list.  objs: list of tracked objects at this iteration; t0_dyn: {token: Agent}
    of the dynamic agents at iteration 0; t_s: seconds since iteration 0."""
    if mode == "true":
        return list(objs)
    statics = [o for o in objs if not is_dynamic(o)]                       # statics: unchanged object, NAVSIM treats them as it does
    if mode == "none":
        return statics
    agents = []
    for tok, a in t0_dyn.items():
        if mode == "t0":
            agents.append(_agent_at(a, a.center.x, a.center.y, a.center.heading, 0.0, 0.0))
        elif mode == "t0v":
            agents.append(_agent_at(a, a.center.x, a.center.y, a.center.heading, a.velocity.x, a.velocity.y))
        elif mode == "cv":
            vx, vy = a.velocity.x, a.velocity.y
            agents.append(_agent_at(a, a.center.x + vx * t_s, a.center.y + vy * t_s, a.center.heading, vx, vy))
        else:
            raise ValueError(mode)
    return statics + agents


def transform_tracks(tracks, times_s, mode):
    """tracks: list of DetectionsTracks (one per iteration, the first is t0); times_s: seconds since the first -> list of
    DetectionsTracks.  mode 'true' returns the INPUT LIST ITSELF (identity, no copy)."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "true":
        return tracks
    t0_dyn = {o.track_token: o for o in tracks[0].tracked_objects.tracked_objects if is_dynamic(o)}
    return [DetectionsTracks(TrackedObjects(transform_one(tr.tracked_objects.tracked_objects, t0_dyn, t, mode)))
            for tr, t in zip(tracks, times_s)]


class HookedScenario:
    """Proxy for `_interpolate_gt_observation`: get_tracked_objects_at_iteration returns the arm's modified tracks.  The true
    tracks are fetched once per iteration from the real scenario (and shared across arms through `cache`)."""

    def __init__(self, real, mode, cache=None):
        if mode not in MODES:
            raise ValueError(mode)
        self._real, self._mode = real, mode
        self._cache = cache if cache is not None else {}
        self._out = {}

    def __getattr__(self, name):
        return getattr(self._real, name)

    @property
    def database_interval(self):
        return self._real.database_interval

    def _true(self, iteration):
        if iteration not in self._cache:
            self._cache[iteration] = self._real.get_tracked_objects_at_iteration(iteration=iteration)
        return self._cache[iteration]

    def get_tracked_objects_at_iteration(self, iteration, future_trajectory_sampling=None):
        if self._mode == "true":
            return self._true(iteration)
        if iteration not in self._out:
            self._out[iteration] = transform_tracks([self._true(0), self._true(iteration)],
                                                    [0.0, iteration * self._real.database_interval], self._mode)[1]
        return self._out[iteration]
