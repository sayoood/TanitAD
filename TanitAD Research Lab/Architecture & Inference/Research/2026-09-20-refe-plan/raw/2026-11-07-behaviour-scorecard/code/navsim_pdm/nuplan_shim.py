"""ADAPTER (not vendored code): the thinnest stand-ins for the nuPlan types that pdm_collision.py's vendored bodies touch, so the scorer runs
without the nuplan package. Stream BEHAV-F1, 2026-11-07.

What the vendored bodies actually use, and where it comes from in nuPlan:
  CollisionType        nuplan/planning/metrics/utils/collision_utils.py:25-32 (IntEnum, same names and values)
  StateSE2             .x .y .heading, hashable (lru_cache key)  -- nuplan/common/actor_state/state_representation.py
  Agent                isinstance() test + .velocity.magnitude() in is_track_stopped; .box.center.heading in get_collision_type
  TrackedObject        every non-Agent (nuPlan StaticObject: cones, barriers, signs, generic objects): is_track_stopped -> True
  AGENT_TYPES          nuplan/common/actor_state/tracked_objects_types.py:53 = {VEHICLE, PEDESTRIAN, BICYCLE, EGO}: exactly nuPlan's Agent subclasses
test_f1_adapter_matches_the_real_nuplan_and_navsim_source_on_fuzzed_geometry fuzzes this adapter against the real nuplan functions when nuplan is importable.
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


@dataclass(frozen=True)
class StateSE2:
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class _Velocity:
    x: float
    y: float

    def magnitude(self):
        return float(np.hypot(self.x, self.y))


@dataclass(frozen=True)
class _Center:
    heading: float


@dataclass(frozen=True)
class _Box:
    center: _Center


class TrackedObject:
    """a non-Agent tracked object (nuPlan StaticObject)"""

    def __init__(self, heading):
        self.box = _Box(_Center(float(heading)))


class Agent(TrackedObject):
    """a moving-capable tracked object (vehicle, pedestrian, bicycle). `speed` is the magnitude of nuPlan's world-frame velocity vector."""

    def __init__(self, heading, speed):
        super().__init__(heading)
        self.velocity = _Velocity(float(speed), 0.0)
