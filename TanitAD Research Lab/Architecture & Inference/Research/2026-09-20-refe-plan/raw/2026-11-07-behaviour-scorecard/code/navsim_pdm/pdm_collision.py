"""NAVSIM v1 / nuPlan collision-type classification, VENDORED UNMODIFIED (Apache-2.0; see LICENSE). Stream BEHAV-F1, 2026-11-07, MM ruling A-10.

The four function definitions below are copied byte-for-byte (line endings normalised CRLF -> LF) from:
  * nuplan idm/utils.py  (C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit/nuplan/planning/simulation/observation/idm/utils.py, file sha256 8f1fb6f7ea8d28fcc3a379d3c5442628eba036225a9a1a3f0326087628745389)
      get_agent_relative_angle (lines 19-30), is_agent_behind (lines 44-52, angle_tolerance=150 deg), is_track_stopped (lines 191-202, stopped_speed_threshhold=5e-02)
  * NAVSIM pdm_scorer_utils.py  (F:/Archive/devbox-C/navsim/devkit/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer_utils.py, file sha256 d483fd4b5b3e871d201a8ebab7e7fde6dcb1c1cf6cc608d1a8bbe3e8f74ab9e3)
      get_collision_type (lines 12-64, stopped_speed_threshold=5e-02)
Only the IMPORT block differs from the sources: nuplan/navsim types are supplied by navsim_pdm/nuplan_shim.py (adapter, not vendored) and
StateIndex by navsim_pdm/pdm_enums.py. test_f1_vendored_bodies_equal_sources re-extracts every body from the sources and compares.
"""
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from shapely import LineString, Polygon

from navsim_pdm.nuplan_shim import CollisionType, StateSE2, TrackedObject, Agent
from navsim_pdm.pdm_enums import StateIndex


@lru_cache(maxsize=256)
def get_agent_relative_angle(ego_state: StateSE2, agent_state: StateSE2) -> float:
    """
    Get the the relative angle of an agent position to the ego
    :param ego_state: pose of ego
    :param agent_state: pose of an agent
    :return: relative angle in radians.
    """
    agent_vector: npt.NDArray[np.float32] = np.array([agent_state.x - ego_state.x, agent_state.y - ego_state.y])
    ego_vector: npt.NDArray[np.float32] = np.array([np.cos(ego_state.heading), np.sin(ego_state.heading)])
    dot_product = np.dot(ego_vector, agent_vector / np.linalg.norm(agent_vector))
    return float(np.arccos(dot_product))


def is_agent_behind(ego_state: StateSE2, agent_state: StateSE2, angle_tolerance: float = 150) -> bool:
    """
    Determines if an agent is behind of the ego
    :param ego_state: ego's pose
    :param agent_state: agent's pose
    :param angle_tolerance: tolerance to consider if agent is behind, where zero is the heading of the ego [deg]
    :return: true if agent is behind, false otherwise
    """
    return bool(get_agent_relative_angle(ego_state, agent_state) > np.deg2rad(angle_tolerance))


def is_track_stopped(tracked_object: TrackedObject, stopped_speed_threshhold: float = 5e-02) -> bool:
    """
    Evaluates if a tracked object is stopped
    :param tracked_object: tracked_object representation
    :param stopped_speed_threshhold: Threshhold for 0 speed due to noise
    :return: True if track is stopped else False.
    """
    return (
        True
        if not isinstance(tracked_object, Agent)
        else bool(tracked_object.velocity.magnitude() <= stopped_speed_threshhold)
    )


def get_collision_type(
    state: npt.NDArray[np.float64],
    ego_polygon: Polygon,
    tracked_object: TrackedObject,
    tracked_object_polygon: Polygon,
    stopped_speed_threshold: float = 5e-02,
) -> CollisionType:
    """
    Classify collision between ego and the track.
    :param ego_state: Ego's state at the current timestamp.
    :param tracked_object: Tracked object.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return Collision type.
    """

    ego_speed = np.hypot(
        state[StateIndex.VELOCITY_X],
        state[StateIndex.VELOCITY_Y],
    )

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold

    center_point = tracked_object_polygon.centroid
    tracked_object_center = StateSE2(center_point.x, center_point.y, tracked_object.box.center.heading)

    ego_rear_axle_pose: StateSE2 = StateSE2(*state[StateIndex.STATE_SE2])

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif is_track_stopped(tracked_object):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString(
        [
            ego_polygon.exterior.coords[0],
            ego_polygon.exterior.coords[3],
        ]
    ).intersects(tracked_object_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type
