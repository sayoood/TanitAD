"""The SUPERSEDED f1 at-fault rule (first implementation, reconstructed from memory, UNVERIFIED -- replaced by AMENDMENT_1 ruling A-10).
Kept ONLY as the deliberate-regression arm: swapping it back in must turn the A-10 tests RED (test_f1_old_bearing_rule_turns_the_a10_scenarios_red,
mutation_check.py kind 'oldrule'). Same signature as the vendored get_collision_type so it can be dropped in. NOT used by the scorer."""
import math

import numpy as np

from navsim_pdm.nuplan_shim import Agent, CollisionType

FRONT_DEG = 45.0
F2_V_STOPPED = 0.005          # the old ego/track "stopped" threshold (NAVSIM source: 5e-02)


def old_collision_type(state, ego_polygon, tracked_object, tracked_object_polygon, stopped_speed_threshold=F2_V_STOPPED):
    ego_speed = float(np.hypot(state[3], state[4]))
    c = ego_polygon.centroid
    a = tracked_object_polygon.centroid
    dx, dy = a.x - c.x, a.y - c.y
    cs, sn = math.cos(state[2]), math.sin(state[2])
    bearing = math.degrees(math.atan2(-sn * dx + cs * dy, cs * dx + sn * dy))   # agent centre in the ego BOX-CENTRE frame
    if not isinstance(tracked_object, Agent):
        return CollisionType.STOPPED_TRACK_COLLISION                             # old: static object = at fault, ahead of the ego-stopped test
    if ego_speed <= stopped_speed_threshold:
        return CollisionType.STOPPED_EGO_COLLISION
    if tracked_object.velocity.magnitude() <= stopped_speed_threshold:
        return CollisionType.STOPPED_TRACK_COLLISION
    if abs(bearing) <= FRONT_DEG:
        return CollisionType.ACTIVE_FRONT_COLLISION
    if abs(bearing) >= 180.0 - FRONT_DEG:
        return CollisionType.ACTIVE_REAR_COLLISION
    return CollisionType.ACTIVE_LATERAL_COLLISION
