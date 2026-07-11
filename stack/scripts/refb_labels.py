"""REF-B pseudo-labels: target-behavior classes + ego-frame waypoint targets.

Pure functions over episode-contract poses ([T, 4] = x, y, yaw, v) — no model,
no data loading, unit-tested on synthetic trajectories (tests/test_refb.py).

Maneuver classes (tanitad.refs.refb.MANEUVER_CLASSES order):
    0 lane_keep   1 turn_left   2 turn_right   3 accelerate   4 brake_stop

Derivation from future kinematics over the LABEL HORIZON (default 20 steps =
2 s @ 10 Hz, matching the tactical head's farthest waypoint):

  curvature test (priority 1 — a braking turn is a TURN):
      dyaw = wrap_to_pi(yaw[t+H] - yaw[t])
      dyaw >  YAW_TURN_RAD  -> turn_left      (CCW-positive yaw, the repo
      dyaw < -YAW_TURN_RAD  -> turn_right      _ego convention: +y = left)
  accel test (priority 2):
      dv = v[t+H] - v[t]
      dv <  DV_BRAKE_MS  OR  (v[t+H] < STOP_V_MS and v[t] >= MOVING_V_MS)
                            -> brake_stop     (includes braking TO a stop)
      dv >  DV_ACCEL_MS     -> accelerate
  else                      -> lane_keep

Documented thresholds (per 2 s horizon):
  YAW_TURN_RAD = 0.15 rad (~8.6 deg over 2 s ~ a deliberate turn/lane change
                 onset; highway curve drift at ~0.02-0.05 rad stays lane_keep)
  DV_ACCEL_MS  = +1.0 m/s (mean accel > +0.5 m/s^2 — deliberate speed-up)
  DV_BRAKE_MS  = -1.0 m/s (mean decel > 0.5 m/s^2 — deliberate braking)
  STOP_V_MS    = 0.3 m/s, MOVING_V_MS = 1.0 m/s (moving -> stopped counts as
                 brake_stop even when dv is small because v[t] was small)

Waypoint targets are ego-frame displacements at the tactical horizons, using
EXACTLY the repo's `_ego` rotation convention (scripts/d1_probe_capacity.py):
rotate the world displacement by -yaw(t); +x = forward, +y = left.

Nav commands (rev2, strategic layer): `nav_command` derives the per-window
command from the NET heading change over 15-25 s of future poses —
|dheading| > 45 deg -> left/right (CCW-left, same sign convention), else
follow. Windows with fewer than NAV_MIN_STEPS of future get follow +
valid=False (excluded from the route-heading aux CE only). Only
follow/left/right are ever emitted: `straight` (index 3 in
refb.NAV_COMMANDS) is reserved for intersection topologies and kept purely
for interface stability. Expected class balance on comma2k19: the corpus is
highway-dominated, so `follow` dominates heavily (turn-class shares in the
low percent range at best) — the trainer therefore weights the route-heading
aux CE by inverse per-batch class frequency, clamped, so the rare left/right
windows are not drowned out.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# ---- documented thresholds (see module docstring) ---------------------------
YAW_TURN_RAD = 0.15
DV_ACCEL_MS = 1.0
DV_BRAKE_MS = -1.0
STOP_V_MS = 0.3
MOVING_V_MS = 1.0

LABEL_HORIZON = 20                 # 2 s @ 10 Hz (tactical farthest waypoint)

LANE_KEEP, TURN_LEFT, TURN_RIGHT, ACCELERATE, BRAKE_STOP = range(5)

# ---- nav-command derivation (rev2, strategic layer) --------------------------
# Indices match tanitad.refs.refb.NAV_COMMANDS; NAV_STRAIGHT is reserved for
# intersection topologies and never emitted here (interface stability only).
NAV_FOLLOW, NAV_LEFT, NAV_RIGHT, NAV_STRAIGHT = range(4)
NAV_TURN_RAD = math.pi / 4         # 45 deg net heading change -> left/right
NAV_HORIZON_STEPS = 250            # 25 s @ 10 Hz (route-scale lookahead)
NAV_MIN_STEPS = 150                # < 15 s of future -> follow + valid=False

# Route-heading aux classes — order matches tanitad.refs.refb.ROUTE_CLASSES.
ROUTE_LEFT, ROUTE_STRAIGHT, ROUTE_RIGHT = range(3)
_NAV_TO_ROUTE = {NAV_FOLLOW: ROUTE_STRAIGHT, NAV_LEFT: ROUTE_LEFT,
                 NAV_RIGHT: ROUTE_RIGHT}


def wrap_to_pi(a: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi] so yaw differences across +-pi stay small."""
    return a - (2 * math.pi) * torch.floor((a + math.pi) / (2 * math.pi))


def ego_frame(dxy: Tensor, yaw: Tensor) -> Tensor:
    """World displacement [..., 2] -> ego frame of `yaw` (d1_probe `_ego`)."""
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


def classify_maneuver(yaw0: Tensor, yaw1: Tensor, v0: Tensor,
                      v1: Tensor) -> Tensor:
    """Vectorized maneuver class from (pose at t, pose at t+H) kinematics.

    All inputs broadcastable [...]; returns int64 class ids [...] per the
    priority order documented in the module docstring (turn > brake > accel).
    """
    dyaw = wrap_to_pi(yaw1 - yaw0)
    dv = v1 - v0
    cls = torch.full(dyaw.shape, LANE_KEEP, dtype=torch.long,
                     device=dyaw.device)
    cls[dv > DV_ACCEL_MS] = ACCELERATE
    brake = (dv < DV_BRAKE_MS) | ((v1 < STOP_V_MS) & (v0 >= MOVING_V_MS))
    cls[brake] = BRAKE_STOP
    cls[dyaw > YAW_TURN_RAD] = TURN_LEFT           # turns override accel/brake
    cls[dyaw < -YAW_TURN_RAD] = TURN_RIGHT
    return cls


def maneuver_labels(poses: Tensor, horizon: int = LABEL_HORIZON) -> Tensor:
    """Per-timestep labels for a whole episode: poses [T, 4] -> [T-horizon].

    Label at t compares pose t with pose t+horizon (future kinematics only —
    a pure function of the trajectory, no actions required)."""
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"poses must be [T, 4], got {tuple(poses.shape)}")
    if poses.shape[0] <= horizon:
        raise ValueError(f"episode too short for labels: T={poses.shape[0]} "
                         f"<= horizon={horizon}")
    p0, p1 = poses[:-horizon], poses[horizon:]
    return classify_maneuver(p0[:, 2], p1[:, 2], p0[:, 3], p1[:, 3])


def window_maneuver_labels(pose_last: Tensor, future_poses: Tensor,
                           horizon: int = LABEL_HORIZON) -> Tensor:
    """Batch labels from window fields: pose_last [B, 4], future_poses
    [B, H, 4] (H >= horizon; future_poses[:, k-1] is k steps past pose_last)."""
    if future_poses.shape[1] < horizon:
        raise ValueError(f"future_poses has only {future_poses.shape[1]} "
                         f"steps; label horizon needs {horizon}")
    p1 = future_poses[:, horizon - 1]
    return classify_maneuver(pose_last[:, 2], p1[:, 2],
                             pose_last[:, 3], p1[:, 3])


def nav_command(poses: Tensor, t: int, horizon_steps: int = NAV_HORIZON_STEPS,
                min_steps: int = NAV_MIN_STEPS) -> tuple[int, bool]:
    """Derive the strategic nav command at timestep ``t`` from future heading.

    Pure function over episode poses [T, 4]. Net heading change over the next
    ``min(horizon_steps, available)`` steps (15-25 s @ 10 Hz at the defaults):
        dyaw >  NAV_TURN_RAD  -> (NAV_LEFT,  True)   (CCW-left, repo `_ego`)
        dyaw < -NAV_TURN_RAD  -> (NAV_RIGHT, True)
        else                  -> (NAV_FOLLOW, True)
    If fewer than ``min_steps`` of future exist the window cannot be judged
    at route scale: returns (NAV_FOLLOW, False) — the command stays a valid
    model input, but the False mask excludes it from the route-heading aux
    CE. NAV_STRAIGHT is never emitted (reserved, see module docstring).
    """
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"poses must be [T, 4], got {tuple(poses.shape)}")
    T = poses.shape[0]
    if not 0 <= t < T:
        raise ValueError(f"t={t} out of range for T={T}")
    if min_steps < 1 or horizon_steps < min_steps:
        raise ValueError(f"need 1 <= min_steps <= horizon_steps, got "
                         f"min_steps={min_steps}, "
                         f"horizon_steps={horizon_steps}")
    h = min(horizon_steps, T - 1 - t)
    if h < min_steps:
        return NAV_FOLLOW, False
    dyaw = float(wrap_to_pi(poses[t + h, 2] - poses[t, 2]))
    if dyaw > NAV_TURN_RAD:
        return NAV_LEFT, True
    if dyaw < -NAV_TURN_RAD:
        return NAV_RIGHT, True
    return NAV_FOLLOW, True


def route_target(nav_cmd: int) -> int:
    """Map a derived nav command to its route-heading aux class (the SAME
    3-way derivation and thresholds by construction)."""
    return _NAV_TO_ROUTE[nav_cmd]


def waypoint_targets(pose_last: Tensor, future_poses: Tensor,
                     horizons: tuple[int, ...]) -> Tensor:
    """Ego-frame 2 s waypoint targets: [B, len(horizons), 2].

    Waypoint at horizon k = ego_frame(xy[t+k] - xy[t], yaw[t]) with t the
    last window pose — the d1_probe `_ego` convention exactly."""
    max_h = max(horizons)
    if future_poses.shape[1] < max_h:
        raise ValueError(f"future_poses has only {future_poses.shape[1]} "
                         f"steps; waypoint horizons need {max_h}")
    yaw = pose_last[:, 2]
    wps = [ego_frame(future_poses[:, k - 1, :2] - pose_last[:, :2], yaw)
           for k in horizons]
    return torch.stack(wps, dim=1)
