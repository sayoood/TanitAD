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


def path_targets(pose_last: Tensor, future_poses: Tensor,
                 dists: tuple[float, ...]) -> Tensor:
    """refbpatch: ego-frame waypoints sampled at fixed ARC-LENGTHS `dists`
    (metres): [B, len(dists), 2]. This is the GT future path resampled by
    cumulative distance travelled, NOT by time — so the target encodes path
    GEOMETRY independent of speed (the TF++ path/speed decouple). Targets beyond
    the realized path length clamp to the final path point (short/slow windows).

    Target-only (operates on GT poses, no model graph) — the path head regresses
    these directly, forcing the tactical latent to encode speed-invariant shape."""
    B, H, _ = future_poses.shape
    yaw = pose_last[:, 2]
    pts = ego_frame(future_poses[:, :, :2] - pose_last[:, None, :2],
                    yaw[:, None])                          # [B, H, 2]
    origin = torch.zeros(B, 1, 2, device=pts.device, dtype=pts.dtype)
    path = torch.cat([origin, pts], dim=1)                 # [B, H+1, 2]
    seg = (path[:, 1:] - path[:, :-1]).norm(dim=-1)        # [B, H]
    cum = torch.cat([origin[:, :, 0], torch.cumsum(seg, dim=1)], dim=1)  # [B,H+1]
    out = []
    for d in dists:
        dd = torch.full((B,), float(d), device=path.device, dtype=path.dtype)
        j = ((cum <= dd[:, None]).long().sum(dim=1) - 1).clamp(0, H - 1)  # [B]
        j1 = (j + 1).clamp(max=H)
        c0 = torch.gather(cum, 1, j[:, None]).squeeze(1)
        c1 = torch.gather(cum, 1, j1[:, None]).squeeze(1)
        p0 = torch.gather(path, 1, j[:, None, None].expand(-1, 1, 2)).squeeze(1)
        p1 = torch.gather(path, 1, j1[:, None, None].expand(-1, 1, 2)).squeeze(1)
        t = ((dd - c0) / (c1 - c0).clamp_min(1e-6)).clamp(0.0, 1.0)[:, None]
        wp = torch.where((dd > cum[:, -1])[:, None], path[:, -1],
                         p0 + t * (p1 - p0))
        out.append(wp)
    return torch.stack(out, dim=1)                         # [B, len(dists), 2]
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


# ============================================================================
# v2 (2026-07-18): curvature-relative derivation — separate ROAD-FOLLOWING
# (lane-keeping through a curve) from genuine ROUTE DECISIONS / junction turns.
# ----------------------------------------------------------------------------
# WHY: the v1 nav/maneuver derivation thresholds the NET heading change over a
# fixed TIME window (|dyaw| > 45 deg route / > 0.15 rad @2 s tactical). Net
# heading-over-time conflates two physically different things because dyaw =
# kappa * v * t: a GENTLE highway curve at speed sweeps a large net heading
# (R=300 m @30 m/s over 25 s => 143 deg) yet is pure lane-keeping, while the
# same 45 deg fires for a tight junction turn (R=15 m). v1 therefore labels a
# curving ROAD a route "turn" (false-turn) — degenerate on 74 %-straight
# highway — and the derivation is CIRCULAR (the fed nav_cmd IS the target).
#
# FIX: decide on PATH CURVATURE  kappa = dyaw/ds  (heading change per unit
# arc-length, signed, +left) — the physical turn tightness kappa = 1/R, which
# is SPEED-INVARIANT. A road-following curve has small |kappa| (large radius)
# no matter how much net heading it sweeps; a junction turn has large |kappa|
# (small radius). Two features from ego poses alone:
#   peak_kappa    max |smoothed kappa| over the horizon  -> tightness (1/R)
#   concentration share of the net heading change falling in the tightest
#                 CONC_WIN sub-window -> TRANSIENT (junction, ~1.0) vs
#                 SUSTAINED (road sweep spread over the window, ~win/horizon)
# A window is a genuine TURN only if it is BOTH tight (peak_kappa >= turn) AND
# transient (concentration >= min); clearly gentle (peak_kappa <= road) is
# road-following; the band between is AMBIGUOUS (gentle fork / exit ramp /
# collector curve — NOT separable from a road curve without a map) and is
# FLAGGED valid=False so the route-from-vision aux never trains on it.
#
# HONEST CEILING: ego kinematics recover "tight junction-scale turn vs road-
# following curve" well, but the true ROUTE INTENT at a junction (which branch
# the driver chose; straight-through vs the cross street; lane-level choice)
# is a MAP/NAV-GT quantity absent from the trajectory. v2 makes the label mean
# "sharp, transient heading change" HONESTLY and flags what it cannot know,
# instead of pretending net-heading-over-time is a route command.
#
# All v1 functions above are unchanged (byte-identical) and remain the default
# path; v2 is opt-in (validate_refb_labels.py / a future trainer flag).
# ----------------------------------------------------------------------------

DT_DEFAULT = 0.1                    # 10 Hz contract
MIN_ARC_M = 0.10                    # arc-length floor: below this kappa is
                                    # undefined (stopped / yaw noise) -> 0
CURV_SMOOTH_STEPS = 5              # ~0.5 s moving-average on per-step kappa

# Turn geometry (documented; swept in validate_refb_labels.py --sweep). A
# genuine (junction-scale) turn is radius <= R_TURN_M; clearly road-following
# is radius >= R_ROAD_M; the band between is the unrecoverable gray zone.
R_TURN_M = 60.0                     # <= 60 m radius -> junction-scale turn
R_ROAD_M = 150.0                    # >= 150 m radius -> road-following curve
CURV_TURN_PER_M = 1.0 / R_TURN_M    # ~0.01667 /m
CURV_ROAD_PER_M = 1.0 / R_ROAD_M    # ~0.00667 /m

# Transience: at least CONCENTRATION_MIN of the net heading change must fall in
# a single CONC_WIN_STEPS sub-window for a route event to count as a discrete
# turn (a junction turn is brief; a highway sweep spreads evenly -> low share).
CONC_WIN_STEPS = 50                 # 5 s @ 10 Hz
CONCENTRATION_MIN = 0.5

# Maneuver-scale (2 s tactical) turn threshold: same physical curvature, so a
# highway curve at speed stays lane_keep. A 2 s window is short, so gate on the
# window-mean curvature (net dyaw / net arc) — no separate smoothing needed.
CURV_TURN_MAN_PER_M = CURV_TURN_PER_M


def _moving_avg(x: Tensor, k: int) -> Tensor:
    """Centered moving average over dim 0, reflect-padded to keep length."""
    if k <= 1 or x.shape[0] <= k // 2:     # reflect-pad needs len > pad
        return x
    pad = k // 2
    xp = torch.nn.functional.pad(x.view(1, 1, -1), (pad, pad), mode="reflect")
    w = torch.ones(1, 1, k, dtype=x.dtype, device=x.device) / k
    y = torch.nn.functional.conv1d(xp, w).view(-1)
    return y[: x.shape[0]]


def path_curvature(poses: Tensor, min_arc: float = MIN_ARC_M) -> Tensor:
    """Signed per-step path curvature kappa[t] = wrap(yaw[t+1]-yaw[t]) / ds[t],
    length T-1, units 1/m (+left, the repo _ego convention).

    ds is the REALIZED arc length ||xy[t+1]-xy[t]|| (geometry, not v*dt), floored
    at ``min_arc`` so a standstill (near-zero motion, noise-dominated yaw) yields
    kappa 0 rather than a blow-up. kappa = 1/R is speed-invariant: it is the same
    for a tight turn taken slowly or a gentle curve taken fast, which is exactly
    what separates a junction turn from lane-keeping through a road curve."""
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"poses must be [T, 4], got {tuple(poses.shape)}")
    if poses.shape[0] < 2:
        return poses.new_zeros(0)
    dyaw = wrap_to_pi(poses[1:, 2] - poses[:-1, 2])
    ds = (poses[1:, :2] - poses[:-1, :2]).norm(dim=-1).clamp_min(min_arc)
    moving = (poses[1:, :2] - poses[:-1, :2]).norm(dim=-1) >= min_arc
    return torch.where(moving, dyaw / ds, torch.zeros_like(dyaw))


def classify_maneuver_v2(sub_poses: Tensor) -> Tensor:
    """Curvature-gated 2 s maneuver class from a window sub-path.

    ``sub_poses`` [B, H+1, 4] is [pose_t, future_1..future_H] per window (the
    intermediate poses ARE needed — v2 reads the whole arc, not just endpoints).
    Priority order matches v1 (turn > brake > accel > lane_keep); ONLY the turn
    test changes: a turn requires the window-mean curvature |dyaw_net/ds_net| to
    exceed CURV_TURN_MAN_PER_M, so a gentle highway curve (large R) stays
    lane_keep even when its net dyaw exceeds v1's 0.15 rad."""
    if sub_poses.ndim != 3 or sub_poses.shape[2] != 4:
        raise ValueError(f"sub_poses must be [B, H+1, 4], got "
                         f"{tuple(sub_poses.shape)}")
    yaw0, yaw1 = sub_poses[:, 0, 2], sub_poses[:, -1, 2]
    v0, v1 = sub_poses[:, 0, 3], sub_poses[:, -1, 3]
    dyaw = wrap_to_pi(yaw1 - yaw0)
    dv = v1 - v0
    seg = (sub_poses[:, 1:, :2] - sub_poses[:, :-1, :2]).norm(dim=-1)  # [B,H]
    arc = seg.sum(dim=1).clamp_min(MIN_ARC_M)
    kappa = dyaw / arc                                 # window-mean curvature
    cls = torch.full(dyaw.shape, LANE_KEEP, dtype=torch.long,
                     device=dyaw.device)
    cls[dv > DV_ACCEL_MS] = ACCELERATE
    brake = (dv < DV_BRAKE_MS) | ((v1 < STOP_V_MS) & (v0 >= MOVING_V_MS))
    cls[brake] = BRAKE_STOP
    turn = kappa.abs() >= CURV_TURN_MAN_PER_M          # curvature-gated turn
    cls[turn & (dyaw > 0)] = TURN_LEFT
    cls[turn & (dyaw < 0)] = TURN_RIGHT
    return cls


def maneuver_labels_v2(poses: Tensor, horizon: int = LABEL_HORIZON) -> Tensor:
    """Per-timestep v2 maneuver labels: poses [T, 4] -> [T-horizon].

    Label at t reads the sub-path poses[t : t+horizon+1] (curvature over the
    window), the v2 curvature-gated analogue of :func:`maneuver_labels`."""
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"poses must be [T, 4], got {tuple(poses.shape)}")
    if poses.shape[0] <= horizon:
        raise ValueError(f"episode too short for labels: T={poses.shape[0]} "
                         f"<= horizon={horizon}")
    windows = poses.unfold(0, horizon + 1, 1).permute(0, 2, 1)  # [T-h, h+1, 4]
    return classify_maneuver_v2(windows)


def window_maneuver_labels_v2(pose_last: Tensor, future_poses: Tensor,
                              horizon: int = LABEL_HORIZON) -> Tensor:
    """Batch v2 maneuver labels from window fields: pose_last [B, 4],
    future_poses [B, H, 4] (H >= horizon). Reads the arc pose_last +
    future_poses[:, :horizon] (the whole 2 s sub-path)."""
    if future_poses.shape[1] < horizon:
        raise ValueError(f"future_poses has only {future_poses.shape[1]} "
                         f"steps; label horizon needs {horizon}")
    sub = torch.cat([pose_last[:, None, :], future_poses[:, :horizon]], dim=1)
    return classify_maneuver_v2(sub)


def route_from_future(poses: Tensor, t: int,
                      horizon_steps: int = NAV_HORIZON_STEPS,
                      min_steps: int = NAV_MIN_STEPS) -> dict:
    """v2 route derivation at timestep ``t`` — the honest, curvature-relative
    strategic signal. Returns a dict:

        route        int  ROUTE_LEFT / ROUTE_STRAIGHT / ROUTE_RIGHT
        valid        bool False when the window cannot be judged: too little
                          future OR AMBIGUOUS (gray-zone radius — a gentle fork/
                          exit/collector curve not separable from a road curve
                          without a map). valid=False windows are excluded from
                          the route-from-vision aux CE (unlearnable targets).
        ambiguous    bool True when future exists but tightness is in the gray
                          band (distinguishes 'no future' from 'unknowable').
        net_dyaw     float signed net heading change (rad) over the window —
                          the v1 signal, kept as a SOFT/graded regression target.
        signed_curv  float signed peak path curvature (1/m) — the graded route
                          tightness target (soft head), +left.
        peak_kappa   float max |smoothed kappa| over the window (1/m).
        concentration float share of |net_dyaw| in the tightest CONC_WIN sub-
                          window (transience: junction ~1, road sweep ~win/h).

    Decision: TURN iff tight (peak_kappa >= CURV_TURN_PER_M) AND transient
    (concentration >= CONCENTRATION_MIN); clearly gentle (peak_kappa <=
    CURV_ROAD_PER_M) -> STRAIGHT/valid; else AMBIGUOUS -> STRAIGHT/invalid."""
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"poses must be [T, 4], got {tuple(poses.shape)}")
    T = poses.shape[0]
    if not 0 <= t < T:
        raise ValueError(f"t={t} out of range for T={T}")
    if min_steps < 1 or horizon_steps < min_steps:
        raise ValueError(f"need 1 <= min_steps <= horizon_steps, got "
                         f"min_steps={min_steps}, horizon_steps={horizon_steps}")
    h = min(horizon_steps, T - 1 - t)
    base = {"route": ROUTE_STRAIGHT, "valid": False, "ambiguous": False,
            "net_dyaw": 0.0, "signed_curv": 0.0, "peak_kappa": 0.0,
            "concentration": 0.0}
    if h < min_steps:
        return base                                    # no route-scale future
    seg = poses[t:t + h + 1]                            # [h+1, 4]
    net_dyaw = float(wrap_to_pi(seg[-1, 2] - seg[0, 2]))
    kappa = path_curvature(seg)                        # [h]
    ks = _moving_avg(kappa, CURV_SMOOTH_STEPS)
    if ks.numel() == 0:
        return base
    peak_i = int(ks.abs().argmax())
    peak_kappa = float(ks.abs()[peak_i])
    signed_curv = float(ks[peak_i])
    # concentration: tightest CONC_WIN heading share of the net heading change.
    win = min(CONC_WIN_STEPS, kappa.shape[0])
    if kappa.shape[0] >= 1 and abs(net_dyaw) > 1e-3:
        # heading change over each length-`win` sub-window = sum of per-step dyaw
        step_dyaw = wrap_to_pi(seg[1:, 2] - seg[:-1, 2])
        sub = step_dyaw.unfold(0, win, 1).sum(dim=1) if step_dyaw.shape[0] >= win \
            else step_dyaw.sum().view(1)
        concentration = float(sub.abs().amax() / (abs(net_dyaw) + 1e-6))
    else:
        concentration = 0.0
    out = dict(base, net_dyaw=net_dyaw, signed_curv=signed_curv,
               peak_kappa=peak_kappa, concentration=concentration)
    is_turn = (peak_kappa >= CURV_TURN_PER_M) and \
              (concentration >= CONCENTRATION_MIN)
    is_road = peak_kappa <= CURV_ROAD_PER_M
    if is_turn:
        # Direction from the sign of the PEAK signed curvature (+left), which is
        # robust to yaw wrapping — a sustained >180 deg turn (roundabout/loop)
        # wraps net_dyaw's sign, but the tightest-point curvature does not.
        out["route"] = ROUTE_LEFT if signed_curv > 0 else ROUTE_RIGHT
        out["valid"] = True
    elif is_road:
        out["route"] = ROUTE_STRAIGHT
        out["valid"] = True
    else:                                              # gray-zone tightness
        out["route"] = ROUTE_STRAIGHT
        out["valid"] = False
        out["ambiguous"] = True
    return out


_ROUTE_TO_NAV = {ROUTE_LEFT: NAV_LEFT, ROUTE_STRAIGHT: NAV_FOLLOW,
                 ROUTE_RIGHT: NAV_RIGHT}


def nav_command_v2(poses: Tensor, t: int, horizon_steps: int = NAV_HORIZON_STEPS,
                   min_steps: int = NAV_MIN_STEPS) -> tuple[int, bool]:
    """v2 drop-in for :func:`nav_command`: (nav_cmd, valid) from the curvature-
    relative :func:`route_from_future`. NAV_STRAIGHT is never emitted (reserved,
    same interface contract as v1)."""
    r = route_from_future(poses, t, horizon_steps, min_steps)
    return _ROUTE_TO_NAV[r["route"]], bool(r["valid"])


def route_target_v2(poses: Tensor, t: int, horizon_steps: int = NAV_HORIZON_STEPS,
                    min_steps: int = NAV_MIN_STEPS) -> int:
    """v2 route-heading aux class (route_left/straight/right) at ``t`` — the
    curvature-relative target, NOT a function of any fed nav command (breaks
    the v1 circularity where the input command and the target were the same
    derivation)."""
    return route_from_future(poses, t, horizon_steps, min_steps)["route"]


# ============================================================================
# v2.1 (2026-07-20): ADAPTIVE-HORIZON route labels that NEVER default to
# `straight`. Fixes the three defects an 800-window audit of the v2 labeler on
# the PhysicalAI val split measured directly.
# ----------------------------------------------------------------------------
# WHAT THE AUDIT FOUND (stack/scripts/route_label_audit.py, 80 episodes):
#   D1 COVERAGE COLLAPSE. v2 needs NAV_MIN_STEPS=150 (15 s) of future and looks
#      ahead NAV_HORIZON_STEPS=250 (25 s). PhysicalAI clips are ~199 frames
#      (~20 s), so only the first ~5 s of every clip is judgeable at all —
#      70 % of windows fell through the `h < min_steps` guard.
#   D2 SILENT STRAIGHT-FALLBACK. That guard returned ROUTE_STRAIGHT (the `base`
#      dict, v2 line ~410). "I cannot judge this" and "the road goes straight"
#      were the SAME emitted class. Any consumer that ignored `valid` — and the
#      label files/exports do, they carry the class — read 70 % unlabeled as
#      70 % straight. That is what poisoned the route prior.
#   D3 net_dyaw COMPUTED BUT UNUSED. v2 decided on peak_kappa+concentration
#      only. A 479 m-radius, 48-degree sweep (val ep_00069) is `is_road` ->
#      ROUTE_STRAIGHT with valid=True, i.e. it TRAINS as straight while the
#      vehicle actually changes heading by half a right angle.
#
# THE v2.1 RULES
#   R1 ADAPTIVE HORIZON — judge with whatever future exists. The gate is not
#      "enough TIME" but "enough ROAD": ARC LENGTH actually travelled
#      (MIN_ARC_ROUTE_M). Curvature kappa = dyaw/ds is ALREADY per-metre, so
#      the TIGHTNESS threshold needs no rescaling by horizon — that is the
#      whole point of a curvature-relative label. What does need rescaling is
#      TRANSIENCE: v2's `concentration` is the share of the net heading change
#      inside a fixed 5 s sub-window, and as the horizon shrinks that share
#      tends to 1 for every window, junction or not. v2.1 therefore (a)
#      measures concentration over a junction-scale stretch of ROAD
#      (CONC_ARC_M metres) instead of a fixed number of steps, and (b) applies
#      the transience gate ONLY when there is enough arc for a sustained
#      alternative to exist (TRANSIENCE_MIN_ARC_M); below that, transience is
#      not measurable and tightness alone decides (flagged in the output).
#   R2 NEVER DEFAULT TO STRAIGHT — unjudgeable windows return ROUTE_UNKNOWN,
#      a SENTINEL that is deliberately NOT one of the three CE classes.
#      n_route stays 3: ROUTE_UNKNOWN always comes with valid=False, and a
#      consumer that forgets the mask gets a loud index error instead of a
#      silent straight. `reason` says which of the four unjudgeable/decided
#      paths fired.
#   R3 net_dyaw IS IN THE DECISION — TURN iff (tight AND transient) OR
#      |net_dyaw| >= NET_DYAW_TURN_RAD. The second clause is the D3 fix: a
#      large-radius sweep that nets 45 degrees of heading is a route event for
#      a head that must predict WHERE THE VEHICLE IS GOING, however gentle the
#      radius. It is a SEPARATE, NAMED rule (`reason="net_heading"`) and can be
#      switched off (`use_net_dyaw=False`) to recover v2's strict
#      junction-only semantics, because the two readings genuinely disagree and
#      the audit reports both.
#   R4 GRADED TARGETS — `mean_curv` (net_dyaw / arc, signed 1/m) and
#      `graded_route` (tanh of mean_curv in junction units) are threshold-free
#      and horizon-invariant by construction: a soft regression target that
#      works on a 2 s window and a 25 s window alike, alongside the 3-class CE.
#
# ADDITIVE: every v1 and v2 function above is unchanged and remains the default
# path, so shipped runs stay reproducible. v2.1 is opt-in.
# ----------------------------------------------------------------------------

# Sentinel route class. NOT a CE class — ROUTE_CLASSES stays 3 wide
# (tanitad.refs.refb.ROUTE_CLASSES / RefBConfig.n_route are untouched).
ROUTE_UNKNOWN = 3
ROUTE_V21_NAMES = {ROUTE_LEFT: "left", ROUTE_STRAIGHT: "straight",
                   ROUTE_RIGHT: "right", ROUTE_UNKNOWN: "unknown"}

MIN_ARC_ROUTE_M = 20.0        # < 20 m of road travelled -> route intent is not
                              # recoverable from the trajectory (stopped, or the
                              # window sits at the very end of the clip)
CONC_ARC_M = 60.0             # junction-scale stretch of ROAD over which a
                              # discrete turn's heading change is concentrated
TRANSIENCE_MIN_ARC_M = 150.0  # below this the window is too short for a
                              # SUSTAINED alternative to exist, so concentration
                              # carries no information -> gate not applied
NET_DYAW_TURN_RAD = math.pi / 4   # 45 deg net heading change -> route turn even
                                  # at road-following radius (the D3 fix)
CURV_SMOOTH_ARC_M = 5.0       # curvature smoothing measured in METRES of road,
                              # so a slow tight turn is not smoothed away and a
                              # fast sweep is not under-smoothed
CURV_SMOOTH_MIN_K, CURV_SMOOTH_MAX_K = 3, 15


def _arc_smooth_k(ds: Tensor, arc_m: float = CURV_SMOOTH_ARC_M) -> int:
    """Moving-average width (in steps) spanning ~`arc_m` metres of road."""
    if ds.numel() == 0:
        return CURV_SMOOTH_MIN_K
    mean_ds = float(ds.mean().clamp_min(1e-3))
    k = int(round(arc_m / mean_ds))
    return max(CURV_SMOOTH_MIN_K, min(CURV_SMOOTH_MAX_K, k))


def _arc_concentration(step_dyaw: Tensor, ds: Tensor, net_dyaw: float,
                       conc_arc_m: float = CONC_ARC_M) -> float:
    """Share of |net_dyaw| falling in the tightest `conc_arc_m` METRES of road.

    v2 used a fixed 50-step (5 s) sub-window, which is a different physical
    stretch of road at 5 m/s than at 30 m/s and degenerates as the horizon
    shrinks. Arc-anchored, "transient" means the same thing at every speed and
    every horizon: the heading change happened inside one junction-length piece
    of road rather than being spread along the whole drive."""
    n = step_dyaw.shape[0]
    if n == 0 or abs(net_dyaw) <= 1e-3:
        return 0.0
    z = step_dyaw.new_zeros(1)
    cum_s = torch.cat([z, torch.cumsum(ds, 0)])              # [n+1]
    cum_d = torch.cat([z, torch.cumsum(step_dyaw, 0)])       # [n+1]
    # first index j > i whose arc from i reaches conc_arc_m (else the window end)
    j = torch.searchsorted(cum_s.contiguous(),
                           (cum_s[:-1] + conc_arc_m).contiguous()).clamp(max=n)
    sub = cum_d[j] - cum_d[:-1]                              # [n] heading per sub
    return float(sub.abs().amax() / (abs(net_dyaw) + 1e-6))


def route_from_future_v21(poses: Tensor, t: int,
                          horizon_steps: int = NAV_HORIZON_STEPS,
                          min_arc_m: float = MIN_ARC_ROUTE_M,
                          net_dyaw_turn: float = NET_DYAW_TURN_RAD,
                          use_net_dyaw: bool = True) -> dict:
    """v2.1 route derivation at ``t`` — adaptive-horizon, never-straight-by-
    default. Drop-in shape-compatible with :func:`route_from_future` plus new
    keys. Returns:

        route        int  ROUTE_LEFT / ROUTE_STRAIGHT / ROUTE_RIGHT /
                          **ROUTE_UNKNOWN** (sentinel, never a CE class)
        valid        bool True iff `route` is a real judgement. ROUTE_UNKNOWN
                          always carries valid=False (and vice versa).
        ambiguous    bool True only for the gray-zone tightness band (a gentle
                          fork/exit not separable from a road curve without a
                          map) — distinguishes "unknowable" from "no data".
        reason       str  which rule fired: ``no_future`` | ``no_arc`` |
                          ``tight_transient`` | ``net_heading`` |
                          ``road_following`` | ``gray_zone``
        net_dyaw     float signed CUMULATIVE heading change (rad) over the
                          AVAILABLE future — graded target, and now a DECISION
                          input. NOTE this differs from v2's field of the same
                          name: v2 took the WRAPPED endpoint difference
                          wrap(yaw[t+h]-yaw[t]), which silently folds a 270 deg
                          roundabout into -90 deg and flips its sign. v2.1 sums
                          the per-step wrapped deltas, so heading change beyond
                          +-180 deg accumulates correctly and the sign is
                          wrap-robust. v2's quantity is kept as
                          ``net_dyaw_wrapped`` for cross-checking.
        net_dyaw_wrapped float the v2 endpoint-difference value
        signed_curv  float signed smoothed peak curvature (1/m, +left)
        peak_kappa   float max |smoothed kappa| (1/m) — tightness, 1/R
        concentration float arc-anchored transience share (see
                          :func:`_arc_concentration`)
        mean_curv    float net_dyaw / arc (signed 1/m) — the threshold-free,
                          horizon-invariant graded route target
        graded_route float tanh(mean_curv / CURV_TURN_PER_M) in (-1, 1) — the
                          same signal squashed into junction units for a soft
                          regression head
        arc_m        float road length actually travelled over the window
        h_steps      int   future steps actually used (adaptive)
        transience_measurable bool whether the concentration gate was applied

    Decision (R3): TURN iff (peak_kappa >= CURV_TURN_PER_M AND transient) OR
    (use_net_dyaw AND |net_dyaw| >= net_dyaw_turn); else clearly gentle
    (peak_kappa <= CURV_ROAD_PER_M) -> STRAIGHT/valid; else UNKNOWN/invalid."""
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"poses must be [T, 4], got {tuple(poses.shape)}")
    T = poses.shape[0]
    if not 0 <= t < T:
        raise ValueError(f"t={t} out of range for T={T}")
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")
    base = {"route": ROUTE_UNKNOWN, "valid": False, "ambiguous": False,
            "reason": "no_future", "net_dyaw": 0.0, "net_dyaw_wrapped": 0.0,
            "signed_curv": 0.0, "peak_kappa": 0.0, "concentration": 0.0,
            "mean_curv": 0.0, "graded_route": 0.0, "arc_m": 0.0, "h_steps": 0,
            "transience_measurable": False}
    h = min(int(horizon_steps), T - 1 - t)
    if h < 1:
        return base                                   # literally no future left
    seg = poses[t:t + h + 1]                          # [h+1, 4]
    d = seg[1:, :2] - seg[:-1, :2]
    ds = d.norm(dim=-1)                               # [h] realized arc per step
    arc = float(ds.sum())
    step_dyaw = wrap_to_pi(seg[1:, 2] - seg[:-1, 2])
    net_dyaw = float(step_dyaw.sum())                 # CUMULATIVE, unwrapped
    out = dict(base, net_dyaw=net_dyaw, arc_m=arc, h_steps=int(h),
               net_dyaw_wrapped=float(wrap_to_pi(seg[-1, 2] - seg[0, 2])))
    if arc < min_arc_m:                               # stopped / clip tail
        out["reason"] = "no_arc"
        return out                                    # UNKNOWN, never straight

    kappa = torch.where(ds >= MIN_ARC_M, step_dyaw / ds.clamp_min(MIN_ARC_M),
                        torch.zeros_like(ds))
    ks = _moving_avg(kappa, _arc_smooth_k(ds))
    peak_i = int(ks.abs().argmax()) if ks.numel() else 0
    peak_kappa = float(ks.abs()[peak_i]) if ks.numel() else 0.0
    signed_curv = float(ks[peak_i]) if ks.numel() else 0.0
    concentration = _arc_concentration(step_dyaw, ds, net_dyaw)
    mean_curv = net_dyaw / max(arc, 1e-6)
    out.update(signed_curv=signed_curv, peak_kappa=peak_kappa,
               concentration=concentration, mean_curv=mean_curv,
               graded_route=float(math.tanh(mean_curv / CURV_TURN_PER_M)),
               transience_measurable=arc >= TRANSIENCE_MIN_ARC_M)

    tight = peak_kappa >= CURV_TURN_PER_M
    # R1: transience only discriminates when a SUSTAINED alternative could fit.
    transient = (concentration >= CONCENTRATION_MIN
                 if out["transience_measurable"] else True)
    big_net = abs(net_dyaw) >= net_dyaw_turn
    if tight and transient:
        # Direction from the PEAK signed curvature: robust to yaw wrap on a
        # >180 deg roundabout, where net_dyaw's sign flips (v2 regression).
        out.update(route=ROUTE_LEFT if signed_curv > 0 else ROUTE_RIGHT,
                   valid=True, reason="tight_transient")
    elif use_net_dyaw and big_net:
        # D3 fix: a gentle-radius sweep that nets >= 45 deg IS a route event.
        # net_dyaw is the CUMULATIVE (unwrapped) heading change, so its sign is
        # correct past +-180 deg too — no wrap ambiguity to guard against.
        out.update(route=ROUTE_LEFT if net_dyaw > 0 else ROUTE_RIGHT,
                   valid=True, reason="net_heading")
    elif peak_kappa <= CURV_ROAD_PER_M:
        out.update(route=ROUTE_STRAIGHT, valid=True, reason="road_following")
    else:                                             # gray-zone tightness
        out.update(route=ROUTE_UNKNOWN, valid=False, ambiguous=True,
                   reason="gray_zone")
    return out


def nav_command_v21(poses: Tensor, t: int,
                    horizon_steps: int = NAV_HORIZON_STEPS,
                    **kw) -> tuple[int, bool]:
    """v2.1 drop-in for :func:`nav_command_v2`: (nav_cmd, valid).

    The nav COMMAND is a model INPUT and must stay inside NAV_COMMANDS, so an
    UNKNOWN route emits NAV_FOLLOW with valid=False — exactly as v1/v2 did for
    short windows. The difference that matters is on the TARGET side
    (:func:`route_target_v21`), which refuses to say `straight` when it does
    not know."""
    r = route_from_future_v21(poses, t, horizon_steps, **kw)
    nav = _ROUTE_TO_NAV.get(r["route"], NAV_FOLLOW)
    return nav, bool(r["valid"])


def route_target_v21(poses: Tensor, t: int,
                     horizon_steps: int = NAV_HORIZON_STEPS,
                     **kw) -> tuple[int, bool]:
    """v2.1 route aux target at ``t`` as ``(target, valid)``.

    Returns the PAIR on purpose: v2's ``route_target_v2`` returned a bare class
    and the caller had to remember to fetch `valid` separately — the exact
    ergonomic hole that let the silent straight-fallback reach the trainer.
    ``target`` is ROUTE_UNKNOWN (=3, out of CE range) whenever valid is False,
    so an unmasked cross-entropy raises instead of training a wrong class."""
    r = route_from_future_v21(poses, t, horizon_steps, **kw)
    return int(r["route"]), bool(r["valid"])


def route_graded_target(poses: Tensor, t: int,
                        horizon_steps: int = NAV_HORIZON_STEPS,
                        **kw) -> tuple[float, float]:
    """Threshold-free soft route target at ``t``: ``(mean_curv, net_dyaw)``.

    mean_curv = net_dyaw / arc is signed curvature per metre — horizon- and
    speed-invariant, defined for ANY amount of future (the 3-class CE is not),
    so it supervises the windows the discrete label must mask. Regress it
    alongside the CE; ``graded_route`` in the full dict is the same quantity
    squashed to (-1, 1)."""
    r = route_from_future_v21(poses, t, horizon_steps, **kw)
    return float(r["mean_curv"]), float(r["net_dyaw"])
