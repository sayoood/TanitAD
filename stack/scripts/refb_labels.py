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
                          # SAYED'S RULING 2026-07-20: a wide sweep is ROAD
                          # FOLLOWING, not a route event — v2's original
                          # semantics. The independent VLM pass agreed with v2
                          # on the contested ep_00069 479 m case. The rule stays
                          # implemented and switchable (pass True to recover it)
                          # but it is NOT the default: `strategic route` means
                          # driver intent at junction scale, not "where the road
                          # bends over the next 20 s".
                          use_net_dyaw: bool = False) -> dict:
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


# ============================================================================
# v3 (2026-07-21): VOCABULARY-COMPLETE route tokens + DISTANCE-TO-MANEUVER,
# and a FACTORIZED tactical labeler (LATMANEUVER x LONMODE, independent).
# ----------------------------------------------------------------------------
# WHY (two defects Sayed found by watching plan-fan clips, both the same disease
# — OUR VOCABULARY IS RICHER THAN ANYTHING THAT MINTS IT):
#
#  DEFECT 1 — ROUTE. tanitad/lake/vocab.py freezes a 9-token ROUTE slot
#    (`follow straight turn_left turn_right exit_left exit_right merge u_turn
#     roundabout`) but this file emits only 3 classes + an UNKNOWN sentinel, and
#    goal_labels.route_at collapses them to `turn_left|turn_right|follow`. FIVE
#    tokens — exit_left, exit_right, merge, u_turn, roundabout — plus `straight`
#    have never been minted by anything. On ep09 f167 the model is conditioned
#    `route left / turn left` in roundabout-exit geometry: the shape descriptor
#    is not wrong, it is not an INSTRUCTION. Note this labeler ALREADY detects
#    the roundabout signature (v2 line ~441 and v2.1 line ~688 both guard the
#    yaw-wrapping a ">180 deg turn (roundabout/loop)" produces) — and throws the
#    information away to keep left/right robust.
#
#  DEFECT 2 — TACTICAL. `N_MANEUVERS = 5` packs THREE LATERAL classes
#    (lane_keep/turn_left/turn_right) and TWO LONGITUDINAL ones
#    (accelerate/brake_stop) into ONE softmax, and `classify_maneuver*` resolves
#    the collision by PRIORITY (turn > brake > accel). vocab.py already models
#    these as independent orthogonal slots: LATMANEUVER (9) x LONMODE (9) x
#    TACPOINT (5). On ep19 the true state is `lane_keep` AND `stop_at_point` AND
#    a stop point ~ahead, simultaneously; the 5-way head can say only one, so it
#    says the lateral one. Measured consequence (both REF-C arms, 881 canonical
#    val windows, results/planfan_clips_tactical_head_val.json): `accelerate`
#    predicted 0/881, `brake_stop` 7/881 base + 4/881 XL, i.e. a 99.5 %
#    lateral-or-neutral prior — which `graft_maneuver` then adds straight into
#    the anchor logits that make the selection.
#
# THE DISTANCE IS THE LOAD-BEARING HALF. `roundabout` is a shape; `roundabout,
# exit in 40 m` is an instruction. Same for `stop_at_point in 12 m`. nuPlan,
# CARLA and every production nav stack hand their planner a distance; we hand
# ours a class name. v3 mints the distance in METRES along the realized future
# arc (see DIST_BAND_TOKENS for the justification of metres over seconds).
#
# ADDITIVE AND CONSERVATIVE BY CONSTRUCTION:
#  * every v1 / v2 / v2.1 function above is untouched and stays the default;
#  * ``route_from_future_v3`` CALLS ``route_from_future_v21`` for the base
#    decision and only ever UPGRADES the token. The 3-class CE field ``route``
#    and the ``valid`` mask are therefore byte-identical to v2.1 for every
#    window, so switching a run to v3 cannot regress coverage (v2->v2.1 raised
#    it 26.0 % -> 81.9 %) and the migration is exactly measurable as
#    "token != the v2.1 token";
#  * a richer token is emitted ONLY when its full signature is confirmed. An
#    unconfirmed roundabout keeps v2.1's defensible turn_left/turn_right and is
#    reported via ``reason``/``roundabout_candidate`` — a low-confidence guess
#    never replaces an honest answer (R3).
# ----------------------------------------------------------------------------

# ---- the frozen ROUTE vocabulary, mirrored (tanitad.lake.vocab.STRATEGIC_TOKENS
# ---- ["ROUTE"]). Pinned equal by tests/test_refb_labels_v3.py so this file and
# ---- the vocabulary can never drift apart.
ROUTE_V3_TOKENS = ("follow", "straight", "turn_left", "turn_right",
                   "exit_left", "exit_right", "merge", "u_turn", "roundabout")
TOKEN_UNKNOWN = "unknown"

# v2.1 3-class -> the v3 token it means when no richer signature is confirmed.
# NOTE `straight` is NOT the image of ROUTE_STRAIGHT: in the v3 vocabulary
# `follow` is "keep following this road" (kinematic) while `straight` means "at
# the junction ahead, go straight through", which asserts a junction exists —
# a MAP fact, not a trajectory fact. goal_labels.route_at already made this
# choice; v3 makes it explicit and never mints `straight`.
_V21_TO_V3_TOKEN = {ROUTE_LEFT: "turn_left", ROUTE_RIGHT: "turn_right",
                    ROUTE_STRAIGHT: "follow", ROUTE_UNKNOWN: TOKEN_UNKNOWN}

# ---- DISTANCE-TO-MANEUVER (metres, not seconds) ------------------------------
# METRES, decided and justified:
#  1. speed-invariant. The same junction is "5 s" at 10 m/s and "2.5 s" at
#     20 m/s — two different tokens for one physical instruction. Curvature is
#     already per-metre in this file for exactly this reason.
#  2. it is what the PLANNER needs: a stopping/deceleration profile is set by
#     distance (v^2 = 2 a d), not by time.
#  3. it is what every external source speaks: nuPlan / CARLA / OpenDRIVE / any
#     nav stack announce "in 200 m". A future map or VLM fills the same slot
#     without a unit conversion that would need the speed at label time.
# Edges (PROPOSED, not frozen): 10 m ~ "you are in it" (1-2 vehicle lengths);
# 25/50 m are the urban decision distances (a comfortable 2 m/s^2 stop from
# 14 m/s needs ~49 m); 100/200 m are the highway announcement distances.
DIST_BAND_EDGES_M = (10.0, 25.0, 50.0, 100.0, 200.0)
DIST_BAND_TOKENS = ("d_now", "d_10_25", "d_25_50", "d_50_100", "d_100_200",
                    "d_200_plus", "d_none", "d_unknown")
# `d_none` and `d_unknown` are DIFFERENT on purpose — this is the D2 lesson of
# v2.1 applied to the distance axis. `d_none` = "we looked over enough road and
# there is no maneuver in range"; `d_unknown` = "the window did not reach far
# enough to say". Collapsing them is exactly the silent-straight-fallback bug.
DIST_LOOKED_ENOUGH_M = 100.0   # arc that must be observed before claiming d_none


def dist_band(dist_m: float | None, observed_arc_m: float = 0.0,
              looked_enough_m: float = DIST_LOOKED_ENOUGH_M) -> str:
    """Metres-to-maneuver -> band token (DIST_BAND_TOKENS).

    ``dist_m is None`` means no maneuver was found in the observed future: that
    is ``d_none`` only when ``observed_arc_m >= looked_enough_m``, else
    ``d_unknown``. Never returns a band for an unobserved horizon."""
    if dist_m is None:
        return "d_none" if observed_arc_m >= looked_enough_m else "d_unknown"
    d = float(dist_m)
    if not math.isfinite(d) or d < 0.0:
        return "d_unknown"
    for edge, tok in zip(DIST_BAND_EDGES_M, DIST_BAND_TOKENS):
        if d < edge:
            return tok
    return DIST_BAND_TOKENS[len(DIST_BAND_EDGES_M)]      # d_200_plus


# ---- roundabout / u-turn / exit / merge geometry (documented, swept-able) -----
# Roundabout: the circulating radius of a real roundabout is small (design
# inscribed radius ~8-25 m for the vehicle path), the arc is SUSTAINED at
# roughly CONSTANT radius, the swept heading is well beyond a junction turn, and
# the vehicle LEAVES by deflecting the other way — that exit reversal is what
# separates "roundabout" from "a very tight long turn".
R_ROUNDABOUT_MAX_M = 30.0
CURV_ROUNDABOUT_PER_M = 1.0 / R_ROUNDABOUT_MAX_M          # ~0.0333 /m
ROUNDABOUT_DYAW_RAD = math.radians(135.0)   # swept heading of the circulating arc
ROUNDABOUT_EXIT_REV_RAD = math.radians(12.0)  # opposite-sign deflection at exit
ROUNDABOUT_KAPPA_CV_MAX = 0.85              # sd/mean of |kappa| inside the arc:
                                            # a constant-radius ring, not a kink
# U-turn: ~180 deg at a very tight radius, ending ANTIPARALLEL, and WITHOUT the
# roundabout's exit reversal.
R_UTURN_MAX_M = 20.0
CURV_UTURN_PER_M = 1.0 / R_UTURN_MAX_M
UTURN_DYAW_LO_RAD, UTURN_DYAW_HI_RAD = math.radians(150.0), math.radians(195.0)
UTURN_ANTIPARALLEL_RAD = math.radians(35.0)  # |final heading - start| within
                                             # this of 180 deg
# Exit / fork / merge: a lateral offset that ACCUMULATES without the sustained
# heading change of a junction turn — the track deflects and then RUNS PARALLEL
# again (heading returns), while the offset does not return. A road curve keeps
# its heading change; a lane change does return but only moves ~one lane.
EXIT_LAT_MIN_M = 8.0            # > 2 lane widths: beyond any lane change
EXIT_NET_DYAW_MAX_RAD = math.radians(45.0)   # net heading stays modest
EXIT_RETURN_FRAC = 0.6          # |final dyaw| <= this * peak |dyaw| = "returns"
EXIT_PEAK_DYAW_MIN_RAD = math.radians(8.0)   # there WAS a deflection at all
EXIT_DV_MS = -1.5               # exit ramps decelerate ...
MERGE_DV_MS = 1.5               # ... on-ramp merges accelerate
# ... but a LAUNCH FROM STANDSTILL also has dv >> +1.5, which is why the first
# pass of this rule minted 13 `merge` windows on ONE stationary episode
# (MEASURED, ep_00073 of the 100-ep val build: v0 = 0.0, dyaw = 0.0). A ramp is
# a road-speed feature: require the whole window to stay above this.
EXIT_MIN_SPEED_MS = 5.0


def _future_track(poses: Tensor, t: int, horizon_steps: int) -> dict:
    """Per-step geometry of the available future from ``t`` (v3 shared core).

    Returns ds [h], step_dyaw [h], cum_dyaw [h] (CUMULATIVE, unwrapped — so a
    >180 deg roundabout accumulates instead of folding), cum_s [h] (arc from t),
    kappa [h] and its arc-smoothed version ks [h], plus the ego-frame future
    track ``ego`` [h+1, 2] (+x forward, +y left, origin at pose t)."""
    T = poses.shape[0]
    h = max(0, min(int(horizon_steps), T - 1 - t))
    if h < 1:
        z = poses.new_zeros(0)
        return {"h": 0, "ds": z, "step_dyaw": z, "cum_dyaw": z, "cum_s": z,
                "kappa": z, "ks": z, "arc": 0.0,
                "ego": poses.new_zeros(1, 2), "seg": poses[t:t + 1]}
    seg = poses[t:t + h + 1]                              # [h+1, 4]
    d = seg[1:, :2] - seg[:-1, :2]
    ds = d.norm(dim=-1)                                   # [h]
    step_dyaw = wrap_to_pi(seg[1:, 2] - seg[:-1, 2])      # [h]
    kappa = torch.where(ds >= MIN_ARC_M, step_dyaw / ds.clamp_min(MIN_ARC_M),
                        torch.zeros_like(ds))
    ks = _moving_avg(kappa, _arc_smooth_k(ds))
    ego = ego_frame(seg[:, :2] - seg[0, :2], seg[0, 2].expand(h + 1))
    return {"h": h, "ds": ds, "step_dyaw": step_dyaw,
            "cum_dyaw": torch.cumsum(step_dyaw, 0), "cum_s": torch.cumsum(ds, 0),
            "kappa": kappa, "ks": ks, "arc": float(ds.sum()),
            "ego": ego, "seg": seg}


# A curvature segment only counts as a MANEUVER if it sweeps real heading.
# kappa = dyaw/ds blows up at low speed (ds floors at MIN_ARC_M = 0.1 m), so
# without this gate parking-lot yaw jitter reads as a junction every window:
# MEASURED on the 100-episode val build, `d_now` was 864/2201 (39 %) before the
# gate and is a fraction of that after. 15 deg is well below any real junction
# turn (~90 deg) or roundabout ring, so nothing real is lost.
SEG_MIN_DYAW_RAD = math.radians(15.0)
# "no exit reversal was observed" is only a claim if there WAS road left to
# observe it in. Without this, every roundabout whose clip ends mid-ring reads
# as a u_turn (MEASURED: 36/2201 u_turns before the gate — implausible for real
# driving, and the same ego track as an unfinished roundabout).
UTURN_MIN_TAIL_M = 30.0


def _curv_segments(ks: Tensor, thresh: float, min_len: int = 2) -> list[tuple]:
    """Contiguous same-sign runs of |smoothed curvature| >= ``thresh``.

    Returns ``[(i0, i1, sign), ...]`` half-open on i1. These are the discrete
    MANEUVER EVENTS in the future track — v2/v2.1 only ever looked at the single
    global peak, which is why they could see "there is a tight bit somewhere"
    but never "the tight bit starts 40 m from here and sweeps 210 deg"."""
    if ks.numel() == 0:
        return []
    sign = torch.sign(ks)
    hot = ks.abs() >= thresh
    out, i = [], 0
    n = int(ks.shape[0])
    while i < n:
        if not bool(hot[i]):
            i += 1
            continue
        s = float(sign[i])
        j = i + 1
        while j < n and bool(hot[j]) and float(sign[j]) == s:
            j += 1
        if j - i >= min_len:
            out.append((i, j, s))
        i = j
    return out


def _seg_stats(tr: dict, i0: int, i1: int) -> dict:
    """Heading swept, arc length, distance-to-start and radius consistency of a
    curvature segment (indices into the per-step arrays of ``_future_track``)."""
    dyaw = float(tr["step_dyaw"][i0:i1].sum())
    arc = float(tr["ds"][i0:i1].sum())
    start_m = float(tr["cum_s"][i0 - 1]) if i0 > 0 else 0.0
    k = tr["ks"][i0:i1].abs()
    mean_k = float(k.mean()) if k.numel() else 0.0
    cv = float(k.std(unbiased=False) / max(mean_k, 1e-9)) if k.numel() > 1 else 0.0
    return {"dyaw": dyaw, "arc_m": arc, "start_m": start_m, "end_m": start_m + arc,
            "mean_kappa": mean_k, "kappa_cv": cv}


def _signed_reversal(tr: dict, i1: int, seg_sign: float) -> float:
    """Heading swept AGAINST ``seg_sign`` after a segment ends (rad, >= 0).

    Leaving a roundabout means deflecting back the other way; this measures how
    much of that counter-deflection is actually observed inside the window."""
    tail = tr["step_dyaw"][i1:]
    if tail.numel() == 0:
        return 0.0
    c = torch.cumsum(tail, 0)
    return float((-seg_sign * c).amax().clamp_min(0.0))


def route_from_future_v3(poses: Tensor, t: int,
                         horizon_steps: int = NAV_HORIZON_STEPS,
                         **v21_kw) -> dict:
    """v3 route derivation at ``t`` — the frozen 9-token ROUTE vocabulary plus
    DISTANCE-TO-MANEUVER. A strict, additive SUPERSET of
    :func:`route_from_future_v21`.

    Returns every v2.1 key unchanged (``route`` int / ``valid`` / ``ambiguous``
    / ``reason`` / the graded targets ...) plus:

        token        str   one of ROUTE_V3_TOKENS or ``unknown`` — the frozen
                           vocabulary token. `straight` is NEVER emitted (it
                           asserts a junction exists = a MAP fact).
        token_v21    str   what v2.1 alone would have said — so the migration
                           is countable as ``token != token_v21``.
        token_valid  bool  the TOKEN is a real judgement. This is NOT
                           ``valid``: ``valid`` gates the 3-class CE and stays
                           byte-identical to v2.1, while a confirmed richer
                           signature (e.g. `merge`) is a judgement even on a
                           window v2.1 had to mask as gray-zone.
        upgraded     bool  token != token_v21
        v3_rule      str   which v3 rule fired: ``v21_base`` | ``roundabout``
                           | ``u_turn`` | ``exit`` | ``merge``. v2.1's own
                           ``reason`` is left UNTOUCHED so the base decision
                           stays auditable next to the upgrade.
        dist_m       float|None arc metres from the ego pose to the START of the
                           next junction-scale maneuver; None = none in range.
        dist_band    str   DIST_BAND_TOKENS (d_none vs d_unknown kept distinct).
        maneuver_arc_m float arc length of that maneuver, 0.0 if none.
        maneuver_dyaw  float signed heading it sweeps (rad, cumulative).
        roundabout_candidate bool a >=135 deg tight sustained arc WAS seen but
                           its exit reversal was not observed inside the window
                           — reported, never silently promoted to `roundabout`.
        uturn_roundabout_confounded bool the segment that produced `u_turn`
                           ALSO satisfies the roundabout ring geometry. Without
                           a map these two are the SAME ego track (MEASURED: on
                           the 100-episode val build one episode produced BOTH a
                           `u_turn` and, at a later window, a roundabout
                           candidate). Treat every `u_turn` carrying this flag
                           as UNVERIFIED pending a map/VLM read.
        n_segments   int   junction-scale curvature segments found ahead.

    Decision order (first confirmed signature wins): u_turn -> roundabout ->
    exit_left/exit_right/merge -> the v2.1 answer. Every richer token requires
    its FULL signature; otherwise the v2.1 token stands (R3: an honest coarse
    label beats an invented precise one)."""
    base = route_from_future_v21(poses, t, horizon_steps, **v21_kw)
    tok21 = _V21_TO_V3_TOKEN[base["route"]]
    out = dict(base, token=tok21, token_v21=tok21, upgraded=False,
               dist_m=None, dist_band="d_unknown", maneuver_arc_m=0.0,
               maneuver_dyaw=0.0, roundabout_candidate=False,
               uturn_roundabout_confounded=False, n_segments=0,
               v3_rule="v21_base",
               token_valid=bool(base["valid"]))
    tr = _future_track(poses, t, horizon_steps)
    if tr["h"] < 1 or tr["arc"] < MIN_ARC_ROUTE_M:
        return out                                  # no road -> no claim at all

    segs = [(i0, i1, sg) for (i0, i1, sg) in _curv_segments(tr["ks"],
                                                            CURV_TURN_PER_M)
            if abs(float(tr["step_dyaw"][i0:i1].sum())) >= SEG_MIN_DYAW_RAD]
    out["n_segments"] = len(segs)

    # ---- DISTANCE-TO-MANEUVER: the first junction-scale event ahead ----------
    if segs:
        st = _seg_stats(tr, segs[0][0], segs[0][1])
        out["dist_m"] = st["start_m"]
        out["maneuver_arc_m"] = st["arc_m"]
        out["maneuver_dyaw"] = st["dyaw"]
        out["dist_band"] = dist_band(st["start_m"], tr["arc"])
    else:
        out["dist_band"] = dist_band(None, tr["arc"])

    # ---- richer TOKENS, each gated on its full signature ---------------------
    for i0, i1, sgn in segs:
        st = _seg_stats(tr, i0, i1)
        swept = abs(st["dyaw"])
        rev = _signed_reversal(tr, i1, sgn)
        tail_m = tr["arc"] - st["end_m"]          # road observed AFTER the arc
        tight_ring = st["mean_kappa"] >= CURV_ROUNDABOUT_PER_M
        # -- u_turn: ~180 deg, very tight, ends ANTIPARALLEL, no exit reversal
        if (st["mean_kappa"] >= CURV_UTURN_PER_M
                and UTURN_DYAW_LO_RAD <= swept <= UTURN_DYAW_HI_RAD
                and abs(abs(st["dyaw"]) - math.pi) <= UTURN_ANTIPARALLEL_RAD
                and tail_m >= UTURN_MIN_TAIL_M
                and rev < ROUNDABOUT_EXIT_REV_RAD):
            out.update(token="u_turn", upgraded=True, token_valid=True,
                       dist_m=st["start_m"], maneuver_arc_m=st["arc_m"],
                       maneuver_dyaw=st["dyaw"], v3_rule="u_turn",
                       dist_band=dist_band(st["start_m"], tr["arc"]),
                       uturn_roundabout_confounded=bool(
                           tight_ring and st["kappa_cv"] <= ROUNDABOUT_KAPPA_CV_MAX))
            return out
        # -- roundabout: sustained constant-radius ring + an OBSERVED exit
        if tight_ring and swept >= ROUNDABOUT_DYAW_RAD \
                and st["kappa_cv"] <= ROUNDABOUT_KAPPA_CV_MAX:
            if rev >= ROUNDABOUT_EXIT_REV_RAD:
                out.update(token="roundabout", upgraded=True, token_valid=True,
                           dist_m=st["start_m"], maneuver_arc_m=st["arc_m"],
                           maneuver_dyaw=st["dyaw"],
                           dist_band=dist_band(st["start_m"], tr["arc"]),
                           v3_rule="roundabout")
                return out
            # ring seen, exit not observed (clip ends mid-circulation): keep
            # v2.1's defensible turn_left/turn_right and SAY SO. This is also
            # where an unfinished ring lands that would otherwise have looked
            # like a u_turn — the two are the SAME ego track until the exit is
            # observed, and guessing between them is not a label.
            out["roundabout_candidate"] = True

    # -- exit / fork / merge: offset accumulates, heading RETURNS ---------------
    if not out["upgraded"]:
        lat = tr["ego"][:, 1]                              # +y = left, metres
        lat_final = float(lat[-1])
        cum = tr["cum_dyaw"]
        peak_dyaw = float(cum.abs().amax()) if cum.numel() else 0.0
        net_dyaw = float(cum[-1]) if cum.numel() else 0.0
        peak_kappa = float(tr["ks"].abs().amax()) if tr["ks"].numel() else 0.0
        dv = float(tr["seg"][-1, 3] - tr["seg"][0, 3])
        diverges = (abs(lat_final) >= EXIT_LAT_MIN_M
                    and abs(lat_final) >= 0.8 * float(lat.abs().amax()))
        returns = (peak_dyaw >= EXIT_PEAK_DYAW_MIN_RAD
                   and abs(net_dyaw) <= EXIT_RETURN_FRAC * peak_dyaw)
        v_min = float(tr["seg"][:, 3].min())
        if (diverges and returns and abs(net_dyaw) <= EXIT_NET_DYAW_MAX_RAD
                and peak_kappa < CURV_TURN_PER_M
                and v_min >= EXIT_MIN_SPEED_MS):
            if dv <= EXIT_DV_MS:                # slowing onto a ramp = leaving
                out.update(token="exit_left" if lat_final > 0 else "exit_right",
                           upgraded=True, token_valid=True, v3_rule="exit")
            elif dv >= MERGE_DV_MS:             # speeding onto a road = joining
                out.update(token="merge", upgraded=True, token_valid=True,
                           v3_rule="merge")
            # else: the two are the SAME ego-track shape and the speed change
            # does not disambiguate -> leave the v2.1 token. Honest gap.
    return out


def route_target_v3(poses: Tensor, t: int,
                    horizon_steps: int = NAV_HORIZON_STEPS, **kw) -> tuple:
    """v3 route target at ``t`` as ``(token, dist_band, valid)`` — the pair-plus
    form for a token-vocabulary head. The 3-class CE target is unchanged: use
    :func:`route_target_v21` for it (v3 does not touch it by construction)."""
    r = route_from_future_v3(poses, t, horizon_steps, **kw)
    return r["token"], r["dist_band"], bool(r["valid"])


# ----------------------------------------------------------------------------
# FACTORIZED TACTICAL — LATMANEUVER x LONMODE x TACPOINT, minted INDEPENDENTLY
# ----------------------------------------------------------------------------
# The 5-way `classify_maneuver*` above is a PRIORITY collapse of two orthogonal
# axes. These functions mint each axis on its own, so `lane_keep` AND
# `stop_at_point` AND a stop point 12 m ahead can all be true at once — which is
# the actual state of the world in the ep19 pedestrian-crossing clip.
#
# HONEST LIMITS, stated up front rather than faked:
#   LONMODE  follow_lead / close_gap / open_gap are LEAD-REFERENCED. `lead_state`
#            is a None stub in this repo, so they are NOT mintable and are never
#            emitted. Their windows fall to free_cruise/coast — which is why the
#            LONMODE mint must be read as "what the EGO is doing", not "why".
#   LATMAN   merge_in / yield_merge need another agent; `abort_lc` and
#            `pull_over` are mintable but rare; the rest are ego-track facts.
#   TACPOINT the POSITION of a stop is a trajectory fact and IS minted
#            (``stop_dist_m``). The NAME of that point — stop_line vs pedestrian
#            crossing vs a queue behind a lead vehicle — is NOT separable from
#            the ego track, so the TACPOINT TOKEN stays `unknown` (VLM/map). This
#            is the single most useful honest gap in the whole labeler: we can
#            always say WHERE, never WHY.
LON_KINEMATIC_TOKENS = ("free_cruise", "stop_at_point", "hold_stop", "launch",
                        "creep", "coast")
LON_LEAD_TOKENS = ("follow_lead", "close_gap", "open_gap")   # NOT mintable
LAT_KINEMATIC_TOKENS = ("lane_keep", "lc_left", "lc_right", "nudge_left",
                        "nudge_right", "abort_lc", "pull_over")
LAT_CONTEXT_TOKENS = ("merge_in", "yield_merge")             # NOT mintable

LON_HORIZON_STEPS = 20         # 2 s — the tactical head's farthest waypoint
STOP_SEARCH_STEPS = 70         # 7 s — how far ahead a stop point is looked for.
                               # The 2 s head CANNOT see a stop 45 m away at
                               # 13 m/s (a comfortable 2.2 m/s^2 stop takes ~6 s);
                               # that is why the distance exists at all. 7 s
                               # covers the whole urban range and is under
                               # nuPlan's 8 s planning horizon.
LON_MIN_STEPS = 10             # < 1 s of future -> a LONMODE claim is not honest
LAT_HORIZON_STEPS = 40         # 4 s — a lane change takes ~3-5 s
CREEP_CEIL_MS = 2.5            # moving but below this, sustained = creep
LANE_HALF_M = 1.75             # half a ~3.5 m lane
LC_NET_YAW_MAX = 0.20          # rad: a lane change nets ~0 heading
PULL_OVER_LAT_M = 2.0          # offset that ends STOPPED and stays stopped
ABORT_LC_RETURN_FRAC = 0.5     # peak offset >= 1 lane, final < this * peak


def lonmode_from_future(poses: Tensor, t: int,
                        horizon: int = LON_HORIZON_STEPS,
                        stop_search: int = STOP_SEARCH_STEPS) -> dict:
    """LONMODE (vocab, 9) minted from the ego speed profile alone + the metric
    DISTANCE to the stop point when one exists.

    Returns ``{token, valid, active, stop_dist_m, stop_dist_band, dv, v0}``.
    ``active`` is True for every mode except ``free_cruise`` — the "is a
    longitudinal decision happening here" bit the 5-way head cannot carry when a
    lateral class outranks it."""
    T = poses.shape[0]
    out = {"token": TOKEN_UNKNOWN, "valid": False, "active": False,
           "stop_dist_m": None, "stop_dist_band": "d_unknown", "dv": 0.0,
           "v0": float(poses[t, 3]) if 0 <= t < T else 0.0}
    fut = poses[t:min(t + horizon + 1, T)]
    if fut.shape[0] < LON_MIN_STEPS:      # not enough future to claim a mode
        return out
    v0 = float(poses[t, 3])
    v_end = float(fut[-1, 3])
    v_med = float(fut[:, 3].median())
    out["dv"] = v_end - v0

    # stop point: the first future step whose speed drops below STOP_V_MS,
    # searched over a LONGER window than the 2 s tactical horizon, with the arc
    # length to it (the planner's actual input).
    ssearch = poses[t:min(t + stop_search + 1, T)]
    arc_obs = 0.0
    stop_i = None
    if ssearch.shape[0] >= 2:
        ds = (ssearch[1:, :2] - ssearch[:-1, :2]).norm(dim=-1)
        arc_obs = float(ds.sum())
        below = (ssearch[1:, 3] < STOP_V_MS).nonzero().flatten()
        if below.numel():
            stop_i = int(below[0])
            out["stop_dist_m"] = float(ds[:stop_i + 1].sum())
    out["stop_dist_band"] = dist_band(out["stop_dist_m"], arc_obs)

    stopped = v0 < STOP_V_MS
    if stopped and v_med < STOP_V_MS:
        tok = "hold_stop"
    elif stopped and out["dv"] > DV_ACCEL_MS:
        tok = "launch"
    elif v0 >= MOVING_V_MS and stop_i is not None:
        tok = "stop_at_point"          # braking TO a stop, wherever it is
    elif not stopped and v0 < CREEP_CEIL_MS and v_med < CREEP_CEIL_MS:
        tok = "creep"
    elif out["dv"] < DV_BRAKE_MS:
        tok = "coast"                  # decelerating, not to a stop
    else:
        tok = "free_cruise"            # moving; WHY is lead-state, not ego
    out.update(token=tok, valid=True, active=tok != "free_cruise")
    return out


def latmaneuver_from_future(poses: Tensor, t: int,
                            horizon: int = LAT_HORIZON_STEPS) -> dict:
    """LATMANEUVER (vocab, 9) minted from the ego-frame lateral profile alone.

    Returns ``{token, valid, active, lat_m, peak_lat_m, net_yaw}``. A net
    heading change beyond ``LC_NET_YAW_MAX`` is a ROUTE turn, not a within-lane
    lateral maneuver, and reads ``lane_keep`` here (the ego does keep its lane
    through a junction turn) — the two axes stay orthogonal on purpose."""
    T = poses.shape[0]
    out = {"token": TOKEN_UNKNOWN, "valid": False, "active": False,
           "lat_m": 0.0, "peak_lat_m": 0.0, "net_yaw": 0.0}
    h = min(int(horizon), T - 1 - t)
    if h < 5:
        return out
    seg = poses[t:t + h + 1]
    net_yaw = float(wrap_to_pi(seg[-1, 2] - seg[0, 2]))
    lat = ego_frame(seg[:, :2] - seg[0, :2], seg[0, 2].expand(h + 1))[:, 1]
    lat_f, peak = float(lat[-1]), float(lat.abs().amax())
    out.update(lat_m=lat_f, peak_lat_m=peak, net_yaw=net_yaw, valid=True)
    if abs(net_yaw) > LC_NET_YAW_MAX:
        out["token"] = "lane_keep"
        return out
    stopped_end = (float(seg[-1, 3]) < STOP_V_MS
                   and float(seg[max(0, h - 10):, 3].median()) < STOP_V_MS)
    if abs(lat_f) >= PULL_OVER_LAT_M and stopped_end:
        out.update(token="pull_over", active=True)
    elif peak >= LANE_HALF_M and abs(lat_f) < ABORT_LC_RETURN_FRAC * peak:
        out.update(token="abort_lc", active=True)
    elif abs(lat_f) >= LANE_HALF_M:
        out.update(token="lc_left" if lat_f > 0 else "lc_right", active=True)
    elif abs(lat_f) >= LANE_HALF_M / 2.0:
        out.update(token="nudge_left" if lat_f > 0 else "nudge_right",
                   active=True)
    else:
        out["token"] = "lane_keep"
    return out


def tactical_from_future_v3(poses: Tensor, t: int,
                            lat_horizon: int = LAT_HORIZON_STEPS,
                            lon_horizon: int = LON_HORIZON_STEPS,
                            stop_search: int = STOP_SEARCH_STEPS) -> dict:
    """One window -> the FACTORIZED tactical label: LATMANEUVER, LONMODE and
    the (honestly unknown) TACPOINT with its metric distance.

        lat / lon        the per-slot dicts above
        tacpoint         always ``unknown`` from kinematics — the POSITION is
                         minted (``stop_dist_m``), the NAME needs vision/map
        man5             the shipped 5-way class at this window (for A/B)
        collapsed        True iff man5 is a LATERAL class while the factorized
                         LONMODE is active — i.e. the 5-way label DESTROYED a
                         live longitudinal decision. This is the count the
                         label-side defect is measured by."""
    lon = lonmode_from_future(poses, t, lon_horizon, stop_search)
    lat = latmaneuver_from_future(poses, t, lat_horizon)
    T = poses.shape[0]
    man5 = None
    if t + LABEL_HORIZON < T:
        sub = poses[t:t + LABEL_HORIZON + 1][None]        # [1, H+1, 4]
        man5 = int(classify_maneuver_v2(sub)[0])
    return {"lat": lat, "lon": lon,
            "tacpoint": {"token": TOKEN_UNKNOWN, "valid": False,
                         "stop_dist_m": lon["stop_dist_m"],
                         "stop_dist_band": lon["stop_dist_band"],
                         "why": "position minted, NAME needs vision/map "
                                "(stop_line vs crossing vs queue-behind-lead)"},
            "man5": man5,
            "collapsed": bool(man5 in (LANE_KEEP, TURN_LEFT, TURN_RIGHT)
                              and lon["active"])}
