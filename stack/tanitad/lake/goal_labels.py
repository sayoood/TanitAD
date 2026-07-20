"""Stage 4 — KINEMATIC v3-goal minting (TanitDataSet rev-3 §7.4 "Label minting").

Mints the slots of the frozen v3 goal (``vocab``) that ego kinematics can derive
HONESTLY, right now, on CPU — every one stamped ``provenance=kinematic``. The
slots that need pixels/lead-state/map (VTARGET sign-cap, VSOURCE=sign_limit,
LONMODE lead modes, HEADWAY, INTERACT, LIGHTSTATE, the STRATEGIC layer, …) stay
``unknown`` for the deferred Cosmos-Reason2 pass to fill (R3: honest gaps, never a
guess).

The maneuver/route derivation is the v2 curvature-relative labeler
(``scripts/refb_labels.py``) — REUSED, not re-implemented, so these labels are
identical to the trainer's on-the-fly ones and inherit the AMBIGUOUS→masked
discipline (a gentle road curve is not a route turn; the gray zone is ``unknown``).

Pure functions over the episode contract poses ``[T,4] = (x, y, yaw, v)``. No
model, no I/O, unit-tested on synthetic trajectories.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor

from tanitad.lake import vocab as V

# --------------------------------------------------------------------------- #
# refb_labels lives in scripts/ (not the package) — append (never prepend, must #
# not shadow stdlib) exactly as tanitad.data.physicalai._refb_labels does.      #
# --------------------------------------------------------------------------- #


def _refb():
    try:
        import refb_labels                                        # noqa: F401
    except ModuleNotFoundError:
        scripts = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts not in sys.path:
            sys.path.append(scripts)
        import refb_labels                                        # noqa: F401
    return refb_labels


# --------------------------------------------------------------------------- #
# Documented kinematic thresholds (10 Hz contract). Shared with refb where it   #
# already defines them (STOP/MOVING/DV) so the two labelers never diverge.      #
# --------------------------------------------------------------------------- #
DT_DEFAULT = 0.1
VTARGET_LO_STEPS = 100         # free-flow lookahead window start (10 s @ 10 Hz)
VTARGET_HI_STEPS = 200         # ...end (20 s) — V3: "85th-pct over 10-20 s"
VTARGET_PCT = 0.85
CREEP_CEIL_MS = 2.5            # sustained speed below this (but moving) = creep
LANE_HALF_M = 1.75            # half a ~3.5 m lane — a lateral offset gate
LC_NET_YAW_MAX = 0.20        # rad: a lane change nets ~0 heading (else it's a turn)
# DYN intensity edges on peak |accel| in the near window [m/s^2]
DYN_EDGES = ((1.0, "gentle"), (2.5, "normal"), (4.0, "firm"))   # else "max"


def _accel(poses: Tensor, dt: float = DT_DEFAULT) -> Tensor:
    """Forward finite-difference longitudinal accel from pose speed v [T]->[T]."""
    v = poses[:, 3]
    a = torch.zeros_like(v)
    if v.shape[0] >= 2:
        a[:-1] = (v[1:] - v[:-1]) / dt
        a[-1] = a[-2]
    return a


# --------------------------------------------------------------------------- #
# Per-slot kinematic minting — each returns a provenance-stamped vocab.slot     #
# (or an unknown_slot when kinematics cannot honestly decide).                  #
# --------------------------------------------------------------------------- #
def vtarget_at(poses: Tensor, t: int, lo: int = VTARGET_LO_STEPS,
               hi: int = VTARGET_HI_STEPS, pct: float = VTARGET_PCT
               ) -> dict[str, str]:
    """VTARGET = the ``pct``-quantile of future speed over ``[t+lo, t+hi]``, banded
    (V3 Q1 non-uniform). This is the KINEMATIC free-flow set-speed proxy; the
    sign/map CAP that tightens it is VLM/map-pending (left to the deferred pass).
    ``provenance=kinematic``; ``unknown`` when the lookahead has no future."""
    T = poses.shape[0]
    a, b = t + lo, min(t + hi, T)
    if b - a < 3:                                     # too little free-flow future
        return V.unknown_slot()
    fv = poses[a:b, 3]
    q = float(torch.quantile(fv, pct))
    return V.slot(V.vtarget_band(q), "kinematic")


def vsource_at(poses: Tensor, t: int, horizon: int = None) -> dict[str, str]:
    """VSOURCE = WHY this set-speed. Kinematics can honestly assert
    ``curve_constrained`` (a tight upcoming curve limits speed) via the v2 peak
    curvature; ``sign_limit`` / ``lead_constrained`` / ``traffic_flow`` are
    VLM/map-pending. Falls back to ``road_class_default`` (kinematic: no curve/lead
    constraint detected). ``provenance=kinematic``."""
    R = _refb()
    horizon = horizon or R.NAV_HORIZON_STEPS
    r = R.route_from_future(poses, t, horizon, R.NAV_MIN_STEPS)
    if r["peak_kappa"] >= R.CURV_TURN_PER_M:          # tight upcoming curve
        return V.slot("curve_constrained", "kinematic")
    return V.slot("road_class_default", "kinematic")


def lonmode_at(poses: Tensor, t: int, horizon: int = 20,
               dt: float = DT_DEFAULT) -> dict[str, str]:
    """LONMODE — the KINEMATIC longitudinal modes only (V3 LONMODE, 9). The
    lead-referenced modes (``follow_lead``/``close_gap``/``open_gap``) need lead
    state → VLM-pending, never emitted here. ``provenance=kinematic``.

    v_now + the 2 s future speed profile decide: hold_stop / launch / creep /
    stop_at_point / coast / free_cruise."""
    R = _refb()
    T = poses.shape[0]
    v0 = float(poses[t, 3])
    fut = poses[t:min(t + horizon + 1, T), 3]
    if fut.numel() < 2:
        return V.unknown_slot()
    v_end = float(fut[-1])
    v_med = float(fut.median())
    dv = v_end - v0
    stopped = v0 < R.STOP_V_MS
    if stopped and v_med < R.STOP_V_MS:
        return V.slot("hold_stop", "kinematic")
    if stopped and dv > R.DV_ACCEL_MS:
        return V.slot("launch", "kinematic")
    if v0 >= R.MOVING_V_MS and v_end < R.STOP_V_MS:
        return V.slot("stop_at_point", "kinematic")
    if v0 < CREEP_CEIL_MS and v_med < CREEP_CEIL_MS and not stopped:
        return V.slot("creep", "kinematic")
    if dv < R.DV_BRAKE_MS:                            # decelerating, not to a stop
        return V.slot("coast", "kinematic")
    return V.slot("free_cruise", "kinematic")         # moving, no lead detectable


def dyn_at(poses: Tensor, t: int, horizon: int = 20,
           dt: float = DT_DEFAULT) -> dict[str, str]:
    """DYN = intensity of the active longitudinal mode from peak |accel| over the
    near window (V3 DYN absorbs decel_soft/hard). ``provenance=kinematic``."""
    T = poses.shape[0]
    a = _accel(poses, dt)[t:min(t + horizon + 1, T)]
    if a.numel() == 0:
        return V.unknown_slot()
    peak = float(a.abs().max())
    for edge, tok in DYN_EDGES:
        if peak < edge:
            return V.slot(tok, "kinematic")
    return V.slot("max", "kinematic")


def route_at(poses: Tensor, t: int, horizon: int = None) -> dict[str, str]:
    """ROUTE (strategic) via the v2 curvature-relative labeler: a genuine, tight,
    transient junction turn → ``turn_left``/``turn_right``; road-following →
    ``follow``; AMBIGUOUS gray-zone (exit/fork not separable without a map) →
    ``unknown`` (R3). Richer route tokens (exit/merge/u_turn/roundabout) need a map
    → VLM/map-pending. ``provenance=kinematic``."""
    R = _refb()
    horizon = horizon or R.NAV_HORIZON_STEPS
    r = R.route_from_future(poses, t, horizon, R.NAV_MIN_STEPS)
    if not r["valid"]:
        return V.unknown_slot()                       # too little future OR ambiguous
    if r["route"] == R.ROUTE_LEFT:
        return V.slot("turn_left", "kinematic")
    if r["route"] == R.ROUTE_RIGHT:
        return V.slot("turn_right", "kinematic")
    return V.slot("follow", "kinematic")


def latmaneuver_at(poses: Tensor, t: int, horizon: int = 40,
                   dt: float = DT_DEFAULT) -> dict[str, str]:
    """LATMANEUVER — the KINEMATIC lateral set only: ``lane_keep`` /
    ``lc_left|lc_right`` / ``nudge_left|nudge_right``. A lane change nets ~0 heading
    (|net yaw| < LC_NET_YAW_MAX) yet accumulates a lateral offset ≳ half a lane;
    a smaller offset is a nudge; ~0 offset is lane_keep. merge/yield/abort/pull_over
    need map/VLM context → never emitted (stays unknown when the signature is
    ambiguous). ``provenance=kinematic``."""
    R = _refb()
    T = poses.shape[0]
    h = min(horizon, T - 1 - t)
    if h < 5:
        return V.unknown_slot()
    seg = poses[t:t + h + 1]                          # [h+1, 4]
    net_yaw = float(R.wrap_to_pi(seg[-1, 2] - seg[0, 2]))
    # lateral offset of the endpoint in the START ego frame (+y = left)
    d_xy = seg[-1, :2] - seg[0, :2]
    lat = float(R.ego_frame(d_xy[None, :], seg[0, 2][None])[0, 1])
    if abs(net_yaw) > LC_NET_YAW_MAX:                 # a net heading change = a turn,
        return V.slot("lane_keep", "kinematic")       # not a within-lane lat maneuver
    if abs(lat) >= LANE_HALF_M:
        return V.slot("lc_left" if lat > 0 else "lc_right", "kinematic")
    if abs(lat) >= LANE_HALF_M / 2.0:
        return V.slot("nudge_left" if lat > 0 else "nudge_right", "kinematic")
    return V.slot("lane_keep", "kinematic")


# --------------------------------------------------------------------------- #
# The per-timestep minter — assemble one full goal tuple                        #
# --------------------------------------------------------------------------- #
def mint_kinematic_goal(poses: Tensor, t: int, *, has_can: bool = False,
                        blinker: str | None = None
                        ) -> dict[str, dict[str, str]]:
    """One full v3 goal at timestep ``t`` with the KINEMATIC slots minted and
    everything else ``unknown`` (VLM/map/engineered-pending).

    Kinematic (now): TACTICAL ``VTARGET, VSOURCE, LONMODE, LATMANEUVER, DYN`` +
    STRATEGIC ``ROUTE``. Honest-optional: ``SIGNAL`` is minted only where CAN
    carries the blinker (``has_can`` + ``blinker`` ∈ indicator_left/right/none) —
    V3's imitation-limited rule — stamped ``human``; else ``unknown``. Every other
    slot (HEADWAY, INTERACT, TACPOINT, LIGHTSTATE, RULECTX, and the rest of the
    STRATEGIC layer) is left ``unknown`` for the deferred Cosmos-Reason2 pass."""
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"poses must be [T,4], got {tuple(poses.shape)}")
    T = poses.shape[0]
    if not 0 <= t < T:
        raise ValueError(f"t={t} out of range for T={T}")

    g = V.empty_goal()
    g["VTARGET"] = vtarget_at(poses, t)
    g["VSOURCE"] = vsource_at(poses, t)
    g["LONMODE"] = lonmode_at(poses, t)
    g["DYN"] = dyn_at(poses, t)
    g["LATMANEUVER"] = latmaneuver_at(poses, t)
    g["ROUTE"] = route_at(poses, t)
    if has_can and blinker in ("indicator_left", "indicator_right", "none"):
        g["SIGNAL"] = V.slot(blinker, "human")        # CAN blinker (e.g. L2D)
    V.validate_goal(g)
    return g


def mint_goal_series(poses: Tensor, *, stride: int = 1, has_can: bool = False
                     ) -> list[dict[str, dict[str, str]]]:
    """Per-timestep kinematic goals for a whole episode (training-time use)."""
    return [mint_kinematic_goal(poses, t, has_can=has_can)
            for t in range(0, poses.shape[0], stride)]


# --------------------------------------------------------------------------- #
# Episode-level SUMMARY goal — the representative tuple for the per-episode      #
# sidecar + the curation strata (dominant kinematic token per slot).            #
# --------------------------------------------------------------------------- #
_KINEMATIC_SLOTS = ("VTARGET", "VSOURCE", "LONMODE", "LATMANEUVER", "DYN", "ROUTE")


def episode_goal_summary(poses: Tensor, *, stride: int = 5,
                         has_can: bool = False) -> dict[str, dict[str, str]]:
    """A single representative goal for the episode: the MODE (most frequent) of
    each kinematic slot's minted token across the episode's valid timesteps.

    This is the per-episode summary the sidecar/catalog stores and curation strata
    read; the trainer still mints per-window via :func:`mint_kinematic_goal`. The
    dominant token's provenance stays ``kinematic``; a slot that is ``unknown`` at
    every step stays ``unknown``."""
    T = poses.shape[0]
    if T < 3:
        return V.empty_goal()
    series = [mint_kinematic_goal(poses, t, has_can=has_can)
              for t in range(0, T, max(1, stride))]
    g = V.empty_goal()
    for s in _KINEMATIC_SLOTS:
        toks = [gg[s]["token"] for gg in series if gg[s]["prov"] == "kinematic"]
        if toks:
            top = Counter(toks).most_common(1)[0][0]
            g[s] = V.slot(top, "kinematic")
    V.validate_goal(g)
    return g
