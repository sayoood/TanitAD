"""D4 (refcv8 data audit) -- pure-numpy label definitions for the refcv8 design, with literal thresholds.

Every threshold is a LITERAL from road geometry / vehicle kinematics or from a cited deployment rule,
never imported from the label builder or the trainer (CONTEXT.md binding rule: a cross-check must be
derived independently of the value it checks).

Frames: provider rows of the v2ep manifest (10 Hz nominal, raw row = provider row + 2), poses (x, y, yaw, v).
Window t: NOW = provider row t + 7 (window 8); future rows NOW+1 .. NOW+60 (6 s); windows t in [0, T-28).
NOW-frame: x forward, y LEFT.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

# ---------------------------------------------------------------- literals ---------------------------
W, MAXH, RAW_OFF, FUT = 8, 20, 2, 60          # trainer window / enumeration horizon / n_stack-1 / 6 s
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)    # the planner's slots (ticks)
# route-following SPEC sec. 3 GT class (re-typed literals, not imported)
TURN_DEG, STRAIGHT_DEG, MIN_LEN_M, STALL_M = 30.0, 10.0, 5.0, 0.05
TAU_C = 0.18063741505146028                   # the run's own --nav-compliance-tau-rad (config.json argv)
# NavSim / OpenScene driving_command (INHERITED, EvalFlyWheel 2026-09-19 route-leak RESULT:
# helpers/driving_command.py:40-100 -> "20 m ahead -> |y| >= 2 m")
NAV_LOOK_M, NAV_LAT_M = 20.0, 2.0
# dense tactical labels (road geometry / kinematics literals)
LAT_TURN_DEG = 30.0        # heading change that is a junction turn, not a lane manoeuvre
LAT_TURN_R_M = 40.0        # minimum radius at or below which a >=30 deg change is a junction turn
LAT_STRAIGHT_PSI_DEG = 10.0  # |heading change| at the horizon end below which lateral offset is readable
LANE_W_MIN_M = 2.5         # a completed lane change moves >= 2.5 m (EU/US lanes 2.75-3.75 m)
NUDGE_M = 1.0              # a deliberate in-lane deviation (half the 2 m clearance budget of a parked car)
STEER_DEG = 3.0            # the path must actually rotate (a pure heading-misalignment drift does not)
V_STOP, V_CREEP = 0.5, 2.0  # m/s
DV_SIG = 1.5               # m/s change over the horizon: the DOCUMENTED v7 ACCELERATE/BRAKE bar (D2: builder used +1.0 in effect)
CURVE_R_M = 60.0           # radius at or below which a speed reduction is "for the curve"
V_FLOOR = 1.0              # kappa = omega / max(v, 1) (rest guard)
SMOOTH = 5                 # 0.5 s moving average for omega
LADDER8_KMH = (20, 30, 50, 70, 80, 100, 120, 130)   # road-law posted limits (label file's pinned ladder)
LADDER4_KMH = (30, 50, 100, 120)                     # the refcv6/7 4-way one-hot ladder (config.json)

LAT_NAMES = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LON_NAMES = ("HOLD", "CREEP", "BRAKE_TO", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE", "CRUISE")


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def snap_up(v_ms, ladder_kmh):
    """Containing-window ladder index: lowest step >= v (km/h), top step caps."""
    v = np.asarray(v_ms, np.float64) * 3.6
    steps = np.asarray(ladder_kmh, np.float64)
    idx = np.searchsorted(steps, v - 1e-9, side="left")
    return np.minimum(idx, len(steps) - 1)


# ---------------------------------------------------------------- window geometry --------------------
def window_block(poses: np.ndarray):
    """Per-clip arrays for every window. poses [T, 4] float64."""
    T = poses.shape[0]
    n = T - (W + MAXH)
    t = np.arange(n)
    r = t + W - 1
    K = np.arange(0, FUT + 1)
    idx = r[:, None] + K[None, :]
    valid = idx <= T - 1
    idc = np.minimum(idx, T - 1)
    x, y, yaw, v = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    c, s = np.cos(yaw[r])[:, None], np.sin(yaw[r])[:, None]
    dx, dy = x[idc] - x[r][:, None], y[idc] - y[r][:, None]
    X, Y = c * dx + s * dy, -s * dx + c * dy
    yaw_u = np.unwrap(yaw)
    dpsi = yaw_u[idc] - yaw_u[r][:, None]
    seg = np.hypot(np.diff(x), np.diff(y))
    S = np.concatenate([[0.0], np.cumsum(seg)])
    return dict(T=T, n=n, t=t, r=r, valid=valid, idc=idc, X=X, Y=Y, dpsi=dpsi, V=v[idc], v0=v[r], S=S,
                yaw_u=yaw_u, v=v)


def gt_class(B):
    """Route RESULT SPEC sec. 3: terminal heading of the slot-50 -> slot-60 segment."""
    h = np.asarray(HORIZONS)
    P = np.stack([B["X"][:, h], B["Y"][:, h]], -1)               # [n, 8, 2]
    d = P[:, -1] - P[:, -2]
    th = np.arctan2(d[:, 1], d[:, 0])
    th = np.where(np.hypot(d[:, 0], d[:, 1]) < STALL_M, 0.0, th)
    Q = np.concatenate([np.zeros((P.shape[0], 1, 2)), P], 1)
    ln = np.linalg.norm(np.diff(Q, axis=1), axis=-1).sum(-1)
    ok = B["valid"][:, 60] & (ln >= MIN_LEN_M)
    deg = np.degrees(np.abs(th))
    cls = np.full(th.shape, "unclassified", dtype=object)
    cls[ok & (deg >= TURN_DEG) & (th > 0)] = "turnL"
    cls[ok & (deg >= TURN_DEG) & (th < 0)] = "turnR"
    cls[ok & (deg < STRAIGHT_DEG)] = "straight"
    cls[ok & (deg >= STRAIGHT_DEG) & (deg < TURN_DEG)] = "gentle"
    return cls, th


def compliance(th, side):
    """the model's nav-compliance predicate: terminal heading has the commanded sign and |th| >= tau."""
    return np.where(side > 0, th >= TAU_C, np.where(side < 0, th <= -TAU_C, False))


# ---------------------------------------------------------------- time-localised nav ----------------
def nav_path_rule_xy(poses, B, look_m=NAV_LOOK_M, lat_m=NAV_LAT_M):
    """NavSim-equivalent command from the ego's own driven path: lateral offset (NOW frame) of the point
    `look_m` of arc ahead; |y| >= lat_m -> L(+1)/R(-1) else straight (0). Undefined (-9) when the clip
    ends before `look_m` of arc is driven. Uses every remaining clip row, not only 6 s (a LABEL may)."""
    S, T, r = B["S"], B["T"], B["r"]
    yaw = poses[:, 2]
    tgt = S[r] + look_m
    j = np.searchsorted(S, tgt, side="left")
    ok = j <= T - 1
    jc = np.clip(j, 1, T - 1)
    # linear interpolation to EXACTLY look_m of arc (a 0.1 s row is up to 3 m at 30 m/s)
    seg = np.maximum(S[jc] - S[jc - 1], 1e-9)
    f = np.clip((tgt - S[jc - 1]) / seg, 0.0, 1.0)
    px = poses[jc - 1, 0] + f * (poses[jc, 0] - poses[jc - 1, 0])
    py = poses[jc - 1, 1] + f * (poses[jc, 1] - poses[jc - 1, 1])
    c, s = np.cos(yaw[r]), np.sin(yaw[r])
    dx, dy = px - poses[r, 0], py - poses[r, 1]
    lat = -s * dx + c * dy
    side = np.where(lat >= lat_m, 1, np.where(lat <= -lat_m, -1, 0)).astype(np.int8)
    side = np.where(ok, side, -9).astype(np.int8)
    return side, lat, ok


def nav_entry_rule(entries, s_now, look_m=NAV_LOOK_M):
    """Time-localised nav from nav_30s.entries (anchor-relative arc distances). s_now [n] = signed arc
    from the anchor to NOW along the driven path. Active entry: start within look_m ahead and not yet
    finished. Returns side [n] (+1/-1/0), dist_to_start [n] (inf if none), entry index."""
    n = len(s_now)
    side = np.zeros(n, np.int8)
    nside = np.zeros(n, np.int8)                 # side of the NEXT not-yet-finished turn (any distance)
    dist = np.full(n, np.inf)
    end = np.full(n, np.inf)
    ents = [e for e in entries if e.get("distance_m") is not None]
    for e in sorted(ents, key=lambda q: q["distance_m"]):
        tok = e.get("token")
        sd = 1 if tok == "NAV_TURN_L" else (-1 if tok == "NAV_TURN_R" else 0)
        if sd == 0:
            continue
        d0 = float(e["distance_m"]) - s_now
        # a turn whose end lies beyond the label horizon carries no distance_end_m: still in progress
        d1 = (float(e["distance_end_m"]) if e.get("distance_end_m") is not None else np.inf) - s_now
        # the NEXT turn not yet completed (d1 > 0) gives the distance/time args
        nxt = (d1 > 0) & ~np.isfinite(dist)
        nside = np.where(nxt, sd, nside)
        dist = np.where(nxt, d0, dist)
        end = np.where(nxt, d1, end)
        act = nxt & (d0 <= look_m)
        side = np.where(act, sd, side)
    nav_entry_rule.next_side = nside
    return side, dist, end


# ---------------------------------------------------------------- dense tactical labels -------------
def dense_tactical(B, min_rows=40):
    """Per-window lat/lon tactical label over the next 6 s (truncated to the available rows, IGNORE when
    fewer than `min_rows` future rows exist). Returns lat (index into LAT_NAMES or -100), lon (LON_NAMES
    or -100), aux dict."""
    n = B["n"]
    valid = B["valid"]
    nfut = valid.sum(1) - 1                                   # future rows available (k >= 1)
    ok = nfut >= min_rows
    X, Y, dpsi, V = B["X"], B["Y"], B["dpsi"], B["V"]
    # last valid column per window
    last = np.minimum(nfut, FUT)
    ar = np.arange(n)
    # curvature over the horizon (smoothed omega / v)
    om = np.diff(dpsi, axis=1) * 10.0                          # rad/s at the nominal 10 Hz
    ker = np.ones(SMOOTH) / SMOOTH
    om_s = np.apply_along_axis(lambda a: np.convolve(a, ker, mode="same"), 1, om)
    v_s = np.apply_along_axis(lambda a: np.convolve(a, ker, mode="same"), 1, V[:, 1:])
    kap = np.abs(om_s) / np.maximum(v_s, V_FLOOR)
    colv = valid[:, 1:]
    kap = np.where(colv, kap, 0.0)
    kmax = kap.max(1)
    dpsi_deg = np.degrees(np.where(valid, dpsi, 0.0))
    absd = np.abs(dpsi_deg)
    big = absd >= LAT_TURN_DEG
    has_big = big.any(1)
    first_big = np.argmax(big, 1)
    side_big = np.sign(dpsi_deg[ar, first_big])
    is_turn = has_big & (kmax >= 1.0 / LAT_TURN_R_M)
    psi_end = dpsi_deg[ar, last]
    y_end = Y[ar, last]
    ymask = valid
    Yv = np.where(ymask, Y, 0.0)
    i_pk = np.argmax(np.abs(Yv), 1)
    y_pk = Yv[ar, i_pk]
    near_straight = (np.abs(psi_end) < LAT_STRAIGHT_PSI_DEG) & (absd.max(1) < 20.0)
    steered = absd.max(1) >= STEER_DEG
    lc = near_straight & steered & (np.abs(y_end) >= LANE_W_MIN_M) & (np.abs(psi_end) <= 5.0)
    ndg = near_straight & steered & ~lc & (np.abs(y_pk) >= NUDGE_M) & (np.abs(y_pk) < LANE_W_MIN_M)
    lat = np.zeros(n, np.int64)                                   # LANE_KEEP
    lat[lc & (y_end > 0)] = 1
    lat[lc & (y_end < 0)] = 2
    lat[ndg & (y_pk > 0)] = 3
    lat[ndg & (y_pk < 0)] = 4
    lat[is_turn & (side_big > 0)] = 5
    lat[is_turn & (side_big < 0)] = 6
    lat = np.where(ok, lat, -100)
    # longitudinal
    Vv = np.where(valid, V, np.nan)
    v0 = B["v0"]
    vmax = np.nanmax(Vv, 1)
    vmin = np.nanmin(Vv, 1)
    i_min = np.nanargmin(np.where(valid, V, np.inf), 1)
    i_max = np.nanargmax(np.where(valid, V, -np.inf), 1)
    hold = (v0 <= V_STOP) & (vmax <= V_STOP)
    creep = ~hold & (vmax <= V_CREEP)
    brake = (vmin <= v0 - DV_SIG)
    accel = (vmax >= v0 + DV_SIG)
    # chronological precedence when both happen in the horizon
    first_brake = brake & (~accel | (i_min <= i_max))
    first_accel = accel & ~first_brake
    curve = kmax >= 1.0 / CURVE_R_M
    lon = np.full(n, 5, np.int64)                                 # CRUISE
    lon[first_accel] = 4
    lon[first_brake & curve] = 3
    lon[first_brake & ~curve] = 2
    lon[creep] = 1
    lon[hold] = 0
    lon = np.where(ok, lon, -100)
    aux = dict(kmax=kmax, dpsi_absmax=absd.max(1), psi_end=psi_end, y_end=y_end, y_pk=y_pk, is_turn=is_turn, ok=ok, nfut=nfut)
    return lat, lon, aux


# ---------------------------------------------------------------- synthetic controls ----------------
def synth(kind: str, v=8.0, T=199):
    """Analytic tracks at 10 Hz: straight, left/right 90-deg junction turn (R 15 m), lane change L 3.5 m,
    nudge R 1.2 m (out and back), stop-and-hold, accelerate."""
    dt = 0.1
    xs, ys, yaws, vs = [0.0], [0.0], [0.0], [v]
    for i in range(1, T):
        tt = i * dt
        vv, w = v, 0.0
        if kind in ("turnL", "turnR") and 9.0 <= tt < 9.0 + (math.pi / 2 * 15.0) / v:
            w = (v / 15.0) * (1 if kind == "turnL" else -1)
        if kind == "lcL" and 9.0 <= tt < 13.0:
            # lateral position follows 3.5 * (1 - cos(pi * u)) / 2 over 4 s -> heading = atan(dy/dx)
            pass
        if kind == "stop":
            vv = max(0.0, v - 1.5 * max(0.0, tt - 9.0))
        if kind == "accel":
            vv = v + 1.0 * max(0.0, min(tt - 9.0, 5.0))
        # midpoint heading: exact chord of a constant-curvature step, so the turn starts at x = 72 m
        yaw = yaws[-1] + w * dt
        ym = 0.5 * (yaws[-1] + yaw)
        xs.append(xs[-1] + vv * math.cos(ym) * dt)
        ys.append(ys[-1] + vv * math.sin(ym) * dt)
        yaws.append(yaw)
        vs.append(vv)
    P = np.stack([xs, ys, yaws, vs], -1)
    if kind in ("lcL", "nudgeR"):
        x = np.arange(T) * dt * v
        tt = np.arange(T) * dt
        u = np.clip((tt - 9.0) / 4.0, 0, 1)
        if kind == "lcL":
            yy = 3.5 * (1 - np.cos(math.pi * u)) / 2
        else:
            yy = -1.2 * np.sin(math.pi * u)
        dy = np.gradient(yy, x)
        P = np.stack([x, yy, np.arctan(dy), np.full(T, v)], -1)
    return P
