"""HiCAP v10 imagination supervision producer (spec `Paper/HiCAP/design/HICAP_V10_SPEC.md` §3.1, paper sec:imag).

Pure numpy (no torch at import): per v9 row, from the corpus's logged ego track and `obstacle.offline` agent cuboids, produce
  * Tier 1 targets for `imagination.EHead` / `FullLoss` (`exo_*`): lead gap / closing speed / lead longitudinal acceleration at
    H_T = 0.5, 1, 2, 3 s, gap log-sigma (track jitter), behaviour mode, onset time (+ q10/q50/q90 band), has_lead, row weight;
  * a teacher cost J(a) with 3 channels (ttc, clearance, progress deficit) for ARBITRARY candidate trajectories against the agents'
    LOGGED futures (swept oriented boxes, vectorised), its scalar form for the Tier-0 listwise target, and the Tier-2 family costs;
  * a per-clip `build_rows(...)` that emits arrays keyed exactly as the trainer consumes them.

INPUT SCHEMA (stated before any formula; MEASURED 2026-10-25 on corpus `Sayood/tanitad-v7-training-corpus`, scratch
`/tmp/claude-0/streamS/`):
  egomotion `egomotion/egomotion_alpamayo.tar` -> `<clip_id>.parquet`: timestamp [us], x, y, z [m, clip-local world frame],
      qx qy qz qw (rig -> world), vx vy vz, ax ay az, curvature; ~100 Hz; starts at ~-0.2 s, runs 55-140 s.
  obstacle.offline `agents/obstacle_offline/obstacle_offline_NNN.parquet` (one row group per clip): timestamp_us [us, SAME clock as
      egomotion], track_id (str), center_x/y/z [m], size_x (length, along the box heading) / size_y (width) / size_z [m],
      orientation_x..w (quaternion), label_class (11 strings incl. protruding_object, train_or_tram_car), reference_frame == 'rig'
      and reference_frame_timestamp_us == timestamp_us on 100 % of rows => every box is in the RIG frame AT ITS OWN TIMESTAMP
      (+x forward, +y LEFT, origin rear axle). Tracks are sampled at ~10 Hz each with per-track jitter (median dt 100.0 ms);
      labels cover ~0-20 s of the clip.
  v9 rows: row j of a clip holds raw row k = k0 + j and `now_s = grid_start_s + k * dt_s` (dt_s ~= 0.10067, NOT 0.1).
      ⚠️ `now_s` is measured from the FIRST EGOMOTION SAMPLE: egomotion time = now_s + t_ego[0] (MEASURED on 20 eval139 clips:
      the speed |v| interpolated at now_s + t_ego[0] matches `row__v_now_ms` to 0.0001-0.015 m/s mean abs; at now_s it is off by
      0.02-0.14 m/s, i.e. a 0.19-0.2 s clock shift). `clip_clock()` below applies this.

CONVENTIONS
  * Frames: world = clip-local egomotion frame. Row frame ("NOW frame") = the ego rig at row j: +x forward, +y left.
  * gap = along-path arc distance from the ego RIG ORIGIN (rear axle) to the lead's REAR FACE (min arc coordinate over the box's
    4 corners) — the programme convention of `taniteval.lead_source` (never bumper-to-bumper).
  * The along-path coordinate is measured on the ego's LOGGED path from NOW (30 s of log, extended straight to 150 m).
  * Tier-1 descriptors are EGO-ACTION-FREE, in exactly the semantics `imagination._lead_pos/tier1_cost` integrate:
        L(t)       = arc position of the lead's rear face at NOW + t (NOW-anchored path),
        exo_gap_h  = L(h) - v0 * h            ("gap if the ego held v0"; == the actual gap at h = 0),
        exo_close_h= v0 - dL/dt(h)            ("closing speed if the ego held v0"),
        exo_acc_h  = d2L/dt2(h)               (lead longitudinal acceleration along the path).
    A target built from the ego's ACTUAL future gap would contain the ego's own action (the echo the paper forbids).
    ⚠️ `exo_gap_h` can be NEGATIVE (ego faster than the lead); `EHead.gap = softplus(.)*10 >= 0` cannot represent it — reported
    as an integration item, the fraction is MEASURED by the CLI.
  * Mode ids in `exo_mode` follow the index space of `imagination.MODE_ACC` = (-3, 0, +1.5, -6) m/s^2, because `tier1_cost`
    integrates MODE_ACC[m] for logit m: 0 brake, 1 hold (cruise), 2 accelerate, 3 brake hard. Label = nearest MODE_ACC value to the
    sustained extreme lead acceleration over [0, 3] s. A CUT-OUT (the lead leaves the corridor) has no MODE_ACC meaning and is
    IGNORE (-100) in `exo_mode`; the spec's 4-way label {cruise, brake, accelerate, cut-out} is `exo_mode4`, cut-in is `exo_cutin`.

⛔ NON-REACTIVE CAVEAT (carried into the paper). The agents' futures are LOGGED: they do not react to a counterfactual ego. J(a) is
an open-loop, non-reactive cost (the nuPlan / NAVSIM open-loop caveat) — a target SHAPE for Tier 0 / Tier 2, not a simulator.
It is pessimistic where real agents would yield (a candidate that brakes late "hits" a follower that would have braked too) and
optimistic where they would not. The Tier-1 descriptors are not affected: under the non-reactive assumption of the paper's
Proposition (ident) the exogenous future W does not depend on the candidate. Labels may use ego and agent futures (PI rule
2026-08-03); NOTHING produced here is an inference input.

Ego footprint constants are ESTIMATED (the fleet vehicle's dimensions are not in the corpus): 4.9 m x 1.95 m, box centre 1.45 m
ahead of the rear axle. Cost weights / thresholds are HYPOTHESIS constants, named below so an ablation can move them.
"""
from __future__ import annotations
import math
import numpy as np

# ------------------------------------------------------------------ constants (kept equal to imagination.py; a test asserts it)
H_T = (0.5, 1.0, 2.0, 3.0)
MODE_ACC = (-3.0, 0.0, 1.5, -6.0)
MODE_NAMES = ("brake", "hold", "accelerate", "brake_hard")           # exo_mode (EHead / tier1_cost index space)
MODE4_NAMES = ("cruise", "brake", "accelerate", "cut_out")            # exo_mode4 (spec §3.1 space)
IGNORE = -100
COST_CHANNELS = ("ttc", "clearance", "progress_deficit")             # HiCAPConfig.n_cost = 3, same order
FAMILIES = ("safety", "comfort", "progress", "lateral")              # imagination.FAMILIES (Tier 2), same order
CF_MASK = (1.0, 1.0, 1.0, 0.0)                                       # lateral has no agent-based counterfactual (no lanes) -> G2 refuses it

CLASSES = ("automobile", "person", "heavy_truck", "rider", "trailer", "bus", "protruding_object", "other_vehicle", "stroller",
           "animal", "train_or_tram_car")
VEHICLE_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "train_or_tram_car")
LEAD_CLASSES = VEHICLE_CLASSES + ("rider",)

EGO_LEN, EGO_WID, EGO_REAR_TO_CENTER = 4.9, 1.95, 1.45              # ESTIMATED
CORRIDOR_HALF_W = 1.25                    # m: ego half-width 0.975 + 0.275 margin, tested against the agent FOOTPRINT (not its centre)
LEAD_MAX_S = 80.0                         # m along path
LEAD_MAX_DYAW = math.radians(60.0)        # a lead moves with the ego's path ...
LEAD_STATIC_V = 1.0                       # ... unless it is (near-)stationary, then any heading blocks
PATH_LOOKAHEAD_S, PATH_S_MAX, PATH_DS, PATH_BACK = 30.0, 150.0, 1.0, 10.0
MAX_TRACK_GAP_S = 0.35                    # resampling: never interpolate a track across a hole longer than this
SG_HALF = 5                               # local quadratic fit over +-5 grid samples (+-0.5 s)
MODE_WIN_S, MODE_SUSTAIN_S = 3.0, 1.0     # mode window and the moving average that defines "sustained"
MIN_MODE_COV_S = 1.5                      # lead observed at least this long in the window, else mode IGNORE
CUT_OUT_MIN_S, CUT_IN_MIN_S = 0.5, 0.3
SIG_FLOOR = 0.005                         # m, floor of the jitter sigma before the log (MEASURED vehicle jitter p10 5 mm, p50 17 mm)
TTC_THR, CLEAR_THR = 3.0, 2.0             # s, m: channel costs saturate at 1 for a collision / contact
J_WEIGHTS = (4.0, 2.0, 1.0)               # scalar J = w . (ttc, clearance, progress deficit)  (HYPOTHESIS)
T_STEPS = 60                              # candidate horizon (vocab.T_STEPS), steps on the v9 row grid
MIN_J_COV_S = 3.0                         # J_w = 1 only if agent labels cover at least this much of the horizon
ONE_SIDED = True                          # ignore agents BEHIND the ego's rear axle (candidate frame): a non-reactive logged follower
                                          # would 'rear-end' every braking candidate (MEASURED: 32 % of rows for a -3 m/s^2 probe)
N_PROBE = 8


# ------------------------------------------------------------------ small geometry
def wrap(a): return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def quat_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def to_frame(x, y, x0, y0, yaw0):
    """World (x, y) -> frame at (x0, y0, yaw0); +x forward, +y left. Broadcasts."""
    dx, dy = x - x0, y - y0; c, s = np.cos(yaw0), np.sin(yaw0)
    return c * dx + s * dy, -s * dx + c * dy


def box_corners(cx, cy, yaw, length, width):
    """Corners [..., 4, 2] of oriented boxes (centre, heading, length along heading, width)."""
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    lx = np.array([0.5, 0.5, -0.5, -0.5]) * np.asarray(length)[..., None]
    ly = np.array([0.5, -0.5, -0.5, 0.5]) * np.asarray(width)[..., None]
    return np.stack([cx[..., None] + c * lx - s * ly, cy[..., None] + s * lx + c * ly], -1)


def _proj(corners, ax):
    """Projection interval of corners [...,4,2] on unit axes ax [...,2] -> (min, max) [...]."""
    p = (corners * ax[..., None, :]).sum(-1); return p.min(-1), p.max(-1)


def obb_overlap(ca, ya, cb, yb):
    """Separating-axis test for 2-D oriented boxes. ca/cb corners [...,4,2], ya/yb headings [...]. True where they overlap."""
    hit = np.ones(np.broadcast_shapes(ca.shape[:-2], cb.shape[:-2]), bool)
    for yaw in (ya, ya + np.pi / 2, yb, yb + np.pi / 2):
        ax = np.stack([np.cos(yaw), np.sin(yaw)], -1)
        a0, a1 = _proj(ca, ax); b0, b1 = _proj(cb, ax)
        hit &= (a1 >= b0) & (b1 >= a0)
    return hit


def _pt_box_dist(p, c, yaw, hl, hw):
    """Distance of points p [...,4,2] to boxes (centre c [...,2], yaw, half sizes) -> [...,4] (0 inside)."""
    d = p - c[..., None, :]; co, si = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    u = np.abs(co * d[..., 0] + si * d[..., 1]) - hl[..., None]; v = np.abs(-si * d[..., 0] + co * d[..., 1]) - hw[..., None]
    return np.hypot(np.maximum(u, 0), np.maximum(v, 0))


def obb_distance(ca, cen_a, ya, la, wa, cb, cen_b, yb, lb, wb):
    """Exact distance between disjoint convex boxes = min over the 8 corner-to-other-box distances (the closest pair of two disjoint
    convex polygons always involves a vertex); 0 where they overlap."""
    shp = np.broadcast_shapes(ca.shape[:-2], cb.shape[:-2])
    ca, cb = np.broadcast_to(ca, shp + (4, 2)), np.broadcast_to(cb, shp + (4, 2))
    da = _pt_box_dist(ca, np.broadcast_to(cen_b, shp + (2,)), np.broadcast_to(yb, shp), np.broadcast_to(lb / 2, shp), np.broadcast_to(wb / 2, shp))
    db = _pt_box_dist(cb, np.broadcast_to(cen_a, shp + (2,)), np.broadcast_to(ya, shp), np.broadcast_to(la / 2, shp), np.broadcast_to(wa / 2, shp))
    d = np.minimum(da.min(-1), db.min(-1))
    return np.where(obb_overlap(ca, ya, cb, yb), 0.0, d)


# ------------------------------------------------------------------ ego track and the v9 clock
class EgoTrack:
    """Egomotion in the clip-local world frame. t in seconds of the EGOMOTION clock (timestamp / 1e6)."""
    def __init__(self, t, x, y, yaw, v):
        o = np.argsort(t)
        self.t, self.x, self.y = np.asarray(t, float)[o], np.asarray(x, float)[o], np.asarray(y, float)[o]
        self.yaw, self.v = np.unwrap(np.asarray(yaw, float)[o]), np.asarray(v, float)[o]

    @classmethod
    def from_columns(cls, d):
        """d: mapping with the egomotion parquet columns (timestamp [us], x, y, qx..qw, vx, vy)."""
        return cls(np.asarray(d["timestamp"], float) / 1e6, d["x"], d["y"], quat_yaw(*(np.asarray(d[k], float) for k in ("qx", "qy", "qz", "qw"))),
                   np.hypot(np.asarray(d["vx"], float), np.asarray(d["vy"], float)))

    def at(self, tq):
        tq = np.asarray(tq, float)
        return np.interp(tq, self.t, self.x), np.interp(tq, self.t, self.y), np.interp(tq, self.t, self.yaw), np.interp(tq, self.t, self.v)


def clip_clock(ego: EgoTrack, grid_start_s: float, dt_s: float, n_grid: int) -> np.ndarray:
    """Egomotion-clock time of raw row k = 0..n_grid-1: now_s(k) + t_ego[0] (see module docstring)."""
    return grid_start_s + np.arange(n_grid) * dt_s + ego.t[0]


# ------------------------------------------------------------------ agents on the row grid
def local_poly(y, w, dt, half=SG_HALF):
    """Weighted local quadratic fit on a uniform grid (Savitzky-Golay with holes). y, w [A,G] -> (pos, vel, acc, rss, n) [A,G]."""
    A, G = y.shape; yz = np.where(w > 0, y, 0.0); wz = (w > 0).astype(float)
    S = np.zeros((5, A, G)); T = np.zeros((3, A, G)); Y2 = np.zeros((A, G))
    yp = np.pad(yz, ((0, 0), (half, half))); wp = np.pad(wz, ((0, 0), (half, half)))
    for o in range(-half, half + 1):
        ww = wp[:, half + o:half + o + G]; yy = yp[:, half + o:half + o + G]; tau = o * dt
        for p in range(5): S[p] += ww * tau ** p
        for p in range(3): T[p] += ww * yy * tau ** p
        Y2 += ww * yy * yy
    M = np.stack([np.stack([S[p + q] for q in range(3)], -1) for p in range(3)], -2)       # [A,G,3,3]
    n = S[0]; ok = n >= 4
    M = np.where(ok[..., None, None], M, np.eye(3)); rhs = np.where(ok[..., None], np.moveaxis(T, 0, -1), 0.0)
    c = np.linalg.solve(M + 1e-9 * np.eye(3), rhs[..., None])[..., 0]
    rss = np.maximum(Y2 - 2 * (c * np.moveaxis(T, 0, -1)).sum(-1) + np.einsum("...i,...ij,...j->...", c, M, c), 0.0)
    nan = ~ok | (w <= 0)
    f = lambda a: np.where(nan, np.nan, a)
    return f(c[..., 0]), f(c[..., 1]), f(2 * c[..., 2]), f(rss / np.maximum(n - 3, 1)), n


class AgentTracks:
    """Agent tracks resampled onto the clip's raw-row grid, in the WORLD frame. Arrays [A, G]; sizes / class [A]."""
    def __init__(self, x, y, yaw, valid, vx, vy, ax, ay, sig, length, width, cls, track_id):
        self.x, self.y, self.yaw, self.valid = x, y, yaw, valid
        self.vx, self.vy, self.ax, self.ay, self.sig = vx, vy, ax, ay, sig
        self.length, self.width, self.cls, self.track_id = length, width, cls, track_id

    @property
    def A(self): return len(self.length)

    @classmethod
    def empty(cls, G):
        z = np.zeros((0, G)); return cls(z, z, z, z.astype(bool), z, z, z, z, z, np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0, object))


def agents_on_grid(obs: dict, ego: EgoTrack, t_grid: np.ndarray, max_gap_s: float = MAX_TRACK_GAP_S) -> AgentTracks:
    """obs: mapping of obstacle.offline columns for ONE clip. Each box (rig frame at its own timestamp) is placed in the world with the
    ego pose interpolated at that timestamp, then every track is linearly resampled onto t_grid (never across a hole > max_gap_s,
    never extrapolated beyond 0.05 s of its first / last sample) and smoothed by a local quadratic fit (vel / acc / jitter)."""
    G = len(t_grid)
    if obs is None or len(obs["timestamp_us"]) == 0: return AgentTracks.empty(G)
    ts = np.asarray(obs["timestamp_us"], float) / 1e6
    ex, ey, eyaw, _ = ego.at(ts)
    cx, cy = np.asarray(obs["center_x"], float), np.asarray(obs["center_y"], float)
    wx = ex + np.cos(eyaw) * cx - np.sin(eyaw) * cy; wy = ey + np.sin(eyaw) * cx + np.cos(eyaw) * cy
    wyaw = eyaw + quat_yaw(*(np.asarray(obs[k], float) for k in ("orientation_x", "orientation_y", "orientation_z", "orientation_w")))
    tid = np.asarray(obs["track_id"]).astype(str); lab = np.asarray(obs["label_class"]).astype(str)
    sx, sy = np.asarray(obs["size_x"], float), np.asarray(obs["size_y"], float)
    order = np.lexsort((ts, tid)); tid_s = tid[order]
    starts = np.flatnonzero(np.r_[True, tid_s[1:] != tid_s[:-1]]); ends = np.r_[starts[1:], len(order)]
    A = len(starts); X = np.full((A, G), np.nan); Y = X.copy(); H = X.copy(); Vd = np.zeros((A, G), bool)
    Ln, Wd, C, ID = np.zeros(A), np.zeros(A), np.zeros(A, int), np.empty(A, object)
    cls_id = {c: i for i, c in enumerate(CLASSES)}
    for a, (s0, s1) in enumerate(zip(starts, ends)):
        ix = order[s0:s1]; t = ts[ix]
        keep = np.r_[True, np.diff(t) > 1e-6]; ix, t = ix[keep], t[keep]
        Ln[a], Wd[a] = np.median(sx[ix]), np.median(sy[ix]); C[a] = cls_id.get(lab[ix[0]], -1); ID[a] = tid_s[s0]
        g0, g1 = np.searchsorted(t_grid, t[0] - 0.05), np.searchsorted(t_grid, t[-1] + 0.05)
        if g1 <= g0: continue
        tq = t_grid[g0:g1]
        if len(t) == 1:
            okq = np.abs(tq - t[0]) <= 0.05
        else:
            j = np.clip(np.searchsorted(t, tq), 1, len(t) - 1); okq = (t[j] - t[j - 1]) <= max_gap_s
        X[a, g0:g1] = np.interp(tq, t, wx[ix]); Y[a, g0:g1] = np.interp(tq, t, wy[ix]); H[a, g0:g1] = np.interp(tq, t, np.unwrap(wyaw[ix]))
        Vd[a, g0:g1] = okq
    dt = float(np.median(np.diff(t_grid))) if G > 1 else 0.1
    px, vx, ax, rx, _ = local_poly(X, Vd, dt); py, vy, ay, ry, _ = local_poly(Y, Vd, dt)
    sig = np.sqrt(rx + ry)
    # single-sample / very short tracks: keep the raw position, zero motion (a near-static box is the safe reading)
    short = Vd & np.isnan(px)
    px = np.where(short, X, px); py = np.where(short, Y, py)
    for arr in (vx, vy, ax, ay): arr[short] = 0.0
    sig[short] = np.nan
    return AgentTracks(np.where(Vd, px, np.nan), np.where(Vd, py, np.nan), np.where(Vd, H, np.nan), Vd, vx, vy, ax, ay, sig, Ln, Wd, C, ID)


# ------------------------------------------------------------------ the logged path and Frenet coordinates
def path_polyline(pts: np.ndarray):
    """pts [N,2] logged ego positions from NOW in the NOW frame -> (P [M,2], s [M], u [M,2] unit tangents) on a 1 m arc grid from
    -PATH_BACK (straight behind) to PATH_S_MAX, extended straight along the last 2 m chord beyond the end of the log (or along +x when
    the ego barely moves)."""
    pts = np.vstack([[0.0, 0.0], pts])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); s = np.r_[0.0, np.cumsum(seg)]
    keep = np.r_[True, seg > 1e-3]; pts, s = pts[keep], s[keep]
    sg = np.arange(0.0, PATH_S_MAX + PATH_DS / 2, PATH_DS)
    if s[-1] < 2.0:
        P = np.stack([sg, np.zeros_like(sg)], 1)
    else:
        x = np.interp(sg, s, pts[:, 0]); y = np.interp(sg, s, pts[:, 1])
        over = sg > s[-1]
        if over.any():
            q = np.array([np.interp(s[-1] - 2.0, s, pts[:, 0]), np.interp(s[-1] - 2.0, s, pts[:, 1])])
            h = pts[-1] - q; h = h / max(np.linalg.norm(h), 1e-9)
            x[over] = pts[-1, 0] + (sg[over] - s[-1]) * h[0]; y[over] = pts[-1, 1] + (sg[over] - s[-1]) * h[1]
        P = np.stack([x, y], 1)
    back = np.arange(-PATH_BACK, 0.0, PATH_DS)
    P = np.vstack([np.stack([back, np.zeros_like(back)], 1), P]); sP = np.r_[back, sg]
    u = np.gradient(P, axis=0); u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return P, sP, u


def frenet(P, sP, u, q):
    """Points q [...,2] -> (s, d) along the polyline; d > 0 = LEFT of the path."""
    sh = q.shape[:-1]; f = q.reshape(-1, 2)
    if f.shape[0] == 0: return np.zeros(sh), np.zeros(sh)
    j = np.argmin(((f[:, None, :] - P[None]) ** 2).sum(-1), 1)
    r = f - P[j]; uj = u[j]
    t = np.clip((r * uj).sum(-1), -PATH_DS, PATH_DS)
    t = np.where((j == len(P) - 1) & ((r * uj).sum(-1) > 0), (r * uj).sum(-1), t)          # beyond the far end: keep going
    d = uj[:, 0] * r[:, 1] - uj[:, 1] * r[:, 0]
    return (sP[j] + t).reshape(sh), d.reshape(sh)


def _rel(ag: AgentTracks, ks, x0, y0, yaw0, idx=slice(None)):
    """States of agents `idx` at grid indices ks in the frame (x0,y0,yaw0): centre x/y, yaw, velocity, acceleration (rotated), valid."""
    g = lambda a: a[idx][:, ks]
    x, y = to_frame(g(ag.x), g(ag.y), x0, y0, yaw0)
    c, s = np.cos(yaw0), np.sin(yaw0)
    rot = lambda a, b: (c * a + s * b, -s * a + c * b)
    vx, vy = rot(g(ag.vx), g(ag.vy)); ax, ay = rot(g(ag.ax), g(ag.ay))
    return x, y, g(ag.yaw) - yaw0, vx, vy, ax, ay, g(ag.valid)


def corridor_overlap(d_corners, half_w=CORRIDOR_HALF_W):
    """Box footprint (lateral interval of its corners) intersects [-half_w, half_w]."""
    return (d_corners.min(-1) <= half_w) & (d_corners.max(-1) >= -half_w)


# ------------------------------------------------------------------ Tier 1
def _ma(a, n):
    """Centred moving average over n samples ignoring NaN; NaN where no sample."""
    if n <= 1: return a
    v = np.isfinite(a); z = np.where(v, a, 0.0); k = np.ones(n)
    num = np.convolve(z, k, "same"); den = np.convolve(v.astype(float), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def lead_row(ag: AgentTracks, k: int, ego_now, path, dt: float, cand_idx: np.ndarray | None = None):
    """Identify the lead at raw row k. Returns (index into ag or -1, info dict). ego_now = (x0, y0, yaw0, v0)."""
    x0, y0, yaw0, v0 = ego_now; P, sP, u = path
    idx = np.flatnonzero(ag.valid[:, k]) if cand_idx is None else cand_idx[ag.valid[cand_idx, k]]
    idx = idx[np.isin(ag.cls[idx], [CLASSES.index(c) for c in LEAD_CLASSES])]
    if idx.size == 0: return -1, {}
    x, y = to_frame(ag.x[idx, k], ag.y[idx, k], x0, y0, yaw0)
    near = (np.hypot(x, y) < LEAD_MAX_S + PATH_BACK + 10) & (x > -PATH_BACK)
    idx, x, y = idx[near], x[near], y[near]
    if idx.size == 0: return -1, {}
    yaw = ag.yaw[idx, k] - yaw0
    cs, cd = frenet(P, sP, u, box_corners(x, y, yaw, ag.length[idx], ag.width[idx]))
    s_rear = cs.min(-1); inc = corridor_overlap(cd)
    sc, _ = frenet(P, sP, u, np.stack([x, y], -1))
    jj = np.clip(np.searchsorted(sP, sc), 0, len(sP) - 1); hp = np.arctan2(u[jj, 1], u[jj, 0])
    spd = np.hypot(ag.vx[idx, k], ag.vy[idx, k]); spd = np.where(np.isfinite(spd), spd, 0.0)
    aligned = (np.abs(wrap(yaw - hp)) <= LEAD_MAX_DYAW) | (spd < LEAD_STATIC_V)
    ok = inc & aligned & (s_rear > 0) & (s_rear <= LEAD_MAX_S)
    if not ok.any(): return -1, {}
    b = np.flatnonzero(ok)[np.argmin(s_rear[ok])]
    return int(idx[b]), dict(gap0=float(s_rear[b]), lat0=float(cd[b].mean()), cls=int(ag.cls[idx[b]]))


def tier1_targets(ag: AgentTracks, a: int, k: int, ego_now, path, dt: float, others: np.ndarray | None = None, s_ego: np.ndarray | None = None):
    """Tier-1 descriptors of lead a from raw row k (see the module docstring for the exact semantics)."""
    x0, y0, yaw0, v0 = ego_now; P, sP, u = path
    G = ag.x.shape[1]; I = int(math.ceil(MODE_WIN_S / dt)) + 1
    ks = np.arange(k, min(k + I, G)); n = len(ks); tau = (ks - k) * dt
    x, y, yaw, vx, vy, axx, ayy, val = (z[0] for z in _rel(ag, ks, x0, y0, yaw0, [a]))
    corners = box_corners(np.nan_to_num(x), np.nan_to_num(y), np.nan_to_num(yaw), ag.length[a], ag.width[a])
    cs, cd = frenet(P, sP, u, corners)
    L = np.where(val, cs.min(-1), np.nan)
    sc, _ = frenet(P, sP, u, np.stack([np.nan_to_num(x), np.nan_to_num(y)], -1))
    jj = np.clip(np.searchsorted(sP, sc), 0, len(sP) - 1); ut = u[jj]
    Ld = np.where(val, vx * ut[:, 0] + vy * ut[:, 1], np.nan); Ldd = np.where(val, axx * ut[:, 0] + ayy * ut[:, 1], np.nan)
    inc = np.where(val, corridor_overlap(cd), True)
    sig = ag.sig[a, ks]
    out = dict(exo_gap=np.zeros(4), exo_close=np.zeros(4), exo_acc=np.zeros(4), exo_logsig=np.zeros(4), exo_lead_s=np.zeros(4), exo_hmask=np.zeros(4))
    fv = np.flatnonzero(val & np.isfinite(Ld))
    if fv.size == 0: return None
    for h_i, h in enumerate(H_T):
        if fv[0] * dt - 1e-6 <= h <= fv[-1] * dt + 1e-6 and n > 1 and val[min(int(round(h / dt)), n - 1)]:
            Lh, Ldh, Lddh = (np.interp(h, tau[fv], z[fv]) for z in (L, Ld, Ldd)); out["exo_hmask"][h_i] = 1.0
        else:                                             # outside the observed span: constant-velocity extrapolation from the last state
            j = fv[-1] if fv[-1] * dt < h else fv[0]
            Lh, Ldh, Lddh = L[j] + Ld[j] * (h - tau[j]), Ld[j], 0.0
        ss = sig[np.abs(tau - h) <= SG_HALF * dt + 1e-9]; ss = ss[np.isfinite(ss)]
        out["exo_lead_s"][h_i] = Lh; out["exo_gap"][h_i] = Lh - v0 * h; out["exo_close"][h_i] = v0 - Ldh; out["exo_acc"][h_i] = Lddh
        out["exo_logsig"][h_i] = math.log(max(float(np.sqrt(np.mean(ss ** 2))) if ss.size else SIG_FLOOR, SIG_FLOOR))
    # ---- mode / onset over [0, MODE_WIN_S]
    cov = val.sum() * dt
    am = _ma(np.where(val, Ldd, np.nan), 2 * int(round(MODE_SUSTAIN_S / dt / 2)) + 1)      # odd window: centred
    out_run = (~inc) & val
    run = 0; t_out = None
    for i in range(n):
        run = run + 1 if out_run[i] else 0
        if run * dt >= CUT_OUT_MIN_S - 1e-9: t_out = float(tau[i - run + 1]); break
    mode = IGNORE; mode4 = IGNORE; onset = MODE_WIN_S
    if t_out is not None:
        mode4, onset = 3, t_out
    elif cov >= MIN_MODE_COV_S and np.isfinite(am).any():
        j = int(np.nanargmax(np.abs(am))); a_ext = float(am[j])
        mode = int(np.argmin(np.abs(np.array(MODE_ACC) - a_ext)))
        mode4 = {0: 1, 3: 1, 1: 0, 2: 2}[mode]
        if mode != 1:
            thr = 0.5 * (MODE_ACC[0] + MODE_ACC[1]) if a_ext < 0 else 0.5 * (MODE_ACC[2] + MODE_ACC[1])
            hit = np.flatnonzero((am <= thr) if a_ext < 0 else (am >= thr))
            onset = float(tau[hit[0]]) if hit.size else float(tau[j])
    # onset band: times at which 10 / 50 / 90 % of the cumulative |speed change| over the window is reached
    v = np.where(val, Ld, np.nan); dv = np.abs(np.diff(v)); dv = np.where(np.isfinite(dv), dv, 0.0); cum = np.r_[0.0, np.cumsum(dv)]
    q = np.array([np.interp(f * cum[-1], cum, tau) for f in (0.1, 0.5, 0.9)]) if cum[-1] > 1e-6 else np.full(3, MODE_WIN_S)
    out.update(exo_mode=mode, exo_mode4=mode4, exo_onset=onset, exo_onset_q=np.maximum.accumulate(q), exo_cutout=int(t_out is not None),
               lead_cov_s=float(cov))
    # ---- cut-in: another vehicle, valid and OUTSIDE the corridor at NOW, that enters it ahead of the ego (and ahead of the lead)
    out["exo_cutin"], out["exo_cutin_t"] = 0, np.nan
    if others is not None and others.size:
        o = others[(others != a) & ag.valid[others, k]]
        t_in = cut_in_time(ag, o, k, ego_now, path, dt, lim=np.minimum(np.where(np.isfinite(L), L, LEAD_MAX_S), LEAD_MAX_S), s_ego=s_ego)
        if t_in is not None: out["exo_cutin"], out["exo_cutin_t"] = 1, t_in
    return out


# ------------------------------------------------------------------ teacher cost J(a)
def _cand_headings(c):
    """Heading of a candidate polyline at each step (0 at NOW; held while it does not move)."""
    d = np.diff(np.concatenate([np.zeros(c.shape[:-2] + (1, 2)), c], -2), axis=-2)
    moving = np.linalg.norm(d, axis=-1) > 0.05
    h = np.where(moving, np.arctan2(d[..., 1], d[..., 0]), np.nan)
    out = np.empty(h.shape); last = np.zeros(h.shape[:-1])
    for i in range(h.shape[-1]):
        last = np.where(np.isnan(h[..., i]), last, h[..., i]); out[..., i] = last
    return out


def comfort_cost(c, v0, dt):
    """numpy port of the comfort term of `imagination.teacher_costs` (|a| > 3 m/s^2, |jerk| > 5 m/s^3 penalised)."""
    v = np.linalg.norm(np.diff(np.concatenate([np.zeros(c.shape[:-2] + (1, 2)), c], -2), axis=-2), axis=-1) / dt
    a = np.diff(np.concatenate([np.full(v.shape[:-1] + (1,), v0), v], -1), axis=-1) / dt; j = np.diff(a, axis=-1) / dt
    return (np.maximum(np.abs(a) - 3.0, 0) ** 2).mean(-1) + 0.01 * (np.maximum(np.abs(j) - 5.0, 0) ** 2).mean(-1)


def candidate_costs(cands: np.ndarray, agents: dict | None, v0: float, fut_log: np.ndarray, dt: float, cov_steps: int | None = None):
    """Teacher cost of K candidate trajectories [K,T,2] (NOW frame, step i at time (i+1) dt) against the agents' LOGGED futures.
    agents: x, y, yaw [A,T] (same frame, same steps), length, width [A], valid [A,T] (or None = empty road).
    Returns dict: ttc_s [K] (first swept-box overlap time; inf = none), clear_m [K] (min box-box distance), collide [K],
    cost3 [K,3] = (relu(TTC_THR - ttc)/TTC_THR, relu(CLEAR_THR - clear)/CLEAR_THR, relu(s_log - s_cand)/max(s_log,1)),
    J [K] = J_WEIGHTS . cost3, cf [K,4] Tier-2 families (safety, comfort, progress, lateral=0).
    Non-reactive: the agents follow their logged futures whatever the candidate does (module docstring). ONE_SIDED: only agents
    whose centre is ahead of the candidate's rear axle at that step count (the 'one-sided form' of the paper; nuPlan's not-at-fault
    rear collision), so a logged follower cannot penalise braking."""
    cands = np.asarray(cands, float); K, T = cands.shape[:2]
    cov = T if cov_steps is None else int(np.clip(cov_steps, 0, T))
    yaw = _cand_headings(cands)
    cen = cands + EGO_REAR_TO_CENTER * np.stack([np.cos(yaw), np.sin(yaw)], -1)
    ce = box_corners(cen[..., 0], cen[..., 1], yaw, EGO_LEN, EGO_WID)                          # [K,T,4,2]
    ttc = np.full(K, np.inf); clear = np.full(K, np.inf)
    if agents is not None and len(agents["length"]) and cov > 0:
        ax_, ay_, ayaw, val = (np.asarray(agents[k])[:, :cov] for k in ("x", "y", "yaw", "valid"))
        L_, W_ = np.asarray(agents["length"], float), np.asarray(agents["width"], float)
        # coarse prefilter: centre distance to any candidate point at the same step within reach
        reach = np.hypot(L_, W_)[:, None] / 2 + math.hypot(EGO_LEN, EGO_WID) / 2 + CLEAR_THR + 1.0
        dmin = np.full(len(L_), np.inf)
        for kk in range(0, K, 64):
            dd = np.hypot(cen[kk:kk + 64, None, :cov, 0] - ax_[None], cen[kk:kk + 64, None, :cov, 1] - ay_[None])
            dmin = np.minimum(dmin, np.where(val[None], dd - reach[None], np.inf).min((0, 2)))
        keep = np.flatnonzero(dmin <= 0)
        if keep.size:
            ax_, ay_, ayaw, val, L_, W_ = ax_[keep], ay_[keep], ayaw[keep], val[keep], L_[keep], W_[keep]
            ca = box_corners(np.nan_to_num(ax_), np.nan_to_num(ay_), np.nan_to_num(ayaw), L_[:, None], W_[:, None])   # [A,cov,4,2]
            ce_, yw = ce[:, None, :cov], yaw[:, None, :cov]                                                          # [K,1,cov,...]
            val = np.broadcast_to(val[None], (K,) + val.shape)
            ov = obb_overlap(ce_, yw, ca[None], np.nan_to_num(ayaw)[None]) & val                                      # [K,A,cov]
            if ONE_SIDED:   # not-at-fault filter: drop agents BEHIND the candidate's rear axle at each step, and drop an agent entirely
                            # for that candidate when its FIRST contact comes from behind (a follower that keeps going penetrates the ego)
                rel = (np.nan_to_num(ax_)[None] - cands[:, None, :cov, 0]) * np.cos(yw) + (np.nan_to_num(ay_)[None] - cands[:, None, :cov, 1]) * np.sin(yw)
                first = np.argmax(ov, -1)
                rear = ov.any(-1) & (np.take_along_axis(rel, first[..., None], -1)[..., 0] <= 0.0)                       # [K,A]
                val = val & (rel > 0.0) & ~rear[..., None]
                ov = ov & val
            dist = obb_distance(ce_, cen[:, None, :cov], yw, EGO_LEN, EGO_WID, ca[None], np.stack([np.nan_to_num(ax_), np.nan_to_num(ay_)], -1)[None],
                                np.nan_to_num(ayaw)[None], L_[None, :, None], W_[None, :, None])
            dist = np.where(val, dist, np.inf)
            clear = dist.min((1, 2))
            hk = ov.any(1)                                                                                            # [K,cov]
            hit_t = np.where(hk.any(-1), np.argmax(hk, axis=-1), -1)
            ttc = np.where(hit_t >= 0, (hit_t + 1) * dt, np.inf)
    s_log = float(np.linalg.norm(np.diff(np.vstack([[0, 0], fut_log]), axis=0), axis=1).sum())
    s_c = np.linalg.norm(np.diff(np.concatenate([np.zeros((K, 1, 2)), cands], 1), axis=1), axis=-1).sum(1)
    c3 = np.stack([np.maximum(TTC_THR - ttc, 0) / TTC_THR, np.maximum(CLEAR_THR - clear, 0) / CLEAR_THR,
                   np.maximum(s_log - s_c, 0) / max(s_log, 1.0)], -1)
    c3 = np.nan_to_num(c3, posinf=0.0)
    J = c3 @ np.array(J_WEIGHTS)
    cf = np.stack([0.5 * (c3[:, 0] + c3[:, 1]), comfort_cost(cands, v0, dt), c3[:, 2], np.zeros(K)], -1)
    return dict(ttc_s=ttc, clear_m=clear, collide=np.isfinite(ttc), cost3=c3.astype(np.float32), J=J.astype(np.float32), cf=cf.astype(np.float32))


def tier0_soft_target(J, star, T: float = 1.0, mix: float = 0.7, valid=None):
    """numpy twin of `imagination.tier0_target`: y = mix softmax(-J/T) + (1-mix) onehot(star). J [B,K], star [B]."""
    z = -np.asarray(J, float) / T
    if valid is not None: z = np.where(valid, z, -1e9)
    z = z - z.max(-1, keepdims=True); e = np.exp(z); p = e / e.sum(-1, keepdims=True)
    return (mix * p + (1 - mix) * np.eye(z.shape[-1])[np.asarray(star)]).astype(np.float32)


def probe_candidates(fut: np.ndarray, v0: float, dt: float) -> np.ndarray:
    """A fixed, vocabulary-free probe set [N_PROBE,T,2] for verification: 0 logged, 1 logged path at constant v0, 2 brake -3 m/s^2,
    3 brake -6 m/s^2, 4 accelerate +1.5 m/s^2, 5/6 logged trajectory shifted +3.5 m / -3.5 m laterally (ramp over 3 s), 7 straight at v0."""
    T = len(fut); t = (np.arange(T) + 1) * dt
    pts = np.vstack([[0.0, 0.0], fut]); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); s = np.r_[0.0, np.cumsum(seg)]
    def along(sq):
        if s[-1] < 0.5: return np.stack([sq, np.zeros_like(sq)], -1)
        ext = sq > s[-1]; h = pts[-1] - pts[max(len(pts) - 3, 0)]; h = h / max(np.linalg.norm(h), 1e-9)
        x = np.interp(sq, s, pts[:, 0]) + np.where(ext, (sq - s[-1]) * h[0], 0); y = np.interp(sq, s, pts[:, 1]) + np.where(ext, (sq - s[-1]) * h[1], 0)
        return np.stack([x, y], -1)
    def prof(a): return np.cumsum(np.clip(v0 + a * t, 0, 30.0) * dt)
    yaw = _cand_headings(fut[None])[0]; nrm = np.stack([-np.sin(yaw), np.cos(yaw)], -1); ramp = np.clip(t / 3.0, 0, 1)[:, None]
    return np.stack([fut, along(prof(0.0)), along(prof(-3.0)), along(prof(-6.0)), along(prof(1.5)), fut + 3.5 * ramp * nrm,
                     fut - 3.5 * ramp * nrm, np.stack([v0 * t, np.zeros(T)], -1)]).astype(np.float32)


# ------------------------------------------------------------------ per clip
def build_rows(ego: EgoTrack, obs: dict | None, grid_start_s: float, dt_s: float, k0: int, n_rows: int, vocab=None, n_cand: int = 5,
               probe: bool = True, obs_span: tuple | None = None) -> dict:
    """All supervision for one clip, one entry per v9 row j (raw row k = k0 + j). Keys:
      exo_gap/exo_close/exo_acc/exo_logsig/exo_lead_s/exo_hmask [n,4] f32, exo_mode/exo_mode4 [n] i64 (-100 IGNORE), exo_onset [n] f32,
      exo_onset_q [n,3], exo_has_lead/exo_w [n] f32, exo_cutin/exo_cutout [n] i8, exo_cutin_t [n], lead_gap0/lead_tg0/lead_lat0/
      lead_cov_s [n], lead_cls [n] i8, lead_tid [n] (object; '' none);
      probe_cost3 [n,8,3], probe_J [n,8], probe_ttc [n,8], probe_clear [n,8] (vocabulary-free verification set, if probe);
      with a FactoredVocab: cand_p/cand_q [n,K] i64, star [n] i64, J_teacher [n,K], cost_t [n,K,3], cf_cost/cf_mask [n,K,4] (cf_mask
      is zeroed where J_w = 0),
      y_tier0 [n,K]; always J_w [n] f32 (1 = agent labels cover >= MIN_J_COV_S of the horizon), J_cov_s [n].
    exo_w = 1 only where agent labels cover [NOW, NOW + 3 s]: an unlabelled stretch is NO_LABEL, never "no lead"."""
    G = k0 + n_rows + T_STEPS + 2
    tg = clip_clock(ego, grid_start_s, dt_s, G)
    ag = agents_on_grid(obs, ego, tg) if obs is not None else AgentTracks.empty(G)
    if obs_span is None and obs is not None and len(obs["timestamp_us"]):
        t_obs = np.asarray(obs["timestamp_us"], float) / 1e6; obs_span = (float(t_obs.min()), float(t_obs.max()))
    ex, ey, eyaw, ev = ego.at(tg)
    n = n_rows; f4 = lambda: np.zeros((n, 4), np.float32)
    R = dict(exo_gap=f4(), exo_close=f4(), exo_acc=f4(), exo_logsig=f4(), exo_lead_s=f4(), exo_hmask=f4(),
             exo_mode=np.full(n, IGNORE, np.int64), exo_mode4=np.full(n, IGNORE, np.int64), exo_onset=np.full(n, MODE_WIN_S, np.float32),
             exo_onset_q=np.full((n, 3), MODE_WIN_S, np.float32), exo_has_lead=np.zeros(n, np.float32), exo_w=np.zeros(n, np.float32),
             exo_cutin=np.zeros(n, np.int8), exo_cutout=np.zeros(n, np.int8), exo_cutin_t=np.full(n, np.nan, np.float32),
             lead_gap0=np.full(n, np.nan, np.float32), lead_tg0=np.full(n, np.nan, np.float32), lead_lat0=np.full(n, np.nan, np.float32),
             lead_cov_s=np.zeros(n, np.float32), lead_cls=np.full(n, -1, np.int8), lead_tid=np.full(n, "", object),
             J_w=np.zeros(n, np.float32), J_cov_s=np.zeros(n, np.float32), v0=np.zeros(n, np.float32))
    if probe:
        R.update(probe_cost3=np.zeros((n, N_PROBE, 3), np.float32), probe_J=np.zeros((n, N_PROBE), np.float32),
                 probe_ttc=np.full((n, N_PROBE), np.inf, np.float32), probe_clear=np.full((n, N_PROBE), np.inf, np.float32))
    if vocab is not None:
        from .data import factor_distances                    # lazy: data.py imports torch
        from . import vocab as Vm
        K = n_cand
        R.update(cand_p=np.zeros((n, K), np.int64), cand_q=np.zeros((n, K), np.int64), star=np.zeros(n, np.int64),
                 J_teacher=np.zeros((n, K), np.float32), cost_t=np.zeros((n, K, 3), np.float32), cf_cost=np.zeros((n, K, 4), np.float32),
                 cf_mask=np.tile(np.array(CF_MASK, np.float32), (n, K, 1)), y_tier0=np.zeros((n, K), np.float32))
    veh = np.flatnonzero(np.isin(ag.cls, [CLASSES.index(c) for c in VEHICLE_CLASSES])) if ag.A else np.zeros(0, int)
    for j in range(n):
        k = k0 + j; x0, y0, yaw0, v0 = ex[k], ey[k], eyaw[k], ev[k]; R["v0"][j] = v0
        t_now = tg[k]
        lab_now = obs_span is not None and obs_span[0] - 0.05 <= t_now and t_now + MODE_WIN_S <= obs_span[1] + 0.05
        # logged path from NOW (30 s of egomotion) and the 6 s future on the row grid
        tp = t_now + np.arange(1, int(PATH_LOOKAHEAD_S / 0.1) + 1) * 0.1
        px, py, _, _ = ego.at(tp); path = path_polyline(np.stack(to_frame(px, py, x0, y0, yaw0), -1))
        fx, fy = to_frame(ex[k + 1:k + 1 + T_STEPS], ey[k + 1:k + 1 + T_STEPS], x0, y0, yaw0); fut = np.stack([fx, fy], -1)
        s_ego = np.cumsum(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], fut]), axis=0), axis=1))           # logged arc at steps 1..T
        if lab_now and ag.A:
            a, info = lead_row(ag, k, (x0, y0, yaw0, v0), path, dt_s)
            R["exo_w"][j] = 1.0
            if a >= 0:
                t1 = tier1_targets(ag, a, k, (x0, y0, yaw0, v0), path, dt_s, others=veh, s_ego=s_ego)
                if t1 is not None:
                    R["exo_has_lead"][j] = 1.0
                    for key in ("exo_gap", "exo_close", "exo_acc", "exo_logsig", "exo_lead_s", "exo_hmask", "exo_onset_q"): R[key][j] = t1[key]
                    for key in ("exo_mode", "exo_mode4", "exo_onset", "exo_cutin", "exo_cutout", "exo_cutin_t", "lead_cov_s"): R[key][j] = t1[key]
                    R["lead_gap0"][j] = info["gap0"]; R["lead_lat0"][j] = info["lat0"]; R["lead_cls"][j] = info["cls"]; R["lead_tid"][j] = str(ag.track_id[a])
                    R["lead_tg0"][j] = info["gap0"] / v0 if v0 > 0.5 else np.nan
            elif veh.size:                                 # no lead: a cut-in can still happen
                t_in = cut_in_time(ag, veh[ag.valid[veh, k]], k, (x0, y0, yaw0, v0), path, dt_s, s_ego=s_ego)
                if t_in is not None: R["exo_cutin"][j], R["exo_cutin_t"][j] = 1, t_in
        elif lab_now:
            R["exo_w"][j] = 1.0                            # labelled stretch with zero agents: a genuine NO_LEAD
        # ---- teacher costs on the row grid
        if obs_span is not None and obs_span[0] - 0.05 <= t_now <= obs_span[1]:
            cov_steps = int(np.clip(np.floor((obs_span[1] - t_now) / dt_s + 1e-9), 0, T_STEPS))
        else:
            cov_steps = 0
        R["J_cov_s"][j] = cov_steps * dt_s; R["J_w"][j] = float(cov_steps * dt_s >= MIN_J_COV_S - 1e-9)
        if not (probe or vocab is not None): continue
        agents = None
        if ag.A and cov_steps > 0:
            ks = slice(k + 1, k + 1 + T_STEPS)
            x, y, yaw, *_r, val = _rel(ag, ks, x0, y0, yaw0)
            # drop agents overlapping the ego AT NOW (label noise / self detections): they would make every candidate "collide"
            v_now = ag.valid[:, k]
            if v_now.any():
                xn, yn = to_frame(ag.x[:, k], ag.y[:, k], x0, y0, yaw0)
                ego_c = box_corners(np.array(EGO_REAR_TO_CENTER), np.array(0.0), np.array(0.0), EGO_LEN, EGO_WID)
                ovn = obb_overlap(ego_c[None], np.zeros(1), box_corners(np.nan_to_num(xn), np.nan_to_num(yn), np.nan_to_num(ag.yaw[:, k] - yaw0), ag.length, ag.width), np.nan_to_num(ag.yaw[:, k] - yaw0)) & v_now
            else:
                ovn = np.zeros(ag.A, bool)
            use = np.flatnonzero(val.any(1) & ~ovn)
            agents = dict(x=x[use], y=y[use], yaw=yaw[use], valid=val[use], length=ag.length[use], width=ag.width[use])
        if probe:
            pc = candidate_costs(probe_candidates(fut, v0, dt_s), agents, v0, fut, dt_s, cov_steps)
            R["probe_cost3"][j] = pc["cost3"]; R["probe_J"][j] = pc["J"]; R["probe_ttc"][j] = pc["ttc_s"]; R["probe_clear"][j] = pc["clear_m"]
        if vocab is not None:
            path_log, dv_log = Vm.extract_path_profile(fut, v0)
            dP, dQ = factor_distances(path_log, dv_log, v0, vocab.paths, vocab.profiles)
            cp, cq = np.argsort(dP, kind="stable")[:K], np.argsort(dQ, kind="stable")[:K]
            cands = np.stack([Vm.compose(vocab.paths[p], vocab.profiles[q], v0) for p, q in zip(cp, cq)])
            cc = candidate_costs(cands, agents, v0, fut, dt_s, cov_steps)
            R["cand_p"][j], R["cand_q"][j] = cp, cq; R["J_teacher"][j] = cc["J"]; R["cost_t"][j] = cc["cost3"]; R["cf_cost"][j] = cc["cf"]
            R["cf_mask"][j] *= R["J_w"][j]                    # G2: no Tier-2 supervision where agent labels do not cover the horizon
            R["y_tier0"][j] = tier0_soft_target(cc["J"][None], np.zeros(1, int))[0]      # star = 0: nearest path x nearest profile
    return R


def cut_in_time(ag: AgentTracks, o: np.ndarray, k: int, ego_now, path, dt: float, lim: np.ndarray | None = None, s_ego: np.ndarray | None = None):
    """Earliest time (s) at which a vehicle of `o` that is valid and LATERALLY outside the corridor at NOW overlaps the corridor ahead
    of the ego's front bumper at that time (s_ego(t) + front < rear-face arc < lim(t); s_ego = the ego's logged arc position, lim
    defaults to LEAD_MAX_S), heading within LEAD_MAX_DYAW of the path, for >= CUT_IN_MIN_S. A follower that drives through the ego's
    NOW position is not a cut-in (it is laterally inside the corridor at NOW)."""
    if o.size == 0: return None
    x0, y0, yaw0, v0 = ego_now; P, sP, u = path
    G = ag.x.shape[1]; ks = np.arange(k, min(k + int(math.ceil(MODE_WIN_S / dt)) + 1, G)); tau = (ks - k) * dt
    x, y, yaw, *_r, val = _rel(ag, ks, x0, y0, yaw0, o)
    near = np.where(val, np.hypot(x, y), np.inf).min(1) < LEAD_MAX_S + 10
    if not near.any(): return None
    x, y, yaw, val, o = x[near], y[near], yaw[near], val[near], o[near]
    cs, cd = frenet(P, sP, u, box_corners(np.nan_to_num(x), np.nan_to_num(y), np.nan_to_num(yaw), ag.length[o][:, None], ag.width[o][:, None]))
    lim = np.full(len(ks), LEAD_MAX_S) if lim is None else lim[:len(ks)]
    sc, _ = frenet(P, sP, u, np.stack([np.nan_to_num(x), np.nan_to_num(y)], -1))
    jj = np.clip(np.searchsorted(sP, sc), 0, len(sP) - 1)
    aligned = np.abs(wrap(np.nan_to_num(yaw) - np.arctan2(u[jj, 1], u[jj, 0]))) <= LEAD_MAX_DYAW         # crossing / oncoming traffic is not a cut-in
    front = (np.zeros(len(ks)) if s_ego is None else np.r_[0.0, s_ego][:len(ks)]) + EGO_REAR_TO_CENTER + EGO_LEN / 2
    lat = corridor_overlap(cd)
    inn = val & aligned & lat & (cs.min(-1) > front[None]) & (cs.min(-1) < lim[None])
    need = max(int(round(CUT_IN_MIN_S / dt)), 1); best = None
    for r in np.flatnonzero(~lat[:, 0] & val[:, 0]):
        run = 0
        for i in range(1, len(ks)):
            run = run + 1 if inn[r, i] else 0
            if run >= need:
                t_in = float(tau[i - run + 1]); best = t_in if best is None or t_in < best else best; break
    return best


def trainer_batch_keys(R: dict, rows) -> dict:
    """Select the keys FullLoss / HiCAPLoss read, for row indices `rows`, as numpy arrays (the caller converts to tensors)."""
    keys = ["exo_gap", "exo_close", "exo_acc", "exo_mode", "exo_onset", "exo_has_lead", "exo_w"]
    keys += [k for k in ("cand_p", "cand_q", "star", "J_teacher", "cost_t", "cf_cost", "cf_mask") if k in R]
    return {k: R[k][rows] for k in keys}
