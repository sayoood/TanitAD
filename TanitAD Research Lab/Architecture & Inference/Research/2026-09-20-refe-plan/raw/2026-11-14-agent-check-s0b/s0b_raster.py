"""S0-B raster builder (agent-check card, section 3). CPU only, no GPU, navtrain only.

Builds the L-OCC rasters of one fit-pack set straight from a nuPlan DB (the "cheap path", FEASIBILITY_AGENTS 2.2) and
exposes the arms R-true, R-t0, R-cv, R-dyn, R-true-2Hz, plus the zero/one regression rasters.

FROZEN MM DECISIONS (2026-11-14, before any data):
  grid     ego REAR-AXLE frame at t0, x in [-8,72) m (rows, 160), y in [-32,32) m (cols, 128), 0.5 m cells.
           row = floor((x+8)/0.5), col = floor((y+32)/0.5).  Cell (r,c) is OCCUPIED by a box iff its CENTRE lies in the
           rotated rectangle, boundary inclusive.  ONE rasteriser (`box_cells`) serves agents AND the ego footprint.
  classes  dynamic = vehicle, pedestrian, bicycle; static = barrier, traffic_cone, czone_sign, generic_object.
  time     primary 10 Hz, 51 steps 0.0..5.0 s, built as NAVSIM's MetricCacheProcessor._interpolate_gt_observation
           (navsim 3e8291b, metric_cache_processor.py:91-205): GT samples at t0+0.5k s (k=0..10) = lidar_pc rows i0+10k,
           per-track LINEAR interpolation to 0.1 s over [first sample, last sample] of that track only (no extrapolation);
           a track seen in exactly ONE sample is present at ALL 51 steps at its single pose (NAVSIM `start_time==end_time`);
           box size = size at the track's first sample.  Time grid literal: np.arange(0,51)*0.1 and np.arange(0,11)*0.5,
           so the float comparisons `start<=t<=end` are NAVSIM's own (MEASURED: every 5th step equals 0.5*k exactly).
  NAVSIM also interpolates STATIC objects' centres the same way (only their velocity is 0); this module mirrors the code,
  not the task text "statics held fixed" -- the difference is the labelling jitter of static boxes (named in HANDOFF).
  no confidence filter (neither devkit filters on it); no radius or max-object cap (update_detections_tracks has none).
  ego pose = ego_pose row of lidar_pc.ego_pose_token; nuPlan builds EgoState.build_from_rear_axle(StateSE2(x, y, q.yaw))
  from it (nuplan_scenario_queries.py:559-598, DriveRL devkit), i.e. ego_pose IS the rear axle.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

X0, Y0, RES = -8.0, -32.0, 0.5
NX, NY = 160, 128
NXY = NX * NY
T_STEPS = 51                                   # 0.0 .. 5.0 s at 10 Hz
N_SAMP = 11                                    # 0.0 .. 5.0 s at 2 Hz
STEP_T = np.arange(0, 51, 1, dtype=float) * 0.1          # NAVSIM literal (interpolated_time_s)
SAMPLE_T = np.arange(0, 11, 1, dtype=float) * 0.5        # NAVSIM literal (relative_time_s)
DYN_NAMES = ("vehicle", "pedestrian", "bicycle")
STATIC_NAMES = ("barrier", "traffic_cone", "czone_sign", "generic_object")
EPS = 1e-9


# ----------------------------------------------------------------------------------------------- frames
def global_to_ego(px, py, pyaw, ex, ey, eyaw):
    """Global map frame -> ego rear-axle frame at t0 (x forward, y left)."""
    c, s = np.cos(eyaw), np.sin(eyaw)
    dx, dy = np.asarray(px, float) - ex, np.asarray(py, float) - ey
    return c * dx + s * dy, -s * dx + c * dy, np.asarray(pyaw, float) - eyaw


def quat_yaw(qw, qx, qy, qz):
    """== pyquaternion Quaternion(qw,qx,qy,qz).yaw_pitch_roll[0], which nuPlan uses for the ego heading."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ----------------------------------------------------------------------------------------------- rasteriser
def box_cells(cx, cy, yaw, hl, hw, radius=None):
    """The ONE rasteriser. Boxes (M,) centre (cx,cy) [ego frame, m], heading yaw, half-length hl, half-width hw.
    Returns rows, cols (M,K) int and inside (M,K) bool (cell centre in the rectangle, boundary inclusive) over a
    (2W+1)^2 window around the centre cell; `valid` = inside AND the cell is in the 160x128 grid.  Also `partial`
    (M,) = box has inside cells outside the grid.  Out-of-grid cells are dropped (they count as FREE)."""
    cx, cy, yaw = (np.atleast_1d(np.asarray(a, float)) for a in (cx, cy, yaw))
    hl = np.broadcast_to(np.asarray(hl, float), cx.shape)
    hw = np.broadcast_to(np.asarray(hw, float), cx.shape)
    if radius is None:
        radius = int(np.ceil(float(np.max(np.hypot(hl, hw))) / RES)) + 1
    w = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(w, w, indexing="ij")
    r0 = np.floor((cx - X0) / RES).astype(np.int64)
    c0 = np.floor((cy - Y0) / RES).astype(np.int64)
    rows = r0[:, None] + dr.ravel()[None, :]
    cols = c0[:, None] + dc.ravel()[None, :]
    xc = X0 + (rows + 0.5) * RES - cx[:, None]
    yc = Y0 + (cols + 0.5) * RES - cy[:, None]
    co, si = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    u = co * xc + si * yc
    v = -si * xc + co * yc
    inside = (np.abs(u) <= hl[:, None] + EPS) & (np.abs(v) <= hw[:, None] + EPS)
    ingrid = (rows >= 0) & (rows < NX) & (cols >= 0) & (cols < NY)
    valid = inside & ingrid
    partial = (inside & ~ingrid).any(axis=1)
    return rows, cols, valid, partial


def rasterize_boxes(R, chan, tidx, cx, cy, yaw, hl, hw):
    """Write boxes into R[chan, t, row, col]. All args (M,) except R (C,T,NX,NY) bool; one size per box."""
    if len(cx) == 0:
        return R
    rows, cols, valid, _ = box_cells(cx, cy, yaw, hl, hw)
    m = np.broadcast_to(np.asarray(tidx)[:, None], rows.shape)
    ch = np.broadcast_to(np.asarray(chan)[:, None], rows.shape)
    R[ch[valid], m[valid], rows[valid], cols[valid]] = True
    return R


def rasterize_tracks(x, y, yaw, present, hl, hw, dyn, n_steps):
    """Tracks (N,T) in the ego frame (+ per-track size/dyn) -> R (2,T,NX,NY) bool; channel 0 dynamic, 1 static."""
    R = np.zeros((2, n_steps, NX, NY), dtype=bool)
    for i in range(x.shape[0]):
        p = present[i]
        if not p.any():
            continue
        reach = max(hl[i], hw[i]) * 1.5 + 2 * RES
        xi, yi = x[i][p], y[i][p]
        keep = (xi > X0 - reach) & (xi < X0 + NX * RES + reach) & (yi > Y0 - reach) & (yi < Y0 + NY * RES + reach)
        if not keep.any():
            continue
        t = np.nonzero(p)[0][keep]
        rasterize_boxes(R, np.full(len(t), 0 if dyn[i] else 1), t, xi[keep], yi[keep], yaw[i][p][keep],
                        np.full(len(t), hl[i]), np.full(len(t), hw[i]))
    return R


# ----------------------------------------------------------------------------------------------- DB
class LogDB:
    """One nuPlan log DB, read-only/immutable, loaded once into numpy. Sets are addressed by lidar_pc token (hex)."""

    def __init__(self, path):
        uri = "file:" + str(path).replace("\\", "/") + "?mode=ro&immutable=1"
        con = sqlite3.connect(uri, uri=True)
        try:
            pc = con.execute("SELECT lower(hex(token)), timestamp, lower(hex(ego_pose_token)) FROM lidar_pc "
                             "ORDER BY timestamp").fetchall()
            self.pc_tokens = [r[0] for r in pc]
            self.pc_ts = np.array([r[1] for r in pc], dtype=np.int64)
            self.pc_index = {t: i for i, t in enumerate(self.pc_tokens)}
            ego_tok = [r[2] for r in pc]
            ego = {r[0]: r[1:] for r in con.execute(
                "SELECT lower(hex(token)), x, y, qw, qx, qy, qz FROM ego_pose")}
            self.ego = np.array([ego[t] for t in ego_tok], dtype=np.float64)       # (n,6) x y qw qx qy qz
            rows = con.execute(
                "SELECT lower(hex(lb.lidar_pc_token)), lb.track_token, c.name, lb.x, lb.y, lb.yaw, lb.width, lb.length, "
                "lb.vx, lb.vy FROM lidar_box lb JOIN track t ON t.token = lb.track_token "
                "JOIN category c ON c.token = t.category_token").fetchall()
        finally:
            con.close()
        n = len(rows)
        pidx = np.fromiter((self.pc_index.get(r[0], -1) for r in rows), dtype=np.int64, count=n)
        trk = {}
        tid = np.fromiter((trk.setdefault(r[1], len(trk)) for r in rows), dtype=np.int64, count=n)
        order = np.argsort(pidx, kind="stable")
        self.b_pc = pidx[order]
        self.b_tid = tid[order]
        self.b_dyn = np.array([rows[j][2] in DYN_NAMES for j in order], dtype=bool)
        self.b_name = [rows[j][2] for j in order]
        f = np.array([[r[3], r[4], r[5], r[6], r[7], r[8], r[9]] for r in rows], dtype=np.float64).reshape(n, 7)[order]
        self.b_x, self.b_y, self.b_yaw, self.b_w, self.b_l, self.b_vx, self.b_vy = f.T
        self.b_start = np.searchsorted(self.b_pc, np.arange(len(self.pc_tokens) + 1))
        dt = np.median(np.diff(self.pc_ts[:400])) / 1e6 if len(self.pc_ts) > 2 else 0.05
        self.stride = int(round(0.5 / dt))                  # 10 at the 20 Hz DB rate (NAVSIM: int(0.5/database_interval))
        self.median_dt = float(dt)

    def ego_pose(self, i):
        x, y, qw, qx, qy, qz = self.ego[i]
        return x, y, float(quat_yaw(qw, qx, qy, qz))

    def sample_rows(self, i):
        a, b = self.b_start[i], self.b_start[i + 1]
        return slice(a, b)


# ----------------------------------------------------------------------------------------------- tracks
@dataclass
class SetTracks:
    """Everything one set needs, in the ego frame at t0. N tracks; raw = the 11 GT samples (dense, NaN-free + mask)."""
    raw_x: np.ndarray        # (N,11)
    raw_y: np.ndarray
    raw_yaw: np.ndarray
    raw_present: np.ndarray  # (N,11) bool
    hl: np.ndarray           # (N,) half length, size at the track's FIRST sample
    hw: np.ndarray
    dyn: np.ndarray          # (N,) bool
    vx0: np.ndarray          # (N,) lidar_box vx at sample 0 in the ego frame (rotated), 0 where absent / static
    vy0: np.ndarray
    ego: tuple


def gather_set_tracks(ldb: LogDB, i0: int) -> SetTracks | None:
    """The 11 GT frames of the set whose t0 lidar_pc row is i0. None if the 5 s future does not exist."""
    last = i0 + ldb.stride * (N_SAMP - 1)
    if last >= len(ldb.pc_tokens):
        return None
    ex, ey, eyaw = ldb.ego_pose(i0)
    tids, ks, xs, ys, yws, ws, ls, vxs, vys, dyns = [], [], [], [], [], [], [], [], [], []
    for k in range(N_SAMP):
        sl = ldb.sample_rows(i0 + ldb.stride * k)
        n = sl.stop - sl.start
        tids.append(ldb.b_tid[sl]); ks.append(np.full(n, k)); xs.append(ldb.b_x[sl]); ys.append(ldb.b_y[sl])
        yws.append(ldb.b_yaw[sl]); ws.append(ldb.b_w[sl]); ls.append(ldb.b_l[sl]); vxs.append(ldb.b_vx[sl])
        vys.append(ldb.b_vy[sl]); dyns.append(ldb.b_dyn[sl])
    tid, k, x, y, yaw, w, l, vx, vy, dyn = (np.concatenate(a) for a in (tids, ks, xs, ys, yws, ws, ls, vxs, vys, dyns))
    lx, ly, lyaw = global_to_ego(x, y, yaw, ex, ey, eyaw)
    c, s = np.cos(eyaw), np.sin(eyaw)
    lvx, lvy = c * vx + s * vy, -s * vx + c * vy
    uniq, inv = np.unique(tid, return_inverse=True)
    N = len(uniq)
    rx = np.zeros((N, N_SAMP)); ry = np.zeros((N, N_SAMP)); ryaw = np.zeros((N, N_SAMP))
    pres = np.zeros((N, N_SAMP), dtype=bool)
    hl = np.zeros(N); hw = np.zeros(N); dd = np.zeros(N, dtype=bool); v0x = np.zeros(N); v0y = np.zeros(N)
    first_k = np.full(N, 99)
    order = np.lexsort((k, inv))                              # by track, then sample: first row of a track = first sample
    for j in order:
        i = inv[j]; kk = k[j]
        rx[i, kk], ry[i, kk], ryaw[i, kk], pres[i, kk] = lx[j], ly[j], lyaw[j], True
        if kk < first_k[i]:
            first_k[i] = kk
            hl[i], hw[i], dd[i] = l[j] / 2.0, w[j] / 2.0, dyn[j]
        if kk == 0 and dyn[j]:
            v0x[i], v0y[i] = lvx[j], lvy[j]
    return SetTracks(rx, ry, ryaw, pres, hl, hw, dd, v0x, v0y, (ex, ey, eyaw))


def interpolate_tracks(raw_x, raw_y, raw_yaw, raw_present, step_t=STEP_T, sample_t=SAMPLE_T):
    """NAVSIM StateInterpolator mirror -> (x,y,yaw,present) each (N,51). Linear, heading unwrapped, per track only inside
    [first, last] sample time; one-sample tracks are present at every step at their single pose."""
    N = raw_x.shape[0]
    T = len(step_t)
    x = np.zeros((N, T)); y = np.zeros((N, T)); yaw = np.zeros((N, T)); pres = np.zeros((N, T), dtype=bool)
    for i in range(N):
        ks = np.nonzero(raw_present[i])[0]
        if len(ks) == 0:
            continue
        if len(ks) == 1:
            j = ks[0]
            x[i], y[i], yaw[i], pres[i] = raw_x[i, j], raw_y[i, j], raw_yaw[i, j], True
            continue
        t = sample_t[ks]
        inr = (step_t >= t[0]) & (step_t <= t[-1])           # float compare exactly as interp1d's range test
        x[i] = np.interp(step_t, t, raw_x[i, ks])
        y[i] = np.interp(step_t, t, raw_y[i, ks])
        yaw[i] = np.interp(step_t, t, np.unwrap(raw_yaw[i, ks]))
        pres[i] = inr
    return x, y, yaw, pres


# ----------------------------------------------------------------------------------------------- arms
ARMS_RASTER = ("R-true", "R-t0", "R-cv", "R-dyn", "R-true-2Hz")


def combine(R, drop_static=False):
    """(2,T,NX,NY) -> flat gather table (T, NXY+1) bool; the last column is a permanent FREE pad for padded indices."""
    T = R.shape[1]
    occ = R[0] if drop_static else (R[0] | R[1])
    out = np.zeros((T, NXY + 1), dtype=bool)
    out[:, :NXY] = occ.reshape(T, NXY)
    return out


def near_sample_index(step_t=STEP_T):
    """10 Hz step -> nearest 0.5 s sample, ties (t=0.25,0.75,..) go to the LATER sample."""
    return np.minimum(np.floor(step_t / 0.5 + 0.5 + 1e-9).astype(int), N_SAMP - 1)


def build_arms(st: SetTracks, which=ARMS_RASTER):
    """-> {arm: (T,NXY+1) bool gather table}. R-dyn is R-true without the static channel."""
    out = {}
    need_true = any(a in which for a in ("R-true", "R-dyn"))
    if need_true:
        x, y, yaw, pres = interpolate_tracks(st.raw_x, st.raw_y, st.raw_yaw, st.raw_present)
        Rt = rasterize_tracks(x, y, yaw, pres, st.hl, st.hw, st.dyn, T_STEPS)
        if "R-true" in which:
            out["R-true"] = combine(Rt)
        if "R-dyn" in which:
            out["R-dyn"] = combine(Rt, drop_static=True)
    p0 = st.raw_present[:, 0]
    if "R-t0" in which:
        R0 = rasterize_tracks(st.raw_x[:, :1], st.raw_y[:, :1], st.raw_yaw[:, :1], p0[:, None], st.hl, st.hw, st.dyn, 1)
        out["R-t0"] = np.repeat(combine(R0), T_STEPS, axis=0)
    if "R-cv" in which:
        tt = STEP_T[None, :]
        x = st.raw_x[:, :1] + st.vx0[:, None] * tt
        y = st.raw_y[:, :1] + st.vy0[:, None] * tt
        yaw = np.repeat(st.raw_yaw[:, :1], T_STEPS, axis=1)
        out["R-cv"] = combine(rasterize_tracks(x, y, yaw, np.repeat(p0[:, None], T_STEPS, axis=1), st.hl, st.hw, st.dyn, T_STEPS))
    if "R-true-2Hz" in which:
        R2 = rasterize_tracks(st.raw_x, st.raw_y, st.raw_yaw, st.raw_present, st.hl, st.hw, st.dyn, N_SAMP)
        out["R-true-2Hz"] = combine(R2)[near_sample_index()]
    return out


def reg_raster(kind):
    """C-REG: 'zero' (all free) or 'one' (every real cell occupied; the pad column stays free)."""
    out = np.zeros((T_STEPS, NXY + 1), dtype=bool)
    if kind == "one":
        out[:, :NXY] = True
    return out
