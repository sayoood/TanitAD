"""PRIVGEOM-SAFE geometry scalars: privileged per-proposal geometry (DIAGNOSTIC ONLY -- never an inference input).

PURE module: shapely + numpy, no nuPlan / NAVSIM import.  The scalars are the FIXED LIST of PREREG_PRIVGEOM_SAFE.md s3 (K = 15):

  DAC row (11):  dac_clear_min        min over the 41 steps x 4 footprint corners of the signed clearance (m, + inside / - outside) to the drivable-area boundary, capped to [-CAP_CLEAR, +CAP_CLEAR]
                 dac_frac_out         fraction of the 41 steps with FEWER than 4 corners inside the drivable area (NAVSIM's non-drivable step test)
                 dac_clear_t0..t40    the per-step signed clearance (min over the 4 corners) at the 9 key times t = 0, 0.5, ..., 4.0 s (steps 0, 5, ..., 40)
  NC row (4):    nc_min_dist          min over steps and over EVERY non-red-light tracked box of the distance (m) from the ego footprint polygon to the box, capped at D_CAP
                 nc_min_tgap          min over steps of (gap to the lead / max(ego speed, V_MIN)), capped at T_CAP (s); T_CAP when there is no lead
                 nc_max_close         max over steps of max(0, ego speed - lead speed along the ego heading), capped at CLOSE_CAP (m/s); 0 when there is no lead
                 nc_min_ttc_margin    min over steps of (TTC - 1.0 s); TTC = 0 if gap <= 0, gap / closing if closing > 1e-3, else T_CAP

LEAD (fixed definition, an APPROXIMATION of "along the path"): at step t the lead is the tracked box, among those whose polygon intersects the straight corridor
{ 0 <= x_ego <= D_LEAD ahead of the FRONT BUMPER, |y_ego| <= CORR_HALF_W } in the ego frame of step t, with the smallest gap; gap = max(0, min over the box's 4 vertices of the longitudinal
coordinate relative to the front bumper).  Agent velocity = forward finite difference of the box centroid between consecutive 0.2 s observation slots (backward at the last slot).

THE SIGN OF dac_clear_min IS THE DAC LABEL (min clearance < 0  <=>  some step has a corner outside the union  <=>  NAVSIM DAC = 0, up to polygon seams) and nc_min_dist == 0 is the
overlap event that the NC label scores: the primary G contains the label-producing test BY CONSTRUCTION (PREREG s3.4).  `unsigned()` / `floored()` give the registered no-contact variants.
"""
from __future__ import annotations

import numpy as np
import shapely

CAP_CLEAR = 10.0          # m, clearance cap (both signs)
D_CAP = 20.0              # m, agent-distance cap
T_CAP = 10.0              # s, time-gap / TTC cap (no lead, or far)
V_MIN = 0.5               # m/s, floor of the ego speed in the time gap
CLOSE_CAP = 20.0          # m/s
D_LEAD = 50.0             # m, corridor length ahead of the front bumper
FRONT_FROM_REAR = 4.049   # m, nuPlan Pacifica: front bumper ahead of the rear axle
CORR_HALF_W = 2.297 / 2.0 + 0.3     # m, ego half width + 0.3 m
KEY_STEPS = (0, 5, 10, 15, 20, 25, 30, 35, 40)      # t = 0, 0.5, ..., 4.0 s on the 0.1 s grid
N_STEPS = 41
DAC_NAMES = ("dac_clear_min", "dac_frac_out") + tuple(f"dac_clear_t{int(round(s * 0.1 * 10)):02d}" for s in KEY_STEPS)
NC_NAMES = ("nc_min_dist", "nc_min_tgap", "nc_max_close", "nc_min_ttc_margin")
K_DAC, K_NC = len(DAC_NAMES), len(NC_NAMES)
G_NAMES = DAC_NAMES + NC_NAMES
DAC_COLS = tuple(range(0, K_DAC))
NC_COLS = tuple(range(K_DAC, K_DAC + K_NC))
IX_X, IX_Y, IX_H, IX_VX, IX_VY = 0, 1, 2, 3, 4          # navsim StateIndex
FL, RL, RR, FR, CTR = 0, 1, 2, 3, 4                     # navsim BBCoordsIndex


class Drivable:
    """The union of the drivable polygons (ROADBLOCK + INTERSECTION + CARPARK_AREA of the PDM drivable map) with a segment tree of its boundary."""

    def __init__(self, polys):
        polys = [p for p in polys if p is not None and not p.is_empty]
        self.empty = len(polys) == 0
        self.n_polys = len(polys)
        self.U = None
        self.tree = None
        self.n_seg = 0
        if self.empty:
            return
        U = shapely.union_all(polys)
        shapely.prepare(U)
        self.U = U
        segs = []
        for ln in shapely.get_parts(U.boundary):
            c = np.asarray(ln.coords, np.float64)
            if len(c) >= 2:
                segs.append(np.stack([c[:-1], c[1:]], axis=1))
        if segs:
            S = np.concatenate(segs)
            self.n_seg = len(S)
            self.tree = shapely.STRtree(shapely.linestrings(S))


def signed_clearance(drv: Drivable, pts: np.ndarray, cap: float = CAP_CLEAR) -> tuple:
    """pts [n, 2] -> (signed clearance [n] in [-cap, cap]; + inside the union, - outside), inside [n] bool."""
    n = len(pts)
    if drv.empty or n == 0:
        return np.full(n, -cap), np.zeros(n, bool)
    inside = shapely.contains_xy(drv.U, pts[:, 0], pts[:, 1])
    d = np.full(n, float(cap))
    if drv.tree is not None:
        idx, dist = drv.tree.query_nearest(shapely.points(pts), max_distance=cap, return_distance=True, all_matches=False)
        d[idx[0]] = np.minimum(dist, cap)
    return np.where(inside, d, -d), inside


def dac_scalars(coords: np.ndarray, drv: Drivable, cap: float = CAP_CLEAR) -> np.ndarray:
    """coords [P, T, 5, 2] (navsim BBCoordsIndex layout; only the 4 corners are read) -> [P, 11] (DAC_NAMES)."""
    P, T = coords.shape[:2]
    if T != N_STEPS:
        raise ValueError(f"expected {N_STEPS} steps, got {T}")
    pts = np.ascontiguousarray(coords[:, :, :4, :]).reshape(-1, 2)
    sc, inside = signed_clearance(drv, pts, cap)
    sc = sc.reshape(P, T, 4)
    inside = inside.reshape(P, T, 4)
    step = sc.min(axis=-1)
    out = np.empty((P, K_DAC), np.float64)
    out[:, 0] = step.min(axis=1)
    out[:, 1] = (inside.sum(axis=-1) < 4).mean(axis=1)
    out[:, 2:] = step[:, list(KEY_STEPS)]
    return out


class Slot:
    """The tracked boxes of ONE observation slot (0.2 s): polygons, STRtree, centroids, velocities, the 4 vertices."""

    def __init__(self, polys, vels=None):
        polys = list(polys)
        self.n = len(polys)
        self.polys = np.empty(self.n, dtype=object)
        for i, p in enumerate(polys):
            self.polys[i] = p
        self.tree = shapely.STRtree(self.polys) if self.n else None
        self.cent = np.array([[p.centroid.x, p.centroid.y] for p in polys], np.float64).reshape(-1, 2)
        self.vels = np.zeros((self.n, 2)) if vels is None else np.asarray(vels, np.float64).reshape(-1, 2)
        v = np.zeros((self.n, 4, 2))
        for i, p in enumerate(polys):
            c = np.asarray(p.exterior.coords, np.float64)[:-1]
            if len(c) >= 4:
                v[i] = c[:4]
            elif len(c) > 0:
                v[i] = np.resize(c, (4, 2))
        self.verts = v


def build_slots(maps, dt_slot: float = 0.2, exclude: str = "red_light") -> list:
    """maps: list over slots of (tokens, geometries).  Velocities by forward finite difference of the centroid per token (backward at the last slot; 0 if the token is absent next door)."""
    keep = []
    for toks, geoms in maps:
        k = [i for i, t in enumerate(toks) if exclude not in str(t)]
        keep.append(([toks[i] for i in k], [geoms[i] for i in k]))
    cents = []
    for toks, geoms in keep:
        cents.append({t: np.array([g.centroid.x, g.centroid.y]) for t, g in zip(toks, geoms)})
    slots = []
    n = len(keep)
    for k, (toks, geoms) in enumerate(keep):
        vel = []
        for t in toks:
            v = np.zeros(2)
            if k + 1 < n and t in cents[k + 1]:
                v = (cents[k + 1][t] - cents[k][t]) / dt_slot
            elif k - 1 >= 0 and t in cents[k - 1]:
                v = (cents[k][t] - cents[k - 1][t]) / dt_slot
            vel.append(v)
        slots.append(Slot(geoms, vel))
    return slots


def nc_scalars(coords: np.ndarray, states: np.ndarray, slots: list, slot_of_step=None) -> np.ndarray:
    """coords [P, T, 5, 2], states [P, T, >=5] (navsim state arrays) -> [P, 4] (NC_NAMES)."""
    P, T = coords.shape[:2]
    if slot_of_step is None:
        slot_of_step = [t // 2 for t in range(T)]
    heading = states[..., IX_H]
    dirv = np.stack([np.cos(heading), np.sin(heading)], axis=-1)                # [P, T, 2]
    speed = np.hypot(states[..., IX_VX], states[..., IX_VY])
    ring = np.ascontiguousarray(coords[:, :, [FL, RL, RR, FR, FL], :])
    ego_polys = shapely.polygons(ring)                                           # [P, T]
    dmin = np.full((P, T), D_CAP)
    gap = np.full((P, T), np.inf)
    close = np.zeros((P, T))
    lead = np.zeros((P, T), bool)
    for t in range(T):
        k = min(int(slot_of_step[t]), len(slots) - 1)
        sl = slots[k] if k >= 0 else None
        if sl is None or sl.n == 0:
            continue
        idx, dist = sl.tree.query_nearest(ego_polys[:, t], max_distance=D_CAP, return_distance=True, all_matches=False)
        if idx.shape[1]:
            dmin[idx[0], t] = np.minimum(dist, D_CAP)
        d = dirv[:, t]
        f = states[:, t, :2] + FRONT_FROM_REAR * d
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=-1)
        c0, c1 = f + CORR_HALF_W * nrm, f - CORR_HALF_W * nrm
        corr = shapely.polygons(np.stack([c0, c1, c1 + D_LEAD * d, c0 + D_LEAD * d, c0], axis=1))
        pa = sl.tree.query(corr, predicate="intersects")
        if pa.shape[1] == 0:
            continue
        pi, ai = pa
        lon = ((sl.verts[ai] - f[pi][:, None, :]) * d[pi][:, None, :]).sum(-1)       # [m, 4]
        g = np.maximum(lon.min(axis=1), 0.0)
        best = np.full(P, np.inf)
        np.minimum.at(best, pi, g)
        sel = np.nonzero(g <= best[pi] + 1e-12)[0]
        first = np.full(P, -1, np.int64)
        first[pi[sel[::-1]]] = sel[::-1]                                              # the lowest pair index wins
        has = first >= 0
        kk = first[has]
        pp, aa = pi[kk], ai[kk]
        gap[pp, t] = g[kk]
        close[pp, t] = speed[pp, t] - (sl.vels[aa] * d[pp]).sum(-1)
        lead[pp, t] = True
    tg = np.where(lead, np.minimum(T_CAP, gap / np.maximum(speed, V_MIN)), T_CAP)
    ttc = np.where(lead, np.where(gap <= 1e-6, 0.0, np.where(close > 1e-3, np.minimum(T_CAP, gap / np.maximum(close, 1e-3)), T_CAP)), T_CAP)
    out = np.empty((P, K_NC), np.float64)
    out[:, 0] = dmin.min(axis=1)
    out[:, 1] = tg.min(axis=1)
    out[:, 2] = np.clip(np.where(lead, close, 0.0), 0.0, CLOSE_CAP).max(axis=1)
    out[:, 3] = (ttc - 1.0).min(axis=1)
    return out


def all_scalars(coords: np.ndarray, states: np.ndarray, drv: Drivable, slots: list) -> np.ndarray:
    """[P, 15] = DAC (11) then NC (4)."""
    return np.concatenate([dac_scalars(coords, drv), nc_scalars(coords, states, slots)], axis=1)


# ---- the registered no-contact variants (PREREG s3.4; descriptive, NO authority over the branches) -----------------------------------------------------------------------------------
def unsigned(g: np.ndarray) -> np.ndarray:
    """DAC no-sign variant: |clearances|, frac_out DROPPED -> [..., 10].  Inside and outside at the same distance look alike; the label is no longer a monotone function of any column."""
    g = np.asarray(g)
    return np.abs(g[..., [0] + list(range(2, K_DAC))])


def floored(g_nc: np.ndarray, floor_d: float = 0.5, floor_t: float = 0.5, floor_m: float = 0.0) -> np.ndarray:
    """NC no-contact variant on the 4 NC columns: distance and time gap floored at 0.5 (m / s), TTC margin floored at 0.0 (TTC >= 1.0 s): contact and near-contact become indistinguishable."""
    g = np.array(g_nc, dtype=np.float64, copy=True)
    g[..., 0] = np.maximum(g[..., 0], floor_d)
    g[..., 1] = np.maximum(g[..., 1], floor_t)
    g[..., 3] = np.maximum(g[..., 3], floor_m)
    return g
