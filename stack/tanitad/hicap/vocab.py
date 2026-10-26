"""Action tree and factored trajectory vocabulary (paper sec:vocab).

Level 1 strategic (5 usable), Level 2 tactical cell = lateral x longitudinal (v9: 3 x 7 = 21 cells, 19 populated,
census `hicap_v9_screen_census_result.json`), Level 3 = factored product of K_P paths x K_V speed profiles composed by
arc-length re-parameterisation. Pure numpy; no torch.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

LAT_V9 = ("LANE_KEEP", "TURN_L", "TURN_R")                      # LAT9 ids 0..2 (lane change ids 3,4 never positive)
LON_V9 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")   # LON9 ids 0..6
N_LAT, N_LON = len(LAT_V9), len(LON_V9)
STRATEGIC_USABLE = 5                                              # v7/v8.1 strategic release (not in v9)
DT = 0.1                                                          # s, trainer clock
T_STEPS = 60                                                      # 6 s at 10 Hz
SLOT_T = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)                 # output slots (s)
SLOT_IDX = tuple(int(round(t / DT)) - 1 for t in SLOT_T)          # index into the 60 steps (t = (i+1) dt)
S_GRID = np.arange(0.0, 151.0, 1.0)                               # arc-length grid for paths (m)
STANDSTILL_V0 = 0.5                                               # m/s: below this the profile comes from the standstill sub-vocabulary


def pair_admissibility(joint_counts: np.ndarray, min_count: int = 1) -> np.ndarray:
    """b mask for the factored cell score: 0 where the pair occurs in the census, -inf where it never does."""
    b = np.where(joint_counts >= min_count, 0.0, -np.inf)
    return b.astype(np.float32)


# ---------------------------------------------------------------- trajectory <-> (path, profile)
def extract_path_profile(traj_xy: np.ndarray, v0: float, dt: float = DT):
    """traj_xy [T,2] ego-frame positions at t=dt..T*dt (x forward, y left) -> (path[len(S_GRID),2], dv[T]).

    path is the polyline re-sampled on S_GRID (m of arc length, extrapolated straight beyond the last point);
    dv[t] = v(t) - v0 with v from finite differences of arc length (v(0)=v0 handled by prepending the origin)."""
    pts = np.vstack([[0.0, 0.0], traj_xy])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    v = seg / dt                                                    # speed on each interval, [T]
    dv = v - v0
    sg = S_GRID
    x = np.interp(sg, s, pts[:, 0]); y = np.interp(sg, s, pts[:, 1])
    over = sg > s[-1]
    if over.any() and len(pts) >= 2 and seg[-1] > 1e-6:             # continue along the last heading
        h = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
        x[over] = pts[-1, 0] + (sg[over] - s[-1]) * h[0]; y[over] = pts[-1, 1] + (sg[over] - s[-1]) * h[1]
    elif over.any():
        x[over], y[over] = pts[-1]
    return np.stack([x, y], 1).astype(np.float32), dv.astype(np.float32)


def compose(path: np.ndarray, dv: np.ndarray, v0: float, dt: float = DT) -> np.ndarray:
    """Compose a trajectory [T,2] from a path polyline (on S_GRID) and a speed-change profile dv[T] over v0.
    v(t) = max(v0 + dv(t), 0); s(t) = cumulative sum v dt; position = path(s(t)). Works for batches via leading dims."""
    v = np.maximum(v0 + dv, 0.0)
    s = np.cumsum(v * dt, axis=-1)
    s = np.minimum(s, S_GRID[-1])
    x = np.interp(s, S_GRID, path[..., 0]); y = np.interp(s, S_GRID, path[..., 1])
    return np.stack([x, y], -1).astype(np.float32)


def slots(traj: np.ndarray) -> np.ndarray:
    return traj[..., list(SLOT_IDX), :]


def path_features(path: np.ndarray, n: int = 32, s_max: float = 60.0) -> np.ndarray:
    """Curvature profile of a path on n arc-length points up to s_max (numeric MLP input). Signed curvature, 1/m."""
    idx = np.linspace(0, int(min(s_max, S_GRID[-1])), n + 2).astype(int)
    p = path[idx]
    d1 = p[2:] - p[:-2]
    th = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
    ds = np.maximum(np.linalg.norm(d1, axis=1), 1e-3)
    return (np.gradient(th) / ds * 2.0).astype(np.float32)


def profile_features(dv: np.ndarray, n: int = 20) -> np.ndarray:
    idx = np.linspace(0, len(dv) - 1, n).astype(int)
    return dv[idx].astype(np.float32)


def path_kmax(path: np.ndarray) -> float:
    return float(np.max(np.abs(path_features(path, 32))))


# ---------------------------------------------------------------- codebooks
def farthest_point(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices of k farthest-point samples of rows of X (deterministic start = medoid-ish nearest to mean)."""
    n = len(X)
    k = min(k, n)
    first = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
    sel = [first]
    d = ((X - X[first]) ** 2).sum(1)
    for _ in range(k - 1):
        j = int(np.argmax(d)); sel.append(j)
        d = np.minimum(d, ((X - X[j]) ** 2).sum(1))
    return np.array(sel)


@dataclass
class FactoredVocab:
    """paths [K_P,len(S_GRID),2], path_cls [K_P] (lateral class), profiles [K_V,T] (dv), prof_cls [K_V] (longitudinal class),
    standstill [K_V] bool (profiles used when v0 < STANDSTILL_V0)."""
    paths: np.ndarray
    path_cls: np.ndarray
    profiles: np.ndarray
    prof_cls: np.ndarray
    standstill: np.ndarray

    @property
    def K_P(self): return len(self.paths)
    @property
    def K_V(self): return len(self.profiles)
    @property
    def N(self): return self.K_P * self.K_V

    def entry(self, p: int, q: int, v0: float) -> np.ndarray:
        return compose(self.paths[p], self.profiles[q], v0)


def build_factored_vocab(paths, lat_cls, dvs, lon_cls, v0s, K_P=1024, K_V=256, floor_P=8, floor_V=4,
                         standstill_frac=0.1) -> FactoredVocab:
    """Farthest-point sampling WITHIN class with a floor per class (k-means starves rare classes). Training split only.
    paths [N,len(S_GRID),2]; dvs [N,T]; v0s [N]. The standstill sub-vocabulary is sampled from windows with v0<STANDSTILL_V0."""
    lat_cls = np.asarray(lat_cls); lon_cls = np.asarray(lon_cls); v0s = np.asarray(v0s)
    pf = np.stack([path_features(p) for p in paths])
    sel_p, cls_p = [], []
    counts = np.array([(lat_cls == c).sum() for c in range(N_LAT)], float)
    quota = np.maximum(floor_P, np.round(K_P * counts / max(counts.sum(), 1))).astype(int)
    for c in range(N_LAT):
        ix = np.where(lat_cls == c)[0]
        if len(ix) == 0: continue
        pick = ix[farthest_point(pf[ix], int(quota[c]))]
        sel_p += list(pick); cls_p += [c] * len(pick)
    moving = v0s >= STANDSTILL_V0
    n_still = int(round(K_V * standstill_frac))
    sel_v, cls_v, still = [], [], []
    for stand, kk in ((False, K_V - n_still), (True, n_still)):
        m = (~moving) if stand else moving
        counts = np.array([((lon_cls == c) & m).sum() for c in range(N_LON)], float)
        quota = np.maximum(floor_V, np.round(kk * counts / max(counts.sum(), 1))).astype(int)
        for c in range(N_LON):
            ix = np.where((lon_cls == c) & m)[0]
            if len(ix) == 0: continue
            pick = ix[farthest_point(np.stack([profile_features(dvs[i]) for i in ix]), int(quota[c]))]
            sel_v += list(pick); cls_v += [c] * len(pick); still += [stand] * len(pick)
    return FactoredVocab(np.asarray(paths)[sel_p], np.array(cls_p), np.asarray(dvs)[sel_v], np.array(cls_v), np.array(still, bool))
