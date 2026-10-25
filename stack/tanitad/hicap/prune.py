"""Prior-encoded pruning R1-R6, guard set, hysteresis, class-conditional episode-blocked conformal keep-sets (paper sec:prune).
Numpy, no torch. All masks are bitsets over the two factors (K_P path bits AND K_V profile bits)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .vocab import (FactoredVocab, compose, path_features, slots, DT, T_STEPS, S_GRID, N_LAT, N_LON, STANDSTILL_V0)

B_PHYS = dict(a_acc=6.0, a_brake=10.0, a_lat=9.0)     # m/s^2 physical limits (R1)


def curvature_along(traj: np.ndarray, dt: float = DT) -> np.ndarray:
    """Signed curvature at each step of a composed trajectory [T,2] (1/m); 0 where speed ~0."""
    p = np.vstack([[0.0, 0.0], traj])
    d = np.gradient(p, axis=0)
    dd = np.gradient(d, axis=0)
    num = d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]
    den = np.maximum((d ** 2).sum(1) ** 1.5, 1e-6)
    k = num / den
    k[(d ** 2).sum(1) < 1e-4] = 0.0
    return k[1:]


def profile_mask(vocab: FactoredVocab, v0: float, a0: float = 0.0, lim=B_PHYS, tube: float = 4.0) -> np.ndarray:
    """R1 profile bits: implied longitudinal acceleration/braking inside the physical limits, and the initial acceleration
    continuous with the measured one (a tube around a0). Standstill sub-vocabulary only when v0 < STANDSTILL_V0 (and vice versa)."""
    dv = vocab.profiles                                            # [K_V,T]
    v = v0 + dv
    acc = np.diff(np.concatenate([np.full((len(dv), 1), v0), v], 1), axis=1) / DT
    ok = (acc.max(1) <= lim["a_acc"]) & (acc.min(1) >= -lim["a_brake"]) & (v.min(1) >= -1e-3)
    ok &= np.abs(acc[:, 0] - a0) <= tube
    ok &= vocab.standstill == (v0 < STANDSTILL_V0)
    return ok


def path_mask(vocab: FactoredVocab, yaw_rate: float, v0: float, tube: float = 0.08) -> np.ndarray:
    """R1 path bits: initial curvature continuity (kappa0 ~ yaw_rate / v0, a tube) -- the measured state, not an input channel
    named curvature. At low speed the tube is open. Exact lateral-acceleration feasibility is tested on the composed entry."""
    if v0 < 1.0: return np.ones(vocab.K_P, bool)
    k0 = yaw_rate / v0
    kp = np.array([path_features(p, 32, 12.0)[0] for p in vocab.paths])
    return np.abs(kp - k0) <= tube


def pair_feasible(traj: np.ndarray, lim=B_PHYS) -> bool:
    """Exact check of a composed entry: lateral acceleration v^2 kappa <= a_lat."""
    v = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], traj]), axis=0), axis=1) / DT
    k = curvature_along(traj)
    return bool(np.all(v ** 2 * np.abs(k) <= lim["a_lat"] + 1e-6))


def guard_set(v0: float, yaw_rate: float, prev_traj: np.ndarray | None = None) -> np.ndarray:
    """Hold-current CTRA, gentle brake, firm brake, max brake in lane, previous plan shifted one tick. [G,T,2]. Injected AFTER every mask."""
    t = (np.arange(T_STEPS) + 1) * DT
    out = []
    for a in (0.0, -1.5, -4.0, -B_PHYS["a_brake"]):
        v = np.maximum(v0 + a * t, 0.0)
        s = np.cumsum(v * DT)
        th = yaw_rate * t if a == 0.0 else np.zeros_like(t)             # CTRA only for hold-current; brakes stay in lane
        x = np.cumsum(v * np.cos(th) * DT); y = np.cumsum(v * np.sin(th) * DT)
        out.append(np.stack([x, y], 1))
    if prev_traj is not None:
        sh = np.vstack([prev_traj[1:], prev_traj[-1:]]) - prev_traj[0]  # shifted one tick, re-origined
        out.append(sh.astype(np.float32))
    return np.stack(out).astype(np.float32)


def hysteresis_bonus(cand: np.ndarray, prev_traj: np.ndarray | None, tube: float = 0.5, bonus: float = 0.5) -> np.ndarray:
    """R5 soft prior bonus for candidates within a tube around the previous plan shifted one tick. cand [K,T,2]."""
    if prev_traj is None: return np.zeros(len(cand), np.float32)
    sh = np.vstack([prev_traj[1:], prev_traj[-1:]]) - prev_traj[0]
    d = np.linalg.norm(cand - sh[None], axis=-1).max(1)
    return np.where(d <= tube, bonus, 0.0).astype(np.float32)


def route_mask(nav: int, lat_cls_of_cell: np.ndarray) -> np.ndarray:
    """R3 (nav-fed arm only): nav 1=left excludes right-turn cells (lat class 2) and conversely; 0 follow keeps all."""
    ok = np.ones(len(lat_cls_of_cell), bool)
    if nav == 1: ok &= lat_cls_of_cell != 2
    if nav == 2: ok &= lat_cls_of_cell != 1
    return ok


# ---------------------------------------------------------------- R6 conformal
@dataclass
class ClassConditionalConformal:
    """Mondrian split conformal: nonconformity = -(score of the true node); threshold per class from the finite-sample quantile
    ceil((n+1)(1-alpha))/n. Keep-set = nodes whose score >= -threshold[class]. Joint recall >= 1 - sum_l alpha_l by the union bound."""
    alpha: float
    q: dict

    @classmethod
    def fit(cls, true_scores: np.ndarray, classes: np.ndarray, alpha: float, min_n: int = 5):
        q = {}
        allq = _quantile(-true_scores, alpha)
        for c in np.unique(classes):
            s = -true_scores[classes == c]
            q[int(c)] = _quantile(s, alpha) if len(s) >= min_n else allq     # sparse classes fall back to the marginal threshold
        q["_all"] = allq
        return cls(alpha, q)

    def keep(self, node_scores: np.ndarray, cls_of_true_guess: int) -> np.ndarray:
        return node_scores >= -self.q.get(int(cls_of_true_guess), self.q["_all"])


def _quantile(s: np.ndarray, alpha: float) -> float:
    n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1]) if n else float("inf")


def episode_blocked_coverage(covered: np.ndarray, episode: np.ndarray) -> dict:
    """Coverage computed with EPISODE as the unit (mean over episodes of the per-episode rate), plus the pooled rate."""
    eps = np.unique(episode)
    per = np.array([covered[episode == e].mean() for e in eps])
    return dict(pooled=float(covered.mean()), episode_mean=float(per.mean()), episode_p05=float(np.percentile(per, 5)), n_episodes=len(eps))
