"""level_floor.py -- the floor operator F of PREREG_LEVELFLOOR_A3 (sha256 977a785e...), section 2. Pure numpy, float64, no torch.

F lifts a plan's ALONG-TRACK progress to a floor derived from a vision-only speed estimate, and leaves the path geometry alone:

  inputs   plan p [30, 2] (ego frame, knots k = 1..30 at t_k = k * 0.2 s, the origin implied), v0 (the tick's measured ego speed),
           and from the floor heads: v_hat (m/s), pi = sigmoid(presence logit), gap_hat (g = 40 * gap_hat, metres).
  1. v_f = kappa * v_hat if v_hat >= 3 m/s, else NO floor.
  2. lead cap: if pi >= 0.5, v_f = min(v_f, max(0, (g - 2) / 1.5)).
  3. s_f(t_k) = integral_0^{t_k} min(v_f, max(v0, 0) + 1.0 * t) dt   (closed form: a ramp at a_f = 1 m/s^2 up to v_f, then v_f; where v0 >= v_f, s_f = v_f * t).
  4. s'_k = max(s_k, s_f(t_k)), s_k = the plan's cumulative arc length from the origin (the chords of the origin-prepended polyline).
  5. the plan is resampled along its OWN origin-prepended polyline at s'_k, extrapolated beyond the last knot along the last segment's direction.
     If s'_k == s_k at every knot the plan OBJECT is returned unchanged.

Choices the card leaves open, fixed here and tested:
  * a knot with s'_k == s_k keeps its original point (the exact value of the resample there; no interpolation round-off);
  * the last segment of length exactly 0 has no direction: the last NON-ZERO segment's direction is used, and a plan with no non-zero segment at all extrapolates along (1, 0), the ego frame's
    forward axis (both counted: ``dir_fallback``);
  * a non-finite v_hat / pi / gap_hat gives no floor / no cap, and a plan with any non-finite value never binds; a non-finite v0 reads as 0;
  * ``lift`` (the dump field ``lf_lift``) = the 0-3 s (knots 1..15) mean plan speed after F minus before, m/s, with the mean speed = the chord length of the origin-prepended polyline over
    those 15 knots / 3 s, the same function on both plans; exactly 0.0 where F does not bind.
No nav, route, limit, label or situation-classifier output enters: the operator reads (plan, v0, v_hat, pi, gap_hat, kappa) and nothing else.
"""
from __future__ import annotations

import numpy as np

DT = 0.2                         # the cache step, s (t_k = k * DT)
K = 30                           # knots per plan (6 s)
V_BAND_MIN = 3.0                 # m/s: the crawl band's lower bound; below it no floor
PI_MIN = 0.5                     # presence threshold
GAP_SCALE_M = 40.0               # g = GAP_SCALE_M * gap_hat
CAP_D0_M = 2.0                   # standstill gap of the car-following equilibrium
CAP_TG_S = 1.5                   # time gap of the car-following equilibrium
A_F = 1.0                        # m/s^2, the floor's comfort acceleration
K_LIFT = 15                      # knots 1..15 = 0-3 s
K_EXEC = 10                      # instrument only: knots 1..10 = 0-2 s, the horizon the read executes at --replan-steps 10 (a bind confined to later knots changes nothing that is driven)
T_LIFT_S = K_LIFT * DT
SHORT_LAST_M = 1e-3              # instrument only: a last segment shorter than this makes the extrapolation direction a near-noise one
FWD = np.array([1.0, 0.0])
CONSTANTS = {"dt_s": DT, "knots": K, "v_band_min_ms": V_BAND_MIN, "pi_min": PI_MIN, "gap_scale_m": GAP_SCALE_M, "cap_standstill_gap_m": CAP_D0_M, "cap_time_gap_s": CAP_TG_S, "a_f_mps2": A_F}
OPERATOR_TEXT = ("v_f = kappa * v_hat if v_hat >= 3 else none; if pi >= 0.5: v_f = min(v_f, max(0, (40 * gap_hat - 2) / 1.5)); s_f(t_k) = integral_0^t_k min(v_f, max(v0, 0) + 1.0 t) dt; "
                 "s'_k = max(s_k, s_f(t_k)); resample the plan's own origin-prepended polyline at s'_k, extrapolating along the last segment's direction; unchanged => the same object")
T_KNOTS = DT * np.arange(1, K + 1, dtype=np.float64)


def sigmoid(x):
    """float64 logistic, overflow-safe."""
    x = np.asarray(x, np.float64)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def plan_arc(plan):
    """``(full, S)`` of a batch of plans [B, K, 2]: ``full`` [B, K+1, 2] the origin-prepended polyline (float64), ``S`` [B, K+1] its cumulative arc length (S[:, 0] = 0, S[:, k] = s_k)."""
    p = np.asarray(plan, np.float64)
    if p.ndim != 3 or p.shape[1:] != (K, 2):
        raise ValueError(f"plan must be [B, {K}, 2], got {p.shape}")
    full = np.concatenate([np.zeros_like(p[:, :1]), p], axis=1)
    seg = np.linalg.norm(np.diff(full, axis=1), axis=2)
    return full, np.concatenate([np.zeros_like(seg[:, :1]), np.cumsum(seg, axis=1)], axis=1)


def floor_speed(v_hat, pi, gap_hat, kappa):
    """Steps 1-2. Returns ``(v_f, band, cap)`` of arrays [B]: ``v_f`` (float64, NaN where there is no floor), ``band`` (v_hat >= 3 and finite), ``cap`` (the lead cap LOWERED v_f)."""
    v_hat, pi, gap_hat = (np.atleast_1d(np.asarray(x, np.float64)) for x in (v_hat, pi, gap_hat))
    kappa = float(kappa)
    if not np.isfinite(kappa):
        raise ValueError(f"kappa must be finite, got {kappa}")
    with np.errstate(invalid="ignore"):
        band = np.isfinite(v_hat) & (v_hat >= V_BAND_MIN)
        v_f = kappa * v_hat
        g = GAP_SCALE_M * gap_hat
        lead = band & np.isfinite(pi) & np.isfinite(g) & (pi >= PI_MIN)
        cap_speed = np.maximum(0.0, (g - CAP_D0_M) / CAP_TG_S)
        capped = np.where(lead, np.minimum(v_f, cap_speed), v_f)
        cap = lead & (cap_speed < v_f)
    return np.where(band, capped, np.nan), band, cap


def ramp_progress(v_f, v0, t=None, a_f: float = A_F):
    """Step 3, closed form: ``s_f(t) = integral_0^t min(v_f, max(v0, 0) + a_f * tau) dtau`` for v_f [B] (>= 0), v0 [B], t [K] -> [B, K] float64.
    A ramp from max(v0, 0) at a_f until it meets v_f at t* = (v_f - v0') / a_f, then v_f; where v0' >= v_f it is v_f * t (t* = 0)."""
    t = T_KNOTS if t is None else np.asarray(t, np.float64)
    v_f = np.atleast_1d(np.asarray(v_f, np.float64))[:, None]
    v0p = np.maximum(np.nan_to_num(np.atleast_1d(np.asarray(v0, np.float64)), nan=0.0, posinf=0.0, neginf=0.0), 0.0)[:, None]
    t_star = np.maximum(v_f - v0p, 0.0) / a_f
    tr = np.minimum(t[None, :], t_star)
    return v0p * tr + 0.5 * a_f * tr * tr + v_f * (t[None, :] - tr)


def last_direction(full):
    """The unit direction of the LAST segment of a polyline [K+1, 2] with non-zero length; ``(d, fallback)``. ``fallback`` is True when the last segment has length exactly 0 (the last non-zero
    segment's direction is then used) or no segment has any length (FWD)."""
    seg = np.diff(full, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    nz = np.nonzero(ln > 0)[0]
    if len(nz) == 0:
        return FWD.copy(), True
    j = nz[-1]
    return seg[j] / ln[j], bool(j != len(ln) - 1)


def resample_polyline(full, S, s_new):
    """Points of the polyline ``full`` [K+1, 2] (arc length ``S`` [K+1]) at the arc lengths ``s_new`` [n] (>= 0): linear inside, along the last segment's direction beyond S[-1].
    Returns ``(points [n, 2], dir_fallback)``."""
    s_new = np.asarray(s_new, np.float64)
    n_seg = len(S) - 1
    idx = np.clip(np.searchsorted(S, s_new, side="right") - 1, 0, n_seg - 1)       # S[idx] <= s < S[idx + 1] inside, so the segment has length > 0
    inside = s_new < S[-1]
    L = S[idx + 1] - S[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(L > 0, (s_new - S[idx]) / L, 0.0)
    pts = full[idx] + u[:, None] * (full[idx + 1] - full[idx])
    fb = False
    if not inside.all():
        d, fb = last_direction(full)
        pts = np.where(inside[:, None], pts, full[-1][None] + (s_new - S[-1])[:, None] * d[None])
    return pts, fb


def apply_floor_batch(plans, v0, v_hat, pi, gap_hat, kappa):
    """F on a batch. ``plans`` [B, 30, 2] (any float dtype), ``v0`` / ``v_hat`` / ``pi`` / ``gap_hat`` [B], ``kappa`` a scalar. Returns ``(plans_out, info)``: ``plans_out`` is ``plans`` ITSELF (the same
    object) when no row binds, else a copy in the input dtype whose bound rows hold the lifted plans and whose other rows are untouched bit for bit. ``info`` (arrays [B]): ``bind`` (s' != s at any
    knot), ``cap`` (the lead cap lowered v_f), ``vf`` (NaN = no floor), ``vhat``, ``pi``, ``gap`` (g in metres), ``lift`` (m/s), ``dir_fallback`` (bool), ``short_last`` (bool, binds only: the last
    segment is shorter than 1 mm), ``band`` (v_hat >= 3), ``bind_k10`` (instrument: s' != s at any of knots 1..10, the 2 s the read executes)."""
    p64 = np.asarray(plans, np.float64)
    full, S = plan_arc(p64)
    B = p64.shape[0]
    v_hat, pi, gap_hat, v0 = (np.broadcast_to(np.atleast_1d(np.asarray(x, np.float64)), (B,)) for x in (v_hat, pi, gap_hat, v0))
    v_f, band, cap = floor_speed(v_hat, pi, gap_hat, kappa)
    s_f = ramp_progress(np.where(band, v_f, 0.0), v0)
    s_k = S[:, 1:]
    s_new = np.maximum(s_k, s_f)
    with np.errstate(invalid="ignore"):
        knot_bind = band[:, None] & np.isfinite(p64).all(axis=(1, 2))[:, None] & (s_new != s_k)       # s' != s (with the max in place s' >= s, so this is s_f > s_k); a non-finite plan never binds
    row_bind = knot_bind.any(axis=1)
    info = {"bind": row_bind, "cap": cap, "vf": v_f, "vhat": v_hat.copy(), "pi": pi.copy(), "gap": GAP_SCALE_M * gap_hat, "lift": np.zeros(B), "band": band,
            "dir_fallback": np.zeros(B, bool), "short_last": np.zeros(B, bool), "bind_k10": knot_bind[:, :K_EXEC].any(axis=1)}
    if not row_bind.any():
        return plans, info
    out = np.array(plans, copy=True)
    for b in np.nonzero(row_bind)[0]:
        pts, fb = resample_polyline(full[b], S[b], s_new[b])
        new = p64[b].copy()
        new[knot_bind[b]] = pts[knot_bind[b]]
        out[b] = new.astype(out.dtype)
        info["dir_fallback"][b] = fb
        info["short_last"][b] = bool(np.linalg.norm(full[b, -1] - full[b, -2]) < SHORT_LAST_M)
        _, S_out = plan_arc(np.asarray(out[b], np.float64)[None])
        info["lift"][b] = (S_out[0, K_LIFT] - S[b, K_LIFT]) / T_LIFT_S
    return out, info


def apply_floor(plan, v0, v_hat, pi, gap_hat, kappa):
    """F on ONE plan [30, 2]: ``(plan_out, info)``; ``plan_out`` is ``plan`` itself when F does not bind. ``info`` holds python scalars."""
    out, info = apply_floor_batch(np.asarray(plan)[None], [v0], [v_hat], [pi], [gap_hat], kappa)
    res = {k: (v[0].item() if hasattr(v[0], "item") else v[0]) for k, v in info.items()}
    return (plan if not res["bind"] else out[0]), res
