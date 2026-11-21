"""imag_plan.py -- IMAGPLAN (card PREREG_V8F_PROOF_ADD40): planning by imagination at T1-kin. Pure numpy functions, batched over windows.

At a replan tick the model's plan P (30 knots at 0.2 s, ego frame of the tick, origin not in the array) gets speed-scaled forms on its own path:
  progress a_c(i) = max(c * a_P(i), a_B(i)),  a_B(i) = progress under braking at B_MAX from the measured speed v0   (card section 2)
  knots = the plan polyline sampled at a_c(i) (dist_rule.plan_path / sample_path), v_c(i) = (a_c(i) - a_c(i-1)) / 0.2 s.
Candidate c is safe iff at both horizons h = 1.0 s, 2.0 s (knots 5, 10)  gap_c(h) >= max(D, TAU * v_c(h)); the executed plan is the safe candidate with
the LARGEST c, and c = 0.0 when none is safe.  The gap forecast gap_c(h) comes from one of three sources (card section 2):
  cand-cv     g_src + (a_l - a_m) + v_L * h - a_c(h)           (head present read + constant-velocity forecast of the lead)
  cand-label  the same with the label gap and the label closing speed (privileged)
  cand-imag   imag_fn(candidate plans): the world model rolled out under the candidate's own actions, lead head read at 1.0 s and 2.0 s, + (a_l - a_m)
The model parts (rollout, head) live in the wrapper; here `imag_fn` is a callback, so that every rule of the card is testable with a planted forecast.
`candidate_actions` converts a candidate plan into the level-1 actions of the corpus by the pinned tool's own functions (IMAGPLAN_DEFS A1).
Tier T1-kin (kinematic rollout on the model's own actions, logged frames); a diagnostic, never driving performance.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, NamedTuple, Optional

sys.dont_write_bytecode = True      # never write __pycache__ next to the sibling tools this file imports
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "distrule"), os.path.join(HERE, "..", "imaghead")):
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import dist_rule as R  # noqa: E402  (plan_path, sample_path, DT; one version, never a copy)

DT = R.DT                    # 0.2 s
D = 3.0                      # m, card section 2 (the constants of card ADD26, unchanged)
TAU = 1.2                    # s
B_MAX = 6.0                  # m/s2
assert (D, TAU, B_MAX) == (R.D_M, R.TAU_S, R.B_MAX), "constants must equal those of dist_rule.py (card ADD26)"
G_MAX_M = R.G_MAX_M          # 60 m: the gate
CAND_CARD = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
KNOT_H = (5, 10)             # knot indices of the two check horizons
H_S = (1.0, 2.0)             # seconds of the two horizons
N_ROLL = 10                  # rollout steps = 2.0 s
SOURCES = ("cand-imag", "cand-cv", "cand-label", "cand-imagkin")
IMAG_SOURCES = ("cand-imag", "cand-imagkin")               # the sources that roll the world model out (cand-imagkin: note PREREG_V8F_PROOF_ADD40_N4 section 3)
FLOORS = ("card", "cap-only")
FALLBACKS = ("zero", "none")
FAMILIES = ("scale", "decel")
DECEL_CARD = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)      # m/s2, note PREREG_V8F_PROOF_ADD40_N3 section 2


# ============================================================================================ candidates
def parse_cand_set(s) -> tuple:
    """'1.0,0.9,...' or an iterable -> a tuple of distinct floats in [0, 1], descending."""
    vals = [float(x) for x in (s.split(",") if isinstance(s, str) else s) if str(x).strip() != ""]
    if not vals or any((not np.isfinite(v)) or v < 0.0 or v > 1.0 for v in vals):
        raise ValueError(f"candidate set must be non-empty values in [0, 1], got {s!r}")
    return tuple(sorted(set(vals), reverse=True))


def parse_decel_set(s) -> tuple:
    """decel family (note ADD40_N3): '0,0.5,...' or an iterable -> a tuple of distinct finite deceleration values d >= 0 m/s2, ASCENDING (smallest d first)."""
    vals = [float(x) for x in (s.split(",") if isinstance(s, str) else s) if str(x).strip() != ""]
    if not vals or any((not np.isfinite(v)) or v < 0.0 for v in vals):
        raise ValueError(f"decel set must be non-empty finite values >= 0, got {s!r}")
    return tuple(sorted(set(vals)))


def effective_set(cset, fallback="zero", family="scale") -> tuple:
    """The candidates actually evaluated.  scale: the set (descending), plus c = 0.0 when the fallback needs it (card: none safe -> c = 0.0).
    decel: the set ASCENDING and nothing added (the fallback zero is the largest d OF THE SET)."""
    if family == "decel":
        return tuple(sorted(set(float(c) for c in cset)))
    if family != "scale":
        raise ValueError(f"family must be one of {FAMILIES}")
    cs = tuple(sorted(set(float(c) for c in cset), reverse=True))
    if fallback == "zero" and 0.0 not in cs:
        cs = cs + (0.0,)
    return cs


def braking_progress(v0, n=30, b_max=B_MAX, dt=DT):
    """a_B(i), i = 0..n [B, n+1]: arc length under constant braking at b_max from speed max(v0, 0) (then standing)."""
    v0 = np.maximum(np.asarray(v0, np.float64), 0.0)
    t = np.arange(n + 1) * dt
    tt = np.minimum(t[None, :], (v0 / b_max)[:, None])
    return v0[:, None] * tt - 0.5 * b_max * tt ** 2


def plan_arcs(plan):
    """plan [B,N,2] -> (aP [B,N+1] cumulative arc with aP[:,0] = 0, [(P, L, ext)] per window), as dist_rule.plan_path."""
    p64 = np.asarray(plan, np.float64)
    B, N = p64.shape[:2]
    aP = np.zeros((B, N + 1))
    paths = []
    for b in range(B):
        P, L, ext = R.plan_path(p64[b])
        aP[b] = L
        paths.append((P, L, ext))
    return aP, paths


def cand_progress(aP, aB, cset, floor="card"):
    """a_c(i) [B, C, N+1].  card: max(c * a_P, a_B) (can lift a plan that brakes harder than B_MAX, IMAGPLAN_DEFS E1); cap-only: min(a_P, that) (never above the plan's own progress)."""
    if floor not in FLOORS:
        raise ValueError(f"floor must be one of {FLOORS}")
    c = np.asarray(cset, np.float64)[None, :, None]
    a = np.maximum(c * aP[:, None, :], aB[:, None, :])
    if floor == "cap-only":
        a = np.minimum(aP[:, None, :], a)
    return a


def decel_step_speeds(aP, dset):
    """decel family: v_d(i) = max(0, v_P(i) - d * DT * i), i = 1..N, v_P(i) = (a_P(i) - a_P(i-1)) / DT.  aP [B, N+1], dset [C] -> [B, C, N]."""
    ds = np.asarray(dset, np.float64)[None, :, None]
    vP = np.diff(aP, axis=1) / DT
    i = np.arange(1, aP.shape[1])[None, None, :]
    return np.maximum(0.0, vP[:, None, :] - ds * DT * i)


def decel_progress(aP, dset):
    """a_d(i) [B, C, N+1] = cumulative sum of v_d * DT (a_d(0) = 0).  d = 0 is a_P itself, bit for bit; never above a_P (rounding clamped), so no floor option applies."""
    vd = decel_step_speeds(aP, dset)
    a = np.zeros((aP.shape[0], len(dset), aP.shape[1]))
    a[..., 1:] = np.cumsum(vd * DT, axis=-1)
    a = np.minimum(a, aP[:, None, :])
    return np.where((np.asarray(dset, np.float64) == 0.0)[None, :, None], aP[:, None, :], a)


def select_decel(dset, safe, fallback="zero"):
    """decel family: safe [B,C] -> (idx [B] of the SMALLEST safe d, or -1; d [B]: that d, or max(dset) when none is safe (fallback zero), or NaN (fallback none))."""
    ds = np.asarray(dset, np.float64)
    idx = np.argmin(np.where(safe, ds[None, :], np.inf), axis=1)
    anys = safe.any(1)
    idx = np.where(anys, idx, -1)
    d = np.where(anys, ds[np.maximum(idx, 0)], ds.max() if fallback == "zero" else np.nan)
    return idx, d


def knot_speed(a, i):
    """v_c(i) = (a_c(i) - a_c(i-1)) / 0.2 for a [..., N+1]."""
    return (a[..., i] - a[..., i - 1]) / DT


def candidate_plan(plan_b, path, a_c, aP_b):
    """One window: the plan polyline sampled at a_c[1:] -> [N,2] in the plan's dtype; where a_c(i) == a_P(i) the original knot is kept (bit for bit)."""
    P, L, ext = path
    p64 = np.asarray(plan_b, np.float64)
    pts = R.sample_path(P, L, ext, a_c[1:])
    keep = a_c[1:] == aP_b[1:]
    pts[keep] = p64[keep]
    return pts.astype(np.asarray(plan_b).dtype)


def safe_flags(gap_h, v_h, d=D, tau=TAU):
    """gap_h, v_h [..., 2] (horizons 1.0 s, 2.0 s) -> safe [...]: both horizons gap >= max(d, tau * v), all finite (a non-finite forecast is never safe)."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(gap_h) & np.isfinite(v_h) & (gap_h >= np.maximum(d, tau * v_h))
    return ok.all(-1)


def select(cset, safe, fallback="zero"):
    """safe [B,C] -> (idx [B] of the largest safe c, or -1; c [B]: that c, or 0.0 when none is safe (fallback zero), or NaN (fallback none))."""
    cs = np.asarray(cset, np.float64)
    masked = np.where(safe, cs[None, :], -np.inf)
    idx = np.argmax(masked, axis=1)
    anys = safe.any(1)
    idx = np.where(anys, idx, -1)
    c = np.where(anys, cs[np.maximum(idx, 0)], 0.0 if fallback == "zero" else np.nan)
    return idx, c


# ============================================================================================ the rule
class PlanOut(NamedTuple):
    plan: np.ndarray        # [B,N,2]; the input object itself when no window changed
    active: np.ndarray      # [B] bool
    chosen_idx: np.ndarray  # [B] index into `cset` (the effective set); -1 = no safe candidate; -2 = inactive
    chosen_c: np.ndarray    # [B] the executed c (0.0 on fallback zero; NaN on inactive or fallback none with none safe)
    any_safe: np.ndarray    # [B] bool
    safe: np.ndarray        # [B,C] bool (False on inactive rows)
    gap_h: np.ndarray       # [B,C,2] gap forecast at 1.0 s, 2.0 s (NaN on inactive rows)
    v_h: np.ndarray         # [B,C,2] candidate speed at the horizons
    a_h: np.ndarray         # [B,C,2] candidate progress at the horizons
    imag_raw: np.ndarray    # [B,C,2] raw imagined gap before the shift and the own-progress term (NaN unless cand-imag)
    n_lift: np.ndarray      # [B] steps where the executed candidate is above the plan's own progress (card floor only)
    n_nonfinite: np.ndarray # [B] candidates whose forecast was non-finite
    changed: np.ndarray     # [B] bool
    cset: tuple
    family: str = "scale"   # decel: `cset` holds the deceleration values d (ascending) and `chosen_c` the chosen d


def imag_plan_rule(plan, v0, active, g_src, a_off, v_L, source, cset=CAND_CARD, floor="card", fallback="zero", imag_fn: Optional[Callable] = None,
                   const_gap=None, shift_m=0.0, d=D, tau=TAU, family="scale") -> PlanOut:
    """The rule of card ADD40 section 2 on a batch of tick plans.
    plan [B,N,2]; v0 [B] measured speed; active [B] the gate (already applied by the caller); g_src [B] source gap; a_off [B] = a_l(k) - a_m(k); v_L [B] = max(0, v_l + c_src).
    imag_fn(cand_plans [Ba,C,N,2], idx [Ba]) -> imagined gap [Ba,C,2] in metres at 1.0 s and 2.0 s (cand-imag).
    const_gap M: every forecast gap is M (control p1); shift_m: added to every imagined gap (control p6).
    family 'decel' (note ADD40_N3): cset = deceleration values d; progress = decel_progress; the safe candidate with the SMALLEST d is executed, none safe -> the largest d of the set (fallback zero)
    or the plan (fallback none).  `floor` is IGNORED for this family (a candidate is never ahead of the plan, so the floor has nothing to lift); PlanOut.chosen_c holds the chosen d."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS}")
    plan_in = plan
    plan = np.asarray(plan)
    B, N = plan.shape[0], plan.shape[1]
    cs = effective_set(cset, fallback, family)
    C = len(cs)
    active = np.asarray(active, bool)
    zi = np.zeros(B, np.int64)
    nanc = np.full((B, C, 2), np.nan)
    empty = PlanOut(plan_in, active, np.full(B, -2, np.int64), np.full(B, np.nan), np.zeros(B, bool), np.zeros((B, C), bool), nanc.copy(), nanc.copy(), nanc.copy(), nanc.copy(),
                    zi.copy(), zi.copy(), np.zeros(B, bool), cs, family)
    if not active.any():
        return empty
    ia = np.nonzero(active)[0]
    g_src, a_off, v_L, v0 = (np.asarray(x, np.float64)[ia] for x in (g_src, a_off, v_L, v0))
    aP, paths = plan_arcs(plan[ia])
    aB = braking_progress(v0, N)
    a_c = decel_progress(aP, cs) if family == "decel" else cand_progress(aP, aB, cs, floor)      # [Ba, C, N+1]
    cand = np.stack([np.stack([candidate_plan(plan[b], paths[k], a_c[k, ci], aP[k]) for ci in range(C)]) for k, b in enumerate(ia)])   # [Ba, C, N, 2]
    a_h = a_c[:, :, list(KNOT_H)]
    v_h = np.stack([knot_speed(a_c, i) for i in KNOT_H], -1)
    h = np.asarray(H_S)[None, None, :]
    imag_raw = np.full(a_h.shape, np.nan)
    if const_gap is not None:
        gap = np.full(a_h.shape, float(const_gap))
    elif source == "cand-imag":
        if imag_fn is None:
            raise ValueError("cand-imag needs imag_fn")
        imag_raw = np.asarray(imag_fn(cand, ia), np.float64)
        gap = imag_raw + float(shift_m) + a_off[:, None, None]
    elif source == "cand-imagkin":
        # note ADD40_N4 section 3: the LEVEL is the imagined gap under the plan (the d = 0 candidate), the difference between candidates is the known one (their own progress).
        if family != "decel" or 0.0 not in cs:
            raise ValueError("cand-imagkin needs the decel family with d = 0 in the set (the plan is the candidate that is imagined)")
        if imag_fn is None:
            raise ValueError("cand-imagkin needs imag_fn")
        i0 = cs.index(0.0)
        imag_raw = np.asarray(imag_fn(cand, ia), np.float64)                   # every candidate is rolled and logged; only column i0 enters the decision
        gap = imag_raw[:, [i0], :] + float(shift_m) + a_off[:, None, None] + (a_h[:, [i0], :] - a_h)
    else:
        gap = (g_src + a_off)[:, None, None] + v_L[:, None, None] * h - a_h
    safe = safe_flags(gap, v_h, d, tau)
    with np.errstate(invalid="ignore"):
        nonfin = (~np.isfinite(gap).all(-1)).sum(1)
    if family == "decel":
        idx, c = select_decel(cs, safe, fallback)
        zero_i = int(np.argmax(cs))                                            # the largest d of the set (fallback zero)
    else:
        idx, c = select(cs, safe, fallback)
        # the executed candidate: the chosen one; fallback zero with none safe = the c = 0.0 candidate (always in the effective set)
        zero_i = cs.index(0.0) if 0.0 in cs else -1
    exec_i = np.where(idx >= 0, idx, zero_i if fallback == "zero" else -1)
    out = plan.copy()
    changed = np.zeros(B, bool)
    n_lift = zi.copy()
    for k, b in enumerate(ia):
        if exec_i[k] < 0:
            continue
        new = cand[k, exec_i[k]]
        changed[b] = bool((new != plan[b]).any())
        out[b] = new
        n_lift[b] = int((a_c[k, exec_i[k], 1:] > aP[k, 1:]).sum())
    full = lambda shp, fill, dt=float: np.full((B,) + shp, fill, dt)          # noqa: E731
    f_idx, f_c = full((), -2, np.int64), full((), np.nan)
    f_idx[ia], f_c[ia] = idx, c
    f_safe, f_gap, f_v, f_a, f_imag = (full((C,), False, bool), full((C, 2), np.nan), full((C, 2), np.nan), full((C, 2), np.nan), full((C, 2), np.nan))
    f_safe[ia], f_gap[ia], f_v[ia], f_a[ia], f_imag[ia] = safe, gap, v_h, a_h, imag_raw
    f_any, f_nf = np.zeros(B, bool), zi.copy()
    f_any[ia], f_nf[ia] = safe.any(1), nonfin
    return PlanOut(out if changed.any() else plan_in, active, f_idx, f_c, f_any, f_safe, f_gap, f_v, f_a, f_imag, n_lift, f_nf, changed, cs, family)


# ============================================================================================ actions of a candidate (IMAGPLAN_DEFS A1)
def candidate_track_rows(T, st, plan_ego, m=N_ROLL):
    """The native 10 Hz track rows 2j .. 2(j+m) the pinned tool would write if it executed `plan_ego` now (`WindowState.commit_plan`, v8f_t1_eval.py:181-193), without writing them.
    T = the pinned module (its envelope, plan_to_world, hermite_frames are called, not copied); st = a WindowState (or anything with j, d['pose'], track)."""
    j = int(st.j)
    pose = st.d["pose"][j]
    pe, _ = T.envelope(np.asarray(plan_ego, np.float64))
    W = T.plan_to_world(pe, pose)
    knots = np.vstack([pose[None, :2], W])
    xy, psi, v = T.hermite_frames(st.track[2 * (j - 1), :2], knots, m, float(pose[2]))
    sl = np.array(st.track[2 * j: 2 * (j + m) + 1], np.float64, copy=True)
    sl[1:, 0:2], sl[1:, 2], sl[1:, 3] = xy, psi, v
    return sl


ACTION_DEFS = ("logged", "executed")


KAPPA_MAX = 0.2              # 1/m: the tightest turn of a car (radius 5 m); note PREREG_V8F_PROOF_ADD40_N2


def limit_yaw(act, v_fr, kappa=KAPPA_MAX):
    """act [m, 2] raw actions (acceleration, yaw rate), v_fr [m] the track speed at the same cache frames -> (act with |yaw rate| <= kappa * |v|, number of limited steps).
    Why (MEASURED, note ADD40_N2): the pinned tool rebuilds the heading of a track from the tangents of the knots; where a path jitters back and forth at standstill the heading
    flips by pi and the yaw channel reads pi / 0.1 s = 31.4 rad/s (17 of 2,399 logged windows; the corpus maximum is 0.55 rad/s). A car cannot yaw faster than kappa * v.
    A value inside the bound is returned unchanged, bit for bit; a NaN speed leaves the step to the caller's nan_to_num."""
    out = np.array(act, np.float64, copy=True)
    lim = float(kappa) * np.abs(np.asarray(v_fr, np.float64))
    w = out[:, 1]
    with np.errstate(invalid="ignore"):
        over = np.abs(w) > lim
    out[over, 1] = np.sign(w[over]) * lim[over]
    return out, int(over.sum())


def candidate_actions(T, st, plan_ego, m=N_ROLL, definition="logged", yaw_limit=True, return_n=False):
    """Raw level-1 actions [m, 2] = (acceleration [m/s2], ch1 = yaw rate [rad/s]) of cache frames j .. j+m-1 for a candidate plan, float32 (IMAGPLAN_DEFS A1, A2).
    definition 'logged' (default) = the corpus' definition, MEASURED on 12 eval clips: the logged acceleration of frame f is the FORWARD difference of the cache-step speed,
      a(f) = (v(f+1) - v(f)) / 0.2 s (cache `a` vs this: max |diff| 1.2e-7; cache `v` = native track speed at the even rows: max |diff| 0.0); ch1(f) = the native track's one-frame backward
      yaw difference (`ego_channels_native`, v8f_mini_data.py:394-401, the yaw-mode spelling of the corpus).  v(j) is the measured track speed, v(j+1..j+m) come from the candidate's executed-form track.
    definition 'executed' = what the pinned tool writes into d['a'] when it executes the plan (`WindowState._finish`, v8f_t1_eval.py:201-224): a = one-frame (0.1 s) BACKWARD track difference; frame j is measured.
    The candidate track is the one the pinned tool would write (`candidate_track_rows`): the same Hermite frames, so a chosen candidate's actions are the ones the executed world then shows."""
    if definition not in ACTION_DEFS:
        raise ValueError(f"definition must be one of {ACTION_DEFS}")
    j = int(st.j)
    sl = candidate_track_rows(T, st, plan_ego, m)
    ch = T.D.ego_channels_native(sl)[0::2]                 # [m+1, 3] (v, dv/dt, dyaw/dt) at cache frames j .. j+m
    out = np.zeros((m, 2))
    out[0, 1] = st.d["ch1"][j]
    out[1:, 1] = ch[1:m, 2]
    if definition == "logged":
        v = sl[0::2, 3]                                    # track speed at the cache frames j .. j+m
        out[:, 0] = (v[1:m + 1] - v[0:m]) / DT
    else:
        out[0, 0] = st.d["a"][j]
        out[1:, 0] = ch[1:m, 1]
    n_lim = 0
    if yaw_limit:
        out, n_lim = limit_yaw(out, sl[0::2, 3][0:m])      # the track speed at the cache frames j .. j+m-1 (note ADD40_N2)
    out = np.nan_to_num(out).astype(np.float32)            # float32 = the dtype of the arrays the pinned tool writes (d['a'], d['ch1']) and of `item`'s act
    return (out, n_lim) if return_n else out


def decel_actions(T, st, plan_ego, d, m=N_ROLL, definition="logged", yaw_limit=True, return_n=False, base=None):
    """Raw level-1 actions [m, 2] (float32) of the decel candidate d of the plan `plan_ego` (note ADD40_N3 section 2): the PLAN's own action sequence (`candidate_actions`, with the yaw bound)
    with ONLY the acceleration channel replaced by the forward difference of the candidate's speed at the cache steps: a(j+s) = (v_d(s+1) - v_d(s)) / DT, s = 0..m-1,
    v_d(0) = the measured track speed v(j), v_d(s) = max(0, v_plan_track(s) - d * DT * s) with v_plan_track the plan's track speeds at the cache frames j+s (`candidate_track_rows`).
    The yaw channel is the plan's, untouched.  d = 0: the plan's actions, bit for bit.  The replacement applies to both action definitions (for 'executed' it also replaces frame j's measured value).
    base = (plan_actions [m,2], n_lim, v_track [m+1]) from a previous call, to avoid recomputing the plan's rows for every d of one window."""
    if base is None:
        act, n_lim = candidate_actions(T, st, plan_ego, m, definition, yaw_limit, return_n=True)
        v = np.asarray(candidate_track_rows(T, st, plan_ego, m)[0::2, 3], np.float64)
        base = (act, n_lim, v)
    act, n_lim, v = base
    out = np.array(act, copy=True)
    if float(d) != 0.0:
        vd = v.copy()
        vd[1:] = np.maximum(0.0, v[1:] - float(d) * DT * np.arange(1, m + 1))
        out[:, 0] = np.nan_to_num((vd[1:m + 1] - vd[0:m]) / DT).astype(np.float32)
    return (out, n_lim) if return_n else out


# ============================================================================================ reading the p4 / p5 logs
def p4_stats(act_cand, act_log):
    """act_cand, act_log [n, 10, 2]: per channel Pearson correlation over all (window, step) pairs and the median absolute difference of the acceleration channel (card control p4)."""
    a, b = np.asarray(act_cand, np.float64), np.asarray(act_log, np.float64)
    ok = np.isfinite(a).all((1, 2)) & np.isfinite(b).all((1, 2))
    a, b = a[ok], b[ok]
    corr = [float(np.corrcoef(a[..., c].ravel(), b[..., c].ravel())[0, 1]) for c in (0, 1)]
    return {"n": int(ok.sum()), "corr_acc": corr[0], "corr_yaw": corr[1], "median_abs_diff_acc": float(np.median(np.abs(a[..., 0] - b[..., 0]))),
            "median_abs_diff_yaw": float(np.median(np.abs(a[..., 1] - b[..., 1]))),
            "bar": {"corr_min": 0.99, "median_abs_diff_acc_max": 0.1}}
