"""VTARGET minting — the tactical set-speed label, and the fix to it.

The V3 vocabulary (``V3_GOAL_VOCABULARY_V1``) mints VTARGET as the *85th-pct
future free-flow speed over 10-20 s*, banded by ``tanitad.lake.vocab.vtarget_band``.
The reference implementation is ``taniteval/planner_p2.py::vtarget_for``, ported
here verbatim as :func:`vtarget_raw` so the two can be diffed.

TWO DEFECTS were measured on the parity corpus before flagship-v1.5 trained on
this label (see the phase-0 validation JSON shipped alongside):

1. **The lookahead floor is never enforced.** ``vtarget_for`` computes
   ``fut = v[L+1 : min(L+VT_LOOK_HI, T)]`` and then only checks
   ``fut.numel() >= VT_MIN_STEPS`` (3 s). ``VT_LOOK_LO`` (10 s) is defined and
   never used. PhysicalAI episodes are **199 frames = 19.9 s**, so the realised
   lookahead is not "10-20 s" — it decays to 3 s at the end of every episode,
   and the label silently changes meaning along the episode.

2. **Pose jitter drives the free-flow gate.** The gate keeps a future step only
   if the step INTO it decelerated less than ``VT_HARD_DECEL`` (1.5 m/s^2).
   Differentiating a jittery speed track at dt=0.1 s amplifies the jitter by 10:
   a +-0.2 m/s wobble fabricates +-2.8 m/s^2 of accel, so the gate fires on
   noise. Worse, it fires ASYMMETRICALLY — a step is dropped exactly when its
   sample came in low — so the surviving sample is biased upward, and then the
   85th percentile is taken on top of that bias.

:func:`vtarget_v2` fixes both: it low-passes the speed track (zero-phase
Savitzky-Golay, order 2 over 1.1 s — preserves real accel/decel ramps, removes
per-frame jitter) BEFORE the gate and the percentile, and it returns an explicit
``valid`` mask plus the realised ``lookahead`` per window so a short-lookahead
label can be routed to the DROPPED token instead of masquerading as a real one.
"""

from __future__ import annotations

import numpy as np

DT = 0.1

# --- planner_p2.py constants, verbatim ---------------------------------------
VT_LOOK_LO = 100          # 10 s — documented floor, NOT enforced by vtarget_raw
VT_LOOK_HI = 200          # 20 s
VT_MIN_STEPS = 30         # 3 s of free-flow samples
VT_PCTL = 0.85
VT_HARD_DECEL = 1.5       # m/s^2

# --- v2 additions ------------------------------------------------------------
SMOOTH_WIN = 11           # 1.1 s Savitzky-Golay window
SMOOTH_POLY = 2
VT_MIN_LOOKAHEAD = 50     # 5 s — the honest floor v2 enforces (see the note)


def savgol(v: np.ndarray, win: int = SMOOTH_WIN,
           poly: int = SMOOTH_POLY) -> np.ndarray:
    """Zero-phase Savitzky-Golay smoother (edge-mirrored), numpy-only.

    Zero-phase matters: a causal filter would shift the speed track in time and
    bias every horizon-indexed label. scipy is not a dependency of the lake.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.shape[0] < win:
        return v.copy()
    half = win // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    a = np.vander(t, poly + 1, increasing=True)
    k = (np.linalg.pinv(a.T @ a) @ a.T)[0]                 # value-at-t=0 weights
    pad = np.concatenate([v[half:0:-1], v, v[-2:-half - 2:-1]])
    return np.convolve(pad, k[::-1], mode="valid")


def vtarget_raw(v: np.ndarray, last: np.ndarray):
    """Verbatim port of ``taniteval/planner_p2.py::vtarget_for``.

    Kept so the defect is reproducible and the v1/v2 diff is auditable. Returns
    ``(v_target [n], valid [n])``.
    """
    t_len = v.shape[0]
    vt = np.empty(last.shape[0], dtype=np.float64)
    valid = np.zeros(last.shape[0], dtype=bool)
    for i, l in enumerate(last):
        hi = min(l + VT_LOOK_HI, t_len)
        fut = v[l + 1:hi]
        if fut.shape[0] >= VT_MIN_STEPS:
            acc = (fut[1:] - fut[:-1]) / DT
            keep = np.ones(fut.shape[0], dtype=bool)
            keep[1:] = acc > -VT_HARD_DECEL
            ff = fut[keep]
            if ff.shape[0] >= VT_MIN_STEPS:
                vt[i] = np.quantile(ff, VT_PCTL)
                valid[i] = True
                continue
        vt[i] = v[l]
    return vt, valid


def vtarget_v2(v: np.ndarray, last: np.ndarray,
               min_lookahead: int = VT_MIN_LOOKAHEAD,
               smooth: bool = True):
    """The fixed mint. Returns ``(v_target, valid, lookahead, v_smoothed)``.

    * the speed track is low-passed before the free-flow gate AND before the
      percentile, so both operate on driver intent rather than on jitter;
    * ``valid`` is False when the realised lookahead is shorter than
      ``min_lookahead`` steps or the free-flow sample is shorter than
      ``VT_MIN_STEPS`` — the caller routes those to the DROPPED token instead of
      silently substituting the current speed;
    * ``v_target`` still carries the hold-speed fallback so a cost function that
      needs a number always has one, but ``valid`` says whether to believe it.
    """
    vs = savgol(v) if smooth else np.asarray(v, dtype=np.float64)
    t_len = vs.shape[0]
    n = last.shape[0]
    vt = np.empty(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    look = np.zeros(n, dtype=np.int64)
    for i, l in enumerate(last):
        hi = min(l + VT_LOOK_HI, t_len)
        fut = vs[l + 1:hi]
        look[i] = fut.shape[0]
        if fut.shape[0] >= max(min_lookahead, VT_MIN_STEPS):
            acc = (fut[1:] - fut[:-1]) / DT
            keep = np.ones(fut.shape[0], dtype=bool)
            keep[1:] = acc > -VT_HARD_DECEL
            ff = fut[keep]
            if ff.shape[0] >= VT_MIN_STEPS:
                vt[i] = np.quantile(ff, VT_PCTL)
                valid[i] = True
                continue
        vt[i] = vs[l]
    return vt, valid, look, vs
