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

⛔ **A THIRD DEFECT — and it is the one that blocks the INPUT, not the label.**
Both mints read ``v[l+1 : l+200]``, a window that **strictly contains the scored
horizon** ``[t, t+2 s]``: the decoder's waypoints live at ``horizons=(5,10,15,20)``
steps and the manoeuvre head's own label is ``dv = v(t+2 s) - v(t)``, i.e. pose
``l+20``. So VTARGET is computed from a **superset of the thing being measured**.
As a *label* that is fine — the PI's 2026-08-03 ruling is explicit that labels may
use ego state and future poses. As an **input at inference** it is the nav-echo
defect again (flagship-v1's route head was an exact bijection of the nav it was
fed, 369/369, and scored 1.0000), which is why the D-SEL stream refused to wire a
``cond_vtarget`` seam for REF-C and escalated instead
(``PREREG_D-SEL_REFC_SELECTION_SURFACE.md`` §9.1).

:func:`vtarget_guarded` is the mint that closes it: the read window starts at
``l + VT_GUARD_STEPS + 1``, so the scored horizon is **excised by construction**
and the excision is checkable from index arithmetic rather than promised in prose.
⚠️ Excision is **necessary and not sufficient** — a speed track is autocorrelated,
so ``v(t+3 s)`` still predicts ``v(t+1 s)``. The residual is a MEASURED quantity,
not an assumption: see ``…/incoming/2026-08-04-target-speed/``. The deployable form
of the lever is therefore a **PREDICTED** target speed (a head trained on this
label, reading only inference-legal inputs), never the label supplied at inference.
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

# --- guarded mint ------------------------------------------------------------
#: Steps of the ego future that the SCORED horizon occupies, and that a
#: leak-guarded label must therefore not read. **Derived, not chosen**:
#: ``RefCConfig.trajectory.horizons[-1] == 20`` (2.0 s @ 10 Hz) is the last
#: predicted waypoint, ``lead_source.K_MAX == 20`` is the same bound in the eval
#: window grid, and the manoeuvre head's label ``dv = v(t+2 s) - v(t)`` reads
#: exactly pose ``l + 20``. Raising a scored horizon MUST raise this constant.
VT_GUARD_STEPS = 20


def savgol(v: np.ndarray, win: int = SMOOTH_WIN,
           poly: int = SMOOTH_POLY) -> np.ndarray:
    """Zero-phase Savitzky-Golay smoother (edge-mirrored), numpy-only.

    Zero-phase matters: a causal filter would shift the speed track in time and
    bias every horizon-indexed label. scipy is not a dependency of the lake.

    Order-2 reproduces a linear ramp EXACTLY in the interior, so genuine
    acceleration survives untouched — the property that makes it safe to smooth
    before a hard-decel gate. The even-mirror padding is not itself linear, so
    the first/last ``win // 2`` samples carry a bounded artefact (< one sample
    step of the underlying ramp; measured max 0.049 m/s on a 10 m/s ramp over
    199 frames). Harmless where it is used: the leading samples fall inside the
    window-8 warm-up and the trailing ones only appear in windows whose
    lookahead is already below the :func:`vtarget_v2` floor. Pinned by
    ``tests/test_flagship_v15.py`` so the behaviour cannot drift silently.
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


def vtarget_guarded(v: np.ndarray, last: np.ndarray, *,
                    guard_steps: int = VT_GUARD_STEPS,
                    min_lookahead: int = VT_MIN_LOOKAHEAD,
                    smooth: bool = True):
    """The LEAK-GUARDED mint. Returns ``(v_target, valid, lookahead, v_smoothed)``.

    Identical to :func:`vtarget_v2` in every respect except two, both of which
    exist to make the label admissible to a *supervised head* whose sibling heads
    are scored on ``[t, t + guard_steps * DT]``:

    1. **The read window is** ``vs[l + guard_steps + 1 : min(l + VT_LOOK_HI, T)]``.
       The scored set is ``{l, …, l + guard_steps}``; the read set starts one step
       past it, so the two are **disjoint by index arithmetic**. With the defaults
       that is ``[t + 2.1 s, t + 20 s]`` against a scored ``[t, t + 2.0 s]``.
       ⚠️ The far end is deliberately NOT shifted: the label's documented meaning
       is "the free-flow speed over the next ~20 s", and sliding the window out to
       ``t + 22 s`` would change the quantity as well as guard it.

    2. **The fallback is the RAW current speed** ``v[l]``, not the smoothed
       ``vs[l]``. ``savgol`` is zero-phase, so ``vs[l]`` is a function of
       ``v[l-5 … l+5]`` — it reads 0.5 s into the future, which would leave a
       small leak living inside the fallback of a function whose entire purpose is
       to remove one. ``v[l]`` is the ego speed the model already receives as
       ``v0`` at inference, so the fallback carries no information the model does
       not already hold. ⚠️ ``valid`` is False on every fallback window; a caller
       that routes those to the DROPPED token never sees the number at all.

    ⛔ **What this does NOT buy.** Disjoint windows are not independent windows.
    A speed track is strongly autocorrelated, so this label still carries real
    information about ``[t, t+2 s]`` — measurably so. The guard makes the label a
    legitimate *supervision target*; it does not make the label safe to **supply**
    at inference. The deployable lever is a head that PREDICTS this quantity from
    inference-legal inputs. Both statements travel with every number derived here.
    """
    guard = int(guard_steps)
    if guard < 0:
        raise ValueError(f"guard_steps must be >= 0, got {guard}")
    v = np.asarray(v, dtype=np.float64)
    vs = savgol(v) if smooth else v
    t_len = vs.shape[0]
    n = last.shape[0]
    vt = np.empty(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    look = np.zeros(n, dtype=np.int64)
    for i, l in enumerate(last):
        hi = min(l + VT_LOOK_HI, t_len)
        lo = min(l + guard + 1, hi)
        fut = vs[lo:hi]
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
        vt[i] = v[l]
    return vt, valid, look, vs


def read_window(last_index: int, t_len: int, guard_steps: int = VT_GUARD_STEPS
                ) -> tuple[int, int]:
    """``(lo, hi)`` pose indices :func:`vtarget_guarded` reads for one window.

    Exported so the admissibility check is a *computation over the same code* the
    label uses, not a re-derivation of its arithmetic in a report. The scored set
    is ``range(last_index, last_index + guard_steps + 1)``; a guard is proved when
    ``set(range(lo, hi))`` and that set are disjoint.
    """
    hi = min(int(last_index) + VT_LOOK_HI, int(t_len))
    return min(int(last_index) + int(guard_steps) + 1, hi), hi
