"""v8f_t1_readout.py -- the causal readouts of the T1-kin own history, promoted from the integ10k study wrapper (card PREREG_V8F_ROLLHIST, item N1).

``backward_frames``, ``causal_frames`` and ``causal_g_frames`` are the functions of card ADD63 (ACCGAIN10K), copied from
``integ10k/v8f_t1_eval_reader_rule_rzg.py`` (at the tip 7b26a8ac, lines 251-296) with their arithmetic UNCHANGED, so that a training chain does not import a
Research-Lab study file by path. Each takes the pinned frames function ``orig`` (``v8f_t1_eval.hermite_frames``) as its first argument and returns
``(xy, psi, v)`` for the executed frames of one committed plan:

  backward_frames   the pinned function given the executed knots only (its last tangent is then a backward difference)
  causal_frames     positions and headings as pinned; the speed column from executed steps only (acc "step" | "zero" | "window")
  causal_g_frames   the speed column ``s[i] - 0.5 x gain x (s[i] - s[i-1])`` at the half step before executed step i; gain 1.0 routes through ``causal`` and
                    gain 0.0 through ``causal_z`` (identities, array for array); ``path_exec`` takes positions and headings from ``backward_frames``

``readout_frames(orig, readout, gain, path_exec)`` (new, no arithmetic) returns the frames function of the named readout; ``pinned`` returns ``orig`` ITSELF.
The pinned tool (``v8f_t1_eval.py``) is not edited; ``v8f_dagger_bank.py`` installs the result in its place for the duration of one generator run.
"""
from __future__ import annotations

import numpy as np

DT_S = 0.2                                          # seconds per cache step (the pinned tool: 30 steps = 6 s)
READOUTS = ("pinned", "causal_g")


def backward_frames(orig, k_prev, knots, m, psi0):
    """the pinned frames function on the executed knots only: its last tangent is then the backward difference."""
    return orig(k_prev, np.asarray(knots, np.float64)[:m + 1], m, psi0)


def causal_frames(orig, k_prev, knots, m, psi0, acc="step"):
    """positions and headings of the pinned frames function; the speed column from executed steps only. s[0] = the step into the current knot (the last executed step
    before this block, or the logged step before the window), s[i] = executed step i; frame 2i-1 (half step) = (s[i-1] + s[i]) / 2, frame 2i (knot) = s[i]."""
    xy, psi, v = orig(k_prev, knots, m, psi0)
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])
    s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S
    v = np.asarray(v, np.float64).copy()
    for i in range(1, m + 1):
        if acc == "zero":
            v[2 * (i - 1)] = s[i]
        elif acc == "window":
            n_w = min(i, 5)
            v[2 * (i - 1)] = s[i] - 0.5 * DT_S * (s[i] - s[i - n_w]) / (DT_S * n_w)
        else:
            v[2 * (i - 1)] = 0.5 * (s[i - 1] + s[i])
        v[2 * (i - 1) + 1] = s[i]
    return xy, psi, v


def causal_g_frames(orig, k_prev, knots, m, psi0, gain, path_exec=False, force_general=False):
    """the readout causal_g. Speed column: at the knot of executed step i the step speed s[i], at the half step before it s[i] - 0.5 * gain * (s[i] - s[i-1]); s as in
    causal_frames. gain == 1.0 -> the formula of causal, gain == 0.0 -> the formula of causal_z (both unchanged, unless force_general, used by the tool check only);
    other gains -> the general formula. path_exec: positions and headings from the pinned frames function given the executed knots only (backward_frames)."""
    base = orig
    if path_exec:
        def base(kp, kn, mm, p0):
            return backward_frames(orig, kp, kn, mm, p0)
    g = float(gain)
    if not force_general and g == 1.0:
        return causal_frames(base, k_prev, knots, m, psi0, "step")
    if not force_general and g == 0.0:
        return causal_frames(base, k_prev, knots, m, psi0, "zero")
    xy, psi, v = base(k_prev, knots, m, psi0)
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])
    s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S
    v = np.asarray(v, np.float64).copy()
    for i in range(1, m + 1):
        v[2 * (i - 1)] = s[i] - 0.5 * g * (s[i] - s[i - 1])
        v[2 * (i - 1) + 1] = s[i]
    return xy, psi, v


def readout_frames(orig, readout="pinned", gain=None, path_exec=False):
    """the frames function of the named readout. ``pinned`` -> ``orig`` itself (nothing is installed: today's generator, bit for bit)."""
    if readout not in READOUTS:
        raise ValueError(f"unknown readout {readout!r}; one of {READOUTS}")
    if readout == "pinned":
        if gain is not None or path_exec:
            raise ValueError("--acc-gain / --path-exec are readout causal_g's parameters: declared with readout pinned they would do nothing")
        return orig
    if gain is None:
        raise ValueError("readout causal_g needs --acc-gain")

    def frames(k_prev, knots, m, psi0):
        return causal_g_frames(orig, k_prev, knots, m, psi0, gain, path_exec)
    frames.__wrapped__ = orig
    return frames
