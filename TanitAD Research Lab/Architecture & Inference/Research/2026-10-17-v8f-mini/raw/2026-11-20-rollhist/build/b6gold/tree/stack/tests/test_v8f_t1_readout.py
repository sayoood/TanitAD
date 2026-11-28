"""Contract tests for stack/scripts/v8f_t1_readout.py (PREREG_V8F_ROLLHIST item N1): the causal readouts of the T1-kin own history, promoted from the integ10k study wrapper.

CPU, fp32 torch not involved (numpy float64), one thread. What is pinned:

* ADD63's identity controls, array for array: gain 1.0 equals ``causal``, gain 0.0 equals ``causal_z``; the general formula agrees with both to <= 1e-12;
* the promoted functions equal the STUDY file's own functions bit for bit: ``GOLDEN_SHA256`` holds the digest of the study functions' outputs on the seeded cases below,
  one per platform, because float64 cos/sin and SIMD paths differ in the last bit across libm/numpy builds (ROLLHIST Amendment 3): the dev box (Windows, numpy 2.5.1)
  and the LoRA pod (Linux, numpy 2.5.4, where ``b1/study_vs_promoted.py`` read the study and promoted functions bit-identical on 2,880 arrays, 2026-11-21);
  (computed by exec'ing ``integ10k/v8f_t1_eval_reader_rule_rzg.py`` at the tip 7b26a8ac, ``b1/study_vs_promoted.py``); a changed digit anywhere in the arithmetic changes it;
* control a3 (ported): the one-frame acceleration read at the last executed knot is ``gain`` x the executed one, to <= 1e-6;
* ``--path-exec``: positions and headings equal ``backward_frames`` bit for bit, the speed column equals that of the same gain without it;
* ``readout_frames("pinned")`` is ``orig`` ITSELF (nothing installed), and a declared-and-inert parameter is refused;
* planted faults are caught: the gain ignored, the path taken from the pinned call although --path-exec is on.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import v8f_t1_eval as T1  # noqa: E402
import v8f_t1_readout as RO  # noqa: E402

GOLDEN_SHA256 = {
    "1c42d85fdbe040c9e10c3967c4df34fa405fe39a0ad45c91a5f5a9cf0cd4a8e9",  # study functions, dev box (Windows, numpy 2.5.1)
    "6b93f03a9642298870f0ac168da40d4fe3222d3ee2a397c8b872f31208ebae87",  # study functions, LoRA pod (Linux, numpy 2.5.4)
}
DT = 0.2


def cases():
    rng = np.random.default_rng(20261120)
    for c in range(40):
        m = int(rng.choice([5, 10]))
        th = np.cumsum(rng.normal(0, 0.05, m + 3))
        sp = 0.2 * np.abs(8 + rng.normal(0, 2.0, m + 3)) * (0.0 if c % 7 == 0 else 1.0)
        pts = np.cumsum(np.stack([sp * np.cos(th), sp * np.sin(th)], -1), 0)
        yield pts[0], pts[1:], m, float(rng.uniform(-3, 3))


def _same(a, b):
    return all(np.array_equal(np.asarray(u), np.asarray(w)) for u, w in zip(a, b))


def test_promoted_functions_equal_the_study_functions_bit_for_bit_golden_digest():
    h = hashlib.sha256()
    for kp, kn, m, p0 in cases():
        outs = [RO.backward_frames(T1.hermite_frames, kp, kn, m, p0)]
        outs += [RO.causal_frames(T1.hermite_frames, kp, kn, m, p0, acc) for acc in ("step", "zero", "window")]
        outs += [RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, g, px, fg) for g in (-0.25, 0.0, 0.5, 1.0, 1.7) for px in (False, True) for fg in (False, True)]
        for o in outs:
            for w in o:
                h.update(np.ascontiguousarray(w, np.float64).tobytes())
    assert h.hexdigest() in GOLDEN_SHA256


def test_gain_one_is_causal_and_gain_zero_is_causal_z_array_for_array():
    n = 0
    for kp, kn, m, p0 in cases():
        for px in (False, True):
            base = (lambda *a: RO.backward_frames(T1.hermite_frames, *a)) if px else T1.hermite_frames
            assert _same(RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, 1.0, px), RO.causal_frames(base, kp, kn, m, p0, "step"))
            assert _same(RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, 0.0, px), RO.causal_frames(base, kp, kn, m, p0, "zero"))
            n += 1
    assert n == 80
    for kp, kn, m, p0 in cases():                                    # the GENERAL formula (forced) agrees with both identities to rounding
        for g, acc in ((1.0, "step"), (0.0, "zero")):
            a = RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, g, False, True)
            b = RO.causal_frames(T1.hermite_frames, kp, kn, m, p0, acc)
            assert max(float(np.abs(np.asarray(u) - np.asarray(w)).max()) for u, w in zip(a, b)) <= 1e-12


@pytest.mark.parametrize("gain", [-0.25, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("path_exec", [False, True])
def test_a3_the_acceleration_at_the_last_executed_knot_is_gain_times_the_executed_one(gain, path_exec):
    worst = 0.0
    for kp, kn, m, p0 in cases():
        xy, psi, v = RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, gain, path_exec)
        K = np.vstack([np.asarray(kp, np.float64)[None], np.asarray(kn, np.float64)[:m + 1]])
        s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT
        a_read = (v[-1] - v[-2]) / (DT / 2.0)
        worst = max(worst, abs(a_read - gain * (s[m] - s[m - 1]) / DT), abs(v[-1] - s[m]))
    assert worst <= 1e-6


def test_path_exec_takes_the_path_from_the_executed_knots_and_leaves_the_speed_column_alone():
    nonvac = 0.0
    for kp, kn, m, p0 in cases():
        xb, pb, _ = RO.backward_frames(T1.hermite_frames, kp, kn, m, p0)
        xo, po, _ = T1.hermite_frames(kp, kn, m, p0)
        nonvac = max(nonvac, float(np.abs(np.asarray(xo) - np.asarray(xb)).max()), float(np.abs(np.asarray(po) - np.asarray(pb)).max()))
        for g in (0.0, 0.5, 1.0):
            xp, pp, vp = RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, g, True)
            xn, pn, vn = RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, g, False)
            assert np.array_equal(xp, xb) and np.array_equal(pp, pb) and np.array_equal(vp, vn)
    assert nonvac > 1e-6, "pinned against backward must differ on these plans, else the comparison says nothing"


def test_planted_faults_are_caught():
    kp, kn, m, p0 = list(cases())[1]
    good = RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, 0.5, True)
    gain_ignored = RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, 1.0, True)
    path_pinned = RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, 0.5, False)
    assert not np.array_equal(good[2], gain_ignored[2])                          # (a) a gain that is ignored changes the speed column: visible
    assert not np.array_equal(good[0], path_pinned[0])                           # (b) a path taken from the pinned call although --path-exec is on: visible


def test_readout_frames_pinned_is_orig_itself_and_causal_g_matches_the_function():
    assert RO.readout_frames(T1.hermite_frames, "pinned") is T1.hermite_frames
    for kp, kn, m, p0 in list(cases())[:6]:
        f = RO.readout_frames(T1.hermite_frames, "causal_g", 1.0, True)
        assert _same(f(kp, kn, m, p0), RO.causal_g_frames(T1.hermite_frames, kp, kn, m, p0, 1.0, True))
        assert f.__wrapped__ is T1.hermite_frames


def test_declared_and_inert_or_missing_parameters_are_refused():
    with pytest.raises(ValueError):
        RO.readout_frames(T1.hermite_frames, "pinned", 1.0)
    with pytest.raises(ValueError):
        RO.readout_frames(T1.hermite_frames, "pinned", None, True)
    with pytest.raises(ValueError):
        RO.readout_frames(T1.hermite_frames, "causal_g")
    with pytest.raises(ValueError):
        RO.readout_frames(T1.hermite_frames, "backward")
