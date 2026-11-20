"""Contract tests for the T1-kin executor of record (stack/scripts/v8f_t1_eval_hb.py, PI 2026-11-20, decision card D16-D).

The wrapper adds one thing to the pinned T1 tool: a heading bound (a heading step is at most kappa x the distance travelled). These
tests pin that it does exactly that and nothing else:

* identity where the bound cannot act (huge kappa; a straight drive through the real ``hermite_frames``), bit for bit;
* a standstill holds the heading exactly, and an acting step turns by exactly kappa x distance (the analytic value);
* the seam at +-pi is a small step, not a full turn;
* positions and speeds pass through untouched, and the pinned module's function is restored after the run;
* the old executor (``--kappa-max none``) installs nothing: the very same function object (the deliberate-regression arm).
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import v8f_t1_eval_hb as HB  # noqa: E402


def _walk(n, step, rng=None, jitter=0.0):
    """n frame positions along +x, ``step`` metres apart, optional jitter"""
    xy = np.stack([np.arange(1, n + 1) * step, np.zeros(n)], 1)
    if jitter and rng is not None:
        xy = xy + rng.normal(0.0, jitter, xy.shape)
    return xy


def test_identity_when_kappa_is_huge():
    rng = np.random.default_rng(0)
    xy = _walk(40, 1.0, rng, 0.1)
    psi = rng.uniform(-math.pi, math.pi, 40)
    rec = HB.kb_new()
    out = HB.bound_heading(xy, psi, np.full(40, 5.0), np.zeros(2), float(psi[0]), 1e9, rec)
    assert np.array_equal(out, psi)                                               # never acted: the tool's own values
    assert rec["frames_bound_acted"] == 0 and rec["frames_total"] == 40


def test_standstill_holds_the_heading_exactly():
    rng = np.random.default_rng(1)
    xy = np.zeros((20, 2))
    psi = rng.uniform(-math.pi, math.pi, 20)
    out = HB.bound_heading(xy, psi, np.zeros(20), np.zeros(2), 0.3, HB.KAPPA_RECORD)
    assert np.max(np.abs(out - 0.3)) <= 1e-15                                     # wrap_pi rounds by at most one ulp


def test_acting_step_turns_by_exactly_kappa_times_distance():
    out = HB.bound_heading(np.array([[1.0, 0.0]]), np.array([0.5]), np.array([5.0]), np.zeros(2), 0.0, 0.2)
    assert out[0] == pytest.approx(0.2, abs=1e-15)
    out = HB.bound_heading(np.array([[0.0, 2.0]]), np.array([-1.0]), np.array([5.0]), np.zeros(2), 0.0, 0.2)
    assert out[0] == pytest.approx(-0.4, abs=1e-15)
    out = HB.bound_heading(np.array([[1.0, 0.0]]), np.array([0.15]), np.array([5.0]), np.zeros(2), 0.0, 0.2)
    assert out[0] == 0.15                                                         # within the bound: the tool's own value


def test_seam_is_a_small_step():
    a0, a1 = math.pi - 0.05, -math.pi + 0.05                                      # +0.1 rad across the seam
    out = HB.bound_heading(np.array([[1.0, 0.0]]), np.array([a1]), np.array([5.0]), np.zeros(2), a0, 0.2)
    assert out[0] == a1
    out = HB.bound_heading(np.array([[1.0, 0.0]]), np.array([a1]), np.array([5.0]), np.zeros(2), a0, 0.05)
    assert out[0] == pytest.approx(HB.wrap_pi(a0 + 0.05), abs=1e-12)


def test_old_executor_installs_nothing():
    def orig(k_prev, knots, m, psi0):
        return None
    assert HB.bounded_frames(orig, None) is orig
    mod = types.SimpleNamespace(hermite_frames=orig)
    with HB.installed(mod, None):
        assert mod.hermite_frames is orig
    assert mod.hermite_frames is orig
    assert HB.parse_kappa("none") is None and HB.parse_kappa("0.2") == 0.2
    for bad in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            HB.parse_kappa(bad)


def test_install_passes_positions_and_speeds_and_restores():
    xy0 = np.array([[0.001, 0.0], [0.002, 0.001], [0.003, -0.001], [0.004, 0.0]])
    v0 = np.array([0.01, 0.01, 0.01, 0.01])
    psi_raw = np.array([0.0, 1.2, -1.3, 0.9])                                    # plan jitter near a standstill

    def orig(k_prev, knots, m, psi0):
        return xy0.copy(), psi_raw.copy(), v0.copy()
    mod = types.SimpleNamespace(hermite_frames=orig)
    rec = HB.kb_new()
    with HB.installed(mod, 0.2, rec):
        assert mod.hermite_frames is not orig
        xy, psi, v = mod.hermite_frames(np.zeros(2), np.zeros((3, 2)), 2, 0.0)
    assert mod.hermite_frames is orig
    assert np.array_equal(xy, xy0) and np.array_equal(v, v0)
    assert rec["frames_bound_acted"] == 3 and rec["by_speed"]["lt2"]["frames_total"] == 4
    assert np.max(np.abs(psi)) <= 0.2 * 0.01                                       # 1 cm travelled in all
    assert rec["max_excess_rad"] <= 1e-12


def test_real_hermite_frames_straight_drive_is_untouched_and_jitter_is_bounded():
    import v8f_t1_eval as T
    knots = np.stack([np.arange(0, 12) * 2.0, np.zeros(12)], 1)                  # 10 m/s straight
    xy, psi, v = T.hermite_frames(np.array([-2.0, 0.0]), knots, 8, 0.0)
    rec = HB.kb_new()
    with HB.installed(T, HB.KAPPA_RECORD, rec):
        xy_b, psi_b, v_b = T.hermite_frames(np.array([-2.0, 0.0]), knots, 8, 0.0)
    assert np.array_equal(np.asarray(xy), np.asarray(xy_b)) and np.array_equal(np.asarray(v), np.asarray(v_b))
    assert np.array_equal(np.asarray(psi, np.float64), psi_b) and rec["frames_bound_acted"] == 0
    rng = np.random.default_rng(3)
    jit = np.cumsum(rng.normal(0.0, 0.004, (12, 2)), 0)                          # ~4 mm per step: the tool's heading swings
    xy, psi, v = T.hermite_frames(np.zeros(2), jit, 8, 0.0)
    with HB.installed(T, HB.KAPPA_RECORD):
        xy_b, psi_b, v_b = T.hermite_frames(np.zeros(2), jit, 8, 0.0)
    assert T.hermite_frames.__name__ == "hermite_frames"
    assert np.array_equal(np.asarray(xy), np.asarray(xy_b)) and np.array_equal(np.asarray(v), np.asarray(v_b))
    prev, pxy = 0.0, np.asarray(jit[0], np.float64)
    for f in range(len(psi_b)):
        dist = float(np.hypot(*(np.asarray(xy_b[f]) - pxy)))
        assert abs(HB.wrap_pi(psi_b[f] - prev)) <= HB.KAPPA_RECORD * dist + 1e-12
        prev, pxy = psi_b[f], np.asarray(xy_b[f])
    assert np.max(np.abs(np.asarray(psi) - psi_b)) > 0.1                          # the bound did act on this plan
