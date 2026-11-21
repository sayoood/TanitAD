"""pytest for the H-ST-RECOVERYLABEL prototype (bank_presteps/recovery/labeller.py). Synthetic logged paths only (no bank, no RAM).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_labeller.py   (cwd = this directory)
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import labeller as L  # noqa: E402

# the stack's own gt_waypoints (taniteval/floors_kin.py at the tip, numpy only), loaded from the copy in this package
_spec = importlib.util.spec_from_file_location("floors_kin_tip", HERE.parent / "src_tip" / "floors_kin.py")
FK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FK)


# ---------------------------------------------------------------------------------------------- synthetic logged paths (0.2 s grid)
def path_straight(T=140, v=10.0):
    i = np.arange(T)
    g = np.stack([v * DT * i, np.zeros(T), np.zeros(T)], 1)
    return g, np.full(T, v)


def path_curved(T=140, v=8.0, kappa=0.02):
    i = np.arange(T)
    th = kappa * v * DT * i
    # constant-curvature arc
    x = np.sin(th) / kappa
    y = (1 - np.cos(th)) / kappa
    return np.stack([x, y, th], 1), np.full(T, v)


def path_accel(T=140, a=1.5, v0=2.0):
    i = np.arange(T)
    t = DT * i
    x = v0 * t + 0.5 * a * t * t
    return np.stack([x, 0.0 * x, np.zeros(T)], 1), v0 + a * t


DT = L.DT
PATHS = {"straight": path_straight, "curved": path_curved, "accel": path_accel}


@pytest.mark.parametrize("name", list(PATHS))
@pytest.mark.parametrize("r", [1, 5, 40])
def test_identity_bit_for_bit(name, r):
    g, v = PATHS[name]()
    res = L.recovery_label(g, v, r, (g[r, 0], g[r, 1], g[r, 2]), float(v[r]))
    assert res["status"] == L.STATUS_IDENTITY
    h = L.human_label(g, r)
    assert np.array_equal(res["label_own"], h)                       # zero tolerance
    gtk = FK.gt_waypoints(np.concatenate([g, v[:, None]], 1), [r], np.arange(1, 31))[0]
    assert np.array_equal(h, gtk)                                    # human_label IS the stack's gt_waypoints expression


def test_identity_on_random_noisy_human_path():
    rng = np.random.default_rng(0)
    T = 140
    g, v = path_straight(T)
    g = g + rng.normal(0, 0.05, g.shape) * np.array([1, 1, 0.01])   # a noisy human path whose own speed/accel limits are violated at 0.2 s
    v = v + rng.normal(0, 1.0, T)
    for r in range(1, 100, 7):
        res = L.recovery_label(g, v, r, (g[r, 0], g[r, 1], g[r, 2]), float(v[r]))
        assert res["status"] == L.STATUS_IDENTITY and np.array_equal(res["label_own"], L.human_label(g, r))


@pytest.mark.parametrize("name", list(PATHS))
def test_continuity_near_zero_offset(name):
    g, v = PATHS[name]()
    r = 20
    h = L.human_label(g, r)
    dev = []
    for eps in (1e-6, 1e-4):
        # tiny lateral and speed offsets: the labeller is NOT in its identity branch, and the label must stay near the human's
        c, s = math.cos(g[r, 2]), math.sin(g[r, 2])
        own = (g[r, 0] - s * eps, g[r, 1] + c * eps, g[r, 2])
        res = L.recovery_label(g, v, r, own, float(v[r]) + eps)
        assert res["status"] == L.STATUS_EMITTED, res["status"]
        # express in the human's frame for the comparison (the own frame is eps-rotated/shifted)
        lab_h = L.to_ego(res["label_world"][1:], (g[r, 0], g[r, 1], g[r, 2]))
        dev.append(float(np.abs(lab_h - h).max()))
    # no trapdoor: the deviation is a bounded discretisation floor (Hermite smoothing of a polyline), not a jump, and it does not grow with eps
    assert max(dev) < 2e-2, dev
    assert abs(dev[1] - dev[0]) < 1e-3, dev


def test_lateral_offset_is_recovered_and_feasible():
    g, v = path_straight()
    r, off = 30, 2.0
    own = (g[r, 0], g[r, 1] + off, 0.0)
    res = L.recovery_label(g, v, r, own, 10.0)
    assert res["status"] == L.STATUS_EMITTED
    assert res["k_target"] in L.LADDER and res["k_rej"] >= 1
    # the checker, called independently on the emitted label over its recovery phase, reads zero
    assert L.violations(res["label_own"], 10.0, res["k_rej"] + 1)["any"] is False
    # rejoin: after k_rej the world label lies on the human path (y == 0) and before it starts at the offset
    Lw = res["label_world"]
    assert Lw[0][1] == pytest.approx(off)
    assert np.abs(Lw[res["k_rej"]:, 1]).max() < 1e-6
    # label differs from the human's where the car is off the path
    assert np.abs(Lw[1] - g[r + 1, :2]).max() > 0.1


def test_impossible_recovery_is_refused_not_emitted():
    g, v = path_straight()
    r = 30
    own = (g[r, 0], g[r, 1] + 3.5, math.pi)               # stationary, facing backwards, 3.5 m off: cannot turn at v = 0 under kappa <= 0.2
    res = L.recovery_label(g, np.where(True, 10.0, v), r, own, 0.0)
    assert res["status"] in ("refused_infeasible", "degenerate", "path_exhausted")
    assert res["label_own"] is None


def test_speed_profile_rate_limit_and_merge():
    vh = np.full(31, 10.0)
    v = L.label_speed(0.0, vh)
    step = np.diff(v)
    assert step[0] == pytest.approx(L.A_LIM * L.DT)
    assert (step <= L.A_LIM * L.DT + 1e-12).all() and (step >= 0).all()
    k_merge = int(np.argmax(v == 10.0))
    assert k_merge > 0 and (v[k_merge:] == 10.0).all()
    # once merged it follows the human even through a human step far above the rate limit
    vh2 = np.concatenate([np.full(16, 10.0), np.full(15, 30.0)])
    v2 = L.label_speed(10.0, vh2)
    assert np.array_equal(v2, vh2)
    # the no-clip arm jumps
    assert L.label_speed(0.0, vh, clip=False)[1] == 10.0


def _panel(n=240, seed=1):
    rng = np.random.default_rng(seed)
    g, v = path_straight(T=160, v=9.0)
    gc, vc = path_curved(T=160, v=7.0, kappa=0.03)
    panel = []
    for i in range(n):
        gg, vv = (g, v) if i % 2 == 0 else (gc, vc)
        r = int(rng.integers(5, 60))
        e = float(rng.uniform(3.0, 8.0)) * rng.choice([-1, 1])        # off_lat band
        dpsi = float(rng.uniform(-35, 35)) * math.pi / 180            # includes beyond-30-degree heading errors
        v_own = float(rng.uniform(0.0, 14.0))
        c, s = math.cos(gg[r, 2]), math.sin(gg[r, 2])
        own = (gg[r, 0] - s * e, gg[r, 1] + c * e, gg[r, 2] + dpsi)
        panel.append((gg, vv, r, own, v_own))
    return panel


def test_gate_real_labeller_passes_and_regression_arms_fail():
    panel = _panel()
    n_emit = n_ref = 0
    real_viol = 0
    dr_viol_records = nc_viol_records = nc_n = 0
    for gg, vv, r, own, v_own in panel:
        res = L.recovery_label(gg, vv, r, own, v_own)
        if res["label_own"] is None:
            n_ref += 1
        else:
            n_emit += 1
            real_viol += L.gate_violation_count(res["label_own"], v_own, res["k_rej"] + 1)
        dr = L.label_ignore_offset(gg, vv, r, own, v_own)
        dr_viol_records += L.gate_violation_count(dr["label_own"], v_own, 6) > 0
        nc = L.recovery_label(gg, vv, r, own, v_own, clip=False, check=False)
        if nc["label_own"] is not None:
            nc_n += 1
            nc_viol_records += L.gate_violation_count(nc["label_own"], v_own, nc["k_rej"] + 1) > 0
    assert n_emit >= 100, (n_emit, n_ref)                  # the labeller is not vacuously refusing everything
    assert real_viol == 0                                  # G-REC: exactly zero violations among emitted labels
    assert dr_viol_records >= 0.9 * len(panel)             # the offset-ignoring labeller FAILS the gate (violates on nearly every record)
    assert nc_n > 0 and nc_viol_records >= 0.3 * nc_n      # the no-clip / unchecked labeller is also caught


def test_mutant_loosened_rate_limit_is_caught_by_the_gate():
    """A labeller whose speed rate limit is 10x too loose must produce violations the gate sees."""
    panel = _panel(120, seed=3)
    orig = L.label_speed
    try:
        L.label_speed = lambda v_own, v_h, a_lim=L.A_LIM, clip=True: orig(v_own, v_h, a_lim=10 * L.A_LIM, clip=clip)
        viol = 0
        for gg, vv, r, own, v_own in panel:
            res = L.recovery_label(gg, vv, r, own, v_own, check=False)
            if res["label_own"] is not None:
                viol += L.gate_violation_count(res["label_own"], v_own, res["k_rej"] + 1)
        assert viol > 0
    finally:
        L.label_speed = orig


def test_checker_known_values():
    # constant 10 m/s straight path: zero violations
    pts = np.stack([np.arange(1, 31) * 2.0, np.zeros(30)], 1)
    assert L.violations(pts, 10.0, 30)["any"] is False
    # a 2 m lateral jump in one step at 10 m/s: curvature / friction violation
    bad = pts.copy()
    bad[5:, 1] += 2.0
    assert L.violations(bad, 10.0, 30)["any"] is True
    # an own speed of 0 followed by a 10 m/s step: acceleration violation
    assert L.violations(pts, 0.0, 30)["accel"] > 0


def test_deterministic():
    panel = _panel(40, seed=5)
    for gg, vv, r, own, v_own in panel:
        a = L.recovery_label(gg, vv, r, own, v_own)
        b = L.recovery_label(gg, vv, r, own, v_own)
        assert a["status"] == b["status"]
        if a["label_own"] is not None:
            assert np.array_equal(a["label_own"], b["label_own"])


def test_no_logged_future_is_not_labelled():
    g, v = path_straight(T=40)
    res = L.recovery_label(g, v, 20, (g[20, 0], g[20, 1] + 4, 0.0), 5.0)
    assert res["status"] == "no_logged_future"
