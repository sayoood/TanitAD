"""Pins for a0k0_headroom_lon_r8.py (longitudinal-only residual scaling): the cross gain fixed at 1 leaves the LATERAL POSITION equal to os
bit for bit; along gain also fixed at 1 -> the plan equals os exactly; a DELIBERATE-REGRESSION arm (letting the cross gain float must change S6
lateral on a fixture, and a mutated box must be caught by the same guard); analytic fit targets as LITERALS; the acceptance-condition function;
cross-fitting keeps the fixed gain and does not leak; the real deliverable's controls (skipped only if A0K0_LON.json is absent).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -m pytest -q test_a0k0_headroom_lon_r8.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0k0_headroom_lon_r8 as M  # noqa: E402

OUT_JSON = HERE.parent / "raw" / "a0k0_lon" / "A0K0_LON.json"
FROZEN = HERE.parent / "raw" / "a0k0_lon" / "FROZEN_GAINS_LON_ema30k.json"
BOX_FIXED_ALL = ((1.0, 1.0), (1.0, 1.0))


def _ob(n=3000, T=6, seed=0):
    rng = np.random.default_rng(seed)
    return (37.0 * rng.normal(size=(n, T, 2))).astype(np.float32), (11.0 * rng.normal(size=(n, T, 2))).astype(np.float32)


def _gt(O, B, along, cross, noise=0.05, seed=5):
    """GT whose true per-axis gains are (along, cross) on the residual (os - prior)."""
    rng = np.random.default_rng(seed)
    G = np.empty(O.shape, np.float64)
    G[..., 0] = along * O[..., 0].astype(np.float64) + (1 - along) * B[..., 0].astype(np.float64)
    G[..., 1] = cross * O[..., 1].astype(np.float64) + (1 - cross) * B[..., 1].astype(np.float64)
    return G + noise * rng.normal(size=O.shape)


def _lateral_untouched(P, O) -> bool:
    return bool(np.array_equal(P[..., 1], O[..., 1]))


# ------------------------------------------------------------------ (a) cross gain fixed at 1 => lateral position == os, bit for bit
@pytest.mark.parametrize("along_true,cross_true", [(0.3, 0.3), (0.3, 1.0), (1.7, 0.4), (0.0, 0.0), (1.0, 1.7)])
def test_a_cross_fixed_lateral_position_equals_os_exactly(along_true, cross_true):
    O, B = _ob()
    G = _gt(O, B, along_true, cross_true)
    W = M.fit_gains(O, B, G, box=M.BOX_LON)
    assert np.all(W[:, 1] == 1.0), W[:, 1]                       # EXACTLY 1.0, not approximately
    P = M.blend(O, B, W)
    assert P.dtype == O.dtype == np.float32
    assert _lateral_untouched(P, O)
    assert (np.abs(W[:, 0]) <= 2.0).all() and (W[:, 0] >= 0.0).all()


def test_a_along_gain_recovers_the_literal_target_when_the_lateral_residual_is_os():
    O, B = _ob()
    G = _gt(O, B, 0.3, 1.0, noise=0.0)                           # lateral truth == os; along truth 0.3 -> fitted 0.30 (literal)
    W = M.fit_gains(O, B, G, box=M.BOX_LON)
    assert np.allclose(W[:, 0], 0.3, atol=1e-4) and np.all(W[:, 1] == 1.0)


# ------------------------------------------------------------------ (b) along gain also fixed at 1 => the plan IS os
def test_b_both_gains_fixed_at_1_plan_equals_os_exactly():
    O, B = _ob(seed=3)
    G = _gt(O, B, 0.2, 0.2, seed=9)
    W = M.fit_gains(O, B, G, box=BOX_FIXED_ALL)
    assert np.all(W == 1.0)
    assert np.array_equal(M.blend(O, B, W), O)
    assert np.array_equal(M.blend(O, B, np.ones((O.shape[1], 2))), O)
    assert np.array_equal(M.blend(O, B, np.zeros((O.shape[1], 2))), B)


# ------------------------------------------------------------------ (c) DELIBERATE REGRESSION: a floating cross gain MUST change S6 lateral
def test_c_deliberate_regression_floating_cross_changes_lateral_on_an_s6_fixture():
    O, B = _ob(n=2500, T=6, seed=11)                              # T = 6: the S6 grid (1..6 s)
    G = _gt(O, B, 0.5, 0.4)                                       # the truth wants the lateral residual shrunk to 0.4
    P_lon = M.blend(O, B, M.fit_gains(O, B, G, box=M.BOX_LON))
    W_free = M.fit_gains(O, B, G, box=M.BOX_RS)                   # the regression: cross gain allowed to float
    P_free = M.blend(O, B, W_free)
    assert np.allclose(W_free[:, 1], 0.4, atol=1e-3) and np.allclose(W_free[:, 0], 0.5, atol=1e-3)
    assert _lateral_untouched(P_lon, O)
    assert not _lateral_untouched(P_free, O)                      # the identity guard goes RED when the cross gain floats
    assert float(np.abs(P_free[..., 1] - O[..., 1]).max()) > 1.0
    # the lateral cross-track error against GT really does change (shrunk toward the truth), so the identity is not vacuous
    assert float(np.abs(P_free[..., 1] - G[..., 1]).mean()) < float(np.abs(P_lon[..., 1] - G[..., 1]).mean())


def test_c_mutating_the_box_is_caught_by_the_guard(monkeypatch):
    O, B = _ob(n=1500, seed=13)
    G = _gt(O, B, 0.5, 0.4)
    monkeypatch.setattr(M, "BOX_LON", M.BOX_RS)                   # reintroduce the defect: the "LON" arm silently fits both axes
    W = M.fit_gains(O, B, G, box=M.BOX_LON)
    P = M.blend(O, B, W)
    assert not np.all(W[:, 1] == 1.0)
    assert not _lateral_untouched(P, O)                           # the control used by read_set would FAIL -> the compute refuses to write


def test_lat1_box_never_shrinks_the_lateral_residual():
    O, B = _ob(n=2000, seed=17)
    W_shrink = M.fit_gains(O, B, _gt(O, B, 0.6, 0.4), box=M.BOX_LAT1)
    assert np.all(W_shrink[:, 1] == 1.0)                           # truth 0.4 < 1: the constraint binds exactly
    W_grow = M.fit_gains(O, B, _gt(O, B, 0.6, 1.5), box=M.BOX_LAT1)
    assert np.allclose(W_grow[:, 1], 1.5, atol=1e-3) and np.allclose(W_grow[:, 0], 0.6, atol=1e-3)
    assert (W_grow[:, 1] >= 1.0).all()


# ------------------------------------------------------------------ the mechanism the prereg names: heading is DERIVED from positions
def test_heading_is_a_function_of_x_as_well_as_y_literal_expectation():
    # the segment (1,1)->(2,2) has a tangent of exactly 45 deg; scaling x by 0.5 with y untouched gives (0.5,1)->(1,2): atan2(1, 0.5) = 63.4349 deg
    ang = lambda a, b: math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))  # noqa: E731
    assert ang((1, 1), (2, 2)) == pytest.approx(45.0, abs=1e-9)
    assert ang((0.5, 1), (1.0, 2)) == pytest.approx(63.4349488, abs=1e-6)       # literal, derived by hand, not from the code under test


# ------------------------------------------------------------------ ACC-LAT: no LATERAL row separated worse (lo > 0) at EITHER seed
def _cells(lo_by_seed):
    out = {}
    for s, lo in lo_by_seed.items():
        out[s] = {k: {"delta": lo + 0.1, "lo": lo, "hi": lo + 0.2, "separated": lo > 0} for k in M.LATERAL_KEYS}
    return out


def test_acc_lat_holds_when_no_row_is_separated_worse():
    r = M.acc_lat(_cells({0: -0.3, 1: -0.1}))
    assert r["HOLDS"] is True and r["separated_worse_at_any_seed"] == []


def test_acc_lat_fails_when_one_seed_has_a_separated_worse_row():
    cells = _cells({0: -0.3, 1: -0.1})
    cells[1]["LAT_heading_mae_deg"]["lo"] = 0.05                      # worse at seed 1 ONLY -> "either seed" must still fail
    r = M.acc_lat(cells)
    assert r["HOLDS"] is False
    assert r["separated_worse_at_any_seed"] == ["LAT_heading_mae_deg"] and r["separated_worse_at_both_seeds"] == []
    cells[0]["LAT_heading_mae_deg"]["lo"] = 0.01
    assert M.acc_lat(cells)["separated_worse_at_both_seeds"] == ["LAT_heading_mae_deg"]


def test_acc_lat_is_undetermined_not_clean_when_a_cell_has_no_interval():
    cells = _cells({0: -0.3, 1: -0.1})
    cells[0]["LAT_cross_mae_m"] = {"status": "REFUSED"}
    assert M.acc_lat(cells)["HOLDS"] is None


def test_acc_lat_uses_the_five_lateral_rows_of_the_battery_metrics():
    import clamp_posthoc_r8 as CP
    _NR, _BP, PR, _ra, _sha = CP.get_readers()
    assert sorted(k for fam, k, *_ in PR.METRICS if fam == "LATERAL") == sorted(M.LATERAL_KEYS)
    assert all(lower for fam, _k, _l, _u, lower, *_ in PR.METRICS if fam == "LATERAL")     # all error metrics: worse == lo > 0


# ------------------------------------------------------------------ cross-fitting keeps the fixed gain, and does not leak
def test_crossfit_lon_keeps_the_cross_gain_exactly_1_and_does_not_leak():
    rng = np.random.default_rng(1)
    n_ep, per = 20, 40
    clip = np.repeat(np.arange(n_ep), per)
    O, B = _ob(n=n_ep * per, T=4, seed=1)
    w_ep = rng.uniform(0.2, 0.9, size=n_ep)[clip][:, None, None]
    G = w_ep * O + (1 - w_ep) * B + 0.05 * rng.normal(size=O.shape)
    folds = M.episode_folds(clip)
    f0 = folds == 0
    W1, _ = M.crossfit(O, B, G, folds, box=M.BOX_LON)
    assert np.all(W1[..., 1] == 1.0)
    assert _lateral_untouched(M.blend(O, B, W1), O)
    G2 = G.copy()
    G2[f0] = 50.0 * O[f0] - 49.0 * B[f0]                              # wreck fold 0's OWN GT: its applied gains must not move
    W2, _ = M.crossfit(O, B, G2, folds, box=M.BOX_LON)
    assert np.array_equal(W1[f0], W2[f0])
    W3, _ = M.crossfit(O, B, G, folds, box=M.BOX_LON, leak=True)
    W4, _ = M.crossfit(O, B, G2, folds, box=M.BOX_LON, leak=True)
    assert not np.array_equal(W3[f0], W4[f0])                         # the DELIBERATE REGRESSION (leaky fit) is detected
    assert np.all(W3[..., 1] == 1.0) and np.all(W4[..., 1] == 1.0)


def test_scalar_box_is_unchanged_by_the_per_axis_generalisation():
    O, B = _ob(n=800, T=4, seed=21)
    G = _gt(O, B, 0.3, 0.7)
    W_scalar = M.fit_gains(O, B, G, box=(0.0, 2.0))
    W_pair = M.fit_gains(O, B, G, box=((0.0, 2.0), (0.0, 2.0)))
    assert np.array_equal(W_scalar, W_pair)
    assert np.allclose(W_scalar[:, 0], 0.3, atol=1e-3) and np.allclose(W_scalar[:, 1], 0.7, atol=1e-3)


# ------------------------------------------------------------------ the real deliverable's controls (skipped if absent)
@pytest.mark.skipif(not OUT_JSON.exists(), reason="A0K0_LON.json not produced yet")
def test_deliverable_controls():
    r = M.jload(OUT_JSON)
    assert r["controls_summary"]["ALL_PASS"] is True
    assert r["controls_summary"]["R0926_exact"] is True
    for tree, tr in r["trees"].items():
        for sd, sr in tr["seeds"].items():
            for surf, ss in sr["surfaces"].items():
                for st, rs in ss["sets"].items():
                    ctl = rs["controls"]
                    assert ctl["a_g1_reproduces_banked_os"]["PASS"] and ctl["b_g0_is_prior"]["PASS"]
                    assert ctl["h_RS_XF_reproduces_banked"]["PASS"], (tree, sd, surf, st)
                    assert ctl["i_lon_lateral_position_is_os"]["PASS"], (tree, sd, surf, st)
                    assert ctl["c_d_optimism"]["IS_known_sign_le_prior_and_os"] is True
                    for arm in ("RS_LON_XF", "GATED_RS_LON_XF"):      # identity, not estimate: cross rows vs os read exactly 0
                        for k in ("LAT_cross_mae_m", "LAT_cross_abs_final_m"):
                            c0 = rs["arms"][arm]["minus_os"][k]
                            assert (c0["delta"], c0["lo"], c0["hi"]) == (0.0, 0.0, 0.0), (tree, sd, surf, st, arm, k, c0)
            for key, row in tr["summary"].items():
                for arm in M.PRIMARY_ARMS:                            # the stored ACC-LAT equals a recomputation from the stored cells
                    surf, st = key.split("|")
                    cells = {s: {k: tr["seeds"][str(s)]["surfaces"][surf]["sets"][st]["arms"][arm]["minus_os"][k] for k in M.LATERAL_KEYS} for s in (0, 1)}
                    assert M.acc_lat(cells)["HOLDS"] == row[arm]["ACC_LAT"]["HOLDS"]


@pytest.mark.skipif(not FROZEN.exists(), reason="frozen gains not produced yet")
def test_frozen_gain_tables_respect_their_boxes():
    fz = M.jload(FROZEN)
    for surf, t in fz["surfaces"].items():
        a = t["all"]
        assert np.all(np.asarray(a["RS_LON"])[:, 1] == 1.0) and np.all(np.asarray(a["RS_LAT1"])[:, 1] >= 1.0)
        assert (np.asarray(a["RS_LON"])[:, 0] >= 0).all() and (np.asarray(a["RS_LON"])[:, 0] <= 2).all()
        for name, v in a["GATED_RS_LON"].items():
            assert np.all(np.asarray(v["g"])[:, 1] == 1.0), (surf, name)
