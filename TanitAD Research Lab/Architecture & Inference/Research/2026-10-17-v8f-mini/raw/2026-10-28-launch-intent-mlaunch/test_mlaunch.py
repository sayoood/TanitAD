"""Tests for the pure parts of mlaunch.py (no GPU, no training).  Run alone:

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider test_mlaunch.py

The estimator tests use the pinned taniteval.ci (`Rig(heavy=False)` loads ci.py and the DAG-ORC rig's loaders only; no torch, no tanitad).
"""
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlaunch as M  # noqa: E402


@pytest.fixture(scope="module")
def rig():
    return M.Rig(heavy=False)


def _g_with_chord(v6, n=1):
    """[n, 30, 2] GT whose LAST chord (knots 29 -> 30) is exactly ``v6`` m/s (0.2 s steps)."""
    g = np.zeros((n, 30, 2))
    g[:, 28, 0] = 10.0
    g[:, 29, 0] = 10.0 + 0.5            # dx = 0.5 -> 0.5 / 0.2 = 2.5 exactly in float64
    return g


# ---------------------------------------------------------------------------------------------------------------- the launch subset (hmc_read.py:386-387)
def test_launch_subset_boundaries():
    g = _g_with_chord(0)
    v6 = float(np.linalg.norm(g[0, 29] - g[0, 28]) / 0.2)
    assert v6 == 2.5
    # v0 + 1 == v6 exactly -> a launch (>=); v0 below the chord by less than 1 -> not
    assert M.launch_subset([v6 - 1.0], g)[0]
    assert not M.launch_subset([v6 - 0.999], g)[0]
    # v0 strictly below 6: v0 == 6.0 is excluded even when the chord is large
    big = np.zeros((2, 30, 2))
    big[:, 29, 0] = 10.0                 # chord 50 m/s
    assert M.launch_subset([5.999, 6.0], big).tolist() == [True, False]
    # NaN v0 gives False, not an exception
    assert M.launch_subset([float("nan")], g).tolist() == [False]


def test_launch_subset_loop_agrees_with_vectorised():
    rng = np.random.default_rng(1)
    n = 500
    g = np.cumsum(rng.uniform(0.0, 1.5, size=(n, 30, 2)), axis=1)
    v0 = rng.uniform(0.0, 9.0, size=n)
    v0[::37] = np.nan
    a, b = M.launch_subset(v0, g), M.launch_subset_loop(v0, g)
    assert a.dtype == bool and np.array_equal(a, b) and 0 < a.sum() < n


# ---------------------------------------------------------------------------------------------------------------- the speeds
def test_closed_form_state_speed_on_constant_velocity():
    v = np.array([0.0, 2.0, 7.5])
    t = (np.arange(1, 31) * M.DT)[None, :]
    g = np.stack([v[:, None] * t, np.zeros((3, 30))], -1)
    for n in (1, 2):
        assert np.allclose(M.gt_state_speed_cf(g, n), v, atol=1e-12)
        assert np.allclose(M.chord_speed(g, n), v, atol=1e-12)
    assert np.allclose(M.own_state_speed_cf(g), v, atol=1e-12)


def test_closed_form_state_speed_is_the_central_difference_not_the_chord():
    """A kink at the tick knot: the central difference (what hermite_frames writes) and the chord differ, and the code must return the central one."""
    g = np.zeros((1, 30, 2))
    g[0, :, 0] = np.arange(1, 31) * 1.0              # 5 m/s ...
    g[0, 10:, 0] = g[0, 9, 0] + np.arange(1, 21) * 3.0   # ... then 15 m/s from knot 11 on
    cf = M.gt_state_speed_cf(g, 1)[0]
    assert cf == pytest.approx((3.0 + 1.0) / 0.4)   # |g[10] - g[8]| = 1 + 3 over 0.4 s
    assert M.chord_speed(g, 1)[0] == pytest.approx(1.0 / 0.2)


def test_window_value_is_the_mean_over_ticks_1_2_of_the_difference():
    own = np.array([[9.0, 1.0, 2.0], [9.0, 3.0, 5.0], [9.0, 7.0, 7.0]])
    gt = np.array([[0.0, 2.0, 4.0], [0.0, 3.0, 8.0], [0.0, 0.0, 0.0]])
    mask = np.array([True, True, False])
    v = M.window_value(own, gt, mask)                # tick 0 (9.0) must not enter
    assert v.tolist() == [(-1.0 + -2.0) / 2, (0.0 + -3.0) / 2]
    assert M.window_value(own, gt, mask, ticks=(2,)).tolist() == [-2.0, -3.0]


# ---------------------------------------------------------------------------------------------------------------- the estimator and the verdict
def _cell(rig, x_shift, n_ep=12, per=10, seed=0):
    rng = np.random.default_rng(seed)
    N = n_ep * per
    eid = np.repeat(np.arange(n_ep), per)
    gt = np.zeros((N, 3))
    ep_eff = np.repeat(rng.normal(0.0, 0.2, n_ep), per)
    y = np.zeros((N, 3))
    y[:, 1:] = -1.0 + ep_eff[:, None] + rng.normal(0.0, 0.05, (N, 2))       # Y slow by ~1 m/s
    x = y.copy()
    x[:, 1:] += x_shift
    mask = np.ones(N, bool)
    return M.cell_stats(rig, x, y, gt, mask, eid, n_boot=300), x, y, gt, mask, eid


def test_x_equals_y_reads_exactly_zero(rig):
    c, x, y, gt, mask, eid = _cell(rig, 0.0)
    r = rig.PB(M.window_value(y, gt, mask), M.window_value(y, gt, mask), eid[mask], n_boot=300, seed=0)
    assert (r["delta"], r["lo"], r["hi"]) == (0.0, 0.0, 0.0) and r["separated"] is False and M.pb_zero(r)
    assert c["x_minus_y"]["delta"] == 0.0 and not c["moved"]


def test_toward_zero_is_moved_away_is_separated_but_not_moved(rig):
    toward = _cell(rig, +0.5)[0]
    away = _cell(rig, -0.5)[0]
    assert toward["x_minus_y"]["separated"] and toward["abs_x_minus_abs_y"]["delta"] < 0 and toward["moved"]
    assert toward["x_minus_y_unrounded"] == pytest.approx(0.5, abs=1e-12)           # the injected effect is read back exactly
    assert away["x_minus_y"]["separated"] and away["abs_x_minus_abs_y"]["delta"] > 0 and not away["moved"]
    assert M.is_moved_sign_reading(toward) and not M.is_moved_sign_reading(away)


def test_overshoot_past_zero_is_separated_but_not_toward_zero(rig):
    """X = Y + 2: |X| = |Y| (mirror image), so the contrast |X| - |Y| is ~0 and the cell must NOT read moved although X - Y is separated."""
    c = _cell(rig, +2.0)[0]
    assert c["x_minus_y"]["separated"]
    assert not c["abs_x_minus_abs_y"]["separated"] and not c["moved"]


def test_v_gt_cancels_in_x_minus_y_but_not_in_the_direction(rig):
    c, x, y, gt, mask, eid = _cell(rig, +0.5)
    shuffled = np.random.default_rng(3).permutation(len(gt))
    gt2 = gt.copy()
    gt2[:, 1:] = np.random.default_rng(4).uniform(5, 15, size=(len(gt), 2))[shuffled]
    c2 = M.cell_stats(rig, x, y, gt2, mask, eid, n_boot=300)
    assert c2["x_minus_y_unrounded"] == pytest.approx(c["x_minus_y_unrounded"], abs=1e-12)
    assert c2["x"]["mean"] != pytest.approx(c["x"]["mean"], abs=1e-3)


def test_route_needs_all_four_cells():
    four = {"main_s0": True, "main_s1": True, "replicate_s0": True, "replicate_s1": True}
    assert M.route(four)[0] == "moved" and "LOOP-STATE" in M.route(four)[1]
    for k in four:                                    # mutation: any single failing cell flips the word
        m = dict(four)
        m[k] = False
        w, nxt = M.route(m)
        assert w == "not moved" and "LAUNCH-INTENT-HI" in nxt
    assert M.route({"main_s0": True, "main_s1": True, "replicate_s0": True})[0] == "not moved"      # a missing cell is not a pass
    assert M.route({})[0] == "not moved"


def test_abs_mean_reducer_name_and_value():
    assert M.abs_mean(np.array([-1.0, -3.0])) == 2.0 and M.abs_mean.__name__ == "abs_mean"


# ---------------------------------------------------------------------------------------------------------------- the real data (skipped where the box-2 outputs are not mounted)
@pytest.mark.skipif(not os.path.exists(M.MANIFEST), reason="launch-intent T1 outputs not mounted")
def test_launch_window_count_on_the_banked_dump():
    import glob
    d = M.PINS  # noqa: F841  (the pins are exercised by Rig(); here only the count)
    man, t1s = M.load_manifest(M.MANIFEST)
    zs = [np.load(f) for f in sorted(glob.glob(os.path.join(t1s[("CTRL", 0)]["dump_dir"], "ep*.npz")))]
    g = np.concatenate([z["g"] for z in zs])
    v0 = np.concatenate([z["v0"] for z in zs])
    eid = np.concatenate([z["eid"] for z in zs])
    m = M.launch_subset(v0, g)
    assert (int(m.sum()), int(len(np.unique(eid[m])))) == M.INHERITED_LAUNCH
