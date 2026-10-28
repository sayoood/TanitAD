"""Small tests for the pure layer of crawl_diag.py (numpy only; no model, no checkpoint, no GPU).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider test_crawl_diag.py
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("crawl_diag", os.path.join(HERE, "crawl_diag.py"))
cd = importlib.util.module_from_spec(_spec)
sys.modules["crawl_diag"] = cd
_spec.loader.exec_module(cd)
_cs = importlib.util.spec_from_file_location("ci", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/taniteval/taniteval/ci.py")
ci = importlib.util.module_from_spec(_cs)
sys.modules["ci"] = ci
_cs.loader.exec_module(ci)


def test_v_bucket_edges():
    v = [0.0, 0.49, 0.5, 1.49, 1.5, 2.99, 3.0, 5.99, 6.0, np.nan, -0.1]
    assert cd.v_bucket(v).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1]


def test_d_class_edges():
    d = [-1.0, -0.25, -0.2499, 0.0, 0.2499, 0.25, 3.0]
    assert cd.d_class(d).tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_table_own_cell_bias_is_exactly_zero():
    rng = np.random.default_rng(1)
    n = 5000
    vb = rng.integers(0, 4, n)
    dc = rng.integers(0, 3, n)
    vgt = rng.gamma(2.0, 2.0, (n, 2)) + vb[:, None]
    t = cd.build_table(vb, dc, vgt)
    cell = cd.cell_of(vb, dc)
    for c in range(12):
        m = cell == c
        assert m.sum() >= cd.MIN_CELL
        assert np.abs(np.median(t["median"][c][None] - vgt[m], axis=0)).max() < 1e-12
        assert np.abs((t["mean"][c][None] - vgt[m]).mean(axis=0)).max() < 1e-9


def test_table_fallback_to_speed_marginal():
    vb = np.array([0] * 100 + [1] * 10)
    dc = np.array([1] * 100 + [1] * 10)
    vgt = np.concatenate([np.full((100, 2), 2.0), np.full((10, 2), 5.0)])
    t = cd.build_table(vb, dc, vgt, min_cell=50)
    assert cd.cell_of(1, 1) in t["fallback_cells"]
    assert np.allclose(t["median"][cd.cell_of(1, 1)], 5.0)            # the marginal of speed bucket 1 is its own windows
    assert np.isnan(t["median"][cd.cell_of(2, 0)]).all() or cd.cell_of(2, 0) in t["fallback_cells"]


def test_predict_nan_without_bucket_and_vonly():
    vb = np.array([0, 0, 1, 1] * 40)
    dc = np.array([0, 2, 1, 1] * 40)
    vgt = np.stack([vb * 1.0 + (dc == 2), vb * 1.0], 1)
    t = cd.build_table(vb, dc, vgt, min_cell=10)
    p = cd.predict(t, np.array([0, -1]), np.array([0, 0]))
    assert not np.isnan(p[0]).any() and np.isnan(p[1]).all()
    pv = cd.predict(t, np.array([0]), np.array([2]), vonly=True)
    assert np.allclose(pv, t["marg_median"][0])


def test_perfect_predictor_gap_is_exactly_zero_and_sign():
    v = np.random.default_rng(2).random((50, 2))
    assert np.all(cd.blind_gap(v, v) == 0.0)
    assert np.allclose(cd.blind_gap(v + 0.3, v), 0.3)                 # a predictor that is too fast reads positive


def test_window_gap_equals_mlaunch_definition():
    rng = np.random.default_rng(3)
    own, gt = rng.random((20, 3)), rng.random((20, 3))
    assert np.allclose(cd.window_gap(own, gt), (own[:, 1] + own[:, 2] - gt[:, 1] - gt[:, 2]) / 2.0)


def _track(speeds):
    """GT track [1, 30, 2] along x with the given 30 chord speeds."""
    x = np.cumsum(np.asarray(speeds, float) * cd.DT)
    return np.stack([x, np.zeros_like(x)], 1)[None]


def test_stay_mask_reads_stay_launch_and_moving():
    stay = _track([0.0] * 30)
    launch = _track([0.0] * 20 + [3.0] * 10)
    blip = _track([0.0] * 5 + [1.5] + [0.0] * 24)                      # one chord over 1 m/s: a launch within the horizon
    g = np.concatenate([stay, launch, blip, stay])
    v0 = np.array([0.0, 0.0, 0.0, 0.8])
    assert cd.stay_mask(v0, g).tolist() == [True, False, False, False]


def test_out_speed_closed_form_on_constant_velocity_and_a_ramp():
    plan = _track([3.0] * 30)
    assert np.allclose(cd.out_speed(plan, 10), 3.0) and np.allclose(cd.out_speed(plan, 20), 3.0) and np.allclose(cd.out_chord_end(plan), 3.0)
    ramp = _track(np.linspace(0.0, 6.0, 30))
    # central difference of positions around knot K[10]; must sit between the chord speeds on either side
    s = cd.out_speed(ramp, 10)[0]
    assert np.linspace(0.0, 6.0, 30)[9] <= s <= np.linspace(0.0, 6.0, 30)[10] + 1e-9


def test_out_speed_agrees_with_mlaunch_cf_when_available():
    p = "E:/Projects/TanitAD-artifacts/dagland/launch_intent_mlaunch/mlaunch.py"
    if not os.path.exists(p):
        pytest.skip("mlaunch not present")
    sp = importlib.util.spec_from_file_location("mlaunch_t", p)
    ml = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(ml)
    plan = np.random.default_rng(4).normal(size=(7, 30, 2)).cumsum(1)
    assert np.allclose(cd.out_speed(plan, 10), ml.own_state_speed_cf(plan, m=10))


def test_speed_counterfactual_functions():
    v = np.array([np.nan, np.nan, 0.2, 0.0, 1.0])
    s = cd.shift_speed(v, -0.5)
    assert np.isnan(s[:2]).all() and np.allclose(s[2:], [0.0, 0.0, 0.5])      # clipped at 0, NaN rows stay NaN
    assert np.array_equal(cd.shift_speed(v, 0.0)[2:], v[2:])                   # delta = 0 is the identity (the exact-zero control)
    f = cd.flat_speed(v, 1.0)
    assert np.isnan(f[:2]).all() and np.allclose(f[2:], 1.0)
    z = cd.zero_speed(v)
    assert np.isnan(z[:2]).all() and np.allclose(z[2:], 0.0)


def test_donors_other_episode_deterministic_and_foreign():
    eid = np.repeat(np.arange(6), 5)
    a = cd.donors_other_episode(eid, np.arange(len(eid)), seed=0)
    b = cd.donors_other_episode(eid, np.arange(len(eid)), seed=0)
    assert np.array_equal(a, b) and (eid[a] != eid).all()
    with pytest.raises(ValueError):
        cd.donors_other_episode(np.zeros(4, int), np.arange(4))


def test_ratio_ci_recovers_a_known_slope_and_flags_undefined():
    rng = np.random.default_rng(5)
    n = 200
    eid = np.repeat(np.arange(20), 10)
    dvi = np.full(n, 0.5)
    dvo = 0.7 * dvi
    r = cd.ratio_ci(ci, dvo, dvi, eid, n_boot=200)
    assert abs(r["S_v"] - 0.7) < 1e-12 and abs(r["lo"] - 0.7) < 1e-9 and abs(r["hi"] - 0.7) < 1e-9
    noisy = dvo + rng.normal(0, 0.2, n)
    r2 = cd.ratio_ci(ci, noisy, dvi, eid, n_boot=300)
    assert r2["lo"] < r2["S_v"] < r2["hi"]
    assert cd.ratio_ci(ci, dvo, np.zeros(n), eid, n_boot=10)["S_v"] is None


def test_ratio_ci_mutation_sign_flip_is_detected():
    eid = np.repeat(np.arange(10), 4)
    dvi = np.full(40, 1.0)
    good = cd.ratio_ci(ci, 0.8 * dvi, dvi, eid, n_boot=100)["S_v"]
    bad = cd.ratio_ci(ci, -0.8 * dvi, dvi, eid, n_boot=100)["S_v"]
    assert good > 0.5 > -0.5 > bad


def test_paired_ratio_difference_is_exact_when_arms_are_constant_and_shares_episodes():
    eid = np.repeat(np.arange(12), 5)
    dvi = np.full(60, 1.0)
    r = cd.paired_ratio_ci(ci, 0.9 * dvi, dvi, 0.4 * dvi, dvi, eid, n_boot=200)
    assert abs(r["delta"] - 0.5) < 1e-9 and r["separated"] and not r["contains_zero"]
    z = cd.paired_ratio_ci(ci, 0.6 * dvi, dvi, 0.6 * dvi, dvi, eid, n_boot=200)
    assert z["delta"] == 0.0 and z["contains_zero"] and not z["separated"]


def test_verdict_h01_total_over_outcomes():
    assert cd.verdict_h01(-0.6, -0.8, -0.3, -0.1)["outcome"] == "a"
    assert cd.verdict_h01(-0.6, -0.8, -0.3, +0.1)["outcome"] == "neither"       # a positive GT-stay gap blocks (a)
    assert cd.verdict_h01(-0.6, -0.8, +0.1, -0.1)["outcome"] == "neither"       # a CI that includes 0 blocks (a)
    assert cd.verdict_h01(0.05, -0.1, 0.15, 0.0)["outcome"] == "b"
    assert cd.verdict_h01(0.05, -0.3, 0.15, 0.0)["outcome"] == "b_point_only"
    assert cd.verdict_h01(-0.3, -0.5, -0.1, 0.0)["outcome"] == "neither"        # between -0.4 and -0.2: neither card outcome
    assert cd.verdict_h01(-0.4, -0.6, -0.1, 0.0)["outcome"] == "a"              # the bar is inclusive, as the card writes "<="


def test_verdict_h02_fires_supports_or_is_between():
    mk = lambda s, lo, hi: {"S_v": s, "lo": lo, "hi": hi}                         # noqa: E731
    assert cd.verdict_h02(mk(0.1, -0.1, 0.3), mk(0.2, 0.0, 0.4))["outcome"] == "falsifier_1_fires"
    assert cd.verdict_h02(mk(0.1, -0.1, 0.3), mk(0.2, 0.0, 0.6))["outcome"] == "between"       # one run's upper CI reaches 0.5
    assert cd.verdict_h02(mk(0.7, 0.6, 0.8), mk(0.9, 0.7, 1.0))["outcome"] == "supported"
    assert cd.verdict_h02(mk(0.7, 0.6, 0.8), mk(0.4, 0.3, 0.5))["outcome"] == "between"
    assert cd.verdict_h02(mk(None, None, None), mk(0.2, 0.0, 0.4))["outcome"] == "undefined"
