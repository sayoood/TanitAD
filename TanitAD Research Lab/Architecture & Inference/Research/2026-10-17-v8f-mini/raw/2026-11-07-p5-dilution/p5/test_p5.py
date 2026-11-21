"""Tests for the P5 read (dev-box CPU).  Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider test_p5.py  (cwd = this folder)
Controls required by the card's brief: (a) a synthetic series with a known noise level returns its known lambda; (b) a0 := a gives lambda = 1; (c) a deliberate-regression mutant
(drop the residualisation / forward difference / 0.2 s difference) changes the number, and a test catches it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import pytest
import p5_lib as L

W = L.synth_world(n_ep=20000, sigma=0.06, alpha=0.5, sigma_u=0.8, seed=0)
EXP = L.synth_expected_lambda(0.06, 0.5, 0.8, m=5)


def test_card_is_the_frozen_one():
    assert L.assert_card_frozen() == L.CARD_SHA


def test_sg_closed_form_equals_polyfit_and_scipy():
    from scipy.signal import savgol_filter
    rng = np.random.default_rng(3)
    v = np.cumsum(rng.normal(0, 0.05, 400)) + 5.0
    for m in (3, 5, 8):
        k0 = np.arange(m, len(v) - m)
        got = L.sg_accel(v, k0, m)
        ref_pf = np.array([np.polyfit(np.arange(-m, m + 1) * L.DT, v[k - m:k + m + 1], 2)[1] for k in k0])      # coefficient of t in c2 t^2 + c1 t + c0 = derivative at the centre
        ref_sp = savgol_filter(v, 2 * m + 1, 2, deriv=1, delta=L.DT)[k0]
        assert np.abs(got - ref_pf).max() < 1e-9
        assert np.abs(got - ref_sp).max() < 1e-9


def test_edges_are_nan_not_wrapped():
    v = np.arange(50, dtype=float)
    assert np.isnan(L.sg_accel(v, np.array([3]), 5)[0])             # frames -2..8: a negative index must not wrap to the end of the array
    assert np.isnan(L.a0_backward(v, np.array([0]), 1)[0])
    assert np.isfinite(L.sg_accel(v, np.array([5]), 5)[0])
    assert abs(L.sg_accel(v, np.array([20]), 5)[0] - 10.0) < 1e-9     # v = k: 10 m/s^2 at 0.1 s per frame
    assert abs(L.a0_backward(v, np.array([20]), 1)[0] - 10.0) < 1e-9


def test_synthetic_known_noise_returns_known_lambda():
    r = L.synth_read(W, "backward")
    g = L.gram_stats(r["a0"], r["a"], W["v0"], W["kappa0"], "lin", W["eid"], len(W["eid"]))
    bs = L.fast_cluster_bootstrap(g, 400, 0)
    assert abs(r["lam"] - EXP["lambda"]) < 3 * bs["sd"], (r["lam"], EXP["lambda"], bs)
    assert abs(r["lam"] - EXP["lambda"]) < 0.015
    assert 0.4 < EXP["lambda"] < 0.8        # a non-trivial known value (the control is not a degenerate 0 or 1)


def test_a0_replaced_by_a_gives_lambda_one():
    r = L.synth_read(W, "a_itself")
    assert abs(r["lam"] - 1.0) < 1e-12
    p = L.lam_parts(W["v0"], W["v0"], W["v0"], W["kappa0"])             # an arbitrary series used as both a0 and a: e = 0 exactly
    assert abs(p["lam"] - 1.0) < 1e-12


def test_mutant_no_residualisation_changes_the_number_and_misses_the_known_value():
    good = L.synth_read(W, "backward", design_kind="lin")["lam"]
    mut = L.synth_read(W, "backward", design_kind="none")["lam"]
    assert abs(good - EXP["lambda"]) < 0.015
    assert abs(mut - good) > 0.05                                         # the mutant moves the number ...
    assert abs(mut - EXP["lambda"]) > 0.05                                # ... and the known-lambda check rejects it
    assert abs(mut - EXP["lambda_without_residualisation"]) < 0.015       # (it computes exactly the quantity it claims to)


def test_mutant_a0_definitions_are_caught_by_the_gate():
    a0_model = L.a0_backward(W["v_flat"], W["k0"], 1)
    assert L.gate_compare(a0_model, L.a0_backward(W["v_flat"], W["k0"], 1))["pass"]
    assert not L.gate_compare(a0_model, L.a0_forward(W["v_flat"], W["k0"]))["pass"]            # forward difference
    assert not L.gate_compare(a0_model, L.a0_backward(W["v_flat"], W["k0"], 2))["pass"]        # the 0.2 s reading of the card's prose
    assert not L.gate_compare(a0_model[:500], a0_model[:500])["pass"]                          # fewer than 1,000 windows is not a pass


def test_gram_form_equals_direct_form():
    r = L.synth_read(W, "backward")
    g = L.gram_stats(r["a0"], r["a"], W["v0"], W["kappa0"], "lin", W["eid"], len(W["eid"]))
    assert abs(L.lam_from_gram(g.sum(0)) - r["lam"]) < 1e-9
    g2 = L.gram_stats(r["a0"], r["a"], W["v0"], W["kappa0"], "p10", W["eid"], len(W["eid"]))
    assert abs(L.lam_from_gram(g2.sum(0)) - L.lam_stat(r["a0"], r["a"], W["v0"], W["kappa0"], "p10")) < 1e-9


def test_fast_bootstrap_uses_ci_pys_resampling():
    ci = L.load_ci()
    rng = np.random.default_rng(5)
    n_ep, per = 60, 7
    eid = np.repeat(np.arange(n_ep), per)
    v0 = rng.uniform(3, 8, len(eid)); kap = rng.normal(0, .02, len(eid)); a = rng.normal(0, 1, len(eid)); a0 = a + rng.normal(0, .8, len(eid)) + 0.3 * v0
    names = np.array([f"ep{e:04d}" for e in eid])
    ref = ci.stat_cluster_bootstrap(list(range(len(eid))), names, lambda it: L.lam_stat(a0[np.asarray(it)], a[np.asarray(it)], v0[np.asarray(it)], kap[np.asarray(it)]), n_boot=200, seed=0)
    uniq, inv = np.unique(names, return_inverse=True)
    g = L.gram_stats(a0, a, v0, kap, "lin", inv, len(uniq))
    fast = L.fast_cluster_bootstrap(g, 200, 0)
    assert abs(ref["lo"] - fast["lo"]) < 1e-9 and abs(ref["hi"] - fast["hi"]) < 1e-9
    assert abs(ref["point"] - L.lam_from_gram(g.sum(0))) < 1e-9


def test_verdict_rule_as_written():
    R = L.RATIO
    assert L.verdict(0.62, 0.70)["word"] == "HOLDS"                      # window for both seeds is [0.566, 0.686]
    assert L.verdict(0.566, 0.70)["word"] == "HOLDS"
    assert L.verdict(0.69, 0.80)["word"] == "FAILS"                      # |0.586 - 0.69| > 0.10
    assert L.verdict(0.55, 0.65)["word"] == "FAILS"                      # |0.666 - 0.55| > 0.10
    assert L.verdict(0.62, 0.90)["word"] == "FAILS"                      # upper bound must be BELOW 0.90
    assert L.verdict(0.62, 0.8999)["word"] == "HOLDS"
    assert abs(L.GAIN_TOTAL["s0"] / L.GAIN_BASE - R["s0"]) < 5e-4 and abs(L.GAIN_TOTAL["s1"] / L.GAIN_BASE - R["s1"]) < 5e-4


@pytest.mark.skipif(not os.path.exists(L.HERE + "/raw/window_table_band.npz"), reason="window table not built")
def test_real_table_mutants_move_the_number_and_model_a0_equals_reimplementation():
    tab = np.load(L.HERE + "/raw/window_table_band.npz")
    rows = np.load(L.HERE + "/raw/p5_sample_rows.npy")
    base = L.lam_stat(tab["a0_model"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    none = L.lam_stat(tab["a0_model"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "none")
    fwd = L.lam_stat(tab["a0_fwd"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    lag2 = L.lam_stat(tab["a0_lag2"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    assert L.gate_compare(tab["a0_model"], tab["a0_indep"])["pass"]
    assert not L.gate_compare(tab["a0_model"], tab["a0_fwd"])["pass"]
    assert not L.gate_compare(tab["a0_model"], tab["a0_lag2"])["pass"]
    # MEASURED shifts of lambda on the real sample: forward 0.0019, 0.2 s 0.0060, no-residualisation 0.00003 (v0 and kappa0 explain ~0 of the variance of a0 here; the synthetic world, where a depends on v0, is
    # what catches that mutant decisively). All three must move the number; the two a0 mutants must move it by more than the bootstrap sd (about 0.001) too.
    assert abs(none - base) > 1e-6, ("none", none, base)
    for name, x in (("fwd", fwd), ("lag2", lag2)):
        assert abs(x - base) > 1e-3, (name, x, base)
