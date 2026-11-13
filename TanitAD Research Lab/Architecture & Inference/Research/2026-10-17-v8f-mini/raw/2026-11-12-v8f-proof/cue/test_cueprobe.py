"""Fast unit tests for cueprobe.py on synthetic data (numpy only, no cache, no torch).  Run: python -m pytest -q test_cueprobe.py   (CUDA_VISIBLE_DEVICES=-1)"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cueprobe as CP  # noqa: E402


def straight_plans(n, rng):
    v = rng.uniform(3.0, 14.0, size=(n, 1))
    x = np.cumsum(np.repeat(v, 30, axis=1) * 0.2, axis=1)
    return np.stack([x, np.zeros_like(x)], axis=2)


def test_conformal_margin_coverage_exchangeable():
    rng = np.random.default_rng(1)
    cov = []
    for _ in range(300):
        d = rng.uniform(5, 60, 100)
        dh = d + rng.normal(2.0, 4.0, 100)
        q = CP.conformal_q(dh - d)
        dt = rng.uniform(5, 60, 5000)
        dht = dt + rng.normal(2.0, 4.0, 5000)
        cov.append(np.mean(dht - q <= dt))
    m = float(np.mean(cov))
    assert 0.89 <= m <= 0.925, m                       # nominal 0.9 (finite-sample rank 91/101 = 0.901)


def test_conformal_q_order_statistic():
    r = np.arange(1.0, 11.0)                           # n = 10, ceil(11 * 0.9) = 10 -> the 10th smallest
    assert CP.conformal_q(r) == 10.0
    assert CP.conformal_q(np.arange(1.0, 101.0)) == 91.0
    assert np.isnan(CP.conformal_q([]))


def test_tau_hits_false_positive_bound_on_training_data():
    rng = np.random.default_rng(2)
    neg = rng.normal(0, 1, 5000)
    tau = CP.tau_for_fpr(neg, 0.02)
    assert (neg >= tau).mean() <= 0.02
    lower = np.unique(neg)[np.unique(neg) < tau].max()  # the next smaller candidate would break the bound: tau is the smallest
    assert (neg >= lower).mean() > 0.02
    tied = np.zeros(1000)                              # all tied: no threshold at or below the tie can satisfy 2 %, tau must sit above
    assert (tied >= CP.tau_for_fpr(tied)).mean() == 0.0
    assert CP.tau_for_fpr(np.array([])) == np.inf
    few = np.array([0.1, 0.2, 0.3])                    # kmax = 0 -> above the largest
    assert (few >= CP.tau_for_fpr(few)).sum() == 0


def test_leak_guard_fires_on_a_later_frame():
    ws = np.array([5, 9, 40, 77])
    assert CP.assert_rows_not_after(ws, ws)
    assert CP.assert_rows_not_after(ws - 3, ws)
    bad = ws.copy()
    bad[2] += 1
    with pytest.raises(AssertionError):
        CP.assert_rows_not_after(bad, ws)
    with pytest.raises(AssertionError):
        CP.assert_rows_not_after(ws + CP.LATE_STEPS, ws)   # the k4 deliberate regression: 2.0 s later


def test_k3_no_active_window_is_bit_identical():
    rng = np.random.default_rng(3)
    plan = straight_plans(40, rng)
    out, rec = CP.apply_d0(plan, [None] * 40)
    assert np.array_equal(out, plan) and all(r is None for r in rec)
    out2, _ = CP.apply_d0(plan, [np.nan] * 40)
    assert np.array_equal(out2, plan)
    stop = np.zeros(40, bool)
    d0, act = CP.arm_d0("PF-PD", plan, stop, np.full(40, 0.3), np.full(40, np.inf), np.full(40, 20.0), np.full(40, 1.0), np.full(40, np.nan))
    assert not act.any() and all(x is None for x in d0)
    out3, _ = CP.apply_d0(plan, d0)
    assert np.array_equal(out3, plan)
    # sanity: a binding budget does change a plan (the test above is not vacuous)
    out4, rec4 = CP.apply_d0(plan[:1], [5.0])
    assert not np.array_equal(out4, plan[:1]) and rec4[0] is not None


def test_arm_definitions():
    rng = np.random.default_rng(4)
    plan = straight_plans(4, rng)
    stop = np.array([True, True, False, False])
    flag, tau = np.array([0.9, 0.1, 0.9, 0.1]), np.full(4, 0.5)
    dhat, q, dtrue = np.full(4, 30.0), np.full(4, 2.0), np.array([25.0, 26.0, np.nan, np.nan])
    pfpd, _ = CP.arm_d0("PF-PD", plan, stop, flag, tau, dhat, q, dtrue)
    ofpd, _ = CP.arm_d0("OF-PD", plan, stop, flag, tau, dhat, q, dtrue)
    pfod, _ = CP.arm_d0("PF-OD", plan, stop, flag, tau, dhat, q, dtrue)
    ofod, _ = CP.arm_d0("OF-OD", plan, stop, flag, tau, dhat, q, dtrue)
    assert pfpd == [28.0, None, 28.0, None]
    assert ofpd == [28.0, 28.0, None, None]
    assert pfod == [25.0, None, 28.0, None]
    assert ofod == [25.0, 26.0, None, None]


def test_folds_keep_episodes_disjoint():
    rng = np.random.default_rng(5)
    eid = np.repeat(np.arange(141), rng.integers(5, 30, 141)).astype(str)
    for seed in range(5):
        f = CP.RP.make_folds(eid, seed=seed)
        for e in np.unique(eid):
            assert len(np.unique(f[eid == e])) == 1
        assert set(f) == set(range(5))


def test_median_gain_bootstrap_sign():
    rng = np.random.default_rng(6)
    ep = np.repeat(np.arange(30), 4)
    ref = np.abs(rng.normal(10, 3, 120))
    good = ref - 4.0
    dr = CP.make_idx_draws(ep, 300)
    g = CP.boot_median_gain(dr, ref, good)
    assert g["gain"] > 3.5 and g["separated_better"]
    assert not CP.boot_median_gain(dr, ref, ref)["separated_better"]


def test_pipeline_runs_and_recovers_planted_signal():
    """Synthetic windows: 60 episodes x 8 windows, tokens = noisy copy of a latent (stop flag, rest distance); the whole fold pipeline must run and find the planted cue."""
    rng = np.random.default_rng(7)
    n_ep, per = 60, 8
    eid = np.repeat(np.arange(n_ep), per)
    N = len(eid)
    stop = rng.random(N) < 0.25
    d = np.where(stop, rng.uniform(8, 40, N), np.nan)
    lat = np.zeros((N, 6))
    lat[:, 0] = stop * 2.0 - 1.0
    lat[:, 1] = np.where(stop, d / 20.0, 0.0)
    X = np.concatenate([lat, rng.normal(0, 0.3, (N, 120))], 1) + rng.normal(0, 0.15, (N, 126))
    K = (X @ X.T).astype(np.float32)
    E = rng.normal(0, 1, (N, 8))
    eid_str = eid.astype(str)
    valid = np.ones(N, bool)
    res = CP.pipeline(0, {"V40": K, "V160": K}, E, eid_str, eid, valid, stop, stop.astype(float), {0: d})
    auc = {s: CP.RP.auc_pooled(stop.astype(float), res["flag_oof"][s]) for s in CP.SETS}
    assert auc["V40"] > 0.9 and abs(auc["E"] - 0.5) < 0.15
    P = res["plan"][0]
    cv = CP.coverage(P, d, stop)
    assert cv["n"] == int(stop.sum()) and cv["share"] >= 0.75           # nominal 0.9 on exchangeable data; loose bound for n ~ 120
    dd = CP.read_dist(P, d, stop, eid, None)
    assert dd["sets"]["V40"]["median_abs_err_m"] < dd["const_floor"]["median_abs_err_m"]
    assert all(c in CP.SETS for c in P["chosen"])
    assert all(f is not None and f <= 0.2 for f in P["tau_in_train_fpr"])
