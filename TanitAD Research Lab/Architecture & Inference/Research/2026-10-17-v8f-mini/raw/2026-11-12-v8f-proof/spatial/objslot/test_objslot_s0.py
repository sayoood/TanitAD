"""Tests of objslot_s0 on synthetic data (no real targets). Run: pytest with --basetemp C:/Users/Admin/v8proof_c/objslot/pt/run1."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import objslot_s0 as O  # noqa: E402


def synth(n=500, seed=1, accel_noise=0.15):
    """Windows with a lead: the ego speed random-walks, the lead speed changes slowly (so a constant-lead-speed rule is good, not perfect)."""
    rng = np.random.default_rng(seed)
    V = 10 + np.cumsum(rng.normal(0, 0.25, (n, 16)), axis=1)
    V = np.abs(V) + 1.0
    vL = V[:, 4] + rng.uniform(-2.5, 2.5, n)
    base = rng.uniform(15, 60, n)
    G = np.full((n, 15), np.nan)
    G[:, 4] = base
    vl = vL.copy()
    vls = [None] * 15
    for f in range(4, 14):
        vl = vl + rng.normal(0, accel_noise, n)
        G[:, f + 1] = G[:, f] + O.DT * (vl - 0.5 * (V[:, f] + V[:, f + 1]))
    vl = vL.copy()
    for f in range(3, -1, -1):
        vl = vl + rng.normal(0, accel_noise, n)
        G[:, f] = G[:, f + 1] - O.DT * (vl - 0.5 * (V[:, f] + V[:, f + 1]))
    fold = rng.integers(0, 5, n)
    return G, V, fold


def test_planted_constant_lead_speed_reads_one():
    G, V, fold = synth()
    r = O.control_o1(V, fold)
    assert r["pass"], r
    assert abs(r["r2_skill"] - 1.0) <= 1e-9 and r["r1_skill"] < 1.0 - 1e-6


def test_copy_reads_zero_and_identity():
    G, V, fold = synth()
    P = O.label_pipeline(G, V, 10, fold)
    r = O.control_o3(P)
    assert r["pass"] and r["copy_skill"] == 0.0 and r["true_as_rule_skill"] == 1.0


def test_shuffled_target_reads_at_most_002():
    G, V, fold = synth(n=800)
    P = O.label_pipeline(G, V, 10, fold)
    ep, n_ep = O.epi_of(np.array(["e%d" % (i // 5) for i in range(len(G))]), P["scored"])
    C = O.RP.boot_counts(n_ep, 50, 0)
    r = O.control_o2(lambda yp: O.label_pipeline(G, V, 10, fold, y=yp), P["y"], P["valid"], ep, n_ep, C)
    assert r["pass"], r


def test_rules_read_real_skill_on_planted_world():
    G, V, fold = synth(n=800)
    P = O.label_pipeline(G, V, 10, fold)
    y, v = P["y"], P["valid"]
    s2 = O.skill_point(y, P["pred"]["R2"], P["pred"]["R0"], v)
    s1 = O.skill_point(y, P["pred"]["R1"], P["pred"]["R0"], v)
    assert s2 > 0.9 and s2 > s1


def test_leak_detector_sees_a_time_shift():
    G, V, fold = synth(n=800)
    r = O.control_o5(G, V, 10, fold)
    assert r["pass"], r
    # a series with NO dynamics (random gap per frame, independent of time) must not pass for free: shifted reads no better
    rng = np.random.default_rng(5)
    Gr = np.repeat(rng.uniform(20, 50, (800, 1)), 15, axis=1) + rng.normal(0, 0.2, (800, 15))
    r2 = O.control_o5(Gr, V, 10, fold)
    assert abs(r2["r1_shifted_history_skill"] - r2["r1_skill"]) < 0.5


def test_fold_leak_in_r3_is_caught():
    G, V, fold = synth(n=600)
    P = O.label_pipeline(G, V, 10, fold)
    inp, y, valid = P["inp"], P["y"], P["valid"]
    p_ok, _ = O.fit_r3(inp, y, valid, fold)
    te = fold == 2
    y2 = y.copy()
    y2[te] = y2[te] + np.random.default_rng(0).normal(0, 30, int(te.sum()))
    p_mut, _ = O.fit_r3(inp, y2, valid, fold)
    assert np.array_equal(p_ok[te], p_mut[te])          # held-out targets do not enter the held-out fold's predictions
    # the leaky variant (fit on all windows, including the held-out fold) must differ, so the test above can fail
    leak = np.zeros(len(y), int) - 1                    # fold = -1 for everybody: the fit sees every window, the apply sees fold f only
    fin = np.isfinite(inp["g0"]) & np.isfinite(inp["Tc"]) & np.isfinite(inp["X"])
    A = np.c_[inp["Tc"][valid & fin], inp["X"][valid & fin], np.ones(int((valid & fin).sum()))]
    coef_all = np.linalg.lstsq(A, (y - inp["g0"])[valid & fin], rcond=None)[0]
    leaky = inp["g0"][te] + np.c_[inp["Tc"][te], inp["X"][te], np.ones(int(te.sum()))] @ coef_all
    assert not np.array_equal(leaky, p_ok[te])
    A2 = np.c_[inp["Tc"][valid & fin], inp["X"][valid & fin], np.ones(int((valid & fin).sum()))]
    coef_mut = np.linalg.lstsq(A2, (y2 - inp["g0"])[valid & fin], rcond=None)[0]
    leaky_mut = inp["g0"][te] + np.c_[inp["Tc"][te], inp["X"][te], np.ones(int(te.sum()))] @ coef_mut
    assert not np.array_equal(leaky, leaky_mut)         # a fold-leaking fit moves when the held-out targets move: the mutation check would catch it
    assert leak.max() == -1


def test_history_run_and_slope():
    G = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                  [np.nan, 2.0, 3.0, 4.0, 5.0],
                  [1.0, 20.0, 3.0, 4.0, 5.0],
                  [1.0, 2.0, 3.0, 4.0, np.nan]])
    m = O.hist_run(G)
    assert m[0].all() and m[1].tolist() == [False, True, True, True, True]
    assert m[2].tolist() == [False, False, True, True, True]      # the jump breaks the run: another lead
    assert not m[3].any()
    c = O.ls_slope(G, m)
    assert abs(c[0] - 5.0) < 1e-12 and abs(c[1] - 5.0) < 1e-12 and np.isnan(c[3])


def test_scored_mask_jump_and_missing():
    G = np.full((3, 15), 30.0)
    G[1, 9] = 40.0          # a 10 m step: another vehicle
    G[2, 7] = np.nan
    ok, cnt = O.scored_mask(G, 10)
    assert ok.tolist() == [True, False, False]
    assert cnt["of_which_jump_above_step_max"] == 1 and cnt["of_which_gap_missing_in_between"] == 1


def test_vision_path_reads_no_state_after_frame_4():
    class Fit:
        def __call__(self, X):
            return X[:, 0] * 2.0
    rng = np.random.default_rng(0)
    Z1 = rng.standard_normal((20, 16, 3)).astype(np.float32)
    fold = np.arange(20) % 5
    fits = {f: Fit() for f in range(5)}
    a = O.vision_gaps(fits, Z1, fold)
    Z2 = Z1.copy()
    Z2[:, 5:] = rng.standard_normal(Z2[:, 5:].shape)
    assert np.array_equal(a, O.vision_gaps(fits, Z2, fold))
    assert a.shape == (20, 5)


def test_signature_control():
    assert O.control_o6_label_inputs()["pass"]
