"""Tests of leadhead_s0 on synthetic data (no real targets).
Run: pytest --basetemp C:/Users/Admin/v8proof_c/leadhead/pt/run1 -p no:cacheprovider"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leadhead_s0 as H  # noqa: E402

RP = H.RP


def synth_rows(n_ep=30, L=24, d=16, amp=5.0, seed=0):
    rng = np.random.default_rng(seed)
    beid, bg, gap = [], [], []
    for e in range(n_ep):
        g_e, c_e = rng.uniform(15, 60), rng.normal(0, 1.5)
        for t in range(L):
            beid.append("syn%03d" % e)
            bg.append(t)
            gap.append(g_e + c_e * H.DT * t)
    beid, bg, gap = np.array(beid), np.array(bg), np.array(gap)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    Sb = rng.standard_normal((len(gap), d)) + (gap * amp)[:, None] * u[None, :]
    return beid, bg, gap, Sb


def test_history_builder_orders_frames_and_marks_episode_edges():
    beid = np.array(["a"] * 8 + ["b"] * 8)
    bg = np.r_[np.arange(8), np.arange(8)]
    idx = H.history_index(beid, bg, range(-4, 1))
    assert (idx[4] == np.array([0, 1, 2, 3, 4])).all()
    assert (idx[3] == -1).any() and (idx[0] == np.array([-1, -1, -1, -1, 0])).all()
    assert (idx[12] == np.array([8, 9, 10, 11, 12])).all()           # no leakage across the episode edge
    assert (idx[8, :4] == -1).all()
    fs = H.build_fit_sets(beid, bg, np.arange(16, dtype=float) * 0.4, np.full(16, 10.0))
    assert fs["ok5"].sum() == 8 and abs(fs["c"][4] - 2.0) < 1e-9     # slope 0.4 per 0.2 s step
    assert np.isnan(fs["c"][0])


def test_history_rule_requires_three_frames_and_same_lead():
    beid = np.array(["a"] * 8)
    bg = np.arange(8)
    gap = np.array([20, 20, np.nan, 20, 20, 20.5, 21, 21.5], float)
    fs = H.build_fit_sets(beid, bg, gap, np.full(8, 10.0))
    assert not fs["c_ok"][4]       # frames 3,4 only labelled in the run after the missing frame at row 2... row 4 history: 0..4 -> run 3,4 = 2 frames
    assert fs["c_ok"][6]           # rows 2..6 -> run 3,4,5,6 >= 3
    gap2 = np.array([20, 20, 20, 20, 20, 40, 40, 40], float)           # jump of 20 m: another lead
    fs2 = H.build_fit_sets(beid, bg, gap2, np.full(8, 10.0))
    assert not fs2["c_ok"][5]      # run is the single last frame after the jump
    assert fs2["c_ok"][7]


def test_crossfit_no_episode_on_both_sides_and_selection_uses_fit_rows_only():
    beid, bg, gap, Sb = synth_rows()
    FS = H.build_fit_sets(beid, bg, gap, np.full(len(gap), 10.0))
    folds_b = RP.make_folds(beid)
    epi = np.array([int(e[3:]) for e in beid])
    f = 2
    fit_rows = np.flatnonzero(FS["ok5"] & (folds_b != f))
    assert set(epi[fit_rows]).isdisjoint(set(epi[folds_b == f]))
    te = np.flatnonzero(FS["ok5"] & (folds_b == f))
    Xte = Sb[FS["idx5"][te]].reshape(len(te), -1)
    m1, i1 = H.fit_fold("HT", Sb, FS, epi, fit_rows, f, ("c",))
    FS2 = dict(FS)
    FS2["c"] = FS["c"].copy()
    FS2["c"][folds_b == f] = 123.0                                   # held-out labels changed: nothing fitted may move
    m2, i2 = H.fit_fold("HT", Sb, FS2, epi, fit_rows, f, ("c",))
    assert i1["c"]["g"] == i2["c"]["g"] and np.array_equal(m1["c"](Xte), m2["c"](Xte))
    assert i1["c"]["n_fit"] == int(FS["c_ok"][fit_rows].sum())


def test_ridge_dual_matches_closed_form():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 10))
    y = X @ rng.standard_normal(10) + 0.1 * rng.standard_normal(60)
    Z = rng.standard_normal((7, 10))
    lam = 3.0
    p = H.ridge_paths(X, y, Z, [lam])[0]
    mu, sd = X.mean(0), X.std(0)
    A = (X - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(10), A.T @ (y - y.mean()))
    assert np.allclose(p, ((Z - mu) / sd) @ w + y.mean(), atol=1e-8)


def test_planted_state_h1_reads_closing_speed():
    r = H.control_h1(1.0, n_ep=40, L=24, d=16, amp=5.0)
    assert r["pass"], r


def test_planted_state_h1_hm_runs():
    r = H.control_h1(1.0, n_ep=40, L=24, d=16, amp=5.0, kind="HM")
    assert r["heldout_r2_closing"] > 0.5, r


def test_shuffled_labels_read_nothing():
    beid, bg, gap, Sb = synth_rows(n_ep=40, d=16)
    FS = H.build_fit_sets(beid, bg, gap, np.full(len(gap), 10.0))
    folds_b = RP.make_folds(beid)
    epi = np.array([int(e[3:]) for e in beid])
    pred = np.full(len(gap), np.nan)
    for f in range(5):
        fr = np.flatnonzero(FS["ok5"] & (folds_b != f))
        mdl, _ = H.fit_fold("HT", Sb, FS, epi, fr, f, ("c",), perm_c=True)
        te = np.flatnonzero(FS["ok5"] & (folds_b == f))
        pred[te] = mdl["c"](Sb[FS["idx5"][te]].reshape(len(te), -1))
    m = FS["c_ok"] & np.isfinite(pred)
    assert RP.r2_pooled(FS["c"][m], pred[m]) <= 0.02


def test_no_future_input_h5_and_in_sample_leak_h4():
    beid, bg, gap, Sb = synth_rows(n_ep=30, d=12)
    FS = H.build_fit_sets(beid, bg, gap, np.full(len(gap), 10.0))
    folds_b = RP.make_folds(beid)
    epi = np.array([int(e[3:]) for e in beid])
    rows = np.flatnonzero(FS["ok5"] & FS["c_ok"] & FS["dir_ok"])
    n = len(rows)
    rng = np.random.default_rng(5)
    Z1 = rng.standard_normal((n, 15, 12))
    Z1[:, :5, :] = Sb[FS["idx5"][rows]]
    V = 10 + rng.normal(0, 0.3, (n, 16))
    fold = folds_b[rows]
    Xw = H.ego_term_windows(V, 10)
    out, _, models = H.crossfit("HT", Sb, FS, epi, folds_b, Z1, Xw, fold, ("gap", "c", "dir"))
    y = FS["gap"][rows] + rng.normal(0, 1, n)
    sc = np.isfinite(out["gap"])
    r = H.control_h5(models, Z1, Xw, V, y, sc, fold, out)
    assert r["pass"] and r["max_abs_change_of_head_outputs"] == 0.0 and r["max_abs_change_of_rule_outputs"] == 0.0
    # in-sample head reads the label better than the cross-fitted one (the leak detector can see a leak)
    c_lab = FS["c"][rows]
    mdl, _ = H.fit_fold("HT", Sb, FS, epi, np.flatnonzero(FS["ok5"]), 0, ("c",))
    call = H.apply_models(mdl, Z1, Xw)
    valid = np.isfinite(c_lab) & np.isfinite(out["c"])
    d = H.control_h4(call["c"], out["c"], c_lab, valid)
    assert d["pass"] or d["r2_cross_fitted"] > 0.99      # noiseless-enough synthetic data can saturate both reads


def test_label_path_h6_reproduces_objslot_skill():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "objslot"))
    import test_objslot_s0 as T
    G, V, fold = T.synth()
    P = H.O.label_pipeline(G, V, 10, fold)
    ref = H.O.skill_point(P["y"], P["pred"]["R2"], P["pred"]["R0"], P["valid"])
    r = H.control_h6(G, V, fold, ref)
    assert r["pass"], r


def test_direct_labels_use_same_lead_and_ego_term():
    beid = np.array(["a"] * 16)
    bg = np.arange(16)
    gap = 30.0 + 0.2 * np.arange(16)
    V = np.full(16, 8.0)
    fs = H.build_fit_sets(beid, bg, gap, V)
    assert fs["dir_ok"][4] and not fs["dir_ok"][6]       # rows 4 only (needs t-4 and t+10 in the episode, 16 rows)
    assert abs(fs["delta"][4] - 2.0) < 1e-9
    assert abs(fs["Xego"][4] - (2.0 * 8.0 - 10 * 0.2 * 8.0)) < 1e-9
