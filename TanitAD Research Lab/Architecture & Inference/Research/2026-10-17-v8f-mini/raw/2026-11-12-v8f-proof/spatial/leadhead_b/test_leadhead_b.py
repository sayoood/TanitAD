"""Tests of leadhead_b on synthetic data (no real targets).
Run: pytest --basetemp C:/Users/Admin/v8proof_c/leadhead_b/pt/run1 -p no:cacheprovider"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leadhead_b as B  # noqa: E402

H = B.H


def synth(n_clip=40, L=24, d=12, amp=5.0, seed=0):
    rng = np.random.default_rng(seed)
    names, t, gap = [], [], []
    for e in range(n_clip):
        g_e, c_e = rng.uniform(15, 60), rng.normal(0, 1.5)
        for k in range(L):
            names.append("clip%03d" % e)
            t.append(k)
            gap.append(g_e + c_e * H.DT * k)
    names, t, gap = np.array(names), np.array(t), np.array(gap)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    Z = (rng.standard_normal((len(gap), d)) + (gap * amp)[:, None] * u[None, :]).astype(np.float32)
    return names, t, gap, Z


def test_train_row_history_builder_orders_frames_and_stops_at_clip_edges():
    names = np.array(["a"] * 8 + ["b"] * 8)
    t = np.r_[np.arange(8), np.arange(8)]
    gap = np.r_[np.arange(8), np.arange(8)] * 0.4 + 20.0
    FS = H.build_fit_sets(names, t, gap, np.full(16, 10.0))
    assert (FS["idx5"][4] == np.array([0, 1, 2, 3, 4])).all()
    assert (FS["idx5"][12] == np.array([8, 9, 10, 11, 12])).all()          # no history across the clip edge
    assert (FS["idx5"][8, :4] == -1).all()
    assert FS["ok5"].sum() == 8 and abs(FS["c"][4] - 2.0) < 1e-9
    Z = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
    X = B.make_getX(Z, FS["idx5"], np.array([4, 12]))(0, 2)
    assert X.shape == (2, 15) and (X[0] == Z[[0, 1, 2, 3, 4]].reshape(-1)).all() and (X[1] == Z[[8, 9, 10, 11, 12]].reshape(-1)).all()


def test_disjointness_check_detects_a_shared_id():
    ok = B.disjoint_check(["a", "b", "c"], ["d", "e"])
    bad = B.disjoint_check(["a", "b", "c"], ["c", "e"])
    assert ok["pass"] and ok["n_common"] == 0
    assert (not bad["pass"]) and bad["common_first5"] == ["c"]


def test_blockwise_gram_ridge_equals_direct_ridge():
    names, t, gap, Z = synth()
    FS = H.build_fit_sets(names, t, gap, np.full(len(gap), 10.0))
    rows = np.flatnonzero(FS["c_ok"])
    y = FS["c"][rows]
    fo = H.inner_folds(names[rows], 4, 100)
    G = B.build_stats(B.make_getX(Z, FS["idx5"], rows), len(rows), y, fo, 4, block=37)           # block size not a divisor of the row count
    tot = B.sum_stats(G)
    P = B.prep(tot, 0)
    lam = 3.7 * tot.n / 1000.0
    u, c0 = B.ridge_uc(P, lam)
    X = np.asarray(Z[FS["idx5"][rows]], np.float64).reshape(len(rows), -1)
    mu, sd = X.mean(0), np.where(X.std(0) < 1e-12, 1.0, X.std(0))
    A = (X - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y - y.mean()))
    direct = A @ w + y.mean()
    assert np.allclose(X @ u + c0, direct, atol=1e-8)
    # the dual form of ADD21 gives the same predictions
    dual = H.ridge_paths(X, y, X, [lam])[0]
    assert np.allclose(dual, direct, atol=1e-7)
    # squared error from the statistics equals the direct one, group by group
    for j in range(4):
        m = fo == j
        assert abs(B.sse_stats(G[j], u, c0, 0) - float(((y[m] - (X[m] @ u + c0)) ** 2).sum())) < 1e-6 * max(1.0, float(((y[m]) ** 2).sum()))


def test_inner_folds_by_clip_are_disjoint_and_cover():
    names, t, gap, Z = synth(n_clip=23)
    fo = H.inner_folds(names, 4, 100)
    assert set(fo) == {0, 1, 2, 3}
    for c in set(names):
        assert len(set(fo[names == c])) == 1                        # a clip is never split
    sizes = [len(set(names[fo == j])) for j in range(4)]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == 23
    assert np.array_equal(fo, H.inner_folds(names, 4, 100))         # deterministic


def test_ht_from_statistics_reads_a_planted_state_and_not_shuffled_labels():
    names, t, gap, Z = synth(n_clip=60, d=16, amp=10.0)
    FS = H.build_fit_sets(names, t, gap, np.full(len(gap), 10.0))
    held = sorted(set(names))[:6]
    r = B.tenth_fit(Z, FS, "c", names, held)
    assert r["r2_heldout_tenth"] > 0.9 and r["r2_in_sample"] >= r["r2_heldout_tenth"] - 1e-9
    pool = np.ones(len(names), bool)
    fo = H.inner_folds(names, 4, 100)
    f = B.fit_target(Z, FS, "c", names, fo, pool, perm_seed=7)
    cl = np.flatnonzero(FS["c_ok"])
    X = np.asarray(Z[FS["idx5"][cl]], np.float64).reshape(len(cl), -1)
    r2_true = 1 - ((FS["c"][cl] - B.predict_lin(f["true"], X)) ** 2).sum() / ((FS["c"][cl] - FS["c"][cl].mean()) ** 2).sum()
    assert r2_true > 0.9


def test_lc_clips_and_clip_draw_are_deterministic():
    names = np.array(["c%d" % i for i in range(10) for _ in range(5)])
    o1, o2 = B.clip_draw(names, 0), B.clip_draw(names, 0)
    assert (o1 == o2).all() and len(o1) == 10
    sel, tot = B.lc_clips(o1, {c: 5 for c in o1}, 12)
    assert tot == 15 and len(sel) == 3


def test_planted_state_control_machinery_runs_on_a_small_layout():
    M = {"keep": np.ones(24 * 30, bool), "beid": np.array(["clip%03d" % e for e in range(30) for _ in range(24)]), "bg": np.tile(np.arange(24), 30)}
    # the control builds 640-wide states: keep it small by using few clips
    tenth = sorted(set(M["beid"]))[:3]
    r = B.control_j5(M, tenth, amp=10.0, s_noise=0.8641)
    assert set(r) >= {"pass", "heldout_r2_closing", "bar", "n_fit", "n_heldout"} and r["heldout_r2_closing"] > 0.1    # 27 clips of 24 frames are far too few rows for 3,200 inputs; the bar 0.9 is for the full layout
