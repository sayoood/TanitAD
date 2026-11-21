"""Unit tests for trigger probe 2 (PREREG_FROZEN.md section 8, C4).  Run (from this directory):
    CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider test_trigger_probe2.py
Needs numpy, sklearn, torch.  The banked-control tests are skipped until the corresponding raw/*.json exists."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import common2 as C  # noqa: E402

L = C.L
RAW = f"{HERE}/raw"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_tp_lib_is_byte_identical_to_the_previous_package():
    assert _sha(f"{HERE}/tp_lib.py") == _sha(f"{C.OLD}/tp_lib.py")


def test_donor_draws_never_from_the_own_clip_and_are_deterministic():
    rng = np.random.default_rng(0)
    clip_all = rng.integers(0, 12, 5000)                                  # few clips: collisions are common, so the redraw loop is exercised
    own = rng.integers(0, 12, 800)
    d1 = C.draw_donors(own, clip_all, 0)
    d2 = C.draw_donors(own, clip_all, 0)
    assert np.array_equal(d1, d2)
    assert (clip_all[d1] != own).all()
    assert not np.array_equal(d1, C.draw_donors(own, clip_all, 1))        # the seed matters
    assert 0 <= d1.min() and d1.max() < len(clip_all)


def test_nav_hist_follows_the_item_rule_and_clips_at_the_clip_start():
    J = {"n": np.array([6, 8]), "off": np.array([0, 6]), "nav": np.arange(14 * 5, dtype=np.float32).reshape(14, 5)}
    h = C.nav_hist(J, 1, 3, 4)                                           # clip 1, td = 3, hmax = 4 -> steps 0..3 -> rows 6..9
    assert np.array_equal(h, J["nav"][6:10])
    h0 = C.nav_hist(J, 1, 1, 4)                                          # steps -2..1 clipped to 0,0,0,1
    assert np.array_equal(h0, J["nav"][[6, 6, 6, 7]])
    hl = C.nav_hist(J, 0, 5, 4)
    assert np.array_equal(hl, J["nav"][2:6])


def test_swap_changes_nav_only_then_nav_and_limit_and_nothing_else():
    torch = pytest.importorskip("torch")
    import extract_znn as XZ
    J = {"clip": np.array(["a", "b"]), "n": np.array([6, 6]), "off": np.array([0, 6]), "nav": np.random.default_rng(0).uniform(size=(12, 5)).astype(np.float32),
         "limit": np.array([[1, 0, 0, 0], [0, 0, 1, 0]], np.float32), "limit_ok": np.array([True, False])}
    jix = {"a": 0, "b": 1}
    A = {"clip_i": np.array([0, 1]), "t": np.array([3, 4])}
    clips = ["a", "b"]

    class Cfg:
        hmax = 4
    feed = {"nav": torch.zeros(1, 4, 5), "limit": torch.tensor([[0.0, 1, 0, 0]]), "limit_ok": torch.tensor([True]), "feats": torch.ones(1, 4, 2), "p10f": torch.ones(1, 4, 8)}
    f1, f2, sn, sl = XZ.swap(feed, J, jix, clips, A, Cfg, np.array([1]))
    assert not torch.equal(f1["nav"], feed["nav"]) and torch.allclose(f1["nav"][0], torch.as_tensor(J["nav"][6 + np.clip(np.arange(1, 5), 0, 5)]))
    assert torch.equal(f1["limit"], feed["limit"]) and torch.equal(f1["limit_ok"], feed["limit_ok"])        # nav-only swap leaves the limit
    assert torch.equal(f2["nav"], f1["nav"]) and torch.equal(f2["limit"], torch.tensor([[0.0, 0, 1, 0]])) and not bool(f2["limit_ok"][0])
    for k in ("feats", "p10f"):                                          # nothing else is touched
        assert f1[k] is feed[k] and f2[k] is feed[k]
    assert not bool(sn[0]) and not bool(sl[0])
    f1s, _, sns, _ = XZ.swap(feed, J, jix, clips, A, Cfg, np.array([0]))   # donor := a window whose nav differs from zeros anyway -> flag false; identity is exercised by control N1
    assert not bool(sns[0])


def test_lag_gather_picks_frames_t_minus_lag():
    torch = pytest.importorskip("torch")
    import extract_v4 as XV
    T = 12
    f4 = torch.arange(T, dtype=torch.float32)[:, None, None].expand(T, 40, 1024).clone()
    ts = np.array([3, 7, 11])
    for lag in (1, 2, 3):
        g = XV.gather_lag(f4, ts, lag)
        assert g.shape == (3, 40 * 1024) and g.dtype == np.float16
        assert np.array_equal(g[:, 0].astype(int), ts - lag)


def test_v4_and_vdiff_readers_equal_dense_computations(tmp_path, monkeypatch):
    import probe2 as P
    n, D = 20, 12
    rng = np.random.default_rng(0)
    Vt, V1, V2, V3 = (rng.normal(size=(n, D)).astype(np.float16) for _ in range(4))
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(), new.mkdir()
    np.save(old / "V_pstop.npy", Vt)
    for l, v in zip((1, 2, 3), (V1, V2, V3)):
        np.save(new / f"Vm{l}_pstop.npy", v)
    monkeypatch.setattr(C, "OLDSCR", str(old).replace("\\", "/"))
    monkeypatch.setattr(C, "SCR", str(new).replace("\\", "/"))
    rows = np.array([0, 3, 4, 19])
    rd4, sh4 = P.reader("V4", "pstop")
    assert sh4 == (n, 4 * D)
    assert np.array_equal(rd4(rows), np.concatenate([V3[rows], V2[rows], V1[rows], Vt[rows]], 1).astype(np.float32))      # chronological: t-3, t-2, t-1, t
    rdd, shd = P.reader("Vdiff", "pstop")
    assert np.array_equal(rdd(rows), Vt[rows].astype(np.float32) - V3[rows].astype(np.float32))
    # and PCA of the streamed concat matches PCA of the dense concat (the reader plugs into StreamPCA unchanged)
    p = L.StreamPCA(k=3, oversample=6, n_iter=2, seed=0, chunk=7).fit(rd4, np.arange(n), 4 * D)
    X = rd4(np.arange(n))
    _, _, Vh = np.linalg.svd((X - X.mean(0)).astype(np.float64), full_matrices=False)
    assert np.linalg.svd(p.comp.astype(np.float64) @ Vh[:3].T, compute_uv=False).min() > 0.99


def test_a_planted_motion_signal_is_seen_through_the_diff_reader_and_a_static_one_is_not():
    """The V4 arm can only help if the DIFFERENCE carries signal: plant a label-dependent drift in V_t - V_{t-3} and none in V_t; the diff probe sees it, the frame-t probe does not."""
    rng = np.random.default_rng(0)
    n, D, K = 1500, 40, 60
    clip = rng.integers(0, K, n)
    y = rng.integers(0, 2, n)
    base = rng.normal(size=(n, D))
    drift = np.zeros((n, D))
    drift[:, 0] = 1.5 * y
    Vt, V3 = base + drift, base                                          # V_t - V_{t-3} = drift; V_t alone has the drift buried under unit-variance noise in column 0 only weakly
    shas = np.array([f"{c * 7919 + 13:012x}" for c in clip])
    outer, inner = L.outer_fold(shas), L.inner_fold(shas)
    o_diff, _ = L.cv_oof(lambda f: Vt - V3, y, outer, inner, "linear")
    o_static, _ = L.cv_oof(lambda f: base, y, outer, inner, "linear")
    a_d = L.auc_ci_fold(o_diff, y, clip, outer, n_boot=200)
    a_s = L.auc_ci_fold(o_static, y, clip, outer, n_boot=200)
    assert a_d["auc"] > 0.85 and a_d["lo"] > 0.8
    assert a_s["lo"] <= 0.5 <= a_s["hi"]


def test_foldmean_metric_unbiased_on_prevalence_shifted_null():
    rng = np.random.default_rng(5)
    K, per = 120, 30
    clip = np.repeat(np.arange(K), per)
    rate = rng.uniform(0.05, 0.95, K)
    y = (rng.uniform(size=K * per) < rate[clip]).astype(int)
    X = rng.normal(size=(K * per, 5))
    shas = np.array([f"{c * 104729 + 7:012x}" for c in clip])
    outer, inner = L.outer_fold(shas), L.inner_fold(shas)
    oof, _ = L.cv_oof(lambda f: X, y, outer, inner, "linear")
    assert abs(L.FoldWAuc(oof, y, outer).auc() - 0.5) < 0.05


def test_decision_rules_at_the_frozen_boundaries():
    import decide2 as Dd
    assert Dd.znn_rule(0.05, 0.001) == "VISION-THROUGH-TRUNK"
    assert Dd.znn_rule(0.05, -0.001) == "INCONCLUSIVE"                    # >= 0.05 but the CI reaches 0
    assert Dd.znn_rule(0.0499, 0.01) == "INCONCLUSIVE"
    assert Dd.znn_rule(0.02, 0.01) == "INCONCLUSIVE"
    assert Dd.znn_rule(0.0199, -0.05) == "NAV-ONLY"
    assert Dd.znn_rule(-0.03, -0.08) == "NAV-ONLY"
    assert Dd.v4_rule(0.03, 0.001) == "MOTION"
    assert Dd.v4_rule(0.03, -0.001) == "INCONCLUSIVE"
    assert Dd.v4_rule(0.0099, 0.0) == "NO-MOTION"
    assert Dd.v4_rule(0.01, 0.0) == "INCONCLUSIVE"


def _need(f):
    p = f"{RAW}/{f}"
    if not os.path.exists(p):
        pytest.skip(f"{f} not banked yet")
    return json.load(open(p))


def test_banked_controls_nav_and_identity():
    n = _need("control_znn.json")
    assert n["N1a_true_forward_vs_banked_Z_max_abs"] <= 1e-5 and n["N1b_selfdonor_swap_vs_banked_Z_max_abs"] <= 1e-5
    assert n["N1b_selfdonor_traj60_max_abs_change"] == 0.0
    assert n["N2_nn_windows_nav_differs"]["frac_relL2_gt_0"] >= 0.99 and n["N2_nn_windows_nav_differs"]["median_max_abs_dtraj60_m"] > 0
    assert n["N2_windows_nav_equal_unchanged_max_relL2"] == 0.0           # a window whose donor nav equals its own is bit-for-bit unchanged


def test_banked_v4_frame_checks():
    v = _need("control_v4.json")
    for p in v["pops"].values():
        assert p["max_abs_frame_t_old_vs_cache"] == 0.0 and p["stored_lag_frames_equal_cache"] and p["all_zero_lag3_rows"] == 0
        assert p["share_rows_where_lag_frame_differs_from_t"]["3"] >= 0.99


#: The ONE frozen control that fails (RESULTS.md section 3): C2 (donor-episode labels, seed 2) on Z_nn+E / P-STOP reads 0.454 [0.418, 0.491], the CI misses 0.5.  It is PINNED here as a failure (so it cannot silently change)
#: and its post-freeze reading against the 11-draw null band is asserted separately; every other frozen control must pass.
KNOWN_FROZEN_FAILURE = ("pstop", "Znn", "C2_donor_labels")


@pytest.mark.parametrize("pop", ["pstop", "pmove"])
def test_banked_label_controls(pop):
    p = f"{RAW}/VERDICT.json"
    if not os.path.exists(p):
        pytest.skip("VERDICT.json not banked yet")
    v = json.load(open(p))["pops"][pop]["controls"]
    for arm, cs in v.items():
        for name, c in cs.items():
            if (pop, arm, name) == KNOWN_FROZEN_FAILURE:
                assert not c["pass"] and c["hi"] < 0.5, c                  # pinned: the frozen reading is a failure
            else:
                assert c["pass"], (pop, arm, name, c)


def test_the_known_frozen_failure_is_inside_its_null_band():
    p = f"{RAW}/VERDICT.json"
    if not os.path.exists(p):
        pytest.skip("VERDICT.json not banked yet")
    b = json.load(open(p))["pops"]["pstop"]["postfreeze_c2_band"]["Znn"]
    assert b["seed2_is_minimum"] and abs(b["mean"] - 0.5) < 0.03 and b["true_minus_null_max"] > 0.1 and b["band_pass"]
