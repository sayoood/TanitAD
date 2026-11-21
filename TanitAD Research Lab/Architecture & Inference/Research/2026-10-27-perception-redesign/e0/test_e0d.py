"""Known-answer + deliberate-regression tests for e0d_score.py.  Pure numpy; CUDA is never touched.
Run: E0D_CI_PATH=<taniteval/taniteval/ci.py> CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_e0d.py   (in a repo checkout the ci.py is found by walking up)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0d_score as E  # noqa: E402


def naive_dilate(m, r):
    h, w = m.shape
    out = np.zeros_like(m)
    for i in range(h):
        for j in range(w):
            if m[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].any():
                out[i, j] = True
    return out


@pytest.mark.parametrize("r", [0, 1, 2, 3, 5])
def test_dilate_equals_naive_square_element(r):
    rng = np.random.default_rng(r)
    m = rng.random((23, 17)) < 0.08
    assert np.array_equal(E.dilate(m, r), naive_dilate(m, r))


def test_analytic_worlds_pass():
    res = E.selftest()
    assert all(v["pass"] for v in res.values()), res


def test_dilated_line_is_exactly_half_a_metre_wide():
    g = np.zeros((20, 40), bool)
    g[:, 20] = True                                                   # a 1-cell (10 cm) line
    assert (E.dilate(g, E.LANE_PRIMARY_R).sum(1) == 5).all()          # 5 cells = 0.5 m


def _rand_world(seed=0, h=40, w=30):
    rng = np.random.default_rng(seed)
    gt = rng.choice(np.array([0, 1, 2, 3, 5, 6, 7, 255], np.uint8), size=(h, w), p=[.1, .35, .08, .02, .05, .02, .3, .08])
    return gt, rng


EDGES = (0, 8, 16, 24, 32, 40)


def test_histograms_equal_brute_force():
    gt, rng = _rand_world(1)
    pdrv = rng.integers(0, 256, gt.shape).astype(np.uint8)
    code = rng.integers(0, 8, gt.shape).astype(np.uint8)
    st = E.gt_static(gt, EDGES)
    ps = E.plane_stats(pdrv, pdrv, code, st, thr_drv=(128,), thr_edge=(128,))
    ref = np.zeros((5, 4, 256), np.int64)
    conf = np.zeros((5, 9, 8), np.int64)
    for i in range(gt.shape[0]):
        b = i // 8
        for j in range(gt.shape[1]):
            g = int(gt[i, j])
            cls = 0 if g in E.DRV_CODES else 1 if g == 0 else 3 if g == 255 else 2
            ref[b, cls, pdrv[i, j]] += 1
            conf[b, 8 if g == 255 else g, code[i, j]] += 1
    assert np.array_equal(ps["drv_hist"], ref) and np.array_equal(ps["conf"], conf)


def _tol_recall_naive(X, Egt, r):
    ys, xs = np.nonzero(Egt)
    hit = 0
    for y, x in zip(ys, xs):
        hit += int(X[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any())
    return hit, len(ys)


def test_edge_stats_equal_naive_tolerance_count():
    gt, rng = _rand_world(2)
    X = rng.random(gt.shape) < 0.1
    st = E.gt_static(gt, EDGES)
    s = E._edge_stats(X, st).sum(0)
    hit, n = _tol_recall_naive(X, gt == E.EDGE, E.TOL_EDGE)
    assert (s[0], s[1]) == (n, hit)


def test_gt_as_prediction_reads_one_and_constants_read_their_known_values():
    gt, _ = _rand_world(3)
    st = E.gt_static(gt, EDGES)
    arms = {k: E.plane_stats(a, b, c, st, thr_drv=(128,), thr_edge=(128,)) for k, (a, b, c) in E.control_planes(gt).items()}
    valid = gt != 255
    G = arms["GTP"]
    h = G["drv_hist"].sum(0).astype(float)
    assert E.iou_from(h[0, 128:].sum(), (h[1] + h[2])[128:].sum(), h[0, :128].sum()) == 1.0
    assert E.f_ioudil(G["lane"][E.LANE_RS.index(2)].sum(0)[:3].astype(float)) == 1.0
    assert E.f_ioudil(G["lane"][E.LANE_RS.index(0)].sum(0)[:3].astype(float)) == 1.0
    assert E.f_tol(G["lane"][E.LANE_RS.index(5)].sum(0).astype(float)) == 1.0
    pe = G["pedge"][0].sum(0).astype(float)
    assert pe[1] == pe[0] and pe[3] == pe[2]                          # edge recall 1, precision 1
    C0 = arms["C0"]
    assert E.f_ioudil(C0["lane"][E.LANE_RS.index(2)].sum(0)[:3].astype(float)) == 0.0
    assert C0["bnd"][0].sum(0)[1] == 0
    CP = arms["CP"]
    hp = CP["drv_hist"].sum(0).astype(float)
    prev = np.isin(gt, E.DRV_CODES)[valid].mean()
    assert abs(E.iou_from(hp[0, 128:].sum(), (hp[1] + hp[2])[128:].sum(), hp[0, :128].sum()) - prev) < 1e-12     # all-positive map reads the class prevalence
    assert CP["bnd"][0].sum(0)[1] == 0                                # a constant plane has no boundary: recall exactly 0


def test_threshold_fit_known_optimum():
    pos, neg = np.zeros(256), np.zeros(256)
    pos[60], neg[40] = 100, 100
    t, iou = E.fit_drv_threshold(pos, neg)
    assert 41 <= t <= 60 and iou == 1.0


def test_disjointness_assertion_raises_on_overlap():
    eid = np.array(["e0", "e0", "e1", "e2"])
    E.assert_disjoint([0, 1], [2, 3], eid)
    with pytest.raises(AssertionError):
        E.assert_disjoint([0, 1], [1, 2], eid)                        # shared window
    with pytest.raises(AssertionError):
        E.assert_disjoint([0, 2], [1, 3], eid)                        # shared episode e0 across the sets


def _fake_bank(fit_pos, fit_neg, score_pos, score_neg):
    n = 2
    h = np.zeros((n, 5, 4, 256), np.int32)
    h[0, :, 0, fit_pos], h[0, :, 2, fit_neg] = 50, 50                  # window 0: episode 0 (FIT)
    h[1, :, 0, score_pos], h[1, :, 2, score_neg] = 50, 50              # window 1: episode 1 (SCORE)
    z = {"e": np.array([0, 1]), "w": np.array([0, 0]), "OWN__drv_hist": h, "OWN__bnd": np.zeros((n, len(E.THR_DRV), 5, 5), np.int32),
         "OWN__pedge": np.zeros((n, len(E.THR_EDGE), 5, 5), np.int32),
         "meta_json": np.array(json.dumps({"dev_fit_episodes": [0], "dev_score_episodes": [1]}))}
    return E.Bank(z=z, quiet=True)


def test_thresholds_use_only_the_fit_windows():
    bank = _fake_bank(60, 40, 220, 200)
    t_fit = E.thresholds_from(bank, bank.fit)
    t_score = E.thresholds_from(bank, bank.score)
    assert all(41 <= t <= 60 for t in t_fit["perband"]["drv"].values())
    assert all(201 <= t <= 220 for t in t_score["perband"]["drv"].values())
    assert t_fit["fit_windows"] == 1 and bank.disjoint["shared_windows"] == 0
    # deliberate regression: a fit on the SCORED windows (the forbidden path) would give the SCORE optimum, so this test goes RED if thresholds_from ignored `idx`
    assert t_fit["perband"]["drv"]["0-20"] != t_score["perband"]["drv"]["0-20"]


def test_split_rule_and_e_index_mapping():
    ids = ["c", "b", "a"]
    sha = {k: hashlib.sha256(k.encode()).hexdigest() for k in ids}
    order = sorted(ids, key=lambda k: sha[k])                         # E0i's known answer: sha256 ascending is c, b, a
    assert order == ["c", "b", "a"]
    clips = sorted(ids)                                               # the windows file is sorted by id: a, b, c -> e0, e1, e2
    split = {"DEV": {"sha12": [sha["c"][:12], sha["a"][:12]]}, "HOLD": {"sha12": [sha["b"][:12]]}}
    m = E.episode_maps(clips, split)
    assert m["dev"] == [2, 0] and m["hold"] == [1]
    assert m["fit"] == [2] and m["score"] == [0]                      # even DEV position = FIT, odd = SCORE
    assert E.episodes_for("dev", m) == [2, 0]
    with pytest.raises(SystemExit):
        E.episodes_for("hold", m)                                     # HOLD stays sealed
    with pytest.raises(AssertionError):
        E.episode_maps(["b", "a"], split)                             # an unsorted windows file would void the e-index rule


def test_pooled_bootstrap_identity_reads_zero():
    ci = E._load_ci()
    comp = np.array([[3.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.0, 1.0]])
    r = E.paired(ci, comp, comp, np.array(["a", "b", "c", "d"]), E.f_ratio, n_boot=200)
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_boundary_clears_the_outer_ring_and_is_the_inner_8_neighbourhood():
    d = np.ones((6, 6), bool)
    d[1, 1] = False                                                   # one hole; its 8 neighbours would be boundary, but row 0 / col 0 never are
    b = E.boundary(d)
    assert set(map(tuple, np.argwhere(b))) == {(1, 2), (2, 1), (2, 2)}
    assert E.boundary(np.ones((6, 6), bool)).sum() == 0 and E.boundary(np.zeros((6, 6), bool)).sum() == 0     # a constant plane has no boundary
