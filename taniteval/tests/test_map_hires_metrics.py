"""refcv7 NEW-2 -- the 10 cm map metrics (``taniteval.map_hires_metrics``).

Every expected value below is a LITERAL derived by hand from the fixture, never an
expression over the code under test:

* a perfect prediction scores IoU = F1 = 1.0 (and an absent class is NaN, not 0);
* a 1-cell (0.1 m) lateral shift of a 2-cell-wide line scores tolerance-F1 = 1.0 at
  0.2 m while its IoU is 1/3;
* a 2-cell shift sits ON the 0.2 m boundary, which is EXCLUDED: F1 = 0.5;
* a 3-cell shift scores F1 = 0.0;
* an all-drivable constant on a GT that is drivable on 80 of 320 columns scores
  drivable IoU = 0.25 in every band and sidewalk IoU = 0.0;
* the 0.5 m hook repeats cell (i, j) onto fine rows 5i..5i+4, cols 5j..5j+4 and
  ignores the not-seen logit.

Deliberate-regression arms (each must go RED): the scorer counting NOT-SEEN cells,
and an INCLUSIVE tolerance. Plus the bootstrap: identical arms are not separated, a
clearly better arm is, and the per-clip pooling shortcut equals brute force.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from taniteval import map_hires_metrics as M

H, W = 600, 320
DRV, LANE, XW, EDGE, SIDE = 1, 2, 3, 5, 7


def _gt_scene() -> np.ndarray:
    g = np.full((H, W), DRV, np.uint8)
    g[:, 140:142] = LANE
    g[:, 200:202] = EDGE
    g[:, 202:] = SIDE
    g[50:70, 20:120] = XW
    g[:10, :] = 255                                    # not seen
    return g


def _line(cols) -> np.ndarray:
    g = np.full((H, W), DRV, np.uint8)
    g[:, cols] = LANE
    return g


def _p(stats) -> dict:
    return M.pooled({k: v[None] for k, v in stats.items()})


# --------------------------------------------------------------------------- #
# analytic targets                                                             #
# --------------------------------------------------------------------------- #
def _assert_perfect_is_one() -> None:
    gt = _gt_scene()
    pred = gt.copy()
    pred[gt == 255] = DRV                     # anything at unseen cells
    r = _p(M.window_stats(pred, gt))
    for k in (DRV, LANE, EDGE, SIDE):
        assert r["iou"][k].tolist() == [1.0, 1.0, 1.0], (k, r["iou"][k])
    assert r["iou"][XW].tolist()[0] == 1.0
    for k in (LANE, EDGE):
        assert r["F1"][k].tolist() == [1.0, 1.0, 1.0]
    assert r["F1"][XW][0] == 1.0
    # classes absent from GT and prediction are UNDEFINED, not 0 or 1
    assert all(math.isnan(v) for v in r["iou"][0].tolist())
    assert math.isnan(r["iou"][XW][1]) and math.isnan(r["F1"][XW][2])


def test_perfect_prediction_scores_one():
    _assert_perfect_is_one()


def test_DELIBERATE_REGRESSION_counting_not_seen_cells_goes_RED(monkeypatch):
    monkeypatch.setattr(M, "_scored_mask",
                        lambda gt, valid: np.ones(gt.shape, bool))
    with pytest.raises(AssertionError):
        _assert_perfect_is_one()


def test_one_cell_shift_of_a_2_cell_line_is_inside_the_tolerance():
    r = _p(M.window_stats(_line([151, 152]), _line([150, 151])))
    assert r["F1"][LANE].tolist() == [1.0, 1.0, 1.0]
    assert r["P"][LANE].tolist() == [1.0, 1.0, 1.0]
    assert r["R"][LANE].tolist() == [1.0, 1.0, 1.0]
    for v in r["iou"][LANE].tolist():
        assert v == pytest.approx(1.0 / 3.0, abs=1e-15) and v < 1.0


def test_two_cell_shift_sits_on_the_boundary_and_is_excluded():
    r = _p(M.window_stats(_line([152, 153]), _line([150, 151])))
    assert r["F1"][LANE].tolist() == [0.5, 0.5, 0.5]
    assert r["iou"][LANE].tolist() == [0.0, 0.0, 0.0]


def _assert_three_cell_shift_is_zero() -> None:
    r = _p(M.window_stats(_line([153, 154]), _line([150, 151])))
    assert r["F1"][LANE].tolist() == [0.0, 0.0, 0.0], r["F1"][LANE]
    assert r["iou"][LANE].tolist() == [0.0, 0.0, 0.0]


def test_three_cell_shift_scores_zero():
    _assert_three_cell_shift_is_zero()


def test_DELIBERATE_REGRESSION_inclusive_tolerance_goes_RED(monkeypatch):
    """`<=` instead of `<`: the 3-cell shift's nearest pair is exactly 0.2 m apart
    and would be matched (F1 0.5) -- the literal test above must fail."""
    real = M._tol_offsets
    monkeypatch.setattr(M, "_tol_offsets",
                        lambda tol_m, cell_m=M.FINE_CELL_M: real(tol_m + 1e-9, cell_m))
    with pytest.raises(AssertionError):
        _assert_three_cell_shift_is_zero()


# --------------------------------------------------------------------------- #
# the tolerance footprint: exact, strict, and scipy-free                       #
# --------------------------------------------------------------------------- #
def test_the_02m_footprint_is_the_3x3_square_and_025m_adds_12_cells():
    """0.1 m cells, STRICT 0.2 m: (+-2, 0) and (0, +-2) sit at exactly 0.2 m and are OUT.
    At 0.25 m: + (+-2, 0), (0, +-2) (d 0.2) and the 8 cells at d 0.2236 -> 21."""
    assert sorted(M._tol_offsets(0.2)) == [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                                           (0, 1), (1, -1), (1, 0), (1, 1)]
    assert len(M._tol_offsets(0.25)) == 21
    assert M._tol_offsets(0.0) == ()
    assert not M._near(np.zeros((7, 5), bool), 0.2).any()


def test_the_footprint_equals_the_distance_transform_where_scipy_exists():
    """The replaced definition, `distance_transform_edt(~m, sampling=0.1) < tol`, on random
    masks of three densities and nine tolerances incl. the exact boundaries. Skipped where
    scipy is absent (the Thor training venv) -- the literal tests above run everywhere."""
    nd = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    for p in (0.002, 0.02, 0.2):
        m = rng.random((120, 90)) < p
        m[0, 0] = True
        for tol in (0.05, 0.1, 0.1414, 0.15, 0.2, 0.2236, 0.25, 0.3, 0.5):
            want = nd.distance_transform_edt(~m, sampling=M.FINE_CELL_M) < tol
            assert np.array_equal(M._near(m, tol), want), (p, tol)


def test_the_module_imports_and_scores_without_scipy(monkeypatch):
    """⛔ The Thor gate's venv has NO scipy (MEASURED 2026-09-27). A module-level scipy import
    made this whole file an import error there. Import the module afresh with scipy BLOCKED
    and score the tolerance fixture with it."""
    import importlib
    import sys
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
    monkeypatch.delitem(sys.modules, "taniteval.map_hires_metrics", raising=False)
    fresh = importlib.import_module("taniteval.map_hires_metrics")
    assert fresh is not M
    r = fresh.pooled({k: v[None] for k, v in
                      fresh.window_stats(_line([151, 152]), _line([150, 151])).items()})
    assert r["F1"][LANE].tolist() == [1.0, 1.0, 1.0]


def test_all_drivable_constant_scores_its_known_value():
    gt = np.full((H, W), SIDE, np.uint8)
    gt[:, :80] = DRV
    gt[:10, :] = 255
    r = _p(M.window_stats(np.full((H, W), DRV, np.uint8), gt))
    assert r["iou"][DRV].tolist() == [0.25, 0.25, 0.25]
    assert r["iou"][SIDE].tolist() == [0.0, 0.0, 0.0]
    assert all(math.isnan(v) for v in r["iou"][LANE].tolist())
    # the thin classes absent everywhere: F1 undefined, never a silent 0 or 1
    assert all(math.isnan(v) for v in r["F1"][LANE].tolist())


def test_bands_are_rows_0_199_200_399_400_599():
    gt = np.full((H, W), DRV, np.uint8)
    pred = gt.copy()
    pred[199, :] = SIDE                     # one wrong row, the last of band 0
    pred[400, :] = SIDE                     # and the first of band 2
    r = _p(M.window_stats(pred, gt))
    assert r["iou"][DRV][0] == pytest.approx(199 / 200)
    assert r["iou"][DRV][1] == 1.0
    assert r["iou"][DRV][2] == pytest.approx(199 / 200)


def test_prediction_must_be_decided():
    with pytest.raises(ValueError):
        M.window_stats(np.full((H, W), 255, np.uint8), _line([1]))
    with pytest.raises(ValueError):
        M.window_stats(np.zeros((120, 64), np.uint8), _line([1]))


# --------------------------------------------------------------------------- #
# the hooks                                                                    #
# --------------------------------------------------------------------------- #
def test_coarse_hook_repeats_each_cell_on_its_5x5_block_and_drops_not_seen():
    p = np.zeros((9, 120, 64), np.float32)
    p[DRV] = 1.0
    p[LANE, 3, 7] = 5.0                         # cell (3, 7) -> lane
    p[8, 10, 10] = 99.0                         # not-seen wins overall here ...
    p[SIDE, 10, 10] = 2.0                       # ... but the class argmax is sidewalk
    f = M.coarse_to_fine_codes(p)
    assert f.shape == (H, W) and f.dtype == np.uint8
    assert (f[15:20, 35:40] == LANE).all()
    assert int(f[14, 35]) == DRV and int(f[15, 34]) == DRV and int(f[20, 40]) == DRV
    assert (f[50:55, 50:55] == SIDE).all()
    assert int(M.coarse_to_fine_codes(p[:8])[16, 36]) == LANE      # 8-channel input
    with pytest.raises(ValueError):
        M.coarse_to_fine_codes(np.zeros((9, 64, 120)))


def test_logits_hook():
    lg = np.zeros((8, H, W), np.float32)
    lg[EDGE, 5, 6] = 1.0
    c = M.logits_to_codes(lg, rule="raw")
    assert int(c[5, 6]) == EDGE and int(c[0, 0]) == 0
    assert np.array_equal(M.logits_to_codes(lg, class_weight=np.ones(8)), c)   # C3
    with pytest.raises(ValueError):
        M.logits_to_codes(np.zeros((9, H, W)), rule="raw")
    with pytest.raises(ValueError, match="class weights"):
        M.logits_to_codes(lg)                  # prior-corrected (the default) needs w


# --------------------------------------------------------------------------- #
# the decision rule (map-signal audit D1): an analytic misfire of the raw rule  #
# --------------------------------------------------------------------------- #
#: median-frequency-like weights: lane up-weighted 20x against drivable
W_MF = np.array([1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def _calibrated_logits(p_lane: float) -> np.ndarray:
    """What a weighted CE converges to: z_c = log(w_c P(c|x)). Truth: a drivable
    scene whose every cell has P(lane) = ``p_lane``, P(drivable) = 1 - p_lane."""
    P = np.full((8, H, W), 1e-9)
    P[DRV], P[LANE] = 1.0 - p_lane, p_lane
    return np.log(W_MF[:, None, None] * P)


def _assert_prior_corrected_decides_drivable() -> None:
    z = _calibrated_logits(0.10)              # w_lane P_lane = 2.0 > 0.9 = w_drv P_drv
    gt = np.full((H, W), DRV, np.uint8)
    raw = _p(M.window_stats(M.logits_to_codes(z, rule="raw"), gt))
    cor = _p(M.window_stats(M.logits_to_codes(z, class_weight=W_MF), gt))
    # the RAW rule calls lane on EVERY cell at a 10 % posterior (the misfire) ...
    assert raw["iou"][DRV].tolist() == [0.0, 0.0, 0.0]
    assert raw["iou"][LANE].tolist() == [0.0, 0.0, 0.0]
    # ... the prior-corrected rule recovers the calibrated decision exactly
    assert cor["iou"][DRV].tolist() == [1.0, 1.0, 1.0]
    assert all(math.isnan(v) for v in cor["iou"][LANE].tolist())
    # and at a 60 % lane posterior BOTH rules call lane (it is a correction, not a veto)
    z6 = _calibrated_logits(0.60)
    assert int(M.logits_to_codes(z6, class_weight=W_MF)[0, 0]) == LANE


def test_the_prior_corrected_rule_fixes_the_raw_misfire():
    _assert_prior_corrected_decides_drivable()


def test_DELIBERATE_REGRESSION_a_correction_that_ignores_the_weights_goes_RED(monkeypatch):
    real = M.logits_to_codes
    monkeypatch.setattr(M, "logits_to_codes",
                        lambda z, *, rule="prior_corrected", class_weight=None:
                        real(z, rule="raw"))
    with pytest.raises(AssertionError):
        _assert_prior_corrected_decides_drivable()


def test_the_numpy_and_torch_rules_are_one_formula():
    torch = pytest.importorskip("torch")
    from tanitad.models import map_head_hires as Hh
    rng = np.random.default_rng(3)
    # multiples of 0.5, so float32 (torch) and float64 (numpy) see the SAME order:
    # the only non-grid term is log 20, whose gap to any grid value is >= 0.004
    z = (rng.integers(-6, 7, size=(8, H, W)) * 0.5).astype(np.float32)
    for rule in ("raw", "prior_corrected"):
        t = Hh.decide(torch.from_numpy(z)[None], rule, torch.from_numpy(W_MF).float())
        n = M.logits_to_codes(z, rule=rule, class_weight=W_MF)
        assert np.array_equal(t[0].numpy().astype(np.uint8), n), rule


# --------------------------------------------------------------------------- #
# the positional prior                                                         #
# --------------------------------------------------------------------------- #
def test_positional_prior_majority_and_fit_set_refusal(tmp_path):
    a = np.full((2, H, W), DRV, np.uint8)
    a[:, :, 0] = SIDE
    b = np.full((1, H, W), DRV, np.uint8)
    b[:, 0, :] = 255                            # row 0 of B unseen
    pr = M.PositionalPrior()
    pr.add("aaaaaaaaaaaa", a)
    pr.add("bbbbbbbbbbbb", b)
    m = pr.predict_for("cccccccccccc")
    assert int(m[300, 0]) == SIDE              # 2 sidewalk vs 1 drivable
    assert int(m[300, 1]) == DRV
    assert int(m[0, 5]) == DRV                 # only A saw it: drivable
    with pytest.raises(ValueError, match="FIT set"):
        pr.predict_for("aaaaaaaaaaaa")
    never = M.PositionalPrior()
    z = np.full((1, H, W), 255, np.uint8)
    z[0, 1:, :] = SIDE
    never.add("dddddddddddd", z)
    assert int(never.predict_for("cccccccccccc")[0, 0]) == SIDE   # global majority
    pr.save(tmp_path / "p.npz")
    q = M.PositionalPrior.load(tmp_path / "p.npz")
    assert q.fit_clips == pr.fit_clips and (q.counts == pr.counts).all()
    assert q.fingerprint() == pr.fingerprint()


# --------------------------------------------------------------------------- #
# the table and the bootstrap                                                  #
# --------------------------------------------------------------------------- #
def _table(n_clips=6, per=3, good="perfect", other="shift3", seed=0):
    rng = np.random.default_rng(seed)
    t = M.WindowTable(arms=("refcv7", "refcv6_38k", "prior"))
    for c in range(n_clips):
        for _ in range(per):
            col = int(rng.integers(20, 280))
            gt = _line([col, col + 1])
            gt[:10] = 255
            preds = {"perfect": _line([col, col + 1]),
                     "shift1": _line([col + 1, col + 2]),
                     "shift3": _line([col + 3, col + 4]),
                     "const": np.full((H, W), DRV, np.uint8)}
            t.add(f"clip{c:08d}", gt, {"refcv7": preds[good],
                                       "refcv6_38k": preds[other],
                                       "prior": preds["const"]})
    return t


def test_clip_pooler_equals_brute_force_on_a_resampled_draw():
    t = _table(n_clips=5, per=4, other="shift1")
    st = t.arrays("refcv6_38k")
    pooler = M._ClipPooler(st, t.eid)
    rows = np.concatenate([np.flatnonzero(np.array(t.eid) == c)
                           for c in ("clip00000003", "clip00000000", "clip00000003")])
    brute = M.pooled({k: v[rows] for k, v in st.items()})
    fast = pooler(rows)
    for k in ("iou", "P", "R", "F1"):
        np.testing.assert_array_equal(np.nan_to_num(fast[k], nan=-1.0),
                                      np.nan_to_num(brute[k], nan=-1.0))
    with pytest.raises(ValueError, match="whole clips"):
        pooler(rows[:-1])


def test_identical_arms_are_not_separated_and_a_better_arm_is():
    same = _table(good="shift1", other="shift1")
    d = M.paired_delta(same, "refcv7", "refcv6_38k", "iou", LANE, 0, n_boot=200)
    assert d["delta"] == 0.0 and d["separated"] is False
    better = _table(good="perfect", other="shift3")
    d = M.paired_delta(better, "refcv7", "refcv6_38k", "iou", LANE, 0, n_boot=200)
    assert d["delta"] == 1.0 and d["separated"] is True
    assert d["n_episodes"] == 6 and d["n_windows"] == 18
    f = M.paired_delta(better, "refcv7", "refcv6_38k", "F1", LANE, 0, n_boot=200)
    assert f["delta"] == 1.0 and f["separated"] is True


def test_summary_carries_n_and_every_class_band():
    t = _table(n_clips=4, per=2, other="shift1")
    s = M.summarize(t, n_boot=100)
    assert (s["n_windows"], s["n_clips"]) == (8, 4)
    rec = s["arms"]["refcv6_38k"]
    assert rec["lane / road line|0-20m|iou"]["mean"] == pytest.approx(1 / 3, abs=1e-4)
    assert rec["lane / road line|0-20m|F1"]["mean"] == 1.0
    assert rec["crosswalk|0-20m|iou"]["mean"] is None          # absent: undefined
    assert len([k for k in rec if k.endswith("|iou")]) == 24   # 8 classes x 3 bands
    M.to_json(s)                                               # NaN-free JSON


def test_bars_are_reachable_in_both_directions():
    """A bar evaluator that can only say PASS (or only FAILED) certifies nothing."""
    t = M.WindowTable(arms=("refcv7", "refcv6_38k", "prior"))
    rng = np.random.default_rng(1)
    for c in range(8):
        for _ in range(2):
            col = int(rng.integers(20, 280))
            gt = np.full((H, W), DRV, np.uint8)
            gt[:, col:col + 2] = LANE
            gt[:, 300:302] = EDGE
            gt[100:120, 30:90] = XW
            base = gt.copy()
            base[:, col:col + 2] = DRV                          # misses the line
            base[:, 300:302] = DRV                              # and the edge
            base[100:120, 30:90] = DRV                          # and the crosswalk
            t.add(f"clip{c:08d}", gt, {"refcv7": gt.copy(), "refcv6_38k": base,
                                       "prior": np.full((H, W), DRV, np.uint8)})
    ok = M.evaluate_bars_m7(t, n_boot=200)
    assert [ok[b]["verdict"] for b in ("BAR-M7-1", "BAR-M7-2", "BAR-M7-3",
                                       "BAR-M7-4")] == ["PASS"] * 4
    bad = M.evaluate_bars_m7(t, arm="refcv6_38k", base="refcv7", n_boot=200)
    assert [bad[b]["verdict"] for b in ("BAR-M7-1", "BAR-M7-2", "BAR-M7-3")] \
        == ["FAILED"] * 3
    # the swapped arm predicts drivable on the line / edge / crosswalk cells, so its
    # drivable IoU is LOWER than the perfect arm's in every band -- separated worse.
    assert bad["BAR-M7-4"]["verdict"] == "FAILED"


def test_table_roundtrip(tmp_path):
    t = _table(n_clips=2, per=2)
    t.save(tmp_path / "t.npz")
    u = M.WindowTable.load(tmp_path / "t.npz")
    assert u.eid == t.eid and u.arms == t.arms
    for a in t.arms:
        for k, v in t.arrays(a).items():
            np.testing.assert_array_equal(u.arrays(a)[k], v)


# --------------------------------------------------------------------------- #
# SPEC_REFCV7 §11.2 / §12 (A6/A7): every band of the 100 m x +-30 m extent       #
# --------------------------------------------------------------------------- #
from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7  # noqa: E402

H7, W7 = 1000, 600


def test_band_names_follow_the_grid():
    assert M.BAND_NAMES == ("0-20m", "20-40m", "40-60m")
    assert M.band_names(H7) == ("0-20m", "20-40m", "40-60m", "60-80m", "80-100m")
    assert M.band_rows(700)[-1] == (600, 700) and M.band_names(700)[-1] == "60-70m"


def test_the_refcv6_hook_on_the_A7_extent_predicts_only_its_own_window():
    p = np.zeros((9, 120, 64), np.float32)
    p[DRV] = 1.0
    p[LANE, 3, 7] = 5.0
    f = M.coarse_to_fine_codes(p, EXTENT_REFCV7)
    assert f.shape == (H7, W7)
    np.testing.assert_array_equal(f[:600, 140:460], M.coarse_to_fine_codes(p))
    outside = np.ones((H7, W7), bool)
    outside[:600, 140:460] = False
    assert (f[outside] == M.NO_PREDICTION).all()
    # scored: refcv6 has no map past 60 m / beyond +-16 m -> its seen GT there is MISSED
    gt = np.full((H7, W7), DRV, np.uint8)
    p2 = np.zeros((9, 120, 64), np.float32)
    p2[DRV] = 1.0                                     # drivable on its whole grid
    r = _p(M.window_stats(M.coarse_to_fine_codes(p2, EXTENT_REFCV7), gt))
    assert r["iou"][DRV].tolist()[:3] == [pytest.approx(320 / 600)] * 3
    assert r["iou"][DRV].tolist()[3:] == [0.0, 0.0]
    perfect = _p(M.window_stats(np.full((H7, W7), DRV, np.uint8), gt))
    assert perfect["iou"][DRV].tolist() == [1.0] * 5
    with pytest.raises(ValueError):
        M.window_stats(np.full((H7, W7), 253, np.uint8), gt)   # neither class nor NO_PRED


def _assert_zero_weight_class_never_decided(fn) -> None:
    z = np.zeros((8, 4, 4))
    z[LANE] = 5.0
    z[DRV] = 1.0
    w = np.ones(8)
    w[LANE] = 0.0
    c = fn(z, rule="prior_corrected", class_weight=w)
    assert int((c == LANE).sum()) == 0 and int((c == DRV).sum()) == 16


def test_a_zero_weight_class_is_never_decided():
    _assert_zero_weight_class_never_decided(M.logits_to_codes)


def test_DELIBERATE_REGRESSION_the_naive_log_of_a_zero_weight_goes_RED():
    def naive(z, *, rule, class_weight):
        with np.errstate(divide="ignore"):
            return np.argmax(z - np.log(class_weight)[:, None, None], axis=0)
    with pytest.raises(AssertionError):
        _assert_zero_weight_class_never_decided(naive)


def test_the_prior_lives_on_its_extent(tmp_path):
    pr = M.PositionalPrior(extent=EXTENT_REFCV7)
    pr.add("aaaaaaaaaaaa", np.full((1, H7, W7), SIDE, np.uint8))
    assert pr.predict_for("bbbbbbbbbbbb").shape == (H7, W7)
    with pytest.raises(ValueError):
        pr.add("cccccccccccc", np.full((1, H, W), SIDE, np.uint8))  # a /2-grid frame
    pr.save(tmp_path / "p7.npz")
    q = M.PositionalPrior.load(tmp_path / "p7.npz")
    assert q.extent == EXTENT_REFCV7 and (q.counts == pr.counts).all()


def test_bars_are_evaluated_in_EVERY_band_with_their_n():
    """SPEC_REFCV7 §11.2 item 1: BAR-M7-1..3 in every band; a band where the class is
    absent from GT and every prediction is UNDEFINED with n 0 -- reported, never
    dropped -- and a bar fails if ANY scored band fails."""
    t = M.WindowTable(arms=("refcv7", "refcv6_38k", "prior"))
    for c in range(6):
        gt = np.full((H7, W7), DRV, np.uint8)
        gt[:, 300 + c:302 + c] = LANE                 # a lane line through all 5 bands
        gt[:, 500:502] = EDGE
        gt[850:870, 100:200] = XW                     # a crosswalk at 85 m only
        base = np.full((H7, W7), DRV, np.uint8)       # refcv6-like: never a thin class
        t.add(f"clip{c:08d}", gt, {"refcv7": gt.copy(), "refcv6_38k": base,
                                   "prior": np.full((H7, W7), DRV, np.uint8)})
    assert t.band_names == ("0-20m", "20-40m", "40-60m", "60-80m", "80-100m")
    r = M.evaluate_bars_m7(t, n_boot=100)
    lane = r["BAR-M7-1"]["per_band"]
    assert list(lane) == list(t.band_names)
    assert all(v["verdict"] == "PASS" and v["n_gt_cells"] > 0 for v in lane.values())
    xw = r["BAR-M7-2"]["per_band"]
    assert [xw[b]["verdict"] for b in t.band_names] == (
        ["UNDEFINED"] * 4 + ["PASS"])                 # n 0 in 0-80 m, never dropped
    assert xw["0-20m"]["n_gt_cells"] == 0 and xw["80-100m"]["n_gt_cells"] == 6 * 20 * 100
    assert r["BAR-M7-1"]["verdict"] == r["BAR-M7-2"]["verdict"] == "PASS"
    assert r["BAR-M7-3"]["verdict"] == "PASS" and len(r["BAR-M7-4"]["per_band"]) == 5
    s = M.summarize(t, n_boot=50)
    assert s["bands"] == list(t.band_names)
    assert len([k for k in s["arms"]["refcv7"] if k.endswith("|iou")]) == 40   # 8 x 5
    assert s["arms"]["refcv7"]["crosswalk|0-20m|n_gt"] == 0
