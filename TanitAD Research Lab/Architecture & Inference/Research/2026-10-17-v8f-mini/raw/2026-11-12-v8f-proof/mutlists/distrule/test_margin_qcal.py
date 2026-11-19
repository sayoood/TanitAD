"""test_margin_qcal.py -- checks for the MARGIN and QCAL forms of DISTRULE (cards ADD30, ADD31), written for the mutation list (the lever folder has no test for them).
numpy only (the wrapper module is imported, never called on a model). Synthetic data; no number here is about a real arm.
Run with the folder of the distrule tool files first on sys.path (the mutation driver does it); plain python through minipytest or pytest."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.environ.get("DISTRULE_SRC", os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import margin_calib as MC  # noqa: E402
import qcal_fit as QF  # noqa: E402
import qcal_map as QM  # noqa: E402
import v8f_t1_eval_rule_margin as MG  # noqa: E402
import v8f_t1_eval_rule_qcal as QC  # noqa: E402


# --------------------------------------------------------------------------------------------- qcal_map
XS = np.array([10.0, 20.0, 40.0])
YS = np.array([6.0, 14.0, 33.0])


def test_qmap_below_between_above_and_min_with_the_head_gap():
    g = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 8.0])
    got = QM.apply_map(g, XS, YS)
    want = np.array([3.0, 6.0, 10.0, 14.0, 23.5, 33.0, 53.0, 4.8])
    assert np.allclose(got, want, atol=1e-9), (got, want)
    assert np.all(got <= g + 1e-12)


def test_qmap_nan_stays_and_identity_map_changes_nothing():
    out = QM.apply_map(np.array([np.nan, 7.0]), XS, YS)
    assert np.isnan(out[0]) and np.isfinite(out[1])
    g = np.linspace(0.5, 120.0, 200)
    assert np.allclose(QM.apply_map(g, np.array([1.0, 100.0]), np.array([1.0, 100.0])), g)


def test_qmap_unit_check_passes_and_refuses_bad_knots():
    assert QM.unit_check(XS, YS)["pass"]
    for xs, ys in ((np.array([10.0, 5.0]), np.array([1.0, 2.0])), (np.array([10.0, 20.0]), np.array([9.0, 5.0])), (np.array([10.0, 20.0]), np.array([11.0, 12.0])),
                   (np.array([0.0, 20.0]), np.array([0.0, 12.0])), (np.array([10.0]), np.array([5.0]))):
        with pytest.raises(ValueError):
            QM.check_map(xs, ys)


# --------------------------------------------------------------------------------------------- qcal_fit
def test_pav_pools_adjacent_violators_with_weights():
    out = QF.pav([1.0, 3.0, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(out, [1.0, 2.5, 2.5, 4.0])
    out = QF.pav([5.0, 1.0], [1.0, 3.0])
    assert np.allclose(out, [2.0, 2.0])


def _world(tmp, n=6000, seed=0, over_max=10.0, logit=3.0):
    rng = np.random.default_rng(seed)
    gl = rng.uniform(3.0, 60.0, n)
    gh = gl + rng.uniform(0.0, over_max, n)
    t = np.arange(n)
    np.savez(os.path.join(tmp, "A_train_lead.npz"), gap_m=gh, has_history=np.ones(n, bool), presence_logit=np.full(n, logit))
    np.savez(os.path.join(tmp, "A_train.npz"), clips=np.array(["c0"]), clip_idx=np.zeros(n, np.int64), t=t)
    pack = {"index": {("c0", int(i)): int(i) for i in range(n)}, "gap": gl, "pres": np.ones(n)}
    return pack, gl, gh


def test_fit_arm_gives_the_20th_percentile_map_monotone_below_the_diagonal(tmp_path):
    pack, gl, gh = _world(str(tmp_path))
    files = {}
    r = QF.fit_arm("A", str(tmp_path), str(tmp_path), pack, files)
    xs, ys = np.array(r["xs"]), np.array(r["ys"])
    assert np.all(np.diff(ys) >= 0) and np.all(ys <= xs + 1e-9) and np.all(np.diff(xs) > 0)
    # the label gap is gh minus U(0, 10): its 20th percentile at a head gap g is about g - 8 (in the interior)
    mid = (xs > 15.0) & (xs < 50.0)
    assert mid.sum() >= 2
    assert np.all(np.abs((xs - ys)[mid] - 8.0) < 1.6), (xs, ys)
    assert r["q_level"] == 20.0 and r["unit_check"]["pass"] and len(files) == 2


def test_fit_arm_refuses_an_evaluation_file(tmp_path):
    pack, gl, gh = _world(str(tmp_path))
    os.replace(os.path.join(str(tmp_path), "A_train_lead.npz"), os.path.join(str(tmp_path), "A_eval_train_lead.npz"))
    with pytest.raises(SystemExit):
        QF.fit_arm("A_eval", str(tmp_path), str(tmp_path), pack, {})


# --------------------------------------------------------------------------------------------- margin_calib
def test_up_half_rounds_up_to_half_metres_and_not_below_zero():
    assert [MC.up_half(x) for x in (-1.0, 0.0, 0.01, 0.5, 2.25, 2.5, 2.51)] == [0.0, 0.0, 0.5, 0.5, 2.5, 2.5, 3.0]


def test_calib_arm_over_read_percentiles_and_recall_on_the_near_rows(tmp_path):
    pack, gl, gh = _world(str(tmp_path), over_max=10.0, logit=3.0)
    r = MC.calib_arm("A", str(tmp_path), str(tmp_path), pack, {})
    pc = r["over_read_percentiles_m"]
    assert abs(pc["p50"] - 5.0) < 0.5 and abs(pc["p80"] - 8.0) < 0.5 and abs(pc["p95"] - 9.5) < 0.5, pc
    assert r["M50"] >= pc["p50"] and r["M50"] - pc["p50"] < 0.5 + 1e-9 and r["M95"] >= pc["p95"]
    assert r["recall_by_P"]["0.5"] == 1.0 and r["P_star"] == 0.5 and r["recall_bar_met"]
    n_near = int(((gl <= 40.0)).sum())
    assert r["n_near_lead"] == n_near and r["n_over_read_rows"] == n_near and r["n_label_absent"] == 0


def test_calib_arm_recall_counts_only_near_rows_and_picks_the_lowest_p_that_meets_the_bar(tmp_path):
    n = 3000
    rng = np.random.default_rng(3)
    gl = rng.uniform(3.0, 60.0, n)
    logit = np.where(gl <= 40.0, 1.5, -6.0)       # near rows: p = 0.82 (>= 0.5 .. all P), far rows: p ~ 0.0025
    np.savez(os.path.join(str(tmp_path), "A_train_lead.npz"), gap_m=gl + 1.0, has_history=np.ones(n, bool), presence_logit=logit)
    np.savez(os.path.join(str(tmp_path), "A_train.npz"), clips=np.array(["c0"]), clip_idx=np.zeros(n, np.int64), t=np.arange(n))
    pack = {"index": {("c0", int(i)): int(i) for i in range(n)}, "gap": gl, "pres": np.ones(n)}
    r = MC.calib_arm("A", str(tmp_path), str(tmp_path), pack, {})
    assert r["recall_by_P"]["0.5"] == 1.0 and r["n_near_lead"] == int((gl <= 40.0).sum())
    assert abs(r["over_read_percentiles_m"]["p50"] - 1.0) < 1e-9


# --------------------------------------------------------------------------------------------- the margin wrapper
def _stub(p, g, c, ok):
    return lambda model, feats, frame_ok: (p, g, c, ok)


def test_margin_lowers_only_finite_head_gaps_never_below_zero_and_zero_margin_is_identity(monkeypatch):
    g = np.array([10.0, 2.0, np.nan, 30.0], np.float32)
    p = np.array([0.9, 0.9, 0.1, 0.9], np.float32)
    c = np.zeros(4, np.float32)
    ok = np.ones(4, bool)
    monkeypatch.setattr(MG, "_ORIG_HEAD_VALUES", _stub(p, g, c, ok))
    monkeypatch.setitem(MG.CONST, "gap_margin_m", 0.0)
    p0, g0, c0, ok0 = MG.head_values_margin(None, None, None)
    assert g0 is g or np.array_equal(g0, g, equal_nan=True)
    monkeypatch.setitem(MG.CONST, "gap_margin_m", 3.0)
    p1, g1, c1, ok1 = MG.head_values_margin(None, None, None)
    assert np.allclose(g1[[0, 1, 3]], [7.0, 0.0, 27.0]) and np.isnan(g1[2]) and g1.dtype == g.dtype
    assert p1 is p and c1 is c and ok1 is ok


def test_margin_main_refusals_and_it_installs_the_constants_and_the_head_read(monkeypatch):
    mix = ["--mix-presence", "head", "--mix-gap", "head", "--mix-closing", "head", "run"]
    for bad in (["--head-p-min", "1.0", "--gap-margin-m", "1.0"], ["--head-p-min", "0.0", "--gap-margin-m", "1.0"], ["--head-p-min", "0.5", "--gap-margin-m", "31.0"],
                ["--head-p-min", "0.5", "--gap-margin-m", "-1.0"]):
        with pytest.raises(SystemExit, match="REFUSED"):
            MG.main(bad + mix)
    with pytest.raises(SystemExit, match="REFUSED"):
        MG.main(["--head-p-min", "0.5", "--gap-margin-m", "1.0", "run"])
    seen = {}
    monkeypatch.setattr(MG.X, "main", lambda rest: seen.setdefault("rest", rest) and 0)
    monkeypatch.setattr(MG.W, "HEAD_P_MIN", 0.5)
    monkeypatch.setattr(MG.W, "head_values", MG.W.head_values)
    MG.main(["--head-p-min", "0.35", "--gap-margin-m", "4.5"] + mix)
    assert MG.W.HEAD_P_MIN == 0.35 and MG.CONST["gap_margin_m"] == 4.5 and MG.W.head_values is MG.head_values_margin and seen["rest"] == mix


# --------------------------------------------------------------------------------------------- the qcal wrapper
def test_qcal_wrapper_applies_the_map_to_finite_head_gaps_only(tmp_path, monkeypatch):
    src = open(os.path.join(HERE, "v8f_t1_eval_rule_qcal.py"), encoding="utf-8").read()
    assert "apply_map" in src and "HEAD_P_MIN" in src
    import json
    mp = os.path.join(str(tmp_path), "m.json")
    json.dump({"xs": [10.0, 20.0, 40.0], "ys": [6.0, 14.0, 33.0]}, open(mp, "w"))
    g = np.array([5.0, 15.0, np.nan, 60.0], np.float32)
    p = np.array([0.9, 0.9, 0.1, 0.9], np.float32)
    monkeypatch.setattr(QC, "_ORIG_HEAD_VALUES", _stub(p, g, np.zeros(4, np.float32), np.ones(4, bool)))
    xs, ys = QM.load_map(mp)
    monkeypatch.setitem(QC.MAP, "xs", xs)
    monkeypatch.setitem(QC.MAP, "ys", ys)
    p1, g1, c1, ok1 = QC.head_values_qcal(None, None, None)
    assert np.allclose(g1[[0, 1, 3]], [3.0, 10.0, 53.0], atol=1e-4) and np.isnan(g1[2])


def test_qcal_main_refuses_bad_p_min_and_a_map_that_fails_and_installs_the_map_and_the_head_read(tmp_path, monkeypatch):
    import json
    mix = ["--mix-presence", "head", "--mix-gap", "head", "--mix-closing", "head", "run"]
    good = os.path.join(str(tmp_path), "good.json")
    json.dump({"xs": [10.0, 20.0, 40.0], "ys": [6.0, 14.0, 33.0]}, open(good, "w"))
    with pytest.raises(SystemExit, match="REFUSED"):
        QC.main(["--head-p-min", "1.0", "--gap-map", good] + mix)
    with pytest.raises(SystemExit, match="REFUSED"):
        QC.main(["--head-p-min", "0.5", "--gap-map", good, "run"])
    bad = os.path.join(str(tmp_path), "bad.json")
    json.dump({"xs": [10.0, 20.0], "ys": [6.0, 25.0]}, open(bad, "w"))
    with pytest.raises(ValueError):
        QC.main(["--head-p-min", "0.5", "--gap-map", bad] + mix)
    monkeypatch.setattr(QC.X, "main", lambda rest: 0)
    monkeypatch.setattr(QC.W, "HEAD_P_MIN", 0.5)
    monkeypatch.setattr(QC.W, "head_values", QC.W.head_values)
    QC.main(["--head-p-min", "0.25", "--gap-map", good] + mix)
    assert QC.W.HEAD_P_MIN == 0.25 and QC.W.head_values is QC.head_values_qcal and list(QC.MAP["xs"]) == [10.0, 20.0, 40.0]
