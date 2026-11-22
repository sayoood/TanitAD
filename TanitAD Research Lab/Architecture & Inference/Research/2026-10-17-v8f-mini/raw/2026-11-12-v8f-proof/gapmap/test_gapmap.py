"""tests of gapmap_t0.py (card ADD112). Synthetic data only."""
import io
import json
import os
import contextlib
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.environ.get("GAPMAP_KIT_GAP", os.path.join(HERE, "..", "gaperr", "gap"))
sys.path.insert(0, HERE)
import gapmap_t0 as gm  # noqa: E402

G = gm.import_kit(KIT)


def world(tmp_path):
    d = str(tmp_path / "world")
    jp, bp, rp, op, lp, lop = G.write_world(d)
    rec_dir = str(tmp_path / "rec")
    G.build(jp, bp, rp, lp, rec_dir)
    return jp, bp, os.path.join(rec_dir, G.RECORD_NAME), op


def make_tests_file(tmp_path, text="3 passed in 0.1s\n", age=10.0, name="t.out"):
    os.makedirs(str(tmp_path), exist_ok=True)
    p = str(tmp_path / name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    t = time.time() - age
    os.utime(p, (t, t))
    return p


# ---------------------------------------------------------------- isotonic
def test_pava_matches_maxmin_and_is_monotone():
    rng = np.random.default_rng(5)
    for t in range(150):
        n = int(rng.integers(2, 61))
        x = rng.integers(0, 10, n).astype(float) if t % 2 else rng.normal(0, 4, n)
        y = x * 0.5 + rng.normal(0, 2, n)
        bx, by, ux, gfit, cnt = gm.pava(x, y)
        assert np.all(np.diff(by) >= 0) and np.all(np.diff(gfit) >= 0)
        u2, gmn, w = gm.group_means(x, y)
        assert np.array_equal(u2, ux)
        assert np.abs(gm.maxmin_fit(gmn, w) - gfit).max() <= 1e-9
        assert np.all(np.diff(bx) > 0)


def test_ties_are_pooled_before_the_fit():
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    y = np.array([0.0, 10.0, 4.0, 4.0, 9.0])
    bx, by, ux, gfit, cnt = gm.pava(x, y)
    assert list(ux) == [1.0, 2.0, 3.0] and list(cnt) == [2, 2, 1]
    # group means 5, 4, 9 -> first two pooled to 4.5 (weights equal)
    assert np.allclose(gfit, [4.5, 4.5, 9.0])
    assert np.allclose(bx, [1.5, 3.0]) and np.allclose(by, [4.5, 9.0])


def test_ends_are_held():
    bx, by = np.array([1.0, 2.0]), np.array([5.0, 7.0])
    assert list(gm.iso_apply(bx, by, np.array([0.0, 1.5, 9.0]))) == [5.0, 6.0, 7.0]
    assert gm.iso_apply(np.array([3.0]), np.array([4.0]), np.array([0.0, 9.0])).tolist() == [4.0, 4.0]


# ---------------------------------------------------------------- folds
def test_fold_assignment_and_no_leak():
    names = ["c%02d" % i for i in range(23)][::-1] + ["nolab"]
    n = len(names)
    ci = np.repeat(np.arange(n), 4)
    err = ci != n - 1
    clips_err, fold_of, rowfold = gm.make_folds(ci, names, err)
    order = sorted(range(n - 1), key=lambda c: names[c])
    assert clips_err == order
    assert all(fold_of[c] == i % 5 for i, c in enumerate(order))
    assert (rowfold[~err] == -1).all()
    x = np.arange(len(ci), dtype=float)
    mapped, diag = gm.cross_fit(ci, names, err, x, x * 2.0, "iso")
    assert diag["leak_rows_total"] == 0 and diag["mapped_once_on_error_rows"] and diag["mapped_zero_off_error_rows"]
    assert diag["n_fold_assignments"] == n - 1
    _, dleak = gm.cross_fit(ci, names, err, x, x * 2.0, "iso", fault="leak")
    assert dleak["leak_rows_total"] > 0


# ---------------------------------------------------------------- planted worlds
def test_planted_monotone_world():
    A = gm.analyse(*gm.planted_world("monotone"))
    assert A["cuts"]["IN"]["cut_e"] >= 0.5 and A["cuts"]["CF"]["cut_e"] >= 0.5


def test_planted_offset_world():
    A = gm.analyse(*gm.planted_world("offset"))
    assert A["cuts"]["CF"]["cut_s"] < 0.05
    assert gm.check_c1() == [] and gm.check_c2() == []


# ---------------------------------------------------------------- identity equals the imported tool
def test_identity_block_equals_t0_table(tmp_path):
    jp, bp, rp, op = world(tmp_path)
    C = G.compose(G.load_npz(jp), G.load_npz(bp))
    A, _ = gm.values_of(C)
    ref = G.t0_table(C)
    for r in gm.RANGES:
        mine = A["blocks"]["identity"][r]
        assert mine["n_error_rows"] == ref[r]["n_error_rows"]
        for k in ("rms_e", "rms_s", "rms_f"):
            assert mine[k] == ref[r][k]


# ---------------------------------------------------------------- names
def test_name_of_four_combinations():
    assert gm.name_of((0.12, 0.3), (0.2, 0.4)) == gm.NAME_CUTS
    assert gm.name_of((-0.1, 0.05), (0.0, 0.08)) == gm.NAME_NOT
    assert gm.name_of((0.12, 0.3), (0.0, 0.08)) == gm.NAME_ND       # both conditions
    assert gm.name_of((0.0, 0.3), (0.0, 0.3)) == gm.NAME_ND         # neither
    assert gm.name_of((0.10, 0.3), (0.0, 0.3)) == gm.NAME_CUTS      # lower end at the bar counts


# ---------------------------------------------------------------- controls and their fault arms
def run(tmp_path, **kw):
    jp, bp, rp, op = world(tmp_path)
    tp = kw.pop("tests", None) or make_tests_file(tmp_path)
    rec = kw.pop("record", rp)
    base = kw.pop("base", bp)
    t_start = kw.pop("t_start", time.time())
    ctrl, _ = gm.run_controls(G.load_npz(jp), G.load_npz(bp), base, rec, tp, KIT, t_start, **kw)
    return ctrl, rp


def test_clean_controls_all_pass(tmp_path):
    ctrl, _ = run(tmp_path)
    assert all(v["pass"] for v in ctrl.values()), ctrl


def test_c0_shifted_record_fails(tmp_path):
    jp, bp, rp, op = world(tmp_path)
    rec = json.load(open(rp, encoding="utf-8"))
    rec["t0"]["near"]["rms_s"] += 1e-3
    sp = str(tmp_path / "shifted.json")
    json.dump(rec, open(sp, "w", encoding="utf-8"))
    ctrl, _ = gm.run_controls(G.load_npz(jp), G.load_npz(bp), bp, sp, make_tests_file(tmp_path), KIT, time.time())
    assert not ctrl["c0"]["pass"] and ctrl["c3"]["pass"]


def test_c3_leak_fails(tmp_path):
    ctrl, _ = run(tmp_path, fault="leak")
    assert not ctrl["c3"]["pass"] and ctrl["c0"]["pass"]


def test_c4_wrong_sha_fails(tmp_path):
    ctrl, _ = run(tmp_path, base=str(tmp_path / "world" / "other_base.npz"))
    assert not ctrl["c4"]["pass"]
    ctrl2, _ = run(tmp_path / "b", kit_prefix="0000000000000000")
    assert not ctrl2["c4"]["pass"]


def test_c5_variants(tmp_path):
    t = time.time()
    assert gm.check_c5(make_tests_file(tmp_path, "4 passed in 1s\n"), t) == []
    assert gm.check_c5(make_tests_file(tmp_path, "1 failed, 3 passed in 1s\n", name="a.out"), t)
    assert gm.check_c5(make_tests_file(tmp_path, "2 errors in 1s\n", name="b.out"), t)
    assert gm.check_c5(make_tests_file(tmp_path, "no tests ran\n", name="c.out"), t)
    late = make_tests_file(tmp_path, "4 passed in 1s\n", age=-100.0, name="d.out")
    assert gm.check_c5(late, t)
    assert gm.check_c5(str(tmp_path / "missing.out"), t)


# ---------------------------------------------------------------- stages
def args_for(tmp_path, out):
    jp, bp, rp, op = world(tmp_path)
    return type("A", (), {"join": jp, "base": bp, "gap_record": rp, "tests_out": make_tests_file(tmp_path), "out": out, "kit_gap": KIT})()


def test_out_exists_is_refused(tmp_path):
    out = str(tmp_path / "o.json")
    open(out, "w").write("{}")
    a = args_for(tmp_path, out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gm.controls_stage(a, time.time())
    assert rc == 2 and open(out).read() == "{}"


def test_stages_end_to_end_and_names_refuse(tmp_path):
    out = str(tmp_path / "r" / "o.json")
    a = args_for(tmp_path, out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gm.controls_stage(a, time.time())
    text = buf.getvalue()
    assert rc == 0 and "ZZ GAPMAP-T0 controls all_pass True" in text
    assert all(ln.startswith(("c", "ZZ")) for ln in text.strip().splitlines()) and "." not in text.replace("GAPMAP-T0", "")
    doc = json.load(open(out, encoding="utf-8"))
    assert "payload" in doc and "withheld" not in doc
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert gm.names_stage(out) == 0
    lines = buf.getvalue().splitlines()
    assert lines[0].startswith("TIER T0") and lines[1].startswith("NAME GAPMAP-T0-") and lines[-1] == "ZZ GAPMAP-T0 names done"
    doc["all_pass"] = False
    bad = str(tmp_path / "bad.json")
    json.dump(doc, open(bad, "w", encoding="utf-8"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert gm.names_stage(bad) == 2
    assert "rms" not in buf.getvalue()


def test_failed_control_withholds_values(tmp_path):
    out = str(tmp_path / "o.json")
    a = args_for(tmp_path, out)
    a.tests_out = make_tests_file(tmp_path, "1 failed in 1s\n", name="f.out")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gm.controls_stage(a, time.time())
    doc = json.load(open(out, encoding="utf-8"))
    assert rc == 1 and doc["all_pass"] is False and "withheld" in doc and "payload" not in doc
    assert "c5 FAIL" in buf.getvalue() and "rms" not in buf.getvalue()


def test_selftest_runs(tmp_path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gm.selftest(KIT)
    assert rc == 0 and "ZZ GAPMAP-T0 SELFTEST OK" in buf.getvalue()


# ---------------------------------------------------------------- the kit's own files
def kit_files():
    return [os.path.join(HERE, n) for n in ("gapmap_t0.py", "test_gapmap.py", "GAPMAP_DEFS.md", "run_gapmap_pod.sh")]


def test_banned_words_absent():
    words = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "closed" + " loop"]
    dp = "driving" + " performance"
    for p in kit_files():
        txt = open(p, encoding="utf-8").read().lower()
        for w in words:
            assert w not in txt, (p, w)
        i = txt.find(dp)
        while i >= 0:
            assert txt[max(0, i - 4):i] == "not ", (p, "driving")
            i = txt.find(dp, i + 1)


def test_sizes_names_and_line_ends():
    for p in kit_files():
        assert os.path.getsize(p) < 5 * 1024 * 1024
        assert len(os.path.abspath(p)) <= 200
        assert os.path.basename(p).split(".")[0] not in ("aux", "con", "prn", "nul")
    assert b"\r" not in open(os.path.join(HERE, "run_gapmap_pod.sh"), "rb").read()
