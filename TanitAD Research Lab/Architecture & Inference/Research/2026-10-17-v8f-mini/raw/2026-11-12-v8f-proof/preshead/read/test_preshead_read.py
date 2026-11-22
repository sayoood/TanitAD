"""test_preshead_read.py -- tests of the T0 reader of card PREREG_V8F_PROOF_ADD118 (preshead_t0.py) on a small planted world built with the synthetic world of the
reader of card ADD116 (imported unchanged). Run from the reader folder with ../fit (the fit kit) and ../../preserr/preserr_t0.py beside it."""
import argparse
import io
import json
import os
import shutil
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = os.path.normpath(os.path.join(HERE, "..", "fit"))
PRESERR = os.path.normpath(os.path.join(HERE, "..", "..", "preserr", "preserr_t0.py"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import preshead_t0 as R  # noqa: E402

PE = R.load_preserr(PRESERR)
NS = argparse.Namespace
LIST_SHA = {"train_list_sha256": "a" * 64, "eval_list_sha256": "b" * 64}


def _planted(L, rng, miss, extra):
    u = rng.random(len(L))
    return np.where(L, np.where(u < miss, 0.2, 0.9), np.where(u < extra, 0.8, 0.1))


def build(base, miss=0.03, extra=0.05, perm="random", fault=None):
    """-> Namespace for run_controls and the expected reader values of the world"""
    t1, jp, _ = PE.make_world(os.path.join(base, "w"))
    root = os.path.join(base, "root")
    os.makedirs(os.path.join(root, "kit_fit"))
    for f in ("PRES_FIT_GEN_RECORD.json", "pres_fit.py"):
        shutil.copy(os.path.join(FIT, f), os.path.join(root, "kit_fit", f))
    vsha = R.sha_file(os.path.join(root, "kit_fit", "pres_fit.py"))
    if fault == "p0":
        with io.open(os.path.join(root, "kit_fit", "pres_fit.py"), "ab") as fh:
            fh.write(b"\n# changed after the record\n")
    J = PE.npz(jp)
    lv, lg = np.asarray(J["lead_valid"]), np.asarray(J["lead_gap"], np.float64)
    Lrow = (lv == 1) & np.isfinite(lg) & (lg >= 0.0) & (lg <= 60.0)
    jsha = R.sha_file(jp)
    ref = os.path.join(base, "ref_record.json")
    json.dump(dict(LIST_SHA), io.open(ref, "w", encoding="utf-8"))
    for i, (name, seed, pm) in enumerate(R.FITS):
        rng = np.random.default_rng(40 + i)
        p = (rng.random(len(Lrow)) if perm == "random" else _planted(Lrow, rng, 0.0, 0.0)) if pm else _planted(Lrow, rng, miss, extra)
        hp = dict(R.HPARAMS)
        rec = {"card": R.CARD, "tier": "T0", "arm": "D", "head": "H5", "seed": seed, "permute_labels": pm, "tool_sha256": vsha, "label": "presence", "hparams": hp, "hparams_equal_card": True,
               "n_samples": {"fit": 1000, "sel": 100}, "n_positive": {"fit": 600, "sel": 60}, "kept_epoch": 2, "kept_sel_loss": 0.3, "seconds": 10.0, "head_pt_sha256": "%064d" % i}
        rec.update(LIST_SHA)
        if fault == "p1" and i == 1:
            rec["eval_list_sha256"] = "c" * 64
        if fault == "p2_epochs" and i == 0:
            rec["hparams"] = dict(hp, epochs=40)
        if fault == "p2_classes" and i == 2:
            rec["n_positive"] = {"fit": 0, "sel": 60}
        if fault == "p2_tool" and i == 1:
            rec["tool_sha256"] = "d" * 64
        os.makedirs(os.path.join(root, "fits", name))
        json.dump(rec, io.open(os.path.join(root, "fits", name, "FIT_RECORD.json"), "w", encoding="utf-8"))
        if fault == "p5_nan" and i == 0:
            p = p.copy()
            p[np.flatnonzero(Lrow)[:3]] = np.nan
        if fault == "p5_rows" and i == 1:
            p = p[:-1]
        os.makedirs(os.path.join(root, "dump", name))
        np.savez(os.path.join(root, "dump", name, "pres.npz"), p=p, has_out=np.isfinite(p))
        meta = {"n_rows": int(len(p)), "head_pt_sha256": rec["head_pt_sha256"] if not (fault == "p5_head" and i == 2) else "e" * 64, "eval_join_sha256": jsha}
        json.dump(meta, io.open(os.path.join(root, "dump", name, "pres_meta.json"), "w", encoding="utf-8"))
    now = time.time()
    tf, tr = os.path.join(base, "lead_fit.out"), os.path.join(base, "lead_read.out")
    io.open(tf, "w", encoding="utf-8").write("12 passed in 50.0s\n")
    io.open(tr, "w", encoding="utf-8").write("9 passed in 1.0s\n" if fault != "p6_failed" else "1 failed, 8 passed in 1.0s\n")
    os.utime(tf, (now - 100, now - 100))
    os.utime(tr, (now - 10, now - 10))
    io.open(os.path.join(root, "PRESHEAD_START.txt"), "w", encoding="utf-8").write("%d 2026-11-22T16:00:00Z pid 1\n" % int(now - (50 if fault != "p6_late" else 500)))
    exp = PE.run_controls(NS(t1=t1, join=jp, tests=tf, n_boot=20))["payload"]
    e = {"M": exp["pooled"]["M_0.5"]["value"], "X": exp["pooled"]["X_0.5"]["value"], "n_frames": exp["per_seed"][0]["n_frames"], "n_L": exp["per_seed"][0]["n_L"], "n_not_L": exp["per_seed"][0]["n_not_L"]}
    return NS(preserr=PRESERR, t1=t1, join=jp, root=root, ref_fit_record=ref, tests_fit=tf, tests_read=tr, n_boot=200), e


def run(base, monkeypatch, card_expect=False, **kw):
    a, e = build(str(base), **kw)
    if not card_expect:
        monkeypatch.setattr(R, "EXPECT", e)
    return R.run_controls(a, now=time.time()), a, e


def test_card_constants():
    assert R.EXPECT == {"M": 0.1558, "X": 0.2244, "n_frames": 10865, "n_L": 3383, "n_not_L": 7482}
    assert abs(R.BAR_M - 0.5 * 0.1558) < 1e-12 and abs(R.BAR_X - 2 * 0.2244) < 1e-12 and R.AUC_PERM == (0.40, 0.60) and R.HPARAMS["epochs"] == 8
    assert R.CONTROLS == tuple("p%d" % i for i in range(7)) and [f[0] for f in R.FITS] == ["D_H5_s0", "D_H5_s1", "D_H5_s0_perm"]


def test_clean_world_good_head_meets_the_bar(tmp_path, monkeypatch):
    res, a, e = run(tmp_path, monkeypatch)
    assert res["all_pass"] is True, {k: v["failures"] for k, v in res["controls"].items() if not v["pass"]}
    P = res["payload"]
    assert abs(P["forms"]["READER"]["pooled"]["M"]["value"] - e["M"]) < 1e-12 and abs(P["forms"]["READER"]["pooled"]["X"]["value"] - e["X"]) < 1e-12
    h, o = P["forms"]["HEAD"]["pooled"], P["forms"]["HEAD-OR-READER"]["pooled"]
    assert h["M"]["value"] < 0.0779 and o["M"]["value"] <= h["M"]["value"] and o["M"]["value"] <= e["M"] and o["X"]["value"] >= max(h["X"]["value"], e["X"]) - 1e-12
    assert P["differences"]["HEAD"]["M"]["interval"][1] < 0 and abs(P["differences"]["HEAD"]["M"]["value"] - (h["M"]["value"] - e["M"])) < 1e-12
    n = R.names_of(P)
    assert n["HEAD"] == "PRESHEAD-T0-HEAD-MEETS-BAR" and n["HEAD-OR-READER"] == "PRESHEAD-T0-HEAD-OR-READER-MEETS-BAR" and n["OVERALL"] == "PRESHEAD-T0-BAR-MET" and n["named_form"] == "HEAD"
    assert all(0.40 <= s["auc"] <= 0.60 for s in P["forms"]["PERMUTED"]["per_seed"]) and set(P["fits"]) == {f[0] for f in R.FITS}
    lines = R.print_names(P, n)
    assert sum(ln.startswith("NAME ") for ln in lines) == 3 and any(ln.startswith("no name: HEAD at the threshold") for ln in lines)


def test_head_that_marks_nothing_is_no_cut(tmp_path, monkeypatch):
    res, _, e = run(tmp_path, monkeypatch, miss=1.0, extra=0.0)
    n = R.names_of(res["payload"])
    assert res["payload"]["forms"]["HEAD"]["pooled"]["M"]["value"] == 1.0 and abs(res["payload"]["forms"]["HEAD-OR-READER"]["pooled"]["M"]["value"] - e["M"]) < 1e-12
    assert n["HEAD"].endswith("HEAD-NO-CUT-OF-MISSES") and n["HEAD-OR-READER"].endswith("-NO-CUT-OF-MISSES") and n["OVERALL"] == "PRESHEAD-T0-NO-CUT" and n["named_form"] is None


def test_partial_head_cuts_but_misses_the_bar(tmp_path, monkeypatch):
    res, _, e = run(tmp_path, monkeypatch, miss=0.5, extra=0.02)
    P = res["payload"]
    n = R.names_of(P)
    o = P["forms"]["HEAD-OR-READER"]["pooled"]["M"]["value"]
    assert e["M"] > 0.2 and 0.0779 < o < e["M"]                                                   # the world's reader misses about 0.3; OR with a head that catches half
    assert n["HEAD"].endswith("HEAD-NO-CUT-OF-MISSES") and n["HEAD-OR-READER"].endswith("-CUTS-MISSES-NOT-TO-BAR")
    assert n["OVERALL"] == "PRESHEAD-T0-MISSES-CUT-BAR-NOT-MET" and n["named_form"] == "HEAD-OR-READER"


def test_extra_marks_above_the_bar_give_no_name_of_a_cut(tmp_path, monkeypatch):
    res, _, _ = run(tmp_path, monkeypatch, miss=0.0, extra=0.6)
    P = res["payload"]
    n = R.names_of(P)
    assert P["forms"]["HEAD"]["pooled"]["M"]["value"] == 0.0 and P["forms"]["HEAD"]["pooled"]["X"]["value"] > 0.4488
    assert n["HEAD"].endswith("HEAD-NO-CUT-OF-MISSES") and n["HEAD-OR-READER"].endswith("-NO-CUT-OF-MISSES") and n["OVERALL"] == "PRESHEAD-T0-NO-CUT"


@pytest.mark.parametrize("fault,ctl", [("p0", "p0"), ("p1", "p1"), ("p2_epochs", "p2"), ("p2_classes", "p2"), ("p2_tool", "p2"), ("p5_nan", "p5"), ("p5_rows", "p5"), ("p5_head", "p5"),
                                       ("p6_failed", "p6"), ("p6_late", "p6")])
def test_a_planted_fault_fails_its_control_and_withholds_the_values(tmp_path, monkeypatch, fault, ctl):
    res, _, _ = run(tmp_path, monkeypatch, fault=fault)
    assert res["all_pass"] is False and res["controls"][ctl]["pass"] is False and res["controls"][ctl]["failures"] and "payload" not in res


def test_permuted_fit_that_reads_the_label_fails_p3(tmp_path, monkeypatch):
    res, _, _ = run(tmp_path, monkeypatch, perm="label")
    assert res["controls"]["p3"]["pass"] is False and "payload" not in res


def test_identity_fails_with_the_card_values_in_the_planted_world(tmp_path, monkeypatch):
    res, _, _ = run(tmp_path, monkeypatch, card_expect=True)
    assert res["controls"]["p4"]["pass"] is False and "payload" not in res and all(res["controls"][k]["pass"] for k in ("p0", "p1", "p2", "p3", "p5", "p6"))


def test_another_reader_of_card_add116_fails_p0(tmp_path, monkeypatch):
    a, e = build(str(tmp_path))
    monkeypatch.setattr(R, "EXPECT", e)
    monkeypatch.setattr(R, "PRESERR_SHA16", "0" * 16)
    assert R.run_controls(a, now=time.time())["controls"]["p0"]["pass"] is False


def test_stages_through_main(tmp_path, monkeypatch, capsys):
    a, e = build(str(tmp_path))
    monkeypatch.setattr(R, "EXPECT", e)
    cj, rj = str(tmp_path / "C.json"), str(tmp_path / "R.json")
    common = ["--preserr", a.preserr, "--t1", a.t1, "--join", a.join, "--root", a.root, "--ref-fit-record", a.ref_fit_record, "--tests-fit", a.tests_fit, "--tests-read", a.tests_read, "--n-boot", "100"]
    assert R.main(["--stage", "controls", "--out", cj] + common) == 0
    out = capsys.readouterr().out
    assert out.count(" PASS") == 7 and "ZZ PRESHEAD-T0 controls all_pass True" in out and "0." not in out          # the controls stage prints no value
    assert R.main(["--stage", "controls", "--out", cj] + common) == 2
    assert R.main(["--stage", "names", "--controls", cj, "--out", rj]) == 0
    out = capsys.readouterr().out
    assert "NAME PRESHEAD-T0-BAR-MET" in out and "ZZ PRESHEAD-T0 names done" in out
    d = json.load(io.open(rj, encoding="utf-8"))
    assert d["names"]["OVERALL"] == "PRESHEAD-T0-BAR-MET" and d["card"] == R.CARD
    c = json.load(io.open(cj, encoding="utf-8"))
    c["controls"]["p3"]["pass"] = False
    bad = str(tmp_path / "C_bad.json")
    json.dump(c, io.open(bad, "w", encoding="utf-8"))
    assert R.main(["--stage", "names", "--controls", bad, "--out", str(tmp_path / "R2.json")]) == 3 and not os.path.exists(str(tmp_path / "R2.json"))
