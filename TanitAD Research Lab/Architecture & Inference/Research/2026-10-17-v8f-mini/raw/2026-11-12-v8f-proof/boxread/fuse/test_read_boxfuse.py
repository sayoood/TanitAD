"""tests of read_boxfuse.py (card PREREG_V8F_PROOF_ADD129). Tiny synthetic worlds of the kits of cards ADD127 and ADD128 (their test modules are imported).
Run one file at a time. BOXREAD_KIT_READ and BOXREAD_KIT_C name the folders of those kits (defaults: ../read and ../read2, or the pod paths)."""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(env, rel, pod, mark):
    for cand in (os.environ.get(env), os.path.join(os.path.dirname(HERE), rel), pod):
        if cand and os.path.isfile(os.path.join(cand, mark)):
            return cand
    raise RuntimeError("kit folder not found: %s" % mark)


KIT = _find("BOXREAD_KIT_READ", "read", "/workspace/v8proof/boxread/kit/read", "read_boxread.py")
KITC = _find("BOXREAD_KIT_C", "read2", "/workspace/v8proof/boxread/kit_c", "read_boxread_c.py")
for p in (KIT, KITC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_read_boxread as T  # noqa: E402
import test_read_boxread_c as TC  # noqa: E402
import read_boxfuse as RF  # noqa: E402
from test_read_boxread import fast_render  # noqa: E402,F401
from test_read_boxread_c import small_floor  # noqa: E402,F401

RB, RC, BS = RF.RB, TC.RC, RF.BS


def world(tmp_path, capsys, with_fuse_test=True, **opt):
    """a world with the controls and the result of card ADD128 in its out folder -> (argv builder of the fuse reader, info)"""
    av, info = TC.world(tmp_path, capsys, **opt)
    RC.main(av("controls"))
    RC.main(av("names"))
    capsys.readouterr()
    d = info["dirs"]
    c128, r128 = os.path.join(d["out"], "BOXREAD_CONTROLS.json"), os.path.join(d["out"], "BOXREAD_RESULT.json")
    ft = os.path.join(d["tests"], "fuse.out")
    BS.wt(ft, "....\n9 passed in 1.0s\n")
    old = time.time() - 500
    os.utime(ft, (old, old))
    fout = str(tmp_path / "fuse_out")

    def fa(mode, extra=()):
        e = ["--add128-controls", c128, "--add128-result", r128] + list(extra)
        if with_fuse_test:
            e += ["--test-out", "fuse=" + ft, "--test-before", "fuse=%f" % (time.time() - 100)]
        return info["argv"](mode, out=fout, extra=e)
    info.update({"c128": c128, "r128": r128, "fout": fout})
    return fa, info


def run(argv, capsys):
    rc = RF.main(argv)
    return rc, capsys.readouterr().out


def cj_of(info):
    return json.load(open(os.path.join(info["fout"], "FUSE_CONTROLS.json"), encoding="utf-8"))


def test_gate_rule():
    g = RF.gate_of(np.array([10.0, 10.0, 10.0, 10.0]), np.array([True, True, True, False]), np.array([14.9, 15.1, 6.6, 10.0]))
    assert g.tolist() == [True, False, False, False]
    assert RF.gate_of(np.array([10.0, np.nan]), np.array([True, False]), np.array([-1.0, 5.0])).tolist() == [False, False]


def _crafted():
    n = 8
    gd = np.array([10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0])
    W = {"nc": 2, "ci": np.repeat([0, 1], 4), "labg": gd * 1.0, "gD": {k: gd.copy() for k in RF.ARM_D},
         "gw": np.array([12.0, 60.0, np.nan, 44.0, 9.0, 21.0, 90.0, np.nan]), "pw": np.array([True, True, False, True, True, True, True, False]),
         "gl": np.array([12.0, 22.0, 31.0, 44.0, 9.0, 21.0, 33.0, 41.0]), "pl": np.ones(n, bool)}
    W["same_pick"] = W["pw"] & (W["gw"] == W["gl"])
    return W, gd


def test_arms_take_the_mean_where_the_reads_agree_and_arm_d_elsewhere():
    W, gd = _crafted()
    gaps, gates = RF.arms_of(W)
    assert gates[0].tolist() == [True, False, False, True, True, True, False, False]
    fg = gaps["FG_0"]
    assert abs(fg[0] - np.sqrt(12.0 * 10.0)) < 1e-12 and fg[1] == 20.0 and fg[2] == 30.0 and fg[6] == 30.0 and fg[7] == 40.0
    fm = gaps["FM_0"]
    assert abs(fm[1] - np.sqrt(60.0 * 20.0)) < 1e-12 and fm[2] == 30.0
    fl = gaps["FL_0"]
    assert abs(fl[1] - np.sqrt(22.0 * 20.0)) < 1e-12 and abs(fl[7] - np.sqrt(41.0 * 40.0)) < 1e-12
    assert gaps["BW_0"][1] == 60.0 and gaps["BW_0"][2] == 30.0
    assert all(np.isfinite(g).all() for g in gaps.values())


def test_values_and_names_on_crafted_reads():
    W, gd = _crafted()
    W["labg"] = gd + np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    P = RF.compute_values(W, n_boot=200)
    assert P["n_rows"] == 8 and set(P["R"]) == {"BW", "FG", "FM", "FL", "FG_over_BW"}
    assert P["mean_E"]["BW"] > P["mean_E"]["FG"]
    assert P["beside"]["picks"]["n_both"] == 6 and P["beside"]["picks"]["n_same"] == 4
    names = RF.derive_names(P)
    assert len(names) == 2 and all(n.startswith("BOXFUSE-T0-") for n in names)


def test_every_name_is_reachable():
    assert RF.name_ratio([0.7, 0.95], 0.85) == "BOXFUSE-T0-BETTER"
    assert RF.name_ratio([0.9, 0.99], 0.95) == "BOXFUSE-T0-SMALL-GAIN"
    assert RF.name_ratio([1.01, 1.2], 1.1) == "BOXFUSE-T0-WORSE"
    assert RF.name_ratio([0.9, 1.1], 1.0) == "BOXFUSE-T0-NOT-SEPARATED"
    P = {"iv": {"FG": [0.9, 1.1]}, "R": {"FG": 1.0}, "far": {"signed": {"FG": -1.0, "D": -6.0}, "iv_diff_FG_minus_D": [1.0, 5.0]}}
    assert RF.derive_names(P)[1] == "BOXFUSE-T0-FAR-LEADS-LESS-SHORT"
    P["far"]["signed"]["FG"] = -4.0
    assert RF.derive_names(P)[1] == "BOXFUSE-T0-FAR-LEADS-STAY-SHORT"


def test_clean_world_controls_pass_print_no_error_and_names_open_after_them(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    monkeypatch.setattr(RF, "F3_MIN_SHARE", 0.05)
    fa, info = world(tmp_path, capsys)
    rc, out = run(fa("names"), capsys)
    assert rc == 3 and "REFUSED" in out
    rc, out = run(fa("controls"), capsys)
    lines = [x for x in out.splitlines() if x.strip()]
    assert rc == 0 and lines[-1] == "ZZ BOXFUSE CONTROLS ALL-PASS", out
    ok = re.compile(r"^(ctl f[0-5] PASS|f1 shares \d\.\d{4} \d\.\d{4}|f3 share \d\.\d{4}|ZZ BOXFUSE CONTROLS ALL-PASS)$")
    assert all(ok.match(x) for x in lines), lines
    cj = cj_of(info)
    assert cj["card"] == "PREREG_V8F_PROOF_ADD129" and set(cj["controls"]["f0"]["identical_to_card_add128"].values()) == {True}
    assert cj["controls"]["f1"]["share_pass_permuted"] <= 0.75 * cj["controls"]["f1"]["share_pass"]
    assert not os.path.exists(os.path.join(info["fout"], "FUSE_RESULT.json"))
    rc, out = run(fa("names"), capsys)
    names = [x.split()[-1] for x in out.splitlines() if x.startswith("ZZ BOXFUSE NAME ")]
    assert rc == 0 and len(names) == 2, out
    rj = json.load(open(os.path.join(info["fout"], "FUSE_RESULT.json"), encoding="utf-8"))
    assert rj["card"] == "PREREG_V8F_PROOF_ADD129" and rj["names"] == names
    r128 = json.load(open(info["r128"], encoding="utf-8"))
    for k in ("D_0", "BW_1"):
        assert abs(rj["measures"]["E"][k] - r128["measures"]["E"][k]) < 1e-9
    txt = open(os.path.join(info["fout"], "FUSE_VERDICT.md"), encoding="utf-8").read()
    assert "ADD129" in txt and "used a second time" in txt and "STRATEGIC: not computed, n = 0" in txt


def test_f1_fails_for_a_gate_that_lets_every_frame_pass(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    monkeypatch.setattr(RF, "F3_MIN_SHARE", 0.05)
    monkeypatch.setattr(RF, "GATE_FACTOR", 1e9)
    fa, info = world(tmp_path, capsys)
    rc, out = run(fa("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl f1 FAIL with arm D's gaps permuted the gate passes on 1\.0000", out, re.M), out


def test_f0_fails_on_a_changed_input_and_on_a_result_that_is_not_reproduced(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    monkeypatch.setattr(RF, "F3_MIN_SHARE", 0.05)
    fa, info = world(tmp_path, capsys)
    rj = json.load(open(info["r128"], encoding="utf-8"))
    rj["measures"]["E"]["BW_0"] += 0.01
    bad = os.path.join(str(tmp_path), "result_changed.json")
    BS.wt(bad, json.dumps(rj))
    argv = fa("controls")
    argv[argv.index("--add128-result") + 1] = bad
    rc, out = run(argv, capsys)
    assert rc == 2 and re.search(r"^ctl f0 FAIL BW_0: its error is not the banked value", out, re.M), out
    T.write_det(os.path.join(info["dirs"]["det/wide"], "zz_one_more.npz"), "wide", T.NFR, {}, T.WIDE, info["wsha"])
    rc, out = run(fa("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl f0 FAIL inputs differ from those of card ADD128: det_wide_listing", out, re.M), out


def test_f0_fails_without_the_files_of_card_add128(tmp_path, capsys, fast_render, small_floor):
    fa, info = world(tmp_path, capsys)
    argv = [x for x in fa("controls")]
    i = argv.index("--add128-controls")
    del argv[i:i + 4]
    rc, out = run(argv, capsys)
    assert rc == 2 and re.search(r"^ctl f0 FAIL no all-pass controls file of card ADD128 given", out, re.M), out


def test_f3_fails_when_no_frame_has_a_pick(tmp_path, capsys, fast_render, small_floor):
    fa, info = world(tmp_path, capsys, loc_offset=30.0)
    rc, out = run(fa("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl f3 FAIL the gate passes on 0\.0000", out, re.M) and re.search(r"^ctl f1 FAIL no picked frame", out, re.M), out


def test_f4_needs_the_test_output_of_this_tool(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    monkeypatch.setattr(RF, "F3_MIN_SHARE", 0.05)
    fa, info = world(tmp_path, capsys, with_fuse_test=False)
    rc, out = run(fa("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl f4 FAIL no test output named fuse", out, re.M), out


def test_f2_fails_when_a_label_file_stands_in_the_place_of_the_locator(tmp_path, capsys, fast_render, small_floor):
    fa, info = world(tmp_path, capsys, loc_label=True)
    rc, out = run(fa("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl f2 FAIL \S", out, re.M), out


def test_names_refuse_when_an_input_changed_after_the_controls(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    monkeypatch.setattr(RF, "F3_MIN_SHARE", 0.05)
    fa, info = world(tmp_path, capsys)
    rc, out = run(fa("controls"), capsys)
    assert rc == 0, out
    T.write_det(os.path.join(info["dirs"]["det/wide"], "zz_one_more.npz"), "wide", T.NFR, {}, T.WIDE, info["wsha"])
    rc, out = run(fa("names"), capsys)
    assert rc == 3 and "REFUSED: inputs differ" in out
