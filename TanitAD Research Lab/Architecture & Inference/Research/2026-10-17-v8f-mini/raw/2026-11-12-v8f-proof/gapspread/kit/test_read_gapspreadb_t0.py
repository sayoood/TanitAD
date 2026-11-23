"""tests of read_gapspreadb_t0.py (card PREREG_V8F_PROOF_ADD133). The worlds of the test of card ADD132 are reused. Run one file at a time."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import test_read_gapspread_t0 as TA  # noqa: E402
import read_gapspreadb_t0 as GB  # noqa: E402
from test_read_gapspread_t0 import fast_render, small_floor  # noqa: E402,F401

GS, BS = GB.GS, GB.BS
FOUR = ("9 passed in 1.0s", "9 passed in 2.0s", "7 passed in 1.0s", "7 passed in 2.0s")


def ns_b(info, x, tag, tmp_path, s3_of_card_a=False, tests=FOUR):
    xb, a = TA.ns(info, x.fit, tag, tmp_path, tests=tests)
    p = os.path.join(info["root"], "add132_controls_%s.json" % tag)
    BS.wt(p, json.dumps({"card": "PREREG_V8F_PROOF_ADD132", "pass_s3": s3_of_card_a, "all_pass": s3_of_card_a}))
    xb.add132_controls = p
    return xb, a


def test_names_of_both_kinds():
    assert GB.flat_name(0.85, [0.80, 0.90], [0.90, 1.05]) == ("GAPSPREADB-T0-PLACED-BETTER-THAN-FLAT", "GAPSPREADB-T0-BAR-MET")
    assert GB.flat_name(0.95, [0.92, 0.98], [0.80, 0.95]) == ("GAPSPREADB-T0-PLACED-BETTER-THAN-FLAT", "GAPSPREADB-T0-BAR-NOT-MET")
    assert GB.flat_name(0.85, [0.80, 0.90], [1.02, 1.10])[0] == "GAPSPREADB-T0-PLACED-TRADES"
    assert GB.flat_name(1.10, [1.02, 1.20], [0.80, 0.95])[0] == "GAPSPREADB-T0-PLACED-WORSE-THAN-FLAT"
    assert GB.flat_name(0.97, [0.92, 1.01], [0.80, 0.95])[0] == "GAPSPREADB-T0-PLACED-NOT-SEPARATED"
    assert GB.mark_name([0.8, 0.9]) == "GAPSPREADB-T0-SPREAD-MARKS-LONG-READS" and GB.mark_name([1.1, 1.2]) == "GAPSPREADB-T0-SPREAD-MARKS-THE-OTHER-WAY"
    assert GB.mark_name([0.9, 1.1]) == "GAPSPREADB-T0-SPREAD-DOES-NOT-MARK" and GB.mark_name(None) == "GAPSPREADB-T0-SPREAD-DOES-NOT-MARK"


def test_the_card_is_not_run_when_control_s3_of_card_add132_passed(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = TA.marked_world(tmp_path, capsys, monkeypatch)
    xb, ab = ns_b(info, x, "order", tmp_path, s3_of_card_a=True)
    assert GB.run_controls(xb, ab, expect=TA.expect_of(W)) is None and "REFUSED" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(xb.out_dir, "GAPSPREADB_CONTROLS.json"))


def test_marked_world_controls_pass_and_both_names_follow(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = TA.marked_world(tmp_path, capsys, monkeypatch)
    xb, ab = ns_b(info, x, "b", tmp_path)
    assert GB.run_name(xb, ab) == 3
    capsys.readouterr()
    rec = GB.run_controls(xb, ab, expect=TA.expect_of(W))
    out = capsys.readouterr().out
    assert rec["all_pass"] is True and out.strip().splitlines()[-1] == "ZZ GAPSPREADB-T0 controls all_pass True", out
    assert "RL" not in out and "long" not in out
    assert GB.run_name(xb, ab) == 0
    rj = json.load(open(os.path.join(xb.out_dir, "GAPSPREADB_RESULT.json"), encoding="utf-8"))
    assert rj["mark_name"] == "GAPSPREADB-T0-SPREAD-MARKS-LONG-READS" and rj["RLp"] < 1.0 and rj["name"].startswith("GAPSPREADB-T0-PLACED-")
    assert rj["card"] == "PREREG_V8F_PROOF_ADD133" and len(rj["families"]) == 4
    b = rj["beside"]
    assert abs(b["long_part_m"]["SP90"] / b["long_part_m"]["PM"] - rj["RLp"]) < 1e-9 and abs(b["long_part_m"]["SP90"] / b["long_part_m"]["A90"] - rj["RL"]) < 1e-9


def test_permuted_arm_keeps_the_factors_and_s5_fails_for_another_k(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = TA.marked_world(tmp_path, capsys, monkeypatch)
    sp = GB.permuted(W, W.s, GB.SEED)
    assert np.allclose(np.sort(sp[W.rows]), np.sort(W.s[W.rows])) and not np.array_equal(sp[W.rows], W.s[W.rows])
    assert GB.ctl_s5(W)["pass"] is True
    c = GB.ctl_s5(W, fault="other-k")
    assert c["pass"] is False and "does not have the k" in c["failures"][0]


def test_s3_fails_when_the_plant_is_not_read(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = TA.marked_world(tmp_path, capsys, monkeypatch)
    monkeypatch.setattr(GS, "planted_spread", lambda w: np.where(w.has, 0.05, np.nan))
    c = GB.ctl_s3(W)
    assert c["pass"] is False and "planted spread is not read" in c["failures"][0]


def test_s4_needs_four_passing_test_outputs(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = TA.marked_world(tmp_path, capsys, monkeypatch)
    xb, ab = ns_b(info, x, "two", tmp_path, tests=FOUR[:2])
    rec = GB.run_controls(xb, ab, expect=TA.expect_of(W))
    assert rec["pass_s4"] is False and rec["all_pass"] is False


def test_unmarked_world_does_not_mark(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = TA.marked_world(tmp_path, capsys, monkeypatch)
    n = len(W.m)
    u = np.abs(np.random.RandomState(9).normal(0.03, 0.01, n)) + 0.005
    fits = TA.write_fits(info["root"], info["base"], u, "nomark")
    x2, a2 = TA.ns(info, fits, "nomark", tmp_path)
    xb, ab = ns_b(info, x2, "nomark_b", tmp_path)
    rec = GB.run_controls(xb, ab, expect=TA.expect_of(W))
    assert rec["all_pass"] is True, capsys.readouterr().out
    assert GB.run_name(xb, ab) == 0
    rj = json.load(open(os.path.join(xb.out_dir, "GAPSPREADB_RESULT.json"), encoding="utf-8"))
    assert rj["mark_name"] == "GAPSPREADB-T0-SPREAD-DOES-NOT-MARK" and rj["bar_name"] != "GAPSPREADB-T0-BAR-MET"
