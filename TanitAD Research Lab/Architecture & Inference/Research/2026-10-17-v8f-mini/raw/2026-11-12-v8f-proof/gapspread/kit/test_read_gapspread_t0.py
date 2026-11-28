"""tests of read_gapspread_t0.py (card PREREG_V8F_PROOF_ADD132). The tiny synthetic world of the kit of card ADD131 is reused (its test module is imported).
Run one file at a time. GAPKIT names the folder of the kit of card ADD131 (default: ../boxfuse10k/gap, or the pod path)."""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("GAPKIT"), os.path.join(os.path.dirname(HERE), "boxfuse10k", "gap"), "/workspace/v8proof/boxfuse10k/kit_gap"):
    if cand and os.path.isfile(os.path.join(cand, "make_joined_gapfiles.py")):
        GAPKIT = cand
        break
else:
    raise RuntimeError("the kit of card ADD131 was not found")
for p in (GAPKIT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_make_joined_gapfiles as TG  # noqa: E402
import read_gapspread_t0 as GS  # noqa: E402
from test_make_joined_gapfiles import fast_render, small_floor  # noqa: E402,F401

MJ, BS = GS.MJ, GS.BS


def write_fits(root, base_path, u, tag):
    """three fit files around the base file: base * (1 - u), base, base * (1 + u); u per row"""
    B = MJ.load(base_path)
    g = np.asarray(B["gap"], np.float64)
    out = []
    for nm, d in zip(GS.FIT_NAMES, (-1.0, 0.0, 1.0)):
        S = {k: np.array(v, copy=True) for k, v in B.items()}
        S["gap"] = g * (1.0 + d * u)
        p = os.path.join(root, "%s_%s.npz" % (nm, tag))
        np.savez(p, **S)
        out.append(p)
    return out


def ns(info, fits, tag, tmp_path, tests=("9 passed in 1.0s", "9 passed in 2.0s")):
    root = info["root"]
    rec = os.path.join(root, "rec106_%s.json" % tag)
    files = {GS.BASE_NAME: {"file_sha256": MJ.sha_file(info["base"])}}
    for nm, p in zip(GS.FIT_NAMES, fits):
        files[nm] = {"file_sha256": MJ.sha_file(p)}
    BS.wt(rec, json.dumps({"card": "PREREG_V8F_PROOF_ADD106", "files": files}))
    touts = []
    for i, t in enumerate(tests):
        q = os.path.join(root, "tests_%s_%d.out" % (tag, i))
        BS.wt(q, "...\n" + t + "\n")
        touts.append(q)
    out = str(tmp_path / ("out_" + tag))
    x = types.SimpleNamespace(base=info["base"], fit=list(fits), rec106=rec, rb_args=info["af"], out_dir=out, tests_out=touts)
    return x, MJ.rb_args(info["af"], out + "_scratch")


def flat_world(tmp_path, capsys, monkeypatch):
    info = TG.world(tmp_path, capsys, monkeypatch)
    n = len(MJ.load(info["base"])["gap"])
    fits = write_fits(info["root"], info["base"], np.full(n, 0.03), "flat")
    x, a = ns(info, fits, "flat", tmp_path)
    return info, x, a, GS.World(x, a)


def marked_world(tmp_path, capsys, monkeypatch):
    info, x0, a0, W0 = flat_world(tmp_path, capsys, monkeypatch)
    ul = 0.02 + 0.05 * np.maximum((W0.m[W0.rows] - W0.lab) / W0.lab, 0.0)
    u = np.random.RandomState(5).choice(ul, size=len(W0.m))  # the other rows carry the same spread values as the label rows, with no mark
    u[W0.rows] = ul
    fits = write_fits(info["root"], info["base"], u, "marked")
    x, a = ns(info, fits, "marked", tmp_path)
    return info, x, a, GS.World(x, a)


def expect_of(W):
    return {"rows": int(len(W.rows)), "clips": W.nc, "e_base": float(np.abs(W.m[W.rows] - W.lab).mean())}


def test_dose_floor_and_second_routine():
    rng = np.random.RandomState(3)
    s = np.abs(rng.normal(0.05, 0.04, 500))
    k = GS.solve_k(s)
    assert abs(GS.factor_of(s, k).mean() - GS.DOSE) < 1e-10 and GS.factor_of(s, k).min() >= GS.FLOOR
    m = rng.uniform(5, 50, 500)
    assert np.abs(GS.sp_gap(m, s, k) - GS.sp_gap_rows(m, s, k)).max() < 1e-12
    kc = GS.solve_k(np.full(500, 0.05))
    assert np.abs(GS.sp_gap(m, np.full(500, 0.05), kc) - GS.DOSE * m).max() < 1e-9
    assert GS.solve_k(np.zeros(10)) is None and GS.solve_k(np.array([0.1] + [0.0] * 99)) is None
    f = np.array([[9.0, 20.0], [10.0, 20.0], [11.0, 23.0]])
    assert np.allclose(GS.spread_of(f, f.mean(axis=0)), [0.2, 3.0 / 21.0])


def test_names_and_the_bar():
    assert GS.name_of(0.85, [0.80, 0.90], [0.90, 1.05]) == ("GAPSPREAD-T0-ALLOCATION-BETTER", "GAPSPREAD-T0-BAR-MET")
    assert GS.name_of(0.95, [0.92, 0.98], [0.80, 0.95]) == ("GAPSPREAD-T0-ALLOCATION-BETTER", "GAPSPREAD-T0-BAR-NOT-MET")
    assert GS.name_of(0.85, [0.80, 0.90], [1.02, 1.10]) == ("GAPSPREAD-T0-ALLOCATION-TRADES", None)
    assert GS.name_of(1.10, [1.02, 1.20], [0.80, 0.95]) == ("GAPSPREAD-T0-ALLOCATION-WORSE", None)
    assert GS.name_of(0.97, [0.92, 1.01], [0.80, 0.95]) == ("GAPSPREAD-T0-ALLOCATION-NOT-SEPARATED", None)
    assert GS.name_of(0.97, None, [0.80, 0.95])[0] == "GAPSPREAD-T0-ALLOCATION-NOT-SEPARATED"


def test_rank_correlation_with_ties():
    a = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    assert GS.rank_avg(a).tolist() == [1.0, 2.5, 2.5, 4.0, 5.0]
    assert abs(GS.rank_corr(a, 3.0 * a + 1.0) - 1.0) < 1e-12 and abs(GS.rank_corr(a, -a) + 1.0) < 1e-12


def test_marked_world_controls_pass_no_value_is_printed_and_the_name_follows(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = marked_world(tmp_path, capsys, monkeypatch)
    assert GS.run_name(x, a) == 3
    capsys.readouterr()
    rec = GS.run_controls(x, a, expect=expect_of(W))
    out = capsys.readouterr().out
    assert rec["all_pass"] is True and out.strip().splitlines()[-1] == "ZZ GAPSPREAD-T0 controls all_pass True", out
    assert "RL" not in out and "long" not in out
    assert GS.run_name(x, a) == 0
    out = capsys.readouterr().out
    rj = json.load(open(os.path.join(x.out_dir, "GAPSPREAD_RESULT.json"), encoding="utf-8"))
    assert rj["name"] == "GAPSPREAD-T0-ALLOCATION-BETTER" and rj["RL"] < 1.0 and ("ZZ GAPSPREAD-T0 NAME " + rj["name"]) in out
    assert rj["card"] == "PREREG_V8F_PROOF_ADD132" and len(rj["families"]) == 4 and rj["beside"]["rank_corr_spread_vs_signed_rel_error"] > 0
    assert abs(rj["beside"]["long_part_m"]["SP90"] / rj["beside"]["long_part_m"]["A90"] - rj["RL"]) < 1e-9


def test_flat_world_reads_a_ratio_of_one(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = flat_world(tmp_path, capsys, monkeypatch)
    rec = GS.run_controls(x, a, expect=expect_of(W))
    assert rec["all_pass"] is True, capsys.readouterr().out
    assert GS.run_name(x, a) == 0
    rj = json.load(open(os.path.join(x.out_dir, "GAPSPREAD_RESULT.json"), encoding="utf-8"))
    assert abs(rj["RL"] - 1.0) < 1e-9 and abs(rj["RS"] - 1.0) < 1e-9 and rj["bar_name"] != "GAPSPREAD-T0-BAR-MET"


def test_s0_fails_for_a_changed_fit_file_and_for_another_hash(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = marked_world(tmp_path, capsys, monkeypatch)
    S = MJ.load(x.fit[2])
    S["gap"] = np.asarray(S["gap"], np.float64) + 0.5
    bad = os.path.join(info["root"], "VGAPL_s4_shifted.npz")
    np.savez(bad, **S)
    x2, a2 = ns(info, [x.fit[0], x.fit[1], bad], "shifted", tmp_path)
    rec = GS.run_controls(x2, a2, expect=expect_of(W))
    assert rec["pass_s0"] is False and "ctl s0 FAIL the mean of the three fits" in capsys.readouterr().out
    rj = json.load(open(x.rec106, encoding="utf-8"))
    rj["files"]["VGAPL_s3"]["file_sha256"] = "0" * 64
    BS.wt(x.rec106, json.dumps(rj))
    rec = GS.run_controls(x, a, expect=expect_of(W))
    assert rec["pass_s0"] is False and rec["all_pass"] is False


def test_s1_fails_against_the_counts_of_the_card_in_a_small_world(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = marked_world(tmp_path, capsys, monkeypatch)
    rec = GS.run_controls(x, a)
    assert rec["pass_s1"] is False and rec["controls"]["s1"]["expected"]["rows"] == 3841


def test_s2_fails_on_a_planted_shift(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = marked_world(tmp_path, capsys, monkeypatch)
    rec = GS.run_controls(x, a, expect=expect_of(W), fault="shift")
    assert rec["pass_s2"] is False and "second routine" in rec["controls"]["s2"]["failures"][0]


def test_s3_fails_when_the_plant_is_not_read_and_when_a_permuted_spread_reads_as_a_mark(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = marked_world(tmp_path, capsys, monkeypatch)
    plant = GS.planted_spread
    monkeypatch.setattr(GS, "planted_spread", lambda w: np.where(w.has, 0.05, np.nan))
    rec = GS.run_controls(x, a, expect=expect_of(W))
    assert rec["pass_s3"] is False and "planted spread is not read" in rec["controls"]["s3"]["failures"][0]
    monkeypatch.setattr(GS, "planted_spread", plant)
    monkeypatch.setattr(GS, "permuted_spread", plant)
    rec = GS.run_controls(x, a, expect=expect_of(W))
    assert rec["pass_s3"] is False and "reads as a mark" in rec["controls"]["s3"]["failures"][0]


def test_s4_needs_two_passing_test_outputs(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W = marked_world(tmp_path, capsys, monkeypatch)
    x1, a1 = ns(info, x.fit, "one", tmp_path, tests=("9 passed in 1.0s",))
    assert GS.run_controls(x1, a1, expect=expect_of(W))["pass_s4"] is False
    x2, a2 = ns(info, x.fit, "red", tmp_path, tests=("9 passed in 1.0s", "1 failed, 8 passed in 1.0s"))
    assert GS.run_controls(x2, a2, expect=expect_of(W))["pass_s4"] is False
