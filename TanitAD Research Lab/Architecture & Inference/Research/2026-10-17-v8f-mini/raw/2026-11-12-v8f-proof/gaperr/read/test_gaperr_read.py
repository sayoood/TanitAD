"""pytest of the reader kit of card ADD111 (GAPERR10K): synthetic data only (tiny worlds of 60 join rows). The model reader of card ADD106 is imported as a sibling module (dev-box copy
cropscale/read/read_cropscale10k.py when the pod path does not exist). One test per control with its planted faults, every name branch on each side of each threshold and the agreement of the two
derivations, the second routines of b0 against hand-computed courses, a4's time rule, the refusal of the names stage, an IDENTITY test against the reader of card ADD110 (same engine calls, same numbers
on one synthetic fixture), the run of the start-parse line of the driver on the start marker as the pod wrote it, the wording scan and the path length."""
import importlib.util
import io
import json
import os
import random
import shutil
import subprocess
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PREV = HERE.parent.parent / "fallattr" / "read"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PREV))
import read_gaperr10k as RE                    # noqa: E402
import read_fallattr10k as RO                  # noqa: E402

RE.bind()
RO.bind()
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
ARMS = RE.ARMS
PRE = "GAPERR10K"


@pytest.fixture()
def world(tmp_path):
    return RE.make_world(str(tmp_path / "w"))


def ev(w):
    a, rec, J, m, wsha, jsha = w
    return RE.compute_controls(a, rec, J, m, wsha, jsha)


# ------------------------------------------------------------------------------------------------ selftest and controls
def test_selftest_ok(capsys):
    assert RE.selftest() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ZZ READ-GAPERR10K SELFTEST OK"


def test_clean_world_all_pass(world):
    c = ev(world)
    assert sorted(c) == sorted(RE.CONTROLS) == ["b0", "b1", "b2", "b3", "b4", "b5"]
    assert all(v["pass"] is True for v in c.values()), {k: v["failures"] for k, v in c.items()}


@pytest.mark.parametrize("fault", RE.CONTROLS)
def test_planted_fault_flips_exactly_that_control(tmp_path, fault):
    c = ev(RE.make_world(str(tmp_path / "f"), fault))
    assert {k: v["pass"] for k, v in c.items()} == {k: (k != fault) for k in RE.CONTROLS}
    assert c[fault]["failures"]


def test_model_constants_agree():
    M = RE.MD
    assert (M.WRAPPER_SHA, M.CHECK_LAST, M.V0_RUN) == (RE.WRAPPER_SHA, RE.CHECK_LAST, RE.V0_RUN)
    assert M.BANKED["RR"] == RE.REFS["RR"] and M.BANKED["RLc"] == RE.REFS["RLc"] and M.RUN["GEc"] == RE.REFS["GEc"]


def test_run_names_and_references():
    assert RE.RUN == {x: "RQ50KF5P30Hb20b" + x for x in ("NR", "BR", "SM", "SL")}
    assert RE.REFS["LA"] == "RQ50KF5P30Hb20aLA" == RO.RUN["LA"]
    assert RE.EXPECTED_LEVELS["LA"] == {"C": 0.0, "F": 0.4877, "U": 0.0839}
    assert RE.EXPECTED_LEVELS["GEc"] == RO.EXPECTED_LEVELS["GEc"] and RE.EXPECTED_LEVELS["RR"] == RO.EXPECTED_LEVELS["RR"]
    assert RE.REFS["GEc"] == RO.REFS["GEc"] and RE.REFS["RLc"] == RO.REFS["RLc"] and RE.REFS["RR"] == RO.REFS["RR"]
    assert RE.rollout_keys() == sorted("%s_s%d" % (x, m) for x in ARMS for m in (0, 1)) and len(RE.rollout_keys()) == 8


# ------------------------------------------------------------------------------------------------ b0
def b0_args(w):
    a, rec, J, m, wsha, jsha = w
    files = {x: RE.gapfile_path(a.gap_dir, x) for x in ARMS}
    bp, lp = rec["base"]["path"], rec["la_file"]["path"]
    return rec, files, J, jsha, RE.JOIN_NAME, RE.load_npz(bp), RE.sha_file(bp), RE.load_npz(lp), RE.sha_file(lp)


def mutate_file(w, arm, fn, rehash=False):
    a, rec = w[0], w[1]
    p = RE.gapfile_path(a.gap_dir, arm)
    z = RE.load_npz(p)
    fn(z)
    np.savez(p, **z)
    if rehash:
        rec["files"][arm]["sha256"] = RE.sha_file(p)


def test_b0_clean_and_values(world):
    r = RE.ctl_b0(*b0_args(world))
    assert r["pass"], r["failures"]
    v = r["values"]
    assert v["n_error_rows"] == 48                                  # 3 clips x 16 label frames + 3 frames at 19, minus the 3 label rows with no base output
    for arm in ARMS:
        assert v["max_abs_diff_" + arm] <= 1e-9
    assert v["max_abs_label_plus_s_plus_f_minus_base"] <= 1e-9


def test_b0_record_top_keys_exact(world):
    rec, *rest = b0_args(world)
    rec["extra"] = 1
    r = RE.ctl_b0(rec, *rest)
    assert not r["pass"] and any("top keys" in x for x in r["failures"])
    del rec["extra"]
    del rec["written_utc"]
    r = RE.ctl_b0(rec, *rest)
    assert not r["pass"] and any("top keys" in x for x in r["failures"])


def test_b0_record_clause_and_flag(world):
    rec, *rest = b0_args(world)
    rec["clauses"]["c2"]["pass"] = False
    r = RE.ctl_b0(rec, *rest)
    assert not r["pass"] and any("c2" in x for x in r["failures"])
    rec["clauses"]["c2"]["pass"] = True
    rec["pass_b0"] = False
    assert not RE.ctl_b0(rec, *rest)["pass"]
    rec["pass_b0"] = True
    rec["clauses"] = {}
    assert not RE.ctl_b0(rec, *rest)["pass"]


def test_b0_files_entries_filled(world):
    rec, *rest = b0_args(world)
    rec["files"]["SM"]["n_diff_rows"] = None
    assert not RE.ctl_b0(rec, *rest)["pass"]
    rec["files"]["SM"]["n_diff_rows"] = 3
    del rec["files"]["SL"]
    assert not RE.ctl_b0(rec, *rest)["pass"]
    rec["files"]["SL"] = {}
    assert not RE.ctl_b0(rec, *rest)["pass"]


def test_b0_t0_block_filled(world):
    rec, *rest = b0_args(world)
    del rec["t0"]["far"]
    assert not RE.ctl_b0(rec, *rest)["pass"]
    rec["t0"]["far"] = {k: 1.0 for k in RE.T0_KEYS}
    del rec["t0"]["far"]["rms_sl"]
    assert not RE.ctl_b0(rec, *rest)["pass"]
    rec["t0"]["far"]["rms_sl"] = 1.0
    assert RE.ctl_b0(rec, *rest)["pass"]


@pytest.mark.parametrize("arm", ARMS)
def test_b0_file_hash_mismatch(world, arm):
    def f(z):
        z["gap"] = z["gap"].copy()
        z["gap"][np.flatnonzero(z["has_out"])[0]] += 1.0
    mutate_file(world, arm, f)
    r = RE.ctl_b0(*b0_args(world))
    assert not r["pass"] and any("sha256" in x for x in r["failures"])


@pytest.mark.parametrize("arm", ARMS)
def test_b0_definition_fault_with_matching_hash(world, arm):
    """a file changed on one row and its record hash updated: only the second routine can catch it"""
    def f(z):
        z["gap"] = z["gap"].copy()
        z["gap"][np.flatnonzero(z["has_out"])[7]] += 0.25
    mutate_file(world, arm, f, rehash=True)
    r = RE.ctl_b0(*b0_args(world))
    assert not r["pass"] and any("definition of card section 2" in x for x in r["failures"])


def test_b0_nr_and_br_swapped_fail(world):
    a, rec = world[0], world[1]
    pn, pb = RE.gapfile_path(a.gap_dir, "NR"), RE.gapfile_path(a.gap_dir, "BR")
    zn, zb = RE.load_npz(pn), RE.load_npz(pb)
    np.savez(pn, **zb)
    np.savez(pb, **zn)
    rec["files"]["NR"]["sha256"], rec["files"]["BR"]["sha256"] = RE.sha_file(pn), RE.sha_file(pb)
    assert not RE.ctl_b0(*b0_args(world))["pass"]


def test_b0_output_flag_fault_in_a_smoothing_file(world):
    def f(z):
        idx = np.flatnonzero(~z["has_out"])[0]
        z["has_out"] = z["has_out"].copy()
        z["has_out"][idx] = True
        z["gap"] = z["gap"].copy()
        z["gap"][idx] = 9.0
    mutate_file(world, "SM", f, rehash=True)
    r = RE.ctl_b0(*b0_args(world))
    assert not r["pass"] and any("SM" in x for x in r["failures"])


def test_b0_raw_field_kept_on_rows_with_no_output(world):
    def f(z):
        z["gap"] = z["gap"].copy()
        z["gap"][np.flatnonzero(~z["has_out"])[0]] = 5.0
    mutate_file(world, "SL", f, rehash=True)
    r = RE.ctl_b0(*b0_args(world))
    assert not r["pass"] and any("keep its base gap field" in x for x in r["failures"])


def test_b0_nr_br_against_la_on_rows_that_are_not_error_rows(world):
    """the file LA changed on a row that is not an error row (frame 18 of clip 0 has no label): NR and BR, correct by the definition, now differ from LA there"""
    lp = world[1]["la_file"]["path"]
    z = RE.load_npz(lp)
    z["gap"] = z["gap"].copy()
    z["gap"][18] += 2.0
    np.savez(lp, **z)
    world[1]["la_file"]["sha256"] = RE.sha_file(lp)
    r = RE.ctl_b0(*b0_args(world))
    assert not r["pass"] and any("file LA" in x for x in r["failures"])


def test_b0_la_file_clauses(world):
    rec, files, J, jsha, jname, base, bsha, la, lsha = b0_args(world)
    assert not RE.ctl_b0(rec, files, J, jsha, jname, base, bsha, la, "0" * 64)["pass"]
    assert not RE.ctl_b0(rec, files, J, jsha, jname, base, bsha, None, None)["pass"]


def test_b0_join_sha_and_name(world):
    rec, files, J, jsha, jname, *rest = b0_args(world)
    assert not RE.ctl_b0(rec, files, J, "0" * 64, jname, *rest)["pass"]
    assert not RE.ctl_b0(rec, files, J, jsha, "other.npz", *rest)["pass"]


def test_b0_base_clauses(world):
    rec, files, J, jsha, jname, base, bsha, la, lsha = b0_args(world)
    assert not RE.ctl_b0(rec, files, J, jsha, jname, base, "0" * 64, la, lsha)["pass"]
    assert not RE.ctl_b0(rec, files, J, jsha, jname, None, None, la, lsha)["pass"]
    rec["base"]["named_in_record_add106"] = False
    assert not RE.ctl_b0(rec, files, J, jsha, jname, base, bsha, la, lsha)["pass"]


def test_b0_duplicate_key_fails(world):
    def f(z):
        z["frame"] = z["frame"].copy()
        z["frame"][1] = z["frame"][0]
        z["clip_index"] = z["clip_index"].copy()
        z["clip_index"][1] = z["clip_index"][0]
    mutate_file(world, "NR", f)
    assert not RE.ctl_b0(*b0_args(world))["pass"]


def test_b0_row_count_in_record(world):
    rec, *rest = b0_args(world)
    rec["files"]["BR"]["n_rows"] = 59
    r = RE.ctl_b0(rec, *rest)
    assert not r["pass"] and any("rows" in x for x in r["failures"])


def test_label_row_rule():
    J = {"clip": np.array(["c"]), "off": np.array([0]), "n": np.array([6]), "lead_gap": np.array([10.0, 60.0, 60.5, -1.0, np.nan, 30.0]), "lead_valid": np.array([1, 1, 1, 1, 1, 0])}
    _, lab = RE.label_rows(J)
    assert lab.tolist() == [True, True, False, False, False, False]


def test_slow_and_fast_part_by_hand():
    J = {"n": np.array([13, 4]), "off": np.array([0, 13])}
    err = np.array([True] * 13 + [True, False, True, True])
    e = np.where(err, np.arange(17.0), np.nan)
    s, f = RE.slow_fast(J, err, e)
    assert s[0] == pytest.approx(2.5) and s[6] == pytest.approx(6.0) and s[12] == pytest.approx(9.5) and s[3] == pytest.approx(np.mean(np.arange(0, 9.0)))
    assert s[13] == pytest.approx(13.0) and s[15] == pytest.approx(15.5) and s[16] == pytest.approx(15.5)
    assert np.isnan(s[14]) and np.isnan(f[14])
    assert f[0] == pytest.approx(-2.5) and f[16] == pytest.approx(0.5)
    assert np.allclose((s + f)[err], e[err])


def test_error_runs_do_not_cross_a_clip_border():
    J = {"n": np.array([3, 3]), "off": np.array([0, 3])}
    err = np.ones(6, bool)
    e = np.array([0.0, 0.0, 0.0, 6.0, 6.0, 6.0])
    s, f = RE.slow_fast(J, err, e)
    assert s.tolist() == [0.0, 0.0, 0.0, 6.0, 6.0, 6.0] and np.allclose(f, 0.0)


def test_compose_attr_rules_by_hand():
    J = {"clip": np.array(["c"]), "off": np.array([0]), "n": np.array([4]), "lead_gap": np.array([10.0, 12.0, 14.0, 99.0]), "lead_valid": np.array([1, 1, 1, 1])}
    bhas = np.array([True, True, False, True])
    bgap = np.array([11.0, 14.0, np.nan, 50.0])
    h, g = RE.compose_attr(J, "NR", bhas, bgap)
    assert h.tolist() == [True, True, True, True]
    assert g[0] == pytest.approx(10.0 + 1.5) and g[1] == pytest.approx(12.0 + 1.5)        # e = 1, 2: s = 1.5 on both rows
    assert g[2] == 14.0 and g[3] == 50.0                                                      # label row with no output takes the label gap; row 3 has no label
    h, g = RE.compose_attr(J, "BR", bhas, bgap)
    assert g[0] == pytest.approx(10.0 - 0.5) and g[1] == pytest.approx(12.0 + 0.5)


def test_trailing_run_by_hand():
    has = np.ones(8, bool)
    gp = np.array([10.0, 11.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
    assert RE.trailing_run_start(has, gp, 7, 0) == 3                                         # at most 5 frames
    assert RE.trailing_run_start(has, gp, 3, 0) == 2                                         # a step of 9 m ends the run
    assert RE.trailing_run_start(has, gp, 1, 0) == 0
    assert RE.trailing_run_start(has, np.array([10.0, 13.0]), 1, 0) == 0                     # a step of exactly 3.0 m stays in the run
    assert RE.trailing_run_start(has, np.array([10.0, 13.0000001]), 1, 0) == 1
    h2 = has.copy()
    h2[5] = False
    assert RE.trailing_run_start(h2, gp, 7, 0) == 6                                          # a frame with no base output ends it
    assert RE.trailing_run_start(has, gp, 4, 3) == 3                                         # never before the first row of the clip


def test_smoothing_forms_by_hand():
    J = {"n": np.array([8]), "off": np.array([0])}
    has = np.ones(8, bool)
    gp = np.array([10.0, 11.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
    sm, sl = RE.smooth_arms(J, has, gp)
    assert sm[0] == 10.0 and sl[0] == 10.0                                                   # one frame: the base gap itself
    assert sm[1] == pytest.approx(10.5) and sl[1] == 11.0                                    # two frames: median of two, the base gap itself for SL
    assert sm[2] == 20.0 and sl[2] == 20.0
    assert sm[3] == pytest.approx(20.5) and sl[3] == 21.0                                    # run 20, 21
    assert sm[4] == 21.0 and sl[4] == pytest.approx(22.0)                                    # run 20, 21, 22
    assert sm[7] == 23.0 and sl[7] == pytest.approx(25.0)                                    # run 21 to 25
    h2 = has.copy()
    h2[4] = False
    sm, sl = RE.smooth_arms(J, h2, gp)
    assert np.isnan(sm[4]) and np.isnan(sl[4]) and sm[5] == 23.0 and sl[5] == 23.0


def test_course_checks_clean_and_planted_faults(monkeypatch):
    assert RE.course_checks() == []
    real = RE.smooth_arms

    def lag3(J, h, g):
        sm, sl = real(J, h, g)
        return np.concatenate([[np.nan] * 3, sm[:-3]]), sl
    monkeypatch.setattr(RE, "smooth_arms", lag3)
    assert RE.course_checks()
    monkeypatch.setattr(RE, "smooth_arms", lambda J, h, g: (real(J, h, g)[0], real(J, h, g)[0]))
    assert any("SL" in x for x in RE.course_checks())
    monkeypatch.setattr(RE, "smooth_arms", lambda J, h, g: (real(J, h, g)[0] + 0.01, real(J, h, g)[1]))
    assert any("constant" in x for x in RE.course_checks())


def test_label_free_detects_a_label_dependent_smoother(world, monkeypatch):
    a, rec, J = world[0], world[1], world[2]
    bhas, bgap, _, _ = RE.file_on_join(J, RE.load_npz(rec["base"]["path"]))
    assert RE.label_free(J, bhas, bgap)
    monkeypatch.setattr(RE, "smooth_arms", lambda J_, h, g: (np.asarray(J_.get("lead_gap", g), float).copy(), np.asarray(J_.get("lead_gap", g), float).copy()))
    assert not RE.label_free(J, bhas, bgap)
    r = RE.ctl_b0(*b0_args(world))
    assert not r["pass"] and any("label fields" in x for x in r["failures"])


# ------------------------------------------------------------------------------------------------ b1 to b5
def test_b1_clauses():
    ok = {"%s_s%d" % (x, m): RE.WRAPPER_SHA + "0" * 48 for x in ARMS for m in (0, 1)}
    chk = "x\n" + RE.CHECK_LAST + "\n"
    assert RE.ctl_b1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, ok)["pass"]
    assert not RE.ctl_b1("0" * 16, chk, RE.WRAPPER_SHA, ok)["pass"]
    assert not RE.ctl_b1(RE.WRAPPER_SHA, "x\nZZ CHECK-WRAPPER-V FAIL\n", RE.WRAPPER_SHA, ok)["pass"]
    assert not RE.ctl_b1(RE.WRAPPER_SHA, chk, "abc", ok)["pass"]
    bad = dict(ok)
    bad["NR_s0"] = "f" * 64
    assert not RE.ctl_b1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, bad)["pass"]
    bad = dict(ok)
    bad["NR_s0"] = None
    assert not RE.ctl_b1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, bad)["pass"]
    assert not RE.ctl_b1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, {k: v for k, v in list(ok.items())[:7]})["pass"]


def rewrite_logs(w, run, m, fn):
    a = w[0]
    d = os.path.join(a.pull, "I10%s_s%d" % (run, m), "ticks")
    for nm in ("rule_log_I10%s_s%d.npz", "rule_log_mid_I10%s_s%d.npz"):
        p = os.path.join(d, nm % (run, m))
        z = RE.load_npz(p)
        fn(z)
        np.savez(p, **z)


def test_b2_clean_has_shares_and_no_label_share(world):
    a, rec, J, *_ = world
    r = RE.ctl_b2(RE.b2_from_files(a, rec, J))
    assert r["pass"], r["failures"]
    sh = r["values"]["rollout_shares"]
    assert sorted(sh) == RE.rollout_keys()
    assert sh["NR_s0"]["share_of_lead_steps_with_nonzero_closing"] == pytest.approx(0.5)
    assert sh["SM_s1"]["share_of_rule_steps_with_a_lead"] is not None
    assert all("share_of_lead_steps_with_label_gap" not in v for v in sh.values())


def test_b2_closing_zero_fails(world):
    a, rec, J, *_ = world
    rewrite_logs(world, RE.RUN["SM"], 1, lambda z: z.update(c_src=np.zeros_like(z["c_src"])))
    r = RE.ctl_b2(RE.b2_from_files(a, rec, J))
    assert not r["pass"] and any("closing" in x for x in r["failures"])


def test_b2_no_output_step_must_equal_v0(world):
    a, rec, J, *_ = world
    def f(z):
        z["g_src"] = z["g_src"].copy()
        z["g_src"][np.flatnonzero((z["j"] == 3) & (z["ci"] == 0))] += 1.0      # clip 0 frame 3 is a step with no output in the base file: the V0 value is expected
    rewrite_logs(world, RE.RUN["SL"], 0, f)
    assert not RE.ctl_b2(RE.b2_from_files(a, rec, J))["pass"]


def test_b2_meta_names_file_and_switch(world):
    a, rec, J, *_ = world
    rewrite_logs(world, RE.RUN["BR"], 0, lambda z: z.update(vis_gap_file_sha256=np.array("0" * 64)))
    assert not RE.ctl_b2(RE.b2_from_files(a, rec, J))["pass"]
    w2 = RE.make_world(str(Path(a.pull).parent.parent / "w2"))
    rewrite_logs(w2, RE.RUN["BR"], 1, lambda z: z.update(vis_closing=np.array(0)))
    assert not RE.ctl_b2(RE.b2_from_files(w2[0], w2[1], w2[2]))["pass"]


def test_b2_missing_rollout_fails(world):
    a, rec, J, *_ = world
    per = RE.b2_from_files(a, rec, J)
    per.pop("SL_s1")
    assert not RE.ctl_b2(per)["pass"]
    shutil.rmtree(os.path.join(a.pull, "I10%s_s0" % RE.RUN["NR"]))
    assert not RE.ctl_b2(RE.b2_from_files(a, rec, J))["pass"]


def test_b3_each_value_including_la():
    lv = {k: {c: v for c, v in d.items()} for k, d in RE.EXPECTED_LEVELS.items()}
    assert sorted(lv) == ["GEc", "LA", "RLc", "RR"] and RE.ctl_b3(lv)["pass"]
    lv["RR"]["U"] += 0.00005
    assert RE.ctl_b3(lv)["pass"]                                       # inside 1e-4
    lv["RR"]["U"] += 0.0001
    assert not RE.ctl_b3(lv)["pass"]
    lv = {k: dict(v) for k, v in RE.EXPECTED_LEVELS.items()}
    lv["RLc"]["F"] = 0.4908 + 0.0002
    assert not RE.ctl_b3(lv)["pass"]
    lv = {k: dict(v) for k, v in RE.EXPECTED_LEVELS.items()}
    lv["LA"]["F"] = 0.4877 + 0.0002
    r = RE.ctl_b3(lv)
    assert not r["pass"] and any("LA F" in x for x in r["failures"])
    lv = {k: dict(v) for k, v in RE.EXPECTED_LEVELS.items()}
    lv["LA"]["U"] = 0.0839 - 0.00009
    assert RE.ctl_b3(lv)["pass"]
    assert not RE.ctl_b3({k: v for k, v in RE.EXPECTED_LEVELS.items() if k != "LA"})["pass"]
    lv = {k: dict(v) for k, v in RE.EXPECTED_LEVELS.items()}
    lv["GEc"]["C"] = float("nan")
    assert not RE.ctl_b3(lv)["pass"]


def test_b4_time_rule():
    g, r = "...\n== 12 passed in 1.0s ==\n", "...\n== 30 passed in 2.0s ==\n\n"
    assert RE.ctl_b4(g, r, 100, 200, 300)["pass"]
    assert RE.ctl_b4(g, r, 200, 100, 300)["pass"]
    assert not RE.ctl_b4(g, r, 100, 300, 300)["pass"]                  # equal to the start: not older
    assert not RE.ctl_b4(g, r, 100, 301, 300)["pass"]
    assert not RE.ctl_b4(g, r, 301, 100, 300)["pass"]
    assert not RE.ctl_b4(g, r, None, 100, 300)["pass"]
    assert not RE.ctl_b4(g, r, 100, 200, None)["pass"]
    assert RE.ctl_b4(g, r, "100", "200", "1970-01-01T00:05:00Z")["pass"]
    assert not RE.ctl_b4(g, r, "100", "400", "1970-01-01T00:05:00Z")["pass"]
    assert RE.ctl_b4(g, r, 1795335000, 1795335800, "1795335883")["pass"]          # the first token of the start marker as the pod wrote it


def test_b4_text_rule():
    ok = "== 5 passed in 1s =="
    assert not RE.ctl_b4("== 2 failed, 5 passed in 1s ==", ok, 1, 1, 9)["pass"]
    assert not RE.ctl_b4(ok, "== 5 passed, 1 error in 1s ==", 1, 1, 9)["pass"]
    assert not RE.ctl_b4("", ok, 1, 1, 9)["pass"]
    assert not RE.ctl_b4(ok, "== no tests ran in 0.1s ==", 1, 1, 9)["pass"]
    assert not RE.ctl_b4(ok, "== 0 passed in 0.1s ==", 1, 1, 9)["pass"]
    assert not RE.ctl_b4("== 5 passed ==\nnote: something", ok, 1, 1, 9)["pass"]          # the last non-empty line decides


def test_b5_parts():
    assert RE.ctl_b5(True, True, True)["pass"]
    assert not RE.ctl_b5(False, True, True)["pass"]
    assert not RE.ctl_b5(True, False, True)["pass"]
    assert not RE.ctl_b5(True, True, None)["pass"]


# ------------------------------------------------------------------------------------------------ names
LOW, HIGH, MID0 = [-0.05, -0.01], [0.01, 0.05], [-0.02, 0.02]
EDGE_LO, EDGE_HI = [-0.02, 0.0], [0.0, 0.02]


@pytest.mark.parametrize("iv,word", [(LOW, "LOWERS"), (HIGH, "RAISES"), (MID0, "NOT-SEPARATED"), (EDGE_LO, "NOT-SEPARATED"), (EDGE_HI, "NOT-SEPARATED"), ([-0.03, -0.0001], "LOWERS"),
                                     ([0.0001, 0.03], "RAISES")])
def test_per_arm_measure_names(iv, word):
    for x in ARMS:
        for m, label in RE.MEASURES:
            kw = {"c": {x: iv}} if m == "C" else ({"u": {x: iv}} if m == "U" else {})
            pl = RE.mk_payload({x: iv}, **kw) if m == "F" else RE.mk_payload(**kw)
            nm = RE.all_names(pl)["%s_%s" % (x, label)]
            want = ("%s-%s-%s-%s" % (PRE, x, word, label)) if word != "NOT-SEPARATED" else ("%s-%s-%s-NOT-SEPARATED" % (PRE, x, label))
            assert nm == want


@pytest.mark.parametrize("nr,br,part", [(LOW, MID0, "-FAST-PART-CARRIES"), (MID0, LOW, "-SLOW-PART-CARRIES"), (LOW, LOW, "-BOTH-PARTS-CARRY"), (MID0, MID0, "-PARTS-NOT-SEPARATED"),
                                         (HIGH, MID0, "-PARTS-NOT-SEPARATED"), (HIGH, HIGH, "-PARTS-NOT-SEPARATED"), (EDGE_LO, LOW, "-SLOW-PART-CARRIES"), (LOW, EDGE_LO, "-FAST-PART-CARRIES"),
                                         (LOW, HIGH, "-FAST-PART-CARRIES"), (HIGH, LOW, "-SLOW-PART-CARRIES"), (EDGE_LO, EDGE_LO, "-PARTS-NOT-SEPARATED")])
def test_parts_branches(nr, br, part):
    pl = RE.mk_payload({"NR": nr, "BR": br})
    assert RE.all_names(pl)["PARTS"] == PRE + part
    assert RE.derive_all(pl)["PARTS"] == PRE + part


def test_parts_read_only_from_nr_and_br_falling_behind():
    pl = RE.mk_payload({"SM": LOW, "SL": LOW}, c={"NR": LOW, "BR": LOW}, u={"NR": LOW})
    assert RE.all_names(pl)["PARTS"] == PRE + "-PARTS-NOT-SEPARATED"


def smooth_payload(x, f=MID0, c=MID0, u=MID0):
    return RE.mk_payload({x: f}, c={x: c}, u={x: u})


@pytest.mark.parametrize("x", RE.SMOOTH_ARMS)
@pytest.mark.parametrize("f,c,u,tail", [
    (MID0, MID0, MID0, "-NO-GAIN"), (LOW, MID0, MID0, "-RECOVERS"), (HIGH, MID0, MID0, "-NO-GAIN"), (EDGE_LO, MID0, MID0, "-NO-GAIN"),
    (LOW, HIGH, MID0, "-PAYS-SAFETY"), (LOW, MID0, HIGH, "-PAYS-SAFETY"), (MID0, HIGH, MID0, "-PAYS-SAFETY"), (MID0, MID0, HIGH, "-PAYS-SAFETY"), (LOW, HIGH, HIGH, "-PAYS-SAFETY"),
    (LOW, LOW, LOW, "-RECOVERS"), (MID0, LOW, MID0, "-NO-GAIN"), (MID0, MID0, LOW, "-NO-GAIN"), (LOW, EDGE_HI, MID0, "-RECOVERS"), (LOW, MID0, EDGE_HI, "-RECOVERS"),
    (MID0, [0.0001, 0.02], MID0, "-PAYS-SAFETY")])
def test_smoothing_arm_names(x, f, c, u, tail):
    pl = smooth_payload(x, f, c, u)
    assert RE.all_names(pl)["SMOOTH_" + x] == "%s-%s%s" % (PRE, x, tail)
    assert RE.derive_all(pl)["SMOOTH_" + x] == "%s-%s%s" % (PRE, x, tail)


def test_smoothing_names_are_only_for_sm_and_sl():
    names = RE.all_names(RE.mk_payload())
    assert sorted(k for k in names if k.startswith("SMOOTH_")) == ["SMOOTH_SL", "SMOOTH_SM"]
    flat = RE.flat_names(names)
    assert PRE + "-NR-PAYS-SAFETY" not in flat and len(flat) == 4 * 3 + 1 + 2


def test_two_derivations_agree_on_random_intervals():
    rnd = random.Random(11)
    pool = [LOW, HIGH, MID0, EDGE_LO, EDGE_HI, [-0.03, -0.0001], [0.0001, 0.03]]
    for _ in range(300):
        pick = lambda: {x: rnd.choice(pool) for x in ARMS}              # noqa: E731
        pl = RE.mk_payload(pick(), pick(), pick(), pick())
        assert RE.names_of(pl) == RE.derive_all(pl)
        assert RE.all_names(pl) == RE.names_of(pl)


def test_disagreement_is_refused(monkeypatch):
    pl = RE.mk_payload({"NR": LOW})
    monkeypatch.setattr(RE, "derive_all", lambda p: dict(RE.names_of(p), PARTS=PRE + "-PARTS-NOT-SEPARATED"))
    with pytest.raises(SystemExit):
        RE.all_names(pl)


# ------------------------------------------------------------------------------------------------ stages
def full_payload():
    pl = RE.mk_payload({"NR": LOW, "SM": LOW}, {"NR": MID0})
    lv = {k: {"C": 0.01, "U": 0.05, "F": 0.5} for k in list(ARMS) + ["GEc", "RR", "RLc", "LA"]}
    pl.update({"levels": lv, "levels_per_seed": {k: {c: [0.1, 0.2] for c in "CUF"} for k in lv}, "families": {}, "strategic": {"pairs": {}},
               "gapfile_counts_from_record": {x: {"n_rows": 60, "n_diff_rows": 5, "n_label_value_rows": 2} for x in ARMS},
               "t0_from_record": {rg: {k: 1.0 for k in RE.T0_KEYS} for rg in RE.T0_RANGES}, "control_values": {"b2": {"rollout_shares": {"NR_s0": {"n_rows": 3}}}}})
    return pl


def controls_file(path, all_pass=True, drop=None, flip=None):
    ctl = {k: {"pass": True, "failures": []} for k in RE.CONTROLS}
    if drop:
        ctl.pop(drop)
    if flip:
        ctl[flip]["pass"] = False
    res = {"card": RE.CARD, "controls": ctl, "all_pass": all_pass, "payload": full_payload()}
    Path(path).write_text(json.dumps(res), encoding="utf-8")


def test_names_stage_refuses_each_missing_or_failed_control(tmp_path, capsys):
    for k in RE.CONTROLS:
        for kind in ("flip", "drop"):
            cf = tmp_path / ("c_%s_%s.json" % (k, kind))
            controls_file(cf, all_pass=(kind == "drop"), **{kind: k})
            out = tmp_path / ("r_%s_%s.json" % (k, kind))
            rc = RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)])
            assert rc == 3 and not out.exists()
    capsys.readouterr()


def test_names_stage_refuses_a_controls_file_of_another_card(tmp_path, capsys):
    cf, out = tmp_path / "c.json", tmp_path / "r.json"
    controls_file(cf)
    d = json.loads(cf.read_text(encoding="utf-8"))
    d["card"] = "PREREG_V8F_PROOF_ADD110"
    cf.write_text(json.dumps(d), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 3 and not out.exists()
    capsys.readouterr()


def test_names_stage_writes_result_and_log(tmp_path, capsys):
    cf, out = tmp_path / "c.json", tmp_path / "r.json"
    controls_file(cf)
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_sha256"] == RE.sha_file(str(cf))
    assert res["names"]["PARTS"] == PRE + "-FAST-PART-CARRIES" and res["names"]["SMOOTH_SM"] == PRE + "-SM-RECOVERS" and res["names"]["SMOOTH_SL"] == PRE + "-SL-NO-GAIN"
    log = (tmp_path / "r_names.log").read_text(encoding="utf-8")
    assert log.strip() == printed.strip()
    lines = log.splitlines()
    assert lines[0].startswith("TIER T1-kin") and lines[-1] == "ZZ READ-GAPERR10K names done"
    assert any(x.startswith(PRE + "-NR-LOWERS-FALLING-BEHIND") for x in lines) and PRE + "-FAST-PART-CARRIES" in lines and PRE + "-SM-RECOVERS" in lines
    assert any(x.startswith("level GEc pooled") for x in lines) and any(x.startswith("level LA pooled") for x in lines)
    assert any(x.startswith("SL-VS-LA:") for x in lines) and any(x.startswith("BR-VS-RR:") for x in lines) and any(x.startswith("families SM-VS-RLc:") for x in lines)
    assert any(x.startswith("gap file SL: ") and "n_diff_rows" in x for x in lines)
    assert [x for x in lines if x.startswith("t0 ")] == ["t0 %s (T0 on the evaluation frames, recorded frames, not driving performance): %s" % (rg, json.dumps(full_payload()["t0_from_record"][rg], sort_keys=True))
                                                          for rg in RE.T0_RANGES]
    assert any(x.startswith("rollout NR_s0:") for x in lines)
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 2      # a second read into the same file is refused


def test_assemble_controls_withholds_on_fail():
    ok = {k: {"pass": True, "failures": [], "values": {"x": 1}} for k in RE.CONTROLS}
    r = RE.assemble_controls(ok, {"levels": {}}, {"h": 1})
    assert r["all_pass"] and "payload" in r and "withheld" not in r and r["payload"]["control_values"]["b0"] == {"x": 1}
    ok["b3"]["pass"] = False
    r = RE.assemble_controls(ok, {"levels": {}}, {"h": 1})
    assert not r["all_pass"] and "withheld" in r and "payload" not in r
    assert sorted(r["controls"]) == sorted(RE.CONTROLS) and "values" not in r["controls"]["b0"]


def test_stdout_of_controls_has_only_the_allowed_lines(tmp_path, monkeypatch, capsys):
    res = RE.assemble_controls({k: {"pass": k != "b4", "failures": ["x"]} for k in RE.CONTROLS}, {}, {})
    monkeypatch.setattr(RE, "run_controls", lambda a: res)
    out = tmp_path / "c.json"
    assert RE.main(["--stage", "controls", "--out", str(out)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["b0 PASS", "b1 PASS", "b2 PASS", "b3 PASS", "b4 FAIL", "b5 PASS", "ZZ READ-GAPERR10K controls all_pass False"]
    assert "withheld" in json.loads(out.read_text(encoding="utf-8"))


def test_run_controls_payload_carries_counts_and_t0_from_the_record(tmp_path, monkeypatch):
    """run_controls with a fake engine: the payload has gapfile_counts_from_record and t0_from_record exactly as the record writes them"""
    w = RE.make_world(str(tmp_path / "w"))
    a, rec, J, m, wsha, jsha = w
    eng = types.SimpleNamespace(**{"levels": m["levels"], "levels_per_seed": {}, "pairs": {}, "families": {}, "strategic": {"pairs": {}}, "grid_all_pass": True, "strategic_a": True, "strategic_b": True})
    monkeypatch.setattr(RE, "measure", lambda a_, E, log: dict(eng.__dict__))
    monkeypatch.setattr(RE.R, "engine", lambda a_: None, raising=False)
    a.out = str(tmp_path / "ctl.json")
    a.gap_record = os.path.join(a.gap_dir, "GAPERR_GAPFILES_RECORD.json")
    a.wrapper_file = ""
    res = RE.run_controls(a)
    pl = res.get("payload") or res["withheld"]
    assert pl["t0_from_record"] == rec["t0"]
    assert pl["gapfile_counts_from_record"] == {x: {k: rec["files"][x][k] for k in ("n_rows", "n_diff_rows", "n_label_value_rows")} for x in ARMS}
    assert res["controls"]["b0"]["pass"], res["controls"]["b0"]["failures"]


# ------------------------------------------------------------------------------------------------ identity with the reader of card ADD110
def h01(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) / 4294967296.0


class FakeS:
    ARCS = (30, 50, 80)

    @staticmethod
    def pair_read(cells, A, B, eid, ticks=(2,)):
        return {"tick2": {"pos": [{"mean": h01(A, B, arc, "m"), "lo": h01(A, B, arc, "l"), "hi": h01(A, B, arc, "h"), "verdict": "v%d" % int(10 * h01(A, B, arc, "v")), "n_per_seed": 3}
                                  for arc in FakeS.ARCS]}}


def fake_engine():
    R = types.SimpleNamespace()
    R.ALL_RUNS = {}
    R.NAMED = ("LONGITUDINAL", "LATERAL")
    R.jl = lambda p: {}
    R.grid_keys = lambda g, clips: list(clips)

    def read_grid(a, E, p, keys, run_list, spread, pairs_def, x, log):
        lev = {r: {c: np.array([h01(r, c, 0), h01(r, c, 1)]) for c in "CUF"} for r in run_list}
        pairs = {nm: {"headway": {m: {"mean": h01(A, B, m, "m"), "interval": [h01(A, B, m, "lo") - 0.5, h01(A, B, m, "hi") - 0.2]} for m in "CUF"}, "family": {f_: h01(A, B, f_) for f_ in R.NAMED}}
                 for nm, (A, B) in pairs_def.items()}
        return {"levels": lev, "pairs": pairs, "controls": {"ALL_PASS": True}}
    R.read_grid = read_grid
    R.ucf = lambda hw: {m: dict(v) for m, v in hw.items()}
    R.cells_of = lambda fam: fam
    R.reading = lambda cells, f_: ("reading", f_, cells[f_])
    R.family_of = lambda vs: "|".join(vs)
    R.strategic_grid = lambda a, keys, runs, p_of, log, x: ({"stage_A_pass": True, "stage_B_pass": True, "controls": {"s1": 1, "s3": {"pass": True}}}, {}, {}, FakeS)
    return R


def run_measure(mod, monkeypatch):
    R = fake_engine()
    monkeypatch.setattr(mod, "R", R)
    monkeypatch.setattr(mod, "EC", types.SimpleNamespace(SETTINGS={"AGE": {"runs": {"RR": "RQ50KF5P30Hb20"}}}))
    H = types.SimpleNamespace(load_join=lambda: {"clip": np.array(["c0", "c1"])})
    a = types.SimpleNamespace(grid_dir="x")
    return mod.measure(a, (H, None, None), lambda s: None), R


def test_identity_measure_same_engine_calls_same_numbers(monkeypatch):
    mn, Rn = run_measure(RE, monkeypatch)
    mo, Ro = run_measure(RO, monkeypatch)
    for k in ("GEc", "RLc", "RR"):
        assert mn["levels"][k] == mo["levels"][k] and mn["levels_per_seed"][k] == mo["levels_per_seed"][k]
    assert mn["levels"]["LA"] == mo["levels"]["LA"] and mn["levels_per_seed"]["LA"] == mo["levels_per_seed"]["LA"]          # the arm LA of card ADD110 is the reference LA here
    assert sorted(mn["levels"]) == sorted(list(ARMS) + ["GEc", "RLc", "RR", "LA"])
    for k in ("GEc", "RLc", "RR", "LA"):
        assert Rn.ALL_RUNS[RE.REFS[k]] == Ro.ALL_RUNS[RE.REFS[k]]
    for mod, mm in ((RE, mn), (RO, mo)):
        refs = ("GEc", "RLc", "RR", "LA") if mod is RE else ("GEc", "RLc", "RR")
        pdef = {"%s-VS-%s" % (x, ref): (mod.RUN[x], mod.REFS[ref]) for x in mod.ARMS for ref in refs}
        want = fake_engine().read_grid(None, None, "30", [], [], "R0Hb20", pdef, {}, None)["pairs"]
        assert sorted(mm["pairs"]) == sorted(pdef)
        for nm in pdef:
            assert mm["pairs"][nm] == {m: dict(v) for m, v in want[nm]["headway"].items()}
            assert mm["families"][nm] == {f_: ("reading", f_, want[nm]["family"][f_]) for f_ in ("LONGITUDINAL", "LATERAL")}
        st = mm["strategic"]["pairs"]
        assert sorted(st) == sorted("%s-VS-%s" % (x, ref) for x in mod.ARMS for ref in ("GEc", "RLc"))
        for nm, v in st.items():
            x, _, ref = nm.split("-")
            exp = FakeS.pair_read(None, mod.RUN[x], mod.REFS[ref], None)["tick2"]["pos"]
            assert v["A"] == mod.RUN[x] and v["B"] == mod.REFS[ref]
            assert [(c["mean"], c["lo"], c["hi"]) for c in v["pos_30_50_80m_tick_t0_plus_20"]] == [(b["mean"], b["lo"], b["hi"]) for b in exp]
    assert mn["grid_all_pass"] is True and mn["strategic_a"] is True and mn["strategic_b"] is True


def test_identity_same_model_reader_and_bootstrap():
    assert RE.MODEL_PATH == RO.MODEL_PATH
    bt = getattr(RE.MD, "boot_ratio", None)
    if bt is None:
        pytest.skip("no boot_ratio in the model reader")
    num, den = np.arange(1.0, 41.0), np.arange(41.0, 81.0)
    assert repr(RE.MD.boot_ratio(num, den, 200, 7)) == repr(RO.MD.boot_ratio(num, den, 200, 7))


def test_identity_shared_routines_equal_outputs(tmp_path):
    wn = RE.make_world(str(tmp_path / "n"))
    J = wn[2]
    z = RE.load_npz(wn[1]["base"]["path"])
    for fn in ("label_rows", "file_on_join"):
        a_, b_ = getattr(RE, fn), getattr(RO, fn)
        ra, rb = (a_(J), b_(J)) if fn == "label_rows" else (a_(J, z), b_(J, z))
        for u, v in zip(ra, rb):
            assert np.array_equal(u, v, equal_nan=True) if isinstance(u, np.ndarray) else u == v
    assert RE.sha_file(wn[1]["base"]["path"]) == RO.sha_file(wn[1]["base"]["path"])
    for x in (None, "", 5, "7", "1970-01-01T00:05:00Z", "1970-01-01T00:05:00", "2026-11-22T08:24:43Z pid 1", "abc"):
        assert RE.to_epoch(x) == RO.to_epoch(x)
    for t in ("", "a\n\nb\n\n", "== 5 passed =="):
        assert RE.last_line(t) == RO.last_line(t)
    ok = {"k%d" % i: RE.WRAPPER_SHA + "0" * 48 for i in range(8)}
    chk = "x\n" + RE.CHECK_LAST + "\n"
    assert RE.ctl_b1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, ok) == RO.ctl_a1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, ok)
    bad = dict(ok, k3="f" * 64)
    assert RE.ctl_b1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, bad) == RO.ctl_a1(RE.WRAPPER_SHA, chk, RE.WRAPPER_SHA, bad)
    lv = {k: dict(v) for k, v in RE.EXPECTED_LEVELS.items()}
    assert RE.ctl_b3(lv)["pass"] == RO.ctl_a3(lv)["pass"] is True
    lv["RLc"]["F"] += 0.001
    assert RE.ctl_b3(lv)["pass"] == RO.ctl_a3(lv)["pass"] is False
    g, r = "== 3 passed ==", "== 4 passed ==\n"
    for args in ((g, r, 1, 2, 3), (g, r, 1, 4, 3), (g, "== 1 failed ==", 1, 2, 3), (g, r, "5", None, "1970-01-01T00:00:09Z")):
        assert RE.ctl_b4(*args) == RO.ctl_a4(*args)
    for args in ((True, True, True), (False, True, True), (True, None, True)):
        assert RE.ctl_b5(*args)["pass"] == RO.ctl_a5(*args)["pass"]


def test_reader_and_scripts_are_what_the_generator_writes():
    spec = importlib.util.spec_from_file_location("make_read_gaperr10k_t", str(HERE / "make_read_gaperr10k.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    for out, src, fn in (("read_gaperr10k.py", "read_fallattr10k.py", mk.make_reader), ("podread_gaperr10k.sh", "podread_fallattr10k_r1.sh", mk.make_driver),
                         ("pull_gaperr.sh", "pull_fallattr.sh", mk.make_pull), ("push_gaperr_read.sh", "push_fallattr_read.sh", mk.make_push)):
        assert fn(mk.rd(str(PREV / src))) == mk.rd(str(HERE / out)), out


def test_generator_refuses_a_changed_source_text():
    spec = importlib.util.spec_from_file_location("make_read_gaperr10k_t2", str(HERE / "make_read_gaperr10k.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    src = mk.rd(str(PREV / "read_fallattr10k.py"))
    with pytest.raises(AssertionError):
        mk.make_reader(src.replace("def file_on_join(J, z):", "def file_on_join_x(J, z):"))
    with pytest.raises(AssertionError):
        mk.sub("abc abc", "abc", "x", 1)


# ------------------------------------------------------------------------------------------------ the start-parse line of the driver
def start_line():
    ls = [x for x in (HERE / "podread_gaperr10k.sh").read_text(encoding="utf-8").splitlines() if "QS=$(" in x]
    assert len(ls) == 1, ls
    return ls[0].strip()


def run_start_parse(tmp_path, first_line):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    with open(str(tmp_path / "GAPERR10K_START.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(first_line + "\nsecond line\n")
    with open(str(tmp_path / "t.sh"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write('set -u\nFA=.\nstop() { echo "STOP $*"; exit 3; }\n' + start_line() + '\necho "QS=$QS"\n')
    return subprocess.run([bash, "t.sh"], cwd=str(tmp_path), capture_output=True, encoding="utf-8")


def test_start_parse_line_on_the_marker_as_the_pod_wrote_it(tmp_path):
    r = run_start_parse(tmp_path, "1795335883 2026-11-22T08:24:43Z pid 774424")
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "QS=1795335883", (r.stdout, r.stderr)


def test_start_parse_line_stops_on_an_iso_first_token(tmp_path):
    r = run_start_parse(tmp_path, "2026-11-22T08:24:43Z 1795335883 pid 774424")
    assert r.returncode == 3 and "STOP START-MARKER-UNREADABLE" in r.stdout and "QS=" not in r.stdout


def test_start_parse_line_stops_on_an_empty_marker(tmp_path):
    r = run_start_parse(tmp_path, "")
    assert r.returncode == 3 and "STOP START-MARKER-UNREADABLE" in r.stdout


# ------------------------------------------------------------------------------------------------ files: wording, paths, scripts
def kit_files():
    return [p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".md")]


def test_wording_scan_and_path_length():
    names = {p.name for p in kit_files()}
    assert {"read_gaperr10k.py", "make_read_gaperr10k.py", "podread_gaperr10k.sh", "pull_gaperr.sh", "push_gaperr_read.sh", "test_gaperr_read.py", "GAPERR_READ_DEFS.md"} <= names
    for p in kit_files():
        low = p.read_text(encoding="utf-8").lower()
        for w in BANNED:
            assert w not in low, (p.name, w)
        rel = "/f/TanitAD-scratch/v8proof/gaperr/read/" + p.name
        assert len(rel) <= 200 and p.stem.lower() not in ("aux", "con", "prn", "nul")


def test_no_file_above_5_mb_and_no_lead_pytest_out():
    for p in HERE.rglob("*"):
        if p.is_file():
            assert p.stat().st_size <= 5 * 1024 * 1024, p
    assert not (HERE / "lead_pytest.out").exists()


def test_scripts_parse_and_hold_their_rules():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    for nm in ("podread_gaperr10k.sh", "pull_gaperr.sh", "push_gaperr_read.sh"):
        r = subprocess.run([bash, "-n", str(HERE / nm)], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (nm, r.stderr)
    pod = (HERE / "podread_gaperr10k.sh").read_text(encoding="utf-8")
    for must in ("GAPERR10K_DONE.txt", "GAPERR10K_STOP.txt", "GAPERR10K_START.txt", "stat -c %Y", "--selftest", "-ge 20", "ALREADY-READ", "bNR bBR bSM bSL aLA gGEc vV0", "CUDA_VISIBLE_DEVICES=-1",
                 "CS=$V/cropscale", "MODEL=$CS/kit_read/read_cropscale10k.py", "FA=$V/gaperr", "KG=$FA/kit_gap", "OUT=$FA/read", "GF=$FA/gapfiles", "GAPERR10K_CONTROLS.json", "GAPERR10K_RESULT.json",
                 "ZZ PODREAD-GAPERR10K", "^b[0-5] (PASS|FAIL)|^ZZ READ-GAPERR10K", "--la-file", "--queue-start \"$QS\"", "VGAPA_NR.npz", "VGAPA_BR.npz", "VGAPA_SM.npz", "VGAPA_SL.npz",
                 "GAPERR_GAPFILES_RECORD.json", "read_gaperr10k.py"):
        assert must in pod, must
    for stale in ("FALLATTR", "fallattr10k", "VGAPA_LN", "VGAPA_LM", "VGAPA_LF", "aLN", "a[0-5]"):
        assert stale not in pod, stale
    for banned_cmd in ("rm ", "mv ", "kill", "pkill"):
        assert banned_cmd not in pod, banned_cmd
    pull = (HERE / "pull_gaperr.sh").read_text(encoding="utf-8")
    assert "sed '/hf_/d'" in pull and "*.npz|*.pt) continue" in pull and "pod_add111" in pull and "sha256sum" in pull and "/workspace/v8proof/gaperr" in pull and "GAPERR_GAPFILES_RECORD.json" in pull
    assert "add110" not in pull and "/fallattr" not in pull
    push = (HERE / "push_gaperr_read.sh").read_text(encoding="utf-8")
    assert "set -C" in push and "sha256sum" in push and "kit_read" in push and "lead_pytest.out" in push and "read_gaperr10k.py podread_gaperr10k.sh" in push and "/workspace/v8proof/gaperr" in push


def test_model_is_imported_not_edited():
    src = (HERE / "read_gaperr10k.py").read_text(encoding="utf-8")
    assert "spec_from_file_location" in src and "def w2_one" not in src and "def lead_multi" not in src
    for fn in ("rows_of", "load_logs", "lead_multi", "w2_one", "gapfile_ref", "meta_of", "run_file_sha", "ctl_g8"):
        assert hasattr(RE.MD, fn)
        assert "MD." + fn in src


def test_previous_reader_is_unchanged():
    import hashlib
    h = hashlib.sha256((PREV / "read_fallattr10k.py").read_bytes()).hexdigest()
    assert h.startswith("84ca638404d482c9")
