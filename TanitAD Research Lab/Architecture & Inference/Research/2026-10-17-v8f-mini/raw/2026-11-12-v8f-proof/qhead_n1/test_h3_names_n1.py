"""Tests of qhead_h3_n1.py (control h3, two parts) and qhead_names_n1.py (controls first, names only if every control passed). Tiny synthetic inputs under pytest's basetemp."""
import copy
import hashlib
import json
import os

import numpy as np
import pytest

import qhead_testlib as TL  # noqa: E402  (sets the source path first)
import qhead_h3_n1 as H3  # noqa: E402
import qhead_model as QM  # noqa: E402
import qhead_names_n1 as NM  # noqa: E402

D = 16
SHAPES3 = [(64, D), (64,), (128, 320), (128,), (3, 128), (3,)]
SHAPES1 = [(64, D), (64,), (128, 320), (128,), (1, 128), (1,)]


def tens(shapes, seed):
    rng = np.random.RandomState(seed)
    return {k: (rng.randn(*s) * 0.03).astype(np.float32) for k, s in zip(QM.TENSOR_KEYS, shapes)}


def h3_fixture(n_ticks=12, seed=0):
    rng = np.random.RandomState(seed)
    names = np.array(["cA", "cB"])
    T = 20
    Z = (rng.randn(2 * T, D)).astype(np.float16)
    es = {"Z": Z, "clip_idx": np.repeat([0, 1], T), "t": np.tile(np.arange(T), 2), "clips": names}
    join = {"clip": names, "off": np.array([0, T]), "n": np.array([T, T]), "lead_valid": np.ones(2 * T), "lead_gap": rng.uniform(5, 40, 2 * T)}
    banked, qt = tens(SHAPES3, 1), tens(SHAPES1, 2)
    ci = rng.randint(0, 2, n_ticks)
    j = rng.randint(4, T, n_ticks)
    k = np.zeros(n_ticks, np.int64)
    k[::3] = 1                                              # some later ticks
    present = np.ones(n_ticks, bool)
    present[1] = False
    rows = np.stack([ci * T + j - 4 + f for f in range(5)], 1)
    z = es["Z"][rows].astype(np.float32)
    g = QM.functional_forward(banked, z)[:, 1] * 50.0
    q = QM.functional_forward(qt, z)[:, 0] * 50.0
    used = np.minimum(g, q)
    rl = {"ci": ci, "j": j, "k": k, "present": present, "g_src": used.astype(np.float64).copy()}
    side = {"g_head": g.astype(np.float32), "q_head": q.astype(np.float32), "g_used": used.astype(np.float32), "p_head": np.ones(n_ticks, np.float32), "ok": np.ones(n_ticks, bool)}
    return rl, side, join, es, banked, qt


def test_h3_passes_on_consistent_data_and_reports_beside_numbers():
    rl, side, join, es, banked, qt = h3_fixture()
    r = H3.check("A", rl, side, join, es, banked, qt)
    assert r["PASS"] and r["a_in_rollout"]["PASS"] and r["b_offline_recompute"]["PASS"] and r["b_offline_recompute"]["counts"]
    assert r["b_offline_recompute"]["planted"]["fails_as_required"] is True and r["b_offline_recompute"]["tol_m"] == 0.02
    assert r["n_first_tick_present"] == r["n_recomputed"] > 0
    assert r["beside"]["mean_amount_taken_off_read_m"] >= 0.0 and 0.0 <= r["beside"]["coverage_eval_windows_share_q_above_label"] <= 1.0


def test_h3_mutation_gap_used_is_max_not_min_fails():
    rl, side, join, es, banked, qt = h3_fixture()
    rl["g_src"] = np.maximum(side["g_head"], side["q_head"]).astype(np.float64)
    side["g_used"] = rl["g_src"].astype(np.float32)
    assert not H3.check("A", rl, side, join, es, banked, qt)["PASS"]


def test_h3_mutation_head_read_unchanged_fails():
    rl, side, join, es, banked, qt = h3_fixture()
    rl["g_src"] = side["g_head"].astype(np.float64)
    side["g_used"] = side["g_head"]
    r = H3.check("A", rl, side, join, es, banked, qt)
    assert not r["PASS"] and not r["b_offline_recompute"]["PASS"]


def test_h3_part_a_bar_is_exactly_zero():
    rl, side, join, es, banked, qt = h3_fixture()
    r = H3.check("A", dict(rl, g_src=rl["g_src"] + 1e-9), side, join, es, banked, qt)
    assert not r["a_in_rollout"]["PASS"] and not r["PASS"] and r["a_in_rollout"]["bar_m"] == 0.0
    side2 = dict(side, g_used=side["g_used"] + np.float32(1e-3))                     # the logged g_used is not min(g_head, q_head)
    assert not H3.check("A", rl, side2, join, es, banked, qt)["a_in_rollout"]["PASS"]


def shifted(rl, side, d):
    """A rollout whose head and quantile reads are d metres off the recomputation, consistent inside the rollout (g_used = min of its own two reads)."""
    g, q = side["g_head"].astype(np.float64) + d, side["q_head"].astype(np.float64) + d
    u = np.minimum(g, q)
    return dict(rl, g_src=u.copy()), dict(side, g_head=g, q_head=q, g_used=u)


def test_h3_part_b_tolerance_002_within_passes_beyond_fails():
    rl, side, join, es, banked, qt = h3_fixture()
    r = H3.check("A", *(shifted(rl, side, 0.015)[0], shifted(rl, side, 0.015)[1]), join, es, banked, qt)
    assert r["a_in_rollout"]["PASS"] and r["b_offline_recompute"]["within_tol"] and r["PASS"], r
    r2 = H3.check("A", *(shifted(rl, side, 0.03)[0], shifted(rl, side, 0.03)[1]), join, es, banked, qt)
    assert r2["a_in_rollout"]["PASS"] and not r2["b_offline_recompute"]["within_tol"] and not r2["PASS"]


def test_h3_planted_difference_is_caught_else_part_b_does_not_count():
    rl, side, join, es, banked, qt = h3_fixture()
    r = H3.check("A", rl, side, join, es, banked, qt, tol=0.5)                       # a tolerance so wide that the planted 0.10 m passes: the guard must refuse to count part (b)
    assert r["b_offline_recompute"]["planted"]["fails_as_required"] is False
    assert r["b_offline_recompute"]["counts"] is False and not r["b_offline_recompute"]["PASS"] and not r["PASS"] and "does not count" in r["why"]
    r2 = H3.check("A", rl, side, join, es, banked, qt, tol=0.02, planted=0.10)
    assert r2["b_offline_recompute"]["planted"]["max_abs_with_offset"] > 0.09 and r2["PASS"]


def test_h3_not_row_for_row_and_unmatched_clip_names_fail():
    rl, side, join, es, banked, qt = h3_fixture()
    s2 = dict(side, g_head=side["g_head"][:-1])
    assert not H3.check("A", rl, s2, join, es, banked, qt)["PASS"]
    j2 = dict(join, clip=np.array(["zz1", "zz2"]))
    r = H3.check("A", rl, side, j2, es, banked, qt)
    assert not r["PASS"] and "matched" in r["why"]


def test_h3_cli_roundtrip(tmp_path):
    rl, side, join, es, banked, qt = h3_fixture()
    d = str(tmp_path)
    np.savez(d + "/rule_log_A.npz", **rl)
    np.savez(d + "/rule_log_A_qh.npz", **side)
    np.savez(d + "/join.npz", **join)
    np.savez(d + "/es.npz", **es)
    np.savez(d + "/bh.npz", **banked)
    np.savez(d + "/A.qhead.npz", **qt, hist=np.int64(5))
    assert H3.main(["--arm", "A", "--rule-log", d + "/rule_log_A.npz", "--join", d + "/join.npz", "--eval-states", d + "/es.npz", "--banked-head", d + "/bh.npz", "--qhead", d + "/A.qhead.npz", "--out", d + "/H3_A.json"]) == 0
    assert json.load(open(d + "/H3_A.json"))["PASS"] is True


# ------------------------------------------------------------------------------------------------ names
def cell(d_s, lo, hi, excl=None):
    return {"d_s": d_s, "D": float(np.mean(d_s)), "A_lo": lo, "A_hi": hi, "A_excludes_0": (hi < 0 or lo > 0) if excl is None else excl}


def reads(u_qh_none=(-0.05, -0.06), u_qh_qc=(-0.02, -0.025), e_qh_qc=(-0.1, -0.12), share=((0.10, 0.04), (0.10, 0.03)), cost_name="NO-COST"):
    hw = {"controls": {"ALL_PASS": True, "h2_planted_regression": {"PASS": True, "named_worse_on_U": True}}, "pairs": {}, "levels": {}}
    q3 = {"controls": {"ALL_PASS": True}, "pairs": {}}
    for ty, m in NM.TYPES.items():
        P = {}
        P[NM.pk(m["qh"], m["none"])] = cell(list(u_qh_none), -0.09, -0.02)
        P[NM.pk(m["qh"], m["qc"])] = cell(list(u_qh_qc), -0.04, -0.005)
        for y in ("rh", "rl", "mc"):
            P[NM.pk(m["qh"], m[y])] = cell([0.0, 0.01], -0.02, 0.03)
        for key, c in P.items():
            hw["pairs"][key] = {"seeds_read": [0, 1], "seeds_requested": [0, 1], "cells": {"U": c, "C": cell([0.0, 0.0], -0.01, 0.01)}}
        un, uq, ul = (0.10, 0.12), (0.05, 0.06), (0.0, 0.0)
        for arm, v in ((m["none"], un), (m["qh"], uq), (m["rl"], ul)):
            hw["levels"][arm] = {"cells": {"U": {"per_seed": list(v)}}}
        q3["pairs"][NM.pk(m["qh"], m["none"])] = {"Q3": {"name": cost_name, "cost_families": [] if cost_name == "NO-COST" else ["LONGITUDINAL"], "gain_families": []}}
        q3["pairs"][NM.pk(m["qh"], m["qc"])] = {"beside": {"ade6_m|all": cell(list(e_qh_qc), -0.2, -0.02)}}
    return hw, q3


class Args:
    pass


def fit_record(base):
    return {"q4_only_train_files": True, "files_read_sha256": {"/x/%s_train.npz" % base: "a"}, "coverage_fit": {"share_q_above_label": 0.2, "n": 20000},
            "coverage_select": {"share_q_above_label": 0.2, "n": 5000}, "coverage_check": {"share_q_above_label": 0.2, "n": 5000},
            "clip_sets": {"fit": [0, 800], "select": [800, 1000], "check": [1000, 1200]},
            "clip_index_range_of_rows_used": {"gradient_steps": [0, 799], "selection": [800, 999], "shift": [800, 999], "check_rows_only_coverage": [1000, 1199]}}


TOOL_BYTES = b"rollout tool bytes\n"


def make_inputs(tmp, hw, q3, fail=None):
    d = str(tmp)
    a = Args()
    a.hw, a.q3 = d + "/hw.json", d + "/q3.json"
    json.dump(hw, open(a.hw, "w"))
    json.dump(q3, open(a.q3, "w"))
    a.fit_dir, a.h3, a.rule_log_dir = d + "/fit", d + "/h3", d + "/pull"
    os.makedirs(a.fit_dir, exist_ok=True)
    os.makedirs(a.h3, exist_ok=True)
    for ty, m in NM.TYPES.items():
        for s in (0, 1):
            base, arm = "%s_s%d" % (m["base"], s), "%s_s%d" % (m["qh"], s)
            json.dump(fit_record(base), open("%s/%s.qhead.json" % (a.fit_dir, base), "w"))
            json.dump({"PASS": True, "beside": {"mean_amount_taken_off_read_m": 1.5, "coverage_eval_windows_share_q_above_label": 0.21}}, open("%s/H3_%s.json" % (a.h3, arm), "w"))
    for arm in ["%s_s%d" % (m["qh"], s) for m in NM.TYPES.values() for s in (0, 1)] + ["LSQHI_s0"]:
        os.makedirs("%s/%s/ticks" % (a.rule_log_dir, arm), exist_ok=True)
        np.savez("%s/%s/ticks/rule_log_%s.npz" % (a.rule_log_dir, arm, arm), lift=np.zeros(50, np.int64))
    a.tool_file = d + "/tool.py"
    open(a.tool_file, "wb").write(TOOL_BYTES)
    sha = hashlib.sha256(TOOL_BYTES).hexdigest()
    a.h1_text = d + "/h1.txt"
    open(a.h1_text, "w").write("[t] qhead: queue start; run script sha256 1a3c tool %s fit 2300\n[t] qhead: h1 the control: CMP files=141 differing=0 max_abs=0\n" % sha[:16])
    a.h6 = d + "/h6.json"
    json.dump({"all_ok": True, "n_mutations": 9}, open(a.h6, "w"))
    if fail:
        fail(a, d)
    return a


def run_names(tmp, hw, q3, fail=None):
    a = make_inputs(tmp, hw, q3, fail)
    C, hw2, q32 = NM.controls(a)
    return C, hw2, q32, a


def test_names_all_controls_pass_gives_names_and_beside(tmp_path):
    hw, q3 = reads()
    out = str(tmp_path / "o.json")
    a = make_inputs(tmp_path, hw, q3)
    assert NM.main(["--hw", a.hw, "--q3", a.q3, "--fit-dir", a.fit_dir, "--h1-text", a.h1_text, "--tool-file", a.tool_file, "--rule-log-dir", a.rule_log_dir, "--h3", a.h3, "--h6", a.h6, "--out", out]) == 0
    r = json.load(open(out))
    assert r["controls_all_pass"] and r["names"]["LS"]["reaches_oracle"] == "QHEAD-REACHES-ORACLE-QE" and r["names"]["LSD"]["U_against_map"] == "QHEAD-CUTS-MORE-QE"
    assert r["names"]["LS"]["cost_against_map"] == "QHEAD-CHEAPER-QE" and r["names"]["LS"]["cost_against_no_rule"] == "QHEAD-NO-COST-QE"
    assert r["beside"]["LS"]["first_tick_mean_amount_taken_off_read_m"] == [1.5, 1.5] and "STRATEGIC: NOT COMPUTED (n = 0;" in r["family_line"]
    assert abs(r["beside"]["LS"]["share_of_label_fed_fall_reached_per_base"][0] - 0.5) < 1e-9


@pytest.mark.parametrize("which", ["h1", "h1tool", "h2", "h3", "h4", "h5", "h5select", "h6", "h7", "h8", "h8check"])
def test_failed_control_withholds_names_and_differences(tmp_path, which):
    hw, q3 = reads()

    def fail(a, d):
        if which == "h1":
            open(a.h1_text, "w").write(open(a.h1_text).read().replace("differing=0", "differing=2"))
        elif which == "h1tool":                                              # the rollout tool file now is not the one the first run recorded
            open(a.tool_file, "wb").write(TOOL_BYTES + b"x")
        elif which == "h5select":                                            # the select coverage is fine, the CHECK coverage is not: h5 must read the check coverage
            j = json.load(open("%s/LS_s0.qhead.json" % a.fit_dir))
            j["coverage_check"]["share_q_above_label"] = 0.40
            json.dump(j, open("%s/LS_s0.qhead.json" % a.fit_dir, "w"))
        elif which == "h8":
            j = json.load(open("%s/LS_s1.qhead.json" % a.fit_dir))
            j["clip_sets"]["select"] = [800, 1100]                           # the sets overlap
            json.dump(j, open("%s/LS_s1.qhead.json" % a.fit_dir, "w"))
        elif which == "h8check":                                             # a check row entered the selection
            j = json.load(open("%s/LSD_s0.qhead.json" % a.fit_dir))
            j["clip_index_range_of_rows_used"]["selection"] = [800, 1150]
            json.dump(j, open("%s/LSD_s0.qhead.json" % a.fit_dir, "w"))
        elif which == "h2":
            np.savez("%s/LSQE_s0/ticks/rule_log_LSQE_s0.npz" % a.rule_log_dir, lift=np.array([0, 1, 0]))
        elif which == "h3":
            json.dump({"PASS": False, "why": "x"}, open("%s/H3_LSDQE_s1.json" % a.h3, "w"))
        elif which == "h4":
            j = json.load(open("%s/LS_s0.qhead.json" % a.fit_dir))
            j["files_read_sha256"]["/x/LS_s0_eval.npz"] = "b"
            json.dump(j, open("%s/LS_s0.qhead.json" % a.fit_dir, "w"))
        elif which == "h5":
            j = json.load(open("%s/LSD_s1.qhead.json" % a.fit_dir))
            j["coverage_check"]["share_q_above_label"] = 0.31
            json.dump(j, open("%s/LSD_s1.qhead.json" % a.fit_dir, "w"))
        elif which == "h6":
            json.dump({"all_ok": False, "n_mutations": 9}, open(a.h6, "w"))
        elif which == "h7":
            h = json.load(open(a.hw))
            h["controls"]["h2_planted_regression"]["named_worse_on_U"] = False
            json.dump(h, open(a.hw, "w"))
    a = make_inputs(tmp_path, hw, q3, fail)
    out = str(tmp_path / "o.json")
    rc = NM.main(["--hw", a.hw, "--q3", a.q3, "--fit-dir", a.fit_dir, "--h1-text", a.h1_text, "--tool-file", a.tool_file, "--rule-log-dir", a.rule_log_dir, "--h3", a.h3, "--h6", a.h6, "--out", out])
    r = json.load(open(out))
    assert rc == 2 and r["names"] is None and r["beside"] is None and not r["controls_all_pass"]
    text = open(out).read()
    assert "U_qh_minus_none" not in text and "QHEAD-" not in text
    bad = [k for k, v in r["controls"].items() if not v["PASS"]]
    assert len(bad) == 1 and bad[0].startswith(which[:2] + "_")


def test_h1_is_carried_when_the_tool_is_unchanged_and_missing_otherwise(tmp_path):
    hw, q3 = reads()
    C, _, _, a = run_names(tmp_path, hw, q3)
    h1 = C["h1_control_rollout_equals_head_fed_dump"]
    assert h1["PASS"] and h1["tool_unchanged"] and h1["cmp"] == ("141", "0")
    open(a.tool_file, "wb").write(TOOL_BYTES + b"y")
    h1b = NM.controls(a)[0]["h1_control_rollout_equals_head_fed_dump"]
    assert not h1b["PASS"] and "missing" in h1b["why"]
    open(a.h1_text, "w").write("CMP files=141 differing=0\n")                  # no recorded tool sha256 at all
    open(a.tool_file, "wb").write(TOOL_BYTES)
    assert not NM.controls(a)[0]["h1_control_rollout_equals_head_fed_dump"]["PASS"]


def test_h5_reads_the_check_coverage_not_the_select_coverage(tmp_path):
    hw, q3 = reads()
    a = make_inputs(tmp_path, hw, q3)
    j = json.load(open("%s/LS_s0.qhead.json" % a.fit_dir))
    j["coverage_select"]["share_q_above_label"] = 0.20
    j["coverage_check"]["share_q_above_label"] = 0.44
    json.dump(j, open("%s/LS_s0.qhead.json" % a.fit_dir, "w"))
    C = NM.controls(a)[0]
    assert not C["h5_check_coverage_in_0.15_0.25"]["PASS"] and C["h5_check_coverage_in_0.15_0.25"]["per_base"]["LS_s0"]["coverage_check"] == 0.44


def test_outcomes_cover_every_branch(tmp_path):
    hw, q3 = reads(u_qh_none=(-0.01, 0.02), u_qh_qc=(0.02, 0.03), e_qh_qc=(0.1, 0.2), share=None, cost_name="COST-LONGITUDINAL")
    for m in NM.TYPES.values():
        hw["pairs"][NM.pk(m["qh"], m["qc"])]["cells"]["U"] = cell([0.02, 0.03], 0.005, 0.05)
        hw["pairs"][NM.pk(m["qh"], m["none"])]["cells"]["U"] = cell([-0.01, 0.02], -0.05, 0.04)
        q3["pairs"][NM.pk(m["qh"], m["qc"])]["beside"]["ade6_m|all"] = cell([0.1, 0.2], 0.05, 0.3)
    N, B = NM.names(hw, q3)
    assert N["LS"]["U_against_map"] == "QHEAD-CUTS-LESS-QE" and N["LS"]["cost_against_map"] == "QHEAD-DEARER-QE" and N["LS"]["reaches_oracle"] is None and N["LS"]["cost_against_no_rule"] == "QHEAD-COSTS-QE"
    hw2, q32 = reads(u_qh_qc=(-0.02, 0.01), e_qh_qc=(-0.1, 0.12))
    for m in NM.TYPES.values():
        hw2["pairs"][NM.pk(m["qh"], m["qc"])]["cells"]["U"] = cell([-0.02, 0.01], -0.03, 0.02)
        q32["pairs"][NM.pk(m["qh"], m["qc"])]["beside"]["ade6_m|all"] = cell([-0.1, 0.12], -0.2, 0.1)
    N2, _ = NM.names(hw2, q32)
    assert N2["LS"]["U_against_map"] == "QHEAD-U-EQUAL-QE" and N2["LS"]["cost_against_map"] == "QHEAD-COST-EQUAL-QE"


def test_share_below_half_in_one_seed_is_not_named(tmp_path):
    hw, q3 = reads()
    for m in NM.TYPES.values():
        hw["levels"][m["qh"]]["cells"]["U"]["per_seed"] = [0.05, 0.095]          # share 0.5 then 0.05 (none 0.10, 0.12 -> second seed share 0.2083)
    N, _ = NM.names(hw, q3)
    assert N["LS"]["reaches_oracle"] is None


def test_one_seed_missing_is_not_named():
    hw, q3 = reads()
    hw["pairs"][NM.pk("LSQE", "LS")]["seeds_read"] = [0]
    N, B = NM.names(hw, q3)
    assert "both seeds" in N["LS"]["all"] and "reaches_oracle" in N["LSD"]
