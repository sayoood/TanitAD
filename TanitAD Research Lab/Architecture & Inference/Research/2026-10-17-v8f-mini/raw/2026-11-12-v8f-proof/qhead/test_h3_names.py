"""Tests of qhead_h3.py (control h3) and qhead_names.py (controls first, names only if every control passed). Tiny synthetic inputs under pytest's basetemp."""
import copy
import json
import os

import numpy as np
import pytest

import qhead_testlib as TL  # noqa: E402  (sets the source path first)
import qhead_h3 as H3  # noqa: E402
import qhead_model as QM  # noqa: E402
import qhead_names as NM  # noqa: E402

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
    r = H3.check("A", rl, side, join, es, banked, qt, tol=1e-4)
    assert r["PASS"] and r["a_in_rollout"]["PASS"] and r["b_card_form"]["PASS"]
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
    assert not r["PASS"] and not r["b_card_form"]["PASS"]


def test_h3_small_tamper_beyond_tolerance_fails_and_within_passes():
    rl, side, join, es, banked, qt = h3_fixture()
    rl2 = dict(rl, g_src=rl["g_src"] + 5e-4)
    assert not H3.check("A", rl2, side, join, es, banked, qt, tol=1e-4)["PASS"]
    rl3 = dict(rl, g_src=rl["g_src"] + 5e-5)
    assert H3.check("A", rl3, side, join, es, banked, qt, tol=1e-4)["PASS"]


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
            json.dump({"q4_only_train_files": True, "files_read_sha256": {"/x/%s_train.npz" % base: "a"}, "coverage_held_out": {"share_q_above_label": 0.2, "n": 5000}}, open("%s/%s.qhead.json" % (a.fit_dir, base), "w"))
            json.dump({"PASS": True, "beside": {"mean_amount_taken_off_read_m": 1.5, "coverage_eval_windows_share_q_above_label": 0.21}}, open("%s/H3_%s.json" % (a.h3, arm), "w"))
    for arm in ["%s_s%d" % (m["qh"], s) for m in NM.TYPES.values() for s in (0, 1)] + ["LSQHI_s0"]:
        os.makedirs("%s/%s/ticks" % (a.rule_log_dir, arm), exist_ok=True)
        np.savez("%s/%s/ticks/rule_log_%s.npz" % (a.rule_log_dir, arm, arm), lift=np.zeros(50, np.int64))
    a.h1_text = d + "/h1.txt"
    open(a.h1_text, "w").write("h1 ...: CMP files=141 differing=0 \n")
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
    assert NM.main(["--hw", a.hw, "--q3", a.q3, "--fit-dir", a.fit_dir, "--h1-text", a.h1_text, "--rule-log-dir", a.rule_log_dir, "--h3", a.h3, "--h6", a.h6, "--out", out]) == 0
    r = json.load(open(out))
    assert r["controls_all_pass"] and r["names"]["LS"]["reaches_oracle"] == "QHEAD-REACHES-ORACLE" and r["names"]["LSD"]["U_against_map"] == "QHEAD-CUTS-MORE"
    assert r["names"]["LS"]["cost_against_map"] == "QHEAD-CHEAPER" and r["names"]["LS"]["cost_against_no_rule"] == "QHEAD-NO-COST"
    assert r["beside"]["LS"]["first_tick_mean_amount_taken_off_read_m"] == [1.5, 1.5] and "STRATEGIC: NOT COMPUTED (n = 0;" in r["family_line"]
    assert abs(r["beside"]["LS"]["share_of_label_fed_fall_reached_per_base"][0] - 0.5) < 1e-9


@pytest.mark.parametrize("which", ["h1", "h2", "h3", "h4", "h5", "h6", "h7"])
def test_failed_control_withholds_names_and_differences(tmp_path, which):
    hw, q3 = reads()

    def fail(a, d):
        if which == "h1":
            open(a.h1_text, "w").write("CMP files=141 differing=2 \n")
        elif which == "h2":
            np.savez("%s/LSQH_s0/ticks/rule_log_LSQH_s0.npz" % a.rule_log_dir, lift=np.array([0, 1, 0]))
        elif which == "h3":
            json.dump({"PASS": False, "why": "x"}, open("%s/H3_LSDQH_s1.json" % a.h3, "w"))
        elif which == "h4":
            j = json.load(open("%s/LS_s0.qhead.json" % a.fit_dir))
            j["files_read_sha256"]["/x/LS_s0_eval.npz"] = "b"
            json.dump(j, open("%s/LS_s0.qhead.json" % a.fit_dir, "w"))
        elif which == "h5":
            j = json.load(open("%s/LSD_s1.qhead.json" % a.fit_dir))
            j["coverage_held_out"]["share_q_above_label"] = 0.31
            json.dump(j, open("%s/LSD_s1.qhead.json" % a.fit_dir, "w"))
        elif which == "h6":
            json.dump({"all_ok": False, "n_mutations": 9}, open(a.h6, "w"))
        elif which == "h7":
            h = json.load(open(a.hw))
            h["controls"]["h2_planted_regression"]["named_worse_on_U"] = False
            json.dump(h, open(a.hw, "w"))
    a = make_inputs(tmp_path, hw, q3, fail)
    out = str(tmp_path / "o.json")
    rc = NM.main(["--hw", a.hw, "--q3", a.q3, "--fit-dir", a.fit_dir, "--h1-text", a.h1_text, "--rule-log-dir", a.rule_log_dir, "--h3", a.h3, "--h6", a.h6, "--out", out])
    r = json.load(open(out))
    assert rc == 2 and r["names"] is None and r["beside"] is None and not r["controls_all_pass"]
    text = open(out).read()
    assert "U_qh_minus_none" not in text and "QHEAD-" not in text
    bad = [k for k, v in r["controls"].items() if not v["PASS"]]
    assert len(bad) == 1 and bad[0].startswith(which + "_")


def test_outcomes_cover_every_branch(tmp_path):
    hw, q3 = reads(u_qh_none=(-0.01, 0.02), u_qh_qc=(0.02, 0.03), e_qh_qc=(0.1, 0.2), share=None, cost_name="COST-LONGITUDINAL")
    for m in NM.TYPES.values():
        hw["pairs"][NM.pk(m["qh"], m["qc"])]["cells"]["U"] = cell([0.02, 0.03], 0.005, 0.05)
        hw["pairs"][NM.pk(m["qh"], m["none"])]["cells"]["U"] = cell([-0.01, 0.02], -0.05, 0.04)
        q3["pairs"][NM.pk(m["qh"], m["qc"])]["beside"]["ade6_m|all"] = cell([0.1, 0.2], 0.05, 0.3)
    N, B = NM.names(hw, q3)
    assert N["LS"]["U_against_map"] == "QHEAD-CUTS-LESS" and N["LS"]["cost_against_map"] == "QHEAD-DEARER" and N["LS"]["reaches_oracle"] is None and N["LS"]["cost_against_no_rule"] == "QHEAD-COSTS"
    hw2, q32 = reads(u_qh_qc=(-0.02, 0.01), e_qh_qc=(-0.1, 0.12))
    for m in NM.TYPES.values():
        hw2["pairs"][NM.pk(m["qh"], m["qc"])]["cells"]["U"] = cell([-0.02, 0.01], -0.03, 0.02)
        q32["pairs"][NM.pk(m["qh"], m["qc"])]["beside"]["ade6_m|all"] = cell([-0.1, 0.12], -0.2, 0.1)
    N2, _ = NM.names(hw2, q32)
    assert N2["LS"]["U_against_map"] == "QHEAD-U-EQUAL" and N2["LS"]["cost_against_map"] == "QHEAD-COST-EQUAL"


def test_share_below_half_in_one_seed_is_not_named(tmp_path):
    hw, q3 = reads()
    for m in NM.TYPES.values():
        hw["levels"][m["qh"]]["cells"]["U"]["per_seed"] = [0.05, 0.095]          # share 0.5 then 0.05 (none 0.10, 0.12 -> second seed share 0.2083)
    N, _ = NM.names(hw, q3)
    assert N["LS"]["reaches_oracle"] is None


def test_one_seed_missing_is_not_named():
    hw, q3 = reads()
    hw["pairs"][NM.pk("LSQH", "LS")]["seeds_read"] = [0]
    N, B = NM.names(hw, q3)
    assert "both seeds" in N["LS"]["all"] and "reaches_oracle" in N["LSD"]
