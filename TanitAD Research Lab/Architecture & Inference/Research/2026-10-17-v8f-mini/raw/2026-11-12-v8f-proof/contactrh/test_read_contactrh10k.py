"""Tests of read_contactrh10k.py on SYNTHETIC fixtures only (card ADD141, control c4). No real file is read."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_contactrh10k as R  # noqa: E402

N_EP, WPE, SEEDS = 6, 4, (0, 1, 2)
GAP = 8.0
EXTRA = 0.5


def kind_of(w):
    return {0: "X", 1: "S", 2: "Y"}.get(w % 6, "N")


def build(tmp, err_x=3.0, err_s_h=1.0, err_s_u=1.0, err_y=4.0, clos_h=2.0, clos_u=1.0, present_x=True):
    """root with H_RQ50K_s<m> and U_RQ50K_s<m> runs; X = RH contact only, S = both, Y = UE only, N = none"""
    root = os.path.join(str(tmp), "pull")
    n = 101
    clip = np.array(["c%d" % i for i in range(N_EP)])
    np.savez(os.path.join(str(tmp), "join.npz"), clip=clip, off=np.arange(N_EP) * n, n=np.full(N_EP, n), lead_valid=np.ones(N_EP * n, np.int8),
             lead_gap=np.full(N_EP * n, GAP, np.float32))
    for m in SEEDS:
        for tag, arm in ((R.TAG_H, "H"), (R.TAG_U, "U")):
            rf = os.path.join(root, "%s_s%d" % (tag, m))
            dd = os.path.join(rf, "dump_%s_s%d_m10_causal_heads" % (tag, m))
            os.makedirs(dd)
            os.makedirs(os.path.join(rf, "ticks"))
            rows = {f: [] for f in R.NEED}
            for e in range(N_EP):
                ws = np.arange(WPE) * 10
                g = np.zeros((WPE, 30, 2))
                cl = np.zeros((WPE, 30, 2))
                for q in range(WPE):
                    w = e * WPE + q
                    kd = kind_of(w)
                    ex = EXTRA if ((arm == "H" and kd in ("X", "S")) or (arm == "U" and kd in ("S", "Y"))) else 0.0
                    g[q, :, 0] = np.arange(1, 31)
                    cl[q, :, 0] = np.arange(1, 31) * (1.0 + ex)
                    for k in (0, 10, 20):
                        pres = not (arm == "H" and kd == "X" and not present_x)
                        err = {("H", "X"): err_x, ("H", "S"): err_s_h, ("U", "S"): err_s_u, ("U", "Y"): err_y}.get((arm, kd), 0.0)
                        clos = {("H", "S"): clos_h, ("U", "S"): clos_u}.get((arm, kd), 0.0)
                        a_l = float(k)
                        a_m = k * (1.0 + ex)
                        gs = GAP + err
                        for f, v in (("ci", e), ("t0", int(ws[q])), ("j", int(ws[q]) + k), ("k", k), ("present", pres), ("g_src", gs), ("c_src", -clos),
                                     ("v0", 5.0), ("v_L", 4.0), ("g0", gs + a_l - a_m), ("a_l", a_l), ("a_m", a_m)):
                            rows[f].append(v)
                np.savez(os.path.join(dd, "ep%03d.npz" % e), g=g, cl=cl, ws=ws, eid=np.array(["e%d" % e] * WPE), clip_index=np.full(WPE, e), v0=np.full(WPE, 5.0))
            np.savez(os.path.join(rf, "ticks", "rule_log_%s_s%d.npz" % (tag, m)), **{f: np.array(v) for f, v in rows.items()})
    return root, os.path.join(str(tmp), "join.npz")


def make_tests_file(tmp, text="5 passed in 1.0s\n"):
    p = os.path.join(str(tmp), "dev_pytest.out")
    with open(p, "w") as f:
        f.write(text)
    return p


def run(tmp, root, join, out, tests="OK", seeds="0,1,2"):
    tp = make_tests_file(tmp) if tests == "OK" else tests
    argv = ["read", "--root", root, "--join", join, "--out-dir", os.path.join(str(tmp), out), "--seeds", seeds, "--expect-files", str(N_EP)]
    if tp:
        argv += ["--tests-out", tp]
    rc = R.main(argv)
    od = os.path.join(str(tmp), out)
    res = json.load(open(os.path.join(od, R.RESULT_NAME))) if os.path.isfile(os.path.join(od, R.RESULT_NAME)) else None
    ctl = json.load(open(os.path.join(od, R.CONTROLS_NAME))) if os.path.isfile(os.path.join(od, R.CONTROLS_NAME)) else None
    return rc, res, ctl, od


def test_controls_pass_and_values(tmp_path):
    root, join = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, "o1")
    assert rc == 0 and ctl["all_pass"], ctl
    v = res["values"]
    assert v["n_X"] == 12 and v["n_S"] == 12 and v["n_Y"] == 12
    assert abs(v["Q1"]["value"] - 2.0) < 1e-9          # X err +3 minus S err +1
    assert abs(v["Q2"]["value"]) < 1e-12
    assert abs(v["Q3"]["value"] - 1.0) < 1e-9          # RH closes at 2.0, UE at 1.0
    assert abs(v["beside_Q1_UE_contact_only"]["value"] - 3.0) < 1e-9   # Y err +4 minus S(UE) err +1
    assert res["name"] == R.N_MISREAD
    assert os.path.isfile(os.path.join(od, R.VERDICT_NAME))


def test_c3_planted_reads_plus_two(tmp_path):
    root, join = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, "o2")
    c3 = ctl["controls"]["c3"]
    assert c3["PASS"] and c3["planted_deviation"] < 1e-6 and c3["unchanged_reads_main_value"]


def test_c3_detects_a_tool_that_ignores_the_plant(tmp_path, monkeypatch):
    root, join = build(tmp_path, err_x=2.0)      # unplanted Q1 = 1.0, so a plant that does nothing cannot read +2.0
    monkeypatch.setattr(R, "plant_logs", lambda runs, seeds, tabs: {m: runs[("H", m)]["log"] for m in seeds})
    rc, res, ctl, od = run(tmp_path, root, join, "o3")
    assert rc == 2 and not ctl["controls"]["c3"]["PASS"]
    assert res["values"] is None and res["name"] is None


def test_failing_c4_withholds_values(tmp_path):
    root, join = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, "o4", tests=make_tests_file(tmp_path, "3 passed, 1 failed in 1s\n"))
    assert rc == 2 and not ctl["controls"]["c4"]["PASS"]
    assert res["values"] is None and res["name"] is None
    assert "withheld" in open(os.path.join(od, R.VERDICT_NAME)).read()


def test_no_tests_file_withholds(tmp_path):
    root, join = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, "o5", tests=None)
    assert rc == 2 and res["values"] is None


def test_failing_c2_inconsistent_gap_withholds(tmp_path):
    root, join = build(tmp_path)
    p = R.log_path(root, R.TAG_H, 0)
    z = dict(np.load(p))
    z["g0"] = z["g0"] + 0.5          # g0 stops matching g_src + a_l - a_m
    np.savez(p, **z)
    rc, res, ctl, od = run(tmp_path, root, join, "o6")
    assert rc == 2 and not ctl["controls"]["c2"]["PASS"] and res["values"] is None


def test_failing_c2_missing_tick_withholds(tmp_path):
    root, join = build(tmp_path)
    p = R.log_path(root, R.TAG_H, 1)
    z = dict(np.load(p))
    keep = ~(z["ci"] == 0)           # episode 0 loses every row, including its X window at the first tick
    np.savez(p, **{k: v[keep] for k, v in z.items()})
    rc, res, ctl, od = run(tmp_path, root, join, "o7")
    assert rc == 2 and not ctl["controls"]["c2"]["PASS"] and res["values"] is None


def test_failing_c1_windows_differ_withholds(tmp_path):
    root, join = build(tmp_path)
    f = os.path.join(R.dump_folder(root, R.TAG_U, 1), "ep000.npz")
    z = dict(np.load(f))
    z["ws"] = z["ws"] + 1
    np.savez(f, **z)
    rc, res, ctl, od = run(tmp_path, root, join, "o8")
    assert rc == 2 and not ctl["controls"]["c1"]["PASS"] and res["values"] is None


def test_q2_presence_misses(tmp_path):
    root, join = build(tmp_path, present_x=False)
    rc, res, ctl, od = run(tmp_path, root, join, "o9")
    assert rc == 0
    v = res["values"]
    assert abs(v["Q2"]["value"] - 1.0) < 1e-12 and v["Q1"]["value"] is None       # every X window missed; no X gap error is defined
    assert res["name"] == R.N_MISREAD


def test_same_read_reachable_end_to_end(tmp_path):
    root, join = build(tmp_path, err_x=1.0, err_s_h=1.0)       # same read in X and S, RH closes faster
    rc, res, ctl, od = run(tmp_path, root, join, "o10")
    assert rc == 0 and abs(res["values"]["Q1"]["value"]) < 1e-9
    assert res["name"] == R.N_SAME


def test_unresolved_reachable_end_to_end(tmp_path):
    root, join = build(tmp_path, err_x=1.0, err_s_h=1.0, clos_h=1.0, clos_u=1.0)
    rc, res, ctl, od = run(tmp_path, root, join, "o11")
    assert rc == 0 and res["name"] == R.N_UNRES


def iv(lo, hi):
    return {"value": 0.0, "interval": [lo, hi], "n_a": 5, "n_b": 5}


def test_names_by_rule():
    v = {"n_X": 12, "Q1": iv(0.5, 1.0), "Q2": iv(-0.1, 0.1), "Q3": iv(-1, 1)}
    assert R.derive_name(v)[0] == R.N_MISREAD
    v = {"n_X": 12, "Q1": iv(-0.5, 0.5), "Q2": iv(0.1, 0.4), "Q3": iv(-1, 1)}
    assert R.derive_name(v)[0] == R.N_MISREAD
    v = {"n_X": 12, "Q1": iv(-0.5, 0.5), "Q2": iv(-0.1, 0.1), "Q3": iv(0.2, 1)}
    assert R.derive_name(v)[0] == R.N_SAME
    v = {"n_X": 12, "Q1": iv(-0.5, 0.5), "Q2": iv(-0.1, 0.1), "Q3": iv(-0.2, 1)}
    assert R.derive_name(v)[0] == R.N_UNRES
    v = {"n_X": 12, "Q1": iv(-2, -1), "Q2": iv(-0.1, 0.1), "Q3": iv(0.2, 1)}
    assert R.derive_name(v)[0] == R.N_UNRES                      # Q1 separated below zero: neither name
    v = {"n_X": 9, "Q1": iv(0.5, 1.0), "Q2": iv(-0.1, 0.1), "Q3": iv(0.2, 1)}
    nm, why = R.derive_name(v)
    assert nm == R.N_UNRES and "too small" in why


def test_estimator_constant_difference_and_determinism():
    idx = R.make_draws(6, 2000, 141)
    assert np.array_equal(idx, R.make_draws(6, 2000, 141)) and idx.shape == (2000, 6)
    ep = np.array([0, 1, 2, 3, 4, 5])
    a = R.mean_diff(ep, np.full(6, 3.0), ep, np.full(6, 1.0), 6, idx)
    assert abs(a["value"] - 2.0) < 1e-12 and a["interval"] == [2.0, 2.0]
    e = R.mean_diff(ep, np.array([]), ep, np.full(6, 1.0), 6, idx)
    assert e["value"] is None


def test_c4_parser(tmp_path):
    assert R.ctl_c4(make_tests_file(tmp_path, "12 passed in 3s\n"))["PASS"]
    assert not R.ctl_c4(make_tests_file(tmp_path, "FAILED x::y\n1 failed, 2 passed\n"))["PASS"]
    assert not R.ctl_c4(make_tests_file(tmp_path, "collected 0 items\n"))["PASS"]
    assert not R.ctl_c4(os.path.join(str(tmp_path), "absent.out"))["PASS"]


def test_refuses_missing_inputs_and_existing_outputs(tmp_path):
    root, join = build(tmp_path)
    assert R.main(["read", "--root", os.path.join(str(tmp_path), "nope"), "--join", join, "--out-dir", os.path.join(str(tmp_path), "o20")]) == 3
    rc, _, _, od = run(tmp_path, root, join, "o21")
    assert rc == 0
    assert R.main(["read", "--root", root, "--join", join, "--out-dir", od, "--expect-files", str(N_EP), "--tests-out", make_tests_file(tmp_path)]) == 3
