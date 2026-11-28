"""Tests of read_contactrh10kc.py (card ADD143, control c4) on SYNTHETIC fixtures only. No real result file is read.

Fixture: 6 episodes x 4 windows (the grid), model seeds 0..2, runs H_RQ50K_P30_s<m> and U_RQ50K_P30_s<m> with an own start point p0 = (DP, 0) at every window (the aged state), a tick file with
pose and speed per tick, a main-phase rule log (ticks k = 0, 10, 20), pre-roll logs and the argv / done files of the queue of card ADD143. Window kinds: X = RH contact only, S = both, Y = UE only.
"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_contactrh10kc as T  # noqa: E402

KIT = "F:/TanitAD-scratch/v8proof/rhacc10k/kit"
sys.path.insert(0, KIT)
import box2_crhc_queue as Q  # noqa: E402

_H = T.bind_headway()
ORIG_ENGINES = (_H.RH.comps, _H.window_cells, bool(getattr(_H.RH, "_note_n4", False)))   # before any patch of this process

N_EP, WPE, SEEDS = 6, 4, (0, 1, 2)
GAP, DP, DT = 8.0, 0.5, 0.2
WS = [0, 10, 20, 30]


def kind_of(w):
    return {0: "X", 1: "S", 2: "Y"}.get(w % 6, "N")


def ex_of(arm, kd, exs_h):
    if arm == "H":
        return {"X": 1.5, "S": exs_h}.get(kd, 0.0)
    return {"S": 0.5, "Y": 0.5}.get(kd, 0.0)


def build(tmp, err_x=3.0, err_s_h=1.0, err_s_u=1.0, err_y=4.0, exs_h=0.6, c_src_u=-1.0, present_x=True, v0_at_t0=False, tick_v_bias=0.0):
    root = os.path.join(str(tmp), "pull")
    n = 101
    clip = np.array(["c%d" % i for i in range(N_EP)])
    join = os.path.join(str(tmp), "join.npz")
    np.savez(join, clip=clip, off=np.arange(N_EP) * n, n=np.full(N_EP, n), lead_valid=np.ones(N_EP * n, np.int8), lead_gap=np.full(N_EP * n, GAP, np.float32))
    grid = os.path.join(str(tmp), "grid.json")
    json.dump({"order": [str(c) for c in clip], "windows": {str(c): WS for c in clip}, "n_windows": N_EP * WPE}, open(grid, "w"))
    want = {r["arm"]: r["argv"] for r in Q.build_runs()}
    for m in SEEDS:
        for tag, arm in ((T.TAG_H, "H"), (T.TAG_U, "U")):
            name = "%s_s%d" % (tag, m)
            rf = os.path.join(root, name)
            dd = os.path.join(rf, "dump_%s_m10_causal_heads" % name)
            td = os.path.join(rf, "ticks")
            os.makedirs(dd)
            os.makedirs(td)
            json.dump({"argv": want[name]}, open(os.path.join(rf, "argv.json"), "w"))
            json.dump({"rc": 0}, open(os.path.join(rf, "RUN_DONE.json"), "w"))
            rows = {f: [] for f in T.NEED}
            trow = {"ci": [], "t0": [], "tick": [], "pose": [], "pose_w0": []}
            offs = {"ci": [], "t0": [], "kind": [], "dpos": []}
            for e in range(N_EP):
                g = np.zeros((WPE, 30, 2))
                cl = np.zeros((WPE, 30, 2))
                for q in range(WPE):
                    w = e * WPE + q
                    kd = kind_of(w)
                    ex = ex_of(arm, kd, exs_h)
                    g[q, :, 0] = np.arange(1, 31)
                    cl[q, :, 0] = DP + (1.0 + ex) * np.arange(1, 31)
                    speed = (1.0 + ex) / DT
                    vt0 = speed + 0.3
                    offs["ci"].append(e), offs["t0"].append(WS[q]), offs["kind"].append("model"), offs["dpos"].append(DP)
                    for tick, k in enumerate((0, 10, 20)):
                        trow["ci"].append(e), trow["t0"].append(WS[q]), trow["tick"].append(tick)
                        trow["pose"].append([DP, 0.0, 0.0, vt0 + tick_v_bias if k == 0 else speed])
                        trow["pose_w0"].append([0.0, 0.0, 0.0])
                        pres = not (arm == "H" and kd == "X" and not present_x)
                        err = {("H", "X"): err_x, ("H", "S"): err_s_h, ("U", "S"): err_s_u, ("U", "Y"): err_y}.get((arm, kd), 0.0)
                        a_l = float(k)
                        a_m = 0.0 if k == 0 else DP + (1.0 + ex) * k
                        gs = GAP + err
                        cs = c_src_u if (arm == "U" and kd == "S") else -1.0
                        v0 = vt0 if (k == 0 or v0_at_t0) else speed
                        for f, v in (("ci", e), ("t0", WS[q]), ("j", WS[q] + k), ("k", k), ("present", pres), ("g_src", gs), ("c_src", cs),
                                     ("v0", v0), ("v_L", 4.0), ("g0", gs + a_l - a_m), ("a_l", a_l), ("a_m", a_m)):
                            rows[f].append(v)
                np.savez(os.path.join(dd, "ep%03d.npz" % e), g=g, cl=cl, ws=np.array(WS), eid=np.array(["e%d" % e] * WPE), clip_index=np.full(WPE, e), v0=np.full(WPE, 5.0))
            np.savez(os.path.join(td, "rule_log_%s.npz" % name), **{f: np.array(v) for f, v in rows.items()})
            np.savez(os.path.join(td, "ticks_%s_m10_causal_heads.npz" % name), **{k: np.array(v) for k, v in trow.items()})
            np.savez(os.path.join(td, "preroll_offsets_%s.npz" % name), **{k: np.array(v) for k, v in offs.items()})
            np.savez(os.path.join(td, "rule_log_pre_%s.npz" % name), ci=np.array([0]), t0=np.array([0]))
            json.dump({"preroll_steps": 30}, open(os.path.join(td, "preroll_log_%s.json" % name), "w"))
    return root, join, grid


def make_tests_file(tmp, text="5 passed in 1.0s\n"):
    p = os.path.join(str(tmp), "dev_pytest.out")
    with open(p, "w") as f:
        f.write(text)
    return p


def run(tmp, root, join, grid, out, tests="OK", seeds="0,1,2"):
    tp = make_tests_file(tmp) if tests == "OK" else tests
    argv = ["read", "--root", root, "--join", join, "--grid", grid, "--kit-dir", KIT, "--out-dir", os.path.join(str(tmp), out), "--seeds", seeds]
    if tp:
        argv += ["--tests-out", tp]
    rc = T.main(argv)
    od = os.path.join(str(tmp), out)
    res = json.load(open(os.path.join(od, T.RESULT_NAME))) if os.path.isfile(os.path.join(od, T.RESULT_NAME)) else None
    ctl = json.load(open(os.path.join(od, T.CONTROLS_NAME))) if os.path.isfile(os.path.join(od, T.CONTROLS_NAME)) else None
    return rc, res, ctl, od


def edit_log(root, tag, m, fn):
    p = T.log_path(root, tag, m)
    z = {k: np.array(v) for k, v in np.load(p).items()}
    z = fn(z)
    np.savez(p, **z)


def failed_and_withheld(rc, res, ctl, name):
    assert rc == 2, (rc, ctl)
    assert not ctl["controls"][name]["PASS"], ctl["controls"][name]
    assert res["values"] is None and res["name"] is None


# ------------------------------------------------------------------------------------------------------------ constants, names, units
def test_constants():
    assert T.CARD == "PREREG_V8F_PROOF_ADD143" and T.PRE == "CONTACTRH10KC"
    assert (T.TAG_H, T.TAG_U) == ("H_RQ50K_P30", "U_RQ50K_P30") and T.BOOT_SEED == 143
    assert T.R3_SPEC == ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2)


def iv(lo, hi):
    return {"value": (lo + hi) / 2, "interval": [lo, hi]}


def vals(q1, q2, q3, nx=19, ns=51):
    return {"Q1": q1, "Q2": q2, "Q3": q3, "n_X": nx, "n_S": ns}


def test_names_each_reachable():
    z, p = iv(-1.0, 1.0), iv(0.2, 0.8)
    assert T.derive_name(vals(z, z, p))[0] == "CONTACTRH10KC-SAME-READ"
    assert T.derive_name(vals(p, z, z))[0] == "CONTACTRH10KC-MISREAD"
    assert T.derive_name(vals(z, p, p))[0] == "CONTACTRH10KC-BOTH"
    assert T.derive_name(vals(z, z, z))[0] == "CONTACTRH10KC-UNRESOLVED"
    assert T.derive_name(vals(z, z, p, nx=9))[0] == "CONTACTRH10KC-UNRESOLVED"
    assert T.derive_name(vals(z, z, p, ns=9))[0] == "CONTACTRH10KC-UNRESOLVED"


def test_own_speed_step_one_from_own_start_not_origin():
    cl = np.zeros((30, 2))
    cl[:, 0] = DP + 1.5 * np.arange(1, 31)
    p0 = np.array([DP, 0.0])
    assert abs(T.own_speed_at(cl, p0, 1, None) - 1.5 / DT) < 1e-12                  # chord from p0
    assert abs(T.own_speed_at(cl, np.zeros(2), 1, None) - (DP + 1.5) / DT) < 1e-12    # the fault: step 1 taken from (0, 0) reads a different speed
    assert abs(T.own_speed_at(cl, p0, 10, None) - 1.5 / DT) < 1e-12
    assert abs(T.own_speed_at(cl, p0, 0, 9.25) - 9.25) < 1e-12
    assert np.isnan(T.own_speed_at(cl, p0, 0, None))


# ------------------------------------------------------------------------------------------------------------ end to end
def test_controls_pass_and_values(tmp_path):
    root, join, grid = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "o1")
    assert rc == 0 and ctl["all_pass"], ctl
    assert set(ctl["controls"]) == {"c0", "c1", "c2", "c3", "c4", "c5"}
    v = res["values"]
    assert v["n_X"] == 12 and v["n_S"] == 12 and v["n_Y"] == 12
    assert abs(v["Q1"]["value"] - 2.0) < 1e-9
    assert abs(v["Q2"]["value"]) < 1e-12
    assert abs(v["Q3"]["value"] - 0.5) < 1e-9            # v0 - v_L: RH 8.0 - 4.0, UE 7.5 - 4.0
    assert abs(v["beside_Q3_minus_c_src"]["value"]) < 1e-12
    assert res["name"] == T.N_BOTH
    assert os.path.isfile(os.path.join(od, T.VERDICT_NAME))
    c2 = ctl["controls"]["c2"]
    assert c2["n_sample_v0_checked"] > 0 and c2["max_abs_v0_dev_mps"] <= 1e-6
    c3 = ctl["controls"]["c3"]
    assert c3["n3_planted_start_offset"]["PASS"] and c3["n2_beside_not_gating"]["PASS"]
    assert ctl["controls"]["c0"]["n_windows_grid"] == 24


def test_same_read_reachable_end_to_end(tmp_path):
    root, join, grid = build(tmp_path, err_x=1.0, err_s_h=1.0)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "o2")
    assert rc == 0 and abs(res["values"]["Q1"]["value"]) < 1e-9
    assert res["name"] == T.N_SAME


def test_c5_fails_end_to_end_when_own_speed_is_identical(tmp_path):
    root, join, grid = build(tmp_path, exs_h=0.5)          # same own speed in S in both arms: Q3 = 0 (c5 fails, so values are withheld)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "o3")
    failed_and_withheld(rc, res, ctl, "c5")


def test_q2_presence_misses(tmp_path):
    root, join, grid = build(tmp_path, present_x=False)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "o4")
    assert rc == 0
    v = res["values"]
    assert abs(v["Q2"]["value"] - 1.0) < 1e-12 and v["Q1"]["value"] is None


# ------------------------------------------------------------------------------------------------------------ c0
def test_c0_fails_on_a_changed_argv(tmp_path):
    root, join, grid = build(tmp_path)
    p = os.path.join(root, "H_RQ50K_P30_s1", "argv.json")
    a = json.load(open(p))
    a["argv"][-1] = "--device cpu"
    json.dump(a, open(p, "w"))
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c0a")
    failed_and_withheld(rc, res, ctl, "c0")


def test_c0_fails_on_a_failed_done_file(tmp_path):
    root, join, grid = build(tmp_path)
    json.dump({"rc": 1}, open(os.path.join(root, "U_RQ50K_P30_s2", "RUN_DONE.json"), "w"))
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c0b")
    failed_and_withheld(rc, res, ctl, "c0")


def test_c0_fails_on_a_missing_preroll_log(tmp_path):
    root, join, grid = build(tmp_path)
    os.remove(os.path.join(root, "H_RQ50K_P30_s0", "ticks", "rule_log_pre_H_RQ50K_P30_s0.npz"))
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c0c")
    failed_and_withheld(rc, res, ctl, "c0")


def test_c0_fails_on_a_wrong_rule_log_window_count(tmp_path):
    root, join, grid = build(tmp_path)
    edit_log(root, T.TAG_U, 0, lambda z: {k: v[~((z["ci"] == 5) & (z["t0"] == 30))] for k, v in z.items()})
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c0d")
    failed_and_withheld(rc, res, ctl, "c0")


def test_c0_fails_when_the_two_dumps_are_identical(tmp_path):
    root, join, grid = build(tmp_path)
    dh, du = T.dump_folder(root, T.TAG_H, 1), T.dump_folder(root, T.TAG_U, 1)
    for f in os.listdir(dh):
        shutil.copy(os.path.join(dh, f), os.path.join(du, f))
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c0e")
    assert rc == 2 and not ctl["controls"]["c0"]["PASS"] and res["values"] is None


def test_c0_fails_on_a_nonfinite_plan(tmp_path):
    root, join, grid = build(tmp_path)
    f = os.path.join(T.dump_folder(root, T.TAG_H, 2), "ep003.npz")
    z = {k: np.array(v) for k, v in np.load(f).items()}
    z["cl"][0, 5, 0] = np.nan
    np.savez(f, **z)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c0f")
    assert rc == 2 and not ctl["controls"]["c0"]["PASS"] and res["values"] is None


def test_refused_when_the_windows_differ_from_the_grid_or_the_tick_file_misses(tmp_path):
    root, join, grid = build(tmp_path)
    f = os.path.join(T.dump_folder(root, T.TAG_U, 1), "ep000.npz")
    z = {k: np.array(v) for k, v in np.load(f).items()}
    z["ws"] = z["ws"] + 1
    np.savez(f, **z)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "r1")
    assert rc == 3 and res is None
    os.makedirs(os.path.join(str(tmp_path), "b"))
    root, join, grid = build(os.path.join(str(tmp_path), "b"))
    os.remove(T.tick_path(root, T.TAG_H, 0))
    rc, res, ctl, od = run(tmp_path, root, join, grid, "r2")
    assert rc == 3 and res is None


# ------------------------------------------------------------------------------------------------------------ c1
def test_c1_fails_when_the_preroll_patch_is_not_installed(tmp_path, monkeypatch):
    root, join, grid = build(tmp_path)
    monkeypatch.setattr(T, "engines_patched", lambda: False)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c1a")
    failed_and_withheld(rc, res, ctl, "c1")


def test_c1_fails_when_the_engine_cells_differ_from_the_recomputation(tmp_path, monkeypatch):
    root, join, grid = build(tmp_path)
    T.bind_longroll()
    orig = _H.window_cells

    def bad(g, cl, L):
        out = dict(orig(g, cl, L))
        out["C"] = np.where(np.isfinite(out["C"]), 0.0, out["C"])
        return out
    bad.__qualname__ = orig.__qualname__
    monkeypatch.setattr(_H, "window_cells", bad)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c1b")
    failed_and_withheld(rc, res, ctl, "c1")


# ------------------------------------------------------------------------------------------------------------ c2
def test_c2_fails_on_an_inconsistent_gap(tmp_path):
    root, join, grid = build(tmp_path)
    for m in SEEDS:
        for tag in (T.TAG_H, T.TAG_U):
            edit_log(root, tag, m, lambda z: dict(z, g0=z["g0"] + 0.5))
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c2a")
    failed_and_withheld(rc, res, ctl, "c2")


def test_c2_fails_on_a_missing_tick(tmp_path):
    root, join, grid = build(tmp_path)
    edit_log(root, T.TAG_H, 1, lambda z: {k: v[z["ci"] != 0] for k, v in z.items()})
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c2b")
    assert rc == 2 and not ctl["controls"]["c2"]["PASS"] and res["values"] is None


def test_c2_fails_when_v0_is_the_speed_at_t0(tmp_path):
    root, join, grid = build(tmp_path, v0_at_t0=True)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c2c")
    failed_and_withheld(rc, res, ctl, "c2")
    assert any("v0" in p for p in ctl["controls"]["c2"]["problems"])


def test_c2_fails_when_the_tick_file_speed_differs(tmp_path):
    root, join, grid = build(tmp_path, tick_v_bias=0.5)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c2d")
    failed_and_withheld(rc, res, ctl, "c2")


def test_c2_fails_when_the_rule_log_holds_a_preroll_row(tmp_path):
    root, join, grid = build(tmp_path)

    def add(z):
        out = {}
        for k, v in z.items():
            extra = {"ci": 0, "t0": -30, "j": -30, "k": 0}.get(k, v[0])
            out[k] = np.concatenate([v, np.array([extra], dtype=v.dtype)])
        return out
    edit_log(root, T.TAG_H, 0, add)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c2e")
    failed_and_withheld(rc, res, ctl, "c2")


# ------------------------------------------------------------------------------------------------------------ c3
def test_c3_planted_reads_plus_two(tmp_path):
    root, join, grid = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c3a")
    c3 = ctl["controls"]["c3"]
    assert c3["PASS"] and c3["planted_deviation"] < 1e-6 and c3["unchanged_reads_main_value"]


def test_c3_detects_a_tool_that_ignores_the_plant(tmp_path, monkeypatch):
    root, join, grid = build(tmp_path, err_x=2.0)
    monkeypatch.setattr(T, "plant_logs", lambda runs, seeds, tabs: {m: runs[("H", m)]["log"] for m in seeds})
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c3b")
    failed_and_withheld(rc, res, ctl, "c3")


def test_c3_n3_fails_when_step_one_is_taken_from_the_origin(tmp_path, monkeypatch):
    if ORIG_ENGINES[2]:
        pytest.skip("the engines were patched before this module was imported")
    T.bind_longroll()
    monkeypatch.setattr(T.N4, "live", lambda x: None)                      # the patched engines then read step 1 from (0, 0)
    plant = T.N4.planted_n3(_H, _H.RH, ORIG_ENGINES[0], ORIG_ENGINES[1])
    assert plant["PASS"] is False
    assert T.n3_judge(plant)["PASS"] is False
    assert T.n3_judge(None)["PASS"] is False
    assert T.n3_judge({"PASS": True})["PASS"] is True


def test_c3_withholds_when_the_n3_plant_fails(tmp_path, monkeypatch):
    root, join, grid = build(tmp_path)
    T.bind_longroll()
    monkeypatch.setitem(T.N4.PLANT, "PASS", False)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c3c")
    failed_and_withheld(rc, res, ctl, "c3")
    assert not ctl["controls"]["c3"]["n3_planted_start_offset"]["PASS"]


def test_n3_of_record_passes_on_the_real_engines():
    T.bind_longroll()
    assert T.N4.PLANT.get("PASS") is True and T.engines_patched()


# ------------------------------------------------------------------------------------------------------------ c4, c5, refusals
def test_failing_c4_withholds_values(tmp_path):
    root, join, grid = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c4a", tests=make_tests_file(tmp_path, "3 passed, 1 failed in 1s\n"))
    failed_and_withheld(rc, res, ctl, "c4")
    assert "withheld" in open(os.path.join(od, T.VERDICT_NAME)).read()


def test_no_tests_file_withholds(tmp_path):
    root, join, grid = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c4b", tests=None)
    assert rc == 2 and res["values"] is None


def test_c4_parser(tmp_path):
    assert T.ctl_c4(make_tests_file(tmp_path, "12 passed in 3s\n"))["PASS"]
    assert not T.ctl_c4(make_tests_file(tmp_path, "12 passed, 1 error in 3s\n"))["PASS"]
    assert not T.ctl_c4(None)["PASS"]


def _tabs(c_h, c_u, c2_h, c2_u):
    a = {"i": np.array([1, 2, 3]), "clos": np.array(c_h, float), "clos2": np.array(c2_h, float), "ep": np.array([0, 1, 2])}
    b = {"i": np.array([1, 2, 3]), "clos": np.array(c_u, float), "clos2": np.array(c2_u, float), "ep": np.array([0, 1, 2])}
    return {0: {"S": a, "S_U": b}}


def test_c5_passes_on_identity_and_fails_otherwise():
    assert T.ctl_c5(_tabs([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [0.5, 2.0, 3.0]))["PASS"]
    assert not T.ctl_c5(_tabs([0.1, 0.2, 0.3], [0.1, 0.2, 0.31], [1.0, 2.0, 3.0], [0.5, 2.0, 3.0]))["PASS"]
    assert not T.ctl_c5(_tabs([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))["PASS"]


def test_c5_fails_end_to_end_when_c_src_differs(tmp_path):
    root, join, grid = build(tmp_path, c_src_u=-1.2)
    rc, res, ctl, od = run(tmp_path, root, join, grid, "c5a")
    failed_and_withheld(rc, res, ctl, "c5")


def test_q3_of_record_is_v0_minus_vl():
    idx = T.make_draws(3, 50, 1)
    tabs = _tabs([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    assert abs(T.paired_closing(tabs, "clos2", 3, idx)["value"] - 1.0) < 1e-12


def test_refuses_missing_inputs_and_existing_outputs(tmp_path):
    root, join, grid = build(tmp_path)
    rc, res, ctl, od = run(tmp_path, root, join, os.path.join(str(tmp_path), "nogrid.json"), "x1")
    assert rc == 3
    rc, res, ctl, od = run(tmp_path, root, join, grid, "x2")
    assert rc == 0
    rc2, _, _, _ = run(tmp_path, root, join, grid, "x2")
    assert rc2 == 3
