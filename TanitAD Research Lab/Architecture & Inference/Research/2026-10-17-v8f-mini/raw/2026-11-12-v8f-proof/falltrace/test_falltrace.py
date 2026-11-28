# test_falltrace.py -- synthetic data only (no real dump, rollout, rule log or result file is opened).
import json
import os
import re
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_fall as X  # noqa: E402

BANNED = tuple(w.replace("_", "") for w in ("vo_id", "super_seded", "re_tracted", "for_merly", "cor_rection", "with_drawn", "no_ longer", "closed_ loop"))   # written in parts: a scan reads this file too
FILES = ("export_fall.py", "podrun_export_fall.sh", "push_falltrace.sh", "pull_falltrace.sh", "FALLTRACE_DEFS.md", "test_falltrace.py")
ARMS = ["VD5c", "F2"]


@pytest.fixture(scope="module")
def eng():
    return X.get_engine("")


def _world(tmp_path, fault="", arms=ARMS):
    root = str(tmp_path)
    exp, lv, nl, nw = X.make_synth(root, arms, fault)
    a = types.SimpleNamespace(pull=os.path.join(root, "pull"), grid_dir=os.path.join(root, "grid"), join=os.path.join(root, "join.npz"))
    return a, exp, lv, nl, nw, root


def _run(eng, tmp_path, fault="", arms=ARMS, banked=True):
    R, H = eng
    a, exp, lv, nl, nw, root = _world(tmp_path, fault, arms)
    return X.run(R, H, a, arms, expect_c=exp, banked=lv if banked else None, n_lead=nl, n_win=nw), root


def _n(**kw):
    d = {"n_acted": 10, "n_acted_no_label_lead": 0, "n_acted_label_lead": 10, "n_gap_low": 0, "n_slope_mismatch": 0}
    d.update(kw)
    return d


# ---------------------------------------------------------------- classes and their order
def test_each_class_alone():
    assert X.classify(_n(n_acted=4, n_acted_label_lead=4)) == "NO-RULE-ACTION"
    assert X.classify(_n(n_acted=0, n_acted_label_lead=0)) == "NO-RULE-ACTION"
    assert X.classify(_n(n_acted_no_label_lead=5, n_acted_label_lead=5)) == "FALSE-PRESENCE"
    assert X.classify(_n(n_gap_low=5)) == "GAP-LOW"
    assert X.classify(_n(n_slope_mismatch=5)) == "SLOPE"
    assert X.classify(_n(n_gap_low=4, n_slope_mismatch=4)) == "MIXED"
    assert X.classify(_n()) == "MIXED"


def test_class_order_is_first_match():
    # everything true at once: the first class of the list wins, then each next one once the earlier ones are false
    assert X.classify(_n(n_acted=3, n_acted_no_label_lead=3, n_acted_label_lead=0, n_slope_mismatch=3)) == "NO-RULE-ACTION"
    assert X.classify(_n(n_acted_no_label_lead=6, n_acted_label_lead=4, n_gap_low=4, n_slope_mismatch=9)) == "FALSE-PRESENCE"
    assert X.classify(_n(n_gap_low=9, n_slope_mismatch=9)) == "GAP-LOW"
    assert X.classify(_n(n_gap_low=4, n_slope_mismatch=5)) == "SLOPE"
    assert X.CLASSES == ("NO-RULE-ACTION", "FALSE-PRESENCE", "GAP-LOW", "SLOPE", "MIXED")


def test_class_thresholds_are_at_the_constants():
    assert X.NO_ACT_MIN == 5 and X.SHARE_MIN == 0.5 and X.GAP_LOW_M == 5.0
    assert X.classify(_n(n_acted=5, n_acted_label_lead=5, n_gap_low=2)) == "MIXED"          # 2 of 5 is below one half
    assert X.classify(_n(n_acted=6, n_acted_label_lead=6, n_gap_low=3)) == "GAP-LOW"        # 3 of 6 is one half


def test_numbers_from_rows(eng):
    R, H = eng
    J = {"off": np.array([0]), "n": np.array([60]), "lead_valid": np.ones(60, np.int64), "lead_gap": np.concatenate([np.full(30, 15.0), np.full(30, 40.0)])}
    j = np.arange(26, 34)
    rows = {"ci": np.zeros(8, np.int64), "t0": np.full(8, 20, np.int64), "j": j, "present": np.ones(8, bool), "changed": np.ones(8, bool), "g_src": np.full(8, 10.0), "c_src": np.full(8, -1.0)}
    nums, outside = X.window_numbers(rows, J, np.array([0, 0]), np.array([20, 21]))
    assert outside == 0 and nums[1]["n_steps"] == 0 and X.classify(nums[1]) == "NO-RULE-ACTION"
    n = nums[0]
    assert n["n_steps"] == 8 and n["n_fed"] == 8 and n["n_acted"] == 8 and n["n_acted_label_lead"] == 8
    # fed 10 m against label 15 m (frames 26 to 29) is -5 m: not below -5; against 40 m (frames 30 to 33) it is -30 m
    assert n["n_gap_low"] == 4
    assert n["n_closing_fed"] == 8
    # label gap constant inside each half, one step of +25 m at frame 29 to 30 (above the 3 m bound: slope undefined there)
    assert n["n_slope_mismatch"] == 7 and abs(n["gap_diff_median_m"] - (-17.5)) < 1e-9


def test_sign_of_closing_input_positive_opens():
    # a closing input of +2 m/s (gap opens) must never count as "the gap closes"
    J = {"off": np.array([0]), "n": np.array([60]), "lead_valid": np.ones(60, np.int64), "lead_gap": np.full(60, 15.0)}
    rows = {"ci": np.zeros(10, np.int64), "t0": np.zeros(10, np.int64), "j": np.arange(10), "present": np.ones(10, bool), "changed": np.ones(10, bool), "g_src": np.full(10, 15.0), "c_src": np.full(10, 2.0)}
    nums, _ = X.window_numbers(rows, J, np.array([0]), np.array([0]))
    assert nums[0]["n_slope_mismatch"] == 0 and X.classify(nums[0]) == "MIXED"


def test_rule_acted_needs_present_and_changed():
    J = {"off": np.array([0]), "n": np.array([60]), "lead_valid": np.ones(60, np.int64), "lead_gap": np.full(60, 15.0)}
    pres = np.array([True] * 6 + [False] * 4)
    rows = {"ci": np.zeros(10, np.int64), "t0": np.zeros(10, np.int64), "j": np.arange(10), "present": pres, "changed": np.ones(10, bool), "g_src": np.full(10, 15.0), "c_src": np.zeros(10)}
    nums, _ = X.window_numbers(rows, J, np.array([0]), np.array([0]))
    assert nums[0]["n_acted"] == 6 and nums[0]["n_fed"] == 6 and nums[0]["n_steps"] == 10


# ---------------------------------------------------------------- the run on synthetic rollouts
def test_clean_run_counts(eng, tmp_path):
    (rc, lines, rec), _ = _run(eng, tmp_path)
    assert rc == 0 and lines[-1] == "ZZ FALLTRACE DONE 2 20"
    t = rec["table_2x2_F"]["VD5c"]
    assert t["vsRLc"]["0"] == {"both": 1, "x_only": 5, "ref_only": 1, "neither": 13}
    assert t["vsRLc"]["pooled"] == {"both": 2, "x_only": 10, "ref_only": 2, "neither": 26}
    assert t["vsRR"]["0"] == {"both": 2, "x_only": 4, "ref_only": 1, "neither": 13}
    for x in ARMS:
        assert {c: rec["class_counts"][x]["excess"][c] for c in X.CLASSES} == {"NO-RULE-ACTION": 2, "FALSE-PRESENCE": 2, "GAP-LOW": 2, "SLOPE": 2, "MIXED": 2}
        assert rec["class_counts"][x]["neither"]["GAP-LOW"] == 2 and rec["class_counts"][x]["neither"]["NO-RULE-ACTION"] == 24
    assert rec["class_counts_pooled_over_arms"]["excess"]["SLOPE"] == 4
    assert any(ln.startswith("FALLTRACE 2x2 VD5c s0 vsRLc both 1 x_only 5 rlc_only 1 neither 13") for ln in lines)
    assert any(ln.startswith("FALLTRACE CLASS pooled neither n 52") for ln in lines)


def test_windows_table_and_coverage(eng, tmp_path):
    (rc, lines, rec), _ = _run(eng, tmp_path)
    w = [r for r in rec["windows"] if r["arm"] == "VD5c" and r["seed"] == 0]
    assert len(w) == 20 and all(r["clip"] == "aaaaaaaa-0000" for r in w)
    by = {r["ws"]: r for r in w}
    assert [by[i]["class"] for i in range(5)] == ["NO-RULE-ACTION", "FALSE-PRESENCE", "GAP-LOW", "SLOPE", "MIXED"]
    assert by[1]["n_acted_no_label_lead"] == 8 and by[1]["n_acted"] == 10
    assert by[2]["gap_diff_median_m"] == -7.0 and by[3]["closing_input_median_mps"] == -2.0 and by[3]["label_slope_median_mps"] == 0.0
    assert by[6]["group_vs_RLc"] == "ref_only" and by[5]["group_vs_RLc"] == "both" and by[0]["group_vs_RR"] == "both"
    assert rec["rule_log_coverage"]["windows_without_rule_rows"]["VD5c_s0"] == 0
    assert rec["rule_log_coverage"]["acted_rows_without_label_lead"]["VD5c_s0"]["of_which_no_label_lead"] == 8
    assert rec["control"]["PASS"] is True and "RR-F" in rec["control"]["banked_checked"]


def test_selection_is_both_log_files(eng, tmp_path):
    # the rows of one window are split over the tick log and the later-look log; dropping the second file must change the counts
    (rc, lines, rec), root = _run(eng, tmp_path)
    key = "I10%s_s0" % X.RUNS["VD5c"]
    os.remove(os.path.join(root, "pull", key, "ticks", "rule_log_mid_%s.npz" % key))
    R, H = eng
    a = types.SimpleNamespace(pull=os.path.join(root, "pull"), grid_dir=os.path.join(root, "grid"), join=os.path.join(root, "join.npz"))
    exp, lv, nl, nw = {"RR": 0.1, "RLc": 0.0, "VD5c": 0.05, "F2": 0.05}, None, 40, 40
    with pytest.raises(SystemExit) as e:
        X.run(R, H, a, ARMS, expect_c=exp, banked=lv, n_lead=nl, n_win=nw)
    assert "MISSING-RULE-LOG" in str(e.value)


# ---------------------------------------------------------------- the control
def test_planted_contact_fault_fails_and_writes_nothing(eng, tmp_path):
    (rc, lines, rec), _ = _run(eng, tmp_path, "contact")
    assert rc == 3 and rec is None and len(lines) == 1 and lines[0].startswith("ZZ FALLTRACE CONTROL FAIL rate-C-VD5c")


def test_banked_f_mismatch_fails(eng, tmp_path):
    (rc, lines, rec), _ = _run(eng, tmp_path, "banked")
    assert rc == 3 and rec is None and lines[0].startswith("ZZ FALLTRACE CONTROL FAIL banked-F-RR")


def test_without_banked_file_only_contact_is_checked(eng, tmp_path):
    (rc, lines, rec), _ = _run(eng, tmp_path, "banked", banked=False)
    assert rc == 0 and rec["control"]["banked_checked"] == []


def test_wrong_counts_fail(eng, tmp_path):
    R, H = eng
    a, exp, lv, nl, nw, _ = _world(tmp_path)
    rc, lines, rec = X.run(R, H, a, ARMS, expect_c=exp, banked=lv, n_lead=nl + 1, n_win=nw)
    assert rc == 3 and lines == ["ZZ FALLTRACE CONTROL FAIL n-lead 40 != 41"] and rec is None


def test_expected_contact_values_of_the_brief():
    assert X.EXPECT_C == {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "F2": 0.0138, "F3": 0.0185, "F4": 0.0123, "E5c": 0.0138}
    assert X.N_LEAD == 650 and X.N_WIN == 1271


def test_load_banked_finds_nested_levels(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"payload": {"x": [1, {"levels": {"RR": {"C": 0.0954, "U": 0.1, "F": 0.2}, "G2": {"C": 0.01, "F": 0.3}}}]}}), encoding="utf-8")
    b, used = X.load_banked([str(p), str(tmp_path / "absent.json"), ""])
    assert b["RR"] == {"C": 0.0954, "F": 0.2} and b["G2"]["F"] == 0.3 and used == ["r.json"]
    assert X.load_banked([])[0] == {}


# ---------------------------------------------------------------- arguments
def test_default_and_extra_arms():
    assert X.parse_arms("") == list(X.ARMS_DEFAULT) == ["VD5c", "F2", "F3", "F4", "E5c"]
    assert X.parse_arms(" G2, G3,G4 ,GEc,E3c,G2") == ["G2", "G3", "G4", "GEc", "E3c"]
    assert all(x in X.ARMS_KNOWN for x in ("G2", "G3", "G4", "GEc", "E3c", "VD5c", "F2", "F3", "F4", "E5c"))
    assert X.RUNS["G2"] == "RQ50KF5P30Hb20gG2" and X.RUNS["E3c"] == "RQ50KF5P30Hb20gE3c" and X.RUNS["E5c"] == "RQ50KF5P30Hb20eE5c" and X.RUNS["F3"] == "RQ50KF5P30Hb20fF3"


def test_main_stops_on_bad_arguments(capsys, tmp_path):
    base = ["--pull", "p", "--grid-dir", "g", "--join", "j", "--code", "c"]
    assert X.main(base) == 2 and capsys.readouterr().out.strip() == "ZZ FALLTRACE STOP missing --out"
    assert X.main(base + ["--out", str(tmp_path / "o"), "--arms", "VD5c,NOPE"]) == 2 and "unknown arm NOPE" in capsys.readouterr().out
    assert X.main(base + ["--out", str(tmp_path)]) == 2 and "output exists" in capsys.readouterr().out


def test_main_end_to_end_with_a_g_arm_list(eng, tmp_path, monkeypatch, capsys):
    arms = ["G2", "GEc"]
    a, exp, lv, nl, nw, root = _world(tmp_path, "", arms)
    monkeypatch.setattr(X, "EXPECT_C", exp)
    monkeypatch.setattr(X, "N_LEAD", nl)
    monkeypatch.setattr(X, "N_WIN", nw)
    monkeypatch.setattr(X, "get_engine", lambda code: eng)
    out = os.path.join(root, "out_fall")
    bj = os.path.join(root, "banked.json")
    json.dump({"levels": lv}, open(bj, "w"))
    rc = X.main(["--pull", a.pull, "--grid-dir", a.grid_dir, "--join", a.join, "--code", "x", "--out", out, "--arms", "G2,GEc", "--banked", bj])
    lines = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and lines[-1] == "ZZ FALLTRACE DONE 2 20"
    rec = json.load(open(os.path.join(out, "FALLTRACE.json"), encoding="utf-8"))
    assert rec["arms"] == arms and len(rec["windows"]) == 80
    s = json.dumps(rec)
    assert len(s) < 400_000 and "/workspace" not in s and ":\\" not in s


def test_main_control_fail_writes_no_folder(eng, tmp_path, monkeypatch, capsys):
    a, exp, lv, nl, nw, root = _world(tmp_path, "contact")
    monkeypatch.setattr(X, "EXPECT_C", exp)
    monkeypatch.setattr(X, "N_LEAD", nl)
    monkeypatch.setattr(X, "N_WIN", nw)
    monkeypatch.setattr(X, "get_engine", lambda code: eng)
    out = os.path.join(root, "out_fall")
    rc = X.main(["--pull", a.pull, "--grid-dir", a.grid_dir, "--join", a.join, "--code", "x", "--out", out, "--arms", "VD5c,F2"])
    assert rc == 3 and not os.path.exists(out) and capsys.readouterr().out.strip().startswith("ZZ FALLTRACE CONTROL FAIL rate-C-VD5c")


# ---------------------------------------------------------------- files
def test_scripts_syntax_line_ends_and_wording():
    bash = shutil.which("bash")
    for f in FILES:
        b = open(os.path.join(HERE, f), "rb").read()
        if f.endswith(".sh"):
            assert b"\r" not in b, f
            assert subprocess.run([bash, "-n", os.path.join(HERE, f)], capture_output=True, encoding="utf-8").returncode == 0, f
        t = b.decode("utf-8").lower()
        if f != "test_falltrace.py":
            for w in BANNED:
                assert w not in t, (f, w)
            for m in re.finditer(r"driving performance", t):
                assert t[max(0, m.start() - 4):m.start()] == "not ", f


def test_dry_mode_makes_nothing():
    r = subprocess.run([shutil.which("bash"), os.path.join(HERE, "podrun_export_fall.sh")], env=dict(os.environ, DRY="1", ARMS="G2,GEc"), capture_output=True, encoding="utf-8")
    assert r.returncode == 0 and re.search(r"ZZ PODRUN-FALLTRACE DRY missing \d+$", r.stdout.strip()) and "MISSING" in r.stdout and "refused if it exists" in r.stdout
    r2 = subprocess.run([shutil.which("bash"), os.path.join(HERE, "podrun_export_fall.sh")], env=dict(os.environ, DRY="1", ARMS="NOPE"), capture_output=True, encoding="utf-8")
    assert r2.returncode == 3 and "UNKNOWN-ARM NOPE" in r2.stdout


def test_pod_script_uses_the_new_root_and_markers():
    t = open(os.path.join(HERE, "podrun_export_fall.sh"), encoding="utf-8").read()
    for s in ("/workspace/v8proof/falltrace", "START_FALLTRACE.txt", "DONE_FALLTRACE.txt", "STOP_FALLTRACE.txt", "CUDA_VISIBLE_DEVICES=-1", "OMP_NUM_THREADS=4", "/workspace/venv/bin/python", "ARMS=${ARMS:-"):
        assert s in t, s
    for f in ("push_falltrace.sh", "pull_falltrace.sh"):
        assert "/workspace/v8proof/falltrace" in open(os.path.join(HERE, f), encoding="utf-8").read()


def test_path_lengths():
    for f in FILES:
        assert len(os.path.join(HERE, f).replace("\\", "/")) <= 200
