"""test_preserr.py -- card ADD116 (PRESERR-T0), lead-written: the tool on small synthetic worlds with known behaviour (clean, a fault per control, a world where a lower
threshold has room and two where it must not), the name rule on hand-made payloads, the label test of a join row, the area under the curve, and the two stages from the
command line. No pod result file is opened."""
import argparse
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "preserr_t0.py")
sys.path.insert(0, HERE)
import preserr_t0 as T  # noqa: E402

NS = argparse.Namespace


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def controls(tmp_path, name, n_boot=200, **kw):
    t1, jp, tp = T.make_world(str(tmp_path / name), **kw)
    return T.run_controls(NS(t1=t1, join=jp, tests=tp, n_boot=n_boot))


def test_selftest(tmp_path):
    assert T.selftest(str(tmp_path / "st")) == 0


def test_clean_world_passes_and_has_the_planted_rates(tmp_path):
    r = controls(tmp_path, "w", miss_p=(0.30, 0.10), extra_p=0.05)
    assert r["all_pass"] is True and sorted(r["controls"]) == sorted(T.CONTROLS)
    v = r["payload"]
    s0, s1 = v["per_seed"]
    assert s0["n_frames"] == s1["n_frames"] > 300 and s0["n_L"] == s1["n_L"] > 100
    assert abs(s0["M_0.5"] - 0.30) < 0.08 and abs(s1["M_0.5"] - 0.10) < 0.06
    assert abs(s0["X_0.5"] - 0.05) < 0.04 and abs(s1["X_0.5"] - 0.05) < 0.04
    P = v["pooled"]
    assert abs(P["M_0.5"]["value"] - 0.5 * (s0["M_0.5"] + s1["M_0.5"])) < 1e-12
    lo, hi = P["M_0.5"]["interval"]
    assert lo <= P["M_0.5"]["value"] <= hi and hi - lo > 0.0
    i5 = T.GRID.index(0.5)
    assert abs(s0["M_by_t"][i5] - s0["M_0.5"]) < 1e-12 and abs(P["by_t"][i5]["M_ratio"]["value"] - 1.0) < 1e-12 and abs(P["by_t"][i5]["X_ratio"]["value"] - 1.0) < 1e-12
    ms = [x["M"] for x in P["by_t"]]
    xs = [x["X"] for x in P["by_t"]]
    assert all(a <= b + 1e-12 for a, b in zip(ms, ms[1:])) and all(a >= b - 1e-12 for a, b in zip(xs, xs[1:]))
    assert s0["auc"] > 0.8 and sum(b["n_L"] for b in s0["bins"]) == s0["n_L"] and sum(b["n_miss"] for b in s0["bins"]) == round(s0["M_0.5"] * s0["n_L"])


def test_same_inputs_give_the_same_values(tmp_path):
    t1, jp, tp = T.make_world(str(tmp_path / "w"))
    a = T.run_controls(NS(t1=t1, join=jp, tests=tp, n_boot=100))["payload"]
    b = T.run_controls(NS(t1=t1, join=jp, tests=tp, n_boot=100))["payload"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("fault", ["h0", "h1", "h2", "h3", "h4", "h5"])
def test_each_fault_turns_its_control_red_and_holds_the_values_back(tmp_path, fault):
    r = controls(tmp_path, "f", n_boot=20, fault=fault)
    assert r["all_pass"] is False and r["controls"][fault]["pass"] is False and r["controls"][fault]["failures"] and "payload" not in r


def test_room_world_gives_room(tmp_path):
    r = controls(tmp_path, "room", miss_p=(0.30, 0.30), extra_p=0.05, sep=4.0)
    nm = T.names_of(r["payload"])
    assert nm["THRESHOLD"] == "PRESERR-T0-THRESHOLD-HAS-ROOM" and nm["t_star"] == 0.3


@pytest.mark.parametrize("sep", [1.0, 40.0])
def test_worlds_without_room_give_no_room(tmp_path, sep):
    r = controls(tmp_path, "noroom", miss_p=(0.30, 0.30), extra_p=0.05, sep=sep)
    nm = T.names_of(r["payload"])
    assert nm["THRESHOLD"] == "PRESERR-T0-THRESHOLD-HAS-NO-ROOM"


def hand(m_ratio_04=(0.4, [0.3, 0.6]), x_ratio_04=1.5, m05=0.2, x05=0.05, ml=0.3, ms=0.1, d=(0.05, 0.3)):
    by = []
    for t in T.GRID:
        if abs(t - 0.4) < 1e-9:
            by.append({"t": t, "M": 0.0, "X": 0.0, "M_ratio": {"value": m_ratio_04[0], "interval": m_ratio_04[1]}, "X_ratio": {"value": x_ratio_04, "interval": [1.0, 2.0]}})
        else:
            by.append({"t": t, "M": 0.0, "X": 0.0, "M_ratio": {"value": 0.9, "interval": [0.8, 1.0]}, "X_ratio": {"value": 1.0, "interval": [1.0, 1.0]}})
    return {"pooled": {"M_0.5": {"value": m05}, "X_0.5": {"value": x05}, "by_t": by, "M_large_gaps": {"value": ml}, "M_small_gaps": {"value": ms}, "large_minus_small": {"value": ml - ms, "interval": list(d) if d else None}}}


def test_name_rule_on_hand_made_payloads():
    R, N = "PRESERR-T0-THRESHOLD-HAS-ROOM", "PRESERR-T0-THRESHOLD-HAS-NO-ROOM"
    assert T.names_of(hand())["THRESHOLD"] == R and T.names_of(hand())["t_star"] == 0.4
    assert T.names_of(hand(m_ratio_04=(0.5, [0.3, 0.74])))["THRESHOLD"] == R
    assert T.names_of(hand(m_ratio_04=(0.51, [0.3, 0.6])))["THRESHOLD"] == N                    # the miss cut is not reached on the grid
    assert T.names_of(hand(m_ratio_04=(0.4, [0.3, 0.75])))["THRESHOLD"] == N                    # the interval does not hold the cut
    assert T.names_of(hand(m_ratio_04=(0.4, None)))["THRESHOLD"] == N
    assert T.names_of(hand(x_ratio_04=2.01))["THRESHOLD"] == N                                  # the extra marks more than double
    assert T.names_of(hand(x_ratio_04=2.0))["THRESHOLD"] == R
    assert T.names_of(hand(m05=0.0))["THRESHOLD"] == N and T.names_of(hand(x05=0.0))["THRESHOLD"] == N
    assert T.names_of(hand())["WHERE"] == "PRESERR-T0-MISSES-AT-LARGE-GAPS"
    assert T.names_of(hand(ml=0.19, ms=0.1))["WHERE"] == "PRESERR-T0-MISSES-NOT-ORDERED-BY-GAP"
    assert T.names_of(hand(d=(-0.01, 0.3)))["WHERE"] == "PRESERR-T0-MISSES-NOT-ORDERED-BY-GAP"
    assert T.names_of(hand(ml=0.1, ms=0.3, d=(-0.3, -0.05)))["WHERE"] == "PRESERR-T0-MISSES-AT-SMALL-GAPS"
    assert T.names_of(hand(d=None))["WHERE"] == "PRESERR-T0-MISSES-NOT-ORDERED-BY-GAP"


def test_label_of_a_join_row():
    JL = {"off": np.array([0, 3]), "n": np.array([3, 2]), "lead_valid": np.array([1, 1, 0, 1, 1]), "lead_gap": np.array([10.0, 60.0, 5.0, 60.5, np.nan])}
    maps, L, g = T.label_of(JL, np.array([0, 0, 0, 1, 1, 1, 2, 0]), np.array([0, 1, 2, 0, 1, 2, 0, -1]))
    assert maps.tolist() == [True, True, True, True, True, False, False, False]
    assert L.tolist() == [True, True, False, False, False, False, False, False]
    assert g[0] == 10.0 and g[3] == 60.5 and np.isnan(g[5])


def test_area_under_the_curve():
    L = np.array([0, 0, 1, 1], bool)
    assert T.auc(np.array([0.1, 0.2, 0.8, 0.9]), L) == 1.0
    assert T.auc(np.array([0.9, 0.8, 0.2, 0.1]), L) == 0.0
    assert T.auc(np.array([0.5, 0.5, 0.5, 0.5]), L) == 0.5
    assert T.auc(np.array([0.1, 0.2]), np.array([1, 1], bool)) is None


def test_two_stages_from_the_command_line(tmp_path):
    t1, jp, tp = T.make_world(str(tmp_path / "w"))
    cf, rf = str(tmp_path / "C.json"), str(tmp_path / "R.json")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, TOOL, "--stage", "controls", "--t1", t1, "--join", jp, "--tests", tp, "--out", cf, "--n-boot", "100"], capture_output=True, encoding="utf-8", env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    lines = r.stdout.strip().splitlines()
    assert lines == ["h%d PASS" % i for i in range(6)] + ["ZZ PRESERR-T0 controls all_pass True"]
    r2 = subprocess.run([sys.executable, TOOL, "--stage", "controls", "--t1", t1, "--join", jp, "--tests", tp, "--out", cf], capture_output=True, encoding="utf-8", env=env, timeout=300)
    assert r2.returncode == 2 and "REFUSED" in r2.stdout
    r3 = subprocess.run([sys.executable, TOOL, "--stage", "names", "--controls", cf, "--out", rf], capture_output=True, encoding="utf-8", env=env, timeout=300)
    assert r3.returncode == 0 and r3.stdout.strip().splitlines()[-1] == "ZZ PRESERR-T0 names done" and len(re.findall(r"^NAME PRESERR-T0-", r3.stdout, re.M)) == 2
    d = json.load(io.open(rf, encoding="utf-8"))
    assert d["card"] == "PREREG_V8F_PROOF_ADD116" and set(d["names"]) == {"THRESHOLD", "WHERE", "t_star"}


def test_names_stage_refuses_a_controls_file_that_is_not_all_pass(tmp_path):
    t1, jp, tp = T.make_world(str(tmp_path / "w"), fault="h3")
    cf = str(tmp_path / "C.json")
    assert T.main(["--stage", "controls", "--t1", t1, "--join", jp, "--tests", tp, "--out", cf, "--n-boot", "20"]) == 0
    assert T.main(["--stage", "names", "--controls", cf, "--out", str(tmp_path / "R.json")]) == 3
    assert not os.path.exists(str(tmp_path / "R.json"))


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
