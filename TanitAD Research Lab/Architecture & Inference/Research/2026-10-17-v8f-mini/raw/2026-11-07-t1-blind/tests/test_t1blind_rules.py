"""The outcome rules of PREREG_T1_BLIND.md section 5 as t1blind_read.py applies them: known-answer cases on synthetic rows (the real SR / GD instruments are loaded by file path).
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_t1blind_rules.py"""
import os
import sys
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import pytest

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
sys.path.insert(0, PK)
import t1blind_read as T  # noqa: E402


@pytest.fixture(scope="module")
def inst():
    spec = importlib.util.spec_from_file_location("lf_for_rules_test", T.LF_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.init()
    return m.SR, m.GD


def rows(ade, m1):
    f = lambda t: {"delta": t[0], "lo": t[1], "hi": t[2], "separated": bool(t[1] > 0 or t[2] < 0)}
    return {T.ADE: f(ade), T.M1: f(m1)}


NS = (0.0, -0.05, 0.05)


def cond(inst, ade, m1):
    SR, GD = inst
    return T.seed_conditions(rows(ade, m1), SR, GD)


def test_ade_separated_worse_above_F_is_used(inst):
    c = cond(inst, (0.20, 0.10, 0.30), NS)
    assert c["ADE_cond"] and c["USED_cond"] and not c["UNUSED_cond"]


def test_ade_separated_worse_below_F_is_neither(inst):
    c = cond(inst, (0.10, 0.05, 0.15), NS)                     # separated, but the point is under F = 0.12074
    assert not c["ADE_cond"] and not c["USED_cond"] and not c["UNUSED_cond"]


def test_ade_separated_better_is_not_used_and_not_unused(inst):
    c = cond(inst, (-0.20, -0.30, -0.10), NS)                  # BLIND better than RIC0 on ADE6: separated, wrong direction for the ADE clause
    assert not c["ADE_cond"] and not c["USED_cond"] and not c["UNUSED_cond"]


def test_m1_separated_either_direction_is_used(inst):
    assert cond(inst, NS, (-0.05, -0.08, -0.02))["USED_cond"]
    assert cond(inst, NS, (0.05, 0.02, 0.08))["USED_cond"]


def test_nothing_separated_and_small_is_unused(inst):
    c = cond(inst, (0.05, -0.05, 0.15), (0.0, -0.1, 0.1))
    assert c["UNUSED_cond"] and not c["USED_cond"]


def test_large_but_unseparated_ade_is_mixed(inst):
    c = cond(inst, (0.15, -0.05, 0.35), NS)                    # |B| >= F and not separated
    assert not c["UNUSED_cond"] and not c["USED_cond"]


def test_outcome_needs_both_seeds(inst):
    used = cond(inst, (0.20, 0.10, 0.30), NS)
    unused = cond(inst, NS, NS)
    mid = cond(inst, (0.10, 0.05, 0.15), NS)
    assert T.outcome({0: used, 1: used})[0] == "VISION-USED"
    assert T.outcome({0: used, 1: unused})[0] == "MIXED"
    assert T.outcome({0: unused, 1: unused})[0] == "VISION-UNUSED"
    assert T.outcome({0: unused, 1: mid})[0] == "MIXED"
    m1only = cond(inst, NS, (0.05, 0.02, 0.08))
    assert T.outcome({0: used, 1: m1only})[0] == "VISION-USED"                                      # the disjunction is per seed
    lab, beside = T.outcome({0: used, 1: m1only})
    assert beside["strict_reading_ADE_both_or_M1_both"] == "MIXED"                                  # the stricter reading is reported beside
