"""Tests for s0l_read.py: CPU, planted data, seconds.  Real-data tests skip when the banked npz files are absent.
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_s0l_read.py   (from this folder; OMP_NUM_THREADS=2)"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import s0l_read as R  # noqa: E402

NB = 200
PARITY = R.THEORY + "/stopdec/out/stopdec_PARITY_DAGOACC_s0.npz"
real = pytest.mark.skipif(not os.path.exists(PARITY), reason="banked STOPDEC npz not present")


def test_planted_gain_counts_and_small_does_not():
    r = R.planted_control(NB)
    assert r["pass"], r
    assert r["gain_dauc"] > R.FLOOR > r["small_dauc"] > 0


def test_planted_floor_zero_mutant_is_red():
    assert not R.planted_control(NB, floor=0.0)["pass"]


def test_self_pair_exact_zero_and_no_gain():
    B, _, _ = R.planted()
    r = R.self_control(B, NB)
    assert r["pass"] and r["dauc"] == 0.0 and (r["lo"], r["hi"]) == (0.0, 0.0) and r["gain"] is False


def test_self_inverted_mutant_is_red():
    B, _, _ = R.planted()
    assert not R.self_control(B, NB, mutate="invert")["pass"]


def test_label_shuffle_covers_half_and_real_labels_do_not():
    B, _, _ = R.planted()
    assert R.shuffle_control(B, NB)["pass"]
    assert not R.shuffle_control(B, NB, shuffle=False)["pass"]


def test_counts_equal_and_mutant_red():
    B, _, _ = R.planted()
    exp = {"n_windows": len(B["eid"]), "n_pos": int(B["stop_hit"].sum()), "n_episodes": 139}
    assert R.counts_control(B, exp)["pass"]
    assert not R.counts_control(B, exp, drop_one=True)["pass"]


def test_counts_assert_refuses_wrong_counts():
    B, Lg, _ = R.planted()
    with pytest.raises(R.S0LRefusal):
        R.read_pair(Lg, B, n_boot=NB)       # planted shape is not STOPCUE's 2,365 / 131 / 139


def test_misaligned_arms_refused():
    B, Lg, _ = R.planted()
    assert R.alignment_control(B)["pass"]
    Lg = dict(Lg)
    Lg["eid"] = Lg["eid"][::-1].copy()
    with pytest.raises(R.S0LRefusal):
        R.check_aligned(Lg, B)


def test_decide_amendment5_needs_point_above_floor_and_ci_lower_above_zero():
    # AMENDMENT_5_S0L.md: GAIN iff point > 0.037 AND CI lower > 0.
    assert R.decide({"dauc": 0.050, "lo": 0.010}) is True
    assert R.decide({"dauc": 0.050, "lo": -0.010}) is False    # the point clears, the interval covers 0: NO-GAIN
    assert R.decide({"dauc": 0.037, "lo": 0.010}) is False     # "exceeds" is strict
    assert R.decide({"dauc": 0.030, "lo": 0.010}) is False
    assert R.decide({"dauc": 0.050, "lo": None}) is False      # no bound, no gain


def test_card_rule_fields_and_point_only_is_beside():
    B, Lg, Ls = R.planted()
    exp = {"n_windows": len(B["eid"]), "n_pos": int(B["stop_hit"].sum()), "n_episodes": 139}
    rg = R.read_pair(Lg, B, n_boot=NB, expect=exp)
    g = rg["goal_stop"]
    assert rg["gain"] is (g["dauc"] > 0.037 and g["lo"] > 0) and rg["point_only_beside"] is (g["dauc"] > 0.037)
    assert "AMENDMENT_5_S0L" in rg["rule"] and rg["tier"].startswith("T1-kin")
    assert rg["families"]["STRATEGIC"].startswith("PAUSED") and rg["families"]["LATERAL"].startswith("NOT COVERED")
    assert rg["counts"] == exp


def test_ci_module_is_pinned():
    S1, CI, sha = R._import_frozen()
    assert sha == R.CI_SHA256


@real
def test_real_counts_match_stopcue():
    c = R.counts_control(R.load_arm(PARITY))
    assert c["pass"] and c["counts"] == {"n_windows": 2365, "n_pos": 131, "n_episodes": 139}


@real
def test_real_self_pair_zero():
    r = R.self_control(R.load_arm(PARITY), NB)
    assert r["pass"] and r["dauc"] == 0.0
