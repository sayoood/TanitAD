"""The unattended chain's DECISION and its liveness probe, proven before it decides alone.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_after_bridge_chain.py

⛔ The chain writes BAR-W3-M1's verdict hours from now with nobody reading. Two things must be
right before that is acceptable: the bar is `A1 > max(STOP, CV)` and NOT "beats a floor" (on
navtest v1 a do-nothing plan scores three times constant velocity, so the easy side is the wrong
side), and the wait must end on the ARTIFACT rather than on a probe that cannot fail.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / "code"


def _mod():
    spec = importlib.util.spec_from_file_location("w3_chain", CODE / "after_bridge_chain.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_chain"] = m
    spec.loader.exec_module(m)
    return m


M = _mod()


def _arms(a1, stop, cv, paired=None):
    d = {"CV": {"x100": {"PDMS": cv}, "n": 10},
         "STOP": {"x100": {"PDMS": stop}, "n": 10},
         "A1": {"x100": {"PDMS": a1}, "n": 10}}
    if paired:
        d["A1"]["paired_interval"] = paired
    return d


def test_beating_both_floors_is_a_pass():
    b = M.bar_verdict(_arms(70.0, 62.5812, 21.8222), "A1", partial=False)
    assert b["VERDICT"] == "PASS"
    assert b["max_floor_x100"] == 62.5812          # LITERAL: STOP is the max, not CV
    assert b["full_split"] is True


def test_beating_only_cv_is_a_fail_this_is_the_measured_case():
    """⭐ The 200-token reading: A1 57.9689, STOP 62.5812, CV 21.8222. Beating CV by +36 does not
    pass a bar written against the maximum — and a rule written with `any` would say it did."""
    b = M.bar_verdict(_arms(57.9689, 62.5812, 21.8222), "A1", partial=True)
    assert b["VERDICT"] == "FAIL"
    assert b["PDMS_x100"]["A1"] > b["PDMS_x100"]["CV"]
    assert b["full_split"] is False
    assert b["_note"].startswith("⛔ A verdict on a PARTIAL seam is PROVISIONAL")


def test_an_exactly_equal_score_does_not_pass_a_strict_greater_bar():
    assert M.bar_verdict(_arms(62.5812, 62.5812, 21.8222), "A1", False)["VERDICT"] == "FAIL"


def test_a_missing_floor_is_inconclusive_never_a_pass():
    """⛔ A floor that failed to load must not silently lower the bar to the floors that did."""
    arms = _arms(70.0, 62.5812, 21.8222)
    del arms["STOP"]
    b = M.bar_verdict(arms, "A1", partial=False)
    assert b["VERDICT"] == "INCONCLUSIVE" and b["missing_floors"] == ["STOP"]


def test_the_paired_intervals_travel_with_the_verdict():
    p = {"STOP": {"delta": -0.0461, "ci95": 0.059, "separated": False},
         "CV": {"delta": 0.3615, "ci95": 0.0599, "separated": True}}
    b = M.bar_verdict(_arms(57.9689, 62.5812, 21.8222, p), "A1", partial=True)
    assert b["paired_vs_floor"]["STOP"]["separated"] is False
    assert b["paired_vs_floor"]["CV"]["delta"] == 0.3615


def test_the_liveness_probe_answers_both_ways_in_one_breath():
    """⚠️ SAME-BREATH CONTROL. A probe that always returns False ends the wait instantly and a
    probe that always returns True never ends it; only a pair of opposite answers shows it reads
    anything. The positive is THIS interpreter — a python.exe running pytest."""
    assert M.bridge_alive("*zzz-no-such-process-zzz*") is False
    assert M.bridge_alive("*pytest*") is True


def test_an_unreadable_probe_assumes_alive_rather_than_finished(monkeypatch):
    """A probe that cannot be read must not be reported as 'the run ended' — that would score a
    seam that is still being written."""
    def boom(*a, **k):
        raise OSError("probe unavailable")
    monkeypatch.setattr(M.subprocess, "run", boom)
    assert M.bridge_alive() is True
