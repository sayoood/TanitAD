"""Literal tests for nav_ann (A6/A7): A5's nav_tl rule over ANNOUNCED entries only, the T3a-c donor series, the reading rule
and the subset view. Expectations are hand-derived literals. Plain process (no tanitad / builder import).

Run:  python test_nav_ann_a6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_ann_a6 as NA  # noqa: E402

L_, R_, F_ = 1, -1, 0


def _e(tok, t0, t1):
    return {"token": tok, "t_start_s": t0, "t_end_s": t1}


def _z(*shas):
    return {"win_sha12": np.array(shas)}


def test_skipping_a_non_announced_entry_lets_the_next_announced_one_count():
    # entry 1: TURN_R 2-4 (NOT announced, e.g. the obstacle pass); entry 2: TURN_L 5-8 (announced)
    recs = {"a": {"entries": [_e("NAV_TURN_R", 2.0, 4.0), _e("NAV_TURN_L", 5.0, 8.0)]}}
    flags = {"a": [False, True]}
    z = _z("a")
    # t_rel 0: A5's nav_tl sees the right turn 2 s ahead; nav_ann skips it and sees the left turn 5 s ahead (<= 6)
    assert NA.nav_side(z, recs, flags, np.array([0.0]), announced_only=False)[0] == R_
    assert NA.nav_side(z, recs, flags, np.array([0.0]), announced_only=True)[0] == L_
    # t_rel -2: tl: right turn 4 s ahead -> right; ann: left turn 7 s ahead (> 6) -> follow
    assert NA.nav_side(z, recs, flags, np.array([-2.0]), announced_only=False)[0] == R_
    assert NA.nav_side(z, recs, flags, np.array([-2.0]), announced_only=True)[0] == F_
    # t_rel 6: everything finished -> follow in both
    assert NA.nav_side(z, recs, flags, np.array([9.0]), announced_only=False)[0] == F_
    assert NA.nav_side(z, recs, flags, np.array([9.0]), announced_only=True)[0] == F_


def test_a_clip_whose_only_turn_is_not_announced_is_follow_everywhere():
    recs = {"a": {"entries": [_e("NAV_TURN_R", 3.8, 8.7)]}}
    for t_rel in (-5.0, 0.0, 4.0, 9.0):
        assert NA.nav_side(_z("a"), recs, {"a": [False]}, np.array([t_rel]), announced_only=True)[0] == F_
        # ... and A5's nav_tl (the foil T3) still announces it while it is within 6 s / under way
    assert NA.nav_side(_z("a"), recs, {"a": [False]}, np.array([0.0]), announced_only=False)[0] == R_


def test_donor_series_uses_the_donor_entries_and_flags_at_the_recipients_t_rel():
    recs = {"a": {"entries": [_e("NAV_FOLLOW_ROAD", 0.0, 30.0)]},
            "b": {"entries": [_e("NAV_TURN_L", 3.0, 6.0)]}}
    flags = {"a": [False], "b": [True]}
    z = _z("a")
    assert NA.nav_side(z, recs, flags, np.array([0.0]), announced_only=True)[0] == F_                       # own series: follow
    assert NA.nav_side(z, recs, flags, np.array([0.0]), donor={"a": "b"}, announced_only=True)[0] == L_     # b's turn, 3 s ahead
    flags_b_supp = {"a": [False], "b": [False]}
    assert NA.nav_side(z, recs, flags_b_supp, np.array([0.0]), donor={"a": "b"}, announced_only=True)[0] == F_


def _bar(c):
    return {"CLEARS": c}


def test_reading_rule_truth_table():
    r = NA.reading_rule
    assert r({"T3a": _bar(True), "T3": _bar(True), "T2a": _bar(False), "T3a-c": _bar(False)})["branch"] == "T3a PASSES, T3a-c FAILS"
    assert r({"T3a": _bar(True), "T3": _bar(False), "T2a": _bar(False), "T3a-c": _bar(False)})["branch"] == "T3a PASSES, T3a-c FAILS"
    assert r({"T3a": _bar(False), "T3": _bar(True), "T2a": _bar(False), "T3a-c": _bar(False)})["branch"] == "T3a FAILS, T3 PASSES"
    assert r({"T3a": _bar(False), "T3": _bar(False), "T2a": _bar(True), "T3a-c": _bar(False)})["branch"] == "BOTH FAIL"
    assert r({"T3a": _bar(False), "T3": _bar(False), "T2a": _bar(False), "T3a-c": _bar(False)})["branch"] == "BOTH FAIL"
    # the control passing overrides everything
    assert r({"T3a": _bar(True), "T3": _bar(True), "T2a": _bar(True), "T3a-c": _bar(True)})["branch"] == "T3a-c PASSES"


def test_subset_view_removes_windows_from_every_class():
    d = {"cls": np.array(["turnL", "straight", "gentle", "turnR"], dtype=object), "target": np.array([1, 0, 1, -1])}
    s = NA.subset_view(d, np.array([True, False, True, False]))
    assert list(s["cls"]) == ["turnL", "unclassified", "gentle", "unclassified"]
    assert list(s["target"]) == [1, 9, 1, 9]
    assert list(d["cls"]) == ["turnL", "straight", "gentle", "turnR"]            # the input is not modified


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
