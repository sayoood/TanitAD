"""Tests for stophard_read.py (CPU, numpy only, no model weights).  Run: python -m pytest -q test_stophard.py

Controls of the card (PREREAD_V8F_STOPHARD.md): 1 IDENT with an unreachable threshold equals OWN exactly; 2 HARD differs from OWN only in the named entries and only
where the decision fires; 3 decision counts equal STOPDEC's / STOPBAL M's (checked in the run on real arms; here the counting logic on planted codes).
Red arms that must go red: a transform that forgets the goal entry, touches a lat or constraint cell, hardens the wrong lon class, fires everywhere (threshold 0),
uses a label instead of T's own decision, writes the goal entry where only the lon decision fires, and an outcome rule that drops the floor.
"""
import hashlib
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stophard_read as H  # noqa: E402

CI = H.SO.load_ci()
HERE = os.path.dirname(os.path.abspath(__file__))


def make_codes(n=60, seed=0):
    rng = np.random.default_rng(seed)
    goal = rng.random((n, H.N_G)).astype(np.float32)
    lat = rng.dirichlet(np.ones(H.N_LAT), n).astype(np.float32)
    lon = rng.dirichlet(np.ones(H.N_LON), n).astype(np.float32)
    cons = rng.normal(size=(n, H.N_CONS)).astype(np.float32)
    own = np.concatenate([goal, lat, lon, cons], 1)
    own[:, 27 + H.LON_STOP_I] = 0.0                             # no lon fire except where set below
    own[32:, H.STOP_GI] = 0.1
    own[:15, H.STOP_GI] = 0.7                                  # goal fires
    own[15:30, H.STOP_GI] = 0.2                                 # goal does not fire
    own[30, H.STOP_GI] = 0.5                                    # exactly on the threshold: fires (>=)
    own[31, H.STOP_GI] = np.nextafter(np.float32(0.5), np.float32(0.0))   # just under: does not fire
    own[10:25, H.LON_SL] = np.float32(0.01)
    own[10:25, 27 + H.LON_STOP_I] = np.float32(0.9)             # lon argmax STOP on windows 10..24 (overlaps goal-fire 10..14)
    return own


def test_card_hash_unchanged():
    p = os.path.join(HERE, "PREREAD_V8F_STOPHARD.md")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == H.CARD_SHA
    line = open(os.path.join(HERE, "PREREAD_V8F_STOPHARD.sha256"), encoding="utf-8").read().split()[0]
    assert line == H.CARD_SHA


def test_decision_rules_match_stopbal_source():
    src = open(os.path.join(H.READ, "stopbal_read.py"), encoding="utf-8").read()
    assert re.search(r"^P_STOP_MIN = 0\.5\b", src, re.M) and re.search(r"^STOP_GI = 6\b", src, re.M) and re.search(r"^LON_STOP_I = 2\b", src, re.M)
    assert '"goal_stop": gp[:, STOP_GI] >= P_STOP_MIN' in src and '"lon_stop": lp.argmax(1) == LON_STOP_I' in src
    assert (H.P_STOP_MIN, H.STOP_GI, H.LON_STOP_I, H.LON_SL) == (0.5, 6, 2, slice(27, 34))


def test_decisions_and_threshold_edge():
    own = make_codes()
    g, l = H.decisions(own)
    assert g[30] and not g[31] and g[:15].all() and not g[15:30].any()
    assert l[10:25].all()


@pytest.mark.parametrize("mode", ["hard", "hgoal", "hlon"])
def test_hard_differs_only_in_named_entries_and_where_it_fires(mode):
    own = make_codes()
    c = H.hard_code(own, mode)
    cnt, prob = H.check_hard(own, c, mode)
    assert prob == [], prob
    g, l = H.decisions(own)
    gc = c[:, H.STOP_GI] != own[:, H.STOP_GI]
    lc = (c[:, H.LON_SL] != own[:, H.LON_SL]).any(1)
    if mode == "hlon":
        assert not gc.any()
    else:
        assert (c[g, H.STOP_GI] == 1.0).all() and not gc[~g].any()
    if mode == "hgoal":
        assert not lc.any()
    else:
        assert (c[l][:, H.LON_SL].argmax(1) == H.LON_STOP_I).all() and (c[l][:, H.LON_SL].sum(1) == 1.0).all() and not lc[~l].any()
    other = np.ones(H.CODE_DIM, bool)
    other[H.STOP_GI] = False
    other[H.LON_SL] = False
    assert np.array_equal(c[:, other].view(np.uint32), own[:, other].view(np.uint32))
    assert cnt["goal_fires"] == int(g.sum()) and cnt["lon_fires"] == int(l.sum()) and cnt["entries_outside_named_changed"] == 0


def test_unreachable_is_identity_bit_for_bit():
    own = make_codes()
    c = H.hard_code(own, "unreach")
    assert np.array_equal(c.view(np.uint32), own.view(np.uint32))
    cnt, prob = H.check_hard(own, c, "unreach")
    assert prob == [] and cnt["windows_changed"] == 0


def test_no_label_input():
    import inspect
    assert list(inspect.signature(H.hard_code).parameters) == ["own", "mode"]      # no GT, no allowed set, no other input
    own = make_codes()
    a = H.hard_code(own, "hard")
    b = H.hard_code(own.copy(), "hard")
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_input_not_mutated():
    own = make_codes()
    keep = own.copy()
    H.hard_code(own, "hard")
    assert np.array_equal(own.view(np.uint32), keep.view(np.uint32))


def test_counts_planted():
    own = make_codes()
    g, l = H.decisions(own)
    cnt, _ = H.check_hard(own, H.hard_code(own, "hard"), "hard")
    assert cnt["goal_fires"] == 16 and cnt["both_fire"] == int((g & l).sum()) == 5 and cnt["lon_fires"] == 15 and cnt["windows_changed"] == 16 + 15 - 5
    sel = np.zeros(len(own), bool)
    sel[:20] = True
    cnt2, _ = H.check_hard(own, H.hard_code(own, "hard"), "hard", sel)
    assert cnt2["n_windows"] == 20 and cnt2["goal_fires"] == 15


# ------------------------------------------------------------------------------------------------ RED ARMS
def test_red_transform_forgets_goal_entry():
    own = make_codes()
    c = H.hard_code(own, "hard")
    c[:, H.STOP_GI] = own[:, H.STOP_GI]
    assert any("not 1.0" in p for p in H.check_hard(own, c, "hard")[1])


def test_red_transform_touches_lat_or_cons():
    own = make_codes()
    for col in (H.N_G + 1, 45):
        c = H.hard_code(own, "hard")
        c[3, col] = np.nextafter(c[3, col], np.float32(10.0))
        assert any("outside the named entries" in p for p in H.check_hard(own, c, "hard")[1])


def test_red_transform_hardens_wrong_lon_class():
    own = make_codes()
    c = H.hard_code(own, "hard")
    row = int(np.where(H.decisions(own)[1])[0][0])
    c[row, H.LON_SL] = np.eye(H.N_LON, dtype=np.float32)[0]
    assert any("one-hot STOP" in p for p in H.check_hard(own, c, "hard")[1])


def test_red_transform_fires_everywhere():
    own = make_codes()
    c = own.copy()
    c[:, H.STOP_GI] = 1.0                                   # threshold 0: also hardens windows that do not fire
    assert any("does not fire" in p for p in H.check_hard(own, c, "hard")[1])
    c2 = own.copy()
    c2[:, H.LON_SL] = np.eye(H.N_LON, dtype=np.float32)[H.LON_STOP_I]
    assert any("does not fire" in p for p in H.check_hard(own, c2, "hard")[1])


def test_red_transform_uses_a_label():
    own = make_codes()
    rng = np.random.default_rng(3)
    gt_stop = rng.random(len(own)) < 0.5                    # a label-driven "decision"
    c = own.copy()
    c[gt_stop, H.STOP_GI] = 1.0
    assert H.check_hard(own, c, "hard")[1] != []


def test_red_goal_written_in_lon_only_mode():
    own = make_codes()
    c = H.hard_code(own, "hard")                            # hlon must not write the goal entry
    assert any("outside the named entries" in p or "does not fire" in p for p in H.check_hard(own, c, "hlon")[1])


def test_red_unreach_that_changes_something():
    own = make_codes()
    c = H.hard_code(own, "hard")
    assert H.check_hard(own, c, "unreach")[1] != []        # a reachable threshold read as IDENT is caught


def test_red_unchanged_window_detected_via_row_check():
    own = make_codes()
    c = H.hard_code(own, "hard")
    c[40, 50] = np.nextafter(c[40, 50], np.float32(99.0))   # a window with no firing decision changes
    assert H.check_hard(own, c, "hard")[1] != []


# ------------------------------------------------------------------------------------------------ outcome rule and cost/gain
def d(point, sep):
    return {"point": point, "separated": sep}


def test_outcome_table():
    mk = lambda g0, g1, c0, c1: {"s0": {"S_gain": g0, "A_cost": c0}, "s1": {"S_gain": g1, "A_cost": c1}}
    assert H.outcome_lever(mk(True, True, False, False)) == "HARD-HELPS"
    assert H.outcome_lever(mk(True, True, True, False)) == "HARD-TRADES"
    assert H.outcome_lever(mk(True, True, False, True)) == "HARD-TRADES"
    assert H.outcome_lever(mk(False, False, False, False)) == "HARD-NULL"
    assert H.outcome_lever(mk(False, False, True, True)) == "HARD-NULL"
    assert H.outcome_lever(mk(True, False, False, False)) == "SEEDS-DISAGREE"
    assert H.outcome_lever(mk(False, True, True, True)) == "SEEDS-DISAGREE"
    assert H.outcome_lever({"s0": {"S_gain": True, "A_cost": False}}) == "NOT-EVALUATED"


def test_gain_and_cost_rules():
    assert H.is_gain_cell(d(-0.5, True), 0.1) and not H.is_gain_cell(d(-0.05, True), 0.1) and not H.is_gain_cell(d(-0.5, False), 0.1) and not H.is_gain_cell(d(0.5, True), 0.1)
    assert H.is_cost(d(0.5, True), 0.1) and not H.is_cost(d(0.05, True), 0.1) and not H.is_cost(d(0.5, False), 0.1) and not H.is_cost(d(-0.5, True), 0.1)
    assert not H.is_gain_cell(d(-0.5, True), None) and not H.is_cost(d(0.5, True), None)      # no floor => never


def test_red_outcome_rule_without_floor_goes_red():
    """a no-floor mutant would call a below-floor separated gain a gain; the real rule must not."""
    below = d(-0.05, True)
    assert not H.is_gain_cell(below, 0.4)
    assert (lambda dd: bool(dd["separated"] and dd["point"] < 0))(below)                    # the mutant's reading differs


def test_floor_from_arms():
    mk = lambda p: {"delta_variant_minus_own": {v: {s: {c: {"point": p} for c in H.CELLS} for s in ("S_stop_point", "A_all")} for v in H.VARIANTS}}
    fl = H.floors_from({H.FLOOR_ARM: mk(-0.3), H.NULL_TAGS[0]: mk(-0.1)})
    assert abs(fl["hard"]["S_stop_point"][H.S_CELL] - 0.2) < 1e-12


def test_estimator_red_arm_identical_paired_vectors_not_separated():
    a = np.random.default_rng(1).normal(size=200)
    eid = np.repeat(np.arange(20), 10)
    r = H.SO.paired(CI, a, a.copy(), eid)
    assert r["point"] == 0.0 and r["separated"] is False


def test_fail_closed_outcome_when_missing():
    assert H.outcome_lever({}) == "NOT-EVALUATED"
