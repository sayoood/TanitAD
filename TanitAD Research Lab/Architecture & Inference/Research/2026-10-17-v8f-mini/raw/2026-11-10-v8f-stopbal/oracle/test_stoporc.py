"""Tests for stoporc_read.py (CPU, numpy only, no model weights).  Run: python -m pytest -q test_stoporc.py

Controls of the card (PREREAD_V8F_STOPORC.md, "Controls that must read known values"):
  1 IDENT is checked in the run on real arms (max abs 0.0); here the check logic is exercised on planted plans.
  2 the forced code differs from OWN exactly in the named entries (counted), everything else bit for bit.
  3 red arm: a mutant that forces nothing reads delta exactly 0.0 and a reaches-O control "not separated".
Red arms that must go red: a forcing that forgets the goal entry, one that touches a lat entry, one that picks the wrong lon class, an outcome rule with the floor removed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stoporc_read as R  # noqa: E402

CI = R.load_ci()


def make_codes(n=40, seed=0):
    rng = np.random.default_rng(seed)
    goal = rng.random((n, R.N_G)).astype(np.float32)
    lat = rng.dirichlet(np.ones(R.N_LAT), n).astype(np.float32)
    lon = rng.dirichlet(np.ones(R.N_LON), n).astype(np.float32)
    cons = rng.normal(size=(n, R.N_CONS)).astype(np.float32)
    own = np.concatenate([goal, lat, lon, cons], 1)
    al = rng.random((n, R.N_LON)) < 0.4
    al[:10, R.LON_STOP_I] = True            # STOP in the allowed set
    al[10:20, R.LON_STOP_I] = False
    al[10:20, 3] = True                     # another class allowed
    al[20:25] = False                       # nothing allowed
    own[0, R.STOP_GI] = 1.0                 # goal entry already 1.0 on window 0
    return own, al


def test_layout_constants():
    assert R.CODE_DIM == 60 and R.LON_SL == slice(27, 34) and R.STOP_GI == 6 and R.LON_STOP_I == 2


@pytest.mark.parametrize("mode", ["fgt", "fgoal", "flon"])
def test_forced_differs_only_in_named_entries(mode):
    own, al = make_codes()
    c = R.force_codes(own, al, mode)
    cnt, prob = R.check_forced(own, c, al, mode)
    assert prob == [], prob
    allowed = R.allowed_cols(mode)
    assert np.array_equal(c[:, ~allowed].view(np.uint32), own[:, ~allowed].view(np.uint32))      # bit for bit elsewhere
    assert cnt["n_windows"] == len(own)
    if mode in ("fgt", "fgoal"):
        assert (c[:, R.STOP_GI] == 1.0).all()
        assert cnt["n_goal_already_ge_1"] == 1
    if mode == "fgoal":
        assert cnt["lon_part_changed"] == 0
    if mode == "flon":
        assert cnt["goal_entry_changed"] == 0


def test_lon_rule_three_cases():
    own, al = make_codes()
    c = R.force_codes(own, al, "fgt")
    lon = c[:, R.LON_SL]
    assert (lon[:10].argmax(1) == R.LON_STOP_I).all() and (lon[:10].sum(1) == 1.0).all()           # STOP allowed -> one-hot STOP
    for i in range(10, 20):                                                                         # else the allowed class T ranks highest (class 3 is the only allowed one there, plus random)
        allowed_idx = np.where(al[i])[0]
        want = allowed_idx[own[i, R.LON_SL][allowed_idx].argmax()]
        assert lon[i].argmax() == want and lon[i].sum() == 1.0 and R.LON_STOP_I not in allowed_idx
    assert np.array_equal(lon[20:25], own[20:25, R.LON_SL])                                         # nothing allowed -> lon part left as T's own (counted)
    cnt, _ = R.check_forced(own, c, al, "fgt")
    assert cnt["lon_case_counts"]["no_allowed_class_lon_left_own"] >= 5
    assert sum(cnt["lon_case_counts"].values()) == len(own)


def test_changed_entries_counted_per_window():
    own, al = make_codes()
    c = R.force_codes(own, al, "fgt")
    cnt, _ = R.check_forced(own, c, al, "fgt")
    assert sum(cnt["changed_entries_per_window"].values()) == len(own)
    assert max(int(k) for k in cnt["changed_entries_per_window"]) <= 1 + R.N_LON
    assert cnt["goal_entry_changed"] + cnt["neither_changed"] + (cnt["lon_part_changed"] - cnt["both_changed"]) == len(own)


# ------------------------------------------------------------------------------------------------ RED ARMS: each defect must be caught by control 2
def test_red_forcing_that_forgets_goal_entry():
    own, al = make_codes()
    c = R.force_codes(own, al, "fgt")
    c[:, R.STOP_GI] = own[:, R.STOP_GI]                  # the defect
    assert any("not 1.0" in p for p in R.check_forced(own, c, al, "fgt")[1])


def test_red_forcing_that_touches_a_lat_entry():
    own, al = make_codes()
    c = R.force_codes(own, al, "fgt")
    c[3, R.N_G] += 1e-7                                    # a lat cell, one ulp-scale nudge
    assert any("outside the named entries" in p for p in R.check_forced(own, c, al, "fgt")[1])


def test_red_forcing_that_touches_a_constraint_cell():
    own, al = make_codes()
    c = R.force_codes(own, al, "flon")
    c[0, 40] = np.nextafter(c[0, 40], np.float32(10.0))   # one ulp in a cons cell
    assert any("outside the named entries" in p for p in R.check_forced(own, c, al, "flon")[1])


def test_red_forcing_with_wrong_lon_class():
    own, al = make_codes()
    c = R.force_codes(own, al, "fgt")
    wrong = (int(c[12, R.LON_SL].argmax()) + 1) % R.N_LON
    c[12, R.LON_SL] = np.eye(R.N_LON, dtype=np.float32)[wrong]
    assert any("one-hot" in p for p in R.check_forced(own, c, al, "fgt")[1])


def test_red_goal_mode_that_also_moves_lon():
    own, al = make_codes()
    c = R.force_codes(own, al, "fgoal")
    c[:, R.LON_SL] = R.force_codes(own, al, "flon")[:, R.LON_SL]
    assert any("outside the named entries" in p for p in R.check_forced(own, c, al, "fgoal")[1])


# ------------------------------------------------------------------------------------------------ control 3: the mutant that forces nothing
def planted_cells(n=60, n_ep=12, shift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), n // n_ep)
    own = {c: rng.random(n) + 1.0 for c in R.CELLS}
    forced = {c: own[c] + shift for c in R.CELLS}
    return {"own": own, "fgt": forced, "fgoal": forced, "flon": forced, "mutant": {c: own[c].copy() for c in R.CELLS}}, eid


def test_mutant_reads_exactly_zero_and_not_separated():
    cells, eid = planted_cells(shift=-0.3)
    sets = {"stop": np.arange(60) < 24, "ctl": np.arange(60) >= 24}
    res = R.analyse_cells(CI, cells, sets, eid)
    m = res["mutant"]
    assert m["stop"][R.PRIMARY_CELL]["point"] == 0.0 and m["ctl"][R.PRIMARY_CELL]["separated"] is False
    assert all(m[s][c]["point"] == 0.0 for s in m for c in R.CELLS)


def test_red_arm_a_forcing_that_moves_the_plan_is_separated():
    cells, eid = planted_cells(shift=-0.3)
    sets = {"stop": np.arange(60) < 24, "ctl": np.arange(60) >= 24}
    res = R.analyse_cells(CI, cells, sets, eid)
    d = res["fgt"]["ctl"][R.PRIMARY_CELL]
    assert d["separated"] is True and d["point"] == pytest.approx(-0.3)


def test_ident_check_logic_goes_red_on_a_nudge():
    plan = np.random.default_rng(0).normal(size=(5, 30, 2)).astype(np.float32)
    assert float(np.abs(plan.astype(np.float64) - plan).max()) == 0.0
    nudged = plan.copy()
    nudged[2, 5, 0] += 1e-3
    assert float(np.abs(nudged.astype(np.float64) - plan).max()) != 0.0


# ------------------------------------------------------------------------------------------------ outcome table (planted)
def cell(point, sep):
    return {"point": point, "separated": sep, "lo": point - 0.1, "hi": point + 0.1}


def arms(p0, p1, prep, r0, r1, rrep=False):
    return {"BAL-NULL_s0": {"primary": cell(*p0), "reach": cell(0.2, r0)}, "BAL-NULL_s1": {"primary": cell(*p1), "reach": cell(0.2, r1)},
            "BAL-NULL_rep": {"primary": cell(*prep), "reach": cell(0.2, rrep)}}


def test_outcome_follows():
    o = R.outcome(arms((-0.8, True), (-0.7, True), (-0.78, False), True, True))      # floor |-0.78 - -0.8| = 0.02; both gains clear it
    assert o["outcome"] == "O-FOLLOWS" and o["floor_m"] == pytest.approx(0.02)


def test_outcome_misreads():
    assert R.outcome(arms((0.5, True), (0.4, True), (0.45, False), True, True))["outcome"] == "O-MISREADS"
    assert R.outcome(arms((-0.01, False), (0.3, False), (0.0, False), True, True))["outcome"] == "O-MISREADS"      # not-a-gain includes unseparated


def test_outcome_ignores():
    assert R.outcome(arms((0.0, False), (0.0, False), (0.0, False), False, False))["outcome"] == "O-IGNORES"


def test_outcome_seeds_disagree():
    assert R.outcome(arms((-0.8, True), (0.1, False), (-0.78, False), True, True))["outcome"] == "SEEDS-DISAGREE"   # gain at one seed only
    assert R.outcome(arms((0.0, False), (0.0, False), (0.0, False), True, False))["outcome"] == "SEEDS-DISAGREE"    # reaches-O at one seed, no gain


def test_gain_needs_separation_better_side_and_floor():
    assert R.is_gain(cell(-0.5, True), 0.1)
    assert not R.is_gain(cell(-0.5, False), 0.1)          # not separated
    assert not R.is_gain(cell(0.5, True), 0.1)            # worse side
    assert not R.is_gain(cell(-0.05, True), 0.1)          # under the floor
    assert not R.is_gain(cell(-0.5, True), None)          # no floor


def test_red_outcome_without_the_floor_would_differ():
    # a planted delta that is separated and better but under the floor: the rule says NOT a gain.  A mutant that drops the floor turns it into one.
    a = arms((-0.05, True), (-0.05, True), (0.5, False), False, False)            # floor |0.5 - -0.05| = 0.55
    assert R.outcome(a)["outcome"] == "O-IGNORES"
    mutant = lambda d, floor: bool(d["separated"] and d["point"] < 0)             # noqa: E731
    assert mutant(a["BAL-NULL_s0"]["primary"], 0.55) and not R.is_gain(a["BAL-NULL_s0"]["primary"], 0.55)


def test_outcome_fails_closed_on_controls_and_missing_arms():
    o = R.outcome(arms((-0.8, True), (-0.7, True), (-0.78, False), True, True), controls_ok=False)
    assert o["outcome"] == "INVALID-CONTROLS" and o["rule_outcome_if_controls_held"] == "O-FOLLOWS"
    a = arms((-0.8, True), (-0.7, True), (-0.78, False), True, True)
    del a["BAL-NULL_rep"]
    assert R.outcome(a)["outcome"] == "NOT-EVALUATED"


def test_card_hash_unchanged():
    import hashlib
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREAD_V8F_STOPORC.md")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == "49eb7d7be6bcdfc655fcb2f7c1c21e6583248c0d3a75f87321fd776806c34535"
