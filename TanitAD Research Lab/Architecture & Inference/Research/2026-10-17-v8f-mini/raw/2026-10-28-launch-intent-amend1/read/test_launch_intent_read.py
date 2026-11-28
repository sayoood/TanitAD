"""Tests of ``launch_intent_read.py`` (PREREG_V8F_LAUNCH_INTENT v2, MM_REDESIGN_2026-10-28): synthetic rows reach EVERY outcome row, both sides of every predicate boundary, every control, the slow-draw flag and the
per-seed AND over the two pairs; ``main_read`` runs END TO END on a FAKE heavy layer (rows scripted per contrast), so the wiring from the tables to the guards to the outcome is exercised, not only the pure functions.
No heavy import: runs in seconds.

Run: python -m pytest -q test_launch_intent_read.py          (LAUNCH_INTENT_READER=<mutant.py> points the whole file at a mutated copy; see mutation_launch_intent_reader.py;
                                                              LI_PREREG=<file> points the PREREG tests at another PREREG text)

Controls that must read as stated:
  * BAR        2 * max(F, RF0): the four crawl bars are 1.1070 / 0.0526 / 0.0136 / 0.0096, each below the CTRL - P10 gap at both seeds; RF is not a term (the function takes none).
  * CLEARS     the improvement must be >= the bar (inclusive), separated-better; FIRES needs all three conditions, with the pair's OWN control in the domain test.
  * SEED       every per-seed predicate and guard is the AND of the main pair and the replicate pair.
  * TABLE      REFUSED before HARMS before the rest; one row per reading; adoption needs R-MOVE at both seeds.
  * SLOW DRAW  all five speed cells separated and each shift > 0.5 m/s (strict); a mirror FAST-T; the bootstrap is the banked method.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
READER = os.environ.get("LAUNCH_INTENT_READER") or str(HERE / "launch_intent_read.py")
PREREG = os.environ.get("LI_PREREG") or str(HERE.parent / "PREREG_V8F_LAUNCH_INTENT.md")
spec = importlib.util.spec_from_file_location("launch_intent_read_under_test", READER)
R = importlib.util.module_from_spec(spec)
sys.modules["launch_intent_read_under_test"] = R
spec.loader.exec_module(R)
SEEDS = R.SEEDS
REAL_N_EPISODES, REAL_N_WINDOWS = R.N_EPISODES, R.N_WINDOWS

RF0 = {"ADE6": 0.0138, "speed MAE": 0.0796, "along-track MAE": 0.0201, "accel MAE": 0.1323, "DK-VIOL": 0.005, "DK-HW": 0.1308, "DK-TG": 0.0041, "heading MAE": 0.3801, "yaw-rate MAE": 0.5535, "curvature MAE": 0.0263,
       "cross-track MAE": 0.032, "L1 lateral-accel MAE": 0.3985, "L2 curvature MAE (GT>=5.2)": 0.0068, "TAC-LAT": 0.0183, "TAC-LON": 0.0125, "S-POS@30": 0.1877, "S-POS@50": 0.44489999999999996, "S-POS@80": 1.0082,
       "M1 crawl share": 0.0048}                                  # P0_RF0.json RF0 (sha256 1608d70b...)
BARS = R.bar_table(RF0)
CRAWL = list(R.CRAWL_SET)
GAP_CTRL_P10 = {"yaw-rate MAE": (3.4726, 8.6844), "curvature MAE": (0.0854, 0.1352), "L2 curvature MAE (GT>=5.2)": (0.0443, 0.0529), "M1 crawl share": (0.0568, 0.0604)}      # DAGORC_READ.json read_a.CTRL_P10 (sha256 70a7a3a4...)


def row(delta, lo=None, hi=None, sep=None):
    lo = delta - 0.05 if lo is None else lo
    hi = delta + 0.05 if hi is None else hi
    return {"delta": delta, "lo": lo, "hi": hi, "separated": (lo > 0 or hi < 0) if sep is None else sep, "n": "2399/141", "mean_arm": 1.0 + delta, "mean_ref": 1.0}


def flat():
    return row(0.0, -0.1, 0.1)


def table(over=None):
    t = {l: flat() for l in R.LABELS}
    t.update(over or {})
    return t


def better(label, amount):
    """a separated-better row improving by ``amount`` in the row's own direction"""
    d = -amount if R.DIRECTION[label] == "lower" else amount
    return row(d, d - 0.001, d + 0.001) if d > 0 else row(d, d - 0.001, d + 0.001)


def worse(label, amount):
    d = amount if R.DIRECTION[label] == "lower" else -amount
    return row(d, d - 0.001, d + 0.001)


def g_ok(t0=0.0, keep=0.0):
    return {"R-TLAT": flat(), "R-TLON": flat(), "T0_point": t0, "KEEP_point": keep}


def tabs3(x_p10=None, x_y=None, y_p10=None):
    return {"X_P10": table(x_p10), "X_Y": table(x_y), "Y_P10": table(y_p10)}


def clearing(extra=None):
    """X - Y clears every crawl row by exactly its bar"""
    d = {l: better(l, BARS[l]) for l in CRAWL}
    d.update(extra or {})
    return d


def pair(**kw):
    t = tabs3(kw.get("x_p10"), kw.get("x_y"), kw.get("y_p10"))
    return R.compose_pair(t, kw.get("g") or g_ok(), BARS)


# ------------------------------------------------------------------------------------------------ the bar
def test_the_bar_is_two_times_max_f_rf0_and_rf_is_not_a_term():
    assert R.BAR_FACTOR == 2.0 and R.F_ADE6 == 0.1207 and R.HALF_F == 0.0604
    assert list(inspect.signature(R.bar_for).parameters) == ["label", "rf0"]                 # no RF argument
    assert R.bar_for("ADE6", RF0) == 2 * 0.1207 and R.bar_for("ADE6", {**RF0, "ADE6": 0.2}) == 0.4
    assert R.bar_for("speed MAE", RF0) == 2 * 0.0796 and R.bar_for("M1 crawl share", {**RF0, "M1 crawl share": -0.004}) == 0.008
    assert R.bar_for("M1 crawl share", {**RF0, "M1 crawl share": 0.0}) == 0.0
    assert R.bar_table(RF0) == {l: R.bar_for(l, RF0) for l in R.LABELS} and set(R.bar_table(RF0)) == set(R.LABELS) and len(R.LABELS) == 19


def test_the_crawl_bars_are_reachable_below_the_ctrl_p10_gap_at_both_seeds():
    want = {"yaw-rate MAE": 1.1070, "curvature MAE": 0.0526, "L2 curvature MAE (GT>=5.2)": 0.0136, "M1 crawl share": 0.0096}
    for l, w in want.items():
        assert round(BARS[l], 4) == w
        assert all(BARS[l] < g for g in GAP_CTRL_P10[l]), l
    v1 = {"yaw-rate MAE": 31.7106, "curvature MAE": 0.2648, "L2 curvature MAE (GT>=5.2)": 0.1740, "M1 crawl share": 0.0776}       # the v1 bars 2 * max(F, RF0, RF): above the gap
    assert all(v1[l] > max(GAP_CTRL_P10[l]) for l in want)


def test_the_labels_directions_and_crawl_set():
    assert R.CRAWL_SET == ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
    assert R.DIRECTION["TAC-LAT"] == R.DIRECTION["TAC-LON"] == "higher" and sum(v == "higher" for v in R.DIRECTION.values()) == 2
    assert {R.FAMILY[l] for l in R.LABELS} == {"PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "crawl"}


# ------------------------------------------------------------------------------------------------ rows, clears, fires, domain
def test_direction_and_separation():
    assert R.sep_worse("ADE6", row(0.05, 0.02, 0.08)) and R.sep_better("ADE6", row(-0.05, -0.08, -0.02))
    assert R.sep_worse("TAC-LAT", row(-0.05, -0.08, -0.02)) and R.sep_better("TAC-LAT", row(0.05, 0.02, 0.08))
    assert not R.sep_worse("ADE6", row(0.05, -0.02, 0.08)) and not R.sep_better("ADE6", row(-0.05, -0.08, 0.02))
    zero = row(0.0, 0.01, 0.05, True)                                                         # separated with a zero point: neither worse nor better, in either direction
    assert not R.sep_worse("ADE6", zero) and not R.sep_better("ADE6", zero) and not R.sep_worse("TAC-LAT", zero) and not R.sep_better("TAC-LAT", zero)


def test_a_row_not_computed_raises():
    nc = {"status": "NOT COMPUTED"}
    for fn in (R.sep_worse, R.sep_better, R.worsening):
        with pytest.raises(R.RowNotComputed):
            fn("ADE6", nc)
    with pytest.raises(R.RowNotComputed):
        R.sep_worse("ADE6", None)


@pytest.mark.parametrize("improvement,want", [(0.0526, True), (0.05259, False), (0.06, True), (0.0, False)])
def test_clears_is_inclusive_at_the_bar(improvement, want):
    b = BARS["curvature MAE"]
    assert abs(b - 0.0526) < 1e-12
    assert R.clears("curvature MAE", row(-improvement, -improvement - 0.001, -0.0005 if improvement else 0.01), b) is want


def test_clears_needs_separation_direction_and_a_zero_bar_clears_on_separation_alone():
    b = BARS["curvature MAE"]
    assert not R.clears("curvature MAE", row(-0.2, -0.4, 0.01), b)                            # large improvement, not separated
    assert not R.clears("curvature MAE", row(0.2, 0.1, 0.3), b)                               # worse
    assert R.clears("TAC-LAT", row(0.04, 0.03, 0.05), BARS["TAC-LAT"] - 0.0066) and not R.clears("TAC-LAT", row(-0.04, -0.05, -0.03), 0.0)     # higher-is-better rows clear upward
    assert R.clears("curvature MAE", row(-0.001, -0.002, -0.0005), 0.0) and not R.clears("curvature MAE", row(0.0, -0.1, 0.1), 0.0)


def test_fires_needs_all_three_conditions_inclusive():
    b = 0.05
    xp, yp, xy = row(0.3, 0.2, 0.4), row(0.0, -0.1, 0.1), row(0.05, 0.01, 0.09)
    assert R.fires("heading MAE", xp, yp, xy, b)
    assert R.fires("heading MAE", xp, yp, row(0.05, 0.01, 0.09), 0.05) and not R.fires("heading MAE", xp, yp, row(0.0499, 0.01, 0.09), 0.05)         # worsening >= bar, inclusive
    assert not R.fires("heading MAE", row(0.3, -0.1, 0.4), yp, xy, b)                          # X - P10 not separated-worse
    assert not R.fires("heading MAE", xp, row(0.3, 0.2, 0.4), xy, b)                           # the control was already worse than P10: not in the domain
    assert not R.fires("heading MAE", xp, yp, row(0.2, -0.01, 0.4), b)                         # X - Y not separated
    assert not R.fires("heading MAE", xp, yp, row(-0.2, -0.3, -0.1), b)                       # X - Y better
    assert R.fires("TAC-LON", row(-0.3, -0.4, -0.2), yp, row(-0.1, -0.2, -0.05), 0.1) and not R.fires("TAC-LON", row(0.3, 0.2, 0.4), yp, row(0.1, 0.05, 0.2), 0.1)       # agreement rows worsen downward


def test_domain_is_the_rows_not_separated_worse_for_the_control_vs_p10():
    d = R.domain(table({"speed MAE": worse("speed MAE", 1.0), "TAC-LAT": worse("TAC-LAT", 0.1), "ADE6": better("ADE6", 1.0)}))
    assert "speed MAE" not in d and "TAC-LAT" not in d and "ADE6" in d and len(d) == 17
    assert R.domain(table()) == list(R.LABELS)
    assert "speed MAE" in R.domain(table({"speed MAE": row(0.5, -0.1, 1.0)}))                    # worse point, not separated: still in the domain


# ------------------------------------------------------------------------------------------------ one pair
def test_a_clean_pair_that_clears_everything():
    p = pair(x_y=clearing())
    assert p["flags"] == {"R-CRAWL": True, "R-MOVE": True, "R-KEEP": True, "R-NOBASE": True} and all(p["guards"].values())
    assert p["crawl_set"] == CRAWL and set(p["R-MOVE_rows"]) == set(CRAWL) and all(p["R-MOVE_rows"].values())
    assert len(p["domain"]) == 19 and not any(p["R-NEW_fires"].values())
    assert set(p["R-NEW_fires"]) == set(p["domain"]) and set(p["R-NOBASE_violations"]) == set(p["domain"])           # the reported rows are the pair's domain
    q = pair(y_p10={"speed MAE": worse("speed MAE", 1.0)}, x_y=clearing())
    assert "speed MAE" not in q["domain"] and set(q["R-NEW_fires"]) == set(q["domain"]) == set(q["R-NOBASE_violations"]) and len(q["domain"]) == 18


def test_r_move_needs_every_crawl_row_to_clear_and_only_crawl_rows():
    for l in CRAWL:
        d = clearing()
        d[l] = better(l, BARS[l] * 0.999)
        p = pair(x_y=d)
        assert p["flags"]["R-MOVE"] is False and p["R-MOVE_rows"][l] is False and sum(p["R-MOVE_rows"].values()) == 3, l
    p = pair(x_y=clearing({"speed MAE": worse("speed MAE", 0.05)}))                            # a non-crawl row does not enter R-MOVE
    assert p["flags"]["R-MOVE"] is True
    assert pair(x_y=table())["flags"]["R-MOVE"] is False


def test_r_crawl_is_x_minus_p10_not_separated_worse_on_each_crawl_row():
    assert pair()["flags"]["R-CRAWL"] is True
    for l in CRAWL:
        assert pair(x_p10={l: worse(l, 0.01)})["flags"]["R-CRAWL"] is False
        assert pair(x_p10={l: row(0.5, -0.1, 1.0)})["flags"]["R-CRAWL"] is True              # worse point, not separated
        assert pair(x_p10={l: better(l, 1.0)})["flags"]["R-CRAWL"] is True
    assert pair(x_p10={"speed MAE": worse("speed MAE", 1.0)})["flags"]["R-CRAWL"] is True       # a non-crawl row is not R-CRAWL's


def test_the_guards_read_the_contrasts_they_are_named_for():
    f1_bad = {"delta": -0.02, "lo": -0.03, "hi": -0.01, "separated": True}
    f1_noise = {"delta": -0.02, "lo": -0.05, "hi": 0.01, "separated": False}
    f1_up = {"delta": 0.02, "lo": 0.01, "hi": 0.03, "separated": True}
    for nm in ("R-TLAT", "R-TLON"):
        assert pair(g={**g_ok(), nm: f1_bad})["guards"][nm] is False
        assert pair(g={**g_ok(), nm: f1_noise})["guards"][nm] is True and pair(g={**g_ok(), nm: f1_up})["guards"][nm] is True
    zero_sep = {"delta": 0.0, "lo": -0.01, "hi": 0.0, "separated": True}
    assert pair(g={**g_ok(), "R-TLAT": zero_sep})["guards"]["R-TLAT"] is True and R.f1_worse(zero_sep) is False and R.f1_worse(f1_bad) is True and R.f1_worse(f1_up) is False
    assert pair(x_y={"ADE6": worse("ADE6", 0.05)})["guards"]["R-ADE"] is False
    assert pair(x_y={"ADE6": row(0.05, -0.02, 0.08)})["guards"]["R-ADE"] is True
    assert pair(x_p10={"ADE6": worse("ADE6", 0.5)})["guards"]["R-ADE"] is True                  # R-ADE reads X - Y, not X - P10
    for pt, want in ((0.0604, True), (0.06041, False), (-1.0, True), (0.0603, True)):
        assert pair(g=g_ok(t0=pt))["guards"]["R-T0"] is want
        assert pair(g=g_ok(keep=pt))["flags"]["R-KEEP"] is want
    assert pair(g=g_ok(t0=1.0))["flags"]["R-KEEP"] is True and pair(g=g_ok(keep=1.0))["guards"]["R-T0"] is True          # the two points are separate inputs


def test_r_new_and_r_nobase_share_one_contrast_and_use_the_pairs_own_domain():
    b = BARS["heading MAE"]
    xp, xy = {"heading MAE": worse("heading MAE", 1.0)}, {"heading MAE": worse("heading MAE", b)}
    p = pair(x_p10=xp, x_y=xy)
    assert p["guards"]["R-NEW"] is False and p["flags"]["R-NOBASE"] is False and p["R-NEW_fires"]["heading MAE"] is True and p["R-NOBASE_violations"]["heading MAE"] is True
    p = pair(x_p10=xp, x_y={"heading MAE": worse("heading MAE", b * 0.99)})
    assert p["guards"]["R-NEW"] is True and p["flags"]["R-NOBASE"] is True                      # below the bar
    p = pair(x_p10=xp, x_y=xy, y_p10={"heading MAE": worse("heading MAE", 1.0)})              # the control was already worse than P10: out of the domain
    assert "heading MAE" not in p["domain"] and p["guards"]["R-NEW"] is True and p["flags"]["R-NOBASE"] is True


def test_r_nobase_needs_the_row_worse_than_p10_as_well_as_worse_than_the_control():
    b = BARS["heading MAE"]
    p = pair(x_y={"heading MAE": worse("heading MAE", b)})                                    # worse than the control by the bar, but not worse than P10: no violation, nothing fires
    assert p["flags"]["R-NOBASE"] is True and p["guards"]["R-NEW"] is True and p["R-NOBASE_violations"]["heading MAE"] is False and p["R-NEW_fires"]["heading MAE"] is False
    p = pair(x_p10={"heading MAE": worse("heading MAE", 0.5)}, x_y={"heading MAE": worse("heading MAE", b)})
    assert p["flags"]["R-NOBASE"] is False and p["R-NOBASE_violations"]["heading MAE"] is True
    p = pair(x_p10={"heading MAE": worse("heading MAE", 0.5)}, x_y={"heading MAE": row(b, -b, 3 * b)})                      # worse by the bar but not separated against the control
    assert p["flags"]["R-NOBASE"] is True
    p = pair(x_p10={"heading MAE": worse("heading MAE", 0.5)}, x_y={"heading MAE": worse("heading MAE", b * 0.999)})        # separated but below the bar
    assert p["flags"]["R-NOBASE"] is True


def test_compose_pair_surfaces_a_row_not_computed():
    t = tabs3()
    t["X_Y"]["speed MAE"] = {"status": "NOT COMPUTED"}
    with pytest.raises(R.RowNotComputed):
        R.compose_pair(t, g_ok(), BARS)


# ------------------------------------------------------------------------------------------------ one seed: the AND over the two pairs
def test_every_per_seed_predicate_and_guard_is_the_and_of_the_two_pairs():
    base = pair(x_y=clearing())
    for key in R.FLAG_KEYS + R.GUARDS:
        for m_ok in (True, False):
            for r_ok in (True, False):
                m, r = json.loads(json.dumps(base)), json.loads(json.dumps(base))
                d = "flags" if key in R.FLAG_KEYS else "guards"
                m[d][key], r[d][key] = m_ok, r_ok
                s = R.compose_seed(m, r)
                assert s[d][key] is (m_ok and r_ok), (key, m_ok, r_ok)
                for other in (R.FLAG_KEYS + R.GUARDS):
                    if other != key:
                        assert s["flags" if other in R.FLAG_KEYS else "guards"][other] is True


def real_pairs(main_over=None, rep_over=None, g_main=None, g_rep=None):
    m = pair(x_y=clearing((main_over or {}).get("x_y")), x_p10=(main_over or {}).get("x_p10"), y_p10=(main_over or {}).get("y_p10"), g=g_main)
    r = pair(x_y=clearing((rep_over or {}).get("x_y")), x_p10=(rep_over or {}).get("x_p10"), y_p10=(rep_over or {}).get("y_p10"), g=g_rep)
    return m, r


def test_r_move_at_a_seed_needs_both_pairs_to_clear():
    m, r = real_pairs()
    assert R.compose_seed(m, r)["flags"]["R-MOVE"] is True
    rep_flat = pair(x_y=table())
    assert R.compose_seed(m, rep_flat)["flags"]["R-MOVE"] is False and R.compose_seed(rep_flat, m)["flags"]["R-MOVE"] is False


def test_r_crawl_t0_new_keep_each_need_both_pairs():
    m, r = real_pairs()
    bad_crawl = pair(x_y=clearing(), x_p10={"yaw-rate MAE": worse("yaw-rate MAE", 1.0)})
    assert R.compose_seed(m, bad_crawl)["flags"]["R-CRAWL"] is False
    assert R.compose_seed(m, pair(x_y=clearing(), g=g_ok(t0=0.07)))["guards"]["R-T0"] is False
    assert R.compose_seed(m, pair(x_y=clearing(), g=g_ok(keep=0.07)))["flags"]["R-KEEP"] is False
    fire = pair(x_y=clearing({"heading MAE": worse("heading MAE", BARS["heading MAE"])}), x_p10={"heading MAE": worse("heading MAE", 1.0)})
    s = R.compose_seed(m, fire)
    assert s["guards"]["R-NEW"] is False and s["flags"]["R-NOBASE"] is False


# ------------------------------------------------------------------------------------------------ the outcome table as one function
def held_all(v=True):
    return {g: {s: v for s in SEEDS} for g in R.GUARDS}


def both(v=True):
    return {s: v for s in SEEDS}


def row_of(**kw):
    ctrl = kw.pop("ctrl", True)
    held = kw.pop("held", held_all())
    d = dict(rc=both(), rm=both(), keep=both(), nobase=both())
    d.update(kw)
    return R.decide_li(ctrl, held, d["rc"], d["rm"], d["keep"], d["nobase"])["OUTCOME"]


def test_every_outcome_row_is_reached():
    assert row_of() == "FIXED / ADOPTED"
    assert row_of(ctrl=False) == "REFUSED"
    assert row_of(held={**held_all(), "R-T0": {0: True, 1: False}}) == "HARMS"
    assert row_of(keep={0: True, 1: False}) == "FIXED / NOT ADOPTED" and row_of(nobase={0: False, 1: True}) == "FIXED / NOT ADOPTED"
    assert row_of(rc={0: True, 1: False}) == "PARTIAL / ADOPTED"
    assert row_of(rc={0: False, 1: False}, keep={0: False, 1: True}) == "PARTIAL / NOT ADOPTED"
    assert row_of(rm={0: True, 1: False}) == "PARTIAL / ONE SEED" and row_of(rm={0: False, 1: True}) == "PARTIAL / ONE SEED"
    assert row_of(rm={0: False, 1: False}) == "NO EFFECT"


def test_precedence_refused_then_harms_then_r_move():
    assert row_of(ctrl=False, held=held_all(False), rm={0: False, 1: False}) == "REFUSED"
    assert row_of(held={**held_all(), "R-NEW": {0: False, 1: True}}) == "HARMS"                    # HARMS even with R-MOVE and R-CRAWL at both seeds
    assert row_of(held={**held_all(), "R-ADE": {0: False, 1: True}}, rm={0: False, 1: False}) == "HARMS"


def test_every_guard_at_every_seed_is_harms_when_it_fails():
    for g in R.GUARDS:
        for s in SEEDS:
            held = held_all()
            held[g][s] = False
            assert row_of(held=held) == "HARMS", (g, s)


def test_adoption_is_reachable_only_with_r_move_at_both_seeds_and_each_cell_matches_one_row():
    import itertools
    names = set()
    for ctrl, nguard, rc, rm, keep, nob in itertools.product((True, False), (None,) + tuple((g, s) for g in R.GUARDS for s in SEEDS), itertools.product((True, False), repeat=2), itertools.product((True, False), repeat=2),
                                                              itertools.product((True, False), repeat=2), itertools.product((True, False), repeat=2)):
        held = held_all()
        if nguard:
            held[nguard[0]][nguard[1]] = False
        d = R.decide_li(ctrl, held, dict(enumerate(rc)), dict(enumerate(rm)), dict(enumerate(keep)), dict(enumerate(nob)))
        assert len(d["matching_rows_in_table_order"]) == 1 and d["OUTCOME"] in R.OUTCOME_NAMES
        names.add(d["OUTCOME"])
        if d["adopts"]:
            assert all(rm) and all(keep) and all(nob) and ctrl and nguard is None
    assert names == set(R.OUTCOME_NAMES)


def test_an_unplaceable_reading_is_unplaced_not_invented(monkeypatch):
    monkeypatch.setattr(R, "outcome_rows", lambda *a, **k: {n: True for n in R.OUTCOME_NAMES})
    d = R.decide_li(True, held_all(), both(), both(), both(), both())
    assert d["OUTCOME"] == "UNPLACED" and len(d["matching_rows_in_table_order"]) == 8
    b = R.outcome_block(d, "A")
    assert b["committed_next"] is None and "UNPLACED" in b["note"]
    monkeypatch.setattr(R, "outcome_rows", lambda *a, **k: {n: False for n in R.OUTCOME_NAMES})
    assert R.decide_li(True, held_all(), both(), both(), both(), both())["OUTCOME"] == "UNPLACED"


def test_decide_arm_reads_the_per_seed_composition():
    m, r = real_pairs()
    per = {s: R.compose_seed(m, r) for s in SEEDS}
    d = R.decide_arm(True, per)
    assert d["OUTCOME"] == "FIXED / ADOPTED" and d["adopts"] is True and d["flags"]["R-MOVE"] == {"s0": True, "s1": True}
    per[1] = R.compose_seed(m, pair(x_y=table()))
    assert R.decide_arm(True, per)["OUTCOME"] == "PARTIAL / ONE SEED"
    # R-KEEP and R-NOBASE are read as their own per-seed columns (the table ANDs them; the record keeps them apart)
    per = {s: R.compose_seed(m, r) for s in SEEDS}
    per[1] = R.compose_seed(pair(x_y=clearing(), g=g_ok(keep=0.07)), r)
    d = R.decide_arm(True, per)
    assert d["OUTCOME"] == "FIXED / NOT ADOPTED" and d["flags"]["R-KEEP"] == {"s0": True, "s1": False} and d["flags"]["R-NOBASE"] == {"s0": True, "s1": True}
    per = {s: R.compose_seed(m, r) for s in SEEDS}
    per[0]["flags"]["R-NOBASE"] = False
    d = R.decide_arm(True, per)
    assert d["OUTCOME"] == "FIXED / NOT ADOPTED" and d["flags"]["R-NOBASE"] == {"s0": False, "s1": True} and d["flags"]["R-KEEP"] == {"s0": True, "s1": True}
    assert R.decide_arm(False, per)["OUTCOME"] == "REFUSED"


def test_the_outcome_table_text_is_the_registered_one():
    assert R.OUTCOME_NAMES == ("REFUSED", "HARMS", "FIXED / ADOPTED", "FIXED / NOT ADOPTED", "PARTIAL / ADOPTED", "PARTIAL / NOT ADOPTED", "PARTIAL / ONE SEED", "NO EFFECT")
    assert R.ADOPTING == ("FIXED / ADOPTED", "PARTIAL / ADOPTED") and R.SENDS_TO_B == ("HARMS", "FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED")
    assert R.table_row("HARMS")[0] == "controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed"


def test_committed_next_per_arm():
    assert R.next_for_arm("HARMS", "A").startswith("not adopted; read B under this table") and "NEXT-LAUNCH" not in R.next_for_arm("HARMS", "A")
    assert R.next_for_arm("HARMS", "B").startswith("not adopted; the launch weight cannot be dosed") and "NEXT-LAUNCH" in R.next_for_arm("HARMS", "B")
    assert "B (the lower dose) under this table" in R.next_for_arm("FIXED / NOT ADOPTED", "A") and R.next_for_arm("FIXED / NOT ADOPTED", "B") == "NEXT-LAUNCH"
    assert R.next_for_arm("PARTIAL / NOT ADOPTED", "A") == R.next_for_arm("FIXED / NOT ADOPTED", "A") and R.next_for_arm("PARTIAL / NOT ADOPTED", "B") == "NEXT-LAUNCH"
    for n in ("REFUSED", "FIXED / ADOPTED", "PARTIAL / ADOPTED", "PARTIAL / ONE SEED", "NO EFFECT"):
        assert R.next_for_arm(n, "A") == R.next_for_arm(n, "B") == R.table_row(n)[1]


def test_the_dose_cascade():
    blk = lambda name: {"OUTCOME": name}              # noqa: E731
    for name in R.SENDS_TO_B:
        assert R.cascade(blk(name), None)["final_arm"] == "A" and "PENDING" in R.cascade(blk(name), None)["b_status"]
        c = R.cascade(blk(name), blk("NO EFFECT"))
        assert c["final_arm"] == "B" and c["final"] == "NO EFFECT"
    for name in set(R.OUTCOME_NAMES) - set(R.SENDS_TO_B):
        c = R.cascade(blk(name), blk("HARMS"))
        assert c["final_arm"] == "A" and c["final"] == name and "beside" in c["b_status"]
        assert R.cascade(blk(name), None)["b_status"] == "not read"
    assert R.cascade(blk("HARMS"), None)["final"] == "HARMS"


def test_next_launch_routing():
    assert R.next_launch(True) == "LOOP-STATE" and R.next_launch(False) == "LAUNCH-INTENT-HI" and R.next_launch(None).startswith("UNDECIDED")


# ------------------------------------------------------------------------------------------------ Read A on adoption
def readA_rows(over=None):
    t = {l: flat() for l in R.LABELS}
    t["ADE6"] = row(-0.30, -0.4, -0.2)
    for l in ("along-track MAE", "heading MAE", "TAC-LAT", "S-POS@30"):
        t[l] = better(l, 0.1)
    t.update(over or {})
    return t


def ra(over0=None, over1=None):
    return R.read_a({0: readA_rows(over0), 1: readA_rows(over1)})


def test_read_a_dominant_and_each_condition():
    assert ra()["outcome"] == "DOMINANT" and ra()["D1_both"] and ra()["D2_both"] and ra()["D3_both"] and ra()["crawl_rows_pass"]
    assert R.TWO_F == 0.2415
    assert ra({"ADE6": row(-0.2415, -0.3, -0.1)})["D1_both"] is True and ra({"ADE6": row(-0.2414, -0.3, -0.1)})["outcome"] == "NO MARGIN"
    assert ra(over1={"ADE6": row(-0.30, -0.4, 0.01)})["outcome"] == "NO MARGIN"                    # one seed not separated
    assert ra({"speed MAE": worse("speed MAE", 0.1)})["outcome"] == "MARGIN, NOT DOMINANT" and ra({"speed MAE": worse("speed MAE", 0.1)})["per_seed"]["s0"]["D2_worse_rows"] == ["speed MAE"]
    no_tac = {"TAC-LAT": flat(), "TAC-LON": flat()}
    assert ra(no_tac)["outcome"] == "MARGIN, NOT DOMINANT" and ra(no_tac)["per_seed"]["s0"]["D3_families"]["TACTICAL"] is False
    assert ra(over1={"M1 crawl share": worse("M1 crawl share", 0.05)})["outcome"] == "MARGIN, NOT DOMINANT"           # worse row is D2 as well
    assert ra({"L2 curvature MAE (GT>=5.2)": row(0.02, -0.05, 0.1)}) ["outcome"] == "DOMINANT"                       # worse point but not separated: crawl passes


def test_read_a_each_family_needs_its_own_better_row():
    for fam, lab in (("LONGITUDINAL", "along-track MAE"), ("LATERAL", "heading MAE"), ("TACTICAL", "TAC-LAT"), ("STRATEGIC", "S-POS@30")):
        r = ra({lab: flat()})                                                                  # that family's only better row is flat at s0
        assert r["outcome"] == "MARGIN, NOT DOMINANT" and r["per_seed"]["s0"]["D3_families"][fam] is False and r["per_seed"]["s1"]["D3_families"][fam] is True and r["D3_both"] is False
        assert all(v for f, v in r["per_seed"]["s0"]["D3_families"].items() if f != fam)
    assert ra()["per_seed"]["s0"]["D3_families"] == {"PRIMARY": True, "LONGITUDINAL": True, "LATERAL": True, "TACTICAL": True, "STRATEGIC": True}


def test_read_a_crawl_rows_are_m1_and_l2_at_either_seed():
    l2, m1 = "L2 curvature MAE (GT>=5.2)", "M1 crawl share"
    r = ra({l2: worse(l2, 0.05)})
    assert r["crawl_rows_pass"] is False and r["per_seed"]["s0"]["L2_sep_worse"] is True and r["per_seed"]["s0"]["M1_sep_worse"] is False and r["per_seed"]["s1"]["L2_sep_worse"] is False
    r = ra(over1={l2: worse(l2, 0.05)})
    assert r["crawl_rows_pass"] is False and r["per_seed"]["s1"]["L2_sep_worse"] is True
    r = ra({m1: worse(m1, 0.05)})
    assert r["crawl_rows_pass"] is False and r["per_seed"]["s0"]["M1_sep_worse"] is True and r["per_seed"]["s0"]["L2_sep_worse"] is False
    r = ra(over1={m1: worse(m1, 0.05)})
    assert r["crawl_rows_pass"] is False and r["per_seed"]["s1"]["M1_sep_worse"] is True
    assert ra()["crawl_rows_pass"] is True


def test_read_a_crawl_open_is_dominant_only_when_the_crawl_rows_are_not_worse():
    # a crawl row separated-worse is also a D2 failure, so the label with D2 and D3 held but the crawl open cannot arise from separated-worse rows alone: it is the unreachable branch
    r = ra({"M1 crawl share": worse("M1 crawl share", 0.05)})
    assert r["crawl_rows_pass"] is False and r["outcome"] != "DOMINANT"


# ------------------------------------------------------------------------------------------------ floors, zeros, families
def test_rf_is_max_over_seeds_of_abs_delta_and_rejects_missing_rows():
    t = {0: table({"ADE6": row(0.4)}), 1: table({"ADE6": row(-0.5)})}
    f = R.floors_from(t)
    assert f["ADE6"] == 0.5 and f["speed MAE"] == 0.0 and set(f) == set(R.LABELS)
    t[1]["speed MAE"] = {"status": "NOT COMPUTED"}
    with pytest.raises(R.RowNotComputed):
        R.floors_from(t)


def test_exact_zero():
    z = {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False}
    assert R.is_exact_zero({l: z for l in R.LABELS}) is True
    assert R.is_exact_zero({"a": z, "b": {**z, "delta": 1e-12}}) is False and R.is_exact_zero({"a": z, "b": {**z, "hi": 1e-12}}) is False and R.is_exact_zero({"a": z, "b": {**z, "lo": -1e-12}}) is False
    assert R.is_exact_zero({}) is False and R.is_exact_zero({"a": {"status": "NOT COMPUTED"}}) is False
    assert R.is_exact_zero({"a": z, "b": {"status": "NOT COMPUTED"}}) is True


def test_families_report():
    f = R.families_report({"X - Y": table(), "X - P10": table()})
    assert set(f) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"} and f["STRATEGIC"]["paused"] is False and "NOT paused" in f["STRATEGIC"]["ruling"]
    assert set(f["LONGITUDINAL"]["rows"]) == {l for l in R.LABELS if R.FAMILY[l] == "LONGITUDINAL"} and "anti-echo" in f["LONGITUDINAL"]["note"]
    assert f["LATERAL"]["rows"]["L2 curvature MAE (GT>=5.2)"]["X - Y"] == {"delta": 0.0, "lo": -0.1, "hi": 0.1, "separated": False}
    t = table()
    t["S-POS@30"] = {"status": "NOT COMPUTED"}
    assert R.families_report({"X - Y": t})["STRATEGIC"]["rows"]["S-POS@30"]["X - Y"] == "NOT COMPUTED"
    assert R.TIER_T1 in f["TACTICAL"]["tier"]


# ------------------------------------------------------------------------------------------------ the slow-draw flag
def cells(mean=-1.0, lo=-1.2, hi=-0.8, **over):
    c = {n: (mean, lo, hi) for n in R.SPEED_CELLS}
    c.update(over)
    return c


def test_slow_draw_label_and_its_boundaries():
    assert R.slow_draw(cells())["label"] == "SLOW-T" and R.SLOW_SHIFT == 0.5
    assert R.slow_draw(cells(-0.5, -0.7, -0.3))["label"] is None                             # exactly 0.5: not more than 0.5
    assert R.slow_draw(cells(-0.5001, -0.7, -0.3))["label"] == "SLOW-T"
    assert R.slow_draw(cells(-1.0, -1.2, 0.0))["label"] is None                              # CI touches 0: not separated
    assert R.slow_draw(cells(-1.0, -1.2, -1e-9))["label"] == "SLOW-T"
    for n in R.SPEED_CELLS:                                                                   # one cell short of the rule is no label
        assert R.slow_draw(cells(**{n: (-0.5, -0.7, -0.3)}))["label"] is None
        assert R.slow_draw(cells(**{n: (-1.0, -1.2, 0.1)}))["label"] is None
        assert R.slow_draw(cells(**{n: (+1.0, 0.8, 1.2)}))["label"] is None
    assert R.slow_draw(cells(1.0, 0.8, 1.2))["label"] == "FAST-T" and R.slow_draw(cells(0.5, 0.3, 0.7))["label"] is None and R.slow_draw(cells(0.5001, 0.3, 0.7))["label"] == "FAST-T"
    assert R.slow_draw(cells(1.0, 0.0, 1.2))["label"] is None
    assert R.slow_draw(cells(-0.6, -0.9, -0.55), thr=0.7)["label"] is None
    with pytest.raises(ValueError):
        R.slow_draw({n: (-1.0, -1.2, -0.8) for n in R.SPEED_CELLS[:4]})
    with pytest.raises(ValueError):
        R.slow_draw({**cells(), "extra": (0.0, 0.0, 0.0)})
    assert set(R.slow_draw(cells())["cells"]) == set(R.SPEED_CELLS) and R.SPEED_CELLS == ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")


def cons_world(n_ep=8, per=10, shift=-1.0, seed=3):
    rng = np.random.default_rng(seed)
    n = n_ep * per
    eid = np.repeat(np.arange(n_ep), per)
    y = rng.normal(size=(n, len(R.CONS_NAMES)))
    x = y.copy()
    for c in R.SPEED_CELLS:
        x[:, R.CONS_NAMES.index(c)] += shift + rng.normal(scale=0.05, size=n)
    return x, y, eid


def test_speed_cells_reads_the_five_speed_columns_and_flags_a_shifted_arm():
    x, y, eid = cons_world()
    c = R.speed_cells(x, y, eid, n_boot=200)
    assert set(c) == set(R.SPEED_CELLS) and all(-1.2 < c[n][0] < -0.8 and c[n][1] <= c[n][0] <= c[n][2] for n in c)
    assert R.slow_draw(c)["label"] == "SLOW-T"
    x2, y2, eid2 = cons_world(shift=0.0)
    assert R.slow_draw(R.speed_cells(x2, y2, eid2, n_boot=200))["label"] is None
    x3 = y.copy()
    x3[:, R.CONS_NAMES.index("lat_theta_deg")] += 5.0                                          # a non-speed column moves: no speed cell does
    assert all(abs(v[0]) < 1e-12 for v in R.speed_cells(x3, y, eid, n_boot=50).values())
    assert R.CONS_NAMES.index("lon_v_target_ms") == 12 and R.CONS_NAMES.index("v_hi_ms") == 25 and len(R.CONS_NAMES) == 26
    with pytest.raises(ValueError):
        R.speed_cells(x[:, :25], y[:, :25], eid)
    with pytest.raises(ValueError):
        R.speed_cells(x, y[:-1], eid)


def test_speed_cells_default_rng_is_the_banked_seed_zero():
    x, y, eid = cons_world()
    c0 = R.speed_cells(x, y, eid, n_boot=200)
    assert c0 == R.speed_cells(x, y, eid, rng=np.random.default_rng(0), n_boot=200) and c0 != R.speed_cells(x, y, eid, rng=np.random.default_rng(1), n_boot=200)


def test_cell_boot_is_the_banked_method_and_consumes_a_shared_rng_in_order():
    d = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0])
    eid = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    m, lo, hi = R.cell_boot(d, eid, np.random.default_rng(0), 300)
    assert abs(m - np.nanmean(d)) < 1e-12 and lo <= m <= hi
    assert R.cell_boot(d, eid, np.random.default_rng(0), 300) == (m, lo, hi)
    rng = np.random.default_rng(0)
    first = R.cell_boot(d, eid, rng, 300)
    second = R.cell_boot(d, eid, rng, 300)
    assert first == (m, lo, hi) and second != first                                                # a shared rng advances: the banked order matters
    # pooled ratio of resampled sums to resampled counts: hand check with two episodes
    d2, e2 = np.array([0.0, 0.0, 10.0]), np.array([0, 0, 1])
    rng = np.random.default_rng(1)
    mean, lo2, hi2 = R.cell_boot(d2, e2, rng, 400)
    assert abs(mean - 10.0 / 3.0) < 1e-12 and 0.0 <= lo2 <= hi2 <= 10.0


def test_cell_boot_matches_an_independent_oracle():
    rng0 = np.random.default_rng(5)
    eid = np.repeat(np.arange(9), 7)
    d = rng0.normal(size=len(eid))
    d[3] = np.nan
    mean, lo, hi = R.cell_boot(d, eid, np.random.default_rng(0), 400)
    # the banked method (rep_s0_speedcells.py boot): per-episode sums and finite counts, ONE integers() draw of len(episodes) per replicate, ratio of resampled sums to counts, 2.5 / 97.5 percentiles
    ue = np.unique(eid)
    sums = np.array([np.nansum(d[eid == e]) for e in ue])
    cnts = np.array([np.isfinite(d[eid == e]).sum() for e in ue])
    g = np.random.default_rng(0)
    bs = []
    for _ in range(400):
        pick = g.integers(0, len(ue), len(ue))
        bs.append(sums[pick].sum() / max(cnts[pick].sum(), 1))
    assert mean == float(np.nanmean(d[np.isfinite(d)])) and lo == float(np.percentile(bs, 2.5)) and hi == float(np.percentile(bs, 97.5))
    assert R.N_BOOT == 2000
    # the banked script reads float32 cons_pred and never casts: the same oracle on a float32 vector, in its own dtype
    d32 = d.astype(np.float32)
    mean32, lo32, hi32 = R.cell_boot(d32, eid, np.random.default_rng(0), 400)
    sums32 = np.array([np.nansum(d32[eid == e]) for e in ue])
    g = np.random.default_rng(0)
    bs32 = []
    for _ in range(400):
        pick = g.integers(0, len(ue), len(ue))
        bs32.append(sums32[pick].sum() / max(cnts[pick].sum(), 1))
    assert mean32 == float(np.nanmean(d32[np.isfinite(d32)])) and lo32 == float(np.percentile(bs32, 2.5)) and hi32 == float(np.percentile(bs32, 97.5)) and (mean32, lo32, hi32) != (mean, lo, hi)


# ------------------------------------------------------------------------------------------------ run-record controls
def argv(seed, lever=None, rep=False, extra=None, drop=()):
    s = R.REP_SEED[seed] if rep else seed
    a = ["--arm", "M2b", "--seed", str(s), "--out", f"D:/r/{'rep' if rep else 'ctrl'}{seed}{'L' if lever else ''}", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2",
         "--ego-perturb-p", "0", "--dagger-p", "0.1", "--dagger-bank-sha256", R.OACC_BANK_SHA[seed], "--dagger-seed", str(seed), "--w-acc", "49.6969",
         "--init-ckpt", f"D:\\tanitad\\runs\\M2b_s{seed}\\ckpt_last.pt", "--kin-mode", "yaw"]
    if lever:
        a += [R.LEVER, lever]
    a += list(extra or [])
    out, i = [], 0
    while i < len(a):
        if a[i] in drop:
            i += 1
            while i < len(a) and not a[i].startswith("--"):
                i += 1
            continue
        out.append(a[i])
        i += 1
    return out


def fl(*a, **k):
    return R.parse_flags(argv(*a, **k))


def test_parse_flags_and_flag_diff_match_the_dag_orc_reader():
    sys.path.insert(0, str(Path(os.environ.get("LI_ORC_DIR", R.ORC_DIR))))
    try:
        import dag_orc_read as O
    except Exception:       # noqa: BLE001
        pytest.skip("dag_orc_read.py not importable here")
    for a in (argv(0), argv(1, "1.58926"), argv(0, rep=True, extra=["--ego-perturb-with-dagger"])):
        assert R.parse_flags(a) == O.parse_flags(a)
    assert R.flag_diff(fl(0), fl(0, "1.58926")) == O.flag_diff(O.parse_flags(argv(0)), O.parse_flags(argv(0, "1.58926"))) == [R.LEVER]
    assert R.flag_diff(R.parse_flags(["--lr", "1e-4"]), R.parse_flags(["--lr", "0.0001"])) == [] and R.flag_diff(R.parse_flags(["--lr", "1e-4"]), R.parse_flags(["--lr", "0.0002"])) == ["--lr"]
    assert R.flag_diff(R.parse_flags(["--out", "a"]), R.parse_flags(["--out", "b"])) == []


@pytest.mark.parametrize("seed", [0, 1])
def test_k_pair_for_each_pair(seed):
    ctrl, rep = fl(seed), fl(seed, rep=True)
    assert R.k_pair("A", seed, fl(seed, "1.58926"), ctrl)[0] is True
    assert R.k_pair("Arep", seed, fl(seed, "1.58926", rep=True), rep)[0] is True
    assert R.k_pair("REP", seed, rep, ctrl)[0] is True
    assert R.k_pair("B", seed, fl(seed, "0.726175"), ctrl, fl(seed, "1.58926"))[0] is True
    assert R.k_pair("Brep", seed, fl(seed, "0.726175", rep=True), rep, fl(seed, "1.58926", rep=True))[0] is True
    # failures
    assert R.k_pair("A", seed, fl(seed, "1.58927"), ctrl)[0] is False                                      # wrong W
    assert R.k_pair("A", seed, fl(seed), ctrl)[0] is False                                                 # lever absent
    assert R.k_pair("A", seed, fl(seed, "1.58926", extra=["--extra-flag", "1"]), ctrl)[0] is False        # another difference
    assert R.k_pair("A", seed, fl(seed, "1.58926", rep=True), ctrl)[0] is False                            # the replicate's seed
    assert R.k_pair("Arep", seed, fl(seed, "1.58926", rep=True), ctrl)[0] is False                         # against the wrong control
    assert R.k_pair("Arep", seed, fl(seed, "1.58926"), rep)[0] is False
    assert R.k_pair("B", seed, fl(seed, "1.58926"), ctrl, fl(seed, "1.58926"))[0] is False                 # B at A's W
    assert R.k_pair("B", seed, fl(seed, "0.726175", extra=["--steps", "10"]), ctrl, fl(seed, "1.58926"))[0] is False
    assert R.k_pair("B", seed, fl(seed, "0.726175"), ctrl, fl(seed, "1.58926", extra=["--x", "1"]))[0] is False    # B differs from A in more than the lever
    assert R.k_pair("REP", seed, fl(seed), ctrl)[0] is False                                               # REP at the main seed
    assert R.k_pair("REP", seed, fl(seed, rep=True, extra=["--ego-perturb-with-dagger"]), ctrl)[0] is False
    assert R.k_pair("REP", 1 - seed, rep, fl(1 - seed))[0] is False                                        # REP of the other seed
    assert R.k_pair("HI", seed, ctrl, ctrl)[0] is False


@pytest.mark.parametrize("role,lever,rep", [("CTRL", None, False), ("REP", None, True), ("A", "1.58926", False), ("Arep", "1.58926", True), ("B", "0.726175", False), ("Brep", "0.726175", True)])
def test_k_registered_names_the_fine_tune(role, lever, rep):
    for seed in (0, 1):
        assert R.k_registered(role, seed, fl(seed, lever, rep=rep))[0] is True
    seed = 1
    for drop in ("--w-acc", "--dagger-bank-sha256", "--dagger-seed", "--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p", "--arm", "--init-ckpt", "--seed"):
        assert R.k_registered(role, seed, fl(seed, lever, rep=rep, drop=(drop,)))[0] is False, drop
    assert R.k_registered(role, 0, fl(1, lever, rep=rep))[0] is False                                   # the other seed's bank, dagger-seed, init
    assert R.k_registered(role, seed, fl(seed, lever, rep=rep, extra=["--ego-perturb-with-dagger"]))[0] is False
    bad = argv(seed, lever, rep=rep)
    bad[bad.index("--w-acc") + 1] = "49.6970"
    assert R.k_registered(role, seed, R.parse_flags(bad))[0] is False
    bad = argv(seed, lever, rep=rep)
    bad[bad.index("--ego-perturb-p") + 1] = "0.1"
    assert R.k_registered(role, seed, R.parse_flags(bad))[0] is False
    bad = argv(seed, lever, rep=rep)
    bad[bad.index("--dagger-p") + 1] = "0.0"
    assert R.k_registered(role, seed, R.parse_flags(bad))[0] is False
    bad = argv(seed, lever, rep=rep)
    bad[bad.index("--init-ckpt") + 1] = "D:/x/M2b_s0/ckpt_last.pt"
    assert R.k_registered(role, seed, R.parse_flags(bad))[0] is False


def test_k_registered_a_flag_named_twice_or_with_two_values_fails():
    assert R.k_registered("CTRL", 0, R.parse_flags(argv(0) + ["--steps", "3000"]))[0] is False
    assert R.k_registered("CTRL", 0, R.parse_flags(argv(0) + ["--w-acc", "49.6969"]))[0] is False
    two = argv(0)
    two.insert(two.index("--steps") + 2, "3000")
    assert R.k_registered("CTRL", 0, R.parse_flags(two))[0] is False
    assert R.one_value({"--a": [["1"]]}, "--a") == (True, "1") and R.one_value({"--a": [["1"], ["1"]]}, "--a")[0] is False and R.one_value({"--a": [["1", "2"]]}, "--a")[0] is False and R.one_value({}, "--a")[0] is False


def test_k_registered_lever_presence_and_value_by_role():
    for role in ("CTRL", "REP"):
        assert R.k_registered(role, 0, fl(0, "1.58926", rep=(role == "REP")))[0] is False             # the lever on a control
    for role, w in (("A", "1.58926"), ("Arep", "1.58926"), ("B", "0.726175"), ("Brep", "0.726175")):
        rep = role.endswith("rep")
        assert R.k_registered(role, 0, fl(0, rep=rep))[0] is False                                    # lever absent on a trained arm
        other = "0.726175" if w == "1.58926" else "1.58926"
        assert R.k_registered(role, 0, fl(0, other, rep=rep))[0] is False
    assert R.k_registered("A", 0, fl(0, "1.58926", rep=True))[0] is False                             # A on the replicate seed
    assert R.k_registered("REP", 0, fl(0))[0] is False                                                # REP at the main seed


def test_k_init():
    assert R.k_init(0, [("a", R.M2B_CKPT_SHA[0]), ("b", R.M2B_CKPT_SHA[0])])[0] is True
    assert R.k_init(0, [])[0] is False and R.k_init(0, [("a", R.M2B_CKPT_SHA[1])])[0] is False and R.k_init(1, [("a", R.M2B_CKPT_SHA[1]), ("b", "0" * 64)])[0] is False
    tr = {"summary.json": {"levers": {"init_ckpt": {"sha256": "x"}}}, "selfcheck.json": {"built_levers": {"init_ckpt": {"sha256": "y"}}}}
    assert R.find_init_values(tr) == [("summary.json:levers.init_ckpt.sha256", "x"), ("selfcheck.json:built_levers.init_ckpt.sha256", "y")]


def summary_for(arm, n_frac=R.F_LAUNCH, eligible=100000, **over):
    w = float(R.W_ARM[arm])
    n = int(round(n_frac * eligible))
    s = {"launch_n": n, "launch_eligible": eligible, "launch_rows": eligible, "launch_intent_w": w, "levers": {"launch_intent_w": w}, "train_window_list_sha256": R.WINDOW_LIST_SHA}
    s["launch_weight_share"] = (1 + w) * n / ((1 + w) * n + (eligible - n))
    s.update(over)
    return s


def test_k_share_band_is_inclusive_per_arm(monkeypatch):
    for arm in ("A", "B", "HI"):
        t = R.SHARE_TARGET[arm]
        w = float(R.W_ARM[arm])
        assert R.k_share(arm, summary_for(arm))[0] is True
        e = 10_000_000
        for share, want in ((t + 0.01999, True), (t - 0.01999, True), (t + 0.02001, False), (t - 0.02001, False)):
            n = e * share / ((1 + w) - share * w)                                     # the n whose recomputed share is `share`
            assert R.k_share(arm, summary_for(arm, n_frac=n / e, eligible=e))[0] is want, (arm, share)
        # the band edges themselves, with the recomputed share pinned to the edge (the count pair cannot hit an edge exactly)
        for edge, want in ((t + 0.02, True), (t - 0.02, True), (t + 0.02 + 1e-9, False), (t - 0.02 - 1e-9, False)):
            monkeypatch.setattr(R, "share_of", lambda n, e_, w_, _s=edge: _s)
            assert R.k_share(arm, {**summary_for(arm), "launch_weight_share": edge})[0] is want, (arm, edge)
        monkeypatch.undo()
    assert R.k_share("A", summary_for("B"))[0] is False                                                  # B's W under A's key
    assert R.k_share("B", summary_for("A"))[0] is False
    assert R.SHARE_TARGET == {"A": 1 / 3, "B": 1 / 4, "HI": 1 / 2}


def test_k_share_reads_the_lever_value_of_the_arm_it_is_asked_about():
    # a summary whose recorded share and lever agree with each other but not with the arm's registered W (1.5893 for 1.58926; B's W under A's key)
    ok = summary_for("A")
    w = 1.5893
    consistent = {**ok, "launch_intent_w": w, "levers": {"launch_intent_w": w}, "launch_weight_share": R.share_of(ok["launch_n"], ok["launch_eligible"], w)}
    assert R.k_share("A", ok)[0] is True and R.k_share("A", consistent)[0] is False
    wb = float(R.W_ARM["B"])
    b_as_a = {**summary_for("A"), "launch_intent_w": wb, "levers": {"launch_intent_w": wb}, "launch_weight_share": R.share_of(ok["launch_n"], ok["launch_eligible"], wb)}
    assert R.k_share("A", b_as_a)[0] is False


def test_k_share_other_failures():
    ok = summary_for("A")
    assert R.k_share("A", {**ok, "launch_rows": ok["launch_eligible"] - 1})[0] is False
    assert R.k_share("A", {**ok, "launch_weight_share": 0.5})[0] is False and R.k_share("A", {**ok, "launch_weight_share": None})[0] is False
    assert R.k_share("A", {k: v for k, v in ok.items() if k != "launch_n"})[0] is False
    assert R.k_share("A", {**ok, "launch_eligible": 0})[0] is False and R.k_share("A", {**ok, "launch_n": ok["launch_eligible"] + 1})[0] is False and R.k_share("A", {**ok, "launch_n": True})[0] is False
    for bad_n in (ok["launch_eligible"] + 1, -1):
        okk, why = R.k_share("A", {**ok, "launch_n": bad_n})
        assert okk is False and "not a count pair" in why[0], (bad_n, why)
    okk, why = R.k_share("A", {**ok, "launch_eligible": 0})
    assert okk is False and "not a count pair" in why[0]
    assert R.k_share("A", {**ok, "levers": {}, "launch_intent_w": None})[0] is False
    assert R.k_share("A", {**ok, "levers": {"launch_intent_w": 1.5893}})[0] is False
    assert R.k_share("A", {**ok, "levers": {}, "launch_intent_w": 1.58926})[0] is True              # the top-level key is read when levers lacks it
    assert R.share_of(10, 100, 1.0) == 20 / 110


def test_k_share_control_form_for_ctrl_and_rep():
    assert R.k_share_control({"levers": {}})[0] is True and R.k_share_control({"levers": {"launch_intent_w": 0.0}})[0] is True and R.k_share_control({})[0] is True
    assert R.k_share_control({"levers": {"launch_intent_w": 1.58926}})[0] is False
    assert R.k_share_control({"levers": {}, "launch_n": 5})[0] is False and R.k_share_control({"launch_weight_share": 0.1})[0] is False
    assert R.k_share_control({"levers": {"launch_n": 3}})[0] is False and R.k_share_control({"levers": {"launch_eligible": 3}})[0] is False and R.k_share_control({"levers": {"launch_n": 0, "launch_eligible": 0}})[0] is True


def test_k_mask_window_list_and_fraction_band():
    assert R.k_mask(summary_for("A"))[0] is True
    assert R.k_mask(summary_for("A", train_window_list_sha256="0" * 64))[0] is False
    for frac, want in ((R.F_LAUNCH + 0.01999, True), (R.F_LAUNCH - 0.01999, True), (R.F_LAUNCH + 0.02001, False), (R.F_LAUNCH - 0.02001, False)):
        s = summary_for("A", n_frac=frac, eligible=10_000_000)
        assert R.k_mask(s)[0] is want, frac
    assert R.k_mask({"train_window_list_sha256": R.WINDOW_LIST_SHA})[0] is False and R.k_mask({**summary_for("A"), "launch_eligible": 0})[0] is False
    assert abs(R.F_LAUNCH - 0.161851) < 1e-6 and R.WINDOW_LIST_SHA.startswith("e144adcd") and R.WINDOW_LIST_SHA.endswith("8168")


def build_rec(changed=None, **over):
    sha = lambda c: c * 64                  # noqa: E731
    rec = {"launch_commit": "a" * 40, "base_commit": "aeb544683b1c6b70e29f241fe7133d4290a1b4a5",
           "files": {"stack/scripts/v8f_mini_train.py": {"sha256_launch": sha("1"), "sha256_base": sha("2")}, "stack/tanitad/data/v8f_mini_data.py": {"sha256_launch": sha("3"), "sha256_base": sha("3")},
                     "stack/tanitad/models/v8f_mini.py": {"sha256_launch": sha("4"), "sha256_base": sha("5")}},
           "changed": [{"file": "stack/scripts/v8f_mini_train.py", "hunk": "h1", "behind_flag": "--launch-intent-w", "default_off": True, "bit_identity_test": "PASS"},
                       {"file": "stack/tanitad/models/v8f_mini.py", "hunk": "h2", "behind_flag": "--launch-intent-w", "default_off": True, "bit_identity_test": "PASS"}]}
    rec.update(over)
    if changed is not None:
        rec["changed"] = changed
    return rec


def test_k_build():
    assert R.k_build(build_rec())[0] is True and R.k_build(None)[0] is False and R.k_build({})[0] is False
    r = build_rec()
    r["files"]["stack/tanitad/data/v8f_mini_data.py"]["sha256_launch"] = "9" * 64                         # Amendment 1: a changed data file must carry DAG-ORC r2's sha256
    r["changed"].append({"file": "stack/tanitad/data/v8f_mini_data.py", "hunk": "h3", "behind_flag": "--ego-perturb-with-dagger", "default_off": True, "bit_identity_test": "PASS"})
    assert R.k_build(r)[0] is False
    r["files"]["stack/tanitad/data/v8f_mini_data.py"]["sha256_launch"] = R.LAUNCH_PINNED_FILES["stack/tanitad/data/v8f_mini_data.py"]
    assert R.k_build(r)[0] is True                                                                       # the pinned bytes with a PASSing flag-gated hunk are admitted
    r["changed"] = [h for h in r["changed"] if h["file"] != "stack/tanitad/data/v8f_mini_data.py"]
    assert R.k_build(r)[0] is False                                                                      # the pinned bytes still need their hunks listed
    r["changed"].append({"file": "stack/tanitad/data/v8f_mini_data.py", "hunk": "h3", "behind_flag": "--ego-perturb-with-dagger", "default_off": True, "bit_identity_test": "UNVERIFIED: x"})
    assert R.k_build(r)[0] is False                                                                      # and each hunk PASSing
    assert R.k_build(build_rec(changed=[]))[0] is False                                                 # changed files, no hunk listed
    for key, val in (("default_off", False), ("bit_identity_test", "FAIL"), ("behind_flag", "")):
        ch = build_rec()["changed"]
        ch[0][key] = val
        assert R.k_build(build_rec(changed=ch))[0] is False, key
    assert R.k_build(build_rec(launch_commit="abc"))[0] is False and R.k_build(build_rec(base_commit="1" * 40))[0] is False
    r = build_rec()
    r["files"]["stack/tanitad/models/v8f_mini.py"]["sha256_base"] = ""
    assert R.k_build(r)[0] is False
    r = build_rec()
    del r["files"]["stack/scripts/v8f_mini_train.py"]
    assert R.k_build(r)[0] is False
    r = build_rec()
    for f in r["files"].values():
        f["sha256_launch"] = f["sha256_base"]
    assert R.k_build({**r, "changed": []})[0] is True                                                   # byte-identical needs no hunk
    assert R.BUILD_FILES == ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py") and R.BUILD_BASE_COMMIT == "aeb54468"
    assert R.LAUNCH_PINNED_FILES == {"stack/tanitad/data/v8f_mini_data.py": "c117e1cf5be08b9739f02e2730ce555be5a707a3183ded6481256e419f751752"}      # PREREG Amendment 1


def licence_inputs(seed=0, **over):
    rep = {"summary": {"done": True, "dagger_n": 1000, "dagger_eligible": 10000, "ego_perturb_n": 0, "levers": {"ego_perturb_stream": 0, "ego_perturb_with_dagger": False},
                       "excluded_sha256": R.EXCLUDED_SHA, "train_window_list_sha256": R.WINDOW_LIST_SHA},
           "token": {"init_ckpt_sha256": R.M2B_CKPT_SHA[seed], "bank_sha256": R.OACC_BANK_SHA[seed], "excluded_sha256": R.EXCLUDED_SHA, "w_acc": 49.6969, "binding": {"commit": "8c24823f" + "0" * 32}},
           "flags": fl(seed, rep=True)}
    oacc = {"summary": {"excluded_sha256": R.EXCLUDED_SHA, "train_window_list_sha256": R.WINDOW_LIST_SHA}, "token": {"init_ckpt_sha256": R.M2B_CKPT_SHA[seed], "bank_sha256": R.OACC_BANK_SHA[seed],
            "excluded_sha256": R.EXCLUDED_SHA, "w_acc": 49.6969, "binding": {"commit": "aeb54468" + "0" * 32}}, "flags": fl(seed)}
    return rep, oacc


def test_k_rep_licence_holds_and_each_condition_breaks_it():
    for seed in (0, 1):
        rep, oacc = licence_inputs(seed)
        assert R.k_rep_licence(rep, oacc, seed)[0] is True
    rep, oacc = licence_inputs(0)

    def broken(path, value, side="rep"):
        r, o = licence_inputs(0)
        tgt = r if side == "rep" else o
        cur = tgt
        for p in path[:-1]:
            cur = cur[p]
        cur[path[-1]] = value
        return R.k_rep_licence(r, o, 0)[0]
    cases = [(("summary", "done"), False), (("summary", "dagger_n"), 1400), (("summary", "dagger_n"), 600), (("summary", "dagger_eligible"), 0), (("summary", "ego_perturb_n"), 1),
             (("summary", "levers", "ego_perturb_with_dagger"), True), (("summary", "levers", "ego_perturb_stream"), 1), (("summary", "excluded_sha256"), "0" * 64),
             (("summary", "train_window_list_sha256"), "0" * 64), (("token", "init_ckpt_sha256"), "0" * 64), (("token", "bank_sha256"), "0" * 64), (("token", "excluded_sha256"), "0" * 64),
             (("token", "w_acc"), 49.6970), (("token", "binding", "commit"), "aeb54468" + "0" * 32)]
    for path, val in cases:
        assert broken(path, val) is False, path
    assert broken(("token", "binding", "commit"), "1" * 40, side="oacc") is False
    assert broken(("token", "w_acc"), 1.0, side="oacc") is False and broken(("summary", "excluded_sha256"), "0" * 64, side="oacc") is False
    assert broken(("token", "bank_sha256"), "0" * 64, side="oacc") is False
    for key in ("init_ckpt_sha256", "bank_sha256"):
        r, o = licence_inputs(0)
        r["token"][key] = o["token"][key] = "0" * 64                                                    # REP equals DAG-OACC, and both differ from the registered value
        assert R.k_rep_licence(r, o, 0)[0] is False, key
    r, o = licence_inputs(0)
    r["flags"] = R.parse_flags(argv(0, rep=True))
    r["flags"]["--w-acc"] = [["49.0"]]
    assert R.k_rep_licence(r, o, 0)[0] is False
    assert R.k_rep_licence(*licence_inputs(0), 1)[0] is False                                          # the seed-1 init for a seed-0 run record
    for edge, want in ((1200, True), (800, True), (1201, False), (799, False)):
        assert broken(("summary", "dagger_n"), edge) is want, edge                                      # swap share 0.1 +- 0.02, inclusive


# ------------------------------------------------------------------------------------------------ the PREREG pin
@pytest.mark.skipif(not os.path.isfile(PREREG), reason="PREREG text not beside the reader")
def test_the_prereg_carries_the_tables_the_reader_codes():
    assert R.prereg_problems(PREREG) == []
    txt = open(PREREG, encoding="utf-8").read()
    assert len(R.PREREG_OUTCOME_ROWS) == 8 and len(R.PREREG_PREDICATE_ROWS) == 7 and R.PREREG_BAR_ROW in txt
    assert "524,288 cells" in txt and "524,288" in txt and "0.726175" in txt and "1.1070" in txt


@pytest.mark.skipif(not os.path.isfile(PREREG), reason="PREREG text not beside the reader")
def test_a_changed_cell_a_missing_row_a_wrong_sha_and_crlf(tmp_path):
    txt = open(PREREG, encoding="utf-8").read()
    p = tmp_path / "p.md"
    for old, new, key in ((R.PREREG_OUTCOME_ROWS[2], R.PREREG_OUTCOME_ROWS[2].replace("both seeds", "one seed"), "outcome"), (R.PREREG_PREDICATE_ROWS[0], R.PREREG_PREDICATE_ROWS[0].replace("**and**", "**or**"), "predicate"),
                          (R.PREREG_BAR_ROW, R.PREREG_BAR_ROW.replace("2 × max(F_r, RF0_r)", "2 × max(F_r, RF0_r, RF_r)"), "bar")):
        p.write_text(txt.replace(old, new, 1), encoding="utf-8", newline="\n")
        bad = R.prereg_problems(str(p))
        assert len(bad) == 1 and (key in bad[0] or "row" in bad[0]), (key, bad)
    p.write_text(txt.replace(R.PREREG_OUTCOME_ROWS[1], "", 1), encoding="utf-8", newline="\n")
    assert len(R.prereg_problems(str(p))) == 1
    p.write_text(txt, encoding="utf-8", newline="\n")
    good = R.sha256_file(str(p))
    assert R.prereg_problems(str(p), good) == [] and len(R.prereg_problems(str(p), "0" * 64)) == 1
    p.write_bytes(txt.replace("\n", "\r\n").encode("utf-8"))
    assert R.prereg_problems(str(p)) == [] and R.prereg_problems(str(p), good) == []                    # LF-normalised before the pin


def test_strip_md_and_split_row():
    assert R.strip_md("**A:** B `x`  y") == "A: B x y" and R.split_row("| a | **b** | c |") == ["a", "**b**", "c"]
    assert R.OUTCOME_TABLE[1][0] == "HARMS" and R.OUTCOME_TABLE[0] == ("REFUSED", "any control fails", "fix and re-read; no row is reported")


# ------------------------------------------------------------------------------------------------ slots
@pytest.fixture(autouse=True)
def _few_episodes(monkeypatch):
    """The fake dump directories hold R.N_EPISODES files; 3 instead of 141 keeps the end-to-end tests off the disk (the real constants are asserted in test_roles_pairs_and_constants)."""
    monkeypatch.setattr(R, "N_EPISODES", 3)


def fake_entry(root, role, seed, with_dump=True, missing=()):
    d = Path(root) / f"{role}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in ("argv.json", "summary.json", "selfcheck.json", "token.json", "t0.json", "t0.npz", "t0f.json", "t0f.npz", "t1.json", "t1.ticks.npz", "l.json", "l.ticks.npz"):
        if f not in missing:
            (d / f).write_bytes(b"{}")
        files[f] = str(d / f)
    dd = d / "dump"
    ld = d / "ldump"
    if with_dump:
        for k in (dd, ld):
            k.mkdir(exist_ok=True)
            for i in range(R.N_EPISODES):
                (k / f"ep{i:03d}.npz").write_bytes(b"")
    e = {"train_dir": str(d), "token": files["token.json"], "ckpt_tail": f"{d.name}/ckpt_last.pt", "t0": {"json": files["t0.json"], "npz": files["t0.npz"]},
         "t1": {"tag": "t", "json": files["t1.json"], "ticks": files["t1.ticks.npz"], "dump_dir": str(dd)}}
    if role in R.ROLES_NEW:
        e["t0_fresh"] = {"json": files["t0f.json"], "npz": files["t0f.npz"]}
        e["t1_logged"] = {"tag": "tl", "json": files["l.json"], "ticks": files["l.ticks.npz"], "dump_dir": str(ld)}
    return e


def test_slot_needs_and_discovery(tmp_path):
    man = {"arms": {r: {str(s): fake_entry(tmp_path, r, s) for s in SEEDS} for r in ("A", "Arep", "CTRL", "REP")}}
    d = R.discover(man)
    assert all(d[r][s]["state"] == "PRESENT" for r in ("A", "Arep", "CTRL", "REP") for s in SEEDS) and all(d[r][s]["state"] == "MISSING" for r in ("B", "Brep") for s in SEEDS)
    needs = {r: [w for w, _ in R.slot_needs(r, man["arms"][r]["0"])] for r in ("A", "CTRL")}
    assert "t1_logged.dump_dir" in needs["A"] and "t0_fresh.json" in needs["A"] and "t1_logged.dump_dir" not in needs["CTRL"] and "t0_fresh.json" not in needs["CTRL"] and "token" in needs["CTRL"]
    man["arms"]["A"]["1"] = fake_entry(tmp_path / "gap", "A", 1, missing=("t0f.npz",))
    assert R.discover(man)["A"][1]["state"] == "PARTIAL" and "t0_fresh.npz" in R.discover(man)["A"][1]["missing"][0]
    man["arms"]["REP"]["0"] = fake_entry(tmp_path / "x", "REP", 0, with_dump=False)
    assert R.discover(man)["REP"][0]["state"] == "PARTIAL" and "t1.dump_dir" in R.discover(man)["REP"][0]["missing"][0]
    Path(man["arms"]["REP"]["0"]["t1"]["dump_dir"]).mkdir()                                              # present but empty: counted, not trusted
    assert R.discover(man)["REP"][0]["state"] == "PARTIAL" and "holds 0 of 3" in R.discover(man)["REP"][0]["missing"][0]
    man["arms"]["CTRL"]["0"] = {}
    assert R.discover(man)["CTRL"][0]["state"] == "MISSING"


# ------------------------------------------------------------------------------------------------ main_read end to end on a FAKE heavy layer
def make_world_files(root, seeds=SEEDS, with_b=False, bad_share=False, bad_lever_w=False):
    """argv / summary / selfcheck / token / lanes for A, Arep, CTRL, REP (+ B, Brep): the run records the real controls read."""
    man = {"seeds": [0, 1], "arms": {}, "build": build_rec()}
    spec_ = [("CTRL", None, False), ("REP", None, True), ("A", "1.58926", False), ("Arep", "1.58926", True)] + ([("B", "0.726175", False), ("Brep", "0.726175", True)] if with_b else [])
    for role, lever, rep in spec_:
        man["arms"][role] = {}
        for s in seeds:
            e = fake_entry(root, role, s)
            a = argv(s, lever, rep=rep)
            d = Path(e["train_dir"])
            (d / "argv.json").write_text(json.dumps({"argv": a, "sha": "x"}), encoding="utf-8")
            arm_key = R.ARM_OF_ROLE.get(role)
            summ = {"done": True, "steps": 3000, "arm": "M2b", "seed": R.REP_SEED[s] if rep else s, "levers": {"init_ckpt": {"sha256": R.M2B_CKPT_SHA[s]}}, "train_window_list_sha256": R.WINDOW_LIST_SHA,
                    "excluded_sha256": R.EXCLUDED_SHA, "dagger_n": 1000, "dagger_eligible": 10000, "ego_perturb_n": 0}
            if arm_key:
                ls = summary_for(arm_key)
                if bad_share and role == "Arep" and s == 1:
                    ls = summary_for(arm_key, n_frac=0.30)
                summ.update({k: v for k, v in ls.items() if k not in ("levers", "train_window_list_sha256")})
                summ["levers"].update({"launch_intent_w": float(R.W_ARM[arm_key]) + (1.0 if (bad_lever_w and role == "A" and s == 0) else 0.0)})
            else:
                summ["levers"].update({"ego_perturb_stream": 0, "ego_perturb_with_dagger": False})
            (d / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
            tok = {"init_ckpt_sha256": R.M2B_CKPT_SHA[s], "bank_sha256": R.OACC_BANK_SHA[s], "excluded_sha256": R.EXCLUDED_SHA, "w_acc": 49.6969, "binding": {"commit": ("8c24823f" if rep and role == "REP" else "aeb54468") + "0" * 32}}
            Path(e["token"]).write_text(json.dumps(tok), encoding="utf-8")
            man["arms"][role][str(s)] = e
    return man


class FakeHeavy:
    """Scripted rows: ``scen(x, y, s, label)`` -> row for contrast X - Y; ``guard(x, y, s)`` -> g dict; ``speed(x, y, s)`` -> cells.  Records every contrast asked for."""
    world = None
    calls = []

    def __init__(self, orc_dir, work_dir, n_boot):
        self.w = FakeHeavy.world
        self.slots = []

    def static_problems(self):
        return []

    rf0_calls = []

    def rf0(self, rollout_files):
        FakeHeavy.rf0_calls.append(list(rollout_files))
        return [], dict(RF0), {"rollout_files_checked": len(rollout_files)}

    def load_slot(self, ctl, role, seed, e, standin=False):
        self.slots.append((role, seed))
        ctl.add(f"FAKE-SLOT:{role}_s{seed}", not (self.w.get("bad_slot") == (role, seed)), "fake")

    def prepare(self, ctl):
        ctl.add("FAKE-PREPARE", True, "fake")

    def contrast(self, x, y):
        FakeHeavy.calls.append((x, y))
        out = {}
        for s in SEEDS:
            if x == y:
                out[s] = {l: {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False, "n": "2399/141"} for l in R.LABELS}
            else:
                out[s] = {l: self.w["scen"](x, y, s, l) for l in R.LABELS}
        return out

    def guard_rows(self, ctl, x, y, s, tab):
        g = g_ok()
        g.update(self.w.get("guard", lambda x, y, s: {})(x, y, s))
        return g

    def speed(self, x, y, s, rng=None):
        return self.w.get("speed", lambda x, y, s: cells(0.0, -0.1, 0.1))(x, y, s)

    def finish(self, ctl):
        ctl.add("FAKE-FINISH", True, "fake")


def world(scen=None, **kw):
    base = lambda x, y, s, l: flat()           # noqa: E731
    return {"scen": scen or base, **kw}


def scen_from(over):
    def scen(x, y, s, l):
        for key in ((x, y, s, l), (x, y, None, l)):
            if key in over:
                return over[key]
        return flat()
    return scen


def clear_all(pairs=(("A", "CTRL"), ("Arep", "REP")), seeds=SEEDS):
    return {(x, y, s, l): better(l, BARS[l]) for x, y in pairs for s in seeds for l in R.CRAWL_SET}


def run_read(tmp_path, monkeypatch, w, with_b=False, name="o", **mk):
    FakeHeavy.world = w
    FakeHeavy.calls = []
    FakeHeavy.rf0_calls = []
    root = tmp_path / name
    man = make_world_files(root, with_b=with_b, **mk)
    mp = tmp_path / f"{name}.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    out = tmp_path / f"{name}_read.json"
    monkeypatch.setattr(R, "Heavy", FakeHeavy)
    rc = R.main(["--manifest", str(mp), "--prereg", PREREG, "--out", str(out)])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


needs_prereg = pytest.mark.skipif(not os.path.isfile(PREREG), reason="PREREG text not beside the reader")


@needs_prereg
def test_e2e_every_outcome_through_main_read(tmp_path, monkeypatch):
    rc, out = run_read(tmp_path, monkeypatch, world(), name="flat")
    assert rc == 0 and out["controls_PASS"] and all(out["controls_PASS"].values())
    assert out["OUTCOME"]["OUTCOME"] == "NO EFFECT" and out["OUTCOME"]["arm"] == "A" and out["cascade"]["final_arm"] == "A" and "NEXT-LAUNCH" in out["OUTCOME"]["committed_next"]
    assert out["next_launch"].startswith("UNDECIDED") and out["read_a_on_adoption"] == "not applicable: no arm adopts" and out["B"] is None
    assert out["strategic_family"].startswith("NOT PAUSED") and len(FakeHeavy.rf0_calls) == 1 and len([f for f in FakeHeavy.rf0_calls[0] if f]) == 2 * 2 * 2 * (3 + 2) and out["provenance"]["claim_row"] == "H-V8F-LAUNCH-INTENT-1"
    assert set(out["four_families"]["A"]["s0"]) == {"main", "rep"} and "M-LAUNCH" in out["not_built"] and out["tier"]["T1"] == R.TIER_T1 and out["estimator"] == R.ESTIMATOR
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(clear_all())), name="fixed")
    assert out["OUTCOME"]["OUTCOME"] == "FIXED / ADOPTED" and out["OUTCOME"]["adopts"] is True and out["next_launch"] is None
    assert out["read_a_on_adoption"]["A"]["main"]["outcome"] in ("NO MARGIN", "MARGIN, NOT DOMINANT", "DOMINANT", "DOMINANT, CRAWL OPEN") and set(out["read_a_on_adoption"]["A"]) == {"main", "rep_beside"}
    # a lever effect needs BOTH pairs: the main pair alone moves nothing
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(clear_all(pairs=(("A", "CTRL"),)))), name="mainonly")
    assert out["OUTCOME"]["OUTCOME"] == "NO EFFECT"
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(clear_all(pairs=(("Arep", "REP"),)))), name="reponly")
    assert out["OUTCOME"]["OUTCOME"] == "NO EFFECT"
    # one seed with both pairs: PARTIAL / ONE SEED
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(clear_all(seeds=(0,)))), name="oneseed")
    assert out["OUTCOME"]["OUTCOME"] == "PARTIAL / ONE SEED" and out["OUTCOME"]["flags"]["R-MOVE"] == {"s0": True, "s1": False} and out["next_launch"].startswith("UNDECIDED")
    # both seeds, main pair at s0 and replicate pair at s1 only: no seed has two draws
    over = {**clear_all(pairs=(("A", "CTRL"),), seeds=(0,)), **clear_all(pairs=(("Arep", "REP"),), seeds=(1,))}
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(over)), name="split")
    assert out["OUTCOME"]["OUTCOME"] == "NO EFFECT"


@needs_prereg
def test_e2e_guards_in_either_pair_and_adoption_through_main_read(tmp_path, monkeypatch):
    allclear = clear_all()
    worse_ade = {("Arep", "REP", 1, "ADE6"): worse("ADE6", 0.05)}                                    # R-ADE fails in the REPLICATE pair at s1 only
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from({**allclear, **worse_ade})), name="ade")
    assert out["OUTCOME"]["OUTCOME"] == "HARMS" and out["OUTCOME"]["flags"]["guards_held_per_seed"]["s1"]["R-ADE"] is False and out["OUTCOME"]["flags"]["guards_held_per_seed"]["s0"]["R-ADE"] is True
    assert out["cascade"]["final_arm"] == "A" and "PENDING" in out["cascade"]["b_status"] and out["OUTCOME"]["committed_next"].startswith("not adopted; read B")
    t0 = {"guard": lambda x, y, s: {"T0_point": 0.07} if (x, y, s) == ("A", "CTRL", 0) else {}}
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(allclear), **t0), name="t0")
    assert out["OUTCOME"]["OUTCOME"] == "HARMS" and out["rows"]["A"]["s0"]["pairs"]["main"]["guards"]["R-T0"] is False and out["rows"]["A"]["s0"]["guards"]["R-T0"] is False
    f1 = {"guard": lambda x, y, s: {"R-TLON": {"delta": -0.05, "lo": -0.06, "hi": -0.04, "separated": True}} if (x, y, s) == ("Arep", "REP", 0) else {}}
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(allclear), **f1), name="f1")
    assert out["OUTCOME"]["OUTCOME"] == "HARMS" and out["rows"]["A"]["s0"]["guards"]["R-TLON"] is False
    # R-NEW in the replicate pair reads REP - P10 for its domain: the row is in the domain only if REP - P10 is not worse
    b = BARS["heading MAE"]
    fire = {("Arep", "P10", None, "heading MAE"): worse("heading MAE", 1.0), ("Arep", "REP", 0, "heading MAE"): worse("heading MAE", b)}
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from({**allclear, **fire})), name="fire")
    assert out["OUTCOME"]["OUTCOME"] == "HARMS" and out["rows"]["A"]["s0"]["pairs"]["rep"]["R-NEW_fires"]["heading MAE"] is True
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from({**allclear, **fire, ("REP", "P10", None, "heading MAE"): worse("heading MAE", 1.0)})), name="firedom")
    assert out["OUTCOME"]["OUTCOME"] == "FIXED / ADOPTED" and "heading MAE" not in out["rows"]["A"]["s0"]["pairs"]["rep"]["domain"]
    # adoption: R-KEEP fails in the replicate pair at s0 -> FIXED / NOT ADOPTED and B is pending
    keep = {"guard": lambda x, y, s: {"KEEP_point": 0.07} if (x, y, s) == ("Arep", "REP", 0) else {}}
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(allclear), **keep), name="keep")
    assert out["OUTCOME"]["OUTCOME"] == "FIXED / NOT ADOPTED" and out["cascade"]["final_arm"] == "A" and "B (the lower dose)" in out["OUTCOME"]["committed_next"] and out["read_a_on_adoption"] == "not applicable: no arm adopts"


@needs_prereg
def test_e2e_b_is_read_after_harms_and_decides_then(tmp_path, monkeypatch):
    harms = {("Arep", "REP", 1, "ADE6"): worse("ADE6", 0.05)}
    scen_b = scen_from({**clear_all(pairs=(("A", "CTRL"), ("Arep", "REP"))), **harms, **clear_all(pairs=(("B", "CTRL"), ("Brep", "REP")))})
    rc, out = run_read(tmp_path, monkeypatch, world(scen_b), with_b=True, name="b")
    assert rc == 0 and out["A"]["OUTCOME"] == "HARMS" and out["B"]["OUTCOME"] == "FIXED / ADOPTED" and out["OUTCOME"]["arm"] == "B" and out["cascade"]["final_arm"] == "B"
    assert ("B", "CTRL") in FakeHeavy.calls and ("Brep", "REP") in FakeHeavy.calls and out["read_a_on_adoption"].keys() == {"B"}
    # A adopted: B (present) is beside and decides nothing
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(clear_all(pairs=(("A", "CTRL"), ("Arep", "REP"), ("B", "CTRL"), ("Brep", "REP"))))), with_b=True, name="b2")
    assert out["OUTCOME"]["arm"] == "A" and out["cascade"]["final_arm"] == "A" and "beside" in out["cascade"]["b_status"]
    # B's HARMS goes to NEXT-LAUNCH
    scen_bb = scen_from({**harms, **{("Brep", "REP", 1, "ADE6"): worse("ADE6", 0.05)}, **clear_all()})
    rc, out = run_read(tmp_path, monkeypatch, world(scen_bb), with_b=True, name="b3")
    assert out["OUTCOME"]["arm"] == "B" and out["OUTCOME"]["OUTCOME"] == "HARMS" and "NEXT-LAUNCH" in out["OUTCOME"]["committed_next"] and out["next_launch"].startswith("UNDECIDED")


@needs_prereg
def test_e2e_the_slow_draw_flag_and_rf_are_beside_and_decide_nothing(tmp_path, monkeypatch):
    slow = {"speed": lambda x, y, s: cells() if (x, y, s) == ("Arep", "REP", 0) else cells(0.0, -0.1, 0.1)}
    rf = lambda x, y, s, l: row(7.0, 6.0, 8.0) if (x, y) == ("REP", "CTRL") else None           # noqa: E731
    over = clear_all()
    base = scen_from(over)

    def scen(x, y, s, l):
        r = rf(x, y, s, l)
        return r if r is not None else base(x, y, s, l)
    rc, out = run_read(tmp_path, monkeypatch, world(scen, **slow), name="slow")
    assert out["OUTCOME"]["OUTCOME"] == "FIXED / ADOPTED"                                           # neither a huge RF nor a slow-T draw changes the word
    sd = out["beside"]["A"]["slow_draw"]
    assert sd["rep:Arep-REP:s0"]["label"] == "SLOW-T" and sd["rep:Arep-REP:s1"]["label"] is None and sd["main:A-CTRL:s0"]["label"] is None and len(sd) == 4
    assert out["floors"]["RF"]["ADE6"] == 7.0 and out["floors"]["bar"]["ADE6"] == BARS["ADE6"] and out["beside"]["A"]["RF_beside_every_row"]["yaw-rate MAE"] == {"RF": 7.0, "RF0": RF0["yaw-rate MAE"], "bar": BARS["yaw-rate MAE"]}


@needs_prereg
def test_e2e_the_tables_asked_for_are_the_registered_contrasts(tmp_path, monkeypatch):
    run_read(tmp_path, monkeypatch, world(), name="calls")
    asked = set(FakeHeavy.calls)
    assert asked == {("REP", "REP"), ("REP", "CTRL"), ("CTRL", "P10"), ("REP", "P10"), ("CTRL", "CTRL"), ("A", "A"), ("Arep", "Arep"), ("A", "P10"), ("A", "CTRL"), ("Arep", "P10"), ("Arep", "REP")}


@needs_prereg
def test_e2e_a_failed_control_refuses_and_reports_no_row(tmp_path, monkeypatch, capsys):
    cases = {
        "slot": dict(w=world(scen_from(clear_all()), bad_slot=("REP", 1)), key="FAKE-SLOT:REP_s1"),
        "share": dict(w=world(scen_from(clear_all())), mk=dict(bad_share=True), key="K-SHARE:Arep_s1"),
        "lever_w": dict(w=world(scen_from(clear_all())), mk=dict(bad_lever_w=True), key="K-SHARE:A_s0"),
    }
    for name, c in cases.items():
        rc, out = run_read(tmp_path, monkeypatch, c["w"], name=name, **c.get("mk", {}))
        assert rc == 2 and out["OUTCOME"] == "REFUSED" and c["key"] in out["failed"] and "rows" not in out and "A" not in out and "read_a" not in out, (name, out and list(out))
    # an unreadable zero: (X, X) must read exactly 0.0
    def scen(x, y, s, l):
        return flat()
    class Dirty(FakeHeavy):
        def contrast(self, x, y):
            t = super().contrast(x, y)
            if (x, y) == ("A", "A"):
                t[0]["ADE6"] = {"delta": 1e-9, "lo": 0.0, "hi": 0.0, "separated": False}
            return t
    monkeypatch.setattr(R, "Heavy", Dirty)
    FakeHeavy.world = world(scen_from(clear_all()))
    man = make_world_files(tmp_path / "zero")
    (tmp_path / "zero.json").write_text(json.dumps(man), encoding="utf-8")
    rc = R.main(["--manifest", str(tmp_path / "zero.json"), "--prereg", PREREG, "--out", str(tmp_path / "zero_read.json")])
    out = json.loads((tmp_path / "zero_read.json").read_text(encoding="utf-8"))
    assert rc == 2 and out["OUTCOME"] == "REFUSED" and "K-ZERO:A-A:s0" in out["failed"]


@needs_prereg
def test_e2e_k_build_k_p0_and_static_are_controls(tmp_path, monkeypatch):
    FakeHeavy.world = world(scen_from(clear_all()))
    man = make_world_files(tmp_path / "b")
    man["build"]["changed"] = []
    p = tmp_path / "b.json"
    p.write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(R, "Heavy", FakeHeavy)
    rc = R.main(["--manifest", str(p), "--prereg", PREREG, "--out", str(tmp_path / "b_out.json")])
    assert rc == 2 and "K-BUILD" in json.loads((tmp_path / "b_out.json").read_text(encoding="utf-8"))["failed"]

    class BadStatic(FakeHeavy):
        def static_problems(self):
            return ["pin moved"]

        def rf0(self, rollout_files):
            return ["RF0 is not the frozen file"], None, {}
    monkeypatch.setattr(R, "Heavy", BadStatic)
    man = make_world_files(tmp_path / "c")
    (tmp_path / "c.json").write_text(json.dumps(man), encoding="utf-8")
    rc = R.main(["--manifest", str(tmp_path / "c.json"), "--prereg", PREREG, "--out", str(tmp_path / "c_out.json")])
    failed = json.loads((tmp_path / "c_out.json").read_text(encoding="utf-8"))["failed"]
    assert rc == 2 and "STATIC" in failed and "K-P0" in failed


@needs_prereg
def test_e2e_missing_slots_bad_prereg_and_row_not_computed_write_nothing_decisive(tmp_path, monkeypatch, capsys):
    FakeHeavy.world = world()
    monkeypatch.setattr(R, "Heavy", FakeHeavy)
    man = make_world_files(tmp_path / "m")
    os.remove(man["arms"]["Arep"]["1"]["t0"]["npz"])
    p, out = tmp_path / "m.json", tmp_path / "m_out.json"
    p.write_text(json.dumps(man), encoding="utf-8")
    assert R.main(["--manifest", str(p), "--prereg", PREREG, "--out", str(out)]) == 4 and not out.exists() and "t0.npz" in capsys.readouterr().err
    man = make_world_files(tmp_path / "m2", with_b=True)
    del man["arms"]["Brep"]["0"]
    p.write_text(json.dumps(man), encoding="utf-8")
    assert R.main(["--manifest", str(p), "--prereg", PREREG, "--out", str(out)]) == 4 and not out.exists() and "half-present" in capsys.readouterr().err
    man = make_world_files(tmp_path / "m2b", with_b=True)
    for r_ in ("B", "Brep"):
        del man["arms"][r_]["1"]                                                                         # B and B_rep at seed 0 only: present, but not a two-seed arm
    p.write_text(json.dumps(man), encoding="utf-8")
    assert R.main(["--manifest", str(p), "--prereg", PREREG, "--out", str(out)]) == 4 and not out.exists() and "half-present" in capsys.readouterr().err
    man = make_world_files(tmp_path / "m3")
    p.write_text(json.dumps(man), encoding="utf-8")
    bad_prereg = tmp_path / "bad.md"
    bad_prereg.write_text("not the PREREG", encoding="utf-8")
    assert R.main(["--manifest", str(p), "--prereg", str(bad_prereg), "--out", str(out)]) == 2 and not out.exists() and "PREREG pin" in capsys.readouterr().err
    assert R.main(["--manifest", str(p), "--out", str(out)]) == 2 and not out.exists()
    nc = {"scen": lambda x, y, s, l: {"status": "NOT COMPUTED"} if (x, y, s, l) == ("A", "CTRL", 0, "speed MAE") else flat()}
    FakeHeavy.world = nc
    rc = R.main(["--manifest", str(p), "--prereg", PREREG, "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 2 and not out.exists() and "RowNotComputed" in err and "speed MAE" in err


@needs_prereg
def test_cli_static_check_and_required_arguments(tmp_path, capsys):
    assert R.main(["--static-check", "--prereg", PREREG]) == 0 and "STATIC-OK" in capsys.readouterr().out
    assert R.main(["--static-check"]) == 2
    assert R.main(["--static-check", "--prereg", PREREG, "--prereg-sha", "0" * 64]) == 2
    with pytest.raises(SystemExit):
        R.main([])


def test_roles_pairs_and_constants():
    assert REAL_N_EPISODES == 141 and REAL_N_WINDOWS == 2399                                    # the autouse fixture shrinks R.N_EPISODES for the fake dumps; the module constant is read at import
    assert R.pair_roles("A") == {"main": ("A", "CTRL"), "rep": ("Arep", "REP")} and R.pair_roles("B") == {"main": ("B", "CTRL"), "rep": ("Brep", "REP")}
    assert R.run_seed_of("A", 1) == 1 and R.run_seed_of("Arep", 1) == 11 and R.run_seed_of("REP", 0) == 10 and R.run_seed_of("CTRL", 0) == 0 and R.run_seed_of("Brep", 0) == 10
    assert R.W_ARM == {"A": "1.58926", "B": "0.726175", "HI": "4.17852"} and R.lever_w("Arep") == "1.58926" and R.lever_w("Brep") == "0.726175" and R.lever_w("CTRL") is None
    assert R.RF0_SHA256.startswith("1608d70b") and R.CLAIM_ROW == "H-V8F-LAUNCH-INTENT-1"
    assert abs(sum([0.0]) + R.SHARE_TARGET["A"] - 1 / 3) < 1e-15 and R.SHARE_TOL == 0.02 and R.MASK_TOL == 0.02
    assert R.GUARDS == ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW") and R.FLAG_KEYS == ("R-CRAWL", "R-MOVE", "R-KEEP", "R-NOBASE")


@needs_prereg
def test_e2e_partial_adopted_and_read_a_on_adoption(tmp_path, monkeypatch):
    # R-CRAWL fails at s1 in the REPLICATE pair only (Arep - P10 yaw-rate separated-worse); R-MOVE, the guards, R-KEEP and R-NOBASE hold: PARTIAL / ADOPTED
    over = {**clear_all(), ("Arep", "P10", 1, "yaw-rate MAE"): worse("yaw-rate MAE", 0.3)}
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(over)), name="pa")
    assert out["OUTCOME"]["OUTCOME"] == "PARTIAL / ADOPTED" and out["OUTCOME"]["adopts"] is True and out["OUTCOME"]["flags"]["R-CRAWL"] == {"s0": True, "s1": False}
    assert out["next_launch"].startswith("UNDECIDED") and out["rows"]["A"]["s1"]["pairs"]["rep"]["R-CRAWL_rows"]["yaw-rate MAE"] is False and out["rows"]["A"]["s1"]["pairs"]["main"]["R-CRAWL_rows"]["yaw-rate MAE"] is True
    # Read A on adoption reads A - P10 (main) and A_rep - P10 (beside), each at both seeds
    over = {**clear_all(), **{("A", "P10", s, l): r for s in SEEDS for l, r in readA_rows().items()}}
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(over)), name="ra")
    assert out["OUTCOME"]["OUTCOME"] == "FIXED / ADOPTED" and out["next_launch"] is None
    assert out["read_a_on_adoption"]["A"]["main"]["outcome"] == "DOMINANT" and out["read_a_on_adoption"]["A"]["rep_beside"]["outcome"] == "NO MARGIN"
    assert set(out["read_a"]) >= {"REP_CTRL", "CTRL_P10", "REP_P10", "A_P10", "A_CTRL", "Arep_P10", "Arep_REP"} and set(out["read_a"]["A_P10"]) == {"s0", "s1"}


@needs_prereg
def test_e2e_unplaced_is_rc_3_and_invents_no_word(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "outcome_rows", lambda *a, **k: {n: True for n in R.OUTCOME_NAMES})
    rc, out = run_read(tmp_path, monkeypatch, world(scen_from(clear_all())), name="unplaced")
    assert rc == 3 and out["OUTCOME"]["OUTCOME"] == "UNPLACED" and out["OUTCOME"]["committed_next"] is None and out["cascade"]["final_arm"] == "A" and out["next_launch"] is None
    assert out["read_a_on_adoption"] == "not applicable: no arm adopts"


@needs_prereg
@pytest.mark.parametrize("pair_,ids", [(("REP", "REP"), ("K-ZERO:REP-REP", "K-ZERO:RF-of-REP-as-CTRL")), (("CTRL", "CTRL"), ("K-ZERO:CTRL-CTRL:s0",)), (("Arep", "Arep"), ("K-ZERO:Arep-Arep:s0",))])
def test_e2e_every_identity_contrast_must_read_exactly_zero(tmp_path, monkeypatch, pair_, ids):
    class Dirty(FakeHeavy):
        def contrast(self, x, y):
            t = super().contrast(x, y)
            if (x, y) == pair_:
                t[0]["ADE6"] = {"delta": 1e-9, "lo": 0.0, "hi": 0.0, "separated": False}
            return t
    FakeHeavy.world = world(scen_from(clear_all()))
    monkeypatch.setattr(R, "Heavy", Dirty)
    man = make_world_files(tmp_path / "z")
    (tmp_path / "z.json").write_text(json.dumps(man), encoding="utf-8")
    rc = R.main(["--manifest", str(tmp_path / "z.json"), "--prereg", PREREG, "--out", str(tmp_path / "z_read.json")])
    out = json.loads((tmp_path / "z_read.json").read_text(encoding="utf-8"))
    assert rc == 2 and out["OUTCOME"] == "REFUSED" and all(i in out["failed"] for i in ids)


@needs_prereg
def test_read_core_and_compose_all_carry_the_control_state(tmp_path):
    FakeHeavy.world = world(scen_from(clear_all()))
    man = make_world_files(tmp_path / "ca")
    ctl = R.Ctl()
    res = R.read_core(FakeHeavy(None, None, None), ctl, man, RF0)
    assert not ctl.failed() and res["xs"] == ["A"] and res["floors"]["bar"] == BARS and set(res["g"]) == {("A", "main"), ("A", "rep")} and set(res["slow"]) == {("A", "main"), ("A", "rep")}
    assert R.compose_all(res, True)["A"]["decision"]["OUTCOME"] == "FIXED / ADOPTED" and R.compose_all(res, False)["A"]["decision"]["OUTCOME"] == "REFUSED"
    assert R.compose_all({**res, "floors": {**res["floors"], "bar": {l: 1e9 for l in R.LABELS}}}, True)["A"]["decision"]["OUTCOME"] == "NO EFFECT"          # the bars in the result are the bars used
    for t in ("REP_CTRL", "CTRL_P10", "REP_P10", "A_P10", "A_CTRL", "Arep_P10", "Arep_REP"):
        assert t in res["tabs"]


def test_read_core_reads_b_only_when_both_b_roles_exist_at_both_seeds(tmp_path):
    FakeHeavy.world = world(scen_from(clear_all()))
    man = make_world_files(tmp_path / "bb", with_b=True)
    res = R.read_core(FakeHeavy(None, None, None), R.Ctl(), man, RF0)
    assert res["xs"] == ["A", "B"] and ("B", "main") in res["g"] and ("B", "rep") in res["g"]
    for r_ in ("B", "Brep"):
        del man["arms"][r_]["1"]
    assert R.read_core(FakeHeavy(None, None, None), R.Ctl(), man, RF0)["xs"] == ["A"]
    man = make_world_files(tmp_path / "bc", with_b=True)
    del man["arms"]["Brep"]["0"]
    assert R.read_core(FakeHeavy(None, None, None), R.Ctl(), man, RF0)["xs"] == ["A"]


def test_trim_keeps_the_reported_keys_and_drops_hidden_rows():
    full = {"delta": 0.1, "lo": 0.0, "hi": 0.2, "separated": False, "n": "2399/141", "mean_arm": 1.0, "mean_ref": 0.9, "family": "PRIMARY", "extra_key": 5}
    t = R.trim({"ADE6": full, "_internal": full, "speed MAE": {"status": "NOT COMPUTED"}})
    assert set(t) == {"ADE6", "speed MAE"} and "extra_key" not in t["ADE6"] and t["ADE6"]["delta"] == 0.1 and t["ADE6"]["family"] == "PRIMARY" and t["speed MAE"] == {"status": "NOT COMPUTED"}


def test_rollout_files_are_the_new_arms_only(tmp_path):
    man = make_world_files(tmp_path / "rf")
    fs = [f for f in R.rollout_files(man) if f]                                                          # absent roles contribute empty names, which K-P0 skips
    assert len(fs) == 2 * 2 * 2 * (3 + 2) and all(("A_s" in f or "Arep_s" in f) for f in fs) and not any(("CTRL_s" in f or ("REP_s" in f and "Arep" not in f)) for f in fs)
    man = make_world_files(tmp_path / "rf2", with_b=True)
    assert len([f for f in R.rollout_files(man) if f]) == 4 * 2 * 2 * (3 + 2)


def test_provenance_and_constants_in_the_result(tmp_path):
    a = types.SimpleNamespace(orc_dir="x")
    pv = R.provenance(a, 0.0, None)
    assert pv["claim_row"] == "H-V8F-LAUNCH-INTENT-1" and pv["rf0_sha256"] == R.RF0_SHA256 and pv["manifest_sha256"] is None and pv["reader_sha256"] == R.sha256_file(R.__file__)
    assert set(R.NOT_BUILT) == {"M-LAUNCH", "anti_echo_block", "M2_and_L2_launch_subset", "t0_four_families"}
