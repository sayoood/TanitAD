"""Literal tests for announced() (SPEC_ADDENDUM_A6). Every expectation is a LITERAL read from the builder's documented rule
(`is_turn`: |dyaw| >= 15 AND R <= 140 m AND v_min <= 8.0 m/s; PI 2026-08-29 suppression) or from a named label record --
never an expression over the code under test. Mutation arms are included.

Run (WINDOWS path for PYTHONPATH):  PYTHONPATH="D:/Projects/TanitAD/stack" python test_announced_a6.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import announced_a6 as AN  # noqa: E402

EVAL = Path("D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
_RECS = None


def eval_by_sha():
    global _RECS
    if _RECS is None:
        _RECS = {AN.sha12(r["clip_id"]): r for r in AN.load(EVAL)}
    return _RECS


# ------------------------------------------------------------------------------------------- the builder's rule
def test_is_turn_boundaries_are_the_builders_literals():
    b = AN.builder()
    assert (b.TURN_MIN_DYAW_DEG, b.TURN_MAX_ARC_R_M, b.TURN_MAX_VMIN_MS) == (15.0, 140.0, 8.0)
    assert b.is_turn((0.0, 3.0, 15.0, 140.0, 8.0))            # all three at their limit -> a turn
    assert b.is_turn((0.0, 3.0, -15.0, 140.0, 8.0))           # sign does not matter
    assert not b.is_turn((0.0, 3.0, 14.9, 140.0, 8.0))        # too little yaw
    assert not b.is_turn((0.0, 3.0, 40.0, 140.1, 8.0))        # too wide: a road curve
    assert not b.is_turn((0.0, 3.0, 40.0, 30.0, 8.01))        # taken too fast: a road curve


def test_command_token_turn_curve_suppressed_and_nothing_ahead():
    assert AN.command_token((3.8, 8.7, -90.1, 19.6, 5.74), False) == "NAV_TURN_R"
    assert AN.command_token((1.0, 4.0, 40.0, 20.0, 5.0), False) == "NAV_TURN_L"
    assert AN.command_token((20.6, 25.1, -60.6, 36.7, 8.68), False) == "NAV_FOLLOW_ROAD"     # the eval 'curve, not a turn' segment
    assert AN.command_token((3.8, 8.7, -90.1, 19.6, 5.74), True) == "NAV_FOLLOW_ROAD"        # PI 2026-08-29: contested
    assert AN.command_token(None, False) == "NAV_FOLLOW_ROAD"


# ------------------------------------------------------------------------------------------- announced(), synthetic
def _rec(seq, entries, supp=None):
    return {"manoeuvre_sequence": seq, "nav_30s": {"entries": entries}, "turn_suppression": supp,
            "nav_command": {"token": "NAV_FOLLOW_ROAD"}}


def _m(t0, t1, dyaw, R, vmin):
    return {"t_start_s": t0, "t_end_s": t1, "dyaw_deg": dyaw, "radius_m": R, "v_min_ms": vmin, "is_turn": True}


def _e(tok, t0, t1, dyaw, R):
    return {"token": tok, "t_start_s": t0, "t_end_s": t1, "dyaw_deg": dyaw, "radius_m": R}


def test_announced_synthetic_known_values():
    seq = [_m(3.0, 6.0, 40.0, 20.0, 5.0)]
    ent = [_e("NAV_TURN_L", 3.0, 6.0, 40.0, 20.0)]
    assert AN.announced(ent[0], _rec(seq, ent)) is True
    # the PI 2026-08-29 suppression flips it (mutation of the record, same entry)
    assert AN.announced(ent[0], _rec(seq, ent, {"applied": True, "side": "left", "t_start_s": 3.0})) is False
    # the key the nav_30s builder READ ('suppressed') is not the key the emitter WRITES ('applied'): it must not count
    assert AN.suppression_applies(_rec(seq, ent, {"suppressed": True})) is False
    assert AN.suppression_applies(_rec(seq, ent, {"applied": True})) is True
    # a FOLLOW entry announces nothing
    fol = [_e("NAV_FOLLOW_ROAD", 0.0, 30.0, None, None)]
    assert AN.announced(fol[0], _rec([], fol)) is False
    # an entry with no matching manoeuvre is an error, never a guess
    try:
        AN.announced(_e("NAV_TURN_L", 9.0, 12.0, 40.0, 20.0), _rec(seq, ent))
        raise AssertionError("expected LookupError")
    except LookupError:
        pass


def test_a_turn_behind_a_curve_is_still_announced_because_the_entry_is_a_real_turn():
    """nav_command looks only at seq[0] (the curve) and says FOLLOW; announced() asks the builder about the ENTRY."""
    seq = [dict(_m(7.2, 9.0, -21.6, 124.7, 12.42), is_turn=False), _m(15.3, 21.5, -69.3, 37.5, 4.21)]
    ent = [_e("NAV_TURN_R", 15.3, 21.5, -69.3, 37.5)]
    r = _rec(seq, ent)
    assert AN.announced(ent[0], r) is True
    assert AN.command_token((7.2, 9.0, -21.6, 124.7, 12.42), False) == "NAV_FOLLOW_ROAD"     # what the shipped token reflects


# ------------------------------------------------------------------------------------------- announced(), real records
def test_the_two_eval_records_with_turn_suppression_announce_nothing():
    R = eval_by_sha()
    one, two = R["4b5c891e90e6"], R["bbd162dfe0eb"]
    for r in (one, two):
        assert AN.suppression_applies(r) is True
        assert r["nav_command"]["token"] == "NAV_FOLLOW_ROAD" and "contested" in r["nav_command"]["reason"]
    assert [e["token"] for e in one["nav_30s"]["entries"]] == ["NAV_TURN_R"]
    assert AN.announced_flags(one) == [False]
    assert [e["token"] for e in two["nav_30s"]["entries"]] == ["NAV_TURN_R", "NAV_TURN_R"]
    assert AN.announced_flags(two) == [False, False]             # clip-level suppression, as the builder applies it


def test_a_curve_not_a_turn_record():
    r = eval_by_sha()["a2bd72d90385"]
    assert r["nav_command"]["token"] == "NAV_FOLLOW_ROAD" and "curve, not a turn" in r["nav_command"]["reason"]
    s = r["manoeuvre_sequence"][0]
    assert (s["t_start_s"], s["dyaw_deg"], s["radius_m"], s["v_min_ms"], s["is_turn"]) == (20.6, -60.6, 36.7, 8.68, False)
    assert AN.builder().is_turn((20.6, 25.1, -60.6, 36.7, 8.68)) is False           # v_min 8.68 > 8.0
    assert AN.announced_flags(r) == [False]                                          # nav_30s holds only the FOLLOW entry
    assert AN.shipped_side(r) == 0


def test_curve_first_eval_record_the_later_turn_is_announced_while_the_shipped_token_is_follow():
    r = eval_by_sha()["c167e6ce11a3"]
    assert r["nav_command"]["token"] == "NAV_FOLLOW_ROAD" and "curve, not a turn" in r["nav_command"]["reason"]
    assert AN.announced_flags(r) == [True]
    assert AN.shipped_side(r) == 0                       # => a literal G1 on entries[0] cannot pass for this record


def test_normal_turn_record_announced_equals_the_shipped_side_and_suppression_mutation_flips_it():
    r = eval_by_sha()["0dfe63214bc2"]
    assert r["nav_command"]["token"] == "NAV_TURN_R" and r["nav_command"]["args"]["time_s"] == 2.6
    assert AN.announced_flags(r) == [True] and AN.shipped_side(r) == -1
    m = copy.deepcopy(r)
    m["turn_suppression"] = {"applied": True}
    assert AN.announced_flags(m) == [False]


def test_turn_beyond_the_30s_cap_has_no_entry_to_announce():
    r = eval_by_sha()["a699b49d4563"]
    assert r["nav_command"]["token"] == "NAV_TURN_R" and r["nav_command"]["args"]["time_s"] == 32.9
    assert [e["token"] for e in r["nav_30s"]["entries"]] == ["NAV_FOLLOW_ROAD"]
    assert AN.announced_flags(r) == [False]


# ------------------------------------------------------------------------------------------- the G1 finding, pinned
def test_G1_finding_is_pinned_if_the_json_is_present():
    p = HERE.parent / "raw" / "a6_g1.json"
    if not p.exists():
        return
    g = json.loads(p.read_text(encoding="utf-8"))
    assert g["G1_literal_entries0"]["n_match"] == 4578 and g["n_records"]["all"] == 4719
    assert g["G1_literal_entries0"]["PASS_all_records_ge_99pct"] is False               # 97.01 % < 99 %
    assert g["G1_suppression_named_records"]["n"] == 68 and g["G1_suppression_named_records"]["n_match"] == 68
    assert g["mismatch_decomposition"]["n_mismatch"] == 141 == sum(g["mismatch_decomposition"]["by_kind"].values())
    assert not any(k.startswith("OTHER") for k in g["mismatch_decomposition"]["by_kind"])
    d = g["diagnostics_NOT_the_registered_control"]
    assert d["builder_rule_on_seq0_reproduces_shipped_nav_command"]["n_match"] == 4719
    assert d["builder_is_turn_on_every_manoeuvre_sequence_element_equals_the_stored_is_turn_flag_mismatches"] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)}/{len(fns)} passed")
