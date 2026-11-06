"""r9_e1_verdict.py: the per-arm E1 verdict.  Known-answer tests, a mapping-completeness test and a DELIBERATE-REGRESSION arm.

KNOWN ANSWERS (literal rows, through ``judge`` -- the very function the CLI calls):
  * floor dx_0_40_mean 1.0 -> 1.25 (floor 0.25, 2 x floor 0.5, lower is better): a lever delta of -0.75 separated KEEPS; -0.5 (exactly 2 x floor)
    DROPS on clause 2; -0.75 not separated DROPS on clause 1; +0.75 separated DROPS on clause 1;
  * the SAME winning lever with ap2m_all separated worse by -0.625 (floor 0.25, 2 x floor 0.5) DROPS on clause 3 naming ap2m_all; at -0.375
    (worse, but inside 2 x floor) or -0.625 NOT separated it KEEPS;
  * a floor of exactly 0 is DEGENERATE: any separated good delta clears clause 2; a separated worse row vetoes at any size;
  * a NOT-COMPUTABLE primary is NOT-COMPUTABLE -- unless clause 3 already fails, then DROP.
END TO END (synthetic dumps through the REAL r9_e1_eval.py -> r9_e1_floor.py -> r9_e1_verdict.py): a lever that wins its primary but adds false
sidewalk flags is DROPPED by clause 3 on a MAP row; a clean winner KEEPS; mismatched windows / gate / arm order / a REFUSED floor REFUSE.
DELIBERATE REGRESSION: the verdict source with its clause-3 call replaced by "no vetoes" must make the clause-3 case go RED (mutate, do not inspect).
MAPWIRE (hold_r9_e1_mapprim): the M1 map primary is ``nf_swedge_prec_lv_5_20``.  KNOWN ANSWERS: M1 KEEPS on a lift-masked rig (rasters from the mapprim
test's builder: lv 5-20 m precision 60/170 for C0, 60/150 for C1, 60/80 for the winner); NOT-COMPUTABLE on a mask-less floor (NO-FLOOR); a box lever is vetoed
by the swedge row where the sidewalk-only proxies cannot see it; a map arm is NOT vetoed by the proxies of its OWN primary (the same rasters judged as
another family are); a --lift-valid whose sha256 differs from the floor's (or one side without a mask) is REFUSED.  CLAUSE-3 COMPLETENESS (MM addendum): a
scope key the floor has no row for is COVERED by a primary's named proxies or makes the verdict NOT-COMPUTABLE ("floor has no row for <key>"); a DROP stands.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_STACK = Path(__file__).resolve().parents[1]
_OPS = _STACK / "ops" / "r9e1_read"
for _p in (_STACK / "scripts", _OPS):
    sys.path.insert(0, str(_p))
import r9_e1_eval as E                                   # noqa: E402
import r9_e1_floor as F                                  # noqa: E402
import r9_e1_verdict as V                                # noqa: E402
import r9e1_read_stub_dump as STUB                       # noqa: E402

BOOT = "50"
CLIPS = ["%012x" % (0xa00 + i) for i in range(6)]
DESIGN_ROWS = {"C0", "C1", "M1", "M2a", "M2b", "M3", "B1", "B2", "B2c", "B3", "B4", "A1", "F1", "T1", "M1-FROZEN"}      # the rows of the section 4.3 table


# ====================================================================================================================== literal rows
def frow(metric, c0, c1, hib=True, family="BOXES"):
    d = c1 - c0
    return {"metric": metric, "family": family, "group": "g", "higher_is_better": hib, "undefined": False, "c0": c0, "c1": c1, "delta": d,
            "ci_lo": d - 0.01, "ci_hi": d + 0.01, "separated": True, "floor": abs(d), "keep_threshold": 2 * abs(d), "floor_degenerate": d == 0.0,
            "ci_degenerate": False}


def base_floor(lv=False, omit=(), **over):
    """A literal floor carrying a row for EVERY clause-3 scope key but the lift-valid one (a floor read WITHOUT a mask).  ``lv=True`` adds the mask rows
    (the primary, plus two rows outside clause 3's scope); ``omit`` drops rows (a floor read by an older evaluator)."""
    rows = [frow("dx_0_40_mean", 1.0, 1.25, hib=False), frow("dx_0_40_median", 1.0, 1.25, hib=False), frow("ap2m_all", 0.5, 0.75),
            frow("ap1m_all", 0.5, 0.75), frow("ap2m_0_20", 0.5, 0.75), frow("ap2m_20_40", 0.5, 0.75), frow("ap2m_40_60", 0.5, 0.75),
            frow("ap2m_20_60", 0.5, 0.75), frow("tp_class_acc", 0.5, 0.75), frow("boxes_per_object_noNMS", 1.0, 1.25, hib=False),
            frow("nf_prec_5_10", 0.5, 0.75, family="MAP"), frow("nf_prec_10_20", 0.5, 0.75, family="MAP"), frow("nf_prec_0_10", 0.5, 0.75, family="MAP"),
            dict(frow("n_pos", 100.0, 100.0), higher_is_better=None)]
    if lv:
        rows += [frow("nf_swedge_prec_lv_5_20", 0.5, 0.75, family="MAP"), frow("nf_swedge_prec_lv_10_20", 0.5, 0.75, family="MAP"),
                 frow("nf_swedge_prec_5_20", 0.5, 0.75, family="MAP")]
    rows = [over.get(r["metric"], r) for r in rows if r["metric"] not in omit]
    return {"schema": "r9e1-floor-v1", "status": "OK", "rows": {r["metric"]: r for r in rows}}


def neutral(floor):
    return {m: {"ref": r["c0"], "arm": r["c0"], "delta": 0.0, "ci_lo": -0.01, "ci_hi": 0.01, "separated": False, "undefined": False}
            for m, r in floor["rows"].items() if r["higher_is_better"] is not None}


def with_delta(pair, metric, delta, separated=True):
    p = dict(pair)
    base = p[metric]["ref"]
    p[metric] = {"ref": base, "arm": base + delta, "delta": delta, "ci_lo": delta - 0.01, "ci_hi": delta + 0.01, "separated": separated, "undefined": False}
    return p


def clauses(v):
    return sorted(f["clause"] for f in v["failing_clauses"])


# ------------------------------------------------------------------------------------------------------------------ the literal keep-rule cases
def case_other_primary_worse_beyond_2x_drops(judge):
    """THE clause-3 case: B1 wins its primary (dx -0.75, beyond 2 x 0.25) but ap2m_all is separated worse by -0.625 (> 2 x 0.25)."""
    fl = base_floor()
    good = with_delta(neutral(fl), "dx_0_40_mean", -0.75)
    assert judge("B1", "C0", fl, good)["verdict"] == "KEEP"                                    # the control: without the other row, it keeps
    v = judge("B1", "C0", fl, with_delta(good, "ap2m_all", -0.625))
    assert v["verdict"] == "DROP", v["verdict"]
    assert clauses(v) == [3] and [x["metric"] for x in v["clause3"]["vetoes"]] == ["ap2m_all"]
    assert "ap2m_all" in v["failing_clauses"][0]["text"]


def test_keep_and_the_three_clauses_literal():
    fl = base_floor()
    n = neutral(fl)
    keep = V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.75))
    assert keep["verdict"] == "KEEP" and keep["failing_clauses"] == [] and keep["not_computable"] == []
    r = keep["reads"][0]
    assert r["key"] == "dx_0_40_mean" and r["state"] == "READ" and r["floor"] == 0.25 and r["keep_threshold"] == 0.5
    assert r["lever_read"]["keep_on_this_metric"] is True and r["lever_read"]["separated_good_direction"] is True
    at = V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.5))                           # exactly 2 x floor: NOT strictly greater
    assert at["verdict"] == "DROP" and clauses(at) == [2]
    just = V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.5000001))
    assert just["verdict"] == "KEEP"
    below = V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.4))
    assert below["verdict"] == "DROP" and clauses(below) == [2]
    unsep = V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.75, separated=False))
    assert unsep["verdict"] == "DROP" and clauses(unsep) == [1]
    wrong = V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", +0.75))                       # lower is better: +0.75 is worse
    assert wrong["verdict"] == "DROP" and 1 in clauses(wrong)
    assert V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.1, separated=False))["verdict"] == "DROP"


def test_clause3_literal_cases():
    case_other_primary_worse_beyond_2x_drops(V.judge)
    fl = base_floor()
    good = with_delta(neutral(fl), "dx_0_40_mean", -0.75)
    for delta, sep, want in [(-0.375, True, "KEEP"),            # worse, separated, but inside 2 x floor (0.5)
                             (-0.625, False, "KEEP"),           # beyond 2 x floor but NOT separated
                             (+0.625, True, "KEEP"),            # separated BETTER is never a veto
                             (-0.5, True, "KEEP"),              # exactly 2 x floor: not beyond
                             (-0.6, True, "DROP")]:
        assert V.judge("B1", "C0", fl, with_delta(good, "ap2m_all", delta, sep))["verdict"] == want, (delta, sep)
    # a lower-is-better row: boxes_per_object_noNMS UP by 0.625 (> 2 x 0.25) separated is worse
    v = V.judge("B1", "C0", fl, with_delta(good, "boxes_per_object_noNMS", +0.625))
    assert v["verdict"] == "DROP" and v["clause3"]["vetoes"][0]["metric"] == "boxes_per_object_noNMS"
    # a MAP proxy row (the map primary's named proxy) also vetoes ...
    v = V.judge("B1", "C0", fl, with_delta(good, "nf_prec_5_10", -0.625))
    assert v["verdict"] == "DROP" and v["clause3"]["vetoes"][0]["family"] == "MAP"
    # ... but a row no arm names (tp_class_acc) is NOT a perception primary: reported as informative, never a veto (sec. 4.3 wording)
    v = V.judge("B1", "C0", fl, with_delta(good, "tp_class_acc", -0.625))
    assert v["verdict"] == "KEEP" and v["clause3"]["vetoes"] == []
    assert [x["metric"] for x in v["clause3"]["informative_worse"]] == ["tp_class_acc"]
    assert V.judge("B1", "C0", fl, with_delta(good, "ap2m_all", -0.625))["clause3"]["vetoes"][0]["in_table_primaries"] is True
    # n_pos (a count: no floor direction) never vetoes and is listed as having no floor
    assert "n_pos" in V.judge("B1", "C0", fl, good)["clause3"]["no_floor"]
    # two vetoes are both named
    v = V.judge("B1", "C0", fl, with_delta(with_delta(good, "ap2m_all", -0.625), "ap1m_all", -0.625))
    assert sorted(x["metric"] for x in v["clause3"]["vetoes"]) == ["ap1m_all", "ap2m_all"]


def test_degenerate_floor_literal():
    fl = base_floor(dx_0_40_mean=frow("dx_0_40_mean", 1.0, 1.0, hib=False), ap2m_all=frow("ap2m_all", 0.5, 0.5))     # floors of exactly 0
    n = neutral(fl)
    k = V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.001))
    assert k["verdict"] == "KEEP" and k["reads"][0]["lever_read"]["floor_degenerate"] is True               # separation alone decides
    assert V.judge("B1", "C0", fl, with_delta(n, "dx_0_40_mean", -0.001, separated=False))["verdict"] == "DROP"
    v = V.judge("B1", "C0", fl, with_delta(with_delta(n, "dx_0_40_mean", -0.001), "ap2m_all", -0.0001))     # any separated worse row vetoes at floor 0
    assert v["verdict"] == "DROP" and v["clause3"]["vetoes"][0]["floor_degenerate"] is True


def test_not_computable_primary_and_unread_rows_literal():
    fl = base_floor()
    n = neutral(fl)
    m1 = V.judge("M1", "C0", fl, n)
    assert m1["verdict"] == "NOT-COMPUTABLE" and m1["failing_clauses"] == [] and "sidewalk+edge" in m1["not_computable"][0]
    assert V.judge("M1", "C0", fl, with_delta(n, "ap2m_all", -0.625))["verdict"] == "DROP"        # clause 3 fails: no primary can rescue it
    a1 = V.judge("A1", "C0", fl, with_delta(n, "ap2m_all", +0.625))                                # primary clears, the NAMED map guard is NC
    assert a1["reads"][0]["lever_read"]["keep_on_this_metric"] is True and a1["verdict"] == "NOT-COMPUTABLE"
    assert "named guard" in a1["not_computable"][0]
    unread = {k: v for k, v in with_delta(n, "dx_0_40_mean", -0.75).items() if not k.startswith("nf_prec")}
    v = V.judge("B1", "C0", fl, unread)                                                              # the MAP rows were not read: never a silent KEEP
    assert v["verdict"] == "NOT-COMPUTABLE" and v["clause3"]["unread"] == ["nf_prec_0_10", "nf_prec_10_20", "nf_prec_5_10"]
    undef = dict(with_delta(n, "dx_0_40_mean", -0.75))
    undef["dx_0_40_mean"] = dict(undef["dx_0_40_mean"], undefined=True)
    assert V.judge("B1", "C0", fl, undef)["verdict"] == "NOT-COMPUTABLE"


def test_alternative_conjunction_reading_and_named_guards_are_reported():
    fl = base_floor()
    v = V.judge("B1", "C0", fl, with_delta(neutral(fl), "dx_0_40_mean", -0.75))                     # dx keeps; AP@1 m unchanged
    assert v["verdict"] == "KEEP" and v["alternative_reading"]["all_listed_primaries_kept"] is False
    both = V.judge("B1", "C0", fl, with_delta(with_delta(neutral(fl), "dx_0_40_mean", -0.75), "ap1m_all", +0.75))
    assert both["alternative_reading"]["all_listed_primaries_kept"] is True
    m2 = V.judge("M2a", "M1", fl, with_delta(neutral(fl), "ap2m_all", -0.4))                         # worse, separated, inside 2 x floor: clause 3 passes ...
    assert m2["named_guards"][0]["separated_worse_any_size"] is True                                  # ... but the stricter 'not worse at all' reading shows it


def test_control_outcomes_literal():
    worse = {"lever_read": {"separated_bad_direction": True, "exceeds_2x_floor": True, "keep_on_this_metric": False}}
    same = {"lever_read": {"separated_bad_direction": False, "exceeds_2x_floor": False, "keep_on_this_metric": False}}
    gain = {"lever_read": {"separated_bad_direction": False, "exceeds_2x_floor": True, "keep_on_this_metric": True}}
    assert V.control_outcome("T1s", "T1", worse).startswith("AS-REQUIRED")
    assert V.control_outcome("T1s", "T1", same).startswith("NOT-WORSE")
    assert V.control_outcome("B2c", "B2", worse).startswith("GAINS-LESS-THAN-B2")
    assert V.control_outcome("B2c", "B2", same).startswith("GAINS-AS-MUCH-AS-B2")
    assert V.control_outcome("B2c", "C0", gain).startswith("GAINS") and V.control_outcome("T1s", "C0", same).startswith("NO-GAIN")
    inside = {"lever_read": {"separated_bad_direction": True, "exceeds_2x_floor": False, "keep_on_this_metric": False}}
    assert V.control_outcome("T1hs", "T1h", worse).startswith("AS-REQUIRED")                 # the addendum's bar: worse beyond 2 x floor ...
    assert V.control_outcome("T1hs", "T1h", inside).startswith("NOT-WORSE-BEYOND-FLOOR")      # ... separated but inside it does NOT attribute
    assert V.control_outcome("T1hs", "T1h", same).startswith("NOT-WORSE:")
    assert V.control_outcome("T1hs", "C0", same).startswith("NO-GAIN") and V.control_outcome("T1s", "T1", inside).startswith("NOT-WORSE")
    fl = base_floor(omit=("nf_prec_0_10",))
    c = V.judge("T1s", "T1", fl, neutral(fl))                                   # a floor read before the pooled key existed has no nf_prec_0_10 row
    assert c["verdict"] == "NOT-COMPUTABLE" and c["control_outcome"]


# ------------------------------------------------------------------------------------------------------------------ MAPWIRE: the map primary, literal
LV = "nf_swedge_prec_lv_5_20"


def test_map_primary_keep_rule_literal_on_a_masked_floor():
    fl = base_floor(lv=True)                                  # lv row 0.5 -> 0.75: floor 0.25, 2 x floor 0.5, higher is better
    n = neutral(fl)
    k = V.judge("M1", "C0", fl, with_delta(n, LV, +0.6))
    assert k["verdict"] == "KEEP" and k["failing_clauses"] == [] and k["not_computable"] == [], (k["verdict"], k["not_computable"])
    r = k["reads"][0]
    assert r["key"] == LV and r["state"] == "READ" and r["floor"] == 0.25 and r["keep_threshold"] == 0.5 and r["higher_is_better"] is True
    at = V.judge("M1", "C0", fl, with_delta(n, LV, +0.5))                                              # exactly 2 x floor: not beyond
    assert at["verdict"] == "DROP" and clauses(at) == [2]
    assert clauses(V.judge("M1", "C0", fl, with_delta(n, LV, +0.6, separated=False))) == [1]           # not separated
    assert 1 in clauses(V.judge("M1", "C0", fl, with_delta(n, LV, -0.6)))                              # higher is better: a fall is not a good direction
    for arm in ("M1", "M2a", "M2b", "M3", "F1", "M1-FROZEN"):                                           # every arm reading the map primary reads it on that row
        assert [m["keys"] for m in V.ARMS[arm]["metrics"] if m["role"] == "primary"] == [(LV,)], arm
    a1 = V.judge("A1", "C0", fl, with_delta(n, "ap2m_all", +0.6))                                       # A1's named guard is the map primary: READ and neutral
    assert a1["verdict"] == "KEEP" and [g["state"] for g in a1["reads"] if g["role"] == "guard"] == ["READ"]


def test_map_primary_is_not_computable_on_a_maskless_floor_and_a_drop_still_stands():
    fl = base_floor()                                         # a floor read WITHOUT --lift-valid has no nf_swedge_prec_lv_5_20 row
    assert LV not in fl["rows"]
    m1 = V.judge("M1", "C0", fl, neutral(fl))
    assert m1["verdict"] == "NOT-COMPUTABLE" and m1["failing_clauses"] == []
    assert m1["reads"][0]["state"] == "NO-FLOOR" and "--lift-valid" in m1["reads"][0]["why"] and "re-read the floor with the current eval" in m1["reads"][0]["why"]
    assert m1["clause3"]["missing_from_floor"] == []          # the missing key IS M1's own primary: excluded from the clause-3 completeness check
    assert V.judge("M2a", "M1", fl, neutral(fl))["verdict"] == "NOT-COMPUTABLE"
    assert V.judge("F1", "C0", fl, neutral(fl))["verdict"] == "NOT-COMPUTABLE"
    # a DROP from clause 3 stands even though the primary cannot be read
    assert V.judge("M1", "C0", fl, with_delta(neutral(fl), "ap2m_all", -0.625))["verdict"] == "DROP"


def test_a_box_lever_is_vetoed_by_the_swedge_row_but_not_by_the_informative_bands():
    fl = base_floor(lv=True)
    good = with_delta(neutral(fl), "dx_0_40_mean", -0.75)
    assert V.judge("B1", "C0", fl, good)["verdict"] == "KEEP"                                           # the control: nothing worse -> KEEP
    v = V.judge("B1", "C0", fl, with_delta(good, LV, -0.625))                                           # the lv primary separated worse beyond 2 x 0.25
    assert v["verdict"] == "DROP" and clauses(v) == [3] and [x["metric"] for x in v["clause3"]["vetoes"]] == [LV]
    assert v["clause3"]["vetoes"][0]["in_table_primaries"] is True and v["clause3"]["vetoes"][0]["family"] == "MAP"
    assert V.judge("B1", "C0", fl, with_delta(good, LV, -0.5))["verdict"] == "KEEP"                     # exactly 2 x floor: not beyond
    # the per-band lv row and the non-lv pooled row are OUT of scope: reported as informative, never a veto
    for k in ("nf_swedge_prec_lv_10_20", "nf_swedge_prec_5_20"):
        w = V.judge("B1", "C0", fl, with_delta(good, k, -0.625))
        assert w["verdict"] == "KEEP" and w["clause3"]["vetoes"] == [] and [x["metric"] for x in w["clause3"]["informative_worse"]] == [k], k


def test_a_map_arm_is_not_vetoed_by_the_proxy_of_its_own_primary():
    fl = base_floor(lv=True)
    win = with_delta(neutral(fl), LV, +0.6)
    worse_proxy = with_delta(with_delta(win, "nf_prec_5_10", -0.625), "nf_prec_10_20", -0.625)          # both sidewalk-only proxies separated worse beyond 2 x floor
    for arm, ref in (("M1", "C0"), ("M2a", "M1"), ("F1", "C0")):
        v = V.judge(arm, ref, fl, worse_proxy)
        assert v["verdict"] == "KEEP" and v["clause3"]["vetoes"] == [], (arm, v["verdict"], v["clause3"]["vetoes"])
        assert {"nf_prec_5_10", "nf_prec_10_20", LV} <= set(v["clause3"]["excluded"]), arm
    # the SAME rows judged as ANOTHER family's lever (B1 with a winning dx) ARE vetoed by those proxies: the proxies keep their scope
    b = V.judge("B1", "C0", fl, with_delta(worse_proxy, "dx_0_40_mean", -0.75))
    assert b["verdict"] == "DROP" and sorted(x["metric"] for x in b["clause3"]["vetoes"]) == ["nf_prec_10_20", "nf_prec_5_10"]
    # a map arm IS still vetoed by every OTHER scope row (a box row), and its guard stays swept
    assert V.judge("M1", "C0", fl, with_delta(win, "ap2m_all", -0.625))["verdict"] == "DROP"
    v = V.judge("M2a", "M1", fl, with_delta(win, "ap2m_all", -0.625))
    assert v["verdict"] == "DROP" and [x["metric"] for x in v["clause3"]["vetoes"]] == ["ap2m_all"]


def test_clause3_completeness_covered_uncovered_and_drop_stands():
    # (1) COVERED: the mask-less floor has no lv row, but both sidewalk-only proxies have rows -> B1 still KEEPS and says why
    fl = base_floor()
    keep = V.judge("B1", "C0", fl, with_delta(neutral(fl), "dx_0_40_mean", -0.75))
    c3 = keep["clause3"]
    assert keep["verdict"] == "KEEP" and keep["not_computable"] == []
    assert c3["missing_from_floor"] == [LV] and c3["missing_uncovered"] == []
    assert c3["missing_covered"] == [{"key": LV, "covered_by_proxies": ["nf_prec_5_10", "nf_prec_10_20"]}]
    # (2) UNCOVERED, a primary WITHOUT proxies: a floor read by the pre-pooled evaluator has no ap2m_20_60 (B2's primary) -> NOT-COMPUTABLE
    old = base_floor(omit=("ap2m_20_60",))
    v = V.judge("B1", "C0", old, with_delta(neutral(old), "dx_0_40_mean", -0.75))
    assert v["verdict"] == "NOT-COMPUTABLE" and v["failing_clauses"] == []
    assert v["clause3"]["missing_uncovered"] == ["ap2m_20_60"] and v["clause3"]["missing_covered"][0]["key"] == LV
    assert "floor has no row for ap2m_20_60: re-read the floor with the current eval" in " ".join(v["not_computable"])
    older = base_floor(omit=("ap2m_20_60", "nf_prec_0_10"))                                              # both pooled keys absent: both named
    w = V.judge("B1", "C0", older, with_delta(neutral(older), "dx_0_40_mean", -0.75))
    assert w["clause3"]["missing_uncovered"] == ["ap2m_20_60", "nf_prec_0_10"] and w["verdict"] == "NOT-COMPUTABLE"
    # (3) a proxy that is itself missing leaves the map primary UNCOVERED (and the proxy, which is in scope, is named too)
    nop = base_floor(omit=("nf_prec_5_10",))
    x = V.judge("B1", "C0", nop, with_delta(neutral(nop), "dx_0_40_mean", -0.75))
    assert x["verdict"] == "NOT-COMPUTABLE" and x["clause3"]["missing_covered"] == []
    assert sorted(x["clause3"]["missing_uncovered"]) == sorted([LV, "nf_prec_5_10"])
    # (4) a DROP from another clause STANDS on an incomplete clause 3 ...
    unsep = V.judge("B1", "C0", old, with_delta(neutral(old), "dx_0_40_mean", -0.75, separated=False))
    assert unsep["verdict"] == "DROP" and clauses(unsep) == [1] and unsep["clause3"]["missing_uncovered"] == ["ap2m_20_60"]
    veto = V.judge("B1", "C0", old, with_delta(with_delta(neutral(old), "dx_0_40_mean", -0.75), "ap2m_all", -0.625))
    assert veto["verdict"] == "DROP" and clauses(veto) == [3] and any("ap2m_20_60" in n for n in veto["not_computable"])
    # (5) the arm's OWN primary key is never "missing" (B2's primary IS ap2m_20_60: it reads NO-FLOOR through its primary instead)
    b2 = V.judge("B2", "C0", old, neutral(old))
    assert b2["verdict"] == "NOT-COMPUTABLE" and b2["reads"][0]["state"] == "NO-FLOOR" and "ap2m_20_60" not in b2["clause3"]["missing_from_floor"]
    # (6) a CONTROL reads its own primary only (its outcome is not a keep rule): an incomplete clause 3 does not turn it NOT-COMPUTABLE
    ctl = V.judge("T1s", "T1", old, neutral(old))
    assert ctl["verdict"] == "CONTROL-READ" and ctl["clause3"]["missing_uncovered"] == ["ap2m_20_60"]


# ------------------------------------------------------------------------------------------------------------------ DELIBERATE REGRESSION
def _mutant_judge(old, new):
    """The verdict source with ``old`` replaced by ``new`` (exactly once), exec'd as a fresh module: a real source mutation."""
    src = Path(V.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, "the mutation target is not unique: %d occurrences" % src.count(old)
    ns = {"__name__": "r9_e1_verdict_mutant", "__file__": V.__file__}
    exec(compile(src.replace(old, new), "r9_e1_verdict_mutant.py", "exec"), ns)
    return ns["judge"]


CLAUSE3_CALL = "c3 = clause3_vetoes(floor, pair_rows, exclude, clause3_scope())"
NO_VETOES = ('c3 = {"vetoes": [], "unread": [], "undefined_in_pair": [], "no_floor": [], "checked": 0, "excluded": [], '
             '"missing_from_floor": [], "missing_covered": [], "missing_uncovered": []}')


def test_regression_arm_a_verdict_that_skips_clause3_goes_red():
    case_other_primary_worse_beyond_2x_drops(V.judge)                                               # GREEN on the real verdict
    case_other_primary_worse_beyond_2x_drops(_mutant_judge(CLAUSE3_CALL, CLAUSE3_CALL))              # GREEN on an identity mutation (the harness works)
    with pytest.raises(AssertionError):                                                              # RED when the clause is skipped
        case_other_primary_worse_beyond_2x_drops(_mutant_judge(CLAUSE3_CALL, NO_VETOES))


def case_clause1_not_separated_drops(judge):
    fl = base_floor()
    assert judge("B1", "C0", fl, with_delta(neutral(fl), "dx_0_40_mean", -0.75, separated=False))["verdict"] == "DROP"


def case_clause2_below_2x_drops(judge):
    fl = base_floor()
    assert judge("B1", "C0", fl, with_delta(neutral(fl), "dx_0_40_mean", -0.4))["verdict"] == "DROP"


def case_worse_inside_2x_does_not_veto(judge):
    fl = base_floor()
    good = with_delta(neutral(fl), "dx_0_40_mean", -0.75)
    assert judge("B1", "C0", fl, with_delta(good, "ap2m_all", -0.375))["verdict"] == "KEEP"


@pytest.mark.parametrize("case,old,new", [
    (case_clause1_not_separated_drops, 'if not lr["separated_good_direction"]:', "if False:"),                       # drop clause 1
    (case_clause2_below_2x_drops, 'if not lr["exceeds_2x_floor"]:', "if False:"),                                    # drop clause 2
    (case_worse_inside_2x_does_not_veto, 'if lr["separated_bad_direction"] and lr["exceeds_2x_floor"]:',
     'if lr["separated_bad_direction"]:'),                                                                           # clause 3 ignores the 2 x floor factor
])
def test_regression_arm_other_mutations_go_red(case, old, new):
    case(V.judge)                                                                                                    # GREEN on the real verdict
    case(_mutant_judge(old, old))                                                                                    # GREEN on an identity mutation
    with pytest.raises(AssertionError):
        case(_mutant_judge(old, new))                                                                                # RED on the mutation


# ------------------------------------------------------------------------------------------------------------------ MAPWIRE: DELIBERATE REGRESSION
def _mutant_ns(old, new):
    """Like ``_mutant_judge`` but returns the whole mutated module namespace (judge AND build_verdict)."""
    src = Path(V.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, "the mutation target is not unique: %d occurrences of %r" % (src.count(old), old)
    ns = {"__name__": "r9_e1_verdict_mutant", "__file__": V.__file__}
    exec(compile(src.replace(old, new), "r9_e1_verdict_mutant.py", "exec"), ns)
    return ns


def case_a_map_arm_is_not_vetoed_by_its_own_proxy(judge):
    fl = base_floor(lv=True)
    worse_proxy = with_delta(with_delta(with_delta(neutral(fl), LV, +0.6), "nf_prec_5_10", -0.625), "nf_prec_10_20", -0.625)
    assert judge("M1", "C0", fl, worse_proxy)["verdict"] == "KEEP"


def case_an_uncovered_missing_scope_key_blocks_keep(judge):
    old = base_floor(omit=("ap2m_20_60",))
    assert judge("B1", "C0", old, with_delta(neutral(old), "dx_0_40_mean", -0.75))["verdict"] == "NOT-COMPUTABLE"


def case_a_proxy_covered_missing_key_still_keeps(judge):
    fl = base_floor()
    assert judge("B1", "C0", fl, with_delta(neutral(fl), "dx_0_40_mean", -0.75))["verdict"] == "KEEP"


@pytest.mark.parametrize("case,old,new", [
    (case_a_map_arm_is_not_vetoed_by_its_own_proxy, 'for x in (*m["keys"], *m["proxies"])', 'for x in (*m["keys"],)'),                 # own proxies stay in the sweep
    (case_an_uncovered_missing_scope_key_blocks_keep,
     'not_computable.append("clause 3 not complete: floor has no row for %s: re-read the floor with the current eval" % k)', "pass"),  # incompleteness ignored
    (case_an_uncovered_missing_scope_key_blocks_keep, "if px and all(_usable_floor_row(floor, p) for p in px):", "if True:"),         # every missing key 'covered'
    (case_a_proxy_covered_missing_key_still_keeps, "if px and all(_usable_floor_row(floor, p) for p in px):", "if False:"),          # no missing key ever covered
])
def test_regression_arm_mapwire_mutations_go_red(case, old, new):
    case(V.judge)                                                                                                    # GREEN on the real verdict
    case(_mutant_ns(old, old)["judge"])                                                                              # GREEN on an identity mutation
    with pytest.raises(AssertionError):
        case(_mutant_ns(old, new)["judge"])                                                                          # RED on the mutation


def test_regression_arm_a_verdict_without_the_mask_sha_check_goes_red(mrig):
    old = 'the same cells)", sv == sf,'
    args, kw = _mask_mismatch_args(mrig)
    assert V.build_verdict(*args, **kw)["status"] == "REFUSED"                                                       # GREEN: the real verdict refuses
    assert _mutant_ns(old, old)["build_verdict"](*args, **kw)["status"] == "REFUSED"                                 # GREEN on an identity mutation
    assert _mutant_ns(old, 'the same cells)", True,')["build_verdict"](*args, **kw)["status"] == "OK"               # RED: with the check gone the mismatch is read


# ====================================================================================================================== the MAPPING
def _design_arm_names():
    cands = [p / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-27-perception-redesign" / "REFCV9_DESIGN.md"
             for p in _STACK.parents]
    cands.append(_STACK.parent / "REFCV9_DESIGN.md")
    for c in cands:
        if c.is_file():
            txt = c.read_text(encoding="utf-8")
            sec = txt[txt.index("### 4.3"):txt.index("### 4.4")]
            names = set()
            for ln in sec.splitlines():
                m = re.match(r"^\| ([A-Za-z0-9][A-Za-z0-9\- /]*?)( \(|\s*\|)", ln)
                if m and not ln.startswith("| arm "):
                    names |= {x.strip() for x in m.group(1).split("/") if x.strip()}
            return names
    return None


def test_mapping_has_every_design_arm():
    assert DESIGN_ROWS <= set(V.ARMS) and set(V.ARMS) - DESIGN_ROWS == {"B3p", "T1s", "T1h", "T1hs"}   # B3p = the table's "B3'" (the argv name); T1s = D3's shuffled control; T1h / T1hs = REFCV9_E1_T1H_PREREG.md
    names = _design_arm_names()
    if names is None:
        pytest.skip("REFCV9_DESIGN.md is not in this checkout")
    assert names == DESIGN_ROWS, (names ^ DESIGN_ROWS)


def test_mapping_not_computable_primaries_carry_reason_and_cure():
    nc = [(a, m) for a, s in V.ARMS.items() for m in s["metrics"] if m["status"] == "NOT-COMPUTABLE"]
    # since hold_r9_e1_mapprim the map primary is COMPUTABLE (nf_swedge_prec_lv_5_20): only M1's two 'beside' metrics (lane tolerance-IoU, boundary recall) remain
    assert {(a, m["id"]) for a, m in nc} == {("M1", "M1.beside.lane"), ("M1", "M1.beside.edge")}
    assert all(m["role"] == "beside" for _, m in nc)
    assert not {a for a, _ in nc} & {"M2a", "M2b", "M3", "A1", "F1", "M1-FROZEN", "B2", "B2c", "T1", "T1s", "T1h", "T1hs"}
    for a, m in nc:
        assert m["keys"] == () and len(m["reason"]) > 40 and len(m["would_compute"]) > 40, (a, m["id"])
        assert "r9_e1_eval" in m["reason"]
        for p in m["proxies"]:
            assert V.key_src(p) is not None, p                                                     # a proxy is an EMITTED key
    for a, s in V.ARMS.items():
        if s["kind"] == "floor":
            continue
        assert [m for m in s["metrics"] if m["role"] == "primary"] and len([m for m in s["metrics"] if m["role"] == "primary"]) == 1, a
        assert s["next_lever"] and s["variable"], a
    # the map primary: ONE key, higher is better, the sidewalk-only proxies kept; every arm that names it names the same spec
    for a in ("M1", "M2a", "M2b", "M3", "A1", "F1", "M1-FROZEN"):
        mp = [m for m in V.ARMS[a]["metrics"] if "lift-valid" in m["text"]]
        assert len(mp) == 1 and mp[0]["keys"] == ("nf_swedge_prec_lv_5_20",) and mp[0]["better"] == "higher", a
        assert mp[0]["proxies"] == ("nf_prec_5_10", "nf_prec_10_20") and mp[0]["status"] == "COMPUTABLE", a
    assert [m["role"] for m in V.ARMS["A1"]["metrics"] if "lift-valid" in m["text"]] == ["guard"]
    assert "nf_swedge_prec_lv_5_20 (r9_e1_eval.py:606)" in V.mapping_md().replace("`", "")
    assert V.own_primary_keys("M1") == {"nf_swedge_prec_lv_5_20", "nf_prec_5_10", "nf_prec_10_20"} and V.own_primary_keys("B1") == {"dx_0_40_mean"}
    assert "nf_swedge_prec_lv_5_20" in V.table_keys() and "nf_swedge_prec_lv_10_20" not in V.clause3_scope()     # the bands are informative, not in scope


def test_mapping_keys_are_emitted_by_the_evaluator_and_cited_lines_exist():
    pk = STUB.pack([7])
    item = E.window_item(pk, (1.0, 2.0), 0.5, None)
    cl, _ = E.pool_by_clip([("c", 0)], [item])
    legacy = set(E.nearfield_stats([np.ones((4, 6))]))                                             # a block measured without a mask: no swedge key
    assert not [k for k in legacy if k.startswith("nf_swedge")]
    masked = set(E.nearfield_stats([np.ones((4, E.NF_COLS))]))                                     # a mask-bearing count unit emits the ten swedge keys
    assert {k for k in masked if k.startswith("nf_swedge")} == {"nf_swedge_prec_%s%s" % (t, b) for t in ("", "lv_") for b in ("0_5", "5_10", "10_20", "20_40", "5_20")}
    emitted = set(E.box_stats(cl)) | legacy | masked
    for a, s in V.ARMS.items():
        for m in s["metrics"]:
            for k in m["keys"]:
                assert k in emitted, "%s: %s is not emitted by r9_e1_eval" % (a, k)
                assert V.key_src(k), k
                assert F.higher_is_better(k) == (m["better"] == "higher"), (a, k)
    for k in emitted:                                                                              # and every emitted key has a cited source line
        assert V.key_src(k) is not None, "no KEY_SRC for the emitted key %s" % k
    for pat, f, ln, needle in V.KEY_SRC:
        lines = (_STACK.parent / f).read_text(encoding="utf-8").splitlines()
        assert needle in lines[ln - 1], "%s:%d does not contain %s: %r" % (f, ln, needle, lines[ln - 1])
    assert not [k for k in emitted if k.startswith("nf_prec") and k.endswith("_5_20")]                      # the sidewalk-only family has no pooled 5-20 m band
    assert "nf_swedge_prec_lv_5_20" in emitted                                                              # the map primary exists (on a mask-bearing block)
    assert {"ap1m_20_60", "ap2m_20_60", "nf_prec_0_10"} <= emitted                                  # the pooled keys (hold_r9_eval_pooled)


def test_mapping_agrees_with_the_launch_argv_files():
    d = _STACK / "ops" / "runs.d"
    seen = 0
    for a, s in V.ARMS.items():
        f = d / ("r9e1_%s.argv.json" % a)
        if not f.is_file():
            continue
        seen += 1
        rec = json.loads(f.read_text(encoding="utf-8"))
        argv = rec["argv"]
        for flag in s["argv_flags"]:
            assert flag in argv, (a, flag)
        if a not in ("C0", "C1"):
            assert rec["derived_from"] == "r9e1_C0"
        base = json.loads((d / "r9e1_C0.argv.json").read_text(encoding="utf-8"))["argv"]
        if a in ("B1", "B2", "B2c", "B3", "B3p", "T1", "T1s", "T1h", "T1hs"):
            extra = {x for x in argv if x.startswith("--") and x not in base}
            assert extra >= set(s["argv_flags"]) - set(base), (a, extra)
    assert seen >= 11
    t1, t1s = (json.loads((d / ("r9e1_%s.argv.json" % x)).read_text(encoding="utf-8"))["argv"] for x in ("T1", "T1s"))
    assert [x for x in t1s if x.startswith("--") and x not in t1] == ["--r9-bev-warp-shuffle"]        # T1s = T1 + the shuffle, nothing else
    t1h, t1hs = (json.loads((d / ("r9e1_%s.argv.json" % x)).read_text(encoding="utf-8"))["argv"] for x in ("T1h", "T1hs"))
    assert [x for x in t1h if x.startswith("--") and x not in t1] == ["--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init",
                                                                       "--r9-bev-warp-fill-only"]
    assert t1h[t1h.index("--r9-bev-warp-fuse-init") + 1] == "identity"
    assert [x for x in t1hs if x.startswith("--") and x not in t1h] == ["--r9-bev-warp-shuffle"]    # T1hs = T1h + the shuffle, nothing else
    b3, b3p = (json.loads((d / ("r9e1_%s.argv.json" % x)).read_text(encoding="utf-8"))["argv"] for x in ("B3", "B3p"))
    assert [x for x in b3p if x.startswith("--") and x not in b3] == ["--r9-box-inloss-nms"]          # B3p = B3 + the in-loss BEV-NMS
    m1, m2a, m2b = (json.loads((d / ("r9e1_%s.argv.json" % x)).read_text(encoding="utf-8"))["argv"] for x in ("M1", "M2a", "M2b"))
    w = lambda av: av[av.index("--w-map-hires") + 1]                                                   # noqa: E731
    assert (w(m1), w(m2a), w(m2b)) == ("1.0", "0.511", "1.018")


def test_resolve_ref_rules():
    assert V.resolve_ref("B1", None) == "C0" and V.resolve_ref("M2a", None) == "M1" and V.resolve_ref("T1s", None) == "T1"
    assert V.resolve_ref("T1h", None) == "C0" and V.resolve_ref("T1hs", None) == "T1h" and V.resolve_ref("T1hs", "C0") == "C0"
    assert V.resolve_ref("B3p", "B3") == "B3" and V.resolve_ref("M3", "M2b") == "M2b" and V.resolve_ref("B4", None) == "B2"
    for arm, ref in [("C0", None), ("C1", "C0"), ("M3", None), ("M3", "M1"), ("M1-FROZEN", None), ("M1-FROZEN", "C0"), ("B1", "M1"), ("nope", None)]:
        with pytest.raises(V.Refused):
            V.resolve_ref(arm, ref)
    assert V.main(["ref", "--arm", "B1"]) == 0 and V.main(["ref", "--arm", "M3"]) == 3


# ====================================================================================================================== end to end
def _strict_json(path):
    def bad(tok):
        raise ValueError("non-finite JSON token %s" % tok)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def _split(tmp, clips):
    p = Path(tmp) / ("split_%d.json" % len(clips))
    p.write_text(json.dumps({"DEV": {"sha12": clips}, "HOLD": {"sha12": ["%012x" % 0xb00]}}), encoding="utf-8")
    return p


def _dump(tmp, name, spec, split):
    tmp = Path(tmp)
    ck = tmp / (name + ".ckpt.json")
    ck.write_text(json.dumps(dict({"step": 4000}, **spec)), encoding="utf-8")
    (tmp / "config.json").write_text("{}", encoding="utf-8")
    out = tmp / ("dump_" + name)
    assert STUB.main(["--ckpt", str(ck), "--config", str(tmp / "config.json"), "--windows", str(split), "--split", "dev",
                      "--s2-windows", str(tmp / "s2.json"), "--out", str(out), "--require-ema"]) == 0
    return out


def _com(split, gate="0.5"):
    return ["--windows", str(split), "--split", "dev", "--gate", gate, "--boot", BOOT, "--seed", "0"]


def _eval_single(tmp, name, d, split, gate="0.5", lv=None):
    out = Path(tmp) / ("EVAL_%s.json" % name)
    assert E.main(["--packs", str(d / "perc_det.pkl"), "--rast-dir", str(d / "rast"), *_com(split, gate), *(["--lift-valid", str(lv)] if lv else []),
                   "--out", str(out)]) == 0
    return out


def _eval_pair(tmp, a, b, da, db, split, gate="0.5", lv=None):
    out = Path(tmp) / ("EVAL_%s_vs_%s.json" % (a, b))
    assert E.main(["--packs", str(da / "perc_det.pkl"), "--packs-b", str(db / "perc_det.pkl"), *_com(split, gate), *(["--lift-valid", str(lv)] if lv else []),
                   "--out", str(out)]) == 0
    return out


class Rig:
    """A floor (C0 miss [7, 8, 9] = AP@2 m 0.85; C1 miss [7, 8] = 0.90: floor 0.05, 2 x floor 0.10) plus any number of synthetic arms."""

    def __init__(self, tmp, clips=CLIPS):
        self.tmp = Path(tmp)
        (self.tmp / "s2.json").write_text("{}", encoding="utf-8")
        self.split = _split(self.tmp, clips)
        self.dumps = {"C0": _dump(self.tmp, "C0", {"miss": [7, 8, 9]}, self.split), "C1": _dump(self.tmp, "C1", {"miss": [7, 8]}, self.split)}
        self.eval0 = _eval_single(self.tmp, "C0", self.dumps["C0"], self.split)
        self.eval1 = _eval_single(self.tmp, "C1", self.dumps["C1"], self.split)
        self.evals = {"C0": self.eval0, "C1": self.eval1}
        pair = _eval_pair(self.tmp, "C1", "C0", self.dumps["C1"], self.dumps["C0"], self.split)
        self.fdir = self.tmp / "floor"
        assert F.main(["--eval-c0", str(self.eval0), "--eval-c1", str(self.eval1), "--paired", str(pair), "--rast-c0", str(self.dumps["C0"] / "rast"),
                       "--rast-c1", str(self.dumps["C1"] / "rast"), "--boot", BOOT, "--out-dir", str(self.fdir)]) == 0
        self.floor = self.fdir / "FLOOR.json"

    def arm(self, name, spec, split=None):
        split = split or self.split
        self.dumps[name] = _dump(self.tmp, name, spec, split)
        self.evals[name] = _eval_single(self.tmp, name, self.dumps[name], split)

    def verdict(self, arm, ref="C0", *, out=None, paired=None, floor=None, extra=(), rasters=True, lever=None):
        """``arm`` names the dump; ``lever`` (default ``arm``) is the sec. 4.3 arm it is judged as -- a test variant of B4 is judged as B4."""
        pair = paired or _eval_pair(self.tmp, arm, ref, self.dumps[arm], self.dumps[ref], self.split)
        od = Path(out or self.tmp / ("v_" + arm + "_" + ref))
        lever = lever or arm
        args = ["--floor", str(floor or self.floor), "--floor-eval-c0", str(self.eval0), "--paired", str(pair), "--eval-arm", str(self.evals[arm]),
                "--eval-ref", str(self.evals[ref]), "--arm", lever, "--ref", ref, "--out-dir", str(od)]
        if rasters:
            args += ["--rast-arm", str(self.dumps[arm] / "rast"), "--rast-ref", str(self.dumps[ref] / "rast")]
        rc = V.main(args + list(extra))
        return rc, _strict_json(od / ("VERDICT_%s.json" % lever)), (od / ("VERDICT_%s.md" % lever)).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    r = Rig(tmp_path_factory.mktemp("verdict"))
    r.arm("B2", {"miss": [7, 8, 9]})                       # an exact stand-in for C0 playing "the B2 winner" (B4's reference)
    r.arm("B4", {"miss": []})                              # AP@2 m 0.85 -> 1.00: delta +0.15 > 2 x 0.05
    r.arm("B4fp", {"miss": [], "fp_rows": 5})              # the same winner, plus 5 rows of false sidewalk flags in the 5-10 m band
    r.arm("B1", {"miss": [7, 8, 9]})                       # an unchanged lever
    return r


def test_end_to_end_a_clean_winner_keeps(rig):
    rc, v, md = rig.verdict("B4", "B2")
    assert rc == 0 and v["status"] == "OK" and v["verdict"] == "KEEP", (v["verdict"], v["failing_clauses"], v["not_computable"])
    r = v["reads"][0]
    assert r["key"] == "ap2m_all" and r["delta"] == pytest.approx(0.15, abs=1e-9) and r["floor"] == pytest.approx(0.05, abs=1e-9)
    assert r["keep_threshold"] == pytest.approx(0.10, abs=1e-9) and r["lever_read"]["keep_on_this_metric"] is True
    assert v["clause3"]["vetoes"] == [] and v["clause3"]["unread"] == [] and v["clause3"]["checked"] >= 5
    assert v["clause3"]["scope"].startswith("sec. 4.3 table keys")
    assert v["tier"] == "T0 (label level)" and v["ref"] == "B2" and v["kind"] == "lever"
    assert "VERDICT: KEEP" in md


def test_end_to_end_winner_that_is_worse_on_a_map_row_is_dropped_by_clause3(rig):
    rc, v, md = rig.verdict("B4fp", "B2", lever="B4")
    assert rc == 0 and v["verdict"] == "DROP", v["verdict"]
    assert v["reads"][0]["lever_read"]["keep_on_this_metric"] is True                       # it WINS its primary ...
    assert clauses(v) == [3]                                                                 # ... and clause 3 is the ONLY failing clause
    vet = v["clause3"]["vetoes"]
    assert vet and all(x["family"] == "MAP" for x in vet) and "nf_prec_5_10" in [x["metric"] for x in vet]
    assert any(x["floor_degenerate"] for x in vet)                                           # the floor's map rows are 0 (C0 = C1 rasters): degenerate
    assert "FAILING CLAUSE 3" in md and "Next lever" in md and "none named" in md           # Rule Zero: the drop names what is next, from the design text


def test_end_to_end_unchanged_lever_drops_on_clause_1_or_2_with_the_next_lever_named(rig):
    rc, v, md = rig.verdict("B1", "C0")
    assert rc == 0 and v["verdict"] == "DROP" and v["reads"][0]["key"] == "dx_0_40_mean"
    assert 1 in clauses(v) and "T1" in v["next_lever_on_drop"] and "FAILING CLAUSE" in md


def test_end_to_end_not_computable_arm_says_what_would_compute_it(rig):
    rig.arm("M1", {"miss": [7, 8, 9]})
    rc, v, md = rig.verdict("M1", "C0")
    # MAPWIRE: this rig's floor was read WITHOUT --lift-valid, so it has no nf_swedge_prec_lv_5_20 row: the primary reads NO-FLOOR, never a guess
    assert rc == 0 and v["verdict"] == "NOT-COMPUTABLE" and v["failing_clauses"] == []
    assert v["reads"][0]["state"] == "NO-FLOOR" and v["reads"][0]["key"] == "nf_swedge_prec_lv_5_20"
    assert "NO-FLOOR" in md and "re-read the floor with the current eval and --lift-valid" in md and "lift-valid" in md
    assert "nf_prec_5_10" in md                                                              # the proxies are listed, never stand in for the primary
    assert "Map primary mask: none" in md and v["clause3"]["vetoes"] == []


def test_end_to_end_records_floor_provenance_and_the_four_families(rig):
    rc, v, md = rig.verdict("B4", "B2", out=rig.tmp / "v_prov")
    import hashlib
    assert v["floor"]["sha256"] == hashlib.sha256(Path(rig.floor).read_bytes()).hexdigest() and v["floor"]["sha256"] in md
    assert v["families"] == [{"family": f, "status": s} for f, s in F.FAMILIES]
    assert md.count("PAUSED (PI 2026-09-19") == 1 and md.count("not computable: the E1 rig builds no planner") == 3
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert "| %s |" % fam in md
    assert "Tier **T0 (label level)**" in md and "paired_episode_cluster_bootstrap_stat" in md
    assert (rig.tmp / "v_prov" / "VERDICT_B4.json").is_file()


def test_end_to_end_the_same_windows_are_enforced(rig, tmp_path):
    # (a) a different gate: the arm eval was read at gate 0.6, the floor at 0.5
    rig.dumps["B4g"] = rig.dumps["B4"]
    rig.evals["B4g"] = _eval_single(tmp_path, "B4g", rig.dumps["B4"], rig.split, gate="0.6")
    pair = _eval_pair(tmp_path, "B4g", "B2", rig.dumps["B4"], rig.dumps["B2"], rig.split, gate="0.6")
    rc, v, md = rig.verdict("B4g", "B2", out=tmp_path / "va", paired=pair, lever="B4")
    assert rc == 2 and v["status"] == "REFUSED" and v["verdict"] == "REFUSED" and "P=R gate" in " ".join(v["problems"])
    assert "STATUS: REFUSED" in md and md.count("PAUSED (PI 2026-09-19") == 1                 # the four families are stated even when refused
    # (b) a dump with one window fewer
    import pickle
    d = _dump(tmp_path, "B4w", {"miss": []}, rig.split)
    recs = pickle.loads((d / "perc_det.pkl").read_bytes())
    (d / "perc_det.pkl").write_bytes(pickle.dumps(recs[:-1]))
    rig.dumps["B4w"] = d
    rig.evals["B4w"] = _eval_single(tmp_path, "B4w", d, rig.split)
    rc, v, _ = rig.verdict("B4w", "B2", out=tmp_path / "vb", lever="B4")
    assert rc == 2 and v["status"] == "REFUSED" and ("same windows" in " ".join(v["problems"]) or "n_windows" in " ".join(v["problems"]))
    # (c) the paired file was built the wrong way round (reference first)
    wrong = _eval_pair(tmp_path, "B2", "B4", rig.dumps["B2"], rig.dumps["B4"], rig.split)
    rc, v, _ = rig.verdict("B4", "B2", out=tmp_path / "vc", paired=wrong)
    assert rc == 2 and v["status"] == "REFUSED"
    # (d) a floor read on a DIFFERENT window set (4 clips, not 6)
    (tmp_path / "other").mkdir()
    other = Rig(tmp_path / "other", CLIPS[:4])
    rc, v, _ = rig.verdict("B4", "B2", out=tmp_path / "vd", floor=other.floor)
    assert rc == 2 and v["status"] == "REFUSED" and ("n_windows" in " ".join(v["problems"]) or "window" in " ".join(v["problems"]))


def test_end_to_end_refuses_a_bad_floor_a_floor_arm_and_unresolvable_references(rig, tmp_path):
    bad = tmp_path / "FLOOR_REFUSED.json"
    bad.write_text(json.dumps({"schema": "r9e1-floor-v1", "status": "REFUSED", "problems": ["x"], "rows": {}}), encoding="utf-8")
    rc, v, _ = rig.verdict("B4", "B2", out=tmp_path / "v1", floor=bad)
    assert rc == 2 and "floor is not OK" in " ".join(v["problems"])
    rig.evals["C1"], rig.dumps["C1"] = rig.eval1, rig.dumps["C1"]
    rc, v, _ = rig.verdict("C1", "C0", out=tmp_path / "v2")
    assert rc == 2 and "replicate floor" in " ".join(v["problems"])
    rc, v, _ = rig.verdict("B4", "C0", out=tmp_path / "v3")                                  # B4's reference is the B2 winner, not C0
    assert rc == 2 and "cannot be read against" in " ".join(v["problems"])
    rig.evals["M3"], rig.dumps["M3"] = rig.evals["B4"], rig.dumps["B4"]
    rc, v, _ = rig.verdict("M3", "B2", out=tmp_path / "v4")
    assert rc == 2 and "M3" in " ".join(v["problems"])
    rc, v, _ = rig.verdict("B4", "B2", out=tmp_path / "v5", extra=["--seed", "3"])           # a different seed than the floor's
    assert rc == 2 and "seed" in " ".join(v["problems"])


# ====================================================================================================================== MAPWIRE end to end: a lift-masked rig
def _mp():
    """The mapprim test module (its synthetic raster builder with EXACT known precisions). Imported lazily so the literal tests do not need scipy."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_r9_e1_eval_mapprim as MP                       # noqa: E402
    return MP


def _sidecar(path, lv, flip_far_cell=False):
    """A shared lift-valid mask for episodes e0..e5 (bool [400, 60]); ``flip_far_cell`` flips ONE cell at 30 m, where no raster has a flag or a GT cell
    of interest, so the numbers are unchanged but the sha256 differs."""
    m = np.array(lv, copy=True)
    if flip_far_cell:
        m[350, 30] = not m[350, 30]
    np.savez_compressed(str(path), **{"e%d" % i: m for i in range(len(CLIPS))})
    return Path(path)


def _paint(dump, arrs):
    """Replace a stub dump's rasters by ``arrs`` (the SAME planes in every episode / window: counts are exact, the paired CI has width 0)."""
    for f in sorted((Path(dump) / "rast").glob("*.npz")):
        np.savez_compressed(f, **arrs)


class MaskedRig(Rig):
    """The REAL eval -> floor -> verdict on rasters from the mapprim builder, with ONE shared lift-valid mask (sidecar A).

    Known answers (band counts of ``MP.BLOCKS``; the mask also marks the 20 x 20 cells of block Y lift-invalid):
      C0  full layout            lv 5-20 m = 60/170 = 0.352941   C1  drop C, H       lv 5-20 m = 60/150 = 0.4   (floor 0.047059, 2 x floor 0.094118)
      M1  drop G                 lv 5-20 m = 60/80  = 0.75       delta +0.397059 -> clears clauses 1-2
      B4x full layout + X        X = 400 false EDGE-only flags on lift-valid drivable cells at 12-14 m: lv 5-20 m = 60/570, psw-only proxies unchanged
      M1p drop G + Y             Y = 400 false SIDEWALK flags on lift-INVALID drivable cells at 7.5-9.5 m: lv unchanged (0.75), nf_prec_5_10 55/75 -> 55/475
    """
    PROFILES = {"C0": (), "C1": ("C", "H"), "M1": ("G",), "M1p": ("G",), "B2": (), "B4x": ()}

    def __init__(self, tmp):
        MP = _mp()
        self.MP, self.tmp = MP, Path(tmp)
        (self.tmp / "s2.json").write_text("{}", encoding="utf-8")
        self.split = _split(self.tmp, CLIPS)
        lv = MP._layout()[1]
        lv[75:95, 0:20] = False                                # block Y (rows 75-95 m*10, cols 270-290) is lift-invalid under the shared mask
        self.lv = lv
        self.mask = _sidecar(self.tmp / "mask_A.npz", lv)
        self.mask_b = _sidecar(self.tmp / "mask_B.npz", lv, flip_far_cell=True)
        self.dumps, self.evals = {}, {}
        for name, spec in (("C0", {"miss": [7, 8, 9]}), ("C1", {"miss": [7, 8]})):
            self.arm(name, spec)
        self.eval0, self.eval1 = self.evals["C0"], self.evals["C1"]
        pair = _eval_pair(self.tmp, "C1", "C0", self.dumps["C1"], self.dumps["C0"], self.split, lv=self.mask)
        self.fdir = self.tmp / "floor"
        assert F.main(["--eval-c0", str(self.eval0), "--eval-c1", str(self.eval1), "--paired", str(pair), "--rast-c0", str(self.dumps["C0"] / "rast"),
                       "--rast-c1", str(self.dumps["C1"] / "rast"), "--boot", BOOT, "--lift-valid", str(self.mask), "--out-dir", str(self.fdir)]) == 0
        self.floor = self.fdir / "FLOOR.json"
        # a floor read WITHOUT a mask (its evals too): the same rasters, no nf_swedge_* row anywhere
        self.eval0n = _eval_single(self.tmp, "C0n", self.dumps["C0"], self.split)
        self.eval1n = _eval_single(self.tmp, "C1n", self.dumps["C1"], self.split)
        pairn = _eval_pair(self.tmp, "C1n", "C0n", self.dumps["C1"], self.dumps["C0"], self.split)
        assert F.main(["--eval-c0", str(self.eval0n), "--eval-c1", str(self.eval1n), "--paired", str(pairn), "--rast-c0", str(self.dumps["C0"] / "rast"),
                       "--rast-c1", str(self.dumps["C1"] / "rast"), "--boot", BOOT, "--out-dir", str(self.tmp / "floor_n")]) == 0
        self.floor_n = self.tmp / "floor_n" / "FLOOR.json"

    def arm(self, name, spec, profile=None):
        profile = profile or name
        self.dumps[name] = _dump(self.tmp, name, spec, self.split)
        arrs, _ = self.MP._layout(drop=self.PROFILES[profile])
        if profile == "B4x":
            arrs["pedge"][120:140, 280:300] = 255              # false EDGE-only flags: the sidewalk-only proxies (psw) cannot see them
        if profile == "M1p":
            arrs["psw"][75:95, 270:290] = 255                  # false SIDEWALK flags on cells the shared mask marks lift-invalid
        _paint(self.dumps[name], arrs)
        self.evals[name] = _eval_single(self.tmp, name, self.dumps[name], self.split, lv=self.mask)

    def nomask_eval(self, name):
        """The single-arm eval of ``name`` read WITHOUT a mask (the evals of a mask-less floor read)."""
        if name == "C0":
            return self.eval0n
        d = self.tmp / "n"
        d.mkdir(exist_ok=True)
        return _eval_single(d, name + "n", self.dumps[name], self.split)

    def verdict(self, arm, ref="C0", *, lv="default", extra=(), **kw):
        mask = self.mask if lv == "default" else lv
        return super().verdict(arm, ref, extra=list(extra) + (["--lift-valid", str(mask)] if mask else []), **kw)


@pytest.fixture(scope="module")
def mrig(tmp_path_factory):
    r = MaskedRig(tmp_path_factory.mktemp("masked"))
    r.arm("M1", {"miss": [7, 8, 9]})                           # C0's boxes exactly: only the map differs
    r.arm("M1p", {"miss": [7, 8, 9]})
    r.arm("B2", {"miss": [7, 8, 9]}, profile="B2")             # C0 playing "the B2 winner" (B4's reference)
    r.arm("B4x", {"miss": []})                                 # AP@2 m +0.15 > 2 x 0.05, plus 400 false edge-only flags
    return r


def test_masked_floor_carries_the_swedge_rows_and_the_mask_sha(mrig):
    fl = json.loads(Path(mrig.floor).read_text(encoding="utf-8"))
    r = fl["rows"][LV]
    assert r["c0"] == pytest.approx(60 / 170, abs=1e-12) and r["c1"] == pytest.approx(60 / 150, abs=1e-12) and r["floor"] == pytest.approx(0.4 - 60 / 170, abs=1e-12)
    assert fl["map_inputs"]["lift_valid"]["sha256"] and fl["map_inputs"]["lift_valid"]["n_episodes"] == len(CLIPS)
    fn = json.loads(Path(mrig.floor_n).read_text(encoding="utf-8"))
    assert not [k for k in fn["rows"] if k.startswith("nf_swedge")] and fn["map_inputs"]["lift_valid"] is None


def test_end_to_end_m1_keeps_on_a_lift_masked_rig(mrig):
    rc, v, md = mrig.verdict("M1", "C0")
    assert rc == 0 and v["status"] == "OK" and v["verdict"] == "KEEP", (v.get("verdict"), v.get("failing_clauses"), v.get("not_computable"), v.get("problems"))
    r = v["reads"][0]
    assert r["key"] == LV and r["state"] == "READ" and r["role"] == "primary" and r["higher_is_better"] is True
    assert r["ref_point"] == pytest.approx(60 / 170, abs=1e-9) and r["arm_point"] == pytest.approx(60 / 80, abs=1e-9)
    assert r["delta"] == pytest.approx(0.75 - 60 / 170, abs=1e-9) and r["floor"] == pytest.approx(0.4 - 60 / 170, abs=1e-9)
    assert r["lever_read"]["keep_on_this_metric"] is True and r["separated"] is True
    assert v["clause3"]["vetoes"] == [] and v["clause3"]["missing_uncovered"] == [] and v["clause3"]["missing_from_floor"] == []
    assert v["lift_valid"]["same_mask"] is True and v["lift_valid"]["verdict"]["sha256"] == v["lift_valid"]["floor"]["sha256"]
    assert "VERDICT: KEEP" in md and "EQUAL to the floor's" in md and v["lift_valid"]["verdict"]["sha256"] in md
    assert (v["reads"][1]["state"], v["reads"][2]["state"]) == ("NOT-COMPUTABLE", "NOT-COMPUTABLE")        # lane tol-IoU / boundary recall stay 'beside', listed, never decide


def test_end_to_end_a_box_lever_is_vetoed_by_the_swedge_row(mrig):
    rc, v, md = mrig.verdict("B4x", "B2", lever="B4")
    assert rc == 0 and v["verdict"] == "DROP", (v["verdict"], v.get("not_computable"), v.get("problems"))
    assert v["reads"][0]["lever_read"]["keep_on_this_metric"] is True and clauses(v) == [3]              # it WINS ap2m_all; clause 3 is the only failing clause
    names = [x["metric"] for x in v["clause3"]["vetoes"]]
    assert names == [LV], names                                                                          # the sidewalk-only proxies cannot see edge-only flags
    x = v["clause3"]["vetoes"][0]
    assert x["delta"] == pytest.approx(60 / 570 - 60 / 170, abs=1e-9) and x["family"] == "MAP" and x["in_table_primaries"] is True
    assert "nf_swedge_prec_lv_10_20" in [y["metric"] for y in v["clause3"]["informative_worse"]]          # the band row is informative only
    assert "FAILING CLAUSE 3" in md and LV in md


def test_end_to_end_a_map_arm_is_not_vetoed_by_its_own_proxy_but_another_family_is(mrig):
    rc, v, md = mrig.verdict("M1p", "C0", lever="M1")
    assert rc == 0 and v["verdict"] == "KEEP", (v["verdict"], v["failing_clauses"], v["clause3"]["vetoes"])
    assert v["reads"][0]["arm_point"] == pytest.approx(0.75, abs=1e-9)                                   # Y sits on lift-invalid cells: the primary is untouched
    assert v["clause3"]["vetoes"] == [] and {"nf_prec_5_10", "nf_prec_10_20", LV} <= set(v["clause3"]["excluded"])
    # the SAME rasters judged as B1 (another family; its own primary dx_0_40_mean is flat -> clause 1/2 fail) are vetoed on the proxy
    rc, b, _ = mrig.verdict("M1p", "C0", lever="B1", out=mrig.tmp / "v_M1p_as_B1")
    assert rc == 0 and b["verdict"] == "DROP" and "nf_prec_5_10" in [x["metric"] for x in b["clause3"]["vetoes"]]
    assert [x["delta"] for x in b["clause3"]["vetoes"] if x["metric"] == "nf_prec_5_10"][0] == pytest.approx(55 / 475 - 55 / 75, abs=1e-9)



def _run_verdict(rig, arm, ref, *, floor, floor_eval, evals, pair, lv, out, lever=None, rasters=True):
    """The verdict CLI with every input spelled out (the Rig helper always uses the rig's own floor / evals)."""
    lever = lever or arm
    args = ["--floor", str(floor), "--floor-eval-c0", str(floor_eval), "--paired", str(pair), "--eval-arm", str(evals[arm]), "--eval-ref", str(evals[ref]),
            "--arm", lever, "--ref", ref, "--out-dir", str(out)]
    if rasters:
        args += ["--rast-arm", str(rig.dumps[arm] / "rast"), "--rast-ref", str(rig.dumps[ref] / "rast")]
    if lv:
        args += ["--lift-valid", str(lv)]
    rc = V.main(args)
    return rc, _strict_json(Path(out) / ("VERDICT_%s.json" % lever)), (Path(out) / ("VERDICT_%s.md" % lever)).read_text(encoding="utf-8")


def test_end_to_end_m1_is_not_computable_on_a_maskless_floor(mrig):
    # the floor AND the evals were read without a mask; the verdict gets none either: the primary has no row -> NO-FLOOR -> NOT-COMPUTABLE, with the reason
    en = {n: mrig.nomask_eval(n) for n in ("C0", "M1", "B2", "B4x")}
    pair = _eval_pair(mrig.tmp / "n", "M1n", "C0n", mrig.dumps["M1"], mrig.dumps["C0"], mrig.split)
    rc, v, md = _run_verdict(mrig, "M1", "C0", floor=mrig.floor_n, floor_eval=mrig.eval0n, evals=en, pair=pair, lv=None, out=mrig.tmp / "n" / "v1")
    assert rc == 0 and v["verdict"] == "NOT-COMPUTABLE" and v["failing_clauses"] == [] and v["reads"][0]["state"] == "NO-FLOOR"
    assert "re-read the floor with the current eval and --lift-valid" in md and "Map primary mask: none" in md
    assert v["lift_valid"] == {"verdict": None, "floor": None, "same_mask": True}
    # a box lever on the same mask-less floor is NOT blocked: the lv primary is COVERED by its two sidewalk-only proxies, and the md says so
    pair = _eval_pair(mrig.tmp / "n", "B4xn", "B2n", mrig.dumps["B4x"], mrig.dumps["B2"], mrig.split)
    rc, v, md = _run_verdict(mrig, "B4x", "B2", floor=mrig.floor_n, floor_eval=mrig.eval0n, evals=en, pair=pair, lv=None, out=mrig.tmp / "n" / "v2", lever="B4")
    assert rc == 0 and v["verdict"] == "KEEP" and v["clause3"]["missing_from_floor"] == [LV] and v["clause3"]["missing_uncovered"] == []
    assert v["clause3"]["missing_covered"][0]["covered_by_proxies"] == ["nf_prec_5_10", "nf_prec_10_20"] and "covered by proxies" in md


def _mask_mismatch_args(mrig):
    """The floor is on mask A; the arm / reference evals AND the verdict are on mask B (B = A with one 35 m cell flipped: every number equal, the sha256
    different).  nearfield_rows' eval cross-check is then satisfied and ONLY the verdict's own mask-sha check can refuse.  -> (positional, keyword) args."""
    d = mrig.tmp / "maskB"
    d.mkdir(exist_ok=True)
    evb = {n: _eval_single(d, n + "B", mrig.dumps[n], mrig.split, lv=mrig.mask_b) for n in ("C0", "M1")}
    pair = _eval_pair(d, "M1B", "C0B", mrig.dumps["M1"], mrig.dumps["C0"], mrig.split)
    return ((str(mrig.floor), str(pair), str(evb["M1"]), str(evb["C0"]), "M1", "C0"),
            dict(rast_arm=str(mrig.dumps["M1"] / "rast"), rast_ref=str(mrig.dumps["C0"] / "rast"), floor_eval_c0=str(mrig.eval0), lift_valid=str(mrig.mask_b)))


def test_the_mask_sha_must_equal_the_floors_or_the_verdict_is_refused(mrig):
    args, kw = _mask_mismatch_args(mrig)
    v = V.build_verdict(*args, **kw)
    assert v["status"] == "REFUSED" and v["verdict"] == "REFUSED", v.get("verdict")
    bad = [x for x in v["comparability"] if x["check"].startswith("lift-valid mask equals the floor's")]
    assert len(bad) == 1 and bad[0]["ok"] is False and v["lift_valid"]["same_mask"] is False
    assert v["problems"][0].count("comparability:") == 1                                                                        # the ONLY problem
    # the control: the very same call on the floor's own mask is accepted (so the refusal above is the sha check, nothing else)
    pair = _eval_pair(mrig.tmp, "M1", "C0", mrig.dumps["M1"], mrig.dumps["C0"], mrig.split, lv=mrig.mask)
    ok = V.build_verdict(str(mrig.floor), str(pair), str(mrig.evals["M1"]), str(mrig.evals["C0"]), "M1", "C0", rast_arm=kw["rast_arm"], rast_ref=kw["rast_ref"],
                         floor_eval_c0=str(mrig.eval0), lift_valid=str(mrig.mask))
    assert ok["status"] == "OK" and ok["lift_valid"]["same_mask"] is True and ok["verdict"] == "KEEP"


def test_one_side_with_a_mask_and_the_other_without_is_refused(mrig, tmp_path):
    en = {n: mrig.nomask_eval(n) for n in ("C0", "M1")}
    pair_n = _eval_pair(mrig.tmp / "n", "M1n", "C0n", mrig.dumps["M1"], mrig.dumps["C0"], mrig.split)
    pair_m = _eval_pair(mrig.tmp, "M1", "C0", mrig.dumps["M1"], mrig.dumps["C0"], mrig.split, lv=mrig.mask)
    # (a) the floor has NO mask (floor_n), the arm / reference evals and the verdict are on mask A
    rc, v, _ = _run_verdict(mrig, "M1", "C0", floor=mrig.floor_n, floor_eval=mrig.eval0n, evals=mrig.evals, pair=pair_m, lv=mrig.mask, out=tmp_path / "va")
    assert rc == 2 and v["status"] == "REFUSED" and "lift-valid mask equals the floor's" in " ".join(v["problems"])
    assert [x["ok"] for x in v["comparability"] if x["check"].startswith("lift-valid mask equals the floor's")] == [False]
    # (b) the floor HAS a mask, the arm / reference evals and the verdict have none
    rc, v, _ = _run_verdict(mrig, "M1", "C0", floor=mrig.floor, floor_eval=mrig.eval0, evals=en, pair=pair_n, lv=None, out=tmp_path / "vb")
    assert rc == 2 and v["status"] == "REFUSED" and "lift-valid mask equals the floor's" in " ".join(v["problems"])
    assert [x["ok"] for x in v["comparability"] if x["check"].startswith("lift-valid mask equals the floor's")] == [False]
    # (c) a mask with no rasters would be silently ignored: refused
    rc, v, _ = _run_verdict(mrig, "M1", "C0", floor=mrig.floor, floor_eval=mrig.eval0, evals=mrig.evals, pair=pair_m, lv=mrig.mask, out=tmp_path / "vc", rasters=False)
    assert rc == 2 and "needs --rast-arm" in " ".join(v["problems"])
    # (d) a mask the evaluator refuses (an episode it lacks) is a REFUSED verdict, never a guess
    short = tmp_path / "short.npz"
    np.savez_compressed(str(short), e0=mrig.lv, e1=mrig.lv)
    rc, v, _ = _run_verdict(mrig, "M1", "C0", floor=mrig.floor, floor_eval=mrig.eval0, evals=mrig.evals, pair=pair_m, lv=short, out=tmp_path / "vd")
    assert rc == 2 and v["status"] == "REFUSED" and "lift-valid mask refused" in " ".join(v["problems"])


def test_end_to_end_clean_winner_on_a_maskless_rig_says_covered_by_proxies(rig):
    rc, v, md = rig.verdict("B4", "B2", out=rig.tmp / "v_cov")
    assert rc == 0 and v["verdict"] == "KEEP" and v["clause3"]["missing_from_floor"] == [LV] and v["clause3"]["missing_uncovered"] == []
    assert "covered by proxies" in md and "Scope keys the floor has NO row for (1)" in md and "Map primary mask: none" in md


def test_dumpcheck_cli(tmp_path):
    a = {"order_digest": "x", "n_windows": 12, "n_clips": 6, "step": 4000, "weights_are_ema": True, "ckpt_kind": "ema_model_only", "SMOKE_RESTRICTED": False}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(a), encoding="utf-8")
    assert V.check_dumps([str(pa), str(pb)]) == []
    pb.write_text(json.dumps(dict(a, order_digest="y", weights_are_ema=False)), encoding="utf-8")
    bad = V.check_dumps([str(pa), str(pb)])
    assert any("order_digest" in b for b in bad) and any("weights_are_ema" in b for b in bad)
    assert V.main(["dumpcheck", "--dump", str(pa), "--dump", str(pb)]) == 1


# ====================================================================================================================== the arm-aware state check
def _lines(*ev):
    return "".join("2026-11-04T%02d:00:00Z %s\n" % (i % 24, e) for i, e in enumerate(ev))


def _run(root, arm, *, done=True, zerograd=True, fail=False, step=4000, ckpt=True, cfg=True):
    d = Path(root) / "runs" / ("r9e1-" + arm)
    d.mkdir(parents=True, exist_ok=True)
    s = {"done": done, "final_step": step, "steps_requested": 4000, "arm": "flat"}      # the SUPERVISOR's done-marker fields (sup_refcv7.sh)
    if zerograd:
        s["zerograd_gate"] = {"ok": True}
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    if ckpt:
        (d / "model_ema.pt").write_bytes(b"x")
    if cfg:
        (d / "config.json").write_text("{}", encoding="utf-8")
    if fail:
        (d / "ZEROGRAD_FAIL.json").write_text("{}", encoding="utf-8")
    return d


def _floor_read(root, status="OK"):
    rd = Path(root) / "read"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "FLOOR.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    return rd


START = ["START commit aaa tree /t arms [C0 C1 B1]"]


def test_state_waits_for_the_floor_then_for_the_arm_then_reads(tmp_path):
    rd = tmp_path / "read"
    s = V.arm_state("B1", "C0", _lines(*START), tmp_path, rd)
    assert s["state"] == "WAIT" and "floor read" in s["reason"]                                          # no FLOOR.json yet
    _floor_read(tmp_path)
    s = V.arm_state("B1", "C0", _lines(*START), tmp_path, rd)
    assert s["state"] == "WAIT" and "not launched yet" in s["reason"]
    b1 = _run(tmp_path, "B1", done=False)
    s = V.arm_state("B1", "C0", _lines(*START, "B1-LAUNCH supervisor x; run dir %s" % b1.as_posix()), tmp_path, rd)
    assert s["state"] == "WAIT" and "training" in s["reason"]
    _run(tmp_path, "B1")
    s = V.arm_state("B1", "C0", _lines(*START, "B1-LAUNCH supervisor x; run dir %s" % b1.as_posix(), "B1-DONE rc 0; ok"), tmp_path, rd)
    assert s["state"] == "READY" and s["run"] == str(b1)


def test_state_aborts_on_a_refused_floor_a_failed_arm_and_disagreeing_markers(tmp_path):
    rd = _floor_read(tmp_path, "REFUSED")
    assert V.arm_state("B1", "C0", _lines(*START), tmp_path, rd)["state"] == "ABORT"
    rd = _floor_read(tmp_path)
    for bad in ("B1-FAILED rc 137: no done-marker", "B1-ZEROGRAD-FAIL rc 0; x", "B1-REFUSED supervisor refused", "B1-NOT-LAUNCHED x"):
        s = V.arm_state("B1", "C0", _lines(*START, "B1-LAUNCH s; run dir /x", bad), tmp_path, rd)
        assert s["state"] == "ABORT" and bad.split()[0] in s["reason"] and s["line"].endswith(bad)
    assert V.arm_state("B1", "C0", _lines(*START, "B1-DONE rc 0; ok"), tmp_path, rd)["state"] == "ABORT"        # DONE logged, no run dir markers
    assert "ALLDONE without B1-DONE" in V.arm_state("B1", "C0", _lines(*START, "ALLDONE chain finished"), tmp_path, rd)["reason"]
    (tmp_path / "read" / "READ_ABORTED").write_text("x", encoding="utf-8")
    assert V.arm_state("B1", "C0", _lines(*START), tmp_path, rd)["state"] == "ABORT"


def test_state_reads_an_arm_finished_in_an_earlier_chain_segment_and_a_non_floor_reference(tmp_path):
    rd = _floor_read(tmp_path)
    b1 = _run(tmp_path, "B1")
    old = ["START commit aaa", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE rc 0; ok", "ALLDONE chain finished", "START commit bbb arms [M1 M2a]"]
    assert V.arm_state("B1", "C0", _lines(*old), tmp_path, rd)["state"] == "READY"                         # only the run dir's markers decide DONE
    m1 = _run(tmp_path, "M1", done=False)
    seg = _lines(*old, "M1-LAUNCH s; run dir %s" % m1.as_posix())
    _run(tmp_path, "M2a")
    s = V.arm_state("M2a", "M1", seg + _lines("M2a-DONE ok"), tmp_path, rd)
    assert s["state"] == "WAIT" and "M1" in s["reason"]                                                       # the reference is not done yet
    _run(tmp_path, "M1")
    s = V.arm_state("M2a", "M1", seg + _lines("M1-DONE ok", "M2a-DONE ok"), tmp_path, rd)
    assert s["state"] == "READY" and s["ref_run"].endswith("r9e1-M1")


def test_state_cli_writes_a_sourceable_file_and_refuses_an_unresolvable_reference(tmp_path):
    rd = _floor_read(tmp_path)
    st = tmp_path / "gate3"
    st.mkdir()
    (st / "CHAIN_STATUS.txt").write_text(_lines(*START), encoding="utf-8")
    out = tmp_path / "state.env"
    assert V.main(["state", "--arm", "B1", "--status", str(st / "CHAIN_STATUS.txt"), "--root", str(tmp_path), "--read-dir", str(rd), "--out", str(out)]) == 0
    env = dict(re.findall(r"^S_(\w+)=(.*)$", out.read_text(encoding="utf-8"), re.M))
    assert env["STATE"] == "WAIT" and set(env) >= {"STATE", "REASON", "LINE", "RUN", "REF_RUN", "SIG"}
    assert V.main(["state", "--arm", "M3", "--status", str(st / "CHAIN_STATUS.txt"), "--root", str(tmp_path), "--read-dir", str(rd), "--out", str(out)]) == 0
    assert "STATE=ABORT" in out.read_text(encoding="utf-8")                                                   # M3 with no --ref: abort at once, do not wait 14 days
