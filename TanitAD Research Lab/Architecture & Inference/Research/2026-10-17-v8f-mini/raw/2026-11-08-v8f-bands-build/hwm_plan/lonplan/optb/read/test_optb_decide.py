"""Tests of the mechanical outcome logic of optb_decide.py (no data needed) plus a reproduction check against the banked LON-PLAN contrast when the smoke output exists.
Run: python -m pytest test_optb_decide.py -q --basetemp <dir under read/tmp>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
import optb_decide as O

HERE = os.path.dirname(os.path.abspath(__file__))


def row(d, w=0.05):
    return {"delta": d, "lo": d - w, "hi": d + w}


def ci(acc=(-0.2, -0.2), yaw=(-0.01, -0.01), floor=(0.05, 0.02), hold_acc=(0.1, 0.1), hold_yaw=(0.3, 0.3), mech=True, bml=None, c23=True):
    return {"dj": {"acc_mae": list(acc), "yaw_mae": list(yaw)},
            "dbar": {"acc_mae": row(sum(acc) / 2), "yaw_mae": row(sum(yaw) / 2, 0.003)},
            "floor": {"acc_mae": floor[0], "yaw_mae": floor[1]},
            "hold": {"a": {"acc_mae": {"hi": hold_acc[0]}, "yaw_mae": {"hi": hold_yaw[0]}}, "b": {"acc_mae": {"hi": hold_acc[1]}, "yaw_mae": {"hi": hold_yaw[1]}}},
            "mech": mech, "band_met_lost": bml, "c2_c3_hold": c23}


def lab(c):
    return O.chassis_label(c)["label"]


def test_better_worse_rule():
    assert O.better_worse(row(-0.2), [-0.1, -0.3], 0.05) == {"better": True, "worse": False}
    assert O.better_worse(row(-0.04), [-0.1, -0.3], 0.05)["better"] is False          # below the floor
    assert O.better_worse(row(-0.2), [-0.1, 0.05], 0.05)["better"] is False           # one lever seed above 0
    assert O.better_worse(row(-0.2, 0.3), [-0.1, -0.3], 0.05)["better"] is False       # CI crosses 0
    assert O.better_worse(row(0.2), [0.1, 0.3], 0.05) == {"better": False, "worse": True}
    assert O.better_worse(row(0.04), [0.1, 0.3], 0.05)["worse"] is False


def test_labels_each_row_of_the_table():
    assert lab(ci(hold_acc=(-0.5, -0.4))) == "OPTB-CEILING"                              # both lever arms below HOLD on acc
    assert lab(ci()) == "OPTB-NARROWS"                                                  # BETTER, HOLD not beaten
    assert lab(ci(acc=(-0.01, -0.02), yaw=(0.0, 0.0), mech=True)) == "OPTB-MECH-ONLY"
    assert lab(ci(acc=(-0.01, -0.02), yaw=(0.0, 0.0), mech=False)) == "OPTB-NULL"
    assert lab(ci(acc=(0.3, 0.3), yaw=(0.0, 0.0))) == "OPTB-HARM"                       # WORSE on acc
    assert lab(ci(yaw=(0.2, 0.2))) == "OPTB-HARM"                                        # WORSE on yaw although acc is BETTER (not WORSE on the other is required)
    assert lab(ci(bml=True)) == "OPTB-HARM"                                              # BAND-MET lost
    assert lab(ci(acc=(-0.3, 0.1))) == "SEEDS-DISAGREE"                                  # lever seeds differ in sign
    assert lab(ci(acc=(-0.01, -0.02), yaw=(0.0, 0.0), mech=None)) == "MECH-NOT-COMPUTABLE"


def test_ceiling_goes_red_when_one_lever_arm_misses_hold():
    c = ci(hold_acc=(-0.5, 0.1), hold_yaw=(0.3, 0.3))
    assert lab(c) != "OPTB-CEILING"


def test_null_without_c2_c3_is_flagged_not_passed():
    r = O.chassis_label(ci(acc=(-0.01, -0.02), yaw=(0.0, 0.0), mech=False, c23=None))
    assert r["label"] == "OPTB-NULL" and any("UNVERIFIED" in n for n in r["notes"])


def L(label):
    return {"label": label, "bw": {}, "notes": []}


def test_overall_labels():
    assert O.overall_label({0: L("OPTB-NARROWS"), 1: L("OPTB-NARROWS")})["label"] == "OPTB-NARROWS"
    assert O.overall_label({0: L("OPTB-CEILING")})["label"] == "INCOMPLETE"
    o = O.overall_label({0: L("OPTB-CEILING"), 1: L("OPTB-MECH-ONLY")})
    assert o["label"] == "SEEDS-DISAGREE" and o["less_favourable"] == "OPTB-MECH-ONLY" and "LON-PLAN" in o["next"]
    assert O.overall_label({0: L("OPTB-CEILING"), 1: L("OPTB-HARM")})["label"] == "OPTB-HARM"
    assert O.overall_label({0: L("SEEDS-DISAGREE"), 1: L("OPTB-NARROWS")})["label"] == "SEEDS-DISAGREE"


@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "smoke", "decide_smoke_B2000.json")), reason="B 2000 smoke output not present")
def test_smoke_reproduces_the_banked_lonplan_contrast_exactly():
    """the stand-in pair FTRSP_s0 := A1_s0, FTNULL_s0 := A0_s0 must reproduce the banked c_A1_A0_s0.json row (same reader, same estimator): delta, lo, hi equal to the banked 4-decimal rounding."""
    banked = json.load(open("F:/TanitAD-scratch/actrsp/lonplan/readout/c_A1_A0_s0.json", encoding="utf-8"))
    mine = json.load(open(os.path.join(HERE, "smoke", "decide_smoke_B2000.json"), encoding="utf-8"))
    n = 0
    for m in ("acc_mae", "yaw_mae"):
        b = [r for r in banked["rows"] if r["kind"] == "contrast" and r["band"] == "0-2s" and r["metric"] == m and r["a"] == "A" and r["b"] == "B"][0]
        x = [r for r in mine["rows"] if r["kind"] == "deciding_pair" and r["arm"] == "FTRSP_s0" and r["metric"] == m][0]
        for k in ("delta", "lo", "hi"):
            assert abs(x[k] - b[k]) <= 6e-5, (m, k, x[k], b[k])    # the banked JSON rounds to 4 decimals
        n += 1
    assert n == 2
