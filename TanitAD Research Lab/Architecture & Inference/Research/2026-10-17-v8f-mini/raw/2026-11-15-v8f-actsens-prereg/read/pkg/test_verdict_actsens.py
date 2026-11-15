"""Synthetic test of verdict_actsens.py: every outcome row of card section 6 fires once, plus deliberate-regression cases. T0 synthetic numbers, no data. Run: python test_verdict_actsens.py (or pytest)."""
import copy, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verdict_actsens as V

QL = ("LONGITUDINAL|0-2s|acc_mae", "LATERAL|0-2s|yaw_mae")


def c(d, w=0.05):
    return {"delta": d, "lo": d - w, "hi": d + w, "label": "x", "n_windows": 994, "n_episodes": 71}


def gain(p, w=0.02):
    return {"point": p, "lo": p - w, "hi": p + w}


def ia(g_lon, g_lat, met=True):
    return {"ia3": {"1|10": {"gain_lon": gain(g_lon), "gain_lat": gain(g_lat)}}, "band_met": {lv: {"beats": met, "healthy": True, "worse": False} for lv in "123"}}


def arm(name, k, acc=0.0, yaw=0.0, hold_acc=0.5, hold_yaw=0.5, g_lon=0.30, g_lat=-0.03, rep=0.001, met=True, w=0.05):
    cells = {QL[0]: {"vs_r35": c(acc, w), "vs_hold": c(hold_acc, w), "rep_arm": c(rep), "rep_r35": c(rep)},
             QL[1]: {"vs_r35": c(yaw, w), "vs_hold": c(hold_yaw, w), "rep_arm": c(rep), "rep_r35": c(rep)}}
    return {"name": name, "chassis": k, "hwm": {"cells": cells}, "ia": ia(g_lon, g_lat, met)}


REF = {k: {"ia3": {"1|10": {"gain_lon": gain(0.30), "gain_lat": gain(-0.03)}}, "passes": {lv: {"beats": True, "healthy": True, "worse": False} for lv in "123"}} for k in (0, 1)}


def world(idm, div, null_delta=0.003):
    """idm / div: dict(k -> arm kwargs). NULL arms read null_delta against R35 and the R35 gains."""
    arms = {}
    for lv, spec in (("IDM", idm), ("DIV", div)):
        for k in (0, 1):
            arms[f"{lv}_s{k}"] = arm(f"{lv}_s{k}", k, **spec[k])
    for k, n in ((0, "NULL_s2"), (1, "NULL_s3")):
        arms[n] = arm(n, k, acc=null_delta, yaw=null_delta)
    return arms


both = lambda **kw: {0: dict(kw), 1: dict(kw)}
CEIL = both(acc=-0.9, hold_acc=-0.4)
NARR = both(acc=-0.20)
MECH = both(g_lon=0.60)
NUL = both()
HARM = both(acc=+0.30)


def labels(idm, div, **kw):
    return V.run(world(idm, div, **kw), REF)["labels"]


def test_each_outcome_fires_once():
    seen = set()
    for spec, want in ((CEIL, "ACTSENS-CEILING"), (NARR, "ACTSENS-NARROWS"), (MECH, "ACTSENS-MECH-ONLY"), (NUL, "ACTSENS-NULL"), (HARM, "ACTSENS-HARM")):
        got = labels(spec, NUL)["IDM"]
        assert got == want, (want, got)
        seen.add(got)
    sd = labels({0: dict(acc=-0.20), 1: dict()}, NUL)["IDM"]
    assert sd == "SEEDS-DISAGREE", sd
    seen.add(sd)
    assert len(seen) == 6


def test_band_met_lost_is_harm():
    spec = {0: dict(acc=-0.20), 1: dict(acc=-0.20, met=False)}
    assert labels(spec, NUL)["IDM"] == "ACTSENS-HARM"


def test_floor_regression_flips_narrows():
    """DELIBERATE REGRESSION: the same lever effect (-0.20) with a training floor of 0.30 (NULL moves 0.30 from R35) must NOT read NARROWS: the floor is what makes it a lever effect."""
    assert labels(NARR, NUL, null_delta=0.003)["IDM"] == "ACTSENS-NARROWS"
    assert labels(NARR, NUL, null_delta=0.30)["IDM"] != "ACTSENS-NARROWS"


def test_inference_floor_is_one_value_over_both_chassis():
    """DELIBERATE REGRESSION (card 4.4: F_I carries no chassis subscript): a lever effect of -0.20 at both chassis, with a large inference replicate (0.30)
    at chassis 1 ONLY, is inside the floor at BOTH chassis and reads ACTSENS-NULL. A per-chassis F_I lets chassis 0 pass on its own small replicate,
    and the label becomes SEEDS-DISAGREE (MEASURED on the per-chassis version: this test goes red)."""
    spec = {0: dict(acc=-0.20, rep=0.001), 1: dict(acc=-0.20, rep=0.30)}
    r = V.run(world(spec, NUL), REF)
    assert r["labels"]["IDM"] == "ACTSENS-NULL", r["labels"]
    f = r["levers"]["IDM"]["floors"]
    assert f[0]["acc_mae"]["F_I"] == f[1]["acc_mae"]["F_I"] == 0.30 and f[0]["acc_mae"]["F_I_k"] == 0.001
    assert labels(both(acc=-0.20, rep=0.001), NUL)["IDM"] == "ACTSENS-NARROWS"


def test_sign_regression():
    """DELIBERATE REGRESSION: flipping the sign convention (a worse arm read as better) must change the label; HARM data never reads NARROWS or CEILING."""
    lab = labels(HARM, NUL)["IDM"]
    assert lab == "ACTSENS-HARM" and lab not in ("ACTSENS-NARROWS", "ACTSENS-CEILING")
    flipped = both(acc=-0.30)
    assert labels(flipped, NUL)["IDM"] == "ACTSENS-NARROWS"


def test_mech_needs_disjoint_ci_and_gap_over_null():
    thin = both(g_lon=0.32)                                  # above R35 by 0.02 but CIs overlap
    assert labels(thin, NUL)["IDM"] == "ACTSENS-NULL"


def test_ceiling_needs_both_chassis_and_not_worse_other():
    one = {0: dict(acc=-0.9, hold_acc=-0.4), 1: dict(acc=-0.9, hold_acc=+0.4)}
    assert labels(one, NUL)["IDM"] != "ACTSENS-CEILING"
    worse_other = both(acc=-0.9, hold_acc=-0.4, yaw=+0.5)
    assert labels(worse_other, NUL)["IDM"] == "ACTSENS-HARM"


def test_incomplete_without_null_or_chassis():
    arms = world(NARR, NARR)
    del arms["NULL_s3"]
    r = V.run(arms, REF)
    assert r["labels"]["IDM"] == "INCOMPLETE" and r["labels"]["DIV"] == "INCOMPLETE"
    arms = world(NARR, NARR); del arms["IDM_s1"]
    r = V.run(arms, REF)
    assert r["labels"]["IDM"] == "INCOMPLETE" and r["labels"]["DIV"] != "INCOMPLETE"


def test_lever_not_run_is_not_null():
    r = V.run(world(NUL, NUL), REF, {"IDM": False, "DIV": True})
    assert r["labels"]["IDM"].startswith("UNDECIDED") and r["labels"]["DIV"] == "ACTSENS-NULL"


def test_lever_of_record():
    r = V.run(world(both(acc=-0.20), both(acc=-0.40)), REF)
    assert r["labels"] == {"IDM": "ACTSENS-NARROWS", "DIV": "ACTSENS-NARROWS"} and r["lever_of_record"]["lever"] == "DIV"


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("PASS", k)
    print(n, "passed")
