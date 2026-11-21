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


# ------------------------------------------------------------------------------------------ STRATEGIC (PI 2026-11-16, the pause ended): the readers state the family, not a pause
HERE_V = os.path.dirname(os.path.abspath(__file__))
OLD_PAUSE = "PAUSED (PI item 25, 2026-09-19): not a pass, not omitted"


def _reader_text(name):
    for d in (HERE_V, os.environ.get("STRATEGIC_CODE_DIR", HERE_V)):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    raise AssertionError(f"{name} not found beside this test or under STRATEGIC_CODE_DIR")


def test_no_pause_string_in_the_three_readers_with_a_control():
    for name in ("hwm_eval.py", "assemble_arm.py", "verdict_actsens.py"):
        t = _reader_text(name)
        assert t.count("PAUSED") == 0, name                                  # the grep that must return 0 ...
        assert t.count("STRATEGIC") > 0, name                                # ... and its same-breath control, a string that exists, which must return non-zero
    assert OLD_PAUSE.count("PAUSED") == 1                                    # the probe can fire on the old text


def _strategic_cells_for(arm):
    arm["hwm"]["cells"]["STRATEGIC|full6s|s_adh"] = {"vs_hold": c(0.40), "vs_logged": c(-0.30)}
    arm["strategic"] = {"state": "REPORTED IN FULL", "arms": {"hwm_oracle": {"tier": "T0"}}, "logged_path": {"s_adh_ceiling": {"mean": 1.0, "n_windows": 8}}}


def test_verdict_carries_the_strategic_section_and_the_labels_do_not_move():
    w = world(NUL, NUL)
    base = V.run(w, REF)["labels"]
    _strategic_cells_for(w["IDM_s0"])
    out = V.run(w, REF)
    assert out["labels"] == base                                             # STRATEGIC is reported beside the labels, never an input of one
    s = out["strategic"]
    assert "REPORTED IN FULL" in s["state"] and "T0 HWM-ORACLE" in s["tier_stamp"] and set(s["not_computed"]) >= {"STRATEGIC.intent_6_30s", "STRATEGIC.derived_goals", "STRATEGIC.detected_info", "STRATEGIC.S_CKPT"}
    assert s["per_arm"]["IDM_s0"]["cells"]["STRATEGIC|full6s|s_adh"]["vs_logged"]["delta"] == -0.30
    assert s["per_arm"]["DIV_s0"]["state"].startswith("not in this arm's read")        # an arm without a block says so
    md = V.md(out)
    assert "PAUSED" not in md and "STRATEGIC: REPORTED IN FULL" in md and "STRATEGIC.S_CKPT" in md and "IDM_s0 STRATEGIC|full6s|s_adh vs_logged" in md


def _write_arm_dir(d, strategic):
    import json
    gain = lambda p: {"point": p, "lo": p - 0.02, "hi": p + 0.02}
    ia3 = {"1|10": {"gain_lon": gain(0.3), "gain_lat": gain(-0.03), "cos": 0.1, "n": 5}}
    row = lambda fam, band, met, a, b: dict(kind="contrast", family=fam, band=band, metric=met, a=a, b=b, delta=0.1, lo=0.0, hi=0.2, label="none", n_windows=994, n_episodes=71)
    rows = [row("LONGITUDINAL", "0-2s", "acc_mae", "A", "B"), row("LONGITUDINAL", "0-2s", "acc_mae", "A", "HOLD"), row("LONGITUDINAL", "0-2s", "acc_mae", "B", "HOLD")]
    if strategic:
        rows += [row("STRATEGIC", "full6s", "s_adh", "A", "logged"), row("STRATEGIC", "full6s", "s_adh", "B", "logged")]
    rep = {"rows": rows, "replicate": [], "n_windows": 994, "n_episodes": 71, "tactical_kappa": {}}
    rep["strategic"] = {"state": "REPORTED IN FULL (test)", "arms": {}} if strategic else OLD_PAUSE
    os.makedirs(d, exist_ok=True)
    json.dump({"zero_act": {}, "ckpt_step": 1, "argv_sha256": "x", "adapter_sha256": "y", "beside": {"ia3": ia3}, "passes": {"1": {"skill": 0.0}}}, open(f"{d}/ia_read.json", "w"))
    for n in ("contrast_hwm.json", "contrast_flat.json"):
        json.dump(rep, open(f"{d}/{n}", "w"))
    json.dump({}, open(f"{d}/c6.json", "w"))


def _assemble(d, out):
    import runpy
    old = sys.argv
    sys.argv = ["assemble_arm.py", d, "IDM_s0", "0", out]
    try:
        runpy.run_path(os.path.join(HERE_V, "assemble_arm.py"), run_name="__main__")
    finally:
        sys.argv = old
    import json
    return json.load(open(out, encoding="utf-8"))


def test_assemble_arm_carries_the_computed_block_or_says_there_is_none(tmp_path):
    _write_arm_dir(str(tmp_path / "new"), True)
    new = _assemble(str(tmp_path / "new"), str(tmp_path / "new.json"))
    assert new["strategic"]["state"] == "REPORTED IN FULL (test)" and new["hwm"]["cells"]["STRATEGIC|full6s|s_adh"]["vs_logged"]["delta"] == 0.1
    assert "r35_vs_logged" in new["hwm"]["cells"]["STRATEGIC|full6s|s_adh"]
    _write_arm_dir(str(tmp_path / "old"), False)
    old = _assemble(str(tmp_path / "old"), str(tmp_path / "old.json"))
    assert old["strategic"]["state"].startswith("NOT COMPUTED IN THIS READ") and "STRATEGIC.S_CKPT" in old["strategic"]["not_computed"]
    assert "PAUSED" not in str(old["strategic"])                              # the old report carried the pause string; the assembled arm must not copy it


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("PASS", k)
    print(n, "passed")
