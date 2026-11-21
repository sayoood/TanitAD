"""Pins for ship_r8_verdict.py (SHIP-R8 clauses (1)-(7) on RETIME_DEPLOY8, AMENDMENT 2).  LITERAL fixtures; every guard has a mutation that must go RED.
    python -m pytest -q test_ship_r8_verdict.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ship_r8_verdict as SV  # noqa: E402

PIN = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
DEPLOY_SHA = "244f313872004745b91a30104244ec9325b1da11bf633b9f31fb532af7075790"
A8, APS = "RETIME_DEPLOY8", "RETIME_DEPLOY"


def cell(delta, lo, hi, sep=None):
    return {"delta": delta, "lo": lo, "hi": hi, "separated": (lo > 0 or hi < 0) if sep is None else sep}


def arm_rows(e_pose, e_ext):
    return {"mean_ade_m": 0.9, "minus_ha0_ext_pose": e_pose, "minus_ha0_ext": e_ext, "NR_minus_refcv7": {}, "minus_os": {}}


def good_fd8():
    """A PASS fixture: E1 / E3 separated below 0 at both seeds; FZR-NR families PASS; FZR-LAT HOLDS."""
    seeds = {}
    for s, (e1, e3) in {"0": (cell(-0.0267, -0.0378, -0.0150), cell(-0.976, -1.277, -0.701)), "1": (cell(-0.0262, -0.0372, -0.0145), cell(-0.973, -1.266, -0.693))}.items():
        seeds[s] = {"surfaces": {"S2": {"sets": {"compliant": {"arms": {A8: arm_rows(e1, cell(-0.0313, -0.043, -0.02)), "CONST_G05": arm_rows(cell(-0.0308, -0.042, -0.019), None)}}}},
                                 "S6": {"sets": {"compliant": {"arms": {A8: arm_rows(e3, cell(-0.948, -1.247, -0.676)), "CONST_G05": arm_rows(cell(-1.1, -1.38, -0.83), None)}}}}}}
    return {"tree_name": "step30000ema_unc", "seeds": seeds, "n_boot": 2000, "frozen_gains_sha256": PIN, "controls": {"ALL_PASS": True},
            "bars": {f"compliant|{A8}": {"FZR-NR": {"per_family": {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "PASS"}}, "FZR-LAT": {"HOLDS": True, "separated_worse_at_any_seed": []}}}}


def good_tf():
    seeds = {}
    for s in ("0", "1"):
        seeds[s] = {"surfaces": {x: {"sets": {"compliant": {"arms": {"RETIME_TRAINFIT": arm_rows(cell(-0.02, -0.03, -0.01), None)}}}} for x in ("S2", "S6")}}
    return {"controls": {"ALL_PASS": True}, "frozen_gains_sha256": PIN, "seeds": seeds, "estimator": "paired episode-cluster bootstrap B 2000, seed 0"}


def good_fd_ps():
    f = good_fd8()
    for s in f["seeds"].values():
        for surf in s["surfaces"].values():
            surf["sets"]["compliant"]["arms"][APS] = surf["sets"]["compliant"]["arms"].pop(A8)
    f["bars"] = {k.replace(A8, APS): v for k, v in f["bars"].items()}
    return f


def good_tac(arm=A8):
    cells = {f"{s}|{k}|s{sd}": {"cell": cell(0.002, -0.004, 0.008), "separated_worse": False} for s in ("S2", "S6") for k in SV.TAC_KEYS for sd in (0, 1)}
    return {"controls": {"ALL_PASS": True}, "clause5": {"status": "PASS", "arm": arm, "cells": cells}, "surfaces": {}}


GOOD_PR = {"strategic": {"status": "NOT APPLICABLE", "n": 0}}
GOOD_PP = {"tree": "step30000ema_unc", "n_windows": 4754, "max_abs_diff_m_by_slot": [2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6], "verdict": "PASS"}


def ev(tree="step30000ema_unc", fd="good", tf="good", tac="good", pr="good", pp="good"):
    return SV.evaluate(tree, good_fd8() if fd == "good" else fd, good_tf() if tf == "good" else tf, good_tac() if tac == "good" else tac, GOOD_PR if pr == "good" else pr,
                       GOOD_PP if pp == "good" else pp, good_fd_ps(), None, DEPLOY_SHA)


def status(v):
    return [c["status"] for c in v["clauses"]]


def test_all_clauses_pass_on_the_pass_fixture():
    v = ev()
    assert status(v) == ["PASS", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS"]
    assert v["overall"] == "PASS" and v["arm"] == "RETIME_DEPLOY8"
    assert v["orientation_only"] is True and "ORIENTATION ONLY" in v["label"]           # 30k is not the condition's tree


def test_condition_tree_label_is_not_orientation():
    v = ev("step48852ema_unc")
    assert v["orientation_only"] is False and v["label"].startswith("SHIP-R8 VERDICT on the condition's tree step48852ema_unc: PASS")


def test_clause_texts_are_the_register_wording():
    assert SV.CLAUSE_TEXT[5].startswith("(5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds")
    assert "if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS" in SV.CLAUSE_TEXT[5]
    assert SV.CLAUSE_TEXT[4].endswith("and S6 lateral HOLDS (FZR-LAT)")
    assert SV.CLAUSE_TEXT[7].startswith("(7) PRIOR PARITY") and "≤ 1e-3 m at all 8 slots on ≥ 200 eval windows" in SV.CLAUSE_TEXT[7]


# ------------------------------------------------------------------ MUTATIONS the brief names
def test_mutation_sign_flip_of_e1_is_a_fail():
    fd = good_fd8()
    for s in ("0", "1"):
        c = fd["seeds"][s]["surfaces"]["S2"]["sets"]["compliant"]["arms"][A8]["minus_ha0_ext_pose"]
        c["delta"], c["lo"], c["hi"] = -c["delta"], -c["hi"], -c["lo"]       # the sign flip
    v = ev(fd=fd)
    assert status(v)[1] == "FAIL" and v["overall"] == "FAIL"


def test_mutation_sign_flip_of_e3_at_one_seed_is_a_fail():
    fd = good_fd8()
    c = fd["seeds"]["1"]["surfaces"]["S6"]["sets"]["compliant"]["arms"][A8]["minus_ha0_ext_pose"]
    c["delta"], c["lo"], c["hi"] = 0.5, 0.2, 0.8
    v = ev(fd=fd)
    assert status(v)[2] == "FAIL" and v["overall"] == "FAIL"


def test_mutation_missing_tac_file_is_incomplete_not_pass():
    v = ev(tac=None)
    assert status(v)[4] == "INCOMPLETE" and v["overall"] == "INCOMPLETE" and "INCOMPLETE" in v["clauses"][4]["reason"]


# ------------------------------------------------------------------ the scored arm is RETIME_DEPLOY8 and nothing else
def test_a_per_surface_form_never_stands_in_for_the_scored_arm():
    fd = good_fd_ps()                                                              # a file that only carries the per-surface RETIME_DEPLOY arm
    v = ev(fd=fd)
    assert status(v)[1] == "INCOMPLETE" and status(v)[2] == "INCOMPLETE" and status(v)[3] == "INCOMPLETE"
    assert status(ev(tac=good_tac(arm="RETIME_DEPLOY")))[4] == "INCOMPLETE"       # an older per-surface TAC file is not clause 5


# ------------------------------------------------------------------ more clause mechanics
def test_e1_negative_but_not_separated_is_a_fail():
    fd = good_fd8()
    fd["seeds"]["0"]["surfaces"]["S2"]["sets"]["compliant"]["arms"][A8]["minus_ha0_ext_pose"]["hi"] = 0.004      # delta stays < 0, CI now spans 0
    assert status(ev(fd=fd))[1] == "FAIL"


def test_e1_cell_without_interval_is_incomplete():
    fd = good_fd8()
    fd["seeds"]["0"]["surfaces"]["S2"]["sets"]["compliant"]["arms"][A8]["minus_ha0_ext_pose"] = {"status": "REFUSED"}
    assert status(ev(fd=fd))[1] == "INCOMPLETE"


def test_missing_deploy8_forward_is_incomplete_everywhere_it_matters():
    v = ev(fd=None)
    assert status(v)[:4] == ["INCOMPLETE"] * 4 and v["overall"] == "INCOMPLETE"


def test_clause1_controls_and_pins():
    tf = good_tf()
    tf["controls"]["ALL_PASS"] = False
    assert status(ev(tf=tf))[0] == "FAIL"
    assert status(ev(tf=None))[0] == "INCOMPLETE"
    tf2 = good_tf()
    tf2["frozen_gains_sha256"] = "0" * 64                                         # the frozen gains changed: the condition says UNCHANGED
    assert status(ev(tf=tf2))[0] == "FAIL"
    fd = good_fd8()
    fd["controls"]["ALL_PASS"] = False
    assert status(ev(fd=fd))[0] == "FAIL"
    fd = good_fd8()
    fd["n_boot"] = 500
    assert status(ev(fd=fd))[0] == "FAIL"
    fd = good_fd8()
    del fd["seeds"]["1"]
    assert status(ev(fd=fd))[0] == "FAIL"
    fd = good_fd8()
    fd["frozen_gains_sha256"] = "1" * 64                                           # FORWARD_DEPLOY8 records a different gains hash
    assert status(ev(fd=fd))[0] == "FAIL"
    fd = good_fd8()
    del fd["frozen_gains_sha256"]                                                  # records no hash at all: cannot show UNCHANGED
    assert status(ev(fd=fd))[0] == "INCOMPLETE"
    fd = good_fd8()
    fd["frozen_gains_sha256"] = DEPLOY_SHA                                         # the deployed table's own hash is acceptable
    assert status(ev(fd=fd))[0] == "PASS"


def test_clause4_nr_and_lat():
    k = f"compliant|{A8}"
    fd = good_fd8()
    fd["bars"][k]["FZR-NR"]["per_family"]["LONGITUDINAL"] = "FAIL: S2:LON_speed_mae_mps"
    assert status(ev(fd=fd))[3] == "FAIL"
    fd = good_fd8()
    fd["bars"][k]["FZR-LAT"]["HOLDS"] = False
    assert status(ev(fd=fd))[3] == "FAIL"
    fd = good_fd8()
    fd["bars"][k]["FZR-LAT"]["HOLDS"] = None
    assert status(ev(fd=fd))[3] == "INCOMPLETE"
    fd = good_fd8()
    fd["bars"][k]["FZR-NR"]["per_family"]["ADE"] = "NOT EVALUABLE"
    assert status(ev(fd=fd))[3] == "INCOMPLETE"
    fd = good_fd8()
    del fd["bars"][k]["FZR-NR"]["per_family"]["LATERAL"]
    assert status(ev(fd=fd))[3] == "INCOMPLETE"


def test_clause5_accuracy_rule_and_cross_checks():
    t = good_tac()
    t["clause5"]["cells"]["S6|TAC_traj_lon_correct|s1"]["cell"] = cell(-0.05, -0.09, -0.01)     # lower accuracy, separated: WORSE
    assert status(ev(tac=t))[4] == "FAIL"
    t = good_tac()
    t["clause5"]["cells"]["S2|TAC_traj_lat_correct|s0"]["cell"] = cell(0.05, 0.01, 0.09)         # a separated IMPROVEMENT must NOT read as worse
    assert status(ev(tac=t))[4] == "PASS"
    t = good_tac()
    t["controls"]["ALL_PASS"] = False
    assert status(ev(tac=t))[4] == "INCOMPLETE"
    t = good_tac()
    del t["clause5"]["cells"]["S2|TAC_traj_lat_correct|s0"]
    assert status(ev(tac=t))[4] == "INCOMPLETE"
    t = good_tac()
    t["clause5"]["status"] = "FAIL"                                                              # the file's own status disagrees with its cells: not PASS
    assert status(ev(tac=t))[4] == "INCOMPLETE"
    t = good_tac()
    t["clause5"]["status"] = "PASS"
    t["clause5"]["cells"]["S6|TAC_traj_lat_correct|s0"]["cell"] = cell(-0.05, -0.09, -0.01)     # the file says PASS but a cell is worse: the verdict re-derives -> FAIL
    assert status(ev(tac=t))[4] == "FAIL"


def test_clause7_prior_parity():
    assert status(ev(pp=None))[6] == "INCOMPLETE" and ev(pp=None)["overall"] == "INCOMPLETE"
    pp = copy.deepcopy(GOOD_PP)
    pp["max_abs_diff_m_by_slot"][5] = 1.0001e-3                                    # one slot just over 1e-3 m
    assert status(ev(pp=pp))[6] == "FAIL" and ev(pp=pp)["overall"] == "FAIL"
    pp = copy.deepcopy(GOOD_PP)
    pp["max_abs_diff_m_by_slot"][7] = 1e-3                                         # exactly at the tolerance passes (<=)
    assert status(ev(pp=pp))[6] == "PASS"
    pp = copy.deepcopy(GOOD_PP)
    pp["n_windows"] = 199
    assert status(ev(pp=pp))[6] == "INCOMPLETE"                                    # < 200 windows: not shown
    pp = copy.deepcopy(GOOD_PP)
    pp["n_windows"] = 200
    assert status(ev(pp=pp))[6] == "PASS"
    pp = copy.deepcopy(GOOD_PP)
    pp["max_abs_diff_m_by_slot"] = pp["max_abs_diff_m_by_slot"][:7]                # only 7 slots reported: schema not recognised -> INCOMPLETE, never PASS
    assert status(ev(pp=pp))[6] == "INCOMPLETE"
    pp = copy.deepcopy(GOOD_PP)
    pp["verdict"] = "PASS"
    pp["max_abs_diff_m_by_slot"][0] = 0.02                                         # the file claims PASS, the numbers say otherwise: the verdict re-derives
    assert status(ev(pp=pp))[6] == "FAIL"
    nested = {"summary": {"n_windows": 512}, "parity": {"max_abs_by_slot_m": [1e-6] * 8}}
    assert status(ev(pp=nested))[6] == "PASS"                                      # a nested layout is found too


def test_fail_dominates_incomplete_and_clause6():
    fd = good_fd8()
    c = fd["seeds"]["0"]["surfaces"]["S2"]["sets"]["compliant"]["arms"][A8]["minus_ha0_ext_pose"]
    c["delta"], c["lo"], c["hi"] = 0.1, 0.05, 0.2
    v = ev(fd=fd, tac=None)                                                        # clause 2 FAIL + clause 5 INCOMPLETE
    assert v["overall"] == "FAIL"
    assert SV.clause6({"strategic": {"status": "ON", "n": 12}})["status"] == "INCOMPLETE"
    assert SV.clause6(GOOD_PR)["status"] == "N/A" and "UNVERIFIED" in SV.clause6(None)["reason"]
    assert SV.overall([{"status": "PASS"}, {"status": "N/A"}]) == "PASS"


def test_beside_block_carries_the_beside_arms():
    v = ev()
    b = v["beside"]
    assert b["pre_ruling_ha0_ext"]["E1_S2"]["s0"]["delta"] == -0.0313
    assert b["flat_0p5_CONST_G05"]["E1_S2_vs_pose"]["s0"]["delta"] == -0.0308
    assert b["per_surface_RETIME_DEPLOY_amendment1"]["clause2_on_this_arm"] == "PASS"
    assert b["RETIME_TRAINFIT_per_surface"]["clause2_on_this_arm"] == "PASS"


# ------------------------------------------------------------------ end to end through files (tmp root): sign flip -> FAIL, missing TAC -> INCOMPLETE
def _write_root(tmp_path, tree="step30000ema_unc", with_tac=True, with_pp=True):
    for d in ("retime_trainfit", "retime_deploy", "paired_read"):
        (tmp_path / d).mkdir()
    (tmp_path / "retime_trainfit" / f"FORWARD_TRAINFIT_{tree}.json").write_text(json.dumps(good_tf()))
    (tmp_path / "retime_deploy" / f"FORWARD_DEPLOY8_{tree}.json").write_text(json.dumps(good_fd8()))
    (tmp_path / "retime_deploy" / f"FORWARD_DEPLOY_{tree}.json").write_text(json.dumps(good_fd_ps()))
    (tmp_path / "paired_read" / f"PAIRED_READ_{tree}.json").write_text(json.dumps(GOOD_PR))
    if with_tac:
        (tmp_path / "retime_deploy" / f"TAC_{tree}.json").write_text(json.dumps(good_tac()))
    if with_pp:
        (tmp_path / "retime_deploy" / "PRIOR_PARITY.json").write_text(json.dumps(GOOD_PP))


def test_end_to_end_files(tmp_path):
    root, out = tmp_path / "r", tmp_path / "o"
    root.mkdir()
    _write_root(root)
    v = SV.run("step30000ema_unc", root, out)
    chk = json.loads((out / "SHIP_R8_step30000ema_unc.json").read_text())
    assert chk["overall"] == "PASS" and (out / "SHIP_R8_step30000ema_unc.md").read_text().startswith("# SHIP_R8_step30000ema_unc")
    assert v["inputs_sha256"]["tac"] is not None and v["inputs_sha256"]["pp"] is not None
    # missing TAC file -> INCOMPLETE
    root2 = tmp_path / "r2"
    root2.mkdir()
    _write_root(root2, with_tac=False)
    assert SV.run("step30000ema_unc", root2, tmp_path / "o2")["overall"] == "INCOMPLETE"
    # missing PRIOR_PARITY -> INCOMPLETE
    root4 = tmp_path / "r4"
    root4.mkdir()
    _write_root(root4, with_pp=False)
    v4 = SV.run("step30000ema_unc", root4, tmp_path / "o4")
    assert v4["overall"] == "INCOMPLETE" and v4["per_clause"]["7"] == "INCOMPLETE"
    # sign flip written into the FORWARD_DEPLOY8 file -> FAIL
    p = root / "retime_deploy" / "FORWARD_DEPLOY8_step30000ema_unc.json"
    j = json.loads(p.read_text())
    for s in ("0", "1"):
        c = j["seeds"][s]["surfaces"]["S2"]["sets"]["compliant"]["arms"][A8]["minus_ha0_ext_pose"]
        c["delta"], c["lo"], c["hi"] = -c["delta"], -c["hi"], -c["lo"]
    p.write_text(json.dumps(j))
    assert SV.run("step30000ema_unc", root, tmp_path / "o3")["overall"] == "FAIL"


def test_raw_weights_beside_falls_back_to_the_per_surface_file():
    v = SV.evaluate("step30000ema_unc", good_fd8(), good_tf(), good_tac(), GOOD_PR, GOOD_PP, good_fd_ps(), None, DEPLOY_SHA, good_fd_ps())
    r = v["beside"]["raw_weights_tree"]
    assert r is not None and "per-surface" in r["form"] and r["E1_S2_vs_pose"]["s0"]["delta"] == -0.0267
    v2 = SV.evaluate("step30000ema_unc", good_fd8(), good_tf(), good_tac(), GOOD_PR, GOOD_PP, good_fd_ps(), good_fd8(), DEPLOY_SHA, good_fd_ps())
    assert v2["beside"]["raw_weights_tree"]["form"] == "RETIME_DEPLOY8"
    assert v["overall"] == "PASS"                                                  # the beside block never changes the verdict


def sibling_pp(n=4754, worst=2e-6):
    slot = [worst * (i + 1) / 8 for i in range(8)]
    return {"parity": [{"tree": t, "inference_seed": s, "n_windows": n, "model_path_vs_rebuilt": {"per_slot_max_m": list(slot), "max_m": max(slot), "n_windows": n}}
                       for t in ("step30000ema_unc", "step30000_unc") for s in (0, 1)],
            "clause7": {"max_abs_m_all_slots_all_windows": worst, "known_values_ok": True, "READING": "PASS (model-path parity)"}, "source_trace_all_verified": True}


def test_clause7_reads_the_siblings_real_schema():
    c = SV.clause7(sibling_pp())
    assert c["status"] == "PASS" and c["evidence"]["min_windows_per_panel"] == 4754 and len(c["evidence"]["panels"]) == 4
    over = sibling_pp()
    over["parity"][2]["model_path_vs_rebuilt"]["per_slot_max_m"][7] = 1.5e-3                  # ONE panel, ONE slot over the tolerance
    assert SV.clause7(over)["status"] == "FAIL"
    liar = sibling_pp()
    liar["parity"][0]["model_path_vs_rebuilt"]["per_slot_max_m"][0] = 0.5                      # the file's own reading still says PASS: the numbers decide
    assert liar["clause7"]["READING"].startswith("PASS") and SV.clause7(liar)["status"] == "FAIL"
    assert SV.clause7(sibling_pp(n=199))["status"] == "INCOMPLETE"
    bad_ctl = sibling_pp()
    bad_ctl["clause7"]["known_values_ok"] = False                                              # the instrument's known-value controls failed: inadmissible
    assert SV.clause7(bad_ctl)["status"] == "INCOMPLETE"
    bad_tr = sibling_pp()
    bad_tr["source_trace_all_verified"] = False
    assert SV.clause7(bad_tr)["status"] == "INCOMPLETE"
    short = sibling_pp()
    short["parity"][1]["model_path_vs_rebuilt"]["per_slot_max_m"] = short["parity"][1]["model_path_vs_rebuilt"]["per_slot_max_m"][:7]
    assert SV.clause7(short)["status"] == "INCOMPLETE"                                         # 7 slots in one panel: schema not recognised, never a clean reading
    assert SV.clause7({"parity": []})["status"] == "INCOMPLETE"


def test_flat_half_arm_is_taken_from_the_file_that_carries_it():
    fd8 = good_fd8()
    for s in fd8["seeds"].values():
        for surf in s["surfaces"].values():
            del surf["sets"]["compliant"]["arms"]["CONST_G05"]                    # the 8-slot file has no flat-0.5 arm
    v = SV.evaluate("step30000ema_unc", fd8, good_tf(), good_tac(), GOOD_PR, GOOD_PP, good_fd_ps(), None, DEPLOY_SHA)
    assert v["beside"]["flat_0p5_CONST_G05"]["E1_S2_vs_pose"]["s0"]["delta"] == -0.0308
