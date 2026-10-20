"""Pins for ship_r8_verdict_pf.py (SHIP-R8 AMENDMENT 3).  LITERAL expectations; every new guard has a mutation arm that must go RED.

The clause-(1b) mutation arms run the REAL ``feed_parity_pf.check`` on a REAL per-frame roll artifact (the CPU smoke train dump, 4 windows,
2 of which carry a per-frame token different from the clip token) with ONE field replaced: the fed token by the per-clip token, the known bit
flipped, the fed args zeroed.  Each must read FAIL in the checker AND make SHIP-R8 INCOMPLETE in the verdict.
    python -m pytest -q tests/test_ship_r8_verdict_pf.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import feed_parity_pf as FP  # noqa: E402
import ship_r8_verdict_pf as V  # noqa: E402

PF = "0123456789abcdef" * 4                     # a literal stand-in for the registered per-frame table sha256
CLIPFED = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
DEPLOY_SHA = "fedcba9876543210" * 4
ARCH = "C:/Users/Admin/refcv8_eval_archive/battery"
TREE = "step48852ema_pf"
A8 = "RETIME_DEPLOY8"
SMOKE_PF = Path("C:/Users/Admin/refcv8_pf/smoke/train_pf_s0")
SMOKE_CLIP = Path("E:/refcv8_eval_kit/trainfit/step30000ema_train139/dump_s0")
TRAIN_V9 = "E:/refcv8_eval_kit/data/refcv8_v9labels/release_train4572/v9_labels_train.npz"


def cell(d, lo, hi):
    return {"delta": d, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}


def rows(e):
    return {"minus_ha0_ext_pose": e, "minus_ha0_ext": e}


def good_fd8():
    seeds = {s: {"surfaces": {"S2": {"sets": {"compliant": {"arms": {A8: rows(cell(-0.03, -0.04, -0.02))}}}},
                              "S6": {"sets": {"compliant": {"arms": {A8: rows(cell(-0.9, -1.2, -0.7))}}}}}} for s in ("0", "1")}
    return {"seeds": seeds, "n_boot": 2000, "frozen_gains_sha256": PF, "controls": {"ALL_PASS": True},
            "bars": {f"compliant|{A8}": {"FZR-NR": {"per_family": {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "PASS"}}, "FZR-LAT": {"HOLDS": True}}}}


def good_tf():
    seeds = {s: {"surfaces": {x: {"sets": {"compliant": {"arms": {"RETIME_TRAINFIT": rows(cell(-0.02, -0.03, -0.01))}}}} for x in ("S2", "S6")}} for s in ("0", "1")}
    return {"controls": {"ALL_PASS": True}, "frozen_gains_sha256": PF, "seeds": seeds, "estimator": "paired episode-cluster bootstrap B 2000"}


def good_tac():
    cells = {f"{s}|{k}|s{sd}": {"cell": cell(0.002, -0.004, 0.008)} for s in ("S2", "S6") for k in V.B.TAC_KEYS for sd in (0, 1)}
    return {"controls": {"ALL_PASS": True}, "clause5": {"status": "PASS", "arm": A8, "cells": cells}}


GOOD_PR = {"strategic": {"status": "NOT APPLICABLE", "n": 0}}
GOOD_PP = {"parity": [{"tree": TREE, "inference_seed": s, "n_windows": 4754, "model_path_vs_rebuilt": {"per_slot_max_m": [1e-6] * 8}} for s in (0, 1)]}
N_EXP = {"0": 4754, "1": 4754}


def good_parity(s):
    return {"status": "PASS", "pf_dump": f"{ARCH}/{TREE}/dump_s{s}", "clip_dump": f"{ARCH}/step40000ema_unc/dump_s{s}", "subset_ok_smoke_only": False,
            "checks": {k: True for k in V.PARITY_CHECKS},
            "counts": {"n_windows": 4754, "n_windows_only_in_pf": 0, "n_windows_only_in_reference": 0, "n_token_differs_from_clip": 812, "n_args_known": 4100}}


def good_reg(tf=None, gate_t=1000.0, rolls=None, registered=PF):
    return V.registration(registered, PF, {"sha256": PF}, {"registered_sha256": PF, "passed_epoch": gate_t}, tf or good_tf(),
                          rolls if rolls is not None else {"0": 2000.0, "1": 3000.0})


def ev(fd8=None, tf=None, parity=None, reg=None, n_exp=None, tree=TREE):
    V.B.FROZEN_SHA256 = PF
    tf = tf or good_tf()
    return V.evaluate_pf(tree, fd8 or good_fd8(), tf, good_tac(), GOOD_PR, GOOD_PP, None, DEPLOY_SHA,
                         parity if parity is not None else {"0": good_parity(0), "1": good_parity(1)}, n_exp or N_EXP, reg or good_reg(tf))


# ------------------------------------------------------------------------------------------ the pass fixture
def test_all_clauses_pass_on_the_pass_fixture():
    v = ev()
    assert v["per_clause"] == {"1": "PASS", "1b": "PASS", "2": "PASS", "3": "PASS", "4": "PASS", "5": "PASS", "6": "N/A", "7": "PASS"}
    assert v["overall"] == "PASS" and v["orientation_only"] is False
    assert v["label"] == "SHIP-R8 VERDICT (AMENDMENT 3, per-frame feed) on step48852ema_pf: PASS"
    assert v["beside"]["per_clip_verdict_deliberate_regression_arm"]["binding"] is False


def test_base_rule_kept_when_1b_passes():
    f = good_fd8()
    f["seeds"]["1"]["surfaces"]["S2"]["sets"]["compliant"]["arms"][A8]["minus_ha0_ext_pose"] = cell(-0.01, -0.03, 0.01)
    v = ev(fd8=f)
    assert v["per_clause"]["2"] == "FAIL" and v["per_clause"]["1b"] == "PASS" and v["overall"] == "FAIL"


# ------------------------------------------------------------------------------------------ clause (1b), literal
@pytest.mark.parametrize("check", list(V.PARITY_CHECKS))
def test_mutation_any_false_check_makes_1b_fail_and_ship_incomplete(check):
    p = good_parity(0)
    p["checks"][check] = False
    p["status"] = "FAIL"
    v = ev(parity={"0": p, "1": good_parity(1)})
    assert v["per_clause"]["1b"] == "FAIL" and v["overall"] == "INCOMPLETE"


def test_1b_fail_beats_another_fail_overall_is_incomplete():
    p = good_parity(1)
    p["checks"]["T_fed_token_equals_v9_per_frame_token"] = False
    f = good_fd8()
    f["controls"]["ALL_PASS"] = False
    v = ev(fd8=f, parity={"0": good_parity(0), "1": p})
    assert v["per_clause"]["1"] == "FAIL" and v["per_clause"]["1b"] == "FAIL" and v["overall"] == "INCOMPLETE"


@pytest.mark.parametrize("mut,expect", [
    ("absent", "INCOMPLETE"), ("smoke_subset", "INCOMPLETE"), ("wrong_dump", "INCOMPLETE"), ("count_short", "INCOMPLETE"),
    ("only_in_ref", "INCOMPLETE"), ("check_unstated", "INCOMPLETE"), ("status_disagrees", "INCOMPLETE"), ("inconclusive", "INCOMPLETE")])
def test_mutation_1b_cannot_be_shown(mut, expect):
    p = good_parity(0)
    if mut == "absent":
        p = None
    elif mut == "smoke_subset":
        p["subset_ok_smoke_only"] = True
    elif mut == "wrong_dump":
        p["pf_dump"] = f"{ARCH}/step48852ema_unc/dump_s0"                    # parity read on the PER-CLIP tree
    elif mut == "count_short":
        p["counts"]["n_windows"] = 4000
    elif mut == "only_in_ref":
        p["counts"]["n_windows_only_in_reference"] = 3
    elif mut == "check_unstated":
        del p["checks"]["A_r8_nav_nonzero_exactly_where_known_and_equals_independent_scaling"]
    elif mut == "status_disagrees":
        p["status"] = "INCONCLUSIVE"
    elif mut == "inconclusive":
        p = {"status": "INCONCLUSIVE", "pf_dump": f"{ARCH}/{TREE}/dump_s0", "checks": {}, "counts": {}}
    v = ev(parity={"0": p, "1": good_parity(1)})
    assert v["per_clause"]["1b"] == expect and v["overall"] == "INCOMPLETE"


def test_1b_tree_manifest_unreadable_is_incomplete():
    v = ev(n_exp={"0": None, "1": 4754})
    assert v["per_clause"]["1b"] == "INCOMPLETE" and v["overall"] == "INCOMPLETE"


# ------------------------------------------------------------------------------------------ (v) a missing clause
def test_missing_clause_reads_incomplete_never_pass():
    v = ev()
    for drop in V.REQUIRED_IDS:
        cl = [c for c in v["clauses"] if str(c["id"]) != drop]
        assert V.overall_pf(cl) == "INCOMPLETE", drop
    cl = copy.deepcopy(v["clauses"])
    cl[3]["status"] = "MAYBE"
    assert V.overall_pf(cl) == "INCOMPLETE"
    assert V.overall_pf(v["clauses"]) == "PASS"


# ------------------------------------------------------------------------------------------ (ii) registration
def test_mutation_registered_is_the_clip_fed_table_fails_clause1():
    tf = good_tf()
    v = ev(tf=tf, reg=good_reg(tf, registered=CLIPFED))
    assert v["per_clause"]["1"] == "FAIL"


def test_mutation_forward_trainfit_on_another_table_fails_clause1():
    tf = good_tf()
    tf["frozen_gains_sha256"] = CLIPFED
    v = ev(tf=tf, reg=good_reg(tf))
    assert v["per_clause"]["1"] == "FAIL" and v["overall"] == "FAIL"


def test_mutation_roll_before_registration_fails_clause1():
    tf = good_tf()
    v = ev(tf=tf, reg=good_reg(tf, gate_t=2500.0))
    assert v["per_clause"]["1"] == "FAIL"


def test_mutation_no_gate_record_is_incomplete():
    tf = good_tf()
    reg = V.registration(PF, PF, {"sha256": PF}, None, tf, {"0": 2000.0, "1": 3000.0})
    v = ev(tf=tf, reg=reg)
    assert v["per_clause"]["1"] == "INCOMPLETE" and v["overall"] == "INCOMPLETE"


def test_mutation_empty_registered_string_is_never_a_match():
    tf = good_tf()
    reg = V.registration("", "", {"sha256": ""}, {"registered_sha256": "", "passed_epoch": 1000.0}, tf, {"0": 2000.0, "1": 3000.0})
    assert reg["ok"] is False and reg["unknown"]


# ------------------------------------------------------------------------------------------ the copied base is the pinned one
def test_base_copy_is_pinned(tmp_path, monkeypatch):
    assert hashlib.sha256(V.BASE_FILE.read_bytes()).hexdigest() == "7543c585d9cc099942862c7f644c5fcae797272cd9b4dfec93708450224660f2"
    bad = tmp_path / "_ship_r8_verdict_base.py"
    bad.write_bytes(V.BASE_FILE.read_bytes() + b"\n# edited\n")
    monkeypatch.setattr(V, "BASE_FILE", bad)
    with pytest.raises(SystemExit):
        V.load_base()


# ------------------------------------------------------------------------------------------ end to end: files in, json + md out
def test_run_writes_eight_clauses_in_order(tmp_path):
    pkg, arch, hold = tmp_path / "pfpkg", tmp_path / "arch", tmp_path / "hold"
    (pkg / "raw" / "retime_trainfit").mkdir(parents=True)
    dep = pkg / "raw" / "retime_deploy"
    dep.mkdir(parents=True)
    json.dump(good_tf(), open(pkg / "raw" / "retime_trainfit" / f"FORWARD_TRAINFIT_{TREE}.json", "w"))
    json.dump(good_fd8(), open(dep / f"FORWARD_DEPLOY8_{TREE}.json", "w"))
    json.dump(good_tac(), open(dep / f"TAC_{TREE}.json", "w"))
    json.dump(GOOD_PP, open(dep / "PRIOR_PARITY.json", "w"))
    (dep / "FROZEN_GAINS_RETIME_deploy.sha256").write_text(DEPLOY_SHA + "\n")
    fz = hold / "FROZEN_GAINS_RETIME_train_pf.json"
    hold.mkdir()
    fz.write_text('{"x": 1}\n')
    fsha = hashlib.sha256(fz.read_bytes()).hexdigest()
    for f in (pkg / "raw" / "retime_trainfit" / f"FORWARD_TRAINFIT_{TREE}.json",):
        j = json.load(open(f))
        j["frozen_gains_sha256"] = fsha
        json.dump(j, open(f, "w"))
    j = json.load(open(dep / f"FORWARD_DEPLOY8_{TREE}.json"))
    j["frozen_gains_sha256"] = fsha
    json.dump(j, open(dep / f"FORWARD_DEPLOY8_{TREE}.json", "w"))
    json.dump({"sha256": fsha, "beside_clip_fed_table": {"gains": {"S2": [0.9]}}}, open(hold / "FROZEN_GAINS_RETIME_train_pf.provenance.json", "w"))
    reg = hold / "REGISTERED_PF_TABLE.sha256"
    reg.write_text(fsha + "\n")
    gate = hold / "REGISTRATION_GATE.json"
    json.dump({"registered_sha256": fsha, "passed_epoch": 1.0}, open(gate, "w"))
    par = hold / "feed_parity"
    par.mkdir()
    for s in (0, 1):
        d = arch / TREE / f"dump_s{s}"
        d.mkdir(parents=True)
        json.dump({"grid": {"n_windows": 4754}}, open(d / "manifest.json", "w"))
        p = good_parity(s)
        p["pf_dump"] = str(d).replace("\\", "/")
        json.dump(p, open(par / f"FEED_PARITY_{TREE}_s{s}.json", "w"))
    json.dump({"strategic": {"status": "NOT APPLICABLE", "n": 0}}, open(arch / TREE / f"PAIRED_READ_{TREE}.json", "w"))
    out = hold / "ship_r8"
    v = V.run(TREE, pkg, reg, fz, par, gate, out, archive=str(arch).replace("\\", "/"), perclip_file=tmp_path / "absent.json")
    j = json.load(open(out / f"SHIP_R8_{TREE}.json", encoding="utf-8"))
    assert [str(c["id"]) for c in j["clauses"]] == ["1", "1b", "2", "3", "4", "5", "6", "7"]
    assert j["overall"] == "PASS" and j["registered_pf_table_sha256"] == fsha
    md = (out / f"SHIP_R8_{TREE}.md").read_text(encoding="utf-8")
    assert "| (1b) FEED PARITY" in md and "deliberate-regression arm, NEVER binding" in md
    # mutation: the registration file is removed -> clause (1) INCOMPLETE, never PASS
    reg.unlink()
    v2 = V.run(TREE, pkg, reg, fz, par, gate, out, archive=str(arch).replace("\\", "/"), perclip_file=tmp_path / "absent.json")
    assert v2["overall"] == "INCOMPLETE" and v2["per_clause"]["1"] in ("FAIL", "INCOMPLETE")


# ------------------------------------------------------------------------------------------ (1b) on a REAL artifact: the checker itself goes RED
def _copy_dump(tmp_path) -> Path:
    if not SMOKE_PF.exists() or not SMOKE_CLIP.exists() or not Path(TRAIN_V9).exists():
        pytest.fail("the real smoke artifacts this test pins are missing (C:/Users/Admin/refcv8_pf/smoke/train_pf_s0, the clip-fed train dump, the train v9 npz)")
    d = tmp_path / "pf"
    shutil.copytree(SMOKE_PF, d)
    return d


def _mutate(d: Path, fn):
    for f in sorted((d / "decisions").glob("ep*.npz")):
        z = dict(np.load(f, allow_pickle=False))
        fn(z)
        np.savez(f, **z)


def _check(d):
    return FP.check(d, SMOKE_CLIP, TRAIN_V9, label="pytest", subset_ok=True)


def test_real_artifact_unmutated_reads_pass(tmp_path):
    r = _check(_copy_dump(tmp_path))
    assert r["status"] == "PASS" and r["counts"]["n_windows"] == 4 and r["counts"]["n_token_differs_from_clip"] == 2


def test_mutation_real_artifact_fed_token_is_the_per_clip_one_goes_red(tmp_path):
    d = _copy_dump(tmp_path)
    _mutate(d, lambda z: z.__setitem__("nav_cmd", z["nav_cmd_clip"].copy()))
    r = _check(d)
    assert r["status"] == "FAIL" and r["checks"]["T_fed_token_equals_v9_per_frame_token"] is False
    p0 = {**r, "pf_dump": f"{ARCH}/{TREE}/dump_s0", "subset_ok_smoke_only": False}
    p0["counts"] = {**r["counts"], "n_windows": 4754, "n_windows_only_in_pf": 0, "n_windows_only_in_reference": 0}
    v = ev(parity={"0": p0, "1": good_parity(1)})
    assert v["per_clause"]["1b"] == "FAIL" and v["overall"] == "INCOMPLETE"


def test_mutation_real_artifact_known_bit_flipped_goes_red(tmp_path):
    d = _copy_dump(tmp_path)
    _mutate(d, lambda z: z.__setitem__("r8_nav_known", ~z["r8_nav_known"].astype(bool)))
    r = _check(d)
    assert r["status"] == "FAIL" and r["checks"]["K_known_equals_release_args_valid"] is False


def test_mutation_real_artifact_args_zeroed_goes_red(tmp_path):
    d = _copy_dump(tmp_path)
    _mutate(d, lambda z: z.__setitem__("r8_nav_fed", np.zeros_like(z["r8_nav_fed"])))
    r = _check(d)
    assert r["status"] == "FAIL" and r["checks"]["A_r8_nav_nonzero_exactly_where_known_and_equals_independent_scaling"] is False


def test_mutation_real_artifact_clip_token_replaced_by_per_frame_goes_red(tmp_path):
    d = _copy_dump(tmp_path)
    _mutate(d, lambda z: z.__setitem__("nav_cmd_clip", z["nav_cmd"].copy()))
    r = _check(d)
    assert r["status"] == "FAIL" and r["checks"]["C_nav_cmd_clip_equals_clip_fed_reference"] is False


@pytest.mark.parametrize("raw,want", [(b"\xef\xbb\xbf" + PF.encode() + b"\r\n", PF), ((PF + "  FROZEN_GAINS_RETIME_train_pf.json\n").encode(), PF),
                                      (PF.upper().encode() + b"\n", PF), (PF[:63].encode() + b"\n", ""), (b"", "")])
def test_read_registered_forms(tmp_path, raw, want):
    f = tmp_path / "REG.sha256"
    f.write_bytes(raw)
    assert V.read_registered(f) == want
