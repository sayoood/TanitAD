"""`run_battery_r7.g0_text_override` (Master Mind ruling 2026-10-04): a coded G0-A6 FAIL proceeds ONLY on a
corrected-judge verdict of the REGISTERED A6 text for the same checkpoint, re-verified on content. No GPU.

    PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" python -m pytest -q test_battery_g0_text_override.py

Every expectation is a literal. The deliberate-regression arm is a text file that CLAIMS PASS while the banked
artifact does not support it: the override must refuse it (it re-runs the judge, it does not trust the file).
"""
import copy
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

KEY = "eval_agent_det_ap1_heavy_truck_0_20"        # n_pos 1: low-support, REPORTED under A2's rule
NPOS = "eval_agent_det_npos_heavy_truck_0_20"
KEY_G = "eval_agent_det_ap1_car_0_20"              # n_pos 40: gating, tolerance max(0.02, 2/40) = 0.05
NPOS_G = "eval_agent_det_npos_car_0_20"
ROW = {"eval_lat": 0.5, KEY: 0.03333, NPOS: 1.0, KEY_G: 0.5, NPOS_G: 40.0}
INRUN = {"eval_lat": 0.5, KEY: 0.07143, NPOS: 1.0, KEY_G: 0.5, NPOS_G: 40.0}
M1 = {"eval_lat": 0.6, KEY: 0.03333, NPOS: 1.0, KEY_G: 0.5, NPOS_G: 40.0}     # moves ONE term: eval_lat


def _g0(inrun=INRUN, amendment="A6"):
    return {"step": 50400, "ckpt_md5": "md5x",
            "model": {"state_dict": {"missing": [], "unexpected": []}, "param_breakdown": {"equal": True},
                      "anchor_file_vs_ckpt_buffers": {}, "declared_vs_built": {"mismatches": []}},
            "wrapper_control": {"clause": "PASS"}, "mutations": {"m1": {"row": dict(M1)}},
            "a6": {"cells": {}, "fp32_s0": {"row": dict(ROW)}},
            "by_seed": {str(s): {"row": dict(ROW), "buffers_unchanged": True} for s in range(24)},
            "verdict": {"G0": "FAIL", "amendment": amendment,
                        "terms": {k: {"inrun": v} for k, v in inrun.items()}}}


def _text(tmp_path, **over):
    t = {"step": 50400, "ckpt_md5": "md5x",
         "verdict_A6_text": {"G0": "PASS", "reasons": [],
                             "mutation_detection": {"m1": {"detected": True, "n_terms_out": 1}}},
         "controls_other_verdicts_unchanged": {"as_registered": {"ok": True}, "A2": {"ok": True},
                                               "A5": {"ok": True}},
         "compare_with_readonly_reconstruction": {"agree": True},
         "record_lines": ["G0-A6 as registered (text): PASS, computed by the corrected judge after the coded "
                          "verdict was read; the criterion is unchanged",
                          "G0-A6 as coded: FAIL (judge defect: the A2 low-support tuple omitted A6)"]}
    for k, v in over.items():
        t[k] = v
    p = tmp_path / "g0_A6_text.json"
    p.write_text(json.dumps(t), encoding="utf-8")
    return p


def _args(p, ruling="Master Mind ruling 2026-10-04"):
    return types.SimpleNamespace(g0_text_verdict=str(p), g0_text_ruling=ruling)


@pytest.fixture(scope="module")
def RB():
    import run_battery_r7 as RB
    return RB


def test_valid_text_verdict_is_accepted_and_stamped(RB, tmp_path):
    ov = RB.g0_text_override(_args(_text(tmp_path)), _g0(), "md5x")
    assert ov["G0_A6_text"] == "PASS" and ov["re_verified_by_judge_in_use"] is True
    assert ov["record_lines"][1].startswith("G0-A6 as coded: FAIL")
    assert ov["mutation_detection_text"] == {"m1": 1}
    assert len(ov["text_verdict_sha256"]) == 64


@pytest.mark.parametrize("case", ["no_ruling", "other_ckpt", "coded_gate_A5", "text_fail", "controls_bad",
                                  "no_agree", "m1_undetected", "refused_file"])
def test_refusals(RB, tmp_path, case):
    p = _text(tmp_path)
    g0, md5, a = _g0(), "md5x", _args(p)
    if case == "no_ruling":
        a = _args(p, ruling=None)
    elif case == "other_ckpt":
        md5 = "md5y"
    elif case == "coded_gate_A5":
        g0 = _g0(amendment="A5")
    elif case == "text_fail":
        p = _text(tmp_path, verdict_A6_text={"G0": "FAIL", "reasons": ["x"], "mutation_detection": {}})
        a = _args(p)
    elif case == "controls_bad":
        p = _text(tmp_path, controls_other_verdicts_unchanged={"A5": {"ok": False}})
        a = _args(p)
    elif case == "no_agree":
        p = _text(tmp_path, compare_with_readonly_reconstruction={"agree": False})
        a = _args(p)
    elif case == "m1_undetected":
        p = _text(tmp_path, verdict_A6_text={"G0": "PASS", "reasons": [],
                                             "mutation_detection": {"m1": {"detected": False, "n_terms_out": 0}}})
        a = _args(p)
    elif case == "refused_file":
        p = _text(tmp_path, REFUSED=["x"])
        a = _args(p)
    with pytest.raises(SystemExit):
        RB.g0_text_override(a, g0, md5)


def test_REGRESSION_a_file_claiming_PASS_is_not_trusted(RB, tmp_path):
    """DELIBERATE REGRESSION: the banked artifact has a GATING cell 0.06 off (n = 40, tolerance 0.05) -- the
    corrected judge reads FAIL. A text file that merely CLAIMS PASS must be refused."""
    inrun = {**INRUN, KEY_G: 0.56}
    with pytest.raises(SystemExit):
        RB.g0_text_override(_args(_text(tmp_path)), _g0(inrun=inrun), "md5x")


def test_mutation_count_mismatch_is_refused(RB, tmp_path):
    p = _text(tmp_path, verdict_A6_text={"G0": "PASS", "reasons": [],
                                         "mutation_detection": {"m1": {"detected": True, "n_terms_out": 7}}})
    with pytest.raises(SystemExit):
        RB.g0_text_override(_args(p), _g0(), "md5x")
