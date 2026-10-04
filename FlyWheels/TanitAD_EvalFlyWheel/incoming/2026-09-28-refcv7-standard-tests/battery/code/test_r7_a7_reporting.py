"""The battery's reporting of SPEC A7: `run_battery_r7.a7_stage_summary` (G0 artifact -> summary stage) and
`result_r7.main` (summary -> RESULT markdown). The registered reporting order is
as registered -> A2 -> A5 -> A6 -> A7; when A7 is THE GATE its line comes last, and a pre-A7 gate keeps its old order.
No GPU, no data.

    REFCV6_REPO=D:/Projects/TanitAD PYTHONPATH=D:/Projects/TanitAD/stack python -m pytest -q test_r7_a7_reporting.py

Every expectation is a literal. The regression arm applies the order check to a text built the OLD way (the gate line
before A5 / A6) and it must fail.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

G0_ART = {
    "verdict_A7": {"G0": "PASS", "reasons": [], "registration": {"registered": True},
                   "a7_lowsupport": {"status": "PASS", "n_members": 238, "N_in": 13, "N_num": 20, "bound": 45,
                                     "why": None, "members": ["not copied"]},
                   "m5": {"status": "UNDETECTED", "detected": False, "N_M5": 0, "N_num": 20, "bound": 45,
                          "blind_spot": "a rare-class index swap is invisible to G0", "restoration_bit_exact": True,
                          "n_gating_detection_out": 0}},
    "a7": {"packs": {"status": "ALL BANKED"},
           "seed_group": {"VERDICT": "B", "status": "COMPUTED", "test_full": {"F": 6.606, "p_one_sided": 0.00876}},
           "seed_draw_correlation": {"status": "COMPUTED", "pairs": [
               {"earlier_step": 30000, "n_cells": 192, "r_cell": 0.866, "extra": "dropped"}]}}}


def test_a7_stage_summary_copies_the_artifact_and_nothing_else():
    import run_battery_r7 as R
    s = R.a7_stage_summary(G0_ART)
    assert s == {"G0_A7": "PASS", "reasons_A7": [], "a7_registration": {"registered": True},
                 "a7_lowsupport": {"status": "PASS", "n_members": 238, "N_in": 13, "N_num": 20, "bound": 45,
                                   "why": None},
                 "a7_m5": {"status": "UNDETECTED", "detected": False, "N_M5": 0, "N_num": 20, "bound": 45,
                           "blind_spot": "a rare-class index swap is invisible to G0", "restoration_bit_exact": True,
                           "n_gating_detection_out": 0},
                 "a7_reports": {"packs": "ALL BANKED", "seed_group": "B", "seed_group_F": 6.606,
                                "seed_group_p": 0.00876,
                                "seed_draw_correlation": [{"earlier_step": 30000, "n_cells": 192, "r_cell": 0.866}]}}


def test_a7_stage_summary_says_when_no_earlier_G0_was_found_instead_of_an_empty_list():
    import run_battery_r7 as R
    s = R.a7_stage_summary({"a7": {"seed_draw_correlation": {"status": "NO EARLIER G0 (comparable) FOUND",
                                                             "pairs": []}}})
    assert s["a7_reports"]["seed_draw_correlation"] == "NO EARLIER G0 (comparable) FOUND"


def test_a7_stage_summary_of_a_pre_A7_artifact_is_all_absent_not_an_error():
    import run_battery_r7 as R
    s = R.a7_stage_summary({"verdict": {"G0": "PASS"}})
    assert s["G0_A7"] is None and s["reasons_A7"] == [] and s["a7_registration"] is None
    assert s["a7_lowsupport"]["status"] is None and s["a7_m5"]["status"] is None
    assert s["a7_reports"]["packs"] is None and s["a7_reports"]["seed_draw_correlation"] is None


GATE_A7 = {"G0": "PASS", "amendment": "A7", "G0_as_registered": "FAIL", "reasons_as_registered": ["one term"],
           "G0_A2": "PASS", "G0_A5": "PASS", "G0_A6": "PASS", "G0_A7": "PASS", "n_seeds": 24, "reasons": [],
           "wrapper_clause": "PASS", "a6_registration": {"registered": True},
           "mutation_detection": {"m1": {"n_terms_out": 5, "detected": True},
                                  "m5": {"n_terms_out": 0, "detected": False, "how": "UNDETECTED"}},
           "a7_registration": {"registered": True},
           "a7_lowsupport": {"status": "PASS", "n_members": 238, "N_in": 13, "N_num": 20, "bound": 45},
           "a7_m5": {"status": "UNDETECTED", "N_M5": 0, "bound": 45, "n_gating_detection_out": 0,
                     "blind_spot": "a rare-class index swap is invisible to G0"},
           "a7_reports": {"packs": "ALL BANKED", "seed_group": "B", "seed_group_F": 6.606, "seed_group_p": 0.00876,
                          "seed_draw_correlation": [{"earlier_step": 30000, "n_cells": 192, "r_cell": 0.866}]}}


def _result_text(tmp_path, g0):
    import result_r7
    root, tag = tmp_path / "battery", "t"
    (root / tag).mkdir(parents=True)
    json.dump({"ckpt_md5": "m", "step": 1, "is_milestone": True, "stages": {"g0": g0}},
              open(root / tag / "battery_summary.json", "w", encoding="utf-8"))
    old, sys.argv = sys.argv, ["result_r7.py", "--tag", tag, "--root", str(root)]
    try:
        result_r7.main()
    finally:
        sys.argv = old
    return (root / tag / f"RESULT_SECTION_{tag}.md").read_text(encoding="utf-8")


def _order(text, *needles):
    idx = [text.index(n) for n in needles]
    return idx


def _assert_registered_A2_A5_A6_A7_order(text):
    i = _order(text, "* **G0 as registered:", "* **G0-A2 (seeds 0..7)", "* **G0-A5 (24 seeds", "* **G0-A6 (REGISTERED",
               "* **G0-A7 (THE GATE")
    assert i == sorted(i)


def test_with_A7_as_the_gate_the_report_reads_registered_A2_A5_A6_then_A7(tmp_path):
    text = _result_text(tmp_path, GATE_A7)
    _assert_registered_A2_A5_A6_A7_order(text)
    assert "* **G0-A7 (THE GATE, SPEC A7; 24 inference seeds): PASS**; mutation terms moved: M1 5 (detected), " \
           "M5 0 (NOT detected); wrapper clause PASS; reasons: []" in text
    assert "* G0-A7 detail: A7.2 guard PASS: N_in 13 vs 2·N_num+5 = 45 (N_num 20, 238 low-support members); " \
           "M5 (bus↔heavy_truck class-logit swap, both slot heads; reported, never gating): UNDETECTED " \
           "(N_M5 0 vs 45, gating DETECTION terms moved 0; BLIND SPOT: a rare-class index swap is invisible to G0)" in text
    assert "A7.3 detection packs ALL BANKED; A7.4 seed-group (seeds 0–7 vs 8–23, eval_traj) B " \
           "(F 6.61, one-sided p 0.0088); seed-draw correlation with earlier G0s: " \
           "[{'earlier_step': 30000, 'n_cells': 192, 'r_cell': 0.866}]" in text
    assert "ONE draw of the DDIM noise" in text


def test_a_not_evaluable_M5_is_not_printed_as_not_detected(tmp_path):
    g0 = json.loads(json.dumps(GATE_A7))
    g0["mutation_detection"]["m5"] = {"n_terms_out": None, "detected": None, "how": "NOT EVALUABLE"}
    assert "M5 None (NOT evaluable)" in _result_text(tmp_path, g0)


def test_REGRESSION_the_old_gate_first_order_fails_the_A7_order_check(tmp_path):
    g0 = dict(GATE_A7, amendment="A6", G0_A7=None)             # an A6-gated report: the gate line comes FIRST
    text = _result_text(tmp_path, g0)
    with pytest.raises(ValueError):                              # no A7 line exists at all in it
        _assert_registered_A2_A5_A6_A7_order(text)
    g6 = dict(GATE_A7, amendment="A6")                           # A6 gate + a DRAFT A7 line beside it
    t6 = _result_text(tmp_path / "x", g6)
    assert _order(t6, "* **G0 as registered:", "* **G0-A2 (seeds 0..7)", "* **G0-A6 (THE GATE, SPEC A6",
                  "* **G0-A5 (24 seeds", "* **G0-A6 (REGISTERED") == sorted(
        _order(t6, "* **G0 as registered:", "* **G0-A2 (seeds 0..7)", "* **G0-A6 (THE GATE, SPEC A6",
               "* **G0-A5 (24 seeds", "* **G0-A6 (REGISTERED"))                      # pre-A7 order unchanged
    assert t6.index("* **G0-A6 (THE GATE, SPEC A6") < t6.index("* **G0-A5 (24 seeds")
    assert "* **G0-A7 (REGISTERED; A6 + the DISCRETE-SMALL-N population guard): PASS**; A7.2 guard PASS" in t6
    with pytest.raises(ValueError):
        _assert_registered_A2_A5_A6_A7_order(t6)                 # A6-gated: there is no 'THE GATE' A7 line


def test_a_pre_A7_artifact_reports_no_A7_text_at_all(tmp_path):
    g0 = {k: v for k, v in GATE_A7.items() if not k.startswith(("a7", "G0_A7"))}
    g0["amendment"] = "A6"
    text = _result_text(tmp_path, g0)
    assert "A7" not in text
    assert text.index("* **G0-A6 (THE GATE, SPEC A6") < text.index("* **G0-A5 (24 seeds")
