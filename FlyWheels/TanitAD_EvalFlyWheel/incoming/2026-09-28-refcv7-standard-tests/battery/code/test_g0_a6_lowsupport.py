"""SPEC A6 item 7 ("Unchanged: ... A2's low-support rule ...") on LITERAL inputs -- no GPU, no data.

    PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" python -m pytest -q test_g0_a6_lowsupport.py

⛔ JUDGE DEFECT (G0-50,400 diagnosis, 2026-10-04): `g0_refcv7.judge` applies A2's low-support rule only when
`amend in ("A2", "A5")`, so under amend="A6" a per-class AP on ONE ground-truth object is gated at abs 0.02.
That is the whole of the step-50,400 G0-A6 FAIL (13 cells, n_pos 1-3) and of the step-30,000 POST-HOC A6 FAIL.

The two `xfail(strict=True)` tests below state what the REGISTERED A6 text requires. They are RED against the
code as it stands (the defect, reproduced on literals independent of any checkpoint). When the Master Mind
lands the two-line fix (`raw/AMENDMENT_A7_DRAFT.md` §A7.1) they XPASS, and strict mode turns that into a
failure that says: remove the markers. The control tests (A2 / A5 on the same literals) are green today and
must stay green -- they show the literal case is the one A2 was written for.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# One heavy-truck GT in band 0-20 m (n_pos = 1): AP = 1/rank(TP). In-run TP at rank 14 (AP 1/14 = 0.07143),
# replay at rank 30 (AP 1/30 = 0.03333) -- the step-50,400 cell, written as literals. |dev| 0.0381 > 0.02.
KEY = "eval_agent_det_ap1_heavy_truck_0_20"
NPOS = "eval_agent_det_npos_heavy_truck_0_20"
# a GATING cell (n_pos = 40 >= 30): tolerance max(0.02, 2/40) = 0.05
KEY_G = "eval_agent_det_ap1_car_0_20"
NPOS_G = "eval_agent_det_npos_car_0_20"
FIX = "JUDGE DEFECT: A6 item 7 (A2 low-support rule) not in force under amend='A6' until A7.1 lands"


def _rec(m1_row):
    return {"model": {"state_dict": {"missing": [], "unexpected": []},
                      "param_breakdown": {"equal": True}, "anchor_file_vs_ckpt_buffers": {},
                      "declared_vs_built": {"mismatches": []}},
            "wrapper_control": {"clause": "PASS"},
            "mutations": {"m1": {"row": m1_row}},
            # A6's measured inputs: a numerics arm identical to seed 0 (phi = 0); no threshold term here
            "a6": {"cells": {}, "fp32_s0": {"row": {"eval_lat": 0.5, KEY: 0.03333, KEY_G: 0.5}}}}


def _by_seed(row, n=24):
    return {s: {"row": dict(row), "buffers_unchanged": True} for s in range(n)}


ROW = {"eval_lat": 0.5, KEY: 0.03333, NPOS: 1.0, KEY_G: 0.5, NPOS_G: 40.0}
INRUN = {"eval_lat": 0.5, KEY: 0.07143, NPOS: 1.0, KEY_G: 0.5, NPOS_G: 40.0}
M1 = {"eval_lat": 0.6, KEY: 0.03333, NPOS: 1.0, KEY_G: 0.5, NPOS_G: 40.0}     # M1 moves lat 20 %


def test_control_A2_and_A5_report_the_low_support_cell():
    import g0_refcv7 as G
    for amend in ("A2", "A5"):
        v = G.judge(INRUN, _by_seed(ROW, 8 if amend == "A2" else 24), _rec(M1), amend=amend)
        t = v["terms"][KEY]
        assert t["cls"] == "DETECTION_LOWSUPPORT" and t["verdict"] == "REPORTED" and t["support_n"] == 1.0
        assert v["G0"] == "PASS", (amend, v["reasons"])
        assert v["mutation_detection"]["m1"]["detected"] is True


def test_control_as_registered_gates_it_at_abs_002():
    """The registered §2 table (no A2) DOES gate it -- that verdict is reported as registered, always."""
    import g0_refcv7 as G
    v = G.judge(INRUN, _by_seed(ROW, 8), _rec(M1))
    assert v["terms"][KEY]["verdict"] == "OUT" and v["G0"] == "FAIL"


def test_A6_reports_the_low_support_cell_as_A6_item_7_requires():
    import g0_refcv7 as G
    v = G.judge(INRUN, _by_seed(ROW), _rec(M1), amend="A6")
    t = v["terms"][KEY]
    assert t["cls"] == "DETECTION_LOWSUPPORT" and t["verdict"] == "REPORTED"
    assert v["G0"] == "PASS", v["reasons"]


def test_A6_gates_an_n_ge_30_cell_at_max_002_2_over_n():
    """The inherited rule has BOTH halves: n >= 30 -> tolerance max(0.02, 2/n). 0.04 off on n = 40 is inside
    0.05 (OK under A2/A5/A6-as-written) and outside the registered 0.02 (OUT as coded)."""
    import g0_refcv7 as G
    inrun = {**INRUN, KEY_G: 0.54}
    v5 = G.judge(inrun, _by_seed(ROW), _rec(M1), amend="A5")
    assert v5["terms"][KEY_G]["verdict"] == "OK"            # control: A5 applies max(0.02, 2/n)
    v6 = G.judge(inrun, _by_seed(ROW), _rec(M1), amend="A6")
    assert v6["terms"][KEY_G]["verdict"] == "OK" and v6["terms"][KEY_G]["support_n"] == 40.0


def test_power_is_kept_a_real_gating_deviation_still_fails_under_A5():
    """Deliberate-regression arm for the RULE (not the code): 0.06 off on an n = 40 cell is beyond
    max(0.02, 0.05) and must FAIL -- the low-support rule never licenses a gating cell."""
    import g0_refcv7 as G
    inrun = {**INRUN, KEY_G: 0.56}
    v = G.judge(inrun, _by_seed(ROW), _rec(M1), amend="A5")
    assert v["terms"][KEY_G]["verdict"] == "OUT" and v["G0"] == "FAIL"
