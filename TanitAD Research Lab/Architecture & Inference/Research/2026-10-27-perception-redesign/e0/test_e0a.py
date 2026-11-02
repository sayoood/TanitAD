"""Known-answer tests for e0a_gap.py (tier 2).  Synthetic packs with literal AP: a perfect detector reads AP = 1.0, a detector whose only slot is 6 m from the GT reads AP = 0.0;
the gap of a population against ITSELF is exactly 0.0."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0a_gap as A  # noqa: E402


def _pack(slot_xy, logit=8.0, gt_xy=(10.0, 0.0)):
    n = 300
    xy = np.full((n, 2), 90.0, np.float32)
    xy[0] = slot_xy
    lg = np.full(n, -12.0, np.float32)
    lg[0] = logit
    return {"ep": 0, "logit": lg, "xy": xy, "cls": np.zeros(n, np.int16), "cls_corr": np.zeros(n, np.int16), "matched": np.zeros(n, bool), "exempt": np.zeros(n, bool),
            "pair_err": np.zeros(0), "pair_size_err": np.zeros(0), "pair_z_err": np.zeros(0), "gt_xy": np.array([gt_xy], np.float32), "gt_cls": np.zeros(1, np.int16),
            "pos": np.ones(1, bool), "ign": np.zeros(1, bool)}


def test_perfect_detector_ap_is_one():
    w = [A.window_rows(_pack((10.0, 0.0)))]
    assert A.ap_cells(w, [0], "all") == 1.0


def test_detector_6m_off_ap_is_zero():
    w = [A.window_rows(_pack((16.0, 0.0)))]
    assert A.ap_cells(w, [0], "all") == 0.0


def test_gap_of_a_population_against_itself_is_exactly_zero():
    w = [A.window_rows(_pack((10.0, 0.0))), A.window_rows(_pack((16.0, 0.0)))]
    assert A.ap_cells(w, [0, 1], "all") - A.ap_cells(w, [0, 1], "all") == 0.0
    assert A.ap_cells(w, [0, 1], "all") == 0.5          # one of two GT found at top confidence, the other missed -> AP 0.5 (literal)


def test_equal_count_subset_is_exact_and_seed_stable():
    by = {e: [(ws, 100 * e + ws) for ws in range(34)] for e in range(5)}
    s = A.equal_count_subset(by, 30, 8, 0)
    assert len(s) == 30 and s == A.equal_count_subset(by, 30, 8, 0) and len(set(s)) == 30
