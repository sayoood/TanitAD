"""Tests of fcread10k.py (card ADD146), synthetic data only."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fcread10k as F  # noqa: E402


def test_planted_controls_pass():
    k2, k3 = F.k2_k3()
    assert k2["PASS"] and k3["PASS"]


def test_reader_row_needs_three_reads():
    R = {(0, x): (10.0, 0.9, 30.0 - 0.4 * x, True) for x in range(10, 30)}
    assert F.reader_row(R, 0, 20)[0] is True
    assert F.reader_row(R, 0, 19)[0] is False                 # f - 10 = 9 has no read
    R[(0, 15)] = (10.0, 0.3, 24.0, True)                     # no reader lead at f - 5
    assert F.reader_row(R, 0, 20)[0] is False


def test_reader_speed_clipped_at_zero():
    R = {(0, x): (1.0, 0.9, 30.0 - 1.0 * x, True) for x in range(0, 30)}     # closing -5 m/s with own speed 1 m/s: lead speed clipped at 0
    ok, c, a = F.reader_row(R, 0, 20)
    assert ok and abs(c + 5.0) < 1e-12 and a == 0.0


def test_label_row_matches_lead_accel():
    sys.path.insert(0, os.path.join(HERE, "..", "build"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("fcw", os.path.join(HERE, "..", "build", "v8f_t1_eval_reader_rule_rzgfhfc.py"))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    lab = F._Lab({(0, 20): (True, 10.0, -1.3), (0, 15): (True, 11.0, 0.4)})
    lv = np.linspace(4.0, 9.0, 40)
    V = {(0, x): float(lv[x]) for x in range(40)}
    ok, c, a = F.label_row(lab, V, 0, 20)
    ref = M.lead_accel(lab, [lv], np.array([0]), np.array([20]))[0][0]
    assert ok and a == ref and c == -1.3


def test_merge_counts_conflicts():
    A = ({(0, 1): (1.0, 0.9, 10.0, True)}, {(0, 1): 1.0})
    B = ({(0, 1): (1.0, 0.9, 10.5, True)}, {(0, 1): 1.0})
    assert F.merge([A, A])[2] == 0
    assert F.merge([A, B])[2] == 1


def test_pearson_spearman():
    x = np.arange(10.0)
    assert abs(F.pearson(x, 2 * x + 1) - 1.0) < 1e-12 and abs(F.spearman(x, x ** 3) - 1.0) < 1e-12
    assert np.isnan(F.pearson(x, np.ones(10)))
