"""Tests for n4_repair_trace.py. Run with the navsim venv python:  python -m pytest -q test_n4_repair_trace.py  (CPU, ~1 min; needs F: run tree + metric cache).
Controls: (1) 3 REPAIR-D-repaired navtest tokens reproduce the shipped applied plan bit for bit; (2) an unrepaired token reproduces applied False;
(3) DELIBERATE REGRESSION: margin forced to -0.05 m (builder mutation arm `margin_neg`) must CHANGE the control (a no-op trace would not notice)."""
import json
import os
import sys

# BLAS thread env MUST be set before numpy is imported anywhere: with default threads the iterative label solve drifts ~3e-6 from the shipped plan
# (measured here, first run of this test); the builder ran with 1 thread (n4_nt_chain.env).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n4_repair_trace as T  # noqa: E402

CONTROL = ["0a08850db6e35ab7", "cc54b0f77c84506d", "cfbf3f55398f5fc6"]       # REPAIR-D iters 1, 3, 5
UNREPAIRED = "12b32c24d2cf5309"
SHIPPED = np.load(T.OUT_BUILD + "/n4_plans_N4D.npz", allow_pickle=False)
SIX = {str(t): i for i, t in enumerate(SHIPPED["tokens"])}


def _run(tokens, mutation, tmp_path, mode="control"):
    out = str(tmp_path / f"{mutation}_{mode}.jsonl")
    toks, logs, plans, ix, rolls = T.load_inputs()
    T.run_tokens({t: logs[ix[t]] for t in tokens}, mode, out, mutation)
    return {json.loads(l)["token"]: json.loads(l) for l in open(out, encoding="utf-8")}


def test_failed_clauses_unit():
    b = {"nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0}
    assert T.failed_clauses({"dac": 1.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0}, b) == []
    assert T.failed_clauses({"dac": 0.0, "nc": 0.5, "ddc": 1.0, "ttc": 0.0, "c": 0.0}, b) == ["dac", "nc", "ttc", "c"]


def test_control_bit_for_bit(tmp_path):
    rows = _run(CONTROL, "none", tmp_path)
    for t in CONTROL:
        d = rows[t]["runs"]["N4D"]
        assert d["applied"]
        assert float(np.abs(np.array(d["plan"]) - SHIPPED["plans20"][SIX[t]]).max()) == 0.0
        assert d["n_label_builds"] == d["applied_param"]["iters"] + 1


def test_unrepaired_reproduces(tmp_path):
    rows = _run([UNREPAIRED], "none", tmp_path)
    d = rows[UNREPAIRED]["runs"]["N4D"]
    assert d["applied"] is False and len(d["iterations"]) == 6


def test_deliberate_regression_margin_negative_changes_control(tmp_path):
    rows = _run([CONTROL[1]], "margin_neg", tmp_path)
    d = rows[CONTROL[1]]["runs"]["N4D"]
    same = d["applied"] and float(np.abs(np.array(d["plan"]) - SHIPPED["plans20"][SIX[CONTROL[1]]]).max()) == 0.0
    assert not same, "a negative margin must change the repaired plan or make it fail; the trace did not notice"
