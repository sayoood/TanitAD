"""Pin E-DEPLOY-O2PRE-E2E's three arms, including the one that MUST fail.

⛔ The negative control is the reason the positive result is evidence: without it
a PASS cannot be distinguished from "this torch/onnx version exports adaptive
pooling anyway".

    pytest -q test_o2pre_e2e.py
"""
import io
import json
import os

import pytest

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "o2pre_e2e.json")


@pytest.fixture(scope="module")
def r():
    if not os.path.exists(RAW):
        pytest.skip(f"artifact not present: {RAW}")
    return json.load(io.open(RAW, encoding="utf-8"))


def test_all_three_arms_matched_their_pre_declared_outcome(r):
    assert r["all_pass"] is True, r["arms"]


def test_the_deployed_geometry_exports_with_no_pooling_op(r):
    a = r["arms"]["deployed_176x624_NONTILING"]
    assert a["token_grid"] == [11, 39] and a["n_tokens"] == 429
    assert a["tiles_exactly"] is False, "must take the repaired non-tiling route"
    assert a["exported"] is True
    assert a["has_pooling_op"] is False
    assert a["has_matmul_or_gemm"] is True
    assert a["ort_vs_eager_max_rel_err"] < 1e-5, a


def test_the_tiling_control_still_exports(r):
    """The repair must not disturb the path every v7 arm trains at."""
    a = r["arms"]["training_256x640_TILING_control"]
    assert a["token_grid"] == [16, 40] and a["tiles_exactly"] is True
    assert a["exported"] is True
    assert a["ort_vs_eager_max_rel_err"] < 1e-5, a


def test_the_negative_control_FAILS_with_the_original_error(r):
    """⛔ If this ever starts exporting, the test has stopped being informative."""
    a = r["arms"]["negative_control_176x624_PRE_REPAIR_OP"]
    assert a["exported"] is False
    assert "adaptive_avg_pool2d" in a["export_error"]
    assert "not factor of input size" in a["export_error"]
