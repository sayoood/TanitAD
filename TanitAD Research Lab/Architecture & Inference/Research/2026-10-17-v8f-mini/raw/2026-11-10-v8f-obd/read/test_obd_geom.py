"""Geometry module test (needs torch and the OTBAND export; CPU): obd_geom.py runs read_otband.py's OWN functions, extracted by ast, not copies.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_obd_geom.py"""
import ast
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, HERE)
G = pytest.importorskip("obd_geom")
import obd_read as R  # noqa: E402


def test_extraction_finds_exactly_the_expected_nodes():
    src = open(G.SRC_PATH, encoding="utf-8").read()
    keep, found = G.extract_nodes(src)
    assert sorted(found) == sorted(["sha", "join", "labels", "arm_arrays", "ART", "LEAD", "LEAD_SHA", "RAW", "blk", "idx", "clip_of", "assert:lead block sha mismatch", "assert:duplicate lead rows"]), found
    # source order is kept (ART before LEAD, blk before idx before clip_of)
    assert found.index("ART") < found.index("LEAD") < found.index("blk") < found.index("idx") < found.index("clip_of")
    names = [n.name for n in keep if isinstance(n, ast.FunctionDef)]
    assert names == ["sha", "join", "labels", "arm_arrays"]


def test_metrics_and_bands_equal_the_cards_cells():
    assert tuple(G.METRICS) == R.METRICS and tuple(G.BANDS) == R.BANDS
    assert G.BANDS[R.OP] == (0, 10) and G.BANDS[R.TAC] == (10, 30)
    assert set(G.BL.FAMILY) <= set(R.FAMILY)


def test_estimator_is_the_trees_ci():
    assert os.path.abspath(G.CI.__file__).replace("\\", "/").startswith("F:/TanitAD-scratch/otband_read/tree")
    assert R.load_ci().__file__.replace("\\", "/").endswith("taniteval/taniteval/ci.py")
