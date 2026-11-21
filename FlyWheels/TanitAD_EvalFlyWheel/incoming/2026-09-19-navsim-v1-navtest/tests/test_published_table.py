"""The published navtest reference numbers live in ONE place, and they are the paper's.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_published_table.py

Three consumers want these numbers: the suite plugin (the leaderboard row), `analyze_navtest.py`
(the SPEC §4 verdicts) and W4's `raw/published_navtest_v1.json`. A second copy of a published
number is how two "independent" tables drift apart, so the plugin's ``PUBLISHED`` is the only
production source — the analysis IMPORTS it — and the literals below (read from the banked PDF,
arXiv 2406.15349v2: Table 1 p. 7, Table 3 p. 9) are the independent pin.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[3]
sys.path[:0] = [str(REPO / "taniteval")]

from taniteval.bench.plugins import navsim_v1 as P                     # noqa: E402

#: verbatim from the banked primary — the independent expectation (never derived from the code)
TABLE_1 = {
    "CV":      {"NC": 68.0, "DAC": 57.8, "TTC": 50.0, "C": 100.0, "EP": 19.4, "PDMS": 20.6},
    "EGO_MLP": {"NC": 93.0, "DAC": 77.3, "TTC": 83.6, "C": 100.0, "EP": 62.8, "PDMS": 65.6},
    "HUMAN":   {"NC": 100.0, "DAC": 100.0, "TTC": 100.0, "C": 99.9, "EP": 87.5, "PDMS": 94.8},
}
TABLE_3_PDMS = {"TransFuser": 83.9, "LTF": 83.5, "Ego Status MLP": 66.4, "Hydra-MDP": 91.3,
                "Constant Velocity": 20.6}
LEADERBOARD_CV = 20.6517          # INHERITED, NAVSIM_PROTOCOL.md §6.1 (HF LB, 2026-08-23)
PAPER_SHA256 = "d3bc66d321cfceccc4f431113dea3f0f481d74d29dd283a33e204a4326f0403c"


def test_plugin_table_is_the_papers_table_1():
    for arm, want in TABLE_1.items():
        got = P.PUBLISHED[arm]
        for k, v in want.items():
            assert got[k] == v, f"{arm}.{k}: {got[k]} != {v}"


def test_plugin_names_the_banked_primary_by_sha256():
    src = P.PUBLISHED["_source"]
    assert src["library_key"] == "2406.15349" and src["sha256"] == PAPER_SHA256
    assert "Table 1" in src["tables"] and "Table 3" in src["tables"]
    assert P.PUBLISHED["_leaderboard_INHERITED"]["CV"] == LEADERBOARD_CV


def test_the_analysis_imports_the_plugins_table_and_keeps_no_copy():
    spec = importlib.util.spec_from_file_location("w3_analyze", PKG / "code" / "analyze_navtest.py")
    A = importlib.util.module_from_spec(spec)
    sys.modules["w3_analyze"] = A
    spec.loader.exec_module(A)
    assert A.PUBLISHED_SOURCE.replace("\\", "/").endswith("bench/plugins/navsim_v1.py")
    assert A.PAPER["CV"] == TABLE_1["CV"] and A.PAPER["HUMAN"] == TABLE_1["HUMAN"]
    assert A.LB["CV"]["PDMS"] == LEADERBOARD_CV
    src = (PKG / "code" / "analyze_navtest.py").read_text(encoding="utf-8")
    assert "20.6517" not in src, "a private copy of a published number reappeared in the analysis"


def test_w4_banked_rows_agree_with_the_plugin():
    p = PKG / "raw" / "published_navtest_v1.json"
    if not p.exists():
        pytest.skip("W4 rows not banked")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = {r["method"]: r for r in d["rows"]}
    assert d["_source_primary"]["sha256"] == PAPER_SHA256
    assert rows["Constant Velocity"]["pdms"] == TABLE_1["CV"]["PDMS"]
    assert rows["Constant Velocity"]["submetrics"] == {k: v for k, v in TABLE_1["CV"].items()
                                                       if k != "PDMS"}
    assert rows["Ego Status MLP"]["pdms"] == TABLE_1["EGO_MLP"]["PDMS"]
    assert rows["Human (privileged)"]["pdms"] == TABLE_1["HUMAN"]["PDMS"]
    assert rows["Hydra-MDP"]["pdms"] == TABLE_3_PDMS["Hydra-MDP"]
    assert rows["Constant Velocity"]["also"]["hf_leaderboard_INHERITED"] == LEADERBOARD_CV
    assert d["_protocol"] == "PDMS_v1_navtest"


def test_verdict_boundaries_are_the_pre_registered_ones_as_amended():
    """SPEC §4 + AMENDMENT A1: both printed-precision readings, REPRODUCED only when both agree."""
    spec = importlib.util.spec_from_file_location("w3_analyze2", PKG / "code" / "analyze_navtest.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    assert A.verdict(20.6, 20.6, 1)["verdict"] == "REPRODUCED"
    assert A.verdict(20.64, 20.6, 1)["verdict"] == "REPRODUCED"      # both readings give 20.6
    # the two conventions DIVERGE here — neither may be assumed, and neither is hidden
    v = A.verdict(20.58, 20.6, 1)
    assert v["verdict"] == "REPRODUCED_UNDER_ROUNDING"
    assert (v["reading_rounded"], v["reading_truncated"]) == (20.6, 20.5)
    v = A.verdict(20.6517, 20.6, 1)
    assert v["verdict"] == "REPRODUCED_UNDER_TRUNCATION"
    assert (v["reading_rounded"], v["reading_truncated"]) == (20.7, 20.6)
    assert A.verdict(21.0, 20.6, 1)["verdict"] == "CLOSE"            # |d| 0.4 <= 0.5
    assert A.verdict(21.2, 20.6, 1)["verdict"] == "NOT REPRODUCED"   # |d| 0.6 > 0.5
    assert A.verdict(20.6517, 20.6517, 4)["verdict"] == "REPRODUCED"
    assert A.verdict(20.6518, 20.6517, 4)["verdict"] == "CLOSE"      # 4 dp is the LB's precision


def test_the_verdict_has_ONE_implementation():
    """`analyze_navtest.verdict` must BE the plugin's `_verdict`, not a second copy of the rule."""
    spec = importlib.util.spec_from_file_location("w3_analyze3", PKG / "code" / "analyze_navtest.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    assert A._VERDICT is not None
    got = A.verdict(20.58, 20.6, 1)
    assert got == P._verdict(20.58, 20.6, 1)
    src = (PKG / "code" / "analyze_navtest.py").read_text(encoding="utf-8")
    assert "REPRODUCED_UNDER" not in src, "the verdict vocabulary was copied into the analysis"


def test_the_convention_probe_refuses_to_assume():
    p = PKG / "raw" / "print_convention_probe.json"
    if not p.exists():
        pytest.skip("probe not run")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["verdict"].startswith("UNSETTLED")
    assert d["internal_test"]["implies"] == "ROUNDING"
    cells = {c["cell"]: c["implies"] for c in d["leaderboard_cells"]}
    assert cells["Tab.3 Constant Velocity PDMS"] == "TRUNCATION"
    assert cells["Tab.3 TransFuser PDMS (3 seeds)"] == "ROUNDING"
    assert d["primary"]["sha256"] == PAPER_SHA256
