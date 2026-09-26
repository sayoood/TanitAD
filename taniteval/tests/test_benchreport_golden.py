"""W5 golden-file test: the report prints E2's OWN numbers, and a mutated report goes RED.

The reference is ``…/2026-09-19-navsim-refcv4b-bridge/RESULT.md`` — written by E2, not by this renderer —
so the literals below are an INDEPENDENTLY AUTHORED reference, not an expression over the code under
test. Each is asserted twice: it must appear in E2's RESULT.md (the control — if E2's document ever
changes, this test says so instead of quietly following it), and it must be what the HTML prints for
that exact summary.json pointer.

Mutations that must go RED (both are faults the verifier exists to catch):
  M1  a swapped arm label in the renderer (A1 <-> STOP)
  M2  a value read from the wrong arm
plus M3: a per-token CSV cell changed by 1e-9, which the ADAPTER's cross-check against E2's own
``scores_summary.json`` must refuse.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from taniteval.benchreport import adapt, render, verify

REPO = Path(__file__).resolve().parents[2]
E2_RAW = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw"
E2_RESULT = E2_RAW.parent / "RESULT.md"
E1_RESULT = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/RESULT.md"
PUBLISHED = REPO / "products/P7-TanitEval/benchmarks/published_results.json"

pytestmark = pytest.mark.skipif(not (E2_RAW / "scores_summary.json").is_file(),
                                reason=f"E2's banked outputs are not in this checkout ({E2_RAW})")

# (summary.json pointer, format, the literal E2's RESULT.md prints)
GOLDEN = [
    ("arms/A1_ego_cmd/statistics/S2_EPDMS_u/value", "0.4670"),
    ("arms/A2_vision_pure/statistics/S2_EPDMS_u/value", "0.5211"),
    ("arms/A3_ego_nocmd/statistics/S2_EPDMS_u/value", "0.4546"),
    ("arms/A4_blind_ego_cmd/statistics/S2_EPDMS_u/value", "0.0950"),
    ("arms/CV/statistics/S2_EPDMS_u/value", "0.3971"),
    ("arms/ECHO/statistics/S2_EPDMS_u/value", "0.4287"),
    ("arms/STOP/statistics/S2_EPDMS_u/value", "0.5212"),
    ("arms/CV/headline/value", "0.1854"),
    ("arms/CV/headline/x100", "18.5356"),        # the protocol's published unit (E1 C8 = the HF warmup LB)
    ("arms/ECHO/headline/value", "0.2248"),
    ("arms/STOP/headline/value", "0.3009"),
    ("arms/A4_blind_ego_cmd/headline/value", "0.0000"),
]
GOLDEN_WTL = [("arms/A1_ego_cmd/paired/CV", (51, 76, 77)), ("arms/A1_ego_cmd/paired/STOP", (108, 40, 56))]
# E2 RESULT.md §3: A1 - CV by t0 speed band, n and the scene-mean delta
GOLDEN_BANDS = [("v_lt1", 36, "+0.031"), ("v_1_4", 64, "+0.100"), ("v_4_8", 59, "+0.151"),
                ("v_ge8", 45, "-0.052")]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2") / "run"
    adapt.build_e2_run(d)
    return d


@pytest.fixture(scope="module")
def rendered(run_dir):
    res = render.render_report(run_dir, published_path=PUBLISHED if PUBLISHED.is_file() else None)
    return Path(res["index"]).read_text(encoding="utf-8")


def shown(html: str, pointer: str) -> list:
    """Every visible text rendered for this summary.json pointer (independent of renderer + verifier)."""
    return re.findall(r'data-k="' + re.escape(pointer) + r'"[^>]*>([^<]+)<', html)


def test_adapter_output_validates_against_W1_schema_v1(run_dir):
    from taniteval.bench.schema_check import load_schema, validate
    for name in ("summary", "bench_run"):
        doc = json.loads((run_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert validate(doc, load_schema(name)) == [], f"{name}.json violates W1 schema v1"
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert s["schema"] == "taniteval.bench.summary/1"
    assert {"STOP", "CV"} <= set(s["floors"]) <= set(s["arms"])          # the mandatory floors ARE arms
    for arm, a in s["arms"].items():                                     # every arm paired against every floor
        assert set(s["floors"]) <= set(a["paired"]), arm


def test_every_golden_number_is_in_E2s_own_result_document():
    """The control: these literals are E2's, not ours."""
    doc = E2_RESULT.read_text(encoding="utf-8")
    doc += E1_RESULT.read_text(encoding="utf-8") if E1_RESULT.is_file() else ""
    for _k, lit in GOLDEN:
        assert lit in doc, f"{lit} is not in E1/E2's RESULT.md — the golden reference moved"
    assert "51 / 76 / 77" in doc and "108 / 40 / 56" in doc
    for _b, n, d in GOLDEN_BANDS:
        assert f"(n {n})" in doc and d.replace("-", "−") in doc or d in doc


def test_report_prints_E2s_numbers(rendered):
    for pointer, lit in GOLDEN:
        texts = shown(rendered, pointer)
        assert texts, f"nothing rendered for {pointer}"
        assert all(t == lit for t in texts), f"{pointer}: rendered {set(texts)}, E2 says {lit}"


def test_report_prints_the_paired_counts(rendered):
    for base, (w, t, lo) in GOLDEN_WTL:
        assert shown(rendered, f"{base}/wins") and set(shown(rendered, f"{base}/wins")) == {str(w)}
        assert set(shown(rendered, f"{base}/ties")) == {str(t)}
        assert set(shown(rendered, f"{base}/losses")) == {str(lo)}


def test_report_prints_the_speed_band_deltas(rendered):
    for band, n, delta in GOLDEN_BANDS:
        assert set(shown(rendered, f"arms/A1_ego_cmd/per_speed_band/bands/{band}/n")) == {str(n)}
        got = shown(rendered, f"arms/A1_ego_cmd/paired/CV/by_speed_band/{band}/scene_mean_delta")
        assert got and all(g == delta for g in got), f"{band}: {got} != {delta}"


def test_hybrid_arms_are_refused_not_printed(rendered, run_dir):
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    hyb = [a for a, v in s["arms"].items() if "value" not in v["headline"]]
    assert hyb, "the E2 fixture must contain HYBRID arms"
    for arm in hyb:
        assert f'data-refusal-k="arms/{arm}/headline"' in rendered
        assert not shown(rendered, f"arms/{arm}/headline/value")
        # and the devkit's hybrid number itself is never printed
        v = s["arms"][arm]["statistics"]["official_rows"]["combined"]["score"]
        assert f"{v:.4f}" not in rendered or f"{v:.4f}" == "0.0000"


def test_verification_passes_on_the_unmutated_report(run_dir, rendered):
    res = verify.verify_report(run_dir / "report" / "index.html", run_dir / "summary.json",
                               PUBLISHED if PUBLISHED.is_file() else None)
    assert res["status"] == "PASS", (res["errors"][:5], res["coverage_missing"][:5])
    assert res["n_numbers_checked"] > 500
    assert res["n_refusals_checked"] > 5


def test_M1_swapped_arm_label_goes_RED(tmp_path, run_dir, monkeypatch):
    d = tmp_path / "m1"
    shutil.copytree(run_dir, d)
    orig = render._arm_label
    swap = {"A1_ego_cmd": "STOP", "STOP": "A1_ego_cmd"}
    monkeypatch.setattr(render, "_arm_label", lambda r, a: orig(r, swap.get(a, a)))
    res = render.render_report(d)
    v = verify.verify_report(res["index"], d / "summary.json")
    assert v["status"] == "FAIL"
    assert any("labelled" in e for e in v["errors"]), v["errors"][:4]


def test_M2_value_read_from_the_wrong_arm_goes_RED(tmp_path, run_dir, monkeypatch):
    d = tmp_path / "m2"
    shutil.copytree(run_dir, d)
    orig = render._n

    def wrong(run, val, f, *path):
        if path[:2] == ("arms", "A1_ego_cmd") and "S2_EPDMS_u" in path and path[-1] == "value":
            val = run.arm("STOP")["statistics"]["S2_EPDMS_u"]["value"]
        return orig(run, val, f, *path)

    monkeypatch.setattr(render, "_n", wrong)
    res = render.render_report(d)
    v = verify.verify_report(res["index"], d / "summary.json")
    assert v["status"] == "FAIL"
    assert any("data-v" in e or "shows" in e for e in v["errors"]), v["errors"][:4]


def test_M3_a_changed_csv_cell_is_refused_by_the_adapter_cross_check(tmp_path):
    """The adapter's agreement with E2's independently computed scores_summary.json has teeth."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for p in E2_RAW.iterdir():
        if p.is_file():
            shutil.copyfile(p, raw / p.name)
    csv = raw / "score_A1_ego_cmd.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    s2col = head.index("no_at_fault_collisions_stage_two")     # a STAGE-2 row: what the cross-check covers
    tok = head.index("token")                                  # col 0 is the devkit's unnamed row index
    i = next(i for i, ln in enumerate(lines[1:], 1)
             if not ln.split(",")[tok].startswith("extended_pdm_score") and ln.split(",")[s2col] != "")
    cells = lines[i].split(",")
    cells[-1] = repr(float(cells[-1]) + 1e-9)                  # `score` is the last column
    lines[i] = ",".join(cells)
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(adapt.AdapterError) as e:
        adapt.build_e2_run(tmp_path / "run", e2_raw=raw, copy_plans=False)
    assert "cross-check FAILED" in str(e.value)
