"""W5 contract tests: what the report does with an INCOMPLETE run — it renders, and it says so.

Three situations the suite really produces:

* **E1** — a run whose mandatory STOP floor was never measured and whose human reference is UNDEFINED
  on a two-stage split (the devkit runner exits 1). Both must be bannered and named, never blank.
* **aggregation failed, per-token scores exist** — MEASURED on navhard CV (2026-09-19 night): the
  devkit wrote every per-token row but no ``extended_pdm_score_*`` summary row. The headline must read
  UNAVAILABLE with that reason while every per-stage / per-log / paired number still renders.
* **cross-protocol** — published rows of another protocol may never reach this run's axis.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from taniteval.benchreport import adapt, contract, render, verify

REPO = Path(__file__).resolve().parents[2]
E1_RAW = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/raw"
E2_RAW = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw"
PUBLISHED = REPO / "products/P7-TanitEval/benchmarks/published_results.json"

pytestmark = pytest.mark.skipif(not (E2_RAW / "scores_summary.json").is_file(),
                                reason="the E1/E2 banked outputs are not in this checkout")


def shown(html: str, pointer: str) -> list:
    return re.findall(r'data-k="' + re.escape(pointer) + r'"[^>]*>([^<]+)<', html)


# ------------------------------------------------------------------ E1: a mandatory floor never ran
@pytest.fixture(scope="module")
def e1_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1") / "run"
    adapt.build_e1_run(d)
    return d


def test_e1_summary_is_schema_valid_even_though_the_run_is_incomplete(e1_run):
    from taniteval.bench.schema_check import load_schema, validate
    for name in ("summary", "bench_run"):
        doc = json.loads((e1_run / f"{name}.json").read_text(encoding="utf-8"))
        assert validate(doc, load_schema(name)) == [], name


def test_missing_mandatory_floor_is_a_critical_issue_and_a_banner(e1_run):
    run = contract.load_run(e1_run)
    crit = [i.what for i in run.issues if i.level == "critical"]
    assert any("STOP" in w and "NOT MEASURED" in w for w in crit), crit
    html = Path(render.render_report(e1_run)["index"]).read_text(encoding="utf-8")
    assert "BLOCKING" in html and "MANDATORY floor STOP" in html


def test_the_undefined_human_reference_is_printed_with_its_reason(e1_run):
    html = Path(render.render_report(e1_run)["index"]).read_text(encoding="utf-8")
    assert 'data-refusal-k="arms/HUMAN/headline"' in html
    assert "two-stage EPDMS UNDEFINED for human_agent" in html          # E1's OWN wording, from its artifact
    assert not shown(html, "arms/HUMAN/headline/value")


def test_e1_report_verifies(e1_run):
    res = render.render_report(e1_run)
    v = verify.verify_report(res["index"], e1_run / "summary.json")
    assert v["status"] == "PASS", (v["errors"][:5], v["coverage_missing"][:5])


# ------------------------------------------ aggregation failed, per-token scores exist (navhard CV)
@pytest.fixture(scope="module")
def agg_failed_run(tmp_path_factory):
    """E2's inputs with every ``extended_pdm_score_*`` summary row removed from one arm — the devkit
    scored every token and then failed to aggregate."""
    base = tmp_path_factory.mktemp("aggfail")
    raw = base / "raw"
    raw.mkdir()
    for p in E2_RAW.iterdir():
        if p.is_file():
            shutil.copyfile(p, raw / p.name)
    csv = raw / "score_CV_official.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    tok = lines[0].split(",").index("token")            # col 0 is the devkit's unnamed row index
    kept = [lines[0]] + [ln for ln in lines[1:] if not ln.split(",")[tok].startswith("extended_pdm_score")]
    assert len(kept) == len(lines) - 3, "expected exactly 3 summary rows to strip"
    csv.write_text("\n".join(kept) + "\n", encoding="utf-8")
    doc = json.loads((raw / "scores_summary.json").read_text(encoding="utf-8"))
    doc["arms"]["CV_official"]["official_summary_rows"] = {}            # the aggregation produced nothing
    (raw / "scores_summary.json").write_text(json.dumps(doc), encoding="utf-8")
    d = base / "run"
    adapt.build_e2_run(d, e2_raw=raw, copy_plans=False)
    return d


def test_aggregation_failure_keeps_every_per_token_number(agg_failed_run):
    s = json.loads((agg_failed_run / "summary.json").read_text(encoding="utf-8"))
    cv = s["arms"]["CV"]
    assert "value" not in cv["headline"] and cv["headline"]["status"] == "UNAVAILABLE"
    assert "did not complete" in cv["headline"]["reason"]
    # …while everything computed from the per-token rows is still there
    assert cv["statistics"]["S2_EPDMS_u"]["value"] > 0
    assert cv["per_stage"]["stage_two"]["submetrics"]["NC"] > 0
    assert len(cv["per_log"]["logs"]) == 7
    assert cv["paired"]["STOP"]["status"] == "OK"
    from taniteval.bench.schema_check import load_schema, validate
    assert validate(s, load_schema("summary")) == []


def test_aggregation_failure_renders_the_refusal_and_still_verifies(agg_failed_run):
    res = render.render_report(agg_failed_run, published_path=PUBLISHED if PUBLISHED.is_file() else None)
    html = Path(res["index"]).read_text(encoding="utf-8")
    assert 'data-refusal-k="arms/CV/headline"' in html
    assert "did not complete" in html
    got = shown(html, "arms/CV/statistics/S2_EPDMS_u/value")            # chart + table twin
    assert got and set(got) == {"0.3971"}
    assert shown(html, "arms/CV/per_stage/stage_two/submetrics/NC")            # the sub-metrics survive
    v = verify.verify_report(res["index"], agg_failed_run / "summary.json",
                             PUBLISHED if PUBLISHED.is_file() else None)
    assert v["status"] == "PASS", (v["errors"][:5], v["coverage_missing"][:5])


# ------------------------------------------------------------------------------- cross-protocol
def test_only_this_protocols_published_rows_are_drawn(tmp_path):
    d = tmp_path / "run"
    adapt.build_e2_run(d, copy_plans=False)
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8")) if PUBLISHED.is_file() else {"results": [], "protocols": {}}
    others = [r["id"] for r in pub.get("results", []) if r.get("protocol") != "EPDMS_v2_warmup_two_stage"]
    assert others, "the published file must carry other protocols for this test to mean anything"
    p = tmp_path / "published.json"
    p.write_text(json.dumps(pub), encoding="utf-8")
    html = Path(render.render_report(d, published_path=p)["index"]).read_text(encoding="utf-8")
    drawn = set(re.findall(r'data-pub="([^"]+)"', html))
    assert not (drawn & set(others)), sorted(drawn & set(others))[:5]
    v = verify.verify_report(d / "report" / "index.html", d / "summary.json", p)
    assert v["status"] == "PASS" and not any("cross-protocol" in e for e in v["errors"])


# ---------------------------------------------- W2: the driving command is a route-level ORACLE
@pytest.fixture(scope="module")
def e2_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2c") / "run"
    adapt.build_e2_run(d, copy_plans=False)
    return d


def test_arms_that_declare_the_command_carry_the_oracle_caveat_BESIDE_the_headline(e2_run):
    """W2 2026-09-19/20: the NavSim command is a route-level oracle; the caveat may not be a footnote."""
    s = json.loads((e2_run / "summary.json").read_text(encoding="utf-8"))
    fed = [a for a, v in s["arms"].items()
           if any("driving_command" in d for d in (v.get("declared_inputs") or []))]
    assert fed, "the E2 fixture must contain command-fed arms"
    html = Path(render.render_report(e2_run)["index"]).read_text(encoding="utf-8")
    assert "ORACLE" in html and "route-following" in html
    head_i, paired_i = html.index('id="headline"'), html.index('id="paired"')
    caveat_i = html.index('data-caveat="navsim.driving_command_is_a_route_oracle"')
    assert head_i < caveat_i < paired_i, "the caveat must sit in the headline section, not at the end"
    for arm in fed:                                   # every fed arm is NAMED in a caveat context
        assert f'data-arm="{arm}"' in html
    assert "oracle cmd" in html                       # and tagged on the chart row itself
    # the STRATEGIC family carries the 'not a route claim' banner
    assert "NOT A ROUTE CLAIM" in html
    for arm, v in s["arms"].items():                  # the caveat is DATA, not only renderer prose
        if arm in fed:
            assert v["caveats"] and v["caveats"][0]["id"] == "navsim.driving_command_is_a_route_oracle"


# ------------------------------------------------------------------- W2: the settled interval rule
def test_every_arm_prints_an_interval_or_the_reason_it_has_none(e2_run):
    s = json.loads((e2_run / "summary.json").read_text(encoding="utf-8"))
    html = Path(render.render_report(e2_run)["index"]).read_text(encoding="utf-8")
    assert "LOG-CLUSTER bootstrap" in html and "log_name" in html
    for arm in s["arms"]:
        assert (f'data-refusal-k="arms/{arm}/interval"' in html
                or shown(html, f"arms/{arm}/interval/lo")), f"{arm}: interval cell is blank"
    assert "run_pdm_score.py:422" in html              # the pre-CSV frame reason, verbatim


def test_an_available_interval_renders_with_its_estimator(tmp_path):
    """A navhard-shaped run (76 log clusters) DOES carry one — it must print with estimator + n."""
    d = tmp_path / "iv"
    adapt.build_e2_run(d, copy_plans=False)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    s["arms"]["CV"]["interval"] = {"status": "OK", "estimator": "log_cluster_bootstrap",
                                   "cluster_unit": "log_name", "lo": 0.1712, "hi": 0.1994,
                                   "n_clusters": 76, "B": 2000}
    (d / "summary.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    html = Path(render.render_report(d)["index"]).read_text(encoding="utf-8")
    assert shown(html, "arms/CV/interval/lo") == ["0.1712"] * len(shown(html, "arms/CV/interval/lo"))
    assert "log_cluster_bootstrap" in html and "clusters log_name" in html and "n 76" in html
    v = verify.verify_report(d / "report" / "index.html", d / "summary.json")
    assert v["status"] == "PASS", v["errors"][:4]


def test_a_published_row_of_a_foreign_protocol_is_caught_by_the_verifier(tmp_path):
    """The cross-protocol check has teeth: relabel a drawn row's protocol and the verifier must FAIL."""
    d = tmp_path / "run2"
    adapt.build_e2_run(d, copy_plans=False)
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    p = tmp_path / "published2.json"
    p.write_text(json.dumps(pub), encoding="utf-8")
    render.render_report(d, published_path=p)
    for r in pub["results"]:                                    # mutate AFTER rendering: the drawn row
        if r.get("protocol") == "EPDMS_v2_warmup_two_stage":    # now claims another protocol
            r["protocol"] = "PDMS_v1_navtest"
    p.write_text(json.dumps(pub), encoding="utf-8")
    v = verify.verify_report(d / "report" / "index.html", d / "summary.json", p)
    assert v["status"] == "FAIL" and any("cross-protocol" in e for e in v["errors"]), v["errors"][:3]
