"""The report must render W1's REAL run directories — and never confuse a pooled count with a per-stage one.

MEASURED 2026-09-20: the first W1 run crashed the report with ``KeyError: 'wins'`` because on a two-stage
protocol the per-scene win/tie/loss counts live PER STAGE (the stages have different token sets). W1 then
added pooled top-level ``wins``/``ties``/``losses`` plus ``_wtl_scope``, keeping the split at
``paired.<floor>.by_stage.<stage>``. Both layouts — and W5's own fixture layout — must render, and every
count must be printed WITH the scope it was counted over.

Two guards:
* over the real run directories when they exist (they are generated, so this part skips on a clean
  checkout — which is why the synthetic guard below exists and always runs);
* over synthetic summaries in all three shapes, which cannot skip.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from taniteval.benchreport import contract, render, verify

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "taniteval" / "results" / "bench"      # every benchmark, not just navsim_v2


def shown(html: str, pointer: str) -> list:
    return re.findall(r'data-k="' + re.escape(pointer) + r'"[^>]*>([^<]+)<', html)


def _real_runs() -> list:
    """<benchmark>/<split>/<run_id>/summary.json — globbed at COLLECTION time: W1's results dir is live."""
    return sorted(p.parent for p in BENCH.glob("*/*/*/summary.json")) if BENCH.is_dir() else []


@pytest.mark.parametrize("run_dir", _real_runs(), ids=lambda p: p.name)
def test_a_real_W1_run_renders_and_verifies(run_dir, tmp_path):
    """No crash, and the page prints exactly what that run's summary.json holds."""
    out = tmp_path / run_dir.name
    res = render.render_report(run_dir, out_dir=out)
    v = verify.verify_report(res["index"], run_dir / "summary.json")
    assert v["status"] == "PASS", (run_dir.name, v["errors"][:5], v["coverage_missing"][:5])
    html = Path(res["index"]).read_text(encoding="utf-8")
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for arm, a in s["arms"].items():
        for fl, p in (a.get("paired") or {}).items():
            if not isinstance(p, dict) or p.get("status") != "OK":
                continue
            for st, b in (p.get("by_stage") or {}).items():          # per-stage counts are rendered
                if isinstance(b, dict) and isinstance(b.get("wins"), int):
                    assert shown(html, f"arms/{arm}/paired/{fl}/by_stage/{st}/wins") == [str(b["wins"])] * len(
                        shown(html, f"arms/{arm}/paired/{fl}/by_stage/{st}/wins"))
                    assert shown(html, f"arms/{arm}/paired/{fl}/by_stage/{st}/wins")
            if isinstance(p.get("wins"), int):                        # pooled counts carry their scope
                assert shown(html, f"arms/{arm}/paired/{fl}/wins")
                assert p.get("_wtl_scope", "")[:40] in html or "pooled" in html


@pytest.mark.parametrize("run_dir", _real_runs(), ids=lambda p: p.name)
def test_lateral_cross_track_carries_its_qualifier_context_and_alternative(run_dir, tmp_path):
    """⛔ W2 2026-09-20: `cross_mae_m` is a lateral OFFSET at matched time index, not a distance to the
    path — on this very run CV and STOP both read 1.0658 m. The page must render the values (not blanks),
    the qualifier, the along-track context and the projection-based alternative, and NAME the tie."""
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    vals = {}
    for arm, a in s["arms"].items():
        lat = (a.get("families") or {}).get("lateral") or {}
        v = (lat.get("metrics") or {}).get("cross_mae_m", lat.get("cross_mae_m"))
        if isinstance(v, (int, float)):
            vals[arm] = float(v)
    if not vals:
        pytest.skip("this run carries no lateral cross-track value")
    html = Path(render.render_report(run_dir, out_dir=tmp_path / run_dir.name)["index"]).read_text(encoding="utf-8")
    assert "lateral offset at matched time index" in html            # the qualifier, verbatim from W2
    assert "pathgeom_crosstrack_m" in html                           # the alternative is NAMED
    for arm, v in vals.items():                                      # the value itself is printed
        assert f"{v:.4f}" in html or f"{v:.3f}" in html
    if len(set(round(v, 6) for v in vals.values())) < len(vals):     # a tie exists -> it must be named
        assert "cannot separate them" in html
    if s.get("benchmark") in ("navsim_v2", "navsim_v1"):
        assert "STOP" in s["arms"], "the mandatory STOP floor pins the tie case"


def _summary(paired: dict) -> dict:
    """A minimal two-stage summary in W1's shape, with the paired block under test."""
    arm = {"kind": "model", "status": "OK", "declared_inputs": [],
           "headline": {"value": 0.25, "x100": 25.0, "column": "score", "statistic": "EPDMS", "n": 220},
           "per_stage": {"stage_one": {"score": 0.4, "n": 16, "submetrics": {"NC": 0.9}},
                         "stage_two": {"score": 0.3, "n": 204, "submetrics": {"NC": 0.8}}},
           "submetrics": {}, "per_log": {}, "paired": {}, "families":
               {f: {"status": "UNAVAILABLE", "n": 0, "reason": "not run"} for f in
                ("longitudinal", "lateral", "tactical", "strategic")},
           "interval": {"status": "UNAVAILABLE", "reason": "7 log groups < 8", "n": 7}, "files": {}}
    a1, cv, stop = copy.deepcopy(arm), copy.deepcopy(arm), copy.deepcopy(arm)
    cv["kind"] = stop["kind"] = "floor"
    a1["paired"] = {"CV": copy.deepcopy(paired), "STOP": copy.deepcopy(paired)}
    cv["paired"] = {"CV": {"status": "SELF"}, "STOP": copy.deepcopy(paired)}
    stop["paired"] = {"STOP": {"status": "SELF"}, "CV": copy.deepcopy(paired)}
    return {"schema": "taniteval.bench.summary/1", "run_id": "20260920T000000Z-navsim_v2-none-aaaaaa",
            "benchmark": "navsim_v2", "protocol": "EPDMS_v2_warmup_two_stage", "split": "warmup_two_stage",
            "claim_bearing": True, "evidence_class": "MEASURED",
            "stamps": {"tier": "T1-family", "loop": {"stage_one": "OPEN"}, "evidence_class": "MEASURED"},
            "headline_metric": {"name": "EPDMS", "column": "score", "higher_is_better": True,
                                "statistic": "official two-stage"},
            "floors": ["STOP", "CV"], "arms": {"A1": a1, "CV": cv, "STOP": stop}}


POOLED = {"status": "OK", "headline_delta": -0.11, "n_common": 220, "wins": 91, "ties": 38, "losses": 91,
          "_wtl_scope": "per-scene wins/ties/losses pooled over the common tokens of BOTH stages "
                        "(tie band 1e-12); per stage in by_stage.<stage>"}
BY_STAGE = {"stage_one": {"n": 16, "score_mean_delta": -0.117, "wins": 8, "ties": 0, "losses": 8},
            "stage_two": {"n": 204, "score_mean_delta": -0.13, "wins": 83, "ties": 38, "losses": 83}}


def _write(tmp_path, summary) -> Path:
    d = Path(tmp_path) / "run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return d


@pytest.mark.parametrize("shape", ["pooled_only", "by_stage_only", "both"])
def test_every_count_layout_renders_with_its_scope(tmp_path, shape):
    paired = dict(POOLED)
    if shape == "by_stage_only":
        paired = {"status": "OK", "headline_delta": -0.11, "n_common": 220, "by_stage": copy.deepcopy(BY_STAGE)}
    elif shape == "both":
        paired = dict(POOLED, by_stage=copy.deepcopy(BY_STAGE))
    d = _write(tmp_path / shape, _summary(paired))
    res = render.render_report(d)
    html = Path(res["index"]).read_text(encoding="utf-8")
    v = verify.verify_report(res["index"], d / "summary.json")
    assert v["status"] == "PASS", (shape, v["errors"][:4], v["coverage_missing"][:4])
    if shape in ("by_stage_only", "both"):
        assert shown(html, "arms/A1/paired/CV/by_stage/stage_two/wins") == ["83"] * len(
            shown(html, "arms/A1/paired/CV/by_stage/stage_two/wins"))
        assert shown(html, "arms/A1/paired/CV/by_stage/stage_one/wins")
        assert "stage two only" in html and "stage one only" in html      # the scope, beside the count
    if shape in ("pooled_only", "both"):
        assert shown(html, "arms/A1/paired/CV/wins") == ["91"] * len(shown(html, "arms/A1/paired/CV/wins"))
        assert "pooled over the common tokens of BOTH stages" in html      # W1's own _wtl_scope text
    if shape == "both":
        # the two scopes are DIFFERENT quantities and both are on the page, each with its own label
        assert shown(html, "arms/A1/paired/CV/wins") and shown(html, "arms/A1/paired/CV/by_stage/stage_one/wins")


def test_a_paired_block_without_counts_is_refused_not_dropped(tmp_path):
    """W1's FAILED run: paired UNAVAILABLE everywhere — the floor must still appear, with the reason."""
    paired = {"status": "UNAVAILABLE", "reason": "this arm was not scored", "n": 0}
    d = _write(tmp_path, _summary(paired))
    res = render.render_report(d)
    html = Path(res["index"]).read_text(encoding="utf-8")
    assert 'data-refusal-k="arms/A1/paired/CV"' in html and "this arm was not scored" in html
    v = verify.verify_report(res["index"], d / "summary.json")
    assert v["status"] == "PASS", (v["errors"][:4], v["coverage_missing"][:4])


def test_the_normaliser_reads_both_per_stage_layouts(tmp_path):
    """W1 names the stage number `score`; the W5 adapter names it `scene_mean`. Same reader."""
    d = _write(tmp_path, _summary(dict(POOLED)))
    run = contract.load_run(d)
    v, k = run.stage_score("A1", "stage_two")
    assert (v, k) == (0.3, "arms/A1/per_stage/stage_two/score")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    s["arms"]["A1"]["per_stage"]["stage_two"] = {"scene_mean": 0.31, "n": 204, "submetrics": {"NC": 0.8}}
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    run2 = contract.load_run(d)
    assert run2.stage_score("A1", "stage_two") == (0.31, "arms/A1/per_stage/stage_two/scene_mean")


# --------------------------------------------- W2's NEW lateral annotations and refusal shape
LATERAL_W2 = {
    "status": "PARTIAL", "n": 16, "reason": "yaw rate refused on this arm",
    "metrics": {
        "cross_mae_m": 1.0658, "heading_mae_deg": 7.4799, "curvature_mae_1pm": 0.015494,
        # W2's refusal shape for a stationary plan / NaN yaw rate — never a blank and never a zero
        "yaw_rate_mae_degps": {"status": "UNAVAILABLE", "reason": "the plan does not move: every step is "
                               "below min_ds_m, so the yaw rate is undefined (NaN)", "n": 0,
                               "n_steps_total": 128, "min_ds_m": 0.25},
    },
    "_cross_is": "`cross_mae_m` is a lateral offset at matched time index, informative only while the "
                 "along-track error is small; read it with the LONGITUDINAL family, and use "
                 "`headline.pathgeom_crosstrack_m` when a distance-to-path is meant.",
    "_along_mae_m_for_context": 1.2669,
    "_projection_based_alternative": "headline.pathgeom_crosstrack_m (lateral.py::frenet_dense)",
}


def test_W2s_lateral_annotations_and_refusal_shape_render(tmp_path):
    s = _summary(dict(POOLED))
    for arm in ("A1", "CV", "STOP"):
        s["arms"][arm]["families"]["lateral"] = copy.deepcopy(LATERAL_W2)
    s["arms"]["STOP"]["families"]["lateral"]["metrics"]["cross_mae_m"] = 1.0658      # the documented tie
    d = _write(tmp_path, s)
    res = render.render_report(d)
    html = Path(res["index"]).read_text(encoding="utf-8")
    # 1. the run's OWN qualifier wins over the built-in, and sits beside the lateral headline
    assert "lateral offset at matched time index" in html
    lat_i = html.index(">LATERAL<")
    assert html.index("lateral.cross_mae_is_an_offset") > lat_i
    # 2. the along-track context number is rendered from the run's own field
    assert shown(html, "arms/A1/families/lateral/_along_mae_m_for_context") == ["1.2669"] * len(
        shown(html, "arms/A1/families/lateral/_along_mae_m_for_context"))
    # 3. the projection-based alternative is named
    assert "headline.pathgeom_crosstrack_m" in html
    # 4. the refusal-shaped term prints UNAVAILABLE + reason, never a blank or a zero
    assert 'data-refusal-k="arms/A1/families/lateral/metrics/yaw_rate_mae_degps"' in html
    assert "the plan does not move" in html
    assert not shown(html, "arms/A1/families/lateral/metrics/yaw_rate_mae_degps")
    # 5. the tie between a moving arm and the STOP floor is NAMED
    assert "cannot separate them" in html
    v = verify.verify_report(res["index"], d / "summary.json")
    assert v["status"] == "PASS", (v["errors"][:4], v["coverage_missing"][:4])


def test_a_lateral_value_is_never_rendered_as_a_blank(tmp_path):
    """The defect this fixes: W1's flat `families.<fam>.metrics.<k>` layout rendered as '—'."""
    s = _summary(dict(POOLED))
    s["arms"]["A1"]["families"]["lateral"] = {"status": "OK", "n": 16,
                                              "metrics": {"cross_mae_m": 1.0658, "heading_mae_deg": 7.48}}
    d = _write(tmp_path, s)
    html = Path(render.render_report(d)["index"]).read_text(encoding="utf-8")
    assert shown(html, "arms/A1/families/lateral/metrics/cross_mae_m"), "a real value rendered as a blank"
    assert shown(html, "arms/A1/families/lateral/metrics/heading_mae_deg")


# ------------------------------------------------------- W1's DRY_RUN_PASSED, and metric direction
def _dry(tmp_path):
    s = _summary(dict(POOLED))
    for a in s["arms"].values():                       # a dry run scores NOTHING, by design
        a["status"] = "OK"
        a["headline"] = {"status": "UNAVAILABLE", "n": 0,
                         "reason": "--dry-run: the preflight passed and no arm was scored"}
        a.pop("statistics", None)
    d = _write(tmp_path, s)
    (d / "bench_run.json").write_text(json.dumps(
        {"schema": "taniteval.bench.bench_run/1", "run_id": s["run_id"], "utc": "2026-09-20T00:00:00Z",
         "git_head": "UNAVAILABLE", "ckpt": {"path": None, "sha256": None, "registry_key": None},
         "benchmark": "navsim_v2", "protocol": s["protocol"], "status": "DRY_RUN_PASSED",
         "devkit": {"repo": "autonomousvision/navsim", "sha": "UNAVAILABLE", "patches": []},
         "split": {"name": "warmup_two_stage", "n_scenes": 220, "n_logs": 7},
         "arms": [{"name": a, "kind": "floor", "declared_inputs": [], "status": "SKIPPED"} for a in s["arms"]],
         "device": {"requested": "cpu", "used": "none"}, "wall_s": 1.0, "claim_bearing": True,
         "stamps": s["stamps"], "report": {"status": "REPORT_PENDING"}}), encoding="utf-8")
    return d


def test_DRY_RUN_PASSED_is_a_pass_not_a_refusal_or_a_failure(tmp_path):
    """⭐ W1/E1 2026-09-20: a passing dry run used to report REFUSED — a failure word on a success."""
    d = _dry(tmp_path)
    run = contract.load_run(d)
    assert run.is_dry_run
    assert [i.what for i in run.issues if i.level == "critical"] == []      # no BLOCKING on a PASS
    html = Path(render.render_report(d)["index"]).read_text(encoding="utf-8")
    assert "DRY RUN PASSED" in html and "nothing was scored" in html
    assert "NOT MEASURED" not in html and "BLOCKING" not in html
    assert "REFUSED" not in html.split('id="provenance"')[0]                # not in the reading path
    v = verify.verify_report(d / "report" / "index.html", d / "summary.json")
    assert v["status"] == "PASS", (v["errors"][:4], v["coverage_missing"][:4])


def test_a_lower_is_better_protocol_says_so_and_is_never_called_a_fraction(tmp_path):
    """MEASURED 2026-09-20 on the real internal_t1 run: the axis read `ADE (fraction)` — metres, and
    lower is better. A unit guessed from nothing is the `true but wrong for the reader` class."""
    s = _summary(dict(POOLED))
    s["protocol"] = "TanitAD_T1_refc_physicalai"
    s["benchmark"] = "internal_t1"
    s["headline_metric"] = {"name": "ADE", "column": "ade_dense_m", "higher_is_better": False,
                            "statistic": "mean ADE over the dense grid"}
    s["floors"] = ["CV"]
    for a in s["arms"].values():
        a["headline"] = {"value": 2.9098, "column": "ade_dense_m", "statistic": "mean ADE", "n": 9}
    d = _write(tmp_path, s)
    html = Path(render.render_report(d)["index"]).read_text(encoding="utf-8")
    assert "LOWER is better" in html
    assert "ADE (fraction)" not in html and "(fraction)" not in html
    assert "NEGATIVE Δ is an improvement" in html
