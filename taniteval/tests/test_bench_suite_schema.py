"""The CONTRACT SCHEMAS and the dependency-free validator (W1). Every expectation is a literal;
each binding rule has a mutation that must go RED.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from taniteval.bench import schema_check as SC

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# the validator itself                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst,schema,n_err", [
    (1, {"type": "integer"}, 0),
    (True, {"type": "integer"}, 1),                     # JSON: a bool is not a number
    (1.0, {"type": "integer"}, 0),
    ("x", {"type": ["string", "null"]}, 0),
    (None, {"type": ["string", "null"]}, 0),
    ({"a": 1}, {"required": ["a", "b"]}, 1),
    ("abc", {"pattern": "^a"}, 0),
    ("bbc", {"pattern": "^a"}, 1),
    ([1, 2], {"items": {"type": "integer"}}, 0),
    ([1, "x"], {"items": {"type": "integer"}}, 1),
    (["STOP"], {"contains": {"const": "STOP"}}, 0),
    (["CV"], {"contains": {"const": "STOP"}}, 1),
    ("score", {"not": {"const": "pdm_score"}}, 0),
    ("pdm_score", {"not": {"const": "pdm_score"}}, 1),
    ({"s": "A"}, {"if": {"properties": {"s": {"const": "A"}}, "required": ["s"]},
                  "then": {"required": ["only_for_a"]}}, 1),
    ({"s": "B"}, {"if": {"properties": {"s": {"const": "A"}}, "required": ["s"]},
                  "then": {"required": ["only_for_a"]}}, 0),
    ({"x": 1}, {"additionalProperties": False}, 1),
    ({"a": 1}, {"minProperties": 2}, 1),
])
def test_validator_literals(inst, schema, n_err):
    assert len(SC.validate(inst, schema)) == n_err


def test_validator_refuses_an_unsupported_keyword():
    """An unknown keyword is REFUSED, never ignored — silent non-checking is the failure class."""
    with pytest.raises(SC.SchemaError):
        SC.validate({"a": 1}, {"uniqueItems": True})


# --------------------------------------------------------------------------- #
# the two contract schemas                                                     #
# --------------------------------------------------------------------------- #
def _fam():
    return {"status": "UNAVAILABLE", "reason": "no GT future on these scenes", "n": 0}


def _arm(kind="floor", name="CV"):
    paired = {f: ({"status": "SELF"} if f == name else
                  {"status": "OK", "headline_delta": -0.1155, "n_common": 220}) for f in ("STOP", "CV")}
    return {"kind": kind, "status": "OK", "declared_inputs": ["ego_velocity[t0]"],
            "headline": {"value": 0.1853562745165113, "column": "score", "statistic": "official two-stage EPDMS", "n": 220},
            "per_stage": {}, "per_log": {}, "submetrics": {},
            "paired": paired,
            "interval": {"status": "UNAVAILABLE", "reason": "7 log groups < the RG-14 floor of 8", "n": 7},
            "families": {k: _fam() for k in ("longitudinal", "lateral", "tactical", "strategic")}, "files": {}}


def _summary():
    return {"schema": "taniteval.bench.summary/1", "run_id": "r", "benchmark": "navsim_v2",
            "protocol": "EPDMS_v2_warmup_two_stage", "split": "warmup_two_stage", "claim_bearing": True,
            "evidence_class": "MEASURED", "stamps": {"tier": "T1-family", "loop": {"stage_one": "OPEN"}},
            "headline_metric": {"name": "EPDMS", "column": "score", "higher_is_better": True, "statistic": "x"},
            "floors": ["STOP", "CV"], "arms": {"CV": _arm(name="CV"), "STOP": _arm(name="STOP")}}


def test_good_summary_validates():
    assert SC.validate(_summary(), SC.load_schema("summary")) == []


@pytest.mark.parametrize("mutate,expect_in", [
    (lambda s: s["headline_metric"].__setitem__("column", "pdm_score"), "pdm_score"),          # M-COL
    (lambda s: s.__setitem__("floors", ["CV"]), "STOP"),                                        # M-FLOOR
    (lambda s: s["arms"]["CV"]["families"].pop("strategic"), "strategic"),
    (lambda s: s["arms"]["CV"]["families"].__setitem__("lateral", {"status": "UNAVAILABLE", "n": 0}), "reason"),
    (lambda s: s["arms"]["CV"].__setitem__("interval", {"status": "OK", "estimator": "overlapping_holdout_se",
                                                        "cluster_unit": "log_name", "lo": 0.1, "hi": 0.2,
                                                        "n_clusters": 8}), "interval"),
    (lambda s: s.__setitem__("protocol", "EPDMS_v2"), "protocol"),
    (lambda s: s["arms"]["CV"].pop("per_stage"), "per_stage"),
    (lambda s: s["arms"]["CV"].__setitem__("kind", "baseline"), "kind"),
])
def test_summary_mutations_go_red(mutate, expect_in):
    s = _summary()
    mutate(s)
    errs = SC.validate(s, SC.load_schema("summary"))
    assert errs and any(expect_in in e for e in errs), errs


def test_nuscenes_summary_must_not_claim():
    s = _summary()
    s.update(benchmark="nuscenes_ol", protocol="nuScenes_OL_L2_stp3", claim_bearing=True, floors=[])
    s["headline_metric"] = {"name": "L2@3s", "column": "l2_3s_m", "higher_is_better": False, "statistic": "x"}
    s["arms"] = {"A": _arm("model", name="A")}
    s["arms"]["A"]["paired"] = {}
    errs = SC.validate(s, SC.load_schema("summary"))
    assert any("claim_bearing" in e for e in errs), errs
    s["claim_bearing"] = False
    assert SC.validate(s, SC.load_schema("summary")) == []


def test_bench_run_schema_literals():
    from taniteval.bench import contract as C
    rec = {"schema": "taniteval.bench.bench_run/1", "run_id": "20260919T193012Z-navsim_v2-none-a1b2c3",
           "utc": "2026-09-19T19:30:12Z", "git_head": "0" * 40,
           "ckpt": {"path": None, "sha256": None, "registry_key": None}, "benchmark": "navsim_v2",
           "protocol": "EPDMS_v2_warmup_two_stage",
           "devkit": {"repo": "autonomousvision/navsim", "sha": "a" * 40, "patches": []},
           "split": {"name": "warmup_two_stage", "n_scenes": 220, "n_logs": 7},
           "arms": [{"name": "CV", "kind": "floor", "declared_inputs": []},
                    {"name": "STOP", "kind": "floor", "declared_inputs": []}],
           "device": {"requested": "auto", "used": "cpu"}, "wall_s": 12.0, "claim_bearing": True,
           "status": "COMPLETE", "stamps": {"tier": "T1-family", "loop": {"a": "b"}, "evidence_class": "MEASURED"},
           "report": {"status": "REPORT_PENDING"}}
    assert C.validate_bench_run(rec) == []
    bad = copy.deepcopy(rec)
    bad["run_id"] = "not-a-run-id"
    assert any("run_id" in e for e in C.validate_bench_run(bad))
    bad = copy.deepcopy(rec)
    bad["arms"] = [a for a in bad["arms"] if a["name"] != "STOP"]
    assert any("STOP" in e for e in C.validate_bench_run(bad))          # M-FLOOR on the run record
    bad = copy.deepcopy(rec)
    bad["git_head"] = "UNAVAILABLE"
    assert C.validate_bench_run(bad) == []                              # honest UNAVAILABLE is allowed


# --------------------------------------------------------------------------- #
# the closed protocol set vs the registry (orchestrator ruling 2026-09-19 (c)) #
# --------------------------------------------------------------------------- #
def _registry_protocol_tags() -> set:
    """W2's public API (`tools/criteria_check.py::registered_protocol_tags`, 2026-09-20) if it is
    there; otherwise the registry walked directly."""
    reg = json.loads((REPO / "products/P7-TanitEval/CRITERIA_REGISTRY.json").read_text(encoding="utf-8"))
    import sys
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import criteria_check as cc
        fn = getattr(cc, "registered_protocol_tags", None)
        if callable(fn):
            out = fn(reg)
            return set(out)          # {tag: where it is registered} -> the TAGS
    except Exception:                                                    # noqa: BLE001
        pass
    tags = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("closed_set", "protocol_tags") and isinstance(v, list):
                    tags.update(str(x) for x in v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(reg.get("benchmarks", {}))
    return tags


def test_protocol_closed_set_agrees_with_the_registry():
    from taniteval.bench.contract import PROTOCOLS
    mine, theirs = set(PROTOCOLS), _registry_protocol_tags()
    extra = theirs - mine
    assert not extra, (f"the registry declares protocol tag(s) the suite's closed set does not know: {sorted(extra)} "
                       "— W4's leaderboard would have no table for them")
    missing = mine - theirs
    if missing:
        pytest.xfail(f"W2 has not landed these protocol tags in CRITERIA_REGISTRY.json yet: {sorted(missing)}")
    assert mine == theirs


def test_navsim_subset_matches_the_registry_gate_exactly():
    reg = json.loads((REPO / "products/P7-TanitEval/CRITERIA_REGISTRY.json").read_text(encoding="utf-8"))
    closed = set(reg["benchmarks"]["navsim"]["GATE_no_cross_protocol_comparison"]["closed_set"])
    from taniteval.bench.contract import PROTOCOLS
    mine = {p for p in PROTOCOLS if p.startswith(("EPDMS_v2", "PDMS_v1"))}
    assert mine == closed
