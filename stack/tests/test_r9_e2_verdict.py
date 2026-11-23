"""test_r9_e2_verdict.py -- the E2 GO / NO-GO judge (REFCV9_DESIGN 4.4), on SYNTHETIC JSONs.  CPU only, no model, no GPU.

Clause (b) is "K2 NOT SEPARATED WORSE than K0" on box AP@2 m (PI 2026-11-17), read from the paired ``box_paired`` record: a tie PASSES, an interval entirely below 0 FAILS,
K2 better PASSES, an absent / non-finite record is NOT-COMPUTABLE (never the old point rule).

The GO case; one RED arm per clause (a)-(e) that must flip to NO-GO NAMING THAT clause (and no other); seed 1 failing alone; a missing family;
a NOT-COMPUTABLE clause is NO-GO (never dropped); STRATEGIC is REPORT-ONLY / NOT-SUPPLIED (PI 2026-11-16; never PAUSED, never a pass, never in clause (d)); the per-window row form goes through taniteval.ci.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "r9_e2_verdict.py"
_spec = importlib.util.spec_from_file_location("r9_e2_verdict", SCRIPT)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


def row(delta, better="lower", sep=False):
    lo, hi = (delta - 0.01, delta + 0.01) if sep else (delta - 1.0, delta + 1.0)
    return {"delta": delta, "lo": lo, "hi": hi, "separated": bool(sep), "better": better, "n": 100}


def good_docs():
    """K2 beats K0 everywhere; no family row separated worse; W4 separated better on every row."""
    eval_k2 = {"boxes": {"ap2m_all": {"point": 0.40}}, "nearfield": {"ci": {V.MAP_KEY: {"point": 0.80}}}, "n_windows": 900, "n_clips": 70}
    eval_k0 = {"boxes": {"ap2m_all": {"point": 0.33}}, "nearfield": {"ci": {V.MAP_KEY: {"point": 0.55}}}}
    fams = {"rows": {"LONGITUDINAL": {"ade_m": row(-0.2, "lower", True), "speed_err": row(0.0)},
                     "LATERAL": {"curv_err": row(-0.01, "lower")},
                     "TACTICAL": {"decision_acc": row(0.02, "higher")}}}
    cb = {"rows": {"ade_m": row(-0.5, "lower", True), "LONGITUDINAL": row(-0.3, "lower", True),
                   "LATERAL": row(-0.2, "lower", True), "TACTICAL": row(0.1, "higher", True)}}
    box_paired = {"better": "higher", "delta": 0.07, "lo": 0.02, "hi": 0.12, "separated": True}              # K2 separated BETTER than K0 on box AP@2 m
    return {"eval_k2": eval_k2, "eval_k0": eval_k0, "box_paired": box_paired, "s6c_k2": {"rate": 0.08, "n": 400}, "s6c_k0": {"rate": 0.18, "n": 400}, "families": fams, "camblind": cb}


def write(tmp, docs_by_seed):
    seeds = {}
    for s, docs in docs_by_seed.items():
        seeds[s] = {}
        for k, d in docs.items():
            p = tmp / f"s{s}_{k}.json"
            p.write_text(json.dumps(d), encoding="utf-8")
            seeds[s][k] = str(p)
    inp = tmp / "INPUTS.json"
    inp.write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
    return inp


def run(tmp, docs_by_seed):
    out = tmp / "out"
    rc = V.main(["--inputs", str(write(tmp, docs_by_seed)), "--out-dir", str(out)])
    return rc, json.loads((out / "E2_VERDICT.json").read_text(encoding="utf-8")), (out / "E2_VERDICT.md").read_text(encoding="utf-8")


def both(d):
    return {"0": copy.deepcopy(d), "1": copy.deepcopy(d)}


def test_go_case(tmp_path):
    rc, v, md = run(tmp_path, both(good_docs()))
    assert rc == 0 and v["verdict"] == "GO" and v["failing_clauses"] == []
    assert [c["status"] for c in v["clauses"]] == ["PASS"] * 5
    assert "E2 verdict: GO" in md or "# E2 verdict: GO" in md


def flip_a(d): d["eval_k2"]["nearfield"]["ci"][V.MAP_KEY]["point"] = 0.50            # below K0's 0.55
def flip_b(d): d["box_paired"].update(delta=-0.09, lo=-0.14, hi=-0.03, separated=True)   # paired interval ENTIRELY below 0: separated worse
def flip_c(d): d["s6c_k2"]["rate"] = 0.20                                             # above K0's 0.18 (lower is better)
def flip_d(d): d["families"]["rows"]["LATERAL"]["curv_err"] = row(0.05, "lower", True)   # separated WORSE (lower is better, delta > 0)
def flip_e(d): d["camblind"]["rows"]["TACTICAL"] = row(0.01, "higher", False)         # not separated vs camera-blind


@pytest.mark.parametrize("cid,flip", [("a", flip_a), ("b", flip_b), ("c", flip_c), ("d", flip_d), ("e", flip_e)])
def test_each_clause_red_arm_names_that_clause(tmp_path, cid, flip):
    d = good_docs()
    flip(d)
    rc, v, md = run(tmp_path, both(d))
    assert rc == 1 and v["verdict"] == "NO-GO"
    assert [f["id"] for f in v["failing_clauses"]] == [cid], v["failing_clauses"]
    assert v["failing_clauses"][0]["seeds"] == ["0", "1"] and v["failing_clauses"][0]["next_lever"]
    assert f"({cid})" in md and "NO-GO names clause" in md


def test_seed1_failing_alone_is_no_go(tmp_path):
    docs = both(good_docs())
    docs["1"]["s6c_k2"]["rate"] = 0.25
    rc, v, _ = run(tmp_path, docs)
    assert rc == 1 and [f["id"] for f in v["failing_clauses"]] == ["c"] and v["failing_clauses"][0]["seeds"] == ["1"]
    assert v["clauses"][2]["per_seed"]["0"]["status"] == "PASS" and v["clauses"][2]["per_seed"]["1"]["status"] == "FAIL"


def test_missing_family_is_no_go_not_dropped(tmp_path):
    d = good_docs()
    del d["families"]["rows"]["TACTICAL"]
    rc, v, _ = run(tmp_path, both(d))
    assert rc == 1 and [f["id"] for f in v["failing_clauses"]] == ["d"] and v["failing_clauses"][0]["status"] == "NOT-COMPUTABLE"
    assert v["four_families"]["0"]["TACTICAL"]["status"] == "NOT-COMPUTABLE"


def test_not_computable_clause_is_no_go(tmp_path):
    d = good_docs()
    del d["eval_k2"]["nearfield"]                                  # the lift-valid primary key absent = NOT-COMPUTABLE (r9_e1_eval emits none without a mask)
    del d["s6c_k0"]["rate"]
    rc, v, _ = run(tmp_path, both(d))
    assert rc == 1 and [(f["id"], f["status"]) for f in v["failing_clauses"]] == [("a", "NOT-COMPUTABLE"), ("c", "NOT-COMPUTABLE")]


def _sub(tmp, name, d):
    p = tmp / name
    p.mkdir()
    return run(p, both(d))


def _failing_d_docs():
    d = good_docs()
    d["families"]["rows"]["LATERAL"]["curv_err"] = row(0.4, "lower", True)         # a LATERAL row separated WORSE: clause (d) FAILS
    return d


def test_strategic_not_supplied_is_never_paused_never_pass_never_omitted(tmp_path):
    """(a) no strategic rows => NOT-SUPPLIED, the reason names the producer, and 'PAUSED' appears nowhere in the JSON or the md."""
    rc, v, md = run(tmp_path, both(good_docs()))
    assert rc == 0 and v["verdict"] == "GO"
    for s in ("0", "1"):
        st = v["four_families"][s]["STRATEGIC"]
        assert st["status"] == "NOT-SUPPLIED" and st["rows"] == {} and "openloop_suite.py:943" in st["reason"]
    blob = json.dumps(v)
    for text in (blob, md):
        assert "PAUSED" not in text and "paused" not in text.lower() and "ITEM 25" not in text
    assert v["strategic"].startswith("REPORT-ONLY") and "STRATEGIC" in md and "NOT-SUPPLIED" in md
    d = good_docs()                                                                 # a stamp dict (no dict-valued rows, as r9_e2_families.build emits) carries no rows
    d["families"]["rows"]["STRATEGIC"] = {"status": "PAUSED", "reason": "stamp copied from a producer"}
    rc, v2, md2 = _sub(tmp_path, "stamp", d)
    assert v2["four_families"]["0"]["STRATEGIC"]["status"] == "NOT-SUPPLIED" and "PAUSED" not in json.dumps(v2) and "PAUSED" not in md2


def test_strategic_separated_worse_is_reported_but_never_enters_the_rule(tmp_path):
    """(b) a supplied strategic row separated WORSE is REPORT-ONLY, visible in JSON and md, and clause (d) + the verdict equal the same input without it."""
    for k, base in (("go", good_docs()), ("nogo", _failing_d_docs())):
        with_row = copy.deepcopy(base)
        with_row["families"]["rows"]["STRATEGIC"] = {"route_acc": row(0.3, "lower", True), "route_zero": row(0.0, "lower", False)}
        rc0, v0, md0 = _sub(tmp_path, k + "_without", base)
        rc1, v1, md1 = _sub(tmp_path, k + "_with", with_row)
        st = v1["four_families"]["0"]["STRATEGIC"]
        assert st["status"] == "REPORT-ONLY" and st["rows"]["route_acc"]["separated"] is True and st["rows"]["route_acc"]["separated_worse"] is True
        assert st["separated_worse_rows"] == ["route_acc"] and "separated worse (report-only, outside clause (d))" in st["rows"]["route_acc"]["report_only_note"]
        assert "STRATEGIC/route_acc" in md1 and "separated worse (report-only, outside clause (d))" in md1 and "PAUSED" not in md1
        assert (rc1, v1["verdict"]) == (rc0, v0["verdict"])
        assert v1["clauses"] == v0["clauses"] and v1["failing_clauses"] == v0["failing_clauses"]          # clause (d) and every other clause are IDENTICAL
        d1 = {c["id"]: c for c in v1["clauses"]}["d"]
        assert not any("STRATEGIC" in str(x) for x in d1["per_seed"]["0"].get("failing_rows", []))


def test_strategic_separated_better_cannot_turn_a_failing_d_into_a_pass(tmp_path):
    """(c) mirror of the old check: a strategic row separated BETTER by a mile cannot rescue a failing clause (d)."""
    d = _failing_d_docs()
    d["families"]["rows"]["STRATEGIC"] = {"x": row(-9.0, "lower", True)}
    rc, v, _ = run(tmp_path, both(d))
    cd = {c["id"]: c for c in v["clauses"]}["d"]
    assert rc == 1 and cd["status"] == "FAIL" and [f["id"] for f in v["failing_clauses"]] == ["d"]
    assert cd["per_seed"]["0"]["failing_rows"] == ["LATERAL/curv_err"]
    st = v["four_families"]["0"]["STRATEGIC"]
    assert st["status"] == "REPORT-ONLY" and st["rows"]["x"]["separated_better"] is True and st["separated_worse_rows"] == []


def test_both_seeds_required(tmp_path):
    inp = tmp_path / "i.json"
    inp.write_text(json.dumps({"seeds": {"0": {}}}), encoding="utf-8")
    assert V.main(["--inputs", str(inp), "--out-dir", str(tmp_path / "o")]) == 2


def test_per_window_rows_use_ci_paired_bootstrap(tmp_path):
    """The per-window form is scored by taniteval.ci (no new estimator): K2 uniformly worse by 1.0 on a `lower`-better row => separated worse."""
    n_ep = 12
    eid = [e for e in range(n_ep) for _ in range(5)]
    a = [2.0 + 0.01 * i for i in range(len(eid))]
    b = [x - 1.0 for x in a]                                         # a - b = +1.0 everywhere
    r = V.row_read({"a": a, "b": b, "eid": eid, "better": "lower"})
    assert r["ok"] and r["separated"] and r["separated_worse"] and not r["separated_better"] and r["src"].startswith("ci.paired")
    r2 = V.row_read({"a": b, "b": a, "eid": eid, "better": "lower"})
    assert r2["ok"] and r2["separated_better"] and not r2["separated_worse"]
    assert V.row_read({"a": [1.0], "b": [1.0, 2.0], "eid": [0], "better": "lower"})["ok"] is False        # unaligned => refused, not guessed


def _b(tmp, rec, seed_docs=None):
    d = good_docs()
    d["box_paired"] = rec
    (tmp).mkdir(exist_ok=True)
    return run(tmp, both(d))


def _status_b(v):
    return {c["id"]: c["status"] for c in v["clauses"]}["b"]


def test_clause_b_tie_passes_not_separated_worse(tmp_path):
    """A tie (delta ~0, CI spans 0) and even a NEGATIVE point delta whose CI reaches 0 PASS: the PI's 'not separated worse'."""
    for i, rec in enumerate(({"better": "higher", "delta": 0.0004, "lo": -0.03, "hi": 0.03, "separated": False},
                             {"better": "higher", "delta": -0.02, "lo": -0.05, "hi": 0.01, "separated": False})):
        rc, v, md = _b(tmp_path / f"t{i}", rec)
        assert rc == 0 and v["verdict"] == "GO" and _status_b(v) == "PASS", v["failing_clauses"]
        assert "not separated worse" in v["clauses"][1]["per_seed"]["0"]["detail"]


def test_clause_b_fails_only_when_the_ci_is_entirely_below_zero(tmp_path):
    rc, v, _ = _b(tmp_path / "w", {"better": "higher", "delta": -0.06, "lo": -0.10, "hi": -0.01, "separated": True})
    assert rc == 1 and [f["id"] for f in v["failing_clauses"]] == ["b"] and v["failing_clauses"][0]["status"] == "FAIL" and v["failing_clauses"][0]["seeds"] == ["0", "1"]
    rc, v, _ = _b(tmp_path / "edge", {"better": "higher", "delta": -0.06, "lo": -0.10, "hi": 0.0, "separated": False})          # hi == 0: the interval reaches 0 => not separated
    assert rc == 0 and _status_b(v) == "PASS"


def test_clause_b_k2_better_passes_and_reports_the_point_delta(tmp_path):
    rc, v, _ = _b(tmp_path / "k", {"better": "higher", "delta": 0.07, "lo": 0.02, "hi": 0.12, "separated": True})
    assert rc == 0 and _status_b(v) == "PASS"
    c = v["clauses"][1]["per_seed"]["0"]
    assert c["delta"] == 0.07 and "delta=+0.07" in c["detail"] and "estimator" in c and "paired" in c["estimator"]


@pytest.mark.parametrize("rec", [None, {}, {"better": "higher", "delta": float("nan"), "lo": -0.1, "hi": 0.1},
                                 {"better": "higher", "delta": 0.01, "lo": None, "hi": 0.1}, {"better": "lower", "delta": 0.01, "lo": -0.1, "hi": 0.1},
                                 {"better": "higher", "delta": 0.01, "lo": -0.1, "hi": 0.1, "undefined": True}])
def test_clause_b_missing_or_nonfinite_record_is_not_computable(tmp_path, rec):
    d = good_docs()
    if rec is None:
        del d["box_paired"]                                                                       # key absent altogether
    else:
        d["box_paired"] = rec
    rc, v, _ = run(tmp_path, both(d))
    assert rc == 1 and [(f["id"], f["status"]) for f in v["failing_clauses"]] == [("b", "NOT-COMPUTABLE")]


def test_RED_the_old_point_rule_would_flip_the_tie_case(tmp_path):
    """Deliberate regression: a tie whose point K2 < K0 FAILED the old rule (K2 > K0 on points); the amended judge PASSES it, and the old rule, applied to
    the same numbers, still FAILS -- so a revert of clause (b) to point_clause is caught."""
    tie = {"better": "higher", "delta": -0.004, "lo": -0.05, "hi": 0.04, "separated": False}
    rc, v, _ = _b(tmp_path / "tie", tie)
    assert rc == 0 and _status_b(v) == "PASS"
    old = V.point_clause("b", 0.33 + tie["delta"], 0.33, True, "boxes.ap2m_all.point")
    assert old["status"] == "FAIL"
    # revert the clause to the point rule inside the judge: the same tie input must now fail the GO
    real = V.box_paired_clause
    try:
        V.box_paired_clause = lambda rec: V.point_clause("b", 0.33 + rec["delta"], 0.33, True, "boxes.ap2m_all.point")
        rc2, v2, _ = _b(tmp_path / "tie2", tie)
    finally:
        V.box_paired_clause = real
    assert rc2 == 1 and [f["id"] for f in v2["failing_clauses"]] == ["b"]


def test_rule_text_and_clause_name_record_the_amendment():
    assert "clause (b) amended by the PI 2026-11-17: not separated worse" in V.RULE
    assert dict(V.CLAUSES)["b"].startswith("box AP@2 m head output no NMS: K2 not separated worse than K0")
    assert "paired" in V.EST["b"] and "NOT SEPARATED WORSE" in V.__doc__
