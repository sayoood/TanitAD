"""test_r9_e2_verdict.py -- the E2 GO / NO-GO judge (REFCV9_DESIGN 4.4), on SYNTHETIC JSONs.  CPU only, no model, no GPU.

The GO case; one RED arm per clause (a)-(e) that must flip to NO-GO NAMING THAT clause (and no other); seed 1 failing alone; a missing family;
a NOT-COMPUTABLE clause is NO-GO (never dropped); STRATEGIC is PAUSED (never a pass, never omitted); the per-window row form goes through taniteval.ci.
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
    return {"eval_k2": eval_k2, "eval_k0": eval_k0, "s6c_k2": {"rate": 0.08, "n": 400}, "s6c_k0": {"rate": 0.18, "n": 400}, "families": fams, "camblind": cb}


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
def flip_b(d): d["eval_k2"]["boxes"]["ap2m_all"]["point"] = 0.30                      # below K0's 0.33
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


def test_strategic_is_paused_never_pass_never_omitted(tmp_path):
    rc, v, md = run(tmp_path, both(good_docs()))
    for s in ("0", "1"):
        assert v["four_families"][s]["STRATEGIC"]["status"] == "PAUSED" and "ITEM 25" in v["four_families"][s]["STRATEGIC"]["reason"]
    assert "PAUSED" in md and v["strategic"].startswith("PAUSED")
    d = good_docs()
    d["families"]["rows"]["STRATEGIC"] = {"x": row(-9.0, "lower", True)}          # a supplied strategic row can never turn the pause into a pass
    (tmp_path / "x").mkdir()
    rc, v, _ = run(tmp_path / "x", both(d))
    assert v["four_families"]["0"]["STRATEGIC"]["status"] == "PAUSED"


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
