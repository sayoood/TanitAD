"""Unit tests for mapft_grade_arm.py -- the RED-arm grader of the refcv8-mapft launch gate (PREREG_MAPFT s1).

The fixtures are LITERAL dicts in the shape `launch_gate.py` writes (an arm token whose `checks` are MISSING because `finalize`
rejects arm evidence; per-check evidence files carrying `arm`, `binding`, `status`, `reasons`), never values computed by the code
under test. Each ESCAPED case removes ONE ingredient of a catch and must flip the grade; `test_the_token_grader_is_blind` re-runs
the replaced token-based rule on the CAUGHT fixture and requires it to MISS -- the defect measured on C3 a0c93c84 (2026-10-27).

Run:  python -m pytest kit/test_mapft_grade_arm.py -q --basetemp=F:/refcv9_mapft_scratch/pt_kit      (no torch needed)
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapft_grade_arm as G  # noqa: E402

ARM = "mapft_planner_trainable"
BIND = {"profile": "refcv8-mapft", "commit": "a" * 40, "tree_sha256": "b" * 64, "argv_sha256": "c" * 64}
MAIN = {"kind": "launch-gate-token", "verdict": "PASS", "reasons": [],
        "binding": {**BIND, "argv": ["--arm", "hier"], "data_sha256": "d" * 64},
        "checks": {"G-DVB": {"status": "PASS"}, "G-LIVE": {"status": "PASS"}, "G-HYG": {"status": "PASS"}}}
ARM_TOK = {"kind": "launch-gate-token", "verdict": "FAIL", "binding": dict(MAIN["binding"]),
           "reasons": [f"a regression arm is active ({ARM}) -- an arm run can never PASS"],
           "checks": {"G-DVB": {"status": "MISSING"}, "G-LIVE": {"status": "MISSING"}, "G-HYG": {"status": "MISSING"}}}
EV = {"G-LIVE": {"kind": "launch-gate-evidence", "check": "G-LIVE", "arm": ARM, "binding": dict(BIND), "status": "FAIL",
                 "reasons": ["the smoke did not complete: SystemExit: [map-only] step 50: 175 FROZEN tensor(s) changed",
                             "metrics.jsonl has no training row at step 100"]},
      "G-DVB": {"kind": "launch-gate-evidence", "check": "G-DVB", "arm": ARM, "binding": dict(BIND), "status": "FAIL",
                "reasons": ["MAP-ONLY: 186 parameter(s) OUTSIDE the map modules are trainable"]}}


def _write(tmp, main=MAIN, tok=ARM_TOK, ev=EV):
    d = tmp / ARM
    (d / "evidence").mkdir(parents=True)
    if tok is not None:
        (d / "FAIL_aaaaaaaaaaaa.json").write_text(json.dumps(tok), encoding="utf-8")
    for c, e in ev.items():
        (d / "evidence" / f"{c}.json").write_text(json.dumps(e), encoding="utf-8")
    mp = tmp / "PASS_aaaaaaaaaaaa.json"
    if main is not None:
        mp.write_text(json.dumps(main), encoding="utf-8")
    return str(d), str(mp)


def _grade(tmp, **kw):
    return G.grade(ARM, *_write(tmp, **kw))


def test_caught_on_the_measured_shape(tmp_path):
    g = _grade(tmp_path)
    assert g.startswith("ARM-mapft_planner_trainable-CAUGHT ")
    assert "red_checks=G-DVB,G-LIVE map_only_reason_on=G-DVB,G-LIVE control_pass=True" in g
    assert g.count("-CAUGHT") == 1


def test_the_token_grader_is_blind(tmp_path):
    """The replaced rule read the ARM TOKEN's checks, which finalize leaves MISSING: it can never report a catch."""
    d, mp = _write(tmp_path)
    tok = json.loads((Path(d) / "FAIL_aaaaaaaaaaaa.json").read_text(encoding="utf-8"))
    bad = [c for c in ("G-DVB", "G-LIVE") if tok["checks"].get(c, {}).get("status") in ("FAIL", "ERROR")]
    assert bad == []                                    # the old grader's red set on the real shape: empty -> ESCAPED
    assert G.grade(ARM, d, mp).startswith("ARM-mapft_planner_trainable-CAUGHT ")


def _ev(changes):
    e = copy.deepcopy(EV)
    for (c, k), v in changes.items():
        e[c][k] = v
    return e


def test_escaped_when_the_checks_pass(tmp_path):
    g = _grade(tmp_path, ev=_ev({("G-LIVE", "status"): "PASS", ("G-DVB", "status"): "PASS"}))
    assert "-ESCAPED-OR-MISCREDITED " in g and "-CAUGHT" not in g and "red_checks= " in g


def test_escaped_without_a_map_only_reason(tmp_path):
    g = _grade(tmp_path, ev=_ev({("G-LIVE", "reasons"): ["metrics.jsonl has no training row at step 100"],
                                   ("G-DVB", "reasons"): ["declared_vs_built: 1 mismatch"]}))
    assert "-CAUGHT" not in g and "red_checks=G-DVB,G-LIVE map_only_reason_on= " in g


def test_escaped_when_the_evidence_is_bound_to_another_commit(tmp_path):
    other = {**BIND, "commit": "e" * 40}
    g = _grade(tmp_path, ev=_ev({("G-LIVE", "binding"): other, ("G-DVB", "binding"): other}))
    assert "-CAUGHT" not in g and "notes=G-DVB:binding-differs,G-LIVE:binding-differs" in g


def test_escaped_when_the_evidence_is_another_arms(tmp_path):
    g = _grade(tmp_path, ev=_ev({("G-LIVE", "arm"): "mapft_map_frozen", ("G-DVB", "arm"): None}))
    assert "-CAUGHT" not in g and "notes=G-DVB:not-this-arm,G-LIVE:not-this-arm" in g


def test_escaped_when_the_control_does_not_pass_the_check(tmp_path):
    main = copy.deepcopy(MAIN)
    main["checks"]["G-LIVE"]["status"] = "FAIL"
    g = _grade(tmp_path, main=main)
    assert "-CAUGHT" not in g and "control_pass=False" in g


def test_escaped_when_the_control_token_is_not_pass(tmp_path):
    main = copy.deepcopy(MAIN)
    main["verdict"] = "INCOMPLETE"
    g = _grade(tmp_path, main=main)
    assert "-CAUGHT" not in g and "control_pass=False" in g


def test_escaped_when_the_arm_token_passes_or_does_not_name_the_arm(tmp_path):
    tok = copy.deepcopy(ARM_TOK)
    tok["verdict"] = "PASS"
    assert "-CAUGHT" not in _grade(tmp_path / "a", tok=tok)
    tok = copy.deepcopy(ARM_TOK)
    tok["reasons"] = ["a regression arm is active (mapft_map_frozen) -- an arm run can never PASS"]
    g = _grade(tmp_path / "b", tok=tok)
    assert "-CAUGHT" not in g and "arm_named=False" in g


def test_ungraded_without_a_token(tmp_path):
    assert _grade(tmp_path / "a", tok=None) == "ARM-mapft_planner_trainable-UNGRADED(no arm token or no control token)"
    assert _grade(tmp_path / "b", main=None) == "ARM-mapft_planner_trainable-UNGRADED(no arm token or no control token)"


def test_one_check_is_enough_and_a_missing_file_is_named(tmp_path):
    ev = {"G-LIVE": EV["G-LIVE"]}
    g = _grade(tmp_path, ev=ev)
    assert g.startswith("ARM-mapft_planner_trainable-CAUGHT ") and "notes=G-DVB:no-evidence" in g
