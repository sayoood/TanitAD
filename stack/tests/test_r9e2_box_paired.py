"""test_r9e2_box_paired.py -- r9e2_box_paired.py (the paired box AP@2 m record for E2 clause (b), PI 2026-11-17 "not separated worse"), on SYNTHETIC JSONs.  CPU only, no scipy.

The record is lifted from ``r9_e1_eval.py --packs-b`` output (``paired_vs_B.boxes.ap2m_all``); the gate comes from both arms' EVAL json (the gates file is Thor-only); a
refused gate / absent block never writes a record; and the record feeds the REAL judge (a tie PASSES, an interval entirely below 0 FAILS).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R9DIR = ROOT / "ops" / "r9e2_read"
for _p in (str(ROOT / "scripts"), str(R9DIR), str(ROOT.parent / "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r9_e2_verdict as V  # noqa: E402
import r9e2_box_paired as BP  # noqa: E402
import r9e2_inputs as INP  # noqa: E402

GATE = {"source": "--gates-json", "sha256": "ab" * 32, "value": 0.37}


def _w(p, obj):
    Path(p).write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def _raw(delta, lo, hi, sep):
    return {"split": "dev", "paired_vs_B": {"alignment": {"n_common": 900, "n_only_a": 0, "n_only_b": 0},
                                           "boxes": {"ap2m_all": {"a": 0.4, "b": 0.4 - delta, "delta": delta, "lo": lo, "hi": hi, "separated": sep, "n_windows": 900,
                                                                  "n_episodes": 70, "n_boot": 2000, "p_delta_gt0": 0.5}}}}


def _rd(tmp, gate2=GATE, gate0=GATE):
    rd = tmp / "read"
    rd.mkdir()
    _w(rd / "EVAL_K2.json", {"gate": gate2, "boxes": {"ap2m_all": {"point": 0.4}}})
    _w(rd / "EVAL_K0.json", {"gate": gate0, "boxes": {"ap2m_all": {"point": 0.41}}})
    return rd


def test_extract_flattens_the_paired_row_and_says_it_serves_both_seeds():
    rec = BP.extract(_raw(-0.01, -0.05, 0.03, False))
    assert rec["better"] == "higher" and rec["metric"] == "boxes.ap2m_all" and (rec["delta"], rec["lo"], rec["hi"], rec["separated"]) == (-0.01, -0.05, 0.03, False)
    assert rec["serves_seeds"] == ["0", "1"] and "deterministic" in rec["note"] and rec["n_episodes"] == 70


@pytest.mark.parametrize("raw", [{}, {"paired_vs_B": {}}, {"paired_vs_B": {"boxes": {}}}, {"boxes": {"ap2m_all": {"delta": 0.1}}}])
def test_extract_refuses_when_there_is_no_paired_box_block(raw):
    with pytest.raises(SystemExit):
        BP.extract(raw)


def test_gate_is_reused_only_when_both_arms_agree():
    assert BP.read_gate({"gate": GATE}, {"gate": GATE})[0] == 0.37
    with pytest.raises(SystemExit):
        BP.read_gate({"gate": GATE}, {"gate": dict(GATE, value=0.5)})
    with pytest.raises(SystemExit):
        BP.read_gate({"gate": GATE}, {"gate": dict(GATE, sha256="cd" * 32)})
    with pytest.raises(SystemExit):
        BP.read_gate({"gate": GATE}, {})


def test_main_extract_only_writes_the_record_with_gate_and_point_check(tmp_path):
    rd = _rd(tmp_path)
    out = tmp_path / "BOX_PAIRED.json"
    assert BP.main(["--rd", str(rd), "--raw", _w(tmp_path / "raw.json", _raw(0.02, -0.01, 0.05, False)), "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["gate"]["value"] == 0.37 and rec["point_check"] == {"eval_k2_point": 0.4, "eval_k0_point": 0.41} and rec["delta"] == 0.02


def test_main_refuses_mismatched_gates_and_writes_nothing(tmp_path):
    rd = _rd(tmp_path, gate0=dict(GATE, value=0.9))
    out = tmp_path / "BOX_PAIRED.json"
    with pytest.raises(SystemExit):
        BP.main(["--rd", str(rd), "--raw", _w(tmp_path / "raw.json", _raw(0.02, -0.01, 0.05, False)), "--out", str(out)])
    assert not out.exists()


def test_run_mode_needs_split_and_eval_script(tmp_path):
    rd = _rd(tmp_path)
    with pytest.raises(SystemExit):
        BP.main(["--rd", str(rd), "--out", str(tmp_path / "o.json")])


def _judge(tmp_path, rec_path):
    """INPUTS via the real writer (BOX_PAIRED.json beside the other reads), then the real judge's clause (b)."""
    rd = tmp_path / "read"
    fam = {"rows": {f: {"r": {"delta": -0.1, "lo": -0.11, "hi": -0.09, "separated": False, "better": "lower"}} for f in V.FAMILIES}}
    cb = {"rows": {k: {"delta": -0.2, "lo": -0.21, "hi": -0.19, "separated": True, "better": "lower"} for k in ("ade_m",) + V.FAMILIES}}
    for s in ("0", "1"):
        _w(rd / f"families_s{s}.json", fam)
        _w(rd / f"camblind_s{s}.json", cb)
    inp = INP.build_inputs(rd)
    return {c["id"]: c for c in V.build(inp)["clauses"]}["b"]


def test_record_end_to_end_through_the_real_inputs_writer_and_judge(tmp_path):
    """tie => PASS, entirely below 0 => FAIL, via BOX_PAIRED.json -> r9e2_inputs.build_inputs -> r9_e2_verdict.build (both seeds read the ONE record)."""
    for name, args, want in (("tie", (-0.004, -0.05, 0.04, False), "PASS"), ("worse", (-0.08, -0.12, -0.03, True), "FAIL"), ("better", (0.06, 0.02, 0.1, True), "PASS")):
        d = tmp_path / name
        d.mkdir()
        rd = _rd(d)
        out = rd / "BOX_PAIRED.json"
        assert BP.main(["--rd", str(rd), "--raw", _w(d / "raw.json", _raw(*args)), "--out", str(out)]) == 0
        c = _judge(d, out)
        assert c["status"] == want and c["per_seed"]["0"]["detail"] == c["per_seed"]["1"]["detail"], (name, c["per_seed"])


def test_real_taniteval_paired_bootstrap_output_is_readable_by_the_judge():
    """The record shape the REAL taniteval.ci.paired_stat_cluster_bootstrap emits (a tie of two identical arms) is a PASS, not a refusal."""
    ci = pytest.importorskip("taniteval.ci")
    items = [float(i % 7) for i in range(60)]
    eid = [str(i // 5) for i in range(60)]
    r = ci.paired_stat_cluster_bootstrap(items, list(items), eid, lambda xs: sum(xs) / len(xs), n_boot=100, seed=0)
    rec = BP.extract({"paired_vs_B": {"alignment": {}, "boxes": {"ap2m_all": r}}})
    assert V.box_paired_clause(rec)["status"] == "PASS"
