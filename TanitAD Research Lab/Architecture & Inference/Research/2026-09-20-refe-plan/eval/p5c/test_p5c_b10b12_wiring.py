"""Reach tests for B10 / B11 / B12: the READ and the RENDERER must consume the new outputs. "A test that the mechanism WORKS is not evidence that it RUNS from its caller."
  * B10: a positive verdict is INCOMPLETE without the determinism result and changes when it is supplied (PASS -> SEPARATED-SINGLE, FAIL -> INVALID).
  * B11: the read verifies SEALED.json itself; a tampered seal makes the read INVALID; the CLI refuses a real read without one.
  * B12: the tactical block rides into the read JSON and the report; an absent block is printed as NOT COMPUTED, never dropped.
  * MUTATION: the source line that consumes each output is deleted in a copy of p5c_read.py and the same assertions must go RED.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_b10b12_wiring.py -q"""
import copy
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_read as R          # noqa: E402
import p5c_render as X        # noqa: E402
import p5c_determinism as D   # noqa: E402
import p5c_seal as S          # noqa: E402
import p5c_tactical as T      # noqa: E402
from test_p5c_read import mk_rows, arms_with_shift, _fam, ent   # noqa: E402


def GOODFAM():
    return _fam(entries_tracked={"speed_mae_mps": ent(-0.1, 0.1)})
from test_p5c_determinism import mk_rows as det_rows, rerun, toks as det_toks   # noqa: E402

H = lambda c: c * 64   # noqa: E731


def jwrite(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f)
    return str(p)


def positive(rows=None, **kw):
    """a read whose tracked d is far above B and whose G2 passes: the verdict depends only on the controls."""
    return R.run(rows or mk_rows(arms_with_shift(0.05)), "C1", "C0", "base", fam_ci=GOODFAM(), **kw)


def b10_result(ok=True):
    pop = det_toks(200)
    sub = D.choose_subset(pop)
    r1 = det_rows(pop)
    r2 = rerun(r1, sub)
    if not ok:
        r2[sub[0]]["arms"]["C1"]["tracked"]["pdms"] += 1e-9
    return D.compare(r1, r2, sub, ["C1", "C0", "base"])


def sealed_doc(**kw):
    arms = {"C1": {"model_final_sha256": H("1"), "hash_source": "declared"}, "C0": {"model_final_sha256": H("2"), "hash_source": "declared"}}
    return S.build_sealed(arms, H("c"), H("b"), kw.get("utc", "2001-01-01T00:00:00Z"))


# ------------------------------------------------------------------------------------------------------ B10 reach
def test_B10_absent_blocks_a_positive_verdict_and_present_changes_it(tmp_path):
    none = positive(require_kinds=("B10_determinism",))
    assert none["verdict"]["verdict"] == "INCOMPLETE" and "B10_determinism" in none["verdict"]["wording"]
    assert none["controls"]["missing_kinds"] == ["B10_determinism"]
    ok = positive(controls=[jwrite(tmp_path / "b10.json", b10_result(True))], require_kinds=("B10_determinism",))
    assert ok["verdict"]["verdict"] == "SEPARATED-SINGLE" and ok["controls"]["missing_kinds"] == []
    assert ok["controls"]["supplied"][0]["kind"] == "B10_determinism" and ok["controls"]["supplied"][0]["pass"] is True


def test_RED_B10_failed_rerun_makes_the_read_INVALID(tmp_path):
    bad = positive(controls=[jwrite(tmp_path / "b10.json", b10_result(False))], require_kinds=("B10_determinism",))
    assert bad["verdict"]["verdict"] == "INVALID" and bad["controls"]["all_pass"] is False


def test_RED_a_different_control_does_not_stand_in_for_B10(tmp_path):
    other = positive(controls=[jwrite(tmp_path / "id.json", {"pass": True})], require_kinds=("B10_determinism",))
    assert other["verdict"]["verdict"] == "INCOMPLETE"             # an unnamed {"pass": true} is not the determinism result


def test_without_required_kinds_the_old_behaviour_holds(tmp_path):
    assert positive(controls=[jwrite(tmp_path / "id.json", {"pass": True})])["verdict"]["verdict"] == "SEPARATED-SINGLE"


# ------------------------------------------------------------------------------------------------------ B11 reach
def test_B11_sealed_is_verified_by_the_read(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}")
    ok = positive(controls=[jwrite(tmp_path / "b10.json", b10_result())], sealed={"doc": sealed_doc(), "file": "S.json"},
                  require_kinds=("B10_determinism", "B11_sealed"), rows_paths=[str(rows)])
    assert ok["verdict"]["verdict"] == "SEPARATED-SINGLE" and ok["sealed"]["verdict"] == "PASS"
    assert {c["kind"] for c in ok["controls"]["supplied"]} == {"B10_determinism", "B11_sealed"}


def test_RED_B11_tampered_seal_makes_the_read_INVALID(tmp_path):
    d = sealed_doc()
    d["arms"]["C1"]["model_final_sha256"] = H("9")
    res = positive(controls=[jwrite(tmp_path / "b10.json", b10_result())], sealed={"doc": d, "file": "S.json"}, require_kinds=("B10_determinism", "B11_sealed"))
    assert res["verdict"]["verdict"] == "INVALID" and res["sealed"]["pass"] is False


def test_RED_B11_rows_older_than_the_seal(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}")
    os.utime(rows, (1.0e9, 1.0e9))
    res = positive(controls=[jwrite(tmp_path / "b10.json", b10_result())], sealed={"doc": sealed_doc(utc=S.utc_now()), "file": "S.json"}, rows_paths=[str(rows)])
    assert res["verdict"]["verdict"] == "INVALID"


def write_rows_file(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")


def cli(*a):
    return subprocess.run([sys.executable, os.path.join(HERE, "p5c_read.py"), *map(str, a)], capture_output=True, text=True, encoding="utf-8")


def test_cli_real_read_needs_a_seal_and_the_required_controls(tmp_path):
    write_rows_file(tmp_path / "rows.jsonl", mk_rows(arms_with_shift(0.05)))
    fam = jwrite(tmp_path / "fam.json", GOODFAM())
    b10 = jwrite(tmp_path / "b10.json", b10_result())
    base = ["--rows", tmp_path / "rows.jsonl", "--c1", "C1", "--c0", "C0", "--base", "base", "--fam-ci", fam, "--n-boot", 200]
    r = cli(*base, "--controls", b10, "--out", tmp_path / "o.json")
    assert r.returncode == 2 and "--sealed" in r.stderr                               # RED: no seal, not a rehearsal
    jwrite(tmp_path / "SEALED.json", sealed_doc())
    r = cli(*base, "--controls", b10, "--sealed", tmp_path / "SEALED.json", "--out", tmp_path / "o.json")
    assert r.returncode == 0, r.stderr
    res = json.load(open(tmp_path / "o.json"))
    assert res["verdict"]["verdict"] == "SEPARATED-SINGLE" and res["controls"]["required_kinds"] == ["B10_determinism", "B11_sealed"]
    r = cli(*base, "--sealed", tmp_path / "SEALED.json", "--controls", jwrite(tmp_path / "x.json", {"pass": True}), "--out", tmp_path / "o2.json")
    assert r.returncode == 0 and json.load(open(tmp_path / "o2.json"))["verdict"]["verdict"] == "INCOMPLETE"       # B10 missing
    r = cli(*base, "--rehearsal", "stand-ins", "--out", tmp_path / "o3.json")           # a rehearsal still needs neither
    assert r.returncode == 0 and json.load(open(tmp_path / "o3.json"))["verdict"]["verdict"].startswith("REHEARSAL:")


# ------------------------------------------------------------------------------------------------------ B12 reach
def test_B12_absent_tactical_is_printed_not_dropped():
    res = positive(rehearsal="t")
    assert res["tactical"]["command_adherence"]["status"] == "NOT COMPUTED" and res["tactical"]["strategic"]["status"] == "PAUSED"
    txt = X.render(res)
    assert "TACTICAL family" in txt and "NOT COMPUTED" in txt and "tactical goal-setting" in txt and "STRATEGIC family: PAUSED" in txt


def fake_block():
    ci = {"point": 0.9, "ci95": [0.8, 0.95]}
    cell = {"n": 40, "accuracy": ci, "kappa": {"point": 0.8, "ci95": [0.6, 0.9]}, "confusion_command_rows_plan_cols": [[9, 1, 0], [0, 20, 0], [0, 2, 8]],
            "rows": ["LEFT", "STRAIGHT", "RIGHT"], "cols": ["LEFT", "STRAIGHT", "RIGHT"], "per_command_recall": {}, "per_command_n": {}}
    b = T.not_computed("synthetic", 40)
    d = {"accuracy_delta": {"point": 0.03, "ci95": [-0.02, 0.08], "separated": False}}
    b["command_adherence"] = {"status": "COMPUTED", "n_tokens": 40, "arms": {r: {"turn_active": cell, "all_tokens": cell} for r in ("C1", "C0", "base")},
                              "paired_C1_minus_x": {"C1-C0": {"turn_active": d, "all_tokens": d}}, "random_proposal_floor": {"status": "NOT COMPUTED", "reason": "no dump", "n": 40},
                              "gates": {"G_FORMAT": {"ok": True}}}
    return b


def test_B12_block_reaches_the_json_and_the_report():
    blk = fake_block()
    res = positive(rehearsal="t", tactical=blk)
    assert res["tactical"] is blk or res["tactical"] == blk
    txt = X.render(res)
    assert "C1 command adherence, turn-active (headline)" in txt and "0.900 [0.800, 0.950]" in txt
    assert "no tracker in this number" in txt.split("TACTICAL family")[1].split("== read controls")[0]
    assert "NOT COMPUTED: no dump" in txt and "pooled" not in txt.lower()


def test_RED_B12_invalid_block_is_refused_by_the_read():
    b = fake_block()
    b["pooled_score"] = 0.5
    with pytest.raises(R.Refused):
        positive(rehearsal="t", tactical=b)
    b = fake_block()
    del b["strategic"]
    with pytest.raises(R.Refused):
        positive(rehearsal="t", tactical=b)


def test_RED_render_refuses_a_read_without_a_tactical_block():
    res = positive(rehearsal="t")
    del res["tactical"]
    with pytest.raises(X.RenderRefused):
        X.render(res)


def test_B12_render_prints_controls_and_missing_kinds():
    res = positive(rehearsal="t", require_kinds=("B10_determinism",))
    txt = X.render(res)
    assert "required controls MISSING: B10_determinism" in txt


# ------------------------------------------------------------------------------------------------------ MUTATION (delete the consuming line, the check must go RED)
def mutated(tmp_path, needle, replacement=""):
    src = open(os.path.join(HERE, "p5c_read.py"), encoding="utf-8").read()
    assert src.count(needle) == 1, f"mutation anchor must occur exactly once: {needle!r}"
    p = tmp_path / "p5c_read_mut.py"
    p.write_text(src.replace(needle, replacement), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("p5c_read_mut", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_MUTATION_dropping_the_tactical_line_is_caught(tmp_path):
    m = mutated(tmp_path, '    res["tactical"] = tactical\n')
    res = m.run(mk_rows(arms_with_shift(0.05)), "C1", "C0", "base", fam_ci=GOODFAM(), rehearsal="t", tactical=fake_block())
    assert "tactical" not in res                                   # the mutant loses it ...
    with pytest.raises(X.RenderRefused):
        X.render(res)                                              # ... and the renderer's check goes RED on it


def test_MUTATION_ignoring_required_kinds_is_caught(tmp_path):
    m = mutated(tmp_path, "(None if (not cs or missing_kinds) else True)", "(None if not cs else True)")
    res = m.run(mk_rows(arms_with_shift(0.05)), "C1", "C0", "base", fam_ci=GOODFAM(), controls=[], require_kinds=("B10_determinism",))
    assert res["verdict"]["verdict"] == "INCOMPLETE"               # no controls at all: still incomplete
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"pass": True}))
    res = m.run(mk_rows(arms_with_shift(0.05)), "C1", "C0", "base", fam_ci=GOODFAM(), controls=[str(f)], require_kinds=("B10_determinism",))
    assert res["verdict"]["verdict"] == "SEPARATED-SINGLE"         # the mutant lets a nameless control stand in for B10: the real module does not (test above)
    assert positive(controls=[str(f)], require_kinds=("B10_determinism",))["verdict"]["verdict"] == "INCOMPLETE"


def test_MUTATION_dropping_the_sealed_check_is_caught(tmp_path):
    m = mutated(tmp_path, '        cs.append({"file": sealed.get("file"), "kind": p5c_seal.KIND_SEALED, "pass": sealed_res["pass"]})\n')
    d = sealed_doc()
    d["arms"]["C1"]["model_final_sha256"] = H("9")
    f = tmp_path / "b10.json"
    f.write_text(json.dumps(b10_result()))
    res = m.run(mk_rows(arms_with_shift(0.05)), "C1", "C0", "base", fam_ci=GOODFAM(), controls=[str(f)], sealed={"doc": d, "file": "S"})
    assert res["verdict"]["verdict"] == "SEPARATED-SINGLE"         # the mutant reads a tampered seal as fine
    assert positive(controls=[str(f)], sealed={"doc": d, "file": "S"})["verdict"]["verdict"] == "INVALID"      # the real read does not
