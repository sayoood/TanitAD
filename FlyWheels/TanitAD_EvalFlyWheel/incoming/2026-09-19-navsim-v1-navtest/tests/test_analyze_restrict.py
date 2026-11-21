"""A RESTRICTED read is paired arithmetic — and it may never be compared to the paper.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_analyze_restrict.py

The bridge banks resumable parts, so a partial navtest pass is scorable on the tokens it holds.
That is a legitimate PAIRED comparison (every arm read on exactly those tokens) and an
ILLEGITIMATE comparison against arXiv 2406.15349, whose numbers are the 12,146-token split.
⛔ The second is the programme's most expensive error class — a true number quoted outside its
scope — so the suppression is a GUARD, and it is tested with a control that must read the
opposite: the same call WITHOUT the restriction has to produce the published verdicts.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / "code"
COLS = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
        "comfort", "ego_progress", "driving_direction_compliance"]


def _mod():
    spec = importlib.util.spec_from_file_location("w3_analyze", CODE / "analyze_navtest.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_analyze"] = m
    spec.loader.exec_module(m)
    return m


def _csv(raw: Path, arm: str, sfx: str, scores: dict):
    d = raw / f"{arm}_{sfx}"
    d.mkdir(parents=True, exist_ok=True)
    head = ["token", "valid", "score"] + COLS
    lines = [",".join(head)]
    for t, s in scores.items():
        lines.append(",".join([t, "True", f"{s}"] + ["1.0"] * len(COLS)))
    lines.append(",".join(["average", "True", f"{sum(scores.values()) / len(scores)}"]
                          + ["1.0"] * len(COLS)))
    (d / f"{arm}_{sfx}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    M = _mod()
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(M, "RAW", str(raw))
    # t1 and t2 are scored by every arm; t3 only by CV.
    _csv(raw, "CV", "navtest", {"t1": 0.2, "t2": 0.4, "t3": 0.9})
    _csv(raw, "HUMAN", "navtest", {"t1": 0.8, "t2": 0.9})
    _csv(raw, "STOP", "navtest", {"t1": 0.6, "t2": 0.6})
    return M, raw


def _out(raw, tag=""):
    return json.load(open(raw / f"analysis_navtest{('_' + tag) if tag else ''}.json",
                          encoding="utf-8"))


def test_unrestricted_navtest_still_carries_the_published_verdicts(rig):
    """⭐ THE CONTROL. If this one did not produce the verdicts, the suppression test below would
    pass for the wrong reason — a check that is green because nothing ran at all."""
    M, raw = rig
    assert M.main(["--split-run", "navtest"]) == 0
    cv = _out(raw)["arms"]["CV"]
    assert "vs_paper_table1" in cv
    assert "published_verdicts" not in cv
    assert cv["n"] == 3


def test_a_restriction_suppresses_every_published_verdict(rig, tmp_path):
    M, raw = rig
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"tokens": ["t1", "t2"]}), encoding="utf-8")
    assert M.main(["--split-run", "navtest", "--restrict-tokens", str(sel), "--tag", "sub"]) == 0
    d = _out(raw, "sub")
    for arm in ("CV", "HUMAN", "STOP"):
        assert "vs_paper_table1" not in d["arms"][arm]
        assert "vs_hf_leaderboard_INHERITED" not in d["arms"][arm]
        assert d["arms"][arm]["published_verdicts"].startswith("SUPPRESSED")


def test_the_restriction_changes_the_arithmetic_to_the_literal_paired_mean(rig, tmp_path):
    """⛔ A flag that only labels the output would pass the test above. The numbers must move:
    CV over {t1, t2} is exactly (0.2 + 0.4)/2 = 0.30, i.e. 30.0 ×100 — a LITERAL, not an
    expression over the code under test. Unrestricted CV includes t3 and reads 50.0."""
    M, raw = rig
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(["t1", "t2"]), encoding="utf-8")       # a bare list, also accepted
    assert M.main(["--split-run", "navtest", "--restrict-tokens", str(sel), "--tag", "sub"]) == 0
    assert M.main(["--split-run", "navtest"]) == 0
    assert _out(raw, "sub")["arms"]["CV"]["x100"]["PDMS"] == 30.0
    assert _out(raw)["arms"]["CV"]["x100"]["PDMS"] == 50.0
    assert _out(raw, "sub")["arms"]["CV"]["n"] == 2


def test_a_token_missing_from_one_arm_is_counted_not_silently_dropped(rig, tmp_path):
    """t3 exists only for CV. A paired read must exclude it — and SAY it excluded it."""
    M, raw = rig
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(["t1", "t2", "t3"]), encoding="utf-8")
    assert M.main(["--split-run", "navtest", "--restrict-tokens", str(sel), "--tag", "sub"]) == 0
    r = _out(raw, "sub")["restricted"]
    assert r["n_requested"] == 3
    assert r["n_scored_by_every_arm"] == 2
    assert r["n_requested_not_in_every_arm"] == 1


def test_a_restriction_with_no_overlap_refuses_rather_than_reporting_an_empty_mean(rig, tmp_path):
    M, raw = rig
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(["nope"]), encoding="utf-8")
    assert M.main(["--split-run", "navtest", "--restrict-tokens", str(sel), "--tag", "sub"]) == 2
    assert not (raw / "analysis_navtest_sub.json").exists()


def test_the_output_tag_keeps_a_subset_read_off_the_full_split_artifact(rig, tmp_path):
    """A subset overwriting analysis_navtest.json is how a 200-token number becomes 'the result'."""
    M, raw = rig
    assert M.main(["--split-run", "navtest"]) == 0
    full = _out(raw)["arms"]["CV"]["x100"]["PDMS"]
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(["t1"]), encoding="utf-8")
    assert M.main(["--split-run", "navtest", "--restrict-tokens", str(sel), "--tag", "sub"]) == 0
    assert _out(raw)["arms"]["CV"]["x100"]["PDMS"] == full      # untouched
    assert _out(raw, "sub")["arms"]["CV"]["x100"]["PDMS"] == 20.0
