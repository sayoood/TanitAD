"""The P-arm runner's ORDERING and GATE (`stack/scripts/p_runner.py`), pinned as literals.

⛔ The rule that matters most: `P0-REPLICATE` runs FIRST and is UNSKIPPABLE. `box3d_centre` has no
measured run-to-run floor, so a lever result without P0 cannot be read at all. A mutation that
lets a later arm run before P0 must turn `test_P0_is_UNSKIPPABLE_even_if_a_later_arm_already_ran`
RED (`stack/scripts/mutate_p_runner.py`).

Nothing here launches anything: every case is a directory of fabricated check files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
# the inner taniteval package, so a bare `pytest -q` behaves like the ambient one
sys.path.insert(0, str(ROOT.parent / "taniteval"))

import p_runner as PR                                     # noqa: E402

A7_ARMS = ("A7-IN-s0", "A7-RND-s0", "A7-IN-s1", "A7-RND-s1")


def _check(root: Path, arm: str, status: str, fname: str = "p_arm_check.json"):
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps({"status": status}), encoding="utf-8")


def _a7_complete(tmp: Path) -> Path:
    d = tmp / "a7"
    for a in A7_ARMS:
        _check(d, a, "VALID", "a7_arm_check.json")
    return d


# ------------------------------------------------------------------ ordering
def test_the_first_arm_is_the_REPLICATE_and_the_order_is_the_prereg_s():
    assert PR.ARMS[0]["name"] == "P0-REPLICATE" and PR.ARMS[0]["kind"] == "replicate"
    assert [a["name"] for a in PR.ARMS] == ["P0-REPLICATE", "P1-STEPS", "P3-SUPERVISION",
                                            "P4-DECODE", "P5-TRUNK"]


def test_with_nothing_done_the_next_arm_is_P0(tmp_path):
    assert PR.next_arm(tmp_path)["name"] == "P0-REPLICATE"


def test_P0_is_UNSKIPPABLE_even_if_a_later_arm_already_ran(tmp_path):
    """⛔ The rule a mutation must break: P1 VALID does not license skipping the floor."""
    _check(tmp_path, "P1-STEPS", "VALID")
    assert PR.next_arm(tmp_path)["name"] == "P0-REPLICATE"
    _check(tmp_path, "P3-SUPERVISION", "VALID")
    _check(tmp_path, "P4-DECODE", "VALID")
    _check(tmp_path, "P5-TRUNK", "VALID")
    assert PR.next_arm(tmp_path)["name"] == "P0-REPLICATE", (
        "every lever VALID and the FLOOR missing must still return P0")


def test_after_P0_the_order_follows_the_prereg(tmp_path):
    _check(tmp_path, "P0-REPLICATE", "VALID")
    assert PR.next_arm(tmp_path)["name"] == "P1-STEPS"
    _check(tmp_path, "P1-STEPS", "VALID")
    assert PR.next_arm(tmp_path)["name"] == "P3-SUPERVISION"
    for a in ("P3-SUPERVISION", "P4-DECODE", "P5-TRUNK"):
        _check(tmp_path, a, "VALID")
    assert PR.next_arm(tmp_path) is None


@pytest.mark.parametrize("state, expect", [
    ("VALID", "VALID"), ("INVALID", "INVALID"), ("nonsense", "INVALID")])
def test_arm_status_reads_the_artifact(tmp_path, state, expect):
    _check(tmp_path, "P0-REPLICATE", state)
    assert PR.arm_status(tmp_path, "P0-REPLICATE") == expect


def test_a_directory_without_a_check_is_PARTIAL_and_is_MOVED_not_deleted(tmp_path):
    (tmp_path / "P0-REPLICATE").mkdir()
    (tmp_path / "P0-REPLICATE" / "train.log").write_text("half a run", encoding="utf-8")
    assert PR.arm_status(tmp_path, "P0-REPLICATE") == "PARTIAL"
    dst = PR.move_aside(tmp_path, "P0-REPLICATE")
    assert dst.exists() and (dst / "train.log").read_text(encoding="utf-8") == "half a run"
    assert not (tmp_path / "P0-REPLICATE").exists()
    assert PR.arm_status(tmp_path, "P0-REPLICATE") == "ABSENT"


# ------------------------------------------------------------------ the gate
@pytest.mark.parametrize("probe, clear", [
    ("2500 8", True),        # exactly at both limits is CLEAR
    ("2501 8", False),       # one MiB over
    ("2500 7", False),       # one GB under
    ("3941 9", False),       # the PI's servers, 2026-09-20
    ("", False), ("x 8", False), ("2500", False),        # unreadable ⇒ INCONCLUSIVE
])
def test_gate_reads_KNOWN_values(probe, clear):
    ok, why = PR.gate_ok(probe)
    assert ok is clear and isinstance(why, str) and why


def test_an_unreadable_probe_says_INCONCLUSIVE_not_busy():
    assert "INCONCLUSIVE" in PR.gate_ok("nope")[1]


# ------------------------------------------------------------------ chaining behind A7
def test_a_running_A7_panel_BLOCKS_even_when_all_its_arms_are_VALID(tmp_path):
    d = _a7_complete(tmp_path)
    assert PR.a7_clear(d, panel_procs=0)[0] is True
    ok, why = PR.a7_clear(d, panel_procs=1)
    assert ok is False and "still running" in why


def test_A7_incomplete_BLOCKS(tmp_path):
    d = tmp_path / "a7"
    for a in A7_ARMS[:3]:
        _check(d, a, "VALID", "a7_arm_check.json")
    ok, why = PR.a7_clear(d)
    assert ok is False and "A7-RND-s1" in why


def test_plan_WAITS_while_A7_runs_even_with_a_clear_gate(tmp_path):
    out, a7 = tmp_path / "p", _a7_complete(tmp_path)
    d = PR.plan(out, a7, "0 32", panel_procs=2)
    assert d["action"] == "WAIT" and d["arm"] == "P0-REPLICATE" and "still running" in d["reason"]


def test_plan_RUNS_P0_when_A7_is_done_and_the_gate_is_clear(tmp_path):
    out, a7 = tmp_path / "p", _a7_complete(tmp_path)
    d = PR.plan(out, a7, "1200 12")
    assert d["action"] == "RUN" and d["arm"] == "P0-REPLICATE"


def test_plan_STOPS_on_an_INVALID_arm(tmp_path):
    out, a7 = tmp_path / "p", _a7_complete(tmp_path)
    _check(out, "P0-REPLICATE", "INVALID")
    d = PR.plan(out, a7, "1200 12")
    assert d["action"] == "STOP" and d["arm"] == "P0-REPLICATE"


def test_plan_is_DONE_when_every_arm_is_VALID(tmp_path):
    out, a7 = tmp_path / "p", _a7_complete(tmp_path)
    for a in PR.ARMS:
        _check(out, a["name"], "VALID")
    assert PR.plan(out, a7, "1200 12")["action"] == "DONE"
