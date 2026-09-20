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
    # ⛔ BOTH replicates come before ANY lever. P0-vs-A8 is seed + an unquantified code delta;
    # only |P0 − P0b| on the same pinned tree is the seed floor, and every lever is judged
    # against it — so a lever that ran before the floor existed could not be read.
    assert [a["name"] for a in PR.ARMS] == ["P0-REPLICATE", "P0B-REPLICATE", "P1-STEPS",
                                            "P3-SUPERVISION", "P4-DECODE", "P5-TRUNK"]
    assert [a["kind"] for a in PR.ARMS[:2]] == ["replicate", "replicate"]
    # three DISTINCT seeds: A8 is 0, so the two replicates must be 1 and 2, never 0
    seeds = [int(a["flags"][a["flags"].index("--seed") + 1]) for a in PR.ARMS[:2]]
    assert seeds == [1, 2] and 0 not in seeds, "a replicate sharing A8's seed is not a replicate"


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
    assert PR.next_arm(tmp_path)["name"] == "P0B-REPLICATE"
    _check(tmp_path, "P0B-REPLICATE", "VALID")
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


# ============================================================ the P0-ONLY authorisation bound
# ⛔ The Master Mind authorised P0-REPLICATE ONLY (2026-09-20): chained behind A7, unchanged gate,
# and it "must not roll on to P1 unattended". These pin that boundary. A mutation that lets the
# runner continue past the bound must turn test_the_bound_REFUSES_the_next_arm RED.
BOUND = "P0-REPLICATE"


def test_without_a_bound_nothing_changes(tmp_path):
    arm, why = PR.bounded_next(tmp_path, None)
    assert arm["name"] == "P0-REPLICATE" and why == "unbounded"


def test_inside_the_bound_P0_still_runs(tmp_path):
    arm, why = PR.bounded_next(tmp_path, BOUND)
    assert arm["name"] == "P0-REPLICATE" and "within the bound" in why


def test_the_bound_REFUSES_the_next_arm(tmp_path):
    """⛔ The rule a mutation must break: P0 VALID ends the authorisation. P1 must NOT start."""
    _check(tmp_path, "P0-REPLICATE", "VALID")
    assert PR.next_arm(tmp_path)["name"] == "P0B-REPLICATE", "unbounded, P0b would be next"
    arm, why = PR.bounded_next(tmp_path, BOUND)
    assert arm is None and "BOUND REACHED" in why


def test_an_unknown_bound_runs_NOTHING(tmp_path):
    arm, why = PR.bounded_next(tmp_path, "P9-NONSENSE")
    assert arm is None and "unknown bound" in why


def test_plan_is_DONE_not_RUN_once_the_bound_is_reached(tmp_path):
    out, a7 = tmp_path / "p", _a7_complete(tmp_path)
    _check(out, "P0-REPLICATE", "VALID")
    d = PR.plan(out, a7, "0 32", stop_after=BOUND)
    assert d["action"] == "DONE" and "BOUND REACHED" in d["reason"]
    assert PR.plan(out, a7, "0 32")["action"] == "RUN", "unbounded, the same state would RUN"


# ------------------------------------------------------------------ the replicate's command line
def test_build_argv_copies_the_base_VERBATIM_except_out_and_seed():
    base = ["--arm", "hier", "--seed", "0", "--steps", "5000", "--out", "OLD",
            "--w-box3d", "1.0"]
    got = PR.build_argv(base, seed=1, out="NEW")
    assert got[:4] == ["--out", "NEW", "--seed", "1"]
    assert got[4:] == ["--arm", "hier", "--steps", "5000", "--w-box3d", "1.0"]
    assert "OLD" not in got and got.count("--seed") == 1 and got.count("--out") == 1


def test_build_argv_ADDS_NOTHING():
    """⛔ An added flag makes the arm a different experiment and the difference stops being
    a floor — in particular no --eval-window-dump."""
    base = ["--arm", "hier", "--seed", "0", "--out", "OLD"]
    got = PR.build_argv(base, seed=1, out="NEW")
    assert set(got) - set(base) - {"NEW", "1"} == set()
    assert len(got) == len(base)


# ------------------------------------------------------------------ preflight
def test_preflight_REFUSES_an_unauthorised_arm(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("x", encoding="utf-8")
    ok, why = PR.preflight("P1-STEPS", "P0-REPLICATE", {"trainer": (f, None)})
    assert ok is False and any("NOT among the authorised arms" in r for r in why)


def test_preflight_REFUSES_when_no_arm_was_authorised(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("x", encoding="utf-8")
    ok, why = PR.preflight("P0-REPLICATE", None, {"trainer": (f, None)})
    assert ok is False and any("launches nothing" in r for r in why)


def test_preflight_REFUSES_a_missing_or_wrong_input(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("x", encoding="utf-8")
    real = PR.md5(f)
    ok, _ = PR.preflight("P0-REPLICATE", "P0-REPLICATE", {"trainer": (f, real)})
    assert ok is True, "the right bytes must pass"
    bad, why = PR.preflight("P0-REPLICATE", "P0-REPLICATE", {"trainer": (f, "0" * 32)})
    assert bad is False and any("md5" in r for r in why)
    miss, why2 = PR.preflight("P0-REPLICATE", "P0-REPLICATE",
                              {"labels": (tmp_path / "nope.gz", None)})
    assert miss is False and any("MISSING" in r for r in why2)


# ================================================ the TWO-replicate bound (MM, 2026-09-20)
# ⛔ The authorisation is now P0 -> P0b, then STOP. P1 must not start unattended. Two INDEPENDENT
# locks guard it: `--stop-after` (the ordering bound) and `--authorise-arm` (the explicit list).
# An arm must pass BOTH, so a mistake in either one alone cannot start a lever.
BOUND2 = "P0B-REPLICATE"


def test_the_two_replicate_bound_runs_P0_then_P0b_then_STOPS(tmp_path):
    a, _w = PR.bounded_next(tmp_path, BOUND2)
    assert a["name"] == "P0-REPLICATE"
    _check(tmp_path, "P0-REPLICATE", "VALID")
    a, _w = PR.bounded_next(tmp_path, BOUND2)
    assert a["name"] == "P0B-REPLICATE", "the floor needs BOTH replicates"
    _check(tmp_path, "P0B-REPLICATE", "VALID")
    a, why = PR.bounded_next(tmp_path, BOUND2)
    assert a is None and "BOUND REACHED" in why
    assert PR.next_arm(tmp_path)["name"] == "P1-STEPS", (
        "unbounded, P1 would start here — the bound is what stops it")


def test_plan_is_DONE_after_BOTH_replicates_not_RUN(tmp_path):
    out, a7 = tmp_path / "p", _a7_complete(tmp_path)
    _check(out, "P0-REPLICATE", "VALID")
    assert PR.plan(out, a7, "0 32", stop_after=BOUND2)["arm"] == "P0B-REPLICATE"
    _check(out, "P0B-REPLICATE", "VALID")
    d = PR.plan(out, a7, "0 32", stop_after=BOUND2)
    assert d["action"] == "DONE" and "BOUND REACHED" in d["reason"]
    assert PR.plan(out, a7, "0 32")["action"] == "RUN", "unbounded, the same state would RUN"


def test_the_AUTHORISE_LIST_is_a_SECOND_lock_the_bound_does_not_cover(tmp_path):
    """⛔ P1 sits inside no bound here, but the list still refuses it — and P0b, which the
    bound DOES allow, is refused when the list omits it. Two locks, independently."""
    f = tmp_path / "t.py"
    f.write_text("x", encoding="utf-8")
    both = ["P0-REPLICATE", "P0B-REPLICATE"]
    assert PR.preflight("P0B-REPLICATE", both, {"t": (f, None)})[0] is True
    ok, why = PR.preflight("P0B-REPLICATE", ["P0-REPLICATE"], {"t": (f, None)})
    assert ok is False and any("NOT among the authorised arms" in r for r in why)
    assert PR.preflight("P1-STEPS", both, {"t": (f, None)})[0] is False


def test_an_empty_authorisation_list_launches_NOTHING(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("x", encoding="utf-8")
    for empty in (None, []):
        ok, why = PR.preflight("P0-REPLICATE", empty, {"t": (f, None)})
        assert ok is False and any("launches nothing" in r for r in why)


def test_the_two_replicates_differ_ONLY_in_seed():
    """⛔ |P0 - P0b| is the seed floor only if the seed is the sole difference. Built from the
    same base argv, the two command lines must differ in exactly the seed and the out dir."""
    base = ["--arm", "hier", "--seed", "0", "--steps", "5000", "--out", "OLD", "--w-box3d", "1.0"]
    p0 = PR.build_argv(base, seed=1, out="A")
    p0b = PR.build_argv(base, seed=2, out="B")
    assert p0[:4] == ["--out", "A", "--seed", "1"] and p0b[:4] == ["--out", "B", "--seed", "2"]
    assert p0[4:] == p0b[4:], "everything after out/seed must be IDENTICAL between replicates"
    assert len(p0) == len(p0b) == len(base)
