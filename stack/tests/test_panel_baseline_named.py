"""D2 -- a panel's baseline must be NAMED, so verdicts cannot flip with run order.

THE DEFECT. ``full_panel.py`` and ``rolled_predict.py`` both took the reference
arm as ``present[0]`` -- whoever is listed first -- and then ruled the rest
against it with ``for arm in present[1:]``. Three consequences, all silent:

  1. reorder the arms and a DIFFERENT arm becomes the baseline, while the
     printed header still claims "vs rdw8";
  2. the first arm gets NO verdict row at all, and a missing rejection reads as
     a pass to anything summarising the JSON;
  3. once "cannot rule" exists, ``None`` is falsy and renders as "passes gate".

THE DELIBERATE REGRESSION. ``_OLD_POSITIONAL_frozen`` below is a verbatim
re-implementation of the defect. Every regression test runs the SAME arms in two
different orders and asserts the old code disagrees with itself while the
repaired code does not.

ASCII-only strings (cp1252 box).
"""
from __future__ import annotations

import pytest

from tanitad.eval.panel_gate import (NO_VERDICT, RULED, relative_verdicts,
                                     render_verdict, select_baseline)

BASELINE = "rdw8"

# participation_val / participation_heldout24 chosen so that `o8w1p0` is
# REJECTED against rdw8 (rank falls on BOTH held-out sets) and `o7w1p0` is not.
ARMS = {
    "rdw8":   {"participation_val": 10.0, "participation_heldout24": 10.0,
               "predictor": {"1": {"z": 5.0}, "2": {"z": 0.5}}},
    "o7w1p0": {"participation_val": 12.0, "participation_heldout24": 11.0,
               "predictor": {"1": {"z": 4.0}, "2": {"z": 3.0}}},
    "o8w1p0": {"participation_val": 4.0, "participation_heldout24": 3.0,
               "predictor": {"1": {"z": 4.5}, "2": {"z": 0.2}}},
}
ORDER_A = ["rdw8", "o7w1p0", "o8w1p0"]
ORDER_B = ["o8w1p0", "o7w1p0", "rdw8"]


def _rule(r, b):
    rank_dead = (r["participation_val"] < b["participation_val"]
                 and r["participation_heldout24"] < b["participation_heldout24"])
    pred_dead = r["predictor"]["1"]["z"] < 2.0
    return (rank_dead or pred_dead,
            "rank fell on BOTH held-out sets" if rank_dead else
            "predictor cos z<2 at h=1" if pred_dead else "passes the gate")


def _OLD_POSITIONAL_frozen(arms: dict, present: list[str]) -> dict:
    """VERBATIM the pre-repair kill-gate: baseline = present[0], skip it."""
    base = arms.get(present[0], {})
    verdicts = {}
    for arm in present[1:]:
        r = arms[arm]
        rank_dead = (r["participation_val"] < base["participation_val"]
                     and r["participation_heldout24"] < base["participation_heldout24"])
        pred_dead = r["predictor"]["1"]["z"] < 2.0
        verdicts[arm] = {
            "REJECTED_by_kill_gate": bool(rank_dead or pred_dead),
            "reason": ("rank fell on BOTH held-out sets" if rank_dead else
                       "predictor cos z<2 at h=1" if pred_dead else
                       "passes the gate")}
    return verdicts


def _new(present):
    return relative_verdicts(ARMS, present, BASELINE, _rule)


# --------------------------------------------------------------------------
# THE REGRESSION TESTS -- these FAIL on the unfixed tree
# --------------------------------------------------------------------------
def test_REGRESSION_old_code_disagrees_with_itself_across_run_order():
    """Establishes the defect is real before asserting the cure."""
    a = _OLD_POSITIONAL_frozen(ARMS, ORDER_A)
    b = _OLD_POSITIONAL_frozen(ARMS, ORDER_B)
    assert a != b, ("if the positional gate agreed across orders there would "
                    "be nothing to fix")
    # concretely: rdw8 has no row at all in order A ...
    assert BASELINE not in a
    # ... and in order B it is RULED against o8w1p0, an arm that order A
    # rejected. The baseline silently became the thing it was meant to judge.
    assert BASELINE in b
    assert a["o8w1p0"]["REJECTED_by_kill_gate"] is True
    assert b[BASELINE]["REJECTED_by_kill_gate"] is False


def test_REGRESSION_named_baseline_is_order_invariant():
    """The cure: same arms, two orders, identical per-arm verdicts."""
    a, b = _new(ORDER_A), _new(ORDER_B)
    assert set(a) == set(b) == set(ARMS)
    for arm in ARMS:
        assert a[arm] == b[arm], f"verdict for {arm} moved with run order"


def test_REGRESSION_every_arm_gets_an_entry_including_the_baseline():
    v = _new(ORDER_A)
    assert BASELINE in v, "the baseline arm must not be silently skipped"
    assert v[BASELINE]["REJECTED_by_kill_gate"] is None
    assert v[BASELINE]["status"] == NO_VERDICT
    assert "IS the baseline" in v[BASELINE]["reason"]
    assert v["o8w1p0"]["status"] == RULED
    assert v["o8w1p0"]["REJECTED_by_kill_gate"] is True
    assert v["o7w1p0"]["REJECTED_by_kill_gate"] is False


def test_REGRESSION_absent_baseline_is_NO_VERDICT_never_a_substitute():
    """No arm is promoted to baseline just because it sorted first."""
    present = ["o7w1p0", "o8w1p0"]
    v = relative_verdicts(ARMS, present, BASELINE, _rule)
    assert set(v) == set(present)
    for arm in present:
        assert v[arm]["REJECTED_by_kill_gate"] is None
        assert v[arm]["status"] == NO_VERDICT
        assert "absent" in v[arm]["reason"]
    # and it must NOT have silently ruled o8w1p0 against o7w1p0
    assert all(x["baseline_arm"] == BASELINE for x in v.values())


def test_REGRESSION_none_does_not_render_as_passes_gate():
    """`None` is falsy; the old renderer turned an unrulable arm into a pass."""
    v = _new(ORDER_A)
    text = render_verdict(v)
    assert f"{BASELINE}: {NO_VERDICT}" in text
    # the exact old expression, reproduced, to show what it would have said
    old_text = "; ".join(
        f"{a}: {'REJECTED' if x['REJECTED_by_kill_gate'] else 'passes gate'}"
        for a, x in v.items())
    assert f"{BASELINE}: passes gate" in old_text, (
        "the frozen renderer must show the false pass, else this is not a "
        "regression test")
    assert f"{BASELINE}: passes gate" not in text


def test_baseline_identity_travels_in_the_record():
    sel = select_baseline(ORDER_B, BASELINE)
    assert sel["baseline_arm"] == BASELINE
    assert sel["baseline_present"] is True
    assert sel["baseline_is_positional"] is False
    absent = select_baseline(["o7w1p0"], BASELINE)
    assert absent["baseline_present"] is False
    assert "no substitute baseline is promoted" in absent["reason"]


def test_a_positional_baseline_is_refused_outright():
    with pytest.raises(ValueError, match="arm NAME"):
        select_baseline(ORDER_A, "")


# --------------------------------------------------------------------------
# the two panels on disk must actually USE the named baseline
# --------------------------------------------------------------------------
@pytest.mark.parametrize("script", ["full_panel.py", "rolled_predict.py"])
def test_the_panels_on_disk_declare_a_named_baseline(script):
    """A source check, deliberately paired with a control that must read
    non-zero -- a grep that returns 0 because it could not read the file is
    indistinguishable from a genuine absence."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    p = (root / "TanitAD Research Lab" / "Architecture & Inference" /
         "Research" / "2026-08-19-simwam-analysis" / "code" / script)
    if not p.is_file():
        pytest.skip(f"{script} not present in this checkout")
    body = p.read_text(encoding="utf-8")
    assert "ARMS" in body, "CONTROL: the file was not readable"      # must hit
    assert 'BASELINE = "rdw8"' in body, f"{script} has no named baseline"
    # ⚠️ COMMENTS ARE NOT CODE. These files DOCUMENT the old positional
    # baseline in prose, so a naive substring search matches the explanation
    # and reports the defect it is describing. Strip comment lines first.
    code = [ln for ln in body.splitlines()
            if not ln.lstrip().startswith("#")]
    joined = "\n".join(code)
    assert "present[1:]" not in joined, (
        f"{script} still skips the first arm (present[1:])")
    # the only admissible surviving present[0] is the target-populating anchor,
    # which is value-identical across arms and asserted to be so.
    residual = joined.replace("present[0] if present else None", "")
    assert "present[0]" not in residual, (
        f"{script} still selects a baseline positionally")
