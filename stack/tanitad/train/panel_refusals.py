"""refcv6 SPEC-v2 refusals 2 and 9 — the REPORT-time half of section 12.

Refusals 3, 4 and the split check live in :mod:`tanitad.train.prelaunch_v2` and
run BEFORE an arm. These two can only run AFTER one, against the panel it emits,
and saying so is part of the correction ``PREREG_REFCV6_V2.ERRATUM-1.md`` section
E4 made: section 12 called all ten "the launch checker", and two of them are not
launch-time checks at all.

**Refusal 2 — a control must read its known value EXACTLY.**
⛔ ``==``, not a tolerance. The programme's own record contains the reason: an
all-zero control read ``0.016340208`` against a base rate of ``0.016197554``
(+0.881 %) and the discrepancy was invisible for weeks because nobody had written
down what the control was SUPPOSED to read. A control whose expected value is not
declared is not a control -- it is a second measurement.

**Refusal 9 — no pooled tactical headline, and no headline quoting a token under
the n = 200 floor.**
⛔ 10 of the 22 behaviours sit under that floor (``refcv6_tactical.py``), and
pooling across tokens turns a handful of well-supported classes into an average
that reads as competence on all 22. Both are REPORTING defects: the numbers are
individually correct and the sentence they support is false.
"""
from __future__ import annotations

__all__ = [
    "ControlDidNotReadItsKnownValue", "TacticalReportRefused",
    "refuse_control_not_reading_known_value",
    "refuse_pooled_or_underpowered_tactical",
    "N_FLOOR",
]

#: The per-token support below which a number may be REPORTED but never quoted in
#: a headline. From `SPEC_REFCV6_V2` section 6 and `refcv6_tactical.py`.
N_FLOOR = 200


class ControlDidNotReadItsKnownValue(RuntimeError):
    """A control's observed value is not EXACTLY its declared known value."""


class TacticalReportRefused(RuntimeError):
    """A tactical headline pools across tokens, or quotes one under the floor."""


def refuse_control_not_reading_known_value(controls) -> dict:
    """Section 12 refusal 2.

    ``controls`` maps ``name -> {"expected": <value>, "observed": <value>}``.

    ⛔ Three ways to fail, and the second is the one that matters:

    1. **no controls at all** -- a panel without them cannot be checked;
    2. **a control with no declared ``expected``** -- REFUSED. An undeclared
       expectation is how a control becomes a second measurement: whatever it
       reads looks like the answer.
    3. **observed != expected** -- exactly, with no tolerance.
    """
    if not controls:
        raise ControlDidNotReadItsKnownValue(
            "the panel declares NO controls. REFUSING rather than reading "
            "'nothing to check': a panel without a control that reads a known "
            "value cannot distinguish a working instrument from a broken one.")
    undeclared, wrong = [], []
    for name, spec in sorted(controls.items()):
        if not isinstance(spec, dict) or "expected" not in spec:
            undeclared.append(name)
            continue
        if "observed" not in spec:
            undeclared.append(name)
            continue
        if spec["observed"] != spec["expected"]:
            wrong.append(f"{name}: observed {spec['observed']!r} != expected "
                         f"{spec['expected']!r}")
    if undeclared:
        raise ControlDidNotReadItsKnownValue(
            f"these controls do not declare BOTH an expected and an observed "
            f"value: {sorted(undeclared)}. A control whose expected value is not "
            f"written down is not a control -- it is a second measurement, and "
            f"whatever it reads will look like the answer.")
    if wrong:
        raise ControlDidNotReadItsKnownValue(
            f"control(s) did not read their known value EXACTLY: {wrong}. "
            f"No tolerance is applied here on purpose: an all-zero control once "
            f"read 0.016340208 against a base rate of 0.016197554 (+0.881 %) and "
            f"the gap went unnoticed because nothing said what it should read.")
    return {"n_controls": len(controls), "all_exact": True}


def refuse_pooled_or_underpowered_tactical(report, *, n_floor: int = N_FLOOR) -> dict:
    """Section 12 refusal 9.

    ``report`` carries ``per_token`` (``token -> {"ap": float, "n": int}``) and
    optionally ``headline`` (a list of token names the report leads with).

    ⛔ A ``pooled`` key anywhere in the report is refused outright: the four metric
    families are never pooled, and neither are the 22 behaviours.
    """
    pooled = [k for k in report if "pool" in k.lower()]
    if pooled:
        raise TacticalReportRefused(
            f"the report carries pooled key(s) {pooled}. Tactical behaviours are "
            f"reported PER CLASS, never pooled: pooling turns a handful of "
            f"well-supported classes into an average that reads as competence on "
            f"all 22.")
    per_token = report.get("per_token")
    if not per_token:
        raise TacticalReportRefused(
            "the report carries no `per_token` block. REFUSING rather than "
            "reading 'no tokens were scored': per-class is the only admissible "
            "form, so its absence is not a degenerate case, it is the defect.")
    missing_n = sorted(t for t, v in per_token.items()
                       if not isinstance(v, dict) or "n" not in v)
    if missing_n:
        raise TacticalReportRefused(
            f"these tokens report no `n`: {missing_n}. An AP without its support "
            f"cannot be placed against the floor, and an unplaceable number is "
            f"quoted as if it were placed.")
    headline = report.get("headline") or []
    under = sorted(t for t in headline
                   if int(per_token.get(t, {}).get("n", 0)) < n_floor)
    if under:
        detail = ", ".join(f"{t} (n={per_token[t]['n']})" for t in under)
        raise TacticalReportRefused(
            f"the headline quotes token(s) under the n = {n_floor} floor: "
            f"{detail}. They may be REPORTED -- they must not lead. 10 of the 22 "
            f"behaviours sit under this floor, so a headline is free to be built "
            f"almost entirely from the least-supported classes.")
    below = sorted(t for t, v in per_token.items() if int(v["n"]) < n_floor)
    return {"n_tokens": len(per_token), "n_below_floor": len(below),
            "below_floor": below, "headline_ok": True}
