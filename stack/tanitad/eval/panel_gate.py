"""Order-invariant panel verdicts: the baseline is NAMED, never positional.

⛔⛔ THE DEFECT THIS EXISTS TO KILL (D2, 2026-09-06). Two analysis panels in this
programme picked their reference arm as ``present[0]`` — *whoever is listed
first* — and then rejected the others relative to it. That one line produces
three failures, none of which announces itself:

1. **Verdicts flip with run order.** Reorder the arm list, or simply run before
   the reference arm's checkpoint has landed, and a DIFFERENT arm silently
   becomes the baseline. The printed column header still said ``vs rdw8``, so
   the table asserted a comparison it had not made.
2. **The first arm never gets a verdict at all.** ``for arm in present[1:]``
   skips it, so the reference arm has NO row in the verdict map — and to every
   downstream summariser an absent rejection is indistinguishable from a pass.
   MEASURED: an arm was recorded as its own baseline.
3. **``None`` is falsy.** Once a "cannot rule" state exists, the natural
   ``'REJECTED' if v else 'passes gate'`` renders it as **passes gate**. A
   three-valued verdict needs a three-valued reader.

⇒ Everything here is keyed by NAME. An arm that is its own baseline, and every
arm in a run whose baseline is absent, reads **NO VERDICT** — never a pass. No
substitute baseline is ever promoted: a panel that cannot rule says so, which is
the same principle ``rank_gate_capacity`` enforces for the rank gate (*a
criterion that CANNOT RULE is worse than no criterion, because the report looks
populated*).
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

__all__ = ["select_baseline", "relative_verdicts", "render_verdict",
           "NO_VERDICT", "RULED"]

NO_VERDICT = "NO VERDICT"
RULED = "RULED"


def select_baseline(present: Sequence[str], baseline: str) -> dict:
    """Resolve a panel's reference arm BY NAME.

    Returns a record that always carries the baseline's identity, so a reader
    of the emitted artifact can never have to infer it from row order.
    ``present_baseline`` False means the relative clauses cannot rule — it does
    NOT mean "use another arm".
    """
    if not baseline or not isinstance(baseline, str):
        raise ValueError("baseline must be a non-empty arm NAME; a positional "
                         "baseline (present[0]) is the defect this replaces")
    have = baseline in tuple(present)
    return {"baseline_arm": baseline,
            "baseline_present": bool(have),
            "baseline_is_positional": False,
            "arms_present": list(present),
            "reason": (None if have else
                       f"baseline arm {baseline!r} is absent from this run; no "
                       f"substitute baseline is promoted, so every RELATIVE "
                       f"verdict is {NO_VERDICT}. Absolute per-arm numbers are "
                       f"order-free and still stand.")}


def relative_verdicts(arms: Mapping[str, dict], present: Iterable[str],
                      baseline: str,
                      rule: Callable[[dict, dict], tuple[bool, str]],
                      *, extra: Callable[[dict], dict] | None = None) -> dict:
    """One entry per PRESENT arm — the baseline included — three-valued.

    ``rule(arm_record, baseline_record) -> (rejected, reason)`` carries the
    panel's own criterion; this function owns only *who is compared to whom*
    and *what an unrulable comparison reads as*.

    ⭐ ORDER-INVARIANT BY CONSTRUCTION: nothing here indexes ``present``. Two
    runs whose ``present`` lists differ only in order produce equal dicts.
    """
    sel = select_baseline(list(present), baseline)
    base_rec = arms.get(baseline) if sel["baseline_present"] else None
    out: dict[str, dict] = {}
    for arm in present:
        rec = arms.get(arm, {})
        entry: dict = {"baseline_arm": baseline}
        if extra is not None:
            entry |= extra(rec)
        if arm == baseline:
            entry |= {"REJECTED_by_kill_gate": None, "status": NO_VERDICT,
                      "reason": "this arm IS the baseline — the relative "
                                "clause cannot be evaluated against itself. "
                                "Absence of a rejection is NOT a pass."}
        elif base_rec is None:
            entry |= {"REJECTED_by_kill_gate": None, "status": NO_VERDICT,
                      "reason": sel["reason"]}
        else:
            rejected, why = rule(rec, base_rec)
            entry |= {"REJECTED_by_kill_gate": bool(rejected),
                      "status": RULED, "reason": why}
        out[arm] = entry
    return out


def render_verdict(verdicts: Mapping[str, dict],
                   suffix: Callable[[dict], str] | None = None) -> str:
    """Human-readable summary that is THREE-VALUED.

    ⛔ ``None`` is not ``False``. The expression this replaces was
    ``'REJECTED' if v['REJECTED_by_kill_gate'] else 'passes gate'``, which
    renders an unrulable arm as a PASS.
    """
    parts = []
    for arm, v in verdicts.items():
        rj = v.get("REJECTED_by_kill_gate")
        if rj is None:
            head = f"{NO_VERDICT} — {v.get('reason', '')}"
        elif rj:
            head = f"REJECTED — {v.get('reason', '')}"
        else:
            head = "passes gate"
        parts.append(f"{arm}: {head}{suffix(v) if suffix else ''}")
    return "; ".join(parts)
