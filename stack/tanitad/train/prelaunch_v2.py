"""refcv6 SPEC-v2 pre-launch refusals 3 and 4 -- the two that did not exist.

``PREREG_REFCV6_V2.ERRATUM-1.md`` section E4 measured the ten refusals section 12
promised: four existed, three were wired to the 2026-09-10 arm set, and **three
did not exist at all**. This module is two of those three.

⛔ **Refusal 4 is the consequential one.** A hyper-parameter chosen on the split it
is later scored on does not crash, does not look wrong, and does not read as a
defect in any curve -- it **manufactures a positive**. Every other refusal in
section 12 catches something that would otherwise be visible.

⭐ **What this module can and cannot do, stated up front so it is not mistaken for
more.** *"Was this threshold fitted on the scored split?"* is not decidable from a
panel. What IS decidable, and is what is enforced here:

1. every tuned quantity **declares** the split it was fitted on, and an
   **undeclared** one is REFUSED -- not assumed innocent. Silence is the defect:
   it is how a tuned threshold passes as a constant.
2. no declared split is the scored one;
3. the scored split is **disjoint** from fit and val, MEASURED on ids rather than
   asserted.

⇒ a run can still lie in its declaration. It can no longer pass by SAYING NOTHING,
which is the failure mode the record actually contains.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "HypothesisNotRegistered", "ScoredSplitLeak",
    "registered_hypotheses", "refuse_unregistered_hypotheses",
    "refuse_scored_split_leak", "refuse_overlapping_splits",
    "prelaunch_v2",
]

#: A hypothesis id as ``GOALS_AND_CLAIMS.md`` writes them: ``E-`` or ``D-``, then
#: upper-case / digit / hyphen groups. ⛔ Word-anchored, so ``E-REFCV6V2-TRUNKY``
#: does not satisfy a claim of ``E-REFCV6V2-TRUNK``.
_ID = re.compile(r"\b([ED]-[A-Z0-9]+(?:-[A-Z0-9]+)*)\b")

#: The keys a panel may use to say where a quantity was fitted. Several spellings
#: are accepted because the record already uses several; ⛔ what is refused is the
#: ABSENCE of all of them, never the choice among them.
_FITTED_KEYS = ("fitted_on", "fit_split", "selected_on", "tuned_on")


class HypothesisNotRegistered(RuntimeError):
    """An arm names a hypothesis id that ``GOALS_AND_CLAIMS.md`` does not carry."""


class ScoredSplitLeak(RuntimeError):
    """A tuned quantity was selected on the split it is scored on -- or will not
    say where it was selected, which is not distinguishable from the same thing."""


def registered_hypotheses(goals_md) -> set:
    """Every hypothesis id ``GOALS_AND_CLAIMS.md`` registers.

    ⛔ RAISES on a missing or idless file rather than returning an empty set. An
    empty set would make :func:`refuse_unregistered_hypotheses` refuse EVERY id,
    which reads as *"the guard is broken"* rather than *"the registry was not
    found"* -- and a guard that refuses everything is switched off within a day.
    """
    p = Path(goals_md)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} does not exist. Refusing to return an empty registry: "
            f"'no ids are registered' and 'the registry could not be read' are "
            f"different facts and must not be collapsed.")
    ids = set(_ID.findall(p.read_text(encoding="utf-8", errors="replace")))
    if not ids:
        raise ValueError(
            f"{p} carries no hypothesis id at all -- refusing to treat that as "
            f"a registry.")
    return ids


def refuse_unregistered_hypotheses(arm_ids, goals_md) -> dict:
    """Section 12 refusal 3. Every id an arm claims must ALREADY be registered."""
    known = registered_hypotheses(goals_md)
    claimed = [str(i) for i in arm_ids]
    if not claimed:
        raise HypothesisNotRegistered(
            "this arm names NO hypothesis id. An unnamed arm cannot be refuted, "
            "because there is nothing written down that it would refute.")
    missing = sorted(set(claimed) - known)
    if missing:
        raise HypothesisNotRegistered(
            f"{missing} are not registered in {Path(goals_md).name}. Register "
            f"the hypothesis BEFORE the arm runs -- a claim written after its "
            f"own result is not a pre-registration. "
            f"({len(known)} ids are registered.)")
    return {"claimed": sorted(claimed), "n_registered": len(known)}


def refuse_scored_split_leak(panel, *, scored: str = "test") -> dict:
    """Section 12 refusal 4. Every tuned quantity declares its split, and none of
    them is ``scored``.

    ``panel["tuned"]`` maps ``name -> {"value": ..., <one of _FITTED_KEYS>: ...}``.
    """
    tuned = panel.get("tuned")
    if tuned is None:
        raise ScoredSplitLeak(
            "the panel carries no `tuned` block. REFUSING rather than reading "
            "'nothing was tuned': a panel that does not enumerate its tuned "
            "quantities cannot be checked, and 'not checked' and 'checked and "
            "fine' are indistinguishable from here. An arm that genuinely tuned "
            "nothing declares `tuned: {}`.")
    undeclared, leaked = [], []
    for name, spec in sorted(tuned.items()):
        if not isinstance(spec, dict):
            undeclared.append(name)
            continue
        where = next((spec[k] for k in _FITTED_KEYS if spec.get(k)), None)
        if where is None:
            undeclared.append(name)
        elif str(where) == str(scored):
            leaked.append(f"{name} (fitted on {where!r})")
    if leaked:
        raise ScoredSplitLeak(
            f"fitted on the SCORED split: {leaked}. This does not crash and does "
            f"not look wrong in any curve -- it MANUFACTURES a positive. Refit on "
            f"the fit split, or score on a split this quantity has never seen.")
    if undeclared:
        raise ScoredSplitLeak(
            f"these tuned quantities do not say where they were fitted: "
            f"{sorted(undeclared)}. Silence is REFUSED, not assumed innocent -- "
            f"an undeclared threshold is exactly how a tuned quantity passes as "
            f"a constant. Declare one of {list(_FITTED_KEYS)}.")
    return {"n_tuned": len(tuned), "scored": scored}


def refuse_overlapping_splits(splits, *, scored: str = "test") -> dict:
    """The measurable half: the scored split shares no id with any other.

    ⛔ An EMPTY scored split RAISES rather than reading 'no overlap' -- the same
    trap the coverage gate names: an empty intersection from a file that could not
    be read is indistinguishable from a genuine absence.
    """
    if scored not in splits:
        raise ScoredSplitLeak(f"no {scored!r} split was declared")
    test = set(splits[scored])
    if not test:
        raise ScoredSplitLeak(
            f"the {scored!r} split is EMPTY. Refusing rather than reporting zero "
            f"overlap -- an empty set overlaps nothing by construction.")
    report = {"n_" + scored: len(test)}
    for other in sorted(splits):
        if other == scored:
            continue
        shared = test & set(splits[other])
        report[f"shared_with_{other}"] = len(shared)
        if shared:
            raise ScoredSplitLeak(
                f"{len(shared)} id(s) appear in BOTH {other!r} and the scored "
                f"{scored!r} split. Every number from this arm is scored on rows "
                f"it was fitted on.")
    return report


def prelaunch_v2(*, arm_ids, goals_md, panel, splits, scored: str = "test") -> dict:
    """All three, cheapest failure first."""
    return {
        "hypotheses": refuse_unregistered_hypotheses(arm_ids, goals_md),
        "splits": refuse_overlapping_splits(splits, scored=scored),
        "tuning": refuse_scored_split_leak(panel, scored=scored),
        "verdict": "PASS",
    }
