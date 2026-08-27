"""Negation scoping for CoT term extraction.

⛔⛔ WHY THIS EXISTS — A TOKEN THAT FIRED 408x MORE OFTEN WHEN THE SCENE WAS
EMPTY. MEASURED 2026-08-24. Alpamayo's `critical_components_analysis` says, on
the 853 clips where nothing is critical:

    "The first 2 seconds show a clear highway with no lead vehicle pedestrians
     cyclists traffic lights or obstacles affecting the ego lane."

A term-matching extractor sees `traffic light` in that sentence and emits
`TRAFFIC_LIGHT_REACT`. The result, over the corpus:

| token | on "nothing critical" clips | on clips naming a component | ratio |
|---|---|---|---|
| `TRAFFIC_LIGHT_REACT` | **60.4 %** | 0.1 % | **408x** |
| `EVADE_IN_CORRIDOR` | 7.0 % | 2.0 % | 3.6x |

Both are inverted: the tokens fire hardest exactly where the source says there is
nothing to react to. Every other perception token runs the correct way round
(`GAP_TARGET` 0.04x, `YIELD` 0.21x), which is what makes these two diagnosable
rather than merely noisy.

⚠️ **I had already measured negation and dismissed it.** An earlier probe found
"166 negated mentions, 3.5 % of clips" and I judged it small. That probe searched
a narrow window (`\\bno\\b` within 40 characters of the term) over the whole
corpus; it could not see a SIX-TERM ENUMERATION under a single `no`, which is the
shape that actually occurs. ⇒ **A negation test whose pattern cannot express the
negation being used will report that negation is rare.**

## What this module does

Marks the character spans of a text that lie inside a negation, so a term found
inside one is not treated as present. Scope runs from a negation cue to the end
of its clause — enumerations (`no A, B, C or D`) are covered because the whole
noun-phrase list sits in one clause.

⚠️ It deliberately does NOT try to decide truth. `negated("...no pedestrians...")`
means *the text asserts absence*, not *there is no pedestrian*. A caller that
wants presence must still find a positive mention.
"""
from __future__ import annotations

import re

#: Cues that open a negative scope.
_CUE = re.compile(
    r"\b(?:no|not|none|without|absent|absence of|free of|clear of|devoid of|"
    r"nor|neither|lacks?|lacking)\b", re.I)

#: Tokens that CLOSE a negative scope. A clause boundary, or a contrastive
#: conjunction that flips polarity back ("no pedestrians BUT a cyclist ahead").
_CLOSE = re.compile(
    r"[.;:!?]|\b(?:but|however|although|though|whereas|while|except|"
    r"instead|yet)\b", re.I)

#: A cue that is itself part of a positive phrase must not open a scope.
#: "no doubt", "not only", "no less than" are assertions, not absences.
_FALSE_CUE = re.compile(
    r"\bno(?:t)?\s+(?:doubt|only|just|merely|less|fewer|more)\b", re.I)


def negated_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of ``text`` that sit inside a negation."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    for m in _CUE.finditer(text):
        if _FALSE_CUE.match(text, m.start()):
            continue
        close = _CLOSE.search(text, m.end())
        spans.append((m.start(), close.start() if close else len(text)))
    return spans


def is_negated(text: str, start: int, end: int) -> bool:
    """Does the match at ``[start, end)`` fall inside a negation scope?"""
    return any(a <= start and end <= b for a, b in negated_spans(text))


def strip_negated(text: str) -> str:
    """``text`` with negated spans blanked out, for term matching.

    Blanking rather than deleting keeps every character offset stable, so a
    caller can still report positions into the ORIGINAL string — deleting would
    silently shift every downstream index.
    """
    if not text:
        return text
    out = list(text)
    for a, b in negated_spans(text):
        for i in range(a, min(b, len(out))):
            if out[i] not in "\n":
                out[i] = " "
    return "".join(out)


def find_positive(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """First match of ``pattern`` that is NOT inside a negation."""
    for m in pattern.finditer(text or ""):
        if not is_negated(text, m.start(), m.end()):
            return m
    return None
