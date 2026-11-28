"""navtok -- NVIDIA PhysicalAI-AV turn-by-turn ``nav_text`` -> (manoeuvre class, distance_m | None, raw).

Source file: ``navigation/navigation.parquet`` (clip_id, timestamp us after clip start, nav_text). Its schema metadata says the text is
"rendered from each drive's GPS trace matched to OpenStreetMap-based routing; street names and road numbers removed".
MEASURED 2026-11-16: 9,864 distinct texts, 53 templates once digits are masked (``navtok_report.json``).

The grammar is closed and frozen here. A text that matches NO rule is returned as class ``UNPARSED`` (never guessed, never a default
FOLLOW), and the report lists every one. ``VOCAB`` is the frozen class list; ``parse`` asserts every class it emits is in it.

Classes are ``VERB[_DEGREE]_SIDE`` with DEGREE in {SHARP, SLIGHT} (absent = ordinary) and SIDE in {L, R, STRAIGHT}; the roundabout and
U-turn classes carry no side. ``coarse`` maps a class to the trainer's 3-token space (FOLLOW 0, TURN_L 1, TURN_R 2) -- see ``COARSE``.
"""
from __future__ import annotations

import re
from typing import NamedTuple, Optional

SIDES = {"left": "L", "right": "R", "straight": "STRAIGHT"}
_DEG = {"sharp": "SHARP", "slight": "SLIGHT"}

# (verb, regex over the text with the distance suffix removed, allowed degrees, allowed sides)
_RULES = [
    ("CONTINUE", r"Continue(?: (?P<deg>sharp|slight))?(?: (?P<side>left|right|straight))?"),
    ("TURN", r"Turn(?: (?P<deg>sharp|slight))? (?P<side>left|right)"),
    ("KEEP", r"Keep(?: (?P<deg>slight))? (?P<side>left|right|straight)"),
    ("MERGE", r"Merge(?: (?P<deg>slight))? (?P<side>left|right)"),
    ("RAMP", r"Take the ramp(?: (?P<deg>sharp|slight))? (?P<side>left|right|straight)"),
    ("UTURN", r"Make a U-turn"),
    ("RB_ENTER", r"Enter roundabout"),
    ("RB_EXIT", r"Exit roundabout"),
    ("ROTARY_EXIT", r"Exit rotary"),
    ("RB_TAKE_EXIT", r"At roundabout, take exit (?P<exit>\d+)"),
    ("RB_NEXT_EXIT", r"Continue around the roundabout and take the next exit"),
]
_DIST = r"(?: in (?P<d>\d+)m)?"
_COMPILED = [(v, re.compile(p + _DIST)) for v, p in _RULES]


def _mk(verb, deg, side):
    return "_".join(x for x in (verb, deg, side) if x)


def _build_vocab():
    out = ["UNPARSED"]
    for verb, degs, sides in (("CONTINUE", ("", "SHARP", "SLIGHT"), ("", "L", "R", "STRAIGHT")),
                              ("TURN", ("", "SHARP", "SLIGHT"), ("L", "R")),
                              ("KEEP", ("", "SLIGHT"), ("L", "R", "STRAIGHT")),
                              ("MERGE", ("", "SLIGHT"), ("L", "R")),
                              ("RAMP", ("", "SHARP", "SLIGHT"), ("L", "R", "STRAIGHT"))):
        for d in degs:
            for s in sides:
                out.append(_mk(verb, d, s))
    out += ["UTURN", "RB_ENTER", "RB_EXIT", "ROTARY_EXIT", "RB_TAKE_EXIT", "RB_NEXT_EXIT"]
    return tuple(out)


#: the frozen class list (index = class id). Order is part of the contract: tests pin it.
VOCAB = _build_vocab()
CLS_ID = {c: i for i, c in enumerate(VOCAB)}

#: trainer token space (v9 ``nav_token``: FOLLOW 0, TURN_L 1, TURN_R 2) <- a class. A TURN (any degree) and a U-turn are turns; everything
#: else announced (continue, keep, merge, ramp, roundabout) is FOLLOW in this coarse space. UTURN has no side: -1 = "turn, side unknown".
def coarse(cls: str) -> int:
    if cls == "UNPARSED":
        return -2
    if cls == "UTURN":
        return -1
    if cls.startswith("TURN_"):
        return 1 if cls.endswith("_L") else 2
    return 0


class NavTok(NamedTuple):
    cls: str
    dist_m: Optional[int]
    raw: str
    verb: str
    degree: str
    side: str
    exit_n: Optional[int]


def parse_full(text: str) -> NavTok:
    for verb, rx in _COMPILED:
        m = rx.fullmatch(text)
        if m is None:
            continue
        g = m.groupdict()
        d = int(g["d"]) if g.get("d") is not None else None
        deg = _DEG.get(g.get("deg") or "", "")
        side = SIDES.get(g.get("side") or "", "")
        ex = int(g["exit"]) if g.get("exit") is not None else None
        cls = _mk(verb, deg, side)
        assert cls in CLS_ID, f"class {cls!r} (from {text!r}) is not in the frozen vocabulary"
        return NavTok(cls, d, text, verb, deg, side, ex)
    return NavTok("UNPARSED", None, text, "", "", "", None)


def parse(text: str):
    """-> (class, distance_m or None, raw)."""
    t = parse_full(text)
    return t.cls, t.dist_m, t.raw
