#!/usr/bin/env python3
"""Fire when `Project Steering/BACKLOG.md` carries work that has already landed.

WHY THIS EXISTS -- MEASURED 2026-09-02, and the backlog says it about itself
============================================================================
`BACKLOG.md` is the pull-list: the answer to *"gated is not idle"*, the thing a
GPU-blocked turn pulls from so the turn still ships something. Its own header
block records the defect, verbatim:

    THIS PULL-LIST IS CARRYING COMPLETED WORK, AND THAT DEFEATS ITS PURPOSE
    (MEASURED 2026-09-02) ... four consecutive items pulled from it tonight were
    already done ... a pull-list whose rows are stale costs exactly the turns it
    exists to make productive -- the puller spends the turn re-deriving state
    instead of doing work, which is the idling the rule forbids, wearing a
    checklist. Sweep before pulling, and close a row in the same turn its work
    lands -- the same discipline `lab_backlog_drift.py` enforces for the Lab's
    list, WHICH THIS LIST HAS NO EQUIVALENT OF.

This is that equivalent. `stack/scripts/lab_backlog_drift.py` asks *"have findings
outrun the Lab backlog?"* by keying on git history. The Programme list's failure is
the mirror image -- **work lands and the row is never closed** -- so this keys on a
contradiction INSIDE each row, which is exact for the same reason git history is:
it compares two fields that the file itself wrote.

WHAT IT CHECKS, AND WHY EACH CHECK CANNOT BE A JUDGEMENT CALL
=============================================================
`BACKLOG.md:8` states the file's own contract: **"Strike items through when done,
with the commit."** Every check below is a violation of that contract as written --
not an opinion about whether a row *feels* finished.

  DONE-BUT-OPEN   the row's OWN body says DONE / CLOSED / STRUCK / DISCHARGED,
                  its id is NOT struck, and no separate line closes it.
                  => it presents as open in a list a puller scans for open work.
  STALE-BLOCKER   the row's blocker text is struck through (the blocker cleared)
                  while the row id is not, and the row does not declare done.
                  => the operating standard's named stale-blocker class: a blocker
                  note that is not revisited when the thing it blocks on lands.
  CLOSED-LATER    (BACKLOG-HYGIENE-2, 2026-10-22) the row LEADS with an open-type
                  status (`**NEXT ...**`, `**QUEUED**`, `**ACTIVE**`, `**BUILDING**`)
                  and LATER, in the same row, a DATED bold span closes it
                  (`**DONE 2026-10-19: ...**`, `**CLOSED 2026-10-20 ...**`,
                  `**UPDATE 2026-10-19: superseded by X**`). A puller scans the
                  LEAD, so the row reads as live work while its own tail says the
                  work is finished -- the same contradiction as DONE-BUT-OPEN, in
                  the position where it is hardest to see. MEASURED 2026-10-22 on
                  tip 1664143b: 25 live rows (20 dated DONE/CLOSED/CANCELLED, 2
                  `UPDATE <date>: superseded`, 2 BLOCKER rows closed by a dated
                  RESOLVED, 1 qualified `READ 1 DONE`). See `later_closure` for
                  the exact grammar, and for the shapes it deliberately REFUSES
                  (undated words, `UPDATE <date>: SEALED`, a qualified DONE that
                  something follows).

THE ROW FAMILY THE TOOL COULD NOT SEE (BACKLOG-HYGIENE-2, MEASURED 2026-10-22)
===============================================================================
`TABLE_ROW` reads ids shaped `A11` / `R4`. The ~80 rows added from 2026-10-17
(`| EP-R1UB | ... |`, `| L2-FRESH-STEPA | ... |`, `| P7.2-ENSEMBLE | ... |`) were not
parsed AT ALL: the tool never classified them, so it could neither read them OPEN
nor catch their closure -- and the `UNPARSED` guard covered bullets only. Of the 91
rows it reported OPEN at that tip, none came from this family. `NAMED_TABLE_ROW`
reads them, but only inside an ITEM table (header `| # |` or `| id |`), because an
indented measurement table elsewhere in the file has first cells like `G-LIVE`; and
an item-table row that NO id regex reads is now counted and printed as UNPARSED.
  CLOSED-ELSEWHERE (INFO, never fails) an id closed by a LATER separate line
                  (`- **R8 -- STRUCK 2026-09-03**`) rather than a struck id. The
                  bullet sections use this convention deliberately and it does
                  announce itself to a reader, so it is reported, never gated on.

It does NOT decide whether a row is "really" done from the world. Doneness is
prose, and a heuristic for it would rot -- the same reasoning that made
`lab_backlog_drift.py` key on git rather than parse the register for "new rows",
and that made `backlog_claim_check.py` check only file existence. A narrow check
that cannot be wrong beats a broad one that quietly is.

ANTI-SILENT-PASS: A SCAN THAT READS ZERO ROWS IS UNTRUSTWORTHY, NOT CLEAN
=========================================================================
This is the whole `tools/` charter -- *"Absence of evidence is an ALARM, not an
all-clear"* -- and this programme has paid for it repeatedly: a fleet monitor
greping run names that had been renamed away printed nothing and was scored as
health; `pod_currency_audit.py` scanned a path MSYS had mangled, found nothing,
and reported **no drift** on a pre-launch gate. So the guards below exit **2
(REFUSING)**, never 0:

  * the backlog is missing, unreadable, or implausibly short
  * zero rows parsed, or fewer than `--min-rows` (a parser regression that drops
    a whole row family halves the count; the floor catches it)
  * zero CLOSED rows -- the file provably contains struck ids, so a close-detector
    that finds none is broken, and it is broken in the direction that reports
    "no problems"
  * every row closed -- the open-detector is then broken, and a pull-list with
    nothing open is a claim a human has to make, not one a regex may make

Guards 3 and 4 are two-sided on purpose: break the strike regex one way and every
row reads open (loud); break it the other and every row reads closed (silent).
Only the silent direction needed a guard, so it has one.

OUTPUT IS ASCII-ONLY. The dev box is cp1252 and the backlog is dense with house
glyphs; a guard that dies printing its own subject matter fails closed for the
wrong reason (MEASURED 2026-08-21). Row text is transliterated before printing.

USAGE
    python tools/backlog_drift.py                      # gate: 0 clean / 1 drift / 2 refusing
    python tools/backlog_drift.py --list-open          # the pull-list, actually-open rows only
    python tools/backlog_drift.py --json               # machine-readable
    python tools/backlog_drift.py --backlog <path>     # any list using the same conventions
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_BACKLOG = "Project Steering/BACKLOG.md"

#: Fewest rows a trustworthy scan of the live list may return. The live file
#: parsed 132 rows on 2026-09-18; a floor of 40 catches a parser regression that
#: drops an entire family (the table family is 60, the bullet family 72) while
#: still allowing the list to be pruned hard by a real sweep.
MIN_ROWS = 40

#: Fewest bytes a trustworthy read may return. A truncated read on a flaky mount
#: is indistinguishable from a short file unless something asserts a floor.
MIN_BYTES = 2000

#: Words that OPEN a closure assertion. ⛔ The word alone is NOT the signal --
#: see `_is_done_phrase`. Substring matching was tried first and was wrong on
#: FIVE live rows (MEASURED 2026-09-18, before this file was staged):
#:   A14  "the AlpaSim **closed**-loop stream"      -- a compound word
#:   F2   "(**done**-marker = off-switch)"          -- a compound word
#:   R30  "so `roll_**closed**` cannot be ported"   -- an identifier
#:   R21  "(D-STEER-INTERFACE-**RESOLVED**)"        -- a decision id
#:   R3   "C-REFAV1-PLAN-NOGOAL fix **landed**"     -- a DEPENDENCY that landed,
#:                                                     which is the row's reason
#:                                                     to be open, not to close
#: Five false alarms out of thirteen hits would have made this the gate nobody
#: reads -- the failure the Lab tool's own comments warn about.
DONE_WORDS = ("done", "closed", "struck", "discharged", "superseded",
              "completed", "cleared", "retracted")

#: ⛔ THERE IS DELIBERATELY NO `NEGATIONS` LIST HERE, AND ITS ABSENCE IS THE
#: FINDING. One was written -- `("partly struck", "not struck", "reframed", ...)`
#: against the live rows `R38 -- PARTLY STRUCK` and `R29 -- REFRAMED (not
#: struck)` -- and the mutation proof then showed it was DEAD CODE: deleting it
#: left the suite GREEN (mutation `M3`, MEASURED 2026-09-18), and removing it
#: changed **no verdict on any of the 117 live rows**. First-word anchoring
#: already rejects both ("partly", "reframed" are not closure verbs). ⭐ It was
#: kept for two hours as "defence in depth", which is what a guard that guards
#: nothing always looks like from the inside. The rule it earns: a check no
#: mutation can turn RED is not a check -- delete it, and let the mechanism that
#: actually does the work be the thing under test.

#: An emphasised span. The closure convention in this file is always emphatic:
#: `**DONE 2026-08-18, 0 GPU**`, `**CLOSED 2026-08-16 ...**`, `**STRUCK ...**`.
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)

#: Leading markup/glyph noise before the first real word of a span.
LEAD_NOISE = re.compile(r"^[^0-9A-Za-z]+")

#: A table item row: `| A3 | ... |`, `| **F1** | ... |`, `| ~~A12~~ | ... |`.
TABLE_ROW = re.compile(
    r"^\|\s*(?P<s1>~~)?\s*(?:\*\*)?\s*(?P<s2>~~)?\s*"
    r"(?P<id>[A-Z]\d+[a-z]?)"
    r"\s*(?P<s3>~~)?\s*(?:\*\*)?\s*(?P<s4>~~)?\s*\|")

#: ⛔ THE NAMED-ID TABLE FAMILY -- `| EP-R1UB | **NEXT, CPU ...** ... |`.
#: MEASURED 2026-10-22 (BACKLOG-HYGIENE-2): `TABLE_ROW` above takes only a
#: letter-plus-digits id (`A11`, `R4`), so the ~80 rows the Master Mind added from
#: 2026-10-17 on -- ids like `EP-R1UB`, `L2-FRESH-STEPA`, `P7.2-ENSEMBLE`,
#: `SO-A16` -- were NOT PARSED AT ALL. The tool did not read them as open and
#: miss their closure; it never saw them. Of the 91 rows it called OPEN, none were
#: from this family. ⭐ That is why a row could say `**DONE 2026-10-19**` in its
#: own text for days: the one check that exists for exactly that was blind to the
#: file's newest and fastest-growing section.
#: Gated on `in_item_table` (a `| # | item |` / `| id | item |` header seen, no
#: blank line since) because indented measurement tables elsewhere in the file
#: have first cells like `G-LIVE` and `**G-SUITE**`, which are NOT backlog items.
NAMED_TABLE_ROW = re.compile(
    r"^\|\s*(?P<s1>~~)?\s*(?:\*\*)?\s*(?P<s2>~~)?\s*"
    r"(?P<id>[A-Z][A-Z0-9]*(?:[-.][A-Za-z0-9]+)+)"
    r"\s*(?P<s3>~~)?\s*(?:\*\*)?\s*(?P<s4>~~)?\s*\|")

#: First cell of a header row that opens an ITEM table.
ITEM_TABLE_HEADERS = ("#", "id")

#: A separator row `|---|---|` / `|:--|--:|`.
TABLE_SEPARATOR = re.compile(r"^\|[\s:|\-]+\|?\s*$")

#: A bullet item row: `- **R26 -- ...`, and `- **R12, R22 -- STRUCK**: ...`
#: (two ids in one bullet -- real, and a parser that takes only the first
#: silently drops half the closures).
BULLET_ROW = re.compile(
    r"^-\s+(?P<s1>~~)?\s*\*\*\s*(?P<s2>~~)?\s*"
    r"(?P<ids>[A-Z]\d+[a-z]?(?:\s*,\s*[A-Z]\d+[a-z]?)*)"
    r"\s*(?:—|–|--|-)\s")

#: A line that LOOKS like a bullet item but did not parse. ⛔ Counted and printed,
#: never swallowed: the live list is written by hand, and a row typed with a plain
#: hyphen where the file's convention is an em-dash would otherwise be INVISIBLE
#: to every check above -- a silent miss inside the tool whose whole subject is
#: silent misses. MEASURED 2026-09-18: the em-dash-only form dropped two
#: hyphen-typed fixtures without a word, which is how this guard was earned.
BULLET_CANDIDATE = re.compile(r"^-\s+(?:~~)?\s*\*\*\s*(?:~~)?\s*[A-Z]\d+[a-z]?\b")

_ASCII_MAP = {
    "—": "--", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "×": "x",
    "≤": "<=", "≥": ">=", "≠": "!=", "±": "+/-",
    "→": "->", "⇒": "=>", "≈": "~", "°": "deg",
    " ": " ", "−": "-",
}


def ascii_only(s: str) -> str:
    """Transliterate to printable ASCII.

    ⚠️ Not cosmetic: the backlog is dense with house glyphs and the console is
    cp1252, so any row excerpt reaching stdout un-mapped is a crash in a guard.
    """
    for k, v in _ASCII_MAP.items():
        s = s.replace(k, v)
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in s)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _is_done_phrase(span: str) -> bool:
    """Does this span ASSERT a closure, as opposed to merely containing a word?

    ⛔ ANCHORED AT THE FIRST WORD. A closure in this file is always written as
    an assertion whose first word IS the verdict -- `DONE 2026-08-18, 0 GPU`,
    `CLOSED 2026-08-16 ...`, `STRUCK 2026-09-03`. Requiring the first word is
    what separates those from `closed-loop`, `roll_closed`, `done-marker` and
    `D-STEER-INTERFACE-RESOLVED`, which are words about something else.
    """
    t = LEAD_NOISE.sub("", _norm(span))         # glyphs, `~~`, `**`, `|`
    head = re.split(r"[^0-9a-z]+", t, maxsplit=1)[0] if t else ""
    return head in DONE_WORDS


def declares_done(body: str, bold_src: str | None = None) -> bool:
    """Does this row's OWN body assert that its work landed?

    Two shapes, both taken from the live file:
      (a) BULLET   `- **R8 -- STRUCK 2026-09-03** ...` -- once the `Rn --` prefix
          is consumed the body OPENS with the verdict.
      (b) TABLE    `| A11 | ~~...~~ (glyph) **DONE 2026-08-18, 0 GPU** ...` --
          the verdict opens an emphasised span.

    ⛔ `bold_src` exists because a BULLET's opening `**` is consumed by the id
    match; without restoring it the row's CLOSING `**` pairs with the NEXT span's
    opener and the resulting phantom span can begin with a closure verb.
    MEASURED: `check X** DONE-marker discipline **applies**` reads DONE without
    the prefix and OPEN with it -- and `done-marker` is verbatim live row F2.
    ⚠️ Inert on today's file (removing it moved 0 of 117 verdicts); it is kept
    because it is cheap and the failure it prevents is a FALSE CLOSURE, which is
    the direction that silently removes work from the list.
    """
    if _is_done_phrase(body):
        return True
    return any(_is_done_phrase(s)
               for s in BOLD.findall(bold_src if bold_src is not None else body))


#: ---------------------------------------------------------------------------
#: CLOSED-LATER grammar (BACKLOG-HYGIENE-2, 2026-10-22)
#: ---------------------------------------------------------------------------
#: A row that LEADS with one of these is still presenting itself as pending work.
#: ⚠️ Deliberately a LABEL selector, not a firing condition: firing needs a LATER
#: DATED closure span (below), so a wide lead set cannot create a false alarm. It
#: only chooses between `CLOSED-LATER` ("the lead lies") and `DONE-BUT-OPEN` ("the
#: row is honest but its id is not struck").
OPEN_LEAD_WORDS = ("next", "fix", "queued", "active", "open", "scheduled", "building",
                   "running", "proposed", "registered", "launch", "draft",
                   "conditional", "blocked", "blocker", "planned", "pending")

#: First words that CLOSE a row when they open a DATED bold span. The Master Mind's
#: list (DONE CLOSED SUPERSEDED STRUCK DISCHARGED WITHDRAWN) plus the three
#: `DONE_WORDS` already treated as closures (completed, cleared, retracted) and
#: `cancelled`, which a live row uses (`NAVTEST-HEADBANK`). ⛔ NOT `resolved` /
#: `unblocked`: those end a BLOCKER, not a row (`HNSCREEN-TOKEN-NAMES` is
#: `UNBLOCKED` and still has a fix to land), and `D-STEER-INTERFACE-RESOLVED` is a
#: decision id that already tripped the substring matcher once.
CLOSURE_HEAD_WORDS = DONE_WORDS + ("withdrawn", "cancelled", "canceled")

_CLOSE_ALT = "|".join(sorted(CLOSURE_HEAD_WORDS, key=len, reverse=True))
_DATE = r"\d{4}-\d{2}-\d{2}"

#: (1) `**DONE 2026-10-19: ...**`, `**CLOSED 2026-10-20, NOT BUILT.**`. The closure
#: word must be the span's FIRST word and a full date must follow within 24
#: characters. ⛔ The date is what separates a verdict from prose such as
#: `**DONE-marker discipline**` (live row F2's phrase); the `(?![-_][a-z])`
#: refuses the compound outright, so the date is a second lock, not the only one.
_DATED_CLOSURE = re.compile(
    r"(?:%s)\b(?![-_][a-z])[^0-9]{0,24}?%s" % (_CLOSE_ALT, _DATE))

#: (2) `**UPDATE 2026-10-19: superseded by PRIVGEOM-SAFE**` -- the row's own update
#: note retiring it. A date AND a closure word right after the colon: an update
#: such as `**UPDATE 2026-10-19: SEALED**` records progress, not closure.
_UPDATE_CLOSURE = re.compile(
    r"update\s+%s(?:\s+[\d:~z]+)?\s*[:\-]\s*(?:%s)\b(?![-_][a-z])"
    % (_DATE, _CLOSE_ALT))

#: (3) A QUALIFIED done: `**READ 1 DONE 2026-10-19 20:12: O3**`, `**STEP 1 DONE ...**`,
#: `**S1b DONE ...**`. ⛔ THIS SHAPE IS A PARTIAL-PROGRESS MARKER, NOT A CLOSURE, in
#: every multi-step live row (`SEL-RANK` STEP 1, `UNSEEN-STRATUM-DATA` S1/S1b,
#: `P6-HN` Step 0, `R0E-PICK` B3/B6/B7, `PAPER-720` C3/C4/C11). Counting it
#: unconditionally would close five live rows that have work left.
#: It is accepted ONLY as the row's LAST bold span of a row that LEADS open-type:
#: the rows are append-only, so a qualified DONE that nothing emphasised follows,
#: under a lead that still says NEXT, is the row's final word -- `L2-FRESH-STEPA`
#: (`NEXT` ... `READ 1 DONE ... O3`; its successor `UNSEEN-STRATUM-DATA` is a row
#: of its own). A qualified DONE with a later open marker stays OPEN.
_QUALIFIED_DONE = re.compile(r"[a-z]{1,12}\s*\d+[a-z]?\s+(?:done|closed)\b\s+%s" % _DATE)

#: (4) A BLOCKER row's own resolution: `**BLOCKER (needs the PI ...)**` ... later
#: `**RESOLVED 2026-10-19 19:15:**` (`D-SPACE`, `L2-STEPB-GATE`). `resolved` is a
#: closure word ONLY under a `blocker` / `blocked` lead: the row IS the blocker,
#: so its resolution closes the row. ⛔ Anywhere else it is a decision id
#: (`D-STEER-INTERFACE-RESOLVED`) or the end of a sub-blocker on a row with work
#: left, so it is deliberately NOT in `CLOSURE_HEAD_WORDS`.
BLOCKER_LEADS = ("blocker", "blocked")
_DATED_RESOLVED = re.compile(r"resolved\b(?![-_][a-z])[^0-9]{0,24}?%s" % _DATE)


#: Like `LEAD_NOISE` but KEEPS `*`: the opening `**` is the thing being tested.
_LEAD_KEEP_STARS = re.compile(r"^[^0-9A-Za-z*]+")


def lead_status(body: str) -> str:
    """The first word of the status span the row OPENS with, lower-cased, or "".

    Only a row whose item cell BEGINS with an emphasised span has a lead status:
    `| X | **NEXT, CPU** ...` -> "next". A struck or plain title
    (`~~Rescue ...~~ **DONE ...**`, `Re-verify ...`) has none, and stays on the
    `DONE-BUT-OPEN` path it always had.
    """
    t = _LEAD_KEEP_STARS.sub("", body.strip())
    if not t.startswith("**"):
        return ""
    m = BOLD.match(t)
    if not m:
        return ""
    span = LEAD_NOISE.sub("", _norm(m.group(1)))
    return re.split(r"[^0-9a-z]+", span, maxsplit=1)[0] if span else ""


def later_closure(body: str) -> str | None:
    """Does a row that LEADS open-type later close itself in a DATED bold span?

    Returns the shape that fired (`dated`, `update`, `qualified`, `resolved`) or
    None.
    Never fires on a row whose lead is not open-type, and never on the lead span.
    """
    if lead_status(body) not in OPEN_LEAD_WORDS:
        return None
    t = _LEAD_KEEP_STARS.sub("", body.strip())
    spans = [LEAD_NOISE.sub("", _norm(s)) for s in BOLD.findall(t)]
    blocker = lead_status(body) in BLOCKER_LEADS
    for i, sp in enumerate(spans[1:], 1):           # spans[0] IS the lead
        if _DATED_CLOSURE.match(sp):
            return "dated"
        if blocker and _DATED_RESOLVED.match(sp):
            return "resolved"
        if _UPDATE_CLOSURE.match(sp):
            return "update"
        if i == len(spans) - 1 and _QUALIFIED_DONE.match(sp):
            return "qualified"
    return None


def has_struck_blocker(cells: list[str]) -> bool:
    """A struck-through blocker in a cell that is neither the id nor the item.

    The C-section's shape: `| C3 | v5 gate verdict | ~~v5 reaching step 2000~~ ...|`
    -- the BLOCKER cleared and was struck while the row id stayed open.

    ⛔ `cells[3:]`, not `cells[2:]`. `split("|")` puts "" at [0], the id at [1]
    and the ITEM at [2]; A11/A13 strike their ITEM text, so including [2] would
    report a struck item as a cleared blocker -- a wrong reason attached to a
    right row, which is how a reader stops trusting the label.
    """
    return any("~~" in c for c in cells[3:])


class Row:
    __slots__ = ("rid", "line_no", "kind", "section", "text", "id_struck",
                 "body_done", "blocker_struck", "closed_later")

    def __init__(self, rid, line_no, kind, section, text, id_struck,
                 body_done, blocker_struck, closed_later=None):
        self.rid = rid
        self.line_no = line_no
        self.kind = kind
        self.section = section
        self.text = text
        self.id_struck = id_struck
        self.body_done = body_done
        self.blocker_struck = blocker_struck
        #: None, or the shape of the later dated closure (`later_closure`).
        self.closed_later = closed_later


#: A heading that is itself an ITEM rather than a container -- the live file
#: carries two (`## BLOCKING INTEGRATION ITEM ...`, `## D-ROLL-1h -- CLOSED ...`).
#: ⛔ They have NO id, so the row parser cannot see them. A tool blind to a whole
#: class of items in its own input is the "absence scored as health" failure this
#: file's docstring is about, so they are counted and printed -- never silently
#: skipped -- even though a human, not a regex, adjudicates them.
#: ⛔ ANCHORED, for the same reason `_is_done_phrase` is. An unanchored
#: `BLOCKING|(blocker glyph)` also matched the SECTION CONTAINER
#: `## B. ONE GPU -- executable now (pod3; (glyph) eval stays free for C64-A)`,
#: whose glyph is a note inside a heading, not a status on it (MEASURED
#: 2026-09-18). An item-heading LEADS with its status.
HEADING_ITEM = re.compile(r"^(?:[⛔✅⏹⚠️\s]+|BLOCKING)")
HEADING_CLOSED = re.compile(r"✅|\bCLOSED\b|\bDONE\b|⏹")


def parse(text: str) -> tuple[list[Row], list[str], list[dict], list[int]]:
    """Return (rows, sections, heading-items, unparsed-bullet-line-numbers)."""
    rows: list[Row] = []
    sections: list[str] = []
    blocks: list[dict] = []
    missed: list[int] = []
    section = "(preamble)"
    in_item_table = False
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s.startswith("|"):
            in_item_table = False               # a blank/prose line ends a table
        if s.startswith("#"):
            raw = s.lstrip("#").strip()
            section = ascii_only(raw)
            sections.append(section)
            level = len(s) - len(s.lstrip("#"))
            if level == 2 and HEADING_ITEM.match(raw):
                blocks.append({"line": n, "title": section[:120],
                               "closed": bool(HEADING_CLOSED.search(raw))})
            continue
        if s.startswith("|"):
            first = (s.split("|") + ["", ""])[1].strip().lower()
            if first in ITEM_TABLE_HEADERS:
                in_item_table = True            # `| # | item |` / `| id | item |`
                continue
        m = TABLE_ROW.match(s)
        named = False
        if not m and in_item_table:
            m = NAMED_TABLE_ROW.match(s)
            named = bool(m)
        if m:
            cells = [c for c in s.split("|")]
            struck = any(m.group(g) for g in ("s1", "s2", "s3", "s4"))
            body = "|".join(cells[2:])
            rows.append(Row(m.group("id"), n, "named" if named else "table",
                            section, s, bool(struck), declares_done(body),
                            has_struck_blocker(cells), later_closure(body)))
            continue
        if in_item_table and s.startswith("|") and not TABLE_SEPARATOR.match(s):
            # ⛔ A ROW OF AN ITEM TABLE THAT NO ID REGEX READS. Counted and printed,
            # never swallowed -- the same anti-silent-miss guard the bullet family
            # has, which the table family LACKED and which is how ~80 named rows
            # went unread for five days.
            missed.append(n)
            continue
        m = BULLET_ROW.match(s)
        if m:
            struck = bool(m.group("s1") or m.group("s2"))
            body = s[m.end():]
            # ⛔ RESTORE THE OPENING `**` THE ID MATCH CONSUMED. Without it the
            # row's CLOSING `**` pairs with the NEXT span's opening one, and
            # every emphasised span in the bullet families is off by one.
            done = declares_done(body, bold_src="**" + body)
            for rid in re.split(r"\s*,\s*", m.group("ids")):
                rows.append(Row(rid, n, "bullet", section, s, struck, done,
                                False))
            continue
        if BULLET_CANDIDATE.match(s):
            missed.append(n)
    return rows, sections, blocks, missed


def classify(rows: list[Row]) -> list[dict]:
    """Per-id verdicts.

    ⛔ AGGREGATION BY ID IS REQUIRED, NOT A REFINEMENT. The bullet sections close
    an item by adding a LATER separate line (`- **R8 -- STRUCK 2026-09-03**`)
    instead of striking the original. A per-occurrence scan reports every such
    original as stale and cries wolf on the majority of the file -- which is how
    a gate gets muted. The id, not the line, is the unit of state.
    """
    by_id: dict[str, list[Row]] = {}
    for r in rows:
        by_id.setdefault(r.rid, []).append(r)

    out: list[dict] = []
    for rid, occ in sorted(by_id.items(), key=lambda kv: (kv[1][0].line_no,)):
        first = occ[0]
        id_struck = any(o.id_struck for o in occ)
        # A closure announced on a LATER, SEPARATE line than the item itself.
        closed_elsewhere = any(o.body_done for o in occ[1:])
        self_done = first.body_done
        blocker_struck = any(o.blocker_struck for o in occ)

        if id_struck:
            verdict = "CLOSED"
        elif closed_elsewhere:
            verdict = "CLOSED-ELSEWHERE"
        elif first.closed_later:
            verdict = "CLOSED-LATER"
        elif self_done:
            verdict = "DONE-BUT-OPEN"
        elif blocker_struck:
            verdict = "STALE-BLOCKER"
        else:
            verdict = "OPEN"

        out.append({
            "id": rid,
            "line": first.line_no,
            "kind": first.kind,
            "section": first.section,
            "verdict": verdict,
            "occurrences": [o.line_no for o in occ],
            "closed_later_shape": first.closed_later,
            "excerpt": ascii_only(first.text)[:200],
        })
    return out


def _git(repo: Path, *args: str) -> str:
    """git with explicit dirs -- the form that survives a flaky mount.

    ⚠️ Borrowed verbatim from `lab_backlog_drift.py`: a plain `cd && git` reports
    "not a repository" intermittently on the Drive mount.
    """
    try:
        out = subprocess.run(
            ["git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        # ⚠️ No git on PATH, or no repo. This is CONTEXT only (see git_context),
        # so it degrades to "unknown" -- it must never turn a readable backlog
        # into a refusal, and it must never be mistaken for "no drift".
        return ""
    return out.stdout if out.returncode == 0 else ""


def git_context(repo: Path, backlog: str) -> dict:
    """Informational only: when was the list last written, and did the repo move?

    ⚠️ Deliberately NEVER decides the exit code. The Lab tool can gate on this
    because its finding-files are a short explicit list; here any commit could
    close a row, so gating on the count would fire every day and train readers to
    ignore the output -- the failure mode the Lab tool's own comments warn about.
    """
    h = _git(repo, "log", "-1", "--format=%H", "--", backlog).strip()
    if not h:
        return {"last_commit": None, "last_date": None, "commits_since": None}
    d = _git(repo, "log", "-1", "--format=%ad", "--date=short", h).strip()
    since = _git(repo, "rev-list", "--count", f"{h}..HEAD").strip()
    return {"last_commit": h[:9], "last_date": d or None,
            "commits_since": int(since) if since.isdigit() else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--backlog", default=DEFAULT_BACKLOG)
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS,
                    help="floor below which the scan is UNTRUSTWORTHY, not clean")
    ap.add_argument("--list-open", action="store_true",
                    help="print the genuinely-open rows (the real pull-list)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args(argv)

    repo = a.repo.resolve()
    bl = repo / a.backlog

    # ---- guards: every one of these exits 2, never 0 -----------------------
    if not bl.is_file():
        print("REFUSING: no backlog at %s" % ascii_only(str(bl)))
        return 2
    try:
        text = bl.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print("REFUSING: could not read %s (%s) -- a failed read is not an "
              "empty file" % (ascii_only(str(bl)), e.__class__.__name__))
        return 2
    if len(text.encode("utf-8", "replace")) < MIN_BYTES:
        print("REFUSING: %s read only %d bytes (floor %d) -- treating a short "
              "read as a failed read, not as a short file"
              % (ascii_only(a.backlog), len(text), MIN_BYTES))
        return 2

    rows, sections, blocks, missed = parse(text)
    if not rows:
        print("REFUSING: parsed 0 rows -- a scan that reads zero rows is "
              "UNTRUSTWORTHY, not clean")
        return 2
    if not sections:
        print("REFUSING: parsed 0 sections -- the file shape is not what this "
              "parser expects; a silent 'no problems' here would be a lie")
        return 2

    verdicts = classify(rows)
    n = len(verdicts)
    if n < a.min_rows:
        print("REFUSING: parsed %d distinct ids, below the --min-rows floor of "
              "%d -- a parser regression that drops a row family looks exactly "
              "like a tidy backlog" % (n, a.min_rows))
        return 2

    counts = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    closed = counts.get("CLOSED", 0) + counts.get("CLOSED-ELSEWHERE", 0)
    if closed == 0:
        print("REFUSING: 0 of %d rows read as closed. The list's own contract "
              "(BACKLOG.md:8) is 'Strike items through when done', so a "
              "close-detector finding none is broken -- and broken in the "
              "direction that reports no problems" % n)
        return 2
    if closed == n:
        print("REFUSING: all %d rows read as closed. Either the open-detector "
              "is broken or the pull-list is empty; both are a human's call, "
              "not a regex's" % n)
        return 2

    # ---- report ------------------------------------------------------------
    ctx = git_context(repo, a.backlog)
    drift = [v for v in verdicts
             if v["verdict"] in ("DONE-BUT-OPEN", "STALE-BLOCKER", "CLOSED-LATER")]

    if a.json:
        print(json.dumps({"backlog": a.backlog, "rows": n, "counts": counts,
                          "git": ctx, "heading_items": blocks,
                          "verdicts": verdicts}, indent=2))
        return 1 if drift else 0

    print("[backlog-drift] %s" % ascii_only(a.backlog))
    print("[backlog-drift] rows read     : %d distinct ids over %d occurrence(s), "
          "%d section(s)" % (n, len(rows), len(sections)))
    print("[backlog-drift] last written  : %s  %s  (%s commit(s) to the repo since)"
          % (ctx["last_commit"] or "?", ctx["last_date"] or "?",
             ctx["commits_since"] if ctx["commits_since"] is not None else "?"))
    print("[backlog-drift] verdicts      : " + "  ".join(
        "%s=%d" % (k, counts[k]) for k in sorted(counts)))

    # ⚠️ Printed ALWAYS, including the zero case. These items have no id, so the
    # row checks above are structurally blind to them; saying so is the whole
    # point -- a reader who is not told a class was skipped assumes it was clean.
    if missed:
        print("[backlog-drift] UNPARSED      : %d bullet line(s) look like items "
              "but did not parse - %s" % (len(missed), missed[:10]))
        print("                 a row the parser cannot see is invisible to every "
              "check above; fix the row's separator or this regex.")
    open_blocks = [b for b in blocks if not b["closed"]]
    print("[backlog-drift] heading items : %d (%d without a closure marker) - NOT "
          "id'd rows, a human adjudicates these" % (len(blocks), len(open_blocks)))
    for b in open_blocks:
        print("                 L%-4d %s" % (b["line"], b["title"][:96]))

    if a.list_open:
        print()
        print("GENUINELY OPEN ROWS - this is the real pull-list:")
        for v in verdicts:
            if v["verdict"] == "OPEN":
                print("  %-5s L%-4d %s" % (v["id"], v["line"], v["excerpt"][:120]))

    if not drift:
        print("[backlog-drift] OK - no row declares itself done while sitting "
              "open in the list.")
        return 0

    print()
    print("STALE ROWS - each contradicts BACKLOG.md:8, 'Strike items through")
    print("when done, with the commit'. A puller scanning for open work reads")
    print("these as work, and spends the turn re-deriving that they are not:")
    for v in drift[:a.max]:
        print("  %-14s %-5s L%-4d  %s"
              % (v["verdict"], v["id"], v["line"], v["excerpt"][:110]))
    if len(drift) > a.max:
        print("  ... and %d more" % (len(drift) - a.max))
    print()
    print("=> DONE-BUT-OPEN : strike the id (~~A11~~) and keep the evidence.")
    print("=> CLOSED-LATER  : the row LEADS open (NEXT/BUILDING/...) but a dated span")
    print("   later in it closes it. Strike the id and keep the evidence -- a puller")
    print("   reads the lead, not the tail.")
    print("=> STALE-BLOCKER : the blocker cleared and the row was not revisited;")
    print("   re-state what is still open, or strike it. Do not leave it reading")
    print("   as a live gap.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
