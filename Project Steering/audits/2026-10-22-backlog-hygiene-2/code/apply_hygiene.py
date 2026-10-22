#!/usr/bin/env python3
"""BACKLOG-HYGIENE-2 -- apply the 2026-10-22 backlog sweep, idempotently.

    python apply_hygiene.py <BACKLOG.md in> <out> [--allow-absent] [--dry-run]

WHAT IT DOES, AND THE CONVENTION IT FOLLOWS
===========================================
`BACKLOG.md:8` -- "Strike items through when done, with the commit" -- and the
tool's own advice (`tools/backlog_drift.py`): "strike the id (~~A11~~) and keep the
evidence". So for every row below this STRIKES THE ID CELL and touches nothing
else. It NEVER deletes text: every other edit is an INSERTION, and the script
verifies that, line by line, before it writes a byte (`_no_text_lost`).

Three kinds of edit, all in `STRIKES` / `RELABELS` / `C2_RESTATE`:

  STRIKES    the row's OWN text proves the closure (`anchor` is a literal of that
             evidence, e.g. `DONE 2026-10-19: MIXED`). The id cell becomes
             `~~id~~`.
  RELABELS   Master Mind rulings of 2026-10-22: a status span is INSERTED at the
             head of the item cell so the LEAD stops lying, and (for the two that
             are decided/closed) the id is struck.
  C2_RESTATE the one STALE-BLOCKER row. Its blocker is struck in the markup but the
             row's own text says UNVERIFIABLE, NOT CLEARED. The two `~~` around the
             blocker are removed (the words stay), and a dated restatement of what
             is still open is appended.

WHY IT REFUSES, AND WHEN
========================
It will be re-applied on whatever the tip is at landing time, so it must not trust
the bytes it was written against:

  * a target id that occurs 0 or >1 times as the FIRST CELL of a table row -> REFUSE
    (`--allow-absent` downgrades only the 0-times case, for a row a later sweep
    already pruned; a duplicate is always a refusal);
  * a row whose closure evidence (`anchor`) is no longer in it -> REFUSE. A row
    that was edited back open after this was written must not be struck blindly;
  * any non-insertion edit it did not intend -> REFUSE.
On any refusal NOTHING is written and the exit code is 2.

IDEMPOTENT. A row whose id is already struck is skipped; a relabel whose marker is
already in the row is skipped; the C2 restatement is skipped once its marker is
present. Applying twice yields the same bytes as applying once (tested).

EOL-TRANSPARENT. The file is read and written as bytes; every line keeps the
line ending it came with (the repo blob is LF; a Windows checkout may be CRLF).
ASCII-only source and ASCII-only inserted text, on purpose (the dev box is cp1252).
"""
from __future__ import annotations

import argparse
import re
import sys

TAG = "BACKLOG-HYGIENE-2"
DATE = "2026-10-22"

# --------------------------------------------------------------------------
# targets. (id, anchor) -- the anchor is a LITERAL taken from the row's own
# closure evidence; it must still be in the row for the strike to apply.
# --------------------------------------------------------------------------
#: the 7 rows the tool already called DONE-BUT-OPEN before this sweep
OLD_FAMILY = [
    ("A11", "DONE 2026-08-18"),
    ("A13", "DONE 2026-09-07"),
    ("B4", "DONE 2026-08-03"),
    ("C3", "CLOSED 2026-08-16"),
    ("R1", "DONE 2026-09-02 (D-REFAV1-LEAD-BLOCK)"),
    ("R2", "DONE 2026-09-02"),
    ("R4", "DONE 2026-09-02 19:26Z"),
]

#: named-id rows that OPEN with their verdict (the old DONE-BUT-OPEN shape)
NAMED_DONE = [
    ("S4R-1", "DONE 2026-10-17"),
    ("P2F-1", "DONE 2026-10-18"),
    ("EP-1", "DONE 2026-10-18, 0 GPU"),
    ("SRTTC-DEV-APPLIER", "DONE 2026-10-19"),
    ("P6-READ-PREP", "DONE 2026-10-19"),
    ("LIT-EP-RL", "DONE 2026-10-18, 0 GPU"),
    ("UNSEEN-STRATUM-DATA", "CLOSED 2026-10-19 ~23:40: THE ROUTE DOES NOT EXIST"),
    ("SEARCH-STALE-T1", "CLOSED, NOT NEEDED"),
    ("P7-SPEC", "DONE 2026-10-22"),
    ("SO-A16", "DONE 2026-10-17"),
    ("TEST-RED-1", "DONE 2026-10-18"),
    ("SO-CARVE", "DONE 2026-10-18"),
    ("R0E-B2B5", "DONE 2026-10-19"),
    ("RL-1", "DONE 2026-10-19, 0 GPU"),
    ("RL-2", "DONE 2026-10-18, 0 GPU"),
    ("LOCK-STEAL-2", "DONE 2026-10-18, 0 GPU"),
    ("P6-RESUME-BANK", "DONE 2026-10-20, for the NEXT ship"),
    ("HNSCREEN-CACHE-CONTENT-FP", "DONE 2026-10-19"),
    ("HNSCREEN-FP-COST", "DONE 2026-10-20"),
    ("HJ-UNITS", "CLOSED 2026-10-18"),
]

#: named-id rows that LEAD open-type and close later (CLOSED-LATER)
NAMED_LATER = [
    ("EP-R1UB", "DONE 2026-10-19: outcome N, NONE-FEASIBLE"),
    ("NAVTEST-HEADBANK", "CANCELLED 2026-10-19"),
    ("HEADROW-PROBE", "DONE 2026-10-19: MIXED"),
    ("FASTNEG-CENSUS", "DONE 2026-10-19: INCONCLUSIVE"),
    ("PICKRULE-AUDIT-NC", "DONE 2026-10-19: ROW-MISS"),
    ("NC-FASTCAL", "DONE 2026-10-19: NO-RECOVER"),
    ("EP-RELIAB-PICKS", "DONE 2026-10-19: EP-OVERPREDICTED"),
    ("EP-ADMISSION-R1", "DONE 2026-10-19"),
    ("EP-ADMISSION-R2-CONFIRM", "DONE 2026-10-19: SEAL AT 1.05"),
    ("PRIVGEOM-SAFE", "DONE 2026-10-19: DAC INFO-UPSTREAM"),
    ("PRIVGEOM-CLEAN", "DONE 2026-10-20 01:37: INCONCLUSIVE"),
    ("PRIVGEOM-DAC", "superseded by PRIVGEOM-SAFE"),
    ("PRIVGEOM-PROBE", "superseded by PRIVGEOM-DAC"),
    ("EPR1UB-GATE-CACHE", "DONE 2026-10-19"),
    ("C2-GATE-SIBLING-LEAK", "DONE 2026-10-19"),
    ("L1-STEPA-RUN", "DONE 2026-10-19: O2 NOT ADMITTED"),
    ("L2-OFFPROP", "DONE 2026-10-19: L2-ADMIT"),
    ("L2-FRESH-STEPA", "READ 1 DONE 2026-10-19 20:12: O3"),
    ("EPADM-SEARCHBANK", "DONE 2026-10-19 ~23:55: ALL THREE DQ6 BARS MISSED"),
    ("EPADM-SEARCHBANK-S0", "CLOSED 2026-10-20, NOT BUILT"),
    ("SEARCH-DOSE", "DONE 2026-10-20: O1_CONFIRMED"),
    ("R0E-B9B10", "DONE 2026-10-19"),
    ("HNSCREEN-TOKEN-NAMES", "Row CLOSED."),
    ("D-SPACE", "RESOLVED 2026-10-19 19:15"),
    ("L2-STEPB-GATE", "STEP B READ 2026-10-19"),
]

STRIKES = OLD_FAMILY + NAMED_DONE + NAMED_LATER

#: Master Mind rulings, 2026-10-22: (id, anchor, marker, prefix, strike_id)
#: `marker` is the idempotency lock; `prefix` is INSERTED at the head of the item
#: cell so the lead states the ruling.
RELABELS = [
    ("SEARCH-P2-CONFIRM", "C2_FAIL_LOWER_CI",
     "CLOSED 2026-10-22: C2_FAIL_LOWER_CI",
     "**CLOSED 2026-10-22: C2_FAIL_LOWER_CI** (MM ruling, " + TAG + "; the C2 read "
     "is recorded in this row: one navtest look, dPDMS -0.155 [-0.350, +0.037]).",
     True),
    ("LATENCY-BUDGET", "DECIDED (PI, 2026-10-20",
     "DECIDED 2026-10-20 (PI, D-LATENCY-BUDGET)",
     "**DECIDED 2026-10-20 (PI, D-LATENCY-BUDGET)** (MM ruling, " + TAG + "; the "
     "decision is recorded verbatim in the DECIDED span at the end of this row).",
     True),
    ("SEARCH-HOST-PORT", "DEFERRED to the optimisation phase",
     "DEFERRED 2026-10-22 (MM ruling)",
     "**DEFERRED 2026-10-22 (MM ruling): to the optimisation phase** "
     "(D-LATENCY-BUDGET; " + TAG + ").",
     False),
    ("THOR-PICK-TICK", "OPTIMISATION phase",
     "DEFERRED 2026-10-22 (MM ruling)",
     "**DEFERRED 2026-10-22 (MM ruling): to the optimisation phase** "
     "(D-LATENCY-BUDGET; " + TAG + ").",
     False),
    ("SEARCH-THOR-LATENCY", "NEXT, 0 GPU to write",
     "DEFERRED 2026-10-22 (MM ruling)",
     "**DEFERRED 2026-10-22 (MM ruling): to the optimisation phase** "
     "(D-LATENCY-BUDGET; " + TAG + "). The search is CLOSED on P2's scorer "
     "(C2, 2026-10-22); re-opens with a P6/P7 scorer.",
     False),
]

#: C2: a STALE-BLOCKER row whose own text says UNVERIFIABLE, NOT CLEARED.
C2_ID = "C2"
C2_ANCHOR = "UNVERIFIABLE, not cleared"
C2_MARKER = "RE-STATED " + DATE + " (" + TAG + ")"
C2_BLOCKER = re.compile(r"~~(v2corpus reaching 30\s*k.*?)~~")
C2_NOTE = (
    "**" + C2_MARKER + ":** what is still open is ONE probe, not the scoring: is "
    "`flagship-v2corpus-30k` finished? `MODEL_REGISTRY.md` section 1.7 (mirror tip "
    "1664143b) still reads RUNNING with ETA 2026-07-29T01:10Z, 85 days past, and "
    "carries the 2026-08-16 annotation that no completion row exists. The blocker "
    "above is therefore NOT cleared and its strike was a display error (the "
    "strike markup is removed, the words are kept); C2 stays open until that run "
    "is probed or the Master Mind declares it moot."
)
C2_INLINE_TAG = " [blocker NOT cleared: see RE-STATED at the end of this row]"


# --------------------------------------------------------------------------
# row addressing
# --------------------------------------------------------------------------
_FIRST_CELL = re.compile(r"^\|([^|]*)\|")


def _row_id(line: str) -> str | None:
    """The id in a table row's FIRST cell, with strike / bold markup removed."""
    m = _FIRST_CELL.match(line)
    if not m:
        return None
    return re.sub(r"[~*]", "", m.group(1)).strip()


def _split_eol(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body):]


def _strike_id_cell(body: str) -> str:
    m = _FIRST_CELL.match(body)
    cell = m.group(1)
    inner = cell.strip()
    lead = cell[: len(cell) - len(cell.lstrip())] or " "
    trail = cell[len(cell.rstrip()):] or " "
    return "|" + lead + "~~" + inner + "~~" + trail + "|" + body[m.end():]


def _insert_prefix(body: str, prefix: str) -> str:
    m = _FIRST_CELL.match(body)
    rest = body[m.end():]
    return body[: m.end()] + " " + prefix + " " + rest.lstrip(" ")


def _append_note(body: str, note: str) -> str:
    stripped = body.rstrip()
    assert stripped.endswith("|"), "row does not end with a cell separator"
    return stripped[:-1].rstrip() + " " + note + " |"


def _no_text_lost(orig: str, new: str, inserted: list[str]) -> bool:
    """True iff `new` is `orig` plus insertions and `~~` markup changes only."""
    t = new
    for ins in inserted:
        if ins not in t:
            return False
        t = t.replace(ins, "", 1)
    # whitespace-insensitive on purpose: an insertion re-spaces its neighbours,
    # and `text|` vs `text |` is not a deletion of text.
    norm = lambda s: re.sub(r"\s+", "", s.replace("~~", ""))  # noqa: E731
    return norm(t) == norm(orig)


# --------------------------------------------------------------------------
def apply(text: str, allow_absent: bool = False) -> tuple[str, dict]:
    """Return (new_text, report). Raises SystemExit(2) on any refusal."""
    lines = text.splitlines(keepends=True)
    index: dict[str, list[int]] = {}
    for i, ln in enumerate(lines):
        rid = _row_id(ln)
        if rid:
            index.setdefault(rid, []).append(i)

    problems: list[str] = []
    report = {"struck": [], "already_struck": [], "relabelled": [],
              "already_relabelled": [], "restated": [], "absent": []}
    edits: dict[int, tuple[str, list[str]]] = {}      # line -> (new body, inserted)

    def locate(rid: str) -> int | None:
        hits = index.get(rid, [])
        if len(hits) == 1:
            return hits[0]
        if not hits and allow_absent:
            report["absent"].append(rid)
            return None
        problems.append("%s: %d occurrence(s) as a table row's first cell "
                        "(need exactly 1)" % (rid, len(hits)))
        return None

    # ---- strikes
    for rid, anchor in STRIKES:
        i = locate(rid)
        if i is None:
            continue
        body, _eol = _split_eol(lines[i])
        if "~~" in _FIRST_CELL.match(body).group(1):
            report["already_struck"].append(rid)
            continue
        if anchor not in body:
            problems.append("%s: closure evidence %r is no longer in the row -- "
                            "refusing to strike a row that may have been reopened"
                            % (rid, anchor))
            continue
        edits[i] = (_strike_id_cell(body), [])
        report["struck"].append(rid)

    # ---- relabels (MM rulings)
    for rid, anchor, marker, prefix, strike in RELABELS:
        i = locate(rid)
        if i is None:
            continue
        body, _eol = _split_eol(lines[i])
        if marker in body:
            report["already_relabelled"].append(rid)
            continue
        if anchor not in body:
            problems.append("%s: expected text %r is not in the row" % (rid, anchor))
            continue
        new = _insert_prefix(body, prefix)
        if strike and "~~" not in _FIRST_CELL.match(new).group(1):
            new = _strike_id_cell(new)
        edits[i] = (new, [prefix])
        report["relabelled"].append(rid)

    # ---- C2 restatement
    i = locate(C2_ID)
    if i is not None:
        body, _eol = _split_eol(lines[i])
        if C2_MARKER in body:
            report["already_relabelled"].append(C2_ID)
        elif C2_ANCHOR not in body:
            problems.append("C2: expected text %r is not in the row" % C2_ANCHOR)
        elif not C2_BLOCKER.search(body):
            problems.append("C2: the struck v2corpus blocker is not where it was; "
                            "re-read the row before restating it")
        else:
            new = C2_BLOCKER.sub(lambda m: m.group(1) + C2_INLINE_TAG, body, count=1)
            new = _append_note(new, C2_NOTE)
            edits[i] = (new, [C2_INLINE_TAG, C2_NOTE])
            report["restated"].append(C2_ID)

    # ---- the contract: never delete text
    for i, (new, inserted) in edits.items():
        orig, _eol = _split_eol(lines[i])
        # a strike's own `~~` pair and the C2 un-strike are the only non-insertions
        if not _no_text_lost(orig, new, inserted):
            problems.append("line %d (%s): the edit would delete text"
                            % (i + 1, _row_id(orig)))

    if problems:
        print("REFUSING (nothing written):")
        for p in problems:
            print("  - " + p)
        raise SystemExit(2)

    for i, (new, _ins) in edits.items():
        _orig, eol = _split_eol(lines[i])
        lines[i] = new + eol
    return "".join(lines), report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("backlog_in")
    ap.add_argument("backlog_out")
    ap.add_argument("--allow-absent", action="store_true",
                    help="skip (and report) a target id that is no longer a row")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args(argv)

    with open(a.backlog_in, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print("REFUSING: input is not valid UTF-8 (%s)" % e)
        return 2
    if len(raw) < 2000:
        print("REFUSING: input is %d bytes -- a short read is not a short file"
              % len(raw))
        return 2

    new, rep = apply(text, allow_absent=a.allow_absent)
    for k in ("struck", "already_struck", "relabelled", "already_relabelled",
              "restated", "absent"):
        print("[apply_hygiene] %-19s %3d  %s" % (k, len(rep[k]), " ".join(rep[k])))
    if a.dry_run:
        print("[apply_hygiene] dry run -- nothing written")
        return 0
    with open(a.backlog_out, "wb") as f:
        f.write(new.encode("utf-8"))
    print("[apply_hygiene] wrote %s (%d bytes, was %d)"
          % (a.backlog_out, len(new.encode("utf-8")), len(raw)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
