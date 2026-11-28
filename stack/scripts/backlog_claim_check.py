#!/usr/bin/env python3
"""Check `Project Steering/BACKLOG.md`'s FILE-EXISTENCE claims against the repo.

WHY THIS EXISTS -- MEASURED 2026-09-02. BACKLOG.md is the live pull-list: it is
the answer to "gated is not idle", the thing a GPU-blocked turn pulls from so the
turn still ships something. On one night, FOUR consecutive items pulled from it
were already done, and a fifth (A12) was three-quarters done behind a stale
parenthetical. A12 read:

    Ship the 4 stale/absent files to Thor -- `sel_winners_curse_law.py` (absent),
    `lf0_bev_lead.py`, `p8_bev_reel.py`, `refc_train.py`

`sel_winners_curse_law.py` was NOT absent. It exists and is git-tracked at
`stack/scripts/sel_winners_curse_law.py`; it had simply never been shipped. The
puller spends the turn re-deriving state instead of working -- which is the
idling the never-idle rule forbids, wearing a checklist.

WHAT THIS CHECKS, AND WHAT IT DELIBERATELY DOES NOT
===================================================
It checks ONE thing exactly: a row that says a named file is **absent/missing**
while the file is in fact TRACKED, and the converse. That is a mechanical,
exact question -- git's index either has the path or it does not.

It does NOT try to decide whether a row is "done". Doneness is prose, and a
heuristic for it would rot -- the same reasoning that made `lab_backlog_drift.py`
key on git history rather than parsing the register for "new rows". A narrow
check that cannot be wrong beats a broad one that quietly is.

WHY `git ls-files` AND NOT `find`/`glob`
========================================
`find`, `rg` and `grep` SILENTLY UNDER-REPORT on the G: Drive mount and still
exit 0 -- measured the same night, when `find` missed `colab/s2_lab_lib.py`
that `ls` returned instantly, and again when `git ls-tree -r` truncated a tree
at 7,535 of 8,525 entries five times in a row. The git index is read once, in
one process, and is exact. ⛔ An empty result from a filesystem sweep on this
mount is not evidence of absence.

KNOWN IMPRECISION, STATED RATHER THAN DISCOVERED
================================================
It OVER-fires on collective phrasing. A row headed *"the 4 stale/absent files"*
puts an absent-word within proximity of all four names, so siblings that were
merely un-shipped are flagged alongside the one genuinely mis-annotated. That is
the acceptable direction for a gate whose failure mode is "look again" -- but it
is not precision, and a run with hits still needs a human to read the row.

⚠️ It has, so far, ZERO true positives on the live backlog: the one real case
(A12) was found and fixed by hand *before* this existed. What proves the checker
works is therefore not a green run -- it is the regression fixture in
`tests/test_backlog_claim_check.py`, which replays A12's pre-fix text and
requires a hit, and replays F5/F7 and requires silence. **A gate that has never
fired on real input is only as trustworthy as its negative control.**

Output is ASCII-only on purpose: the dev box is cp1252 and a checker that dies
printing its own house glyphs is worse than no checker.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: Words a row uses when it asserts a file is NOT there. Matched only inside the
#: same table cell as the filename, never across the whole row.
ABSENT_WORDS = ("absent", "missing", "does not exist", "doesn't exist",
                "not present", "no such file", "never written", "not in repo")

#: A backticked token that looks like a source/artifact filename.
FILE_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|sh|md|json|jsonl|yaml|yml|"
                     r"toml|cfg|csv|tsv|txt|pt))`")


def tracked_paths(repo: Path) -> set[str]:
    """Every path in git's index. Exact; see the module docstring."""
    out = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True)
    return {l.strip() for l in out.stdout.decode("utf-8", "replace").splitlines()
            if l.strip()}


def basenames(paths: set[str]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for p in paths:
        idx.setdefault(p.rsplit("/", 1)[-1], []).append(p)
    return idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--backlog", default="Project Steering/BACKLOG.md")
    a = ap.parse_args(argv)

    repo = a.repo.resolve()
    bl = repo / a.backlog
    if not bl.exists():
        print("REFUSING: no backlog at %s" % bl)
        return 2

    text = bl.read_text(encoding="utf-8", errors="replace")
    tracked = tracked_paths(repo)
    by_base = basenames(tracked)
    if not tracked:
        # ⛔ An empty index is a FAILED READ, not a repo with no files. Reporting
        # "every claim is wrong" off that would be the exact class of error this
        # tool exists to catch.
        print("REFUSING: git ls-files returned nothing -- treating as a failed "
              "read, not as an empty repo")
        return 2

    rows = [l for l in text.splitlines() if l.strip().startswith("|")]
    checked = flagged = skipped = 0
    for row in rows:
        if row.strip().startswith("| #") or set(row.strip()) <= set("|- "):
            continue
        # ⛔ SKIP CLOSED ROWS -- AND THIS IS NOT A CONVENIENCE, IT IS A CORRECTNESS
        # FIX THE FIRST RUN FORCED. A row closed by writing *why* it was stale
        # ("the row's own '(absent)' annotation was STALE") still CONTAINS the
        # word "absent", so the checker read the post-mortem as a live claim and
        # flagged 7 rows that had already been fixed -- including the very row
        # that motivated the tool. That is the echo trap this programme has met
        # before: a filter matching text its own subject wrote about it. A closed
        # row is not a claim about the world; it is a record of one.
        if "~~" in row:
            skipped += 1
            continue
        for cell in row.split("|"):
            for m in FILE_RE.finditer(cell):
                name = m.group(1)
                base = name.rsplit("/", 1)[-1]
                hits = ([name] if name in tracked else []) or by_base.get(base, [])
                checked += 1

                # ⛔ PROXIMITY, NOT CELL-MEMBERSHIP. The first version asked
                # "does this CELL contain an absent-word", and on the real
                # backlog that was wrong every time it fired. Row F5 reads
                # "`onnx` absent (1), a Windows-basename assert in
                # `test_resim.py`" -- the absent-word belongs to a PACKAGE and
                # the filenames name where failures LIVE, so cell-membership
                # attributed the claim to two files nobody said were missing.
                # An absent-claim about a file is written NEXT TO the file.
                near = cell[max(0, m.start() - 40):m.end() + 40].lower()
                claims_absent = any(w in near for w in ABSENT_WORDS)

                # ⛔ "(new)" MARKS A FILE TO BE CREATED, NOT ONE THAT WENT
                # MISSING. Row F7 plans `stack/scripts/v6_preflight.sh` (new);
                # it is correctly untracked and flagging it invents work.
                planned = "(new)" in cell[m.end():m.end() + 12].lower()

                if claims_absent and hits:
                    flagged += 1
                    print("STALE-ABSENT  %s -- the row calls it absent/missing, "
                          "but it IS tracked at: %s"
                          % (base, ", ".join(sorted(hits)[:3])))
                elif (not claims_absent) and ("/" in name) and not hits \
                        and not planned:
                    flagged += 1
                    print("NOT-TRACKED   %s -- named as a repo path, absent from "
                          "the index" % name)

    print("checked %d file claim(s); %d row(s), %d closed row(s) skipped; %d problem(s)"
          % (checked, len(rows), skipped, flagged))
    # ⭐ Non-zero exit so this can gate, the way lab_backlog_drift.py does. A
    # check that only prints is a check somebody forgets to read.
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
