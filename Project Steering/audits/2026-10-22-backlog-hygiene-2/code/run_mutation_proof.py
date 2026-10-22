#!/usr/bin/env python3
"""Mutation proof for BACKLOG-HYGIENE-2's `tools/backlog_drift.py` changes.

For each mutation: copy the tool + its tests + fixture into a scratch tree, patch ONE
thing in the tool's source, run the suite, record the exit code and the tests that
went RED. The UNMUTATED copy runs first and MUST be green (the control: a harness
that can only report red proves nothing). The tool in `code/` is never touched --
its md5 is checked before and after.

    python run_mutation_proof.py <BACKLOG.md for the live-file tests> > mutation_check.out

A mutation that leaves the suite GREEN is a finding, not a footnote: it names a
branch no test can see. The exit code is the number of SURVIVORS.
"""
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL_REL = "tools/backlog_drift.py"
TEST_REL = "tools/tests/test_backlog_drift.py"
FIX_REL = "tools/tests/fixtures/backlog_rows_2026-10-22.md"

#: (name, what it disables, find, replace)
MUTATIONS = [
    ("M1_later_closure_off",
     "CLOSED-LATER detection disabled entirely (later_closure always None)",
     "    if lead_status(body) not in OPEN_LEAD_WORDS:\n        return None\n",
     "    return None\n"),
    ("M2_named_family_unread",
     "the named-id table family is not parsed (back to the pre-2026-10-22 tool)",
     "        if not m and in_item_table:\n",
     "        if False:\n"),
    ("M3_date_requirement_dropped",
     "a closure word closes WITHOUT a date",
     'r"(?:%s)\\b(?![-_][a-z])[^0-9]{0,24}?%s" % (_CLOSE_ALT, _DATE))',
     'r"(?:%s)\\b(?![-_][a-z])" % (_CLOSE_ALT,))'),
    ("M4_compound_guard_dropped",
     "`DONE-marker ... <date>` reads as a closure (compound lock removed)",
     'r"(?:%s)\\b(?![-_][a-z])[^0-9]{0,24}?%s" % (_CLOSE_ALT, _DATE))',
     'r"(?:%s)\\b[^0-9]{0,24}?%s" % (_CLOSE_ALT, _DATE))'),
    ("M5_qualified_not_last_only",
     "a qualified DONE closes even when something follows it",
     "        if i == len(spans) - 1 and _QUALIFIED_DONE.match(sp):\n",
     "        if _QUALIFIED_DONE.match(sp):\n"),
    ("M6_update_branch_off",
     "`UPDATE <date>: superseded` is not read",
     "        if _UPDATE_CLOSURE.match(sp):\n",
     "        if False:\n"),
    ("M7_resolved_for_any_lead",
     "`RESOLVED` closes a row under ANY lead, not only BLOCKER",
     "        if blocker and _DATED_RESOLVED.match(sp):\n",
     "        if _DATED_RESOLVED.match(sp):\n"),
    ("M8_item_table_gate_off",
     "named ids are read in ANY table (G-LIVE / G-SUITE become backlog rows)",
     "        if not m and in_item_table:\n",
     "        if not m:\n"),
    ("M9_unparsed_table_rows_swallowed",
     "an item-table row no regex reads is NOT reported",
     '        if in_item_table and s.startswith("|") and not TABLE_SEPARATOR.match(s):\n',
     "        if False:\n"),
    ("M10_classify_ignores_closed_later",
     "classify never emits CLOSED-LATER (rows fall back to DONE-BUT-OPEN)",
     "        elif first.closed_later:\n",
     "        elif False:\n"),
    ("M11_lead_gate_removed",
     "any lead may make a row CLOSED-LATER (the open-type lead gate removed)",
     "    if lead_status(body) not in OPEN_LEAD_WORDS:\n        return None\n",
     "    if False:\n        return None\n"),
    ("M12_cancelled_not_a_closure_word",
     "`CANCELLED` is no longer a closure word",
     'CLOSURE_HEAD_WORDS = DONE_WORDS + ("withdrawn", "cancelled", "canceled")',
     'CLOSURE_HEAD_WORDS = DONE_WORDS + ("withdrawn",)'),
]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def stage(dst: Path, backlog: Path) -> None:
    for rel in (TOOL_REL, TEST_REL, FIX_REL):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / rel, dst / rel)
    bl = dst / "Project Steering" / "BACKLOG.md"
    bl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(backlog, bl)


def run_suite(root: Path):
    r = subprocess.run(
        ["uv", "run", "--no-project", "--with", "pytest", "python", "-m", "pytest",
         "-q", "-p", "no:cacheprovider", "-rf", TEST_REL],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    red = sorted(set(re.findall(r"^FAILED\s+\S+::(\S+)", r.stdout, re.M)))
    tail = [ln for ln in r.stdout.splitlines()
            if re.search(r"^\d+ (passed|failed)|, \d+ passed|^=+ .*(passed|failed)", ln)
            and not ln.startswith("FAILED")]
    return r.returncode, red, (tail[-1] if tail else r.stdout[-200:])


def main() -> int:
    backlog = Path(sys.argv[1])
    before = md5(HERE / TOOL_REL)
    print("tool md5 before: %s" % before)
    src = (HERE / TOOL_REL).read_text(encoding="utf-8")
    survivors = 0

    with tempfile.TemporaryDirectory(prefix="blhyg_mut_") as td:
        root = Path(td)
        stage(root, backlog)
        rc, red, tail = run_suite(root)
        print("\nCONTROL (unmutated)       rc=%d  %s" % (rc, tail))
        if rc != 0:
            print("  CONTROL IS RED -- the harness proves nothing; failing: %s" % red)
            return 99

        for name, what, find, repl in MUTATIONS:
            if find not in src:
                print("\n%-34s PATCH TARGET NOT FOUND -- mutation not applied" % name)
                survivors += 1
                continue
            assert src.count(find) >= 1
            (root / TOOL_REL).write_text(src.replace(find, repl, 1), encoding="utf-8")
            rc, red, tail = run_suite(root)
            verdict = "RED (killed)" if rc != 0 else "GREEN (SURVIVED)"
            if rc == 0:
                survivors += 1
            print("\n%-34s rc=%d  %s" % (name, rc, verdict))
            print("  mutation : %s" % what)
            print("  result   : %s" % tail)
            for t in red[:12]:
                print("    RED  %s" % t)
            if len(red) > 12:
                print("    ... and %d more" % (len(red) - 12))
        (root / TOOL_REL).write_text(src, encoding="utf-8")     # restore the copy

    after = md5(HERE / TOOL_REL)
    print("\ntool md5 after : %s  (%s)" % (after, "UNCHANGED" if after == before
                                         else "CHANGED -- BUG IN THIS HARNESS"))
    print("survivors: %d of %d" % (survivors, len(MUTATIONS)))
    return survivors


if __name__ == "__main__":
    sys.exit(main())
