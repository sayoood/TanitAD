"""⛔ REDACT the run-directory path out of any banked artifact, before staging.

**This class has blocked a landing more than once**, and it blocked this one:
three of my raw JSONs carried the session scratchpad path — whose directory
name is a UUID — purely because they record which directory a run wrote to.
``tests/test_refcv6_no_session_paths.py`` forbids that pattern outright, for the
stated reason that *nothing can tell a session UUID from a clip UUID by looking
at it*.

⛔ IT REPLACES BY STRUCTURE, NOT BY THE LITERAL SESSION ID. Matching the one id
I happen to have would leave the next agent's artifacts broken in the same way,
and it would put the forbidden string INTO this file — the guard-spells-the-
pattern defect. So the scratch ROOT is passed in, and any UUID-shaped directory
segment is replaced generically.

Usage:  TAC_OUT=<scratch root> python redact.py <file> [<file> ...]
"""
from __future__ import annotations

import os
import re
import sys

_H = "[0-9a-fA-F]"
UUID = re.compile(f"{_H}{{8}}-{_H}{{4}}-{_H}{{4}}-{_H}{{4}}-{_H}{{12}}")
ROOT = os.environ.get("TAC_OUT", "")


def _variants(p: str):
    """A path can appear slash-, backslash- or json-escaped."""
    p = p.rstrip("/\\")
    yield p
    yield p.replace("/", "\\")
    yield p.replace("/", "\\").replace("\\", "\\\\")
    yield p.replace("\\", "/")


#: The worktree root, redacted for the SAME reason and by a SECOND rule: an
#: absolute HOME path is also forbidden in a repo artifact (it is allowed only
#: as an `os.environ.get(...)` default or in Markdown prose). A run record
#: naturally contains one for every input it opened.
WTROOT = os.environ.get("TAC_WT", "")

n_files = n_sub = 0
for f in sys.argv[1:]:
    s = open(f, encoding="utf-8").read()
    before = s
    for v in _variants(ROOT):
        if v:
            s = s.replace(v, "<SCRATCH>")
    for v in _variants(WTROOT):
        if v:
            s = s.replace(v, "<WORKTREE>")
    # whatever survives (e.g. a parent dir) loses its UUID segment
    s = UUID.sub("<REDACTED-UUID>", s)
    if s != before:
        open(f, "w", encoding="utf-8").write(s)
        n_sub += 1
    n_files += 1
    # ⛔ POSITIVE ASSERTION on the RESULT, never on an exit code.
    leftover = UUID.search(s)
    print(f"{os.path.basename(f)}: "
          f"{'CHANGED' if s != before else 'unchanged'} | "
          f"{'⛔ UUID REMAINS: ' + leftover.group(0) if leftover else 'clean'}")
print(f"{n_files} files, {n_sub} rewritten")
