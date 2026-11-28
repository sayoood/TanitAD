#!/usr/bin/env python3
"""Refuse a commit whose index spans several agents' in-flight deliverable dirs.

**This exists because a reminder was not enough.** The whole-index sweep has now
happened **four** times in one session (`7e6b123`, `2dc2795`, `19a0b87`, `6bf905d`):
each time a pathspec-free ``git commit`` swept a sibling agent's staged files into a
commit whose subject was about something else. Nothing was ever lost — the content is
in HEAD and byte-identical — but **the work becomes unfindable by commit message**,
which is the whole point of a commit message.

The escalation that produced this tool said it exactly right: *"the rule needs a
mechanism (e.g. a hook refusing an index spanning multiple ``incoming/`` dirs), not
another reminder."*

Why it is a WARNING and not a hard refusal
------------------------------------------
``git commit -- <pathspec>`` **segfaults on this repo** (measured 2026-07-25: exit 139
under MSYS git *and* ``0xC0000005`` under native Windows git — not the shell, not
fsmonitor). So the pathspec-free whole-index commit is the *documented, working*
procedure, and a hard block would leave no way to commit at all. CLAUDE.md's rule is
therefore: whole-index is admissible **after listing the index and confirming every
entry is intended program work**, naming siblings in the message.

This tool automates the *listing* half — the half a human reliably skips.

Usage
-----
    python tools/index_span_check.py            # report; exit 1 if it spans
    python tools/index_span_check.py --quiet    # exit code only

Exit 1 means "your commit message must name every directory listed"; it does not mean
"do not commit".
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

# The console here is cp1252; a non-ASCII byte in a WARNING must never be the
# thing that stops the warning being seen. (This tool crashed on its own emoji.)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass
from collections import defaultdict

# A deliverable dir is one agent's work-product: ".../incoming/<dated-stream>/..."
_STREAM = re.compile(r"^(?P<root>.*?/incoming/[^/]+)/")


def staged_paths() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                         capture_output=True, text=True, check=False, encoding="utf-8").stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def group(paths: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    streams: dict[str, list[str]] = defaultdict(list)
    other: list[str] = []
    for p in paths:
        m = _STREAM.match(p)
        (streams[m.group("root")] if m else other).append(p)
    return dict(streams), other


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    paths = staged_paths()
    if not paths:
        if not a.quiet:
            print("index_span_check: nothing staged.")
        return 0

    streams, other = group(paths)
    if not a.quiet:
        print(f"index_span_check: {len(paths)} staged file(s), "
              f"{len(streams)} deliverable stream(s).")
        for root, files in sorted(streams.items()):
            print(f"  [{len(files):3d}]  {root}")
        if other:
            print(f"  [{len(other):3d}]  (outside incoming/: code, steering, tests)")

    if len(streams) <= 1:
        return 0

    if not a.quiet:
        print()
        print("!! THE INDEX SPANS MULTIPLE AGENTS' DELIVERABLE DIRS.")
        print("    This is ADMISSIBLE — pathspec commits segfault on this repo, so")
        print("    whole-index is the documented path — but the commit message MUST")
        print("    NAME every directory above, or that work becomes unfindable by")
        print("    message. Four sweeps in one session is what earned this check.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
