#!/usr/bin/env python3
"""Is this path in HEAD?  -- the ONLY admissible way to ask, on this mount.

WHY THIS EXISTS.  On the G: mount a failed git-object read is INDISTINGUISHABLE from a
genuine absence: `git show HEAD:<path>` returns empty and exits non-zero in BOTH cases,
and `git rev-parse` reports "fatal: not a git repository" while `.git/HEAD` reads fine.

That ambiguity has produced FOUR wrong answers about the SAME files (refcv5's model
seams): two of mine, one from a completeness review, and one from the Training FlyWheel
-- each reported "not in HEAD" from a read taken during a flap.  The files were in HEAD
every time.

THE FIX IS NOT MORE CARE, IT IS A THIRD OUTCOME.  A presence check must be able to say
INCONCLUSIVE, and must prove its own channel with a control that MUST read non-zero in
the same breath.  Exit codes:

    0  every path is IN HEAD
    1  the channel is proven UP and at least one path is genuinely ABSENT
    3  INCONCLUSIVE -- the channel could not be proven; the result says nothing

Never treat 3 as 1.  "0 hits" is a claim about the SEARCH unless the control passed.

ASCII only (this box is cp1252 and non-ASCII in print() is fatal).
"""
from __future__ import annotations

import subprocess
import sys
import time

#: A path that must exist in HEAD and be large.  If reading THIS fails, nothing else
#: read in the same breath is admissible.
CONTROL_PATH = "CLAUDE.md"
CONTROL_MIN_LINES = 100

EXIT_ALL_PRESENT = 0
EXIT_SOME_ABSENT = 1
EXIT_INCONCLUSIVE = 3


def _show(path: str) -> str | None:
    """`git show HEAD:<path>`, or None if the command itself failed."""
    try:
        r = subprocess.run(["git", "show", f"HEAD:{path}"],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def _lines(text: str | None) -> int:
    return 0 if not text else len(text.splitlines())


def channel_is_up() -> tuple[bool, int]:
    """Prove the git object channel with a control that must read non-zero."""
    n = _lines(_show(CONTROL_PATH))
    return n >= CONTROL_MIN_LINES, n


def wait_for_channel(attempts: int = 1, sleep_s: float = 15.0) -> tuple[bool, int]:
    up, n = channel_is_up()
    for _ in range(max(0, attempts - 1)):
        if up:
            break
        time.sleep(sleep_s)
        up, n = channel_is_up()
    return up, n


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    attempts = 1
    if "--wait" in argv:
        i = argv.index("--wait")
        argv.pop(i)
        attempts = int(argv.pop(i)) if i < len(argv) and argv[i].isdigit() else 40
    if not argv:
        print("usage: in_head.py [--wait N] <path> [<path> ...]", file=sys.stderr)
        return EXIT_INCONCLUSIVE

    up, ctl = wait_for_channel(attempts)
    print(f"CONTROL {CONTROL_PATH} = {ctl} lines from HEAD "
          f"(need >= {CONTROL_MIN_LINES})")
    if not up:
        print("INCONCLUSIVE: the git object channel could not be proven. "
              "This says NOTHING about the paths -- do not read it as absence.")
        return EXIT_INCONCLUSIVE

    absent = []
    for p in argv:
        n = _lines(_show(p))
        if n > 0:
            print(f"  IN_HEAD  {n:6d} lines  {p}")
        else:
            print(f"  ABSENT              {p}")
            absent.append(p)
    if absent:
        print(f"ABSENT (channel proven up): {len(absent)} of {len(argv)}")
        return EXIT_SOME_ABSENT
    print(f"ALL PRESENT: {len(argv)} of {len(argv)}")
    return EXIT_ALL_PRESENT


if __name__ == "__main__":
    sys.exit(main())
