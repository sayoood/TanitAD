#!/usr/bin/env python3
"""⛔ CLIP IDS APPEAR ONLY AS sha12 IN THE REPO. Rewrite every UUID-shaped clip id in a
banked artifact to `sha256(clip_id)[:12]`, in place, and REFUSE to leave one behind.

    python scrub_sha12.py <file> [<file> ...]

Prints the count per file. Run on every file before it is `git add`ed; the check at the
end of `RESULT.md`'s manifest is this script returning 0 replacements on a second pass.
"""
from __future__ import annotations

import hashlib
import re
import sys

UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


def sha12(m: re.Match) -> str:
    return hashlib.sha256(m.group(0).encode()).hexdigest()[:12]


def main() -> int:
    bad = 0
    for p in sys.argv[1:]:
        with open(p, encoding="utf-8", errors="surrogateescape") as fh:
            s = fh.read()
        s2, n = UUID.subn(sha12, s)
        if n:
            with open(p, "w", encoding="utf-8", errors="surrogateescape") as fh:
                fh.write(s2)
        left = len(UUID.findall(s2))
        bad += left
        print(f"{n:6d} replaced · {left} left · {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
