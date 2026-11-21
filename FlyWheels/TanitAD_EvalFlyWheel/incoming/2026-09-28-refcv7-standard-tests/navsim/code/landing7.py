#!/usr/bin/env python3
"""Write ``LANDING_READY.txt`` for the Master Mind (the single committer; PI ruling 2026-09-26).
Any python. This stream never runs ``git add``.

    python code/landing7.py            # -> LANDING_READY.txt (+ prints the refusals, if any)

Format (the brief): a heading ``## 2026-09-28 REFCV7-NAVSIM``, then one path per line relative to
``D:/Projects/TanitAD``, each followed by a ``#`` comment carrying its ``git hash-object
--no-filters`` blob and size. Included: every file under the package EXCEPT the excluded globs
below. REFUSED (listed, never landed): a file over 20 MB, and a file whose text carries a canonical
UUID (8-4-4-4-12 hex) -- PhysicalAI clip ids never land (clip-id rule).
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = "D:/Projects/TanitAD"
MAX_BYTES = 20 * 2**20
UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
EXCLUDE = (
    "LANDING_READY.txt", "*/__pycache__/*", "*.pyc", "*.nohup.txt",
    "raw/inputs/speed_limits_navtest.json",         # a byte-identical gunzip of a landed .gz
    "*/score_*_wrapper/*_hooks*", "*.tmp",
    # LIVE at hand-over (still being written; they land in the next batch): the navhard KH chain,
    # its log, and the armed waiter's state
    "raw/harness_repro_navhard/*", "raw/harness7.log", "raw/harness7b.nohup.txt",
    "raw/milestones/*", "raw/floors/navhard_two_stage/*",
)
LIVE_NOTE = ("# NOT IN THIS BATCH (live at hand-over): raw/harness_repro_navhard/ + "
             "raw/floors/navhard_two_stage/ (the navhard KH, running), raw/harness7.log, "
             "raw/milestones/ (the armed step-5,000 waiter and, later, its outputs)")


def blob(p: str) -> str:
    r = subprocess.run(["git", "hash-object", "--no-filters", p], capture_output=True, text=True)
    b = r.stdout.strip()
    return b if len(b) == 40 else "INCONCLUSIVE"


def main() -> int:
    lines = ["## 2026-09-28 REFCV7-NAVSIM",
             "# EvalFlyWheel NavSim stream: the refcv7 NavSim suite (SPEC + amendment A1, code, tests,",
             "# harness known values, step-1,500 VALIDATION ONLY run, the armed milestone waiter).",
             "# Every refcv7 number in this package below step 5,000 is VALIDATION ONLY (SPEC §6)."]
    refused = []
    n = 0
    for dp, dns, fns in os.walk(PKG):
        dns.sort()
        for fn in sorted(fns):
            full = os.path.join(dp, fn)
            rel_pkg = os.path.relpath(full, PKG).replace("\\", "/")
            if any(fnmatch.fnmatch(rel_pkg, g) or fnmatch.fnmatch("/" + rel_pkg, "*/" + g)
                   for g in EXCLUDE):
                continue
            size = os.path.getsize(full)
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            if size > MAX_BYTES:
                refused.append(f"# REFUSED (> 20 MB, {size} B): {rel}")
                continue
            with open(full, "rb") as fh:
                if UUID.search(fh.read()):
                    refused.append(f"# REFUSED (canonical UUID in the text): {rel}")
                    continue
            lines.append(rel)
            lines.append(f"#   blob {blob(full)}  {size} B")
            n += 1
    lines += refused
    lines.append(LIVE_NOTE)
    with open(os.path.join(PKG, "LANDING_READY.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"LANDING_READY.txt: {n} paths, {len(refused)} refused")
    for r in refused:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
