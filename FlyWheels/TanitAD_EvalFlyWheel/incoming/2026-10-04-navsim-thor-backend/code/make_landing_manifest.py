#!/usr/bin/env python3
"""Write LANDING_READY_NAVSIM_THOR.txt: ``<md5>  <repo-relative path>  -- <what>`` for the backend package
AND the Thor-only score tree under refcv7 step50400/thor_scores/ (long paths read via the \\\\?\\ prefix)."""
import hashlib
import os

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend"
TREL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/thor_scores"
TS = "D:/Projects/TanitAD/" + TREL


def long(p):
    p = os.path.abspath(p)
    return "\\\\?\\" + p if os.name == "nt" else p


lines, bad = [], 0
for base, rel, what in ((PK, REL, "backend package"), (TS, TREL, "Thor production scores (backend: thor)")):
    for root, _, files in os.walk(long(base)):
        for f in sorted(files):
            if f.endswith(".pid") or f.startswith("LANDING_READY"):
                continue
            full = os.path.join(root, f)
            r = os.path.relpath(full, long(base)).replace(os.sep, "/")
            try:
                md5 = hashlib.md5(open(full, "rb").read()).hexdigest()
            except OSError:
                md5, bad = "UNREADABLE", bad + 1
            lines.append(f"{md5}  {rel}/{r}  -- {what}")
lines.sort(key=lambda x: x.split("  ")[1])
with open(os.path.join(PK, "LANDING_READY_NAVSIM_THOR.txt"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# EvalFlyWheel navsim-thor-backend + Thor production (2026-10-04/05). NOT staged / NOT committed. "
             "md5  repo-relative path  -- what\n# Includes the Thor-only score tree under refcv7 step50400/thor_scores/ "
             "(never the milestone's standard names). *.pid excluded.\n" + "\n".join(lines) + "\n")
print(len(lines), "files;", bad, "unreadable")
