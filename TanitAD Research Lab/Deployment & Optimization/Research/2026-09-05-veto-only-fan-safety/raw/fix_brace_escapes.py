#!/usr/bin/env python3
"""Fix three leaked `{{...}}` format-brace escapes in this session's register rows.

The row text was built with `str.replace` on placeholders, not `str.format`, so the doubled
braces I wrote to escape `{0,1}` were never collapsed and shipped literally as
`raw/run/s{{0,1}}/veto200/arm_summary.json`. Cosmetic, but the artifact path is the part of a
register row a reader actually follows, and a path that does not exist is worse than none.
Scoped to the three rows this session added.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF

raw = open(PATH, "rb").read()
is_crlf = CRLF.encode() in raw
s = raw.decode("utf-8").replace(CRLF, LF)
n = s.count("{{")
if n == 0:
    raise SystemExit("[braces] none present - nothing to fix")
s2 = s.replace("{{", "{").replace("}}", "}")
open(PATH, "wb").write((s2.replace(LF, CRLF) if is_crlf else s2).encode("utf-8"))
print("[braces] collapsed %d doubled-brace escape(s) (%s)"
      % (n, "CRLF" if is_crlf else "LF"))
