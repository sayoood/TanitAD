#!/usr/bin/env python3
"""Fix the --analyze-only fallback: it takes the DUMP DIR as its VALUE, not a bare flag.

`openloop_suite.py:2256` declares `--analyze-only` with `metavar="DUMP_DIR"`, so
`--analyze-only --dump-dir <dir>` would have swallowed `--dump-dir` as the value and then
failed on a directory literally named "--dump-dir". The whole point of that branch is to
recover a COMPLETED rollout whose analysis step died -- a recovery path that itself fails is
worse than none, because it turns a recoverable run into a re-roll.

Verified against the tool's own usage block (line 85):
    openloop_suite.py --analyze-only <dump-dir> --out-dir <dir> --tag <tag>
"""
import io
import os

P = r"C:\Users\Admin\veto_run\run_eval_veto.sh"
LF = chr(10)
BS = chr(92)

s = io.open(P, encoding="utf-8").read().replace(chr(13) + LF, LF)
old = ('      "$PY" -u "$REPO/taniteval/tools/openloop_suite.py" --analyze-only ' + BS + LF
       + '         --dump-dir "$OUT/eval/${tag}_dump" --out-dir "$OUT/eval" ' + BS + LF)
new = ('      "$PY" -u "$REPO/taniteval/tools/openloop_suite.py" ' + BS + LF
       + '         --analyze-only "$OUT/eval/${tag}_dump" --out-dir "$OUT/eval" ' + BS + LF)
if new in s:
    raise SystemExit("[patch] already fixed")
if s.count(old) != 1:
    raise SystemExit("[patch] anchor found %d times" % s.count(old))
io.open(P, "w", encoding="utf-8", newline=LF).write(s.replace(old, new))
print("[patch] --analyze-only now takes the dump dir as its value")
