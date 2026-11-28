#!/usr/bin/env python3
"""Idempotent, marker-guarded APPEND to the live claims register.

⛔ APPEND ONLY. A sibling's register block was silently swept today by another
agent rewriting the file from an older base, so this never rewrites: it reads,
checks for its own marker, and appends only if absent.
⚠️ The G: mount flaps, so the read is retried and the write is verified by
re-reading the marker back. ASCII in print() only (cp1252 dev box).
"""
import io
import os
import sys
import time

REG = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Project Steering"
       r"\GOALS_AND_CLAIMS.md")
BLOCK = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\wkfront\register_block.md"
MARKER = sys.argv[2] if len(sys.argv) > 2 else "WKAPPA-FRONTIER-2026-09-06"


def read_retry(path, tries=8):
    """⚠️ A failed read on this mount is INCONCLUSIVE, never 'absent'."""
    for i in range(tries):
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                s = f.read()
            if s:
                return s
        except OSError as e:
            print("  read attempt %d failed: %s" % (i + 1, e.__class__.__name__))
        time.sleep(3)
    return None


def main():
    with io.open(BLOCK, "r", encoding="utf-8") as f:
        block = f.read()
    if MARKER not in block:
        print("REFUSED: the block does not carry its own marker %r" % MARKER)
        return 2
    cur = read_retry(REG)
    if cur is None:
        print("INCONCLUSIVE: could not read the register (mount). Nothing written.")
        return 3
    print("register read OK: %d bytes, %d lines" % (len(cur), cur.count("\n") + 1))
    if MARKER in cur:
        print("ALREADY PRESENT: marker %r found -- no write (idempotent)." % MARKER)
        return 0
    new = cur.rstrip("\n") + "\n" + block.rstrip("\n") + "\n"
    tmp = REG + ".wkf.tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    os.replace(tmp, REG)
    back = read_retry(REG)
    if back is None:
        print("WROTE but VERIFY INCONCLUSIVE (mount) -- re-check by hand.")
        return 4
    ok = MARKER in back
    grew = len(back) - len(cur)
    print("VERIFY: marker present=%s  grew_by=%d bytes  now %d lines"
          % (ok, grew, back.count("\n") + 1))
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
