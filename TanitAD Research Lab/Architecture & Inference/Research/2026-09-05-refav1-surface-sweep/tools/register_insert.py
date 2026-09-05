"""INSERT one row into a register table, on the line after the line starting with AFTER.

Read-modify-write with retries on the flapping G: mount, idempotent by the row's own id marker,
verified by re-reading. NEVER rewrites any other line, and REFUSES if the output would be
shorter than the input.

Usage: python register_insert.py <file> <after_marker_file> <row_file> <row_id_marker>

MEASURED 2026-09-05 and the reason this is not the predecessor's `insert_rows.py`:
`GOALS_AND_CLAIMS.md` has MIXED line endings — 2,776 LF and 24 CR. A `"\\r\\n" in s` test is
therefore True, and splitting a 2,776-line file on CRLF yields 25 "lines"; joining that back
would MANGLE the whole register. Splitting on LF keeps each CRLF line's trailing CR inside the
line, so an LF join round-trips the bytes exactly whatever the mix. The same flaw is live in
`Research/2026-09-05-refav1-ccos-eval/tools/insert_rows.py` and is escalated, not silently
worked around here.
"""
import os
import sys
import time

LF = chr(10)


def rd(p):
    for i in range(30):
        try:
            with open(p, encoding="utf-8", newline="") as fh:
                return fh.read()
        except OSError as e:
            print("  retry read", i, os.path.basename(p), e, flush=True)
            time.sleep(10)
    raise SystemExit("cannot read " + p)


def wr(p, s):
    for i in range(30):
        try:
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(s)
            if rd(p) == s:
                return True
            print("  re-read differs after write; retrying", flush=True)
        except OSError as e:
            print("  retry write", i, os.path.basename(p), e, flush=True)
            time.sleep(10)
    return False


def main():
    path, after_f, row_f, row_mark = sys.argv[1:5]
    after = rd(after_f).strip()
    row = rd(row_f).strip()
    s = rd(path)
    lines = s.split(LF)
    if any(l.startswith(row_mark) for l in lines):
        print("PRESENT already:", row_mark)
        return
    hit = [i for i, l in enumerate(lines) if l.startswith(after)]
    if not hit:
        raise SystemExit("ANCHOR NOT FOUND: " + after[:80])
    lines.insert(hit[0] + 1, row)
    out = LF.join(lines)
    if len(out) < len(s):
        raise SystemExit("REFUSING: output shorter than input")
    if not wr(path, out):
        raise SystemExit("write failed")
    got = rd(path).split(LF)
    n = sum(1 for l in got if l.startswith(row_mark))
    print("INSERTED after line %d; row present %dx; file %d -> %d chars"
          % (hit[0] + 1, n, len(s), len(out)))


if __name__ == "__main__":
    main()
