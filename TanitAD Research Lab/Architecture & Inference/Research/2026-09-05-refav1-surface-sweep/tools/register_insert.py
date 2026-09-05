"""INSERT one row into a register table, after the line that starts with MARKER.

Read-modify-write with retries on the flapping G: mount, idempotent by the row's own id
marker, verified by re-reading. NEVER rewrites any other line, and REFUSES if the output
would be shorter than the input.

Usage: python register_insert.py <file> <after_marker_file> <row_file> <row_id_marker>
"""
import os
import sys
import time


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
    nl = "\r\n" if "\r\n" in s else "\n"
    lines = s.split(nl)
    if any(l.startswith(row_mark) for l in lines):
        print("PRESENT already:", row_mark)
        return
    hit = [i for i, l in enumerate(lines) if l.startswith(after)]
    if not hit:
        raise SystemExit("ANCHOR NOT FOUND: " + after[:80])
    lines.insert(hit[0] + 1, row)
    out = nl.join(lines)
    if len(out) < len(s):
        raise SystemExit("REFUSING: output shorter than input")
    if not wr(path, out):
        raise SystemExit("write failed")
    got = rd(path).split(nl)
    n = sum(1 for l in got if l.startswith(row_mark))
    print(f"INSERTED after line {hit[0]+1}; row present {n}x; file {len(s)} -> {len(out)} B")


if __name__ == "__main__":
    main()
