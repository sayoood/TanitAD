#!/usr/bin/env python3
"""make_podread_r1.py -- lead's tool (card ADD110, note N1): writes read/podread_fallattr10k_r1.sh as a copy of read/podread_fallattr10k.sh with ONE line changed, the line that reads the queue
start from the start marker. The queue writes the first token of the marker as seconds since the epoch; the driver as built handed that token to `date -d`, which refuses a bare number.
The new line takes the token as it is when it is a number. Asserted: the old line is there once, every other line is equal, the new file does not exist yet."""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "read", "podread_fallattr10k.sh")
DST = os.path.join(HERE, "read", "podread_fallattr10k_r1.sh")
OLD = '  QS=$(date -d "$(head -n 1 "$FA/FALLATTR10K_START.txt" | awk \'{print $1}\')" +%s) || stop "START-MARKER-UNREADABLE"'
NEW = '  QS=$(head -n 1 "$FA/FALLATTR10K_START.txt" | awk \'{print $1}\'); [[ "$QS" =~ ^[0-9]+$ ]] || stop "START-MARKER-UNREADABLE"'
if os.path.exists(DST):
    print("ZZ PODREAD-R1 STOP: the new file exists")
    sys.exit(2)
raw = io.open(SRC, encoding="utf-8", newline="").read()
eol = "\r\n" if "\r\n" in raw else "\n"
lines = raw.split(eol)
hits = [i for i, x in enumerate(lines) if x == OLD]
if len(hits) != 1:
    print("ZZ PODREAD-R1 STOP: old line found %d times" % len(hits))
    sys.exit(2)
out = list(lines)
out[hits[0]] = NEW
io.open(DST, "w", encoding="utf-8", newline="").write(eol.join(out))
back = io.open(DST, encoding="utf-8", newline="").read().split(eol)
diff = [i for i, (x, y) in enumerate(zip(lines, back)) if x != y]
ok = len(back) == len(lines) and diff == hits
print("ZZ PODREAD-R1 DONE line %d changed, lines %d, other lines equal %s" % (hits[0] + 1, len(lines), ok))
sys.exit(0 if ok else 2)
