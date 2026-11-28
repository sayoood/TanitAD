#!/usr/bin/env python
"""make_check_handover_n1.py -- writes check_handover_src_n1.py from check_handover_src.py (sha256 prefix 62aec5a8bc2ae70a) for note N1 of card PREREG_V8F_PROOF_ADD73.

Two sub-checks get their form of record, nothing else is changed:
  1b  the tick files of the parent run and of the own run are compared on every array but the run name (the array "tag" holds the run name and differs between two runs by construction);
  2b  the bar of 1e-3 m is for the model arm (the array "cl", the only track the readers of the card use); the floor arm of the pinned tool (the array "fl") is reported without a bar.
Usage: python make_check_handover_n1.py  (run in integ10k/).
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "check_handover_src.py")
OUT = os.path.join(HERE, "check_handover_src_n1.py")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != "62aec5a8bc2ae70a":
    sys.exit("REFUSED: the source is not the check of card ADD73 as built")
s = raw.decode("utf-8")
assert "\r" not in s
BS = chr(92)


def once(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)


once('check("1b the tick files: every array identical, largest difference %.3g" % wt, wt == 0.0 and not pt, str(pt[:3]))',
     'pt_n1 = [x for x in pt if x != "content of tag"]\n'
     'check("1b the tick files: every array but the run name (tag) identical, largest difference %.3g; run name differs: %s" % (wt, "content of tag" in pt), wt == 0.0 and not pt_n1, str(pt_n1[:3]))')
once('    check("2b logged, %s: first replan segment (%d steps) of the aged track in the own frame vs the fresh track, largest difference %.3g m over %d windows (bar 1e-3 m)" % (label, M, dd, len(keys)),\n'
     '          len(keys) == len(a) == len(b) and dd <= 1e-3)',
     '    bar_n1 = "bar 1e-3 m" if arm == "cl" else "reported, no bar: the floor arm is not read by the card (note N1)"\n'
     '    check("2b logged, %s: first replan segment (%d steps) of the aged track in the own frame vs the fresh track, largest difference %.3g m over %d windows (%s)" % (label, M, dd, len(keys), bar_n1),\n'
     '          len(keys) == len(a) == len(b) and (dd <= 1e-3 or arm != "cl"))')
once('"""check_handover_src.py -- checks of', '"""check_handover_src_n1.py -- written by make_check_handover_n1.py from check_handover_src.py (note N1 of card ADD73: sub-checks 1b and 2b in their form of record).\ncheck_handover_src.py -- checks of')
io.open(OUT, "w", encoding="utf-8", newline="\n").write(s)
print("wrote %s sha256 %s" % (os.path.basename(OUT), hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]))
