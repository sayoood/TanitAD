"""make_read_k8.py -- writes imagplan/read_k8.py from imagplan/read_k.py (card PREREG_V8F_PROOF_ADD40, note N6).

read_k.py reads the source cand-imagkin on one arm at the bases 0 and 1 (fixed in its loop). Note N6 repeats the read at the bases 2 and 3. The only
change is that the two bases come from an argument (--bases, default "0,1", which gives the first version's behaviour). Two edits, each asserted to
match exactly once; read_k.py stays as banked. Written 2026-11-18 before any tool run on the new bases existed.
"""
import io
import os

H = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(H, "read_k.py"), encoding="utf-8", newline="").read()
edits = [
    ('    ap.add_argument("--out", required=True)\n',
     '    ap.add_argument("--bases", default="0,1")                     # note N6: the two bases of the read (first version: 0 and 1)\n'
     '    ap.add_argument("--out", required=True)\n'),
    ('    for k in (0, 1):\n',
     '    for k in [int(x) for x in a.bases.split(",")]:\n'),
]
out = src
for old, new in edits:
    assert out.count(old) == 1, (old, out.count(old))
    out = out.replace(old, new)
with io.open(os.path.join(H, "read_k8.py"), "w", encoding="utf-8", newline="") as fh:
    fh.write(out)
a, b = src.splitlines(), out.splitlines()
print("read_k8.py written: lines", len(a), "->", len(b), "; CR count", out.count("\r"))
