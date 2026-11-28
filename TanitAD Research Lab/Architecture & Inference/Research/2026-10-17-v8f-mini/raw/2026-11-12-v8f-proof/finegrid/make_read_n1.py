"""make_read_n1.py -- writes finegrid/finegrid_read_n1.py from finegrid/finegrid_read.py (card PREREG_V8F_PROOF_ADD41, note N1).

The reader has to run on the pod (its state files do not fit on the dev box). There the reproduction of the banked TK values (control g7, TK side) differs
from the banked numbers in the seventh decimal (another numpy and CPU), measured before any pilot number existed. Note N1 gives that part of the control a
tolerance on the pod. The only change is an argument --g7-tol (default 0.0, which is the first version's behaviour: for a difference that is never negative,
`<= 0.0` and `== 0.0` are the same test). Four edits, each asserted to match exactly once; finegrid_read.py stays as built and tested.
"""
import io
import os

H = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(H, "finegrid_read.py"), encoding="utf-8", newline="").read()
edits = [
    ('def repro_row(R, banked):\n',
     'def repro_row(R, banked, tol=0.0):\n'),
    ('    return {"pass": bool(mx == 0.0), "max_abs_diff": mx, "by_part": d}\n',
     '    return {"pass": bool(mx <= tol), "max_abs_diff": mx, "tol": tol, "by_part": d}\n'),
    ('            repro[arm] = repro_row(R[arm], banked[arm])\n',
     '            repro[arm] = repro_row(R[arm], banked[arm], a.g7_tol)\n'),
    ('    ap.add_argument("--B", type=int, default=2000)\n',
     '    ap.add_argument("--g7-tol", type=float, default=0.0, help="note N1: tolerance of the TK reproduction of control g7 (0.0 = the card as frozen; 1e-5 on the pod)")\n'
     '    ap.add_argument("--B", type=int, default=2000)\n'),
]
out = src
for old, new in edits:
    assert out.count(old) == 1, (old, out.count(old))
    out = out.replace(old, new)
with io.open(os.path.join(H, "finegrid_read_n1.py"), "w", encoding="utf-8", newline="") as fh:
    fh.write(out)
print("finegrid_read_n1.py written: lines", len(src.splitlines()), "->", len(out.splitlines()), "; CR count", out.count("\r"))
