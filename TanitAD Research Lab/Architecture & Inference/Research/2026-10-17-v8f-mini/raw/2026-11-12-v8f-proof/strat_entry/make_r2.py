"""make_r2.py : reader versions r2 for the STRATEGIC entry (row H-V8F-PROOF-83, work item 1).

The PI ended the pause of the STRATEGIC family on 2026-11-16. Three readers of this study write the pause stamp as a fixed string. This script writes a second version of
each reader beside the first one (the first versions are banked and stay untouched): the ONLY change is the text of the STRATEGIC entry. It refuses to overwrite, and it
checks that exactly the expected lines differ and that nothing else does.
"""
import io
import os
import sys

ROOT = "F:/TanitAD-scratch/v8proof"
OLD = "PAUSED (PI item 25)"
NEW = "NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)"
FILES = (("spatial/q3/q3_read.py", "spatial/q3/q3_read_r2.py", 2), ("headway/headway_read.py", "headway/headway_read_r2.py", 2), ("imaghead/imag_read.py", "imaghead/imag_read_r2.py", 2))
ok = True
for src, dst, n_exp in FILES:
    ps, pd = os.path.join(ROOT, src), os.path.join(ROOT, dst)
    b = io.open(ps, "rb").read()
    n = b.count(OLD.encode("utf-8"))
    nb = b.replace(OLD.encode("utf-8"), NEW.encode("utf-8"))
    if os.path.exists(pd):
        same = io.open(pd, "rb").read() == nb
        print("exists", dst, "equal to the expected content:", same)
        ok = ok and same
        continue
    if n != n_exp:
        print("REFUSED", src, "has", n, "occurrences, expected", n_exp)
        ok = False
        continue
    with io.open(pd, "wb") as fh:
        fh.write(nb)
    la, lb = b.split(b"\n"), nb.split(b"\n")
    diff = [i + 1 for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    only = all(la[i - 1].replace(OLD.encode(), NEW.encode()) == lb[i - 1] for i in diff)
    print("written", dst, "lines", len(lb), "same line count", len(la) == len(lb), "differing lines", diff, "each differs by the entry only", only, "CR bytes", nb.count(b"\r"), "of", b.count(b"\r"))
    ok = ok and len(la) == len(lb) and len(diff) == n_exp and only
print("ZZ MAKE-R2", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
