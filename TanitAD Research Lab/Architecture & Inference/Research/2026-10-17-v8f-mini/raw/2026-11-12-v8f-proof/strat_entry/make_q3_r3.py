"""make_q3_r3.py -- writes spatial/q3/q3_read_r3.py from spatial/q3/q3_read_r2.py.

Found on 2026-11-19 while reading the dose arms: the verdict of the r2 family reader states the STRATEGIC entry as NOT COMPUTED in its header, and its section
"Not measured by this read" still printed the engine's fixed list RH.GAPS, whose STRATEGIC line carries the pause text of PI item 25. The PI ruling of
2026-11-16 ended that pause. r3 changes that one list entry and nothing else: one edit, asserted to match exactly once; the engine file and the r2 reader stay
as banked. Every number of a read is unchanged (checked by strat_entry/cmp_json.py on the read of the dose arms).
"""
import io
import os

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_p = os.path.join(H, "spatial", "q3", "q3_read_r2.py")
dst_p = os.path.join(H, "spatial", "q3", "q3_read_r3.py")
src = io.open(src_p, encoding="utf-8", newline="").read()
old = '"gaps": list(RH.GAPS),'
new = ('"gaps": [("STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; '
       'PI ruling of 2026-11-16)" if g.startswith("STRATEGIC") else g) for g in RH.GAPS],')
assert src.count(old) == 1, src.count(old)
out = src.replace(old, new)
with io.open(dst_p, "w", encoding="utf-8", newline="") as fh:
    fh.write(out)
print("q3_read_r3.py written: lines", len(src.splitlines()), "->", len(out.splitlines()), "; CR count", out.count("\r"))
