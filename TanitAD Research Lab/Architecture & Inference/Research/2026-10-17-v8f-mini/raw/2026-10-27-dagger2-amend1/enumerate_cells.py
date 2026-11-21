"""PREREG_V8F_DAGGER2 Amendment 1: the 16 guarded (R-CRAWL, R-MOVE) x 2-seed cells under the table as landed and as amended.
python enumerate_cells.py   (prints the counts the amendment quotes; exits 1 if any placed cell moves)"""
import itertools
from collections import Counter

old, new = {}, {}
for c0, c1, m0, m1 in itertools.product([1, 0], repeat=4):
    cr, mv = (c0, c1), (m0, m1)
    both = [cr[i] and mv[i] for i in range(2)]
    fixed, pa, ne = all(both), all(mv) and not all(cr), not any(mv)
    old[(c0, c1, m0, m1)] = "FIXED" if fixed else "PARTIAL" if (pa or sum(both) == 1) else "NO EFFECT" if ne else None
    new[(c0, c1, m0, m1)] = "FIXED" if fixed else "PARTIAL" if (pa or sum(mv) == 1) else "NO EFFECT" if ne else None
moved = [k for k in old if old[k] and old[k] != new[k]]
print("as landed", dict(Counter(old.values())))
print("amended  ", dict(Counter(new.values())))
print("placed cells that move:", moved)
print("newly placed (c0, c1, m0, m1):", [(k, new[k]) for k in old if old[k] is None])
print("PARTIAL as landed with R-MOVE at one seed only:", [k for k in old if old[k] == "PARTIAL" and sum(k[2:]) == 1])
raise SystemExit(1 if moved or None in new.values() else 0)
