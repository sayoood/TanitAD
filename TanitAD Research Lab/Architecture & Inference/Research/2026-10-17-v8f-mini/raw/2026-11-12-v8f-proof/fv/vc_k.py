import json, sys
from z3 import Reals, Solver, Not, unsat, sat
import z3
r = {}
s1, s2, v, cap, dt = Reals("s1 s2 v cap dt")
S = Solver(); S.add(dt > 0, v >= 0, v <= cap, s2 - s1 == v * dt, Not(s2 - s1 <= cap * dt)); r["VC-K1"] = "PROVED" if S.check() == unsat else "NOT-PROVED"
ax, ay, bx, by, cx, cy, x, y = Reals("ax ay bx by cx cy x y")
S = Solver(); S.set("timeout", 300000)
S.add(x >= 0, y >= 0, (ax - bx) ** 2 + (ay - by) ** 2 <= x * x, (bx - cx) ** 2 + (by - cy) ** 2 <= y * y, Not((ax - cx) ** 2 + (ay - cy) ** 2 <= (x + y) ** 2))
c = S.check(); r["VC-K2"] = "PROVED" if c == unsat else ("COUNTEREXAMPLE" if c == sat else "UNKNOWN")
S = Solver(); S.add(x >= 0, y >= 0, (ax - bx) ** 2 + (ay - by) ** 2 <= x * x, (bx - cx) ** 2 + (by - cy) ** 2 <= y * y, Not((ax - cx) ** 2 + (ay - cy) ** 2 <= x * x + y * y))
r["M-K2"] = "REFUTED" if S.check() == sat else "NOT-REFUTED"
r["z3"] = z3.get_version_string()
print(json.dumps(r))
