"""Discharge FV_SPEC_V5_POOL.md with Z3 (same judge and conventions as fv_check.py, fv_check_v2.py and fv_check_v4.py)."""
import hashlib, json, sys, time
from z3 import Reals, Solver, And, Not, sat, unsat
import z3
SPEC, OUT = sys.argv[1], sys.argv[2]
with open(SPEC, "rb") as fh:
    sha = hashlib.sha256(fh.read()).hexdigest()
res, mut = {}, {}


def run(h, c):
    s = Solver(); s.set("timeout", 120000); s.add(*h); s.add(Not(c)); t0 = time.time(); r = s.check()
    return r, round(time.time() - t0, 3), (str(s.model())[:300] if r == sat else None)


def prove(i, h, c):
    r, dt, m = run(h, c)
    res[i] = {"status": "PROVED" if r == unsat else ("COUNTEREXAMPLE" if r == sat else "UNKNOWN"), "s": dt, **({"model": m} if m else {})}


def refute(i, h, c):
    r, dt, m = run(h, c)
    mut[i] = {"status": "REFUTED" if r == sat else ("NOT-REFUTED" if r == unsat else "UNKNOWN"), "s": dt, **({"counterexample": m} if m else {})}


y1, y2, g = Reals("y1 y2 g")
u1, u2, u3, u4, s, w, b = Reals("u1 u2 u3 u4 s w b")
k1, k2, s1, s2 = Reals("k1 k2 s1 s2")
a = lambda x: w * x + b
mean = lambda xs: (a(xs[0]) + a(xs[1]) + a(xs[2]) + a(xs[3])) / 4
grid = lambda xs: [a(x) for x in xs]
block = lambda xs: [(a(xs[0]) + a(xs[1])) / 2, (a(xs[2]) + a(xs[3])) / 2]
A = [u1 + s, u2, u3, u4]
B = [u1, u2 + s, u3, u4]
C = [u1, u2, u3 + s, u4]
rd = lambda G: G[0] - G[1]

prove("VC-P1", [], (y1 - g) ** 2 + (y2 - g) ** 2 >= (y1 - y2) ** 2 / 2)
prove("VC-P2", [g == (y1 + y2) / 2], (y1 - g) ** 2 + (y2 - g) ** 2 == (y1 - y2) ** 2 / 2)
prove("VC-P3", [], mean(A) == mean(B))
prove("VC-P4", [], ((1 - g) ** 2 + g ** 2) / 2 >= z3.Q(1, 4))
prove("VC-P5", [], ((1 - g) ** 2 + 3 * g ** 2) / 4 >= z3.Q(3, 16))
prove("VC-P6", [w != 0, s != 0], And(rd(grid(A)) - rd(grid(B)) == 2 * w * s, rd(grid(A)) - rd(grid(B)) != 0))
# k objects of strength s on the same background shift the sum of the cell features by k*s
m_k = lambda k: (a(u1) + a(u2) + a(u3) + a(u4) + w * k * s) / 4
prove("VC-P7", [w != 0, s != 0, k1 != k2], And(m_k(k1) - m_k(k2) == (k1 - k2) * w * s / 4, m_k(k1) - m_k(k2) != 0))
prove("VC-P8", [], And(mean(B) - mean(A) == 0, grid(B)[0] - grid(A)[0] == -w * s, grid(B)[1] - grid(A)[1] == w * s))
prove("VC-P9", [w != 0, s != 0], And(block(A)[0] == block(B)[0], block(A)[1] == block(B)[1],
                                      block(A)[0] - block(C)[0] == w * s / 2, block(A)[0] - block(C)[0] != 0))

refute("M-P1", [], (y1 - g) ** 2 + (y2 - g) ** 2 >= (y1 - y2) ** 2)
refute("M-P4", [], ((1 - g) ** 2 + g ** 2) / 2 >= z3.Q(26, 100))
refute("M-P5", [], ((1 - g) ** 2 + 3 * g ** 2) / 4 >= z3.Q(1, 4))
refute("M-P6", [s != 0], rd(grid(A)) - rd(grid(B)) != 0)
refute("M-P7", [w != 0, s1 != 0, s2 != 0, k1 != k2], (k1 * s1 - k2 * s2) * w / 4 != 0)
refute("M-P8", [w != 0, s != 0], mean(B) - mean(A) != 0)
refute("M-P9", [w != 0, s != 0], block(A)[0] - block(B)[0] != 0)

bad = [i for i, r in res.items() if r["status"] != "PROVED"] + [i for i, r in mut.items() if r["status"] != "REFUTED"]
out = {"spec": "FV_SPEC_V5_POOL.md", "spec_sha256": sha, "z3": z3.get_version_string(), "n_vc": len(res),
       "n_proved": sum(r["status"] == "PROVED" for r in res.values()), "n_mut": len(mut),
       "n_refuted": sum(r["status"] == "REFUTED" for r in mut.values()), "not_ok": bad, "vc": res, "mutations": mut}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("ZZ FV5 spec=" + sha[:12], "proved=" + str(out["n_proved"]) + "/" + str(out["n_vc"]),
      "mutations_refuted=" + str(out["n_refuted"]) + "/" + str(out["n_mut"]), "not_ok=" + str(bad))
sys.exit(1 if bad else 0)
