"""Discharge FV_SPEC_V4_GATE.md with Z3 (same judge and conventions as fv_check.py and fv_check_v2.py)."""
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


p, t, f, dc, db = Reals("p t f dc db")
dom = [p > 0, p < 1, t >= 0, t <= 1, f >= 0, f <= 1]
DF = p * t * dc + (1 - p) * f * db
DH = -p * (1 - t) * dc - (1 - p) * (1 - f) * db
iff = lambda a, b: And(z3.Implies(a, b), z3.Implies(b, a))
prove("VC-J1", dom, DF - DH == p * dc + (1 - p) * db)
prove("VC-J2", dom + [dc < 0, db > 0], iff(DF < 0, t * p * (-dc) > f * (1 - p) * db))
prove("VC-J3", dom + [dc < 0, db > 0], iff(DH < 0, (1 - t) * p * (-dc) < (1 - f) * (1 - p) * db))
prove("VC-J4", dom + [dc < 0, db > 0, t == 1, f == 0], And(DF < 0, DH < 0))
prove("VC-J5", dom + [t == f], Not(And(DF < 0, DH < 0)))
prove("VC-J6", dom + [dc < 0, db > 0, DF < 0, DH < 0], t > f)
prove("VC-J7", dom + [dc < 0, db < 0], DH >= 0)
prove("VC-J7b", dom + [dc < 0, db < 0, t == 1, f == 1], DH == 0)
prove("VC-J8", dom + [dc < 0, db > 0, t >= f], Not(And(DF > 0, DH > 0)))
refute("M-J2", dom + [dc < 0, db > 0], iff(DF < 0, t * (-dc) > f * db))
refute("M-J6", dom + [dc < 0, db > 0, t > f], And(DF < 0, DH < 0))
refute("M-J5", dom + [t == f], DF <= 0)
refute("M-J8", dom + [dc < 0, db > 0], Not(And(DF > 0, DH > 0)))
bad = [i for i, r in res.items() if r["status"] != "PROVED"] + [i for i, r in mut.items() if r["status"] != "REFUTED"]
out = {"spec": "FV_SPEC_V4_GATE.md", "spec_sha256": sha, "z3": z3.get_version_string(), "n_vc": len(res),
       "n_proved": sum(r["status"] == "PROVED" for r in res.values()), "n_mut": len(mut),
       "n_refuted": sum(r["status"] == "REFUTED" for r in mut.values()), "not_ok": bad, "vc": res, "mutations": mut}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("ZZ FV4 spec=" + sha[:12], "proved=" + str(out["n_proved"]) + "/" + str(out["n_vc"]),
      "mutations_refuted=" + str(out["n_refuted"]) + "/" + str(out["n_mut"]), "not_ok=" + str(bad))
sys.exit(1 if bad else 0)
