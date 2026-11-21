"""Discharge FV_SPEC_V2.md with Z3 (same judge and conventions as fv_check.py)."""
import hashlib, json, sys, time
from z3 import Reals, Solver, And, Not, sat, unsat, RealVal
import z3
SPEC, OUT = sys.argv[1], sys.argv[2]
sha = hashlib.sha256(open(SPEC, "rb").read()).hexdigest(); res, mut = {}, {}
def run(h, c):
    s = Solver(); s.set("timeout", 120000); s.add(*h); s.add(Not(c)); t = time.time(); r = s.check()
    return r, round(time.time() - t, 3), (str(s.model())[:300] if r == sat else None)
def prove(i, h, c):
    r, dt, m = run(h, c); res[i] = {"status": "PROVED" if r == unsat else ("COUNTEREXAMPLE" if r == sat else "UNKNOWN"), "s": dt, **({"model": m} if m else {})}
def refute(i, h, c):
    r, dt, m = run(h, c); mut[i] = {"status": "REFUTED" if r == sat else ("NOT-REFUTED" if r == unsat else "UNKNOWN"), "s": dt, **({"counterexample": m} if m else {})}
a, b, y, K, S1, S2 = Reals("a b y K S1 S2")
sq = lambda t: t*t
prove("VC-G1", [], sq((a+b)/2 - y) == sq(a-y)/2 + sq(b-y)/2 - sq(a-b)/4)
refute("M-G1", [], sq((a+b)/2 - y) == sq(a-y)/2 + sq(b-y)/2 - sq(a-b)/2)
prove("VC-G2", [K > 0], sq(S1/K - y) == (S2 - 2*y*S1 + K*y*y)/K - (S2 - S1*S1/K)/K)
prove("VC-G3", [K > 0, S1*S1 <= K*S2], sq(S1/K - y) <= (S2 - 2*y*S1 + K*y*y)/K)
refute("M-G3", [K > 0], sq(S1/K - y) <= (S2 - 2*y*S1 + K*y*y)/K)
D2, C, G2, s2, rho, P, k, s, f, h, D, w, c = Reals("D2 C G2 s2 rho P k s f h D w c")
prove("VC-H1", [D2 > 0, C == G2], And(D2 - 2*C + G2 == D2 - G2, 1 - (D2 - G2)/D2 == G2/D2))
H2 = [s2 > 0, rho > -1, rho < 1]
prove("VC-H2", H2, 1 - (s2*(1 - rho*rho))/(2*s2*(1 - rho)) == (1 - rho)/2)
refute("M-H2", H2, 1 - (s2*(1 - rho*rho))/(2*s2*(1 - rho)) == (1 - rho))
prove("VC-H3", [P >= 1, k > 0, k <= P, s > 0], And(k*k*s/P <= k*s, (k*k*s/P)/(k*s) == k/P))
refute("M-H3", [P >= 1, k > 0, s > 0], And(k*k*s/P <= k*s, (k*k*s/P)/(k*s) == k/P))
H4 = [y > 0, f > 0, h > 0, D == f*h/y]
prove("VC-H4", H4, f*h/y - f*h/(y+1) == D*D/(f*h + D))
refute("M-H4", H4, f*h/y - f*h/(y+1) == D*D/(f*h))
prove("VC-H5", [f > 0, h > 0, w > 0, D > 0], f*h/D - f*h/(D+w) == f*h*w/(D*(D+w)))
prove("VC-H6", [y > 0, y <= c, f > 0, h > 0, D == f*h/y], D >= f*h/c)
refute("M-H6", [y > 0, c > 0, f > 0, h > 0, D == f*h/y], D >= f*h/c)
bad = [i for i, r in res.items() if r["status"] != "PROVED"] + [i for i, r in mut.items() if r["status"] != "REFUTED"]
json.dump({"spec": "FV_SPEC_V2.md", "spec_sha256": sha, "z3": z3.get_version_string(), "n_vc": len(res), "n_proved": sum(r["status"] == "PROVED" for r in res.values()), "n_mut": len(mut),
           "n_refuted": sum(r["status"] == "REFUTED" for r in mut.values()), "not_ok": bad, "vc": res, "mutations": mut}, open(OUT, "w", encoding="utf-8"), indent=1)
print(f"ZZ FV2 spec={sha[:12]} proved={sum(r['status']=='PROVED' for r in res.values())}/{len(res)} mutations_refuted={sum(r['status']=='REFUTED' for r in mut.values())}/{len(mut)} not_ok={bad}")
sys.exit(1 if bad else 0)
