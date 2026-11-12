"""Discharge FV_SPEC_V1.md with Z3. A VC is PROVED when its negation is unsat; a mutation is REFUTED when the solver finds a counterexample."""
import hashlib, json, sys, time
from z3 import (Real, Reals, Bool, Int, IntVal, RealVal, Function, RealSort, Solver, And, Or, Not, Implies, If, sat, unsat, simplify)

SPEC = sys.argv[1]; OUT = sys.argv[2]
spec_sha = hashlib.sha256(open(SPEC, "rb").read()).hexdigest()
res, mut = {}, {}

def run(hyp, claim, timeout_ms=120000):
    s = Solver(); s.set("timeout", timeout_ms)
    s.add(*hyp); s.add(Not(claim))
    t = time.time(); r = s.check(); dt = round(time.time() - t, 3)
    return r, dt, (s.model() if r == sat else None)

def prove(vid, hyp, claim):
    r, dt, m = run(hyp, claim)
    res[vid] = {"status": "PROVED" if r == unsat else ("COUNTEREXAMPLE" if r == sat else "UNKNOWN"), "s": dt}
    if m is not None: res[vid]["model"] = str(m)[:300]

def refute(mid, hyp, claim):
    r, dt, m = run(hyp, claim)
    mut[mid] = {"status": "REFUTED" if r == sat else ("NOT-REFUTED" if r == unsat else "UNKNOWN"), "s": dt}
    if m is not None: mut[mid]["counterexample"] = str(m)[:300]

iff = lambda a, b: And(Implies(a, b), Implies(b, a))
ab = lambda x: If(x >= 0, x, -x)
mx = lambda a, b: If(a >= b, a, b)
mn = lambda a, b: If(a <= b, a, b)

# ---- Group A
L, e, e2, b, eps, P, Q, R, epsk, m, k = Reals("L e e2 b eps P Q R epsk m k")
prove("VC-A1", [L >= 0, e <= b, e2 <= L*e + eps], e2 <= L*b + eps)
refute("M-A1", [e <= b, e2 <= L*e + eps], e2 <= L*b + eps)
prove("VC-A2", [L != 1, b == eps*(P-1)/(L-1)], And(L*b + eps == eps*(P*L-1)/(L-1), Implies(P == 1, b == 0)))
A3h = [L > 1, Q > 1, R > 1, eps > 0, epsk >= 0]
prove("VC-A3", A3h, iff(epsk < eps*(Q-1)/(L-1), epsk*(R-1)/(Q-1) < eps*(R-1)/(L-1)))
refute("M-A3", A3h, iff(epsk <= eps*(Q-1)/(L-1), epsk*(R-1)/(Q-1) < eps*(R-1)/(L-1)))
prove("VC-A4", [m > 0, k > 0, eps > 0, epsk >= 0], iff(epsk < k*eps, epsk*m < eps*(m*k)))
x, y, Sx, Sy = Reals("x y Sx Sy")
prove("VC-A5", [0 <= x, x <= y, 0 <= Sx, Sx <= Sy], x*Sx + 1 <= y*Sy + 1)

# ---- Group B
S0, S1, S2, c = Reals("S0 S1 S2 c")
prove("VC-B1", [S0 > 0], S2 - 2*c*S1 + c*c*S0 == (S2 - S1*S1/S0) + S0*(S1/S0 - c)*(S1/S0 - c))
prove("VC-B2", [S0 > 0, c != S1/S0], S2 - 2*c*S1 + c*c*S0 > S2 - S1*S1/S0)
refute("M-B2", [S0 > 0], S2 - 2*c*S1 + c*c*S0 > S2 - S1*S1/S0)
p, D, r, tol, da, db, Da, Db, ya, yb = Reals("p D r tol da db Da Db ya yb")
B3 = lambda sep: [p > 0, p <= RealVal(1)/2, D > 0, r >= 0, tol >= 0, sep, da <= r, db <= r, Da == p*D, Db == (1-p)*D, ya >= Da - da, yb >= Db - db]
prove("VC-B3", B3(p*D > tol + r), And(ya > tol, yb > tol))
refute("M-B3", B3(p*D > tol), And(ya > tol, yb > tol))
prove("VC-B4", [da <= r, r <= tol], da <= tol)

# ---- Group C
d, v, u, q, dt = Reals("d v u q dt")
Bf = Function("B", RealSort(), RealSort())
vb = mx(v - q, 0); uc = mx(u, vb); d2 = d - v*dt
def shield(vnext, checkd):
    hyp = [dt > 0, q > 0, v >= 0, Bf(0) == 0, Bf(v) >= 0, Bf(vb) >= 0, Bf(uc) >= 0, Implies(v > 0, Bf(v) == v*dt + Bf(vb)), Bf(v) <= d]
    return hyp, And(Bf(vnext) <= d2, d2 >= 0, vnext >= vb, vnext >= 0)
vn = If(Bf(uc) <= d2, uc, vb)
prove("VC-C1", *shield(vn, d2))
refute("M-C1a", *shield(uc, d2))
refute("M-C1b", *shield(If(Bf(uc) <= d, uc, vb), d))
n = Real("n")
Bcf = lambda vv, nn: dt*((nn+1)*vv - q*nn*(nn+1)/2)
prove("VC-C2", [n >= 1, n*q <= v], And(Bcf(v, n) == v*dt + Bcf(v - q, n - 1), Bcf(v, 0) == v*dt))

def masked(K, use_mask=True):
    s = [Real(f"s{i}") for i in range(K)]; a = [Bool(f"a{i}") for i in range(K)]
    sel = lambda idx: (lambda f: f(f, 0))(lambda f, i: s[i] if i == K-1 else If(idx == i, s[i], f(f, i+1)))
    best = IntVal(-1); arg = IntVal(-1)
    for i in range(K):
        ok = a[i] if use_mask else True
        best = If(And(ok, Or(best == -1, s[i] > sel(best))), IntVal(i), best)
        arg = If(Or(arg == -1, s[i] > sel(arg)), IntVal(i), arg)
    adm = lambda idx: Or(*[And(idx == i, a[i]) for i in range(K)])
    p1 = Implies(Or(*a), adm(best))
    p2 = And(*[Implies(And(a[i], best != -1), s[i] <= sel(best)) for i in range(K)])
    p3 = Implies(adm(arg), best == arg)
    return And(p1, p2, p3)
for K in range(1, 7):
    prove(f"VC-C3[K={K}]", [], masked(K))
refute("M-C3", [], masked(3, use_mask=False))
J, J0, Jinf, V, lam = Reals("J J0 Jinf V lam")
prove("VC-C4", [lam > 0, V >= 0, J + lam*V <= J0, J >= Jinf], V <= (J0 - Jinf)/lam)
refute("M-C4", [lam > 0, V >= 0, J + lam*V <= J0], V <= (J0 - Jinf)/lam)
z = Real("z"); w = mn(v, u)
prove("VC-C5", [z <= u], And(w <= u, ab(v - w) <= ab(v - z)))

# ---- Group D
G, J1, I1, I0, J2, I2 = Reals("G J1 I1 I0 J2 I2")
D1 = lambda gain: [G >= 0, ab(J1 - I1) <= G, ab(J0 - I0) <= G, gain]
prove("VC-D1", D1(I1 - I0 > 2*G), J1 > J0)
refute("M-D1", D1(I1 - I0 > G), J1 > J0)
prove("VC-D2", [ab(J1 - I1) <= G, ab(J2 - I2) <= G, I1 <= I2], J1 <= J2 + 2*G)

# ---- Group E
a_, b_ = Reals("a_ b_")
prove("VC-E1", [x > 0, y > 0], a_*a_/x + b_*b_/y >= (a_+b_)*(a_+b_)/(x+y))
refute("M-E1", [x > 0], a_*a_/x + b_*b_/y >= (a_+b_)*(a_+b_)/(x+y))
prove("VC-E2", [x > 0, y > 0, a_*y == b_*x], a_*a_/x + b_*b_/y == (a_+b_)*(a_+b_)/(x+y))
pb, pi, pib = Reals("pb pi pib")
E3h = [pb > 0, pb < 1, pi > 0, pi < 1, pib > 0, pib < 1]
odds = (pb/(1-pb)) * (pi/(1-pi)) / (pib/(1-pib)); post = odds/(1+odds)
rr = (pib*(1-pi))/((1-pib)*pi); tau = rr/(1+rr)
prove("VC-E3", E3h, iff(post >= RealVal(1)/2, pb >= tau))
refute("M-E3", E3h, iff(post >= RealVal(1)/2, pb >= RealVal(1)/2))

# ---- Group F
k1, k2, m11, m12, m21, m22, l1, l2 = Reals("k1 k2 m11 m12 m21 m22 l1 l2")
fro = m11*m11 + m12*m12 + m21*m21 + m22*m22; kk = k1*k1 + k2*k2
con = [l1 == m11*k1 + m12*k2, l2 == m21*k1 + m22*k2]
prove("VC-F1", [Or(k1 != 0, k2 != 0)] + con, fro*kk >= l1*l1 + l2*l2)
refute("M-F1", [Or(k1 != 0, k2 != 0)], fro*kk >= l1*l1 + l2*l2)
e11, e12, e21, e22 = l1*k1/kk, l1*k2/kk, l2*k1/kk, l2*k2/kk
prove("VC-F2", [Or(k1 != 0, k2 != 0)], And(e11*k1 + e12*k2 == l1, e21*k1 + e22*k2 == l2, (e11*e11 + e12*e12 + e21*e21 + e22*e22)*kk == l1*l1 + l2*l2))
al, w0, w1, xx = Reals("al w0 w1 xx")
prove("VC-F3", [], (1-al)*w0*xx + al*w1*xx == ((1-al)*w0 + al*w1)*xx)

import z3
bad = [k_ for k_, r_ in res.items() if r_["status"] != "PROVED"] + [k_ for k_, r_ in mut.items() if r_["status"] != "REFUTED"]
out = {"spec": SPEC.replace("\\", "/").split("/")[-1], "spec_sha256": spec_sha, "z3": z3.get_version_string(), "n_vc": len(res), "n_proved": sum(r_["status"] == "PROVED" for r_ in res.values()),
       "n_mut": len(mut), "n_refuted": sum(r_["status"] == "REFUTED" for r_ in mut.values()), "not_ok": bad, "vc": res, "mutations": mut}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
print(f"ZZ FV spec={spec_sha[:12]} proved={out['n_proved']}/{out['n_vc']} mutations_refuted={out['n_refuted']}/{out['n_mut']} not_ok={bad}")
sys.exit(1 if bad else 0)
