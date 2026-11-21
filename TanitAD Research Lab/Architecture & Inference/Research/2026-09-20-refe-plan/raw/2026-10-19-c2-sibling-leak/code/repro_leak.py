"""Reproduce the C2 sibling leak on a code dir: python -B -I repro_leak.py <code dir with srttc_c2_gate.py ...>"""
import os, shutil, sys, tempfile
code = os.path.abspath(sys.argv[1]); sys.path.insert(0, code)
import srttc_c2_gate as CG
tmp = tempfile.mkdtemp(prefix="c2leak_repro_", dir="C:/Users/Admin/par18/c2leak/work")
names = ("srttc_r1_gate.py", "srttc_r1.py", "srttc_c2.py")
def mk(n):
    d = os.path.join(tmp, n); os.makedirs(d)
    for f in names: shutil.copy(os.path.join(code, f), d)
    return d
A, B = mk("A"), mk("B")
path0 = list(sys.path); mods0 = set(sys.modules)
rA, cA, _ = CG.load_verified_pair(A, {})
print("A: R.__file__ in A:", os.path.dirname(rA.__file__) == A, "| sys.path grew:", len(sys.path) - len(path0), "| sys.path[0] =", sys.path[0])
shutil.rmtree(A)
rB, cB, _ = CG.load_verified_pair(B, {})
print("B (identical bytes, A deleted): same srttc_r1 object as A:", rB is rA, "| same c2 object:", cB is cA)
print("B: R.__file__ =", os.path.relpath(rB.__file__, tmp), "| exists:", os.path.exists(rB.__file__))
print("B: c2mod.R is rB:", cB.R is rB)
try:
    open(os.path.abspath(cB.R.__file__), "rb").read(1); print("B: reading R.__file__ (as run_c2 does at srttc_c2.py:110) OK")
except Exception as e:
    print("B: reading R.__file__ (as run_c2 does at srttc_c2.py:110) ->", type(e).__name__)
shutil.rmtree(tmp, ignore_errors=True)
