"""gen_argv_s1.py -- DSCALE10K block 2 (pod): writes argv/{B_s1,U_s1}.A.json and argv/{B_s1,U_s1}.B.tmpl.json from the block-1 files of B_s0 / U_s0 (card 2.3: seed 1; stage B --seed 1 and --dagger-seed 1).
Only --seed, --dagger-seed and --out change (the checker's argv_diff_problems pins --seed / --dagger-seed to the arm's own value and refuses any other difference). Idempotent; prints the arms' argv."""
import json
import os
import sys

D = os.environ.get("DS_ROOT", "/root/dscale10k")
L = lambda n: json.load(open(f"{D}/argv/{n}.json"))["argv"]


def setf(t, f, vals):
    t = list(t)
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + list(vals) + t[j:]


def W(n, t):
    json.dump({"argv": t}, open(f"{D}/argv/{n}.json", "w"), indent=1)


for arm, src in (("B_s1", "B_s0"), ("U_s1", "U_s0")):
    a = setf(setf(L(f"{src}.A"), "--seed", ["1"]), "--out", [f"{D}/runs/{arm}/A"])
    b = setf(setf(L(f"{src}.B.tmpl"), "--seed", ["1"]), "--dagger-seed", ["1"])
    b = setf(b, "--out", [f"{D}/runs/{arm}/B"])
    W(f"{arm}.A", a)
    W(f"{arm}.B.tmpl", b)
    for n, t in ((f"{arm}.A", a), (f"{arm}.B.tmpl", b)):
        print(n, " ".join(t))
        print()
