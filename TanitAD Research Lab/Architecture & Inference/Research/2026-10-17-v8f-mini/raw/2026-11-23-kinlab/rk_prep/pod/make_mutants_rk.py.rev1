"""make_mutants_rk.py -- one-line mutants of dscale10k_check_rk.py, each removing ONE RK pin; each must turn test_rk_checker.py RED. usage: make_mutants_rk.py SRC OUTDIR"""
import os, sys
SRC, OUT = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)
M = {
    "mut_nolabelpin.py": ('if a.get("--dagger-label") != RK_LABEL:', 'if False:'),
    "mut_nodose.py": [('if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RK_P]:', 'if False:'),
                      ('({"--dagger-p"} if kind == "RHp3" else set())', '({"--dagger-p"} if kind in ("RHp3", "RK") else set())')],
    "mut_nobankpin.py": ('if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n                bad.append(f"RK --dagger-bank', 'if False:\n                bad.append(f"RK --dagger-bank'),
    "mut_labelnotadmitted.py": ('({"--dagger-label"} if kind in ("RL", "RK") else set())', '({"--dagger-label"} if kind == "RL" else set())'),
    "mut_labelkeptforpair.py": ('            if kind in ("RL", "RK"):\n                li = ', '            if kind == "RL":\n                li = '),
}
for name, subs in M.items():
    subs = subs if isinstance(subs, list) else [subs]
    t = s
    for old, new in subs:
        assert t.count(old) == 1, (name, t.count(old))
        t = t.replace(old, new)
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="").write(t)
    print("wrote", name)
