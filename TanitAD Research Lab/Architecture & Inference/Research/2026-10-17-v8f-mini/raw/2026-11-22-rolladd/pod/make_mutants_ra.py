"""make_mutants_ra.py -- one-line mutants of dscale10k_check_ra.py, each removing ONE RA pin; each must turn test_ra_checker.py RED. usage: make_mutants_ra.py SRC OUTDIR"""
import os, sys
SRC, OUT = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)
M = {
    "mut_nomodepin.py": ('if a.get("--dagger-mode") != RA_MODE:', 'if False:'),
    "mut_nodose.py": [('if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RA_P]:', 'if False:'),
                      ('({"--dagger-p"} if kind == "RHp3" else set())', '({"--dagger-p"} if kind in ("RHp3", "RA") else set())')],
    "mut_nobankpin.py": ('if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n                bad.append(f"RA --dagger-bank', 'if False:\n                bad.append(f"RA --dagger-bank'),
    "mut_modenotadmitted.py": ('| ({"--dagger-mode"} if kind == "RA" else set())', ''),
    "mut_modekeptforpair.py": ('            if kind == "RA":\n                mi = ', '            if False:\n                mi = '),
}
for name, subs in M.items():
    subs = subs if isinstance(subs, list) else [subs]
    t = s
    for old, new in subs:
        assert t.count(old) == 1, (name, t.count(old))
        t = t.replace(old, new)
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="").write(t)
    print("wrote", name)
