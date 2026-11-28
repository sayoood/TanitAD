"""make_mutants_rl.py -- one-line mutants of dscale10k_check_rl.py, each removing ONE RL pin; each must turn test_rl_checker.py RED. usage: make_mutants_rl.py SRC OUTDIR"""
import os, sys
SRC, OUT = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)
M = {
    "mut_nolabelpin.py": ('if a.get("--dagger-label") != RL_LABEL:', 'if False:'),
    "mut_nodose.py": [('if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RL_P]:', 'if False:'),            # the literal pin AND the template-equality it rides with (the dose is pinned twice)
                      ('({"--dagger-p"} if kind == "RHp3" else set())', '({"--dagger-p"} if kind in ("RHp3", "RL") else set())')],
    "mut_nobankpin.py": ('if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:', 'if False:'),
    "mut_labelnotadmitted.py": ('set(RH_DECL) | ({"--dagger-label"} if kind == "RL" else set())', 'set(RH_DECL)'),
    "mut_labelkeptforpair.py": ('            if kind == "RL":\n                li = ', '            if False:\n                li = '),
}
for name, subs in M.items():
    subs = subs if isinstance(subs, list) else [subs]
    t = s
    for old, new in subs:
        assert t.count(old) == 1, (name, t.count(old))
        t = t.replace(old, new)
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="").write(t)
    print("wrote", name)
