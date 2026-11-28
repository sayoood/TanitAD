"""make_mutants.py -- three one-line mutants of dscale10k_check_rhk.py, each removing ONE RHK pin. Each must turn at least one test_k_ test RED. usage: make_mutants.py SRC OUTDIR"""
import os
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)
M = {
    "mut_nodose.py": [('if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RHK_P]:', 'if False:'),            # the literal 0.1 pin AND the template-equality it rides with (the dose is pinned twice)
                      ('({"--dagger-p"} if kind == "RHp3" else set())', '({"--dagger-p"} if kind in ("RHp3", "RHK") else set())')],
    "mut_notoken.py": ('if tk not in (RHK_TOKENS if kind == "RHK" else RH_TOKENS):', 'if tk not in (RHK_TOKENS + RH_TOKENS if kind == "RHK" else RH_TOKENS):'),
    "mut_nopath.py": ('if not bk.endswith(f"/{RHK_BANK_DIR}/s{seed}"):', 'if False:'),
}
for name, subs in M.items():
    subs = subs if isinstance(subs, list) else [subs]
    t = s
    for old, new in subs:
        assert t.count(old) == 1, (name, t.count(old))
        t = t.replace(old, new)
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="").write(t)
    print("wrote", name)
