"""make_mutants_ki.py -- One-line mutants of dscale10k_check_ki.py, each removing ONE KI pin; each must turn test/test_ki_checker.py RED.
usage: python -I make_mutants_ki.py SRC TESTDIR OUTDIR PYTHON   (control copy first: must be GREEN; PYTHON = an interpreter that has pytest)"""
import os
import subprocess
import sys

SRC, TESTDIR, OUT, PYX = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]), sys.argv[4]
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)
M = {
    "mut_nobasepin": ('if a.get(KI_BASE_FLAG) != KI_BASE:', 'if False:'),
    "mut_nodose": [('if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [KI_P]:', 'if False:'),
                   ('({"--dagger-p"} if kind == "RHp3" else set())', '({"--dagger-p"} if kind in ("RHp3", "KI") else set())')],
    "mut_nobankpin": ('if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n                bad.append(f"KI --dagger-bank', 'if False:\n                bad.append(f"KI --dagger-bank'),
    "mut_basenotadmitted": ('({KI_BASE_FLAG} if kind == "KI" else set())', 'set()'),
    "mut_basekeptforpair": ('            if kind == "KI":                                                                      # the base pair', '            if False:                                                                             # the base pair'),
    "mut_kinotrhlike": ('RH_LIKE = ("RH", "RHp3", "RHK", "RL", "RA", "RK", "KI")', 'RH_LIKE = ("RH", "RHp3", "RHK", "RL", "RA", "RK")'),
}


def run_test(path):
    env = dict(os.environ, DS_CHECK_KI_PATH=path)
    env.pop("MSYS_NO_PATHCONV", None)
    p = subprocess.run([PYX, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", os.path.join(OUT, "bt_" + os.path.basename(path)), "test_ki_checker.py"],
                       cwd=TESTDIR, env=env, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, [l for l in (p.stdout + p.stderr).splitlines() if l.startswith("FAILED") or " passed" in l or " failed" in l or "rror" in l][-3:]


ctl = os.path.join(OUT, "ctl_dscale10k_check_ki.py")
open(ctl, "w", encoding="utf-8", newline="").write(s)
rc, info = run_test(ctl)
print(f"M0 control rc={rc} ({'GREEN as required' if rc == 0 else 'NOT GREEN'}) {info}")
allred = rc == 0
for name, subs in M.items():
    subs = subs if isinstance(subs, list) else [subs]
    t = s
    for old, new in subs:
        assert t.count(old) == 1, (name, old[:60], t.count(old))
        t = t.replace(old, new)
    p = os.path.join(OUT, name + ".py")
    open(p, "w", encoding="utf-8", newline="").write(t)
    rc, info = run_test(p)
    print(f"{name} test_rc={rc} ({'RED as required' if rc != 0 else 'STILL GREEN: the test missed it'}) {info}")
    allred &= rc != 0
print("ZZ KI-CHECKER-MUTANTS", "PASS" if allred else "FAIL")
sys.exit(0 if allred else 1)
