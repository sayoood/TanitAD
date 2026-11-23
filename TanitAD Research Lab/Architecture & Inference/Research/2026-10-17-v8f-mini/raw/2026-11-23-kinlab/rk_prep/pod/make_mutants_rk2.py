"""make_mutants_rk2.py -- REVISION 2. One-line mutants of dscale10k_check_rk.py (rev2), each removing ONE RK pin; each must turn test/test_rk_checker.py RED.
The five of revision 1 are kept; new: the cache pair not admitted, the cache pin removed, the cache pair kept in the pair check's argv.
usage: python -I make_mutants_rk2.py SRC TESTDIR OUTDIR   (control copy first: must be GREEN)"""
import os
import subprocess
import sys

SRC, TESTDIR, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)
M = {
    "mut_nolabelpin": ('if a.get("--dagger-label") != RK_LABEL:', 'if False:'),
    "mut_nodose": [('if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RK_P]:', 'if False:'),
                   ('({"--dagger-p"} if kind == "RHp3" else set())', '({"--dagger-p"} if kind in ("RHp3", "RK") else set())')],
    "mut_nobankpin": ('if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n                bad.append(f"RK --dagger-bank', 'if False:\n                bad.append(f"RK --dagger-bank'),
    "mut_labelnotadmitted": ('({"--dagger-label"} if kind in ("RL", "RK") else set())', '({"--dagger-label"} if kind == "RL" else set())'),
    "mut_labelkeptforpair": ('            if kind in ("RL", "RK"):\n                li = ', '            if kind == "RL":\n                li = '),
    # revision 2
    "mut_cachenotadmitted": ('({RK_CACHE_FLAG} if kind == "RK" else set())', 'set()'),
    "mut_nocachepin": ('if len(kc) != 1 or not kc[0].endswith(f"/kin_cache/s{seed}_kin_v1.npz") or "/kin_cache/s" not in kc[0]:', 'if False:'),
    "mut_cachekeptforpair": ('if ci >= 0 and kind == "RK":', 'if False:'),
}


def run_test(path):
    env = dict(os.environ, DS_CHECK_RK_PATH=path)
    p = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", os.path.join(OUT, "bt_" + os.path.basename(path)), "test_rk_checker.py"],
                       cwd=TESTDIR, env=env, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, [l for l in (p.stdout + p.stderr).splitlines() if l.startswith("FAILED") or " passed" in l or " failed" in l][-3:]


ctl = os.path.join(OUT, "ctl_dscale10k_check_rk.py")
open(ctl, "w", encoding="utf-8", newline="").write(s)
rc, info = run_test(ctl)
print(f"M0 control rc={rc} ({'GREEN as required' if rc == 0 else 'NOT GREEN'}) {info}")
allred = rc == 0
for name, subs in M.items():
    subs = subs if isinstance(subs, list) else [subs]
    t = s
    for old, new in subs:
        assert t.count(old) == 1, (name, t.count(old))
        t = t.replace(old, new)
    p = os.path.join(OUT, name + ".py")
    open(p, "w", encoding="utf-8", newline="").write(t)
    rc, info = run_test(p)
    print(f"{name} rc={rc} ({'RED as required' if rc != 0 else 'STILL GREEN'}) {info[-1:]}")
    allred &= rc != 0
print("ZZ RK-CHECKER-MUTANTS", "PASS" if allred else "FAIL")
sys.exit(0 if allred else 1)
