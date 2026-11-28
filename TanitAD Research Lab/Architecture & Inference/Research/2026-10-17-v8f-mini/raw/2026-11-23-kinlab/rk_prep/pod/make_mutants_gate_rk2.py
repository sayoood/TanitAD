"""make_mutants_gate_rk2.py -- REVISION 2. One-line mutants of rk_gate_record.py (rev2), each removing ONE check; each must turn test_rk_gate_record.py RED (rc != 0).
Revision-1's six (census bar, label check, G-SUITE commit, bank sha, verify's census re-derivation, n_label_kin) are kept, plus the cache checks of Amendment 1 and the pre-lock precheck.
usage: python -I make_mutants_gate_rk2.py SRC TEST SCRATCH_DIR   (runs the test on a control copy, which must be GREEN, then on each mutant)"""
import os
import subprocess
import sys

SRC, TEST, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)


def blk(indent, cond):
    """the two-line anchor `<indent>if/elif <cond>:` + the next line's `bad.append`, with the newline in front so the indentation is part of the match"""
    pad = " " * indent
    return "\n" + pad + cond + ":\n" + pad + "    bad.append"


M = {
    "mut_nocensusbar": (blk(4, "elif share < CENSUS_BAR"), blk(4, "elif False")),
    "mut_nolabelcheck": ('if m.get("--dagger-label") != LABEL:', "if False:"),
    "mut_nogsuitecommit": ('if gs.get("new") != COMMIT and gs.get("new_commit") != COMMIT:', "if False:"),
    "mut_nobankshacheck": ('if m.get("--dagger-bank-sha256") != bind["bank_json"]["sha256"]:', "if False:"),
    "mut_noverifycensus": ("if share is None or share < CENSUS_BAR or c.get(\"relabelled_share_of_ok\") != share or c.get(\"n_ok\") != n_ok:", "if False:"),
    "mut_nonkin": ("if not nk or max(nk) <= 0:", "if False:"),
    # revision 2: the cache checks of the mint
    "mut_nocachepath": ('if m.get("--dagger-label-cache") != want_cache:', "if False:"),
    "mut_nocachesha": ("if set(lev_sha) != {csha}:", "if False:"),
    "mut_nocachelevpath": ("if set(lev_path) != {want_cache}:", "if False:"),
    "mut_nocpusha": (blk(4, "if set(cl) != {csha}"), blk(4, "if False")),
    "mut_nocpucensus": ("if counts_c != counts or share_c != share:", "if False:"),
    # the pre-lock precheck
    "mut_precheck_nosha": (blk(8, "if set(cl) != {csha}"), blk(8, "if False")),
    "mut_precheck_nocensus": (blk(8, "elif share < CENSUS_BAR"), blk(8, "elif False")),
    "mut_precheck_nopath": (blk(8, "if set(lp) != {p}"), blk(8, "if False")),
    "mut_precheck_nolabel": (blk(8, "if lv != {LABEL}"), blk(8, "if False")),
    # binding
    "mut_cachenotbound": ('f["label_cache"] = cache_path(seed)', 'pass'),
    "mut_cpuscnotbound": ('f["cache_selfcheck"] = f"{R}/kin_cache/s{seed}_selfcheck_cpu.json"', 'pass'),
    "mut_verifynocachebind": ('if not cp or cp != cache_path(a.tag.split("_s")[1]) or rec.get("label_cache_sha256") != (rec.get("binds", {}).get("label_cache") or {}).get("sha256"):', "if False:"),
}


def run_test(path):
    env = dict(os.environ)
    p = subprocess.run([sys.executable, "-I", TEST, path, os.path.join(OUT, "t_" + os.path.basename(path))], env=env, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, [l for l in (p.stdout + p.stderr).splitlines() if "FAIL" in l or "Error" in l][:4]


ctl = os.path.join(OUT, "rk_gate_record_ctl.py")
open(ctl, "w", encoding="utf-8", newline="").write(s)
rc, _ = run_test(ctl)
print(f"M0 control rc={rc} ({'GREEN as required' if rc == 0 else 'NOT GREEN: the control must pass'})")
allred = rc == 0
for name, (old, new) in M.items():
    assert s.count(old) == 1, (name, s.count(old))
    p = os.path.join(OUT, name + ".py")
    open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    rc, fails = run_test(p)
    print(f"{name} test_rc={rc} ({'RED as required' if rc != 0 else 'STILL GREEN: the test missed it'}): {len(fails)} fail lines {fails[:2]}")
    allred &= rc != 0
print("ZZ RK-GATE-MUTANTS", "PASS" if allred else "FAIL")
sys.exit(0 if allred else 1)
