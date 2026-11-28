"""make_mutants_chain_ki.py -- one-line mutants of rh_ki_chain.sh (the pre-lock validations), each must turn test_chain_dry_ki.py RED. Each mutant is a COPY of the kit whose SHA256_KI.txt is regenerated (so the `kit` step does not mask the
mutation); the control (unmutated copy) must be GREEN.
usage: python -I make_mutants_chain_ki.py <kit dir> <scratch dir> <bash exe> <python exe>"""
import hashlib
import os
import shutil
import subprocess
import sys

KIT, OUT, BASH, PYX = (os.path.abspath(a) for a in sys.argv[1:5])
TEST = os.path.join(KIT, "test_chain_dry_ki.py")
FILES = [l.split("*", 1)[1].strip() for l in open(os.path.join(KIT, "SHA256_KI.txt"), encoding="utf-8")]
chain = open(os.path.join(KIT, "rh_ki_chain.sh"), encoding="utf-8").read()
M = {
    "mut_nokitcheck": ("  vstep kit chk_kit\n  vstep tree_prep tp", "  vstep tree_prep tp"),
    "mut_notreecheck": ("  vstep tree_prep tp\n  vstep gsuite gs\n  vstep argv chk_argv\n  vstep gate_inputs chk_gate_inputs\n  say \"would now", "  vstep gsuite gs\n  vstep argv chk_argv\n  vstep gate_inputs chk_gate_inputs\n  say \"would now"),
    "mut_nogsuitestep": ("  vstep gsuite gs\n  vstep argv chk_argv", "  vstep argv chk_argv"),
    "mut_noargvstep": ("  vstep argv chk_argv\n  vstep gate_inputs", "  vstep gate_inputs"),
    "mut_nogateinputs": ("  vstep gate_inputs chk_gate_inputs\n  say \"would now", "  say \"would now"),
    "mut_gsalwaysvalid": ("ok = (r.get(\"new_commit\") == sys.argv[2] or r.get(\"new\") == sys.argv[2]) and not r.get(\"n_regressions\") and not r.get(\"new_test_failures\") and not r.get(\"missing_in_new\")", "ok = True"),
    "mut_treealwaysok": ("  chk_tree || { echo \"tree_ki fails the blob check", "  true || { echo \"tree_ki fails the blob check"),
    "mut_gsnolog_ignored": ("  gs_log_done || { echo \"$GSUITE_KI is valid but", "  true || { echo \"$GSUITE_KI is valid but"),
}


def stage(name, text):
    d = os.path.join(OUT, name, "kit")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(KIT):
        p = os.path.join(KIT, f)
        if os.path.isfile(p):
            shutil.copy(p, d)
    open(os.path.join(d, "rh_ki_chain.sh"), "w", encoding="utf-8", newline="\n").write(text)
    with open(os.path.join(d, "SHA256_KI.txt"), "w", encoding="utf-8", newline="\n") as f:
        for fn in FILES:
            f.write(hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest() + " *" + fn + "\n")
    return d


def run(d, name):
    p = subprocess.run([sys.executable, "-I", os.path.join(d, "test_chain_dry_ki.py"), d, os.path.join(OUT, name, "scr"), BASH, PYX], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, [l for l in (p.stdout + p.stderr).splitlines() if "FAIL" in l or "Error" in l][:3]


if os.path.exists(OUT):
    shutil.rmtree(OUT)
d = stage("ctl", chain)
rc, _ = run(d, "ctl")
print(f"M0 control rc={rc} ({'GREEN as required' if rc == 0 else 'NOT GREEN'})", flush=True)
allred = rc == 0
for name, (old, new) in M.items():
    assert chain.count(old) == 1, (name, chain.count(old))
    d = stage(name, chain.replace(old, new))
    rc, fails = run(d, name)
    print(f"{name} test_rc={rc} ({'RED as required' if rc != 0 else 'STILL GREEN: the test missed it'}): {fails[:2]}", flush=True)
    allred &= rc != 0
print("ZZ KI-CHAIN-MUTANTS", "PASS" if allred else "FAIL")
sys.exit(0 if allred else 1)
