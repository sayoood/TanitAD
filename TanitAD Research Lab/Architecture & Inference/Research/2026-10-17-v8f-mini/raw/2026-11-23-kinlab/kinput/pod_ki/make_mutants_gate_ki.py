"""make_mutants_gate_ki.py -- One-line mutants of ki_gate_record.py, each removing ONE check; each must turn test_ki_gate_record.py RED (rc != 0).
The KI checks (card PREREG_V8F_KINPUT section 3, lever-acts): the argv base, the built levers' base and label, the probe's base_source / rows_changed.base / selected rows, the shared RK checks that still apply
(G-SUITE commit, bank sha, grad_diff, the p = 0 control), the lever flags KI must not carry, verify's probe re-derivation and the pre-lock precheck. There is no cache check in KI.
usage: python -I make_mutants_gate_ki.py SRC TEST SCRATCH_DIR   (runs the test on a control copy, which must be GREEN, then on each mutant)"""
import os
import subprocess
import sys

SRC, TEST, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
s = open(SRC, encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)

M = {
    "mut_noargvbase": ('if m.get("--dagger-base") != BASE:', "if False:"),
    "mut_nolevbase": ('if lb != {BASE}:', "if False:"),
    "mut_nolevlabelhuman": ('if ll != {"human"}:', "if False:"),
    "mut_nobasesource": ('if set(bs) != {BASE}:', "if False:"),
    "mut_norowschangedbase": ('if len(base_rows) != 1 or base_rows[0] != 0 or isinstance(base_rows[0], bool):', "if False:"),
    "mut_noselectedrows": ('if not ns or max(ns) <= 0:', "if False:"),
    "mut_noprobepass": ('if pr.get("pass") is not True:', "if False:"),
    "mut_nogradcheck": ('if not gd or max(gd) <= 0:', "if False:"),
    "mut_nop0control": ('if not z or max(z) != 0.0:', "if False:"),
    "mut_nogsuitecommit": ('if gs.get("new") != COMMIT and gs.get("new_commit") != COMMIT:', "if False:"),
    "mut_nobankshacheck": ('if m.get("--dagger-bank-sha256") != bind["bank_json"]["sha256"]:', "if False:"),
    "mut_nootherleverflags": ('        if f in m:\n', "        if False:\n"),
    "mut_nobankpath": ('if m.get("--dagger-bank") != f"{R}/bank/s{seed}":', "if False:"),
    "mut_nopaircheck": ('if "ZZ ROLLHIST-PAIR PASS" not in open(F["pair_log"]).read():', "if False:"),
    "mut_noverifyprobe": ('if pb or rec.get("probe") != vals:', "if False:"),
    "mut_noverifycommit": ('if rec.get("commit") != COMMIT or not rec.get("pass") or rec.get("base") != BASE or a.tag not in TAGS or rec.get("tag") != a.tag:', "if False:"),
    "mut_notagcheck": ('if tag not in TAGS or not re.fullmatch(r"[0-9a-f]{40}", COMMIT):', 'if not re.fullmatch(r"[0-9a-f]{40}", COMMIT):'),
    "mut_precheck_noinputs": ("            bad += input_problems(tag, ro, argv_map([str(t) for t in argv]), bind)\n    for q in bad:", "            pass\n    for q in bad:"),
    "mut_selfcheckunbound": ('"selfcheck": f"{d}/selfcheck/selfcheck.json", ', ""),
}


def run_test(path):
    env = dict(os.environ)
    p = subprocess.run([sys.executable, "-I", TEST, path, os.path.join(OUT, "t_" + os.path.basename(path))], env=env, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, [l for l in (p.stdout + p.stderr).splitlines() if "FAIL" in l or "Error" in l][:4]


ctl = os.path.join(OUT, "ki_gate_record_ctl.py")
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
print("ZZ KI-GATE-MUTANTS", "PASS" if allred else "FAIL")
sys.exit(0 if allred else 1)
