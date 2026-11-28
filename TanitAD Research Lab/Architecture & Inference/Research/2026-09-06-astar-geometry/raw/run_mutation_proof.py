"""Two-sided mutation proof for D-REFCV4B-ASTAR-GEOMETRY.

Builds a MUTANT copy of the banked gate in which `_pair()` re-introduces the
defect (the hoisted `decoder.anchors` binding) as if the tool had never been
fixed, and runs it. The repo's own copy is never touched, so the suite is never
left red.

REQUIRED OUTCOME
  CORRECTED  -> the banked gate passes 5/5.
  MUTATED    -> the v0-conditioned clauses go RED (the gate CAN fail), while
                the FIXED-vocabulary clause stays GREEN (the defect is
                invisible on refcv3 -- which is why it survived review, and
                why correcting it cannot move refcv3's numbers).

A gate that cannot fail measures nothing; a gate that fails on both arms would
not be measuring THIS defect.
"""
import io
import os
import subprocess
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "taniteval", "tests",
                   "test_refcv3_arm_astar_geometry.py")
MUT = os.path.join(SP, "test_MUTANT_astar_geometry.py")
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"

s = io.open(SRC, encoding="utf-8").read()

# make the mutant self-locating (it lives outside the repo tree)
s = s.replace(
    'TOOL = Path(__file__).resolve().parents[1] / "tools" / "refcv3_arm.py"',
    'TOOL = Path(r"%s") / "taniteval" / "tools" / "refcv3_arm.py"' % REPO)

# THE MUTATION: `right` is computed from the hoisted reference-speed family,
# i.e. exactly the binding the fix removed.
OLD = "        right.append(tool._oracle_anchor_index(good, tgt, sv))"
NEW = ("        right.append(tool._oracle_anchor_index(bad, tgt, sv))"
       "   # MUTANT: the defect")
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)
io.open(MUT, "w", encoding="utf-8", newline="").write(s)

env = dict(os.environ)
env["OMP_NUM_THREADS"] = "6"
env["PYTHONPATH"] = os.pathsep.join(
    [os.path.join(REPO, "stack"), os.path.join(REPO, "stack", "scripts"),
     REPO])


def run(path, label):
    p = subprocess.run([PY, "-m", "pytest", path, "-q", "--no-header",
                        "-p", "no:cacheprovider"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-14:]
    print("=" * 68)
    print(label, " exit=", p.returncode)
    print("=" * 68)
    for l in tail:
        print("   ", l)
    return p.returncode, p.stdout or ""


rc_ok, out_ok = run(SRC, "CORRECTED (the banked gate)")
rc_mut, out_mut = run(MUT, "MUTATED (defect re-introduced)")

print()
print("#" * 68)
print("MUTATION PROOF VERDICT")
print("#" * 68)
ok = True
if rc_ok != 0:
    print("FAIL: the corrected gate did not pass.")
    ok = False
else:
    print("PASS: corrected gate green (5/5).")
if rc_mut == 0:
    print("FAIL: the MUTANT passed -- the gate cannot detect the defect and "
          "therefore measures nothing.")
    ok = False
else:
    print("PASS: mutant RED -- the gate detects the re-introduced defect.")

# third clause: the fixed-vocabulary (refcv3) control must survive the mutation
inv = "test_fixed_vocabulary_mutation_is_invisible"
mut_failed_fixed = (inv in out_mut and "FAILED" in out_mut
                    and any(inv in l and "FAILED" in l
                            for l in out_mut.splitlines()))
if mut_failed_fixed:
    print("FAIL: the mutant also broke the FIXED-vocabulary control, so the "
          "gate is not specific to v0-conditioned builds.")
    ok = False
else:
    print("PASS: the FIXED-vocabulary (refcv3) control stayed GREEN under the "
          "mutation -- the defect is invisible there, as claimed.")
print()
print("OVERALL:", "MUTATION PROOF COMPLETE" if ok else "MUTATION PROOF FAILED")
sys.exit(0 if ok else 1)
