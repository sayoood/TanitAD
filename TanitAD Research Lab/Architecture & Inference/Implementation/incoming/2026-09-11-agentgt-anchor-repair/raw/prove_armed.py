# -*- coding: utf-8 -*-
"""THE COMPLEMENTARY PROOF: each ARM's green depends on the substitution.

`prove_red.py` shows the three mutations are real behaviour changes.  It cannot
show that the mutation ARMS are armed, because applying a mutation in place
consumes the very anchor those arms mutate on.

So: neuter `_mutate` itself -- make the substitution a NO-OP (`nb = ob`), so the
"mutant" is byte-identical to the shipped trainer -- and require every arm to go
RED.  A test that stays GREEN under a no-op mutation is passing for a reason
other than the mutation, which is exactly the disarmed-regression class.

Controls are EXPECTED to stay green: they assert the shipped trainer's normal
behaviour, and a no-op mutant is the shipped trainer.
"""
import hashlib
import os
import re
import subprocess

STACK = r"D:\Projects\TanitAD\stack"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
TESTS = ["tests/test_refc_v3_agent_gt_head_drop.py",
         "tests/test_refc_v3_agent_gt_reaches_forward.py"]
FILES = [os.path.join(STACK, t.replace("/", os.sep)) for t in TESTS]
OUT = os.path.dirname(os.path.abspath(__file__))

LIVE = b'    nb = new.encode("utf-8").replace(b"\\n", _eol(src))'
DISARM = b'    nb = ob  # DISARM SIMULATION: substitution is a no-op'

#: the arms -- tests whose green must DEPEND on the substitution biting.
ARMS = {
    "test_a_head_build_REFUSES_a_supplied_agent_gt",
    "test_the_message_names_the_two_ways_out",
    "test_DELIBERATE_REGRESSION_the_mutation_anchor_is_armed",
    "test_case_1_still_refuses_agent_gt_with_no_seam",
    "test_the_HEAD_row_is_now_DISCRIMINATING",
    "test_REVERTING_THE_PATCH_REPRODUCES_THE_DEFECT",
    "test_the_OFF_parity_comparison_CAN_FAIL",
    "test_the_ORACLE_value_assertion_CAN_FAIL",
}


def run():
    env = dict(os.environ)
    env.update(PYTHONPATH=STACK, CUDA_VISIBLE_DEVICES="-1",
               PYTHONIOENCODING="utf-8")
    p = subprocess.run([PY, "-m", "pytest", "-q", "--no-header", "-rf"] + TESTS,
                       cwd=STACK, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.stdout + p.stderr


orig = {f: open(f, "rb").read() for f in FILES}
base = {f: hashlib.md5(b).hexdigest() for f, b in orig.items()}
for f, h in base.items():
    print("baseline %-48s %s" % (os.path.basename(f), h))

try:
    for f in FILES:
        assert orig[f].count(LIVE) == 1, "%s: live line not found once" % f
        open(f, "wb").write(orig[f].replace(LIVE, DISARM))
    log = run()
finally:
    for f in FILES:
        open(f, "wb").write(orig[f])

now = {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in FILES}
assert now == base, "RESTORE FAILED"

open(os.path.join(OUT, "disarm_sim.log"), "w", encoding="utf-8").write(log)
failed = {x.split("::")[-1].split("[")[0] for x in
          re.findall(r"^FAILED (\S+)", log, re.M)}
tail = [l for l in log.splitlines() if re.match(r"^\d+ (failed|passed)", l)]

print("\nsuite under DISARM SIMULATION: %s" % (tail[-1] if tail else "?"))
print("\nARMS -- must be RED (green here = passing without the mutation):")
bad = []
for a in sorted(ARMS):
    ok = a in failed
    print("   %-58s %s" % (a, "RED  ok" if ok else "GREEN  *** DISARMED ***"))
    if not ok:
        bad.append(a)
print("\nnon-arm tests that also went red (controls are expected green):")
for x in sorted(failed - ARMS):
    print("   %s" % x)
print("\ntest files restored:")
for f in FILES:
    print("   %-48s %s  MATCHES" % (os.path.basename(f), now[f]))
print("\nVERDICT: %s" % ("ALL %d ARMS ARE ARMED" % len(ARMS) if not bad
                         else "DISARMED ARMS: %s" % bad))
