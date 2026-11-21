# -*- coding: utf-8 -*-
"""PROVE each re-anchored mutation is ARMED, by reintroducing the defect in the
REAL trainer and requiring the guards to go RED.

A green suite is not evidence that a mutation arm can fail -- that is the
disarmed-regression class this whole file set exists to prevent.  So for each
anchor: apply it to `stack/scripts/refc_v3_train.py` IN PLACE, run both test
files, record exactly which tests turn red, restore from a byte backup, and
md5-verify the trainer.

Two kinds of red are reported separately:
  BEHAVIOUR  a test asserting the SHIPPED trainer's behaviour now fails
             -- this is the evidence that the mutation is a real change.
  ANCHOR     a `_mutate` arm reports its anchor missing, because the in-place
             mutation consumed the very text that arm anchors on.  Expected
             bookkeeping; NOT behaviour evidence.
"""
import hashlib
import os
import re
import subprocess
import sys

STACK = r"D:\Projects\TanitAD\stack"
TRAINER = os.path.join(STACK, "scripts", "refc_v3_train.py")
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
TESTS = ["tests/test_refc_v3_agent_gt_head_drop.py",
         "tests/test_refc_v3_agent_gt_reaches_forward.py"]
OUT = os.path.dirname(os.path.abspath(__file__))

# The three anchors, verbatim from the test modules (canonical LF form).
GATE_FROM = ('    if (_ag_cfg is not None and getattr(_ag_cfg, "enable", False)\n'
             '            and getattr(_ag_cfg, "oracle", False)):')
GATE_TO = "    if True:"
REVERT_FROM = ("                v_max_ms=v_max_ms, v_max_valid=v_max_valid,\n"
               "                agent_gt=agent_gt)")
REVERT_TO = "                v_max_ms=v_max_ms, v_max_valid=v_max_valid)"
VALUE_FROM = '        agent_gt = {"box": batch["agent_box"].to(device),'
VALUE_TO = '        agent_gt = {"box": batch["agent_box"].to(device) * 2.0,'

MUTATIONS = [
    ("GATE   (oracle gate removed -> `if True:`)", GATE_FROM, GATE_TO),
    ("REVERT (agent_gt dropped from forward call)", REVERT_FROM, REVERT_TO),
    ("VALUE  (in-branch box corruption * 2.0)", VALUE_FROM, VALUE_TO),
]


def resolve(src, anchor):
    ab = anchor.encode("utf-8")
    forms = [ab] if b"\n" not in ab else [ab, ab.replace(b"\n", b"\r\n")]
    counts = [(f, src.count(f)) for f in forms]
    return next((f for f, n in counts if n), ab), sum(n for _f, n in counts)


def eol(src):
    return b"\r\n" if b"\r\n" in src else b"\n"


def run_tests():
    env = dict(os.environ)
    env.update(PYTHONPATH=STACK, CUDA_VISIBLE_DEVICES="-1",
               PYTHONIOENCODING="utf-8")
    p = subprocess.run([PY, "-m", "pytest", "-q", "--no-header", "-rf"] + TESTS,
                       cwd=STACK, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.stdout + p.stderr


def classify(log):
    """-> (behaviour_reds, anchor_reds, tail)"""
    failed = re.findall(r"^FAILED (\S+)", log, re.M)
    disarmed = set()
    for m in re.finditer(r"_{5,}\s+(\S+?)\s+_{5,}(.*?)(?=\n_{5,}|\n=+ )", log,
                         re.S):
        if "DISARMED" in m.group(2):
            disarmed.add(m.group(1).split("[")[0])
    beh, anc = [], []
    for f in failed:
        node = f.split("::")[-1].split("[")[0]
        (anc if node in disarmed else beh).append(f)
    tail = [l for l in log.splitlines() if re.match(r"^\d+ (failed|passed)", l)
            or " passed" in l and ("failed" in l or "error" in l)]
    return beh, anc, (tail[-1] if tail else log.strip().splitlines()[-1])


orig = open(TRAINER, "rb").read()
base_md5 = hashlib.md5(orig).hexdigest()
backup = os.path.join(OUT, "refc_v3_train.py.bak")
open(backup, "wb").write(orig)
print("baseline trainer md5 = %s  (%d bytes)  backup -> %s\n"
      % (base_md5, len(orig), backup))

report = []
for label, frm, to in MUTATIONS:
    ob, n = resolve(orig, frm)
    assert n == 1, "%s: anchor resolved %d times, not 1" % (label, n)
    nb = to.encode("utf-8").replace(b"\n", eol(orig))
    mutated = orig.replace(ob, nb)
    assert mutated != orig
    assert len(orig) - len(mutated) == len(ob) - len(nb)
    try:
        open(TRAINER, "wb").write(mutated)
        log = run_tests()
    finally:
        open(TRAINER, "wb").write(orig)
    now = hashlib.md5(open(TRAINER, "rb").read()).hexdigest()
    assert now == base_md5, "RESTORE FAILED: %s != %s" % (now, base_md5)

    beh, anc, tail = classify(log)
    with open(os.path.join(OUT, "red_%s.log" % label.split()[0]), "w",
              encoding="utf-8") as fh:
        fh.write(log)
    report.append((label, beh, anc, tail, now))
    print("=" * 78)
    print("MUTATION: %s" % label)
    print("  anchor resolved: 1 match, %d bytes (%s form)"
          % (len(ob), "CRLF" if b"\r\n" in ob else "LF"))
    print("  suite: %s" % tail)
    print("  BEHAVIOUR RED (%d) -- the mutation is a real change:" % len(beh))
    for f in beh:
        print("      %s" % f)
    print("  ANCHOR RED (%d) -- arm's own anchor consumed by the in-place edit:"
          % len(anc))
    for f in anc:
        print("      %s" % f)
    print("  trainer restored, md5 = %s  MATCHES BASELINE" % now)

print("=" * 78)
print("ALL MUTATIONS RESTORED. final md5 = %s"
      % hashlib.md5(open(TRAINER, "rb").read()).hexdigest())
print("expected                        = %s" % base_md5)
