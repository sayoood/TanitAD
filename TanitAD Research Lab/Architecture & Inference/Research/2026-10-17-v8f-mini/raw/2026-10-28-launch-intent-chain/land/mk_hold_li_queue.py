"""Hold: the LAUNCH-INTENT queue stage (:launchintentarm plus four dispatch lines in v8f_mini_box2_queue.cmd; mods-rw on a base blob the
tip must still carry) and the box-2 chain package banked beside it (raw/2026-10-28-launch-intent-chain/: the shipped files, the
generators, the self-test and its proof, the dry run, LAUNCH.txt, NOTES.md, MANIFEST.txt).  python mk_hold_li_queue.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
D = "E:/Projects/TanitAD-artifacts/dagland/launch_intent_chain/"
H = "C:/Users/Admin/qland/hold_li_queue/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-launch-intent-chain/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

Q = V8 + "code/v8f_mini_box2_queue.cmd"
base = blob(Q)
assert base == rd(D + "queue_stage/queue_tip_b09ac89f.cmd"), "the tip's queue moved since the stage was derived"
new = rd(D + "queue_stage/v8f_mini_box2_queue.cmd")
assert sha(new) == "195db471625359f344f6e4a02406a46b07925205c06ac3da725854b1084e0d9e"
bl, nl = base.split(b"\r\n"), new.split(b"\r\n")
assert all(l in nl for l in bl if l), "a base line is missing from the staged queue"
assert new.count(b":launchintentarm") == 3 and base.count(b"launchintent") == 0
put("mods-rw/" + Q, new)

n = 0
SKIP_DIRS = ("__pycache__",)
for root, dirs, files in os.walk(D):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (root == D.rstrip("/") and d == "land")]
    for f in files:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(D):]
        if rel.startswith("land/"):
            continue
        b = rd(full)
        if not b:
            continue
        assert len(b) < 400_000, rel
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
put("stage/" + PK + "land/mk_hold_li_queue.py", rd(D + "land/mk_hold_li_queue.py"))
n += 1
put("msg.txt", ("LAUNCH-INTENT box-2 queue stage: :launchintentarm (DAG-OACC's banked argv by calling :dagoaccarm untouched, plus"
                " --launch-intent-w 1.58926 and nothing else; launchintentrep sets the trainer seed to 10 + SEED as :dagoaccorcarm does"
                " for REP) and four dispatch lines; every other line of the queue unchanged. Chain package banked in"
                " raw/2026-10-28-launch-intent-chain/ (%d files): the seven shipped files, the generators, the fail-closed self-test"
                " (390 / 390 tests, 185 / 185 mutants killed; guard regression arms: a positive control and 11 defective harnesses all"
                " REFUSED), the dry run (four runs SAME and ONE-LEVER against their pair controls, REP licence PASS at both seeds),"
                " LAUNCH.txt, NOTES.md, MANIFEST.txt.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode("utf-8"))
print("hold written", H, n)
