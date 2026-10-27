"""Hold: the box-2 queue's DAGGER2 stages (insertions only: bankdag2; arms dag2 / dag20) + the build's proof package (self-test, mutation
proof, reachability, chain sources), bytes as the build wrote them.  python mk_hold_dag2_queue.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
Q = "E:/Projects/TanitAD-artifacts/dag2q/"
H = "C:/Users/Admin/qland/hold_dag2_queue/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
QF = PK + "code/v8f_mini_box2_queue.cmd"
RAW = PK + "raw/2026-10-27-dagger2-queue/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


# 1. the queue: the tip's file with 60 inserted lines and none removed (the build's base is 8e121ea8; the file is unchanged since)
tipq = blob(QF)
assert hashlib.sha1(b"blob %d\0" % len(tipq) + tipq).hexdigest() == "0c2f234510cfe4f1a2fd616a8caed973644cac6a"
newq = open(Q + "stage/" + QF, "rb").read()
assert hashlib.sha256(newq).hexdigest() == "1e038f8bca8d016d5799ee2888d4c15eac8684a200085e4480ca45153e4015c0"
a, b = tipq.split(b"\r\n"), newq.split(b"\r\n")
i = j = 0
while i < len(a):          # every tip line survives, in order (insertions only)
    while a[i] != b[j]:
        j += 1
    i += 1
    j += 1
assert len(b) - len(a) == 60 and newq.count(b"\r\n") == newq.count(b"\n")
put("mods-rw/" + QF, newq)

# 2. the proof package, bytes as written (MANIFEST.txt carries their sha256)
skip = ("stage/", "tip/")
n = 0
for root, _, files in os.walk(Q):
    for f in files:
        p = os.path.join(root, f).replace("\\", "/")
        rel = p[len(Q):]
        if rel.startswith(skip) or "__pycache__" in rel:
            continue
        assert blob(RAW + rel) is None
        put("stage/" + RAW + rel, open(p, "rb").read())
        n += 1
put("stage/" + RAW + "mk_hold_dag2_queue.py", open("E:/Projects/TanitAD-artifacts/dagland/mk_hold_dag2_queue.py", "rb").read().replace(b"\r\n", b"\n"))
put("msg.txt", ("Box-2 queue: PREREG_V8F_DAGGER2's stages, insertions only (60 lines, 0 removed): bankdag2 (generator = DAG_s, iteration-2 bank"
                " dir) and arms dag2 (--dagger-p fixed 0.1, init DAG_s) / dag20 (--dagger-p 0). Every existing stage prints byte-identical"
                " lines (47 dispatched stages swept). Proof package raw/2026-10-27-dagger2-queue/: self-test 113/113 with 93 deliberate"
                " regressions caught, mutation proof 44/44 killed, reachability 26/26 on the real DAG checkpoints and 24/24 through the real"
                " --selfcheck-only gate; chain sources (dev-box side). Bytes as the build wrote them (%d files)\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n, "package files")
