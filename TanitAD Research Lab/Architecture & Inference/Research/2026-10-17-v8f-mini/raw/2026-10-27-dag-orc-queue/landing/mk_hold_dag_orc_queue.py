"""Hold: DAG-ORC's box-2 queue stages (insertions only), the 6 label-set tests widened by their labels, and the build / proof package
raw/2026-10-27-dag-orc-queue/.  Nothing is launched (needs the PI's go).   python mk_hold_dag_orc_queue.py <live tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/dag_orc_chain/"
H = "C:/Users/Admin/qland/hold_dag_orc_queue/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-dag-orc-queue/"
assert not os.path.exists(H), H


def blob(p, at=None):
    r = subprocess.run(GIT + ["show", f"{at or TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


QP = V8 + "code/v8f_mini_box2_queue.cmd"
old, new = blob(QP), open(S + "stage/" + QP, "rb").read()
assert old == blob(QP, "3b7b404e"), "the tip's queue moved since the build's base"
ol, nl = old.split(b"\r\n"), new.split(b"\r\n")
assert b"\n" not in b"".join(nl) and len(nl) == len(ol) + 20
it = iter(nl)
assert all(any(x == y for y in it) for x in ol), "an existing queue line is not kept in order"
put("mods-rw/" + QP, new)
for t in ("test_v8f_orc_chain", "test_v8f_flat", "test_v8f_oacc", "test_v8f_kind", "test_v8f_mixo", "test_v8f_lora_ar2"):
    T = f"stack/tests/{t}.py"
    assert blob(T) == blob(T, "3b7b404e"), T + ": the tip moved since the build's base"
    nb = open(S + "stage/" + T, "rb").read()
    assert b"\r" not in nb
    put("mods-rw/" + T, nb)
last = {}
for l in open(S + "pytest_sweep.log", encoding="utf-8", errors="replace"):
    m = re.search(r"##### (\S+) rc=(\d+)", l)
    if m:
        last[m.group(1)] = m.group(2)
assert len(last) == 18 and all(v == "0" for v in last.values()), last
n = 0
for root, dirs, files in os.walk(S):
    dirs[:] = [d for d in dirs if d not in ("stage", "tip", "__pycache__")]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), S).replace("\\", "/")
        b = open(S + rel, "rb").read()
        if rel.endswith((".py", ".md", ".sh", ".in")):
            b = b.replace(b"\r\n", b"\n")
        if not b:
            continue
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b); n += 1
put("stage/" + PK + "landing/mk_hold_dag_orc_queue.py", open(__file__, "rb").read().replace(b"\r\n", b"\n")); n += 1
put("msg.txt", ("DAG-ORC box-2 queue (PREREG_V8F_DAG_ORC): 20 queue lines inserted, 0 removed, every one of the 245 pre-existing stage"
                " invocations byte-identical in output; arms A / REP / B through the new :dagorcarm (init DAG_s; A = DAG2-0's argv + the"
                " perturbation flags, REP = DAG2-0's argv at --seed 10 / 11, B = A at p 0.1); the 6 label-set tests widened by the new labels"
                " only. Proof: selftest 150/150 + 230/230 regressions caught (the real chain on a fake box 2), mutation 83/83 killed, pytest 18"
                " files green on the composed tree (1,519 passed), flag-to-loss traced for all 30 flags, no launch blocker. NOT installed, NOT"
                " launched: needs the PI's go for box-2 time. Package raw/2026-10-27-dag-orc-queue/ (%d files).\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
