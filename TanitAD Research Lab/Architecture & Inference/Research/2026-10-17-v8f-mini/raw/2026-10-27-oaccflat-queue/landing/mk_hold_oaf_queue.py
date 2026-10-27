"""Hold: the DAG-OACC + FLAT-DAG box-2 queue stages (insertions only, 0 removed), the 6 label-set tests widened by their labels, and the
build/proof package raw/2026-10-27-oaccflat-queue/ (+ the per-file pytest sweep of the composed tree).  python mk_hold_oaf_queue.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/oaccflatq/"
DL = "E:/Projects/TanitAD-artifacts/dagland/"
PRE = "C:/Users/Admin/qland/work/ofx_pre_oaf/"
H = "C:/Users/Admin/qland/hold_oaf_queue/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-oaccflat-queue/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


sha = lambda b: hashlib.sha256(b).hexdigest()
QP = V8 + "code/v8f_mini_box2_queue.cmd"
old, new = blob(QP), open(S + "stage/" + QP, "rb").read()
assert sha(old).startswith("1e038f8bca8d016d") and sha(new).startswith("6d26a613c1bcd6c1"), "queue base / build moved"
ol, nl = old.split(b"\r\n"), new.split(b"\r\n")
assert b"\n" not in b"".join(nl) and len(nl) == len(ol) + 153
it = iter(nl)
assert all(any(x == y for y in it) for x in ol), "an existing queue line is not kept in order"   # insertions only
put("mods-rw/" + QP, new)
for t in ("test_v8f_orc_chain", "test_v8f_flat", "test_v8f_oacc", "test_v8f_kind", "test_v8f_mixo", "test_v8f_lora_ar2"):
    T = f"stack/tests/{t}.py"
    tb, nb = blob(T), open(S + "stage/" + T, "rb").read()
    assert tb == open(PRE + t + ".py", "rb").read(), T + ": the tip moved since the sweep's base"
    assert b"\r" not in nb
    put("mods-rw/" + T, nb)
n = 0
for root, dirs, files in os.walk(S):
    dirs[:] = [d for d in dirs if d not in ("stage", "tip", "__pycache__")]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), S).replace("\\", "/")
        b = open(S + rel, "rb").read()
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b.replace(b"\r\n", b"\n") if rel.endswith(".py") else b); n += 1
for f in ("sweep_oaf.sh", "pytest_sweep_oaf.log", "sweep_oaf_rerun.sh", "pytest_sweep_oaf_rerun.log", "sweep_oaf_rerun2.sh", "pytest_sweep_oaf_rerun2.log", "oaf_queue/mk_hold_oaf_queue.py"):
    put("stage/" + PK + "landing/" + os.path.basename(f), open(DL + f, "rb").read().replace(b"\r\n", b"\n")); n += 1
last = {}                                                     # each file's LAST result over the sweep then its rerun (3 files the disk watchdog stopped)
for lg in ("pytest_sweep_oaf.log", "pytest_sweep_oaf_rerun.log", "pytest_sweep_oaf_rerun2.log"):
    t = open(DL + lg, "rb").read().decode()
    assert "SWEEP-DONE" in t, lg
    for l in t.splitlines():
        if l.startswith("stack/tests/") and " rc=" in l:
            last[l.split(" rc=")[0]] = l.split(" rc=")[1].split()[0]
assert len(last) == 19 and all(v == "0" for v in last.values()), ("sweep not green", last)
put("msg.txt", ("DAG-OACC + FLAT-DAG box-2 queue (PREREG_V8F_DAG_OACC, PREREG_V8F_FLATDAG): 153 queue lines inserted, 0 removed (queue sha256"
                " 6d26a613..., base 1e038f8b...), every pre-existing stage byte-identical in output (selftest S); the 6 label-set tests widened by"
                " the new labels only. Proof: selftest 135/135, mutation 85/85 killed, control green; both chains dry-run rc 0. Pytest on the"
                " composed tree (tip stack/taniteval blob-verified + this queue + these tests), one run per file, all 19 files (the 15 queue"
                " readers + init-ckpt / DAgger tests) green: landing/pytest_sweep_oaf.log, with the 3 files its C: disk watchdog stopped"
                " (orc_chain, dagger_train, ego_perturb; killed, not failed) re-run: orc_chain green in landing/pytest_sweep_oaf_rerun.log (C: ran to 0 MB there and stopped the other two),"
                " dagger_train and ego_perturb green in landing/pytest_sweep_oaf_rerun2.log (temp on F:). Not run against a real box 2: the ssh glue,"
                " waccdag's model route (C-O3 compares it to the offline 49.6969 at launch). Package raw/2026-10-27-oaccflat-queue/ (%d files).\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
