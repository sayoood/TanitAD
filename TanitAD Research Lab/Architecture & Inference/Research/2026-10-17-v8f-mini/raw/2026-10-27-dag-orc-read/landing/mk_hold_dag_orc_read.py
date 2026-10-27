"""Hold: DAG-ORC's reader (dag_orc_read.py) and its frozen replicate floor RF0 (P0_RF0.json): PREREG_V8F_DAG_ORC launch
preconditions 1 and 2, built and proven, nothing read on DAG-ORC data (none exists).   python mk_hold_dag_orc_read.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/dag_orc_read/"
H = "C:/Users/Admin/qland/hold_dag_orc_read/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-dag-orc-read/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


n = 0
for root, dirs, files in os.walk(S):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", "_work")]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), S).replace("\\", "/")
        if rel == "PREREG_V8F_DAG_ORC.md":
            continue                                   # the landed PREREG is in the repo (the reader pins its blob sha)
        b = open(S + rel, "rb").read()
        if rel.endswith((".py", ".md", ".sh")):
            assert b"\r" not in b, rel
        assert len(b) > 0 and blob(PK + rel) is None and len(H + "stage/" + PK + rel) < 260, rel
        os.makedirs(os.path.dirname(H + "stage/" + PK + rel), exist_ok=True)
        open(H + "stage/" + PK + rel, "wb").write(b)
        n += 1
os.makedirs(H + "stage/" + PK + "landing", exist_ok=True)
open(H + "stage/" + PK + "landing/mk_hold_dag_orc_read.py", "wb").write(open(__file__, "rb").read().replace(b"\r\n", b"\n"))
open(H + "msg.txt", "wb").write((
    "DAG-ORC reader + frozen RF0 (PREREG_V8F_DAG_ORC launch preconditions 1 and 2; nothing read on DAG-ORC data, none exists)."
    " P0_RF0.json (sha256 1608d70b...): RF0 per Read-A row = max over seeds of |DAG-0 - ORC-0| from the banked causal dumps, Read A's reader"
    " unchanged, identity 0.0 and RC 0 differences. dag_orc_read.py (sha256 a4907f4e...): PREREG pinned by the 85d0b125/3b7b404e blob,"
    " every table cell verbatim; bar 2 x max(F, RF0, RF), R-MOVE / R-CRAWL / guards / R-KEEP / R-NOBASE, A then B on HARMS. Self-test 288/288,"
    " mutation 135/135 killed (control green), K-RC 0 differences vs DAG2_READ.json, dry run OK with A / REP / B missing. Not built: the T0"
    " four-family block, M-LAUNCH, the anti-echo block, M2/L2 on the GT-launch subset (work items before the read). Package"
    " raw/2026-10-27-dag-orc-read/ (%d files).\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % (n + 1)).encode())
print("hold written", H, n + 1)
