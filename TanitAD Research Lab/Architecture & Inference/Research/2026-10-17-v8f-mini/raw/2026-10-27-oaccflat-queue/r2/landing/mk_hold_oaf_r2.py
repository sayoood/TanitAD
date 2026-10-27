"""Hold: DAG-OACC r2 (the gate asserter's read_o branch, keyed on levers.residual_zero_init; the re-run chain that moves the first attempt's
box-2 gate dirs aside) into raw/2026-10-27-oaccflat-queue/r2/.  .py/.md are landed LF (MANIFEST_r2.txt hashes the E: bytes, CRLF in two files).
python mk_hold_oaf_r2.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/oaccflatq/r2/"
H = "C:/Users/Admin/qland/hold_oaf_r2/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-oaccflat-queue/r2/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


n = 0
src = [(os.path.relpath(os.path.join(r, f), S).replace("\\", "/"), os.path.join(r, f)) for r, _, fs in os.walk(S) for f in fs]
src.append(("landing/mk_hold_oaf_r2.py", __file__))
for rel, p in src:
    b = open(p, "rb").read()
    if rel.endswith((".py", ".md", ".sh")):
        b = b.replace(b"\r\n", b"\n")
        assert b"\r" not in b, rel
    assert len(b) > 0 and blob(PK + rel) is None and len(H + "stage/" + PK + rel) < 260, rel
    os.makedirs(os.path.dirname(H + "stage/" + PK + rel), exist_ok=True)
    open(H + "stage/" + PK + rel, "wb").write(b)
    n += 1
open(H + "msg.txt", "wb").write((
    "DAG-OACC r2 (PREREG_V8F_DAG_OACC, instrument fix, nothing trained): the first box-2 attempt passed C-O3 (w_acc 49.6969 exact) and the"
    " trainer's gate, then the dev-box asserter refused read_o_grad_diff 864.79 as 'must be exactly 0.0'. That value is not a registered"
    " control: it read 0.0 in OACC's gate only because OACC started from scratch with O's residual head zero-initialised"
    " (levers.residual_zero_init True); DAG-OACC fine-tunes the trained M2b, where the term must reach read_o. r2 keys the check on"
    " residual_zero_init (True: exactly 0.0; False: finite > 0; else refused); the four registered controls stay exactly 0.0. Proof: selftest"
    " 63/63 + 104/104 regressions caught; mutation 45/45 killed; the real gate: r1 refuses, r2 passes. The r2 chain moves the first"
    " attempt's box-2 gate dirs and pins aside (rename, never delete) after BOX2_BUSY. Queued on the dev box behind FLAT-DAG. Package"
    " raw/2026-10-27-oaccflat-queue/r2/ (%d files).\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
