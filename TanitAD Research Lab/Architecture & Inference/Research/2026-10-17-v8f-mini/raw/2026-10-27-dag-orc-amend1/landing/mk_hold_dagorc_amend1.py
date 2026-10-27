"""Hold: PREREG_V8F_DAG_ORC Amendment 1 (A1-2026-10-27, re-based on DAG-OACC under launch precondition 4, before any DAG-ORC run) + the
DAG-OACC read it rests on (raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json, its canonical package path) + a register note on H-V8F-DAG-ORC-1.
python mk_hold_dagorc_amend1.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
HERE = "E:/Projects/TanitAD-artifacts/dagland/dagorc_amend1/"
H = "C:/Users/Admin/qland/hold_dagorc_amend1/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
am = rd(HERE + "AMEND1.md")
assert b"\r" not in am and am.startswith(b"\n## Amendment 1 (A1-2026-10-27): re-based on DAG-OACC")
PR = "Project Steering/PREREG_V8F_DAG_ORC.md"
p = blob(PR)
assert p.endswith(b"\n") and b"## Amendment 1" not in p
put("mods/" + PR, p + am)
RJ = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/read/DAGOACC_READ.json"
rj = rd(RJ)
assert hashlib.sha256(rj).hexdigest() == "9e6bd4f6940fb3e868d4d8aede50cf47c082a3833f014c34bb87ff201c99a894"
for rel, b in ((V8 + "raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json", rj),
               (V8 + "raw/2026-10-27-dag-orc-amend1/AMEND1.md", am),
               (V8 + "raw/2026-10-27-dag-orc-amend1/landing/mk_hold_dagorc_amend1.py", rd(HERE + "mk_hold_dagorc_amend1.py")),
               (V8 + "raw/2026-10-27-dag-orc-amend1/dagorc_waiter.out", rd("C:/Users/Admin/box2_ship/logs/dagorc_waiter.out"))):
    assert blob(rel) is None, rel
    put("stage/" + rel, b)

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
END = b" is running. | MEASURED (N0); HYPOTHESIS (A) | `Project Steering/PREREG_V8F_DAG_ORC.md` |"
assert g.count(END) == 1
add = (" is running. **Amendment 1 (A1-2026-10-27), before any DAG-ORC run (launch precondition 4):** DAG-OACC read LONGITUDINAL FIXED"
       " (`raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json`), so the arm is re-based in the COMPOSE form: A = DAG-OACC's argv + the"
       " perturbation + a new default-off `--ego-perturb-with-dagger` (perturb only non-swapped samples), CTRL = DAG-OACC banked, REP ="
       " DAG-OACC's argv at seeds 10 / 11; crawl set C = yaw-rate, curvature, L2, M1; no exempt rows; ≈ 2.8 h for A + REP, B not queued."
       " The DAG-based waiter was stopped before it fired. Launch blocker: the composition flag and its reachability proofs. |"
       " MEASURED (N0); HYPOTHESIS (A) | `Project Steering/PREREG_V8F_DAG_ORC.md` |").encode("utf-8")
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in add.lower(), w
g2 = g.replace(END, add)
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)
put("msg.txt", ("PREREG_V8F_DAG_ORC Amendment 1 (A1-2026-10-27): re-based on DAG-OACC under its launch precondition 4, before any DAG-ORC run"
                " (the DAG-based waiter was stopped before it fired; nothing reached box 2). DAG-OACC read LONGITUDINAL FIXED (60/60 controls;"
                " the read JSON is banked at raw/2026-10-27-dag-oacc/read/). COMPOSE form: A = DAG-OACC's argv + the perturbation + a new"
                " default-off composition flag, CTRL = DAG-OACC (banked), REP new; crawl set = yaw-rate, curvature, L2, M1. Register note.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
