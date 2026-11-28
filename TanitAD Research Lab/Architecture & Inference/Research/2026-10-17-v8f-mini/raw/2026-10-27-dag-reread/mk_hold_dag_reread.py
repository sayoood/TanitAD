"""Hold: the DAGGER read's ROBUST outcome (DAG_READ.json + the re-read log, bytes as written) + PREREG_V8F_DAG_REREAD (the committed next's
re-reads, registered before any DAG - P10 / DAG - FLAT row) + the register rows.  python mk_hold_dag_reread.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
H = "C:/Users/Admin/qland/hold_dag_reread/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RUN = PK + "raw/2026-10-27-dagger-run/"
RR = PK + "raw/2026-10-27-dag-reread/"
READ_SHA = "3393fdd56bd561fd5181a8219d2c75d700036b9373da6ec12a8bcba51decde3d"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def rd(p):
    return open(p, "rb").read()


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def stage(rel, b):
    assert blob(rel) is None, f"{rel} exists on the tip"
    put("stage/" + rel, b)


# 1. the read, bytes as the reader wrote them (CRLF; the lander hashes with --no-filters, so the sha256 below is the landed blob's)
b = rd(A + "v8f-dagger-t1/read/DAG_READ.json")
assert hashlib.sha256(b).hexdigest() == READ_SHA
stage(RUN + "read/DAG_READ.json", b)
for f in ("reread.log", "reread_cmd.txt"):
    stage(RUN + "read/" + f, rd(A + "dagland/" + f))

# 2. the re-read PREREG (authored here; LF)
pre = rd(A + "dagland/PREREG_V8F_DAG_REREAD.md").replace(b"\r\n", b"\n")
assert b"\r" not in pre and pre.startswith(b"# PREREG_V8F_DAG_REREAD")
stage("Project Steering/PREREG_V8F_DAG_REREAD.md", pre)
stage(RR + "mk_hold_dag_reread.py", rd(A + "dagland/mk_hold_dag_reread.py").replace(b"\r\n", b"\n"))

# 3. the register: H-V8F-DAGGER-1 gets its outcome; H-V8F-DAG-REREAD-1 is a new row after it
g = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-DAGGER-1**")]
assert len(row) == 1 and "| **H-V8F-DAG-REREAD-1**" not in g
cells = row[0].split(" | ")
lead = "**PRE-REGISTERED (2026-10-26), NOT RUN.**"
assert cells[2].startswith(lead)
cells[2] = (
    "**ROBUST (MEASURED 2026-10-27, read by the pre-registered reader at Amendment 2; all 176 controls PASS).** Both heads clear their bar at"
    " both seeds, at T1-kin (ticks >= 1, DAG - DAG-0, paired episode-cluster bootstrap, B = 2,000, 141 episodes): R-TLAT +0.1082 [+0.0491,"
    " +0.1572] / +0.1184 [+0.0755, +0.1585] (bar 0.0248 / 0.0248); R-TLON +0.0554 [+0.0300, +0.0847] / +0.0806 [+0.0560, +0.1043] (bar"
    " 0.0502 / 0.0278, set by the replicate floor RF 0.0251 / 0.0139). Guards held: R-ADE (T1-kin ADE6) -0.5435 [-0.7564, -0.3690] /"
    " -0.4046 [-0.5496, -0.2782] m, separated-better; R-T0 +0.0019 [-0.0230, +0.0266] / +0.0483 [+0.0178, +0.0807] m, both <= F/2 = 0.0604."
    " Committed next, verbatim: *\"adopt DAG as the base. Re-read T1 dominance over P10 and FLAT's committed repeat on it. If the crawl"
    " beside rows (M1, L2) still fail, next is DAgger iteration 2 (DAG's own states), under its own read\"*; registered as"
    " `H-V8F-DAG-REREAD-1`. `raw/2026-10-27-dagger-run/read/DAG_READ.json` (sha256 `3393fdd5...`). Pre-registered 2026-10-26: "
    + cells[2][len(lead):].lstrip())
new = " | ".join([
    "| **H-V8F-DAG-REREAD-1** (2026-10-27)",
    "On the adopted DAG base, at T1-kin (m = 10, causal bucket, eval-141): (A) DAG DOMINATES the kinematic floor P10, i.e. at BOTH seeds ADE6"
    " is separated-better by >= 2F (0.2415 m), no row of the four families is separated-worse, and every family has a separated-better row;"
    " and (B) DAG beats its capacity-matched FLAT control on ADE6 by >= 2F, separated at both seeds (B-1 at FLAT's registered open-loop tier,"
    " B-2 at T1-kin; B-2 is confounded by DAG's DAgger fine-tune and is attributed only by FLAT-DAG)",
    "**PRE-REGISTERED (2026-10-27), NOT RUN.** Written before any DAG - P10 or DAG - FLAT row and before any FLAT T1-kin rollout; the S-POS"
    " vs CTRV rows were seen in the DAGGER read and are disclosed as seen (section 0). Zero training; dev-box CPU.",
    "HYPOTHESIS",
    "`Project Steering/PREREG_V8F_DAG_REREAD.md` |",
])
g2 = g.replace(row[0], " | ".join(cells) + "\n" + new)
assert g2.count("| **H-V8F-DAG-REREAD-1**") == 1
put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g2.encode("utf-8"))
put("msg.txt", ("PREREG_V8F_DAGGER read ROBUST (176/176 controls): DAG - DAG-0 at T1-kin clears both heads at both seeds (R-TLAT +0.108 / +0.118,"
                " R-TLON +0.055 / +0.081), T1-kin ADE6 -0.54 / -0.40 m, T0 within F/2. DAG_READ.json + the re-read log. PREREG_V8F_DAG_REREAD:"
                " the committed next's re-reads (dominance over P10 with a registered bar, FLAT's repeat at its tier and at T1-kin), written"
                " before any of their rows. Register: H-V8F-DAGGER-1 ROBUST, H-V8F-DAG-REREAD-1 new\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
