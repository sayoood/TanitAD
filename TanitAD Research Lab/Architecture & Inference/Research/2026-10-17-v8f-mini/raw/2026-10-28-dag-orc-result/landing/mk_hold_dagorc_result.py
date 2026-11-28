"""Hold: RESULT_V8F_DAG_ORC (HARMS, arm A) + the package raw/2026-10-28-dag-orc-result/ (the read JSONs, the T1-kin and box-2 logs, the four run records,
the K-BUILD record, R2C_NOTE, the audit) + register row H-V8F-DAG-ORC-1 (mods-rw, line count unchanged).
python mk_hold_dagorc_result.py <live tip> [--out <hold dir>]
The default hold dir is C:/Users/Admin/qland/hold_dagorc_result/. The Master Mind lands it; this script only writes the hold."""
import hashlib
import os
import re
import subprocess
import sys

args = sys.argv[1:]
TIP = args[0]
HOLD = "C:/Users/Admin/qland/hold_dagorc_result/"
if "--out" in args:
    HOLD = args[args.index("--out") + 1].replace("\\", "/").rstrip("/") + "/"
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
HERE = A + "dagland/dagorc_result/"
RD = "F:/TanitAD-scratch/dagorc_t1_out/read/"
T1L = A + "dagorc_t1/logs/"
RUNS = A + "v8f-dag-orc-oacc/"
LG = "C:/Users/Admin/box2_ship/logs/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-dag-orc-result/"
H = HOLD
assert not os.path.exists(H), H

FORBID = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
HFTOK = re.compile(rb"hf_[A-Za-z0-9]{8,}")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not HFTOK.search(b), "hf_ token in " + rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- the RESULT
res = rd(HERE + "RESULT_V8F_DAG_ORC.md")
assert b"\r" not in res and res.startswith(b"# RESULT_V8F_DAG_ORC: ") and b"Outcome **HARMS**" in res
assert blob(V8 + "RESULT_V8F_DAG_ORC.md") is None, "RESULT already at the tip"
for w in FORBID:
    assert w not in res.lower(), w
put("stage/" + V8 + "RESULT_V8F_DAG_ORC.md", res)

# ---- the package
files = {}   # repo-relative (under PK) -> (source path, expected sha256 or None)
READ_PIN = {"DAGORC_READ.json": "70a7a3a46e520311ade0cf2e1c71203010a25185ef5eb82b9d34b988b9548112",
            "RF_REP.json": "64c9444d486939be8dbbf61ef67028c35aa56b13c5d4bdbbc71d6e17e6fc497a"}
for f, h in READ_PIN.items():
    files["read/" + f] = (RD + f, h)
files["logs/chain.out"] = (T1L + "chain.out", "db5fa5a813818082d61956b937fdd42b17ffbeddd076df5acdd3a06161200a8d")
files["logs/dagorc_read.out"] = (T1L + "dagorc_read.out", "5a8679b7db7076acd8df874a437deddb6d2aedafbfb8442a16f666e170c5563b")
for f in ("chain_v8f_dagoaccorc_r2.out", "chain_v8f_dagoaccorc.out", "chain_v8f_dagoaccorc_r2.out_r2a_marker_stop",
          "chain_v8f_dagoaccorc_r2.out_r2b_queue_edit_stop", "chain_v8f_dagoaccorc_r2.out_r2c_diskfloor_stop",
          "v8fdagoaccorc10280348_t_M2b_s1_dagoaccorc.log"):
    files["box2/logs/" + f] = (LG + f, None)
files["box2/run_chain_v8f_dagoaccorc_r2c_pin.cmd"] = ("C:/Users/Admin/box2_ship/run_chain_v8f_dagoaccorc_r2c_pin.cmd", None)
files["notes/R2C_NOTE.md"] = (A + "dagland/dagorc_r2a/R2C_NOTE.md", "df02ff2f8b46ab05f45d437bd32d6a69596ea49b07d968f731c8b72cc9a98fac")
files["kbuild/kbuild_r2_launch_8c24823f.json"] = (A + "dagorc_t1/kbuild_r2_launch_8c24823f.json", "b1c5c36b6c3cb483f7d735546ac5f104fedcc6e8cf96df499da06a6f5fba1d65")
CK = {}
for n in ("M2b_s0_dagoaccorc", "M2b_s0_dagoaccorcrep", "M2b_s1_dagoaccorc", "M2b_s1_dagoaccorcrep"):
    for f in ("summary.json", "selfcheck.json", "argv.json", "launch_gate_token.json", "SHA256SUMS.txt"):
        files[f"runs/{n}/{f}"] = (RUNS + f"{n}/{f}", None)
    sums = dict((l.split(" *")[1], l.split(" *")[0]) for l in rd(RUNS + f"{n}/SHA256SUMS.txt").decode().splitlines())
    CK[n] = sums["ckpt_last.pt"]
    for f, h in sums.items():
        if f in ("summary.json", "selfcheck.json", "argv.json", "launch_gate_token.json"):
            assert sha(rd(RUNS + f"{n}/{f}")) == h, (n, f)      # the staged run record is the one the chain pulled
assert CK["M2b_s0_dagoaccorc"].startswith("380b230d") and CK["M2b_s0_dagoaccorcrep"].startswith("b03fd6ef")
assert CK["M2b_s1_dagoaccorc"].startswith("870dded7") and CK["M2b_s1_dagoaccorcrep"].startswith("77853e56")
for f in ("NUMBERS_AUDIT.md", "gen_result.py", "audit_cover.py", "REGISTER_ROW.md"):
    files["audit/" + f] = (HERE + f, None)
files["landing/mk_hold_dagorc_result.py"] = (HERE + "mk_hold_dagorc_result.py", None)
# the originals the RESULT cites that are already in the repo: identical bytes, not re-staged
for repo_rel, src, h in (("raw/2026-10-28-dag-orc-build/rebase/read/P0_RF0.json", A + "dagorc_rebase/read/P0_RF0.json", "1608d70b21b6b6a45136a6d96828f70b5bbabd9c0ceb84db136927593c79ad0f"),
                         ("raw/2026-10-28-dag-orc-build/rebase/read/dag_orc_read.py", A + "dagorc_rebase/read/dag_orc_read.py", "2e5f9f1778e104615d4480f51c6246703c9e3bb8f6c0eeaa3fc0c5621bd8adcf"),
                         ("raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md", A + "dagland/dagorc_r2a/R2A_NOTE.md", "825fa0856b91b04937cc906b1cae70b443b1615ef8c07c4ceebf7ebc116823c3")):
    bl = blob(V8 + repo_rel)
    assert bl is not None and sha(bl) == h == sha(rd(src).replace(b"\r\n", b"\n")), repo_rel

SKIP = (".pt", ".npz", ".npy", ".pyc")
n = 0
lfn = []
sums_out = []
for rel, (src, want) in sorted(files.items()):
    assert not src.endswith(SKIP), src
    b = rd(src)
    if want is not None:
        assert sha(b) == want, (rel, sha(b))
    assert b and len(b) < 400_000, (rel, len(b))
    if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
        b = b.replace(b"\r\n", b"\n")
        assert b"\r" not in b, rel
        lfn.append(rel)
    if rel.endswith((".py", ".md")):
        assert b"\r" not in b, rel
    assert blob(PK + rel) is None, rel
    put("stage/" + PK + rel, b)
    sums_out.append(f"{sha(b)} *{rel}")
    n += 1
if lfn:
    b = ("CRLF -> LF on banking (content otherwise unchanged):\n" + "\n".join(lfn) + "\n").encode()
    put("stage/" + PK + "LF_NORMALISED.txt", b)
    sums_out.append(f"{sha(b)} *LF_NORMALISED.txt")
    n += 1
put("stage/" + PK + "PACKAGE_SHA256SUMS.txt", ("\n".join(sums_out) + "\n").encode())
n += 1

# ---- the register row (mods-rw; one line replaced, line count unchanged)
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g is not None and b"\r" not in g
rr = rd(HERE + "REGISTER_ROW.md").decode("utf-8")
m = re.search(r"<!-- STATUS-BEGIN -->\n(.*?)\n<!-- STATUS-END -->", rr, re.S)
status = m.group(1).strip()
ev_old = re.search(r"<!-- EVID-OLD: (.*?) -->", rr).group(1)
ev_new = re.search(r"<!-- EVID-NEW: (.*?) -->", rr).group(1)
assert status.startswith("**HARMS (MEASURED 2026-10-28") and status.endswith("Was:") and "\n" not in status and "|" not in status
for w in FORBID:
    assert w not in status.lower().encode(), w
ID = b"| **H-V8F-DAG-ORC-1** (2026-10-27) |"
assert g.count(ID) == 1
i = g.index(ID)
j = g.index(b"\n", i)
row = g[i:j]
OLDHEAD = b"| **N0 READ: NOT NAVMOVE (MEASURED 2026-10-27, all 17 controls PASS; A, REP, B NOT RUN).**"
TAIL_OLD = b"| " + ev_old.encode() + b" | `Project Steering/PREREG_V8F_DAG_ORC.md` |"
TAIL_NEW = b"| " + ev_new.encode() + b" | `Project Steering/PREREG_V8F_DAG_ORC.md` |"
assert row.count(OLDHEAD) == 1 and row.endswith(TAIL_OLD) and row.count(TAIL_OLD) == 1, row[-160:]
row2 = row.replace(OLDHEAD, b"| " + status.encode("utf-8") + b" " + OLDHEAD[2:])
row2 = row2[: -len(TAIL_OLD)] + TAIL_NEW
assert row2.startswith(ID) and row2.count(b"\n") == 0
# the old status text survives after "Was:" byte for byte
assert (b"Was: " + OLDHEAD[2:]) in row2
g2 = g[:i] + row2 + g[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
for w in FORBID:
    assert w not in row2[: row2.index(b"Was: ")].lower(), w
put("mods-rw/" + GC, g2)

# ---- commit message
MSG = ("RESULT_V8F_DAG_ORC: HARMS (H-V8F-DAG-ORC-1), arm A (O-RECOVER's perturbation, p 0.3, composed into DAG-OACC's recipe), all 162 reader controls PASS. Guards: R-ADE"
       " fails at s0 (ADE6 A - CTRL +0.1434 m, separated-worse), R-T0 fails at both seeds (T0 ADE6 +0.1106 / +0.0722 m against <= +0.0604); R-TLAT, R-TLON and R-NEW"
       " held. R-MOVE false at both seeds: the four crawl rows improve against CTRL, each separated-better, at 8.8 to 33.8 percent of their bar (the replicate pair sets"
       " the bar; REP s0 - CTRL ADE6 +0.4025 m). Committed next: B (p 0.1) under the same table, blocked on the PI's go. T1-kin four families reported;"
       " the T0 four families are NOT RUN and three beside rows are NOT BUILT (work items). Run history: r1 refused at K-REACH, r2's first launch at its run"
       " record (shared aside marker), A s1 at its run record (a self-test replaced box 2's queue mid-run), final launch pinned to 8c24823f; box 2 about 4.9 h"
       f" by the logs against the ~3 h asked. Package raw/2026-10-28-dag-orc-result/ ({n} files: read JSONs, logs, run records, K-BUILD record, notes, audit)."
       " Register row updated.\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n")
for w in FORBID:
    assert w not in MSG.lower().encode(), w
put("msg.txt", MSG.encode("utf-8"))

# ---- last checks on everything staged
bad = []
for dp, _, fs in os.walk(H):
    for f in fs:
        p = os.path.join(dp, f)
        b = rd(p)
        if HFTOK.search(b):
            bad.append(("hf_", p))
        if p.endswith((".md", ".py", ".sh")) and b"\r" in b:
            bad.append(("CR", p))
        if p.endswith(SKIP):
            bad.append(("binary", p))
assert not bad, bad
print("hold written", H, n, "package files; register line count", len(g.split(b"\n")), "unchanged")
